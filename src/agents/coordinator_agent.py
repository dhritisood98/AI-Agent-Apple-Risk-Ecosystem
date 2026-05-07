"""
Coordinator Agent
=================
The brain. Reads today's triage_results from the Sentinel, compares against
risk_history to detect changes, runs cross-file cluster analysis, and writes
alerts to risk_alerts.

Feedback loop:
  - Risk escalation:   file moved from No Impact / Low → Medium / High
  - Risk de-escalation: file moved from High / Medium → lower (positive signal)
  - Similarity trending: max_sim score rising across last N runs even if still
    below threshold — early warning before a file tips over
  - New bulletin coverage: bulletin_count went from 0 → >0

Cross-file clustering:
  - Groups files sharing the same signal_category + effective_risk
  - If a whole cluster is at High, that's more urgent than a single file

risk_history schema:
  file_name       TEXT
  effective_risk  TEXT
  zs_risk_level   TEXT
  max_sim_score   FLOAT
  bulletin_count  INT
  run_at          TIMESTAMPTZ

risk_alerts schema:
  alert_type      TEXT   (escalation | de-escalation | trending | coverage | cluster)
  severity        TEXT   (high | medium | low | info)
  file_name       TEXT   (null for cluster alerts)
  cluster         TEXT   (signal_category, for cluster alerts)
  message         TEXT
  previous_value  TEXT
  current_value   TEXT
  run_at          TIMESTAMPTZ
  acknowledged    BOOL   (default false — UI can mark these)
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from collections import defaultdict
from .base_agent import BaseAgent

# How many past runs to look at for trending
TREND_WINDOW = 5
# Minimum sim score increase per run to flag as trending
TREND_DELTA_THRESHOLD = 0.03
# Risk level ordering for escalation detection
RISK_ORDER = {"No Impact": 0, "Low": 1, "Medium": 2, "High": 3}


class CoordinatorAgent(BaseAgent):
    name = "coordinator"

    def _load_current(self) -> list[dict]:
        return self.sb.table("triage_results").select("*").execute().data or []

    def _load_history(self, file_name: str, limit: int = TREND_WINDOW) -> list[dict]:
        return (
            self.sb.table("risk_history")
            .select("file_name, effective_risk, zs_risk_level, max_sim_score, bulletin_count, run_at")
            .eq("file_name", file_name)
            .order("run_at", desc=True)
            .limit(limit)
            .execute()
            .data or []
        )

    def _write_history(self, verdicts: list[dict]) -> None:
        rows = [
            {
                "file_name":      v["file_name"],
                "effective_risk": v["effective_risk"],
                "zs_risk_level":  v["zs_risk_level"],
                "max_sim_score":  v["max_sim_score"],
                "bulletin_count": v["bulletin_count"],
                "run_at":         v["run_at"],
            }
            for v in verdicts
        ]
        if rows:
            self.sb.table("risk_history").insert(rows).execute()

    def _write_alerts(self, alerts: list[dict]) -> None:
        if alerts:
            self.sb.table("risk_alerts").insert(alerts).execute()

    # ── feedback loop checks ────────────────────────────────────────────────

    def _check_escalation(self, current: dict, history: list[dict]) -> dict | None:
        if not history:
            return None
        prev_risk = history[0]["effective_risk"]
        curr_risk = current["effective_risk"]
        if RISK_ORDER.get(curr_risk, 0) > RISK_ORDER.get(prev_risk, 0):
            severity = "high" if curr_risk == "High" else "medium"
            return {
                "alert_type":     "escalation",
                "severity":       severity,
                "file_name":      current["file_name"],
                "cluster":        current.get("signal_category", ""),
                "message":        (
                    f"{current['file_name']} risk escalated: "
                    f"{prev_risk} → {curr_risk}. "
                    f"Top bulletin similarity: {current['max_sim_score']:.3f}."
                ),
                "previous_value": prev_risk,
                "current_value":  curr_risk,
                "run_at":         current["run_at"],
                "acknowledged":   False,
            }
        return None

    def _check_de_escalation(self, current: dict, history: list[dict]) -> dict | None:
        if not history:
            return None
        prev_risk = history[0]["effective_risk"]
        curr_risk = current["effective_risk"]
        if RISK_ORDER.get(curr_risk, 0) < RISK_ORDER.get(prev_risk, 0):
            return {
                "alert_type":     "de-escalation",
                "severity":       "info",
                "file_name":      current["file_name"],
                "cluster":        current.get("signal_category", ""),
                "message":        (
                    f"{current['file_name']} risk reduced: "
                    f"{prev_risk} → {curr_risk}. "
                    f"Top similarity now {current['max_sim_score']:.3f}."
                ),
                "previous_value": prev_risk,
                "current_value":  curr_risk,
                "run_at":         current["run_at"],
                "acknowledged":   False,
            }
        return None

    def _check_trending(self, current: dict, history: list[dict]) -> dict | None:
        """Flag files where similarity score is rising even if still below threshold."""
        if len(history) < 3:
            return None
        # Only care about files currently at No Impact — they could tip over
        if current["effective_risk"] != "No Impact":
            return None
        scores = [h["max_sim_score"] for h in history[:TREND_WINDOW]]
        scores = [current["max_sim_score"]] + scores  # prepend current
        # Check if each step is strictly increasing with meaningful delta
        rising_steps = sum(
            1 for i in range(len(scores) - 1)
            if scores[i] - scores[i + 1] >= TREND_DELTA_THRESHOLD
        )
        if rising_steps >= 2:
            return {
                "alert_type":     "trending",
                "severity":       "medium",
                "file_name":      current["file_name"],
                "cluster":        current.get("signal_category", ""),
                "message":        (
                    f"{current['file_name']} similarity score trending up "
                    f"({scores[-1]:.3f} → {scores[0]:.3f} over {len(scores)} runs). "
                    f"Still below threshold but rising — early warning."
                ),
                "previous_value": str(round(scores[-1], 3)),
                "current_value":  str(round(scores[0], 3)),
                "run_at":         current["run_at"],
                "acknowledged":   False,
            }
        return None

    def _check_new_coverage(self, current: dict, history: list[dict]) -> dict | None:
        """Flag when a file that had 0 matching bulletins now has some."""
        if not history:
            return None
        prev_count = history[0]["bulletin_count"]
        curr_count = current["bulletin_count"]
        if prev_count == 0 and curr_count > 0:
            return {
                "alert_type":     "coverage",
                "severity":       "medium",
                "file_name":      current["file_name"],
                "cluster":        current.get("signal_category", ""),
                "message":        (
                    f"{current['file_name']} now has {curr_count} matching Apple bulletin(s) "
                    f"(previously none). New Apple content may be relevant."
                ),
                "previous_value": "0 bulletins",
                "current_value":  f"{curr_count} bulletins",
                "run_at":         current["run_at"],
                "acknowledged":   False,
            }
        return None

    # ── cross-file cluster analysis ────────────────────────────────────────

    def _check_clusters(self, verdicts: list[dict]) -> list[dict]:
        """If a whole signal category has multiple High-risk files, raise a cluster alert."""
        cluster_risks: dict[str, list[str]] = defaultdict(list)
        for v in verdicts:
            cluster_risks[v.get("signal_category", "Other")].append(v["effective_risk"])

        alerts = []
        for category, risks in cluster_risks.items():
            high_count = risks.count("High")
            medium_count = risks.count("Medium")
            total = len(risks)
            if high_count >= 2:
                alerts.append({
                    "alert_type":     "cluster",
                    "severity":       "high",
                    "file_name":      None,
                    "cluster":        category,
                    "message":        (
                        f"Cluster alert: {high_count}/{total} files in '{category}' "
                        f"are High-risk. This entire signal category may be compromised."
                    ),
                    "previous_value": "",
                    "current_value":  f"{high_count} High, {medium_count} Medium",
                    "run_at":         self._utc_now(),
                    "acknowledged":   False,
                })
        return alerts

    # ── main ───────────────────────────────────────────────────────────────

    def run(self) -> dict:
        self._log_start()

        verdicts = self._load_current()
        if not verdicts:
            print("  No triage_results found. Run Sentinel first.")
            return {"status": "no_data"}

        print(f"\n  Analyzing {len(verdicts)} files for feedback signals...")

        alerts = []

        # Per-file feedback loop
        for v in verdicts:
            history = self._load_history(v["file_name"])
            for check in [
                self._check_escalation,
                self._check_de_escalation,
                self._check_trending,
                self._check_new_coverage,
            ]:
                alert = check(v, history)
                if alert:
                    alerts.append(alert)
                    print(f"  ⚠️  [{alert['alert_type'].upper()}] {alert['message']}")

        # Cross-file cluster analysis
        cluster_alerts = self._check_clusters(verdicts)
        alerts.extend(cluster_alerts)
        for a in cluster_alerts:
            print(f"  🔴  [CLUSTER] {a['message']}")

        # Persist history snapshot and alerts
        self._write_history(verdicts)
        self._write_alerts(alerts)

        if not alerts:
            print("  ✅  No risk changes detected. All signals stable.")

        result = {
            "files_analyzed": len(verdicts),
            "alerts_generated": len(alerts),
            "alert_types": list({a["alert_type"] for a in alerts}),
        }
        self._log_done(result)
        return result
