"""
Sentinel Triage Agent
=====================
For every Swift file in code_knowledge, retrieves relevant Apple bulletins,
runs zero-shot risk classification, and writes a verdict to triage_results.

This is the "per-file intelligence" layer — it replaces the real-time
computation currently done at Streamlit load time.

triage_results schema:
  file_name       TEXT
  file_path       TEXT
  feature         TEXT
  signal_category TEXT
  zs_risk_level   TEXT   (High/Medium/Low — intrinsic zero-shot)
  effective_risk  TEXT   (High/Medium/Low/No Impact — bulletin-backed)
  max_sim_score   FLOAT  (top cosine similarity across retrieved bulletins)
  bulletin_count  INT    (number of qualifying bulletins ≥ threshold)
  top_bulletin_preview TEXT
  run_at          TIMESTAMPTZ
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .base_agent import BaseAgent
from src.embedders import get_embedder_spec, Embedder
from src.retriever import Retriever
from src.similarity import rerank_by_cosine
from src.zero_shot import classify_risk_zs_with_scores
from src.llm_clients import NIMClient, NoLLM
from src.prompts import build_rationale_prompt
from src.config import settings

BULLETIN_MIN_SIM = 0.55
LLM_MODEL = "meta/llama-3.1-70b-instruct"

RECOMMENDED_ACTIONS = {
    "High":   "Immediate review required — signal may be blocked or restricted in current iOS.",
    "Medium": "Monitor next 2 Apple releases — API behavior change possible.",
    "Low":    "No action needed — low disruption risk.",
}

def _extract_cve(text: str) -> str:
    """Pull the first CVE ID from bulletin text."""
    import re
    m = re.search(r'CVE-\d{4}-\d{4,7}', text or "")
    return m.group(0) if m else ""


def _signal_category(text: str) -> str:
    t = (text or "").lower()
    if any(x in t for x in ["kernel", "os version", "os build", "boot time", "time zone"]):
        return "OS / System"
    if any(x in t for x in ["identifierforvendor", "vendor identifier", "idfv"]):
        return "Identifiers"
    if any(x in t for x in ["device name", "device type", "memory", "cpu", "screen resolution"]):
        return "Device Attributes"
    if any(x in t for x in ["biometrics", "passcode", "authentication", "local authentication"]):
        return "Authentication"
    if any(x in t for x in ["locale", "user interface style", "ui style", "language", "region"]):
        return "Locale / UI"
    if any(x in t for x in ["cellular", "carrier", "network", "reachability", "mobile data", "sim"]):
        return "Network / Cellular"
    if any(x in t for x in ["keychain", "disk space", "disk", "storage", "identifier storage"]):
        return "Storage / Keychain"
    if any(x in t for x in ["sha256", "sha-256", "hash", "digest", "fingerprint function", "checksum"]):
        return "Hashing"
    if any(x in t for x in ["fingerprint tree", "compound", "tree builder", "tree calculator"]):
        return "Fingerprint Core"
    if any(x in t for x in ["configuration", "factory", "builder", "provider", "library"]):
        return "App Infrastructure"
    return "Other"


class SentinelAgent(BaseAgent):
    name = "sentinel"

    def __init__(self, supabase_url: str, supabase_key: str):
        super().__init__(supabase_url, supabase_key)
        spec = get_embedder_spec("nomic_768")
        self._embedder = Embedder(spec)
        self._retriever = Retriever(supabase_url, supabase_key)
        self._llm = (
            NIMClient(settings.nvidia_api_key, settings.nim_base_url)
            if settings.nvidia_api_key else NoLLM()
        )

    def _triage_file(self, file_path: str, feature: str, content: str) -> dict:
        """Run full triage for a single Swift file and return the verdict."""
        file_name = file_path.split("/")[-1]

        # Zero-shot classification (intrinsic sensitivity)
        zs_level, _, _, _ = classify_risk_zs_with_scores(content)

        # Bulletin retrieval + reranking
        vec = self._embedder.embed_query(content)
        raw = self._retriever.retrieve_with_rpc("match_chunks_768", vec, k=9)
        ranked = rerank_by_cosine(vec, raw)[:3]
        qualified = [
            b for b in ranked
            if b.get("cosine_similarity", b.get("similarity", 0.0)) >= BULLETIN_MIN_SIM
        ]

        max_sim = max(
            (b.get("cosine_similarity", b.get("similarity", 0.0)) for b in ranked),
            default=0.0,
        )
        effective_risk = zs_level if qualified else "No Impact"
        top_preview = ""
        if ranked:
            top_preview = (ranked[0].get("chunk_text") or "")[:600]

        # ── LLM rationale (only for High / Medium — keeps costs low) ──────────
        triggering_cve = _extract_cve(top_preview)
        rationale = ""
        if effective_risk in ("High", "Medium") and top_preview:
            try:
                prompt = build_rationale_prompt(
                    file_name=file_name,
                    summary=content,
                    effective_risk=effective_risk,
                    zs_level=zs_level,
                    top_bulletin=top_preview,
                    triggering_cve=triggering_cve,
                )
                rationale = self._llm.generate(prompt, model=LLM_MODEL).text.strip()
            except Exception as e:
                print(f"    ⚠️  Rationale LLM call failed for {file_name}: {e}")

        return {
            "file_name":            file_name,
            "file_path":            file_path,
            "feature":              feature,
            "signal_category":      _signal_category(content),
            "zs_risk_level":        zs_level,
            "effective_risk":       effective_risk,
            "max_sim_score":        round(max_sim, 4),
            "bulletin_count":       len(qualified),
            "top_bulletin_preview": top_preview,
            "triggering_cve":       triggering_cve,
            "rationale":            rationale,
            "recommended_action":   RECOMMENDED_ACTIONS.get(effective_risk, ""),
            "run_at":               self._utc_now(),
        }

    def run(self) -> dict:
        self._log_start()

        code_rows = (
            self.sb.table("code_knowledge")
            .select("file_path, feature, content")
            .execute()
            .data or []
        )
        print(f"\n  Triaging {len(code_rows)} Swift files...")

        verdicts = []
        for i, row in enumerate(code_rows, 1):
            file_name = (row.get("file_path") or "").split("/")[-1]
            print(f"  [{i:02d}/{len(code_rows)}] {file_name}...", end=" ", flush=True)
            verdict = self._triage_file(
                row.get("file_path", ""),
                row.get("feature", ""),
                row.get("content", ""),
            )
            verdicts.append(verdict)
            print(f"{verdict['effective_risk']} (sim={verdict['max_sim_score']:.3f})")

        # Upsert into triage_results (replace previous run)
        if verdicts:
            # Delete old results and reinsert fresh
            self.sb.table("triage_results").delete().neq("file_name", "__never__").execute()
            self.sb.table("triage_results").insert(verdicts).execute()

        counts = {}
        for v in verdicts:
            counts[v["effective_risk"]] = counts.get(v["effective_risk"], 0) + 1

        result = {"files_triaged": len(verdicts), "breakdown": counts}
        self._log_done(result)
        return result
