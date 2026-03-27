from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

from transformers import pipeline


RISK_LEVEL_LABELS = [
    "high risk: privacy restriction or signal blocked by Apple",
    "medium risk: API behavior change or partial degradation",
    "low risk: stable, minimal disruption expected",
]

RISK_TYPE_LABELS = [
    "Signal Degradation",
    "API Deprecation",
    "Privacy Violation",
    "Permission Change",
    "Operational",
]

# Map the raw label text back to the short display name
_LEVEL_MAP = {
    "high risk: privacy restriction or signal blocked by Apple": "High",
    "medium risk: API behavior change or partial degradation": "Medium",
    "low risk: stable, minimal disruption expected": "Low",
}

_LEVEL_REASON_MAP = {
    "High": "Likely exposed to Apple privacy restrictions or reliability changes.",
    "Medium": "Depends on APIs that may remain available but could change in behavior.",
    "Low": "Lower sensitivity; less immediate disruption risk.",
}


@lru_cache(maxsize=1)
def _get_pipeline():
    """Load the NLI pipeline once and cache it for the process lifetime."""
    return pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-deberta-v3-small",
        device=-1,  # CPU; set to 0 for GPU
    )


def classify_risk_zs_with_scores(text: str) -> Tuple[str, str, str, Dict[str, float]]:
    """Zero-shot risk classification with per-label confidence scores.

    Returns (level, risk_type, reason, scores) where scores is a dict
    mapping short label names ("High", "Medium", "Low") to probabilities.
    """
    t = (text or "").strip()
    if not t:
        scores = {"High": 0.0, "Medium": 0.0, "Low": 1.0}
        return "Low", "Operational", _LEVEL_REASON_MAP["Low"], scores

    clf = _get_pipeline()

    # Score risk level — returns labels sorted by descending score
    level_result = clf(t[:512], candidate_labels=RISK_LEVEL_LABELS, multi_label=False)
    raw_level = level_result["labels"][0]
    level = _LEVEL_MAP.get(raw_level, "Low")

    scores: Dict[str, float] = {
        _LEVEL_MAP.get(lbl, lbl): round(float(sc), 4)
        for lbl, sc in zip(level_result["labels"], level_result["scores"])
    }

    # Score risk type
    type_result = clf(t[:512], candidate_labels=RISK_TYPE_LABELS, multi_label=False)
    risk_type = type_result["labels"][0]

    reason = _LEVEL_REASON_MAP[level]
    return level, risk_type, reason, scores


def classify_risk_zs(text: str) -> Tuple[str, str, str]:
    """Convenience wrapper — drops the scores dict for callers that don't need it."""
    level, risk_type, reason, _ = classify_risk_zs_with_scores(text)
    return level, risk_type, reason
