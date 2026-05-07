from __future__ import annotations

import numpy as np
from typing import Any, Dict, List


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute exact cosine similarity between two vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rerank_by_cosine(
    query_vec: List[float],
    results: List[Dict[str, Any]],
    embedding_key: str = "embedding",
    score_key: str = "cosine_similarity",
) -> List[Dict[str, Any]]:
    """Re-rank a list of result dicts by client-side cosine similarity.

    If a result has an embedding vector under `embedding_key`, the score is
    computed exactly via cosine_similarity().  If no embedding is present
    (e.g. the DB RPC didn't return the vector), the pre-computed pgvector
    'similarity' field is used as a fallback so the list is still sorted
    consistently.
    """
    scored = []
    for r in results:
        emb = r.get(embedding_key)
        if emb is not None:
            score = cosine_similarity(query_vec, emb)
        else:
            score = float(r.get("similarity", 0.0))
        scored.append({**r, score_key: score})
    return sorted(scored, key=lambda x: x[score_key], reverse=True)
