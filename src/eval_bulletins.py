"""
Bulletin Relevance Evaluation Script
=====================================
Evaluates whether the Related Apple Bulletins retrieved for each Swift file
are actually relevant, using keyword-based ground truth.

Usage:
    python -m src.eval_bulletins
    python -m src.eval_bulletins --k 5 --verbose
    python -m src.eval_bulletins --file OSInfoHarvester.swift
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import settings
from src.embedders import get_embedder_spec, Embedder
from src.retriever import Retriever
from supabase import create_client

GROUND_TRUTH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "eval", "bulletin_relevance.json"
)

# Similarity thresholds for quality badges
STRONG_THRESHOLD = 0.75
PLAUSIBLE_THRESHOLD = 0.55


def load_ground_truth(file_filter: str | None = None) -> List[Dict]:
    with open(GROUND_TRUTH_PATH, "r") as f:
        data = json.load(f)
    if file_filter:
        data = [d for d in data if file_filter.lower() in d["swift_file"].lower()]
    return data


def load_code_knowledge(sb) -> List[Dict]:
    res = sb.table("code_knowledge").select("file_path, feature, content").execute()
    return res.data or []


def score_bulletin(bulletin_text: str, entry: Dict) -> Tuple[bool, bool, List[str], List[str]]:
    """
    Check whether a retrieved bulletin is relevant or irrelevant.

    Returns (has_relevant, has_irrelevant, matched_relevant, matched_irrelevant).
    """
    text_lower = bulletin_text.lower()
    matched_rel = [kw for kw in entry["relevant_keywords"] if kw.lower() in text_lower]
    matched_irr = [kw for kw in entry["irrelevant_keywords"] if kw.lower() in text_lower]
    return bool(matched_rel), bool(matched_irr), matched_rel, matched_irr


def badge_for_score(sim: float) -> str:
    if sim >= STRONG_THRESHOLD:
        return "STRONG  ✅"
    elif sim >= PLAUSIBLE_THRESHOLD:
        return "PLAUSIBLE  ⚠️"
    else:
        return "WEAK    ❌"


def run_eval(k: int = 3, verbose: bool = False, file_filter: str | None = None) -> None:
    print("=" * 70)
    print("  Bulletin Relevance Evaluation")
    print(f"  k={k} | threshold_strong={STRONG_THRESHOLD} | threshold_plausible={PLAUSIBLE_THRESHOLD}")
    print("=" * 70)

    gt_entries = load_ground_truth(file_filter)
    if not gt_entries:
        print("No ground truth entries found (check --file filter).")
        return

    sb = create_client(settings.supabase_url, settings.supabase_key)
    code_rows = load_code_knowledge(sb)

    spec = get_embedder_spec("nomic_768")
    embedder = Embedder(spec)
    retriever = Retriever(settings.supabase_url, settings.supabase_key)

    # Build lookup: short filename -> content/summary
    content_map: Dict[str, str] = {}
    for row in code_rows:
        short = (row.get("file_path") or "").split("/")[-1]
        content_map[short] = row.get("content", "")

    total_hits = 0
    total_bulletins = 0
    total_irrelevant = 0
    results: List[Dict[str, Any]] = []

    for entry in gt_entries:
        swift_file = entry["swift_file"]
        summary = content_map.get(swift_file, "")

        if not summary:
            print(f"\n⚠️  {swift_file} — no summary found in code_knowledge, skipping.")
            continue

        query_vec = embedder.embed_query(summary)
        raw = retriever.retrieve_with_rpc("match_chunks_768", query_vec, k=k * 3)
        from src.similarity import rerank_by_cosine
        bulletins = rerank_by_cosine(query_vec, raw)[:k]

        file_hits = 0
        file_irrel = 0
        bulletin_details = []

        for b in bulletins:
            sim = b.get("cosine_similarity", b.get("similarity", 0.0))
            text = b.get("chunk_text") or b.get("content") or ""
            has_rel, has_irr, matched_rel, matched_irr = score_bulletin(text, entry)

            if has_rel:
                file_hits += 1
            if has_irr and not has_rel:
                file_irrel += 1

            bulletin_details.append({
                "sim": sim,
                "badge": badge_for_score(sim),
                "relevant": has_rel,
                "irrelevant_only": has_irr and not has_rel,
                "matched_rel": matched_rel,
                "matched_irr": matched_irr,
                "preview": text[:120].replace("\n", " "),
            })

        hit_rate = file_hits / max(len(bulletins), 1)
        total_hits += file_hits
        total_bulletins += len(bulletins)
        total_irrelevant += file_irrel

        results.append({
            "swift_file": swift_file,
            "num_bulletins": len(bulletins),
            "hits": file_hits,
            "hit_rate": hit_rate,
            "irrelevant": file_irrel,
            "details": bulletin_details,
        })

        status = "✅" if hit_rate >= 0.5 else ("⚠️" if hit_rate > 0 else "❌")
        print(f"\n{status}  {swift_file}  —  hit rate: {hit_rate:.0%}  ({file_hits}/{len(bulletins)} bulletins relevant)")

        if verbose:
            for i, bd in enumerate(bulletin_details, 1):
                rel_mark = "✅" if bd["relevant"] else ("⚠️" if bd["irrelevant_only"] else "—")
                print(f"   [{i}] {bd['badge']}  sim={bd['sim']:.3f}  {rel_mark}")
                print(f"       preview: {bd['preview']}")
                if bd["matched_rel"]:
                    print(f"       matched relevant: {bd['matched_rel']}")
                if bd["matched_irr"]:
                    print(f"       matched irrelevant: {bd['matched_irr']}")

    # Summary
    overall_hit_rate = total_hits / max(total_bulletins, 1)
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print(f"  Files evaluated : {len(results)}")
    print(f"  Total bulletins : {total_bulletins}")
    print(f"  Relevant hits   : {total_hits}  ({overall_hit_rate:.0%})")
    print(f"  Irrelevant only : {total_irrelevant}")
    print("=" * 70)

    # Per-file table
    print("\n  Per-file breakdown:")
    print(f"  {'File':<45} {'Hit Rate':>9} {'Hits':>5} {'k':>3} {'Irrel':>6}")
    print("  " + "-" * 70)
    for r in sorted(results, key=lambda x: x["hit_rate"]):
        flag = "✅" if r["hit_rate"] >= 0.5 else ("⚠️ " if r["hit_rate"] > 0 else "❌")
        print(
            f"  {flag} {r['swift_file']:<43} "
            f"{r['hit_rate']:>8.0%} "
            f"{r['hits']:>5} "
            f"{r['num_bulletins']:>3} "
            f"{r['irrelevant']:>6}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate bulletin relevance per Swift file")
    parser.add_argument("--k", type=int, default=3, help="Number of bulletins to retrieve per file (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Show per-bulletin detail")
    parser.add_argument("--file", type=str, default=None, help="Filter to a specific Swift file name")
    args = parser.parse_args()

    run_eval(k=args.k, verbose=args.verbose, file_filter=args.file)


if __name__ == "__main__":
    main()
