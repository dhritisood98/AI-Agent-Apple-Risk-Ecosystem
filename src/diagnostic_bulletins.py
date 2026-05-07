"""
Bulletin Diagnostic / Root Cause Checker
=========================================
Checks the health of the bulletin embedding pipeline end-to-end.

Checks:
  1. How many rows are in vector_chunks_768 (bulletins indexed)?
  2. Avg / min / max similarity score across all bulletin chunks
  3. Do snapshot_chunks have chunk_text populated?
  4. Are source URLs present in the sources table?
  5. Sample retrieval: try a few known-good queries, report top similarities

Usage:
    python -m src.diagnostic_bulletins
    python -m src.diagnostic_bulletins --query "iOS privacy biometrics"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from supabase import create_client
from src.config import settings
from src.embedders import get_embedder_spec, Embedder
from src.retriever import Retriever
from src.similarity import rerank_by_cosine

SAMPLE_QUERIES = [
    "identifierForVendor privacy restriction iOS",
    "CoreTelephony carrier cellular network deprecated",
    "LocalAuthentication biometrics passcode",
    "sysctl kernel os version hardware model",
    "SHA256 cryptography hashing iOS security",
    "keychain SecItem storage identifier",
]


def run_diagnostic(extra_query: str | None = None) -> None:
    sb = create_client(settings.supabase_url, settings.supabase_key)
    spec = get_embedder_spec("nomic_768")
    embedder = Embedder(spec)
    retriever = Retriever(settings.supabase_url, settings.supabase_key)

    print("=" * 70)
    print("  Bulletin Diagnostic Report")
    print("=" * 70)

    # ── 1. Row counts ──────────────────────────────────────────────────────────
    print("\n[1] Table row counts")
    for table in ["snapshot_chunks", "vector_chunks_768", "sources", "snapshots"]:
        try:
            count = len(sb.table(table).select("id", count="exact").execute().data or [])
            print(f"    {table:<30} {count:>6} rows")
        except Exception as e:
            print(f"    {table:<30} ERROR: {e}")

    # ── 2. Bulletin chunks with text ───────────────────────────────────────────
    print("\n[2] snapshot_chunks — text coverage")
    try:
        all_chunks = sb.table("snapshot_chunks").select("id, chunk_text").execute().data or []
        with_text = sum(1 for c in all_chunks if c.get("chunk_text"))
        print(f"    Total chunks:       {len(all_chunks)}")
        print(f"    Chunks with text:   {with_text}")
        print(f"    Chunks without:     {len(all_chunks) - with_text}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ── 3. vector_chunks_768 — model_name breakdown ───────────────────────────
    print("\n[3] vector_chunks_768 — model_name breakdown")
    try:
        vecs = sb.table("vector_chunks_768").select("model_name").execute().data or []
        from collections import Counter
        counts = Counter(r.get("model_name", "unknown") for r in vecs)
        for model, cnt in counts.most_common():
            print(f"    {model:<40} {cnt:>5} embeddings")
        if not vecs:
            print("    ⚠️  Table is EMPTY — run chunk_andembed.py first!")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ── 4. Sources with URLs ───────────────────────────────────────────────────
    print("\n[4] sources — URL coverage")
    try:
        sources = sb.table("sources").select("id, url, agent_name").execute().data or []
        with_url = sum(1 for s in sources if s.get("url"))
        print(f"    Total sources:      {len(sources)}")
        print(f"    Sources with URL:   {with_url}")
        for s in sources[:5]:
            print(f"    [{s.get('agent_name','?')}] {(s.get('url','')[:80])}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ── 5. Sample retrieval tests ─────────────────────────────────────────────
    print("\n[5] Sample retrieval — top similarities")
    queries = SAMPLE_QUERIES + ([extra_query] if extra_query else [])
    for q in queries:
        try:
            vec = embedder.embed_query(q)
            raw = retriever.retrieve_with_rpc("match_chunks_768", vec, k=10)
            ranked = rerank_by_cosine(vec, raw)[:3]
            if not ranked:
                print(f"    [{q[:50]:<50}]  → NO RESULTS")
            else:
                sims = [f"{r.get('cosine_similarity', r.get('similarity', 0)):.3f}" for r in ranked]
                print(f"    [{q[:50]:<50}]  → top sims: {', '.join(sims)}")
        except Exception as e:
            print(f"    [{q[:50]:<50}]  → ERROR: {e}")

    # ── 6. Average similarity audit ───────────────────────────────────────────
    print("\n[6] Similarity audit — aggregate across all sample queries")
    all_sims = []
    for q in SAMPLE_QUERIES:
        try:
            vec = embedder.embed_query(q)
            raw = retriever.retrieve_with_rpc("match_chunks_768", vec, k=5)
            ranked = rerank_by_cosine(vec, raw)[:5]
            for r in ranked:
                all_sims.append(r.get("cosine_similarity", r.get("similarity", 0.0)))
        except Exception:
            pass

    if all_sims:
        avg = sum(all_sims) / len(all_sims)
        strong = sum(1 for s in all_sims if s >= 0.75)
        plausible = sum(1 for s in all_sims if 0.55 <= s < 0.75)
        weak = sum(1 for s in all_sims if s < 0.55)
        print(f"    Scores sampled:  {len(all_sims)}")
        print(f"    Average sim:     {avg:.4f}")
        print(f"    Strong (≥0.75):  {strong}  ({strong/len(all_sims):.0%})")
        print(f"    Plausible (≥0.55): {plausible}  ({plausible/len(all_sims):.0%})")
        print(f"    Weak (<0.55):    {weak}  ({weak/len(all_sims):.0%})")
    else:
        print("    Could not compute — no similarity scores returned.")

    print("\n" + "=" * 70)
    print("  Diagnostic complete.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run bulletin pipeline diagnostics")
    parser.add_argument("--query", type=str, default=None, help="Extra test query to run")
    args = parser.parse_args()
    run_diagnostic(extra_query=args.query)


if __name__ == "__main__":
    main()
