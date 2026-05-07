from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.database_manager import DatabaseManager
from .config import settings
from .embedders import get_embedder_spec, Embedder
from .retriever import Retriever
from .prompts import build_prompt
from .llm_clients import NoLLM, NIMClient


def load_evalset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    # Ensure .env is loaded when running via `python -m src.runner`
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", required=True, choices=["bge_small", "e5_base", "mpnet", "nomic_768"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--llm", default="none", choices=["none", "nim"])
    ap.add_argument("--model", default="")  # LLM model name if llm enabled
    ap.add_argument("--evalset", default="src/rag/evalset.jsonl")
    args = ap.parse_args()

    # 1) Initialize Components
    spec = get_embedder_spec(args.embedder)
    embedder = Embedder(spec)
    retriever = Retriever(settings.supabase_url, settings.supabase_key)
    db_manager = DatabaseManager()

    if args.llm == "nim":
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY not set in .env")
        llm = NIMClient(settings.nvidia_api_key, settings.nim_base_url)
        if not args.model:
            raise RuntimeError('Pass --model for NIM, e.g. --model "meta/llama-3.1-8b-instruct"')
    else:
        llm = NoLLM()

    eval_rows = load_evalset(args.evalset)

    os.makedirs("src/rag/results", exist_ok=True)

    # Use ONE timestamp for both run_id and output filename
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = f"src/rag/results/run_{args.embedder}_{args.llm}_{ts}.jsonl"
    run_id = f"{args.embedder}_{args.llm}_{ts}"

    all_records = []

    with open(out_path, "w", encoding="utf-8") as out:
        for row in eval_rows:
            qid = row.get("id")
            question = row["question"]

            t0 = time.time()
            qvec = embedder.embed_query(question)
            t_embed = time.time()

            retrieved = retriever.retrieve(spec, qvec, k=args.k)
            t_ret = time.time()

            prompt = build_prompt(question, retrieved)

            ans = llm.generate(prompt, model=args.model).text
            t_llm = time.time()

            record = {
                "id": qid,
                "embedder": spec.model_name,
                "embed_dim": spec.dim,
                "k": args.k,
                "llm": args.llm,
                "llm_model": args.model,
                "question": question,
                "retrieved": [
                    {
                        "rank": i + 1,
                        "similarity": r.get("similarity"),
                        "snapshot_chunk_id": r.get("snapshot_chunk_id"),
                        "chunk_text_preview": (r.get("chunk_text") or "")[:300],
                    }
                    for i, r in enumerate(retrieved)
                ],
                "prompt": prompt,
                "answer": ans,
                "timing_ms": {
                    "embed": int((t_embed - t0) * 1000),
                    "retrieve": int((t_ret - t_embed) * 1000),
                    "llm": int((t_llm - t_ret) * 1000),
                    "total": int((t_llm - t0) * 1000),
                },
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.append(record)

            print(
                f"✅ {qid}: embed={record['timing_ms']['embed']}ms "
                f"ret={record['timing_ms']['retrieve']}ms "
                f"llm={record['timing_ms']['llm']}ms"
            )

    # 2) Database Storage Step
    print(f"\nSaved local results to: {out_path}")
    print("DEBUG: records_written =", len(all_records))
    print(
        "DEBUG: out_path exists =", os.path.exists(out_path),
        "size =", os.path.getsize(out_path) if os.path.exists(out_path) else None
    )

    try:
        db_manager.store_run_results(out_path, run_id=run_id)
        print("🚀 Successfully synced all results to Supabase!")
    except Exception as e:
        print(f"❌ Database sync failed: {e}")


# ✅ THIS is the critical part you were missing
if __name__ == "__main__":
    main()