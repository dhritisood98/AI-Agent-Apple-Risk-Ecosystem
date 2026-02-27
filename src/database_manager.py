import json
from supabase import create_client
from dotenv import load_dotenv
from src.config import settings  # adjust if your settings import differs

class DatabaseManager:
    def __init__(self):
        load_dotenv()
        self.sb = create_client(settings.supabase_url, settings.supabase_key)

    def store_run_results(self, jsonl_path: str, run_id: str | None = None, batch_size: int = 200):
        rows = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)

                # Build context from retrieved chunk previews
                ctx_parts = []
                for item in r.get("retrieved", []):
                    rank = item.get("rank")
                    sim = item.get("similarity")
                    cid = item.get("snapshot_chunk_id")
                    preview = (item.get("chunk_text_preview") or "").strip()
                    ctx_parts.append(f"[{rank}] sim={sim} chunk_id={cid}\n{preview}")
                context = "\n\n".join(ctx_parts)

                t = r.get("timing_ms") or {}

                rows.append({
                    "query_id": str(r.get("id") or ""),          # eval question id
                    "query_text": r.get("question") or "",      # question
                    "context": context,                         # retrieved context
                    "llm_response": r.get("answer") or "",      # final answer
                    "embed_time_ms": t.get("embed"),
                    "retrieval_time_ms": t.get("retrieve"),
                    "llm_time_ms": t.get("llm"),
                    "run_id": run_id,
                })

        # Insert batches + print real errors
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            resp = self.sb.table("rag_results").insert(batch).execute()
            print("DB insert resp:", resp)  # keep for debugging
            if getattr(resp, "error", None):
                raise RuntimeError(resp.error)

        return True