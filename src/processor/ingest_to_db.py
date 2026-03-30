import os
import hashlib
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Any

from supabase import create_client


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]  # recommended for inserts
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

CHUNK_SIZE = 1600
OVERLAP = 200
MIN_LEN = 300


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """
    Simple char-based chunking with overlap.
    Great MVP choice for web pages.
    """
    t = (text or "").strip()
    if not t:
        return []

    chunks: List[str] = []
    start = 0
    n = len(t)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(t[start:end])
        if end == n:
            break
        start = max(end - overlap, 0)

    return chunks


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def snapshot_already_chunked(snapshot_id: int) -> bool:
    resp = sb.table("snapshot_chunks").select("id").eq("snapshot_id", snapshot_id).limit(1).execute()
    return bool(resp.data)


def main(limit: int = 200):
    # Fetch recent snapshots (adjust limit as needed)
    resp = (
        sb.table("snapshots")
        .select("id, source_id, clean_text")
        .order("fetched_at", desc=True)
        .limit(limit)
        .execute()
    )

    snapshots: List[Dict[str, Any]] = resp.data or []
    print(f"Fetched {len(snapshots)} snapshots")

    chunked = 0
    skipped = 0

    for s in snapshots:
        snapshot_id = int(s["id"])
        clean_text = s.get("clean_text") or ""

        if len(clean_text) < MIN_LEN:
            skipped += 1
            continue

        if snapshot_already_chunked(snapshot_id):
            skipped += 1
            continue

        chunks = chunk_text(clean_text)

        rows = []
        for idx, ch in enumerate(chunks):
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "chunk_index": idx,
                    "chunk_text": ch,
                    "chunk_hash": sha256(ch),
                }
            )

        if rows:
            sb.table("snapshot_chunks").insert(rows).execute()
            chunked += 1
            print(f"✅ snapshot_id={snapshot_id} chunks={len(rows)}")
        else:
            skipped += 1

    print(f"Done. chunked={chunked} skipped={skipped}")


if __name__ == "__main__":
    main()