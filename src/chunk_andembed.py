from .db import get_supabase_client
from .embedders import Embedder, get_embedder_spec


def main():
    sb = get_supabase_client()

    spec = get_embedder_spec("bge_small")
    embedder = Embedder(spec)

    print("Fetching snapshot chunks from Supabase...")

    response = (
        sb.table("snapshot_chunks")
        .select("id, snapshot_id, chunk_index, chunk_text")
        .execute()
    )
    chunks = response.data

    if not chunks:
        print("❓ No rows found in snapshot_chunks.")
        return

    print(f"📋 Found {len(chunks)} chunks. Starting embedding...")

    inserted_count = 0
    skipped_count = 0
    error_count = 0

    for chunk in chunks:
        chunk_id = chunk.get("id")
        snapshot_id = chunk.get("snapshot_id")
        chunk_index = chunk.get("chunk_index")
        text = chunk.get("chunk_text")

        if not text:
            print(f"⚠️ Skipping chunk {chunk_id} - No chunk_text found.")
            skipped_count += 1
            continue

        print(
            f"🚀 Processing chunk_id={chunk_id} | "
            f"snapshot_id={snapshot_id} | chunk_index={chunk_index}"
        )

        try:
            existing = (
                sb.table(spec.table)
                .select("snapshot_chunk_id")
                .eq("snapshot_chunk_id", chunk_id)
                .limit(1)
                .execute()
            )

            if existing.data:
                print(f"⏭️ Skipping chunk {chunk_id} - already embedded.")
                skipped_count += 1
                continue

            vector = embedder.embed_query(text)

            payload = {
                "snapshot_chunk_id": chunk_id,
                "embedding": vector,
                "model_name": "bge-small-en-v1.5",
                "agent_name": "ios-risk-agent",
            }

            sb.table(spec.table).insert(payload).execute()
            inserted_count += 1

        except Exception as e:
            print(f"❌ Error processing chunk {chunk_id}: {e}")
            error_count += 1

    print("\n✅ Embedding run complete.")
    print(f"Inserted: {inserted_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total checked: {len(chunks)}")


if __name__ == "__main__":
    main()