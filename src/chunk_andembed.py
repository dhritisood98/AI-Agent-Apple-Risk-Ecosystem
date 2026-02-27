from .db import get_supabase_client
from .config import settings
from .processor.chunker import IOSDocumentChunker
from .embedders import Embedder, get_embedder_spec

def main():
    sb = get_supabase_client()
    
    # 1. Initialize Chunker and Embedder (using bge_small as per your runner)
    chunker = IOSDocumentChunker(chunk_size=1000, chunk_overlap=150)
    spec = get_embedder_spec("bge_small")
    embedder = Embedder(spec)

    # 2. Fetch snapshots that need processing
    print("Fetching raw snapshots...")
    # Adjust 'snapshots' to whatever your table name is
    snapshots = sb.table("snapshots").select("*").eq("agent_name", "ios-risk-agent").execute().data

    for snap in snapshots:
        print(f"Processing: {snap['url']}")
        
        # Prepare metadata for context injection
        metadata = {
            "source_url": snap['url'],
            "doc_id": snap['id'],
            "version": "iOS 18" # You can add logic to parse this from the URL
        }

        # 3. Chunk the document
        chunks = chunker.chunk_document(snap['content'], metadata)

        # 4. Embed and Upload
        for chunk in chunks:
            # Generate vector using your embedder.py logic
            vector = embedder.embed_query(chunk['content'])
            
            payload = {
                "agent_name": "ios-risk-agent",
                "content": chunk['content'],
                "metadata": chunk['metadata'],
                "embedding": vector,
                "source_id": snap['id']
            }
            
            # Upsert into the table defined in your EmbedderSpec
            sb.table(spec.table).upsert(payload).execute()
        
    print(f"✅ Finished processing {len(snapshots)} documents.")

if __name__ == "__main__":
    main()