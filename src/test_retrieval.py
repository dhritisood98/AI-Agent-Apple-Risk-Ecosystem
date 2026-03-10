import os
from dotenv import load_dotenv

from .embedders import Embedder, get_embedder_spec
from .retriever import Retriever

load_dotenv()


def main():

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]

    query = "latest iOS security vulnerabilities"

    spec = get_embedder_spec("bge_small")
    embedder = Embedder(spec)

    retriever = Retriever(supabase_url, supabase_key)

    query_embedding = embedder.embed_query(query)

    results = retriever.retrieve(
        query_embedding=query_embedding,
        k=5
    )

    print("\nTop Results\n")

    for r in results:
        print(r)
        print()


if __name__ == "__main__":
    main()