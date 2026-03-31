import math
from typing import Any, Dict, List, Tuple

from .embedders import Embedder, get_embedder_spec
from .scrape_ios_sources_2 import fetch_parse_and_clean
from .hybrid_chunker import IOSHybridChunker


TEST_URL = "https://support.apple.com/en-us/120304"
TOP_K = 5

TEST_QUERIES = [
    "kernel privilege arbitrary code execution",
    "privacy issue temporary files",
    "malicious webpage fingerprint the user",
    "legacy RSA PKCS#1 v1.5 ciphertext decrypt without private key",
]


class IOSDocumentChunker:
    """
    Baseline chunker copied from team version.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def clean_text(self, text: str) -> str:
        import re

        text = re.sub(r"\s+", " ", text)
        boilerplate = [
            "Shop the Latest Mac iPad iPhone",
            "Apple Store Shop",
            "Accessories Apple Trade In",
        ]
        for phrase in boilerplate:
            text = text.replace(phrase, "")
        return text.strip()

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return [text]

        separator = separators[0]
        remaining_seps = separators[1:]

        splits = text.split(separator)
        final_chunks = []
        current_chunk = ""

        for s in splits:
            if len(current_chunk) + len(s) + len(separator) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())

                if len(s) > self.chunk_size and remaining_seps:
                    final_chunks.extend(self._split_text(s, remaining_seps))
                    current_chunk = ""
                else:
                    current_chunk = s
            else:
                current_chunk += (separator if current_chunk else "") + s

        if current_chunk:
            final_chunks.append(current_chunk.strip())

        return final_chunks

    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        cleaned_text = self.clean_text(text)
        raw_chunks = self._split_text(cleaned_text, self.separators)

        processed_chunks = []
        for i, content in enumerate(raw_chunks):
            version = metadata.get("version", "iOS")
            contextual_content = f"[{version} UPDATE] {content}"

            processed_chunks.append(
                {
                    "content": contextual_content,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "char_count": len(content),
                        "chunk_type": "BASELINE",
                    },
                }
            )

        return processed_chunks


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_chunks(embedder: Embedder, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    embedded = []
    for chunk in chunks:
        item = dict(chunk)
        item["embedding"] = embedder.embed_query(chunk["content"])
        embedded.append(item)
    return embedded


def rank_chunks(
    embedder: Embedder,
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    q_emb = embedder.embed_query(query)

    scored = []
    for chunk in chunks:
        score = cosine_similarity(q_emb, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def print_ranked_results(title: str, ranked: List[Tuple[float, Dict[str, Any]]]) -> None:
    print(f"\n--- {title} ---")
    for i, (score, chunk) in enumerate(ranked, start=1):
        meta = chunk["metadata"]
        print(
            f"\n[{i}] score={score:.4f} "
            f"| type={meta.get('chunk_type')} "
            f"| section={meta.get('section_title', 'N/A')} "
            f"| chunk_index={meta.get('chunk_index')}"
        )
        print(chunk["content"][:500])


def main():
    print(f"Fetching source page: {TEST_URL}")
    parsed = fetch_parse_and_clean(TEST_URL)

    print("\nDocument info:")
    print(f"page_title: {parsed['page_title']}")
    print(f"clean_text_len: {parsed['clean_text_len']}")
    print(f"section_count: {len(parsed['sections'])}")

    # Build baseline chunks
    baseline_chunker = IOSDocumentChunker(chunk_size=1000, chunk_overlap=150)
    baseline_chunks = baseline_chunker.chunk_document(
        parsed["clean_text_full"],
        metadata={
            "page_title": parsed["page_title"],
            "version": parsed["ios_version"] or "iOS",
            "source_url": TEST_URL,
        },
    )

    # Build hybrid chunks
    hybrid_chunker = IOSHybridChunker(
        max_tokens=800,
        min_tokens=50,
        overlap=100,
        summary_tokens=120,
    )
    hybrid_chunks = hybrid_chunker.chunk_parsed_document(
        parsed=parsed,
        source_metadata={
            "source_id": "debug_source",
            "source_name": "debug_name",
            "source_url": TEST_URL,
        },
    )

    # Filter out summary chunks for a fairer retrieval comparison
    hybrid_chunks = [
        c for c in hybrid_chunks
        if c["metadata"].get("chunk_type") != "SECTION_SUMMARY"
    ]

    print(f"\nBaseline chunk count: {len(baseline_chunks)}")
    print(f"Hybrid chunk count (without summaries): {len(hybrid_chunks)}")

    spec = get_embedder_spec("bge_small")
    embedder = Embedder(spec)

    print("\nEmbedding baseline chunks...")
    baseline_chunks = embed_chunks(embedder, baseline_chunks)

    print("Embedding hybrid chunks...")
    hybrid_chunks = embed_chunks(embedder, hybrid_chunks)

    for query in TEST_QUERIES:
        print(f"\n==============================")
        print(f"QUERY: {query}")
        print(f"==============================")

        baseline_ranked = rank_chunks(embedder, query, baseline_chunks, top_k=TOP_K)
        hybrid_ranked = rank_chunks(embedder, query, hybrid_chunks, top_k=TOP_K)

        print_ranked_results("BASELINE", baseline_ranked)
        print_ranked_results("HYBRID", hybrid_ranked)


if __name__ == "__main__":
    main()