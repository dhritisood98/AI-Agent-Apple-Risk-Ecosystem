from src.processor.hybrid_chunker import IOSHybridChunker

html = """
<h2>Kernel</h2>
<p>CVE-2024-1111 Test vulnerability description here.</p>
<p>More details here for the same CVE.</p>

<p>CVE-2024-2222 Another vulnerability details.</p>
"""

metadata = {"source": "test_html"}

chunker = IOSHybridChunker()
chunks = chunker.chunk_document(html, metadata)

print(f"Total chunks: {len(chunks)}")

for i, c in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---")
    print("Metadata:", c["metadata"])
    print("Content:", c["content"][:200])