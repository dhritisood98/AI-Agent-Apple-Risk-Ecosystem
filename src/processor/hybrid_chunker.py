import re
from bs4 import BeautifulSoup


class IOSHybridChunker:
    def __init__(self, max_tokens=800, min_tokens=50, overlap=100):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap = overlap

        # Pattern: "CVE-2024-12345"
        self.cve_pattern = re.compile(r"CVE-\d{4}-\d+")

    # =========================
    # Utility Helpers
    # =========================
    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def recursive_split(self, text):
        """
        Split long CVE blocks into smaller subchunks with overlap
        """
        tokens = self.tokenize(text)
        n = len(tokens)

        if n <= self.max_tokens:
            return [text]

        chunks = []
        start = 0

        while start < n:
            end = min(start + self.max_tokens, n)
            chunk = self.detokenize(tokens[start:end])
            chunks.append(chunk)

            if end == n:
                break

            start = max(end - self.overlap, 0)

        return chunks

    # =========================
    # CVE Extraction per H2 Section
    # =========================
    def extract_cve_blocks(self, h2_text):
        """
        Extract CVE blocks by detecting CVE start lines
        """
        lines = h2_text.split("\n")
        blocks = []
        current = []

        for line in lines:
            if self.cve_pattern.search(line):
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
                current.append(line)
            else:
                if current:
                    current.append(line)

        if current:
            blocks.append("\n".join(current).strip())

        return blocks

    # =========================
    # Main Chunking Logic
    # =========================
    def chunk_document(self, html, metadata):
        soup = BeautifulSoup(html, "html.parser")
        all_chunks = []

        h2_tags = soup.find_all("h2")

        for h2 in h2_tags:
            section_title = h2.get_text().strip()

            section_metadata = {
                **metadata,
                "section": section_title,
                "chunk_type": "H2",
            }

            # ✅ FIX: use "\n" instead of " "
            section_content = []
            node = h2.find_next_sibling()

            while node and node.name != "h2":
                section_content.append(node.get_text("\n", strip=True))
                node = node.find_next_sibling()

            h2_text = "\n".join(section_content).strip()

            if not h2_text:
                continue

            # =========================
            # Step 1: Section Summary
            # =========================
            summary = h2_text[:600]

            all_chunks.append({
                "content": f"[SECTION SUMMARY]\n{section_title}\n\n{summary}...",
                "metadata": section_metadata
            })

            # =========================
            # Step 2: CVE Blocks
            # =========================
            cve_blocks = self.extract_cve_blocks(h2_text)

            for block in cve_blocks:
                tokens = self.tokenize(block)

                # 🔹 Too short → keep but mark merged
                if len(tokens) < self.min_tokens:
                    all_chunks.append({
                        "content": f"[MERGED CVE]\n{section_title}\n{block}",
                        "metadata": {
                            **section_metadata,
                            "chunk_type": "CVE_MERGED"
                        }
                    })
                    continue

                # 🔹 Too long → split
                if len(tokens) > self.max_tokens:
                    parts = self.recursive_split(block)

                    for idx, part in enumerate(parts):
                        all_chunks.append({
                            "content": part,
                            "metadata": {
                                **section_metadata,
                                "chunk_type": "CVE_LONG_SPLIT",
                                "part": idx + 1
                            }
                        })
                    continue

                # 🔹 Normal CVE chunk
                all_chunks.append({
                    "content": block,
                    "metadata": {
                        **section_metadata,
                        "chunk_type": "CVE"
                    }
                })

        return all_chunks