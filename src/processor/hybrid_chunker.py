import re
from typing import Any, Dict, List, Optional


class IOSHybridChunker:
    """
    IOSHybridChunker v2

    Designed for section-level input from scraper output, e.g.:
    parsed = {
        "page_title": "...",
        "ios_version": "...",
        "product_family": "...",
        "sections": [
            {
                "section_title": "...",
                "section_text": "...",
                "section_order": 1
            },
            ...
        ]
    }

    Main goals:
    - preserve source metadata
    - split CVE-heavy Apple security content cleanly
    - fallback safely for non-CVE pages
    """

    def __init__(
        self,
        max_tokens: int = 800,
        min_tokens: int = 50,
        overlap: int = 100,
        summary_tokens: int = 120,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap = overlap
        self.summary_tokens = summary_tokens

        self.cve_pattern = re.compile(r"\bCVE-\d{4}-\d+\b", re.IGNORECASE)
        self.marker_pattern = re.compile(
            r"(?=(?:\bCVE-\d{4}-\d+\b|\bImpact:\b|\bDescription:\b|\bReleased\b))",
            re.IGNORECASE,
        )

    # =========================
    # Token helpers
    # =========================
    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def token_len(self, text: str) -> int:
        return len(self.tokenize(text))

    # =========================
    # Generic splitting
    # =========================
    def recursive_split(self, text: str) -> List[str]:
        """
        Token-based split with overlap.
        Used as fallback for long blocks.
        """
        tokens = self.tokenize(text)
        n = len(tokens)

        if n <= self.max_tokens:
            return [text]

        chunks = []
        start = 0

        while start < n:
            end = min(start + self.max_tokens, n)
            chunk = self.detokenize(tokens[start:end]).strip()
            if chunk:
                chunks.append(chunk)

            if end == n:
                break

            start = max(end - self.overlap, 0)

        return chunks

    # =========================
    # Summary builder
    # =========================
    def build_summary(self, text: str) -> str:
        """
        Use leading tokens as a lightweight summary preview.
        Better than fixed chars because it avoids awkward truncation.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return ""

        summary = self.detokenize(tokens[: self.summary_tokens]).strip()
        return summary

    # =========================
    # Apple security block splitting
    # =========================
    def split_security_blocks(self, text: str) -> List[str]:
        """
        Split a section into meaningful security blocks using markers:
        - Released
        - Impact:
        - Description:
        - CVE-xxxx-xxxxx

        This works well for Apple security content pages.
        """
        if not text.strip():
            return []

        parts = [p.strip() for p in self.marker_pattern.split(text) if p.strip()]
        return parts if parts else [text]

    def extract_cve_blocks(self, text: str) -> List[str]:
        """
        Keep only blocks that contain at least one CVE reference.
        """
        blocks = self.split_security_blocks(text)
        return [b for b in blocks if self.cve_pattern.search(b)]

    # =========================
    # Metadata helper
    # =========================
    def build_chunk_metadata(
        self,
        base_metadata: Dict[str, Any],
        section: Dict[str, Any],
        chunk_type: str,
        chunk_index: int,
        part: Optional[int] = None,
    ) -> Dict[str, Any]:
        metadata = {
            **base_metadata,
            "section_title": section.get("section_title"),
            "section_order": section.get("section_order"),
            "chunk_type": chunk_type,
            "chunk_index": chunk_index,
        }

        if part is not None:
            metadata["part"] = part

        return metadata

    # =========================
    # Chunk a single section
    # =========================
    def chunk_section(
        self,
        section: Dict[str, Any],
        base_metadata: Dict[str, Any],
        start_chunk_index: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Input:
            section = {
                "section_title": "...",
                "section_text": "...",
                "section_order": 1
            }

        Output:
            list of chunk dicts
        """
        all_chunks: List[Dict[str, Any]] = []
        text = (section.get("section_text") or "").strip()

        if not text:
            return all_chunks

        chunk_index = start_chunk_index

        # -----------------------------------
        # 1) Add a lightweight section summary
        # -----------------------------------
        summary = self.build_summary(text)
        if summary:
            all_chunks.append(
                {
                    "content": f"[SECTION SUMMARY]\n{section.get('section_title', 'Untitled Section')}\n\n{summary}",
                    "metadata": self.build_chunk_metadata(
                        base_metadata=base_metadata,
                        section=section,
                        chunk_type="SECTION_SUMMARY",
                        chunk_index=chunk_index,
                    ),
                }
            )
            chunk_index += 1

        # -----------------------------------
        # 2) Split into security-aware blocks
        # -----------------------------------
        security_blocks = self.split_security_blocks(text)

        for block in security_blocks:
            block = block.strip()
            if not block:
                continue

            tokens = self.tokenize(block)
            has_cve = self.cve_pattern.search(block) is not None

            # Too short block
            if len(tokens) < self.min_tokens:
                all_chunks.append(
                    {
                        "content": block,
                        "metadata": self.build_chunk_metadata(
                            base_metadata=base_metadata,
                            section=section,
                            chunk_type="CVE_SHORT" if has_cve else "GENERIC_SHORT",
                            chunk_index=chunk_index,
                        ),
                    }
                )
                chunk_index += 1
                continue

            # Normal-size block
            if len(tokens) <= self.max_tokens:
                all_chunks.append(
                    {
                        "content": block,
                        "metadata": self.build_chunk_metadata(
                            base_metadata=base_metadata,
                            section=section,
                            chunk_type="CVE" if has_cve else "GENERIC_BLOCK",
                            chunk_index=chunk_index,
                        ),
                    }
                )
                chunk_index += 1
                continue

            # Long block -> recursive split
            split_parts = self.recursive_split(block)
            for idx, part_text in enumerate(split_parts, start=1):
                all_chunks.append(
                    {
                        "content": part_text,
                        "metadata": self.build_chunk_metadata(
                            base_metadata=base_metadata,
                            section=section,
                            chunk_type="CVE_LONG_SPLIT" if has_cve else "GENERIC_LONG_SPLIT",
                            chunk_index=chunk_index,
                            part=idx,
                        ),
                    }
                )
                chunk_index += 1

        return all_chunks

    # =========================
    # Chunk an entire parsed doc
    # =========================
    def chunk_parsed_document(self, parsed: Dict[str, Any], source_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Accepts parsed output from scraper, e.g.:

        parsed = {
            "page_title": "...",
            "ios_version": "...",
            "product_family": "...",
            "published_date": "...",
            "sections": [...]
        }

        source_metadata can additionally include:
        {
            "source_id": ...,
            "source_name": ...,
            "source_url": ...
        }
        """
        source_metadata = source_metadata or {}

        base_metadata = {
            "page_title": parsed.get("page_title"),
            "ios_version": parsed.get("ios_version"),
            "product_family": parsed.get("product_family"),
            "published_date": parsed.get("published_date"),
            **source_metadata,
        }

        sections = parsed.get("sections") or []
        all_chunks: List[Dict[str, Any]] = []
        chunk_index = 1

        for section in sections:
            section_chunks = self.chunk_section(
                section=section,
                base_metadata=base_metadata,
                start_chunk_index=chunk_index,
            )
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return all_chunks