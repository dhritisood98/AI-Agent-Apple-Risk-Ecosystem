import re
from typing import List, Dict

class IOSDocumentChunker:
    """
    Splits long iOS security documents into smaller, semantically consistent 
    pieces for vector search.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Hierarchy of separators: Paragraphs -> Newlines -> Sentences -> Words
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def clean_text(self, text: str) -> str:
        """Removes HTML artifacts and excessive whitespace."""
        # Collapse all whitespace into single spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove common Apple support boilerplate phrases
        boilerplate = [
            "Shop the Latest Mac iPad iPhone",
            "Apple Store Shop",
            "Accessories Apple Trade In"
        ]
        for phrase in boilerplate:
            text = text.replace(phrase, "")
        return text.strip()

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting logic to respect document structure."""
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0]
        remaining_seps = separators[1:]
        
        splits = text.split(separator)
        final_chunks = []
        current_chunk = ""

        for s in splits:
            # Check if adding the next split exceeds our limit
            if len(current_chunk) + len(s) + len(separator) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                
                # If a single split is still too large, go to the next separator level
                if len(s) > self.chunk_size:
                    final_chunks.extend(self._split_text(s, remaining_seps))
                    current_chunk = ""
                else:
                    current_chunk = s
            else:
                # Build the chunk with the current separator
                current_chunk += (separator if current_chunk else "") + s

        if current_chunk:
            final_chunks.append(current_chunk.strip())

        return final_chunks

    def chunk_document(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Main entry point: cleans, splits, and injects context.
        """
        cleaned_text = self.clean_text(text)
        raw_chunks = self._split_text(cleaned_text, self.separators)
        
        processed_chunks = []
        for i, content in enumerate(raw_chunks):
            # Context Injection: Prepend the version so the vector is 'aware' of it
            version = metadata.get("version", "iOS 18")
            contextual_content = f"[{version} UPDATE] {content}"
            
            processed_chunks.append({
                "content": contextual_content,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "char_count": len(content)
                }
            })
            
        return processed_chunks