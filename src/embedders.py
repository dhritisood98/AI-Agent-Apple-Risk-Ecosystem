from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional

from sentence_transformers import SentenceTransformer

EmbedderName = Literal["bge_small", "e5_base", "mpnet", "nomic_768"]

@dataclass
class EmbedderSpec:
    name: EmbedderName
    model_name: str
    dim: int
    table: str
    rpc: str

def get_embedder_spec(name: EmbedderName) -> EmbedderSpec:
    if name == "bge_small":
        return EmbedderSpec(
            name=name,
            model_name="BAAI/bge-small-en-v1.5",
            dim=384,
            table="vector_chunks_384",
            rpc="match_chunks_384",
        )
    if name == "e5_base":
        return EmbedderSpec(
            name=name,
            model_name="intfloat/e5-base-v2",
            dim=768,
            table="vector_chunks_768",
            rpc="match_chunks_768",
        )
    if name == "mpnet":
        return EmbedderSpec(
            name=name,
            model_name="sentence-transformers/all-mpnet-base-v2",
            dim=768,
            table="vector_chunks_768",
            rpc="match_chunks_768",
        )
    if name == "nomic_768":
        return EmbedderSpec(
            name=name,
            model_name="nomic-ai/nomic-embed-text-v1.5",
            dim=768,
            table="vector_chunks_768",
            rpc="match_chunks_768",
        )
    raise ValueError(f"Unknown embedder: {name}")

class Embedder:
    def __init__(self, spec: EmbedderSpec):
        self.spec = spec
        trust = True if "nomic" in spec.model_name else False
        self.model = SentenceTransformer(spec.model_name, trust_remote_code=trust)

    def embed_query(self, text: str) -> List[float]:
        # Nomic benefits from prefixes: search_query/search_document
        if self.spec.name.startswith("nomic"):
            text = f"search_query: {text}"
        return self.model.encode([text], normalize_embeddings=True).tolist()[0]