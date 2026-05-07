from __future__ import annotations
from typing import Any, Dict, List, Optional
from supabase import create_client
from src.similarity import rerank_by_cosine


class Retriever:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.sb = create_client(supabase_url, supabase_key)

    def _run_rpc(
        self,
        rpc_name: str,
        query_embedding: List[float],
        k: int = 5,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        rpc_params = {
            "query_embedding": query_embedding,
            "match_count": k,
        }

        if extra_params:
            rpc_params.update(extra_params)

        try:
            resp = self.sb.rpc(rpc_name, rpc_params).execute()
            return resp.data or []
        except Exception as e:
            print(f"❌ Retrieval Error in RPC '{rpc_name}': {e}")
            return []

    def retrieve(
        self,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Default bulletin retrieval path.
        Keep this on the RPC that matches your bulletin embeddings.
        """
        return self._run_rpc(
            rpc_name="match_chunks_384",
            query_embedding=query_embedding,
            k=k,
        )

    def retrieve_code_knowledge(
        self,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Code-specific retrieval path for Nomic 768 embeddings.
        """
        return self._run_rpc(
            rpc_name="match_code_knowledge_768",
            query_embedding=query_embedding,
            k=k,
        )

    def retrieve_with_rpc(
        self,
        rpc_name: str,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Flexible retrieval path when you want to choose the RPC dynamically.
        """
        return self._run_rpc(
            rpc_name=rpc_name,
            query_embedding=query_embedding,
            k=k,
        )

    def retrieve_and_rerank(
        self,
        query_embedding: List[float],
        k: int = 5,
        fetch_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch fetch_k bulletin candidates then re-rank top-k by cosine similarity."""
        candidates = self.retrieve(query_embedding, k=fetch_k)
        return rerank_by_cosine(query_embedding, candidates)[:k]

    def retrieve_code_knowledge_with_rerank(
        self,
        query_embedding: List[float],
        k: int = 5,
        fetch_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch fetch_k code-knowledge candidates then re-rank top-k by cosine similarity."""
        candidates = self.retrieve_code_knowledge(query_embedding, k=fetch_k)
        return rerank_by_cosine(query_embedding, candidates)[:k]