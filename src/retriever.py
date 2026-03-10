from __future__ import annotations
from typing import Any, Dict, List
from supabase import create_client


class Retriever:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.sb = create_client(supabase_url, supabase_key)

    def retrieve(
        self,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[Dict[str, Any]]:

        rpc_params = {
            "query_embedding": query_embedding,
            "match_count": k
        }

        try:
            resp = self.sb.rpc("match_chunks_384", rpc_params).execute()
            return resp.data or []
        except Exception as e:
            print(f"❌ Retrieval Error: {e}")
            return []