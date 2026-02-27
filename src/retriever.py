from __future__ import annotations
from typing import Any, Dict, List, Optional
from supabase import create_client

from .embedders import EmbedderSpec

class Retriever:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.sb = create_client(supabase_url, supabase_key)

    def retrieve(
        self,
        spec: EmbedderSpec,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        # For 384 RPC: expects (query_embedding, match_count)
        if spec.dim == 384:
            resp = self.sb.rpc(spec.rpc, {"query_embedding": query_embedding, "match_count": k}).execute()
            return resp.data or []

        # For 768 RPC: expects (query_embedding, match_count, filter_model)
        resp = self.sb.rpc(
            spec.rpc,
            {"query_embedding": query_embedding, "match_count": k, "filter_model": spec.model_name},
        ).execute()
        return resp.data or []