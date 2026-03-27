"""Base class shared by all agents."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from supabase import create_client, Client


class BaseAgent:
    name: str = "base"

    def __init__(self, supabase_url: str, supabase_key: str):
        self.sb: Client = create_client(supabase_url, supabase_key)
        self.log = logging.getLogger(self.name)
        self._started_at: datetime | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Entry point. Returns a result dict."""
        raise NotImplementedError

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _log_start(self):
        self._started_at = datetime.now(timezone.utc)
        self.log.info(f"[{self.name}] starting at {self._started_at.isoformat()}")
        print(f"\n{'='*60}")
        print(f"  {self.name.upper()} AGENT  —  {self._started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{'='*60}")

    def _log_done(self, result: dict):
        elapsed = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        print(f"\n  Done in {elapsed:.1f}s  |  {result}")
        self.log.info(f"[{self.name}] done in {elapsed:.1f}s: {result}")
