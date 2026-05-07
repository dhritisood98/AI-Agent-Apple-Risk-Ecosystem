"""
Scraper Agent
=============
Fetches Apple bulletin URLs from the sources table, scrapes each one,
stores new/changed snapshots, chunks them, and embeds the chunks.

Wraps the existing scrape_ios_sources + processor + chunk_andembed pipeline.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .base_agent import BaseAgent
import src.scrape_ios_sources as _scraper_mod
import src.processor.ingest_to_db as _ingest_mod
import src.chunk_andembed as _embed_mod


class ScraperAgent(BaseAgent):
    name = "scraper"

    def run(self) -> dict:
        self._log_start()

        # Step 1: Scrape all registered Apple bulletin URLs
        print("\n[1/3] Scraping Apple bulletin sources...")
        _scraper_mod.main()

        # Step 2: Chunk new snapshots into snapshot_chunks
        print("\n[2/3] Chunking new snapshots...")
        _ingest_mod.main()

        # Step 3: Embed new chunks into vector_chunks_768
        print("\n[3/3] Embedding new chunks...")
        _embed_mod.main()

        result = {"status": "ok"}
        self._log_done(result)
        return result
