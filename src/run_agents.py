"""
Local Agent Orchestrator
========================
Runs the full Sentinel system pipeline locally.

Usage:
    python -m src.run_agents                    # full pipeline
    python -m src.run_agents --only scraper     # scraper only
    python -m src.run_agents --only sentinel    # sentinel only
    python -m src.run_agents --only coordinator # coordinator only (needs sentinel results)
    python -m src.run_agents --skip scraper     # skip scraping (use existing bulletins)
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import settings
from src.agents import ScraperAgent, SentinelAgent, CoordinatorAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def run_pipeline(only: str | None = None, skip: str | None = None) -> None:
    started = datetime.now(timezone.utc)
    print(f"\n{'#'*60}")
    print(f"  SENTINEL SYSTEM  —  {started.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'#'*60}")

    url = settings.supabase_url
    key = settings.supabase_key

    results = {}

    # ── Scraper ────────────────────────────────────────────────────────────────
    if only in (None, "scraper") and skip != "scraper":
        try:
            results["scraper"] = ScraperAgent(url, key).run()
        except Exception as e:
            print(f"\n❌  ScraperAgent failed: {e}")
            results["scraper"] = {"error": str(e)}
    else:
        print("\n  [scraper] skipped")

    # ── Sentinel ───────────────────────────────────────────────────────────────
    if only in (None, "sentinel") and skip != "sentinel":
        try:
            results["sentinel"] = SentinelAgent(url, key).run()
        except Exception as e:
            print(f"\n❌  SentinelAgent failed: {e}")
            results["sentinel"] = {"error": str(e)}
    else:
        print("\n  [sentinel] skipped")

    # ── Coordinator ────────────────────────────────────────────────────────────
    if only in (None, "coordinator") and skip != "coordinator":
        try:
            results["coordinator"] = CoordinatorAgent(url, key).run()
        except Exception as e:
            print(f"\n❌  CoordinatorAgent failed: {e}")
            results["coordinator"] = {"error": str(e)}
    else:
        print("\n  [coordinator] skipped")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE  —  {elapsed:.1f}s total")
    for agent, result in results.items():
        print(f"  {agent:<15} {result}")
    print(f"{'#'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run the Sentinel multi-agent pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only", choices=["scraper", "sentinel", "coordinator"],
                       help="Run only this agent")
    group.add_argument("--skip", choices=["scraper", "sentinel", "coordinator"],
                       help="Skip this agent, run the rest")
    args = parser.parse_args()
    run_pipeline(only=args.only, skip=args.skip)


if __name__ == "__main__":
    main()
