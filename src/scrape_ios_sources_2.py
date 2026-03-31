# src/scrape_ios_sources_2.py
# iOS-only scraper: pulls sources where agent_name == "ios-risk-agent"
# Records changes ONLY if content_hash differs from the last stored version.
# Improved with:
# - noise cleaning
# - title extraction
# - DB insert error handling (won't crash whole run)
# - clearer logging

import hashlib
import re
from datetime import datetime, timezone
from typing import Tuple

import requests
from bs4 import BeautifulSoup

from .db import get_supabase_client
from .config import USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}

MIN_CLEAN_TEXT_LEN = 300
MAX_CLEAN_TEXT_CHARS = 25000
REQUEST_TIMEOUT_S = 45

NOISE_PATTERNS = [
    r"Helpful\?\s*Yes\s*No",
    r"Character limit:\s*\d+",
    r"Maximum character limit is \d+\.",
    r"Please don.?t include any personal information.*",
    r"Submit\s*Thanks for your feedback\.?",
    r"Apple makes no representations regarding third-party website accuracy or reliability\.",
    r"Contact the vendor\s*for additional information\.",
    r"Published Date:\s*[A-Za-z]+\s+\d{1,2},\s+\d{4}",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_noise_tags(root) -> None:
    for tag in root.find_all(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()
    for tag in root.find_all(["nav", "footer", "header", "aside"]):
        tag.decompose()


def _remove_noise_text(text: str) -> str:
    cleaned = text or ""
    for pat in NOISE_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return _normalize_ws(cleaned)


def _pick_root(soup: BeautifulSoup):
    return soup.find("main") or soup.find("article") or soup.find("body") or soup


def _cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3) :]
    return head.rstrip() + "\n\n[...truncated...]\n\n" + tail.lstrip()


def _extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.text:
        return _normalize_ws(soup.title.text)
    h1 = soup.find("h1")
    if h1:
        return _normalize_ws(h1.get_text(" ", strip=True))
    return "Untitled Apple Support Page"


def fetch_raw_and_clean(url: str) -> Tuple[str, str, str]:
    resp = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT_S,
        allow_redirects=True,
    )
    resp.raise_for_status()

    raw_html = resp.text or ""
    soup = BeautifulSoup(raw_html, "html.parser")

    title = _extract_title(soup)

    root = _pick_root(soup)
    _remove_noise_tags(root)

    clean_text = _normalize_ws(root.get_text("\n", strip=True))
    clean_text = _remove_noise_text(clean_text)
    clean_text = _cap_text(clean_text, MAX_CLEAN_TEXT_CHARS)

    return raw_html, clean_text, title


def main():
    sb = get_supabase_client()

    sources = (
        sb.table("sources")
        .select("id,name,url,agent_name")
        .eq("active", True)
        .eq("agent_name", "ios-risk-agent")
        .execute()
        .data
    )

    print(f"Found {len(sources)} active iOS sources", flush=True)

    inserted = 0
    skipped = 0
    failed = 0

    for s in sources:
        now = _utc_now_iso()

        src_id = s["id"]
        source_name = s["name"]
        url = s["url"]
        agent_name = s["agent_name"]

        # 1) Fetch page
        try:
            raw_html, clean_text, page_title = fetch_raw_and_clean(url)
        except Exception as e:
            failed += 1
            print(f"⚠️ Fetch failed (skip): {source_name} url={url} err={e}", flush=True)
            continue

        # 2) Skip very short pages
        if len(clean_text) < MIN_CLEAN_TEXT_LEN:
            skipped += 1
            print(
                f"Skipped (too short): {source_name} | title={page_title} | len={len(clean_text)}",
                flush=True,
            )
            continue

        # 3) Optional: skip obvious non-security/update pages
        # Uncomment if you want stricter filtering
        # title_lower = page_title.lower()
        # if "security content" not in title_lower and "ios 18 update" not in title_lower:
        #     skipped += 1
        #     print(f"Skipped (non-security page): {source_name} | title={page_title}", flush=True)
        #     continue

        # 4) Change detection
        new_hash = _sha256(clean_text)

        try:
            last_entry = (
                sb.table("snapshots")
                .select("content_hash")
                .eq("source_id", src_id)
                .order("fetched_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as e:
            failed += 1
            print(f"⚠️ Failed reading last snapshot: {source_name} err={e}", flush=True)
            continue

        if last_entry.data and last_entry.data[0].get("content_hash") == new_hash:
            skipped += 1
            print(
                f"⏭️ No change detected: {source_name} | title={page_title}",
                flush=True,
            )
            continue

        # 5) Insert new snapshot
        payload = {
            "source_id": src_id,
            "fetched_at": now,
            "content_hash": new_hash,
            "raw_text": raw_html,
            "clean_text": clean_text,
            "agent_name": agent_name,
        }

        try:
            sb.table("snapshots").insert(payload).execute()
            inserted += 1
            print(
                f"✅ Stored NEW iOS snapshot: {source_name} | title={page_title}",
                flush=True,
            )
        except Exception as e:
            failed += 1
            print(
                f"⚠️ DB insert failed: {source_name} | title={page_title} | err={e}",
                flush=True,
            )
            continue

    print(f"🏁 Done. inserted={inserted} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()