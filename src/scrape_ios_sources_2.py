# src/scrape_ios_sources_2.py

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup, Tag

from .db import get_supabase_client
from .config import USER_AGENT
from .hybrid_chunker import IOSHybridChunker

HEADERS = {"User-Agent": USER_AGENT}

MIN_CLEAN_TEXT_LEN = 300
MAX_PREVIEW_TEXT_CHARS = 25000
REQUEST_TIMEOUT_S = 45

# Debug flags
DEBUG_SINGLE_URL = True
DEBUG_URL = "https://support.apple.com/en-us/120304"
DEBUG_FETCH = False

# Chunk export
ENABLE_CHUNK_EXPORT = True
EXPORT_DIR = "data/chunk_debug"
EXPORT_LIMIT = 1 

NOISE_PATTERNS = [
    r"Helpful\?\s*Yes\s*No",
    r"Character limit:\s*\d+",
    r"Maximum character limit is \d+\.",
    r"Please don.?t include any personal information.*",
    r"Submit\s*Thanks for your feedback\.?",
    r"Apple makes no representations regarding third-party website accuracy or reliability\.",
    r"Contact the vendor\s*for additional information\.",
    r"Published Date:\s*[A-Za-z]+\s+\d{1,2},\s+\d{4}",
    r"\bBack to top\b",
    r"\bLearn more\b",
]

IOS_VERSION_PATTERNS = [
    r"\biOS\s+\d+(?:\.\d+){0,2}\b",
    r"\biPadOS\s+\d+(?:\.\d+){0,2}\b",
    r"\bwatchOS\s+\d+(?:\.\d+){0,2}\b",
    r"\bmacOS\s+\d+(?:\.\d+){0,2}\b",
    r"\bSafari\s+\d+(?:\.\d+){0,2}\b",
    r"\bvisionOS\s+\d+(?:\.\d+){0,2}\b",
    r"\btvOS\s+\d+(?:\.\d+){0,2}\b",
]

PUBLISHED_DATE_PATTERNS = [
    r"Published Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    r"Released\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    r"Updated\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
]

NAV_LINES_TO_DROP = {
    "Apple",
    "Store",
    "Mac",
    "iPad",
    "iPhone",
    "Watch",
    "Vision",
    "AirPods",
    "TV & Home",
    "Entertainment",
    "Accessories",
    "Support",
    "+",
    "0",
}

NOISE_LINE_PATTERNS = [
    r"^Apple Support$",
    r"^Table of Contents$",
    r"^United States$",
    r"^Copyright © Apple Inc\..*$",
    r"^All rights reserved\.$",
    r"^View in English$",
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


def _cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return head.rstrip() + "\n\n[...truncated...]\n\n" + tail.lstrip()


def _slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "untitled"


def _remove_noise_tags(root: Tag) -> None:
    for tag in root.find_all(["script", "style", "noscript"]):
        tag.decompose()


def _remove_navigation_lines(text: str) -> str:
    lines = [line.strip() for line in text.split("\n")]
    kept: List[str] = []

    for line in lines:
        if not line:
            continue

        if line in NAV_LINES_TO_DROP:
            continue

        should_drop = False
        for pat in NOISE_LINE_PATTERNS:
            if re.match(pat, line, flags=re.IGNORECASE):
                should_drop = True
                break

        if should_drop:
            continue

        kept.append(line)

    return "\n".join(kept).strip()


def _remove_noise_text(text: str) -> str:
    cleaned = text or ""
    cleaned = _remove_navigation_lines(cleaned)

    for pat in NOISE_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)

    return _normalize_ws(cleaned)


def _extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return _normalize_ws(h1.get_text(" ", strip=True))

    if soup.title and soup.title.text:
        title = _normalize_ws(soup.title.text)
        title = re.sub(r"\s*-\s*Apple Support\s*$", "", title, flags=re.IGNORECASE)
        return title

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return _normalize_ws(og_title["content"])

    return "Untitled Apple Support Page"


def _dedupe_paragraphs(text: str, min_para_len: int = 25) -> str:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    seen = set()
    kept: List[str] = []

    for p in paragraphs:
        p_norm = _normalize_ws(p)
        if len(p_norm) < min_para_len:
            continue

        p_key = p_norm.lower()
        if p_key in seen:
            continue

        seen.add(p_key)
        kept.append(p_norm)

    return "\n".join(kept).strip()


def _extract_published_date(raw_text: str) -> Optional[str]:
    for pat in PUBLISHED_DATE_PATTERNS:
        m = re.search(pat, raw_text, flags=re.IGNORECASE)
        if m:
            return _normalize_ws(m.group(1))
    return None


def _extract_ios_version(text: str) -> Optional[str]:
    for pat in IOS_VERSION_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return _normalize_ws(m.group(0))
    return None


def _extract_product_family(text: str) -> Optional[str]:
    lowered = text.lower()
    if "ipados" in lowered:
        return "iPadOS"
    if "ios" in lowered:
        return "iOS"
    if "watchos" in lowered:
        return "watchOS"
    if "macos" in lowered:
        return "macOS"
    if "safari" in lowered:
        return "Safari"
    if "visionos" in lowered:
        return "visionOS"
    if "tvos" in lowered:
        return "tvOS"
    return None


def _extract_sections_from_text(text: str) -> List[Dict[str, Any]]:
    if not text.strip():
        return []

    marker_pattern = re.compile(
        r"(?=(?:\bCVE-\d{4}-\d+\b|\bImpact:\b|\bDescription:\b|\bReleased\b))"
    )

    parts = [p.strip() for p in marker_pattern.split(text) if p.strip()]
    sections: List[Dict[str, Any]] = []
    order = 1

    for part in parts:
        if len(part) < 80:
            continue

        first_line = part.split("\n", 1)[0].strip()
        section_title = first_line[:120] if first_line else f"Section {order}"

        sections.append(
            {
                "section_title": section_title,
                "section_text": part,
                "section_order": order,
            }
        )
        order += 1

    if sections:
        return sections

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    sections = []
    order = 1

    for block in blocks:
        if len(block) < 120:
            continue

        first_line = block.split("\n", 1)[0].strip()
        section_title = first_line[:120] if first_line else f"Section {order}"

        sections.append(
            {
                "section_title": section_title,
                "section_text": block,
                "section_order": order,
            }
        )
        order += 1

    if not sections and text.strip():
        sections.append(
            {
                "section_title": "Full Page Content",
                "section_text": text.strip(),
                "section_order": 1,
            }
        )

    return sections


def fetch_parse_and_clean(url: str) -> Dict[str, Any]:
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

    _remove_noise_tags(soup)
    raw_text = soup.get_text("\n", strip=True)

    if DEBUG_FETCH:
        print("\n===== DEBUG FETCH =====", flush=True)
        print(f"url: {url}", flush=True)
        print(f"raw_html_len: {len(raw_html)}", flush=True)
        print(f"body_text_preview_before_clean: {raw_text[:500]}", flush=True)

    clean_text_full = _normalize_ws(raw_text)
    clean_text_full = _remove_noise_text(clean_text_full)
    clean_text_full = _dedupe_paragraphs(clean_text_full)

    clean_text_preview = _cap_text(clean_text_full, MAX_PREVIEW_TEXT_CHARS)
    sections = _extract_sections_from_text(clean_text_full)

    published_date = _extract_published_date(raw_html) or _extract_published_date(clean_text_full)
    ios_version = _extract_ios_version(f"{title}\n{clean_text_full}")
    product_family = _extract_product_family(f"{title}\n{clean_text_full}")

    return {
        "raw_html": raw_html,
        "page_title": title,
        "clean_text_full": clean_text_full,
        "clean_text_preview": clean_text_preview,
        "clean_text_len": len(clean_text_full),
        "sections": sections,
        "published_date": published_date,
        "ios_version": ios_version,
        "product_family": product_family,
    }


def _build_snapshot_payload(
    source_id: Any,
    source_url: str,
    source_name: str,
    agent_name: str,
    fetched_at: str,
    content_hash: str,
    parsed: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "source_id": source_id,
        "fetched_at": fetched_at,
        "content_hash": content_hash,
        "raw_text": parsed["raw_html"],
        "clean_text": parsed["clean_text_preview"],
        "agent_name": agent_name,
    }


def _get_last_snapshot_hash(sb, source_id: Any) -> Optional[str]:
    result = (
        sb.table("snapshots")
        .select("content_hash")
        .eq("source_id", source_id)
        .order("fetched_at", desc=True)
        .limit(1)
        .execute()
    )

    if result.data and len(result.data) > 0:
        return result.data[0].get("content_hash")

    return None


def _generate_chunks(parsed: Dict[str, Any], src_id: Any, source_name: str, url: str) -> List[Dict[str, Any]]:
    chunker = IOSHybridChunker(
        max_tokens=800,
        min_tokens=50,
        overlap=100,
        summary_tokens=120,
    )

    return chunker.chunk_parsed_document(
        parsed=parsed,
        source_metadata={
            "source_id": src_id,
            "source_name": source_name,
            "source_url": url,
        },
    )


def _export_chunks_to_json(
    parsed: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    src_id: Any,
    source_name: str,
    url: str,
) -> str:
    export_path = Path(EXPORT_DIR)
    export_path.mkdir(parents=True, exist_ok=True)

    title_slug = _slugify(parsed.get("page_title", "untitled"))
    file_name = f"{src_id}_{title_slug}.json"
    file_path = export_path / file_name

    payload = {
        "source_id": src_id,
        "source_name": source_name,
        "source_url": url,
        "page_title": parsed.get("page_title"),
        "ios_version": parsed.get("ios_version"),
        "product_family": parsed.get("product_family"),
        "published_date": parsed.get("published_date"),
        "clean_text_len": parsed.get("clean_text_len"),
        "section_count": len(parsed.get("sections", [])),
        "chunk_count": len(chunks),
        "sections": parsed.get("sections", []),
        "chunks": chunks,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(file_path)


def main():
    sb = get_supabase_client()

    if DEBUG_SINGLE_URL:
        parsed = fetch_parse_and_clean(DEBUG_URL)
        print("\n===== DEBUG RESULT =====", flush=True)
        print("page_title:", parsed["page_title"], flush=True)
        print("clean_text_len:", parsed["clean_text_len"], flush=True)
        print("ios_version:", parsed["ios_version"], flush=True)
        print("product_family:", parsed["product_family"], flush=True)
        print("section_count:", len(parsed["sections"]), flush=True)

        chunks = _generate_chunks(parsed, "debug_source", "debug_name", DEBUG_URL)
        print("chunk_count:", len(chunks), flush=True)

        if ENABLE_CHUNK_EXPORT:
            out = _export_chunks_to_json(parsed, chunks, "debug_source", "debug_name", DEBUG_URL)
            print("chunk_export_path:", out, flush=True)
        return

    try:
        sources = (
            sb.table("sources")
            .select("id,name,url,agent_name")
            .eq("active", True)
            .eq("agent_name", "ios-risk-agent")
            .execute()
            .data
        )
    except Exception as e:
        print(f"❌ Failed loading sources: err={e}", flush=True)
        return

    print(f"Found {len(sources)} active iOS sources", flush=True)

    inserted = 0
    skipped = 0
    failed = 0
    exported = 0

    for idx, s in enumerate(sources, start=1):
        now = _utc_now_iso()

        src_id = s["id"]
        source_name = s["name"]
        url = s["url"]
        agent_name = s["agent_name"]

        print(f"\n🔎 Processing: {source_name} | url={url}", flush=True)

        try:
            parsed = fetch_parse_and_clean(url)
        except Exception as e:
            failed += 1
            print(
                f"⚠️ Fetch/parse failed (skip): {source_name} | url={url} | err={type(e).__name__}: {e}",
                flush=True,
            )
            continue

        clean_text_full = parsed["clean_text_full"]
        page_title = parsed["page_title"]

        print(
            f"🧠 Parsed: title={page_title} | len={parsed['clean_text_len']} | ios_version={parsed['ios_version']} | sections={len(parsed['sections'])}",
            flush=True,
        )

        if len(clean_text_full) < MIN_CLEAN_TEXT_LEN:
            skipped += 1
            print(
                f"⏭️ Skipped (too short): {source_name} | title={page_title} | len={len(clean_text_full)}",
                flush=True,
            )
            continue

        new_hash = _sha256(clean_text_full)

        try:
            last_hash = _get_last_snapshot_hash(sb, src_id)
        except Exception as e:
            failed += 1
            print(f"⚠️ Failed reading last snapshot: {source_name} | err={e}", flush=True)
            continue

        if last_hash == new_hash:
            skipped += 1
            print(
                f"⏭️ No change detected: {source_name} | title={page_title}",
                flush=True,
            )
        else:
            payload = _build_snapshot_payload(
                source_id=src_id,
                source_url=url,
                source_name=source_name,
                agent_name=agent_name,
                fetched_at=now,
                content_hash=new_hash,
                parsed=parsed,
            )

            try:
                sb.table("snapshots").insert(payload).execute()
                inserted += 1
                print(
                    f"✅ Stored NEW iOS snapshot: {source_name} | title={page_title} | len={parsed['clean_text_len']}",
                    flush=True,
                )
            except Exception as e:
                failed += 1
                print(
                    f"⚠️ DB insert failed: {source_name} | title={page_title} | err={type(e).__name__}: {e}",
                    flush=True,
                )
                continue

        if ENABLE_CHUNK_EXPORT and exported < EXPORT_LIMIT:
            chunks = _generate_chunks(parsed, src_id, source_name, url)
            out = _export_chunks_to_json(parsed, chunks, src_id, source_name, url)
            exported += 1
            print(f"📦 Exported chunks -> {out}", flush=True)

    print(
        f"\n🏁 Done. inserted={inserted} skipped={skipped} failed={failed} exported={exported}",
        flush=True,
    )


if __name__ == "__main__":
    main()