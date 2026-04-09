import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from supabase import create_client
from src.config import settings
from src.zero_shot import classify_risk_zs, classify_risk_zs_with_scores
from src.embedders import get_embedder_spec, Embedder
from src.retriever import Retriever
from src.similarity import rerank_by_cosine
from src.llm_clients import NIMClient, NoLLM
from src.prompts import build_prompt


st.set_page_config(
    page_title="iOS Risk Sentinel",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #ffffff !important;
    color: #111827 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 1400px !important;
}

.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.75rem;
}
.page-title { font-size: 1.3rem; font-weight: 700; color: #111827; letter-spacing: -0.02em; }
.page-subtitle { font-size: 0.81rem; color: #6b7280; margin-top: 0.2rem; }
.live-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    color: #15803d; font-size: 0.72rem; font-weight: 600;
    padding: 0.3rem 0.75rem; border-radius: 999px;
}
.live-dot {
    width: 6px; height: 6px; background: #16a34a;
    border-radius: 50%; animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* kpi cards */
.kpi {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 1rem 1.15rem;
    position: relative; overflow: hidden;
}
.kpi::after {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px; border-radius: 10px 10px 0 0;
}
.kpi-blue::after  { background: #2563eb; }
.kpi-slate::after { background: #64748b; }
.kpi-red::after   { background: #dc2626; }
.kpi-amber::after { background: #d97706; }
.kpi-green::after { background: #16a34a; }
.kpi-label { font-size: 0.7rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: .07em; margin-bottom: .45rem; }
.kpi-value { font-size: 1.7rem; font-weight: 700; color: #111827; line-height: 1; margin-bottom: .25rem; }
.kpi-sub   { font-size: 0.73rem; color: #9ca3af; }
.kpi-bar-bg { height: 4px; background: #f3f4f6; border-radius: 99px; margin-top: .55rem; overflow: hidden; }
.kpi-bar-fill { height: 4px; border-radius: 99px; }

.sec { font-size: 0.72rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: .08em; margin: 0 0 .9rem 0; }

.rb { display: inline-block; font-size: 0.7rem; font-weight: 600; padding: .18rem .6rem; border-radius: 999px; letter-spacing: .03em; }
.rb-high   { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
.rb-medium { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.rb-low    { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }

.fc {
    border: 1px solid #e5e7eb; border-left: 3px solid #e5e7eb;
    border-radius: 10px; padding: .9rem 1rem; margin-bottom: .6rem; background: #ffffff;
}
.fc.high   { border-left-color: #dc2626; }
.fc.medium { border-left-color: #d97706; }
.fc.low    { border-left-color: #16a34a; }
.fc-name { font-size: .85rem; font-weight: 600; color: #111827; margin-bottom: .15rem; font-family: 'Courier New', monospace; }
.fc-meta { font-size: .73rem; color: #9ca3af; margin-bottom: .4rem; }
.fc-body { font-size: .81rem; color: #4b5563; line-height: 1.55; }

.ai-result {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-top: 3px solid #2563eb; border-radius: 10px; padding: 1.2rem 1.35rem;
}
.ai-label { font-size: .7rem; font-weight: 600; color: #2563eb; text-transform: uppercase; letter-spacing: .07em; margin-bottom: .7rem; }
.ai-body  { font-size: .87rem; color: #1f2937; line-height: 1.75; }

.ctx-panel {
    background: #fafafa; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 1rem 1.15rem; margin-bottom: .9rem;
}
.ctx-row { display: flex; justify-content: space-between; margin-bottom: .45rem; }
.ctx-key { font-size: .8rem; color: #6b7280; }
.ctx-val { font-size: .8rem; font-weight: 600; color: #111827; }

.rex {
    border: 1px solid #e5e7eb; border-radius: 8px;
    padding: .7rem .9rem; margin-bottom: .55rem;
}
.rex-title { font-size: .73rem; font-weight: 700; margin-bottom: .2rem; }
.rex-body  { font-size: .77rem; color: #4b5563; line-height: 1.5; }

div[data-testid="stDataFrame"] { border: 1px solid #e5e7eb !important; border-radius: 10px !important; overflow: hidden !important; }

div[data-testid="stTextInput"] input {
    border: 1px solid #d1d5db !important; border-radius: 8px !important;
    font-size: .87rem !important; color: #111827 !important; background: #fff !important;
}
div[data-testid="stSelectbox"] > div { border: 1px solid #d1d5db !important; border-radius: 8px !important; }

div[data-testid="stButton"] button {
    background: #374151 !important; color: #f9fafb !important; border: none !important;
    border-radius: 8px !important; font-size: .83rem !important; font-weight: 600 !important;
}
div[data-testid="stButton"] button:hover { background: #1f2937 !important; }

div[data-testid="stTabs"] button { font-size: .82rem !important; font-weight: 500 !important; color: #6b7280 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #1d4ed8 !important; font-weight: 600 !important; }

hr.div { border: none; border-top: 1px solid #f3f4f6; margin: 1.4rem 0; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def signal_category(text: str) -> str:
    t = (text or "").lower()
    if any(x in t for x in ["kernel","os version","os build","boot time","time zone"]):
        return "OS / System"
    if any(x in t for x in ["identifierforvendor","vendor identifier","idfv"]):
        return "Identifiers"
    if any(x in t for x in ["device name","device type","memory","cpu","screen resolution"]):
        return "Device Attributes"
    if any(x in t for x in ["biometrics","passcode","authentication","local authentication"]):
        return "Authentication"
    if any(x in t for x in ["locale","user interface style","ui style","language","region"]):
        return "Locale / UI"
    if any(x in t for x in ["cellular","carrier","network","reachability","mobile data","sim"]):
        return "Network / Cellular"
    if any(x in t for x in ["keychain","disk space","disk","storage","identifier storage",
                             "secitem","persistent","kSecClass","secure storage",
                             "storeidentifier","loadidentifier","uuid","store identifier",
                             "load identifier"]):
        return "Storage / Keychain"
    if any(x in t for x in ["sha256","sha-256","hash","digest","fingerprint function","checksum"]):
        return "Hashing"
    if any(x in t for x in ["fingerprint tree","compound","tree builder","tree calculator","stability level","compound tree"]):
        return "Fingerprint Core"
    if any(x in t for x in ["configuration","factory","builder","provider","aggregat","library","coordinator"]):
        return "App Infrastructure"
    return "Other"

def shorten(path: str) -> str:
    return (path or "Unknown").split("/")[-1]

def rb_html(level: str) -> str:
    cls = {"High":"rb-high","Medium":"rb-medium","Low":"rb-low"}.get(level,"rb-low")
    return f'<span class="rb {cls}">{level}</span>'

def fc_cls(level: str) -> str:
    return {"High":"fc high","Medium":"fc medium","Low":"fc low"}.get(level,"fc low")

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#374151"),
    margin=dict(l=0, r=0, t=24, b=0),
)


# ── Supabase connection ───────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    # Prefer st.secrets (Streamlit Cloud), fall back to env vars (local)
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except (KeyError, FileNotFoundError):
        url = settings.supabase_url
        key = settings.supabase_key
    return create_client(url, key)

@st.cache_resource
def get_embedder():
    spec = get_embedder_spec("nomic_768")
    return Embedder(spec)

@st.cache_resource
def get_retriever():
    return Retriever(settings.supabase_url, settings.supabase_key)

@st.cache_resource
def get_llm():
    if settings.nvidia_api_key:
        return NIMClient(settings.nvidia_api_key, settings.nim_base_url)
    return NoLLM()

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data(ttl=3600)
def get_raw_source(file_path: str, max_lines: int = 40) -> str:
    """Read the raw Swift source file from the local repo."""
    abs_path = os.path.join(_BASE_DIR, file_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[:max_lines]
        return "".join(lines)
    except FileNotFoundError:
        return f"(source file not found: {file_path})"

@st.cache_data(ttl=300)
def get_related_bulletins(content: str, k: int = 3) -> list:
    """Embed the file summary with bge_small and retrieve related Apple bulletin chunks.

    Bulletins are indexed via chunk_andembed.py using bge_small (384-dim) into
    vector_chunks_384, so the query vector must use the same model.
    """
    embedder = get_embedder()
    retriever = get_retriever()
    vec = embedder.embed_query(content)
    results = retriever.retrieve_with_rpc("match_chunks_768", vec, k=k * 3)
    from src.similarity import rerank_by_cosine
    return rerank_by_cosine(vec, results)[:k]

@st.cache_data(ttl=3600)
def load_chunk_source_map() -> dict:
    """Returns {snapshot_chunk_id: source_url} via vector_chunks_768 → sources join."""
    sb = get_supabase()
    # Step 1: get snapshot_chunk_id → source_id from vector_chunks_768 (549 rows, no pagination issue)
    vec_rows = (
        sb.table("vector_chunks_768")
        .select("snapshot_chunk_id, source_id")
        .execute()
        .data or []
    )
    # Step 2: get source_id → url from sources
    src_rows = (
        sb.table("sources")
        .select("id, url")
        .execute()
        .data or []
    )
    source_url_map = {r["id"]: r.get("url", "") for r in src_rows}
    # Step 3: join locally
    return {
        r["snapshot_chunk_id"]: source_url_map.get(r["source_id"], "")
        for r in vec_rows
    }

@st.cache_data(ttl=300)
def load_code_knowledge():
    sb = get_supabase()
    res = sb.table("code_knowledge").select("file_path, feature, content").execute()
    return res.data or []

@st.cache_data(ttl=300)
def load_triage_results() -> dict:
    """Returns {file_name: row} from triage_results for quick lookup."""
    sb = get_supabase()
    rows = (
        sb.table("triage_results")
        .select("file_name, top_bulletin_preview, triggering_cve, rationale, recommended_action")
        .execute()
        .data or []
    )
    return {r["file_name"]: r for r in rows}

BETA_IOS = "26.4 beta 3"
BETA_IOS_URL = "https://support.apple.com/en-us/126792"

@st.cache_data(ttl=300)
def load_stable_ios_version() -> str:
    """Returns the latest stable iOS version by scanning snapshot clean_text."""
    sb = get_supabase()
    src_ids = [s["id"] for s in
               sb.table("sources").select("id").eq("agent_name", "ios-risk-agent").execute().data]
    if not src_ids:
        return "—"
    snaps = (
        sb.table("snapshots")
        .select("clean_text")
        .in_("source_id", src_ids)
        .order("fetched_at", desc=True)
        .limit(50)
        .execute()
        .data or []
    )
    stable_versions = []
    for s in snaps:
        for v in re.findall(r'iOS\s+(\d+\.\d+(?:\.\d+)?)', s.get("clean_text", "")):
            major = int(v.split(".")[0])
            if 15 <= major < 20:
                stable_versions.append(v)
    if not stable_versions:
        return "—"
    return max(stable_versions, key=lambda v: [int(x) for x in v.split(".")])

# ── load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data from Supabase…"):
    code_data = load_code_knowledge()
    triage_map = load_triage_results()
    CURRENT_IOS = load_stable_ios_version()

BULLETIN_MIN_SIM = 0.55

@st.cache_data(ttl=3600)
def build_records(code_data_json: str) -> list:
    """Build risk records from code_knowledge. Cached to avoid re-running
    classify_risk_zs_with_scores + get_related_bulletins on every Streamlit rerun."""
    import json
    items = json.loads(code_data_json)
    records = []
    for item in items:
        content = item.get("content", "")
        lvl, rtype, reason, zs_scores = classify_risk_zs_with_scores(content)
        raw_buls = get_related_bulletins(content)
        has_impact = any(
            b.get("cosine_similarity", b.get("similarity", 0.0)) >= BULLETIN_MIN_SIM
            for b in raw_buls
        )
        effective_risk = lvl if has_impact else "No Impact"
        records.append({
            "Swift File":      shorten(item.get("file_path", "")),
            "file_path":       item.get("file_path", ""),
            "Feature":         item.get("feature", ""),
            "Risk Level":      lvl,
            "Effective Risk":  effective_risk,
            "Risk Type":       rtype,
            "Risk Reason":     reason,
            "Signal Category": signal_category(content),
            "Summary":         content,
            "zs_scores":       zs_scores,
        })
    return records

import json as _json
records = build_records(_json.dumps(code_data))

df = pd.DataFrame(records) if records else pd.DataFrame(
    columns=["Swift File","Feature","Risk Level","Effective Risk","Risk Type","Risk Reason","Signal Category","Summary"])

high_n      = int((df["Effective Risk"]=="High").sum())      if not df.empty else 0
medium_n    = int((df["Effective Risk"]=="Medium").sum())    if not df.empty else 0
low_n       = int((df["Effective Risk"]=="Low").sum())       if not df.empty else 0
no_impact_n = int((df["Effective Risk"]=="No Impact").sum()) if not df.empty else 0
total_n     = len(df)

# Weighted risk score 0–100 (No Impact counts as 0)
risk_score = round((high_n * 100 + medium_n * 50 + low_n * 10) / max(total_n, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="page-header">
    <div>
        <div class="page-title">📱 iOS Risk Sentinel</div>
        <div class="page-subtitle">AI-driven monitoring of Apple ecosystem changes and their impact on FingerprintJS Swift signal collection.</div>
    </div>
    <div class="live-badge"><div class="live-dot"></div>Active Monitoring</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
kpi_defs = [
    (c1, "kpi-blue",  "Stable iOS",       CURRENT_IOS,      "Latest monitored release",   None,                         "#2563eb"),
    (c2, "kpi-slate", "Beta Track",        BETA_IOS,         "Forward-looking changes",    None,                         "#64748b"),
    (c3, "kpi-red",   "High Impact",       str(high_n),      "Bulletin-confirmed risk",    high_n / max(total_n, 1),         "#dc2626"),
    (c4, "kpi-amber", "Medium Impact",     str(medium_n),    "Monitor for API drift",      medium_n / max(total_n, 1),   "#d97706"),
    (c5, "kpi-green", "Low Impact",        str(low_n),       "Unlikely to break near-term", low_n / max(total_n, 1),       "#16a34a"),
]
for col, accent, label, value, sub, pct, bar_color in kpi_defs:
    bar_html = ""
    if pct is not None:
        bar_html = f"""
        <div class="kpi-bar-bg">
            <div class="kpi-bar-fill" style="width:{int(pct*100)}%;background:{bar_color};"></div>
        </div>"""
    value_html = (
        f'<a href="{BETA_IOS_URL}" target="_blank" '
        f'style="color:inherit;text-decoration:none;border-bottom:1px dashed #94a3b8;">{value}</a>'
        if label == "Beta Track" else value
    )
    with col:
        st.markdown(f"""
        <div class="kpi {accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value_html}</div>
            <div class="kpi-sub">{sub}</div>
            {bar_html}
        </div>""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_beta_summary(version: str) -> tuple[list[str], str]:
    """Returns (bullets, url) from the beta snapshot."""
    sb = get_supabase()
    src_rows = (
        sb.table("sources").select("id, url")
        .eq("agent_name", "ios-risk-agent").execute().data or []
    )
    if not src_rows:
        return [], ""
    src_id_to_url = {r["id"]: r.get("url", "") for r in src_rows}
    major = version.split(".")[0]
    snaps = (
        sb.table("snapshots").select("clean_text, source_id")
        .in_("source_id", list(src_id_to_url.keys()))
        .order("fetched_at", desc=True).limit(50).execute().data or []
    )
    for s in snaps:
        text = s.get("clean_text", "")
        if f"iOS {major}" in text or version.split(" ")[0] in text:
            bullets, seen = [], set()
            for m in re.finditer(r'Impact:\s*(.+?)(?=\nDescription:|\nCVE-|\Z)', text, re.DOTALL):
                line = m.group(1).replace("\n", " ").strip()
                if line not in seen:
                    seen.add(line)
                    bullets.append(line)
                if len(bullets) >= 6:
                    break
            url = src_id_to_url.get(s.get("source_id"), "")
            return bullets, url
    return [], ""

if True:
    _beta_bullets, _beta_url = load_beta_summary(BETA_IOS)
    if _beta_bullets:
        _src_link = (
            f'<a href="{_beta_url}" target="_blank" style="font-size:.72rem;color:#64748b;'
            f'text-decoration:none;border-bottom:1px solid #cbd5e1;">View full release notes ↗</a>'
            if _beta_url else ""
        )
        _items_html = "".join(
            f'<li style="margin-bottom:.4rem;color:#1e3a5f;font-size:.8rem;line-height:1.55;">{b}</li>'
            for b in _beta_bullets
        )
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:3px solid #64748b;
                    border-radius:8px;padding:.75rem 1rem;margin-bottom:1rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;">
                <span style="font-size:.7rem;font-weight:700;color:#64748b;text-transform:uppercase;
                             letter-spacing:.07em;">iOS {BETA_IOS} — Key changes</span>
                {_src_link}
            </div>
            <ul style="margin:0;padding-left:1.1rem;">{_items_html}</ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"iOS {BETA_IOS} has not been scraped yet. Run the Scraper Agent once Apple publishes the release notes.")

PRIVACY_KEYWORDS = [
    "app may be able", "access sensitive", "user data", "privacy", "fingerprint",
    "identify", "track", "enumerate", "installed apps", "location", "microphone",
    "camera", "contacts", "siri", "accessibility", "data protection", "entitlement",
    "sandbox", "bypass", "permission", "consent",
]

def _extract_bullets(text: str) -> list[tuple[str, str, list[str]]]:
    """Returns list of (component, impact, cve_ids) tuples filtered by privacy keywords."""
    bullets = []
    seen = set()
    # Capture the full entry block: component, impact, description, then CVE lines
    for match in re.finditer(
        r'(?:^|\n)([A-Za-z][^\n]{0,60})\n+Impact:\s*(.+?)(?=\nAvailable for:|\n[A-Z][a-z]|\Z)',
        text, re.DOTALL
    ):
        component = match.group(1).strip()
        block = match.group(2)
        # Impact is everything up to Description:
        impact_m = re.match(r'(.+?)(?=\nDescription:|\nCVE-|\Z)', block, re.DOTALL)
        impact = impact_m.group(1).replace("\n", " ").strip() if impact_m else block.strip()
        # Extract CVE IDs from the block
        cves = re.findall(r'CVE-\d{4}-\d{4,7}', block)
        if len(component) > 60 or component[0].islower():
            component = ""
        if any(kw in impact.lower() for kw in PRIVACY_KEYWORDS) and impact not in seen:
            seen.add(impact)
            bullets.append((component, impact, cves))
    return bullets[:5]

@st.cache_data(ttl=300)
def load_ios_impact_bullets(version: str) -> tuple[list[str], str]:
    """Extract privacy/API-relevant Impact lines plus the source URL."""
    if not version or version == "—":
        return [], ""
    sb = get_supabase()
    src_rows = (
        sb.table("sources")
        .select("id, url")
        .eq("agent_name", "ios-risk-agent")
        .execute()
        .data or []
    )
    if not src_rows:
        return [], ""
    src_id_to_url = {r["id"]: r.get("url", "") for r in src_rows}
    src_ids = list(src_id_to_url.keys())
    snaps = (
        sb.table("snapshots")
        .select("clean_text, source_id")
        .in_("source_id", src_ids)
        .order("fetched_at", desc=True)
        .limit(50)
        .execute()
        .data or []
    )
    major = version.split(".")[0]  # e.g. "18" from "18.7.6"

    def _try(snaps_list, match_fn):
        for s in snaps_list:
            text = s.get("clean_text", "")
            if match_fn(text):
                bullets = _extract_bullets(text)
                if bullets:
                    url = src_id_to_url.get(s.get("source_id"), "")
                    return bullets, url
        return None, None

    bullets, url = _try(snaps, lambda t: f"security content of iOS {version}" in t)
    if bullets:
        return bullets, url

    bullets, url = _try(snaps, lambda t: f"security content of iOS {major}" in t and f"iOS {major} and" in t)
    if bullets:
        return bullets, url

    bullets, url = _try(snaps, lambda t: f"iOS {major}" in t)
    if bullets:
        return bullets, url

    return [], ""

import html as _html

def _linkify_cves_stable(raw: str, base_url: str = "") -> str:
    escaped = _html.escape(raw)
    def _make_link(m):
        cve = m.group(1)
        href = f"{base_url}#{cve.lower()}" if base_url else f"https://nvd.nist.gov/vuln/detail/{cve}"
        return (
            f'<a href="{href}" target="_blank"'
            f' style="color:#2563eb;font-weight:600;text-decoration:none;'
            f'border-bottom:1px solid #bfdbfe;">{cve}</a>'
        )
    return re.sub(r'(CVE-\d{4}-\d{4,7})', _make_link, escaped)

ios_bullets, ios_source_url = load_ios_impact_bullets(CURRENT_IOS)
if ios_bullets:
    def _bullet_row(component: str, impact: str, cves: list) -> str:
        badge = (
            f'<span style="display:inline-block;background:#dbeafe;color:#1d4ed8;'
            f'font-size:.65rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;'
            f'padding:.15rem .45rem;border-radius:4px;margin-right:.5rem;'
            f'white-space:nowrap;">{_html.escape(component)}</span>'
            if component else ""
        )
        cve_links = " ".join(
            f'<a href="{ios_source_url}#{c.lower()}" target="_blank"'
            f' style="font-size:.7rem;color:#2563eb;font-weight:600;text-decoration:none;'
            f'border-bottom:1px solid #bfdbfe;white-space:nowrap;">{c}</a>'
            for c in cves
        )
        return (
            f'<li style="margin-bottom:.6rem;display:flex;align-items:flex-start;gap:.3rem;">'
            f'<span style="color:#2563eb;margin-top:.1rem;flex-shrink:0;">›</span>'
            f'<span style="color:#1e3a5f;font-size:.82rem;line-height:1.6;">'
            f'{badge}{_html.escape(impact)}'
            f'{"&nbsp;&nbsp;" + cve_links if cve_links else ""}'
            f'</span>'
            f'</li>'
        )
    bullets_html = "".join(_bullet_row(c, i, v) for c, i, v in ios_bullets)
    source_link = (
        f'<a href="{ios_source_url}" target="_blank" '
        f'style="color:#2563eb;font-size:.72rem;font-weight:500;text-decoration:none;'
        f'border-bottom:1px solid #bfdbfe;">View Apple Security Release ↗</a>'
        if ios_source_url else ""
    )
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#eff6ff 0%,#f0f9ff 100%);
                border:1px solid #bfdbfe;border-radius:12px;
                padding:1rem 1.25rem 1rem 1.25rem;margin-bottom:1.25rem;
                box-shadow:0 1px 4px rgba(37,99,235,.07);">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.6rem;">
            <div style="display:flex;align-items:center;gap:.5rem;">
                <span style="background:#2563eb;color:#fff;font-size:.65rem;font-weight:700;
                             letter-spacing:.08em;text-transform:uppercase;
                             padding:.2rem .55rem;border-radius:999px;">iOS {CURRENT_IOS}</span>
                <span style="font-size:.78rem;font-weight:600;color:#1e40af;">
                    Privacy &amp; API changes affecting signal collection
                </span>
            </div>
            {source_link}
        </div>
        <ul style="font-size:.82rem;line-height:1.7;margin:.25rem 0 0 0;
                   padding-left:0;list-style:none;">{bullets_html}</ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["Executive Overview", "AI Risk Analysis", "File Risk Table"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  ·  EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if "t1_filter" not in st.session_state:
        st.session_state.t1_filter = {"type": None, "risk": None, "category": None}

    left, right = st.columns([3, 2], gap="large")

    with left:
        _donut_col, _ = st.columns([1, 1])

        # ── Donut chart ───────────────────────────────────────────────────────
        with _donut_col:
            st.markdown('<div class="sec">Risk Distribution</div>', unsafe_allow_html=True)
            donut = go.Figure(go.Pie(
                labels=["High", "Medium", "Low"],
                values=[high_n, medium_n, low_n],
                hole=0.62,
                marker=dict(colors=["#dc2626","#d97706","#16a34a"],
                            line=dict(color="#ffffff", width=2)),
                textinfo="percent",
                textfont=dict(size=11, color="#ffffff", family="Inter"),
                hovertemplate="<b>%{label}</b><br>%{value} files<extra></extra>",
            ))
            donut.add_annotation(
                text=f"<b>{total_n}</b><br><span style='font-size:10px;color:#9ca3af'>files</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color="#111827", family="Inter"),
                align="center",
            )
            donut.update_layout(
                **PLOTLY_BASE,
                showlegend=True,
                legend=dict(
                    orientation="h", x=0.5, xanchor="center", y=-0.08,
                    font=dict(size=11, color="#374151"),
                ),
                height=240,
                clickmode="event+select",
            )
            donut_ev = st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False}, on_select="rerun", key="donut_chart")
            if donut_ev and donut_ev.selection.points:
                lbl = donut_ev.selection.points[0].get("label")
                if lbl and st.session_state.t1_filter.get("risk") != lbl:
                    st.session_state.t1_filter = {"type": "risk", "risk": lbl, "category": None}
                    st.rerun()

        # ── Stacked bar: Risk breakdown per signal category (linked to donut) ──
        _ftype = st.session_state.t1_filter.get("type")
        _active_risk = (
            st.session_state.t1_filter.get("risk")
            if _ftype in ("risk", "both") else None
        )
        _active_cat = (
            st.session_state.t1_filter.get("category")
            if _ftype == "both" else None
        )
        _risk_color = {"High": "#dc2626", "Medium": "#d97706", "Low": "#16a34a"}

        if _active_risk and _active_cat:
            _stack_title = (
                f'Signal Category — <span style="color:{_risk_color.get(_active_risk,"#374151")}">'
                f'{_active_cat} · {_active_risk}</span>'
            )
        elif _active_risk:
            _stack_title = (
                f'Signal Categories — <span style="color:{_risk_color.get(_active_risk,"#374151")}">'
                f'{_active_risk} files only</span>'
            )
        else:
            _stack_title = "Risk Breakdown by Signal Category"
        st.markdown(f'<div class="sec" style="margin-top:.5rem;">{_stack_title}</div>', unsafe_allow_html=True)

        if not df.empty:
            fig_stack = go.Figure()
            if _active_risk:
                # Filtered view: single bar per category for the selected risk level, highlight active cat
                _filtered_df = df[df["Effective Risk"] == _active_risk]
                _counts = _filtered_df["Signal Category"].value_counts().sort_values(ascending=True)
                _cat_order = _counts.index.tolist()
                _bar_colors = [
                    _risk_color[_active_risk] if (not _active_cat or cat == _active_cat)
                    else "#e5e7eb"
                    for cat in _cat_order
                ]
                fig_stack.add_trace(go.Bar(
                    name=_active_risk,
                    x=_counts.values.tolist(),
                    y=_cat_order,
                    orientation="h",
                    marker=dict(color=_bar_colors, line=dict(width=0)),
                    hovertemplate="<b>%{y}</b><br>" + _active_risk + ": %{x} file(s)<extra></extra>",
                    text=_counts.values.tolist(),
                    textposition="outside",
                    textfont=dict(size=11, color="#374151"),
                ))
                fig_stack.update_layout(
                    **PLOTLY_BASE,
                    barmode="stack",
                    xaxis=dict(visible=False),
                    yaxis=dict(tickfont=dict(size=11, color="#374151")),
                    height=max(240, len(_cat_order) * 38),
                    bargap=0.35,
                    showlegend=False,
                )
            else:
                # Default view: stacked bar for all levels
                _cat_order = df["Signal Category"].value_counts().index.tolist()
                _full_palette = {"High": "#dc2626", "Medium": "#d97706", "Low": "#16a34a", "No Impact": "#d1d5db"}
                for level in ["High", "Medium", "Low", "No Impact"]:
                    _counts = df[df["Effective Risk"] == level]["Signal Category"].value_counts()
                    fig_stack.add_trace(go.Bar(
                        name=level,
                        x=[_counts.get(cat, 0) for cat in _cat_order],
                        y=_cat_order,
                        orientation="h",
                        marker=dict(color=_full_palette[level], line=dict(width=0)),
                        hovertemplate=f"<b>%{{y}}</b><br>{level}: %{{x}} file(s)<extra></extra>",
                    ))
                fig_stack.update_layout(
                    **PLOTLY_BASE,
                    barmode="stack",
                    xaxis=dict(visible=False),
                    yaxis=dict(tickfont=dict(size=11, color="#374151")),
                    height=320,
                    bargap=0.3,
                    legend=dict(
                        orientation="h", x=0.5, xanchor="center", y=-0.04,
                        font=dict(size=11, color="#374151"),
                    ),
                    showlegend=True,
                )
            stack_ev = st.plotly_chart(fig_stack, use_container_width=True, config={"displayModeBar": False}, on_select="rerun", key="stack_chart")
            if stack_ev and stack_ev.selection.points:
                pt = stack_ev.selection.points[0]
                cat_val = pt.get("y")
                curve = pt.get("curveNumber", 0)
                risk_val = _active_risk if _active_risk else ["High", "Medium", "Low", "No Impact"][min(curve, 3)]
                if cat_val and (
                    st.session_state.t1_filter.get("category") != cat_val
                    or st.session_state.t1_filter.get("risk") != risk_val
                ):
                    st.session_state.t1_filter = {"type": "both", "risk": risk_val, "category": cat_val}
                    st.rerun()

        # ── File list: shown when a category is selected via stacked bar ──────
        if _active_risk and _active_cat:
            _risk_color2 = {"High": "#dc2626", "Medium": "#d97706", "Low": "#16a34a"}
            _color = _risk_color2.get(_active_risk, "#6b7280")
            _matched = df[
                (df["Effective Risk"] == _active_risk) &
                (df["Signal Category"] == _active_cat)
            ][["Swift File", "Feature", "Effective Risk", "Risk Type"]].reset_index(drop=True)

            st.markdown(
                f'<div class="sec" style="margin-top:.75rem;">'
                f'Files — <span style="color:{_color}">{_active_cat} · {_active_risk}</span>'
                f'&nbsp;<span style="font-size:.72rem;color:#6b7280;font-weight:400;">'
                f'({len(_matched)} file{"s" if len(_matched) != 1 else ""})</span>'
                f'&nbsp;&nbsp;<span style="font-size:.72rem;color:#9ca3af;cursor:pointer;" '
                f'onclick="void(0)">— click a row in the File Risk Table to inspect</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            for _, row in _matched.iterrows():
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:.6rem;padding:.45rem .6rem;'
                    f'border:1px solid #e5e7eb;border-radius:6px;margin-bottom:.3rem;background:#fff;">'
                    f'<span style="background:{_color}22;color:{_color};font-size:.65rem;font-weight:700;'
                    f'padding:.15rem .4rem;border-radius:4px;white-space:nowrap;">{_active_risk}</span>'
                    f'<span style="font-size:.8rem;color:#111827;font-weight:500;font-family:monospace;">{row["Swift File"]}</span>'
                    f'<span style="font-size:.75rem;color:#6b7280;margin-left:auto;">{row["Feature"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            if st.button("✕ Clear filter", key="clear_filter_btn"):
                st.session_state.t1_filter = {"type": None, "risk": None, "category": None}
                st.rerun()

    # ── Right column ──────────────────────────────────────────────────────────
    with right:

        st.markdown('<div class="sec">Release Context</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ctx-panel">
            <div class="ctx-row"><span class="ctx-key">Stable release</span><span class="ctx-val">iOS {CURRENT_IOS}</span></div>
            <div class="ctx-row"><span class="ctx-key">Beta release</span><span class="ctx-val">iOS {BETA_IOS}</span></div>
            <div class="ctx-row"><span class="ctx-key">Intelligence source</span><span class="ctx-val">Apple bulletins + Swift RAG</span></div>
            <div class="ctx-row" style="margin-bottom:0"><span class="ctx-key">Indexed components</span><span class="ctx-val">{len(df)} files</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec">Risk Level Guide</div>', unsafe_allow_html=True)
        for lvl, title_color, bg, border, desc in [
            ("High",   "#b91c1c", "#fef2f2", "#fecaca", "File depends on signals Apple is actively restricting or making unreliable."),
            ("Medium", "#92400e", "#fffbeb", "#fde68a", "File uses APIs permitted today but subject to behavioral changes."),
            ("Low",    "#15803d", "#f0fdf4", "#bbf7d0", "File uses low-sensitivity signals; unlikely to break near-term."),
        ]:
            st.markdown(f"""
            <div class="rex" style="background:{bg};border-color:{border};">
                <div class="rex-title" style="color:{title_color};">{lvl} Risk</div>
                <div class="rex-body">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec" style="margin-top:.9rem;">Priority Files</div>', unsafe_allow_html=True)
        for row in [r for r in records if r["Effective Risk"]=="High"][:3]:
            st.markdown(f"""
            <div class="{fc_cls(row['Risk Level'])}">
                {rb_html(row['Risk Level'])}
                <div class="fc-name" style="margin-top:.35rem;">{row['Swift File']}</div>
                <div class="fc-meta">{row['Feature']} · {row['Risk Type']}</div>
                <div class="fc-body">{row['Summary'][:140]}…</div>
            </div>""", unsafe_allow_html=True)

    # ── Drill-down panel (shown when a chart element is clicked) ──────────────
    f = st.session_state.t1_filter
    if f["type"]:
        _risk_bg   = {"High": "#fef2f2", "Medium": "#fffbeb", "Low": "#f0fdf4", "No Impact": "#f9fafb"}.get(f.get("risk",""), "#f9fafb")
        _risk_border = {"High": "#fecaca", "Medium": "#fde68a", "Low": "#bbf7d0", "No Impact": "#e5e7eb"}.get(f.get("risk",""), "#e5e7eb")
        _risk_col  = {"High": "#b91c1c",  "Medium": "#92400e", "Low": "#15803d", "No Impact": "#6b7280"}.get(f.get("risk",""), "#374151")

        if f["type"] == "risk":
            filtered = [r for r in records if r["Effective Risk"] == f["risk"]]
            title = f"{f['risk']} Risk — All Files"
            badge_color = _risk_bg
        elif f["type"] == "category":
            filtered = [r for r in records if r["Signal Category"] == f["category"]]
            title = f"{f['category']} — All Files"
            badge_color = "#eff6ff"
        elif f["type"] == "both":
            filtered = [r for r in records if r["Signal Category"] == f["category"] and r["Effective Risk"] == f["risk"]]
            title = f"{f['category']}  ·  {f['risk']} Risk"
            badge_color = _risk_bg
        elif f["type"] == "file":
            filtered = [r for r in records if r["Swift File"] == f.get("file")]
            title = f["file"]
            badge_color = "#f9fafb"
        else:
            filtered = []
            title = ""
            badge_color = "#f9fafb"

        st.markdown(
            f"<div id='drill-anchor' style='border:1.5px solid {_risk_border};"
            f"border-radius:10px;padding:.6rem 1rem;background:{_risk_bg};"
            f"margin:1.5rem 0 .75rem 0;display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-size:.82rem;font-weight:700;color:{_risk_col};'>{title}"
            f"  <span style='font-size:.78rem;font-weight:400;color:#6b7280;'>— {len(filtered)} file(s)</span></span>"
            f"</div>",
            unsafe_allow_html=True
        )
        hd_col, btn_col = st.columns([5, 1])
        with btn_col:
            if st.button("✕ Clear", key="t1_clear"):
                st.session_state.t1_filter = {"type": None, "risk": None, "category": None}
                st.rerun()

        if filtered:
            dc1, dc2 = st.columns(2)
            for i, row in enumerate(filtered):
                with (dc1 if i % 2 == 0 else dc2):
                    st.markdown(f"""
                    <div class="{fc_cls(row['Risk Level'])}" style="background:{badge_color};">
                        {rb_html(row['Risk Level'])}
                        <div class="fc-name" style="margin-top:.35rem;">{row['Swift File']}</div>
                        <div class="fc-meta">{row['Feature']} · {row['Signal Category']}</div>
                        <div class="fc-body">{row['Summary'][:180]}…</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.caption("No files match this selection.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  ·  AI RISK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: .75rem 1rem;
    font-size: .87rem;
    color: #1e3a8a;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-top: 2px solid #2563eb;
    border-radius: 10px;
    padding: .85rem 1rem;
    font-size: .87rem;
    color: #1f2937;
    line-height: 1.7;
}
[data-testid="stChatInput"] textarea {
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-size: .87rem !important;
    background: #ffffff !important;
    color: #111827 !important;
}
[data-testid="stChatInput"] button {
    background: #374151 !important;
    border-radius: 6px !important;
}
details.ctx-exp summary {
    font-size: .72rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: .07em;
    cursor: pointer;
    margin-top: .9rem;
    list-style: none;
}
details.ctx-exp summary::before { content: '▸  '; }
details.ctx-exp[open] summary::before { content: '▾  '; }
</style>
""", unsafe_allow_html=True)

with tab2:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = []

    st.markdown("""
    <div style="font-size:.83rem;color:#4b5563;margin-bottom:1rem;line-height:1.65;">
        Ask anything about iOS changes, at-risk signals, or affected Swift files.
        The model retrieves context from the indexed codebase on every turn.
    </div>""", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📱"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📎  {len(msg['sources'])} supporting files retrieved", expanded=False):
                    src_c1, src_c2, src_c3 = st.columns(3)
                    for j, src in enumerate(msg["sources"]):
                        with [src_c1, src_c2, src_c3][j % 3]:
                            st.markdown(f"""
                            <div class="{fc_cls(src['Risk Level'])}">
                                {rb_html(src['Risk Level'])}
                                <div class="fc-name" style="margin-top:.3rem;">{src['Swift File']}</div>
                                <div class="fc-meta">{src['Feature']} · {src['Signal Category']}</div>
                                <div class="fc-body">{src['Summary'][:160]}…</div>
                            </div>""", unsafe_allow_html=True)

    user_input = st.chat_input("Ask about iOS risk, signal changes, or Swift files…")
    query = user_input

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query, "sources": []})
        st.session_state.chat_context.append({"role": "user", "content": query})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="📱"):
            with st.spinner("Retrieving context…"):
                embedder = get_embedder()
                retriever = get_retriever()
                llm = get_llm()

                query_vec = embedder.embed_query(query)
                raw_results = retriever.retrieve_code_knowledge_with_rerank(
                    query_vec, k=6, fetch_k=20
                )

                # Map retrieved results to the same shape as `records` for display
                sources = []
                for r in raw_results:
                    lvl, rtype, reason, zs_scores = classify_risk_zs_with_scores(r.get("content", ""))
                    sources.append({
                        "Swift File":        shorten(r.get("file_path", "")),
                        "file_path":         r.get("file_path", ""),
                        "Feature":           r.get("feature", ""),
                        "Risk Level":        lvl,
                        "Risk Type":         rtype,
                        "Risk Reason":       reason,
                        "Signal Category":   signal_category(r.get("content", "")),
                        "Summary":           r.get("content", ""),
                        "cosine_similarity": r.get("cosine_similarity", r.get("similarity", 0.0)),
                        "zs_scores":         zs_scores,
                    })

                system_instructions = (
                    "You are an iOS risk analysis assistant. "
                    "Answer concisely using the retrieved Swift file context. "
                    "Highlight the most at-risk files and explain why."
                )
                prompt = build_prompt(
                    query,
                    [{
                        "snapshot_chunk_id": s["Swift File"],
                        "similarity": s["cosine_similarity"],
                        "chunk_text": s["Summary"],
                    } for s in sources],
                    system_instructions=system_instructions,
                )
                response_text = llm.generate(
                    prompt, model="meta/llama-3.1-70b-instruct"
                ).text

                st.markdown(response_text)

                with st.expander(f"📎  {len(sources)} supporting files retrieved", expanded=False):
                    src_c1, src_c2, src_c3 = st.columns(3)
                    for j, src in enumerate(sources):
                        with [src_c1, src_c2, src_c3][j % 3]:
                            st.markdown(f"""
                            <div class="{fc_cls(src['Risk Level'])}">
                                {rb_html(src['Risk Level'])}
                                <div class="fc-name" style="margin-top:.3rem;">{src['Swift File']}</div>
                                <div class="fc-meta">{src['Feature']} · {src['Signal Category']}</div>
                                <div class="fc-body" style="margin-top:.3rem;">{src['Summary'][:160]}…</div>
                                <div class="fc-meta" style="margin-top:.3rem;color:#6b7280;">
                                    similarity: {src['cosine_similarity']:.3f}
                                </div>
                            </div>""", unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_text,
            "sources": sources,
        })
        st.session_state.chat_context.append({"role": "assistant", "content": response_text})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  ·  FILE RISK TABLE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if df.empty:
        st.info("No indexed files found. Run the ingestion script to sync your Swift codebase.")
    else:
        f1, f2 = st.columns([1, 1])
        with f1:
            sel_risk = st.selectbox("Filter by risk level", ["All","High","Medium","Low"], key="t3r")
        with f2:
            sel_feat = st.selectbox("Filter by feature",
                ["All"] + sorted(df["Feature"].dropna().unique().tolist()), key="t3f")

        view = df.copy()
        if sel_risk != "All":
            view = view[view["Effective Risk"] == sel_risk]
        if sel_feat != "All":
            view = view[view["Feature"] == sel_feat]

        disp = view[["Swift File","Feature","Signal Category","Effective Risk","Risk Type","Risk Reason"]].copy()
        st.dataframe(disp, use_container_width=True, height=280)

        st.markdown('<div class="sec" style="margin-top:1.1rem;">File Cards</div>', unsafe_allow_html=True)
        for row in view.to_dict("records"):
            eff = row["Effective Risk"]
            label = f"{row['Swift File']}  ·  {eff}  ·  {row['Feature']}"
            with st.expander(label, expanded=False):
                col_l, col_r = st.columns([3, 2])

                with col_l:
                    st.markdown(
                        f'<div style="margin-bottom:.5rem;">'
                        f'<span style="font-size:.65rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Feature</span><br>'
                        f'<span style="color:#1e3a5f;font-size:.9rem;">{_html.escape(row.get("Feature",""))}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div style="margin-bottom:.5rem;">'
                        f'<span style="font-size:.65rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Signal Category</span><br>'
                        f'<span style="color:#1e3a5f;font-size:.9rem;">{_html.escape(row.get("Signal Category",""))}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div style="margin-bottom:.5rem;">'
                        f'<span style="font-size:.65rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Summary</span><br>'
                        f'<span style="color:#1e3a5f;font-size:.85rem;">{_html.escape(row.get("Summary",""))}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with col_r:
                    # ── Load bulletins first so effective risk can be computed
                    bulletins = get_related_bulletins(row["Summary"])
                    source_map = load_chunk_source_map()
                    BULLETIN_MIN_SIM = 0.55
                    bulletins = [
                        b for b in bulletins
                        if b.get("cosine_similarity", b.get("similarity", 0.0)) >= BULLETIN_MIN_SIM
                    ]

                    # ── Effective risk banner (uses pre-computed value from record)
                    eff_level = row["Effective Risk"]
                    if bulletins:
                        top_sim = max(
                            b.get("cosine_similarity", b.get("similarity", 0.0))
                            for b in bulletins
                        )
                        eff_sub = f"Backed by {len(bulletins)} matching Apple bulletin(s). Top similarity: {top_sim:.3f}."
                    else:
                        eff_sub = "No matching Apple bulletins found. Not currently impacted by any indexed Apple changes."
                    if eff_level == "High":
                        eff_bg, eff_border, eff_color = "#fef2f2", "#fecaca", "#b91c1c"
                    elif eff_level == "Medium":
                        eff_bg, eff_border, eff_color = "#fffbeb", "#fde68a", "#92400e"
                    else:
                        eff_bg, eff_border, eff_color = "#f0fdf4", "#bbf7d0", "#15803d"

                    st.markdown(f"""
                    <div style="background:{eff_bg};border:1px solid {eff_border};
                                border-radius:8px;padding:.6rem .85rem;margin-bottom:.85rem;">
                        <div style="font-size:.68rem;font-weight:600;color:#6b7280;
                                    text-transform:uppercase;letter-spacing:.07em;margin-bottom:.25rem;">
                            Effective Risk
                        </div>
                        <div style="font-size:1rem;font-weight:700;color:{eff_color};">
                            {eff_level}
                        </div>
                        <div style="font-size:.74rem;color:#6b7280;margin-top:.2rem;line-height:1.45;">
                            {eff_sub}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # ── Rationale + Recommended Action (LLM-generated, High/Medium only) ──
                    _rationale = row.get("rationale", "")
                    _action    = row.get("recommended_action", "")
                    _cve       = row.get("triggering_cve", "")
                    if _rationale and eff_level in ("High", "Medium"):
                        _action_color = "#b91c1c" if eff_level == "High" else "#92400e"
                        _action_bg    = "#fef2f2" if eff_level == "High" else "#fffbeb"
                        _action_border= "#fecaca" if eff_level == "High" else "#fde68a"
                        _cve_html = (
                            f'<span style="font-size:.68rem;font-weight:700;background:#dbeafe;'
                            f'color:#1d4ed8;border-radius:4px;padding:.1rem .4rem;margin-left:.5rem;">'
                            f'{_cve}</span>'
                            if _cve else ""
                        )
                        st.markdown(f"""
                        <div style="background:#f8fafc;border:1px solid #e2e8f0;
                                    border-radius:8px;padding:.65rem .85rem;margin-bottom:.85rem;">
                            <div style="font-size:.68rem;font-weight:600;color:#6b7280;
                                        text-transform:uppercase;letter-spacing:.07em;margin-bottom:.35rem;">
                                Why this is at risk {_cve_html}
                            </div>
                            <div style="font-size:.8rem;color:#1e3a5f;line-height:1.6;">
                                {_html.escape(_rationale)}
                            </div>
                            <div style="margin-top:.5rem;background:{_action_bg};border:1px solid {_action_border};
                                        border-radius:5px;padding:.3rem .6rem;display:inline-block;">
                                <span style="font-size:.72rem;font-weight:600;color:{_action_color};">
                                    ⚡ {_html.escape(_action)}
                                </span>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    # ── Classification confidence (intrinsic signal sensitivity)
                    st.markdown("**Signal Sensitivity** *(zero-shot)*")
                    level_colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
                    scores = row.get("zs_scores", {})
                    for lbl, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                        color = level_colors.get(lbl, "#6b7280")
                        st.markdown(f"""
                        <div style="margin-bottom:.5rem;">
                            <div style="display:flex;justify-content:space-between;font-size:.82rem;">
                                <span style="color:{color};font-weight:600;">{lbl}</span>
                                <span style="color:#374151;">{score:.1%}</span>
                            </div>
                            <div style="background:#f3f4f6;border-radius:4px;height:6px;margin-top:.25rem;">
                                <div style="background:{color};width:{score*100:.1f}%;height:100%;border-radius:4px;"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    # ── Related Apple bulletins
                    st.markdown("**Related Apple Bulletins**")
                    if bulletins:
                        import html as _html

                        def _linkify_cves(raw: str, base_url: str = "") -> str:
                            escaped = _html.escape(raw)
                            def _make_link(m):
                                cve = m.group(1)
                                # Deep-link to the CVE paragraph on the Apple bulletin page
                                href = f"{base_url}#{cve.lower()}" if base_url else f"https://nvd.nist.gov/vuln/detail/{cve}"
                                return (
                                    f'<a href="{href}" target="_blank"'
                                    f' style="color:#2563eb;font-weight:600;text-decoration:none;'
                                    f'border-bottom:1px solid #bfdbfe;">{cve}</a>'
                                )
                            return re.sub(r'(CVE-\d{4}-\d{4,7})', _make_link, escaped)

                        for b in bulletins:
                            sim = b.get("cosine_similarity", b.get("similarity", 0.0))
                            chunk_id = b.get("snapshot_chunk_id")
                            url = source_map.get(chunk_id, "")
                            raw_text = b.get("chunk_text") or b.get("content") or ""
                            full_text = _linkify_cves(raw_text, base_url=url)
                            link_html = (
                                f'<a href="{url}" target="_blank" style="font-size:.75rem;'
                                f'color:#2563eb;text-decoration:none;font-weight:500;">↗ View source</a>'
                                if url else ""
                            )
                            # Quality badge based on similarity threshold
                            if sim >= 0.75:
                                badge_html = (
                                    '<span style="font-size:.68rem;font-weight:600;'
                                    'background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;'
                                    'padding:.1rem .45rem;border-radius:999px;">Strong</span>'
                                )
                            elif sim >= 0.55:
                                badge_html = (
                                    '<span style="font-size:.68rem;font-weight:600;'
                                    'background:#fffbeb;color:#92400e;border:1px solid #fde68a;'
                                    'padding:.1rem .45rem;border-radius:999px;">Plausible</span>'
                                )
                            else:
                                badge_html = (
                                    '<span style="font-size:.68rem;font-weight:600;'
                                    'background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;'
                                    'padding:.1rem .45rem;border-radius:999px;">Weak</span>'
                                )
                            full_section = (
                                f'<div style="max-height:180px;overflow-y:auto;margin-top:.4rem;'
                                f'background:#f9fafb;border-radius:4px;padding:.5rem .6rem;'
                                f'font-size:.78rem;color:#374151;line-height:1.55;'
                                f'border:1px solid #f3f4f6;">{full_text}</div>'
                                if len(full_text) > 200 else
                                f'<div style="font-size:.8rem;color:#374151;line-height:1.55;">{full_text}</div>'
                            )
                            st.markdown(f"""
                            <div style="border:1px solid #e5e7eb;border-radius:6px;
                                        padding:.65rem .8rem;margin-bottom:.5rem;">
                                <div style="display:flex;justify-content:space-between;
                                            align-items:center;margin-bottom:.3rem;">
                                    <div style="display:flex;align-items:center;gap:.45rem;">
                                        <span style="font-size:.74rem;color:#6b7280;">
                                            similarity: {sim:.3f}
                                        </span>
                                        {badge_html}
                                    </div>
                                    {link_html}
                                </div>
                                {full_section}
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.caption("None — see Effective Risk above.")
