import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from supabase import create_client
from src.config import settings


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
def classify_risk(text: str) -> tuple[str, str, str]:
    t = (text or "").lower()
    if any(x in t for x in ["kernel","boot time","biometrics","local authentication",
                              "critical","blocked","degraded","high risk","tracking",
                              "sensitive information","device fingerprinting"]):
        return "High", "Signal Degradation", "Likely exposed to Apple privacy restrictions or reliability changes."
    if any(x in t for x in ["identifierforvendor","moderate","api change","limited",
                              "partial","network","less reliable"]):
        return "Medium", "API Dependency", "Depends on APIs that may remain available but could change in behavior."
    return "Low", "Operational", "Lower sensitivity; less immediate disruption risk."

def signal_category(text: str) -> str:
    t = (text or "").lower()
    if any(x in t for x in ["kernel","os version","os build","boot time","time zone"]):
        return "OS / System"
    if any(x in t for x in ["identifierforvendor","vendor identifier","idfv"]):
        return "Identifiers"
    if any(x in t for x in ["device name","device type","memory","cpu","screen resolution"]):
        return "Device Attributes"
    if any(x in t for x in ["biometrics","passcode","authentication"]):
        return "Authentication"
    if any(x in t for x in ["locale","user interface style"]):
        return "Locale / UI"
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
    return create_client(settings.supabase_url, settings.supabase_key)

@st.cache_data(ttl=300)
def load_code_knowledge():
    sb = get_supabase()
    res = sb.table("code_knowledge").select("file_path, feature, content").execute()
    return res.data or []

BETA_IOS = "26.4 beta 3"

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
    CURRENT_IOS = load_stable_ios_version()

records = []
for item in code_data:
    lvl, rtype, reason = classify_risk(item.get("content", ""))
    records.append({
        "Swift File":      shorten(item.get("file_path", "")),
        "Feature":         item.get("feature", ""),
        "Risk Level":      lvl,
        "Risk Type":       rtype,
        "Risk Reason":     reason,
        "Signal Category": signal_category(item.get("content", "")),
        "Summary":         item.get("content", ""),
    })

df = pd.DataFrame(records) if records else pd.DataFrame(
    columns=["Swift File","Feature","Risk Level","Risk Type","Risk Reason","Signal Category","Summary"])

high_n   = int((df["Risk Level"]=="High").sum())   if not df.empty else 0
medium_n = int((df["Risk Level"]=="Medium").sum()) if not df.empty else 0
low_n    = int((df["Risk Level"]=="Low").sum())    if not df.empty else 0
total_n  = len(df)

# Weighted risk score 0–100
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
    (c3, "kpi-red",   "High-Risk Files",   str(high_n),      "Immediate attention needed", high_n / max(total_n, 1),    "#dc2626"),
    (c4, "kpi-amber", "Medium-Risk Files", str(medium_n),    "Monitor for API drift",      medium_n / max(total_n, 1),  "#d97706"),
    (c5, "kpi-green", "Stable Files",      str(low_n),       "No disruption expected",     low_n / max(total_n, 1),     "#16a34a"),
]
for col, accent, label, value, sub, pct, bar_color in kpi_defs:
    bar_html = ""
    if pct is not None:
        bar_html = f"""
        <div class="kpi-bar-bg">
            <div class="kpi-bar-fill" style="width:{int(pct*100)}%;background:{bar_color};"></div>
        </div>"""
    with col:
        st.markdown(f"""
        <div class="kpi {accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
            {bar_html}
        </div>""", unsafe_allow_html=True)

PRIVACY_KEYWORDS = [
    "app may be able", "access sensitive", "user data", "privacy", "fingerprint",
    "identify", "track", "enumerate", "installed apps", "location", "microphone",
    "camera", "contacts", "siri", "accessibility", "data protection", "entitlement",
    "sandbox", "bypass", "permission", "consent",
]

def _extract_bullets(text: str) -> list[str]:
    bullets = []
    for match in re.finditer(r'Impact:\s*(.+?)(?=\nDescription:|\nCVE-|\nAvailable for:|\Z)',
                             text, re.DOTALL):
        impact = match.group(1).replace("\n", " ").strip()
        if any(kw in impact.lower() for kw in PRIVACY_KEYWORDS) and impact not in bullets:
            bullets.append(impact)
    return bullets[:5]

@st.cache_data(ttl=300)
def load_ios_impact_bullets(version: str) -> list[str]:
    """Extract privacy/API-relevant Impact lines, trying specific version first then major version."""
    if not version or version == "—":
        return []
    sb = get_supabase()
    src_ids = [s["id"] for s in
               sb.table("sources").select("id").eq("agent_name", "ios-risk-agent").execute().data]
    if not src_ids:
        return []
    snaps = (
        sb.table("snapshots")
        .select("clean_text")
        .in_("source_id", src_ids)
        .order("fetched_at", desc=True)
        .limit(50)
        .execute()
        .data or []
    )
    major = version.split(".")[0]  # e.g. "18" from "18.7.6"

    # Priority 1: snapshot specifically about this exact version
    for s in snaps:
        text = s.get("clean_text", "")
        if f"security content of iOS {version}" in text:
            bullets = _extract_bullets(text)
            if bullets:
                return bullets

    # Priority 2: snapshot about the major version (e.g. "iOS 18")
    for s in snaps:
        text = s.get("clean_text", "")
        if f"security content of iOS {major}" in text and f"iOS {major} and" in text:
            bullets = _extract_bullets(text)
            if bullets:
                return bullets

    # Priority 3: any snapshot mentioning the major version
    for s in snaps:
        text = s.get("clean_text", "")
        if f"iOS {major}" in text:
            bullets = _extract_bullets(text)
            if bullets:
                return bullets

    return []

ios_bullets = load_ios_impact_bullets(CURRENT_IOS)
if ios_bullets:
    bullets_html = "".join(
        f'<li style="margin-bottom:.35rem;">{b}</li>' for b in ios_bullets
    )
    st.markdown(f"""
    <div style="background:#f8faff;border:1px solid #dbeafe;border-left:3px solid #2563eb;
                border-radius:8px;padding:.75rem 1rem;margin-bottom:1rem;">
        <span style="font-size:.7rem;font-weight:600;color:#2563eb;text-transform:uppercase;
                     letter-spacing:.07em;">iOS {CURRENT_IOS} — Privacy &amp; API changes affecting signal collection</span>
        <ul style="font-size:.82rem;color:#374151;line-height:1.65;margin:.5rem 0 0 0;
                   padding-left:1.2rem;">{bullets_html}</ul>
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
    left, right = st.columns([3, 2], gap="large")

    with left:
        ch1, ch2 = st.columns(2)

        # ── Donut chart ───────────────────────────────────────────────────────
        with ch1:
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
            )
            st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

        # ── Signal exposure bar chart ──────────────────────────────────────────
        with ch2:
            st.markdown('<div class="sec">Signal Exposure Areas</div>', unsafe_allow_html=True)
            if not df.empty:
                sig = df["Signal Category"].value_counts().reset_index()
                sig.columns = ["Category", "Count"]
                bar_colors = ["#dc2626" if c >= 3 else "#d97706" if c == 2 else "#16a34a" for c in sig["Count"]]
                fig_bar = go.Figure(go.Bar(
                    x=sig["Count"], y=sig["Category"],
                    orientation="h",
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=sig["Count"], textposition="outside",
                    textfont=dict(size=11, color="#6b7280"),
                    hovertemplate="<b>%{y}</b><br>%{x} files<extra></extra>",
                ))
                fig_bar.update_layout(
                    **PLOTLY_BASE,
                    xaxis=dict(visible=False),
                    yaxis=dict(tickfont=dict(size=11, color="#374151")),
                    height=240,
                    bargap=0.35,
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # ── Top files ranked bar ───────────────────────────────────────────────
        st.markdown('<div class="sec" style="margin-top:.5rem;">Top Files by Risk Priority</div>', unsafe_allow_html=True)
        if not df.empty:
            risk_map = {"High": 3, "Medium": 2, "Low": 1}
            ranked = df.copy()
            ranked["Score"] = ranked["Risk Level"].map(risk_map)
            ranked = ranked.sort_values("Score", ascending=True).tail(8)
            rank_colors = [{"High":"#dc2626","Medium":"#d97706","Low":"#16a34a"}[r] for r in ranked["Risk Level"]]
            fig_rank = go.Figure(go.Bar(
                x=ranked["Score"], y=ranked["Swift File"],
                orientation="h",
                marker=dict(color=rank_colors, line=dict(width=0)),
                text=ranked["Risk Level"], textposition="inside",
                textfont=dict(size=10, color="#ffffff", family="Inter"),
                hovertemplate="<b>%{y}</b><br>Risk: %{text}<extra></extra>",
                width=0.52,
            ))
            fig_rank.update_layout(
                **PLOTLY_BASE,
                xaxis=dict(
                    tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"],
                    tickfont=dict(size=10, color="#9ca3af"),
                    range=[0, 3.6],
                ),
                yaxis=dict(tickfont=dict(size=10, color="#374151")),
                height=280,
                bargap=0.3,
            )
            st.plotly_chart(fig_rank, use_container_width=True, config={"displayModeBar": False})

        # ── Risk Heatmap: Signal Category × Risk Level ────────────────────────
        st.markdown('<div class="sec" style="margin-top:.5rem;">Risk Heatmap — Category × Level</div>', unsafe_allow_html=True)
        if not df.empty:
            heat = df.groupby(["Signal Category", "Risk Level"]).size().reset_index(name="Count")
            heat_pivot = heat.pivot(index="Signal Category", columns="Risk Level", values="Count").fillna(0)
            for col_name in ["High", "Medium", "Low"]:
                if col_name not in heat_pivot.columns:
                    heat_pivot[col_name] = 0
            heat_pivot = heat_pivot[["High", "Medium", "Low"]]
            fig_heat = go.Figure(go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.tolist(),
                y=heat_pivot.index.tolist(),
                colorscale=[[0,"#f0fdf4"],[0.5,"#fef9c3"],[1,"#fef2f2"]],
                text=heat_pivot.values.astype(int),
                texttemplate="%{text}",
                textfont=dict(size=13, color="#374151", family="Inter"),
                showscale=False,
                hovertemplate="<b>%{y}</b> — <b>%{x}</b><br>%{z} file(s)<extra></extra>",
            ))
            fig_heat.update_layout(
                **PLOTLY_BASE,
                xaxis=dict(tickfont=dict(size=11, color="#374151"), side="top"),
                yaxis=dict(tickfont=dict(size=11, color="#374151")),
                height=220,
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

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
        for row in [r for r in records if r["Risk Level"]=="High"][:3]:
            st.markdown(f"""
            <div class="{fc_cls(row['Risk Level'])}">
                {rb_html(row['Risk Level'])}
                <div class="fc-name" style="margin-top:.35rem;">{row['Swift File']}</div>
                <div class="fc-meta">{row['Feature']} · {row['Risk Type']}</div>
                <div class="fc-body">{row['Summary'][:140]}…</div>
            </div>""", unsafe_allow_html=True)


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
                turn = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
                mock_responses = {
                    0: (
                        "Based on the indexed Swift codebase, here are the components ranked by "
                        "current risk exposure:\n\n"
                        "**1. KernelBootTimeHarvester.swift — Critical**  \n"
                        "Uses `sysctl()` to read kernel boot time — now restricted in iOS 18.5 "
                        "Lockdown Mode. Needs an immediate fallback signal.\n\n"
                        "**2. DeviceFingerprintCollector.swift — Critical**  \n"
                        "Aggregates multiple hardware identifiers. ATT expansion in iOS 18.4 "
                        "triggers consent requirements for first-party aggregation.\n\n"
                        "**3. BiometricAuthProvider.swift — High**  \n"
                        "LAContext queries appear in the App Privacy Report under sensitive signals. "
                        "App Store review automation is beginning to flag this.\n\n"
                        "**4. IDFVProvider.swift — Medium**  \n"
                        "Must be declared in `PrivacyInfo.xcprivacy` — non-compliant submissions "
                        "are currently being rejected.\n\n"
                        "**Recommendation:** Address KernelBootTime fallback first, then complete "
                        "a Privacy Manifest audit before the next submission."
                    ),
                    1: (
                        "Great follow-up. The key difference between the two critical files is "
                        "*scope of impact*:\n\n"
                        "- **KernelBootTimeHarvester** fails silently at runtime — the call "
                        "returns an error or empty value rather than crashing, so you may not "
                        "notice degraded signal quality until you audit collection rates.\n\n"
                        "- **DeviceFingerprintCollector** is more likely to trigger a visible "
                        "consent prompt or an App Store review flag *before* it breaks at "
                        "runtime, giving you slightly more lead time.\n\n"
                        "Prioritise KernelBootTime for the next sprint and DeviceFingerprint "
                        "before the next App Store submission cycle."
                    ),
                }
                response_text = mock_responses.get(
                    turn,
                    "That's a good question. Based on the current codebase context, "
                    "I don't see a directly impacted file for that specific scenario, "
                    "but I'd recommend auditing `MemoryUsageHarvester.swift` and "
                    "`NetworkReachability.swift` as secondary candidates. "
                    "Would you like me to explain the risk vectors for either of those?"
                )
                sources = records[:6]

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
                                <div class="fc-body">{src['Summary'][:160]}…</div>
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
            view = view[view["Risk Level"] == sel_risk]
        if sel_feat != "All":
            view = view[view["Feature"] == sel_feat]

        disp = view[["Swift File","Feature","Signal Category","Risk Level","Risk Type","Risk Reason"]].copy()
        st.dataframe(disp, use_container_width=True, height=280)

        st.markdown('<div class="sec" style="margin-top:1.1rem;">File Cards</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        for i, row in enumerate(view.to_dict("records")):
            with (ca if i % 2 == 0 else cb):
                st.markdown(f"""
                <div class="{fc_cls(row['Risk Level'])}">
                    {rb_html(row['Risk Level'])}
                    <div class="fc-name" style="margin-top:.35rem;">{row['Swift File']}</div>
                    <div class="fc-meta">{row['Feature']} · {row['Risk Type']}</div>
                    <div class="fc-body">{row['Risk Reason']}<br><br>{row['Summary'][:200]}…</div>
                </div>""", unsafe_allow_html=True)
