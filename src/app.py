import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.config import settings
from src.embedders import get_embedder_spec, Embedder
from src.retriever import Retriever
from src.llm_clients import NIMClient
from src.prompts import build_prompt

st.set_page_config(page_title="iOS Risk Sentinel", page_icon="📱", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
    color: #1f2937;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.5rem;
}
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.small-label {
    color: #6b7280;
    font-size: 0.85rem;
}
.big-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #111827;
}
.badge-high, .badge-medium, .badge-low {
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-high { background:#fee2e2; color:#b91c1c; }
.badge-medium { background:#fef3c7; color:#b45309; }
.badge-low { background:#dcfce7; color:#15803d; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_sentinel():
    spec = get_embedder_spec("bge_small")
    embedder = Embedder(spec)
    retriever = Retriever(settings.supabase_url, settings.supabase_key)
    llm = NIMClient(settings.nvidia_api_key, settings.nim_base_url)
    return spec, embedder, retriever, llm


spec, embedder, retriever, llm = init_sentinel()


def classify_risk(text: str) -> tuple[str, str]:
    t = text.lower()
    if any(x in t for x in ["kernel", "break", "blocked", "critical", "degraded", "privacy restriction"]):
        return "High", "Signal Degradation"
    if any(x in t for x in ["reduced", "partial", "limited", "network", "api change"]):
        return "Medium", "API Dependency"
    return "Low", "Operational"


def risk_badge(level: str):
    if level == "High":
        return '<span class="badge-high">High</span>'
    if level == "Medium":
        return '<span class="badge-medium">Medium</span>'
    return '<span class="badge-low">Low</span>'


st.title("📱 iOS Risk Sentinel")
st.caption("Track Apple ecosystem changes against iOS security bulletin intelligence.")

CURRENT_IOS = "26.3.1"
BETA_IOS = "26.4 beta 3"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="card"><div class="small-label">Current Stable iOS</div><div class="big-value">{CURRENT_IOS}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><div class="small-label">Current Beta</div><div class="big-value">{BETA_IOS}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card"><div class="small-label">Knowledge Source</div><div class="big-value">Apple Security Bulletins</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="card"><div class="small-label">Vector Engine</div><div class="big-value">BGE Small 384</div></div>', unsafe_allow_html=True)

st.caption("Apple currently lists iOS 26.3.1 as the latest stable release, while Developer releases include iOS 26.4 beta 3.")

tab1, tab2, tab3 = st.tabs(["Overview", "Chatbot Analysis", "Affected Components"])

with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Risk Breakdown")
        risk_df = pd.DataFrame({
            "Risk Type": ["Signal Degradation", "API Dependency", "Privacy Restriction", "Operational"],
            "Count": [4, 3, 2, 1]
        })

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(risk_df["Count"], labels=risk_df["Risk Type"], autopct="%1.0f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    with right:
        st.subheader("Latest Project View")
        st.markdown("""
        <div class="card">
        <b>Observed risk themes in current demo pipeline</b><br><br>
        - Security bulletin changes from Apple releases<br>
        - Potential signal degradation from platform changes<br>
        - Impact to fraud and device intelligence workflows<br>
        - Need for monitoring patched vulnerabilities and affected devices
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Sentinel Chatbot")

    query = st.text_input(
        "Ask about iOS vulnerabilities, Apple security changes, or business impact",
        placeholder="What are the latest iOS WebKit risks and how could they affect device security monitoring?"
    )

    k = st.slider("Top-k retrieval", 1, 10, 4)

    if st.button("Run Analysis") and query:
        with st.spinner("Analyzing..."):
            qvec = embedder.embed_query(query)
            results = retriever.retrieve(query_embedding=qvec, k=k)

            system_instructions = """
            You are an iOS digital risk analyst.
            Explain:
            1. the main security issue or platform change
            2. affected products or components
            3. likely business or fraud-risk impact
            4. practical recommendation
            """

            prompt = build_prompt(query, results, system_instructions=system_instructions)
            response = llm.generate(prompt, model="meta/llama-3.1-70b-instruct")

            level, rtype = classify_risk(response.text)

            st.markdown("### Sentinel Response")
            st.markdown(risk_badge(level), unsafe_allow_html=True)
            st.write(f"**Risk Type:** {rtype}")
            st.write(response.text)

            st.markdown("### Retrieved Context")
            for r in results:
                st.markdown(f"""
                <div class="card">
                <b>Chunk ID: {r.get('snapshot_chunk_id', 'unknown')}</b><br><br>
                {r.get('chunk_text', '')}<br><br>
                <span class="small-label">Similarity: {r.get('similarity', 0):.3f}</span>
                </div>
                """, unsafe_allow_html=True)

with tab3:
    st.subheader("Affected Components")
    st.write("This section can later be driven from saved retrieval results or structured risk outputs.")

    demo_df = pd.DataFrame([
        {"component": "WebKit", "risk_level": "High", "risk_type": "Remote Code Execution / Web Content Risk"},
        {"component": "Kernel", "risk_level": "High", "risk_type": "Privilege Escalation"},
        {"component": "Apple Security Releases", "risk_level": "Medium", "risk_type": "Patch Monitoring"},
    ])

    st.dataframe(demo_df, use_container_width=True)