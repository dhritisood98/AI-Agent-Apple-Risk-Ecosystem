import streamlit as st
import sys
import os

# Ensure the root directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change these from relative (.) to absolute (src.)
from src.config import settings
from src.embedders import get_embedder_spec, Embedder
from src.retriever import Retriever
from src.llm_clients import NIMClient
from src.prompts import build_prompt

# 1. Page Configuration
st.set_page_config(page_title="iOS Risk Sentinel", page_icon="🍎", layout="wide")
st.title("🍎 iOS Security & Risk Sentinel")
st.markdown("Query Apple security releases to identify platform fraud risks.")

# 2. Initialize Backend (Cached for speed)
@st.cache_resource
def init_tools():
    spec = get_embedder_spec("bge_small")
    embedder = Embedder(spec)
    retriever = Retriever(settings.supabase_url, settings.supabase_key)
    llm = NIMClient(settings.nvidia_api_key, settings.nim_base_url)
    return spec, embedder, retriever, llm

spec, embedder, retriever, llm = init_tools()

# 3. Sidebar for Settings
with st.sidebar:
    st.header("Search Settings")
    k_value = st.slider("Number of context chunks (k)", 1, 10, 5)
    model_choice = st.text_input("NIM Model", "meta/llama3-8b-instruct")

# 4. Main Chat Interface
question = st.text_input("Enter your security query:", placeholder="e.g., WebKit vulnerabilities in iOS 18")

if st.button("Analyze Risk"):
    if question:
        with st.spinner("Searching Apple security database..."):
            # Step A: Embed & Retrieve
            qvec = embedder.embed_query(question)
            retrieved = retriever.retrieve(spec, qvec, k=k_value)
            
            # Step B: Generate Answer
            prompt = build_prompt(question, retrieved)
            response = llm.generate(prompt, model=model_choice)
            
            # 5. Display Results
            st.subheader("🤖 AI Risk Analysis")
            st.write(response.text)
            
            # 6. Show Sources (Expander)
            with st.expander("View Retrieved Context Chunks"):
                for i, res in enumerate(retrieved):
                    st.info(f"**Source {i+1}** (Similarity: {res['similarity']:.4f})")
                    st.write(res['chunk_text'])
                    st.divider()
    else:
        st.warning("Please enter a question first.")