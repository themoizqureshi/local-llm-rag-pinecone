"""
Streamlit UI for Local LLM RAG — LlamaIndex + Ollama + Pinecone.

Run with:
  1. ollama pull llama3.2:3b       (one-time model download)
  2. streamlit run app.py

Requires PINECONE_API_KEY in .env.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Local LLM RAG", page_icon="🦙", layout="wide")


# ── Dependency checks ─────────────────────────────────────────────────────────
def check_ollama() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def check_pinecone() -> bool:
    return bool(os.getenv("PINECONE_API_KEY"))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")

    ollama_ok = check_ollama()
    pinecone_ok = check_pinecone()

    st.markdown(f"{'✅' if ollama_ok else '❌'} **Ollama** (llama3.2:3b)")
    if not ollama_ok:
        st.caption("Start Ollama: `ollama serve` then `ollama pull llama3.2:3b`")

    st.markdown(f"{'✅' if pinecone_ok else '❌'} **Pinecone** API key")
    if not pinecone_ok:
        st.caption("Add `PINECONE_API_KEY` to `.env`")

    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("LlamaIndex · Ollama (llama3.2:3b)")
    st.markdown("BAAI/bge-small-en-v1.5 (local)")
    st.markdown("Pinecone (managed vector DB)")

    if st.button("🗑 Clear & Upload New PDF") and "query_engine" in st.session_state:
        del st.session_state["query_engine"]
        del st.session_state["doc_name"]
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🦙 Local LLM RAG — Chat with Your PDF")
st.caption("LlamaIndex · Ollama (llama3.2:3b local) · Pinecone · BAAI/bge-small-en-v1.5")

if not ollama_ok or not pinecone_ok:
    st.error(
        "Prerequisites missing — check the System Status panel in the sidebar. "
        "Both Ollama and Pinecone must be ready before uploading a document."
    )
    st.stop()


# ── Session state ─────────────────────────────────────────────────────────────
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
    st.session_state.doc_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Upload & Ingest ───────────────────────────────────────────────────────────
if st.session_state.query_engine is None:
    st.subheader("Upload a PDF to get started")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded:
        with st.spinner("Ingesting PDF into Pinecone... (first run downloads the embedding model)"):
            from src.embeddings import get_hf_embeddings
            from src.pinecone_store import get_or_create_pinecone_index, get_vector_store
            from src.ingestion import load_pdf
            from src.chain import configure_settings, build_index, build_query_engine

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded.getvalue())
            tmp.flush()

            embed_model = get_hf_embeddings()
            configure_settings(embed_model)

            pinecone_index = get_or_create_pinecone_index()
            vector_store = get_vector_store(pinecone_index)
            documents = load_pdf(tmp.name)
            index = build_index(vector_store, documents)
            st.session_state.query_engine = build_query_engine(index)
            st.session_state.doc_name = uploaded.name
            os.unlink(tmp.name)

        st.success(f"✅ Indexed **{uploaded.name}** ({len(documents)} pages). Ask a question below.")
        st.rerun()
else:
    st.success(f"📄 Document loaded: **{st.session_state.doc_name}**")


# ── Chat Interface ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Source chunks"):
                for src in msg["sources"]:
                    st.markdown(f"> {src}")

if prompt := st.chat_input(
    "Ask a question about your document...",
    disabled=st.session_state.query_engine is None,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Querying Ollama (local inference)..."):
            response = st.session_state.query_engine.query(prompt)
            answer = str(response)
            sources = []
            if hasattr(response, "source_nodes"):
                sources = [node.text[:300] + "..." for node in response.source_nodes]

        st.markdown(answer)
        if sources:
            with st.expander("Source chunks"):
                for src in sources:
                    st.markdown(f"> {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
