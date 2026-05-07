"""
LlamaIndex RAG chain with Ollama (local Llama 3.2).

LlamaIndex vs LangChain (used in Projects 1 & 2):
- LlamaIndex: data-centric, optimised for document retrieval,
  first-class support for structured data, cleaner indexing API
- LangChain: agent-centric, more integrations, LCEL composability
- Both are industry standard — knowing both differentiates you

Settings is LlamaIndex's global config singleton (replaced ServiceContext
in 0.10+). All nodes, indexes, and query engines inherit from it.
"""

import logging
import os

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)


def configure_settings(embed_model) -> None:
    """
    Set LlamaIndex global settings once at app startup.

    chunk_size=512 / overlap=50: smaller than Project 1's 1000/200
    because Llama 3.2 (3B) has a smaller effective context window
    than Gemini 2.0 Flash for maintaining coherence.

    OLLAMA_BASE_URL allows overriding when running inside Docker
    (set to http://host.docker.internal:11434).
    """
    Settings.embed_model = embed_model
    Settings.llm = Ollama(
        model="llama3.2:3b",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=120.0,  # CPU inference is slow — give it time
        temperature=0.0,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    logger.info("LlamaIndex Settings configured: llama3.2 + HuggingFace bge-small")


def build_index(vector_store, documents) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex backed by Pinecone.

    StorageContext wires the index to an external vector store.
    Documents are chunked (by Settings.node_parser) and embedded
    (by Settings.embed_model) before being upserted to Pinecone.
    """
    logger.info(f"Building index from {len(documents)} documents...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    logger.info("Index built")
    return index


def build_query_engine(index: VectorStoreIndex, top_k: int = 4) -> RetrieverQueryEngine:
    """
    Build a retriever-backed query engine.

    top_k=4 is the default from Project 1 — retrieve 4 chunks,
    inject them as context, and let the LLM answer.
    """
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    return RetrieverQueryEngine(retriever=retriever)
