"""
HuggingFace embeddings — runs 100% locally, no API key needed.

Model: BAAI/bge-small-en-v1.5
- 384 dimensions (vs OpenAI text-embedding-3-small's 1536)
- ~130MB download, runs on CPU
- Ranked highly on MTEB (Massive Text Embedding Benchmark)
- Perfect for: privacy-sensitive data, cost-zero experiments

When to use HuggingFace vs API-based embeddings:
- HuggingFace (local): private data, zero cost, offline environments
- OpenAI / Gemini: production, highest accuracy on multilingual content
"""

import logging
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


def get_hf_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbedding:
    """
    Load HuggingFace embedding model.

    BAAI/bge-small-en-v1.5 beats all-MiniLM-L6-v2 on retrieval benchmarks
    while using the same 384-dim output — matching our Pinecone index dimension.

    First call downloads the model (~130MB) to ~/.cache/huggingface.
    Subsequent calls load from disk cache.
    """
    logger.info(f"Loading HuggingFace embedding model: {model_name}")
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    logger.info("Embedding model ready")
    return embed_model
