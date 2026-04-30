"""
Pinecone vector store setup.

Pinecone is a fully managed vector database — no infrastructure to manage.
Free tier: 1 serverless index, 100K vectors, 1GB storage.

Key concepts:
- Index: like a database table, holds vectors + metadata
- Namespace: logical separation within an index (per user, per document)
- Dimension: must match the embedding model output (384 for bge-small)
- Metric: cosine similarity is standard for semantic search

ChromaDB (Project 1) vs Pinecone (this project):
- ChromaDB: local disk, zero setup, dev only, not scalable
- Pinecone: managed cloud, scales to billions of vectors, production-ready
"""

import logging
import os

from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore

logger = logging.getLogger(__name__)

INDEX_NAME = "rag-demo"
EMBEDDING_DIMENSION = 384  # Must match BAAI/bge-small-en-v1.5 output dimension


def get_or_create_pinecone_index():
    """
    Return existing Pinecone index or create it if absent.

    Uses serverless spec (free-tier friendly).
    aws/us-east-1 is the free-tier region.
    Creating an index is idempotent — safe to call repeatedly.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        logger.info(f"Creating Pinecone index '{INDEX_NAME}' (dim={EMBEDDING_DIMENSION})")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Index created")
    else:
        logger.info(f"Using existing Pinecone index '{INDEX_NAME}'")

    return pc.Index(INDEX_NAME)


def get_vector_store(index) -> PineconeVectorStore:
    """Wrap the Pinecone index as a LlamaIndex VectorStore."""
    return PineconeVectorStore(pinecone_index=index)
