"""
FastAPI REST API for the local LLM RAG system.

Endpoints:
  GET  /health  — Liveness check
  POST /ingest  — Upload and index a PDF into Pinecone
  POST /query   — Ask a question against the indexed document

Design decisions:
- lifespan context manager: initialise heavy resources once at startup
  (embedding model + Pinecone connection), not per-request
- AppState singleton: avoids global variables while keeping state accessible
- Pydantic models: request/response validation with automatic OpenAPI docs
- CORS middleware: enables calls from any frontend during development
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import SimpleDirectoryReader
from pydantic import BaseModel

from .chain import build_index, build_query_engine, configure_settings
from .embeddings import get_hf_embeddings
from .pinecone_store import get_or_create_pinecone_index, get_vector_store

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppState:
    query_engine = None
    index = None
    vector_store = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once at startup: load embeddings and connect to Pinecone."""
    logger.info("Starting up: loading HuggingFace embeddings...")
    embed_model = get_hf_embeddings()
    configure_settings(embed_model)

    logger.info("Connecting to Pinecone...")
    pinecone_index = get_or_create_pinecone_index()
    state.vector_store = get_vector_store(pinecone_index)
    logger.info("Ready — POST /ingest to index a document")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Local LLM RAG API",
    description="RAG API using Llama 3.2 (Ollama) + HuggingFace embeddings + Pinecone",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4


class QueryResponse(BaseModel):
    answer: str
    model: str = "llama3.2:3b"


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "llama3.2:3b",
        "vectorstore": "pinecone",
        "document_loaded": state.query_engine is not None,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload a PDF, embed its chunks, and upsert them into Pinecone."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, file.filename)
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)

        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        state.index = build_index(state.vector_store, documents)
        state.query_engine = build_query_engine(state.index)

    logger.info(f"Indexed {file.filename}: {len(documents)} pages")
    return IngestResponse(
        message=f"Successfully indexed {file.filename}",
        chunks_indexed=len(documents),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question. Requires /ingest to have been called first."""
    if state.query_engine is None:
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet. POST to /ingest first.",
        )

    response = state.query_engine.query(request.question)
    return QueryResponse(answer=str(response))
