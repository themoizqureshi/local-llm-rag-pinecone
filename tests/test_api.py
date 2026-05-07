"""
FastAPI endpoint tests using TestClient (synchronous, no real API calls).

Strategy: patch all external dependencies at the src.api module level so
the lifespan function completes without touching Pinecone or Ollama.
Reset state between tests to avoid cross-test contamination.
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient with all external dependencies mocked."""
    with patch("src.api.get_hf_embeddings", return_value=MagicMock()), \
         patch("src.api.get_or_create_pinecone_index", return_value=MagicMock()), \
         patch("src.api.get_vector_store", return_value=MagicMock()), \
         patch("src.api.configure_settings"):
        from src.api import app, state
        state.query_engine = None
        state.index = None
        with TestClient(app) as c:
            yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "llama3.2:3b"
    assert data["vectorstore"] == "pinecone"
    assert data["document_loaded"] is False


def test_query_before_ingest_returns_400(client):
    response = client.post("/query", json={"question": "What is this document about?"})
    assert response.status_code == 400
    assert "No document indexed" in response.json()["detail"]


def test_ingest_rejects_non_pdf(client):
    files = {"file": ("notes.txt", io.BytesIO(b"some text"), "text/plain")}
    response = client.post("/ingest", files=files)
    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]


def test_ingest_indexes_document(client):
    mock_docs = [MagicMock(), MagicMock(), MagicMock()]

    with patch("src.api.SimpleDirectoryReader") as mock_reader, \
         patch("src.api.build_index", return_value=MagicMock()), \
         patch("src.api.build_query_engine", return_value=MagicMock()):
        mock_reader.return_value.load_data.return_value = mock_docs

        pdf_bytes = b"%PDF-1.4 fake pdf content"
        files = {"file": ("sample.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        response = client.post("/ingest", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["chunks_indexed"] == 3
    assert "sample.pdf" in data["message"]


def test_query_after_ingest_returns_answer(client):
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "The document discusses RAG architecture."
    mock_qe = MagicMock()
    mock_qe.query.return_value = mock_response

    with patch("src.api.SimpleDirectoryReader") as mock_reader, \
         patch("src.api.build_index", return_value=MagicMock()), \
         patch("src.api.build_query_engine", return_value=mock_qe):
        mock_reader.return_value.load_data.return_value = [MagicMock()]
        pdf_bytes = b"%PDF-1.4 fake"
        files = {"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        client.post("/ingest", files=files)

    response = client.post("/query", json={"question": "What is this about?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The document discusses RAG architecture."
    assert data["model"] == "llama3.2:3b"
