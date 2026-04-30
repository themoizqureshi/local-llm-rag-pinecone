# Local LLM RAG — Llama 3.2 + HuggingFace + Pinecone

> Same RAG concept as Project 1, but fully cloud-agnostic: local Llama 3.2 via Ollama, local HuggingFace embeddings, Pinecone for production-grade vector storage, and a FastAPI REST backend. **Proves you're not locked to OpenAI.**

![Python](https://img.shields.io/badge/python-3.11-blue)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Pinecone](https://img.shields.io/badge/Pinecone-serverless-purple)

## What This Does

```
POST /ingest  (PDF file)
      ↓
SimpleDirectoryReader → SentenceSplitter (chunk_size=512)
      ↓
BAAI/bge-small-en-v1.5 embeddings (384-dim, local, free)
      ↓
Pinecone serverless index (cloud, free tier)

POST /query  {"question": "..."}
      ↓
Embed question → Pinecone similarity search (top_k=4)
      ↓
Llama 3.2 via Ollama (local, free, private)
      ↓
{"answer": "...", "model": "llama3.2"}
```

## Quick Start

```bash
# 1. Install Ollama and pull the model (one-time)
brew install ollama          # Mac
ollama pull llama3.2
ollama serve                 # Runs at http://localhost:11434

# 2. Clone and set up
git clone https://github.com/YOUR_USERNAME/local-llm-rag-pinecone
cd local-llm-rag-pinecone

cp .env.example .env
# Add your PINECONE_API_KEY from pinecone.io (free tier)

uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Run the API
uvicorn src.api:app --reload --port 8000
# API docs: http://localhost:8000/docs

# 4. Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@path/to/your.pdf"

# 5. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

## Running with Docker

```bash
# Copy .env first, then:
docker-compose up --build
# API available at http://localhost:8000
# Note: Ollama must be running on the host machine
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Llama 3.2 (via Ollama) | Free, local, private — no API key |
| Embeddings | BAAI/bge-small-en-v1.5 | Top MTEB retrieval score at 384 dims |
| Vector Store | Pinecone (serverless) | Production-grade managed infra |
| Framework | LlamaIndex 0.11 | Data-centric RAG, global Settings API |
| API | FastAPI | Testable REST backend with auto OpenAPI docs |
| Container | Docker + docker-compose | One-command deployment |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check, shows if document is loaded |
| POST | `/ingest` | Upload PDF → chunks → Pinecone |
| POST | `/query` | Question → Pinecone retrieval → Llama 3.2 answer |

Full interactive docs at `http://localhost:8000/docs` when running.

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/how_it_works.md](docs/how_it_works.md) | Deep-dive: Ollama, HuggingFace, Pinecone, LlamaIndex Settings, FastAPI lifespan |
| [docs/interview_prep.md](docs/interview_prep.md) | Q&A on local LLMs, Pinecone vs ChromaDB, FastAPI vs Streamlit, trade-offs |
| [docs/architecture.md](docs/architecture.md) | Pipeline diagram and component comparison table |

## Lessons Learned

- *Fill in after building.*

---

*Part of the [AI Engineer Portfolio](https://github.com/YOUR_USERNAME) — Project 3 of 5.*
