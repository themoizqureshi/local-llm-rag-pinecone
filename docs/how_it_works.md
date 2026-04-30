# How It Works — Local LLM RAG with Pinecone

## What Makes This Different From Project 1

Project 1 used LangChain + Gemini (cloud LLM) + ChromaDB (local vector store). This project flips two of those three choices:

- **Cloud LLM → Local LLM** (Gemini → Llama 3.2 via Ollama)
- **Local vector store → Cloud vector store** (ChromaDB → Pinecone)
- **LangChain → LlamaIndex** (different but equivalent framework)

The result: zero LLM API costs, a production-grade vector DB, and a proper REST API instead of a UI. This combination proves you're not locked to any one vendor.

---

## Component 1: Ollama (Local LLM Server)

Ollama is a tool that runs open-source LLMs as a local HTTP server. When you run `ollama serve`, it starts an API at `http://localhost:11434`. When you run `ollama pull llama3.2`, it downloads the Llama 3.2 3B model (a ~2GB quantized GGUF file).

```
ollama pull llama3.2
ollama run llama3.2 "test"   ← interactive
# or via API:
curl http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"hello"}'
```

LlamaIndex's `Ollama` class wraps this HTTP API:

```python
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(
    model="llama3.2",
    base_url="http://localhost:11434",
    request_timeout=120.0,  # CPU inference can be 10-30s
    temperature=0.0,
)
```

**Why `temperature=0`?** Same reason as Project 1 — factual Q&A needs deterministic, grounded answers. Temperature 0 = always pick the highest-probability token.

**CPU vs GPU performance:**
- On a modern MacBook (Apple Silicon), Llama 3.2 3B runs at ~15-25 tokens/second
- On CPU-only Linux, expect ~3-8 tokens/second
- GPU is 10-20x faster but not required for this project

---

## Component 2: HuggingFace Embeddings (Local)

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

The `sentence-transformers` library downloads the model from HuggingFace Hub on first use, then caches it at `~/.cache/huggingface/`. Every subsequent load is instant.

**BAAI/bge-small-en-v1.5 properties:**
- Output dimensions: **384** (this is your Pinecone index dimension)
- Model size: ~130MB
- Ranking on MTEB benchmark: better than all-MiniLM-L6-v2 at similar size
- Language: English-only (bge-m3 is the multilingual version)

The embedding process is identical to Project 1 — convert text to a dense vector, then measure angular distance (cosine similarity) between the query vector and stored chunk vectors.

---

## Component 3: LlamaIndex Settings API

LangChain uses LCEL chains: `retriever | prompt | llm | parser`. LlamaIndex uses a different pattern — a **global Settings singleton** that every component inherits from:

```python
from llama_index.core import Settings

Settings.embed_model = embed_model    # Used by indexing + retrieval
Settings.llm = Ollama(...)           # Used by query engines
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
```

You set these once at startup. Every index, retriever, and query engine you build automatically uses them. No need to thread them through every function call.

**Why chunk_size=512 (smaller than Project 1's 1000)?**

Llama 3.2 3B has a smaller effective context window than Gemini 2.0 Flash. Injecting 4 chunks × 512 chars = ~2048 chars of context keeps the total prompt well within the model's comfortable range and produces more coherent answers.

---

## Component 4: Pinecone (Managed Cloud Vector Store)

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.create_index(
    name="rag-demo",
    dimension=384,       # Must match bge-small output
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

**Serverless spec** means Pinecone manages the infrastructure entirely — no pods to provision, no idle costs on the free tier. The free tier gives you one index with 100K vectors and 1GB storage.

**Why cosine metric?**

Cosine similarity measures the angle between two vectors, ignoring magnitude. This is ideal for embeddings where the direction encodes meaning and the magnitude encodes confidence. A normalized embedding of "What is revenue?" and "Tell me about earnings" will have a small angle (similar direction) → high cosine similarity → correctly retrieved as related.

---

## Component 5: FastAPI Lifespan Events

The standard pattern for FastAPI apps that need to initialize expensive resources:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before yield runs ONCE at startup
    embed_model = get_hf_embeddings()    # Downloads/loads ~130MB model
    configure_settings(embed_model)
    pinecone_index = get_or_create_pinecone_index()
    state.vector_store = get_vector_store(pinecone_index)
    yield
    # Everything after yield runs ONCE at shutdown (cleanup)
    logger.info("Shutting down")

app = FastAPI(lifespan=lifespan)
```

Without lifespan: if you loaded the embedding model inside the `/ingest` handler, it would download 130MB on every request. That would be ~30s per request instead of ~30s once at startup.

---

## Component 6: The Ingestion Pipeline

```
POST /ingest (PDF bytes)
       │
       ▼
Save to tempfile
       │
       ▼
SimpleDirectoryReader(input_files=[path]).load_data()
→ List[Document] (one Document per page, with file_path/page metadata)
       │
       ▼
VectorStoreIndex.from_documents(documents, storage_context)
     internally:
     ├── SentenceSplitter → List[TextNode] (chunks with chunk_size=512)
     ├── embed_model.get_text_embedding(node.text) → 384-dim vector
     └── pinecone_index.upsert(vectors=[...])
       │
       ▼
state.index = index
state.query_engine = RetrieverQueryEngine(retriever)
```

---

## Component 7: The Query Pipeline

```
POST /query {"question": "..."}
       │
       ▼
embed_model.get_query_embedding(question) → 384-dim vector
       │
       ▼
pinecone_index.query(vector, top_k=4)
→ 4 most similar TextNodes with similarity scores
       │
       ▼
RetrieverQueryEngine assembles context string from node texts
       │
       ▼
Ollama (Llama 3.2): "Answer this question using only the context: ..."
       │
       ▼
QueryResponse(answer=str(response))
```

---

## Why a REST API Instead of Streamlit?

Projects 1 and 4 use Streamlit (great for demos). This project uses FastAPI to show you can build a production-grade backend:

- **Testable**: each endpoint has clear inputs and outputs, easy to write pytest tests
- **Integrable**: any frontend (React, mobile, CLI) can call it
- **Containerizable**: Dockerfile + docker-compose ship the entire service
- **Documentable**: FastAPI auto-generates OpenAPI docs at `http://localhost:8000/docs`

In a real product, the RAG logic would be a FastAPI service that your frontend calls — not a monolithic Streamlit app.
