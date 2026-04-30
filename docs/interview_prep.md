# Interview Prep — Local LLM RAG with Pinecone

> This project answers the interviewer question: "Are you just an OpenAI wrapper, or do you actually understand the underlying infrastructure?" Building with local LLMs, local embeddings, and a managed cloud vector store proves you can navigate the full stack.

---

## Core Concept Questions

### Q: Walk me through this project and why it's different from your first RAG project.

> "Both projects build RAG systems, but they make opposite choices on every axis. Project 1 uses cloud APIs for everything (Gemini for LLM, Google for embeddings, ChromaDB for local storage) and a Streamlit demo UI. This project runs the LLM locally via Ollama (zero API cost), runs embeddings locally via HuggingFace sentence-transformers, uses Pinecone as a production-grade managed vector store, and exposes a proper FastAPI REST backend instead of a UI.
>
> The architecture proves I understand both the 'convenient cloud' path and the 'cost-controlled, privacy-preserving' path — and can articulate the trade-offs between them."

---

### Q: Why would you choose a local LLM over a cloud LLM?

> "Three main reasons:
>
> **Privacy**: If the data is medical records, financial documents, or proprietary code, you cannot send it to a third-party API. Local LLMs let you process sensitive data without it ever leaving your infrastructure.
>
> **Cost at scale**: Cloud LLM APIs charge per token. At 100K queries/day, GPT-4o costs thousands of dollars monthly. Running Llama 3.2 on your own hardware has a one-time infrastructure cost and then near-zero marginal cost per query.
>
> **Latency**: For some use cases (real-time code completion, embedded devices), the network round-trip to a cloud API is unacceptable. Local inference runs in milliseconds without network latency.
>
> The trade-off is model quality — Llama 3.2 3B is significantly less capable than Gemini 2.0 Flash or GPT-4o for complex reasoning. I choose local LLMs when the queries are focused (RAG-grounded factual Q&A), not open-ended reasoning tasks."

---

### Q: What is Ollama and how does it work under the hood?

> "Ollama is a local model server. When you run `ollama pull llama3.2`, it downloads a GGUF-quantized version of the model (about 2GB for 3B parameters at 4-bit quantization). When you run `ollama serve`, it starts an HTTP server at `localhost:11434` that exposes a REST API identical in structure to OpenAI's API.
>
> The quantization is key — GGUF Q4 format compresses the model's float32 weights to 4-bit integers, reducing memory from ~6GB to ~2GB. There's a quality trade-off (roughly 1-3% degradation on benchmarks), but it makes the 3B model runnable on any modern laptop without a GPU.
>
> In this project, LlamaIndex's `Ollama` class just sends HTTP requests to `localhost:11434/api/generate`. Setting `OLLAMA_BASE_URL=http://host.docker.internal:11434` in docker-compose makes this work from inside a container."

---

### Q: Why BAAI/bge-small-en-v1.5 over all-MiniLM-L6-v2?

> "Both are 384-dimension, CPU-friendly models. `bge-small-en-v1.5` (BGE = BAAI General Embedding) consistently ranks higher on the MTEB retrieval benchmark — it was specifically trained with contrastive learning on retrieval tasks, while `all-MiniLM-L6-v2` was trained for general sentence similarity.
>
> For RAG specifically, retrieval quality matters more than semantic similarity quality. BGE models were trained to pull back exactly the chunks that contain the answer — which is the RAG retrieval task. The practical difference is a few percentage points on context_recall scores, but those percentage points matter when you're trying to stay above the 0.75 quality threshold."

---

### Q: Explain the Pinecone dimension mismatch error and how to prevent it.

> "When you create a Pinecone index, you specify a `dimension` parameter that must match the output size of your embedding model exactly. BAAI/bge-small-en-v1.5 outputs 384-dimensional vectors. If you accidentally create an index with `dimension=768` (the size of the larger bge-base model) and then try to upsert 384-dim vectors, Pinecone will throw a `dimension mismatch` error at upsert time.
>
> Prevention: always define `EMBEDDING_DIMENSION = 384` as a named constant in `pinecone_store.py` and reference it in both `create_index` and as documentation for the embedding model choice. If you change the embedding model, you must also delete and recreate the index. In production I'd add an assertion at startup: `assert embed_model.embed_dim == EMBEDDING_DIMENSION`."

---

### Q: What is the LlamaIndex Settings API and how does it differ from LangChain?

> "LangChain uses LCEL — you explicitly thread the LLM and embeddings through every chain: `retriever | prompt | llm | parser`. Every component must receive its dependencies as arguments.
>
> LlamaIndex uses a global `Settings` singleton: you set `Settings.llm`, `Settings.embed_model`, and `Settings.node_parser` once at startup, and every index, retriever, and query engine inherits those settings automatically. It's cleaner for data-centric use cases where you want one consistent configuration across a complex pipeline.
>
> Neither is objectively better — LangChain's explicit threading makes dependencies clearer in multi-LLM systems (e.g., one LLM for retrieval ranking, another for generation). LlamaIndex's global settings are simpler when you have one consistent configuration. Knowing both is a differentiator."

---

### Q: Why FastAPI over Streamlit for this project?

> "Streamlit (used in Projects 1 and 4) is perfect for demos and internal tools — you can build a UI in 20 lines of Python. But it's a monolith: the UI and backend are coupled, you can't version or test individual endpoints, and it's not designed to be called by other services.
>
> FastAPI is a proper backend:
> - **Testable**: each endpoint has a defined contract (Pydantic models) and can be tested with `TestClient` without a browser
> - **Integrable**: any frontend (React, mobile app, CLI tool) can call the same API
> - **Documented**: auto-generates OpenAPI spec at `/docs` — product managers and frontend engineers can explore it without reading code
> - **Containerizable**: a Dockerfile and docker-compose file are all you need to run the service anywhere
>
> In production, the RAG logic would be a FastAPI service that your frontend calls, not a monolithic app."

---

### Q: What does the FastAPI lifespan context manager do and why use it?

> "The `lifespan` context manager runs once at startup (before the first request) and once at shutdown. I use it to initialize expensive resources: loading the 130MB HuggingFace embedding model and establishing the Pinecone connection.
>
> Without lifespan: if I loaded the embedding model inside the `/ingest` handler, it would download 130MB on the first request — adding 30 seconds of cold-start latency. For a Pinecone connection, re-creating it per request would mean unnecessary network handshakes and authentication overhead.
>
> The pattern is: slow initialization happens once at startup, fast request handling calls pre-initialized objects. Same pattern as database connection pools, ML model loading, or any resource that's expensive to create but cheap to use."

---

### Q: What are the trade-offs between Pinecone vs ChromaDB vs FAISS?

> "ChromaDB (Project 1): zero setup, runs on local disk, great for development. Fails in production because it doesn't scale horizontally and requires managing your own persistence.
>
> FAISS (Facebook AI Similarity Search): ultra-fast in-memory similarity search, used inside ChromaDB and many other vector stores. Pure vector math, no metadata filtering, no persistence built-in — you manage serialization yourself. Great for research, not for building products.
>
> Pinecone: fully managed, scales to billions of vectors, handles replication and monitoring, supports metadata filtering (filter by document ID, date range, etc.), serverless pricing means you pay only for what you use. The trade-off is vendor lock-in and cost at extreme scale.
>
> My rule: ChromaDB for local development, Pinecone for production or anything that needs to scale beyond one machine."

---

### Q: How would you handle the case where Ollama is slow (30+ seconds per query)?

> "Three approaches in priority order:
>
> **1. Use a smaller quantized model**: Llama 3.2 1B Q4 is half the size of 3B and twice as fast. For simple factual Q&A over short contexts, the quality difference is acceptable.
>
> **2. Use Metal / CUDA GPU acceleration**: On Apple Silicon, Ollama automatically uses Metal for acceleration — you get 15-25 tokens/second instead of 3-8 on CPU. On Linux with NVIDIA, set `OLLAMA_METAL=1` or ensure CUDA is detected.
>
> **3. Stream the response**: Instead of waiting for the full answer, stream tokens as they're generated using `StreamingResponse` in FastAPI. The user sees the answer building in real-time, so perceived latency drops dramatically even if total generation time is the same. FastAPI's `StreamingResponse` + Ollama's streaming API make this straightforward to add."

---

## Connecting to Your Production Experience

> "At Speridian, I built document extraction pipelines on Google Cloud — essentially a managed RAG system using Google Discovery Engine. This project is me implementing the same concept from scratch with open-source components to deeply understand what managed services abstract away.
>
> The most valuable lesson: Pinecone's serverless spec is doing the same thing as Google Discovery Engine's data store — it's handling vector storage, indexing, and similarity search so I don't have to manage infrastructure. Understanding both lets me make informed build-vs-buy decisions in production."

> "The FastAPI backend pattern here mirrors how I'd expose any internal ML service at Speridian — a REST API with clear contracts, startup health checks, and Pydantic validation. The lifespan pattern for loading models is the same pattern we use for loading ML models at inference serving time."
