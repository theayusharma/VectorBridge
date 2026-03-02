# VectorBridge

A Retrieval-Augmented Generation (RAG) API for answering questions about documents.

## What it does

VectorBridge processes documents (PDF, DOCX, DOC, TXT) from URLs and answers questions about their content using AI. It combines:
- Multi-format document parsing (including PDF table extraction)
- Semantic text chunking
- Hybrid retrieval (FAISS + BM25 + reranking)
- Google Gemini for answer generation

## Quick Start

```bash
# Run the server
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hackrx/run` | POST | Process document & answer questions |
| `/health` | GET | Health check |
| `/cache/purge` | POST | Purge cached data |

```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `CHUNK_SIZE` | 800 | Text chunk size in characters |
| `CHUNK_OVERLAP` | 150 | Chunk overlap in characters |
| `TOP_K_VECTOR` | 12 | Number of chunks to retrieve via FAISS |
| `TOP_K_BM25` | 8 | Number of chunks to retrieve via BM25 |
| `TOP_K_RERANKED` | 8 | Final number of chunks after reranking |
| `EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| `RERANKER_MODEL` | cross-encoder/ms-marco-MiniLM-L-12-v2 | Reranker model |
| `LLM_MODEL` | gemini-2.5-flash | LLM model for answer generation |
| `MAX_FILE_SIZE` | 200MB | Maximum document file size |

## Project Structure

```
src/
├── app.py                  # FastAPI entry point
├── config.py               # Configuration
├── chunker.py              # Semantic text chunking
├── doc_downloader.py       # Document downloader
├── parsers.py              # Multi-format parsers (PDF, DOCX, DOC, TXT)
├── vector_store.py         # FAISS vector management
├── retriever.py            # Hybrid retrieval (FAISS + BM25 + reranking)
├── question_processor.py   # LLM answer generation
├── geminiloadbalance.py    # API key rotation
└── cache.py                # Processing cache
```
