import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_community.retrievers import BM25Retriever
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from cache import CacheManager
from chunker import DocumentChunker
from config import Config
from doc_downloader import DocumentDownloader
from parsers import PARSER_MAPPING, DocumentParser
from question_processor import QuestionProcessor
from retriever import ContextRetriever
from vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)

load_dotenv()


# data models
class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]


class HealthCheck(BaseModel):
    status: str
    model_versions: dict
    cache_stats: dict


security_scheme = HTTPBearer(auto_error=False)


async def validate_api_key(
    token: HTTPAuthorizationCredentials = Depends(security_scheme),
):
    if not token or token.credentials != os.getenv("HACKRX_AUTH_TOKEN"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )
    return token.credentials


# Application Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    config = Config()
    app.state.config = config

    app.state.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    app.state.downloader = DocumentDownloader(
        max_size=config.MAX_FILE_SIZE, timeout=config.REQUEST_TIMEOUT
    )

    app.state.parser = DocumentParser(
        parsers=PARSER_MAPPING,
        max_workers=4,
        max_file_size=config.MAX_FILE_SIZE,
        timeout=600.0,
    )

    app.state.chunker = DocumentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        semantic_threshold=config.SEMANTIC_SIMILARITY_THRESHOLD,
        embeddings=config.embeddings,
    )

    app.state.processing_cache = CacheManager(cache_dir=config.CACHE_DIR)

    app.state.vector_mgr = VectorStoreManager(
        embeddings=config.embeddings,
        cache_dir=config.CACHE_DIR,
    )

    app.state.retriever = ContextRetriever(
        cross_encoder=config.cross_encoder,
    )

    app.state.processor = QuestionProcessor(retriever=app.state.retriever)

    yield

    logger.info("Shutting down...")
    app.state.executor.shutdown(wait=True)


# FastAPI app
app = FastAPI(title="HackRx RAG API", version="0.2.2", lifespan=lifespan)

Instrumentator().instrument(app).expose(app)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# API endpoints
@app.post("/hackrx/run", response_model=List[str])
async def process_document(
    request: DocumentRequest, _api_key: str = Depends(validate_api_key)
):
    try:
        # Step 1: Download document and calculate hash
        raw_content, file_ext = await app.state.downloader.download(request.documents)
        doc_hash = hashlib.sha256(raw_content.getvalue()).hexdigest()

        # Step 2: Check cache for processed (chunked) document
        cached_data = app.state.processing_cache.load(doc_hash)

        if cached_data:
            chunks, chunk_metadata = cached_data
            logger.info(
                f"Loaded {len(chunks)} chunks from cache for doc_hash: {doc_hash}"
            )
        else:
            # Step 2a: If not in cache, parse the document
            logger.info(f"Parsing document (hash: {doc_hash})")
            raw_text, _, _ = await app.state.parser.parse(raw_content, file_ext)

            # Step 2b: Chunk the raw text
            logger.info(f"Chunking document (hash: {doc_hash})")
            chunks, chunk_metadata = app.state.chunker.chunk(raw_text)

            # Step 2c: Save the results to the new processing cache
            app.state.processing_cache.save(doc_hash, chunks, chunk_metadata)

        # Step 3: Create BM25 retriever for this document
        bm25_retriever = BM25Retriever.from_texts(texts=chunks)

        # Step 4: Create/load vector store (this has its own separate cache)
        vector_store, _ = app.state.vector_mgr.create_vector_store(
            chunks=chunks,
            chunk_metadata=[cm.__dict__ for cm in chunk_metadata],
            doc_hash=doc_hash,
        )

        # Step 5: Process questions using the retrieved and chunked data
        results, _ = await app.state.processor.process_questions(
            questions=request.questions,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            top_k=app.state.config.TOP_K_CHUNKS,
            executor=app.state.executor,
            chunk_metadata=chunk_metadata,
        )

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed",
        )


@app.get("/health", response_model=HealthCheck)
async def health_check(request: Request):
    """System health status and component versions"""
    config = request.app.state.config
    return {
        "status": "healthy",
        "model_versions": {
            "embedding": config.EMBEDDING_MODEL,
            "reranker": config.RERANKER_MODEL,
            "llm": config.GEMINI_MODEL,
        },
        "cache_stats": request.app.state.vector_mgr.cache_stats(),
    }


@app.post("/cache/purge")
async def purge_cache(
    request: Request, documents: str, _api_key: str = Depends(validate_api_key)
):
    """Purge all cached data for a document."""
    try:
        content, _ = await request.app.state.downloader.download(documents)
        doc_hash = hashlib.sha256(content.getvalue()).hexdigest()

        # Purge from both the vector store and the processing cache
        request.app.state.vector_mgr.purge_document(doc_hash)
        request.app.state.processing_cache.purge(doc_hash)

        return {"status": "purged", "doc_hash": doc_hash}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_request, exc):
    logger.critical(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# main entry point
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        log_config=None,
        proxy_headers=True,
    )
