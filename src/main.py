import asyncio
import hashlib
import io
import json
import logging
import joblib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, cast, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from keybert import KeyBERT
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, HttpUrl
from sentence_transformers import CrossEncoder
import en_core_web_md

from geminiloadbalance import LLMGenerationError, generate_text_with_retry
from parsers import (
    PARSER_MAPPING,
    DocumentProcessingError,
    DocumentStructure,
    ChunkMetadata,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

HACKRX_AUTH_TOKEN = os.getenv("HACKRX_AUTH_TOKEN")


class Config:
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    REQUEST_TIMEOUT = 30.0
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    TOP_K_CHUNKS = 12
    RERANK_TOP_K = 8
    BM25_TOP_K = 8
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    GEMINI_MODEL = "gemini-2.5-flash"
    MAX_LLM_RETRIES = 3

    # context parameters
    CONTEXT_EXPANSION_RATIO = 1.5  # expand context by 50%
    SEMANTIC_SIMILARITY_THRESHOLD = 0.3
    KEYWORD_BOOST_FACTOR = 1.2

    # cache
    EMBEDDING_MODEL_PATH = f"./models/{EMBEDDING_MODEL}"
    CACHE_DIR = Path("cache")
    DOCUMENT_CACHE_DIR = CACHE_DIR / "document"
    VECTORDB_CACHE_DIR = CACHE_DIR / "vectordb"
    METADATA_CACHE_DIR = CACHE_DIR / "metadata"
    BM25_CACHE_DIR = CACHE_DIR / "bm25"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model...")
    app.state.embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL, cache_folder=Config.EMBEDDING_MODEL_PATH
    )
    logger.info("Loading reranker model...")
    app.state.reranker = CrossEncoder(Config.RERANKER_MODEL, device="cpu")

    app.state.executor = ThreadPoolExecutor(max_workers=4)

    Config.DOCUMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Config.VECTORDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Config.METADATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Config.BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down...")
    app.state.executor.shutdown(wait=False)


try:
    nlp = en_core_web_md.load()
except Exception as e:
    logger.warning(
        f"Failed to load spaCy model: {e}. Falling back to basic keyword extraction."
    )
    nlp = None

# FastAPI app
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
app = FastAPI(title="HackRx RAG API", version="0.1.1", lifespan=lifespan)

if os.getenv("ENVIRONMENT") == "production":
    from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

    app.add_middleware(HTTPSRedirectMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# exception handlers
@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(
    _request: Request, exc: DocumentProcessingError
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


@app.exception_handler(LLMGenerationError)
async def llm_generation_exception_handler(_request: Request, exc: LLMGenerationError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


# auth check
async def get_api_key(api_key_header: str = Security(api_key_header)):
    parts = api_key_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    token = parts[1]
    if token != HACKRX_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return token


# Data models
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]


class HackRxResponse(BaseModel):
    answers: List[str]
    processing_time: float
    document_metadata: Optional[Dict[str, Any]]
    context_metadata: Optional[Dict[str, Any]]


# keyword extraction
def extract_keywords(text: str, use_spacy: bool = True) -> List[str]:
    if use_spacy and nlp:
        doc = nlp(text)
        keywords: List[str] = [
            token.text.lower()
            for token in doc
            if token.pos_ in ["NOUN", "PROPN", "VERB"]
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 2
        ]
    else:
        kw_model = KeyBERT()
        keywords = cast(
            List[str], [kw[0] for kw in kw_model.extract_keywords(text, top_n=10)]
        )

    seen = set()
    return [k for k in keywords if not (k in seen or seen.add(k))]


# doc downloader
async def get_document_content_from_url(
    url: Union[str, HttpUrl],
) -> Tuple[io.BytesIO, str]:
    logger.info(f"Downloading document from {url}")

    url_str = str(url) if isinstance(url, HttpUrl) else url

    try:
        async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
            head_response = await client.head(url_str, follow_redirects=True)
            head_response.raise_for_status()

            content_length = head_response.headers.get("content-length")
            if content_length and int(content_length) > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Document exceeds maximum size of {Config.MAX_FILE_SIZE/1024/1024}MB",
                )

            # stream download
            buffer = io.BytesIO()
            async with client.stream("GET", url_str, follow_redirects=True) as response:
                response.raise_for_status()

                total_bytes = 0
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    total_bytes += len(chunk)
                    if total_bytes > Config.MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Document exceeds maximum size of {Config.MAX_FILE_SIZE/1024/1024}MB",
                        )
                    buffer.write(chunk)

            # determine extension
            file_ext = Path(urlparse(url_str).path).suffix.lower()
            if not file_ext or file_ext not in PARSER_MAPPING:
                file_ext = ".txt"

            buffer.seek(0)
            return buffer, file_ext

    except httpx.RequestError as e:
        logger.error(f"Failed to download document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document: {str(e)}",
        )


# text processing with intelligent chunking
def clean_and_chunk_text(
    raw_text: str,
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
) -> Tuple[List[str], List[ChunkMetadata]]:
    logger.info("Processing and chunking of text")

    # text cleaning
    cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)

    # Preserve sentence boundaries after punctuation
    cleaned_text = re.sub(r"(\d+\.\s+|\-\s+|\*\s+)", r"\n\1", cleaned_text)

    # Protect table markdown structure by temporarily replacing pipes in tables
    table_pattern = r"(--- Table \d+ on Page \d+ ---.*?)(?=\n\n---|$)"
    tables = re.findall(table_pattern, cleaned_text, re.DOTALL)
    table_placeholders = [f"__TABLE_{i}__" for i in range(len(tables))]

    for i, table in enumerate(tables):
        cleaned_text = cleaned_text.replace(table, table_placeholders[i])

    # text splitter with custom separators for semantic chunking
    text_splitter = SemanticChunker(
        embeddings=app.state.embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    chunks = text_splitter.split_text(raw_text)

    # Restore tables in chunks
    for i, placeholder in enumerate(table_placeholders):
        chunks = [chunk.replace(placeholder, tables[i]) for chunk in chunks]

    if not chunks:
        raise DocumentProcessingError(
            "Could not extract meaningful text chunks from document"
        )

    # metadata for each chunk
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        # page number
        page_match = re.search(r"--- Page (\d+)", chunk)
        page_num = int(page_match.group(1)) if page_match else None

        # section
        section_match = re.search(r"Section: ([^\n]+)", chunk)
        section = section_match.group(1).strip() if section_match else "General"

        # keywords
        keywords = extract_keywords(chunk, use_spacy=nlp is not None)

        # determine chunk type
        chunk_type = "text"
        if "--- Table" in chunk:
            chunk_type = "table"
        elif any(
            keyword in chunk.lower()
            for keyword in ["exclusion", "coverage", "benefit", "claim", "policy"]
        ):
            chunk_type = "policy_rule"

        # importance score
        importance_score = 1.0
        if chunk_type == "policy_rule":
            importance_score = 2.0
        if chunk_type == "table":
            importance_score = 1.5

        metadata = ChunkMetadata(
            chunk_id=f"chunk_{i}",
            page_num=page_num,
            section=section,
            keywords=keywords,
            chunk_type=chunk_type,
            importance_score=importance_score,
        )
        chunk_metadata.append(metadata)

    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks, chunk_metadata


# caching
def get_document_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save_metadata(
    doc_hash: str,
    metadata: Dict[str, Any],
    doc_structure: Optional[DocumentStructure] = None,
    chunk_metadata: Optional[List[ChunkMetadata]] = None,
):
    cache_data = {
        "metadata": metadata,
        "doc_structure": doc_structure.__dict__ if doc_structure else None,
        "chunk_metadata": (
            [
                {
                    "chunk_id": cm.chunk_id,
                    "page_num": cm.page_num,
                    "section": cm.section,
                    "keywords": cm.keywords,
                    "chunk_type": cm.chunk_type,
                    "importance_score": cm.importance_score,
                }
                for cm in chunk_metadata
            ]
            if chunk_metadata
            else None
        ),
    }

    with open(Config.METADATA_CACHE_DIR / f"{doc_hash}.json", "w") as f:
        json.dump(cache_data, f, indent=2)


def load_metadata(
    doc_hash: str,
) -> Tuple[Dict[str, Any], Optional[DocumentStructure], Optional[List[ChunkMetadata]]]:
    try:
        with open(Config.METADATA_CACHE_DIR / f"{doc_hash}.json", "r") as f:
            cache_data = json.load(f)

        metadata = cache_data.get("metadata", {})

        doc_structure = None
        if cache_data.get("doc_structure"):
            doc_structure = DocumentStructure()
            doc_structure.__dict__.update(cache_data["doc_structure"])

        chunk_metadata = None
        if cache_data.get("chunk_metadata"):
            chunk_metadata = []
            for cm_data in cache_data["chunk_metadata"]:
                cm = ChunkMetadata(
                    chunk_id=cm_data["chunk_id"],
                    page_num=cm_data.get("page_num"),
                    section=cm_data.get("section"),
                    keywords=cm_data.get("keywords", []),
                    chunk_type=cm_data.get("chunk_type", "text"),
                    importance_score=cm_data.get("importance_score", 1.0),
                )
                chunk_metadata.append(cm)

        return metadata, doc_structure, chunk_metadata
    except (FileNotFoundError, json.JSONDecodeError):
        return {}, None, None


def save_bm25_retriever(bm25_retriever: BM25Retriever, doc_hash: str) -> None:
    try:
        bm25_cache_path = Config.BM25_CACHE_DIR / f"{doc_hash}.bm25"
        joblib.dump(bm25_retriever, bm25_cache_path)
        logger.info(f"Saved BM25 retriever to cache: {bm25_cache_path}")
    except Exception as e:
        logger.error(f"Failed to save BM25 retriever: {str(e)}")
        raise DocumentProcessingError(f"Failed to save BM25 retriever: {str(e)}")


def load_bm25_retriever(doc_hash: str) -> Optional[BM25Retriever]:
    try:
        bm25_cache_path = Config.BM25_CACHE_DIR / f"{doc_hash}.bm25"
        if bm25_cache_path.exists():
            bm25_retriever = joblib.load(bm25_cache_path)
            logger.info(f"Loaded BM25 retriever from cache: {bm25_cache_path}")
            return bm25_retriever
        return None
    except Exception as e:
        logger.warning(f"Failed to load BM25 retriever: {str(e)}")
        return None


# vector store creation
def create_vector_store(
    chunks: List[str],
    embeddings: HuggingFaceEmbeddings,
    chunk_metadata: Optional[List[ChunkMetadata]] = None,
) -> FAISS:
    if not chunks:
        raise ValueError("Cannot create vector store from empty text chunks")

    logger.info(f"Creating vector store with {len(chunks)} chunks")
    try:
        # metadata for FAISS
        metadatas = []
        for i, _ in enumerate(chunks):
            metadata = {"chunk_index": i}
            if chunk_metadata and i < len(chunk_metadata):
                cm = chunk_metadata[i]
                cm_data = {
                    "chunk_id": cm.chunk_id,
                    "page_num": cm.page_num,
                    "section": cm.section,
                    "keywords": cm.keywords,
                    "chunk_type": cm.chunk_type,
                    "importance_score": cm.importance_score,
                }
                metadata.update(cm_data)
            metadatas.append(metadata)

        vector_store = FAISS.from_texts(
            texts=chunks, embedding=embeddings, metadatas=metadatas
        )
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        raise DocumentProcessingError("Failed to generate document embeddings")


# context retrieval
def get_context(
    question: str,
    vector_store: FAISS,
    bm25_retriever: BM25Retriever,
    top_k: int,
    chunk_metadata: Optional[List[ChunkMetadata]] = None,
) -> Tuple[str, Dict[str, Any]]:
    try:
        # Vector search (FAISS)
        primary_docs = vector_store.similarity_search(question, k=top_k)
        if not primary_docs:
            logger.warning("No documents found in vector search.")

        # BM25 search
        bm25_retriever.k = Config.BM25_TOP_K
        bm25_docs = bm25_retriever.get_relevant_documents(question)

        # Combine and deduplicate
        combined_docs = primary_docs + [
            doc
            for doc in bm25_docs
            if doc.page_content not in {d.page_content for d in primary_docs}
        ]
        if not combined_docs:
            return "", {
                "primary_chunks": 0,
                "bm25_chunks": 0,
                "strategies_used": ["vector_similarity", "bm25"],
                "total_chunks": 0,
            }

        # Reranking
        query_chunk_pairs = [(question, doc.page_content) for doc in combined_docs]
        reranker_scores = app.state.reranker.predict(query_chunk_pairs)

        # Combine scores
        selected_chunks = []
        for doc, reranker_score in zip(combined_docs, reranker_scores):
            importance_score = doc.metadata.get("importance_score", 1.0)
            combined_score = (
                reranker_score * 0.6
                + importance_score * 0.3
                + (0.1 if doc in bm25_docs else 0.0)
            )
            selected_chunks.append((doc.page_content, combined_score, doc.metadata))

        # Keyword-based boost
        seen_chunks = set(chunk[0] for chunk in selected_chunks)
        context_metadata = {
            "primary_chunks": len(primary_docs),
            "bm25_chunks": len(bm25_docs),
            "strategies_used": ["vector_similarity", "bm25", "reranking"],
            "keyword_matches": 0,
            "expanded_chunks": 0,
            "total_chunks": 0,
        }
        if chunk_metadata:
            question_keywords = extract_keywords(
                question.lower(), use_spacy=nlp is not None
            )
            keyword_matches = []
            for cm in chunk_metadata:
                text = cm.text if hasattr(cm, "text") else None
                if not text:
                    continue
                text_keywords = [kw.lower() for kw in cm.keywords]
                overlap = set(text_keywords).intersection(set(question_keywords))
                if overlap and text not in seen_chunks:
                    boost_score = (
                        len(overlap) * Config.KEYWORD_BOOST_FACTOR * cm.importance_score
                    )
                    keyword_matches.append((text, boost_score, cm))
            keyword_matches.sort(key=lambda x: x[1], reverse=True)
            for text, score, cm in keyword_matches[:2]:
                selected_chunks.append((text, score, cm.__dict__))
                seen_chunks.add(text)
            context_metadata["keyword_matches"] = len(keyword_matches[:2])
            if keyword_matches:
                context_metadata["strategies_used"].append("keyword_matching")

        # Context expansion
        expanded_chunks = []
        primary_indices = [doc.metadata.get("chunk_index", -1) for doc in combined_docs]
        for idx in primary_indices:
            if idx == -1:
                continue
            if idx > 0:
                try:
                    prev_doc = vector_store.similarity_search("", k=idx + 1)[idx - 1]
                    if prev_doc.page_content not in seen_chunks:
                        expanded_chunks.append(
                            (prev_doc.page_content, 0.8, prev_doc.metadata)
                        )
                        seen_chunks.add(prev_doc.page_content)
                except IndexError:
                    pass
            try:
                next_doc = vector_store.similarity_search("", k=idx + 2)[idx + 1]
                if next_doc.page_content not in seen_chunks:
                    expanded_chunks.append(
                        (next_doc.page_content, 0.8, next_doc.metadata)
                    )
                    seen_chunks.add(next_doc.page_content)
            except IndexError:
                pass
        selected_chunks.extend(expanded_chunks[:2])
        context_metadata["expanded_chunks"] = len(expanded_chunks[:2])
        if expanded_chunks:
            context_metadata["strategies_used"].append("context_expansion")

        # Sort by combined score and select top RERANK_TOP_K
        selected_chunks.sort(key=lambda x: x[1], reverse=True)
        selected_chunks = selected_chunks[: Config.RERANK_TOP_K]
        context_metadata["total_chunks"] = len(selected_chunks)

        # Combine chunks with metadata annotations
        context_parts = []
        for i, (chunk_text, score, metadata) in enumerate(selected_chunks):
            chunk_type = metadata.get("chunk_type", "text")
            page_num = metadata.get("page_num", "Unknown")
            section = metadata.get("section", "General")
            header = f"--- Context Chunk {i+1} (Score: {score:.2f}, Type: {chunk_type}, Page: {page_num}, Section: {section}) ---"
            context_parts.append(f"{header}\n{chunk_text}")

        final_context = "\n\n".join(context_parts)
        return final_context, context_metadata

    except Exception as e:
        logger.error(f"Error in get_context: {str(e)}")
        raise DocumentProcessingError(f"Failed to retrieve context: {str(e)}")


# LLM
async def generate_answer_with_llm(
    question: str, context: str, context_metadata: Dict[str, Any]
) -> str:
    strategies_used = ", ".join(context_metadata.get("strategies_used", []))
    total_chunks = context_metadata.get("total_chunks", 0)

    prompt = f"""
Role: You are an expert insurance claims adjudicator with deep knowledge of policy interpretation. Provide clear, accurate information based strictly on the policy document.

Context Analysis: 
    - Retrieved {total_chunks} relevant document sections using: {strategies_used}
    - Each section is scored by relevance and importance

Instructions:
    Communication Style:
        - Write in a single, clear paragraph that a customer can easily understand
        - Use exact policy wording when referencing specific terms, amounts, or time periods
        - Maintain original formatting of numbers and periods (e.g., "thirty (30) days")
        - Be conversational but precise - avoid bullet points or structured lists

    Content Requirements:
        - Start with a clear answer: "Yes, this is covered" / "No, this is not covered" / "This is partially covered" / "Based on the available information..."
        - Include the specific policy section reference (e.g., "According to Section 2.21...")
        - Mention any coverage amounts, limits, or time periods
        - Explain any conditions that must be met
        - Note any exclusions or limitations that apply
        - If information is incomplete, clearly state what additional details would be needed

    Quality Standards:
        - Never extrapolate beyond the provided policy text
        - Use policy-defined terms consistently
        - If there are contradictions or ambiguities, acknowledge them
        - Be honest about the completeness of the information available

Policy Context (Analyzed from {total_chunks} relevant sections):
{context}

Question: {question}

Required Output: Write a single, comprehensive paragraph that directly answers the customer's question with all relevant policy details, conditions, and limitations in a natural, flowing narrative format.
    """

    logger.info(f"Generating answer for question: {question}...")
    return await generate_text_with_retry(
        prompt=prompt,
        model_name=Config.GEMINI_MODEL,
        # max_retries=Config.MAX_LLM_RETRIES,
    )


# process questions in parallel
async def process_questions(
    questions: List[str],
    vector_store: FAISS,
    bm25_retriever: BM25Retriever,
    top_k: int,
    chunk_metadata: Optional[List[ChunkMetadata]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    loop = asyncio.get_event_loop()
    tasks = []
    context_metadatas = []

    for question in questions:
        context, ctx_metadata = await loop.run_in_executor(
            app.state.executor,
            partial(
                get_context,
                question,
                vector_store,
                bm25_retriever,
                top_k,
                chunk_metadata,
            ),
        )
        context_metadatas.append(ctx_metadata)
        tasks.append(generate_answer_with_llm(question, context, ctx_metadata))

    answers = await asyncio.gather(*tasks)
    return answers, context_metadatas


# API endpoints
@app.get("/")
async def root():
    return {"message": "Letsgoo, its up running........", "version": "0.1.1"}


@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    req: HackRxRequest, _api_key: str = Security(get_api_key)
) -> HackRxResponse:
    start_time = asyncio.get_event_loop().time()

    try:
        content_bytes, file_ext = await get_document_content_from_url(req.documents)
        doc_hash = get_document_hash(content_bytes.getvalue())
        vector_store_path = Config.VECTORDB_CACHE_DIR / f"{doc_hash}.faiss"

        # Cache check
        vector_store = None
        bm25_retriever = None
        if vector_store_path.exists():
            logger.info(f"Loading vector store from cache: {vector_store_path}")
            vector_store = FAISS.load_local(
                folder_path=str(Config.VECTORDB_CACHE_DIR),
                index_name=doc_hash,
                embeddings=app.state.embeddings,
                # allow_dangerous_deserialization=True,
            )
            bm25_retriever = load_bm25_retriever(doc_hash)
            metadata, doc_structure, chunk_metadata = load_metadata(doc_hash)
            if not metadata:
                metadata = {"source": "cache"}
        else:
            logger.info("Processing new document.")
            parser = PARSER_MAPPING.get(file_ext)
            if not parser:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"File type '{file_ext}' not supported",
                )

            (
                raw_text,
                metadata,
                doc_structure,
            ) = await asyncio.get_event_loop().run_in_executor(
                app.state.executor, parser.parse, io.BytesIO(content_bytes.getvalue())
            )

            text_chunks, chunk_metadata = clean_and_chunk_text(
                raw_text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP
            )

            vector_store = await asyncio.get_event_loop().run_in_executor(
                app.state.executor,
                create_vector_store,
                text_chunks,
                app.state.embeddings,
                chunk_metadata,
            )

            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_texts(
                text_chunks,
                metadatas=[
                    {
                        "chunk_index": i,
                        "chunk_id": cm.chunk_id,
                        "page_num": cm.page_num,
                        "section": cm.section,
                        "keywords": cm.keywords,
                        "chunk_type": cm.chunk_type,
                        "importance_score": cm.importance_score,
                    }
                    for i, cm in enumerate(chunk_metadata)
                ],
            )

            # Save to cache
            vector_store.save_local(
                folder_path=str(Config.VECTORDB_CACHE_DIR), index_name=doc_hash
            )
            save_bm25_retriever(bm25_retriever, doc_hash)
            save_metadata(doc_hash, metadata, doc_structure, chunk_metadata)

            with open(Config.DOCUMENT_CACHE_DIR / doc_hash, "wb") as f:
                f.write(content_bytes.getvalue())
            logger.info(
                f"Saved new document, vector store, and BM25 retriever to cache with hash: {doc_hash}"
            )

        if not vector_store or not bm25_retriever:
            raise DocumentProcessingError(
                "Failed to load or create vector store or BM25 retriever"
            )

        # Process questions with context retrieval
        answers, context_metadatas = await process_questions(
            req.questions,
            vector_store,
            bm25_retriever,  # Pass BM25 retriever
            Config.TOP_K_CHUNKS,
            chunk_metadata,
        )

        # Aggregate context metadata
        aggregated_context_metadata = {
            "total_questions": len(req.questions),
            "average_chunks_per_question": (
                sum(cm.get("total_chunks", 0) for cm in context_metadatas)
                / len(context_metadatas)
                if context_metadatas
                else 0
            ),
            "strategies_summary": {
                strategy: sum(
                    1
                    for cm in context_metadatas
                    if strategy in cm.get("strategies_used", [])
                )
                for strategy in [
                    "vector_similarity",
                    "bm25",
                    "reranking",
                    "keyword_matching",
                    "context_expansion",
                ]
            },
        }

        processing_time = asyncio.get_event_loop().time() - start_time

        # Metadata for response
        response_metadata = metadata.copy()
        if doc_structure:
            doc_struct_data = {
                "document_structure": {
                    "sections_found": len(doc_structure.sections),
                    "headers_found": len(doc_structure.headers),
                    "tables_found": len(doc_structure.tables),
                    "key_value_pairs": len(doc_structure.key_value_pairs),
                }
            }
            response_metadata.update(doc_struct_data)  # type:ignore

        if chunk_metadata:
            chunk_types = {}
            for cm in chunk_metadata:
                chunk_types[cm.chunk_type] = chunk_types.get(cm.chunk_type, 0) + 1
            response_metadata["chunk_analysis"] = {  # type:ignore
                "total_chunks": len(chunk_metadata),
                "chunk_types": chunk_types,
                "avg_importance_score": (
                    sum(cm.importance_score for cm in chunk_metadata)
                    / len(chunk_metadata)
                    if chunk_metadata
                    else 0
                ),
            }

        return HackRxResponse(
            answers=answers,
            processing_time=round(processing_time, 2),
            document_metadata=response_metadata,
            context_metadata=aggregated_context_metadata,
        )

    except (HTTPException, DocumentProcessingError, LLMGenerationError) as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_config=None,
    )
