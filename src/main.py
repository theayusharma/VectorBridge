import asyncio
import email
import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import docx
import google.generativeai as genai
import httpx
import pdfplumber
import textract
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field, HttpUrl

from geminiloadbalance import get_next_api_key

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#no 
# load_dotenv()


class Config:
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    REQUEST_TIMEOUT = 30.0
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_CHUNKS = 3
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_PATH = "./models/all-MiniLM-L6-v2"
    GEMINI_MODEL = "gemini-1.5-flash"

API_HIT_COUNT = 0
API_LOG_FILE = "api_logs.txt"

def load_api_hit_count():
    global API_HIT_COUNT
    try:
        with open(API_LOG_FILE, "r") as f:
            for line in f:
                if "API hit count" in line:
                    API_HIT_COUNT = int(line.split(":")[-1].strip())
                    logger.info(f"Loaded API hit count: {API_HIT_COUNT}")
    except FileNotFoundError:
        logger.info("api_logs.txt not found, starting hit count from 0")
        API_HIT_COUNT = 0

def save_api_hit_count():
    with open(API_LOG_FILE, "a") as f:
        f.write(f"API hit count: {API_HIT_COUNT}\n")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_api_hit_count()
    logger.info("Loading embedding model...")
    app.state.embeddings = SentenceTransformerEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        cache_folder=Config.EMBEDDING_MODEL_PATH
    )
    yield
    # Clean up resources if needed on shutdown
    logger.info("Shutting down...")

# FastAPI app
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
app = FastAPI(title="HackRx RAG API", version="0.2.0", lifespan=lifespan)

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



# custom exceptions
class DocumentProcessingError(Exception):
    pass

class LLMGenerationError(Exception):
    pass


# exception handlers
@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(
    request: Request, exc: DocumentProcessingError
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )

@app.exception_handler(LLMGenerationError)
async def llm_generation_exception_handler(request: Request, exc: LLMGenerationError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


# auth check
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing.",
        )

    parts = api_key_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Must be 'Bearer <key>'.",
        )

    token = parts[1]
    if token != os.getenv("HACKRX_AUTH_TOKEN"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token",
        )
    return token


# Data models
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

    chunk_size: Optional[int] = Field(
        default=Config.CHUNK_SIZE,
        ge=100,
        le=2000,
        description="Size of text chunks for processing",
    )
    chunk_overlap: Optional[int] = Field(
        default=Config.CHUNK_OVERLAP,
        ge=0,
        le=500,
        description="Overlap between text chunks",
    )
    top_k: Optional[int] = Field(
        default=Config.TOP_K_CHUNKS,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve",
    )

class HackRxResponse(BaseModel):
    answers: List[str]
    processing_time: float
    document_metadata: Optional[Dict[str, Any]]


# Parsers
def parse_pdf(content: io.BytesIO) -> Tuple[str, Dict[str, Any]]:
    full_text = []
    metadata = {"pages": 0, "tables": 0, "images": 0}

    try:
        with pdfplumber.open(content) as pdf:
            metadata["pages"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=2) or ""
                full_text.append(f"--- Page {i+1} ---\n{page_text}")

                tables = page.extract_tables()
                if tables:
                    metadata["tables"] += len(tables)
                    full_text.append(f"\n--- Tables on Page {i+1} ---\n")
                    for table in tables:
                        markdown_table = "\n".join(
                            [
                                "| " + " | ".join(map(str, row)) + " |"
                                for row in table
                                if row
                            ]
                        )
                        full_text.append(markdown_table + "\n")

                if page.images:
                    metadata["images"] += len(page.images)
    except Exception as e:
        raise DocumentProcessingError(f"Failed to parse PDF: {str(e)}")

    return "\n".join(full_text), metadata


def parse_docx(content: io.BytesIO) -> Tuple[str, Dict[str, Any]]:
    try:
        doc = docx.Document(content)
        metadata = {
            "paragraphs": len(doc.paragraphs),
            "tables": len(doc.tables),
            "sections": len(doc.sections),
        }
        return "\n".join([para.text for para in doc.paragraphs if para.text]), metadata
    except Exception as e:
        raise DocumentProcessingError(f"Failed to parse DOCX: {str(e)}")


def parse_doc(content: io.BytesIO) -> Tuple[str, Dict[str, Any]]:
    temp_file_path = "temp_file.doc"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(content.read())

        text = textract.process(temp_file_path).decode("utf-8")
        return text, {"format": "DOC"}
    except Exception as e:
        raise DocumentProcessingError(f"Failed to parse DOC: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def parse_eml(content: io.BytesIO) -> Tuple[str, Dict[str, Any]]:
    try:
        msg = email.message_from_bytes(content.read())
        metadata = {
            "subject": msg["subject"],
            "from": msg["from"],
            "to": msg["to"],
            "date": msg["date"],
            "content_type": msg.get_content_type(),
        }

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        return (
            f"Subject: {msg['subject']}\nFrom: {msg['from']}\nTo: {msg['to']}\n\n{body}",
            metadata,
        )
    except Exception as e:
        raise DocumentProcessingError(f"Failed to parse EML: {str(e)}")


def parse_txt(content: io.BytesIO) -> Tuple[str, Dict[str, Any]]:
    try:
        text = content.read().decode("utf-8", errors="ignore")
        return text, {"size": len(text)}
    except Exception as e:
        raise DocumentProcessingError(f"Failed to parse TXT: {str(e)}")


# parser mapping
PARSER_MAPPING: Dict[str, callable] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".doc": parse_doc,
    ".eml": parse_eml,
    ".txt": parse_txt,
}


# doc downloader
async def get_document_content_from_url(url: Union[str, HttpUrl]) -> Tuple[io.BytesIO, str]:
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
                    detail=f"Document exceeds maximum size of {Config.MAX_FILE_SIZE/1024/1024}MB"
                )
            
            response = await client.get(url_str, follow_redirects=True)
            response.raise_for_status()
            
            file_ext = Path(urlparse(url_str).path).suffix.lower()
            if not file_ext or file_ext not in PARSER_MAPPING:
                file_ext = ".txt"
            
            content = io.BytesIO(await response.aread())
            if content.getbuffer().nbytes > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Document exceeds maximum size of {Config.MAX_FILE_SIZE/1024/1024}MB"
                )
            
            return content, file_ext
    except httpx.RequestError as e:
        logger.error(f"Failed to download document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document: {str(e)}",
        )


# text processing
def clean_and_chunk_text(
    raw_text: str,
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
) -> List[str]:
    logger.info("Processing and chunking text")

    cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(cleaned_text)

    if not chunks:
        raise DocumentProcessingError(
            "Could not extract meaningful text chunks from document"
        )

    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


# vector store
def create_vector_store(chunks: List[str]) -> FAISS:
    if not chunks:
        raise ValueError("Cannot create vector store from empty text chunks")

    logger.info(f"Creating vector store with {len(chunks)} chunks")
    try:
        vector_store = FAISS.from_texts(texts=chunks, embedding=app.state.embeddings)
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        raise DocumentProcessingError("Failed to generate document embeddings")


# LLM
async def generate_answer_with_llm(question: str, context: str) -> str:
    genai.configure(api_key=get_next_api_key())

    prompt = f"""
    Role: You are an expert insurance policy analyst specialized in health insurance policies. Your task is to extract precise information from policy documents and answer questions with exact details including numbers, conditions, and limitations.

    Instructions:
    1. Answer ONLY using information from the provided policy context
    2. Be extremely precise with numbers, periods, and conditions
    3. Format answers as complete sentences mirroring the policy language
    4. Include all relevant conditions and limitations
    5. If the answer contains multiple points, present them as a single cohesive answer
    6. For definitions, quote the exact policy wording when possible
    7. For coverage questions, always specify:
       - Whether it's covered (Yes/No)
       - Any waiting periods
       - Specific conditions
       - Limitations or sub-limits

    Policy Context:
    ---
    {context}
    ---

    Question: {question}

    Answer Structure Guidelines:
    1. Begin with a direct answer to the question
    2. Include all numerical values exactly as in the policy (e.g., "thirty (30) days" not "30 days")
    3. List conditions as part of the sentence flow
    4. For coverage questions, use this pattern:
       "[Yes/No], [coverage details]. [Conditions]: [specific requirements]. [Limitations]: [any caps or exclusions]."

    Current Question: {question}

    Required Answer Format: A single, well-structured paragraph containing all relevant details from the policy document.
    """

    logger.info(f"Generating answer for question: {question[:50]}...")
    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        raise LLMGenerationError("Failed to generate answer from document")


# process questions in parallel
async def process_questions_parallel(
    questions: List[str], vector_store: FAISS, top_k: int
) -> List[str]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = []
        for question in questions:
            relevant_docs = await loop.run_in_executor(
                executor, partial(vector_store.similarity_search, question, k=top_k)
            )
            context = "\n---\n".join([doc.page_content for doc in relevant_docs])

            tasks.append(generate_answer_with_llm(question, context))

        return await asyncio.gather(*tasks)


# API endpoints
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Letsgoo, its up running........"}


@app.post(
    "/api/v1/hackrx/run",
    response_model=HackRxResponse,
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid input or document processing error"},
        401: {"description": "Unauthorized"},
        413: {"description": "Document too large"},
        422: {"description": "Unprocessable document"},
        503: {"description": "LLM service unavailable"},
    },
)
async def hackrx_run(
    req: HackRxRequest, api_key: str = Security(get_api_key)
) -> HackRxResponse:
    global API_HIT_COUNT
    API_HIT_COUNT += 1
    save_api_hit_count()

    start_time = asyncio.get_event_loop().time()

    try:
        content, file_ext = await get_document_content_from_url(req.documents)
        parser = PARSER_MAPPING.get(file_ext)
        if not parser:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File type '{file_ext}' not supported",
            )

        raw_text, metadata = await asyncio.get_event_loop().run_in_executor(
            None, parser, content
        )

        text_chunks = clean_and_chunk_text(
            raw_text, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap
        )

        vector_store = await asyncio.get_event_loop().run_in_executor(
            None, create_vector_store, text_chunks
        )

        answers = await process_questions_parallel(
            req.questions, vector_store, req.top_k
        )

        processing_time = asyncio.get_event_loop().time() - start_time

        return HackRxResponse(
            answers=answers,
            processing_time=round(processing_time, 2),
            document_metadata=metadata,
        )

    except HTTPException:
        raise
    except DocumentProcessingError as e:
        raise
    except LLMGenerationError as e:
        raise
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
        reload=True if os.getenv("ENVIRONMENT") == "development" else False
    )