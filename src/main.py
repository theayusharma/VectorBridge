import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


import email
from urllib.parse import urlparse

import docx
import google.generativeai as genai
import httpx
import pdfplumber
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
import textract

API_KEY = os.getenv("HACKRX_AUTH_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("missing env var: GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# FastAPI app
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
app = FastAPI(title="HackRx RAG by LetsGoo...", version="0.1.0")


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
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token",
        )
    return token


# data models
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]


# parsers
def parse_pdf(content: io.BytesIO) -> str:
    full_text = []
    with pdfplumber.open(content) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text(x_tolerance=2) or ""
            full_text.append(f"--- Page {i+1} ---\n{page_text}")

            tables = page.extract_tables()
            if tables:
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
    return "\n".join(full_text)


def parse_docx(content: io.BytesIO) -> str:
    doc = docx.Document(content)
    return "\n".join([para.text for para in doc.paragraphs])


def parse_doc(content: io.BytesIO) -> str:
    temp_file_path = "temp_file.doc"
    with open(temp_file_path, "wb") as f:
        f.write(content.read())
    try:
        text = textract.process(temp_file_path).decode("utf-8")
    finally:
        os.remove(temp_file_path)
    return text


def parse_eml(content: io.BytesIO) -> str:
    msg = email.message_from_bytes(content.read())
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                body = part.get_payload(decode=True).decode(errors="ignore")
                break
    else:
        body = msg.get_payload(decode=True).decode(errors="ignore")
    return f"Subject: {msg['subject']}\nFrom: {msg['from']}\nTo: {msg['to']}\n\n{body}"


# parsing functions mapping
PARSER_MAPPING: Dict[str, callable] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".doc": parse_doc,
    ".eml": parse_eml,
    ".txt": lambda c: c.read().decode("utf-8", errors="ignore"),
}


# get document via url
async def get_document_content_from_url(url: str) -> (io.BytesIO, str):
    print(f"E1: Downloading document from {url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            file_ext = Path(urlparse(url).path).suffix.lower() or ".txt"
            return io.BytesIO(await response.aread()), file_ext
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download or access URL: {e}",
        )


# cleaning & chunking
def clean_and_chunk_text(raw_text: str) -> List[str]:
    print("E3: Starting text cleaning and chunking...")
    cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(cleaned_text)
    print(f"Text successfully cleaned and split into {len(chunks)} chunks.")
    return chunks


# put embeddings & store into db
def create_vector_store(chunks: List[str]) -> FAISS:
    if not chunks:
        raise ValueError(
            "Cannot create vector store from an empty list of text chunks."
        )

    embedding_model = "all-MiniLM-L6-v2"
    print(f"E4: Initializing sentence-transformer embedding model {embedding_model}...")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

    print("E5: Generating embeddings and creating FAISS vector store...")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    print("done: doc processed")
    return vector_store


def process_and_expand_query(
    question: str, embeddings: SentenceTransformerEmbeddings
) -> List[float]:
    # F2: Query Expansion (Simple placeholder version)
    # In a more advanced system, you would use an LLM to generate multiple variations.
    # For now, we will just use the original question as it's very effective.
    print(f"F2: Processing query: '{question}'")
    # expanded_queries = [question, f"explain in detail {question}"] # Example of simple expansion

    # F3: Generate Query Embedding
    # We use the same embedding model to embed the user's query.
    print("F3: Generating embedding for the query...")
    # LangChain's vector store search methods handle this automatically,
    # but we can do it explicitly to see the step.
    # The search function is more efficient, so we'll let it handle the embedding.
    # query_embedding = embeddings.embed_query(question)
    # print("Query embedding generated.")
    # For simplicity and efficiency, we will pass the raw text to the search function.
    return question


# semantic search
def retrieve_relevant_chunks(
    question: str, vector_store: FAISS, k: int = 3
) -> List[Any]:
    print(f"\nG/H: Retrieving top-{k} chunks for question: '{question}'")
    # The `similarity_search` method performs the following:
    # 1. Accepts the raw query string (our processed question).
    # 2. Implicitly performs step F3 (Generate Query Embedding) using the store's embedding function.
    # 3. Performs step G (Semantic Search) to find the most similar vectors in the FAISS index.
    # 4. Performs step H (Retrieve Top-K Chunks).
    relevant_docs = vector_store.similarity_search(question, k=k)
    print(f"Found {len(relevant_docs)} relevant document chunks.")
    return relevant_docs


async def generate_decision_with_llm(question: str, context: str) -> str:
    prompt = f"""
    You are a helpful assistant. Based *only* on the context provided below, answer the following question concisely.
    Do not use any external knowledge or make assumptions. If the answer is not found in the context,
    clearly state that "The answer could not be found in the provided document."

    **Context:**
    ---
    {context}
    ---

    **Question:** {question}

    **Answer:**
    """

    print("I1: Prompt constructed. Calling Gemini API...")

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = await model.generate_content_async(prompt)

        answer = response.text.strip()
        print("I3: Received and parsed response from Gemini.")
        return answer
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"An error occurred while communicating with the AI model. Please check the server logs."


# api endpoint
@app.get("/")
def hola():
    return "LetsGooooooo, its up ye.........."

@app.post(
    "/api/v1/hackrx/run",
    response_model=HackRxResponse,
)
async def hackrx_run(req: HackRxRequest, api_key: str = Security(get_api_key)):
    # doc processing
    content_bytes, file_ext = await get_document_content_from_url(req.documents)
    parser = PARSER_MAPPING.get(file_ext)
    if not parser:
        raise HTTPException(415, f"File type '{file_ext}' not supported.")
    raw_text = parser(content_bytes)
    text_chunks = clean_and_chunk_text(raw_text)
    if not text_chunks:
        raise HTTPException(400, "Could not extract text from document.")
    vector_store = create_vector_store(text_chunks)

    # query processing
    final_answers = []
    for question in req.questions:
        relevant_chunks = retrieve_relevant_chunks(question, vector_store)
        context = "\n---\n".join([doc.page_content for doc in relevant_chunks])

        if not context:
            answer_text = "Could not find any relevant information in the document to answer this question."
        else:
            answer_text = await generate_decision_with_llm(question, context)

        final_answers.append(answer_text)

    return HackRxResponse(answers=final_answers)


# server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
