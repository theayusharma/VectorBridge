import asyncio
import logging
from concurrent.futures import Executor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from chunker import ChunkMetadata
from config import Config
from geminiloadbalance import generate_text_with_retry
from retriever import ContextRetriever

logger = logging.getLogger(__name__)


class QuestionProcessor:
    def __init__(self, retriever: ContextRetriever):
        self.retriever = retriever
        self.llm_client = generate_text_with_retry
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent LLM requests

    async def process_questions(
        self,
        questions: List[str],
        vector_store: FAISS,
        bm25_retriever: BM25Retriever,
        top_k: int,
        executor: Executor,
        chunk_metadata: Optional[List[ChunkMetadata]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        loop = asyncio.get_event_loop()
        tasks = []
        context_metadatas = []

        for question in questions:
            context, ctx_metadata = await loop.run_in_executor(
                executor,
                partial(
                    self.retriever.get_context,
                    question,
                    vector_store,
                    bm25_retriever,
                    top_k,
                    chunk_metadata,
                ),
            )
            context_metadatas.append(ctx_metadata)
            tasks.append(
                self._generate_answer_with_llm(question, context, ctx_metadata)
            )

        answers = await asyncio.gather(*tasks)
        return answers, context_metadatas

    async def _generate_answer_with_llm(
        self, question: str, context: str, context_metadata: Dict[str, Any]
    ) -> str:
        strategies_used = ", ".join(context_metadata.get("strategies_used", []))
        total_chunks = context_metadata.get("total_chunks", 0)

        prompt = f"""
Role: You are an expert subject-matter analyst with deep knowledge of interpreting and explaining complex documents. Your job is to provide a clear, accurate answer strictly based on the retrieved document excerpts.

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
        - Start with a clear answer
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
        )
