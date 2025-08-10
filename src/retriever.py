import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import en_core_web_md
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from chunker import ChunkMetadata
from config import Config
from parsers import DocumentProcessingError

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    retrieval_score: float = 0.0
    retrieval_strategy: str = "unknown"

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        return isinstance(other, RetrievedChunk) and self.text == other.text


class ContextRetriever:
    def __init__(
        self,
        cross_encoder: CrossEncoder,
    ):
        self.reranker = cross_encoder
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.nlp = en_core_web_md.load()

        # Retrieval parameters
        self.reranker_weight = 0.6
        self.retrieval_weight = 0.4
        self.score_threshold = 0.15
        self.max_rerank_candidates = 50
        self.keyword_boost_factor = Config.KEYWORD_BOOST_FACTOR

    def get_context(
        self,
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
            reranker_scores = self.reranker.predict(query_chunk_pairs)

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
                question_keywords = self._extract_keywords(question.lower())
                keyword_matches = []
                for cm in chunk_metadata:
                    text = cm.text if hasattr(cm, "text") else None
                    if not text:
                        continue
                    text_keywords = [kw.lower() for kw in cm.keywords]
                    overlap = set(text_keywords).intersection(set(question_keywords))
                    if overlap and text not in seen_chunks:
                        boost_score = (
                            len(overlap)
                            * Config.KEYWORD_BOOST_FACTOR
                            * cm.importance_score
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
            primary_indices = [
                doc.metadata.get("chunk_index", -1) for doc in combined_docs
            ]
            for idx in primary_indices:
                if idx == -1:
                    continue
                if idx > 0:
                    try:
                        prev_doc = vector_store.similarity_search("", k=idx + 1)[
                            idx - 1
                        ]
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

    def _extract_keywords(self, text: str) -> List[str]:
        doc = self.nlp(text)
        keywords: List[str] = [
            token.text.lower()
            for token in doc
            if token.pos_ in ["NOUN", "PROPN", "VERB"]
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 2
        ]

        seen = set()
        return [k for k in keywords if not (k in seen or seen.add(k))]
