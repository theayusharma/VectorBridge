import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import en_core_web_md
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from parsers import DocumentProcessingError, DocumentStructure

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    def __init__(
        self,
        chunk_id: str,
        page_num: Optional[int] = None,
        section: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        chunk_type: str = "text",
        importance_score: float = 1.0,
        text: Optional[str] = None,
    ):
        self.chunk_id = chunk_id
        self.page_num = page_num
        self.section = section
        self.keywords = keywords or []
        self.chunk_type = chunk_type
        self.importance_score = importance_score
        self.text = text


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        semantic_threshold: float,
        embeddings: HuggingFaceEmbeddings,
        batch_size: int = 10  # For NLP processing
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold
        self.text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        self.nlp = en_core_web_md.load()
        self.batch_size = batch_size
        self._table_regex = re.compile(r"(--- Table \d+ on Page \d+ ---.*?)(?=\n\n---|$)")
        self._page_regex = re.compile(r"--- Page (\d+)")
        self._section_regex = re.compile(r"Section: ([^\n]+)")
        self._whitespace_regex = re.compile(r"\n{3,}|[ \t]{2,}")
        self._policy_keywords = {"exclusion", "coverage", "benefit", "claim", "policy"}

    def chunk(self, raw_text: str) -> Tuple[List[str], List[ChunkMetadata]]:
        if not raw_text or not isinstance(raw_text, str):
            raise DocumentProcessingError("Invalid input text")

        try:
            # Initial clean once
            cleaned_text = self._clean_text(raw_text)
            
            # Split first, then process chunks individually
            chunks = self.text_splitter.split_text(cleaned_text)
            if not chunks:
                raise DocumentProcessingError("No chunks generated from text")

            # Process metadata in batches for efficiency
            chunk_metadata = self._process_chunks_batch(chunks)
            
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks, chunk_metadata

        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            raise DocumentProcessingError(f"Document chunking failed: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """One-time text cleaning"""
        text = self._whitespace_regex.sub(" ", text).strip()
        return re.sub(r"(\d+\.\s+|\-\s+|\*\s+)", r"\n\1", text)

    def _process_chunks_batch(self, chunks: List[str]) -> List[ChunkMetadata]:
        """Process chunks in batches for efficiency"""
        metadata_list = []
        
        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start:batch_start + self.batch_size]
            
            # Process structural metadata first
            batch_metadata = [
                self._extract_structural_metadata(chunk, i + batch_start)
                for i, chunk in enumerate(batch)
            ]
            
            # Batch process for NLP keywords
            docs = list(self.nlp.pipe(
                [chunk[:1000] for chunk in batch],  # Limit text for keywords
                disable=["parser", "ner"],  # Only need POS tagging
                batch_size=self.batch_size
            ))
            
            # Combine results
            for i, (metadata, doc) in enumerate(zip(batch_metadata, docs)):
                metadata.keywords = self._extract_keywords_from_doc(doc)
                metadata_list.append(metadata)
                
        return metadata_list

    def _extract_structural_metadata(self, chunk: str, chunk_id: int) -> ChunkMetadata:
        """Extract non-NLP metadata from chunk"""
        page_num = None
        if page_match := self._page_regex.search(chunk):
            try:
                page_num = int(page_match.group(1))
            except (ValueError, IndexError):
                pass

        section = "General"
        if section_match := self._section_regex.search(chunk):
            section = section_match.group(1).strip()

        chunk_type = "text"
        importance = 1.0
        
        if "--- Table" in chunk:
            chunk_type = "table"
            importance = 1.5
        elif any(kw in chunk.lower() for kw in self._policy_keywords):
            chunk_type = "policy_rule"
            importance = 2.0

        return ChunkMetadata(
            chunk_id=f"chunk_{chunk_id}",
            page_num=page_num,
            section=section,
            chunk_type=chunk_type,
            importance_score=importance,
            text=chunk[:200]  # Store snippet for debugging
        )

    def _extract_keywords_from_doc(self, doc) -> List[str]:
        """Extract keywords from pre-processed Spacy doc"""
        seen = set()
        keywords = []
        
        for token in doc:
            if (token.pos_ in {"NOUN", "PROPN", "VERB"} and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                kw = token.text.lower()
                if kw not in seen:
                    seen.add(kw)
                    keywords.append(kw)
                    
        return keywords[:10]
