from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


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
    CACHE_DIR = Path("cache")

    @property
    def embeddings(self):
        if not hasattr(self, "_embeddings"):
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL,
                cache_folder=str(self.CACHE_DIR / "models" / self.EMBEDDING_MODEL),
            )
        return self._embeddings

    @property
    def cross_encoder(self):
        if not hasattr(self, "_cross_encoder"):
            self._cross_encoder = CrossEncoder(self.RERANKER_MODEL)
        return self._cross_encoder
