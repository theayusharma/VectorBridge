import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from chunker import ChunkMetadata

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of chunked documents to avoid redundant parsing and chunking."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "processed_docs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ProcessingCacheManager at {self.cache_dir}")

    def _get_cache_path(self, doc_hash: str) -> Path:
        """Generates the file path for a given document hash."""
        return self.cache_dir / f"{doc_hash}.pkl"

    def load(self, doc_hash: str) -> Optional[Tuple[List[str], List[ChunkMetadata]]]:
        """
        Loads chunked document data from the cache.

        Returns:
            A tuple of (chunks, chunk_metadata) or None if not found.
        """
        cache_path = self._get_cache_path(doc_hash)
        if not cache_path.exists():
            logger.info(f"Processing cache miss for doc_hash: {doc_hash}")
            return None

        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            logger.info(f"Processing cache hit for doc_hash: {doc_hash}")
            return cached_data
        except (pickle.UnpicklingError, EOFError, IOError) as e:
            logger.warning(
                f"Could not load processing cache file {cache_path}. Error: {e}"
            )
            cache_path.unlink(missing_ok=True)
            return None

    def save(
        self, doc_hash: str, chunks: List[str], chunk_metadata: List[ChunkMetadata]
    ):
        """Saves chunked document data to the cache."""
        cache_path = self._get_cache_path(doc_hash)
        temp_path = cache_path.with_suffix(".tmp")

        try:
            with open(temp_path, "wb") as f:
                pickle.dump((chunks, chunk_metadata), f)
            temp_path.rename(cache_path)
            logger.info(f"Saved processed document to cache for doc_hash: {doc_hash}")
        except (pickle.PicklingError, IOError) as e:
            logger.error(f"Failed to save processed cache for {doc_hash}: {e}")
            temp_path.unlink(missing_ok=True)

    def purge(self, doc_hash: str):
        """Removes a document's processed data from the cache."""
        cache_path = self._get_cache_path(doc_hash)
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info(f"Purged processed cache for doc_hash: {doc_hash}")
            except OSError as e:
                logger.error(f"Error deleting processed cache file {cache_path}: {e}")
