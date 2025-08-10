import fcntl
import logging
import os
import pickle
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Union

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from parsers import DocumentProcessingError

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, embeddings: HuggingFaceEmbeddings, cache_dir: Union[str, Path]):
        self.embeddings = embeddings
        self.cache_dir = Path(cache_dir)
        self.cache_version = "v3"
        self.lock_timeout = 15
        self.vector_db_dir = self.cache_dir / "vectordb"
        self._ensure_cache_dirs()

    def _ensure_cache_dirs(self):
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "temp").mkdir(parents=True, exist_ok=True)

    def create_vector_store(
        self,
        chunks: List[str],
        chunk_metadata: List[Dict],
        doc_hash: str,
    ) -> Tuple[FAISS, Dict]:
        cache_key = self._generate_cache_key(doc_hash, self.embeddings.model_name)
        cache_path = self.vector_db_dir / f"{cache_key}.faiss"
        metadata_path = self.vector_db_dir / f"{cache_key}.meta"

        # 1. Check for a valid cache entry first to avoid locking
        if self._is_cache_valid(cache_path, metadata_path):
            try:
                logger.info(f"Loading vector store from cache for doc_hash: {doc_hash}")
                return self._load_cached_store(cache_path, metadata_path)
            except Exception as e:
                logger.warning(
                    f"Failed to load a valid cache file, will rebuild. Error: {e}"
                )

        # 2. If no valid cache, build a new store within a lock
        logger.info(
            f"No valid cache found. Building new vector store for doc_hash: {doc_hash}"
        )
        with self._get_cache_lock(cache_path):
            if self._is_cache_valid(cache_path, metadata_path):
                return self._load_cached_store(cache_path, metadata_path)

            vector_store, metadata = self._build_vector_store(
                chunks, chunk_metadata, doc_hash
            )

            self._save_store_atomically(
                vector_store, metadata, cache_path, metadata_path
            )

        return vector_store, metadata

    def purge_document(self, doc_hash: str):
        cache_key = self._generate_cache_key(doc_hash, self.embeddings.model_name)
        logger.info(f"Purging cache for doc_hash: {doc_hash} (key: {cache_key})")

        base_path = self.vector_db_dir / cache_key
        paths_to_delete = [
            base_path.with_suffix(".faiss"),
            base_path.with_suffix(".pkl"),
            base_path.with_suffix(".meta"),
            base_path.with_suffix(".faiss.lock"),
        ]

        for path in paths_to_delete:
            try:
                if path.exists():
                    path.unlink()
                    logger.info(f"Deleted {path}")
            except OSError as e:
                logger.error(f"Error deleting cache file {path}: {e}")

    def cache_stats(self) -> Dict:
        try:
            files = list(self.vector_db_dir.glob("*.faiss"))
            total_size = sum(f.stat().st_size for f in self.vector_db_dir.glob("*"))
            return {
                "cached_items": len(files),
                "total_cache_size_mb": round(total_size / (1024 * 1024), 2),
            }
        except Exception as e:
            logger.error(f"Could not calculate cache stats: {e}")
            return {"cached_items": 0, "total_cache_size_mb": 0}

    def _build_vector_store(
        self, chunks: List[str], chunk_metadata: List[Dict], doc_hash: str
    ) -> Tuple[FAISS, Dict]:
        """construct a FAISS vector store."""
        logger.info(f"Building FAISS index for {len(chunks)} chunks...")

        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            metadatas=chunk_metadata,
        )

        index_metadata = {
            "doc_hash": doc_hash,
            "chunk_count": len(chunks),
            "embedding_model": self.embeddings.model_name,
            "cache_version": self.cache_version,
            "index_dimensions": vector_store.index.d,
        }
        return vector_store, index_metadata

    def _save_store_atomically(
        self, store: FAISS, metadata: Dict, cache_path: Path, metadata_path: Path
    ):
        temp_index_name = f"{cache_path.stem}_temp_{os.urandom(6).hex()}"
        
        temp_faiss_path = cache_path.parent / f"{temp_index_name}.faiss"
        temp_pkl_path = cache_path.parent / f"{temp_index_name}.pkl"
        meta_temp_path = metadata_path.with_suffix(".tmp")

        final_faiss_path = cache_path
        final_pkl_path = cache_path.with_suffix(".pkl")
        
        try:
            store.save_local(folder_path=str(cache_path.parent), index_name=temp_index_name)

            with open(meta_temp_path, "wb") as f:
                pickle.dump(metadata, f)

            os.rename(temp_faiss_path, final_faiss_path)
            os.rename(temp_pkl_path, final_pkl_path)
            os.rename(meta_temp_path, metadata_path)

            logger.info(f"Successfully saved new cache to {final_faiss_path}")

        except Exception as e:
            if temp_faiss_path.exists():
                temp_faiss_path.unlink()
            if temp_pkl_path.exists():
                temp_pkl_path.unlink()
            if meta_temp_path.exists():
                meta_temp_path.unlink()
            raise DocumentProcessingError(f"Failed to save cache files: {e}") from e

    def _load_cached_store(
        self, cache_path: Path, metadata_path: Path
    ) -> Tuple[FAISS, Dict]:
        with self._get_cache_lock(cache_path):
            vector_store = FAISS.load_local(
                folder_path=str(cache_path.parent),
                index_name=cache_path.stem,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        return vector_store, metadata

    def _is_cache_valid(self, cache_path: Path, metadata_path: Path) -> bool:
        """Checks if a cache entry is present and valid."""
        pkl_path = cache_path.with_suffix(".pkl")
        if not all([cache_path.exists(), metadata_path.exists(), pkl_path.exists()]):
            return False

        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            if metadata.get("cache_version") != self.cache_version:
                logger.info("Cache version mismatch. Invalidating cache.")
                return False
            if metadata.get("embedding_model") != self.embeddings.model_name:
                logger.info("Embedding model mismatch. Invalidating cache.")
                return False
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            logger.warning(f"Metadata file {metadata_path} is corrupt or invalid: {e}")
            return False

        return True

    def _generate_cache_key(self, doc_hash: str, model_name: str) -> str:
        model_name_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        return f"{doc_hash}_{self.cache_version}_{model_name_safe}"

    @contextmanager
    def _get_cache_lock(self, path: Path):
        lock_path = path.with_suffix(".lock")
        lock_file_pointer = None

        try:
            lock_file_pointer = open(lock_path, "w")
            fcntl.flock(lock_file_pointer, fcntl.LOCK_EX)
            yield
        finally:
            if lock_file_pointer:
                fcntl.flock(lock_file_pointer, fcntl.LOCK_UN)
                lock_file_pointer.close()
