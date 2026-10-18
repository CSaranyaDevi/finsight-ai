"""OpenAI embeddings with LangChain's on-disk cache (index/embeddings_cache/, git-ignored)."""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from finsight.config import EMBEDDING_MODEL, INDEX_DIR

try:  # LangChain >= 1.0 moved these to langchain-classic
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain.storage import LocalFileStore
except ImportError:  # pragma: no cover
    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore

CACHE_DIR = INDEX_DIR / "embeddings_cache"


def cached_embeddings(model: str = EMBEDDING_MODEL) -> CacheBackedEmbeddings:
    """Embeddings keyed by text hash, so re-running never pays twice for the same text.

    `query_embedding_cache=True` caches query vectors too (useful for repeated eval runs).
    """
    store = LocalFileStore(str(CACHE_DIR))
    return CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(model=model), store, namespace=model, key_encoder="sha256",
        query_embedding_cache=True,
    )
