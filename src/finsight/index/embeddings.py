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


NATIVE_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


def cached_embeddings(model: str = EMBEDDING_MODEL, dimensions: int | None = None) -> CacheBackedEmbeddings:
    """Embeddings keyed by text hash, so re-running never pays twice for the same text.

    `dimensions` shortens text-embedding-3 vectors (e.g. large at 1536); None or the model's native size
    means full vectors. The cache namespace includes a non-native size so different sizes never mix.
    `query_embedding_cache=True` caches query vectors too (useful for repeated eval runs).
    """
    if dimensions == NATIVE_DIMS.get(model):
        dimensions = None
    store = LocalFileStore(str(CACHE_DIR))
    kwargs = {"dimensions": dimensions} if dimensions else {}
    namespace = f"{model}-{dimensions}" if dimensions else model
    return CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(model=model, **kwargs), store, namespace=namespace, key_encoder="sha256",
        query_embedding_cache=True,
    )
