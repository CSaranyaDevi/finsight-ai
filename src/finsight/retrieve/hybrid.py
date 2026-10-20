"""Dense, BM25 and hybrid retrieval over one chunking strategy, with metadata filters.

- dense:  Chroma similarity on `embed_text`, mapped to full chunks (LangChain MultiVectorRetriever)
- bm25:   bm25s on `bm25_text` (BM25SRetriever)
- hybrid: both, fused with reciprocal rank fusion (LangChain EnsembleRetriever), deduplicated on chunk_id
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document

from finsight.index.bm25 import BM25SRetriever, load_bm25
from finsight.index.vectorstore import load_docstore, open_readonly_vectorstore

try:  # LangChain >= 1.0 moved these to langchain-classic
    from langchain.retrievers import EnsembleRetriever, MultiVectorRetriever
    from langchain.storage import InMemoryStore
except ImportError:  # pragma: no cover
    from langchain_classic.retrievers import EnsembleRetriever, MultiVectorRetriever
    from langchain_classic.storage import InMemoryStore

MODES = ("dense", "bm25", "hybrid")


@dataclass
class RetrievalIndex:
    strategy: str
    vectorstore: object
    docstore: dict[str, Document]
    byte_docstore: InMemoryStore
    bm25_model: object
    bm25_meta: list[dict]


@lru_cache(maxsize=4)
def load_index(strategy: str) -> RetrievalIndex:
    """Open Chroma (a temporary copy) + bm25s + docstore for a strategy (cached; read-only)."""
    docstore = load_docstore(strategy)
    mem = InMemoryStore()
    mem.mset(list(docstore.items()))
    model, meta = load_bm25(strategy)
    return RetrievalIndex(strategy, open_readonly_vectorstore(strategy), docstore, mem, model, meta)


def chroma_filter(flt: dict | None) -> dict | None:
    """{'company': 'Amazon', 'fiscal_year': 2024} -> Chroma where-clause."""
    if not flt:
        return None
    clauses = [{k: {"$eq": v}} for k, v in flt.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def make_filter(company: str | None = None, fiscal_year: int | None = None, kind: str | None = None) -> dict | None:
    flt = {k: v for k, v in {"company": company, "fiscal_year": fiscal_year, "kind": kind}.items() if v is not None}
    return flt or None


def dense_retriever(idx: RetrievalIndex, k: int = 30, flt: dict | None = None) -> MultiVectorRetriever:
    search_kwargs = {"k": k}
    if flt:
        search_kwargs["filter"] = chroma_filter(flt)
    return MultiVectorRetriever(
        vectorstore=idx.vectorstore, docstore=idx.byte_docstore, id_key="chunk_id", search_kwargs=search_kwargs
    )


def bm25_retriever(idx: RetrievalIndex, k: int = 30, flt: dict | None = None) -> BM25SRetriever:
    return BM25SRetriever(model=idx.bm25_model, meta=idx.bm25_meta, docstore=idx.docstore, k=k, filter=flt)


def search(
    query: str,
    strategy: str = "fixed",
    mode: str = "hybrid",
    company: str | None = None,
    fiscal_year: int | None = None,
    kind: str | None = None,
    k: int = 10,
    candidates: int = 30,
    weights: tuple[float, float] = (0.5, 0.5),
) -> list[Document]:
    """Top-k chunks for `query`. `candidates` = how many each retriever contributes before fusion."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    idx = load_index(strategy)
    flt = make_filter(company, fiscal_year, kind)
    if mode == "dense":
        return dense_retriever(idx, k, flt).invoke(query)[:k]
    if mode == "bm25":
        return bm25_retriever(idx, k, flt).invoke(query)[:k]
    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever(idx, candidates, flt), bm25_retriever(idx, candidates, flt)],
        weights=list(weights),
        id_key="chunk_id",
    )
    return ensemble.invoke(query)[:k]
