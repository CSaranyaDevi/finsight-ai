"""bm25s keyword index (one per chunking strategy) + a LangChain retriever wrapper.

Indexes each chunk's `bm25_text` (for tables: labels and figures without markdown).
Metadata filters (company, fiscal year, ...) use bm25s `weight_mask`, so BM25 and
Chroma search the same candidate set.
"""
from __future__ import annotations

import json
import re
from typing import Any

import bm25s
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from finsight.chunking.base import load_chunks
from finsight.chunking.pipeline import chunks_path
from finsight.config import INDEX_DIR
from finsight.index.vectorstore import chunk_metadata

BM25_DIR = INDEX_DIR / "bm25"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "which", "with", "what",
    "how", "did", "do", "does", "our", "we", "us",
}
# numbers like $1,234.5  (48,133)  12%  -3.2  |  words like fy2024, year-over-year, google's
_TOKEN = re.compile(r"\$?\(?-?\d[\d,]*(?:\.\d+)?%?\)?|[a-z][a-z0-9]*(?:['’-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase; numbers kept whole ('107,556') plus a comma-free form ('107556'); no stemming."""
    tokens: list[str] = []
    for t in _TOKEN.findall(text.lower()):
        if t[0].isalpha():
            if t not in _STOPWORDS:
                tokens.append(t)
            continue
        num = t.strip("$()%").lstrip("-")
        if not num or not num[0].isdigit():
            continue
        num = num.rstrip(".,")
        tokens.append(num)
        if "," in num:
            tokens.append(num.replace(",", ""))
    return tokens


def build_bm25(strategy: str) -> None:
    """Build and save the index plus the chunk ids / metadata that row i refers to."""
    chunks = load_chunks(chunks_path(strategy))
    model = bm25s.BM25()
    model.index([tokenize(c.bm25_text) for c in chunks], show_progress=False)
    out = BM25_DIR / strategy
    model.save(str(out), show_progress=False)
    (out / "chunks.json").write_text(
        json.dumps([chunk_metadata(c) for c in chunks], ensure_ascii=False), encoding="utf8"
    )


def load_bm25(strategy: str) -> tuple[bm25s.BM25, list[dict]]:
    path = BM25_DIR / strategy
    model = bm25s.BM25.load(str(path), show_progress=False)
    meta = json.loads((path / "chunks.json").read_text(encoding="utf8"))
    return model, meta


def matches(meta: dict, flt: dict | None) -> bool:
    return not flt or all(meta.get(k) == v for k, v in flt.items())


class BM25SRetriever(BaseRetriever):
    """LangChain retriever over a saved bm25s index; returns full chunk Documents from the docstore."""

    model: Any
    meta: list[dict]
    docstore: dict[str, Document]
    k: int = 30
    filter: dict | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []
        mask = None
        if self.filter:
            mask = np.array([matches(m, self.filter) for m in self.meta], dtype=float)
            if not mask.any():
                return []
        idx, scores = self.model.retrieve(
            [tokens], k=min(self.k, len(self.meta)), weight_mask=mask, show_progress=False
        )
        docs = []
        for i, score in zip(idx[0], scores[0]):
            if score <= 0:  # masked out or no overlapping terms
                continue
            doc = self.docstore[self.meta[i]["chunk_id"]]
            docs.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "bm25_score": float(score)}, id=doc.id))
        return docs
