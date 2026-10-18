"""Strategy B: semantic text chunks via LangChain's SemanticChunker.

SemanticChunker embeds every sentence and cuts where the embedding distance between
neighbouring sentences jumps (percentile breakpoint). It has no size limits, so we add:
- merge splits shorter than `min_tokens` into the following split
- re-split anything longer than `max_tokens` with the token splitter (no overlap)

Sentence embeddings go through a LangChain CacheBackedEmbeddings file cache, so
re-running never pays twice for the same sentence. This step calls the OpenAI API.
"""
from __future__ import annotations

from bisect import bisect_right

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from finsight.chunking.base import Block, Chunk, count_tokens, make_prefix
from finsight.chunking.fixed import make_splitter
from finsight.config import DocMeta, EMBEDDING_MODEL, INDEX_DIR

try:  # LangChain >= 1.0 moved these to langchain-classic
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain.storage import LocalFileStore
except ImportError:  # pragma: no cover
    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore

CACHE_DIR = INDEX_DIR / "embeddings_cache"


def cached_embeddings(model: str = EMBEDDING_MODEL) -> CacheBackedEmbeddings:
    """OpenAI embeddings with an on-disk cache keyed by text hash (committed with the index)."""
    store = LocalFileStore(str(CACHE_DIR))
    return CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(model=model), store, namespace=model, key_encoder="sha256"
    )


def make_semantic_splitter(embeddings=None, percentile: float = 95) -> SemanticChunker:
    return SemanticChunker(
        embeddings or cached_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile,
        add_start_index=True,
    )


def _sized(pieces: list[tuple[int, str]], min_tokens: int, max_tokens: int) -> list[tuple[int, str]]:
    """Apply min/max token limits to (start_offset, text) pieces from SemanticChunker."""
    merged: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    for start, text in pieces:
        if pending:
            start, text = pending[0], pending[1] + " " + text
            pending = None
        if count_tokens(text) < min_tokens:
            pending = (start, text)
        else:
            merged.append((start, text))
    if pending:  # trailing short piece: attach to the previous one, or keep alone
        if merged:
            s, t = merged.pop()
            merged.append((s, t + " " + pending[1]))
        else:
            merged.append(pending)

    splitter = make_splitter(max_tokens, 0)
    out: list[tuple[int, str]] = []
    for start, text in merged:
        if count_tokens(text) <= max_tokens:
            out.append((start, text))
        else:
            for d in splitter.create_documents([text]):
                out.append((start + max(0, d.metadata.get("start_index", 0)), d.page_content))
    return out


def chunk_text_semantic(
    meta: DocMeta,
    blocks: list[Block],
    splitter: SemanticChunker | None = None,
    min_tokens: int = 150,
    max_tokens: int = 800,
    skip_below: int = 25,
) -> list[Chunk]:
    """Split each block independently (never across a heading or a table)."""
    splitter = splitter or make_semantic_splitter()
    chunks: list[Chunk] = []
    for block in blocks:
        text, offsets = block.reflowed()
        if count_tokens(text) < skip_below:
            continue

        docs = splitter.create_documents([text])
        pieces = [(max(0, d.metadata.get("start_index", 0)), d.page_content) for d in docs]
        for start, body in _sized(pieces, min_tokens, max_tokens):
            page = block.lines[max(0, bisect_right(offsets, start) - 1)][0]
            section = block.section or "General"
            full = f"{make_prefix(meta, section, page)}\n{body}"
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.doc_id}-semantic-{len(chunks):05d}",
                    doc_id=meta.doc_id,
                    company=meta.company,
                    fiscal_year=meta.fiscal_year,
                    doc_type=meta.doc_type,
                    kind="text",
                    strategy="semantic",
                    section=section,
                    page=page,
                    text=full,
                    embed_text=full,
                    bm25_text=full,
                    n_tokens=count_tokens(full),
                    extra={"min_tokens": min_tokens, "max_tokens": max_tokens},
                )
            )
    return chunks
