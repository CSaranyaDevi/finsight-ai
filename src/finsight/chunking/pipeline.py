"""Parse -> blocks/tables -> chunks for one strategy, saved as JSONL under index/chunks/."""
from __future__ import annotations

from pathlib import Path

from finsight.chunking.base import Chunk, iter_blocks, save_chunks
from finsight.chunking.tables import chunk_tables
from finsight.config import INDEX_DIR, DocMeta, list_documents
from finsight.ingest.parsers import Element, parse_document

CHUNKS_DIR = INDEX_DIR / "chunks"


def split_elements(elements: list[Element]):
    blocks, tables = [], []
    for kind, item in iter_blocks(elements):
        (blocks if kind == "block" else tables).append(item)
    return blocks, tables


def chunk_document(meta: DocMeta, elements: list[Element], strategy: str, **kwargs) -> list[Chunk]:
    blocks, tables = split_elements(elements)
    if strategy == "fixed":
        from finsight.chunking.fixed import chunk_text_fixed
        text_chunks = chunk_text_fixed(meta, blocks, **kwargs)
    elif strategy == "semantic":
        from finsight.chunking.semantic import chunk_text_semantic
        text_chunks = chunk_text_semantic(meta, blocks, **kwargs)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return text_chunks + chunk_tables(meta, tables)


def build_chunks(strategy: str, parsed: dict[str, list[Element]] | None = None, **kwargs) -> list[Chunk]:
    """Chunk every document. Pass `parsed` (doc_id -> elements) to reuse an earlier parse."""
    chunks: list[Chunk] = []
    for meta in list_documents():
        elements = parsed[meta.doc_id] if parsed else parse_document(meta)
        chunks += chunk_document(meta, elements, strategy, **kwargs)
    return chunks


def chunks_path(strategy: str) -> Path:
    return CHUNKS_DIR / f"{strategy}.jsonl"


def build_and_save(strategy: str, parsed=None, **kwargs) -> list[Chunk]:
    chunks = build_chunks(strategy, parsed, **kwargs)
    save_chunks(chunks, chunks_path(strategy))
    return chunks
