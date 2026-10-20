"""Chroma vector store (one collection per chunking strategy) + chunk docstore.

Multi-vector pattern: Chroma holds the vector of each chunk's `embed_text` (for tables,
the short description). Retrieval maps the hit's `chunk_id` to the full chunk `text`
(the markdown table for tables) through the docstore, which is what the LLM sees.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_core.documents import Document

from finsight.chunking.base import Chunk, load_chunks
from finsight.chunking.pipeline import chunks_path
from finsight.config import EMBEDDING_DIM, EMBEDDING_MODEL, INDEX_DIR
from finsight.index.embeddings import cached_embeddings

CHROMA_DIR = INDEX_DIR / "chroma_db"
MANIFEST = INDEX_DIR / "index_manifest.json"
BATCH = 500


def chunk_metadata(c: Chunk) -> dict:
    """Scalar metadata only (Chroma rejects None, lists and dicts)."""
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "company": c.company,
        "fiscal_year": c.fiscal_year,
        "doc_type": c.doc_type,
        "kind": c.kind,
        "section": c.section,
        "page": c.page or 0,  # 0 = no page (DOCX)
        "title": c.extra.get("title", "") if c.kind == "table" else "",
    }


def to_document(c: Chunk) -> Document:
    """The document the LLM sees: full chunk text + metadata."""
    return Document(page_content=c.text, metadata=chunk_metadata(c), id=c.chunk_id)


def open_vectorstore(strategy: str, embeddings=None) -> Chroma:
    return Chroma(
        collection_name=strategy,
        embedding_function=embeddings or cached_embeddings(EMBEDDING_MODEL, EMBEDDING_DIM),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _embed_hash(c: Chunk) -> str:
    return hashlib.sha256(c.embed_text.encode("utf8")).hexdigest()[:16]


def build_vectorstore(strategy: str, rebuild: bool = False) -> Chroma:
    """Embed every chunk's `embed_text` into the `strategy` collection (calls the OpenAI API).

    Skips the work only if the collection holds the same chunk ids with the same embed_text
    (compared by hash), unless `rebuild`. Unchanged texts come from the embedding cache.
    """
    chunks = load_chunks(chunks_path(strategy))
    store = open_vectorstore(strategy)
    got = store.get(include=["metadatas"])
    existing = {i: (m or {}).get("embed_hash") for i, m in zip(got["ids"], got["metadatas"])}
    wanted = {c.chunk_id: _embed_hash(c) for c in chunks}
    if existing == wanted and not rebuild:
        return store
    if existing:
        store.delete_collection()
        store = open_vectorstore(strategy)

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        store.add_texts(
            texts=[c.embed_text for c in batch],
            metadatas=[{**chunk_metadata(c), "embed_hash": _embed_hash(c)} for c in batch],
            ids=[c.chunk_id for c in batch],
        )
    _update_manifest(strategy, len(chunks))
    return store


def build_all(strategies: tuple[str, ...] = ("fixed", "semantic")) -> dict[str, int]:
    """Rebuild `index/chroma_db/` from scratch for all strategies.

    Deleting and re-adding collections in place leaves orphaned segment folders and grows
    chroma.sqlite3 (it logs every add/delete), so the committed index is always built into an
    empty folder. Vectors come from the embedding cache, so a rebuild costs nothing unless
    chunk text changed. Restart the kernel first if Chroma is already open in this process.
    """
    if CHROMA_DIR.exists():
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError as e:
            raise PermissionError(
                f"{CHROMA_DIR} is open in another process (another notebook kernel, e.g. notebook 04, "
                "or the Streamlit app). Shut down every kernel and stop Streamlit, then rerun. "
                f"Original error: {e}"
            ) from e
    counts = {}
    for strategy in strategies:
        store = build_vectorstore(strategy, rebuild=True)
        counts[strategy] = store._collection.count()
    return counts


def load_docstore(strategy: str) -> dict[str, Document]:
    """chunk_id -> full Document, built from the committed chunk JSONL."""
    return {c.chunk_id: to_document(c) for c in load_chunks(chunks_path(strategy))}


def _update_manifest(strategy: str, n_chunks: int) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf8")) if MANIFEST.exists() else {}
    manifest.update({"embedding_model": EMBEDDING_MODEL, "embedding_dim": EMBEDDING_DIM})
    manifest.setdefault("collections", {})[strategy] = {
        "chunks": n_chunks,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf8")


def check_manifest() -> None:
    """Fail fast if the index was built with a different embedding model than the app uses."""
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} not found; build the index first")
    m = json.loads(MANIFEST.read_text(encoding="utf8"))
    if m.get("embedding_model") != EMBEDDING_MODEL or m.get("embedding_dim") != EMBEDDING_DIM:
        raise ValueError(
            f"Index built with {m.get('embedding_model')} ({m.get('embedding_dim')}d), "
            f"app configured for {EMBEDDING_MODEL} ({EMBEDDING_DIM}d)"
        )
