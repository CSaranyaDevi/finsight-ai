"""Strategy A: fixed-size text chunks via LangChain's RecursiveCharacterTextSplitter."""
from __future__ import annotations

from bisect import bisect_right

from langchain_text_splitters import RecursiveCharacterTextSplitter

from finsight.chunking.base import Block, Chunk, count_tokens, make_prefix
from finsight.config import DocMeta


def make_splitter(chunk_size: int = 512, chunk_overlap: int = 64) -> RecursiveCharacterTextSplitter:
    """Token-based splitter (same tokenizer as the OpenAI embedding models).

    Tries to break on paragraph, then line, then sentence, then word boundaries.
    chunk_size counts the body only; the metadata prefix adds ~15-25 tokens.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def chunk_text_fixed(
    meta: DocMeta,
    blocks: list[Block],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_tokens: int = 25,
) -> list[Chunk]:
    """Split each block independently, so chunks never cross a heading or a table.

    Blocks shorter than `min_tokens` are skipped: they are mostly table captions and
    header lines ("Year Ended December 31, 2023 2024") that the table chunks already carry.
    """
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for block in blocks:
        # wrapped PDF lines joined into paragraphs; offsets map a split back to its page
        text, offsets = block.reflowed()
        if count_tokens(text) < min_tokens:
            continue

        for doc in splitter.create_documents([text]):
            start = max(0, doc.metadata.get("start_index", 0))
            page = block.lines[max(0, bisect_right(offsets, start) - 1)][0]
            section = block.section or "General"
            full = f"{make_prefix(meta, section, page)}\n{doc.page_content}"
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.doc_id}-fixed-{len(chunks):05d}",
                    doc_id=meta.doc_id,
                    company=meta.company,
                    fiscal_year=meta.fiscal_year,
                    doc_type=meta.doc_type,
                    kind="text",
                    strategy="fixed",
                    section=section,
                    page=page,
                    text=full,
                    embed_text=full,
                    bm25_text=full,
                    n_tokens=count_tokens(full),
                    extra={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                )
            )
    return chunks
