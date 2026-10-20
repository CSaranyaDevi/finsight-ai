"""LLM reranking with LangChain's LLMListwiseRerank.

The model reads the question together with all candidate chunks and returns them ordered by
relevance, keeping the top `top_n`. Unlike the retrievers, it sees the full chunk text (the
markdown table, not its embedded description), so it can tell which table actually holds the
asked-for figure.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from finsight.agent.llm import chat_model
from finsight.config import RERANK_MODEL

try:  # LangChain >= 1.0 moved these to langchain-classic
    from langchain.retrievers.document_compressors import LLMListwiseRerank
except ImportError:  # pragma: no cover
    from langchain_classic.retrievers.document_compressors import LLMListwiseRerank

log = logging.getLogger(__name__)


def llm_rerank(question: str, docs: list[Document], top_n: int, model: str = RERANK_MODEL) -> list[Document]:
    """Reorder `docs` by relevance to `question` and keep `top_n`. Falls back to the input order on failure."""
    if len(docs) <= 1:
        return docs[:top_n]
    try:
        reranker = LLMListwiseRerank.from_llm(chat_model(model, "low"), top_n=top_n)
        ranked = list(reranker.compress_documents(docs, question))
    except Exception as e:  # rate limit, malformed output, ...: never block the answer
        log.warning("rerank failed, keeping retrieval order: %s", e)
        return docs[:top_n]
    # Top up with unranked docs in retrieval order if the model returned fewer than top_n
    seen = {d.metadata["chunk_id"] for d in ranked}
    rest = [d for d in docs if d.metadata["chunk_id"] not in seen]
    return (ranked + rest)[:top_n]
