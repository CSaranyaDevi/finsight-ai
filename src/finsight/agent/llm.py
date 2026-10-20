"""Chat model factory: one place that knows reasoning vs non-reasoning model settings."""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

# Reasoning models reject a custom temperature and take reasoning_effort instead.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_PREFIXES) and "chat" not in model  # gpt-5-chat-latest is non-reasoning


@lru_cache(maxsize=8)
def chat_model(model: str, effort: str = "medium") -> ChatOpenAI:
    """temperature=0 for non-reasoning models (deterministic); reasoning_effort for reasoning models."""
    if is_reasoning_model(model):
        return ChatOpenAI(model=model, reasoning_effort=effort)
    return ChatOpenAI(model=model, temperature=0)
