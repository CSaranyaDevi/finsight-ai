"""LangGraph agent: analyze -> decompose -> retrieve -> rerank -> grade -> (retry) -> answer | refuse.

    START -> analyze --(out of scope)--> refuse -> END
                |
             decompose -> retrieve -> rerank -> grade --(yes/partial)--> answer -> END
                              ^                    |
                              +----(no, 1 retry)---+--(no, retried)--> refuse -> END

`rerank` is a pass-through until step 7 plugs in the LLM reranker.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from finsight.agent import prompts
from finsight.agent.llm import chat_model
from finsight.config import ANALYSIS_MODEL, ANSWER_MODEL, HYBRID_WEIGHTS
from finsight.retrieve.hybrid import search

MAX_CONTEXT = 12  # chunks passed to the answer step
MAX_SUBQUERIES = 9  # 3 companies x 3 years
MAX_RETRIES = 1

Company = Literal["Microsoft", "Amazon", "Alphabet"]


class QueryAnalysis(BaseModel):
    in_scope: bool = Field(description="Answerable from the Microsoft / Amazon / Alphabet filings")
    companies: list[Company] = Field(description="Companies the question is about")
    fiscal_years: list[int] = Field(description="Requested fiscal years (2023-2025); empty if none given")
    question_type: Literal["lookup", "comparison", "narrative"]
    search_query: str = Field(description="Standalone search query without years")


class Grade(BaseModel):
    answerable: Literal["yes", "partial", "no"]
    missing: str = Field(description="What the excerpts lack; empty if nothing")


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}, ...]
    strategy: str  # chunk set: "fixed" | "semantic"
    mode: str  # retrieval: "hybrid" (default) | "dense" | "bm25"
    weights: tuple  # hybrid fusion weights (dense, bm25); default HYBRID_WEIGHTS
    analysis: dict
    subqueries: list[dict]  # [{"query", "company", "fiscal_year"}]
    docs: list[Document]
    grade: dict
    attempts: int
    answer: str
    citations: list[dict]
    route: str  # how the answer was produced: answered | not_found | out_of_scope


# --- helpers ---------------------------------------------------------------

def _history_messages(history: list[dict] | None, limit: int = 6):
    msgs = []
    for turn in (history or [])[-limit:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        msgs.append(cls(content=turn["content"]))
    return msgs


def source_label(doc: Document) -> str:
    m = doc.metadata
    page = f", p.{m['page']}" if m.get("page") else ""
    return f"{m['company']} {m['doc_type'].replace('_', ' ')} FY{m['fiscal_year']} — {m['section']}{page}"


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {source_label(d)} ({d.metadata['kind']})\n{d.page_content}" for i, d in enumerate(docs, 1))


def _merge(groups: list[list[Document]], limit: int) -> list[Document]:
    """Round-robin across sub-query results so every company/year gets represented; dedupe on chunk_id."""
    seen, merged = set(), []
    for rank in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if rank < len(g) and g[rank].metadata["chunk_id"] not in seen:
                seen.add(g[rank].metadata["chunk_id"])
                merged.append(g[rank])
    return merged[:limit]


# --- nodes -----------------------------------------------------------------

def analyze(state: AgentState) -> dict:
    llm = chat_model(ANALYSIS_MODEL, "low").with_structured_output(QueryAnalysis)
    result: QueryAnalysis = llm.invoke(
        [SystemMessage(prompts.ANALYZE), *_history_messages(state.get("history")), HumanMessage(state["question"])]
    )
    analysis = result.model_dump()
    analysis["fiscal_years"] = sorted({y for y in analysis["fiscal_years"] if 2023 <= y <= 2025})
    analysis["companies"] = list(dict.fromkeys(analysis["companies"]))
    return {"analysis": analysis, "attempts": 0}


def decompose(state: AgentState) -> dict:
    """One sub-query per (company, year); no filter on an axis the question doesn't specify."""
    a = state["analysis"]
    companies = a["companies"] or [None]
    years = a["fiscal_years"] or [None]
    subqueries = [
        {"query": a["search_query"], "company": c, "fiscal_year": y} for c in companies for y in years
    ][:MAX_SUBQUERIES]
    return {"subqueries": subqueries}


def retrieve(state: AgentState) -> dict:
    strategy = state.get("strategy", "fixed")
    subs = state["subqueries"]
    per_sub = max(3, MAX_CONTEXT // len(subs))
    missing = state.get("grade", {}).get("missing", "") if state.get("attempts") else ""
    groups = []
    for s in subs:
        query = f"{s['query']} {missing}".strip()  # on retry, search for what the grader said was missing
        groups.append(search(query, strategy, state.get("mode", "hybrid"), company=s["company"], fiscal_year=s["fiscal_year"],
                             k=per_sub, weights=tuple(state.get("weights", HYBRID_WEIGHTS))))
    new_docs = _merge(groups, MAX_CONTEXT * 2)
    if state.get("attempts"):  # keep what we had, add the new hits
        new_docs = _merge([state.get("docs", []), new_docs], MAX_CONTEXT * 2)
    return {"docs": new_docs}


def rerank(state: AgentState) -> dict:
    """Pass-through for now (step 7 replaces this with an LLM reranker)."""
    return {"docs": state["docs"][:MAX_CONTEXT]}


def grade(state: AgentState) -> dict:
    llm = chat_model(ANALYSIS_MODEL, "low").with_structured_output(Grade)
    result: Grade = llm.invoke(
        [
            SystemMessage(prompts.GRADE),
            HumanMessage(f"Question: {state['question']}\n\nExcerpts:\n{format_context(state['docs'])}"),
        ]
    )
    return {"grade": result.model_dump(), "attempts": state.get("attempts", 0) + 1}


def answer(state: AgentState) -> dict:
    docs = state["docs"]
    llm = chat_model(ANSWER_MODEL, "medium")
    msg = llm.invoke(
        [
            SystemMessage(prompts.ANSWER),
            *_history_messages(state.get("history")),
            HumanMessage(f"Excerpts:\n{format_context(docs)}\n\nQuestion: {state['question']}"),
        ]
    )
    text = msg.content if isinstance(msg.content, str) else "".join(
        p.get("text", "") for p in msg.content if isinstance(p, dict)
    )
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(docs)})
    citations = [
        {"n": n, "label": source_label(docs[n - 1]), **{k: docs[n - 1].metadata[k] for k in ("chunk_id", "kind", "page")}}
        for n in used
    ]
    return {"answer": text, "citations": citations, "route": "answered"}


def refuse(state: AgentState) -> dict:
    if not state["analysis"]["in_scope"]:
        return {"answer": prompts.REFUSE_OUT_OF_SCOPE, "citations": [], "route": "out_of_scope"}
    return {"answer": prompts.REFUSE_NOT_FOUND, "citations": [], "route": "not_found"}


# --- routing ---------------------------------------------------------------

def after_analyze(state: AgentState) -> str:
    return "decompose" if state["analysis"]["in_scope"] else "refuse"


def after_grade(state: AgentState) -> str:
    if state["grade"]["answerable"] in ("yes", "partial"):
        return "answer"
    return "retrieve" if state["attempts"] <= MAX_RETRIES else "refuse"


# --- graph -----------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("analyze", analyze), ("decompose", decompose), ("retrieve", retrieve),
                     ("rerank", rerank), ("grade", grade), ("answer", answer), ("refuse", refuse)]:
        g.add_node(name, fn)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", after_analyze, ["decompose", "refuse"])
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade")
    g.add_conditional_edges("grade", after_grade, ["answer", "retrieve", "refuse"])
    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_agent():
    return build_graph()


def ask(question: str, history: list[dict] | None = None, strategy: str = "fixed") -> AgentState:
    """Run the agent. Returns the final state: answer, citations, route, plus analysis/docs/grade for inspection."""
    return get_agent().invoke({"question": question, "history": history or [], "strategy": strategy})
