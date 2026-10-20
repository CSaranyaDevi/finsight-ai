"""Retrieval evaluation against the gold set.

Runs each answerable gold question through the agent's own analyze -> decompose -> retrieve
steps (the real pipeline, including company/year extraction), for every combination of
chunking strategy and retrieval mode, and scores the retrieved chunks against the gold
evidence rules.

A chunk satisfies an evidence item if it comes from one of the item's `docs` and contains
every string in `contains` (case-insensitive). Metrics per question:
- hit@k:          at least one evidence item satisfied in the top k
- recall@k:       share of evidence items satisfied in the top k (matters for comparisons)
- all@k:          every evidence item satisfied in the top k
- mrr:            1 / rank of the first chunk that satisfies any evidence item (0 if none)

Query analysis is an LLM call; results are cached in eval/cache/ so the grid reuses them.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from finsight.config import ANALYSIS_MODEL, ROOT

GOLD_PATH = ROOT / "eval" / "gold_questions.jsonl"
CACHE_DIR = ROOT / "eval" / "cache"
REPORTS_DIR = ROOT / "eval" / "reports"
KS = (3, 6, 12)
STRATEGIES = ("fixed", "semantic")
MODES = ("dense", "bm25", "hybrid")


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def satisfies(doc: Document, item: dict) -> bool:
    text = doc.page_content.lower()
    return doc.metadata["doc_id"] in item["docs"] and all(s.lower() in text for s in item["contains"])


def score(docs: list[Document], evidence: list[dict]) -> dict:
    """Metrics for one ranked list of chunks against a question's evidence items."""
    out = {}
    for k in KS:
        top = docs[:k]
        found = [any(satisfies(d, item) for d in top) for item in evidence]
        out[f"hit@{k}"] = float(any(found))
        out[f"recall@{k}"] = sum(found) / len(evidence)
        out[f"all@{k}"] = float(all(found))
    first = next((i for i, d in enumerate(docs, 1) if any(satisfies(d, item) for item in evidence)), None)
    out["mrr"] = 1 / first if first else 0.0
    out["first_rank"] = first
    return out


# --- query analysis (cached) -----------------------------------------------

def _key(g: dict) -> str:
    from finsight.agent.prompts import ANALYZE

    raw = json.dumps({"q": g["question"], "h": g.get("history", []), "m": ANALYSIS_MODEL, "p": ANALYZE}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def analyze_all(gold: list[dict], refresh: bool = False) -> dict[str, dict]:
    """gold id -> agent analysis. One LLM call per uncached question."""
    from finsight.agent.graph import analyze

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "analysis.json"
    cache = json.loads(path.read_text(encoding="utf8")) if path.exists() and not refresh else {}
    out = {}
    for g in gold:
        k = _key(g)
        if k not in cache:
            cache[k] = analyze({"question": g["question"], "history": g.get("history", [])})["analysis"]
            path.write_text(json.dumps(cache, indent=1), encoding="utf8")
        out[g["id"]] = cache[k]
    return out


# --- retrieval grid --------------------------------------------------------

def retrieve_for(analysis: dict, strategy: str, mode: str, weights=None, rerank: str = "none", question: str = "") -> list[Document]:
    """The agent's decompose + retrieve + rerank steps for one configuration (rerank "none" | "llm")."""
    from finsight.agent.graph import decompose, retrieve
    from finsight.agent.graph import rerank as rerank_node

    state = {"analysis": analysis, "strategy": strategy, "mode": mode, "attempts": 0, "rerank": rerank, "question": question}
    if weights:
        state["weights"] = weights
    state.update(decompose(state))
    state.update(retrieve(state))
    return rerank_node(state)["docs"]


def run_retrieval_eval(
    strategies=STRATEGIES, modes=MODES, gold: list[dict] | None = None, save: bool = True, weights=None, tag: str = "", rerank: str = "none"
) -> pd.DataFrame:
    """One row per (question, strategy, mode). Refusal questions are skipped (no evidence).

    `weights` (dense, bm25) overrides HYBRID_WEIGHTS for hybrid mode; `tag` labels the run in the report file.
    `rerank="llm"` adds the LLM reranker (one LLM call per question and configuration).
    """
    gold = [g for g in (gold or load_gold()) if g.get("evidence")]
    analyses = analyze_all(gold)
    rows = []
    for strategy in strategies:
        for mode in modes:
            for g in gold:
                a = analyses[g["id"]]
                docs = retrieve_for(a, strategy, mode, weights, rerank, g["question"]) if a["in_scope"] else []
                rows.append({
                    "id": g["id"], "type": g["type"], "variant": g.get("variant", "original"),
                    "paraphrase_of": g.get("paraphrase_of", ""), "strategy": strategy, "mode": mode, "rerank": rerank,
                    "weights": ",".join(map(str, weights)) if (weights and mode == "hybrid") else "",
                    "in_scope": a["in_scope"], "companies": ",".join(a["companies"]),
                    "years": ",".join(map(str, a["fiscal_years"])), "n_evidence": len(g["evidence"]),
                    **score(docs, g["evidence"]),
                    "top_chunks": " ".join(d.metadata["chunk_id"] for d in docs[:6]),
                })
    df = pd.DataFrame(rows)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(REPORTS_DIR / f"retrieval_{tag + '_' if tag else ''}{datetime.now():%Y%m%d_%H%M}.csv", index=False)
    return df


def summarize(df: pd.DataFrame, by=("strategy", "mode")) -> pd.DataFrame:
    metrics = [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["all@6", "mrr"]
    return df.groupby(list(by))[metrics].mean().round(3)
