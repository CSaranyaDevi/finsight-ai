"""Generate the high-level design diagram for docs/HLD.md with the OpenAI image API.

Usage:  python scripts/generate_hld_diagram.py [--model gpt-image-2] [--out docs/images/finsight_hld.png]
Needs OPENAI_API_KEY (.env). One image per run.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import finsight.config  # noqa: E402,F401  (loads .env)
from openai import OpenAI  # noqa: E402

PROMPT = """A landscape presentation slide (16:9 feel) showing a system architecture. Clean white background,
structured slide layout with aligned boxes and straight arrows, navy and teal accents, simple line icons.
ALL text in a casual handwritten marker-style font that is still neat and clearly legible.
Spell every label exactly as written. No extra text, no logos, no people.

TITLE (top left, large): "FinSight AI: Table-Aware Agentic RAG Architecture"
SUBTITLE: "Question answering over Microsoft, Amazon and Alphabet annual reports and Form 10-K filings, FY2023–2025"

TOP ROW, a band labeled "Offline Knowledge Base Construction", four boxes left to right joined by arrows:
1 "Document Ingestion" — "9 filings (Word, PDF)"
2 "Table-Aware Parsing" — "~500 tables; values aligned to row, fiscal year and units"
3 "Chunking" — "512-token text chunks; tables preserved as complete units with metadata"
4 "Hybrid Index" — "Dense: text-embedding-3-large in Chroma · Sparse: BM25"

BOTTOM ROW, a larger band labeled "Online Query Processing (LangGraph Agent)", six boxes left to right joined by arrows:
1 "Query Analysis" — "gpt-4.1: entities, fiscal years, terminology"
2 "Query Decomposition" — "one retrieval per company and year"
3 "Hybrid Retrieval" — "dense + BM25, rank fusion, metadata filters"
4 "Reranking" — "gpt-5.4-mini: top 12 of 30"
5 "Evidence Validation" — "gpt-4.1: sufficiency check, one retry"
6 "Answer Generation" — "gpt-5.4: grounded, cited response"
Below box 5, a small box with a thin red border: "Insufficient evidence: explicit refusal"

A dashed arrow from "Hybrid Index" down to "Hybrid Retrieval".

RIGHT SIDE, a narrow vertical panel titled "Results" with three lines:
"Evidence in top 3: 98%"
"Evidence in top 6: 100%"
"MRR: 0.96"
and under it in small text: "56 evaluated questions"

FOOTER (small): "Interface: Streamlit chat with citations and source tables  |  Stack: LangChain · LangGraph · OpenAI · Chroma"
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-image-2")
    ap.add_argument("--out", default=str(ROOT / "docs" / "images" / "finsight_hld.png"))
    ap.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--size", default="1536x1024", help="1024x1536 portrait, 1536x1024 landscape")
    args = ap.parse_args()

    result = OpenAI().images.generate(model=args.model, prompt=PROMPT, size=args.size, quality=args.quality, n=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(result.data[0].b64_json))
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB) with {args.model}")


if __name__ == "__main__":
    main()
