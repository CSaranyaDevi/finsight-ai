"""Paths, env settings and corpus metadata."""
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

def _path(env_name: str, default: str) -> Path:
    """Relative paths in .env are resolved against the project root, not the working directory."""
    p = Path(os.getenv(env_name) or default)
    return p if p.is_absolute() else (ROOT / p).resolve()


DATA_DIR = _path("DATA_DIR", "DataSource")
INDEX_DIR = _path("INDEX_DIR", "index")

# Defaults = the final, evaluated setup, so a deployment only needs OPENAI_API_KEY
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))  # index vectors; large shortened from its native 3072
# Semantic chunk boundaries use their own model so switching the index model keeps the chunk sets unchanged
CHUNKER_EMBEDDING_MODEL = os.getenv("CHUNKER_EMBEDDING_MODEL", "text-embedding-3-small")
# Chat models per role (any OpenAI chat model; reasoning models get reasoning_effort instead of temperature)
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "gpt-4.1")  # query analysis + context check
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-5.4")  # final answer (reasoning model, effort "medium")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-5.4-mini")  # reranking (reasoning effort "low")
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "llm")  # "llm" (LLM listwise rerank) | "none"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))  # retrieved before reranking down to the context size

# Hybrid retrieval fusion weights (dense, bm25); tuned with the retrieval eval
HYBRID_WEIGHTS = tuple(float(x) for x in os.getenv("HYBRID_WEIGHTS", "0.5,0.5").split(","))

# Microsoft's fiscal year ends June 30; Amazon and Alphabet end Dec 31.
FISCAL_YEAR_END = {"Microsoft": "June 30", "Amazon": "December 31", "Alphabet": "December 31"}


@dataclass(frozen=True)
class DocMeta:
    path: Path
    company: str
    fiscal_year: int
    doc_type: str  # annual_report | 10-K

    @property
    def doc_id(self) -> str:
        return f"{self.company.lower()}_{self.fiscal_year}"


def doc_meta(path: Path) -> DocMeta:
    """Infer company / fiscal year from the file name."""
    name = path.name.lower()
    year = int(re.search(r"(20\d\d)", name).group(1))
    if name.startswith("amazon"):
        return DocMeta(path, "Amazon", year, "annual_report")
    if name.startswith("goog"):
        return DocMeta(path, "Alphabet", year, "10-K")
    return DocMeta(path, "Microsoft", year, "annual_report")  # 20xx_Annual_Report.docx


def list_documents() -> list[DocMeta]:
    files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".pdf", ".docx"}]
    return sorted((doc_meta(p) for p in files), key=lambda m: (m.company, m.fiscal_year))
