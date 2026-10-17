"""Paths, env settings and corpus metadata."""
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "DataSource"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")

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
