"""Shared chunk model, token counting and section/heading tracking."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken

from finsight.config import DocMeta
from finsight.ingest.parsers import Element

_ENC = tiktoken.get_encoding("cl100k_base")  # tokenizer used by OpenAI embedding models


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company: str
    fiscal_year: int
    doc_type: str
    kind: str  # "text" | "table"
    strategy: str  # "fixed" | "semantic" | "shared" (tables, used by both)
    section: str
    page: int | None
    text: str  # what the LLM sees (prefix + body)
    embed_text: str  # what gets embedded (tables: a short description)
    bm25_text: str  # what BM25 indexes
    n_tokens: int
    extra: dict = field(default_factory=dict)  # table title, units, row range, ...

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def make_prefix(meta: DocMeta, section: str, page: int | None) -> str:
    parts = [meta.company, f"FY{meta.fiscal_year}", meta.doc_type, section or "General"]
    if page:
        parts.append(f"p.{page}")
    return " | ".join(parts)


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for c in chunks:
            f.write(c.to_json() + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf8") as f:
        return [Chunk(**json.loads(line)) for line in f]


# --- section tracking -------------------------------------------------------

_ITEM_OR_PART = re.compile(r"(?i)^(item\s+\d+[a-c]?\.?\s.{0,90}|part\s+[iv]+\b.{0,60})$")


_ROMAN = re.compile(r"^[IVX]+$")


def _is_spaced_letters(line: str) -> bool:
    """'A N N U A L R E P O R T' style cover text."""
    tokens = line.split()
    return len(tokens) >= 4 and sum(len(t) == 1 for t in tokens) / len(tokens) >= 0.7


def section_name(line: str) -> str:
    """Title-case ALL-CAPS headings but keep roman numerals and form names ('PART II', '10-K')."""
    if not line.isupper():
        return line
    return " ".join(w if _ROMAN.match(w) or any(ch.isdigit() for ch in w) else w.capitalize() for w in line.split())


def is_heading(line: str) -> bool:
    """Heuristic: 10-K 'Item 7. ...' / 'PART II' lines and short ALL-CAPS lines.

    Rejects table-of-contents entries ('Item 12. ... 90'), cross-references
    ('Part II, Item 7 of this Annual Report on Form 10-K.'), cover-page checkbox
    lines ('☒ ANNUAL REPORT PURSUANT ...') and spaced-out cover text.
    """
    line = line.strip()
    if not line or len(line) > 100 or re.fullmatch(r"[\d\s.,$%()\-]+", line):
        return False
    if any(ch in line for ch in "☒☐") or _is_spaced_letters(line):
        return False
    if _ITEM_OR_PART.match(line):
        is_toc_entry = re.search(r"\s\d{1,3}$", line)
        is_sentence = line.endswith(".") or "of this annual report" in line.lower()
        return not (is_toc_entry or is_sentence)
    letters = [c for c in line if c.isalpha()]
    return (
        len(letters) >= 5
        and not line.endswith((".", ",", ";"))
        and sum(c.isupper() for c in letters) / len(letters) > 0.9
    )


@dataclass
class Block:
    """A run of text lines in one section, uninterrupted by a heading or a table."""
    section: str
    lines: list[tuple[int | None, str]]  # (page, line)

    def reflowed(self) -> tuple[str, list[int]]:
        """Block text with PDF line wraps joined, plus the start offset of each line.

        PDF text breaks at every visual line wrap, so splitting on newlines cuts sentences.
        A line break is kept only after sentence-ending punctuation or before a bullet;
        other wraps become spaces. DOCX lines (page None) are real paragraphs and keep newlines.
        Each separator is one character, so `offsets` map a character position back to a line/page.
        """
        parts, offsets, pos = [], [], 0
        for i, (page, line) in enumerate(self.lines):
            offsets.append(pos)
            parts.append(line)
            pos += len(line) + 1
            if i < len(self.lines) - 1:
                nxt = self.lines[i + 1][1]
                keep_break = page is None or line.endswith((".", "!", "?", ":", ";")) or nxt[:1] in "•●-–"
                parts.append("\n" if keep_break else " ")
        return "".join(parts), offsets


def iter_blocks(elements: list[Element]):
    """Yield ("block", Block) for text runs and ("table", (Element, section, context_lines)).

    A heading starts a new block and updates the section; a table ends the current block.
    Chunks therefore never cross a heading or a table. `context_lines` are the few lines
    of text just before a table (used for its caption and units).
    """
    section = ""
    lines: list[tuple[int | None, str]] = []
    recent: list[str] = []

    for el in elements:
        if el.kind == "text":
            for raw in el.text.split(chr(10)):
                line = raw.strip()
                if not line:
                    continue
                if is_heading(line):
                    if lines:
                        yield "block", Block(section, lines)
                        lines = []
                    section = section_name(line)
                    recent = [line]
                    continue
                lines.append((el.page, line))
                recent = (recent + [line])[-6:]
        else:
            if lines:
                yield "block", Block(section, lines)
                lines = []
            yield "table", (el, section, list(recent))
            recent = []
    if lines:
        yield "block", Block(section, lines)
