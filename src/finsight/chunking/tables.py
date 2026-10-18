"""Table chunks, shared by both chunking strategies.

Each parsed table becomes one chunk if it fits `max_tokens`, otherwise several row-group
chunks that each repeat the header row, title and units, so every chunk stands alone.

Three views per chunk (multi-vector pattern):
- text:       prefix + title + units + markdown table  -> what the LLM sees
- embed_text: short description (title, units, columns, row labels) -> what gets embedded
- bm25_text:  raw labels and figures without markdown syntax -> what BM25 indexes
"""
from __future__ import annotations

import re

from finsight.chunking.base import Chunk, count_tokens, make_prefix
from finsight.config import DocMeta
from finsight.ingest.parsers import Element, rows_to_markdown

_YEAR = re.compile(r"^(?:19|20)\d\d$")
_UNITS = re.compile(r"\(?\b(in (?:millions|thousands|billions)(?:,? except [^)]{0,80})?)\)?", re.IGNORECASE)
_PERIOD_LINE = re.compile(r"(?i)^(years?|as of|three|six|nine|twelve|fiscal|quarter)\b.*(ended|ending|of)\b")
_VALUE = re.compile(r"^\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?$")
_UNITS_ONLY = re.compile(r"(?i)^\(?\s*in (?:millions|thousands|billions)[^)]{0,80}\)?:?$")


def _is_year_line(line: str) -> bool:
    tokens = line.split()
    return len(tokens) >= 2 and all(_YEAR.match(t) for t in tokens)


def find_units(context: list[str], rows: list[list[str]]) -> str:
    """'in millions, except per share amounts' from the caption lines or the first table rows."""
    haystack = context[-4:] + [" ".join(r) for r in rows[:2]]
    for text in reversed(haystack):
        m = _UNITS.search(text)
        if m:
            return m.group(1).strip().rstrip(":").lower()
    return ""


def find_title(context: list[str], section: str) -> str:
    """Last meaningful line before the table, skipping header-ish lines
    ('2023 2024', 'Year Ended December 31,', '(in millions)')."""
    for line in reversed(context):
        line = line.strip()
        if not line or _is_year_line(line) or _PERIOD_LINE.match(line) or _UNITS_ONLY.match(line) or len(line) < 4:
            continue
        return line[:200]
    return section or "Table"


def split_header(rows: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    """Treat the first row as a header unless it holds data values (numbers other than years)."""
    if not rows:
        return [], []
    first = [c.strip() for c in rows[0][1:] if c.strip()]
    holds_data = any(_VALUE.match(c.replace(" ", "")) and not _YEAR.match(c) for c in first)
    return ([], rows) if holds_data else ([rows[0]], rows[1:])


def describe(meta: DocMeta, title: str, section: str, units: str, header: list[list[str]], body: list[list[str]]) -> str:
    """Rule-based description for embedding. Row labels carry most of the retrieval signal.

    Can later be replaced by an LLM-written summary (multi-vector pattern) without changing callers.
    """
    columns = [c for c in (header[0][1:] if header else []) if c]
    labels = [r[0] for r in body if r and r[0]][:30]
    parts = [
        f"{meta.company} {meta.doc_type} FY{meta.fiscal_year}.",
        f"Table: {title}.",
        f"Section: {section}." if section else "",
        f"Units: {units}." if units else "",
        f"Columns: {', '.join(columns)}." if columns else "",
        f"Rows: {'; '.join(labels)}." if labels else "",
    ]
    return " ".join(p for p in parts if p)


def _flat(rows: list[list[str]]) -> str:
    return "\n".join(" ".join(c for c in r if c) for r in rows)


def _row_groups(header: list[list[str]], body: list[list[str]], budget: int) -> list[tuple[int, int]]:
    """(start, end) indexes into `body` so each group plus the header fits `budget` tokens."""
    header_tokens = count_tokens(rows_to_markdown(header)) if header else 0
    groups, start, used = [], 0, header_tokens
    for i, row in enumerate(body):
        t = count_tokens("| " + " | ".join(row) + " |")
        if i > start and used + t > budget:
            groups.append((start, i))
            start, used = i, header_tokens
        used += t
    groups.append((start, len(body)))
    return groups


def chunk_tables(
    meta: DocMeta,
    tables: list[tuple[Element, str, list[str]]],
    max_tokens: int = 800,
) -> list[Chunk]:
    """`tables` are the ("table", (element, section, context_lines)) items from `iter_blocks`."""
    chunks: list[Chunk] = []
    for t_idx, (el, section, context) in enumerate(tables):
        section = section or "General"
        title = find_title(context, section)
        units = find_units(context, el.rows)
        header, body = split_header(el.rows)
        prefix = make_prefix(meta, section, el.page)
        units_line = f"\nUnits: {units}" if units else ""

        def intro(label: str = "") -> str:
            return f"{prefix}\nTable: {title}{label}{units_line}"

        budget = max_tokens - count_tokens(intro(" (rows 100-200 of 200)"))
        groups = [(0, len(body))]
        if count_tokens(rows_to_markdown(header + body)) > budget:
            groups = _row_groups(header, body, budget)

        for g_idx, (start, end) in enumerate(groups):
            part = body[start:end]
            markdown = rows_to_markdown(header + part)
            label = f" (rows {start + 1}-{end} of {len(body)})" if len(groups) > 1 else ""
            text = f"{intro(label)}\n{markdown}"
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.doc_id}-table-{t_idx:04d}" + (f"-g{g_idx}" if len(groups) > 1 else ""),
                    doc_id=meta.doc_id,
                    company=meta.company,
                    fiscal_year=meta.fiscal_year,
                    doc_type=meta.doc_type,
                    kind="table",
                    strategy="shared",  # identical in the fixed and semantic chunk sets
                    section=section,
                    page=el.page,
                    text=text,
                    embed_text=describe(meta, title, section, units, header, part),
                    bm25_text=f"{prefix}\n{title}\n{units}\n{_flat(header + part)}",
                    n_tokens=count_tokens(text),
                    extra={
                        "title": title,
                        "units": units,
                        "columns": header[0] if header else [],
                        "table_index": t_idx,
                        "row_range": [start, end],
                        "n_groups": len(groups),
                    },
                )
            )
    return chunks
