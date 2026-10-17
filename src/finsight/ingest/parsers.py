"""Parse DOCX / PDF into an ordered stream of text and table elements."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from finsight.config import DocMeta


@dataclass
class Element:
    kind: str  # "text" | "table"
    page: int | None  # PDF page (1-based); None for DOCX
    text: str = ""  # text body, or markdown for a table
    rows: list[list[str]] = field(default_factory=list)  # table cells (tables only)
    order: int = 0  # position within the document


def clean_cell(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _dedupe_row(cells: list[str]) -> list[str]:
    """DOCX merged cells repeat the same cell object; drop adjacent repeats."""
    out: list[str] = []
    for c in cells:
        if not out or c != out[-1] or c == "":
            out.append(c)
    return out


def _is_useful_table(rows: list[list[str]]) -> bool:
    cells = [c for r in rows for c in r if c]
    return len(rows) >= 2 and len(cells) >= 4


def parse_docx(meta: DocMeta) -> list[Element]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(meta.path))
    elements: list[Element] = []
    para_buf: list[str] = []

    def flush():
        if para_buf:
            elements.append(Element("text", None, "\n".join(para_buf), order=len(elements)))
            para_buf.clear()

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            t = clean_cell(Paragraph(child, doc).text)
            if t:
                para_buf.append(t)
        elif tag == "tbl":
            flush()
            table = Table(child, doc)
            rows = []
            for r in table.rows:
                cells = _dedupe_row([clean_cell(c.text) for c in r.cells])
                if any(cells):
                    rows.append(cells)
            if _is_useful_table(rows):
                elements.append(Element("table", None, rows_to_markdown(rows), rows, len(elements)))
    flush()
    return elements


def parse_pdf(meta: DocMeta) -> list[Element]:
    import pdfplumber

    elements: list[Element] = []
    with pdfplumber.open(str(meta.path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_boxes = [t.bbox for t in tables]

            def outside(obj):
                x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                return not any(x0 >= b[0] - 1 and x1 <= b[2] + 1 and top >= b[1] - 1 and bottom <= b[3] + 1
                               for b in table_boxes)

            body = page.filter(outside).extract_text() if table_boxes else page.extract_text()
            if body and body.strip():
                elements.append(Element("text", pno, body.strip(), order=len(elements)))
            for t in tables:
                rows = [[clean_cell(c) for c in r] for r in t.extract() if r]
                rows = [r for r in rows if any(r)]
                if _is_useful_table(rows):
                    elements.append(Element("table", pno, rows_to_markdown(rows), rows, len(elements)))
    return elements


def parse_document(meta: DocMeta) -> list[Element]:
    return parse_docx(meta) if meta.path.suffix.lower() == ".docx" else parse_pdf(meta)
