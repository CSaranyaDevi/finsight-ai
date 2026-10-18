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


_NUM = r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?"
_NUMERIC_CELL = re.compile(rf"^{_NUM}$|^[—–-]$")
# Trailing run of numbers/symbols at the end of a row label: "AWS 90,757 107,556", "North America 12 % 10 %"
_TAIL = re.compile(rf"^(?P<label>.*?[^\d\s$(%,.\-—–])?\s*(?P<tail>(?:(?:{_NUM}|[—–-]|\$|%)\s*)+)$")


_DATE = re.compile(
    r"(?i)^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}$"
)


def _split_numeric_row(text: str) -> list[str]:
    """Split a one-cell row like 'AWS 90,757 107,556' into ['AWS', '90,757', '107,556']."""
    if _DATE.match(text.strip()):  # 'June 30, 2024' is a column label, not label + values
        return [text.strip()]
    m = _TAIL.match(text.strip())
    if not m or not any(ch.isdigit() for ch in m.group("tail")):
        return [text.strip()]
    tokens = m.group("tail").split()
    values: list[str] = []
    for tok in tokens:
        if tok in {"%", ")"} and values:
            values[-1] += tok
        elif values and values[-1] == "$":
            values[-1] = "$" + tok
        else:
            values.append(tok)
    label = (m.group("label") or "").strip()
    # 'Basic net income per share (Note 12) $4.59 ...': '12)' closes the label's bracket, it is not a value
    if values and label.count("(") > label.count(")") and values[0].endswith(")") and not values[0].startswith("("):
        label = f"{label} {values.pop(0)}"
    return [label, *values]


def _merge_symbol_cells(row: list[str]) -> list[str]:
    """Join stray '$' / '%' / ')' cells onto the neighbouring value cell."""
    out = list(row)
    for i, c in enumerate(out):
        if c == "$":
            j = next((k for k in range(i + 1, len(out)) if out[k]), None)
            if j is not None:
                out[j], out[i] = "$" + out[j], ""
        elif c in {"%", ")"}:
            j = next((k for k in range(i - 1, -1, -1) if out[k]), None)
            if j is not None:
                out[j], out[i] = out[j] + c, ""
    return out


def _drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    return [[r[j] for j in keep] for r in rows if any(r[j] for j in keep)]


def normalize_pdf_table(rows: list[list[str]]) -> list[list[str]]:
    """Clean a pdfplumber table: merge symbols, drop empty columns, recover collapsed rows.

    Line-less tables often come back with a row's label and all its values in the first
    cell ('Google Cloud 33,088 43,229' | '' | ''). Such rows are split into label + values,
    and the values are right-aligned so they sit under the value columns.
    """
    width = max((len(r) for r in rows), default=0)
    rows = _drop_empty_columns([_merge_symbol_cells(r + [""] * (width - len(r))) for r in rows])
    if not rows:
        return rows

    split_rows: list[list[str]] = []
    for r in rows:
        nonempty = [c for c in r if c]
        parts = _split_numeric_row(r[0]) if len(nonempty) == 1 and r[0] else None
        split_rows.append(parts if parts and len(parts) > 1 else r)
    width = max(len(r) for r in split_rows)
    aligned = []
    for r, original in zip(split_rows, rows):
        if r is original:  # untouched row: pad on the right
            aligned.append(r + [""] * (width - len(r)))
        else:  # split row: label first, values right-aligned under the value columns
            label, values = r[0], r[1:]
            aligned.append([label] + [""] * (width - 1 - len(values)) + values)
    return _drop_empty_columns(aligned)


def _looks_like_prose(rows: list[list[str]]) -> bool:
    """Paragraph text that the PDF table finder split into cells."""
    cells = [c for r in rows for c in r if c]
    if not cells:
        return True
    fragments = sum(1 for c in cells if c[0].islower() or c[0] in ",.;:")
    return fragments / len(cells) >= 0.3 or any(len(c) > 200 for c in cells)


def _is_useful_table(rows: list[list[str]]) -> bool:
    cells = [c for r in rows for c in r if c]
    if len(rows) < 2 or len(cells) < 4:
        return False
    numeric = sum(1 for c in cells if _NUMERIC_CELL.match(c.replace(" ", "")))
    return numeric >= 2 and not _looks_like_prose(rows)


_YEAR = re.compile(r"(?:19|20)\d\d")
_LABEL_START = re.compile(r"(?i)^(years?|as of|three|six|nine|twelve|fiscal|quarter|december|june|september|march)")


def place_years(rows: list[list[str]], years: list[str], label: str = "") -> list[list[str]] | None:
    """Put a header row of `years` over the columns that hold numbers, then drop spacer columns.

    Returns None if the number of value columns doesn't match the number of years.
    """
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    value_cols = [j for j in range(1, width) if any(_NUMERIC_CELL.match(r[j].replace(" ", "")) for r in rows)]
    if len(years) < 2 or len(value_cols) != len(years):
        return None
    header = [label] + [""] * (width - 1)
    for j, y in zip(value_cols, years):
        header[j] = y
    return _drop_empty_columns([header] + rows)


def year_header(body: str) -> tuple[str, list[str]] | None:
    """Find a column-header line (e.g. '2023 2024') just above a table, outside its detected box.

    Looks at the last lines of the text above the table for a line ending in 2+ years.
    Returns (label, years), e.g. ('Year Ended December 31,', ['2023', '2024']).
    """
    lines = [l.strip() for l in body.split(chr(10)) if l.strip()][-3:]
    for k in range(len(lines) - 1, -1, -1):
        tokens = lines[k].split()
        years: list[str] = []
        while tokens and _YEAR.fullmatch(tokens[-1]):
            years.insert(0, tokens.pop())
        if len(years) >= 2:
            label = " ".join(tokens).strip()
            if not label and k > 0 and _LABEL_START.match(lines[k - 1]):
                label = lines[k - 1]
            return label, years
    return None


def parse_docx(meta: DocMeta) -> list[Element]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(meta.path))
    elements: list[Element] = []
    para_buf: list[str] = []

    def flush():
        if para_buf:
            elements.append(Element("text", None, chr(10).join(para_buf), order=len(elements)))
            para_buf.clear()

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            t = clean_cell(Paragraph(child, doc).text)
            if t:
                para_buf.append(t)
        elif tag == "tbl":
            table = Table(child, doc)
            rows = []
            for r in table.rows:
                cells = _dedupe_row([clean_cell(c.text) for c in r.cells])
                if any(cells):
                    rows.append(cells)
            if _is_useful_table(rows):
                flush()
                elements.append(Element("table", None, rows_to_markdown(rows), rows, len(elements)))
            else:  # layout / text-only table: keep its content as ordinary text
                para_buf.extend(" ".join(c for c in r if c) for r in rows)
    flush()
    return elements


def parse_pdf(meta: DocMeta) -> list[Element]:
    """Per page: text above / between / below the accepted tables, interleaved in reading order."""
    import pdfplumber

    elements: list[Element] = []

    def text_of(page, y0: float, y1: float) -> str:
        if y1 - y0 < 2:
            return ""
        # x_tolerance=1: the default (3) glues words together in some PDFs (Amazon shareholder letter)
        return (page.crop((0, y0, page.width, y1)).extract_text(x_tolerance=1) or "").strip()

    def line_years(page, table) -> list[str]:
        """Years on the table's first line across the full page width, left to right.

        pdfplumber's first row sometimes covers only the last header cell ('2024') while
        '2022 2023' sit on the same line outside it.
        """
        r0 = table.rows[0].bbox
        words = page.crop((0, max(0, r0[1] - 2), page.width, min(page.height, r0[3] + 2))).extract_words(x_tolerance=1)
        return [w["text"] for w in sorted(words, key=lambda w: w["x0"]) if _YEAR.fullmatch(w["text"])]

    with pdfplumber.open(str(meta.path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            accepted = []  # (top, bottom, rows, years on the first line)
            for t in sorted(page.find_tables(), key=lambda t: t.bbox[1]):
                rows = [[clean_cell(c) for c in r] for r in t.extract() if r]
                rows = normalize_pdf_table([r for r in rows if any(r)])
                if _is_useful_table(rows) and (not accepted or t.bbox[1] >= accepted[-1][1]):
                    accepted.append((t.bbox[1], t.bbox[3], rows, line_years(page, t)))

            cursor = 0.0
            for top, bottom, rows, years in accepted:
                body = text_of(page, cursor, top)
                if body:
                    elements.append(Element("text", pno, body, order=len(elements)))
                in_first_row = [c for c in rows[0] if _YEAR.fullmatch(c.strip())]
                if in_first_row:
                    if len(years) > len(in_first_row):  # partial header row: rebuild it from the full line
                        rows = place_years(rows[1:], years) or rows
                else:
                    found = year_header(body)  # header line sits above the detected table
                    if found:
                        rows = place_years(rows, found[1], found[0]) or rows
                elements.append(Element("table", pno, rows_to_markdown(rows), rows, len(elements)))
                cursor = bottom
            body = text_of(page, cursor, page.height)
            if body:
                elements.append(Element("text", pno, body, order=len(elements)))
    return elements


def parse_document(meta: DocMeta) -> list[Element]:
    return parse_docx(meta) if meta.path.suffix.lower() == ".docx" else parse_pdf(meta)
