# Implementation Log

Running record of what was built, why, and what went wrong. Newest step last. Design lives in [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md).

## Step 0 — Planning and setup (2026-10-18)

**Done**
- Read the project handout and inspected `DataSource/`: 9 files, 3 companies × FY2023–25. Microsoft annual reports (`.docx`, native tables), Amazon annual reports and Alphabet 10-Ks (`.pdf`).
- Wrote the technical design (framework fields, LangGraph agent, table-aware chunking, evaluation grid).
- Created the project structure: `src/finsight/{ingest,chunking,index,retrieve,agent,api}`, `eval/`, `ui/`, `notebooks/`, `index/`. Added `requirements.txt`, `.gitignore`, `.env.example` and `README.md`.
- First commit pushed to `main` (`594716d`).

**Decisions**
| Decision | Choice | Reason |
|---|---|---|
| Build track | Code (LangChain + LangGraph) | Control over chunking, rerank and evals |
| LLM provider | OpenAI (key in `.env`, ~$10 credit) | Simplest; Nebius, Fireworks and Azure were considered |
| Embeddings | `text-embedding-3-small`, 1536-dim | Cheap; 8K-token limit leaves room for table chunks |
| Vector store | Chroma (embedded) | No server; works on Replit |
| Sparse retrieval | `bm25s` | Pure Python, fast, saves to disk; preferred over `rank_bm25` |
| Reranker | LLM-based by default; local cross-encoder as an option | OpenAI has no reranker; Replit memory is limited |
| Deployment | Build the index locally in VS Code and commit it; Replit only serves retrieval | Chunk embeddings are part of the repo (in Chroma); no re-embedding on Replit |
| Code layout | Logic in `src/finsight/`, notebooks call it | Reusable by the Streamlit app |

## Step 1 — Parse and inspect (2026-10-18)

**Code:** `src/finsight/config.py`, `src/finsight/ingest/parsers.py`, `notebooks/01_parse_and_inspect.ipynb`.

**Approach:** `python-docx` reads DOCX tables as native objects. `pdfplumber` finds tables and extracts text in PDFs. Output is an ordered stream of `Element(kind=text|table)` with page numbers.

**Problems found while reviewing the notebook output, and fixes**
| Problem | Fix |
|---|---|
| Amazon tables came out as one column (`AWS 90,757 107,556`) | Split rows into label + value columns; merge stray `$`, `%`, `)`; drop empty columns |
| Prose paragraphs detected as tables (Alphabet) | Require 2+ numeric cells and reject fragment-heavy cells; rejected tables are kept as plain text |
| Missing spaces in the Amazon letter (`DearShareholders:`) | `extract_text(x_tolerance=1)` |
| Table captions separated from tables | Interleave PDF text and tables by vertical position |
| Tables had no column headers (years sat above the detected box) | `year_header()` recovers `2023 \| 2024` rows from the text just above a table |

**Result:** 9 documents parse; 33–48 tables per PDF and 63–70 per Microsoft file. Committed as `638786d`.

**Known limits**
- Microsoft DOCX tables have no page numbers; citations will use section names.
- Only about half of PDF tables have year headers; the rest are probably non-year tables, not yet checked individually.
- Alphabet table count dropped from 81 to about 40 after filtering; assumed to be false positives, not fully verified.

## Step 2 — Chunking (2026-10-18 to 2026-10-19, done)

**Decision: use LangChain and LangGraph wherever they have the capability** (recorded in TECHNICAL_DESIGN §10).

Does LangChain support table-aware chunking? We checked:
- There is no built-in table-aware splitter. LangChain's own [benchmark on tables](https://blog.langchain.dev/benchmarking-rag-on-tables) says table extraction is hard and some tools miss table headers.
- Extraction is left to loaders (`pdfplumber`, PyMuPDF, Unstructured, Docling). LangChain does not fix borderless tables, year headers, units or captions.
- For retrieval, LangChain recommends the multi-vector pattern ([semi-structured RAG](https://langchain.com/blog/semi-structured-multi-modal-rag)): embed a summary of each table, return the raw table to the LLM.
- **Conclusion:** keep our parsing and table chunking; use LangChain splitters for text and `MultiVectorRetriever`-style retrieval for tables. Optional later check: compare an Unstructured or PyMuPDF loader against our parser on one Amazon and one Alphabet PDF.

**Chunk size and embedding dimension** (questions raised during review)
- `text-embedding-3-small` produces 1536-dimension vectors (a vector is the list of numbers; the dimension is its length). Every chunk, whatever its length, becomes 1536 numbers.
- Dimension and maximum input are independent properties. The hard limit on chunk size is the model's input limit (8,191 tokens), not the dimension.
- The handout tip ("a 512-token chunk on a 384-dim embedding is wasted") is a quality guideline: larger chunks need a more capable model. 1536 dims comfortably supports 512–1024 token chunks.
- Starting sizes: fixed 512 tokens with 64 overlap (~12%); semantic 150–800 tokens, no overlap; tables up to ~800 tokens, no overlap (row groups repeat the header instead). The eval sweeps fixed 256 / 512 / 1024.
- Both strategies are built and evaluated; the app ships the winner. Both indexes are kept in the repo for now so the comparison is reproducible.

**Code**
| File | Status | Contents |
|---|---|---|
| `chunking/base.py` | done | `Chunk` record, `count_tokens` (tiktoken `cl100k_base`), `make_prefix`, JSONL save/load, `is_heading`, `iter_blocks` (text blocks never cross a heading or a table; passes the lines before each table for captions) |
| `chunking/fixed.py` | done | Strategy A with LangChain `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (512 / 64), per block, page mapped from `start_index`; blocks under 25 tokens skipped (mostly table captions) |
| `chunking/semantic.py` | done | Strategy B with LangChain `SemanticChunker` (percentile 95 breakpoint) per block; adds min 150 / max 800 tokens (merge short splits, re-split long ones with the token splitter). Sentence embeddings through LangChain `CacheBackedEmbeddings` + `LocalFileStore` at `index/embeddings_cache/` (sha256 keys), so re-runs don't pay twice. Calls the OpenAI API |
| `chunking/pipeline.py` | done | `split_elements` → blocks + tables; `chunk_document` (text chunks for the chosen strategy + shared table chunks); `build_and_save` writes `index/chunks/{fixed,semantic}.jsonl` |
| `chunking/tables.py` | done | One chunk per table if ≤ 800 tokens, else row groups that repeat header, title and units. Title = last meaningful line before the table; units = "in millions…" from caption or first rows; first row treated as header unless it holds data values. Three views: `text` (markdown for the LLM), `embed_text` (rule-based description with columns and row labels, swappable for an LLM summary later), `bm25_text` (labels and figures without markdown). `strategy="shared"`: identical in both chunk sets |
| `notebooks/02_chunking.ipynb` | done (3 runs) | Parse once; check section detection; build fixed + table chunks (no API); inspect table titles, units, headers and coverage; semantic on one document, then all (API); compare token distributions |

**Dependencies added:** `langchain`, `langchain-text-splitters`, `langchain-experimental`, `langchain-classic`, `langchain-chroma`. In LangChain 1.x, `CacheBackedEmbeddings` and `LocalFileStore` live in `langchain-classic`; `semantic.py` imports from `langchain` first and falls back to `langchain_classic`.

**How tables are kept away from LangChain's splitters:** parsing already emits tables and text as separate elements (PDF text is cropped around table areas; DOCX tables are separate objects). `iter_blocks` tags each item as a text block or a table, `split_elements` puts them in two lists, and `chunk_document` sends blocks to the LangChain splitters and tables to `chunk_tables`. LangChain only ever receives plain text strings. Rejected "tables" (prose) are text on purpose; caption and year-header lines stay in the text just before a table (usually dropped by the 25-token minimum).

**Vector DB:** Chroma, embedded mode (no server), via `langchain-chroma`; one collection per strategy; chunk metadata (company, fiscal year, doc type, section, page, kind) stored for filtering. Built locally, committed, opened read-only on Replit.

**Storage layout** (all under `index/`, configurable with `INDEX_DIR`; committed with the code except the embedding cache)
| Path | Contents | Created by |
|---|---|---|
| `index/chunks/fixed.jsonl` | fixed text chunks + table chunks (one JSON record per line) | notebook 02 §3 |
| `index/chunks/semantic.jsonl` | semantic text chunks + the same table chunks | notebook 02 §5 |
| `index/embeddings_cache/` | cached OpenAI embeddings (LangChain `LocalFileStore`); **git-ignored, local only** | semantic chunking; chunk embeddings in step 3 |
| `index/chroma_db/` | Chroma vector store | step 3 |
| `index/bm25/` | `bm25s` index | step 3 |

**Decision (cache):** keep LangChain `LocalFileStore` (one file per text) and do **not** commit `index/embeddings_cache/` (added to `.gitignore`). The chunk vectors are committed inside `index/chroma_db/`. `SQLStore` (single SQLite file, `langchain-community`, needs `sqlalchemy[asyncio]`) was considered and not chosen.

**Known risks:** `is_heading` is a heuristic and needs checking on real output; text without a detected heading falls under "General".

**First run of notebook 02 (all sections, incl. semantic via the API)**
| | Fixed | Semantic |
|---|---|---|
| Text chunks | 1,508 | 1,341 |
| Median tokens | 499 | 359 |
| Range | 43–550 | 28–831 |
| Table chunks (shared) | 455 | 455 |

- Output: `index/chunks/fixed.jsonl` (10.3 MB), `semantic.jsonl` (9.6 MB). Embedding cache: 11,776 files, 375 MB (~32 KB per vector stored as text), which confirms keeping it out of git.
- Table samples correct (Alphabet revenue by type with `2022 | 2023 | 2024` header; Microsoft segment results; Amazon cash flow split into row groups with repeated header). Units found in ~65% of PDF tables and 93–94% of Microsoft tables; header row in 58–75% of PDF tables, 100% of Microsoft.

**Problems found and fixes**
| Problem | Fix |
|---|---|
| Table-of-contents lines as sections (`Item 12. … 90`) | Reject Item/Part lines ending in a page number |
| Cross-references as sections (`Part II, Item 7 of this Annual Report on Form 10-K.`) | Reject Item/Part lines ending with "." or containing "of this annual report" |
| Cover-page lines as sections (`☒ ANNUAL REPORT PURSUANT …`, `A N N U A L R E P O R T`) | Reject checkbox lines and spaced-out letters |
| `PART II` became `Part Ii` | `section_name()` keeps roman numerals and words with digits (`10-K`) |
| Amazon table titles were `(in millions)` | `find_title` skips units-only lines |
| Row-group label after the units line | Label moved to the `Table:` line |

Verified on the problem strings, then both chunk sets were regenerated.

**Second run:** section names clean (`Part I`–`Part IV`, `Item 1. Business`, `Item 1A. Risk Factors`, Microsoft `Operating Segments`, …); Amazon title now `CONSOLIDATED STATEMENTS OF CASH FLOWS (rows 31-35 of 35)`. Fixed 1,526 / semantic 1,351 text chunks.

**Mid-sentence chunks:** 30% of fixed and 14% of semantic text chunks started with a lowercase word. Cause: PDF text has a newline at every visual line wrap, and LangChain's splitter treats `\n` as a preferred split point, so it cut sentences at wraps. Fix: `Block.reflowed()` joins wrapped PDF lines with a space and keeps a line break only after `. ! ? : ;` or before a bullet (DOCX paragraphs unchanged); both chunkers use it. Tested on three documents in memory: lowercase starts fell to ~1% (Amazon 1%, Alphabet 1%, Microsoft 0%). Semantic sentences change text, so the next semantic run re-embeds them (a few cents).

**Final run (step 2 result)**
| | Fixed | Semantic |
|---|---|---|
| Total chunks | 1,984 | 1,825 |
| Text chunks | 1,529 | 1,370 |
| Table chunks (shared) | 455 | 455 |
| Median text tokens | 418 | 338 |
| Text token range | 32–536 | 28–832 |
| Starts mid-sentence | 0.5% (was 30%) | 3.4% (was 14%) |

Fixed median fell from 497 to 418 tokens because the splitter now cuts at sentence ends instead of filling to exactly 512. Files: `index/chunks/fixed.jsonl` (9.9 MB), `semantic.jsonl` (9.6 MB).

**Open / minor:** A few table chunks reach ~850 tokens (target 800); some semantic chunks are under 150 tokens because their whole block is short; 3.4% of semantic chunks still start mid-sentence (mostly from the no-overlap re-split of oversized pieces). All accepted for now.

**Lessons**
- Check chunk boundaries on real output, not just counts: the 30% mid-sentence rate only showed up when we counted chunks starting lowercase.
- PDF line breaks are layout, not structure. Reflow before any splitter that prefers `\n`.
- Changing chunk text invalidates the embedding cache (keys are text hashes), so the semantic set re-embeds after text-level fixes. Each full semantic run cost a few cents.

## All fixes so far

| # | Step | Problem | Fix | File |
|---|---|---|---|---|
| 1 | 1 | Amazon tables had one column | Split rows into label + values; merge `$ % )`; drop empty columns | `parsers.py` |
| 2 | 1 | Prose detected as tables (Alphabet) | Numeric-cell and fragment checks; rejected tables kept as text | `parsers.py` |
| 3 | 1 | Glued words in Amazon PDF text | `extract_text(x_tolerance=1)` | `parsers.py` |
| 4 | 1 | Captions separated from tables | Interleave text and tables by vertical position | `parsers.py` |
| 5 | 1 | PDF tables missing year headers | `year_header()` from the text above the table | `parsers.py` |
| 6 | 2 | TOC lines as sections | Reject Item/Part lines ending in a page number | `base.py` |
| 7 | 2 | Cross-references as sections | Reject Item/Part sentences | `base.py` |
| 8 | 2 | Cover-page checkbox / spaced-letter lines as sections | Reject them | `base.py` |
| 9 | 2 | `Part Ii` | `section_name()` keeps roman numerals and `10-K` | `base.py` |
| 10 | 2 | Amazon table titles `(in millions)` | `find_title` skips units-only lines | `tables.py` |
| 11 | 2 | Row-group label after units line | Moved to the `Table:` line | `tables.py` |
| 12 | 2 | 30% of chunks started mid-sentence | `Block.reflowed()` joins PDF line wraps | `base.py`, `fixed.py`, `semantic.py` |
| 13 | 2 | Embedding cache too big for git (11,776 files, 375 MB) | Git-ignored; chunk vectors committed via Chroma | `.gitignore` |
| 14 | 3 | 15% of PDF table chunks had collapsed rows (label + values in the first cell) | Split each such row and right-align values; dates not split | `parsers.py` |
| 15 | 3 | Rebuild skipped when ids matched but content changed | Store `embed_hash` per vector; rebuild on any mismatch | `vectorstore.py` |
| 16 | 3 | Partial header rows put the wrong year over value columns | `line_years()` reads the whole first line; `place_years()` aligns years to numeric columns | `parsers.py` |
| 17 | 3 | `(Note 12)` split into a value, adding a column | Keep a `)` that closes the label's bracket in the label | `parsers.py` |
| 18 | 3 | Chroma folder doubled (orphaned segments, growing sqlite log) | `build_all()` rebuilds into an empty folder every run | `vectorstore.py`, notebook 03 |

## Step 3 — Index and retrieve (2026-10-19, in progress)

**Approach:** both indexes are built from the same chunks with the same chunk IDs, so hybrid fusion can merge hits on `chunk_id`. Each index reads a different view of a chunk:
| | Chroma (dense) | bm25s (keyword) | LLM |
|---|---|---|---|
| Text chunk | `embed_text` (prefix + body) | `bm25_text` (same) | `text` (same) |
| Table chunk | `embed_text` (short description) | `bm25_text` (labels and figures, no markdown) | `text` (markdown table) |

Dense search matches meaning, so tables are embedded as a description; keyword search needs the exact labels and figures; the LLM needs the table structure. This is LangChain's multi-vector pattern: Chroma returns a `chunk_id`, the docstore returns the full chunk.

**Code**
| File | Status | Contents |
|---|---|---|
| `index/embeddings.py` | written, not run | `cached_embeddings()`: `OpenAIEmbeddings` + `CacheBackedEmbeddings` (moved here from `semantic.py`, now shared); query vectors cached too |
| `index/vectorstore.py` | written, not run | Chroma collection per strategy (cosine), `add_texts(embed_text, metadata, ids)` in batches of 500; skips rebuild if the collection already holds the same ids; scalar metadata (`page` 0 = DOCX); docstore `chunk_id → Document(text)` built from the committed JSONL; `index_manifest.json` with model, dimension and counts, checked on load |
| `index/bm25.py` | written, toy-tested | Tokenizer: lowercase, stopwords removed, no stemming, numbers kept whole plus comma-free form (`107,556` → `107,556`, `107556`), `$ % ( )` stripped. bm25s index saved to `index/bm25/{strategy}/` with row → chunk metadata. `BM25SRetriever` (LangChain `BaseRetriever`); filters via bm25s `weight_mask`, masked or zero-score hits dropped |
| `retrieve/hybrid.py` | written, not run | `search(query, strategy, mode, company, fiscal_year, kind, k)`; dense = `MultiVectorRetriever` (Chroma where-filter), bm25 = `BM25SRetriever`, hybrid = `EnsembleRetriever` (RRF, weights 0.5/0.5, dedup on `chunk_id`), 30 candidates each before fusion |
| `notebooks/03_index_and_retrieve.ipynb` | written, not run | Build both strategies' indexes; six test questions with company/year filters; dense vs BM25 vs hybrid side by side; top hit text; fixed vs semantic |

**Checks before running:** imports resolve on LangChain 1.x (`EnsembleRetriever`, `MultiVectorRetriever`, `InMemoryStore` from `langchain-classic`); tokenizer output checked on sample strings; `BM25SRetriever` with and without a company/year filter tested on toy data.

**Dependencies:** `bm25s` 0.3.14 installed. `bm25s` and `chromadb` pinned in `requirements.txt` because the indexes built locally are loaded on Replit.

**Filters:** company and fiscal year are passed explicitly for now; extracting them from the question is the agent's `analyze_query` step (step 6).

**First run of notebook 03**
- Built: `fixed` 1,984 vectors, `semantic` 1,825; manifest `text-embedding-3-small`, 1536-dim. Sizes: `index/chroma_db/` 72 MB, `index/bm25/` 4.9 MB.

| Question (filter) | Hybrid top hit | Verdict |
|---|---|---|
| AWS operating income 2024 (Amazon 2024) | MD&A text on operating income (p.39); segment operating income table at rank 2 | Good: the table with the AWS figure is in the top 2 |
| Google Cloud revenue 2024 (Alphabet 2024) | Revenues-by-type table (p.37) | Right table, but its rows were broken (see fix below) |
| Intelligent Cloud revenue FY2024 (Microsoft 2024) | Segment results text; a segment table at rank 3 | Partial: the main "Segment Results of Operations" table is not in the top 5; a case for reranking |
| Free cash flow 2024 (Amazon 2024) | Free cash flow section text (p.40); FCF table at rank 4 | Good |
| AI regulation risks (Alphabet 2025) | Risk Factors bullet on AI laws (EU AI Act) | Good with fixed; semantic's top hit is a less relevant risk bullet |
| Microsoft employees (Microsoft 2025) | Human Capital Resources: ~228,000 employees as of June 30, 2025 | Good |

Dense and BM25 often agree on the top hit; hybrid pulls tables up when only one retriever ranks them.

**Problem: collapsed table rows.** 15% of table chunks (about 36 Alphabet, 11 Amazon) had rows where the label and all values sat in the first cell with the other columns empty (`Google Cloud 33,088 43,229 | | |`). The step 1 row split only ran when the whole table was one column. Fix in `parsers.py`: `normalize_pdf_table` now splits each such row into label + values and right-aligns the values under the value columns; dates like `June 30, 2024` are not split. Verified: the revenues-by-type table now reads `Google Cloud | 33,088 | 43,229` under `2023 | 2024`. (Microsoft hits from the same check were false positives: date sub-headers.)

**Problem: stale vectors on rebuild.** `build_vectorstore` skipped rebuilding when chunk ids were unchanged, but the fix changes table content under the same ids. Fix: each vector stores an `embed_hash` of its `embed_text`; the collection is rebuilt when any hash differs. Unchanged texts come from the embedding cache, so a rebuild costs only the changed tables.

**Second run (after the collapsed-row fix):** 0 collapsed rows left. Table chunks rose from 455 to 506: about 50 tables were previously rejected as "not numeric" because all their numbers sat in one cell. Totals: fixed 2,035 chunks, semantic 1,893. Google Cloud's top hit became the segment table (p.89), the revenues-by-type table (p.37) is correct at rank 3.

**Problem: partial header rows (wrong years).** The segment table's header read `| | 2024 | | |` over three value columns, so `Google Cloud 26,280` (the 2022 figure) would read as 2024. The page line is `Year Ended December 31, 2022 2023 2024`, but pdfplumber's first row only covered the `2024` cell. Fixes in `parsers.py`:
- `line_years()` reads the years across the full width of the table's first line, left to right.
- `place_years()` puts each year over a column that actually holds numbers and drops empty spacer columns (used for both partial header rows and header lines above the table, replacing the old "years = columns or columns − 1" rule).
- `_split_numeric_row` no longer splits `(Note 12)` into a value `12)`; that extra column had blocked the income statement header.

Verified on one parse: Alphabet segment table and income statement now `2022 | 2023 | 2024`; revenues by type, Amazon free cash flow, net sales and tax tables unchanged and correct. Tables with a year header: 29–33 per Alphabet document, 19–21 per Amazon document. Remaining tables without one are mostly statements of equity, debt schedules and share tables whose columns are not years.

**Third run:** header fixes confirmed in the chunks and the index (Alphabet income statement p.54 and segment table p.89 both `2022 | 2023 | 2024`); no errors. All six test questions return a relevant top hit; Google Cloud's top hit is the corrected segment table.

**Problem: Chroma folder bloat.** `index/chroma_db/` grew from 72 MB to 140 MB after in-place rebuilds: 8 segment folders for 2 live collections, and `chroma.sqlite3` at 78 MB because it logs every add and delete (close to GitHub's 100 MB per-file limit). Fix: `build_all()` deletes `index/chroma_db/` and builds both collections into an empty folder every run; vectors come from the embedding cache, so this is free. Notebook 03 §1 now calls `build_all()` (restart the kernel first so no Chroma client holds the folder open).

**Status at end of 2026-10-19:** the first `build_all()` run failed with a Windows `PermissionError` because the kernel had not been restarted and still held the Chroma files open; the folder was left partly deleted. Committed everything except `index/chroma_db/`. To resume: restart the kernel, Run All in notebook 03, check the folder size (expected ~70 MB, `chroma.sqlite3` well under 100 MB), then commit `index/chroma_db/` and close step 3.

## Later steps (not started)

4 gold questions and eval harness · 5 rerank · 6 LangGraph agent · 7 Streamlit UI · 8 comparison report and demo.
