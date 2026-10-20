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
| 19 | 6 | Microsoft FY2025 DOCX tables fragmented, header years over the wrong columns | `normalize_docx_table()`: merge `$ )`, align header labels to value columns, drop spacer columns | `parsers.py` |

## Step 3 — Index and retrieve (2026-10-19, done)

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

**Final build:** after shutting down all notebook kernels, `build_all()` ran cleanly: `fixed` 2,035 and `semantic` 1,893 vectors, `index/chroma_db/` 71 MB with exactly 2 segment folders, `chroma.sqlite3` 52 MB (GitHub warns above 50 MB, blocks at 100 MB). Same top hits for the six test questions. A `PermissionError` on the first attempt came from a kernel still holding the Chroma files open; shutting down all kernels resolved it.

**Carried forward:** Intelligent Cloud's main segment table is not in the top 5 (step 5, rerank); company/year filters are passed by hand (step 6, agent); the six questions are spot checks, not measurements (step 4, gold set).

## Step 4 — LangGraph agent (2026-10-20, done)

**Model choice** (discussed before building)
| Role | Model | Reason |
|---|---|---|
| Query analysis + context check | `gpt-4.1` (`ANALYSIS_MODEL`) | Non-reasoning, fast, `temperature=0` for consistent extraction and yes/no judgments |
| Answer | `gpt-5.4`, reasoning effort medium (`ANSWER_MODEL`) | Reading the right table column and year, comparing and computing changes benefit from reasoning; the prompt also makes it quote input figures before any calculation |
| Rerank (step 7) | `gpt-4.1-mini` default (`RERANK_MODEL`) | Runs on ~30 chunks per question, so speed and cost matter |
| Embeddings | `text-embedding-3-small` | Unchanged; the index was built with it |

Considered: `gpt-4o-mini` (cheapest), `gpt-5.4` (reasoning; checked available on the account via the models list), a split with `gpt-5.4` for answers only. Chosen: `gpt-4.1` for analysis and the context check, `gpt-5.4` for answers; the step 6 evaluation can compare answer models. `GENERATION_MODEL` replaced by `ANALYSIS_MODEL` / `ANSWER_MODEL` / `RERANK_MODEL`. `llm.chat_model()` sends `temperature=0` to non-reasoning models and `reasoning_effort` to reasoning models (`gpt-5*`, `o*`), so any model can be swapped in `.env`.

**Graph** (`agent/graph.py`, LangGraph `StateGraph`)
```
START -> analyze --(out of scope)--> refuse -> END
            |
         decompose -> retrieve -> rerank -> grade --(yes/partial)--> answer -> END
                          ^                   |
                          +---(no, 1 retry)---+--(no, retried)--> refuse -> END
```
| Node | What it does |
|---|---|
| `analyze` | Structured output (`QueryAnalysis`): in_scope, companies (brands mapped to parent: AWS → Amazon, Google Cloud → Alphabet), fiscal years 2023–2025, question type, standalone search query without years. Uses chat history for follow-ups |
| `decompose` | One sub-query per (company, year), up to 9; no filter on an axis the question doesn't specify |
| `retrieve` | Hybrid search (fixed chunks) per sub-query, merged round-robin so each company/year is represented, deduplicated on `chunk_id`. On retry, adds the grader's "missing" text to the query and keeps earlier hits |
| `rerank` | Pass-through, keeps the top 12 (step 7 plugs in the LLM reranker here) |
| `grade` | Structured output (`Grade`): answerable yes / partial / no, plus what is missing |
| `answer` | Numbered excerpts with source labels; rules: excerpts only, cite every figure `[n]`, match table values to column years, quote inputs before calculating, Microsoft fiscal year ends June 30, state what is missing for partial answers. Citations parsed from the `[n]` markers |
| `refuse` | Out of scope, or not found after one retry; fixed messages, no LLM call |

`ask(question, history, strategy)` returns the final state: answer, citations, route (`answered` / `not_found` / `out_of_scope`) and the trace (analysis, sub-queries, docs, grade).

**Checks before running:** graph compiles (Mermaid diagram printed); `decompose` and `after_grade` routing tested without API calls; reasoning-model detection tested.

**Notebook 04:** single lookups, comparisons (decomposed), narrative, three refusal cases (other company, year outside the corpus, fact not in filings), a follow-up using chat history, and the context of one answer.

**First run of notebook 04** (answers checked against the filings)
| # | Question | Result | Verdict |
|---|---|---|---|
| 1 | AWS operating income 2024 | $39,834M, segment table cited | Correct |
| 2 | Google Cloud revenue 2024 | $43,229M, two tables agree | Correct |
| 3 | Microsoft employees | ~228,000 as of June 30, 2025 (plus FY2023–24) | Correct |
| 4 | AWS vs Google Cloud revenue 2024 | $107,556M vs $43,229M, difference $64,327M with calculation shown | Correct |
| 5 | Intelligent Cloud FY2023 → FY2024 | +$17.5B (+20%); says absolute totals not in excerpts | Correct but incomplete: segment table (105,362 vs 87,907) not retrieved; rerank (step 7) |
| 6 | Alphabet AI-regulation risks (2025) | Structured, cited summary | Good |
| 7 | Apple revenue 2024 | Refused, out of scope | Correct |
| 8 | Amazon revenue 2019 | Refused, routed as out of scope | Outcome correct; "not in the filings" would be the more precise label |
| 9 | AWS CEO's dog | Refused | Correct |
| 10 | Follow-up "And what was it in 2023?" | Resolved to Google Cloud 2023: $33,088M | Value correct; added a stray remark about the previous turn's year |

All decompositions, filters and brand mappings (AWS → Amazon, Google Cloud → Alphabet) were right; every answer graded "yes" on the first attempt.

**Fixes:** answer prompt rules 8–9: answer only the latest question (history is for resolving references, not for commentary), and no closing offers ("If you want, I can…").

**Open:** context noise (about half of the 12 excerpts loosely related, e.g. shareholder-letter paragraphs) and the Intelligent Cloud table miss are for step 7 (rerank).

## Step 5 — Streamlit UI (2026-10-20, in progress)

**Code:** `ui/app.py` (Streamlit 1.66, installed in `.venv`; already in `requirements.txt`). Run with `streamlit run ui/app.py`.

| Part | Behaviour |
|---|---|
| Startup | `st.cache_resource` checks the index manifest, opens Chroma + bm25s for both strategies once, compiles the agent; clear error if `OPENAI_API_KEY` or the index is missing |
| Chat | `st.chat_input`; each question calls `ask(question, history, strategy)` with the last 6 messages for follow-ups |
| Answer | Markdown answer, then a **Sources** expander: each citation `[n]` with its label (company, doc, fiscal year, section, page) and the excerpt the model saw (tables render as markdown tables) |
| Trace (toggle) | Route, extracted companies/years/type, search query, sub-queries, excerpts retrieved, context-check verdict |
| Sidebar | Corpus and fiscal-year note, chunking strategy (fixed / semantic), five example questions, clear chat, model names |

**Checks:** headless smoke test with Streamlit `AppTest` (renders the page and loads the index, no questions asked, no LLM calls): no exceptions or errors; sidebar, strategy radio, 6 buttons and chat input present.

**Note:** querying Chroma modifies `index/chroma_db/chroma.sqlite3` on disk even without writes; left uncommitted, to resolve before deployment.

## Step 6 — Gold set and evaluation (2026-10-20, in progress)

**Gold set:** `eval/gold_questions.jsonl`, 44 questions. Answers were taken from the parsed filings (table rows printed from `index/chunks/fixed.jsonl`), not from memory.

| Type | Count | What it tests |
|---|---|---|
| lookup | 24 | One figure from one company and year: segment and total revenue, operating and net income, EPS, free cash flow, capex, R&D, employees; includes a FY2022 comparative (Microsoft revenue) that only appears in later filings |
| comparison | 8 | Across companies (AWS vs Google Cloud; net income of all three) or years (operating income, R&D, Intelligent Cloud); one growth-rate question |
| narrative | 5 | Explanations and definitions: AWS operating income drivers, AI-regulation risks, Intelligent Cloud growth drivers, free cash flow definition, Other Bets |
| followup | 2 | Chat history needed to resolve the question |
| refusal | 5 | Other companies, a year outside the filings, a forecast, a fact not in the filings |

**Record format:** `id`, `type`, `question`, optional `history`, `answer` (human-readable expected answer), `figures` (strings the answer must contain), `keywords` (narrative), `expect_refusal`, and `evidence`: a list of `{docs, contains}` rules. A chunk is relevant to an evidence item if it belongs to one of `docs` and contains every string in `contains`. Rules instead of chunk ids keep the gold set valid for both chunking strategies and across re-chunking.

**Verification:** every evidence item matches at least one chunk in both the fixed and semantic chunk sets (0 unmatched).

**Cost of evaluation:** building the gold set used no API calls. Retrieval metrics need only the question embeddings (cached). A full answer run is about 3–4 chat calls per question (~20K tokens), so full runs are limited to the final configurations; answers are scored with rule checks, not an LLM judge.

**Found while building:** Microsoft FY2025 DOCX tables are fragmented (`| Revenue |  | $ | 281,724 |  |  | $ | 245,122 |`, headers shifted); the DOCX path does not apply the `$` merging and empty-column cleanup used for PDFs. Figures are intact (gold set unaffected) but the tables are harder to read.

**Fix (DOCX tables):** cause was twofold: `$` and `)` in their own cells with empty spacer columns, and merged header cells collapsing to fewer cells than the data rows, so `2025` sat over a spacer column. `normalize_docx_table()` merges symbol cells, re-places header labels over the columns that hold values (when the label count matches), and drops empty columns. Verified on one parse: FY2025 tables wider than 6 columns fell from 58 of 63 to 15 (the rest are genuinely wide: share repurchases, dividends, stock performance, fair-value tables); `| Revenue | $281,724 | $245,122 | 15% |` under `2025 | 2024 | Percentage Change`; FY2023 and FY2024 unchanged.

**Regenerated:** notebook 02 re-chunked (fixed 2,027 / semantic 1,885 chunks; 498 tables, down from 506 because cleaned Microsoft tables fit in fewer row groups). Gold evidence still matches in both sets (0 unmatched). Notebook 03's `build_all()` failed again with `PermissionError`: a second kernel (notebook 04, whose agent queries had opened Chroma) still held the files. Stopped the project's kernel processes and rebuilt from the terminal: `fixed` 2,027 and `semantic` 1,885 vectors, 71 MB, 2 segment folders, `chroma.sqlite3` 51.8 MB; manifest updated; hybrid search returns the clean FY2025 table. `build_all()` now raises a clearer error naming the likely holders (another kernel or Streamlit). Notebook 03's saved output still shows the failed run.

**Retrieval eval harness** (`src/finsight/evaluation/retrieval.py`, `notebooks/05_retrieval_eval.ipynb`)
- Runs the 39 answerable gold questions through the agent's own `analyze → decompose → retrieve → rerank` steps, so it scores the real pipeline including company/year extraction. Grid: strategy (fixed / semantic) × mode (dense / BM25 / hybrid); the agent's `retrieve` node now reads a `mode` from state (default hybrid).
- Relevance: a chunk satisfies an evidence item if it comes from one of the item's filings and contains all its strings.
- Metrics at k = 3, 6, 12 (the agent passes 12 chunks to the answer step): hit@k, recall@k (share of evidence items found), all@k (every item found), MRR.
- Query analysis results cached in `eval/cache/analysis.json` (keyed by question, history and model), so the grid makes 39 LLM calls once. Per-question results saved to `eval/reports/retrieval_<timestamp>.csv`.
- Scoring checked on a toy case (wrong filing not counted; comparison needs both items for all@k).

**First retrieval eval run** (39 questions, `eval/reports/retrieval_20261020_2154.csv`)
| Strategy | Mode | hit@3 | hit@6 | hit@12 | recall@12 | all@6 | MRR |
|---|---|---|---|---|---|---|---|
| fixed | BM25 | 0.718 | 0.949 | 0.974 | 0.949 | 0.923 | 0.538 |
| fixed | dense | 0.462 | 0.641 | 0.795 | 0.765 | 0.615 | 0.410 |
| fixed | hybrid (0.5/0.5) | 0.615 | 0.769 | 0.949 | 0.927 | 0.718 | 0.491 |
| semantic | BM25 | 0.718 | 0.923 | 0.974 | 0.949 | 0.897 | 0.574 |
| semantic | dense | 0.436 | 0.667 | 0.744 | 0.726 | 0.615 | 0.334 |
| semantic | hybrid (0.5/0.5) | 0.590 | 0.795 | 0.949 | 0.927 | 0.744 | 0.448 |

Findings:
- **BM25 > hybrid > dense.** With fixed chunks, 7 questions (L07, L11, L15, L18, L21, L23, C08) have evidence in BM25's top 6 but not hybrid's; none the other way round. Dense misses most table lookups: table descriptions carry labels but not figures, and MD&A text outranks them. The 50/50 fusion lets dense push good BM25 hits down.
- **Fixed vs semantic roughly tie** (hit@12 0.949 both). Fixed ranks narrative evidence higher (hit@3 1.0 vs 0.6); semantic slightly better on comparisons at top 3.
- **Comparisons are weakest** (all@6 0.50): with 2–3 companies or years the 12-chunk budget is split and one company's table drops out (C01, C05, C08).
- **Query analysis** extracted the right companies and years for every question except L24 (FY2022 revenue), marked out of scope because 2022 is outside 2023–2025, although the FY2023/FY2024 filings report it as a comparative.
- **Caveat:** gold questions mostly use the filings' own wording ("purchases of property and equipment"), which favours BM25. Paraphrased questions ("capex", "cloud sales") would favour dense; worth adding a few.

Changes:
- Analysis prompt: years before 2023 are in scope (prior-year comparatives) with no year filter; forecasts out of scope. The analysis cache key now includes the prompt, so prompt changes re-run the analysis.
- Hybrid fusion weights configurable: `HYBRID_WEIGHTS` (default 0.5,0.5) in config, `weights` in agent state, `weights=` in `run_retrieval_eval`. Notebook 05 §5 sweeps 0.5/0.5, 0.3/0.7, 0.2/0.8 against BM25 only.

**Paraphrased questions added:** 17 gold questions (`P01`–`P17`, `variant: "paraphrase"`, `paraphrase_of: <original id>`) reword existing questions in everyday terms instead of the filings' labels: capex, EPS, profit, R&D, headcount, "Google's cloud unit", "cloud and server segment", "cash left after capex", "experimental businesses". Each keeps its original's evidence rules and expected figures. Originals unchanged so earlier runs stay comparable. Gold set now 61 questions (56 with evidence); all evidence still matches in both chunk sets. Notebook 05 §3a compares the 17 paraphrases with their 17 originals by strategy and mode.

**Second retrieval eval run** (56 questions incl. paraphrases; `retrieval_20261020_2224.csv` + weight sweep files). L24 now in scope.

Fixed chunks, 17 paraphrases vs the same 17 originals:
| Mode | hit@6 original | hit@6 paraphrased | hit@12 paraphrased |
|---|---|---|---|
| BM25 | 1.000 | 0.235 | 0.471 |
| dense | 0.647 | 0.588 | 0.647 |
| hybrid 0.5/0.5 | 0.765 | 0.353 | 0.588 |
| hybrid 0.2/0.8 | 1.000 | 0.235 | 0.471 |

Fixed chunks, all 39 originals: BM25 hit@6 0.949; hybrid 0.2/0.8 0.897; 0.3/0.7 0.821; 0.5/0.5 0.744; dense 0.641.

Findings:
- BM25 collapses on everyday wording (capex never matches "purchases of property and equipment", EPS never matches "diluted earnings per share"); dense barely drops but is mediocre either way.
- Weights alone can't fix it: BM25-heavy fusion is best for filing wording, 0.5/0.5 is best for paraphrases, and none is good on paraphrases. The gap is vocabulary, not fusion.
- L24 (FY2022) is in scope but still not retrieved: with no year filter and the year removed from the query, nothing points at the 2022 column.

Fix: the analysis prompt now rewrites `search_query` in 10-K terminology, keeping the user's term (capex → purchases of property and equipment, EPS → diluted earnings per share / diluted net income per share, profit → net income, headcount → employees, R&D, cash after capex → free cash flow, each company's cloud segment name), and keeps years before 2023 in the query. Same LLM call, no extra cost. Weight default to be chosen after re-running with this.

**Decision: larger embedding model for the index (`text-embedding-3-large`).** Reason: dense retrieval understands paraphrases (hit@6 dropped only 0.647 → 0.588) but ranks topical MD&A text above the table that holds the figure; a stronger embedding model should separate them better. Choices:
- Reduced to **1536 dimensions** (`dimensions` parameter): same index size as today (~71 MB, `chroma.sqlite3` ~52 MB) while keeping most of the large model's quality; full 3072 was considered (~90–100 MB) and rejected for repo size.
- Semantic chunk boundaries stay on `text-embedding-3-small` (`CHUNKER_EMBEDDING_MODEL`), so the fixed and semantic chunk sets are unchanged and comparable with earlier runs; only the index switches.
- Code: `cached_embeddings(model, dimensions)` passes `dimensions` only when it differs from the model's native size and includes it in the cache namespace; the vector store uses `EMBEDDING_MODEL`/`EMBEDDING_DIM`; the manifest check catches an index built with a different model.
- Order, for clean attribution: (1) re-run notebook 05 on the small model to measure query rewriting alone; (2) set `EMBEDDING_MODEL=text-embedding-3-large`, `EMBEDDING_DIM=1536` in `.env`, rebuild (notebook 03), re-run notebook 05 to measure the embedding upgrade. Re-index cost about 1.2M tokens.

**Run A (small model + query rewriting) and run B (large model at 1536 + query rewriting)**, 56 questions; reports `retrieval_A_small_rewrite_*` and `retrieval_B_large1536_rewrite_*`. Index rebuilt with `text-embedding-3-large` at 1536 (2,027 / 1,885 vectors, still 71 MB, `chroma.sqlite3` 51.8 MB); notebook kernels stopped first so the folder could be replaced.

Fixed chunks, all 56 questions:
| Mode | hit@6 A → B | all@6 A → B | MRR A → B |
|---|---|---|---|
| BM25 | 0.929 → 0.929 | 0.893 → 0.893 | 0.544 → 0.544 |
| dense | 0.696 → 0.875 | 0.661 → 0.839 | 0.438 → 0.664 |
| hybrid 0.2/0.8 | 0.875 → 0.911 | 0.804 → 0.875 | 0.515 → 0.595 |
| hybrid 0.3/0.7 | 0.821 → 0.911 | 0.732 → 0.875 | 0.538 → 0.664 |
| hybrid 0.5/0.5 | 0.786 → 0.929 | 0.714 → 0.893 | 0.523 → 0.682 |

Semantic chunks, run B: hybrid 0.2/0.8 hit@6 0.964, all@6 0.929, MRR 0.623; hybrid 0.5/0.5 0.946 / 0.893 / 0.631.
Fixed, 17 paraphrases only (hit@6): BM25 0.882 (was 0.235 before rewriting); hybrid 0.5/0.5 with large 0.941.

Findings:
- Query rewriting fixed BM25 on everyday wording (paraphrase hit@6 0.235 → 0.882).
- The large model mostly lifts dense retrieval (hit@6 0.70 → 0.88, MRR 0.44 → 0.66); BM25 is unaffected, as expected.
- With a strong dense side, 0.5/0.5 hybrid finds as much as BM25 (hit@6 0.929) and ranks it higher (MRR 0.682 vs 0.544), and is best on paraphrases. The earlier case for BM25-heavy weights came from the weak small model.
- One question ≈ 0.018 here; differences under ~0.04 are not meaningful. Fixed vs semantic remains close.
- Remaining misses at top 12 (fixed, hybrid 0.2/0.8): C01 and P14, the AWS vs Google Cloud 2025 comparison.

**Decision:** retrieval defaults stay **fixed chunks + hybrid 0.5/0.5** (`HYBRID_WEIGHTS=0.5,0.5`), now on the large-model index (`EMBEDDING_MODEL=text-embedding-3-large`, `EMBEDDING_DIM=1536` in `.env`). Notebooks 03 and 05 outputs predate this run (it was run from the terminal).

## Step 7 — Reranking (2026-10-20, done)

**Code**
| File | Contents |
|---|---|
| `retrieve/rerank.py` | `llm_rerank(question, docs, top_n)`: LangChain `LLMListwiseRerank` with `RERANK_MODEL` (`.env`: `gpt-5.4-mini`, reasoning effort low). The model reads the question with all candidates (full chunk text, so tables as markdown, not their descriptions) and returns them in relevance order. Falls back to retrieval order on any error; tops up if fewer than `top_n` come back |
| `agent/graph.py` | `rerank` node: LLM rerank down to 12 when `RERANK_BACKEND=llm`, else plain truncation. With reranking on, `retrieve` fetches `RERANK_CANDIDATES` (30) instead of 12. State keys `rerank`, `question` |
| `config.py` | `RERANK_BACKEND` (default `llm`), `RERANK_CANDIDATES` (30) |
| `evaluation/retrieval.py` | `rerank="none" | "llm"` option, recorded per row |
| notebook 05 §6 | Rerank off vs on (hybrid 0.5/0.5, both strategies), by question type, and which questions changed |

**Bug caught before running:** in `retrieve_for`, the `rerank` parameter was shadowed by importing the agent's `rerank` function under the same name, so the reranker would never have switched on. Renamed the import; verified offline with stubbed search and reranker (off: 12 candidates, no rerank call; llm: 30 candidates → reranker → 12).

**First live check (C01, AWS vs Google Cloud revenue 2025, the remaining miss):** without rerank, all@6 0 (top 2: Amazon letter text and an Alphabet table); with rerank, all@6 1, first relevant chunk at rank 1, top 2 the Amazon net sales table and the Alphabet revenue table. 3.3 s per question vs 0.3 s.

**Note:** reranking is now on by default for the agent and the UI, which adds ~3 s and one `RERANK_MODEL` call per question.

**Rerank impact** (hybrid 0.5/0.5, 56 questions; reports `retrieval_rerank_none_*`, `retrieval_rerank_llm_*`; run from the terminal, 112 rerank calls in 260 s, ~2.3 s each)
| Strategy | Rerank | hit@3 | hit@6 | all@6 | MRR |
|---|---|---|---|---|---|
| fixed | off | 0.804 | 0.929 | 0.893 | 0.682 |
| fixed | on | 0.982 | 1.000 | 1.000 | 0.963 |
| semantic | off | 0.768 | 0.946 | 0.893 | 0.631 |
| semantic | on | 1.000 | 1.000 | 0.982 | 0.943 |

Fixed chunks by type (all@6 off → on): comparison 0.70 → 1.00, lookup 0.92 → 1.00; narrative and follow-up already 1.00, MRR rose to 1.00.
Improved by rerank (fixed): C01, C06, L03, L07, P01, P14; worse: none. Semantic with rerank still misses C01 at top 12.

**Decision (final retrieval setup):** fixed chunks, hybrid 0.5/0.5, `text-embedding-3-large` at 1536, LLM rerank of 30 candidates to 12 (`gpt-5.4-mini`). Already the default for the agent and the UI.

**Caveat:** scores are near the ceiling, so this gold set no longer separates good from great configurations; further tuning needs harder questions (multi-step comparisons, rarer line items, vaguer wording).

**Coverage guarantee for multi-part questions:** decomposition and round-robin merging guarantee every company/year is among the 30 candidates, but the reranker chooses the final 12 by relevance and could drop one. Added `ensure_coverage()` in the `rerank` node: `retrieve` marks each sub-query's top chunks as protected (2 each, or 1 when there are more than 6 sub-queries, so they never exceed 12), and any protected chunk the reranker left out replaces the lowest-ranked unprotected chunk. Applied only when there is more than one sub-query; single-company questions keep the reranker's order untouched. Verified offline (a reranker that drops Alphabet entirely: Alphabet's top 2 restored; 9 sub-queries: all 9 represented in the final 12; single lookup: output identical to the reranker's) and live on the 10 multi-part gold questions (fixed, hybrid, rerank): all@6 = 1.0, first relevant chunk at rank 1 for every one.

**Cleanup:** removed 16 superseded eval reports (small-model runs, pre-rewriting runs and repeated sweeps; their numbers are recorded above and the files remain in git history). Kept the 5 reports for the final setup: `retrieval_B_large1536_rewrite_*` (mode and weight comparison) and `retrieval_rerank_{none,llm}_*`. Trimmed `eval/cache/analysis.json` from 151 to the 56 entries for the current prompt. Cleared stale outputs in notebooks 03 and 05 (they showed a failed build and small-model results); re-run to regenerate.

## Later steps (not started)

Order changed on 2026-10-19: agent and UI first, evaluation and reranking after.

| Step | Work |
|---|---|
| 4 | LangGraph agent: analyze query (company, year), decompose comparisons, retrieve (hybrid, fixed), rerank node as a pass-through, check the context is sufficient, answer with citations or refuse |
| 5 | Streamlit UI |
| 6 | Gold question set and evaluation harness (scores the full pipeline, including filter extraction) |
| 7 | Rerank (LLM-based), plugged into the existing node; before/after measurement |
| 8 | Chunking comparison report, reranking impact analysis, demo |

Until steps 6–7: refusal uses an LLM sufficiency check instead of a tuned rerank-score threshold; strategy and top-k are untested defaults (fixed chunks, hybrid).
