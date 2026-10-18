# FinSight AI — Technical Design (v0.1)

Table-aware RAG agent over annual reports, with a chatbot UI. Track 2 (LangChain + LangGraph), per the Project Handout.

## 1. Framework (from the handout)

| Field | Decision |
|---|---|
| Use case | Analysts ask factual and comparative questions ("What was AWS operating income in 2024?", "How did Microsoft's cloud revenue change FY23→FY25?") through a chat UI, with answers cited to document, page and table. |
| Corpus | 9 files, 3 companies × 3 years: Microsoft annual reports FY23–25 (`.docx`, 69–74 tables each), Amazon annual reports 2023–25 (`.pdf`), Alphabet 10-Ks 2023–25 (`.pdf`). English. Public filings; source of truth is each company's investor-relations site. |
| Ingestion + cleaning | One-off batch ingest with a parser per format. Strip headers/footers/page numbers, repair hyphenation, normalise whitespace and unicode, keep tables as structured objects (never flattened into prose). Tag every element with `company, fiscal_year, doc_type, section, page`. |
| Freshness | Static corpus. Re-ingest is idempotent (content-hash per file), so a new filing is "drop file, run `ingest`". No SLA needed for the demo. |
| Chunking + embedding | Text: **fixed-size vs semantic** (compared, see §5). Tables: dedicated table chunks (§3). Embeddings via OpenAI (`text-embedding-3-small`, 1536-dim, 8K token limit) (model chosen to match chunk size, ~512 tokens). |
| Retrieve | Hybrid: dense + BM25, fused with RRF, metadata pre-filter (company/year), then cross-encoder rerank. Top-k 30 → rerank → top 6. |
| "I don't know" path | Designed first: if the best reranker score is below a threshold, or the grader finds the context doesn't support the answer, the agent replies "I could not find this in the provided filings" and shows what it did find. |

## 2. Architecture

```
DataSource/ ──► Parsers ──► Cleaner ──► Element stream (text blocks + tables, with metadata)
                                              │
                         ┌────────────────────┴───────────────────┐
                         ▼                                        ▼
                 Text chunkers (A: fixed, B: semantic)     Table chunker (§3)
                         └────────────────────┬───────────────────┘
                                              ▼
                          Embeddings ──► Vector store  +  BM25 index
                                              │
 UI (chat) ◄──► API ◄──► LangGraph agent ─────┘
```

**LangGraph agent** (stateful graph, the "agent" part):
1. `analyze_query` — extract companies, fiscal years, metric, and whether it is a table lookup, narrative, comparison, or out of scope. Build metadata filter.
2. `decompose` — comparisons / multi-year questions split into one sub-query per (company, year).
3. `retrieve` — hybrid search per sub-query with the filter.
4. `rerank` — cross-encoder, keep top 6, apply refusal threshold.
5. `grade` — is the context sufficient? If not, rewrite the query once and retry; else go to `refuse`.
6. `generate` — answer only from context, cite `[doc, p., table]`, show the source table as markdown; compute simple deltas/ratios in code, not in the LLM.
7. `refuse` — the explicit "not found" response.

## 3. Table-awareness (the core differentiator)

Plain text splitters destroy financial tables: row labels get separated from column headers, units ("in millions") and years are lost. Approach:

- **Extraction**: DOCX via `python-docx` (native table objects). PDF via `pdfplumber`/PyMuPDF table detection, with a fallback to layout text for borderless tables (10-K tables often have no ruling lines). Validate by checking extracted-table counts per doc against a manual spot check.
- **Normalise** each table to a header-aware grid: merged cells resolved, multi-row headers joined, `$`/`%`/parentheses-negatives handled, numbers kept as written plus a parsed value in metadata.
- **Representation** — each table yields:
  1. a **whole-table chunk** (markdown, prefixed with caption, section heading, units, company, year, page) if it fits ~800 tokens;
  2. otherwise **row-group chunks**, each repeating the header row and caption so every chunk is self-contained;
  3. a short **generated description** ("Amazon segment operating income by segment, 2022–2024, $ millions") which is what gets embedded for dense search, while the markdown table is what the LLM sees (summary-embedding / raw-table-return pattern).
- BM25 indexes the raw table text so exact labels and figures ("Intelligent Cloud", "AWS") match lexically.
- Tables are never split mid-row and never merged with neighbouring prose.

## 4. Stack

- Python 3.13, LangChain + LangGraph, LangSmith (optional) for traces.
- Vector store: Qdrant local or Chroma (embedded, no server) — default **Chroma** for zero-setup; BM25 via **`bm25s`** (pure Python, fast, saves/loads its index to disk). Tokenization is custom: lowercase, keep numbers like `$1,234.5` and `12.4%` whole, light stopword removal, no stemming, table markup characters dropped. One BM25 index per chunking strategy, using the same chunk IDs as the vector store; company/year filters are applied to BM25 candidates too.
- Models: OpenAI API for embeddings and generation. OpenAI has no reranker, so rerank is either LLM-based (chat model scores each candidate) or a local `bge-reranker` cross-encoder; both are compared in the rerank analysis.
- API: FastAPI with streaming. UI: **Streamlit** chat (fast to build, supports tables and expandable citations); can swap to a React front-end later since the API is separate.

### Deployment split: build locally, serve on Replit

- **Local (VS Code):** run ingestion, chunking, embedding and indexing. Output is a prebuilt artifact: `chroma_db/` plus the saved `bm25s` index directory and a chunk metadata file. Pin the `bm25s` version in `requirements.txt` so the saved index loads identically on Replit.
- **Embeddings are part of the repo.** Committed artifacts: `index/chunks/*.jsonl`, `index/chroma_db/` (holds the chunk vectors), the `bm25s` index and `index_manifest.json`. The LangChain embedding cache (`index/embeddings_cache/`, `LocalFileStore`, one file per text) stays local and is git-ignored; it only saves API cost on re-runs. Expected size is tens of MB (about 1536 floats per chunk, for both chunking strategies). Cloning the repo gives a working retrieval system without an API call or the source filings.
- **Replit:** ships only the app code and the prebuilt index, and opens Chroma read-only. It does **not** include `DataSource/` or the ingest code path, and never re-embeds the corpus.
- At query time Replit only embeds the user's query, runs retrieval and rerank, and calls the generator, all through the provider API, so no local models are needed and memory stays low.
- The index must be built with the same embedding model that Replit uses for queries. Record model name and dimension in an `index_manifest.json` and have the app check it on startup.
- LLM API key goes in Replit Secrets, never in the repo.
- Streamlit calls the agent directly in-process on Replit (no separate FastAPI service) to keep it to one process. The agent stays in a module so FastAPI can wrap it later.
- Keep `chroma_db/` small enough to commit or upload (expected tens of MB). If Replit's SQLite is too old for Chroma, use the `pysqlite3-binary` shim.

## 5. Evaluation and the two required comparisons

**Gold set** (~40 questions, written by hand with source page): table lookups, narrative, cross-year, cross-company, and ~6 unanswerable ones to test refusal.

Experiment grid (same questions throughout):

| Axis | Variants |
|---|---|
| Chunking | fixed-size (e.g. 512 tok / 64 overlap) vs semantic (embedding-similarity breakpoints) |
| Table handling | naive (tables as text) vs table-aware |
| Retrieval | dense vs BM25 vs hybrid |
| Rerank | off vs on |

Metrics: Hit@k, MRR, nDCG@k (retrieval, against gold pages/tables); answer correctness incl. exact numeric match; faithfulness/groundedness (LLM judge); refusal precision/recall; latency and cost.
Output: the **chunking comparison report** and **reranking impact analysis** the handout requires.

## 6. Repo layout

```
DataSource/            raw filings (untracked)
src/finsight/
  ingest/   parsers, cleaning, table extraction
  chunking/ fixed.py, semantic.py, tables.py
  index/    embeddings, vector store, bm25
  retrieve/ hybrid, rerank
  agent/    LangGraph graph + prompts
  api/      FastAPI
eval/       gold_questions.jsonl, run_eval.py, reports/
ui/         app.py (Streamlit)
docs/       this file, comparison report
```

## 7. Build plan

1. **Parse & inspect** — extract text + tables from all 9 files; report table counts, eyeball samples. *(biggest risk: PDF tables)*
2. **Chunk** — both text strategies + table chunker; stats per strategy.
3. **Index & retrieve** — embeddings, Chroma, BM25, RRF, metadata filters.
4. **Gold set + eval harness** — build before tuning, so every change is measured.
5. **Rerank** — add, measure delta.
6. **LangGraph agent** — graph, citations, refusal path.
7. **UI** — Streamlit chat wired to the API.
8. **Reports + demo** — comparison report, reranking analysis, recording.

## 8. Risks

- PDF table extraction on borderless 10-K tables → mitigated by fallback + spot-check + a chunk viewer.
- Fiscal-year mismatch: Microsoft FY ends June, Amazon/Alphabet December; store fiscal year and period end in metadata, and state it in answers.
- Same metric restated across years: prefer the latest filing's figure unless the user asks for "as originally reported"; cite which.
- LLM arithmetic errors → compute in code.
- Gold-set quality drives the whole comparison; write it by hand, not LLM-generated.

## 9. Open questions

1. Rerank backend for Replit: LLM-based (light on memory, adds latency and cost) vs local cross-encoder (heavier install). Default to LLM-based.
2. Streamlit acceptable for the UI, or is a custom web front-end wanted?
3. Is `DataSource/` allowed in the GitHub repo (≈10 MB), or should it be git-ignored with a download note? (Replit doesn't need it either way; only the built index.)

## 10. LangChain / LangGraph usage (decision: use the framework wherever it has the capability)

| Piece | Implementation |
|---|---|
| Parsing, cleaning, section tracking, table cleanup | Own code (`pdfplumber`, `python-docx`); LangChain has no equivalent for borderless financial tables |
| Fixed-size chunks | `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (512 / 64) |
| Semantic chunks | `SemanticChunker` (`langchain_experimental`) with `OpenAIEmbeddings`; min/max token limits added on top |
| Tables | Table chunks built from our parsed tables; retrieval via the `MultiVectorRetriever` pattern (description embedded, full markdown table returned) |
| Embeddings / vector store | `OpenAIEmbeddings`, `Chroma` |
| Sparse + fusion | BM25 retriever (`bm25s`) + `EnsembleRetriever` (RRF); custom retriever wrapper if `bm25s` is not supported directly |
| Rerank | `ContextualCompressionRetriever` with an LLM reranker (`LLMListwiseRerank`) |
| Agent | LangGraph state machine (§2) |
| Tracing / evals | LangSmith (optional) |

Chunks are `langchain_core.documents.Document` objects with metadata, converted to/from the `Chunk` record for JSONL storage.
