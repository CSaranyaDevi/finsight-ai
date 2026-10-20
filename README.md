# FinSight AI

Table-aware RAG agent over Microsoft, Amazon and Alphabet annual reports (FY2023–25), with a chat UI.
Design: [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then set OPENAI_API_KEY
python -m ipykernel install --user --name finsight --display-name "finsight (.venv)"
```

## Run the chat UI

```powershell
streamlit run ui/app.py
```

Opens at http://localhost:8501. Uses the committed `index/` and `OPENAI_API_KEY` from `.env`.

## Deploy on Replit

1. In Replit: **Create Repl → Import from GitHub** → this repository.
2. **Secrets** (padlock icon): add `OPENAI_API_KEY`. Nothing else is needed; the code defaults are the final, evaluated setup.
3. Click **Run**. `.replit` starts Streamlit on port 8080; the Webview shows the app (first start installs `requirements-app.txt`, a few minutes).
4. For a public URL: **Deploy → Autoscale** (build and run commands come from `.replit`).

The app serves the committed `index/` (Chroma, BM25, chunks); it never re-embeds the filings and doesn't need `DataSource/`.

## Run step by step

| Step | Notebook / code | Status |
|---|---|---|
| 1. Parse & inspect | `notebooks/01_parse_and_inspect.ipynb`, `src/finsight/ingest/parsers.py` | code ready |
| 2. Chunking (fixed, semantic, tables) | `src/finsight/chunking/` | todo |
| 3. Embeddings, Chroma, bm25s, hybrid retrieval | `src/finsight/index/`, `src/finsight/retrieve/` | todo |
| 4. Gold questions + eval harness | `eval/` | todo |
| 5. Rerank | `src/finsight/retrieve/` | todo |
| 6. LangGraph agent | `src/finsight/agent/` | todo |
| 7. Streamlit UI | `ui/app.py` | done |

Select the **finsight (.venv)** kernel in VS Code when opening a notebook. Embeddings and indexes are committed under `index/`.
