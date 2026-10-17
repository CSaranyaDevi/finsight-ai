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

## Run step by step

| Step | Notebook / code | Status |
|---|---|---|
| 1. Parse & inspect | `notebooks/01_parse_and_inspect.ipynb`, `src/finsight/ingest/parsers.py` | code ready |
| 2. Chunking (fixed, semantic, tables) | `src/finsight/chunking/` | todo |
| 3. Embeddings, Chroma, bm25s, hybrid retrieval | `src/finsight/index/`, `src/finsight/retrieve/` | todo |
| 4. Gold questions + eval harness | `eval/` | todo |
| 5. Rerank | `src/finsight/retrieve/` | todo |
| 6. LangGraph agent | `src/finsight/agent/` | todo |
| 7. Streamlit UI | `ui/app.py` | todo |

Select the **finsight (.venv)** kernel in VS Code when opening a notebook. Embeddings and indexes are committed under `index/`.
