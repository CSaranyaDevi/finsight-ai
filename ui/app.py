"""FinSight AI chat UI (Streamlit).

Run locally:   streamlit run ui/app.py
On Replit:     streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8080
Needs OPENAI_API_KEY (from .env locally, from Secrets on Replit) and the committed index/ folder.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.config import ANALYSIS_MODEL, ANSWER_MODEL  # noqa: E402  (also loads .env)

st.set_page_config(page_title="FinSight AI", page_icon="📊", layout="wide")

EXAMPLES = [
    "What was AWS operating income in 2024?",
    "Compare AWS and Google Cloud revenue in 2024.",
    "How did Microsoft Intelligent Cloud revenue change from fiscal 2023 to fiscal 2024?",
    "What does Alphabet say about risks from AI regulation in 2025?",
    "How many employees does Microsoft have?",
]
HISTORY_TURNS = 6  # previous messages passed to the agent for follow-ups


@st.cache_resource(show_spinner="Loading the index…")
def load_agent():
    """Check the index manifest, open Chroma + bm25s for both strategies once, compile the graph."""
    from finsight.agent.graph import get_agent
    from finsight.index.vectorstore import check_manifest
    from finsight.retrieve.hybrid import load_index

    check_manifest()
    for strategy in ("fixed", "semantic"):
        load_index(strategy)
    return get_agent()


def excerpt_body(text: str) -> str:
    """Drop the 'Company | FY | ...' prefix line; the source label already shows it."""
    return text.split("\n", 1)[1] if "\n" in text else text


def render_sources(msg: dict) -> None:
    if not msg.get("citations"):
        return
    with st.expander(f"Sources ({len(msg['citations'])})"):
        for c in msg["citations"]:
            icon = "📋" if c["kind"] == "table" else "📄"
            st.markdown(f"**[{c['n']}]** {icon} {c['label']}")
            body = msg["excerpts"].get(c["chunk_id"], "")
            if body:
                with st.container(border=True):
                    st.markdown(excerpt_body(body))


def render_trace(msg: dict) -> None:
    t = msg.get("trace")
    if not t or not st.session_state.get("show_trace"):
        return
    with st.expander("How this was answered"):
        a = t["analysis"]
        st.markdown(
            f"- **Route:** `{t['route']}`\n"
            f"- **Companies:** {', '.join(a['companies']) or 'any'} · **Years:** {', '.join(map(str, a['fiscal_years'])) or 'any'}"
            f" · **Type:** {a['question_type']}\n"
            f"- **Search query:** `{a['search_query']}`\n"
            f"- **Sub-queries:** {len(t['subqueries'])} · **Excerpts retrieved:** {t['n_docs']}"
            + (f" · **Context check:** {t['grade']['answerable']}" if t.get("grade") else "")
        )


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg)
            render_trace(msg)


def answer(question: str, strategy: str) -> dict:
    from finsight.agent.graph import ask

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-HISTORY_TURNS:]]
    state = ask(question, history, strategy)
    docs = {d.metadata["chunk_id"]: d.page_content for d in state.get("docs", [])}
    cited = {c["chunk_id"]: docs.get(c["chunk_id"], "") for c in state.get("citations", [])}
    return {
        "role": "assistant",
        "content": state["answer"],
        "citations": state.get("citations", []),
        "excerpts": cited,
        "trace": {
            "route": state.get("route"),
            "analysis": state["analysis"],
            "subqueries": state.get("subqueries", []),
            "n_docs": len(state.get("docs", [])),
            "grade": state.get("grade"),
        },
    }


# --- sidebar ---------------------------------------------------------------

with st.sidebar:
    st.title("📊 FinSight AI")
    st.caption("Answers from annual reports and 10-K filings, with citations.")
    st.markdown(
        "**Corpus**\n"
        "- Microsoft · FY2023–FY2025 (year ends June 30)\n"
        "- Amazon · 2023–2025\n"
        "- Alphabet · 2023–2025"
    )
    strategy = st.radio("Chunking strategy", ["fixed", "semantic"], horizontal=True,
                        help="Which chunk set to search. Tables are the same in both.")
    st.session_state.show_trace = st.toggle("Show how each answer was found", value=False)
    st.markdown("**Try asking**")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = []
    st.caption(f"Models: {ANALYSIS_MODEL} (analysis) · {ANSWER_MODEL} (answers)")

# --- main ------------------------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to `.env` locally or to Secrets on Replit.")
    st.stop()

try:
    load_agent()
except Exception as e:  # missing or mismatched index
    st.error(f"Could not load the index: {e}")
    st.stop()

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    render_message(m)

question = st.chat_input("Ask about Microsoft, Amazon or Alphabet filings…") or st.session_state.pop("pending", None)
if question:
    user_msg = {"role": "user", "content": question}
    render_message(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Searching the filings…"):
            try:
                reply = answer(question, strategy)
            except Exception as e:
                reply = {"role": "assistant", "content": f"Something went wrong: `{e}`", "citations": [], "excerpts": {}}
        st.markdown(reply["content"])
        render_sources(reply)
        render_trace(reply)
    st.session_state.messages += [user_msg, reply]
