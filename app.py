"""
Phase 8 UI: a Streamlit front end over the same pipeline the CLI uses
(src/pipeline.py, src/ingest.py) — no logic is duplicated here, this is just
a presentation layer. Upload documents, see what's ingested, ask questions,
and see citations + faithfulness score + timing for each answer.

Usage:
    streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import ingest_document  # noqa: E402
from pipeline import answer_question, get_collection  # noqa: E402

st.set_page_config(page_title="Multi-Document Audit", page_icon="📄", layout="wide")
st.title("📄 Multi-Document Audit & Grounded RAG")
st.caption(
    "Upload documents, ask questions, and get grounded answers with citations and a faithfulness score — "
    "built incrementally as a learning project (see ROADMAP.md)."
)


def get_ingested_sources() -> dict[str, list[int]]:
    collection = get_collection()
    if collection.count() == 0:
        return {}
    data = collection.get()
    by_source: dict[str, list[int]] = {}
    for meta in data["metadatas"]:
        by_source.setdefault(meta["source"], []).append(meta["page"])
    return by_source


with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf documents", type=["txt", "pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Ingest uploaded files"):
        with st.spinner("Chunking and embedding..."):
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)
                # Store under the original filename so citations show the real name.
                target_path = tmp_path.with_name(uploaded_file.name)
                tmp_path.rename(target_path)
                num_chunks = ingest_document(target_path)
                st.success(f"{uploaded_file.name}: {num_chunks} chunks stored")

    st.divider()
    sources = get_ingested_sources()
    if sources:
        st.subheader(f"{len(sources)} document(s) ingested")
        for name, pages in sorted(sources.items()):
            st.text(f"• {name} ({len(pages)} chunks, pages {min(pages)}-{max(pages)})")
    else:
        st.info("No documents ingested yet — upload one above.")

    st.divider()
    scope_to_selection = st.checkbox("Scope question to selected documents only")
    selected_sources = st.multiselect("Documents", options=sorted(sources.keys())) if sources else []

question = st.text_input("Ask a question about the ingested documents", placeholder="e.g. What's the cure period in the vendor contract?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not get_ingested_sources():
        st.error("No documents ingested yet. Upload one in the sidebar first.")
    elif not question.strip():
        st.error("Type a question first.")
    else:
        with st.spinner("Retrieving and generating..."):
            try:
                result = answer_question(
                    question,
                    sources=selected_sources if (scope_to_selection and selected_sources) else None,
                )
            except RuntimeError as e:
                st.error(str(e))
                result = None

        if result:
            st.subheader("Answer")
            st.write(result["answer"])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sources")
                for i, meta in enumerate(result["metadatas"], 1):
                    st.text(f"[{i}] {meta['source']}, page {meta['page']}")

            with col2:
                st.subheader("Faithfulness")
                score = result["faithfulness_score"]
                if score is None:
                    st.warning("Could not score this answer.")
                else:
                    st.metric("Score", f"{score:.2f}")
                    if score < 0.8:
                        st.warning("Low confidence — verify against the source documents directly.")
                if result["unsupported_claims"]:
                    st.caption("Unsupported claims flagged:")
                    for claim in result["unsupported_claims"]:
                        st.caption(f"- {claim}")

            t = result["timing_ms"]
            tok = result["tokens"]
            st.caption(
                f"Timing — embed {t['embed']}ms · retrieve {t['retrieve']}ms · generate {t['generate']}ms · "
                f"evaluate {t['evaluate']}ms · total {t['total']}ms · tokens {tok['input']} in / {tok['output']} out"
            )
