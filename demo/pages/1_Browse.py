"""Browse page — pick corpus, pick document, view raw content + metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from lib import corpus, artifacts, viewers  # noqa: E402

st.set_page_config(page_title="Browse · InfoGuide", layout="wide")
st.title("Browse Corpus")

# --- Selection controls (sidebar) -------------------------------------------
st.sidebar.header("Document selection")

corpus_choice = st.sidebar.radio(
    "Corpus",
    options=["synthetic", "real"],
    format_func=lambda v: "Synthetic (data/synthetic/raw)" if v == "synthetic" else "Real (data/raw)",
    key="corpus_choice",
)

if corpus_choice == "synthetic":
    docs = corpus.list_synthetic()
else:
    docs = corpus.list_real()

if not docs:
    st.warning(f"No documents found for corpus '{corpus_choice}'.")
    st.stop()

# Source filter (real corpus only)
if corpus_choice == "real":
    sources = corpus.list_sources(docs)
    selected_sources = st.sidebar.multiselect(
        "Source folders",
        options=sources,
        default=sources,
    )
    docs = [d for d in docs if d.source in selected_sources]
    if not docs:
        st.warning("No documents match the selected source filters.")
        st.stop()

# Persist selection across pages via session_state
prev_id = st.session_state.get("selected_doc_id")
default_idx = next((i for i, d in enumerate(docs) if d.doc_id == prev_id), 0)

doc = st.sidebar.selectbox(
    "Document",
    options=docs,
    index=default_idx,
    format_func=lambda d: d.label,
    key="selected_doc_obj",
)
st.session_state["selected_doc_id"] = doc.doc_id
st.session_state["selected_corpus"] = doc.corpus

# --- Main pane ---------------------------------------------------------------
left, right = st.columns([3, 1], gap="large")

with right:
    st.subheader("Metadata")
    meta = {
        "doc_id": doc.doc_id,
        "corpus": doc.corpus,
        "source": doc.source or "—",
        "filename": doc.filename,
        "format": doc.path.suffix.lstrip(".").lower() or "—",
        "size": viewers.file_size_human(doc.path),
        "path": str(doc.path.relative_to(corpus.config.PROJECT_ROOT)),
    }

    # Enrich with extracted/processed metadata when available (real docs).
    extracted = artifacts.get_extracted(doc.doc_id)
    if extracted:
        for k in ("language", "publication_date", "retrieval_date", "license", "simulated_banking_role"):
            v = extracted.get(k)
            if v:
                meta[k] = v
        eq = extracted.get("extraction_quality") or {}
        if eq:
            meta["extraction_status"] = eq.get("status", "—")

    processed = artifacts.get_processed(doc.doc_id)
    if processed:
        ps = processed.get("processed_stats") or {}
        if ps:
            meta["word_count"] = ps.get("word_count", "—")
        meta["sentence_count"] = processed.get("sentence_count", "—")

    for k, v in meta.items():
        st.markdown(f"**{k}**: {v}")

    st.caption(
        f"Showing {len(docs)} document(s) in '{corpus_choice}' corpus"
        + (" after source filter" if corpus_choice == "real" else "")
        + "."
    )

with left:
    st.subheader(doc.label)
    viewers.render_raw_document(doc.path, height=720)

st.divider()
st.caption(
    "Selection persists in session state (`selected_doc_id`, `selected_corpus`) — "
    "open **Pipeline View** to inspect this document's stage outputs."
)
