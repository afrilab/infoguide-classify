"""Design Evaluations — render per-module design-and-evaluation reports (markdown)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import config  # noqa: E402

st.set_page_config(page_title="Design Evaluations · InfoGuide", layout="wide")
st.title("Design Evaluations")
st.caption(
    "Per-module design and evaluation reports — narrative documents that explain "
    "design decisions, trade-offs, and findings (sourced from "
    f"`{config.DOCS.relative_to(config.PROJECT_ROOT)}/`)."
)

available = {
    label: path
    for label, path in config.DESIGN_EVALUATIONS.items()
    if path.exists()
}
missing = {
    label: path
    for label, path in config.DESIGN_EVALUATIONS.items()
    if not path.exists()
}

if not available:
    st.warning("No design evaluation reports were found.")
    if missing:
        st.write("Expected files:")
        for label, p in missing.items():
            st.markdown(f"- **{label}** — `{p.relative_to(config.PROJECT_ROOT)}`")
    st.stop()

module = st.sidebar.radio(
    "Module",
    options=list(available.keys()),
    key="design_eval_module",
)

path = available[module]
text = path.read_text(encoding="utf-8")

st.caption(f"Source: `{path.relative_to(config.PROJECT_ROOT)}`")

c1, c2 = st.columns([1, 1])
c1.metric("Lines", text.count("\n") + 1)
c2.download_button(
    "Download markdown",
    data=text,
    file_name=path.name,
    mime="text/markdown",
)

st.divider()
st.markdown(text)

if missing:
    with st.expander("Modules without a design evaluation yet"):
        for label, p in missing.items():
            st.markdown(f"- **{label}** — `{p.relative_to(config.PROJECT_ROOT)}` (not found)")
