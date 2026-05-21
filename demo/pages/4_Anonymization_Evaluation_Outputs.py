"""Anonymization Evaluation Outputs — browse aggregate evaluation artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from lib import artifacts  # noqa: E402

st.set_page_config(page_title="Anonymization Evaluation Outputs · InfoGuide", layout="wide")
st.title("Anonymization Evaluation Outputs")
st.caption(
    "Evaluation artifacts from "
    f"`{config.ANON_EVAL_DIR.relative_to(config.PROJECT_ROOT)}/<approach>/`."
)


def _fmt_score(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


approaches = artifacts.list_anon_eval_approaches()
if not approaches:
    st.warning(
        "No anonymization evaluation outputs found at "
        f"`{config.ANON_EVAL_DIR.relative_to(config.PROJECT_ROOT)}`."
    )
    st.stop()

approach = st.selectbox(
    "Approach",
    options=approaches,
    format_func=lambda k: f"{k}  —  {config.ANON_EVAL_APPROACHES.get(k, k)}",
)

summary = artifacts.anon_eval_summary(approach)
if summary:
    st.subheader("Summary")
    strict = summary.get("strict") or {}
    relaxed = summary.get("relaxed") or {}
    neg = summary.get("negative_cases") or {}

    sc = st.columns(4)
    sc[0].metric("Strict P", _fmt_score(strict.get("precision")))
    sc[1].metric("Strict R", _fmt_score(strict.get("recall")))
    sc[2].metric("Strict F1", _fmt_score(strict.get("f1")))
    sc[3].metric(
        "TP / FP / FN (strict)",
        f"{strict.get('tp', '—')} / {strict.get('fp', '—')} / {strict.get('fn', '—')}",
    )

    rc = st.columns(4)
    rc[0].metric("Relaxed P", _fmt_score(relaxed.get("precision")))
    rc[1].metric("Relaxed R", _fmt_score(relaxed.get("recall")))
    rc[2].metric("Relaxed F1", _fmt_score(relaxed.get("f1")))
    rc[3].metric(
        "TP / FP / FN (relaxed)",
        f"{relaxed.get('tp', '—')} / {relaxed.get('fp', '—')} / {relaxed.get('fn', '—')}",
    )

    if neg:
        nc = st.columns(3)
        nc[0].metric("Negative cases — total", neg.get("total", "—"))
        nc[1].metric("Incorrectly detected", neg.get("incorrectly_detected", "—"))
        nc[2].metric("False-positive rate", _fmt_score(neg.get("false_positive_rate")))

    with st.expander("Raw summary.json"):
        st.json(summary)
else:
    st.info("No summary.json found for this approach.")

st.subheader("Per-slice breakdowns")
slice_tabs = st.tabs(
    [
        "Per type",
        "Per difficulty",
        "Per doc type",
        "Per type variation",
        "Negative cases",
        "Comparison table",
    ]
)
slice_keys = [
    "per_type",
    "per_difficulty",
    "per_doc_type",
    "per_type_variation",
    "negative_cases",
    "comparison_table",
]
for sub_tab, key in zip(slice_tabs, slice_keys):
    with sub_tab:
        df = artifacts.anon_eval_csv(approach, key)
        if df is None:
            st.info(f"`{key}.csv` not present for this approach.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Errors")
errors = artifacts.anon_eval_errors(approach)
if not errors:
    st.info("No errors.jsonl found for this approach.")
else:
    st.caption(f"{len(errors)} error rows")
    st.dataframe(errors, use_container_width=True, hide_index=True)
