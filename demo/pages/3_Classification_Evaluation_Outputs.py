"""Classification Evaluation Outputs — browse aggregate evaluation artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from lib import artifacts  # noqa: E402

st.set_page_config(page_title="Classification Evaluation Outputs · InfoGuide", layout="wide")
st.title("Classification Evaluation Outputs")
st.caption(
    "Evaluation artifacts from "
    f"`{config.CLASSIFICATION_OUTPUTS_DIR.relative_to(config.PROJECT_ROOT)}`."
)

if not config.CLASSIFICATION_OUTPUTS_DIR.exists():
    st.warning(
        "No classification evaluation output directory found at "
        f"`{config.CLASSIFICATION_OUTPUTS_DIR.relative_to(config.PROJECT_ROOT)}`."
    )
    st.stop()

tab_compare, tab_reports, tab_synthetic = st.tabs(
    ["Model comparison", "Per-model reports", "Anonymization impact"]
)

with tab_compare:
    st.subheader("Model comparison")

    md = artifacts.classification_markdown("classification_model_comparison.md")
    csv = artifacts.classification_csv("classification_model_comparison.csv")

    if md is None and csv is None:
        st.info("No classification model comparison outputs found.")
    else:
        if md is not None:
            st.markdown(md)
        if csv is not None:
            st.markdown("### Comparison table")
            st.dataframe(csv, use_container_width=True, hide_index=True)

with tab_reports:
    st.subheader("Per-model reports")

    report_names = artifacts.classification_report_names()
    if not report_names:
        st.info("No `classification_report__<name>.md` files found.")
    else:
        report_name = st.selectbox(
            "Report",
            options=report_names,
            format_func=lambda name: name.replace("_", " ").title(),
        )

        report_md = artifacts.classification_markdown(
            f"classification_report__{report_name}.md"
        )
        doc_level = artifacts.classification_csv(f"doc_level_eval__{report_name}.csv")

        if report_md is not None:
            st.markdown(report_md)
        else:
            st.info(f"`classification_report__{report_name}.md` not found.")

        st.markdown("### Document-level evaluation")
        if doc_level is None:
            st.info(f"`doc_level_eval__{report_name}.csv` not found.")
        else:
            st.dataframe(doc_level, use_container_width=True, hide_index=True)

with tab_synthetic:
    st.subheader("Synthetic TF-IDF anonymization comparison")

    md = artifacts.classification_markdown(
        "synthetic_tfidf_anonymization_comparison.md"
    )
    csv = artifacts.classification_csv(
        "synthetic_tfidf_anonymization_comparison.csv"
    )

    if md is None and csv is None:
        st.info("No synthetic anonymization comparison outputs found.")
    else:
        if md is not None:
            st.markdown(md)
        if csv is not None:
            st.markdown("### Comparison table")
            st.dataframe(csv, use_container_width=True, hide_index=True)
