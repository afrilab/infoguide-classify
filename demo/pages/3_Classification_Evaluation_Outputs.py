"""Classification Evaluation Outputs — browse aggregate evaluation artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from lib import artifacts  # noqa: E402


def _rel(p: Path) -> str:
    return str(p.relative_to(config.PROJECT_ROOT))


st.set_page_config(page_title="Classification Evaluation Outputs · InfoGuide", layout="wide")
st.title("Classification Evaluation Outputs")
st.caption(
    f"Sources: `{_rel(config.CLASSIFICATION_RESULTS_DIR)}` (consolidated) · "
    f"`{_rel(config.CLASSIFICATION_OUTPUTS_DIR)}` (per-method)."
)

results_present = config.CLASSIFICATION_RESULTS_DIR.exists()
outputs_present = config.CLASSIFICATION_OUTPUTS_DIR.exists()

if not results_present and not outputs_present:
    st.warning("No classification evaluation artifacts found.")
    st.stop()

tab_overview, tab_compare, tab_confusion, tab_threshold, tab_cv, tab_reports, tab_synthetic = st.tabs(
    [
        "Overview",
        "Model comparison",
        "Confusion matrices",
        "Threshold calibration",
        "Cross-validation",
        "Per-method reports",
        "Anonymization impact",
    ]
)


with tab_overview:
    st.subheader("Ground-truth class distribution")
    dist_md = artifacts.classification_result_markdown("class_distribution.md")
    if dist_md is not None:
        st.markdown(dist_md)
    else:
        st.info("`class_distribution.md` not found.")

    st.subheader("Methods overview")
    methods_img = artifacts.classification_result_image("methods_overview.png")
    methods_img_2col = artifacts.classification_result_image("methods_overview_2col.png")
    if methods_img is not None:
        st.image(str(methods_img), use_container_width=True)
    if methods_img_2col is not None:
        with st.expander("Two-column variant", expanded=False):
            st.image(str(methods_img_2col), use_container_width=True)
    if methods_img is None and methods_img_2col is None:
        st.info("No `methods_overview*.png` found.")


with tab_compare:
    st.subheader("Model comparison — 5-fold stratified CV")

    md = artifacts.classification_result_markdown("model_comparison.md")
    csv = artifacts.classification_result_csv("model_comparison.csv")
    bar_img = artifacts.classification_result_image("model_comparison_bar.png")

    # Fall back to legacy classification_outputs comparison if the new one is missing.
    if md is None and csv is None:
        md = artifacts.classification_markdown("classification_model_comparison.md")
        csv = artifacts.classification_csv("classification_model_comparison.csv")

    if md is None and csv is None and bar_img is None:
        st.info("No model comparison artifacts found.")
    else:
        if md is not None:
            st.markdown(md)
        if bar_img is not None:
            st.image(str(bar_img), use_container_width=True)
        if csv is not None:
            with st.expander("Comparison table (CSV)", expanded=False):
                st.dataframe(csv, use_container_width=True, hide_index=True)

    st.subheader("Per-class precision / recall / F1")
    pc_md = artifacts.classification_result_markdown("per_class_metrics.md")
    pc_csv = artifacts.classification_result_csv("per_class_metrics.csv")
    pc_bar = artifacts.classification_result_image("per_class_f1_bar.png")

    if pc_md is None and pc_csv is None and pc_bar is None:
        st.info("No per-class metrics artifacts found.")
    else:
        if pc_bar is not None:
            st.image(str(pc_bar), use_container_width=True)
        if pc_md is not None:
            with st.expander("Per-class metrics (markdown)", expanded=False):
                st.markdown(pc_md)
        if pc_csv is not None:
            with st.expander("Per-class metrics (CSV)", expanded=False):
                st.dataframe(pc_csv, use_container_width=True, hide_index=True)


with tab_confusion:
    st.subheader("Confusion matrices")
    cm_names = artifacts.confusion_matrix_names()
    if not cm_names:
        st.info("No `confusion_matrix__*.png` files found.")
    else:
        view = st.radio(
            "Layout",
            options=("Grid", "Single"),
            horizontal=True,
            label_visibility="collapsed",
        )

        def _label(name: str) -> str:
            return name.replace("_", " ").title()

        if view == "Single":
            choice = st.selectbox("Model", options=cm_names, format_func=_label)
            img = artifacts.classification_result_image(f"confusion_matrix__{choice}.png")
            if img is not None:
                st.image(str(img), use_container_width=True)
        else:
            cols = st.columns(2)
            for i, name in enumerate(cm_names):
                img = artifacts.classification_result_image(f"confusion_matrix__{name}.png")
                if img is None:
                    continue
                with cols[i % 2]:
                    st.markdown(f"**{_label(name)}**")
                    st.image(str(img), use_container_width=True)


with tab_threshold:
    st.subheader("Threshold calibration — TF-IDF cosine")
    pareto_img = artifacts.classification_result_image("threshold_pareto.png")
    if pareto_img is not None:
        st.image(str(pareto_img), use_container_width=True)

    md = artifacts.classification_markdown("threshold_calibration.md")
    if md is not None:
        st.markdown(md)
    elif pareto_img is None:
        st.info("No threshold-calibration artifacts found.")


with tab_cv:
    st.subheader("Supervised baselines — stratified 5-fold CV")
    sup_md = artifacts.classification_markdown("supervised_baselines_cv.md")
    if sup_md is not None:
        st.markdown(sup_md)
    else:
        st.info("`supervised_baselines_cv.md` not found.")

    st.subheader("Fine-tuned transformer — DistilBERT 5-fold CV")
    ft_md = artifacts.classification_markdown("finetune_distilbert_cv.md")
    if ft_md is not None:
        st.markdown(ft_md)
    else:
        st.info("`finetune_distilbert_cv.md` not found.")


with tab_reports:
    st.subheader("Per-method reports")

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
