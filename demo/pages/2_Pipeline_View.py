"""Pipeline View — inspect every stage's output for the selected document."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from lib import artifacts, corpus, viewers  # noqa: E402

st.set_page_config(page_title="Pipeline View · InfoGuide", layout="wide")
st.title("Pipeline View")

# --- Resolve selection from session state -----------------------------------
doc_id = st.session_state.get("selected_doc_id")
if not doc_id:
    st.info("No document selected yet. Open the **Browse** page and pick one.")
    st.stop()

doc = corpus.find_by_id(doc_id)
if doc is None:
    st.error(f"Selected document '{doc_id}' not found on disk.")
    st.stop()

st.caption(
    f"**doc_id**: `{doc.doc_id}` · **corpus**: {doc.corpus}"
    + (f" · **source**: {doc.source}" if doc.source else "")
    + f" · **filename**: {doc.filename}"
)


def _fmt_score(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


tab_raw, tab_pre, tab_anon, tab_anon_eval, tab_cls, tab_tax = st.tabs(
    [
        "Raw",
        "Preprocessed",
        "Anonymized",
        "Anonymization Evaluation",
        "Classification",
        "Taxonomy",
    ]
)


# --- Raw ---------------------------------------------------------------------
with tab_raw:
    st.subheader("Raw document")
    viewers.render_raw_document(doc.path, height=720)


# --- Preprocessed ------------------------------------------------------------
with tab_pre:
    st.subheader("Preprocessed text")

    if doc.corpus == "synthetic":
        # Synthetic docs bypass preprocessing — show the extraction-equivalent
        # record (raw text from synthetic_raw.jsonl, falling back to the .txt file).
        st.info(
            "Preprocessing is **bypassed** for the synthetic corpus. "
            "Showing the extraction output (raw text record) instead."
        )
        rec = artifacts.get_synthetic_raw(doc.doc_id)
        text = (rec or {}).get("text")
        if text is None:
            try:
                text = doc.path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = doc.path.read_text(encoding="latin-1")

        c1, c2 = st.columns(2)
        c1.metric("Characters", len(text))
        c2.metric("Words", len(text.split()))

        st.text_area(
            "extraction_text",
            value=text,
            height=620,
            label_visibility="collapsed",
        )
        if rec:
            with st.expander("Raw record (synthetic_raw.jsonl)"):
                st.json({k: v for k, v in rec.items() if k != "text"})
    else:
        processed = artifacts.get_processed(doc.doc_id)
        if not processed:
            st.warning(
                "No preprocessed output found for this document in "
                f"`{config.PROCESSED_FILE.relative_to(config.PROJECT_ROOT)}`."
            )
        else:
            stats = processed.get("processed_stats") or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Words", stats.get("word_count", "—"))
            c2.metric("Lines", stats.get("line_count", "—"))
            c3.metric("Sentences", processed.get("sentence_count", "—"))
            c4.metric(
                "Quality",
                (processed.get("processed_quality") or {}).get("status", "—"),
            )

            view_mode = st.radio(
                "View",
                options=["Full text", "Sentences"],
                horizontal=True,
                key="pre_view_mode",
            )
            if view_mode == "Full text":
                st.text_area(
                    "processed_text",
                    value=processed.get("processed_text", ""),
                    height=600,
                    label_visibility="collapsed",
                )
            else:
                sents = processed.get("sentences") or []
                if not sents:
                    st.info("No sentence-level output available.")
                else:
                    st.write(f"Showing {len(sents)} sentences.")
                    for i, s in enumerate(sents):
                        st.markdown(f"**{i+1}.** {s}")


# --- Anonymized --------------------------------------------------------------
with tab_anon:
    st.subheader("Anonymized text")

    variants = list(config.ANONYMIZATION_VARIANTS.keys())
    default_idx = variants.index(config.ANONYMIZATION_DEFAULT)
    variant = st.selectbox(
        "Anonymization variant",
        options=variants,
        index=default_idx,
        help="Each variant corresponds to a different Presidio configuration "
             "(NER size sm/lg/trf, regex-only, NER-only, generic placeholders, etc.).",
        key="anon_variant",
    )

    rec = artifacts.get_anonymized(doc.doc_id, variant)
    if not rec:
        if doc.corpus != "synthetic":
            st.info(
                "No anonymization output is available for this document. "
                "Anonymization variants are currently produced for the **synthetic** "
                "corpus only."
            )
        else:
            st.warning(f"Variant '{variant}' has no entry for `{doc.doc_id}`.")
    else:
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("**Original (raw)**")
            viewers.render_raw_document(doc.path, height=600)
        with col_b:
            st.markdown(f"**Anonymized — _{variant}_**")
            st.text_area(
                "anonymized_text",
                value=rec.get("anonymized_text", ""),
                height=600,
                label_visibility="collapsed",
            )


# --- Anonymization Evaluation -----------------------------------------------
with tab_anon_eval:
    st.subheader("Anonymization Evaluation")
    st.caption(
        "Aggregate evaluation outputs from "
        f"`{config.ANON_EVAL_DIR.relative_to(config.PROJECT_ROOT)}/<approach>/`. "
        "These metrics are dataset-level (not per-document)."
    )

    approaches = artifacts.list_anon_eval_approaches()
    if not approaches:
        st.warning(
            "No evaluation outputs found at "
            f"`{config.ANON_EVAL_DIR.relative_to(config.PROJECT_ROOT)}`."
        )
    else:
        approach = st.selectbox(
            "Approach",
            options=approaches,
            format_func=lambda k: f"{k}  —  {config.ANON_EVAL_APPROACHES.get(k, k)}",
            key="anon_eval_approach",
        )

        # Summary metrics
        summary = artifacts.anon_eval_summary(approach)
        if summary:
            st.markdown("### Summary")
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

        # Per-slice CSVs
        st.markdown("### Per-slice breakdowns")
        slice_tabs = st.tabs(
            ["Per type", "Per difficulty", "Per doc type", "Per type variation",
             "Negative cases", "Comparison table"]
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

        # Errors (errors.jsonl) — with optional doc-id filter
        st.markdown("### Errors (errors.jsonl)")
        errors = artifacts.anon_eval_errors(approach)
        if not errors:
            st.info("No errors.jsonl found for this approach.")
        else:
            only_this_doc = st.checkbox(
                f"Filter to current document (`{doc.doc_id}`)",
                value=(doc.corpus == "synthetic"),
                key="anon_eval_errors_filter",
            )
            shown = (
                [e for e in errors if e.get("doc_id") == doc.doc_id]
                if only_this_doc
                else errors
            )
            st.caption(f"{len(shown)} of {len(errors)} error rows shown")
            if shown:
                st.dataframe(shown, use_container_width=True, hide_index=True)


# --- Classification ----------------------------------------------------------
with tab_cls:
    st.subheader("Classification")

    variants = list(config.CLASSIFICATION_VARIANTS.keys())
    default = artifacts.classification_default_for(doc.corpus)
    default_idx = variants.index(default) if default in variants else 0
    variant = st.selectbox(
        "Classification model / method",
        options=variants,
        index=default_idx,
        help="All registered classifiers are listed. If a variant has no record "
             "for this document, a note is shown instead.",
        key="cls_variant",
    )

    rec = artifacts.get_classification(doc.doc_id, variant)
    if not rec:
        st.warning(
            f"Variant '{variant}' has no entry for `{doc.doc_id}`. "
            "It may not have been run on this corpus yet."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted", rec.get("predicted_label", "—"))
        c2.metric("Confidence", _fmt_score(rec.get("confidence")))
        c3.metric("Margin", _fmt_score(rec.get("margin")))

        st.caption(
            f"Second best: **{rec.get('second_best', '—')}** "
            f"(score={_fmt_score(rec.get('second_score'))}) · "
            f"used_field=`{rec.get('used_field', '—')}` · "
            f"text_len={rec.get('text_len', '—')} · "
            f"method=`{rec.get('method', '—')}` · "
            f"model=`{rec.get('model', '—')}`"
        )

        top = rec.get("top_labels") or []
        if top:
            st.markdown("**Top labels**")
            rows = [
                {
                    "label": t.get("label"),
                    "score": t.get("score"),
                    "description": (t.get("raw_label") or "")[:200] + (
                        "…" if t.get("raw_label") and len(t["raw_label"]) > 200 else ""
                    ),
                }
                for t in top
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            chart_data = {t["label"]: t["score"] for t in top if "label" in t}
            if chart_data:
                st.bar_chart(chart_data, horizontal=True)

        with st.expander("Raw record"):
            st.json(rec)


# --- Taxonomy ---------------------------------------------------------------
with tab_tax:
    st.subheader("Taxonomy mapping")
    st.info(
        "Taxonomy outputs are not yet generated by the pipeline. "
        "Once the taxonomy module produces per-document level-1/2/3 mappings, "
        "this tab will display the predicted path, scores, and (where available) "
        "ground-truth comparison."
    )
