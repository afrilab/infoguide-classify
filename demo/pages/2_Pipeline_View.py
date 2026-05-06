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


tab_raw, tab_pre, tab_anon, tab_cls, tab_tax = st.tabs(
    ["Raw", "Preprocessed", "Anonymized", "Classification", "Taxonomy"]
)


# --- Raw ---------------------------------------------------------------------
with tab_raw:
    st.subheader("Raw document")
    viewers.render_raw_document(doc.path, height=720)


# --- Preprocessed ------------------------------------------------------------
with tab_pre:
    st.subheader("Preprocessed text")
    processed = artifacts.get_processed(doc.doc_id)
    if not processed:
        st.info(
            "No preprocessed output found for this document. "
            "Preprocessing is currently produced for the real corpus only "
            f"(`{config.PROCESSED_FILE.relative_to(config.PROJECT_ROOT)}`)."
        )
    else:
        stats = processed.get("processed_stats") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Words", stats.get("word_count", "—"))
        c2.metric("Lines", stats.get("line_count", "—"))
        c3.metric("Sentences", processed.get("sentence_count", "—"))
        c4.metric("Quality", (processed.get("processed_quality") or {}).get("status", "—"))

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
        # Helpful diagnostic: anonymization in this repo currently covers synthetic only.
        if doc.corpus != "synthetic":
            st.info(
                "No anonymization output is available for this document. "
                "Anonymization variants are currently produced for the **synthetic** "
                "corpus only."
            )
        else:
            st.warning(
                f"Variant '{variant}' has no entry for `{doc.doc_id}`."
            )
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


# --- Classification ----------------------------------------------------------
with tab_cls:
    st.subheader("Classification")

    variants = artifacts.classification_variants_for(doc.corpus)
    if not variants:
        st.info("No classification variants registered for this corpus.")
    else:
        labels = list(variants.keys())
        default = artifacts.classification_default_for(doc.corpus)
        default_idx = labels.index(default) if default in labels else 0
        variant = st.selectbox(
            "Classification model / method",
            options=labels,
            index=default_idx,
            key="cls_variant",
        )

        rec = artifacts.get_classification(doc.doc_id, variant, doc.corpus)
        if not rec:
            st.warning(f"Variant '{variant}' has no entry for `{doc.doc_id}`.")
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
                        "description": (t.get("raw_label") or "")[: 200] + (
                            "…" if t.get("raw_label") and len(t["raw_label"]) > 200 else ""
                        ),
                    }
                    for t in top
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
                # Lightweight bar chart of scores (Streamlit native).
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
