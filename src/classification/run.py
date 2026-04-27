# src/classification/run.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from file_io import read_jsonl, write_jsonl, pick_text


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_template_for_hf(template: str) -> str:
    t = (template or "").strip()
    if not t:
        return "This document is about {}."
    if "{label}" in t:
        t = t.replace("{label}", "{}")
    return t


def _normalize_label(raw_label: str, base_labels: List[str]) -> str:
    s = (raw_label or "").strip()

    for sep in ["—", " - ", ":", "|"]:
        if sep in s:
            left = s.split(sep, 1)[0].strip()
            if left in base_labels:
                return left

    if s in base_labels:
        return s

    lower_map = {b.lower(): b for b in base_labels}
    if s.lower() in lower_map:
        return lower_map[s.lower()]

    return s


def _build_candidate_labels(
    base_labels: List[str],
    label_descriptions: Dict[str, str] | None,
    use_label_descriptions: bool,
) -> List[str]:
    if not use_label_descriptions or not label_descriptions:
        return list(base_labels)

    candidates: List[str] = []
    for lbl in base_labels:
        desc = (label_descriptions.get(lbl) or "").strip()
        if desc:
            candidates.append(f"{lbl} - {desc}")
        else:
            candidates.append(lbl)
    return candidates


def _slice_text(text: str, max_chars: int, head_only: bool) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    if head_only:
        return text[:max_chars]
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def _build_output_row(
    *,
    doc_id: str,
    used_field: str,
    text_len: int,
    scored_pairs: List[Tuple[str, float]],
    base_labels: List[str],
    top_k: int,
    min_score: float,
    min_margin: float,
    method: str,
    model_name: str,
    head_only: bool,
    max_chars: int,
) -> Dict[str, Any]:
    """Apply thresholds and format the final output row from a list of (raw_label, score)."""

    if not scored_pairs:
        return {
            "doc_id": doc_id,
            "predicted_label": "Needs_Review",
            "confidence": 0.0,
            "second_best": None,
            "second_score": 0.0,
            "margin": 0.0,
            "min_margin": min_margin,
            "top_labels": [],
            "used_field": used_field,
            "text_len": text_len,
            "method": method,
            "model": model_name,
            "head_only": head_only,
            "max_chars": max_chars,
        }

    pairs = sorted(scored_pairs, key=lambda x: x[1], reverse=True)
    pairs = pairs[: max(1, min(top_k, len(pairs)))]

    top_labels: List[Dict[str, Any]] = []
    for raw_lbl, sc in pairs:
        norm = _normalize_label(raw_lbl, base_labels)
        top_labels.append({"label": norm, "score": float(sc), "raw_label": raw_lbl})

    best = top_labels[0]
    second = top_labels[1] if len(top_labels) > 1 else None

    predicted = best["label"]
    confidence = float(best["score"])
    second_best = second["label"] if second else None
    second_score = float(second["score"]) if second else 0.0
    margin = confidence - second_score

    if predicted not in base_labels:
        predicted = "Needs_Review"
    if confidence < min_score:
        predicted = "Needs_Review"
    if predicted != "Needs_Review" and min_margin > 0.0 and margin < min_margin:
        predicted = "Needs_Review"

    return {
        "doc_id": doc_id,
        "predicted_label": predicted,
        "confidence": confidence,
        "second_best": second_best,
        "second_score": second_score,
        "margin": margin,
        "min_margin": min_margin,
        "top_labels": top_labels,
        "used_field": used_field,
        "text_len": text_len,
        "method": method,
        "model": model_name,
        "head_only": head_only,
        "max_chars": max_chars,
    }


# ---------------------------------------------------------------------------
# Method 1: zero-shot NLI (BART-MNLI)
# ---------------------------------------------------------------------------

def _classify_zero_shot(
    *,
    rows_in: List[Dict[str, Any]],
    candidate_labels: List[str],
    base_labels: List[str],
    cfg: Dict[str, Any],
    fields_priority: List[str],
) -> List[Dict[str, Any]]:
    model_name = cfg.get("model_name") or "facebook/bart-large-mnli"
    device = cfg.get("device", -1)
    hypothesis_template = _normalize_template_for_hf(
        cfg.get("hypothesis_template") or "This document is about {}."
    )
    multi_label = bool(cfg.get("multi_label", False))
    top_k = int(cfg.get("top_k", len(base_labels) if base_labels else 1))
    min_score = float(cfg.get("min_score", 0.0))
    min_margin = float(cfg.get("min_margin", 0.0))
    head_only = bool(cfg.get("head_only", False))
    max_chars = int(cfg.get("max_chars", 6000))

    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "transformers is required for zero-shot classification. "
            "Install with: pip install transformers"
        ) from e

    clf = pipeline("zero-shot-classification", model=model_name, device=device)

    rows_out: List[Dict[str, Any]] = []
    for row in rows_in:
        doc_id = row.get("doc_id") or row.get("id") or row.get("filename") or "unknown"
        text, used_field = pick_text(row, fields_priority)
        text = _slice_text(text, head_only=head_only, max_chars=max_chars)

        if not text.strip():
            scored: List[Tuple[str, float]] = []
        else:
            result = clf(
                sequences=text,
                candidate_labels=candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=multi_label,
            )
            scored = list(zip(
                list(result.get("labels", [])),
                [float(x) for x in result.get("scores", [])],
            ))

        rows_out.append(_build_output_row(
            doc_id=doc_id,
            used_field=used_field,
            text_len=len(text),
            scored_pairs=scored,
            base_labels=base_labels,
            top_k=top_k,
            min_score=min_score,
            min_margin=min_margin,
            method="zero_shot",
            model_name=model_name,
            head_only=head_only,
            max_chars=max_chars,
        ))

    return rows_out


# ---------------------------------------------------------------------------
# Method 2: TF-IDF cosine similarity to label descriptions
# ---------------------------------------------------------------------------

def _classify_tfidf(
    *,
    rows_in: List[Dict[str, Any]],
    candidate_labels: List[str],
    base_labels: List[str],
    cfg: Dict[str, Any],
    fields_priority: List[str],
) -> List[Dict[str, Any]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for tfidf classification."
        ) from e

    top_k = int(cfg.get("top_k", len(base_labels) if base_labels else 1))
    min_score = float(cfg.get("min_score", 0.0))
    min_margin = float(cfg.get("min_margin", 0.0))
    head_only = bool(cfg.get("head_only", False))
    max_chars = int(cfg.get("max_chars", 6000))

    ngram_min = int(cfg.get("tfidf_ngram_min", 1))
    ngram_max = int(cfg.get("tfidf_ngram_max", 2))
    stop_words = cfg.get("tfidf_stop_words", "english")
    min_df = int(cfg.get("tfidf_min_df", 1))

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for row in rows_in:
        doc_id = row.get("doc_id") or row.get("id") or row.get("filename") or "unknown"
        text, used_field = pick_text(row, fields_priority)
        text = _slice_text(text, head_only=head_only, max_chars=max_chars)
        texts.append(text)
        metas.append({"doc_id": doc_id, "used_field": used_field, "text_len": len(text)})

    nonempty_indices = [i for i, t in enumerate(texts) if t.strip()]
    nonempty_texts = [texts[i] for i in nonempty_indices]

    sim_by_doc_idx: Dict[int, List[float]] = {}
    if nonempty_texts:
        vec = TfidfVectorizer(
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            stop_words=stop_words,
            lowercase=True,
        )
        doc_mat = vec.fit_transform(nonempty_texts)
        label_mat = vec.transform(candidate_labels)
        sim = cosine_similarity(doc_mat, label_mat)  # (n_nonempty, n_labels)
        for local_i, doc_i in enumerate(nonempty_indices):
            sim_by_doc_idx[doc_i] = [float(s) for s in sim[local_i]]

    rows_out: List[Dict[str, Any]] = []
    for i, meta in enumerate(metas):
        if i in sim_by_doc_idx:
            scored = list(zip(candidate_labels, sim_by_doc_idx[i]))
        else:
            scored = []

        rows_out.append(_build_output_row(
            doc_id=meta["doc_id"],
            used_field=meta["used_field"],
            text_len=meta["text_len"],
            scored_pairs=scored,
            base_labels=base_labels,
            top_k=top_k,
            min_score=min_score,
            min_margin=min_margin,
            method="tfidf",
            model_name=f"tfidf(ngram=[{ngram_min},{ngram_max}],min_df={min_df})",
            head_only=head_only,
            max_chars=max_chars,
        ))

    return rows_out


# ---------------------------------------------------------------------------
# Method 3: sentence-embedding cosine similarity to label descriptions
# ---------------------------------------------------------------------------

def _classify_embedding(
    *,
    rows_in: List[Dict[str, Any]],
    candidate_labels: List[str],
    base_labels: List[str],
    cfg: Dict[str, Any],
    fields_priority: List[str],
) -> List[Dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required for embedding classification. "
            "Install with: pip install sentence-transformers"
        ) from e

    model_name = cfg.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"
    top_k = int(cfg.get("top_k", len(base_labels) if base_labels else 1))
    min_score = float(cfg.get("min_score", 0.0))
    min_margin = float(cfg.get("min_margin", 0.0))
    head_only = bool(cfg.get("head_only", False))
    max_chars = int(cfg.get("max_chars", 6000))

    model = SentenceTransformer(model_name)

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for row in rows_in:
        doc_id = row.get("doc_id") or row.get("id") or row.get("filename") or "unknown"
        text, used_field = pick_text(row, fields_priority)
        text = _slice_text(text, head_only=head_only, max_chars=max_chars)
        texts.append(text)
        metas.append({"doc_id": doc_id, "used_field": used_field, "text_len": len(text)})

    nonempty_indices = [i for i, t in enumerate(texts) if t.strip()]
    nonempty_texts = [texts[i] for i in nonempty_indices]

    sim_by_doc_idx: Dict[int, List[float]] = {}
    if nonempty_texts:
        doc_emb = model.encode(nonempty_texts, convert_to_numpy=True, show_progress_bar=False)
        label_emb = model.encode(candidate_labels, convert_to_numpy=True, show_progress_bar=False)
        sim = cosine_similarity(doc_emb, label_emb)
        for local_i, doc_i in enumerate(nonempty_indices):
            sim_by_doc_idx[doc_i] = [float(s) for s in sim[local_i]]

    rows_out: List[Dict[str, Any]] = []
    for i, meta in enumerate(metas):
        scored = list(zip(candidate_labels, sim_by_doc_idx[i])) if i in sim_by_doc_idx else []
        rows_out.append(_build_output_row(
            doc_id=meta["doc_id"],
            used_field=meta["used_field"],
            text_len=meta["text_len"],
            scored_pairs=scored,
            base_labels=base_labels,
            top_k=top_k,
            min_score=min_score,
            min_margin=min_margin,
            method="embedding",
            model_name=model_name,
            head_only=head_only,
            max_chars=max_chars,
        ))

    return rows_out


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def classify_documents(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    method = (cfg.get("method") or "zero_shot").strip().lower()

    input_path = cfg.get("input_path")
    output_path = cfg.get("output_path")
    if not input_path or not output_path:
        raise ValueError("Config must include 'input_path' and 'output_path'.")

    base_labels = cfg.get("labels")
    if not isinstance(base_labels, list) or not base_labels:
        raise ValueError("Config must include non-empty list 'labels'.")

    fields_priority = cfg.get("text_fields_priority") or [
        "processed_text", "clean_text", "text"
    ]
    use_label_descriptions = bool(cfg.get("use_label_descriptions", False))
    label_descriptions = cfg.get("label_descriptions") or {}
    candidate_labels = _build_candidate_labels(
        base_labels=base_labels,
        label_descriptions=label_descriptions,
        use_label_descriptions=use_label_descriptions,
    )

    rows_in = read_jsonl(str(input_path))

    if method in {"zero_shot", "zeroshot", "zero-shot"}:
        rows_out = _classify_zero_shot(
            rows_in=rows_in, candidate_labels=candidate_labels,
            base_labels=base_labels, cfg=cfg, fields_priority=fields_priority,
        )
    elif method == "tfidf":
        rows_out = _classify_tfidf(
            rows_in=rows_in, candidate_labels=candidate_labels,
            base_labels=base_labels, cfg=cfg, fields_priority=fields_priority,
        )
    else:
        raise ValueError(f"Unsupported classification method: {method}")

    write_jsonl(str(output_path), rows_out)
    return rows_out
