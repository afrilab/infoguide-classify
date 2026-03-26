# src/infoguide/classification/run.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from file_io import read_jsonl, write_jsonl, pick_text


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
    head = text[:half]
    tail = text[-half:]

    return head + "\n...\n" + tail

def classify_documents(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:

    method = (cfg.get("method") or "zero_shot").strip().lower()
    if method not in {"zero_shot", "zeroshot", "zero-shot"}:
        raise ValueError(f"Unsupported classification method: {method}")

    input_path = cfg.get("input_path")
    output_path = cfg.get("output_path")
    if not input_path or not output_path:
        raise ValueError("Config must include 'input_path' and 'output_path'.")

    base_labels = cfg.get("labels")
    if not isinstance(base_labels, list) or not base_labels:
        raise ValueError("Config must include non-empty list 'labels'.")

    model_name = cfg.get("model_name") or "facebook/bart-large-mnli"
    device = cfg.get("device", -1)

    hypothesis_template = _normalize_template_for_hf(
        cfg.get("hypothesis_template") or "This document is about {}."
    )

    multi_label = bool(cfg.get("multi_label", False))
    top_k = int(cfg.get("top_k", len(base_labels) if len(base_labels) else 1))

    min_score = float(cfg.get("min_score", 0.0))
    min_margin = float(cfg.get("min_margin", 0.0))

    head_only = bool(cfg.get("head_only", False))
    max_chars = int(cfg.get("max_chars", 6000))

    fields_priority = cfg.get("text_fields_priority") or ["processed_text", "clean_text", "text"]

    use_label_descriptions = bool(cfg.get("use_label_descriptions", False))
    label_descriptions = cfg.get("label_descriptions") or {}

    candidate_labels = _build_candidate_labels(
        base_labels=base_labels,
        label_descriptions=label_descriptions,
        use_label_descriptions=use_label_descriptions,
    )

    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "transformers is required for zero-shot classification. "
            "Install it with: pip install transformers"
        ) from e

    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )

    rows_in = read_jsonl(str(input_path))
    rows_out: List[Dict[str, Any]] = []

    for row in rows_in:
        doc_id = row.get("doc_id") or row.get("id") or row.get("filename") or "unknown"

        text, used_field = pick_text(row, fields_priority)
        text = _slice_text(text, head_only=head_only, max_chars=max_chars)

        if not text.strip():
            rows_out.append(
                {
                    "doc_id": doc_id,
                    "predicted_label": "Needs_Review",
                    "confidence": 0.0,
                    "second_best": None,
                    "second_score": 0.0,
                    "margin": 0.0,
                    "min_margin": min_margin,
                    "top_labels": [],
                    "used_field": used_field,
                    "text_len": 0,
                    "method": "zero_shot",
                    "model": model_name,
                    "head_only": head_only,
                    "max_chars": max_chars,
                }
            )
            continue

        result = clf(
            sequences=text,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label,
        )
        raw_labels: List[str] = list(result.get("labels", []))
        raw_scores: List[float] = [float(x) for x in result.get("scores", [])]

        pairs: List[Tuple[str, float]] = list(zip(raw_labels, raw_scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[: max(1, min(top_k, len(pairs)))]

        top_labels: List[Dict[str, Any]] = []
        for raw_lbl, sc in pairs:
            norm = _normalize_label(raw_lbl, base_labels)
            top_labels.append({"label": norm, "score": float(sc), "raw_label": raw_lbl})

        best = top_labels[0] if top_labels else None
        second = top_labels[1] if len(top_labels) > 1 else None

        predicted = best["label"] if best else "Needs_Review"
        confidence = float(best["score"]) if best else 0.0
        second_best = second["label"] if second else None
        second_score = float(second["score"]) if second else 0.0
        margin = confidence - second_score

        if predicted not in base_labels:
            predicted = "Needs_Review"

        if confidence < min_score:
            predicted = "Needs_Review"

        if predicted != "Needs_Review" and min_margin > 0.0:
            if margin < min_margin:
                predicted = "Needs_Review"

        rows_out.append(
            {
                "doc_id": doc_id,
                "predicted_label": predicted,
                "confidence": confidence,
                "second_best": second_best,
                "second_score": second_score,
                "margin": margin,
                "min_margin": min_margin,
                "top_labels": top_labels,
                "used_field": used_field,
                "text_len": len(text),
                "method": "zero_shot",
                "model": model_name,
                "head_only": head_only,
                "max_chars": max_chars,
            }
        )

    write_jsonl(str(output_path), rows_out)
    return rows_out
