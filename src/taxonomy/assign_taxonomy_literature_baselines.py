"""Literature-style taxonomy baselines for hierarchical text classification.

The baselines intentionally avoid handwritten banking rules:

- flat_tfidf_logreg: one flat classifier over Level 3 labels.
- topdown_tfidf_logreg: local classifiers from Level 1 to Level 2 to Level 3.
- flat_tfidf_svm: one flat Linear SVM classifier over Level 3 labels.
- topdown_tfidf_svm: local Linear SVM classifiers from Level 1 to Level 3.

By default, predictions are generated with leave-one-out training for documents
that appear in the label file. This keeps the comparison usable on a small
corpus without evaluating a document on a classifier trained with its own label.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


LEVEL_KEYS = ("level_1", "level_2", "level_3")
TaxonomyPath = Tuple[str, str, str]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_text(doc: Dict[str, Any], max_chars: int) -> str:
    for key in ("processed_text", "text", "clean_text", "content", "document_text", "body"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value[:max_chars]
    return ""


def get_taxonomy(row: Dict[str, Any]) -> Optional[TaxonomyPath]:
    taxonomy = row.get("taxonomy")
    if not isinstance(taxonomy, dict):
        return None
    values: List[str] = []
    for key in LEVEL_KEYS:
        value = taxonomy.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        values.append(value.strip())
    return values[0], values[1], values[2]


def make_pipeline(max_features: int, classifier: str) -> Pipeline:
    if classifier == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
    elif classifier == "svm":
        clf = LinearSVC(class_weight="balanced", dual="auto")
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            ("clf", clf),
        ]
    )


def majority_label(labels: Sequence[str]) -> Optional[str]:
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def fit_predict_label(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    text: str,
    max_features: int,
    classifier: str,
) -> tuple[Optional[str], float, str]:
    labels = [label for label in train_labels if label]
    if not labels:
        return None, 0.0, "no_training_labels"

    unique_labels = sorted(set(labels))
    if len(unique_labels) == 1:
        return unique_labels[0], 1.0, "single_class_fallback"

    model = make_pipeline(max_features, classifier)
    model.fit(list(train_texts), list(train_labels))
    predicted = str(model.predict([text])[0])

    confidence = 0.0
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        confidence = float(max(model.predict_proba([text])[0]))
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(model.named_steps["tfidf"].transform([text]))
        if hasattr(scores, "ravel"):
            raw_scores = [float(value) for value in scores.ravel()]
        elif isinstance(scores, list):
            raw_scores = [float(value) for value in scores]
        else:
            raw_scores = [float(scores)]
        if raw_scores:
            confidence = max(raw_scores)

    return predicted, confidence, f"tfidf_{classifier}"


def fallback_path(paths: Sequence[TaxonomyPath]) -> TaxonomyPath:
    if not paths:
        return "Unassigned", "Unassigned", "Unassigned"
    return Counter(paths).most_common(1)[0][0]


def flat_predict(
    train_texts: Sequence[str],
    train_paths: Sequence[TaxonomyPath],
    text: str,
    max_features: int,
    classifier: str,
) -> tuple[TaxonomyPath, float, Dict[str, Any]]:
    level3_to_path: Dict[str, TaxonomyPath] = {}
    for path in train_paths:
        level3_to_path.setdefault(path[2], path)

    pred_l3, confidence, backend = fit_predict_label(
        train_texts,
        [path[2] for path in train_paths],
        text,
        max_features,
        classifier,
    )
    if pred_l3 and pred_l3 in level3_to_path:
        return level3_to_path[pred_l3], confidence, {"backend": backend}

    return fallback_path(train_paths), confidence, {"backend": backend, "fallback": "majority_path"}


def topdown_predict(
    train_texts: Sequence[str],
    train_paths: Sequence[TaxonomyPath],
    text: str,
    max_features: int,
    classifier: str,
) -> tuple[TaxonomyPath, float, Dict[str, Any]]:
    pred_l1, conf_l1, backend_l1 = fit_predict_label(
        train_texts,
        [path[0] for path in train_paths],
        text,
        max_features,
        classifier,
    )
    if pred_l1 is None:
        return fallback_path(train_paths), 0.0, {"fallback": "missing_level_1"}

    l1_indices = [idx for idx, path in enumerate(train_paths) if path[0] == pred_l1]
    l1_paths = [train_paths[idx] for idx in l1_indices]
    l1_texts = [train_texts[idx] for idx in l1_indices]

    pred_l2, conf_l2, backend_l2 = fit_predict_label(
        l1_texts,
        [path[1] for path in l1_paths],
        text,
        max_features,
        classifier,
    )
    if pred_l2 is None:
        path = fallback_path(l1_paths or train_paths)
        return path, conf_l1, {"level_1_backend": backend_l1, "fallback": "missing_level_2"}

    l2_indices = [idx for idx, path in enumerate(l1_paths) if path[1] == pred_l2]
    l2_paths = [l1_paths[idx] for idx in l2_indices]
    l2_texts = [l1_texts[idx] for idx in l2_indices]

    pred_l3, conf_l3, backend_l3 = fit_predict_label(
        l2_texts,
        [path[2] for path in l2_paths],
        text,
        max_features,
        classifier,
    )
    if pred_l3 is None:
        path = fallback_path(l2_paths or l1_paths or train_paths)
        return path, min(conf_l1, conf_l2), {
            "level_1_backend": backend_l1,
            "level_2_backend": backend_l2,
            "fallback": "missing_level_3",
        }

    for path in l2_paths:
        if path[2] == pred_l3:
            return path, min(conf_l1, conf_l2, conf_l3), {
                "level_1_backend": backend_l1,
                "level_2_backend": backend_l2,
                "level_3_backend": backend_l3,
            }

    path = fallback_path(l2_paths or l1_paths or train_paths)
    return path, min(conf_l1, conf_l2, conf_l3), {"fallback": "invalid_level_3_path"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/clean_documents.jsonl")
    ap.add_argument("--labels", default="data/labels/taxonomy_gold_labels.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--method",
        choices=["flat_tfidf_logreg", "topdown_tfidf_logreg", "flat_tfidf_svm", "topdown_tfidf_svm"],
        required=True,
    )
    ap.add_argument("--max_chars", type=int, default=12000)
    ap.add_argument("--max_features", type=int, default=30000)
    ap.add_argument(
        "--train_mode",
        choices=["leave_one_out", "all_labels"],
        default="leave_one_out",
        help="leave_one_out excludes the current document's label when available.",
    )
    args = ap.parse_args()

    docs = {doc["doc_id"]: doc for doc in read_jsonl(args.input)}
    label_rows = {row["doc_id"]: row for row in read_jsonl(args.labels)}

    labeled_items: List[tuple[str, str, TaxonomyPath]] = []
    for doc_id, row in label_rows.items():
        doc = docs.get(doc_id)
        taxonomy = get_taxonomy(row)
        if doc is None or taxonomy is None:
            continue
        labeled_items.append((doc_id, extract_text(doc, args.max_chars), taxonomy))

    records: List[Dict[str, Any]] = []
    for doc_id in sorted(docs):
        doc = docs[doc_id]
        text = extract_text(doc, args.max_chars)
        train_items = labeled_items
        if args.train_mode == "leave_one_out":
            train_items = [item for item in labeled_items if item[0] != doc_id]

        train_texts = [item[1] for item in train_items]
        train_paths = [item[2] for item in train_items]

        classifier = "svm" if args.method.endswith("_svm") else "logreg"
        if args.method.startswith("flat_"):
            path, confidence, debug = flat_predict(train_texts, train_paths, text, args.max_features, classifier)
        else:
            path, confidence, debug = topdown_predict(train_texts, train_paths, text, args.max_features, classifier)

        records.append(
            {
                "doc_id": doc_id,
                "taxonomy": {
                    "level_1": path[0],
                    "level_2": path[1],
                    "level_3": path[2],
                },
                "status": "assigned" if path[0] != "Unassigned" else "unassigned",
                "confidence": round(confidence, 6),
                "method": args.method,
                "debug": {
                    "train_mode": args.train_mode,
                    "baseline_family": "hierarchical_text_classification_literature",
                    **debug,
                },
            }
        )

    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} records to {args.output}")
    print(f"Method: {args.method}; train_mode={args.train_mode}; labels={len(labeled_items)}")


if __name__ == "__main__":
    main()
