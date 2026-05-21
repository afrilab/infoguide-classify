# src/classification/eval_harness.py
"""Shared dataset loading and CV harness for classification experiments.

Used by:
- supervised_baselines.py: TF-IDF + (LogReg, LinearSVM, MultinomialNB, RandomForest)
- threshold_calibration.py: grid sweep over min_score/min_margin
- finetune_transformer.py: DistilBERT fine-tuning
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

GT_PATH = REPO_ROOT / "data" / "eval" / "classification_gt.jsonl"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "clean_documents.jsonl"

TEXT_FIELD_PRIORITY = ("anonymized_text", "processed_text", "clean_text", "text")

DEFAULT_MAX_CHARS = 6000
DEFAULT_HEAD_ONLY = True

# Classes with fewer than this many examples are excluded from CV.
# Stratified k-fold with k=5 needs at least 5 examples per class; we use a
# slightly higher floor so each fold has at least one positive example for
# every class kept in the experiment.
MIN_EXAMPLES_PER_CLASS = 5


@dataclass(frozen=True)
class Example:
    doc_id: str
    text: str
    label: str


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _pick_text(row: dict) -> str:
    for key in TEXT_FIELD_PRIORITY:
        v = row.get(key)
        if v:
            return v
    return ""


def _slice_text(text: str, max_chars: int, head_only: bool) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    if head_only:
        return text[:max_chars]
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def load_dataset(
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    head_only: bool = DEFAULT_HEAD_ONLY,
    drop_singleton_classes: bool = True,
) -> list[Example]:
    """Join classification_gt.jsonl with clean_documents.jsonl.

    Drops GT entries whose doc_id is missing from the processed corpus and,
    by default, drops classes that fall below MIN_EXAMPLES_PER_CLASS so CV
    splits are well-defined.
    """
    gt = {r["doc_id"]: r["true_label"] for r in _read_jsonl(GT_PATH)}
    corpus = {r["doc_id"]: r for r in _read_jsonl(CORPUS_PATH)}

    joined_ids = sorted(set(gt) & set(corpus))

    examples: list[Example] = []
    for doc_id in joined_ids:
        text = _slice_text(_pick_text(corpus[doc_id]), max_chars, head_only)
        if not text:
            continue
        examples.append(Example(doc_id=doc_id, text=text, label=gt[doc_id]))

    if drop_singleton_classes:
        counts = Counter(ex.label for ex in examples)
        keep = {lbl for lbl, n in counts.items() if n >= MIN_EXAMPLES_PER_CLASS}
        examples = [ex for ex in examples if ex.label in keep]

    return examples


def class_distribution(examples: list[Example]) -> dict[str, int]:
    return dict(Counter(ex.label for ex in examples))


def stratified_kfold_indices(
    labels: list[str], k: int = 5, seed: int = 42
) -> list[tuple[list[int], list[int]]]:
    """Return [(train_idx, test_idx), ...] for k-fold stratified CV."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    dummy_x = [[0]] * len(labels)
    return [(list(tr), list(te)) for tr, te in skf.split(dummy_x, labels)]


def aggregate_cv_metrics(fold_metrics: list[dict]) -> dict:
    """Average per-fold metric dicts. Computes mean and std for each numeric key."""
    import statistics

    keys = set()
    for m in fold_metrics:
        keys.update(m.keys())

    agg: dict = {}
    for k in sorted(keys):
        vals = [m[k] for m in fold_metrics if isinstance(m.get(k), (int, float))]
        if not vals:
            continue
        agg[f"{k}_mean"] = statistics.mean(vals)
        agg[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return agg


def score_predictions(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Compute accuracy, macro/weighted F1, and per-class F1."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    for i, lbl in enumerate(labels):
        out[f"f1_{lbl}"] = float(f[i])
        out[f"precision_{lbl}"] = float(p[i])
        out[f"recall_{lbl}"] = float(r[i])
    return out
