"""
Evaluate taxonomy assignment accuracy against manual gold labels.

Usage:
  python src/taxonomy/evaluate_taxonomy_accuracy.py \
    --predictions outputs/taxonomy/taxonomy_assignments_evidence.jsonl \
    --gold outputs/taxonomy/labels/taxonomy_gold_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def get_label(row: Dict[str, Any]) -> str | None:
    label = row.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()

    taxonomy = row.get("taxonomy")
    if isinstance(taxonomy, dict):
        for key in ("level_3", "level_2", "level_1"):
            value = taxonomy.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Predicted taxonomy JSONL")
    ap.add_argument("--gold", default="outputs/taxonomy/labels/taxonomy_gold_labels.jsonl", help="Gold label JSONL")
    ap.add_argument("--show_errors", action="store_true", help="Print each mismatched document")
    args = ap.parse_args()

    gold_rows = {row["doc_id"]: row for row in read_jsonl(Path(args.gold))}
    pred_rows = {row["doc_id"]: row for row in read_jsonl(Path(args.predictions))}

    missing_predictions = sorted(set(gold_rows) - set(pred_rows))
    extra_predictions = sorted(set(pred_rows) - set(gold_rows))
    common_ids = sorted(set(gold_rows) & set(pred_rows))

    total = len(common_ids)
    correct = 0
    gold_counts: Counter[str] = Counter()
    pred_counts: Counter[str | None] = Counter()
    label_stats: Dict[str, Counter[str]] = defaultdict(Counter)
    confusion: Counter[tuple[str, str | None]] = Counter()
    errors: List[Dict[str, Any]] = []

    for doc_id in common_ids:
        gold_label = get_label(gold_rows[doc_id])
        pred_label = get_label(pred_rows[doc_id])

        if gold_label is None:
            continue

        gold_counts[gold_label] += 1
        pred_counts[pred_label] += 1
        confusion[(gold_label, pred_label)] += 1

        if pred_label == gold_label:
            correct += 1
            label_stats[gold_label]["tp"] += 1
        else:
            label_stats[gold_label]["fn"] += 1
            if pred_label is not None:
                label_stats[pred_label]["fp"] += 1
            errors.append(
                {
                    "doc_id": doc_id,
                    "filename": gold_rows[doc_id].get("filename"),
                    "gold": gold_label,
                    "predicted": pred_label,
                }
            )

    accuracy = correct / total if total else 0.0

    print(f"Predictions: {args.predictions}")
    print(f"Gold:        {args.gold}")
    print(f"Compared:    {total}")
    print(f"Correct:     {correct}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Missing predictions: {len(missing_predictions)}")
    print(f"Extra predictions:   {len(extra_predictions)}")

    print("\nPer-label metrics:")
    labels = sorted(set(gold_counts) | {label for label in pred_counts if label})
    for label in labels:
        tp = label_stats[label]["tp"]
        fp = label_stats[label]["fp"]
        fn = label_stats[label]["fn"]
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        print(
            f"- {label}: "
            f"support={gold_counts[label]} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f}"
        )

    if args.show_errors and errors:
        print("\nErrors:")
        for err in errors:
            print(
                f"- {err['doc_id']} | {err['filename']} | "
                f"gold={err['gold']} | predicted={err['predicted']}"
            )


if __name__ == "__main__":
    main()
