"""
Evaluate taxonomy assignments against manual single-annotator taxonomy labels.

The evaluator reports the hierarchy level by level instead of collapsing the
taxonomy to only Level 3. This makes weak spots visible when a prediction gets
the broad banking domain right but misses the functional category or topic.

Usage:
  python src/taxonomy/evaluate_taxonomy_accuracy.py \
    --predictions data/outputs/taxonomy_assignments_evidence.jsonl \
    --labels data/labels/taxonomy_gold_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LEVEL_KEYS = ("level_1", "level_2", "level_3")


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


def get_label(row: Dict[str, Any]) -> Optional[str]:
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


def get_taxonomy(row: Dict[str, Any]) -> Dict[str, Optional[str]]:
    taxonomy = row.get("taxonomy")
    if isinstance(taxonomy, dict):
        return {
            key: taxonomy.get(key).strip() if isinstance(taxonomy.get(key), str) and taxonomy.get(key).strip() else None
            for key in LEVEL_KEYS
        }

    label = get_label(row)
    return {"level_1": None, "level_2": None, "level_3": label}


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_labels(
    gold_values: Dict[str, Optional[str]],
    pred_values: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    common_ids = sorted(set(gold_values) & set(pred_values))
    correct = 0
    gold_counts: Counter[str] = Counter()
    pred_counts: Counter[Optional[str]] = Counter()
    label_stats: Dict[str, Counter[str]] = defaultdict(Counter)
    confusion: Counter[tuple[str, Optional[str]]] = Counter()
    errors: List[Dict[str, Any]] = []

    for doc_id in common_ids:
        gold_label = gold_values[doc_id]
        pred_label = pred_values[doc_id]
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
            errors.append({"doc_id": doc_id, "gold": gold_label, "predicted": pred_label})

    total = sum(gold_counts.values())
    labels = sorted(set(gold_counts) | {label for label in pred_counts if label})
    macro_f1_values: List[float] = []
    per_label: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = label_stats[label]["tp"]
        fp = label_stats[label]["fp"]
        fn = label_stats[label]["fn"]
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        if gold_counts[label] > 0:
            macro_f1_values.append(f1)
        per_label[label] = {
            "support": float(gold_counts[label]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0,
        "per_label": per_label,
        "confusion": confusion,
        "errors": errors,
    }


def path_value(row: Dict[str, Any]) -> Optional[str]:
    taxonomy = get_taxonomy(row)
    if all(taxonomy.get(key) for key in LEVEL_KEYS):
        return " > ".join(str(taxonomy[key]) for key in LEVEL_KEYS)
    return get_label(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Predicted taxonomy JSONL")
    ap.add_argument("--labels", default="data/labels/taxonomy_gold_labels.jsonl", help="Manual single-annotator label JSONL")
    ap.add_argument("--gold", dest="labels", help=argparse.SUPPRESS)
    ap.add_argument("--show_errors", action="store_true", help="Print each mismatched document")
    ap.add_argument(
        "--write_json",
        help="Optional path for a machine-readable metrics report.",
    )
    args = ap.parse_args()

    gold_rows = {row["doc_id"]: row for row in read_jsonl(Path(args.labels))}
    pred_rows = {row["doc_id"]: row for row in read_jsonl(Path(args.predictions))}

    missing_predictions = sorted(set(gold_rows) - set(pred_rows))
    extra_predictions = sorted(set(pred_rows) - set(gold_rows))
    common_ids = sorted(set(gold_rows) & set(pred_rows))

    level_reports: Dict[str, Dict[str, Any]] = {}
    for level_key in LEVEL_KEYS:
        level_reports[level_key] = evaluate_labels(
            {doc_id: get_taxonomy(row).get(level_key) for doc_id, row in gold_rows.items()},
            {doc_id: get_taxonomy(row).get(level_key) for doc_id, row in pred_rows.items()},
        )
    level_reports["path"] = evaluate_labels(
        {doc_id: path_value(row) for doc_id, row in gold_rows.items()},
        {doc_id: path_value(row) for doc_id, row in pred_rows.items()},
    )

    print(f"Predictions: {args.predictions}")
    print(f"Labels:      {args.labels} (manual single-annotator)")
    print(f"Compared:    {len(common_ids)} common documents")
    print(f"Missing predictions: {len(missing_predictions)}")
    print(f"Extra predictions:   {len(extra_predictions)}")

    print("\nHierarchy metrics:")
    for key, title in [
        ("level_1", "Level 1"),
        ("level_2", "Level 2"),
        ("level_3", "Level 3"),
        ("path", "Full path"),
    ]:
        report = level_reports[key]
        print(
            f"- {title}: correct={report['correct']}/{report['total']} "
            f"accuracy={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f}"
        )

    print("\nPer-label metrics at Level 3:")
    for label, metrics in sorted(level_reports["level_3"]["per_label"].items()):
        print(
            f"- {label}: "
            f"support={int(metrics['support'])} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )

    if args.show_errors and level_reports["path"]["errors"]:
        print("\nFull-path errors:")
        for err in level_reports["path"]["errors"]:
            filename = gold_rows[err["doc_id"]].get("filename")
            print(
                f"- {err['doc_id']} | {filename} | "
                f"gold={err['gold']} | predicted={err['predicted']}"
            )

    if args.write_json:
        serializable = {
            key: {
                metric_key: metric_value
                for metric_key, metric_value in report.items()
                if metric_key != "confusion"
            }
            for key, report in level_reports.items()
        }
        out_path = Path(args.write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
