"""
Run taxonomy ablation experiments.

This script reruns taxonomy assigners with controlled settings and evaluates
each output against the current label file.

Important: the repository label file is a manual single-annotator label set.
The ablation intentionally excludes deterministic rule-normalization because it
can produce circular agreement when rules and labels are developed together.

Usage:
  python src/taxonomy/run_taxonomy_ablation.py

Outputs:
  outputs/taxonomy/ablation/taxonomy_ablation_results.csv
  outputs/taxonomy/ablation/taxonomy_ablation_results.md
  outputs/taxonomy/ablation/*.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LEVEL_KEYS = ("level_1", "level_2", "level_3")


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    output_path: Path
    args: List[str]
    notes: str


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
        value = taxonomy.get("level_3") or taxonomy.get("level_2") or taxonomy.get("level_1")
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


def get_path_label(row: Dict[str, Any]) -> Optional[str]:
    taxonomy = get_taxonomy(row)
    if all(taxonomy.get(key) for key in LEVEL_KEYS):
        return " > ".join(str(taxonomy[key]) for key in LEVEL_KEYS)
    return get_label(row)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_label_values(
    common_ids: List[str],
    gold_values: Dict[str, Optional[str]],
    pred_values: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    correct = 0
    gold_counts: Counter[str] = Counter()
    pred_counts: Counter[Optional[str]] = Counter()
    stats: Dict[str, Counter[str]] = defaultdict(Counter)

    for doc_id in common_ids:
        gold_label = gold_values[doc_id]
        pred_label = pred_values[doc_id]

        if gold_label is None:
            continue

        gold_counts[gold_label] += 1
        pred_counts[pred_label] += 1

        if pred_label == gold_label:
            correct += 1
            stats[gold_label]["tp"] += 1
        else:
            stats[gold_label]["fn"] += 1
            if pred_label is not None:
                stats[pred_label]["fp"] += 1

    total = sum(gold_counts.values())
    labels = sorted(set(gold_counts) | {label for label in pred_counts if label})
    macro_f1_values: List[float] = []
    per_label: Dict[str, Dict[str, float]] = {}

    for label in labels:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]
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
        "agreement": correct / total if total else 0.0,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0,
        "per_label": per_label,
    }


def evaluate(prediction_path: Path, gold_path: Path) -> Dict[str, Any]:
    gold_rows = {row["doc_id"]: row for row in read_jsonl(gold_path)}
    pred_rows = {row["doc_id"]: row for row in read_jsonl(prediction_path)}
    common_ids = sorted(set(gold_rows) & set(pred_rows))

    status_counts: Counter[str] = Counter()
    for doc_id in common_ids:
        status = pred_rows[doc_id].get("status")
        if isinstance(status, str):
            status_counts[status] += 1

    level_metrics = {
        level_key: evaluate_label_values(
            common_ids,
            {doc_id: get_taxonomy(row).get(level_key) for doc_id, row in gold_rows.items()},
            {doc_id: get_taxonomy(row).get(level_key) for doc_id, row in pred_rows.items()},
        )
        for level_key in LEVEL_KEYS
    }
    path_metrics = evaluate_label_values(
        common_ids,
        {doc_id: get_path_label(row) for doc_id, row in gold_rows.items()},
        {doc_id: get_path_label(row) for doc_id, row in pred_rows.items()},
    )

    return {
        "total": path_metrics["total"],
        "correct": path_metrics["correct"],
        "agreement": path_metrics["agreement"],
        "macro_f1": path_metrics["macro_f1"],
        "level_1_agreement": level_metrics["level_1"]["agreement"],
        "level_2_agreement": level_metrics["level_2"]["agreement"],
        "level_3_agreement": level_metrics["level_3"]["agreement"],
        "path_agreement": path_metrics["agreement"],
        "level_1_macro_f1": level_metrics["level_1"]["macro_f1"],
        "level_2_macro_f1": level_metrics["level_2"]["macro_f1"],
        "level_3_macro_f1": level_metrics["level_3"]["macro_f1"],
        "assigned": status_counts.get("assigned", 0),
        "needs_review": status_counts.get("needs_review", 0),
        "unassigned": status_counts.get("unassigned", 0),
        "missing_predictions": len(set(gold_rows) - set(pred_rows)),
        "extra_predictions": len(set(pred_rows) - set(gold_rows)),
    }


def build_experiments(
    out_dir: Path,
    input_path: Path,
    chunks_path: Path,
    taxonomy_path: Path,
) -> List[Experiment]:
    experiments: List[Experiment] = [
        Experiment(
            name="baseline_keyword",
            kind="baseline",
            output_path=out_dir / "baseline_keyword.jsonl",
            args=[
                "src/taxonomy/assign_taxonomy_baseline.py",
                "--input",
                str(input_path),
                "--taxonomy",
                str(taxonomy_path),
                "--output",
                str(out_dir / "baseline_keyword.jsonl"),
            ],
            notes="Rule-based keyword baseline.",
        )
    ]

    experiments.extend(
        [
            Experiment(
                name="flat_tfidf_logreg_loo",
                kind="flat_classifier",
                output_path=out_dir / "flat_tfidf_logreg_loo.jsonl",
                args=[
                    "src/taxonomy/assign_taxonomy_literature_baselines.py",
                    "--input",
                    str(input_path),
                    "--labels",
                    "outputs/taxonomy/labels/taxonomy_gold_labels.jsonl",
                    "--output",
                    str(out_dir / "flat_tfidf_logreg_loo.jsonl"),
                    "--method",
                    "flat_tfidf_logreg",
                    "--train_mode",
                    "leave_one_out",
                ],
                notes="Literature baseline: flat TF-IDF + logistic regression over Level 3 labels; leave-one-out.",
            ),
            Experiment(
                name="topdown_tfidf_logreg_loo",
                kind="topdown_hierarchical_classifier",
                output_path=out_dir / "topdown_tfidf_logreg_loo.jsonl",
                args=[
                    "src/taxonomy/assign_taxonomy_literature_baselines.py",
                    "--input",
                    str(input_path),
                    "--labels",
                    "outputs/taxonomy/labels/taxonomy_gold_labels.jsonl",
                    "--output",
                    str(out_dir / "topdown_tfidf_logreg_loo.jsonl"),
                    "--method",
                    "topdown_tfidf_logreg",
                    "--train_mode",
                    "leave_one_out",
                ],
                notes="Literature baseline: top-down local TF-IDF + logistic regression classifiers; leave-one-out.",
            ),
            Experiment(
                name="flat_tfidf_svm_loo",
                kind="flat_classifier",
                output_path=out_dir / "flat_tfidf_svm_loo.jsonl",
                args=[
                    "src/taxonomy/assign_taxonomy_literature_baselines.py",
                    "--input",
                    str(input_path),
                    "--labels",
                    "outputs/taxonomy/labels/taxonomy_gold_labels.jsonl",
                    "--output",
                    str(out_dir / "flat_tfidf_svm_loo.jsonl"),
                    "--method",
                    "flat_tfidf_svm",
                    "--train_mode",
                    "leave_one_out",
                ],
                notes="Literature baseline: flat TF-IDF + Linear SVM over Level 3 labels; leave-one-out.",
            ),
            Experiment(
                name="topdown_tfidf_svm_loo",
                kind="topdown_hierarchical_classifier",
                output_path=out_dir / "topdown_tfidf_svm_loo.jsonl",
                args=[
                    "src/taxonomy/assign_taxonomy_literature_baselines.py",
                    "--input",
                    str(input_path),
                    "--labels",
                    "outputs/taxonomy/labels/taxonomy_gold_labels.jsonl",
                    "--output",
                    str(out_dir / "topdown_tfidf_svm_loo.jsonl"),
                    "--method",
                    "topdown_tfidf_svm",
                    "--train_mode",
                    "leave_one_out",
                ],
                notes="Literature baseline: top-down local TF-IDF + Linear SVM classifiers; leave-one-out.",
            ),
        ]
    )

    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        name = f"hierarchical_alpha_{alpha:.1f}".replace(".", "_")
        experiments.append(
            Experiment(
                name=name,
                kind="hierarchical_embeddings",
                output_path=out_dir / f"{name}.jsonl",
                args=[
                    "src/taxonomy/assign_taxonomy_embeddings.py",
                    "--input",
                    str(input_path),
                    "--taxonomy",
                    str(taxonomy_path),
                    "--output",
                    str(out_dir / f"{name}.jsonl"),
                    "--alpha",
                    str(alpha),
                ],
                notes=f"Whole-document hierarchical assigner; alpha={alpha}.",
            )
        )

    for alpha in [0.0, 0.3, 0.5, 0.68, 0.85, 1.0]:
        name = f"evidence_alpha_{alpha:.2f}".replace(".", "_")
        experiments.append(
            Experiment(
                name=name,
                kind="chunk_evidence",
                output_path=out_dir / f"{name}.jsonl",
                args=[
                    "src/taxonomy/assign_taxonomy_evidence.py",
                    "--input",
                    str(input_path),
                    "--chunks",
                    str(chunks_path),
                    "--taxonomy",
                    str(taxonomy_path),
                    "--output",
                    str(out_dir / f"{name}.jsonl"),
                    "--model",
                    "local_ngram",
                    "--alpha",
                    str(alpha),
                ],
                notes=f"Chunk-evidence assigner with local n-gram backend; alpha={alpha}.",
            )
        )

    for top_k in [1, 2, 4, 8]:
        name = f"evidence_topk_{top_k}"
        experiments.append(
            Experiment(
                name=name,
                kind="chunk_evidence",
                output_path=out_dir / f"{name}.jsonl",
                args=[
                    "src/taxonomy/assign_taxonomy_evidence.py",
                    "--input",
                    str(input_path),
                    "--chunks",
                    str(chunks_path),
                    "--taxonomy",
                    str(taxonomy_path),
                    "--output",
                    str(out_dir / f"{name}.jsonl"),
                    "--model",
                    "local_ngram",
                    "--alpha",
                    "0.68",
                    "--top_k_chunks",
                    str(top_k),
                ],
                notes=f"Chunk-evidence assigner with local n-gram backend; top_k_chunks={top_k}.",
            )
        )

    return experiments


def run_experiment(experiment: Experiment, skip_existing: bool) -> None:
    if skip_existing and experiment.output_path.exists():
        print(f"[SKIP] {experiment.name} -> {experiment.output_path}")
        return

    print(f"[RUN] {experiment.name}")
    subprocess.run([sys.executable, *experiment.args], check=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank",
        "experiment",
        "kind",
        "evaluation_role",
        "agreement",
        "path_agreement",
        "level_1_agreement",
        "level_2_agreement",
        "level_3_agreement",
        "macro_f1",
        "level_1_macro_f1",
        "level_2_macro_f1",
        "level_3_macro_f1",
        "correct",
        "total",
        "assigned",
        "needs_review",
        "unassigned",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Taxonomy Ablation Results",
        "",
        "These scores are agreement with the current taxonomy label file. Deterministic rule-normalized outputs are excluded to avoid circular 100% agreement.",
        "",
        "| Rank | Experiment | Kind | Role | Path agree. | L1 agree. | L2 agree. | L3 agree. | L3 macro F1 | Correct/Total | Notes |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{experiment}` | {kind} | {evaluation_role} | {path_agreement:.4f} | "
            "{level_1_agreement:.4f} | {level_2_agreement:.4f} | {level_3_agreement:.4f} | "
            "{level_3_macro_f1:.4f} | {correct}/{total} | {notes} |".format(**row)
        )
    lines.append("")
    lines.append("Gold labels: `outputs/taxonomy/labels/taxonomy_gold_labels.jsonl`.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/clean_documents.jsonl")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--labels", default="outputs/taxonomy/labels/taxonomy_gold_labels.jsonl")
    ap.add_argument("--gold", dest="labels", help=argparse.SUPPRESS)
    ap.add_argument("--out_dir", default="outputs/taxonomy/ablation")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(
        out_dir=out_dir,
        input_path=Path(args.input),
        chunks_path=Path(args.chunks),
        taxonomy_path=Path(args.taxonomy),
    )

    for experiment in experiments:
        run_experiment(experiment, skip_existing=args.skip_existing)

    rows: List[Dict[str, Any]] = []
    for experiment in experiments:
        metrics = evaluate(experiment.output_path, Path(args.labels))
        rows.append(
            {
                "experiment": experiment.name,
                "kind": experiment.kind,
                "evaluation_role": "baseline agreement",
                "agreement": metrics["agreement"],
                "macro_f1": metrics["macro_f1"],
                "level_1_agreement": metrics["level_1_agreement"],
                "level_2_agreement": metrics["level_2_agreement"],
                "level_3_agreement": metrics["level_3_agreement"],
                "path_agreement": metrics["path_agreement"],
                "level_1_macro_f1": metrics["level_1_macro_f1"],
                "level_2_macro_f1": metrics["level_2_macro_f1"],
                "level_3_macro_f1": metrics["level_3_macro_f1"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "assigned": metrics["assigned"],
                "needs_review": metrics["needs_review"],
                "unassigned": metrics["unassigned"],
                "notes": experiment.notes,
            }
        )

    rows.sort(
        key=lambda row: (
            row["path_agreement"],
            row["level_3_macro_f1"],
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    csv_path = out_dir / "taxonomy_ablation_results.csv"
    md_path = out_dir / "taxonomy_ablation_results.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print("\nTop experiments:")
    for row in rows[:10]:
        print(
            f"{row['rank']:>2}. {row['experiment']}: "
            f"path_agreement={row['path_agreement']:.4f}, "
            f"level_3_macro_f1={row['level_3_macro_f1']:.4f}, "
            f"correct={row['correct']}/{row['total']}"
        )
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
