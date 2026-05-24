"""
Create visual comparisons for taxonomy ablation results.

Usage:
  python src/taxonomy/plot_taxonomy_ablation.py \
    --input data/outputs/ablation/taxonomy_ablation_results.csv \
    --out_dir data/outputs/ablation/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def label_from_row(row: Dict[str, Any]) -> Optional[str]:
    label = row.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()

    taxonomy = row.get("taxonomy")
    if isinstance(taxonomy, dict):
        value = taxonomy.get("level_3") or taxonomy.get("level_2") or taxonomy.get("level_1")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def as_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def score_value(row: Dict[str, str]) -> float:
    return as_float(row, "path_agreement") or as_float(row, "agreement") or as_float(row, "accuracy")


def f1_value(row: Dict[str, str]) -> float:
    return as_float(row, "level_3_macro_f1") or as_float(row, "macro_f1")


def short_name(name: str) -> str:
    return (
        name.replace("hierarchical_", "hier_")
        .replace("evidence_", "ev_")
        .replace("baseline_keyword", "baseline")
    )


def color_for(kind: str) -> str:
    return {
        "hierarchical_embeddings": "#2f6fbd",
        "chunk_evidence": "#2c8f5b",
        "flat_classifier": "#c05a2b",
        "topdown_hierarchical_classifier": "#7b4dbb",
        "baseline": "#8a8f98",
    }.get(kind, "#666666")


def save_bar_accuracy(rows: List[Dict[str, str]], out_path: Path) -> None:
    rows = sorted(rows, key=score_value)
    names = [short_name(row["experiment"]) for row in rows]
    values = [score_value(row) for row in rows]
    colors = [color_for(row["kind"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("Full-path agreement")
    ax.set_title("Taxonomy Ablation Agreement")
    ax.set_xlim(0, max(0.6, max(values) + 0.06))
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for idx, value in enumerate(values):
        ax.text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_metric_grouped(rows: List[Dict[str, str]], out_path: Path) -> None:
    rows = sorted(rows, key=lambda row: int(row["rank"]))[:10]
    names = [short_name(row["experiment"]) for row in rows]
    accuracy = [score_value(row) for row in rows]
    macro_f1 = [f1_value(row) for row in rows]

    x = list(range(len(rows)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar([i - width / 2 for i in x], accuracy, width, label="Path agreement", color="#2f6fbd")
    ax.bar([i + width / 2 for i in x], macro_f1, width, label="Level 3 macro F1", color="#d28b26")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0, max(0.65, max(max(accuracy), max(macro_f1)) + 0.08))
    ax.set_ylabel("Score")
    ax.set_title("Top 10 Taxonomy Ablation Runs")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_alpha_curve(rows: List[Dict[str, str]], out_path: Path) -> None:
    series = {
        "hierarchical_embeddings": [],
        "chunk_evidence": [],
    }

    for row in rows:
        name = row["experiment"]
        if name.startswith("hierarchical_alpha_"):
            alpha = float(name.replace("hierarchical_alpha_", "").replace("_", "."))
            series["hierarchical_embeddings"].append((alpha, score_value(row), f1_value(row)))
        elif name.startswith("evidence_alpha_"):
            alpha = float(name.replace("evidence_alpha_", "").replace("_", "."))
            series["chunk_evidence"].append((alpha, score_value(row), f1_value(row)))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for kind, values in series.items():
        values.sort(key=lambda item: item[0])
        if not values:
            continue
        label = "Hierarchical" if kind == "hierarchical_embeddings" else "Chunk evidence"
        color = color_for(kind)
        ax.plot(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            linewidth=2,
            label=f"{label} path agreement",
            color=color,
        )
        ax.plot(
            [item[0] for item in values],
            [item[2] for item in values],
            marker="s",
            linewidth=1.5,
            linestyle="--",
            label=f"{label} Level 3 macro F1",
            color=color,
            alpha=0.75,
        )

    ax.set_xlabel("Embedding weight alpha")
    ax.set_ylabel("Score")
    ax.set_title("Effect of Embedding/Keyword Weight")
    ax.set_ylim(0, 0.65)
    ax.grid(linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_topk_curve(rows: List[Dict[str, str]], out_path: Path) -> None:
    values = []
    for row in rows:
        name = row["experiment"]
        if name.startswith("evidence_topk_"):
            top_k = int(name.replace("evidence_topk_", ""))
            values.append((top_k, score_value(row), f1_value(row)))

    values.sort(key=lambda item: item[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([item[0] for item in values], [item[1] for item in values], marker="o", label="Path agreement", color="#2c8f5b")
    ax.plot([item[0] for item in values], [item[2] for item in values], marker="s", linestyle="--", label="Level 3 macro F1", color="#d28b26")
    ax.set_xticks([item[0] for item in values])
    ax.set_xlabel("Top evidence chunks aggregated")
    ax.set_ylabel("Score")
    ax.set_title("Chunk Evidence Aggregation Ablation")
    ax.set_ylim(0, 0.6)
    ax.grid(linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(
    gold_path: Path,
    prediction_path: Path,
    out_path: Path,
    title: str,
) -> None:
    gold = {row["doc_id"]: label_from_row(row) for row in read_jsonl(gold_path)}
    predictions = {row["doc_id"]: label_from_row(row) for row in read_jsonl(prediction_path)}

    labels = sorted(set(gold.values()) | set(predictions.values()))
    if any(predictions.get(doc_id) is None for doc_id in gold):
        labels.append("Unassigned")

    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for doc_id, gold_label in gold.items():
        pred_label = predictions.get(doc_id) or "Unassigned"
        if gold_label not in index:
            continue
        if pred_label not in index:
            labels.append(pred_label)
            index[pred_label] = len(labels) - 1
            for row in matrix:
                row.append(0)
            matrix.append([0 for _ in labels])
        matrix[index[gold_label]][index[pred_label]] += 1

    short_labels = [
        label.replace(" / ", "/").replace(" Documents", "").replace(" Communications", "").replace(" (Financial / Incident / Audit)", "")
        for label in labels
    ]

    fig_size = max(6.5, len(labels) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Manual gold label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short_labels, rotation=35, ha="right")
    ax.set_yticklabels(short_labels)

    max_value = max(max(row) for row in matrix) if matrix else 0
    threshold = max_value / 2 if max_value else 0
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            color = "white" if value > threshold else "#1f2933"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/outputs/ablation/taxonomy_ablation_results.csv")
    ap.add_argument("--out_dir", default="data/outputs/ablation/figures")
    ap.add_argument("--labels", default="data/labels/taxonomy_gold_labels.jsonl")
    ap.add_argument("--gold", dest="labels", help=argparse.SUPPRESS)
    ap.add_argument("--learned_predictions", default="data/outputs/taxonomy_assignments_embeddings.jsonl")
    ap.add_argument("--flat_predictions", default="data/outputs/ablation/flat_tfidf_logreg_loo.jsonl")
    ap.add_argument("--topdown_predictions", default="data/outputs/ablation/topdown_tfidf_logreg_loo.jsonl")
    ap.add_argument("--flat_svm_predictions", default="data/outputs/ablation/flat_tfidf_svm_loo.jsonl")
    ap.add_argument("--topdown_svm_predictions", default="data/outputs/ablation/topdown_tfidf_svm_loo.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.input))

    save_bar_accuracy(rows, out_dir / "taxonomy_ablation_accuracy.png")
    save_metric_grouped(rows, out_dir / "taxonomy_ablation_top10.png")
    save_alpha_curve(rows, out_dir / "taxonomy_ablation_alpha_curve.png")
    save_topk_curve(rows, out_dir / "taxonomy_ablation_topk_curve.png")
    save_confusion_matrix(
        gold_path=Path(args.labels),
        prediction_path=Path(args.learned_predictions),
        out_path=out_dir / "taxonomy_confusion_hierarchical_alpha_0_7.png",
        title="Confusion Matrix: Best Learned Baseline",
    )
    if Path(args.flat_predictions).exists():
        save_confusion_matrix(
            gold_path=Path(args.labels),
            prediction_path=Path(args.flat_predictions),
            out_path=out_dir / "taxonomy_confusion_flat_tfidf_logreg_loo.png",
            title="Confusion Matrix: Flat TF-IDF Logistic Regression",
        )
    if Path(args.topdown_predictions).exists():
        save_confusion_matrix(
            gold_path=Path(args.labels),
            prediction_path=Path(args.topdown_predictions),
            out_path=out_dir / "taxonomy_confusion_topdown_tfidf_logreg_loo.png",
            title="Confusion Matrix: Top-Down TF-IDF Logistic Regression",
        )
    if Path(args.flat_svm_predictions).exists():
        save_confusion_matrix(
            gold_path=Path(args.labels),
            prediction_path=Path(args.flat_svm_predictions),
            out_path=out_dir / "taxonomy_confusion_flat_tfidf_svm_loo.png",
            title="Confusion Matrix: Flat TF-IDF Linear SVM",
        )
    if Path(args.topdown_svm_predictions).exists():
        save_confusion_matrix(
            gold_path=Path(args.labels),
            prediction_path=Path(args.topdown_svm_predictions),
            out_path=out_dir / "taxonomy_confusion_topdown_tfidf_svm_loo.png",
            title="Confusion Matrix: Top-Down TF-IDF Linear SVM",
        )
    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
