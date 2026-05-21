# src/classification/threshold_calibration.py
"""Calibrate (min_score, min_margin) thresholds for the unsupervised TF-IDF
cosine classifier against the controlled label descriptions.

Methodology:
- Compute cosine similarity between each document's TF-IDF vector and each
  label-description TF-IDF vector (identical preprocessing to
  configs/classification_tfidf.yaml).
- For each (min_score, min_margin) grid point, derive a prediction (or
  Needs_Review) and measure macro F1 against the ground truth.
- We grid-search on the *full* joined dataset (no train/test split) because
  this method has no learned parameters — thresholds are calibrated, not
  trained. The result describes the operating point trade-off, not a held-out
  generalisation estimate.

Outputs:
- outputs/classification_outputs/threshold_calibration.json
- outputs/classification_outputs/threshold_calibration.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from eval_harness import REPO_ROOT, class_distribution, load_dataset, score_predictions

CONFIG_PATH = REPO_ROOT / "configs" / "classification_tfidf.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "classification_outputs"

# Grid covers the range called out in the report's "Open Issues" section.
MIN_SCORE_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]
MIN_MARGIN_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]


def build_candidate_labels(base_labels: list[str], descs: dict[str, str]) -> list[str]:
    return [f"{lbl} - {descs.get(lbl, '').strip()}" for lbl in base_labels]


def predict_with_thresholds(
    sims: np.ndarray,
    base_labels: list[str],
    min_score: float,
    min_margin: float,
) -> list[str]:
    preds: list[str] = []
    for row in sims:
        order = np.argsort(-row)
        best, second = order[0], order[1]
        top_score = float(row[best])
        margin = top_score - float(row[second])
        if top_score < min_score or margin < min_margin:
            preds.append("Needs_Review")
        else:
            preds.append(base_labels[best])
    return preds


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    base_labels = cfg["labels"]
    descs = cfg["label_descriptions"]
    candidate_texts = build_candidate_labels(base_labels, descs)

    # Restrict to classes present in the (CV-clean) joined dataset; the report
    # is honest that two labels have zero ground truth examples.
    examples = load_dataset()
    eval_class_labels = sorted(set(ex.label for ex in examples))

    print(f"Loaded {len(examples)} examples — {class_distribution(examples)}")
    print(f"Candidate label space (from config): {base_labels}")
    print(f"Evaluating against classes present in GT: {eval_class_labels}")

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        min_df=1,
    )
    doc_mat = vec.fit_transform([ex.text for ex in examples])
    label_mat = vec.transform(candidate_texts)
    sims = cosine_similarity(doc_mat, label_mat)  # (N, n_labels)

    y_true = [ex.label for ex in examples]

    # We report a Pareto frontier (best macro F1 at each coverage floor) plus
    # one "deployed" operating point at MIN_COVERAGE.
    MIN_COVERAGE = 0.90
    COVERAGE_FLOORS = [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]

    grid: list[dict] = []
    best = {"macro_f1_on_covered": -1.0, "coverage": -1.0}
    for ms in MIN_SCORE_GRID:
        for mm in MIN_MARGIN_GRID:
            y_pred = predict_with_thresholds(sims, base_labels, ms, mm)
            needs_review = sum(1 for p in y_pred if p == "Needs_Review")
            coverage = 1.0 - needs_review / len(y_pred)
            # Restrict scoring to docs we actually classified
            kept_idx = [i for i, p in enumerate(y_pred) if p != "Needs_Review"]
            if kept_idx:
                m = score_predictions(
                    [y_true[i] for i in kept_idx],
                    [y_pred[i] for i in kept_idx],
                    eval_class_labels,
                )
            else:
                m = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

            entry = {
                "min_score": ms,
                "min_margin": mm,
                "needs_review": needs_review,
                "coverage": coverage,
                "accuracy_on_covered": m["accuracy"],
                "macro_f1_on_covered": m["macro_f1"],
                "weighted_f1_on_covered": m["weighted_f1"],
            }
            grid.append(entry)
            # Selection: best macro F1 subject to coverage >= MIN_COVERAGE.
            if coverage >= MIN_COVERAGE:
                score_tuple = (m["macro_f1"], coverage)
                best_tuple = (best["macro_f1_on_covered"], best["coverage"])
                if score_tuple > best_tuple:
                    best = {**entry, "y_pred": y_pred}

    # Also report metrics if we *force* a prediction (no Needs_Review) at the
    # selected operating point — comparable to the supervised baselines that
    # always emit a label.
    y_pred_forced = predict_with_thresholds(sims, base_labels, 0.0, 0.0)
    forced_metrics = score_predictions(y_true, y_pred_forced, eval_class_labels)

    # Pareto frontier: best macro F1 attainable subject to each coverage floor.
    frontier = []
    for floor in COVERAGE_FLOORS:
        best_at = None
        for g in grid:
            if g["coverage"] < floor:
                continue
            if best_at is None or g["macro_f1_on_covered"] > best_at["macro_f1_on_covered"]:
                best_at = g
        frontier.append({"coverage_floor": floor, "best": best_at})

    summary = {
        "n_examples": len(examples),
        "class_distribution": class_distribution(examples),
        "label_space_in_config": base_labels,
        "label_space_in_eval": eval_class_labels,
        "min_score_grid": MIN_SCORE_GRID,
        "min_margin_grid": MIN_MARGIN_GRID,
        "grid": grid,
        "min_coverage": MIN_COVERAGE,
        "selected": {k: v for k, v in best.items() if k != "y_pred"},
        "forced_no_threshold": forced_metrics,
        "pareto_frontier": frontier,
    }

    json_path = OUTPUT_DIR / "threshold_calibration.json"
    md_path = OUTPUT_DIR / "threshold_calibration.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(summary, md_path)
    print(f"\nSelected operating point: {summary['selected']}")
    print(f"Forced (no threshold): {forced_metrics}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def write_markdown(summary: dict, path: Path) -> None:
    lines = []
    lines.append("# Threshold Calibration — TF-IDF Cosine\n\n")
    lines.append(f"- N = {summary['n_examples']} documents\n")
    lines.append("- Class distribution: " + ", ".join(
        f"{k}={v}" for k, v in sorted(summary['class_distribution'].items())
    ) + "\n")
    lines.append("- Label space in config: " + ", ".join(summary["label_space_in_config"]) + "\n")
    lines.append("- Classes with GT examples: " + ", ".join(summary["label_space_in_eval"]) + "\n\n")

    sel = summary["selected"]
    lines.append(f"## Selected operating point (coverage floor = {summary['min_coverage']})\n\n")
    if "min_score" not in sel:
        lines.append("- No grid point met the coverage floor — see frontier below.\n\n")
        sel = None
    if sel is not None:
        lines.append(
            f"- min_score = **{sel['min_score']}**, min_margin = **{sel['min_margin']}**\n"
        f"- coverage = {sel['coverage']:.3f} ({summary['n_examples'] - sel['needs_review']}/{summary['n_examples']})\n"
        f"- accuracy on covered = {sel['accuracy_on_covered']:.3f}\n"
        f"- macro F1 on covered = {sel['macro_f1_on_covered']:.3f}\n"
            f"- weighted F1 on covered = {sel['weighted_f1_on_covered']:.3f}\n\n"
        )

    lines.append("## Pareto frontier (best macro F1 at each coverage floor)\n\n")
    lines.append("| Coverage ≥ | min_score | min_margin | Coverage | Macro F1 | Accuracy | Needs_Review |\n")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |\n")
    for f in summary["pareto_frontier"]:
        b = f["best"]
        if b is None:
            lines.append(f"| {f['coverage_floor']:.2f} | – | – | – | – | – | – |\n")
        else:
            lines.append(
                f"| {f['coverage_floor']:.2f} "
                f"| {b['min_score']:.3f} | {b['min_margin']:.3f} "
                f"| {b['coverage']:.3f} | {b['macro_f1_on_covered']:.3f} "
                f"| {b['accuracy_on_covered']:.3f} | {b['needs_review']} |\n"
            )
    lines.append("\n")

    f = summary["forced_no_threshold"]
    lines.append("## No-threshold baseline (always predict)\n\n")
    lines.append(
        f"- accuracy = {f['accuracy']:.3f}\n"
        f"- macro F1 = {f['macro_f1']:.3f}\n"
        f"- weighted F1 = {f['weighted_f1']:.3f}\n\n"
    )

    lines.append("## Grid (macro F1 on covered docs)\n\n")
    header = "| min_score \\ min_margin |"
    sep = "| --- |"
    for mm in summary["min_margin_grid"]:
        header += f" {mm:.2f} |"
        sep += " --- |"
    lines.append(header + "\n")
    lines.append(sep + "\n")
    for ms in summary["min_score_grid"]:
        row = f"| {ms:.3f} |"
        for mm in summary["min_margin_grid"]:
            entry = next(
                g for g in summary["grid"]
                if abs(g["min_score"] - ms) < 1e-9 and abs(g["min_margin"] - mm) < 1e-9
            )
            row += f" {entry['macro_f1_on_covered']:.2f} ({entry['needs_review']}) |"
        lines.append(row + "\n")

    lines.append("\nCell format: `macro_f1 (n_needs_review)`\n")
    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
