# src/classification/supervised_baselines.py
"""Supervised TF-IDF baselines evaluated with stratified k-fold CV.

Models: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest.
All share the same TF-IDF preprocessing as the unsupervised TF-IDF cosine method
(ngram=[1,2], english stop words, lowercase, min_df=1) so the comparison is fair.

Outputs:
- outputs/classification_outputs/supervised_baselines_cv.json (raw per-fold + aggregate)
- outputs/classification_outputs/supervised_baselines_cv.md (human-readable table)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from eval_harness import (
    REPO_ROOT,
    aggregate_cv_metrics,
    class_distribution,
    load_dataset,
    score_predictions,
    stratified_kfold_indices,
)

OUTPUT_DIR = REPO_ROOT / "outputs" / "classification_outputs"

TFIDF_KWARGS = dict(
    ngram_range=(1, 2),
    stop_words="english",
    lowercase=True,
    min_df=1,
)

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0, random_state=42
    ),
    "LinearSVM": lambda: LinearSVC(C=1.0, class_weight="balanced", random_state=42),
    "MultinomialNB": lambda: MultinomialNB(),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}


def run_cv(examples, k: int = 5, seed: int = 42) -> dict:
    doc_ids = [ex.doc_id for ex in examples]
    texts = [ex.text for ex in examples]
    labels = [ex.label for ex in examples]
    class_labels = sorted(set(labels))
    folds = stratified_kfold_indices(labels, k=k, seed=seed)

    results: dict = {
        "n_examples": len(examples),
        "k_folds": k,
        "class_distribution": class_distribution(examples),
        "class_labels": class_labels,
        "models": {},
    }

    for model_name, factory in MODELS.items():
        fold_metrics = []
        predictions: list[dict] = []
        for fold_i, (train_idx, test_idx) in enumerate(folds):
            X_train = [texts[i] for i in train_idx]
            y_train = [labels[i] for i in train_idx]
            X_test = [texts[i] for i in test_idx]
            y_test = [labels[i] for i in test_idx]

            pipe = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                    ("clf", factory()),
                ]
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            for idx, pred in zip(test_idx, y_pred):
                predictions.append(
                    {
                        "doc_id": doc_ids[idx],
                        "fold": fold_i,
                        "y_true": labels[idx],
                        "y_pred": str(pred),
                    }
                )

            m = score_predictions(list(y_test), list(y_pred), class_labels)
            m["fold"] = fold_i
            fold_metrics.append(m)

        agg = aggregate_cv_metrics(fold_metrics)
        results["models"][model_name] = {
            "fold_metrics": fold_metrics,
            "aggregate": agg,
            "predictions": predictions,
        }
        print(
            f"{model_name:>20s}: "
            f"acc={agg['accuracy_mean']:.3f}±{agg['accuracy_std']:.3f}  "
            f"macroF1={agg['macro_f1_mean']:.3f}±{agg['macro_f1_std']:.3f}"
        )

    return results


def write_markdown(results: dict, path: Path) -> None:
    class_labels = results["class_labels"]
    lines = []
    lines.append("# Supervised Baselines — Stratified 5-Fold CV\n")
    lines.append(f"- N = {results['n_examples']} documents\n")
    lines.append(f"- k = {results['k_folds']} folds\n")
    lines.append("- Class distribution: " + ", ".join(
        f"{k}={v}" for k, v in sorted(results['class_distribution'].items())
    ) + "\n")
    lines.append(
        "- Shared TF-IDF preprocessing: ngram=[1,2], english stop words, "
        "lowercase, min_df=1, head_only=True, max_chars=6000\n"
    )
    lines.append("\n## Aggregate metrics (mean ± std across folds)\n\n")

    header = "| Model | Accuracy | Macro F1 | Weighted F1 |"
    sep = "| --- | --- | --- | --- |"
    for lbl in class_labels:
        header += f" F1 ({lbl}) |"
        sep += " --- |"
    lines.append(header + "\n")
    lines.append(sep + "\n")

    for model_name, info in results["models"].items():
        a = info["aggregate"]
        row = (
            f"| {model_name} "
            f"| {a['accuracy_mean']:.3f} ± {a['accuracy_std']:.3f} "
            f"| {a['macro_f1_mean']:.3f} ± {a['macro_f1_std']:.3f} "
            f"| {a['weighted_f1_mean']:.3f} ± {a['weighted_f1_std']:.3f} |"
        )
        for lbl in class_labels:
            key = f"f1_{lbl}_mean"
            row += f" {a.get(key, 0.0):.3f} |"
        lines.append(row + "\n")

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_dataset()
    print(f"Loaded {len(examples)} examples — {class_distribution(examples)}")

    results = run_cv(examples, k=5, seed=42)

    json_path = OUTPUT_DIR / "supervised_baselines_cv.json"
    md_path = OUTPUT_DIR / "supervised_baselines_cv.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    write_markdown(results, md_path)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
