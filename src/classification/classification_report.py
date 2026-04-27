from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
PRED_PATH = ROOT / "data" / "classified" / "classification_results.jsonl"
GT_PATH = ROOT / "data" / "eval" / "classification_gt.jsonl"
OUT_DIR = ROOT / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LABELS = ["Policy_Procedure_Contract", "Reports", "Internal_Communications", "Emails", "HR_Documents", "Forms_Structured"]


def load_ground_truth(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df[["doc_id", "true_label"]]


def load_predictions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    wanted_cols = [
        "doc_id",
        "predicted_label",
        "confidence",
        "second_best",
        "second_score",
        "margin",
    ]
    for col in wanted_cols:
        if col not in df.columns:
            df[col] = None

    return df[wanted_cols]


def main():
    pred_df = load_predictions(PRED_PATH)
    gt_df = load_ground_truth(GT_PATH)

    eval_df = (
        gt_df.merge(pred_df, on="doc_id", how="left")
        .sort_values("doc_id")
        .reset_index(drop=True)
    )

    # any label outside target set stays as-is in doc-level output,
 
    y_true = eval_df["true_label"].tolist()
    y_pred = eval_df["predicted_label"].fillna("MISSING").tolist()

    eval_df["is_correct"] = eval_df["true_label"] == eval_df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)

    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=TARGET_LABELS,
        zero_division=0,
    )

    metrics_df = pd.DataFrame(
        {
            "label": TARGET_LABELS,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support,
        }
    )

    macro_precision = metrics_df["precision"].mean()
    macro_recall = metrics_df["recall"].mean()
    macro_f1 = metrics_df["f1"].mean()
    weighted_f1 = (metrics_df["f1"] * metrics_df["support"]).sum() / metrics_df["support"].sum()

    summary_df = pd.DataFrame(
        [
            {
                "n_docs": len(eval_df),
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "correct_docs": int(eval_df["is_correct"].sum()),
                "wrong_docs": int((~eval_df["is_correct"]).sum()),
                "needs_review_count": int((eval_df["predicted_label"] == "Needs_Review").sum()),
            }
        ]
    )

    cm = confusion_matrix(y_true, y_pred, labels=TARGET_LABELS)
    cm_df = pd.DataFrame(cm, index=TARGET_LABELS, columns=TARGET_LABELS)

    wrong_df = eval_df.loc[~eval_df["is_correct"]].copy()

    # Confidence stats by predicted label
    conf_stats_df = (
        eval_df.groupby("predicted_label", dropna=False)
        .agg(
            count=("doc_id", "count"),
            avg_confidence=("confidence", "mean"),
            avg_margin=("margin", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    class_report_text = classification_report(
        y_true,
        y_pred,
        labels=TARGET_LABELS,
        zero_division=0,
        digits=4,
    )

    
    # Markdown report
    md_path = OUT_DIR / "classification_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Classification Evaluation Report\n\n")

        f.write("## 1. Overall Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 2. Per-Class Metrics\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 3. Confusion Matrix\n\n")
        f.write(cm_df.to_markdown())
        f.write("\n\n")

        f.write("## 4. Misclassified Documents\n\n")
        if wrong_df.empty:
            f.write("No misclassified documents.\n\n")
        else:
            cols = [
                "doc_id",
                "true_label",
                "predicted_label",
                "confidence",
                "second_best",
                "second_score",
                "margin",
            ]
            f.write(wrong_df[cols].to_markdown(index=False))
            f.write("\n\n")

        f.write("## 5. Prediction Confidence Statistics\n\n")
        f.write(conf_stats_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 6. sklearn Classification Report\n\n")
        f.write("```text\n")
        f.write(class_report_text)
        f.write("\n```\n")

    print("Saved:")
    print(OUT_DIR / "classification_report.md")
    print(OUT_DIR / "classification_summary.csv")
    print(OUT_DIR / "classification_metrics.csv")
    print(OUT_DIR / "confusion_matrix.csv")
    print(OUT_DIR / "misclassified_docs.csv")
    print(OUT_DIR / "prediction_confidence_stats.csv")
    print(OUT_DIR / "doc_level_eval.csv")

    print("\nOverall Summary:")
    print(summary_df.to_string(index=False))

    print("\nPer-Class Metrics:")
    print(metrics_df.to_string(index=False))

    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    print("\nClassification Report:")
    print(class_report_text)


if __name__ == "__main__":
    main()