from pathlib import Path
import argparse
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
GT_PATH = ROOT / "data" / "eval" / "classification_gt.jsonl"
DEFAULT_OUT_DIR = ROOT / "outputs" / "classification_outputs"

TARGET_LABELS = [
    "Policy_Procedure_Contract",
    "Reports",
    "Internal_Communications",
    "Emails",
    "HR_Documents",
    "Forms_Structured",
]
LABEL_SHORT = {
    "Policy_Procedure_Contract": "Policy",
    "Reports": "Reports",
    "Internal_Communications": "Internal",
    "Emails": "Emails",
    "HR_Documents": "HR",
    "Forms_Structured": "Forms",
    "Needs_Review": "Needs_Review",
    "MISSING": "MISSING",
}


def load_ground_truth(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)[["doc_id", "true_label"]]


def load_predictions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    for col in ["doc_id", "predicted_label", "confidence", "second_best", "second_score", "margin", "method"]:
        if col not in df.columns:
            df[col] = None
    return df[["doc_id", "predicted_label", "confidence", "second_best", "second_score", "margin", "method"]]


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "data" / "classified" / "classification_results.jsonl"),
                    help="Path to predictions JSONL")
    ap.add_argument("--name", default="", help="Report name suffix, e.g. v1_zeroshot")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Directory to write report files")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.pred)
    suffix = f"__{args.name}" if args.name else ""
    method_name = args.name.replace("_", " ").upper() if args.name else pred_path.stem

    pred_df = load_predictions(pred_path)
    gt_df = load_ground_truth(GT_PATH)

    eval_df = (
        gt_df.merge(pred_df, on="doc_id", how="left")
        .sort_values("doc_id")
        .reset_index(drop=True)
    )

    y_true = eval_df["true_label"].tolist()
    y_pred = eval_df["predicted_label"].fillna("MISSING").tolist()
    eval_df["is_correct"] = eval_df["true_label"] == eval_df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)
    correct = int(eval_df["is_correct"].sum())
    needs_review = int((eval_df["predicted_label"] == "Needs_Review").sum())

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_LABELS, zero_division=0,
    )
    active = [i for i, s in enumerate(support) if s > 0]
    macro_f1 = f1[active].mean() if len(active) else 0.0
    weighted_f1 = (
        (f1 * support).sum() / support.sum() if support.sum() else 0.0
    )

    cm = confusion_matrix(y_true, y_pred, labels=TARGET_LABELS)

    wrong_df = eval_df.loc[~eval_df["is_correct"]].copy()

    # --- Write Markdown ---
    md_path = out_dir / f"classification_report{suffix}.md"
    with md_path.open("w", encoding="utf-8") as f:

        f.write(f"# Classification Report — {method_name}\n\n")

        # 1. Overview
        f.write("## Overview\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Documents evaluated | {len(eval_df)} |\n")
        f.write(f"| Correct predictions | {correct} / {len(eval_df)} ({fmt_pct(accuracy)}) |\n")
        f.write(f"| Needs Review | {needs_review} |\n")
        f.write(f"| Macro F1 (active classes) | {fmt_pct(macro_f1)} |\n")
        f.write(f"| Weighted F1 | {fmt_pct(weighted_f1)} |\n\n")

        # 2. Per-class performance
        f.write("## Per-Class Performance\n\n")
        rows = []
        for i, lbl in enumerate(TARGET_LABELS):
            if support[i] == 0:
                continue
            correct_i = cm[i, i]
            rows.append({
                "Label": lbl,
                "Support": int(support[i]),
                "Correct": correct_i,
                "Accuracy": fmt_pct(correct_i / support[i]),
                "Precision": fmt_pct(p[i]),
                "Recall": fmt_pct(r[i]),
                "F1": fmt_pct(f1[i]),
            })
        f.write(pd.DataFrame(rows).to_markdown(index=False))
        f.write("\n\n")

        # 3. Confusion matrix (short label names)
        f.write("## Confusion Matrix\n\n")
        active_labels = [TARGET_LABELS[i] for i in active]
        short_labels = [LABEL_SHORT[l] for l in active_labels]
        cm_active = cm[active][:, active]
        cm_df = pd.DataFrame(cm_active, index=[f"TRUE {s}" for s in short_labels], columns=short_labels)
        f.write(cm_df.to_markdown())
        f.write("\n\n")

        # 4. Misclassified docs
        f.write("## Misclassified Documents\n\n")
        if wrong_df.empty:
            f.write("No misclassified documents.\n\n")
        else:
            mis = wrong_df[["doc_id", "true_label", "predicted_label", "confidence", "margin"]].copy()
            mis["confidence"] = mis["confidence"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            mis["margin"] = mis["margin"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            f.write(mis.to_markdown(index=False))
            f.write("\n\n")

    # --- Write single CSV ---
    csv_path = out_dir / f"doc_level_eval{suffix}.csv"
    eval_df.to_csv(csv_path, index=False)

    print(f"Saved: {md_path}")
    print(f"Saved: {csv_path}")
    print(f"\nAccuracy : {fmt_pct(accuracy)}  ({correct}/{len(eval_df)})")
    print(f"Macro F1 : {fmt_pct(macro_f1)}   Weighted F1: {fmt_pct(weighted_f1)}")
    print(f"Needs Review: {needs_review}")
    print()
    for row in rows:
        print(f"  {row['Label']:<35} F1={row['F1']}  ({row['Correct']}/{row['Support']})")


if __name__ == "__main__":
    main()
