from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


DEFAULT_RUNS = {
    "tfidf": ROOT / "data" / "classified" / "classification_results__tfidf.jsonl",
    "embedding": ROOT / "data" / "classified" / "classification_results__embedding.jsonl",
    "embedding_large": ROOT / "data" / "classified" / "classification_results__embedding_large.jsonl",
    "zeroshot": ROOT / "data" / "classified" / "classification_results__zeroshot.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ground_truth() -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(GT_PATH))[["doc_id", "true_label"]]


def load_predictions(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Skipping {name}: {path} not found")
        return pd.DataFrame(columns=["doc_id", f"{name}_prediction", f"{name}_confidence", f"{name}_margin", f"{name}_used_field"])

    df = pd.DataFrame(read_jsonl(path))
    for col in ["doc_id", "predicted_label", "confidence", "margin", "used_field"]:
        if col not in df.columns:
            df[col] = None

    return df[["doc_id", "predicted_label", "confidence", "margin", "used_field"]].rename(
        columns={
            "predicted_label": f"{name}_prediction",
            "confidence": f"{name}_confidence",
            "margin": f"{name}_margin",
            "used_field": f"{name}_used_field",
        }
    )


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def parse_run_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--run values must use name=path format")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Run name cannot be empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_arg,
        help="Classification run in name=path format. Defaults to tfidf, embedding, zeroshot.",
    )
    parser.add_argument("--name", default="classification_model_comparison")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Directory to write output files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = dict(args.run) if args.run else {k: v for k, v in DEFAULT_RUNS.items() if v.exists()}
    if not runs:
        print("No prediction files found — skipping comparison.")
        return
    gt_df = load_ground_truth()

    summary_rows: list[dict[str, Any]] = []
    comparison_df = gt_df.copy()

    for name, path in runs.items():
        pred_df = load_predictions(name, Path(path))
        eval_df = gt_df.merge(pred_df, on="doc_id", how="left")
        y_true = eval_df["true_label"].tolist()
        y_pred = eval_df[f"{name}_prediction"].fillna("MISSING").tolist()
        correct = eval_df["true_label"] == eval_df[f"{name}_prediction"]

        _, _, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=TARGET_LABELS,
            zero_division=0,
        )
        active = [i for i, count in enumerate(support) if count > 0]
        macro_f1 = f1[active].mean() if active else 0.0
        weighted_f1 = (f1 * support).sum() / support.sum() if support.sum() else 0.0

        summary_rows.append(
            {
                "model": name,
                "prediction_file": str(path),
                "documents": len(eval_df),
                "correct": int(correct.sum()),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "needs_review": int((eval_df[f"{name}_prediction"] == "Needs_Review").sum()),
                "anonymized_field_docs": int((eval_df[f"{name}_used_field"] == "anonymized_text").sum()),
            }
        )

        comparison_df = comparison_df.merge(pred_df, on="doc_id", how="left")
        comparison_df[f"{name}_correct"] = (
            comparison_df["true_label"] == comparison_df[f"{name}_prediction"]
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "weighted_f1"], ascending=False
    )

    base_name = args.name.strip() or "classification_model_comparison"
    csv_path = out_dir / f"{base_name}.csv"
    md_path = out_dir / f"{base_name}.md"

    comparison_df.to_csv(csv_path, index=False)

    display_summary = summary_df.copy()
    for col in ["accuracy", "macro_f1", "weighted_f1"]:
        display_summary[col] = display_summary[col].map(fmt_pct)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Classification Model Comparison\n\n")
        f.write("## Summary\n\n")
        f.write(display_summary.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Per-Document Predictions\n\n")
        prediction_cols = ["doc_id", "true_label"]
        for name in runs:
            prediction_cols.extend([f"{name}_prediction", f"{name}_confidence", f"{name}_correct"])
        per_doc = comparison_df[prediction_cols].copy()
        for col in per_doc.columns:
            if col.endswith("_confidence"):
                per_doc[col] = per_doc[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        f.write(per_doc.to_markdown(index=False))
        f.write("\n")

    print(f"Saved: {md_path}")
    print(f"Saved: {csv_path}")
    print()
    print(display_summary[["model", "correct", "documents", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
