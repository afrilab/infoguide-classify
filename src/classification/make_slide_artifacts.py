# src/classification/make_slide_artifacts.py
"""Build slide-ready artifacts (PNGs, CSVs, markdown tables) under
outputs/classification_results/.

What it produces:
- confusion_matrix__<method>.png  (one per supervised method + TF-IDF cosine)
- model_comparison_bar.png        (accuracy + macro F1 across all methods)
- per_class_f1_bar.png            (per-class F1 across methods)
- threshold_pareto.png            (coverage vs. macro F1 frontier)
- model_comparison.csv / .md      (clean side-by-side table)
- per_class_metrics.csv / .md     (per-class precision/recall/F1)

Sources:
- Runs lexical supervised baselines (LR / SVM / NB / RF) inline (~10s).
- Runs unsupervised TF-IDF cosine at calibrated thresholds inline (~5s).
- Loads DistilBERT predictions from outputs/classification_outputs/finetune_distilbert_cv.json
  (run src/classification/finetune_transformer.py first).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
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

OUTPUT_DIR = REPO_ROOT / "outputs" / "classification_results"
DISTILBERT_JSON = REPO_ROOT / "outputs" / "classification_outputs" / "finetune_distilbert_cv.json"
TFIDF_CFG = REPO_ROOT / "configs" / "classification_tfidf.yaml"
BGE_SMALL_CFG = REPO_ROOT / "configs" / "classification_embedding.yaml"
BGE_LARGE_CFG = REPO_ROOT / "configs" / "classification_embedding_large.yaml"
ZEROSHOT_BART_CFG = REPO_ROOT / "configs" / "classification.yaml"
ZEROSHOT_DEBERTA_CFG = REPO_ROOT / "configs" / "classification_zeroshot_large.yaml"

TFIDF_KWARGS = dict(ngram_range=(1, 2), stop_words="english", lowercase=True, min_df=1)

LEXICAL_MODELS = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0, random_state=42
    ),
    "Linear SVM": lambda: LinearSVC(C=1.0, class_weight="balanced", random_state=42),
    "Multinomial NB": lambda: MultinomialNB(),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

# Selected by threshold_calibration.py — see docs/classification_design_evaluation.md
TFIDF_COSINE_MIN_SCORE = 0.0
TFIDF_COSINE_MIN_MARGIN = 0.01

PLOT_STYLE = {
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "DejaVu Sans",
}


def cv_predictions_lexical(
    examples, model_factory, class_labels, folds
) -> tuple[list[str], list[str], list[dict]]:
    """Run CV and return (y_true_all, y_pred_all, fold_metrics)."""
    texts = [ex.text for ex in examples]
    labels = [ex.label for ex in examples]

    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    fold_metrics: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train = [texts[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test = [texts[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
            ("clf", model_factory()),
        ])
        pipe.fit(X_train, y_train)
        y_pred = list(pipe.predict(X_test))

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        m = score_predictions(y_test, y_pred, class_labels)
        m["fold"] = fold_i
        fold_metrics.append(m)

    return y_true_all, y_pred_all, fold_metrics


def tfidf_cosine_predictions(examples, class_labels) -> tuple[list[str], list[str], dict]:
    """Unsupervised TF-IDF cosine to label descriptions, at the calibrated thresholds.
    Forces a prediction (no Needs_Review) so the confusion matrix is comparable to
    the supervised models that always emit a label.
    """
    cfg = yaml.safe_load(TFIDF_CFG.read_text(encoding="utf-8"))
    base_labels = cfg["labels"]
    descs = cfg["label_descriptions"]
    candidate_texts = [f"{lbl} - {descs.get(lbl, '').strip()}" for lbl in base_labels]

    vec = TfidfVectorizer(**TFIDF_KWARGS)
    doc_mat = vec.fit_transform([ex.text for ex in examples])
    label_mat = vec.transform(candidate_texts)
    sims = cosine_similarity(doc_mat, label_mat)

    y_true = [ex.label for ex in examples]
    y_pred = [base_labels[int(np.argmax(row))] for row in sims]

    m = score_predictions(y_true, y_pred, class_labels)
    return y_true, y_pred, m


def load_existing_predictions(
    examples, class_labels, pred_path: Path
) -> tuple[list[str], list[str], dict] | None:
    """Load predictions from data/classified/classification_results__*.jsonl and
    join with the 94-doc CV subset. Only docs present in both are scored.
    """
    if not pred_path.exists():
        print(f"    SKIP: {pred_path} not found")
        return None
    pred_map: dict[str, str] = {}
    with pred_path.open() as f:
        for line in f:
            r = json.loads(line)
            pred_map[r["doc_id"]] = r["predicted_label"]
    y_true: list[str] = []
    y_pred: list[str] = []
    missing = 0
    for ex in examples:
        if ex.doc_id not in pred_map:
            missing += 1
            continue
        y_true.append(ex.label)
        y_pred.append(pred_map[ex.doc_id])
    if missing:
        print(f"    WARN: {missing} docs missing from {pred_path.name}")
    if not y_true:
        return None
    m = score_predictions(y_true, y_pred, class_labels)
    return y_true, y_pred, m


def _build_candidates_from_cfg(cfg_path: Path) -> tuple[list[str], list[str], int]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base_labels = cfg["labels"]
    descs = cfg.get("label_descriptions", {})
    candidate_texts = [f"{lbl} - {descs.get(lbl, '').strip()}" for lbl in base_labels]
    max_chars = int(cfg.get("max_chars", 6000))
    return base_labels, candidate_texts, max_chars


def bge_embedding_predictions(
    examples, class_labels, cfg_path: Path
) -> tuple[list[str], list[str], dict]:
    """Single-pass cosine similarity between BGE document embeddings and BGE label
    description embeddings. No CV (no training)."""
    from sentence_transformers import SentenceTransformer

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_name = cfg["model_name"]
    query_instruction = (cfg.get("query_instruction") or "").strip()
    base_labels, candidate_texts, max_chars = _build_candidates_from_cfg(cfg_path)

    texts = [ex.text[:max_chars] for ex in examples]
    encode_texts = [query_instruction + " " + t if query_instruction else t for t in texts]

    model = SentenceTransformer(model_name)
    doc_emb = model.encode(encode_texts, convert_to_numpy=True, show_progress_bar=False)
    label_emb = model.encode(candidate_texts, convert_to_numpy=True, show_progress_bar=False)
    sim = cosine_similarity(doc_emb, label_emb)

    y_true = [ex.label for ex in examples]
    y_pred = [base_labels[int(np.argmax(row))] for row in sim]
    m = score_predictions(y_true, y_pred, class_labels)
    return y_true, y_pred, m


def zero_shot_predictions(
    examples, class_labels, cfg_path: Path
) -> tuple[list[str], list[str], dict]:
    """Zero-shot NLI: candidate-label entailment scoring. No CV (no training)."""
    import torch
    from transformers import pipeline

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_name = cfg["model_name"]
    hypothesis_template = cfg.get("hypothesis_template") or "The type of this document is {}."
    base_labels, candidate_texts, max_chars = _build_candidates_from_cfg(cfg_path)
    label_map = dict(zip(candidate_texts, base_labels))

    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
    clf = pipeline("zero-shot-classification", model=model_name, device=device)

    y_true: list[str] = []
    y_pred: list[str] = []
    for i, ex in enumerate(examples):
        text = ex.text[:max_chars]
        if not text.strip():
            y_pred.append("Needs_Review")
            y_true.append(ex.label)
            continue
        result = clf(
            sequences=text,
            candidate_labels=candidate_texts,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        top = result["labels"][0]
        y_pred.append(label_map.get(top, top))
        y_true.append(ex.label)
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(examples)}] processed")

    m = score_predictions(y_true, y_pred, class_labels)
    return y_true, y_pred, m


def load_distilbert_predictions(class_labels) -> tuple[list[str], list[str], dict] | None:
    if not DISTILBERT_JSON.exists():
        return None
    data = json.loads(DISTILBERT_JSON.read_text())
    if "predictions" not in data:
        return None
    y_true = [r["y_true"] for r in data["predictions"]]
    y_pred = [r["y_pred"] for r in data["predictions"]]
    m = score_predictions(y_true, y_pred, class_labels)
    return y_true, y_pred, m


def plot_confusion_matrix(
    y_true, y_pred, class_labels, title: str, out_path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(title, weight="bold", pad=12)
    ax.set_xlabel("Predicted", labelpad=6)
    ax.set_ylabel("True", labelpad=6)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.savefig(out_path)
    plt.close(fig)


def plot_model_comparison_bar(rows: list[dict], out_path: Path) -> None:
    """rows: [{name, accuracy, macro_f1, type}]"""
    rows = sorted(rows, key=lambda r: r["macro_f1"])
    names = [r["name"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    types = [r["type"] for r in rows]

    palette = {
        "Lexical (supervised)": "#3b82c4",
        "Transformer (supervised)": "#7c3aed",
        "Cosine (unsupervised)": "#f59e0b",
        "Zero-shot NLI": "#ef4444",
    }
    type_to_legend = {
        "Supervised (lexical)": "Lexical (supervised)",
        "Supervised (transformer)": "Transformer (supervised)",
        "Unsupervised (label desc)": "Cosine (unsupervised)",
        "Zero-shot NLI": "Zero-shot NLI",
    }
    types = [type_to_legend.get(t, t) for t in types]
    bar_colors = [palette.get(t, "#888") for t in types]

    y = np.arange(len(names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 0.65 * len(names) + 1.5))
    bars_acc = ax.barh(y - width / 2, accs, width, color=bar_colors,
                       alpha=0.45, edgecolor="black", linewidth=0.5)
    bars_f1 = ax.barh(y + width / 2, f1s, width, color=bar_colors,
                      alpha=1.0, edgecolor="black", linewidth=0.5,
                      hatch="//")

    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(a + 0.006, i - width / 2, f"{a:.3f}", va="center", fontsize=9.5)
        ax.text(f + 0.006, i + width / 2, f"{f:.3f}", va="center", fontsize=9.5, weight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score (5-fold CV, N=94)")
    ax.set_title("Model Comparison — Accuracy vs. Macro F1", weight="bold")
    ax.grid(axis="x", alpha=0.25)

    from matplotlib.patches import Patch
    family_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=l)
                      for l, c in palette.items()]
    metric_handles = [
        Patch(facecolor="white", edgecolor="black", linewidth=0.5, alpha=0.45, label="Accuracy (solid)"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.5, hatch="//", label="Macro F1 (hatched)"),
    ]
    leg1 = ax.legend(handles=family_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), fontsize=9, title="Method family",
                     title_fontsize=9, framealpha=0.95, borderaxespad=0.0)
    ax.add_artist(leg1)
    ax.legend(handles=metric_handles, loc="upper left",
              bbox_to_anchor=(1.02, 0.55), fontsize=9, title="Metric",
              title_fontsize=9, framealpha=0.95, borderaxespad=0.0)
    fig.savefig(out_path, bbox_extra_artists=(leg1,), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_per_class_f1(per_class: dict, class_labels, out_path: Path) -> None:
    """per_class: {method_name: {class: f1, ...}}"""
    methods = list(per_class.keys())
    x = np.arange(len(class_labels))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for i, m in enumerate(methods):
        vals = [per_class[m].get(c, 0.0) for c in class_labels]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m, color=colors[i])
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width - 0.4 + width / 2, v + 0.015,
                    f"{v:.2f}", ha="center", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=10, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-Class F1 Across Models", weight="bold")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    fontsize=9, title="Model", title_fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, bbox_extra_artists=(leg,), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_threshold_pareto(out_path: Path) -> None:
    """Read threshold_calibration.json and plot coverage vs macro F1."""
    src = REPO_ROOT / "outputs" / "classification_outputs" / "threshold_calibration.json"
    data = json.loads(src.read_text())
    grid = data["grid"]
    coverage = np.array([g["coverage"] for g in grid])
    macrof1 = np.array([g["macro_f1_on_covered"] for g in grid])
    # Pareto frontier: at each coverage level, the max F1 achievable at coverage >= that level
    order = np.argsort(-coverage)
    cov_sorted = coverage[order]
    f1_sorted = macrof1[order]
    frontier_cov, frontier_f1 = [], []
    best = -1.0
    for c, f in zip(cov_sorted, f1_sorted):
        if f > best:
            best = f
            frontier_cov.append(c)
            frontier_f1.append(f)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(coverage, macrof1, s=24, alpha=0.4, color="#888", label="grid point")
    ax.plot(frontier_cov, frontier_f1, "o-", color="#c0392b", linewidth=2,
            markersize=7, label="Pareto frontier")
    sel = data["selected"]
    if "coverage" in sel:
        ax.scatter([sel["coverage"]], [sel["macro_f1_on_covered"]],
                   marker="*", s=300, color="#2ecc71", edgecolor="black",
                   zorder=5, label=f"selected (min_score={sel['min_score']}, min_margin={sel['min_margin']})")
    ax.set_xlabel("Coverage (fraction confidently labeled)")
    ax.set_ylabel("Macro F1 on covered docs")
    ax.set_title("Threshold Calibration — Coverage vs. Macro F1", weight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)


def safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)

    examples = load_dataset()
    class_labels = sorted(set(ex.label for ex in examples))
    folds = stratified_kfold_indices([ex.label for ex in examples], k=5, seed=42)

    print(f"N={len(examples)}, classes={class_distribution(examples)}")

    # ---- Lexical supervised (CV) ----
    comparison_rows: list[dict] = []
    per_class_f1: dict[str, dict[str, float]] = {}
    per_class_table_rows: list[dict] = []

    for name, factory in LEXICAL_MODELS.items():
        y_true, y_pred, fold_metrics = cv_predictions_lexical(
            examples, factory, class_labels, folds
        )
        agg = aggregate_cv_metrics(fold_metrics)
        m_all = score_predictions(y_true, y_pred, class_labels)
        plot_confusion_matrix(
            y_true, y_pred, class_labels,
            title=f"{name} — aggregated 5-fold CV",
            out_path=OUTPUT_DIR / f"confusion_matrix__{safe_filename(name)}.png",
        )
        comparison_rows.append({
            "name": name,
            "type": "Supervised (lexical)",
            "accuracy": agg["accuracy_mean"],
            "accuracy_std": agg["accuracy_std"],
            "macro_f1": agg["macro_f1_mean"],
            "macro_f1_std": agg["macro_f1_std"],
            "weighted_f1": agg["weighted_f1_mean"],
        })
        per_class_f1[name] = {c: m_all[f"f1_{c}"] for c in class_labels}
        for c in class_labels:
            per_class_table_rows.append({
                "model": name, "class": c,
                "precision": m_all[f"precision_{c}"],
                "recall": m_all[f"recall_{c}"],
                "f1": m_all[f"f1_{c}"],
            })
        print(f"  {name}: acc={agg['accuracy_mean']:.3f}  macroF1={agg['macro_f1_mean']:.3f}")

    # ---- Unsupervised TF-IDF cosine (forced) ----
    yt, yp, m_cos = tfidf_cosine_predictions(examples, class_labels)
    plot_confusion_matrix(
        yt, yp, class_labels,
        title="TF-IDF cosine (label descriptions) — single pass",
        out_path=OUTPUT_DIR / "confusion_matrix__tfidf_cosine.png",
    )
    comparison_rows.append({
        "name": "TF-IDF cosine",
        "type": "Unsupervised (label desc)",
        "accuracy": m_cos["accuracy"],
        "accuracy_std": 0.0,
        "macro_f1": m_cos["macro_f1"],
        "macro_f1_std": 0.0,
        "weighted_f1": m_cos["weighted_f1"],
    })
    per_class_f1["TF-IDF cosine"] = {c: m_cos[f"f1_{c}"] for c in class_labels}
    for c in class_labels:
        per_class_table_rows.append({
            "model": "TF-IDF cosine", "class": c,
            "precision": m_cos[f"precision_{c}"],
            "recall": m_cos[f"recall_{c}"],
            "f1": m_cos[f"f1_{c}"],
        })
    print(f"  TF-IDF cosine: acc={m_cos['accuracy']:.3f}  macroF1={m_cos['macro_f1']:.3f}")

    # ---- BGE embeddings + Zero-shot NLI (load existing predictions) ----
    CLASSIFIED_DIR = REPO_ROOT / "data" / "classified"
    existing_runs = [
        ("BGE Small embedding",   "Unsupervised (label desc)", CLASSIFIED_DIR / "classification_results__embedding.jsonl"),
        ("BGE Large embedding",   "Unsupervised (label desc)", CLASSIFIED_DIR / "classification_results__embedding_large.jsonl"),
        ("BART zero-shot NLI",    "Zero-shot NLI",             CLASSIFIED_DIR / "classification_results__zeroshot.jsonl"),
        ("DeBERTa zero-shot NLI", "Zero-shot NLI",             CLASSIFIED_DIR / "classification_results__zeroshot_large.jsonl"),
    ]
    for label, mtype, pred_path in existing_runs:
        result = load_existing_predictions(examples, class_labels, pred_path)
        if result is None:
            continue
        yt, yp, m = result
        plot_confusion_matrix(
            yt, yp, class_labels,
            title=f"{label} — single pass",
            out_path=OUTPUT_DIR / f"confusion_matrix__{safe_filename(label)}.png",
        )
        comparison_rows.append({
            "name": label, "type": mtype,
            "accuracy": m["accuracy"], "accuracy_std": 0.0,
            "macro_f1": m["macro_f1"], "macro_f1_std": 0.0,
            "weighted_f1": m["weighted_f1"],
        })
        per_class_f1[label] = {c: m[f"f1_{c}"] for c in class_labels}
        for c in class_labels:
            per_class_table_rows.append({
                "model": label, "class": c,
                "precision": m[f"precision_{c}"], "recall": m[f"recall_{c}"], "f1": m[f"f1_{c}"],
            })
        print(f"  {label}: acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")

    # ---- DistilBERT (load) ----
    dbert = load_distilbert_predictions(class_labels)
    if dbert is None:
        print("WARN: DistilBERT predictions not yet available — skipping its panels.")
    else:
        yt, yp, m_db = dbert
        plot_confusion_matrix(
            yt, yp, class_labels,
            title="DistilBERT fine-tuned — aggregated 5-fold CV",
            out_path=OUTPUT_DIR / "confusion_matrix__distilbert.png",
        )
        db_json = json.loads(DISTILBERT_JSON.read_text())
        agg = db_json["aggregate"]
        comparison_rows.append({
            "name": "DistilBERT",
            "type": "Supervised (transformer)",
            "accuracy": agg["accuracy_mean"],
            "accuracy_std": agg["accuracy_std"],
            "macro_f1": agg["macro_f1_mean"],
            "macro_f1_std": agg["macro_f1_std"],
            "weighted_f1": agg["weighted_f1_mean"],
        })
        per_class_f1["DistilBERT"] = {c: m_db[f"f1_{c}"] for c in class_labels}
        for c in class_labels:
            per_class_table_rows.append({
                "model": "DistilBERT", "class": c,
                "precision": m_db[f"precision_{c}"],
                "recall": m_db[f"recall_{c}"],
                "f1": m_db[f"f1_{c}"],
            })
        print(f"  DistilBERT: acc={agg['accuracy_mean']:.3f}  macroF1={agg['macro_f1_mean']:.3f}")

    # ---- Combined plots ----
    plot_model_comparison_bar(comparison_rows, OUTPUT_DIR / "model_comparison_bar.png")
    plot_per_class_f1(per_class_f1, class_labels, OUTPUT_DIR / "per_class_f1_bar.png")
    plot_threshold_pareto(OUTPUT_DIR / "threshold_pareto.png")

    # ---- Tables ----
    write_comparison_table(comparison_rows, class_labels, per_class_f1, OUTPUT_DIR)
    write_per_class_table(per_class_table_rows, OUTPUT_DIR)
    write_class_distribution(examples, OUTPUT_DIR)

    print(f"\nArtifacts written under {OUTPUT_DIR}/")


def write_comparison_table(rows, class_labels, per_class_f1, out_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: -r["macro_f1"])
    lines = [
        "# Model Comparison — 5-Fold Stratified CV (N=94)\n\n",
        "| Rank | Model | Type | Accuracy | Macro F1 | Weighted F1 |"
        + "".join(f" F1 ({c}) |" for c in class_labels) + "\n",
        "| --- | --- | --- | --- | --- | --- |" + " --- |" * len(class_labels) + "\n",
    ]
    csv_lines = ["rank,model,type,accuracy,accuracy_std,macro_f1,macro_f1_std,weighted_f1,"
                 + ",".join(f"f1_{c}" for c in class_labels) + "\n"]

    for rank, r in enumerate(rows, start=1):
        acc = f"{r['accuracy']:.3f}"
        if r["accuracy_std"] > 0:
            acc += f" ± {r['accuracy_std']:.3f}"
        f1 = f"{r['macro_f1']:.3f}"
        if r["macro_f1_std"] > 0:
            f1 += f" ± {r['macro_f1_std']:.3f}"
        wf1 = f"{r['weighted_f1']:.3f}"
        per_class = "".join(f" {per_class_f1[r['name']].get(c, 0.0):.3f} |" for c in class_labels)
        lines.append(f"| {rank} | {r['name']} | {r['type']} | {acc} | {f1} | {wf1} |{per_class}\n")

        csv_lines.append(
            f"{rank},{r['name']},{r['type']},{r['accuracy']:.4f},{r['accuracy_std']:.4f},"
            f"{r['macro_f1']:.4f},{r['macro_f1_std']:.4f},{r['weighted_f1']:.4f},"
            + ",".join(f"{per_class_f1[r['name']].get(c, 0.0):.4f}" for c in class_labels)
            + "\n"
        )

    (out_dir / "model_comparison.md").write_text("".join(lines), encoding="utf-8")
    (out_dir / "model_comparison.csv").write_text("".join(csv_lines), encoding="utf-8")


def write_per_class_table(rows, out_dir: Path) -> None:
    md = ["# Per-Class Precision / Recall / F1\n\n",
          "| Model | Class | Precision | Recall | F1 |\n| --- | --- | --- | --- | --- |\n"]
    csv = ["model,class,precision,recall,f1\n"]
    for r in rows:
        md.append(f"| {r['model']} | {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |\n")
        csv.append(f"{r['model']},{r['class']},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")
    (out_dir / "per_class_metrics.md").write_text("".join(md), encoding="utf-8")
    (out_dir / "per_class_metrics.csv").write_text("".join(csv), encoding="utf-8")


def write_class_distribution(examples, out_dir: Path) -> None:
    counts = class_distribution(examples)
    md = ["# Ground-Truth Class Distribution (CV subset)\n\n",
          f"- N = {len(examples)} documents\n", "- 3 classes used in CV (HR singleton dropped, Internal/Emails have no GT)\n\n",
          "| Class | Count | Share |\n| --- | --- | --- |\n"]
    total = sum(counts.values())
    for c, n in sorted(counts.items(), key=lambda x: -x[1]):
        md.append(f"| {c} | {n} | {n/total:.1%} |\n")
    (out_dir / "class_distribution.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
