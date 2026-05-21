# src/classification/render_methods_table.py
"""Render a slide-ready PNG of the methods overview table, styled to match
the screenshot referenced in the slide deck (dark card, monospace gridless).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from eval_harness import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "outputs" / "classification_results"

ROWS = [
    ("TF-IDF Cosine",         "Matches document text with label descriptions using keywords",    "0.916"),
    ("BGE Large Embedding",   "Semantic similarity using BGE large embeddings",                  "0.905"),
    ("Logistic Regression",   "Learns class weights over TF-IDF features from labeled examples", "0.902 ± 0.07"),
    ("Random Forest",         "Ensembles 300 decision trees over TF-IDF features",               "0.899 ± 0.08"),
    ("BGE Small Embedding",   "Semantic similarity using BGE small embeddings",                  "0.889"),
    ("Linear SVM",            "Finds maximum-margin separator in TF-IDF feature space",          "0.880 ± 0.09"),
    ("Multinomial NB",        "Bayesian model assuming feature independence",                    "0.787 ± 0.02"),
    ("DistilBERT Fine-tuned", "Fine-tunes a transformer end-to-end on labeled documents",        "0.771 ± 0.11"),
    ("BART Zero-Shot NLI",    "Uses BART entailment to classify without training",               "0.449"),
    ("DeBERTa Zero-Shot NLI", "Uses DeBERTa entailment to classify without training",            "0.333"),
]
HEADERS = ["Approach", "What It Does", "Macro F1"]

BG = "#0b1220"
ROW_BG = "#111827"
ROW_ALT = "#0f172a"
HEADER_BG = "#1f2937"
TEXT = "#e5e7eb"
TEXT_DIM = "#9ca3af"
BORDER = "#1f2937"


def render(out_path: Path, include_metric: bool = True) -> None:
    cols = HEADERS if include_metric else HEADERS[:2]
    rows = [[r[0], r[1]] + ([r[2]] if include_metric else []) for r in ROWS]

    col_widths = [0.22, 0.55, 0.18] if include_metric else [0.28, 0.72]
    n_rows = len(rows) + 1  # +1 for header
    fig_w = 13 if include_metric else 11
    fig_h = 0.55 * n_rows + 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    # x-edges from col widths (normalized)
    edges = [0.0]
    for w in col_widths:
        edges.append(edges[-1] + w)

    def draw_row(y_top, cells, fill, weight="normal", color=TEXT, size=12):
        ax.add_patch(plt.Rectangle((0, y_top - 1), 1, 1, facecolor=fill,
                                   edgecolor=BORDER, linewidth=1.2, zorder=1))
        for i, txt in enumerate(cells):
            x = edges[i] + 0.012
            ax.text(x, y_top - 0.5, str(txt),
                    ha="left", va="center",
                    fontsize=size, color=color, weight=weight,
                    family="DejaVu Sans", zorder=2)

    y = n_rows
    draw_row(y, cols, HEADER_BG, weight="bold", color="#ffffff", size=13)
    y -= 1
    for i, r in enumerate(rows):
        fill = ROW_BG if i % 2 == 0 else ROW_ALT
        draw_row(y, r, fill, size=12)
        y -= 1

    # caption
    if include_metric:
        ax.text(0.012, -0.15, "Macro F1 — supervised methods on 5-fold CV (mean ± std); unsupervised / zero-shot on the same 94-doc subset, single pass",
                ha="left", va="top", fontsize=9.5, color=TEXT_DIM,
                style="italic", family="DejaVu Sans", transform=ax.transData)

    fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render(OUTPUT_DIR / "methods_overview.png", include_metric=True)
    render(OUTPUT_DIR / "methods_overview_2col.png", include_metric=False)
    print(f"Wrote {OUTPUT_DIR}/methods_overview.png")
    print(f"Wrote {OUTPUT_DIR}/methods_overview_2col.png")


if __name__ == "__main__":
    main()
