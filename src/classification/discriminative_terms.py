# src/classification/discriminative_terms.py
"""Per-class discriminative TF-IDF terms.

Fits the same TF-IDF preprocessing used by `supervised_baselines.py` and the
TF-IDF cosine classifier (ngram=[1,2], english stop words, lowercase, min_df=1)
on the full labeled corpus, then ranks each class's terms by

    discriminative_score(term, class) = mean(TF-IDF | in class) - mean(TF-IDF | other classes)

A positive score means the term carries more weight in `class` than elsewhere.

Outputs:
- outputs/classification_outputs/discriminative_terms.md
- outputs/classification_outputs/discriminative_terms.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from eval_harness import REPO_ROOT, class_distribution, load_dataset

OUTPUT_DIR = REPO_ROOT / "outputs" / "classification_outputs"

TFIDF_KWARGS = dict(
    ngram_range=(1, 2),
    stop_words="english",
    lowercase=True,
    min_df=1,
)

TOP_N = 20


def compute_discriminative_terms(examples, top_n: int = TOP_N) -> dict:
    texts = [ex.text for ex in examples]
    labels = np.array([ex.label for ex in examples])
    class_labels = sorted(set(labels.tolist()))

    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    X = vectorizer.fit_transform(texts)  # (n_docs, n_terms), sparse
    terms = np.array(vectorizer.get_feature_names_out())

    per_class: dict[str, list[dict]] = {}
    for cls in class_labels:
        in_mask = labels == cls
        out_mask = ~in_mask

        # Mean TF-IDF per term, in-class and out-of-class.
        in_mean = np.asarray(X[in_mask].mean(axis=0)).ravel()
        out_mean = np.asarray(X[out_mask].mean(axis=0)).ravel()
        score = in_mean - out_mean

        # Top-N terms with positive discriminative score, descending.
        order = np.argsort(-score)
        rows = []
        for rank, i in enumerate(order[:top_n], start=1):
            if score[i] <= 0:
                break
            rows.append(
                {
                    "rank": rank,
                    "term": str(terms[i]),
                    "in_class_mean_tfidf": float(in_mean[i]),
                    "out_class_mean_tfidf": float(out_mean[i]),
                    "discriminative_score": float(score[i]),
                }
            )
        per_class[cls] = rows

    return {
        "n_examples": len(examples),
        "class_distribution": dict(Counter(labels.tolist())),
        "class_labels": class_labels,
        "vectorizer": {
            "ngram_range": list(TFIDF_KWARGS["ngram_range"]),
            "stop_words": TFIDF_KWARGS["stop_words"],
            "lowercase": TFIDF_KWARGS["lowercase"],
            "min_df": TFIDF_KWARGS["min_df"],
            "vocab_size": int(len(terms)),
        },
        "top_n": top_n,
        "per_class": per_class,
    }


def write_markdown(results: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Top Discriminative TF-IDF Terms per Class\n\n")
    lines.append(f"- N = {results['n_examples']} documents\n")
    dist = results["class_distribution"]
    lines.append(
        "- Class distribution: "
        + ", ".join(f"{k}={dist[k]}" for k in sorted(dist))
        + "\n"
    )
    v = results["vectorizer"]
    lines.append(
        f"- TF-IDF preprocessing: ngram={v['ngram_range']}, "
        f"stop_words={v['stop_words']}, lowercase={v['lowercase']}, "
        f"min_df={v['min_df']}, vocab_size={v['vocab_size']}\n"
    )
    lines.append(
        "- Score = mean(TF-IDF | in class) − mean(TF-IDF | other classes); "
        "higher means more distinctive to the class.\n\n"
    )

    for cls in results["class_labels"]:
        rows = results["per_class"].get(cls, [])
        lines.append(f"## {cls}  (N={dist.get(cls, 0)})\n\n")
        if not rows:
            lines.append("_No terms with positive discriminative score._\n\n")
            continue
        lines.append("| Rank | Term | In-class mean | Out-class mean | Score |\n")
        lines.append("| ---: | --- | ---: | ---: | ---: |\n")
        for r in rows:
            lines.append(
                f"| {r['rank']} | `{r['term']}` "
                f"| {r['in_class_mean_tfidf']:.4f} "
                f"| {r['out_class_mean_tfidf']:.4f} "
                f"| {r['discriminative_score']:.4f} |\n"
            )
        lines.append("\n")

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_dataset()
    print(f"Loaded {len(examples)} examples — {class_distribution(examples)}")

    results = compute_discriminative_terms(examples, top_n=TOP_N)

    json_path = OUTPUT_DIR / "discriminative_terms.json"
    md_path = OUTPUT_DIR / "discriminative_terms.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    write_markdown(results, md_path)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    for cls, rows in results["per_class"].items():
        head = ", ".join(r["term"] for r in rows[:8])
        print(f"  {cls}: {head}")


if __name__ == "__main__":
    main()
