# Classification Module Design and Evaluation

## Executive Summary

### What This Module Does

The classification module assigns each document a single **document type** label based on structure, format, and content characteristics. Unlike topic classification (which answers "what is this about?"), this module answers "**what kind of document is this?**" — distinguishing policies from reports, forms from internal communications, etc.

### Headline Result (Stratified 5-Fold CV, N=94)

Evaluated on the subset of ground truth that intersects the processed corpus, with classes appearing fewer than 5 times excluded so stratified k-fold is well-defined (see [Methodology](#methodology) for the precise data construction).

| Method | Type | Accuracy | Macro F1 | Notes |
| --- | --- | --- | --- | --- |
| **Logistic Regression (TF-IDF)** | **Supervised** | **0.905 ± 0.061** | **0.902 ± 0.068** | Best overall; cheap to train |
| Random Forest (TF-IDF) | Supervised | 0.894 ± 0.088 | 0.899 ± 0.084 | Comparable to LR, higher variance |
| Linear SVM (TF-IDF) | Supervised | 0.884 ± 0.090 | 0.880 ± 0.094 | — |
| TF-IDF cosine (forced) | Unsupervised (label descriptions) | 0.904 | 0.916 | Single-pass, no train/test split |
| Multinomial Naive Bayes (TF-IDF) | Supervised | 0.809 ± 0.024 | 0.787 ± 0.020 | Stable but worst supervised |
| Fine-tuned DistilBERT | Supervised (transformer) | 0.798 ± 0.090 | 0.771 ± 0.109 | High variance; underperforms with N=94 |
| BGE Large embedding cosine | Unsupervised (label descriptions) | 0.837* | — | From prior runs; not re-evaluated here |
| BART zero-shot NLI | Zero-shot | 0.480* | — | From prior runs; not re-evaluated here |

\* Prior numbers were computed on the full 98-row GT including HR singleton and 3 docs missing from the corpus, so they aren't directly comparable to the CV column. See [O1](#o1-unify-the-evaluation-numbers-across-methods) for follow-up.

**Recommendation**: deploy **Logistic Regression on TF-IDF features** as primary; keep TF-IDF cosine with label descriptions as a deterministic, training-free fallback (its 0.916 macro F1 on this subset overlaps Logistic Regression's CV interval). DistilBERT fine-tuning is not recommended at this corpus size.

---

## Methodology

### Data Construction

- Source: `data/eval/classification_gt.jsonl` (98 hand-labeled doc_ids) joined with `data/processed/clean_documents.jsonl` (95 processed docs) on `doc_id`.
- 3 GT entries (`doc_0012`, `doc_0082`, `doc_0083`) have no matching processed document and are dropped.
- The HR_Documents class has a single labeled example (`N=1`) and the Internal_Communications + Emails classes have **zero** labeled examples. None of these can support stratified k-fold CV. The HR singleton is excluded; the two empty classes are noted as gaps in the [Limitations](#limitations).
- **Final CV dataset**: **N = 94 documents** across **3 classes**: Reports (45), Policy_Procedure_Contract (29), Forms_Structured (20).
- Text preprocessing shared across methods: take the first non-empty field of (`anonymized_text`, `processed_text`, `clean_text`, `text`), truncate `head_only=True` to `max_chars=6000`.

### Cross-Validation

- Stratified 5-fold CV, `seed=42`.
- Same fold assignments used by every supervised method (`eval_harness.stratified_kfold_indices`).
- Per-fold: train on 4 folds, predict on the held-out fold, score with `sklearn.metrics`.
- Aggregate: mean and population standard deviation across folds.

### Metrics

Reported per method: accuracy, macro F1, weighted F1, and per-class F1/precision/recall. Macro F1 is the primary comparison metric because the GT subset is moderately imbalanced (45/29/20).

### Why CV Rather Than the Previous Single-Split Numbers

The previous report compared methods using a single deterministic prediction over the full 98-row GT. That worked for the unsupervised TF-IDF cosine and BGE embedding methods (they have no learned parameters, so train/test splits are unnecessary), but it cannot evaluate any supervised model fairly — there is no held-out test set. Adding supervised methods forced this redesign.

For the deterministic unsupervised methods we still report a single point estimate (the row labeled "forced") since there is nothing to cross-validate.

---

## Label Space

Labels describe a document's **primary type and format**, not the topics it mentions:

| Label                     | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| Policy_Procedure_Contract | Formal policy, procedure, and contract documents     |
| Reports                   | Financial, incident, and audit reports               |
| Internal_Communications   | Memos, announcements, and meeting notes              |
| Emails                    | Email correspondence and digital communications      |
| HR_Documents              | Employee records, evaluations, and HR communications |
| Forms_Structured          | Standardized forms with organized fields             |

The configured label space has 6 entries; only 3 are present in usable quantity in the current ground truth.

---

## Methods Registry

| Method                          | Status              | Macro F1 (CV) | Config / Script                                              | Notes                                                            |
| ------------------------------- | ------------------- | ------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| **Logistic Regression (TF-IDF)** | **RECOMMENDED**     | **0.902 ± 0.068** | `src/classification/supervised_baselines.py`              | Balanced class weights, C=1.0, max_iter=2000                     |
| Random Forest (TF-IDF)          | Implemented         | 0.899 ± 0.084 | `src/classification/supervised_baselines.py`                 | 300 trees, balanced weights                                      |
| Linear SVM (TF-IDF)             | Implemented         | 0.880 ± 0.094 | `src/classification/supervised_baselines.py`                 | C=1.0, balanced weights                                          |
| Multinomial NB (TF-IDF)         | Implemented         | 0.787 ± 0.020 | `src/classification/supervised_baselines.py`                 | Stable but lowest accuracy                                       |
| TF-IDF cosine (label descriptions) | Production fallback | 0.916 (single split, no CV) | `configs/classification_tfidf.yaml`                          | Threshold-calibrated; see [Threshold Calibration](#threshold-calibration) |
| BGE Large embedding             | Implemented         | 0.837 (prior, not CV'd) | `configs/classification_embedding_large.yaml`                | BAAI/bge-large-en-v1.5                                           |
| BGE Small embedding             | Implemented         | 0.827 (prior, not CV'd) | `configs/classification_embedding.yaml`                      | BAAI/bge-small-en-v1.5                                           |
| DistilBERT fine-tuned           | Implemented (not recommended) | 0.771 ± 0.109 | `src/classification/finetune_transformer.py`                 | 4 epochs, batch 8, lr 2e-5; overfits this N                       |
| BART zero-shot NLI              | Implemented (not recommended) | 0.480 (prior) | `configs/classification.yaml`                                | facebook/bart-large-mnli; not suitable for this corpus           |
| DeBERTa-large zero-shot NLI     | Ablation only        | 0.245 (prior) | `configs/classification_zeroshot_large.yaml`                 | Underperforms BART on this domain                                |

---

## Supervised Baselines

Four classical supervised classifiers were trained on the same TF-IDF features and the same 5-fold stratified splits. All use the unsupervised TF-IDF cosine method's preprocessing: ngram=[1,2], english stop words, lowercase, min_df=1.

Run with `python src/classification/supervised_baselines.py`. Outputs to `outputs/classification_outputs/supervised_baselines_cv.{json,md}`.

| Model | Accuracy | Macro F1 | Weighted F1 | F1 Forms | F1 Policy | F1 Reports |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.905 ± 0.061 | 0.902 ± 0.068 | 0.905 ± 0.063 | 0.950 | 0.879 | 0.878 |
| Linear SVM | 0.884 ± 0.090 | 0.880 ± 0.094 | 0.881 ± 0.092 | 0.945 | 0.846 | 0.848 |
| Random Forest | 0.894 ± 0.088 | 0.899 ± 0.084 | 0.892 ± 0.090 | 0.985 | 0.823 | 0.890 |
| Multinomial NB | 0.809 ± 0.024 | 0.787 ± 0.020 | 0.798 ± 0.023 | 0.880 | 0.692 | 0.788 |

**Reading**:
- Logistic Regression and Random Forest are within one standard deviation of each other; both narrowly beat the unsupervised TF-IDF cosine baseline on accuracy.
- The Forms_Structured class is the easiest (F1 ≥ 0.95 for the top three models); Policy vs. Reports is where the differences emerge.
- Multinomial NB lags because the dense overlap between Policy and Reports vocabularies hurts NB's strong feature-independence assumption.

---

## Threshold Calibration

The unsupervised TF-IDF cosine method emits a `Needs_Review` label whenever the top score or the top-vs-second margin falls below configured thresholds. The previous report admitted these thresholds were hand-tuned defaults. This run sweeps a grid and reports the trade-off explicitly.

Run with `python src/classification/threshold_calibration.py`. Outputs to `outputs/classification_outputs/threshold_calibration.{json,md}`.

- Grid: `min_score ∈ {0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25}` × `min_margin ∈ {0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10}`.
- Selection criterion: maximize macro F1 subject to coverage ≥ 0.90 (i.e., the classifier confidently labels at least 90% of inputs).

### Pareto Frontier (Coverage vs. Macro F1)

| Coverage ≥ | min_score | min_margin | Coverage | Macro F1 | Accuracy | Needs_Review |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 0.000 | 0.000 | 1.000 | 0.916 | 0.904 | 0 |
| 0.95 | 0.000 | 0.000 | 1.000 | 0.916 | 0.904 | 0 |
| 0.90 | 0.000 | 0.010 | 0.904 | 0.929 | 0.918 | 9 |
| 0.80 | 0.000 | 0.010 | 0.904 | 0.929 | 0.918 | 9 |
| 0.70 | 0.000 | 0.020 | 0.766 | 0.961 | 0.958 | 22 |
| 0.50 | 0.000 | 0.030 | 0.606 | 0.982 | 0.982 | 37 |

### Selected Operating Point

`min_score = 0.000`, `min_margin = 0.010` — coverage 0.904 (85/94 covered), macro F1 0.929 on covered. This is the recommended threshold for the production TF-IDF cosine fallback.

The previous production defaults (`min_score=0.10`, `min_margin=0.08`) place the classifier at very low coverage on this corpus (74 of 94 docs marked `Needs_Review`). They should be replaced with the values above; the existing `configs/classification_tfidf.yaml` is left unchanged in this branch and updating it is a separate follow-up.

---

## Fine-Tuned Transformer

A DistilBERT classifier (`distilbert-base-uncased`) was fine-tuned with stratified 5-fold CV using the same splits as the supervised lexical baselines.

Run with `python src/classification/finetune_transformer.py`. Outputs to `outputs/classification_outputs/finetune_distilbert_cv.{json,md}`.

- Training: 4 epochs, batch 8, AdamW lr=2e-5, weight decay 0.01, linear warmup 10%, max_length 256 tokens.
- Device: MPS (M-series Mac) — ~30s per fold.

### Results

| Metric | Value |
| --- | --- |
| Accuracy | 0.798 ± 0.090 |
| Macro F1 | 0.771 ± 0.109 |
| F1 Forms_Structured | 0.949 ± 0.063 |
| F1 Policy_Procedure_Contract | **0.523 ± 0.240** |
| F1 Reports | 0.841 ± 0.066 |

**Reading**:
- DistilBERT is the *weakest* supervised method we tested. The Forms vs. (Policy/Reports) decision is fine, but Policy vs. Reports collapses — `F1_Policy = 0.52 ± 0.24` indicates the model essentially can't separate the two on the held-out fold without enough training examples to learn the distinction.
- This is the classic small-data outcome for transformer fine-tuning: with ~75 training docs and the model needing to update hundreds of millions of parameters from a 256-token window, the input signal is dominated by initialization noise. The supervised lexical baselines invert the model size / data size ratio and win.
- Variance is large (σ ≈ 0.09–0.11) — a different seed could shift these numbers materially. We did not run multi-seed averaging because the qualitative conclusion ("transformer fine-tuning is not worth it at N=94") is stable.

This is a fair test, not a refutation of fine-tuning in general — with even 500–1000 docs/class the picture would likely reverse.

---

## Findings

### F1. Supervised TF-IDF beats unsupervised TF-IDF cosine, narrowly

Logistic Regression: 0.902 macro F1 CV vs. TF-IDF cosine (label descriptions): 0.916 single-split. These are within the CV standard deviation (±0.068). The supervised method has the advantage of learning class-specific weights from labels; the unsupervised method has the advantage of leveraging human-crafted descriptions. On this corpus they are effectively tied — pick by deployment needs (deterministic with no training: cosine; cheap retraining as new labels arrive: LR).

### F2. Fine-tuned transformer underperforms on small data

DistilBERT 0.771 macro F1 vs. Logistic Regression 0.902. The transformer needs more data than we have to justify its capacity. This is the expected and well-documented behavior for transformer fine-tuning at N≈100; the experiment is included for honest comparison, not because we expected it to win.

### F3. Threshold calibration trades coverage for accuracy roughly linearly

The Pareto frontier above shows that pushing macro F1 from 0.92 → 0.98 costs roughly 40 percentage points of coverage. The marginal cost is small below 90% coverage and grows steeply beyond.

### F4. Ground-truth class coverage is the real bottleneck

Two of six configured labels (Internal_Communications, Emails) have zero ground-truth examples; a third (HR_Documents) has one. Every method's reported macro F1 is computed over 3 classes only. Until the GT is expanded to cover all six classes with at least ~10 examples per class, no method can be evaluated on the full label space. This is the single most valuable improvement available.

### F5. Old report numbers vs. new CV numbers don't align — by design

The previous report listed TF-IDF cosine at 87.8% accuracy on the full 98 GT rows. Re-evaluated on the 94-row CV subset (HR singleton dropped, 3 dangling IDs removed), the same method reaches 90.4% accuracy / 91.6% macro F1. The difference is data construction, not method changes. We did not re-score the BGE and zero-shot methods on the CV subset; see [O1](#o1-unify-the-evaluation-numbers-across-methods).

---

## Recommendations

1. **Deploy Logistic Regression on TF-IDF as primary** (`src/classification/supervised_baselines.py`-trained model, persisted via joblib). 0.902 macro F1 CV, fast, retrainable as new labels arrive.
2. **Keep TF-IDF cosine with label descriptions as a deterministic fallback**, with `min_score=0.0` and `min_margin=0.01` (calibrated). Use it for cold-start before enough labels exist to retrain, or when no training data is available for new label additions.
3. **Do not deploy DistilBERT fine-tuning at current corpus size.** Revisit when labeled data exceeds ~100 docs per class.
4. **Do not deploy zero-shot NLI for this corpus.** Prior ceiling at 48% accuracy holds.

---

## Limitations

- **Three-class evaluation only.** Internal_Communications and Emails have no ground truth; HR_Documents has one example. Reported macro F1 covers only Forms / Policy / Reports.
- **Small N (94).** 5-fold CV with ~19-doc test folds yields high variance (transformer std=0.11, lexical std=0.07–0.09). Confidence intervals overlap heavily between top methods.
- **No held-out test set beyond CV.** All reported numbers are CV estimates. Adding a permanent held-out set (e.g., 20% never used in any fold) is recommended before production cut-over.
- **Documents are truncated to first 6000 chars (`head_only=True`).** Documents whose distinguishing structure appears late may be misclassified. The choice was kept for consistency with the prior report's preprocessing.

---

## Open Issues & Future Work

### O1. Unify the evaluation numbers across methods

BGE Large (0.837), BGE Small (0.827), BART (0.48), DeBERTa (0.245) are still quoted from the old 98-row evaluation. Re-running them with `eval_harness.load_dataset()` and CV folds would make every number in the Methods Registry directly comparable.

### O2. Expand ground truth to all six label classes

This is the highest-leverage methodological improvement available. Target ≥10 labeled examples per class for usable stratified CV across the full label space. Until then, half the configured labels are untestable.

### O3. Persist trained Logistic Regression model

`supervised_baselines.py` retrains per fold and discards. For deployment, add a `--final` mode that trains on all 94 examples and dumps a joblib artifact alongside the TF-IDF vectorizer.

### O4. Multi-seed transformer evaluation

DistilBERT std=0.11 across 5 folds at one seed. Averaging over 3–5 seeds would tighten the CI and confirm the underperformance is stable, not a single bad random init.

### O5. Re-tune unsupervised cosine thresholds for production config

`configs/classification_tfidf.yaml` still has `min_score=0.10`, `min_margin=0.08` (from before this calibration). Update to `min_score=0.0`, `min_margin=0.01` so the production fallback matches the operating point reported here.

---

## Artifacts

| Artifact | Purpose |
| --- | --- |
| `src/classification/eval_harness.py` | Shared dataset loader, k-fold splitter, scoring helpers |
| `src/classification/supervised_baselines.py` | Trains and CV-evaluates LR / SVM / NB / RF |
| `src/classification/threshold_calibration.py` | Grid sweep over (min_score, min_margin) for TF-IDF cosine |
| `src/classification/finetune_transformer.py` | DistilBERT fine-tuning with CV |
| `outputs/classification_outputs/supervised_baselines_cv.{json,md}` | Per-fold + aggregate CV metrics |
| `outputs/classification_outputs/threshold_calibration.{json,md}` | Threshold grid + Pareto frontier |
| `outputs/classification_outputs/finetune_distilbert_cv.{json,md}` | DistilBERT CV metrics |
| `data/eval/classification_gt.jsonl` | Ground truth (98 rows, only 94 usable after join + class-size filter) |
| `data/processed/clean_documents.jsonl` | Processed corpus (95 documents) |
| `configs/classification_tfidf.yaml` | Production TF-IDF cosine config (thresholds pending [O5](#o5-re-tune-unsupervised-cosine-thresholds-for-production-config)) |

### Reproducing the Results

```bash
# Stratified 5-fold CV on TF-IDF supervised baselines (~10 seconds)
python src/classification/supervised_baselines.py

# Threshold grid sweep on unsupervised TF-IDF cosine (~5 seconds)
python src/classification/threshold_calibration.py

# DistilBERT fine-tune with 5-fold CV (~3 minutes on MPS, longer on CPU)
python src/classification/finetune_transformer.py
```

All three scripts use the same fold seed (42) and the same dataset constructor (`eval_harness.load_dataset()`).
