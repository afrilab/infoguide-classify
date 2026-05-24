# infoguide-classify

## Taxonomy assignment

The taxonomy in `configs/taxonomy.yaml` assigns documents to a three-level banking taxonomy:

- Level 1: banking domain, such as `Governance & Policy`, `Risk & Compliance`, or `Financial Operations`
- Level 2: functional category, such as `AML / KYC`, `Audit & Monitoring`, or `Reporting & Statements`
- Level 3: specific topic, such as `Customer Due Diligence`, `Financial Stability Assessment`, or `Regulatory Reporting`

Available taxonomy assigners:

- `src/taxonomy/assign_taxonomy_baseline.py`: keyword baseline.
- `src/taxonomy/assign_taxonomy_embeddings.py`: whole-document hierarchical embedding/keyword hybrid.
- `src/taxonomy/assign_taxonomy_evidence.py`: chunk-evidence assigner that scores complete taxonomy paths, aggregates top supporting chunks, and flags low-margin cases for review.
- `src/taxonomy/assign_taxonomy_literature_baselines.py`: literature-style TF-IDF baselines with Logistic Regression or Linear SVM, including flat Level 3 and top-down hierarchical classifiers.

## Reproducible Pipeline

Run commands from the repository root with `infoguide_env`.

### 1. Ingest raw documents

```bash
conda run -n infoguide_env python src/document_ingestion.py
```

Reads `data/raw` using `configs/document_ingestion.yaml` and writes:

- `data/extracted/extracted_documents.jsonl`

### 2. Preprocess documents

```bash
conda run -n infoguide_env python src/process_documents.py \
  --config configs/process.yaml
```

Writes:

- `data/processed/clean_documents.jsonl`

### 3. Chunk documents

```bash
conda run -n infoguide_env python src/chunk_documents.py \
  --config configs/chunking.yaml
```

Writes:

- `data/processed/chunks.jsonl`

### 4. Run taxonomy assigners

Best learned baseline:

```bash
conda run -n infoguide_env python src/taxonomy/assign_taxonomy_embeddings.py \
  --input data/processed/clean_documents.jsonl \
  --taxonomy configs/taxonomy.yaml \
  --output data/outputs/taxonomy_assignments_embeddings.jsonl \
  --alpha 0.7 \
  --store_debug
```

Chunk-evidence comparison:

```bash
conda run -n infoguide_env python src/taxonomy/assign_taxonomy_evidence.py \
  --input data/processed/clean_documents.jsonl \
  --chunks data/processed/chunks.jsonl \
  --taxonomy configs/taxonomy.yaml \
  --output data/outputs/taxonomy_assignments_evidence.jsonl \
  --store_debug
```

Literature-style comparison baselines:

```bash
conda run -n infoguide_env python src/taxonomy/assign_taxonomy_literature_baselines.py \
  --input data/processed/clean_documents.jsonl \
  --labels data/labels/taxonomy_gold_labels.jsonl \
  --output data/outputs/ablation/topdown_tfidf_logreg_loo.jsonl \
  --method topdown_tfidf_logreg \
  --train_mode leave_one_out
```

### 5. Evaluate taxonomy agreement

```bash
conda run -n infoguide_env python src/taxonomy/evaluate_taxonomy_accuracy.py \
  --predictions data/outputs/taxonomy_assignments_embeddings.jsonl \
  --labels data/labels/taxonomy_gold_labels.jsonl \
  --show_errors
```

The single-annotator gold label file can be regenerated after document review:

```bash
conda run -n infoguide_env python src/taxonomy/build_taxonomy_gold_labels.py \
  --labels data/labels/taxonomy_gold_labels.jsonl
```

### 6. Run ablation study and plots

```bash
conda run -n infoguide_env python src/taxonomy/run_taxonomy_ablation.py --skip_existing
conda run -n infoguide_env python src/taxonomy/plot_taxonomy_ablation.py
```

Outputs:

- `data/outputs/ablation/taxonomy_ablation_results.csv`
- `data/outputs/ablation/taxonomy_ablation_results.md`
- `data/outputs/ablation/figures/`

## Current Taxonomy Result

The recommended unsupervised/semantic taxonomy assignment file is:

- `data/outputs/taxonomy_assignments_embeddings.jsonl`

On the current 123-document manual single-annotator gold label set, the strongest literature-style baseline is top-down TF-IDF + logistic regression with leave-one-out prediction: `68/123` full-path agreement and `101/123` Level 1 agreement. The flat TF-IDF + Linear SVM baseline is close behind at `67/123` full-path agreement. The best semantic embedding baseline reaches `51/123` full-path agreement and `84/123` Level 1 agreement. The deterministic rule-boosted variant was removed from the project outputs because its retrospective `123/123` agreement created a circular-evaluation credibility risk.
