# infoguide-classify

## Taxonomy assignment

The taxonomy in `configs/taxonomy.yaml` classifies documents into six document/PII-oriented labels:

- `Policy / Procedure / Contract Documents`
- `Reports (Financial / Incident / Audit)`
- `Internal Communications`
- `Emails`
- `HR Documents / Communications`
- `Forms / Structured Documents`

Available taxonomy assigners:

- `src/assign_taxonomy_baseline.py`: keyword baseline.
- `src/assign_taxonomy_embeddings.py`: whole-document hierarchical embedding/keyword hybrid.
- `src/assign_taxonomy_evidence.py`: chunk-evidence assigner that scores complete taxonomy paths, aggregates top supporting chunks, and flags low-margin cases for review.
- `src/assign_taxonomy_rule_boosted.py`: recommended final taxonomy output; applies deterministic document-genre rules on top of the best hybrid output.

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
conda run -n infoguide_env python src/assign_taxonomy_embeddings.py \
  --input data/processed/clean_documents.jsonl \
  --taxonomy configs/taxonomy.yaml \
  --output data/outputs/taxonomy_assignments_embeddings.jsonl \
  --alpha 0.7 \
  --store_debug
```

Chunk-evidence comparison:

```bash
conda run -n infoguide_env python src/assign_taxonomy_evidence.py \
  --input data/processed/clean_documents.jsonl \
  --chunks data/processed/chunks.jsonl \
  --taxonomy configs/taxonomy.yaml \
  --output data/outputs/taxonomy_assignments_evidence.jsonl \
  --store_debug
```

Rule-boosted final output:

```bash
conda run -n infoguide_env python src/assign_taxonomy_rule_boosted.py \
  --documents data/processed/clean_documents.jsonl \
  --base_predictions data/outputs/taxonomy_assignments_embeddings.jsonl \
  --output data/outputs/taxonomy_assignments_rule_boosted.jsonl
```

### 5. Evaluate taxonomy accuracy

```bash
conda run -n infoguide_env python src/evaluate_taxonomy_accuracy.py \
  --predictions data/outputs/taxonomy_assignments_rule_boosted.jsonl \
  --gold data/labels/taxonomy_gold_labels.jsonl \
  --show_errors
```

### 6. Run ablation study and plots

```bash
conda run -n infoguide_env python src/run_taxonomy_ablation.py --skip_existing
conda run -n infoguide_env python src/plot_taxonomy_ablation.py
```

Outputs:

- `data/outputs/ablation/taxonomy_ablation_results.csv`
- `data/outputs/ablation/taxonomy_ablation_results.md`
- `data/outputs/ablation/figures/`

## Current Taxonomy Result

The recommended final taxonomy assignment file is:

- `data/outputs/taxonomy_assignments_rule_boosted.jsonl`

On the current 39-document gold set, the rule-boosted hybrid reaches `39/39` accuracy. This should be interpreted as a corpus-specific validation result, not a broad generalization claim; see `docs/taxonomy_evaluation.md` for limitations.
