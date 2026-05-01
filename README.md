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

Recommended current run:

```bash
python src/assign_taxonomy_evidence.py \
  --input data/processed/clean_documents.jsonl \
  --chunks data/processed/chunks.jsonl \
  --taxonomy configs/taxonomy.yaml \
  --output data/outputs/taxonomy_assignments_evidence.jsonl \
  --store_debug
```
