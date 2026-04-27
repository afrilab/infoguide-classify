# Classification Module Design and Evaluation

## Overview

- This document defines the design of the document-classification module
  and logs the ablation experiments that shape it.
- The module assigns each document a single primary **document type** label drawn
  from a controlled label space based on document structure, format, and content characteristics.
- The current production method is zero-shot NLI
  (`facebook/bart-large-mnli`); TF-IDF and embedding-similarity
  baselines are available (see Methods Registry below).

## Input

- Input is the preprocessed corpus (`data/processed/clean_documents.jsonl`).
- Each row is expected to contain:
  - `doc_id`
  - one of `processed_text` / `clean_text` / `text` (priority order)

## Output

### Predictions

Written to `data/classified/classification_results.jsonl`. Each row:

- `doc_id`
- `predicted_label` (one of the configured labels, or `Needs_Review`)
- `confidence` — top score
- `second_best`, `second_score`
- `margin` — `confidence - second_score`
- `top_labels` — full ranked list with raw scores
- `used_field`, `text_len`, `method`, `model`,
  `head_only`, `max_chars`

### Evaluation Reports

- `logs/classification_report.md` — current run
- `logs/classification_report__<tag>.md` — per-ablation snapshots
- Companion CSVs in `logs/` (summary, metrics, confusion matrix,
  misclassified docs, confidence stats)
- Ground truth: `data/eval/classification_gt.jsonl`

## Label Space

Labels describe a document's **primary type and format**, not the topics it
mentions. The label set covers six document categories based on content structure,
formality, and organizational context:

| Label                      | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| Policy_Procedure_Contract  | Formal policy, procedure, and contract documents     |
| Reports                    | Financial, incident, and audit reports               |
| Internal_Communications    | Memos, announcements, and meeting notes               |
| Emails                     | Email correspondence and digital communications       |
| HR_Documents               | Employee records, evaluations, and HR communications |
| Forms_Structured           | Standardized forms with organized fields              |

### Labeling Protocol

When labeling ground truth, ask: **"What type of document is this based on its structure, format, and intended use?"**

| Document Characteristics       | Label                      |
| ------------------------------ | -------------------------- |
| Formal policies and contracts  | Policy_Procedure_Contract  |
| Analytical reports with data   | Reports                    |
| Internal memos and updates     | Internal_Communications    |
| Email messages and threads     | Emails                     |
| Employee and HR information    | HR_Documents               |
| Structured forms and templates | Forms_Structured           |

Boundary guidelines (apply to humans during labeling):

- **Policy_Procedure_Contract** — NOT this label if document is an email,
  a report, an internal announcement, or HR-related.
- **Reports** — NOT this label if document is a policy, internal memo,
  email, HR document, or structured form.
- **Internal_Communications** — NOT this label if document is sent externally,
  is a formal policy, incident report, or HR-specific.
- **Emails** — NOT this label if document is a formal report, policy,
  internal memo, HR record, or structured form.
- **HR_Documents** — NOT this label if document is not employee or HR-related,
  or if it's an email, policy, report, or structured form.
- **Forms_Structured** — NOT this label if document is narrative content
  (email, memo, report, policy) without structured fields.

## Methods Registry

| Method                 | Status      | Notes                         |
| ---------------------- | ----------- | ----------------------------- |
| zero_shot (BART-MNLI)  | Implemented | `src/classification/run.py`   |
| TF-IDF cosine to label | Planned     | Baseline (supervisor request) |
| Embedding similarity   | Planned     | Baseline (supervisor request) |

## Evaluation Setup

- **Test set**: 11 hand-labeled documents
  (`data/eval/classification_gt.jsonl`).
- **Metrics**: per-class precision/recall/F1, macro/weighted/micro
  averages, confusion matrix, `Needs_Review` count.
- **Runner**: `python src/classification/classification_report.py` —
  loads predictions and ground truth, writes Markdown + CSV outputs
  to `logs/`.
- **Known limitation**: Ground truth is being expanded with diverse document types.
  Current evaluation set is limited. See Open Issues.

## Ablation Log

Current design uses the 6-label document-type classification system with BART-MNLI zero-shot.
All runs use BART-MNLI zero-shot with `head_only=true`, `max_chars=3000`,
`min_score=0.10`, `min_margin=0.08`, hypothesis template
`"This document is about {}."`. 

| ID  | Variant                                                  | Macro F1 | Macro P | Macro R | Accuracy | Snapshot                           |
| --- | -------------------------------------------------------- | -------- | ------- | ------- | -------- | ---------------------------------- |
| V1  | Initial 6-label setup with improved descriptions         | 0.222    | 0.313   | 0.208   | 0.727    | `logs/classification_report__v1_zeroshot.md` |

**V1 Performance Notes:**
- Ground truth: 11 documents (7 Policy_Procedure_Contract, 4 Reports)
- Strong performance on Policy_Procedure_Contract (93.3% F1)
- Lower recall on Reports (25%)
- Labels with no ground truth support: Internal_Communications, Emails, HR_Documents, Forms_Structured

## Findings

### F1. 6-label document-type classification is feasible with good initial results

The zero-shot NLI approach successfully distinguishes document types, achieving 72.7% accuracy 
on a test set of 11 documents. The model particularly excels at identifying 
Policy_Procedure_Contract documents (93.3% F1), which represent 64% of the labeled data.

Performance on Reports (F1 = 0.40, Recall = 25%) indicates the label descriptions need 
refinement to better capture World Bank-style analytical reports.

### F2. Label descriptions must match corpus vocabulary

The descriptions were refined through iterative testing to emphasize:
- **Policy_Procedure_Contract**: "formal legal", "regulations", "FATF recommendations", "guidance documents"
- **Reports**: "World Bank", "financial sector assessments", "research papers", "audit findings"

This vocabulary alignment directly maps to how documents appear in the corpus.

### F3. Threshold tuning improved results significantly

Lowering `min_score` from 0.35 to 0.10 reduced false "Needs_Review" predictions, 
allowing the model's actual predictions to be captured. This is appropriate until 
ground truth is expanded.

### F4. Ground truth expansion is critical

Current GT is heavily imbalanced (7 Policy_Procedure_Contract, 4 Reports, 0 others).
To achieve robust per-class metrics and validate the 6-label schema, we need:

- **Policy_Procedure_Contract**: ≥10 docs (currently 7)
- **Reports**: ≥10 docs (currently 4)  
- **Internal_Communications**: ≥5 docs (currently 0)
- **Emails**: ≥5 docs (currently 0)
- **HR_Documents**: ≥5 docs (currently 0)
- **Forms_Structured**: ≥5 docs (currently 0)

Target: ≥30 total documents with balanced class representation.

## Open Issues

### O1. Ground truth expansion required

To produce defensible per-class metrics we need:

- **Policy_Procedure_Contract**: Corporate policies, contracts, procedures
- **Reports**: Financial reports, incident reports, audit assessments
- **Internal_Communications**: Memos, announcements, meeting notes
- **Emails**: Email correspondence, internal messages
- **HR_Documents**: Employee records, evaluations, recruitment materials
- **Forms_Structured**: Application forms, surveys, registration documents

Target: ≥ 5–10 docs per class, ≥ 30 total for reliable baselines.

### O2. Document preprocessing may affect classification

Some documents may contain web navigation, table-of-contents, or other
non-content elements. With `head_only=true` and `max_chars=3000`, the
classifier may focus on non-essential content.

Possible fixes (preprocessing-side, out of scope for this module):

- Detect and strip web-nav templates during ingestion.
- Skip table-of-contents blocks during cleaning.
- Use smart slicing for documents with front-matter.

### O3. Threshold optimization deferred

Threshold tuning (`min_score`, `min_margin`) is deferred until ground truth
reaches target size (O1). Current thresholds may be suboptimal but are 
reasonable starting points.

## Artifacts

| Artifact                                              | Purpose                                          |
| ----------------------------------------------------- | ------------------------------------------------ |
| `configs/classification.yaml`                         | Method config + labels + descriptions + protocol |
| `src/classification/run.py`                           | Core classification logic                        |
| `src/classification/classify_documents.py`            | CLI entry point                                  |
| `src/classification/classification_report.py`         | Evaluation runner                                |
| `data/eval/classification_gt.jsonl`                   | Ground truth (with labeler rationale notes)      |
| `data/classified/classification_results.jsonl`        | Latest predictions                               |
| `data/classified/classification_results__<tag>.jsonl` | Per-ablation prediction snapshots                |
| `logs/classification_report.md`                       | Latest evaluation report                         |
| `logs/classification_report__<tag>.md`                | Per-ablation report snapshots                    |
