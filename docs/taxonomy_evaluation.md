# Taxonomy Evaluation

This document describes the corrected taxonomy system. The earlier document-type labels were classification labels; the active taxonomy now uses banking-domain levels.

## Taxonomy Levels

The taxonomy is stored in `configs/taxonomy.yaml` and uses three levels:

- Level 1: Banking domain
- Level 2: Functional category
- Level 3: Specific topic

| Level 1 | Level 2 categories |
|---|---|
| `Governance & Policy` | `Internal Policies`; `Procedures & Guidelines` |
| `Risk & Compliance` | `AML / KYC`; `Audit & Monitoring`; `Incident & Fraud` |
| `Financial Operations` | `Reporting & Statements`; `Transactions & Processing` |
| `Customer & Accounts` | `Account Management`; `Customer Communication` |
| `Human Resources` | `Employee Management`; `Internal HR Communication` |
| `IT & Security` | `Access & Identity`; `Data Protection` |

Level 3 was added under each Level 2 to make assignments specific enough for evaluation. Examples include `Customer Due Diligence`, `Financial Stability Assessment`, `Regulatory Reporting`, `Account Authorization`, `Workforce Analytics`, and `Data Quality and Integrity`.

## Gold Label File

Gold labels are stored in `outputs/taxonomy/labels/taxonomy_gold_labels.jsonl`. Each row now contains a full hierarchical path:

```json
{
  "label": "Account Authorization",
  "taxonomy": {
    "level_1": "Customer & Accounts",
    "level_2": "Account Management",
    "level_3": "Account Authorization"
  }
}
```

The current label set covers all 123 processed documents.

| Level 1 | Count |
|---|---:|
| `Financial Operations` | 52 |
| `Risk & Compliance` | 41 |
| `Customer & Accounts` | 16 |
| `IT & Security` | 7 |
| `Governance & Policy` | 6 |
| `Human Resources` | 1 |

The split files were regenerated from the new hierarchy:

| Split | Documents |
|---|---:|
| Development | 86 |
| Test | 37 |

The taxonomy labels were assigned manually by a single annotator after reviewing the documents and applying the finalized three-level banking taxonomy. Each row includes provenance fields:

- `annotation_status: manual_single_annotator`
- `label_source: manual_document_review`
- `independent_second_review: false`

Because the labels were not independently double-annotated, inter-annotator agreement could not be measured. The reported scores should therefore be interpreted as accuracy against a single-annotator manual gold set, not as independently adjudicated multi-annotator performance. The deterministic rule-boosted variant was removed from the taxonomy workflow because it produced circular `100%` agreement and harmed the credibility of the evaluation.

## Evaluation Method

Evaluation now reports the hierarchy level by level:

- Level 1 agreement for the banking domain.
- Level 2 agreement for the functional category.
- Level 3 agreement for the specific topic.
- Full-path agreement for the complete `Level 1 > Level 2 > Level 3` path.

This avoids hiding cases where the model predicts the correct broad domain but misses the more specific category.

Command:

```bash
conda run -n infoguide_env python src/taxonomy/evaluate_taxonomy_accuracy.py \
  --predictions outputs/taxonomy/taxonomy_assignments_embeddings.jsonl \
  --labels outputs/taxonomy/labels/taxonomy_gold_labels.jsonl
```

## Current Results

| Method | Correct / Total | Path agreement | L1 agreement | L2 agreement | L3 macro F1 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Top-down TF-IDF + Logistic Regression | 68 / 123 | 0.5528 | 0.8211 | 0.7236 | 0.3141 | Literature baseline using local classifiers at each hierarchy level. |
| Flat TF-IDF + Linear SVM | 67 / 123 | 0.5447 | 0.8130 | 0.6829 | 0.3425 | Literature baseline using a flat max-margin classifier over Level 3 classes. |
| Top-down TF-IDF + Linear SVM | 65 / 123 | 0.5285 | 0.8130 | 0.7154 | 0.2860 | Literature baseline using local max-margin classifiers at each hierarchy level. |
| Flat TF-IDF + Logistic Regression | 64 / 123 | 0.5203 | 0.7805 | 0.6748 | 0.3239 | Literature baseline over Level 3 classes with hierarchical post-mapping. |
| Hierarchical embeddings, alpha=0.7 | 51 / 123 | 0.4146 | 0.6829 | 0.5285 | 0.3872 | Best semantic retrieval baseline by full-path agreement. |
| Chunk evidence, top-1 local n-gram | 46 / 123 | 0.3740 | 0.5610 | 0.4959 | 0.4953 | Best chunk-evidence variant by macro F1 among the top runs. |
| Keyword baseline | 29 / 123 | 0.2358 | 0.5203 | 0.4390 | 0.2540 | Simple keyword baseline. |

The literature-style top-down logistic regression classifier is the strongest full-path baseline, reaching 55.3% agreement overall and 54.1% on the held-out test split. The added Linear SVM baselines are close competitors, with the flat SVM reaching 54.5% full-path agreement and the top-down SVM reaching 52.9%. The semantic embedding method remains useful as a zero-shot/retrieval-style baseline, but it no longer outperforms supervised TF-IDF methods on this small manually labeled corpus. The gap between Level 1 and full-path performance remains important: even the strongest method reaches 82.1% Level 1 agreement but 55.3% full-path agreement, showing that fine-grained Level 2 and Level 3 topics are the hard part.

The removed rule-boosted result is not included in the reported ablation because it measured deterministic consistency rather than generalization.

## What Changed

- Replaced the old six document-type labels with banking-domain taxonomy levels.
- Added Level 3 topics under the requested Level 1 and Level 2 structure.
- Updated embedding, evidence, and keyword loaders so they read real Level 3 topics from `configs/taxonomy.yaml`.
- Added literature-style hierarchical text classification comparisons: flat and top-down TF-IDF classifiers with Logistic Regression and Linear SVM, using leave-one-out prediction.
- Updated evaluation and plotting utilities to compare Level 3 labels first.
- Updated evaluation to report Level 1, Level 2, Level 3, and full-path agreement separately.
- Removed the rule-boosted taxonomy algorithm and its generated outputs to avoid presenting circular `100%` agreement.
- Added provenance fields to the label files to document that they are manual single-annotator labels.
- Regenerated `taxonomy_gold_labels.jsonl`, `taxonomy_gold_dev.jsonl`, `taxonomy_gold_test.jsonl`, and `taxonomy_assignments_embeddings.jsonl`.

## Limitations

- Some banking domains have low support in the current corpus, especially `Human Resources`, `Governance & Policy`, and `IT & Security`.
- The corpus is dominated by public reports, regulatory guidance, tax forms, and account forms, so it does not fully cover customer communications, internal HR communication, access/identity workflows, or incident/fraud examples.
- The current setup is single-label. Some documents naturally fit multiple paths, such as account agreements that are both policy/legal documents and account-management documents.
- The label file was produced by one annotator, so no inter-annotator agreement score is available.
- A second independent annotation pass would strengthen the evaluation, but it was outside the project timeline.
