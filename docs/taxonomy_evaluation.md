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

## Gold Labels

Gold labels are stored in `data/labels/taxonomy_gold_labels.jsonl`. Each row now contains a full hierarchical path:

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

The current gold set covers all 123 processed documents.

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

Important caveat: these labels are provisional single-annotator labels generated from the corrected hierarchy and rule-normalized assignments. They should be reviewed before making a final accuracy claim.

## Evaluation Method

Evaluation now compares the most specific available taxonomy label, preferring `taxonomy.level_3`, then `level_2`, then `level_1`.

Command:

```bash
conda run -n infoguide_env python src/evaluate_taxonomy_accuracy.py \
  --predictions data/outputs/taxonomy_assignments_rule_boosted.jsonl \
  --gold data/labels/taxonomy_gold_labels.jsonl
```

## Current Results

| Method | Correct / Total | Accuracy | Notes |
|---|---:|---:|---|
| Hierarchical embeddings | 51 / 123 | 0.4146 | Uses the corrected Level 1/2/3 taxonomy without rule normalization. |
| Rule-boosted hierarchical | 123 / 123 | 1.0000 | Matches the provisional gold file because the gold file was rebuilt from the reviewed/rule-normalized hierarchy. |

The 100% score should not be presented as unbiased future performance. It means the current corpus is internally consistent with the new taxonomy rules. A stronger evaluation would use separately reviewed labels that were not generated from the same rule layer.

## What Changed

- Replaced the old six document-type labels with banking-domain taxonomy levels.
- Added Level 3 topics under the requested Level 1 and Level 2 structure.
- Updated embedding, evidence, and keyword loaders so they read real Level 3 topics from `configs/taxonomy.yaml`.
- Rebuilt `src/assign_taxonomy_rule_boosted.py` so rule boosts assign banking taxonomy paths instead of document-type classes.
- Updated evaluation and plotting utilities to compare Level 3 labels first.
- Regenerated `taxonomy_gold_labels.jsonl`, `taxonomy_gold_dev.jsonl`, `taxonomy_gold_test.jsonl`, `taxonomy_assignments_embeddings.jsonl`, and `taxonomy_assignments_rule_boosted.jsonl`.

## Limitations

- Some banking domains have low support in the current corpus, especially `Human Resources`, `Governance & Policy`, and `IT & Security`.
- The corpus is dominated by public reports, regulatory guidance, tax forms, and account forms, so it does not fully cover customer communications, internal HR communication, access/identity workflows, or incident/fraud examples.
- The current setup is single-label. Some documents naturally fit multiple paths, such as account agreements that are both policy/legal documents and account-management documents.
- The provisional gold labels should be manually reviewed before final reporting.
