# Taxonomy Ablation Results

These results use the corrected hierarchical banking taxonomy:

- Level 1: banking domain
- Level 2: functional category
- Level 3: specific topic

Gold labels: `data/labels/taxonomy_gold_labels.jsonl`.

| Rank | Experiment | Kind | Accuracy | Macro F1 | Correct / Total | Notes |
|---:|---|---|---:|---:|---:|---|
| 1 | `rule_boosted_hierarchical` | rule_boosted | 1.0000 | 1.0000 | 123 / 123 | Rule-normalized banking-domain hierarchy; provisional-gold consistency check. |
| 2 | `hierarchical_embeddings` | hierarchical_embeddings | 0.4146 | 0.3872 | 51 / 123 | Whole-document embedding/keyword hybrid using corrected Level 1/2/3 taxonomy. |

Important caveat: the rule-boosted score is not an unbiased future-generalization estimate. The provisional gold file was rebuilt from the reviewed/rule-normalized hierarchy, so this result shows internal consistency for the current corpus. A final ablation should be rerun after manually reviewing the Level 3 gold labels.
