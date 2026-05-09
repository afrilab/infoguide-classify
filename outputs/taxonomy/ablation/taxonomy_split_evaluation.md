# Taxonomy Split Evaluation

The split uses the corrected three-level banking taxonomy and is stratified by Level 1 banking domain.

| Method | Split | Accuracy | Macro F1 | Correct / Total | Interpretation |
|---|---|---:|---:|---:|---|
| `hierarchical_embeddings` | Development | 0.4535 | 0.3661 | 39 / 86 | Corrected Level 3 taxonomy without rule normalization. |
| `hierarchical_embeddings` | Test | 0.3243 | 0.2751 | 12 / 37 | Corrected Level 3 taxonomy without rule normalization. |
| `rule_boosted_hierarchical` | Development | 1.0000 | 1.0000 | 86 / 86 | Provisional-gold consistency check. |
| `rule_boosted_hierarchical` | Test | 1.0000 | 1.0000 | 37 / 37 | Retrospective split; not an unbiased future-generalization estimate. |

Important caveat: the rule-boosted score is expected because the provisional gold file was regenerated from the reviewed/rule-normalized hierarchy. Manual review is still needed before final reporting.
