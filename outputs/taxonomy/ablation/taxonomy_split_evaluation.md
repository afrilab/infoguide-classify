# Taxonomy Split Evaluation

The split uses the corrected three-level banking taxonomy and is stratified by Level 1 banking domain. Scores are measured against the current manual single-annotator gold label file.

| Method | Split | Path agree. | L1 agree. | L2 agree. | L3 macro F1 | Correct / Total | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| `topdown_tfidf_logreg_loo` | Development | 0.5581 | 0.8605 | 0.7674 | 0.2951 | 48 / 86 | Literature baseline: top-down local classifiers. |
| `topdown_tfidf_logreg_loo` | Test | 0.5405 | 0.7297 | 0.6216 | 0.4427 | 20 / 37 | Literature baseline: top-down local classifiers. |
| `flat_tfidf_logreg_loo` | Development | 0.5233 | 0.8256 | 0.7093 | 0.2946 | 45 / 86 | Literature baseline: flat Level 3 classifier. |
| `flat_tfidf_logreg_loo` | Test | 0.5135 | 0.6757 | 0.5946 | 0.5021 | 19 / 37 | Literature baseline: flat Level 3 classifier. |
| `hierarchical_embeddings` | Development | 0.4535 | 0.7209 | 0.5465 | 0.3661 | 39 / 86 | Non-rule baseline on manual single-annotator labels. |
| `hierarchical_embeddings` | Test | 0.3243 | 0.5946 | 0.4865 | 0.2751 | 12 / 37 | Non-rule baseline on manual single-annotator labels. |

Important caveat: labels were assigned by one annotator, so inter-annotator agreement is not available. The deterministic rule-boosted variant was removed because it created circular `100%` agreement.
