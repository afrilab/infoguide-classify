# Taxonomy Ablation Results

These scores are agreement with the current taxonomy label file. Deterministic rule-normalized outputs are excluded to avoid circular 100% agreement.

| Rank | Experiment | Kind | Role | Path agree. | L1 agree. | L2 agree. | L3 agree. | L3 macro F1 | Correct/Total | Notes |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `topdown_tfidf_logreg_loo` | topdown_hierarchical_classifier | baseline agreement | 0.5528 | 0.8211 | 0.7236 | 0.5528 | 0.3141 | 68/123 | Literature baseline: top-down local TF-IDF + logistic regression classifiers; leave-one-out. |
| 2 | `flat_tfidf_svm_loo` | flat_classifier | baseline agreement | 0.5447 | 0.8130 | 0.6829 | 0.5447 | 0.3425 | 67/123 | Literature baseline: flat TF-IDF + Linear SVM over Level 3 labels; leave-one-out. |
| 3 | `topdown_tfidf_svm_loo` | topdown_hierarchical_classifier | baseline agreement | 0.5285 | 0.8130 | 0.7154 | 0.5285 | 0.2860 | 65/123 | Literature baseline: top-down local TF-IDF + Linear SVM classifiers; leave-one-out. |
| 4 | `flat_tfidf_logreg_loo` | flat_classifier | baseline agreement | 0.5203 | 0.7805 | 0.6748 | 0.5203 | 0.3239 | 64/123 | Literature baseline: flat TF-IDF + logistic regression over Level 3 labels; leave-one-out. |
| 5 | `hierarchical_alpha_0_7` | hierarchical_embeddings | baseline agreement | 0.4146 | 0.6829 | 0.5285 | 0.4146 | 0.3872 | 51/123 | Whole-document hierarchical assigner; alpha=0.7. |
| 6 | `hierarchical_alpha_1_0` | hierarchical_embeddings | baseline agreement | 0.4065 | 0.6585 | 0.5366 | 0.4065 | 0.3425 | 50/123 | Whole-document hierarchical assigner; alpha=1.0. |
| 7 | `evidence_topk_1` | chunk_evidence | baseline agreement | 0.3740 | 0.5610 | 0.4959 | 0.3740 | 0.4953 | 46/123 | Chunk-evidence assigner with local n-gram backend; top_k_chunks=1. |
| 8 | `hierarchical_alpha_0_5` | hierarchical_embeddings | baseline agreement | 0.3740 | 0.6667 | 0.5041 | 0.3740 | 0.3642 | 46/123 | Whole-document hierarchical assigner; alpha=0.5. |
| 9 | `hierarchical_alpha_0_3` | hierarchical_embeddings | baseline agreement | 0.3740 | 0.6667 | 0.4959 | 0.3740 | 0.3535 | 46/123 | Whole-document hierarchical assigner; alpha=0.3. |
| 10 | `evidence_topk_2` | chunk_evidence | baseline agreement | 0.3659 | 0.5772 | 0.4959 | 0.3659 | 0.4808 | 45/123 | Chunk-evidence assigner with local n-gram backend; top_k_chunks=2. |
| 11 | `evidence_alpha_0_68` | chunk_evidence | baseline agreement | 0.3577 | 0.5610 | 0.4878 | 0.3577 | 0.4746 | 44/123 | Chunk-evidence assigner with local n-gram backend; alpha=0.68. |
| 12 | `evidence_topk_4` | chunk_evidence | baseline agreement | 0.3577 | 0.5610 | 0.4878 | 0.3577 | 0.4746 | 44/123 | Chunk-evidence assigner with local n-gram backend; top_k_chunks=4. |
| 13 | `evidence_alpha_0_50` | chunk_evidence | baseline agreement | 0.3496 | 0.6098 | 0.5285 | 0.3496 | 0.4958 | 43/123 | Chunk-evidence assigner with local n-gram backend; alpha=0.5. |
| 14 | `evidence_topk_8` | chunk_evidence | baseline agreement | 0.3496 | 0.5366 | 0.4715 | 0.3496 | 0.4675 | 43/123 | Chunk-evidence assigner with local n-gram backend; top_k_chunks=8. |
| 15 | `evidence_alpha_0_30` | chunk_evidence | baseline agreement | 0.3415 | 0.5772 | 0.5041 | 0.3415 | 0.4714 | 42/123 | Chunk-evidence assigner with local n-gram backend; alpha=0.3. |
| 16 | `evidence_alpha_0_00` | chunk_evidence | baseline agreement | 0.3252 | 0.5854 | 0.4959 | 0.3252 | 0.3985 | 40/123 | Chunk-evidence assigner with local n-gram backend; alpha=0.0. |
| 17 | `hierarchical_alpha_0_0` | hierarchical_embeddings | baseline agreement | 0.2520 | 0.5772 | 0.3821 | 0.2520 | 0.2885 | 31/123 | Whole-document hierarchical assigner; alpha=0.0. |
| 18 | `baseline_keyword` | baseline | baseline agreement | 0.2358 | 0.5203 | 0.4390 | 0.2358 | 0.2540 | 29/123 | Rule-based keyword baseline. |
| 19 | `evidence_alpha_0_85` | chunk_evidence | baseline agreement | 0.0650 | 0.0894 | 0.0894 | 0.0650 | 0.1447 | 8/123 | Chunk-evidence assigner with local n-gram backend; alpha=0.85. |
| 20 | `evidence_alpha_1_00` | chunk_evidence | baseline agreement | 0.0325 | 0.0325 | 0.0325 | 0.0325 | 0.0884 | 4/123 | Chunk-evidence assigner with local n-gram backend; alpha=1.0. |

Manual single-annotator gold label file: `data/labels/taxonomy_gold_labels.jsonl`.