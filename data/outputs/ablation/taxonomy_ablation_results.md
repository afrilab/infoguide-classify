# Taxonomy Ablation Results

| Rank | Experiment | Kind | Accuracy | Macro F1 | Correct/Total | Assigned | Review | Unassigned | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `rule_boosted_hybrid` | rule_boosted | 1.0000 | 1.0000 | 39/39 | 39 | 0 | 0 | Best hybrid output with deterministic document-genre rule boosts. |
| 2 | `hierarchical_alpha_0_7` | hierarchical_embeddings | 0.5128 | 0.5527 | 20/39 | 38 | 0 | 1 | Whole-document hierarchical assigner; alpha=0.7. |
| 3 | `evidence_alpha_0_85` | chunk_evidence | 0.4615 | 0.5092 | 18/39 | 33 | 6 | 0 | Chunk-evidence assigner; alpha=0.85. |
| 4 | `hierarchical_alpha_0_3` | hierarchical_embeddings | 0.4615 | 0.5051 | 18/39 | 38 | 0 | 1 | Whole-document hierarchical assigner; alpha=0.3. |
| 5 | `evidence_topk_8` | chunk_evidence | 0.4615 | 0.4971 | 18/39 | 36 | 3 | 0 | Chunk-evidence assigner; top_k_chunks=8. |
| 6 | `hierarchical_alpha_0_5` | hierarchical_embeddings | 0.4615 | 0.4687 | 18/39 | 38 | 0 | 1 | Whole-document hierarchical assigner; alpha=0.5. |
| 7 | `evidence_alpha_0_68` | chunk_evidence | 0.4359 | 0.4669 | 17/39 | 36 | 3 | 0 | Chunk-evidence assigner; alpha=0.68. |
| 8 | `evidence_topk_2` | chunk_evidence | 0.4359 | 0.4669 | 17/39 | 34 | 5 | 0 | Chunk-evidence assigner; top_k_chunks=2. |
| 9 | `evidence_topk_4` | chunk_evidence | 0.4359 | 0.4669 | 17/39 | 36 | 3 | 0 | Chunk-evidence assigner; top_k_chunks=4. |
| 10 | `evidence_alpha_1_00` | chunk_evidence | 0.4359 | 0.4066 | 17/39 | 31 | 8 | 0 | Chunk-evidence assigner; alpha=1.0. |
| 11 | `evidence_alpha_0_00` | chunk_evidence | 0.4103 | 0.4617 | 16/39 | 36 | 2 | 1 | Chunk-evidence assigner; alpha=0.0. |
| 12 | `evidence_alpha_0_30` | chunk_evidence | 0.4103 | 0.4607 | 16/39 | 36 | 2 | 1 | Chunk-evidence assigner; alpha=0.3. |
| 13 | `evidence_topk_1` | chunk_evidence | 0.4103 | 0.4593 | 16/39 | 37 | 2 | 0 | Chunk-evidence assigner; top_k_chunks=1. |
| 14 | `evidence_alpha_0_50` | chunk_evidence | 0.3846 | 0.4470 | 15/39 | 36 | 2 | 1 | Chunk-evidence assigner; alpha=0.5. |
| 15 | `hierarchical_alpha_1_0` | hierarchical_embeddings | 0.3590 | 0.3819 | 14/39 | 29 | 0 | 9 | Whole-document hierarchical assigner; alpha=1.0. |
| 16 | `baseline_keyword` | baseline | 0.2821 | 0.3436 | 11/39 | 38 | 0 | 1 | Rule-based keyword baseline. |
| 17 | `hierarchical_alpha_0_0` | hierarchical_embeddings | 0.2564 | 0.3476 | 10/39 | 38 | 0 | 1 | Whole-document hierarchical assigner; alpha=0.0. |

Gold labels: `data/labels/taxonomy_gold_labels.jsonl`.