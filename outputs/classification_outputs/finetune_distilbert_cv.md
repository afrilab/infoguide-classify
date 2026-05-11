# Fine-Tuned Transformer — distilbert-base-uncased

- N = 94 documents, k = 5 folds
- Class distribution: Forms_Structured=20, Policy_Procedure_Contract=29, Reports=45
- Hyperparameters: epochs=4, batch=8, lr=2e-05, weight_decay=0.01, max_len=256, warmup=0.1, seed=42

## Aggregate metrics (mean ± std across folds)

| Metric | Value |
| --- | --- |
| Accuracy | 0.798 ± 0.090 |
| Macro F1 | 0.771 ± 0.109 |
| Weighted F1 | 0.766 ± 0.112 |
| F1 (Forms_Structured) | 0.949 ± 0.063 |
| F1 (Policy_Procedure_Contract) | 0.523 ± 0.240 |
| F1 (Reports) | 0.841 ± 0.066 |

## Per-fold metrics

| Fold | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- |
| 0 | 0.684 | 0.634 | 0.627 |
| 1 | 0.737 | 0.689 | 0.671 |
| 2 | 0.789 | 0.771 | 0.776 |
| 3 | 0.947 | 0.952 | 0.946 |
| 4 | 0.833 | 0.810 | 0.810 |
