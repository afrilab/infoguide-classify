# Supervised Baselines — Stratified 5-Fold CV
- N = 94 documents
- k = 5 folds
- Class distribution: Forms_Structured=20, Policy_Procedure_Contract=29, Reports=45
- Shared TF-IDF preprocessing: ngram=[1,2], english stop words, lowercase, min_df=1, head_only=True, max_chars=6000

## Aggregate metrics (mean ± std across folds)

| Model | Accuracy | Macro F1 | Weighted F1 | F1 (Forms_Structured) | F1 (Policy_Procedure_Contract) | F1 (Reports) |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.905 ± 0.061 | 0.902 ± 0.068 | 0.903 ± 0.063 | 0.911 | 0.883 | 0.911 |
| LinearSVM | 0.884 ± 0.090 | 0.880 ± 0.094 | 0.880 ± 0.094 | 0.911 | 0.835 | 0.894 |
| MultinomialNB | 0.809 ± 0.024 | 0.787 ± 0.020 | 0.792 ± 0.027 | 0.838 | 0.689 | 0.834 |
| RandomForest | 0.894 ± 0.088 | 0.899 ± 0.084 | 0.892 ± 0.088 | 0.978 | 0.822 | 0.897 |
