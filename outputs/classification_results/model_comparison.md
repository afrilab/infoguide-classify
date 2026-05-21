# Model Comparison — 5-Fold Stratified CV (N=94)

| Rank | Model | Type | Accuracy | Macro F1 | Weighted F1 | F1 (Forms_Structured) | F1 (Policy_Procedure_Contract) | F1 (Reports) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | TF-IDF cosine | Unsupervised (label desc) | 0.904 | 0.916 | 0.914 | 0.950 | 0.877 | 0.921 |
| 2 | BGE Large embedding | Unsupervised (label desc) | 0.862 | 0.905 | 0.903 | 0.950 | 0.846 | 0.920 |
| 3 | Logistic Regression | Supervised (lexical) | 0.905 ± 0.061 | 0.902 ± 0.068 | 0.903 | 0.923 | 0.881 | 0.911 |
| 4 | Random Forest | Supervised (lexical) | 0.894 ± 0.088 | 0.899 ± 0.084 | 0.892 | 0.976 | 0.821 | 0.901 |
| 5 | BGE Small embedding | Unsupervised (label desc) | 0.851 | 0.889 | 0.886 | 0.952 | 0.808 | 0.907 |
| 6 | Linear SVM | Supervised (lexical) | 0.884 ± 0.090 | 0.880 ± 0.094 | 0.880 | 0.923 | 0.836 | 0.894 |
| 7 | Multinomial NB | Supervised (lexical) | 0.809 ± 0.024 | 0.787 ± 0.020 | 0.792 | 0.857 | 0.711 | 0.833 |
| 8 | DistilBERT | Supervised (transformer) | 0.798 ± 0.090 | 0.771 ± 0.109 | 0.766 | 0.950 | 0.558 | 0.838 |
| 9 | BART zero-shot NLI | Zero-shot NLI | 0.500 | 0.449 | 0.494 | 0.567 | 0.000 | 0.779 |
| 10 | DeBERTa zero-shot NLI | Zero-shot NLI | 0.255 | 0.333 | 0.363 | 0.400 | 0.054 | 0.545 |
