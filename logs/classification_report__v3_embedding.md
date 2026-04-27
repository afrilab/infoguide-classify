# Classification Evaluation Report

## 1. Overall Summary

|   n_docs |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |   correct_docs |   wrong_docs |   needs_review_count |
|---------:|-----------:|------------------:|---------------:|-----------:|--------------:|---------------:|-------------:|---------------------:|
|       11 |   0.363636 |          0.111111 |       0.166667 |   0.133333 |      0.290909 |              4 |            7 |                    5 |

## 2. Per-Class Metrics

| label                     |   precision |   recall |   f1 |   support |
|:--------------------------|------------:|---------:|-----:|----------:|
| Policy_Procedure_Contract |    0        |        0 |  0   |         7 |
| Reports                   |    0.666667 |        1 |  0.8 |         4 |
| Internal_Communications   |    0        |        0 |  0   |         0 |
| Emails                    |    0        |        0 |  0   |         0 |
| HR_Documents              |    0        |        0 |  0   |         0 |
| Forms_Structured          |    0        |        0 |  0   |         0 |

## 3. Confusion Matrix

|                           |   Policy_Procedure_Contract |   Reports |   Internal_Communications |   Emails |   HR_Documents |   Forms_Structured |
|:--------------------------|----------------------------:|----------:|--------------------------:|---------:|---------------:|-------------------:|
| Policy_Procedure_Contract |                           0 |         2 |                         0 |        0 |              0 |                  0 |
| Reports                   |                           0 |         4 |                         0 |        0 |              0 |                  0 |
| Internal_Communications   |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| Emails                    |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| HR_Documents              |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| Forms_Structured          |                           0 |         0 |                         0 |        0 |              0 |                  0 |

## 4. Misclassified Documents

| doc_id   | true_label                | predicted_label   |   confidence | second_best               |   second_score |      margin |
|:---------|:--------------------------|:------------------|-------------:|:--------------------------|---------------:|------------:|
| doc_0003 | Policy_Procedure_Contract | Reports           |     0.348382 | Policy_Procedure_Contract |       0.181333 | 0.167048    |
| doc_0006 | Policy_Procedure_Contract | Needs_Review      |     0.236098 | Policy_Procedure_Contract |       0.235286 | 0.000812054 |
| doc_0007 | Policy_Procedure_Contract | Needs_Review      |     0.236098 | Policy_Procedure_Contract |       0.235286 | 0.000812054 |
| doc_0008 | Policy_Procedure_Contract | Needs_Review      |     0.296546 | Policy_Procedure_Contract |       0.248741 | 0.0478058   |
| doc_0009 | Policy_Procedure_Contract | Needs_Review      |     0.24438  | HR_Documents              |       0.166934 | 0.0774461   |
| doc_0010 | Policy_Procedure_Contract | Reports           |     0.317432 | Policy_Procedure_Contract |       0.216911 | 0.100522    |
| doc_0011 | Policy_Procedure_Contract | Needs_Review      |     0.203051 | Policy_Procedure_Contract |       0.176034 | 0.0270173   |

## 5. Prediction Confidence Statistics

| predicted_label   |   count |   avg_confidence |   avg_margin |
|:------------------|--------:|-----------------:|-------------:|
| Reports           |       6 |         0.36875  |    0.198225  |
| Needs_Review      |       5 |         0.243235 |    0.0307787 |

## 6. sklearn Classification Report

```text
                           precision    recall  f1-score   support

Policy_Procedure_Contract     0.0000    0.0000    0.0000         7
                  Reports     0.6667    1.0000    0.8000         4
  Internal_Communications     0.0000    0.0000    0.0000         0
                   Emails     0.0000    0.0000    0.0000         0
             HR_Documents     0.0000    0.0000    0.0000         0
         Forms_Structured     0.0000    0.0000    0.0000         0

                micro avg     0.6667    0.3636    0.4706        11
                macro avg     0.1111    0.1667    0.1333        11
             weighted avg     0.2424    0.3636    0.2909        11

```
