# Classification Evaluation Report

## 1. Overall Summary

|   n_docs |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |   correct_docs |   wrong_docs |   needs_review_count |
|---------:|-----------:|------------------:|---------------:|-----------:|--------------:|---------------:|-------------:|---------------------:|
|       11 |   0.727273 |            0.3125 |       0.208333 |   0.222222 |      0.739394 |              8 |            3 |                    2 |

## 2. Per-Class Metrics

| label                     |   precision |   recall |       f1 |   support |
|:--------------------------|------------:|---------:|---------:|----------:|
| Policy_Procedure_Contract |       0.875 |     1    | 0.933333 |         7 |
| Reports                   |       1     |     0.25 | 0.4      |         4 |
| Internal_Communications   |       0     |     0    | 0        |         0 |
| Emails                    |       0     |     0    | 0        |         0 |
| HR_Documents              |       0     |     0    | 0        |         0 |
| Forms_Structured          |       0     |     0    | 0        |         0 |

## 3. Confusion Matrix

|                           |   Policy_Procedure_Contract |   Reports |   Internal_Communications |   Emails |   HR_Documents |   Forms_Structured |
|:--------------------------|----------------------------:|----------:|--------------------------:|---------:|---------------:|-------------------:|
| Policy_Procedure_Contract |                           7 |         0 |                         0 |        0 |              0 |                  0 |
| Reports                   |                           1 |         1 |                         0 |        0 |              0 |                  0 |
| Internal_Communications   |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| Emails                    |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| HR_Documents              |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| Forms_Structured          |                           0 |         0 |                         0 |        0 |              0 |                  0 |

## 4. Misclassified Documents

| doc_id   | true_label   | predicted_label           |   confidence | second_best               |   second_score |    margin |
|:---------|:-------------|:--------------------------|-------------:|:--------------------------|---------------:|----------:|
| doc_0001 | Reports      | Policy_Procedure_Contract |     0.519035 | Reports                   |       0.302849 | 0.216186  |
| doc_0002 | Reports      | Needs_Review              |     0.236244 | Internal_Communications   |       0.222102 | 0.0141426 |
| doc_0005 | Reports      | Needs_Review              |     0.417167 | Policy_Procedure_Contract |       0.340361 | 0.0768065 |

## 5. Prediction Confidence Statistics

| predicted_label           |   count |   avg_confidence |   avg_margin |
|:--------------------------|--------:|-----------------:|-------------:|
| Policy_Procedure_Contract |       8 |         0.561753 |    0.387601  |
| Needs_Review              |       2 |         0.326706 |    0.0454745 |
| Reports                   |       1 |         0.5219   |    0.165866  |

## 6. sklearn Classification Report

```text
                           precision    recall  f1-score   support

Policy_Procedure_Contract     0.8750    1.0000    0.9333         7
                  Reports     1.0000    0.2500    0.4000         4
  Internal_Communications     0.0000    0.0000    0.0000         0
                   Emails     0.0000    0.0000    0.0000         0
             HR_Documents     0.0000    0.0000    0.0000         0
         Forms_Structured     0.0000    0.0000    0.0000         0

                micro avg     0.8889    0.7273    0.8000        11
                macro avg     0.3125    0.2083    0.2222        11
             weighted avg     0.9205    0.7273    0.7394        11

```
