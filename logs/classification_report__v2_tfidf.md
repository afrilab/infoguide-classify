# Classification Evaluation Report

## 1. Overall Summary

|   n_docs |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |   correct_docs |   wrong_docs |   needs_review_count |
|---------:|-----------:|------------------:|---------------:|-----------:|--------------:|---------------:|-------------:|---------------------:|
|       11 |   0.727273 |          0.333333 |       0.261905 |   0.287879 |      0.826446 |              8 |            3 |                    3 |

## 2. Per-Class Metrics

| label                     |   precision |   recall |       f1 |   support |
|:--------------------------|------------:|---------:|---------:|----------:|
| Policy_Procedure_Contract |           1 | 0.571429 | 0.727273 |         7 |
| Reports                   |           1 | 1        | 1        |         4 |
| Internal_Communications   |           0 | 0        | 0        |         0 |
| Emails                    |           0 | 0        | 0        |         0 |
| HR_Documents              |           0 | 0        | 0        |         0 |
| Forms_Structured          |           0 | 0        | 0        |         0 |

## 3. Confusion Matrix

|                           |   Policy_Procedure_Contract |   Reports |   Internal_Communications |   Emails |   HR_Documents |   Forms_Structured |
|:--------------------------|----------------------------:|----------:|--------------------------:|---------:|---------------:|-------------------:|
| Policy_Procedure_Contract |                           4 |         0 |                         0 |        0 |              0 |                  0 |
| Reports                   |                           0 |         4 |                         0 |        0 |              0 |                  0 |
| Internal_Communications   |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| Emails                    |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| HR_Documents              |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| Forms_Structured          |                           0 |         0 |                         0 |        0 |              0 |                  0 |

## 4. Misclassified Documents

| doc_id   | true_label                | predicted_label   |   confidence | second_best               |   second_score |    margin |
|:---------|:--------------------------|:------------------|-------------:|:--------------------------|---------------:|----------:|
| doc_0003 | Policy_Procedure_Contract | Needs_Review      |    0.0638513 | Policy_Procedure_Contract |      0.0309567 | 0.0328946 |
| doc_0008 | Policy_Procedure_Contract | Needs_Review      |    0.0580098 | Reports                   |      0.0433313 | 0.0146785 |
| doc_0011 | Policy_Procedure_Contract | Needs_Review      |    0.0455594 | Reports                   |      0.0413964 | 0.004163  |

## 5. Prediction Confidence Statistics

| predicted_label           |   count |   avg_confidence |   avg_margin |
|:--------------------------|--------:|-----------------:|-------------:|
| Policy_Procedure_Contract |       4 |        0.142295  |    0.104048  |
| Reports                   |       4 |        0.14134   |    0.109859  |
| Needs_Review              |       3 |        0.0558068 |    0.0172454 |

## 6. sklearn Classification Report

```text
                           precision    recall  f1-score   support

Policy_Procedure_Contract     1.0000    0.5714    0.7273         7
                  Reports     1.0000    1.0000    1.0000         4
  Internal_Communications     0.0000    0.0000    0.0000         0
                   Emails     0.0000    0.0000    0.0000         0
             HR_Documents     0.0000    0.0000    0.0000         0
         Forms_Structured     0.0000    0.0000    0.0000         0

                micro avg     1.0000    0.7273    0.8421        11
                macro avg     0.3333    0.2619    0.2879        11
             weighted avg     1.0000    0.7273    0.8264        11

```
