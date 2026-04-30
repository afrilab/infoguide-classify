# Classification Evaluation Report

## 1. Overall Summary

|   n_docs |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |   correct_docs |   wrong_docs |   needs_review_count |
|---------:|-----------:|------------------:|---------------:|-----------:|--------------:|---------------:|-------------:|---------------------:|
|       98 |   0.877551 |          0.545779 |       0.601347 |   0.559141 |      0.896879 |             86 |           12 |                    0 |

## 2. Per-Class Metrics

| label                     |   precision |   recall |       f1 |   support |
|:--------------------------|------------:|---------:|---------:|----------:|
| Policy_Procedure_Contract |    0.892857 | 0.833333 | 0.862069 |        30 |
| Reports                   |    0.931818 | 0.911111 | 0.921348 |        45 |
| Internal_Communications   |    0        | 0        | 0        |         0 |
| Emails                    |    0        | 0        | 0        |         0 |
| HR_Documents              |    0.5      | 1        | 0.666667 |         1 |
| Forms_Structured          |    0.95     | 0.863636 | 0.904762 |        22 |

## 3. Confusion Matrix

|                           |   Policy_Procedure_Contract |   Reports |   Internal_Communications |   Emails |   HR_Documents |   Forms_Structured |
|:--------------------------|----------------------------:|----------:|--------------------------:|---------:|---------------:|-------------------:|
| Policy_Procedure_Contract |                          25 |         3 |                         0 |        0 |              1 |                  0 |
| Reports                   |                           3 |        41 |                         0 |        0 |              0 |                  1 |
| Internal_Communications   |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| Emails                    |                           0 |         0 |                         0 |        0 |              0 |                  0 |
| HR_Documents              |                           0 |         0 |                         0 |        0 |              1 |                  0 |
| Forms_Structured          |                           0 |         0 |                         0 |        1 |              0 |                 19 |

## 4. Misclassified Documents

| doc_id   | true_label                | predicted_label           |   confidence | second_best               |   second_score |       margin |
|:---------|:--------------------------|:--------------------------|-------------:|:--------------------------|---------------:|-------------:|
| doc_0004 | Reports                   | Policy_Procedure_Contract |    0.070459  | Reports                   |      0.0591095 |   0.0113494  |
| doc_0012 | Policy_Procedure_Contract | nan                       |  nan         | nan                       |    nan         | nan          |
| doc_0020 | Policy_Procedure_Contract | Reports                   |    0.0409368 | Policy_Procedure_Contract |      0.0364448 |   0.00449209 |
| doc_0025 | Policy_Procedure_Contract | Reports                   |    0.0812496 | Policy_Procedure_Contract |      0.0533929 |   0.0278567  |
| doc_0030 | Reports                   | Policy_Procedure_Contract |    0.0335701 | Reports                   |      0.0233055 |   0.0102646  |
| doc_0036 | Reports                   | Forms_Structured          |    0.0450799 | Policy_Procedure_Contract |      0.0426615 |   0.00241849 |
| doc_0043 | Policy_Procedure_Contract | Reports                   |    0.0656828 | Policy_Procedure_Contract |      0.0301433 |   0.0355395  |
| doc_0066 | Reports                   | Policy_Procedure_Contract |    0.0808498 | Reports                   |      0.0594661 |   0.0213838  |
| doc_0082 | Forms_Structured          | nan                       |  nan         | nan                       |    nan         | nan          |
| doc_0083 | Forms_Structured          | nan                       |  nan         | nan                       |    nan         | nan          |
| doc_0091 | Forms_Structured          | Emails                    |    0.0504679 | Forms_Structured          |      0.0435359 |   0.00693201 |
| doc_0096 | Policy_Procedure_Contract | HR_Documents              |    0.0286171 | Forms_Structured          |      0.0215609 |   0.0070562  |

## 5. Prediction Confidence Statistics

| predicted_label           |   count |   avg_confidence |   avg_margin |
|:--------------------------|--------:|-----------------:|-------------:|
| Reports                   |      44 |        0.0678519 |   0.0398282  |
| Policy_Procedure_Contract |      28 |        0.0933805 |   0.0663902  |
| Forms_Structured          |      20 |        0.104441  |   0.0736441  |
| nan                       |       3 |      nan         | nan          |
| HR_Documents              |       2 |        0.0700277 |   0.0445877  |
| Emails                    |       1 |        0.0504679 |   0.00693201 |

## 6. sklearn Classification Report

```text
                           precision    recall  f1-score   support

Policy_Procedure_Contract     0.8929    0.8333    0.8621        30
                  Reports     0.9318    0.9111    0.9213        45
  Internal_Communications     0.0000    0.0000    0.0000         0
                   Emails     0.0000    0.0000    0.0000         0
             HR_Documents     0.5000    1.0000    0.6667         1
         Forms_Structured     0.9500    0.8636    0.9048        22

                micro avg     0.9053    0.8776    0.8912        98
                macro avg     0.5458    0.6013    0.5591        98
             weighted avg     0.9196    0.8776    0.8969        98

```
