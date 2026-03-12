# Classification Evaluation Report

## 1. Overall Summary

| n_docs | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | correct_docs | wrong_docs | needs_review_count |
| -----: | -------: | --------------: | -----------: | -------: | ----------: | -----------: | ---------: | -----------------: |
|     11 | 0.272727 |        0.407407 |          0.4 | 0.232323 |    0.217631 |            3 |          8 |                  1 |

## 2. Per-Class Metrics

| label      | precision | recall |       f1 | support |
| :--------- | --------: | -----: | -------: | ------: |
| Finance    |         0 |      0 |        0 |       4 |
| Compliance |         1 |    0.2 | 0.333333 |       5 |
| Risk       |  0.222222 |      1 | 0.363636 |       2 |

## 3. Confusion Matrix

|            | Finance | Compliance | Risk |
| :--------- | ------: | ---------: | ---: |
| Finance    |       0 |          0 |    4 |
| Compliance |       0 |          1 |    3 |
| Risk       |       0 |          0 |    2 |

## 4. Misclassified Documents

| doc_id   | true_label | predicted_label | confidence | second_best | second_score |    margin |
| :------- | :--------- | :-------------- | ---------: | :---------- | -----------: | --------: |
| doc_0001 | Finance    | Risk            |   0.384671 | Policy      |     0.282966 |  0.101705 |
| doc_0002 | Finance    | Risk            |   0.331338 | Policy      |     0.252827 | 0.0785101 |
| doc_0003 | Compliance | Needs_Review    |   0.267908 | Finance     |     0.255858 | 0.0120499 |
| doc_0004 | Finance    | Risk            |   0.394173 | Finance     |     0.231132 |  0.163041 |
| doc_0005 | Finance    | Risk            |   0.575453 | Compliance  |     0.191976 |  0.383477 |
| doc_0007 | Compliance | Risk            |   0.356323 | Policy      |     0.253337 |  0.102986 |
| doc_0009 | Compliance | Risk            |    0.58875 | Policy      |      0.18613 |   0.40262 |
| doc_0010 | Compliance | Risk            |   0.361803 | Compliance  |      0.30978 | 0.0520229 |

## 5. Prediction Confidence Statistics

| predicted_label | count | avg_confidence | avg_margin |
| :-------------- | ----: | -------------: | ---------: |
| Risk            |     9 |       0.417259 |   0.173767 |
| Compliance      |     1 |       0.395571 |   0.044185 |
| Needs_Review    |     1 |       0.267908 |  0.0120499 |

## 6. sklearn Classification Report

```text
              precision    recall  f1-score   support

     Finance     0.0000    0.0000    0.0000         4
  Compliance     1.0000    0.2000    0.3333         5
        Risk     0.2222    1.0000    0.3636         2

   micro avg     0.3000    0.2727    0.2857        11
   macro avg     0.4074    0.4000    0.2323        11
weighted avg     0.4949    0.2727    0.2176        11

```

## 7. Interpretation

- The model over-predicts **Risk**.
- **Finance** is completely missed in this evaluation slice.
- **Compliance** has high precision but low recall.
- `Needs_Review` is treated as an incorrect prediction for label metrics.
