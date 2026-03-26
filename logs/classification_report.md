# Classification Evaluation Report

## 1. Overall Summary

|   n_docs |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |   correct_docs |   wrong_docs |   needs_review_count |
|---------:|-----------:|------------------:|---------------:|-----------:|--------------:|---------------:|-------------:|---------------------:|
|       11 |   0.545455 |          0.833333 |           0.55 |   0.642857 |      0.662338 |              6 |            5 |                    2 |

## 2. Per-Class Metrics

| label      |   precision |   recall |       f1 |   support |
|:-----------|------------:|---------:|---------:|----------:|
| Finance    |         1   |     0.75 | 0.857143 |         4 |
| Compliance |         1   |     0.4  | 0.571429 |         5 |
| Risk       |         0.5 |     0.5  | 0.5      |         2 |

## 3. Confusion Matrix

|            |   Finance |   Compliance |   Risk |
|:-----------|----------:|-------------:|-------:|
| Finance    |         3 |            0 |      0 |
| Compliance |         0 |            2 |      1 |
| Risk       |         0 |            0 |      1 |

## 4. Misclassified Documents

| doc_id   | true_label   | predicted_label   |   confidence | second_best   |   second_score |    margin |
|:---------|:-------------|:------------------|-------------:|:--------------|---------------:|----------:|
| doc_0002 | Finance      | Policy            |     0.48089  | Risk          |       0.238133 | 0.242757  |
| doc_0006 | Risk         | Needs_Review      |     0.389842 | Risk          |       0.368616 | 0.0212257 |
| doc_0007 | Compliance   | Policy            |     0.511105 | Compliance    |       0.197849 | 0.313256  |
| doc_0009 | Compliance   | Risk              |     0.419025 | Policy        |       0.260236 | 0.158788  |
| doc_0010 | Compliance   | Needs_Review      |     0.36183  | Risk          |       0.312061 | 0.0497682 |

## 5. Prediction Confidence Statistics

| predicted_label   |   count |   avg_confidence |   avg_margin |
|:------------------|--------:|-----------------:|-------------:|
| Finance           |       3 |         0.537573 |     0.331687 |
| Compliance        |       2 |         0.522608 |     0.342134 |
| Needs_Review      |       2 |         0.375836 |     0.035497 |
| Policy            |       2 |         0.495998 |     0.278006 |
| Risk              |       2 |         0.506289 |     0.284761 |

## 6. sklearn Classification Report

```text
              precision    recall  f1-score   support

     Finance     1.0000    0.7500    0.8571         4
  Compliance     1.0000    0.4000    0.5714         5
        Risk     0.5000    0.5000    0.5000         2

   micro avg     0.8571    0.5455    0.6667        11
   macro avg     0.8333    0.5500    0.6429        11
weighted avg     0.9091    0.5455    0.6623        11

```

## 7. Interpretation

- The model over-predicts **Risk**.
- **Finance** is completely missed in this evaluation slice.
- **Compliance** has high precision but low recall.
- `Needs_Review` is treated as an incorrect prediction for label metrics.
