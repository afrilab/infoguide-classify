# Classification Report — TFIDF

## Overview

| Metric | Value |
|---|---|
| Documents evaluated | 98 |
| Correct predictions | 86 / 98 (87.8%) |
| Needs Review | 0 |
| Macro F1 (active classes) | 83.9% |
| Weighted F1 | 89.7% |

## Per-Class Performance

| Label                     |   Support |   Correct | Accuracy   | Precision   | Recall   | F1    |
|:--------------------------|----------:|----------:|:-----------|:------------|:---------|:------|
| Policy_Procedure_Contract |        30 |        25 | 83.3%      | 89.3%       | 83.3%    | 86.2% |
| Reports                   |        45 |        41 | 91.1%      | 93.2%       | 91.1%    | 92.1% |
| HR_Documents              |         1 |         1 | 100.0%     | 50.0%       | 100.0%   | 66.7% |
| Forms_Structured          |        22 |        19 | 86.4%      | 95.0%       | 86.4%    | 90.5% |

## Confusion Matrix

|              |   Policy |   Reports |   HR |   Forms |
|:-------------|---------:|----------:|-----:|--------:|
| TRUE Policy  |       25 |         3 |    1 |       0 |
| TRUE Reports |        3 |        41 |    0 |       1 |
| TRUE HR      |        0 |         0 |    1 |       0 |
| TRUE Forms   |        0 |         0 |    0 |      19 |

## Misclassified Documents

| doc_id   | true_label                | predicted_label           | confidence   | margin   |
|:---------|:--------------------------|:--------------------------|:-------------|:---------|
| doc_0004 | Reports                   | Policy_Procedure_Contract | 0.056        | 0.018    |
| doc_0012 | Policy_Procedure_Contract | nan                       | —            | —        |
| doc_0020 | Policy_Procedure_Contract | Reports                   | 0.039        | 0.001    |
| doc_0025 | Policy_Procedure_Contract | Reports                   | 0.078        | 0.034    |
| doc_0030 | Reports                   | Policy_Procedure_Contract | 0.024        | 0.003    |
| doc_0036 | Reports                   | Forms_Structured          | 0.047        | 0.011    |
| doc_0043 | Policy_Procedure_Contract | Reports                   | 0.056        | 0.026    |
| doc_0066 | Reports                   | Policy_Procedure_Contract | 0.062        | 0.021    |
| doc_0082 | Forms_Structured          | nan                       | —            | —        |
| doc_0083 | Forms_Structured          | nan                       | —            | —        |
| doc_0091 | Forms_Structured          | Emails                    | 0.043        | 0.007    |
| doc_0096 | Policy_Procedure_Contract | HR_Documents              | 0.031        | 0.012    |

