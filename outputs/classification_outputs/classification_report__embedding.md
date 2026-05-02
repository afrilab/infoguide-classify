# Classification Report — EMBEDDING

## Overview

| Metric | Value |
|---|---|
| Documents evaluated | 98 |
| Correct predictions | 81 / 98 (82.7%) |
| Needs Review | 8 |
| Macro F1 (active classes) | 90.2% |
| Weighted F1 | 87.3% |

## Per-Class Performance

| Label                     |   Support |   Correct | Accuracy   | Precision   | Recall   | F1     |
|:--------------------------|----------:|----------:|:-----------|:------------|:---------|:-------|
| Policy_Procedure_Contract |        30 |        21 | 70.0%      | 91.3%       | 70.0%    | 79.2%  |
| Reports                   |        45 |        39 | 86.7%      | 95.1%       | 86.7%    | 90.7%  |
| HR_Documents              |         1 |         1 | 100.0%     | 100.0%      | 100.0%   | 100.0% |
| Forms_Structured          |        22 |        20 | 90.9%      | 90.9%       | 90.9%    | 90.9%  |

## Confusion Matrix

|              |   Policy |   Reports |   HR |   Forms |
|:-------------|---------:|----------:|-----:|--------:|
| TRUE Policy  |       21 |         2 |    0 |       1 |
| TRUE Reports |        2 |        39 |    0 |       1 |
| TRUE HR      |        0 |         0 |    1 |       0 |
| TRUE Forms   |        0 |         0 |    0 |      20 |

## Misclassified Documents

| doc_id   | true_label                | predicted_label           | confidence   | margin   |
|:---------|:--------------------------|:--------------------------|:-------------|:---------|
| doc_0003 | Policy_Procedure_Contract | Needs_Review              | 0.598        | 0.005    |
| doc_0012 | Policy_Procedure_Contract | nan                       | —            | —        |
| doc_0020 | Policy_Procedure_Contract | Reports                   | 0.675        | 0.026    |
| doc_0025 | Policy_Procedure_Contract | Reports                   | 0.686        | 0.059    |
| doc_0026 | Policy_Procedure_Contract | Needs_Review              | 0.674        | 0.000    |
| doc_0028 | Reports                   | Forms_Structured          | 0.680        | 0.041    |
| doc_0030 | Reports                   | Policy_Procedure_Contract | 0.694        | 0.012    |
| doc_0033 | Policy_Procedure_Contract | Needs_Review              | 0.642        | 0.006    |
| doc_0035 | Reports                   | Policy_Procedure_Contract | 0.680        | 0.022    |
| doc_0037 | Policy_Procedure_Contract | Needs_Review              | 0.682        | 0.002    |
| doc_0038 | Policy_Procedure_Contract | Needs_Review              | 0.693        | 0.007    |
| doc_0047 | Reports                   | Needs_Review              | 0.597        | 0.002    |
| doc_0051 | Reports                   | Needs_Review              | 0.641        | 0.007    |
| doc_0066 | Reports                   | Needs_Review              | 0.646        | 0.009    |
| doc_0082 | Forms_Structured          | nan                       | —            | —        |
| doc_0083 | Forms_Structured          | nan                       | —            | —        |
| doc_0096 | Policy_Procedure_Contract | Forms_Structured          | 0.660        | 0.012    |

