# Classification Report — EMBEDDING LARGE

## Overview

| Metric | Value |
|---|---|
| Documents evaluated | 98 |
| Correct predictions | 82 / 98 (83.7%) |
| Needs Review | 8 |
| Macro F1 (active classes) | 83.0% |
| Weighted F1 | 88.6% |

## Per-Class Performance

| Label                     |   Support |   Correct | Accuracy   | Precision   | Recall   | F1    |
|:--------------------------|----------:|----------:|:-----------|:------------|:---------|:------|
| Policy_Procedure_Contract |        30 |        22 | 73.3%      | 95.7%       | 73.3%    | 83.0% |
| Reports                   |        45 |        40 | 88.9%      | 95.2%       | 88.9%    | 92.0% |
| HR_Documents              |         1 |         1 | 100.0%     | 50.0%       | 100.0%   | 66.7% |
| Forms_Structured          |        22 |        19 | 86.4%      | 95.0%       | 86.4%    | 90.5% |

## Confusion Matrix

|              |   Policy |   Reports |   HR |   Forms |
|:-------------|---------:|----------:|-----:|--------:|
| TRUE Policy  |       22 |         2 |    0 |       0 |
| TRUE Reports |        1 |        40 |    0 |       1 |
| TRUE HR      |        0 |         0 |    1 |       0 |
| TRUE Forms   |        0 |         0 |    1 |      19 |

## Misclassified Documents

| doc_id   | true_label                | predicted_label           | confidence   | margin   |
|:---------|:--------------------------|:--------------------------|:-------------|:---------|
| doc_0011 | Policy_Procedure_Contract | Needs_Review              | 0.578        | 0.009    |
| doc_0012 | Policy_Procedure_Contract | nan                       | —            | —        |
| doc_0020 | Policy_Procedure_Contract | Needs_Review              | 0.587        | 0.003    |
| doc_0025 | Policy_Procedure_Contract | Reports                   | 0.563        | 0.061    |
| doc_0028 | Reports                   | Forms_Structured          | 0.572        | 0.019    |
| doc_0030 | Reports                   | Needs_Review              | 0.605        | 0.007    |
| doc_0033 | Policy_Procedure_Contract | Needs_Review              | 0.562        | 0.002    |
| doc_0035 | Reports                   | Policy_Procedure_Contract | 0.621        | 0.030    |
| doc_0037 | Policy_Procedure_Contract | Needs_Review              | 0.616        | 0.004    |
| doc_0043 | Policy_Procedure_Contract | Needs_Review              | 0.541        | 0.003    |
| doc_0047 | Reports                   | Needs_Review              | 0.557        | 0.001    |
| doc_0051 | Reports                   | Needs_Review              | 0.566        | 0.009    |
| doc_0082 | Forms_Structured          | nan                       | —            | —        |
| doc_0083 | Forms_Structured          | nan                       | —            | —        |
| doc_0087 | Forms_Structured          | HR_Documents              | 0.551        | 0.049    |
| doc_0096 | Policy_Procedure_Contract | Reports                   | 0.622        | 0.010    |

