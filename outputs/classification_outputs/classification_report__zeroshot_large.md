# Classification Report — ZEROSHOT LARGE

## Overview

| Metric | Value |
|---|---|
| Documents evaluated | 98 |
| Correct predictions | 24 / 98 (24.5%) |
| Needs Review | 25 |
| Macro F1 (active classes) | 24.2% |
| Weighted F1 | 35.0% |

## Per-Class Performance

| Label                     |   Support |   Correct | Accuracy   | Precision   | Recall   | F1    |
|:--------------------------|----------:|----------:|:-----------|:------------|:---------|:------|
| Policy_Procedure_Contract |        30 |         1 | 3.3%       | 12.5%       | 3.3%     | 5.3%  |
| Reports                   |        45 |        18 | 40.0%      | 85.7%       | 40.0%    | 54.5% |
| HR_Documents              |         1 |         0 | 0.0%       | 0.0%        | 0.0%     | 0.0%  |
| Forms_Structured          |        22 |         5 | 22.7%      | 100.0%      | 22.7%    | 37.0% |

## Confusion Matrix

|              |   Policy |   Reports |   HR |   Forms |
|:-------------|---------:|----------:|-----:|--------:|
| TRUE Policy  |        1 |         3 |   11 |       0 |
| TRUE Reports |        4 |        18 |    3 |       0 |
| TRUE HR      |        0 |         0 |    0 |       0 |
| TRUE Forms   |        3 |         0 |    3 |       5 |

## Misclassified Documents

| doc_id   | true_label                | predicted_label           | confidence   | margin   |
|:---------|:--------------------------|:--------------------------|:-------------|:---------|
| doc_0003 | Policy_Procedure_Contract | Emails                    | 0.793        | 0.741    |
| doc_0006 | Policy_Procedure_Contract | Emails                    | 0.563        | 0.443    |
| doc_0007 | Policy_Procedure_Contract | Emails                    | 0.563        | 0.443    |
| doc_0008 | Policy_Procedure_Contract | Emails                    | 0.337        | 0.051    |
| doc_0009 | Policy_Procedure_Contract | HR_Documents              | 0.341        | 0.055    |
| doc_0010 | Policy_Procedure_Contract | Reports                   | 0.374        | 0.130    |
| doc_0011 | Policy_Procedure_Contract | Needs_Review              | 0.254        | 0.017    |
| doc_0012 | Policy_Procedure_Contract | nan                       | —            | —        |
| doc_0013 | Policy_Procedure_Contract | HR_Documents              | 0.222        | 0.044    |
| doc_0015 | Policy_Procedure_Contract | Emails                    | 0.259        | 0.032    |
| doc_0016 | Policy_Procedure_Contract | Needs_Review              | 0.216        | 0.026    |
| doc_0019 | Reports                   | Needs_Review              | 0.265        | 0.025    |
| doc_0020 | Policy_Procedure_Contract | Reports                   | 0.685        | 0.607    |
| doc_0021 | Policy_Procedure_Contract | Reports                   | 0.277        | 0.072    |
| doc_0024 | Policy_Procedure_Contract | HR_Documents              | 0.316        | 0.099    |
| doc_0025 | Policy_Procedure_Contract | Needs_Review              | 0.234        | 0.025    |
| doc_0026 | Policy_Procedure_Contract | HR_Documents              | 0.534        | 0.386    |
| doc_0027 | Reports                   | Needs_Review              | 0.183        | 0.001    |
| doc_0028 | Reports                   | Needs_Review              | 0.228        | 0.001    |
| doc_0029 | Policy_Procedure_Contract | HR_Documents              | 0.280        | 0.097    |
| doc_0030 | Reports                   | Emails                    | 0.339        | 0.099    |
| doc_0031 | Policy_Procedure_Contract | Emails                    | 0.356        | 0.089    |
| doc_0032 | Policy_Procedure_Contract | Needs_Review              | 0.208        | 0.007    |
| doc_0033 | Policy_Procedure_Contract | HR_Documents              | 0.252        | 0.033    |
| doc_0034 | Policy_Procedure_Contract | HR_Documents              | 0.319        | 0.078    |
| doc_0035 | Reports                   | HR_Documents              | 0.287        | 0.033    |
| doc_0036 | Reports                   | Needs_Review              | 0.271        | 0.013    |
| doc_0037 | Policy_Procedure_Contract | HR_Documents              | 0.224        | 0.041    |
| doc_0038 | Policy_Procedure_Contract | Needs_Review              | 0.200        | 0.001    |
| doc_0039 | Policy_Procedure_Contract | HR_Documents              | 0.344        | 0.074    |
| doc_0040 | Policy_Procedure_Contract | Needs_Review              | 0.213        | 0.014    |
| doc_0041 | Policy_Procedure_Contract | Needs_Review              | 0.327        | 0.012    |
| doc_0042 | Policy_Procedure_Contract | Needs_Review              | 0.209        | 0.004    |
| doc_0043 | Policy_Procedure_Contract | HR_Documents              | 0.397        | 0.213    |
| doc_0044 | Reports                   | Emails                    | 0.508        | 0.288    |
| doc_0045 | Reports                   | Needs_Review              | 0.256        | 0.004    |
| doc_0046 | Reports                   | Needs_Review              | 0.255        | 0.027    |
| doc_0048 | Reports                   | Policy_Procedure_Contract | 0.301        | 0.089    |
| doc_0049 | Reports                   | Policy_Procedure_Contract | 0.342        | 0.072    |
| doc_0051 | Reports                   | Needs_Review              | 0.209        | 0.010    |
| doc_0052 | Reports                   | HR_Documents              | 0.291        | 0.045    |
| doc_0054 | Reports                   | Emails                    | 0.319        | 0.058    |
| doc_0055 | Reports                   | Emails                    | 0.317        | 0.112    |
| doc_0056 | Reports                   | Needs_Review              | 0.248        | 0.016    |
| doc_0057 | Reports                   | Policy_Procedure_Contract | 0.304        | 0.035    |
| doc_0060 | Reports                   | Emails                    | 0.266        | 0.067    |
| doc_0061 | Reports                   | Needs_Review              | 0.242        | 0.018    |
| doc_0062 | Reports                   | HR_Documents              | 0.463        | 0.295    |
| doc_0064 | Reports                   | Needs_Review              | 0.248        | 0.029    |
| doc_0066 | Reports                   | Emails                    | 0.562        | 0.251    |
| doc_0067 | Reports                   | Emails                    | 0.914        | 0.881    |
| doc_0068 | Reports                   | Policy_Procedure_Contract | 0.281        | 0.093    |
| doc_0071 | Reports                   | Emails                    | 0.314        | 0.121    |
| doc_0073 | Reports                   | Emails                    | 0.572        | 0.458    |
| doc_0074 | Reports                   | Needs_Review              | 0.263        | 0.029    |
| doc_0075 | HR_Documents              | Emails                    | 0.536        | 0.178    |
| doc_0076 | Forms_Structured          | Emails                    | 0.260        | 0.058    |
| doc_0080 | Forms_Structured          | Emails                    | 0.260        | 0.058    |
| doc_0081 | Forms_Structured          | Needs_Review              | 0.234        | 0.015    |
| doc_0082 | Forms_Structured          | nan                       | —            | —        |
| doc_0083 | Forms_Structured          | nan                       | —            | —        |
| doc_0084 | Forms_Structured          | Policy_Procedure_Contract | 0.259        | 0.033    |
| doc_0085 | Forms_Structured          | HR_Documents              | 0.337        | 0.069    |
| doc_0086 | Forms_Structured          | Needs_Review              | 0.212        | 0.026    |
| doc_0087 | Forms_Structured          | Needs_Review              | 0.237        | 0.028    |
| doc_0088 | Forms_Structured          | HR_Documents              | 0.252        | 0.059    |
| doc_0089 | Forms_Structured          | HR_Documents              | 0.292        | 0.075    |
| doc_0090 | Forms_Structured          | Needs_Review              | 0.239        | 0.001    |
| doc_0091 | Forms_Structured          | Policy_Procedure_Contract | 0.413        | 0.074    |
| doc_0092 | Forms_Structured          | Needs_Review              | 0.236        | 0.006    |
| doc_0094 | Forms_Structured          | Needs_Review              | 0.278        | 0.005    |
| doc_0095 | Forms_Structured          | Emails                    | 0.211        | 0.043    |
| doc_0096 | Policy_Procedure_Contract | HR_Documents              | 0.327        | 0.155    |
| doc_0098 | Forms_Structured          | Policy_Procedure_Contract | 0.391        | 0.072    |

