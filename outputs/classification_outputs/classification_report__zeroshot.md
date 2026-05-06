# Classification Report — ZEROSHOT

## Overview

| Metric | Value |
|---|---|
| Documents evaluated | 98 |
| Correct predictions | 47 / 98 (48.0%) |
| Needs Review | 10 |
| Macro F1 (active classes) | 33.2% |
| Weighted F1 | 48.1% |

## Per-Class Performance

| Label                     |   Support |   Correct | Accuracy   | Precision   | Recall   | F1    |
|:--------------------------|----------:|----------:|:-----------|:------------|:---------|:------|
| Policy_Procedure_Contract |        30 |         0 | 0.0%       | 0.0%        | 0.0%     | 0.0%  |
| Reports                   |        45 |        30 | 66.7%      | 93.8%       | 66.7%    | 77.9% |
| HR_Documents              |         1 |         0 | 0.0%       | 0.0%        | 0.0%     | 0.0%  |
| Forms_Structured          |        22 |        17 | 77.3%      | 42.5%       | 77.3%    | 54.8% |

## Confusion Matrix

|              |   Policy |   Reports |   HR |   Forms |
|:-------------|---------:|----------:|-----:|--------:|
| TRUE Policy  |        0 |         2 |    0 |      16 |
| TRUE Reports |        0 |        30 |    0 |       7 |
| TRUE HR      |        0 |         0 |    0 |       0 |
| TRUE Forms   |        0 |         0 |    0 |      17 |

## Misclassified Documents

| doc_id   | true_label                | predicted_label         | confidence   | margin   |
|:---------|:--------------------------|:------------------------|:-------------|:---------|
| doc_0003 | Policy_Procedure_Contract | Needs_Review            | 0.284        | 0.009    |
| doc_0006 | Policy_Procedure_Contract | Internal_Communications | 0.217        | 0.035    |
| doc_0007 | Policy_Procedure_Contract | Internal_Communications | 0.217        | 0.035    |
| doc_0008 | Policy_Procedure_Contract | Internal_Communications | 0.222        | 0.038    |
| doc_0009 | Policy_Procedure_Contract | Needs_Review            | 0.233        | 0.029    |
| doc_0010 | Policy_Procedure_Contract | Forms_Structured        | 0.318        | 0.143    |
| doc_0011 | Policy_Procedure_Contract | Reports                 | 0.229        | 0.040    |
| doc_0012 | Policy_Procedure_Contract | nan                     | —            | —        |
| doc_0013 | Policy_Procedure_Contract | Reports                 | 0.276        | 0.038    |
| doc_0014 | Policy_Procedure_Contract | Needs_Review            | 0.219        | 0.012    |
| doc_0015 | Policy_Procedure_Contract | Emails                  | 0.371        | 0.209    |
| doc_0016 | Policy_Procedure_Contract | Needs_Review            | 0.250        | 0.025    |
| doc_0017 | Reports                   | Emails                  | 0.339        | 0.150    |
| doc_0018 | Reports                   | Needs_Review            | 0.240        | 0.014    |
| doc_0020 | Policy_Procedure_Contract | Forms_Structured        | 0.379        | 0.144    |
| doc_0021 | Policy_Procedure_Contract | Emails                  | 0.263        | 0.060    |
| doc_0024 | Policy_Procedure_Contract | Forms_Structured        | 0.310        | 0.094    |
| doc_0025 | Policy_Procedure_Contract | Forms_Structured        | 0.354        | 0.145    |
| doc_0026 | Policy_Procedure_Contract | Forms_Structured        | 0.280        | 0.060    |
| doc_0027 | Reports                   | Emails                  | 0.255        | 0.053    |
| doc_0028 | Reports                   | Forms_Structured        | 0.288        | 0.097    |
| doc_0029 | Policy_Procedure_Contract | Forms_Structured        | 0.265        | 0.078    |
| doc_0030 | Reports                   | Emails                  | 0.614        | 0.464    |
| doc_0031 | Policy_Procedure_Contract | Forms_Structured        | 0.317        | 0.125    |
| doc_0032 | Policy_Procedure_Contract | Forms_Structured        | 0.361        | 0.187    |
| doc_0033 | Policy_Procedure_Contract | Forms_Structured        | 0.235        | 0.052    |
| doc_0034 | Policy_Procedure_Contract | Forms_Structured        | 0.307        | 0.113    |
| doc_0035 | Reports                   | Emails                  | 0.298        | 0.098    |
| doc_0037 | Policy_Procedure_Contract | Forms_Structured        | 0.239        | 0.050    |
| doc_0038 | Policy_Procedure_Contract | Forms_Structured        | 0.264        | 0.032    |
| doc_0039 | Policy_Procedure_Contract | Forms_Structured        | 0.243        | 0.032    |
| doc_0040 | Policy_Procedure_Contract | Forms_Structured        | 0.358        | 0.167    |
| doc_0041 | Policy_Procedure_Contract | Forms_Structured        | 0.314        | 0.129    |
| doc_0042 | Policy_Procedure_Contract | Forms_Structured        | 0.317        | 0.137    |
| doc_0043 | Policy_Procedure_Contract | Internal_Communications | 0.276        | 0.036    |
| doc_0048 | Reports                   | Forms_Structured        | 0.271        | 0.080    |
| doc_0050 | Reports                   | Needs_Review            | 0.293        | 0.011    |
| doc_0054 | Reports                   | Forms_Structured        | 0.241        | 0.043    |
| doc_0060 | Reports                   | Internal_Communications | 0.249        | 0.048    |
| doc_0062 | Reports                   | Forms_Structured        | 0.272        | 0.057    |
| doc_0065 | Reports                   | Forms_Structured        | 0.314        | 0.111    |
| doc_0067 | Reports                   | Forms_Structured        | 0.273        | 0.075    |
| doc_0068 | Reports                   | Needs_Review            | 0.226        | 0.029    |
| doc_0073 | Reports                   | Forms_Structured        | 0.268        | 0.081    |
| doc_0075 | HR_Documents              | Emails                  | 0.843        | 0.785    |
| doc_0078 | Forms_Structured          | Needs_Review            | 0.250        | 0.010    |
| doc_0082 | Forms_Structured          | nan                     | —            | —        |
| doc_0083 | Forms_Structured          | nan                     | —            | —        |
| doc_0084 | Forms_Structured          | Emails                  | 0.278        | 0.049    |
| doc_0087 | Forms_Structured          | Needs_Review            | 0.240        | 0.001    |
| doc_0096 | Policy_Procedure_Contract | Needs_Review            | 0.264        | 0.010    |

