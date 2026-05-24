# Threshold Calibration — TF-IDF Cosine

- N = 94 documents
- Class distribution: Forms_Structured=20, Policy_Procedure_Contract=29, Reports=45
- Label space in config: Policy_Procedure_Contract, Reports, Internal_Communications, Emails, HR_Documents, Forms_Structured
- Classes with GT examples: Forms_Structured, Policy_Procedure_Contract, Reports

## Selected operating point (coverage floor = 0.9)

- min_score = **0.0**, min_margin = **0.01**
- coverage = 0.904 (85/94)
- accuracy on covered = 0.918
- macro F1 on covered = 0.929
- weighted F1 on covered = 0.929

## Pareto frontier (best macro F1 at each coverage floor)

| Coverage ≥ | min_score | min_margin | Coverage | Macro F1 | Accuracy | Needs_Review |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 0.000 | 0.000 | 1.000 | 0.916 | 0.904 | 0 |
| 0.95 | 0.000 | 0.000 | 1.000 | 0.916 | 0.904 | 0 |
| 0.90 | 0.000 | 0.010 | 0.904 | 0.929 | 0.918 | 9 |
| 0.80 | 0.000 | 0.010 | 0.904 | 0.929 | 0.918 | 9 |
| 0.70 | 0.000 | 0.020 | 0.766 | 0.961 | 0.958 | 22 |
| 0.50 | 0.000 | 0.030 | 0.606 | 0.982 | 0.982 | 37 |

## No-threshold baseline (always predict)

- accuracy = 0.904
- macro F1 = 0.916
- weighted F1 = 0.914

## Grid (macro F1 on covered docs)

| min_score \ min_margin | 0.00 | 0.01 | 0.02 | 0.03 | 0.05 | 0.07 | 0.10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 0.92 (0) | 0.93 (9) | 0.96 (22) | 0.98 (37) | 1.00 (55) | 1.00 (74) | 0.67 (82) |
| 0.010 | 0.92 (0) | 0.93 (9) | 0.96 (22) | 0.98 (37) | 1.00 (55) | 1.00 (74) | 0.67 (82) |
| 0.020 | 0.92 (0) | 0.93 (9) | 0.96 (22) | 0.98 (37) | 1.00 (55) | 1.00 (74) | 0.67 (82) |
| 0.030 | 0.91 (3) | 0.93 (10) | 0.96 (22) | 0.98 (37) | 1.00 (55) | 1.00 (74) | 0.67 (82) |
| 0.050 | 0.92 (26) | 0.92 (26) | 0.95 (34) | 0.98 (40) | 1.00 (55) | 1.00 (74) | 0.67 (82) |
| 0.075 | 0.95 (59) | 0.95 (59) | 0.95 (59) | 1.00 (61) | 1.00 (65) | 1.00 (74) | 0.67 (82) |
| 0.100 | 1.00 (75) | 1.00 (75) | 1.00 (75) | 1.00 (75) | 1.00 (76) | 1.00 (76) | 0.67 (82) |
| 0.125 | 0.67 (80) | 0.67 (80) | 0.67 (80) | 0.67 (80) | 0.67 (80) | 0.67 (80) | 0.67 (83) |
| 0.150 | 0.67 (84) | 0.67 (84) | 0.67 (84) | 0.67 (84) | 0.67 (84) | 0.67 (84) | 0.67 (84) |
| 0.200 | 0.67 (88) | 0.67 (88) | 0.67 (88) | 0.67 (88) | 0.67 (88) | 0.67 (88) | 0.67 (88) |
| 0.250 | 0.33 (93) | 0.33 (93) | 0.33 (93) | 0.33 (93) | 0.33 (93) | 0.33 (93) | 0.33 (93) |

Cell format: `macro_f1 (n_needs_review)`
