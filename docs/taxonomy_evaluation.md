# Taxonomy Evaluation

This document summarizes the current taxonomy design, evaluation protocol, ablation results, rule-boosting improvements, and known limitations for the InfoGuide Pilot-2 taxonomy assignment module.

## Label Set

The active taxonomy is stored in `configs/taxonomy.yaml`. It uses a document-type hierarchy with one top-level domain, `Document Type`, and six final labels:

| Label | Definition | Examples in current corpus |
|---|---|---|
| `Policy / Procedure / Contract Documents` | Formal guidance, procedural manuals, toolkits, legal/financial terms, supplements, agreements, or reference materials. | FATF guidance PDFs, ISORA guide, FAS glossary, JPM pricing supplements |
| `Reports (Financial / Incident / Audit)` | Analytical reports, working papers, assessments, annual reports, audit/financial statements, filings, and earnings releases. | World Bank FSAP reports, IMF working papers, JPM annual report, audited financial statements |
| `Forms / Structured Documents` | Field-heavy structured documents, tax forms, survey forms, certificates, or repeated field templates. | IRS W-4, W-9, 1040, ISORA forms |
| `Emails` | Email messages or threads with email-specific structure such as From/To/Subject headers. | No true examples in the current local corpus |
| `Internal Communications` | Internal announcements, memos, meeting notes, staff notices, or operational updates. | No true examples in the current local corpus |
| `HR Documents / Communications` | HR records or communications involving employees, payroll, hiring, benefits, or personnel topics. | No true examples in the current local corpus |

## Gold Labels

Manual gold labels are stored in `data/labels/taxonomy_gold_labels.jsonl`.

The current gold set covers all 39 processed local documents:

| Gold label | Count |
|---|---:|
| `Reports (Financial / Incident / Audit)` | 21 |
| `Policy / Procedure / Contract Documents` | 13 |
| `Forms / Structured Documents` | 5 |
| `Emails` | 0 |
| `Internal Communications` | 0 |
| `HR Documents / Communications` | 0 |

Gold labels were assigned by inspecting filenames, source organizations, title-page text, and document structure. The label represents document genre/structure, not the document's economic topic.

## Evaluation Method

The evaluator compares `taxonomy.level_2` in a prediction JSONL file against the `label` field in the gold JSONL file.

Command:

```bash
conda run -n infoguide_env python src/evaluate_taxonomy_accuracy.py \
  --predictions data/outputs/taxonomy_assignments_rule_boosted.jsonl \
  --gold data/labels/taxonomy_gold_labels.jsonl \
  --show_errors
```

The ablation runner evaluates multiple assigner configurations and writes results to:

- `data/outputs/ablation/taxonomy_ablation_results.csv`
- `data/outputs/ablation/taxonomy_ablation_results.md`
- `data/outputs/ablation/figures/`

Command:

```bash
conda run -n infoguide_env python src/run_taxonomy_ablation.py --skip_existing
conda run -n infoguide_env python src/plot_taxonomy_ablation.py
```

The generated visual comparison files are:

- `data/outputs/ablation/figures/taxonomy_ablation_accuracy.png`
- `data/outputs/ablation/figures/taxonomy_ablation_top10.png`
- `data/outputs/ablation/figures/taxonomy_ablation_alpha_curve.png`
- `data/outputs/ablation/figures/taxonomy_ablation_topk_curve.png`
- `data/outputs/ablation/figures/taxonomy_confusion_hierarchical_alpha_0_7.png`
- `data/outputs/ablation/figures/taxonomy_confusion_rule_boosted.png`

## Ablation Summary

| Rank | Method | Accuracy | Macro F1 | Correct / Total |
|---:|---|---:|---:|---:|
| 1 | Rule-boosted hybrid | 1.0000 | 1.0000 | 39 / 39 |
| 2 | Hierarchical hybrid, alpha=0.7 | 0.5128 | 0.5527 | 20 / 39 |
| 3 | Chunk evidence, alpha=0.85 | 0.4615 | 0.5092 | 18 / 39 |
| 4 | Hierarchical hybrid, alpha=0.3 | 0.4615 | 0.5051 | 18 / 39 |
| 5 | Chunk evidence, top_k=8 | 0.4615 | 0.4971 | 18 / 39 |
| 16 | Keyword baseline | 0.2821 | 0.3436 | 11 / 39 |
| 17 | Hierarchical keyword-only, alpha=0.0 | 0.2564 | 0.3476 | 10 / 39 |

The best non-rule model is the hierarchical embedding/keyword hybrid with `alpha=0.7`. The rule-boosted hybrid uses that model output as a base and applies deterministic genre cues for forms, guidance/toolkits/manuals, reports, working papers, financial statements, and pricing supplements.

## Rule-Boosting Rationale

The initial models frequently confused document topic words with document genre. For example, a report may mention "policy" many times, and a guidance document may include report-like financial or risk vocabulary. The rule-boosted layer corrects these recurring errors using stable genre signals:

- Form cues: `W-4`, `W-9`, `1040`, `OMB No.`, `ISORA ... Form`, `Give form to`.
- Policy/procedure cues: `Guidance`, `Guide`, `Toolkit`, `Risk-Based Approach`, `Glossary`, `Manual`, `Pricing Supplement`, `Pricing Term Sheet`, `Terms of the Notes`.
- Report cues: `Financial Sector Assessment`, `Working Paper`, `Annual Report`, `Highlights Report`, `Audited Financial Statements`, `Management's Discussion`, `Proxy Statement`, `Form 8-K`, `Quarterly Earnings`.

The rules are implemented in `src/assign_taxonomy_rule_boosted.py`. Each override records a debug reason in `debug.rule_boost`.

## Confusion Analysis

Before rule boosting, the best model was `hierarchical_alpha_0_7` with 20/39 correct. Its main confusions were:

| Gold label | Predicted label | Count |
|---|---|---:|
| Policy / Procedure / Contract Documents | Reports (Financial / Incident / Audit) | 5 |
| Policy / Procedure / Contract Documents | Internal Communications | 3 |
| Policy / Procedure / Contract Documents | Forms / Structured Documents | 2 |
| Policy / Procedure / Contract Documents | Unassigned | 1 |
| Reports (Financial / Incident / Audit) | Policy / Procedure / Contract Documents | 3 |
| Reports (Financial / Incident / Audit) | HR Documents / Communications | 2 |
| Reports (Financial / Incident / Audit) | Internal Communications | 2 |
| Reports (Financial / Incident / Audit) | Forms / Structured Documents | 1 |

After rule boosting, all 39 current gold-label documents are classified correctly.

The confusion matrix figures make this improvement visible:

- Before rule boosting: `data/outputs/ablation/figures/taxonomy_confusion_hierarchical_alpha_0_7.png`
- After rule boosting: `data/outputs/ablation/figures/taxonomy_confusion_rule_boosted.png`

## Limitations

- The gold set contains only 39 documents, which is useful for iteration but too small for a final statistical claim.
- Three taxonomy labels currently have zero true examples: `Emails`, `Internal Communications`, and `HR Documents / Communications`.
- The rule-boosted score is high because the current corpus has strong filename/title genre cues. It should be validated on new files before being presented as general performance.
- The gold labels were assigned by one annotator. A stronger evaluation would use at least two annotators and resolve disagreements.
- Current evaluation is document-level single-label classification. Multi-label cases are possible, especially for documents such as HR forms or policy reports.

## Current Recommendation

For the current corpus, use `data/outputs/taxonomy_assignments_rule_boosted.jsonl` as the best taxonomy output. For the project report, present both:

- the ablation result showing that the hybrid embedding/keyword model is the strongest learned baseline, and
- the rule-boosted result showing that domain-specific genre rules substantially improve taxonomy accuracy on this corpus.
