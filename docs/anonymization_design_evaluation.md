# Anonymization Module: Design and Evaluation

## 1. Purpose

The anonymization module detects sensitive entities in processed documents and replaces them with standardized placeholders. It is designed as a configurable preprocessing stage for the larger document pipeline, so downstream classification and taxonomy steps can operate on text with reduced exposure of personally identifiable or sensitive information.

The module focuses on eight entity categories:

| Category | Placeholder | Detection method |
|---|---:|---|
| PERSON | `[PERSON]` | NER |
| ORGANIZATION / ORG | `[ORG]` | NER |
| LOCATION | `[LOCATION]` | NER |
| EMAIL | `[EMAIL]` | Regex |
| PHONE | `[PHONE]` | Regex |
| DATE | `[DATE]` | Regex |
| IBAN | `[IBAN]` | Regex |
| CREDIT_CARD | `[CREDIT_CARD]` | Regex |

This scope separates contextual entities, which require language understanding, from structured identifiers, which can be detected reliably through format-based recognizers.

## 2. Design Overview

The anonymization module uses a hybrid detection architecture:

1. Regex recognizers detect structured PII such as email addresses, phone numbers, dates, IBANs, and credit card numbers.
2. NER recognizers detect contextual entities such as people, organizations, and locations.
3. A precedence policy resolves overlaps between recognizers.
4. Optional post-processing filters reduce false positives.
5. Final spans are replaced with placeholders in the anonymized output text.

The main implementation is in `src/anonymization/anonymizer.py`. Configuration is externalized in YAML files under `configs/anonymization/`, which keeps model choice, regex patterns, enabled entity types, placeholder mappings, thresholds, and output paths reproducible across experiments.

## 3. Hybrid Detection Strategy

### 3.1 Regex-Based Structured Detection

Regex detection is used for entity types whose surface form is relatively regular:

- `EMAIL`
- `PHONE`
- `DATE`
- `IBAN`
- `CREDIT_CARD`

This approach gives strong precision for clearly formatted identifiers, but it does not cover contextual entities and can still produce false positives when non-sensitive values look similar to PII. Examples include template IDs, reference numbers, record codes, and transaction-like strings.

### 3.2 NER-Based Contextual Detection

NER is used for contextual entity types:

- `PERSON`
- `ORGANIZATION` / `ORG`
- `LOCATION`

These entities depend more heavily on surrounding language. For example, a person name may appear with a title, as a partial name, in lowercase, or inside a sentence where fixed patterns are not enough. The project evaluates several contextual NER options:

- spaCy `en_core_web_sm`
- spaCy `en_core_web_lg`
- spaCy `en_core_web_trf`
- DeBERTa NER baseline
- GLiNER open-vocabulary NER baseline

The spaCy variants are evaluated inside the primary hybrid pipeline. DeBERTa is evaluated both as a contextual baseline and as a replacement NER component in a regex + DeBERTa hybrid. GLiNER is evaluated as an external open-vocabulary baseline.

## 4. Overlap and Precedence Policy

Hybrid systems can produce overlapping spans. The module applies a deterministic precedence rule:

- Regex spans are treated as high-confidence structured PII.
- When a regex span overlaps an NER span, the regex span is kept.
- The overlapping NER span is removed.
- Remaining spans are sorted by character offset before anonymization.

This prevents duplicate masking and avoids cases where a contextual NER prediction partially overrides a structured identifier. The same principle is used when combining regex predictions with DeBERTa predictions in `src/anonymization/combine_regex_deberta_hybrid.py`.

## 5. Post-Processing Filters

### 5.1 Generic Organization Filter

Organization detection is especially prone to false positives because many generic business phrases look like organization names. The module filters generic institutional unit names ending in terms such as:

- `division`
- `department`
- `team`
- `unit`
- `office`
- `board`
- `panel`
- `committee`

For example, phrases such as `Compliance Division` or `Internal Audit Committee` are treated as functional descriptions rather than sensitive organizations. This filter helps preserve specific organization masking while avoiding unnecessary anonymization of generic internal structures.

### 5.2 Regex Context Filter

Structured identifiers can also create false positives. The context filter in `src/anonymization/context_filter_regex_predictions.py` checks a configurable character window around each regex match. The current configuration uses a 50-character window.

The filter removes a regex match when negative context appears without positive support. For example:

| Entity type | Positive cues | Negative cues |
|---|---|---|
| PHONE | `phone`, `mobile`, `contact`, `tel`, `call` | `reference`, `code`, `ticket`, `case`, `template` |
| CREDIT_CARD | `card`, `credit card`, `payment`, `billing` | `reference`, `ticket`, `case`, `record`, `code` |
| IBAN | `iban`, `bank account`, `account`, `transfer`, `payment` | `reference`, `sample`, `example`, `template`, `code` |
| DATE | `date`, `effective`, `deadline`, `issued`, `signed` | `version`, `template`, `reference`, `record`, `code` |
| EMAIL | `email`, `contact`, `mail` | `example`, `sample`, `placeholder`, `test` |

This filter is intentionally conservative. It mainly targets obvious lookalike values while preserving true sensitive identifiers when the surrounding text supports them.

## 6. Synthetic Benchmark Dataset

The anonymization module is evaluated on a synthetic benchmark because real enterprise documents contain limited labeled sensitive information. The benchmark contains 66 documents across policies, reports, emails, internal communications, HR documents, and structured forms.

The dataset includes 745 labeled entities across the target PII categories and 80 negative cases designed to test lookalike non-PII patterns. The synthetic dataset is used only as a controlled benchmark for entity-level evaluation; it is not treated as representative production data.

## 7. Evaluation Methodology

The evaluator compares predicted entities against ground truth spans and reports precision, recall, and F1.

Two matching modes are used:

| Mode | Definition |
|---|---|
| Strict | Entity type, start offset, and end offset must match exactly. |
| Relaxed | Entity type must match and the predicted span must overlap the gold span. |

The evaluator also reports:

- per-entity-type metrics
- per-document-type metrics
- per-difficulty metrics
- per-variation metrics
- negative-case false positive rate
- error files containing false positives, false negatives, and boundary errors

Runtime metrics are tracked separately under `outputs/anonymizer_metrics/`, including average runtime, throughput, and document-level latency.

## 8. Evaluation Blocks

| Evaluation block | Configurations | Purpose |
|---|---|---|
| Component contribution | Regex-only, spaCy-sm NER-only, regex + spaCy-sm hybrid | Measures the separate and combined value of structured regex detection and contextual NER. |
| spaCy backbone comparison | Regex + spaCy-sm, regex + spaCy-lg, regex + spaCy-trf | Tests whether stronger spaCy NER backbones improve contextual detection. |
| Regex context filtering | Regex + spaCy-trf, regex + spaCy-trf + context filter | Measures whether local context reduces structured-PII false positives. |
| External hybrid replacement | Regex + spaCy-trf + context filter, regex + DeBERTa + context filter | Tests whether replacing spaCy with DeBERTa improves anonymization quality. |
| Open-vocabulary baseline | GLiNER | Evaluates a prompt-driven NER architecture as an alternative baseline. |

## 9. Overall Results

| Model | Strict precision | Strict recall | Strict F1 | Relaxed precision | Relaxed recall | Relaxed F1 | Negative FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regex-only | 0.8546 | 0.3235 | 0.4693 | 0.8794 | 0.3329 | 0.4830 | 0.1625 |
| spaCy-sm NER-only | 0.5066 | 0.3624 | 0.4225 | 0.6329 | 0.4282 | 0.5108 | 0.1375 |
| Regex + spaCy-sm | 0.6348 | 0.6859 | 0.6594 | 0.7297 | 0.7611 | 0.7451 | 0.3000 |
| Regex + spaCy-lg | 0.6691 | 0.7463 | 0.7056 | 0.7669 | 0.8215 | 0.7933 | 0.3375 |
| Regex + spaCy-trf | 0.7727 | 0.8443 | 0.8069 | 0.8808 | 0.9221 | 0.9010 | 0.2875 |
| Regex + spaCy-trf + context filter | 0.7793 | 0.8389 | 0.8080 | 0.8893 | 0.9168 | 0.9028 | 0.2125 |
| DeBERTa NER baseline | 0.8868 | 0.8737 | 0.8802 | 0.9473 | 0.9074 | 0.9269 | 0.0750 |
| Regex + DeBERTa + context filter | 0.8797 | 0.8738 | 0.8768 | 0.9310 | 0.9060 | 0.9184 | 0.1625 |
| GLiNER baseline | 0.6519 | 0.7691 | 0.7057 | 0.6884 | 0.7946 | 0.7377 | 0.2250 |

## 10. Interpretation

### 10.1 Regex and NER Are Complementary

Regex-only detection has high precision but low recall because it covers only structured identifiers. NER-only detection finds contextual entities but misses structured types and produces more false positives. The hybrid spaCy-sm model improves over both isolated components, confirming that the two approaches cover different parts of the anonymization problem.

### 10.2 Larger spaCy Backbones Improve Contextual Detection

The spaCy backbone comparison shows a steady improvement from `sm` to `lg` to `trf`. The regex-detected entity types remain mostly unchanged across these runs because the regex component is fixed. The gains come mainly from contextual entity types: `PERSON`, `ORG`, and `LOCATION`.

The transformer-based spaCy model gives the best spaCy-based quality:

- strict F1 improves from 0.6594 with spaCy-sm to 0.8069 with spaCy-trf
- relaxed F1 improves from 0.7451 with spaCy-sm to 0.9010 with spaCy-trf

This improvement comes with higher runtime cost.

| Model | Average runtime | Average latency |
|---|---:|---:|
| Regex + spaCy-sm | 4.06 s | 47.74 ms |
| Regex + spaCy-lg | 4.81 s | 51.95 ms |
| Regex + spaCy-trf | 12.53 s | 153.18 ms |

### 10.3 Context Filtering Reduces Structured False Positives

Adding the regex context filter to the spaCy-trf hybrid slightly improves precision and reduces negative-case false positives:

- strict precision increases from 0.7727 to 0.7793
- relaxed precision increases from 0.8808 to 0.8893
- negative-case false positive rate decreases from 0.2875 to 0.2125

The clearest per-type improvements are for `IBAN`, `PHONE`, and `CREDIT_CARD`. `IBAN` reaches perfect precision and recall after filtering in the current benchmark. The filter has a smaller effect on `EMAIL`, which is already handled by a precise regex pattern.

### 10.4 DeBERTa Improves Accuracy but Costs More Runtime

The regex + DeBERTa + context-filtered hybrid achieves the strongest F1 among the full hybrid systems:

- strict F1: 0.8768
- relaxed F1: 0.9184

Compared with regex + spaCy-trf + context filter, the DeBERTa hybrid has much higher precision and fewer false positives. However, DeBERTa is slower than the spaCy-trf hybrid backbone:

| Model | Average runtime | Average latency |
|---|---:|---:|
| Regex + spaCy-trf | 12.53 s | 153.18 ms |
| DeBERTa NER baseline | 23.71 s | 291.62 ms |

The context-filtered hybrid combination metrics are not used for runtime comparison because that stage measures only post-processing and combination, not full NER inference.

### 10.5 GLiNER Is Flexible but Not Best for This Setup

GLiNER is useful as an open-vocabulary NER baseline because it can detect prompted entity types without being restricted to a fixed model label set. In this benchmark, however, it produces more false positives and runs slower than the strongest alternatives:

- strict F1: 0.7057
- relaxed F1: 0.7377
- average runtime: 39.50 s
- average latency: 408.44 ms

For this module, GLiNER is better treated as an exploratory baseline than as the selected anonymization approach.

## 11. Final Model Selection

The selected pipeline configuration is:

**Regex + spaCy-trf + regex context filter**

This configuration is selected because it provides the best practical balance for the full pipeline:

- it is the strongest spaCy-based hybrid configuration
- it achieves high relaxed F1 at 0.9028
- it improves precision through context filtering
- it is substantially faster than the DeBERTa and GLiNER alternatives
- it keeps the implementation simple and configurable within the main anonymization pipeline

The regex + DeBERTa + context-filtered hybrid achieves higher F1, but it introduces a larger inference cost. It is therefore a strong accuracy-oriented alternative, while the spaCy-trf context-filtered hybrid is the preferred production-facing choice for the current pipeline.

## 12. Outputs

The anonymization module produces two main outputs:

| Output | Purpose |
|---|---|
| `data/anonymized/*_documents.jsonl` | Document-level anonymized text used by downstream pipeline stages. |
| `outputs/anonymizer_outputs/*_predictions.jsonl` | Entity predictions used for evaluation and error analysis. |

Evaluation artifacts are written under `outputs/anonymizer_evaluation/`, and runtime summaries are written under `outputs/anonymizer_metrics/`.

## 13. Limitations and Next Steps

The current system is effective for the defined PII categories, but several limitations remain:

- organization names remain difficult because the boundary between sensitive organizations and generic business units is sometimes ambiguous
- context filtering is rule-based and may miss subtler false positives
- regex patterns for structured identifiers can be further tuned for domain-specific formats
- DeBERTa improves quality but requires a runtime trade-off
- evaluation depends on synthetic benchmark coverage, so future validation on manually labeled real documents would strengthen confidence

Potential improvements include weighted context filtering, more entity-specific negative rules, calibration of NER thresholds, and a more formal comparison of accuracy versus runtime for production deployment.
