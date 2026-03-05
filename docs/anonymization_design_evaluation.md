# Anonymization Module Design

## Overview

- This document defines the design of the PII Detection and Anonymization module in InfoGuide Pilot-2
- The module detects and masks personally identifiable information (PII)
- Detected entities are replaced with typed placeholders such as [PERSON], [ORG], [EMAIL]

## Input

- The module takes input from the preprocessing stage
- Each document contains:
  - doc_id
  - metadata fields (source, url, etc.)
  - processed_text

## Output

### Anonymized Documents

- Each document is extended with:
  - anonymized_text
- The original processed_text is preserved

### PII Detection Logs

- Each detected entity is stored with:
  - doc_id
  - pii_type
  - start
  - end
  - score

## PII Scope

- PERSON
- ORG
- LOCATION
- EMAIL
- URL
- PHONE
- IP_ADDRESS
- DATE
- CREDIT_CARD
- IBAN

## Placeholder Schema

- PERSON → [PERSON]
- ORG → [ORG]
- LOCATION → [LOCATION]
- EMAIL → [EMAIL]
- URL → [URL]
- PHONE → [PHONE]
- IP_ADDRESS → [IP]
- DATE → [DATE]
- CREDIT_CARD → [CREDIT_CARD]
- IBAN → [IBAN]

## Detection Strategy

### Rule-based (Regex)

- EMAIL
- URL
- PHONE
- IP_ADDRESS
- IBAN
- CREDIT_CARD
- DATE

### NER-based (spaCy via Presidio)

- PERSON
- ORG
- LOCATION

### Orchestration (Presidio)

- Combines regex and NER detections into a unified pipeline
- Assigns confidence scores to detected entities
- Applies anonymization using defined placeholders

## Precedence and Overlap Handling

- Regex detections are applied first
- Regex spans are treated as high-confidence and frozen
- NER detections are applied afterward
- If overlap occurs: regex span is preserved

## Confidence Thresholding

- PERSON / ORG / LOCATION (NER-based): ≥ 0.50
- Regex-based detections (EMAIL, URL, PHONE, IP_ADDRESS, IBAN, CREDIT_CARD, DATE): no confidence threshold is applied
  - These are pattern-matched and treated as high-precision detections
- Only entities above threshold (for NER) are anonymized

## Language Assumption

- Documents are assumed to be in English

## Failure Handling

- If anonymization fails, the document remains unchanged
- anonymized_text = processed_text

## Configurability

- All parameters are configurable via YAML files (not hardcoded in Python)

- Configurable elements include:
  - PII entity types
  - placeholder mappings
  - confidence thresholds
  - regex patterns
  - enabled/disabled recognizers

- Motivation:
  - enables experimentation
  - supports comparison of configurations
  - enables ablation studies

## Evaluation Plan

The goal of this evaluation is to assess the accuracy of the anonymization module in detecting and replacing personally identifiable information (PII). The evaluation was conducted on a small dataset consisting of 11 synthetic documents created by generative ai, specifically for the task of testing the anonymization pipeline. Document id's for this task are: 0012, 0013...0022. Each document contains one or more instances of PII belonging to the entity types defined in the anonymization configuration.

For evalution an answer key was manually created for each document. The evaluation was performed manually using the following process:

- For each document, the processed text, anonymized output, and answer key were inspected.

- Each PII instance listed in the answer key was checked to determine whether:

  - it was correctly detected and anonymized

  - it was missed by the system

  - it was incorrectly labeled.

- The following outcomes were recorded:

  - True Positive (TP): A PII entity that exists in the answer key and was correctly detected and anonymized.

  - False Negative (FN): A PII entity present in the answer key that was not detected or anonymized.

  - False Positive (FP): A span that was anonymized by the system but does not correspond to a true PII instance.

- Counts were aggregated across all documents to compute evaluation metrics.

## Results Summary

### Document-Level Results

| Document | True Positives | False Negatives | False Positives |
|--------|--------|--------|--------|
| Document 1 | 6 | 2 | 0 |
| Document 2 | 5 | 2 | 1 |
| Document 3 | 5 | 3 | 0 |
| Document 4 | 4 | 2 | 1 |
| Document 5 | 3 | 3 | 1 |
| Document 6 | 4 | 1 | 0 |
| Document 7 | 3 | 2 | 0 |
| Document 8 | 2 | 2 | 0 |
| Document 9 | 2 | 1 | 0 |
| Document 10 | 2 | 2 | 0 |
| Document 11 | 6 | 2 | 0 |

---

### Aggregated Results

| Metric | Count |
|------|------|
| True Positives (TP) | 42 |
| False Negatives (FN) | 20 |
| False Positives (FP) | 3 |

---

### Precision

Precision measures the proportion of detected entities that are correct.

Precision = TP / (TP + FP)

Precision = 42 / (42 + 3) = **0.93**

A high precision indicates that the anonymization system rarely replaces non-PII text.

---

### Recall

Recall measures the proportion of true PII instances that were successfully detected.

Recall = TP / (TP + FN)

Recall = 42 / (42 + 20) = **0.68**

This indicates that while most detected entities are correct, the system fails to identify some true PII instances.

---

### F1 Score

The F1 score combines precision and recall into a single metric.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

F1 ≈ **0.78**

---

### Error Analysis

Most **false negatives** occurred in the following situations:

- organization names that were not recognized by the spaCy NER model  
- URLs that were not captured by the regex recognizers  
- secondary mentions of person names in free-text paragraphs  

False positives were relatively rare and typically occurred when numeric patterns were incorrectly matched as phone numbers or other identifiers.

Overall, the system demonstrates **high precision but moderate recall**, indicating that the anonymization approach is conservative: it avoids incorrect anonymizations but sometimes fails to detect all PII instances.

---

### Limitations

This evaluation has several limitations.

- The dataset size is small (11 documents), which limits the ability to generalize the results.
- The documents used for testing were synthetically generated rather than collected from real-world sources.
- Manual annotation and manual evaluation may introduce minor inconsistencies or human errors.

Future work will evaluate the anonymization system on larger and more diverse real-world corpora in order to better assess its performance in practical scenarios.