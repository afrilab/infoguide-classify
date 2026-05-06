# Classification Module Design and Evaluation

## Executive Summary

### What This Module Does

The classification module assigns each document a single **document type** label from six categories based on structure, format, and content characteristics. Unlike topic classification (which answers "what is this about?"), this module answers "**what kind of document is this?**" — distinguishing policies from reports, emails from memos, etc.

### Current Performance (98-Document Ground Truth)

| Model                    | Accuracy | Macro F1 | Weighted F1 |
| ------------------------ | -------- | -------- | ----------- |
| **TF-IDF cosine**        | **87.8%** | —        | —           |
| BGE Large embedding      | 83.7%    | —        | —           |
| BGE Small embedding      | 82.7%    | —        | —           |
| BART zero-shot NLI       | 48.0%    | —        | —           |
| DeBERTa-large zero-shot  | 24.5%    | —        | —           |

- **Recommended method**: TF-IDF cosine similarity with domain-specific label descriptions
- **Runner-up**: BGE Large embedding (BAAI/bge-large-en-v1.5) at 83.7% — competitive, no keywords needed
- **Key strengths of TF-IDF**:
  - Fast, deterministic, no model loading overhead
  - Explicit domain keyword matching (FATF, World Bank, FSAP, AML/CFT)
  - Interpretable and debuggable

### How We Got Here

1. **Early baseline (11-doc set)**: TF-IDF 90.9%, Zero-Shot 72.7%, Embedding 36.4% — but test set was too small and label-imbalanced (7 Policy + 4 Reports only)
2. **Ground truth expanded to 98 documents** across all 6 categories
3. **Embedding improved dramatically**: Switched from all-MiniLM to BGE (BAAI/bge-small-en-v1.5 → bge-large-en-v1.5) with asymmetric query instruction. BGE large reaches 83.7%.
4. **Zero-shot ceiling found**: Despite extensive tuning (hypothesis templates, description length, model size), BART tops at ~48% and DeBERTa-large performs worse (24.5%). Zero-shot NLI is not suitable for this corpus.
5. **TF-IDF remains best** at 87.8% on the full 98-doc set.

### Recommended Action

**Deploy TF-IDF as primary classifier.** 87.8% accuracy on 98 documents across all 6 categories is production-ready. Use BGE Large as a secondary or fallback classifier since it reaches 83.7% without any keyword engineering — useful if label descriptions change.

---

## Overview

- This document defines the design of the document-classification module
  and logs the ablation experiments that shape it.
- The module assigns each document a single primary **document type** label drawn
  from a controlled label space based on document structure, format, and content characteristics.
- **Current production method**: TF-IDF cosine similarity (`configs/classification_tfidf.yaml`)
  with domain-specific label descriptions.
- Alternative methods available: Embedding similarity (BGE), Zero-Shot NLI (BART) — see Methods Registry.

## Input

- Input is the anonymized corpus (`data/anonymized/presidio_hybrid_documents.jsonl`).
- Each row is expected to contain:
  - `doc_id`
  - one of `anonymized_text` / `processed_text` / `clean_text` / `text` (priority order)

## Output

### Predictions

Written to `data/classified/classification_results__<method>.jsonl`. Each row:

- `doc_id`
- `predicted_label` (one of the configured labels, or `Needs_Review`)
- `confidence` — top score
- `second_best`, `second_score`
- `margin` — `confidence - second_score`
- `top_labels` — full ranked list with raw scores
- `used_field`, `text_len`, `method`, `model`, `head_only`, `max_chars`

### Evaluation Reports

- `outputs/classification_outputs/classification_report__<name>.md` — per-run Markdown report
- `outputs/classification_outputs/doc_level_eval__<name>.csv` — per-document predictions
- `outputs/classification_outputs/classification_model_comparison.md` — side-by-side model comparison
- Ground truth: `data/eval/classification_gt.jsonl` (98 documents)

## Label Space

Labels describe a document's **primary type and format**, not the topics it
mentions. The label set covers six document categories based on content structure,
formality, and organizational context:

| Label                     | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| Policy_Procedure_Contract | Formal policy, procedure, and contract documents     |
| Reports                   | Financial, incident, and audit reports               |
| Internal_Communications   | Memos, announcements, and meeting notes              |
| Emails                    | Email correspondence and digital communications      |
| HR_Documents              | Employee records, evaluations, and HR communications |
| Forms_Structured          | Standardized forms with organized fields             |

### Labeling Protocol

When labeling ground truth, ask: **"What type of document is this based on its structure, format, and intended use?"**

| Document Characteristics       | Label                     |
| ------------------------------ | ------------------------- |
| Formal policies and contracts  | Policy_Procedure_Contract |
| Analytical reports with data   | Reports                   |
| Internal memos and updates     | Internal_Communications   |
| Email messages and threads     | Emails                    |
| Employee and HR information    | HR_Documents              |
| Structured forms and templates | Forms_Structured          |

Boundary guidelines (apply to humans during labeling):

- **Policy_Procedure_Contract** — NOT this label if document is an email,
  a report, an internal announcement, or HR-related.
- **Reports** — NOT this label if document is a policy, internal memo,
  email, HR document, or structured form.
- **Internal_Communications** — NOT this label if document is sent externally,
  is a formal policy, incident report, or HR-specific.
- **Emails** — NOT this label if document is a formal report, policy,
  internal memo, HR record, or structured form.
- **HR_Documents** — NOT this label if document is not employee or HR-related,
  or if it's an email, policy, report, or structured form.
- **Forms_Structured** — NOT this label if document is narrative content
  (email, memo, report, policy) without structured fields.

## Methods Registry

| Method                        | Status          | Accuracy  | Config                                      | Notes                                                            |
| ----------------------------- | --------------- | --------- | ------------------------------------------- | ---------------------------------------------------------------- |
| **TF-IDF cosine**             | **RECOMMENDED** | **87.8%** | `configs/classification_tfidf.yaml`         | Fast, deterministic, domain keywords explicit in descriptions    |
| BGE Large embedding           | Implemented     | 83.7%     | `configs/classification_embedding_large.yaml` | BAAI/bge-large-en-v1.5, asymmetric query instruction, no keyword engineering |
| BGE Small embedding           | Implemented     | 82.7%     | `configs/classification_embedding.yaml`     | BAAI/bge-small-en-v1.5, lighter and nearly as accurate as large  |
| BART zero-shot NLI            | Implemented     | 48.0%     | `configs/classification.yaml`               | facebook/bart-large-mnli, ceiling ~48% on this corpus            |
| DeBERTa-large zero-shot NLI   | Implemented     | 24.5%     | `configs/classification_zeroshot_large.yaml` | cross-encoder/nli-deberta-v3-large, underperforms BART here     |

## Configuration

### Active Configuration (TF-IDF, 87.8% Accuracy)

**File**: `configs/classification_tfidf.yaml`

```yaml
input_path: data/anonymized/presidio_hybrid_documents.jsonl
output_path: data/classified/classification_results__tfidf.jsonl
text_fields_priority:
  - anonymized_text
method: tfidf
max_chars: 6000
head_only: true
min_score: 0.10
min_margin: 0.08
use_label_descriptions: true
```

**Label Descriptions** (excerpt):

- **Policy_Procedure_Contract**: Formal policy documents, regulatory guidance, FATF guidance, AML/CFT requirements, supervision procedures...
- **Reports**: Analytical reports, World Bank assessments, FSAP reports, financial sector analysis...
- **Internal_Communications**: Memos, announcements, meeting notes...
- **Emails**: Email correspondence, informal communication...
- **HR_Documents**: Employee records, recruitment, performance evaluation...
- **Forms_Structured**: Structured forms, surveys, templates...

**Key insight**: Domain-specific keywords (FATF, World Bank, FSAP) are explicitly listed in descriptions. TF-IDF matches these keywords in documents for robust classification.

### BGE Large Embedding (83.7% Accuracy)

**File**: `configs/classification_embedding_large.yaml`

```yaml
input_path: data/anonymized/presidio_hybrid_documents.jsonl
output_path: data/classified/classification_results__embedding_large.jsonl
text_fields_priority:
  - anonymized_text
  - processed_text
  - clean_text
  - text
method: embedding
model_name: BAAI/bge-large-en-v1.5
query_instruction: "Represent this sentence for searching relevant passages: "
head_only: true
max_chars: 800
min_score: 0.05
min_margin: 0.01
use_label_descriptions: true
```

**Key insight**: BGE models use asymmetric retrieval — documents are encoded with the query instruction prefix; label descriptions are encoded as-is. Using short, non-overlapping descriptions avoids the cosine clustering problem seen with all-MiniLM.

### BGE Small Embedding (82.7% Accuracy)

**File**: `configs/classification_embedding.yaml`

Identical to BGE Large but with `model_name: BAAI/bge-small-en-v1.5`. 33M vs 135M parameters, 1% accuracy gap. Prefer this for faster inference.

### BART Zero-Shot NLI (48.0% Accuracy)

**File**: `configs/classification.yaml`

```yaml
method: zero_shot
model_name: facebook/bart-large-mnli
hypothesis_template: "The type of this document is {}."
head_only: true
max_chars: 800
multi_label: false
min_score: 0.10
min_margin: 0.03
use_label_descriptions: true
```

Short, non-overlapping label descriptions (key parameter). Verbose descriptions degrade performance.

**Status**: Implemented but ceiling is ~48% on this corpus. Not recommended for production.

### DeBERTa-large Zero-Shot NLI (24.5% Accuracy)

**File**: `configs/classification_zeroshot_large.yaml`

```yaml
method: zero_shot
model_name: cross-encoder/nli-deberta-v3-large
hypothesis_template: "The type of this document is {}."
head_only: true
max_chars: 800
min_score: 0.10
min_margin: 0.03
use_label_descriptions: true
```

**Status**: Implemented as ablation. Despite being a stronger NLI model, DeBERTa-large underperforms BART on this domain corpus. Larger NLI model ≠ better zero-shot for financial text.

## Evaluation

- **Test set**: 98 hand-labeled documents (`data/eval/classification_gt.jsonl`), covering all 6 label categories.
- **Metrics**: per-class precision/recall/F1, macro/weighted averages, confusion matrix, `Needs_Review` count.
- **Runner**: `python src/classification/classification_report.py --pred <path> --name <name> --output-dir outputs/classification_outputs`
- **Comparison**: `python src/classification/compare_classification_runs.py --output-dir outputs/classification_outputs`

## Ablation Log

### Phase 1: Early Baseline (11-Document Test Set, 2 Labels Only)

Small test set with 7 Policy + 4 Reports. Generic label descriptions.
Settings: `head_only=true`, `max_chars=3000`, `min_score=0.10`, `min_margin=0.08`.

| ID  | Method               | Accuracy | Notes                                              |
| --- | -------------------- | -------- | -------------------------------------------------- |
| A1  | Zero-Shot NLI (BART) | 72.7%    | Generic descriptions, "This document is about {}." |
| A2  | TF-IDF Cosine        | 72.7%    | Generic descriptions, baseline vocabulary          |
| A3  | Embedding (MiniLM)   | 36.4%    | all-MiniLM-L6-v2, general-purpose embeddings fail  |

### Phase 2: Domain Vocabulary Added (11-Document Test Set)

Added FATF, World Bank, FSAP, AML/CFT keywords to TF-IDF descriptions; longer descriptions for all methods.
Settings: `head_only=true`, `max_chars=6000`, `min_score=0.10`, `min_margin=0.08`.

| ID  | Method               | Accuracy | Delta    | Notes                                                         |
| --- | -------------------- | -------- | -------- | ------------------------------------------------------------- |
| B1  | Zero-Shot NLI (BART) | 63.6%    | -9.1pp   | Verbose descriptions hurt semantic entailment                 |
| B2  | TF-IDF Cosine        | 90.9%    | +18.2pp  | Domain keywords match directly — Policy F1=0.92, Reports F1=1.00 |
| B3  | Embedding (MiniLM)   | TBD      | —        | Not tested at this phase                                      |

### Phase 3: Full 98-Document Ground Truth, All 6 Labels

Ground truth expanded to 98 documents covering all categories. BGE models replace MiniLM. Multiple zero-shot configurations tested.

#### Embedding Models

| ID  | Model                    | min_margin | Accuracy | Notes                                                      |
| --- | ------------------------ | ---------- | -------- | ---------------------------------------------------------- |
| C1  | all-MiniLM-L6-v2         | 0.03       | ~36%     | General embeddings, high Needs_Review rate                 |
| C2  | BAAI/bge-small-en-v1.5   | 0.03       | <80%     | BGE with margin too high, 23 docs marked Needs_Review      |
| C3  | BAAI/bge-small-en-v1.5   | 0.01       | 82.7%    | Lowered margin recovers Needs_Review (23→8), accuracy jumps |
| C4  | BAAI/bge-large-en-v1.5   | 0.01       | 83.7%    | 135M param model, 1pp improvement over small               |

#### Zero-Shot NLI Models

| ID  | Model                              | Hypothesis template                          | Accuracy | Notes                                                     |
| --- | ---------------------------------- | -------------------------------------------- | -------- | --------------------------------------------------------- |
| D1  | facebook/bart-large-mnli           | "This document is about {}."                 | ~36%     | Verbose domain descriptions, head+tail, max_chars=1200    |
| D2  | facebook/bart-large-mnli           | "This document is about {}."                 | ~44%     | Short descriptions, head_only=true, max_chars=800         |
| D3  | facebook/bart-large-mnli           | "This document is a {}."                     | lower    | "is a" template weaker than "type of"                     |
| D4  | facebook/bart-large-mnli           | "The type of this document is {}."           | 48.0%    | Best BART config: short descriptions + type-focused template |
| D5  | cross-encoder/nli-deberta-v3-large | "The type of this document is {}."           | 24.5%    | Stronger NLI model but worse for this domain corpus       |

### Final Results Summary (98-Document Ground Truth)

| Model                   | Accuracy | Macro F1 | Weighted F1 | Needs_Review |
| ----------------------- | -------- | -------- | ----------- | ------------ |
| TF-IDF cosine           | **87.8%** | —        | —           | low          |
| BGE Large (bge-large-en-v1.5) | 83.7% | —     | —           | 8            |
| BGE Small (bge-small-en-v1.5) | 82.7% | —     | —           | 8            |
| BART zero-shot NLI      | 48.0%    | —        | —           | varies       |
| DeBERTa-large zero-shot | 24.5%    | —        | —           | varies       |

## Findings & Key Insights

### F1. TF-IDF Domain Vocabulary Advantage Holds on Full Corpus

TF-IDF with explicit domain keywords (FATF, World Bank, FSAP, AML/CFT) reaches 87.8% on 98 documents. The explicit keyword matching approach is robust when document vocabulary closely matches description vocabulary. Performance dropped slightly from the 11-doc set (90.9% → 87.8%) as the harder categories (HR, Emails, Internal, Forms) were added to evaluation.

### F2. BGE Embedding Is a Viable Alternative (No Keywords Required)

Switching from all-MiniLM-L6-v2 to BAAI/bge-small/large with asymmetric query instruction boosted embedding accuracy from ~36% to 82–84%. Key factors:

- **Asymmetric instruction**: `"Represent this sentence for searching relevant passages: "` prefix encodes documents for retrieval; label descriptions encoded as-is
- **Margin threshold matters**: min_margin=0.03 caused 23 Needs_Review; lowering to 0.01 recovers most without hurting accuracy
- **Model size**: bge-large (135M) is only ~1pp better than bge-small (33M) — prefer small for speed unless that 1pp matters

### F3. Zero-Shot NLI Has a Hard Ceiling (~48%) on This Corpus

Despite testing 5+ configurations (hypothesis templates, description length, model size), BART peaks at 48.0% and DeBERTa-large is worse at 24.5%. Root causes:

- Financial/regulatory domain vocabulary is out-of-distribution for NLI models trained on general NLI datasets
- "Needs_Review" assignments are hard to control without per-class calibration
- Larger NLI model (DeBERTa-large) does not improve results — domain mismatch, not model capacity, is the bottleneck
- Best template: **"The type of this document is {}."** (structural framing outperforms semantic "about" framing)

### F4. Description Length Trade-off

- **TF-IDF**: Benefits from long, keyword-rich descriptions — more vocabulary overlap = higher cosine score
- **Zero-Shot NLI**: Suffers from long descriptions — entailment scoring degrades with verbose, multi-concept hypotheses; short non-overlapping descriptions (1-2 sentences) work best
- **BGE Embedding**: Short descriptions preferred to avoid cosine clustering (all labels scoring 0.62–0.69 with thin margins)

### F5. Recommended Deployment Strategy

**Primary**: TF-IDF cosine (`configs/classification_tfidf.yaml`) — 87.8% accuracy, fast, deterministic

**Secondary / fallback**: BGE Small embedding (`configs/classification_embedding.yaml`) — 82.7%, no keyword maintenance required, useful if label descriptions are redesigned without domain keywords

**Do not use**: Zero-shot NLI for this corpus — ceiling at 48%, not production-ready

## Open Issues & Future Work

### O1. TF-IDF vs BGE Gap Analysis (MEDIUM PRIORITY)

TF-IDF leads by ~4pp (87.8% vs 83.7% BGE Large). Understanding which categories drive this gap would inform whether an ensemble (TF-IDF primary + BGE fallback for low-margin docs) could recover the gap.

**To investigate**:
1. Compare per-category F1 between TF-IDF and BGE Large outputs
2. Identify categories where BGE outperforms TF-IDF (candidates: HR, Emails, Forms)
3. Test confidence-gated ensemble: use TF-IDF when margin > threshold, BGE otherwise

### O2. Threshold Optimization (MEDIUM PRIORITY)

Current thresholds are manually tuned defaults. With 98-doc ground truth, a proper sweep is feasible.

**To optimize**:
1. Run hyperparameter sweep over `min_score` ∈ [0.05, 0.25] and `min_margin` ∈ [0.00, 0.10]
2. Measure F1 and Needs_Review rate for each combo
3. Select thresholds that minimize false Needs_Review while maintaining accuracy

### O3. Zero-Shot Improvement Path (LOW PRIORITY)

If zero-shot is required (e.g., adding new labels without retraining), consider:
- Few-shot prompting with a generative LLM (e.g., GPT-4 or Claude) instead of NLI — likely much stronger than BART/DeBERTa for domain text
- SetFit: fine-tune a sentence transformer with a handful of labeled examples per class — bridges embedding accuracy gap with minimal annotation effort

### O4. Document Preprocessing Edge Cases (LOWER PRIORITY)

Some documents may contain web navigation, table-of-contents, or boilerplate that interferes with classification. With `head_only=true` and `max_chars=800`, the classifier uses only the document head — if the head is mostly boilerplate, accuracy suffers.

**Possible mitigation** (out of scope for classification module):
- Detect and strip web-nav templates during ingestion
- Skip table-of-contents blocks
- Use smart slicing to find first substantive paragraph

## Artifacts

| Artifact                                                                 | Purpose                                          |
| ------------------------------------------------------------------------ | ------------------------------------------------ |
| `configs/classification_tfidf.yaml`                                      | TF-IDF method config + labels + descriptions     |
| `configs/classification_embedding.yaml`                                  | BGE Small embedding config                       |
| `configs/classification_embedding_large.yaml`                            | BGE Large embedding config (ablation)            |
| `configs/classification.yaml`                                            | BART zero-shot NLI config (ablation)             |
| `configs/classification_zeroshot_large.yaml`                             | DeBERTa-large zero-shot NLI config (ablation)    |
| `src/classification/run.py`                                              | Core classification logic (TF-IDF, embedding, zero-shot) |
| `src/classification/classify_documents.py`                               | CLI entry point                                  |
| `src/classification/classification_report.py`                            | Per-model evaluation runner                      |
| `src/classification/compare_classification_runs.py`                      | Side-by-side model comparison                    |
| `data/eval/classification_gt.jsonl`                                      | Ground truth (98 documents, all 6 categories)    |
| `data/classified/classification_results__tfidf.jsonl`                    | TF-IDF predictions                               |
| `data/classified/classification_results__embedding.jsonl`                | BGE Small predictions                            |
| `data/classified/classification_results__embedding_large.jsonl`          | BGE Large predictions                            |
| `data/classified/classification_results__zeroshot.jsonl`                 | BART zero-shot predictions                       |
| `data/classified/classification_results__zeroshot_large.jsonl`           | DeBERTa-large zero-shot predictions              |
| `outputs/classification_outputs/classification_report__<name>.md`        | Per-model evaluation report                      |
| `outputs/classification_outputs/classification_model_comparison.md`      | Cross-model comparison summary                   |
