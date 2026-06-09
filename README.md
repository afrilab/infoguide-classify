# InfoGuide Pilot-2

InfoGuide Pilot-2 is a modular document-intelligence pipeline for enterprise and banking documents. It takes raw documents in heterogeneous formats and turns them into structured outputs that can support search, retrieval, analytics, and later downstream tasks.

Large organizations often store policies, reports, forms, emails, HR documents, regulatory files, and operational documents in inconsistent formats. These collections are hard to use directly because the text may be noisy, the files may not share a common structure, and some documents may contain sensitive information. InfoGuide addresses this problem with a configuration-driven pipeline that extracts text, cleans it, anonymizes sensitive entities, assigns document-type labels, and maps documents into a banking-domain taxonomy.

The project is not designed around a single fixed model. It is an engineering prototype and evaluation framework. Each major stage can be configured, replaced, evaluated, and compared against alternatives under shared inputs and output formats.

## Pipeline Overview

The full workflow is:

```text
raw documents
  -> ingestion and text extraction
  -> preprocessing and normalization
  -> anonymization
  -> document-type classification
  -> banking taxonomy assignment
  -> evaluation artifacts and dashboard inspection
```

The pipeline uses JSONL files between stages so that each module can be inspected independently. YAML files under `configs/` control inputs, outputs, models, thresholds, taxonomy labels, and experiment variants.

## Data

The real corpus contains open-access financial, regulatory, and institutional documents from sources such as the World Bank, FATF, the IMF, the European Central Bank, BIS, the Federal Reserve, the IRS, and JPMorgan. The corpus is tracked through `docs/corpus_manifest.csv`, which records metadata such as `doc_id`, filename, file format, source, source URL, language, retrieval date, publication date, license, and simulated banking role.

The project also includes a synthetic benchmark dataset for anonymization evaluation. This dataset was created because real public documents do not provide reliable span-level labels for sensitive information. It contains enterprise-style policies, reports, internal communications, emails, HR documents, and structured forms with controlled PII categories, entity variants, difficulty levels, and negative lookalike examples.

## Repository Structure

```text
configs/      YAML configuration files for pipeline stages and experiments
data/         raw, synthetic, extracted, processed, anonymized, and classified data
demo/         Streamlit dashboard for inspecting precomputed artifacts
docs/         design notes, evaluation summaries, corpus metadata, and reports
outputs/      evaluation outputs, metrics, plots, and model comparison artifacts
src/          ingestion, preprocessing, anonymization, classification, taxonomy, and runner code
```

## Pipeline Stages

### 1. Ingestion and Text Extraction

The ingestion stage converts raw files into a common JSONL representation. It supports PDF, TXT, and HTML files in the current corpus configuration.

For each real document, ingestion checks the corpus manifest before processing. Files without a manifest entry are skipped so undocumented files do not enter the pipeline accidentally. PDF text is extracted with `pdfplumber`, TXT files are read directly, and HTML files are parsed with BeautifulSoup.

The output is `data/extracted/extracted_documents.jsonl`. Each row combines manifest metadata, extracted raw text, basic text statistics, and an extraction quality status.

Main files:

- `src/ingestion/document_ingestion.py`
- `configs/document_ingestion.yaml`
- `docs/corpus_manifest.csv`

### 2. Preprocessing and Normalization

The preprocessing stage cleans extracted text before downstream processing. It normalizes whitespace, removes common boilerplate patterns, removes repeated corpus-level headers and footers, computes quality statistics, and splits text into sentences with a lightweight spaCy sentencizer.

This stage is especially useful for long regulatory PDFs, where repeated page headers, footers, page numbers, and navigation text can distort document representation.

The output is `data/processed/clean_documents.jsonl`. Each row contains metadata, processed text, optional sentence lists, processing statistics, quality status, and debug information about removed content.

Main files:

- `src/preprocess/process_documents.py`
- `src/preprocess/chunk_documents.py`
- `configs/process.yaml`
- `configs/chunking.yaml`

### 3. Anonymization

The anonymization module detects sensitive entities in processed documents and replaces them with typed placeholders. It is a configurable preprocessing stage for the rest of the pipeline. It reduces exposure of sensitive values.

The module focuses on eight entity categories:

| Entity type | Placeholder | Main detection method |
| --- | --- | --- |
| `PERSON` | `[PERSON]` | NER |
| `ORGANIZATION` / `ORG` | `[ORG]` | NER |
| `LOCATION` | `[LOCATION]` | NER |
| `EMAIL` | `[EMAIL]` | Regex |
| `PHONE` | `[PHONE]` | Regex |
| `DATE` | `[DATE]` | Regex |
| `IBAN` | `[IBAN]` | Regex |
| `CREDIT_CARD` | `[CREDIT_CARD]` | Regex |

The design separates structured identifiers from contextual entities. Email addresses, phone numbers, dates, IBANs, and credit card numbers have regular surface forms, so regex recognizers give strong precision. Person names, organization names, and locations depend more on language context, so the module uses NER for those categories.

The main pipeline is hybrid:

- Regex recognizers detect structured entities such as `EMAIL`, `PHONE`, `DATE`, `IBAN`, and `CREDIT_CARD`.
- NER recognizers detect contextual entities such as `PERSON`, `ORGANIZATION`, and `LOCATION`.
- A deterministic overlap policy resolves conflicting spans.
- Post-processing filters reduce false positives.
- Final spans are replaced with typed placeholders in the anonymized text.

When regex and NER spans overlap, the regex span is kept and the overlapping NER span is removed. This avoids duplicate masking and prevents a contextual model from partially overriding a high-confidence structured identifier.

The module also uses two post-processing filters. The generic organization filter removes broad unit names such as `Compliance Division`, `Internal Audit Committee`, or other phrases ending in words like `department`, `unit`, `office`, `board`, `panel`, and `committee`. The regex context filter checks a 50-character window around structured matches. It removes lookalike values when negative cues such as `reference`, `template`, `record`, or `code` appear without positive support such as `phone`, `email`, `contact`, `payment`, or `IBAN`.

Several recognizer variants are evaluated:

| Evaluation block | Variants | Purpose |
| --- | --- | --- |
| Component contribution | Regex-only, spaCy-sm NER-only, regex + spaCy-sm | Measures how much regex and NER each contribute. |
| spaCy backbone comparison | spaCy-sm, spaCy-lg, spaCy-trf hybrids | Tests whether stronger contextual NER improves results. |
| Regex context filtering | spaCy-trf hybrid with and without context filtering | Tests whether local context reduces structured false positives. |
| External NER replacement | spaCy-trf hybrid vs. DeBERTa hybrid | Tests whether replacing spaCy improves quality. |
| Open-vocabulary baseline | GLiNER | Tests a prompt-driven NER baseline. |

The synthetic anonymization benchmark contains 66 documents, 745 labeled entities, and 80 negative cases. Evaluation is span-based. Strict matching requires entity type, start offset, and end offset to match exactly. Relaxed matching requires the entity type to match and the predicted span to overlap the gold span. The evaluator also reports per-type metrics, per-document-type metrics, per-difficulty metrics, negative-case false positive rate, and error files.

Summary results:

| Model | Strict F1 | Relaxed F1 | Negative FP rate |
| --- | ---: | ---: | ---: |
| Regex-only | 0.4693 | 0.4830 | 0.1625 |
| spaCy-sm NER-only | 0.4225 | 0.5108 | 0.1375 |
| Regex + spaCy-sm | 0.6594 | 0.7451 | 0.3000 |
| Regex + spaCy-lg | 0.7056 | 0.7933 | 0.3375 |
| Regex + spaCy-trf | 0.8069 | 0.9010 | 0.2875 |
| Regex + spaCy-trf + context filter | 0.8080 | 0.9028 | 0.2125 |
| Regex + DeBERTa + context filter | 0.8768 | 0.9184 | 0.1625 |
| GLiNER baseline | 0.7057 | 0.7377 | 0.2250 |

The selected practical configuration is `Regex + spaCy-trf + regex context filter`. It gives the best balance between quality, runtime, and integration simplicity among the spaCy-based systems. In the synthetic benchmark, it reached 0.8080 strict F1 and 0.9028 relaxed F1. A regex + DeBERTa hybrid achieved higher F1, but with a larger inference cost.

The main limitation is that evaluation depends on a synthetic benchmark. This makes controlled entity-level measurement possible, but it does not fully represent real enterprise privacy risk. Organization names also remain difficult because the boundary between sensitive organizations and generic business units can be ambiguous.

Main files:

- `src/anonymization/anonymizer.py`
- `src/anonymization/context_filter_regex_predictions.py`
- `src/anonymization/combine_regex_deberta_hybrid.py`
- `src/anonymization/evaluator.py`
- `configs/anonymization/`
- `configs/anonymization_evaluation/`
- `docs/anonymization_design_evaluation.md`

Primary outputs:

- `data/anonymized/*_documents.jsonl`
- `outputs/anonymizer_outputs/*_predictions.jsonl`
- `outputs/anonymizer_evaluation/`
- `outputs/anonymizer_metrics/`

### 4. Document Classification

The classification module assigns each document one primary document-type label. It answers "what kind of document is this?" rather than "what topic does this document discuss?"

The labels describe the primary type and format of a document:

| Label | Meaning |
| --- | --- |
| `Policy_Procedure_Contract` | Formal policy, procedure, and contract documents. |
| `Reports` | Financial, incident, institutional, and audit reports. |
| `Internal_Communications` | Memos, announcements, and meeting notes. |
| `Emails` | Email correspondence and digital communications. |
| `HR_Documents` | Employee records, evaluations, and HR communications. |
| `Forms_Structured` | Standardized forms with organized fields. |

The module is designed as a comparison framework rather than a single classifier. It evaluates simple lexical methods, embedding methods, zero-shot NLI models, supervised classical models, and a fine-tuned transformer under a shared label space.

The current main evaluation joins `data/eval/classification_gt.jsonl` with `data/processed/clean_documents.jsonl`. The original ground-truth file has 98 hand-labeled document IDs. Three entries do not have matching processed documents and are dropped. The `HR_Documents` class has one labeled example, while `Internal_Communications` and `Emails` have no usable examples. These classes cannot support stratified cross-validation, so the final CV dataset contains 94 documents across three classes: 45 `Reports`, 29 `Policy_Procedure_Contract`, and 20 `Forms_Structured`.

All compared methods use a shared text construction rule. The classifier takes the first available non-empty field from `anonymized_text`, `processed_text`, `clean_text`, or `text`. Text is truncated to the first 6000 characters. Macro F1 is the main comparison metric because the evaluated classes are moderately imbalanced.

Main method comparison:

| Method | Type | Accuracy | Macro F1 | Role |
| --- | --- | ---: | ---: | --- |
| Logistic Regression + TF-IDF | Supervised | 0.905 +/- 0.061 | 0.902 +/- 0.068 | Recommended primary method. |
| Random Forest + TF-IDF | Supervised | 0.894 +/- 0.088 | 0.899 +/- 0.084 | Competitive, but higher variance. |
| Linear SVM + TF-IDF | Supervised | 0.884 +/- 0.090 | 0.880 +/- 0.094 | Strong lexical baseline. |
| TF-IDF cosine with label descriptions | Unsupervised | 0.904 | 0.916 | Deterministic fallback. |
| Multinomial Naive Bayes + TF-IDF | Supervised | 0.809 +/- 0.024 | 0.787 +/- 0.020 | Stable but weaker. |
| Fine-tuned DistilBERT | Supervised transformer | 0.798 +/- 0.090 | 0.771 +/- 0.109 | Not recommended at this data size. |

Logistic Regression over TF-IDF features is the recommended primary method. It is fast, interpretable, cheap to retrain, and performs best among the supervised models. Random Forest is close, but its variance is higher. Linear SVM is also strong. Multinomial Naive Bayes is stable but weaker because the vocabulary overlap between policies and reports hurts its independence assumptions.

TF-IDF cosine similarity with label descriptions remains useful as a deterministic fallback. It does not need training data and performs competitively on the current subset. The fallback also supports a `Needs_Review` mechanism. Threshold calibration sweeps the top score and top-vs-second margin. The recommended operating point is `min_score = 0.0` and `min_margin = 0.01`, which gives 90.4% coverage and macro F1 of 0.929 on covered documents. This means 85 of 94 documents are auto-labeled and 9 are routed to review.

Fine-tuned DistilBERT was tested with the same stratified 5-fold setup. It underperformed the lexical baselines because the dataset is too small for transformer fine-tuning. Its weakest point is separating `Policy_Procedure_Contract` from `Reports`, where the model needs more examples to learn stable boundaries.

The main limitation is label coverage. Only three of the six configured classes have enough labeled examples for the reported CV experiment. The next major improvement is to expand ground truth for `Internal_Communications`, `Emails`, and `HR_Documents`, then re-run all methods on the full label space.

Main files:

- `src/classification/classify_documents.py`
- `src/classification/supervised_baselines.py`
- `src/classification/threshold_calibration.py`
- `src/classification/finetune_transformer.py`
- `src/classification/classification_report.py`
- `configs/classification_tfidf.yaml`
- `configs/classification_embedding.yaml`
- `configs/classification_embedding_large.yaml`
- `configs/classification_zeroshot_large.yaml`
- `docs/classification_design_evaluation.md`

Primary outputs:

- `data/classified/*.jsonl`
- `outputs/classification_outputs/`
- `outputs/classification_results/`

### 5. Banking Taxonomy Assignment

The taxonomy module adds a banking-domain organization layer on top of document-type classification. Classification identifies the document format, while taxonomy assignment identifies the banking domain, functional category, and specific topic.

The active taxonomy is stored in `configs/taxonomy.yaml` and has three levels:

- Level 1: banking domain
- Level 2: functional category
- Level 3: specific topic

Current Level 1 and Level 2 structure:

| Level 1 | Level 2 categories |
| --- | --- |
| `Governance & Policy` | `Internal Policies`; `Procedures & Guidelines` |
| `Risk & Compliance` | `AML / KYC`; `Audit & Monitoring`; `Incident & Fraud` |
| `Financial Operations` | `Reporting & Statements`; `Transactions & Processing` |
| `Customer & Accounts` | `Account Management`; `Customer Communication` |
| `Human Resources` | `Employee Management`; `Internal HR Communication` |
| `IT & Security` | `Access & Identity`; `Data Protection` |

Level 3 topics make the assignments specific enough to evaluate. Examples include `Customer Due Diligence`, `Financial Stability Assessment`, `Regulatory Reporting`, `Account Authorization`, `Workforce Analytics`, and `Data Quality and Integrity`.

Gold labels are stored in `outputs/taxonomy/labels/taxonomy_gold_labels.jsonl`. Each row contains a complete path with `level_1`, `level_2`, and `level_3`. The current label set covers all 123 processed documents, but it was produced by a single annotator. Scores should therefore be read as agreement with a manual single-annotator gold set, not as independently adjudicated performance.

Gold label distribution by Level 1:

| Level 1 | Documents |
| --- | ---: |
| `Financial Operations` | 52 |
| `Risk & Compliance` | 41 |
| `Customer & Accounts` | 16 |
| `IT & Security` | 7 |
| `Governance & Policy` | 6 |
| `Human Resources` | 1 |

The evaluation reports agreement separately for Level 1, Level 2, Level 3, and the full path. This avoids hiding partial success. A model may identify the correct broad banking domain while missing the more specific functional category or topic.

The project compares several assignment strategies:

| Method | Correct / Total | Path agreement | L1 agreement | L2 agreement | L3 macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-down TF-IDF + Logistic Regression | 68 / 123 | 0.5528 | 0.8211 | 0.7236 | 0.3141 |
| Flat TF-IDF + Linear SVM | 67 / 123 | 0.5447 | 0.8130 | 0.6829 | 0.3425 |
| Top-down TF-IDF + Linear SVM | 65 / 123 | 0.5285 | 0.8130 | 0.7154 | 0.2860 |
| Flat TF-IDF + Logistic Regression | 64 / 123 | 0.5203 | 0.7805 | 0.6748 | 0.3239 |
| Hierarchical embeddings, alpha=0.7 | 51 / 123 | 0.4146 | 0.6829 | 0.5285 | 0.3872 |
| Chunk evidence, top-1 local n-gram | 46 / 123 | 0.3740 | 0.5610 | 0.4959 | 0.4953 |
| Keyword baseline | 29 / 123 | 0.2358 | 0.5203 | 0.4390 | 0.2540 |

The strongest full-path baseline is top-down TF-IDF + Logistic Regression. It follows the hierarchy directly by predicting Level 1 first, then Level 2 inside that branch, then Level 3 inside the selected branch. This approach reaches 55.3% full-path agreement overall and 54.1% on the held-out test split.

Flat TF-IDF + Linear SVM is close behind at 54.5% full-path agreement. The semantic embedding baseline remains useful as a zero-shot or retrieval-style method, but it does not outperform supervised TF-IDF on this small manually labeled corpus. The removed rule-boosted variant is not reported because it produced circular 100% agreement by measuring deterministic consistency rather than generalization.

The main challenge is fine-grained topic assignment. The best method reaches 82.1% Level 1 agreement but 55.3% full-path agreement, which shows that broad banking domains are much easier than Level 2 and Level 3 distinctions. Low-support domains such as `Human Resources`, `Governance & Policy`, and `IT & Security` are also less reliable. Some documents naturally belong to more than one taxonomy path, but the current system is single-label.

Main files:

- `src/taxonomy/assign_taxonomy_embeddings.py`
- `src/taxonomy/assign_taxonomy_evidence.py`
- `src/taxonomy/assign_taxonomy_literature_baselines.py`
- `src/taxonomy/evaluate_taxonomy_accuracy.py`
- `src/taxonomy/run_taxonomy_ablation.py`
- `configs/taxonomy.yaml`
- `docs/taxonomy_evaluation.md`

Primary outputs:

- `outputs/taxonomy/taxonomy_assignments*.jsonl`
- `outputs/taxonomy/labels/taxonomy_gold_labels.jsonl`
- `outputs/taxonomy/ablation/`

### 6. Evaluation and Inspection

The repository includes evaluation scripts and precomputed artifacts for anonymization, classification, and taxonomy assignment. These outputs include JSON, JSONL, CSV, Markdown summaries, and plots.

The Streamlit dashboard in `demo/` reads these artifacts and lets users inspect pipeline outputs without rerunning the full pipeline. It supports document browsing, stage-by-stage pipeline inspection, classification evaluation review, anonymization evaluation review, and design report viewing.

Main files:

- `demo/app.py`
- `demo/pages/`
- `docs/*_design_evaluation.md`
- `outputs/`

You can access the demo interfcace from: https://npxmbjs2d2pfut5jpwbpna.streamlit.app/ 


## Running the Project

Run commands from the repository root. The original project was developed with a Conda environment named `infoguide_env`. If that environment exists, prefix commands with `conda run -n infoguide_env`.

### Run the configured pipeline

First inspect the commands:

```bash
python src/run_pipeline.py --config configs/pipeline.yaml --dry-run
```

Then run enabled steps:

```bash
python src/run_pipeline.py --config configs/pipeline.yaml
```

The enabled steps in `configs/pipeline.yaml` currently focus on ingestion, preprocessing, anonymization variants, and anonymization evaluation. Several classification and taxonomy experiments are present in the same file but disabled by default.

### Run individual stages

```bash
python src/ingestion/document_ingestion.py --config configs/document_ingestion.yaml
python src/preprocess/process_documents.py --config configs/process.yaml
python src/preprocess/chunk_documents.py --config configs/chunking.yaml
```

Run the selected anonymization variant:

```bash
python src/anonymization/anonymizer.py --config configs/anonymization/regex_spacy_hybrid_trf.yaml
python src/anonymization/context_filter_regex_predictions.py --config configs/anonymization/context_filter_hybrid_trf.yaml
```

Run document classification:

```bash
python src/classification/classify_documents.py --config configs/classification_tfidf.yaml
python src/classification/supervised_baselines.py
python src/classification/threshold_calibration.py
```

Run taxonomy assignment and evaluation:

```bash
python src/taxonomy/assign_taxonomy_literature_baselines.py \
  --input data/processed/clean_documents.jsonl \
  --labels outputs/taxonomy/labels/taxonomy_gold_labels.jsonl \
  --output outputs/taxonomy/ablation/topdown_tfidf_logreg_loo.jsonl \
  --method topdown_tfidf_logreg \
  --train_mode leave_one_out

python src/taxonomy/evaluate_taxonomy_accuracy.py \
  --predictions outputs/taxonomy/taxonomy_assignments_embeddings.jsonl \
  --labels outputs/taxonomy/labels/taxonomy_gold_labels.jsonl
```

### Run the dashboard

Install the lightweight dashboard dependencies if needed:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run demo/app.py
```

The dashboard reads precomputed artifacts from `data/`, `docs/`, and `outputs/`. It does not run the full pipeline.

## Key Evaluation Takeaways

- Hybrid anonymization is necessary because regex and NER cover different entity types.
- The selected anonymization setup is `Regex + spaCy-trf + regex context filter` because it balances accuracy and runtime.
- Document classification works well with TF-IDF features on the current corpus. Logistic Regression is the recommended primary supervised method.
- TF-IDF cosine classification remains a useful fallback when training data is limited or unavailable.
- Taxonomy assignment is easier at broad Level 1 domains than at full Level 1 > Level 2 > Level 3 paths.
- The strongest taxonomy baseline is top-down TF-IDF + Logistic Regression, but fine-grained topic assignment still needs more labeled data.

## Limitations and Future Work

The current corpus is useful for prototyping and controlled comparison, but it is still small and imbalanced for some labels. Classification evaluation currently has strong support for only three of the six configured document-type labels.

The anonymization benchmark is synthetic. It supports controlled span-level evaluation, but future validation on manually labeled real enterprise documents would strengthen confidence.

The taxonomy gold labels were produced by a single annotator. A second independent review would make the evaluation stronger and would allow inter-annotator agreement to be measured.

Future work should add more labeled examples for underrepresented document types and taxonomy branches, persist the final trained classification model, improve run metadata tracking, and support multi-label taxonomy assignment for documents that naturally belong to more than one banking topic.
