## (A) Document Classification Models (SoTA)

This section surveys state-of-the-art approaches for document-level and segment-level classification. In InfoGuide Pilot-2, these approaches inform the **Document Classifier** module and provide alternative baselines under different constraints (availability of labels, document length, and domain shift across sources).

### A.1 — BERT / RoBERTa-style Transformer Classifiers (Supervised Fine-tuning)

**What it does:**  
Assigns one or more semantic labels to a document (or document segment), such as _Policy_, _Regulation_, _Guideline_, or _Report_, using supervised learning when labeled examples exist.

**How it works (high level):**  
A Transformer encoder (e.g., BERT/RoBERTa) encodes the input text into contextual representations. A classification head (often a linear layer) predicts class probabilities from a pooled representation (e.g., [CLS]) or an aggregated segment representation. For long documents, common strategies include truncation, chunking into segments, and aggregating segment-level predictions (majority vote, max confidence, or weighted averaging).

**Why it is relevant to InfoGuide Pilot-2:**  
This is the most standard and strong baseline for the **Document Classification** step after preprocessing and (optionally) anonymization. It also provides confidence scores that can be used in governance checks (e.g., abstain or flag low-confidence assignments).

**Usage decision:**

### A.2 — DeBERTa (Stronger Transformer Encoder Baseline)

**What it does:**  
Performs high-accuracy document/segment classification, often improving performance over older BERT-style encoders on many benchmarks.

**How it works (high level):**  
DeBERTa improves Transformer encoding by modeling content and position information more effectively than classic BERT-style setups. In practice, it is used similarly: encode input text and fine-tune a classification head for the target label set. It can also support chunk-level inference with aggregation for long documents.

**Why it is relevant to InfoGuide Pilot-2:**  
Pilot-2 processes heterogeneous enterprise-style documents where subtle domain cues matter (e.g., compliance vs. governance vs. consumer guidance). A stronger encoder baseline is important to demonstrate that model choice was considered systematically, not arbitrarily.

**Usage decision:**

### A.3 — Zero-shot Classification via NLI (No Labeled Training Data)

**What it does:**  
Assigns labels to documents **without task-specific training data**, using only candidate label descriptions (e.g., “This document is about AML compliance”).

**How it works (high level):**  
A Natural Language Inference (NLI) model evaluates whether the document text (premise) entails a label hypothesis (e.g., “This text is a compliance policy”). For each candidate label, an entailment score is computed; the label(s) with highest scores are selected. For long documents, the method is typically applied per chunk and then aggregated across chunks.

**Why it is relevant to InfoGuide Pilot-2:**  
Pilot-2 may not have a labeled dataset for its specific banking/governance taxonomy. Zero-shot NLI provides a realistic baseline that can run immediately, supports rapid iteration, and helps bootstrap early taxonomy/category assignments before supervised training is possible.

**Usage decision:**

### A.4 — Embedding-based Topic/Cluster Assignment (SBERT + Clustering + Labeling)

**What it does:**  
Groups documents into semantically coherent clusters and assigns cluster-level topics/labels, enabling classification under weak supervision or unsupervised settings.

**How it works (high level):**  
A document embedding model (commonly SBERT-style) maps each document or chunk to a fixed-dimensional vector. Clustering algorithms (e.g., KMeans or HDBSCAN) group similar vectors. Labels can then be assigned by: (i) extracting cluster keywords (TF-IDF), (ii) mapping clusters to taxonomy nodes, or (iii) using a lightweight classifier/zero-shot step to name clusters.

**Why it is relevant to InfoGuide Pilot-2:**  
This approach supports taxonomy-oriented organization when labels are missing or noisy, and it helps check whether the taxonomy is coherent (clusters should align with taxonomy paths). It also scales well because embeddings and clustering can be batched and parallelized.

**Usage decision:**

**Summary (A):**  
We cover supervised Transformer classifiers (BERT/RoBERTa, DeBERTa) as high-accuracy baselines, zero-shot NLI classification as a no-label alternative, and embedding-based clustering as a weakly supervised approach for discovering structure in unlabeled corpora. These represent complementary strategies under Pilot-2 constraints (limited labels, long documents, and heterogeneous sources).

## (B) PII Detection and Anonymization Systems (SoTA)

### B.1 — Rule-based Approaches (Regex, Pattern Matching)

**What it does**

Rule-based anonymization approaches aim to detect and mask sensitive information (PII) in text using predefined patterns. These methods are primarily used for identifying structured PII such as email addresses, phone numbers, URLs, and identification numbers, where the format is well-defined and predictable. 

**How it works (high level)**
  
Rule-based approaches rely on explicitly defined detection rules to identify PII spans in text. The most common mechanism is regular expression (regex) matching, where patterns are designed to capture the structural characteristics of entities such as email addresses, phone numbers, or identification numbers. These patterns typically encode constraints such as allowed character sets, positional structure (e.g., local-part@domain for emails), length, and formatting variations.

In addition to regex-based rules, rule-based systems may also incorporate dictionary or blacklist-based matching, where known sensitive terms (e.g., organization names, person names, or domain-specific entities) are detected through lookup tables. Rule-based systems can also use contextual clues to support pattern matching. For example, if a detected string matches an email format and appears near words such as “contact” or “email,” it is more likely to be treated as valid sensitive information.

Once a span is detected, different anonymization strategies can be applied depending on the use case. Common approaches include replacing the span with a placeholder (e.g., [EMAIL]), masking parts of the value (e.g., j***@mail.com), deleting the span entirely, or applying irreversible transformations such as hashing. The choice of transformation depends on the desired balance between privacy protection, information retention and text preservation.

Rule-based systems are deterministic and interpretable. Given the same input and rule set, they always produce the same output. Their effectiveness is therefore directly tied to the completeness and quality of the defined rules.

**Why it is relevant to InfoGuide Pilot-2**  

Rule-based anonymization is directly relevant to the anonymization component of the InfoGuide pipeline. In the system, regex-based detection is used as a high-precision layer for structured PII types such as URLs, emails, and phone numbers. It complements model-based approaches (NER) by ensuring detection of entities that follow strict formatting rules. This makes it an essential building block within a hybrid anonymization pipeline.

**Usage decision**

Rule-based approaches are used directly in the InfoGuide Pilot-2 implementation as part of the hybrid anonymization module. However, they are not used as a standalone solution due to their inability to detect unstructured entities such as person or organization names. Instead, they serve as a high-confidence component combined with NER-based component.

**Strengths:**
- High precision for structured PII (e.g., emails, URLs)
- Fast and computationally efficient
- Deterministic and interpretable
- Easy to control and customize

**Limitations:**
- Cannot detect unstructured or context-dependent entities (e.g., names)
- Requires manual rule design and maintenance
- Limited generalization to unseen patterns
- Performance depends heavily on rule coverage

Rule-based anonymization remains widely used in production systems as a high-precision component, often integrated with machine learning models in hybrid architectures (e.g., Microsoft Presidio).

### B.2 — NER-based Approaches (Statistical and Neural)

**What it does**

NER-based anonymization approaches aim to detect sensitive information (PII) by identifying and classifying named entities in text. Unlike rule-based methods that rely on predefined patterns, these approaches recognize entities such as person names, organizations and locations based on linguistic context, enabling detection of unstructured and context-dependent PII.

**How it works (high level)**

NER-based systems treat PII detection as a sequence labeling task, where each token in a sentence is assigned a label (e.g., PERSON, ORG, LOCATION). These approaches can be broadly divided into statistical and neural methods.

Statistical NER models rely on manually engineered features and probabilistic sequence models. Features may include capitalization, word shape, surrounding tokens, and part-of-speech tags. These models learn patterns from labeled data but depend heavily on feature design.

Neural NER models use deep learning architectures to automatically learn representations from data. Common approaches include BiLSTM-based models and transformer-based models such as BERT or RoBERTa. 

Once entities are detected, anonymization is applied. 

**Why it is relevant to InfoGuide Pilot-2**  

NER-based approaches are essential for detecting unstructured and context-dependent PII that cannot be captured by rule-based methods. In the InfoGuide pipeline, they complement regex-based detection by identifying entities such as person names and organization names, which do not follow fixed patterns. In this project, a spaCy-based NER model (en_core_web_sm) is used as the primary entity recognition component. This makes NER a critical component of the hybrid anonymization system.

**Usage decision**

NER-based approaches are used directly in the InfoGuide Pilot-2 implementation as the primary mechanism for detecting context-dependent entities. In this project a spaCy-based NER model is combined with rule-based methods in a hybrid architecture.

**Strengths:**
- Capable of detecting unstructured and context-dependent PII (e.g., names, organizations)
- Contextual understanding
- Reduces need for manual rule design
- More adaptable to different domains compared to rule-based systems

**Limitations:**
- Performance depends on training data and domain alignment
- Lower precision compared to rule-based methods for structured entities
- Computationally more expensive than rule-based approaches
- Limited interpretability

### B.3 — Hybrid and Production-Oriented Anonymization Systems

**What it does**

Hybrid and production-oriented anonymization systems aim to detect and anonymize sensitive information (PII) by combining multiple approaches within a unified pipeline. These systems integrate rule-based methods and NER-based models to achieve both high precision and high recall, while also supporting scalable, configurable and deployable anonymization workflows.

**How it works (high level)**

Hybrid anonymization systems combine deterministic and model-based components in a multi-stage pipeline. Typically, rule-based methods (e.g., regex or dictionary matching) are applied first to detect structured PII such as email addresses, phone numbers, or URLs with high precision. Then, NER-based models are used to identify context-dependent entities such as person and organizations names.

A key component of hybrid systems is conflict resolution and span management. When multiple detection methods produce overlapping spans, predefined policies are applied to ensure consistency. For example, rule-based detections may take precedence over model-based detections. Additionally, confidence thresholds can be applied to filter low-confidence model predictions.

Production-oriented systems extend this pipeline by adding engineering components such as configuration management, policy-driven anonymization rules, and logging mechanisms. A representative example of such a system is Microsoft Presidio, which combines rule-based recognizers and NLP-based entity detection within a unified framework, allowing customizable anonymization strategies and detailed detection logs. Detected entities are processed according to predefined anonymization strategies (e.g., replacement with placeholders, masking, or deletion).

**Why it is relevant to InfoGuide Pilot-2**  

Hybrid anonymization is directly aligned with the anonymization requirements of InfoGuide Pilot-2, where both structured and unstructured PII must be detected and processed in a reproducible and auditable manner.

**Usage decision**

Hybrid and production-oriented approaches are used directly in the InfoGuide Pilot-2 implementation as the core anonymization strategy. Rather than relying on a single method, the system combines rule-based and NER-based components to leverage their complementary strengths.
In this project, a hybrid approach is implemented using:
- regex-based detection for structured entities,
- a spaCy-based NER model for context-dependent entities,
- a precedence policy where rule-based detections override overlapping NER outputs,
- configurable thresholds and entity selection via external configuration files.

In particular, Microsoft Presidio is used as the primary framework, serving as a strong baseline for Phase-2 comparisons.

### B.4 — LLM-based Anonymization Approaches

**What it does**

LLM-based anonymization approaches aim to remove or transform sensitive information (PII) using large language models (LLMs). Unlike traditional methods that only detect and mask entities, these approaches can rewrite or transform text to preserve meaning while eliminating sensitive information, enabling more flexible and context-aware anonymization.

**How it works (high level)**

LLM-based anonymization leverages large pretrained language models to process entire text sequences and generate anonymized outputs. LLM-based anonymization treats the task as a text transformation or generation problem rather than a sequence labeling task.

The model can be prompted or fine-tuned to identify sensitive information and produce a modified version of the input text where PII is removed, replaced, or generalized. For example, a sentence containing a person’s name may be rewritten using a placeholder (e.g., [PERSON]) or transformed into a more generic expression (e.g., “a customer”).

LLM-based anonymization offers strong contextual understanding and can handle ambiguous or implicit PII by rewriting text while preserving meaning. However, it is non-deterministic, may fail to fully remove sensitive information, and is less interpretable than rule-based methods. Additionally, it introduces challenges in reproducibility, auditability, and computational cost.

**Why it is relevant to InfoGuide Pilot-2**  

LLM-based approaches represent a more advanced and flexible method for anonymization, particularly for complex documents where sensitive information is embedded in complex linguistic contexts. They are relevant as a potential extension to the InfoGuide pipeline, especially for handling cases where rule-based and NER-based methods fail.

**Usage decision**

LLM-based anonymization approaches are not directly used in the current InfoGuide Pilot-2 implementation. However, they may be considered as a potential enhancement for future iterations, particularly in scenarios requiring context-aware rewriting or handling of ambiguous PII.

**References for the anonymization section**
- Deußer et al., *A Survey on Current Trends and Recent Advances in Text Anonymization*, 2025.
- Mishra et al., *A hybrid rule-based NLP and machine learning approach for PII detection and anonymization in financial documents*, 2025.
- Microsoft Presidio, *Presidio: Data Protection and Anonymization SDK*, 2025.

## (C) Taxonomy, Ontology, and Document Organization Systems (SoTA)