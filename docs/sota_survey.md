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
