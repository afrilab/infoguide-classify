# Internal Progress Report – InfoGuide Pilot-2

This document records internal role assignments, concrete responsibilities, completed tasks and remaining tasks for the InfoGuide Pilot-2 project.  

---
## Dilan Sataloğlu – Corpus, Data Governance & Anonymization Lead

Is responsible for all data-centric, governance-critical, and privacy-sensitive components of the project.

### Responsibilities
- Selection and justification of open-access document corpora
- Verification and documentation of data licensing, provenance, and usage constraints
- Design and implementation of the document ingestion pipeline
  - Text extraction from PDF / HTML
- Design and maintenance of corpus manifest file where document metadata is hold
  - Metadata extraction (document ID, source, filename, retrieval date, license)
- Implementation of PII detection and anonymization pipelines (e.g., names, identifiers, organizations where applicable)
- Definition and documentation of anonymization and governance rules (what is anonymized, what is preserved, and why)
- Ensuring traceability of all experiments (runs) and maintaining corpus versioning

---
## Doğa İzci – Preprocessing & Document Classification Lead

Is responsible for performing preprocessing and document-level classification.

### Responsibilities
- Design and implementation of Step 2: Text Preprocessing and Normalization
  - Boilerplate removal (headers, footers, page numbers)
  - Whitespace and encoding normalization
  - Sentence or paragraph segmentation
- Documentation of preprocessing rules and reproducibility guarantees
- Design of the document classification problem
  - Label definitions, scope, and assumptions
- Implementation of document representation and classification models (e.g., embedding-based, transformer-based, or hybrid)
- Evaluation of classification quality (accuracy, F1-score, confusion analysis)

---
## Arda Keleş – Taxonomy, System Integration & Evaluation Lead

Is responsible for taxonomy ownership, system-level integration, and evaluation.

### Responsibilities
- Design of the document taxonomy
  - Definition of hierarchical taxonomy levels (Domain → Subdomain → Topic)
  - Documentation of taxonomy rationale and assumptions
- Implementation of taxonomy assignment logic
- Implementation of automated and manual evaluation pipelines
- Design and deployment of the human-in-the-loop review interface
- Consolidation of experimental results into progress and final reports
- Ensuring system-level reproducibility, including one-command execution aligned with DoD-4 requirements
- Ensuring repository structure, documentation quality, and DoD compliance