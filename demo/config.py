"""Static configuration: paths and variant registry.

Adding a new variant = one entry here. Pages never hardcode filenames.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
OUTPUTS = PROJECT_ROOT / "outputs"
DOCS = PROJECT_ROOT / "docs"

# Module label -> design-evaluation markdown path. Add a module = one entry.
DESIGN_EVALUATIONS: dict[str, Path] = {
    "Anonymization": DOCS / "anonymization_design_evaluation.md",
    "Classification": DOCS / "classification_design_evaluation.md",
}

RAW_SYNTHETIC_DIR = DATA / "synthetic" / "raw"
RAW_REAL_DIR = DATA / "raw"
SYNTHETIC_RAW_JSONL = DATA / "synthetic" / "synthetic_raw.jsonl"

EXTRACTED_FILE = DATA / "extracted" / "extracted_documents.jsonl"
PROCESSED_FILE = DATA / "processed" / "clean_documents.jsonl"
ANONYMIZED_DIR = DATA / "anonymized"
CLASSIFIED_DIR = DATA / "classified"

ANON_EVAL_DIR = OUTPUTS / "anonymizer_evaluation"
CLASSIFICATION_OUTPUTS_DIR = OUTPUTS / "classification_outputs"
CLASSIFICATION_RESULTS_DIR = OUTPUTS / "classification_results"

ANONYMIZATION_VARIANTS: dict[str, str] = {
    "regex + spaCy hybrid (sm)": "regex_spacy_hybrid_documents.jsonl",
    "regex + spaCy hybrid (lg)": "regex_spacy_hybrid_lg_documents.jsonl",
    "regex + spaCy hybrid (trf)": "regex_spacy_hybrid_trf_documents.jsonl",
    "regex + spaCy hybrid (trf, context-filtered)": "regex_spacy_hybrid_trf_context_filtered_documents.jsonl",
    "regex + spaCy hybrid (trf, generic placeholders)": "regex_spacy_hybrid_trf_generic_placeholder_documents.jsonl",
    "regex + DeBERTa hybrid (context-filtered)": "regex_deberta_hybrid_context_filtered_documents.jsonl",
    "DeBERTa NER baseline": "deberta_ner_baseline_documents.jsonl",
    "GLiNER baseline": "gliner_baseline_documents.jsonl",
    "spaCy NER only": "spacy_ner_only_documents.jsonl",
    "regex only": "regex_only_documents.jsonl",
}
ANONYMIZATION_DEFAULT = "regex + spaCy hybrid (sm)"

# Folder name -> human label, for outputs/anonymizer_evaluation/<folder>/
ANON_EVAL_APPROACHES: dict[str, str] = {
    "regex_spacy_hybrid": "regex + spaCy hybrid (sm)",
    "regex_spacy_hybrid_lg": "regex + spaCy hybrid (lg)",
    "regex_spacy_hybrid_trf": "regex + spaCy hybrid (trf)",
    "regex_spacy_hybrid_trf_context_filtered": "regex + spaCy hybrid (trf, context-filtered)",
    "regex_deberta_hybrid_context_filtered": "regex + DeBERTa hybrid (context-filtered)",
    "deberta_ner_baseline": "DeBERTa NER baseline",
    "deberta_ner_baseline_full": "DeBERTa NER baseline (full)",
    "gliner_baseline": "GLiNER baseline",
    "spacy_ner_only": "spaCy NER only",
    "regex_only": "regex only",
}

# Unified classification registry — every model is selectable regardless of corpus.
# If a given variant has no record for the selected doc_id, the UI says so cleanly.
CLASSIFICATION_VARIANTS: dict[str, str] = {
    "TF-IDF": "classification_results__tfidf.jsonl",
    "Embedding (base)": "classification_results__embedding.jsonl",
    "Embedding (large)": "classification_results__embedding_large.jsonl",
    "Zero-shot (base)": "classification_results__zeroshot.jsonl",
    "Zero-shot (large)": "classification_results__zeroshot_large.jsonl",
    "TF-IDF — synthetic (raw)": "syn_tfidf_raw.jsonl",
    "TF-IDF — synthetic (generic placeholders)": "syn_tfidf_generic.jsonl",
    "TF-IDF — synthetic (specific placeholders)": "syn_tfidf_specific.jsonl",
}
CLASSIFICATION_DEFAULTS = {
    "real": "Embedding (base)",
    "synthetic": "TF-IDF — synthetic (specific placeholders)",
}
