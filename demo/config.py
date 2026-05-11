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
    "presidio_hybrid (sm, base)": "presidio_hybrid_documents.jsonl",
    "presidio_hybrid (lg)": "presidio_hybrid_lg_documents.jsonl",
    "presidio_hybrid (trf)": "presidio_hybrid_trf_documents.jsonl",
    "presidio_hybrid (trf, context-filtered)": "presidio_hybrid_trf_context_filtered_documents.jsonl",
    "presidio_hybrid (trf, generic placeholders)": "presidio_hybrid_trf_generic_placeholder_documents.jsonl",
    "presidio (NER only)": "presidio_ner_only_documents.jsonl",
    "presidio (regex only)": "presidio_regex_only_documents.jsonl",
}
ANONYMIZATION_DEFAULT = "presidio_hybrid (sm, base)"

# Folder name -> human label, for outputs/anonymizer_evaluation/<folder>/
ANON_EVAL_APPROACHES: dict[str, str] = {
    "presidio_hybrid": "presidio_hybrid (sm, base)",
    "presidio_hybrid_lg": "presidio_hybrid (lg)",
    "presidio_hybrid_trf": "presidio_hybrid (trf)",
    "presidio_hybrid_trf_context_filtered": "presidio_hybrid (trf, context-filtered)",
    "presidio_ner_only": "presidio (NER only)",
    "presidio_regex_only": "presidio (regex only)",
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
