"""Static configuration: paths and variant registry.

Adding a new variant = one entry here. Pages never hardcode filenames.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"

RAW_SYNTHETIC_DIR = DATA / "synthetic" / "raw"
RAW_REAL_DIR = DATA / "raw"

EXTRACTED_FILE = DATA / "extracted" / "extracted_documents.jsonl"
PROCESSED_FILE = DATA / "processed" / "clean_documents.jsonl"
ANONYMIZED_DIR = DATA / "anonymized"
CLASSIFIED_DIR = DATA / "classified"

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

CLASSIFICATION_VARIANTS_REAL: dict[str, str] = {
    "TF-IDF": "classification_results__tfidf.jsonl",
    "Embedding (base)": "classification_results__embedding.jsonl",
    "Embedding (large)": "classification_results__embedding_large.jsonl",
    "Zero-shot (base)": "classification_results__zeroshot.jsonl",
    "Zero-shot (large)": "classification_results__zeroshot_large.jsonl",
}
CLASSIFICATION_DEFAULT_REAL = "Embedding (base)"

CLASSIFICATION_VARIANTS_SYN: dict[str, str] = {
    "TF-IDF (raw)": "syn_tfidf_raw.jsonl",
    "TF-IDF (generic placeholders)": "syn_tfidf_generic.jsonl",
    "TF-IDF (specific placeholders)": "syn_tfidf_specific.jsonl",
}
CLASSIFICATION_DEFAULT_SYN = "TF-IDF (specific placeholders)"
