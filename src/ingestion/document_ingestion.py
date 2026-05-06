import os
import csv
import json
from pathlib import Path
import yaml
import pdfplumber
import logging
from bs4 import BeautifulSoup
import argparse

logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Function to load a YAML configuration file and convert it into a Python dictionary
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Function to load corpus_manifest.csv file and creates a python dictionary (doc_id --> metadata)
def load_manifest(manifest_path: str) -> dict:
    manifest = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        header = f.readline()
        f.seek(0)
        delimiter = ";" if header.count(";") > header.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            doc_id = (row.get("doc_id") or "").strip()
            if doc_id:
                manifest[doc_id] = row
    return manifest

# Function to go over all directories under input_dir, extact file type (ex:pdf), and only accept the ones specified in config (supported_types)
# Saves file path as an Path object
def discover_files(input_dir: str, supported_types: list[str]) -> list[Path]:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for name in filenames:
            extension = name.split(".")[-1].lower()
            if extension in supported_types:
                files.append(Path(root) / name)
    return files

# Function to extract text from pdf's using pdfplumber
def extract_text_from_pdf(pdf_path: Path) -> str:
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)

    return "\n".join(text_chunks)

def extract_text_from_txt(txt_path: Path, encoding: str = "utf-8") -> str:
    with open(txt_path, "r", encoding=encoding) as f:
        return f.read()


def extract_text_from_html(html_path: Path, encoding: str = "utf-8") -> str:
    html = html_path.read_text(encoding=encoding, errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    raw_text = soup.get_text(separator="\n")

    cleaned_lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]
    return "\n".join(cleaned_lines)


def extract_raw_text(file_path: Path, encoding: str = "utf-8") -> str:
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".txt":
        return extract_text_from_txt(file_path, encoding)
    if ext == ".html":
        return extract_text_from_html(file_path, encoding)

    raise ValueError(f"Unsupported file type: {ext}")

# Function to find the matching corpus_manifest.csv entry for pdf filenames 
def find_manifest_entry(filename: str, manifest: dict) -> dict | None:
    for entry in manifest.values():
        if entry["filename"] == filename:
            return entry
    return None

# Function to compute basic statistics about extracted text
def compute_text_stats(text: str) -> dict:
    lines = text.splitlines()
    non_empty_lines = [l for l in lines if l.strip()]
    all_words = text.split()
    alpha_words = [w for w in all_words if w.isalpha()]

    return {
        "word_count": len(all_words),
        "alpha_word_count": len(alpha_words),
        "avg_words_per_line": (
            len(all_words) / len(non_empty_lines)
            if non_empty_lines else 0
        ),
        "avg_word_length": (
            sum(len(w) for w in alpha_words) / len(alpha_words)
            if alpha_words else 0
        ),
    }

# Function to evaluate the quality of extracted text using simple heuristic rules
# and classify it as either 'healthy' or 'warning'
def assess_extraction_quality(stats: dict) -> dict:
    issues = []

    # Very low content
    if stats["word_count"] < 200:
        issues.append("low_content")

    # Mostly non-alphabetic
    alpha_ratio = (
        stats["alpha_word_count"] / stats["word_count"]
        if stats["word_count"] else 0
    )
    if alpha_ratio < 0.3:
        issues.append("low_alphabetic_content")

    # Fragmented words
    if stats["avg_word_length"] < 4:
        issues.append("fragmented_words")

    # Weak / fragmented line structure
    if stats["avg_words_per_line"] < 3:
        issues.append("weak_line_structure")

    return {
        "status": "healthy" if not issues else "warning",
        "issues": issues
    }

# Function to combine manifest metadata + extracted text + analysis into a single record
def build_document_record(manifest_entry: dict, raw_text: str) -> dict:
    stats = compute_text_stats(raw_text)
    quality = assess_extraction_quality(stats)

    record = dict(manifest_entry)

    record["extraction_stats"] = stats
    record["extraction_quality"] = quality
    record["raw_text"] = raw_text

    return record

def build_synthetic_manifest_entry(file_path: Path) -> dict:
    doc_id = file_path.stem

    return {
        "doc_id": doc_id,
        "filename": file_path.name,
        "file_format": file_path.suffix.replace(".", "").lower(),
        "source": "synthetic",
        "source_url": "",
        "language": "en",
        "retrieval_date": "",
        "publication_date": "",
        "license": "synthetic",
        "simulated_banking_role": "",
        "document_type": "synthetic",
    }


def process_manifest_source(
    source_cfg: dict,
    output_file,
    encoding: str,
) -> list[dict]:
    input_dir = source_cfg["input_dir"]
    supported_types = source_cfg["supported_types"]
    manifest_path = source_cfg["manifest_path"]

    manifest = load_manifest(manifest_path)
    files = discover_files(input_dir, supported_types)

    results = []

    for file_path in files:
        filename = file_path.name
        entry = find_manifest_entry(filename, manifest)

        if entry is None:
            print(f"[WARN] No manifest entry for {filename}, skipping.")
            continue

        raw_text = extract_raw_text(file_path, encoding)

        doc = build_document_record(
            manifest_entry=entry,
            raw_text=raw_text,
        )

        output_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
        results.append(doc)

        print(
            f"[INGEST][corpus] {doc['doc_id']} | "
            f"{doc['extraction_quality']['status']}"
        )

    return results


def process_synthetic_source(
    source_cfg: dict,
    output_file,
    encoding: str,
) -> list[dict]:
    input_dir = source_cfg["input_dir"]
    supported_types = source_cfg["supported_types"]

    files = discover_files(input_dir, supported_types)

    results = []

    for file_path in files:
        raw_text = extract_raw_text(file_path, encoding)
        entry = build_synthetic_manifest_entry(file_path)

        doc = build_document_record(
            manifest_entry=entry,
            raw_text=raw_text,
        )

        output_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
        results.append(doc)

        print(
            f"[INGEST][synthetic] {doc['doc_id']} | "
            f"{doc['extraction_quality']['status']}"
        )

    return results


def print_ingestion_summary(results: list[dict]) -> None:
    total = len(results)
    warnings = sum(
        1 for record in results
        if record["extraction_quality"]["status"] == "warning"
    )

    print("\n--- INGESTION SUMMARY ---")
    print(f"Total documents: {total}")
    print(f"Warnings: {warnings}")
    print(f"Healthy: {total - warnings}")


def run_ingestion(config_path: str) -> None:
    cfg = load_config(config_path)

    output_path = cfg["output_path"]
    encoding = cfg.get("encoding", "utf-8")
    sources = cfg.get("sources", {})

    os.makedirs(Path(output_path).parent, exist_ok=True)

    results = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        corpus_cfg = sources.get("corpus", {})
        if corpus_cfg.get("enabled", False):
            results.extend(
                process_manifest_source(
                    source_cfg=corpus_cfg,
                    output_file=out_f,
                    encoding=encoding,
                )
            )
        else:
            print("[INGEST] corpus source disabled.")

        synthetic_cfg = sources.get("synthetic", {})
        if synthetic_cfg.get("enabled", False):
            results.extend(
                process_synthetic_source(
                    source_cfg=synthetic_cfg,
                    output_file=out_f,
                    encoding=encoding,
                )
            )
        else:
            print("[INGEST] synthetic source disabled.")

    print_ingestion_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_ingestion(args.config)
