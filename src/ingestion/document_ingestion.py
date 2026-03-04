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
        reader = csv.DictReader(f)
        for row in reader:
            manifest[row["doc_id"]] = row
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

# Function to extract text from .txt files
def extract_text_from_txt(txt_path: Path) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

# Function to extract text from html files using BeautifulSoup
def extract_text_from_html(html_path: Path) -> str:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    raw_text = soup.get_text(separator="\n")  # Extract all text

    # Remove empty lines and extra whitespace
    cleaned_lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]
    return "\n".join(cleaned_lines)

# Function to call the right text extraction function based on file format
def extract_raw_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".txt":
        return extract_text_from_txt(file_path)
    if ext == ".html":
        return extract_text_from_html(file_path)

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
def build_document_record(doc_id: str, manifest_entry: dict, raw_text: str) -> dict:
    stats = compute_text_stats(raw_text)
    quality = assess_extraction_quality(stats)

    record = dict(manifest_entry)

    record["extraction_stats"] = stats
    record["extraction_quality"] = quality
    record["raw_text"] = raw_text

    return record

# Function to print a health report for the extraction process
def print_ingestion_summary(results: list[dict]):
    total = len(results)
    warnings = sum(
        1 for r in results
        if r["extraction_quality"]["status"] == "warning"
    )

    print("\n--- INGESTION SUMMARY ---")
    print(f"Total documents: {total}")
    print(f"Warnings: {warnings}")
    print(f"Healthy: {total - warnings}")


# Function that serves as main orcestrator, to run the whole pdf extraction pipeline
def run_ingestion(config_path: str):
    cfg = load_config(config_path)
    manifest = load_manifest(cfg["manifest_path"])
    files = discover_files(cfg["input_dir"], cfg["supported_types"])

    os.makedirs(Path(cfg["output_path"]).parent, exist_ok=True)

    results = []

    with open(cfg["output_path"], "w", encoding="utf-8") as out_f:
        for file_path in files:
            filename = file_path.name
            entry = find_manifest_entry(filename, manifest)

            if entry is None:
                print(f"[WARN] No manifest entry for {filename}, skipping.")
                continue

            raw_text = extract_raw_text(file_path)

            doc = build_document_record(
                doc_id=entry["doc_id"],
                manifest_entry=entry,
                raw_text=raw_text
            )

            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            results.append(doc)

            print(
                f"[INGEST] {doc['doc_id']} | "
                f"{doc['extraction_quality']['status']} "
            )

    print_ingestion_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_ingestion(args.config)