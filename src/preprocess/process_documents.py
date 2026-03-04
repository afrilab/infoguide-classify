
# Usage:
#   python scripts/process_documents.py --config configs/process.yaml
#

#   pip install spacy pyyaml

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import spacy


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)



_ws_re = re.compile(r"[ \t]+")


def normalize_whitespace(text: str) -> str:
    # Normalize line endings + collapse spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ws_re.sub(" ", text)
    # strip trailing spaces per line
    text = "\n".join([ln.strip() for ln in text.splitlines()])
    # collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_boilerplate(text: str) -> Tuple[str, Dict[str, Any]]:

    debug: Dict[str, Any] = {"removed_lines": 0, "rules": []}
    lines = text.splitlines()
    out: List[str] = []

    page_number_re = re.compile(r"^\s*(page\s*)?\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE)
    toc_dots_re = re.compile(r"\.{3,}\s*\d+\s*$")
    urlish_re = re.compile(r"(https?://|www\.)", re.IGNORECASE)

    removed = 0
    for ln in lines:
        raw = ln.strip()
        if not raw:
            out.append("")
            continue

        if page_number_re.match(raw):
            removed += 1
            continue

        if toc_dots_re.search(raw):
            removed += 1
            continue

        if urlish_re.search(raw) and len(raw) < 120:
            removed += 1
            continue

        out.append(ln)

    debug["removed_lines"] = removed
    debug["rules"] = ["page_number", "toc_dots", "urlish_footer"]
    return "\n".join(out), debug


def build_repeated_line_set(
    texts: List[str],
    min_len: int,
    min_occ: int,
    max_len: int,
) -> set:

    counter = Counter()
    for t in texts:
        for line in t.splitlines():
            s = normalize_whitespace(line).strip()
            if len(s) < min_len:
                continue
            if len(s) > max_len:
                continue
            counter[s] += 1

    repeated = {s for s, c in counter.items() if c >= min_occ}
    return repeated


def drop_repeated_lines(text: str, repeated: set) -> Tuple[str, int]:
    removed = 0
    out_lines: List[str] = []
    for ln in text.splitlines():
        s = normalize_whitespace(ln).strip()
        if s in repeated:
            removed += 1
            continue
        out_lines.append(ln)
    return "\n".join(out_lines), removed


def compute_stats(text: str) -> Dict[str, int]:
    lines = [ln.strip() for ln in text.splitlines()]
    non_empty = [ln for ln in lines if ln]
    words = sum(len(ln.split()) for ln in non_empty)
    chars = sum(len(ln) for ln in non_empty)
    return {
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty),
        "word_count": words,
        "char_count": chars,
    }


def assess_quality(stats: Dict[str, int], cfg: Dict[str, Any]) -> Dict[str, Any]:
    thr = cfg["quality_thresholds"]
    issues: List[str] = []

    if stats["word_count"] < thr["min_words_warning"]:
        issues.append("too_short")
    if stats["char_count"] < thr["min_chars_warning"]:
        issues.append("very_low_content")

    denom = stats["non_empty_line_count"] or 1
    avg_words_per_line = stats["word_count"] / denom
    if avg_words_per_line < thr["min_avg_words_per_line_warning"]:
        issues.append("fragmented_text")

    return {"status": "ok" if not issues else "warning", "issues": issues}


def make_sentencizer(lang: str = "en"):
    nlp = spacy.blank(lang)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(nlp, text: str) -> List[str]:
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents]
    return [s for s in sents if s]


def main(config_path: str) -> None:
    cfg = load_config(Path(config_path))

    in_path = Path(cfg["input_path"])
    out_path = Path(cfg["output_path"])
    lang = cfg.get("language", "en")

    print(f"Reading:  {in_path}")
    docs = read_jsonl(in_path)
    print(f"Loaded {len(docs)} docs")

    rl = cfg["repeated_lines"]
    min_len = int(rl["min_len"])
    min_occ = int(rl["min_occ"])
    max_len = int(rl["max_len"])

    raw_texts = [d.get("raw_text", "") or "" for d in docs]
    repeated_lines = build_repeated_line_set(
        raw_texts,
        min_len=min_len,
        min_occ=min_occ,
        max_len=max_len,
    )
    print(f"Repeated lines detected: {len(repeated_lines)}")

    include_sentences = bool(cfg["output_fields"].get("include_sentences", True))
    include_debug = bool(cfg["output_fields"].get("include_debug", True))

    nlp = make_sentencizer(lang)

    cleaned_records: List[Dict[str, Any]] = []
    warnings = 0

    for d in docs:
        doc_id = d.get("doc_id") or d.get("document_id") or d.get("id")
        raw_text = d.get("raw_text", "") or ""

        step_debug: Dict[str, Any] = {}

        text0 = normalize_whitespace(raw_text)
        text1, dbg1 = remove_boilerplate(text0)
        step_debug["remove_boilerplate"] = dbg1

        text2, removed_rep = drop_repeated_lines(text1, repeated_lines)
        step_debug["drop_repeated_lines"] = {"removed_lines": removed_rep}

        cleaned_text = normalize_whitespace(text2)

        sentences = split_sentences(nlp, cleaned_text)

        stats = compute_stats(cleaned_text)
        quality = assess_quality(stats, cfg)
        if quality["status"] != "ok":
            warnings += 1

        record: Dict[str, Any] = {
            "doc_id": doc_id,
            "source": d.get("source"),
            "source_url": d.get("source_url"),
            "filename": d.get("filename"),
            "document_type": d.get("document_type"),
            "language": d.get("language", cfg.get("language", "en")),
            "retrieval_date": d.get("retrieval_date"),
            "license": d.get("license"),
            "processed_text": cleaned_text,
            "processed_stats": stats,
            "processed_quality": quality,
            "sentence_count": len(sentences),
        }

        if include_sentences:
            record["sentences"] = sentences
        if include_debug:
            record["processing_debug"] = step_debug

        cleaned_records.append(record)

    print(f"\nWriting: {out_path}")
    write_jsonl(out_path, cleaned_records)

    print("\n--- PREPROCESS SUMMARY ---")
    print(f"Total documents: {len(cleaned_records)}")
    print(f"Warnings: {warnings}")
    print(f"Healthy: {len(cleaned_records) - warnings}")
    print("✅ Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to process.yaml")
    args = parser.parse_args()

    main(args.config)
