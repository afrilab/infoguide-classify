"""
assign_taxonomy_baseline.py

Baseline taxonomy assignment (rule-based keyword matching) for InfoGuide Pilot-2.

Input : data/processed/clean_documents.jsonl   (or any JSONL with doc_id + text fields)
Config: configs/taxonomy.yaml                  (3-level hierarchy)
Output: outputs/taxonomy/taxonomy_assignments.jsonl

Usage:
  python src/taxonomy/assign_taxonomy_baseline.py \
    --input data/processed/clean_documents.jsonl \
    --taxonomy configs/taxonomy.yaml \
    --output outputs/taxonomy/taxonomy_assignments.jsonl
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_phrase(phrase: str) -> str:
    phrase = phrase.strip().lower()
    phrase = re.escape(phrase)
    # replace escaped spaces "\ " with regex whitespace "\s+"
    phrase = phrase.replace(r"\ ", r"\s+")
    return phrase



def compile_taxonomy_patterns(taxonomy: Dict[str, Any]) -> List[Tuple[str, str, str, re.Pattern]]:
    """
    taxonomy.yaml expected structure:

    taxonomy:
      Level1:
        Final Label:
          description: label description
          keywords:
            - keyword_or_phrase_1
            - keyword_or_phrase_2
      ...

    Returns list of tuples: (l1, l2, l3, compiled_regex)
    """
    if "taxonomy" not in taxonomy or not isinstance(taxonomy["taxonomy"], dict):
        raise ValueError("taxonomy.yaml must contain a top-level key 'taxonomy' with a nested mapping.")

    patterns: List[Tuple[str, str, str, re.Pattern]] = []
    for l1, l2_map in taxonomy["taxonomy"].items():
        if not isinstance(l2_map, dict):
            continue
        for l2, l2_info in l2_map.items():
            if isinstance(l2_info, dict):
                keywords_raw = l2_info.get("keywords", [])
            elif isinstance(l2_info, list):
                keywords_raw = l2_info
            else:
                keywords_raw = []

            if not isinstance(keywords_raw, list):
                keywords_raw = []

            l2_label = str(l2)
            l2_keywords = list(dict.fromkeys([l2_label, *[
                str(item).strip()
                for item in keywords_raw
                if isinstance(item, str) and str(item).strip()
            ]]))

            topics = l2_info.get("topics", {}) if isinstance(l2_info, dict) else {}
            if isinstance(topics, dict) and topics:
                for l3, l3_info in topics.items():
                    if isinstance(l3_info, dict):
                        l3_keywords_raw = l3_info.get("keywords", [])
                    else:
                        l3_keywords_raw = []
                    if not isinstance(l3_keywords_raw, list):
                        l3_keywords_raw = []
                    l3_label = str(l3)
                    keywords = list(dict.fromkeys([l3_label, *[
                        str(item).strip()
                        for item in l3_keywords_raw
                        if isinstance(item, str) and str(item).strip()
                    ], *l2_keywords]))
                    for keyword in keywords:
                        if not keyword:
                            continue
                        phrase = tokenize_phrase(keyword)
                        rx = re.compile(rf"(?<!\w){phrase}(?!\w)")
                        patterns.append((str(l1), l2_label, l3_label, rx))
            else:
                for keyword in l2_keywords:
                    if not keyword:
                        continue
                    phrase = tokenize_phrase(keyword)
                    rx = re.compile(rf"(?<!\w){phrase}(?!\w)")
                    patterns.append((str(l1), l2_label, l2_label, rx))
    if not patterns:
        raise ValueError("No taxonomy patterns found. Check your taxonomy.yaml structure.")
    return patterns


def extract_doc_id(doc: Dict[str, Any]) -> Optional[str]:
    for key in ("doc_id", "document_id", "id"):
        if key in doc and isinstance(doc[key], str) and doc[key].strip():
            return doc[key].strip()
    meta = doc.get("metadata")
    if isinstance(meta, dict):
        for key in ("doc_id", "document_id", "id"):
            if key in meta and isinstance(meta[key], str) and meta[key].strip():
                return meta[key].strip()
    return None


def extract_text(doc: dict) -> str:
    val = doc.get("processed_text")
    if isinstance(val, str) and val.strip():
        return val

    for key in ("text", "clean_text", "content", "document_text", "text_clean", "body", "chunk_text"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val

    return ""



def assign_taxonomy_to_text(
    text: str,
    patterns: List[Tuple[str, str, str, re.Pattern]],
    max_hits: int = 1,
) -> Tuple[Optional[str], Optional[str], Optional[str], List[Dict[str, Any]]]:
    """
    Baseline: choose the (l1,l2,l3) with the most matches.
    If tie, earliest match wins.
    """
    t = normalize_text(text)
    scored: List[Tuple[int, int, str, str, str, str]] = []  # (count, first_pos, l1, l2, l3, matched_span)

    for l1, l2, l3, rx in patterns:
        matches = list(rx.finditer(t))
        if not matches:
            continue
        count = len(matches)
        first_pos = matches[0].start()
        matched_span = t[matches[0].start() : matches[0].end()]
        scored.append((count, first_pos, l1, l2, l3, matched_span))

    if not scored:
        return None, None, None, []

    scored.sort(key=lambda x: (-x[0], x[1]))  # most hits, then earliest appearance
    top = scored[:max_hits]

    debug_hits = [
        {
            "level_1": l1,
            "level_2": l2,
            "level_3": l3,
            "match_count": count,
            "first_pos": first_pos,
            "first_match": span,
        }
        for (count, first_pos, l1, l2, l3, span) in top
    ]

    _, _, best_l1, best_l2, best_l3, _ = top[0]
    return best_l1, best_l2, best_l3, debug_hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/clean_documents.jsonl", help="Input JSONL path")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml", help="Taxonomy YAML path")
    ap.add_argument("--output", default="outputs/taxonomy/taxonomy_assignments.jsonl", help="Output JSONL path")
    ap.add_argument("--store_debug", action="store_true", help="Store top match debug info per document")
    ap.add_argument("--max_debug_hits", type=int, default=3, help="How many top hits to store in debug")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not os.path.exists(args.taxonomy):
        raise FileNotFoundError(f"Taxonomy config not found: {args.taxonomy}")

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        tax = yaml.safe_load(f)

    patterns = compile_taxonomy_patterns(tax)

    out_records = []
    missing_id = 0
    empty_text = 0
    unassigned = 0

    for doc in read_jsonl(args.input):
        doc_id = extract_doc_id(doc)
        if not doc_id:
            missing_id += 1
            doc_id = f"unknown_{missing_id}"

        text = extract_text(doc)
        print(doc_id, len(text))
        if not text.strip():
            empty_text += 1

        l1, l2, l3, debug_hits = assign_taxonomy_to_text(
            text=text,
            patterns=patterns,
            max_hits=max(1, args.max_debug_hits),
        )

        if l1 is None:
            unassigned += 1

        rec: Dict[str, Any] = {
            "doc_id": doc_id,
            "taxonomy": {
                "level_1": l1,
                "level_2": l2,
                "level_3": l3,
            },
            "status": "assigned" if l1 is not None else "unassigned",
        }

        if args.store_debug:
            rec["debug"] = {
                "top_hits": debug_hits,
                "note": "Baseline keyword matching; refine with embeddings later if needed.",
            }

        out_records.append(rec)

    write_jsonl(args.output, out_records)

    print("Taxonomy assignment complete")
    print(f"Input:   {args.input}")
    print(f"Taxonomy:{args.taxonomy}")
    print(f"Output:  {args.output}")
    print(f"Stats: total={len(out_records)} missing_id={missing_id} empty_text={empty_text} unassigned={unassigned}")


if __name__ == "__main__":
    # pip install pyyaml
    main()
