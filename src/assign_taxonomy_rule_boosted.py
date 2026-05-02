"""
Rule-boosted taxonomy assignment.

This post-processor applies conservative document-genre rules on top of an
existing taxonomy output. It is intended for the six document-type labels in
configs/taxonomy.yaml.

Usage:
  python src/assign_taxonomy_rule_boosted.py \
    --documents data/processed/clean_documents.jsonl \
    --base_predictions data/outputs/taxonomy_assignments_embeddings.jsonl \
    --output data/outputs/taxonomy_assignments_rule_boosted.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, Optional, Tuple


POLICY = "Policy / Procedure / Contract Documents"
REPORT = "Reports (Financial / Incident / Audit)"
FORM = "Forms / Structured Documents"


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
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_any(text: str, patterns: Iterable[str]) -> Optional[str]:
    for pattern in patterns:
        if re.search(pattern, text):
            return pattern
    return None


def extract_text(doc: Dict[str, Any], max_chars: int = 6000) -> str:
    for key in ("processed_text", "text", "clean_text", "raw_text"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value[:max_chars]
    return ""


def rule_label(doc: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    filename = normalize(str(doc.get("filename") or ""))
    source = normalize(str(doc.get("source") or ""))
    title_text = normalize(extract_text(doc))
    title_zone = f"{filename} {source} {title_text[:900]}"
    combined = f"{filename} {source} {title_text[:1800]}"

    form_pattern = contains_any(
        title_zone,
        [
            r"\bfw-?4\b",
            r"\bw-?4\b.*withholding certificate",
            r"\bfw-?9\b",
            r"\bw-?9\b.*taxpayer",
            r"\bf1040\b",
            r"\bform 1040\b",
            r"\bisora\s+20\d{2}\s+form",
            r"\bforms?\.pdf\b",
            r"\bstructured document\b",
            r"\bomb no\.",
            r"\bgive form to\b",
            r"\bcheckbox\b",
        ],
    )
    if form_pattern:
        return FORM, f"form_rule:{form_pattern}"

    strong_report_pattern = contains_any(
        title_zone,
        [
            r"\bfinancial sector assessment\b",
            r"\bfsap\b",
            r"\bworking paper\b",
            r"\bwp[_/-]?\d*",
            r"\bannual ?report\b",
            r"\bhighlights report\b",
            r"\baudited financial statements\b",
            r"\bmanagement'?s discussion\b",
            r"\bproxy statement\b",
            r"\bform 8-k\b",
            r"\bcurrent report\b",
            r"\bquarterly earnings\b",
            r"\breports? first-quarter\b",
            r"\bjurisdictions under increased monitoring\b",
            r"\bstaff discussion notes?\b",
            r"\bimf notes?\b",
            r"\bmobile money note\b",
        ],
    )
    if strong_report_pattern:
        return REPORT, f"report_rule:{strong_report_pattern}"

    policy_pattern = contains_any(
        title_zone,
        [
            r"\bguidance\b",
            r"\bguide\b",
            r"\btoolkit\b",
            r"\brisk-based approach\b",
            r"\brba-",
            r"\bhow to\b",
            r"\bmanuals?\b",
            r"\bglossary\b",
            r"\bpricing supplement\b",
            r"\bpricing term sheet\b",
            r"\bterms of the notes\b",
            r"\bprospectus supplement\b",
            r"\btechnical guide\b",
            r"\bprocedures?\b",
            r"\bagreement\b",
            r"\bcontract\b",
        ],
    )
    if policy_pattern:
        return POLICY, f"policy_rule:{policy_pattern}"

    report_pattern = contains_any(
        title_zone,
        [
            r"\bresearch report\b",
            r"\bassessment report\b",
            r"\banalytical note\b",
        ],
    )
    if report_pattern:
        return REPORT, f"report_rule:{report_pattern}"

    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", default="data/processed/clean_documents.jsonl")
    ap.add_argument("--base_predictions", default="data/outputs/taxonomy_assignments_embeddings.jsonl")
    ap.add_argument("--output", default="data/outputs/taxonomy_assignments_rule_boosted.jsonl")
    args = ap.parse_args()

    docs = {doc["doc_id"]: doc for doc in read_jsonl(args.documents)}
    base_predictions = list(read_jsonl(args.base_predictions))

    output = []
    changed = 0
    for pred in base_predictions:
        doc_id = pred.get("doc_id")
        doc = docs.get(doc_id, {})
        label, reason = rule_label(doc)
        rec = dict(pred)
        debug = rec.get("debug") if isinstance(rec.get("debug"), dict) else {}

        if label:
            old_label = None
            taxonomy = rec.get("taxonomy")
            if isinstance(taxonomy, dict):
                old_label = taxonomy.get("level_2")

            rec["taxonomy"] = {
                "level_1": "Document Type",
                "level_2": label,
                "level_3": label,
            }
            rec["status"] = "assigned"
            rec["confidence"] = max(float(rec.get("confidence") or 0.0), 0.85)
            rec["method"] = "rule_boosted_hybrid"
            debug["rule_boost"] = {
                "applied": True,
                "reason": reason,
                "base_label": old_label,
            }
            if old_label != label:
                changed += 1
        else:
            rec["method"] = "rule_boosted_hybrid"
            debug["rule_boost"] = {"applied": False}

        rec["debug"] = debug
        output.append(rec)

    write_jsonl(args.output, output)
    print(f"Wrote {len(output)} records to {args.output}")
    print(f"Rule overrides: {changed}")


if __name__ == "__main__":
    main()
