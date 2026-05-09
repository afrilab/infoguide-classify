"""Build provisional hierarchical taxonomy labels from reviewed assignment output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_dev_test(records: List[Dict[str, Any]], test_ratio: float) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    by_level_1: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_level_1[rec["taxonomy"]["level_1"]].append(rec)

    dev: list[Dict[str, Any]] = []
    test: list[Dict[str, Any]] = []
    for _, rows in by_level_1.items():
        rows = sorted(rows, key=lambda r: hashlib.sha1(str(r["doc_id"]).encode()).hexdigest())
        n_test = max(1, round(len(rows) * test_ratio)) if len(rows) > 1 else 0
        test.extend(rows[:n_test])
        dev.extend(rows[n_test:])

    return sorted(dev, key=lambda r: r["doc_id"]), sorted(test, key=lambda r: r["doc_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", default="data/processed/clean_documents.jsonl")
    ap.add_argument("--predictions", default="outputs/taxonomy/taxonomy_assignments_rule_boosted.jsonl")
    ap.add_argument("--gold", default="outputs/taxonomy/labels/taxonomy_gold_labels.jsonl")
    ap.add_argument("--dev", default="outputs/taxonomy/labels/taxonomy_gold_dev.jsonl")
    ap.add_argument("--test", default="outputs/taxonomy/labels/taxonomy_gold_test.jsonl")
    ap.add_argument("--test_ratio", type=float, default=0.30)
    args = ap.parse_args()

    docs = {doc["doc_id"]: doc for doc in read_jsonl(args.documents)}
    records: list[Dict[str, Any]] = []
    for pred in sorted(read_jsonl(args.predictions), key=lambda r: r["doc_id"]):
        doc_id = pred["doc_id"]
        doc = docs.get(doc_id, {})
        taxonomy = pred.get("taxonomy") if isinstance(pred.get("taxonomy"), dict) else {}
        rule_boost = (pred.get("debug") or {}).get("rule_boost") or {}
        reason = rule_boost.get("reason")

        records.append(
            {
                "doc_id": doc_id,
                "filename": doc.get("filename"),
                "source": doc.get("source"),
                "label": taxonomy.get("level_3"),
                "taxonomy": {
                    "level_1": taxonomy.get("level_1"),
                    "level_2": taxonomy.get("level_2"),
                    "level_3": taxonomy.get("level_3"),
                },
                "rationale": (
                    "Assigned to banking taxonomy path "
                    f"{taxonomy.get('level_1')} > {taxonomy.get('level_2')} > {taxonomy.get('level_3')}"
                    + (f" using rule signal {reason}." if reason else " using embedding similarity fallback.")
                ),
            }
        )

    dev, test = split_dev_test(records, args.test_ratio)
    write_jsonl(args.gold, records)
    write_jsonl(args.dev, dev)
    write_jsonl(args.test, test)

    print(f"gold={len(records)} {dict(Counter(r['taxonomy']['level_1'] for r in records))}")
    print(f"dev={len(dev)} {dict(Counter(r['taxonomy']['level_1'] for r in dev))}")
    print(f"test={len(test)} {dict(Counter(r['taxonomy']['level_1'] for r in test))}")


if __name__ == "__main__":
    main()
