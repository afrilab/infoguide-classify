#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def ensure_src_on_path() -> None:
  
    here = Path(__file__).resolve()
    project_root = here.parents[1]         
    src_dir = project_root / "src"         
    sys.path.insert(0, str(src_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg_path = args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ensure_src_on_path()
    from run import classify_documents

    # Optional: merge anonymized_text from a separate anonymizer output file.
    # If input_path already points at that file, classify it directly.
    anon_path = cfg.get("anonymized_docs_path")
    input_path = cfg.get("input_path")
    if (
        anon_path
        and input_path
        and Path(anon_path).exists()
        and Path(anon_path).resolve() != Path(input_path).resolve()
    ):
        import json
        anon_map = {}
        with open(anon_path, "r", encoding="utf-8") as af:
            for line in af:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    anon_map[rec["doc_id"]] = rec.get("anonymized_text", "")
        if anon_map:
            from file_io import read_jsonl
            rows_in = read_jsonl(cfg["input_path"])
            for row in rows_in:
                doc_id = row.get("doc_id")
                if doc_id in anon_map:
                    row["anonymized_text"] = anon_map[doc_id]
            # Patch cfg so classify_documents reads from memory
            import tempfile
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
            for r in rows_in:
                tmp.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.close()
            cfg = dict(cfg)
            cfg["input_path"] = tmp.name
            print(f"[CLASSIFY] Merged anonymized_text for {len(anon_map)} docs")

    rows = classify_documents(cfg)
    print(f"[CLASSIFY] Read/Wrote: {len(rows)} docs")
    print(f"[CLASSIFY] Output: {cfg['output_path']}")


if __name__ == "__main__":
    main()
