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

    rows = classify_documents(cfg)
    print(f"[CLASSIFY] Read/Wrote: {len(rows)} docs")
    print(f"[CLASSIFY] Output: {cfg['output_path']}")


if __name__ == "__main__":
    main()
