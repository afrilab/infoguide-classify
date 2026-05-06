# src/infoguide/classification/file_io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {e}") from e
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pick_text(row: Dict[str, Any], fields_priority: List[str]) -> Tuple[str, str]:
   
    for field in fields_priority:
        val = row.get(field, None)
        if isinstance(val, str) and val.strip():
            return val.strip(), field
    return "", ""
