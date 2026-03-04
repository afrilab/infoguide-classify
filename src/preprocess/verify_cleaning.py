from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_jsonl(path: Path, encoding: str="utf-8") -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding=encoding) as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out

def get_doc_id(d: Dict[str, Any]) -> Optional[str]:
    return d.get("doc_id") or d.get("document_id") or d.get("id")

def get_text(d: Dict[str, Any]) -> str:
    for k in ["processed_text","clean_text","cleaned_text","text","raw_text"]:
        v=d.get(k)
        if isinstance(v,str) and v.strip():
            return v
    return ""

def word_count(t: str) -> int:
    return len(t.split())

def top_repeated_lines(text: str, min_len: int, topk: int) -> List[Tuple[str,int]]:
    lines=[ln.strip() for ln in text.splitlines()]
    lines=[ln for ln in lines if len(ln)>=min_len]
    c=Counter(lines)
    return c.most_common(topk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/verify.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    raw_path  = Path(cfg["raw_path"])
    clean_path= Path(cfg["clean_path"])
    encoding  = cfg.get("encoding","utf-8")
    min_text_len = int(cfg.get("min_text_length",0))

    checks = cfg.get("checks", {})
    drop_warn_pct = float(checks.get("drop_warn_pct", 85))
    repeat_min_len = int(checks.get("repeat_min_len", 30))
    repeat_topk = int(checks.get("repeat_topk", 8))
    repeat_warn_count = int(checks.get("repeat_warn_count", 10))

    raw_docs = load_jsonl(raw_path, encoding)
    clean_docs = load_jsonl(clean_path, encoding)

    raw_map = {get_doc_id(d): d for d in raw_docs if get_doc_id(d)}
    clean_map= {get_doc_id(d): d for d in clean_docs if get_doc_id(d)}

    matched = sorted(set(raw_map.keys()) & set(clean_map.keys()))
    missing_clean = sorted(set(raw_map.keys()) - set(clean_map.keys()))

    print(f"Loaded raw:   {len(raw_docs)} from {raw_path}")
    print(f"Loaded clean: {len(clean_docs)} from {clean_path}")
    print(f"Matched ids:  {len(matched)}")
    if missing_clean:
        print(f"[WARN] raw has ids missing in clean: {missing_clean[:5]}{'...' if len(missing_clean)>5 else ''}")
    print()

    suspicious = 0
    for doc_id in matched:
        r = raw_map[doc_id]
        c = clean_map[doc_id]
        rt = (r.get("raw_text") or get_text(r) or "").strip()
        ct = (get_text(c) or "").strip()

        rw = word_count(rt)
        cw = word_count(ct)
        drop = (1 - (cw / rw))*100 if rw>0 else 0

        print(f"{doc_id} | raw_words={rw} clean_words={cw} drop={drop:.1f}%")

        bad = False
        if rw>0 and drop >= drop_warn_pct:
            print(f"  [WARN] too much removed (>{drop_warn_pct:.0f}%)")
            bad = True
        if min_text_len and len(ct) < min_text_len:
            print(f"  [WARN] cleaned text length {len(ct)} < {min_text_len}")
            bad = True

        reps = top_repeated_lines(ct, repeat_min_len, repeat_topk)
        for line, cnt in reps:
            flag = " <== POSSIBLE BOILERPLATE" if cnt >= repeat_warn_count else ""
            print(f"    ({cnt}x){flag} {line[:120]}")
            if cnt >= repeat_warn_count:
                bad = True

        if bad:
            suspicious += 1
        print()

    print("---- SUMMARY ----")
    print(f"Total matched: {len(matched)}")
    print(f"Suspicious:    {suspicious}")
    print(" Done")

if __name__ == "__main__":
    main()
