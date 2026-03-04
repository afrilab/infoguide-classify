
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional

import yaml


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_clean_text(doc: Dict[str, Any]) -> str:
  
    for key in ["processed_text", "clean_text", "cleaned_text", "text", "raw_text"]:
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""



_blankline_re = re.compile(r"\n\s*\n+", re.MULTILINE)

def split_into_paragraphs(text: str) -> List[str]:
    parts = _blankline_re.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


_sentence_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _sentence_re.split(text)
    return [s.strip() for s in parts if s.strip()]


def _tokens(s: str) -> List[str]:
    # token ~= word
    return s.split()


def _join(tokens: List[str]) -> str:
    return " ".join(tokens).strip()


def chunk_token_stream(tokens: List[str], max_tokens: int, overlap_tokens: int, min_tokens: int) -> List[str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))

    out: List[str] = []
    i = 0
    n = len(tokens)

    while i < n:
        j = min(i + max_tokens, n)
        chunk = tokens[i:j]
        if len(chunk) >= min_tokens:
            out.append(_join(chunk))
        if j == n:
            break
        i = j - overlap_tokens

    return out


def chunk_from_units(units: List[str], max_tokens: int, overlap_tokens: int, min_tokens: int) -> List[str]:
  
    out: List[str] = []
    current: List[str] = []
    current_tokens: List[str] = []

    def flush_with_overlap():
        nonlocal current, current_tokens
        if len(current_tokens) >= min_tokens:
            out.append(_join(current_tokens))
        if overlap_tokens > 0:
            current_tokens = current_tokens[-overlap_tokens:]
            current = [_join(current_tokens)] if current_tokens else []
        else:
            current_tokens = []
            current = []

    for u in units:
        utoks = _tokens(u)
        if not utoks:
            continue

        if len(utoks) > max_tokens:
            if current_tokens:
                flush_with_overlap()
            big_chunks = chunk_token_stream(utoks, max_tokens, overlap_tokens, min_tokens)
            out.extend(big_chunks)
            current, current_tokens = [], []
            continue

        if len(current_tokens) + len(utoks) > max_tokens and current_tokens:
            flush_with_overlap()

        current_tokens.extend(utoks)

    if current_tokens:
        if len(current_tokens) >= min_tokens:
            out.append(_join(current_tokens))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chunking.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    in_path = Path(cfg["input_path"])
    out_path = Path(cfg["output_path"])

    chcfg = cfg.get("chunking", {})
    strategy = chcfg.get("strategy", "paragraph").strip().lower()
    max_tokens = int(chcfg.get("max_tokens", 512))
    overlap_tokens = int(chcfg.get("overlap_tokens", 50))
    min_tokens = int(chcfg.get("min_tokens", 60))  

    print(f"Reading:  {in_path}")
    docs = load_jsonl(in_path)
    print(f"Loaded {len(docs)} cleaned docs")

    out_rows: List[Dict[str, Any]] = []
    total_chunks = 0
    empty_docs = 0

    for d in docs:
        doc_id = d.get("document_id") or d.get("doc_id")
        filename = d.get("filename", "?")
        source = d.get("source", "?")

        clean_text = pick_clean_text(d).strip()
        if not clean_text:
            empty_docs += 1
            print(f"[SKIP] {doc_id} has no clean text")
            continue

        if strategy == "paragraph":
            units = split_into_paragraphs(clean_text)
            chunks = chunk_from_units(units, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens)

        elif strategy == "sentence":
            units = split_into_sentences(clean_text)
            chunks = chunk_from_units(units, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens)

        elif strategy == "fixed":
            toks = _tokens(clean_text)
            chunks = chunk_token_stream(toks, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens)

        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}. Use paragraph|sentence|fixed")

        for i, ch in enumerate(chunks):
            out_rows.append({
                "document_id": doc_id,
                "chunk_id": f"{doc_id}_{i:04d}",
                "source": source,
                "filename": filename,
                "text": ch,
                "chunk_meta": {
                    "strategy": strategy,
                    "max_tokens": max_tokens,
                    "overlap_tokens": overlap_tokens,
                }
            })

        total_chunks += len(chunks)
        print(f"[CHUNK] {doc_id} | {filename} | chunks={len(chunks)}")

    write_jsonl(out_path, out_rows)
    print("\n--- CHUNK SUMMARY ---")
    print(f"Docs: {len(docs)}")
    print(f"Empty docs skipped: {empty_docs}")
    print(f"Total chunks: {total_chunks}")
    print(f"Written: {out_path}")
    print("Done")


if __name__ == "__main__":
    main()
