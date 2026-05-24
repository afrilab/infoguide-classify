"""
assign_taxonomy_evidence.py

Chunk-evidence taxonomy assignment for InfoGuide.

This assigner is intended to improve on whole-document keyword or embedding
matching by:
1) scoring every chunk against complete taxonomy paths,
2) avoiding greedy level-by-level choices,
3) aggregating only the strongest pieces of evidence for each document, and
4) exposing the chunks that drove the final decision.

Input : data/processed/clean_documents.jsonl
Chunks: data/processed/chunks.jsonl
Config: configs/taxonomy.yaml
Output: data/outputs/taxonomy_assignments_evidence.jsonl

Usage:
  python src/taxonomy/assign_taxonomy_evidence.py \
    --input data/processed/clean_documents.jsonl \
    --chunks data/processed/chunks.jsonl \
    --taxonomy configs/taxonomy.yaml \
    --output data/outputs/taxonomy_assignments_evidence.jsonl \
    --store_debug

Install:
  pip install sentence-transformers pyyaml torch
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class TaxonomyPath:
    level_1: str
    level_2: str
    level_3: str
    description: str
    keywords: Tuple[str, ...]
    rep_text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    source: Optional[str] = None
    filename: Optional[str] = None


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9-]+", normalize_text(text))


def vectorize_text(text: str) -> Dict[str, float]:
    tokens = tokenize(text)
    counts: Dict[str, float] = {}

    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0

    for left, right in zip(tokens, tokens[1:]):
        bigram = f"{left} {right}"
        counts[bigram] = counts.get(bigram, 0.0) + 1.5

    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm <= 0:
        return {}
    return {term: value / norm for term, value in counts.items()}


def sparse_cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(weight * vec_b.get(term, 0.0) for term, weight in vec_a.items())


def load_sentence_transformer(model_name: str) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    if model_name.strip().lower() in {"local", "local_ngram", "none", "disabled"}:
        return None, None, "embedding backend disabled by --model"
    try:
        from sentence_transformers import SentenceTransformer, util

        return SentenceTransformer(model_name), util, None
    except Exception as e:  # pragma: no cover - depends on local ML environment
        return None, None, f"{type(e).__name__}: {e}"


def extract_doc_id(doc: Dict[str, Any]) -> Optional[str]:
    for key in ("doc_id", "document_id", "id"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    meta = doc.get("metadata")
    if isinstance(meta, dict):
        for key in ("doc_id", "document_id", "id"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def extract_text(doc: Dict[str, Any]) -> str:
    val = doc.get("processed_text")
    if isinstance(val, str) and val.strip():
        return val

    for key in ("text", "clean_text", "content", "document_text", "text_clean", "body", "chunk_text"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val

    return ""


def phrase_occurrences(text: str, phrase: str) -> int:
    phrase = phrase.strip().lower()
    if not phrase:
        return 0
    pattern = re.escape(phrase).replace(r"\ ", r"\s+")
    return len(re.findall(rf"(?<!\w){pattern}(?!\w)", text))


def keyword_score(text: str, keywords: Sequence[str]) -> Tuple[float, List[Dict[str, Any]]]:
    normalized = normalize_text(text)
    hits: List[Dict[str, Any]] = []
    weighted_hits = 0.0

    for kw in keywords:
        count = phrase_occurrences(normalized, kw)
        if count <= 0:
            continue
        token_count = max(1, len(kw.split()))
        weight = 1.0 + math.log1p(token_count)
        weighted_hits += count * weight
        hits.append({"keyword": kw, "count": count, "weight": round(weight, 4)})

    score = weighted_hits / (weighted_hits + 3.0) if weighted_hits > 0 else 0.0
    hits.sort(key=lambda x: (-x["count"], x["keyword"]))
    return score, hits


def build_representation_text(l1: str, l2: str, l3: str, description: str, keywords: Sequence[str]) -> str:
    parts = [
        f"Taxonomy path: {l1} > {l2} > {l3}.",
        f"Document label: {l3}.",
        f"Document family: {l2}.",
        f"Taxonomy domain: {l1}.",
    ]
    if description:
        parts.append(f"Definition: {description}.")
    if keywords:
        parts.append("Signals and related terms: " + ", ".join(keywords) + ".")
    return " ".join(parts)


def load_taxonomy_paths(taxonomy_yaml: Dict[str, Any]) -> List[TaxonomyPath]:
    if "taxonomy" not in taxonomy_yaml or not isinstance(taxonomy_yaml["taxonomy"], dict):
        raise ValueError("taxonomy.yaml must contain a top-level key 'taxonomy' with a nested mapping.")

    paths: List[TaxonomyPath] = []
    for l1, l2_map in taxonomy_yaml["taxonomy"].items():
        if not isinstance(l2_map, dict):
            continue
        for l2, l2_info in l2_map.items():
            description = ""
            keywords_raw: Any = []

            if isinstance(l2_info, dict):
                description = str(l2_info.get("description", "")).strip()
                keywords_raw = l2_info.get("keywords", [])
            elif isinstance(l2_info, list):
                keywords_raw = l2_info

            if not isinstance(keywords_raw, list):
                keywords_raw = []

            keywords = tuple(
                str(item).strip()
                for item in keywords_raw
                if isinstance(item, str) and str(item).strip()
            )

            topics = l2_info.get("topics", {}) if isinstance(l2_info, dict) else {}
            if isinstance(topics, dict) and topics:
                for l3, l3_info in topics.items():
                    if isinstance(l3_info, dict):
                        l3_description = str(l3_info.get("description", "")).strip()
                        l3_keywords_raw = l3_info.get("keywords", [])
                    else:
                        l3_description = ""
                        l3_keywords_raw = []
                    if not isinstance(l3_keywords_raw, list):
                        l3_keywords_raw = []
                    l3_keywords = tuple(
                        str(item).strip()
                        for item in l3_keywords_raw
                        if isinstance(item, str) and str(item).strip()
                    )
                    expanded_keywords = tuple(dict.fromkeys((str(l3), *l3_keywords, str(l2), *keywords)))
                    path_description = " ".join(x for x in (l3_description, description) if x)
                    paths.append(
                        TaxonomyPath(
                            level_1=str(l1),
                            level_2=str(l2),
                            level_3=str(l3),
                            description=path_description,
                            keywords=expanded_keywords,
                            rep_text=build_representation_text(
                                str(l1),
                                str(l2),
                                str(l3),
                                path_description,
                                expanded_keywords,
                            ),
                        )
                    )
            else:
                label = str(l2)
                expanded_keywords = tuple(dict.fromkeys((label, *keywords)))
                paths.append(
                    TaxonomyPath(
                        level_1=str(l1),
                        level_2=label,
                        level_3=label,
                        description=description,
                        keywords=expanded_keywords,
                        rep_text=build_representation_text(
                            str(l1),
                            label,
                            label,
                            description,
                            expanded_keywords,
                        ),
                    )
                )

    if not paths:
        raise ValueError("No valid taxonomy paths found in taxonomy.yaml.")
    return paths


def load_chunks(path: str) -> Dict[str, List[Chunk]]:
    chunks_by_doc: Dict[str, List[Chunk]] = {}
    if not path or not os.path.exists(path):
        return chunks_by_doc

    for rec in read_jsonl(path):
        doc_id = rec.get("document_id") or rec.get("doc_id")
        text = rec.get("text") or rec.get("chunk_text") or rec.get("processed_text")
        if not isinstance(doc_id, str) or not doc_id.strip() or not isinstance(text, str) or not text.strip():
            continue
        chunk = Chunk(
            chunk_id=str(rec.get("chunk_id") or f"{doc_id}_{len(chunks_by_doc.get(doc_id, [])):04d}"),
            document_id=doc_id.strip(),
            text=text.strip(),
            source=rec.get("source") if isinstance(rec.get("source"), str) else None,
            filename=rec.get("filename") if isinstance(rec.get("filename"), str) else None,
        )
        chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

    return chunks_by_doc


def fallback_chunks(doc_id: str, text: str, max_words: int = 380, overlap_words: int = 60) -> List[Chunk]:
    words = normalize_text(text).split()
    if not words:
        return []

    chunks: List[Chunk] = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if not chunk_words:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_fallback_{len(chunks):04d}",
                document_id=doc_id,
                text=" ".join(chunk_words),
            )
        )
        if start + max_words >= len(words):
            break
    return chunks


def summarize_candidate(
    path: TaxonomyPath,
    chunk_scores: List[Dict[str, Any]],
    top_k: int,
    support_threshold: float,
) -> Dict[str, Any]:
    ranked = sorted(chunk_scores, key=lambda x: x["score"], reverse=True)
    top = ranked[: max(1, top_k)]
    max_score = top[0]["score"] if top else 0.0
    top_mean = sum(item["score"] for item in top) / len(top) if top else 0.0
    support_count = sum(1 for item in ranked if item["score"] >= support_threshold)
    support_ratio = support_count / max(1, len(ranked))

    aggregate = (0.55 * max_score) + (0.35 * top_mean) + (0.10 * min(1.0, support_ratio * 4.0))
    return {
        "path": path,
        "aggregate_score": aggregate,
        "max_chunk_score": max_score,
        "top_chunk_mean": top_mean,
        "support_count": support_count,
        "support_ratio": support_ratio,
        "evidence": top,
    }


def classify_document(
    doc_id: str,
    text: str,
    chunks: List[Chunk],
    paths: List[TaxonomyPath],
    model: Optional[Any],
    sentence_util: Optional[Any],
    path_embeddings: Any,
    path_vectors: List[Dict[str, float]],
    alpha: float,
    top_k_chunks: int,
    support_threshold: float,
    min_score: float,
    min_margin: float,
) -> Dict[str, Any]:
    if not text.strip() and not chunks:
        return {
            "taxonomy": {"level_1": None, "level_2": None, "level_3": None},
            "status": "unassigned",
            "confidence": 0.0,
            "debug": {"reason": "empty_text"},
        }

    usable_chunks = chunks or fallback_chunks(doc_id, text)
    if not usable_chunks:
        return {
            "taxonomy": {"level_1": None, "level_2": None, "level_3": None},
            "status": "unassigned",
            "confidence": 0.0,
            "debug": {"reason": "no_usable_chunks"},
        }

    if model is not None and sentence_util is not None and path_embeddings is not None:
        chunk_texts = [chunk.text for chunk in usable_chunks]
        chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True, normalize_embeddings=True)
        cosine_matrix = sentence_util.cos_sim(chunk_embeddings, path_embeddings)
        backend = "sentence_transformer"
    else:
        chunk_vectors = [vectorize_text(chunk.text) for chunk in usable_chunks]
        cosine_matrix = None
        backend = "local_ngram"

    per_path_scores: List[List[Dict[str, Any]]] = [[] for _ in paths]
    for chunk_idx, chunk in enumerate(usable_chunks):
        for path_idx, path in enumerate(paths):
            if cosine_matrix is not None:
                embedding_score = (float(cosine_matrix[chunk_idx][path_idx].item()) + 1.0) / 2.0
            else:
                embedding_score = sparse_cosine(chunk_vectors[chunk_idx], path_vectors[path_idx])
            kw_score, kw_hits = keyword_score(chunk.text, path.keywords)
            score = (alpha * embedding_score) + ((1.0 - alpha) * kw_score)
            per_path_scores[path_idx].append(
                {
                    "chunk_id": chunk.chunk_id,
                    "score": score,
                    "embedding_score": embedding_score,
                    "keyword_score": kw_score,
                    "keyword_hits": kw_hits[:5],
                    "snippet": normalize_text(chunk.text)[:420],
                }
            )

    ranked = [
        summarize_candidate(path, scores, top_k_chunks, support_threshold)
        for path, scores in zip(paths, per_path_scores)
    ]
    ranked.sort(key=lambda item: item["aggregate_score"], reverse=True)

    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    best_path: TaxonomyPath = best["path"]
    margin = best["aggregate_score"] - (runner_up["aggregate_score"] if runner_up else 0.0)
    confidence = max(0.0, min(1.0, best["aggregate_score"]))

    status = "assigned"
    reason = None
    if best["aggregate_score"] < min_score:
        status = "unassigned"
        reason = "low_score"
    elif margin < min_margin:
        status = "needs_review"
        reason = "low_margin"

    taxonomy = {
        "level_1": best_path.level_1 if status != "unassigned" else None,
        "level_2": best_path.level_2 if status != "unassigned" else None,
        "level_3": best_path.level_3 if status != "unassigned" else None,
    }

    def public_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
        path = item["path"]
        return {
            "taxonomy": {
                "level_1": path.level_1,
                "level_2": path.level_2,
                "level_3": path.level_3,
            },
            "aggregate_score": round(float(item["aggregate_score"]), 6),
            "max_chunk_score": round(float(item["max_chunk_score"]), 6),
            "top_chunk_mean": round(float(item["top_chunk_mean"]), 6),
            "support_count": item["support_count"],
            "support_ratio": round(float(item["support_ratio"]), 6),
            "evidence": [
                {
                    "chunk_id": ev["chunk_id"],
                    "score": round(float(ev["score"]), 6),
                    "embedding_score": round(float(ev["embedding_score"]), 6),
                    "keyword_score": round(float(ev["keyword_score"]), 6),
                    "keyword_hits": ev["keyword_hits"],
                    "snippet": ev["snippet"],
                }
                for ev in item["evidence"]
            ],
        }

    debug = {
        "reason": reason,
        "backend": backend,
        "margin": round(float(margin), 6),
        "chunk_count": len(usable_chunks),
        "top_candidates": [public_candidate(item) for item in ranked[:5]],
    }

    return {
        "taxonomy": taxonomy,
        "status": status,
        "confidence": confidence,
        "debug": debug,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/clean_documents.jsonl", help="Input document JSONL path")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl", help="Optional chunk JSONL path")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml", help="Taxonomy YAML path")
    ap.add_argument(
        "--output",
        default="data/outputs/taxonomy_assignments_evidence.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    ap.add_argument("--alpha", type=float, default=0.68, help="Embedding score weight")
    ap.add_argument("--top_k_chunks", type=int, default=4, help="Evidence chunks to aggregate per taxonomy path")
    ap.add_argument("--support_threshold", type=float, default=0.32, help="Chunk score counted as supporting evidence")
    ap.add_argument("--min_score", type=float, default=0.30, help="Minimum aggregate score for assignment")
    ap.add_argument("--min_margin", type=float, default=0.005, help="Minimum gap between best and second-best path")
    ap.add_argument("--store_debug", action="store_true", help="Store candidate and evidence chunk details")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not os.path.exists(args.taxonomy):
        raise FileNotFoundError(f"Taxonomy config not found: {args.taxonomy}")

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy_yaml = yaml.safe_load(f)

    print(f"Loading taxonomy: {args.taxonomy}")
    paths = load_taxonomy_paths(taxonomy_yaml)
    print(f"Loaded {len(paths)} taxonomy paths")

    print(f"Loading chunks: {args.chunks}")
    chunks_by_doc = load_chunks(args.chunks)
    print(f"Loaded chunks for {len(chunks_by_doc)} documents")

    print(f"Loading embedding model: {args.model}")
    model, sentence_util, model_error = load_sentence_transformer(args.model)
    if model is not None:
        path_embeddings = model.encode(
            [path.rep_text for path in paths],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        path_vectors: List[Dict[str, float]] = []
        print("Using backend: sentence_transformer")
    else:
        path_embeddings = None
        path_vectors = [vectorize_text(path.rep_text + " " + " ".join(path.keywords)) for path in paths]
        print("Using backend: local_ngram")
        print(f"SentenceTransformer unavailable; fallback reason: {model_error}")

    out_records: List[Dict[str, Any]] = []
    stats = {"assigned": 0, "needs_review": 0, "unassigned": 0, "missing_id": 0, "fallback_chunked": 0}

    for idx, doc in enumerate(read_jsonl(args.input), start=1):
        doc_id = extract_doc_id(doc)
        if not doc_id:
            stats["missing_id"] += 1
            doc_id = f"unknown_{stats['missing_id']}"

        text = extract_text(doc)
        doc_chunks = chunks_by_doc.get(doc_id, [])
        if not doc_chunks:
            stats["fallback_chunked"] += 1

        result = classify_document(
            doc_id=doc_id,
            text=text,
            chunks=doc_chunks,
            paths=paths,
            model=model,
            sentence_util=sentence_util,
            path_embeddings=path_embeddings,
            path_vectors=path_vectors,
            alpha=args.alpha,
            top_k_chunks=max(1, args.top_k_chunks),
            support_threshold=args.support_threshold,
            min_score=args.min_score,
            min_margin=args.min_margin,
        )

        rec: Dict[str, Any] = {
            "doc_id": doc_id,
            "taxonomy": result["taxonomy"],
            "status": result["status"],
            "confidence": round(float(result["confidence"]), 6),
            "method": "chunk_evidence_embedding_keyword",
        }
        if args.store_debug:
            rec["debug"] = result["debug"]

        out_records.append(rec)
        stats[result["status"]] += 1

        if idx % 25 == 0:
            print(f"Processed {idx} documents...")

    write_jsonl(args.output, out_records)

    print("\nTaxonomy assignment complete")
    print(f"Input:   {args.input}")
    print(f"Chunks:  {args.chunks}")
    print(f"Taxonomy:{args.taxonomy}")
    print(f"Output:  {args.output}")
    print(
        "Stats: "
        f"total={len(out_records)} "
        f"assigned={stats['assigned']} "
        f"needs_review={stats['needs_review']} "
        f"unassigned={stats['unassigned']} "
        f"missing_id={stats['missing_id']} "
        f"fallback_chunked={stats['fallback_chunked']}"
    )


if __name__ == "__main__":
    main()
