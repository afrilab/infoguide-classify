"""
assign_taxonomy_embeddings.py

Hierarchical taxonomy assignment for InfoGuide using:
1) embedding similarity
2) keyword matching
3) hybrid scoring

Input : data/processed/clean_documents.jsonl
Config: configs/taxonomy.yaml
Output: data/outputs/taxonomy_assignments_embeddings.jsonl

Usage:
  python src/assign_taxonomy_embeddings.py \
    --input data/processed/clean_documents.jsonl \
    --taxonomy configs/taxonomy.yaml \
    --output data/outputs/taxonomy_assignments_embeddings.jsonl \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --alpha 0.7 \
    --store_debug

Install:
  pip install sentence-transformers pyyaml torch
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from sentence_transformers import SentenceTransformer, util


# ----------------------------
# IO helpers
# ----------------------------

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


# ----------------------------
# Text helpers
# ----------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phrase_occurrences(text: str, phrase: str) -> int:
    """
    Count phrase matches with flexible whitespace.
    """
    phrase = phrase.strip().lower()
    if not phrase:
        return 0
    pattern = re.escape(phrase).replace(r"\ ", r"\s+")
    rx = re.compile(rf"(?<!\w){pattern}(?!\w)")
    return len(rx.findall(text))


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


def extract_text(doc: Dict[str, Any]) -> str:
    val = doc.get("processed_text")
    if isinstance(val, str) and val.strip():
        return val

    for key in ("text", "clean_text", "content", "document_text", "text_clean", "body", "chunk_text"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val

    return ""


# ----------------------------
# Taxonomy structures
# ----------------------------

@dataclass
class Level3Node:
    level_1: str
    level_2: str
    level_3: str
    keywords: List[str]
    rep_text: str
    embedding: Any = None


@dataclass
class Level2Node:
    level_1: str
    level_2: str
    keywords: List[str]
    rep_text: str
    embedding: Any = None
    children: List[Level3Node] = field(default_factory=list)


@dataclass
class Level1Node:
    level_1: str
    keywords: List[str]
    rep_text: str
    embedding: Any = None
    children: List[Level2Node] = field(default_factory=list)


def build_representation_text(
    level_1: str,
    level_2: Optional[str],
    level_3: Optional[str],
    keywords: List[str],
) -> str:
    """
    Build a natural-language representation for embedding.
    """
    parts: List[str] = []
    parts.append(f"Domain category: {level_1}.")
    if level_2:
        parts.append(f"Subtopic group: {level_2}.")
    if level_3:
        parts.append(f"Granular subtopic: {level_3}.")
    if keywords:
        parts.append("Related terms: " + ", ".join(keywords) + ".")
    return " ".join(parts)


def load_taxonomy(taxonomy_yaml: Dict[str, Any]) -> List[Level1Node]:
    if "taxonomy" not in taxonomy_yaml or not isinstance(taxonomy_yaml["taxonomy"], dict):
        raise ValueError("taxonomy.yaml must contain a top-level key 'taxonomy' with a nested mapping.")

    level1_nodes: List[Level1Node] = []

    for l1, l2_map in taxonomy_yaml["taxonomy"].items():
        if not isinstance(l2_map, dict):
            continue

        level1_keywords: List[str] = []
        level2_nodes: List[Level2Node] = []

        for l2, l2_info in l2_map.items():
            if not isinstance(l2_info, dict):
                continue

            description = str(l2_info.get("description", "")).strip()
            keywords_raw = l2_info.get("keywords", [])
            if not isinstance(keywords_raw, list):
                keywords_raw = []

            keywords = [str(x).strip() for x in keywords_raw if isinstance(x, str) and str(x).strip()]
            level1_keywords.extend(keywords)

            level3_nodes: List[Level3Node] = []
            for l3 in keywords:
                rep_text_l3 = build_representation_text(
                    level_1=str(l1),
                    level_2=str(l2),
                    level_3=str(l3),
                    keywords=[l3],
                )
                if description:
                    rep_text_l3 += f" Parent description: {description}"
                level3_nodes.append(
                    Level3Node(
                        level_1=str(l1),
                        level_2=str(l2),
                        level_3=str(l3),
                        keywords=[l3],
                        rep_text=rep_text_l3,
                    )
                )

            rep_text_l2 = build_representation_text(
                level_1=str(l1),
                level_2=str(l2),
                level_3=None,
                keywords=keywords,
            )
            if description:
                rep_text_l2 += f" Description: {description}"

            level2_nodes.append(
                Level2Node(
                    level_1=str(l1),
                    level_2=str(l2),
                    keywords=keywords,
                    rep_text=rep_text_l2,
                    children=level3_nodes,
                )
            )

        rep_text_l1 = build_representation_text(
            level_1=str(l1),
            level_2=None,
            level_3=None,
            keywords=level1_keywords,
        )

        level1_nodes.append(
            Level1Node(
                level_1=str(l1),
                keywords=level1_keywords,
                rep_text=rep_text_l1,
                children=level2_nodes,
            )
        )

    if not level1_nodes:
        raise ValueError("No valid taxonomy nodes found in taxonomy.yaml.")

    return level1_nodes


# ----------------------------
# Embedding helpers
# ----------------------------

def encode_texts(model: SentenceTransformer, texts: List[str]):
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)


def cosine_score(vec_a, vec_b) -> float:
    return float(util.cos_sim(vec_a, vec_b).item())


# ----------------------------
# Keyword scoring
# ----------------------------

def keyword_score(text: str, keywords: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Returns:
      normalized keyword score in [0, 1] approximately
      debug list with phrase counts
    """
    normalized = normalize_text(text)
    hits: List[Dict[str, Any]] = []
    total_matches = 0

    for kw in keywords:
        count = phrase_occurrences(normalized, kw)
        if count > 0:
            hits.append({"keyword": kw, "count": count})
            total_matches += count

    # simple saturation-style normalization
    score = total_matches / (total_matches + 2.0) if total_matches > 0 else 0.0
    hits.sort(key=lambda x: (-x["count"], x["keyword"]))
    return score, hits


# ----------------------------
# Hierarchical prediction
# ----------------------------

def score_candidates(
    doc_embedding,
    doc_text: str,
    candidate_nodes: List[Any],
    alpha: float,
) -> List[Dict[str, Any]]:
    """
    alpha = embedding weight
    (1 - alpha) = keyword weight
    """
    results: List[Dict[str, Any]] = []

    for node in candidate_nodes:
        emb_score = cosine_score(doc_embedding, node.embedding)
        kw_score, kw_hits = keyword_score(doc_text, node.keywords)
        hybrid = alpha * emb_score + (1.0 - alpha) * kw_score

        label = getattr(node, "level_1", None)
        if hasattr(node, "level_2") and getattr(node, "level_2", None):
            label = getattr(node, "level_2")
        if hasattr(node, "level_3") and getattr(node, "level_3", None):
            label = getattr(node, "level_3")

        results.append(
            {
                "label": label,
                "embedding_score": emb_score,
                "keyword_score": kw_score,
                "hybrid_score": hybrid,
                "keyword_hits": kw_hits[:5],
                "node": node,
            }
        )

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results


def classify_document(
    text: str,
    level1_nodes: List[Level1Node],
    model: SentenceTransformer,
    alpha: float = 0.7,
    min_level1_score: float = 0.15,
    min_level2_score: float = 0.15,
    min_level3_score: float = 0.10,
) -> Dict[str, Any]:
    if not text.strip():
        return {
            "taxonomy": {
                "level_1": None,
                "level_2": None,
                "level_3": None,
            },
            "status": "unassigned",
            "confidence": 0.0,
            "debug": {"reason": "empty_text"},
        }

    normalized_text = normalize_text(text)
    doc_embedding = encode_texts(model, [normalized_text])[0]

    # Level 1
    l1_scores = score_candidates(doc_embedding, normalized_text, level1_nodes, alpha=alpha)
    best_l1 = l1_scores[0]
    if best_l1["hybrid_score"] < min_level1_score:
        return {
            "taxonomy": {
                "level_1": None,
                "level_2": None,
                "level_3": None,
            },
            "status": "unassigned",
            "confidence": best_l1["hybrid_score"],
            "debug": {
                "reason": "low_level1_confidence",
                "level_1_candidates": [
                    {k: v for k, v in x.items() if k != "node"} for x in l1_scores[:3]
                ],
            },
        }

    chosen_l1: Level1Node = best_l1["node"]

    # Level 2
    l2_scores = score_candidates(doc_embedding, normalized_text, chosen_l1.children, alpha=alpha)
    best_l2 = l2_scores[0]
    if best_l2["hybrid_score"] < min_level2_score:
        return {
            "taxonomy": {
                "level_1": chosen_l1.level_1,
                "level_2": None,
                "level_3": None,
            },
            "status": "partially_assigned",
            "confidence": best_l2["hybrid_score"],
            "debug": {
                "reason": "low_level2_confidence",
                "level_1_candidates": [
                    {k: v for k, v in x.items() if k != "node"} for x in l1_scores[:3]
                ],
                "level_2_candidates": [
                    {k: v for k, v in x.items() if k != "node"} for x in l2_scores[:3]
                ],
            },
        }

    chosen_l2: Level2Node = best_l2["node"]

    # Level 3
    l3_scores = score_candidates(doc_embedding, normalized_text, chosen_l2.children, alpha=alpha)
    best_l3 = l3_scores[0]
    if best_l3["hybrid_score"] < min_level3_score:
        return {
            "taxonomy": {
                "level_1": chosen_l1.level_1,
                "level_2": chosen_l2.level_2,
                "level_3": None,
            },
            "status": "partially_assigned",
            "confidence": best_l3["hybrid_score"],
            "debug": {
                "reason": "low_level3_confidence",
                "level_1_candidates": [
                    {k: v for k, v in x.items() if k != "node"} for x in l1_scores[:3]
                ],
                "level_2_candidates": [
                    {k: v for k, v in x.items() if k != "node"} for x in l2_scores[:3]
                ],
                "level_3_candidates": [
                    {k: v for k, v in x.items() if k != "node"} for x in l3_scores[:5]
                ],
            },
        }

    return {
        "taxonomy": {
            "level_1": chosen_l1.level_1,
            "level_2": chosen_l2.level_2,
            "level_3": best_l3["label"],
        },
        "status": "assigned",
        "confidence": best_l3["hybrid_score"],
        "debug": {
            "level_1_candidates": [{k: v for k, v in x.items() if k != "node"} for x in l1_scores[:3]],
            "level_2_candidates": [{k: v for k, v in x.items() if k != "node"} for x in l2_scores[:3]],
            "level_3_candidates": [{k: v for k, v in x.items() if k != "node"} for x in l3_scores[:5]],
        },
    }


# ----------------------------
# Precompute taxonomy embeddings
# ----------------------------

def attach_embeddings(level1_nodes: List[Level1Node], model: SentenceTransformer) -> None:
    l1_texts = [node.rep_text for node in level1_nodes]
    l1_embs = encode_texts(model, l1_texts)
    for node, emb in zip(level1_nodes, l1_embs):
        node.embedding = emb

        if node.children:
            l2_texts = [child.rep_text for child in node.children]
            l2_embs = encode_texts(model, l2_texts)
            for child, child_emb in zip(node.children, l2_embs):
                child.embedding = child_emb

                if child.children:
                    l3_texts = [grandchild.rep_text for grandchild in child.children]
                    l3_embs = encode_texts(model, l3_texts)
                    for grandchild, grandchild_emb in zip(child.children, l3_embs):
                        grandchild.embedding = grandchild_emb


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/clean_documents.jsonl", help="Input JSONL path")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml", help="Taxonomy YAML path")
    ap.add_argument(
        "--output",
        default="data/outputs/taxonomy_assignments_embeddings.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Embedding weight in hybrid score; keyword weight is (1-alpha)",
    )
    ap.add_argument("--store_debug", action="store_true", help="Store debug scoring information")
    ap.add_argument("--min_level1_score", type=float, default=0.15)
    ap.add_argument("--min_level2_score", type=float, default=0.15)
    ap.add_argument("--min_level3_score", type=float, default=0.10)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not os.path.exists(args.taxonomy):
        raise FileNotFoundError(f"Taxonomy config not found: {args.taxonomy}")

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        tax = yaml.safe_load(f)

    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Loading taxonomy...")
    level1_nodes = load_taxonomy(tax)

    print("Encoding taxonomy nodes...")
    attach_embeddings(level1_nodes, model)

    out_records: List[Dict[str, Any]] = []
    missing_id = 0
    empty_text = 0
    assigned = 0
    partially_assigned = 0
    unassigned = 0

    for idx, doc in enumerate(read_jsonl(args.input), start=1):
        doc_id = extract_doc_id(doc)
        if not doc_id:
            missing_id += 1
            doc_id = f"unknown_{missing_id}"

        text = extract_text(doc)
        if not text.strip():
            empty_text += 1

        result = classify_document(
            text=text,
            level1_nodes=level1_nodes,
            model=model,
            alpha=args.alpha,
            min_level1_score=args.min_level1_score,
            min_level2_score=args.min_level2_score,
            min_level3_score=args.min_level3_score,
        )

        rec: Dict[str, Any] = {
            "doc_id": doc_id,
            "taxonomy": result["taxonomy"],
            "status": result["status"],
            "confidence": round(float(result["confidence"]), 6),
        }

        if args.store_debug:
            rec["debug"] = result["debug"]

        out_records.append(rec)

        if result["status"] == "assigned":
            assigned += 1
        elif result["status"] == "partially_assigned":
            partially_assigned += 1
        else:
            unassigned += 1

        if idx % 25 == 0:
            print(f"Processed {idx} documents...")

    write_jsonl(args.output, out_records)

    print("\nTaxonomy assignment complete")
    print(f"Input:   {args.input}")
    print(f"Taxonomy:{args.taxonomy}")
    print(f"Output:  {args.output}")
    print(
        "Stats: "
        f"total={len(out_records)} "
        f"missing_id={missing_id} "
        f"empty_text={empty_text} "
        f"assigned={assigned} "
        f"partially_assigned={partially_assigned} "
        f"unassigned={unassigned}"
    )


if __name__ == "__main__":
    main()