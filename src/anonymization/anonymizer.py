import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import spacy

from metrics import build_run_metrics, default_metrics_path, write_metrics

DEFAULT_MODEL_NAME = "anonymizer"
DEFAULT_NER_SCORE = 0.85
GENERIC_ORG_UNIT_SUFFIXES = ("division", "department", "team", "unit", "office", "board", "panel", "committee")
LEADING_ORG_ARTICLES = ("the ", "a ", "an ")
SPACY_TO_PIPELINE_ENTITY = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
}


@dataclass
class DetectionResult:
    entity_type: str
    start: int
    end: int
    score: float | None

# Load YAML config into a dict
def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Read JSONL file
def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


# Write JSONL file and create parent dirs
def write_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    # overlap happens if ranges intersect: [a0,a1) and [b0,b1)
    return not (a[1] <= b[0] or b[1] <= a[0])


def normalize_org_text_for_unit_filter(text: str) -> str:
    normalized = " ".join(text.strip().split()).lower()
    for article in LEADING_ORG_ARTICLES:
        if normalized.startswith(article):
            return normalized[len(article):]
    return normalized


def is_generic_org_unit_detection(text: str) -> bool:
    normalized = normalize_org_text_for_unit_filter(text)
    return any(
        normalized.endswith(f" {suffix}") or normalized == suffix
        for suffix in GENERIC_ORG_UNIT_SUFFIXES
    )


# Enforce design policy: (1) regex + NER scope, (2) NER threshold, (3) regex wins overlaps
def filter_by_policy(
    results,
    regex_entities: set[str],
    ner_entities: set[str],
    ner_min_score: float,
    text: str = "",
) -> List:
    regex_results = []
    ner_results = []

    for r in results:
        et = r.entity_type
        if et in regex_entities:
            regex_results.append(r)
        elif et in ner_entities:
            ner_results.append(r)
        else:
            # ignore anything outside defined scope
            continue

    # Apply NER confidence threshold
    ner_results = [r for r in ner_results if (r.score or 0.0) >= ner_min_score]
    ner_results = [
        r
        for r in ner_results
        if not (
            r.entity_type == "ORGANIZATION"
            and is_generic_org_unit_detection(text[r.start:r.end])
        )
    ]

    # Freeze regex spans
    frozen_spans = [(r.start, r.end) for r in regex_results]

    # Remove overlapping NER spans: drop any NER span that overlaps a frozen regex span
    filtered_ner = []
    for r in ner_results:
        nspan = (r.start, r.end)
        if any(spans_overlap(nspan, fspan) for fspan in frozen_spans):
            continue
        filtered_ner.append(r)

    final_results = regex_results + filtered_ner
    # Sort by offset for stable output/logs
    final_results.sort(key=lambda x: (x.start, x.end))
    return final_results


def build_regex_patterns(regex_cfg: Dict[str, Any], regex_entities: set[str]) -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []

    for entity_type, cfg in regex_cfg.items():
        if entity_type not in regex_entities or not bool(cfg.get("enabled", True)):
            continue

        for pattern_str in cfg.get("patterns", []):
            patterns.append(
                (
                    entity_type,
                    re.compile(pattern_str, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL),
                )
            )

    return patterns


def detect_regex_entities(text: str, patterns: List[Tuple[str, re.Pattern]]) -> List[DetectionResult]:
    results: List[DetectionResult] = []

    for entity_type, pattern in patterns:
        for match in pattern.finditer(text):
            if match.start() == match.end():
                continue
            results.append(
                DetectionResult(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    score=1.0,
                )
            )

    return results


def detect_ner_entities(nlp, text: str, ner_entities: set[str]) -> List[DetectionResult]:
    if nlp is None or not ner_entities:
        return []

    results: List[DetectionResult] = []
    doc = nlp(text)

    for ent in doc.ents:
        entity_type = SPACY_TO_PIPELINE_ENTITY.get(ent.label_)
        if entity_type not in ner_entities:
            continue

        results.append(
            DetectionResult(
                entity_type=entity_type,
                start=ent.start_char,
                end=ent.end_char,
                score=DEFAULT_NER_SCORE,
            )
        )

    return results


def resolve_overlapping_results(results: List[DetectionResult]) -> List[DetectionResult]:
    selected: List[DetectionResult] = []

    for result in sorted(
        results,
        key=lambda r: (-(r.score or 0.0), -(r.end - r.start), r.start, r.end),
    ):
        span = (result.start, result.end)
        if any(spans_overlap(span, (kept.start, kept.end)) for kept in selected):
            continue
        selected.append(result)

    selected.sort(key=lambda r: (r.start, r.end))
    return selected


def anonymize_text(text: str, results: List[DetectionResult], placeholders: Dict[str, str]) -> str:
    anonymized = text

    for result in sorted(results, key=lambda r: (r.start, r.end), reverse=True):
        placeholder = placeholders.get(result.entity_type, f"[{result.entity_type}]")
        anonymized = anonymized[: result.start] + placeholder + anonymized[result.end :]

    return anonymized


ENTITY_TYPE_MAP = {
    "ORGANIZATION": "ORG",
}

def build_prediction_output(
    doc_id: str,
    text: str,
    final_results,
    model_name: str,
) -> Dict[str, Any]:
    entities = []

    for result in final_results:
        mapped_type = ENTITY_TYPE_MAP.get(result.entity_type, result.entity_type)

        entities.append(
            {
                "type": mapped_type,
                "text": text[result.start:result.end],
                "start": result.start,
                "end": result.end,
                "score": float(result.score) if result.score is not None else None,
            }
        )

    return {
        "doc_id": doc_id,
        "model": model_name,
        "entities": entities,
    }


# CLI entry: read config, run detection + anonymization, write pipeline and evaluation outputs
def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an anonymization config YAML file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    model_name = cfg.get("model_name", DEFAULT_MODEL_NAME)
    spacy_model = cfg.get("spacy_model", "en_core_web_sm")
    placeholders = cfg["placeholders"]

    regex_entities = set(cfg.get("regex_entities", []))

    ner_entities_defined = set(cfg.get("ner_entities", []))
    ner_toggles = cfg.get("ner", {}) or {}

    ner_entities = {
        entity_type
        for entity_type in ner_entities_defined
        if bool(ner_toggles.get(entity_type, {}).get("enabled", True))
    }

    ner_min_score = float(cfg.get("ner_min_score", 0.50))
    regex_cfg = cfg.get("regex", {})

    regex_patterns = build_regex_patterns(regex_cfg, regex_entities)
    nlp = spacy.load(spacy_model) if ner_entities else None

    paths = cfg["paths"]

    input_path = paths["input"]

    # Pipeline output. Keep this path because the next pipeline step depends on it.
    out_docs_path = paths["output_docs"]

    # Evaluation output. Used to compare this anonymizer against ground truth.
    out_pred_path = paths["output_predictions"]

    docs = read_jsonl(input_path)

    out_docs: List[Dict[str, Any]] = []
    prediction_outputs: List[Dict[str, Any]] = []
    document_latencies_seconds: List[float] = []
    char_count = 0

    for doc in docs:
        doc_started = time.perf_counter()
        doc_id = doc.get("doc_id")
        text = doc.get("processed_text", "")

        if not isinstance(text, str) or not text:
            out_docs.append(
                {
                    "doc_id": doc_id,
                    "anonymized_text": "",
                }
            )

            prediction_outputs.append(
                {
                    "doc_id": doc_id,
                    "model": model_name,
                    "entities": [],
                }
            )

            document_latencies_seconds.append(time.perf_counter() - doc_started)
            continue

        char_count += len(text)

        try:
            results = detect_regex_entities(text, regex_patterns)
            results.extend(detect_ner_entities(nlp, text, ner_entities))

            final_results = filter_by_policy(
                results=results,
                regex_entities=regex_entities,
                ner_entities=ner_entities,
                ner_min_score=ner_min_score,
                text=text,
            )
            final_results = resolve_overlapping_results(final_results)
            anonymized_text = anonymize_text(text, final_results, placeholders)

            out_docs.append(
                {
                    "doc_id": doc_id,
                    "anonymized_text": anonymized_text,
                }
            )

            prediction_outputs.append(
                build_prediction_output(
                    doc_id=doc_id,
                    text=text,
                    final_results=final_results,
                    model_name=model_name,
                )
            )

        except Exception as exc:
            out_docs.append(
                {
                    "doc_id": doc_id,
                    "anonymized_text": text,
                    "error": str(exc),
                }
            )

            prediction_outputs.append(
                {
                    "doc_id": doc_id,
                    "model": model_name,
                    "entities": [],
                    "error": str(exc),
                }
            )
        finally:
            document_latencies_seconds.append(time.perf_counter() - doc_started)

    write_jsonl(out_docs_path, out_docs)
    write_jsonl(out_pred_path, prediction_outputs)

    runtime_seconds = time.perf_counter() - run_started
    metrics = build_run_metrics(
        variant_name=model_name,
        stage_name="anonymization",
        doc_count=len(docs),
        char_count=char_count,
        entity_count=sum(len(doc.get("entities", [])) for doc in prediction_outputs),
        runtime_seconds=runtime_seconds,
        document_latencies_seconds=document_latencies_seconds,
        config_path=args.config,
    )
    metrics_path = cfg.get("metrics_path", default_metrics_path(model_name))
    write_metrics(metrics_path, metrics)
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
