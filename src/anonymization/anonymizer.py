import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import spacy
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

DEFAULT_MODEL_NAME = "presidio"
GENERIC_ORG_UNIT_SUFFIXES = ("division", "department", "team", "unit", "office", "board", "panel", "committee")
LEADING_ORG_ARTICLES = ("the ", "a ", "an ")

def build_regex_recognizers(regex_cfg: Dict[str, Any]) -> List[PatternRecognizer]:
    """
    Builds Presidio PatternRecognizers from configs/anonymization.yaml.
    """
    recognizers: List[PatternRecognizer] = []

    for entity_type, cfg in regex_cfg.items():
        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            continue

        presidio_patterns: List[Pattern] = []

        for i, pattern_str in enumerate(cfg.get("patterns", [])):
            presidio_patterns.append(
                Pattern(
                    name=f"{entity_type}_pattern_{i}",
                    regex=pattern_str,
                    score=1.0,
                )
            )

        recognizer = PatternRecognizer(
            supported_entity=entity_type,
            patterns=presidio_patterns,
            name=f"REGEX_{entity_type}",
        )

        recognizers.append(recognizer)

    return recognizers

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


# Build Presidio AnalyzerEngine with spaCy NER + custom regex recognizers from YAML
def build_analyzer(spacy_model: str, regex_cfg: Dict[str, Any]) -> AnalyzerEngine:
    # Ensure spaCy model is available
    spacy.load(spacy_model)

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }
    )
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

    # Add regex recognizers (PatternRecognizers) from YAML
    recognizers = build_regex_recognizers(regex_cfg)
    for rec in recognizers:
        analyzer.registry.add_recognizer(rec)

    return analyzer


def build_operators(placeholders: Dict[str, str]) -> Dict[str, OperatorConfig]:
    # Replace entity spans with placeholders
    ops: Dict[str, OperatorConfig] = {}
    for entity_type, placeholder in placeholders.items():
        ops[entity_type] = OperatorConfig("replace", {"new_value": placeholder})
    return ops


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

    analyzer = build_analyzer(spacy_model, regex_cfg)
    anonymizer = AnonymizerEngine()
    operators = build_operators(placeholders)

    paths = cfg["paths"]

    input_path = paths["input"]

    # Pipeline output. Keep this path because the next pipeline step depends on it.
    out_docs_path = paths["output_docs"]

    # Evaluation output. Used to compare this anonymizer against ground truth.
    out_pred_path = paths["output_predictions"]

    docs = read_jsonl(input_path)

    out_docs: List[Dict[str, Any]] = []
    prediction_outputs: List[Dict[str, Any]] = []

    for doc in docs:
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

            continue

        try:
            entities_to_detect = sorted(regex_entities.union(ner_entities))

            results = analyzer.analyze(
                text=text,
                language="en",
                entities=entities_to_detect,
            )

            final_results = filter_by_policy(
                results=results,
                regex_entities=regex_entities,
                ner_entities=ner_entities,
                ner_min_score=ner_min_score,
                text=text,
            )

            anonymized_result = anonymizer.anonymize(
                text=text,
                analyzer_results=final_results,
                operators=operators,
            )

            out_docs.append(
                {
                    "doc_id": doc_id,
                    "anonymized_text": anonymized_result.text,
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

    write_jsonl(out_docs_path, out_docs)
    write_jsonl(out_pred_path, prediction_outputs)


if __name__ == "__main__":
    main()
