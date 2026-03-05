import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from regex_recognizers import build_regex_recognizers

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


# Enforce design policy: (1) regex + NER scope, (2) NER threshold, (3) regex wins overlaps
def filter_by_policy(results, regex_entities: set[str], ner_entities: set[str], ner_min_score: float, ) -> List:
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


# CLI entry: read config, run detection + anonymization, write outputs/logs
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/anonymization.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    spacy_model = cfg.get("spacy_model", "en_core_web_sm")
    placeholders = cfg["placeholders"]

    # Entity scope: regex list is direct, NER list is filtered by per-entity toggles
    regex_entities = set(cfg.get("regex_entities", []))
    ner_entities_defined = set(cfg.get("ner_entities", []))
    ner_toggles = cfg.get("ner", {}) or {}
    ner_entities = {
        et for et in ner_entities_defined
        if bool(ner_toggles.get(et, {}).get("enabled", True))
    }
    ner_min_score = float(cfg.get("ner_min_score", 0.50))

    regex_cfg = cfg.get("regex", {})

    analyzer = build_analyzer(spacy_model, regex_cfg)
    anonymizer = AnonymizerEngine()
    operators = build_operators(placeholders)

    paths = cfg["paths"]
    input_path = paths["input"]
    out_docs_path = paths["output_docs"]
    out_log_path = paths["output_logs"]
    docs = read_jsonl(input_path)

    out_docs: List[Dict[str, Any]] = []
    pii_logs: List[Dict[str, Any]] = []

    for doc in docs:
        doc_id = doc.get("doc_id")
        text = doc.get("processed_text", "")

        # Fail-safe: if missing text, keep unchanged
        if not isinstance(text, str) or not text:
            doc_out = {
                 "doc_id": doc_id, 
                 "anonymized_text": doc.get("processed_text", "")
                 }
            out_docs.append(doc_out)
            continue

        try:
            # Detect configured entity types
            entities_to_detect = sorted(regex_entities.union(ner_entities))
            results = analyzer.analyze(text=text, language="en", entities=entities_to_detect)

            # Apply overlap + threshold policy before anonymizing/logging
            final_results = filter_by_policy(
                results=results,
                regex_entities=regex_entities,
                ner_entities=ner_entities,
                ner_min_score=ner_min_score,
            )

            # Anonymize (replace with placeholders)
            anon_res = anonymizer.anonymize(text=text, analyzer_results=final_results, operators=operators)
            anonymized_text = anon_res.text

            doc_out = {"doc_id": doc_id, "anonymized_text": anonymized_text }
            out_docs.append(doc_out)

        
            # Log the spans that are anonymized (offsets in original processed_text)
            for r in final_results:
                pii_logs.append(
                    {
                        "doc_id": doc_id,
                        "pii_type": r.entity_type,
                        "start": r.start,
                        "end": r.end,
                        "score": float(r.score) if r.score is not None else None,
                    }
                )

            
        except Exception:
            # Failure handling: keep doc unchanged
            doc_out = {
                "doc_id": doc_id,
                "anonymized_text": text}
            out_docs.append(doc_out)

    write_jsonl(out_docs_path, out_docs)
    write_jsonl(out_log_path, pii_logs)

if __name__ == "__main__":
    main()