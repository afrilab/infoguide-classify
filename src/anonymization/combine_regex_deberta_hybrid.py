import argparse
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

from context_filter_regex_predictions import (
    build_anonymized_docs,
    filter_predictions,
    load_json_or_jsonl,
    make_text_map,
    write_jsonl,
)
from entity_filters import filter_generic_org_units


Document = Dict[str, Any]
Entity = Dict[str, Any]


def spans_overlap(a: Entity, b: Entity) -> bool:
    return int(a["start"]) < int(b["end"]) and int(b["start"]) < int(a["end"])


def make_prediction_map(prediction_docs: List[Document]) -> Dict[str, Document]:
    return {
        str(doc["doc_id"]): doc
        for doc in prediction_docs
        if doc.get("doc_id")
    }


def filter_entity_types(entities: List[Entity], allowed_types: Set[str]) -> List[Entity]:
    if not allowed_types:
        return list(entities)
    return [
        entity
        for entity in entities
        if str(entity.get("type")) in allowed_types
    ]


def deduplicate_entities(entities: List[Entity]) -> List[Entity]:
    best_by_key: Dict[Tuple[str, int, int], Entity] = {}

    for entity in entities:
        key = (
            str(entity.get("type")),
            int(entity["start"]),
            int(entity["end"]),
        )
        previous = best_by_key.get(key)
        if previous is None or entity.get("score", 0.0) > previous.get("score", 0.0):
            best_by_key[key] = entity

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda entity: (int(entity["start"]), int(entity["end"]), str(entity.get("type"))))
    return deduped


def combine_predictions(
    source_docs: List[Document],
    regex_docs: List[Document],
    ner_docs: List[Document],
    regex_entities: Set[str],
    ner_entities: Set[str],
    output_model_name: str,
) -> List[Document]:
    regex_map = make_prediction_map(regex_docs)
    ner_map = make_prediction_map(ner_docs)

    combined_docs: List[Document] = []

    for source_doc in source_docs:
        doc_id = str(source_doc.get("doc_id", ""))
        if not doc_id:
            continue

        regex_predictions = filter_entity_types(
            regex_map.get(doc_id, {}).get("entities", []),
            regex_entities,
        )
        ner_predictions = filter_entity_types(
            ner_map.get(doc_id, {}).get("entities", []),
            ner_entities,
        )

        kept_ner = [
            ner_entity
            for ner_entity in ner_predictions
            if not any(spans_overlap(ner_entity, regex_entity) for regex_entity in regex_predictions)
        ]

        combined_docs.append(
            {
                "doc_id": doc_id,
                "model": output_model_name,
                "entities": deduplicate_entities(regex_predictions + kept_ner),
            }
        )

    return combined_docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to regex + DeBERTa hybrid config")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_docs = cfg.get("input_docs", cfg["benchmark_path"])
    source_docs = load_json_or_jsonl(input_docs)
    regex_docs = load_json_or_jsonl(cfg["regex_predictions"])
    ner_docs = load_json_or_jsonl(cfg["ner_predictions"])

    regex_entities = set(cfg.get("regex_entities", []))
    ner_entities = set(cfg.get("ner_entities", []))
    output_model_name = cfg.get("output_model_name", "regex_deberta_hybrid_context_filtered")
    window_chars = int(cfg.get("window_chars", 50))
    context_rules = cfg.get("context_rules", {}) or {}
    placeholders = cfg.get("placeholders", {}) or {}
    apply_generic_org_filter = bool(cfg.get("apply_generic_org_filter", True))

    text_map = make_text_map(source_docs)

    filtered_regex_docs = filter_predictions(
        prediction_docs=regex_docs,
        text_map=text_map,
        regex_entities=regex_entities,
        context_rules=context_rules,
        window_chars=window_chars,
        output_model_name="regex_only_context_filtered",
    )

    combined_docs = combine_predictions(
        source_docs=source_docs,
        regex_docs=filtered_regex_docs,
        ner_docs=ner_docs,
        regex_entities=regex_entities,
        ner_entities=ner_entities,
        output_model_name=output_model_name,
    )

    if apply_generic_org_filter:
        for doc in combined_docs:
            doc["entities"] = filter_generic_org_units(doc.get("entities", []))

    write_jsonl(cfg["output_predictions"], combined_docs)

    output_docs = cfg.get("output_docs")
    if output_docs:
        anonymized_docs = build_anonymized_docs(
            source_docs=source_docs,
            prediction_docs=combined_docs,
            placeholders=placeholders,
        )
        write_jsonl(output_docs, anonymized_docs)

    total_regex_before = sum(len(doc.get("entities", [])) for doc in regex_docs)
    total_regex_after = sum(len(doc.get("entities", [])) for doc in filtered_regex_docs)
    total_ner = sum(len(doc.get("entities", [])) for doc in ner_docs)
    total_combined = sum(len(doc.get("entities", [])) for doc in combined_docs)

    print("Regex + DeBERTa hybrid completed.")
    print(f"Regex entities before filter: {total_regex_before}")
    print(f"Regex entities after filter: {total_regex_after}")
    print(f"DeBERTa entities: {total_ner}")
    print(f"Combined entities: {total_combined}")
    print(f"Output predictions: {cfg['output_predictions']}")
    if output_docs:
        print(f"Output anonymized docs: {output_docs}")


if __name__ == "__main__":
    main()
