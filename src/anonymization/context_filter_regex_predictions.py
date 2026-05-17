import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

from metrics import build_run_metrics, default_metrics_path, write_metrics

Document = Dict[str, Any]
Entity = Dict[str, Any]
CsvRow = Dict[str, str]

DEFAULT_BEFORE_PATH = (
    "outputs/anonymizer_evaluation/regex_spacy_hybrid_trf/per_type.csv"
)
DEFAULT_AFTER_PATH = (
    "outputs/anonymizer_evaluation/"
    "regex_spacy_hybrid_trf_context_filtered/per_type.csv"
)
DEFAULT_COMPARISON_OUTPUT_PATH = (
    "outputs/anonymizer_evaluation/"
    "regex_spacy_hybrid_trf_context_filtered/comparison_table.csv"
)
DEFAULT_REGEX_TYPES = [
    "EMAIL",
    "PHONE",
    "DATE",
    "CREDIT_CARD",
    "IBAN",
]


def load_json_or_jsonl(path: str | Path) -> List[Document]:
    path_obj = Path(path)

    with path_obj.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    if content.startswith("["):
        return json.loads(content)

    docs = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))

    return docs


def write_jsonl(path: str | Path, records: List[Document]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_per_type_csv(path: str | Path) -> Dict[str, CsvRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["type"]: row for row in reader}


def int_value(row: CsvRow, key: str) -> int:
    return int(row.get(key, 0) or 0)


def float_value(row: CsvRow, key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def format_delta(value: float) -> str:
    return f"{value:.4f}"


def write_csv(path: str | Path, fieldnames: List[str], rows: List[CsvRow]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_text_map(docs: List[Document]) -> Dict[str, str]:
    text_map = {}

    for doc in docs:
        doc_id = doc.get("doc_id")
        text = doc.get("text") or doc.get("processed_text") or ""

        if doc_id and isinstance(text, str):
            text_map[doc_id] = text

    return text_map


def make_prediction_map(prediction_docs: List[Document]) -> Dict[str, Document]:
    return {
        doc["doc_id"]: doc
        for doc in prediction_docs
        if doc.get("doc_id")
    }


def get_context(text: str, start: int, end: int, window_chars: int) -> str:
    left = max(0, start - window_chars)
    right = min(len(text), end + window_chars)
    return text[left:right].lower()


def contains_any(context: str, keywords: List[str]) -> bool:
    return any(keyword.lower() in context for keyword in keywords)


def should_keep_entity(
    entity: Entity,
    text: str,
    regex_entities: Set[str],
    context_rules: Dict[str, Any],
    window_chars: int,
) -> bool:
    entity_type = entity.get("type")

    # Only filter regex-based entities.
    # NER entities such as PERSON, ORG, LOCATION are kept unchanged.
    if entity_type not in regex_entities:
        return True

    start = int(entity["start"])
    end = int(entity["end"])

    context = get_context(text, start, end, window_chars)

    rules = context_rules.get(entity_type, {})
    positive_keywords = rules.get("positive", []) or []
    negative_keywords = rules.get("negative", []) or []

    has_positive_context = contains_any(context, positive_keywords)
    has_negative_context = contains_any(context, negative_keywords)

    # Main filtering rule:
    # If a regex match appears in negative context and no positive context supports it,
    # treat it as a likely false positive and remove it.
    if has_negative_context and not has_positive_context:
        return False

    return True


def filter_predictions(
    prediction_docs: List[Document],
    text_map: Dict[str, str],
    regex_entities: Set[str],
    context_rules: Dict[str, Any],
    window_chars: int,
    output_model_name: str | None = None,
    document_latencies_seconds: List[float] | None = None,
) -> List[Document]:
    filtered_docs = []

    for pred_doc in prediction_docs:
        doc_started = time.perf_counter()
        doc_id = pred_doc.get("doc_id")
        text = text_map.get(doc_id, "")

        kept_entities = []
        removed_entities = []

        for entity in pred_doc.get("entities", []):
            keep = should_keep_entity(
                entity=entity,
                text=text,
                regex_entities=regex_entities,
                context_rules=context_rules,
                window_chars=window_chars,
            )

            if keep:
                kept_entities.append(entity)
            else:
                removed_entities.append(entity)

        filtered_doc = {
            "doc_id": doc_id,
            "model": output_model_name or pred_doc.get("model", "context_filtered"),
            "entities": kept_entities,
        }

        # Helpful for debugging, but evaluator will ignore this field.
        if removed_entities:
            filtered_doc["removed_entities"] = removed_entities

        filtered_docs.append(filtered_doc)
        if document_latencies_seconds is not None:
            document_latencies_seconds.append(time.perf_counter() - doc_started)

    return filtered_docs


def anonymize_text(text: str, entities: List[Entity], placeholders: Dict[str, str]) -> str:
    anonymized = text

    for entity in sorted(
        entities,
        key=lambda item: (int(item["start"]), int(item["end"])),
        reverse=True,
    ):
        entity_type = entity.get("type", "")
        start = int(entity["start"])
        end = int(entity["end"])

        if start < 0 or end > len(anonymized) or start >= end:
            continue

        placeholder = placeholders.get(entity_type, f"[{entity_type}]")
        anonymized = anonymized[:start] + placeholder + anonymized[end:]

    return anonymized


def build_anonymized_docs(
    source_docs: List[Document],
    prediction_docs: List[Document],
    placeholders: Dict[str, str],
) -> List[Document]:
    prediction_map = make_prediction_map(prediction_docs)
    anonymized_docs = []

    for source_doc in source_docs:
        doc_id = source_doc.get("doc_id")
        text = source_doc.get("processed_text") or source_doc.get("text") or ""
        prediction_doc = prediction_map.get(doc_id, {})
        entities = prediction_doc.get("entities", [])

        anonymized_docs.append(
            {
                "doc_id": doc_id,
                "anonymized_text": anonymize_text(text, entities, placeholders)
                if isinstance(text, str)
                else "",
            }
        )

    return anonymized_docs


def build_comparison_rows(
    before_rows: Dict[str, CsvRow],
    after_rows: Dict[str, CsvRow],
    regex_types: List[str],
) -> List[CsvRow]:
    comparison_rows = []

    for entity_type in regex_types:
        before = before_rows.get(entity_type, {})
        after = after_rows.get(entity_type, {})

        strict_fp_before = int_value(before, "strict_fp")
        strict_fp_after = int_value(after, "strict_fp")
        relaxed_fp_before = int_value(before, "relaxed_fp")
        relaxed_fp_after = int_value(after, "relaxed_fp")

        strict_precision_before = float_value(before, "strict_precision")
        strict_precision_after = float_value(after, "strict_precision")
        strict_recall_before = float_value(before, "strict_recall")
        strict_recall_after = float_value(after, "strict_recall")
        relaxed_precision_before = float_value(before, "relaxed_precision")
        relaxed_precision_after = float_value(after, "relaxed_precision")
        relaxed_recall_before = float_value(before, "relaxed_recall")
        relaxed_recall_after = float_value(after, "relaxed_recall")

        comparison_rows.append(
            {
                "type": entity_type,
                "strict_fp_before": str(strict_fp_before),
                "strict_fp_after": str(strict_fp_after),
                "strict_fp_delta": str(strict_fp_after - strict_fp_before),
                "strict_precision_before": format_delta(strict_precision_before),
                "strict_precision_after": format_delta(strict_precision_after),
                "strict_precision_delta": format_delta(
                    strict_precision_after - strict_precision_before
                ),
                "strict_recall_before": format_delta(strict_recall_before),
                "strict_recall_after": format_delta(strict_recall_after),
                "strict_recall_delta": format_delta(
                    strict_recall_after - strict_recall_before
                ),
                "relaxed_fp_before": str(relaxed_fp_before),
                "relaxed_fp_after": str(relaxed_fp_after),
                "relaxed_fp_delta": str(relaxed_fp_after - relaxed_fp_before),
                "relaxed_precision_before": format_delta(relaxed_precision_before),
                "relaxed_precision_after": format_delta(relaxed_precision_after),
                "relaxed_precision_delta": format_delta(
                    relaxed_precision_after - relaxed_precision_before
                ),
                "relaxed_recall_before": format_delta(relaxed_recall_before),
                "relaxed_recall_after": format_delta(relaxed_recall_after),
                "relaxed_recall_delta": format_delta(
                    relaxed_recall_after - relaxed_recall_before
                ),
            }
        )

    return comparison_rows


def write_comparison_table(
    before_path: str | Path,
    after_path: str | Path,
    output_path: str | Path,
    regex_types: List[str],
) -> None:
    comparison_rows = build_comparison_rows(
        before_rows=load_per_type_csv(before_path),
        after_rows=load_per_type_csv(after_path),
        regex_types=regex_types,
    )
    write_csv(
        output_path,
        [
            "type",
            "strict_fp_before",
            "strict_fp_after",
            "strict_fp_delta",
            "strict_precision_before",
            "strict_precision_after",
            "strict_precision_delta",
            "strict_recall_before",
            "strict_recall_after",
            "strict_recall_delta",
            "relaxed_fp_before",
            "relaxed_fp_after",
            "relaxed_fp_delta",
            "relaxed_precision_before",
            "relaxed_precision_after",
            "relaxed_precision_delta",
            "relaxed_recall_before",
            "relaxed_recall_after",
            "relaxed_recall_delta",
        ],
        comparison_rows,
    )


def run_context_filter(config_path: str) -> None:
    run_started = time.perf_counter()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    benchmark_path = cfg["benchmark_path"]
    input_docs = cfg.get("input_docs", benchmark_path)
    input_predictions = cfg["input_predictions"]
    output_predictions = cfg["output_predictions"]
    output_docs = cfg.get("output_docs")

    window_chars = int(cfg.get("window_chars", 50))
    regex_entities = set(cfg.get("regex_entities", []))
    context_rules = cfg.get("context_rules", {}) or {}
    output_model_name = cfg.get("output_model_name")
    placeholders = cfg.get("placeholders", {}) or {}

    source_docs = load_json_or_jsonl(input_docs)
    prediction_docs = load_json_or_jsonl(input_predictions)

    text_map = make_text_map(source_docs)
    document_latencies_seconds: List[float] = []

    filtered_docs = filter_predictions(
        prediction_docs=prediction_docs,
        text_map=text_map,
        regex_entities=regex_entities,
        context_rules=context_rules,
        window_chars=window_chars,
        output_model_name=output_model_name,
        document_latencies_seconds=document_latencies_seconds,
    )

    write_jsonl(output_predictions, filtered_docs)

    if output_docs:
        anonymized_docs = build_anonymized_docs(
            source_docs=source_docs,
            prediction_docs=filtered_docs,
            placeholders=placeholders,
        )
        write_jsonl(output_docs, anonymized_docs)

    total_before = sum(len(doc.get("entities", [])) for doc in prediction_docs)
    total_after = sum(len(doc.get("entities", [])) for doc in filtered_docs)
    removed = total_before - total_after
    runtime_seconds = time.perf_counter() - run_started
    variant_name = output_model_name or "context_filtered"
    char_count = sum(len(text_map.get(doc.get("doc_id"), "")) for doc in prediction_docs)
    metrics = build_run_metrics(
        variant_name=variant_name,
        stage_name="context_filtering",
        doc_count=len(filtered_docs),
        char_count=char_count,
        entity_count=total_after,
        runtime_seconds=runtime_seconds,
        document_latencies_seconds=document_latencies_seconds,
        config_path=config_path,
    )
    metrics_path = cfg.get("metrics_path", default_metrics_path(variant_name))
    write_metrics(metrics_path, metrics)

    print("Context filtering completed.")
    print(f"Input predictions: {input_predictions}")
    print(f"Output predictions: {output_predictions}")
    if output_docs:
        print(f"Output anonymized docs: {output_docs}")
    print(f"Entities before: {total_before}")
    print(f"Entities after: {total_after}")
    print(f"Removed entities: {removed}")
    print(f"Metrics: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=False,
        help="Path to context filtering YAML config.",
    )
    parser.add_argument(
        "--compare-regex-pii",
        action="store_true",
        help="Write before/after regex PII comparison table.",
    )
    parser.add_argument("--before", default=DEFAULT_BEFORE_PATH)
    parser.add_argument("--after", default=DEFAULT_AFTER_PATH)
    parser.add_argument("--output", default=DEFAULT_COMPARISON_OUTPUT_PATH)
    parser.add_argument(
        "--regex-types",
        nargs="+",
        default=DEFAULT_REGEX_TYPES,
        help="Regex-backed PII types to include in the comparison.",
    )
    args = parser.parse_args()

    if args.compare_regex_pii:
        write_comparison_table(
            before_path=args.before,
            after_path=args.after,
            output_path=args.output,
            regex_types=args.regex_types,
        )
        return

    if not args.config:
        parser.error("--config is required unless --compare-regex-pii is used.")

    run_context_filter(args.config)


if __name__ == "__main__":
    main()
