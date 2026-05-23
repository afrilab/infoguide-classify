import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from context_filter_regex_predictions import build_anonymized_docs
from entity_filters import filter_generic_org_units
from metrics import build_run_metrics, default_metrics_path, write_metrics


Document = Dict[str, Any]
Entity = Dict[str, Any]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json_or_jsonl(path: str | Path) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    if content.startswith("["):
        return json.loads(content)

    docs: List[Document] = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs


def write_jsonl(path: str | Path, records: Iterable[Document]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def should_process_doc(doc_id: str, prefixes: List[str]) -> bool:
    if not prefixes:
        return True
    return any(doc_id.startswith(prefix) for prefix in prefixes)


def iter_text_windows(
    text: str,
    max_chars: int,
    overlap_chars: int,
    max_words: Optional[int] = None,
    overlap_words: int = 0,
) -> Iterable[Tuple[int, str]]:
    if max_words and max_words > 0:
        word_spans = list(re.finditer(r"\S+", text))
        if len(word_spans) <= max_words:
            yield 0, text
            return

        step_words = max(1, max_words - max(0, overlap_words))
        word_start = 0

        while word_start < len(word_spans):
            word_end = min(word_start + max_words, len(word_spans))
            start = word_spans[word_start].start()
            end = word_spans[word_end - 1].end()

            yield start, text[start:end]

            if word_end >= len(word_spans):
                break

            word_start += step_words

        return

    if max_chars <= 0 or len(text) <= max_chars:
        yield 0, text
        return

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        if end < text_length:
            split_at = text.rfind(" ", start, end)
            if split_at > start + max_chars // 2:
                end = split_at

        yield start, text[start:end]

        if end >= text_length:
            break

        start = max(0, end - overlap_chars)


def normalize_gliner_entity(
    raw_entity: Entity,
    offset: int,
    label_to_type: Dict[str, str],
    text: str,
) -> Entity | None:
    raw_label = str(raw_entity.get("label", "")).lower()
    entity_type = label_to_type.get(raw_label)
    if not entity_type:
        return None

    start = int(raw_entity["start"]) + offset
    end = int(raw_entity["end"]) + offset

    if start < 0 or end <= start or end > len(text):
        return None

    return {
        "type": entity_type,
        "text": text[start:end],
        "start": start,
        "end": end,
        "score": float(raw_entity.get("score", 0.0)),
    }


def deduplicate_entities(entities: List[Entity]) -> List[Entity]:
    best_by_key: Dict[Tuple[str, int, int], Entity] = {}

    for entity in entities:
        key = (entity["type"], entity["start"], entity["end"])
        previous = best_by_key.get(key)
        if previous is None or entity.get("score", 0.0) > previous.get("score", 0.0):
            best_by_key[key] = entity

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda entity: (entity["start"], entity["end"], entity["type"]))
    return deduped


def predict_doc_entities(
    model: Any,
    text: str,
    labels: List[str],
    label_to_type: Dict[str, str],
    threshold: float,
    max_chars: int,
    overlap_chars: int,
    max_words: Optional[int],
    overlap_words: int,
) -> List[Entity]:
    entities: List[Entity] = []

    for offset, window_text in iter_text_windows(
        text=text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        max_words=max_words,
        overlap_words=overlap_words,
    ):
        raw_entities = model.predict_entities(
            window_text,
            labels,
            threshold=threshold,
        )

        for raw_entity in raw_entities:
            entity = normalize_gliner_entity(
                raw_entity=raw_entity,
                offset=offset,
                label_to_type=label_to_type,
                text=text,
            )
            if entity is not None:
                entities.append(entity)

    return deduplicate_entities(entities)


def load_gliner_model(model_id: str) -> Any:
    try:
        from gliner import GLiNER  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "GLiNER is required for this baseline. Install it in the project "
            "environment with: pip install gliner"
        ) from exc

    return GLiNER.from_pretrained(model_id)


def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to GLiNER baseline YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    model_name = cfg.get("model_name", "gliner_baseline")
    model_id = cfg.get("model_id", "urchade/gliner_multi-v2.1")
    placeholders = cfg.get("placeholders", {}) or {}
    threshold = float(cfg.get("threshold", 0.5))
    max_chars = int(cfg.get("max_chars", 3000))
    overlap_chars = int(cfg.get("overlap_chars", 300))
    max_words = cfg.get("max_words")
    max_words = int(max_words) if max_words is not None else None
    overlap_words = int(cfg.get("overlap_words", 0))
    doc_id_prefixes = [str(prefix) for prefix in cfg.get("doc_id_prefixes", [])]
    text_field = str(cfg.get("text_field", "processed_text"))
    apply_generic_org_filter = bool(cfg.get("apply_generic_org_filter", True))

    label_rows = cfg["labels"]
    labels = [str(row["label"]) for row in label_rows]
    label_to_type = {
        str(row["label"]).lower(): str(row["type"])
        for row in label_rows
    }

    model = load_gliner_model(model_id)
    docs = read_json_or_jsonl(paths["input"])

    outputs: List[Document] = []
    document_latencies_seconds: List[float] = []
    char_count = 0
    for doc in docs:
        doc_started = time.perf_counter()
        doc_id = str(doc.get("doc_id", ""))
        if not should_process_doc(doc_id, doc_id_prefixes):
            continue

        text = doc.get(text_field, "")
        if not isinstance(text, str) or not text:
            outputs.append({"doc_id": doc_id, "model": model_name, "entities": []})
            document_latencies_seconds.append(time.perf_counter() - doc_started)
            continue

        char_count += len(text)
        entities = predict_doc_entities(
            model=model,
            text=text,
            labels=labels,
            label_to_type=label_to_type,
            threshold=threshold,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            max_words=max_words,
            overlap_words=overlap_words,
        )
        if apply_generic_org_filter:
            entities = filter_generic_org_units(entities)
        outputs.append({"doc_id": doc_id, "model": model_name, "entities": entities})
        document_latencies_seconds.append(time.perf_counter() - doc_started)

    write_jsonl(paths["output_predictions"], outputs)

    output_docs = paths.get("output_docs")
    if output_docs:
        anonymized_docs = build_anonymized_docs(
            source_docs=docs,
            prediction_docs=outputs,
            placeholders=placeholders,
        )
        write_jsonl(output_docs, anonymized_docs)

    runtime_seconds = time.perf_counter() - run_started
    metrics = build_run_metrics(
        variant_name=model_name,
        stage_name="anonymization",
        doc_count=len(outputs),
        char_count=char_count,
        entity_count=sum(len(doc.get("entities", [])) for doc in outputs),
        runtime_seconds=runtime_seconds,
        document_latencies_seconds=document_latencies_seconds,
        config_path=args.config,
    )
    metrics_path = cfg.get("metrics_path", default_metrics_path(model_name))
    write_metrics(metrics_path, metrics)
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
