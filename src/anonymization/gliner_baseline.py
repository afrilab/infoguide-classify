import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


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
) -> Iterable[Tuple[int, str]]:
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
) -> List[Entity]:
    entities: List[Entity] = []

    for offset, window_text in iter_text_windows(text, max_chars, overlap_chars):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to GLiNER baseline YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    model_name = cfg.get("model_name", "gliner_baseline")
    model_id = cfg.get("model_id", "urchade/gliner_multi-v2.1")
    threshold = float(cfg.get("threshold", 0.5))
    max_chars = int(cfg.get("max_chars", 3000))
    overlap_chars = int(cfg.get("overlap_chars", 300))
    doc_id_prefixes = [str(prefix) for prefix in cfg.get("doc_id_prefixes", [])]
    text_field = str(cfg.get("text_field", "processed_text"))

    label_rows = cfg["labels"]
    labels = [str(row["label"]) for row in label_rows]
    label_to_type = {
        str(row["label"]).lower(): str(row["type"])
        for row in label_rows
    }

    model = load_gliner_model(model_id)
    docs = read_json_or_jsonl(paths["input"])

    outputs: List[Document] = []
    for doc in docs:
        doc_id = str(doc.get("doc_id", ""))
        if not should_process_doc(doc_id, doc_id_prefixes):
            continue

        text = doc.get(text_field, "")
        if not isinstance(text, str) or not text:
            outputs.append({"doc_id": doc_id, "model": model_name, "entities": []})
            continue

        entities = predict_doc_entities(
            model=model,
            text=text,
            labels=labels,
            label_to_type=label_to_type,
            threshold=threshold,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        outputs.append({"doc_id": doc_id, "model": model_name, "entities": entities})

    write_jsonl(paths["output_predictions"], outputs)


if __name__ == "__main__":
    main()
