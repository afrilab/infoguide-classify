import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


Entity = Dict[str, Any]
Document = Dict[str, Any]

def format_score(value: float) -> str:
    return f"{value:.4f}"

def load_json_or_jsonl(path: str) -> List[Document]:
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


def safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def compute_scores(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def get_variations(entity: Entity) -> List[str]:
    variations = entity.get("variation")

    if isinstance(variations, list):
        return [str(variation) for variation in variations] or ["unknown"]

    if variations:
        return [str(variations)]

    return ["unknown"]

def spans_overlap(a: Entity, b: Entity) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def is_strict_match(gt: Entity, pred: Entity) -> bool:
    return (
        gt["type"] == pred["type"]
        and gt["start"] == pred["start"]
        and gt["end"] == pred["end"]
    )


def is_relaxed_match(gt: Entity, pred: Entity) -> bool:
    return gt["type"] == pred["type"] and spans_overlap(gt, pred)


def span_contains(container: Entity, inner: Entity) -> bool:
    return container["start"] <= inner["start"] and inner["end"] <= container["end"]


def predictions_cover_gold_text(gt: Entity, predictions: List[Entity]) -> bool:
    gold_text = gt.get("text", "")
    if not gold_text:
        return False

    covered_offsets = set()
    for pred in predictions:
        overlap_start = max(gt["start"], pred["start"])
        overlap_end = min(gt["end"], pred["end"])
        covered_offsets.update(range(overlap_start, overlap_end))

    for relative_index, char in enumerate(gold_text):
        if char.isalnum() and gt["start"] + relative_index not in covered_offsets:
            return False

    return True


def normalize_entity(entity: Entity) -> Entity:
    return {
        "type": entity["type"],
        "text": entity.get("text", ""),
        "start": int(entity["start"]),
        "end": int(entity["end"]),
        **({"score": entity["score"]} if "score" in entity else {}),
        **({"difficulty": entity["difficulty"]} if "difficulty" in entity else {}),
        **({"variation": entity["variation"]} if "variation" in entity else {}),
    }


def make_doc_map(docs: List[Document]) -> Dict[str, Document]:
    return {doc["doc_id"]: doc for doc in docs if doc["doc_id"].startswith("syn_doc")}


def match_entities(
    gt_entities: List[Entity],
    pred_entities: List[Entity],
    mode: str,
) -> Tuple[List[Tuple[Entity, Entity]], List[Entity], List[Entity]]:
    matcher = is_strict_match if mode == "strict" else is_relaxed_match

    matches = []
    used_pred_indexes = set()

    for gt in gt_entities:
        best_index = None

        for pred_index, pred in enumerate(pred_entities):
            if pred_index in used_pred_indexes:
                continue

            if matcher(gt, pred):
                best_index = pred_index
                break

        if best_index is not None:
            used_pred_indexes.add(best_index)
            matches.append((gt, pred_entities[best_index]))
            continue

        if mode == "strict":
            candidate_indexes = [
                pred_index
                for pred_index, pred in enumerate(pred_entities)
                if pred_index not in used_pred_indexes and is_relaxed_match(gt, pred)
            ]
            candidate_predictions = [
                pred_entities[pred_index] for pred_index in candidate_indexes
            ]

            if predictions_cover_gold_text(gt, candidate_predictions):
                used_pred_indexes.update(candidate_indexes)
                matches.append((gt, candidate_predictions[0]))

    matched_gt_ids = {id(gt) for gt, _ in matches}

    false_negatives = [gt for gt in gt_entities if id(gt) not in matched_gt_ids]
    false_positives = [
        pred
        for pred_index, pred in enumerate(pred_entities)
        if pred_index not in used_pred_indexes
    ]

    if mode == "relaxed":
        matched_gold_spans = [gt for gt, _ in matches]
        false_positives = [
            pred
            for pred in false_positives
            if not any(
                pred["type"] == gt["type"] and span_contains(gt, pred)
                for gt in matched_gold_spans
            )
        ]

    return matches, false_negatives, false_positives


def update_counter(counter: Dict[str, Dict[str, int]], key: str, tp=0, fp=0, fn=0):
    if key not in counter:
        counter[key] = {"tp": 0, "fp": 0, "fn": 0}

    counter[key]["tp"] += tp
    counter[key]["fp"] += fp
    counter[key]["fn"] += fn


def evaluate_negative_cases(
    benchmark_doc: Document,
    pred_entities: List[Entity],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    negative_cases = benchmark_doc.get("negative_cases", [])
    total = len(negative_cases)
    incorrectly_detected = 0
    rows = []

    for case in negative_cases:
        case_type = case.get("type")
        case_text = case.get("text")

        matched_predictions = [
            pred
            for pred in pred_entities
            if pred["type"] == case_type and pred.get("text") == case_text
        ]

        is_detected = len(matched_predictions) > 0

        if is_detected:
            incorrectly_detected += 1

        rows.append(
            {
                "doc_id": benchmark_doc["doc_id"],
                "type": case_type,
                "text": case_text,
                "incorrectly_detected": is_detected,
                "matched_predictions": json.dumps(matched_predictions, ensure_ascii=False),
            }
        )

    return total, incorrectly_detected, rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(model_config: Dict[str, Any], benchmark_path: str):
    model_name = model_config["model"]
    prediction_path = model_config["prediction_path"]
    output_dir = Path(model_config["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_docs = make_doc_map(load_json_or_jsonl(benchmark_path))
    prediction_docs = make_doc_map(load_json_or_jsonl(prediction_path))

    overall = {
        "strict": {"tp": 0, "fp": 0, "fn": 0},
        "relaxed": {"tp": 0, "fp": 0, "fn": 0},
    }

    per_type = {
        "strict": {},
        "relaxed": {},
    }

    per_type_variation = {
        "strict": {},
        "relaxed": {},
    }

    per_difficulty = {
        "strict": {},
        "relaxed": {},
    }

    per_doc_type = {
        "strict": {},
        "relaxed": {},
    }

    errors = []
    negative_rows = []
    negative_total = 0
    negative_incorrect = 0

    for doc_id, benchmark_doc in benchmark_docs.items():
        prediction_doc = prediction_docs.get(doc_id, {})

        gt_entities = [
            normalize_entity(entity)
            for entity in benchmark_doc.get("entities", [])
        ]

        pred_entities = [
            normalize_entity(entity)
            for entity in prediction_doc.get("entities", [])
        ]

        doc_type = benchmark_doc.get("doc_type", "unknown")

        neg_total, neg_incorrect, neg_rows = evaluate_negative_cases(
            benchmark_doc,
            pred_entities,
        )

        negative_total += neg_total
        negative_incorrect += neg_incorrect
        negative_rows.extend(neg_rows)

        for mode in ["strict", "relaxed"]:
            matches, false_negatives, false_positives = match_entities(
                gt_entities,
                pred_entities,
                mode,
            )

            overall[mode]["tp"] += len(matches)
            overall[mode]["fp"] += len(false_positives)
            overall[mode]["fn"] += len(false_negatives)

            for gt, pred in matches:
                entity_type = gt["type"]
                difficulty = gt.get("difficulty", "unknown")

                update_counter(per_type[mode], entity_type, tp=1)
                for variation in get_variations(gt):
                    update_counter(
                        per_type_variation[mode],
                        (entity_type, variation),
                        tp=1,
                    )
                update_counter(per_difficulty[mode], difficulty, tp=1)
                update_counter(per_doc_type[mode], doc_type, tp=1)

            for gt in false_negatives:
                entity_type = gt["type"]
                difficulty = gt.get("difficulty", "unknown")

                update_counter(per_type[mode], entity_type, fn=1)
                for variation in get_variations(gt):
                    update_counter(
                        per_type_variation[mode],
                        (entity_type, variation),
                        fn=1,
                    )
                update_counter(per_difficulty[mode], difficulty, fn=1)
                update_counter(per_doc_type[mode], doc_type, fn=1)

                errors.append(
                    {
                        "error_type": "false_negative",
                        "match_mode": mode,
                        "doc_id": doc_id,
                        "type": entity_type,
                        "ground_truth": {
                            "text": gt.get("text", ""),
                            "start": gt["start"],
                            "end": gt["end"],
                        },
                    }
                )

            for pred in false_positives:
                entity_type = pred["type"]

                update_counter(per_type[mode], entity_type, fp=1)
                update_counter(
                    per_type_variation[mode],
                    (entity_type, "unknown"),
                    fp=1,
                )
                update_counter(per_difficulty[mode], "unknown", fp=1)
                update_counter(per_doc_type[mode], doc_type, fp=1)

                errors.append(
                    {
                        "error_type": "false_positive",
                        "match_mode": mode,
                        "doc_id": doc_id,
                        "type": entity_type,
                        "prediction": {
                            "text": pred.get("text", ""),
                            "start": pred["start"],
                            "end": pred["end"],
                        },
                    }
                )

        strict_matches, _, _ = match_entities(gt_entities, pred_entities, "strict")
        relaxed_matches, _, _ = match_entities(gt_entities, pred_entities, "relaxed")

        strict_pairs = {
            (
                gt["type"],
                gt["start"],
                gt["end"],
                pred["start"],
                pred["end"],
            )
            for gt, pred in strict_matches
        }

        for gt, pred in relaxed_matches:
            pair_key = (
                gt["type"],
                gt["start"],
                gt["end"],
                pred["start"],
                pred["end"],
            )

            if pair_key not in strict_pairs:
                errors.append(
                    {
                        "error_type": "boundary_error",
                        "doc_id": doc_id,
                        "type": gt["type"],
                        "ground_truth": {
                            "text": gt.get("text", ""),
                            "start": gt["start"],
                            "end": gt["end"],
                        },
                        "prediction": {
                            "text": pred.get("text", ""),
                            "start": pred["start"],
                            "end": pred["end"],
                        },
                    }
                )

    summary = {
        "model": model_name,
        "strict": compute_scores(**overall["strict"]),
        "relaxed": compute_scores(**overall["relaxed"]),
        "negative_cases": {
            "total": negative_total,
            "incorrectly_detected": negative_incorrect,
            "false_positive_rate": round(
                safe_div(negative_incorrect, negative_total),
                4,
            ),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    type_names = sorted(
        set(per_type["strict"].keys()) | set(per_type["relaxed"].keys())
    )

    per_type_rows = []
    for entity_type in type_names:
        strict_scores = compute_scores(**per_type["strict"].get(entity_type, {"tp": 0, "fp": 0, "fn": 0}))
        relaxed_scores = compute_scores(**per_type["relaxed"].get(entity_type, {"tp": 0, "fp": 0, "fn": 0}))

        per_type_rows.append(
            {
                "type": entity_type,
                "strict_precision": format_score(strict_scores["precision"]),
                "strict_recall": format_score(strict_scores["recall"]),
                "strict_f1": format_score(strict_scores["f1"]),
                "strict_tp": strict_scores["tp"],
                "strict_fp": strict_scores["fp"],
                "strict_fn": strict_scores["fn"],
                "relaxed_precision": format_score(relaxed_scores["precision"]),
                "relaxed_recall": format_score(relaxed_scores["recall"]),
                "relaxed_f1": format_score(relaxed_scores["f1"]),
                "relaxed_tp": relaxed_scores["tp"],
                "relaxed_fp": relaxed_scores["fp"],
                "relaxed_fn": relaxed_scores["fn"],
            }
        )

    write_csv(
        output_dir / "per_type.csv",
        [
            "type",
            "strict_precision",
            "strict_recall",
            "strict_f1",
            "strict_tp",
            "strict_fp",
            "strict_fn",
            "relaxed_precision",
            "relaxed_recall",
            "relaxed_f1",
            "relaxed_tp",
            "relaxed_fp",
            "relaxed_fn",
        ],
        per_type_rows,
    )

    type_variation_names = sorted(
        set(per_type_variation["strict"].keys())
        | set(per_type_variation["relaxed"].keys())
    )

    per_type_variation_rows = []
    for entity_type, variation in type_variation_names:
        strict_scores = compute_scores(
            **per_type_variation["strict"].get(
                (entity_type, variation),
                {"tp": 0, "fp": 0, "fn": 0},
            )
        )
        relaxed_scores = compute_scores(
            **per_type_variation["relaxed"].get(
                (entity_type, variation),
                {"tp": 0, "fp": 0, "fn": 0},
            )
        )

        per_type_variation_rows.append(
            {
                "type": entity_type,
                "variation": variation,
                "strict_precision": format_score(strict_scores["precision"]),
                "strict_recall": format_score(strict_scores["recall"]),
                "strict_f1": format_score(strict_scores["f1"]),
                "strict_tp": strict_scores["tp"],
                "strict_fp": strict_scores["fp"],
                "strict_fn": strict_scores["fn"],
                "relaxed_precision": format_score(relaxed_scores["precision"]),
                "relaxed_recall": format_score(relaxed_scores["recall"]),
                "relaxed_f1": format_score(relaxed_scores["f1"]),
                "relaxed_tp": relaxed_scores["tp"],
                "relaxed_fp": relaxed_scores["fp"],
                "relaxed_fn": relaxed_scores["fn"],
            }
        )

    write_csv(
        output_dir / "per_type_variation.csv",
        [
            "type",
            "variation",
            "strict_precision",
            "strict_recall",
            "strict_f1",
            "strict_tp",
            "strict_fp",
            "strict_fn",
            "relaxed_precision",
            "relaxed_recall",
            "relaxed_f1",
            "relaxed_tp",
            "relaxed_fp",
            "relaxed_fn",
        ],
        per_type_variation_rows,
    )

    difficulty_names = sorted(
        set(per_difficulty["strict"].keys()) | set(per_difficulty["relaxed"].keys())
    )

    per_difficulty_rows = []
    for difficulty in difficulty_names:
        strict_counts = per_difficulty["strict"].get(difficulty, {"tp": 0, "fp": 0, "fn": 0})
        relaxed_counts = per_difficulty["relaxed"].get(difficulty, {"tp": 0, "fp": 0, "fn": 0})

        strict_scores = compute_scores(**strict_counts)
        relaxed_scores = compute_scores(**relaxed_counts)

        per_difficulty_rows.append(
            {
                "difficulty": difficulty,
                "strict_precision": format_score(strict_scores["precision"]),
                "strict_recall": format_score(strict_scores["recall"]),
                "strict_f1": format_score(strict_scores["f1"]),
                "relaxed_precision": format_score(relaxed_scores["precision"]),
                "relaxed_recall": format_score(relaxed_scores["recall"]),
                "relaxed_f1": format_score(relaxed_scores["f1"]),
                "tp": strict_counts["tp"],
                "fp": strict_counts["fp"],
                "fn": strict_counts["fn"],
            }
        )

    write_csv(
        output_dir / "per_difficulty.csv",
        [
            "difficulty",
            "strict_precision",
            "strict_recall",
            "strict_f1",
            "relaxed_precision",
            "relaxed_recall",
            "relaxed_f1",
            "tp",
            "fp",
            "fn",
        ],
        per_difficulty_rows,
    )

    doc_type_names = sorted(
        set(per_doc_type["strict"].keys()) | set(per_doc_type["relaxed"].keys())
    )

    per_doc_type_rows = []
    for doc_type in doc_type_names:
        strict_counts = per_doc_type["strict"].get(doc_type, {"tp": 0, "fp": 0, "fn": 0})
        relaxed_counts = per_doc_type["relaxed"].get(doc_type, {"tp": 0, "fp": 0, "fn": 0})

        strict_scores = compute_scores(**strict_counts)
        relaxed_scores = compute_scores(**relaxed_counts)

        per_doc_type_rows.append(
            {
                "doc_type": doc_type,
                "strict_precision": format_score(strict_scores["precision"]),
                "strict_recall": format_score(strict_scores["recall"]),
                "strict_f1": format_score(strict_scores["f1"]),
                "relaxed_precision": format_score(relaxed_scores["precision"]),
                "relaxed_recall": format_score(relaxed_scores["recall"]),
                "relaxed_f1": format_score(relaxed_scores["f1"]),
                "tp": strict_counts["tp"],
                "fp": strict_counts["fp"],
                "fn": strict_counts["fn"],
            }
        )

    write_csv(
        output_dir / "per_doc_type.csv",
        [
            "doc_type",
            "strict_precision",
            "strict_recall",
            "strict_f1",
            "relaxed_precision",
            "relaxed_recall",
            "relaxed_f1",
            "tp",
            "fp",
            "fn",
        ],
        per_doc_type_rows,
    )

    write_csv(
        output_dir / "negative_cases.csv",
        [
            "doc_id",
            "type",
            "text",
            "incorrectly_detected",
            "matched_predictions",
        ],
        negative_rows,
    )

    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + "\n")

    print(f"Evaluation completed for model: {model_name}")
    print(f"Outputs written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to anonymization evaluator YAML config.",
    )
    parser.add_argument(
        "--model",
        required=False,
        help="Optional model name. If omitted, all models in config are evaluated.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    benchmark_path = config["benchmark_path"]
    models = config["models"]

    for model_config in models:
        if args.model and model_config["model"] != args.model:
            continue

        evaluate_model(model_config, benchmark_path)


if __name__ == "__main__":
    main()
