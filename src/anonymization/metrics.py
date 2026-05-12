import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def default_metrics_path(variant_name: str) -> Path:
    safe_name = variant_name.replace("/", "_").replace(" ", "_")
    return Path("outputs/anonymizer_metrics") / f"{safe_name}.json"


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_run_metrics(
    *,
    variant_name: str,
    stage_name: str,
    doc_count: int,
    char_count: int,
    entity_count: int,
    runtime_seconds: float,
    document_latencies_seconds: List[float],
    config_path: str,
) -> Dict[str, Any]:
    latency_ms = [value * 1000.0 for value in document_latencies_seconds]
    throughput_docs = doc_count / runtime_seconds if runtime_seconds > 0 else 0.0
    throughput_chars = char_count / runtime_seconds if runtime_seconds > 0 else 0.0

    return {
        "variant": variant_name,
        "stage": stage_name,
        "config_path": config_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "doc_count": doc_count,
        "char_count": char_count,
        "entity_count": entity_count,
        "runtime_seconds": runtime_seconds,
        "throughput_docs_per_sec": throughput_docs,
        "throughput_chars_per_sec": throughput_chars,
        "latency_avg_ms": statistics.fmean(latency_ms) if latency_ms else 0.0,
        "latency_p50_ms": percentile(latency_ms, 0.50),
        "latency_p95_ms": percentile(latency_ms, 0.95),
        "latency_max_ms": max(latency_ms) if latency_ms else 0.0,
    }


def write_metrics(path: str | Path, metrics: Dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
