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


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _run_metrics_dir(path: Path) -> Path:
    return path.parent / "runs" / path.stem


def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


def _numeric_metric_keys(runs: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for run in runs:
        for key, value in run.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def build_metrics_summary(
    *,
    summary_path: Path,
    runs_dir: Path,
) -> Dict[str, Any]:
    run_files = sorted(runs_dir.glob("*.json"))
    runs = [
        run
        for run_file in run_files
        if (run := _read_json(run_file)) is not None
    ]

    if not runs:
        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_count": 0,
            "run_files": [],
        }

    latest_run = max(runs, key=lambda run: str(run.get("created_at", "")))
    summary: Dict[str, Any] = {
        "variant": latest_run.get("variant"),
        "stage": latest_run.get("stage"),
        "config_path": latest_run.get("config_path"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "runs_dir": str(runs_dir),
        "run_files": [str(path) for path in run_files],
        "latest_run": latest_run,
        "averages": {},
        "metric_stats": {},
    }

    for key in _numeric_metric_keys(runs):
        values = [
            float(run[key])
            for run in runs
            if isinstance(run.get(key), (int, float)) and not isinstance(run.get(key), bool)
        ]
        if not values:
            continue

        average = statistics.fmean(values)
        summary["averages"][key] = average
        summary["metric_stats"][key] = {
            "avg": average,
            "min": min(values),
            "max": max(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

        # Keep the historical top-level shape useful for quick inspection.
        summary[key] = average

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    return summary


def _migrate_existing_metric(path: Path, runs_dir: Path) -> None:
    if not path.exists() or any(runs_dir.glob("*.json")):
        return

    existing = _read_json(path)
    if not existing or "run_count" in existing or "runtime_seconds" not in existing:
        return

    migrated = dict(existing)
    migrated.setdefault("run_id", "migrated_existing")
    run_path = runs_dir / f"{migrated['run_id']}.json"
    with run_path.open("w", encoding="utf-8") as f:
        json.dump(migrated, f, indent=2)
        f.write("\n")


def write_metrics(path: str | Path, metrics: Dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    runs_dir = _run_metrics_dir(path_obj)
    runs_dir.mkdir(parents=True, exist_ok=True)
    _migrate_existing_metric(path_obj, runs_dir)

    run_metrics = dict(metrics)
    run_metrics.setdefault("run_id", _run_id())
    run_path = runs_dir / f"{run_metrics['run_id']}.json"
    suffix = 1
    while run_path.exists():
        run_path = runs_dir / f"{run_metrics['run_id']}_{suffix}.json"
        suffix += 1

    with run_path.open("w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)
        f.write("\n")

    summary = build_metrics_summary(summary_path=path_obj, runs_dir=runs_dir)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
