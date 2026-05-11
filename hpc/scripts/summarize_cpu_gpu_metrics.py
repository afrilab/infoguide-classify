#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "hpc" / "outputs"

CLASSIFICATION_RUNS = [
    "tfidf",
    "embedding",
    "embedding_large",
    "zeroshot",
    "zeroshot_large",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_step_metrics(run_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(run_dir / "pipeline_metrics.jsonl")


def step_runtime_map(metrics: list[dict[str, Any]]) -> dict[str, float]:
    return {
        str(row["step"]): float(row.get("runtime_seconds") or 0.0)
        for row in metrics
    }


def parse_gnu_time(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {"available": True}

    elapsed = re.search(r"Elapsed \(wall clock\) time.*:\s*(.+)", text)
    max_rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    user_time = re.search(r"User time \(seconds\):\s*([\d.]+)", text)
    sys_time = re.search(r"System time \(seconds\):\s*([\d.]+)", text)
    cpu_pct = re.search(r"Percent of CPU this job got:\s*(.+)", text)

    if elapsed:
        out["elapsed_raw"] = elapsed.group(1).strip()
    if max_rss:
        out["max_resident_set_kb"] = int(max_rss.group(1))
        out["max_resident_set_mb"] = round(int(max_rss.group(1)) / 1024, 2)
    if user_time:
        out["user_time_seconds"] = float(user_time.group(1))
    if sys_time:
        out["system_time_seconds"] = float(sys_time.group(1))
    if cpu_pct:
        out["cpu_percent_raw"] = cpu_pct.group(1).strip()
    return out


def parse_gpu_monitor(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"available": False}

    def nums(column: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            raw = (row.get(column) or "").strip()
            if raw:
                try:
                    values.append(float(raw))
                except ValueError:
                    pass
        return values

    util = nums("utilization.gpu [%]")
    mem_used = nums("memory.used [MiB]")
    mem_total = nums("memory.total [MiB]")

    out: dict[str, Any] = {
        "available": True,
        "samples": len(rows),
    }
    if util:
        out["gpu_utilization_max_pct"] = max(util)
        out["gpu_utilization_mean_pct"] = round(sum(util) / len(util), 2)
    if mem_used:
        out["gpu_memory_used_max_mib"] = max(mem_used)
        out["gpu_memory_used_mean_mib"] = round(sum(mem_used) / len(mem_used), 2)
    if mem_total:
        out["gpu_memory_total_mib"] = max(mem_total)
    return out


def classification_output_path(run_name: str, device: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "classified"
        / f"classification_results__{run_name}_{device}_full_pipeline.jsonl"
    )


def compare_classification_outputs() -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for run_name in CLASSIFICATION_RUNS:
        cpu_path = classification_output_path(run_name, "cpu")
        gpu_path = classification_output_path(run_name, "gpu")
        cpu_rows = read_jsonl(cpu_path)
        gpu_rows = read_jsonl(gpu_path)

        cpu_by_doc = {str(row.get("doc_id")): row for row in cpu_rows}
        gpu_by_doc = {str(row.get("doc_id")): row for row in gpu_rows}
        common_ids = sorted(set(cpu_by_doc).intersection(gpu_by_doc))
        cpu_only = sorted(set(cpu_by_doc).difference(gpu_by_doc))
        gpu_only = sorted(set(gpu_by_doc).difference(cpu_by_doc))

        same_label = 0
        confidence_abs_diffs: list[float] = []
        for doc_id in common_ids:
            cpu_row = cpu_by_doc[doc_id]
            gpu_row = gpu_by_doc[doc_id]
            if cpu_row.get("predicted_label") == gpu_row.get("predicted_label"):
                same_label += 1
            try:
                confidence_abs_diffs.append(
                    abs(float(cpu_row.get("confidence") or 0.0) - float(gpu_row.get("confidence") or 0.0))
                )
            except (TypeError, ValueError):
                pass

        comparisons.append(
            {
                "run": run_name,
                "cpu_file": str(cpu_path.relative_to(PROJECT_ROOT)),
                "gpu_file": str(gpu_path.relative_to(PROJECT_ROOT)),
                "cpu_rows": len(cpu_rows),
                "gpu_rows": len(gpu_rows),
                "common_docs": len(common_ids),
                "cpu_only_docs": len(cpu_only),
                "gpu_only_docs": len(gpu_only),
                "same_predicted_label": same_label,
                "label_match_rate": round(same_label / len(common_ids), 6) if common_ids else None,
                "mean_confidence_abs_diff": (
                    round(sum(confidence_abs_diffs) / len(confidence_abs_diffs), 6)
                    if confidence_abs_diffs
                    else None
                ),
                "cpu_sha256": file_sha256(cpu_path),
                "gpu_sha256": file_sha256(gpu_path),
            }
        )
    return comparisons


def build_summary(cpu_run_dir: Path, gpu_run_dir: Path) -> dict[str, Any]:
    cpu_metrics = load_step_metrics(cpu_run_dir)
    gpu_metrics = load_step_metrics(gpu_run_dir)
    cpu_steps = step_runtime_map(cpu_metrics)
    gpu_steps = step_runtime_map(gpu_metrics)
    all_steps = sorted(set(cpu_steps).union(gpu_steps))

    total_docs = count_jsonl(PROJECT_ROOT / "data" / "processed" / "clean_documents.jsonl")
    cpu_total = sum(cpu_steps.values())
    gpu_total = sum(gpu_steps.values())

    per_step: list[dict[str, Any]] = []
    for step in all_steps:
        cpu_time = cpu_steps.get(step)
        gpu_time = gpu_steps.get(step)
        speedup = None
        if cpu_time is not None and gpu_time and gpu_time > 0:
            speedup = cpu_time / gpu_time
        per_step.append(
            {
                "step": step,
                "cpu_runtime_seconds": round(cpu_time, 4) if cpu_time is not None else None,
                "gpu_runtime_seconds": round(gpu_time, 4) if gpu_time is not None else None,
                "speedup_cpu_over_gpu": round(speedup, 4) if speedup is not None else None,
            }
        )

    return {
        "documents": total_docs,
        "cpu_run_dir": str(cpu_run_dir.relative_to(PROJECT_ROOT)),
        "gpu_run_dir": str(gpu_run_dir.relative_to(PROJECT_ROOT)),
        "total_runtime": {
            "cpu_seconds": round(cpu_total, 4),
            "gpu_seconds": round(gpu_total, 4),
            "speedup_cpu_over_gpu": round(cpu_total / gpu_total, 4) if gpu_total > 0 else None,
            "cpu_docs_per_second": round(total_docs / cpu_total, 6) if cpu_total > 0 else None,
            "gpu_docs_per_second": round(total_docs / gpu_total, 6) if gpu_total > 0 else None,
            "parallel_efficiency": None,
            "parallel_efficiency_note": "Not applicable for the current single CPU job vs single GPU job comparison.",
        },
        "resource_usage": {
            "cpu": parse_gnu_time(cpu_run_dir / "resource_usage.txt"),
            "gpu": parse_gnu_time(gpu_run_dir / "resource_usage.txt"),
            "gpu_monitor": parse_gpu_monitor(gpu_run_dir / "gpu_monitor.csv"),
        },
        "per_step_runtime": per_step,
        "output_consistency": {
            "classification": compare_classification_outputs(),
        },
    }


def write_outputs(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "cpu_gpu_metrics_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = out_dir / "cpu_gpu_step_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["step", "cpu_runtime_seconds", "gpu_runtime_seconds", "speedup_cpu_over_gpu"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["per_step_runtime"])

    consistency_path = out_dir / "cpu_gpu_output_consistency.csv"
    with consistency_path.open("w", encoding="utf-8", newline="") as f:
        rows = summary["output_consistency"]["classification"]
        fieldnames = [
            "run",
            "cpu_rows",
            "gpu_rows",
            "common_docs",
            "cpu_only_docs",
            "gpu_only_docs",
            "same_predicted_label",
            "label_match_rate",
            "mean_confidence_abs_diff",
            "cpu_sha256",
            "gpu_sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fieldnames} for row in rows)

    md_path = out_dir / "cpu_gpu_metrics_summary.md"
    total = summary["total_runtime"]
    resources = summary["resource_usage"]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# CPU vs GPU Pipeline Metrics\n\n")
        f.write("## End-to-End\n\n")
        f.write("| Metric | CPU | GPU |\n|---|---:|---:|\n")
        f.write(f"| Runtime seconds | {total['cpu_seconds']} | {total['gpu_seconds']} |\n")
        f.write(f"| Documents/second | {total['cpu_docs_per_second']} | {total['gpu_docs_per_second']} |\n")
        f.write(f"| Speedup CPU/GPU |  | {total['speedup_cpu_over_gpu']} |\n\n")
        f.write(f"Parallel efficiency: {total['parallel_efficiency_note']}\n\n")

        f.write("## Resource Usage\n\n")
        f.write("| Metric | CPU job | GPU job |\n|---|---:|---:|\n")
        f.write(
            "| Max resident set MB | "
            f"{resources['cpu'].get('max_resident_set_mb', 'n/a')} | "
            f"{resources['gpu'].get('max_resident_set_mb', 'n/a')} |\n"
        )
        f.write(
            "| GPU memory max MiB | n/a | "
            f"{resources['gpu_monitor'].get('gpu_memory_used_max_mib', 'n/a')} |\n"
        )
        f.write(
            "| GPU utilization mean % | n/a | "
            f"{resources['gpu_monitor'].get('gpu_utilization_mean_pct', 'n/a')} |\n\n"
        )

        f.write("## Per-Step Runtime\n\n")
        f.write("| Step | CPU sec | GPU sec | Speedup |\n|---|---:|---:|---:|\n")
        for row in summary["per_step_runtime"]:
            f.write(
                f"| {row['step']} | {row['cpu_runtime_seconds']} | "
                f"{row['gpu_runtime_seconds']} | {row['speedup_cpu_over_gpu']} |\n"
            )
        f.write("\n## Output Consistency\n\n")
        f.write("| Run | CPU rows | GPU rows | Common docs | Label match rate | Mean confidence abs diff |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary["output_consistency"]["classification"]:
            f.write(
                f"| {row['run']} | {row['cpu_rows']} | {row['gpu_rows']} | "
                f"{row['common_docs']} | {row['label_match_rate']} | "
                f"{row['mean_confidence_abs_diff']} |\n"
            )

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {consistency_path}")
    print(f"Saved {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full-pipeline CPU/GPU HPC metrics.")
    parser.add_argument("--cpu-run-dir", required=True, help="Example: hpc/runs/full_cpu_12345")
    parser.add_argument("--gpu-run-dir", required=True, help="Example: hpc/runs/full_gpu_12346")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    cpu_run_dir = (PROJECT_ROOT / args.cpu_run_dir).resolve()
    gpu_run_dir = (PROJECT_ROOT / args.gpu_run_dir).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()

    summary = build_summary(cpu_run_dir=cpu_run_dir, gpu_run_dir=gpu_run_dir)
    write_outputs(summary, out_dir)


if __name__ == "__main__":
    main()
