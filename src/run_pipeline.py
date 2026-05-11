import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Load YAML file and return as dictionary
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Print and run command
def run_cmd(cmd: list[str], dry_run: bool = False) -> dict:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return {"runtime_seconds": 0.0, "returncode": None}
    start = time.time()
    result = subprocess.run(cmd)
    runtime = time.time() - start
    # Stop pipeline if command failed
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return {"runtime_seconds": round(runtime, 4), "returncode": result.returncode}

# Ensure file exists
def must_exist(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Not found: {path}")

def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without running them")
    parser.add_argument("--metrics-output", help="Optional JSONL file for per-step runtime metrics")
    args = parser.parse_args()

    # Load pipeline configuration
    pipeline_cfg = load_yaml(args.config)
    # Get steps section
    steps = pipeline_cfg.get("steps", {})

    # Validate steps structure
    if not isinstance(steps, dict) or not steps:
        raise ValueError("pipeline.yaml must contain a non-empty 'steps' mapping.")

    metrics = []

    # Iterate over pipeline steps
    for step_name, step_cfg in steps.items():

        # Each step must be a dictionary
        if not isinstance(step_cfg, dict):
            raise ValueError(f"Step '{step_name}' must be a mapping/object in YAML.")

        # Skip disabled steps
        if not step_cfg.get("enabled", True):
            print(f"\nSkipping {step_name} (enabled: false)")
            continue

        # Ensure required fields exist
        if "script" not in step_cfg:
            raise KeyError(f"Step '{step_name}' must define 'script'.")
        if "config" not in step_cfg and "args" not in step_cfg:
            raise KeyError(f"Step '{step_name}' must define 'config' or 'args'.")

        script = step_cfg["script"]
        must_exist(script)

        # Build command — support both --config and free-form args
        cmd = [sys.executable, script]
        if "config" in step_cfg:
            must_exist(step_cfg["config"])
            cmd += ["--config", step_cfg["config"]]
        if "args" in step_cfg:
            cmd += [str(a) for a in step_cfg["args"]]

        # Run pipeline step
        print(f"\nRunning {step_name}")
        step_metric = run_cmd(cmd, dry_run=args.dry_run)
        metrics.append(
            {
                "step": step_name,
                "script": script,
                "config": step_cfg.get("config"),
                "args": step_cfg.get("args", []),
                **step_metric,
            }
        )

    if args.dry_run:
        print("\nPipeline dry run finished.")
    else:
        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                for metric in metrics:
                    f.write(json.dumps(metric, ensure_ascii=False) + "\n")
            print(f"\nPipeline metrics written to {args.metrics_output}")
        print("\nPipeline finished.")


if __name__ == "__main__":
    main()
