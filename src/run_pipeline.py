import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# Load YAML file and return as dictionary
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Print and run command
def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    result = subprocess.run(cmd)
    # Stop pipeline if command failed
    if result.returncode != 0:
        raise SystemExit(result.returncode)

# Ensure file exists
def must_exist(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Not found: {path}")

def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml")
    args = parser.parse_args()

    # Load pipeline configuration
    pipeline_cfg = load_yaml(args.config)
    # Get steps section
    steps = pipeline_cfg.get("steps", {})

    # Validate steps structure
    if not isinstance(steps, dict) or not steps:
        raise ValueError("pipeline.yaml must contain a non-empty 'steps' mapping.")

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
        if "script" not in step_cfg or "config" not in step_cfg:
            raise KeyError(f"Step '{step_name}' must define both 'script' and 'config'.")

        script = step_cfg["script"]
        cfg = step_cfg["config"]

        # Check files exist
        must_exist(script)
        must_exist(cfg)

        # Run pipeline step
        print(f"\nRunning {step_name}")
        run_cmd([sys.executable, script, "--config", cfg])

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
