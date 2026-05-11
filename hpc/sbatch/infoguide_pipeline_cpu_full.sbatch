#!/bin/bash
#SBATCH --job-name=infoguide_full_cpu
#SBATCH --account=users
#SBATCH --partition=short_mdbf
#SBATCH --qos=short_mdbf
#SBATCH --output=hpc/logs/full_cpu_%j.out
#SBATCH --error=hpc/logs/full_cpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

mkdir -p hpc/logs

cd "${SLURM_SUBMIT_DIR}"

# Adjust this block to match the environment setup on your cluster.
# Examples:
# module load miniconda
# conda activate infoguide_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

RUN_DIR="hpc/runs/full_cpu_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}" hpc/outputs

date

/usr/bin/time -v -o "${RUN_DIR}/resource_usage.txt" \
  python3 src/run_pipeline.py \
    --config hpc/configs/pipeline_full_cpu.yaml \
    --metrics-output "${RUN_DIR}/pipeline_metrics.jsonl"

echo "${RUN_DIR}" > hpc/outputs/latest_cpu_run_dir.txt

date
