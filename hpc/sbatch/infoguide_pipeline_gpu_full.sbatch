#!/bin/bash
#SBATCH --job-name=infoguide_full_gpu
#SBATCH --account=users
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --output=hpc/logs/full_gpu_%j.out
#SBATCH --error=hpc/logs/full_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p hpc/logs

cd "${SLURM_SUBMIT_DIR}"

# Adjust this block to match the environment setup on your cluster.
# Examples:
# module load miniconda cuda
# conda activate infoguide_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export INFOGUIDE_ST_DEVICE=cuda
export INFOGUIDE_SPACY_GPU=1

RUN_DIR="hpc/runs/full_gpu_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}" hpc/outputs
GPU_MONITOR_PID=""

cleanup_gpu_monitor() {
  if [ -n "${GPU_MONITOR_PID}" ]; then
    kill "${GPU_MONITOR_PID}" || true
  fi
}
trap cleanup_gpu_monitor EXIT

date
nvidia-smi

if command -v nvidia-smi >/dev/null 2>&1; then
  bash hpc/scripts/monitor_gpu.sh "${RUN_DIR}/gpu_monitor.csv" 5 &
  GPU_MONITOR_PID=$!
fi

/usr/bin/time -v -o "${RUN_DIR}/resource_usage.txt" \
  python3 src/run_pipeline.py \
    --config hpc/configs/pipeline_full_gpu.yaml \
    --metrics-output "${RUN_DIR}/pipeline_metrics.jsonl"

cleanup_gpu_monitor
GPU_MONITOR_PID=""

echo "${RUN_DIR}" > hpc/outputs/latest_gpu_run_dir.txt

if [ -n "${CPU_RUN_DIR:-}" ]; then
  python3 hpc/scripts/summarize_cpu_gpu_metrics.py \
    --cpu-run-dir "${CPU_RUN_DIR}" \
    --gpu-run-dir "${RUN_DIR}" \
    --output-dir hpc/outputs
elif [ -f hpc/outputs/latest_cpu_run_dir.txt ]; then
  python3 hpc/scripts/summarize_cpu_gpu_metrics.py \
    --cpu-run-dir "$(cat hpc/outputs/latest_cpu_run_dir.txt)" \
    --gpu-run-dir "${RUN_DIR}" \
    --output-dir hpc/outputs
else
  echo "CPU_RUN_DIR not provided and hpc/outputs/latest_cpu_run_dir.txt not found; skipping CPU/GPU summary."
fi

nvidia-smi
date
