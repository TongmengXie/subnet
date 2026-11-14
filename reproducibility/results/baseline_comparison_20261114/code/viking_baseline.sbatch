#!/usr/bin/env bash
#SBATCH --job-name=neurostrike-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=neurostrike-baseline-%A_%a.log
#SBATCH --error=neurostrike-baseline-%A_%a.err
set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
ARGS=()
if [[ "${3:-}" == "smoke" ]]; then ARGS+=(--smoke); fi
python -u "$1/code/run_baseline_task.py" --root "$1" --kind "$2" --task "${SLURM_ARRAY_TASK_ID:-0}" "${ARGS[@]}"
