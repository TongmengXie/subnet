#!/usr/bin/env bash
#SBATCH --job-name=neurostrike-width
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-32%2
#SBATCH --output=neurostrike-width-%A_%a.log
#SBATCH --error=neurostrike-width-%A_%a.err
set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u "${1:?sweep directory required}/code/run_width_sweep.py" --sweep "$1" --task "${SLURM_ARRAY_TASK_ID:?}"
