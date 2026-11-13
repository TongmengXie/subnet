#!/usr/bin/env bash
#SBATCH --job-name=neurostrike-only-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u test_neurostrike_only.py
python -u evaluate_neurostrike_only.py --out "results/neurostrike_only_${SLURM_JOB_ID}"
