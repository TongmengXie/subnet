#!/usr/bin/env bash
#SBATCH --job-name=neurostrike-variant-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-3%2
#SBATCH --output=neurostrike-variants-%A_%a.log
#SBATCH --error=neurostrike-variants-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CROSS_SCOPE=all-adjacent CROSS_PLACEMENT=selected_mlp CROSS_SEED=42
export CROSS_RANK=8 SMOKE=1 OMP_NUM_THREADS=1
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) export CROSS_METHOD=connected CROSS_ACTIVATION=tanh CROSS_BIAS=1 ;;
  1) export CROSS_METHOD=connected CROSS_ACTIVATION=identity CROSS_BIAS=0 ;;
  2) export CROSS_METHOD=connected CROSS_ACTIVATION=identity CROSS_BIAS=1 ;;
  3) export CROSS_METHOD=direct CROSS_ACTIVATION=identity CROSS_BIAS=1 ;;
  *) echo "Unexpected task ID" >&2; exit 2 ;;
esac
echo "Variant task=${SLURM_ARRAY_TASK_ID} method=${CROSS_METHOD} activation=${CROSS_ACTIVATION} bias=${CROSS_BIAS}"
bash viking_cross_layer.sbatch
