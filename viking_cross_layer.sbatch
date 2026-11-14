#!/usr/bin/env bash
#SBATCH --job-name=neurostrike-cross-layer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
# Supply your project account in the sbatch command (do not hardcode secrets).
set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PLACEMENT="${CROSS_PLACEMENT:-selected_mlp}"
SCOPE="${CROSS_SCOPE:-pair}"
METHOD="${CROSS_METHOD:-connected}"
SEED="${CROSS_SEED:-42}"
ACTIVATION="${CROSS_ACTIVATION:-}"
BIAS="${CROSS_BIAS:-0}"
NEURON_SELECTION="${CROSS_NEURON_SELECTION:-neurostrike}"
OUTPUT="${SLURM_SUBMIT_DIR}/results/${SCOPE}_${METHOD}_${PLACEMENT}_seed${SEED}_${SLURM_JOB_ID}"
if [[ -n "$ACTIVATION" || "$BIAS" == "1" ]]; then
  OUTPUT="${OUTPUT}_activation${ACTIVATION:-default}_bias${BIAS}"
fi
if [[ "$NEURON_SELECTION" == "random" ]]; then
  OUTPUT="${OUTPUT}_random${CROSS_SELECTION_SEED:?required for random selection}"
fi
mkdir -p "$OUTPUT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python -u test_cross_layer.py
python -u test_cross_layer_hf_cpu.py
python -u test_adjacent_adapters.py
if [[ "$METHOD" == "direct" ]]; then python -u test_direct_edges.py; fi
if [[ -n "$ACTIVATION" || "$BIAS" == "1" ]]; then python -u test_architecture_variants.py; fi
ARGS=()
ARGS+=(--neuron-selection "$NEURON_SELECTION")
if [[ "$NEURON_SELECTION" == "random" ]]; then ARGS+=(--selection-seed "$CROSS_SELECTION_SEED"); fi
if [[ -n "$ACTIVATION" ]]; then ARGS+=(--activation "$ACTIVATION"); fi
if [[ "$BIAS" == "1" ]]; then ARGS+=(--bias); fi
if [[ "${SMOKE:-0}" == "1" ]]; then ARGS+=(--smoke); fi
python -u train_cross_layer.py --repo NeuroStrike --source 7 --target 8 \
  --scope "$SCOPE" --method "$METHOD" --seed "$SEED" \
  --rank "${CROSS_RANK:-8}" --placement "$PLACEMENT" --out "$OUTPUT" "${ARGS[@]}"
