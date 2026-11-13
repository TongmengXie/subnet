#!/usr/bin/env bash
#SBATCH --job-name=neurostrike-followup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=0-08:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
# Supply --account=YOUR_PROJECT_CODE when calling sbatch (Viking requires it).
set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
cd "${SLURM_SUBMIT_DIR:?Submit this job from the Viking project directory}"
WORKDIR="$PWD"
source "$WORKDIR/.venv/bin/activate"
export NEUROSTRIKE_REPO="$WORKDIR/NeuroStrike"
export HF_HOME="$WORKDIR/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Require a previously completed full run; this follow-up reuses both trained adapters.
: "${ADAPTER_CHECKPOINT_DIR:?Set ADAPTER_CHECKPOINT_DIR to results/OLD_JOB_ID}"
export ADAPTER_CHECKPOINT_DIR="$(realpath "$ADAPTER_CHECKPOINT_DIR")"
[[ -f "$ADAPTER_CHECKPOINT_DIR/bottleneck_adapter.pt" && -f "$ADAPTER_CHECKPOINT_DIR/independent_adapter.pt" ]] || {
  echo 'Missing adapter checkpoints in ADAPTER_CHECKPOINT_DIR' >&2; exit 2;
}
export SUBNETWORK_RESULTS_DIR="$WORKDIR/results/${SLURM_JOB_ID}"
export EVAL_HARMFUL_N="${EVAL_HARMFUL_N:-200}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
export EVAL_LOGIT_N="${EVAL_LOGIT_N:-8}"
export NEXT_SCIQ_N="${NEXT_SCIQ_N:-100}"
export NEXT_NORM_N="${NEXT_NORM_N:-8}"
export NEXT_JUDGE="${NEXT_JUDGE:-1}"
mkdir -p "$SUBNETWORK_RESULTS_DIR"
if [[ "${SMOKE:-0}" == '1' ]]; then
  # Checkpoints were trained on 500 examples: do NOT change SFT_TRAIN_N/VAL_N.
  export EVAL_HARMFUL_N=8 EVAL_MAX_NEW_TOKENS=32 EVAL_LOGIT_N=2
  export NEXT_SCIQ_N=8 NEXT_NORM_N=2 NEXT_JUDGE=0
fi
printf 'Job %s on %s at %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA; check driver/wheel'
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),flush=True)
PY
python -m pip freeze > "$SUBNETWORK_RESULTS_DIR/pip_freeze.txt"
git -C "$NEUROSTRIKE_REPO" rev-parse HEAD > "$SUBNETWORK_RESULTS_DIR/neurostrike_commit.txt"
python -u run_viking.py \
  --notebook NeuroStrike_subnetwork_localization_next.ipynb \
  --repo "$NEUROSTRIKE_REPO" \
  --executed "$SUBNETWORK_RESULTS_DIR/executed.ipynb"
printf 'Finished at %s. Results: %s\n' "$(date -Is)" "$SUBNETWORK_RESULTS_DIR"
