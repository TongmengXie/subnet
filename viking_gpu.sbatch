#!/usr/bin/env bash
#SBATCH --job-name=neurostrike-subnetwork
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
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
export SUBNETWORK_RESULTS_DIR="$WORKDIR/results/${SLURM_JOB_ID}"
mkdir -p "$SUBNETWORK_RESULTS_DIR"
if [[ "${SMOKE:-0}" == '1' ]]; then
  export SFT_TRAIN_N=32 SFT_VAL_N=16 SFT_VAL_EVERY=4
  export EVAL_HARMFUL_N=8 EVAL_MAX_NEW_TOKENS=32 EVAL_LOGIT_N=2
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
  --notebook NeuroStrike_subnetwork_localization_viking.ipynb \
  --repo "$NEUROSTRIKE_REPO" \
  --executed "$SUBNETWORK_RESULTS_DIR/executed.ipynb"
printf 'Finished at %s. Results: %s\n' "$(date -Is)" "$SUBNETWORK_RESULTS_DIR"
