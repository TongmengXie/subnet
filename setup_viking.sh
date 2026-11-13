#!/usr/bin/env bash
# Run ONCE from the directory containing this file, after connecting to Viking.
set -euo pipefail
cd "$(dirname "$0")"
WORKDIR="$PWD"
module purge
module load Python/3.11.3-GCCcore-12.3.0
if [[ ! -d NeuroStrike/.git ]]; then
  git clone --depth 1 https://github.com/wu-lichao/NeuroStrike-Neuron-Level-Attacks-on-Aligned-LLMs.git NeuroStrike
fi
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# CUDA 12.1 wheels: driver must support CUDA 12.1 or later. Check nvidia-smi on a GPU node.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-viking.txt
python -m ipykernel install --sys-prefix --name neurostrike-viking --display-name 'Python (NeuroStrike Viking)'
mkdir -p "$WORKDIR/hf_cache" "$WORKDIR/results"
cat <<NEXT
Setup complete.

Next (run on the Viking login node):
  export HF_HOME="$WORKDIR/hf_cache"
  source "$WORKDIR/.venv/bin/activate"
  hf auth login

Ensure you have accepted Meta's access terms for meta-llama/Llama-3.2-1B-Instruct.
Then submit from THIS directory with your actual project code:
  sbatch --account=YOUR_PROJECT_CODE viking_gpu.sbatch
Smoke test:
  sbatch --account=YOUR_PROJECT_CODE --export=ALL,SMOKE=1 viking_gpu.sbatch
NEXT
