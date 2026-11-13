#!/usr/bin/env bash
# Manual invocation only. Submit five independent training jobs on Viking.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash submit_placement_sweep.sh YOUR_PROJECT_CODE" >&2
  exit 2
fi
ACCOUNT="$1"
for mode in pre_attention post_input_norm_pre_attention post_attention_pre_norm post_norm_pre_mlp selected_mlp; do
  echo "Submitting: $mode"
  sbatch --account="$ACCOUNT" --export="ALL,CROSS_PLACEMENT=$mode,CROSS_RANK=8" viking_cross_layer.sbatch
done
