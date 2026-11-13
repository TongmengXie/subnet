> Historical document, archived 2026-11-13. Superseded by [CURRENT_EXPERIMENT.md](../../CURRENT_EXPERIMENT.md). Commands and plans below are not current authorization.

# NeuroStrike follow-up: Viking HPC

This is a **patch** for your existing `edge_learning/` directory. It does not replace `setup_viking.sh`, `run_viking.py`, or the original notebook. Copy the three files into the directory containing your existing Slurm script.

## What changes

- `NeuroStrike_subnetwork_localization_next.ipynb`: keeps the old pipeline, but when `ADAPTER_CHECKPOINT_DIR` is set it **loads existing trained adapters instead of training again**. It also accepts optional extra harmful prompts in a CSV with a `prompt` column.
- `next_experiment.py`: uses the existing StrongREJECT generations; adds a self-channel-only bottleneck counterfactual; evaluates responses with Llama Guard 3 1B; tests SciQ **test** split multiple-choice accuracy; saves intervention RMS diagnostics and a comparison CSV.
- `viking_next.sbatch`: submits the new notebook with existing `run_viking.py` and Hugging Face cache. It requires a completed previous run containing both adapters and training history.

## Requirements

- Existing `edge_learning/` setup completed and `.venv`, `NeuroStrike/`, `run_viking.py`, and `hf_cache/` present.
- Existing **full run** checkpoint directory `results/OLD_JOB_ID/` containing `bottleneck_adapter.pt`, `independent_adapter.pt`, `training_history.csv`.
- Hugging Face account must have access to **both** `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-Guard-3-1B`. The second is used only for response-level judging; it may require separate approval.
- Reuse original `SFT_SEED`, `SFT_TRAIN_N`, `SFT_VAL_N`, `SFT_SCIQ_SHARE`, `RANK` (currently 8). Defaults assume the original 500/100 run, seed 42.

## Commands (from Viking interactive SSH session)

```bash
cd ~/edge_learning             # or your actual project path
ls results/OLD_JOB_ID/{bottleneck_adapter.pt,independent_adapter.pt,training_history.csv}

# Smoke: reuses checkpoints, judges disabled, 8 harmful + 8 SciQ prompts.
sbatch --account=YOUR_PROJECT_CODE \
  --export=ALL,ADAPTER_CHECKPOINT_DIR="$PWD/results/OLD_JOB_ID",SMOKE=1 \
  viking_next.sbatch

# Full: 200 StrongREJECT harmful prompts, Llama Guard response-level scoring,
# 100 SciQ test examples. Reuses trained adapters, no additional training.
sbatch --account=YOUR_PROJECT_CODE \
  --export=ALL,ADAPTER_CHECKPOINT_DIR="$PWD/results/OLD_JOB_ID" \
  viking_next.sbatch

squeue -u "$USER"
sacct -j NEW_JOB_ID --format=JobID,State,ExitCode,Elapsed
cat results/NEW_JOB_ID/next_comparison.csv
cat results/NEW_JOB_ID/mixing_intervention_norms.csv | head
```

To add a harder/adversarial prompt set, prepare `hard_prompts.csv` with a header `prompt`, then submit with:

```bash
sbatch --account=YOUR_PROJECT_CODE \
  --export=ALL,ADAPTER_CHECKPOINT_DIR="$PWD/results/OLD_JOB_ID",EVAL_EXTRA_PROMPTS_CSV="$PWD/hard_prompts.csv" \
  viking_next.sbatch
```

Check Llama Guard access **interactively** before full run:

```bash
export HF_HOME="$PWD/hf_cache"
source .venv/bin/activate
hf download meta-llama/Llama-Guard-3-1B config.json
```

Do not paste tokens into Slurm scripts. If access is denied, obtain access to the judge model first; `NEXT_JUDGE=0` permits a **non-judged** run, but it cannot support safety claims.

## Outputs

- `results/NEW_JOB_ID/next_comparison.csv`: `unsafe_response_rate` (lower is preferable), `benign_accuracy` (higher preferable), counts and invalid-choice rate for Original, Independent, Bottleneck, and `self_only`.
- `harmful_generations.csv`, `harmful_judged.csv`: individual prompt/response and Llama Guard verdict (if enabled).
- `sciq_test_predictions.csv`: per-question SciQ accuracy.
- `mixing_intervention_norms.csv`: full vs self-only correction magnitudes on a small set of untouched model activations.
- Existing notebook results/figures remain, including the pruning red-team baseline, which is **excluded from the new main comparison table**.

## Interpretation limits

- **Llama Guard unsafe-response rate is not a direct attack success rate (ASR)**. An unsafe classification does not necessarily imply the model complied with a harmful instruction; judge quality must be audited on examples.
- `self_only` preserves each channel's nonlinear single-input contribution `sum_h B[i,h] tanh(A[h,i] x[i])` while removing dependence on other input channels. This is **not** a retrained parameter-matched independent adapter. It may change the correction norm; inspect `mixing_intervention_norms.csv` before attributing changes to cross-channel connectivity.
- The first 200 StrongREJECT prompts may still be saturated. If the unaltered model remains 0% unsafe under the judge, add a genuinely harder evaluation set or choose a task where the base model has measurable errors.
- This patch has passed Python parsing and shell syntax checks but has **not** been executed on Viking or against gated Hugging Face models.
