> Historical document, archived 2026-11-13. Superseded by [CURRENT_EXPERIMENT.md](../../CURRENT_EXPERIMENT.md). Commands and plans below are not current authorization.

# Codex handoff — NeuroStrike cross-layer subnetwork localisation

**Working directory:** `~/edge_learning` (Codex starts here). **Date:** 2026-11-13.

## Objective

Study whether pretrained, NeuroStrike-selected safety-related MLP channels can be adapted with a small **trainable cross-layer subnetwork**, while freezing the Llama backbone. The research ambition is **subnetwork localisation, local editability, and reconfigurability**, not merely lowering language-modelling loss. Avoid claiming learned paths are recovered pretrained/causal edges.

Model: `meta-llama/Llama-3.2-1B-Instruct` (gated); NeuroStrike selection artifact expected at `NeuroStrike/pre_computed_sn/weights_Llama-3.2-1B-Instruct.p` (trusted local pickle). Existing `.venv` and `hf_cache` are under the working directory.

## Crucial architecture distinction

**Earlier pilot (different experiment):** adapters on selected channels in 16 layers. Independent-neuron updates versus *within-layer* rank-8 mixing `n_l → 8 → n_l`. Roughly 1,313 selected channel occurrences across 16 layers; both old adapters ~21,008 trainable parameters. Old held-out completion CE ~2.51 (original), ~2.51 (independent), ~2.46 (bottleneck). Lexical refusal on 200 StrongREJECT prompts ~99%, 98%, 98%; pruned ~0%. Training CE was noisy (~1.8–4.9), without clear convergence. These results **are not** the new cross-layer experiment, and lexical refusal is not judge-based ASR.

**Current cross-layer prototype:** one adjacent pair, **source layer 7 → target layer 8**, not all layers. Capture selected SwiGLU MLP intermediate activations immediately before source `mlp.down_proj`; compute `delta = W2 * tanh(W1 * source_selected)` with rank 8, then inject into target. `102 → 8 → 202` is an illustrative channel-count example; read **actual** counts from the NeuroStrike selection. For 102 and 202 channels, two bias-free dense matrices have `(102 + 202)*8 = 2,432` trainable weights, so this is **not parameter-matched** to the old ~21k-parameter adapters. The original intention was ultimately to study connections across **all relevant consecutive layers**; the current 7→8 scope is a prototype, not the full intended experiment.

**Transformer structure:** `x → input_layernorm → self_attn → residual → post_attention_layernorm → SwiGLU MLP → down_proj → residual`. Layer-8 attention lies between source layer-7 MLP and target layer-8 MLP. The new skip pathway can bypass it without deleting or changing original attention parameters. Placement changes can affect attention activations even if attention weights remain frozen.

## Current implementation / files to inspect

These names were provided in the latest `NeuroStrike_cross_layer_placement_ablation_with_plots.zip` package; check what actually exists in this working directory before assuming it is installed:

- `cross_layer_adapter.py` — `CrossLayerAdapter`, NeuroStrike channel-map loader, hooks, five target placements.
- `train_cross_layer.py` — frozen-model training on 500 PKU-SafeRLHF + SciQ examples, 125 steps (gradient accumulation 4), 100 held-out examples; checkpoints/CSV/manifest.
- `test_cross_layer.py` — synthetic CPU hook-order, identity-at-init, gradient, frozen-weights tests.
- `viking_cross_layer.sbatch` — one GPU Slurm job; default `--source 7 --target 8 --rank 8`.
- `submit_placement_sweep.sh` — manually submits five placement jobs.
- `plot_placement_results.py` — CPU-only plotting script added **after** the five jobs were submitted.
- `README.md` — details; verify it matches the installed code.

Five placement options (`--placement` / `CROSS_PLACEMENT`):

| Name | Injection point | Important caveat |
|---|---|---|
| `pre_attention` | input to layer 8, before attention RMSNorm | changes what attention receives |
| `post_input_norm_pre_attention` | after attention RMSNorm, before attention | changes attention input |
| `post_attention_pre_norm` | after attention, before MLP RMSNorm | attention unchanged for this layer |
| `post_norm_pre_mlp` | after MLP RMSNorm, before MLP | attention unchanged |
| `selected_mlp` | selected layer-8 MLP channels before `down_proj` | direct selected-channel injection |

Only `selected_mlp` injects into selected **MLP intermediate channels**. Earlier modes map the 202-dimensional learned correction into the full residual stream using the **frozen selected columns of layer-8 `down_proj`**. Consequently, the five-way sweep changes both **placement and representation space**; it is not a clean isolated test of normalisation/attention ordering. The source hook stays at layer-7 `down_proj` in every mode.

## Slurm jobs already submitted

| Placement | Job ID |
|---|---|
| `pre_attention` | `38621775` |
| `post_input_norm_pre_attention` | `38621776` |
| `post_attention_pre_norm` | `38621777` |
| `post_norm_pre_mlp` | `38621778` |
| `selected_mlp` | `38621779` |

**Do not assume these jobs succeeded or that their outputs are available.** They were submitted using the original batch script, which did **not** produce PNGs. Their output directories are expected under `results/cross_layer_<placement>_<job_id>/`, with `training_history.csv`, `heldout_completion_ce.csv`, `cross_layer_adapter.pt`, `manifest.json`; logs likely `neurostrike-cross-layer-<job_id>.log/.err` in the submission directory. Newer batch script may plot automatically, but cannot retroactively change already-submitted jobs.

Check status safely:

```bash
pwd && git status --short 2>/dev/null || true
ls -la
sacct -j 38621775,38621776,38621777,38621778,38621779 -X --format=JobID,JobName%25,State,ExitCode,Elapsed
find results -maxdepth 3 \( -name 'training_history.csv' -o -name 'heldout_completion_ce.csv' -o -name 'manifest.json' \) -print
```

Plot completed results **without retraining**, if the script and CSVs exist:

```bash
./.venv/bin/python plot_placement_results.py --root results --out results/cross_layer_figures
```

Expected figures: `training_diagnostics.png`, `heldout_ce_change.png`, `heldout_ce_absolute.png`. Check output integrity and whether all five modes, seeds, and held-out prompt hashes match. Plot script currently scans `cross_layer_*` folders, and may require Matplotlib in `.venv`.

## First Codex task (execute in this order)

1. **Inspect the actual repository state:** list scripts and results; read `README.md`, adapter, training, batch, plot, and tests. Do not assume the latest ZIP was extracted. Preserve unrelated work and inspect `git status` before edits.
2. **Recover outcomes:** check the five job statuses/logs and output CSVs. Report exactly which completed/failed and why; do not invent numbers.
3. **Generate and validate figures** from existing results if possible; fix plotter issues if needed. Distinguish noisy per-example train CE from fixed-batch train CE and mini-validation CE. Show original vs adapted held-out CE overall and separately for PKU-SafeRLHF and SciQ.
4. **Audit scientific and code correctness:** real Hugging Face Llama hook signatures/order, source/target channel indexing, frozen backbone, attention effects, autograd, identity at initialization, reproducibility, parameter count, data leakage, checkpoint/reload, Slurm correctness, and whether placement comparisons are genuinely controlled. Pay particular attention to earlier placements' residual-space mapping and to the actual selected channel counts.
5. **Explain the scope mismatch:** prototype only 7→8 versus intended all relevant adjacent layer pairs; propose a minimal, testable generalisation to all selected adjacent layers, with per-pair trainable parameters and clear handling of overlaps/caches. Do not implement or launch a large sweep until the existing five runs are understood.
6. Provide a short **status report**, confirmed evidence, code changes (if any), and the **single next recommended experiment**. Separate facts from hypotheses and keep conclusions conservative.

## Execution and safety constraints

- **Do not submit Slurm jobs, start GPU training, download gated weights, install packages, or change cluster modules without explicit user approval.** CPU-only inspection/tests/plotting using existing environment are permitted.
- Do not touch global Python environments, user secrets, tokens, or unrelated files. Avoid destructive commands. Treat `.p` and `.pt` files as potentially unsafe pickle; only load trusted local artifacts.
- No network access or credential exposure unless necessary and approved. Keep outputs inside `results/` and code edits scoped to the experiment.
- The original scientific question concerns subnetwork reconfigurability. A lower held-out CE alone is **not** a safety improvement; lexical refusal at ~100% is saturated. Do not report judged safety ASR without an actual judge.
- CPU synthetic tests previously passed; **real Llama GPU integration has not been verified in this chat**. The cluster job outputs, not earlier assertions, must determine what ran successfully.

## Suggested opening reply to the user

“I'm checking the five submitted Slurm jobs and the actual `edge_learning` files first. I'll recover any CSV results, produce the missing plots without retraining, then audit the cross-layer hooks and identify what must change to extend from 7→8 to all relevant layer pairs. I won't submit new jobs without approval.”
