# NeuroStrike experiment: current specification

Updated 2026-11-13 against the latest user decisions. This document describes
current intent and verified implementation; archived documents describe history.
User instructions take precedence. Job status below is a snapshot, not live state.

## Latest confirmed user decisions

- Keep the current eight-hidden-neuron version as an architecture to test. Do not
  discard its results or replace the current figure with unrun direct connections.
- Treat direct full connections **without new hidden neurons** as another
  architecture. They were the original intention, but are not implemented or tested yet.
- Next compare hidden layers with/without `tanh`, with/without biases, and direct
  full connections without a hidden layer. See [NEXT_STEPS.md](NEXT_STEPS.md).
- Use plain labels **with edges** and **without edges**. Here, edges means added
  connections between NeuroStrike-selected neurons. The original network remains
  connected in both settings. Use neuron/node in explanations, not channel.
- Finish the existing all-layer figure using actual loss bars, lower is better;
  omit the untrained zero-output adapter because it equals the original model.
- Evaluate NeuroStrike neurons alone as a distinct intervention; the user approved
  evaluation-only job 38653072. New training sweeps have not been approved.

## Implemented and tested architectures

Model: `meta-llama/Llama-3.2-1B-Instruct`; all original weights frozen.
Selections: trusted local `NeuroStrike/pre_computed_sn/weights_Llama-3.2-1B-Instruct.p`,
threshold 3, union of gate/up selected neuron indices.

| Figure label | Actual computation | Added trainable weights |
|---|---|---:|
| Original | Full pretrained model, no added adjustment | 0 |
| Without edges, trained | Each target neuron: `z += sum b*tanh(a*z)` using only its own activation | 19,896 |
| With edges, trained | Per consecutive pair: `target += W2*tanh(W1*source)` through 8 new shared hidden neurons, no biases | 19,896 |

The without-edges version has 7 or 8 added tanh units per target neuron, with
learned input/output weights. It is not a set of disconnected neurons, nor direct
training of the original neuron weights. The with-edges version has two dense
stages, not a separate direct edge for every selected source–target pair. Eight
is a tested design choice, not an established optimum. Do not claim that the
current fixed-width, bias-free version is a universal function approximator.

All-layer scope: 15 literal consecutive pairs, 0→1 through 14→15. Not every pair
of layers. Source-only layer 0 is not adjusted in the without-edges control;
both trained versions target layers 1–15. Incoming adjustments are applied before
outgoing source capture, preserving gradients through the whole computation.

Selected neuron counts by layer:
`50, 51, 63, 75, 92, 89, 99, 102, 97, 94, 80, 95, 86, 76, 75, 89`.
Thus the single-pair example is **102→8→97, 1,592 weights**, not 102→8→202.

## Insertion points: implemented scope matters

The source is always read immediately before its MLP down projection.

| Target insertion point | CLI placement | Tested scope |
|---|---|---|
| Before attention RMSNorm | `pre_attention` | 7→8 only |
| After attention RMSNorm, before attention | `post_input_norm_pre_attention` | 7→8 only; repaired rerun |
| After attention residual addition, before MLP RMSNorm | `post_attention_pre_norm` | 7→8 only |
| After MLP RMSNorm, before MLP projections | `post_norm_pre_mlp` | 7→8 only |
| Selected MLP neurons before down projection | `selected_mlp` | 7→8 and all 15 consecutive pairs |

Only the last option directly adjusts selected target neurons. The first four
map the learned output through frozen target down-projection columns into hidden
state space. They differ in representation and branch routing as well as location.
In `post_attention_pre_norm`, the addition enters only the MLP branch, not the
already-saved residual bypass. Only the first two change target-layer attention
activations in the single-pair experiment; downstream attention can change in all cases.

The all-layer runs use `selected_mlp` to directly address selected neurons. This
is not a finding that it is the best insertion point. Other all-layer placements
are not supported by the current all-layer implementation.

## Data, results and limits

Per seed: train on 400 PKU-SafeRLHF safe responses + 100 SciQ correct answers;
one epoch, 125 AdamW updates, gradient accumulation 4, learning rate 0.0002.
Held out: 80 PKU test examples + 20 SciQ validation examples. Seeds 42–44 change
data samples as well as initialization. Within each seed, trained methods match
ordered prompts, tokenized data, original losses and optimizer settings.

Metric: token-weighted, teacher-forced completion cross-entropy. It measures
reference-answer prediction, not generated-answer accuracy or judged safety.
PKU responses dominate the token-weighted overall result; report domains separately.
Mini-validation overlaps final evaluation; future tuning needs a separate split.

The with-edges version has lower held-out loss than without-edges at all three
seeds; both improve over original. This does not isolate connectivity alone:
functional form differs despite equal parameter counts. No judged safety result
has been established for these all-layer runs.

## Jobs and figures

- 38621775–38621779: historical single-pair placement sweep; 38621776 failed on
  keyword attention inputs. Four completed runs were recovered.
- 38651211–38651220: completed smoke tests, repaired pair run, and all-layer runs.
  Full matched pairs: 38651215/16 (42), 38651217/18 (43), 38651219/20 (44).
- 38651214: completed repaired pair, but its original-model loss differs from the
  old placement sweep despite matching prompt hashes. Cause unresolved; do not
  merge its absolute loss into the old placement ranking.
- 38653072: neurons-only evaluation. Last live check during this documentation
  update: **PENDING**, no results. Recheck Slurm before making a new status claim.

Neurons-only means zeroing all other MLP intermediate neurons across all 16 layers,
while keeping attention, normalization, embeddings, output head and residual paths.
It is not a standalone network made only of selected nodes. Older `pruned` results
instead removed the selected neurons: the opposite intervention.

Current outputs under `results/all_layer_figures/`:

- `heldout_three_settings.png`: actual loss bars for original / without edges /
  with edges, mean across seeds with individual seed dots; zero-based axes.
- `three_settings_summary.csv`: underlying current loss values.
- `architectures_all_layers.png`: representative neuron-level computations.
- `edge_placement_variants.png`: insertion-point inventory.
- `training_comparison.png`: fixed-train and mini-validation diagnostics.
- `heldout_four_settings.png`: legacy alias of the three-setting chart, not proof
  of a fourth evaluation. `four_settings_summary.csv` is a historical export;
  use `three_settings_summary.csv` for the current chart.

Generate plots on CPU with `.venv/bin/python plot_all_layer_results.py`. Once all
three neurons-only results pass comparison checks, the script additionally writes
`heldout_with_neurostrike_only.png`; the three-setting figure remains three settings.
No missing evaluation is plotted as a measured zero.

## Execution boundaries

No new GPU training jobs, package installs, model downloads or environment changes
without user approval. The approved neurons-only evaluation may be inspected and
its results plotted. Preserve existing results and use offline cached assets.
Historical launch commands are examples, not ongoing permission to submit jobs.
