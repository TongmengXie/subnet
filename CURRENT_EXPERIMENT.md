# NeuroStrike experiment: current specification

Updated 2026-11-13 against the latest user decisions. This document describes
current intent and verified implementation; archived documents describe history.
User instructions take precedence. Job status below is a snapshot, not live state.

## Latest confirmed user decisions

- Keep the current eight-hidden-neuron version as an architecture to test. Do not
  discard its results or replace the current figure with unrun direct connections.
- Treat direct full connections **without new hidden neurons** as another
  architecture. They were the original intention; the bias-free version is now
  implemented and CPU-tested, but has no pretrained-model training results yet.
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

## Direct full connections: implementation ready, no measured result yet

The next step implemented is `--scope all-adjacent --method direct`: a single
bias-free matrix per consecutive pair, `target += W × source`. No hidden neurons
or tanh. All 15 pairs use selected-MLP insertion with inject-before-capture ordering.
Total weights: 105,698 (7→8: 9,894), versus 19,896 for the existing trained controls.
No parameter-matching claim is made. Checkpoints/manifests record method `direct`,
null rank and architecture `selected-adjacent-direct-v1`. All six bias/tanh/direct variants are now implemented and CPU-tested;
see NEXT_STEPS.md for switches and budgets. Only the bias-free direct version has
a completed pretrained GPU smoke run; the other new variants remain CPU-verified only.

`test_direct_edges.py` passes exact-map, zero-init identity, gradients/cascading,
selected-neuron isolation, frozen-backbone, reload and cleanup checks on tiny actual
HF Llama, CPU only. Existing architecture tests pass. Pretrained GPU integration passed smoke job 38654553 (exit 0, 33 seconds),
including zero-init and checkpoint reload. Full-run performance remains unmeasured.
`plot_direct_results.py` will compare measured direct runs at seeds 42–44 with the
existing controls after matching data/settings/baselines, and explicitly show the
different budgets. Existing figures do not include an unmeasured direct result.

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
- 38653072: neurons-only evaluation. Live Slurm check: **COMPLETED**, exit 0, 30 seconds.
  All three seeds passed matching-data checks; original per-sequence loss differences are zero.

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
- `heldout_four_settings.png` / `heldout_with_neurostrike_only.png`: four measured
  settings including neurons only. `four_settings_summary.csv` contains their values.

Generate plots on CPU with `.venv/bin/python plot_all_layer_results.py`. The completed
three neurons-only results pass comparison checks, so the script additionally writes
`heldout_with_neurostrike_only.png`; the three-setting figure remains three settings.
No missing evaluation is plotted as a measured zero.

## Execution boundaries

No new GPU training jobs, package installs, model downloads or environment changes
without user approval. The approved neurons-only evaluation may be inspected and
its results plotted. Preserve existing results and use offline cached assets.
Historical launch commands are examples, not ongoing permission to submit jobs.

Neurons-only result (job 38653072): mean overall loss 13.5324 across seeds, versus
2.4549 original, 2.4350 trained without edges, 2.4243 trained with edges. This shows
selected MLP neurons alone do not preserve reference-answer prediction under this
masking intervention; it does not establish judged safety or a standalone safety circuit.

## Direct GPU smoke submission

User approved trying submission. Job **38654553** accepted: direct full connections,
all 15 consecutive pairs, selected-MLP insertion, no hidden neurons/tanh/bias;
seed 42, 32 training and 16 validation examples, 8 optimizer updates, 105,698 weights.
One GPU, 20-minute limit, cached/offline assets. Status must be checked live;
submission alone does not establish GPU correctness or performance. Inspect
zero-init identity, finite training/evaluation, serialized reload and output metadata
before proceeding to full comparisons. Full runs have not been submitted.

Latest implementation status: activation/bias settings are stored in checkpoints
and manifests. Existing performance figures retain their original architectures;
nondefault variants are excluded until an explicit multi-architecture plot is added.
No additional GPU jobs were submitted during this implementation.

## Remaining variant GPU smoke tests submitted

User authorized submission. Array **38654607**, tasks 0–3, covers:
0: eight hidden neurons + tanh + biases; 1: eight hidden neurons, no tanh/biases;
2: eight hidden neurons, no tanh, with biases; 3: direct full connections with biases.
All tasks use seed 42, 32 training / 16 validation examples, 8 updates, selected-MLP
insertion across all 15 consecutive pairs. Maximum two concurrent GPU tasks,
20-minute limit each, existing cached assets offline. Check live status and identity/
reload/finite-loss outputs before full comparisons; no full sweep submitted.
Submission record: `results/batch_submissions/variant_smokes_38654607.json`.

## Completed variant smoke comparison

All four tasks in 38654607 completed with exit 0. All six architectures now have
GPU smoke evidence; full comparisons are still missing. The matched smoke chart
is `results/architecture_variant_figures/train32_seeds42/architecture_loss.png`,
generated by `plot_architecture_variants.py --train-n 32 --seeds 42`. Data/settings/
baseline checks pass; the plotter rejects mismatches and missing variants.
Direct+bias has lowest smoke CE (2.418387), but uses 106,961 weights. Biases help
in these short runs; tanh results are mixed. One seed, eight updates, 16 held-out
examples do not establish a final ranking. Paired bootstrap CSV estimates only
held-out-example uncertainty, not training uncertainty. See
`results/architecture_variant_figures/SMOKE_FINDINGS.md` for caveats, including
initialization differences and the smoke/full ranking reversal in existing controls.
No additional GPU submissions were made during this analysis.

## Authorized width sweep — 2026-11-14

Submitted array **38655122**, tasks 0–32 (at most two concurrent GPUs).
Widths 4/8/16/32/64 × tanh/identity × seeds 42/43/44, biases enabled, plus
three matched direct+bias references. Full 500 training / 100 validation examples,
125 updates, unchanged optimizer and selected-MLP placement. These held-out
examples are used as architecture-selection validation, not untouched final tests.
Bias-enabled hidden budgets: 11,271 / 21,279 / 41,295 / 81,327 / 161,391;
direct+bias: 106,961. No bias-free width sweep or new insertion-point sweep is included.
Training code snapshot, hashes and exact task map:
`results/width_sweep_20261114/plan.json` and `code/`.
After all runs finish, `.venv/bin/python plot_width_sweep.py` checks matching data,
settings, code hashes, selections, per-sequence baselines and aggregate losses,
then writes loss_vs_width.png, loss_vs_parameters.png and summary.csv.
Submission is not evidence of completed runs; check Slurm and output manifests.

## Research interpretation and baselines — 2026-11-14

Verified current code freezes ALL original model weights and optimizes only added
adapter weights (`requires_grad_(False)` for model parameters; optimizer receives
`adapter.parameters()`). Full-model outputs are evaluated. This is localized
adaptation through selected neurons, not updating those neurons' original weights.
Do not describe it as a trained standalone subnetwork or as directly fine-tuning
only original safety-neuron weights. Changes propagate through the frozen backbone.

User agreed to a random-neuron baseline using the chosen safety architecture.
Implemented `--neuron-selection random --selection-seed N`: uniform sampling from
all MLP neuron indices, without replacement within each layer, matching NeuroStrike
counts at every layer. Overlap with selected neurons is allowed and recorded; this
is not a disjoint-neuron control. A separate local RNG preserves training RNG and
adapter initialization at matched seeds. Actual neuron IDs, mask seed and overlap
are recorded; output naming and existing plotters distinguish random selection.
`test_random_neurons.py` passes count/budget/init matching, repeatability, training,
unchanged original model weights and serialized reload for all three methods.
No random baseline jobs submitted yet: choose architecture after width validation,
then use multiple mask seeds without tuning architecture on random-mask results.
Suggested mask seeds 1001–1003 are a proposal, not submitted jobs. Use matched
training seeds/data/order and budgets; additional runs can separate mask and
training-seed variance if needed.

User also proposed full-model fine-tuning without architectural changes. Meta-device
configuration inspection counts 1,235,814,400 original parameters. A conventional
16-byte/parameter mixed-precision Adam planning estimate is 18.42 GiB for parameter,
gradient and optimizer state alone; activations/buffers/allocator overhead add memory.
This is an estimate, not measured GPU memory or a tested training recipe. A GPU
memory smoke test should precede any full baseline; tune learning rate appropriately
with a separate tuning budget rather than assume adapter LR is fair. No full-model
training code or jobs have been added. Keep architecture unchanged for this control.

Exact data statistics are in `results/data_audit/REPORT.md` and data_stats.csv/json.
Prompt-date bug found: Llama embeds today's date. Historical reference hashes match
when explicitly using 13 Nov 2026. Current trainer/evaluation now record or pin this
date. All 33 original width tasks were still pending, so array 38655122 was cancelled
without training and replaced by **38655694** at the same scope/budget. Replacement
snapshot: `results/width_sweep_20261114_fixeddate/`; the width plotter defaults there.
The old snapshot is preserved and marked superseded. No sweep winner is available
while replacement tasks remain queued. New submissions beyond this replacement
were not performed during this work.

## Baseline submissions — 2026-11-14

User authorized working on and submitting random-neuron and unchanged full-model
fine-tuning baselines. Source snapshots and task definitions are under
`results/baseline_comparison_20261114/`; accepted job IDs are in submissions.json.

- 38655875: full-model GPU smoke, 32 train / 16 validation, 8 updates; records peak
  allocated/reserved VRAM and device capacity. One GPU, 48G host RAM, 30-minute cap.
- 38655876: full-model full runs, array 0–2 (seeds 42/43/44), maximum one concurrent;
  dependent on successful 38655875. Unchanged Llama architecture, all original
  parameters trainable, FP32 parameters/AdamW states and BF16 autocast. Evaluation
  is BF16, matching the baseline precision. Saves full BF16 model checkpoints.
  LR 2e-5 is a declared provisional setting, not an optimized full-model result.
- 38655877: random-neuron GPU smoke; depends on successful width sweep 38655694.
  Chooses the minimum mean overall validation CE across three seeds, with smaller
  budget breaking exact ties. This fixes architecture before testing random masks.
- 38655878: random full runs, array 0–8, three training seeds × three mask seeds
  (1001/1002/1003); maximum two concurrent, after successful 38655877.

Every full run uses 500 training / 100 validation examples, 125 updates, fixed
prompt date 13 Nov 2026. Data hashes and original per-sequence losses must match
historical references. Random subsets are uniform over all 8,192 MLP neurons per
layer and match NeuroStrike counts; overlap is allowed and recorded. The original
model stays frozen in random runs. The CPU tests establish matching counts/budgets/
initialization and unchanged backbone, plus full-model embedding/attention/head
updates with unchanged architecture and checkpoint reload. Full-model peak memory
remains unmeasured until its smoke finishes. No baseline has measured success just
because submission was accepted. These evaluations are validation, not final tests.

The random comparison is trained adapter-on-selected-neurons versus adapter-on-
random-neurons; it does not compare original-weight-only neuron fine-tuning.
Full-model fine-tuning changes optimizer LR, parameter count and optimization
geometry, so compare held-out benefit and cost without claiming equal capacity.
