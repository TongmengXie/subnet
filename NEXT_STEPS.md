# NeuroStrike experiment: next steps

Current specification: [CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md).
This is the future comparison plan, not authorization to launch it.

## Confirmed next phase: larger, safety-only training

The user confirmed on 2026-11-14 that the eventual main experiment must:

1. **Scale to more training data points.** Increase beyond the current 500-example
   pilot and use larger, separate validation and final-test sets. Exact sizes are
   not selected yet; keep data and training budgets matched across architectures,
   NeuroStrike-selected neurons, random-neuron controls and full-model baselines.
2. **Eliminate SciQ from training.** Train on safety-alignment examples, initially
   safe-response examples from PKU-SafeRLHF. SciQ may be used only as an optional
   held-out utility evaluation, not as part of the training objective.

The current runs jointly train on 400 PKU + 100 SciQ examples per seed. Retain
them as a **mixed-task pilot**, not evidence from safety-only training. The clean
safety-only comparison must restart from the original pretrained model rather
than continue from mixed-training checkpoints. Separate per-domain evaluation
does not undo mixed training.

This records the intended next phase; it does not cancel or alter current jobs
or authorize additional GPU submissions. Select data sizes and independent
splits before launching the larger safety-only comparisons.

## Planned architecture comparison

Compare connections between NeuroStrike-selected neurons using:

- Hidden layers **with and without `tanh`**.
- **With and without biases** (both hidden-layer transformations).
- **Direct full connections without a hidden layer**, with and without an output bias.

Proposed test matrix: at the current width of eight, this gives four hidden-layer variants and two
direct-connection variants. Preserve the existing eight-node, `tanh`, bias-free
version as one architecture, not as the original direct-connection design.
The direct version adds `W × source` (or `W × source + b`) to the target;
it introduces no hidden neurons or added activation function.

Use matched data, selected neurons, consecutive layer pairs, insertion point,
training settings and seeds. Start at selected MLP neurons before down projection
(the insertion point used by the completed all-layer runs); vary placement
separately. Report parameter counts and held-out loss overall and by dataset.
Keep a separate tuning split and untouched final test set for future comparisons.
Plot actual loss as bars, with individual seed results shown. Do not interpret
completion-loss improvements as demonstrated safety improvements.

All six variants are implemented and CPU-tested. The bias-free direct version
completed pretrained GPU smoke job 38654553; other new variants have not been
GPU-tested. No full direct or variant comparison has been run. New GPU jobs
require approval.

## Implemented next step: direct full connections

`--scope all-adjacent --method direct` adds `W × source` to selected target neurons
for every consecutive layer pair. One dense matrix per pair, no hidden neurons,
no tanh, no biases. All original model weights remain frozen. Incoming adjustments
are applied before outgoing capture. Placement remains `selected_mlp` for a controlled
comparison with existing all-layer runs; other all-layer placements are not enabled.

Actual budget: **105,698 weights** across 15 pairs; **9,894** for pair 7→8.
This is not parameter-matched to the 19,896-weight controls. `--rank` does not apply;
checkpoint/manifest rank is null. Existing defaults and checkpoints remain supported.

CPU verification (no downloads):

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python test_direct_edges.py
```

After GPU approval, use the existing batch script with `CROSS_SCOPE=all-adjacent`,
`CROSS_METHOD=direct`, `CROSS_PLACEMENT=selected_mlp` and `SMOKE=1` first.
Full runs should use seeds 42, 43, 44 with the existing sizes/settings, and record
matched tokenized hashes and original losses before comparison. Existing runs use
mini-validation overlapping final evaluation: do not tune on this comparison and
claim an independent final test. New tuning requires separate data.

Once measured full direct runs exist:

```bash
OMP_NUM_THREADS=1 .venv/bin/python plot_direct_results.py
```

This writes actual-loss bars in `results/direct_edge_figures/direct_vs_hidden_loss.png`
and a summary CSV, checking matched data/settings and baseline losses. It refuses
to fabricate a plot when runs are missing. Existing figures and results are preserved.

## Existing runs and implementation

Jobs 38651211–38651220 completed: three smoke tests, the repaired attention-input
placement, and matched all-layer with-edges/without-edges runs at seeds 42–44.
The with-edges version uses eight shared hidden neurons and `tanh`, without biases.
The without-edges version learns separate `sum b*tanh(a*z)` adjustments for each
selected neuron; it adds no connections between selected neurons.
Both versions have 19,896 trained weights and keep the original model frozen.

Evaluation-only job 38653072 completed successfully for all three seeds.
Data and baseline checks passed; mean overall neurons-only loss is 13.5324.
The four-setting figure includes this result; the three-setting figure stays separate.

Current figures: `results/all_layer_figures/heldout_three_settings.png`, generated
by `plot_all_layer_results.py`; the neurons-only result gets a separate comparison
figure when all three seeds are available. Existing results remain preserved.

The following describes the current implementation, not the planned variants.

`train_cross_layer.py` retains its single-pair default. New options:

- `--scope all-adjacent --method connected`: independent rank-r matrices for
  every literal (i,i+1) pair with nonempty selections. Selected-MLP placement only.
  Shared hooks inject incoming corrections before capturing outgoing sources,
  preserving the graph. At rank 8 the current selection gives 19,896 parameters.
- `--scope all-adjacent --method direct`: one full source-to-target matrix per pair,
  selected-MLP placement only; implemented/CPU-tested, pretrained GPU run pending approval.
- `--scope all-adjacent --method independent`: exactly the same total parameter
  budget, distributed over the same target channels/layers as channel-separable
  `sum b*tanh(a*z_channel)` corrections. No cross-channel inputs or source-layer
  skip inputs. Zero b gives identity initialization. Unit allocation is
  deterministic; capacity and initialization geometry still differ from connected
  adapters. This is a trained control, not proof that any difference is solely edges.

Both methods use the same seed-controlled examples/order, optimizer settings and
completion loss. Source-only layer 0 is not adapted in the independent control,
matching the connected network's target coverage. Model remains frozen/eval;
checkpointing and compile remain unsupported. Full-model GPU runs completed;
peak GPU memory has not been established here.

New outputs: step-zero fixed diagnostics; incremental training history;
`heldout_per_sequence.csv` with paired CE, completion-token counts and prompt/answer
hashes; tokenized-data and code hashes in the manifest; early run configuration.
Final adapted evaluation reconstructs a fresh adapter from the serialized
checkpoint. Outputs must be empty to prevent accidental overwriting. Checkpoints
support inference reconstruction, not optimizer/RNG training resume. Mini-validation
still overlaps final validation; do not tune on it and then claim independent test
performance. Semantic overlap, uncertainty and judged safety remain future work.

The batch script accepts CROSS_SCOPE, CROSS_METHOD, CROSS_SEED, CROSS_RANK,
CROSS_PLACEMENT and SMOKE. It runs CPU tests first and enforces cached/offline HF
access. Its output naming now includes scope/method/seed. `plot_placement_results.py` is for
historical placement directories only; `plot_all_layer_results.py` handles the
current all-layer comparison. Existing submission scripts are historical tools,
not authorization to submit new sweeps.

For the planned variants, run CPU identity/gradient/reload tests and approved GPU
smoke tests before full runs. Retain the frozen baseline for each evaluation set;
require matching tokenized hashes and record the different parameter budgets.
Separate judged safety and utility evaluation is still needed.

CPU validation (no downloads):

```
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python test_adjacent_adapters.py
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python test_cross_layer_hf_cpu.py
OMP_NUM_THREADS=1 .venv/bin/python test_cross_layer.py
```

## Direct GPU smoke submission

User approved trying submission. Job **38654553** accepted: direct full connections,
all 15 consecutive pairs, selected-MLP insertion, no hidden neurons/tanh/bias;
seed 42, 32 training and 16 validation examples, 8 optimizer updates, 105,698 weights.
One GPU, 20-minute limit, cached/offline assets. Status must be checked live;
submission alone does not establish GPU correctness or performance. Inspect
zero-init identity, finite training/evaluation, serialized reload and output metadata
before proceeding to full comparisons. Full runs have not been submitted.

## Implemented: tanh and bias switches

Use `--activation tanh` or `--activation identity` (no added activation), and
`--bias` to enable biases in both hidden transforms or the direct output.
Direct connections require identity; the without-edges control retains its
original tanh, bias-free definition. Existing defaults and checkpoint shapes
remain unchanged. Manifests/checkpoints record activation and bias explicitly.

| Architecture | Activation | Bias | Weights across all pairs |
|---|---|---|---:|
| Eight hidden nodes | tanh | no | 19,896 |
| Eight hidden nodes | tanh | yes | 21,279 |
| Eight hidden nodes | identity | no | 19,896 |
| Eight hidden nodes | identity | yes | 21,279 |
| Direct full connections | identity | no | 105,698 |
| Direct full connections | identity | yes | 106,961 |

`test_architecture_variants.py` passes identity at initialization, exact equations,
training gradients, frozen backbone and serialized reload for all six variants
on tiny actual HF Llama, offline CPU. Existing tests also pass. Batch options
`CROSS_ACTIVATION=tanh|identity` and `CROSS_BIAS=0|1` select variants and distinguish
output folder names. Current loss plotters exclude nondefault variants, preventing
silent overwriting of existing architecture results; a combined six-variant
performance plot remains future work after approved matched runs exist.

Smoke result verified: **38654553 COMPLETED**, exit 0, 33 seconds. Full-model
zero-init identity and serialized reload passed; overall completion loss
2.44088194 → 2.42986922 on 16 held-out examples after 8 updates. This is integration
evidence, not the full 500/100 three-seed comparison. No further jobs submitted.

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
