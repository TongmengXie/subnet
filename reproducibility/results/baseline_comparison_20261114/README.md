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
