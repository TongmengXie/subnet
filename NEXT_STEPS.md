# NeuroStrike experiment: next steps

Current specification: [CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md).
This is the future comparison plan, not authorization to launch it.

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

These variants are planned, not yet implemented or run. New GPU training jobs
require approval.

## Existing runs and implementation

Jobs 38651211–38651220 completed: three smoke tests, the repaired attention-input
placement, and matched all-layer with-edges/without-edges runs at seeds 42–44.
The with-edges version uses eight shared hidden neurons and `tanh`, without biases.
The without-edges version learns separate `sum b*tanh(a*z)` adjustments for each
selected neuron; it adds no connections between selected neurons.
Both versions have 19,896 trained weights and keep the original model frozen.

Evaluation-only job 38653072 was submitted for the selected-MLP-neurons-only
comparison. Its results must pass data and baseline checks before inclusion.
Check live status rather than assuming completion.

Current figures: `results/all_layer_figures/heldout_three_settings.png`, generated
by `plot_all_layer_results.py`; the neurons-only result gets a separate comparison
figure when all three seeds are available. Existing results remain preserved.

The following describes the current implementation, not the planned variants.

`train_cross_layer.py` retains its single-pair default. New options:

- `--scope all-adjacent --method connected`: independent rank-r matrices for
  every literal (i,i+1) pair with nonempty selections. Selected-MLP placement only.
  Shared hooks inject incoming corrections before capturing outgoing sources,
  preserving the graph. At rank 8 the current selection gives 19,896 parameters.
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
