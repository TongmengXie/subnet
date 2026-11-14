# All-layer figures

Current specification: [CURRENT_EXPERIMENT.md](../../CURRENT_EXPERIMENT.md).
These figures use the six full all-layer runs at seeds 42, 43, 44; smoke and
single-pair runs are excluded.

- `heldout_three_settings.png`: actual held-out completion loss bars, zero-based
  axes; original / trained without edges / trained with edges. Bars are means
  across three seed-specific datasets; dots are seeds, not confidence intervals.
- `three_settings_summary.csv`: values underlying the current loss chart.
- `architectures_all_layers.png`: representative computations and trainable weights.
- `edge_placement_variants.png`: five insertion points tested in the pair study;
  only selected-MLP insertion is implemented/tested for all consecutive layers.
- `training_comparison.png`: fixed-train and mini-validation changes from step zero.
- `heldout_four_settings.png` / `heldout_with_neurostrike_only.png`: four-setting
  comparison including the completed neurons-only evaluation; `four_settings_summary.csv`
  contains all four measured settings.

With edges means **eight new shared hidden neurons with tanh and no biases per
consecutive pair**; it is not a direct full-connection result. Without edges means
separate trained adjustments to each selected neuron, with no added connections
between selected neurons. The original network is retained and frozen in both.
Both trained versions use 19,896 weights. The untrained adapter adds zero and is
omitted because it equals the original. Direct connections without hidden neurons
and the tanh/bias variants are planned comparisons, not measured bars.

Job 38653072 completed successfully for all three seeds. Tokenized hashes, prompts,
answers and token counts match; original per-sequence losses match exactly.
The plotter writes a separate `heldout_with_neurostrike_only.png` comparison.
Neurons-only zeros the nonselected MLP neurons but retains attention and residuals;
it is different from the old experiment removing selected neurons.

No per-layer ablation scores are available. No safety superiority follows from
completion loss. Mini-validation overlaps final evaluation. Regenerate on CPU:

```bash
OMP_NUM_THREADS=1 .venv/bin/python plot_all_layer_results.py
```
