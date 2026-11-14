# Comparable architecture results: preliminary smoke comparison

All four tasks 38654607_0–3 completed successfully (exit 0). Their logs report
full-model zero-init identity and serialized reload checks passed. Combined with
38651212 (eight nodes/tanh/no bias), 38651213 (without edges), and 38654553
(direct/no bias), seven trained variants have matched seed-42 smoke data.

The combined plotter independently verifies training settings, selection, train/val
tokenized hashes, ordered prompts, per-sequence answer hashes/token counts,
original losses, and reconstruction of domain and overall CE from sequence losses.
All match. Runs span different code versions, so implementation provenance is
recorded in their manifests rather than claimed identical.

| Setting | Overall CE | Trained weights |
|---|---:|---:|
| Original | 2.440882 | 0 |
| Without edges | 2.439922 | 19,896 |
| Eight nodes, tanh, no bias | 2.440545 | 19,896 |
| Eight nodes, tanh, bias | 2.428240 | 21,279 |
| Eight nodes, no tanh, no bias | 2.437260 | 19,896 |
| Eight nodes, no tanh, bias | 2.429698 | 21,279 |
| Direct, no bias | 2.429869 | 105,698 |
| Direct, bias | 2.418387 | 106,961 |

All trained models retain the full original backbone. All use 32 training and 16
held-out examples, eight updates, seed 42, selected-MLP insertion across 15 pairs.
This is not the 500/100 three-seed comparison. Do not mix their absolute losses.

Tentative findings: bias-containing variants have lower CE than their corresponding
bias-free variants. Removing tanh improves the bias-free hidden version but slightly
worsens the biased hidden version: no consistent tanh advantage. Direct+bias is
lowest in this smoke comparison, but uses far more weights. The without-edges
control slightly beats the original eight-node configuration in smoke, whereas
that ordering reverses in the completed full runs: short-run rankings are unstable.

Adding biases changes parameter count and RNG consumption during initialization.
Equal seeds therefore do not guarantee identical shared initial weights across
bias settings. This is a comparison of these complete training configurations,
not an isolated causal demonstration of bias usefulness. A controlled bias study
should explicitly copy shared initial weights and vary only bias presence.

New plotter: plot_architecture_variants.py --train-n 32 --seeds 42.
Outputs: architecture_loss.png, summary.csv, paired_bootstrap.csv under
results/architecture_variant_figures/train32_seeds42/.
Paired bootstrap: 2,000 joint resamples of held-out examples, token-weighted CE
difference versus the eight-node/tanh/no-bias version. Exploratory 95% intervals
exclude training-seed uncertainty and do not account for multiple comparisons.
Only 16 held-out examples (13 PKU, 3 SciQ); domain conclusions are especially weak.

Next: approved matched full runs of all six architectures at multiple seeds,
with parameter budgets reported, plus separate tuning and final-test data before
hyperparameter selection. No additional GPU jobs submitted during this analysis.
