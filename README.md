# NeuroStrike selected-neuron experiments

Start with [CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md) for the current user
intent, implemented architectures, verified results, and execution boundaries.
See [NEXT_STEPS.md](NEXT_STEPS.md) for the planned architecture comparison.

The completed all-layer comparison uses eight added hidden neurons with `tanh`
and no biases for the with-edges version. Direct full connections without hidden
neurons are a separate planned architecture, not the architecture in those results.

Current loss figure: [heldout_three_settings.png](results/all_layer_figures/heldout_three_settings.png).
Regenerate from existing results only:

```bash
OMP_NUM_THREADS=1 .venv/bin/python plot_all_layer_results.py
```

[CODEX_HANDOFF.md](CODEX_HANDOFF.md) is the restart checklist.
[Documentation audit](docs/DOCUMENTATION_AUDIT.md) identifies superseded material.
Older notebook setup/launch instructions are preserved in
[docs/archive/README.md](docs/archive/README.md); they are not the current plan.
The upstream `NeuroStrike/` documentation remains unchanged.
