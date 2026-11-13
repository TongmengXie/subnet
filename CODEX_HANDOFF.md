# Codex handoff: read current decisions first

This replaces the initial 7→8-only recovery handoff. Read
[CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md) and [NEXT_STEPS.md](NEXT_STEPS.md).
The original handoff is retained as [history](docs/archive/CODEX_HANDOFF.md).

## Resume checklist

1. Inspect actual files and preserve unrelated work. Git metadata has not been
   usable here; do not assume a clean Git diff is available.
2. Check authorized evaluation job **38653072**, its logs, and
   `results/neurostrike_only_38653072/seed{42,43,44}/`. Last documentation check
   found it pending, with no results; this is not a live status guarantee.
3. If results exist, verify all three data/baseline checks, then run
   `.venv/bin/python plot_all_layer_results.py`. Retain the three-setting figure
   and put the neurons-only comparison in its separate figure.
4. Keep the existing eight-node + tanh + no-bias results. The user's latest
   decision is to treat the no-hidden-node version as another architecture,
   not to discard or silently replace existing results.
5. Next architectures: with/without tanh, with/without biases, and direct full
   connections without hidden neurons. These are documented plans, not completed
   implementation or approval for a new GPU sweep.

Use neuron/node in explanations; figure labels are **with edges** and **without
edges**, referring to added connections between selected neurons. Always disclose
that the current with-edges version includes eight added shared nodes per pair.
Only selected-MLP insertion has been tested across all 15 consecutive pairs.
Completion loss is not demonstrated safety performance.

CPU inspection/tests/plotting are permitted. New GPU training, downloads,
installs and environment changes require user approval. Do not submit legacy
sweep scripts simply because a historical document lists them.
