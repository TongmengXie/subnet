# Documentation audit — 2026-11-13

Compared project-owned Markdown documents with the latest user decisions and
actual scripts/manifests. Upstream `NeuroStrike/` documentation and experiment
artifacts (CSVs, checkpoints, manifests and notebooks) were not rewritten.

## Latest user intent used for reconciliation

1. Keep the eight-hidden-neuron architecture and finish the existing three-setting
   loss figure; treat direct full connections without hidden neurons as another
   architecture, not an immediate replacement.
2. Document future comparisons with/without tanh, with/without biases, and direct
   full connections without a hidden layer. No new training sweep is approved.
3. Say neurons/nodes and with edges/without edges; explain that edges are added
   connections between selected neurons and disclose the eight added nodes.
4. Plot actual loss as bars and omit the zero-output untrained duplicate baseline.
5. Neurons-only evaluation is separately authorized; new GPU training/environment
   changes still require approval. Last checked job 38653072: PENDING.

The proposed six-variant matrix, a separate future tuning/test split, and the
suggestion to vary placement separately are experimental-design recommendations,
not completed experiments or additional user-approved launches.

## Findings and resolution

| Document | Outdated or misleading content found | Resolution |
|---|---|---|
| README.md | Earlier notebook setup described as current; obsolete no-GPU-verification claim | Current entry point; exact original retained under docs/archive/README.md |
| CODEX_HANDOFF.md | 7→8-only scope, missing figures/recovery tasks still pending, unverified GPU claim, 202-neuron examples | Current restart checklist; original archived |
| README_corsslayer78.md | 102→8→202 / 2,432-weight example, other-three count despite four earlier sites, pair-only/current-test claims | Historical pointer with actual 102→8→97 / 1,592 correction; original archived |
| NEXT_README.md | Old within-layer/checkpoint/judge patch presented as next experiment; not-executed claims | Historical pointer; original archived; no implied judge results or launch approval |
| NEXT_STEPS.md | Needed links to current intent; inherited plotting/submission wording and proposed-vs-approved ambiguity | Explicit current reference and proposed matrix; accurate current plotter and authorization boundary |
| results/cross_layer_figures/AUDIT_REPORT.md | Repaired-pair smoke and all-layer implementation still recommended as future work | Marked historical, preserving measurements and original audit conclusions |
| results/new_runs_check/CHECK.md | Old note that NEXT_STEPS was stale and live Slurm was unavailable | Marked dated historical snapshot; measurements preserved |
| results/new_runs_check/NEUROSTRIKE_ONLY_EVALUATION.md | Plot instructions could imply changing the three-setting figure; status not timestamped | Clarified separate comparison output and last-checked pending status |
| results/all_layer_figures/README.md | Legacy four-setting filename, current CSV absent, eight-node/no-node distinction unclear | Current file map and architecture definitions; template in plot_all_layer_results.py updated too |

## Authority and navigation

[CURRENT_EXPERIMENT.md](../CURRENT_EXPERIMENT.md) is the consolidated specification.
[NEXT_STEPS.md](../NEXT_STEPS.md) contains the future comparison plan.
[CODEX_HANDOFF.md](../CODEX_HANDOFF.md) is the resume checklist.
Archived text may contain superseded commands and numbers; its banner explicitly
marks it as history, not current requirements or permission.

No experiment training code, measured results, or GPU submissions were changed
by this documentation audit. Only the plotter's generated README template changed
alongside Markdown documents, so regenerating figures preserves the documentation.
