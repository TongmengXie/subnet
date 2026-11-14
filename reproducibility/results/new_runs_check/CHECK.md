> Historical batch-check snapshot. Statements about NEXT_STEPS.md being stale and unavailable live Slurm access describe that earlier check; documentation and access have since changed. Recorded run measurements remain evidence. Current intent and status: [CURRENT_EXPERIMENT.md](../../CURRENT_EXPERIMENT.md).

# New run check — 2026-11-13

All ten jobs 38651211–38651220 have saved COMPLETED (exit 0) summaries and complete result CSVs. Live squeue timed out; sacct could not connect. Error files contain offline dataset-cache notices, not tracebacks.

Three smoke jobs: 38651211–13. Full repaired attention placement: 38651214. Full all-adjacent connected/independent pairs: 38651215/16 (seed 42), 38651217/18 (43), 38651219/20 (44). Each full run uses 500 train examples, 100 validation examples and 125 optimizer updates. All-adjacent methods each have 19,896 parameters.

Connected minus independent overall completion CE: -0.00952428, -0.01148775, -0.01119994 at seeds 42, 43, 44. Both methods improve relative to their shared baseline. Each seed pair matches tokenized-data hashes, ordered prompt hashes, code hashes, budget, LR and optimizer steps. Per-sequence prompt/answer hashes, token counts and original losses also match. All ten per-sequence CSVs reproduce saved aggregate/domain CE to 1e-12.

Connected implementation covers all 15 adjacent pairs with inject-before-capture cascading semantics and preserved autograd. Independent control is a channel-separable nonlinear correction at the same target layers, with equal total weights; its functional capacity and initialization differ. This comparison does not isolate cross-layer connections alone. Seeds also change sampled datasets; comparisons are paired within seed, not repeated initialization on a fixed evaluation set.

Logs report full-model zero-init identity and serialized checkpoint reload identity in every new job. These checks use one probe sequence. No fresh GPU execution was performed for this check.

Important historical comparability issue: repaired pair job 38651214 matches the old sweep's ordered prompt hashes but baseline overall CE is 2.463116153 versus 2.477557614 previously (PKU 2.446492692 versus 2.460868200; SciQ 4.230996024 versus 4.252451445). Its adapted CE is 2.462016797, a within-run change of -0.001099356. Cause is unresolved; matching prompts does not establish matching tokenized inputs, answers, model revision or numerical execution. Do not append this absolute CE to the old placement ranking without resolving baseline drift. Old manifests lack tokenized-data/code provenance needed to establish historical equality.

NEXT_STEPS.md still says no jobs submitted and memory/runtime unmeasured; those statements predate this batch and are stale. The old placement plotter scans only cross_layer_* directories and will not include the new result naming automatically.

No evidence of judged safety improvement: results measure teacher-forced completion CE. Next analytical step: investigate baseline drift and compute paired uncertainty from the saved per-sequence files before making scientific claims. No new submissions or environment changes were made.
