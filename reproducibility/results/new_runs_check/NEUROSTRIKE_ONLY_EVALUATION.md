# NeuroStrike neurons-only evaluation

Current specification: [CURRENT_EXPERIMENT.md](../../CURRENT_EXPERIMENT.md).
Latest live Slurm check: COMPLETED, exit 0, 30 seconds. All three results validated.
Original per-sequence loss differences are exactly zero; prompts, answers, token
counts, tokenized hashes and aggregate losses match the reference files.

User authorized this evaluation on 2026-11-13. Slurm job 38653072 accepted; evaluation only, no training. One GPU, 30-minute limit; three held-out datasets (seeds 42, 43, 44), loaded from exactly the original sampler/tokenizer settings. Uses cached weights and datasets offline.

Definition: at all 16 MLP down-projection inputs, keep the union of NeuroStrike-selected gate/up neurons and zero every other intermediate neuron. Attention, layer normalization, embeddings, output head and residual paths remain intact. This is a selected-MLP-neurons-only intervention inside the full transformer, not a standalone network containing only selected nodes.

Offline tiny actual HF Llama test passes: selected values retained exactly, complement zeroed, all-neuron selection gives original output, model weights unchanged, hooks removed on normal and exceptional exits.

Before evaluating this intervention, the job requires both training and validation tokenized-data hashes to match the historical reference and per-sequence original CE differences no larger than 1e-5. A mismatch stops execution rather than adding an incomparable bar.

Outputs expected at results/neurostrike_only_38653072/seed{42,43,44}/: heldout_completion_ce.csv, heldout_per_sequence.csv, manifest.json. No trained checkpoint is produced because this intervention has no learned weights.

After all three seeds finish, run `.venv/bin/python plot_all_layer_results.py` to generate the separate heldout_with_neurostrike_only.png comparison. The plotter validates all three results before adding them; heldout_three_settings.png stays a three-setting chart. Missing results are never treated as measured zeros.

Initial sandbox submission timed out without acceptance; outside-sandbox Slurm queue/accounting checks showed no matching job before retry. Accepted retry ID recorded in results/batch_submissions/neurostrike_only_submission.json.

Measured overall CE: seed 42 = 13.5017884; seed 43 = 13.5894664; seed 44 = 13.5060238.
Mean = 13.5324262. Updated four-setting comparison is in
`results/all_layer_figures/heldout_with_neurostrike_only.png`; three-setting figure retained.
