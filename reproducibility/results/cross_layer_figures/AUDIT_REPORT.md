> Historical recovery audit. Its pair-only scope, integration gaps, and proposed next smoke run describe the earlier state; all-layer runs and repaired-pair runs subsequently completed. Numeric findings remain historical evidence. Current intent and status: [CURRENT_EXPERIMENT.md](../../CURRENT_EXPERIMENT.md).

# NeuroStrike cross-layer recovery and audit — 2026-11-13

## Evidence and recovered outcomes

Inspected local handoff, adapter/trainer, tests, batch/sweep scripts, both READMEs, installed Transformers Llama source, four manifests/CSV sets/checkpoints, and all five job logs/errors. No usable Git repository metadata is exposed here (`git status` fails); changes cannot be compared against Git. `plot_placement_results.py` is present at takeover, along with this earlier audit, the keyword-aware hook repair, the HF CPU test, and three PNG figures. Those code changes predate this takeover; no Git history is available to establish their provenance.

Live `sacct` for 38621775–38621779 failed connecting to infra01:6819. Status below is verified from saved Slurm utilisation summaries plus output artifacts, not a fresh accounting query.

| Job | Placement | Saved status | Overall adapted CE | Change | PKU adapted CE | SciQ adapted CE |
|---|---|---|---:|---:|---:|---:|
|38621775|pre_attention|COMPLETED, exit 0|2.474501|−0.003057|2.457989|4.230572|
|38621776|post_input_norm_pre_attention|FAILED, exit 1|unavailable|unavailable|unavailable|unavailable|
|38621777|post_attention_pre_norm|COMPLETED, exit 0|2.476353|−0.001204|2.459798|4.237017|
|38621778|post_norm_pre_mlp|COMPLETED, exit 0|2.477250|−0.000307|2.460627|4.245091|
|38621779|selected_mlp|COMPLETED, exit 0|2.475385|−0.002173|2.458745|4.245038|

Identical baseline: overall 2.477557614, PKU 2.460868200, SciQ 4.252451445. These are completion-only, teacher-forced, token-weighted CE. Four successful runs each have 125 sequential history rows, manifest and checkpoint. All checkpoint tensors are finite; encode is 8×102 and decode is 97×8, with nonzero trained decode. Failed run has no recoverable trained checkpoint/CSVs: a fifth result cannot be generated without rerunning it.

Created three PNGs and placement_summary.csv from existing CSVs only. Missing mode is explicitly labelled. Ordered train and validation prompt hashes and recorded model/rank/layers/channel counts/threshold/seed/steps/learning rate/accumulation/epochs match across the four completed runs; baselines match exactly. Each has 500 unique train and 100 unique validation prompt hashes, with no exact normalized prompt overlap.

## Implementation findings

**Confirmed historical bug; repair already present at takeover:** HF `LlamaDecoderLayer` calls `self_attn(hidden_states=hidden_states, ...)`. Original target pre-hook did not use `with_kwargs=True`; positional `args` was empty. Its combined error condition incorrectly reported “Target reached before source” even when source cache was available. This is exactly the failed job traceback. The current file has a keyword-aware wrapper preserving all other kwargs and positional behavior. Existing results are from the original code and were not overwritten.

Original toy attention was called positionally, so all toy tests passed despite this incompatibility. The existing `test_cross_layer_hf_cpu.py` exercises: randomly initialized tiny actual HF Llama, CPU only, no downloads. All five placements now pass exact identity at zero initialization, nonzero output after optimization, finite gradients, encode learning, frozen unchanged backbone, clean cache and hook removal. Original toy tests also pass, including activation placement and state-dict roundtrip. These checks do not verify the fixed mode on the full pretrained BF16 GPU model.

**Placement semantics:**
- Source is selected SwiGLU intermediates immediately before source down_proj. Channel indices are sorted, unique and bounds checked; positive-score absolute z-score >3 selection matches upstream criterion and unions gate/up indices.
- pre_attention adds to layer input before input RMSNorm. Both target attention input and the attention residual bypass change.
- post_input_norm_pre_attention adds to attention input after input RMSNorm. Attention changes, but its residual bypass is unchanged.
- post_attention_pre_norm hooks the input of post_attention_layernorm, after attention residual addition. The correction enters the normalized MLP branch only; the MLP residual bypass was already saved and does NOT receive it directly.
- post_norm_pre_mlp adds to normalized MLP input; only the MLP branch changes.
- selected_mlp adds to selected intermediate channels before down_proj; the correction reaches the layer output directly through selected down_proj columns.

The four earlier sites use frozen target down_proj columns to map a 97-vector to residual width. Equal parameter counts do not equalize activation scale, RMSNorm attenuation, branch routing or functional capacity. Only selected_mlp edits selected target MLP activations. This is a controlled comparison of five intervention definitions, not an isolated causal test of attention/normalization order. Attention parameters stay frozen in every mode; only the first two can change target-layer attention activations. Downstream layers can change in all modes.

**Autograd/freezing:** trainer explicitly freezes pretrained parameters, keeps model in eval mode, disables cache and gradient checkpointing, and optimizes only adapter matrices. Source cache is not detached, preserving gradients needed for a future multi-pair network. Hooks clone selected targets and clear cached source on consumption/finally. Zero decode ensures identity, but encode gradient is zero on the first update by construction. Synthetic CPU checks establish weight immutability; logs attest training configuration, but saved artifacts alone cannot independently prove original full-model weights were unchanged.

**Reproducibility/checkpoints:** checkpoint contains channel IDs, pair, rank, placement and state dict; sufficient to manually reconstruct an inference adapter against the same backbone. No dedicated reload CLI or full training-resume state (optimizer/RNG) exists. Manifests omit code hash, model/tokenizer/dataset revisions, package versions, tokenized-data/answer hashes, token limits and attention backend. Seeds are set, but deterministic GPU kernels are not enforced. Prompt equality is verified; exact tokenized/answer equality and historical code identity are not independently provable. Trainer saves history only after training, not incrementally, so failures lose history. Slurm uses strict shell error handling and the expected placement export; it neither plots nor resumes. No submissions made during this audit.

## Scientific limits

Raw train_ce averages four changing examples with equal example weights; held-out/fixed/mini-validation CE is token weighted. These are different estimands. Raw CE spans approximately 1.71–4.93; fixed-train CE stays near 2.52 and mini-validation near 2.73, with very small changes. There is no clear evidence of strong convergence. Step-1 diagnostics are already after an update; no step-0 fixed diagnostic is saved.

Dataset mix is nominally 80% PKU / 20% SciQ by example. Overall CE is dominated by PKU completion tokens (baseline means imply only about 0.93% SciQ token weight). Report domains separately; aggregate CE is not a balanced utility/safety score. Mini-validation is the first 24 shuffled validation examples, not a stratified independent split, and those examples also occur in final validation. Repeated monitoring can become selection leakage if used to tune the experiment.

Exact normalized train/validation prompt disjointness is confirmed; semantic duplicates, answer overlap and pretraining contamination remain unaudited. Data-selection code excludes exact StrongREJECT prompts, not semantic overlap. Only one seed, no per-example losses/token counts or confidence intervals, and no safety judge: tiny CE reductions do not establish placement superiority, improved safety, causal localization, or recovered pretrained connections. Comparison to older ~21k-parameter 16-layer adapters is not parameter/scope matched (this prototype has 1,592 weights). READMEs retain stale 202-channel examples and an incorrect “other three hooks” count; README.md primarily documents the earlier experiment.

## Extension to all relevant consecutive layers (design only)

Trusted local selection gives counts:

`0:50, 1:51, 2:63, 3:75, 4:92, 5:89, 6:99, 7:102, 8:97, 9:94, 10:80, 11:95, 12:86, 13:76, 14:75, 15:89`.

All layers are nonempty, hence the literal consecutive pairs are 0→1 through 14→15 (15 pairs). Per-pair rank-8 weights in that order:

`808, 912, 1104, 1336, 1448, 1504, 1608, 1592, 1528, 1392, 1400, 1448, 1296, 1208, 1312`.

Total **19,896** independent trainable weights, versus 1,592 for 7→8. This total is close to, but not equal to, the earlier 21,008 and still represents a different intervention architecture.

Minimal generalization: a ModuleDict of independent pair adapters; explicit pair/placement/channel/rank metadata; one orchestrated hook manager and optimizer/checkpoint covering all pairs. Define consecutive as `(i,i+1)` with both selections present, not adjacent entries in a sparse selection list. Start with selected_mlp to preserve the intended selected-channel interpretation.

Crucial overlap policy: at intermediate layer i, incoming (i−1)→i injection and outgoing i→(i+1) source capture share down_proj. Decide whether outgoing source is the corrected activation (cascading network, recommended default for the stated subnetwork ambition) or original activation (parallel skips). Implement a single hook that injects THEN captures for cascading semantics; do not let installation order silently choose this. Preserve the graph across pairs, avoid detach, maintain pair/forward-specific caches, clear on exceptions, and reject repeated/unconsumed sources. Freeze backbone, not the graph. Test at least three consecutive layers, all-pair zero-init identity, nonzero upstream gradient propagation, intended capture ordering, no cross-forward cache leakage, reload, and hook cleanup. Continue to disallow checkpointing/compile until explicitly supported.

Memory and GPU runtime of joint training are unmeasured: early adapters require backpropagation through much more of the frozen backbone. Do not infer cost from 7→8 or parameter count alone. Add model/dataset/code provenance, token counts, per-example held-out losses, independent evaluation split, randomized-channel/norm-matched controls, equal-budget comparisons, multiple seeds and a separate judged safety/utility evaluation before scientific claims.

## Single next recommended experiment

After approval: one full-model GPU **smoke run of the repaired post_input_norm_pre_attention mode** (32 train / 16 validation, existing --smoke), checking zero-init identity, gradient flow and checkpoint reload under actual BF16/installed HF behavior. This closes the demonstrated integration gap before spending on a replacement full run or implementing an all-pair sweep. It will not fill the missing comparable 500/100 result; that requires a subsequent approved full rerun with the original settings. No GPU jobs, downloads, package/module changes, or environment changes were performed here.

## Takeover verification

On 2026-11-13, reran both CPU test scripts successfully (all five modes, including actual randomly initialized HF Llama), regenerated all three PNGs from the existing CSVs, and reran manifest/hash/baseline consistency assertions. Independently loaded all four checkpoints with weights_only=True: finite tensors, shapes 8×102 and 97×8, and channel IDs matching the trusted selection. Selection counts and all 15 pair budgets above were independently recomputed. Live sacct failed; a bounded squeue query timed out. Existing saved job states remain the available scheduler evidence. No experiment source edits, GPU jobs, downloads, installs or module changes were made during this takeover. Corrected this report to acknowledge pre-existing repair/test/plot artifacts.
