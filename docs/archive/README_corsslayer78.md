> Historical document, archived 2026-11-13. Superseded by [CURRENT_EXPERIMENT.md](../../CURRENT_EXPERIMENT.md). Commands and plans below are not current authorization.

# Cross-layer NeuroStrike adapter: attention/normalisation placement ablation (v2)

## Finding from review of the original implementation

The original `cross_layer_adapter.py` hard-coded the **source and destination** at
`model.layers.{i}.mlp.down_proj` **forward pre-hooks**, so its only placement was
**after target attention, MLP RMSNorm, and SwiGLU**. Its direct `102 → 8 → 202`
connection added to layer-8 selected **MLP intermediate activations**. It did
not vary the bottleneck's position relative to attention or normalisation.

## What this patch changes

The source remains the layer-7 selected MLP activations just before
`layers.7.mlp.down_proj`. All modes use the **same two trainable matrices**:

    selected_layer7 -> W1 -> tanh -> W2 -> 202-dimensional correction
       102 inputs         8 bottleneck         202 outputs

Rank=8, n7=102, n8=202 means `(102+202)*8=2432` trainable weights in every mode.
Actual channel counts are read from the NeuroStrike local selection artifact.

The **target hook** is selectable using `--placement` / `CROSS_PLACEMENT`:

| mode | target injection site in layer 8 | target-layer attention sees new signal? | target MLP RMSNorm sees new signal? |
| --- | --- | --- | --- |
| `pre_attention` | input to layer 8 (`model.layers.8`): **before attention RMSNorm** | **Yes** | Yes (indirectly) |
| `post_input_norm_pre_attention` | input to `model.layers.8.self_attn`: **after attention RMSNorm, before attention** | **Yes** | Yes (indirectly) |
| `post_attention_pre_norm` | input to `model.layers.8.post_attention_layernorm` | No | **Yes** |
| `post_norm_pre_mlp` | input to `model.layers.8.mlp` | No | No |
| `selected_mlp` | input to `model.layers.8.mlp.down_proj` | No | No |

These are five insertion points covering both RMSNorm operations and attention, not
all arbitrary permutations of the operators. The **source** is always captured after layer-7 normalisation, attention and
SwiGLU; only **destination placement** varies. The original transformer
execution order stays unchanged; this patch does not move, bypass, replace,
or learn new attention weights. `pre_attention` may **change the attention
activations**, but not the pretrained attention parameters.

**Important difference between placements:** only `selected_mlp` adds the learned
202-dimensional vector directly to the selected layer-8 **MLP intermediate
activations**. The other three hooks operate in the 2,048-dimensional Llama-3.2-1B
**residual stream**, before the selected MLP activations exist. To retain the
same trainable dimensions and connect to the chosen selected-neuron output
subspace, those modes insert

    residual_delta = frozen_down_proj_8[:, selected_8] @ (W2 @ tanh(W1 @ z7_selected))

using the original (frozen) `down_proj` columns. This produces a **residual
injection in a fixed subspace**, not a direct modification of layer-8 selected
MLP neuron activations. Thus comparisons vary *both* injection position and
representation space: interpret carefully, not as a perfect causal ablation of
attention/norm order.

**Why not test every possible ordering?** Llama's block order is fixed:

    layer 8 input -> attention RMSNorm -> attention -> residual
                  -> MLP RMSNorm -> SwiGLU -> down_proj -> residual

In particular, moving the MLP RMSNorm before attention, or running attention
within the new 8-neuron bottleneck, would change the transformer architecture
and introduce a different experiment. These are not implemented here.

## Files

- `cross_layer_adapter.py`: five target hook modes; selected output-basis mapping.
- `train_cross_layer.py`: `--placement` CLI, frozen Llama training, fixed-train
  and held-out CE, saves placement in checkpoint/manifest.
- `test_cross_layer.py`: CPU toy transformer, checks exact hook ordering,
  identity at initialization, no attention/MLP weight modifications, gradients,
  target-only activation changes for `selected_mlp`, and checkpoint reload.
- `viking_cross_layer.sbatch`: accepts `CROSS_PLACEMENT`, `CROSS_RANK`, `SMOKE`.
- `submit_placement_sweep.sh`: **explicit manual submission** of five Slurm jobs.

## Running on Viking

Reuse the existing `~/edge_learning/.venv`, an accessible gated
`meta-llama/Llama-3.2-1B-Instruct` checkpoint, and a trusted local NeuroStrike
selection at `NeuroStrike/pre_computed_sn/weights_Llama-3.2-1B-Instruct.p`.
Do not execute `.p` files from untrusted sources: pickle is executable code.

Run tests locally or on the cluster login node:

```bash
cd ~/edge_learning
./.venv/bin/python test_cross_layer.py
```

Single placement smoke (do not run GPU work on a login node):

```bash
sbatch --account=YOUR_PROJECT_CODE \
  --export=ALL,CROSS_PLACEMENT=post_attention_pre_norm,SMOKE=1 \
  viking_cross_layer.sbatch
```

Full jobs, one per placement (five jobs):

```bash
bash submit_placement_sweep.sh YOUR_PROJECT_CODE
```

The submission script does **not** auto-run; inspect it first. Output directories
include placement and Slurm job ID, e.g. `results/cross_layer_pre_attention_1234/`.

For a single manual GPU run:

```bash
./.venv/bin/python train_cross_layer.py --repo NeuroStrike \
  --source 7 --target 8 --rank 8 --placement selected_mlp \
  --out results/cross_layer_manual
```

## How to compare

Use identical train/validation prompt hashes, seed, rank, selected channels,
optimizer steps, LR, and validation set. Inspect `fixed_train_ce`, `mini_val_ce`,
`grad_norm`, PKU-SafeRLHF/SciQ held-out CE separately; compare to a frozen
baseline. Raw mini-batch CE is inherently noisy. Add independent seeds and
safety evaluations before claiming a benefit. Different target sites alter the
forward computation differently even at identical parameter count. The adapter
still tests only **one layer pair (7 → 8)**, not the earlier 16-layer baseline.

## Verification status

CPU synthetic integration tests pass for all five hook modes (including a
Llama-like attention + two-norm block). `python -m py_compile` passes.
**Not tested on the real Hugging Face Llama model or a Viking GPU** in this
environment. Run the supplied tests and Slurm smoke job before any full sweep.
No judged safety ASR is provided; lower CE does not prove improved safety.
