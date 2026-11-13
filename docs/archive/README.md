> Historical document, archived 2026-11-13. Superseded by [CURRENT_EXPERIMENT.md](../../CURRENT_EXPERIMENT.md). Commands and plans below are not current authorization.

# NeuroStrike on University of York Viking (Slurm)

This package runs the supplied **v2 PKU-SafeRLHF + SciQ subnetwork notebook** on one A40 GPU.
It does not reproduce NeuroStrike's Llama Guard 8B judged ASR; the existing judge-free
StrongREJECT metrics remain unchanged.

## Before starting

- Obtain Viking login + project account code. Connect via University VPN if off-campus.
- Obtain Hugging Face access to `meta-llama/Llama-3.2-1B-Instruct`.
- Transfer this directory to your Viking **scratch** workspace (not the login node's
  small home directory). Example on your own computer:

  `scp -r neurostrike_viking YOUR_USERNAME@viking.york.ac.uk:~/scratch/`

  Adjust the destination to your actual scratch path; Viking documentation shows
  scratch storage under `/mnt/scratch/users/` and user-specific `scratch` links.

## Setup (once)

```bash
ssh YOUR_USERNAME@viking.york.ac.uk
cd ~/scratch/neurostrike_viking   # or the directory where you uploaded the package
bash setup_viking.sh
export HF_HOME="$PWD/hf_cache"
source .venv/bin/activate
hf auth login
```

The setup uses `Python/3.11.3-GCCcore-12.3.0`, documented by Viking. If the module
changes, use `module spider Python` and edit both shell scripts to use the same version.
The PyTorch wheel is CUDA 12.1; check the driver via `nvidia-smi` on a GPU node.

## Run

```bash
# Small run to check dataset/model downloads, adapters and figures:
sbatch --account=YOUR_PROJECT_CODE --export=ALL,SMOKE=1 viking_gpu.sbatch

# After smoke test succeeds, run the full pilot (500 training, 100 validation,
# 64 StrongREJECT harmful prompts, 128 generated tokens):
sbatch --account=YOUR_PROJECT_CODE viking_gpu.sbatch

squeue -u "$USER"
# inspect the job log, replacing JOBID with the printed Slurm ID:
tail -f neurostrike-subnetwork-JOBID.log
```

Default resources: `gpu` (1× NVIDIA A40 48GB), 6 CPU cores, 32GB RAM, 6h.
For H100 use `--partition=gpuplus` when submitting; no other changes are needed.

Results appear under `results/JOBID/`:
- `bottleneck_adapter.pt`, `independent_adapter.pt` (saved before evaluation)
- `training_history.csv`, `heldout_completion_ce.csv`, `adaptation_dataset_manifest.json`
- `strongreject_predictions.csv`, `refusal_rates.csv`, `logit_shift_*`
- `figures/*.png`, `figures/*.html` (interactive overlay)
- `executed.ipynb`, `pip_freeze.txt`, `neurostrike_commit.txt`

## Important modifications from the Colab notebook

1. Remove `google.colab` secrets and shell magics. HF credentials are provided by
   `hf auth login` with a persistent `HF_HOME` (never embed tokens in scripts).
2. Clone NeuroStrike **once** during setup and run with working directory
   `NeuroStrike/white_box` to preserve `../pre_computed_sn` paths.
3. Load the Llama model once. Skip the duplicate-load and demo-pruning cells;
   A4's pruned baseline remains intact.
4. Save trained adapter checkpoints **immediately** after A3, before A4's long
   generation loop, so a job timeout during evaluation doesn't lose training.
5. Use `SUBNETWORK_RESULTS_DIR` for all results and save Plotly as standalone HTML.
6. Override experiment sizes through environment variables rather than editing
   code (`SFT_TRAIN_N`, `SFT_VAL_N`, `EVAL_HARMFUL_N`, `EVAL_MAX_NEW_TOKENS`,
   `EVAL_LOGIT_N`, `SFT_EPOCHS`, `SFT_GRAD_ACCUM`, `SFT_VAL_EVERY`).

## Notes

- Viking login nodes must not be used for model execution or training.
- The base model is loaded in BF16 by upstream `util_model.load_model`, supported
  by A40/H100. This is **not** an 8-bit/4-bit quantized run.
- The notebook downloads three Hugging Face datasets. If compute nodes cannot
  access Hugging Face, pre-cache the model and datasets using a network-enabled
  allocated node or an approved transfer workflow. Do not assume compute-node
  internet access is available.
- Re-running the job trains again; it does not automatically resume from saved
  checkpoints. For long multi-epoch runs, add checkpoint/resume to A3.
- Python dependencies and CUDA compatibility have not been verified on a live
  Viking node. The notebook conversion and source compilation were tested locally.
