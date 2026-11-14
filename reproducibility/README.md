# Reproducibility records

Small saved manifests, metric/history CSVs, job plans and the exact submitted code
snapshots are copied from results/ at commit preparation. No checkpoints, secrets,
model weights, dataset contents, or model cache are included. Per-sequence files
contain prompt/answer hashes and losses, not raw text. Historical absolute paths
must be adjusted when using another checkout. Job states in records are snapshots.

Use source snapshots under results/*/code/ here when reproducing a submitted run;
the root implementation evolved after some submissions. environment.txt records
installed packages; upstream.json records the local NeuroStrike revision and the
trusted selection artifact hash. Obtain the gated model and datasets separately.
Pin prompt date to 13 Nov 2026 for comparisons to the November 13 reference runs.
The superseded unpinned sweep is explicitly marked in its saved plan.

Current results are mixed PKU/SciQ pilots. The agreed next phase is larger,
safety-only training; see ../NEXT_STEPS.md. No reproducibility command authorizes
GPU submissions. Existing checkpoints remain local and are not backed up by Git.
