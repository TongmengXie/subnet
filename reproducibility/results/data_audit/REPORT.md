# Train and validation data audit — 2026-11-14

All six historical train/validation tokenized fingerprints (seeds 42–44) were reproduced exactly with Llama prompt date fixed to **13 Nov 2026**. No pretrained model weights were loaded for this CPU audit.

| Data | Source split | Used per seed | Reference |
|---|---|---:|---|
| PKU-SafeRLHF training | train (73,907 rows) | 400 | Chosen response labelled safe; preferred response if safe, otherwise another safe response |
| SciQ training | train (11,679) | 100 | Correct answer text; support and distractors unused |
| PKU-SafeRLHF validation | test (8,211) | 80 | Same safe-response rule |
| SciQ validation | validation (1,000) | 20 | Correct answer text |

SciQ official test split (1,000 examples) is not used by the current all-layer training/evaluation pipeline. The selected 100-example evaluation is validation for architecture selection, not an untouched final test.

| Seed | Train completion tokens | Validation completion tokens | PKU share of validation tokens |
|---|---:|---:|---:|
| 42 | 36,038 | 7,085 | 99.07% |
| 43 | 34,604 | 7,789 | 98.99% |
| 44 | 35,673 | 6,918 | 98.93% |

Limits: up to 256 total tokens; reference answer truncated to 128 tokens, with EOS appended for complete answers. CE excludes prompt tokens. PKU answers are much longer, so overall token-weighted CE is approximately 99% PKU. Report SciQ separately. Raw training loss averages per-example CE, while evaluation is token weighted.

Training answer truncation counts: 101 / 77 / 91 out of 500 at seeds 42 / 43 / 44. Validation: 20 / 24 / 17 out of 100. Prompt truncation: train 3 / 1 / 1, validation 0 / 2 / 0. All truncated examples are PKU.

Within-seed exact normalized prompt overlap: zero. Across seeds: 1,469 unique training prompts, 295 unique validation prompts, four prompts shared between combined train and combined validation. This prevents treating the three combined datasets as one independent final test. No semantic-duplicate or pretraining-contamination audit was performed.

The chat template defaults to the current date. This explains the reproduced hash difference between 13 and 14 Nov; pinning 13 Nov restores the stored hashes exactly. Older 11-vs-13 Nov baseline drift may have the same cause, but that older case has not been independently verified.

Machine-readable detail: data_stats.csv and data_stats.json.
