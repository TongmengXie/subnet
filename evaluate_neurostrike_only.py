"""Evaluate selected MLP neurons only, against the three saved all-layer runs.

No optimizer or training. Retains the full attention/residual backbone and zeros
the complement of the gate/up union immediately before every MLP down_proj.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import sys
from functools import partial

import numpy as np
import torch

from cross_layer_adapter import CrossLayerAdapter, load_neurostrike_channel_map
from train_cross_layer import evaluate, get_examples, normalized, tokenize_pairs, write_csv


@contextmanager
def selected_only(model, channels):
    layers = model.model.layers
    if set(channels) != set(range(len(layers))):
        raise ValueError('Require an explicit nonempty selection for every MLP layer')
    handles = []
    try:
        for i, layer in enumerate(layers):
            down = layer.mlp.down_proj
            ids = CrossLayerAdapter._validate_channels(channels[i], down.in_features, 'selected')
            mask = torch.zeros(down.in_features, device=down.weight.device, dtype=down.weight.dtype)
            mask[list(ids)] = 1
            def hook(_module, args, keep=mask):
                return (args[0] * keep,) + args[1:]
            handles.append(down.register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def fingerprint(samples):
    return hashlib.sha256(json.dumps(
        [(ids.tolist(), labels.tolist(), domain) for ids, labels, _, domain in samples]
    ).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Run pretrained-model evaluation in a Slurm GPU allocation')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / 'NeuroStrike/white_box'))
    import util_model
    references = []
    for seed in (42, 43, 44):
        paths = list((root / 'results').glob(f'all-adjacent_connected_selected_mlp_seed{seed}_*/manifest.json'))
        paths = [p for p in paths if json.loads(p.read_text())['train_n'] == 500]
        if len(paths) != 1:
            raise ValueError(f'Expected one full connected reference for seed {seed}')
        references.append(paths[0])
    first = json.loads(references[0].read_text())
    model_id = first['model']
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if str(dtype) != first['dtype']:
        raise ValueError('GPU dtype differs from reference')
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                               local_files_only=True).to('cuda').eval()
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template,
                                           date_string=first.get('prompt_date','13 Nov 2026'))
    selection = root / 'NeuroStrike/pre_computed_sn' / f'weights_{model_id.split("/")[-1]}.p'
    channels = load_neurostrike_channel_map(selection, threshold=first['selection_threshold'])
    def prompt_fn(q):
        return util_model.construct_prompt(tokenizer, model_id.split('/')[-1], [q])[0]
    for path in references:
        meta = json.loads(path.read_text())
        seed = meta['seed']
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        # Reconstruct the exact sampler settings, including training prompts used
        # by its duplicate-exclusion filter. No training is performed.
        config = json.loads((path.parent / 'run_config.json').read_text())
        cfg = argparse.Namespace(**config)
        train_records, val_records = get_examples(cfg)
        train = tokenize_pairs(train_records, tokenizer, cfg, prompt_fn)
        val = tokenize_pairs(val_records, tokenizer, cfg, prompt_fn)
        assert fingerprint(train) == meta['tokenized_sha256']['train']
        assert fingerprint(val) == meta['tokenized_sha256']['val']
        assert {int(i): ids for i, ids in meta['network_config']['channels'].items()} == channels
        before, after = [], []
        baseline = evaluate(model, val, 'cuda', rows=before)
        # Baseline drift invalidates a direct comparison: fail rather than append
        # an incomparable neurons-only bar to the historical results.
        import csv
        old = list(csv.DictReader((path.parent / 'heldout_per_sequence.csv').open()))
        for row, prior in zip(before, old):
            assert row['sample_id'] == int(prior['sample_id'])
            assert row['completion_tokens'] == int(prior['completion_tokens'])
            if abs(row['ce'] - float(prior['original_ce'])) > 1e-5:
                raise ValueError(f'Baseline drift for seed {seed}, sample {row["sample_id"]}')
        with selected_only(model, channels):
            only = evaluate(model, val, 'cuda', rows=after)
        assert all(torch.isfinite(torch.tensor(r['ce'])) for r in after)
        folder = args.out / f'seed{seed}'
        folder.mkdir()
        rows = []
        for b, a, (q, answer, _) in zip(before, after, val_records):
            rows.append(dict(sample_id=b['sample_id'], domain=b['domain'],
                prompt_sha256=hashlib.sha256(normalized(q).encode()).hexdigest(),
                answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                completion_tokens=b['completion_tokens'], original_ce=b['ce'],
                adapted_ce=a['ce'], change=a['ce']-b['ce']))
        write_csv(folder / 'heldout_per_sequence.csv', rows, list(rows[0]))
        metrics = [dict(method=m, domain=d, heldout_ce=value)
                   for m, scores in [('original', baseline), ('neurostrike_only', only)]
                   for d, value in scores.items()]
        write_csv(folder / 'heldout_completion_ce.csv', metrics, ['method', 'domain', 'heldout_ce'])
        manifest = dict(method='neurostrike_only', seed=seed, model=model_id,
            reference_manifest=str(path), tokenized_sha256=meta['tokenized_sha256'],
            selected_neurons=channels, selected_counts={i:len(ids) for i,ids in channels.items()},
            intervention='Keep selected SwiGLU neurons only before down_proj at all 16 layers; attention and residuals retained',
            trainable_parameters=0, optimizer_steps=0, evaluation_only=True,
            baseline_max_per_sequence_difference=max(abs(b['ce']-float(p['original_ce'])) for b,p in zip(before,old)),
            code_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                         for name in ['evaluate_neurostrike_only.py', 'train_cross_layer.py', 'cross_layer_adapter.py']},
            selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
            dtype=str(dtype), torch=torch.__version__)
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'Seed {seed}: {metrics}', flush=True)


if __name__ == '__main__':
    main()
