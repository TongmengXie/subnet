#!/usr/bin/env python3
"""Train a cross-layer NeuroStrike safety-neuron adapter on frozen Llama-3.2-1B.

Reuse original experiment's PKU-SafeRLHF/SciQ splits, seed, prompt formatting,
completion-only CE, and validation. No existing pretrained parameters are changed.

Run in the prior Viking .venv. The NeuroStrike repository and gated model access
must already exist. Baseline and adapter use the same held-out samples.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from math import isfinite as math_isfinite
import os
from pathlib import Path
import random
import re
import sys
from functools import partial

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from cross_layer_adapter import CrossLayerAdapter, PLACEMENTS, load_neurostrike_channel_map
from adjacent_adapters import AdjacentAdapter
from neuron_selection import count_matched_random_neurons


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('NeuroStrike'))
    ap.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct')
    ap.add_argument('--selection', type=Path, default=None,
                    help='Precomputed NeuroStrike .p (trusted local pickle)')
    ap.add_argument('--threshold', type=float, default=3.)
    ap.add_argument('--neuron-selection', choices=['neurostrike', 'random'], default='neurostrike')
    ap.add_argument('--selection-seed', type=int, default=None,
                    help='Independent random-neuron mask seed; required for random selection')
    ap.add_argument('--prompt-date', default='13 Nov 2026',
                    help='Fixed Llama chat-template date, matching the completed reference runs')
    ap.add_argument('--source', type=int, default=7)
    ap.add_argument('--target', type=int, default=8)
    ap.add_argument('--scope', choices=['pair', 'all-adjacent'], default='pair')
    ap.add_argument('--method', choices=['connected', 'independent', 'direct'], default='connected',
                    help='direct: one dense source-to-target matrix, no hidden neurons/tanh/bias')
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--activation', choices=['tanh', 'identity'], default=None)
    ap.add_argument('--bias', action='store_true', help='Biases in both hidden transforms, or direct output bias')
    ap.add_argument('--placement', choices=PLACEMENTS, default='selected_mlp',
                    help='Target injection site; see README for interpretation')
    ap.add_argument('--train-n', type=int, default=500)
    ap.add_argument('--val-n', type=int, default=100)
    ap.add_argument('--sciq-share', type=float, default=.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--max-answer-tokens', type=int, default=128)
    ap.add_argument('--val-every', type=int, default=25)
    ap.add_argument('--mini-val-n', type=int, default=24)
    ap.add_argument('--out', type=Path, default=Path('results/cross_layer'))
    ap.add_argument('--smoke', action='store_true', help='32 train / 16 val, fast integration check')
    a = ap.parse_args()
    if a.neuron_selection == 'random' and (a.selection_seed is None or a.selection_seed < 0):
        ap.error('Random selection requires a nonnegative --selection-seed')
    if a.neuron_selection == 'neurostrike' and a.selection_seed is not None:
        ap.error('--selection-seed applies only to random selection')
    a.activation = a.activation or ('identity' if a.method == 'direct' else 'tanh')
    if a.method == 'direct' and a.activation != 'identity':
        ap.error('Direct connections have no activation function; use identity')
    if a.method == 'independent' and (a.activation != 'tanh' or a.bias):
        ap.error('Independent control retains tanh without biases')
    if a.smoke:
        a.train_n, a.val_n, a.val_every, a.mini_val_n = 32, 16, 4, 8
    if not (0 < a.sciq_share < 1) or min(a.train_n, a.val_n, a.grad_accum, a.epochs) < 1:
        ap.error('Expected positive sizes/steps and SciQ share in (0,1)')
    if a.scope == 'pair' and a.method != 'connected':
        ap.error('independent/direct methods require --scope all-adjacent')
    if a.method == 'direct' and a.rank != 8:
        ap.error('--rank does not apply to direct connections; omit it')
    if a.scope == 'all-adjacent' and a.placement != 'selected_mlp':
        ap.error('All-adjacent currently supports selected_mlp only')
    if min(a.rank, a.val_every, a.mini_val_n) < 1:
        ap.error('rank/val-every/mini-val-n must be positive')
    return a


def normalized(q):
    return re.sub(r'\s+', ' ', str(q)).strip().casefold()


def get_examples(args):
    """Same selection strategy as earlier notebook (not exact semantic-overlap audit)."""
    from datasets import load_dataset
    strong = load_dataset('walledai/StrongREJECT', split='train')
    forbidden = {normalized(q) for q in strong['prompt']}
    pku_train = load_dataset('PKU-Alignment/PKU-SafeRLHF', split='train')
    pku_val = load_dataset('PKU-Alignment/PKU-SafeRLHF', split='test')
    sci_train = load_dataset('allenai/sciq', split='train')
    sci_val = load_dataset('allenai/sciq', split='validation')
    needed = {'prompt', 'response_0', 'response_1', 'is_response_0_safe',
              'is_response_1_safe', 'better_response_id'}
    for split in (pku_train, pku_val):
        if not needed.issubset(split.column_names):
            raise RuntimeError('PKU-SafeRLHF dataset schema changed')

    def pku_pair(ex):
        valid = [i for i in (0, 1) if bool(ex[f'is_response_{i}_safe'])]
        if not valid:
            return None
        better = ex.get('better_response_id')
        selected = int(better) if better in (0, 1) and int(better) in valid else valid[0]
        q, a = str(ex['prompt']).strip(), str(ex[f'response_{selected}']).strip()
        return (q, a, 'PKU-SafeRLHF') if q and a else None

    def sci_pair(ex):
        q, a = str(ex['question']).strip(), str(ex['correct_answer']).strip()
        return (q, a, 'SciQ') if q and a else None

    def sample(split, fn, n, seed):
        if n == 0:
            return []
        records = []
        for i in np.random.default_rng(seed).permutation(len(split)):
            record = fn(split[int(i)])
            if record is None or normalized(record[0]) in forbidden:
                continue
            forbidden.add(normalized(record[0]))
            records.append(record)
            if len(records) == n:
                break
        if len(records) != n:
            raise RuntimeError(f'Could only collect {len(records)}/{n} distinct samples')
        return records

    ns, nv = round(args.train_n * args.sciq_share), round(args.val_n * args.sciq_share)
    train = sample(pku_train, pku_pair, args.train_n-ns, args.seed+1) + \
            sample(sci_train, sci_pair, ns, args.seed+2)
    val = sample(pku_val, pku_pair, args.val_n-nv, args.seed+3) + \
          sample(sci_val, sci_pair, nv, args.seed+4)
    shuffled = random.Random(args.seed)
    shuffled.shuffle(train)
    shuffled.shuffle(val)
    return train, val


def tokenize_pairs(records, tokenizer, args, prompt_fn):
    tensors = []
    for q, a, domain in records:
        prefix = tokenizer(prompt_fn(q), add_special_tokens=False)['input_ids']
        answer_full = tokenizer(a, add_special_tokens=False)['input_ids']
        answer = answer_full[:args.max_answer_tokens]
        if len(answer) == len(answer_full) and tokenizer.eos_token_id is not None:
            answer = answer + [tokenizer.eos_token_id]
        if not answer or len(answer) >= args.max_length - 16:
            raise ValueError('Answer too long; adjust token limits')
        prefix = prefix[-(args.max_length-len(answer)):]
        ids = prefix + answer
        labels = [-100] * len(prefix) + answer
        tensors.append((
            torch.tensor([ids], dtype=torch.long),
            torch.tensor([labels], dtype=torch.long),
            torch.ones((1, len(ids)), dtype=torch.long), domain,
        ))
    return tensors


def evaluate(model, samples, device, adapter=None, rows=None):
    """Token-weighted teacher-forced CE; assume hooks active when adapter supplied."""
    total, total_tokens = {}, {}
    with torch.no_grad():
        for sample_id, (input_ids, labels, attention_mask, domain) in enumerate(samples):
            inputs = {'input_ids': input_ids.to(device), 'labels': labels.to(device),
                      'attention_mask': attention_mask.to(device), 'use_cache': False}
            loss = model(**inputs).loss
            ntok = int((labels[:, 1:] != -100).sum().item())
            if ntok < 1:
                raise ValueError('Sample has no completion tokens')
            if rows is not None:
                rows.append(dict(sample_id=sample_id, domain=domain, completion_tokens=ntok, ce=loss.item()))
            for key in ('all', domain):
                total[key] = total.get(key, 0.) + loss.item() * ntok
                total_tokens[key] = total_tokens.get(key, 0) + ntok
    return {key: total[key] / total_tokens[key] for key in total}


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU; run test_cross_layer.py on CPU instead')
    if args.source >= args.target:
        raise ValueError('Source layer must precede target layer')
    args.repo = args.repo.resolve()
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    selection_path = (args.selection or args.repo / 'pre_computed_sn' /
                      f'weights_{args.model.split("/")[-1]}.p').resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f'NeuroStrike weights not found: {selection_path}')
    channels = load_neurostrike_channel_map(selection_path, threshold=args.threshold)
    if args.scope == 'pair' and (args.source not in channels or args.target not in channels):
        raise ValueError(f'Selections unavailable for requested layers; available: {sorted(channels)}')
    source, target = channels.get(args.source, []), channels.get(args.target, [])
    print(f'Neuron counts: layer {args.source}: {len(source)}, layer {args.target}: {len(target)}', flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo_white = args.repo / 'white_box'
    if not (repo_white / 'util_model.py').is_file():
        raise FileNotFoundError(f'NeuroStrike white_box/util_model.py missing: {repo_white}')
    sys.path.insert(0, str(repo_white))
    import util_model

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f'Model: {args.model}; dtype: {dtype}; output: {args.out}', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to('cuda')
    model.eval()
    model.config.use_cache = False
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    for p in model.parameters():
        p.requires_grad_(False)
    reference_channels = channels
    if args.neuron_selection == 'random':
        modules = dict(model.named_modules())
        dimensions = {i: modules[f'model.layers.{i}.mlp.down_proj'].in_features for i in channels}
        channels = count_matched_random_neurons(channels, dimensions, seed=args.selection_seed)
        source, target = channels.get(args.source, []), channels.get(args.target, [])
    overlap = {i: len(set(ids) & set(reference_channels[i])) for i, ids in channels.items()}
    print(f'Neuron selection: {args.neuron_selection}; mask seed: {args.selection_seed}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, date_string=args.prompt_date)
    model_name = args.model.split('/')[-1]

    def prompt_fn(q):
        # Explicitly reuse the original experiment's prompt construction.
        generated = util_model.construct_prompt(tokenizer, model_name, [q])
        if len(generated) != 1 or not isinstance(generated[0], str):
            raise TypeError('Expected one text-only chat prompt')
        return generated[0]

    if args.scope == 'all-adjacent':
        adapter = AdjacentAdapter(model, channels, rank=args.rank, method=args.method,
                                  activation=args.activation, bias=args.bias)
    else:
        adapter = CrossLayerAdapter(model, source_layer=args.source, target_layer=args.target,
                                    source_channels=source, target_channels=target, rank=args.rank,
                                    placement=args.placement,
                                    activation=args.activation, bias=args.bias)
    print(f'Adapter scope={args.scope}, method={args.method}, parameters={adapter.parameter_count}', flush=True)
    print('All pretrained model weights frozen:',
          all(not p.requires_grad for p in model.parameters()), flush=True)

    with open(args.out / 'run_config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, f, indent=2)
    train_records, val_records = get_examples(args)
    train = tokenize_pairs(train_records, tokenizer, args, prompt_fn)
    val = tokenize_pairs(val_records, tokenizer, args, prompt_fn)
    device = model.get_input_embeddings().weight.device
    fixed_train = train[:min(8, len(train))]
    mini_val = val[:min(args.mini_val_n, len(val))]
    before_rows = []
    baseline = evaluate(model, val, device, rows=before_rows)
    print('Original CE, held-out:', baseline, flush=True)
    write_csv(args.out / 'step_zero_diagnostics.csv',
              [dict(split=name, domain=d, ce=v) for name, samples in [('fixed_train', fixed_train), ('mini_val', mini_val)]
               for d, v in evaluate(model, samples, device).items()], ['split', 'domain', 'ce'])

    probe = val[0][0].to(device)
    with torch.no_grad():
        original_logits = model(probe, use_cache=False).logits
        with adapter.installed(model):
            torch.testing.assert_close(model(probe, use_cache=False).logits,
                                       original_logits, rtol=0, atol=0)
    del original_logits
    print('PASS full-model zero-init identity', flush=True)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.)
    order = [int(i) for epoch in range(args.epochs)
             for i in np.random.default_rng(args.seed+epoch).permutation(len(train))]
    history = []
    print('Optimizing adapter only:', flush=True)
    adapter.train()
    with adapter.installed(model):
        for offset in range(0, len(order), args.grad_accum):
            batch = order[offset:offset+args.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for idx in batch:
                ids, labels, attention_mask, _domain = train[idx]
                out = model(input_ids=ids.to(device), labels=labels.to(device),
                            attention_mask=attention_mask.to(device), use_cache=False)
                loss = out.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite training loss at example {idx}')
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
            grad_norm = float(clip_grad_norm_(adapter.parameters(), max_norm=1.))
            if not math_isfinite(grad_norm):
                raise RuntimeError('Non-finite gradient norm')
            optimizer.step()
            step = len(history) + 1
            row = {'step': step, 'train_ce': float(np.mean(losses)),
                   'grad_norm': grad_norm, 'fixed_train_ce': '', 'mini_val_ce': '',
                   'mini_val_pku_ce': '', 'mini_val_sciq_ce': ''}
            if step == 1 or step % args.val_every == 0 or offset+len(batch) == len(order):
                adapter.eval()
                ft = evaluate(model, fixed_train, device)
                mv = evaluate(model, mini_val, device)
                adapter.train()
                row.update(fixed_train_ce=ft['all'], mini_val_ce=mv['all'],
                           mini_val_pku_ce=mv.get('PKU-SafeRLHF', ''),
                           mini_val_sciq_ce=mv.get('SciQ', ''))
                print(f'step {step}: train {row["train_ce"]:.3f}; '
                      f'fixed {ft["all"]:.3f}; mini-val {mv["all"]:.3f}; '
                      f'gradient norm {grad_norm:.4f}', flush=True)
            history.append(row)
            write_csv(args.out / 'training_history.csv', history, list(row))

    adapter.eval()
    with torch.no_grad(), adapter.installed(model):
        trained_probe = model(probe, use_cache=False).logits.detach().cpu()
    checkpoint = {
        'architecture': ('selected-adjacent-variants-v1' if args.bias or args.activation != ('identity' if args.method == 'direct' else 'tanh') else
                         'selected-adjacent-direct-v1' if args.method == 'direct' else
                         'selected-adjacent-v3' if args.scope == 'all-adjacent' else 'cross-layer-selected-output-basis-skip-v2'),
        'scope': args.scope, 'method': args.method,
        'neuron_selection': args.neuron_selection, 'selection_seed': args.selection_seed,
        'activation': args.activation, 'bias': args.bias,
        'prompt_date': args.prompt_date,
        'network_config': adapter.config() if args.scope == 'all-adjacent' else None,
        'model': args.model, 'source_layer': args.source, 'target_layer': args.target,
        'source_channels': source, 'target_channels': target,
        'rank': None if args.method == 'direct' else args.rank,
        'placement': args.placement,
        'state_dict': {k:v.detach().cpu() for k,v in adapter.state_dict().items()},
    }
    torch.save(checkpoint, args.out / 'cross_layer_adapter.pt')
    write_csv(args.out / 'training_history.csv', history,
              ['step','train_ce','grad_norm','fixed_train_ce','mini_val_ce',
               'mini_val_pku_ce','mini_val_sciq_ce'])

    # Reload the serialized checkpoint into a fresh adapter before final evaluation.
    saved = torch.load(args.out / 'cross_layer_adapter.pt', map_location='cpu', weights_only=True)
    if args.scope == 'all-adjacent':
        cfg = dict(saved['network_config'])
        cfg.pop('overlap_policy')
        adapter = AdjacentAdapter(model, **cfg)
    else:
        adapter = CrossLayerAdapter(model, source_layer=args.source, target_layer=args.target,
            source_channels=source, target_channels=target, rank=args.rank, placement=args.placement,
                                    activation=args.activation, bias=args.bias)
    adapter.load_state_dict(saved['state_dict'], strict=True)
    adapter.eval()
    with torch.no_grad(), adapter.installed(model):
        torch.testing.assert_close(model(probe, use_cache=False).logits.cpu(),
                                   trained_probe, rtol=0, atol=0)
    del trained_probe
    print('PASS serialized checkpoint reload identity', flush=True)
    after_rows = []
    with adapter.installed(model):
        after = evaluate(model, val, device, rows=after_rows)
    paired = []
    for before, changed, (q, answer, _) in zip(before_rows, after_rows, val_records):
        paired.append(dict(sample_id=before['sample_id'], domain=before['domain'],
            prompt_sha256=hashlib.sha256(normalized(q).encode()).hexdigest(),
            answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            completion_tokens=before['completion_tokens'], original_ce=before['ce'],
            adapted_ce=changed['ce'], change=changed['ce']-before['ce']))
    write_csv(args.out / 'heldout_per_sequence.csv', paired, list(paired[0]))
    results = [{'method': m, 'domain': d, 'heldout_ce': value}
               for m, metric in [('original', baseline),
                                 (args.method if args.scope == 'all-adjacent' else 'cross_layer', after)]
               for d, value in metric.items()]
    write_csv(args.out / 'heldout_completion_ce.csv', results,
              ['method','domain','heldout_ce'])
    print('Heldout CE:', results, flush=True)

    meta = {'model': args.model, 'rank': None if args.method == 'direct' else args.rank, 'source_layer': args.source,
            'target_layer': args.target, 'source_count': len(source),
            'target_count': len(target), 'trainable_parameters': adapter.parameter_count,
            'placement': args.placement,
            'scope': args.scope, 'method': args.method,
            'neuron_selection': args.neuron_selection, 'selection_seed': args.selection_seed,
            'overlap_with_neurostrike': overlap,
            'activation': args.activation, 'bias': args.bias,
            'prompt_date': args.prompt_date,
            'network_config': adapter.config() if args.scope == 'all-adjacent' else None,
            'source_module': getattr(adapter, 'source_name', None),
            'target_module': getattr(adapter, 'target_name', None),
            'intervention': ('per-neuron sum b*tanh(a*z)' if args.method == 'independent' else
                             f"{'direct W*source' if args.method == 'direct' else 'W2*'+args.activation+'(W1*source+b1)'}; "
                             f"bias={args.bias}; placement={args.placement}"),
            'pretrained_frozen': True, 'selection_threshold': args.threshold,
            'trusted_selection_path': str(selection_path),
            'train_n': args.train_n, 'val_n': args.val_n, 'seed': args.seed,
            'epochs': args.epochs, 'grad_accum': args.grad_accum,
            'optimizer_steps': len(history), 'lr': args.lr,
            'datasets': ['PKU-Alignment/PKU-SafeRLHF', 'allenai/sciq'],
            'train_prompt_sha256': [hashlib.sha256(normalized(q).encode()).hexdigest()
                                    for q,_,_ in train_records],
            'val_prompt_sha256': [hashlib.sha256(normalized(q).encode()).hexdigest()
                                  for q,_,_ in val_records],
            'max_length': args.max_length, 'max_answer_tokens': args.max_answer_tokens,
            'sciq_share': args.sciq_share, 'dtype': str(dtype),
            'python': platform.python_version(), 'torch': torch.__version__,
            'code_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in ('train_cross_layer.py', 'cross_layer_adapter.py', 'adjacent_adapters.py', 'neuron_selection.py')},
            'tokenized_sha256': {name: hashlib.sha256(json.dumps(
                [(ids.tolist(), labels.tolist(), domain) for ids, labels, _, domain in samples]).encode()).hexdigest()
                for name, samples in [('train', train), ('val', val)]},
            'notes': 'No safety ASR judged. CE does not establish safety.'}
    with open(args.out / 'manifest.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved checkpoint, training history, validation CE, and manifest to', args.out, flush=True)


if __name__ == '__main__':
    main()
