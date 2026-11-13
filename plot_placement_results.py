#!/usr/bin/env python3
"""Plot saved placement CSVs on CPU; never load a model or retrain."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/neurostrike-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cross_layer_adapter import PLACEMENTS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, default=Path('results/cross_layer_figures'))
    args = ap.parse_args()
    runs = []
    for path in sorted(args.root.glob('cross_layer_*/manifest.json')):
        meta = json.loads(path.read_text())
        folder = path.parent
        with (folder / 'heldout_completion_ce.csv').open() as f:
            ce = {(r['method'], r['domain']): float(r['heldout_ce']) for r in csv.DictReader(f)}
        with (folder / 'training_history.csv').open() as f:
            history = list(csv.DictReader(f))
        runs.append((folder, meta, ce, history))
    if not runs:
        raise SystemExit('No completed runs found')
    reference = runs[0][1]
    fields = ['model', 'rank', 'source_layer', 'target_layer', 'source_count', 'target_count',
              'trainable_parameters', 'selection_threshold', 'train_n', 'val_n', 'seed',
              'epochs', 'grad_accum', 'optimizer_steps', 'lr',
              'train_prompt_sha256', 'val_prompt_sha256']
    for folder, meta, ce, history in runs:
        for key in fields:
            if meta[key] != reference[key]:
                raise ValueError(f'Incomparable manifest {key}: {folder}')
        assert len(history) == meta['optimizer_steps']
        assert [int(r['step']) for r in history] == list(range(1, len(history)+1))
        assert not set(meta['train_prompt_sha256']) & set(meta['val_prompt_sha256'])
        for domain in ('all', 'PKU-SafeRLHF', 'SciQ'):
            assert ce['original', domain] == runs[0][2]['original', domain]
    args.out.mkdir(parents=True, exist_ok=True)
    present = {m['placement'] for _, m, _, _ in runs}
    missing = [p for p in PLACEMENTS if p not in present]
    note = 'Missing result: ' + ', '.join(missing) if missing else 'All placements available'
    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 3*len(runs)), squeeze=False)
    for row, (folder, meta, ce, history) in enumerate(runs):
        for col, metrics in enumerate((('train_ce',), ('fixed_train_ce', 'mini_val_ce'), ('grad_norm',))):
            ax = axes[row, col]
            for metric in metrics:
                points = [(int(r['step']), float(r[metric])) for r in history if r[metric] != '']
                ax.plot(*zip(*points), label=metric, marker='.' if col == 1 else None)
            ax.set_title(meta['placement'])
            ax.set_xlabel('Optimizer step'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Training diagnostics — changing-example mean CE vs fixed token-weighted CE\n'+note)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(args.out/'training_diagnostics.png', dpi=160); plt.close(fig)
    summary = []
    for delta in (True, False):
        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        for ax, domain in zip(axes, ('all', 'PKU-SafeRLHF', 'SciQ')):
            for i, placement in enumerate(PLACEMENTS):
                matches = [r for r in runs if r[1]['placement'] == placement]
                if not matches:
                    ax.text(i, 0 if delta else .1, 'missing', rotation=90, ha='center', va='bottom')
                for folder, meta, ce, history in matches:
                    baseline, adapted = ce['original', domain], ce['cross_layer', domain]
                    if delta:
                        ax.scatter(i, adapted-baseline)
                        summary.append(dict(run=folder.name, placement=placement, domain=domain,
                                            original=baseline, adapted=adapted, change=adapted-baseline))
                    else:
                        ax.scatter(i-.08, baseline, c='black', marker='x', label='original' if i == 0 else None)
                        ax.scatter(i+.08, adapted, c='tab:blue', label='adapted' if i == 0 else None)
            if delta: ax.axhline(0, color='black', lw=.8)
            else: ax.legend()
            ax.set_xticks(range(len(PLACEMENTS)), PLACEMENTS, rotation=35, ha='right')
            ax.set_title(domain); ax.set_ylabel('Adapted − original CE' if delta else 'Completion CE'); ax.grid(alpha=.2)
        fig.suptitle('Single seed; token-weighted held-out CE; no uncertainty estimate\n'+note)
        fig.subplots_adjust(bottom=.38, top=.86, wspace=.3); fig.savefig(args.out/('heldout_ce_change.png' if delta else 'heldout_ce_absolute.png'), dpi=160); plt.close(fig)
    with (args.out/'placement_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(json.dumps({'completed_runs': [r[0].name for r in runs], 'missing': missing,
                      'manifest_comparison': 'matched', 'summary': summary}, indent=2))
if __name__ == '__main__':
    main()
