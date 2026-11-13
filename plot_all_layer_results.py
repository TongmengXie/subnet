"""CPU-only figures for saved all-adjacent runs; no model loading."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/neurostrike-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent / 'results'
OUT = ROOT / 'all_layer_figures'
OUT.mkdir(exist_ok=True)
COLORS = {'connected': '#2467a8', 'independent': '#cc701a'}
DOMAINS = ('all', 'PKU-SafeRLHF', 'SciQ')
runs = {}
for path in ROOT.glob('all-adjacent_*/manifest.json'):
    meta = json.loads(path.read_text())
    if meta['train_n'] != 500:
        continue
    folder = path.parent
    def read(name):
        with (folder / name).open() as f:
            return list(csv.DictReader(f))
    ce = {(r['method'], r['domain']): float(r['heldout_ce'])
          for r in read('heldout_completion_ce.csv')}
    runs[meta['seed'], meta['method']] = (meta, ce, read('training_history.csv'),
                                         read('step_zero_diagnostics.csv'))
seeds = sorted({s for s, _ in runs})
assert seeds == [42, 43, 44]
for seed in seeds:
    a, b = runs[seed, 'connected'], runs[seed, 'independent']
    for key in ('tokenized_sha256', 'train_prompt_sha256', 'val_prompt_sha256',
                'trainable_parameters', 'code_sha256', 'lr', 'optimizer_steps'):
        assert a[0][key] == b[0][key], (seed, key)
    assert a[0]['trainable_parameters'] == 19896
    for domain in DOMAINS:
        assert a[1]['original', domain] == b[1]['original', domain]

only_runs = {}
for path in ROOT.glob('neurostrike_only_*/seed*/manifest.json'):
    meta = json.loads(path.read_text())
    seed = meta['seed']
    assert seed not in only_runs, f'Duplicate neurons-only result for seed {seed}'
    assert meta['tokenized_sha256'] == runs[seed, 'connected'][0]['tokenized_sha256']
    assert meta['baseline_max_per_sequence_difference'] <= 1e-5
    with (path.parent / 'heldout_completion_ce.csv').open() as f:
        ce = {(r['method'], r['domain']): float(r['heldout_ce']) for r in csv.DictReader(f)}
    for domain in DOMAINS:
        assert abs(ce['original', domain]-runs[seed, 'connected'][1]['original', domain]) <= 1e-5
    only_runs[seed] = ce
if only_runs and sorted(only_runs) != seeds:
    raise ValueError('Wait for neurons-only results for all three seeds before plotting')
def plot_heldout(include_only=False):
    labels = ['Original', 'NeuroStrike\nwithout edges', 'NeuroStrike\nwith edges']
    if include_only:
        labels.insert(1, 'NeuroStrike\nneurons only')
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    summary = []
    for ax, domain in zip(axes, DOMAINS):
        values = [[] for _ in labels]
        for seed in seeds:
            ce_c, ce_i = runs[seed, 'connected'][1], runs[seed, 'independent'][1]
            base = ce_c['original', domain]
            absolute = [base, ce_i['independent', domain], ce_c['connected', domain]]
            if include_only:
                absolute.insert(1, only_runs[seed]['neurostrike_only', domain])
            for idx, (name, val) in enumerate(zip(labels, absolute)):
                values[idx].append(val)
                summary.append(dict(seed=seed, domain=domain, setting=name.replace('\n', ' '),
                                    original_ce=base, heldout_ce=val, ce_change=val-base,
                                    evidence='saved held-out CE'))
        means = [sum(v)/len(v) for v in values]
        colors = ['.65', COLORS['independent'], COLORS['connected']]
        if include_only:
            colors.insert(1, '#8361a9')
        ax.bar(range(len(labels)), means, color=colors, width=.6)
        for i, (mean, vals) in enumerate(zip(means, values)):
            ax.scatter([i-.09, i, i+.09], vals, color='.15', s=20, zorder=3)
            ax.annotate(f'{mean:.4f}', (i, max(vals)), xytext=(0, 9), textcoords='offset points', ha='center')
        ax.set_ylim(0, max(max(v) for v in values)*1.28)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title(domain if domain!='all' else 'Overall')
        ax.set_ylabel('Held-out completion loss (lower is better)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Original model vs trained NeuroStrike neurons · all consecutive layers')
    fig.text(.5, .025, 'Bars: mean held-out loss; dots: three seed-specific datasets (42, 43, 44), not confidence intervals.\n'
             'Trained versions: 19,896 weights each. With edges uses 8 added shared nodes per consecutive layer pair.\n'
             'Training per seed: 400 PKU + 100 SciQ; held-out: 80 PKU + 20 SciQ. This measures answer prediction, not safety.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .19, 1, .93))
    if include_only:
        fig.savefig(OUT / 'heldout_with_neurostrike_only.png', dpi=180)
    else:
        fig.savefig(OUT / 'heldout_four_settings.png', dpi=180)  # legacy filename
        fig.savefig(OUT / 'heldout_three_settings.png', dpi=180)
    plt.close(fig)

    return summary

summary = plot_heldout()
if only_runs:
    plot_heldout(include_only=True)

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for col, seed in enumerate(seeds):
    for method in ('connected', 'independent'):
        meta, ce, history, zero = runs[seed, method]
        for row, (metric, split) in enumerate([('fixed_train_ce', 'fixed_train'), ('mini_val_ce', 'mini_val')]):
            base = next(float(r['ce']) for r in zero if r['split']==split and r['domain']=='all')
            pts = [(0, 0)] + [(int(r['step']), float(r[metric])-base) for r in history if r[metric]]
            axes[row, col].plot(*zip(*pts), marker='.', color=COLORS[method], label=('with edges' if method=='connected' else 'without edges')+' trained')
            axes[row, col].axhline(0, color='.4', ls='--', lw=1, label='original / untrained' if method=='connected' else None)
            axes[row, col].grid(alpha=.2)
            axes[row, col].set_title(f'Seed {seed} · {split.replace("_", " ")}')
            axes[row, col].set_ylabel('CE change from step 0')
            axes[row, col].set_xlabel('Optimizer update')
axes[0, 0].legend(fontsize=9)
fig.suptitle('Training diagnostics · fixed examples, token-weighted CE\nMini-validation overlaps final evaluation; it is not an independent test set')
fig.tight_layout(rect=(0, 0, 1, .92))
fig.savefig(OUT / 'training_comparison.png', dpi=180)
plt.close(fig)

cfg = runs[42, 'connected'][0]['network_config']
counts = {int(k): len(v) for k, v in cfg['channels'].items()}
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis('off')
def arrow(ax, start, end, color='.65', width=1.3, label=None):
    ax.annotate('', end, start, arrowprops=dict(arrowstyle='->', color=color, lw=width), zorder=1)
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2, label,
                ha='center', va='center', fontsize=11, color=color, backgroundcolor='white', zorder=5)
def node(ax, x, y, label, color='.4'):
    ax.text(x, y, label, ha='center', va='center', fontsize=11, zorder=4,
            bbox=dict(boxstyle='round,pad=.4', facecolor='white', edgecolor=color))
ax=axes[0];ax.set_title('Original network', fontsize=16)
ax.text(5,11.5,'Neuron activations',ha='center',fontsize=13)
for x,n in [(2.5,'1'),(7.5,'2')]:
    node(ax,x,10,'neuron '+n)
    arrow(ax,(x,9.65),(x,5.4))
    node(ax,x,5,'unchanged')
ax.text(5,7.5,'No added weights',ha='center',fontsize=13,backgroundcolor='white')
ax.text(5,3.4,'Original model weights frozen',ha='center',fontsize=12)
ax.text(5,1.6,'No adapter training',ha='center',fontsize=13)
ax=axes[1];ax.set_title('Without edges · trained',fontsize=16)
c=COLORS['independent']
ax.text(5,11.5,'Each unit reads ONE neuron',ha='center',fontsize=13)
for x,n in [(2.5,'1'),(7.5,'2')]:
    node(ax,x,10,'neuron '+n)
    for offset,k in [(-1,'1'),(1,'2')]:
        xx=x+offset
        arrow(ax,(x,9.65),(xx,8.1),c,label='a'+n+k)
        node(ax,xx,7.7,'tanh')
        arrow(ax,(xx,7.3),(x,5.4),c,label='b'+n+k)
    node(ax,x,5,'sum + neuron '+n)
ax.text(5,3.8,'7 or 8 tanh units per neuron\nEach added unit reads only its own neuron',ha='center',fontsize=11)
ax.text(5,2.1,'Training changes the orange weights a and b',ha='center',fontsize=12,color=c)
ax.text(5,.8,'Applied at layers 1–15 · 19,896 weights',ha='center',fontsize=12)
ax=axes[2];ax.set_title('With edges · trained',fontsize=16)
c=COLORS['connected']
ax.text(5,11.5,'Each unit reads MULTIPLE neurons',ha='center',fontsize=13)
for x,n in [(2.5,'1'),(7.5,'2')]: node(ax,x,10,'source '+n)
for x in [2.5,7.5]: node(ax,x,7.5,'tanh')
for x,n in [(2.5,'1'),(7.5,'2')]: node(ax,x,5,'sum + target '+n)
for x in [2.5,7.5]:
    for xx in [2.5,7.5]:
        arrow(ax,(x,9.65),(xx,7.9),c)
        arrow(ax,(x,7.1),(xx,5.4),c)
ax.text(5,8.8,'W1',ha='center',fontsize=12,color=c,backgroundcolor='white')
ax.text(5,6.2,'W2',ha='center',fontsize=12,color=c,backgroundcolor='white')
ax.text(5,3.8,'8 shared tanh units per layer pair\nTwo shown; source layer i → target layer i+1',ha='center',fontsize=11)
ax.text(5,2.1,'Training changes the blue weights W1 and W2',ha='center',fontsize=12,color=c)
ax.text(5,.8,'Pairs 0→1 through 14→15 · 19,896 weights',ha='center',fontsize=12)
fig.suptitle('Without edges vs with edges between selected neurons',fontsize=18)
fig.text(.5,.035,'Training for both adapters: model predicts answer → compare with reference answer → backpropagation updates colored weights.\n'
         'Edges here means added connections between selected neurons. The original network remains connected and frozen.',ha='center',fontsize=12)
fig.tight_layout(rect=(0,.11,1,.94))
fig.savefig(OUT / 'architectures_all_layers.png', dpi=180)
plt.close(fig)

# Compact insertion-point inventory; the source is the same in every variant.
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(-.6, 8.6); ax.set_ylim(-1.3, 6.8); ax.axis('off')
stages = ['Layer input', 'Attention\nRMSNorm', 'Attention', '+ residual',
          'MLP\nRMSNorm', 'SwiGLU', 'Selected MLP\nneurons', 'Down\nprojection', '+ residual']
for i, stage in enumerate(stages):
    ax.text(i, 5.8, stage, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=.45', facecolor='#f1f1f1', edgecolor='.7'))
    if i < 8:
        arrow(ax, (i+.36,5.8),(i+.64,5.8))
variants = [
    ('Before attention RMSNorm', 0, '7→8 pair only'),
    ('After attention RMSNorm; before attention', 1.5, '7→8 pair only; repaired rerun'),
    ('After attention + residual; before MLP RMSNorm', 3.5, '7→8 pair only; MLP branch receives addition'),
    ('After MLP RMSNorm; before SwiGLU', 4.5, '7→8 pair only'),
    ('Selected MLP neurons; before down projection', 6, 'USED for all 15 layer pairs'),
]
for row, (name, x, status) in enumerate(variants):
    y=4.6-row*1.05
    color=COLORS['connected'] if row==4 else '.4'
    ax.plot([x,x],[y+.12,5.35],color=color,lw=1,alpha=.5)
    ax.scatter(x,y,s=95,color=color,zorder=3)
    ax.text(-.5,y+.2,f'{row+1}. {name}',ha='left',va='bottom',fontsize=11,
            color=color,backgroundcolor='white')
    ax.text(-.5,y-.12,status,ha='left',va='top',fontsize=10,color=color,backgroundcolor='white')
fig.suptitle('Five tested ways to add a signal from selected neurons in an earlier layer',fontsize=16)
fig.text(.5,.045,'All variants read source neurons immediately before the source down projection. Only variant 5 adds directly to selected target neurons.\n'
         'Variants 1–4 map the signal to hidden-state space using frozen down-projection weights. Placement, representation and branch routing differ.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.11,1,.94))
fig.savefig(OUT / 'edge_placement_variants.png', dpi=180)
plt.close(fig)
with (OUT / 'three_settings_summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
(OUT / 'README.md').write_text('# All-layer figures\n\nCurrent specification: [CURRENT_EXPERIMENT.md](../../CURRENT_EXPERIMENT.md).\nThese figures use the six full all-layer runs at seeds 42, 43, 44; smoke and\nsingle-pair runs are excluded.\n\n- `heldout_three_settings.png`: actual held-out completion loss bars, zero-based\n  axes; original / trained without edges / trained with edges. Bars are means\n  across three seed-specific datasets; dots are seeds, not confidence intervals.\n- `three_settings_summary.csv`: values underlying the current loss chart.\n- `architectures_all_layers.png`: representative computations and trainable weights.\n- `edge_placement_variants.png`: five insertion points tested in the pair study;\n  only selected-MLP insertion is implemented/tested for all consecutive layers.\n- `training_comparison.png`: fixed-train and mini-validation changes from step zero.\n- `heldout_four_settings.png`: legacy alias of the three-setting chart, not a fourth\n  measured setting. `four_settings_summary.csv` is an older export.\n\nWith edges means **eight new shared hidden neurons with tanh and no biases per\nconsecutive pair**; it is not a direct full-connection result. Without edges means\nseparate trained adjustments to each selected neuron, with no added connections\nbetween selected neurons. The original network is retained and frozen in both.\nBoth trained versions use 19,896 weights. The untrained adapter adds zero and is\nomitted because it equals the original. Direct connections without hidden neurons\nand the tanh/bias variants are planned comparisons, not measured bars.\n\nIf all three neurons-only evaluations pass matching-data/baseline checks,\n`plot_all_layer_results.py` also writes `heldout_with_neurostrike_only.png`.\nNeurons-only zeros the nonselected MLP neurons but retains attention and residuals;\nit is different from the old experiment removing selected neurons.\n\nNo per-layer ablation scores are available. No safety superiority follows from\ncompletion loss. Mini-validation overlaps final evaluation. Regenerate on CPU:\n\n```bash\nOMP_NUM_THREADS=1 .venv/bin/python plot_all_layer_results.py\n```\n')

print(f'Saved three figures and source summary to {OUT}')
