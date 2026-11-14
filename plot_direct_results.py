"""Plot measured direct-edge results beside matched existing controls, CPU only."""
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/neurostrike-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, default=Path('results/direct_edge_figures'))
    args = ap.parse_args()
    runs = {}
    for p in args.root.glob('all-adjacent_*/manifest.json'):
        m = json.loads(p.read_text())
        if m.get('neuron_selection', 'neurostrike') != 'neurostrike':
            continue
        if m.get('bias', False) or m.get('activation', 'identity' if m.get('method') == 'direct' else 'tanh') != ('identity' if m.get('method') == 'direct' else 'tanh'):
            continue
        if m['train_n'] != 500 or m['method'] not in ('connected','independent','direct'):
            continue
        key = m['seed'], m['method']
        if key in runs:
            raise ValueError(f'Duplicate run for {key}; select a clean results root')
        with (p.parent / 'heldout_completion_ce.csv').open() as f:
            ce = {(r['method'],r['domain']):float(r['heldout_ce']) for r in csv.DictReader(f)}
        runs[key] = m, ce
    if not all((seed,'direct') in runs for seed in (42,43,44)):
        raise SystemExit('No complete direct-edge comparison yet: need measured full runs at seeds 42, 43, 44. No figure generated.')
    fields = ['model','selection_threshold','placement','train_n','val_n','epochs',
              'grad_accum','optimizer_steps','lr','max_length','max_answer_tokens',
              'sciq_share','dtype','tokenized_sha256','train_prompt_sha256','val_prompt_sha256']
    domains = ['all','PKU-SafeRLHF','SciQ']
    for seed in (42,43,44):
        ref, baseline = runs[seed,'connected']
        for method in ('independent','direct'):
            m, ce = runs[seed,method]
            for field in fields:
                if m[field] != ref[field]:
                    raise ValueError(f'Incomparable {field} for seed {seed}, {method}')
            assert m['network_config']['channels'] == ref['network_config']['channels']
            for domain in domains:
                assert abs(ce['original',domain]-baseline['original',domain]) < 1e-5
        m = runs[seed,'direct'][0]
        assert m['rank'] is None and m['placement']=='selected_mlp'
        assert m['network_config']['method']=='direct'
        c = {int(i):len(v) for i,v in m['network_config']['channels'].items()}
        assert m['trainable_parameters'] == sum(c[i]*c[i+1] for i in c if i+1 in c)
    labels = ['Original','Without edges\ntrained','With edges\n8 tanh nodes','With edges\ndirect, no hidden nodes']
    methods = ['original','independent','connected','direct']
    fig, axes = plt.subplots(1,3,figsize=(17,6))
    rows = []
    for ax,domain in zip(axes,domains):
        for i,method in enumerate(methods):
            vals=[]
            for seed in (42,43,44):
                meta, ce = runs[seed,'connected' if method=='original' else method]
                value=ce[method,domain]
                assert math.isfinite(value)
                vals.append(value)
                rows.append(dict(seed=seed,domain=domain,method=method,heldout_ce=value,
                                 parameters=0 if method=='original' else meta['trainable_parameters']))
            mean=sum(vals)/3
            ax.bar(i,mean,color=['.65','#cc701a','#2467a8','#39815a'][i],width=.6)
            ax.scatter([i-.08,i,i+.08],vals,color='.15',s=18)
            ax.annotate(f'{mean:.4f}',(i,max(vals)),xytext=(0,8),textcoords='offset points',ha='center')
        ax.set_ylim(bottom=0,top=ax.get_ylim()[1]*1.12)
        ax.set_xticks(range(4),labels,fontsize=9)
        ax.set_title('Overall' if domain=='all' else domain)
        ax.set_ylabel('Held-out completion loss (lower is better)')
    budget=runs[42,'direct'][0]['trainable_parameters']
    fig.suptitle('Measured architecture comparison: eight tanh nodes vs direct full connections')
    fig.text(.5,.025,f'Means across three seed-specific datasets; dots are seeds, not confidence intervals.\n'
             f'Hidden-node / without-edges controls: 19,896 weights each; direct: {budget:,} weights. Budgets differ.\n'
             'Full frozen backbone retained; same selected-MLP insertion. Completion loss is not a safety score.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.18,1,.93))
    args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/'direct_vs_hidden_loss.png',dpi=180)
    plt.close(fig)
    with (args.out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Saved measured comparison to {args.out}')


if __name__=='__main__':
    main()
