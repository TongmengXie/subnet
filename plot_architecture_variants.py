"""Compare measured architectures with matched data/steps; never mix smoke/full."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/neurostrike-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARIANTS = [('independent','tanh',False), ('connected','tanh',False),
            ('connected','tanh',True), ('connected','identity',False),
            ('connected','identity',True), ('direct','identity',False),
            ('direct','identity',True)]
LABELS = ['Without edges', '8 nodes\ntanh', '8 nodes\ntanh + bias',
          '8 nodes\nno tanh', '8 nodes\nno tanh + bias', 'Direct', 'Direct\n+ bias']


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def collect(root, train_n, seeds):
    runs = {}
    for p in root.glob('all-adjacent_*/manifest.json'):
        m = json.loads(p.read_text())
        if m.get('neuron_selection', 'neurostrike') != 'neurostrike':
            continue
        if m['train_n'] != train_n or m['seed'] not in seeds:
            continue
        key = (m['method'], m.get('activation', 'identity' if m['method']=='direct' else 'tanh'), m.get('bias',False))
        if key not in VARIANTS:
            continue
        k = m['seed'], key
        if k in runs:
            raise ValueError(f'Duplicate configuration {k}: select a results root with one run each')
        runs[k] = m, read(p.parent/'heldout_per_sequence.csv'), p.parent
    required = [(s,k) for s in seeds for k in VARIANTS]
    missing = [k for k in required if k not in runs]
    if missing:
        raise ValueError(f'Missing measured runs: {missing}; no plot generated')
    fields = ['model','placement','selection_threshold','train_n','val_n','seed','epochs',
              'grad_accum','optimizer_steps','lr','max_length','max_answer_tokens',
              'sciq_share','dtype','tokenized_sha256','train_prompt_sha256','val_prompt_sha256']
    for seed in seeds:
        ref, samples, _ = runs[seed,VARIANTS[1]]
        for key in VARIANTS:
            m, rows, folder = runs[seed,key]
            for field in fields:
                if m[field] != ref[field]:
                    raise ValueError(f'Incomparable {field}: {folder}')
            if m['network_config']['channels'] != ref['network_config']['channels']:
                raise ValueError(f'Selection mismatch: {folder}')
            assert len(rows)==len(samples)==m['val_n']
            for a,b in zip(rows,samples):
                for field in ['sample_id','domain','prompt_sha256','answer_sha256','completion_tokens']:
                    if a[field] != b[field]:
                        raise ValueError(f'Per-sequence mismatch {field}: {folder}')
                assert abs(float(a['original_ce'])-float(b['original_ce'])) < 1e-5
                assert all(np.isfinite(float(a[k])) for k in ['original_ce','adapted_ce'])
            ce = {(r['method'],r['domain']):float(r['heldout_ce']) for r in read(folder/'heldout_completion_ce.csv')}
            for domain in ['all','PKU-SafeRLHF','SciQ']:
                rr = [r for r in rows if domain=='all' or r['domain']==domain]
                for col,method in [('original_ce','original'),('adapted_ce',key[0])]:
                    loss = sum(float(r[col])*int(r['completion_tokens']) for r in rr)/sum(int(r['completion_tokens']) for r in rr)
                    assert abs(loss-ce[method,domain])<1e-10
    return runs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('results'))
    ap.add_argument('--out',type=Path,default=Path('results/architecture_variant_figures'))
    ap.add_argument('--train-n',type=int,default=32)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42])
    args=ap.parse_args()
    runs=collect(args.root,args.train_n,args.seeds)
    fig,axes=plt.subplots(1,3,figsize=(20,6))
    summary=[]; uncertainty=[]
    for ax,domain in zip(axes,['all','PKU-SafeRLHF','SciQ']):
        vals=[[] for _ in range(8)]
        for seed in args.seeds:
            baseline_rows=runs[seed,VARIANTS[1]][1]
            for idx,key in enumerate([None]+VARIANTS):
                meta,rows,_=runs[seed,key or VARIANTS[1]]
                rr=[r for r in rows if domain=='all' or r['domain']==domain]
                w=np.array([int(r['completion_tokens']) for r in rr])
                losses=np.array([float(r['adapted_ce'] if key else r['original_ce']) for r in rr])
                loss=float(np.dot(w,losses)/sum(w));vals[idx].append(loss)
                summary.append(dict(seed=seed,domain=domain,setting='Original' if key is None else LABELS[idx-1].replace('\n',' '),
                                    heldout_ce=loss,parameters=meta['trainable_parameters'] if key else 0,
                                    train_n=meta['train_n'],val_n=meta['val_n'],steps=meta['optimizer_steps']))
                if key:
                    ref=np.array([float(r['adapted_ce']) for r in baseline_rows if domain=='all' or r['domain']==domain])
                    rng=np.random.default_rng(42)
                    ids=rng.integers(0,len(rr),size=(2000,len(rr)))
                    boot=(w[ids]*(losses-ref)[ids]).sum(axis=1)/w[ids].sum(axis=1)
                    lo,hi=np.quantile(boot,[.025,.975])
                    uncertainty.append(dict(seed=seed,domain=domain,setting=LABELS[idx-1].replace('\n',' '),
                        change_vs_8_tanh=float(np.dot(w,losses-ref)/sum(w)),bootstrap_low=float(lo),bootstrap_high=float(hi),
                        examples=len(rr),note='Paired example bootstrap only; excludes training/seed uncertainty; exploratory'))
        for i,ys in enumerate(vals):
            avg=sum(ys)/len(ys)
            ax.bar(i,avg,color='.65' if i==0 else '#447aa3')
            ax.scatter([i]*len(ys),ys,c='.15',s=12)
            ax.annotate(f'{avg:.4f}',(i,max(ys)),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(8),['Original']+LABELS,rotation=35,ha='right',fontsize=9)
        ax.set_ylim(0,max(max(v) for v in vals)*1.2)
        ax.set_ylabel('Held-out completion loss (lower is better)');ax.set_title(domain)
    m=next(iter(runs.values()))[0]
    fig.suptitle(f'Matched architecture variants · {args.train_n} training / {m["val_n"]} held-out · {m["optimizer_steps"]} updates')
    fig.text(.5,.025,f'Seeds: {args.seeds}. '+('SMOKE TEST: preliminary, not a final architecture ranking. ' if args.train_n==32 else '')+
             '\nBudgets and initialization differ; all original weights frozen. Completion loss does not measure safety.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.12,1,.94))
    dest=args.out/f"train{args.train_n}_seeds{'-'.join(map(str,args.seeds))}"
    dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'architecture_loss.png',dpi=180);plt.close(fig)
    for name,rows in [('summary.csv',summary),('paired_bootstrap.csv',uncertainty)]:
        with (dest/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Validated settings, per-sequence data/baselines and aggregate loss. Saved {dest}')


if __name__=='__main__':
    main()
