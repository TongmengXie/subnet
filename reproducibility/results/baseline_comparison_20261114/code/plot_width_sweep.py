"""Plot a completed, matched width sweep; never substitute missing measurements."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/neurostrike-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--sweep',type=Path,default=Path('results/width_sweep_20261114_fixeddate'))
 args=ap.parse_args();plan=json.loads((args.sweep/'plan.json').read_text());runs={};summary=[]
 for task in plan['tasks']:
  folder=args.sweep/f"run_{task['task']:02d}"
  if not (folder/'manifest.json').exists():raise SystemExit(f'Missing completed result: {folder}; no plot generated')
  m=json.loads((folder/'manifest.json').read_text())
  assert m.get('neuron_selection','neurostrike')=='neurostrike'
  assert m.get('prompt_date')==plan['prompt_date']
  for field in ['method','rank','activation','bias','seed']:
   assert m[field]==task[field],(folder,field)
  assert m['trainable_parameters']==task['parameters'] and m['optimizer_steps']==125
  with (folder/'heldout_per_sequence.csv').open() as f:rows=list(csv.DictReader(f))
  assert len(rows)==100
  with (folder/'heldout_completion_ce.csv').open() as f:
   ce={(r['method'],r['domain']):float(r['heldout_ce']) for r in csv.DictReader(f)}
  for domain in ['all','PKU-SafeRLHF','SciQ']:
   rr=[r for r in rows if domain=='all' or r['domain']==domain]
   for col,method in [('original_ce','original'),('adapted_ce',m['method'])]:
    val=sum(float(r[col])*int(r['completion_tokens']) for r in rr)/sum(int(r['completion_tokens']) for r in rr)
    assert math.isfinite(val) and abs(val-ce[method,domain])<1e-10
   summary.append(dict(task=task['task'],seed=m['seed'],method=m['method'],width=m['rank'],activation=m['activation'],parameters=m['trainable_parameters'],domain=domain,loss=ce[m['method'],domain],original_loss=ce['original',domain]))
  runs[task['task']]=m,rows,ce
 for seed in [42,43,44]:
  group=[runs[t['task']] for t in plan['tasks'] if t['seed']==seed]
  ref,refrows,refce=group[0]
  for m,rows,ce in group[1:]:
   for field in ['tokenized_sha256','train_prompt_sha256','val_prompt_sha256','model','placement','selection_threshold','epochs','lr','grad_accum','max_length','max_answer_tokens','dtype','train_n','val_n','code_sha256']:
    assert m[field]==ref[field],field
   assert m['network_config']['channels']==ref['network_config']['channels']
   for a,b in zip(rows,refrows):
    assert all(a[k]==b[k] for k in ['prompt_sha256','answer_sha256','domain','completion_tokens','sample_id'])
    assert abs(float(a['original_ce'])-float(b['original_ce']))<1e-5
 widths=[4,8,16,32,64]
 for measure in ['width','parameters']:
  fig,axes=plt.subplots(1,3,figsize=(16,5))
  for ax,domain in zip(axes,['all','PKU-SafeRLHF','SciQ']):
   for activation,color in [('tanh','#2467a8'),('identity','#cc701a')]:
    xs=[];ys=[]
    for width in widths:
     rr=[r for r in summary if r['domain']==domain and r['method']=='connected' and r['activation']==activation and r['width']==width]
     x=width if measure=='width' else rr[0]['parameters'];xs.append(x);ys.append(sum(r['loss'] for r in rr)/3)
     ax.scatter([x]*3,[r['loss'] for r in rr],color=color,s=12,alpha=.5)
    ax.plot(xs,ys,marker='o',color=color,label='tanh' if activation=='tanh' else 'no tanh')
   dr=[r for r in summary if r['domain']==domain and r['method']=='direct'];mean=sum(r['loss'] for r in dr)/3
   if measure=='width':ax.axhline(mean,color='#39815a',ls='--',label='Direct reference (no hidden nodes)')
   else:ax.scatter([106961], [mean],color='#39815a',marker='D',s=65,label='Direct reference')
   ax.set_title(domain);ax.set_xlabel('Hidden neurons per pair' if measure=='width' else 'Total trained parameters')
   ax.set_ylabel('Validation completion loss (lower is better)');ax.grid(alpha=.2)
   if measure=='width':ax.set_xticks(widths)
  axes[0].legend(fontsize=8)
  fig.suptitle('Width sweep: biases enabled, fixed selected-MLP insertion; mean of seeds 42–44')
  fig.text(.5,.025,'Dots show seed-specific results; not confidence intervals. Validation used for architecture selection, not an untouched final test.\nAll original weights frozen; parameter budgets differ. Completion loss is not a safety score.',ha='center',fontsize=10)
  fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(args.sweep/f'loss_vs_{measure}.png',dpi=180);plt.close(fig)
 with (args.sweep/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 print('Validated all 33 runs; generated width and parameter-count comparisons.')

if __name__=='__main__':main()
