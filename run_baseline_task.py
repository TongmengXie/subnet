"""Run a declared random-neuron or full-model baseline from a code snapshot."""
import argparse,csv,json,subprocess,sys,hashlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--kind',choices=['random','full'],required=True)
ap.add_argument('--task',type=int,default=0);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
plan=json.loads((a.root/'plan.json').read_text());sweep=Path(plan['sweep']);sp=json.loads((sweep/'plan.json').read_text())
seed=42 if a.smoke else [42,43,44][a.task//3 if a.kind=='random' else a.task]
out=a.root/(a.kind+'_smoke' if a.smoke else f'{a.kind}_{a.task:02d}')
ref=Path(plan['references'][str(seed)])
if a.kind=='random':
 if not (a.root/'architecture_choice.json').exists():
  subprocess.run([sys.executable,'plot_width_sweep.py','--sweep',str(sweep)],check=True)
 groups={}
 for t in sp['tasks']:
  folder=sweep/f"run_{t['task']:02d}";m=json.loads((folder/'manifest.json').read_text())
  assert m['prompt_date']=='13 Nov 2026'
  assert m['method']==t['method'] and m['rank']==t['rank'] and m['activation']==t['activation'] and m['bias']
  assert m['tokenized_sha256']==json.loads(Path(plan['references'][str(t['seed'])]).read_text())['tokenized_sha256']
  with (folder/'heldout_completion_ce.csv').open() as f:ce=next(float(r['heldout_ce']) for r in csv.DictReader(f) if r['method']==t['method'] and r['domain']=='all')
  groups.setdefault((t['method'],t['rank'],t['activation']),[]).append((ce,t['parameters']))
 assert len(groups)==11 and all(len(v)==3 for v in groups.values())
 key=min(groups,key=lambda k:(sum(x[0] for x in groups[k])/3,groups[k][0][1]))
 choice=dict(method=key[0],rank=key[1],activation=key[2],bias=True,selection_metric='Mean validation overall completion CE across seeds; ties smaller budget',mean_validation_ce=sum(x[0] for x in groups[key])/3)
 choicefile=a.root/'architecture_choice.json'
 try:
  with choicefile.open('x') as f:json.dump(choice,f,indent=2)
 except FileExistsError:assert json.loads(choicefile.read_text())==choice
 mask_seed=1001 if a.smoke else [1001,1002,1003][a.task%3]
 cmd=[sys.executable,'-u','train_cross_layer.py','--repo',str(Path(plan['project'])/'NeuroStrike'),'--scope','all-adjacent','--method',key[0],'--rank',str(key[1] or 8),'--activation',key[2],'--bias','--neuron-selection','random','--selection-seed',str(mask_seed),'--seed',str(seed),'--prompt-date','13 Nov 2026','--out',str(out)]
else:
 cmd=[sys.executable,'-u','train_full_model.py','--repo',str(Path(plan['project'])/'NeuroStrike'),'--reference',str(ref),'--seed',str(seed),'--lr','2e-5','--out',str(out)]
if a.smoke:cmd.append('--smoke')
print(json.dumps({'kind':a.kind,'task':a.task,'command':cmd}),flush=True)
subprocess.run(cmd,cwd=a.root/'code',check=True)
if not a.smoke:
 m=json.loads((out/'manifest.json').read_text());r=json.loads(ref.read_text());assert m['tokenized_sha256']==r['tokenized_sha256']
 with (out/'heldout_per_sequence.csv').open() as f:rows=list(csv.DictReader(f))
 with (ref.parent/'heldout_per_sequence.csv').open() as f:old=list(csv.DictReader(f))
 assert len(rows)==len(old)==100
 for x,y in zip(rows,old):
  assert all(x[k]==y[k] for k in ['prompt_sha256','answer_sha256','completion_tokens','domain'])
  assert abs(float(x['original_ce'])-float(y['original_ce']))<1e-5
 print('PASS matched data and original baseline',flush=True)
