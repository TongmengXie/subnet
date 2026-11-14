"""Execute one predeclared sweep task using its snapshotted training code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ap=argparse.ArgumentParser()
ap.add_argument('--sweep',type=Path,required=True)
ap.add_argument('--task',type=int,required=True)
a=ap.parse_args()
plan=json.loads((a.sweep/'plan.json').read_text())
task=plan['tasks'][a.task]
assert task['task']==a.task
cmd=[sys.executable,'-u','train_cross_layer.py','--repo',str(Path(plan['project'])/'NeuroStrike'),
     '--scope','all-adjacent','--method',task['method'],'--rank',str(task['rank'] or 8),
     '--activation',task['activation'],'--bias','--placement','selected_mlp',
     '--seed',str(task['seed']),'--train-n','500','--val-n','100',
     '--out',str(a.sweep/f'run_{a.task:02d}')]
print(json.dumps({'task':task,'command':cmd}),flush=True)
subprocess.run(cmd,cwd=a.sweep/'code',check=True)
