"""Rebuild the Viking notebook from the v2 Colab notebook (run locally, not on Viking)."""
from pathlib import Path
import os, nbformat
root=Path(__file__).resolve().parent
source=Path(os.environ.get('SOURCE_NOTEBOOK',root.parent/'NeuroStrike_subnetwork_localization_v2_PKU_overlay.ipynb'))
n=nbformat.read(source,as_version=4)

n.cells[3].source='''# Viking HPC initialization: repository is cloned once by setup_viking.sh.
import os, sys, subprocess
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg', force=True)
REPO_ROOT = Path(os.environ.get('NEUROSTRIKE_REPO', Path.cwd().parent)).expanduser().resolve()
WHITE_BOX = REPO_ROOT / 'white_box'
assert (WHITE_BOX / 'util_model.py').is_file(), f'NeuroStrike white_box missing: {WHITE_BOX}'
assert (REPO_ROOT / 'pre_computed_sn' / 'weights_Llama-3.2-1B-Instruct.p').is_file(), (
    'Precomputed safety-neuron file missing; check NeuroStrike checkout')
os.chdir(WHITE_BOX)
sys.path.insert(0, str(WHITE_BOX))
sys.path.insert(0, str(REPO_ROOT))
print('Working directory:', os.getcwd())
print('NeuroStrike commit:', subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip())
'''
n.cells[5].source='''# Authentication is handled by `hf auth login` on Viking (or HF_TOKEN in the job environment).
# Never place access tokens directly in the notebook or the Slurm script.
import os
from huggingface_hub import get_token
if not get_token():
    raise RuntimeError('Hugging Face token unavailable. Run `hf auth login` with the same HF_HOME used by the job; '
                       'request Llama-3.2 access from Meta if needed.')
print('Hugging Face authentication available (token not displayed).')
'''
n.cells[9].source=n.cells[9].source.replace('model_id = 0',"model_id = int(os.environ.get('MODEL_ID', '0'))",1)
for i in [15,17,19]:
 n.cells[i]=nbformat.v4.new_markdown_cell('**Viking:** skipped the Colab-only demonstration step. The original pretrained model remains loaded once; A4 still evaluates NeuroStrike pruning as a separate variant.')

s=n.cells[25].source
s=s.replace('SFT_SEED = 42',"SFT_SEED = int(os.environ.get('SFT_SEED', '42'))",1)
for name,default in [('SFT_TRAIN_N','500'),('SFT_VAL_N','100'),('SFT_EPOCHS','1'),('SFT_GRAD_ACCUM','4'),('SFT_VAL_EVERY','50'),('SFT_MAX_LENGTH','256'),('SFT_MAX_COMPLETION_TOKENS','128')]:
 import re
 s,c=re.subn(r'^'+name+r'\s*=\s*\d+[^\n]*$',f"{name} = int(os.environ.get('{name}', '{default}'))",s,count=1,flags=re.M)
 assert c==1,(name,c)
s=s.replace('SFT_SCIQ_SHARE = 0.20',"SFT_SCIQ_SHARE = float(os.environ.get('SFT_SCIQ_SHARE', '0.20'))",1)
s=s.replace('SFT_LR = 2e-4',"SFT_LR = float(os.environ.get('SFT_LR', '2e-4'))",1)
s=s.replace('# For Colab, GPU memory is dominated by backward through the frozen 1B model.', '# On Viking, GPU memory is dominated by backward through the frozen 1B model.')
s='import os\n'+s
n.cells[25].source=s

# Save trained adapters before long generation/evaluation: a job interruption should not lose training.
checkpoint_code='''# A3b. Save checkpoints immediately after training, BEFORE StrongREJECT generation.
import os, json
from pathlib import Path
EARLY_OUT = Path(os.environ.get('SUBNETWORK_RESULTS_DIR', 'subnetwork_localization_pilot')).expanduser().resolve()
EARLY_OUT.mkdir(parents=True, exist_ok=True)
for _kind, _adapter in [('bottleneck', bottleneck_adapter), ('independent', independent_adapter)]:
    torch.save({'architecture':'gated-mlp-selected-channel-residual', 'kind':_kind,
                'rank':RANK, 'selected_channels':selected_channels,
                'state_dict':{k:v.detach().cpu() for k,v in _adapter.state_dict().items()}},
               EARLY_OUT/f'{_kind}_adapter.pt')
training_history.to_csv(EARLY_OUT/'training_history.csv',index=False)
periodic_validation_history.to_csv(EARLY_OUT/'periodic_validation_ce.csv',index=False)
heldout_ce_table.to_csv(EARLY_OUT/'heldout_completion_ce.csv',index=False)
with (EARLY_OUT/'adaptation_dataset_manifest.json').open('w') as _f:
    json.dump(adaptation_dataset_manifest,_f,indent=2)
print('Training checkpoints safely saved:',EARLY_OUT)
'''
n.cells.insert(26,nbformat.v4.new_code_cell(checkpoint_code))
# Shift original cell 27 -> 28, 28 -> 29, 35 -> 36
s=n.cells[28].source
s=s.replace('EVAL_HARMFUL_N = 64        # Fixed sampled subset for EACH model variant.',
'''_eval_harmful_n = os.environ.get('EVAL_HARMFUL_N', '64')
EVAL_HARMFUL_N = None if _eval_harmful_n.lower() == 'none' else int(_eval_harmful_n)''',1)
s=s.replace('EVAL_SEED = 42',"EVAL_SEED = int(os.environ.get('EVAL_SEED', '42'))",1)
s=s.replace('EVAL_MAX_NEW_TOKENS = 128 # Full NeuroStrike script sets 512 for non-reasoning models.',
            "EVAL_MAX_NEW_TOKENS = int(os.environ.get('EVAL_MAX_NEW_TOKENS', '128'))",1)
s=s.replace('EVAL_LOGIT_N = 12         # Lightweight sensitivity check, beyond string-level refusal.',
            "EVAL_LOGIT_N = int(os.environ.get('EVAL_LOGIT_N', '12'))",1)
s=s.replace("EVAL_OUT = Path('subnetwork_localization_pilot')",
            "EVAL_OUT = Path(os.environ.get('SUBNETWORK_RESULTS_DIR', 'subnetwork_localization_pilot')).expanduser().resolve()",1)
s='import os\n'+s
n.cells[28].source=s
s=n.cells[29].source.replace("RESULT_DIR = Path('subnetwork_localization_pilot')",
 "RESULT_DIR = Path(os.environ.get('SUBNETWORK_RESULTS_DIR', 'subnetwork_localization_pilot')).expanduser().resolve()",1)
s='import os\n'+s
n.cells[29].source=s
s=n.cells[36].source.replace('fig_overview.show()',"print('Overview HTML saved after figure construction')",1).replace('fig.show()',"print('Layer overlay HTML saved after figure construction')",1)
n.cells[36].source=s
n.cells.insert(0,nbformat.v4.new_markdown_cell('''# NeuroStrike subnetwork localization — Viking HPC batch edition

Run with `viking_gpu.sbatch` and `run_viking.py` from the accompanying package. The original model is loaded once. Precomputed safety-neuron weights come from the NeuroStrike repository; A3 trains the bottleneck and independent controls; A3b saves checkpoints before A4 evaluation; A6/A7 save PNG and HTML figures. Configure smoke/full sample counts using environment variables rather than editing cells. Do not run compute-intensive cells on Viking login nodes.'''))
for cell in n.cells:
 if cell.cell_type=='code':
  cell.outputs=[]
  cell.execution_count=None
n.metadata['kernelspec']={'display_name':'Python (NeuroStrike Viking)','language':'python','name':'neurostrike-viking'}
n.metadata['language_info']={'name':'python','version':'3.11'}
out=root/'NeuroStrike_subnetwork_localization_viking.ipynb'
nbformat.validate(n)
nbformat.write(n,out)
print(out,len(n.cells),'cells')
