"""Offline CPU audit of exact train/validation samples and token counts; no weights."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from functools import partial
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from train_cross_layer import get_examples, tokenize_pairs, normalized


def main():
 root=Path(__file__).resolve().parent
 out=root/'results/data_audit';out.mkdir(exist_ok=True)
 sys.path.insert(0,str(root/'NeuroStrike/white_box'));import util_model
 model='meta-llama/Llama-3.2-1B-Instruct'
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
 cfg=AutoConfig.from_pretrained(model,local_files_only=True)
 # Historical full runs were formatted on 13 Nov 2026. Explicitly reconstruct
 # the date-dependent Llama chat template and verify both saved data hashes.
 tok.apply_chat_template=partial(tok.apply_chat_template,date_string='13 Nov 2026')
 def prompt(q):return util_model.construct_prompt(tok,model.split('/')[-1],[q])[0]
 rows=[];seed_sets={};checks=[]
 for seed in [42,43,44]:
  p=next(p for p in (root/'results').glob(f'all-adjacent_connected_selected_mlp_seed{seed}_*/manifest.json') if json.loads(p.read_text())['train_n']==500 and not json.loads(p.read_text()).get('bias',False))
  meta=json.loads(p.read_text());a=argparse.Namespace(**json.loads((p.parent/'run_config.json').read_text()))
  tr,va=get_examples(a)
  sets={name:{normalized(q) for q,_,_ in records} for name,records in [('train',tr),('validation',va)]}
  assert not sets['train']&sets['validation'];seed_sets[seed]=sets
  for split,records in [('train',tr),('val',va)]:
   ts=tokenize_pairs(records,tok,a,prompt)
   fingerprint=hashlib.sha256(json.dumps([(ids.tolist(),labels.tolist(),domain) for ids,labels,_,domain in ts]).encode()).hexdigest()
   assert fingerprint==meta['tokenized_sha256'][split]
   checks.append(dict(seed=seed,split=split,tokenized_hash_matched=True))
   for domain in ['all','PKU-SafeRLHF','SciQ']:
    subset=[(r,t) for r,t in zip(records,ts) if domain=='all' or r[2]==domain]
    completions=[int((t[1][:,1:]!=-100).sum()) for _,t in subset]
    lengths=[t[0].shape[-1] for _,t in subset]
    answer_truncated=sum(len(tok(r[1],add_special_tokens=False)['input_ids'])>a.max_answer_tokens for r,t in subset)
    prompt_truncated=0
    for r,t in subset:
     prefix=len(tok(prompt(r[0]),add_special_tokens=False)['input_ids'])
     if prefix+int((t[1]!=-100).sum())>a.max_length:prompt_truncated+=1
    rows.append(dict(seed=seed,split='validation' if split=='val' else split,domain=domain,examples=len(subset),
      completion_tokens=sum(completions),mean_completion_tokens=float(np.mean(completions)),
      mean_total_tokens=float(np.mean(lengths)),max_total_tokens=max(lengths),
      answer_truncated=answer_truncated,prompt_truncated=prompt_truncated))
 dataset_sizes={}
 for dataset,splits in [('PKU-Alignment/PKU-SafeRLHF',['train','test']),('allenai/sciq',['train','validation','test'])]:
  dataset_sizes[dataset]={s:len(load_dataset(dataset,split=s)) for s in splits}
 summary=dict(model=model,prompt_date='13 Nov 2026',model_config=dict(hidden_size=cfg.hidden_size,intermediate_size=cfg.intermediate_size,num_hidden_layers=cfg.num_hidden_layers),
  source_split_sizes=dataset_sizes,rows=rows,checks=checks,exact_prompt_overlap={str(s):len(d['train']&d['validation']) for s,d in seed_sets.items()},
  cross_seed_train_validation_overlap={f'train{a}_val{b}':len(seed_sets[a]['train']&seed_sets[b]['validation']) for a in seed_sets for b in seed_sets},
  unique_prompts_across_seeds={split:len(set.union(*(d[split] for d in seed_sets.values()))) for split in ['train','validation']},
  notes=['Exact normalized prompt overlap only; no semantic-overlap or pretraining-contamination claim.',
   '100 held-out examples serve as validation for architecture selection, not an untouched final test.',
   'Only completion tokens enter CE. Training averages per-example mean losses; reported evaluation CE is token-weighted.'])
 (out/'data_stats.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (out/'data_stats.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
