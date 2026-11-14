"""Full-model supervised fine-tuning baseline, unchanged architecture, cached assets."""
import argparse,csv,hashlib,json,math,random,sys,time
from functools import partial
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from train_cross_layer import get_examples,tokenize_pairs,evaluate,normalized,write_csv


def fingerprint(samples):
 return hashlib.sha256(json.dumps([(i.tolist(),l.tolist(),d) for i,l,_,d in samples]).encode()).hexdigest()


def optimize(model,samples,device,lr,accum,seed,amp=False):
 """All original weights trainable; one epoch in the matched seeded order."""
 params=list(model.parameters())
 assert all(p.requires_grad for p in params)
 opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.)
 history=[]
 order=np.random.default_rng(seed).permutation(len(samples))
 for offset in range(0,len(order),accum):
  opt.zero_grad(set_to_none=True);batch=order[offset:offset+accum];losses=[]
  for idx in batch:
   ids,labels,mask,_=samples[int(idx)]
   with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
    loss=model(input_ids=ids.to(device),labels=labels.to(device),attention_mask=mask.to(device),use_cache=False).loss
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite full-model loss')
   (loss/len(batch)).backward();losses.append(float(loss.detach()))
  norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
  if not math.isfinite(norm):raise RuntimeError('Nonfinite gradient norm')
  opt.step();history.append(dict(step=len(history)+1,train_ce=float(np.mean(losses)),grad_norm=norm))
 return history


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--lr',type=float,default=2e-5);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists() and any(a.out.iterdir()):raise FileExistsError(a.out)
 a.out.mkdir(parents=True,exist_ok=True)
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16-capable GPU required')
 torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 ref=json.loads(a.reference.read_text())
 cfg=argparse.Namespace(**json.loads((a.reference.parent/'run_config.json').read_text()))
 cfg.seed=a.seed
 if a.smoke:cfg.train_n,cfg.val_n=32,16
 sys.path.insert(0,str(a.repo/'white_box'));import util_model
 tok=AutoTokenizer.from_pretrained(cfg.model,local_files_only=True)
 tok.apply_chat_template=partial(tok.apply_chat_template,date_string='13 Nov 2026')
 tr,va=get_examples(cfg)
 prompt=lambda q:util_model.construct_prompt(tok,cfg.model.split('/')[-1],[q])[0]
 train=tokenize_pairs(tr,tok,cfg,prompt);val=tokenize_pairs(va,tok,cfg,prompt)
 hashes={'train':fingerprint(train),'val':fingerprint(val)}
 if not a.smoke:assert hashes==ref['tokenized_sha256']
 torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 model=AutoModelForCausalLM.from_pretrained(cfg.model,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
 model.config.use_cache=False;model.gradient_checkpointing_disable()
 before=[];baseline=evaluate(model,val,'cuda',rows=before)
 if not a.smoke:
  old=list(csv.DictReader((a.reference.parent/'heldout_per_sequence.csv').open()))
  assert len(old)==len(before)
  assert max(abs(b['ce']-float(r['original_ce'])) for b,r in zip(before,old))<1e-5
 # Start from the exact BF16 reference weights, represented by FP32 masters.
 model.float().train()
 for p in model.parameters():p.requires_grad_(True)
 count=sum(p.numel() for p in model.parameters())
 print('Training all original parameters:',count,'lr',a.lr,flush=True)
 probe_ids=train[0][0][0,:16].to('cuda')
 head_before=model.lm_head.weight.detach()[probe_ids,:16].clone();embed_before=model.get_input_embeddings().weight.detach()[probe_ids,:16].clone()
 history=optimize(model,train,'cuda',a.lr,cfg.grad_accum,a.seed,amp=True)
 assert not torch.equal(head_before,model.lm_head.weight[probe_ids,:16]) and not torch.equal(embed_before,model.get_input_embeddings().weight[probe_ids,:16])
 del head_before,embed_before
 model.zero_grad(set_to_none=True);model.to(dtype=torch.bfloat16).eval()
 afterrows=[];after=evaluate(model,val,'cuda',rows=afterrows)
 assert all(math.isfinite(v) for v in after.values())
 write_csv(a.out/'training_history.csv',history,list(history[0]))
 results=[dict(method=m,domain=d,heldout_ce=v) for m,scores in [('original',baseline),('full_model',after)] for d,v in scores.items()]
 write_csv(a.out/'heldout_completion_ce.csv',results,['method','domain','heldout_ce'])
 pairs=[dict(sample_id=b['sample_id'],domain=b['domain'],completion_tokens=b['completion_tokens'],
  prompt_sha256=hashlib.sha256(normalized(q).encode()).hexdigest(),answer_sha256=hashlib.sha256(ans.encode()).hexdigest(),original_ce=b['ce'],adapted_ce=c['ce'])
  for b,c,(q,ans,_) in zip(before,afterrows,va)]
 write_csv(a.out/'heldout_per_sequence.csv',pairs,list(pairs[0]))
 model.save_pretrained(a.out/'model');tok.save_pretrained(a.out/'model')
 memory=dict(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,total_device_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start)
 meta=dict(method='full_model',unchanged_architecture=True,trainable_parameters=count,seed=a.seed,lr=a.lr,train_n=len(train),val_n=len(val),optimizer_steps=len(history),grad_accum=cfg.grad_accum,tokenized_sha256=hashes,prompt_date='13 Nov 2026',reference=str(a.reference),training_precision='FP32 parameters and AdamW states; BF16 autocast',evaluation_precision='BF16 as reference',memory=memory,code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['train_full_model.py','train_cross_layer.py']},notes='Provisional LR, not independently optimized. Validation is not an untouched final test.')
 (a.out/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(dict(results=results,memory=memory)),flush=True)

if __name__=='__main__':main()
