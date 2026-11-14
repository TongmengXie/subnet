"""Offline CPU control checks: matched counts/budget, seed independence, frozen model."""
import io
import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from neuron_selection import count_matched_random_neurons
from adjacent_adapters import AdjacentAdapter

torch.set_num_threads(1)
reference={0:[0,2],1:[1,3,5],2:[0,4],3:[2,6]}
dimensions={i:24 for i in reference}
np.random.seed(42);torch.manual_seed(42)
np_before=np.random.get_state();torch_before=torch.get_rng_state().clone()
a=count_matched_random_neurons(reference,dimensions,seed=1001)
assert np.array_equal(np_before[1],np.random.get_state()[1])
assert torch.equal(torch_before,torch.get_rng_state())
assert a==count_matched_random_neurons(reference,dimensions,seed=1001)
assert a!=count_matched_random_neurons(reference,dimensions,seed=1002)
assert {k:len(v) for k,v in a.items()}=={k:len(v) for k,v in reference.items()}
assert all(ids==sorted(set(ids)) and min(ids)>=0 and max(ids)<24 for ids in a.values())
model=LlamaForCausalLM(LlamaConfig(hidden_size=16,intermediate_size=24,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=2,vocab_size=32)).eval()
for p in model.parameters():p.requires_grad_(False)
original={n:p.detach().clone() for n,p in model.named_parameters()}
ids=torch.tensor([[1,2,3,4]]);base=model(ids,use_cache=False).logits.detach()
for method in ['connected','independent','direct']:
 torch.manual_seed(17);safe=AdjacentAdapter(model,reference,rank=4,method=method,bias=(method!='independent'))
 torch.manual_seed(17);random=AdjacentAdapter(model,a,rank=4,method=method,bias=(method!='independent'))
 assert safe.parameter_count==random.parameter_count
 for (n,p),(m,q) in zip(safe.named_parameters(),random.named_parameters()):
  assert n==m and torch.equal(p,q)  # same initialization, only neuron IDs differ
 with random.installed(model):
  torch.testing.assert_close(model(ids,use_cache=False).logits,base,rtol=0,atol=0)
  opt=torch.optim.AdamW(random.parameters(),lr=.01)
  for _ in range(3):
   opt.zero_grad();model(ids,labels=ids,use_cache=False).loss.backward();opt.step()
  changed=model(ids,use_cache=False).logits.detach()
  assert not torch.equal(changed,base)
 assert all(torch.equal(p,original[n]) and p.grad is None for n,p in model.named_parameters())
 buf=io.BytesIO();torch.save(dict(config=random.config(),state_dict=random.state_dict()),buf);buf.seek(0)
 saved=torch.load(buf,weights_only=True);cfg=saved['config'];cfg.pop('overlap_policy')
 copy=AdjacentAdapter(model,**cfg);copy.load_state_dict(saved['state_dict'])
 with copy.installed(model):torch.testing.assert_close(model(ids,use_cache=False).logits,changed,rtol=0,atol=0)
 print('PASS random',method,'counts/budget/initialization matched; independent mask seed; identity, training, frozen backbone, reload')
