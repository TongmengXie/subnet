"""Offline real-HF checks of the six planned tanh/bias/direct variants."""
import io
import itertools
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from adjacent_adapters import AdjacentAdapter

torch.set_num_threads(1)
torch.manual_seed(42)
model=LlamaForCausalLM(LlamaConfig(hidden_size=16,intermediate_size=24,
    num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=2,vocab_size=32)).eval()
for p in model.parameters(): p.requires_grad_(False)
weights={n:p.detach().clone() for n,p in model.named_parameters()}
channels={0:[0,2],1:[1,3,5],2:[0,4],3:[2,6]}
ids=torch.tensor([[1,2,3,4]])
base=model(ids,use_cache=False).logits.detach()
variants=[('connected',a,b) for a,b in itertools.product(['tanh','identity'],[False,True])]+[('direct','identity',b) for b in [False,True]]
for method,activation,bias in variants:
 a=AdjacentAdapter(model,channels,rank=2,method=method,activation=activation,bias=bias)
 expected=(16 if method=='direct' else 28)+(7+(0 if method=='direct' else 6) if bias else 0)
 assert a.parameter_count==expected
 with a.installed(model):
  torch.testing.assert_close(model(ids,use_cache=False).logits,base,rtol=0,atol=0)
  opt=torch.optim.AdamW(a.parameters(),lr=.02)
  for _ in range(3):
   opt.zero_grad();model(ids,labels=ids,use_cache=False).loss.backward()
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
   opt.step()
  assert all(p.grad.abs().sum()>0 for p in a.parameters())
  trained=model(ids,use_cache=False).logits.detach()
  assert not torch.equal(trained,base)
 assert all(torch.equal(p,weights[n]) and p.grad is None for n,p in model.named_parameters())
 block=a.blocks['0'];x=torch.tensor([[[-10.,20.]]])
 if method=='direct': expected_output=block.edges(x)
 else:
  h=block.encode(x);expected_output=block.decode(torch.tanh(h) if activation=='tanh' else h)
 torch.testing.assert_close(block(x),expected_output,rtol=0,atol=0)
 buf=io.BytesIO();torch.save(dict(config=a.config(),state_dict=a.state_dict()),buf);buf.seek(0)
 saved=torch.load(buf,weights_only=True);cfg=saved['config'];cfg.pop('overlap_policy')
 restored=AdjacentAdapter(model,**cfg);restored.load_state_dict(saved['state_dict'])
 with restored.installed(model):torch.testing.assert_close(model(ids,use_cache=False).logits,trained,rtol=0,atol=0)
 torch.testing.assert_close(model(ids,use_cache=False).logits,base,rtol=0,atol=0)
 print('PASS',method,activation,'bias='+str(bias),'identity, exact equation, trainable, frozen weights, serialized reload')
