"""CPU integration for full-model updates with unchanged HF architecture."""
import tempfile,torch
from transformers import LlamaConfig,LlamaForCausalLM
from train_full_model import optimize

torch.set_num_threads(1);torch.manual_seed(42)
m=LlamaForCausalLM(LlamaConfig(hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=32)).eval()
ids=torch.tensor([[1,2,3,4]]);samples=[(ids,ids,torch.ones_like(ids),'test')]*4
before={n:p.detach().clone() for n,p in m.named_parameters()};names=set(before)
history=optimize(m,samples,'cpu',.001,2,42)
assert len(history)==2 and set(dict(m.named_parameters()))==names
assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
assert not torch.equal(before['lm_head.weight'],m.lm_head.weight)
assert not torch.equal(before['model.embed_tokens.weight'],m.model.embed_tokens.weight)
assert not torch.equal(before['model.layers.0.self_attn.q_proj.weight'],m.model.layers[0].self_attn.q_proj.weight)
m.eval();expected=m(ids,use_cache=False).logits.detach()
with tempfile.TemporaryDirectory() as tmp:
 m.save_pretrained(tmp);restored=LlamaForCausalLM.from_pretrained(tmp).eval()
 torch.testing.assert_close(restored(ids,use_cache=False).logits,expected,rtol=0,atol=0)
print('PASS full-model: unchanged parameter structure, all trainable, gradients, embedding/attention/head updates, save/reload')
