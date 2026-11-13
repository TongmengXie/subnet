"""Offline CPU test: complement masking, unchanged selected values and cleanup."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from evaluate_neurostrike_only import selected_only

torch.set_num_threads(1)
torch.manual_seed(42)
model = LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=24,
    num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2, vocab_size=32)).eval()
for p in model.parameters():
    p.requires_grad_(False)
ids = torch.tensor([[1, 2, 3, 4]])
base = model(ids, use_cache=False).logits.detach()
weights = {n:p.clone() for n,p in model.named_parameters()}
channels = {i:[1,3,5] for i in range(3)}
raw, masked, handles = {}, {}, []
for i, layer in enumerate(model.model.layers):
    def capture(_mod, args, i=i):
        raw[i] = args[0].clone()
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
try:
    with selected_only(model, channels):
        for i, layer in enumerate(model.model.layers):
            def capture(_mod, args, i=i):
                masked[i] = args[0].clone()
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
        changed = model(ids, use_cache=False).logits.detach()
        for i in channels:
            torch.testing.assert_close(raw[i][...,channels[i]], masked[i][...,channels[i]], rtol=0, atol=0)
            complement = [j for j in range(24) if j not in channels[i]]
            assert torch.count_nonzero(masked[i][...,complement]) == 0
        assert not torch.equal(base, changed)
finally:
    for h in handles:
        h.remove()
torch.testing.assert_close(model(ids, use_cache=False).logits, base, rtol=0, atol=0)
with selected_only(model, {i:list(range(24)) for i in range(3)}):
    torch.testing.assert_close(model(ids, use_cache=False).logits, base, rtol=0, atol=0)
try:
    with selected_only(model, channels):
        raise RuntimeError('intentional cleanup check')
except RuntimeError:
    pass
assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.model.layers)
assert all(torch.equal(p,weights[n]) for n,p in model.named_parameters())
print('PASS actual HF CPU: selected values retained, complement zeroed, all-selected identity, frozen weights, exception cleanup')
