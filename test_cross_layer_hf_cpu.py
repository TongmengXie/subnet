"""Offline CPU regression against installed HF Llama; no pretrained weights."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from cross_layer_adapter import CrossLayerAdapter, PLACEMENTS

def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        vocab_size=32, attention_dropout=0.)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    ids = torch.tensor([[1, 2, 3, 4]])
    base = model(ids, use_cache=False).logits.detach()
    for placement in PLACEMENTS:
        adapter = CrossLayerAdapter(model, source_layer=0, target_layer=1,
            source_channels=[0, 2], target_channels=[1, 3], rank=2, placement=placement)
        with adapter.installed(model):
            torch.testing.assert_close(model(ids, use_cache=False).logits, base, rtol=0, atol=0)
            opt = torch.optim.AdamW(adapter.parameters(), lr=.01)
            for _ in range(3):
                opt.zero_grad()
                loss = model(ids, labels=ids, use_cache=False).loss
                loss.backward()
                assert torch.isfinite(adapter.decode.weight.grad).all()
                opt.step()
            assert adapter.encode.weight.grad.abs().sum() > 0
            assert not torch.equal(model(ids, use_cache=False).logits, base)
            assert adapter._cache is None
        assert all(torch.equal(p, original[n]) and p.grad is None for n, p in model.named_parameters())
        torch.testing.assert_close(model(ids, use_cache=False).logits, base, rtol=0, atol=0)
        print('PASS real HF CPU:', placement)
if __name__ == '__main__':
    main()
