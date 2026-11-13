"""Offline real-HF CPU tests for cascading adapters and independent controls."""
import io
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from adjacent_adapters import AdjacentAdapter, consecutive_pairs
from train_cross_layer import evaluate


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=24,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        vocab_size=32, attention_dropout=0.)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    channels = {0: [0, 2], 1: [1, 3, 5], 2: [0, 4], 3: [2, 6]}
    assert consecutive_pairs({0:[0], 2:[1], 3:[2]}) == [(2,3)]
    ids = torch.tensor([[1, 2, 3, 4]])
    base = model(ids, use_cache=False).logits.detach()
    for method in ('connected', 'independent'):
        adapter = AdjacentAdapter(model, channels, rank=4, method=method)
        assert adapter.parameter_count == 56
        with adapter.installed(model):
            torch.testing.assert_close(model(ids, use_cache=False).logits, base, rtol=0, atol=0)
            opt = torch.optim.AdamW(adapter.parameters(), lr=.02)
            for _ in range(3):
                opt.zero_grad()
                model(ids, labels=ids, use_cache=False).loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
                opt.step()
            trained = model(ids, use_cache=False).logits.detach()
            assert not torch.equal(trained, base)
            if method == 'connected':
                # Incoming injection must be visible to outgoing capture.
                seen = {}
                block = adapter.blocks['1']
                capture = block._capture_source
                def check(module, args):
                    seen['captured'] = args[0].detach().clone()
                    return capture(module, args)
                block._capture_source = check
                handle = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(
                    lambda module, args: seen.update(actual=args[0].detach().clone()))
                model(ids, use_cache=False)
                handle.remove()
                block._capture_source = capture
                torch.testing.assert_close(seen['captured'], seen['actual'], rtol=0, atol=0)
                # Upstream correction affects downstream correction through a live graph.
                with torch.no_grad():
                    for b in adapter.blocks.values():
                        b.encode.weight.fill_(.1)
                        b.decode.weight.fill_(.1)
                adapter.zero_grad()
                model(ids, use_cache=False)
                # Caches are consumed after each forward.
                assert all(b._cache is None for b in adapter.blocks.values())
                outputs = {}
                handle = adapter.blocks['2'].decode.register_forward_hook(
                    lambda module, args, output: outputs.update(last=output))
                model(ids, use_cache=False)
                handle.remove()
                upstream = torch.autograd.grad(outputs['last'].sum(),
                    adapter.blocks['0'].decode.weight)[0]
                assert upstream.abs().sum() > 0
                model(ids, labels=ids, use_cache=False).loss.backward()
                assert adapter.blocks['0'].decode.weight.grad.abs().sum() > 0
            else:
                b = adapter.blocks['1']
                x = torch.randn(1,1,3, requires_grad=True)
                values = b.b * torch.tanh(b.a*x.index_select(-1,b.unit_channel))
                delta = torch.zeros_like(x).index_add(-1,b.unit_channel,values)
                grad = torch.autograd.grad(delta[...,0].sum(),x)[0]
                assert torch.count_nonzero(grad[...,1:]) == 0
        assert all(torch.equal(p,weights[n]) and p.grad is None for n,p in model.named_parameters())
        torch.testing.assert_close(model(ids, use_cache=False).logits,base,rtol=0,atol=0)
        buf = io.BytesIO()
        torch.save(adapter.state_dict(),buf); buf.seek(0)
        restored = AdjacentAdapter(model,channels,rank=4,method=method)
        restored.load_state_dict(torch.load(buf,weights_only=True))
        with adapter.installed(model):
            expected = model(ids,use_cache=False).logits.detach()
        with restored.installed(model):
            torch.testing.assert_close(model(ids,use_cache=False).logits,expected,rtol=0,atol=0)
        try:
            with adapter.installed(model):
                raise ValueError('deliberate cleanup test')
        except ValueError:
            pass
        with adapter.installed(model):
            model(ids,use_cache=False)
        print('PASS',method, 'budget, identity, training, overlap/separability, reload, cleanup')
    rows=[]
    samples=[(ids,ids,torch.ones_like(ids),'test')]
    metrics=evaluate(model,samples,'cpu',rows=rows)
    assert rows[0]['completion_tokens']==3 and rows[0]['ce']==metrics['all']
    print('PASS per-sequence token counts and aggregation')

if __name__ == '__main__':
    main()
