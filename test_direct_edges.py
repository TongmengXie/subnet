"""Offline CPU integration tests of direct full connections on actual HF Llama."""
import io
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from adjacent_adapters import AdjacentAdapter


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=24,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        vocab_size=32, attention_dropout=0.)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    original = {n:p.detach().clone() for n,p in model.named_parameters()}
    channels = {0:[0,2], 1:[1,3,5], 2:[0,4], 3:[2,6]}
    ids = torch.tensor([[1,2,3,4]])
    baseline = model(ids, use_cache=False).logits.detach()
    adapter = AdjacentAdapter(model, channels, method='direct')
    assert adapter.parameter_count == 2*3 + 3*2 + 2*2 == 16
    assert adapter.rank is None
    for b in adapter.blocks.values():
        assert len(list(b.parameters())) == 1
        assert sum(isinstance(m, nn.Linear) for m in b.modules()) == 1
        assert not hasattr(b, 'encode') and not hasattr(b, 'decode')
        assert b.edges.bias is None
    # Direct output must be precisely W*x, including negative and large inputs.
    b = adapter.blocks['0']
    with torch.no_grad():
        b.edges.weight.copy_(torch.tensor([[1.,2.],[3.,4.],[5.,6.]]))
    x = torch.tensor([[[-10.,20.]]], requires_grad=True)
    torch.testing.assert_close(b(x), x @ b.edges.weight.T, rtol=0, atol=0)
    with torch.no_grad():
        b.edges.weight.zero_()
    with adapter.installed(model):
        torch.testing.assert_close(model(ids, use_cache=False).logits, baseline, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.02)
        for _ in range(3):
            optimizer.zero_grad()
            model(ids, labels=ids, use_cache=False).loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       and p.grad.abs().sum() > 0 for p in adapter.parameters())
            optimizer.step()
        assert not torch.equal(model(ids, use_cache=False).logits, baseline)
        assert all(b._cache is None for b in adapter.blocks.values())
        # Verify that outgoing capture sees the incoming adjustment; unselected
        # neurons at this hook must remain exactly unchanged.
        seen = {}
        def before(_m, args):
            seen['before'] = args[0].detach().clone()
        # prepend puts the observer before the already-installed adapter hook.
        h1 = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(before, prepend=True)
        def after(_m, args):
            seen['after'] = args[0].detach().clone()
            seen['captured'] = adapter.blocks['1']._cache.detach().clone()
        h2 = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(after)
        model(ids, use_cache=False)
        h1.remove(); h2.remove()
        torch.testing.assert_close(seen['captured'], seen['after'][...,channels[1]], rtol=0, atol=0)
        complement = [i for i in range(24) if i not in channels[1]]
        torch.testing.assert_close(seen['before'][...,complement], seen['after'][...,complement], rtol=0, atol=0)
        assert not torch.equal(seen['before'], seen['after'])
        # Later direct corrections retain a gradient to earlier learned edges.
        with torch.no_grad():
            for b in adapter.blocks.values():
                b.edges.weight.fill_(.1)
        captured = {}
        def capture(_m, _args, output):
            captured['last'] = output
        h = adapter.blocks['2'].edges.register_forward_hook(capture)
        model(ids, use_cache=False); h.remove()
        grad = torch.autograd.grad(captured['last'].sum(), adapter.blocks['0'].edges.weight)[0]
        assert grad.abs().sum() > 0
        expected = model(ids, use_cache=False).logits.detach()
    assert all(torch.equal(p,original[n]) and p.grad is None for n,p in model.named_parameters())
    torch.testing.assert_close(model(ids, use_cache=False).logits, baseline, rtol=0, atol=0)
    buf = io.BytesIO()
    torch.save(dict(config=adapter.config(), state_dict=adapter.state_dict()),buf); buf.seek(0)
    saved = torch.load(buf, weights_only=True)
    cfg = saved['config']; cfg.pop('overlap_policy')
    restored = AdjacentAdapter(model, **cfg)
    restored.load_state_dict(saved['state_dict'], strict=True)
    with restored.installed(model):
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, rtol=0, atol=0)
    try:
        with adapter.installed(model):
            raise RuntimeError('cleanup check')
    except RuntimeError:
        pass
    assert all(b._cache is None for b in adapter.blocks.values())
    assert all(not l.mlp.down_proj._forward_pre_hooks for l in model.model.layers)
    print('PASS direct: exact linear map, no hidden neurons/tanh/bias, identity, gradients, cascading, frozen backbone, reload, cleanup')


if __name__ == '__main__':
    main()
