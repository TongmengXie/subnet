#!/usr/bin/env python3
"""CPU-only tests of five target injection positions in a Llama-like block.

The toy block deliberately includes both separate normalisation sites and
attention, so hook placement/order is tested rather than assumed.
"""
from contextlib import nullcontext
import tempfile
from pathlib import Path

import torch
from torch import nn
from cross_layer_adapter import CrossLayerAdapter, PLACEMENTS


class ToyMLP(nn.Module):
    def __init__(self, d_model=12, d_ff=16):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(12)
        self.post_attention_layernorm = nn.LayerNorm(12)
        self.self_attn = nn.Linear(12, 12, bias=False)  # stand-in for attention
        self.mlp = ToyMLP()
        self.attention_calls = 0

    def forward(self, x):
        self.attention_calls += 1
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyLayer() for _ in range(9)])
        self.readout = nn.Linear(12, 5, bias=False)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return self.readout(x)


def traced_run(model, x, adapter=None):
    layer = model.model.layers[8]
    modules = {
        'input_norm_in': layer.input_layernorm,
        'attn_in': layer.self_attn,
        'post_norm_in': layer.post_attention_layernorm,
        'mlp_in': layer.mlp,
        'down_in': layer.mlp.down_proj,
    }
    seen = {}
    with (adapter.installed(model) if adapter is not None else nullcontext()):
        handles = []
        try:
            for label, mod in modules.items():
                def capture(_mod, args, _label=label):
                    seen[_label] = args[0].detach().clone()
                handles.append(mod.register_forward_pre_hook(capture))
            output = model(x)
        finally:
            for handle in handles:
                handle.remove()
    return output, seen


def has_changed(a, b):
    return bool(torch.max(torch.abs(a - b)) > 1e-7)


def main():
    torch.manual_seed(7)
    model = ToyModel()
    for p in model.parameters():
        p.requires_grad_(False)
    original_weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    source_ids = [0, 3, 4, 6]
    target_ids = [2, 5, 7, 9, 11]
    x = torch.randn(2, 3, 12)
    base, base_trace = traced_run(model, x)

    for placement in PLACEMENTS:
        adapter = CrossLayerAdapter(
            model, source_layer=7, target_layer=8,
            source_channels=source_ids, target_channels=target_ids,
            rank=3, placement=placement,
        )
        assert adapter.parameter_count == 3 * (4 + 5) == 27
        assert len(tuple(adapter.parameters())) == 2
        before_calls = model.model.layers[8].attention_calls
        zero, _ = traced_run(model, x, adapter)
        torch.testing.assert_close(base, zero, rtol=0, atol=0)
        assert model.model.layers[8].attention_calls == before_calls + 1
        assert adapter._cache is None

        # Make the correction nonzero and check which operations it can affect.
        with torch.no_grad():
            adapter.decode.weight.fill_(0.15)
            modified, seen = traced_run(model, x, adapter)
            assert has_changed(base, modified), f'No output difference for {placement}'
            changed = {key: has_changed(base_trace[key], seen[key]) for key in base_trace}
            expected = {
                'pre_attention': (True, True, True, True, True),
                'post_input_norm_pre_attention': (False, True, True, True, True),
                'post_attention_pre_norm': (False, False, True, True, True),
                'post_norm_pre_mlp': (False, False, False, True, True),
                'selected_mlp': (False, False, False, False, True),
            }[placement]
            assert tuple(changed.values()) == expected, (placement, changed, expected)
            if placement == 'selected_mlp':
                complement = [i for i in range(16) if i not in target_ids]
                torch.testing.assert_close(base_trace['down_in'][..., complement],
                                           seen['down_in'][..., complement], rtol=0, atol=0)
            after_removing, _ = traced_run(model, x)
            torch.testing.assert_close(base, after_removing, rtol=0, atol=0)

        opt = torch.optim.AdamW(adapter.parameters(), lr=1e-2)
        with adapter.installed(model):
            for _ in range(3):
                opt.zero_grad()
                loss = model(x).square().mean()
                loss.backward()
                assert adapter.decode.weight.grad is not None
                assert torch.isfinite(adapter.decode.weight.grad).all()
                opt.step()
        assert adapter.encode.weight.grad is not None
        assert torch.isfinite(adapter.encode.weight.grad).all()
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(p.detach(), original_weights[n]) for n,p in model.named_parameters())
        assert adapter._cache is None

        with tempfile.TemporaryDirectory() as tmp:
            savepath = Path(tmp) / 'adapter.pt'
            torch.save(adapter.state_dict(), savepath)
            copy = CrossLayerAdapter(model, source_layer=7, target_layer=8,
                                     source_channels=source_ids, target_channels=target_ids,
                                     rank=3, placement=placement)
            copy.load_state_dict(torch.load(savepath, weights_only=True))
            with torch.no_grad():
                out1, _ = traced_run(model, x, adapter)
                out2, _ = traced_run(model, x, copy)
            torch.testing.assert_close(out1, out2, rtol=0, atol=0)

        print(f'PASS {placement}: correct order; identity; trainable; frozen backbone; reload')

    print('PASS: all placements; 4→3→5 trainable weights; unchanged frozen attention/MLPs')


if __name__ == '__main__':
    main()
