"""Selected-channel networks; joint hooks enforce inject-before-capture semantics."""
from contextlib import contextmanager
import torch
from torch import nn
from cross_layer_adapter import CrossLayerAdapter


def consecutive_pairs(channels):
    pairs = [(i, i + 1) for i in sorted(channels) if channels[i] and channels.get(i + 1)]
    if not pairs:
        raise ValueError('No selected consecutive layer pairs')
    return pairs


class AdjacentAdapter(nn.Module):
    """Connected skips or an exactly parameter-matched channel-separable control.

    Independent control: each target channel receives sum b*tanh(a*z_channel).
    Units are distributed deterministically across target channels. No cross-channel
    or cross-layer inputs enter the correction. Equal budgets do not imply equal
    functional capacity; record this control definition when reporting results.
    """
    def __init__(self, model, channels, rank=8, method='connected'):
        super().__init__()
        if method not in ('connected', 'independent') or rank < 1:
            raise ValueError('Invalid method/rank')
        self.pairs = consecutive_pairs(channels)
        self.channels = {i: list(channels[i]) for i in sorted({j for p in self.pairs for j in p})}
        self.method, self.rank = method, rank
        self.blocks = nn.ModuleDict()
        self._active = False
        budget = sum(rank * (len(channels[i]) + len(channels[j])) for i, j in self.pairs)
        if method == 'connected':
            for i, j in self.pairs:
                self.blocks[str(i)] = CrossLayerAdapter(model, source_layer=i, target_layer=j,
                    source_channels=channels[i], target_channels=channels[j], rank=rank)
        else:
            targets = [j for _, j in self.pairs]
            count = sum(len(channels[j]) for j in targets)
            if budget % 2 or budget // 2 < count:
                raise ValueError('Budget must allow at least one two-weight unit per target channel')
            units, extra = divmod(budget // 2, count)
            for j in targets:
                block = nn.Module()
                ids = []
                for k in range(len(channels[j])):
                    ids.extend([k] * (units + (extra > 0)))
                    extra = max(0, extra - 1)
                down = model.model.layers[j].mlp.down_proj
                CrossLayerAdapter._validate_channels(channels[j], down.in_features, 'target')
                block.register_buffer('unit_channel', torch.tensor(ids, device=down.weight.device))
                block.a = nn.Parameter(torch.randn(len(ids), device=down.weight.device))
                block.b = nn.Parameter(torch.zeros(len(ids), device=down.weight.device))
                self.blocks[str(j)] = block
        assert self.parameter_count == budget

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def config(self):
        return dict(channels=self.channels, rank=self.rank, method=self.method,
                    overlap_policy='inject_then_capture')

    @contextmanager
    def installed(self, model):
        if self._active or getattr(model, 'is_gradient_checkpointing', False):
            raise RuntimeError('Active hooks or gradient checkpointing unsupported')
        self._active = True
        handles = []
        incoming = {j: i for i, j in self.pairs}
        outgoing = {i for i, _ in self.pairs}
        def hook(layer):
            def apply(module, args):
                if self.method == 'connected':
                    if layer in incoming:
                        args = self.blocks[str(incoming[layer])]._inject_target(module, args)
                    if layer in outgoing:
                        self.blocks[str(layer)]._capture_source(module, args)
                    return args
                if layer not in incoming:
                    return None
                x = args[0]
                ids = torch.tensor(self.channels[layer], device=x.device)
                selected = x.index_select(-1, ids).float()
                block = self.blocks[str(layer)]
                values = block.b * torch.tanh(block.a * selected.index_select(-1, block.unit_channel))
                delta = torch.zeros_like(selected).index_add(-1, block.unit_channel, values)
                changed = x.clone()
                changed.index_copy_(-1, ids, (selected + delta).to(x.dtype))
                return (changed,) + args[1:]
            return apply
        try:
            for block in self.blocks.values():
                if self.method == 'connected':
                    block._cache = None
            for layer in self.channels:
                handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook(layer)))
            yield model
            if self.method == 'connected' and any(b._cache is not None for b in self.blocks.values()):
                raise RuntimeError('Unconsumed source activation')
        finally:
            for handle in handles:
                handle.remove()
            if self.method == 'connected':
                for block in self.blocks.values():
                    block._cache = None
            self._active = False
