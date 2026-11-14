"""NeuroStrike-selected, cross-layer gated-MLP activation adapter for Llama.

For source layer i and target layer j>i:
  z_i = SiLU(gate_proj_i(x_i)) * up_proj_i(x_i)  [before down_proj_i]
  z_j = SiLU(gate_proj_j(x_j)) * up_proj_j(x_j)  [before down_proj_j]
  correction_j = B tanh(A z_i[S_i])  [n_target_selected]
  Placement "selected_mlp": z_j[S_j] += correction_j
  Earlier placements: hidden_state += down_proj_j[:, S_j] @ correction_j

Five placements are supported: before target attention RMSNorm, after attention
RMSNorm but before attention, after attention but before MLP RMSNorm,
after MLP RMSNorm but before MLP projections, or
at the selected MLP channels before target down_proj (original prototype).
All have the same trainable parameters. The earlier-placement interventions use
frozen down_proj columns to embed the learned correction in residual space;
these are NOT injections into selected target MLP neuron activations.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import pickle
import re
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def load_neurostrike_channel_map(path: str | Path, *, threshold: float = 3.0) -> Dict[int, list[int]]:
    """Reproduce original NeuroStrike z-score selection, unioning gate/up channels.

    WARNING: NeuroStrike .p files are Python pickles. Only use files from a trusted
    local copy of the original repository. Never deserialize untrusted pickles.
    """
    with open(path, 'rb') as handle:
        raw = pickle.load(handle)  # noqa: S301 -- only trusted local NeuroStrike artifact
    if not isinstance(raw, dict):
        raise TypeError('Expected NeuroStrike weight dictionary in the .p artifact')
    channels: Dict[int, set[int]] = {}
    for key, value in raw.items():
        match = re.search(r'(?:^|\.)layers\.(\d+)\.mlp\.(?:gate_proj|up_proj)$', str(key))
        if not match:
            continue
        layer = int(match.group(1))
        scores = np.asarray(value, dtype=np.float64).reshape(-1)
        sigma = np.std(scores)
        if not np.isfinite(sigma) or sigma == 0:
            continue
        z = (scores - scores.mean()) / sigma
        chosen = np.flatnonzero((np.abs(z) > threshold) & (scores > 0))
        channels.setdefault(layer, set()).update(chosen.tolist())
    return {layer: sorted(indices) for layer, indices in sorted(channels.items()) if indices}


PLACEMENTS = (
    'pre_attention',
    'post_input_norm_pre_attention',
    'post_attention_pre_norm',
    'post_norm_pre_mlp',
    'selected_mlp',
)


class CrossLayerAdapter(nn.Module):
    """Selected-neuron skip: bottleneck+tanh, or one direct dense linear map.

    connection='direct' has no hidden neurons or activation function; bias is optional.
    Its zero-initialized matrix maps all selected sources to all selected targets.

    Source: down_proj pre-hook on source MLP selected channels.
    Target hook depends on placement; see PLACEMENTS. Constructor does not
    register the frozen model as a submodule, so adapter.parameters() only covers
    the new trainable weights. In earlier placements the output is projected
    into the residual stream using selected columns of frozen target down_proj.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        source_layer: int,
        target_layer: int,
        source_channels: Iterable[int],
        target_channels: Iterable[int],
        rank: int = 8,
        placement: str = 'selected_mlp',
        connection: str = 'bottleneck',
        activation: str | None = None,
        bias: bool = False,
    ):
        super().__init__()
        if not (0 <= source_layer < target_layer):
            raise ValueError('Require source_layer < target_layer')
        if rank < 1:
            raise ValueError('rank must be positive')
        if connection not in ('bottleneck', 'direct'):
            raise ValueError('connection must be bottleneck or direct')
        if placement not in PLACEMENTS:
            raise ValueError(f'placement must be one of {PLACEMENTS}, got {placement!r}')
        activation = activation or ('identity' if connection == 'direct' else 'tanh')
        if activation not in ('tanh', 'identity') or (connection == 'direct' and activation != 'identity'):
            raise ValueError('Direct connections require identity; bottleneck supports tanh/identity')
        self.activation, self.bias = activation, bool(bias)
        self.source_layer = int(source_layer)
        self.target_layer = int(target_layer)
        self.placement = placement
        self.source_name = f'model.layers.{source_layer}.mlp.down_proj'
        self.target_down_name = f'model.layers.{target_layer}.mlp.down_proj'
        self.target_name = {
            'pre_attention': f'model.layers.{target_layer}',
            'post_input_norm_pre_attention': f'model.layers.{target_layer}.self_attn',
            'post_attention_pre_norm': f'model.layers.{target_layer}.post_attention_layernorm',
            'post_norm_pre_mlp': f'model.layers.{target_layer}.mlp',
            'selected_mlp': self.target_down_name,
        }[placement]
        modules = dict(model.named_modules())
        if any(name not in modules for name in
               (self.source_name, self.target_down_name, self.target_name)):
            raise ValueError('Model must expose Llama-style layers with post_attention_layernorm '
                             'and separate mlp.down_proj modules')
        source_mod, target_mod = modules[self.source_name], modules[self.target_down_name]
        if not isinstance(source_mod, nn.Linear) or not isinstance(target_mod, nn.Linear):
            raise TypeError('Only separate nn.Linear down_proj modules are supported')
        self.source_channels = self._validate_channels(source_channels, source_mod.in_features, 'source')
        self.target_channels = self._validate_channels(target_channels, target_mod.in_features, 'target')
        self.connection = connection
        self.rank = int(rank) if connection == 'bottleneck' else None
        self.target_hidden_size = int(target_mod.out_features)
        if connection == 'direct':
            self.edges = nn.Linear(len(self.source_channels), len(self.target_channels), bias=bias)
            nn.init.zeros_(self.edges.weight)  # identity at init, no hidden units
            if bias:
                nn.init.zeros_(self.edges.bias)
        else:
            self.encode = nn.Linear(len(self.source_channels), rank, bias=bias)
            self.decode = nn.Linear(rank, len(self.target_channels), bias=bias)
            nn.init.normal_(self.encode.weight, std=1 / len(self.source_channels) ** .5)
            nn.init.zeros_(self.decode.weight)  # exact zero correction at initialization
            if bias:
                nn.init.zeros_(self.encode.bias)
                nn.init.zeros_(self.decode.bias)
        # Keep a non-registered reference to frozen down_proj. PyTorch module
        # registration would incorrectly include the whole pretrained model.
        object.__setattr__(self, '_target_down_proj', target_mod)
        # Keep adapter on destination device in float32 for stable small updates.
        self.to(device=target_mod.weight.device, dtype=torch.float32)
        self._cache: Optional[Tensor] = None
        self._active = False

    @staticmethod
    def _validate_channels(indices: Iterable[int], dimension: int, label: str) -> tuple[int, ...]:
        ids = tuple(sorted(int(i) for i in indices))
        if not ids or len(set(ids)) != len(ids) or ids[0] < 0 or ids[-1] >= dimension:
            raise ValueError(f'Invalid {label} channel IDs for dimension {dimension}')
        return ids

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, source_selected: Tensor) -> Tensor:
        """Return target-channel correction; does not modify any model activations."""
        if self.connection == 'direct':
            return self.edges(source_selected.to(device=self.edges.weight.device, dtype=torch.float32))
        x = source_selected.to(device=self.encode.weight.device, dtype=torch.float32)
        hidden = self.encode(x)
        return self.decode(torch.tanh(hidden) if self.activation == 'tanh' else hidden)

    def _capture_source(self, _module: nn.Module, args: tuple):
        if not args:
            raise RuntimeError('Source down_proj has no input')
        if self._cache is not None:
            raise RuntimeError('Source called again before target; unsupported model execution order')
        x = args[0]
        if x.ndim not in (2, 3):
            raise RuntimeError(f'Unexpected source tensor shape {tuple(x.shape)}')
        index = torch.as_tensor(self.source_channels, device=x.device, dtype=torch.long)
        self._cache = x.index_select(-1, index)
        return None  # source MLP computation unchanged

    def _inject_target(self, _module: nn.Module, args: tuple):
        if not args or self._cache is None:
            raise RuntimeError('Target reached before source: check hooks and layer order')
        x = args[0]
        source = self._cache
        self._cache = None  # do not keep the previous autograd graph alive
        if x.shape[:-1] != source.shape[:-1]:
            raise RuntimeError('Token/batch dimensions differ across source/target; cannot align tokens')
        correction = self(source)  # [batch, seq, n_target_selected], float32
        if self.placement == 'selected_mlp':
            idx = torch.as_tensor(self.target_channels, device=x.device, dtype=torch.long)
            if x.shape[-1] <= idx[-1]:
                raise RuntimeError('Expected MLP intermediate channels at target hook')
            chosen = x.index_select(-1, idx)
            changed = x.clone()
            changed.index_copy_(-1, idx, chosen + correction.to(device=x.device, dtype=x.dtype))
        else:
            if x.shape[-1] != self.target_hidden_size:
                raise RuntimeError('Expected full residual hidden size at earlier target hook')
            # Use selected *frozen* down_proj columns as the output-space basis.
            # No new parameters are introduced: encode/decode have identical
            # shapes in all five placements. This is a direction-based residual
            # stream injection, NOT a selected MLP activation injection.
            target_weight = self._target_down_proj.weight
            target_idx = torch.as_tensor(self.target_channels, device=target_weight.device,
                                         dtype=torch.long)
            basis = target_weight.index_select(1, target_idx).to(dtype=torch.float32)
            residual_correction = F.linear(correction, basis)
            changed = x + residual_correction.to(device=x.device, dtype=x.dtype)
        return (changed,) + tuple(args[1:])

    def _inject_target_with_kwargs(self, module: nn.Module, args: tuple, kwargs: dict):
        """HF attention passes hidden_states by keyword; preserve other inputs."""
        if args:
            return self._inject_target(module, args), kwargs
        if 'hidden_states' not in kwargs:
            raise RuntimeError('Target has no positional input or hidden_states keyword')
        changed = self._inject_target(module, (kwargs['hidden_states'],))[0]
        return args, {**kwargs, 'hidden_states': changed}

    @contextmanager
    def installed(self, model: nn.Module):
        """Install temporary hooks. Never enable with gradient checkpointing or torch.compile."""
        if self._active:
            raise RuntimeError('Cross-layer adapter hooks are already active')
        modules = dict(model.named_modules())
        if self.source_name not in modules or self.target_name not in modules:
            raise RuntimeError('Model module names changed since adapter construction')
        self._cache = None
        handles = []
        self._active = True
        try:
            handles.append(modules[self.source_name].register_forward_pre_hook(self._capture_source))
            handles.append(modules[self.target_name].register_forward_pre_hook(
                self._inject_target_with_kwargs, with_kwargs=True))
            yield model
        finally:
            for handle in handles:
                handle.remove()
            self._cache = None
            self._active = False
