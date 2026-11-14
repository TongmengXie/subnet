"""Count-matched random MLP neuron selection, independent of training RNGs."""
import numpy as np


def count_matched_random_neurons(reference, dimensions, *, seed):
    """Uniform subsets from all MLP neurons; overlap with NeuroStrike is allowed.

    Draw without replacement within each layer; sort IDs to preserve the adapter's
    indexing convention. Uses a local generator, never numpy/torch global RNGs.
    """
    if not reference or set(reference)-set(dimensions):
        raise ValueError('Missing reference selections or layer dimensions')
    rng = np.random.default_rng(seed)
    result = {}
    for layer, selected in sorted(reference.items()):
        dim = int(dimensions[layer])
        if not selected or len(set(selected)) != len(selected) or min(selected)<0 or max(selected)>=dim:
            raise ValueError(f'Invalid reference selection at layer {layer}')
        result[layer] = sorted(rng.choice(dim, size=len(selected), replace=False).tolist())
    return result
