#!/usr/bin/env python3
"""Plot a compact, checkpoint-grounded NeuroStrike safety-neuron overlay.

Inputs: trained bottleneck_adapter.pt written by the Viking notebook (A3b/A5).
Outputs: a single PNG and SVG in the checkpoint directory (or --output).
No model download, training, Hugging Face login, or changes to checkpoint/model.

Original Llama MLP connectivity is architectural (not recovered original weights).
Selected channel IDs, per-layer counts, bottleneck rank, and coupling statistics
are read from the actual trained checkpoint. No invented neuron locations/edges.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="Path to trained bottleneck_adapter.pt")
    p.add_argument("--layer", type=int, default=None,
                   help="Layer index (default: layer with most selected channels)")
    p.add_argument("--output", type=Path, default=None, help="Output PNG path")
    p.add_argument("--intermediate-size", type=int, default=8192,
                   help="Original Llama-3.2-1B MLP width (default 8192)")
    p.add_argument("--hidden-size", type=int, default=2048,
                   help="Original Llama-3.2-1B hidden width (default 2048)")
    p.add_argument("--num-layers", type=int, default=16,
                   help="Original Llama-3.2-1B decoder layers (default 16)")
    return p.parse_args()


def read_checkpoint(path):
    # weights_only=True avoids arbitrary Python-object unpickling.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as e:
        raise SystemExit("Requires PyTorch supporting torch.load(weights_only=True)") from e
    if not isinstance(ckpt, dict) or ckpt.get("kind") != "bottleneck":
        raise SystemExit("Expected a NeuroStrike bottleneck_adapter.pt checkpoint")
    selection = ckpt.get("selected_channels")
    state = ckpt.get("state_dict")
    if not isinstance(selection, dict) or not isinstance(state, dict) or not selection:
        raise SystemExit("Missing selected_channels or state_dict in checkpoint")
    return selection, state


def layer_index(name):
    m = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj$", name)
    if not m:
        raise ValueError(f"Unexpected target module: {name!r}")
    return int(m.group(1))


def draw_box(ax, x, y, w, h, label, sub="", face="#F2F4F7", edge="#CDD4DD", fg="#263346"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=1.25))
    ax.text(x+w/2, y+h/2+(0.12 if sub else 0), label, ha="center", va="center",
            fontsize=11.5, weight="bold", color=fg)
    if sub:
        ax.text(x+w/2, y+h/2-0.20, sub, ha="center", va="center", fontsize=9.2, color=fg)


def arrow(ax, xy0, xy1, color="#8B98A8", lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, connectionstyle="arc3"))


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    selected, state = read_checkpoint(args.checkpoint)
    names = list(selected)  # matches ModuleDict insertion order in the training notebook
    counts = {layer_index(n): len(selected[n]) for n in names}
    if len(counts) != len(names):
        raise SystemExit("More than one down_proj target found in the same layer")
    if any(i >= args.num_layers or i < 0 for i in counts):
        raise SystemExit("Layer indices inconsistent with --num-layers")
    chosen_layer = args.layer if args.layer is not None else max(counts, key=counts.get)
    matching = [n for n in names if layer_index(n) == chosen_layer]
    if len(matching) != 1:
        raise SystemExit(f"Layer {chosen_layer} unavailable; selected layers: {sorted(counts)}")
    name = matching[0]
    ids = np.array([int(i) for i in selected[name]], dtype=np.int64)
    if len(ids) != len(set(ids.tolist())) or np.any(ids < 0) or np.any(ids >= args.intermediate_size):
        raise SystemExit("Selected channel indices invalid for --intermediate-size")
    key = f"selected_layer_{names.index(name)}"
    try:
        A = state[f"blocks.{key}.up.weight"].detach().float().numpy()
        B = state[f"blocks.{key}.down.weight"].detach().float().numpy()
    except KeyError as e:
        raise SystemExit(f"Missing adapter weights for {name}: {e}") from e
    k = len(ids)
    rank = A.shape[0]
    if A.shape != (rank, k) or B.shape != (k, rank):
        raise SystemExit(f"Unexpected shapes: A={A.shape}, B={B.shape}, selected={k}")
    J = B @ A  # derivative of delta z = B tanh(A z) at z=0
    off = J.copy()
    np.fill_diagonal(off, 0)
    off_norm = float(np.linalg.norm(off))
    total_norm = float(np.linalg.norm(J))
    off_frac = off_norm / total_norm if total_norm else 0.0

    bg = "#FFFFFF"
    dark = "#1B2737"
    muted = "#627184"
    gray = "#8B98A8"
    orange = "#D68C19"
    purple = "#7156AC"
    fig = plt.figure(figsize=(13.8, 7.7), facecolor=bg)
    fig.text(.055, .945, "Where the safety-neuron adapter acts", fontsize=21, weight="bold", color=dark)
    fig.text(.055, .901,
             f"Llama-3.2-1B-Instruct  ·  actual NeuroStrike selection  ·  trained bottleneck checkpoint  ·  layer {chosen_layer}",
             fontsize=11, color=muted)

    # Original architecture + adapter insertion point.
    ax = fig.add_axes([.055, .42, .89, .42]); ax.set_xlim(0, 13.2); ax.set_ylim(0, 4.1); ax.axis("off")
    draw_box(ax, .0, 1.12, 1.35, .92, "Input", f"{args.hidden_size}-d")
    draw_box(ax, 2.0, 2.05, 1.65, .85, "gate_proj", "frozen")
    draw_box(ax, 2.0, .38, 1.65, .85, "up_proj", "frozen")
    draw_box(ax, 4.18, 1.12, 1.5, .92, "SwiGLU", f"{args.intermediate_size}-d")
    draw_box(ax, 9.4, 1.12, 1.65, .92, "down_proj", "frozen")
    draw_box(ax, 11.68, 1.12, 1.4, .92, "Output", f"{args.hidden_size}-d")
    arrow(ax, (1.35,1.79),(2.0,2.45));arrow(ax,(1.35,1.35),(2.0,.80))
    arrow(ax, (3.65,2.47),(4.18,1.82));arrow(ax,(3.65,.80),(4.18,1.35))
    arrow(ax, (5.68,1.58),(6.17,1.58));arrow(ax,(8.81,1.58),(9.4,1.58))
    arrow(ax,(11.05,1.58),(11.68,1.58))
    # Actual selected-channel locus inside original MLP.
    draw_box(ax, 6.17, 1.06, 2.64, 1.04, f"{k} selected channels", "original MLP coordinates",
             face="#FFF4DF", edge="#DCA54D", fg="#7D5010")
    # Actual adapter is applied via down_proj forward_pre_hook to its input.
    ax.plot([7.48,7.48],[2.10,2.50],color=purple,lw=2.5)
    arrow(ax,(7.48,2.50),(7.48,2.70),color=purple,lw=2.5)
    draw_box(ax, 5.98, 2.72, 3.02, .82, "Learned bottleneck", f"A: {k}→{rank}  ·  tanh  ·  B: {rank}→{k}",
             face="#F1ECFA",edge="#BBA9DE",fg="#57408B")
    ax.plot([9.00,9.22,9.22],[3.12,3.12,1.90],color=purple,lw=2.3)
    arrow(ax,(9.22,1.90),(8.81,1.80),color=purple,lw=2.3)
    ax.text(7.50,.52,"z′ₛ = zₛ + B tanh(Azₛ)",fontsize=12.2,color=purple,ha="center",weight="bold")
    ax.text(7.50,.16,"Only selected original channels are edited; all pretrained weights remain frozen.",
            fontsize=9.8,color=muted,ha="center")

    # Real positions of ALL selected channel IDs (not a fabricated neural graph).
    strip = fig.add_axes([.075, .30, .63, .080])
    strip.set_xlim(0,args.intermediate_size-1);strip.set_ylim(0,1)
    strip.add_patch(Rectangle((0,.35),args.intermediate_size-1,.30,fc="#E7EBF0",ec="none"))
    strip.vlines(ids,.25,.75,color=orange,linewidth=.9,alpha=.95)
    strip.set_yticks([])
    strip.set_xlabel(f"Actual channel index within layer {chosen_layer} MLP (0–{args.intermediate_size-1})",fontsize=10,color=dark)
    strip.spines[['top','right','left']].set_visible(False)
    strip.tick_params(axis='x',labelsize=9,colors=muted)
    fig.text(.075,.397, f"Actual selected locations: {k}/{args.intermediate_size} channels ({100*k/args.intermediate_size:.2f}%)",
             fontsize=11.2,color=dark,weight="bold")
    fig.text(.075,.220, "Amber ticks = original neuron/channel IDs from the saved checkpoint (all shown).",
             fontsize=9.8,color=muted)

    # Compact empirical summary: real per-layer selection counts, and trained coupling.
    info = fig.add_axes([.75,.20,.22,.20]);info.axis("off")
    info.text(0,1.0,"Checkpoint measurements",fontsize=11.2,color=dark,weight="bold",va="top")
    info.text(0,.72,f"Selected layers: {len(counts)}/{args.num_layers}",fontsize=10.3,color=dark)
    info.text(0,.51,f"Total selected occurrences: {sum(counts.values()):,}",fontsize=10.3,color=dark)
    info.text(0,.30,f"Effective mixing ‖BA‖F: {total_norm:.3g}",fontsize=10.3,color=dark)
    info.text(0,.09,f"Off-diagonal share: {off_frac:.1%}",fontsize=10.3,color=dark)

    fig.text(.055,.08,
             "Topology: original Llama SwiGLU MLP.  Channel IDs, counts and BA statistics: actual trained checkpoint.",
             fontsize=9.5,color=muted)
    fig.text(.055,.049,
             "BA is the adapter's local Jacobian at z=0, not an original pretrained weight matrix or a causal connectivity estimate.",
             fontsize=9.3,color=muted)
    output = args.output if args.output else args.checkpoint.parent/f"layer_{chosen_layer}_real_overlay.png"
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.suffix.lower() != '.png':
        raise SystemExit('--output must end in .png')
    fig.savefig(output,dpi=200,facecolor=bg,bbox_inches='tight',pad_inches=.15)
    svg = output.with_suffix('.svg')
    fig.savefig(svg,facecolor=bg,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    print(f"Saved: {output} and {svg}")
    print(f"Layer {chosen_layer}: {k} real selected channels; rank={rank}; ||BA||F={total_norm:.5g}; offdiag fraction={off_frac:.3%}")

if __name__ == '__main__':
    main()
