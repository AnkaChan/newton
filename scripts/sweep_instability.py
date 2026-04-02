#!/usr/bin/env python
"""Sweep grid resolution and layer count, plot oscillation amplitude.

Usage:
    CUDA_VISIBLE_DEVICES=3 uv run --extra examples python scripts/sweep_instability.py
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer

# Re-use the Example class
sys.path.insert(0, os.path.dirname(__file__))
from grid_on_table import Example, find_settled_substep


def measure_oscillation(example) -> float:
    """Return max z-position std after horizontal motion settles."""
    if len(example.positions_history) < 4:
        return 0.0
    positions = np.array(example.positions_history)
    start = find_settled_substep(positions)
    z = positions[start:, :, 2]
    if z.shape[0] < 2:
        return 0.0
    return float(np.std(z, axis=0).max())


def run_config(frames=120, **kwargs) -> float:
    """Run one configuration headless and return oscillation metric."""
    viewer = newton.viewer.ViewerGL(headless=True)
    example = Example(viewer=viewer, **kwargs)

    for f in range(frames):
        example.step()
        example.render()

    wp.synchronize()
    viewer.close()
    return measure_oscillation(example)


def main():
    wp.init()
    frames = 100
    results = {}

    for fold in [False, True]:
        tag = "folded" if fold else "flat"
        print(f"\n{'#'*60}")
        print(f"# {tag.upper()}")
        print(f"{'#'*60}")

        # --- Sweep 1: Resolution (2 layers) ---
        resolutions = [1, 2, 4, 8, 10, 16, 20, 32, 48, 64]
        res_osc = []
        print(f"\n=== Sweep 1: Resolution, 2 layers, {tag} ===")
        for n in resolutions:
            t0 = time.time()
            osc = run_config(frames=frames, grid_n=n, layers=2, fold=fold)
            dt = time.time() - t0
            res_osc.append(osc)
            print(f"  grid_n={n:3d} -> osc={osc:.4f} cm  ({dt:.1f}s)")

        # --- Sweep 2: Layers (10x10) ---
        layer_counts = [1, 2, 3, 4, 5, 8, 10, 15, 20]
        layer_osc = []
        print(f"\n=== Sweep 2: Layers, 10x10, {tag} ===")
        for nl in layer_counts:
            t0 = time.time()
            osc = run_config(frames=frames, grid_n=10, layers=nl, fold=fold)
            dt = time.time() - t0
            layer_osc.append(osc)
            print(f"  layers={nl:3d} -> osc={osc:.4f} cm  ({dt:.1f}s)")

        results[tag] = {
            "resolutions": resolutions,
            "res_osc": res_osc,
            "layer_counts": layer_counts,
            "layer_osc": layer_osc,
        }

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for row, tag in enumerate(["flat", "folded"]):
        r = results[tag]
        color_res = "tab:blue" if tag == "flat" else "tab:purple"
        color_lay = "tab:red" if tag == "flat" else "tab:orange"

        axes[row, 0].plot(r["resolutions"], r["res_osc"], "o-", color=color_res, linewidth=2)
        axes[row, 0].set_xlabel("Grid resolution (NxN)")
        axes[row, 0].set_ylabel("Max Z-position std (cm)")
        axes[row, 0].set_title(f"Oscillation vs Resolution — 2 layers, {tag}")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(r["layer_counts"], r["layer_osc"], "s-", color=color_lay, linewidth=2)
        axes[row, 1].set_xlabel("Number of layers")
        axes[row, 1].set_ylabel("Max Z-position std (cm)")
        axes[row, 1].set_title(f"Oscillation vs Layers — 10x10 grid, {tag}")
        axes[row, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "instability_sweep.png")
    out_path = os.path.abspath(out_path)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
