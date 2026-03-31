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
from grid_on_table import Example


def measure_oscillation(example) -> float:
    """Return max z-position std over last 50% of frames."""
    if len(example.positions_history) < 4:
        return 0.0
    positions = np.array(example.positions_history)
    n_frames = positions.shape[0]
    start = n_frames // 2
    z = positions[start:, :, 2]
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

    # --- Sweep 1: Resolution (2 layers) ---
    resolutions = [1, 2, 4, 8, 10, 16, 20, 32, 48, 64]
    res_osc = []
    print("=== Sweep 1: Resolution (2 layers) ===")
    for n in resolutions:
        t0 = time.time()
        osc = run_config(frames=frames, grid_n=n, layers=2)
        dt = time.time() - t0
        res_osc.append(osc)
        print(f"  grid_n={n:3d} -> osc={osc:.4f} cm  ({dt:.1f}s)")

    # --- Sweep 2: Layers (10x10) ---
    layer_counts = [1, 2, 3, 4, 5, 8, 10, 15, 20]
    layer_osc = []
    print("\n=== Sweep 2: Layers (10x10 grid) ===")
    for nl in layer_counts:
        t0 = time.time()
        osc = run_config(frames=frames, grid_n=10, layers=nl)
        dt = time.time() - t0
        layer_osc.append(osc)
        print(f"  layers={nl:3d} -> osc={osc:.4f} cm  ({dt:.1f}s)")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(resolutions, res_osc, "o-", color="tab:blue", linewidth=2)
    ax1.set_xlabel("Grid resolution (NxN)")
    ax1.set_ylabel("Max Z-position std (cm)")
    ax1.set_title("Oscillation vs Resolution (2 layers)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(layer_counts, layer_osc, "s-", color="tab:red", linewidth=2)
    ax2.set_xlabel("Number of layers")
    ax2.set_ylabel("Max Z-position std (cm)")
    ax2.set_title("Oscillation vs Layers (10x10 grid)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "instability_sweep.png")
    out_path = os.path.abspath(out_path)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
