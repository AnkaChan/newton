# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate figures for the exploratory HTML report from the reference functions."""

from pathlib import Path

import numpy as np
from numpy_probe import MU, sorted_stress, tensor_stress


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(__file__).parent
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "alm-svd-exploration",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    values = np.concatenate([np.linspace(-0.01, -1.0e-8, 200), np.linspace(1.0e-8, 0.01, 200)])
    history = MU * np.array([1.5, 0.5, 0.4])
    material_history = MU * np.diag([0.5, 1.5, 0.4])
    scalar, tensor = [], []
    for t in values:
        F = np.diag([1 + t, 1 - t, 0.7])
        scalar.append(sorted_stress(F, history)[0, 0] / 1000)
        tensor.append(tensor_stress(F, material_history)[0, 0] / 1000)
    axes[0].plot(values, scalar, color="#b45d25", label="Three sorted histories", lw=2)
    axes[0].plot(values, tensor, color="#087568", label="Material tensor history", lw=2)
    axes[0].set(
        xlabel="t in F = diag(1+t, 1-t, 0.7)", ylabel="Stress P11 [kPa]", title="Positive stretches exchange order"
    )
    axes[0].legend(loc="center right", fontsize=9)
    axes[0].set_xticks(np.linspace(-0.01, 0.01, 5))
    values = np.linspace(-0.004, 0.004, 401)
    history = MU * np.diag([1.0, 1.0, 0.8])
    unregularized, regularized = [], []
    for t in values:
        F = np.diag([1.0, 1.0, t])
        unregularized.append(tensor_stress(F, history)[2, 2] / 1000 if t != 0 else np.nan)
        regularized.append(tensor_stress(F, history, epsilon=1.0e-3)[2, 2] / 1000)
    axes[1].plot(values, unregularized, color="#b45d25", label="Unregularized", lw=2)
    axes[1].plot(values, regularized, color="#087568", label="epsilon = 0.001", lw=2)
    axes[1].set(
        xlabel="Signed thickness in F = diag(1, 1, t)", ylabel="Stress P33 [kPa]", title="A tet passes through flat"
    )
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].set_xticks(np.linspace(-0.004, 0.004, 5))
    for ax in axes:
        ax.axvline(0, color="#9aacaa", lw=0.8, ls="--")
        ax.grid(alpha=0.15)
    fig.savefig(folder / "stress_comparison.svg", metadata={"Date": None})
    svg = folder / "stress_comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(folder / "stress_comparison.png", dpi=150)
    print("Wrote stress_comparison.svg and stress_comparison.png")


if __name__ == "__main__":
    main()
