# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build an experimental, static inspector for voxel-center polar frames.

The axes are computed from the actual shared-corner geometry after projection.
They are the trilinear deformation gradient at each cell's center; one matrix
does not capture all corner warping of a general hexahedron. This module's API
and artifact schema are experimental and may change.
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import VoxelGridData, augment_grid, generate_cuboid
from .vbd_samples import CornerProjector

__all__ = ["CellFrames", "compute_cell_frames", "generate_frame_sample", "write_frame_inspector"]


@dataclass(frozen=True)
class CellFrames:
    """Experimental center kinematics, with zero frame/axes for invalid cells."""

    centers: np.ndarray
    """Current corner-average centers [m], shape [cell_count, 3]."""
    deformation: np.ndarray
    """Dimensionless world deformation F, shape [cell_count, 3, 3]."""
    frames: np.ndarray
    """Proper orthonormal frame R as world columns, shape [cell_count, 3, 3]."""
    local_axes: np.ndarray
    """Dimensionless columns of R.T @ F, shape [cell_count, 3, 3]."""
    determinants: np.ndarray
    """Center volume ratios det(F), shape [cell_count]."""
    valid: np.ndarray
    """Whether F is positively oriented and numerically nonsingular, shape [cell_count]."""


def compute_cell_frames(rest: VoxelGridData, positions: np.ndarray) -> CellFrames:
    """Compute experimental cell-center gradients and their polar frames.

    Material face labels use the fixed rest topology. Each column of F is the
    difference of opposite face centroids divided by rest cell size. Its
    magnitude is preserved, so the axes contain both stretch and shear.

    Args:
        rest: Canonical cubic-cell topology and rest dimensions.
        positions: Current shared corner positions [m], shape [corner_count, 3].

    Returns:
        Center gradients, frames, local axes and validity flags. Inverted or
        collapsed cells have zero frame and local-axis matrices; consumers must
        check ``valid`` before using them. Valid cells satisfy F = R @ U, with
        a proper orthonormal R and symmetric positive-definite U.

    Raises:
        ValueError: Positions have the wrong shape or contain nonfinite values.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != rest.corner_rest_positions.shape:
        raise ValueError("positions must have shape [corner_count, 3]")
    if not np.isfinite(positions).all():
        raise ValueError("positions must contain only finite values")
    corners = positions[rest.cell_corner_indices]
    signs = 2 * np.indices((2, 2, 2)).reshape(3, -1).T - 1
    # Subtract one corner before differencing to reduce cancellation under translation.
    relative = corners - corners[:, :1]
    deformation = np.einsum("nki,kj->nij", relative, signs) / (4 * rest.cell_size)
    left, singular_values, right_transpose = np.linalg.svd(deformation)
    determinants = np.linalg.det(deformation)
    valid = (determinants > 0) & (singular_values[:, -1] > 1e-10 * np.maximum(1.0, singular_values[:, 0]))
    frames = np.zeros_like(deformation)
    frames[valid] = left[valid] @ right_transpose[valid]
    local_axes = np.zeros_like(deformation)
    local_axes[valid] = frames[valid].transpose(0, 2, 1) @ deformation[valid]
    return CellFrames(corners.mean(axis=1), deformation, frames, local_axes, determinants, valid)


def generate_frame_sample(
    *,
    cell_counts: tuple[int, int, int] = (10, 10, 40),
    cell_size: float = 0.025,
    deformation_amplitude: float = 0.15,
    seed: int = 0,
) -> tuple[VoxelGridData, np.ndarray, float]:
    """Generate an experimental seeded compatible geometry for the inspector.

    Args:
        cell_counts: Number of cells along the three material axes.
        cell_size: Canonical cubic voxel edge length [m].
        deformation_amplitude: Independent cell perturbation bound, in [0, 1/3).
        seed: Explicit random seed passed to the augmenter.

    Returns:
        Canonical rest grid, current shared positions [m] of shape
        [corner_count, 3], and projection edge RMS residual [m]. The material
        z=0 side remains fixed. This is a static initialization, not simulation.
    """
    rest = generate_cuboid(cell_counts, cell_size=cell_size)
    augmented = augment_grid(rest, deformation_amplitude=deformation_amplitude, velocity_amplitude=0.0, seed=seed)
    positions, _, residual = CornerProjector(rest).project(augmented)
    return rest, positions, residual


def write_frame_inspector(
    output: Path,
    *,
    seed: int = 0,
    deformation_amplitude: float = 0.15,
    cell_size: float = 0.025,
) -> dict:
    """Write an experimental offline-capable HTML inspector and exact sample.

    Args:
        output: Destination directory for index.html and its relative assets.
        seed: Explicit augmenter seed for the 10 by 10 by 40 grid.
        deformation_amplitude: Independent cell perturbation bound, in [0, 1/3).
        cell_size: Canonical cubic voxel edge length [m].

    Returns:
        Numeric diagnostics and generation metadata for verification.
    """
    rest, positions, residual = generate_frame_sample(
        seed=seed, deformation_amplitude=deformation_amplitude, cell_size=cell_size
    )
    result = compute_cell_frames(rest, positions)
    valid_frames = result.frames[result.valid]
    valid_axes = result.local_axes[result.valid]
    diagnostics = {
        "valid_cell_count": int(result.valid.sum()),
        "invalid_cell_count": int((~result.valid).sum()),
        "min_determinant": float(result.determinants.min()),
        "max_determinant": float(result.determinants.max()),
        "max_orthogonality_error": float(
            np.max(np.abs(valid_frames.transpose(0, 2, 1) @ valid_frames - np.eye(3)), initial=0)
        ),
        "max_reconstruction_error": float(
            np.max(np.abs(valid_frames @ valid_axes - result.deformation[result.valid]), initial=0)
        ),
    }
    metadata = {
        "cell_counts": list(rest.cell_counts),
        "cell_size": rest.cell_size,
        "seed": seed,
        "deformation_amplitude": deformation_amplitude,
        "projection_residual_m": residual,
        "numpy_version": np.__version__,
        "description": "Seeded cell augmentation projected to shared corners, with material z=0 fixed. Static geometry.",
        "gradient": "Opposite material-face centroid differences / rest edge length; evaluated at the cell center.",
        "frame": "Polar rotation R of the actual current center gradient F. Local axes U = R.T @ F.",
        "invalid_policy": "Nonpositive determinant or near-singular F is flagged; no local frame is displayed.",
    }
    payload = {
        "metadata": metadata,
        "rest_positions": rest.corner_rest_positions.tolist(),
        "positions": positions.tolist(),
        "cells": rest.cell_corner_indices.tolist(),
        "centers": result.centers.tolist(),
        "deformation": result.deformation.tolist(),
        "frames": result.frames.tolist(),
        "local_axes": result.local_axes.tolist(),
        "determinants": result.determinants.tolist(),
        "valid": result.valid.tolist(),
        "diagnostics": diagnostics,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(__file__).with_name("inspector_web"), output, dirs_exist_ok=True)
    (output / "data.js").write_text(
        "window.CELL_FRAME_DATA=" + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n",
        encoding="utf-8",
    )
    (output / "metadata.json").write_text(
        json.dumps({"metadata": metadata, "diagnostics": diagnostics}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "sample.npz",
        cell_counts=rest.cell_counts,
        cell_size=rest.cell_size,
        seed=seed,
        deformation_amplitude=deformation_amplitude,
        rest_positions=rest.corner_rest_positions,
        positions=positions,
        cell_corner_indices=rest.cell_corner_indices,
        centers=result.centers,
        deformation=result.deformation,
        frames=result.frames,
        local_axes=result.local_axes,
        determinants=result.determinants,
        valid=result.valid,
    )
    return {"metadata": metadata, "diagnostics": diagnostics}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deformation-amplitude", type=float, default=0.15)
    parser.add_argument("--cell-size", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated" / "cell_frames")
    args = parser.parse_args()
    diagnostics = write_frame_inspector(
        args.output, seed=args.seed, deformation_amplitude=args.deformation_amplitude, cell_size=args.cell_size
    )
    print(json.dumps({"output": str(args.output), **diagnostics}, indent=2))


if __name__ == "__main__":
    _main()
