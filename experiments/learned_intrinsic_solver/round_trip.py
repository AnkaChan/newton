# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test shared-corner recovery from the existing cell-center polar encoding.

Experimental: this is an information-loss experiment, not a dynamics solver.
Both decoders receive only the rest grid, stored R/U (and optionally centers),
and explicitly supplied positions of the material z-min boundary. The original
free corners are used only after reconstruction to measure recovery error.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .cell_frames import CellFrames, compute_cell_frames
from .data import VoxelGridData, augment_grid, generate_cuboid
from .multiscale import generate_multiscale
from .vbd_samples import CornerProjector

__all__ = ["FrameDecoder", "build_measurement_operators", "make_hourglass", "write_round_trip_report"]


def build_measurement_operators(rest: VoxelGridData):
    """Return exact center-gradient [1/m] and corner-average center operators.

    Gradient rows run (cell, material-axis); multiplying positions [P,3]
    returns rows of F.T. Center rows are dimensionless, so their product
    with positions has units of meters. No per-cell independent corners occur.
    """
    from scipy import sparse

    cells = rest.cell_corner_indices
    count = len(cells)
    signs = 2 * np.indices((2, 2, 2)).reshape(3, -1).T - 1
    gradient = sparse.csr_matrix(
        (
            np.tile(signs.T.reshape(-1) / (4 * rest.cell_size), count),
            (np.repeat(np.arange(3 * count), 8), np.repeat(cells[:, None, :], 3, axis=1).reshape(-1)),
        ),
        shape=(3 * count, len(rest.corner_rest_positions)),
    )
    centers = sparse.csr_matrix(
        (np.full(8 * count, 1 / 8), (np.repeat(np.arange(count), 8), cells.reshape(-1))),
        shape=(count, len(rest.corner_rest_positions)),
    )
    return gradient, centers


class FrameDecoder:
    """Choose minimum free displacement among compatible shared-corner states.

    LSQR starts at zero displacement, with no damping, regularizer, extra
    anchors, or original free-corner input. On this rank-deficient system its
    solution lies in the operator's row space: unresolved null modes are lost.
    The supplied boundary is eliminated exactly. Center rows, when used, are
    divided by h so all residuals and row weights are dimensionless.
    """

    def __init__(self, rest: VoxelGridData, fixed_indices: np.ndarray, *, include_centers: bool):
        from scipy import sparse

        self.rest = rest
        self.fixed = np.asarray(fixed_indices, dtype=np.int64)
        self.free = np.ones(len(rest.corner_rest_positions), dtype=bool)
        self.free[self.fixed] = False
        self.include_centers = include_centers
        self.gradient, self.center = build_measurement_operators(rest)
        self.matrix = (
            sparse.vstack((self.gradient, self.center / rest.cell_size), format="csr")
            if include_centers
            else self.gradient
        )
        self.free_matrix = self.matrix[:, self.free]
        self.fixed_matrix = self.matrix[:, self.fixed]
        self.rest_measurement = self.matrix @ rest.corner_rest_positions

    def decode(self, encoded: CellFrames, fixed_positions: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Recover corners using encoded quantities and supplied boundary only.

        Args:
            encoded: Cell centers, R, U, and validity from the existing encoder.
                Only R/U and, when enabled, centers are read; its stored F is
                deliberately unused, so this performs the requested R@U decode.
            fixed_positions: Explicit global positions [m] of fixed corners in
                the same order as fixed_indices supplied to the constructor.
        """
        from scipy.sparse.linalg import lsqr

        if not encoded.valid.all():
            raise ValueError("Cannot decode invalid polar frames")
        fixed_positions = np.asarray(fixed_positions, dtype=np.float64)
        if fixed_positions.shape != (len(self.fixed), 3) or not np.isfinite(fixed_positions).all():
            raise ValueError("fixed_positions must contain finite xyz positions for each fixed corner")
        deformation = encoded.frames @ encoded.local_axes
        target = deformation.transpose(0, 2, 1).reshape(-1, 3)
        if self.include_centers:
            target = np.vstack((target, encoded.centers / self.rest.cell_size))
        fixed_displacement = fixed_positions - self.rest.corner_rest_positions[self.fixed]
        right_hand_side = target - self.rest_measurement - self.fixed_matrix @ fixed_displacement
        displacement = np.zeros_like(self.rest.corner_rest_positions)
        displacement[self.fixed] = fixed_displacement
        diagnostics = []
        for axis in range(3):
            result = lsqr(
                self.free_matrix,
                right_hand_side[:, axis],
                atol=1e-13,
                btol=1e-13,
                conlim=1e12,
                iter_lim=20000,
                damp=0.0,
            )
            if result[1] not in (0, 1, 2, 4, 5):
                raise RuntimeError(f"LSQR did not converge: status={result[1]}, iterations={result[2]}")
            displacement[self.free, axis] = result[0]
            diagnostics.append(
                {
                    "world_axis": axis,
                    "stop_code": int(result[1]),
                    "iterations": int(result[2]),
                    "weighted_residual_norm": float(result[3]),
                    "condition_estimate": float(result[6]),
                }
            )
        positions = self.rest.corner_rest_positions + displacement
        positions[self.fixed] = fixed_positions
        return positions, diagnostics


def make_hourglass(rest: VoxelGridData, *, amplitude: float = 0.001) -> np.ndarray:
    """Add an exactly invisible alternating xy corner warp in world x [m].

    delta_x(i,j,k) = amplitude * (-1)**(i+j) * k/nz. It vanishes on z=0.
    Opposite-face averages and cell centers cancel it. For h=0.025 m and
    amplitude=0.001 m, the trilinear map has det F >= 0.92 at every point.
    """
    coordinates = np.indices(tuple(count + 1 for count in rest.cell_counts)).reshape(3, -1).T
    positions = rest.corner_rest_positions.copy()
    positions[:, 0] += (
        amplitude * (-1.0) ** (coordinates[:, 0] + coordinates[:, 1]) * coordinates[:, 2] / rest.cell_counts[2]
    )
    return positions


def _error_metrics(rest, original, recovered, encoded, fixed):
    decoded = compute_cell_frames(rest, recovered)
    corner_error = np.linalg.norm(recovered - original, axis=1)
    center_error = np.linalg.norm(decoded.centers - encoded.centers, axis=1)
    gradient_error = decoded.deformation - encoded.frames @ encoded.local_axes
    return {
        "corner_rmse_m": float(np.sqrt(np.mean(corner_error**2))),
        "corner_max_error_m": float(corner_error.max()),
        "center_rmse_m": float(np.sqrt(np.mean(center_error**2))),
        "center_max_error_m": float(center_error.max()),
        "gradient_component_rmse": float(np.sqrt(np.mean(gradient_error**2))),
        "gradient_max_abs_error": float(np.abs(gradient_error).max()),
        "boundary_max_error_m": float(np.linalg.norm(recovered[fixed] - original[fixed], axis=1).max()),
        "recovered_min_center_det": float(decoded.determinants.min()),
    }


def _cases(rest, seeds):
    canonical = rest.corner_rest_positions
    yield "identity", "Identity", "Canonical cuboid; a zero-displacement baseline.", canonical, {}
    affine = np.array([[1.0, 0.0, 0.18], [0.0, 1.0, -0.10], [0.0, 0.0, 1.05]])
    yield (
        "affine",
        "Known affine shear + stretch",
        "F is constant. The odd 11-by-11 corner cross section allows the minimum-displacement decoder to remove an invisible checkerboard component even from this affine map.",
        canonical @ affine.T,
        {"affine_F": affine.tolist()},
    )
    sample = augment_grid(rest, deformation_amplitude=0.15, velocity_amplitude=0.75, seed=0)
    projected, _, residual = CornerProjector(rest).project(sample)
    yield (
        "projected_0",
        "Original projected initializer · seed 0",
        "The existing independent cell augmenter, projected to shared corners with all 12 cell edges. Shown before the VBD gallery's rigid world transform; the rest face stays fixed.",
        projected,
        {"seed": 0, "deformation_amplitude": 0.15, "edge_projection_rms_m": residual},
    )
    for seed in seeds:
        sample = generate_multiscale(rest, seed=seed)
        yield (
            f"multiscale_{seed}",
            f"Multiscale initializer · seed {seed}",
            "The current compatible coarse-to-fine corner-displacement augmenter, with its supplied fixed face.",
            sample.positions,
            {
                "seed": seed,
                "effective_scale": sample.effective_scale,
                "control_counts": [level.control_counts for level in sample.levels],
                "amplitudes_m": [level.amplitude_m for level in sample.levels],
            },
        )
    yield (
        "hourglass",
        "Invisible corner warp · 1 mm",
        "delta_x(i,j,k) = 0.001 (-1)^(i+j) k/nz. Every cell center and center gradient matches identity, and the fixed face is unchanged, but the free corners differ by up to 1 mm.",
        make_hourglass(rest),
        {"amplitude_m": 0.001, "minimum_trilinear_det_bound": 1 - 0.002 / rest.cell_size},
    )


def write_round_trip_report(output: Path, *, cell_counts=(10, 10, 40), cell_size=0.025, seeds=(0, 7)) -> dict:
    """Run the round trip, archive exact data, and write an independent report."""
    rest = generate_cuboid(tuple(cell_counts), cell_size=cell_size)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
    decoders = {
        "axes": FrameDecoder(rest, fixed, include_centers=False),
        "centers_axes": FrameDecoder(rest, fixed, include_centers=True),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    report = {
        "conclusion": "R and U recover the sampled center gradient, but cell-center frames do not uniquely determine shared corners, even when origins and the fixed face are stored.",
        "cell_counts": list(cell_counts),
        "cell_size_m": cell_size,
        "cell_count": len(rest.cell_corner_indices),
        "corner_count": len(rest.corner_rest_positions),
        "fixed_corner_count": len(fixed),
        "fixed_face": "material z-min",
        "decoder": "Zero-start undamped LSQR of displacement from the canonical rest grid; minimum free displacement within the solution set. No free original corners, regularizer, extra anchors, or original-state initial guess.",
        "decoder_inputs": "Canonical rest topology/dimensions, stored polar R and local U, optional stored cell centers, and explicitly supplied global positions of the fixed face.",
        "axes_row_units": "Dimensionless F residuals; coefficients +/- 1/(4h)",
        "center_row_units": "Center residual in meters divided by h; coefficients 1/(8h); equal dimensionless row weight to F",
        "solver_atol": 1e-13,
        "solver_btol": 1e-13,
        "numpy_version": np.__version__,
        "corner_error_definition": "sqrt(mean over corners of squared Euclidean xyz error)",
        "cases": [],
    }
    payload = {
        "rest_positions": rest.corner_rest_positions.tolist(),
        "cells": rest.cell_corner_indices.tolist(),
        "fixed_indices": fixed.tolist(),
        "cases": [],
    }
    cases = list(_cases(rest, seeds))
    cases.sort(key=lambda case: 0 if case[0].startswith("multiscale_") else 1 if case[0] == "projected_0" else 2)
    for case_id, label, description, original, extra in cases:
        start = time.perf_counter()
        encoded = compute_cell_frames(rest, original)
        if not encoded.valid.all():
            raise RuntimeError(f"Case {case_id} has invalid center frames")
        info = {
            "id": case_id,
            "label": label,
            "description": description,
            "parameters": extra,
            "min_original_center_det": float(encoded.determinants.min()),
            "polar_F_max_abs_error": float(np.abs(encoded.frames @ encoded.local_axes - encoded.deformation).max()),
            "decoders": {},
        }
        archive = {
            "rest_positions": rest.corner_rest_positions,
            "original_positions": original,
            "cell_corner_indices": rest.cell_corner_indices,
            "fixed_indices": fixed,
            "supplied_fixed_positions": original[fixed],
            "stored_centers": encoded.centers,
            "stored_R": encoded.frames,
            "stored_U": encoded.local_axes,
        }
        item = {"id": case_id, "original": original.tolist(), "recovered": {}}
        for name, decoder in decoders.items():
            recovered, diagnostics = decoder.decode(encoded, original[fixed])
            metrics = _error_metrics(rest, original, recovered, encoded, fixed)
            info["decoders"][name] = {**metrics, "lsqr": diagnostics}
            archive[f"recovered_{name}"] = recovered
            item["recovered"][name] = recovered.tolist()
        if case_id == "hourglass":
            canonical_encoding = compute_cell_frames(rest, rest.corner_rest_positions)
            info["encoding_difference_from_identity"] = {
                "centers_max_abs_m": float(np.abs(encoded.centers - canonical_encoding.centers).max()),
                "F_max_abs": float(np.abs(encoded.deformation - canonical_encoding.deformation).max()),
                "R_max_abs": float(np.abs(encoded.frames - canonical_encoding.frames).max()),
                "U_max_abs": float(np.abs(encoded.local_axes - canonical_encoding.local_axes).max()),
                "boundary_max_abs_m": float(np.abs(original[fixed] - rest.corner_rest_positions[fixed]).max()),
            }
        info["wall_seconds"] = time.perf_counter() - start
        np.savez_compressed(output / "data" / f"{case_id}.npz", **archive)
        report["cases"].append(info)
        payload["cases"].append(item)
        print(
            json.dumps({"case": case_id, "wall_seconds": info["wall_seconds"], "decoders": info["decoders"]}),
            flush=True,
        )
    # A small exact-topology dense audit makes rank deficiency directly visible.
    small = generate_cuboid((2, 2, 3), cell_size=cell_size)
    small_fixed = np.flatnonzero(small.corner_rest_positions[:, 2] == 0)
    report["small_grid_rank_audit"] = {}
    for name in decoders:
        matrix = FrameDecoder(small, small_fixed, include_centers=name == "centers_axes").free_matrix.toarray()
        rank = int(np.linalg.matrix_rank(matrix))
        report["small_grid_rank_audit"][name] = {
            "cell_counts": [2, 2, 3],
            "rows": matrix.shape[0],
            "free_scalar_corner_unknowns": matrix.shape[1],
            "rank": rank,
            "nullity_per_world_component": matrix.shape[1] - rank,
        }
    payload["report"] = report
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "data.js").write_text(
        "window.ROUND_TRIP_DATA=" + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n"
    )
    shutil.copytree(Path(__file__).with_name("round_trip_web"), output, dirs_exist_ok=True)
    shutil.copytree(Path(__file__).with_name("inspector_web") / "vendor", output / "vendor", dirs_exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    shutil.copy2(__file__, output / "source" / "round_trip.py")
    return report


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs=3, default=(10, 10, 40))
    parser.add_argument("--cell-size", type=float, default=0.025)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 7), help="Multiscale seeds to include.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated" / "round_trip")
    args = parser.parse_args()
    report = write_round_trip_report(
        args.output, cell_counts=tuple(args.cells), cell_size=args.cell_size, seeds=args.seeds
    )
    print(f"Wrote {len(report['cases'])} round-trip cases to {args.output}")


if __name__ == "__main__":
    _main()
