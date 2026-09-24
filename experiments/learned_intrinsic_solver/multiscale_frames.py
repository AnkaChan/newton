# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export experimental seeded multiscale initial states to the frame inspector."""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .cell_frames import compute_cell_frames
from .data import generate_cuboid
from .multiscale import generate_multiscale

__all__ = ["write_multiscale_inspector"]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def write_multiscale_inspector(
    output: Path, *, seeds: tuple[int, ...] = tuple(range(20)), strength: float = 0.5
) -> dict:
    """Write an experimental static inspector with seed and level comparisons.

    Args:
        output: Destination directory for HTML, local browser assets and data.
        seeds: Nonempty unique nonnegative seeds; default zero through nineteen.
        strength: Requested amplitude budget as a fraction of the shortest side.

    Returns:
        The complete catalog of automatically derived grids and state metrics.
        JSON visualization arrays use ten decimal places; NumPy downloads retain
        exact float64 positions, controls and level contributions.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("states", "samples"):
        (output / folder).mkdir(exist_ok=True)
    shutil.copytree(Path(__file__).with_name("inspector_web"), output, dirs_exist_ok=True)
    rest = generate_cuboid((10, 10, 40), cell_size=0.025)
    catalog = {
        "seeds": list(seeds),
        "modes": [{"id": "combined", "label": "Combined"}],
        "levels": [],
        "states": [],
        "strength": strength,
        "description": "Seeded displacements on automatically generated coarse, middle and fine control grids.",
        "hierarchy_rule": "Double target spacing from the base voxel size until every axis has at most four intervals; retain three representative levels including both endpoints. Every level spans the exact rest bounds.",
        "amplitude_rule": "Uniform component bounds share a budget of strength times the shortest rest side, weighted by target spacing to power 1.5. Values are displacement-component half-widths, not vector-norm bounds.",
        "orientation_rule": "Halve a common amplitude scale until combined and individual contributions have volume ratios at least 0.2 for Newton's alternating five tets and trilinear Jacobians at eight corners, eight Gauss points, and the center of every cell.",
        "limitation": "Finite orientation samples do not prove continuous global injectivity or exclude distant self-intersections. These are initial shapes, not simulated equilibria.",
    }
    initial = None
    for seed in seeds:
        sample = generate_multiscale(rest, seed=seed, strength=strength)
        if not catalog["levels"]:
            catalog["levels"] = [asdict(level) for level in sample.levels]
            catalog["modes"] += [
                {"id": level.name, "label": f"{level.name.capitalize()} only"} for level in sample.levels
            ]
        modes = [("combined", sample.positions, sample.screen)] + [
            (level.name, rest.corner_rest_positions + sample.effective_scale * field, screen)
            for level, field, screen in zip(
                sample.levels, sample.level_displacements, sample.level_screens, strict=True
            )
        ]
        for mode, positions, screen in modes:
            result = compute_cell_frames(rest, positions)
            if not result.valid.all():
                raise ValueError(f"Invalid center frame for seed {seed}, mode {mode}")
            amplitudes = [
                level.amplitude_m * sample.effective_scale if mode in ("combined", level.name) else 0.0
                for level in sample.levels
            ]
            displacement = positions - rest.corner_rest_positions
            state = {
                "seed": seed,
                "mode": mode,
                "file": f"states/seed_{seed:02d}_{mode}.json",
                "download": f"samples/seed_{seed:02d}.npz",
                "effective_scale": sample.effective_scale,
                "backtracking_steps": sample.backtracking_steps,
                "effective_amplitudes_m": amplitudes,
                "max_displacement_m": float(np.linalg.norm(displacement, axis=1).max()),
                "rms_displacement_m": float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))),
                **screen,
            }
            catalog["states"].append(state)
            diagnostics = {
                "valid_cell_count": int(result.valid.sum()),
                "invalid_cell_count": 0,
                "min_determinant": float(result.determinants.min()),
                "max_orthogonality_error": float(
                    np.abs(result.frames.transpose(0, 2, 1) @ result.frames - np.eye(3)).max()
                ),
                "max_reconstruction_error": float(np.abs(result.frames @ result.local_axes - result.deformation).max()),
                **screen,
            }
            payload = {
                "metadata": {
                    "cell_counts": list(rest.cell_counts),
                    "cell_size": rest.cell_size,
                    "seed": seed,
                    "mode": mode,
                    "effective_scale": sample.effective_scale,
                    "effective_amplitudes_m": amplitudes,
                    "description": catalog["description"],
                    "strength": strength,
                },
                "positions": positions.round(10).tolist(),
                "centers": result.centers.round(10).tolist(),
                "deformation": result.deformation.round(10).tolist(),
                "frames": result.frames.round(10).tolist(),
                "local_axes": result.local_axes.round(10).tolist(),
                "determinants": result.determinants.round(10).tolist(),
                "valid": result.valid.tolist(),
                "diagnostics": diagnostics,
            }
            _write_json(output / state["file"], payload)
            if initial is None:
                initial = {
                    **payload,
                    "cells": rest.cell_corner_indices.tolist(),
                    "rest_positions": rest.corner_rest_positions.tolist(),
                }
        np.savez_compressed(
            output / f"samples/seed_{seed:02d}.npz",
            seed=seed,
            strength=strength,
            positions=sample.positions,
            rest_positions=rest.corner_rest_positions,
            cell_corner_indices=rest.cell_corner_indices,
            cell_counts=rest.cell_counts,
            cell_size=rest.cell_size,
            level_displacements=sample.level_displacements,
            effective_scale=sample.effective_scale,
            control_counts=np.array([level.control_counts for level in sample.levels]),
            requested_amplitudes_m=np.array([level.amplitude_m for level in sample.levels]),
            **{
                f"controls_{level.name}": control for level, control in zip(sample.levels, sample.controls, strict=True)
            },
        )
    _write_json(output / "catalog.json", catalog)
    (output / "data.js").write_text(
        "window.CELL_FRAME_CATALOG="
        + json.dumps(catalog, separators=(",", ":"), allow_nan=False)
        + ";\n"
        + "window.CELL_FRAME_DATA="
        + json.dumps(initial, separators=(",", ":"), allow_nan=False)
        + ";\n",
        encoding="utf-8",
    )
    return catalog


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated" / "multiscale_frames")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    catalog = write_multiscale_inspector(
        args.output, seeds=tuple(range(args.seed_start, args.seed_start + args.count)), strength=args.strength
    )
    combined = [state for state in catalog["states"] if state["mode"] == "combined"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seeds": len(combined),
                "views": len(catalog["states"]),
                "levels": catalog["levels"],
                "rms_displacement_range_m": [
                    min(s["rms_displacement_m"] for s in combined),
                    max(s["rms_displacement_m"] for s in combined),
                ],
                "min_tet_volume_ratio": min(s["min_tet_volume_ratio"] for s in catalog["states"]),
                "min_sampled_jacobian": min(s["min_sampled_jacobian"] for s in catalog["states"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
