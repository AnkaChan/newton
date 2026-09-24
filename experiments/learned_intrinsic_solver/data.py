# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate experimental voxel-cell inputs without running a physics solver.

This entire module is experimental and may change without API compatibility.
Nodes are cells. Independently augmented cell fields do not imply a compatible
deformed corner mesh or a shared-corner velocity field.
"""

import argparse
from dataclasses import dataclass, fields, replace
from math import prod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["VoxelGridData", "augment_grid", "generate_cuboid"]


@dataclass(frozen=True)
class VoxelGridData:
    """Hold shared rest geometry and independent cell fields.

    Experimental: this container and its fields may change without notice.
    Arrays use float64 for physical values and int64 for indices. Cell and
    corner indices run in x/y/z order with z varying fastest. A cell's eight
    local corners follow the same order over offsets in {0, 1}.
    """

    cell_counts: tuple[int, int, int]
    """Number of cells along each rest-material axis."""

    cell_size: float
    """Uniform rest edge length [m]."""

    corner_rest_positions: NDArray[np.float64]
    """Shared rest corner positions [m], shape (corner_count, 3)."""

    cell_corner_indices: NDArray[np.int64]
    """Shared corner indices, shape (cell_count, 8)."""

    cell_rest_centers: NDArray[np.float64]
    """Rest cell centers [m], shape (cell_count, 3)."""

    cell_neighbors: NDArray[np.int64]
    """Face neighbors in (-x, +x, -y, +y, -z, +z) order; -1 marks an exposed face."""

    cell_deformation: NDArray[np.float64]
    """Dimensionless transformed material axes as columns, shape (cell_count, 3, 3)."""

    cell_velocity: NDArray[np.float64]
    """Cell velocities in the common coordinate frame [m/s], shape (cell_count, 3)."""

    @property
    def cell_exposed_faces(self) -> NDArray[np.bool_]:
        """Return exposed-face flags in fixed material order, shape (cell_count, 6)."""
        return self.cell_neighbors == -1


def generate_cuboid(
    cell_counts: tuple[int, int, int],
    *,
    cell_size: float = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VoxelGridData:
    """Generate a full cuboid of cubic voxels with shared rest corners.

    Experimental: the signature and returned data schema may change.

    Args:
        cell_counts: Positive integer cell counts along x, y, and z.
        cell_size: Common rest voxel edge length [m].
        origin: Minimum rest corner position [m].

    Returns:
        Grid data with identity deformation axes and zero cell velocities.

    Raises:
        ValueError: If counts, spacing, or origin are invalid.
    """
    try:
        counts = tuple(cell_counts)
    except TypeError as error:
        raise ValueError("cell_counts must contain three positive integers") from error
    if len(counts) != 3 or any(
        isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0
        for count in counts
    ):
        raise ValueError("cell_counts must contain three positive integers")
    counts = tuple(int(count) for count in counts)
    corner_counts = tuple(count + 1 for count in counts)
    if prod(corner_counts) > np.iinfo(np.int64).max:
        raise ValueError("cell_counts exceed the supported index range")

    size = float(cell_size)
    origin_array = np.asarray(origin, dtype=np.float64)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("cell_size must be finite and positive")
    if origin_array.shape != (3,) or not np.isfinite(origin_array).all():
        raise ValueError("origin must contain three finite coordinates")

    cell_coordinates = np.indices(counts, dtype=np.int64).reshape(3, -1).T
    corner_coordinates = np.indices(corner_counts, dtype=np.int64).reshape(3, -1).T
    local_corners = np.indices((2, 2, 2), dtype=np.int64).reshape(3, -1).T
    corner_strides = np.array([corner_counts[1] * corner_counts[2], corner_counts[2], 1], dtype=np.int64)
    cell_corner_indices = (cell_coordinates[:, None, :] + local_corners[None, :, :]) @ corner_strides

    with np.errstate(over="ignore", invalid="ignore"):
        corner_positions = origin_array + size * corner_coordinates
        cell_centers = origin_array + size * (cell_coordinates + 0.5)
    if not np.isfinite(corner_positions).all() or not np.isfinite(cell_centers).all():
        raise ValueError("grid coordinates exceed the supported numeric range")

    cell_count = prod(counts)
    cell_indices = np.arange(cell_count, dtype=np.int64)
    cell_strides = (counts[1] * counts[2], counts[2], 1)
    neighbors = np.full((cell_count, 6), -1, dtype=np.int64)
    for axis, stride in enumerate(cell_strides):
        lower = cell_coordinates[:, axis] > 0
        upper = cell_coordinates[:, axis] < counts[axis] - 1
        neighbors[lower, 2 * axis] = cell_indices[lower] - stride
        neighbors[upper, 2 * axis + 1] = cell_indices[upper] + stride

    return VoxelGridData(
        cell_counts=counts,
        cell_size=size,
        corner_rest_positions=corner_positions,
        cell_corner_indices=cell_corner_indices,
        cell_rest_centers=cell_centers,
        cell_neighbors=neighbors,
        cell_deformation=np.broadcast_to(np.eye(3), (cell_count, 3, 3)).copy(),
        cell_velocity=np.zeros((cell_count, 3), dtype=np.float64),
    )


def _validate_state(deformation: np.ndarray, velocity: np.ndarray, cell_count: int) -> None:
    if deformation.shape != (cell_count, 3, 3) or not np.isfinite(deformation).all():
        raise ValueError("cell_deformation must contain finite 3-by-3 matrices, one per cell")
    signs, log_determinants = np.linalg.slogdet(deformation)
    if not np.all(signs > 0.0) or not np.isfinite(log_determinants).all():
        raise ValueError("cell_deformation must have positive, nonsingular determinants")
    if velocity.shape != (cell_count, 3) or not np.isfinite(velocity).all():
        raise ValueError("cell_velocity must contain three finite components per cell")


def augment_grid(
    grid: VoxelGridData,
    *,
    deformation_amplitude: float = 0.1,
    velocity_amplitude: float = 1.0,
    seed: int | None = None,
) -> VoxelGridData:
    """Copy a grid and independently perturb each cell's axes and velocity.

    Experimental: the augmentation distribution and interface may change.
    Deformation increments are I + E with independent uniform entries in E.
    Apply each increment on the left: F_new = (I + E) @ F_old. Velocities
    receive independent uniform component increments. Neither operation
    modifies rest geometry or reconstructs current shared-corner positions.

    Args:
        grid: Source sample with finite velocities and valid deformation matrices.
        deformation_amplitude: Dimensionless entry bound a for E, with 0 <= a < 1/3.
        velocity_amplitude: Nonnegative velocity increment bound per component [m/s].
        seed: Seed for a local NumPy generator; does not change global random state.

    Returns:
        A sample with independent arrays. Zero amplitudes preserve existing fields.

    Raises:
        ValueError: If amplitudes or source fields are invalid, or results are nonfinite.
    """
    deformation_amplitude = float(deformation_amplitude)
    velocity_amplitude = float(velocity_amplitude)
    if not np.isfinite(deformation_amplitude) or not 0.0 <= deformation_amplitude < 1.0 / 3.0:
        raise ValueError("deformation_amplitude must be finite and in [0, 1/3)")
    if not np.isfinite(velocity_amplitude) or velocity_amplitude < 0.0:
        raise ValueError("velocity_amplitude must be finite and nonnegative")

    cell_count = len(grid.cell_corner_indices)
    deformation = np.asarray(grid.cell_deformation, dtype=np.float64)
    velocity = np.asarray(grid.cell_velocity, dtype=np.float64)
    _validate_state(deformation, velocity, cell_count)
    rng = np.random.default_rng(seed)
    # ||E||_2 <= ||E||_F <= 3*a < 1 keeps I + E nonsingular and orientation preserving.
    increments = np.eye(3) + rng.uniform(-deformation_amplitude, deformation_amplitude, deformation.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        new_deformation = increments @ deformation
        new_velocity = velocity + rng.uniform(-velocity_amplitude, velocity_amplitude, velocity.shape)
    _validate_state(new_deformation, new_velocity, cell_count)
    return replace(
        grid,
        corner_rest_positions=grid.corner_rest_positions.copy(),
        cell_corner_indices=grid.cell_corner_indices.copy(),
        cell_rest_centers=grid.cell_rest_centers.copy(),
        cell_neighbors=grid.cell_neighbors.copy(),
        cell_deformation=new_deformation,
        cell_velocity=new_velocity,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate augmented cuboid-cell test inputs as a NumPy archive.")
    parser.add_argument("--cells", type=int, nargs=3, default=(4, 3, 2), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--cell-size", type=float, default=1.0, help="Rest voxel edge length in meters.")
    parser.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--deformation-amplitude", type=float, default=0.1, help="Matrix-entry bound in [0, 1/3).")
    parser.add_argument(
        "--velocity-amplitude", type=float, default=1.0, help="Velocity component increment bound in m/s."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Destination .npz file.")
    args = parser.parse_args()

    try:
        grid = generate_cuboid(tuple(args.cells), cell_size=args.cell_size, origin=tuple(args.origin))
        grid = augment_grid(
            grid,
            deformation_amplitude=args.deformation_amplitude,
            velocity_amplitude=args.velocity_amplitude,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as output:
            np.savez_compressed(
                output,
                **{field.name: getattr(grid, field.name) for field in fields(grid)},
                cell_exposed_faces=grid.cell_exposed_faces,
                seed=args.seed,
                deformation_amplitude=args.deformation_amplitude,
                velocity_amplitude=args.velocity_amplitude,
            )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(
        f"Wrote {len(grid.cell_corner_indices)} cells and {len(grid.corner_rest_positions)} shared corners to {args.output}"
    )


if __name__ == "__main__":
    _main()
