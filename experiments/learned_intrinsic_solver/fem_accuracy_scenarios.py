# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared beam scenarios, schedules, and metrics for the FEM accuracy comparison.

Experimental. Both the learned intrinsic hex solver and Newton's VBD solver run
the same four cantilever scenarios (extension under gravity, uniaxial stretch,
twist, and compression with release) on the training beam. This module holds
everything the two drivers share so their runs are comparable: the beam and
material constants, the clamp and far-face corner sets, the prescribed
far-face motion per scenario, the geometric metrics, and the trajectory file
layout consumed by :mod:`experiments.learned_intrinsic_solver.render_learned`.
The solver drivers themselves live elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .data import VoxelGridData, generate_cuboid

__all__ = [
    "BEAM_LENGTH",
    "BEAM_WIDTH",
    "CELL_COUNTS",
    "CELL_SIZE",
    "DAMPING",
    "DENSITY",
    "FPS",
    "GRAVITY_MAGNITUDE",
    "LAME_LAMBDA",
    "LAME_MU",
    "POISSON_RATIO",
    "SCENARIOS",
    "SUBSTEPS",
    "TIME_STEP",
    "YOUNG_MODULUS",
    "Scenario",
    "Trajectory",
    "beam_rest",
    "bulk_volume_ratio",
    "cell_volumes",
    "centre_jacobian_ratios",
    "clamp_indices",
    "compute_metrics",
    "far_face_indices",
    "frame_times",
    "lateral_contraction",
    "mid_length_indices",
    "read_metrics",
    "read_trajectory",
    "substep_times",
    "tip_displacement",
    "write_metrics",
    "write_trajectory",
]

CELL_COUNTS = (10, 10, 40)
"""Hex cell counts along x, y, and the beam axis z."""

CELL_SIZE = 0.025
"""Rest cell edge length h [m]."""

BEAM_LENGTH = CELL_COUNTS[2] * CELL_SIZE
"""Rest beam length L along z [m]."""

BEAM_WIDTH = CELL_COUNTS[0] * CELL_SIZE
"""Rest cross-section edge length [m]."""

YOUNG_MODULUS = 5.0e5
"""Young's modulus E [Pa]."""

POISSON_RATIO = 0.3
"""Poisson ratio nu."""

LAME_LAMBDA = YOUNG_MODULUS * POISSON_RATIO / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
"""First Lame parameter lambda [Pa]."""

LAME_MU = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
"""Shear modulus mu [Pa]."""

DENSITY = 1000.0
"""Mass density rho [kg/m^3]."""

DAMPING = 100.0
"""Viscous damping coefficient [Pa s]."""

GRAVITY_MAGNITUDE = 9.81
"""Gravitational acceleration g [m/s^2]."""

FPS = 30
"""Recorded frames per second."""

SUBSTEPS = 10
"""Solver substeps per recorded frame."""

TIME_STEP = 1.0 / (FPS * SUBSTEPS)
"""Solver substep dt [s]."""

_PLANE_TOLERANCE = 1e-6
"""Absolute tolerance [m] for selecting corners on a rest plane."""

_TIME_TOLERANCE = 0.5 * TIME_STEP
"""Half a substep; absorbs floating-point drift when comparing schedule times."""

_CORNER_SIGNS = (2 * np.indices((2, 2, 2)).reshape(3, -1).T - 1).astype(np.float64)
"""Reference-cube corner coordinates in generate_cuboid corner order, shape (8, 3)."""

_GAUSS_POINTS = _CORNER_SIGNS / np.sqrt(3.0)
"""Full 2x2x2 Gauss points on [-1, 1]^3 (unit weights), shape (8, 3)."""


def beam_rest() -> VoxelGridData:
    """Return the rest grid of the training beam (0.25 x 0.25 x 1.0 m, 4000 cells)."""
    return generate_cuboid(CELL_COUNTS, cell_size=CELL_SIZE)


def _rest_positions(rest) -> np.ndarray:
    """Return float64 rest corner positions from a grid or an [N, 3] array."""
    positions = rest.corner_rest_positions if isinstance(rest, VoxelGridData) else rest
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("rest positions must have shape [N, 3]")
    return positions


def _plane_indices(rest, z_value: float) -> np.ndarray:
    positions = _rest_positions(rest)
    return np.flatnonzero(np.abs(positions[:, 2] - z_value) <= _PLANE_TOLERANCE)


def clamp_indices(rest) -> np.ndarray:
    """Return the corner indices of the clamped face (rest z = 0)."""
    return _plane_indices(rest, 0.0)


def far_face_indices(rest) -> np.ndarray:
    """Return the corner indices of the driven or free far face (rest z = L)."""
    positions = _rest_positions(rest)
    return _plane_indices(rest, positions[:, 2].max())


def mid_length_indices(rest) -> np.ndarray:
    """Return the corner indices of the cross-section at half the rest length."""
    positions = _rest_positions(rest)
    return _plane_indices(rest, 0.5 * (positions[:, 2].min() + positions[:, 2].max()))


@dataclass(frozen=True)
class Scenario:
    """Describe one beam scenario shared by both solvers.

    Frame ``k`` is recorded at ``k / FPS`` seconds with frame 0 the initial
    state. The far face follows a rigid motion during the first ``ramp_frames``
    frames, is held for ``hold_frames`` frames, and is released (left free)
    for any remaining frames. Scenarios with ``motion == "none"`` never drive
    the far face.

    Attributes:
        name: Registry key and output directory name.
        gravity: Gravity vector [m/s^2].
        frame_count: Number of recorded frames after the initial state.
        ramp_frames: Frames over which the prescribed motion ramps linearly.
        hold_frames: Frames the far face is held after the ramp; ``None`` holds
            to the end of the run.
        motion: ``"none"``, ``"translation"`` along z, or ``"rotation"`` about
            the beam axis through the far-face centroid.
        amplitude: Total translation [m] or rotation angle [rad] at the ramp end.
    """

    name: str
    gravity: tuple[float, float, float]
    frame_count: int
    ramp_frames: int
    hold_frames: int | None
    motion: str
    amplitude: float

    def __post_init__(self):
        if self.motion not in ("none", "translation", "rotation"):
            raise ValueError("motion must be 'none', 'translation', or 'rotation'")
        if self.frame_count < 1 or self.ramp_frames < 0 or (self.hold_frames is not None and self.hold_frames < 0):
            raise ValueError("frame counts must be nonnegative and frame_count positive")
        if self.motion != "none" and self.ramp_frames < 1:
            raise ValueError("driven scenarios need a positive ramp")

    @property
    def ramp_seconds(self) -> float:
        """Return the ramp duration [s]."""
        return self.ramp_frames / FPS

    @property
    def release_frame(self) -> int | None:
        """Return the last frame at which the far face is still driven, or ``None``."""
        if self.motion == "none":
            return None
        if self.hold_frames is None:
            return None
        return self.ramp_frames + self.hold_frames

    def is_driven(self, time_seconds: float) -> bool:
        """Return whether the far face is prescribed at ``time_seconds``."""
        if self.motion == "none":
            return False
        if self.release_frame is None:
            return True
        return float(time_seconds) <= self.release_frame / FPS + _TIME_TOLERANCE

    def ramp_fraction(self, time_seconds: float) -> float:
        """Return ``min(t / T_ramp, 1)`` for the linear ramp."""
        return min(max(float(time_seconds), 0.0) / self.ramp_seconds, 1.0)

    def ramp_rate(self, time_seconds: float) -> float:
        """Return the time derivative of :meth:`ramp_fraction` (zero while holding)."""
        if float(time_seconds) < self.ramp_seconds:
            return 1.0 / self.ramp_seconds
        return 0.0

    def prescribed_far_face(self, rest, time_seconds: float):
        """Return the prescribed far-face state at ``time_seconds``.

        Args:
            rest: Beam rest grid or its [N, 3] rest corner positions.
            time_seconds: Physical time [s].

        Returns:
            ``(driven, positions, velocities)``. ``positions`` and ``velocities``
            are float64 arrays of shape [K, 3] in :func:`far_face_indices` order
            when ``driven`` is true and ``None`` otherwise. Velocities are the
            analytic time derivative of the schedule and vanish while holding.
        """
        if not self.is_driven(time_seconds):
            return False, None, None
        rest_positions = _rest_positions(rest)
        face = rest_positions[far_face_indices(rest_positions)]
        fraction = self.ramp_fraction(time_seconds)
        rate = self.ramp_rate(time_seconds)
        if self.motion == "translation":
            offset = np.array([0.0, 0.0, self.amplitude * fraction])
            positions = face + offset
            velocities = np.broadcast_to(np.array([0.0, 0.0, self.amplitude * rate]), face.shape).copy()
            return True, positions, velocities
        theta = self.amplitude * fraction
        omega = self.amplitude * rate
        centroid = face.mean(axis=0)
        cos, sin = np.cos(theta), np.sin(theta)
        rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        radial = (face - centroid) @ rotation.T
        positions = centroid + radial
        velocities = omega * np.stack([-radial[:, 1], radial[:, 0], np.zeros(len(radial))], axis=1)
        return True, positions, velocities


SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            name="extension",
            gravity=(0.0, 0.0, GRAVITY_MAGNITUDE),
            frame_count=300,
            ramp_frames=0,
            hold_frames=None,
            motion="none",
            amplitude=0.0,
        ),
        Scenario(
            name="stretch",
            gravity=(0.0, 0.0, 0.0),
            frame_count=400,
            ramp_frames=200,
            hold_frames=None,
            motion="translation",
            amplitude=BEAM_LENGTH,
        ),
        Scenario(
            name="twist",
            gravity=(0.0, 0.0, 0.0),
            frame_count=300,
            ramp_frames=200,
            hold_frames=None,
            motion="rotation",
            amplitude=2.0 * np.pi,
        ),
        Scenario(
            name="compression_release",
            gravity=(0.0, 0.0, 0.0),
            frame_count=400,
            ramp_frames=100,
            hold_frames=50,
            motion="translation",
            amplitude=-0.5 * BEAM_LENGTH,
        ),
    )
}
"""Scenario registry keyed by name."""


def frame_times(scenario: Scenario) -> np.ndarray:
    """Return recorded frame times [s], shape [frame_count + 1], starting at zero."""
    return np.arange(scenario.frame_count + 1, dtype=np.float64) / FPS


def substep_times(scenario: Scenario) -> np.ndarray:
    """Return substep end times [s], shape [frame_count, SUBSTEPS].

    Row ``k - 1`` holds the end times of the substeps that advance frame
    ``k - 1`` to frame ``k``; its last entry equals ``frame_times(scenario)[k]``.
    """
    steps = np.arange(1, scenario.frame_count * SUBSTEPS + 1, dtype=np.float64)
    return (steps / (FPS * SUBSTEPS)).reshape(scenario.frame_count, SUBSTEPS)


def _shape_gradients(points: np.ndarray) -> np.ndarray:
    """Return trilinear shape-function gradients d N_a / d xi at reference points.

    Args:
        points: Reference coordinates in [-1, 1]^3, shape [P, 3].

    Returns:
        Gradients of shape [P, 8, 3] in generate_cuboid corner order.
    """
    factors = 1.0 + points[:, None, :] * _CORNER_SIGNS[None, :, :]
    gradients = np.empty((len(points), 8, 3), dtype=np.float64)
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        gradients[:, :, axis] = _CORNER_SIGNS[None, :, axis] * np.prod(factors[:, :, other], axis=-1) / 8.0
    return gradients


def _cell_corners(positions: np.ndarray, cells: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    cells = np.asarray(cells)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    if cells.ndim != 2 or cells.shape[1] != 8:
        raise ValueError("cells must have shape [C, 8]")
    return positions[cells]


def _reference_jacobians(corners: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return d x / d xi at reference points, shape [C, P, 3, 3]."""
    gradients = _shape_gradients(points)
    return np.einsum("cai,paj->cpij", corners, gradients)


def cell_volumes(positions: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return the exact volume [m^3] of each trilinear hexahedron, shape [C].

    The Jacobian determinant of the trilinear map is at most quadratic in each
    reference coordinate, so the full 2x2x2 Gauss rule integrates it exactly.
    Inverted cells report negative volume.
    """
    corners = _cell_corners(positions, cells)
    jacobians = _reference_jacobians(corners, _GAUSS_POINTS)
    return np.linalg.det(jacobians).sum(axis=1)


def centre_jacobian_ratios(positions: np.ndarray, rest_positions: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return det(F) at each cell centre, shape [C]; one at rest.

    ``F`` is the deformation gradient of the trilinear map evaluated at the
    reference centre ``xi = 0``.
    """
    corners = _cell_corners(positions, cells)
    rest_corners = _cell_corners(rest_positions, cells)
    centre = np.zeros((1, 3))
    deformed = np.linalg.det(_reference_jacobians(corners, centre))[:, 0]
    rest = np.linalg.det(_reference_jacobians(rest_corners, centre))[:, 0]
    return deformed / rest


def bulk_volume_ratio(positions: np.ndarray, rest_positions: np.ndarray, cells: np.ndarray) -> float:
    """Return the total deformed volume divided by the total rest volume."""
    return float(cell_volumes(positions, cells).sum() / cell_volumes(rest_positions, cells).sum())


def tip_displacement(positions: np.ndarray, rest, far_indices: np.ndarray):
    """Return the mean far-face z displacement [m] from rest.

    Args:
        positions: Deformed corners, shape [N, 3] or [F, N, 3].
        rest: Beam rest grid or its [N, 3] rest corner positions.
        far_indices: Far-face corner indices.

    Returns:
        A float for one frame or a float64 array of shape [F] for a series.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rest_positions = _rest_positions(rest)
    far_indices = np.asarray(far_indices)
    displacement = positions[..., far_indices, 2].mean(axis=-1) - rest_positions[far_indices, 2].mean()
    return float(displacement) if displacement.ndim == 0 else displacement


def lateral_contraction(positions: np.ndarray, rest, mid_indices: np.ndarray) -> float:
    """Return the mid-length cross-section width divided by its rest width.

    The width is the mean of the x and y extents of the mid-length corners.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rest_positions = _rest_positions(rest)
    mid_indices = np.asarray(mid_indices)
    section = positions[mid_indices, :2]
    rest_section = rest_positions[mid_indices, :2]
    width = (section.max(axis=0) - section.min(axis=0)).mean()
    rest_width = (rest_section.max(axis=0) - rest_section.min(axis=0)).mean()
    return float(width / rest_width)


def analytic_extension_tip_displacement() -> float:
    """Return the small-strain tip extension rho g L^2 / (2 E) [m] of a hanging bar."""
    return DENSITY * GRAVITY_MAGNITUDE * BEAM_LENGTH**2 / (2.0 * YOUNG_MODULUS)


def compute_metrics(scenario: Scenario, positions: np.ndarray, rest, cells: np.ndarray, far_indices, times) -> dict:
    """Return the JSON-serialisable metrics of one recorded trajectory.

    Args:
        scenario: Scenario the trajectory was recorded under.
        positions: Recorded corners, shape [F, N, 3]; frame 0 is the initial state.
        rest: Beam rest grid or its [N, 3] rest corner positions.
        cells: Cell corner indices, shape [C, 8].
        far_indices: Far-face corner indices.
        times: Recorded frame times [s], shape [F].

    Returns:
        A dictionary with the scenario name, recorded frame count, completion
        flag, and the scenario-specific metrics described in the module
        documentation. Truncated trajectories report ``None`` for metrics
        whose frame was not reached.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rest_positions = _rest_positions(rest)
    cells = np.asarray(cells)
    far_indices = np.asarray(far_indices)
    times = np.asarray(times, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[0] < 1 or positions.shape[1:] != rest_positions.shape:
        raise ValueError("positions must have shape [F, N, 3] matching the rest corners")
    if times.shape != (len(positions),):
        raise ValueError("times must hold one entry per recorded frame")
    recorded = len(positions) - 1
    final = positions[-1]

    def min_ratio(frame: np.ndarray) -> float:
        return float(centre_jacobian_ratios(frame, rest_positions, cells).min())

    metrics: dict = {
        "scenario": scenario.name,
        "frame_count": recorded,
        "completed": recorded >= scenario.frame_count,
        "final_time_seconds": float(times[-1]),
    }
    if scenario.name == "extension":
        series = tip_displacement(positions, rest_positions, far_indices)
        metrics.update(
            {
                "tip_displacement_final": float(series[-1]),
                "tip_displacement_series": [float(value) for value in series],
                "series_times": [float(value) for value in times],
                "analytic_tip_displacement": analytic_extension_tip_displacement(),
                "bulk_volume_ratio": bulk_volume_ratio(final, rest_positions, cells),
                "min_centre_jacobian_ratio": min_ratio(final),
            }
        )
    elif scenario.name == "stretch":
        metrics.update(
            {
                "bulk_volume_ratio": bulk_volume_ratio(final, rest_positions, cells),
                "lateral_contraction": lateral_contraction(final, rest_positions, mid_length_indices(rest_positions)),
                "min_centre_jacobian_ratio": min_ratio(final),
            }
        )
    elif scenario.name == "twist":
        peak = scenario.ramp_frames
        reached = recorded >= peak
        metrics.update(
            {
                "peak_frame": peak,
                "bulk_volume_ratio_peak": bulk_volume_ratio(positions[peak], rest_positions, cells)
                if reached
                else None,
                "min_centre_jacobian_ratio_peak": min_ratio(positions[peak]) if reached else None,
                "bulk_volume_ratio_final": bulk_volume_ratio(final, rest_positions, cells),
                "min_centre_jacobian_ratio_final": min_ratio(final),
            }
        )
    elif scenario.name == "compression_release":
        last_driven = min(scenario.release_frame, recorded)
        compression = min(min_ratio(frame) for frame in positions[: last_driven + 1])
        rest_length = rest_positions[far_indices, 2].mean()
        metrics.update(
            {
                "release_frame": scenario.release_frame,
                "min_centre_jacobian_ratio_compression": compression,
                "length_recovery_ratio": float(final[far_indices, 2].mean() / rest_length),
                "bulk_volume_ratio_final": bulk_volume_ratio(final, rest_positions, cells),
            }
        )
    else:
        raise ValueError(f"unknown scenario {scenario.name!r}")
    return metrics


class Trajectory(NamedTuple):
    """Arrays stored in one ``trajectory.npz``."""

    positions: np.ndarray
    """Recorded corners, float32 [F, N, 3]."""

    times: np.ndarray
    """Frame times [s], float64 [F]."""

    rest_positions: np.ndarray
    """Rest corners, float64 [N, 3]."""

    fixed_indices: np.ndarray
    """Clamp corner indices, int64 [P]."""

    cell_corner_indices: np.ndarray
    """Cell corner indices, int64 [C, 8]."""


def write_trajectory(path, positions, times, rest, clamp, cells) -> Path:
    """Write a trajectory in the layout read by ``render_learned._load_trajectory``.

    Only the clamp corners are listed as ``fixed_indices``; the driven far face
    moves and therefore must not be listed.

    Args:
        path: Output ``.npz`` path; parent directories are created.
        positions: Recorded corners, shape [F, N, 3]; stored as float32.
        times: Frame times [s], shape [F], starting at zero.
        rest: Beam rest grid or its [N, 3] rest corner positions.
        clamp: Clamp corner indices.
        cells: Cell corner indices, shape [C, 8].

    Returns:
        The written path.

    Raises:
        ValueError: If shapes are inconsistent, values are nonfinite, times do
            not start at zero and increase, or the clamp corners move.
    """
    path = Path(path)
    positions = np.asarray(positions, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    rest_positions = _rest_positions(rest)
    clamp = np.asarray(clamp, dtype=np.int64)
    cells = np.asarray(cells, dtype=np.int64)
    if positions.ndim != 3 or positions.shape[0] < 1 or positions.shape[1:] != rest_positions.shape:
        raise ValueError("positions must have shape [F, N, 3] matching the rest corners")
    if times.shape != (len(positions),) or abs(times[0]) > 1e-9 or np.any(np.diff(times) <= 0):
        raise ValueError("times must hold one entry per frame, start at zero, and strictly increase")
    if not np.isfinite(positions).all() or not np.isfinite(times).all() or not np.isfinite(rest_positions).all():
        raise ValueError("positions, times, and rest positions must be finite")
    if clamp.ndim != 1 or len(clamp) < 1 or len(np.unique(clamp)) != len(clamp):
        raise ValueError("clamp must be a nonempty unique index vector")
    if cells.ndim != 2 or cells.shape[1] != 8:
        raise ValueError("cells must have shape [C, 8]")
    if (
        np.any(clamp < 0)
        or np.any(clamp >= len(rest_positions))
        or np.any(cells < 0)
        or np.any(cells >= len(rest_positions))
    ):
        raise ValueError("clamp and cell indices must address the rest corners")
    if not np.allclose(positions[:, clamp], positions[0, clamp], rtol=0, atol=1e-5):
        raise ValueError("clamp corners move; only the fixed clamp may be listed")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=positions,
        times=times,
        rest_positions=rest_positions,
        fixed_indices=clamp,
        cell_corner_indices=cells,
    )
    return path


def read_trajectory(path) -> Trajectory:
    """Read a trajectory written by :func:`write_trajectory`."""
    with np.load(Path(path)) as data:
        required = ("positions", "times", "rest_positions", "fixed_indices", "cell_corner_indices")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"trajectory is missing {missing}")
        return Trajectory(
            positions=np.asarray(data["positions"], dtype=np.float32),
            times=np.asarray(data["times"], dtype=np.float64),
            rest_positions=np.asarray(data["rest_positions"], dtype=np.float64),
            fixed_indices=np.asarray(data["fixed_indices"], dtype=np.int64),
            cell_corner_indices=np.asarray(data["cell_corner_indices"], dtype=np.int64),
        )


def write_metrics(path, metrics: dict) -> Path:
    """Write metrics as indented JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def read_metrics(path) -> dict:
    """Read a metrics JSON file."""
    return json.loads(Path(path).read_text())


def scenario_summary(scenario: Scenario) -> dict:
    """Return the scenario fields plus derived timing as a JSON-serialisable dict."""
    summary = asdict(scenario)
    summary.update(
        {
            "gravity": list(scenario.gravity),
            "release_frame": scenario.release_frame,
            "fps": FPS,
            "substeps": SUBSTEPS,
            "time_step": TIME_STEP,
        }
    )
    return summary
