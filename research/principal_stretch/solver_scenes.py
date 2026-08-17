# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Common-objective adaptations of the SolverVBD validation scenes.

The source scenes come from Newton PR #2901.  These builders preserve the
geometry, boundary-condition idea, density, and public Lamé parameters while
making the comparison objective exact: contact and damping are disabled and
the current branch's stored stable-NH lambda is ``lambda_public + mu``.

The driven builders default to the first nontrivial PR schedule increment at
the original atomic substep. Large-deformation and release checkpoints are
history dependent; callers must supply the preceding state rather than
pretending that a one-shot affine field replays the PR trajectory.
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Mapping, Sequence

import newton
import numpy as np
import warp as wp

from .solver_benchmark import TetBenchmarkScene, scene_from_model

_PR_REVISION = "a513d446e42477a8ada78070f92ffb60d3108eeb"
_PR_SOURCE = "newton PR #2901 VBD validation examples"


def _validated_dimensions(values: Sequence[int]) -> tuple[int, int, int]:
    raw = tuple(values)
    if len(raw) != 3 or any(
        isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0 for value in raw
    ):
        raise ValueError("dimensions must contain three positive integers")
    return tuple(int(value) for value in raw)


def _stored_lambda(mu: float, public_lambda: float) -> float:
    """Convert PR/public small-strain Lamé lambda to this branch's storage."""
    return float(mu + public_lambda)


def _build_grid_model(
    dimensions: Sequence[int],
    cell_size: Sequence[float],
    origin: Sequence[float],
    *,
    density: float,
    mu: float,
    public_lambda: float,
    gravity: Sequence[float],
    fix_left: bool = False,
) -> newton.Model:
    dims = _validated_dimensions(dimensions)
    cells = tuple(float(value) for value in cell_size)
    start = tuple(float(value) for value in origin)
    if len(cells) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in cells):
        raise ValueError("cell_size must contain three finite positive values")
    if len(start) != 3 or not np.isfinite(start).all():
        raise ValueError("origin must contain three finite values")
    if not np.isfinite((density, mu, public_lambda)).all() or min(density, mu, public_lambda) <= 0.0:
        raise ValueError("density and material parameters must be finite and positive")

    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_soft_grid(
        pos=wp.vec3(*start),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dims[0],
        dim_y=dims[1],
        dim_z=dims[2],
        cell_x=cells[0],
        cell_y=cells[1],
        cell_z=cells[2],
        density=density,
        k_mu=mu,
        k_lambda=_stored_lambda(mu, public_lambda),
        k_damp=0.0,
        fix_left=fix_left,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        add_surface_mesh_edges=False,
        particle_radius=0.0,
    )
    builder.color()
    model = builder.finalize(device="cpu")
    model.set_gravity(gravity)
    return model


def _set_inactive(model: newton.Model, mask: np.ndarray) -> None:
    if mask.shape != (model.particle_count,):
        raise ValueError("pin mask has the wrong shape")
    flags = model.particle_flags.numpy().copy()
    flags[mask] &= ~int(newton.ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags, dtype=wp.int32, device=model.device)


def _metadata(
    scene: str,
    *,
    dimensions: Sequence[int],
    cell_size: Sequence[float],
    density: float,
    mu: float,
    public_lambda: float,
    pr_substeps: int,
    pr_iterations: int,
    pr_damping: float,
    state_kind: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "adapted_from": _PR_SOURCE,
        "source_revision": _PR_REVISION,
        "source_scene": scene,
        "dimensions_cells": list(dimensions),
        "cell_size_m": list(cell_size),
        "density_kg_m3": density,
        "mu_public_pa": mu,
        "lambda_public_pa": public_lambda,
        "lambda_stored_pa": _stored_lambda(mu, public_lambda),
        "coefficient_convention": "a727e58c-stored-lambda-equals-public-lambda-plus-mu",
        "state_kind": state_kind,
        "common_objective_adaptations": [
            "set tet damping to zero",
            "omit contact and ground plane",
            "use one implicit step at manifest dt",
            "retain zero-energy boundary triangles for CUDA tiled SolverVBD",
        ],
        "pr_original_substeps_per_frame": pr_substeps,
        "pr_original_frame_rate_hz": 60,
        "pr_original_substep_dt_seconds": 1.0 / (60.0 * pr_substeps),
        "pr_original_vbd_iterations_per_substep": pr_iterations,
        "pr_original_tet_damping": pr_damping,
    }
    if extra is not None:
        value.update(extra)
    return value


def build_extension_scene(
    *,
    dim_xy: int = 4,
    dim_z: int = 20,
    cell: float = 0.05,
    dt: float = 1.0 / 360.0,
) -> TetBenchmarkScene:
    """Build the PR hanging-beam extension scene on a common objective."""
    dimensions = (dim_xy, dim_xy, dim_z)
    cells = (cell, cell, cell)
    density = 1000.0
    mu = 5.0e4
    public_lambda = 5.0e4
    origin = (0.0, 0.0, 2.0)
    model = _build_grid_model(
        dimensions,
        cells,
        origin,
        density=density,
        mu=mu,
        public_lambda=public_lambda,
        gravity=(0.0, 0.0, -9.81),
    )
    rest = model.particle_q.numpy()
    top_z = origin[2] + dim_z * cell
    top = np.isclose(rest[:, 2], top_z, rtol=0.0, atol=1.0e-6)
    _set_inactive(model, top)
    return scene_from_model(
        model,
        name=f"pr2901-extension-{dim_xy}x{dim_xy}x{dim_z}-common-step",
        source="newton/examples/vbd/example_soft_beam_extension.py",
        dt=dt,
        metadata=_metadata(
            "soft_beam_extension",
            dimensions=dimensions,
            cell_size=cells,
            density=density,
            mu=mu,
            public_lambda=public_lambda,
            pr_substeps=6,
            pr_iterations=100,
            pr_damping=0.1,
            state_kind="rest-state gravity substep with fixed top face",
            extra={"pr_schedule_frame_index": 0, "pr_schedule_substep_index": 0},
        ),
    )


def build_stretch_scene(
    *,
    dimensions: Sequence[int] = (10, 3, 3),
    cell: float = 0.05,
    stretch_ratio: float = 1.0 + 1.0 / 200.0,
    one_shot_diagnostic: bool = False,
    dt: float = 1.0 / 300.0,
) -> TetBenchmarkScene:
    """Build one driven-boundary increment of the PR 2x beam stretch."""
    if not isinstance(one_shot_diagnostic, bool):
        raise ValueError("one_shot_diagnostic must be a bool")
    if not np.isfinite(stretch_ratio) or stretch_ratio <= 0.0:
        raise ValueError("stretch_ratio must be finite and positive")
    default_ratio = 1.0 + 1.0 / 200.0
    if stretch_ratio != default_ratio and not one_shot_diagnostic:
        raise ValueError("a non-default stretch target requires audited history or one_shot_diagnostic=True")
    dims = _validated_dimensions(dimensions)
    cells = (cell, cell, cell)
    density = 1000.0
    mu = 1.0e4
    public_lambda = 1.0e5
    origin = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    model = _build_grid_model(
        dims,
        cells,
        origin,
        density=density,
        mu=mu,
        public_lambda=public_lambda,
        gravity=(0.0, 0.0, 0.0),
        fix_left=True,
    )
    rest = model.particle_q.numpy().astype(np.float64)
    length = dims[0] * cell
    left = np.isclose(rest[:, 0], origin[0], rtol=0.0, atol=1.0e-6)
    right = np.isclose(rest[:, 0], origin[0] + length, rtol=0.0, atol=1.0e-6)
    _set_inactive(model, right)
    targets = rest.copy()
    targets[right, 0] = origin[0] + stretch_ratio * length
    return scene_from_model(
        model,
        name=f"pr2901-stretch-{stretch_ratio:g}x-{dims[0]}x{dims[1]}x{dims[2]}-boundary-step",
        source="newton/examples/vbd/example_soft_beam_stretch.py",
        dt=dt,
        x_current=rest,
        pin_targets=targets[left | right],
        metadata=_metadata(
            "soft_beam_stretch",
            dimensions=dims,
            cell_size=cells,
            density=density,
            mu=mu,
            public_lambda=public_lambda,
            pr_substeps=5,
            pr_iterations=20,
            pr_damping=1.0e-3,
            state_kind=(
                "one-shot stretch diagnostic from rest; not a PR trajectory checkpoint"
                if one_shot_diagnostic
                else "rest-state driven-right-boundary increment"
            ),
            extra={
                "stretch_ratio": stretch_ratio,
                "one_shot_diagnostic": one_shot_diagnostic,
                "default_is_first_nontrivial_pr_increment": stretch_ratio == default_ratio,
                "pr_schedule_frame_index": 1 if stretch_ratio == default_ratio else None,
                "pr_schedule_substep_index": 0 if stretch_ratio == default_ratio else None,
                "pr_original_ramp_frames": 200,
            },
        ),
    )


def build_twist_scene(
    *,
    dimensions: Sequence[int] = (3, 3, 16),
    cell: float = 0.05,
    twist_angle: float = 2.0 * np.pi / 200.0,
    one_shot_diagnostic: bool = False,
    dt: float = 1.0 / 300.0,
) -> TetBenchmarkScene:
    """Build one driven-boundary increment of the PR 360-degree twist."""
    if not isinstance(one_shot_diagnostic, bool):
        raise ValueError("one_shot_diagnostic must be a bool")
    if not np.isfinite(twist_angle):
        raise ValueError("twist_angle must be finite")
    if abs(twist_angle) > 1.0e-12 and abs(np.sin(0.5 * twist_angle)) < 1.0e-8:
        raise ValueError("a full-turn endpoint needs a history-bearing current state")
    default_angle = 2.0 * np.pi / 200.0
    if twist_angle != default_angle and not one_shot_diagnostic:
        raise ValueError("a non-default twist target requires audited history or one_shot_diagnostic=True")
    dims = _validated_dimensions(dimensions)
    cells = (cell, cell, cell)
    density = 1000.0
    mu = 1.0e4
    public_lambda = 1.0e4
    origin = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    model = _build_grid_model(
        dims,
        cells,
        origin,
        density=density,
        mu=mu,
        public_lambda=public_lambda,
        gravity=(0.0, 0.0, 0.0),
    )
    rest = model.particle_q.numpy().astype(np.float64)
    height = dims[2] * cell
    bottom = np.isclose(rest[:, 2], origin[2], rtol=0.0, atol=1.0e-6)
    top = np.isclose(rest[:, 2], origin[2] + height, rtol=0.0, atol=1.0e-6)
    _set_inactive(model, bottom | top)
    center = origin[:2] + 0.5 * np.array((dims[0] * cell, dims[1] * cell))
    relative = rest[top, :2] - center[None, :]
    cosine = np.cos(twist_angle)
    sine = np.sin(twist_angle)
    targets = rest.copy()
    targets[top, 0] = center[0] + cosine * relative[:, 0] - sine * relative[:, 1]
    targets[top, 1] = center[1] + sine * relative[:, 0] + cosine * relative[:, 1]
    return scene_from_model(
        model,
        name=f"pr2901-twist-{np.degrees(twist_angle):g}deg-{dims[0]}x{dims[1]}x{dims[2]}-boundary-step",
        source="newton/examples/vbd/example_soft_beam_twist.py",
        dt=dt,
        x_current=rest,
        pin_targets=targets[bottom | top],
        metadata=_metadata(
            "soft_beam_twist",
            dimensions=dims,
            cell_size=cells,
            density=density,
            mu=mu,
            public_lambda=public_lambda,
            pr_substeps=5,
            pr_iterations=20,
            pr_damping=1.0e-3,
            state_kind=(
                "one-shot twist diagnostic from rest; not a PR trajectory checkpoint"
                if one_shot_diagnostic
                else "rest-state driven-top-boundary twist increment"
            ),
            extra={
                "twist_angle_rad": twist_angle,
                "one_shot_diagnostic": one_shot_diagnostic,
                "default_is_first_nontrivial_pr_increment": twist_angle == default_angle,
                "pr_schedule_frame_index": 1 if twist_angle == default_angle else None,
                "pr_schedule_substep_index": 0 if twist_angle == default_angle else None,
                "pr_original_ramp_frames": 200,
            },
        ),
    )


def build_compression_scene(
    *,
    dim: int = 6,
    cell: float = 0.05,
    compression_ratio: float = 1.0 - 0.5 / 149.0,
    released: bool = False,
    one_shot_diagnostic: bool = False,
    dt: float = 1.0 / 300.0,
) -> TetBenchmarkScene:
    """Build one driven-boundary increment of the PR compression ramp.

    Release is history dependent and deliberately rejected here. A release
    checkpoint must come from an audited trajectory chain, not a caller-made
    affine state with an unverified provenance label.
    """
    if not isinstance(released, bool) or not isinstance(one_shot_diagnostic, bool):
        raise ValueError("released and one_shot_diagnostic must be bools")
    if not np.isfinite(compression_ratio) or not 0.0 < compression_ratio <= 1.0:
        raise ValueError("compression_ratio must lie in (0, 1]")
    if released:
        raise ValueError("released compression requires an audited trajectory checkpoint")
    default_ratio = 1.0 - 0.5 / 149.0
    if compression_ratio != default_ratio and not one_shot_diagnostic:
        raise ValueError("a non-default compression target requires audited history or one_shot_diagnostic=True")
    if isinstance(dim, bool) or not isinstance(dim, numbers.Integral) or dim <= 0:
        raise ValueError("dim must be a positive integer")
    dim = int(dim)
    dimensions = (dim, dim, dim)
    cells = (cell, cell, cell)
    density = 1000.0
    mu = 1.0e4
    public_lambda = 1.0e4
    origin = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    model = _build_grid_model(
        dimensions,
        cells,
        origin,
        density=density,
        mu=mu,
        public_lambda=public_lambda,
        gravity=(0.0, 0.0, 0.0),
    )
    rest = model.particle_q.numpy().astype(np.float64)
    height = dim * cell
    bottom = np.isclose(rest[:, 2], origin[2], rtol=0.0, atol=1.0e-6)
    top = np.isclose(rest[:, 2], origin[2] + height, rtol=0.0, atol=1.0e-6)
    _set_inactive(model, bottom | top)
    targets = rest.copy()
    targets[top, 2] = origin[2] + compression_ratio * height
    return scene_from_model(
        model,
        name=f"pr2901-compression-{compression_ratio:g}-driven-{dim}cubed-step",
        source="newton/examples/vbd/example_soft_cube_compression.py",
        dt=dt,
        x_current=rest,
        pin_targets=targets[bottom | top],
        metadata=_metadata(
            "soft_cube_compression",
            dimensions=dimensions,
            cell_size=cells,
            density=density,
            mu=mu,
            public_lambda=public_lambda,
            pr_substeps=5,
            pr_iterations=30,
            pr_damping=1.0e-2,
            state_kind=(
                "one-shot compression diagnostic from rest; not a PR trajectory checkpoint"
                if one_shot_diagnostic
                else "rest-state driven-top-boundary compression increment"
            ),
            extra={
                "compression_ratio": compression_ratio,
                "released": False,
                "one_shot_diagnostic": one_shot_diagnostic,
                "default_is_first_nontrivial_pr_increment": compression_ratio == default_ratio,
                "pr_schedule_frame_index": 1 if compression_ratio == default_ratio else None,
                "pr_schedule_substep_index": 0 if compression_ratio == default_ratio else None,
                "pr_original_compress_frames": 150,
                "pr_original_settle_frames": 250,
            },
        ),
    )


def build_sliver_scene(*, dt: float = 1.0 / 360.0) -> TetBenchmarkScene:
    """Build the PR 10:1-cell sliver beam under gravity."""
    dimensions = (2, 2, 10)
    cells = (0.2, 0.2, 0.02)
    density = 1000.0
    mu = 5.0e4
    public_lambda = 5.0e4
    origin = (0.0, 0.0, 2.0)
    model = _build_grid_model(
        dimensions,
        cells,
        origin,
        density=density,
        mu=mu,
        public_lambda=public_lambda,
        gravity=(0.0, 0.0, -9.81),
    )
    rest = model.particle_q.numpy()
    top_z = origin[2] + dimensions[2] * cells[2]
    top = np.isclose(rest[:, 2], top_z, rtol=0.0, atol=1.0e-6)
    _set_inactive(model, top)
    return scene_from_model(
        model,
        name="pr2901-sliver-10to1-common-step",
        source="newton/examples/vbd/example_soft_sliver_elements.py",
        dt=dt,
        metadata=_metadata(
            "soft_sliver_elements",
            dimensions=dimensions,
            cell_size=cells,
            density=density,
            mu=mu,
            public_lambda=public_lambda,
            pr_substeps=6,
            pr_iterations=20,
            pr_damping=0.1,
            state_kind="rest-state gravity substep with fixed top face",
            extra={
                "cell_aspect_ratio": 10.0,
                "pr_schedule_frame_index": 0,
                "pr_schedule_substep_index": 0,
            },
        ),
    )


def build_refinement_scene(level: str, *, dt: float = 1.0 / 360.0) -> TetBenchmarkScene:
    """Build one PR same-domain hanging-beam refinement level."""
    configurations = {
        "coarse": (2, 10, 0.1),
        "medium": (4, 20, 0.05),
        "fine": (8, 40, 0.025),
    }
    if level not in configurations:
        raise ValueError(f"unknown refinement level {level!r}; expected one of {tuple(configurations)}")
    dim_xy, dim_z, cell = configurations[level]
    scene = build_extension_scene(dim_xy=dim_xy, dim_z=dim_z, cell=cell, dt=dt)
    metadata = dict(scene.metadata)
    metadata.update(
        {
            "source_scene": "soft_convergence_refinement",
            "refinement_level": level,
            "pr_original_vbd_iterations_per_substep": 5 * dim_z,
        }
    )
    return dataclasses.replace(
        scene,
        name=f"pr2901-refinement-{level}-common-step",
        source="newton/examples/vbd/example_soft_convergence_refinement.py",
        metadata=metadata,
    )
