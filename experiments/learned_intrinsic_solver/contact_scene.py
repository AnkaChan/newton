# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental contact partners, scene sampling and brute-force pair detection.

Version 1 of the learned intrinsic solver's contact handling
(``notes/contact-design-20260927.md``, sections 3.2, 3.3, 4 and 7) pairs the
body's exposed-face samples with static partners: one optional ground plane
and up to 64 artificial static contact points with normals and a lateral
radius of influence. Static points are drawn outside the rest body (one cell
of clearance) and act as one-sided disks: a sample whose center lies more than
its own radius behind the disk has passed through it and is not a candidate,
so a point can never pull a far face of the body toward it. A point is
rejected and redrawn while its disk would be a detection candidate against the
rest surface samples or its normal does not oppose the nearest exposed face
within :data:`POINT_NORMAL_MAX_ANGLE` degrees, and no point lies in front of
the clamped z-minimum face, so no disk slices the rest body. The detector can
in turn drop pairs whose partner normal does not oppose the sample's face
normal. This module owns the per-scene partner record
(:class:`ContactPartners`), the seeded scene generator
(:func:`sample_contact_partners`), the frozen per-step pair list
(:class:`ContactPairs`) and the CPU detector (:func:`detect_contacts`).

Sampling uses NumPy; detection reads and writes CPU ``torch`` tensors so the
result can be handed to the energy and network input code without further
conversion. Every numeric range in :func:`sample_contact_partners` is
provisional, as recorded in section 7 of the design note.

Amendment 2026-09-29 (design note, section 7): the sampled stiffness
``ke = kappa * E * h`` may be raised to a load-based floor
:func:`contact_stiffness_floor`, ``m g / (n_face * d_max)``, so a body at rest
on its largest flat face penetrates at most ``d_max`` (a fraction of the sample
radius) under its own weight. The floor is off unless the caller passes the
body's density, the gravity magnitude and ``static_penetration_max``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch  # noqa: TID253 -- Explicit opt-in PyTorch implementation.
from torch import Tensor  # noqa: TID253

from .contact_geometry import exposed_face_samples, sample_points
from .data import VoxelGridData

__all__ = [
    "KIND_PLANE",
    "KIND_POINT",
    "KIND_SELF",
    "PLANE_PARTNER_RADIUS",
    "POINT_BOX_DEPTH_MARGIN",
    "POINT_BOX_LATERAL_MARGIN",
    "POINT_CLEARANCE_CELLS",
    "POINT_NORMAL_MAX_ANGLE",
    "POINT_PLACEMENT_ATTEMPTS",
    "POINT_REJECTION_SAMPLE_RADIUS_CELLS",
    "ContactPairs",
    "ContactPartners",
    "contact_stiffness_floor",
    "detect_contacts",
    "sample_contact_partners",
]

KIND_PLANE = 0
"""Pair kind of the ground plane."""

KIND_POINT = 1
"""Pair kind of an artificial static contact point."""

KIND_SELF = 2
"""Pair kind reserved for self-contact; never produced in version 1."""

PLANE_PARTNER_RADIUS = 1e9
"""Large finite lateral radius [m] standing in for the plane's infinite extent."""

POINT_BOX_LATERAL_MARGIN = 0.10
"""Extension of the static point box beyond the rest body along x and z [m]."""

POINT_BOX_DEPTH_MARGIN = 0.35
"""Extension of the static point box below the rest body along -y [m]."""

POINT_CLEARANCE_CELLS = 1.0
"""Clearance between static points and the rest bounding box in units of ``cell_size``.

The rest bounding box grown by this margin is excluded from the static point
box, so a point starts at least one cell (two sample radii) away from every
rest surface sample.
"""

POINT_NORMAL_MAX_ANGLE = 60.0
"""Largest angle [deg] between a static point normal and the inward normal of its nearest rest sample.

The disk must face the exposed face it is closest to instead of grazing it.
"""

POINT_REJECTION_SAMPLE_RADIUS_CELLS = 0.5
"""Sample radius ``r`` used by the rest-shape rejection test, in units of ``cell_size``.

A candidate point is redrawn while any rest surface sample lies within its
disk band ``-r <= gap < r + cell_size``; the one-cell margin matches
:data:`POINT_CLEARANCE_CELLS`.
"""

POINT_PLACEMENT_ATTEMPTS = 1000
"""Position draws allowed per static point before :func:`sample_contact_partners` raises."""

_PLANE_NORMAL = (0.0, 1.0, 0.0)
_UNIT_NORMAL_TOLERANCE = 1e-3
_NORMAL_ATTEMPTS_PER_POSITION = 8


def _as_float32(value: Any, *, shape: tuple[int, ...], name: str) -> Tensor:
    """Coerce ``value`` to a finite CPU float32 tensor with the requested shape.

    A ``-1`` in ``shape`` accepts any extent along that axis, including zero.
    """
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().to("cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be convertible to a float32 tensor") from error
    if tensor.numel() == 0 and -1 in shape:
        tensor = tensor.reshape(shape)
    if tensor.ndim != len(shape) or any(
        expected != -1 and actual != expected for expected, actual in zip(shape, tensor.shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor.contiguous()


def _finite_scalar(value: Any, *, name: str, minimum: float | None = None, strict: bool = False) -> float:
    """Return ``value`` as a finite float, optionally bounded below."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if minimum is not None and (result < minimum or (strict and result <= minimum)):
        comparison = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {comparison} {minimum}")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    """Return ``value`` as a nonnegative Python integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _ordered_range(value: Any, *, name: str, minimum: float | None = None, strict: bool = False) -> tuple[float, float]:
    """Return ``value`` as an ordered pair of finite floats."""
    try:
        lower, upper = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an ordered (lower, upper) pair") from error
    lower = _finite_scalar(lower, name=f"{name}[0]", minimum=minimum, strict=strict)
    upper = _finite_scalar(upper, name=f"{name}[1]", minimum=minimum, strict=strict)
    if lower > upper:
        raise ValueError(f"{name} must be an ordered (lower, upper) pair")
    return lower, upper


def _check_unit_normals(normals: Tensor, *, name: str) -> None:
    """Reject normals whose length deviates noticeably from one."""
    if normals.numel() == 0:
        return
    norms = torch.linalg.vector_norm(normals.double(), dim=-1)
    if not torch.all((norms - 1.0).abs() <= _UNIT_NORMAL_TOLERANCE):
        raise ValueError(f"{name} must contain unit vectors")


@dataclass(eq=False)
class ContactPartners:
    """Static contact partners of one scene: an optional floor and static points.

    Experimental. All tensors are CPU float32. The plane normal is
    ``(0, 1, 0)`` because gravity acts along -y. ``plane_point`` is kept even
    when ``plane_present`` is False so the sampled height stays visible in the
    payload; detection ignores it in that case. Every version-1 partner has
    self flag 0.
    """

    plane_present: bool
    """Whether the ground plane takes part in detection."""

    plane_point: Tensor
    """A point on the ground plane [m], shape [3]."""

    plane_normal: Tensor
    """Unit plane normal, shape [3]; ``(0, 1, 0)`` in version 1."""

    point_positions: Tensor
    """Static contact point positions [m], shape [N, 3]."""

    point_normals: Tensor
    """Unit normals of the static points, shape [N, 3]."""

    point_radii: Tensor
    """Lateral radius of influence ``r_p`` of each static point [m], shape [N]."""

    ke: float
    """Contact stiffness [N/m]."""

    kd: float
    """Contact damping [N·s/m]."""

    mu: float
    """Friction coefficient."""

    ke_floor: float = 0.0
    """Load-based stiffness floor [N/m] of :func:`contact_stiffness_floor`; 0 when the floor was disabled."""

    floor_bound: bool = False
    """Whether ``ke`` was raised to ``ke_floor`` (the sampled ``kappa * E * h`` fell below it)."""

    def __post_init__(self) -> None:
        """Coerce fields to CPU float32 tensors and validate shapes and ranges."""
        if not isinstance(self.plane_present, (bool, np.bool_)):
            raise ValueError("plane_present must be a bool")
        self.plane_present = bool(self.plane_present)
        self.plane_point = _as_float32(self.plane_point, shape=(3,), name="plane_point")
        self.plane_normal = _as_float32(self.plane_normal, shape=(3,), name="plane_normal")
        _check_unit_normals(self.plane_normal, name="plane_normal")
        self.point_positions = _as_float32(self.point_positions, shape=(-1, 3), name="point_positions")
        self.point_normals = _as_float32(self.point_normals, shape=(-1, 3), name="point_normals")
        self.point_radii = _as_float32(self.point_radii, shape=(-1,), name="point_radii")
        count = self.point_positions.shape[0]
        if self.point_normals.shape[0] != count or self.point_radii.shape[0] != count:
            raise ValueError("point_positions, point_normals and point_radii must share the point count")
        _check_unit_normals(self.point_normals, name="point_normals")
        if count and not torch.all(self.point_radii > 0.0):
            raise ValueError("point_radii must be positive")
        self.ke = _finite_scalar(self.ke, name="ke", minimum=0.0)
        self.kd = _finite_scalar(self.kd, name="kd", minimum=0.0)
        self.mu = _finite_scalar(self.mu, name="mu", minimum=0.0)
        self.ke_floor = _finite_scalar(self.ke_floor, name="ke_floor", minimum=0.0)
        if not isinstance(self.floor_bound, (bool, np.bool_)):
            raise ValueError("floor_bound must be a bool")
        self.floor_bound = bool(self.floor_bound)
        if self.floor_bound and self.ke != self.ke_floor:
            raise ValueError("ke must equal ke_floor when floor_bound is set")

    @property
    def point_count(self) -> int:
        """Return the number of static contact points N."""
        return int(self.point_positions.shape[0])

    @classmethod
    def contact_free(cls) -> ContactPartners:
        """Return partners without a plane, without points and with zero coefficients."""
        return cls(
            plane_present=False,
            plane_point=torch.zeros(3, dtype=torch.float32),
            plane_normal=torch.tensor(_PLANE_NORMAL, dtype=torch.float32),
            point_positions=torch.zeros((0, 3), dtype=torch.float32),
            point_normals=torch.zeros((0, 3), dtype=torch.float32),
            point_radii=torch.zeros((0,), dtype=torch.float32),
            ke=0.0,
            kd=0.0,
            mu=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of plain lists and scalars."""
        return {
            "plane_present": bool(self.plane_present),
            "plane_point": self.plane_point.tolist(),
            "plane_normal": self.plane_normal.tolist(),
            "point_positions": self.point_positions.tolist(),
            "point_normals": self.point_normals.tolist(),
            "point_radii": self.point_radii.tolist(),
            "ke": float(self.ke),
            "kd": float(self.kd),
            "mu": float(self.mu),
            "ke_floor": float(self.ke_floor),
            "floor_bound": bool(self.floor_bound),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContactPartners:
        """Rebuild partners from :meth:`to_dict` output.

        Payloads written before the load-based floor lack ``ke_floor`` and
        ``floor_bound``; they rebuild with the floor disabled.

        Raises:
            ValueError: If a required field is missing or fails validation.
        """
        if not isinstance(data, dict):
            raise ValueError("data must be a mapping produced by ContactPartners.to_dict")
        names = (
            "plane_present",
            "plane_point",
            "plane_normal",
            "point_positions",
            "point_normals",
            "point_radii",
            "ke",
            "kd",
            "mu",
        )
        missing = [name for name in names if name not in data]
        if missing:
            raise ValueError(f"data is missing contact partner fields: {missing}")
        optional = {name: data[name] for name in ("ke_floor", "floor_bound") if name in data}
        return cls(**{name: data[name] for name in names}, **optional)


def _largest_face_sample_count(face_index: Tensor) -> int:
    """Return the largest number of exposed face samples sharing one material face index."""
    if face_index.numel() == 0:
        return 0
    return int(torch.bincount(face_index, minlength=6).max())


def contact_stiffness_floor(
    rest: VoxelGridData,
    *,
    density: float,
    gravity_magnitude: float,
    static_penetration_max: float,
    sample_radius: float,
) -> float:
    """Return the load-based contact stiffness floor ``m g / (n_face * d_max)`` [N/m].

    A body resting on its largest flat face spreads its weight ``m g`` over the
    ``n_face`` exposed face samples of that face; with a penalty stiffness
    ``ke`` per sample the static penetration is ``m g / (n_face * ke)``.
    Requiring it to stay below ``d_max = static_penetration_max * r`` gives
    the floor. Here ``m = rho * V`` with ``V = C h^3`` the rest volume of the
    ``C`` cells, and ``n_face`` is the largest count of exposed face samples
    sharing one material face index (400 for the 10x10x40 beam). For the
    canonical beam with ``E = 1e3`` Pa, ``rho = 1e4`` kg/m^3 and ``g = 9.81``
    the floor is ``6131 / (400 * 0.00625) = 2452`` N/m. Zero gravity gives a
    zero floor.

    Args:
        rest: Rest geometry whose cell count and exposed faces set ``V`` and ``n_face``.
        density: Rest density rho [kg/m^3], positive.
        gravity_magnitude: Scene gravity magnitude ``g`` [m/s^2], nonnegative.
        static_penetration_max: Allowed static penetration in units of ``sample_radius``, positive.
        sample_radius: Surface sample radius ``r`` [m], positive.

    Raises:
        ValueError: If the rest geometry is not :class:`VoxelGridData`, has no
            exposed face, or a scalar is outside its range.
    """
    if not isinstance(rest, VoxelGridData):
        raise ValueError("rest must be VoxelGridData")
    density = _finite_scalar(density, name="density", minimum=0.0, strict=True)
    gravity_magnitude = _finite_scalar(gravity_magnitude, name="gravity_magnitude", minimum=0.0)
    static_penetration_max = _finite_scalar(
        static_penetration_max, name="static_penetration_max", minimum=0.0, strict=True
    )
    sample_radius = _finite_scalar(sample_radius, name="sample_radius", minimum=0.0, strict=True)
    face_count = _largest_face_sample_count(exposed_face_samples(rest).face_index)
    if face_count == 0:
        raise ValueError("rest has no exposed face to rest on")
    mass = density * len(rest.cell_corner_indices) * float(rest.cell_size) ** 3
    return mass * gravity_magnitude / (face_count * static_penetration_max * sample_radius)


def sample_contact_partners(
    rest: VoxelGridData,
    *,
    master_seed: int,
    seed: int,
    youngs_modulus: float,
    cell_size: float,
    time_step: float,
    plane_probability: float = 0.8,
    plane_height_range: tuple[float, float] = (-0.35, -0.02),
    max_points: int = 64,
    point_radius_range: tuple[float, float] = (0.5, 2.0),
    kappa_range: tuple[float, float] = (0.1, 10.0),
    beta_range: tuple[float, float] = (0.0, 1.0),
    mu_range: tuple[float, float] = (0.0, 1.0),
    density: float | None = None,
    gravity_magnitude: float | None = None,
    static_penetration_max: float | None = None,
    sample_radius: float | None = None,
) -> ContactPartners:
    """Draw one reproducible contact scene for a trajectory.

    The generator is seeded from ``SeedSequence([master_seed, seed, 2203])``
    and draws, in this order, ``kappa``, ``beta``, ``mu``, the plane flag, the
    plane height and the point count, so the coefficients never depend on the
    plane or point draws: ``ke = kappa * E * h`` with ``kappa`` log-uniform in
    ``kappa_range``, ``kd = beta * ke * dt`` with ``beta`` uniform in
    ``beta_range`` and ``mu`` uniform in ``mu_range``. With
    ``static_penetration_max`` given, ``ke`` is raised to the load-based floor
    :func:`contact_stiffness_floor` when the sampled value falls below it
    (``floor_bound``), and ``kd`` uses the floored ``ke``; the draws are the
    same either way, so a seed reproduces its scene with the floor on or off.
    The plane, present with
    ``plane_probability``, passes through the rest bounding-box center at a
    height uniform in ``plane_height_range`` relative to the body's rest
    y-minimum; its height is drawn even when the plane is absent. The point
    count is uniform in ``{0, ..., max_points}``.

    The points themselves come from the first spawned child of that seed
    sequence, one point at a time, so their rejection loop never shifts the
    draws above. A position is uniform in the rest bounding box extended by
    :data:`POINT_BOX_LATERAL_MARGIN` in x and z and by
    :data:`POINT_BOX_DEPTH_MARGIN` toward -y, restricted to
    ``z >= z_min + cell_size`` (nothing in front of the clamped z-minimum
    face) and outside the rest bounding box grown by
    :data:`POINT_CLEARANCE_CELLS` cells. Its normal is uniform on the sphere,
    flipped toward the rest bounding-box center and redrawn until it opposes
    the outward normal of the nearest rest surface sample within
    :data:`POINT_NORMAL_MAX_ANGLE` degrees; the radius is uniform in
    ``point_radius_range`` times ``cell_size``. The point is then rejected and
    the position redrawn if :func:`detect_contacts` on the resting rest-surface
    samples (sample radius :data:`POINT_REJECTION_SAMPLE_RADIUS_CELLS` cells,
    search band widened to ``r + cell_size``) would report any candidate, so
    no disk slices or grazes the rest body.

    Args:
        rest: Rest geometry whose corner positions bound the scene.
        master_seed: Nonnegative campaign seed.
        seed: Nonnegative trajectory seed.
        youngs_modulus: Sampled Young's modulus E [Pa].
        cell_size: Rest cell edge length h [m].
        time_step: Physical time step dt [s].
        plane_probability: Probability that the ground plane is present.
        plane_height_range: Plane height bounds relative to the rest y-minimum [m].
        max_points: Largest static point count.
        point_radius_range: Lateral radius bounds in units of ``cell_size``.
        kappa_range: Positive log-uniform bounds of the stiffness factor.
        beta_range: Nonnegative uniform bounds of the damping factor.
        mu_range: Nonnegative uniform bounds of the friction coefficient.
        density: Rest density rho [kg/m^3] of the body; required by the floor.
        gravity_magnitude: Gravity magnitude ``g`` [m/s^2] of the scene; required by the floor.
        static_penetration_max: Allowed static penetration in units of
            ``sample_radius``; None disables the floor (the default).
        sample_radius: Surface sample radius ``r`` [m]; None means ``0.5 * cell_size``.

    Returns:
        The sampled partners with CPU float32 tensors.

    Raises:
        ValueError: If the rest geometry, seeds, scalars or ranges are invalid,
            if the floor is enabled without ``density`` or
            ``gravity_magnitude``, if ``max_points > 0`` and the clearance
            leaves no room for points inside the box (``cell_size`` at least
            :data:`POINT_BOX_DEPTH_MARGIN`), or if a point cannot be placed
            within :data:`POINT_PLACEMENT_ATTEMPTS` position draws.
    """
    if not isinstance(rest, VoxelGridData):
        raise ValueError("rest must be VoxelGridData")
    corners = np.asarray(rest.corner_rest_positions, dtype=np.float64)
    if corners.ndim != 2 or corners.shape[1] != 3 or corners.shape[0] == 0 or not np.isfinite(corners).all():
        raise ValueError("rest.corner_rest_positions must be a finite nonempty [P, 3] array")
    master_seed = _nonnegative_int(master_seed, name="master_seed")
    seed = _nonnegative_int(seed, name="seed")
    youngs_modulus = _finite_scalar(youngs_modulus, name="youngs_modulus", minimum=0.0, strict=True)
    cell_size = _finite_scalar(cell_size, name="cell_size", minimum=0.0, strict=True)
    time_step = _finite_scalar(time_step, name="time_step", minimum=0.0, strict=True)
    plane_probability = _finite_scalar(plane_probability, name="plane_probability", minimum=0.0)
    if plane_probability > 1.0:
        raise ValueError("plane_probability must lie in [0, 1]")
    plane_height_range = _ordered_range(plane_height_range, name="plane_height_range")
    max_points = _nonnegative_int(max_points, name="max_points")
    point_radius_range = _ordered_range(point_radius_range, name="point_radius_range", minimum=0.0, strict=True)
    kappa_range = _ordered_range(kappa_range, name="kappa_range", minimum=0.0, strict=True)
    beta_range = _ordered_range(beta_range, name="beta_range", minimum=0.0)
    mu_range = _ordered_range(mu_range, name="mu_range", minimum=0.0)
    if density is not None:
        density = _finite_scalar(density, name="density", minimum=0.0, strict=True)
    if gravity_magnitude is not None:
        gravity_magnitude = _finite_scalar(gravity_magnitude, name="gravity_magnitude", minimum=0.0)
    sample_radius = (
        0.5 * cell_size
        if sample_radius is None
        else _finite_scalar(sample_radius, name="sample_radius", minimum=0.0, strict=True)
    )
    ke_floor = 0.0
    if static_penetration_max is not None:
        if density is None or gravity_magnitude is None:
            raise ValueError("static_penetration_max requires density and gravity_magnitude")
        ke_floor = contact_stiffness_floor(
            rest,
            density=density,
            gravity_magnitude=gravity_magnitude,
            static_penetration_max=static_penetration_max,
            sample_radius=sample_radius,
        )

    root = np.random.SeedSequence([master_seed, seed, 2203])
    rng = np.random.default_rng(root)

    kappa = math.exp(rng.uniform(math.log(kappa_range[0]), math.log(kappa_range[1])))
    beta = float(rng.uniform(*beta_range))
    mu = float(rng.uniform(*mu_range))
    ke = kappa * youngs_modulus * cell_size
    floor_bound = ke_floor > ke
    if floor_bound:
        ke = ke_floor
    kd = beta * ke * time_step

    lower = corners.min(axis=0)
    upper = corners.max(axis=0)
    center = 0.5 * (lower + upper)

    plane_present = bool(rng.random() < plane_probability)
    plane_height = float(rng.uniform(*plane_height_range))
    plane_point = np.array([center[0], lower[1] + plane_height, center[2]], dtype=np.float64)

    count = int(rng.integers(0, max_points, endpoint=True))
    box_lower = lower - np.array([POINT_BOX_LATERAL_MARGIN, POINT_BOX_DEPTH_MARGIN, POINT_BOX_LATERAL_MARGIN])
    box_upper = upper + np.array([POINT_BOX_LATERAL_MARGIN, 0.0, POINT_BOX_LATERAL_MARGIN])
    # The clamped face is the z-minimum face: keep every point at least one cell behind it.
    box_lower[2] = lower[2] + cell_size
    clearance_lower = lower - POINT_CLEARANCE_CELLS * cell_size
    clearance_upper = upper + POINT_CLEARANCE_CELLS * cell_size
    if max_points and np.all(clearance_lower <= box_lower) and np.all(clearance_upper >= box_upper):
        raise ValueError("cell_size is too large for the static point box: the clearance region covers it")
    if count:
        faces = exposed_face_samples(rest)
        rest_samples = sample_points(torch.from_numpy(corners)[None], faces.corners)[0]
        points = _draw_static_points(
            np.random.default_rng(root.spawn(1)[0]),
            count,
            box_lower=box_lower,
            box_upper=box_upper,
            clearance_lower=clearance_lower,
            clearance_upper=clearance_upper,
            center=center,
            rest_samples=rest_samples,
            rest_normals=faces.rest_normals.double(),
            radius_range=(point_radius_range[0] * cell_size, point_radius_range[1] * cell_size),
            cell_size=cell_size,
        )
        positions, normals, radii = points.positions, points.normals, points.radii
    else:
        positions = np.zeros((0, 3))
        normals = np.zeros((0, 3))
        radii = np.zeros((0,))

    return ContactPartners(
        plane_present=plane_present,
        plane_point=plane_point,
        plane_normal=np.array(_PLANE_NORMAL, dtype=np.float64),
        point_positions=positions,
        point_normals=normals,
        point_radii=radii,
        ke=ke,
        kd=kd,
        mu=mu,
        ke_floor=ke_floor,
        floor_bound=floor_bound,
    )


class _StaticPoints(NamedTuple):
    """Accepted static points of one scene and the number of position draws they took."""

    positions: np.ndarray
    normals: np.ndarray
    radii: np.ndarray
    attempts: int


def _point_candidates(
    positions: Tensor,
    threshold: Tensor,
    radius: float,
    point_positions: Tensor,
    point_normals: Tensor,
    point_radii: Tensor,
    sample_normals: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the ``[S, N]`` candidate mask and gaps of samples against static-point disks.

    A sample is a candidate of a point when its lateral distance from the
    point's axis is below ``r_p`` and ``-radius <= gap < threshold``; with
    ``sample_normals`` the point normal must additionally oppose the sample's
    face normal. All tensors are CPU float64; ``threshold`` has shape ``[S]``.
    """
    offsets = positions[:, None, :] - point_positions[None, :, :]
    gap = (offsets * point_normals[None, :, :]).sum(dim=-1)
    lateral = torch.linalg.vector_norm(offsets - gap[..., None] * point_normals[None, :, :], dim=-1)
    candidate = (lateral < point_radii[None, :]) & (gap < threshold[:, None]) & (gap >= -radius)
    if sample_normals is not None:
        candidate &= (sample_normals @ point_normals.T) < 0.0
    return candidate, gap


def _draw_static_points(
    rng: np.random.Generator,
    count: int,
    *,
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    clearance_lower: np.ndarray,
    clearance_upper: np.ndarray,
    center: np.ndarray,
    rest_samples: Tensor,
    rest_normals: Tensor,
    radius_range: tuple[float, float],
    cell_size: float,
) -> _StaticPoints:
    """Draw ``count`` static points one at a time, rejecting disks that reach the rest body.

    Each position is uniform in the box and redrawn while it lies in the
    cleared region, while no normal within :data:`_NORMAL_ATTEMPTS_PER_POSITION`
    draws opposes the nearest rest sample's face normal within
    :data:`POINT_NORMAL_MAX_ANGLE`, or while :func:`_point_candidates` reports
    a rest sample inside the disk band ``-r <= gap < r + cell_size`` with
    ``r`` of :data:`POINT_REJECTION_SAMPLE_RADIUS_CELLS` cells.

    Args:
        rng: Generator dedicated to the points so retries shift no other draw.
        count: Number of points to place.
        box_lower: Lower corner of the point box [m], shape [3].
        box_upper: Upper corner of the point box [m], shape [3].
        clearance_lower: Lower corner of the excluded cleared region [m], shape [3].
        clearance_upper: Upper corner of the excluded cleared region [m], shape [3].
        center: Rest bounding-box center the normals are flipped toward [m], shape [3].
        rest_samples: Rest surface sample positions [m], float64, shape [S, 3].
        rest_normals: Outward rest face normals, float64, shape [S, 3].
        radius_range: Lateral radius bounds [m].
        cell_size: Rest cell edge length h [m].

    Raises:
        ValueError: If a point finds no admissible placement within
            :data:`POINT_PLACEMENT_ATTEMPTS` position draws.
    """
    radius = POINT_REJECTION_SAMPLE_RADIUS_CELLS * cell_size
    threshold = torch.full((rest_samples.shape[0],), radius + cell_size, dtype=torch.float64)
    max_cos = -math.cos(math.radians(POINT_NORMAL_MAX_ANGLE))
    positions = np.zeros((count, 3))
    normals = np.zeros((count, 3))
    radii = np.zeros(count)
    attempts = 0
    for index in range(count):
        for _ in range(POINT_PLACEMENT_ATTEMPTS):
            attempts += 1
            position = rng.uniform(box_lower, box_upper)
            if np.all((position > clearance_lower) & (position < clearance_upper)):
                continue
            position_tensor = torch.from_numpy(position)
            nearest = int(torch.argmin(torch.linalg.vector_norm(rest_samples - position_tensor, dim=1)))
            face_normal = rest_normals[nearest].numpy()
            for _ in range(_NORMAL_ATTEMPTS_PER_POSITION):
                normal = rng.normal(size=3)
                norm = float(np.linalg.norm(normal))
                normal = np.array(_PLANE_NORMAL) if norm <= 0.0 else normal / norm
                # Flip the normal so it faces the body; a point at the center keeps its draw.
                if float(normal @ (center - position)) < 0.0:
                    normal = -normal
                if float(normal @ face_normal) <= max_cos:
                    break
            else:
                continue
            point_radius = float(rng.uniform(*radius_range))
            candidate, _ = _point_candidates(
                rest_samples,
                threshold,
                radius,
                position_tensor[None],
                torch.from_numpy(normal)[None],
                torch.tensor([point_radius], dtype=torch.float64),
            )
            if not bool(candidate.any()):
                positions[index] = position
                normals[index] = normal
                radii[index] = point_radius
                break
        else:
            raise ValueError(
                f"could not place static point {index} within {POINT_PLACEMENT_ATTEMPTS} draws: "
                "the point box leaves no admissible disk outside the rest body"
            )
    return _StaticPoints(positions=positions, normals=normals, radii=radii, attempts=attempts)


class ContactPairs(NamedTuple):
    """Frozen list of candidate (sample, partner) pairs for one physical step.

    Experimental. Rows are ordered by sample index, then kind, then gap, then
    partner index. ``partner_index`` is -1 for the plane. ``partner_radius`` is
    the lateral radius ``r_p`` [m]; plane rows carry
    :data:`PLANE_PARTNER_RADIUS`. All tensors live on the CPU.

    Attributes:
        sample_index: Owning surface sample of each pair, int64, shape [Q].
        partner_index: Static point index or -1 for the plane, int64, shape [Q].
        kind: Partner kind (0 plane, 1 point, 2 self reserved), int64, shape [Q].
        partner_point: Partner point ``p`` [m], float32, shape [Q, 3]; the plane's
            foot point of the sample position for plane rows.
        partner_normal: Unit partner normal ``n``, float32, shape [Q, 3].
        partner_radius: Lateral radius of influence [m], float32, shape [Q].
    """

    sample_index: Tensor
    partner_index: Tensor
    kind: Tensor
    partner_point: Tensor
    partner_normal: Tensor
    partner_radius: Tensor

    @classmethod
    def empty(cls) -> ContactPairs:
        """Return a pair list with Q = 0."""
        return cls(
            sample_index=torch.zeros((0,), dtype=torch.int64),
            partner_index=torch.zeros((0,), dtype=torch.int64),
            kind=torch.zeros((0,), dtype=torch.int64),
            partner_point=torch.zeros((0, 3), dtype=torch.float32),
            partner_normal=torch.zeros((0, 3), dtype=torch.float32),
            partner_radius=torch.zeros((0,), dtype=torch.float32),
        )


def _detection_input(value: Any, *, name: str) -> Tensor:
    """Return ``value`` as a finite CPU float64 tensor of shape [S, 3]."""
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor of shape [S, 3]")
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [S, 3], got {list(value.shape)}")
    result = value.detach().to("cpu", torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def detect_contacts(
    sample_positions: Tensor,
    sample_velocities: Tensor,
    partners: ContactPartners,
    *,
    radius: float,
    time_step: float,
    max_pairs_per_sample: int = 4,
    sample_normals: Tensor | None = None,
) -> ContactPairs:
    """Collect candidate pairs between surface samples and static partners.

    Brute-force but vectorized (all S x N sample-point distances at once) and
    deterministic. With ``margin_s = radius + |v_s| dt`` per sample, the plane
    is a candidate when ``plane_present`` and ``gap < radius + margin_s``, where
    ``gap = (x_s - plane_point) . plane_normal`` and the partner point is the
    foot point of ``x_s`` on the plane. A static point is a candidate when its
    lateral distance ``|(x_s - p) - gap n| < r_p`` and
    ``-radius <= gap < radius + margin_s``: the disk is one-sided, and a sample
    whose sphere no longer reaches the disk plane from behind (a far face of the
    body, say) is not pulled toward it, so a point pair starts at most
    ``2 radius`` deep. The plane has no such bound. When ``sample_normals`` is
    given, a plane or point pair is a candidate only if the partner normal
    opposes the sample's face normal, ``n . n_s < 0``, so a partner cannot push
    on a face it grazes or sees from behind. Only the ``max_pairs_per_sample``
    nearest points by gap survive per sample, ties resolved toward the lower
    point index. The result holds at most ``S * (1 + max_pairs_per_sample)``
    rows.

    Args:
        sample_positions: Step-start sample positions [m], shape [S, 3].
        sample_velocities: Step-start sample velocities [m/s], shape [S, 3].
        partners: Static partners of the scene.
        radius: Sample radius r [m].
        time_step: Physical time step dt [s].
        max_pairs_per_sample: Cap on point pairs per sample.
        sample_normals: Outward unit face normals of the samples, shape [S, 3];
            None keeps every pair regardless of orientation.

    Returns:
        The candidate pairs, or :meth:`ContactPairs.empty` when nothing is near.

    Raises:
        ValueError: If inputs have the wrong shape, are non-finite or scalars are invalid.
    """
    positions = _detection_input(sample_positions, name="sample_positions")
    velocities = _detection_input(sample_velocities, name="sample_velocities")
    if positions.shape != velocities.shape:
        raise ValueError("sample_positions and sample_velocities must share shape [S, 3]")
    if sample_normals is not None:
        sample_normals = _detection_input(sample_normals, name="sample_normals")
        if sample_normals.shape != positions.shape:
            raise ValueError("sample_normals must share shape [S, 3] with sample_positions")
    if not isinstance(partners, ContactPartners):
        raise ValueError("partners must be ContactPartners")
    radius = _finite_scalar(radius, name="radius", minimum=0.0, strict=True)
    time_step = _finite_scalar(time_step, name="time_step", minimum=0.0, strict=True)
    max_pairs_per_sample = _nonnegative_int(max_pairs_per_sample, name="max_pairs_per_sample")

    sample_count = positions.shape[0]
    if sample_count == 0:
        return ContactPairs.empty()
    threshold = 2.0 * radius + torch.linalg.vector_norm(velocities, dim=1) * time_step

    samples: list[Tensor] = []
    indices: list[Tensor] = []
    kinds: list[Tensor] = []
    points: list[Tensor] = []
    normals: list[Tensor] = []
    radii: list[Tensor] = []
    gaps: list[Tensor] = []

    if partners.plane_present:
        plane_normal = partners.plane_normal.double()
        plane_gap = (positions - partners.plane_point.double()) @ plane_normal
        plane_hit = plane_gap < threshold
        if sample_normals is not None:
            plane_hit &= (sample_normals @ plane_normal) < 0.0
        hit = torch.nonzero(plane_hit, as_tuple=False).flatten()
        if hit.numel():
            samples.append(hit)
            indices.append(torch.full_like(hit, -1))
            kinds.append(torch.full_like(hit, KIND_PLANE))
            points.append(positions[hit] - plane_gap[hit, None] * plane_normal)
            normals.append(plane_normal.expand(hit.numel(), 3))
            radii.append(torch.full((hit.numel(),), PLANE_PARTNER_RADIUS, dtype=torch.float64))
            gaps.append(plane_gap[hit])

    point_count = partners.point_count
    if point_count and max_pairs_per_sample:
        point_positions = partners.point_positions.double()
        point_normals = partners.point_normals.double()
        point_radii = partners.point_radii.double()
        candidate, point_gap = _point_candidates(
            positions, threshold, radius, point_positions, point_normals, point_radii, sample_normals
        )
        ranked_gap, order = torch.sort(torch.where(candidate, point_gap, torch.inf), dim=1, stable=True)
        keep = min(max_pairs_per_sample, point_count)
        kept_gap = ranked_gap[:, :keep]
        kept_index = order[:, :keep]
        valid = torch.isfinite(kept_gap)
        if valid.any():
            sample_index = torch.arange(sample_count, dtype=torch.int64)[:, None].expand(sample_count, keep)[valid]
            partner_index = kept_index[valid]
            samples.append(sample_index)
            indices.append(partner_index)
            kinds.append(torch.full_like(partner_index, KIND_POINT))
            points.append(point_positions[partner_index])
            normals.append(point_normals[partner_index])
            radii.append(point_radii[partner_index])
            gaps.append(kept_gap[valid])

    if not samples:
        return ContactPairs.empty()

    sample_index = torch.cat(samples)
    partner_index = torch.cat(indices)
    kind = torch.cat(kinds)
    gap = torch.cat(gaps)
    order = torch.from_numpy(
        np.lexsort((partner_index.numpy(), gap.numpy(), kind.numpy(), sample_index.numpy())).astype(np.int64)
    )
    return ContactPairs(
        sample_index=sample_index[order].contiguous(),
        partner_index=partner_index[order].contiguous(),
        kind=kind[order].contiguous(),
        partner_point=torch.cat(points)[order].to(torch.float32).contiguous(),
        partner_normal=torch.cat(normals)[order].to(torch.float32).contiguous(),
        partner_radius=torch.cat(radii)[order].to(torch.float32).contiguous(),
    )
