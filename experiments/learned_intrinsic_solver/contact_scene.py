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
so a point can never pull a far face of the body toward it. This module owns
the per-scene partner record (:class:`ContactPartners`), the seeded scene generator
(:func:`sample_contact_partners`), the frozen per-step pair list
(:class:`ContactPairs`) and the CPU detector (:func:`detect_contacts`).

Sampling uses NumPy; detection reads and writes CPU ``torch`` tensors so the
result can be handed to the energy and network input code without further
conversion. Every numeric range in :func:`sample_contact_partners` is
provisional, as recorded in section 7 of the design note.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch  # noqa: TID253 -- Explicit opt-in PyTorch implementation.
from torch import Tensor  # noqa: TID253

from .data import VoxelGridData

__all__ = [
    "KIND_PLANE",
    "KIND_POINT",
    "KIND_SELF",
    "PLANE_PARTNER_RADIUS",
    "POINT_BOX_DEPTH_MARGIN",
    "POINT_BOX_LATERAL_MARGIN",
    "POINT_CLEARANCE_CELLS",
    "ContactPairs",
    "ContactPartners",
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

_PLANE_NORMAL = (0.0, 1.0, 0.0)
_UNIT_NORMAL_TOLERANCE = 1e-3


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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContactPartners:
        """Rebuild partners from :meth:`to_dict` output.

        Raises:
            ValueError: If a field is missing or fails validation.
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
        return cls(**{name: data[name] for name in names})


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
) -> ContactPartners:
    """Draw one reproducible contact scene for a trajectory.

    The generator is seeded from ``SeedSequence([master_seed, seed, 2203])``.
    Coefficients are drawn first so they never depend on the plane or point
    draws: ``ke = kappa * E * h`` with ``kappa`` log-uniform in ``kappa_range``,
    ``kd = beta * ke * dt`` with ``beta`` uniform in ``beta_range`` and ``mu``
    uniform in ``mu_range``. The plane, present with ``plane_probability``,
    passes through the rest bounding-box center at a height uniform in
    ``plane_height_range`` relative to the body's rest y-minimum; its height is
    drawn even when the plane is absent. The point count is uniform in
    ``{0, ..., max_points}``; positions are uniform in the rest bounding box
    extended by :data:`POINT_BOX_LATERAL_MARGIN` in x and z and by
    :data:`POINT_BOX_DEPTH_MARGIN` toward -y, minus the rest bounding box
    grown by :data:`POINT_CLEARANCE_CELLS` cells (rejection sampling, so no
    point lies inside or within one cell of the rest body); normals are uniform
    on the sphere and flipped toward the rest bounding-box center, and radii
    are uniform in ``point_radius_range`` times ``cell_size``.

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

    Returns:
        The sampled partners with CPU float32 tensors.

    Raises:
        ValueError: If the rest geometry, seeds, scalars or ranges are invalid,
            or if ``max_points > 0`` and the clearance leaves no room for
            points inside the box (``cell_size`` at least
            :data:`POINT_BOX_DEPTH_MARGIN`).
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

    rng = np.random.default_rng(np.random.SeedSequence([master_seed, seed, 2203]))

    kappa = math.exp(rng.uniform(math.log(kappa_range[0]), math.log(kappa_range[1])))
    beta = float(rng.uniform(*beta_range))
    mu = float(rng.uniform(*mu_range))
    ke = kappa * youngs_modulus * cell_size
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
    clearance_lower = lower - POINT_CLEARANCE_CELLS * cell_size
    clearance_upper = upper + POINT_CLEARANCE_CELLS * cell_size
    if max_points and np.all(clearance_lower <= box_lower) and np.all(clearance_upper >= box_upper):
        raise ValueError("cell_size is too large for the static point box: the clearance region covers it")
    positions = rng.uniform(box_lower, box_upper, size=(count, 3))
    # Redraw points inside the cleared body region until every point lies in the surrounding shell.
    inside = np.all((positions > clearance_lower) & (positions < clearance_upper), axis=1)
    while inside.any():
        positions[inside] = rng.uniform(box_lower, box_upper, size=(int(inside.sum()), 3))
        inside = np.all((positions > clearance_lower) & (positions < clearance_upper), axis=1)
    normals = rng.normal(size=(count, 3))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = norms[:, 0] <= 0.0
    normals[degenerate] = _PLANE_NORMAL
    norms[degenerate] = 1.0
    normals = normals / norms
    # Flip each normal so it faces the body; points coincident with the center keep their draw.
    facing_away = np.einsum("ij,ij->i", normals, center[None, :] - positions) < 0.0
    normals[facing_away] *= -1.0
    radii = rng.uniform(point_radius_range[0] * cell_size, point_radius_range[1] * cell_size, size=count)

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
    )


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
    ``2 radius`` deep. The plane has no such bound. Only the
    ``max_pairs_per_sample`` nearest points by gap survive per sample, ties
    resolved toward the lower point index. The result holds at most
    ``S * (1 + max_pairs_per_sample)`` rows.

    Args:
        sample_positions: Step-start sample positions [m], shape [S, 3].
        sample_velocities: Step-start sample velocities [m/s], shape [S, 3].
        partners: Static partners of the scene.
        radius: Sample radius r [m].
        time_step: Physical time step dt [s].
        max_pairs_per_sample: Cap on point pairs per sample.

    Returns:
        The candidate pairs, or :meth:`ContactPairs.empty` when nothing is near.

    Raises:
        ValueError: If inputs have the wrong shape, are non-finite or scalars are invalid.
    """
    positions = _detection_input(sample_positions, name="sample_positions")
    velocities = _detection_input(sample_velocities, name="sample_velocities")
    if positions.shape != velocities.shape:
        raise ValueError("sample_positions and sample_velocities must share shape [S, 3]")
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
        hit = torch.nonzero(plane_gap < threshold, as_tuple=False).flatten()
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
        offsets = positions[:, None, :] - point_positions[None, :, :]
        point_gap = (offsets * point_normals[None, :, :]).sum(dim=-1)
        lateral = torch.linalg.vector_norm(offsets - point_gap[..., None] * point_normals[None, :, :], dim=-1)
        candidate = (lateral < point_radii[None, :]) & (point_gap < threshold[:, None]) & (point_gap >= -radius)
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
