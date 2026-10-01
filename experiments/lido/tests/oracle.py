# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Black-box oracle over the OLD learned hex solver for the LIDO parity tests.

Test-only. This module is imported by tests under ``experiments/lido/tests``
and must never be imported by the ``experiments.lido`` package itself. It
calls ``experiments.learned_intrinsic_solver`` (the old package) as a black
box and exposes plain functions with numpy float64 in/out and SI units at the
API (metres, seconds, pascals, kilograms per cubic metre, pascal seconds,
joules, newtons). Array layouts at this API: corners ``[P, 3]``, cells
``[C, 8]``, per-cell mode vectors ``[C, 7, 3]`` (vector index first; the old
code packs them as ``[C, 3, 7]`` with the vectors in columns, see
:func:`conventions`).

Units. The old training step (``mixed_physics.MixedHexSolverStep``, the one
the v4 checkpoint was trained through) evaluates the objective in the cell
units ``h = mu = dt = 1`` and converts at its API; the old single-material
step (``solver_step.LearnedHexSolverStep``) evaluates the same objective in
SI. The oracle uses the old package's SI objective
``hex_energy.HexImplicitEulerLoss`` in float64. The normalised path equals it
exactly in real arithmetic (``E_SI = mu h^3 E'``, ``dE_SI/dX = mu h^2
dE'/dX'``, ``X' = X / h``); ``test_oracle_selfcheck`` checks this identity
with the old code's own normalised construction. The one network input that
is not dimensionless by construction, the fusion-projected gradient, is
returned in joules by default; the old feature pipeline divides it by the
energy unit ``mu h^3`` (:func:`energy_unit`, keyword ``energy_unit`` of
:func:`project_gradient`; :func:`network_inputs` applies it).

Gaps (everything else calls the old functions unchanged):

* ``fuse`` and ``project_gradient`` accept ``X`` for API symmetry only. The
  old fusion operators (``K = G^T W G``, ``B = (G^T M)_f``) are rest-grid
  constants and the prescribed displacement is zero, so neither result
  depends on ``X``; ``fuse`` returns ``fused - X``, which is ``X``-free.
* Fusion weights: the old training step uses unit cell weights on the unit
  grid; the old SI step uses ``stiffness h^3``. A weighted least squares fit
  is invariant under a uniform rescaling of its weights, so the oracle uses
  unit weights on the SI grid (identical fit; ``d_SI = h d'``).
* Contact is not exposed (the parity brief excludes it): ``energy_and_grad``
  is the contact-free incremental potential and ``network_inputs`` feeds the
  network the contact-free path (17 zero contact channels).
* Frames are computed in float64 here; the old production path runs float32
  and switches to float64 only for the tie cells. Same functions, same rule.
* ``network_forward`` builds the OLD network for the requested (small) grid
  and copies the v4 parameters one to one; the two neighbourhood buffers
  ``neighbor_indices_1`` / ``neighbor_mask_1`` are grid dependent and are
  rebuilt rather than loaded (they are the only keys not loaded).
* The old package factorises the fusion matrix with oneMKL PARDISO on the
  CPU (``pardiso.PardisoFactor``); the oracle inherits that dependency.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch

from experiments.learned_intrinsic_solver.data import VoxelGridData, generate_cuboid
from experiments.learned_intrinsic_solver.features import (
    CONDITIONING_CHANNELS,
    CONTACT_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    conditioning_channels,
    state_feature_dim,
)
from experiments.learned_intrinsic_solver.frames import (
    closest_proper_rotations,
    reference_rotation,
    select_reference_corners,
)
from experiments.learned_intrinsic_solver.fusion import HexFusion
from experiments.learned_intrinsic_solver.hex_energy import HexImplicitEulerLoss, hex_gauss_quadrature
from experiments.learned_intrinsic_solver.hex_modes import MODE_COUNT, corner_local_coordinates, mode_vectors
from experiments.learned_intrinsic_solver.input_assembly import OptimizerHistory, assemble_inputs
from experiments.learned_intrinsic_solver.material_sampling import lame_from_youngs_modulus
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.network_geometry import build_grid_neighborhood

__all__ = [
    "CANONICAL_CELL_COUNTS",
    "CANONICAL_CELL_SIZE",
    "V4_CHECKPOINT",
    "conditioning",
    "contact_token_layout",
    "conventions",
    "edge_feature_layout",
    "energy_and_grad",
    "energy_terms",
    "energy_unit",
    "frames",
    "fuse",
    "lame",
    "modes",
    "network_forward",
    "network_inputs",
    "network_state_dict_summary",
    "node_feature_layout",
    "project_gradient",
    "reference_corners",
    "rest_grid",
]

_DTYPE = torch.float64
CANONICAL_CELL_COUNTS = (10, 10, 40)
CANONICAL_CELL_SIZE = 0.025
V4_CHECKPOINT = Path("generated/training_v4_20260928/checkpoints/best_validation.pt")
"""Path (relative to the repository root) of the v4 checkpoint; read only."""


# -- helpers -----------------------------------------------------------------------------------------------


def _counts(cell_counts) -> tuple[int, int, int]:
    counts = tuple(int(count) for count in cell_counts)
    if len(counts) != 3 or any(count <= 0 for count in counts):
        raise ValueError("cell_counts must be three positive integers")
    return counts


def _t(array, dtype=_DTYPE) -> torch.Tensor:
    """Copy a numpy array (or array-like) into a contiguous CPU tensor."""
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    return torch.from_numpy(np.ascontiguousarray(np.asarray(array, dtype=numpy_dtype)))


def _np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy().copy()


class _Grid(NamedTuple):
    rest: VoxelGridData
    cells: torch.Tensor
    fixed: np.ndarray
    pinned_mask: np.ndarray
    reference_corners: np.ndarray | None


@functools.cache
def _grid(cell_counts: tuple[int, int, int], h: float) -> _Grid:
    """Canonical rest grid at origin (0, 0, 0) with the training pin convention (z-min face)."""
    rest = generate_cuboid(cell_counts, cell_size=h)
    z = rest.corner_rest_positions[:, 2]
    # train_mixed.py: fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
    fixed = np.flatnonzero(z == z.min()).astype(np.int64)
    pinned = np.zeros(len(z), dtype=bool)
    pinned[fixed] = True
    reference = select_reference_corners(rest.corner_rest_positions, fixed)
    return _Grid(rest, torch.as_tensor(rest.cell_corner_indices, dtype=torch.long), fixed, pinned, reference)


@functools.cache
def _fusion(cell_counts: tuple[int, int, int], h: float) -> HexFusion:
    """Unit-weight seven-mode fusion on the SI grid (the old training step's weights, see module docstring)."""
    grid = _grid(cell_counts, h)
    return HexFusion(grid.rest, grid.fixed, dtype=_DTYPE, target_modes=MODE_COUNT)


@functools.cache
def _energy_module(cell_counts, h, dt, E, nu, rho, eta) -> HexImplicitEulerLoss:
    grid = _grid(cell_counts, h)
    lam, mu = lame_from_youngs_modulus(E, nu)
    return HexImplicitEulerLoss(grid.rest, lam, mu, rho, dt, damping=eta, dtype=_DTYPE)


@functools.cache
def _checkpoint(path: str) -> dict:
    # Read only: torch.load never writes. weights_only=False because the old format stores RNG/controller state.
    return torch.load(path, map_location="cpu", weights_only=False)


# -- conventions -------------------------------------------------------------------------------------------


def lame(E: float, nu: float) -> tuple[float, float]:
    """Return ``(lambda, mu)`` [Pa] from Young's modulus and Poisson's ratio as the old sampler does."""
    return lame_from_youngs_modulus(float(E), float(nu))


def energy_unit(h: float, E: float, nu: float) -> float:
    """Return ``mu h^3`` [J], the unit of the old gradient feature (``axis_gradient_world``, ``log_gradient_rms``)."""
    _, mu = lame(E, nu)
    return mu * float(h) ** 3


def reference_corners(cell_counts, h) -> np.ndarray | None:
    """Return the three ordered corner IDs of the frame tie-break reference for this grid, or None."""
    grid = _grid(_counts(cell_counts), float(h))
    return None if grid.reference_corners is None else grid.reference_corners.copy()


def node_feature_layout() -> list[tuple[str, int]]:
    """Return ``(name, width)`` slices of the 159-wide node input in order (sum 159)."""
    return [
        ("local_axes: R^T V, [3, 7] row-major (value 7 i + m)", 21),
        ("inertial_axis_offset: R^T (V_Y - V)", 21),
        ("physical_axis_change: R^T (V - V_prev)", 21),
        ("current_axis_gradient: clip(R^T G / rms_G, +-10), G in energy_unit", 21),
        ("previous_axis_gradient: clip(R^T G_prev / rms_G, +-10); zero when history invalid", 21),
        ("previous_axis_update: clip(R^T U_prev / rms_U, +-10); zero when history invalid", 21),
        ("exposed_face_flags: -x, +x, -y, +y, -z, +z", 6),
        ("fixed_corner_flags: corner_order (z fastest)", 8),
        ("log_gradient_rms: ln max(rms_G, 1e-12)", 1),
        ("history_valid: 1.0 or 0.0", 1),
        ("contact_pooled: zero-init Linear(128 -> 16) of [masked mean, masked max]", 16),
        ("contact_count_fraction: valid tokens / M (M = 24)", 1),
    ]


def edge_feature_layout() -> list[tuple[str, int]]:
    """Return ``(name, width)`` slices of the 24-wide directed edge feature (receiver i, sender j)."""
    return [
        ("rest_offset: (c_j - c_i)_rest / h", 3),
        ("current_offset: R_i^T (c_j - c_i) / h, c = mean of the 8 corners", 3),
        ("relative_frame: R_i^T R_j, [3, 3] row-major", 9),
        ("transported_axes: R_i^T R_j A_j = R_i^T F_j, [3, 3] row-major (centre F only)", 9),
    ]


def contact_token_layout() -> list[tuple[str, int]]:
    """Return ``(name, width)`` slices of the 19-wide contact token (owner-cell frame, dimensionless)."""
    return [
        ("contact_point: R_i^T (x_s - c_i) / h", 3),
        ("partner_point: R_i^T (p - c_i) / h", 3),
        ("partner_normal: R_i^T n", 3),
        ("gap / r with gap = (x_s - p) . n", 1),
        ("approach_rate: -(n . (x_s - x_s0)) / r over the current step", 1),
        ("radius_ratio: min(r_p / r, 10)", 1),
        ("log1p(kappa), kappa = ke / (E h)", 1),
        ("beta = kd / (ke dt)", 1),
        ("mu_friction", 1),
        ("kind one-hot: plane, point, self", 3),
        ("self flag (always 0 in v1)", 1),
    ]


def conventions() -> dict:
    """Return the old code's ordering and scaling conventions (values computed from the old functions)."""
    corners = [tuple(int(v) for v in row) for row in corner_local_coordinates(dtype=_DTYPE).tolist()]
    rule = hex_gauss_quadrature(1.0, dtype=np.float64)
    canonical = _grid(CANONICAL_CELL_COUNTS, CANONICAL_CELL_SIZE)
    canonical_reference = [int(index) for index in canonical.reference_corners]
    _nx, ny, nz = (count + 1 for count in CANONICAL_CELL_COUNTS)
    reference_coordinates = [(p // (ny * nz), p // nz % ny, p % nz) for p in canonical_reference]
    small = _grid((2, 2, 3), CANONICAL_CELL_SIZE)
    return {
        "corner_order": corners,
        "corner_order_rule": (
            "local corner k = 4 x + 2 y + z (bits x, y, z in {0, 1}), xi_k = (2x - 1, 2y - 1, 2z - 1); z fastest; "
            "corner k of cell (ix, iy, iz) is the grid corner (ix + x, iy + y, iz + z)"
        ),
        "gauss_points": [tuple(float(v) for v in row) for row in rule.points.tolist()],
        "gauss_points_rule": "xi_q = corner_order[q] / sqrt(3), same z-fast order as the corners",
        "gauss_weights": "h^3 / 8 per point in the energy (rest Jacobian included); the fusion normalises to 1/8",
        "mode_order": ("a1", "a2", "a3", "w12", "w13", "w23", "w123"),
        "mode_scalar_functions": ("xi1", "xi2", "xi3", "xi1 xi2", "xi1 xi3", "xi2 xi3", "xi1 xi2 xi3"),
        "mode_definition": (
            "v_m = (2 / h) c_m with c_m = (1/8) sum_k e_m(xi_k) x_k = (1 / (4 h)) sum_k E[k, m] (x_k - x_0); "
            "v_1..3 are the columns of the centre deformation gradient F (dimensionless), v_4..7 the warping vectors; "
            "F(xi) = sum_m v_m d_m(xi)^T with d_m = grad_xi e_m"
        ),
        "mode_scaled_by_two_over_h": True,
        "mode_packing_old": (
            "[..., 3, 7] with the seven vectors in columns; the 21 values are the row-major flatten, "
            "value index 7 i + m (i = component 0..2, m = mode 0..6)"
        ),
        "mode_packing_oracle": "[C, 7, 3] (vector m first, then component i); transpose of the old packing",
        "local_representation": "R^T V ([3, 7]) in the frozen frame R with world columns; the network sees its row-major flatten",
        "pin_convention": (
            "corners whose rest z equals the minimum rest z (the z-min face, iz = 0), all of them; "
            "prescribed positions default to the rest positions (zero displacement)"
        ),
        "corner_numbering": (
            "p = ix (ny + 1)(nz + 1) + iy (nz + 1) + iz; z fastest, x slowest; rest position = origin + h (ix, iy, iz), "
            "origin (0, 0, 0)"
        ),
        "cell_numbering": "c = ix ny nz + iy nz + iz; z fastest, x slowest",
        "exposed_face_order": ("-x", "+x", "-y", "+y", "-z", "+z"),
        "neighbor_slots": (
            "27 slots: slot 0 = self, slots 1..26 = offsets (dx, dy, dz) in lexicographic order over [-1, 1]^3 with "
            "max(|dx|, |dy|, |dz|) = 1; out-of-grid slots have index 0 and mask False (edge features zeroed)"
        ),
        "reference_corners_rule": (
            "among the pinned corners: p0 = lexicographically smallest rest position (x, then y, then z); "
            "p1 = farthest from p0; p2 = maximises |(p1 - p0) x (p2 - p0)|; ties (rel 1e-9) take the smallest corner ID; "
            "None when fewer than three noncollinear pins"
        ),
        "reference_corners_canonical_10x10x40": canonical_reference,
        "reference_corners_canonical_grid_coordinates": reference_coordinates,
        "reference_corners_2x2x3": [int(index) for index in small.reference_corners],
        "reference_rotation": (
            "from the CURRENT positions x0, x1, x2 of the three corners: e1 = normalize(x1 - x0), "
            "n = normalize(e1 x (x2 - x0)), e2 = n x e1; R_ref = [e1 e2 n] as columns"
        ),
        "frame_rule": (
            "U, s, Vh = svd(F_centre); d = sign det(U Vh) (0 -> +1); R = U diag(1, 1, d) Vh; "
            "gap = s2 + s3 (d > 0) or s2 - s3 (d < 0); tie if gap <= 1e-4 max(s1, 1); tie cells redo the formula on "
            "F + eps R_ref with eps = 1e-4 max(s1, 1) in float64; frames detached (frozen), recomputed per query"
        ),
        "frame_tie_tolerance": 1e-4,
        "energy_unit": "mu h^3 [J]; unit of axis_gradient_world (gradient feature) and of log_gradient_rms",
        "mass_lumping": (
            "m_p = sum over the cells incident to corner p of rho_c h^3 / 8 (each cell gives one eighth of its rest "
            "mass to each of its eight corners); pinned corners carry mass too; inertia = sum_p m_p |X_p - Y_p|^2 / (2 dt^2)"
        ),
        "lame_from_E_nu": "lambda = E nu / ((1 + nu)(1 - 2 nu)), mu = E / (2 (1 + nu)); stable NH uses mu_NH = mu, lambda_NH = lambda + mu",
        "conditioning_channels": tuple(CONDITIONING_CHANNELS),
        "node_feature_layout": node_feature_layout(),
        "edge_feature_layout": edge_feature_layout(),
        "contact_token_layout": contact_token_layout(),
    }


# -- geometry ----------------------------------------------------------------------------------------------


def rest_grid(cell_counts, h) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rest [P, 3] m, cells [C, 8] int64, pinned_mask [P] bool)`` of the canonical grid at origin 0."""
    grid = _grid(_counts(cell_counts), float(h))
    return grid.rest.corner_rest_positions.copy(), grid.rest.cell_corner_indices.copy(), grid.pinned_mask.copy()


# -- objective ---------------------------------------------------------------------------------------------


def energy_terms(cell_counts, h, dt, X, Y, X_prev, E, nu, rho, eta) -> dict[str, float]:
    """Return the incremental potential and its parts [J]: ``total, elastic, inertia, damping``.

    ``elastic = sum_q w_q psi(F_q)`` with the stable Neo-Hookean density (``mu_NH = mu``, ``lambda_NH = lambda +
    mu``) at the 8 Gauss points, ``w_q = h^3 / 8``; ``inertia = sum_p m_p |X_p - Y_p|^2 / (2 dt^2)`` with the
    lumped masses of :func:`conventions` (``mass_lumping``); ``damping = eta / (2 dt) sum_q w_q ||F_q^T F_q -
    F_prev,q^T F_prev,q||_F^2``. Gravity is inside ``Y``; no contact.
    """
    module = _energy_module(_counts(cell_counts), float(h), float(dt), float(E), float(nu), float(rho), float(eta))
    terms = module(_t(X)[None], _t(Y)[None], previous_positions=_t(X_prev)[None])
    return {
        "total": float(terms.total[0]),
        "elastic": float(terms.elastic[0]),
        "inertia": float(terms.inertia[0]),
        "damping": float(terms.damping[0]),
    }


def energy_and_grad(cell_counts, h, dt, X, Y, X_prev, E, nu, rho, eta) -> tuple[float, np.ndarray]:
    """Return ``(energy [J], dE/dX [N] of shape [P, 3])`` of the potential the old solver minimises.

    The gradient is the full autograd gradient, pinned rows included (the old feature pipeline zeroes the
    pinned rows before projecting; :func:`project_gradient` does that itself). See :func:`energy_terms`.
    """
    module = _energy_module(_counts(cell_counts), float(h), float(dt), float(E), float(nu), float(rho), float(eta))
    candidate = _t(X)[None].requires_grad_(True)
    total = module(candidate, _t(Y)[None], previous_positions=_t(X_prev)[None]).total.sum()
    (gradient,) = torch.autograd.grad(total, candidate)
    return float(total.detach()), _np(gradient[0])


# -- frames and modes --------------------------------------------------------------------------------------


def frames(cell_counts, h, X, *, return_tie_mask: bool = False):
    """Return the frozen per-cell frames ``R [C, 3, 3]`` (world columns) with the clamped-face tie-break.

    ``R`` is the closest proper rotation to the centre deformation gradient of each cell; tie cells use the
    reference rotation built from the current positions of :func:`reference_corners`. With
    ``return_tie_mask`` the boolean tie mask ``[C]`` is returned as well.
    """
    grid = _grid(_counts(cell_counts), float(h))
    positions = _t(X)[None]
    centre = mode_vectors(positions, grid.cells, float(h))[..., :3]
    reference = None
    if grid.reference_corners is not None:
        reference = reference_rotation(positions, torch.as_tensor(grid.reference_corners, dtype=torch.long))
    result = closest_proper_rotations(centre, reference)
    rotations = _np(result.frames[0])
    if return_tie_mask:
        return rotations, _np(result.tie_mask[0])
    return rotations


def modes(cell_counts, h, X) -> np.ndarray:
    """Return the seven world mode vectors ``[C, 7, 3]`` (``a1, a2, a3, w12, w13, w23, w123``, scaled ``2 / h``)."""
    grid = _grid(_counts(cell_counts), float(h))
    vectors = mode_vectors(_t(X)[None], grid.cells, float(h))[0]
    return _np(vectors.transpose(1, 2))


# -- fusion ------------------------------------------------------------------------------------------------


def fuse(cell_counts, h, X, dm_world) -> np.ndarray:
    """Return the corner displacement ``d [P, 3]`` [m] fusing the per-cell target increments ``dm_world [C, 7, 3]``.

    Weighted least squares over the 8 Gauss points of ``||grad(d)(xi_q) - Delta F_{c,q}||^2`` with
    ``Delta F_{c,q} = sum_m dm_{c,m} d_m(xi_q)^T``, pinned corners fixed at zero displacement (rows of ``d``
    exactly zero). The result does not depend on ``X`` (see the module docstring).
    """
    counts, size = _counts(cell_counts), float(h)
    grid = _grid(counts, size)
    positions = _t(X)[None]
    targets = _t(dm_world).transpose(1, 2).contiguous()[None]  # [1, C, 3, 7], old packing
    fused = _fusion(counts, size).fuse(positions, targets, positions[:, grid.fixed])
    return _np((fused - positions)[0])


def project_gradient(cell_counts, h, X, gX, *, energy_unit: float = 1.0) -> np.ndarray:
    """Return the fusion-adjoint projection ``[C, 7, 3]`` of a position gradient ``gX [P, 3]`` onto the modes.

    ``unpack(B^T K_ff^{-1} g_free)`` with the pinned rows of ``gX`` zeroed first (as the old
    ``objective_gradient`` does). In joules for ``energy_unit = 1``; the old feature pipeline divides by
    ``mu h^3`` (pass :func:`energy_unit`) to obtain ``axis_gradient_world``. Independent of ``X``.
    """
    counts, size = _counts(cell_counts), float(h)
    grid = _grid(counts, size)
    gradient = _t(gX).clone()
    gradient[grid.fixed] = 0
    projected = _fusion(counts, size).project_gradient(gradient[None])[0]  # [C, 3, 7]
    return _np(projected.transpose(1, 2) / float(energy_unit))


# -- network inputs and the old network ----------------------------------------------------------------------


def conditioning(
    h, dt, E, nu, rho, eta, gravity=(0.0, -9.81, 0.0), *, contact_ke=0.0, contact_kd=0.0, contact_mu=0.0
) -> np.ndarray:
    """Return the 7 dimensionless FiLM channels ``[7]`` in the order of ``conventions()['conditioning_channels']``."""
    lam, mu = lame(E, nu)

    def scalar(value):
        return torch.tensor([float(value)], dtype=_DTYPE)

    channels = conditioning_channels(
        scalar(lam),
        scalar(mu),
        scalar(rho),
        scalar(eta),
        float(h),
        float(dt),
        gravity,
        contact_ke=scalar(contact_ke),
        contact_kd=scalar(contact_kd),
        contact_mu=scalar(contact_mu),
    )
    return _np(channels[0])


def network_inputs(
    cell_counts, h, dt, X, Y, X_prev, E, nu, rho, eta, *, gravity=(0.0, -9.81, 0.0), history=None
) -> dict[str, np.ndarray]:
    """Run the old feature pipeline (``input_assembly.assemble_inputs``) on one contact-free object.

    ``history`` is None (no history: zero blocks, ``history_valid = 0``) or a pair ``(G_prev [C, 7, 3],
    U_prev [C, 7, 3])`` in world axes (``G_prev`` in energy units, as ``axis_gradient_world`` returns it).

    Returns numpy float64 arrays: ``frames [C, 3, 3]``, ``tie_mask [C]``, ``local_axes [C, 7, 3]`` (``R^T V``,
    oracle layout), ``state_features [C, 121]``, ``node_features [C, 159]`` (the exact old node input with 17
    zero contact channels), ``edge_features [C, 27, 24]``, ``neighbor_indices [C, 27]``, ``neighbor_mask
    [C, 27]``, ``conditioning [7]``, ``axis_gradient_world [C, 7, 3]`` (energy units), ``position_gradient
    [P, 3]`` [N] with pinned rows zeroed.
    """
    counts, size, step = _counts(cell_counts), float(h), float(dt)
    grid = _grid(counts, size)
    module = _energy_module(counts, size, step, float(E), float(nu), float(rho), float(eta))
    fusion = _fusion(counts, size)
    unit = energy_unit(size, E, nu)
    cell_count = len(grid.rest.cell_corner_indices)
    flags = torch.zeros(len(grid.rest.corner_rest_positions), dtype=_DTYPE)
    flags[grid.fixed] = 1
    boundary = torch.cat((torch.as_tensor(grid.rest.cell_exposed_faces, dtype=_DTYPE), flags[grid.cells]), -1)
    indices, mask = build_grid_neighborhood(counts, 1)
    geometry = SimpleNamespace(
        target_modes=MODE_COUNT,
        cell_corner_indices=grid.cells,
        center_gradients=corner_local_coordinates(dtype=_DTYPE) / (4 * size),
        rest_centers=torch.as_tensor(grid.rest.cell_rest_centers, dtype=_DTYPE),
        cell_size=size,
        reference_corners=(
            torch.empty(0, dtype=torch.long)
            if grid.reference_corners is None
            else torch.as_tensor(grid.reference_corners, dtype=torch.long)
        ),
        fixed_indices=torch.as_tensor(grid.fixed, dtype=torch.long),
        boundary_features=boundary,
        network=SimpleNamespace(hops=(1,), neighborhood=lambda hop: (indices, mask)),
    )
    channels = _t(conditioning(size, step, E, nu, rho, eta, gravity))
    optimizer_history = None
    if history is not None:
        previous_gradient, previous_update = history
        optimizer_history = OptimizerHistory(
            _t(previous_gradient).transpose(1, 2).contiguous()[None],
            _t(previous_update).transpose(1, 2).contiguous()[None],
            torch.ones(1, dtype=torch.bool),
        )
    inputs = assemble_inputs(
        geometry,
        _t(X)[None],
        _t(Y)[None],
        _t(X_prev)[None],
        energy_total=lambda candidate, target, previous: module(candidate, target, previous_positions=previous).total,
        project_gradient=lambda gradient: fusion.project_gradient(gradient) / unit,
        conditioning=channels[None, None].expand(1, cell_count, -1),
        history=optimizer_history,
    )
    local_axes = inputs.local_axes[0]
    node = torch.cat(
        (local_axes.flatten(-2), inputs.state_features[0], torch.zeros(cell_count, CONTACT_FEATURE_DIM, dtype=_DTYPE)),
        dim=-1,
    )
    return {
        "frames": _np(inputs.frames[0]),
        "tie_mask": _np(inputs.tie_mask[0]),
        "local_axes": _np(local_axes.transpose(1, 2)),
        "state_features": _np(inputs.state_features[0]),
        "node_features": _np(node),
        "edge_features": _np(inputs.edge_features[1][0]),
        "neighbor_indices": _np(indices),
        "neighbor_mask": _np(mask),
        "conditioning": _np(channels),
        "axis_gradient_world": _np(inputs.axis_gradient_world[0].transpose(1, 2)),
        "position_gradient": _np(inputs.position_gradient[0]),
    }


def network_state_dict_summary(checkpoint_path=V4_CHECKPOINT) -> list[tuple[str, tuple[int, ...]]]:
    """Return ``(name, shape)`` of every entry of the checkpoint's ``network_state`` in state_dict order."""
    state = _checkpoint(str(checkpoint_path))
    state = state["network_state"] if "network_state" in state else state
    return [(name, tuple(int(size) for size in tensor.shape)) for name, tensor in state.items()]


def _old_network(checkpoint_path, cell_counts) -> IntrinsicSolverNetwork:
    checkpoint = _checkpoint(str(checkpoint_path))
    config = checkpoint["config"]
    network = IntrinsicSolverNetwork(
        cell_counts,
        state_feature_dim(config["target_modes"]),
        target_modes=config["target_modes"],
        conditioning_dim=len(CONDITIONING_CHANNELS),
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        edge_input_dim=EDGE_FEATURE_DIM,
        edge_hidden_dim=config["edge_hidden_dim"],
        hops=tuple(config["hops"]),
        max_step_size=config["max_step_size"],
        query_chunk_size=config["query_chunk_size"],
        edge_network=config["edge_network"],
        checkpoint_chunks=config["checkpoint_chunks"],
        contact_tokens=config["contact"],
    )
    state = {name: tensor for name, tensor in checkpoint["network_state"].items() if not name.startswith("neighbor_")}
    missing, unexpected = network.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("neighbor_") for name in missing):
        raise RuntimeError(f"checkpoint does not match the old network: missing {missing}, unexpected {unexpected}")
    return network.eval()


def network_forward(checkpoint_path, cell_counts, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run the OLD network (v4 weights, float32, CPU) on the output of :func:`network_inputs`.

    Contact-free: the contact encoder receives no tokens (its 17 channels are zero, as in training on
    contact-free objects). Returns ``local_target [C, 7, 3]`` (``R^T V`` plus the update, local frame),
    ``correction [C, 7, 3]`` (bounded, joint norm below one) and ``step_size [C]`` in ``(0, 0.05)``.
    """
    counts = _counts(cell_counts)
    network = _old_network(checkpoint_path, counts)
    local_axes = _t(inputs["local_axes"], torch.float32).transpose(1, 2).contiguous()[None]
    state = _t(inputs["state_features"], torch.float32)[None]
    edges = {1: _t(inputs["edge_features"], torch.float32)[None]}
    channels = _t(inputs["conditioning"], torch.float32)[None]
    with torch.no_grad():
        output = network(local_axes, state, edges, channels)
    return {
        "local_target": _np(output.local_target_axes[0].transpose(1, 2)),
        "correction": _np(output.axis_correction[0].transpose(1, 2)),
        "step_size": _np(output.step_size[0]),
    }
