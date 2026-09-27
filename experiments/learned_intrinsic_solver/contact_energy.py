# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental penalty contact energy whose negative gradient is Newton's VBD contact force.

The energy reproduces ``_compute_body_particle_contact_force`` from
:mod:`newton._src.solvers.vbd.rigid_vbd_kernels` in its quadratic penalty
branch (``use_log_barrier=False``) for a static partner. For a surface sample
at ``x`` with step-start position ``x0``, partner point ``p``, unit partner
normal ``n``, sample radius ``r`` and ``delta = x - x0``:

- penetration depth ``d = r - (x - p) . n``; Newton's penalty force norm is
  ``-dE/dd = ke * d`` with no additional factor, so the normal energy is
  ``E_n = ke / 2 * relu(d)^2`` whose spatial force is ``ke * relu(d) * n``;
- damping ``E_d = kd / (2 dt) * relu(-(n . delta))^2`` gives Newton's
  ``-(kd / dt) (n . delta) n`` while approaching and nothing while separating;
- friction uses ``f_n = ke * relu(d)`` as a detached constant, exactly like
  Newton, which treats the normal load as constant when differentiating the
  friction term. With ``u = delta - (n . delta) n``, ``eps_u = friction_epsilon
  * dt`` and the IPC smoothing ``f0(y) = -y^3 / (3 eps_u^2) + y^2 / eps_u +
  eps_u / 3`` for ``y < eps_u`` and ``f0(y) = y`` otherwise,
  ``E_f = mu * f_n * f0(|u|)``. Its gradient is ``mu f_n (f1(y) / y) u`` with
  ``f1(y) / y = (-y / eps_u + 2) / eps_u`` inside the band and ``1 / y``
  outside, matching ``compute_projected_isotropic_friction``.

Newton's caller ``_eval_body_particle_contact`` evaluates the force law only
while the current penetration depth is positive. The damping and friction terms
therefore switch on with ``d > 0`` here as well, so a pair that does not
penetrate contributes exactly zero energy and zero gradient regardless of its
motion. The normal is used as given; callers supply unit normals.

Masked pairs contribute exactly zero even when their payload rows hold NaN or
an index of -1: the payload is replaced by finite placeholders before any
arithmetic, and the slip norm is guarded so ``|u| = 0`` has a finite gradient.
Units are SI: metres, seconds, newtons per metre for ``ke``, newton seconds per
metre for ``kd``; energies are in joules.
"""

from __future__ import annotations

import math

import torch  # noqa: TID253 -- Explicit opt-in PyTorch implementation.
from torch import Tensor  # noqa: TID253

__all__ = ["contact_energy", "contact_penetration"]


def _check_scalar(value: float, name: str, *, allow_zero: bool) -> float:
    """Return ``value`` as a float after checking finiteness and sign."""
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {bound} scalar")
    return result


def _batch_parameter(value: Tensor | float, batch: int, reference: Tensor, name: str) -> Tensor:
    """Return ``value`` as a tensor of shape ``[B]`` matching ``reference``'s dtype and device."""
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tensor.shape != (batch,):
        raise ValueError(f"{name} must have shape [B] = [{batch}]")
    return tensor


def _prepare_pairs(
    sample_positions: Tensor,
    sample_index: Tensor,
    partner_point: Tensor,
    partner_normal: Tensor,
    pair_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Validate shapes and return ``(safe_index, x, p, n, mask)`` with finite masked rows.

    Args:
        sample_positions: Sample positions [m], shape [B,S,3].
        sample_index: Sample id of each pair, shape [B,Q]; ignored where masked.
        partner_point: Partner point [m], shape [B,Q,3]; ignored where masked.
        partner_normal: Unit partner normal, shape [B,Q,3]; ignored where masked.
        pair_mask: True for valid pairs, shape [B,Q].

    Returns:
        The sanitized index [B,Q], the gathered sample positions [B,Q,3], the
        sanitized partner points and normals [B,Q,3], and the boolean mask.
    """
    if sample_positions.ndim != 3 or sample_positions.shape[-1] != 3:
        raise ValueError("sample_positions must have shape [B,S,3]")
    batch, sample_count, _ = sample_positions.shape
    if sample_index.ndim != 2 or sample_index.shape[0] != batch:
        raise ValueError("sample_index must have shape [B,Q]")
    if sample_index.dtype not in (torch.int64, torch.int32):
        raise ValueError("sample_index must be an integer tensor")
    pair_shape = tuple(sample_index.shape)
    if tuple(partner_point.shape) != (*pair_shape, 3):
        raise ValueError("partner_point must have shape [B,Q,3]")
    if tuple(partner_normal.shape) != (*pair_shape, 3):
        raise ValueError("partner_normal must have shape [B,Q,3]")
    if tuple(pair_mask.shape) != pair_shape or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must be a boolean tensor of shape [B,Q]")
    for name, tensor in (("partner_point", partner_point), ("partner_normal", partner_normal)):
        if tensor.dtype != sample_positions.dtype or tensor.device != sample_positions.device:
            raise ValueError(f"{name} must share the dtype and device of sample_positions")

    index = sample_index.to(torch.int64)
    mask = pair_mask
    invalid = mask & ((index < 0) | (index >= sample_count))
    if bool(invalid.any()):
        raise ValueError("sample_index of valid pairs must lie in [0, S)")
    safe_index = torch.where(mask, index, torch.zeros_like(index))
    gathered = torch.gather(sample_positions, 1, safe_index.unsqueeze(-1).expand(-1, -1, 3))
    row_mask = mask.unsqueeze(-1)
    point = torch.where(row_mask, partner_point, torch.zeros_like(partner_point))
    normal = torch.where(row_mask, partner_normal, torch.zeros_like(partner_normal))
    return safe_index, gathered, point, normal, mask


def _penetration_depth(gathered: Tensor, point: Tensor, normal: Tensor, radius: float) -> Tensor:
    """Return the signed penetration depth ``r - (x - p) . n`` with shape [B,Q]."""
    return radius - ((gathered - point) * normal).sum(-1)


def contact_penetration(
    sample_positions: Tensor,
    sample_index: Tensor,
    partner_point: Tensor,
    partner_normal: Tensor,
    pair_mask: Tensor,
    *,
    radius: float,
) -> Tensor:
    """Return the positive penetration depth ``relu(r - (x - p) . n)`` of every pair.

    Experimental. Diagnostics companion of :func:`contact_energy`; masked pairs
    report exactly zero.

    Args:
        sample_positions: Sample positions [m], shape [B,S,3].
        sample_index: Sample id of each pair, shape [B,Q]; ignored where masked.
        partner_point: Partner point [m], shape [B,Q,3]; ignored where masked.
        partner_normal: Unit partner normal, shape [B,Q,3]; ignored where masked.
        pair_mask: True for valid pairs, shape [B,Q].
        radius: Sample radius [m], finite and non-negative.

    Returns:
        Penetration depth [m] per pair, shape [B,Q].
    """
    radius = _check_scalar(radius, "radius", allow_zero=True)
    _, gathered, point, normal, mask = _prepare_pairs(
        sample_positions, sample_index, partner_point, partner_normal, pair_mask
    )
    depth = torch.relu(_penetration_depth(gathered, point, normal, radius))
    return torch.where(mask, depth, torch.zeros_like(depth))


def contact_energy(
    sample_positions: Tensor,
    sample_start_positions: Tensor,
    sample_index: Tensor,
    partner_point: Tensor,
    partner_normal: Tensor,
    pair_mask: Tensor,
    *,
    radius: float,
    ke: Tensor | float,
    kd: Tensor | float,
    mu: Tensor | float,
    time_step: float,
    friction_epsilon: float = 1e-2,
) -> Tensor:
    """Return the penalty contact energy [J] of every batch member, shape [B].

    Experimental. The negative gradient with respect to ``sample_positions``
    equals Newton's ``_compute_body_particle_contact_force`` for each pair
    (quadratic penalty, static partner); see the module docstring for the exact
    terms. ``sample_positions`` and ``sample_start_positions`` retain their
    autograd history; the friction normal load is detached.

    Args:
        sample_positions: Current sample positions [m], shape [B,S,3].
        sample_start_positions: Step-start sample positions [m], same shape;
            the friction and damping anchor.
        sample_index: Sample id of each pair, shape [B,Q]; ignored where masked.
        partner_point: Partner point [m], shape [B,Q,3]; ignored where masked.
        partner_normal: Unit partner normal, shape [B,Q,3]; ignored where masked.
        pair_mask: True for valid pairs, shape [B,Q].
        radius: Sample radius [m], finite and non-negative.
        ke: Normal penalty stiffness [N/m] per batch member, shape [B] or scalar.
        kd: Normal damping [N s/m] per batch member, shape [B] or scalar.
        mu: Friction coefficient per batch member, shape [B] or scalar.
        time_step: Physical step [s], finite and positive.
        friction_epsilon: IPC smoothing band as a fraction of ``time_step``;
            ``eps_u = friction_epsilon * time_step`` in metres.

    Returns:
        Contact energy [J], shape [B]; exactly zero for members without a
        penetrating valid pair.
    """
    radius = _check_scalar(radius, "radius", allow_zero=True)
    time_step = _check_scalar(time_step, "time_step", allow_zero=False)
    friction_epsilon = _check_scalar(friction_epsilon, "friction_epsilon", allow_zero=False)
    if sample_start_positions.shape != sample_positions.shape:
        raise ValueError("sample_start_positions must have the same shape as sample_positions")
    if (
        sample_start_positions.dtype != sample_positions.dtype
        or sample_start_positions.device != sample_positions.device
    ):
        raise ValueError("sample_start_positions must share the dtype and device of sample_positions")
    safe_index, current, point, normal, mask = _prepare_pairs(
        sample_positions, sample_index, partner_point, partner_normal, pair_mask
    )
    batch = sample_positions.shape[0]
    stiffness = _batch_parameter(ke, batch, sample_positions, "ke").unsqueeze(-1)
    damping = _batch_parameter(kd, batch, sample_positions, "kd").unsqueeze(-1)
    friction = _batch_parameter(mu, batch, sample_positions, "mu").unsqueeze(-1)

    start = torch.gather(sample_start_positions, 1, safe_index.unsqueeze(-1).expand(-1, -1, 3))
    depth = _penetration_depth(current, point, normal, radius)
    penetration = torch.relu(depth)
    penetrating = depth > 0
    translation = current - start
    normal_translation = (normal * translation).sum(-1)

    normal_energy = 0.5 * stiffness * penetration.square()

    approach = torch.relu(-normal_translation)
    damping_energy = (0.5 * damping / time_step) * approach.square()
    damping_energy = torch.where(penetrating, damping_energy, torch.zeros_like(damping_energy))

    slip = translation - normal_translation.unsqueeze(-1) * normal
    slip_sq = slip.square().sum(-1)
    # clamp_min keeps sqrt' finite at zero slip; its gradient vanishes in the clamped region.
    slip_norm = torch.sqrt(slip_sq.clamp_min(torch.finfo(slip_sq.dtype).tiny))
    eps_u = friction_epsilon * time_step
    smoothed = -slip_norm.pow(3) / (3.0 * eps_u * eps_u) + slip_norm.square() / eps_u + eps_u / 3.0
    smoothing = torch.where(slip_norm < eps_u, smoothed, slip_norm)
    normal_load = (stiffness * penetration).detach()
    friction_energy = friction * normal_load * smoothing

    pair_energy = normal_energy + damping_energy + friction_energy
    pair_energy = torch.where(mask, pair_energy, torch.zeros_like(pair_energy))
    return pair_energy.sum(-1)
