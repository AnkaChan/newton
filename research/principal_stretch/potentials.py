# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Incremental potentials and elastic energies for stretch-solver research.

The stable Neo-Hookean functions mirror the volumetric material used by
``SolverVBD``.  The older St. Venant--Kirchhoff helpers remain available for
reproducing the original self-supervised experiments, but they must not be
mixed with VBD convergence measurements.

Variational implicit Euler: minimising L_total wrt x_{t+1} solves
    M (x_{t+1} - 2 x_t + x_{t-1}) / dt^2 = - dW/dx + f_ext + m g
which is one backward-Euler step of the elastic body.
"""

from __future__ import annotations

import math

import torch

from .torch_solver import compute_F


def _determinant_3x3(matrix: torch.Tensor) -> torch.Tensor:
    """Polynomial 3x3 determinant with finite derivatives at rank loss.

    ``torch.linalg.det`` has a finite first derivative at a singular 3x3
    matrix, but its generic backward-of-backward path can use an inverse and
    return NaN. Newton needs the Hessian at exactly flat/inverted tets, so keep
    the determinant in its explicit cubic polynomial form.
    """
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 0, 2]
    d = matrix[..., 1, 0]
    e = matrix[..., 1, 1]
    f = matrix[..., 1, 2]
    g = matrix[..., 2, 0]
    h = matrix[..., 2, 1]
    i = matrix[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def stable_neo_hookean_energy_density(
    F: torch.Tensor,
    mu: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Newton VBD's volumetric stable Neo-Hookean density.

    The density is

    ``0.5 * mu * (tr(F F^T) - 3) + 0.5 * lam * (det(F) - alpha)^2``

    where ``alpha = 1 + mu / max(lam, 1e-6)``.  For the physical parameter
    range ``lam >= 1e-6`` this is the usual ``alpha = 1 + mu / lam`` and the
    first Piola stress is zero at ``F = I``. The energy itself has a harmless
    material-dependent constant at rest; it is intentionally retained to
    match the VBD implementation at this worktree's baseline revision. Here
    ``lam`` is the coefficient stored and consumed directly by that kernel; it
    is not the later PR #2901 branch's converted
    ``lambda_NH = lambda_Lame + mu`` convention. A row with ``mu = lam = 0``
    denotes a disabled tet and returns zero, matching the solver's dispatch.

    Args:
        F: Deformation gradients, shape ``[..., T, 3, 3]``.
        mu: First Lamé-like material coefficient, broadcastable to
            ``F.shape[:-2]``.
        lam: Second Lamé-like material coefficient, positive for active tets
            and broadcastable to ``F.shape[:-2]``.

    Returns:
        Energy density for each deformation gradient, shape ``F.shape[:-2]``.

    Raises:
        ValueError: If a material coefficient is non-finite or negative, or an
            active tet has non-positive ``lam``.
    """
    if F.shape[-2:] != (3, 3):
        raise ValueError(f"F must end in (3, 3), got {tuple(F.shape)}")
    if not torch.isfinite(mu).all() or not torch.isfinite(lam).all():
        raise ValueError("stable Neo-Hookean material coefficients must be finite")
    if (mu < 0.0).any() or (lam < 0.0).any():
        raise ValueError("stable Neo-Hookean material coefficients must be non-negative")
    active = (mu > 0.0) | (lam > 0.0)
    if (active & (lam <= 0.0)).any():
        raise ValueError("stable Neo-Hookean lambda must be positive on active tets")

    lam_safe = torch.clamp_min(lam, 1.0e-6)
    alpha = 1.0 + mu / lam_safe
    frob_sq = (F * F).sum(dim=(-2, -1))
    det_f = _determinant_3x3(F)
    density = 0.5 * mu * (frob_sq - 3.0) + 0.5 * lam * (det_f - alpha) ** 2
    return torch.where(active, density, torch.zeros_like(density))


def stable_neo_hookean_energy(
    F: torch.Tensor,
    mu: torch.Tensor,
    lam: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Sum stable Neo-Hookean elastic energy over the final tet axis.

    Args:
        F: Deformation gradients, shape ``[..., T, 3, 3]``.
        mu: Per-tet first material coefficient, shape ``[T]``.
        lam: Per-tet second material coefficient, shape ``[T]``.
        volume: Positive per-tet rest volumes [m^3], shape ``[T]``.

    Returns:
        Per-sample energy, shape ``F.shape[:-3]``.  A non-batched input returns
        a scalar tensor.
    """
    if (volume <= 0.0).any() or not torch.isfinite(volume).all():
        raise ValueError("rest volumes must be finite and strictly positive")
    density = stable_neo_hookean_energy_density(F, mu, lam)
    return (density * volume).sum(dim=-1)


def incremental_potential_stable_neo_hookean(
    x_next: torch.Tensor,
    inertial_target: torch.Tensor,
    mass: torch.Tensor,
    tets: torch.Tensor,
    J: torch.Tensor,
    mu: torch.Tensor,
    lam: torch.Tensor,
    volume: torch.Tensor,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Evaluate the common contact-free implicit-Euler potential.

    Gravity and external force are incorporated in ``inertial_target``:

    ``y = x_n + dt * v_n + dt^2 * (gravity + f_ext / mass)``.

    Dirichlet constraints are imposed by callers through free-variable
    elimination rather than a penalty term.

    Args:
        x_next: Candidate positions [m], shape ``[V, 3]``.
        inertial_target: Force-shifted inertial target positions [m], shape
            ``[V, 3]``.
        mass: Lumped vertex masses [kg], shape ``[V]``.
        tets: Tet vertex indices, shape ``[T, 4]``.
        J: Shape-function gradients [1/m], shape ``[T, 4, 3]``.
        mu: Per-tet first material coefficient [Pa], shape ``[T]``.
        lam: Per-tet second material coefficient [Pa], shape ``[T]``.
        volume: Per-tet rest volume [m^3], shape ``[T]``.
        dt: Substep duration [s].

    Returns:
        Scalar ``total``, ``inertia``, and ``elastic`` tensors.  Component
        entries are not detached so derivative tests can inspect them.

    Raises:
        ValueError: If ``dt`` is non-positive or inputs have incompatible
            shapes.
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if x_next.ndim != 2 or x_next.shape[-1] != 3:
        raise ValueError(f"x_next must have shape (V, 3), got {tuple(x_next.shape)}")
    if inertial_target.shape != x_next.shape:
        raise ValueError("inertial_target must have the same shape as x_next")
    if mass.shape != x_next.shape[:1]:
        raise ValueError("mass must have shape (V,)")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tets must have shape (T, 4), got {tuple(tets.shape)}")
    n_tets = tets.shape[0]
    expected_shapes = {
        "J": (n_tets, 4, 3),
        "mu": (n_tets,),
        "lam": (n_tets,),
        "volume": (n_tets,),
    }
    for name, tensor in (("J", J), ("mu", mu), ("lam", lam), ("volume", volume)):
        if tensor.shape != expected_shapes[name]:
            raise ValueError(f"{name} must have shape {expected_shapes[name]}, got {tuple(tensor.shape)}")
    if tets.dtype != torch.int64:
        raise ValueError(f"tets must have dtype torch.int64, got {tets.dtype}")
    tensors = (inertial_target, mass, tets, J, mu, lam, volume)
    if any(tensor.device != x_next.device for tensor in tensors):
        raise ValueError("all common-potential tensors must be on the same device")
    floating_tensors = (inertial_target, mass, J, mu, lam, volume)
    if not x_next.is_floating_point() or any(tensor.dtype != x_next.dtype for tensor in floating_tensors):
        raise ValueError("all floating common-potential tensors must share x_next's floating dtype")
    if not torch.isfinite(inertial_target).all() or not torch.isfinite(mass).all() or not torch.isfinite(J).all():
        raise ValueError("inertial_target, mass, and J must be finite")
    if (mass < 0.0).any():
        raise ValueError("mass must be non-negative")

    delta = x_next - inertial_target
    inertia = 0.5 / (dt * dt) * (mass[:, None] * delta * delta).sum()
    F = compute_F(x_next, tets, J)
    elastic = stable_neo_hookean_energy(F, mu, lam, volume)
    return {"total": inertia + elastic, "inertia": inertia, "elastic": elastic}


def incremental_potential_batched(
    x_next: torch.Tensor,  # (B, V, 3)
    x_t: torch.Tensor,  # (B, V, 3)
    x_prev: torch.Tensor,  # (B, V, 3)
    mass: torch.Tensor,  # (V,)
    gravity: torch.Tensor,  # (3,)
    f_ext: torch.Tensor,  # (B, V, 3)
    tets: torch.Tensor,  # (T, 4)
    J: torch.Tensor,  # (T, 4, 3)
    mu: torch.Tensor,  # (T,)
    lam: torch.Tensor,  # (T,)
    volume: torch.Tensor,  # (T,)
    dt: float,
) -> torch.Tensor:
    """Per-sample incremental potential, summed across the batch (scalar).

    The pin penalty of :func:`incremental_potential` is omitted: it weights by
    ``mass[pin_idx]`` and pinned particles are pinned *by having zero mass*, so
    the term is identically zero, and the decoder hard-pins anyway.
    """
    inv_dt2 = 1.0 / (dt * dt)
    delta = x_next - 2.0 * x_t + x_prev  # (B, V, 3)
    L_inertia = 0.5 * inv_dt2 * (mass[None, :, None] * delta * delta).sum(dim=(-2, -1))

    x_tet = x_next[:, tets]  # (B, T, 4, 3)
    F = torch.einsum("tac,btad->btdc", J, x_tet)  # (B, T, 3, 3)
    Ft = F.transpose(-1, -2)
    eye = torch.eye(3, dtype=F.dtype, device=F.device).expand_as(F)
    E = 0.5 * (Ft @ F - eye)
    tr_E = E.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, T)
    frob_E2 = (E * E).sum(dim=(-2, -1))  # (B, T)
    L_elastic = ((mu * frob_E2 + 0.5 * lam * tr_E * tr_E) * volume).sum(dim=-1)  # (B,)

    L_gravity = -(mass[None, :, None] * x_next * gravity[None, None, :]).sum(dim=(-2, -1))
    L_ext = -(f_ext * x_next).sum(dim=(-2, -1))

    return (L_inertia + L_elastic + L_gravity + L_ext).sum()


def stvk_energy(F: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    """Per-tet St. Venant-Kirchhoff strain energy, summed.

    Args:
        F: (T, 3, 3) deformation gradients.
        mu, lam: (T,) Lame parameters per tet.
        volume: (T,) rest volumes.
    """
    Ft = F.transpose(-1, -2)
    eye = torch.eye(3, dtype=F.dtype, device=F.device).expand_as(F)
    E = 0.5 * (Ft @ F - eye)
    tr_E = E.diagonal(dim1=-2, dim2=-1).sum(-1)
    frob_E2 = (E * E).sum(dim=(-2, -1))
    psi = mu * frob_E2 + 0.5 * lam * tr_E * tr_E
    return (psi * volume).sum()


def incremental_potential(
    x_next: torch.Tensor,  # (V, 3)
    x_t: torch.Tensor,  # (V, 3)
    x_prev: torch.Tensor,  # (V, 3)
    mass: torch.Tensor,  # (V,)
    gravity: torch.Tensor,  # (3,)
    f_ext: torch.Tensor,  # (V, 3)
    tets: torch.Tensor,  # (T, 4)
    J: torch.Tensor,  # (T, 4, 3)
    mu: torch.Tensor,  # (T,)
    lam: torch.Tensor,  # (T,)
    volume: torch.Tensor,  # (T,)
    dt: float,
    pin_idx: torch.Tensor | None = None,
    pin_target: torch.Tensor | None = None,
    pin_weight: float = 1e6,
) -> dict[str, torch.Tensor]:
    inv_dt2 = 1.0 / (dt * dt)
    delta = x_next - 2.0 * x_t + x_prev  # (V, 3)
    L_inertia = 0.5 * inv_dt2 * (mass[:, None] * delta * delta).sum()

    F = compute_F(x_next, tets, J)
    L_elastic = stvk_energy(F, mu, lam, volume)

    L_gravity = -(mass[:, None] * x_next * gravity[None, :]).sum()
    L_ext = -(f_ext * x_next).sum()

    if pin_idx is not None and pin_idx.numel() > 0:
        diff = x_next[pin_idx] - pin_target
        L_pin = 0.5 * pin_weight * (mass[pin_idx, None] * diff * diff).sum()
    else:
        L_pin = torch.zeros((), dtype=x_next.dtype, device=x_next.device)

    L_total = L_inertia + L_elastic + L_gravity + L_ext + L_pin
    return {
        "total": L_total,
        "inertia": L_inertia.detach(),
        "elastic": L_elastic.detach(),
        "gravity": L_gravity.detach(),
        "ext": L_ext.detach(),
        "pin": L_pin.detach(),
    }
