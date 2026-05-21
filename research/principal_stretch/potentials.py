# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Incremental potential and elastic energies for self-supervised training.

L_total(x_{t+1}) = L_inertia + L_StVK + L_gravity + L_ext + L_pin

Variational implicit Euler: minimising L_total wrt x_{t+1} solves
    M (x_{t+1} - 2 x_t + x_{t-1}) / dt^2 = - dW/dx + f_ext + m g
which is one backward-Euler step of the elastic body.
"""

from __future__ import annotations

import torch

from .torch_solver import compute_F


def stvk_energy(F: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor,
                volume: torch.Tensor) -> torch.Tensor:
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
    x_next: torch.Tensor,         # (V, 3)
    x_t: torch.Tensor,            # (V, 3)
    x_prev: torch.Tensor,         # (V, 3)
    mass: torch.Tensor,           # (V,)
    gravity: torch.Tensor,        # (3,)
    f_ext: torch.Tensor,          # (V, 3)
    tets: torch.Tensor,           # (T, 4)
    J: torch.Tensor,              # (T, 4, 3)
    mu: torch.Tensor,             # (T,)
    lam: torch.Tensor,            # (T,)
    volume: torch.Tensor,         # (T,)
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
