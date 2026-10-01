# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incremental potential in normalised units (design spec 4.3): stable Neo-Hookean with Newton's parameter
mapping at 8 Gauss points, VBD metric damping against the step-start shape, inertia against Y, contact."""

from __future__ import annotations

import torch

from . import contact as _contact
from . import hex as hx
from .contact import contact_energy
from .energy_kernel import elastic_damping_warp, energy_and_grad_warp

Tensor = torch.Tensor

USE_WARP = True  # float32 CUDA elastic + damping energy through the Warp kernel; the torch path stays the reference


def seg_sum(values: Tensor, seg: Tensor, count: int) -> Tensor:
    return torch.zeros(count, dtype=values.dtype, device=values.device).index_add_(0, seg, values)


def stable_neo_hookean(F: Tensor, lam: Tensor) -> Tensor:
    """Energy density with mu = 1 and lambda_NH = lam + 1 (Newton mapping); zero at F = I. F [...,3,3], lam broadcast."""
    I_C = (F * F).sum((-1, -2))
    J = torch.linalg.det(F)
    lam_nh = lam + 1.0
    return 0.5 * (I_C - 3.0) + 0.5 * lam_nh * (J - 1.0) ** 2 - (J - 1.0)


def neo_hookean_stress(F: Tensor, lam: Tensor) -> Tensor:
    """Newton's stress mu F + (lambda_NH (J - 1) - mu) cof(F) with mu = 1 (test reference)."""
    J = torch.linalg.det(F)
    cof = J[..., None, None] * torch.linalg.inv(F).transpose(-1, -2)
    return F + ((lam + 1.0) * (J - 1.0) - 1.0)[..., None, None] * cof


def modes_and_center(x: Tensor, batch) -> tuple[Tensor, Tensor]:
    m = hx.modes(x[batch.cells], batch.hc)
    return m, hx.center_deformation(m)


def elastic_damping(batch, x: Tensor) -> tuple[Tensor, Tensor]:
    """Per-cell elastic and damping energies [C] at the corners x."""
    F = hx.gauss_deformation(x.index_select(0, batch.cells.reshape(-1)).view(-1, 8, 3), batch.hc)  # [C,8,3,3]
    lam = batch.material.lam[batch.cell_obj]
    w = batch.hc.weights
    E_el = (w * stable_neo_hookean(F, lam[:, None])).sum(-1)
    C_now = hx.mat3_tn(F, F)
    eta = batch.material.eta[batch.cell_obj]
    E_damp = 0.5 * eta * (w * ((C_now - batch.C_prev) ** 2).sum((-1, -2))).sum(-1)
    return E_el, E_damp


def inertia(batch, x: Tensor) -> Tensor:
    """Per-corner inertia [N]; pinned rows contribute nothing."""
    rho = batch.material.rho[batch.corner_obj]
    e = 0.5 * rho * batch.mass * ((x - batch.Y) ** 2).sum(-1)
    return e.masked_fill(batch.pinned, 0.0)


def elastic_damping_total(batch, x: Tensor) -> Tensor:
    """Per-cell elastic + damping energy [C]: the Warp kernel on float32 CUDA, the torch path otherwise."""
    if USE_WARP and x.is_cuda and x.dtype == torch.float32:
        return elastic_damping_warp(batch, x)
    E_el, E_damp = elastic_damping(batch, x)
    return E_el + E_damp


def energy(batch, x: Tensor) -> Tensor:
    """Incremental potential per object [O] in mu h^3 units."""
    E = seg_sum(elastic_damping_total(batch, x), batch.cell_obj, batch.O) + seg_sum(
        inertia(batch, x), batch.corner_obj, batch.O
    )
    return E + contact_energy(batch, x)


def fused_pass_applies(batch, x: Tensor) -> bool:
    """The fused Warp pass (`energy_kernel.energy_and_grad_warp`) covers float32 CUDA candidates without a graph
    whose contact pairs are absent or in the capacity layout (the inference path); training keeps autograd."""
    pairs = batch.pairs
    return (
        USE_WARP
        and x.is_cuda
        and x.dtype == torch.float32
        and not x.requires_grad
        and (pairs is None or pairs.count == 0 or (pairs.padded and _contact.USE_WARP))
    )


def energy_and_grad(batch, x: Tensor) -> tuple[Tensor, Tensor]:
    """E [O] (with graph if x has one) and the detached position gradient [N,3] with pinned rows zeroed."""
    if not x.requires_grad:
        if fused_pass_applies(batch, x):
            return energy_and_grad_warp(batch, x)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            E = energy(batch, x)
            (gX,) = torch.autograd.grad(E.sum(), x)
        return E.detach(), gX.masked_fill(batch.pinned[:, None], 0.0)
    E = energy(batch, x)
    (gX,) = torch.autograd.grad(E.sum(), x, retain_graph=True)
    return E, gX.detach().masked_fill(batch.pinned[:, None], 0.0)


def residual(gX: Tensor, batch) -> Tensor:
    """Free-corner gradient norm per object [O] (normalised force units)."""
    return seg_sum((gX * gX).sum(-1).masked_fill_(batch.pinned, 0.0), batch.corner_obj, batch.O).sqrt()


def corner_mass(batch) -> Tensor:
    """Normalised corner masses m_i = rho_obj mass_i [N] (derivation section 1; `mass` is lumped in h^3 units)."""
    return batch.material.rho[batch.corner_obj] * batch.mass


def total_mass(batch) -> Tensor:
    """M_tot = sum_i m_i per object [O]."""
    return seg_sum(corner_mass(batch), batch.corner_obj, batch.O)


def centroid(batch, x: Tensor) -> Tensor:
    """Mass-weighted centroid c(x) = sum_i m_i x_i / M_tot per object [O,3] (eq. 1.10); the same linear map applied
    to velocities gives the centroid velocity."""
    m = corner_mass(batch)
    num = torch.zeros(batch.O, 3, dtype=x.dtype, device=x.device).index_add_(0, batch.corner_obj, m[:, None] * x)
    return num / seg_sum(m, batch.corner_obj, batch.O)[:, None]


def inverted_cells(F_center: Tensor, batch) -> Tensor:
    return seg_sum((torch.linalg.det(F_center) <= 0).to(F_center.dtype), batch.cell_obj, batch.O)
