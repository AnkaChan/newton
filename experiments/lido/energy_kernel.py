# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Elastic + damping energy per cell with analytic corner forces: one float32 Warp kernel behind a
torch.autograd.Function, so inference and training (dE/dx through the fused candidate) share one call; plus
the fused inference pass `energy_and_grad_warp` (per-object energies and the full corner gradient in three
launches).

Per cell the kernel gathers the 8 corners and at each of the 8 Gauss points forms F[r,a] = sum_k x_k[r] Gq[q,k,a],
psi = 0.5 (I_C - 3) + 0.5 lam_nh (J - 1)^2 - (J - 1) with mu = 1, lam_nh = lam + 1 (ADR 0002),
P = F + (lam_nh (J - 1) - 1) cof(F); damping D = F^T F - C_prev, psi_d = 0.5 eta |D|_F^2, dpsi_d/dF = 2 eta F D.
E_c = mu_scale sum_q w_q (psi + psi_d) and g_k = mu_scale sum_q w_q (P + 2 eta F D) Gq[q,k] with w_q = 1/8 and
mu_scale = mu / mu_norm the object's factor into the common unit of energy (1 in body mode; physics.py). The
material is read per object (lam, eta, mu_scale [O] through cell_obj).

Two output modes. Per cell (the autograd Function): E_c to [C] and the corner gradients to [C,8,3] (no atomics);
backward scatters them to [N,3] with torch index_add_, which measured faster than a separate atomic Warp scatter
kernel on an L40 (0.048 ms against 0.077 ms at 4000 cells). Fused (inference, `energy_and_grad_warp`): E_c is
atomically added to the owning object's slot of [O] and the 8 corner gradients are atomically added to [N,3],
then `corner_kernel` adds the inertia energy 0.5 rho m |x - Y|^2 per object and its gradient rho m (x - Y) per
corner and zeroes the pinned rows; with the contact pairs scattered by `contact_kernel` in between, the whole of
`physics.energy_and_grad` is one zero fill and three launches instead of some forty kernels (the per-object segment
sums, the inertia chain, the contact chain and its autograd backward). The torch path in physics.py is the
reference; this module only handles float32 CUDA.
"""

from __future__ import annotations

import torch
import warp as wp

from .contact import R_SAMPLE
from .contact_kernel import launch_contact

WEIGHT = 1.0 / 8.0  # Gauss weight, cell volume 1

mat83 = wp.types.matrix(shape=(8, 3), dtype=wp.float32)


@wp.func
def cofactor(F: wp.mat33) -> wp.mat33:
    """cof(F) = det(F) F^-T, defined at singular F: column j is the cross product of the other two columns."""
    c0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    c2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    return wp.matrix_from_cols(wp.cross(c1, c2), wp.cross(c2, c0), wp.cross(c0, c1))


@wp.kernel
def elastic_damping_kernel(
    x: wp.array[wp.vec3],
    cells: wp.array2d[wp.int64],
    cell_obj: wp.array[wp.int64],  # [C]
    lam: wp.array[float],  # [O]
    eta: wp.array[float],  # [O]
    mu_scale: wp.array[float],  # [O]
    C_prev: wp.array2d[wp.mat33],
    Gq: wp.array2d[wp.vec3],
    inv_factor: float,  # lambda at an inverted point = inv_factor x max(1, lam) (physics.effective_lambda; 0 = off)
    fused: int,  # 0: energy[c], grad_cells[c, k]; 1: energy_obj[cell_obj[c]] += E, grad_x[cells[c, k]] += g_k
    energy: wp.array[float],  # [C] (fused 0)
    grad_cells: wp.array2d[wp.vec3],  # [C,8] (fused 0)
    energy_obj: wp.array[float],  # [O] (fused 1)
    grad_x: wp.array[wp.vec3],  # [N] (fused 1)
):
    c = wp.tid()
    X = mat83()
    for k in range(8):
        xk = x[int(cells[c, k])]
        X[k, 0] = xk[0]
        X[k, 1] = xk[1]
        X[k, 2] = xk[2]
    o = int(cell_obj[c])
    lam_o = lam[o]
    eta_c = eta[o]
    wq = WEIGHT * mu_scale[o]  # Gauss weight times the object's unit factor
    E = float(0.0)
    G = mat83()
    for q in range(8):
        F = wp.mat33(0.0)
        for k in range(8):
            F += wp.outer(wp.vec3(X[k, 0], X[k, 1], X[k, 2]), Gq[q, k])
        J = wp.determinant(F)
        lam_q = lam_o
        if inv_factor > 0.0 and J <= 0.0:  # inverted point: the stiffened lambda (physics.effective_lambda)
            lam_q = inv_factor * wp.max(1.0, lam_o)
        lam_nh = lam_q + 1.0
        Jm1 = J - 1.0
        psi = 0.5 * (wp.ddot(F, F) - 3.0) + 0.5 * lam_nh * Jm1 * Jm1 - Jm1
        D = wp.transpose(F) * F - C_prev[c, q]
        psi_d = 0.5 * eta_c * wp.ddot(D, D)
        E += wq * (psi + psi_d)
        P = F + (lam_nh * Jm1 - 1.0) * cofactor(F) + (2.0 * eta_c) * (F * D)  # first Piola stress plus damping
        for k in range(8):
            g = wq * (P * Gq[q, k])
            G[k, 0] = G[k, 0] + g[0]
            G[k, 1] = G[k, 1] + g[1]
            G[k, 2] = G[k, 2] + g[2]
    if fused == 0:
        energy[c] = E
        for k in range(8):
            grad_cells[c, k] = wp.vec3(G[k, 0], G[k, 1], G[k, 2])
    else:
        wp.atomic_add(energy_obj, o, E)
        for k in range(8):
            wp.atomic_add(grad_x, int(cells[c, k]), wp.vec3(G[k, 0], G[k, 1], G[k, 2]))


@wp.kernel
def corner_kernel(
    x: wp.array[wp.vec3],
    Y: wp.array[wp.vec3],
    corner_obj: wp.array[wp.int64],  # [N]
    rho: wp.array[float],  # [O] the body's own group
    mu_scale: wp.array[float],  # [O] its factor into the object's unit of energy
    mass: wp.array[float],  # [N]
    pinned: wp.array[wp.bool],  # [N]
    energy_obj: wp.array[float],  # [O], accumulated
    grad_x: wp.array[wp.vec3],  # [N]: in, the elastic + contact scatter; out, plus inertia, pinned rows zero
):
    """Inertia 0.5 rho m |x - Y|^2 per corner into the object's energy (rho = mu_scale times the body's group), its
    gradient rho m (x - Y) onto the corner's gradient, pinned corners contribute nothing and end with a zero gradient."""
    n = wp.tid()
    if pinned[n]:
        grad_x[n] = wp.vec3(0.0)
        return
    o = int(corner_obj[n])
    dx = x[n] - Y[n]
    k = (mu_scale[o] * rho[o]) * mass[n]
    wp.atomic_add(energy_obj, o, 0.5 * k * wp.dot(dx, dx))
    grad_x[n] = grad_x[n] + k * dx


def _array(t: torch.Tensor | None, dtype):
    """Warp array descriptor aliasing t without a copy; requires_grad=False keeps Warp from allocating and
    attaching a .grad to the candidate, whose gradient is handled by the Function's backward."""
    return None if t is None else wp.from_torch(t.contiguous(), dtype=dtype, requires_grad=False, return_ctype=True)


def _stream(x: torch.Tensor):
    return wp.stream_from_torch(torch.cuda.current_stream(x.device))


def _launch_cells(batch, x, energy, grad_cells, energy_obj, grad_x) -> None:
    C = batch.cells.shape[0]
    if C == 0:
        return
    wp.init()
    from . import physics  # the inversion factor lives next to the torch reference (physics imports this module)

    m = batch.material
    inputs = [
        _array(x, wp.vec3),
        _array(batch.cells, wp.int64),
        _array(batch.cell_obj, wp.int64),
        _array(m.lam, wp.float32),
        _array(m.eta, wp.float32),
        _array(m.mu_scale, wp.float32),
        _array(batch.C_prev, wp.mat33),
        _array(batch.hc.Gq, wp.vec3),
        float(physics.INVERSION_LAMBDA_FACTOR),
        int(energy is None),
        _array(energy, wp.float32),
        _array(grad_cells, wp.vec3),
        _array(energy_obj, wp.float32),
        _array(grad_x, wp.vec3),
    ]
    wp.launch(elastic_damping_kernel, dim=C, inputs=inputs, stream=_stream(x))


def _launch_corners(batch, x, energy_obj, grad_x) -> None:
    N = x.shape[0]
    if N == 0:
        return
    m = batch.material
    inputs = [
        _array(x, wp.vec3),
        _array(batch.Y, wp.vec3),
        _array(batch.corner_obj, wp.int64),
        _array(m.rho, wp.float32),
        _array(m.mu_scale, wp.float32),
        _array(batch.mass, wp.float32),
        _array(batch.pinned, wp.bool),
        _array(energy_obj, wp.float32),
        _array(grad_x, wp.vec3),
    ]
    wp.launch(corner_kernel, dim=N, inputs=inputs, stream=_stream(x))


class ElasticDamping(torch.autograd.Function):
    """energies [C] = elastic + damping energy per cell; backward returns dE/dx [N,3] (x only: the material and
    C_prev are step constants). `batch` supplies cells, cell_obj, material, C_prev and the hex constants."""

    @staticmethod
    def forward(ctx, x, batch):
        x = x.contiguous()
        C = batch.cells.shape[0]
        energy = torch.empty(C, dtype=x.dtype, device=x.device)
        grad_cells = torch.empty(C, 8, 3, dtype=x.dtype, device=x.device)
        _launch_cells(batch, x, energy, grad_cells, None, None)
        ctx.save_for_backward(batch.cells, grad_cells)
        ctx.N = x.shape[0]
        return energy

    @staticmethod
    def backward(ctx, grad_out):
        cells, grad_cells = ctx.saved_tensors
        grad_x = torch.zeros(ctx.N, 3, dtype=grad_cells.dtype, device=grad_cells.device)
        grad_x.index_add_(0, cells.reshape(-1), (grad_out[:, None, None] * grad_cells).reshape(-1, 3))
        return grad_x, None


def elastic_damping_warp(batch, x: torch.Tensor) -> torch.Tensor:
    """Per-cell elastic + damping energy [C] at the corners x (float32 CUDA), differentiable in x."""
    return ElasticDamping.apply(x, batch)


def energy_and_grad_warp(batch, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused inference pass: E [O] = elastic + damping + inertia + contact per object and the detached gradient
    [N,3] with pinned rows zeroed, as `physics.energy_and_grad` returns them (float32 CUDA, no graph on x; the
    contact pairs in the capacity layout or absent). One zero fill and three launches."""
    x = x.detach().contiguous()
    N, O = x.shape[0], batch.O
    buf = torch.zeros(3 * N + O, dtype=x.dtype, device=x.device)  # one fill for both accumulators
    grad_x, energy = buf[: 3 * N].view(N, 3), buf[3 * N :]
    _launch_cells(batch, x, None, None, energy, grad_x)
    pairs = batch.pairs
    if pairs is not None and pairs.count > 0:
        launch_contact(batch, x, pairs, R_SAMPLE, energy, None, grad_x)
    _launch_corners(batch, x, energy, grad_x)
    return energy, grad_x
