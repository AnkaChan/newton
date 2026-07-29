# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: is the custom ``polar_R`` backward a correct gradient?

``torch_solver.polar_R`` computes the forward polar rotation with a detached
SVD and then re-routes the gradient through ``R = M @ inv(sym(R0^T M))``.
This script compares its Jacobian against a central finite-difference
Jacobian of the true polar rotation, at several distances from ``S = I``.
"""

from __future__ import annotations

import torch

from .torch_solver import polar_R


def true_polar_R(M: torch.Tensor) -> torch.Tensor:
    """Reference polar rotation (no gradient tricks)."""
    U, _s, Vh = torch.linalg.svd(M)
    det = torch.linalg.det(U @ Vh)
    D = torch.eye(3, dtype=M.dtype, device=M.device).expand(M.shape[0], -1, -1).clone()
    D[:, 2, 2] = det
    return U @ D @ Vh


def jac_autograd(fn, M):
    """d vec(R) / d vec(M) via autograd, shape (9, 9)."""
    rows = []
    for i in range(3):
        for j in range(3):
            Mv = M.clone().requires_grad_(True)
            R = fn(Mv)
            g = torch.autograd.grad(R[0, i, j], Mv, retain_graph=False)[0]
            rows.append(g[0].reshape(-1))
    return torch.stack(rows)


def jac_fd(M, eps=1e-6):
    rows = []
    for a in range(3):
        for b in range(3):
            Mp = M.clone()
            Mp[0, a, b] += eps
            Mm = M.clone()
            Mm[0, a, b] -= eps
            d = (true_polar_R(Mp) - true_polar_R(Mm)) / (2 * eps)
            rows.append(d[0].reshape(-1))
    # rows indexed by (a,b) = input; we want (out, in) -> transpose
    return torch.stack(rows).T


def main():
    torch.manual_seed(0)
    dt = torch.float64
    print(f"{'stretch dev':>12} {'|J_auto|':>12} {'|J_fd|':>12} {'rel err':>12} {'cos sim':>10}")
    for dev in (0.0, 0.01, 0.05, 0.2, 0.5):
        # M = R * S with S = I + dev * sym random
        A = torch.randn(3, 3, dtype=dt)
        Q, _ = torch.linalg.qr(A)
        if torch.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        B = torch.randn(3, 3, dtype=dt)
        Ssym = 0.5 * (B + B.T)
        Ssym = Ssym / Ssym.norm()
        S = torch.eye(3, dtype=dt) + dev * Ssym
        M = (Q @ S).unsqueeze(0)

        J_auto = jac_autograd(polar_R, M)
        J_ref = jac_fd(M)
        rel = (J_auto - J_ref).norm() / J_ref.norm().clamp(min=1e-30)
        cos = (J_auto.reshape(-1) @ J_ref.reshape(-1)) / (
            J_auto.norm().clamp(min=1e-30) * J_ref.norm().clamp(min=1e-30)
        )
        print(f"{dev:12.3f} {J_auto.norm():12.4e} {J_ref.norm():12.4e} {rel:12.4e} {cos:10.4f}")

    # Also: what happens without the symmetrisation (the docstring's stated intent)?
    print("\nWithout the 0.5*(S+S^T) symmetrisation the gradient collapses to zero:")

    def polar_R_nosym(M):
        with torch.no_grad():
            U, _s, Vh = torch.linalg.svd(M)
            det = torch.linalg.det(U @ Vh)
            D = torch.eye(3, dtype=M.dtype, device=M.device).expand(M.shape[0], -1, -1).clone()
            D[:, 2, 2] = det
            R0 = U @ D @ Vh
        S = R0.transpose(-1, -2) @ M
        return M @ torch.linalg.inv(S)

    A = torch.randn(3, 3, dtype=dt)
    Q, _ = torch.linalg.qr(A)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    S = torch.eye(3, dtype=dt) + 0.2 * torch.diag(torch.tensor([0.3, -0.1, 0.05], dtype=dt))
    M = (Q @ S).unsqueeze(0)
    print(f"  |J| (no sym)   = {jac_autograd(polar_R_nosym, M).norm():.4e}")
    print(f"  |J| (with sym) = {jac_autograd(polar_R, M).norm():.4e}")
    print(f"  |J| (true)     = {jac_fd(M).norm():.4e}")


if __name__ == "__main__":
    main()
