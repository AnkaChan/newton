# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit trilinear hex: corner order, Gauss points, shape-function gradients and the 21-mode maps.

Everything is in cell units (h = 1), so these tensors are constants shared by every grid.
Local corner k = 4 i + 2 j + l for the corner offset (i, j, l), z fastest; xi_k = 2 (i, j, l) - 1.
Gauss point q uses the same ordering with xi_q = xi_k / sqrt(3).

Mode order (derivation note section 8): alpha^1, alpha^2, alpha^3, omega^12, omega^13, omega^23, omega^123,
each a 3-vector, scaled by 2/h so that the first three are the columns of the centre deformation gradient.
"""

from __future__ import annotations

import math
from typing import ClassVar

import torch

MODE_COUNT = 7
CORNER_OFFSETS = torch.tensor([(i, j, l) for i in (0, 1) for j in (0, 1) for l in (0, 1)], dtype=torch.int64)  # [8,3]
XI_CORNERS = (2 * CORNER_OFFSETS - 1).to(torch.float64)  # [8,3]
XI_GAUSS = XI_CORNERS / math.sqrt(3.0)  # [8,3]

# Faces: -x, +x, -y, +y, -z, +z. Corner order of each face is counter-clockwise seen from outside.
FACE_AXIS = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
FACE_SIGN = torch.tensor([-1, 1, -1, 1, -1, 1], dtype=torch.int64)
FACE_NORMALS = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=torch.float64)


def _face_corners() -> torch.Tensor:
    out = []
    for f in range(6):
        axis, sign = int(FACE_AXIS[f]), int(FACE_SIGN[f])
        ks = [k for k in range(8) if int(CORNER_OFFSETS[k, axis]) == (1 if sign > 0 else 0)]
        # order the four corners counter-clockwise around the outward normal
        others = [a for a in range(3) if a != axis]
        centre = CORNER_OFFSETS[ks].to(torch.float64).mean(0)
        n = FACE_NORMALS[f]
        u = torch.zeros(3, dtype=torch.float64)
        u[others[0]] = 1.0
        v = torch.linalg.cross(n, u)

        def angle(k, centre=centre, u=u, v=v):
            d = CORNER_OFFSETS[k].to(torch.float64) - centre
            return math.atan2(float(d @ v), float(d @ u))

        out.append(sorted(ks, key=angle))
    return torch.tensor(out, dtype=torch.int64)


FACE_CORNERS = _face_corners()  # [6,4] local corner ids


def shape_gradients(xi: torch.Tensor) -> torch.Tensor:
    """d N_k / d xi^a at the points xi [Q,3], times 2 (the 2/h factor with h = 1). Returns [Q,8,3]."""
    xi = xi.to(torch.float64)
    factors = 1.0 + XI_CORNERS[None, :, :] * xi[:, None, :]  # [Q,8,3]
    out = torch.empty(xi.shape[0], 8, 3, dtype=torch.float64)
    for a in range(3):
        others = [b for b in range(3) if b != a]
        out[:, :, a] = XI_CORNERS[None, :, a] * factors[:, :, others[0]] * factors[:, :, others[1]] / 8.0
    return 2.0 * out


GQ = shape_gradients(XI_GAUSS)  # [8,8,3]  F[r,a] = sum_k x_k[r] GQ[q,k,a]
WEIGHTS = torch.full((8,), 1.0 / 8.0, dtype=torch.float64)  # Gauss weights, cell volume 1


def _p_modes() -> torch.Tensor:
    xi = XI_CORNERS
    rows = [
        xi[:, 0],
        xi[:, 1],
        xi[:, 2],
        xi[:, 0] * xi[:, 1],
        xi[:, 0] * xi[:, 2],
        xi[:, 1] * xi[:, 2],
        xi[:, 0] * xi[:, 1] * xi[:, 2],
    ]
    return torch.stack(rows) * (2.0 / 8.0)  # 1/8 Hadamard projection times the 2/h scale


P_MODES = _p_modes()  # [7,8]  m[v] = sum_k P_MODES[v,k] x_k


def _mode_coefficients(xi: torch.Tensor) -> torch.Tensor:
    """coef[q, v, a]: column a of F(xi_q) = sum_v coef[q,v,a] m_v (derivation eq. 8.4)."""
    xi = xi.to(torch.float64)
    q = xi.shape[0]
    c = torch.zeros(q, MODE_COUNT, 3, dtype=torch.float64)
    for a in range(3):
        c[:, a, a] = 1.0
    pairs = {3: (0, 1), 4: (0, 2), 5: (1, 2)}
    for v, (a, b) in pairs.items():
        c[:, v, a] = xi[:, b]
        c[:, v, b] = xi[:, a]
    for a in range(3):
        others = [b for b in range(3) if b != a]
        c[:, 6, a] = xi[:, others[0]] * xi[:, others[1]]
    return c


def _gamma(xi: torch.Tensor) -> torch.Tensor:
    """Gamma_q [9,21] with vec F index 3 r + a (row-major) and mode index 3 v + r."""
    coef = _mode_coefficients(xi)  # [Q,7,3]
    q = xi.shape[0]
    g = torch.zeros(q, 9, 3 * MODE_COUNT, dtype=torch.float64)
    for r in range(3):
        for a in range(3):
            for v in range(MODE_COUNT):
                g[:, 3 * r + a, 3 * v + r] = coef[:, v, a]
    return g


GAMMA = _gamma(XI_GAUSS)  # [8,9,21]


class HexConstants:
    """The unit-hex tensors on one device and dtype (cached per device)."""

    _cache: ClassVar[dict] = {}

    def __init__(self, device, dtype=torch.float32):
        self.Gq = GQ.to(device=device, dtype=dtype)
        self.weights = WEIGHTS.to(device=device, dtype=dtype)
        self.P_modes = P_MODES.to(device=device, dtype=dtype)
        self.Gamma = GAMMA.to(device=device, dtype=dtype)
        self.face_corners = FACE_CORNERS.to(device)
        self.face_normals = FACE_NORMALS.to(device=device, dtype=dtype)

    @classmethod
    def get(cls, device, dtype=torch.float32) -> HexConstants:
        key = (str(device), dtype)
        if key not in cls._cache:
            cls._cache[key] = cls(torch.device(device), dtype)
        return cls._cache[key]


def mat3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched 3x3 product a @ b as an elementwise multiply-sum (cuBLAS batched tiny gemm is far slower)."""
    return (a[..., :, :, None] * b[..., None, :, :]).sum(-2)


def mat3_tn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched a^T @ b for [...,3,3] matrices, elementwise."""
    return (a[..., :, :, None] * b[..., :, None, :]).sum(-3)


def modes(x_cells: torch.Tensor, hc: HexConstants) -> torch.Tensor:
    """21 mode values per cell from the gathered corners x_cells [C,8,3] -> [C,7,3] (world vectors).

    matmul broadcasts P_modes over the cells as one batched GEMM with a contiguous result; the einsum form
    permuted and copied the corners first and returned a strided view."""
    return torch.matmul(hc.P_modes, x_cells)


def center_deformation(m: torch.Tensor) -> torch.Tensor:
    """Centre deformation gradient [C,3,3] from the modes: columns are the three axis vectors."""
    return m[:, :3, :].transpose(1, 2)


def gauss_deformation(x_cells: torch.Tensor, hc: HexConstants) -> torch.Tensor:
    """Deformation gradients at the 8 Gauss points, [C,8,3,3]."""
    return torch.einsum("ckr,qka->cqra", x_cells, hc.Gq)


def modes_to_gauss(dm: torch.Tensor, hc: HexConstants) -> torch.Tensor:
    """Per-Gauss-point deformation increments [C,8,3,3] from mode increments dm [C,7,3] (eq. 8.4)."""
    return torch.einsum("qij,cj->cqi", hc.Gamma, dm.reshape(dm.shape[0], 3 * MODE_COUNT)).reshape(-1, 8, 3, 3)


def gauss_to_modes(dF: torch.Tensor, hc: HexConstants) -> torch.Tensor:
    """Adjoint of modes_to_gauss: sum_q Gamma_q^T vec dF_q, [C,8,3,3] -> [C,7,3]."""
    return torch.einsum("qij,cqi->cj", hc.Gamma, dF.reshape(dF.shape[0], 8, 9)).reshape(-1, MODE_COUNT, 3)
