# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local-global ARAP-with-target-stretch recovery solver (reference solver A).

Energy per tet: E_e = (w_e/2) || F_e(x) - R_e * S_e^* ||_F^2.

Local step (parallel, closed-form):
    R_e = polar_R(F_e * (S_e^*)^T)

Global step (single sparse linear solve):
    L @ x = rhs(R, S^*),   L is the rest-mesh ARAP Laplacian (constant across
    iterations AND across frames, since S_e^* only enters the RHS).

Dirichlet BCs at pinned vertices are eliminated from the system.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import warp as wp

from .kernels import assemble_global_rhs, compute_F_only, local_step_R


@dataclasses.dataclass
class RecoveryResult:
    x: np.ndarray
    iters: int
    stretch_err: float
    converged: bool
    history: list


class LocalGlobalRecover:
    """One-shot setup, then solve(...) per frame with new target stretches."""

    def __init__(
        self,
        rest_q: np.ndarray,  # (V, 3)
        tet_indices: np.ndarray,  # (T, 4) int32
        tet_poses: np.ndarray,  # (T, 3, 3) Dm_inv
        pinned_indices: np.ndarray,  # (P,) int
        device: str = "cuda:0",
        tikhonov: float = 0.0,
    ):
        self.device = device
        self.rest_q = rest_q.astype(np.float32)
        self.tet_indices = tet_indices.astype(np.int32)
        self.tet_poses = tet_poses.astype(np.float32)
        self.pinned = np.asarray(pinned_indices, dtype=np.int64)

        self.n_verts = rest_q.shape[0]
        self.n_tets = tet_indices.shape[0]

        # Per-tet rest volume as the ARAP weight w_e.
        det_inv = np.linalg.det(self.tet_poses)
        self.w = (1.0 / (6.0 * det_inv)).astype(np.float32)
        if (self.w <= 0).any():
            raise ValueError("non-positive rest volumes — check tet orientation")

        # Per-tet Jacobian rows J_e in R^{4x3}: row a = dF_col_c / dx_a for c=0..2.
        J = np.zeros((self.n_tets, 4, 3), dtype=np.float64)
        J[:, 1] = self.tet_poses[:, 0]
        J[:, 2] = self.tet_poses[:, 1]
        J[:, 3] = self.tet_poses[:, 2]
        J[:, 0] = -(J[:, 1] + J[:, 2] + J[:, 3])
        self.J = J
        self.tikhonov = float(tikhonov)

        self._assemble_L()
        self._partition_BC()
        self._factorize()
        self._init_gpu_buffers()

    # ---- one-time setup ----------------------------------------------------

    def _assemble_L(self):
        import scipy.sparse as sp

        K = self.w.astype(np.float64)[:, None, None] * np.einsum("eac,ebc->eab", self.J, self.J)
        rows = np.repeat(self.tet_indices, 4, axis=1)  # (T, 16)
        cols = np.tile(self.tet_indices, 4)  # (T, 16)
        self.L = sp.csr_matrix(
            (K.reshape(-1), (rows.reshape(-1), cols.reshape(-1))),
            shape=(self.n_verts, self.n_verts),
        )

    def _partition_BC(self):
        import scipy.sparse as sp

        mask = np.ones(self.n_verts, dtype=bool)
        mask[self.pinned] = False
        self.free = np.where(mask)[0]
        L_ff = self.L[self.free][:, self.free]
        if self.tikhonov > 0.0:
            L_ff = L_ff + self.tikhonov * sp.eye(self.free.size, format="csr")
        self.L_ff = L_ff.tocsc()
        self.L_fp = self.L[self.free][:, self.pinned].tocsc()

    def _factorize(self):
        import scipy.sparse.linalg as spla

        if self.free.size == 0:
            self._solve = None
            return
        self._solve = spla.factorized(self.L_ff)

    def _init_gpu_buffers(self):
        self.x_wp = wp.zeros(self.n_verts, dtype=wp.vec3, device=self.device)
        self.F_wp = wp.zeros(self.n_tets, dtype=wp.mat33, device=self.device)
        self.R_wp = wp.zeros(self.n_tets, dtype=wp.mat33, device=self.device)
        self.S_wp = wp.zeros(self.n_tets, dtype=wp.mat33, device=self.device)
        self.rhs_wp = wp.zeros(self.n_verts, dtype=wp.vec3, device=self.device)
        self.tet_idx_wp = wp.array(self.tet_indices, dtype=wp.int32, device=self.device)
        self.tet_pose_wp = wp.array(self.tet_poses.reshape(-1, 3, 3), dtype=wp.mat33, device=self.device)
        self.w_wp = wp.array(self.w, dtype=wp.float32, device=self.device)

    # ---- per-frame solve ---------------------------------------------------

    def solve(
        self,
        S_target: np.ndarray,  # (T, 3, 3)
        pinned_targets: np.ndarray,  # (P, 3) — world-frame target positions
        x_init: np.ndarray | None = None,  # (V, 3)
        max_iters: int = 200,
        tol: float = 1e-8,
    ) -> RecoveryResult:
        if x_init is None:
            x = self.rest_q.copy()
        else:
            x = x_init.astype(np.float32).copy()
        x[self.pinned] = pinned_targets.astype(np.float32)

        S_wp = wp.array(S_target.reshape(-1, 3, 3), dtype=wp.mat33, device=self.device)
        # Boundary contribution to RHS (only depends on pinned target positions).
        bc_rhs = self.L_fp @ pinned_targets  # (|free|, 3)

        history = []
        last_err = np.inf
        converged = False
        iters = max_iters

        for it in range(max_iters):
            self.x_wp.assign(x)

            wp.launch(
                compute_F_only,
                dim=self.n_tets,
                inputs=[self.x_wp, self.tet_idx_wp, self.tet_pose_wp],
                outputs=[self.F_wp],
                device=self.device,
            )
            wp.launch(
                local_step_R,
                dim=self.n_tets,
                inputs=[self.F_wp, S_wp],
                outputs=[self.R_wp],
                device=self.device,
            )
            self.rhs_wp.zero_()
            wp.launch(
                assemble_global_rhs,
                dim=self.n_tets,
                inputs=[self.R_wp, S_wp, self.tet_idx_wp, self.tet_pose_wp, self.w_wp],
                outputs=[self.rhs_wp],
                device=self.device,
            )
            rhs = self.rhs_wp.numpy().astype(np.float64)  # (V, 3)

            b_free = rhs[self.free] - bc_rhs
            x_free = np.column_stack([self._solve(b_free[:, a]) for a in range(3)])
            x_new = x.copy()
            x_new[self.free] = x_free.astype(np.float32)
            x_new[self.pinned] = pinned_targets.astype(np.float32)

            # Measure stretch error using current F vs S_target.
            F_np = self.F_wp.numpy().reshape(-1, 3, 3)
            S_now = _polar_S_batch(F_np)
            stretch_err = float(np.linalg.norm(S_now - S_target, axis=(1, 2)).mean())

            step_norm = float(np.linalg.norm(x_new - x))
            history.append((it, stretch_err, step_norm))
            x = x_new

            if step_norm < tol:
                iters = it + 1
                converged = True
                break
            last_err = stretch_err

        return RecoveryResult(x=x, iters=iters, stretch_err=last_err, converged=converged, history=history)


def _polar_S_batch(F: np.ndarray) -> np.ndarray:
    """CPU polar-S for diagnostics: S = (F^T F)^{1/2}."""
    A = np.einsum("eij,eik->ejk", F, F)  # F^T F, symmetric PSD
    # Eigendecompose symmetric positive matrices.
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    sqrt_w = np.sqrt(w)
    return np.einsum("eij,ej,ekj->eik", V, sqrt_w, V)
