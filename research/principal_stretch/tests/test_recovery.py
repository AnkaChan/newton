# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for principal-stretch inverse recovery."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

from ..kernels import compute_F_polar
from ..recover_local_global import LocalGlobalRecover


def _tiny_grid(dim=(3, 2, 2), cell=0.1):
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim[0],
        dim_y=dim[1],
        dim_z=dim[2],
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        density=1.0e3,
        k_mu=1.0e5,
        k_lambda=1.0e5,
        k_damp=1e-3,
        fix_left=True,
    )
    builder.color()
    model = builder.finalize()
    pinned = np.where(np.asarray(builder.particle_mass, dtype=np.float64) == 0.0)[0]
    rest_q = np.asarray(builder.particle_q, dtype=np.float32)
    tet_indices = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)
    tet_poses = model.tet_poses.numpy().reshape(-1, 3, 3).astype(np.float32)
    return rest_q, tet_indices, tet_poses, pinned, model


def _stretches_from_positions(x, tet_indices, tet_poses, model):
    n_tets = tet_indices.shape[0]
    device = model.device
    x_wp = wp.array(x, dtype=wp.vec3, device=device)
    ti = wp.array(tet_indices, dtype=wp.int32, device=device)
    tp = wp.array(tet_poses.reshape(-1, 3, 3), dtype=wp.mat33, device=device)
    F = wp.zeros(n_tets, dtype=wp.mat33, device=device)
    R = wp.zeros(n_tets, dtype=wp.mat33, device=device)
    S = wp.zeros(n_tets, dtype=wp.mat33, device=device)
    wp.launch(compute_F_polar, dim=n_tets, inputs=[x_wp, ti, tp], outputs=[F, R, S], device=device)
    return F.numpy().reshape(-1, 3, 3), R.numpy().reshape(-1, 3, 3), S.numpy().reshape(-1, 3, 3)


class TestPrincipalStretchRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.rest_q, cls.tet_indices, cls.tet_poses, cls.pinned, cls.model = _tiny_grid()
        cls.device = "cuda:0" if wp.get_device().is_cuda else "cpu"
        cls.rec = LocalGlobalRecover(cls.rest_q, cls.tet_indices, cls.tet_poses, cls.pinned, device=cls.device)

    def test_round_trip_identity(self):
        """S_target = S(rest) ⇒ recovered shape is the rest pose."""
        _, _, S = _stretches_from_positions(self.rest_q, self.tet_indices, self.tet_poses, self.model)
        # At rest F = I exactly, so S = I and R = I.
        np.testing.assert_allclose(S, np.broadcast_to(np.eye(3), S.shape), atol=1e-5)

        res = self.rec.solve(
            S_target=S,
            pinned_targets=self.rest_q[self.pinned],
            x_init=self.rest_q,
            max_iters=10,
            tol=1e-12,
        )
        err = np.linalg.norm(res.x - self.rest_q, axis=1).mean()
        self.assertLess(err, 1e-5)

    def test_uniform_stretch(self):
        """S_target = 1.1 * I everywhere ⇒ recovered = 1.1 * rest (anchored at rest)."""
        # NB: with pinned vertices held at rest, the recovered shape can't be a
        # uniform scaling of rest; instead the solver finds the closest non-uniform
        # shape that pins the BC. So we test the *unpinned* (Tikhonov) case here.
        n_tets = self.tet_indices.shape[0]
        S_target = np.broadcast_to(1.1 * np.eye(3), (n_tets, 3, 3)).copy()
        rec_free = LocalGlobalRecover(
            self.rest_q,
            self.tet_indices,
            self.tet_poses,
            pinned_indices=np.array([], dtype=np.int64),
            device=self.device,
            tikhonov=1e-8,
        )
        res = rec_free.solve(
            S_target=S_target,
            pinned_targets=np.zeros((0, 3), dtype=np.float32),
            x_init=self.rest_q,
            max_iters=200,
            tol=1e-12,
        )
        _, _, S_recovered = _stretches_from_positions(res.x, self.tet_indices, self.tet_poses, self.model)
        err = np.linalg.norm(S_recovered - S_target, axis=(1, 2)).mean()
        self.assertLess(err, 1e-3)

    def test_recovery_from_synthetic_deformation(self):
        """Apply a known deformation, extract stretches, recover, compare."""
        # Synthetic: shear + stretch along x.
        A = np.array(
            [
                [1.3, 0.2, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        x_target = (self.rest_q @ A.T).astype(np.float32)
        _, _, S_target = _stretches_from_positions(x_target, self.tet_indices, self.tet_poses, self.model)

        res = self.rec.solve(
            S_target=S_target,
            pinned_targets=x_target[self.pinned],
            x_init=self.rest_q,
            max_iters=200,
            tol=1e-12,
        )
        err = np.linalg.norm(res.x - x_target, axis=1).mean()
        self.assertLess(err, 1e-4)

    def test_se3_ambiguity(self):
        """Stretches are SE(3)-invariant; recovery without anchors works up to SE(3)."""
        # Reuse the synthetic deformation, but apply an extra random rigid transform.
        rng = np.random.default_rng(seed=0)
        A_def = np.array([[1.2, 0.1, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        x_def = (self.rest_q @ A_def.T).astype(np.float32)

        # Apply a known rigid Q, t.
        ax = rng.standard_normal(3)
        ax = ax / np.linalg.norm(ax)
        theta = 0.7
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Q = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        t = np.array([1.5, -0.3, 2.1])
        x_target_rigid = (x_def @ Q.T + t).astype(np.float32)

        _, _, S_target = _stretches_from_positions(x_target_rigid, self.tet_indices, self.tet_poses, self.model)
        _, _, S_def = _stretches_from_positions(x_def, self.tet_indices, self.tet_poses, self.model)
        # Stretches must be invariant to rigid transform.
        np.testing.assert_allclose(S_target, S_def, atol=1e-4)

        # Recover with NO pins, Tikhonov regularised.
        rec_free = LocalGlobalRecover(
            self.rest_q,
            self.tet_indices,
            self.tet_poses,
            pinned_indices=np.array([], dtype=np.int64),
            device=self.device,
            tikhonov=1e-8,
        )
        res = rec_free.solve(
            S_target=S_target,
            pinned_targets=np.zeros((0, 3), dtype=np.float32),
            x_init=self.rest_q,
            max_iters=200,
            tol=1e-12,
        )
        # Raw vertex error should be huge (different SE(3) frame).
        raw = np.linalg.norm(res.x - x_target_rigid, axis=1).mean()
        # Procrustes-aligned should be small.
        a = res.x - res.x.mean(0)
        b = x_target_rigid - x_target_rigid.mean(0)
        H = a.T @ b
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        aligned = a @ R.T + x_target_rigid.mean(0)
        aligned_err = np.linalg.norm(aligned - x_target_rigid, axis=1).mean()

        self.assertGreater(raw, 0.5, f"raw err unexpectedly small: {raw}")
        self.assertLess(aligned_err, 1e-2, f"aligned err too large: {aligned_err}")


if __name__ == "__main__":
    unittest.main()
