# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Authenticated source-operator geometry tests for iterative v5."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from .. import torch_solver as ts


def _pose(rest: np.ndarray, tets: np.ndarray) -> np.ndarray:
    origin = rest[tets[:, 0]]
    matrix = np.stack(
        (
            rest[tets[:, 1]] - origin,
            rest[tets[:, 2]] - origin,
            rest[tets[:, 3]] - origin,
        ),
        axis=-1,
    )
    return np.linalg.inv(matrix).astype(rest.dtype, copy=False)


def _structured_zero_inverse() -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(
        [
            [
                [float.fromhex("0x0.0p+0"), float.fromhex("0x0.0p+0"), -float.fromhex("0x1.279a74p-4")],
                [-float.fromhex("0x1.d8f720p-5"), -float.fromhex("0x1.d8f720p-5"), -float.fromhex("0x1.d8f720p-6")],
                [-float.fromhex("0x1.d8f71cp-5"), float.fromhex("0x0.0p+0"), -float.fromhex("0x1.d8f718p-6")],
            ]
        ],
        dtype=np.float32,
    )
    inverse = np.asarray(
        [
            [
                [float.fromhex("0x1.bb67acp+2"), float.fromhex("0x1.754da2p-50"), -float.fromhex("0x1.1520d0p+4")],
                [float.fromhex("0x1.e00006p-21"), -float.fromhex("0x1.1520cep+4"), float.fromhex("0x1.1520d0p+4")],
                [-float.fromhex("0x1.bb67b0p+3"), -float.fromhex("0x0.0p+0"), -float.fromhex("0x0.0p+0")],
            ]
        ],
        dtype=np.float32,
    )
    return source, inverse


class TestAuthenticatedOperatorGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.rest32 = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.125, 0.125, 0.0],
                [0.0, 0.875, 0.0625],
                [0.03125, 0.0, 1.25],
            ],
            dtype=np.float32,
        )
        self.tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        self.pinned = np.asarray([0, 1, 2], dtype=np.int64)

    def _build_promoted(self, poses: np.ndarray | None = None) -> ts.SolverState:
        return ts.build_solver(
            self.rest32,
            self.tets,
            _pose(self.rest32, self.tets) if poses is None else poses,
            self.pinned,
            torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
        )

    def test_float32_source_is_preserved_and_exactly_promoted_to_float64(self) -> None:
        poses = _pose(self.rest32, self.tets)
        state = self._build_promoted(poses)

        self.assertEqual(state.source_rest_q_exact.dtype, torch.float32)
        self.assertEqual(state.source_tet_poses.dtype, torch.float32)
        self.assertEqual(state.source_tet_indices.dtype, torch.int64)
        self.assertTrue(torch.equal(state.source_rest_q_exact, torch.from_numpy(self.rest32)))
        self.assertTrue(torch.equal(state.source_tet_poses, torch.from_numpy(poses)))
        self.assertTrue(torch.equal(state.rest_q, torch.from_numpy(self.rest32).to(torch.float64)))
        self.assertTrue(torch.equal(state.Dm_inv, torch.from_numpy(poses).to(torch.float64)))
        self.assertEqual(ts.validate_authenticated_operator_geometry(state), state.operator_geometry_sha256)
        self.assertEqual(state.static_mesh_sha256, ts.static_mesh_sha256(self.rest32, self.tets))

    def test_source_bytes_policy_and_runtime_projection_are_separately_bound(self) -> None:
        rest32 = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        pose32 = _pose(rest32, self.tets)
        promoted = ts.build_solver(
            rest32,
            self.tets,
            pose32,
            self.pinned,
            torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
        )
        rest64 = rest32.astype(np.float64)
        pose64 = _pose(rest64, self.tets)
        canonical = ts.build_solver(
            rest64,
            self.tets,
            pose64,
            self.pinned,
            torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )

        self.assertEqual(promoted.static_mesh_sha256, canonical.static_mesh_sha256)
        for name in ("rest_q", "Dm_inv", "J", "w", "L", "L_ff_chol", "L_fp"):
            self.assertTrue(torch.equal(getattr(promoted, name), getattr(canonical, name)), name)
        self.assertNotEqual(promoted.operator_geometry_sha256, canonical.operator_geometry_sha256)
        self.assertNotEqual(promoted.projection_state_sha256, canonical.projection_state_sha256)
        with self.assertRaisesRegex(ValueError, "requires exact source rest_q and tet_poses dtype"):
            ts.build_solver(
                self.rest32,
                self.tets,
                _pose(self.rest32, self.tets),
                self.pinned,
                torch.device("cpu"),
                dtype=torch.float64,
                operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
            )

    def test_source_mutation_and_policy_relabel_fail_closed(self) -> None:
        state = self._build_promoted()
        state.source_tet_poses[0, 0, 0] = torch.nextafter(
            state.source_tet_poses[0, 0, 0],
            torch.tensor(float("inf"), dtype=torch.float32),
        )
        with self.assertRaisesRegex(ValueError, "operator_geometry_sha256 verification failed"):
            ts.validate_authenticated_operator_geometry(state)

        state = self._build_promoted()
        state.operator_geometry_policy = ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE
        with self.assertRaisesRegex(ValueError, "requires exact source rest_q and tet_poses dtype"):
            ts.validate_authenticated_operator_geometry(state)

    def test_same_topology_can_have_distinct_authenticated_source_operators(self) -> None:
        poses = _pose(self.rest32, self.tets)
        perturbed = poses.copy()
        perturbed[0, 0, 0] = np.nextafter(perturbed[0, 0, 0], np.float32(np.inf))
        first = self._build_promoted(poses)
        second = self._build_promoted(perturbed)

        self.assertEqual(first.static_mesh_sha256, second.static_mesh_sha256)
        self.assertNotEqual(first.operator_geometry_sha256, second.operator_geometry_sha256)
        self.assertFalse(torch.equal(first.Dm_inv, second.Dm_inv))

    def test_structural_zero_roundoff_authenticates(self) -> None:
        source, inverse = _structured_zero_inverse()

        ts._require_inverse_backward_error_numpy(source, inverse)

    def test_structural_zero_roundoff_does_not_hide_pose_tampering(self) -> None:
        source, inverse = _structured_zero_inverse()
        inverse[0, 0, 1] += np.float32(1.0e-8)

        with self.assertRaisesRegex(ValueError, "backward-error bound"):
            ts._require_inverse_backward_error_numpy(source, inverse)

    def test_nonfinite_contribution_scale_cannot_authenticate(self) -> None:
        huge = np.finfo(np.float32).max
        source = np.asarray([[[huge, huge, 0.0], [huge, -huge, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
        inverse = np.asarray([[[huge, huge, 0.0], [huge, -huge, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(ValueError, "backward-error bound"):
                ts._require_inverse_backward_error_numpy(source, inverse)

    def test_legacy_path_remains_unverified_and_keeps_v2_projection_digest(self) -> None:
        rest = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        state = ts.build_solver(
            rest,
            self.tets,
            _pose(rest, self.tets),
            self.pinned,
            torch.device("cpu"),
            dtype=torch.float64,
        )

        self.assertEqual(state.operator_geometry_policy, "legacy-unverified")
        self.assertIsNone(state.operator_geometry_sha256)
        self.assertEqual(
            state.projection_state_sha256,
            "f2039579d0b2d83351631f5060c067a5fd781c1f0b4a3213fceede68751a12db",
        )
        with self.assertRaisesRegex(ValueError, "no authenticated v5 operator geometry"):
            ts.validate_authenticated_operator_geometry(state)

    def test_legacy_promoted_float32_values_still_build_but_cannot_authenticate_as_float64(self) -> None:
        poses32 = _pose(self.rest32, self.tets)
        promoted_rest64 = self.rest32.astype(np.float64)
        promoted_poses64 = poses32.astype(np.float64)
        legacy = ts.build_solver(
            promoted_rest64,
            self.tets,
            promoted_poses64,
            self.pinned,
            torch.device("cpu"),
            dtype=torch.float64,
        )

        self.assertEqual(legacy.operator_geometry_policy, "legacy-unverified")
        self.assertEqual(
            legacy.projection_state_sha256,
            "01447f835397889b64dd72635275d0f6b0a7570862d68457ea42343a7ed721b9",
        )
        with self.assertRaisesRegex(ValueError, "backward-error bound"):
            ts.build_solver(
                promoted_rest64,
                self.tets,
                promoted_poses64,
                self.pinned,
                torch.device("cpu"),
                dtype=torch.float64,
                operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
            )

    def test_legacy_keeps_numpy_pose_alias_while_authenticated_sources_are_owned(self) -> None:
        rest64 = self.rest32.astype(np.float64)
        legacy_poses = _pose(rest64, self.tets)
        legacy = ts.build_solver(
            rest64,
            self.tets,
            legacy_poses,
            self.pinned,
            torch.device("cpu"),
            dtype=torch.float64,
        )
        legacy_poses[0, 0, 0] += 0.25
        self.assertEqual(float(legacy.Dm_inv[0, 0, 0]), legacy_poses[0, 0, 0])

        authenticated_poses = _pose(rest64, self.tets)
        authenticated = ts.build_solver(
            rest64,
            self.tets,
            authenticated_poses,
            self.pinned,
            torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        expected = authenticated.Dm_inv.clone()
        authenticated_poses[0, 0, 0] += 0.25
        self.assertTrue(torch.equal(authenticated.Dm_inv, expected))
        self.assertTrue(torch.equal(authenticated.source_tet_poses, expected))


if __name__ == "__main__":
    unittest.main()
