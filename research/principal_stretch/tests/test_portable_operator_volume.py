# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for portable host-authenticated tetrahedron volumes."""

from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from .. import torch_solver as ts
from ..iterative_solver import validate_projection_objective_volume_binding
from ..v5_objective import CommonObjectiveContext
from .test_v5_operator_geometry import _structured_zero_inverse


def _structural_zero_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_matrix, source_pose = _structured_zero_inverse()
    rest = np.zeros((4, 3), dtype=np.float32)
    rest[1:] = source_matrix[0].T
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    return rest, tets, source_pose


def _portable_state(device: torch.device) -> ts.SolverState:
    rest, tets, poses = _structural_zero_geometry()
    return ts.build_solver(
        rest,
        tets,
        poses,
        np.asarray([0, 1, 2], dtype=np.int64),
        device,
        dtype=torch.float64,
        operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
    )


def _bound_objective(state: ts.SolverState) -> CommonObjectiveContext:
    return CommonObjectiveContext(
        tets=state.tets,
        J=state.J,
        volume=state.w,
        mass=torch.tensor([0.8, 1.0, 1.2, 1.4], dtype=torch.float64, device=state.rest_q.device),
        mu=torch.tensor([17.0], dtype=torch.float64, device=state.rest_q.device),
        lam=torch.tensor([31.0], dtype=torch.float64, device=state.rest_q.device),
        inertial_target=state.rest_q + 0.015,
        pinned=state.pinned,
        dt=0.08,
        operator_geometry_sha256=state.operator_geometry_sha256,
        operator_volume_policy=state.operator_volume_policy,
        operator_volume_sha256=state.operator_volume_sha256,
    )


def _little_endian_bytes(value: torch.Tensor) -> bytes:
    return np.asarray(value.detach().contiguous().cpu().numpy(), dtype="<f8").tobytes(order="C")


class TestPortableOperatorVolume(unittest.TestCase):
    def test_scalar_order_preserves_structural_zeros_and_canonical_little_endian_bytes(self) -> None:
        """Preserve the registered scalar order and canonical byte representation."""
        rest, tets, poses = _structural_zero_geometry()

        payload = ts.canonical_operator_volume(
            rest,
            tets,
            poses,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
        )

        self.assertEqual(payload.policy, ts.OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT)
        self.assertEqual(payload.source_rest_determinant.dtype.str, "<f8")
        self.assertEqual(payload.source_tet_pose_determinant.dtype.str, "<f8")
        self.assertEqual(payload.rest_volume.dtype.str, "<f8")
        self.assertTrue(payload.source_rest_determinant.flags.c_contiguous)
        self.assertTrue(payload.source_tet_pose_determinant.flags.c_contiguous)
        self.assertTrue(payload.rest_volume.flags.c_contiguous)
        self.assertFalse(payload.source_rest_determinant.flags.writeable)
        self.assertFalse(payload.source_tet_pose_determinant.flags.writeable)
        self.assertFalse(payload.rest_volume.flags.writeable)
        self.assertEqual(payload.source_rest_determinant.tobytes().hex(), "5a88cfb9f0872f3f")
        self.assertEqual(payload.source_tet_pose_determinant.tobytes().hex(), "7f86b84bec3cb040")
        self.assertEqual(payload.rest_volume.tobytes().hex(), "b555600a4b05053f")
        self.assertEqual(payload.source_rest_determinant[0].item().hex(), "0x1.f87f0b9cf885ap-13")
        self.assertEqual(payload.source_tet_pose_determinant[0].item().hex(), "0x1.03cec4bb8867fp+12")
        self.assertEqual(payload.rest_volume[0].item().hex(), "0x1.5054b0a6055b5p-15")
        self.assertEqual(
            payload.operator_geometry_sha256,
            "c60676b376f2aaa98b71a4d3224429c247fdb5772895ced4b261bbc4fdaa53b3",
        )
        self.assertEqual(
            payload.operator_volume_sha256,
            "d7838623eceff3aa70dc5f91e5e552781cb4b3a6447742d80a28154be20bd68f",
        )
        big_endian_volume = payload.rest_volume.astype(">f8")
        self.assertEqual(
            ts.operator_volume_sha256(
                payload.operator_geometry_sha256,
                big_endian_volume,
                policy=payload.policy,
            ),
            payload.operator_volume_sha256,
        )

    def test_inverted_rest_order_and_negative_pose_orientation_fail_closed(self) -> None:
        """Reject either source-rest or source-pose orientation inversion."""
        rest, tets, poses = _structural_zero_geometry()
        inverted_tets = tets.copy()
        inverted_tets[:, [1, 2]] = inverted_tets[:, [2, 1]]
        with self.assertRaisesRegex(ValueError, "positive orientation"):
            ts.canonical_operator_volume(
                rest,
                inverted_tets,
                poses,
                operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
            )

        inverted_poses = poses.copy()
        inverted_poses[:, [0, 1]] = inverted_poses[:, [1, 0]]
        with self.assertRaisesRegex(ValueError, "positive orientation"):
            ts.canonical_operator_volume(
                rest,
                tets,
                inverted_poses,
                operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
            )

    def test_solver_copies_canonical_host_volume_bytes_and_binds_new_projection_domain(self) -> None:
        """Copy canonical host volume bytes into the portable solver state."""
        state = _portable_state(torch.device("cpu"))
        rest, tets, poses = _structural_zero_geometry()
        payload = ts.canonical_operator_volume(
            rest,
            tets,
            poses,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
        )

        self.assertEqual(state.operator_volume_policy, payload.policy)
        self.assertEqual(state.operator_volume_sha256, payload.operator_volume_sha256)
        self.assertEqual(_little_endian_bytes(state.source_rest_determinants), payload.source_rest_determinant.tobytes())
        self.assertEqual(
            _little_endian_bytes(state.source_tet_pose_determinants),
            payload.source_tet_pose_determinant.tobytes(),
        )
        self.assertEqual(_little_endian_bytes(state.source_tet_volumes), payload.rest_volume.tobytes())
        self.assertEqual(_little_endian_bytes(state.w), payload.rest_volume.tobytes())
        self.assertEqual(ts.validate_authenticated_operator_geometry(state), state.operator_geometry_sha256)
        self.assertEqual(state.projection_state_sha256, ts.projection_state_sha256(state))
        self.assertEqual(
            state.projection_state_sha256,
            "98369fec364618fb21f4b8b21bcf360c2f16712911df8e69dfcf3a4470fda902",
        )

    def test_volume_corruption_and_policy_relabel_fail_closed(self) -> None:
        """Reject runtime volume corruption and policy relabeling."""
        state = _portable_state(torch.device("cpu"))
        state.w[0] = torch.nextafter(state.w[0], torch.tensor(float("inf"), dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "canonical host volume"):
            ts.validate_authenticated_operator_geometry(state)

        state = _portable_state(torch.device("cpu"))
        state.source_tet_volumes[0] = torch.nextafter(
            state.source_tet_volumes[0],
            torch.tensor(float("inf"), dtype=torch.float64),
        )
        with self.assertRaisesRegex(ValueError, "canonical host volume"):
            ts.validate_authenticated_operator_geometry(state)

        state = _portable_state(torch.device("cpu"))
        state.operator_volume_policy = "host-float64-scalar-pose-determinant-v2"
        with self.assertRaisesRegex(ValueError, "volume policy"):
            ts.validate_authenticated_operator_geometry(state)

        state = _portable_state(torch.device("cpu"))
        state.operator_geometry_policy = ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED
        with self.assertRaisesRegex(ValueError, "portable volume binding"):
            ts.validate_authenticated_operator_geometry(state)

    def test_bound_objective_authenticates_volume_and_rejects_corruption_or_relabeling(self) -> None:
        """Bind canonical operator-volume identity into a common objective."""
        state = _portable_state(torch.device("cpu"))
        context = _bound_objective(state)

        self.assertEqual(context.operator_geometry_sha256, state.operator_geometry_sha256)
        self.assertEqual(context.operator_volume_policy, state.operator_volume_policy)
        self.assertEqual(context.operator_volume_sha256, state.operator_volume_sha256)
        context.validate_immutable()
        self.assertEqual(
            context.common_objective_sha256,
            "9955590cf76499fb9d13aa1ba5dedfa705bb92da024a9939d285646af01ff254",
        )

        owned_volume = object.__getattribute__(context, "volume")
        owned_volume[0] = torch.nextafter(owned_volume[0], torch.tensor(float("inf"), dtype=torch.float64))
        with self.assertRaisesRegex(RuntimeError, "changed after authentication"):
            context.validate_immutable()

        context = _bound_objective(state)
        object.__setattr__(context, "operator_volume_policy", "host-float64-scalar-pose-determinant-v2")
        with self.assertRaisesRegex(RuntimeError, "changed after authentication"):
            context.validate_immutable()

    def test_partial_or_forged_objective_volume_binding_is_rejected(self) -> None:
        """Reject incomplete and forged objective volume identity fields."""
        state = _portable_state(torch.device("cpu"))
        kwargs = {
            "tets": state.tets,
            "J": state.J,
            "volume": state.w,
            "mass": torch.tensor([0.8, 1.0, 1.2, 1.4], dtype=torch.float64),
            "mu": torch.tensor([17.0], dtype=torch.float64),
            "lam": torch.tensor([31.0], dtype=torch.float64),
            "inertial_target": state.rest_q + 0.015,
            "pinned": state.pinned,
            "dt": 0.08,
        }
        with self.assertRaisesRegex(ValueError, "all be provided together"):
            CommonObjectiveContext(
                **kwargs,
                operator_geometry_sha256=state.operator_geometry_sha256,
            )
        with self.assertRaisesRegex(ValueError, "operator-volume SHA-256"):
            CommonObjectiveContext(
                **kwargs,
                operator_geometry_sha256=state.operator_geometry_sha256,
                operator_volume_policy=state.operator_volume_policy,
                operator_volume_sha256="0" * 64,
            )

        unbound = CommonObjectiveContext(**kwargs)
        object.__setattr__(unbound, "operator_geometry_sha256", state.operator_geometry_sha256)
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            unbound.validate_immutable()

    def test_projection_objective_binding_rejects_unbound_or_self_consistent_relabels(self) -> None:
        """Reject missing and independently valid but relabeled objective identities."""
        state = _portable_state(torch.device("cpu"))
        common = {
            "tets": state.tets,
            "J": state.J,
            "volume": state.w,
            "mass": torch.tensor([0.8, 1.0, 1.2, 1.4], dtype=torch.float64),
            "mu": torch.tensor([17.0], dtype=torch.float64),
            "lam": torch.tensor([31.0], dtype=torch.float64),
            "inertial_target": state.rest_q + 0.015,
            "pinned": state.pinned,
            "dt": 0.08,
        }
        unbound = CommonObjectiveContext(**common)
        with self.assertRaisesRegex(ValueError, "requires a bound"):
            validate_projection_objective_volume_binding(state, unbound)

        relabeled_geometry = "0" * 64
        relabeled_volume = ts.operator_volume_sha256(
            relabeled_geometry,
            state.w.detach().cpu().numpy(),
            policy=state.operator_volume_policy,
        )
        relabeled = CommonObjectiveContext(
            **common,
            operator_geometry_sha256=relabeled_geometry,
            operator_volume_policy=state.operator_volume_policy,
            operator_volume_sha256=relabeled_volume,
        )
        with self.assertRaisesRegex(ValueError, "identities differ"):
            validate_projection_objective_volume_binding(state, relabeled)

    @unittest.skipUnless(
        os.environ.get("PSS_RUN_CUDA_PARITY") == "1" and torch.cuda.is_available(),
        "set PSS_RUN_CUDA_PARITY=1 after claiming a GPU",
    )
    def test_cpu_cuda_raw_volume_bytes_and_bound_objective_identity_are_exact(self) -> None:
        """Match raw volume bytes and bound objective SHA across CPU and CUDA."""
        cpu_state = _portable_state(torch.device("cpu"))
        cuda_state = _portable_state(torch.device("cuda"))
        cpu_context = _bound_objective(cpu_state)
        cuda_context = _bound_objective(cuda_state)

        for name in (
            "source_rest_determinants",
            "source_tet_pose_determinants",
            "source_tet_volumes",
            "w",
        ):
            self.assertEqual(_little_endian_bytes(getattr(cpu_state, name)), _little_endian_bytes(getattr(cuda_state, name)))
        self.assertEqual(cpu_state.operator_geometry_sha256, cuda_state.operator_geometry_sha256)
        self.assertEqual(cpu_state.operator_volume_sha256, cuda_state.operator_volume_sha256)
        self.assertEqual(cpu_context.common_objective_sha256, cuda_context.common_objective_sha256)


if __name__ == "__main__":
    unittest.main()
