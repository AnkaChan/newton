# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for portable host-authenticated tetrahedron volumes."""

from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from .. import torch_solver as ts
from ..graph_transformer import GraphTransformerConfig
from ..iterative_solver import (
    IterativeSolverConfig,
    PhysicalStepContext,
    solve_iterative_principal_stretch,
    validate_projection_objective_volume_binding,
)
from ..predictor import StretchPredictor, build_stretch_predictor
from ..v5_ablation import (
    AttestedVBDK1Start,
    V5AblationConfig,
    VBDK1MethodRecord,
    pin_binding_sha256,
    run_v5_identical_corrector_ablation,
)
from ..v5_corrector import CorrectorConfig, FixedPCGConfig
from ..v5_objective import CommonObjectiveContext
from .test_graph_transformer import _chain_mesh, _tet_poses
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


def _portable_chain_source() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rest, tets = _chain_mesh(2)
    source_rest = np.array(rest, dtype=np.float32, order="C", copy=True)
    source_tets = np.array(tets, dtype=np.int64, order="C", copy=True)
    source_poses = np.array(_tet_poses(source_rest, source_tets), dtype=np.float32, order="C", copy=True)
    return source_rest, source_tets, source_poses


def _strided_last_axis(value: np.ndarray) -> np.ndarray:
    padded_shape = (*value.shape[:-1], 2 * value.shape[-1])
    padded = np.empty(padded_shape, dtype=value.dtype)
    view = padded[..., ::2]
    view[...] = value
    assert not view.flags.c_contiguous
    return view


def _portable_execution_problem() -> tuple[
    ts.SolverState,
    StretchPredictor,
    PhysicalStepContext,
    dict[str, object],
]:
    rest, tets, poses = _portable_chain_source()
    state = ts.build_solver(
        rest,
        tets,
        poses,
        np.asarray([0, 1, 2], dtype=np.int64),
        torch.device("cpu"),
        dtype=torch.float64,
        operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
    )
    predictor = build_stretch_predictor(
        "graph-transformer",
        rest,
        tets,
        torch.device("cpu"),
        torch.float64,
        residual=True,
        graph_config=GraphTransformerConfig(
            hidden_dim=8,
            num_heads=2,
            n_levels=2,
            cluster_size=2,
            max_hencky_update=0.01,
            max_rotation_update=0.015,
            architecture_version=5,
        ),
    )
    predictor.eval()
    positions = state.rest_q.clone()
    mu = torch.full((state.n_tets,), 17.0, dtype=torch.float64)
    lam = torch.full((state.n_tets,), 31.0, dtype=torch.float64)
    physical_step = PhysicalStepContext(
        x_current=positions,
        x_previous=positions,
        force=torch.zeros_like(positions),
        gravity=torch.zeros(3, dtype=torch.float64),
        mu=mu,
        lam=lam,
        pin=torch.isin(state.tets, state.pinned).any(dim=-1).to(torch.float64),
        pinned_targets=positions[state.pinned],
    )
    objective_fields = {
        "tets": state.tets,
        "J": state.J,
        "volume": state.w,
        "mass": torch.ones(state.n_verts, dtype=torch.float64),
        "mu": mu,
        "lam": lam,
        "inertial_target": positions,
        "pinned": state.pinned,
        "dt": 1.0,
    }
    return state, predictor, physical_step, objective_fields


def _portable_objective(state: ts.SolverState, fields: dict[str, object], binding: str) -> CommonObjectiveContext:
    if binding == "unbound":
        return CommonObjectiveContext(**fields)
    if binding == "matched":
        return CommonObjectiveContext(
            **fields,
            operator_geometry_sha256=state.operator_geometry_sha256,
            operator_volume_policy=state.operator_volume_policy,
            operator_volume_sha256=state.operator_volume_sha256,
        )
    if binding != "mismatched":
        raise AssertionError(f"unknown test binding {binding!r}")
    relabeled_geometry = "0" * 64
    relabeled_volume = ts.operator_volume_sha256(
        relabeled_geometry,
        state.w.detach().cpu().numpy(),
        policy=state.operator_volume_policy,
    )
    return CommonObjectiveContext(
        **fields,
        operator_geometry_sha256=relabeled_geometry,
        operator_volume_policy=state.operator_volume_policy,
        operator_volume_sha256=relabeled_volume,
    )


def _portable_vbd_k1(
    state: ts.SolverState,
    objective: CommonObjectiveContext,
    physical_step: PhysicalStepContext,
) -> AttestedVBDK1Start:
    return AttestedVBDK1Start(
        positions=physical_step.x_current,
        physical_step_sha256=physical_step.physical_step_sha256,
        common_objective_sha256=objective.common_objective_sha256,
        static_mesh_sha256=state.static_mesh_sha256,
        operator_geometry_sha256=state.operator_geometry_sha256,
        projection_state_sha256=state.projection_state_sha256,
        pin_binding_sha256=pin_binding_sha256(state.pinned, physical_step.pinned_targets),
        method_record=VBDK1MethodRecord(source_run_sha256="a" * 64),
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
    def test_portable_source_arrays_require_exact_c_contiguity_at_every_entry(self) -> None:
        """Reject Fortran and strided sources before portable canonicalization."""
        rest, tets, poses = _portable_chain_source()
        variants = {
            "rest_q-fortran": (np.asfortranarray(rest), tets, poses, "rest_q"),
            "rest_q-strided": (_strided_last_axis(rest), tets, poses, "rest_q"),
            "tet_indices-fortran": (rest, np.asfortranarray(tets), poses, "tet_indices"),
            "tet_indices-strided": (rest, _strided_last_axis(tets), poses, "tet_indices"),
            "tet_poses-fortran": (rest, tets, np.asfortranarray(poses), "tet_poses"),
            "tet_poses-strided": (rest, tets, _strided_last_axis(poses), "tet_poses"),
        }
        for label, (source_rest, source_tets, source_poses, source_name) in variants.items():
            with self.subTest(layout=label):
                self.assertFalse(
                    {
                        "rest_q": source_rest,
                        "tet_indices": source_tets,
                        "tet_poses": source_poses,
                    }[source_name].flags.c_contiguous
                )
                with self.assertRaisesRegex(ValueError, rf"{source_name}.*C-contiguous"):
                    ts.canonical_operator_volume(
                        source_rest,
                        source_tets,
                        source_poses,
                        operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
                    )
                with self.assertRaisesRegex(ValueError, rf"{source_name}.*C-contiguous"):
                    ts.operator_geometry_sha256(
                        source_rest,
                        source_tets,
                        source_poses,
                        policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
                    )
                with self.assertRaisesRegex(ValueError, rf"{source_name}.*C-contiguous"):
                    ts.build_solver(
                        source_rest,
                        source_tets,
                        source_poses,
                        np.asarray([0, 1, 2], dtype=np.int64),
                        torch.device("cpu"),
                        dtype=torch.float64,
                        operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
                    )

    def test_nonportable_source_policy_preserves_layout_canonicalization(self) -> None:
        """Keep legacy promoted-policy acceptance and identities unchanged."""
        rest, tets, poses = _portable_chain_source()
        expected_geometry = ts.operator_geometry_sha256(
            rest,
            tets,
            poses,
            policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
        )
        source_rest = np.asfortranarray(rest)
        source_tets = np.asfortranarray(tets)
        source_poses = np.asfortranarray(poses)
        self.assertFalse(source_rest.flags.c_contiguous)
        self.assertFalse(source_tets.flags.c_contiguous)
        self.assertFalse(source_poses.flags.c_contiguous)
        self.assertEqual(
            ts.operator_geometry_sha256(
                source_rest,
                source_tets,
                source_poses,
                policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
            ),
            expected_geometry,
        )
        state = ts.build_solver(
            source_rest,
            source_tets,
            source_poses,
            np.asarray([0, 1, 2], dtype=np.int64),
            torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
        )
        self.assertEqual(state.operator_geometry_sha256, expected_geometry)

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
        self.assertFalse(payload.source_rest_determinant.flags["W"])
        self.assertFalse(payload.source_tet_pose_determinant.flags["W"])
        self.assertFalse(payload.rest_volume.flags["W"])
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
        self.assertEqual(
            _little_endian_bytes(state.source_rest_determinants), payload.source_rest_determinant.tobytes()
        )
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

    def test_public_solver_rejects_unbound_and_mismatched_portable_objectives(self) -> None:
        """Enforce the volume binding at the exact public solve boundary."""
        state, predictor, physical_step, fields = _portable_execution_problem()
        expected_errors = {
            "unbound": "requires a bound",
            "mismatched": "identities differ",
        }
        for binding, error in expected_errors.items():
            with self.subTest(binding=binding), self.assertRaisesRegex(ValueError, error):
                solve_iterative_principal_stretch(
                    predictor=predictor,
                    projection_state=state,
                    objective=_portable_objective(state, fields, binding),
                    physical_step=physical_step,
                    expected_physical_step_sha256=physical_step.physical_step_sha256,
                    config=IterativeSolverConfig(
                        iterations=1,
                        objective_policy="record",
                        residual_policy="record",
                        head_mode="zero",
                    ),
                )

    def test_public_solver_and_ablation_accept_matching_portable_objective(self) -> None:
        """Execute both public boundaries with one matching portable identity."""
        state, predictor, physical_step, fields = _portable_execution_problem()
        objective = _portable_objective(state, fields, "matched")
        result = solve_iterative_principal_stretch(
            predictor=predictor,
            projection_state=state,
            objective=objective,
            physical_step=physical_step,
            expected_physical_step_sha256=physical_step.physical_step_sha256,
            config=IterativeSolverConfig(
                iterations=1,
                objective_policy="record",
                residual_policy="record",
                head_mode="zero",
            ),
        )
        self.assertEqual(len(result.trace), 1)
        self.assertTrue(torch.isfinite(result.positions).all())

        ablation = run_v5_identical_corrector_ablation(
            predictor=predictor,
            projection_state=state,
            objective=objective,
            physical_step=physical_step,
            expected_physical_step_sha256=physical_step.physical_step_sha256,
            corrector_config=CorrectorConfig(
                pcg=FixedPCGConfig(iterations=1),
                candidate_alphas=(0.0,),
            ),
            vbd_k1=_portable_vbd_k1(state, objective, physical_step),
            config=V5AblationConfig(iterations=1, head_permutation=(1, 0)),
        )
        self.assertEqual(len(ablation.arms), 6)

    def test_public_ablation_rejects_unbound_and_mismatched_portable_objectives(self) -> None:
        """Enforce the volume binding at the exact public ablation boundary."""
        state, predictor, physical_step, fields = _portable_execution_problem()
        expected_errors = {
            "unbound": "requires a bound",
            "mismatched": "identities differ",
        }
        for binding, error in expected_errors.items():
            objective = _portable_objective(state, fields, binding)
            with self.subTest(binding=binding), self.assertRaisesRegex(ValueError, error):
                run_v5_identical_corrector_ablation(
                    predictor=predictor,
                    projection_state=state,
                    objective=objective,
                    physical_step=physical_step,
                    expected_physical_step_sha256=physical_step.physical_step_sha256,
                    corrector_config=CorrectorConfig(
                        pcg=FixedPCGConfig(iterations=1),
                        candidate_alphas=(0.0,),
                    ),
                    vbd_k1=_portable_vbd_k1(state, objective, physical_step),
                    config=V5AblationConfig(iterations=1, head_permutation=(1, 0)),
                )

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
            self.assertEqual(
                _little_endian_bytes(getattr(cpu_state, name)), _little_endian_bytes(getattr(cuda_state, name))
            )
        self.assertEqual(cpu_state.operator_geometry_sha256, cuda_state.operator_geometry_sha256)
        self.assertEqual(cpu_state.operator_volume_sha256, cuda_state.operator_volume_sha256)
        self.assertEqual(cpu_context.common_objective_sha256, cuda_context.common_objective_sha256)


if __name__ == "__main__":
    unittest.main()
