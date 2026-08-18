# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the standalone architecture-v5 checkpoint contract."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import unittest
from unittest import mock

import numpy as np
import torch

from .. import torch_solver as ts
from .. import v5_checkpoint as checkpoint_contract
from ..graph_transformer import GraphTransformerConfig
from ..iterative_solver import (
    PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64,
    PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
    IdentityConstraintHook,
    IterativeSolverConfig,
    PhysicalStepContext,
    ProposalSafeguardConfig,
    SolverVBDStagedFloat32Evidence,
    solve_iterative_principal_stretch,
)
from ..predictor import build_stretch_predictor
from ..v5_checkpoint import (
    ConstraintContract,
    CorrectorContract,
    OptimizerContract,
    ParentLineage,
    ProjectionContract,
    ProposalSafeguardContract,
    RepresentationContract,
    ResidualContract,
    SafeguardContract,
    TrainingStage,
    V5SolverContract,
    V6SolverContract,
    build_v5_checkpoint,
    build_v5_evaluation_binding,
    build_v6_checkpoint,
    build_v6_evaluation_binding,
    learned_state_sha256,
    verify_v5_checkpoint,
    verify_v5_evaluation_binding,
    verify_v5_runtime_compatibility,
    verify_v6_checkpoint,
    verify_v6_evaluation_binding,
    verify_v6_runtime_compatibility,
)
from ..v5_dataset import (
    DataAccessLedger,
    NumericContentIdentity,
    SamplingSchedule,
    SplitManifest,
    TrajectoryProvenance,
    TrajectoryRecord,
    TrajectorySampleRecord,
    build_sampling_schedule,
)
from ..v5_objective import CommonObjectiveContext
from ..v5_training import (
    CompatibleStateLossConfig,
    PotentialExcessLossConfig,
    PrincipalStretchLabelConfig,
    RepresentationLossConfig,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _trajectory(
    name: str,
    topology: str,
    *,
    operator_geometry_sha256: str | None = None,
    physical_step_sha256: str | None = None,
    common_objective_sha256: str | None = None,
    dt_seconds: float = 1.0 / 120.0,
    physical_integration_policy: str = PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64,
    source_integration_evidence_sha256: str | None = None,
) -> TrajectoryRecord:
    material = _digest(f"material:{name}")
    dt = dt_seconds
    provenance = TrajectoryProvenance(
        generation_spec_sha256=_digest(f"generation:{name}"),
        history_manifest_sha256=_digest(f"history:{name}"),
        root_checkpoint_sha256=_digest(f"root:{name}"),
        final_checkpoint_sha256=_digest(f"final:{name}"),
        artifact_bundle_uri=f"artifact://dataset/{name}.npz",
        artifact_bundle_sha256=_digest(f"bundle:{name}"),
        artifact_source_uri=f"source://history/{name}.json",
        artifact_source_sha256=_digest(f"source:{name}"),
        static_bundle_sha256=_digest(f"static:{name}"),
        density_kg_m3=1000.0,
        initial_velocity_m_s=(0.0, 0.0, 0.0),
        pin_schedule_sha256=_digest(f"pins:{name}"),
        event_inventory_sha256=_digest(f"events:{name}"),
        coordinate_start_sha256=_digest(f"start:{name}"),
        coordinate_stop_sha256=_digest(f"stop:{name}"),
        coordinate_range_sha256=_digest(f"range:{name}"),
        dt_seconds=dt,
        generation_seed=17,
    )
    components = {
        component: NumericContentIdentity(
            identifier=f"{name}:{component}",
            sha256=_digest(f"bytes:{name}:{component}"),
        )
        for component in (
            "observed_f",
            "input_f",
            "reference_f",
            "observed_state",
            "input_state",
            "reference_state",
        )
    }
    sample = TrajectorySampleRecord(
        sample_id=f"{name}:0",
        ordinal=0,
        topology_sha256=topology,
        operator_geometry_sha256=operator_geometry_sha256 or _digest(f"operator:{name}"),
        material_sha256=material,
        pin_signature_sha256=provenance.pin_schedule_sha256,
        dt_seconds=dt,
        physical_step_sha256=physical_step_sha256 or _digest(f"physical-step:{name}"),
        physical_integration_policy=physical_integration_policy,
        source_integration_evidence_sha256=source_integration_evidence_sha256,
        common_objective_sha256=common_objective_sha256 or _digest(f"common-objective:{name}"),
        **components,
    )
    return TrajectoryRecord(
        trajectory_id=name,
        scene_family=f"scene:{name}",
        load_program_id=f"load:{name}",
        load_program_sha256=_digest(f"load:{name}"),
        source_chain_sha256=_digest(f"chain:{name}"),
        topology_sha256=topology,
        operator_geometry_sha256=sample.operator_geometry_sha256,
        material_sha256=material,
        provenance=provenance,
        source_transition_count=1,
        samples=(sample,),
    )


def _mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rest = np.asarray(
        [
            [0.013, -0.007, 0.011],
            [1.037, 0.021, -0.017],
            [0.029, 0.983, 0.043],
            [-0.019, 0.031, 1.071],
        ],
        dtype=np.float64,
    )
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    rest_matrix = np.stack((rest[1] - rest[0], rest[2] - rest[0], rest[3] - rest[0]), axis=1)
    pose = np.linalg.inv(rest_matrix)[None]
    return rest, tets, pose


class TestV5CheckpointContract(unittest.TestCase):
    def setUp(self):
        self.rest, self.tets, self.pose = _mesh()
        self.projection = ts.build_solver(
            self.rest,
            self.tets,
            self.pose,
            np.asarray([0, 1, 2], dtype=np.int64),
            torch.device("cpu"),
            dtype=torch.float64,
            projection_backend="sparse_pcg",
            pcg_relative_tolerance=1.0e-6,
            pcg_absolute_tolerance=0.0,
            pcg_max_iterations=24,
            pcg_raise_on_nonconvergence=True,
            pcg_preconditioner="jacobi",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        self.positions = torch.as_tensor(self.rest, dtype=torch.float64)
        self.mu = torch.tensor([2.0e4], dtype=torch.float64)
        self.lam = torch.tensor([3.0e4], dtype=torch.float64)
        self.physical_step = PhysicalStepContext(
            x_current=self.positions,
            x_previous=self.positions,
            force=torch.zeros_like(self.positions),
            gravity=torch.zeros(3, dtype=torch.float64),
            mu=self.mu,
            lam=self.lam,
            pin=torch.ones(1, dtype=torch.float64),
            pinned_targets=self.positions[self.projection.pinned],
        )
        self.objective = CommonObjectiveContext(
            tets=self.projection.tets,
            J=self.projection.J,
            volume=self.projection.w,
            mass=torch.ones(4, dtype=torch.float64),
            mu=self.mu,
            lam=self.lam,
            inertial_target=self.positions,
            pinned=self.projection.pinned,
            dt=1.0 / 120.0,
        )
        self.train = _trajectory("train", _digest("train-topology"))
        self.held_out = _trajectory(
            "held-out",
            self.projection.static_mesh_sha256,
            operator_geometry_sha256=self.projection.operator_geometry_sha256,
            physical_step_sha256=self.physical_step.physical_step_sha256,
            common_objective_sha256=self.objective.common_objective_sha256,
        )
        self.manifest = SplitManifest(train=(self.train,), validation=(), confirmation=(self.held_out,))
        self.schedule = build_sampling_schedule(self.manifest, steps=8, batch_size=1, seed=1701)
        self.graph_config = GraphTransformerConfig(
            hidden_dim=8,
            num_heads=2,
            n_levels=1,
            cluster_size=2,
            max_hencky_update=0.25,
            max_rotation_update=0.5,
            architecture_version=5,
        )
        self.predictor = build_stretch_predictor(
            "graph-transformer",
            self.rest,
            self.tets,
            torch.device("cpu"),
            torch.float64,
            residual=True,
            graph_config=self.graph_config,
        )
        self.predictor.eval()
        self.state = self.predictor.model.state_dict()
        self.optimizer_state = {
            "state": {0: {"step": torch.tensor(7), "exp_avg": torch.arange(3, dtype=torch.float32)}},
            "param_groups": [{"lr": 1.0e-4, "params": [0]}],
        }
        self.rng_state = {
            "torch_cpu": torch.arange(16, dtype=torch.uint8),
            "numpy_pcg64_state_sha256": _digest("numpy-rng"),
        }
        self.contract = self._contract()
        self.checkpoint = build_v5_checkpoint(
            self.state,
            solver_contract=self.contract,
            optimizer_state=self.optimizer_state,
            rng_state=self.rng_state,
            batch_stream=self.schedule,
            completed_updates=7,
            parent_lineage=ParentLineage.root(),
        )

    def _contract(
        self,
        architecture_version: int = 5,
        graph_config: dict[str, object] | None = None,
        learned_parameter_dtype: str = "torch.float64",
        training_split: SplitManifest | None = None,
        sampling_schedule: SamplingSchedule | None = None,
    ) -> V5SolverContract:
        if graph_config is None:
            graph_config = self.predictor.checkpoint_config()["graph_transformer"]
        else:
            graph_config = dict(graph_config)
        graph_config["architecture_version"] = architecture_version
        return V5SolverContract.build(
            graph_config=graph_config,
            learned_parameter_dtype=learned_parameter_dtype,
            training_split=training_split or self.manifest,
            sampling_schedule=sampling_schedule or self.schedule,
            stages=(
                TrainingStage(
                    name="representation",
                    start_update=0,
                    end_update=4,
                    label_config=PrincipalStretchLabelConfig(
                        max_hencky_update=0.25,
                        max_rotation_update=0.5,
                        minimum_principal_stretch=0.05,
                    ),
                    representation_loss_config=RepresentationLossConfig(
                        max_hencky_update=0.25,
                        max_rotation_update=0.5,
                        hencky_weight=1.0,
                        rotation_weight=1.0,
                    ),
                ),
                TrainingStage(
                    name="physics",
                    start_update=4,
                    end_update=8,
                    compatible_state_loss_config=CompatibleStateLossConfig(
                        characteristic_length_m=1.0,
                        position_denominator_floor_kg_m2=1.0e-16,
                        deformation_denominator_floor_m3=1.0e-16,
                        position_weight=1.0,
                        deformation_weight=1.0,
                    ),
                    potential_excess_loss_config=PotentialExcessLossConfig(
                        denominator_floor_joules=1.0e-12,
                        negative_baseline_tolerance_joules=1.0e-12,
                        weight=0.1,
                    ),
                ),
            ),
            trained_iterations=2,
            inference_iterations=3,
            residual=ResidualContract(
                definition="exact-common-objective-gradient-at-current-iterate-v1",
                normalization="divide-by-common-objective-context-residual-scale",
                scale_source="derived-max-material-or-inertial-force-with-1e-12N-floor-v1",
                detach_features=True,
            ),
            representation=RepresentationContract(
                minimum_principal_stretch=0.05,
                max_hencky_update=0.25,
                max_rotation_update=0.5,
            ),
            projection=ProjectionContract(
                backend="sparse_pcg",
                relative_tolerance=1.0e-6,
                absolute_tolerance=0.0,
                max_iterations=24,
                warm_start="current-iterate",
                raise_on_nonconvergence=True,
                preconditioner="jacobi",
                operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
            ),
            constraint=ConstraintContract.build(
                {
                    "schema_version": 1,
                    "kind": "identity",
                    "refresh_policy": "none",
                    "displacement_reference": "current-iterate",
                },
                prepare_cadence="before-every-learned-iteration",
                apply_cadence="after-every-projection-before-residual",
                gradient_policy="hook-defined-autograd",
            ),
            corrector=CorrectorContract(
                kind="identity",
                iterations=0,
                residual_operator_calls=0,
                preconditioner_calls=0,
                line_search_candidates=0,
            ),
            safeguards=SafeguardContract(
                minimum_determinant=0.0,
                minimum_singular_value=0.0,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                objective_increase_tolerance=0.0,
                normalized_residual_increase_tolerance=0.0,
                replay_relative_tolerance=1.0e-6 if learned_parameter_dtype == "torch.float32" else 1.0e-12,
                replay_absolute_tolerance=1.0e-7 if learned_parameter_dtype == "torch.float32" else 1.0e-12,
                initializer_policy="persistence",
            ),
            optimizer=OptimizerContract.build(
                "AdamW",
                {"learning_rate": 1.0e-4, "weight_decay": 1.0e-6, "gradient_clip_norm": 1.0},
            ),
            physical_timestep_source="common-objective-context-per-sample",
            rng_algorithm="torch-cpu-plus-numpy-pcg64",
            batch_stream_contract="pss-v5-static-layout-homogeneous-trajectory-first-sampling-v3",
        )

    def _v6_contract(
        self,
        fractions: tuple[float, ...] = (1.0, 0.5, 0.0),
        *,
        training_split: SplitManifest | None = None,
        sampling_schedule: SamplingSchedule | None = None,
    ) -> V6SolverContract:
        direct = (
            self.contract
            if training_split is None and sampling_schedule is None
            else self._contract(training_split=training_split, sampling_schedule=sampling_schedule)
        )
        return V6SolverContract.build(
            graph_config=direct.graph_config,
            learned_parameter_dtype=direct.learned_parameter_dtype,
            training_split=self.manifest if training_split is None else training_split,
            sampling_schedule=self.schedule if sampling_schedule is None else sampling_schedule,
            stages=direct.stages,
            trained_iterations=direct.trained_iterations,
            inference_iterations=direct.inference_iterations,
            residual=direct.residual,
            representation=direct.representation,
            projection=direct.projection,
            constraint=direct.constraint,
            corrector=direct.corrector,
            safeguards=direct.safeguards,
            optimizer=direct.optimizer,
            physical_timestep_source=direct.physical_timestep_source,
            rng_algorithm=direct.rng_algorithm,
            batch_stream_contract=direct.batch_stream_contract,
            training_proposal_policy="direct-unselected-record-v1",
            inference_proposal_safeguard=ProposalSafeguardContract(
                candidate_step_fractions=fractions,
            ),
        )

    def _v6_checkpoint(self, fractions: tuple[float, ...] = (1.0, 0.5, 0.0)) -> dict[str, object]:
        return build_v6_checkpoint(
            self.state,
            solver_contract=self._v6_contract(fractions),
            optimizer_state=self.optimizer_state,
            rng_state=self.rng_state,
            batch_stream=self.schedule,
            completed_updates=7,
            parent_lineage=ParentLineage.root(),
        )

    def _v6_runtime(self):
        fractions = (1.0, 0.5, 0.0)
        checkpoint = self._v6_checkpoint(fractions)
        config = IterativeSolverConfig(
            iterations=3,
            minimum_determinant=0.0,
            minimum_singular_value=0.0,
            objective_policy="require-nonincreasing",
            residual_policy="require-nonincreasing",
            objective_increase_tolerance=0.0,
            normalized_residual_increase_tolerance=0.0,
            initializer_policy="persistence",
            return_projection_diagnostics=True,
            head_mode="learned",
            proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=fractions),
        )
        constraint = IdentityConstraintHook()
        result = solve_iterative_principal_stretch(
            predictor=self.predictor,
            projection_state=self.projection,
            objective=self.objective,
            physical_step=self.physical_step,
            expected_physical_step_sha256=self.physical_step.physical_step_sha256,
            config=config,
            constraint=constraint,
        )
        ledger = DataAccessLedger(self.manifest).record_access(
            self.held_out.trajectory_id,
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("common_objective", "physical_step", "reference_state"),
        )
        binding = build_v6_evaluation_binding(
            checkpoint,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            projection_state=self.projection,
            predictor=self.predictor,
            selected_sample_ids=(self.held_out.samples[0].sample_id,),
            physical_dt_seconds=self.objective.dt,
            residual_scale=self.objective.residual_scale,
        )
        return checkpoint, config, constraint, result, ledger, binding

    def _staged_float32_inputs(self):
        dt32 = np.float32(1.0 / 120.0)
        pinned = self.projection.pinned.cpu().numpy()
        pre_event = np.asarray(self.rest, dtype=np.float32)
        velocity = np.zeros_like(pre_event)
        velocity[3] = np.asarray((0.0123, -0.0042, 0.0081), dtype=np.float32)
        displacement = (velocity * dt32).astype(np.float32)
        x_previous = (pre_event - displacement).astype(np.float32)
        inertial_target = (pre_event + displacement).astype(np.float32)
        inertial_target[pinned] = pre_event[pinned]
        source_mass = np.ones(pre_event.shape[0], dtype=np.float32)
        source_evidence = SolverVBDStagedFloat32Evidence(
            source_transition_sha256=_digest("source-transition-v6"),
            dt_seconds=float(dt32),
            pre_event_positions=torch.from_numpy(pre_event),
            velocity=torch.from_numpy(velocity),
            mass=torch.from_numpy(source_mass),
            inverse_mass=torch.from_numpy(source_mass.copy()),
        )
        physical_step = PhysicalStepContext(
            x_current=torch.from_numpy(pre_event).double(),
            x_previous=torch.from_numpy(x_previous).double(),
            force=torch.zeros_like(self.positions),
            gravity=torch.zeros(3, dtype=torch.float64),
            mu=self.mu,
            lam=self.lam,
            pin=torch.ones(1, dtype=torch.float64),
            pinned_targets=torch.from_numpy(pre_event[pinned]).double(),
            integration_policy=PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
            source_evidence=source_evidence,
        )
        objective = CommonObjectiveContext(
            tets=self.projection.tets,
            J=self.projection.J,
            volume=self.projection.w,
            mass=torch.ones(pre_event.shape[0], dtype=torch.float64),
            mu=self.mu,
            lam=self.lam,
            inertial_target=torch.from_numpy(inertial_target).double(),
            pinned=self.projection.pinned,
            dt=float(dt32),
        )
        held_out = _trajectory(
            "source-held-out-v6",
            self.projection.static_mesh_sha256,
            operator_geometry_sha256=self.projection.operator_geometry_sha256,
            physical_step_sha256=physical_step.physical_step_sha256,
            common_objective_sha256=objective.common_objective_sha256,
            dt_seconds=float(dt32),
            physical_integration_policy=PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
            source_integration_evidence_sha256=source_evidence.evidence_sha256,
        )
        manifest = SplitManifest(train=(self.train,), validation=(), confirmation=(held_out,))
        schedule = build_sampling_schedule(manifest, steps=8, batch_size=1, seed=1701)
        return physical_step, objective, source_evidence, held_out, manifest, schedule

    def test_roundtrip_and_deterministic_tensor_hash(self):
        expected_hash = learned_state_sha256(self.state)
        self.assertEqual(self.checkpoint["learned_state_sha256"], expected_hash)
        self.assertEqual(expected_hash, learned_state_sha256(dict(reversed(tuple(self.state.items())))))
        self.assertEqual(self.contract.training_dataset_sha256, self.manifest.manifest_sha256)
        self.assertEqual(self.contract.sampling_schedule_sha256, self.schedule.schedule_sha256)

        buffer = io.BytesIO()
        torch.save(self.checkpoint, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=False)
        verified = verify_v5_checkpoint(loaded)

        self.assertEqual(verified.checkpoint_payload_sha256, self.checkpoint["checkpoint_payload_sha256"])
        self.assertEqual(verified.learned_state_sha256, expected_hash)
        self.assertEqual(verified.solver_contract, self.contract)
        self.assertEqual(verified.solver_contract.inference_iterations, 3)
        self.assertEqual(verified.solver_contract.inference_work["predictor_passes"], 3)
        self.assertEqual(verified.solver_contract.inference_work["common_residual_evaluations"], 4)
        self.assertEqual(verified.solver_contract.inference_work["physical_step_authentications"], 9)
        self.assertEqual(verified.solver_contract.inference_work["common_objective_authentications"], 9)
        self.assertEqual(verified.solver_contract.inference_work["maximum_projection_iterations"], 72)
        self.assertEqual(verified.completed_updates, 7)
        continuation = loaded["metadata"]["continuation_snapshot"]
        self.assertFalse(continuation["resume_capability"])
        self.assertEqual(continuation["semantics"], "integrity-only-not-a-resume-proof")

        reconstructed = build_stretch_predictor(
            "graph-transformer",
            self.rest,
            self.tets,
            torch.device("cpu"),
            torch.float64,
            residual=True,
            graph_config=GraphTransformerConfig(**verified.solver_contract.graph_config),
        )
        reconstructed.model.load_state_dict(loaded["state_dict"], strict=True)
        self.assertEqual(learned_state_sha256(reconstructed.model.state_dict()), expected_hash)

    def test_schema_v6_roundtrip_binds_candidate_inference_without_changing_v5(self):
        self.assertEqual(
            self.contract.solver_contract_sha256,
            "84cb02c2b4e60f4205b6e6f6524f2c9700fb2623d074a395d747b07a193f77b1",
        )
        contract = self._v6_contract()
        self.assertEqual(V6SolverContract.from_dict(contract.as_dict()), contract)
        self.assertEqual(contract.trained_work, self.contract.trained_work)
        self.assertEqual(contract.inference_work["predictor_passes"], 3)
        self.assertEqual(contract.inference_work["global_compatibility_projections"], 3)
        self.assertEqual(contract.inference_work["common_residual_evaluations"], 10)
        self.assertEqual(contract.inference_work["common_objective_evaluations"], 10)
        self.assertEqual(contract.inference_work["state_validity_evaluations"], 10)
        self.assertEqual(contract.inference_work["constraint_applications"], 9)
        self.assertEqual(contract.inference_work["physical_step_authentications"], 15)
        self.assertEqual(contract.inference_work["common_objective_authentications"], 15)
        self.assertEqual(contract.training_proposal_policy, "direct-unselected-record-v1")
        self.assertEqual(
            contract.inference_proposal_safeguard.as_config(),
            ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
        )

        checkpoint = self._v6_checkpoint()
        verified = verify_v6_checkpoint(checkpoint)
        self.assertEqual(verified.solver_contract, contract)
        self.assertEqual(checkpoint["schema_version"], 6)
        self.assertEqual(checkpoint["contract"], "pss-iterative-principal-stretch-checkpoint-v6")
        with self.assertRaisesRegex(ValueError, "schema-v5 checkpoint identity"):
            verify_v5_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, "schema-v6 checkpoint identity"):
            verify_v6_checkpoint(self.checkpoint)

        tampered = copy.deepcopy(contract.as_dict())
        tampered["inference_proposal_safeguard"]["candidate_step_fractions"][1] = 0.25
        with self.assertRaisesRegex(ValueError, "schema-6 solver-contract SHA-256"):
            V6SolverContract.from_dict(tampered)
        with self.assertRaisesRegex(ValueError, "registered core policy"):
            ProposalSafeguardContract(candidate_step_fractions=(1.0, 0.0, 0.5))

    def test_schema_v6_rejects_execution_dtype_collapsed_candidates_and_safeguards(self):
        collapsed_candidates = self._v6_contract((1.0, 1.0e-50, 0.0))
        float32_projection = dataclasses.replace(
            collapsed_candidates.projection,
            execution_dtype="torch.float32",
        )
        with self.assertRaisesRegex(ValueError, "candidate_step_fractions.*execution dtype"):
            dataclasses.replace(collapsed_candidates, projection=float32_projection)

        contract = self._v6_contract()
        float32_projection = dataclasses.replace(contract.projection, execution_dtype="torch.float32")
        underflowed_safeguards = dataclasses.replace(
            contract.safeguards,
            objective_increase_tolerance=1.0e-50,
        )
        with self.assertRaisesRegex(ValueError, "objective_increase_tolerance.*execution dtype"):
            dataclasses.replace(
                contract,
                projection=float32_projection,
                safeguards=underflowed_safeguards,
            )

    def test_schema_v6_runtime_replays_every_candidate_and_rejects_forgery(self):
        checkpoint, config, constraint, result, ledger, binding = self._v6_runtime()
        self.assertEqual(binding.as_dict()["schema_version"], 4)
        self.assertEqual(
            binding.as_dict()["contract"],
            "pss-v6-fixed-candidate-held-out-evaluation-binding-v4",
        )
        self.assertEqual(binding.as_dict()["checkpoint"]["schema_version"], 6)
        self.assertEqual(
            verify_v6_evaluation_binding(
                binding,
                checkpoint=checkpoint,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                projection_state=self.projection,
                predictor=self.predictor,
            ),
            binding,
        )
        verified = verify_v6_runtime_compatibility(
            checkpoint,
            evaluation_binding=binding,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            predictor=self.predictor,
            solver_config=config,
            projection_state=self.projection,
            constraint=constraint,
            objective=self.objective,
            physical_step=self.physical_step,
            result=result,
        )
        self.assertEqual(verified.proposal_accepted_iterations, 3)
        self.assertEqual(verified.zero_step_iterations, 0)
        self.assertEqual(verified.learned_contribution_retained_iterations, 0)
        self.assertTrue(all(len(trace.candidate_evaluations) == 3 for trace in result.trace))
        self.assertTrue(all(trace.learned_displacement_retention is None for trace in result.trace))
        self.assertTrue(
            all(
                trace.selection_reason == "first-admissible-positive-candidate-zero-projected-displacement"
                for trace in result.trace
            )
        )

        runtime_arguments = {
            "evaluation_binding": binding,
            "held_out_trajectory": self.held_out,
            "split_manifest": self.manifest,
            "access_ledger": ledger,
            "predictor": self.predictor,
            "solver_config": config,
            "projection_state": self.projection,
            "constraint": constraint,
            "objective": self.objective,
            "physical_step": self.physical_step,
        }
        with self.assertRaisesRegex(ValueError, "proposal safeguard differs"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **{
                    **runtime_arguments,
                    "solver_config": dataclasses.replace(
                        config,
                        proposal_safeguard=ProposalSafeguardConfig(
                            candidate_step_fractions=(1.0, 0.25, 0.0),
                        ),
                    ),
                },
                result=result,
            )
        revalidation_tamper = dataclasses.replace(config)
        object.__setattr__(revalidation_tamper, "return_projection_diagnostics", 1)
        with self.assertRaisesRegex(TypeError, "return_projection_diagnostics must be a bool"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **{**runtime_arguments, "solver_config": revalidation_tamper},
                result=result,
            )

        first_trace = result.trace[0]
        candidates = list(first_trace.candidate_evaluations)
        candidates[0] = dataclasses.replace(candidates[0], candidate_index=1)
        forged_trace = (dataclasses.replace(first_trace, candidate_evaluations=tuple(candidates)), *result.trace[1:])
        with self.assertRaisesRegex(ValueError, "candidate order"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, trace=forged_trace),
            )

        candidates = list(first_trace.candidate_evaluations)
        candidates[0] = dataclasses.replace(candidates[0], objective=candidates[0].objective + 1.0)
        forged_trace = (dataclasses.replace(first_trace, candidate_evaluations=tuple(candidates)), *result.trace[1:])
        with self.assertRaisesRegex(ValueError, r"candidate\[0\].objective"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, trace=forged_trace),
            )

        candidates = list(first_trace.candidate_evaluations)
        candidates[0] = dataclasses.replace(candidates[0], admissible=False)
        forged_trace = (dataclasses.replace(first_trace, candidate_evaluations=tuple(candidates)), *result.trace[1:])
        with self.assertRaisesRegex(ValueError, "candidate gate admissible"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, trace=forged_trace),
            )

        candidates = list(first_trace.candidate_evaluations)
        candidates[0] = dataclasses.replace(
            candidates[0],
            displacement_retention=torch.zeros((), dtype=self.positions.dtype),
        )
        forged_trace = (dataclasses.replace(first_trace, candidate_evaluations=tuple(candidates)), *result.trace[1:])
        with self.assertRaisesRegex(ValueError, "displacement retention must be absent"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, trace=forged_trace),
            )

        zero_candidate = first_trace.candidate_evaluations[-1]
        moved_zero = zero_candidate.constrained_positions.clone()
        moved_zero[-1, 0] += 1.0e-4
        candidates = list(first_trace.candidate_evaluations)
        candidates[-1] = dataclasses.replace(zero_candidate, constrained_positions=moved_zero)
        forged_trace = (dataclasses.replace(first_trace, candidate_evaluations=tuple(candidates)), *result.trace[1:])
        with self.assertRaisesRegex(ValueError, "constrained_positions"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, trace=forged_trace),
            )

        nonselected = first_trace.candidate_evaluations[1]
        tiny_candidate = nonselected.candidate_positions.clone()
        tiny_constrained = nonselected.constrained_positions.clone()
        tiny_candidate[-1, 0] += 5.0e-13
        tiny_constrained[-1, 0] += 5.0e-13
        candidates = list(first_trace.candidate_evaluations)
        candidates[1] = dataclasses.replace(
            nonselected,
            candidate_positions=tiny_candidate,
            constrained_positions=tiny_constrained,
        )
        forged_trace = (dataclasses.replace(first_trace, candidate_evaluations=tuple(candidates)), *result.trace[1:])
        with self.assertRaisesRegex(ValueError, "exact interpolated candidate"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, trace=forged_trace),
            )

        forged_objective_after = first_trace.objective_after.clone() + 5.0e-13
        forged_trace = (
            dataclasses.replace(first_trace, objective_after=forged_objective_after),
            *result.trace[1:],
        )
        with self.assertRaisesRegex(ValueError, "objective_after internal selection"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, trace=forged_trace),
            )

        forged_second_before = result.trace[1].positions_before.clone()
        forged_second_before[-1, 0] += 5.0e-13
        forged_trace = (
            first_trace,
            dataclasses.replace(result.trace[1], positions_before=forged_second_before),
            *result.trace[2:],
        )
        with self.assertRaisesRegex(ValueError, r"trace\[1\].positions_before internal chain"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, trace=forged_trace),
            )

        with self.assertRaisesRegex(ValueError, "result.objective internal chain"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, objective=result.objective + 5.0e-13),
            )

        with self.assertRaisesRegex(ValueError, "selected candidate index"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(
                    result,
                    trace=(dataclasses.replace(first_trace, selected_candidate_index=1), *result.trace[1:]),
                ),
            )
        with self.assertRaisesRegex(ValueError, "aggregate proposal_accepted_iterations"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(
                    result,
                    proposal_accepted_iterations=result.proposal_accepted_iterations + 1,
                ),
            )
        with self.assertRaisesRegex(ValueError, "observed work constraint_applications"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(
                    result,
                    work=dataclasses.replace(
                        result.work,
                        constraint_applications=result.work.constraint_applications - 1,
                    ),
                ),
            )

    def test_schema_v6_runtime_authenticates_exact_zero_selection(self):
        predictor = copy.deepcopy(self.predictor)
        with torch.no_grad():
            predictor.model.output_head[-1].bias[0] = 1.0e-4
        checkpoint = build_v6_checkpoint(
            predictor.model.state_dict(),
            solver_contract=self._v6_contract(),
            optimizer_state=self.optimizer_state,
            rng_state=self.rng_state,
            batch_stream=self.schedule,
            completed_updates=7,
            parent_lineage=ParentLineage.root(),
        )
        config = IterativeSolverConfig(
            iterations=3,
            objective_increase_tolerance=0.0,
            normalized_residual_increase_tolerance=0.0,
            proposal_safeguard=ProposalSafeguardConfig(
                candidate_step_fractions=(1.0, 0.5, 0.0),
            ),
        )
        constraint = IdentityConstraintHook()
        result = solve_iterative_principal_stretch(
            predictor=predictor,
            projection_state=self.projection,
            objective=self.objective,
            physical_step=self.physical_step,
            expected_physical_step_sha256=self.physical_step.physical_step_sha256,
            config=config,
            constraint=constraint,
        )
        ledger = DataAccessLedger(self.manifest).record_access(
            self.held_out.trajectory_id,
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("common_objective", "physical_step", "reference_state"),
        )
        binding = build_v6_evaluation_binding(
            checkpoint,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            projection_state=self.projection,
            predictor=predictor,
            selected_sample_ids=(self.held_out.samples[0].sample_id,),
            physical_dt_seconds=self.objective.dt,
            residual_scale=self.objective.residual_scale,
        )
        verified = verify_v6_runtime_compatibility(
            checkpoint,
            evaluation_binding=binding,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            predictor=predictor,
            solver_config=config,
            projection_state=self.projection,
            constraint=constraint,
            objective=self.objective,
            physical_step=self.physical_step,
            result=result,
        )
        self.assertEqual(verified.proposal_accepted_iterations, 0)
        self.assertEqual(verified.zero_step_iterations, 3)
        self.assertEqual(verified.learned_contribution_retained_iterations, 0)
        self.assertTrue(torch.equal(result.positions, self.physical_step.x_current))
        for trace in result.trace:
            self.assertEqual(trace.selected_step_fraction, 0.0)
            self.assertEqual(trace.selection_reason, "no-admissible-positive")
            self.assertFalse(trace.proposal_accepted)
            self.assertFalse(trace.learned_contribution_retained)
            self.assertTrue(torch.equal(trace.positions, trace.positions_before))
            self.assertTrue(torch.equal(trace.candidate_evaluations[-1].constrained_positions, trace.positions_before))

    def test_schema_v6_candidate_runtime_authenticates_staged_float32_source_integration(self):
        physical_step, objective, source_evidence, held_out, manifest, schedule = self._staged_float32_inputs()
        contract = self._v6_contract(training_split=manifest, sampling_schedule=schedule)
        checkpoint = build_v6_checkpoint(
            self.state,
            solver_contract=contract,
            optimizer_state=self.optimizer_state,
            rng_state=self.rng_state,
            batch_stream=schedule,
            completed_updates=7,
            parent_lineage=ParentLineage.root(),
        )
        config = IterativeSolverConfig(
            iterations=3,
            objective_increase_tolerance=0.0,
            normalized_residual_increase_tolerance=0.0,
            proposal_safeguard=ProposalSafeguardConfig(
                candidate_step_fractions=(1.0, 0.5, 0.0),
            ),
        )
        constraint = IdentityConstraintHook()
        result = solve_iterative_principal_stretch(
            predictor=self.predictor,
            projection_state=self.projection,
            objective=objective,
            physical_step=physical_step,
            expected_physical_step_sha256=physical_step.physical_step_sha256,
            config=config,
            constraint=constraint,
        )
        ledger = DataAccessLedger(manifest).record_access(
            held_out.trajectory_id,
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("common_objective", "physical_step", "reference_state"),
        )
        binding = build_v6_evaluation_binding(
            checkpoint,
            held_out_trajectory=held_out,
            split_manifest=manifest,
            access_ledger=ledger,
            projection_state=self.projection,
            predictor=self.predictor,
            selected_sample_ids=(held_out.samples[0].sample_id,),
            physical_dt_seconds=objective.dt,
            residual_scale=objective.residual_scale,
        )
        runtime_arguments = {
            "evaluation_binding": binding,
            "held_out_trajectory": held_out,
            "split_manifest": manifest,
            "access_ledger": ledger,
            "predictor": self.predictor,
            "solver_config": config,
            "projection_state": self.projection,
            "constraint": constraint,
            "objective": objective,
            "physical_step": physical_step,
        }
        verified = verify_v6_runtime_compatibility(checkpoint, **runtime_arguments, result=result)
        self.assertEqual(verified.iterations, 3)
        self.assertEqual(
            verified.physical_integration_policy,
            PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
        )
        self.assertEqual(verified.source_integration_evidence_sha256, source_evidence.evidence_sha256)
        self.assertTrue(all(len(trace.candidate_evaluations) == 3 for trace in result.trace))

        first = result.trace[0]
        forged_index = (first.selected_candidate_index + 1) % len(first.candidate_evaluations)
        forged_trace = (dataclasses.replace(first, selected_candidate_index=forged_index), *result.trace[1:])
        with self.assertRaisesRegex(ValueError, "selected candidate index"):
            verify_v6_runtime_compatibility(
                checkpoint,
                **runtime_arguments,
                result=dataclasses.replace(result, trace=forged_trace),
            )

    def test_schema_v6_nonfinite_candidate_metrics_fail_closed(self):
        config = IterativeSolverConfig(
            iterations=3,
            objective_increase_tolerance=0.0,
            normalized_residual_increase_tolerance=0.0,
            proposal_safeguard=ProposalSafeguardConfig(
                candidate_step_fractions=(1.0, 0.5, 0.0),
            ),
        )
        before = checkpoint_contract._runtime_state_metrics(
            self.objective,
            self.projection,
            self.positions,
            detach_residual=True,
        )
        projected = self.positions.clone()
        projected[-1, 0] += 0.1
        constrained = projected.clone()
        constrained[-1, 1] = torch.nan
        metrics = checkpoint_contract._runtime_candidate_metrics(
            objective=self.objective,
            projection_state=self.projection,
            config=config,
            current=self.positions,
            projected_positions=projected,
            constrained_positions=constrained,
            pinned_targets=self.physical_step.pinned_targets,
            step_fraction=1.0,
            objective_before=before["objective"],
            normalized_residual_norm_before=before["normalized_residual_norm"],
        )
        self.assertFalse(metrics["positions_finite"])
        self.assertFalse(metrics["residual_finite"])
        self.assertFalse(metrics["state_valid"])
        self.assertFalse(metrics["learned_contribution_retained"])
        self.assertIsNone(metrics["displacement_retention"])
        self.assertIn("non-finite-positions", metrics["rejection_reasons"])
        self.assertIn("non-finite-residual", metrics["rejection_reasons"])
        checkpoint_contract._require_same_v6_candidate_tensor(
            "matching-nonfinite-candidate",
            constrained.clone(),
            constrained,
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-12,
        )

        opposite = self.positions.clone()
        opposite[-1, 0] -= 0.1
        opposite_metrics = checkpoint_contract._runtime_candidate_metrics(
            objective=self.objective,
            projection_state=self.projection,
            config=config,
            current=self.positions,
            projected_positions=projected,
            constrained_positions=opposite,
            pinned_targets=self.physical_step.pinned_targets,
            step_fraction=1.0,
            objective_before=before["objective"],
            normalized_residual_norm_before=before["normalized_residual_norm"],
        )
        self.assertLess(opposite_metrics["displacement_retention"], 0.0)
        self.assertFalse(opposite_metrics["learned_contribution_retained"])

    def test_schema_v6_runtime_rejects_nonfinite_replayed_state_norm(self):
        checkpoint, config, constraint, result, ledger, binding = self._v6_runtime()
        original = checkpoint_contract._runtime_state_metrics
        injected = False

        def overflow_first_norm(*args, **kwargs):
            nonlocal injected
            metrics = original(*args, **kwargs)
            if not injected:
                metrics["raw_residual_norm"] = torch.full_like(metrics["raw_residual_norm"], torch.inf)
                injected = True
            return metrics

        with (
            mock.patch.object(checkpoint_contract, "_runtime_state_metrics", side_effect=overflow_first_norm),
            self.assertRaisesRegex(ValueError, "initial state metric raw_residual_norm is non-finite"),
        ):
            verify_v6_runtime_compatibility(
                checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=result,
            )

    def test_schema_v6_runtime_rejects_work_and_projection_diagnostic_primitive_aliases(self):
        checkpoint, config, constraint, result, ledger, binding = self._v6_runtime()
        runtime_arguments = {
            "evaluation_binding": binding,
            "held_out_trajectory": self.held_out,
            "split_manifest": self.manifest,
            "access_ledger": ledger,
            "predictor": self.predictor,
            "solver_config": config,
            "projection_state": self.projection,
            "constraint": constraint,
            "objective": self.objective,
            "physical_step": self.physical_step,
        }
        work_aliases = (
            dataclasses.replace(result.work, projection_diagnostics_recorded=1),
            dataclasses.replace(
                result.work,
                projection_factor_solves=bool(result.work.projection_factor_solves),
            ),
        )
        for forged_work in work_aliases:
            with self.subTest(work=forged_work), self.assertRaisesRegex(ValueError, "canonical primitive type"):
                verify_v6_runtime_compatibility(
                    checkpoint,
                    **runtime_arguments,
                    result=dataclasses.replace(result, work=forged_work),
                )

        diagnostics = result.trace[0].projection_diagnostics
        self.assertIsNotNone(diagnostics)
        diagnostic_aliases = (
            dataclasses.replace(diagnostics, converged=1),
            dataclasses.replace(diagnostics, rhs_count=float(diagnostics.rhs_count)),
            dataclasses.replace(diagnostics, absolute_tolerance=0),
        )
        for forged_diagnostics in diagnostic_aliases:
            forged_first = dataclasses.replace(result.trace[0], projection_diagnostics=forged_diagnostics)
            with (
                self.subTest(diagnostics=forged_diagnostics),
                self.assertRaisesRegex(ValueError, "canonical primitive type"),
            ):
                verify_v6_runtime_compatibility(
                    checkpoint,
                    **runtime_arguments,
                    result=dataclasses.replace(result, trace=(forged_first, *result.trace[1:])),
                )

        primitive_alias_traces = (
            dataclasses.replace(result.trace[0], iteration_fraction=False),
            dataclasses.replace(result.trace[0], constraint_prepare_diagnostics={"refreshes": False}),
            dataclasses.replace(
                result.trace[0],
                candidate_evaluations=(
                    dataclasses.replace(
                        result.trace[0].candidate_evaluations[0],
                        constraint_diagnostics={"truncation_calls": False, "minimum_fraction": True},
                    ),
                    *result.trace[0].candidate_evaluations[1:],
                ),
            ),
        )
        for forged_first in primitive_alias_traces:
            with self.subTest(trace=forged_first), self.assertRaisesRegex(ValueError, "canonical primitive type"):
                verify_v6_runtime_compatibility(
                    checkpoint,
                    **runtime_arguments,
                    result=dataclasses.replace(result, trace=(forged_first, *result.trace[1:])),
                )

    def test_schema_v6_runtime_rejects_a_consistent_signed_zero_trace_forgery(self):
        checkpoint, config, constraint, result, ledger, binding = self._v6_runtime()
        last = result.trace[-1]
        selected_index = last.selected_candidate_index
        selected = last.candidate_evaluations[selected_index]
        zero_mask = (selected.normalized_residual == 0.0) & ~torch.signbit(selected.normalized_residual)
        self.assertTrue(zero_mask.any())
        flat_index = int(torch.nonzero(zero_mask.reshape(-1), as_tuple=False)[0])

        flipped_residual = selected.normalized_residual.clone()
        flipped_residual.reshape(-1)[flat_index] = -0.0
        candidates = list(last.candidate_evaluations)
        candidates[selected_index] = dataclasses.replace(selected, normalized_residual=flipped_residual)
        forged_last = dataclasses.replace(
            last,
            candidate_evaluations=tuple(candidates),
            residual_after=flipped_residual.clone(),
        )
        forged_result = dataclasses.replace(
            result,
            trace=(*result.trace[:-1], forged_last),
            normalized_residual=flipped_residual.clone(),
        )
        with self.assertRaisesRegex(ValueError, "candidate.*normalized_residual"):
            verify_v6_runtime_compatibility(
                checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=forged_result,
            )

    def test_graph_config_requires_exact_canonical_v5_fields(self):
        partial = self.predictor.checkpoint_config()["graph_transformer"]
        partial.pop("hidden_dim")
        with self.assertRaisesRegex(ValueError, "exact canonical architecture-v5 fields"):
            self._contract(graph_config=partial)

        extra = self.predictor.checkpoint_config()["graph_transformer"]
        extra["unregistered"] = 1
        with self.assertRaisesRegex(ValueError, "valid architecture-v5 GraphTransformerConfig"):
            self._contract(graph_config=extra)

    def test_tensor_and_restart_state_tampering_fail_closed(self):
        state_tamper = copy.deepcopy(self.checkpoint)
        first_name = next(iter(state_tamper["state_dict"]))
        state_tamper["state_dict"][first_name].view(-1)[0] += 1.0
        with self.assertRaisesRegex(ValueError, "learned-state SHA-256"):
            verify_v5_checkpoint(state_tamper)

        optimizer_tamper = copy.deepcopy(self.checkpoint)
        optimizer_tamper["optimizer_state"]["state"][0]["step"] += 1
        with self.assertRaisesRegex(ValueError, "optimizer-state SHA-256"):
            verify_v5_checkpoint(optimizer_tamper)

        nonfinite_optimizer = copy.deepcopy(self.optimizer_state)
        nonfinite_optimizer["state"][0]["exp_avg"][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "state tensors must be finite"):
            build_v5_checkpoint(
                self.state,
                solver_contract=self.contract,
                optimizer_state=nonfinite_optimizer,
                rng_state=self.rng_state,
                batch_stream=self.schedule,
                completed_updates=7,
                parent_lineage=ParentLineage.root(),
            )

        rng_tamper = copy.deepcopy(self.checkpoint)
        rng_tamper["rng_state"]["torch_cpu"][0] += 1
        with self.assertRaisesRegex(ValueError, "RNG-state SHA-256"):
            verify_v5_checkpoint(rng_tamper)

        stream_tamper = copy.deepcopy(self.checkpoint)
        stream_tamper["batch_stream"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "batch-stream SHA-256"):
            verify_v5_checkpoint(stream_tamper)

    def test_k_corrector_and_constraint_tampering_fail_closed(self):
        mutations = {
            "inference K": lambda solver: solver.update(inference_iterations=4),
            "corrector budget": lambda solver: solver["corrector"].update(iterations=1),
            "constraint": lambda solver: solver["constraint"]["descriptor"].update(kind="dat"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                tampered = copy.deepcopy(self.checkpoint)
                mutate(tampered["metadata"]["solver_contract"])
                with self.assertRaisesRegex(ValueError, "solver-contract SHA-256"):
                    verify_v5_checkpoint(tampered)

        stage = self.contract.stages[0].as_dict()
        stage["loss_contract"]["principal_stretch_representation"]["formula"] = "arbitrary-loss"
        with self.assertRaisesRegex(ValueError, "formula or config is not canonical"):
            TrainingStage.from_dict(stage)

        stage = self.contract.stages[1].as_dict()
        del stage["loss_contract"]["compatible_state"]["config"]["characteristic_length_m"]
        with self.assertRaisesRegex(ValueError, "config is (invalid|not canonical)"):
            TrainingStage.from_dict(stage)

    def test_schema_v3_and_v4_cannot_be_relabelled_as_v5(self):
        for old_version in (3, 4):
            with self.subTest(old_version=old_version):
                with self.assertRaisesRegex(ValueError, "architecture_version must be exactly 5"):
                    self._contract(architecture_version=old_version)

                relabelled = copy.deepcopy(self.checkpoint)
                relabelled["schema_version"] = old_version
                relabelled["contract"] = f"legacy-schema-v{old_version}"
                with self.assertRaisesRegex(ValueError, "schema-v5 checkpoint identity"):
                    verify_v5_checkpoint(relabelled)

    def test_fake_legacy_and_mesh_static_state_are_rejected(self):
        fake = {"output_head.0.weight": torch.zeros(1), "rotation_head.0.weight": torch.zeros(1)}
        with self.assertRaisesRegex(ValueError, "missing required key families"):
            build_v5_checkpoint(
                fake,
                solver_contract=self.contract,
                optimizer_state=self.optimizer_state,
                rng_state=self.rng_state,
                batch_stream=self.schedule,
                completed_updates=7,
                parent_lineage=ParentLineage.root(),
            )

        nonfinite = {name: value.clone() for name, value in self.state.items()}
        first_name = next(iter(nonfinite))
        nonfinite[first_name].reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            build_v5_checkpoint(
                nonfinite,
                solver_contract=self.contract,
                optimizer_state=self.optimizer_state,
                rng_state=self.rng_state,
                batch_stream=self.schedule,
                completed_updates=7,
                parent_lineage=ParentLineage.root(),
            )

        mixed_dtype = {name: value.clone() for name, value in self.state.items()}
        mixed_dtype[next(iter(mixed_dtype))] = mixed_dtype[next(iter(mixed_dtype))].float()
        with self.assertRaisesRegex(ValueError, "expected torch.float64"):
            build_v5_checkpoint(
                mixed_dtype,
                solver_contract=self.contract,
                optimizer_state=self.optimizer_state,
                rng_state=self.rng_state,
                batch_stream=self.schedule,
                completed_updates=7,
                parent_lineage=ParentLineage.root(),
            )

        legacy_config = dataclasses.replace(self.graph_config, architecture_version=3)
        legacy = build_stretch_predictor(
            "graph-transformer",
            self.rest,
            self.tets,
            torch.device("cpu"),
            torch.float64,
            residual=True,
            graph_config=legacy_config,
        ).model.state_dict()
        with self.assertRaisesRegex(ValueError, "v5_context_encoder"):
            build_v5_checkpoint(
                legacy,
                solver_contract=self.contract,
                optimizer_state=self.optimizer_state,
                rng_state=self.rng_state,
                batch_stream=self.schedule,
                completed_updates=7,
                parent_lineage=ParentLineage.root(),
            )

        static_state = dict(self.state)
        static_state["tets"] = torch.as_tensor(self.tets)
        with self.assertRaisesRegex(ValueError, "mesh-static"):
            build_v5_checkpoint(
                static_state,
                solver_contract=self.contract,
                optimizer_state=self.optimizer_state,
                rng_state=self.rng_state,
                batch_stream=self.schedule,
                completed_updates=7,
                parent_lineage=ParentLineage.root(),
            )

    def test_exact_runtime_floor_and_split_manifest_are_required(self):
        with self.assertRaisesRegex(ValueError, "minimum_principal_stretch must be exactly 0.05"):
            RepresentationContract(
                minimum_principal_stretch=1.0e-8,
                max_hencky_update=0.25,
                max_rotation_update=0.5,
            )
        with self.assertRaisesRegex(ValueError, "canonical SplitManifest"):
            V5SolverContract.build(
                graph_config=self.predictor.checkpoint_config()["graph_transformer"],
                learned_parameter_dtype="torch.float64",
                training_split={},
                sampling_schedule=self.schedule,
                stages=self.contract.stages,
                trained_iterations=2,
                inference_iterations=3,
                residual=self.contract.residual,
                representation=self.contract.representation,
                projection=self.contract.projection,
                constraint=self.contract.constraint,
                corrector=self.contract.corrector,
                safeguards=self.contract.safeguards,
                optimizer=self.contract.optimizer,
                physical_timestep_source="common-objective-context-per-sample",
                rng_algorithm="torch-cpu-plus-numpy-pcg64",
                batch_stream_contract="pss-v5-static-layout-homogeneous-trajectory-first-sampling-v3",
            )

        with self.assertRaisesRegex(ValueError, "no greater than"):
            ProjectionContract(
                backend="sparse_pcg",
                relative_tolerance=1.0e300,
                absolute_tolerance=0.0,
                max_iterations=24,
                warm_start="current-iterate",
                raise_on_nonconvergence=True,
                preconditioner="jacobi",
            )
        with self.assertRaisesRegex(ValueError, "registered evidence maximum"):
            dataclasses.replace(self.contract.safeguards, objective_increase_tolerance=1.0e300)
        with self.assertRaisesRegex(ValueError, "registered learned-parameter dtype policy"):
            dataclasses.replace(
                self.contract,
                safeguards=dataclasses.replace(
                    self.contract.safeguards,
                    replay_relative_tolerance=5.0e-7,
                ),
            )
        unrepresentable_graph = self.predictor.checkpoint_config()["graph_transformer"]
        unrepresentable_graph["max_hencky_update"] = 1.0e300
        with self.assertRaisesRegex(ValueError, "remain finite and positive"):
            self._contract(graph_config=unrepresentable_graph, learned_parameter_dtype="torch.float32")

    def test_nonidentity_corrector_requires_a_fixed_positive_budget(self):
        with self.assertRaisesRegex(ValueError, "registered identity corrector"):
            CorrectorContract(
                kind="matrix-free-gauss-newton",
                iterations=0,
                residual_operator_calls=0,
                preconditioner_calls=0,
                line_search_candidates=0,
            )

    def test_tiny_dense_operator_forgery_is_not_hidden_by_an_absolute_tolerance(self):
        edge = 1.0e-16
        rest = np.asarray(
            [[0.0, 0.0, 0.0], [edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]],
            dtype=np.float64,
        )
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        state = ts.build_solver(
            rest,
            tets,
            (np.eye(3, dtype=np.float64) / edge)[None],
            np.asarray([0, 1, 2], dtype=np.int64),
            torch.device("cpu"),
            dtype=torch.float64,
            projection_backend="dense",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        contract = ProjectionContract(
            backend="dense",
            relative_tolerance=None,
            absolute_tolerance=None,
            max_iterations=0,
            warm_start="not-applicable",
            raise_on_nonconvergence=True,
            preconditioner="none",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        checkpoint_contract._verify_projection_operator(state, contract)

        forged_l = state.L.clone()
        forged_l[3, 3] = 1.0e-14
        forged_cholesky = state.L_ff_chol.clone()
        forged_cholesky[0, 0] = 1.0e-7
        forged = dataclasses.replace(state, L=forged_l, L_ff_chol=forged_cholesky)
        forged.projection_state_sha256 = ts.projection_state_sha256(forged)
        self.assertEqual(forged.projection_state_sha256, ts.projection_state_sha256(forged))
        with self.assertRaisesRegex(ValueError, "dense compatibility operator"):
            checkpoint_contract._verify_projection_operator(forged, contract)

    def test_dense_factor_allows_only_locally_scaled_cholesky_cancellation(self):
        rest = np.asarray(
            [
                [0.1257302210933933, -0.1321048632913019, 0.6404226504432821],
                [0.10490011715303971, -0.535669373161111, 0.36159505490948474],
                [1.3040000451301372, 0.9470809631292422, -0.7037352358069926],
                [-1.2654214710460525, -0.6232744625373522, 0.0413259793472436],
                [-2.3250307746388343, -0.21879166393254573, -1.2459109472530652],
                [-0.7322673547034516, -0.5442589828573099, -0.31630015636915454],
                [0.4116305363741328, 1.0425133694426776, -0.12853466294403426],
                [1.3664634705496859, -0.6651946734866135, 0.3515100700930197],
            ],
            dtype=np.float64,
        )
        tets = np.asarray(
            [
                [2, 5, 4, 7],
                [2, 5, 6, 4],
                [1, 0, 7, 6],
                [1, 5, 0, 6],
                [1, 2, 6, 7],
                [1, 2, 7, 5],
                [1, 2, 5, 6],
                [3, 1, 7, 5],
                [3, 1, 5, 0],
                [3, 5, 4, 6],
                [3, 5, 6, 0],
            ],
            dtype=np.int64,
        )
        origin = rest[tets[:, 0]]
        rest_matrix = np.stack(
            (rest[tets[:, 1]] - origin, rest[tets[:, 2]] - origin, rest[tets[:, 3]] - origin),
            axis=-1,
        )
        state = ts.build_solver(
            rest,
            tets,
            np.linalg.inv(rest_matrix),
            np.asarray([0], dtype=np.int64),
            torch.device("cpu"),
            dtype=torch.float64,
            projection_backend="dense",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        contract = ProjectionContract(
            backend="dense",
            relative_tolerance=None,
            absolute_tolerance=None,
            max_iterations=0,
            warm_start="not-applicable",
            raise_on_nonconvergence=True,
            preconditioner="none",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        reduced = state.L[state.free][:, state.free]
        product = state.L_ff_chol @ state.L_ff_chol.transpose(0, 1)
        structural_zero = reduced == 0.0
        self.assertTrue(structural_zero.any())
        self.assertGreater(float(product[structural_zero].abs().max()), 0.0)
        checkpoint_contract._verify_projection_operator(state, contract)

    def test_sparse_operator_uses_local_assembly_contribution_scale(self):
        rest = np.asarray(
            [
                [-0.9891213503478509, -0.3677866514678832, 1.2879252612892487],
                [0.1939744191326132, 0.9202308996398569, 0.5771037912572513],
                [-0.6364636463709805, 0.5419522204102933, -0.3165954511658161],
                [-0.32238911615896015, 0.09716731867045719, -1.5259304065189514],
                [1.1921661041016585, -0.6710896751741096, 1.0002694196594604],
                [0.1363211238531175, 1.5320330796287964, -0.6599694137918207],
                [-0.31179485646991756, 0.337769126558826, -2.2074710981998042],
                [0.8279214415587369, 1.541630394690618, 1.126806793265028],
            ],
            dtype=np.float64,
        )
        tets = np.asarray(
            [
                [1, 7, 0, 4],
                [1, 2, 4, 0],
                [1, 2, 0, 7],
                [3, 2, 0, 4],
                [3, 2, 6, 0],
                [3, 1, 2, 4],
                [5, 1, 7, 2],
                [5, 2, 7, 0],
                [5, 3, 2, 6],
                [5, 3, 1, 2],
                [5, 3, 6, 4],
                [5, 1, 4, 7],
                [5, 3, 4, 1],
            ],
            dtype=np.int64,
        )
        origin = rest[tets[:, 0]]
        rest_matrix = np.stack(
            (rest[tets[:, 1]] - origin, rest[tets[:, 2]] - origin, rest[tets[:, 3]] - origin),
            axis=-1,
        )
        poses = np.linalg.inv(rest_matrix)
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                state = ts.build_solver(
                    rest,
                    tets,
                    poses,
                    np.asarray([0], dtype=np.int64),
                    torch.device("cpu"),
                    dtype=dtype,
                    projection_backend="sparse_pcg",
                    pcg_relative_tolerance=1.0e-6,
                    pcg_absolute_tolerance=0.0,
                    pcg_max_iterations=24,
                    pcg_raise_on_nonconvergence=True,
                    pcg_preconditioner="jacobi",
                    operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
                )
                contract = ProjectionContract(
                    backend="sparse_pcg",
                    relative_tolerance=1.0e-6,
                    absolute_tolerance=0.0,
                    max_iterations=24,
                    warm_start="current-iterate",
                    raise_on_nonconvergence=True,
                    preconditioner="jacobi",
                    execution_dtype=str(dtype),
                    operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
                )
                checkpoint_contract._verify_projection_operator(state, contract)

    def test_numpy_inverse_is_bound_by_local_backward_error(self):
        rest = np.random.default_rng(639).normal(size=(8, 3))
        tets = np.asarray(
            [
                [3, 4, 7, 6],
                [5, 4, 6, 0],
                [5, 3, 0, 6],
                [5, 3, 6, 4],
                [5, 2, 4, 0],
                [5, 3, 2, 0],
                [5, 1, 4, 2],
                [5, 3, 4, 7],
                [5, 1, 7, 4],
                [5, 3, 7, 2],
                [5, 1, 2, 7],
            ],
            dtype=np.int64,
        )
        origin = rest[tets[:, 0]]
        rest_matrix = np.stack(
            (rest[tets[:, 1]] - origin, rest[tets[:, 2]] - origin, rest[tets[:, 3]] - origin),
            axis=-1,
        )
        poses = np.linalg.inv(rest_matrix)
        state = ts.build_solver(
            rest,
            tets,
            poses,
            np.asarray([0], dtype=np.int64),
            torch.device("cpu"),
            dtype=torch.float64,
            projection_backend="dense",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        torch_inverse = torch.linalg.inv(torch.as_tensor(rest_matrix, dtype=torch.float64))
        relative_eps = (state.Dm_inv - torch_inverse).abs() / torch_inverse.abs() / torch.finfo(torch.float64).eps
        self.assertGreater(float(relative_eps.nan_to_num().max()), 1.0e5)
        contract = ProjectionContract(
            backend="dense",
            relative_tolerance=None,
            absolute_tolerance=None,
            max_iterations=0,
            warm_start="not-applicable",
            raise_on_nonconvergence=True,
            preconditioner="none",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        checkpoint_contract._verify_projection_operator(state, contract)

    def test_source_promoted_operator_is_reconstructed_from_exact_float32_bytes(self):
        rest32 = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        poses32 = np.eye(3, dtype=np.float32)[None]
        state = ts.build_solver(
            rest32,
            tets,
            poses32,
            np.asarray([0, 1, 2], dtype=np.int64),
            torch.device("cpu"),
            dtype=torch.float64,
            projection_backend="dense",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
        )
        contract = ProjectionContract(
            backend="dense",
            relative_tolerance=None,
            absolute_tolerance=None,
            max_iterations=0,
            warm_start="not-applicable",
            raise_on_nonconvergence=True,
            preconditioner="none",
            execution_dtype="torch.float64",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
        )

        checkpoint_contract._verify_projection_operator(state, contract)
        self.assertTrue(torch.equal(state.Dm_inv, state.source_tet_poses.to(torch.float64)))
        with self.assertRaisesRegex(ValueError, "policy differs"):
            checkpoint_contract._verify_projection_operator(
                state,
                dataclasses.replace(
                    contract,
                    operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
                ),
            )

    def test_cross_topology_evaluation_uses_a_separate_verified_binding(self):
        self.assertNotEqual(self.train.topology_sha256, self.held_out.topology_sha256)
        ledger = DataAccessLedger(self.manifest).record_access(
            self.held_out.trajectory_id,
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("common_objective", "physical_step", "reference_state"),
        )
        binding = build_v5_evaluation_binding(
            self.checkpoint,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            projection_state=self.projection,
            predictor=self.predictor,
            selected_sample_ids=(self.held_out.samples[0].sample_id,),
            physical_dt_seconds=1.0 / 120.0,
            residual_scale=self.objective.residual_scale,
        )
        verified = verify_v5_evaluation_binding(
            binding,
            checkpoint=self.checkpoint,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            projection_state=self.projection,
            predictor=self.predictor,
        )

        self.assertEqual(verified, binding)
        self.assertEqual(binding.checkpoint_payload_sha256, self.checkpoint["checkpoint_payload_sha256"])
        self.assertEqual(binding.held_out_topology_sha256, self.held_out.topology_sha256)
        self.assertEqual(binding.held_out_operator_geometry_sha256, self.projection.operator_geometry_sha256)
        self.assertEqual(binding.selected_samples[0].operator_geometry_sha256, self.projection.operator_geometry_sha256)
        self.assertEqual(binding.physical_dt_seconds, 1.0 / 120.0)
        self.assertEqual(binding.physical_timestep_source, "common-objective-context-per-sample")

        forged_sparse = self.projection.L_ff_sparse.clone()
        forged_sparse.values().mul_(2.0)
        forged_projection = dataclasses.replace(
            self.projection,
            L_ff_sparse=forged_sparse,
            L_ff_inverse_diagonal=self.projection.L_ff_inverse_diagonal.clone() * 0.5,
        )
        forged_projection.projection_state_sha256 = ts.projection_state_sha256(forged_projection)
        with self.assertRaisesRegex(ValueError, "canonical.*reconstruction"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                projection_state=forged_projection,
                predictor=self.predictor,
                selected_sample_ids=(self.held_out.samples[0].sample_id,),
                physical_dt_seconds=self.objective.dt,
                residual_scale=self.objective.residual_scale,
            )

        forged_predictor = copy.deepcopy(self.predictor)
        forged_predictor.model.corner_force_weight.add_(0.125)
        forged_predictor.model.static_graph_sha256 = forged_predictor.model.compute_static_graph_sha256()
        with self.assertRaisesRegex(ValueError, "canonical reconstruction"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                projection_state=self.projection,
                predictor=forged_predictor,
                selected_sample_ids=(self.held_out.samples[0].sample_id,),
                physical_dt_seconds=self.objective.dt,
                residual_scale=self.objective.residual_scale,
            )

        float32_projection = ts.build_solver(
            self.rest,
            self.tets,
            self.pose,
            np.asarray([0, 1, 2], dtype=np.int64),
            torch.device("cpu"),
            dtype=torch.float32,
            projection_backend="sparse_pcg",
            pcg_relative_tolerance=1.0e-6,
            pcg_absolute_tolerance=0.0,
            pcg_max_iterations=24,
            pcg_raise_on_nonconvergence=True,
            pcg_preconditioner="jacobi",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        float32_contract = dataclasses.replace(
            self.contract,
            projection=dataclasses.replace(self.contract.projection, execution_dtype="torch.float32"),
        )
        float32_checkpoint = build_v5_checkpoint(
            self.state,
            solver_contract=float32_contract,
            optimizer_state=self.optimizer_state,
            rng_state=self.rng_state,
            batch_stream=self.schedule,
            completed_updates=7,
            parent_lineage=ParentLineage.root(),
        )
        float32_binding = build_v5_evaluation_binding(
            float32_checkpoint,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            projection_state=float32_projection,
            predictor=self.predictor,
            selected_sample_ids=(self.held_out.samples[0].sample_id,),
            physical_dt_seconds=self.objective.dt,
            residual_scale=self.objective.residual_scale,
        )
        self.assertEqual(float32_binding.held_out_topology_sha256, self.held_out.topology_sha256)

        incomplete_ledger = DataAccessLedger(self.manifest).record_access(
            self.held_out.trajectory_id,
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("reference_state",),
        )
        with self.assertRaisesRegex(ValueError, "must include physical_step and common_objective"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=incomplete_ledger,
                projection_state=self.projection,
                predictor=self.predictor,
                selected_sample_ids=(self.held_out.samples[0].sample_id,),
                physical_dt_seconds=self.objective.dt,
                residual_scale=self.objective.residual_scale,
            )

        with self.assertRaisesRegex(ValueError, "canonical TrajectoryRecord"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.held_out.as_dict(),
                split_manifest=self.manifest,
                access_ledger=ledger,
                projection_state=self.projection,
                predictor=self.predictor,
                selected_sample_ids=(self.held_out.samples[0].sample_id,),
                physical_dt_seconds=1.0 / 120.0,
                residual_scale=self.objective.residual_scale,
            )

        with self.assertRaisesRegex(ValueError, "physical_dt_seconds differs from a selected sample"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                projection_state=self.projection,
                predictor=self.predictor,
                selected_sample_ids=(self.held_out.samples[0].sample_id,),
                physical_dt_seconds=1.0 / 60.0,
                residual_scale=self.objective.residual_scale,
            )

        train_ledger = DataAccessLedger(self.manifest).record_access(
            self.train.trajectory_id,
            purpose="model_selection",
            scope="payload",
            payload_names=("common_objective", "physical_step", "reference_state"),
        )
        with self.assertRaisesRegex(ValueError, "held-out split role"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.train,
                split_manifest=self.manifest,
                access_ledger=train_ledger,
                projection_state=self.projection,
                predictor=self.predictor,
                selected_sample_ids=(self.train.samples[0].sample_id,),
                physical_dt_seconds=1.0 / 120.0,
                residual_scale=self.objective.residual_scale,
            )

    def test_runtime_verifier_binds_concrete_execution_and_rejects_mismatches(self):
        self.predictor.model.eval()
        config = IterativeSolverConfig(
            iterations=3,
            minimum_determinant=0.0,
            minimum_singular_value=0.0,
            objective_policy="require-nonincreasing",
            residual_policy="require-nonincreasing",
            objective_increase_tolerance=0.0,
            normalized_residual_increase_tolerance=0.0,
            initializer_policy="persistence",
            return_projection_diagnostics=True,
            head_mode="learned",
        )
        constraint = IdentityConstraintHook()
        result = solve_iterative_principal_stretch(
            predictor=self.predictor,
            projection_state=self.projection,
            objective=self.objective,
            physical_step=self.physical_step,
            expected_physical_step_sha256=self.physical_step.physical_step_sha256,
            config=config,
            constraint=constraint,
        )
        ledger = DataAccessLedger(self.manifest).record_access(
            self.held_out.trajectory_id,
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("common_objective", "physical_step", "reference_state"),
        )
        binding = build_v5_evaluation_binding(
            self.checkpoint,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            projection_state=self.projection,
            predictor=self.predictor,
            selected_sample_ids=(self.held_out.samples[0].sample_id,),
            physical_dt_seconds=self.objective.dt,
            residual_scale=self.objective.residual_scale,
        )
        verified = verify_v5_runtime_compatibility(
            self.checkpoint,
            evaluation_binding=binding,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            predictor=self.predictor,
            solver_config=config,
            projection_state=self.projection,
            constraint=constraint,
            objective=self.objective,
            physical_step=self.physical_step,
            result=result,
        )
        self.assertEqual(verified.iterations, 3)
        self.assertEqual(verified.physical_step_sha256, self.physical_step.physical_step_sha256)
        self.assertEqual(verified.common_objective_sha256, self.objective.common_objective_sha256)
        self.assertEqual(verified.operator_geometry_sha256, self.projection.operator_geometry_sha256)

        with self.assertRaisesRegex(ValueError, "iteration count"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=dataclasses.replace(config, iterations=2),
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=result,
            )
        with self.assertRaisesRegex(ValueError, "projection (state|policy)"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=dataclasses.replace(self.projection, pcg_max_iterations=12),
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=result,
            )
        with self.assertRaisesRegex(ValueError, "safeguard policy"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=dataclasses.replace(config, objective_policy="record"),
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=result,
            )
        with self.assertRaisesRegex(ValueError, "projection-diagnostics policy"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=dataclasses.replace(config, return_projection_diagnostics=False),
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=result,
            )
        with self.assertRaisesRegex(ValueError, "learned, unpermuted"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=dataclasses.replace(result, head_mode="zero"),
            )
        with self.assertRaisesRegex(ValueError, "registered identity-constraint scope"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=dataclasses.replace(
                    result,
                    constraint_registration="unregistered-custom-no-authenticated-execution",
                ),
            )

        wrong_mesh_projection = dataclasses.replace(
            self.projection,
            static_mesh_sha256=_digest("other-mesh"),
        )
        wrong_mesh_projection.projection_state_sha256 = ts.projection_state_sha256(wrong_mesh_projection)
        with self.assertRaisesRegex(ValueError, "static mesh differs"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=wrong_mesh_projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=result,
            )

        altered_predictor = copy.deepcopy(self.predictor)
        with torch.no_grad():
            next(altered_predictor.model.parameters()).reshape(-1)[0] += 1.0
        with self.assertRaisesRegex(ValueError, "learned-state SHA-256"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=altered_predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=result,
            )

        altered_buffers = copy.deepcopy(self.predictor)
        buffer = next(altered_buffers.model.buffers())
        buffer.reshape(-1)[0] += 1
        with self.assertRaisesRegex(ValueError, "static graph"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=altered_buffers,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=result,
            )

        altered_projection = dataclasses.replace(
            self.projection,
            L_ff_inverse_diagonal=self.projection.L_ff_inverse_diagonal.clone(),
        )
        altered_projection.L_ff_inverse_diagonal[0] *= 2.0
        with self.assertRaisesRegex(ValueError, "projection state"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=altered_projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=result,
            )

        other_physical_step = PhysicalStepContext(
            x_current=self.positions,
            x_previous=self.positions,
            force=torch.full_like(self.positions, 1.0e-9),
            gravity=torch.zeros(3, dtype=torch.float64),
            mu=self.mu,
            lam=self.lam,
            pin=torch.ones(1, dtype=torch.float64),
            pinned_targets=self.positions[self.projection.pinned],
        )
        with self.assertRaisesRegex(ValueError, "physical step differs from the selected sample"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=other_physical_step,
                result=result,
            )

        altered_objective = CommonObjectiveContext(
            tets=self.projection.tets,
            J=self.projection.J,
            volume=self.projection.w,
            mass=torch.full((4,), 2.0, dtype=torch.float64),
            mu=self.mu,
            lam=self.lam,
            inertial_target=self.positions,
            pinned=self.projection.pinned,
            dt=self.objective.dt,
        )
        with self.assertRaisesRegex(ValueError, "common objective differs from the selected sample"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=altered_objective,
                physical_step=self.physical_step,
                result=result,
            )

        forged_positions = result.positions.clone()
        forged_positions[-1, 0] += 1.0e-3
        with self.assertRaisesRegex(ValueError, "result.positions"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=dataclasses.replace(result, positions=forged_positions),
            )

        forged_trace = list(result.trace)
        forged_trace[0] = dataclasses.replace(
            forged_trace[0],
            projection_diagnostics=dataclasses.replace(
                forged_trace[0].projection_diagnostics,
                matrix_vector_products=0,
                initial_residual_norm_max=123.0,
                hierarchy_levels=999,
                preconditioner_matrix_vector_products=999,
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact work replay"):
            verify_v5_runtime_compatibility(
                self.checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=self.objective,
                physical_step=self.physical_step,
                result=dataclasses.replace(result, trace=tuple(forged_trace)),
            )

    def test_runtime_replay_authenticates_staged_float32_source_integration(self):
        dt32 = np.float32(1.0 / 120.0)
        pinned = self.projection.pinned.cpu().numpy()
        pre_event = np.asarray(self.rest, dtype=np.float32)
        velocity = np.zeros_like(pre_event)
        velocity[3] = np.asarray((0.0123, -0.0042, 0.0081), dtype=np.float32)
        displacement = (velocity * dt32).astype(np.float32)
        x_previous = (pre_event - displacement).astype(np.float32)
        inertial_target = (pre_event + displacement).astype(np.float32)
        inertial_target[pinned] = pre_event[pinned]
        source_mass = np.ones(pre_event.shape[0], dtype=np.float32)
        source_evidence = SolverVBDStagedFloat32Evidence(
            source_transition_sha256=_digest("source-transition"),
            dt_seconds=float(dt32),
            pre_event_positions=torch.from_numpy(pre_event),
            velocity=torch.from_numpy(velocity),
            mass=torch.from_numpy(source_mass),
            inverse_mass=torch.from_numpy(source_mass.copy()),
        )
        physical_step = PhysicalStepContext(
            x_current=torch.from_numpy(pre_event).double(),
            x_previous=torch.from_numpy(x_previous).double(),
            force=torch.zeros_like(self.positions),
            gravity=torch.zeros(3, dtype=torch.float64),
            mu=self.mu,
            lam=self.lam,
            pin=torch.ones(1, dtype=torch.float64),
            pinned_targets=torch.from_numpy(pre_event[pinned]).double(),
            integration_policy=PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
            source_evidence=source_evidence,
        )
        objective = CommonObjectiveContext(
            tets=self.projection.tets,
            J=self.projection.J,
            volume=self.projection.w,
            mass=torch.ones(pre_event.shape[0], dtype=torch.float64),
            mu=self.mu,
            lam=self.lam,
            inertial_target=torch.from_numpy(inertial_target).double(),
            pinned=self.projection.pinned,
            dt=float(dt32),
        )
        held_out = _trajectory(
            "source-held-out",
            self.projection.static_mesh_sha256,
            operator_geometry_sha256=self.projection.operator_geometry_sha256,
            physical_step_sha256=physical_step.physical_step_sha256,
            common_objective_sha256=objective.common_objective_sha256,
            dt_seconds=float(dt32),
            physical_integration_policy=PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
            source_integration_evidence_sha256=source_evidence.evidence_sha256,
        )
        manifest = SplitManifest(train=(self.train,), validation=(), confirmation=(held_out,))
        schedule = build_sampling_schedule(manifest, steps=8, batch_size=1, seed=1701)
        contract = self._contract(training_split=manifest, sampling_schedule=schedule)
        checkpoint = build_v5_checkpoint(
            self.state,
            solver_contract=contract,
            optimizer_state=self.optimizer_state,
            rng_state=self.rng_state,
            batch_stream=schedule,
            completed_updates=7,
            parent_lineage=ParentLineage.root(),
        )
        config = IterativeSolverConfig(
            iterations=3,
            minimum_determinant=0.0,
            minimum_singular_value=0.0,
            objective_policy="require-nonincreasing",
            residual_policy="require-nonincreasing",
            objective_increase_tolerance=0.0,
            normalized_residual_increase_tolerance=0.0,
            initializer_policy="persistence",
            return_projection_diagnostics=True,
            head_mode="learned",
        )
        constraint = IdentityConstraintHook()
        result = solve_iterative_principal_stretch(
            predictor=self.predictor,
            projection_state=self.projection,
            objective=objective,
            physical_step=physical_step,
            expected_physical_step_sha256=physical_step.physical_step_sha256,
            config=config,
            constraint=constraint,
        )
        ledger = DataAccessLedger(manifest).record_access(
            held_out.trajectory_id,
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("common_objective", "physical_step", "reference_state"),
        )
        binding = build_v5_evaluation_binding(
            checkpoint,
            held_out_trajectory=held_out,
            split_manifest=manifest,
            access_ledger=ledger,
            projection_state=self.projection,
            predictor=self.predictor,
            selected_sample_ids=(held_out.samples[0].sample_id,),
            physical_dt_seconds=objective.dt,
            residual_scale=objective.residual_scale,
        )
        verified = verify_v5_runtime_compatibility(
            checkpoint,
            evaluation_binding=binding,
            held_out_trajectory=held_out,
            split_manifest=manifest,
            access_ledger=ledger,
            predictor=self.predictor,
            solver_config=config,
            projection_state=self.projection,
            constraint=constraint,
            objective=objective,
            physical_step=physical_step,
            result=result,
        )
        self.assertEqual(
            verified.physical_integration_policy,
            PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
        )
        self.assertEqual(verified.source_integration_evidence_sha256, source_evidence.evidence_sha256)

        class StringSubclass(str):
            pass

        with self.assertRaisesRegex(ValueError, "result physical integration policy"):
            verify_v5_runtime_compatibility(
                checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=held_out,
                split_manifest=manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=objective,
                physical_step=physical_step,
                result=dataclasses.replace(
                    result,
                    physical_integration_policy=PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64,
                ),
            )
        with self.assertRaisesRegex(ValueError, "result physical integration policy"):
            verify_v5_runtime_compatibility(
                checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=held_out,
                split_manifest=manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=objective,
                physical_step=physical_step,
                result=dataclasses.replace(
                    result,
                    physical_integration_policy=StringSubclass(result.physical_integration_policy),
                ),
            )
        with self.assertRaisesRegex(ValueError, "result source integration evidence"):
            verify_v5_runtime_compatibility(
                checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=held_out,
                split_manifest=manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=objective,
                physical_step=physical_step,
                result=dataclasses.replace(result, source_integration_evidence_sha256="b" * 64),
            )
        with self.assertRaisesRegex(ValueError, "result source integration evidence"):
            verify_v5_runtime_compatibility(
                checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=held_out,
                split_manifest=manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=objective,
                physical_step=physical_step,
                result=dataclasses.replace(
                    result,
                    source_integration_evidence_sha256=StringSubclass(result.source_integration_evidence_sha256),
                ),
            )

        relabelled_selection = dataclasses.replace(
            binding.selected_samples[0],
            physical_integration_policy=PHYSICAL_INTEGRATION_POLICY_ALGEBRAIC_FLOAT64,
            source_integration_evidence_sha256=None,
        )
        relabelled_binding = dataclasses.replace(binding, selected_samples=(relabelled_selection,))
        with self.assertRaisesRegex(ValueError, "evaluation binding.*held-out selection"):
            verify_v5_runtime_compatibility(
                checkpoint,
                evaluation_binding=relabelled_binding,
                held_out_trajectory=held_out,
                split_manifest=manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=objective,
                physical_step=physical_step,
                result=result,
            )

        with self.assertRaisesRegex(ValueError, "physical integration policy is not registered"):
            dataclasses.replace(
                binding.selected_samples[0],
                physical_integration_policy=StringSubclass(binding.selected_samples[0].physical_integration_policy),
            )

        algebraic_step = PhysicalStepContext(
            x_current=physical_step.x_current,
            x_previous=physical_step.x_previous,
            force=physical_step.force,
            gravity=physical_step.gravity,
            mu=physical_step.mu,
            lam=physical_step.lam,
            pin=physical_step.pin,
            pinned_targets=physical_step.pinned_targets,
        )
        with self.assertRaisesRegex(ValueError, "physical step differs from the selected sample"):
            verify_v5_runtime_compatibility(
                checkpoint,
                evaluation_binding=binding,
                held_out_trajectory=held_out,
                split_manifest=manifest,
                access_ledger=ledger,
                predictor=self.predictor,
                solver_config=config,
                projection_state=self.projection,
                constraint=constraint,
                objective=objective,
                physical_step=algebraic_step,
                result=result,
            )

    def test_predictor_hooks_and_noncanonical_execution_surface_are_rejected(self):
        self.predictor.model.eval()
        config = IterativeSolverConfig(
            iterations=3,
            objective_increase_tolerance=0.0,
            normalized_residual_increase_tolerance=0.0,
            return_projection_diagnostics=True,
        )
        constraint = IdentityConstraintHook()
        ledger = DataAccessLedger(self.manifest).record_access(
            self.held_out.trajectory_id,
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("common_objective", "physical_step", "reference_state"),
        )
        binding = build_v5_evaluation_binding(
            self.checkpoint,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            projection_state=self.projection,
            predictor=self.predictor,
            selected_sample_ids=(self.held_out.samples[0].sample_id,),
            physical_dt_seconds=self.objective.dt,
            residual_scale=self.objective.residual_scale,
        )

        handle = self.predictor.model.output_head.register_forward_hook(
            lambda _module, _arguments, output: torch.zeros_like(output)
        )
        try:
            hooked_result = solve_iterative_principal_stretch(
                predictor=self.predictor,
                projection_state=self.projection,
                objective=self.objective,
                physical_step=self.physical_step,
                expected_physical_step_sha256=self.physical_step.physical_step_sha256,
                config=config,
                constraint=constraint,
            )
            self.assertEqual(
                float(torch.stack([trace.delta_h.abs().max() for trace in hooked_result.trace]).max()), 0.0
            )
            with self.assertRaisesRegex(ValueError, "active _forward_hooks hook"):
                verify_v5_runtime_compatibility(
                    self.checkpoint,
                    evaluation_binding=binding,
                    held_out_trajectory=self.held_out,
                    split_manifest=self.manifest,
                    access_ledger=ledger,
                    predictor=self.predictor,
                    solver_config=config,
                    projection_state=self.projection,
                    constraint=constraint,
                    objective=self.objective,
                    physical_step=self.physical_step,
                    result=hooked_result,
                )
        finally:
            handle.remove()

        overridden = copy.deepcopy(self.predictor)
        overridden.forward = lambda *_args, **_kwargs: None
        with self.assertRaisesRegex(ValueError, "overrides method 'forward'"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                projection_state=self.projection,
                predictor=overridden,
                selected_sample_ids=(self.held_out.samples[0].sample_id,),
                physical_dt_seconds=self.objective.dt,
                residual_scale=self.objective.residual_scale,
            )

        class NoncanonicalSiLU(torch.nn.SiLU):
            pass

        altered_tree = copy.deepcopy(self.predictor)
        altered_tree.model.output_head[1] = NoncanonicalSiLU()
        altered_tree.eval()
        with self.assertRaisesRegex(ValueError, "module tree differs"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                projection_state=self.projection,
                predictor=altered_tree,
                selected_sample_ids=(self.held_out.samples[0].sample_id,),
                physical_dt_seconds=self.objective.dt,
                residual_scale=self.objective.residual_scale,
            )

        training_child = copy.deepcopy(self.predictor)
        next(module for module in training_child.modules() if isinstance(module, torch.nn.Dropout)).train()
        with self.assertRaisesRegex(ValueError, "must be in evaluation mode"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                projection_state=self.projection,
                predictor=training_child,
                selected_sample_ids=(self.held_out.samples[0].sample_id,),
                physical_dt_seconds=self.objective.dt,
                residual_scale=self.objective.residual_scale,
            )

        class ModelSubclass(type(self.predictor.model)):
            pass

        subclassed_model = copy.deepcopy(self.predictor)
        subclassed_model.model.__class__ = ModelSubclass
        with self.assertRaisesRegex(ValueError, "exact PrincipalStretchGraphTransformer type"):
            build_v5_evaluation_binding(
                self.checkpoint,
                held_out_trajectory=self.held_out,
                split_manifest=self.manifest,
                access_ledger=ledger,
                projection_state=self.projection,
                predictor=subclassed_model,
                selected_sample_ids=(self.held_out.samples[0].sample_id,),
                physical_dt_seconds=self.objective.dt,
                residual_scale=self.objective.residual_scale,
            )

    def test_runtime_replay_supports_bound_float32_learned_dtype(self):
        predictor = build_stretch_predictor(
            "graph-transformer",
            self.rest,
            self.tets,
            torch.device("cpu"),
            torch.float32,
            residual=True,
            graph_config=self.graph_config,
        )
        predictor.eval()
        contract = self._contract(
            graph_config=predictor.checkpoint_config()["graph_transformer"],
            learned_parameter_dtype="torch.float32",
        )
        checkpoint = build_v5_checkpoint(
            predictor.model.state_dict(),
            solver_contract=contract,
            optimizer_state=self.optimizer_state,
            rng_state=self.rng_state,
            batch_stream=self.schedule,
            completed_updates=7,
            parent_lineage=ParentLineage.root(),
        )
        config = IterativeSolverConfig(
            iterations=3,
            objective_increase_tolerance=0.0,
            normalized_residual_increase_tolerance=0.0,
            return_projection_diagnostics=True,
        )
        result = solve_iterative_principal_stretch(
            predictor=predictor,
            projection_state=self.projection,
            objective=self.objective,
            physical_step=self.physical_step,
            expected_physical_step_sha256=self.physical_step.physical_step_sha256,
            config=config,
            constraint=IdentityConstraintHook(),
        )
        ledger = DataAccessLedger(self.manifest).record_access(
            self.held_out.trajectory_id,
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("common_objective", "physical_step", "reference_state"),
        )
        binding = build_v5_evaluation_binding(
            checkpoint,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            projection_state=self.projection,
            predictor=predictor,
            selected_sample_ids=(self.held_out.samples[0].sample_id,),
            physical_dt_seconds=self.objective.dt,
            residual_scale=self.objective.residual_scale,
        )
        verified = verify_v5_runtime_compatibility(
            checkpoint,
            evaluation_binding=binding,
            held_out_trajectory=self.held_out,
            split_manifest=self.manifest,
            access_ledger=ledger,
            predictor=predictor,
            solver_config=config,
            projection_state=self.projection,
            constraint=IdentityConstraintHook(),
            objective=self.objective,
            physical_step=self.physical_step,
            result=result,
        )
        self.assertEqual(verified.learned_state_sha256, checkpoint["learned_state_sha256"])
        self.assertEqual(
            verified.claim_scope,
            "authenticated-development-replay-not-learned-contribution-or-promotion-evidence",
        )


if __name__ == "__main__":
    unittest.main()
