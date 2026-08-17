# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import hashlib
import tempfile
import unittest

import numpy as np
import torch

from .. import torch_solver as ts
from ..graph_transformer import GraphTransformerConfig
from ..iterative_solver import PhysicalStepContext
from ..train_pr_history_v5 import (
    TRAINER_EXECUTION_CONTRACT_SHA256,
    SharedTopologyPredictorBank,
    V5TrainingSample,
    _optimizer_options,
    build_v5_adamw_optimizer_contract,
    canonical_training_tensor_sha256,
    compute_v5_training_batch_loss,
    train_pr_history_v5,
    trainer_execution_contract,
)
from ..v5_checkpoint import (
    ConstraintContract,
    CorrectorContract,
    ProjectionContract,
    RepresentationContract,
    ResidualContract,
    SafeguardContract,
    TrainingStage,
    V5SolverContract,
    canonical_json_sha256,
    verify_v5_checkpoint,
)
from ..v5_dataset import (
    DataAccessLedger,
    DatasetRole,
    NumericContentIdentity,
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


def _pose(rest: np.ndarray, tets: np.ndarray) -> np.ndarray:
    result = []
    for tet in tets:
        matrix = np.stack(
            (rest[tet[1]] - rest[tet[0]], rest[tet[2]] - rest[tet[0]], rest[tet[3]] - rest[tet[0]]),
            axis=1,
        )
        result.append(np.linalg.inv(matrix))
    return np.asarray(result)


def _provenance(name: str, dt: float) -> TrajectoryProvenance:
    return TrajectoryProvenance(
        generation_spec_sha256=_digest(f"generation:{name}"),
        history_manifest_sha256=_digest(f"history:{name}"),
        root_checkpoint_sha256=_digest(f"root:{name}"),
        final_checkpoint_sha256=_digest(f"final:{name}"),
        artifact_bundle_uri=f"artifact://v5-trainer/{name}.npz",
        artifact_bundle_sha256=_digest(f"bundle:{name}"),
        artifact_source_uri=f"source://v5-trainer/{name}.json",
        artifact_source_sha256=_digest(f"source:{name}"),
        static_bundle_sha256=_digest(f"static:{name}"),
        density_kg_m3=1.0,
        initial_velocity_m_s=(0.0, 0.0, 0.0),
        pin_schedule_sha256=_digest(f"pins:{name}"),
        event_inventory_sha256=_digest(f"events:{name}"),
        coordinate_start_sha256=_digest(f"start:{name}"),
        coordinate_stop_sha256=_digest(f"stop:{name}"),
        coordinate_range_sha256=_digest(f"range:{name}"),
        dt_seconds=dt,
        generation_seed=31,
    )


def _numeric(name: str, component: str, tensor: torch.Tensor | None = None) -> NumericContentIdentity:
    digest = canonical_training_tensor_sha256(tensor) if tensor is not None else _digest(f"{name}:{component}:bytes")
    return NumericContentIdentity(identifier=f"{name}:{component}", sha256=digest)


def _sample_and_record(
    name: str,
    rest: np.ndarray,
    *,
    target_offset: tuple[float, float, float],
    ordinal: int = 0,
    trajectory_name: str | None = None,
    mu_value: float = 0.0,
    lam_value: float = 0.0,
) -> tuple[TrajectorySampleRecord, V5TrainingSample]:
    trajectory_id = name if trajectory_name is None else trajectory_name
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    pinned = np.asarray([0, 1, 2], dtype=np.int64)
    state = ts.build_solver(
        rest,
        tets,
        _pose(rest, tets),
        pinned,
        torch.device("cpu"),
        dtype=torch.float64,
        projection_backend="dense",
    )
    dt = 0.1
    x_current = torch.as_tensor(rest, dtype=torch.float64).clone()
    x_current[3] += torch.tensor([0.045, -0.035, 0.025], dtype=torch.float64)
    reference = torch.as_tensor(rest, dtype=torch.float64).clone()
    reference[3] += torch.tensor(target_offset, dtype=torch.float64)
    x_previous = x_current.clone()
    mass = torch.ones(4, dtype=torch.float64)
    force = torch.zeros_like(x_current)
    force[3] = (reference[3] - x_current[3]) / (dt * dt)
    mu = torch.full((1,), mu_value, dtype=torch.float64)
    lam = torch.full((1,), lam_value, dtype=torch.float64)
    pin = torch.ones(1, dtype=torch.float64)
    physical = PhysicalStepContext(
        x_current=x_current,
        x_previous=x_previous,
        force=force,
        gravity=torch.zeros(3, dtype=torch.float64),
        mu=mu,
        lam=lam,
        pin=pin,
        pinned_targets=x_current[state.pinned],
    )
    objective = CommonObjectiveContext(
        tets=state.tets,
        J=state.J,
        volume=state.w,
        mass=mass,
        mu=mu,
        lam=lam,
        inertial_target=reference,
        pinned=state.pinned,
        dt=dt,
    )
    reference_f = ts.compute_F(reference, state.tets, state.J)
    sample_id = f"{name}:{ordinal}"
    sample = TrajectorySampleRecord(
        sample_id=sample_id,
        ordinal=ordinal,
        topology_sha256=state.static_mesh_sha256,
        material_sha256=_digest(f"material:{trajectory_id}"),
        pin_signature_sha256=_digest(f"pins:{trajectory_id}"),
        dt_seconds=dt,
        physical_step_sha256=physical.physical_step_sha256,
        common_objective_sha256=objective.common_objective_sha256,
        observed_f=_numeric(sample_id, "observed_f"),
        input_f=_numeric(sample_id, "input_f"),
        reference_f=_numeric(sample_id, "reference_f", reference_f),
        observed_state=_numeric(sample_id, "observed_state"),
        input_state=_numeric(sample_id, "input_state"),
        reference_state=_numeric(sample_id, "reference_state", reference),
    )
    payload = V5TrainingSample(
        trajectory_id=trajectory_id,
        sample_record=sample,
        physical_step=physical,
        common_objective=objective,
        projection_state=state,
        producer_attested_reference_positions=reference,
        producer_attested_reference_deformation_gradient=reference_f,
    )
    return sample, payload


def _trajectory(
    name: str,
    rest: np.ndarray,
    target_offset: tuple[float, float, float],
    *,
    mu_value: float = 0.0,
    lam_value: float = 0.0,
) -> tuple[TrajectoryRecord, V5TrainingSample]:
    sample, payload = _sample_and_record(
        name,
        rest,
        target_offset=target_offset,
        mu_value=mu_value,
        lam_value=lam_value,
    )
    provenance = _provenance(name, sample.dt_seconds)
    record = TrajectoryRecord(
        trajectory_id=name,
        scene_family=f"scene:{name}",
        load_program_id=f"load:{name}",
        load_program_sha256=_digest(f"load-program:{name}"),
        source_chain_sha256=_digest(f"chain:{name}"),
        topology_sha256=sample.topology_sha256,
        material_sha256=sample.material_sha256,
        provenance=provenance,
        source_transition_count=1,
        samples=(sample,),
    )
    return record, payload


def _graph_config() -> GraphTransformerConfig:
    return GraphTransformerConfig(
        hidden_dim=8,
        num_heads=2,
        n_levels=0,
        cluster_size=2,
        dropout=0.0,
        max_hencky_update=0.4,
        max_rotation_update=0.75,
        architecture_version=5,
    )


def _contract(
    manifest: SplitManifest,
    schedule,
    *,
    representation_end: int,
    trained_iterations: int = 1,
) -> V5SolverContract:
    graph = dataclasses.asdict(_graph_config())
    del graph["max_multiplicative_update"]
    return V5SolverContract.build(
        graph_config=graph,
        learned_parameter_dtype="torch.float64",
        training_split=manifest,
        sampling_schedule=schedule,
        stages=(
            TrainingStage(
                name="representation",
                start_update=0,
                end_update=representation_end,
                label_config=PrincipalStretchLabelConfig(
                    max_hencky_update=0.4,
                    max_rotation_update=0.75,
                    minimum_principal_stretch=0.05,
                ),
                representation_loss_config=RepresentationLossConfig(
                    max_hencky_update=0.4,
                    max_rotation_update=0.75,
                    hencky_weight=1.0,
                    rotation_weight=1.0,
                ),
            ),
            TrainingStage(
                name="physics",
                start_update=representation_end,
                end_update=schedule.steps,
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
        trained_iterations=trained_iterations,
        inference_iterations=2,
        residual=ResidualContract(
            definition="exact-common-objective-gradient-at-current-iterate-v1",
            normalization="divide-by-common-objective-context-residual-scale",
            scale_source="derived-max-material-or-inertial-force-with-1e-12N-floor-v1",
            detach_features=True,
        ),
        representation=RepresentationContract(
            minimum_principal_stretch=0.05,
            max_hencky_update=0.4,
            max_rotation_update=0.75,
        ),
        projection=ProjectionContract(
            backend="dense",
            relative_tolerance=None,
            absolute_tolerance=None,
            max_iterations=0,
            warm_start="not-applicable",
            raise_on_nonconvergence=True,
            preconditioner="none",
            require_runtime_diagnostics=True,
            execution_dtype="torch.float64",
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
            replay_relative_tolerance=1.0e-12,
            replay_absolute_tolerance=1.0e-12,
            initializer_policy="persistence",
        ),
        optimizer=build_v5_adamw_optimizer_contract(
            learning_rate=5.0e-3,
            weight_decay=0.0,
            gradient_clip_norm=10.0,
        ),
        physical_timestep_source="common-objective-context-per-sample",
        rng_algorithm="torch-cpu-plus-numpy-pcg64",
        batch_stream_contract="pss-v5-static-layout-homogeneous-trajectory-first-sampling-v1",
    )


class TestExecutableV5Trainer(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rest_a = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.rest_b = np.asarray(
            [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.05, 0.9, 0.0], [0.08, -0.04, 1.1]],
            dtype=np.float64,
        )

    def _fixture(self, *, steps: int = 12, representation_end: int = 8):
        trajectory_a, payload_a = _trajectory("train-a", self.rest_a, (0.012, 0.018, -0.008))
        trajectory_b, payload_b = _trajectory("train-b", self.rest_b, (-0.014, 0.016, -0.006))
        manifest = SplitManifest(train=(trajectory_a, trajectory_b), validation=(), confirmation=())
        schedule = build_sampling_schedule(manifest, steps=steps, batch_size=1, seed=19)
        contract = _contract(manifest, schedule, representation_end=representation_end)
        payloads = {payload_a.key: payload_a, payload_b.key: payload_b}
        return manifest, schedule, contract, payloads

    def test_train_reduces_loss_records_gradients_and_verifies_checkpoint(self):
        manifest, schedule, contract, payloads = self._fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/v5.pt"
            result = train_pr_history_v5(
                solver_contract=contract,
                sampling_schedule=schedule,
                access_ledger=DataAccessLedger(manifest),
                payloads=payloads,
                seed=7,
                checkpoint_path=path,
            )
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                verify_v5_checkpoint(loaded).checkpoint_payload_sha256,
                result.verified_checkpoint.checkpoint_payload_sha256,
            )

        self.assertEqual(result.completed_updates, schedule.steps)
        self.assertEqual(result.next_batch_index, schedule.steps)
        self.assertEqual(
            [update.stage for update in result.updates],
            ["representation"] * 8 + ["physics"] * 4,
        )
        representation_losses = [update.loss for update in result.updates if update.stage == "representation"]
        self.assertLess(sum(representation_losses[-2:]), sum(representation_losses[:2]))
        maxima = {
            family: max(update.gradient_norms[family] for update in result.updates)
            for family in ("backbone", "context_encoder", "output_head", "rotation_head")
        }
        self.assertTrue(all(value > 0.0 and np.isfinite(value) for value in maxima.values()))
        for topology in {update.topology_sha256 for update in result.updates}:
            topology_updates = tuple(update for update in result.updates if update.topology_sha256 == topology)
            for family in ("backbone", "context_encoder", "output_head", "rotation_head"):
                self.assertGreater(max(update.gradient_norms[family] for update in topology_updates), 0.0)
        self.assertEqual(result.trainer_execution_contract_sha256, TRAINER_EXECUTION_CONTRACT_SHA256)
        self.assertFalse(trainer_execution_contract()["resume_capability"])
        continuation = result.checkpoint["metadata"]["continuation_snapshot"]
        self.assertFalse(continuation["resume_capability"])
        self.assertEqual(len(result.access_ledger.accesses), 2)
        self.assertTrue(all(access.role is DatasetRole.TRAIN for access in result.access_ledger.accesses))
        self.assertEqual(canonical_json_sha256(result.training_run_record), result.training_run_sha256)
        self.assertEqual(result.training_run_record["final_access_ledger_sha256"], result.access_ledger.ledger_sha256)
        rng_snapshot = result.checkpoint["rng_state"]
        self.assertEqual(rng_snapshot["training_run_sha256"], result.training_run_sha256)
        self.assertEqual(rng_snapshot["training_run_record"], result.training_run_record)
        self.assertEqual(rng_snapshot["model_initialization_torch_seed"], 7)
        self.assertEqual(rng_snapshot["numpy_pcg64_sampling_schedule_seed"], schedule.seed)

    def test_cross_topology_contexts_share_parameter_objects(self):
        _manifest, schedule, contract, payloads = self._fixture(steps=8, representation_end=4)
        bank = SharedTopologyPredictorBank(_graph_config(), torch.float64)
        ordered_payloads = [
            payloads[(batch.samples[0].trajectory_id, batch.samples[0].sample_id)] for batch in schedule.batches
        ]
        for payload in ordered_payloads:
            bank.ensure(payload)
        self.assertEqual(len(bank.predictors), 2)
        parameter_maps = [dict(predictor.model.named_parameters()) for predictor in bank.predictors.values()]
        for name in bank.parameter_names:
            self.assertIs(parameter_maps[0][name], parameter_maps[1][name])
        shared = bank.shared_parameters()
        self.assertEqual(len(shared), len({id(parameter) for parameter in shared}))
        result = train_pr_history_v5(
            solver_contract=contract,
            sampling_schedule=schedule,
            access_ledger=DataAccessLedger(schedule.manifest),
            payloads=payloads,
            seed=5,
        )
        expected_topologies = tuple(batch.topology_sha256 for batch in schedule.batches)
        self.assertEqual(tuple(update.topology_sha256 for update in result.updates), expected_topologies)
        self.assertGreaterEqual(len(set(expected_topologies)), 2)
        trained_maps = [dict(predictor.model.named_parameters()) for predictor in result.bank.predictors.values()]
        for name in result.bank.parameter_names:
            self.assertIs(trained_maps[0][name], trained_maps[1][name])

    def test_reference_mutation_and_manifest_role_fail_before_training(self):
        manifest, schedule, contract, payloads = self._fixture(steps=8, representation_end=4)
        payload = next(iter(payloads.values()))
        self.assertFalse(payload.producer_attested_reference_positions.requires_grad)
        self.assertFalse(payload.producer_attested_reference_deformation_gradient.requires_grad)
        payload.producer_attested_reference_positions[3, 0] += 0.25
        with self.assertRaisesRegex(ValueError, "reference positions changed"):
            train_pr_history_v5(
                solver_contract=contract,
                sampling_schedule=schedule,
                access_ledger=DataAccessLedger(manifest),
                payloads=payloads,
            )

        validation_record, validation_payload = _trajectory("validation-only", self.rest_a, (0.01, 0.01, 0.0))
        split = SplitManifest(train=manifest.train, validation=(validation_record,), confirmation=())
        train_schedule = build_sampling_schedule(split, steps=8, batch_size=1, seed=19)
        train_contract = _contract(split, train_schedule, representation_end=4)
        clean_payloads = {key: value for key, value in payloads.items() if value is not payload}
        _record, replacement = _trajectory("train-a", self.rest_a, (0.012, 0.018, -0.008))
        clean_payloads[replacement.key] = replacement
        clean_payloads[validation_payload.key] = validation_payload
        with self.assertRaisesRegex(ValueError, "exactly the samples"):
            train_pr_history_v5(
                solver_contract=train_contract,
                sampling_schedule=train_schedule,
                access_ledger=DataAccessLedger(split),
                payloads=clean_payloads,
            )

    def test_projection_execution_dtype_must_match_solver_contract(self):
        manifest, schedule, contract, payloads = self._fixture(steps=8, representation_end=4)
        mismatched = dataclasses.replace(
            contract,
            projection=dataclasses.replace(contract.projection, execution_dtype="torch.float32"),
        )
        with self.assertRaisesRegex(ValueError, "projection execution dtype"):
            train_pr_history_v5(
                solver_contract=mismatched,
                sampling_schedule=schedule,
                access_ledger=DataAccessLedger(manifest),
                payloads=payloads,
            )

    def test_checkpoint_and_sample_digest_tampering_are_rejected(self):
        manifest, schedule, contract, payloads = self._fixture(steps=8, representation_end=4)
        result = train_pr_history_v5(
            solver_contract=contract,
            sampling_schedule=schedule,
            access_ledger=DataAccessLedger(manifest),
            payloads=payloads,
            seed=3,
        )
        tampered = copy.deepcopy(result.checkpoint)
        state = tampered["state_dict"]
        key = next(iter(state))
        state[key].reshape(-1)[0] += 1.0
        with self.assertRaisesRegex(ValueError, "learned-state"):
            verify_v5_checkpoint(tampered)

        sample = next(iter(payloads.values()))
        wrong_record = dataclasses.replace(
            sample.sample_record,
            reference_state=NumericContentIdentity(
                identifier=sample.sample_record.reference_state.identifier,
                sha256=_digest("wrong-reference"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "reference positions SHA-256"):
            V5TrainingSample(
                trajectory_id=sample.trajectory_id,
                sample_record=wrong_record,
                physical_step=sample.physical_step,
                common_objective=sample.common_objective,
                projection_state=sample.projection_state,
                producer_attested_reference_positions=sample.producer_attested_reference_positions,
                producer_attested_reference_deformation_gradient=(
                    sample.producer_attested_reference_deformation_gradient
                ),
            )

        x_current, x_previous, force, gravity, mu, lam, pin, pinned_targets = sample.physical_step._owned_tensors()
        differentiable_physical = PhysicalStepContext(
            x_current=x_current.detach().clone().requires_grad_(),
            x_previous=x_previous,
            force=force,
            gravity=gravity,
            mu=mu,
            lam=lam,
            pin=pin,
            pinned_targets=pinned_targets,
        )
        self.assertEqual(
            differentiable_physical.physical_step_sha256,
            sample.physical_step.physical_step_sha256,
        )
        with self.assertRaisesRegex(ValueError, "x_current must not require gradients"):
            V5TrainingSample(
                trajectory_id=sample.trajectory_id,
                sample_record=sample.sample_record,
                physical_step=differentiable_physical,
                common_objective=sample.common_objective,
                projection_state=sample.projection_state,
                producer_attested_reference_positions=sample.producer_attested_reference_positions,
                producer_attested_reference_deformation_gradient=(
                    sample.producer_attested_reference_deformation_gradient
                ),
            )

    def test_two_sample_physics_batch_routes_distinct_objectives(self):
        sample0, payload0 = _sample_and_record(
            "pair-0",
            self.rest_a,
            target_offset=(0.01, 0.018, -0.005),
            ordinal=0,
            trajectory_name="pair",
        )
        sample1, payload1 = _sample_and_record(
            "pair-1",
            self.rest_a,
            target_offset=(-0.018, 0.006, 0.012),
            ordinal=1,
            trajectory_name="pair",
        )
        provenance = _provenance("pair", sample0.dt_seconds)
        trajectory = TrajectoryRecord(
            trajectory_id="pair",
            scene_family="scene:pair",
            load_program_id="load:pair",
            load_program_sha256=_digest("load-program:pair"),
            source_chain_sha256=_digest("chain:pair"),
            topology_sha256=sample0.topology_sha256,
            material_sha256=sample0.material_sha256,
            provenance=provenance,
            source_transition_count=2,
            samples=(sample0, sample1),
        )
        manifest = SplitManifest(train=(trajectory,), validation=(), confirmation=())
        schedule = build_sampling_schedule(manifest, steps=2, batch_size=2, seed=4)
        contract = _contract(manifest, schedule, representation_end=1)
        payloads = {payload0.key: payload0, payload1.key: payload1}
        bank = SharedTopologyPredictorBank(_graph_config(), torch.float64)
        batch = schedule.batches[1]
        for reference in batch.samples:
            bank.ensure(payloads[(reference.trajectory_id, reference.sample_id)])
        loss = compute_v5_training_batch_loss(
            bank=bank,
            batch=batch,
            payloads=payloads,
            stage=contract.stages[1],
            solver_contract=contract,
        )
        loss.total.backward()
        batch_gradient = torch.cat(
            [parameter.grad.reshape(-1) for parameter in bank.shared_parameters() if parameter.grad is not None]
        ).clone()

        for parameter in bank.shared_parameters():
            parameter.grad = None
        individual_losses = []
        for reference in batch.samples:
            individual_batch = dataclasses.replace(batch, samples=(reference,))
            individual_losses.append(
                compute_v5_training_batch_loss(
                    bank=bank,
                    batch=individual_batch,
                    payloads=payloads,
                    stage=contract.stages[1],
                    solver_contract=contract,
                ).total
            )
        loop_loss = torch.stack(individual_losses).mean()
        loop_loss.backward()
        loop_gradient = torch.cat(
            [parameter.grad.reshape(-1) for parameter in bank.shared_parameters() if parameter.grad is not None]
        ).clone()
        torch.testing.assert_close(loss.total, loop_loss, rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(batch_gradient, loop_gradient, rtol=1.0e-10, atol=1.0e-12)

        swapped = dict(payloads)
        references = tuple(batch.samples)
        first_key = (references[0].trajectory_id, references[0].sample_id)
        second_key = (references[1].trajectory_id, references[1].sample_id)
        swapped[first_key], swapped[second_key] = swapped[second_key], swapped[first_key]
        with self.assertRaisesRegex(ValueError, "scheduled reference"):
            compute_v5_training_batch_loss(
                bank=bank,
                batch=batch,
                payloads=swapped,
                stage=contract.stages[1],
                solver_contract=contract,
            )
        self.assertTrue(torch.isfinite(batch_gradient).all())
        self.assertGreater(float(torch.linalg.vector_norm(batch_gradient)), 0.0)

    def test_active_elastic_k2_recurrent_training_and_optimizer_dtype_guard(self):
        trajectory, payload = _trajectory(
            "elastic",
            self.rest_a,
            (0.014, 0.012, -0.006),
            mu_value=0.1,
            lam_value=0.2,
        )
        manifest = SplitManifest(train=(trajectory,), validation=(), confirmation=())
        schedule = build_sampling_schedule(manifest, steps=4, batch_size=1, seed=23)
        contract = _contract(manifest, schedule, representation_end=2, trained_iterations=2)
        result = train_pr_history_v5(
            solver_contract=contract,
            sampling_schedule=schedule,
            access_ledger=DataAccessLedger(manifest),
            payloads={payload.key: payload},
            seed=11,
        )
        self.assertEqual(result.verified_checkpoint.solver_contract.trained_iterations, 2)
        self.assertEqual([update.stage for update in result.updates], ["representation"] * 2 + ["physics"] * 2)
        self.assertGreater(float(payload.common_objective.mu.item()), 0.0)
        self.assertTrue(all(np.isfinite(update.loss) for update in result.updates))

        underflow = build_v5_adamw_optimizer_contract(
            learning_rate=1.0e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            beta1=1.0e-300,
        )
        with self.assertRaisesRegex(ValueError, "beta1.*representable"):
            _optimizer_options(underflow, torch.float32)


if __name__ == "__main__":
    unittest.main()
