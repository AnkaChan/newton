# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate detached mixed-query training and exact pool continuation."""

import csv
import importlib.util
import itertools
import json
import math
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.learned_intrinsic_solver.train_mixed import MixedTrainConfig, _scheduled_learning_rate

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253 -- Optional experimental training tests.

from experiments.learned_intrinsic_solver import features, mixed_validation, train_mixed
from experiments.learned_intrinsic_solver.contact_scene import ContactPartners, sample_contact_partners
from experiments.learned_intrinsic_solver.curriculum import MixedCurriculum
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.history import HISTORY_KEYS, store_history
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.multiscale import screen_geometry
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.train_mixed import (
    PERTURBED_CANDIDATE_PROBABILITY,
    MixedTrainConfig,
    _allow_early_stop,
    _batch,
    _checked_forward,
    _full_horizon_iterations,
    _TrajectoryFactory,
    local_objective,
    run_training,
)
from experiments.learned_intrinsic_solver.training_schedule import PlateauController

_REMOVED_UPDATE_KEYS = ("shortened_query_count", "mean_acceptance_scale")
_CONTACT_PAYLOAD_KEYS = (
    "contact_sample_index",
    "contact_kind",
    "contact_partner_point",
    "contact_partner_normal",
    "contact_partner_radius",
)
_FLOOR_SCENE = {"contact_plane_probability": 1.0, "contact_plane_height_range": (-0.02, -0.01)}
"""Floor 1-2 cm below the rest bottom faces: inside r = 5 cm at reset, so every trajectory starts penetrating."""


def _candidate_factory(rest, config):
    """Build a factory around a stand-in step exposing only CPU topology."""
    step = SimpleNamespace(
        fixed_indices=torch.tensor(np.flatnonzero(rest.corner_rest_positions[:, 2] == 0), dtype=torch.long),
        cell_corner_indices=torch.zeros(len(rest.cell_corner_indices), 8, dtype=torch.long),
    )
    return _TrajectoryFactory(step, rest, config, rank=0)


def _payload(seed, inertial, fixed, *, physical=None, physical_age=0):
    return {
        "seed": seed,
        "physical_age": physical_age,
        "physical_positions": (inertial if physical is None else physical).clone(),
        "candidate": inertial.clone(),
        "inertial_prediction": inertial.clone(),
        "fixed_positions": inertial[fixed].clone(),
    }


class TestMixedTraining(unittest.TestCase):
    """Check the optimizer contract rather than a copied implementation."""

    @staticmethod
    def config(epochs, **overrides):
        """Use a tiny real hex problem with physical and inner transitions."""
        values = {
            "cell_counts": (1, 1, 2),
            "cell_size": 0.1,
            "hidden_dim": 8,
            "edge_hidden_dim": 4,
            "num_heads": 2,
            "batch_size": 2,
            "pool_multiplier": 2,
            "queries_per_epoch": 8,
            "max_epochs": epochs,
            "stage_epochs": 1,
            "iteration_counts": (1, 2),
            "physical_step_counts": (1, 2),
            "validation_count": 2,
            "validation_iterations": 3,
            "validation_physical_steps": 2,
            "validation_physical_iterations": 2,
            "validation_full_count": 1,
            "validation_full_interval": 2,
            "device": "cpu",
            "cpu_threads": 1,
            "preparation_workers": 1,
            "verbose": False,
            "early_stopping": False,
        }
        values.update(overrides)
        return MixedTrainConfig(**values)

    def _step(self, config):
        """Build the network and step exactly as run_training does from ``config``."""
        rest = generate_cuboid(config.cell_counts, cell_size=config.cell_size)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        network = IntrinsicSolverNetwork(
            rest.cell_counts,
            config.state_feature_dim,
            conditioning_dim=config.conditioning_dim,
            hidden_dim=config.hidden_dim,
            edge_hidden_dim=config.edge_hidden_dim,
            num_heads=config.num_heads,
            edge_network=config.edge_network,
            contact_tokens=config.contact,
        )
        step = MixedHexSolverStep(
            rest,
            fixed,
            network=network,
            time_step=config.time_step,
            contact_max_pairs=config.contact_max_pairs,
            contact_tokens_per_cell=config.contact_tokens_per_cell,
            contact_friction_epsilon=config.contact_friction_epsilon,
        )
        self.addCleanup(step.close)
        return step, rest

    def test_revised_schema_defaults_and_validation(self):
        """Default to the contact-aware schema with the edge network, the floor and the validation settings."""
        config = MixedTrainConfig()
        self.assertEqual(config.feature_schema_version, features.FEATURE_SCHEMA_VERSION)
        self.assertEqual(
            (config.state_feature_dim, config.conditioning_dim), (features.STATE_FEATURE_DIM, features.CONDITIONING_DIM)
        )
        self.assertEqual(config.conditioning_dim, 9)
        self.assertEqual(config.energy_floor_scale, 1.0)
        self.assertEqual(config.validation_full_count, 16)
        self.assertEqual(config.validation_full_interval, 5)
        self.assertEqual((config.validation_interval, config.validation_full_iterations), (1, None))
        self.assertEqual(config.plateau_min_final_stage_epochs, 20)
        self.assertEqual(config.stage_descent_rate, 0.8)
        self.assertEqual(config.stage_max_epochs, 20)
        self.assertIs(config.edge_network, True)
        self.assertIs(config.contact, True)
        self.assertIs(config.checkpoint_chunks, False)
        self.assertEqual(config.contact_plane_probability, 0.8)
        self.assertEqual(config.contact_plane_height_range, (-0.15, -0.005))
        self.assertEqual(config.contact_max_points, 64)
        self.assertEqual(config.contact_point_radius_range, (0.5, 2.0))
        self.assertEqual(config.contact_kappa_range, (0.1, 10.0))
        self.assertEqual(config.contact_beta_range, (0.0, 1.0))
        self.assertEqual(config.contact_mu_range, (0.0, 1.0))
        self.assertEqual(config.contact_max_pairs, 4)
        self.assertEqual(config.contact_tokens_per_cell, 24)
        self.assertEqual(config.contact_friction_epsilon, 1e-2)
        self.assertEqual((config.hops, config.hidden_dim, config.num_heads, config.max_step_size), ((1,), 128, 4, 0.05))
        self.assertFalse(hasattr(config, "candidate_probabilities"))
        self.assertFalse(hasattr(config, "geometry_backtracking"))
        self.assertEqual(MixedTrainConfig.from_checkpoint_config(asdict(config)), config)
        for field, value in (
            ("energy_floor_scale", 0.0),
            ("energy_floor_scale", math.inf),
            ("validation_full_count", 0),
            ("validation_full_interval", True),
            ("validation_interval", 0),
            ("validation_interval", True),
            ("validation_interval", 2.0),
            ("validation_full_iterations", 0),
            ("validation_full_iterations", True),
            ("validation_full_iterations", 2.5),
            ("plateau_min_final_stage_epochs", -1),
            ("stage_max_epochs", 0),
            ("stage_max_epochs", 9),
            ("stage_max_epochs", 20.5),
            ("edge_network", 1),
            ("contact", "yes"),
            ("checkpoint_chunks", 1),
            ("contact_plane_probability", 1.5),
            ("contact_plane_probability", -0.1),
            ("contact_plane_probability", math.nan),
            ("contact_plane_height_range", (0.0, -0.1)),
            ("contact_plane_height_range", (-math.inf, 0.0)),
            ("contact_max_points", -1),
            ("contact_max_points", True),
            ("contact_point_radius_range", (0.0, 1.0)),
            ("contact_kappa_range", (0.0, 1.0)),
            ("contact_beta_range", (-0.1, 1.0)),
            ("contact_mu_range", (0.5, 0.1)),
            ("contact_max_pairs", -1),
            ("contact_tokens_per_cell", 0),
            ("contact_friction_epsilon", 0.0),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                MixedTrainConfig(**{field: value})
        self.assertEqual(MixedTrainConfig(plateau_min_final_stage_epochs=0).plateau_min_final_stage_epochs, 0)
        sparse = MixedTrainConfig(contact_max_points=0, contact_max_pairs=0, contact_beta_range=(0.0, 0.0))
        self.assertEqual((sparse.contact_max_points, sparse.contact_max_pairs), (0, 0))
        self.assertEqual(MixedTrainConfig(contact_kappa_range=[0.5, 2]).contact_kappa_range, (0.5, 2))
        # Checkpoints written before the memory knob default to dense backpropagation.
        without_knob = {k: v for k, v in asdict(config).items() if k != "checkpoint_chunks"}
        self.assertIs(MixedTrainConfig.from_checkpoint_config(without_knob).checkpoint_chunks, False)
        # Checkpoints written before the validation budget knobs validated every epoch at the largest K.
        budget_knobs = ("validation_interval", "validation_full_iterations")
        without_budget = {k: v for k, v in asdict(config).items() if k not in budget_knobs}
        rebuilt = MixedTrainConfig.from_checkpoint_config(without_budget)
        self.assertEqual((rebuilt.validation_interval, rebuilt.validation_full_iterations), (1, None))
        self.assertEqual(MixedTrainConfig(validation_interval=4, validation_full_iterations=8).validation_interval, 4)

    def test_legacy_checkpoint_configurations_are_rejected_explicitly(self):
        """Never reshape a 38/86-feature checkpoint into the revised schema."""
        current = asdict(self.config(1))
        legacy_variants = {
            "schema_1": {**current, "feature_schema_version": 1},
            "schema_2": {**current, "feature_schema_version": 2},
            "missing_schema": {k: v for k, v in current.items() if k != "feature_schema_version"},
            "candidate_probabilities": {**current, "candidate_probabilities": (0.5, 0.35, 0.1, 0.05)},
            "geometry_backtracking": {**current, "geometry_backtracking": False},
        }
        for name, values in legacy_variants.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "legacy"):
                MixedTrainConfig.from_checkpoint_config(values)
        for version in (1, 2, 3, 5):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "legacy"):
                replace(self.config(1), feature_schema_version=version)

    def test_local_objective_matches_the_leco_formula_with_the_floor(self):
        """Scale by max(|E_before|, floor), detach everything but the new energy."""
        after = torch.tensor([1.0, 0.3, 0.0, -0.2], requires_grad=True)
        before = torch.tensor([2.0, 0.2, 0.0, -0.1], requires_grad=True)
        floor = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
        losses = local_objective(after, before, floor, increase_weight=2.0)
        scale = np.array([2.0, 0.5, 0.5, 0.5])
        expected = np.arcsinh(after.detach().numpy() / scale) + 2.0 * np.maximum(
            (after.detach().numpy() - before.detach().numpy()) / scale, 0
        )
        np.testing.assert_allclose(losses.detach().numpy(), expected, rtol=1e-6, atol=1e-7)
        losses.sum().backward()
        self.assertIsNone(before.grad)
        self.assertIsNone(floor.grad)
        # d/dE_after asinh(E/s) = 1 / (s sqrt(1 + (E/s)^2)); the penalty adds w/s where E_after > E_before.
        gradient = 1 / (scale * np.sqrt(1 + (after.detach().numpy() / scale) ** 2))
        gradient[1] += 2.0 / scale[1]
        np.testing.assert_allclose(after.grad.numpy(), gradient, rtol=1e-6, atol=1e-7)
        scalar = local_objective(torch.tensor([3.0]), torch.tensor([0.0]), 1.5)
        self.assertAlmostEqual(float(scalar), math.asinh(2.0) + 2.0, places=6)
        for bad in (0.0, -1.0, math.nan):
            with self.subTest(floor=bad), self.assertRaises(ValueError):
                local_objective(after.detach(), before.detach(), bad)

    def test_candidate_modes_are_equiprobable_and_deterministic(self):
        """Draw inertial and perturbed-inertial candidates 50/50 without any screening."""
        rest = generate_cuboid((1, 1, 2), cell_size=0.1)
        config = self.config(1)
        factory = _candidate_factory(rest, config)
        fixed = factory.fixed_indices
        base = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)
        base[:, 0] += 0.3 * base[:, 2]
        modes, seeds = Counter(), 240
        for seed in range(seeds):
            prepared = factory._candidate(_payload(seed, base, fixed))
            again = factory._candidate(_payload(seed, base, fixed))
            self.assertIn(prepared["candidate_mode"], ("inertial", "perturbed_inertial"))
            self.assertNotIn("initializer_fallback", prepared)
            self.assertNotIn("initializer_halvings", prepared)
            torch.testing.assert_close(prepared["candidate"], again["candidate"], rtol=0, atol=0)
            torch.testing.assert_close(prepared["candidate"][fixed], base[fixed], rtol=0, atol=0)
            if prepared["candidate_mode"] == "inertial":
                torch.testing.assert_close(prepared["candidate"], base, rtol=0, atol=0)
            else:
                noise = (prepared["candidate"] - base).square().sum(-1).mean().sqrt().item()
                self.assertGreater(noise, 0)
                self.assertLess(noise, 0.1 * config.cell_size * 1.01)
            modes[prepared["candidate_mode"]] += 1
        self.assertEqual(PERTURBED_CANDIDATE_PROBABILITY, 0.5)
        self.assertEqual(sum(modes.values()), seeds)
        self.assertGreaterEqual(modes["perturbed_inertial"] / seeds, 0.4)
        self.assertLessEqual(modes["perturbed_inertial"] / seeds, 0.6)
        different = factory._candidate(_payload(0, base, fixed, physical_age=1))
        self.assertFalse(
            torch.equal(different["candidate"], factory._candidate(_payload(0, base, fixed))["candidate"])
            and different["candidate_mode"] == "inertial"
        )

    def test_inverted_perturbed_candidate_is_accepted_without_fallback(self):
        """Keep inverted candidates; only nonfinite initializations raise."""
        rest = generate_cuboid((1, 1, 2), cell_size=0.1)
        factory = _candidate_factory(rest, self.config(1))
        fixed = factory.fixed_indices
        base = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)
        inverted = base.clone()
        free = torch.ones(len(base), dtype=torch.bool)
        free[fixed] = False
        inverted[free, 2] = -inverted[free, 2]
        self.assertLess(min(screen_geometry(rest, inverted.numpy()).values()), 0)
        seen = set()
        for seed in range(64):
            prepared = factory._candidate(_payload(seed, inverted, fixed, physical=base))
            seen.add(prepared["candidate_mode"])
            self.assertTrue(torch.isfinite(prepared["candidate"]).all())
            self.assertLess(min(screen_geometry(rest, prepared["candidate"].numpy()).values()), 0)
            self.assertFalse(torch.equal(prepared["candidate"], base))
            if prepared["candidate_mode"] == "inertial":
                torch.testing.assert_close(prepared["candidate"], inverted, rtol=0, atol=0)
            else:
                self.assertFalse(torch.equal(prepared["candidate"], inverted))
                self.assertLess((prepared["candidate"] - inverted).abs().max().item(), 0.1)
        self.assertEqual(seen, {"inertial", "perturbed_inertial"})
        nonfinite = base.clone()
        nonfinite[-1, 1] = math.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            factory._candidate(_payload(0, nonfinite, fixed))

    def test_history_is_stored_after_a_query_carried_by_advance_and_absent_after_reset(self):
        """Carry the detached world gradient and achieved update across physical steps only."""
        config = self.config(1)
        step, rest = self._step(config)
        factory = _TrajectoryFactory(step, rest, config, rank=0)
        cells = len(rest.cell_corner_indices)
        payload = factory.reset(0)
        self.addCleanup(lambda: step.context_specs and factory.retire(payload))
        self.assertFalse(payload["history_valid"])
        for name in HISTORY_KEYS[:2]:
            self.assertEqual(payload[name].shape, (cells, 3, 3))
            self.assertEqual(payload[name].abs().sum().item(), 0.0)
        batch = _batch([payload], torch.device("cpu"), cell_count=cells)
        self.assertEqual(batch["history"].valid.tolist(), [False])
        inputs = step.prepare_inputs(
            batch["candidate"],
            batch["inertial_prediction"],
            batch["context_ids"],
            previous_positions=batch["physical_positions"],
            history=batch["history"],
        )
        self.assertEqual(inputs.state_features[..., -1].unique().tolist(), [0.0])
        result = _checked_forward(step, step, batch)
        store_history([payload], result)
        self.assertTrue(payload["history_valid"])
        torch.testing.assert_close(
            payload["history_axis_gradient_world"], result.axis_gradient_world[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            payload["history_axis_update_world"], result.achieved_axis_update_world[0], rtol=0, atol=0
        )
        self.assertFalse(payload["history_axis_gradient_world"].requires_grad)
        self.assertGreater(payload["history_axis_gradient_world"].abs().sum().item(), 0)
        payload["candidate"] = result.positions[0].detach()
        advanced = factory.advance(payload)
        self.assertTrue(advanced["history_valid"])
        for name in HISTORY_KEYS[:2]:
            torch.testing.assert_close(advanced[name], payload[name], rtol=0, atol=0)
        self.assertEqual(advanced["physical_age"], 1)
        carried = _batch([advanced], torch.device("cpu"))
        self.assertEqual(carried["history"].valid.tolist(), [True])
        torch.testing.assert_close(carried["history"].axis_gradient_world[0], payload[HISTORY_KEYS[0]], rtol=0, atol=0)
        inputs = step.prepare_inputs(
            carried["candidate"],
            carried["inertial_prediction"],
            carried["context_ids"],
            previous_positions=carried["physical_positions"],
            history=carried["history"],
        )
        self.assertEqual(inputs.state_features[..., -1].unique().tolist(), [1.0])
        fresh = factory.reset(1)
        self.assertFalse(fresh["history_valid"])
        self.assertEqual(fresh[HISTORY_KEYS[0]].abs().sum().item(), 0.0)
        factory.retire(fresh)
        factory.retire(payload)
        self.assertEqual(step.context_specs, {})

    def test_batch_infers_history_or_reports_none(self):
        """Collate stored blocks without a cell count and treat historyless payloads as none."""
        records = [
            {
                "candidate": torch.zeros(8, 3),
                "physical_positions": torch.zeros(8, 3),
                "inertial_prediction": torch.zeros(8, 3),
                "fixed_positions": torch.zeros(4, 3),
                "context_id": "a",
                "history_axis_gradient_world": torch.ones(2, 3, 3),
                "history_axis_update_world": 2 * torch.ones(2, 3, 3),
                "history_valid": True,
            },
            {
                "candidate": torch.zeros(8, 3),
                "physical_positions": torch.zeros(8, 3),
                "inertial_prediction": torch.zeros(8, 3),
                "fixed_positions": torch.zeros(4, 3),
                "context_id": "b",
            },
        ]
        batch = _batch(records, torch.device("cpu"))
        self.assertEqual(batch["history"].valid.tolist(), [True, False])
        self.assertEqual(batch["history"].axis_gradient_world.shape, (2, 2, 3, 3))
        self.assertEqual(batch["history"].axis_update_world[0].sum().item(), 36.0)
        self.assertEqual(batch["history"].axis_gradient_world[1].sum().item(), 0.0)
        self.assertIsNone(_batch(records[1:], torch.device("cpu"))["history"])
        with self.assertRaises(ValueError):
            _batch([{**records[1], "history_valid": True}], torch.device("cpu"))

    def test_training_smoke_reports_residuals_history_and_selection(self):
        """Run two tiny CPU epochs end to end and check the revised report and checkpoints."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run_training(output, self.config(2))
            self.assertEqual(report["completed_epochs"], 2)
            self.assertEqual(report["completed_updates"], 8)
            for row in report["updates"]:
                for key in ("loss", "before_joule", "after_joule", "mean_force_residual_n", "tie_cell_count"):
                    self.assertIn(key, row)
                self.assertTrue(math.isfinite(row["gradient_norm"]) and row["gradient_norm"] >= 0)
                self.assertGreaterEqual(row["mean_force_residual_n"], 0)
                self.assertLessEqual(row["step_size_min"], row["step_size_mean"])
                self.assertLessEqual(row["step_size_mean"], row["step_size_max"])
                self.assertLess(row["step_size_max"], self.config(2).max_step_size)
                self.assertTrue(
                    math.isfinite(row["contact_max_penetration_r"]) and row["contact_max_penetration_r"] >= 0
                )
                self.assertTrue(math.isfinite(row["contact_pair_mean"]) and row["contact_pair_mean"] >= 0)
                for key in _REMOVED_UPDATE_KEYS:
                    self.assertNotIn(key, row)
            for row in report["epochs"]:
                for key in _REMOVED_UPDATE_KEYS:
                    self.assertNotIn(key, row)
                self.assertEqual(sum(row["candidate_modes"].values()), row["query_count"])
                self.assertTrue(set(row["candidate_modes"]) <= {"inertial", "perturbed_inertial"})
                self.assertIn("selection", row["validation"])
                self.assertIn("force_residual", row["validation"])
                self.assertIn("penetration", row["validation"])
                self.assertIn("allow_early_stop", row)
                self.assertLessEqual(row["gradient_norm_mean"], row["gradient_norm_max"])
                self.assertTrue(0.0 <= row["contact_scene_fraction"] <= 1.0)
                self.assertTrue(0.0 <= row["contact_realized_fraction"] <= row["contact_scene_fraction"])
                self.assertGreaterEqual(row["contact_max_penetration_r"], 0.0)
            self.assertIsNone(report["epochs"][0]["full_horizon_validation"])
            row = report["epochs"][1]
            full = row["full_horizon_validation"]
            # The full-horizon check runs at the largest K and H available to the current stage.
            self.assertEqual(
                (full["iterations"], full["physical_steps"], full["sample_count"]),
                (max(row["available_K"]), max(row["available_H"]), 1),
            )
            self.assertEqual((full["iterations"], full["physical_steps"]), (1, 2))
            self.assertIn("final_free_force_residual_norm_n", full)
            self.assertIn("final_max_penetration_r", full)
            selection = report["epochs"][-1]["validation"]["selection"]
            if selection["eligible"]:
                self.assertIsNotNone(report["best_selection"])
                self.assertTrue((output / "checkpoints/best_validation.pt").is_file())
                self.assertLessEqual(report["best_selection"]["metric"], selection["metric"])
            saved = torch.load(output / "checkpoints/latest.pt", weights_only=False)
            records = saved["rank_states"][0]["pool"]["records"]
            self.assertTrue(all(set(HISTORY_KEYS) <= set(record["payload"]) for record in records))
            for record in records:
                # History exists exactly when this trajectory has received at least one query.
                queried = record["inner_iteration"] > 0 or record["physical_step"] > 0
                self.assertEqual(record["payload"]["history_valid"], queried, record)
                self.assertEqual(record["payload"]["history_axis_gradient_world"].device.type, "cpu")
                self.assertEqual(record["payload"]["history_axis_gradient_world"].abs().sum().item() > 0, queried)
                self.assertIn(record["payload"]["candidate_mode"], ("inertial", "perturbed_inertial"))
                # The scene and its frozen pair list travel with the payload.
                scene = ContactPartners.from_dict(record["payload"]["contact_partners"])
                self.assertEqual(scene.to_dict(), record["payload"]["contact_partners"])
                for key in _CONTACT_PAYLOAD_KEYS:
                    self.assertEqual(record["payload"][key].device.type, "cpu")
                self.assertIn("contact", record["payload"]["metadata"])
            for name in (
                "report.json",
                "updates.csv",
                "epochs.csv",
                "loss_curve.svg",
                "residual_curve.svg",
                "penetration_curve.svg",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertNotIn("shortened_query_count", (output / "updates.csv").read_text())
            header = (output / "updates.csv").read_text().splitlines()[0]
            self.assertIn("contact_max_penetration_r,contact_pair_mean", header)
            self.assertIn(
                "contact_scene_fraction,contact_realized_fraction,contact_max_penetration_r,"
                "validation_final_max_penetration_r",
                (output / "epochs.csv").read_text().splitlines()[0],
            )

    def test_contact_scenes_follow_the_seeded_generator_and_the_sampled_material(self):
        """Register each trajectory's scene from [master_seed, seed, 2203] with ke scaled by its own Young's modulus."""
        config = self.config(1)
        step, rest = self._step(config)
        training = _TrajectoryFactory(step, rest, config, rank=0)
        validation = _TrajectoryFactory(step, rest, config, rank=0, validation=True)
        for factory in (training, validation):
            with self.subTest(validation=factory is validation):
                payload = factory.reset(5)
                material = payload["context_spec"]
                youngs = (
                    material["lame_mu"]
                    * (3 * material["lame_lambda"] + 2 * material["lame_mu"])
                    / (material["lame_lambda"] + material["lame_mu"])
                )
                expected = sample_contact_partners(
                    rest,
                    master_seed=factory.master_seed,
                    seed=5,
                    youngs_modulus=youngs,
                    cell_size=config.cell_size,
                    time_step=config.time_step,
                    plane_probability=config.contact_plane_probability,
                    plane_height_range=config.contact_plane_height_range,
                    max_points=config.contact_max_points,
                    point_radius_range=config.contact_point_radius_range,
                    kappa_range=config.contact_kappa_range,
                    beta_range=config.contact_beta_range,
                    mu_range=config.contact_mu_range,
                )
                self.assertEqual(payload["contact_partners"], expected.to_dict())
                scene = payload["metadata"]["contact"]
                self.assertEqual(set(scene), {"plane_present", "point_count", "ke", "kd", "mu", "kappa", "beta"})
                self.assertEqual(
                    (scene["plane_present"], scene["point_count"]), (expected.plane_present, expected.point_count)
                )
                self.assertEqual((scene["ke"], scene["kd"], scene["mu"]), (expected.ke, expected.kd, expected.mu))
                self.assertAlmostEqual(scene["kappa"], expected.ke / (youngs * config.cell_size), places=9)
                self.assertAlmostEqual(scene["beta"], expected.kd / (expected.ke * config.time_step), places=9)
                self.assertTrue(config.contact_kappa_range[0] <= scene["kappa"] <= config.contact_kappa_range[1])
                # The step's conditioning carries the same ratios for the registered context.
                batch = _batch([payload], torch.device("cpu"))
                inputs = step.prepare_inputs(
                    batch["candidate"],
                    batch["inertial_prediction"],
                    batch["context_ids"],
                    previous_positions=batch["physical_positions"],
                    contact=batch["contact"],
                )
                torch.testing.assert_close(
                    inputs.conditioning[0, 0, 6:],
                    torch.tensor([math.log1p(scene["kappa"]), scene["beta"], scene["mu"]], dtype=torch.float32),
                    rtol=1e-5,
                    atol=1e-6,
                )
                factory.retire(payload)
        self.assertNotEqual(training.reset(5)["contact_partners"], validation.reset(5)["contact_partners"])
        step.close()
        self.assertEqual(step.context_specs, {})

    def test_batch_collates_contact_pairs_padded_to_the_largest_count(self):
        """Zero-pad the per-payload pair tensors to the batch maximum Q with a validity mask; Q = 0 is allowed."""
        config = self.config(1, **_FLOOR_SCENE)
        step, rest = self._step(config)
        factory = _TrajectoryFactory(step, rest, config, rank=0)
        payloads = [factory.reset(seed) for seed in range(2)]
        self.addCleanup(lambda: [factory.retire(p) for p in payloads if p["context_id"] in step.context_specs])
        counts = [int(p["contact_sample_index"].shape[0]) for p in payloads]
        self.assertTrue(all(count >= 2 for count in counts), counts)
        # Drop one pair from the second payload so the two rows need different padding.
        for key in _CONTACT_PAYLOAD_KEYS:
            payloads[1][key] = payloads[1][key][:-1]
        counts[1] -= 1
        batch = _batch(payloads, torch.device("cpu"))
        contact = batch["contact"]
        width = max(counts)
        self.assertEqual(
            set(contact), {"sample_index", "kind", "partner_point", "partner_normal", "partner_radius", "mask"}
        )
        self.assertEqual(contact["sample_index"].shape, (2, width))
        self.assertEqual(contact["partner_point"].shape, (2, width, 3))
        self.assertEqual((contact["sample_index"].dtype, contact["kind"].dtype), (torch.int64, torch.int64))
        self.assertEqual(contact["partner_radius"].dtype, torch.float32)
        self.assertEqual(contact["mask"].dtype, torch.bool)
        self.assertEqual(contact["mask"].sum(1).tolist(), counts)
        for row, payload in enumerate(payloads):
            count = counts[row]
            torch.testing.assert_close(
                contact["sample_index"][row, :count], payload["contact_sample_index"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                contact["partner_point"][row, :count], payload["contact_partner_point"], rtol=0, atol=0
            )
            self.assertEqual(contact["partner_point"][row, count:].abs().sum().item(), 0.0)
        # Payloads without pair tensors give None; a mixture treats the missing ones as empty.
        plain = {k: v for k, v in payloads[0].items() if k not in _CONTACT_PAYLOAD_KEYS}
        self.assertIsNone(_batch([plain], torch.device("cpu"))["contact"])
        mixed = _batch([plain, payloads[1]], torch.device("cpu"))["contact"]
        self.assertEqual(mixed["mask"].sum(1).tolist(), [0, counts[1]])
        # Q = 0 everywhere yields empty [B, 0] entries, which the step treats as no contact.
        for payload in payloads:
            for key in _CONTACT_PAYLOAD_KEYS:
                payload[key] = payload[key][:0]
        empty = _batch(payloads, torch.device("cpu"))["contact"]
        self.assertEqual(empty["sample_index"].shape, (2, 0))
        self.assertEqual(empty["mask"].shape, (2, 0))
        terms = step.energy(
            batch["candidate"],
            batch["inertial_prediction"],
            batch["context_ids"],
            previous_positions=batch["physical_positions"],
            contact=empty,
        )
        torch.testing.assert_close(terms.contact, torch.zeros(2), rtol=0, atol=0)

    def test_contact_smoke_with_a_guaranteed_floor_records_pairs_penetration_and_scenes(self):
        """Run two tiny epochs on floor scenes: finite losses, pairs in every update, penetration metrics, full scene fraction."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run_training(output, self.config(2, **_FLOOR_SCENE))
            self.assertEqual(report["completed_epochs"], 2)
            self.assertEqual(len(report["updates"]), 8)
            for row in report["updates"]:
                self.assertTrue(math.isfinite(row["loss"]))
                self.assertTrue(math.isfinite(row["after_joule"]) and math.isfinite(row["before_joule"]))
                # Two exposed bottom faces per object touch the floor, so every object carries at least two pairs.
                self.assertGreaterEqual(row["contact_pair_mean"], 2.0)
                self.assertTrue(math.isfinite(row["contact_max_penetration_r"]))
            # 1-2 cm gap against r = 5 cm: the deepest penetration of the first update is 0.6-0.8 r.
            self.assertGreater(max(row["contact_max_penetration_r"] for row in report["updates"]), 0.3)
            for row in report["epochs"]:
                self.assertEqual(row["contact_scene_fraction"], 1.0)
                self.assertEqual(row["contact_realized_fraction"], 1.0)
                self.assertGreater(row["contact_max_penetration_r"], 0.0)
                validation = row["validation"]
                curve = validation["penetration"]
                self.assertEqual(len(curve), self.config(2).validation_iterations + 1)
                self.assertEqual(set(curve[0]), {"iteration", "mean", "max", "valid_count"})
                self.assertGreater(curve[0]["max"], 0.3)
                self.assertTrue(all(point["valid_count"] == validation["sample_count"] for point in curve))
                self.assertGreater(validation["physical_curves"][0]["max_penetration_r"]["mean"], 0.0)
                for sample in validation["samples"]:
                    self.assertGreater(sample["contact_energy_joule"][0], 0.0)
                    self.assertEqual(len(sample["max_penetration_r"]), len(sample["energies"]))
            full = report["epochs"][1]["full_horizon_validation"]
            self.assertGreater(full["final_max_penetration_r"]["max"], 0.0)
            self.assertIn("mean", full["final_max_penetration_r"])
            svg = (output / "penetration_curve.svg").read_text()
            self.assertIn('aria-label="max"', svg)
            self.assertNotIn("Waiting for completed measurements", svg)
            self.assertIn("Validation deepest contact penetration", (output / "index.html").read_text())
            saved = torch.load(output / "checkpoints/latest.pt", weights_only=False)
            for record in saved["rank_states"][0]["pool"]["records"]:
                payload = record["payload"]
                self.assertTrue(payload["contact_partners"]["plane_present"])
                self.assertTrue(payload["metadata"]["contact"]["plane_present"])
                self.assertGreaterEqual(payload["contact_sample_index"].shape[0], 2)
            self.assertTrue(any(key.startswith("contact_encoder.") for key in saved["network_state"]))
            epochs_csv = (output / "epochs.csv").read_text().splitlines()
            self.assertEqual(len(epochs_csv), 3)
            self.assertIn("1.0", epochs_csv[1])

    def test_realized_contact_fraction_separates_scene_presence_from_detected_pairs(self):
        """A floor far below every trajectory gives scene fraction 1 but realized contact 0 within the horizon."""
        config = self.config(
            1, contact_plane_probability=1.0, contact_plane_height_range=(-0.35, -0.34), contact_max_points=0
        )
        with tempfile.TemporaryDirectory() as directory:
            report = run_training(Path(directory), config)
            self.assertTrue(all(row["contact_pair_mean"] == 0.0 for row in report["updates"]))
            row = report["epochs"][0]
            self.assertEqual(row["contact_scene_fraction"], 1.0)
            self.assertEqual(row["contact_realized_fraction"], 0.0)
            self.assertEqual(row["contact_max_penetration_r"], 0.0)

    def test_contact_disabled_reproduces_the_schema_four_zero_path(self):
        """Without contact: contact-free partners, no pairs, zero contact channels, no encoder, zero metrics."""
        config = self.config(1, contact=False)
        step, rest = self._step(config)
        self.assertIsNone(step.network.contact_encoder)
        factory = _TrajectoryFactory(step, rest, config, rank=0)
        payload = factory.reset(3)
        self.assertEqual(payload["contact_partners"], ContactPartners.contact_free().to_dict())
        self.assertEqual(
            payload["metadata"]["contact"],
            {"plane_present": False, "point_count": 0, "ke": 0.0, "kd": 0.0, "mu": 0.0, "kappa": 0.0, "beta": 0.0},
        )
        self.assertEqual(payload["contact_sample_index"].shape, (0,))
        batch = _batch([payload], torch.device("cpu"))
        self.assertEqual(batch["contact"]["sample_index"].shape, (1, 0))
        inputs = step.prepare_inputs(
            batch["candidate"],
            batch["inertial_prediction"],
            batch["context_ids"],
            previous_positions=batch["physical_positions"],
            contact=batch["contact"],
        )
        self.assertEqual(inputs.conditioning.shape[-1], features.CONDITIONING_DIM)
        torch.testing.assert_close(
            inputs.conditioning[..., 6:], torch.zeros_like(inputs.conditioning[..., 6:]), rtol=0, atol=0
        )
        self.assertIsNone(inputs.contact_tokens)
        result = _checked_forward(step, step, batch)
        torch.testing.assert_close(result.contact_energy, torch.zeros(1), rtol=0, atol=0)
        torch.testing.assert_close(result.contact_max_penetration, torch.zeros(1), rtol=0, atol=0)
        factory.retire(payload)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run_training(output, config)
            self.assertTrue(all(row["contact_pair_mean"] == 0.0 for row in report["updates"]))
            self.assertTrue(all(row["contact_max_penetration_r"] == 0.0 for row in report["updates"]))
            row = report["epochs"][0]
            self.assertEqual((row["contact_scene_fraction"], row["contact_max_penetration_r"]), (0.0, 0.0))
            self.assertEqual(row["contact_realized_fraction"], 0.0)
            self.assertTrue(all(point["max"] == 0.0 for point in row["validation"]["penetration"]))
            saved = torch.load(output / "checkpoints/latest.pt", weights_only=False)
            self.assertFalse(any(key.startswith("contact_encoder.") for key in saved["network_state"]))
            for record in saved["rank_states"][0]["pool"]["records"]:
                self.assertEqual(record["payload"]["contact_partners"], ContactPartners.contact_free().to_dict())
                self.assertEqual(record["payload"]["contact_sample_index"].shape, (0,))

    def test_exact_resume_preserves_updates_and_active_trajectories(self):
        """A restart reproduces Adam and independently progressing pool members."""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "experiments.learned_intrinsic_solver.curriculum.MixedCurriculum.available_counts",
                new=property(lambda self: ((1, 2), (1, 2))),
            ),
        ):
            root = Path(directory)
            full = run_training(root / "full", self.config(2, **_FLOOR_SCENE))
            run_training(root / "split", self.config(1, **_FLOOR_SCENE))
            resumed = run_training(
                root / "split", self.config(2, **_FLOOR_SCENE), resume=root / "split/checkpoints/latest.pt"
            )
            a = torch.load(root / "full/checkpoints/latest.pt", weights_only=False)
            b = torch.load(root / "split/checkpoints/latest.pt", weights_only=False)
            for key in a["network_state"]:
                torch.testing.assert_close(a["network_state"][key], b["network_state"][key], rtol=0, atol=0)
            for identity, state in a["optimizer_state"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, b["optimizer_state"]["state"][identity][key], rtol=0, atol=0)
            self.assertEqual(a["curriculum_state"], b["curriculum_state"])
            self.assertEqual(a["controller_state"], b["controller_state"])
            self.assertEqual(full["updates"], resumed["updates"])
            self.assertTrue(all(row["contact_pair_mean"] > 0 for row in full["updates"]))
            self.assertEqual(full["best_selection"], resumed["best_selection"])
            self.assertEqual(full["completed_updates"], 8)
            self.assertEqual(full["epochs"][-1]["query_count"], 8)
            self.assertEqual(full["epochs"][-1]["validation"]["sample_count"], 2)
            self.assertEqual(len(full["epochs"][-1]["validation"]["relative_energy"]), 4)
            self.assertEqual(len(full["epochs"][-1]["validation"]["force_residual"]), 4)
            self.assertEqual(a["rank_states"][0]["pool"]["next_seed"], b["rank_states"][0]["pool"]["next_seed"])
            for record_a, record_b in zip(
                a["rank_states"][0]["pool"]["records"], b["rank_states"][0]["pool"]["records"], strict=True
            ):
                self.assertEqual(record_a["payload"]["history_valid"], record_b["payload"]["history_valid"])
                torch.testing.assert_close(
                    record_a["payload"]["history_axis_gradient_world"],
                    record_b["payload"]["history_axis_gradient_world"],
                    rtol=0,
                    atol=0,
                )
                # The contact scene and its frozen pair list survive the checkpoint round trip.
                self.assertEqual(record_a["payload"]["contact_partners"], record_b["payload"]["contact_partners"])
                self.assertEqual(record_a["payload"]["metadata"]["contact"], record_b["payload"]["metadata"]["contact"])
                self.assertTrue(record_a["payload"]["contact_partners"]["plane_present"])
                for key in _CONTACT_PAYLOAD_KEYS:
                    torch.testing.assert_close(record_a["payload"][key], record_b["payload"][key], rtol=0, atol=0)
            for name in ("report.json", "updates.csv", "epochs.csv", "loss_curve.svg", "index.html"):
                self.assertTrue((root / "split" / name).is_file(), name)

    def test_resume_rejects_material_or_architecture_change_before_writing(self):
        """A checkpoint cannot silently change its physical problem or model."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_training(output, self.config(1))
            checkpoint = output / "checkpoints/latest.pt"
            before = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "resume configuration"):
                run_training(output, replace(self.config(2), time_step=0.01), resume=checkpoint)
            with self.assertRaisesRegex(ValueError, "resume configuration"):
                run_training(output, replace(self.config(2), energy_floor_scale=2.0), resume=checkpoint)
            saved = torch.load(checkpoint, weights_only=False)
            saved["config"]["feature_schema_version"] = 2
            legacy = output / "checkpoints/legacy.pt"
            torch.save(saved, legacy)
            with self.assertRaisesRegex(ValueError, "legacy"):
                run_training(output, self.config(2), resume=legacy)
            for field in ("contact", "edge_network", "contact_plane_probability", "contact_tokens_per_cell"):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "resume configuration"):
                    changed = (
                        {field: not getattr(self.config(2), field)} if field in ("contact", "edge_network") else {}
                    )
                    if field == "contact_plane_probability":
                        changed = {field: 0.5}
                    elif field == "contact_tokens_per_cell":
                        changed = {field: 12}
                    run_training(output, replace(self.config(2), **changed), resume=checkpoint)
            self.assertEqual(before, checkpoint.read_bytes())

    def test_resume_requires_the_stored_contact_scene_of_every_context(self):
        """A checkpoint whose pool payload lost its contact partners is rejected instead of silently going contact-free."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_training(root / "base", self.config(1))
            saved = torch.load(root / "base/checkpoints/latest.pt", weights_only=False)
            del saved["rank_states"][0]["pool"]["records"][0]["payload"]["contact_partners"]
            broken = root / "broken.pt"
            torch.save(saved, broken)
            with self.assertRaisesRegex(ValueError, "no stored contact partners"):
                run_training(root / "resumed", self.config(2), resume=broken)

    def test_resume_updates_only_descent_gate_and_records_effective_boundary(self):
        """Change the gate without rewriting prior decisions or resetting training state."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original_config = replace(self.config(1), stage_descent_rate=0.9)
            run_training(output, original_config)
            checkpoint = output / "checkpoints/latest.pt"
            saved = torch.load(checkpoint, weights_only=False)
            # The memory knob may change on resume: it does not alter outputs or gradients.
            resumed = run_training(
                output, replace(original_config, stage_descent_rate=0.8, checkpoint_chunks=True), resume=checkpoint
            )
            restored = torch.load(output / "checkpoints/final.pt", weights_only=False)

            def assert_same(expected, actual):
                if isinstance(expected, torch.Tensor):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                elif isinstance(expected, np.ndarray):
                    np.testing.assert_array_equal(expected, actual)
                elif isinstance(expected, dict):
                    self.assertEqual(expected.keys(), actual.keys())
                    for name, value in expected.items():
                        assert_same(value, actual[name])
                elif isinstance(expected, (tuple, list)):
                    self.assertEqual(len(expected), len(actual))
                    for left, right in zip(expected, actual, strict=True):
                        assert_same(left, right)
                else:
                    self.assertEqual(expected, actual)

            for name in ("network_state", "optimizer_state", "controller_state", "rank_states"):
                assert_same(saved[name], restored[name])
            self.assertEqual(restored["curriculum_state"], {**saved["curriculum_state"], "min_descent_rate": 0.8})
            self.assertEqual(resumed["epochs"], saved["report"]["epochs"])
            self.assertEqual(resumed["updates"], saved["report"]["updates"])
            self.assertEqual(
                resumed["configuration_changes"],
                [
                    {
                        "field": "stage_descent_rate",
                        "previous": 0.9,
                        "current": 0.8,
                        "effective_from_epoch": 2,
                        "completed_updates": saved["report"]["completed_updates"],
                        "source": "checkpoint_resume",
                    }
                ],
            )
            self.assertEqual(restored["config"]["stage_descent_rate"], 0.8)

    def test_resume_applies_overdue_cap_without_new_validation(self):
        """Promote once at resume while leaving checkpoint history and active budgets intact."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = replace(self.config(1), stage_max_epochs=None, stage_descent_rate=1.0)
            run_training(output, config)
            checkpoint = output / "checkpoints/latest.pt"
            saved = torch.load(checkpoint, weights_only=False)
            self.assertEqual(saved["curriculum_state"]["stage"], 0)
            report = run_training(output, replace(config, stage_max_epochs=1), resume=checkpoint)
            restored = torch.load(output / "checkpoints/final.pt", weights_only=False)
            self.assertEqual(restored["curriculum_state"]["stage"], 1)
            self.assertEqual(restored["curriculum_state"]["stage_epochs"], 0)
            self.assertEqual(report["epochs"], saved["report"]["epochs"])
            self.assertEqual(report["updates"], saved["report"]["updates"])
            self.assertEqual(report["configuration_changes"][0]["field"], "stage_max_epochs")
            self.assertEqual(
                report["curriculum_events"],
                [
                    {
                        "source": "checkpoint_resume",
                        "advance_reason": "max_stage_epochs",
                        "previous_stage": 0,
                        "stage": 1,
                        "previous_stage_epochs": 1,
                        "effective_from_epoch": 2,
                        "completed_updates": saved["report"]["completed_updates"],
                    }
                ],
            )
            original_pool = saved["rank_states"][0]["pool"]
            resumed_pool = restored["rank_states"][0]["pool"]
            self.assertEqual(original_pool["iteration_counts"], resumed_pool["iteration_counts"])
            self.assertEqual(original_pool["physical_step_counts"], resumed_pool["physical_step_counts"])

    def test_later_inner_iterations_normalize_by_the_carried_candidate_energy(self):
        """E_before at inner iteration 2 is the previous update's E_after, not the solve's initial energy."""
        with (
            tempfile.TemporaryDirectory() as directory,
            # Stage 0 caps K at 1, so force K=2/H=1 dispatch to reach a second inner iteration.
            patch(
                "experiments.learned_intrinsic_solver.curriculum.MixedCurriculum.available_counts",
                new=property(lambda self: ((2,), (1,))),
            ),
        ):
            report = run_training(Path(directory), self.config(1))
        updates = report["updates"]
        self.assertEqual(len(updates), 4)
        # FIFO pool of four with batches of two: rows 0/1 are iteration 1 of members A,B and C,D;
        # rows 2/3 are their second iterations, starting from the carried candidates.
        self.assertEqual(updates[2]["before_joule"], updates[0]["after_joule"])
        self.assertEqual(updates[3]["before_joule"], updates[1]["after_joule"])
        # The zero-initialised heads make update 0 an energy no-op, so only the second pair can
        # distinguish per-update normalization from the forbidden initial-energy denominator.
        self.assertNotEqual(updates[1]["after_joule"], updates[1]["before_joule"])
        self.assertNotEqual(updates[3]["before_joule"], updates[1]["before_joule"])

    def _run_with_scripted_full_horizon(self, config, adjust_full):
        """Run with an always-qualifying cheap validation and a wrapped full-horizon check."""
        real_validate = mixed_validation.validate
        real_full = mixed_validation.validate_full_horizon
        full_calls = []

        def qualifying_validate(*args, **kwargs):
            # Wrap the real summary: the controller and report writers read its other keys.
            summary = real_validate(*args, **kwargs)
            return {
                **summary,
                "failed_count": 0,
                "physical_survivors": summary["sample_count"],
                "mean_before_joule": 10.0,
                "mean_after_joule": 9.0,
                "descent_rate": 1.0,
            }

        def wrapped_full(*args, **kwargs):
            summary = real_full(*args, **kwargs)
            full_calls.append(kwargs)
            return {**summary, **adjust_full(summary)}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(train_mixed, "_validate", qualifying_validate),
            # run_training imports validate_full_horizon locally, so patch its source module.
            patch.object(mixed_validation, "validate_full_horizon", wrapped_full),
        ):
            report = run_training(Path(directory), config)
        return report, full_calls

    def test_full_horizon_check_runs_before_validation_gated_advancement(self):
        """The full check runs whenever the curriculum could advance, not only on interval epochs."""
        config = self.config(2, validation_full_interval=50)
        report, calls = self._run_with_scripted_full_horizon(
            config, lambda summary: {"failed_count": 0, "physical_survivors": summary["sample_count"]}
        )
        # Epoch 1 cannot advance (patience 2); epoch 2 can, so the check runs there despite the interval.
        self.assertEqual(calls, [{"iterations": 1, "physical_steps": 2}])
        self.assertIsNone(report["epochs"][0]["full_horizon_validation"])
        self.assertIsNotNone(report["epochs"][1]["full_horizon_validation"])
        self.assertFalse(report["epochs"][0]["curriculum"]["advanced"])
        decision = report["epochs"][1]["curriculum"]
        self.assertEqual(
            (decision["advanced"], decision["advance_reason"], decision["stage"], decision["full_horizon_qualified"]),
            (True, "validation", 1, True),
        )

    def test_failed_full_horizon_check_blocks_validation_gated_advancement(self):
        """A full check with a dead trajectory vetoes the otherwise satisfied validation gate."""
        config = self.config(2, validation_full_interval=50)
        report, calls = self._run_with_scripted_full_horizon(
            config, lambda summary: {"failed_count": 1, "physical_survivors": 0}
        )
        self.assertEqual(len(calls), 1)
        decision = report["epochs"][1]["curriculum"]
        self.assertFalse(decision["advanced"])
        self.assertIs(decision["full_horizon_qualified"], False)
        self.assertEqual((decision["stage"], decision["qualified_epochs"]), (0, 0))

    def test_full_horizon_check_runs_before_the_stage_cap_advances(self):
        """Reaching the hard residence limit also triggers the check on a non-interval epoch."""
        with tempfile.TemporaryDirectory() as directory:
            report = run_training(Path(directory), self.config(1, validation_full_interval=50, stage_max_epochs=1))
        row = report["epochs"][0]
        self.assertIsNotNone(row["full_horizon_validation"])
        self.assertEqual(row["curriculum"]["advance_reason"], "max_stage_epochs")

    def test_best_checkpoint_requires_eligibility_and_strict_improvement(self):
        """Ineligible epochs never become best; equal metrics do not overwrite the best checkpoint."""
        script = [(True, 1.0), (False, 0.1), (True, 0.5), (True, 0.5)]
        real_validate = train_mixed._validate
        calls = []

        def scripted(*args, **kwargs):
            summary = real_validate(*args, **kwargs)
            eligible, metric = script[len(calls)]
            calls.append(metric)
            summary["selection"] = dict(summary["selection"], metric=metric, eligible=eligible)
            if not eligible:
                summary["physical_survivors"] = summary["sample_count"] - 1
            return summary

        with tempfile.TemporaryDirectory() as directory, patch.object(train_mixed, "_validate", scripted):
            output = Path(directory)
            report = run_training(output, self.config(4))
            self.assertEqual(calls, [1.0, 0.1, 0.5, 0.5])
            selections = [row["validation"]["selection"] for row in report["epochs"]]
            self.assertEqual([(s["eligible"], s["metric"]) for s in selections], script)
            # Epoch 2 has the smallest metric but a dead trajectory; epoch 4 ties epoch 3 and must not replace it.
            self.assertEqual((report["best_selection"]["epoch"], report["best_selection"]["metric"]), (3, 0.5))
            saved = torch.load(output / "checkpoints/best_validation.pt", weights_only=False)
            self.assertEqual(saved["report"]["completed_epochs"], 3)
            self.assertEqual(saved["report"]["best_selection"]["epoch"], 3)

    def test_validation_interval_skips_epochs_and_writes_reports_with_gaps(self):
        """Validate on interval epochs and the final epoch only; skipped rows carry None and never become best."""
        script = [(True, 1.0), (True, 0.5)]
        real_validate = train_mixed._validate
        real_full = mixed_validation.validate_full_horizon
        validated, full_calls = [], []

        def scripted(*args, **kwargs):
            summary = real_validate(*args, **kwargs)
            eligible, metric = script[len(validated)]
            validated.append(metric)
            summary["selection"] = dict(summary["selection"], metric=metric, eligible=eligible)
            return summary

        def counted_full(*args, **kwargs):
            full_calls.append(kwargs)
            return real_full(*args, **kwargs)

        config = self.config(3, validation_interval=2, validation_full_interval=2)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(train_mixed, "_validate", scripted),
            patch.object(mixed_validation, "validate_full_horizon", counted_full),
        ):
            output = Path(directory)
            report = run_training(output, config)
            # Epoch 1 skips validation entirely; epoch 2 is an interval epoch and epoch 3 the final one.
            self.assertEqual(validated, [1.0, 0.5])
            rows = report["epochs"]
            self.assertEqual([row["epoch"] for row in rows], [1, 2, 3])
            self.assertIsNone(rows[0]["validation"])
            self.assertIsNone(rows[0]["full_horizon_validation"])
            self.assertIsInstance(rows[1]["validation"], dict)
            self.assertIsInstance(rows[2]["validation"], dict)
            # Full-horizon checks happen on validated epochs only; the interval-2 check runs at epoch 2.
            self.assertIsNotNone(rows[1]["full_horizon_validation"])
            self.assertEqual(len(full_calls), sum(row["full_horizon_validation"] is not None for row in rows))
            # The skipped epoch counts toward stage residence without qualifying or advancing.
            self.assertEqual(
                rows[0]["curriculum"],
                {
                    "stage": 0,
                    "advanced": False,
                    "advance_reason": None,
                    "qualified": False,
                    "full_horizon_qualified": None,
                    "stage_epochs": 1,
                    "qualified_epochs": 0,
                },
            )
            self.assertEqual(rows[1]["curriculum"]["stage_epochs"], 2)
            self.assertFalse(rows[0]["allow_early_stop"])
            # The cosine schedule is by epoch and unchanged by skipped validation.
            self.assertEqual(
                [row["learning_rate"] for row in rows],
                [_scheduled_learning_rate(config, epoch, 0.0) for epoch in (1, 2, 3)],
            )
            self.assertEqual((report["best_selection"]["epoch"], report["best_selection"]["metric"]), (3, 0.5))
            self.assertEqual(report["status"], "epoch_limit")
            self.assertTrue((output / "checkpoints/best_validation.pt").is_file())
            best = torch.load(output / "checkpoints/best_validation.pt", weights_only=False)
            self.assertEqual(best["report"]["completed_epochs"], 3)
            # The controller saw two validations at training epochs 2 and 3, both improvements.
            controller = torch.load(output / "checkpoints/latest.pt", weights_only=False)["controller_state"]
            self.assertEqual((controller["last_epoch"], controller["best_loss"], controller["bad_epochs"]), (3, 0.5, 0))
            with (output / "epochs.csv").open() as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual([r["selection_metric"] for r in csv_rows], ["", "1.0", "0.5"])
            self.assertEqual([r["sample_count"] for r in csv_rows], ["", "2", "2"])
            self.assertEqual(csv_rows[0]["query_count"], "8")
            page = (output / "index.html").read_text()
            self.assertIn("Validation runs every 2 epochs and on the final epoch.", page)
            for name in ("report.json", "loss_curve.svg", "validation_curve.svg", "residual_curve.svg"):
                self.assertTrue((output / name).is_file(), name)
            self.assertIsNone(json.loads((output / "report.json").read_text())["epochs"][0]["validation"])

    def test_skipped_epochs_count_stage_residence_and_defer_the_cap_to_a_validated_epoch(self):
        """Residence grows on skipped epochs; a cap reached there promotes at the next validated epoch after the full check."""
        real_validate = train_mixed._validate

        def stalled(*args, **kwargs):
            # Never qualifies, so only the hard residence limit can advance the stage.
            return {**real_validate(*args, **kwargs), "descent_rate": 0.0}

        config = self.config(4, validation_interval=2, validation_full_interval=50, stage_max_epochs=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(train_mixed, "_validate", stalled):
            report = run_training(Path(directory), config)
        rows = report["epochs"]
        decisions = [row["curriculum"] for row in rows]
        # Epochs 1 and 3 reach the one-epoch cap without validation and wait; epochs 2 and 4 run the
        # full-horizon check first and then promote by the cap.
        self.assertEqual([d["stage_epochs"] for d in decisions], [1, 0, 1, 0])
        self.assertEqual([d["advanced"] for d in decisions], [False, True, False, True])
        self.assertEqual([d["advance_reason"] for d in decisions], [None, "max_stage_epochs", None, "max_stage_epochs"])
        self.assertEqual([d["stage"] for d in decisions], [0, 1, 1, 2])
        self.assertEqual([row["full_horizon_validation"] is not None for row in rows], [False, True, False, True])
        self.assertEqual([list(row["available_K"]) for row in rows], [[1], [1], [1, 2], [1, 2]])
        self.assertEqual([row["validation"] is None for row in rows], [True, False, True, False])

    def test_resume_after_a_skipped_epoch_keeps_an_unchanged_cap_for_the_next_validated_epoch(self):
        """Resuming from a skipped epoch that reached the unchanged cap reproduces the uninterrupted run exactly."""
        config = self.config(
            2, validation_interval=2, checkpoint_interval=1, stage_max_epochs=1, validation_full_interval=50
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            straight = run_training(root / "straight", config)
            checkpoint = root / "straight/checkpoints/epoch_0001.pt"
            saved = torch.load(checkpoint, weights_only=False)
            # Epoch 1 skipped validation and reached the one-epoch cap, so the checkpoint holds an overdue stage.
            self.assertIsNone(saved["report"]["epochs"][0]["validation"])
            self.assertEqual((saved["curriculum_state"]["stage"], saved["curriculum_state"]["stage_epochs"]), (0, 1))
            resumed = run_training(root / "resumed", config, resume=checkpoint)
            # The unchanged cap promotes at epoch 2 after the full-horizon check, never at resume.
            self.assertNotIn("curriculum_events", resumed)
            self.assertEqual(
                [row["curriculum"] for row in resumed["epochs"]], [row["curriculum"] for row in straight["epochs"]]
            )
            self.assertEqual([list(row["available_K"]) for row in resumed["epochs"]], [[1], [1]])
            self.assertEqual(
                (resumed["epochs"][1]["curriculum"]["advanced"], resumed["epochs"][1]["curriculum"]["advance_reason"]),
                (True, "max_stage_epochs"),
            )
            self.assertIsNotNone(resumed["epochs"][1]["full_horizon_validation"])
            self.assertEqual(resumed["updates"], straight["updates"])
            self.assertEqual(resumed["best_selection"], straight["best_selection"])
            latest = torch.load(root / "resumed/checkpoints/latest.pt", weights_only=False)
            reference = torch.load(root / "straight/checkpoints/latest.pt", weights_only=False)
            for name in ("curriculum_state", "controller_state"):
                self.assertEqual(latest[name], reference[name], name)

    def test_resume_accepts_a_changed_validation_budget_and_resets_the_selection_record(self):
        """Every validation setting may change on resume; an incomparable best record and the plateau patience restart."""
        real_validate = train_mixed._validate
        metric = [0.5]

        def eligible(*args, **kwargs):
            summary = real_validate(*args, **kwargs)
            summary["selection"] = dict(summary["selection"], metric=metric[0], eligible=True)
            return summary

        changes = {
            "validation_count": 3,
            "validation_iterations": 2,
            "validation_interval": 2,
            "validation_physical_steps": 1,
            "validation_physical_iterations": 1,
            "validation_full_count": 2,
            "validation_full_interval": 3,
            "validation_full_iterations": 1,
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(train_mixed, "_validate", eligible):
            output = Path(directory)
            base = run_training(output, self.config(1))
            checkpoint = output / "checkpoints/latest.pt"
            self.assertEqual(base["best_selection"]["epoch"], 1)
            self.assertEqual(torch.load(checkpoint, weights_only=False)["controller_state"]["best_loss"], 0.5)
            with self.assertRaisesRegex(ValueError, "resume configuration"):
                run_training(output, self.config(2, learning_rate=2e-4), resume=checkpoint)
            # The cheaper budget measures a larger residual that must not be judged against the old best.
            metric[0] = 1.0
            resumed = run_training(output, self.config(3, **changes), resume=checkpoint)
            self.assertEqual(resumed["completed_epochs"], 3)
            self.assertEqual(
                resumed["best_selection_history"],
                [{"reset_at_epoch": 1, "reason": "validation budget changed", "record": base["best_selection"]}],
            )
            # Epoch 2 is the first validated epoch under the new budget and becomes the new best; the
            # equal metric of epoch 3 does not replace it.
            self.assertEqual(resumed["best_selection"]["epoch"], 2)
            self.assertEqual(resumed["epochs"][1]["validation"]["sample_count"], 3)
            self.assertEqual(len(resumed["epochs"][1]["validation"]["force_residual"]), 3)
            best = torch.load(output / "checkpoints/best_validation.pt", weights_only=False)
            self.assertEqual(best["report"]["completed_epochs"], 2)
            self.assertEqual(best["config"]["validation_iterations"], 2)
            # The controller forgot the old best too: epoch 2 improved from nothing and only epoch 3's
            # equal metric counts against patience.
            controller = torch.load(output / "checkpoints/latest.pt", weights_only=False)["controller_state"]
            self.assertEqual(
                (
                    controller["best_loss"],
                    controller["bad_epochs"],
                    controller["lr_bad_epochs"],
                    controller["last_epoch"],
                ),
                (1.0, 1, 1, 3),
            )
            self.assertEqual(len(controller["recent_good"]), 2)
            full = resumed["epochs"][2]["full_horizon_validation"]
            self.assertEqual((full["iterations"], full["sample_count"]), (1, 2))
            recorded = {change["field"]: change for change in resumed["configuration_changes"]}
            self.assertEqual(set(recorded), set(changes))
            for field, current in changes.items():
                self.assertEqual(
                    (recorded[field]["previous"], recorded[field]["current"], recorded[field]["effective_from_epoch"]),
                    (getattr(self.config(1), field), current, 2),
                )
            # An unchanged budget keeps the record, the controller's patience and writes no history.
            again = run_training(output, self.config(4, **changes), resume=output / "checkpoints/latest.pt")
            self.assertEqual(again["best_selection"], resumed["best_selection"])
            self.assertEqual(len(again["best_selection_history"]), 1)
            controller = torch.load(output / "checkpoints/latest.pt", weights_only=False)["controller_state"]
            self.assertEqual((controller["best_loss"], controller["bad_epochs"]), (1.0, 2))

    def test_full_horizon_iterations_cap_limits_k_below_the_largest_available(self):
        """validation_full_iterations caps the full check's K and the summary records the capped value."""
        config = self.config(1, validation_full_interval=1)
        self.assertEqual(_full_horizon_iterations(config, (1, 2, 4)), 4)
        self.assertEqual(_full_horizon_iterations(replace(config, validation_full_iterations=2), (1, 2, 4)), 2)
        self.assertEqual(_full_horizon_iterations(replace(config, validation_full_iterations=8), (1, 2, 4)), 4)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "experiments.learned_intrinsic_solver.curriculum.MixedCurriculum.available_counts",
                new=property(lambda self: ((1, 2), (1, 2))),
            ),
        ):
            root = Path(directory)
            capped = run_training(root / "capped", replace(config, validation_full_iterations=1))
            uncapped = run_training(root / "uncapped", config)
        for report, expected in ((capped, 1), (uncapped, 2)):
            with self.subTest(expected=expected):
                row = report["epochs"][0]
                full = row["full_horizon_validation"]
                self.assertEqual(max(row["available_K"]), 2)
                self.assertEqual((full["iterations"], full["physical_steps"]), (expected, 2))

    @staticmethod
    def _curriculum_at(stage, stage_epochs):
        """Return a curriculum restored to the given progress under a 20-epoch stage cap."""
        curriculum = MixedCurriculum(min_stage_epochs=1, max_stage_epochs=20)
        curriculum.load_state_dict(
            {**curriculum.state_dict(), "stage": stage, "stage_epochs": stage_epochs, "qualified_epochs": 0}
        )
        return curriculum

    def test_allow_early_stop_requires_final_stage_residence_after_observation(self):
        """Plateau stopping is considered only after plateau_min_final_stage_epochs in the final stage."""
        config = self.config(1, early_stopping=True)
        self.assertEqual(config.plateau_min_final_stage_epochs, 20)
        final = MixedCurriculum().final_stage
        for stage, stage_epochs, early_stopping, expected in (
            (final - 1, 25, True, False),
            (final, 19, True, False),
            (final, 20, True, True),
            (final, 25, True, True),
            (final, 20, False, False),
        ):
            with self.subTest(stage=stage, stage_epochs=stage_epochs, early_stopping=early_stopping):
                curriculum = self._curriculum_at(stage, stage_epochs)
                self.assertIs(_allow_early_stop(replace(config, early_stopping=early_stopping), curriculum), expected)
        # A stage entered this epoch counts zero: the cap promotes 4/19 to 5/0 on observe.
        curriculum = self._curriculum_at(final - 1, 19)
        self.assertFalse(_allow_early_stop(config, curriculum))
        decision = curriculum.observe({})
        self.assertEqual(
            (decision["stage"], decision["advance_reason"], curriculum.stage_epochs), (final, "max_stage_epochs", 0)
        )
        self.assertFalse(_allow_early_stop(config, curriculum))
        self.assertTrue(_allow_early_stop(replace(config, plateau_min_final_stage_epochs=0), curriculum))

    def test_plateau_stopping_is_allowed_only_after_final_stage_residence(self):
        """A resumed run passes allow_early_stop=True to the controller exactly when the gate is met."""
        real_observe = PlateauController.observe
        seen = []

        def observe(self, epoch, validation, *, allow_early_stop=True):
            seen.append(allow_early_stop)
            return real_observe(self, epoch, validation, allow_early_stop=allow_early_stop)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_training(root / "base", self.config(1))
            saved = torch.load(root / "base/checkpoints/latest.pt", weights_only=False)
            self.assertFalse(saved["report"]["epochs"][0]["allow_early_stop"])
            final = MixedCurriculum().final_stage
            for stage_epochs, expected in ((19, True), (18, False)):
                with self.subTest(stage_epochs=stage_epochs):
                    saved["curriculum_state"].update(stage=final, stage_epochs=stage_epochs, qualified_epochs=0)
                    checkpoint = root / f"final_stage_{stage_epochs}.pt"
                    torch.save(saved, checkpoint)
                    seen.clear()
                    with patch.object(PlateauController, "observe", observe):
                        report = run_training(
                            root / f"resumed_{stage_epochs}", self.config(2, early_stopping=True), resume=checkpoint
                        )
                    # The gate is evaluated after observe: 19 -> 20 final-stage epochs allows, 18 -> 19 does not.
                    self.assertEqual(seen, [expected])
                    self.assertEqual([row["allow_early_stop"] for row in report["epochs"]], [False, expected])
                    self.assertEqual(
                        (report["epochs"][1]["curriculum"]["stage"], report["epochs"][1]["curriculum"]["stage_epochs"]),
                        (final, stage_epochs + 1),
                    )


class TestLearningRateSchedule(unittest.TestCase):
    def test_cosine_constant_and_plateau_rates(self):
        """Cosine runs from the initial rate to lr_final at the cap; constant and plateau pass through."""
        cosine = MixedTrainConfig(learning_rate=1e-4, lr_final=1e-6, max_epochs=100, lr_schedule="cosine")
        self.assertAlmostEqual(_scheduled_learning_rate(cosine, 0, 5e-5), 1e-4)
        self.assertAlmostEqual(_scheduled_learning_rate(cosine, 50, 5e-5), 1e-6 + (1e-4 - 1e-6) / 2)
        self.assertAlmostEqual(_scheduled_learning_rate(cosine, 100, 5e-5), 1e-6)
        self.assertAlmostEqual(_scheduled_learning_rate(cosine, 250, 5e-5), 1e-6)
        rates = [_scheduled_learning_rate(cosine, epoch, 0.0) for epoch in range(101)]
        self.assertTrue(all(a >= b for a, b in itertools.pairwise(rates)))
        constant = MixedTrainConfig(learning_rate=1e-4, lr_schedule="constant")
        self.assertEqual(_scheduled_learning_rate(constant, 37, 5e-5), 1e-4)
        plateau = MixedTrainConfig(learning_rate=1e-4, lr_schedule="plateau")
        self.assertEqual(_scheduled_learning_rate(plateau, 37, 5e-5), 5e-5)
        with self.assertRaises(ValueError):
            MixedTrainConfig(lr_schedule="linear")
        with self.assertRaises(ValueError):
            MixedTrainConfig(learning_rate=1e-4, lr_final=1e-3)
        saved = {k: v for k, v in asdict(MixedTrainConfig()).items() if k not in ("lr_schedule", "lr_final")}
        legacy = MixedTrainConfig.from_checkpoint_config(saved)
        self.assertEqual(legacy.lr_schedule, "plateau")
        self.assertTrue(math.isfinite(legacy.lr_final))

    def test_weight_decay_and_clipping_defaults_and_legacy_values(self):
        """AdamW decay and clipping follow LeCO by default; older checkpoints rebuild plain Adam without clipping."""
        config = MixedTrainConfig()
        self.assertEqual((config.weight_decay, config.gradient_clip_norm, config.lr_final), (1e-6, 1.0, 2.5e-5))
        self.assertIsNone(MixedTrainConfig(gradient_clip_norm=None).gradient_clip_norm)
        for bad in ({"weight_decay": -1e-6}, {"gradient_clip_norm": 0.0}, {"gradient_clip_norm": math.inf}):
            with self.assertRaises(ValueError):
                MixedTrainConfig(**bad)
        saved = {k: v for k, v in asdict(MixedTrainConfig()).items() if k not in ("weight_decay", "gradient_clip_norm")}
        legacy = MixedTrainConfig.from_checkpoint_config(saved)
        self.assertEqual((legacy.weight_decay, legacy.gradient_clip_norm), (0.0, None))


if __name__ == "__main__":
    unittest.main()
