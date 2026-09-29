# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check normalization and complete-population rollout statistics, and the mixed rollout's contact records."""

import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features, history, train_mixed
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.material_sampling import MaterialSample
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.render_learned import _CONTACT_KEYS, _load_trajectory
from experiments.learned_intrinsic_solver.rollout_report import build_report, summarize_rollout
from experiments.learned_intrinsic_solver.simulate_mixed import _contact_batch, _run_query, _Trajectory, run_rollouts
from experiments.learned_intrinsic_solver.train_mixed import MixedTrainConfig

_TINY = {
    "cell_counts": (1, 1, 2),
    "cell_size": 0.1,
    "hidden_dim": 8,
    "edge_hidden_dim": 4,
    "num_heads": 2,
    "batch_size": 2,
    "pool_multiplier": 2,
    "queries_per_epoch": 8,
    "max_epochs": 1,
    "stage_epochs": 1,
    "iteration_counts": (1, 2),
    "physical_step_counts": (1, 2),
    "validation_count": 2,
    "validation_iterations": 3,
    "validation_physical_steps": 2,
    "validation_physical_iterations": 2,
    "validation_full_count": 1,
    "validation_full_interval": 2,
    "target_modes": 7,
    "device": "cpu",
    "cpu_threads": 1,
    "preparation_workers": 1,
    "verbose": False,
    "early_stopping": False,
}
"""Two-cell seven-mode CPU configuration shared by the rollout tests."""


def _network(config: MixedTrainConfig) -> IntrinsicSolverNetwork:
    return IntrinsicSolverNetwork(
        config.cell_counts,
        config.state_feature_dim,
        target_modes=config.target_modes,
        conditioning_dim=config.conditioning_dim,
        hidden_dim=config.hidden_dim,
        edge_hidden_dim=config.edge_hidden_dim,
        num_heads=config.num_heads,
        hops=config.hops,
        max_step_size=config.max_step_size,
        query_chunk_size=config.query_chunk_size,
        edge_network=config.edge_network,
        contact_tokens=config.contact,
    )


def _youngs_modulus(material: dict) -> float:
    """Return E [Pa] of a rollout report's ``material`` block (the context's Lamé parameters and density)."""
    return MaterialSample(material["lame_lambda"], material["lame_mu"], material["density"]).youngs_modulus


class TestRolloutReport(unittest.TestCase):
    def test_normalize_each_case_before_aggregating(self):
        rows = summarize_rollout(np.array([[2.0, 100.0, 4.0], [1.0, 10.0, 8.0]]))
        self.assertEqual(rows[0]["mean"], 1.0)
        self.assertEqual(rows[1]["max"], 2.0)
        self.assertEqual(rows[1]["median"], 0.5)
        self.assertAlmostEqual(rows[1]["mean"], (0.5 + 0.1 + 2.0) / 3)

    def test_invalid_case_is_not_silently_omitted_from_full_statistics(self):
        rows = summarize_rollout(np.array([[2.0, 100.0, 4.0], [1.0, np.nan, 8.0]]))
        self.assertEqual(rows[1]["failed_count"], 1)
        self.assertEqual(rows[1]["valid_count"], 2)
        self.assertIsNone(rows[1]["mean"])
        self.assertIsNone(rows[1]["median"])
        self.assertIsNone(rows[1]["max"])
        self.assertAlmostEqual(rows[1]["valid_only_mean"], 1.25)

    def test_zero_initial_energy_cannot_be_normalized(self):
        with self.assertRaisesRegex(ValueError, "initial"):
            summarize_rollout(np.array([[0.0, 1.0], [0.0, 0.5]]))

    def test_first_iteration_failure_still_produces_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            energies = np.array([[2.0, 4.0], [np.nan, 2.0]])
            np.savez(
                output / "rank_0.npz",
                physical_seeds=[10000, 10001],
                energies=energies,
                relative_energies=energies / energies[0],
            )
            (output / "rank_0.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": "test-checkpoint",
                        "iterations": 1,
                        "checkpoint_epoch": 198,
                        "parameter_state_unchanged": True,
                        "failures": [{"physical_seed": 10000, "iteration": 1}],
                    }
                )
            )
            report = build_report(output, world_size=1)
            self.assertIsNone(report["step1_normalized_change_for_training_comparison"])
            self.assertEqual(report["rows"][1]["failed_count"], 1)
            self.assertTrue((output / "index.html").is_file())


class TestMixedRolloutContact(unittest.TestCase):
    """Contact plumbing of ``simulate_mixed``: pair collation, saved partners and the report block."""

    def test_contact_batch_pads_pairs_to_the_batch_maximum(self):
        """Collate payloads with different pair counts into masked [B, Q] tensors; Q = 0 stays valid."""
        rich = {
            "contact_sample_index": torch.tensor([3, 7]),
            "contact_kind": torch.tensor([0, 1]),
            "contact_partner_point": torch.tensor([[0.0, -0.1, 0.0], [0.2, 0.0, 0.1]]),
            "contact_partner_normal": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            "contact_partner_radius": torch.tensor([1e9, 0.05]),
        }
        empty = {key: value[:0] for key, value in rich.items()}
        batch = _contact_batch([empty, rich], torch.device("cpu"))
        self.assertEqual(
            {key: tuple(value.shape) for key, value in batch.items()},
            {
                "sample_index": (2, 2),
                "kind": (2, 2),
                "partner_point": (2, 2, 3),
                "partner_normal": (2, 2, 3),
                "partner_radius": (2, 2),
                "mask": (2, 2),
            },
        )
        self.assertEqual(batch["mask"].tolist(), [[False, False], [True, True]])
        self.assertEqual(batch["sample_index"].tolist(), [[0, 0], [3, 7]])
        self.assertEqual(batch["kind"].dtype, torch.int64)
        self.assertEqual(batch["partner_radius"].dtype, torch.float32)
        self.assertEqual(batch["mask"].dtype, torch.bool)
        torch.testing.assert_close(batch["partner_normal"][1], rich["contact_partner_normal"])
        none = _contact_batch([empty, {}], torch.device("cpu"))
        self.assertEqual(tuple(none["sample_index"].shape), (2, 0))
        self.assertEqual(tuple(none["partner_point"].shape), (2, 0, 3))
        self.assertEqual(tuple(none["mask"].shape), (2, 0))

    def test_query_tolerates_payloads_without_pair_tensors(self):
        """Treat a payload without contact_* keys as contact-free instead of failing on a None batch entry."""
        config = MixedTrainConfig(**_TINY)
        rest = generate_cuboid(config.cell_counts, cell_size=config.cell_size)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
        step = MixedHexSolverStep(
            rest,
            fixed,
            network=_network(config),
            time_step=config.time_step,
            gravity=config.gravity,
            target_modes=config.target_modes,
        )
        step.eval()
        factory = train_mixed._TrajectoryFactory(step, rest, config, rank=0, validation=True)
        payload = factory.reset(0)
        legacy = {key: value for key, value in payload.items() if not key.startswith("contact_")}
        legacy["contact_partners"] = payload["contact_partners"]
        self.assertIsNone(train_mixed._batch([legacy], torch.device("cpu"), cell_count=2)["contact"])
        trajectory = _Trajectory(0, legacy, config.time_step)
        _run_query(
            step,
            [trajectory],
            cell_count=2,
            device=torch.device("cpu"),
            train_mixed=train_mixed,
            history_module=history,
            features=features,
        )
        self.assertIsNone(trajectory.failure)
        self.assertEqual((trajectory.calls, trajectory.max_pair_count, trajectory.max_penetration_r), (1, 0, 0.0))
        self.assertEqual(len(trajectory.energies), 1)

    def test_rollout_saves_contact_partners_and_reports_penetration(self):
        """Write the seed's plane and static points beside the frames and summarize contact, gravity and modes in report.json.

        The checkpoint is a seven-mode (schema 6) configuration, so the rollout
        rebuilds a seven-mode network and step; each seed's gravity is the one
        its validation context sampled.
        """
        config = MixedTrainConfig(**_TINY)
        network = _network(config)
        self.assertEqual(network.target_modes, 7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "format": "mixed_pool_v2",
                    "config": asdict(config),
                    "network_state": network.state_dict(),
                    "report": {"completed_epochs": 0, "best_selection": None},
                },
                checkpoint,
            )
            dt = config.time_step
            reports = run_rollouts(
                checkpoint, root / "rollout", seeds=(0, 1), iterations=1, duration=2 * dt, fps=300, device="cpu"
            )
            self.assertEqual([report["seed"] for report in reports], [0, 1])
            low, high = config.gravity_magnitude_range
            gravities = []
            for report in reports:
                self.assertEqual(report["status"], "complete", report["failure"])
                self.assertEqual(report["target_modes"], 7)
                gravity = report["gravity"]
                self.assertEqual((len(gravity), gravity[0], gravity[2]), (3, 0.0, 0.0))
                self.assertTrue(low <= -gravity[1] <= high, gravity)
                gravities.append(tuple(gravity))
                block = report["contact"]
                self.assertLessEqual(
                    {"plane_present", "plane_height", "point_count", "ke", "kd", "mu", "max_penetration_r"}, set(block)
                )
                self.assertIsInstance(block["plane_present"], bool)
                self.assertGreaterEqual(block["point_count"], 0)
                self.assertGreaterEqual(block["ke"], 0.0)
                # The scene's material-relative factor of the floored stiffness and whether the floor bound.
                self.assertIsInstance(block["floor_bound"], bool)
                self.assertAlmostEqual(
                    block["kappa"], block["ke"] / (_youngs_modulus(report["material"]) * config.cell_size), places=6
                )
                low_kappa, high_kappa = config.contact_kappa_range
                self.assertGreaterEqual(block["kappa"], low_kappa)
                self.assertTrue(block["kappa"] <= high_kappa or block["floor_bound"], block)
                self.assertGreaterEqual(block["max_penetration_r"], 0.0)
                self.assertGreaterEqual(block["max_pair_count"], 0)
                trajectory = root / "rollout" / f"seed_{report['seed']}" / "trajectory.npz"
                with np.load(trajectory) as data:
                    self.assertLessEqual(set(_CONTACT_KEYS), set(data.files))
                    self.assertEqual(data["contact_plane_present"].dtype, np.bool_)
                    self.assertEqual(data["contact_point_positions"].shape, (block["point_count"], 3))
                    self.assertEqual(data["contact_point_radii"].shape, (block["point_count"],))
                positions, times, _, _, _, contact = _load_trajectory(trajectory)
                self.assertEqual(positions.shape[0], 3)
                self.assertEqual(len(times), 3)
                self.assertEqual(contact.plane_present, block["plane_present"])
                self.assertEqual(contact.point_count, block["point_count"])
                self.assertAlmostEqual(float(contact.plane_point[1]), block["plane_height"], places=6)
                written = json.loads((trajectory.parent / "report.json").read_text())
                self.assertEqual(written["contact"], block)
                self.assertEqual((written["gravity"], written["target_modes"]), (report["gravity"], 7))
            self.assertEqual(len(set(gravities)), 2)

    def test_rollout_reports_the_legacy_kappa_of_a_scene_recorded_before_the_stiffness_floor(self):
        """A payload whose scene metadata predates the floor records ``kappa``; the report reads it and no floor flag."""
        config = MixedTrainConfig(**_TINY)
        real_reset = train_mixed._TrajectoryFactory.reset

        def legacy_reset(self, seed):
            payload = real_reset(self, seed)
            scene = payload["metadata"]["contact"]
            scene["kappa"] = scene.pop("kappa_effective")
            del scene["ke_floor"], scene["floor_bound"]
            return payload

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(train_mixed._TrajectoryFactory, "reset", legacy_reset),
        ):
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "format": "mixed_pool_v2",
                    "config": asdict(config),
                    "network_state": _network(config).state_dict(),
                    "report": {"completed_epochs": 0, "best_selection": None},
                },
                checkpoint,
            )
            dt = config.time_step
            (report,) = run_rollouts(
                checkpoint, root / "rollout", seeds=(0,), iterations=1, duration=2 * dt, fps=300, device="cpu"
            )
        self.assertEqual(report["status"], "complete", report["failure"])
        block = report["contact"]
        self.assertIsNone(block["floor_bound"])
        self.assertAlmostEqual(
            block["kappa"], block["ke"] / (_youngs_modulus(report["material"]) * config.cell_size), places=6
        )


if __name__ == "__main__":
    unittest.main()
