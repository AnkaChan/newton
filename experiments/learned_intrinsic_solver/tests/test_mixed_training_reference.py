# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify one-update checkpoint comparison detects gradient averaging errors."""

import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.mixed_training_reference import verify_first_update
from experiments.learned_intrinsic_solver.train_mixed import MixedTrainConfig, run_training


class TestMixedTrainingReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Save one bounded two-member CPU update from the real trainer."""
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        config = MixedTrainConfig(
            cell_counts=(1, 1, 2),
            cell_size=0.1,
            hidden_dim=8,
            edge_hidden_dim=4,
            num_heads=2,
            batch_size=2,
            pool_multiplier=2,
            queries_per_epoch=2,
            max_epochs=1,
            iteration_counts=(1,),
            physical_step_counts=(1,),
            validation_count=2,
            validation_iterations=1,
            validation_physical_steps=1,
            validation_physical_iterations=1,
            validation_full_count=1,
            validation_full_interval=1,
            device="cpu",
            cpu_threads=1,
            preparation_workers=1,
            verbose=False,
            early_stopping=False,
        )
        run_training(cls.root / "run", config)
        cls.initial_path = cls.root / "run/checkpoints/initial.pt"
        cls.after_path = cls.root / "run/checkpoints/latest.pt"
        cls.initial = torch.load(cls.initial_path, map_location="cpu", weights_only=False)
        cls.after = torch.load(cls.after_path, map_location="cpu", weights_only=False)

    def _write(self, name, value):
        path = self.root / name
        torch.save(value, path)
        return path

    def test_real_first_update_matches_all_parameters_and_adam_state(self):
        """Reconstruct the trainer's complete first update from detached pool inputs."""
        report = verify_first_update(self.initial_path, self.after_path, device="cpu")
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["global_batch_size"], 2)
        self.assertEqual(report["world_size"], 1)
        self.assertEqual(report["distinct_material_count"], 2)
        self.assertTrue(report["rank_parameter_agreement"])
        self.assertGreater(report["parameters"]["element_count"], 0)
        self.assertEqual(report["optimizer"]["tensor_count"], 3 * report["parameters"]["tensor_count"])
        json.dumps(report, allow_nan=False)

    def test_reference_loss_matches_the_recorded_first_update_loss(self):
        """State the LeCO objective independently and reproduce the trainer's batch loss."""
        report = verify_first_update(self.initial_path, self.after_path, device="cpu")
        recorded = self.after["report"]["updates"][0]["loss"]
        self.assertAlmostEqual(report["reference_loss"], recorded, places=5)

    def test_different_viscosities_are_heterogeneous_materials(self):
        """Accept a real matching update whose only material variation is viscosity."""
        config = replace(
            MixedTrainConfig.from_checkpoint_config(self.initial["config"]),
            youngs_modulus_range=(1e4, 1e4),
            poissons_ratio_range=(0.3, 0.3),
            density_range=(1000.0, 1000.0),
            cpu_threads=2,
        )
        output = self.root / "damping-only"
        run_training(output, config)
        report = verify_first_update(output / "checkpoints/initial.pt", output / "checkpoints/latest.pt")
        self.assertTrue(report["parameters"]["passed"], report)
        self.assertTrue(report["optimizer"]["passed"], report)
        self.assertEqual(report["distinct_material_count"], 2)
        self.assertTrue(report["heterogeneous_materials"])
        self.assertTrue(report["passed"], report)

    def test_concatenated_rank_batches_have_the_same_global_reference(self):
        """Treat two saved rank batches as one equally weighted global batch."""
        initial, after = copy.deepcopy(self.initial), copy.deepcopy(self.after)
        for saved in (initial, after):
            saved["world_size"] = 2
            saved["config"]["batch_size"] = 1
            saved["report"]["world_size"] = 2
            saved["report"]["config"]["batch_size"] = 1
        source = initial["rank_states"][0]
        ranks = []
        for rank in range(2):
            state = copy.deepcopy(source)
            pool = state["pool"]
            pool["records"] = pool["records"][rank::2]
            selected = {record["id"] for record in pool["records"]}
            for queue in ("dispatch", "ready", "pending"):
                pool[queue] = [identity for identity in pool[queue] if identity in selected]
            pool["batch_size"], pool["capacity"] = 1, 2
            keys = {record["payload"]["context_id"] for record in pool["records"]}
            state["context_specs"] = {key: value for key, value in state["context_specs"].items() if key in keys}
            ranks.append(state)
        initial["rank_states"] = ranks
        after["rank_states"] = [copy.deepcopy(after["rank_states"][0]) for _ in range(2)]
        report = verify_first_update(self._write("two-initial.pt", initial), self._write("two-after.pt", after))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["rank_batch_sizes"], [1, 1])
        self.assertEqual(report["world_size"], 2)

    def test_optimizer_moments_detect_sum_instead_of_mean_gradients(self):
        """Reject incorrectly scaled gradients even when Adam parameters look similar."""
        corrupted = copy.deepcopy(self.after)
        for state in corrupted["optimizer_state"]["state"].values():
            state["exp_avg"] *= 2
            state["exp_avg_sq"] *= 4
        report = verify_first_update(self.initial_path, self._write("wrong-moments.pt", corrupted))
        self.assertFalse(report["passed"])
        self.assertTrue(report["parameters"]["passed"])
        self.assertFalse(report["optimizer"]["passed"])
        self.assertGreater(report["optimizer"]["relative_l2_error"], 0.9)

    def test_parameter_and_rank_hash_corruption_fail_independently(self):
        """Reject changed parameters and divergent distributed parameter fingerprints."""
        corrupted = copy.deepcopy(self.after)
        corrupted["network_state"]["correction_head.weight"] += 0.01
        report = verify_first_update(self.initial_path, self._write("wrong-weights.pt", corrupted))
        self.assertFalse(report["parameters"]["passed"])
        self.assertFalse(report["rank_parameter_agreement"])
        corrupted = copy.deepcopy(self.after)
        corrupted["rank_states"][0]["parameter_sha256"] = "different-rank-weights"
        report = verify_first_update(self.initial_path, self._write("wrong-hash.pt", corrupted))
        self.assertTrue(report["parameters"]["passed"])
        self.assertFalse(report["rank_parameter_agreement"])
        self.assertFalse(report["passed"])

    def test_reject_noninitial_or_incomplete_rank_batch_checkpoints(self):
        """Reject verification inputs that cannot represent exactly one full update."""
        corrupted = copy.deepcopy(self.after)
        corrupted["report"]["completed_updates"] = 2
        with self.assertRaisesRegex(ValueError, "one update"):
            verify_first_update(self.initial_path, self._write("two-updates.pt", corrupted))
        corrupted = copy.deepcopy(self.initial)
        corrupted["rank_states"][0]["pool"]["dispatch"] = [0]
        with self.assertRaisesRegex(ValueError, "full batch"):
            verify_first_update(self._write("short-batch.pt", corrupted), self.after_path)

    def test_reject_initial_dispatch_with_optimizer_history_or_legacy_config(self):
        """Require a historyless first dispatch and the revised schema."""
        corrupted = copy.deepcopy(self.initial)
        record = corrupted["rank_states"][0]["pool"]["records"][0]
        record["payload"]["history_valid"] = True
        with self.assertRaisesRegex(ValueError, "optimizer history"):
            verify_first_update(self._write("with-history.pt", corrupted), self.after_path)
        corrupted = copy.deepcopy(self.initial)
        record = corrupted["rank_states"][0]["pool"]["records"][0]
        record["payload"]["history_axis_gradient_world"] += 1
        with self.assertRaisesRegex(ValueError, "optimizer history"):
            verify_first_update(self._write("nonzero-history.pt", corrupted), self.after_path)
        legacy_initial, legacy_after = copy.deepcopy(self.initial), copy.deepcopy(self.after)
        for saved in (legacy_initial, legacy_after):
            saved["config"]["feature_schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "legacy"):
            verify_first_update(
                self._write("legacy-initial.pt", legacy_initial), self._write("legacy-after.pt", legacy_after)
            )


if __name__ == "__main__":
    unittest.main()
