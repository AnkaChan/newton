# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise viscosity sampling through the detached mixed training boundary."""

import importlib.util
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.tests import test_train_mixed
from experiments.learned_intrinsic_solver.train_mixed import MixedTrainConfig, _batch, run_training


class TestDampingTraining(unittest.TestCase):
    def test_default_schema_and_legacy_checkpoint_config(self):
        """Enable sampled damping for new runs and explicitly preserve old zero-viscosity layouts."""
        config = MixedTrainConfig()
        self.assertEqual(config.damping_range, (10.0, 1000.0))
        self.assertEqual((config.state_feature_dim, config.conditioning_dim), (86, 6))
        old = asdict(config)
        del old["damping_range"], old["feature_schema_version"]
        legacy = MixedTrainConfig.from_checkpoint_config(old)
        self.assertEqual(legacy.damping_range, (0.0, 0.0))
        self.assertEqual((legacy.state_feature_dim, legacy.conditioning_dim), (38, 5))
        with self.assertRaisesRegex(ValueError, "legacy"):
            replace(config, feature_schema_version=1)

    def test_batch_preserves_physical_anchor_separately_from_candidate(self):
        """Collate the physical-step start without replacing it with an inner optimizer iterate."""
        start = torch.ones(8, 3, requires_grad=True)
        records = [
            {
                "candidate": 2 * start,
                "physical_positions": start,
                "inertial_prediction": 3 * start,
                "fixed_positions": start[:4],
                "context_id": "test",
            }
        ]
        batch = _batch(records, torch.device("cpu"))
        torch.testing.assert_close(batch["physical_positions"], start.detach()[None])
        self.assertFalse(batch["physical_positions"].requires_grad)
        self.assertFalse(torch.equal(batch["physical_positions"], batch["candidate"]))

    def test_legacy_resume_is_explicit_and_new_damping_cannot_replace_it(self):
        """Resume legacy weights only with legacy physics and feature dimensions."""
        config = replace(
            test_train_mixed.TestMixedTraining.config(1), feature_schema_version=1, damping_range=(0.0, 0.0)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_training(root, config)
            path = root / "checkpoints/latest.pt"
            saved = torch.load(path, weights_only=False)
            del saved["config"]["feature_schema_version"], saved["config"]["damping_range"]
            for spec in saved["rank_states"][0]["context_specs"].values():
                spec.pop("damping", None)
            for record in saved["rank_states"][0]["pool"]["records"]:
                record["payload"]["context_spec"].pop("damping", None)
                record["payload"]["metadata"]["material"].pop("damping", None)
                record["payload"]["metadata"]["material_ranges"].pop("damping", None)
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "resume configuration"):
                run_training(root, test_train_mixed.TestMixedTraining.config(2), resume=path)
            resumed = run_training(root, replace(config, max_epochs=2), resume=path)
            self.assertEqual(resumed["completed_epochs"], 2)

    def test_legacy_and_fresh_payloads_share_zero_damping_diagnostics(self):
        """Resume a batch mixing old missing-viscosity metadata with newly reset objects."""
        config = replace(
            test_train_mixed.TestMixedTraining.config(1),
            queries_per_epoch=2,
            feature_schema_version=1,
            damping_range=(0.0, 0.0),
            cpu_threads=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_training(root, config)
            path = root / "checkpoints/initial.pt"
            saved = torch.load(path, weights_only=False)
            pool = saved["rank_states"][0]["pool"]
            legacy_id = pool["dispatch"][1]
            legacy = next(record["payload"] for record in pool["records"] if record["id"] == legacy_id)
            del legacy["context_spec"]["damping"]
            del legacy["metadata"]["material"]["damping"]
            torch.save(saved, path)
            resumed = run_training(root, config, resume=path)
            self.assertEqual(resumed["epochs"][0]["rank_0_material_ranges"]["damping"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
