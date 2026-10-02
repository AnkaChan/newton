# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""v5 validation (design spec section 11): held-out scene records with the new keys, the report summaries with
scene samples, and the momentum drift of the contact-free check with the zero-init network."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.learned_intrinsic_solver.mixed_report import write_mixed_report
from experiments.lido import contact, scenes_v5
from experiments.lido import report as R
from experiments.lido import validation as V
from experiments.lido.augment import Augmenter
from experiments.lido.fusion import Fusion
from experiments.lido.grid import GridCache
from experiments.lido.network import Net
from experiments.lido.runner import scene_batch
from experiments.lido.step import Step
from experiments.lido.tests.test_body_contact import two_boxes
from experiments.lido.tests.test_report import cheap_samples, epoch_record
from experiments.lido.tests.test_scene_runner import MASTER, scene_cfg

NEW_KEYS = ("interbody_penetration_r", "plane_penetration_r", "contact_pairs")


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


class TestValidationV5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = scene_cfg()
        cls.grids = GridCache("cpu")
        cls.aug = Augmenter("cpu")
        cls.step = Step(Net.from_config(cls.cfg), Fusion(), cls.aug)  # zero-init heads: no shape update

    def test_cheap_records_and_summary(self):
        cfg = self.cfg
        records = V.validate_cheap_v5(self.step, cfg, self.grids, self.aug, "cpu", MASTER)
        self.assertEqual([r["seed"] for r in records], list(range(cfg.validation_scene_count)))
        K = cfg.validation_iterations
        for r in records:
            self.assertTrue(r["survived"] and not r["failed_first_update"] and r["contact"])
            self.assertEqual(r["scene"]["validation"], True)
            for key in ("residual_n", "energy_joule", "penetration_r", "inverted_cells", *NEW_KEYS[:2]):
                self.assertEqual(len(r[key]), K + 1, key)
                self.assertTrue(all(_finite(v) for v in r[key]), key)
            self.assertTrue(all(v >= 0 for v in r["interbody_penetration_r"] + r["plane_penetration_r"]))
            self.assertEqual(set(r["contact_pairs"]), {"total", "plane", "point", "body"})
            self.assertEqual(r["contact_pairs"]["point"], 0)
            self.assertGreater(r["scale_joule"], 0.0)
            self.assertGreater(r["energy_joule"][0], 0.0)
        summary = R.summarize_cheap_validation(records, K)
        self.assertEqual(summary["sample_count"], cfg.validation_scene_count)
        self.assertEqual(summary["physical_survivors"], cfg.validation_scene_count)
        self.assertTrue(summary["selection"]["eligible"] and _finite(summary["selection"]["metric"]))
        for key in NEW_KEYS[:2]:
            self.assertEqual([row["iteration"] for row in summary[key]], list(range(K + 1)))
            self.assertTrue(all(_finite(row["max"]) for row in summary[key]))
        self.assertEqual(set(summary["contact_pairs"]), {"total", "plane", "body"})
        # body-mode samples (without the scene keys) keep the previous key set
        plain = R.summarize_cheap_validation(cheap_samples(), iterations=3)
        for key in (*NEW_KEYS, "momentum_drift"):
            self.assertNotIn(key, plain)

    def test_full_horizon_records_summary_and_epoch_record(self):
        cfg = self.cfg
        K, H = 1, 2
        records, seconds = V.validate_full_horizon_v5(self.step, cfg, self.grids, self.aug, "cpu", MASTER, K, H)
        self.assertEqual(len(records), cfg.validation_full_scene_count)
        self.assertGreater(seconds, 0.0)
        r = records[0]
        self.assertTrue(r["survived"])
        self.assertEqual(len(r["physical_records"]), H)
        for row in r["physical_records"]:
            for key in ("residual_n", "energy_joule", "penetration_r", "inverted_cells", *NEW_KEYS[:2]):
                self.assertTrue(_finite(row[key]), key)
            self.assertEqual(set(row["contact_pairs"]), {"total", "plane", "point", "body"})
        self.assertTrue(_finite(r["momentum_drift"]))
        full = R.summarize_full_horizon(records, K, H, seconds)
        self.assertTrue(full["selection"]["eligible"] and _finite(full["selection"]["metric"]))
        for key in NEW_KEYS[:2]:
            self.assertTrue(_finite(full[f"final_{key}"]["max"]), key)
        self.assertEqual(set(full["final_contact_pairs"]), {"total", "plane", "body"})
        self.assertAlmostEqual(full["momentum_drift"], r["momentum_drift"])
        # the epoch record passes the scene keys through and stays JSON-clean
        record, _, _ = epoch_record()
        self.assertNotIn("scene_regime", record)
        cheap = R.summarize_cheap_validation(
            V.validate_cheap_v5(self.step, cfg, self.grids, self.aug, "cpu", MASTER), cfg.validation_iterations
        )
        record = R.build_epoch_record(
            epoch=1,
            loss=0.5,
            query_count=10,
            updates=2,
            seconds=1.0,
            lr=1e-4,
            grad_norm_mean=0.1,
            grad_norm_max=0.2,
            step_mean=0.02,
            step_min=0.01,
            step_max=0.03,
            tie_cell_count=0,
            mean_force_residual_n=1.0,
            resets=0,
            failures=[],
            regime={"stage": 0, "k_max": 1, "h_max": 8, "queries": 10, "filler_queries": 0, "updates": 2},
            available_K=[1],
            available_H=[8],
            contact_scene_fraction=1.0,
            contact_realized_fraction=0.0,
            contact_max_penetration_r=0.0,
            validation=cheap,
            full_horizon_validation=full,
            scene_regime={"scenes": 3, "scenes_served": [{"seed": 0, "bodies": 4}]},
        )
        self.assertEqual(record["scene_regime"]["scenes"], 3)
        self.assertIn("momentum_drift", record["full_horizon_validation"])
        self.assertIn("interbody_penetration_r", record["validation"])
        R.dumps(record)  # no NaN, no tensors
        # the existing dashboard renders a run with the scene keys
        with tempfile.TemporaryDirectory() as tmp:
            run = R.RunReport(Path(tmp) / "run", cfg, 1)
            run.set_best_selection(full, epoch=1)
            run.log_epoch(record)
            report = json.loads((Path(tmp) / "run" / "report.json").read_text())
            self.assertEqual(report["epochs"][0]["scene_regime"]["scenes"], 3)
            write_mixed_report(Path(tmp) / "public", report)
            self.assertTrue((Path(tmp) / "public" / "index.html").is_file())

    def test_momentum_drift_of_the_contact_free_check(self):
        cfg = self.cfg
        for index in range(2):
            scene = scenes_v5.held_out_scene(MASTER, index, cfg)
            free = V.contact_free_copy(scene, 8)
            b = scene_batch(free, self.grids, self.aug, "cpu", plane=False)
            self.step.prepare(b, b.active, None)
            self.assertFalse(bool(b.scene.plane_present.any()))
            self.assertEqual(b.pairs.count, 0)  # body-body detection on, nothing within reach
            drift = V.momentum_drift_check(self.step, cfg, self.grids, self.aug, "cpu", MASTER, 1, 8, scene)
            self.assertTrue(math.isfinite(drift))
            self.assertLess(drift, 1e-3, (index, drift))

    def test_kind_penetration(self):
        b = two_boxes(gap=0.4, plane_d=-0.3)
        b.pairs = contact.detect(b, b.X, b.V)
        kinds = contact.kind_penetration(b, b.x)
        self.assertEqual(tuple(kinds.shape), (3,))
        self.assertGreater(kinds[contact.KIND_BODY].item(), 0.0)
        self.assertGreater(kinds[0].item(), 0.0)
        self.assertEqual(kinds[1].item(), 0.0)
        self.assertAlmostEqual(kinds.max().item(), contact.penetration(b, b.x).max().item(), places=12)
        apart = two_boxes(gap=2.0)  # beyond the detection reach: no pairs, zero everywhere
        apart.pairs = contact.detect(apart, apart.X, apart.V)
        self.assertEqual(apart.pairs.count, 0)
        self.assertEqual(contact.kind_penetration(apart, apart.x).abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
