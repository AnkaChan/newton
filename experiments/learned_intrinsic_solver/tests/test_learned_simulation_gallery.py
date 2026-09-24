# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that partial learned-physics clips are labelled with actual duration."""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.learned_intrinsic_solver.learned_simulation_gallery import build_gallery


class TestLearnedSimulationGallery(unittest.TestCase):
    def test_default_discovers_ten_seed_directories(self):
        """Find report files under their seed directories without an explicit seed list."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(10000, 10010):
                simulation = root / "simulation" / f"seed_{seed}"
                render = root / "render" / f"seed_{seed}"
                simulation.mkdir(parents=True)
                render.mkdir(parents=True)
                (simulation / "report.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "checkpoint_epoch": 198,
                            "time_step": 1 / 300,
                            "optimizer_iterations_per_step": 2,
                            "requested_duration_seconds": 10,
                            "actual_duration_seconds": 10,
                            "status": "complete",
                            "failure": None,
                        }
                    )
                )
                (render / "render.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "video": "simulation.mp4",
                            "initial_image": "initial.png",
                            "final_image": "final.png",
                            "last_physical_time_seconds": 10,
                        }
                    )
                )
                for name in ("simulation.mp4", "initial.png", "final.png"):
                    (render / name).touch()
            result = build_gallery(root / "simulation", root / "render", output=root / "web")
            self.assertEqual(result["sample_count"], 10)
            self.assertEqual(result["complete_count"], 10)

    def test_failed_case_shows_actual_time_and_failure_without_claiming_completion(self):
        """Show seed, physical setup, and the stopped time beside the partial clip."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulation = root / "simulation" / "seed_10000"
            render = root / "render" / "seed_10000"
            simulation.mkdir(parents=True)
            render.mkdir(parents=True)
            (simulation / "report.json").write_text(
                json.dumps(
                    {
                        "seed": 10000,
                        "checkpoint_epoch": 198,
                        "time_step": 1 / 300,
                        "optimizer_iterations_per_step": 2,
                        "requested_duration_seconds": 10.0,
                        "actual_duration_seconds": 0.37,
                        "status": "failed",
                        "failure": {"error": "invalid learned output", "time_seconds": 0.373},
                        "initial_velocity_rms_m_per_s": 0.2,
                        "initial_velocity_max_m_per_s": 0.4,
                    }
                )
            )
            (render / "render.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "video": "simulation.mp4",
                        "initial_image": "initial.png",
                        "final_image": "final.png",
                        "rendered_frame_count": 13,
                    }
                )
            )
            for name in ("simulation.mp4", "initial.png", "final.png"):
                (render / name).touch()
            result = build_gallery(root / "simulation", root / "render", output=root / "web", expected_seeds=(10000,))
            html = (root / "web" / "index.html").read_text()
            self.assertEqual(result["failed_count"], 1)
            self.assertIn("0.37 s", html)
            self.assertIn("10 s requested", html)
            self.assertIn("invalid learned output", html)
            self.assertIn("Stopped when a learned update failed physical validity checks", html)
            self.assertIn("<details>", html)
            self.assertIn('href="../index.html"', html)
            self.assertIn("Epoch 198", html)
            self.assertIn("2 learned iterations", html)
            self.assertIn("seed_10000/simulation.mp4", html)
            self.assertNotIn("10 s simulated", html)
            self.assertTrue((root / "web/seed_10000/simulation.mp4").samefile(render / "simulation.mp4"))
            self.assertTrue((root / "web/seed_10000/initial.png").samefile(render / "initial.png"))
            self.assertTrue((root / "web/seed_10000/simulation_report.json").samefile(simulation / "report.json"))
            self.assertTrue((root / "web/seed_10000/render.json").samefile(render / "render.json"))


if __name__ == "__main__":
    unittest.main()
