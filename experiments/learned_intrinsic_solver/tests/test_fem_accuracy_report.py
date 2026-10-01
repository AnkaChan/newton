# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the FEM accuracy comparison page, the side-by-side composite, and the index card."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.learned_intrinsic_solver import fem_accuracy_report as report
from experiments.learned_intrinsic_solver import fem_accuracy_scenarios as scenarios

try:
    import imageio.v2 as imageio
    import imageio_ffmpeg  # noqa: F401

    HAVE_FFMPEG = True
except ImportError:
    imageio = None
    HAVE_FFMPEG = False


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _series(frames: int, final: float) -> tuple[list[float], list[float]]:
    times = [frame / scenarios.FPS for frame in range(frames + 1)]
    values = [final * (1.0 - math.exp(-frame / 20.0)) for frame in range(frames + 1)]
    return times, values


def _metrics(name: str, solver: str) -> dict:
    """Return complete metrics with learned values that agree, deviate, or invert per scenario.

    The extension tip agrees with the 100-iteration VBD reference of the
    fixture but not with the tabulated 20-iteration VBD value.
    """
    scenario = scenarios.SCENARIOS[name]
    base = {"scenario": name, "frame_count": scenario.frame_count, "completed": True}
    base["final_time_seconds"] = scenario.frame_count / scenarios.FPS
    learned = solver == "learned"
    if name == "extension":
        tip = 0.0095 if learned else 0.01525
        times, values = _series(scenario.frame_count, tip)
        base.update(
            {
                "tip_displacement_final": tip,
                "tip_displacement_series": values,
                "series_times": times,
                "analytic_tip_displacement": scenarios.analytic_extension_tip_displacement(),
                "bulk_volume_ratio": 1.0064 if learned else 1.00643,
                "min_centre_jacobian_ratio": 1.0001 if learned else 1.00012,
            }
        )
    elif name == "stretch":
        base.update(
            {
                "bulk_volume_ratio": 1.30 if learned else 1.223,
                "lateral_contraction": 0.75 if learned else 0.770,
                "min_centre_jacobian_ratio": 1.25 if learned else 1.199,
            }
        )
    elif name == "twist":
        base.update(
            {
                "peak_frame": scenario.ramp_frames,
                "bulk_volume_ratio_peak": 0.90 if learned else 0.934,
                "min_centre_jacobian_ratio_peak": -0.05 if learned else 0.862,
                "bulk_volume_ratio_final": 0.90 if learned else 0.934,
                "min_centre_jacobian_ratio_final": -0.05 if learned else 0.862,
            }
        )
    else:
        base.update(
            {
                "release_frame": scenario.release_frame,
                "min_centre_jacobian_ratio_compression": -0.03 if learned else -0.021,
                "length_recovery_ratio": 0.96 if learned else 0.953,
                "bulk_volume_ratio_final": 1.0 if learned else 0.9995,
            }
        )
    if learned:
        base.update({"solver": "learned", "failure": None})
    return base


def _fake_root(root: Path, *, learned_names=None, composites=("extension", "twist")) -> None:
    learned_names = set(scenarios.SCENARIOS if learned_names is None else learned_names)
    for name in scenarios.SCENARIOS:
        _write_json(root / "vbd" / name / "metrics.json", _metrics(name, "vbd"))
        _write_json(
            root / "vbd" / name / "run.json",
            {
                "solver": {"name": "Newton SolverVBD", "iterations_per_substep": 20},
                "environment": {"newton_version": "1.7.0.dev0", "newton_git_revision": "0db3a304abb7"},
            },
        )
        if name in learned_names:
            _write_json(root / "learned" / name / "metrics.json", _metrics(name, "learned"))
            _write_json(
                root / "learned" / name / "run.json",
                {
                    "solver": "learned",
                    "status": "complete",
                    "failure": None,
                    "checkpoint": "generated/training_v4/checkpoints/best_validation.pt",
                    "checkpoint_sha256": "abcdef0123456789",
                    "checkpoint_epoch": 20,
                    "iterations_per_substep": 8,
                },
            )
        for solver, _ in report.SOLVERS:
            if solver == "learned" and name not in learned_names:
                continue
            render = root / "renders" / f"{solver}_{name}"
            render.mkdir(parents=True, exist_ok=True)
            (render / "simulation.mp4").write_bytes(b"mp4")
            (render / "initial.png").write_bytes(b"png")
        if name in composites and name in learned_names:
            (root / "renders" / f"side_by_side_{name}.mp4").write_bytes(b"mp4")
    times, values = _series(300, 0.00978)
    _write_json(
        root / "vbd_iter100" / "extension" / "metrics.json",
        {
            "scenario": "extension",
            "frame_count": 300,
            "completed": True,
            "tip_displacement_final": 0.00978,
            "tip_displacement_series": values,
            "series_times": times,
        },
    )


class TestHomogeneousReference(unittest.TestCase):
    def test_stretch_reference_matches_the_stable_neo_hookean_stationary_point(self):
        """Stretching to 2 L gives J = 1 + mu / (2 (lambda + mu)) and lateral sqrt(J / 2)."""
        volume_ratio, lateral = report.homogeneous_stretch_reference(2.0)
        ratio = scenarios.LAME_MU / (scenarios.LAME_LAMBDA + scenarios.LAME_MU)
        self.assertAlmostEqual(volume_ratio, 1.0 + 0.5 * ratio)
        self.assertAlmostEqual(volume_ratio, 1.2)
        self.assertAlmostEqual(lateral, math.sqrt(0.6))
        self.assertEqual(report.homogeneous_stretch_reference(1.0), (1.0, 1.0))
        with self.assertRaises(ValueError):
            report.homogeneous_stretch_reference(0.0)


class TestBuildReport(unittest.TestCase):
    def test_page_has_every_scenario_with_videos_tables_plot_and_verdicts(self):
        """Stage clips, prefer composites, fall back to two clips, and grade each scenario from its metrics."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_root(root)
            result = report.build_report(root, site=root / "site")
            page = (root / "site" / "index.html").read_text()
            self.assertIn(report.TITLE, page)
            self.assertIn(report.PULL_REQUEST_URL, page)
            self.assertEqual(page.count('<article class="card" id="'), 4)
            self.assertIn("epoch 20 checkpoint of the v4 training campaign", page)
            self.assertIn("K = 8 learned iterations per substep", page)
            self.assertIn("20 Gauss-Seidel iterations per substep", page)
            self.assertIn("9.81 mm", page)
            self.assertIn("1.2000", page)
            self.assertIn("videos/side_by_side_extension.mp4", page)
            self.assertNotIn("videos/side_by_side_stretch.mp4", page)
            self.assertIn("videos/learned_stretch.mp4", page)
            self.assertIn("videos/vbd_stretch.mp4", page)
            self.assertIn("plots/extension_tip_displacement.png", page)
            self.assertTrue((root / "site" / "plots" / "extension_tip_displacement.png").is_file())
            self.assertTrue((root / "site" / "videos" / "side_by_side_extension.mp4").is_file())
            self.assertTrue((root / "site" / "videos" / "vbd_compression_release.mp4").is_file())
            self.assertTrue((root / "site" / "data" / "learned_extension_metrics.json").is_file())
            self.assertTrue((root / "renders" / "learned_extension.mp4").is_file())
            statuses = {entry["name"]: entry["status"] for entry in result["scenarios"]}
            self.assertEqual(
                statuses,
                {
                    "extension": "agreement",
                    "stretch": "deviation",
                    "twist": "inversion",
                    "compression_release": "inversion",
                },
            )
            extension = result["scenarios"][0]
            self.assertTrue(any("100 iterations 9.78 mm" in sentence for sentence in extension["verdict"]))
            self.assertTrue(any("largest learned-to-VBD difference" in sentence for sentence in extension["verdict"]))
            self.assertTrue(any("converged reference" in sentence for sentence in extension["verdict"]))
            self.assertIn("Newton VBD with more iterations per substep", page)
            self.assertFalse(any("agree within" in sentence for sentence in result["scenarios"][1]["verdict"]))
            twist = result["scenarios"][2]
            self.assertTrue(any("negative centre Jacobian" in sentence for sentence in twist["verdict"]))
            self.assertEqual(result["checkpoint"]["epoch"], 20)
            self.assertEqual(result["vbd"]["iterations_per_substep"], 20)
            saved = json.loads((root / "site" / "report.json").read_text())
            self.assertEqual(saved["verdict_counts"], {"agreement": 1, "deviation": 1, "inversion": 2})

    def test_failed_and_missing_learned_runs_are_reported_without_claiming_agreement(self):
        """A nonfinite failure names its frame, a missing run is marked not available, and None metrics render."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_root(root, learned_names=("extension", "stretch", "twist"), composites=())
            failed = _metrics("twist", "learned")
            failed.update(
                {
                    "frame_count": 57,
                    "completed": False,
                    "final_time_seconds": 57 / scenarios.FPS,
                    "bulk_volume_ratio_peak": None,
                    "min_centre_jacobian_ratio_peak": None,
                    "bulk_volume_ratio_final": 0.99,
                    "min_centre_jacobian_ratio_final": 0.80,
                    "failure": {"frame": 58, "substep": 575, "reason": "nonfinite proposal"},
                }
            )
            _write_json(root / "learned" / "twist" / "metrics.json", failed)
            result = report.build_report(root, site=root / "site")
            statuses = {entry["name"]: entry["status"] for entry in result["scenarios"]}
            self.assertEqual(statuses["twist"], "failure")
            self.assertEqual(statuses["compression_release"], "missing")
            twist = result["scenarios"][2]
            self.assertTrue(any("nonfinite proposal at frame 58" in sentence for sentence in twist["verdict"]))
            self.assertFalse(any("agree within" in sentence for sentence in twist["verdict"]))
            page = (root / "site" / "index.html").read_text()
            self.assertIn("not reached", page)
            self.assertIn("no run", page)
            self.assertIn("No Learned intrinsic solver clip for compression_release", page)
            self.assertIn("Not available", page)


class TestIndexCard(unittest.TestCase):
    def test_card_is_inserted_first_once_and_existing_cards_survive(self):
        """Insert the card after the section opening, never twice, and keep the earlier cards."""
        anchor = '<section class="cards" aria-label="Project webpages">'
        existing = '<a class="card" href="older/index.html" data-older><h2>Older</h2></a>'
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.html"
            index.write_text(f"<html><body><main>{anchor}{existing}</section></main></body></html>")
            arguments = {
                "href": "fem-accuracy/index.html",
                "badge": "FEM accuracy",
                "title": "Learned vs <VBD>",
                "description": "Four scenarios & verdicts.",
                "marker": "fem-accuracy",
            }
            self.assertTrue(report.add_index_card(index, **arguments))
            text = index.read_text()
            self.assertIn(existing, text)
            self.assertLess(text.index("data-fem-accuracy"), text.index("data-older"))
            self.assertIn("Learned vs &lt;VBD&gt;", text)
            self.assertIn("Four scenarios &amp; verdicts.", text)
            self.assertFalse(report.add_index_card(index, **arguments))
            self.assertEqual(index.read_text().count("data-fem-accuracy"), 1)
            index.write_text("<html><body>no cards</body></html>")
            with self.assertRaises(ValueError):
                report.add_index_card(index, **arguments)


@unittest.skipUnless(HAVE_FFMPEG, "imageio_ffmpeg is not installed")
class TestComposeSideBySide(unittest.TestCase):
    def test_shorter_clip_is_held_to_the_longer_frame_count(self):
        """Stack a 5-frame and an 8-frame clip into one 8-frame clip twice as wide."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = []
            for name, frames, shade in (("left.mp4", 5, 40), ("right.mp4", 8, 200)):
                path = root / name
                with imageio.get_writer(str(path), fps=30, codec="libx264", pixelformat="yuv420p") as writer:
                    for _ in range(frames):
                        writer.append_data(np.full((48, 64, 3), shade, dtype=np.uint8))
                clips.append(path)
            output = root / "composite.mp4"
            result = report.compose_side_by_side(clips[0], clips[1], output, labels=("L", "R"))
            self.assertEqual(result, {"frames": 8, "left_frames": 5, "right_frames": 8, "padded": True})
            self.assertTrue(output.is_file())
            self.assertEqual(report._frame_count(output), 8)
            reader = imageio.get_reader(str(output))
            try:
                self.assertEqual(tuple(reader.get_meta_data()["size"]), (128, 48))
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
