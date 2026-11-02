# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record the selected cloth recipes with upstream rigid-soft DAT and OGC.

Use standard particle elasticity and native fast math off. Keep the historical
compiler experiment separate. All outcomes use the scene's own measurements.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import sys
from pathlib import Path

import warp as wp

from .fast_math.provenance import sha256, validate_output, verify_physics

HERE = Path(__file__).resolve().parent
CASES = (
    "j3_h1_jacket",
    "r7_h1_towel",
    "m2_franka_pouch",
    "m3_franka_tote_empty",
    "r5_allegro_towel",
    "r5_franka_dress",
    "r5_franka_patch_contact_v2",
    "r5_franka_towel_contact_v2",
    "r5_franka_trousers",
)


def parse_arguments(argv=None):
    """Validate the fixed campaign before importing any scene factory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hold-seconds", type=float, default=60.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--h1-initial-cache", type=Path)
    args = parser.parse_args(argv)
    for name in ("hold_seconds", "video_fps"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration < 0):
        parser.error("--duration must be finite and nonnegative")
    if args.h1_initial_cache and args.case not in ("j3_h1_jacket", "r7_h1_towel"):
        parser.error("--h1-initial-cache is only valid for the two H1 scenes")
    return args


def main(argv=None):
    """Run one recorded scene with the upstream solver and source guards."""
    args = parse_arguments(argv)
    out = args.out.resolve()
    validate_output(out)
    if args.h1_initial_cache:
        os.environ["NEWTON_COMPARE_H1_INITIAL_IMPORT"] = str(args.h1_initial_cache.resolve())
    from . import comparison_runtime as runtime
    from . import run_comparison as runner
    from .fast_math.run_fast_math import package_provenance, task_result
    from .fast_math.task_observer import TaskObserver

    runtime.configure_main_dat(args.case)
    manifest_path = HERE / "dat_ogc_physics_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    physics = verify_physics(runner.ROOT, manifest, warp_version=wp.__version__)
    guarded = set(HERE.rglob("*.py"))
    guarded.update(runner.ROOT / name for name in physics["files_sha256"])
    guarded.update((manifest_path, HERE / "cases.json", HERE / "recipes/h1/manifest.json"))
    hashes = {str(path): sha256(path) for path in sorted(guarded)}
    out.mkdir(parents=True, exist_ok=True)
    save_json = runner.save_json
    provenance = {
        "schema_version": 1,
        "campaign": "main-rigid-soft-dat-ogc-20261102",
        "physics_base_revision": physics["base_revision"],
        "newton_revision": physics["revision"],
        "physics_manifest_sha256": sha256(manifest_path),
        "physics_source_sha256": physics["files_sha256"],
        "guarded_source_sha256": hashes,
        "warp_package": package_provenance(),
        "warp_distribution_version": importlib.metadata.version("warp-lang"),
        "fast_math_enabled": False,
        "particle_elasticity": "standard VBD",
        "rigid_soft_enable_dat": True,
        "ogc_scope": "Merged PR4936 full-surface edge-force rule, Eq14; not every part of OGC",
        "kinematic_limit": "Prescribed kinematic hands move outside DAT truncation",
        "comparison_limit": "Current main and Warp differ from the saved baseline; not an isolated DAT ablation",
        "initial_import": str(args.h1_initial_cache.resolve()) if args.h1_initial_cache else None,
        "command": [sys.executable, "-m", "experiments.collision_benchmark.run_dat_ogc", *(argv or sys.argv[1:])],
    }
    tasks = []
    contact_summary = {
        "sample_count": 0,
        "max_soft_rows": 0,
        "max_rigid_rows": 0,
        "collision_calls": 0,
        "self_contact_peak_counts": [0, 0, 0],
        "self_contact_peak_excess": [0, 0, 0],
        "self_detection_calls": 0,
        "overflow": False,
        "scope": (
            "Sticky global rigid/full-surface and soft-self row counts across every owned detection, "
            "including preparation; excludes per-body adjacency capacities. "
            "Self-contact order: vertex-triangle, edge-edge, triangle-vertex."
        ),
    }
    base_observer = runner.RunObserver

    class MainObserver(base_observer):
        """Keep the existing criteria and stop on incomplete contact coverage."""

        def __init__(self, scene, observer):
            super().__init__(scene, observer)
            self.task = TaskObserver(scene)
            tasks.append(self.task)
            solver = runtime.scene_solver(scene)
            if not solver.rigid_soft_enable_dat or solver.integrate_with_external_rigid_solver:
                raise AssertionError("The requested native rigid-soft DAT path is not enabled")
            if solver.collision_pipeline is None:
                raise AssertionError("DAT needs a solver-owned collision pipeline")
            provenance["effective_solver_options"] = {
                key: value
                for key, value in solver.comparison_options.items()
                if value is None or isinstance(value, (int, float, str, bool))
            }
            provenance["initial_fingerprint"] = runtime.input_fingerprint(scene)
            provenance["common_start"] = getattr(scene.raw_scene, "comparison_start", None)
            provenance["runtime_configuration"] = runtime.main_dat_configuration()
            save_json(out / "dat-ogc-provenance.json", provenance)

        def after_substep(self, state, dt):
            super().after_substep(state, dt)
            if self.observer._first_nonfinite is not None:
                raise FloatingPointError(f"Nonfinite physical state at {self.observer._first_nonfinite:.12g} s")
            self.task.sample(state, self.time, self.scene.phase)
            solver = runtime.scene_solver(self.scene)
            contacts = solver.contacts
            counts = solver.collision_pipeline.dat_contact_latch.numpy()
            self_counts = solver.collision_pipeline.dat_self_contact_latch.numpy()
            contact_summary["sample_count"] += 1
            contact_summary["max_rigid_rows"] = max(contact_summary["max_rigid_rows"], int(counts[0]))
            contact_summary["max_soft_rows"] = max(contact_summary["max_soft_rows"], int(counts[1]))
            contact_summary["collision_calls"] = int(counts[4])
            contact_summary["self_contact_peak_counts"] = self_counts[:3].tolist()
            contact_summary["self_contact_peak_excess"] = self_counts[3:6].tolist()
            contact_summary["self_detection_calls"] = int(self_counts[6])
            if (
                counts[0] > contacts.rigid_contact_max
                or counts[1] > contacts.soft_contact_max
                or any(counts[2:4])
                or any(self_counts[3:6])
            ):
                contact_summary["overflow"] = True
                raise RuntimeError(
                    f"Contact buffer overflow at {self.time:.12g} s: "
                    f"global={counts.tolist()}, self={self_counts.tolist()}"
                )

    runner.RunObserver = MainObserver

    def save(path, value):
        if path.name in ("results.json", "progress.json"):
            value = {**value, "math_mode": "off", "fast_math_enabled": False, "rigid_soft_enable_dat": True}
            if path.name == "results.json":
                value["task_measurements"] = task_result(tasks[-1]) if tasks else None
                value["dat_contact_coverage"] = dict(contact_summary)
        return save_json(path, value)

    runner.save_json = save
    arguments = [args.case, "--mode", "vbd", "--out", str(out), "--contact-max-ratio", "10"]
    for name in ("device", "hold_seconds", "video_fps", "width", "height"):
        arguments.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.duration is not None:
        arguments.extend(["--duration", str(args.duration)])
    if args.no_video:
        arguments.append("--no-video")
    shutil.copy2(__file__, out / "run_dat_ogc.py")
    sys.argv = [str(__file__), *arguments]
    try:
        runner.main()
    finally:
        provenance["source_guard_unchanged"] = all(sha256(Path(path)) == digest for path, digest in hashes.items())
        provenance["dat_contact_coverage"] = dict(contact_summary)
        if tasks:
            save_json(out / "task-observations.json", task_result(tasks[-1]))
        save_json(out / "dat-ogc-provenance.json", provenance)
        if not provenance["source_guard_unchanged"]:
            raise AssertionError("Source changed during the recorded run")


if __name__ == "__main__":
    main()
