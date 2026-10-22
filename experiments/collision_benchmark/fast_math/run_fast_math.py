# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one elasticity/math combination with the pinned PR4830 compiler scope.

Both arms use identical kernel bodies and module organization. Preparation is
math-off; the requested flag is installed only at the common task start.
"""

# ruff: noqa: PLC0415
# Keep solver and recipe imports after CLI parsing and before their runtime patch.

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import warp as wp

from .provenance import sha256, validate_output, verify_physics

HERE = Path(__file__).resolve().parent
PR_HEAD = "d1b848a043022ac0e55cee31d3117b599399f8c0"


def require_fingerprint(actual, expected):
    if actual != expected:
        fields = sorted(
            k
            for k in actual.get("fields", {}).keys() | expected.get("fields", {}).keys()
            if actual.get("fields", {}).get(k) != expected.get("fields", {}).get(k)
        )
        raise AssertionError(f"Initial fingerprint differs from reference: {fields}")


def task_result(observer):
    """Keep physical results writable if auxiliary evidence cannot serialize."""
    try:
        return observer.result()
    except Exception as exc:  # Preserve physical measurements if auxiliary evidence fails.
        return {"status": "observer_error", "error": f"{type(exc).__name__}: {exc}"}


class CompilerScope:
    """Keep unchanged kernel functions; isolate only the PR's hot modules."""

    def __init__(self):
        from newton._src.solvers.vbd import solver_vbd as solver

        self.solver_module = solver
        self.enabled = False
        self.tiles = {}
        self.constructed = []
        self.original_rigid_module = solver.rigid_vbd_kernels
        self.untreated = {
            "rigid_remaining": solver.rigid_vbd_kernels,
            "particle_alm_separate": solver.particle_alm_kernels,
            "coupling": solver.vbd_coupling_kernels,
        }
        self.untreated_initial = {key: wp.get_module_options(module).copy() for key, module in self.untreated.items()}
        self.rigid_solve = wp.kernel(
            solver.solve_rigid_body.func,
            module="unique",
            enable_backward=False,
            module_options={"enable_backward": False, "fast_math": False},
        )
        self.rigid_duals = wp.kernel(
            solver.update_duals_joint.func,
            module="unique",
            enable_backward=False,
            module_options={"enable_backward": False, "fast_math": False},
        )
        solver.solve_rigid_body = self.rigid_solve
        solver.update_duals_joint = self.rigid_duals
        solver.rigid_vbd_kernels.solve_rigid_body = self.rigid_solve
        solver.rigid_vbd_kernels.update_duals_joint = self.rigid_duals
        factory = solver.make_solve_elasticity_tile

        def make_tile(*args, **kwargs):
            kernel = factory(*args, **kwargs)
            self.tiles[kernel.module.name] = kernel
            wp.set_module_options({"fast_math": self.enabled}, module=kernel.module)
            return kernel

        solver.make_solve_elasticity_tile = make_tile
        self.set_math(False)

    def treated_modules(self):
        return {
            "particle_vbd": self.solver_module.particle_vbd_kernels,
            "rigid_solve": self.rigid_solve.module,
            "rigid_joint_duals": self.rigid_duals.module,
            **{"elasticity_tile:" + name: kernel.module for name, kernel in self.tiles.items()},
        }

    def set_math(self, enabled):
        self.enabled = bool(enabled)
        for module in self.treated_modules().values():
            wp.set_module_options({"fast_math": self.enabled}, module=module)

    def snapshot(self):
        return {
            key: {
                "module_name": wp.get_module(module.__name__).name
                if not isinstance(module, wp.Module)
                else module.name,
                "options": dict(wp.get_module_options(module)),
            }
            for key, module in {**self.treated_modules(), **self.untreated}.items()
        }

    def finish_constructor(self, solver, before):
        # Match the PR's constructor propagation into unique rigid modules.
        rigid_options = wp.get_module_options(self.original_rigid_module)
        for module in (self.rigid_solve.module, self.rigid_duals.module):
            solver._set_module_options(
                {
                    "deterministic": rigid_options["deterministic"],
                    "deterministic_max_records": rigid_options["deterministic_max_records"],
                    "enable_backward": False,
                    "fast_math": self.enabled,
                },
                module=module,
            )
        particle = self.solver_module.particle_vbd_kernels
        if solver.model.particle_count:
            options = dict(solver._module_options.get(particle, {}))
            solver._set_module_options({**options, "fast_math": self.enabled}, module=particle)
        tile = getattr(solver, "_tiled_elasticity_kernel", None)
        if tile is not None:
            solver._set_module_options(
                {"enable_backward": False, "fast_math": self.enabled},
                module=tile.module,
            )
        after = self.snapshot()
        for key in self.treated_modules():
            if after[key]["options"]["fast_math"] is not self.enabled:
                raise AssertionError(f"Compiler treatment was reset: {key}")
        for key in self.untreated:
            if after[key]["options"]["fast_math"] != self.untreated_initial[key]["fast_math"]:
                raise AssertionError(f"Fast math leaked outside the PR scope: {key}")
        self.constructed.append(
            {
                "fast_math": self.enabled,
                "particle_count": solver.model.particle_count,
                "body_count": solver.model.body_count,
                "before": before,
                "after": after,
            }
        )


def repair_duck(scene, runtime):
    from newton.solvers import SolverFeatherstone

    before = runtime.input_fingerprint(scene)
    example = scene.raw_scene.example
    old = example.robot_solver
    names = (
        "angular_damping",
        "update_mass_matrix_interval",
        "friction_smoothing",
        "use_tile_gemm",
        "fuse_cholesky",
    )
    config = {name: getattr(old, name) for name in names}
    new = SolverFeatherstone(example.model, update_mass_matrix_interval=example.sim_substeps)
    if {name: getattr(new, name) for name in names} != config:
        raise AssertionError("Fresh Featherstone constructor settings differ")
    if old._step % old.update_mass_matrix_interval != new._step % new.update_mass_matrix_interval:
        raise AssertionError("Featherstone mass-update phase differs")
    example.robot_solver = new
    require_fingerprint(runtime.input_fingerprint(scene), before)
    return {
        "config": config,
        "old_step": old._step,
        "new_step": new._step,
        "repair": "Fresh identical Featherstone constructor outside discarded constructor capture; no state change.",
    }


def package_provenance():
    distribution = importlib.metadata.distribution("warp-lang")
    files = {}
    for item in distribution.files or []:
        if str(item).endswith(("/RECORD", "/METADATA", "/warp.so")) or str(item) in (
            "warp/__init__.py",
            "warp/_src/context.py",
        ):
            path = Path(distribution.locate_file(item))
            if path.is_file():
                files[str(path)] = sha256(path)
    return {
        "version": wp.__version__,
        "distribution": "warp-lang",
        "files_sha256": files,
    }


def parse_arguments(argv=None):
    """Parse one complete run without importing scene factories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="Case ID from the bundled comparison registry")
    parser.add_argument("--mode", choices=("vbd", "alm"), required=True)
    parser.add_argument("--math", choices=("off", "on"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hold-seconds", type=float, default=60.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument(
        "--initial-reference", type=Path, help="Prior results.json whose full initial fingerprint must match"
    )
    parser.add_argument("--without-task-observer", action="store_true")
    return parser.parse_args(argv)


def runner_arguments(args):
    """Forward the unchanged simulation and recording arguments."""
    arguments = [args.case, "--mode", args.mode, "--out", str(args.out)]
    for name in ("device", "hold_seconds", "video_fps", "width", "height"):
        arguments.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.duration is not None:
        arguments.extend(["--duration", str(args.duration)])
    if args.no_video:
        arguments.append("--no-video")
    return arguments


def main(argv=None):
    """Run a single process; preserve all original solver and observer settings."""
    args = parse_arguments(argv)
    out = args.out.resolve()
    validate_output(out)
    reference_path = args.initial_reference.resolve() if args.initial_reference else None
    expected = json.loads(reference_path.read_text())["input_fingerprint"] if reference_path else None
    pr = json.loads((HERE / "pr_scope.json").read_text())
    if pr["head"] != PR_HEAD:
        raise AssertionError("PR head changed; audit compilation scope before running")
    from experiments.collision_benchmark import comparison_cases
    from experiments.collision_benchmark import comparison_runtime as runtime
    from experiments.collision_benchmark import run_comparison as runner

    manifest = json.loads((HERE / "physics_manifest.json").read_text())
    physics = verify_physics(runner.ROOT, manifest, warp_version=wp.__version__)
    revision = physics["revision"]
    guarded = set(Path(runtime.__file__).parent.rglob("*.py"))
    guarded.update(runner.ROOT / name for name in physics["files_sha256"])
    guarded.update((HERE / "physics_manifest.json", HERE / "pr_scope.json"))
    hashes = {str(path): sha256(path) for path in sorted(guarded)}
    out.mkdir(parents=True, exist_ok=True)
    scope = CompilerScope()
    treatment = args.math == "on"
    active_treatment = False
    task_observers = []
    provenance = {
        "schema_version": 1,
        "math_mode": args.math,
        "fast_math_enabled": treatment,
        "adapter_sha256": sha256(__file__),
        "newton_revision": revision,
        "pr_number": 4830,
        "pr_head": PR_HEAD,
        "pr_base": pr["base"],
        "pr_metadata_sha256": sha256(HERE / "pr_scope.json"),
        "pr_diff_sha256": pr["upstream_diff_sha256"],
        "physics_base_revision": physics["base_revision"],
        "physics_manifest_sha256": sha256(HERE / "physics_manifest.json"),
        "physics_source_sha256": physics["files_sha256"],
        "warp_package": package_provenance(),
        "guarded_source_sha256": hashes,
        "initial_reference": str(reference_path) if reference_path else None,
        "initial_reference_sha256": sha256(reference_path) if reference_path else None,
        "preparation": "Math-off, standard elasticity, frozen paired initialization/cache before treatment.",
        "kernel_scope": "particle_vbd module, generated elasticity tile, unique original rigid solve and joint-dual kernels only. Separate particle_alm, coupling and remaining rigid modules unchanged. Inlined ALM functions inside treated modules share their compilation flag.",
        "organization": "Identical native kernel bodies and unique rigid modules for math-off and math-on; no equation backport from the newer PR base.",
        "finite_guard": "Stop only after the frozen observer records the first nonfinite physical substep, identically in every arm.",
    }
    original_solver = runtime.ObservedSolverVBD

    class ScopedObservedSolver(original_solver):
        def __init__(self, model, **kwargs):
            scope.set_math(treatment if active_treatment else False)
            before = scope.snapshot()
            super().__init__(model, **kwargs)
            scope.finish_constructor(self, before)

    runtime.ObservedSolverVBD = ScopedObservedSolver
    original_build = comparison_cases.build_case

    def build(case, **kwargs):
        nonlocal expected
        scene = original_build(case, **kwargs)
        initial = runtime.input_fingerprint(scene)
        if expected is None:
            expected = initial
        else:
            require_fingerprint(initial, expected)
        if case == "r2_franka_duck":
            provenance["duck_runtime_adapter"] = repair_duck(scene, runtime)
        require_fingerprint(runtime.input_fingerprint(scene), initial)
        provenance["initial_fingerprint"] = initial
        provenance["common_start"] = getattr(scene.raw_scene, "comparison_start", None)
        scene.protocol["fast_math_adapter"] = {
            "adapter_sha256": provenance["adapter_sha256"],
            "pr_head": PR_HEAD,
            "same_kernel_organization_both_math_modes": True,
            "finite_guard": provenance["finite_guard"],
            "duck_repair": case == "r2_franka_duck",
        }
        return scene

    comparison_cases.build_case = build
    original_replace = runtime.replace_solver

    def replace(scene, mode):
        nonlocal active_treatment
        active_treatment = True
        scope.set_math(treatment)
        solver = original_replace(scene, mode)
        require_fingerprint(runtime.input_fingerprint(scene), expected)
        provenance["effective_options_at_task_start"] = scope.snapshot()
        runner.save_json(out / "fast-math-provenance.json", provenance)
        return solver

    runtime.replace_solver = replace
    task_type = None
    if not args.without_task_observer:
        # The task-side observer never changes the physical state or calls collide.
        from .task_observer import TaskObserver

        task_type = TaskObserver
        observer_source = HERE / "task_observer.py"
        provenance["task_observer_sha256"] = sha256(observer_source)
        hashes[str(observer_source)] = provenance["task_observer_sha256"]
        shutil.copy2(observer_source, out / "task_observer.py")
    base_observer = runner.RunObserver

    class GuardedRunObserver(base_observer):
        def __init__(self, scene, observer):
            super().__init__(scene, observer)
            self.task = task_type(scene) if task_type else None
            if self.task:
                task_observers.append(self.task)

        def after_substep(self, state, dt):
            super().after_substep(state, dt)
            # Pinned frozen observer exposes this already-computed flag. No second reduction.
            bad_time = self.observer._first_nonfinite
            if bad_time is not None:
                raise FloatingPointError(f"Nonfinite physical state recorded at {bad_time:.12g} s")
            if self.task:
                self.task.sample(state, self.time, self.scene.phase)

    runner.RunObserver = GuardedRunObserver
    save_json = runner.save_json

    def save(path, value):
        if path.name in ("results.json", "progress.json"):
            value = {
                **value,
                "math_mode": args.math,
                "fast_math_enabled": treatment,
                "fast_math_adapter_sha256": provenance["adapter_sha256"],
                "pr4830_head": PR_HEAD,
            }
            if path.name == "results.json":
                value["compiler_effective_options"] = scope.snapshot()
                value["task_measurements"] = task_result(task_observers[-1]) if task_observers else None
        return save_json(path, value)

    runner.save_json = save
    shutil.copy2(__file__, out / "run_fast_math.py")
    sys.argv = [str(__file__), *runner_arguments(args)]
    try:
        runner.main()
    finally:
        provenance["constructors"] = scope.constructed
        provenance["final_effective_options"] = scope.snapshot()
        provenance["source_guard_unchanged"] = all(sha256(Path(path)) == digest for path, digest in hashes.items())
        if task_observers:
            task = task_result(task_observers[-1])
            save_json(out / "task-observations.json", task)
            if task.get("status") == "observer_error":
                provenance["task_observer_error"] = task["error"]
        save_json(out / "fast-math-provenance.json", provenance)
        if not provenance["source_guard_unchanged"]:
            raise AssertionError("Frozen source changed during this run")
    if provenance.get("task_observer_error"):
        raise RuntimeError("Physical results preserved; task observer failed: " + provenance["task_observer_error"])


if __name__ == "__main__":
    main()
