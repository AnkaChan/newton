# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run reproducible Newton/SolverVBD baselines on audited elastic scenes.

Each selected scene uses the exact default state from :mod:`solver_scenes`.
The runner solves one common implicit objective with the dense CPU Newton
reference, restarts SolverVBD from that same state for every iteration budget,
and writes the existing self-checking JSON/NPZ bundle. A compact, self-hashed
index binds the generated bundles together.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import hashlib
import json
import pathlib
import sys
from collections.abc import Callable, Sequence

from .solver_benchmark import (
    TetBenchmarkScene,
    build_common_problem,
    run_newton,
    run_vbd,
    write_benchmark_bundle,
)
from .solver_scenes import (
    build_compression_scene,
    build_extension_scene,
    build_refinement_scene,
    build_sliver_scene,
    build_stretch_scene,
    build_twist_scene,
)

_INDEX_SCHEMA_VERSION = 1
_DEFAULT_VBD_ITERATIONS = (1, 2, 4, 8, 16, 32)
_REFINEMENT_KEYS = ("refinement-coarse", "refinement-medium", "refinement-fine")
_DEFAULT_SCENE_KEYS = (
    "extension",
    "stretch",
    "twist",
    "compression",
    "sliver",
    "refinement-coarse",
    "refinement-medium",
)

_SCENE_FACTORIES: dict[str, Callable[[], TetBenchmarkScene]] = {
    "extension": build_extension_scene,
    "stretch": build_stretch_scene,
    "twist": build_twist_scene,
    "compression": build_compression_scene,
    "sliver": build_sliver_scene,
    "refinement-coarse": functools.partial(build_refinement_scene, "coarse"),
    "refinement-medium": functools.partial(build_refinement_scene, "medium"),
    "refinement-fine": functools.partial(build_refinement_scene, "fine"),
}


@dataclasses.dataclass(frozen=True)
class SolverSceneRunConfig:
    """Configuration for one multi-scene baseline invocation."""

    output_dir: pathlib.Path
    scene_selectors: tuple[str, ...] = _DEFAULT_SCENE_KEYS
    device: str = "cpu"
    tile_solve: bool = False
    repeats: int = 5
    vbd_iterations: tuple[int, ...] = _DEFAULT_VBD_ITERATIONS
    max_newton_free_dofs: int = 2_000

    def validate(self) -> None:
        """Reject ambiguous or unsafe benchmark configurations."""
        if not self.scene_selectors:
            raise ValueError("at least one scene must be selected")
        _resolve_scene_keys(self.scene_selectors)
        if not self.device:
            raise ValueError("device must not be empty")
        if not isinstance(self.tile_solve, bool):
            raise ValueError("tile_solve must be a bool")
        if not isinstance(self.repeats, int) or isinstance(self.repeats, bool) or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        _normalize_vbd_iterations(self.vbd_iterations)
        if (
            not isinstance(self.max_newton_free_dofs, int)
            or isinstance(self.max_newton_free_dofs, bool)
            or self.max_newton_free_dofs < 1
        ):
            raise ValueError("max_newton_free_dofs must be a positive integer")


def _resolve_scene_keys(selectors: Sequence[str]) -> tuple[str, ...]:
    """Expand selector aliases and remove duplicates without changing order."""
    expanded: list[str] = []
    for selector in selectors:
        if selector == "all":
            values = tuple(_SCENE_FACTORIES)
        elif selector == "refinement":
            values = _REFINEMENT_KEYS
        elif selector in _SCENE_FACTORIES:
            values = (selector,)
        else:
            choices = ("all", "refinement", *_SCENE_FACTORIES)
            raise ValueError(f"unknown scene {selector!r}; expected one of {choices}")
        for value in values:
            if value not in expanded:
                expanded.append(value)
    if not expanded:
        raise ValueError("at least one scene must be selected")
    return tuple(expanded)


def _normalize_vbd_iterations(values: Sequence[int]) -> tuple[int, ...]:
    """Return a canonical ascending sequence of distinct positive budgets."""
    iterations = tuple(values)
    if not iterations:
        raise ValueError("at least one VBD iteration budget is required")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in iterations):
        raise ValueError("VBD iteration budgets must be positive integers")
    if len(set(iterations)) != len(iterations):
        raise ValueError("VBD iteration budgets must be distinct")
    return tuple(sorted(iterations))


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _planned_paths(output_dir: pathlib.Path, scene_keys: Sequence[str]) -> tuple[pathlib.Path, ...]:
    paths = [output_dir / "index.json"]
    for key in scene_keys:
        paths.extend((output_dir / f"{key}.json", output_dir / f"{key}.npz"))
    return tuple(paths)


def _assert_output_paths_available(output_dir: pathlib.Path, scene_keys: Sequence[str]) -> None:
    existing = [path for path in _planned_paths(output_dir, scene_keys) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing benchmark outputs: {joined}")


def _validate_bundle(
    bundle_path: pathlib.Path,
    raw_path: pathlib.Path,
    expected_scene_sha256: str,
) -> dict[str, object]:
    if not bundle_path.is_file() or not raw_path.is_file():
        raise RuntimeError(f"benchmark writer did not create {bundle_path.name} and {raw_path.name}")
    payload = json.loads(bundle_path.read_text())
    if payload["scene"]["scene_sha256"] != expected_scene_sha256:
        raise RuntimeError(f"bundle {bundle_path.name} contains the wrong scene hash")
    raw_record = payload["raw_npz"]
    if raw_record["path"] != raw_path.name:
        raise RuntimeError(f"bundle {bundle_path.name} points to an unexpected raw archive")
    actual_raw_sha256 = _file_sha256(raw_path)
    if raw_record["sha256"] != actual_raw_sha256:
        raise RuntimeError(f"bundle {bundle_path.name} raw archive hash does not match")
    return payload


def run_solver_scenes(config: SolverSceneRunConfig) -> pathlib.Path:
    """Run all selected baselines and return the generated index path.

    Scene construction and the dense free-DOF ceiling are preflighted before
    creating the output directory or invoking either solver.
    """
    config.validate()
    scene_keys = _resolve_scene_keys(config.scene_selectors)
    iterations = _normalize_vbd_iterations(config.vbd_iterations)

    scenes: list[tuple[str, TetBenchmarkScene, int]] = []
    oversized: list[tuple[str, int]] = []
    for key in scene_keys:
        # Factories are intentionally called without overrides: their audited
        # first-increment/default state is part of the benchmark definition.
        scene = _SCENE_FACTORIES[key]()
        free_dofs = int(scene.free_indices.size * 3)
        scenes.append((key, scene, free_dofs))
        if free_dofs > config.max_newton_free_dofs:
            oversized.append((key, free_dofs))
    if oversized:
        details = ", ".join(f"{key}={free_dofs}" for key, free_dofs in oversized)
        raise ValueError(
            f"dense Newton free-DOF ceiling {config.max_newton_free_dofs} exceeded: {details}; "
            "select a smaller scene or explicitly raise --max-newton-free-dofs"
        )

    output_dir = pathlib.Path(config.output_dir).resolve()
    _assert_output_paths_available(output_dir, scene_keys)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_scenes: list[dict[str, object]] = []
    for key, scene, free_dofs in scenes:
        print(
            f"[{key}] {scene.n_vertices} vertices, {scene.n_tets} tets, {free_dofs} free DOFs",
            file=sys.stderr,
            flush=True,
        )
        problem = build_common_problem(scene)
        newton_run = run_newton(scene, problem, warmup=True, repeats=config.repeats)
        if not newton_run.reference_accepted:
            raise RuntimeError(f"{key}: Newton reference gates failed: {newton_run.reference_failures}")
        vbd_results = [
            run_vbd(
                scene,
                budget,
                device=config.device,
                tile_solve=config.tile_solve,
                warmup=True,
                repeats=config.repeats,
            )
            for budget in iterations
        ]

        bundle_path = output_dir / f"{key}.json"
        raw_path = output_dir / f"{key}.npz"
        write_benchmark_bundle(bundle_path, scene, problem, newton_run, vbd_results)
        scene_sha256 = str(scene.manifest()["scene_sha256"])
        _validate_bundle(bundle_path, raw_path, scene_sha256)
        index_scenes.append(
            {
                "key": key,
                "name": scene.name,
                "vertices": scene.n_vertices,
                "tetrahedra": scene.n_tets,
                "free_dofs": free_dofs,
                "scene_sha256": scene_sha256,
                "bundle": {
                    "path": bundle_path.name,
                    "sha256": _file_sha256(bundle_path),
                },
                "raw_npz": {
                    "path": raw_path.name,
                    "sha256": _file_sha256(raw_path),
                },
            }
        )

    index: dict[str, object] = {
        "schema_version": _INDEX_SCHEMA_VERSION,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "runner": "research.principal_stretch.run_solver_scenes",
        "configuration": {
            "scene_keys": list(scene_keys),
            "device": config.device,
            "tile_solve": config.tile_solve,
            "repeats": config.repeats,
            "vbd_iterations": list(iterations),
            "max_newton_free_dofs": config.max_newton_free_dofs,
            "scene_parameters": "audited builder defaults",
            "newton_baseline_records_per_scene": 1,
            "vbd_restart_contract": "fresh identical physical state for each iteration budget and repeat",
        },
        "scenes": index_scenes,
    }
    index["index_sha256"] = _json_sha256(index)
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return index_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=_DEFAULT_SCENE_KEYS,
        metavar="SCENE",
        help=(
            "scene keys; use 'refinement' for all refinement levels or 'all' for every key "
            f"(default: {' '.join(_DEFAULT_SCENE_KEYS)})"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tile-solve", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--vbd-iterations",
        "--budgets",
        dest="vbd_iterations",
        nargs="+",
        type=int,
        default=_DEFAULT_VBD_ITERATIONS,
    )
    parser.add_argument(
        "--max-newton-free-dofs",
        type=int,
        default=2_000,
        help="refuse dense Newton above this free-DOF count (default: 2000)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = SolverSceneRunConfig(
        output_dir=args.output_dir,
        scene_selectors=tuple(args.scenes),
        device=args.device,
        tile_solve=args.tile_solve,
        repeats=args.repeats,
        vbd_iterations=tuple(args.vbd_iterations),
        max_newton_free_dofs=args.max_newton_free_dofs,
    )
    try:
        index_path = run_solver_scenes(config)
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
