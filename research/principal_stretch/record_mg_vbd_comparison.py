# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Quantify, store, and render small MG-VBD trajectory comparisons.

This module deliberately separates simulation from visualization.  The
``generate`` command advances three history-bearing solvers, evaluates every
stored state with the independent common-objective evaluator, and seals the
arrays and provenance into content-addressed NPZ/JSON files.  The ``render``
command accepts only a verified bundle and turns it into a fixed-camera MP4.

The left lane is a high-iteration public :class:`newton.solvers.SolverVBD`
trajectory.  It is a numerical comparison reference, not Newton and not
ground truth.  Authenticated dense/sparse Newton endpoints may be attached to
the medium-beam JSON, but only as static first-substep quantitative evidence;
they are never repeated to manufacture a reference animation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .correction_graph_vbd import DirectGraphVBDConfig
from .gaia_assets import GAIA_ASSETS, GAIA_REPOSITORY_URL, GAIA_SOURCE_REVISION, build_registered_gaia_scene
from .solver_benchmark import (
    CommonStateMetrics,
    TetBenchmarkScene,
    _build_vbd_model,
    build_common_problem,
    common_objective_manifest,
    evaluate_common_state,
)
from .solver_scenes import build_refinement_scene, build_twist_scene

SCHEMA_V1 = "principal-stretch-mg-vbd-recording-v1"
SCHEMA = "principal-stretch-mg-vbd-recording-v2"
STATIC_REFERENCE_SCHEMA = "authenticated-medium-newton-endpoints-v2"
GAIA_ASSET_BUNDLE_SCHEMA = "principal-stretch-gaia-asset-bundle-v1"
GENERATION_SOURCE_SCHEMA_V2 = "principal-stretch-mg-vbd-generation-source-v2"
GENERATION_SOURCE_SCHEMA = "principal-stretch-mg-vbd-generation-source-v3"
RENDER_SOURCE_SCHEMA = "principal-stretch-mg-vbd-render-source-v1"
RENDER_RECORD_SCHEMA = "principal-stretch-mg-vbd-render-record-v1"
GENERATION_RECEIPT_SCHEMA = "principal-stretch-mg-vbd-process-local-generation-receipt-v1"
METHOD_IDS = ("reference", "mg_vbd", "vbd_k4")
METRIC_NAMES = (
    "objective",
    "gradient_norm",
    "relative_residual",
    "determinant_min",
    "inverted_tet_fraction",
    "free_rms_error_m",
    "mass_weighted_rms_error_m",
    "max_pin_error_m",
)
_COMMON_METRIC_NAMES = (
    "objective",
    "inertia",
    "elastic",
    "gradient_norm",
    "relative_residual",
    "determinant_min",
    "determinant_max",
    "inverted_tet_fraction",
    "minimum_singular_value",
    "free_rms_error_m",
    "mass_weighted_rms_error_m",
    "max_pin_error_m",
    "position_sha256",
)
_GENERATION_SOURCE_PATHS_V2 = (
    "research/principal_stretch/record_mg_vbd_comparison.py",
    "research/principal_stretch/mg_vbd_rollout.py",
    "research/principal_stretch/captured_graph_vbd.py",
    "research/principal_stretch/correction_graph_vbd.py",
    "research/principal_stretch/correction_mg_vbd.py",
    "research/principal_stretch/correction_multigrid.py",
    "research/principal_stretch/solver_benchmark.py",
    "research/principal_stretch/solver_scenes.py",
    "research/principal_stretch/newton_baseline.py",
    "research/principal_stretch/potentials.py",
    "research/principal_stretch/captured_mg_vbd.py",
    "research/principal_stretch/captured_vbd_baseline.py",
    "research/principal_stretch/correction_gpu.py",
    "research/principal_stretch/correction_gpu_warp.py",
    "research/principal_stretch/correction_multigrid_warp.py",
    "research/principal_stretch/correction_multigrid_warp_scalar_fused.py",
    "research/principal_stretch/polar.py",
    "research/principal_stretch/sparse_newton_reference.py",
    "research/principal_stretch/torch_solver.py",
)
_GENERATION_SOURCE_PATHS = (
    "research/principal_stretch/__init__.py",
    *_GENERATION_SOURCE_PATHS_V2,
    "research/principal_stretch/gaia_assets.py",
)
_STATIC_REFERENCE_METHODS = {
    "dense": ("dense-cpu-newton-float64", "fresh-dense-newton-accepted-reference-v1"),
    "sparse": ("sparse-exact-cpu-newton-float64", "fresh-sparse-exact-newton-accepted-reference-v2"),
}
_NEWTON_CONFIG_NAMES = frozenset(
    {
        "max_iterations",
        "gradient_absolute_tolerance",
        "gradient_relative_tolerance",
        "step_relative_tolerance",
        "armijo",
        "backtrack",
        "max_line_search_steps",
        "minimum_eigenvalue_relative",
        "regularization_growth",
        "max_regularization_attempts",
    }
)
_DENSE_SOURCE_RECORD_NAMES = frozenset(
    {
        "contract",
        "method",
        "config",
        "scene_sha256",
        "objective_instance_sha256",
        "accepted",
        "failures",
        "native_converged",
        "native_reason",
        "accepted_iterations",
        "final_objective",
        "final_gradient_norm",
        "final_relative_residual",
        "verification_converged",
        "verification_reason",
        "verification_displacement_relative",
        "alternate_start_converged",
        "alternate_start_reason",
        "alternate_start_displacement_relative",
        "position_sha256",
    }
)
_SPARSE_SOURCE_RECORD_NAMES = frozenset(
    {
        *_DENSE_SOURCE_RECORD_NAMES,
        "alternate_start_gradient_norm",
        "alternate_start_relative_residual",
        "repeat_count",
        "repeat_deterministic_sha256",
        "native_result",
        "verification_result",
        "alternate_start_result",
    }
)
_SPARSE_RESULT_NAMES = frozenset(
    {
        "contract",
        "hessian_contract",
        "linear_solver",
        "eigen_policy",
        "factor_certificate",
        "factor_equilibration",
        "factor_unit_diagonal_limit",
        "factor_pivot_relative_margin",
        "factor_relation_relative_limit",
        "factorization_relative_residual_limit",
        "linear_residual_limit",
        "maximum_refinement_steps",
        "positions_sha256",
        "converged",
        "reason",
        "accepted_iterations",
        "residual_scale",
        "scipy_version",
        "final_objective",
        "final_gradient_norm",
        "final_relative_residual",
        "work",
        "trace",
    }
)
_SPARSE_WORK_NAMES = frozenset(
    {
        "objective_evaluations",
        "gradient_evaluations",
        "hessian_evaluations",
        "eigenvalue_evaluations",
        "factorization_attempts",
        "factor_certificate_attempts",
        "linear_solve_attempts",
        "line_search_trials",
    }
)
_SPARSE_TRACE_NAMES = frozenset(
    {
        "iteration",
        "objective",
        "gradient_norm",
        "relative_residual",
        "minimum_determinant",
        "hessian_nnz",
        "minimum_eigenvalue",
        "eigenpair_residual",
        "diagonal_scale",
        "ritz_regularization",
        "gershgorin_lower_bound",
        "gershgorin_rescue_regularization",
        "gershgorin_rescue_used",
        "regularization",
        "last_attempted_regularization",
        "factor_nnz",
        "factorization_attempts",
        "factor_certificate_attempts",
        "linear_solve_attempts",
        "linear_refinement_steps",
        "line_search_trials",
        "factor_permutations_match",
        "factor_l_unit_diagonal_error",
        "factor_minimum_diagonal",
        "factor_maximum_diagonal_magnitude",
        "factor_minimum_diagonal_relative",
        "factor_relation_relative_residual",
        "factorization_relative_residual",
        "factor_certificate_passed",
        "linear_relative_residual",
        "directional_derivative",
        "accepted_step_norm",
        "accepted_step_size",
    }
)
_STATIC_COMPARISON_NAMES = (
    "free_max_m",
    "free_rms_m",
    "gradient_delta_N_sparse_minus_dense",
    "objective_delta_sparse_minus_dense",
)
_NON_METRIC_ARRAY_NAMES = (
    "positions",
    "velocities",
    "objective_input_positions",
    "objective_input_velocities",
    "pin_targets",
    "time_seconds",
    "source_frame_index",
    "solve_seconds",
    "transfer_seconds",
    "mg_last_gate_accepted",
    "mg_last_gate_reason_code",
    "mg_frame_gate_accept_count",
)
_ARRAY_NAMES = frozenset((*_NON_METRIC_ARRAY_NAMES, *(f"metric_{name}" for name in METRIC_NAMES)))
_UNSEALED_RECORD_NAMES_V1 = frozenset(
    {
        "schema",
        "scene_key",
        "scene_display_name",
        "scene_manifest",
        "scene_physical_sha256",
        "git_revision",
        "generation_source",
        "methods",
        "method_order",
        "reference_policy",
        "simulation",
        "metrics",
        "mg_gate_reason_names",
        "setup_seconds_diagnostic",
        "device",
        "camera",
        "static_first_step_newton_reference",
    }
)
_UNSEALED_RECORD_NAMES = frozenset({*_UNSEALED_RECORD_NAMES_V1, "gaia_asset_bundle", "execution_authentication"})
_SEALED_RECORD_NAMES_V1 = frozenset(
    {
        *_UNSEALED_RECORD_NAMES_V1,
        "npz_filename",
        "npz_file_sha256",
        "arrays",
        "record_sha256",
    }
)
_SEALED_RECORD_NAMES = frozenset(
    {
        *_UNSEALED_RECORD_NAMES,
        "npz_filename",
        "npz_file_sha256",
        "arrays",
        "record_sha256",
    }
)


@dataclasses.dataclass(frozen=True)
class RecordingSpec:
    """One audited source-scene schedule used by the recording proof."""

    key: str
    display_name: str
    substeps_per_source_frame: int
    reference_iterations: int
    camera_direction: tuple[float, float, float]


_SPECS = {
    "refinement-medium": RecordingSpec(
        key="refinement-medium",
        display_name="PR #2901 refinement: medium hanging beam",
        substeps_per_source_frame=6,
        reference_iterations=100,
        camera_direction=(1.0, -2.0, 0.25),
    ),
    "twist": RecordingSpec(
        key="twist",
        display_name="PR #2901 twist ramp",
        substeps_per_source_frame=5,
        reference_iterations=20,
        camera_direction=(1.4, -2.0, 0.35),
    ),
    "gaia-bunny-small": RecordingSpec(
        key="gaia-bunny-small",
        display_name="Gaia bunny_small: static support under gravity and tip load",
        substeps_per_source_frame=6,
        reference_iterations=40,
        camera_direction=(1.25, -1.8, 0.55),
    ),
    "gaia-armadilo-lowres": RecordingSpec(
        key="gaia-armadilo-lowres",
        display_name="Gaia Armadilo_lowres: static support under gravity and tip load",
        substeps_per_source_frame=6,
        reference_iterations=40,
        camera_direction=(1.4, -1.8, 0.35),
    ),
}
_GAIA_SCENE_ASSETS: Mapping[str, tuple[str, float]] = types.MappingProxyType(
    {
        "gaia-bunny-small": ("bunny_small", 0.1),
        "gaia-armadilo-lowres": ("Armadilo_lowres", 1.0),
    }
)
_GAIA_BUNDLE_ROOT_RELATIVE = f"gaia-{GAIA_SOURCE_REVISION[:10]}"
_GAIA_BUNDLE_FILES: Mapping[str, tuple[int, str, str]] = types.MappingProxyType(
    {
        "LICENSE": (
            11357,
            "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
            "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
        ),
        "Data/mesh_models/t/bunny_small.t": (
            227758,
            "5052f098fd0eba9efa20c6dbb4a8915f50df09948a4b9d438a44976e86f9b746",
            "200198bea3ceb65dad808d5c7058ff6913008772",
        ),
        "Data/mesh_models/t/Armadilo_lowres.t": (
            584313,
            "6226e096aa61f27ec4de582fcf82d834bf2647bbfcbaefb0ba9c320d99809644",
            "15553dbe3eaabd1a946beb7df1389ca9b3423567",
        ),
    }
)


def recording_spec(scene_key: str) -> RecordingSpec:
    """Return the immutable recording policy for ``scene_key``."""
    try:
        return _SPECS[scene_key]
    except KeyError as error:
        raise ValueError(f"scene must be one of {tuple(_SPECS)}") from error


def _resolve_gaia_asset_root(value: str | os.PathLike[str] | None) -> pathlib.Path:
    raw = os.environ.get("PSS_GAIA_ASSET_ROOT") if value is None else value
    if raw is None or not str(raw):
        raise ValueError("Gaia recording scenes require --gaia-asset-root or PSS_GAIA_ASSET_ROOT")
    root = pathlib.Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Gaia asset root is not a directory: {root}")
    return root


def _gaia_asset_bundle_manifest(scene_key: str) -> dict[str, object] | None:
    if scene_key not in _GAIA_SCENE_ASSETS:
        return None
    asset_name, unit_scale = _GAIA_SCENE_ASSETS[scene_key]
    payload: dict[str, object] = {
        "contract": GAIA_ASSET_BUNDLE_SCHEMA,
        "source_repository_url": GAIA_REPOSITORY_URL,
        "source_revision": GAIA_SOURCE_REVISION,
        "bundle_root_relative_path": _GAIA_BUNDLE_ROOT_RELATIVE,
        "selected_asset_name": asset_name,
        "selected_asset_relative_path": GAIA_ASSETS[asset_name].relative_path,
        "unit_scale_m_per_source_unit": unit_scale,
        "files": {
            relative: {"bytes": size, "sha256": sha256, "git_blob_oid": blob_oid}
            for relative, (size, sha256, blob_oid) in _GAIA_BUNDLE_FILES.items()
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_gaia_asset_bundle_manifest(value: object, scene_key: str) -> None:
    expected = _gaia_asset_bundle_manifest(scene_key)
    if _canonical_json(value) != _canonical_json(expected):
        raise ValueError("recording Gaia asset bundle manifest is not canonical")


def _copy_gaia_asset_bundle(source_root: pathlib.Path, output_directory: pathlib.Path) -> pathlib.Path:
    destination_root = output_directory / _GAIA_BUNDLE_ROOT_RELATIVE
    if destination_root.exists():
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise RuntimeError("existing Gaia bundle root must be a real directory")
    else:
        destination_root.mkdir()
    for relative, (expected_size, expected_sha256, _blob_oid) in _GAIA_BUNDLE_FILES.items():
        source = source_root / pathlib.PurePosixPath(relative)
        if not source.is_file() or source.stat().st_size != expected_size or _file_sha256(source) != expected_sha256:
            raise ValueError(f"Gaia bundle source file differs from its pinned bytes: {relative}")
        destination = destination_root / pathlib.PurePosixPath(relative)
        parent = destination_root
        for component in pathlib.PurePosixPath(relative).parts[:-1]:
            parent /= component
            if parent.exists():
                if parent.is_symlink() or not parent.is_dir():
                    raise RuntimeError(f"existing Gaia bundle directory must not redirect: {relative}")
            else:
                parent.mkdir()
        if destination.exists():
            if destination.is_symlink():
                raise RuntimeError(f"existing Gaia bundle file must not be a symlink: {relative}")
            if destination.stat().st_size != expected_size or _file_sha256(destination) != expected_sha256:
                raise RuntimeError(f"existing Gaia bundle file differs from its pinned bytes: {relative}")
            continue
        with tempfile.NamedTemporaryFile(prefix=".gaia-asset-", dir=destination.parent, delete=False) as stream:
            temporary = pathlib.Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            if temporary.stat().st_size != expected_size or _file_sha256(temporary) != expected_sha256:
                raise RuntimeError(f"copied Gaia bundle file differs from its pinned bytes: {relative}")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return destination_root


def _verify_gaia_asset_bundle_files(root: pathlib.Path) -> None:
    for relative, (expected_size, expected_sha256, _blob_oid) in _GAIA_BUNDLE_FILES.items():
        path = root
        redirected = False
        for component in pathlib.PurePosixPath(relative).parts:
            path /= component
            redirected = redirected or path.is_symlink()
        if (
            redirected
            or not path.is_file()
            or path.stat().st_size != expected_size
            or _file_sha256(path) != expected_sha256
        ):
            raise ValueError(f"Gaia asset bundle file differs from its pinned bytes: {relative}")


def _bundle_gaia_asset_root(
    scene_key: object,
    bundle_directory: pathlib.Path,
    explicit_root: str | os.PathLike[str] | None,
) -> str | os.PathLike[str] | None:
    if type(scene_key) is not str or scene_key not in _GAIA_SCENE_ASSETS or explicit_root is not None:
        return explicit_root
    bundled = bundle_directory / _GAIA_BUNDLE_ROOT_RELATIVE
    if bundled.is_symlink():
        raise ValueError("packaged Gaia bundle root must not redirect outside the recording directory")
    return bundled if bundled.is_dir() else None


def build_recording_scene(
    scene_key: str,
    *,
    gaia_asset_root: str | os.PathLike[str] | None = None,
) -> TetBenchmarkScene:
    """Build the rest-state scene used to start a recording.

    Args:
        scene_key: Canonical recording scene key.
        gaia_asset_root: Gaia checkout root for a Gaia scene. Uses
            ``PSS_GAIA_ASSET_ROOT`` when omitted.
    """
    recording_spec(scene_key)
    if scene_key == "refinement-medium":
        if gaia_asset_root is not None:
            raise ValueError("gaia_asset_root is valid only for Gaia recording scenes")
        return build_refinement_scene("medium")
    if scene_key == "twist":
        if gaia_asset_root is not None:
            raise ValueError("gaia_asset_root is valid only for Gaia recording scenes")
        return build_twist_scene(twist_angle=0.0, one_shot_diagnostic=True)
    asset_name, unit_scale = _GAIA_SCENE_ASSETS[scene_key]
    return build_registered_gaia_scene(
        asset_name,
        _resolve_gaia_asset_root(gaia_asset_root),
        unit_scale_m_per_source_unit=unit_scale,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    dtype = array.dtype if array.dtype.byteorder == "|" else array.dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=dtype)


def array_sha256(value: np.ndarray) -> str:
    """Hash an array together with its canonical dtype and shape."""
    array = _canonical_array(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _execution_authentication_policy() -> dict[str, object]:
    return {
        "contract": GENERATION_RECEIPT_SCHEMA,
        "producer_process_receipt_required_at_seal": True,
        "receipt_is_one_shot": True,
        "offline_cryptographic_attestation": False,
        "transition_replay_at_load": False,
        "claim_scope": "process-local accidental producer mixup detection at generation-to-seal boundary",
    }


def _trajectory_content_sha256(
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> str:
    payload = {
        "metadata": metadata,
        "arrays": {
            name: {
                "dtype": array.dtype.name,
                "shape": list(array.shape),
                "array_sha256": array_sha256(array),
            }
            for name, array in sorted(arrays.items())
        },
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _make_generated_trajectory_registry():
    """Create one process-local, identity-bound generation receipt registry."""
    registry: dict[int, tuple[Mapping[str, object], Mapping[str, np.ndarray], str, bool]] = {}
    lock = threading.Lock()

    def issue(metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> None:
        content_sha256 = _trajectory_content_sha256(metadata, arrays)
        with lock:
            key = id(metadata)
            if key in registry:
                raise RuntimeError("trajectory metadata already has an unconsumed generation receipt")
            # Strong references prevent Python from reusing either identity.
            registry[key] = (metadata, arrays, content_sha256, False)

    def consume(
        metadata: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
        frozen_metadata: Mapping[str, object],
        canonical_arrays: Mapping[str, np.ndarray],
    ) -> None:
        with lock:
            registered = registry.get(id(metadata))
            if registered is None or registered[0] is not metadata or registered[1] is not arrays:
                raise RuntimeError("recording was not returned by this process's exact generation run")
            if registered[3]:
                raise RuntimeError("recording generation receipt is already reserved by a seal attempt")
            if _trajectory_content_sha256(frozen_metadata, canonical_arrays) != registered[2]:
                raise RuntimeError("recording changed after its process-local generation receipt was issued")
            registry[id(metadata)] = (*registered[:3], True)

    def finalize(metadata: Mapping[str, object], *, success: bool) -> None:
        with lock:
            registered = registry.get(id(metadata))
            if registered is None or registered[0] is not metadata or not registered[3]:
                raise RuntimeError("recording generation receipt reservation is not live")
            if success:
                del registry[id(metadata)]
            else:
                registry[id(metadata)] = (*registered[:3], False)

    return issue, consume, finalize


_issue_generated_trajectory, _consume_generated_trajectory, _finalize_generated_trajectory = (
    _make_generated_trajectory_registry()
)
del _make_generated_trajectory_registry


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _git_revision(repository: pathlib.Path | None = None) -> str | None:
    root = pathlib.Path(__file__).resolve().parents[2] if repository is None else repository.resolve()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _require_exact_keys(value: object, expected: set[str] | frozenset[str], *, name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(expected):
        actual = set(value) if type(value) is dict else set()
        raise ValueError(
            f"{name} keys differ: missing={sorted(set(expected) - actual)}, extra={sorted(actual - set(expected))}"
        )
    return value


def _require_finite_number(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_git_revision(value: object, *, name: str = "git revision") -> str:
    if type(value) is not str or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase 40-character Git revision")
    return value


def _require_git_oid(value: object, *, object_format: str, name: str) -> str:
    lengths = {"sha1": 40, "sha256": 64}
    if object_format not in lengths:
        raise ValueError(f"unsupported Git object format {object_format!r}")
    length = lengths[object_format]
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase {object_format} object ID")
    return value


def _run_git(repository: pathlib.Path, arguments: list[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if binary else result.stderr
        raise RuntimeError(f"git {' '.join(arguments)} failed: {stderr.strip()}")
    return result.stdout


def _version_string(module: object, distribution: str) -> str:
    value = getattr(module, "__version__", None)
    if type(value) is str and value:
        return value
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"cannot resolve installed {distribution!r} version") from error


def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"rendering requires the {distribution!r} package") from error


def _generation_source_manifest(repository: pathlib.Path | None = None) -> dict[str, object]:
    """Bind generated trajectories to clean, committed solver source bytes."""
    root = pathlib.Path(__file__).resolve().parents[2] if repository is None else repository.resolve()
    object_format = str(_run_git(root, ["rev-parse", "--show-object-format"])).strip()
    revision = _require_git_oid(_git_revision(root), object_format=object_format, name="generation source Git revision")
    committed_revision = str(_run_git(root, ["rev-parse", "--verify", f"{revision}^{{commit}}"])).strip()
    if committed_revision != revision:
        raise RuntimeError("recording generation HEAD is not a direct commit")
    tree_oid = str(_run_git(root, ["rev-parse", f"{revision}^{{tree}}"])).strip()
    _require_git_oid(tree_oid, object_format=object_format, name="generation source Git tree")
    status = _run_git(
        root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        binary=True,
    )
    assert isinstance(status, bytes)
    if status:
        changed = status.replace(b"\0", b"\n").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"recording generation repository must be completely clean and committed:\n{changed}")

    files: dict[str, dict[str, str]] = {}
    for relative in _GENERATION_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"recording generation source is missing: {relative}")
        worktree_oid = str(_run_git(root, ["hash-object", "--", relative])).strip()
        committed_oid = str(_run_git(root, ["rev-parse", f"{revision}:{relative}"])).strip()
        _require_git_oid(worktree_oid, object_format=object_format, name=f"{relative} worktree blob")
        _require_git_oid(committed_oid, object_format=object_format, name=f"{relative} committed blob")
        if worktree_oid != committed_oid:
            raise RuntimeError(f"recording generation source differs from {revision}: {relative}")
        committed_contents = _run_git(root, ["cat-file", "blob", committed_oid], binary=True)
        assert isinstance(committed_contents, bytes)
        if path.read_bytes() != committed_contents:
            raise RuntimeError(f"recording generation source bytes differ from {revision}: {relative}")
        files[relative] = {
            "sha256": hashlib.sha256(committed_contents).hexdigest(),
            "git_blob_oid": worktree_oid,
        }

    payload: dict[str, object] = {
        "contract": GENERATION_SOURCE_SCHEMA,
        "git_revision": revision,
        "git_tree_oid": tree_oid,
        "git_object_format": object_format,
        "repository_clean": True,
        "repository_status_sha256": hashlib.sha256(status).hexdigest(),
        "files": files,
        "newton_version": _version_string(newton, "newton"),
        "warp_version": _version_string(wp, "warp-lang"),
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_generation_source_manifest(value: object) -> dict[str, object]:
    manifest = _require_exact_keys(
        value,
        {
            "contract",
            "git_revision",
            "git_tree_oid",
            "git_object_format",
            "repository_clean",
            "repository_status_sha256",
            "files",
            "newton_version",
            "warp_version",
            "manifest_sha256",
        },
        name="generation source manifest",
    )
    contract = manifest["contract"]
    if contract == GENERATION_SOURCE_SCHEMA:
        expected_source_paths = _GENERATION_SOURCE_PATHS
    elif contract == GENERATION_SOURCE_SCHEMA_V2:
        expected_source_paths = _GENERATION_SOURCE_PATHS_V2
    else:
        raise ValueError("generation source contract is not canonical")
    object_format = manifest["git_object_format"]
    if type(object_format) is not str:
        raise ValueError("generation source Git object format must be a string")
    _require_git_oid(manifest["git_revision"], object_format=object_format, name="generation source Git revision")
    _require_git_oid(manifest["git_tree_oid"], object_format=object_format, name="generation source Git tree")
    if manifest["repository_clean"] is not True:
        raise ValueError("generation source repository was not clean")
    if manifest["repository_status_sha256"] != hashlib.sha256(b"").hexdigest():
        raise ValueError("clean generation source status digest is not canonical")
    for name in ("newton_version", "warp_version"):
        if type(manifest[name]) is not str or not manifest[name]:
            raise ValueError(f"generation source {name} must be a non-empty string")
    files = _require_exact_keys(manifest["files"], set(expected_source_paths), name="generation source files")
    for relative, item_value in files.items():
        item = _require_exact_keys(item_value, {"sha256", "git_blob_oid"}, name=f"generation source {relative}")
        _require_sha256(item["sha256"], name=f"generation source {relative} SHA-256")
        _require_git_oid(item["git_blob_oid"], object_format=object_format, name=f"generation source {relative} blob")
    expected = _require_sha256(manifest["manifest_sha256"], name="generation source manifest SHA-256")
    unsigned = dict(manifest)
    del unsigned["manifest_sha256"]
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != expected:
        raise ValueError("generation source manifest digest does not match")
    return manifest


def _verify_generation_source_git_objects(
    value: object,
    *,
    repository: pathlib.Path | None = None,
) -> None:
    """Verify a source manifest solely against its recorded Git commit."""
    manifest = _validate_generation_source_manifest(value)
    root = pathlib.Path(__file__).resolve().parents[2] if repository is None else repository.resolve()
    revision = str(manifest["git_revision"])
    object_format = str(manifest["git_object_format"])
    try:
        current_object_format = str(_run_git(root, ["rev-parse", "--show-object-format"])).strip()
        if current_object_format != object_format:
            raise ValueError("generation source Git object format differs from the repository")
        committed_revision = str(_run_git(root, ["rev-parse", "--verify", f"{revision}^{{commit}}"])).strip()
        if committed_revision != revision:
            raise ValueError("generation source revision does not resolve to its recorded commit")
        tree_oid = str(_run_git(root, ["rev-parse", f"{revision}^{{tree}}"])).strip()
        if tree_oid != manifest["git_tree_oid"]:
            raise ValueError("generation source tree differs from its recorded commit")
        if str(_run_git(root, ["cat-file", "-t", tree_oid])).strip() != "tree":
            raise ValueError("generation source tree object is not a Git tree")
        for relative, item in manifest["files"].items():
            committed_oid = str(_run_git(root, ["rev-parse", f"{revision}:{relative}"])).strip()
            if committed_oid != item["git_blob_oid"]:
                raise ValueError(f"generation source blob differs at recorded revision: {relative}")
            if str(_run_git(root, ["cat-file", "-t", committed_oid])).strip() != "blob":
                raise ValueError(f"generation source object is not a blob: {relative}")
            contents = _run_git(root, ["cat-file", "blob", committed_oid], binary=True)
            assert isinstance(contents, bytes)
            if hashlib.sha256(contents).hexdigest() != item["sha256"]:
                raise ValueError(f"generation source raw blob SHA-256 differs: {relative}")
    except RuntimeError as error:
        raise ValueError(f"generation source Git object verification failed for revision {revision}") from error


def _scene_physical_sha256(scene: TetBenchmarkScene) -> str:
    names = (
        "rest_q",
        "tet_indices",
        "tet_poses",
        "mass",
        "particle_inv_mass",
        "tet_materials",
        "tri_indices",
        "tri_poses",
        "tri_materials",
        "tri_areas",
        "particle_flags",
        "color_group_offsets",
        "color_group_particles",
        "x_current",
        "velocity",
        "gravity",
        "external_force",
        "pinned_indices",
        "pin_targets",
        "vbd_inertial_target",
    )
    payload = {
        "contract": "tet-recording-physical-scene-v1",
        "name": scene.name,
        "source": scene.source,
        "dt_seconds": scene.dt,
        "arrays": {name: array_sha256(getattr(scene, name)) for name in names},
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def twist_pin_targets(scene: TetBenchmarkScene, source_frame: int) -> np.ndarray:
    """Return the exact float32-published PR twist targets for one frame."""
    if type(source_frame) is not int or source_frame < 0:
        raise ValueError("source_frame must be a non-negative built-in int")
    if scene.name != "pr2901-twist-0deg-3x3x16-boundary-step":
        raise ValueError("twist targets require the canonical rest-state twist scene")

    angle = 2.0 * np.pi if source_frame >= 200 else (source_frame / 200.0) * (2.0 * np.pi)
    rest = scene.rest_q.astype(np.float32)
    targets = rest[scene.pinned_indices].copy()
    pinned_rest = rest[scene.pinned_indices]
    top_local = np.where(np.isclose(pinned_rest[:, 2], np.float32(1.8), rtol=0.0, atol=1.0e-6))[0]
    center = np.array([3 * 0.05 / 2.0, 3 * 0.05 / 2.0])
    cosine = np.cos(angle)
    sine = np.sin(angle)
    for local_index in top_local:
        rx = pinned_rest[local_index, 0] - center[0]
        ry = pinned_rest[local_index, 1] - center[1]
        targets[local_index, 0] = center[0] + cosine * rx - sine * ry
        targets[local_index, 1] = center[1] + sine * rx + cosine * ry
    return targets.astype(np.float64)


def pin_targets_for_frame(scene_key: str, scene: TetBenchmarkScene, source_frame: int) -> np.ndarray:
    """Return targets held constant through one source frame's substeps."""
    recording_spec(scene_key)
    if type(source_frame) is not int or source_frame < 0:
        raise ValueError("source_frame must be a non-negative built-in int")
    if scene_key == "twist":
        return twist_pin_targets(scene, source_frame)
    return np.array(scene.pin_targets, dtype=np.float64, copy=True)


def fixed_camera(
    positions: np.ndarray,
    *,
    panel_width: int,
    panel_height: int,
    direction: tuple[float, float, float],
    fov_degrees: float = 32.0,
) -> dict[str, object]:
    """Fit one fixed perspective camera to the union of all trajectories."""
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim < 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError("positions must be a finite array ending in (V, 3)")
    if type(panel_width) is not int or type(panel_height) is not int or min(panel_width, panel_height) < 1:
        raise ValueError("panel dimensions must be positive built-in integers")
    if not math.isfinite(fov_degrees) or not 1.0 <= fov_degrees < 179.0:
        raise ValueError("fov_degrees must lie in [1, 179)")

    points = array.reshape(-1, 3)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    target = 0.5 * (lower + upper)
    radius = float(np.linalg.norm(points - target[None, :], axis=1).max())
    radius = max(radius, 1.0e-3)
    view = np.asarray(direction, dtype=np.float64)
    if view.shape != (3,) or not np.isfinite(view).all() or np.linalg.norm(view) == 0.0:
        raise ValueError("direction must be one finite nonzero 3-vector")
    view /= np.linalg.norm(view)
    half_vertical = math.radians(0.5 * fov_degrees)
    half_horizontal = math.atan((panel_width / panel_height) * math.tan(half_vertical))
    limiting_half_angle = min(half_vertical, half_horizontal)
    distance = 1.25 * radius / math.sin(limiting_half_angle)
    camera_position = target + distance * view
    return {
        "contract": "union-aabb-fixed-perspective-camera-v1",
        "position": camera_position.tolist(),
        "target": target.tolist(),
        "fov_degrees": fov_degrees,
        "panel_width": panel_width,
        "panel_height": panel_height,
        "union_aabb_min_m": lower.tolist(),
        "union_aabb_max_m": upper.tolist(),
        "union_radius_m": radius,
        "margin_factor": 1.25,
    }


def _validate_source_record(record_value: object, expected_sha256: object, *, role: str) -> dict[str, object]:
    """Authenticate exact source-record structure before canonical semantics."""
    expected_names = _DENSE_SOURCE_RECORD_NAMES if role == "dense" else _SPARSE_SOURCE_RECORD_NAMES
    record = _require_exact_keys(record_value, expected_names, name=f"{role} Newton source record")
    expected = _require_sha256(expected_sha256, name=f"{role} source record SHA-256")
    if hashlib.sha256(_canonical_json(record)).hexdigest() != expected:
        raise ValueError(f"{role} source record digest is stale")
    expected_method, expected_contract = _STATIC_REFERENCE_METHODS[role]
    if record.get("method") != expected_method or record.get("contract") != expected_contract:
        raise ValueError(f"{role} Newton endpoint method or contract is not canonical")
    _require_exact_keys(record["config"], _NEWTON_CONFIG_NAMES, name=f"{role} Newton config")
    if role == "dense" and (
        record["accepted"] is not True
        or record["failures"] != []
        or record["native_converged"] is not True
        or record["native_reason"] != "gradient"
        or record["verification_converged"] is not True
        or record["verification_reason"] != "gradient"
        or record["alternate_start_converged"] is not True
        or record["alternate_start_reason"] != "gradient"
    ):
        raise ValueError("dense Newton source record is not accepted canonical evidence")
    if role == "dense" and (type(record["accepted_iterations"]) is not int or record["accepted_iterations"] < 0):
        raise ValueError("dense Newton accepted iteration count must be a non-negative built-in int")
    if role == "sparse":
        for name in ("native_result", "verification_result", "alternate_start_result"):
            result = _require_exact_keys(record[name], _SPARSE_RESULT_NAMES, name=f"sparse {name}")
            _require_exact_keys(result["work"], _SPARSE_WORK_NAMES, name=f"sparse {name} work")
            trace = result["trace"]
            if type(trace) is not list or not trace:
                raise ValueError(f"sparse {name} trace must be a non-empty list")
            for index, item in enumerate(trace):
                _require_exact_keys(item, _SPARSE_TRACE_NAMES, name=f"sparse {name} trace[{index}]")
    return record


def _require_static_metric_record(value: object, *, role: str) -> dict[str, object]:
    record = _require_exact_keys(value, set(_COMMON_METRIC_NAMES), name=f"{role} independent metrics")
    for name in _COMMON_METRIC_NAMES:
        if name == "position_sha256":
            _require_sha256(record[name], name=f"{role} independent position SHA-256")
        else:
            _require_finite_number(record[name], name=f"{role} independent {name}")
    return record


def _validate_static_endpoint_metrics(metrics: CommonStateMetrics, *, role: str) -> None:
    if metrics.inverted_tet_fraction != 0.0 or metrics.determinant_min <= 0.0:
        raise ValueError(f"authenticated reference {role} has an inverted or non-positive-determinant tetrahedron")
    if metrics.max_pin_error_m != 0.0:
        raise ValueError(f"authenticated reference {role} does not preserve exact pins")


def _static_comparison(
    scene: TetBenchmarkScene,
    dense_positions: np.ndarray,
    sparse_positions: np.ndarray,
    dense_metrics: CommonStateMetrics,
    sparse_metrics: CommonStateMetrics,
) -> dict[str, float]:
    difference = sparse_positions[scene.free_indices] - dense_positions[scene.free_indices]
    distances = np.linalg.norm(difference, axis=1)
    return {
        "free_max_m": float(distances.max(initial=0.0)),
        "free_rms_m": float(np.sqrt(np.mean(np.sum(difference * difference, axis=1)))),
        "gradient_delta_N_sparse_minus_dense": sparse_metrics.gradient_norm - dense_metrics.gradient_norm,
        "objective_delta_sparse_minus_dense": sparse_metrics.objective - dense_metrics.objective,
    }


def _validate_static_comparison(stored: object, recomputed: Mapping[str, float]) -> None:
    record = _require_exact_keys(stored, set(_STATIC_COMPARISON_NAMES), name="static Newton comparison")
    for name in _STATIC_COMPARISON_NAMES:
        value = _require_finite_number(record[name], name=f"static Newton comparison {name}")
        if not math.isclose(value, recomputed[name], rel_tol=1.0e-12, abs_tol=1.0e-18):
            raise ValueError(f"authenticated static comparison {name} no longer reproduces")


def load_authenticated_medium_reference(
    json_path: str | os.PathLike[str],
    *,
    expected_json_sha256: str,
    expected_npz_sha256: str,
    npz_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load and authenticate the frozen dense/sparse medium endpoint bundle."""
    source_json = pathlib.Path(json_path).resolve()
    if _file_sha256(source_json) != _require_sha256(expected_json_sha256, name="reference JSON SHA-256"):
        raise ValueError("authenticated reference JSON file digest does not match")
    record = json.loads(source_json.read_text(encoding="utf-8"))
    record = _require_exact_keys(
        record,
        {
            "schema",
            "scene_name",
            "scene_sha256",
            "scene_physical_sha256",
            "objective_instance_sha256",
            "git_revision",
            "vertices",
            "tets",
            "free_dofs",
            "npz_path",
            "arrays",
            "dense",
            "sparse",
            "comparison",
        },
        name="authenticated reference record",
    )
    if record.get("schema") != STATIC_REFERENCE_SCHEMA:
        raise ValueError(f"reference JSON must use schema {STATIC_REFERENCE_SCHEMA!r}")
    if record.get("scene_name") != "pr2901-refinement-medium-common-step":
        raise ValueError("authenticated reference belongs to another scene")
    _require_sha256(record.get("scene_sha256"), name="reference scene SHA-256")
    _require_sha256(record.get("scene_physical_sha256"), name="reference physical scene SHA-256")
    _require_sha256(record.get("objective_instance_sha256"), name="reference objective SHA-256")
    _require_git_revision(record.get("git_revision"), name="reference Git revision")
    if (record.get("vertices"), record.get("tets"), record.get("free_dofs")) != (525, 1600, 1500):
        raise ValueError("authenticated reference topology counts are not canonical")
    if type(record.get("npz_path")) is not str or not record["npz_path"]:
        raise ValueError("authenticated reference NPZ path must be a non-empty string")

    declared_npz = pathlib.Path(str(record["npz_path"]))
    if npz_path is not None:
        source_npz = pathlib.Path(npz_path).resolve()
    elif declared_npz.is_absolute():
        source_npz = declared_npz.resolve()
    else:
        source_npz = (source_json.parent / declared_npz).resolve()
    if _file_sha256(source_npz) != _require_sha256(expected_npz_sha256, name="reference NPZ SHA-256"):
        raise ValueError("authenticated reference NPZ file digest does not match")
    expected_keys = {"dense_positions", "sparse_positions"}
    with np.load(source_npz, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("authenticated reference NPZ has unexpected arrays")
        arrays = {name: np.array(archive[name], copy=True) for name in sorted(expected_keys)}

    array_records = record.get("arrays")
    if type(array_records) is not dict or set(array_records) != expected_keys:
        raise ValueError("authenticated reference array manifest is incomplete")
    for name, array in arrays.items():
        item = _require_exact_keys(
            array_records[name], {"dtype", "shape", "array_sha256"}, name=f"authenticated reference {name} manifest"
        )
        if type(array) is not np.ndarray or array.dtype != np.dtype(np.float64) or array.shape != (525, 3):
            raise ValueError(f"authenticated reference {name} must be finite float64 (525, 3)")
        if not np.isfinite(array).all():
            raise ValueError(f"authenticated reference {name} must be finite float64 (525, 3)")
        if item.get("dtype") != "float64" or item.get("shape") != [525, 3]:
            raise ValueError(f"authenticated reference {name} dtype/shape is stale")
        _require_sha256(item["array_sha256"], name=f"authenticated reference {name} array SHA-256")
        if array_sha256(array) != item["array_sha256"]:
            raise ValueError(f"authenticated reference {name} array digest does not match")

    scene = build_recording_scene("refinement-medium")
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    if record["scene_sha256"] != scene_sha256:
        raise ValueError("authenticated reference scene differs from the canonical medium scene")
    if record["scene_physical_sha256"] != _scene_physical_sha256(scene):
        raise ValueError("authenticated reference physical scene differs from the canonical medium scene")
    problem = build_common_problem(scene)
    objective_sha256 = str(common_objective_manifest(scene, problem)["objective_instance_sha256"])
    if record["objective_instance_sha256"] != objective_sha256:
        raise ValueError("authenticated reference objective differs from the canonical medium scene")
    from .correction_mg_vbd import _supplied_reference_evidence  # noqa: PLC0415

    evaluated: dict[str, CommonStateMetrics] = {}
    for role, array_name in (("dense", "dense_positions"), ("sparse", "sparse_positions")):
        entry = _require_exact_keys(
            record.get(role),
            {"method", "source_record", "source_record_sha256", "independent_metrics"},
            name=f"authenticated reference {role} entry",
        )
        expected_method, _expected_contract = _STATIC_REFERENCE_METHODS[role]
        if entry.get("method") != expected_method:
            raise ValueError(f"authenticated reference {role} method is not canonical")
        source_record = _validate_source_record(entry["source_record"], entry["source_record_sha256"], role=role)
        independent_metrics = _require_static_metric_record(entry["independent_metrics"], role=role)
        metrics = evaluate_common_state(problem, arrays[array_name], reference_positions=arrays[array_name])
        evaluated[role] = metrics
        _validate_static_endpoint_metrics(metrics, role=role)
        evidence = _supplied_reference_evidence(
            arrays[array_name],
            source_record,
            entry["source_record_sha256"],
            scene_sha256=scene_sha256,
            objective_instance_sha256=objective_sha256,
            metrics=metrics,
            residual_scale=problem.residual_scale,
        )
        if evidence.provenance != f"externally-pinned-supplied-reference:{entry['method']}":
            raise ValueError(f"authenticated reference {role} canonical provenance is stale")
        if independent_metrics["position_sha256"] != evidence.position_sha256:
            raise ValueError(f"authenticated reference {role} independent position binding is stale")
        for field in _COMMON_METRIC_NAMES:
            actual = getattr(metrics, field)
            stored = independent_metrics[field]
            matches = (
                actual == stored
                if field == "position_sha256"
                else math.isclose(float(actual), float(stored), rel_tol=1.0e-12, abs_tol=1.0e-14)
            )
            if not matches:
                raise ValueError(f"authenticated reference {role} independent {field} no longer reproduces")
    comparison = _static_comparison(
        scene,
        arrays["dense_positions"],
        arrays["sparse_positions"],
        evaluated["dense"],
        evaluated["sparse"],
    )
    _validate_static_comparison(record["comparison"], comparison)
    return record, arrays


def static_medium_reference_evidence(
    scene: TetBenchmarkScene,
    json_path: str | os.PathLike[str],
    *,
    expected_json_sha256: str,
    expected_npz_sha256: str,
    npz_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Verify endpoints independently and return a compact static-only record."""
    if scene.name != "pr2901-refinement-medium-common-step":
        raise ValueError("static medium reference evidence requires the medium scene")
    record, arrays = load_authenticated_medium_reference(
        json_path,
        expected_json_sha256=expected_json_sha256,
        expected_npz_sha256=expected_npz_sha256,
        npz_path=npz_path,
    )
    if record.get("git_revision") != _git_revision():
        raise ValueError("authenticated reference was produced from another committed revision")
    current_scene_sha256 = str(scene.manifest()["scene_sha256"])
    if record["scene_sha256"] != current_scene_sha256:
        raise ValueError("authenticated reference scene manifest differs from the current medium scene")
    current_physical_sha256 = _scene_physical_sha256(scene)
    if record.get("scene_physical_sha256") != current_physical_sha256:
        raise ValueError("authenticated reference physical scene differs from the current medium scene")

    problem = build_common_problem(scene)
    current_objective_sha256 = str(common_objective_manifest(scene, problem)["objective_instance_sha256"])
    if record["objective_instance_sha256"] != current_objective_sha256:
        raise ValueError("authenticated reference objective differs from the current medium scene")
    verified: dict[str, dict[str, object]] = {}
    evaluated: dict[str, CommonStateMetrics] = {}
    for role, array_name in (("dense", "dense_positions"), ("sparse", "sparse_positions")):
        metrics = evaluate_common_state(problem, arrays[array_name], reference_positions=arrays[array_name])
        evaluated[role] = metrics
        residual_limit = max(1.0e-10, 1.0e-10 * problem.residual_scale)
        _validate_static_endpoint_metrics(metrics, role=role)
        if metrics.gradient_norm > residual_limit:
            raise ValueError(f"authenticated reference {role} fails independent endpoint acceptance gates")
        stored = record[role]["independent_metrics"]
        for field in _COMMON_METRIC_NAMES:
            actual = getattr(metrics, field)
            if field == "position_sha256":
                matches = actual == stored[field]
            else:
                matches = math.isclose(float(actual), float(stored[field]), rel_tol=1.0e-12, abs_tol=1.0e-14)
            if not matches:
                raise ValueError(f"authenticated reference {role} independent {field} no longer reproduces")
        verified[role] = {
            "method": record[role]["method"],
            "source_record_sha256": record[role]["source_record_sha256"],
            "position_sha256": metrics.position_sha256,
            "objective": metrics.objective,
            "gradient_norm": metrics.gradient_norm,
            "relative_residual": metrics.relative_residual,
            "determinant_min": metrics.determinant_min,
            "inverted_tet_fraction": metrics.inverted_tet_fraction,
            "max_pin_error_m": metrics.max_pin_error_m,
        }
    comparison = _static_comparison(
        scene,
        arrays["dense_positions"],
        arrays["sparse_positions"],
        evaluated["dense"],
        evaluated["sparse"],
    )
    _validate_static_comparison(record["comparison"], comparison)
    return {
        "role": "authenticated-static-first-atomic-step-endpoint-only",
        "scope": "medium rest-state gravity substep at t=dt; excluded from all video trajectory lanes",
        "video_reference_lane": False,
        "ground_truth_claim": False,
        "source_schema": record["schema"],
        "source_json_sha256": expected_json_sha256,
        "source_npz_sha256": expected_npz_sha256,
        "source_scene_sha256": record["scene_sha256"],
        "source_scene_physical_sha256": record["scene_physical_sha256"],
        "source_objective_instance_sha256": record["objective_instance_sha256"],
        "current_scene_sha256": current_scene_sha256,
        "current_scene_physical_sha256": current_physical_sha256,
        "dense": verified["dense"],
        "sparse": verified["sparse"],
        "comparison": comparison,
    }


class _PublicVBDTrajectory:
    """History-bearing trajectory built only from public ``SolverVBD.step``."""

    def __init__(self, scene: TetBenchmarkScene, iterations: int, device: str):
        self.scene = scene
        self.iterations = iterations
        self.model = _build_vbd_model(scene, device)
        self.solver = SolverVBD(
            model=self.model,
            iterations=iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.positions = np.array(scene.x_current, dtype=np.float32, copy=True).astype(np.float64)
        self.velocities = np.array(scene.velocity, dtype=np.float32, copy=True).astype(np.float64)
        self.external_force = np.array(scene.external_force, dtype=np.float32, copy=True).astype(np.float64)
        self.pin_targets = np.array(scene.pin_targets, dtype=np.float32, copy=True).astype(np.float64)
        self.set_pin_targets(self.pin_targets)

    def set_pin_targets(self, targets: np.ndarray) -> None:
        targets32 = np.asarray(targets, dtype=np.float32)
        if targets32.shape != (self.scene.pinned_indices.size, 3) or not np.isfinite(targets32).all():
            raise ValueError("public VBD pin targets have the wrong shape or non-finite values")
        self.pin_targets = targets32.astype(np.float64)
        self.positions[self.scene.pinned_indices] = self.pin_targets
        self.velocities[self.scene.pinned_indices] = 0.0

    def step(self) -> tuple[float, float]:
        self.state_in.clear_forces()
        self.state_in.particle_q.assign(np.asarray(self.positions, dtype=np.float32))
        self.state_in.particle_qd.assign(np.asarray(self.velocities, dtype=np.float32))
        self.state_in.particle_f.assign(np.asarray(self.external_force, dtype=np.float32))
        with wp.ScopedTimer("trajectory-public-vbd", print=False, synchronize=True) as timer:
            self.solver.step(self.state_in, self.state_out, self.control, None, self.scene.dt)
        transfer_start = time.perf_counter()
        positions = self.state_out.particle_q.numpy().astype(np.float64)
        velocities = self.state_out.particle_qd.numpy().astype(np.float64)
        transfer_seconds = time.perf_counter() - transfer_start
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise RuntimeError("public VBD trajectory produced a non-finite state")
        if not np.array_equal(positions[self.scene.pinned_indices], self.pin_targets):
            raise RuntimeError("public VBD trajectory changed an exact pin")
        self.positions = positions
        self.velocities = velocities
        return timer.elapsed * 1.0e-3, transfer_seconds


class _MGVBDTrajectory:
    """Thin trajectory owner around the committed captured rollout adapter."""

    def __init__(self, scene: TetBenchmarkScene, device: str):
        from .mg_vbd_rollout import MGVBDRollout, MGVBDRolloutCapturedBackend  # noqa: PLC0415

        self.scene = scene
        self.config = DirectGraphVBDConfig()
        self.config.validate()
        start = time.perf_counter()
        self.backend = MGVBDRolloutCapturedBackend.build(
            scene,
            device=device,
            config=self.config,
            tile_solve=False,
            warmup_replays=1,
        )
        self.setup_seconds = time.perf_counter() - start
        self.rollout = MGVBDRollout(scene, self.backend)

    @property
    def positions(self) -> np.ndarray:
        return self.rollout.state.positions

    @property
    def velocities(self) -> np.ndarray:
        return self.rollout.state.velocities

    def set_pin_targets(self, targets: np.ndarray) -> None:
        self.rollout.set_pin_targets(targets)

    def step(self) -> tuple[object, float]:
        start = time.perf_counter()
        state = self.rollout.step(self.scene.dt)
        return state, time.perf_counter() - start


def _empty_metric_arrays(frame_count: int) -> dict[str, np.ndarray]:
    return {f"metric_{name}": np.full((len(METHOD_IDS), frame_count), np.nan) for name in METRIC_NAMES}


def _store_metrics(arrays: dict[str, np.ndarray], method: int, frame: int, metrics: CommonStateMetrics) -> None:
    for name in METRIC_NAMES:
        value = getattr(metrics, name)
        arrays[f"metric_{name}"][method, frame] = np.nan if value is None else float(value)


def _method_records(spec: RecordingSpec) -> list[dict[str, object]]:
    config = DirectGraphVBDConfig()
    return [
        {
            "id": "reference",
            "panel_title": "REFERENCE*",
            "method": "public SolverVBD",
            "iterations_per_atomic_step": spec.reference_iterations,
            "role": "high-budget numerical trajectory comparison",
            "ground_truth_claim": False,
            "newton_claim": False,
        },
        {
            "id": "mg_vbd",
            "panel_title": "MG-VBD",
            "method": "MGVBDRollout + MGVBDRolloutCapturedBackend",
            "config": dataclasses.asdict(config),
            "role": "captured four-outer/two-V-cycle candidate trajectory",
        },
        {
            "id": "vbd_k4",
            "panel_title": "VBD K4",
            "method": "public SolverVBD",
            "iterations_per_atomic_step": 4,
            "role": "public four-sweep baseline trajectory",
        },
    ]


def _reference_policy(spec: RecordingSpec) -> dict[str, object]:
    return {
        "video_lane": "high-budget public SolverVBD numerical trajectory",
        "iterations_per_atomic_step": spec.reference_iterations,
        "ground_truth_claim": False,
        "newton_trajectory_claim": False,
        "required_on_frame_label": "numerical reference; not Newton/ground truth",
    }


def _metrics_policy() -> dict[str, object]:
    return {
        "contract": "independent-float64-common-objective-per-method-history-v1",
        "residual_scope": "each method's own final atomic-step implicit objective",
        "error_scope": "free-particle displacement from the high-budget public VBD trajectory",
        "initial_frame_metrics": "not applicable; stored as NaN in NPZ and rendered as n/a",
        "timing_policy": "diagnostic only; setup and transfers are not cross-method performance evidence",
    }


def _source_schedule(scene_key: str) -> str:
    recording_spec(scene_key)
    if scene_key == "twist":
        return "angle(f)=2*pi if f>=200 else (f/200)*2*pi; target held through five substeps"
    if scene_key == "refinement-medium":
        return "fixed top face under gravity"
    return "fixed support slab under gravity and tip load"


def _gaia_protocol_annotation(scene_key: object) -> str | None:
    """Return the physical Gaia protocol shown directly on rendered frames."""
    if type(scene_key) is not str or scene_key not in _GAIA_SCENE_ASSETS:
        return None
    _asset_name, unit_scale = _GAIA_SCENE_ASSETS[scene_key]
    return (
        f"scale {unit_scale:g} m/source unit · fixed min-y 2% slab · "
        "+x 10 N total on max-y 2% slab · gravity -y 9.81 m/s²"
    )


def _require_encoder_dimensions(panel_width: object, panel_height: object) -> tuple[int, int]:
    if (
        type(panel_width) is not int
        or type(panel_height) is not int
        or min(panel_width, panel_height) < 16
        or panel_width % 16
        or panel_height % 16
    ):
        raise ValueError("imageio H.264 panel dimensions must be positive multiples of 16")
    return panel_width, panel_height


def generate_trajectory(
    scene_key: str,
    *,
    source_frames: int,
    device: str,
    panel_width: int = 640,
    panel_height: int = 720,
    static_reference: dict[str, object] | None = None,
    gaia_asset_root: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Generate and independently score one three-method trajectory.

    Args:
        scene_key: Canonical recording scene key.
        source_frames: Number of 60 Hz source frames to simulate.
        device: Claimed CUDA device visible to this process.
        panel_width: Width of each output panel in pixels.
        panel_height: Height of each output panel in pixels.
        static_reference: Optional authenticated medium-scene Newton evidence.
        gaia_asset_root: Gaia checkout root for a Gaia scene.
    """
    if type(source_frames) is not int or source_frames < 1:
        raise ValueError("source_frames must be a positive built-in int")
    spec = recording_spec(scene_key)
    if scene_key in _GAIA_SCENE_ASSETS:
        gaia_asset_root = _resolve_gaia_asset_root(gaia_asset_root)
        _verify_gaia_asset_bundle_files(gaia_asset_root)
    scene = build_recording_scene(scene_key, gaia_asset_root=gaia_asset_root)
    requested_device = wp.get_device(device)
    if not requested_device.is_cuda:
        raise ValueError("MG-VBD recording generation requires a claimed CUDA device")
    if static_reference is not None and scene_key != "refinement-medium":
        raise ValueError("static Newton endpoint evidence is valid only for refinement-medium")
    _require_encoder_dimensions(panel_width, panel_height)
    generation_source = _generation_source_manifest()

    setup_start = time.perf_counter()
    reference = _PublicVBDTrajectory(scene, spec.reference_iterations, str(requested_device))
    reference_setup_seconds = time.perf_counter() - setup_start
    setup_start = time.perf_counter()
    k4 = _PublicVBDTrajectory(scene, 4, str(requested_device))
    k4_setup_seconds = time.perf_counter() - setup_start
    mg = _MGVBDTrajectory(scene, str(requested_device))

    stored_frames = source_frames + 1
    shape = (len(METHOD_IDS), stored_frames, scene.n_vertices, 3)
    positions = np.empty(shape, dtype=np.float64)
    velocities = np.empty(shape, dtype=np.float64)
    objective_input_positions = np.empty(shape, dtype=np.float64)
    objective_input_velocities = np.empty(shape, dtype=np.float64)
    pin_targets = np.empty((stored_frames, scene.pinned_indices.size, 3), dtype=np.float64)
    time_seconds = np.arange(stored_frames, dtype=np.float64) * spec.substeps_per_source_frame * scene.dt
    source_frame_index = np.arange(-1, source_frames, dtype=np.int64)
    solve_seconds = np.zeros((len(METHOD_IDS), stored_frames), dtype=np.float64)
    transfer_seconds = np.zeros((len(METHOD_IDS), stored_frames), dtype=np.float64)
    mg_last_gate_accepted = np.full((stored_frames, 4), -1, dtype=np.int8)
    mg_last_gate_reason_code = np.full((stored_frames, 4), -1, dtype=np.int16)
    mg_frame_gate_accept_count = np.zeros(stored_frames, dtype=np.int64)

    initial_targets = pin_targets_for_frame(scene_key, scene, 0)
    runners: tuple[Any, ...] = (reference, mg, k4)
    for runner in runners:
        runner.set_pin_targets(initial_targets)
    for method, runner in enumerate(runners):
        positions[method, 0] = runner.positions
        velocities[method, 0] = runner.velocities
        objective_input_positions[method, 0] = runner.positions
        objective_input_velocities[method, 0] = runner.velocities
    pin_targets[0] = initial_targets

    reason_names: list[str] = []
    for source_frame in range(source_frames):
        output_frame = source_frame + 1
        targets = pin_targets_for_frame(scene_key, scene, source_frame)
        pin_targets[output_frame] = targets
        for runner in runners:
            runner.set_pin_targets(targets)

        frame_gate_accept_count = 0
        last_mg_state = None
        for substep in range(spec.substeps_per_source_frame):
            if substep == spec.substeps_per_source_frame - 1:
                for method, runner in enumerate(runners):
                    objective_input_positions[method, output_frame] = runner.positions
                    objective_input_velocities[method, output_frame] = runner.velocities

            solve, transfer = reference.step()
            solve_seconds[0, output_frame] += solve
            transfer_seconds[0, output_frame] += transfer
            last_mg_state, elapsed = mg.step()
            solve_seconds[1, output_frame] += elapsed
            frame_gate_accept_count += sum(last_mg_state.accepted)
            solve, transfer = k4.step()
            solve_seconds[2, output_frame] += solve
            transfer_seconds[2, output_frame] += transfer

        assert last_mg_state is not None
        if not reason_names:
            from .captured_graph_vbd import REASON_NAMES  # noqa: PLC0415

            reason_names = list(REASON_NAMES)
        mg_last_gate_accepted[output_frame] = np.asarray(last_mg_state.accepted, dtype=np.int8)
        mg_last_gate_reason_code[output_frame] = np.asarray(
            [reason_names.index(reason) for reason in last_mg_state.reasons],
            dtype=np.int16,
        )
        mg_frame_gate_accept_count[output_frame] = frame_gate_accept_count
        for method, runner in enumerate(runners):
            positions[method, output_frame] = runner.positions
            velocities[method, output_frame] = runner.velocities

    arrays: dict[str, np.ndarray] = {
        "positions": positions,
        "velocities": velocities,
        "objective_input_positions": objective_input_positions,
        "objective_input_velocities": objective_input_velocities,
        "pin_targets": pin_targets,
        "time_seconds": time_seconds,
        "source_frame_index": source_frame_index,
        "solve_seconds": solve_seconds,
        "transfer_seconds": transfer_seconds,
        "mg_last_gate_accepted": mg_last_gate_accepted,
        "mg_last_gate_reason_code": mg_last_gate_reason_code,
        "mg_frame_gate_accept_count": mg_frame_gate_accept_count,
    }
    arrays.update(_empty_metric_arrays(stored_frames))

    for frame in range(1, stored_frames):
        reference_positions = positions[0, frame]
        for method in range(len(METHOD_IDS)):
            objective_scene = dataclasses.replace(
                scene,
                x_current=objective_input_positions[method, frame],
                velocity=objective_input_velocities[method, frame],
                pin_targets=pin_targets[frame],
            )
            metrics = evaluate_common_state(
                build_common_problem(objective_scene),
                positions[method, frame],
                reference_positions=reference_positions,
            )
            _store_metrics(arrays, method, frame, metrics)

    camera = fixed_camera(
        positions,
        panel_width=panel_width,
        panel_height=panel_height,
        direction=spec.camera_direction,
    )
    device_record = {
        "requested": device,
        "resolved": str(requested_device),
        "is_cuda": bool(requested_device.is_cuda),
        "name": requested_device.name,
    }
    metadata: dict[str, object] = {
        "schema": SCHEMA,
        "scene_key": scene_key,
        "scene_display_name": spec.display_name,
        "scene_manifest": scene.manifest(),
        "scene_physical_sha256": _scene_physical_sha256(scene),
        "git_revision": generation_source["git_revision"],
        "generation_source": generation_source,
        "methods": _method_records(spec),
        "method_order": list(METHOD_IDS),
        "reference_policy": _reference_policy(spec),
        "simulation": {
            "source_frames": source_frames,
            "stored_frames_including_initial": stored_frames,
            "source_frame_rate_hz": 60,
            "substeps_per_source_frame": spec.substeps_per_source_frame,
            "atomic_dt_seconds": scene.dt,
            "stored_duration_seconds": float(time_seconds[-1]),
            "source_schedule": _source_schedule(scene_key),
        },
        "metrics": _metrics_policy(),
        "mg_gate_reason_names": reason_names,
        "setup_seconds_diagnostic": {
            "reference_public_vbd": reference_setup_seconds,
            "mg_vbd_capture_and_setup": mg.setup_seconds,
            "vbd_k4_public": k4_setup_seconds,
        },
        "device": device_record,
        "camera": camera,
        "static_first_step_newton_reference": static_reference,
        "gaia_asset_bundle": _gaia_asset_bundle_manifest(scene_key),
        "execution_authentication": _execution_authentication_policy(),
    }
    _validate_recording_bundle(
        metadata,
        arrays,
        sealed=False,
        require_current_generation_source=True,
        gaia_asset_root=gaia_asset_root,
    )
    _issue_generated_trajectory(metadata, arrays)
    return metadata, arrays


def _historical_scene_manifest(scene: TetBenchmarkScene, revision: str) -> dict[str, object]:
    manifest = json.loads(_canonical_json(scene.manifest()))
    del manifest["scene_sha256"]
    manifest["metadata"]["newton_revision"] = revision
    manifest["metadata"]["dirty_tree_sha256"] = None
    manifest["scene_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    return manifest


def _validate_scene_manifest(
    value: object,
    scene: TetBenchmarkScene,
    *,
    require_current: bool,
    expected_revision: str,
) -> None:
    expected_names = {
        "schema_version",
        "name",
        "source",
        "dt_seconds",
        "n_vertices",
        "n_tets",
        "n_triangles",
        "n_pinned",
        "metadata",
        "arrays",
        "scene_sha256",
    }
    manifest = _require_exact_keys(value, expected_names, name="scene manifest")
    expected_sha256 = _require_sha256(manifest["scene_sha256"], name="scene manifest SHA-256")
    unsigned = dict(manifest)
    del unsigned["scene_sha256"]
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != expected_sha256:
        raise ValueError("scene manifest semantic digest does not match")
    current = scene.manifest()
    manifest_bytes = _canonical_json(manifest)
    current_bytes = _canonical_json(current)
    historical_bytes = _canonical_json(_historical_scene_manifest(scene, expected_revision))
    if require_current and manifest_bytes not in (current_bytes, historical_bytes):
        raise ValueError("scene manifest differs from the current canonical scene")
    if not require_current and manifest_bytes != historical_bytes:
        raise ValueError("historical scene manifest differs from its exact clean physical reconstruction")


def _historical_static_reference_identities(scene: TetBenchmarkScene, revision: str) -> tuple[str, str, str]:
    historical_scene = str(_historical_scene_manifest(scene, revision)["scene_sha256"])
    historical_objective = json.loads(_canonical_json(common_objective_manifest(scene, build_common_problem(scene))))
    del historical_objective["objective_instance_sha256"]
    historical_objective["scene_sha256"] = historical_scene
    historical_objective_sha256 = hashlib.sha256(_canonical_json(historical_objective)).hexdigest()
    return historical_scene, historical_scene, historical_objective_sha256


def _validate_compact_static_reference(
    value: object,
    scene: TetBenchmarkScene,
    *,
    require_current: bool,
    expected_revision: str | None = None,
) -> None:
    record = _require_exact_keys(
        value,
        {
            "role",
            "scope",
            "video_reference_lane",
            "ground_truth_claim",
            "source_schema",
            "source_json_sha256",
            "source_npz_sha256",
            "source_scene_sha256",
            "source_scene_physical_sha256",
            "source_objective_instance_sha256",
            "current_scene_sha256",
            "current_scene_physical_sha256",
            "dense",
            "sparse",
            "comparison",
        },
        name="compact static Newton reference",
    )
    if record["role"] != "authenticated-static-first-atomic-step-endpoint-only":
        raise ValueError("compact static Newton role is not canonical")
    if record["scope"] != "medium rest-state gravity substep at t=dt; excluded from all video trajectory lanes":
        raise ValueError("compact static Newton scope is not canonical")
    if record["video_reference_lane"] is not False or record["ground_truth_claim"] is not False:
        raise ValueError("static Newton endpoint must not claim a video lane or ground truth")
    if record["source_schema"] != STATIC_REFERENCE_SCHEMA:
        raise ValueError("compact static Newton source schema is not canonical")
    for name in (
        "source_json_sha256",
        "source_npz_sha256",
        "source_scene_sha256",
        "source_scene_physical_sha256",
        "source_objective_instance_sha256",
        "current_scene_sha256",
        "current_scene_physical_sha256",
    ):
        _require_sha256(record[name], name=f"compact static Newton {name}")
    current_physical = _scene_physical_sha256(scene)
    current_scene = str(scene.manifest()["scene_sha256"])
    current_objective = str(common_objective_manifest(scene, build_common_problem(scene))["objective_instance_sha256"])
    current_identities = (current_scene, current_scene, current_objective)
    actual_identities = (
        record["source_scene_sha256"],
        record["current_scene_sha256"],
        record["source_objective_instance_sha256"],
    )
    if require_current:
        allowed_identities = {current_identities}
        if expected_revision is not None:
            allowed_identities.add(_historical_static_reference_identities(scene, expected_revision))
        if actual_identities not in allowed_identities:
            raise ValueError("static Newton scene manifest or objective is not current")
    else:
        if expected_revision is None:
            raise ValueError("historical static Newton validation requires its recorded revision")
        if actual_identities != _historical_static_reference_identities(scene, expected_revision):
            raise ValueError("historical static Newton scene or objective identity is not canonical")
    if record["source_scene_physical_sha256"] != current_physical:
        raise ValueError("static Newton source physical scene is not current")
    if record["current_scene_physical_sha256"] != current_physical:
        raise ValueError("static Newton current physical scene digest is stale")
    for role in ("dense", "sparse"):
        item = _require_exact_keys(
            record[role],
            {
                "method",
                "source_record_sha256",
                "position_sha256",
                "objective",
                "gradient_norm",
                "relative_residual",
                "determinant_min",
                "inverted_tet_fraction",
                "max_pin_error_m",
            },
            name=f"compact static Newton {role}",
        )
        if item["method"] != _STATIC_REFERENCE_METHODS[role][0]:
            raise ValueError(f"compact static Newton {role} method is not canonical")
        for name in ("source_record_sha256", "position_sha256"):
            _require_sha256(item[name], name=f"compact static Newton {role} {name}")
        for name in (
            "objective",
            "gradient_norm",
            "relative_residual",
            "determinant_min",
            "inverted_tet_fraction",
            "max_pin_error_m",
        ):
            _require_finite_number(item[name], name=f"compact static Newton {role} {name}")
        if item["inverted_tet_fraction"] != 0.0 or item["determinant_min"] <= 0.0:
            raise ValueError(f"compact static Newton {role} has an inverted or non-positive-determinant tetrahedron")
        if item["max_pin_error_m"] != 0.0:
            raise ValueError(f"compact static Newton {role} does not preserve exact pins")
    comparison = _require_exact_keys(
        record["comparison"], set(_STATIC_COMPARISON_NAMES), name="compact static Newton comparison"
    )
    for name in _STATIC_COMPARISON_NAMES:
        _require_finite_number(
            comparison[name],
            name=f"compact static Newton comparison {name}",
            nonnegative=name in {"free_max_m", "free_rms_m"},
        )
    recomputed_deltas = {
        "gradient_delta_N_sparse_minus_dense": record["sparse"]["gradient_norm"] - record["dense"]["gradient_norm"],
        "objective_delta_sparse_minus_dense": record["sparse"]["objective"] - record["dense"]["objective"],
    }
    for name, expected in recomputed_deltas.items():
        if not math.isclose(float(comparison[name]), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise ValueError(f"compact static Newton comparison {name} is stale")


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    finite: bool = True,
) -> np.ndarray:
    value = arrays[name]
    if type(value) is not np.ndarray or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f"recording array {name!r} must have dtype {np.dtype(dtype).name} and shape {shape}")
    if finite and value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValueError(f"recording array {name!r} must be finite")
    return value


def _validate_metric_arrays(
    arrays: Mapping[str, np.ndarray],
    scene: TetBenchmarkScene,
    stored_frames: int,
) -> None:
    for name in METRIC_NAMES:
        values = _require_array(
            arrays,
            f"metric_{name}",
            dtype=np.float64,
            shape=(len(METHOD_IDS), stored_frames),
            finite=False,
        )
        if not np.isnan(values[:, 0]).all() or not np.isfinite(values[:, 1:]).all():
            raise ValueError(f"metric_{name} must be NaN only on the initial frame")

    if np.any(arrays["metric_determinant_min"][:, 1:] <= 0.0):
        raise ValueError("recording trajectories contain a non-positive tetrahedron determinant")
    if np.any(arrays["metric_inverted_tet_fraction"][:, 1:] != 0.0):
        raise ValueError("recording trajectories contain inverted tetrahedra")

    for frame in range(1, stored_frames):
        reference_positions = arrays["positions"][0, frame]
        for method in range(len(METHOD_IDS)):
            input_positions = arrays["objective_input_positions"][method, frame]
            tets = scene.tet_indices
            edges = np.stack(
                (
                    input_positions[tets[:, 1]] - input_positions[tets[:, 0]],
                    input_positions[tets[:, 2]] - input_positions[tets[:, 0]],
                    input_positions[tets[:, 3]] - input_positions[tets[:, 0]],
                ),
                axis=2,
            )
            if np.any(np.linalg.det(edges @ scene.tet_poses) <= 0.0):
                raise ValueError("recording objective-input trajectory contains an inverted tetrahedron")
            objective_scene = dataclasses.replace(
                scene,
                x_current=input_positions,
                velocity=arrays["objective_input_velocities"][method, frame],
                pin_targets=arrays["pin_targets"][frame],
            )
            recomputed = evaluate_common_state(
                build_common_problem(objective_scene),
                arrays["positions"][method, frame],
                reference_positions=reference_positions,
            )
            for name in METRIC_NAMES:
                actual = float(arrays[f"metric_{name}"][method, frame])
                expected = getattr(recomputed, name)
                if expected is None or not math.isclose(actual, float(expected), rel_tol=1.0e-12, abs_tol=1.0e-14):
                    raise ValueError(f"stored {name} does not reproduce for method {method}, frame {frame}")


def _validate_array_manifest(value: object, arrays: Mapping[str, np.ndarray]) -> None:
    manifest = _require_exact_keys(value, set(_ARRAY_NAMES), name="recording array manifest")
    for name, array in arrays.items():
        item = _require_exact_keys(
            manifest[name], {"dtype", "shape", "array_sha256"}, name=f"recording array manifest {name!r}"
        )
        if item["dtype"] != array.dtype.name or _canonical_json(item["shape"]) != _canonical_json(list(array.shape)):
            raise ValueError(f"recording array {name!r} dtype/shape differs from its manifest")
        _require_sha256(item["array_sha256"], name=f"recording array {name!r} SHA-256")
        if item["array_sha256"] != array_sha256(array):
            raise ValueError(f"recording array {name!r} digest differs from its manifest")


def _validate_recording_bundle(
    record_value: Mapping[str, object],
    arrays_value: Mapping[str, np.ndarray],
    *,
    sealed: bool,
    require_current_generation_source: bool = False,
    gaia_asset_root: str | os.PathLike[str] | None = None,
) -> None:
    """Reject a byte-valid recording whose physical or presentation semantics drifted."""
    schema = record_value.get("schema")
    if schema == SCHEMA:
        expected_record_names = _SEALED_RECORD_NAMES if sealed else _UNSEALED_RECORD_NAMES
    elif schema == SCHEMA_V1:
        expected_record_names = _SEALED_RECORD_NAMES_V1 if sealed else _UNSEALED_RECORD_NAMES_V1
    else:
        raise ValueError(f"recording must use schema {SCHEMA!r} or {SCHEMA_V1!r}")
    record = _require_exact_keys(record_value, expected_record_names, name="recording metadata")
    arrays = _require_exact_keys(arrays_value, set(_ARRAY_NAMES), name="recording arrays")
    scene_key = record["scene_key"]
    if type(scene_key) is not str:
        raise ValueError("recording scene_key must be a string")
    spec = recording_spec(scene_key)
    if sealed:
        npz_sha256 = _require_sha256(record["npz_file_sha256"], name="recording NPZ SHA-256")
        if record["npz_filename"] != f"{scene_key}-{npz_sha256}.npz":
            raise ValueError("recording NPZ filename is not exactly content addressed")
        _validate_array_manifest(record["arrays"], arrays)
        expected_record_sha256 = _require_sha256(record["record_sha256"], name="recording record SHA-256")
        unsigned = dict(record)
        del unsigned["record_sha256"]
        if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != expected_record_sha256:
            raise ValueError("recording semantic digest does not match")
    scene = build_recording_scene(scene_key, gaia_asset_root=gaia_asset_root)
    if record["scene_display_name"] != spec.display_name:
        raise ValueError("recording scene display name is not canonical")
    revision = _require_git_revision(record["git_revision"], name="recording Git revision")
    _validate_scene_manifest(
        record["scene_manifest"],
        scene,
        require_current=not sealed,
        expected_revision=revision,
    )
    if record["scene_physical_sha256"] != _scene_physical_sha256(scene):
        raise ValueError("recording physical scene digest differs from the current scene")
    generation_source = _validate_generation_source_manifest(record["generation_source"])
    if schema == SCHEMA_V1:
        if scene_key in _GAIA_SCENE_ASSETS or generation_source["contract"] != GENERATION_SOURCE_SCHEMA_V2:
            raise ValueError("v1 recordings require a non-Gaia v2 generation source contract")
    else:
        _validate_gaia_asset_bundle_manifest(record["gaia_asset_bundle"], scene_key)
        if _canonical_json(record["execution_authentication"]) != _canonical_json(_execution_authentication_policy()):
            raise ValueError("recording execution-authentication boundary is not canonical")
        if generation_source["contract"] != GENERATION_SOURCE_SCHEMA:
            raise ValueError("v2 recordings require the v3 generation source contract")
        if scene_key in _GAIA_SCENE_ASSETS:
            _verify_gaia_asset_bundle_files(_resolve_gaia_asset_root(gaia_asset_root))
    if generation_source["git_revision"] != revision:
        raise ValueError("recording Git revision differs from its generation source")
    if require_current_generation_source and _canonical_json(generation_source) != _canonical_json(
        _generation_source_manifest()
    ):
        raise ValueError("recording generation source changed between simulation and sealing")
    if _canonical_json(record["method_order"]) != _canonical_json(list(METHOD_IDS)) or _canonical_json(
        record["methods"]
    ) != _canonical_json(_method_records(spec)):
        raise ValueError("recording method order or claims are not canonical")
    if _canonical_json(record["reference_policy"]) != _canonical_json(_reference_policy(spec)):
        raise ValueError("recording numerical-reference policy is not canonical")
    if _canonical_json(record["metrics"]) != _canonical_json(_metrics_policy()):
        raise ValueError("recording metric policy is not canonical")

    simulation = _require_exact_keys(
        record["simulation"],
        {
            "source_frames",
            "stored_frames_including_initial",
            "source_frame_rate_hz",
            "substeps_per_source_frame",
            "atomic_dt_seconds",
            "stored_duration_seconds",
            "source_schedule",
        },
        name="recording simulation policy",
    )
    source_frames = simulation["source_frames"]
    stored_frames = simulation["stored_frames_including_initial"]
    if (
        type(source_frames) is not int
        or source_frames < 1
        or type(stored_frames) is not int
        or stored_frames != source_frames + 1
    ):
        raise ValueError("recording frame counts are not canonical")
    if type(simulation["source_frame_rate_hz"]) is not int or simulation["source_frame_rate_hz"] != 60:
        raise ValueError("recording source frame rate must be exactly 60 Hz")
    if (
        type(simulation["substeps_per_source_frame"]) is not int
        or simulation["substeps_per_source_frame"] != spec.substeps_per_source_frame
    ):
        raise ValueError("recording substep count differs from its scene policy")
    if (
        type(simulation["atomic_dt_seconds"]) is not float
        or simulation["atomic_dt_seconds"] != scene.dt
        or _canonical_json(simulation["source_schedule"]) != _canonical_json(_source_schedule(scene_key))
    ):
        raise ValueError("recording timestep or source schedule is not canonical")

    method_count = len(METHOD_IDS)
    vertex_count = scene.n_vertices
    pin_count = scene.pinned_indices.size
    state_shape = (method_count, stored_frames, vertex_count, 3)
    positions = _require_array(arrays, "positions", dtype=np.float64, shape=state_shape)
    velocities = _require_array(arrays, "velocities", dtype=np.float64, shape=state_shape)
    objective_positions = _require_array(arrays, "objective_input_positions", dtype=np.float64, shape=state_shape)
    objective_velocities = _require_array(arrays, "objective_input_velocities", dtype=np.float64, shape=state_shape)
    pin_targets = _require_array(arrays, "pin_targets", dtype=np.float64, shape=(stored_frames, pin_count, 3))
    time_seconds = _require_array(arrays, "time_seconds", dtype=np.float64, shape=(stored_frames,))
    source_indices = _require_array(arrays, "source_frame_index", dtype=np.int64, shape=(stored_frames,), finite=False)
    solve_seconds = _require_array(arrays, "solve_seconds", dtype=np.float64, shape=(method_count, stored_frames))
    transfer_seconds = _require_array(arrays, "transfer_seconds", dtype=np.float64, shape=(method_count, stored_frames))
    if np.any(solve_seconds < 0.0) or np.any(transfer_seconds < 0.0):
        raise ValueError("recording timings must be non-negative")
    if np.any(solve_seconds[:, 0] != 0.0) or np.any(transfer_seconds[:, 0] != 0.0):
        raise ValueError("recording initial-frame timings must be zero")

    expected_times = np.arange(stored_frames, dtype=np.float64) * spec.substeps_per_source_frame * scene.dt
    if not np.array_equal(time_seconds, expected_times):
        raise ValueError("recording time grid is not canonical")
    if type(simulation["stored_duration_seconds"]) is not float or simulation["stored_duration_seconds"] != float(
        expected_times[-1]
    ):
        raise ValueError("recording stored duration differs from its time grid")
    if not np.array_equal(source_indices, np.arange(-1, source_frames, dtype=np.int64)):
        raise ValueError("recording source-frame indices are not canonical")
    for frame in range(stored_frames):
        schedule_frame = max(0, frame - 1)
        expected_targets = pin_targets_for_frame(scene_key, scene, schedule_frame)
        if not np.array_equal(pin_targets[frame], expected_targets):
            raise ValueError(f"recording pin schedule differs at stored frame {frame}")
    expected_all_pins = np.broadcast_to(pin_targets[None, :, :, :], (method_count, stored_frames, pin_count, 3))
    if not np.array_equal(positions[:, :, scene.pinned_indices], expected_all_pins):
        raise ValueError("recording trajectories do not preserve exact pin targets")
    if np.any(velocities[:, :, scene.pinned_indices] != 0.0):
        raise ValueError("recording trajectories do not preserve zero pinned velocities")
    expected_objective_pins = np.broadcast_to(
        pin_targets[None, 1:, :, :], (method_count, stored_frames - 1, pin_count, 3)
    )
    if not np.array_equal(objective_positions[:, 1:, scene.pinned_indices], expected_objective_pins):
        raise ValueError("recording objective inputs do not preserve exact pin targets")
    if np.any(objective_velocities[:, 1:, scene.pinned_indices] != 0.0):
        raise ValueError("recording objective inputs do not preserve zero pinned velocities")
    initial_positions = np.array(scene.x_current, dtype=np.float32).astype(np.float64)
    initial_positions[scene.pinned_indices] = pin_targets[0]
    initial_velocities = np.array(scene.velocity, dtype=np.float32).astype(np.float64)
    initial_velocities[scene.pinned_indices] = 0.0
    if not np.array_equal(positions[:, 0], np.broadcast_to(initial_positions, positions[:, 0].shape)):
        raise ValueError("recording initial positions differ from the canonical scene")
    if not np.array_equal(velocities[:, 0], np.broadcast_to(initial_velocities, velocities[:, 0].shape)):
        raise ValueError("recording initial velocities differ from the canonical scene")
    if not np.array_equal(objective_positions[:, 0], positions[:, 0]) or not np.array_equal(
        objective_velocities[:, 0], velocities[:, 0]
    ):
        raise ValueError("recording initial objective inputs differ from the initial state")

    from .captured_graph_vbd import REASON_NAMES  # noqa: PLC0415

    if record["mg_gate_reason_names"] != list(REASON_NAMES):
        raise ValueError("recording MG gate reason table is not canonical")
    accepted = _require_array(arrays, "mg_last_gate_accepted", dtype=np.int8, shape=(stored_frames, 4), finite=False)
    reasons = _require_array(arrays, "mg_last_gate_reason_code", dtype=np.int16, shape=(stored_frames, 4), finite=False)
    accept_count = _require_array(
        arrays, "mg_frame_gate_accept_count", dtype=np.int64, shape=(stored_frames,), finite=False
    )
    if np.any(accepted[0] != -1) or np.any(reasons[0] != -1) or accept_count[0] != 0:
        raise ValueError("recording initial MG gate sentinels are not canonical")
    if np.any((accepted[1:] < 0) | (accepted[1:] > 1)):
        raise ValueError("recording MG gate acceptance values must be binary")
    if np.any((reasons[1:] < 0) | (reasons[1:] >= len(REASON_NAMES))):
        raise ValueError("recording MG gate reason code is out of range")
    if np.any((accepted[1:] == 1) != (reasons[1:] == REASON_NAMES.index("accepted"))):
        raise ValueError("recording MG gate acceptance and reason codes disagree")
    last_counts = np.count_nonzero(accepted[1:] == 1, axis=1)
    if np.any(accept_count[1:] < last_counts) or np.any(accept_count[1:] > 4 * spec.substeps_per_source_frame):
        raise ValueError("recording MG per-frame gate acceptance count is out of range")

    setup = _require_exact_keys(
        record["setup_seconds_diagnostic"],
        {"reference_public_vbd", "mg_vbd_capture_and_setup", "vbd_k4_public"},
        name="recording setup timings",
    )
    for name, value in setup.items():
        _require_finite_number(value, name=f"recording setup timing {name}", nonnegative=True)
    device = _require_exact_keys(
        record["device"], {"requested", "resolved", "is_cuda", "name"}, name="recording device"
    )
    if any(type(device[name]) is not str or not device[name] for name in ("requested", "resolved", "name")):
        raise ValueError("recording device strings must be non-empty")
    if device["is_cuda"] is not True:
        raise ValueError("recording generation device must be CUDA")

    camera = record["camera"]
    if type(camera) is not dict:
        raise ValueError("recording camera must be a JSON object")
    panel_width, panel_height = _require_encoder_dimensions(camera.get("panel_width"), camera.get("panel_height"))
    expected_camera = fixed_camera(
        positions,
        panel_width=panel_width,
        panel_height=panel_height,
        direction=spec.camera_direction,
    )
    if _canonical_json(camera) != _canonical_json(expected_camera):
        raise ValueError("recording fixed camera does not match the full trajectory union")
    static_reference = record["static_first_step_newton_reference"]
    if static_reference is not None:
        if scene_key != "refinement-medium":
            raise ValueError("static Newton evidence may only accompany refinement-medium")
        _validate_compact_static_reference(
            static_reference,
            scene,
            require_current=not sealed,
            expected_revision=revision,
        )

    _validate_metric_arrays(arrays, scene, stored_frames)


def _write_frozen_content_addressed_bundle(
    directory: pathlib.Path,
    frozen_metadata: dict[str, object],
    canonical_arrays: Mapping[str, np.ndarray],
    gaia_asset_root: pathlib.Path | None,
) -> pathlib.Path:
    scene_key = frozen_metadata.get("scene_key")
    is_gaia = type(scene_key) is str and scene_key in _GAIA_SCENE_ASSETS
    if is_gaia:
        assert gaia_asset_root is not None
        _copy_gaia_asset_bundle(gaia_asset_root, directory)

    with tempfile.NamedTemporaryFile(prefix=".trajectory-", suffix=".npz", dir=directory, delete=False) as stream:
        temporary_npz = pathlib.Path(stream.name)
    try:
        np.savez_compressed(temporary_npz, **canonical_arrays)
        npz_sha256 = _file_sha256(temporary_npz)
        scene_key_name = str(frozen_metadata.get("scene_key", "recording"))
        npz_name = f"{scene_key_name}-{npz_sha256}.npz"
        npz_path = directory / npz_name
        if npz_path.is_symlink():
            raise RuntimeError("existing content-addressed NPZ must not be a symlink")
        if npz_path.exists():
            if _file_sha256(npz_path) != npz_sha256:
                raise RuntimeError("existing content-addressed NPZ failed its filename digest")
            temporary_npz.unlink()
        else:
            os.replace(temporary_npz, npz_path)
    finally:
        if temporary_npz.exists():
            temporary_npz.unlink()

    array_records = {
        name: {
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "array_sha256": array_sha256(array),
        }
        for name, array in canonical_arrays.items()
    }
    payload = dict(frozen_metadata)
    payload.update(
        {
            "npz_filename": npz_name,
            "npz_file_sha256": npz_sha256,
            "arrays": array_records,
        }
    )
    record_sha256 = hashlib.sha256(_canonical_json(payload)).hexdigest()
    payload["record_sha256"] = record_sha256
    json_path = directory / f"{frozen_metadata.get('scene_key', 'recording')}-{record_sha256}.json"
    if json_path.is_symlink():
        raise RuntimeError("existing content-addressed JSON must not be a symlink")
    serialized = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if json_path.exists():
        if json_path.read_bytes() != serialized:
            raise RuntimeError("existing semantic content-addressed JSON has different bytes")
    else:
        with tempfile.NamedTemporaryFile(prefix=".record-", suffix=".json", dir=directory, delete=False) as stream:
            temporary_json = pathlib.Path(stream.name)
            stream.write(serialized)
        try:
            os.replace(temporary_json, json_path)
        finally:
            if temporary_json.exists():
                temporary_json.unlink()
    return json_path


def save_content_addressed_bundle(
    output_directory: str | os.PathLike[str],
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    *,
    gaia_asset_root: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    """Atomically write one self-verifying content-addressed NPZ/JSON pair.

    Args:
        output_directory: Destination directory for the sealed bundle.
        metadata: Unsealed trajectory metadata.
        arrays: Named trajectory arrays.
        gaia_asset_root: Gaia checkout root for a Gaia scene. Exact source
            assets and the license are copied into the sealed bundle.
    """
    frozen_metadata = json.loads(_canonical_json(metadata))
    if type(frozen_metadata) is not dict:
        raise ValueError("recording metadata must be a JSON object")
    directory = pathlib.Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    scene_key = frozen_metadata.get("scene_key")
    is_gaia = type(scene_key) is str and scene_key in _GAIA_SCENE_ASSETS
    resolved_gaia_root = _resolve_gaia_asset_root(gaia_asset_root) if is_gaia else None
    canonical_arrays = {
        name: _canonical_array(np.asarray(value)).copy(order="C") for name, value in sorted(arrays.items())
    }
    _validate_recording_bundle(
        frozen_metadata,
        canonical_arrays,
        sealed=False,
        require_current_generation_source=True,
        gaia_asset_root=resolved_gaia_root,
    )
    _consume_generated_trajectory(metadata, arrays, frozen_metadata, canonical_arrays)
    try:
        result = _write_frozen_content_addressed_bundle(
            directory,
            frozen_metadata,
            canonical_arrays,
            resolved_gaia_root,
        )
    except BaseException:
        _finalize_generated_trajectory(metadata, success=False)
        raise
    _finalize_generated_trajectory(metadata, success=True)
    return result


def load_content_addressed_bundle(
    json_path: str | os.PathLike[str],
    *,
    gaia_asset_root: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load a recording only after verifying every semantic and byte digest.

    Args:
        json_path: Content-addressed recording JSON path.
        gaia_asset_root: Optional Gaia checkout root. A packaged sibling asset
            root is used automatically when present.
    """
    unresolved_path = pathlib.Path(json_path).expanduser()
    if unresolved_path.is_symlink():
        raise ValueError("recording JSON must not be a symlink")
    path = unresolved_path.resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if type(record) is not dict or record.get("schema") not in {SCHEMA, SCHEMA_V1}:
        raise ValueError(f"recording must use schema {SCHEMA!r} or {SCHEMA_V1!r}")
    expected_record_sha256 = _require_sha256(record.get("record_sha256"), name="record SHA-256")
    unsigned = dict(record)
    del unsigned["record_sha256"]
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != expected_record_sha256:
        raise ValueError("recording JSON semantic digest does not match")
    if path.name != f"{record.get('scene_key')}-{expected_record_sha256}.json":
        raise ValueError("recording JSON filename is not content addressed")
    _verify_generation_source_git_objects(record.get("generation_source"))

    npz_sha256 = _require_sha256(record.get("npz_file_sha256"), name="NPZ SHA-256")
    expected_npz_filename = f"{record.get('scene_key')}-{npz_sha256}.npz"
    if record.get("npz_filename") != expected_npz_filename:
        raise ValueError("recording NPZ filename is not content addressed")
    npz_path = path.parent / expected_npz_filename
    if npz_path.is_symlink():
        raise ValueError("recording NPZ must not be a symlink")
    if _file_sha256(npz_path) != npz_sha256:
        raise ValueError("recording NPZ bytes or filename do not match")
    array_records = record.get("arrays")
    if type(array_records) is not dict:
        raise ValueError("recording array manifest is missing")
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(array_records):
            raise ValueError("recording NPZ keys differ from its manifest")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    for name, array in arrays.items():
        item = array_records[name]
        if type(item) is not dict:
            raise ValueError(f"recording array manifest for {name!r} is invalid")
        if item.get("dtype") != array.dtype.name or _canonical_json(item.get("shape")) != _canonical_json(
            list(array.shape)
        ):
            raise ValueError(f"recording array {name!r} dtype/shape differs from its manifest")
        if item.get("array_sha256") != array_sha256(array):
            raise ValueError(f"recording array {name!r} digest differs from its manifest")
    resolved_gaia_root = _bundle_gaia_asset_root(record.get("scene_key"), path.parent, gaia_asset_root)
    _validate_recording_bundle(record, arrays, sealed=True, gaia_asset_root=resolved_gaia_root)
    return record, arrays


def _render_source_manifest(tools_root: pathlib.Path) -> dict[str, object]:
    files = {
        "research/principal_stretch/record_mg_vbd_comparison.py": pathlib.Path(__file__).resolve(),
        "newton_capture/__init__.py": tools_root / "newton_capture/__init__.py",
        "newton_capture/_deps.py": tools_root / "newton_capture/_deps.py",
        "newton_capture/_display.py": tools_root / "newton_capture/_display.py",
        "newton_capture/_video.py": tools_root / "newton_capture/_video.py",
    }
    for name, path in files.items():
        if not path.is_file():
            raise RuntimeError(f"render source is missing: {name}")
    payload: dict[str, object] = {
        "contract": RENDER_SOURCE_SCHEMA,
        "files": {name: _file_sha256(path) for name, path in files.items()},
        "newton_version": _version_string(newton, "newton"),
        "warp_version": _version_string(wp, "warp-lang"),
        "pillow_version": _distribution_version("Pillow"),
        "imageio_version": _distribution_version("imageio"),
        "imageio_ffmpeg_version": _distribution_version("imageio-ffmpeg"),
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_render_source_manifest(value: object) -> dict[str, object]:
    manifest = _require_exact_keys(
        value,
        {
            "contract",
            "files",
            "newton_version",
            "warp_version",
            "pillow_version",
            "imageio_version",
            "imageio_ffmpeg_version",
            "manifest_sha256",
        },
        name="render source manifest",
    )
    if manifest["contract"] != RENDER_SOURCE_SCHEMA:
        raise ValueError("render source contract is not canonical")
    expected_files = {
        "research/principal_stretch/record_mg_vbd_comparison.py",
        "newton_capture/__init__.py",
        "newton_capture/_deps.py",
        "newton_capture/_display.py",
        "newton_capture/_video.py",
    }
    files = _require_exact_keys(manifest["files"], expected_files, name="render source files")
    for name, value_sha256 in files.items():
        _require_sha256(value_sha256, name=f"render source {name} SHA-256")
    for name in ("newton_version", "warp_version", "pillow_version", "imageio_version", "imageio_ffmpeg_version"):
        if type(manifest[name]) is not str or not manifest[name]:
            raise ValueError(f"render source {name} must be a non-empty string")
    expected = _require_sha256(manifest["manifest_sha256"], name="render source manifest SHA-256")
    unsigned = dict(manifest)
    del unsigned["manifest_sha256"]
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != expected:
        raise ValueError("render source manifest digest does not match")
    return manifest


def _resolve_render_fps(record: Mapping[str, object], requested_fps: int | None) -> tuple[int, float]:
    simulation = record.get("simulation")
    if type(simulation) is not dict or type(simulation.get("source_frame_rate_hz")) is not int:
        raise ValueError("recording source frame rate is missing")
    source_fps = simulation["source_frame_rate_hz"]
    fps = source_fps if requested_fps is None else requested_fps
    if type(fps) is not int or fps < 1:
        raise ValueError("fps must be a positive built-in int or None")
    return fps, fps / source_fps


def _playback_annotation(playback_rate: float) -> str:
    if not math.isfinite(playback_rate) or playback_rate <= 0.0:
        raise ValueError("playback rate must be finite and positive")
    if math.isclose(playback_rate, 1.0, rel_tol=0.0, abs_tol=1.0e-15):
        return ""
    qualifier = "slow motion" if playback_rate < 1.0 else "fast motion"
    return f"playback {playback_rate:.3g}x {qualifier}"


def _render_sidecar_path(output: pathlib.Path) -> pathlib.Path:
    return output.with_name(f"{output.name}.render.json")


def _inspect_encoded_mp4(
    path: pathlib.Path,
    *,
    expected_fps: int,
    expected_frame_count: int,
    expected_width: int,
    expected_height: int,
) -> dict[str, object]:
    import imageio_ffmpeg  # noqa: PLC0415

    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    video_lines = [line for line in probe.stderr.splitlines() if " Video: " in line]
    if len(video_lines) != 1:
        raise RuntimeError(f"encoded MP4 has {len(video_lines)} video streams, expected exactly one")
    match = re.search(
        r"Video:\s*(?P<codec>[^,\s]+).*?,\s*"
        r"(?P<pixel_format>[A-Za-z0-9_]+)(?:\([^)]*\))?,\s*"
        r"(?P<width>\d+)x(?P<height>\d+).*?,\s*"
        r"(?P<fps>[0-9]+(?:\.[0-9]+)?)\s+fps\b",
        video_lines[0],
    )
    if match is None:
        raise RuntimeError(f"could not parse encoded MP4 video stream: {video_lines[0].strip()}")
    frame_count, duration_seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    codec = match.group("codec")
    pixel_format = match.group("pixel_format")
    fps = float(match.group("fps"))
    size = (int(match.group("width")), int(match.group("height")))
    if codec != "h264" or pixel_format != "yuv420p":
        raise RuntimeError(f"encoded MP4 stream is {codec}/{pixel_format}, expected h264/yuv420p")
    if not math.isclose(float(fps), expected_fps, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"encoded MP4 rate is {fps}, expected {expected_fps} fps")
    if frame_count != expected_frame_count:
        raise RuntimeError(f"encoded MP4 has {frame_count} frames, expected {expected_frame_count}")
    if tuple(size) != (expected_width, expected_height):
        raise RuntimeError(f"encoded MP4 size is {size}, expected {(expected_width, expected_height)}")
    return {
        "codec": codec,
        "pixel_format": pixel_format,
        "encoded_duration_seconds": float(duration_seconds),
    }


def _build_render_record(
    *,
    bundle_json: pathlib.Path,
    bundle_record_sha256: str,
    output_mp4: pathlib.Path,
    fps: int,
    source_fps: int,
    frame_count: int,
    width: int,
    height: int,
    render_source: Mapping[str, object],
    encoded_stream: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": RENDER_RECORD_SCHEMA,
        "bundle_json_relative_path": os.path.relpath(bundle_json, output_mp4.parent),
        "bundle_json_file_sha256": _file_sha256(bundle_json),
        "bundle_record_sha256": _require_sha256(bundle_record_sha256, name="render bundle record SHA-256"),
        "mp4_filename": output_mp4.name,
        "mp4_file_sha256": _file_sha256(output_mp4),
        "fps": fps,
        "source_frame_rate_hz": source_fps,
        "playback_rate": fps / source_fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "encoder": "libx264",
        "codec": encoded_stream["codec"],
        "pixel_format": encoded_stream["pixel_format"],
        "encoded_duration_seconds": encoded_stream["encoded_duration_seconds"],
        "render_source": dict(render_source),
    }
    payload["render_record_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_render_record(value: object) -> dict[str, object]:
    record = _require_exact_keys(
        value,
        {
            "schema",
            "bundle_json_relative_path",
            "bundle_json_file_sha256",
            "bundle_record_sha256",
            "mp4_filename",
            "mp4_file_sha256",
            "fps",
            "source_frame_rate_hz",
            "playback_rate",
            "frame_count",
            "width",
            "height",
            "encoder",
            "codec",
            "pixel_format",
            "encoded_duration_seconds",
            "render_source",
            "render_record_sha256",
        },
        name="render record",
    )
    if record["schema"] != RENDER_RECORD_SCHEMA:
        raise ValueError("render record schema is not canonical")
    for name in ("bundle_json_file_sha256", "bundle_record_sha256", "mp4_file_sha256"):
        _require_sha256(record[name], name=f"render record {name}")
    bundle_relative = record["bundle_json_relative_path"]
    if type(bundle_relative) is not str or not bundle_relative or pathlib.Path(bundle_relative).is_absolute():
        raise ValueError("render record bundle_json_relative_path must be a non-empty relative path")
    if (
        type(record["mp4_filename"]) is not str
        or not record["mp4_filename"]
        or pathlib.Path(record["mp4_filename"]).name != record["mp4_filename"]
    ):
        raise ValueError("render record mp4_filename must be one basename")
    fps = record["fps"]
    source_fps = record["source_frame_rate_hz"]
    frame_count = record["frame_count"]
    width = record["width"]
    height = record["height"]
    if any(type(value_int) is not int or value_int < 1 for value_int in (fps, source_fps, frame_count)):
        raise ValueError("render record rates and frame count must be positive built-in integers")
    _require_encoder_dimensions(width, height)
    expected_rate = fps / source_fps
    if record["playback_rate"] != expected_rate:
        raise ValueError("render record playback rate differs from its frame rates")
    if record["encoder"] != "libx264" or record["codec"] != "h264" or record["pixel_format"] != "yuv420p":
        raise ValueError("render record codec configuration is not canonical")
    duration = _require_finite_number(
        record["encoded_duration_seconds"], name="render record encoded duration", nonnegative=True
    )
    if duration == 0.0:
        raise ValueError("render record encoded duration must be positive")
    expected_duration = frame_count / fps
    if abs(duration - expected_duration) > 1.0 / fps:
        raise ValueError("render record encoded duration differs from its frame count and rate")
    _validate_render_source_manifest(record["render_source"])
    expected = _require_sha256(record["render_record_sha256"], name="render record SHA-256")
    unsigned = dict(record)
    del unsigned["render_record_sha256"]
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != expected:
        raise ValueError("render record semantic digest does not match")
    return record


def _save_render_record(path: pathlib.Path, record: Mapping[str, object]) -> pathlib.Path:
    _validate_render_record(record)
    serialized = (json.dumps(dict(record), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(prefix=".render-", suffix=".json", dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        stream.write(serialized)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_render_record(
    path: str | os.PathLike[str],
    *,
    gaia_asset_root: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Load and verify an MP4 render sidecar and both files it binds.

    Args:
        path: Render sidecar JSON path.
        gaia_asset_root: Optional Gaia checkout root for a Gaia recording.
    """
    sidecar = pathlib.Path(path).resolve()
    record = _validate_render_record(json.loads(sidecar.read_text(encoding="utf-8")))
    if sidecar.name != f"{record['mp4_filename']}.render.json":
        raise ValueError("render sidecar filename does not match its MP4")
    bundle = (sidecar.parent / record["bundle_json_relative_path"]).resolve()
    output = sidecar.parent / record["mp4_filename"]
    if _file_sha256(bundle) != record["bundle_json_file_sha256"]:
        raise ValueError("render record bundle JSON bytes differ")
    bundle_record, bundle_arrays = load_content_addressed_bundle(bundle, gaia_asset_root=gaia_asset_root)
    if bundle_record["record_sha256"] != record["bundle_record_sha256"]:
        raise ValueError("render record bundle semantic identity differs")
    if record["source_frame_rate_hz"] != bundle_record["simulation"]["source_frame_rate_hz"]:
        raise ValueError("render record source rate differs from its bundle")
    camera = bundle_record["camera"]
    expected_render_shape = (
        int(camera["panel_width"]) * len(METHOD_IDS),
        int(camera["panel_height"]),
        int(bundle_arrays["positions"].shape[1]),
    )
    if (record["width"], record["height"], record["frame_count"]) != expected_render_shape:
        raise ValueError("render record dimensions or frame count differ from its bundle")
    if _file_sha256(output) != record["mp4_file_sha256"]:
        raise ValueError("render record MP4 bytes differ")
    encoded_stream = _inspect_encoded_mp4(
        output,
        expected_fps=record["fps"],
        expected_frame_count=record["frame_count"],
        expected_width=record["width"],
        expected_height=record["height"],
    )
    if encoded_stream["codec"] != record["codec"] or encoded_stream["pixel_format"] != record["pixel_format"]:
        raise ValueError("render record codec metadata differs from its MP4")
    if not math.isclose(
        float(encoded_stream["encoded_duration_seconds"]),
        float(record["encoded_duration_seconds"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("render record duration differs from its MP4")
    return record


def _load_font(size: int, *, bold: bool = False):
    from PIL import ImageFont  # noqa: PLC0415

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    if not bold:
        candidates = candidates[::-1]
    for candidate in candidates:
        if pathlib.Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _format_metric(value: float, *, scale: float = 1.0, suffix: str = "") -> str:
    if not math.isfinite(value):
        return "n/a"
    scaled = value * scale
    if scaled == 0.0:
        return f"0{suffix}"
    if abs(scaled) < 1.0e-2 or abs(scaled) >= 1.0e4:
        return f"{scaled:.2e}{suffix}"
    return f"{scaled:.3g}{suffix}"


def _apply_header_safe_area(frame: np.ndarray) -> np.ndarray:
    """Shift the fixed-camera image below overlays without changing its scale."""
    source = np.asarray(frame, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("rendered panel must have RGB shape (H, W, 3)")
    shift = max(1, int(round(source.shape[0] * 0.08)))
    if shift >= source.shape[0]:
        raise ValueError("rendered panel is too short for its header safe area")
    corners = np.concatenate(
        (
            source[:8, :8].reshape(-1, 3),
            source[:8, -8:].reshape(-1, 3),
            source[-8:, :8].reshape(-1, 3),
            source[-8:, -8:].reshape(-1, 3),
        )
    )
    background = np.median(corners, axis=0).astype(np.uint8)
    shifted = np.empty_like(source)
    shifted[:] = background
    shifted[shift:] = source[:-shift]
    return shifted


def label_panel(
    frame: np.ndarray,
    *,
    method_index: int,
    frame_index: int,
    record: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Overlay the declared method role and independent metrics on one panel."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    if method_index not in range(len(METHOD_IDS)):
        raise ValueError("method_index is outside the three-panel layout")
    image = Image.fromarray(_apply_header_safe_area(frame), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(image, mode="RGBA")
    width, height = image.size
    title_font = _load_font(max(18, width // 24), bold=True)
    body_font = _load_font(max(13, width // 38))
    small_font = _load_font(max(11, width // 46))
    colors = ((255, 214, 92, 255), (92, 220, 255, 255), (255, 155, 92, 255))
    method = record["methods"][method_index]

    draw.rectangle((0, 0, width, 142), fill=(8, 13, 20, 218))
    draw.rectangle((0, 0, width, 6), fill=colors[method_index])
    draw.text((18, 14), method["panel_title"], font=title_font, fill=colors[method_index])
    if method_index == 0:
        subtitle = f"Public VBD K{method['iterations_per_atomic_step']} · numerical only"
    elif method_index == 1:
        subtitle = "Captured rollout · 4 outer x 2 V-cycles"
    else:
        subtitle = "Public SolverVBD · 4 sweeps"
    draw.text((18, 54), subtitle, font=small_font, fill=(225, 230, 238, 255))

    relative = float(arrays["metric_relative_residual"][method_index, frame_index])
    rms = float(arrays["metric_free_rms_error_m"][method_index, frame_index])
    determinant = float(arrays["metric_determinant_min"][method_index, frame_index])
    inversions = float(arrays["metric_inverted_tet_fraction"][method_index, frame_index])
    pin_error = float(arrays["metric_max_pin_error_m"][method_index, frame_index])
    first_line = (
        f"rel residual  {_format_metric(relative)}    RMS to ref  {_format_metric(rms, scale=1000.0, suffix=' mm')}"
    )
    pin_text = "exact" if pin_error == 0.0 else _format_metric(pin_error, scale=1000.0, suffix=" mm")
    second_line = (
        f"min det(F)  {_format_metric(determinant)}    inverted  "
        f"{_format_metric(inversions, scale=100.0, suffix='%')}    pins  {pin_text}"
    )
    draw.text((18, 82), first_line, font=body_font, fill=(248, 248, 248, 255))
    draw.text((18, 108), second_line, font=body_font, fill=(248, 248, 248, 255))

    if method_index == 1 and frame_index > 0:
        accepted = arrays["mg_last_gate_accepted"][frame_index]
        gate_text = f"last atomic gates: {int(np.count_nonzero(accepted == 1))}/{accepted.size} accepted"
        footer_height = 62 if _gaia_protocol_annotation(record.get("scene_key")) is not None else 34
        badge_bottom = height - footer_height - 10
        draw.rounded_rectangle((16, badge_bottom - 38, width - 16, badge_bottom), radius=8, fill=(8, 13, 20, 205))
        draw.text((28, badge_bottom - 30), gate_text, font=small_font, fill=colors[method_index])
    return np.asarray(image.convert("RGB"))


def label_composite(
    frame: np.ndarray,
    *,
    frame_index: int,
    record: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    playback_rate: float = 1.0,
) -> np.ndarray:
    """Add the common time/scope footer to one three-panel frame."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(image, mode="RGBA")
    width, height = image.size
    font = _load_font(max(12, width // 145))
    source_frame = int(arrays["source_frame_index"][frame_index])
    time_value = float(arrays["time_seconds"][frame_index])
    frame_label = "initial state" if source_frame < 0 else f"source frame {source_frame}"
    left = f"{record['scene_display_name']}  ·  {frame_label}  ·  t={time_value:.4f} s"
    playback = _playback_annotation(playback_rate)
    if playback:
        left += f"  ·  {playback}"
    right = "* high-budget public VBD trajectory; numerical comparison, not Newton or ground truth"
    protocol = _gaia_protocol_annotation(record.get("scene_key"))
    footer_height = 62 if protocol is not None else 34
    draw.rectangle((0, height - footer_height, width, height), fill=(4, 7, 12, 230))
    if protocol is not None:
        draw.text((16, height - 55), protocol, font=font, fill=(170, 220, 255, 255))
    draw.text((16, height - 27), left, font=font, fill=(230, 235, 242, 255))
    right_box = draw.textbbox((0, 0), right, font=font)
    draw.text((width - (right_box[2] - right_box[0]) - 16, height - 27), right, font=font, fill=(255, 214, 92, 255))
    return np.asarray(image.convert("RGB"))


def render_bundle(
    bundle_json: str | os.PathLike[str],
    output_mp4: str | os.PathLike[str],
    *,
    fps: int | None = None,
    gaia_asset_root: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    """Render one verified trajectory bundle using a single fixed-camera viewer.

    Args:
        bundle_json: Content-addressed recording JSON path.
        output_mp4: Destination MP4 path.
        fps: Output frame rate. Uses the source rate when omitted.
        gaia_asset_root: Optional Gaia checkout root. Packaged bundle assets
            are used automatically when present.
    """
    bundle = pathlib.Path(bundle_json).resolve()
    record, arrays = load_content_addressed_bundle(bundle, gaia_asset_root=gaia_asset_root)
    resolved_fps, playback_rate = _resolve_render_fps(record, fps)
    if record.get("method_order") != list(METHOD_IDS):
        raise ValueError("recording method order is not Reference | MG-VBD | VBD K4")
    resolved_gaia_root = _bundle_gaia_asset_root(record["scene_key"], bundle.parent, gaia_asset_root)
    scene = build_recording_scene(str(record["scene_key"]), gaia_asset_root=resolved_gaia_root)
    if _scene_physical_sha256(scene) != record.get("scene_physical_sha256"):
        raise ValueError("current render scene physical content differs from the recording")
    positions = arrays["positions"]
    expected_shape = (len(METHOD_IDS), positions.shape[1], scene.n_vertices, 3)
    if positions.shape != expected_shape:
        raise ValueError("recording positions have an invalid three-method shape")

    camera = record["camera"]
    panel_width = int(camera["panel_width"])
    panel_height = int(camera["panel_height"])
    output = pathlib.Path(output_mp4).resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("output path must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    tools_root = pathlib.Path(os.environ.get("AI_LOGS", "/home/horde/Code/AI-Docs/AI-Logs")) / "Newton/tools"
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    from newton_capture import Capture  # noqa: PLC0415
    from newton_capture._video import VideoWriter  # noqa: PLC0415

    model = _build_vbd_model(scene, "cpu")
    state = model.state()
    render_source = None
    with Capture(
        out_dir=str(output.parent),
        width=panel_width,
        height=panel_height,
        camera_pos=tuple(camera["position"]),
        camera_target=tuple(camera["target"]),
        camera_fov=float(camera["fov_degrees"]),
        shading_style="studio",
    ) as capture:
        # Capture installs its imageio-ffmpeg dependency on entry, so seal the
        # renderer versions only after that setup has completed.
        render_source = _render_source_manifest(tools_root)
        viewer = capture._get_viewer(model)
        capture._apply_camera(viewer)
        writer = VideoWriter(str(output), fps=resolved_fps, quality=8)
        writer.open()
        try:
            for frame_index in range(positions.shape[1]):
                panels = []
                for method_index in range(len(METHOD_IDS)):
                    state.particle_q.assign(np.asarray(positions[method_index, frame_index], dtype=np.float32))
                    panel = capture._render_frame(viewer, model, state)
                    panels.append(
                        label_panel(
                            panel,
                            method_index=method_index,
                            frame_index=frame_index,
                            record=record,
                            arrays=arrays,
                        )
                    )
                composite = label_composite(
                    np.concatenate(panels, axis=1),
                    frame_index=frame_index,
                    record=record,
                    arrays=arrays,
                    playback_rate=playback_rate,
                )
                writer.write_frame(composite)
        finally:
            result = pathlib.Path(writer.close()).resolve()
    if result != output or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("MP4 encoding failed; install imageio plus imageio-ffmpeg and rerun render")
    assert render_source is not None
    encoded_stream = _inspect_encoded_mp4(
        output,
        expected_fps=resolved_fps,
        expected_frame_count=positions.shape[1],
        expected_width=panel_width * len(METHOD_IDS),
        expected_height=panel_height,
    )
    render_record = _build_render_record(
        bundle_json=bundle,
        bundle_record_sha256=record["record_sha256"],
        output_mp4=output,
        fps=resolved_fps,
        source_fps=record["simulation"]["source_frame_rate_hz"],
        frame_count=positions.shape[1],
        width=panel_width * len(METHOD_IDS),
        height=panel_height,
        render_source=render_source,
        encoded_stream=encoded_stream,
    )
    _save_render_record(_render_sidecar_path(output), render_record)
    return output


def _generate_command(args: argparse.Namespace) -> int:
    static_reference = None
    if args.static_reference_json is not None:
        scene = build_recording_scene(args.scene, gaia_asset_root=args.gaia_asset_root)
        static_reference = static_medium_reference_evidence(
            scene,
            args.static_reference_json,
            expected_json_sha256=args.static_reference_json_sha256,
            expected_npz_sha256=args.static_reference_npz_sha256,
            npz_path=args.static_reference_npz,
        )
    metadata, arrays = generate_trajectory(
        args.scene,
        source_frames=args.frames,
        device=args.device,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        static_reference=static_reference,
        gaia_asset_root=args.gaia_asset_root,
    )
    output = save_content_addressed_bundle(
        args.out_dir,
        metadata,
        arrays,
        gaia_asset_root=args.gaia_asset_root,
    )
    print(output)
    return 0


def _render_command(args: argparse.Namespace) -> int:
    print(render_bundle(args.bundle, args.out, fps=args.fps, gaia_asset_root=args.gaia_asset_root))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate and seal one quantitative trajectory")
    generate.add_argument("--scene", choices=tuple(_SPECS), required=True)
    generate.add_argument("--frames", type=int, required=True, help="number of original 60 Hz source frames")
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument("--panel-width", type=int, default=640)
    generate.add_argument("--panel-height", type=int, default=720)
    generate.add_argument("--out-dir", required=True)
    generate.add_argument("--static-reference-json")
    generate.add_argument("--static-reference-npz")
    generate.add_argument("--static-reference-json-sha256")
    generate.add_argument("--static-reference-npz-sha256")
    generate.add_argument(
        "--gaia-asset-root",
        help="root of the digest-pinned Gaia checkout; defaults to PSS_GAIA_ASSET_ROOT for Gaia scenes",
    )
    generate.set_defaults(handler=_generate_command)

    render = commands.add_parser("render", help="render one verified bundle to MP4")
    render.add_argument("--bundle", required=True)
    render.add_argument("--out", required=True)
    render.add_argument(
        "--fps",
        type=int,
        default=None,
        help="output rate; defaults to the bundle's 60 Hz source rate, with explicit retiming labeled on-frame",
    )
    render.add_argument(
        "--gaia-asset-root",
        help="root of the digest-pinned Gaia checkout; defaults to PSS_GAIA_ASSET_ROOT for Gaia scenes",
    )
    render.set_defaults(handler=_render_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the quantitative generator or verified-bundle renderer."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    static_values = (
        getattr(args, "static_reference_json", None),
        getattr(args, "static_reference_json_sha256", None),
        getattr(args, "static_reference_npz_sha256", None),
    )
    if any(value is not None for value in static_values) and not all(value is not None for value in static_values):
        parser.error("static reference JSON and both expected file SHA-256 values must be supplied together")
    if getattr(args, "static_reference_npz", None) is not None and getattr(args, "static_reference_json", None) is None:
        parser.error("a static reference NPZ override requires the authenticated static reference JSON")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
