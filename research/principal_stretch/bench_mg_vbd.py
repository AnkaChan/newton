# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run the authenticated CPU development suite for multiplicative MG-VBD.

The suite deliberately fixes the numerical policy before seeing any result:
three relinearized outer corrections, four PCG iterations per correction, and
one tuned rigid static-rest multigrid hierarchy reused throughout.  Every
scene starts fresh VBD K1, fresh VBD K4, and a fresh dense Newton reference via
:func:`correction_mg_vbd.run_multiplicative_mg_vbd`.

The JSON checkpoint contains the complete timing-free quality records and
separate diagnostic-only CPU timings.  It is content addressed at the scene,
quality, diagnostic, entry, and suite levels.  Partial checkpoints may be
resumed only under the exact same configuration, scene manifests, and pinned
source files.  Completed output is never overwritten.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .correction_mg_vbd import MGVBDCorrectionConfig, MGVBDRunResult, run_multiplicative_mg_vbd
from .solver_benchmark import TetBenchmarkScene
from .solver_scenes import (
    build_compression_scene,
    build_extension_scene,
    build_refinement_scene,
    build_sliver_scene,
    build_stretch_scene,
    build_twist_scene,
)

_SCHEMA_VERSION = 1
_SUITE_CONTRACT = "pss-multiplicative-mg-vbd-cpu-development-suite-v1"
_ENTRY_CONTRACT = "pss-multiplicative-mg-vbd-cpu-development-scene-v1"
_DIAGNOSTIC_CONTRACT = "pss-multiplicative-mg-vbd-cpu-suite-diagnostic-v1"
_FAILURE_DIAGNOSTIC_CONTRACT = "pss-multiplicative-mg-vbd-cpu-suite-failure-diagnostic-v1"
_MAX_NEWTON_FREE_DOFS = 2_000
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
}

# These are the reviewed source contents at the benchmark decision point.  A
# suite cannot silently resume or start against a locally modified solver.
_PINNED_SOURCE_SHA256 = {
    "correction_mg_vbd.py": "ff4ea309392577a68061b8ef0972425755b80d36b66db3bcaad98d5f20aa87f8",
    "correction_gpu.py": "a80e12ab04306c7d5d964902d9e23d30d23b89076af5d090630e64d69110e53f",
    "correction_multigrid.py": "b0d6eee9cc150b5f691950a9f1afaebd8a5ed0b21a7acbe017e534ddf9f3a8a9",
    "solver_benchmark.py": "0ca95df0c511c716aa9b05969bb700cd50ed75c00cc1e37dbe97c7b4498fc877",
    "solver_scenes.py": "aedd680a31e0ed6126ce803bfcf91d47321144639a4b78eba078ef8ae4342c92",
}

_METRIC_FIELDS = (
    "objective",
    "gradient_norm",
    "relative_residual",
    "free_rms_error_m",
    "mass_weighted_rms_error_m",
    "determinant_min",
    "determinant_max",
    "inverted_tet_fraction",
    "minimum_singular_value",
    "max_pin_error_m",
    "position_sha256",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _validate_sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_self_hash(record: Mapping[str, object], field: str, name: str) -> None:
    stored = _validate_sha256(record.get(field), f"{name} {field}")
    payload = dict(record)
    payload.pop(field)
    if _canonical_sha256(payload) != stored:
        raise ValueError(f"{name} does not match its {field}")


def _same_float64_measurement(left: object, right: object) -> bool:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
        or not math.isfinite(left)
        or not math.isfinite(right)
    ):
        return False
    guard = 128.0 * np.finfo(np.float64).eps * max(1.0, abs(left), abs(right))
    return abs(left - right) <= guard


def _fixed_config() -> MGVBDCorrectionConfig:
    config = MGVBDCorrectionConfig()
    config.validate()
    if (
        config.outer_corrections != 3
        or config.correction.pcg_iterations != 4
        or config.mode_kind != "rigid"
        or config.smoother_safety != 0.9
    ):
        raise RuntimeError("MG-VBD defaults no longer match the frozen 3x4 tuned-rigid policy")
    return config


def _fixed_config_record() -> dict[str, object]:
    return _fixed_config().deterministic_record()


def _resolve_scene_keys(keys: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(keys)
    if not resolved:
        raise ValueError("at least one scene must be selected")
    if len(set(resolved)) != len(resolved):
        raise ValueError("scene keys must be distinct")
    unknown = tuple(key for key in resolved if key not in _SCENE_FACTORIES)
    if unknown:
        raise ValueError(f"unknown scene keys {unknown}; expected a subset of {tuple(_SCENE_FACTORIES)}")
    return resolved


def _configuration_record(scene_keys: Sequence[str], max_newton_free_dofs: int) -> dict[str, object]:
    return {
        "scene_keys": list(scene_keys),
        "scene_parameters": "audited-builder-defaults-called-without-overrides",
        "max_newton_free_dofs": max_newton_free_dofs,
        "mg_vbd": _fixed_config_record(),
        "vbd": {
            "device": "cpu",
            "tile_solve": False,
            "k1_iterations": 1,
            "k4_iterations": 4,
            "warmup": False,
            "diagnostic_repeats": 1,
            "restart_policy": "separate-fresh-runs-from-identical-physical-state",
        },
        "reference": {
            "method": "fresh-dense-cpu-newton",
            "warmup": False,
            "diagnostic_repeats": 1,
            "maximum_free_dofs": max_newton_free_dofs,
        },
        "classification": "development-suite-fixed-before-results-not-confirmation-evidence",
        "performance_evidence": False,
    }


def _source_manifest() -> dict[str, object]:
    module_dir = pathlib.Path(__file__).resolve().parent
    records: dict[str, object] = {}
    for filename, expected in _PINNED_SOURCE_SHA256.items():
        path = module_dir / filename
        actual = _file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned source {filename} changed: expected {expected}, found {actual}")
        records[filename] = {"sha256": actual, "pinned_sha256": expected, "reviewed": True}
    benchmark_path = pathlib.Path(__file__).resolve()
    records[benchmark_path.name] = {
        "sha256": _file_sha256(benchmark_path),
        "pinned_sha256": None,
        "reviewed": False,
    }
    payload = {
        "root": "research/principal_stretch",
        "files": records,
    }
    payload["source_manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _verify_source_manifest(record: Mapping[str, object], *, verify_current_sources: bool) -> None:
    _validate_self_hash(record, "source_manifest_sha256", "source manifest")
    files = record.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("source manifest files must be a mapping")
    for filename, expected in _PINNED_SOURCE_SHA256.items():
        item = files.get(filename)
        if not isinstance(item, Mapping):
            raise ValueError(f"source manifest is missing {filename}")
        if item.get("sha256") != expected or item.get("pinned_sha256") != expected or item.get("reviewed") is not True:
            raise ValueError(f"source manifest does not bind reviewed {filename}")
    benchmark_item = files.get(pathlib.Path(__file__).name)
    if not isinstance(benchmark_item, Mapping):
        raise ValueError("source manifest does not bind the benchmark driver")
    _validate_sha256(benchmark_item.get("sha256"), "benchmark driver source SHA-256")
    if benchmark_item.get("pinned_sha256") is not None or benchmark_item.get("reviewed") is not False:
        raise ValueError("benchmark driver source status is invalid")
    if verify_current_sources:
        current = _source_manifest()
        if record != current:
            raise ValueError("suite source manifest does not match current source files")


def _verify_scene_manifest(record: Mapping[str, object]) -> None:
    _validate_self_hash(record, "scene_sha256", "scene manifest")
    for name in ("n_vertices", "n_tets", "n_pinned"):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"scene manifest {name} must be a non-negative integer")
    if record["n_vertices"] < 1 or record["n_tets"] < 1 or record["n_pinned"] >= record["n_vertices"]:
        raise ValueError("scene manifest has invalid vertex, tet, or pin counts")


def _scene_counts(scene: TetBenchmarkScene) -> dict[str, int]:
    free_vertices = int(scene.free_indices.size)
    return {
        "vertices": scene.n_vertices,
        "tetrahedra": scene.n_tets,
        "pinned_vertices": int(scene.pinned_indices.size),
        "free_vertices": free_vertices,
        "free_dofs": 3 * free_vertices,
    }


def _selected_metrics(record: Mapping[str, object], name: str) -> dict[str, object]:
    selected: dict[str, object] = {}
    for field in _METRIC_FIELDS:
        if field not in record:
            raise ValueError(f"{name} is missing metric {field}")
        value = record[field]
        if field == "position_sha256":
            _validate_sha256(value, f"{name} position_sha256")
        elif value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            raise ValueError(f"{name} metric {field} must be finite or null")
        selected[field] = value
    return selected


def _summary_from_quality(quality: Mapping[str, object]) -> dict[str, object]:
    metrics: dict[str, dict[str, object]] = {}
    for role, key in (
        ("reference", "reference_metrics"),
        ("k1", "k1_metrics"),
        ("mg_corrected", "final_metrics"),
        ("k4", "k4_metrics"),
    ):
        value = quality.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"quality record is missing {key}")
        metrics[role] = _selected_metrics(value, key)

    hierarchy = quality.get("hierarchy")
    gate = quality.get("gate")
    outer = quality.get("outer_corrections")
    if not isinstance(hierarchy, Mapping) or not isinstance(gate, Mapping) or not isinstance(outer, list):
        raise ValueError("quality record is missing hierarchy, gate, or outer correction evidence")
    versus_k4 = gate.get("versus_k4")
    if not isinstance(versus_k4, Mapping):
        raise ValueError("quality gate is missing the K4 comparison")

    per_outer = []
    total_operator_applications = 0
    total_residual_verification_applications = 0
    total_preconditioner_applications = 0
    total_v_cycles = 0
    total_current_operator_builds = 0
    for index, item in enumerate(outer):
        if not isinstance(item, Mapping):
            raise ValueError("outer correction evidence must be a mapping")
        correction = item.get("correction")
        v_cycles = item.get("v_cycle_work")
        if not isinstance(correction, Mapping) or not isinstance(v_cycles, list):
            raise ValueError("outer correction is missing correction or V-cycle evidence")
        pcg = correction.get("pcg")
        pcg_work = {} if pcg is None else pcg.get("work")
        if pcg is not None and (not isinstance(pcg, Mapping) or not isinstance(pcg_work, Mapping)):
            raise ValueError("outer correction has invalid PCG work evidence")
        correction_work = correction.get("work")
        if not isinstance(correction_work, Mapping):
            raise ValueError("outer correction has invalid work evidence")
        applications = 0 if pcg is None else int(pcg_work["preconditioner_applications"])
        operator_applications = 0 if pcg is None else int(pcg_work["operator_applications"])
        verification_applications = 0 if pcg is None else int(pcg_work["residual_verification_applications"])
        operator_builds = int(correction_work["operator_builds"])
        total_operator_applications += operator_applications
        total_residual_verification_applications += verification_applications
        total_preconditioner_applications += applications
        total_v_cycles += len(v_cycles)
        total_current_operator_builds += operator_builds
        per_outer.append(
            {
                "outer_index": index,
                "accepted": correction.get("accepted"),
                "fallback_used": not bool(correction.get("accepted")),
                "reason": correction.get("reason"),
                "alpha": correction.get("alpha"),
                "exact_work_completed": item.get("exact_work_completed"),
                "requested_pcg_iterations": None if pcg is None else pcg.get("requested_iterations"),
                "completed_pcg_iterations": None if pcg is None else pcg.get("completed_iterations"),
                "operator_applications": operator_applications,
                "residual_verification_applications": verification_applications,
                "preconditioner_applications": applications,
                "v_cycle_records": len(v_cycles),
                "start_position_sha256": item.get("start_position_sha256"),
                "end_position_sha256": item.get("end_position_sha256"),
                "true_linear_residual_norm": None if pcg is None else pcg.get("true_residual_norm"),
            }
        )

    return {
        "gate_passed": gate.get("passed"),
        "metrics": metrics,
        "versus_k4": _json_copy(versus_k4),
        "safeguards": {
            "exact_pins": gate.get("exact_pins"),
            "inversion_free": gate.get("inversion_free"),
            "fallback_used": gate.get("fallback_used"),
            "all_outer_work_completed": gate.get("all_outer_work_completed"),
            "final_max_pin_error_m": metrics["mg_corrected"]["max_pin_error_m"],
            "final_inverted_tet_fraction": metrics["mg_corrected"]["inverted_tet_fraction"],
            "final_determinant_min": metrics["mg_corrected"]["determinant_min"],
        },
        "work": {
            "configured_outer_corrections": 3,
            "configured_pcg_iterations_per_outer": 4,
            "accepted_outer_corrections": quality.get("accepted_outer_correction_count"),
            "recorded_outer_corrections": len(outer),
            "total_current_operator_builds": total_current_operator_builds,
            "total_operator_applications": total_operator_applications,
            "total_residual_verification_applications": total_residual_verification_applications,
            "total_preconditioner_applications": total_preconditioner_applications,
            "total_v_cycle_records": total_v_cycles,
            "per_outer": per_outer,
        },
        "hierarchy": {
            "hierarchy_sha256": hierarchy.get("hierarchy_sha256"),
            "static_model_sha256": hierarchy.get("static_model_sha256"),
            "storage_sha256": hierarchy.get("storage_sha256"),
            "total_bytes": hierarchy.get("total_bytes"),
            "level_shapes": _json_copy(hierarchy.get("level_shapes")),
            "mode_kind": hierarchy.get("mode_kind"),
            "smoother_safety": hierarchy.get("smoother_safety"),
            "preconditioner_identity": hierarchy.get("preconditioner_identity"),
            "static_reuse": hierarchy.get("static_reuse"),
        },
        "identities": {
            "scene_sha256": quality.get("scene_sha256"),
            "objective_instance_sha256": quality.get("objective_instance_sha256"),
            "quality_sha256": quality.get("quality_sha256"),
            "reference_provenance": (
                quality.get("reference", {}).get("provenance")
                if isinstance(quality.get("reference"), Mapping)
                else None
            ),
        },
    }


def _validate_quality_record(quality: Mapping[str, object], scene_sha256: str) -> None:
    _validate_self_hash(quality, "quality_sha256", "MG-VBD quality record")
    if quality.get("performance_evidence") is not False:
        raise ValueError("MG-VBD quality record must not claim performance evidence")
    if quality.get("scene_sha256") != scene_sha256:
        raise ValueError("MG-VBD quality record belongs to another scene")
    if quality.get("config") != _fixed_config_record():
        raise ValueError("MG-VBD quality record changed the frozen 3x4 tuned-rigid config")
    reference = quality.get("reference")
    vbd_k1 = quality.get("vbd_k1")
    vbd_k4 = quality.get("vbd_k4")
    hierarchy = quality.get("hierarchy")
    outer = quality.get("outer_corrections")
    gate = quality.get("gate")
    if not all(isinstance(value, Mapping) for value in (reference, vbd_k1, vbd_k4, hierarchy, gate)):
        raise ValueError("MG-VBD quality record is missing identity evidence")
    if not isinstance(outer, list) or not outer:
        raise ValueError("MG-VBD quality record must retain outer correction evidence")
    assert isinstance(reference, Mapping)
    assert isinstance(vbd_k1, Mapping)
    assert isinstance(vbd_k4, Mapping)
    assert isinstance(hierarchy, Mapping)
    assert isinstance(gate, Mapping)
    objective_instance_sha256 = _validate_sha256(
        quality.get("objective_instance_sha256"), "quality objective_instance_sha256"
    )
    if reference.get("provenance") != "fresh-dense-newton":
        raise ValueError("suite requires a fresh dense Newton reference for every scene")
    reference_source = reference.get("source_record")
    if not isinstance(reference_source, Mapping) or reference_source.get("accepted") is not True:
        raise ValueError("fresh dense Newton reference lacks accepted source evidence")
    _validate_sha256(reference.get("position_sha256"), "reference position_sha256")
    reference_source_sha256 = _validate_sha256(reference.get("source_record_sha256"), "reference source_record_sha256")
    if _canonical_sha256(reference_source) != reference_source_sha256:
        raise ValueError("fresh dense Newton source record does not match its SHA-256")
    reference_bindings = {
        "contract": "fresh-dense-newton-accepted-reference-v1",
        "method": "dense-cpu-newton-float64",
        "accepted": True,
        "scene_sha256": scene_sha256,
        "objective_instance_sha256": objective_instance_sha256,
        "position_sha256": reference.get("position_sha256"),
    }
    if any(reference_source.get(name) != expected for name, expected in reference_bindings.items()):
        raise ValueError("fresh dense Newton source record does not bind its accepted reference")
    if (
        reference.get("scene_sha256") != scene_sha256
        or reference.get("objective_instance_sha256") != objective_instance_sha256
    ):
        raise ValueError("fresh dense Newton evidence belongs to another scene or objective")
    if reference_source.get("failures") != []:
        raise ValueError("fresh dense Newton accepted reference contains failures")
    for evidence, role, iterations in ((vbd_k1, "vbd-k1", 1), (vbd_k4, "vbd-k4", 4)):
        _validate_self_hash(evidence, "evidence_sha256", f"{role} evidence")
        if (
            evidence.get("role") != role
            or evidence.get("iterations") != iterations
            or evidence.get("execution") != "fresh-scalar-cpu-run_vbd"
            or evidence.get("device") != "cpu"
            or evidence.get("requested_tile_solve") is not False
            or evidence.get("effective_tile_solve") is not False
        ):
            raise ValueError(f"suite {role} evidence is not the required fresh scalar-CPU run")
        if (
            evidence.get("scene_sha256") != scene_sha256
            or evidence.get("objective_instance_sha256") != objective_instance_sha256
        ):
            raise ValueError(f"suite {role} evidence belongs to another scene or objective")
    if vbd_k1.get("physical_state_sha256") != vbd_k4.get("physical_state_sha256") or vbd_k1.get(
        "iterate_zero_sha256"
    ) != vbd_k4.get("iterate_zero_sha256"):
        raise ValueError("fresh K1 and K4 evidence does not share one restart state")
    if hierarchy.get("mode_kind") != "rigid" or hierarchy.get("smoother_safety") != 0.9:
        raise ValueError("hierarchy evidence changed the tuned rigid policy")
    for name in ("hierarchy_sha256", "static_model_sha256", "storage_sha256"):
        _validate_sha256(hierarchy.get(name), f"hierarchy {name}")
    if (
        hierarchy.get("static_reuse") is not True
        or isinstance(hierarchy.get("total_bytes"), bool)
        or not isinstance(hierarchy.get("total_bytes"), int)
        or hierarchy["total_bytes"] < 1
    ):
        raise ValueError("hierarchy evidence lacks static reuse or storage")
    hierarchy_config = quality["config"]
    for name in (
        "mode_kind",
        "target_aggregate_size",
        "minimum_aggregate_size",
        "coarse_node_limit",
        "maximum_levels",
        "pre_smooth_steps",
        "post_smooth_steps",
        "smoother_safety",
    ):
        if hierarchy.get(name) != hierarchy_config[name]:
            raise ValueError(f"hierarchy evidence changed configured {name}")
    preconditioner_identity = hierarchy.get("preconditioner_identity")
    if preconditioner_identity != (
        f"{hierarchy.get('solver_contract')}:rest-a0-vcycle:{hierarchy.get('hierarchy_sha256')}"
    ):
        raise ValueError("hierarchy preconditioner identity does not bind its content")

    expected_start = vbd_k1.get("position_sha256")
    fallback_used = False
    total_v_cycles = 0
    accepted_count = 0
    all_outer_work_completed = len(outer) == 3
    for index, item in enumerate(outer):
        if not isinstance(item, Mapping) or item.get("outer_index") != index:
            raise ValueError("outer correction indices are not contiguous")
        if item.get("start_position_sha256") != expected_start:
            raise ValueError("outer correction chain does not start at the previous endpoint")
        correction = item.get("correction")
        v_cycles = item.get("v_cycle_work")
        start_metrics = item.get("start_metrics")
        end_metrics = item.get("metrics")
        if not isinstance(correction, Mapping) or not isinstance(v_cycles, list):
            raise ValueError("outer correction lacks correction or V-cycle evidence")
        if not isinstance(start_metrics, Mapping) or not isinstance(end_metrics, Mapping):
            raise ValueError("outer correction lacks independently evaluated endpoint metrics")
        if start_metrics.get("position_sha256") != expected_start or end_metrics.get("position_sha256") != item.get(
            "end_position_sha256"
        ):
            raise ValueError("outer endpoint metrics do not bind the correction chain")
        if correction.get("config") != _fixed_config_record()["correction"]:
            raise ValueError("outer correction changed the fixed PCG policy")
        if correction.get("preconditioner_identity") != preconditioner_identity:
            raise ValueError("outer correction changed the static hierarchy identity")
        if any(
            not isinstance(work, Mapping) or work.get("hierarchy_sha256") != hierarchy.get("hierarchy_sha256")
            for work in v_cycles
        ):
            raise ValueError("outer correction contains a foreign V-cycle record")
        for work in v_cycles:
            assert isinstance(work, Mapping)
            for name in ("rhs_sha256", "result_sha256", "content_sha256"):
                _validate_sha256(work.get(name), f"outer {index} V-cycle {name}")
        scalar_bindings = (
            (correction.get("initial_objective"), start_metrics.get("objective"), "initial objective"),
            (correction.get("initial_gradient_norm"), start_metrics.get("gradient_norm"), "initial gradient"),
            (
                correction.get("initial_minimum_determinant"),
                start_metrics.get("determinant_min"),
                "initial determinant",
            ),
            (correction.get("final_objective"), end_metrics.get("objective"), "final objective"),
            (correction.get("final_gradient_norm"), end_metrics.get("gradient_norm"), "final gradient"),
            (
                correction.get("final_minimum_determinant"),
                end_metrics.get("determinant_min"),
                "final determinant",
            ),
        )
        if any(not _same_float64_measurement(measured, independent) for measured, independent, _ in scalar_bindings):
            failed = next(
                name
                for measured, independent, name in scalar_bindings
                if not _same_float64_measurement(measured, independent)
            )
            raise ValueError(f"outer correction {failed} does not match independent metrics")
        pcg = correction.get("pcg")
        expected_vcycles = 0
        exact_pcg = False
        if pcg is not None:
            if not isinstance(pcg, Mapping) or not isinstance(pcg.get("work"), Mapping):
                raise ValueError("outer correction PCG record is invalid")
            expected_vcycles = int(pcg["work"]["preconditioner_applications"])
            exact_pcg = bool(
                pcg.get("requested_iterations") == 4
                and pcg.get("completed_iterations") == 4
                and pcg.get("consumed_exact_iteration_count") is True
            )
            if pcg.get("preconditioner_identity") != preconditioner_identity:
                raise ValueError("outer PCG changed the static hierarchy identity")
            trace = pcg.get("trace")
            if not isinstance(trace, list) or len(trace) != pcg.get("completed_iterations"):
                raise ValueError("outer PCG trace does not bind its completed iteration count")
        if len(v_cycles) != expected_vcycles:
            raise ValueError("V-cycle record count does not match preconditioner applications")
        accepted = correction.get("accepted") is True
        fallback_used |= not accepted
        accepted_count += int(accepted)
        exact_outer = bool(accepted and exact_pcg and item.get("exact_work_completed") is True)
        all_outer_work_completed &= exact_outer
        if not accepted and index != len(outer) - 1:
            raise ValueError("a rejected correction did not terminate the outer sequence")
        expected_start = item.get("end_position_sha256")
        total_v_cycles += len(v_cycles)
    final_metrics = quality.get("final_metrics")
    if not isinstance(final_metrics, Mapping) or final_metrics.get("position_sha256") != expected_start:
        raise ValueError("final metrics do not bind the last outer endpoint")
    if (
        quality.get("accepted_outer_correction_count") != accepted_count
        or quality.get("total_v_cycle_count") != total_v_cycles
    ):
        raise ValueError("quality aggregate work counts do not match retained outer evidence")

    k4_metrics = quality.get("k4_metrics")
    k1_metrics = quality.get("k1_metrics")
    reference_metrics = quality.get("reference_metrics")
    if not all(isinstance(value, Mapping) for value in (k4_metrics, k1_metrics, reference_metrics)):
        raise ValueError("quality record lacks K1, K4, or reference metrics")
    assert isinstance(k4_metrics, Mapping)
    assert isinstance(k1_metrics, Mapping)
    assert isinstance(reference_metrics, Mapping)
    metric_bindings = (
        (reference_metrics.get("position_sha256"), reference.get("position_sha256"), "reference"),
        (k1_metrics.get("position_sha256"), vbd_k1.get("position_sha256"), "K1"),
        (k4_metrics.get("position_sha256"), vbd_k4.get("position_sha256"), "K4"),
    )
    if any(measured != expected for measured, expected, _ in metric_bindings):
        failed = next(name for measured, expected, name in metric_bindings if measured != expected)
        raise ValueError(f"quality {failed} metrics do not bind their state evidence")
    for name, metric in (
        ("final_objective", reference_metrics.get("objective")),
        ("final_gradient_norm", reference_metrics.get("gradient_norm")),
        ("final_relative_residual", reference_metrics.get("relative_residual")),
    ):
        if not _same_float64_measurement(reference_source.get(name), metric):
            raise ValueError(f"fresh dense Newton source record changed {name}")
    comparison = gate.get("versus_k4")
    if not isinstance(comparison, Mapping):
        raise ValueError("quality gate lacks a K4 comparison")
    if (
        comparison.get("candidate_position_sha256") != final_metrics.get("position_sha256")
        or comparison.get("comparator_position_sha256") != k4_metrics.get("position_sha256")
        or comparison.get("comparator_role") != "fresh-vbd-k4"
    ):
        raise ValueError("quality K4 comparison does not bind candidate and comparator states")
    tiny = np.finfo(np.float64).tiny
    expected_ratios = {
        "objective_magnitude_ratio": abs(float(final_metrics["objective"]))
        / max(abs(float(k4_metrics["objective"])), tiny),
        "objective_delta": float(final_metrics["objective"]) - float(k4_metrics["objective"]),
        "residual_ratio": float(final_metrics["relative_residual"]) / max(float(k4_metrics["relative_residual"]), tiny),
        "free_rms_ratio": float(final_metrics["free_rms_error_m"]) / max(float(k4_metrics["free_rms_error_m"]), tiny),
        "mass_weighted_rms_ratio": float(final_metrics["mass_weighted_rms_error_m"])
        / max(float(k4_metrics["mass_weighted_rms_error_m"]), tiny),
    }
    if any(comparison.get(name) != value for name, value in expected_ratios.items()):
        raise ValueError("quality K4 ratios do not match raw common metrics")
    exact_pins = final_metrics.get("max_pin_error_m") == 0.0
    inversion_free = bool(
        final_metrics.get("inverted_tet_fraction") == 0.0 and float(final_metrics["determinant_min"]) > 0.0
    )
    objective_no_worse = bool(comparison.get("objective_delta") <= comparison.get("objective_roundoff_guard"))
    expected_passed = bool(
        exact_pins
        and inversion_free
        and all_outer_work_completed
        and not fallback_used
        and objective_no_worse
        and comparison.get("residual_ratio") <= 1.0
        and comparison.get("free_rms_ratio") <= 1.0
        and comparison.get("mass_weighted_rms_ratio") <= 1.0
    )
    expected_gate = {
        "exact_pins": exact_pins,
        "inversion_free": inversion_free,
        "all_outer_work_completed": all_outer_work_completed,
        "fallback_used": fallback_used,
        "passed": expected_passed,
    }
    if any(gate.get(name) != value for name, value in expected_gate.items()):
        raise ValueError("quality gate booleans do not match independently recomputed evidence")


def _diagnostic_record(result: MGVBDRunResult, scene_wall_seconds: float) -> dict[str, object]:
    timing = result.timing.deterministic_record()
    _validate_self_hash(timing, "timing_sha256", "MG-VBD diagnostic timing")
    payload: dict[str, object] = {
        "contract": _DIAGNOSTIC_CONTRACT,
        "performance_evidence": False,
        "measurement_provenance": "eager-scalar-cpu-diagnostic-only-not-performance-evidence",
        "scene_wall_seconds": scene_wall_seconds,
        "integration_timing": timing,
    }
    payload["diagnostic_sha256"] = _canonical_sha256(payload)
    return payload


def _failure_diagnostic_record(error: Exception, scene_wall_seconds: float) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract": _FAILURE_DIAGNOSTIC_CONTRACT,
        "performance_evidence": False,
        "measurement_provenance": "eager-scalar-cpu-diagnostic-only-not-performance-evidence",
        "scene_wall_seconds": scene_wall_seconds,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
    }
    payload["diagnostic_sha256"] = _canonical_sha256(payload)
    return payload


def _failure_summary(diagnostic: Mapping[str, object]) -> dict[str, object]:
    return {
        "gate_passed": False,
        "execution_failure": {
            "exception_type": diagnostic.get("exception_type"),
            "exception_message": diagnostic.get("exception_message"),
        },
        "metrics": None,
        "versus_k4": None,
        "safeguards": None,
        "work": None,
        "hierarchy": None,
        "identities": None,
    }


def _build_entry(
    key: str,
    scene: TetBenchmarkScene,
    counts: Mapping[str, int],
    result: MGVBDRunResult | None,
    diagnostic: Mapping[str, object],
) -> dict[str, object]:
    manifest = scene.manifest()
    if result is None:
        status = "execution-failed"
        quality = None
        summary = _failure_summary(diagnostic)
    else:
        status = "completed"
        quality = result.quality.deterministic_record()
        summary = _summary_from_quality(quality)
    payload: dict[str, object] = {
        "contract": _ENTRY_CONTRACT,
        "key": key,
        "status": status,
        "scene_manifest": manifest,
        "counts": dict(counts),
        "quality": quality,
        "diagnostic_timing": _json_copy(diagnostic),
        "summary": summary,
    }
    payload["entry_sha256"] = _canonical_sha256(payload)
    _validate_entry(payload)
    return payload


def _validate_diagnostic(record: Mapping[str, object], quality_sha256: str | None) -> None:
    _validate_self_hash(record, "diagnostic_sha256", "scene diagnostic record")
    if record.get("performance_evidence") is not False:
        raise ValueError("scene timing must be explicitly diagnostic-only")
    seconds = record.get("scene_wall_seconds")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds < 0.0
    ):
        raise ValueError("scene diagnostic wall time must be finite and non-negative")
    if quality_sha256 is None:
        if record.get("contract") != _FAILURE_DIAGNOSTIC_CONTRACT:
            raise ValueError("failed scene has the wrong diagnostic contract")
        if type(record.get("exception_type")) is not str or type(record.get("exception_message")) is not str:
            raise ValueError("failed scene lacks exact exception evidence")
    else:
        if record.get("contract") != _DIAGNOSTIC_CONTRACT:
            raise ValueError("completed scene has the wrong diagnostic contract")
        timing = record.get("integration_timing")
        if not isinstance(timing, Mapping):
            raise ValueError("completed scene lacks integration timing")
        _validate_self_hash(timing, "timing_sha256", "integration timing")
        if timing.get("quality_sha256") != quality_sha256 or timing.get("performance_evidence") is not False:
            raise ValueError("diagnostic timing does not bind quality or claims performance evidence")


def _validate_entry(entry: Mapping[str, object]) -> None:
    _validate_self_hash(entry, "entry_sha256", "scene entry")
    if entry.get("contract") != _ENTRY_CONTRACT or type(entry.get("key")) is not str:
        raise ValueError("scene entry has the wrong contract or key")
    scene = entry.get("scene_manifest")
    counts = entry.get("counts")
    diagnostic = entry.get("diagnostic_timing")
    summary = entry.get("summary")
    if not isinstance(scene, Mapping) or not isinstance(counts, Mapping) or not isinstance(diagnostic, Mapping):
        raise ValueError("scene entry lacks manifest, counts, or diagnostic evidence")
    if not isinstance(summary, Mapping):
        raise ValueError("scene entry summary must be a mapping")
    _verify_scene_manifest(scene)
    expected_counts = {
        "vertices": scene["n_vertices"],
        "tetrahedra": scene["n_tets"],
        "pinned_vertices": scene["n_pinned"],
        "free_vertices": scene["n_vertices"] - scene["n_pinned"],
        "free_dofs": 3 * (scene["n_vertices"] - scene["n_pinned"]),
    }
    if counts != expected_counts:
        raise ValueError("scene entry counts do not match its manifest")
    quality = entry.get("quality")
    if entry.get("status") == "completed":
        if not isinstance(quality, Mapping):
            raise ValueError("completed scene lacks quality evidence")
        _validate_quality_record(quality, str(scene["scene_sha256"]))
        _validate_diagnostic(diagnostic, str(quality["quality_sha256"]))
        if summary != _summary_from_quality(quality):
            raise ValueError("scene summary does not match its quality evidence")
    elif entry.get("status") == "execution-failed":
        if quality is not None:
            raise ValueError("failed scene must not contain partial quality evidence")
        _validate_diagnostic(diagnostic, None)
        if summary != _failure_summary(diagnostic):
            raise ValueError("failed-scene summary does not match its diagnostic evidence")
    else:
        raise ValueError("scene entry has an unknown status")


def _quality_suite_payload(payload: Mapping[str, object]) -> dict[str, object]:
    entries = payload.get("scenes")
    if not isinstance(entries, list):
        raise ValueError("suite scenes must be a list")
    return {
        "contract": payload.get("contract"),
        "configuration": _json_copy(payload.get("configuration")),
        "source_manifest_sha256": (
            payload.get("source_manifest", {}).get("source_manifest_sha256")
            if isinstance(payload.get("source_manifest"), Mapping)
            else None
        ),
        "scenes": [
            {
                "key": entry.get("key"),
                "status": entry.get("status"),
                "scene_sha256": (
                    entry.get("scene_manifest", {}).get("scene_sha256")
                    if isinstance(entry.get("scene_manifest"), Mapping)
                    else None
                ),
                "quality_sha256": (
                    entry.get("quality", {}).get("quality_sha256")
                    if isinstance(entry.get("quality"), Mapping)
                    else None
                ),
                "summary": _json_copy(entry.get("summary")),
            }
            for entry in entries
            if isinstance(entry, Mapping)
        ],
    }


def _suite_summary(
    entries: Sequence[Mapping[str, object]], scene_keys: Sequence[str], status: str
) -> dict[str, object]:
    passed = [entry["key"] for entry in entries if entry["summary"]["gate_passed"] is True]
    failed = [entry["key"] for entry in entries if entry["summary"]["gate_passed"] is not True]
    execution_failed = [entry["key"] for entry in entries if entry["status"] == "execution-failed"]
    complete = status == "complete"
    return {
        "requested_scene_count": len(scene_keys),
        "completed_scene_count": len(entries),
        "gate_pass_count": len(passed),
        "gate_fail_count": len(failed),
        "passed_scene_keys": passed,
        "failed_scene_keys": failed,
        "execution_failed_scene_keys": execution_failed,
        "all_requested_gates_passed": bool(complete and len(passed) == len(scene_keys)),
        "development_only": True,
        "confirmation_tuning_performed": False,
    }


def _new_suite_payload(scene_keys: Sequence[str], max_newton_free_dofs: int) -> dict[str, object]:
    created = datetime.datetime.now(datetime.UTC).isoformat()
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "contract": _SUITE_CONTRACT,
        "status": "partial",
        "performance_evidence": False,
        "classification": "fixed-policy-development-suite-not-confirmation-evidence",
        "created_at_utc": created,
        "completed_at_utc": None,
        "runner": "research.principal_stretch.bench_mg_vbd",
        "configuration": _configuration_record(scene_keys, max_newton_free_dofs),
        "source_manifest": _source_manifest(),
        "scenes": [],
    }
    return _seal_suite(payload)


def _seal_suite(payload: Mapping[str, object]) -> dict[str, object]:
    sealed = dict(_json_copy(payload))
    sealed.pop("quality_suite_sha256", None)
    sealed.pop("suite_sha256", None)
    configuration = sealed.get("configuration")
    scene_keys = configuration.get("scene_keys") if isinstance(configuration, Mapping) else None
    entries = sealed.get("scenes")
    if not isinstance(scene_keys, list) or not isinstance(entries, list):
        raise ValueError("cannot seal suite without configuration and scene entries")
    sealed["summary"] = _suite_summary(entries, scene_keys, str(sealed.get("status")))
    sealed["quality_suite_sha256"] = _canonical_sha256(_quality_suite_payload(sealed))
    sealed["suite_sha256"] = _canonical_sha256(sealed)
    return sealed


def verify_suite_payload(payload: Mapping[str, object], *, verify_current_sources: bool = False) -> dict[str, object]:
    """Verify all suite hashes, identities, arithmetic, and fixed-work policy."""
    record = _json_copy(payload)
    if not isinstance(record, dict):
        raise ValueError("suite payload must be a JSON object")
    stored_suite_sha256 = _validate_sha256(record.get("suite_sha256"), "suite_sha256")
    unsealed = dict(record)
    unsealed.pop("suite_sha256")
    if _canonical_sha256(unsealed) != stored_suite_sha256:
        raise ValueError("suite payload does not match suite_sha256")
    if record.get("schema_version") != _SCHEMA_VERSION or record.get("contract") != _SUITE_CONTRACT:
        raise ValueError("suite schema or contract is unsupported")
    if record.get("performance_evidence") is not False:
        raise ValueError("CPU development suite must not claim performance evidence")
    if record.get("status") not in ("partial", "complete"):
        raise ValueError("suite status must be partial or complete")
    configuration = record.get("configuration")
    source_manifest = record.get("source_manifest")
    entries = record.get("scenes")
    if (
        not isinstance(configuration, Mapping)
        or not isinstance(source_manifest, Mapping)
        or not isinstance(entries, list)
    ):
        raise ValueError("suite lacks configuration, sources, or scenes")
    scene_keys = configuration.get("scene_keys")
    if not isinstance(scene_keys, list):
        raise ValueError("suite scene_keys must be a list")
    resolved = _resolve_scene_keys(scene_keys)
    maximum = configuration.get("max_newton_free_dofs")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1 or maximum > _MAX_NEWTON_FREE_DOFS:
        raise ValueError("suite has an invalid dense Newton free-DOF ceiling")
    if configuration != _configuration_record(resolved, maximum):
        raise ValueError("suite configuration changed the frozen development policy")
    _verify_source_manifest(source_manifest, verify_current_sources=verify_current_sources)
    if len(entries) > len(resolved):
        raise ValueError("suite contains more entries than requested scenes")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or entry.get("key") != resolved[index]:
            raise ValueError("suite entries are not an ordered prefix of requested scenes")
        _validate_entry(entry)
        counts = entry["counts"]
        if counts["free_dofs"] > maximum:
            raise ValueError("suite entry exceeds the dense Newton free-DOF ceiling")
    if record["status"] == "complete" and len(entries) != len(resolved):
        raise ValueError("complete suite does not contain every requested scene")
    if record["status"] == "partial" and len(entries) == len(resolved):
        raise ValueError("partial suite already contains every requested scene")
    expected_summary = _suite_summary(entries, resolved, str(record["status"]))
    if record.get("summary") != expected_summary:
        raise ValueError("suite summary does not match scene evidence")
    expected_quality_sha256 = _canonical_sha256(_quality_suite_payload(record))
    if record.get("quality_suite_sha256") != expected_quality_sha256:
        raise ValueError("suite quality identity does not match timing-free evidence")
    return record


def verify_suite_file(path: pathlib.Path, *, verify_current_sources: bool = False) -> dict[str, object]:
    """Load and verify one self-hashed suite JSON file."""
    path = pathlib.Path(path).resolve()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read suite JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("suite JSON root must be an object")
    return verify_suite_payload(payload, verify_current_sources=verify_current_sources)


def _atomic_write_json(path: pathlib.Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = pathlib.Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@dataclasses.dataclass(frozen=True)
class MGVBDSuiteRunConfig:
    """Output and immutable scene selection for one CPU development suite."""

    output_path: pathlib.Path
    report_path: pathlib.Path | None = None
    scene_keys: tuple[str, ...] = _DEFAULT_SCENE_KEYS
    max_newton_free_dofs: int = _MAX_NEWTON_FREE_DOFS
    resume: bool = False

    def validate(self) -> None:
        _resolve_scene_keys(self.scene_keys)
        if (
            isinstance(self.max_newton_free_dofs, bool)
            or not isinstance(self.max_newton_free_dofs, int)
            or self.max_newton_free_dofs < 1
            or self.max_newton_free_dofs > _MAX_NEWTON_FREE_DOFS
        ):
            raise ValueError(f"max_newton_free_dofs must lie in [1, {_MAX_NEWTON_FREE_DOFS}]")
        if type(self.resume) is not bool:
            raise ValueError("resume must be an exact bool")
        output = pathlib.Path(self.output_path)
        if output.suffix.lower() != ".json":
            raise ValueError("output_path must end in .json")
        if self.report_path is not None:
            report = pathlib.Path(self.report_path)
            if report.suffix.lower() != ".md":
                raise ValueError("report_path must end in .md")
            if report.resolve() == output.resolve():
                raise ValueError("report_path must differ from output_path")


def _preflight_scenes(scene_keys: Sequence[str], maximum: int) -> list[tuple[str, TetBenchmarkScene, dict[str, int]]]:
    scenes = []
    oversized = []
    for key in scene_keys:
        scene = _SCENE_FACTORIES[key]()
        if type(scene) is not TetBenchmarkScene:
            raise TypeError(f"{key} factory must return an exact TetBenchmarkScene")
        counts = _scene_counts(scene)
        _verify_scene_manifest(scene.manifest())
        scenes.append((key, scene, counts))
        if counts["free_dofs"] > maximum:
            oversized.append((key, counts["free_dofs"]))
    if oversized:
        details = ", ".join(f"{key}={dofs}" for key, dofs in oversized)
        raise ValueError(f"dense Newton free-DOF ceiling {maximum} exceeded: {details}")
    return scenes


def _markdown_report(payload: Mapping[str, object], json_path: pathlib.Path) -> str:
    summary = payload["summary"]
    lines = [
        "# Multiplicative MG-VBD CPU development suite",
        "",
        f"- Status: `{payload['status']}`",
        f"- Suite SHA-256: `{payload['suite_sha256']}`",
        f"- Timing-free quality suite SHA-256: `{payload['quality_suite_sha256']}`",
        f"- JSON file SHA-256: `{_file_sha256(json_path)}`",
        "- Fixed policy: 3 nonlinear outer corrections x 4 PCG iterations, rigid tuned static-A0 MG",
        "- Classification: development-only; no confirmation tuning; CPU timings are diagnostic-only",
        f"- Gate result: {summary['gate_pass_count']}/{summary['requested_scene_count']} scenes passed",
        "",
        "| Scene | Gate | K4 residual ratio | K4 free-RMS ratio | K4 objective ratio | V-cycles | Fallback | det(F) min | Pin error (m) | Diagnostic wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in payload["scenes"]:
        scene_summary = entry["summary"]
        seconds = entry["diagnostic_timing"]["scene_wall_seconds"]
        if entry["status"] == "execution-failed":
            lines.append(
                f"| {entry['key']} | FAIL (execution) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | {seconds:.6g} |"
            )
            continue
        comparison = scene_summary["versus_k4"]
        safeguards = scene_summary["safeguards"]
        work = scene_summary["work"]
        lines.append(
            "| "
            f"{entry['key']} | {'PASS' if scene_summary['gate_passed'] else 'FAIL'} | "
            f"{comparison['residual_ratio']:.9g} | {comparison['free_rms_ratio']:.9g} | "
            f"{comparison['objective_magnitude_ratio']:.9g} | {work['total_v_cycle_records']} | "
            f"{safeguards['fallback_used']} | {safeguards['final_determinant_min']:.9g} | "
            f"{safeguards['final_max_pin_error_m']:.9g} | {seconds:.6g} |"
        )
    if summary["execution_failed_scene_keys"]:
        lines.extend(["", "## Execution failures", ""])
        for entry in payload["scenes"]:
            if entry["status"] == "execution-failed":
                failure = entry["summary"]["execution_failure"]
                lines.append(f"- `{entry['key']}`: `{failure['exception_type']}` — {failure['exception_message']}")
    lines.extend(
        [
            "",
            "The ratios are MG-corrected / fresh K4 on the independently evaluated common objective. "
            "A suite failure is retained as evidence; it is not tuned away or retried with another numerical policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_report_no_overwrite(path: pathlib.Path, payload: Mapping[str, object], json_path: pathlib.Path) -> None:
    expected = _markdown_report(payload, json_path)
    if path.exists():
        if path.read_text() != expected:
            raise FileExistsError(f"refusing to overwrite existing report {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected)


def run_scene_suite(config: MGVBDSuiteRunConfig) -> pathlib.Path:
    """Run or resume the fixed CPU development suite and return its JSON path."""
    config.validate()
    scene_keys = _resolve_scene_keys(config.scene_keys)
    scenes = _preflight_scenes(scene_keys, config.max_newton_free_dofs)
    output_path = pathlib.Path(config.output_path).resolve()
    report_path = None if config.report_path is None else pathlib.Path(config.report_path).resolve()

    if output_path.exists():
        if not config.resume:
            raise FileExistsError(
                f"refusing to overwrite existing suite {output_path}; pass resume=True to verify/resume"
            )
        payload = verify_suite_file(output_path, verify_current_sources=True)
        if payload["configuration"] != _configuration_record(scene_keys, config.max_newton_free_dofs):
            raise ValueError("existing suite configuration does not match this invocation")
    else:
        if config.resume:
            raise FileNotFoundError(f"cannot resume missing suite {output_path}")
        if report_path is not None and report_path.exists():
            raise FileExistsError(f"refusing to overwrite existing report {report_path}")
        payload = _new_suite_payload(scene_keys, config.max_newton_free_dofs)
        _atomic_write_json(output_path, payload)

    completed = len(payload["scenes"])
    for index, (key, scene, counts) in enumerate(scenes[:completed]):
        entry = payload["scenes"][index]
        if entry["key"] != key or entry["scene_manifest"] != scene.manifest() or entry["counts"] != counts:
            raise ValueError(f"resume scene {key} does not match its authenticated checkpoint")

    if payload["status"] == "complete":
        if report_path is not None:
            _write_report_no_overwrite(report_path, payload, output_path)
        return output_path

    for key, scene, counts in scenes[completed:]:
        print(
            f"[{key}] {counts['vertices']} vertices, {counts['tetrahedra']} tets, {counts['free_dofs']} free DOFs",
            file=sys.stderr,
            flush=True,
        )
        start = time.perf_counter()
        try:
            result = run_multiplicative_mg_vbd(
                scene,
                config=_fixed_config(),
                vbd_warmup=False,
                vbd_repeats=1,
                newton_warmup=False,
                newton_repeats=1,
            )
        except Exception as error:  # retain an honest development-suite failure and continue
            elapsed = time.perf_counter() - start
            diagnostic = _failure_diagnostic_record(error, elapsed)
            entry = _build_entry(key, scene, counts, None, diagnostic)
            print(f"[{key}] FAIL ({type(error).__name__}: {error})", file=sys.stderr, flush=True)
        else:
            elapsed = time.perf_counter() - start
            diagnostic = _diagnostic_record(result, elapsed)
            entry = _build_entry(key, scene, counts, result, diagnostic)
            gate = result.quality.gate
            print(
                f"[{key}] {'PASS' if gate.passed else 'FAIL'} "
                f"res/K4={gate.versus_k4.residual_ratio:.6g} "
                f"rms/K4={gate.versus_k4.free_rms_ratio:.6g} "
                f"work={result.quality.total_v_cycle_count}",
                file=sys.stderr,
                flush=True,
            )
        payload["scenes"].append(entry)
        if len(payload["scenes"]) == len(scenes):
            payload["status"] = "complete"
            payload["completed_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        payload = _seal_suite(payload)
        _atomic_write_json(output_path, payload)
        verify_suite_file(output_path, verify_current_sources=True)

    if payload["status"] != "complete":
        raise RuntimeError("suite stopped before every preflighted scene produced an evidence row")
    if report_path is not None:
        _write_report_no_overwrite(report_path, payload, output_path)
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--scenes", nargs="+", default=_DEFAULT_SCENE_KEYS)
    parser.add_argument("--max-newton-free-dofs", type=int, default=_MAX_NEWTON_FREE_DOFS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", type=pathlib.Path, metavar="JSON")
    parser.add_argument("--verify-current-sources", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.verify is not None:
            payload = verify_suite_file(args.verify, verify_current_sources=args.verify_current_sources)
            print(
                f"verified {args.verify}: status={payload['status']} "
                f"gates={payload['summary']['gate_pass_count']}/{payload['summary']['requested_scene_count']} "
                f"sha256={payload['suite_sha256']}"
            )
            return 0
        if args.output is None:
            raise ValueError("--output is required unless --verify is used")
        path = run_scene_suite(
            MGVBDSuiteRunConfig(
                output_path=args.output,
                report_path=args.report,
                scene_keys=tuple(args.scenes),
                max_newton_free_dofs=args.max_newton_free_dofs,
                resume=args.resume,
            )
        )
        payload = verify_suite_file(path)
    except (FileExistsError, FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {path}: gates={payload['summary']['gate_pass_count']}/{payload['summary']['requested_scene_count']} "
        f"sha256={payload['suite_sha256']}"
    )
    return 0 if payload["summary"]["all_requested_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
