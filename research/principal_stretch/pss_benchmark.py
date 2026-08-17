# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic principal-stretch candidates for the common objective.

This adapter intentionally does not describe the learned pipeline as a
common-objective iterative solver.  One graph-transformer pass predicts an
SPD target right-stretch field, then ``K`` local-global decoder sweeps minimize
a separate stretch-compatibility surrogate.  The returned positions are left
unmodified and can be scored by :func:`solver_benchmark.evaluate_common_state`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import numbers
import pathlib
import statistics
import time
from collections.abc import Mapping

import numpy as np
import torch

from . import torch_solver
from .graph_transformer import GraphTransformerConfig
from .predictor import build_stretch_predictor, checkpoint_predictor_config, load_stretch_predictor_state
from .solver_benchmark import (
    CommonStateMetrics,
    TetBenchmarkScene,
    _array_digest,
    _canonical_digest,
    _git_dirty_digest,
    _git_revision,
    _readonly_array,
    build_common_problem,
    common_objective_manifest,
    evaluate_common_state,
)

_METHOD = "principal-stretch-graph-transformer-diagnostic"
_WORK_SEMANTICS = "one predictor pass plus K surrogate local-global decoder sweeps"
_HISTORY_CONTRACT = "x_previous = x_current - checkpoint_dt * velocity; pinned velocity must be zero"
# CUDA hierarchy pooling and decoder assembly use atomic ``index_add_``
# reductions.  A nonzero-head v2 checkpoint measured target discrepancies up
# to 1.0132789611816406e-6 on the audited 12,800-tet refinement scene and
# decoded-position discrepancies up to 2.488853612092612e-8 m across
# K=1,4,16,32 on the 1,600-tet extension scene (NVIDIA L40).  Bind repeat
# acceptance to explicit float32-error scales with conservative margins rather
# than requiring bit equality that only holds for hierarchy-free checkpoints.
_CUDA_TARGET_REPEAT_EPS_MULTIPLIER = 64.0
_CUDA_POSITION_REPEAT_EPS_MULTIPLIER = 16.0
_FLOAT32_EPSILON = float(np.finfo(np.float32).eps)

_GRAPH_ARG_CONFIG_KEYS = {
    "gt_hidden": "hidden_dim",
    "gt_heads": "num_heads",
    "gt_levels": "n_levels",
    "gt_cluster_size": "cluster_size",
    "gt_dropout": "dropout",
    "gt_max_delta": "max_hencky_update",
    "gt_architecture_version": "architecture_version",
    "dt": "dt",
}
_INTEGER_GRAPH_CONFIG_KEYS = {
    "hidden_dim",
    "num_heads",
    "n_levels",
    "cluster_size",
    "architecture_version",
}


def _canonical_json(value: object, name: str) -> str:
    """Return stable JSON or reject checkpoint metadata that cannot be bound."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint {name} must contain finite JSON-compatible values") from exc


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _target_repeat_scale(target: torch.Tensor) -> float:
    """Return the dimensionless scale used by the CUDA target repeat gate."""
    return max(1.0, float(target.detach().abs().max()))


def _position_repeat_scale_m(scene: TetBenchmarkScene) -> float:
    """Return a translation-invariant mesh length for the position repeat gate."""
    extent = np.ptp(np.asarray(scene.rest_q), axis=0)
    scale = float(np.linalg.norm(extent))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scene rest bounding-box diagonal must be finite and positive")
    return scale


def _cuda_target_repeat_tolerance(target_scale: float) -> float:
    return _CUDA_TARGET_REPEAT_EPS_MULTIPLIER * _FLOAT32_EPSILON * target_scale


def _cuda_position_repeat_tolerance_m(position_scale_m: float) -> float:
    return _CUDA_POSITION_REPEAT_EPS_MULTIPLIER * _FLOAT32_EPSILON * position_scale_m


def _validate_decoder_connectivity(scene: TetBenchmarkScene) -> None:
    """Require every free vertex component to be anchored by a pin."""
    parent = np.arange(scene.n_vertices, dtype=np.int64)
    incident = np.zeros(scene.n_vertices, dtype=bool)

    def find(vertex: int) -> int:
        root = vertex
        while parent[root] != root:
            root = int(parent[root])
        while parent[vertex] != vertex:
            next_vertex = int(parent[vertex])
            parent[vertex] = root
            vertex = next_vertex
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for tet in scene.tet_indices:
        vertices = [int(vertex) for vertex in tet]
        incident[vertices] = True
        for vertex in vertices[1:]:
            union(vertices[0], vertex)

    free = scene.free_indices
    isolated = free[~incident[free]]
    if isolated.size:
        raise ValueError(f"local-global decoder has isolated free vertices: {isolated.tolist()}")

    pinned_roots = {find(int(vertex)) for vertex in scene.pinned_indices if incident[int(vertex)]}
    floating_roots = sorted({find(int(vertex)) for vertex in free} - pinned_roots)
    if floating_roots:
        raise ValueError("every connected tet component with free vertices must contain a pinned vertex")


def _validate_checkpoint_graph_args(checkpoint_args: dict[str, object], graph_config: dict[str, object]) -> None:
    """Reject stale CLI metadata that changes effective inference semantics."""
    for argument_name, config_name in _GRAPH_ARG_CONFIG_KEYS.items():
        if argument_name not in checkpoint_args:
            continue
        actual = checkpoint_args[argument_name]
        expected = graph_config[config_name]
        if config_name in _INTEGER_GRAPH_CONFIG_KEYS:
            matches = (
                isinstance(actual, numbers.Integral) and not isinstance(actual, bool) and int(actual) == int(expected)
            )
        else:
            matches = (
                isinstance(actual, numbers.Real)
                and not isinstance(actual, bool)
                and math.isfinite(float(actual))
                and float(actual) == float(expected)
            )
        if not matches:
            raise ValueError(
                f"checkpoint args {argument_name}={actual!r} disagrees with predictor_config {config_name}={expected!r}"
            )


def _checkpoint_training_hierarchy_levels(checkpoint: dict[str, object], graph_config: dict[str, object]) -> int:
    """Return the deepest hierarchy level known to have run during training."""
    architecture_version = int(graph_config["architecture_version"])
    configured_levels = int(graph_config["n_levels"])
    runtime = checkpoint.get("runtime", {})
    if runtime is None:
        runtime = {}
    if not isinstance(runtime, Mapping):
        raise ValueError("checkpoint runtime must be a mapping when present")
    recorded_depth = checkpoint.get("training_realized_hierarchy_levels")
    runtime_depth = runtime.get("realized_hierarchy_levels")
    if recorded_depth is not None and runtime_depth is not None and recorded_depth != runtime_depth:
        raise ValueError("checkpoint hierarchy-depth metadata disagrees between training contract and runtime")
    if recorded_depth is None:
        recorded_depth = runtime_depth

    if architecture_version >= 2:
        if recorded_depth is None and configured_levels == 0:
            return 0
        if recorded_depth is None:
            raise ValueError(
                "architecture-v2 checkpoint must record training_realized_hierarchy_levels "
                "when configured n_levels is positive"
            )
        if (
            not isinstance(recorded_depth, numbers.Integral)
            or isinstance(recorded_depth, bool)
            or not 0 <= int(recorded_depth) <= configured_levels
        ):
            raise ValueError(
                "checkpoint training_realized_hierarchy_levels must be an integer between zero and configured n_levels"
            )
        return int(recorded_depth)

    state_dict = checkpoint["state_dict"]
    encoder_levels: set[int] = set()
    for name in state_dict:
        parts = str(name).split(".")
        if len(parts) >= 3 and parts[0] == "encoders" and parts[1].isdigit():
            encoder_levels.add(int(parts[1]))
    if not encoder_levels or encoder_levels != set(range(max(encoder_levels) + 1)):
        raise ValueError("legacy graph-transformer state_dict does not encode a contiguous hierarchy depth")
    inferred_depth = max(encoder_levels)
    if recorded_depth is not None:
        if (
            not isinstance(recorded_depth, numbers.Integral)
            or isinstance(recorded_depth, bool)
            or int(recorded_depth) != inferred_depth
        ):
            raise ValueError("checkpoint recorded hierarchy depth disagrees with its architecture-v1 state_dict")
    return inferred_depth


def _load_graph_checkpoint(
    checkpoint_path: pathlib.Path,
) -> tuple[dict[str, object], dict[str, object], str, str, str, int, float]:
    """Load and validate the exact checkpoint bytes used by the run."""
    start = time.perf_counter()
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if "state_dict" not in checkpoint or not isinstance(checkpoint["state_dict"], Mapping):
        raise ValueError("checkpoint must contain a state_dict mapping")

    raw_predictor_config = checkpoint.get("predictor_config")
    if isinstance(raw_predictor_config, Mapping):
        raw_graph_config = raw_predictor_config.get("graph_transformer")
        if isinstance(raw_graph_config, Mapping):
            raw_architecture_version = raw_graph_config.get("architecture_version")
            if (
                isinstance(raw_architecture_version, numbers.Integral)
                and not isinstance(raw_architecture_version, bool)
                and int(raw_architecture_version) >= 3
                and "max_rotation_update" not in raw_graph_config
            ):
                raise ValueError("architecture-v3 checkpoint is missing max_rotation_update metadata")

    predictor_config = checkpoint_predictor_config(checkpoint)
    if predictor_config.get("kind") != "graph-transformer":
        raise ValueError("PSS common adapter requires a graph-transformer checkpoint")
    if predictor_config.get("residual") is not True:
        raise ValueError("graph-transformer checkpoint must use the log-stretch residual parameterization")
    graph_config = predictor_config.get("graph_transformer")
    if not isinstance(graph_config, dict):
        raise ValueError("graph-transformer checkpoint is missing architecture metadata")
    # Versioned fields introduced after the preserved v1/v2 checkpoints have
    # defaults inside GraphTransformerConfig. The original semantic fields
    # (except the state-dict-inferred architecture version) must be explicit.
    required_graph_keys = set(_GRAPH_ARG_CONFIG_KEYS.values()) - {"architecture_version"}
    missing_graph_keys = sorted(required_graph_keys - graph_config.keys())
    if missing_graph_keys:
        raise ValueError(f"graph-transformer checkpoint is missing metadata: {missing_graph_keys}")
    if int(graph_config["architecture_version"]) >= 3 and "max_rotation_update" not in graph_config:
        raise ValueError("architecture-v3 checkpoint is missing max_rotation_update metadata")
    try:
        graph_config = dataclasses.asdict(GraphTransformerConfig(**graph_config))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid graph-transformer predictor configuration: {exc}") from exc
    if int(graph_config["architecture_version"]) > 2:
        raise ValueError("PSS target-stretch diagnostic does not support full-deformation-gradient checkpoints")
    predictor_config["graph_transformer"] = graph_config

    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        raise ValueError("checkpoint args must be a dictionary when present")
    if checkpoint_args.get("predictor", "graph-transformer") != "graph-transformer":
        raise ValueError("checkpoint args disagree with the graph-transformer predictor configuration")
    blocks = checkpoint_args.get("blocks", 1)
    if not isinstance(blocks, numbers.Integral) or isinstance(blocks, bool) or int(blocks) != 1:
        raise ValueError("graph-transformer diagnostic requires checkpoint blocks=1")
    if checkpoint_args.get("warm", "prev") not in ("inertial", "prev"):
        raise ValueError("checkpoint warm-start metadata must be 'inertial' or 'prev'")
    if "residual" in checkpoint_args and checkpoint_args["residual"] is not True:
        raise ValueError("checkpoint args disagree with the required residual parameterization")

    checkpoint_dt = float(graph_config["dt"])
    if not math.isfinite(checkpoint_dt) or checkpoint_dt <= 0.0:
        raise ValueError("checkpoint graph-transformer dt must be finite and positive")
    _validate_checkpoint_graph_args(checkpoint_args, graph_config)
    training_hierarchy_levels = _checkpoint_training_hierarchy_levels(checkpoint, graph_config)

    predictor_config_json = _canonical_json(predictor_config, "predictor_config")
    checkpoint_args_json = _canonical_json(checkpoint_args, "args")
    return (
        checkpoint,
        predictor_config,
        predictor_config_json,
        checkpoint_args_json,
        checkpoint_sha256,
        training_hierarchy_levels,
        time.perf_counter() - start,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class PSSRunResult:
    """One raw learned-stretch candidate with provenance and synchronized timings."""

    positions: np.ndarray
    target_stretch: np.ndarray
    previous_positions: np.ndarray
    decoder_iterations: int
    checkpoint_name: str
    checkpoint_sha256: str
    predictor_config_json: str
    checkpoint_args_json: str
    parameter_count: int
    training_hierarchy_levels: int
    realized_hierarchy_levels: int
    device: str
    setup_seconds: float
    checkpoint_load_seconds: float
    common_problem_seconds: float
    decoder_setup_seconds: float
    predictor_setup_seconds: float
    input_setup_seconds: float
    warmup_seconds: float
    repeat_seconds: tuple[float, ...]
    predictor_seconds: tuple[float, ...]
    decoder_seconds: tuple[float, ...]
    transfer_seconds: tuple[float, ...]
    target_repeat_scale: float
    target_repeat_tolerance: float
    position_repeat_tolerance_m: float
    position_repeat_scale_m: float
    position_repeat_max_abs_discrepancy_m: float
    target_repeat_max_abs_discrepancy: float
    scene_sha256: str
    objective_instance_sha256: str
    physical_state_sha256: str
    previous_state_sha256: str
    iterate_zero_sha256: str
    target_stretch_sha256: str
    result_state_sha256: str
    newton_revision: str | None
    dirty_tree_sha256: str | None
    run_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _readonly_array(self.positions, np.float64, "positions"))
        object.__setattr__(
            self,
            "target_stretch",
            _readonly_array(self.target_stretch, np.float32, "target_stretch"),
        )
        object.__setattr__(
            self,
            "previous_positions",
            _readonly_array(self.previous_positions, np.float64, "previous_positions"),
        )

    @property
    def median_solve_seconds(self) -> float:
        """Median end-to-end predictor plus decoder time [s]."""
        return statistics.median(self.repeat_seconds)


def _pss_run_digest(result: PSSRunResult) -> str:
    payload = {
        "method": _METHOD,
        "work_semantics": _WORK_SEMANTICS,
        "history_contract": _HISTORY_CONTRACT,
        "positions_sha256": _array_digest(result.positions),
        "target_stretch_actual_sha256": _array_digest(result.target_stretch),
        "previous_positions_actual_sha256": _array_digest(result.previous_positions),
        "decoder_iterations": result.decoder_iterations,
        "checkpoint_name": result.checkpoint_name,
        "checkpoint_sha256": result.checkpoint_sha256,
        "predictor_config_json": result.predictor_config_json,
        "checkpoint_args_json": result.checkpoint_args_json,
        "parameter_count": result.parameter_count,
        "training_hierarchy_levels": result.training_hierarchy_levels,
        "realized_hierarchy_levels": result.realized_hierarchy_levels,
        "device": result.device,
        "setup_seconds": result.setup_seconds,
        "checkpoint_load_seconds": result.checkpoint_load_seconds,
        "common_problem_seconds": result.common_problem_seconds,
        "decoder_setup_seconds": result.decoder_setup_seconds,
        "predictor_setup_seconds": result.predictor_setup_seconds,
        "input_setup_seconds": result.input_setup_seconds,
        "warmup_seconds": result.warmup_seconds,
        "repeat_seconds": list(result.repeat_seconds),
        "predictor_seconds": list(result.predictor_seconds),
        "decoder_seconds": list(result.decoder_seconds),
        "transfer_seconds": list(result.transfer_seconds),
        "target_repeat_scale": result.target_repeat_scale,
        "target_repeat_tolerance": result.target_repeat_tolerance,
        "position_repeat_tolerance_m": result.position_repeat_tolerance_m,
        "position_repeat_scale_m": result.position_repeat_scale_m,
        "position_repeat_max_abs_discrepancy_m": result.position_repeat_max_abs_discrepancy_m,
        "target_repeat_max_abs_discrepancy": result.target_repeat_max_abs_discrepancy,
        "scene_sha256": result.scene_sha256,
        "objective_instance_sha256": result.objective_instance_sha256,
        "physical_state_sha256": result.physical_state_sha256,
        "previous_state_sha256": result.previous_state_sha256,
        "iterate_zero_sha256": result.iterate_zero_sha256,
        "target_stretch_sha256": result.target_stretch_sha256,
        "result_state_sha256": result.result_state_sha256,
        "newton_revision": result.newton_revision,
        "dirty_tree_sha256": result.dirty_tree_sha256,
    }
    return _canonical_digest(payload)


def _validate_candidate(
    target: torch.Tensor, positions: torch.Tensor, pinned: torch.Tensor, pin_targets: torch.Tensor
) -> None:
    if not torch.isfinite(target).all():
        raise RuntimeError("PSS predictor returned a non-finite target stretch")
    if not torch.isfinite(positions).all():
        raise RuntimeError("PSS decoder returned non-finite positions")
    symmetry_error = float((target - target.transpose(-1, -2)).abs().max())
    if symmetry_error > 2.0e-5:
        raise RuntimeError(f"PSS target stretch is not symmetric (max error {symmetry_error:.3e})")
    if float(torch.linalg.eigvalsh(target).min()) <= 0.0:
        raise RuntimeError("PSS target stretch is not positive definite")
    if pinned.numel() and not torch.equal(positions[pinned], pin_targets):
        raise RuntimeError("PSS decoder changed a Dirichlet target")


def run_pss(
    scene: TetBenchmarkScene,
    checkpoint_path: pathlib.Path,
    decoder_iterations: int,
    *,
    device: str = "cuda:0",
    warmup: bool = True,
    repeats: int = 5,
) -> PSSRunResult:
    """Produce an unmodified PSS candidate for independent common scoring.

    ``decoder_iterations`` counts local-global sweeps of the stretch
    compatibility surrogate. It is not a common-objective convergence count.
    Every repeat reconstructs the target and restarts the decoder from the
    exact common inertial iterate.
    """
    if not isinstance(decoder_iterations, int) or isinstance(decoder_iterations, bool) or decoder_iterations < 1:
        raise ValueError("decoder_iterations must be a positive integer")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(warmup, bool):
        raise ValueError("warmup must be a boolean")
    checkpoint_path = pathlib.Path(checkpoint_path)
    torch_device = torch.device(device)
    if torch_device.type not in ("cpu", "cuda"):
        raise ValueError("PSS diagnostic device must be CPU or CUDA")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    setup_start = time.perf_counter()
    (
        checkpoint,
        predictor_config,
        predictor_config_json,
        checkpoint_args_json,
        checkpoint_sha256,
        training_hierarchy_levels,
        checkpoint_load_seconds,
    ) = _load_graph_checkpoint(checkpoint_path)
    graph_config = predictor_config["graph_transformer"]
    checkpoint_dt = float(graph_config["dt"])
    if np.float32(checkpoint_dt) != np.float32(scene.dt):
        raise ValueError(
            f"checkpoint dt {checkpoint_dt:.9g} does not match scene dt {scene.dt:.9g} at SolverVBD float32 precision"
        )

    if not np.array_equal(scene.x_current[scene.pinned_indices], scene.pin_targets):
        raise ValueError("scene x_current must contain the exact Dirichlet pin targets")
    if scene.pinned_indices.size and not np.array_equal(
        scene.velocity[scene.pinned_indices], np.zeros_like(scene.pin_targets)
    ):
        raise ValueError("PSS history reconstruction requires exactly zero pinned velocity")
    _validate_decoder_connectivity(scene)

    previous_positions = np.asarray(scene.x_current) - checkpoint_dt * np.asarray(scene.velocity)
    if not np.isfinite(previous_positions).all():
        raise ValueError("derived previous positions must be finite")
    if not np.array_equal(previous_positions[scene.pinned_indices], scene.pin_targets):
        raise ValueError("derived previous positions do not preserve the static pin history")

    common_problem_start = time.perf_counter()
    problem = build_common_problem(scene)
    objective_manifest = common_objective_manifest(scene, problem)
    common_problem_seconds = time.perf_counter() - common_problem_start

    predictor_setup_start = time.perf_counter()
    predictor = build_stretch_predictor(
        "graph-transformer",
        np.array(scene.rest_q, copy=True),
        np.array(scene.tet_indices, copy=True),
        torch_device,
        torch.float32,
        residual=True,
        graph_config=graph_config,
    )
    realized_hierarchy_levels = int(predictor.model.n_levels)
    architecture_version = int(graph_config["architecture_version"])
    if architecture_version >= 2 and realized_hierarchy_levels > training_hierarchy_levels:
        raise ValueError(
            f"inference hierarchy depth {realized_hierarchy_levels} exceeds the checkpoint's "
            f"training-realized depth {training_hierarchy_levels}"
        )
    if architecture_version < 2 and realized_hierarchy_levels != training_hierarchy_levels:
        raise ValueError(
            f"architecture-v{architecture_version} checkpoint was trained at hierarchy depth "
            f"{training_hierarchy_levels}, but inference realized depth {realized_hierarchy_levels}"
        )
    load_stretch_predictor_state(predictor, checkpoint)
    predictor.eval()
    _synchronize(torch_device)
    predictor_setup_seconds = time.perf_counter() - predictor_setup_start

    decoder_setup_start = time.perf_counter()
    decoder_state = torch_solver.build_solver(
        np.array(scene.rest_q, copy=True),
        np.array(scene.tet_indices, copy=True),
        np.array(scene.tet_poses, copy=True),
        np.array(scene.pinned_indices, copy=True),
        device=torch_device,
        dtype=torch.float64,
    )
    _synchronize(torch_device)
    decoder_setup_seconds = time.perf_counter() - decoder_setup_start

    input_setup_start = time.perf_counter()
    x_current = torch.as_tensor(np.array(scene.x_current, copy=True), dtype=torch.float64, device=torch_device)
    x_previous = torch.as_tensor(previous_positions.copy(), dtype=torch.float64, device=torch_device)
    force = torch.as_tensor(np.array(scene.external_force, copy=True), dtype=torch.float64, device=torch_device)
    gravity = torch.as_tensor(np.array(scene.gravity, copy=True), dtype=torch.float64, device=torch_device)
    mu = torch.as_tensor(np.array(scene.tet_materials[:, 0], copy=True), dtype=torch.float32, device=torch_device)
    lam = torch.as_tensor(np.array(scene.tet_materials[:, 1], copy=True), dtype=torch.float32, device=torch_device)
    vertex_is_pinned = np.zeros(scene.n_vertices, dtype=bool)
    vertex_is_pinned[scene.pinned_indices] = True
    pin_flag = torch.as_tensor(
        vertex_is_pinned[scene.tet_indices].any(axis=1).astype(np.float32),
        dtype=torch.float32,
        device=torch_device,
    )
    pin_targets = torch.as_tensor(np.array(scene.pin_targets, copy=True), dtype=torch.float64, device=torch_device)
    iterate_zero_np = problem.inertial_target.detach().numpy().copy()
    iterate_zero = torch.as_tensor(iterate_zero_np, dtype=torch.float64, device=torch_device)
    _synchronize(torch_device)
    input_setup_seconds = time.perf_counter() - input_setup_start
    setup_seconds = time.perf_counter() - setup_start

    def pipeline() -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
        x_init = iterate_zero.clone()
        _synchronize(torch_device)
        total_start = time.perf_counter()
        predictor_start = total_start
        target = predictor.model(
            decoder_state,
            x_current,
            x_previous,
            force,
            gravity,
            mu,
            lam,
            pin_flag,
        )
        _synchronize(torch_device)
        predictor_elapsed = time.perf_counter() - predictor_start
        decoder_start = time.perf_counter()
        positions = torch_solver.solve(
            decoder_state,
            target.to(dtype=torch.float64),
            pin_targets,
            x_init=x_init,
            n_iters=decoder_iterations,
        )
        _synchronize(torch_device)
        decoder_elapsed = time.perf_counter() - decoder_start
        return target, positions, time.perf_counter() - total_start, predictor_elapsed, decoder_elapsed

    warmup_seconds = 0.0
    warmup_target = None
    warmup_positions = None
    target_repeat_scale = 0.0
    target_repeat_tolerance = 0.0
    position_repeat_scale_m = _position_repeat_scale_m(scene)
    position_repeat_tolerance_m = (
        _cuda_position_repeat_tolerance_m(position_repeat_scale_m) if torch_device.type == "cuda" else 0.0
    )
    position_repeat_max_abs_discrepancy_m = 0.0
    target_repeat_max_abs_discrepancy = 0.0

    def compare_repeat(
        candidate_target: torch.Tensor,
        candidate_positions: torch.Tensor,
        reference_target: torch.Tensor,
        reference_positions: torch.Tensor,
        label: str,
    ) -> None:
        nonlocal position_repeat_max_abs_discrepancy_m, target_repeat_max_abs_discrepancy
        target_discrepancy = float((candidate_target - reference_target).abs().max())
        position_discrepancy = float((candidate_positions - reference_positions).abs().max())
        target_repeat_max_abs_discrepancy = max(target_repeat_max_abs_discrepancy, target_discrepancy)
        position_repeat_max_abs_discrepancy_m = max(position_repeat_max_abs_discrepancy_m, position_discrepancy)
        if target_discrepancy > target_repeat_tolerance:
            raise RuntimeError(
                f"{label} PSS predictor target differs from the representative repeat by "
                f"{target_discrepancy:.3e}, exceeding {target_repeat_tolerance:.3e}"
            )
        if position_discrepancy > position_repeat_tolerance_m:
            raise RuntimeError(
                f"{label} PSS decoder positions differ from the representative repeat by "
                f"{position_discrepancy:.3e} m, exceeding {position_repeat_tolerance_m:.3e} m"
            )

    with torch.no_grad():
        if warmup:
            warmup_target, warmup_positions, warmup_seconds, _predictor_seconds, _decoder_seconds = pipeline()
            _validate_candidate(warmup_target, warmup_positions, decoder_state.pinned, pin_targets)
            warmup_target = warmup_target.detach().clone()
            warmup_positions = warmup_positions.detach().clone()

        repeat_seconds: list[float] = []
        predictor_seconds: list[float] = []
        decoder_seconds: list[float] = []
        transfer_seconds: list[float] = []
        reference_target = None
        reference_positions = None
        target_np = None
        positions_np = None
        for _ in range(repeats):
            target, positions, total_elapsed, predictor_elapsed, decoder_elapsed = pipeline()
            _validate_candidate(target, positions, decoder_state.pinned, pin_targets)
            if reference_target is None:
                reference_target = target.detach().clone()
                reference_positions = positions.detach().clone()
                target_repeat_scale = _target_repeat_scale(reference_target)
                if torch_device.type == "cuda":
                    target_repeat_tolerance = _cuda_target_repeat_tolerance(target_repeat_scale)
            else:
                compare_repeat(target, positions, reference_target, reference_positions, "repeated")
            transfer_start = time.perf_counter()
            current_target_np = target.detach().cpu().numpy().astype(np.float32, copy=True)
            current_positions_np = positions.detach().cpu().numpy().astype(np.float64, copy=True)
            transfer_seconds.append(time.perf_counter() - transfer_start)
            if target_np is None:
                target_np = current_target_np
                positions_np = current_positions_np
            repeat_seconds.append(total_elapsed)
            predictor_seconds.append(predictor_elapsed)
            decoder_seconds.append(decoder_elapsed)

        if warmup_target is not None:
            compare_repeat(warmup_target, warmup_positions, reference_target, reference_positions, "warmup")

    scene_sha256 = str(scene.manifest()["scene_sha256"])
    objective_instance_sha256 = str(objective_manifest["objective_instance_sha256"])
    target_stretch_sha256 = _array_digest(target_np)
    result_state_sha256 = _array_digest(positions_np)
    result = PSSRunResult(
        positions=positions_np,
        target_stretch=target_np,
        previous_positions=previous_positions,
        decoder_iterations=decoder_iterations,
        checkpoint_name=checkpoint_path.name,
        checkpoint_sha256=checkpoint_sha256,
        predictor_config_json=predictor_config_json,
        checkpoint_args_json=checkpoint_args_json,
        parameter_count=sum(parameter.numel() for parameter in predictor.parameters()),
        training_hierarchy_levels=training_hierarchy_levels,
        realized_hierarchy_levels=realized_hierarchy_levels,
        device=str(torch_device),
        setup_seconds=setup_seconds,
        checkpoint_load_seconds=checkpoint_load_seconds,
        common_problem_seconds=common_problem_seconds,
        decoder_setup_seconds=decoder_setup_seconds,
        predictor_setup_seconds=predictor_setup_seconds,
        input_setup_seconds=input_setup_seconds,
        warmup_seconds=warmup_seconds,
        repeat_seconds=tuple(repeat_seconds),
        predictor_seconds=tuple(predictor_seconds),
        decoder_seconds=tuple(decoder_seconds),
        transfer_seconds=tuple(transfer_seconds),
        target_repeat_scale=target_repeat_scale,
        target_repeat_tolerance=target_repeat_tolerance,
        position_repeat_tolerance_m=position_repeat_tolerance_m,
        position_repeat_scale_m=position_repeat_scale_m,
        position_repeat_max_abs_discrepancy_m=position_repeat_max_abs_discrepancy_m,
        target_repeat_max_abs_discrepancy=target_repeat_max_abs_discrepancy,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        physical_state_sha256=_array_digest(scene.x_current),
        previous_state_sha256=_array_digest(previous_positions),
        iterate_zero_sha256=_array_digest(iterate_zero_np),
        target_stretch_sha256=target_stretch_sha256,
        result_state_sha256=result_state_sha256,
        newton_revision=_git_revision(),
        dirty_tree_sha256=_git_dirty_digest(),
        run_sha256="",
    )
    return dataclasses.replace(result, run_sha256=_pss_run_digest(result))


def pss_run_record(
    result: PSSRunResult,
    metrics: CommonStateMetrics | None = None,
    *,
    scene: TetBenchmarkScene | None = None,
    reference_positions: np.ndarray | torch.Tensor | None = None,
) -> dict[str, object]:
    """Return a JSON record, independently verifying any supplied metrics.

    A scored record requires the source scene and, when error metrics are
    present, the exact reference positions. The common evaluator is rerun so
    metrics from a different objective or reference cannot be relabeled.
    """
    if result.target_stretch_sha256 != _array_digest(result.target_stretch):
        raise ValueError("PSS target stretch was modified after the bound run")
    if result.result_state_sha256 != _array_digest(result.positions):
        raise ValueError("PSS result positions were modified after the bound run")
    if result.previous_state_sha256 != _array_digest(result.previous_positions):
        raise ValueError("PSS derived history was modified after the bound run")
    if result.run_sha256 != _pss_run_digest(result):
        raise ValueError("PSS execution or configuration record was modified after the bound run")

    metrics_provenance = None
    if metrics is None:
        if reference_positions is not None:
            raise ValueError("reference_positions requires common metrics")
    else:
        if scene is None:
            raise ValueError("scene is required to bind common metrics to the PSS result")
        scene_sha256 = str(scene.manifest()["scene_sha256"])
        if scene_sha256 != result.scene_sha256:
            raise ValueError("common metrics scene does not belong to this PSS result")
        if _array_digest(scene.x_current) != result.physical_state_sha256:
            raise ValueError("common metrics scene physical state does not belong to this PSS result")
        problem = build_common_problem(scene)
        objective_instance_sha256 = str(common_objective_manifest(scene, problem)["objective_instance_sha256"])
        if objective_instance_sha256 != result.objective_instance_sha256:
            raise ValueError("common metrics objective does not belong to this PSS result")
        if _array_digest(problem.inertial_target.detach().numpy()) != result.iterate_zero_sha256:
            raise ValueError("common metrics objective has the wrong inertial iterate")
        expected_metrics = evaluate_common_state(
            problem,
            result.positions,
            reference_positions=reference_positions,
        )
        if metrics != expected_metrics:
            raise ValueError("common metrics were not evaluated with the supplied objective and reference")
        if metrics.position_sha256 != result.result_state_sha256:
            raise ValueError("common metrics do not belong to this PSS result state")
        reference_state_sha256 = None
        if reference_positions is not None:
            if isinstance(reference_positions, torch.Tensor):
                reference_array = reference_positions.detach().to(dtype=torch.float64, device="cpu").numpy().copy()
            else:
                reference_array = np.array(reference_positions, dtype=np.float64, copy=True)
            reference_state_sha256 = _array_digest(reference_array)
        metrics_provenance = {
            "evaluator": "solver_benchmark.evaluate_common_state",
            "objective_instance_sha256": objective_instance_sha256,
            "candidate_position_sha256": result.result_state_sha256,
            "reference_state_sha256": reference_state_sha256,
        }

    record: dict[str, object] = {
        "method": _METHOD,
        "claim_boundary": {
            "common_objective_convergence": False,
            "description": _WORK_SEMANTICS,
            "decoder_objective": "target-stretch compatibility surrogate, not the common stable-NH potential",
        },
        "config": {
            "decoder_iterations": result.decoder_iterations,
            "predictor_passes": 1,
            "device": result.device,
            "network_dtype": "float32",
            "decoder_dtype": "float64",
            "history_contract": _HISTORY_CONTRACT,
            "predictor": json.loads(result.predictor_config_json),
        },
        "checkpoint": {
            "name": result.checkpoint_name,
            "sha256": result.checkpoint_sha256,
            "args": json.loads(result.checkpoint_args_json),
            "parameter_count": result.parameter_count,
            "training_hierarchy_levels": result.training_hierarchy_levels,
            "realized_hierarchy_levels": result.realized_hierarchy_levels,
        },
        "run_sha256": result.run_sha256,
        "scene_sha256": result.scene_sha256,
        "objective_instance_sha256": result.objective_instance_sha256,
        "physical_state_sha256": result.physical_state_sha256,
        "previous_state_sha256": result.previous_state_sha256,
        "iterate_zero_sha256": result.iterate_zero_sha256,
        "target_stretch_sha256": result.target_stretch_sha256,
        "result_state_sha256": result.result_state_sha256,
        "timing_seconds": {
            "setup": result.setup_seconds,
            "checkpoint_load": result.checkpoint_load_seconds,
            "common_problem": result.common_problem_seconds,
            "decoder_setup": result.decoder_setup_seconds,
            "predictor_setup": result.predictor_setup_seconds,
            "input_setup": result.input_setup_seconds,
            "untimed_warmup_pipeline": result.warmup_seconds,
            "pipeline_repeats": list(result.repeat_seconds),
            "pipeline_median": result.median_solve_seconds,
            "predictor_repeats": list(result.predictor_seconds),
            "decoder_repeats": list(result.decoder_seconds),
            "transfer_repeats": list(result.transfer_seconds),
        },
        "repeat_determinism": {
            "representative_repeat_index": 0,
            "target_scale": result.target_repeat_scale,
            "target_max_abs_discrepancy": result.target_repeat_max_abs_discrepancy,
            "target_tolerance": result.target_repeat_tolerance,
            "target_required_exact": result.target_repeat_tolerance == 0.0,
            "position_scale_m": result.position_repeat_scale_m,
            "position_max_abs_discrepancy_m": result.position_repeat_max_abs_discrepancy_m,
            "position_tolerance_m": result.position_repeat_tolerance_m,
        },
        "work": {
            "predictor_passes_per_repeat": 1,
            "surrogate_decoder_sweeps_per_repeat": result.decoder_iterations,
            "surrogate_polar_local_steps_per_repeat": result.decoder_iterations,
            "surrogate_global_triangular_solves_per_repeat": result.decoder_iterations,
        },
        "environment": {
            "newton_revision": result.newton_revision,
            "dirty_tree_sha256": result.dirty_tree_sha256,
            "torch": torch.__version__,
        },
    }
    if metrics is not None:
        record["metrics"] = metrics.as_dict()
        record["metrics_provenance"] = metrics_provenance
    return record
