# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Provenance-bound v3 training and evaluation on audited PR transitions.

This module is the common-objective counterpart to the legacy trajectory
trainer.  Every sample is an accepted :class:`HistoryTransition` from
``pr_scene_history``.  The graph transformer observes the post-callback state
``A_k.q`` and the exact float32 reconstruction of the preceding position,

``x_previous = float32(C_k.q - float32(C_k.qd * dt32))``.

Pin indices and targets are transition-local, so moving Dirichlet data are not
collapsed to a rest-pose constant.  Architecture v3 predicts a full
deformation-gradient field, which is decoded by exactly one weighted global
projection.  The default dense Cholesky projection is the only backend this
initial trainer accepts; a checkpoint records that choice explicitly.

The position loss is a dimensionless mass-weighted free-vertex error,

``sum_i m_i ||x_i - x_i_ref||^2 / (sum_i m_i * ell^2)``,

where ``ell`` is the static RMS rest-edge length.  An optional deformation-
gradient term is the rest-volume-weighted mean squared component error.  Both
normalizations are fixed by the authenticated static bundle.

Checkpoint tensor state and JSON metadata have separate SHA-256 digests.  The
metadata binds the history manifest, complete static bundle, selected
transition hashes and prefixes, exact model configuration, realized hierarchy
depth, seeds, loss contract, and decoder work.  A selected transition set is
never described as a complete history.

This remains research infrastructure rather than a public Newton API.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from . import torch_solver
from .graph_transformer import GraphTransformerConfig
from .pr_scene_history import HistoryTransition, PRHistoryStaticBundle, PRSceneHistory
from .predictor import (
    StretchPredictor,
    build_stretch_predictor,
    load_stretch_predictor_state,
    predictor_decoder_work,
)
from .solver_benchmark import build_common_problem, common_objective_manifest, evaluate_common_state

_SCHEMA_VERSION = 1
_CHECKPOINT_CONTRACT = "pr2901-history-v3-checkpoint-v1"
_EVALUATION_CONTRACT = "pr2901-history-v3-evaluation-v1"
_SUPPORTED_PROJECTION_BACKEND = "dense_cholesky"


def _jsonable(value: object) -> object:
    """Return a finite, canonical-JSON-compatible copy."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("authenticated JSON metadata must be finite")
        return value
    raise TypeError(f"unsupported authenticated metadata type {type(value).__name__}")


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    dtype = array.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=canonical_dtype)


def _array_digest(value: np.ndarray) -> str:
    array = _canonical_array(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and canonical CPU bytes."""
    records: list[dict[str, object]] = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"state_dict entry {name!r} is not a tensor")
        array = tensor.detach().cpu().contiguous().numpy()
        records.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": _array_digest(array),
            }
        )
    return _canonical_digest(records)


def _seed_everything(seed: int) -> np.random.Generator:
    """Seed NumPy and Torch generators and return the sampling generator."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _source_provenance() -> dict[str, str | None]:
    """Return HEAD and a content digest for every dirty source path."""
    repository = pathlib.Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "-z"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    dirty_sha256 = None
    if status.returncode == 0 and status.stdout:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        digest = hashlib.sha256()
        digest.update(status.stdout)
        digest.update(diff.stdout)
        if untracked.returncode == 0:
            for relative_bytes in sorted(item for item in untracked.stdout.split(b"\0") if item):
                path = repository / relative_bytes.decode("utf-8", errors="surrogateescape")
                digest.update(relative_bytes)
                digest.update(path.read_bytes())
        dirty_sha256 = digest.hexdigest()
    return {
        "newton_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty_tree_sha256": dirty_sha256,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _characteristic_length(rest_q: np.ndarray, tets: np.ndarray) -> float:
    """Return the RMS length over all six rest edges of every tet."""
    corners = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    squared = [
        np.sum((rest_q[tets[:, a]].astype(np.float64) - rest_q[tets[:, b]].astype(np.float64)) ** 2, axis=1)
        for a, b in corners
    ]
    length = float(np.sqrt(np.mean(np.concatenate(squared))))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("static mesh has no finite positive characteristic edge length")
    return length


def _pin_flag(pinned_indices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pinned = np.zeros(int(tets.max()) + 1, dtype=bool)
    pinned[pinned_indices] = True
    return pinned[tets].any(axis=1).astype(np.float32)


def _float32_previous(transition: HistoryTransition) -> np.ndarray:
    dt32 = np.float32(transition.dt_seconds)
    displacement = (transition.input_state.qd * dt32).astype(np.float32)
    return (transition.input_state.q - displacement).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class PRV3TrainingConfig:
    """Optimization and loss settings authenticated in every checkpoint."""

    steps: int = 1000
    batch_size: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    position_loss_weight: float = 1.0
    deformation_gradient_loss_weight: float = 0.0
    gradient_clip_norm: float = 5.0
    seed: int = 0
    log_every: int = 50
    projection_backend: str = _SUPPORTED_PROJECTION_BACKEND

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.log_every, bool) or self.log_every < 1:
            raise ValueError("log_every must be a positive integer")
        for name in (
            "learning_rate",
            "position_loss_weight",
            "deformation_gradient_loss_weight",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        if self.position_loss_weight == 0.0 and self.deformation_gradient_loss_weight == 0.0:
            raise ValueError("at least one loss weight must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.projection_backend != _SUPPORTED_PROJECTION_BACKEND:
            raise ValueError(
                f"unsupported projection_backend {self.projection_backend!r}; "
                f"this trainer currently requires {_SUPPORTED_PROJECTION_BACKEND!r}"
            )


@dataclasses.dataclass(frozen=True)
class _PreparedSample:
    transition: HistoryTransition
    x_current: np.ndarray
    x_previous: np.ndarray
    pinned_indices: np.ndarray
    pin_targets: np.ndarray
    reference_positions: np.ndarray
    pin_signature: tuple[int, ...]


@dataclasses.dataclass
class _PreparedDataset:
    history: PRSceneHistory
    static: PRHistoryStaticBundle
    samples: tuple[_PreparedSample, ...]
    characteristic_length_m: float
    selection_record: dict[str, object]


@dataclasses.dataclass
class PRV3TrainingResult:
    """In-memory result of one authenticated training run."""

    predictor: StretchPredictor
    checkpoint: dict[str, object]


def _validate_static_bundle(history: PRSceneHistory) -> PRHistoryStaticBundle:
    static = history.static_bundle
    manifest = history.manifest
    expected = {
        "manifest_sha256": manifest.manifest_sha256,
        "base_physical_sha256": manifest.base_physical_sha256,
        "topology_sha256": manifest.topology_sha256,
        "material_sha256": manifest.material_sha256,
    }
    for name, value in expected.items():
        if getattr(static, name) != value:
            raise ValueError(f"history static bundle has inconsistent {name}")
    if static.as_dict().get("static_sha256") != static.static_sha256:
        raise ValueError("history static bundle record is not self-consistent")
    return static


def _prepare_dataset(
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
) -> _PreparedDataset:
    """Validate and content-address one same-topology transition selection."""
    static = _validate_static_bundle(history)
    transitions = tuple(transitions)
    if not transitions:
        raise ValueError("at least one accepted history transition is required")
    hashes = [transition.transition_sha256 for transition in transitions]
    if len(hashes) != len(set(hashes)):
        raise ValueError("training transition selection contains duplicates")

    samples: list[_PreparedSample] = []
    transition_records: list[dict[str, object]] = []
    for transition in transitions:
        if not isinstance(transition, HistoryTransition):
            raise TypeError("every sample must be a HistoryTransition")
        if transition.manifest_sha256 != history.manifest.manifest_sha256:
            raise ValueError("transition belongs to a different history manifest")
        if transition.topology_sha256 != static.topology_sha256:
            raise ValueError("multi-topology training is not yet supported")
        if transition.material_sha256 != static.material_sha256:
            raise ValueError("transition material does not match the static bundle")
        if transition.dt_seconds != history.manifest.dt_seconds:
            raise ValueError("transition dt does not exactly match the history manifest")
        if transition.reference_record.get("accepted") is not True:
            raise ValueError("training requires an explicitly accepted dense-Newton reference")

        model_inputs = transition.model_inputs()
        expected_keys = (
            "x_current",
            "x_previous",
            "pinned_indices",
            "pin_targets",
            "inertial_target",
        )
        if tuple(model_inputs) != expected_keys:
            raise ValueError("transition model-input schema is not the expected v1 order")
        if not np.array_equal(model_inputs["x_current"], transition.applied_state.q):
            raise ValueError("transition model x_current is not the post-callback state A_k.q")
        if not np.array_equal(model_inputs["x_previous"], _float32_previous(transition)):
            raise ValueError("transition model x_previous violates the float32 velocity contract")
        if not np.array_equal(model_inputs["pinned_indices"], transition.applied_state.pinned_indices):
            raise ValueError("transition model pinned_indices do not match A_k")
        if not np.array_equal(model_inputs["pin_targets"], transition.applied_state.pin_targets):
            raise ValueError("transition model pin_targets do not match A_k")
        if not np.array_equal(model_inputs["inertial_target"], transition.inertial_target):
            raise ValueError("transition model inertial_target is inconsistent")

        scene = history.build_atomic_scene(transition.input_state, transition.applied_state)
        problem = build_common_problem(scene)
        objective = common_objective_manifest(scene, problem)
        if scene.manifest()["scene_sha256"] != transition.scene_sha256:
            raise ValueError("transition scene hash does not match reconstructed common scene")
        if objective["objective_instance_sha256"] != transition.objective_instance_sha256:
            raise ValueError("transition objective hash does not match reconstructed common objective")

        pinned_indices = model_inputs["pinned_indices"]
        samples.append(
            _PreparedSample(
                transition=transition,
                x_current=model_inputs["x_current"],
                x_previous=model_inputs["x_previous"],
                pinned_indices=pinned_indices,
                pin_targets=model_inputs["pin_targets"],
                reference_positions=transition.reference_positions,
                pin_signature=tuple(int(index) for index in pinned_indices),
            )
        )
        training_record = transition.training_record()
        transition_records.append(
            {
                "transition_sha256": transition.transition_sha256,
                "training_record_sha256": _canonical_digest(training_record),
                "input_prefix_sha256": transition.input_prefix_sha256,
                "input_state_sha256": transition.input_state_sha256,
                "coordinate": transition.coordinate.as_dict(),
                "scene_sha256": transition.scene_sha256,
                "objective_instance_sha256": transition.objective_instance_sha256,
            }
        )

    selection_record: dict[str, object] = {
        "contract": "pr2901-history-selected-transition-set-v1",
        "provenance_scope": "selected-content-addressed-transitions-not-a-complete-history-claim",
        "history_manifest_sha256": history.manifest.manifest_sha256,
        "static_sha256": static.static_sha256,
        "transitions": transition_records,
    }
    selection_record["selection_sha256"] = _canonical_digest(selection_record)
    return _PreparedDataset(
        history=history,
        static=static,
        samples=tuple(samples),
        characteristic_length_m=_characteristic_length(static.rest_q, static.tet_indices),
        selection_record=selection_record,
    )


def _validate_graph_config(config: GraphTransformerConfig, dt: float) -> None:
    if config.architecture_version != 3:
        raise ValueError("PR history common training requires graph-transformer architecture version 3")
    if float(np.float32(config.dt)) != dt or config.dt != dt:
        raise ValueError(
            "graph-transformer dt must exactly equal the transition float32 dt; "
            f"got config.dt={config.dt!r}, transition.dt={dt!r}"
        )


def _decoder_work(projection_backend: str) -> dict[str, object]:
    if projection_backend != _SUPPORTED_PROJECTION_BACKEND:
        raise ValueError(f"unsupported projection backend {projection_backend!r}")
    # Use the shared work contract, then make the backend and factorization
    # explicit rather than inferring Cholesky from "one global solve".
    work = {
        "schema_version": 1,
        "target": "full-deformation-gradient",
        "decoder": "weighted-global-projection",
        "predictor_passes": 1,
        "global_triangular_solves": 1,
        "local_polar_sweeps": 0,
        "projection_backend": projection_backend,
        "factorization": "dense-prefactored-cholesky",
        "linear_solve_rhs": "three-coordinate-columns-per-sample",
    }
    return work


def _build_predictor_and_solvers(
    dataset: _PreparedDataset,
    graph_config: GraphTransformerConfig,
    device: torch.device,
) -> tuple[StretchPredictor, dict[tuple[int, ...], torch_solver.SolverState]]:
    static = dataset.static
    predictor = build_stretch_predictor(
        "graph-transformer",
        np.array(static.rest_q, copy=True),
        np.array(static.tet_indices, copy=True),
        device,
        torch.float32,
        residual=True,
        graph_config=graph_config,
    )
    shared_work = predictor_decoder_work(predictor, solver_iterations=1, blocks=1)
    if shared_work["target"] != "full-deformation-gradient" or shared_work["global_triangular_solves"] != 1:
        raise RuntimeError("shared predictor work contract is not the v3 one-shot projection")

    return predictor, _build_solvers(dataset, device)


def _build_solvers(
    dataset: _PreparedDataset,
    device: torch.device,
) -> dict[tuple[int, ...], torch_solver.SolverState]:
    """Build one dense projection factorization per dynamic pin signature."""
    static = dataset.static

    solvers: dict[tuple[int, ...], torch_solver.SolverState] = {}
    for sample in dataset.samples:
        if sample.pin_signature not in solvers:
            solvers[sample.pin_signature] = torch_solver.build_solver(
                np.array(static.rest_q, copy=True),
                np.array(static.tet_indices, copy=True),
                np.array(static.tet_poses, copy=True),
                np.array(sample.pinned_indices, copy=True),
                device=device,
                dtype=torch.float64,
            )
    return solvers


def _grouped_prediction(
    predictor: StretchPredictor,
    solvers: Mapping[tuple[int, ...], torch_solver.SolverState],
    static: PRHistoryStaticBundle,
    samples: Sequence[_PreparedSample],
    device: torch.device,
) -> list[torch.Tensor]:
    """Predict samples in pin-signature batches while preserving input order."""
    groups: dict[tuple[int, ...], list[tuple[int, _PreparedSample]]] = defaultdict(list)
    for output_index, sample in enumerate(samples):
        groups[sample.pin_signature].append((output_index, sample))

    output: list[torch.Tensor | None] = [None] * len(samples)
    force_one = torch.tensor(static.external_force, dtype=torch.float64, device=device)
    gravity = torch.tensor(static.gravity, dtype=torch.float64, device=device)
    mu = torch.tensor(static.tet_materials[:, 0], dtype=torch.float32, device=device)
    lam = torch.tensor(static.tet_materials[:, 1], dtype=torch.float32, device=device)

    for signature, indexed_samples in groups.items():
        state = solvers[signature]
        group_samples = [sample for _index, sample in indexed_samples]
        x_current = torch.as_tensor(
            np.stack([sample.x_current for sample in group_samples]),
            dtype=torch.float64,
            device=device,
        )
        x_previous = torch.as_tensor(
            np.stack([sample.x_previous for sample in group_samples]),
            dtype=torch.float64,
            device=device,
        )
        force = force_one[None].expand(len(group_samples), -1, -1)
        pin = torch.as_tensor(
            _pin_flag(group_samples[0].pinned_indices, static.tet_indices),
            dtype=torch.float32,
            device=device,
        )
        pinned_targets = torch.as_tensor(
            np.stack([sample.pin_targets for sample in group_samples]),
            dtype=torch.float64,
            device=device,
        )
        target_F = predictor.predict_deformation_gradient(
            state,
            x_current,
            x_previous,
            force,
            gravity,
            mu,
            lam,
            pin,
        )
        predicted = torch_solver.project_deformation_gradient(state, target_F, pinned_targets)
        for local_index, (output_index, _sample) in enumerate(indexed_samples):
            output[output_index] = predicted[local_index]

    if any(value is None for value in output):
        raise RuntimeError("internal grouped prediction did not fill every sample")
    return [value for value in output if value is not None]


def _sample_loss(
    prediction: torch.Tensor,
    sample: _PreparedSample,
    state: torch_solver.SolverState,
    static: PRHistoryStaticBundle,
    characteristic_length_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = torch.tensor(sample.reference_positions, dtype=torch.float64, device=prediction.device)
    mass = torch.tensor(static.mass, dtype=torch.float64, device=prediction.device)
    difference = prediction[state.free] - reference[state.free]
    free_mass = mass[state.free]
    position_loss = (free_mass[:, None] * difference.square()).sum()
    position_loss = position_loss / (free_mass.sum() * characteristic_length_m**2)

    predicted_F = torch_solver.compute_F(prediction, state.tets, state.J)
    reference_F = torch_solver.compute_F(reference, state.tets, state.J)
    deformation_loss = (state.w[:, None, None] * (predicted_F - reference_F).square()).sum()
    deformation_loss = deformation_loss / (9.0 * state.w.sum())
    return position_loss, deformation_loss


def train_pr_history_v3(
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
    *,
    graph_config: GraphTransformerConfig | None = None,
    training_config: PRV3TrainingConfig | None = None,
    device: torch.device | str = "cpu",
    output_path: pathlib.Path | str | None = None,
) -> PRV3TrainingResult:
    """Train architecture v3 on accepted, authenticated PR transitions.

    Args:
        history: Exact PR schedule and static common-objective bundle.
        transitions: One or more accepted same-topology transition samples.
        graph_config: Exact v3 graph configuration.  When omitted, defaults are
            used with the history's canonical float32 timestep.
        training_config: Optimizer, seed, losses, and projection backend.
        device: Torch device for predictor and dense projection.
        output_path: Optional checkpoint path written with :func:`torch.save`.

    Returns:
        Predictor plus authenticated in-memory checkpoint.
    """
    output = None if output_path is None else pathlib.Path(output_path)
    if output is not None and output.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint {output}")
    dataset = _prepare_dataset(history, transitions)
    train_config = PRV3TrainingConfig() if training_config is None else training_config
    dt = history.manifest.dt_seconds
    model_config = GraphTransformerConfig(dt=dt) if graph_config is None else graph_config
    _validate_graph_config(model_config, dt)
    device = torch.device(device)
    rng = _seed_everything(train_config.seed)

    predictor, solvers = _build_predictor_and_solvers(dataset, model_config, device)
    predictor.train()
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    decoder_work = _decoder_work(train_config.projection_backend)

    log: list[dict[str, object]] = []
    start = time.perf_counter()
    sample_count = len(dataset.samples)
    for step in range(train_config.steps):
        selected_indices = rng.choice(
            sample_count,
            size=train_config.batch_size,
            replace=train_config.batch_size > sample_count,
        )
        selected = [dataset.samples[int(index)] for index in selected_indices]
        optimizer.zero_grad(set_to_none=True)
        predictions = _grouped_prediction(predictor, solvers, dataset.static, selected, device)
        position_loss = torch.zeros((), dtype=torch.float64, device=device)
        deformation_loss = torch.zeros((), dtype=torch.float64, device=device)
        for prediction, sample in zip(predictions, selected, strict=True):
            sample_position, sample_deformation = _sample_loss(
                prediction,
                sample,
                solvers[sample.pin_signature],
                dataset.static,
                dataset.characteristic_length_m,
            )
            position_loss = position_loss + sample_position
            deformation_loss = deformation_loss + sample_deformation
        position_loss = position_loss / len(selected)
        deformation_loss = deformation_loss / len(selected)
        loss = (
            train_config.position_loss_weight * position_loss
            + train_config.deformation_gradient_loss_weight * deformation_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), train_config.gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite predictor gradient at step {step}")
        optimizer.step()

        if step % train_config.log_every == 0 or step + 1 == train_config.steps:
            log.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "normalized_position_loss": float(position_loss.detach()),
                    "volume_weighted_deformation_gradient_loss": float(deformation_loss.detach()),
                    "gradient_norm_before_clipping": float(gradient_norm.detach()),
                    "sample_indices": [int(index) for index in selected_indices],
                    "transition_sha256": [sample.transition.transition_sha256 for sample in selected],
                }
            )

    _synchronize(device)
    train_seconds = time.perf_counter() - start
    predictor.eval()
    state_dict = {name: value.detach().cpu().clone() for name, value in predictor.model.state_dict().items()}
    state_dict_sha256 = _state_dict_digest(state_dict)
    predictor_config = predictor.checkpoint_config()
    realized_levels = int(predictor.model.n_levels)
    metadata: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "contract": _CHECKPOINT_CONTRACT,
        "history_manifest": history.manifest.as_dict(),
        "static_bundle": dataset.static.as_dict(),
        "transition_selection": dataset.selection_record,
        "predictor_config": predictor_config,
        "training_realized_hierarchy_levels": realized_levels,
        "decoder_work": decoder_work,
        "training_config": dataclasses.asdict(train_config),
        "seed_contract": {
            "numpy_generator": "PCG64",
            "numpy_generator_seed": train_config.seed,
            "torch_manual_seed": train_config.seed,
            "torch_cuda_manual_seed_all": train_config.seed,
        },
        "loss_contract": {
            "position": "sum_free(m_i*||x_i-x_ref_i||^2)/(sum_free(m_i)*rms_rest_edge^2)",
            "deformation_gradient": "sum_t(V_t*||F_t-F_ref_t||_F^2)/(9*sum_t(V_t))",
            "characteristic_length_m": dataset.characteristic_length_m,
            "pins": "transition-local exact Dirichlet indices and targets excluded from position loss",
        },
        "training_log": log,
        "runtime": {
            "train_seconds": train_seconds,
            "device_type": device.type,
            "parameter_count": sum(parameter.numel() for parameter in predictor.parameters()),
        },
        "source_provenance": _source_provenance(),
        "software": {"torch_version": str(torch.__version__), "numpy_version": str(np.__version__)},
    }
    if device.type == "cuda":
        metadata["runtime"]["device_name"] = torch.cuda.get_device_name(device)
    metadata_sha256 = _canonical_digest(metadata)
    checkpoint: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "contract": _CHECKPOINT_CONTRACT,
        "state_dict": state_dict,
        "state_dict_sha256": state_dict_sha256,
        "metadata": metadata,
        "metadata_sha256": metadata_sha256,
        "predictor_config": predictor_config,
        "training_realized_hierarchy_levels": realized_levels,
        "decoder_work": decoder_work,
    }
    checkpoint["checkpoint_payload_sha256"] = _canonical_digest(
        {
            "contract": _CHECKPOINT_CONTRACT,
            "state_dict_sha256": state_dict_sha256,
            "metadata_sha256": metadata_sha256,
        }
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also closes the race between the early check and
        # a different process publishing the same experiment path.
        with output.open("xb") as checkpoint_file:
            torch.save(checkpoint, checkpoint_file)
    return PRV3TrainingResult(predictor=predictor, checkpoint=checkpoint)


def _verify_checkpoint(
    checkpoint: Mapping[str, object],
    dataset: _PreparedDataset,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if checkpoint.get("schema_version") != _SCHEMA_VERSION or checkpoint.get("contract") != _CHECKPOINT_CONTRACT:
        raise ValueError("unsupported PR history v3 checkpoint schema")
    metadata = checkpoint.get("metadata")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint is missing metadata or state_dict")
    if _canonical_digest(metadata) != checkpoint.get("metadata_sha256"):
        raise ValueError("checkpoint metadata SHA-256 verification failed")
    tensor_state: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint state_dict must map strings to tensors")
        tensor_state[name] = value
    if _state_dict_digest(tensor_state) != checkpoint.get("state_dict_sha256"):
        raise ValueError("checkpoint state_dict SHA-256 verification failed")
    expected_payload = _canonical_digest(
        {
            "contract": _CHECKPOINT_CONTRACT,
            "state_dict_sha256": checkpoint["state_dict_sha256"],
            "metadata_sha256": checkpoint["metadata_sha256"],
        }
    )
    if expected_payload != checkpoint.get("checkpoint_payload_sha256"):
        raise ValueError("checkpoint payload SHA-256 verification failed")

    if _jsonable(metadata.get("history_manifest")) != _jsonable(dataset.history.manifest.as_dict()):
        raise ValueError("checkpoint history manifest does not match evaluation history")
    if _jsonable(metadata.get("static_bundle")) != _jsonable(dataset.static.as_dict()):
        raise ValueError("checkpoint static bundle does not match evaluation history")
    predictor_config = metadata.get("predictor_config")
    if not isinstance(predictor_config, Mapping):
        raise ValueError("checkpoint predictor_config is missing")
    predictor_config_copy = dict(predictor_config)
    graph = predictor_config_copy.get("graph_transformer")
    if predictor_config_copy.get("kind") != "graph-transformer" or not isinstance(graph, Mapping):
        raise ValueError("checkpoint is not a graph-transformer checkpoint")
    if graph.get("architecture_version") != 3:
        raise ValueError("checkpoint is not architecture version 3")
    if graph.get("dt") != dataset.history.manifest.dt_seconds:
        raise ValueError("checkpoint graph timestep does not exactly match evaluation transitions")
    if checkpoint.get("predictor_config") != predictor_config:
        raise ValueError("checkpoint predictor config copies disagree")
    if checkpoint.get("decoder_work") != metadata.get("decoder_work"):
        raise ValueError("checkpoint decoder work copies disagree")
    if metadata.get("decoder_work") != _decoder_work(_SUPPORTED_PROJECTION_BACKEND):
        raise ValueError("checkpoint decoder work is not the supported dense one-shot projection")
    if checkpoint.get("training_realized_hierarchy_levels") != metadata.get("training_realized_hierarchy_levels"):
        raise ValueError("checkpoint realized hierarchy depth copies disagree")
    return predictor_config_copy, tensor_state


def load_pr_history_v3_checkpoint(
    checkpoint_or_path: Mapping[str, object] | pathlib.Path | str,
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
    *,
    device: torch.device | str = "cpu",
) -> tuple[StretchPredictor, _PreparedDataset, dict[str, object]]:
    """Verify and load a v3 checkpoint against an exact history dataset."""
    dataset = _prepare_dataset(history, transitions)
    device = torch.device(device)
    if isinstance(checkpoint_or_path, Mapping):
        checkpoint = dict(checkpoint_or_path)
    else:
        checkpoint = torch.load(checkpoint_or_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint file did not contain a dictionary")
    predictor_config, state_dict = _verify_checkpoint(checkpoint, dataset)
    graph_config = GraphTransformerConfig(**dict(predictor_config["graph_transformer"]))
    _validate_graph_config(graph_config, history.manifest.dt_seconds)
    predictor, _solvers = _build_predictor_and_solvers(dataset, graph_config, device)
    load_stretch_predictor_state(predictor, {"state_dict": state_dict})
    predictor.eval()
    realized = int(predictor.model.n_levels)
    if realized != checkpoint["training_realized_hierarchy_levels"]:
        raise ValueError(
            "evaluation hierarchy realizes a different depth from training; "
            f"evaluation={realized}, training={checkpoint['training_realized_hierarchy_levels']}"
        )
    return predictor, dataset, checkpoint


def evaluate_pr_history_v3(
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
    checkpoint_or_path: Mapping[str, object] | pathlib.Path | str,
    *,
    device: torch.device | str = "cpu",
    warmup: int = 1,
    repeats: int = 5,
) -> dict[str, object]:
    """Evaluate v3 positions with the independent common-objective scorer.

    The requested transitions may be held-out samples, but they must share the
    exact authenticated history manifest, static mesh, material, and timestep
    with the checkpoint.  Timings cover one predictor pass plus one projection;
    common-objective scoring is timed separately and never included in solver
    inference time.
    """
    if isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    device = torch.device(device)
    predictor, dataset, checkpoint = load_pr_history_v3_checkpoint(
        checkpoint_or_path,
        history,
        transitions,
        device=device,
    )
    solvers = _build_solvers(dataset, device)
    # CUDA graph aggregation uses atomic index additions.  On the audited PR
    # mesh their ordering changes positions at roughly 1e-10 m, so retain a
    # strict one-nanometre ceiling while CPU inference remains bit-exact.
    repeat_tolerance_m = 0.0 if device.type == "cpu" else 1.0e-9

    sample_records: list[dict[str, object]] = []
    timing_records: list[dict[str, object]] = []
    with torch.no_grad():
        for sample in dataset.samples:
            for _ in range(warmup):
                _grouped_prediction(predictor, solvers, dataset.static, [sample], device)
            _synchronize(device)

            durations: list[float] = []
            repeat_positions: list[np.ndarray] = []
            for _ in range(repeats):
                _synchronize(device)
                start = time.perf_counter()
                candidate = _grouped_prediction(predictor, solvers, dataset.static, [sample], device)[0]
                _synchronize(device)
                durations.append(time.perf_counter() - start)
                # Device-to-host transfer is deliberately outside the solver
                # interval.  It provides an independent repeat comparison and
                # the representative array consumed by the CPU evaluator.
                repeat_positions.append(candidate.detach().cpu().numpy())
            if not repeat_positions:
                raise RuntimeError("evaluation produced no prediction")
            prediction = repeat_positions[0]
            repeat_max_discrepancy_m = max(
                float(np.max(np.abs(candidate - prediction))) for candidate in repeat_positions
            )
            if repeat_max_discrepancy_m > repeat_tolerance_m:
                raise RuntimeError(
                    "repeat inference outputs exceeded the device-aware discrepancy tolerance: "
                    f"observed={repeat_max_discrepancy_m:.3e} m, tolerance={repeat_tolerance_m:.3e} m"
                )

            # The common evaluator intentionally differentiates its scalar
            # objective even though network inference above is no-grad.
            with torch.enable_grad():
                scene = history.build_atomic_scene(sample.transition.input_state, sample.transition.applied_state)
                problem = build_common_problem(scene)
                objective = common_objective_manifest(scene, problem)
                if objective["objective_instance_sha256"] != sample.transition.objective_instance_sha256:
                    raise ValueError("evaluation transition objective hash changed after loading")
                metric_start = time.perf_counter()
                metrics = evaluate_common_state(
                    problem,
                    prediction,
                    reference_positions=sample.reference_positions,
                )
                metric_seconds = time.perf_counter() - metric_start
            sample_records.append(
                {
                    "transition_sha256": sample.transition.transition_sha256,
                    "coordinate": sample.transition.coordinate.as_dict(),
                    "scene_sha256": sample.transition.scene_sha256,
                    "objective_instance_sha256": sample.transition.objective_instance_sha256,
                    "reference_position_sha256": _array_digest(sample.reference_positions),
                    "metrics": metrics.as_dict(),
                    "decoder_work": checkpoint["decoder_work"],
                }
            )
            timing_records.append(
                {
                    "transition_sha256": sample.transition.transition_sha256,
                    "warmup_runs": warmup,
                    "repeat_seconds": durations,
                    "median_inference_seconds": statistics.median(durations),
                    "minimum_inference_seconds": min(durations),
                    "repeat_max_discrepancy_m": repeat_max_discrepancy_m,
                    "repeat_discrepancy_tolerance_m": repeat_tolerance_m,
                    "common_evaluator_seconds": metric_seconds,
                    "inference_scope": "one predictor pass plus one dense global projection",
                }
            )

    free_errors = [float(record["metrics"]["free_rms_error_m"]) for record in sample_records]
    residuals = [float(record["metrics"]["relative_residual"]) for record in sample_records]
    deterministic: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "contract": _EVALUATION_CONTRACT,
        "checkpoint_payload_sha256": checkpoint["checkpoint_payload_sha256"],
        "history_manifest_sha256": history.manifest.manifest_sha256,
        "static_sha256": dataset.static.static_sha256,
        "evaluation_selection": dataset.selection_record,
        "predictor_config": checkpoint["predictor_config"],
        "training_realized_hierarchy_levels": checkpoint["training_realized_hierarchy_levels"],
        "decoder_work": checkpoint["decoder_work"],
        "samples": sample_records,
        "summary": {
            "sample_count": len(sample_records),
            "mean_free_rms_error_m": statistics.fmean(free_errors),
            "maximum_free_rms_error_m": max(free_errors),
            "mean_relative_residual": statistics.fmean(residuals),
            "maximum_relative_residual": max(residuals),
        },
    }
    deterministic["evaluation_sha256"] = _canonical_digest(deterministic)
    timing = {
        "samples": timing_records,
        "timing_scope": "excluded from deterministic evaluation_sha256",
    }
    timing["timing_sha256"] = _canonical_digest(timing)
    return {"deterministic": deterministic, "timing": timing}
