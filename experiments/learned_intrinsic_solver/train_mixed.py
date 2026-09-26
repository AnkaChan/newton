# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental detached mixed-pool trainer; no retained intermediate dataset.

One update evaluates one proposal per ready trajectory. Epochs count global
optimizer queries, not unique trajectories. CPU preparation overlaps other
ready queries; differentiable CPU fusion still synchronizes each proposal.

Candidates start from the inertial prediction, half of them with smooth
multiscale noise; inverted or collapsed cells are accepted and only nonfinite
values raise. Every query stores its un-normalized world axis gradient and the
achieved world change of the center deformation as detached optimizer history,
which is carried across physical steps and cleared only at trajectory reset.
The per-update objective is the LeCO asinh form with the material-aware energy
floor of the revised nine-value schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import features
from .material_sampling import MaterialRanges
from .mixed_report import write_progress
from .mixed_validation import validate as _validate
from .mixed_validation import validation_chunk as _validation_chunk  # noqa: F401 -- Keep the existing test seam.

__all__ = ["PERTURBED_CANDIDATE_PROBABILITY", "MixedTrainConfig", "local_objective", "run_training"]

PERTURBED_CANDIDATE_PROBABILITY = 0.5
"""Probability that a new candidate adds smooth multiscale noise to the inertial prediction."""

_LEGACY_CONFIG_FIELDS = ("candidate_probabilities", "geometry_backtracking")


@dataclass(frozen=True)
class MixedTrainConfig:
    """Experimental V2 settings. Material is independently sampled per reset."""

    cell_counts: tuple[int, int, int] = (10, 10, 40)
    cell_size: float = 0.025
    time_step: float = 1 / 300
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    hidden_dim: int = 128
    edge_hidden_dim: int = 64
    num_heads: int = 4
    hops: tuple[int, ...] = (1,)
    query_chunk_size: int = 128
    max_step_size: float = 0.05
    learning_rate: float = 1e-4
    energy_increase_weight: float = 1.0
    energy_floor_scale: float = 1.0
    """Multiplier c of the material-aware energy floor in the per-update loss scale."""
    batch_size: int = 16
    pool_multiplier: int = 4
    queries_per_epoch: int = 8192
    max_epochs: int = 500
    stage_epochs: int = 10
    stage_max_epochs: int | None = 20
    """Hard residence limit per curriculum stage; None disables the cap."""
    stage_patience: int = 2
    stage_descent_rate: float = 0.8
    plateau_min_final_stage_epochs: int = 20
    """Final-stage epochs required before plateau stopping may be considered."""
    iteration_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    physical_step_counts: tuple[int, ...] = (8, 16, 32, 64, 128)
    validation_count: int = 512
    validation_iterations: int = 100
    validation_physical_steps: int = 8
    validation_physical_iterations: int = 2
    validation_full_count: int = 16
    """Held-out seeds of the full-horizon check, disjoint from the cheap set."""
    validation_full_interval: int = 5
    """Epoch interval of the full-horizon check; curriculum advancement also triggers it."""
    checkpoint_interval: int = 5
    early_stopping: bool = True
    youngs_modulus_range: tuple[float, float] = (1e3, 1e6)
    poissons_ratio_range: tuple[float, float] = (0.2, 0.49)
    density_range: tuple[float, float] = (100.0, 10000.0)
    damping_range: tuple[float, float] = (10.0, 1000.0)
    """Independent log-uniform absolute VBD viscosity [Pa·s]."""
    feature_schema_version: int = features.FEATURE_SCHEMA_VERSION
    """Only the revised nine-value schema (3) is supported; legacy runs restart."""
    strength_range: tuple[float, float] = (0.02, 0.1)
    velocity_dt_range: tuple[float, float] = (0.0, 0.1)
    perturbation_scale_range: tuple[float, float] = (0.0, 1.0)
    seed: int = 73
    device: str = "cuda"
    cpu_threads: int = 2
    preparation_workers: int = 2
    verbose: bool = True

    def __post_init__(self):
        """Reject invalid budgets and physical ranges before creating outputs."""
        for name in (
            "cell_counts",
            "gravity",
            "hops",
            "iteration_counts",
            "physical_step_counts",
            "youngs_modulus_range",
            "poissons_ratio_range",
            "density_range",
            "damping_range",
            "strength_range",
            "velocity_dt_range",
            "perturbation_scale_range",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in (
            "hidden_dim",
            "edge_hidden_dim",
            "num_heads",
            "query_chunk_size",
            "batch_size",
            "pool_multiplier",
            "queries_per_epoch",
            "max_epochs",
            "stage_epochs",
            "validation_count",
            "validation_iterations",
            "validation_physical_steps",
            "validation_physical_iterations",
            "validation_full_count",
            "validation_full_interval",
            "checkpoint_interval",
            "cpu_threads",
            "preparation_workers",
            "stage_patience",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        limit = self.plateau_min_final_stage_epochs
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("plateau_min_final_stage_epochs must be a nonnegative integer")
        for name in ("cell_counts", "hops", "iteration_counts", "physical_step_counts"):
            values = getattr(self, name)
            if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integers")
        if len(self.cell_counts) != 3 or self.hidden_dim % self.num_heads or self.pool_multiplier < 2:
            raise ValueError("invalid grid, attention heads, or pool multiplier (minimum 2)")
        if max(self.iteration_counts) > 32 or max(self.physical_step_counts) > 128:
            raise ValueError("V2 budgets are limited to K <= 32 and H <= 128")
        if 1 not in self.iteration_counts or min(self.physical_step_counts) > 8:
            raise ValueError("initial curriculum requires K=1 and an H <= 8")
        for name in ("cell_size", "time_step", "max_step_size", "learning_rate", "energy_floor_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.energy_increase_weight) or self.energy_increase_weight < 0:
            raise ValueError("energy_increase_weight must be finite and nonnegative")
        if not 0 <= self.stage_descent_rate <= 1:
            raise ValueError("stage_descent_rate must lie in [0,1]")
        if self.stage_max_epochs is not None and (
            isinstance(self.stage_max_epochs, bool)
            or not isinstance(self.stage_max_epochs, int)
            or self.stage_max_epochs < self.stage_epochs
        ):
            raise ValueError("stage_max_epochs must be None or an integer at least stage_epochs")
        if len(self.gravity) != 3 or not all(math.isfinite(v) for v in self.gravity):
            raise ValueError("gravity must be a finite three-vector")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("strength_range", "velocity_dt_range", "perturbation_scale_range"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 <= bounds[0] <= bounds[1]:
                raise ValueError(f"invalid {name}")
        self.material_ranges()
        if (
            isinstance(self.feature_schema_version, bool)
            or self.feature_schema_version != features.FEATURE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"feature_schema_version must be {features.FEATURE_SCHEMA_VERSION}: the revised nine-value "
                "schema; legacy checkpoints require fresh initialization"
            )

    @property
    def state_feature_dim(self):
        """Return the revised per-cell state width."""
        return features.STATE_FEATURE_DIM

    @property
    def conditioning_dim(self):
        """Return the FiLM width of the revised schema."""
        return features.CONDITIONING_DIM

    @classmethod
    def from_checkpoint_config(cls, values):
        """Rebuild a revised-schema configuration; reject legacy checkpoints explicitly.

        Raises:
            ValueError: The saved ``feature_schema_version`` is not the revised
                schema or the configuration carries the removed
                ``candidate_probabilities``/``geometry_backtracking`` fields.
        """
        values = dict(values)
        legacy = [name for name in _LEGACY_CONFIG_FIELDS if name in values]
        if values.get("feature_schema_version") != features.FEATURE_SCHEMA_VERSION or legacy:
            raise ValueError(
                "incompatible legacy checkpoint; start a fresh run: the revised nine-value schema "
                f"(feature_schema_version {features.FEATURE_SCHEMA_VERSION}) has no {', '.join(_LEGACY_CONFIG_FIELDS)}"
            )
        return cls(**values)

    def material_ranges(self):
        """Return independent E, nu, rho, and absolute viscosity sampling bounds."""
        return MaterialRanges(
            self.youngs_modulus_range, self.poissons_ratio_range, self.density_range, self.damping_range
        )


def local_objective(after, before, floor, *, increase_weight=1.0):
    """Return per-member LeCO losses for one update; gradients reach only ``after``.

    ``scale = max(|before|, floor)`` is detached, where ``before`` is the
    energy of the candidate immediately before this update and ``floor`` the
    material-aware energy floor [J]. The loss is
    ``asinh(after / scale) + increase_weight * relu((after - before) / scale)``.

    Args:
        after: Energies after the update, shape [B]; the only differentiable input.
        before: Energies immediately before the update, shape [B]; detached.
        floor: Positive finite floor [J], shape [B] or scalar; detached.
        increase_weight: Nonnegative weight of the energy-increase penalty.
    """
    import torch

    before = before.detach()
    floor = torch.as_tensor(floor, dtype=before.dtype, device=before.device).detach()
    if not torch.isfinite(floor).all() or (floor <= 0).any():
        raise ValueError("energy floor must be finite and positive")
    scale = torch.maximum(before.abs(), floor)
    return torch.asinh(after / scale) + increase_weight * torch.relu((after - before) / scale)


class _TrajectoryFactory:
    """Prepare detached CPU trajectories without accessing network parameters."""

    def __init__(self, step, rest, config, *, rank, validation=False):
        self.step, self.rest, self.config = step, rest, config
        self.prefix = f"{'validation' if validation else 'train'}-{rank}"
        self.master_seed = config.seed + (1000000007 if validation else 0)
        self.seed_parity = int(validation)
        # Preparation workers use CPU topology without synchronizing CUDA.
        self.fixed_indices = step.fixed_indices.detach().cpu().clone()
        self.cell_count = len(step.cell_corner_indices)

    def reset(self, seed):
        import torch

        from .history import empty_history  # noqa: PLC0415 -- Optional training boundary.
        from .initial_state import InitialStateAugmenter  # noqa: PLC0415 -- Optional training boundary.

        c = self.config
        initial = InitialStateAugmenter(
            self.rest,
            master_seed=self.master_seed,
            time_step=c.time_step,
            material_ranges=c.material_ranges(),
            strength_range=c.strength_range,
            velocity_dt_range=c.velocity_dt_range,
            perturbation_scale_range=c.perturbation_scale_range,
        ).reset(2 * seed + self.seed_parity)
        key = f"{self.prefix}-{seed}"
        specification = asdict(initial.material)
        self.step.register_context(key, **specification)
        try:
            payload = self.step.prepare(key, torch.from_numpy(initial.positions), torch.from_numpy(initial.velocities))
            payload.update(context_spec=specification, metadata=initial.metadata, seed=seed, physical_age=0)
            payload.update(empty_history(self.cell_count))
            return self._candidate(payload)
        except BaseException:
            self.step.discard_context(key)
            raise

    def advance(self, payload):
        from .history import carry_history  # noqa: PLC0415 -- Optional training boundary.
        from .train_epochs import _cpu  # noqa: PLC0415 -- Optional training boundary.

        payload = _cpu(payload)
        prepared = self.step.advance(payload)
        for key in ("context_spec", "metadata", "seed"):
            prepared[key] = payload[key]
        prepared["physical_age"] = payload["physical_age"] + 1
        carry_history(payload, prepared)
        return self._candidate(prepared)

    def retire(self, payload):
        self.step.discard_context(payload["context_id"])

    def _candidate(self, payload):
        """Draw one of two equiprobable initializers; never screen, shorten or repair.

        ``inertial`` uses the inertial prediction with prescribed corners set;
        ``perturbed_inertial`` adds smooth multiscale noise with an RMS of
        1-10 percent of the cell size. The rigid initializer computed by
        ``prepare`` is replaced and never used as the learned candidate.
        Only a nonfinite candidate raises.
        """
        import torch

        from .multiscale import generate_multiscale  # noqa: PLC0415 -- Optional training boundary.

        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, payload["seed"], payload["physical_age"], 911])
        )
        perturbed = rng.random() < PERTURBED_CANDIDATE_PROBABILITY
        candidate = payload["inertial_prediction"].clone()
        if perturbed:
            sample = generate_multiscale(self.rest, seed=int(rng.integers(2**32)), strength=0.1)
            noise = sample.positions - self.rest.corner_rest_positions
            rms = float(np.sqrt(np.mean(np.sum(noise**2, axis=-1))))
            noise *= float(rng.uniform(0.01, 0.1) * self.config.cell_size) / rms if rms else 0
            candidate = candidate + torch.from_numpy(noise.astype(np.float32))
        candidate[self.fixed_indices] = payload["fixed_positions"]
        if not torch.isfinite(candidate).all():
            raise ValueError("nonfinite candidate initialization")
        payload.update(candidate=candidate.detach(), candidate_mode="perturbed_inertial" if perturbed else "inertial")
        return payload


def _batch(records, device, *, cell_count=None):
    """Collate detached payload tensors and the stacked optimizer history on ``device``.

    ``cell_count`` validates stored history blocks; when omitted it is read
    from the first stored block, and payloads without any history entries
    yield ``history=None`` (no history for the whole batch).
    """
    import torch

    from .history import HISTORY_KEYS, batch_history  # noqa: PLC0415 -- Optional training boundary.

    payloads = [getattr(record, "payload", record) for record in records]
    values = {
        name: torch.stack([p[name].detach().to(device) for p in payloads])
        for name in ("candidate", "inertial_prediction", "fixed_positions", "physical_positions")
    }
    values["context_ids"] = tuple(p["context_id"] for p in payloads)
    if cell_count is None:
        blocks = [p.get(HISTORY_KEYS[0]) for p in payloads]
        blocks = [block for block in blocks if isinstance(block, torch.Tensor) and block.ndim == 3]
        if blocks:
            cell_count = int(blocks[0].shape[0])
        elif any(p.get("history_valid", False) for p in payloads):
            raise ValueError("payload history_valid is set without stored history blocks")
    values["history"] = batch_history(payloads, device, cell_count=cell_count) if cell_count is not None else None
    return values


def _checked_forward(module, step, batch):
    """Evaluate one learned proposal; reject only nonfinite outputs or moved pins."""
    import torch

    result = module(
        batch["candidate"],
        batch["inertial_prediction"],
        batch["context_ids"],
        fixed_positions=batch["fixed_positions"],
        previous_positions=batch["physical_positions"],
        history=batch.get("history"),
    )
    if not torch.isfinite(result.loss.total).all() or not torch.isfinite(result.positions).all():
        raise ValueError("nonfinite learned proposal; trajectory retained as a failure")
    if not torch.equal(result.positions[:, step.fixed_indices], batch["fixed_positions"]):
        raise ValueError("learned proposal moved prescribed corners")
    return result


def _write_report(output, report):
    from .mixed_report import write_mixed_report  # noqa: PLC0415 -- Optional training boundary.

    write_mixed_report(output, report)


def _selection_metric(validation):
    """Return the finite selection metric when the validation is eligible, else None."""
    selection = validation.get("selection") or {}
    metric = selection.get("metric")
    if selection.get("eligible") and isinstance(metric, (int, float)) and math.isfinite(metric):
        return float(metric)
    return None


def _allow_early_stop(config, curriculum) -> bool:
    """Return whether plateau stopping may be considered after this epoch's curriculum observation.

    Stopping is permitted only when enabled and the curriculum has completed at
    least ``plateau_min_final_stage_epochs`` epochs in its final stage; call
    this after ``curriculum.observe`` so a stage entered this epoch counts zero.
    """
    return bool(
        config.early_stopping
        and curriculum.stage == curriculum.final_stage
        and curriculum.stage_epochs >= config.plateau_min_final_stage_epochs
    )


def run_training(output: Path, config: MixedTrainConfig, *, resume: Path | None = None):
    """Run an explicitly requested V2 campaign or a bounded verification run.

    Checkpoints restore the same rank count, curriculum, pool queues, optimizer
    history and Adam sequence. The configured descent-rate gate and stage limit
    may change on resume; the gate applies only to future validation and an
    overdue hard limit promotes one stage immediately. These changes record
    their epoch boundary. Native factors are rebuilt; they are never
    serialized. Legacy checkpoints are rejected explicitly.

    Every epoch runs the cheap validation; every ``validation_full_interval``
    epochs, or when the curriculum could advance, the full-horizon check at the
    largest available K and H follows. The best checkpoint is selected by the
    validation ``selection`` metric (mean final free-corner force residual with
    survival required). Plateau stopping is considered only after
    ``plateau_min_final_stage_epochs`` epochs in the final curriculum stage.
    """
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from .curriculum import MixedCurriculum  # noqa: PLC0415 -- Optional training boundary.
    from .data import generate_cuboid  # noqa: PLC0415 -- Optional training boundary.
    from .history import store_history  # noqa: PLC0415 -- Optional training boundary.
    from .mixed_physics import MixedHexSolverStep  # noqa: PLC0415 -- Optional training boundary.
    from .mixed_validation import validate_full_horizon  # noqa: PLC0415 -- Optional training boundary.
    from .network import IntrinsicSolverNetwork  # noqa: PLC0415 -- Optional training boundary.
    from .train_epochs import _all_ranks_ok, _atomic_torch, _cpu  # noqa: PLC0415 -- Optional training boundary.
    from .training_schedule import PlateauController  # noqa: PLC0415 -- Optional training boundary.
    from .trajectory_pool import ActiveTrajectoryPool  # noqa: PLC0415 -- Optional training boundary.

    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if config.queries_per_epoch % (config.batch_size * world_size):
        raise ValueError("queries_per_epoch must be divisible by batch_size * world_size")
    output = Path(output).resolve()
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if saved:
        if saved.get("format") != "mixed_pool_v2" or saved["world_size"] != world_size:
            raise ValueError("incompatible checkpoint format or rank count")
        allowed = {"max_epochs", "verbose", "early_stopping", "stage_descent_rate", "stage_max_epochs"}
        saved_config = asdict(MixedTrainConfig.from_checkpoint_config(saved["config"]))
        if any(saved_config.get(k) != v for k, v in asdict(config).items() if k not in allowed):
            raise ValueError("resume configuration differs from saved physical/training configuration")
        if config.max_epochs < saved["report"]["completed_epochs"]:
            raise ValueError("max_epochs precedes the checkpoint")
    elif (output / "report.json").exists() or (output / "checkpoints").exists():
        raise FileExistsError("use a fresh output directory or an explicit resume checkpoint")
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(config.device)
    if device.type == "cuda":
        if world_size > 1 and (torch.cuda.device_count() != 1 or int(os.environ.get("LOCAL_RANK", "0")) != 0):
            raise ValueError(
                "distributed training requires one exclusively claimed CUDA device per rank; use launch_training"
            )
        torch.cuda.set_device(0)
    owned_group = world_size > 1 and not dist.is_initialized()
    if owned_group:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    random.seed(config.seed + rank)
    np.random.seed((config.seed + rank) % 2**32)  # noqa: NPY002 -- Preserve process RNG in checkpoints.
    torch.manual_seed(config.seed)
    rest = generate_cuboid(config.cell_counts, cell_size=config.cell_size)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
    network = IntrinsicSolverNetwork(
        config.cell_counts,
        config.state_feature_dim,
        conditioning_dim=config.conditioning_dim,
        hidden_dim=config.hidden_dim,
        edge_hidden_dim=config.edge_hidden_dim,
        num_heads=config.num_heads,
        hops=config.hops,
        max_step_size=config.max_step_size,
        query_chunk_size=config.query_chunk_size,
    ).to(device)
    step = MixedHexSolverStep(
        rest,
        fixed,
        network=network,
        time_step=config.time_step,
        gravity=config.gravity,
        energy_floor_scale=config.energy_floor_scale,
    ).to(device)
    cell_count = len(step.cell_corner_indices)
    factory = _TrajectoryFactory(step, rest, config, rank=rank)
    validation_factory = _TrajectoryFactory(step, rest, config, rank=rank, validation=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    controller = PlateauController(
        config.learning_rate, min_epochs=1, max_epochs=config.max_epochs, min_lr=min(1e-6, config.learning_rate)
    )
    curriculum = MixedCurriculum(
        config.iteration_counts,
        config.physical_step_counts,
        min_stage_epochs=config.stage_epochs,
        max_stage_epochs=config.stage_max_epochs,
        patience=config.stage_patience,
        min_descent_rate=config.stage_descent_rate,
    )
    pool = None
    try:
        if saved:
            network.load_state_dict(saved["network_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            controller.load_state_dict(saved["controller_state"])
            curriculum.load_state_dict(saved["curriculum_state"])
            previous_descent_rate = curriculum.min_descent_rate
            previous_stage_limit = curriculum.max_stage_epochs
            curriculum.min_descent_rate = config.stage_descent_rate
            curriculum.max_stage_epochs = config.stage_max_epochs
            previous_stage, previous_stage_epochs = curriculum.stage, curriculum.stage_epochs
            overdue_advanced = curriculum.advance_if_overdue()
            state = saved["rank_states"][rank]
            for key, spec in state["context_specs"].items():
                step.register_context(key, **spec)
            pool = ActiveTrajectoryPool.from_state_dict(
                state["pool"],
                reset=factory.reset,
                advance=factory.advance,
                retire=factory.retire,
                workers=config.preparation_workers,
            )
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])  # noqa: NPY002 -- Preserve process RNG in checkpoints.
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"], device)
            report = saved["report"]
            for field, previous, current in (
                ("stage_descent_rate", previous_descent_rate, config.stage_descent_rate),
                ("stage_max_epochs", previous_stage_limit, config.stage_max_epochs),
            ):
                if previous != current:
                    report.setdefault("configuration_changes", []).append(
                        {
                            "field": field,
                            "previous": previous,
                            "current": current,
                            "effective_from_epoch": report["completed_epochs"] + 1,
                            "completed_updates": report["completed_updates"],
                            "source": "checkpoint_resume",
                        }
                    )
            if overdue_advanced:
                report.setdefault("curriculum_events", []).append(
                    {
                        "source": "checkpoint_resume",
                        "advance_reason": "max_stage_epochs",
                        "previous_stage": previous_stage,
                        "stage": curriculum.stage,
                        "previous_stage_epochs": previous_stage_epochs,
                        "effective_from_epoch": report["completed_epochs"] + 1,
                        "completed_updates": report["completed_updates"],
                    }
                )
            report["config"] = asdict(config)
            report["status"] = "running"
        else:
            counts = curriculum.available_counts
            pool = ActiveTrajectoryPool(
                capacity=config.pool_multiplier * config.batch_size,
                batch_size=config.batch_size,
                reset=factory.reset,
                advance=factory.advance,
                retire=factory.retire,
                iteration_counts=counts[0],
                physical_step_counts=counts[1],
                seed=config.seed + rank * 100000000,
                workers=config.preparation_workers,
            )
            report = {
                "format": "mixed_pool_v2",
                "config": asdict(config),
                "world_size": world_size,
                "parameter_count": sum(p.numel() for p in network.parameters()),
                "completed_updates": 0,
                "completed_epochs": 0,
                "epochs": [],
                "updates": [],
                "best_selection": None,
                "status": "running",
            }
        best_metric = (report.get("best_selection") or {}).get("metric")
        module = (
            DistributedDataParallel(step, device_ids=[0] if device.type == "cuda" else None, broadcast_buffers=False)
            if world_size > 1
            else step
        )
        if rank == 0:
            (output / "checkpoints").mkdir(parents=True, exist_ok=True)
            write_progress(
                output,
                report,
                phase="initializing",
                epoch=report["completed_epochs"] + 1,
                available_K=curriculum.available_counts[0],
                available_H=curriculum.available_counts[1],
            )

        def checkpoint(name):
            checkpoint_error = None
            try:
                pool_state = pool.state_dict()
            except Exception as caught:
                checkpoint_error = repr(caught)
            failures = _all_ranks_ok(checkpoint_error, device, world_size)
            if failures:
                raise RuntimeError(f"checkpoint preparation failed: {failures}")
            local = {
                "pool": pool_state,
                "context_specs": step.context_specs,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),  # noqa: NPY002 -- Preserve process RNG in checkpoints.
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "parameter_sha256": hashlib.sha256(
                    b"".join(p.detach().cpu().numpy().tobytes() for p in network.parameters())
                ).hexdigest(),
            }
            states = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(states, local)
            else:
                states[0] = local
            if rank == 0:
                _atomic_torch(
                    output / "checkpoints" / name,
                    {
                        "format": "mixed_pool_v2",
                        "config": asdict(config),
                        "world_size": world_size,
                        "network_state": _cpu(network.state_dict()),
                        "optimizer_state": _cpu(optimizer.state_dict()),
                        "controller_state": controller.state_dict(),
                        "curriculum_state": curriculum.state_dict(),
                        "rank_states": states,
                        "report": report,
                    },
                )

        if not saved:
            checkpoint("initial.pt")
        if rank == 0:
            _write_report(output, report)
        for epoch in range(report["completed_epochs"] + 1, config.max_epochs + 1):
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            counts = curriculum.available_counts
            pool.set_available_counts(*counts)
            if rank == 0:
                write_progress(
                    output, report, phase="training", epoch=epoch, available_K=counts[0], available_H=counts[1]
                )
            totals = Counter()
            budgets, ages, steps, modes, materials, perturbations = Counter(), Counter(), Counter(), Counter(), [], []
            timings = Counter()
            step_size_min, step_size_max = math.inf, -math.inf
            for _ in range(config.queries_per_epoch // (config.batch_size * world_size)):
                began = time.perf_counter()
                error, records = None, None
                try:
                    records = pool.take_batch()
                    batch = _batch(records, device, cell_count=cell_count)
                    with torch.no_grad():
                        previous = step.energy(
                            batch["candidate"],
                            batch["inertial_prediction"],
                            batch["context_ids"],
                            previous_positions=batch["physical_positions"],
                        ).total
                        floor = step.energy_floor(batch["context_ids"])
                    if not torch.isfinite(previous).all():
                        raise ValueError("nonfinite input energy")
                except Exception as caught:
                    error = repr(caught)
                failures = _all_ranks_ok(error, device, world_size)
                if failures:
                    raise RuntimeError(f"mixed input preparation failed: {failures}")
                timings["prepare_wait_and_transfer_seconds"] += time.perf_counter() - began
                optimizer.zero_grad(set_to_none=True)
                began = time.perf_counter()
                error = None
                try:
                    with torch.autocast(device_type=device.type, enabled=False):
                        result = _checked_forward(module, step, batch)
                        losses = local_objective(
                            result.loss.total, previous, floor, increase_weight=config.energy_increase_weight
                        )
                        loss = losses.mean()
                except Exception as caught:
                    error = repr(caught)
                failures = _all_ranks_ok(error, device, world_size)
                if failures:
                    raise RuntimeError(f"learned proposal failed: {failures}")
                timings["forward_seconds"] += time.perf_counter() - began
                began = time.perf_counter()
                loss.backward()
                gradient_error = (
                    None
                    if all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
                    else "nonfinite gradient"
                )
                failures = _all_ranks_ok(gradient_error, device, world_size)
                if failures:
                    raise RuntimeError(f"backward failed: {failures}")
                optimizer.step()
                timings["backward_and_adam_seconds"] += time.perf_counter() - began
                after = result.loss.total.detach()
                residual = result.force_residual_norm.detach()
                step_size = result.step_size.detach()
                ties = (
                    result.tie_mask.sum().to(after.dtype) if result.tie_mask is not None else torch.zeros_like(after[0])
                )
                sums = torch.stack(
                    (losses.detach().sum(), previous.sum(), after.sum(), residual.sum(), step_size.sum(), ties)
                ).double()
                extremes = torch.stack((-step_size.min(), step_size.max())).double()
                if world_size > 1:
                    dist.all_reduce(sums)
                    dist.all_reduce(extremes, op=dist.ReduceOp.MAX)
                sums = sums.cpu().tolist()
                extremes = extremes.cpu().tolist()
                query_count = config.batch_size * world_size
                step_size_min = min(step_size_min, -extremes[0])
                step_size_max = max(step_size_max, extremes[1])
                report["completed_updates"] += 1
                report["updates"].append(
                    {
                        "update": report["completed_updates"],
                        "epoch": epoch,
                        "loss": sums[0] / query_count,
                        "before_joule": sums[1] / query_count,
                        "after_joule": sums[2] / query_count,
                        "mean_force_residual_n": sums[3] / query_count,
                        "step_size_mean": sums[4] / (query_count * cell_count),
                        "step_size_min": -extremes[0],
                        "step_size_max": extremes[1],
                        "tie_cell_count": int(round(sums[5])),
                    }
                )
                totals.update(
                    loss=sums[0],
                    query_count=query_count,
                    force_residual_sum=sums[3],
                    step_size_sum=sums[4],
                    tie_cell_count=int(round(sums[5])),
                )
                payloads = [record.payload for record in records]
                store_history(payloads, result)
                for i, record in enumerate(records):
                    budgets[f"{record.iteration_budget}/{record.step_budget}"] += 1
                    ages[str(record.inner_iteration)] += 1
                    steps[str(record.physical_step)] += 1
                    modes[record.payload["candidate_mode"]] += 1
                    materials.append(
                        {
                            "damping": 0.0,
                            **record.payload["context_spec"],
                            **record.payload["metadata"]["material_parameters"],
                        }
                    )
                    perturbations.append(record.payload["metadata"]["perturbation_scale"])
                    record.payload["candidate"] = result.positions[i].detach()
                pool.finish_batch(records)
                if rank == 0 and report["completed_updates"] % 8 == 0:
                    write_progress(
                        output, report, phase="training", epoch=epoch, available_K=counts[0], available_H=counts[1]
                    )
                del result, loss, losses, records, batch, previous, floor, after, residual, step_size, payloads
            if rank == 0:
                write_progress(
                    output, report, phase="validation", epoch=epoch, available_K=counts[0], available_H=counts[1]
                )
            validation = _validate(step, validation_factory, config, device, rank, world_size)
            need_full = epoch % config.validation_full_interval == 0 or curriculum.needs_full_horizon(validation)
            full = (
                validate_full_horizon(
                    step,
                    validation_factory,
                    config,
                    device,
                    rank,
                    world_size,
                    iterations=max(counts[0]),
                    physical_steps=max(counts[1]),
                )
                if need_full
                else None
            )
            curriculum_decision = curriculum.observe(validation, full_horizon=full)
            allow_early_stop = _allow_early_stop(config, curriculum)
            decision = controller.observe(epoch, validation, allow_early_stop=allow_early_stop)
            for group in optimizer.param_groups:
                group["lr"] = decision["learning_rate"]
            diagnostics = {
                "rank": rank,
                "budgets": dict(budgets),
                "inner_ages": dict(ages),
                "physical_ages": dict(steps),
                "candidate_modes": dict(modes),
                "pool": dict(pool.stats),
                "timings": dict(timings),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "material_histograms": {},
                "perturbation_histogram": np.histogram(perturbations, bins=np.linspace(0, 1, 11))[0].tolist(),
            }
            for name, bounds, logarithmic in (
                ("youngs_modulus", config.youngs_modulus_range, True),
                ("poissons_ratio", config.poissons_ratio_range, False),
                ("density", config.density_range, True),
                ("damping", config.damping_range, True),
            ):
                # Constant configured materials still need nonzero histogram bins.
                if bounds[0] == bounds[1]:
                    edges = np.linspace(bounds[0] - 0.5, bounds[1] + 0.5, 11)
                else:
                    edges = np.geomspace(*bounds, 11) if logarithmic else np.linspace(*bounds, 11)
                diagnostics["material_histograms"][name] = {
                    "edges": edges.tolist(),
                    "counts": np.histogram([m.get(name, 0.0) for m in materials], bins=edges)[0].tolist(),
                }
            rank_diagnostics = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(rank_diagnostics, diagnostics)
            else:
                rank_diagnostics[0] = diagnostics
            candidate_modes = Counter()
            for part in rank_diagnostics:
                candidate_modes.update(part["candidate_modes"])
            metric = _selection_metric(validation)
            is_best = metric is not None and (best_metric is None or metric < best_metric)
            if is_best:
                best_metric = metric
                report["best_selection"] = {
                    "epoch": epoch,
                    "metric": metric,
                    "aggregation": validation["selection"].get("aggregation"),
                    "completed_updates": report["completed_updates"],
                }
            row = {
                "epoch": epoch,
                "loss": totals["loss"] / totals["query_count"],
                "query_count": totals["query_count"],
                "mean_force_residual_n": totals["force_residual_sum"] / totals["query_count"],
                "step_size_mean": totals["step_size_sum"] / (totals["query_count"] * cell_count),
                "step_size_min": step_size_min,
                "step_size_max": step_size_max,
                "tie_cell_count": totals["tie_cell_count"],
                "candidate_modes": dict(candidate_modes),
                "seconds": time.perf_counter() - epoch_start,
                "available_K": counts[0],
                "available_H": counts[1],
                "rank_0_budgets": dict(budgets),
                "rank_0_inner_ages": dict(ages),
                "rank_0_physical_ages": dict(steps),
                "rank_0_material_ranges": {
                    k: [min(m[k] for m in materials), max(m[k] for m in materials)] for k in materials[0]
                },
                "rank_0_perturbation_scale": [min(perturbations), max(perturbations)],
                "rank_0_timings": dict(timings),
                "validation": validation,
                "full_horizon_validation": full,
                "learning_rate": decision["learning_rate"],
                "curriculum": curriculum_decision,
                "allow_early_stop": bool(allow_early_stop),
                "rank_0_pool": dict(pool.stats),
                "rank_0_peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "rank_diagnostics": rank_diagnostics,
            }
            report["epochs"].append(row)
            report.update(completed_epochs=epoch, status=decision["status"])
            checkpoint("latest.pt")
            if is_best:
                checkpoint("best_validation.pt")
            if epoch % config.checkpoint_interval == 0:
                checkpoint(f"epoch_{epoch:04d}.pt")
            if rank == 0:
                _write_report(output, report)
                if config.verbose:
                    selection = validation.get("selection") or {}
                    print(
                        f"epoch {epoch}: loss={row['loss']:.6g}, K={counts[0]}, H={counts[1]}, "
                        f"selection={selection.get('metric')} (eligible={selection.get('eligible')}), "
                        f"validation failures={validation['failed_count']}",
                        flush=True,
                    )
            if decision["stop"]:
                break
        checkpoint("final.pt")
        if rank == 0:
            write_progress(
                output,
                report,
                phase="complete",
                epoch=report["completed_epochs"],
                available_K=curriculum.available_counts[0],
                available_H=curriculum.available_counts[1],
            )
        return report
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "error": repr(error),
            "rank": rank,
            "config": asdict(config),
            "completed_updates": locals().get("report", {}).get("completed_updates", 0),
            "inputs": _cpu([vars(record) for record in (locals().get("records") or [])]),
            "active_records": _cpu([vars(record) for record in pool.records]) if pool is not None else [],
            "context_specs": step.context_specs,
            "network_state": _cpu(network.state_dict()),
            "optimizer_state": _cpu(optimizer.state_dict()),
        }
        _atomic_torch(output / f"failure_rank_{rank}.pt", failure)
        if rank == 0:
            (output / "failure.json").write_text(
                json.dumps(
                    {"error": repr(error), "completed_updates": locals().get("report", {}).get("completed_updates", 0)},
                    indent=2,
                )
            )
            if "report" in locals():
                report["status"] = "failed"
                _write_report(output, report)
                write_progress(output, report, phase="failed", epoch=locals().get("epoch", 0))
        raise
    finally:
        error_in_flight = sys.exc_info()[0] is not None
        try:
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    if not error_in_flight:
                        raise
        finally:
            try:
                step.close()
            finally:
                if owned_group:
                    dist.destroy_process_group()


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--config", type=Path, help="JSON overrides for MixedTrainConfig; resume starts from saved config"
    )
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    values = {}
    if args.resume:
        import torch

        values = asdict(
            MixedTrainConfig.from_checkpoint_config(
                torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
            )
        )
    if args.config:
        values.update(json.loads(args.config.read_text()))
    if args.max_epochs is not None:
        values["max_epochs"] = args.max_epochs
    if args.device is not None:
        values["device"] = args.device
    run_training(args.output, MixedTrainConfig(**values), resume=args.resume)


if __name__ == "__main__":
    _main()
