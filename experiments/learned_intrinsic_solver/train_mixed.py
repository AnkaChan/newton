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

Every trajectory draws a static contact scene (``contact_scene``): an optional
ground plane and artificial static points with per-scene stiffness, damping
and friction. The scene is registered with the physical context, stored in the
payload as ``contact_partners`` and its frozen per-step pair list is collated
into every batch so the contact energy, the contact conditioning channels and
the network's contact tokens follow the physics (schema 4).

Two training regimes share this loop. The ``pool`` regime draws an open-ended
stream of trajectories with curriculum-capped budgets and counts an epoch as
``queries_per_epoch`` global queries. The ``fixed_states`` regime reuses the
same ``state_count`` training initial states every epoch, samples per state
and epoch a budget ``K x H <= budget_cap`` under a fixed growth timetable
(:func:`sample_epoch_jobs`), balances the states over ranks and pads every
rank to a common number of full-size updates (:func:`assign_jobs`).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

__all__ = [
    "PERTURBED_CANDIDATE_PROBABILITY",
    "JobAssignment",
    "MixedTrainConfig",
    "assign_jobs",
    "fixed_state_stage",
    "local_objective",
    "run_training",
    "sample_epoch_jobs",
]

PERTURBED_CANDIDATE_PROBABILITY = 0.5
"""Probability that a new candidate adds smooth multiscale noise to the inertial prediction."""

_LEGACY_CONFIG_FIELDS = ("candidate_probabilities", "geometry_backtracking")

_VALIDATION_BUDGET_FIELDS = (
    "validation_count",
    "validation_iterations",
    "validation_interval",
    "validation_physical_steps",
    "validation_physical_iterations",
    "validation_full_count",
    "validation_full_interval",
    "validation_full_iterations",
)
"""Validation settings that may change on resume; none of them affects a training update."""

_SELECTION_BUDGET_FIELDS = ("validation_count", "validation_iterations")
"""Validation settings whose change makes earlier checkpoint-selection metrics incomparable."""

_ARCHITECTURE_FIELDS = (
    "cell_counts",
    "cell_size",
    "time_step",
    "gravity",
    "hidden_dim",
    "edge_hidden_dim",
    "num_heads",
    "hops",
    "query_chunk_size",
    "max_step_size",
    "feature_schema_version",
    "edge_network",
    "contact",
)
"""Fields a weights-only initialization must share with its checkpoint; every other field may differ."""

_REGIME_DEFAULTS = {
    "regime": "pool",
    "state_count": 2048,
    "budget_cap": 1024,
    "growth_stage_epochs": 2,
    "growth_stages": ((1, 8), (2, 16), (4, 32), (8, 64), (16, 128), (32, 128)),
}
"""Fixed-state regime settings absent from checkpoints written before the regime existed."""


@dataclass(frozen=True)
class MixedTrainConfig:
    """Experimental V2 settings. Material and the contact scene are independently sampled per reset."""

    cell_counts: tuple[int, int, int] = (10, 10, 40)
    cell_size: float = 0.025
    time_step: float = 1 / 300
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    hidden_dim: int = 128
    edge_hidden_dim: int = 64
    num_heads: int = 4
    hops: tuple[int, ...] = (1,)
    query_chunk_size: int = 128
    checkpoint_chunks: bool = False
    """Recompute attention chunks during backpropagation to trade compute for memory; outputs unchanged."""
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
    regime: str = "pool"
    """``pool``: open-ended trajectory pool under the validation-gated curriculum; ``fixed_states``: fixed-state epochs.

    The fixed-state regime runs the same ``state_count`` training initial states
    (training seeds ``0 .. state_count - 1``) once per epoch with per-epoch
    sampled budgets ``K x H <= budget_cap`` under the ``growth_stages``
    timetable. ``queries_per_epoch``, ``iteration_counts``,
    ``physical_step_counts`` and the curriculum settings are unused there.
    """
    state_count: int = 2048
    """Fixed training initial states of every epoch in the fixed-state regime."""
    budget_cap: int = 1024
    """Largest ``K x H`` (queries per state and epoch) of the fixed-state regime."""
    growth_stage_epochs: int = 2
    """Epochs per growth stage of the fixed-state regime; the final stage persists."""
    growth_stages: tuple[tuple[int, int], ...] = ((1, 8), (2, 16), (4, 32), (8, 64), (16, 128), (32, 128))
    """Growth timetable ``(K_max, H_max)`` per stage: K_max a power of two <= 32, H_max <= 128, non-decreasing."""
    validation_count: int = 512
    validation_iterations: int = 100
    validation_interval: int = 1
    """Epoch interval of the cheap validation; the final epoch is always validated.

    Epochs between validations skip validation entirely: they record
    ``validation=None``, never select a best checkpoint and only count toward
    curriculum stage residence.
    """
    validation_physical_steps: int = 8
    validation_physical_iterations: int = 2
    validation_full_count: int = 16
    """Held-out seeds of the full-horizon check, disjoint from the cheap set."""
    validation_full_interval: int = 5
    """Epoch interval of the full-horizon check; curriculum advancement also triggers it.

    Interval-driven checks run only on validated epochs, that is at epochs
    divisible by both intervals; one that falls on a skipped epoch is not
    deferred. A check the curriculum needs before advancing runs on the next
    validated epoch.
    """
    validation_full_iterations: int | None = None
    """Cap on K of the full-horizon check, ``min(largest available K, cap)``; None uses the largest available K."""
    checkpoint_interval: int = 5
    early_stopping: bool = True
    """Allow plateau/stall stopping before ``max_epochs``; False runs to the epoch cap."""
    lr_schedule: str = "cosine"
    """``cosine``: decay from learning_rate to lr_final over max_epochs; ``constant``; ``plateau``: legacy halving controller."""
    lr_final: float = 2.5e-5
    """Learning rate the cosine schedule reaches at ``max_epochs``."""
    weight_decay: float = 1e-6
    """Decoupled (AdamW) weight decay; 0 reproduces plain Adam."""
    gradient_clip_norm: float | None = 1.0
    """Clip the global gradient norm to this value before each update; None disables clipping."""
    youngs_modulus_range: tuple[float, float] = (1e3, 1e6)
    poissons_ratio_range: tuple[float, float] = (0.2, 0.49)
    density_range: tuple[float, float] = (100.0, 10000.0)
    damping_range: tuple[float, float] = (10.0, 1000.0)
    """Independent log-uniform absolute VBD viscosity [Pa·s]."""
    edge_network: bool = True
    """Give every transformer block the zero-initialized state-dependent edge update (ablation A02)."""
    contact: bool = True
    """Sample a contact scene per trajectory, add its energy and feed contact tokens to the network."""
    contact_plane_probability: float = 0.8
    """Probability that a sampled scene contains the ground plane."""
    contact_plane_height_range: tuple[float, float] = (-0.15, -0.005)
    """Uniform plane height bounds relative to the rest y-minimum [m].

    Shallow enough that soft and moderately stiff beams reach the floor within
    the longest training horizon (128 steps, 0.43 s).
    """
    contact_max_points: int = 64
    """Largest static point count per scene; the count is uniform in ``{0, ..., max}``."""
    contact_point_radius_range: tuple[float, float] = (0.5, 2.0)
    """Uniform lateral radius bounds of the static points in units of ``cell_size``."""
    contact_kappa_range: tuple[float, float] = (0.1, 10.0)
    """Log-uniform bounds of the stiffness factor ``kappa = ke / (E h)``."""
    contact_beta_range: tuple[float, float] = (0.0, 1.0)
    """Uniform bounds of the damping factor ``beta = kd / (ke dt)``."""
    contact_mu_range: tuple[float, float] = (0.0, 1.0)
    """Uniform bounds of the friction coefficient."""
    contact_max_pairs: int = 4
    """Largest number of static-point pairs kept per surface sample."""
    contact_tokens_per_cell: int = 24
    """Contact token slots per cell of the network's contact encoder."""
    contact_friction_epsilon: float = 1e-2
    """IPC friction smoothing band as a fraction of the time step."""
    feature_schema_version: int = features.FEATURE_SCHEMA_VERSION
    """Only the contact-aware schema (4) is supported; legacy runs restart."""
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
            "contact_plane_height_range",
            "contact_point_radius_range",
            "contact_kappa_range",
            "contact_beta_range",
            "contact_mu_range",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "growth_stages", tuple(tuple(stage) for stage in self.growth_stages))
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
            "validation_interval",
            "validation_physical_steps",
            "validation_physical_iterations",
            "validation_full_count",
            "validation_full_interval",
            "checkpoint_interval",
            "cpu_threads",
            "preparation_workers",
            "stage_patience",
            "contact_tokens_per_cell",
            "state_count",
            "budget_cap",
            "growth_stage_epochs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.regime not in ("pool", "fixed_states"):
            raise ValueError("regime must be pool or fixed_states")
        if not self.growth_stages:
            raise ValueError("growth_stages must not be empty")
        previous = (1, 1)
        for stage in self.growth_stages:
            if len(stage) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in stage):
                raise ValueError("growth_stages must contain (K_max, H_max) pairs of positive integers")
            k_max, h_max = stage
            if k_max & (k_max - 1) or k_max > 32 or h_max > 128:
                raise ValueError("growth_stages are limited to K_max a power of two <= 32 and H_max <= 128")
            if k_max < previous[0] or h_max < previous[1]:
                raise ValueError("growth_stages must be non-decreasing in K_max and H_max")
            previous = stage
        for name in ("plateau_min_final_stage_epochs", "contact_max_points", "contact_max_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("edge_network", "contact", "checkpoint_chunks"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
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
        for name in (
            "cell_size",
            "time_step",
            "max_step_size",
            "learning_rate",
            "energy_floor_scale",
            "contact_friction_epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        probability = self.contact_plane_probability
        if isinstance(probability, bool) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("contact_plane_probability must lie in [0, 1]")
        if self.lr_schedule not in ("cosine", "constant", "plateau"):
            raise ValueError("lr_schedule must be cosine, constant or plateau")
        if (
            isinstance(self.lr_final, bool)
            or not math.isfinite(self.lr_final)
            or not 0 < self.lr_final <= self.learning_rate
        ):
            raise ValueError("lr_final must be finite, positive and at most learning_rate")
        if isinstance(self.weight_decay, bool) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.gradient_clip_norm is not None and (
            isinstance(self.gradient_clip_norm, bool)
            or not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be None or finite and positive")
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
        if self.validation_full_iterations is not None and (
            isinstance(self.validation_full_iterations, bool)
            or not isinstance(self.validation_full_iterations, int)
            or self.validation_full_iterations < 1
        ):
            raise ValueError("validation_full_iterations must be None or a positive integer")
        if len(self.gravity) != 3 or not all(math.isfinite(v) for v in self.gravity):
            raise ValueError("gravity must be a finite three-vector")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("strength_range", "velocity_dt_range", "perturbation_scale_range"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 <= bounds[0] <= bounds[1]:
                raise ValueError(f"invalid {name}")
        # Plane heights may be negative; radii and stiffness factors must be positive; beta and mu nonnegative.
        for name, lowest in (
            ("contact_plane_height_range", -math.inf),
            ("contact_point_radius_range", 0.0),
            ("contact_kappa_range", 0.0),
            ("contact_beta_range", 0.0),
            ("contact_mu_range", 0.0),
        ):
            bounds = getattr(self, name)
            strict = name in ("contact_point_radius_range", "contact_kappa_range")
            if (
                len(bounds) != 2
                or any(isinstance(v, bool) or not math.isfinite(v) for v in bounds)
                or not bounds[0] <= bounds[1]
                or bounds[0] < lowest
                or (strict and bounds[0] <= lowest)
            ):
                raise ValueError(f"invalid {name}")
        self.material_ranges()
        if (
            isinstance(self.feature_schema_version, bool)
            or self.feature_schema_version != features.FEATURE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"feature_schema_version must be {features.FEATURE_SCHEMA_VERSION}: the contact-aware revised "
                "nine-value schema; legacy checkpoints require fresh initialization"
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
        # Checkpoints written before the schedule option used the plateau controller.
        values.setdefault("lr_schedule", "plateau")
        values.setdefault("lr_final", 1e-6)
        # Checkpoints written before AdamW and clipping used plain Adam without clipping.
        values.setdefault("weight_decay", 0.0)
        values.setdefault("gradient_clip_norm", None)
        # Checkpoints written before the memory knob existed trained without chunk checkpointing.
        values.setdefault("checkpoint_chunks", False)
        # Checkpoints written before the validation budget knobs validated every epoch at the largest K.
        values.setdefault("validation_interval", 1)
        values.setdefault("validation_full_iterations", None)
        # Checkpoints written before the fixed-state regime trained with the open-ended pool.
        for name, default in _REGIME_DEFAULTS.items():
            values.setdefault(name, default)
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

    def __init__(self, step, rest, config, *, rank, validation=False, unique_contexts=False):
        self.step, self.rest, self.config = step, rest, config
        self.prefix = f"{'validation' if validation else 'train'}-{rank}"
        self.master_seed = config.seed + (1000000007 if validation else 0)
        self.seed_parity = int(validation)
        # Preparation workers use CPU topology without synchronizing CUDA.
        self.fixed_indices = step.fixed_indices.detach().cpu().clone()
        self.cell_count = len(step.cell_corner_indices)
        # Job lists may run one seed twice at once (filler jobs), so each reset needs its own
        # context key; ``next`` on a count is atomic under the GIL for the worker threads.
        self._context_ids = itertools.count() if unique_contexts else None

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
        contact = self._contact_partners(seed, initial.material.youngs_modulus)
        key = f"{self.prefix}-{seed}"
        if self._context_ids is not None:
            key = f"{key}-{next(self._context_ids)}"
        specification = asdict(initial.material)
        self.step.register_context(key, **specification, contact=contact)
        try:
            payload = self.step.prepare(key, torch.from_numpy(initial.positions), torch.from_numpy(initial.velocities))
            payload.update(
                context_spec=specification,
                metadata={
                    **initial.metadata,
                    "contact": _contact_metadata(contact, initial.material.youngs_modulus, c),
                },
                contact_partners=contact.to_dict(),
                seed=seed,
                physical_age=0,
            )
            payload.update(empty_history(self.cell_count))
            return self._candidate(payload)
        except BaseException:
            self.step.discard_context(key)
            raise

    def _contact_partners(self, seed, youngs_modulus):
        """Draw the trajectory's static contact scene; contact-free partners when contact is disabled.

        The scene stream is ``[master_seed, seed, 2203]`` (see
        :func:`.contact_scene.sample_contact_partners`), so training and
        validation factories draw different scenes for the same seed.
        """
        from .contact_scene import (  # noqa: PLC0415 -- Optional training boundary.
            ContactPartners,
            sample_contact_partners,
        )

        c = self.config
        if not c.contact:
            return ContactPartners.contact_free()
        return sample_contact_partners(
            self.rest,
            master_seed=self.master_seed,
            seed=seed,
            youngs_modulus=youngs_modulus,
            cell_size=c.cell_size,
            time_step=c.time_step,
            plane_probability=c.contact_plane_probability,
            plane_height_range=c.contact_plane_height_range,
            max_points=c.contact_max_points,
            point_radius_range=c.contact_point_radius_range,
            kappa_range=c.contact_kappa_range,
            beta_range=c.contact_beta_range,
            mu_range=c.contact_mu_range,
        )

    def advance(self, payload):
        from .history import carry_history  # noqa: PLC0415 -- Optional training boundary.
        from .train_epochs import _cpu  # noqa: PLC0415 -- Optional training boundary.

        payload = _cpu(payload)
        prepared = self.step.advance(payload)
        for key in ("context_spec", "metadata", "seed", "contact_partners"):
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


def _contact_metadata(partners, youngs_modulus, config):
    """Summarize one scene for the payload metadata: presence, count and its dimensionless coefficients."""
    ke, kd = float(partners.ke), float(partners.kd)
    return {
        "plane_present": bool(partners.plane_present),
        "point_count": int(partners.point_count),
        "ke": ke,
        "kd": kd,
        "mu": float(partners.mu),
        "kappa": ke / (youngs_modulus * config.cell_size),
        "beta": kd / (ke * config.time_step) if ke > 0 else 0.0,
    }


def _saved_contact_partners(pool_state):
    """Rebuild every checkpointed trajectory's :class:`.ContactPartners`, keyed by context id.

    The pool drains preparation before it is checkpointed, so each record
    carries the payload whose ``contact_partners`` was written at reset.
    """
    from .contact_scene import ContactPartners  # noqa: PLC0415 -- Optional training boundary.

    partners = {}
    for record in pool_state.get("records", []):
        payload = record.get("payload") or {}
        if "context_id" in payload and "contact_partners" in payload:
            partners[payload["context_id"]] = ContactPartners.from_dict(payload["contact_partners"])
    return partners


_CONTACT_PAYLOAD_KEYS = (
    ("sample_index", "contact_sample_index", ()),
    ("kind", "contact_kind", ()),
    ("partner_point", "contact_partner_point", (3,)),
    ("partner_normal", "contact_partner_normal", (3,)),
    ("partner_radius", "contact_partner_radius", ()),
)
"""Batch entry, payload key and trailing shape of every frozen contact pair tensor."""


def _batch_contact(payloads, device):
    """Collate the frozen contact pairs of a batch, zero-padded to the largest Q with a validity mask.

    Returns None when no payload carries pair tensors (synthetic or legacy
    payloads). A payload without pair tensors among others contributes zero
    pairs; ``Q = 0`` for every payload yields ``[B, 0]`` entries, which the
    step treats as no contact.
    """
    import torch

    counts = [
        int(payload["contact_sample_index"].shape[0]) if "contact_sample_index" in payload else None
        for payload in payloads
    ]
    if all(count is None for count in counts):
        return None
    counts = [count or 0 for count in counts]
    width = max(counts)
    contact = {}
    for name, key, tail in _CONTACT_PAYLOAD_KEYS:
        dtype = torch.int64 if name in ("sample_index", "kind") else torch.float32
        padded = torch.zeros((len(payloads), width, *tail), dtype=dtype)
        for row, (payload, count) in enumerate(zip(payloads, counts, strict=True)):
            if count:
                padded[row, :count] = payload[key].detach().to(dtype)
        contact[name] = padded.to(device)
    contact["mask"] = (torch.arange(width)[None, :] < torch.tensor(counts)[:, None]).to(device)
    return contact


def _batch(records, device, *, cell_count=None):
    """Collate detached payload tensors, the stacked optimizer history and the contact pairs on ``device``.

    ``cell_count`` validates stored history blocks; when omitted it is read
    from the first stored block, and payloads without any history entries
    yield ``history=None`` (no history for the whole batch). ``contact`` is the
    padded pair batch of :func:`_batch_contact` or None.
    """
    import torch

    from .history import HISTORY_KEYS, batch_history  # noqa: PLC0415 -- Optional training boundary.

    payloads = [getattr(record, "payload", record) for record in records]
    values = {
        name: torch.stack([p[name].detach().to(device) for p in payloads])
        for name in ("candidate", "inertial_prediction", "fixed_positions", "physical_positions")
    }
    values["context_ids"] = tuple(p["context_id"] for p in payloads)
    values["contact"] = _batch_contact(payloads, device)
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
        contact=batch.get("contact"),
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
    """Return the finite selection metric when the validation is eligible, else None (also for no validation)."""
    selection = (validation or {}).get("selection") or {}
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


def _full_horizon_iterations(config, available_iterations):
    """Return K of the full-horizon check: the largest available count, capped by ``validation_full_iterations``."""
    largest = max(available_iterations)
    return largest if config.validation_full_iterations is None else min(largest, config.validation_full_iterations)


def _scheduled_learning_rate(config, epoch, plateau_rate):
    """Return the learning rate for the epoch after ``epoch`` under the configured schedule.

    ``cosine`` follows ``lr_final + (learning_rate - lr_final) * (1 + cos(pi * epoch / max_epochs)) / 2``
    with ``epoch`` the number of completed epochs, so it starts at ``learning_rate`` and
    reaches ``lr_final`` exactly at the epoch cap; ``constant`` keeps ``learning_rate``;
    ``plateau`` returns the legacy controller's rate. Raising ``max_epochs`` on resume
    stretches the cosine over the new cap from the current epoch onward.
    """
    if config.lr_schedule == "plateau":
        return float(plateau_rate)
    if config.lr_schedule == "constant":
        return float(config.learning_rate)
    fraction = min(max(epoch / config.max_epochs, 0.0), 1.0)
    return float(config.lr_final + (config.learning_rate - config.lr_final) * (1 + math.cos(math.pi * fraction)) / 2)


@dataclass(frozen=True)
class JobAssignment:
    """Per-rank job lists of one fixed-state epoch with the common update count."""

    rank_jobs: tuple[tuple[tuple[int, int, int], ...], ...]
    """Jobs ``(seed, K, H)`` of every rank: the sampled jobs in execution order, then the filler jobs."""
    updates: int
    """Full-size updates every rank performs in the epoch."""
    queries: int
    """Queries of the sampled jobs over all ranks, without fillers."""
    filler_queries: tuple[int, ...]
    """Padding queries per rank, each a ``(seed, 1, 1)`` job on one of the rank's own seeds."""


def fixed_state_stage(config, epoch):
    """Return ``(stage, K_max, H_max)`` of ``epoch`` under the fixed-state growth timetable.

    Stages advance every ``growth_stage_epochs`` epochs from epoch 1; the last
    stage persists.
    """
    stage = min(len(config.growth_stages) - 1, (epoch - 1) // config.growth_stage_epochs)
    k_max, h_max = config.growth_stages[stage]
    return stage, k_max, h_max


def _powers_of_two(limit):
    """Return the powers of two no greater than ``limit`` in increasing order."""
    return tuple(1 << i for i in range(limit.bit_length()) if 1 << i <= limit)


def _fixed_state_counts(config, stage):
    """Return the available ``(K, H)`` counts of ``stage`` for the validation checks.

    K are the powers of two up to ``min(K_max, budget_cap)`` and H is the single
    stage horizon ``H_max``, so the full-horizon check runs the largest K
    (capped by ``validation_full_iterations``) over ``H_max`` steps.
    """
    k_max, h_max = config.growth_stages[stage]
    return _powers_of_two(min(k_max, config.budget_cap)), (h_max,)


def sample_epoch_jobs(config, epoch, master_seed):
    """Draw the fixed-state jobs ``(seed, K, H)`` of one epoch, one per training state.

    With ``(K_max, H_max)`` the growth stage of ``epoch``, ``H`` is uniform
    over ``1 .. H_max`` and ``K`` uniform over the powers of two ``<= K_max``
    whose ``K x H <= budget_cap``; ``K = 1`` is always offered. The stream is
    ``SeedSequence([master_seed, epoch, 7331])``, so the draw is identical on
    every rank and after a resume.

    Args:
        config: Training configuration with the fixed-state fields.
        epoch: One-based epoch number.
        master_seed: Campaign seed, normally ``config.seed``.

    Returns:
        ``[(seed, K, H)]`` for ``seed`` in ``range(config.state_count)``.
    """
    _, k_max, h_max = fixed_state_stage(config, epoch)
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, epoch, 7331]))
    powers = _powers_of_two(k_max)
    jobs = []
    for seed in range(config.state_count):
        steps = int(rng.integers(1, h_max + 1))
        options = [k for k in powers if k * steps <= config.budget_cap] or [1]
        jobs.append((seed, int(rng.choice(options)), steps))
    return jobs


def assign_jobs(jobs, world_size, batch_size, *, shuffle_seed=None):
    """Balance jobs over ranks by ``K x H`` and pad every rank to the same update count.

    Longest-processing-time greedy: jobs sorted by decreasing ``K x H`` (ties
    by seed) go to the least loaded rank (ties by rank index). The update
    count ``U`` is the smallest number of full batches that holds the heaviest
    rank and gives the longest job one query per update (a trajectory receives
    at most one query per batch). Each rank's sampled jobs are then listed in
    execution order: the LPT order (decreasing ``K x H``) when ``shuffle_seed``
    is None, otherwise the permutation drawn from
    ``SeedSequence([*shuffle_seed, rank, 7332])``. The trainer shuffles so the
    pool, which starts jobs in list order, sees a stationary mixture of budgets
    over the epoch instead of drifting from the longest budgets to the shortest;
    the pool completes every batch whatever the order. Each rank finally
    appends filler jobs ``(seed, 1, 1)``, cycling through its own seeds in
    increasing order, until it holds exactly ``U * batch_size`` queries, so
    every rank performs ``U`` full-size updates. The result depends only on
    the arguments and is therefore identical on every rank.

    Args:
        jobs: ``(seed, K, H)`` triples of one epoch.
        world_size: Number of ranks.
        batch_size: Queries per update on every rank.
        shuffle_seed: Integers seeding the per-rank execution order, normally
            ``(master_seed, epoch)``; None keeps the LPT order.

    Raises:
        ValueError: A count is not a positive integer or a rank receives no job.
    """
    for name, value in (("world_size", world_size), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    ordered = sorted(jobs, key=lambda job: (-(job[1] * job[2]), job[0]))
    loads = [0] * world_size
    rank_jobs = [[] for _ in range(world_size)]
    for job in ordered:
        rank = min(range(world_size), key=loads.__getitem__)
        rank_jobs[rank].append(job)
        loads[rank] += job[1] * job[2]
    if any(not queue for queue in rank_jobs):
        raise ValueError("every rank needs at least one job; increase state_count")
    longest = ordered[0][1] * ordered[0][2]
    updates = max(-(-max(loads) // batch_size), longest)
    fillers = []
    for rank, queue in enumerate(rank_jobs):
        if shuffle_seed is not None:
            rng = np.random.default_rng(np.random.SeedSequence([*shuffle_seed, rank, 7332]))
            queue[:] = [queue[index] for index in rng.permutation(len(queue))]
        seeds = sorted({job[0] for job in queue})
        count = updates * batch_size - loads[rank]
        queue.extend((seeds[index % len(seeds)], 1, 1) for index in range(count))
        fillers.append(count)
    return JobAssignment(tuple(tuple(queue) for queue in rank_jobs), updates, sum(loads), tuple(fillers))


def _parameter_shape_mismatches(saved_state, current_state):
    """Return the parameter names one state dict lacks or whose shapes differ between the two.

    A weights-only start compares the checkpoint's ``network_state`` with the
    freshly built network before loading, so a feature change that kept the
    architecture fields (for example a conditioning width without a schema
    bump) is rejected as an architecture mismatch instead of a raw size error.
    """
    return [
        name
        for name in sorted(set(saved_state) | set(current_state))
        if name not in saved_state
        or name not in current_state
        or tuple(saved_state[name].shape) != tuple(current_state[name].shape)
    ]


def _allow_early_stop_fixed(config, epoch):
    """Return whether plateau stopping may be considered after ``epoch`` in the fixed-state regime.

    The analogue of :func:`_allow_early_stop`: stopping is permitted only when
    enabled and at least ``plateau_min_final_stage_epochs`` epochs, ``epoch``
    included, ran in the final growth stage.
    """
    final_stage_start = (len(config.growth_stages) - 1) * config.growth_stage_epochs
    return bool(config.early_stopping and epoch - final_stage_start >= config.plateau_min_final_stage_epochs)


def run_training(
    output: Path, config: MixedTrainConfig, *, resume: Path | None = None, resume_weights_only: bool = False
):
    """Run an explicitly requested V2 campaign or a bounded verification run.

    ``config.regime`` selects the open-ended trajectory pool (``pool``) or the
    fixed-state epoch regime (``fixed_states``). In the latter every epoch
    draws :func:`sample_epoch_jobs`, distributes them with :func:`assign_jobs`
    and performs exactly ``U`` full-size updates on every rank; the growth
    stage supplies the available K/H of the full-horizon check, the curriculum
    is unused (epoch rows carry ``curriculum=None`` and a ``regime`` block) and
    checkpoints hold no pool state because every epoch restarts its job list.

    With ``resume_weights_only`` the checkpoint only initializes the network
    and the AdamW state of a fresh run: the epoch counter, report, curriculum,
    controller and pool are new, only the architecture fields
    (``_ARCHITECTURE_FIELDS``) must match and the origin is recorded under
    ``report["initialized_from"]`` together with every differing field. The
    learning rate follows the new schedule from epoch 1.

    Checkpoints restore the same rank count, curriculum, pool queues, optimizer
    history and Adam sequence. The configured descent-rate gate, stage limit
    and validation budget may change on resume; the gate applies only to future
    validation and an overdue hard limit promotes one stage immediately. These
    changes record their epoch boundary. A changed ``validation_count`` or
    ``validation_iterations`` makes earlier selection metrics incomparable, so
    ``best_selection`` restarts from None, the previous record moves to
    ``best_selection_history`` and the plateau controller forgets its best
    metric and patience. Native factors are rebuilt; they are never
    serialized. Legacy checkpoints are rejected explicitly.

    Every ``validation_interval`` epochs, and on the final epoch, the cheap
    validation runs; on validated epochs divisible by
    ``validation_full_interval``, or when the curriculum could advance, the
    full-horizon check at the largest available H and the largest available K
    (capped by ``validation_full_iterations``) follows. Skipped epochs record
    ``validation=None``, count toward curriculum stage residence only, never
    select a checkpoint and are invisible to the plateau controller. The best
    checkpoint is selected by the validation ``selection`` metric (mean final
    free-corner force residual with survival required). Plateau stopping is
    considered only after ``plateau_min_final_stage_epochs`` epochs in the
    final curriculum stage.
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
    fixed_states = config.regime == "fixed_states"
    if not fixed_states and config.queries_per_epoch % (config.batch_size * world_size):
        raise ValueError("queries_per_epoch must be divisible by batch_size * world_size")
    if fixed_states and config.state_count < world_size:
        raise ValueError("state_count must be at least the rank count")
    if resume_weights_only and resume is None:
        raise ValueError("resume_weights_only requires a checkpoint")
    output = Path(output).resolve()
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    initialized_from = initial_weights = None
    if saved and resume_weights_only:
        if saved.get("format") != "mixed_pool_v2":
            raise ValueError("incompatible checkpoint format")
        saved_config = asdict(MixedTrainConfig.from_checkpoint_config(saved["config"]))
        current_config = asdict(config)
        mismatched = [name for name in _ARCHITECTURE_FIELDS if saved_config[name] != current_config[name]]
        if mismatched:
            raise ValueError(
                f"weights-only initialization requires the checkpoint architecture; differing: {mismatched}"
            )
        if (output / "report.json").exists() or (output / "checkpoints").exists():
            raise FileExistsError("weights-only initialization starts a fresh run; use a fresh output directory")
        initialized_from = {
            "checkpoint": str(Path(resume).resolve()),
            "sha256": None,  # Filled once the parameter shapes match the freshly built network.
            "completed_epochs": saved["report"]["completed_epochs"],
            "completed_updates": saved["report"]["completed_updates"],
            "best_selection": saved["report"].get("best_selection"),
            "config_differences": {
                name: {"checkpoint": saved_config.get(name), "current": value}
                for name, value in current_config.items()
                if saved_config.get(name) != value
            },
        }
        initial_weights, saved = saved, None
    if saved:
        if saved.get("format") != "mixed_pool_v2" or saved["world_size"] != world_size:
            raise ValueError("incompatible checkpoint format or rank count")
        allowed = {
            "max_epochs",
            "verbose",
            "early_stopping",
            "stage_descent_rate",
            "stage_max_epochs",
            "lr_schedule",
            "lr_final",
            "weight_decay",
            "gradient_clip_norm",
            "checkpoint_chunks",
            *_VALIDATION_BUDGET_FIELDS,
        }
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
        edge_network=config.edge_network,
        checkpoint_chunks=config.checkpoint_chunks,
        contact_tokens=config.contact,
    ).to(device)
    if initial_weights is not None:
        # The architecture fields do not cover every parameter shape (a feature change without a
        # schema bump keeps them equal), so compare the tensors themselves before touching the output.
        mismatched = _parameter_shape_mismatches(initial_weights["network_state"], network.state_dict())
        if mismatched:
            if owned_group:
                dist.destroy_process_group()
            raise ValueError(
                "weights-only initialization requires the checkpoint architecture; "
                f"parameters missing or differing in shape: {mismatched}"
            )
        initialized_from["sha256"] = hashlib.sha256(Path(resume).read_bytes()).hexdigest()
    step = MixedHexSolverStep(
        rest,
        fixed,
        network=network,
        time_step=config.time_step,
        gravity=config.gravity,
        energy_floor_scale=config.energy_floor_scale,
        contact_max_pairs=config.contact_max_pairs,
        contact_tokens_per_cell=config.contact_tokens_per_cell,
        contact_friction_epsilon=config.contact_friction_epsilon,
    ).to(device)
    cell_count = len(step.cell_corner_indices)
    factory = _TrajectoryFactory(step, rest, config, rank=rank, unique_contexts=fixed_states)
    validation_factory = _TrajectoryFactory(step, rest, config, rank=rank, validation=True)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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

    def available_counts(epoch):
        """Return the (K, H) counts new resets or the growth stage of ``epoch`` offer."""
        if fixed_states:
            return _fixed_state_counts(config, fixed_state_stage(config, epoch)[0])
        return curriculum.available_counts

    def stage_regime(epoch):
        """Return the fixed-state regime block of ``epoch`` before its jobs are drawn, else None.

        Progress heartbeats carry this block in every phase; the epoch loop adds
        the query and update counts once the jobs are assigned.
        """
        if not fixed_states:
            return None
        stage, k_max, h_max = fixed_state_stage(config, epoch)
        return {"name": "fixed_states", "stage": stage, "k_max": k_max, "h_max": h_max}

    def fixed_state_pool():
        """Return an empty pool for job lists; ``set_jobs`` starts every epoch."""
        return ActiveTrajectoryPool(
            capacity=config.pool_multiplier * config.batch_size,
            batch_size=config.batch_size,
            reset=factory.reset,
            advance=factory.advance,
            retire=factory.retire,
            iteration_counts=(1,),
            physical_step_counts=(1,),
            seed=config.seed + rank * 100000000,
            workers=config.preparation_workers,
            initialize=False,
        )

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
            # Only a changed cap promotes here. An unchanged cap reached on a skipped epoch waits for
            # the next validated epoch and its full-horizon check, as the uninterrupted run would.
            # The fixed-state regime never consults the curriculum.
            overdue_advanced = (
                not fixed_states and previous_stage_limit != config.stage_max_epochs and curriculum.advance_if_overdue()
            )
            state = saved["rank_states"][rank]
            if fixed_states:
                # Every epoch restarts its job list, so the checkpoint holds no pool or contexts.
                pool = fixed_state_pool()
            else:
                partners = _saved_contact_partners(state["pool"])
                for key, spec in state["context_specs"].items():
                    if key not in partners:
                        raise ValueError(f"checkpoint context {key!r} has no stored contact partners")
                    step.register_context(key, **spec, contact=partners[key])
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
                *((field, saved_config[field], getattr(config, field)) for field in _VALIDATION_BUDGET_FIELDS),
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
            if any(saved_config[field] != getattr(config, field) for field in _SELECTION_BUDGET_FIELDS):
                # Metrics measured under a different validation budget are not comparable: the best
                # record and the controller's patience restart and best_validation.pt is rewritten at
                # the first eligible epoch.
                controller.reset_metric_history()
                report.setdefault("best_selection_history", []).append(
                    {
                        "reset_at_epoch": report["completed_epochs"],
                        "reason": "validation budget changed",
                        "record": report.get("best_selection"),
                    }
                )
                report["best_selection"] = None
                if rank == 0 and config.verbose:
                    print(
                        f"resume: validation budget changed after epoch {report['completed_epochs']}; "
                        f"best_selection reset (previous record: {report['best_selection_history'][-1]['record']})",
                        flush=True,
                    )
            report["config"] = asdict(config)
            report["status"] = "running"
        else:
            if fixed_states:
                pool = fixed_state_pool()
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
            if initial_weights is not None:
                # AdamW moments carry over; the schedule overwrites the group rates at epoch 1.
                network.load_state_dict(initial_weights["network_state"])
                optimizer.load_state_dict(initial_weights["optimizer_state"])
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
            if initialized_from is not None:
                report["initialized_from"] = initialized_from
        best_metric = (report.get("best_selection") or {}).get("metric")
        module = (
            DistributedDataParallel(step, device_ids=[0] if device.type == "cuda" else None, broadcast_buffers=False)
            if world_size > 1
            else step
        )
        regime = stage_regime(report["completed_epochs"] + 1)
        if rank == 0:
            (output / "checkpoints").mkdir(parents=True, exist_ok=True)
            initial_counts = available_counts(report["completed_epochs"] + 1)
            write_progress(
                output,
                report,
                phase="initializing",
                epoch=report["completed_epochs"] + 1,
                available_K=initial_counts[0],
                available_H=initial_counts[1],
                regime=regime,
            )

        def checkpoint(name):
            checkpoint_error = None
            try:
                pool_state = None if fixed_states else pool.state_dict()
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
            # The rate for this epoch follows the schedule of the CURRENT configuration, so a
            # resume with a changed schedule or epoch cap takes effect immediately and a resumed
            # run matches an uninterrupted one with the same configuration.
            for group in optimizer.param_groups:
                group["lr"] = _scheduled_learning_rate(config, epoch - 1, controller.learning_rate)
                group["weight_decay"] = config.weight_decay
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if fixed_states:
                counts = available_counts(epoch)
                assignment = assign_jobs(
                    sample_epoch_jobs(config, epoch, config.seed),
                    world_size,
                    config.batch_size,
                    shuffle_seed=(config.seed, epoch),
                )
                pool.set_jobs(assignment.rank_jobs[rank])
                updates = assignment.updates
                regime = {
                    **stage_regime(epoch),
                    "queries": assignment.queries,
                    "filler_queries": list(assignment.filler_queries),
                    "updates": updates,
                }
            else:
                counts = curriculum.available_counts
                pool.set_available_counts(*counts)
                updates = config.queries_per_epoch // (config.batch_size * world_size)
                regime = None
            if rank == 0:
                write_progress(
                    output,
                    report,
                    phase="training",
                    epoch=epoch,
                    available_K=counts[0],
                    available_H=counts[1],
                    regime=regime,
                )
            totals = Counter()
            budgets, ages, steps, modes, materials, perturbations = Counter(), Counter(), Counter(), Counter(), [], []
            timings = Counter()
            step_size_min, step_size_max = math.inf, -math.inf
            gradient_norm_max = 0.0
            contact_penetration_max = 0.0
            contact_scenes = {}
            contact_hits = {}
            update_count = 0
            for _ in range(updates):
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
                            contact=batch["contact"],
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
                # Gradients are identical on all ranks after the DDP all-reduce, so this is the
                # global norm before clipping; an infinite bound only measures it.
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        network.parameters(),
                        config.gradient_clip_norm if config.gradient_clip_norm is not None else math.inf,
                    )
                )
                optimizer.step()
                timings["backward_and_adam_seconds"] += time.perf_counter() - began
                after = result.loss.total.detach()
                residual = result.force_residual_norm.detach()
                step_size = result.step_size.detach()
                ties = (
                    result.tie_mask.sum().to(after.dtype) if result.tie_mask is not None else torch.zeros_like(after[0])
                )
                pair_counts = (
                    batch["contact"]["mask"].sum(dim=1)
                    if batch["contact"] is not None
                    else torch.zeros(len(records), dtype=torch.int64, device=after.device)
                )
                pairs = pair_counts.sum().to(after.dtype)
                pair_counts = pair_counts.tolist()
                penetration = (
                    result.contact_max_penetration.detach().max()
                    if result.contact_max_penetration is not None
                    else torch.zeros_like(after[0])
                )
                sums = torch.stack(
                    (losses.detach().sum(), previous.sum(), after.sum(), residual.sum(), step_size.sum(), ties, pairs)
                ).double()
                extremes = torch.stack((-step_size.min(), step_size.max(), penetration)).double()
                if world_size > 1:
                    dist.all_reduce(sums)
                    dist.all_reduce(extremes, op=dist.ReduceOp.MAX)
                sums = sums.cpu().tolist()
                extremes = extremes.cpu().tolist()
                query_count = config.batch_size * world_size
                step_size_min = min(step_size_min, -extremes[0])
                step_size_max = max(step_size_max, extremes[1])
                report["completed_updates"] += 1
                update_count += 1
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
                        "gradient_norm": gradient_norm,
                        "contact_max_penetration_r": extremes[2],
                        "contact_pair_mean": sums[6] / query_count,
                    }
                )
                gradient_norm_max = max(gradient_norm_max, gradient_norm)
                contact_penetration_max = max(contact_penetration_max, extremes[2])
                totals.update(
                    loss=sums[0],
                    gradient_norm_sum=gradient_norm,
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
                    scene = record.payload["metadata"].get("contact") or {}
                    contact_scenes[record.seed] = bool(scene.get("plane_present")) or scene.get("point_count", 0) > 0
                    # Realized contact: the trajectory carried at least one detected pair in some update.
                    contact_hits[record.seed] = contact_hits.get(record.seed, False) or pair_counts[i] > 0
                    record.payload["candidate"] = result.positions[i].detach()
                pool.finish_batch(records)
                if rank == 0 and report["completed_updates"] % 8 == 0:
                    write_progress(
                        output,
                        report,
                        phase="training",
                        epoch=epoch,
                        available_K=counts[0],
                        available_H=counts[1],
                        regime=regime,
                    )
                del result, loss, losses, records, batch, previous, floor, after, residual, step_size, payloads
            if fixed_states:
                # Coordinated like every other failure of the loop, so a pool invariant violation
                # fails all ranks together instead of stranding the others in the next collective.
                exhaustion_error = None if pool.exhausted else f"{pool.remaining_queries} queries unserved"
                failures = _all_ranks_ok(exhaustion_error, device, world_size)
                if failures:
                    raise RuntimeError(f"fixed-state epoch left queries unserved: {failures}")
            validated = epoch % config.validation_interval == 0 or epoch == config.max_epochs
            if validated:
                if rank == 0:
                    write_progress(
                        output,
                        report,
                        phase="validation",
                        epoch=epoch,
                        available_K=counts[0],
                        available_H=counts[1],
                        regime=regime,
                    )
                validation = _validate(step, validation_factory, config, device, rank, world_size)
                need_full = epoch % config.validation_full_interval == 0 or (
                    not fixed_states and curriculum.needs_full_horizon(validation)
                )
                full = (
                    validate_full_horizon(
                        step,
                        validation_factory,
                        config,
                        device,
                        rank,
                        world_size,
                        iterations=_full_horizon_iterations(config, counts[0]),
                        physical_steps=max(counts[1]),
                    )
                    if need_full
                    else None
                )
                if fixed_states:
                    curriculum_decision = None
                    allow_early_stop = _allow_early_stop_fixed(config, epoch)
                else:
                    curriculum_decision = curriculum.observe(validation, full_horizon=full)
                    allow_early_stop = _allow_early_stop(config, curriculum)
                decision = controller.observe(epoch, validation, allow_early_stop=allow_early_stop)
            else:
                # No validation work: the curriculum counts stage residence only, the controller
                # never sees this epoch and no full-horizon check runs; one the curriculum needs
                # before advancing runs on the next validated epoch.
                validation = full = None
                curriculum_decision = None if fixed_states else curriculum.skip()
                allow_early_stop = False
                decision = {"learning_rate": controller.learning_rate, "stop": False, "status": "running"}
            # Recorded as the rate the next epoch will use under the current schedule.
            decision["learning_rate"] = _scheduled_learning_rate(config, epoch, decision["learning_rate"])
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
                "gradient_norm_mean": totals["gradient_norm_sum"] / max(update_count, 1),
                "gradient_norm_max": gradient_norm_max,
                "contact_scene_fraction": sum(contact_scenes.values()) / len(contact_scenes) if contact_scenes else 0.0,
                "contact_realized_fraction": sum(contact_hits.values()) / len(contact_hits) if contact_hits else 0.0,
                "contact_max_penetration_r": contact_penetration_max,
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
            if regime is not None:
                row["regime"] = regime
            report["epochs"].append(row)
            report.update(completed_epochs=epoch, status=decision["status"])
            checkpoint("latest.pt")
            if is_best:
                checkpoint("best_validation.pt")
            if epoch % config.checkpoint_interval == 0:
                checkpoint(f"epoch_{epoch:04d}.pt")
            if rank == 0:
                _write_report(output, report)
                stage_text = (
                    f"stage {regime['stage']} (K<={regime['k_max']}, H<={regime['h_max']}, U={regime['updates']}), "
                    if regime is not None
                    else ""
                )
                if config.verbose and validation is None:
                    print(
                        f"epoch {epoch}: loss={row['loss']:.6g}, {stage_text}K={counts[0]}, H={counts[1]}, "
                        f"validation skipped (every {config.validation_interval} epochs)",
                        flush=True,
                    )
                elif config.verbose:
                    selection = validation.get("selection") or {}
                    print(
                        f"epoch {epoch}: loss={row['loss']:.6g}, {stage_text}K={counts[0]}, H={counts[1]}, "
                        f"selection={selection.get('metric')} (eligible={selection.get('eligible')}), "
                        f"validation failures={validation['failed_count']}",
                        flush=True,
                    )
            if decision["stop"]:
                break
        checkpoint("final.pt")
        if rank == 0:
            final_counts = available_counts(max(report["completed_epochs"], 1))
            write_progress(
                output,
                report,
                phase="complete",
                epoch=report["completed_epochs"],
                available_K=final_counts[0],
                available_H=final_counts[1],
                regime=regime,
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
                write_progress(
                    output, report, phase="failed", epoch=locals().get("epoch", 0), regime=locals().get("regime")
                )
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
        "--resume-weights-only",
        action="store_true",
        help="initialize the network and AdamW state from --resume into a fresh run; "
        "--config supplies the whole configuration and only the architecture must match",
    )
    parser.add_argument(
        "--config", type=Path, help="JSON overrides for MixedTrainConfig; resume starts from saved config"
    )
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.resume_weights_only and not args.resume:
        parser.error("--resume-weights-only requires --resume")
    values = {}
    if args.resume and not args.resume_weights_only:
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
    run_training(
        args.output, MixedTrainConfig(**values), resume=args.resume, resume_weights_only=args.resume_weights_only
    )


if __name__ == "__main__":
    _main()
