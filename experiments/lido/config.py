# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Training configuration: the v4 JSON key set (design spec 7.2). Unknown keys are rejected."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # geometry and integration (SI)
    cell_counts: tuple = (10, 10, 40)
    cell_size: float = 0.025
    time_step: float = 1.0 / 300.0
    gravity: tuple = (0.0, -9.81, 0.0)
    gravity_magnitude_range: tuple = (2.0, 40.0)
    pins: str = "zmin_face"
    # network
    hidden_dim: int = 192
    edge_hidden_dim: int = 96
    num_heads: int = 6
    contact_hidden_dim: int = 64
    target_modes: int = 7
    max_step_size: float = 0.05
    feature_schema_version: int = 6
    edge_network: bool = True
    edge_module: str = "a02"  # "a02": edge encoder + A02 update (the trained checkpoints); "pair": one MLP on the pair
    compile_network: bool = True  # torch.compile the cell-graph layer on CUDA (implementation only)
    # regime
    regime: str = "fixed_states"
    batch_size: int = 16
    state_count: int = 2048
    budget_cap: int = 2048
    growth_stages: tuple = ((1, 8), (2, 16), (4, 32), (8, 64), (16, 128), (32, 128))
    growth_stage_epochs: int = 2
    max_epochs: int = 48
    iteration_counts: tuple = (1, 2, 4, 8, 16, 32)
    # optimisation
    learning_rate: float = 1e-4
    lr_schedule: str = "cosine"
    lr_final: float = 2.5e-5
    weight_decay: float = 1e-6
    gradient_clip_norm: float = 1.0
    energy_increase_weight: float = 1.0
    energy_floor_scale: float = 1.0
    early_stopping: bool = False
    # materials and augmentation
    youngs_modulus_range: tuple = (1e3, 1e6)
    poissons_ratio_range: tuple = (0.2, 0.49)
    density_range: tuple = (100.0, 1e4)
    damping_range: tuple = (10.0, 1000.0)
    strength_range: tuple = (0.02, 0.1)
    velocity_dt_range: tuple = (0.0, 0.1)
    perturbation_scale_range: tuple = (0.0, 1.0)
    candidate_noise_range: tuple = (0.01, 0.10)
    # contact
    contact: bool = True
    contact_plane_probability: float = 0.8
    contact_plane_height_range: tuple = (-0.15, -0.005)
    contact_max_points: int = 64
    contact_point_radius_range: tuple = (0.5, 2.0)
    contact_kappa_range: tuple = (10.0, 1000.0)
    contact_static_penetration_max: float = 0.5
    contact_beta_range: tuple = (0.0, 1.0)
    contact_mu_range: tuple = (0.0, 1.0)
    contact_max_pairs: int = 4
    contact_tokens_per_cell: int = 24
    contact_friction_epsilon: float = 0.01
    # v5 scenes (scenes_v5.py, design spec section 11): one multi-body scene per batch in a shared world frame
    scene_mode: str = "body"  # "body": one object per slot (jobs.py); "v5": one scene of free bodies per batch
    scene_cells: int = 64000  # bodies are added until the scene reaches this many cells
    body_sides: tuple = (3, 12)  # cells per axis, drawn independently per axis
    drift_speed_range: tuple = (0.1, 0.5)  # m/s scene-wide drift speed
    placement_height: float = 0.5  # m: a body's lowest point lies between 2 cells and this above the ground
    placement_gap_cells: tuple = (1, 3)  # cells between the bounding boxes of placed bodies
    scene_count: int = 64  # fixed scenes per epoch
    validation_scene_count: int = 8
    validation_full_scene_count: int = 2
    # validation and selection
    validation_count: int = 64
    validation_iterations: int = 8
    validation_interval: int = 1
    validation_full_count: int = 16
    validation_full_interval: int = 1
    validation_full_iterations: int = 8
    validation_full_steps: int = 128
    selection_source: str = "full_horizon"
    # run
    seed: int = 73
    device: str = "cuda"
    log_every: int = 50
    checkpoint_interval: int = 5
    verbose: bool = True
    # accepted legacy keys of the v4 JSON (ignored)
    hops: tuple = (1,)
    query_chunk_size: int = 128
    pool_multiplier: int = 4
    queries_per_epoch: int = 8192
    stage_epochs: int = 10
    stage_patience: int = 2
    stage_descent_rate: float = 0.8
    stage_max_epochs: int = 20
    physical_step_counts: tuple = (8, 16, 32, 64, 128)
    validation_physical_steps: int = 8
    validation_physical_iterations: int = 2
    cpu_threads: int = 2
    preparation_workers: int = 2
    plateau_min_final_stage_epochs: int = 20
    updates_history_limit: int = 8192
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> TrainConfig:
        names = {f.name for f in dataclasses.fields(TrainConfig)} - {"extra"}
        unknown = sorted(set(d) - names)
        if unknown:
            raise KeyError(f"unknown config keys: {unknown}")
        kwargs = {}
        for f in dataclasses.fields(TrainConfig):
            if f.name in d:
                v = d[f.name]
                kwargs[f.name] = tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
        return TrainConfig(**kwargs)

    @staticmethod
    def load(path: str) -> TrainConfig:
        with open(path) as f:
            return TrainConfig.from_dict(json.load(f))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("extra")
        return d

    @property
    def K_values(self) -> tuple:
        return tuple(int(k) for k in self.iteration_counts)
