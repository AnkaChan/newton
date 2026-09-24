# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic fresh physical trajectory starts without retained state files.

Reset always samples from an owned canonical rest grid. Shape and velocity use
the original smoke sampler's seed stream and interpolation; material uses a
separate stream. Returned arrays are float32 owned copies and never become the
next reset's source. This module is NumPy-only and does not build Newton state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .data import VoxelGridData, generate_cuboid
from .material_sampling import MaterialRanges, MaterialSample, sample_material
from .multiscale import generate_multiscale, interpolate_control_grid, screen_geometry

__all__ = ["InitialState", "InitialStateAugmenter"]

_DEFAULT_MATERIAL_RANGES = MaterialRanges()
_GENERATOR_VERSION = "initial_state_v2"
_MULTISCALE_MAX_LEVELS = 3
_MIN_VOLUME_RATIO = 0.2
_VELOCITY_CONTROL_CAPS = (3, 3, 5)
_MAX_FLOAT32_HALVINGS = 25


@dataclass(frozen=True)
class InitialState:
    """Owned physical X [m], V [m/s], z-min pins, and trajectory material."""

    positions: np.ndarray
    velocities: np.ndarray
    fixed_indices: np.ndarray
    material: MaterialSample
    seed: int
    metadata: dict


def _seed(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _range(value, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(bound, bool) or not isinstance(bound, (int, float)) for bound in value)
        or not all(math.isfinite(bound) for bound in value)
        or value[0] < 0
        or value[0] > value[1]
    ):
        raise ValueError(f"{name} must be finite nonnegative (lower, upper) with lower <= upper")
    return float(value[0]), float(value[1])


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(values, dtype=np.float64) ** 2, axis=-1))))


class InitialStateAugmenter:
    """Regenerate seeded trajectory starts from an immutable canonical rest copy.

    The physical seed stream ``[master_seed, seed, 701]`` reproduces the smoke
    sampler's X/V distribution when ranges, timestep, and perturbation scale
    agree at one. Separate ``[master_seed, seed, 1103]`` and
    ``[master_seed, seed, 1301]`` streams select trajectory-level material and
    a shared global perturbation multiplier.
    Calling ``reset()`` repeats the current seed; ``reset(seed)`` switches it.
    """

    def __init__(
        self,
        rest: VoxelGridData,
        *,
        seed: int = 0,
        master_seed: int = 73,
        time_step: float = 1 / 300,
        material_ranges: MaterialRanges = _DEFAULT_MATERIAL_RANGES,
        strength_range: tuple[float, float] = (0.02, 0.1),
        velocity_dt_range: tuple[float, float] = (0.0, 0.1),
        perturbation_scale_range: tuple[float, float] = (1.0, 1.0),
    ):
        if not isinstance(rest, VoxelGridData):
            raise TypeError("rest must be canonical VoxelGridData")
        canonical = generate_cuboid(
            rest.cell_counts,
            cell_size=rest.cell_size,
            origin=tuple(float(value) for value in rest.corner_rest_positions[0]),
        )
        if not np.array_equal(rest.corner_rest_positions, canonical.corner_rest_positions) or not np.array_equal(
            rest.cell_corner_indices, canonical.cell_corner_indices
        ):
            raise ValueError("rest must have canonical cuboid positions and topology")
        if not isinstance(material_ranges, MaterialRanges):
            raise TypeError("material_ranges must be MaterialRanges")
        if (
            isinstance(time_step, bool)
            or not isinstance(time_step, (int, float))
            or not math.isfinite(time_step)
            or time_step <= 0
        ):
            raise ValueError("time_step must be finite and positive")
        self._rest = canonical
        self._fixed = np.flatnonzero(
            canonical.corner_rest_positions[:, 2] == canonical.corner_rest_positions[:, 2].min()
        ).astype(np.int64)
        self.seed = _seed(seed, "seed")
        self.master_seed = _seed(master_seed, "master_seed")
        self.time_step = float(time_step)
        self.material_ranges = material_ranges
        self.strength_range = _range(strength_range, "strength_range")
        self.velocity_dt_range = _range(velocity_dt_range, "velocity_dt_range")
        self.perturbation_scale_range = _range(perturbation_scale_range, "perturbation_scale_range")

    def _float32_positions(self, sampled: np.ndarray) -> tuple[np.ndarray, dict[str, float], int]:
        rest = self._rest
        rest32 = rest.corner_rest_positions.astype(np.float32)
        for halvings in range(_MAX_FLOAT32_HALVINGS):
            if halvings == 0:
                positions = sampled.astype(np.float32)
            else:
                positions = (
                    rest.corner_rest_positions + (0.5**halvings) * (sampled - rest.corner_rest_positions)
                ).astype(np.float32)
            positions[self._fixed] = rest32[self._fixed]
            if not np.isfinite(positions).all():
                continue
            screen = screen_geometry(rest, positions)
            if min(screen.values()) >= _MIN_VOLUME_RATIO:
                return positions, screen, halvings
        raise ValueError("float32 augmented shape remained invalid after deterministic backtracking")

    def reset(self, seed: int | None = None) -> InitialState:
        """Return an independent, screened initial state for the current seed."""
        current = self.seed if seed is None else _seed(seed, "seed")
        rest = self._rest
        physical_stream = [self.master_seed, current, 701]
        rng = np.random.default_rng(np.random.SeedSequence(physical_stream))
        strength = float(rng.uniform(*self.strength_range))
        sample = generate_multiscale(
            rest,
            seed=current,
            strength=strength,
            max_levels=_MULTISCALE_MAX_LEVELS,
            min_volume_ratio=_MIN_VOLUME_RATIO,
        )
        perturbation_stream = [self.master_seed, current, 1301]
        perturbation_scale = float(
            np.random.default_rng(np.random.SeedSequence(perturbation_stream)).uniform(*self.perturbation_scale_range)
        )
        if perturbation_scale == 1.0:
            sampled_positions = sample.positions  # Preserve the legacy float32 conversion bit for bit.
        elif perturbation_scale == 0.0:
            sampled_positions = rest.corner_rest_positions
        else:
            sampled_positions = rest.corner_rest_positions + perturbation_scale * (
                sample.positions - rest.corner_rest_positions
            )
        positions, screen, extra_halvings = self._float32_positions(sampled_positions)
        counts = tuple(min(n + 1, cap) for n, cap in zip(rest.cell_counts, _VELOCITY_CONTROL_CAPS, strict=True))
        controls = rng.uniform(-1, 1, size=(*counts, 3))
        controls[:, :, 0] = 0
        velocity = interpolate_control_grid(
            controls,
            rest.corner_rest_positions,
            origin=rest.corner_rest_positions[0],
            extent=np.asarray(rest.cell_counts) * rest.cell_size,
        )
        velocity[self._fixed] = 0
        unscaled_target_displacement_rms = float(rng.uniform(*self.velocity_dt_range) * rest.cell_size)
        norm = _rms(velocity)
        velocity *= unscaled_target_displacement_rms / (self.time_step * norm) if norm else 0
        velocity = velocity.astype(np.float32)
        if perturbation_scale == 0.0:
            velocity.fill(0)
        elif perturbation_scale != 1.0:
            velocity *= np.float32(perturbation_scale)
        velocity[self._fixed] = 0
        if not np.isfinite(velocity).all():
            raise ValueError("float32 initial velocity is nonfinite")

        material_stream = [self.master_seed, current, 1103]
        material_seed = int(np.random.SeedSequence(material_stream).generate_state(1, dtype=np.uint64)[0])
        material = sample_material(material_seed, ranges=self.material_ranges)
        metadata = {
            "schema_version": 1,
            "generator_version": _GENERATOR_VERSION,
            "physical_seed": current,
            "master_seed": self.master_seed,
            "physical_seed_sequence": physical_stream,
            "material_seed_sequence": material_stream,
            "material_seed": material_seed,
            "perturbation_seed_sequence": perturbation_stream,
            "cell_counts": list(rest.cell_counts),
            "cell_size": rest.cell_size,
            "origin": [float(value) for value in rest.corner_rest_positions[0]],
            "time_step": self.time_step,
            "strength_range": list(self.strength_range),
            "velocity_dt_range": list(self.velocity_dt_range),
            "perturbation_scale_range": list(self.perturbation_scale_range),
            "material_ranges": asdict(self.material_ranges),
            "multiscale_max_levels": _MULTISCALE_MAX_LEVELS,
            "minimum_volume_ratio": _MIN_VOLUME_RATIO,
            "velocity_control_point_caps": list(_VELOCITY_CONTROL_CAPS),
            "max_float32_backtracking_steps": _MAX_FLOAT32_HALVINGS - 1,
            "material": asdict(material),
            "strength": strength,
            "perturbation_scale": perturbation_scale,
            "augmentation_scale": sample.effective_scale * perturbation_scale * (0.5**extra_halvings),
            "multiscale_backtracking_steps": sample.backtracking_steps,
            "float32_backtracking_steps": extra_halvings,
            "velocity_dt_rms_m": _rms(self.time_step * velocity),
            "requested_velocity_dt_rms_m": perturbation_scale * unscaled_target_displacement_rms,
            "unscaled_requested_velocity_dt_rms_m": unscaled_target_displacement_rms,
            "screen": screen,
        }
        self.seed = current
        return InitialState(positions, velocity, self._fixed.copy(), material, current, metadata)
