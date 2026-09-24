# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample one deterministic, uniform material for an experimental trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["MaterialRanges", "MaterialSample", "sample_material"]


@dataclass(frozen=True)
class MaterialRanges:
    """Independent log-uniform Lamé [Pa] and density [kg/m³] bounds."""

    lame_lambda: tuple[float, float] = (1e3, 1e6)
    """First Lamé parameter bounds [Pa]."""

    lame_mu: tuple[float, float] = (1e3, 1e6)
    """Shear modulus bounds [Pa]."""

    density: tuple[float, float] = (100.0, 10000.0)
    """Rest density bounds [kg/m³]."""

    def __post_init__(self) -> None:
        """Reject bounds that cannot define a positive log-uniform range."""
        for name in ("lame_lambda", "lame_mu", "density"):
            bounds = getattr(self, name)
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bounds)
                or bounds[0] <= 0
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"{name} must be finite positive (lower, upper) bounds with lower < upper")


@dataclass(frozen=True)
class MaterialSample:
    """Object-level scalar Lamé parameters [Pa] and density [kg/m³]."""

    lame_lambda: float
    """First Lamé parameter [Pa]."""

    lame_mu: float
    """Shear modulus [Pa]."""

    density: float
    """Rest density [kg/m³]."""


_DEFAULT_RANGES = MaterialRanges()


def sample_material(seed: int, *, ranges: MaterialRanges = _DEFAULT_RANGES) -> MaterialSample:
    """Draw one reproducible material, held fixed over a physical trajectory.

    Each component uses an independent seeded uniform variate in logarithmic
    space. The returned scalar fields can be passed to ``build_newton_hex_model``.

    Args:
        seed: Nonnegative trajectory material seed.
        ranges: Per-component lower and upper bounds.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a nonnegative integer")
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(ranges, MaterialRanges):
        raise TypeError("ranges must be MaterialRanges")
    draws = np.random.default_rng(seed).random(3)

    def transform(bounds: tuple[float, float], draw: float) -> float:
        lower, upper = bounds
        return math.exp(math.log(lower) + float(draw) * (math.log(upper) - math.log(lower)))

    return MaterialSample(
        lame_lambda=transform(ranges.lame_lambda, draws[0]),
        lame_mu=transform(ranges.lame_mu, draws[1]),
        density=transform(ranges.density, draws[2]),
    )
