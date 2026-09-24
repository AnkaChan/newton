# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample one deterministic isotropic material for an experimental trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["MaterialRanges", "MaterialSample", "lame_from_youngs_modulus", "sample_material"]


@dataclass(frozen=True)
class MaterialRanges:
    """Young's modulus [Pa], Poisson's ratio, and density [kg/m³] bounds."""

    youngs_modulus: tuple[float, float] = (1e3, 1e6)
    """Log-uniform Young's modulus bounds [Pa]."""

    poissons_ratio: tuple[float, float] = (0.2, 0.49)
    """Linear-uniform Poisson's ratio bounds; equal values fix the ratio."""

    density: tuple[float, float] = (100.0, 10000.0)
    """Rest density bounds [kg/m³]."""

    def __post_init__(self) -> None:
        """Reject bounds outside the compressible nonnegative-Lamé domain."""
        for name in ("youngs_modulus", "poissons_ratio", "density"):
            bounds = getattr(self, name)
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    for value in bounds
                )
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"{name} must be finite ordered (lower, upper) bounds")
            if name == "poissons_ratio":
                if bounds[0] < 0 or bounds[1] >= 0.5:
                    raise ValueError("poissons_ratio bounds must satisfy 0 <= lower <= upper < 0.5")
            elif bounds[0] <= 0:
                raise ValueError(f"{name} bounds must be positive")


@dataclass(frozen=True)
class MaterialSample:
    """Object-level scalar Lamé parameters [Pa] and density [kg/m³]."""

    lame_lambda: float
    """First Lamé parameter [Pa]."""

    lame_mu: float
    """Shear modulus [Pa]."""

    density: float
    """Rest density [kg/m³]."""

    @property
    def youngs_modulus(self) -> float:
        """Recover Young's modulus [Pa] from the derived Lamé parameters."""
        return self.lame_mu * (3 * self.lame_lambda + 2 * self.lame_mu) / (self.lame_lambda + self.lame_mu)

    @property
    def poissons_ratio(self) -> float:
        """Recover Poisson's ratio from the derived Lamé parameters."""
        return self.lame_lambda / (2 * (self.lame_lambda + self.lame_mu))


def lame_from_youngs_modulus(youngs_modulus: float, poissons_ratio: float) -> tuple[float, float]:
    """Convert isotropic Young's modulus [Pa] and Poisson's ratio to Lamé [Pa].

    Args:
        youngs_modulus: Positive finite Young's modulus [Pa].
        poissons_ratio: Finite ratio in [0, 0.5), yielding nonnegative λ.

    Returns:
        First Lamé parameter λ and shear modulus μ [Pa].
    """
    if (
        not isinstance(youngs_modulus, (int, float))
        or isinstance(youngs_modulus, bool)
        or not math.isfinite(youngs_modulus)
        or youngs_modulus <= 0
    ):
        raise ValueError("youngs_modulus must be finite and positive")
    if (
        not isinstance(poissons_ratio, (int, float))
        or isinstance(poissons_ratio, bool)
        or not math.isfinite(poissons_ratio)
        or not 0 <= poissons_ratio < 0.5
    ):
        raise ValueError("poissons_ratio must satisfy 0 <= nu < 0.5")
    lame_lambda = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    lame_mu = youngs_modulus / (2 * (1 + poissons_ratio))
    return float(lame_lambda), float(lame_mu)


_DEFAULT_RANGES = MaterialRanges()


def sample_material(seed: int, *, ranges: MaterialRanges = _DEFAULT_RANGES) -> MaterialSample:
    """Draw one reproducible material, held fixed over a physical trajectory.

    Young's modulus and density are log-uniform; Poisson's ratio is linear-
    uniform. The derived Lamé scalars can be passed to ``build_newton_hex_model``.

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

    def log_uniform(bounds: tuple[float, float], draw: float) -> float:
        lower, upper = bounds
        if lower == upper:
            return float(lower)
        return math.exp(math.log(lower) + float(draw) * (math.log(upper) - math.log(lower)))

    youngs_modulus = log_uniform(ranges.youngs_modulus, draws[0])
    lower_nu, upper_nu = ranges.poissons_ratio
    poissons_ratio = lower_nu + float(draws[1]) * (upper_nu - lower_nu)
    lame_lambda, lame_mu = lame_from_youngs_modulus(youngs_modulus, poissons_ratio)
    return MaterialSample(
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        density=log_uniform(ranges.density, draws[2]),
    )
