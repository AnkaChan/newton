# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build experimental native Newton particle models with hexahedral metadata.

This opt-in helper is CPU/float32 only and may change without compatibility
notice. Hex mechanics belong to the learned solver, not Newton tetrahedra.
PyTorch is imported only when building a model to reuse material and mass
validation from the experimental hexahedral energy implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

import newton

from .data import VoxelGridData, generate_cuboid

__all__ = ["build_newton_hex_model"]


def build_newton_hex_model(
    rest: VoxelGridData,
    fixed_indices: Sequence[int] | np.ndarray,
    *,
    lame_lambda,
    lame_mu,
    density,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> newton.Model:
    """Build a standard Newton model for a canonical shared-corner hex grid.

    Experimental. The result is finalized explicitly on CPU. Initial particles
    use the canonical rest positions and zero velocities. Fixed particles keep
    their physical positive masses and inverse masses; their ACTIVE flag is
    cleared and ``model.learned_intrinsic.fixed`` records the exact constraints.
    The learned solver must enforce these positions explicitly.

    All custom attributes use assignment MODEL and namespace learned_intrinsic:
    ``cell_counts`` (vec3i, ONCE), ``cell_size`` (float32, ONCE),
    ``rest_positions`` (vec3, PARTICLE), ``fixed`` (bool, PARTICLE),
    ``cell_corner_indices`` (8-component int32 vector, custom frequency hex),
    and ``lame_lambda``, ``lame_mu``, ``density`` (float32, hex).
    Hex corner indices have references="particle". The custom frequency count
    is available through model.get_custom_frequency_count("learned_intrinsic:hex").
    ONCE arrays have one element. No tetrahedra, springs, surfaces, collision
    shapes, or proxy rigid bodies are added. Time step belongs to the solver.

    Args:
        rest: Canonical cubic grid, including shifted origins, in z-fast order.
        fixed_indices: Unique shared-corner indices to constrain; may be empty.
        lame_lambda: Nonnegative scalar or per-hex first Lamé parameter [Pa].
        lame_mu: Positive scalar or per-hex shear modulus [Pa].
        density: Positive scalar or per-hex physical density [kg/m^3].
        gravity: Finite world-space acceleration [m/s^2], length three.

    Returns:
        Native Newton Model with CPU float32 geometry/material arrays and normal
        state()/control() behavior. Float32 rest metadata should be compared
        with float32 tolerance when reconstructing the canonical rest grid.
    """
    from .hex_energy import HexImplicitEulerLoss  # noqa: PLC0415 - Keep optional PyTorch behind model construction.

    canonical = generate_cuboid(rest.cell_counts, cell_size=rest.cell_size, origin=tuple(rest.corner_rest_positions[0]))
    if not np.array_equal(rest.cell_corner_indices, canonical.cell_corner_indices):
        raise ValueError("rest hex corner indices must use the canonical full-cuboid topology")
    # Reuse the physical law's canonical-geometry/material checks and its exact
    # lumped-mass construction; dt=1 is incidental and is not stored in Model.
    physical = HexImplicitEulerLoss(rest, lame_lambda, lame_mu, density, 1.0)
    masses = physical.lumped_mass.numpy()
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("hex lumped masses must remain finite and positive in float32")
    particle_count = len(rest.corner_rest_positions)
    fixed = np.asarray(fixed_indices)
    if fixed.ndim != 1 or (fixed.size and not np.issubdtype(fixed.dtype, np.integer)):
        raise ValueError("fixed_indices must be a one-dimensional sequence of integer indices")
    if fixed.size and (np.any(fixed < 0) or np.any(fixed >= particle_count)):
        raise ValueError("fixed_indices must reference existing shared corners")
    fixed = fixed.astype(np.int64)
    if len(np.unique(fixed)) != len(fixed):
        raise ValueError("fixed_indices must be unique")
    fixed_flags = np.zeros(particle_count, dtype=bool)
    fixed_flags[fixed] = True
    gravity_array = np.asarray(gravity, dtype=np.float32)
    if gravity_array.shape != (3,) or not np.isfinite(gravity_array).all():
        raise ValueError("gravity must be a finite three-vector")

    namespace = "learned_intrinsic"
    frequency = f"{namespace}:hex"
    builder = newton.ModelBuilder(gravity=tuple(gravity_array.tolist()))
    builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="hex", namespace=namespace))
    vec8i = wp.types.vector(length=8, dtype=wp.int32)

    def register(name, dtype, attribute_frequency, *, values=None, references=None):
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name=name,
                dtype=dtype,
                frequency=attribute_frequency,
                assignment=newton.Model.AttributeAssignment.MODEL,
                namespace=namespace,
                values=values,
                references=references,
            )
        )

    register("cell_counts", wp.vec3i, newton.Model.AttributeFrequency.ONCE, values={0: tuple(rest.cell_counts)})
    register("cell_size", wp.float32, newton.Model.AttributeFrequency.ONCE, values={0: float(rest.cell_size)})
    register("rest_positions", wp.vec3, newton.Model.AttributeFrequency.PARTICLE)
    register("fixed", wp.bool, newton.Model.AttributeFrequency.PARTICLE)
    register("cell_corner_indices", vec8i, frequency, references="particle")
    for name in ("lame_lambda", "lame_mu", "density"):
        register(name, wp.float32, frequency)

    positions = rest.corner_rest_positions.astype(np.float32).tolist()
    active = int(newton.ParticleFlags.ACTIVE)
    flags = np.full(particle_count, active, dtype=np.int32)
    flags[fixed_flags] &= ~active
    builder.add_particles(
        pos=positions,
        vel=[(0.0, 0.0, 0.0)] * particle_count,
        mass=masses.tolist(),
        radius=[0.0] * particle_count,
        flags=flags.tolist(),
        custom_attributes={
            f"{namespace}:rest_positions": positions,
            f"{namespace}:fixed": fixed_flags.tolist(),
        },
    )
    lam = physical.lame_lambda.numpy()
    mu = physical.lame_mu.numpy()
    rho = physical.density.numpy()
    builder.add_custom_values_batch(
        [
            {
                f"{namespace}:cell_corner_indices": tuple(int(index) for index in corners),
                f"{namespace}:lame_lambda": float(lam[cell]),
                f"{namespace}:lame_mu": float(mu[cell]),
                f"{namespace}:density": float(rho[cell]),
            }
            for cell, corners in enumerate(rest.cell_corner_indices)
        ]
    )
    return builder.finalize(device="cpu")
