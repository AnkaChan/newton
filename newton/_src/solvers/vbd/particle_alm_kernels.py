# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-address elasticity history and element passes for particle ALM."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from .rigid_vbd_kernels import _alm_relaxed_ascent, _compliant_alm_coefficients, _reset_world_selected

wp.set_module_options({"enable_backward": False})


@wp.struct
class ParticleElasticityAlmState:
    """Solver-owned structural stresses, numerical metrics, and pending reseeds."""

    enabled: int
    deviatoric: int
    rho_scale: float
    tet_lambda_mu: wp.array[wp.mat33]
    tet_lambda_pressure: wp.array[float]
    tet_rho_mu: wp.array[float]
    tet_rho_pressure: wp.array[float]
    spring_lambda: wp.array[float]
    spring_rho: wp.array[float]
    bend_lambda: wp.array[float]
    bend_rho: wp.array[float]
    tet_pending: wp.array[int]
    spring_pending: wp.array[int]
    bend_pending: wp.array[int]


@wp.func
def particle_alm_coefficients(material_k: float, rho: float):
    """Return stable retention, effective stiffness, and relaxation coefficients."""
    return _compliant_alm_coefficients(material_k, rho)


@wp.func
def particle_alm_ascent(lam: Any, constraint: Any, material_k: float, rho: float) -> Any:
    """Advance scalar or matrix history without forming an overflowing rho*C."""
    return _alm_relaxed_ascent(lam, constraint, material_k, rho)


@wp.func
def particle_alm_cofactor(F: wp.mat33) -> wp.mat33:
    """Compute the determinant gradient without inverting the deformation."""
    c0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    c2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    return wp.matrix_from_cols(wp.cross(c1, c2), wp.cross(c2, c0), wp.cross(c0, c1))


@wp.func
def particle_alm_tet_weight(Dm_inv: wp.mat33, vertex_order: int) -> wp.vec3:
    """Select a row of the inverse rest matrix for F=Ds*Dm_inverse."""
    if vertex_order == 0:
        return -(Dm_inv[0] + Dm_inv[1] + Dm_inv[2])
    return Dm_inv[vertex_order - 1]


@wp.func
def _tet_deformation(tet: int, pos: wp.array[wp.vec3], indices: wp.array2d[int], Dm_inv: wp.mat33):
    x0 = pos[indices[tet, 0]]
    return wp.matrix_from_cols(pos[indices[tet, 1]] - x0, pos[indices[tet, 2]] - x0, pos[indices[tet, 3]] - x0) * Dm_inv


@wp.func
def _particle_mobility(particle: int, inv_mass: wp.array[float], flags: wp.array[int]) -> float:
    if (flags[particle] & ParticleFlags.ACTIVE) != 0:
        return inv_mass[particle]
    return 0.0


@wp.func
def _bounded_rho(value: wp.float64) -> float:
    """Round positive metrics into the finite float32 range without retiring rows."""
    return float(wp.clamp(value, wp.float64(1.401298464324817e-45), wp.float64(3.4028234663852886e38)))


@wp.func
def _hinge_gradient(
    n1: wp.vec3,
    n2: wp.vec3,
    e: wp.vec3,
    norm1: float,
    norm2: float,
    dn1: wp.mat33,
    dn2: wp.mat33,
    sine: float,
    cosine: float,
) -> wp.vec3:
    I = wp.identity(n=3, dtype=float)
    dn1_hat = (1.0 / norm1) * (I - wp.outer(n1, n1)) * dn1
    dn2_hat = (1.0 / norm2) * (I - wp.outer(n2, n2)) * dn2
    dsin = wp.transpose(wp.skew(n1) * dn2_hat - wp.skew(n2) * dn1_hat) * e
    dcos = wp.transpose(dn1_hat) * n2 + wp.transpose(dn2_hat) * n1
    return dsin * cosine - dcos * sine


@wp.func
def particle_alm_hinge_geometry(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    """Return the existing signed hinge angle, its four gradients, and validity."""
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)
    norm1 = wp.length(n1)
    norm2 = wp.length(n2)
    norm_e = wp.length(e)
    if norm1 < 1.0e-6 or norm2 < 1.0e-6 or norm_e < 1.0e-6:
        return 0.0, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), 0
    n1_hat = n1 / norm1
    n2_hat = n2 / norm2
    e_hat = e / norm_e
    sine = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cosine = wp.dot(n1_hat, n2_hat)
    theta = wp.atan2(sine, cosine)
    g0 = _hinge_gradient(n1_hat, n2_hat, e_hat, norm1, norm2, wp.skew(e), wp.mat33(0.0), sine, cosine)
    g1 = _hinge_gradient(n1_hat, n2_hat, e_hat, norm1, norm2, wp.mat33(0.0), -wp.skew(e), sine, cosine)
    g2 = _hinge_gradient(n1_hat, n2_hat, e_hat, norm1, norm2, -wp.skew(x03), wp.skew(x13), sine, cosine)
    g3 = _hinge_gradient(n1_hat, n2_hat, e_hat, norm1, norm2, wp.skew(x02), -wp.skew(x12), sine, cosine)
    return theta, g0, g1, g2, g3, 1


@wp.kernel
def _prepare_tets(
    pos: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    dt: float,
    state: ParticleElasticityAlmState,
):
    tet = wp.tid()
    mu = materials[tet, 0]
    pressure_k = materials[tet, 1] + mu
    Dm_inv = poses[tet]
    rest_det = wp.determinant(Dm_inv)
    F = _tet_deformation(tet, pos, indices, Dm_inv)
    cof = particle_alm_cofactor(F)
    mobility_mu = float(0.0)
    mobility_pressure = float(0.0)
    for vertex in range(4):
        w = particle_alm_tet_weight(Dm_inv, vertex)
        g = cof * w
        mobility = _particle_mobility(indices[tet, vertex], inv_mass, flags)
        mobility_mu += mobility * wp.dot(w, w)
        mobility_pressure += mobility * wp.dot(g, g)
    # Pending tet bits independently track mu and pressure through degeneracies.
    pending = state.tet_pending[tet]
    numerator = wp.float64(state.rho_scale) * wp.float64(6.0) * wp.float64(rest_det)
    dt_squared = wp.float64(dt) * wp.float64(dt)
    if state.deviatoric != 0:
        state.tet_rho_mu[tet] = 0.0
        if mu > 0.0 and rest_det > 0.0 and mobility_mu > 0.0:
            state.tet_rho_mu[tet] = _bounded_rho(numerator / (dt_squared * wp.float64(mobility_mu)))
            if (pending & 1) != 0:
                state.tet_lambda_mu[tet] = mu * F
            pending = pending & ~1
        else:
            state.tet_lambda_mu[tet] = wp.mat33(0.0)
            pending = pending | 1
    state.tet_rho_pressure[tet] = 0.0
    if pressure_k > 0.0 and rest_det > 0.0 and mobility_pressure > 0.0:
        state.tet_rho_pressure[tet] = _bounded_rho(numerator / (dt_squared * wp.float64(mobility_pressure)))
        if (pending & 2) != 0:
            # Keep the standing -mu stress when 1+mu/K rounds to 1.
            state.tet_lambda_pressure[tet] = pressure_k * (wp.determinant(F) - 1.0) - pressure_k * (
                mu / wp.max(pressure_k, 1.0e-6)
            )
        pending = pending & ~2
    else:
        state.tet_lambda_pressure[tet] = 0.0
        pending = pending | 2
    state.tet_pending[tet] = pending


@wp.kernel
def _update_tets(
    pos: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    state: ParticleElasticityAlmState,
):
    tet = wp.tid()
    mu = materials[tet, 0]
    pressure_k = materials[tet, 1] + mu
    F = _tet_deformation(tet, pos, indices, poses[tet])
    if state.deviatoric != 0:
        state.tet_lambda_mu[tet] = particle_alm_ascent(state.tet_lambda_mu[tet], F, mu, state.tet_rho_mu[tet])
    if pressure_k > 0.0:
        residual = (wp.determinant(F) - 1.0) - mu / wp.max(pressure_k, 1.0e-6)
        state.tet_lambda_pressure[tet] = particle_alm_ascent(
            state.tet_lambda_pressure[tet], residual, pressure_k, state.tet_rho_pressure[tet]
        )
    else:
        state.tet_lambda_pressure[tet] = 0.0


@wp.kernel
def _prepare_springs(
    pos: wp.array[wp.vec3],
    indices: wp.array[int],
    rest_length: wp.array[float],
    stiffness: wp.array[float],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    dt: float,
    state: ParticleElasticityAlmState,
):
    spring = wp.tid()
    i = indices[2 * spring]
    j = indices[2 * spring + 1]
    length = wp.length(pos[i] - pos[j])
    mobility = _particle_mobility(i, inv_mass, flags) + _particle_mobility(j, inv_mass, flags)
    material_k = stiffness[spring]
    state.spring_rho[spring] = 0.0
    if material_k > 0.0 and length > 1.0e-8 and mobility > 0.0:
        inertia = wp.float64(state.rho_scale) / (wp.float64(dt) * wp.float64(dt) * wp.float64(mobility))
        state.spring_rho[spring] = _bounded_rho(wp.max(inertia, wp.float64(9.0) * wp.float64(material_k)))
        if state.spring_pending[spring] != 0:
            state.spring_lambda[spring] = material_k * (length - rest_length[spring])
        state.spring_pending[spring] = 0
    else:
        state.spring_lambda[spring] = 0.0
        state.spring_pending[spring] = 1


@wp.kernel
def _update_springs(
    pos: wp.array[wp.vec3],
    indices: wp.array[int],
    rest_length: wp.array[float],
    stiffness: wp.array[float],
    state: ParticleElasticityAlmState,
):
    spring = wp.tid()
    length = wp.length(pos[indices[2 * spring]] - pos[indices[2 * spring + 1]])
    if length > 1.0e-8:
        state.spring_lambda[spring] = particle_alm_ascent(
            state.spring_lambda[spring], length - rest_length[spring], stiffness[spring], state.spring_rho[spring]
        )
    else:
        state.spring_lambda[spring] = 0.0
        state.spring_pending[spring] = 1


@wp.kernel
def _prepare_bends(
    pos: wp.array[wp.vec3],
    indices: wp.array2d[int],
    rest_angle: wp.array[float],
    rest_length: wp.array[float],
    properties: wp.array2d[float],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    dt: float,
    state: ParticleElasticityAlmState,
):
    edge = wp.tid()
    state.bend_rho[edge] = 0.0
    i0 = indices[edge, 0]
    i1 = indices[edge, 1]
    i2 = indices[edge, 2]
    i3 = indices[edge, 3]
    if i0 < 0 or i1 < 0 or i2 < 0 or i3 < 0:
        state.bend_lambda[edge] = 0.0
        state.bend_pending[edge] = 1
        return
    theta, g0, g1, g2, g3, valid = particle_alm_hinge_geometry(pos[i0], pos[i1], pos[i2], pos[i3])
    mobility = (
        _particle_mobility(i0, inv_mass, flags) * wp.dot(g0, g0)
        + _particle_mobility(i1, inv_mass, flags) * wp.dot(g1, g1)
        + _particle_mobility(i2, inv_mass, flags) * wp.dot(g2, g2)
        + _particle_mobility(i3, inv_mass, flags) * wp.dot(g3, g3)
    )
    material_k = properties[edge, 0] * rest_length[edge]
    if material_k > 0.0 and valid != 0 and mobility > 0.0:
        state.bend_rho[edge] = _bounded_rho(
            wp.float64(state.rho_scale) / (wp.float64(dt) * wp.float64(dt) * wp.float64(mobility))
        )
        if state.bend_pending[edge] != 0:
            state.bend_lambda[edge] = material_k * (theta - rest_angle[edge])
        state.bend_pending[edge] = 0
    else:
        state.bend_lambda[edge] = 0.0
        state.bend_pending[edge] = 1


@wp.kernel
def _update_bends(
    pos: wp.array[wp.vec3],
    indices: wp.array2d[int],
    rest_angle: wp.array[float],
    rest_length: wp.array[float],
    properties: wp.array2d[float],
    state: ParticleElasticityAlmState,
):
    edge = wp.tid()
    i0 = indices[edge, 0]
    i1 = indices[edge, 1]
    i2 = indices[edge, 2]
    i3 = indices[edge, 3]
    if i0 < 0 or i1 < 0 or i2 < 0 or i3 < 0:
        state.bend_lambda[edge] = 0.0
        return
    theta, _g0, _g1, _g2, _g3, valid = particle_alm_hinge_geometry(pos[i0], pos[i1], pos[i2], pos[i3])
    if valid != 0:
        state.bend_lambda[edge] = particle_alm_ascent(
            state.bend_lambda[edge],
            theta - rest_angle[edge],
            properties[edge, 0] * rest_length[edge],
            state.bend_rho[edge],
        )
    else:
        state.bend_lambda[edge] = 0.0
        state.bend_pending[edge] = 1


@wp.kernel
def _reset_history(
    world_mask: wp.array[wp.bool],
    reset_all: bool,
    world_count: int,
    particle_world: wp.array[int],
    tet_indices: wp.array2d[int],
    spring_indices: wp.array[int],
    bend_indices: wp.array2d[int],
    state: ParticleElasticityAlmState,
):
    element = wp.tid()
    if element < state.tet_pending.shape[0]:
        selected = bool(False)
        for vertex in range(4):
            selected = selected or _reset_world_selected(
                particle_world[tet_indices[element, vertex]], world_mask, reset_all, world_count
            )
        if selected:
            state.tet_pending[element] = 2
            if state.deviatoric != 0:
                state.tet_pending[element] = 3
                state.tet_lambda_mu[element] = wp.mat33(0.0)
                state.tet_rho_mu[element] = 0.0
            state.tet_lambda_pressure[element] = 0.0
            state.tet_rho_pressure[element] = 0.0
    if element < state.spring_pending.shape[0]:
        selected = bool(False)
        for vertex in range(2):
            selected = selected or _reset_world_selected(
                particle_world[spring_indices[2 * element + vertex]], world_mask, reset_all, world_count
            )
        if selected:
            state.spring_pending[element] = 1
            state.spring_lambda[element] = 0.0
            state.spring_rho[element] = 0.0
    if element < state.bend_pending.shape[0]:
        selected = bool(False)
        for vertex in range(4):
            particle = bend_indices[element, vertex]
            if particle >= 0:
                selected = selected or _reset_world_selected(
                    particle_world[particle], world_mask, reset_all, world_count
                )
        if selected:
            state.bend_pending[element] = 1
            state.bend_lambda[element] = 0.0
            state.bend_rho[element] = 0.0


def create_particle_elasticity_alm_state(model, enabled: bool, deviatoric: bool, rho_scale: float):
    """Allocate histories before capture and validate immutable material settings."""
    float32 = np.finfo(np.float32)
    if not math.isfinite(rho_scale) or rho_scale < float(float32.smallest_subnormal) or rho_scale > float(float32.max):
        raise ValueError("particle_elasticity_alm_rho_scale must be positive and finite in float32")
    if enabled and model.tet_count:
        materials = model.tet_materials.numpy()[:, :2].astype(np.float64)
        mu = materials[:, 0]
        pressure_k = materials[:, 1] + mu
        inactive = (mu == 0.0) & (pressure_k == 0.0)
        if (
            not np.isfinite(materials).all()
            or np.any(mu < 0.0)
            or np.any((pressure_k <= 0.0) & ~inactive)
            or np.any(pressure_k > np.finfo(np.float32).max)
        ):
            raise ValueError(
                "Particle elasticity ALM requires nonnegative tet mu and positive finite lambda + mu, or an all-zero material"
            )
    if enabled and model.spring_count:
        stiffness = model.spring_stiffness.numpy()
        if not np.isfinite(stiffness).all() or np.any(stiffness < 0.0):
            raise ValueError("Particle elasticity ALM requires finite nonnegative spring stiffness")
    if enabled and model.edge_count:
        stiffness = model.edge_bending_properties.numpy()[:, 0].astype(np.float64)
        material_k = stiffness * model.edge_rest_length.numpy().astype(np.float64)
        if (
            not np.isfinite(stiffness).all()
            or np.any(stiffness < 0.0)
            or not np.isfinite(material_k).all()
            or np.any(material_k < 0.0)
            or np.any(material_k > float(float32.max))
        ):
            raise ValueError(
                "Particle elasticity ALM requires finite nonnegative hinge stiffness and rest-length product"
            )
    state = ParticleElasticityAlmState()
    state.enabled = int(enabled)
    state.deviatoric = int(deviatoric)
    state.rho_scale = rho_scale
    tet_count = model.tet_count if enabled else 0
    spring_count = model.spring_count if enabled else 0
    bend_count = model.edge_count if enabled else 0
    state.tet_lambda_mu = wp.zeros(tet_count if deviatoric else 0, dtype=wp.mat33, device=model.device)
    state.tet_lambda_pressure = wp.zeros(tet_count, dtype=float, device=model.device)
    state.tet_rho_mu = wp.zeros(tet_count if deviatoric else 0, dtype=float, device=model.device)
    state.tet_rho_pressure = wp.zeros(tet_count, dtype=float, device=model.device)
    state.spring_lambda = wp.zeros(spring_count, dtype=float, device=model.device)
    state.spring_rho = wp.zeros(spring_count, dtype=float, device=model.device)
    state.bend_lambda = wp.zeros(bend_count, dtype=float, device=model.device)
    state.bend_rho = wp.zeros(bend_count, dtype=float, device=model.device)
    state.tet_pending = wp.full(tet_count, 3 if deviatoric else 2, dtype=int, device=model.device)
    state.spring_pending = wp.ones(spring_count, dtype=int, device=model.device)
    state.bend_pending = wp.ones(bend_count, dtype=int, device=model.device)
    return state


def prepare_particle_elasticity_alm(model, pos: wp.array, dt: float, state: ParticleElasticityAlmState):
    """Reseed pending rows from the incoming pose and freeze this step's metrics."""
    if not state.enabled:
        return
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Particle elasticity ALM requires a positive finite timestep")
    if model.tet_count:
        wp.launch(
            _prepare_tets,
            model.tet_count,
            inputs=[
                pos,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                model.particle_inv_mass,
                model.particle_flags,
                dt,
                state,
            ],
            device=model.device,
        )
    if model.spring_count:
        wp.launch(
            _prepare_springs,
            model.spring_count,
            inputs=[
                pos,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.particle_inv_mass,
                model.particle_flags,
                dt,
                state,
            ],
            device=model.device,
        )
    if model.edge_count:
        wp.launch(
            _prepare_bends,
            model.edge_count,
            inputs=[
                pos,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_rest_length,
                model.edge_bending_properties,
                model.particle_inv_mass,
                model.particle_flags,
                dt,
                state,
            ],
            device=model.device,
        )


def update_particle_elasticity_alm(model, pos: wp.array, state: ParticleElasticityAlmState):
    """Advance each element's dual once after a complete primal color sweep."""
    if not state.enabled:
        return
    if model.tet_count:
        wp.launch(
            _update_tets,
            model.tet_count,
            inputs=[pos, model.tet_indices, model.tet_poses, model.tet_materials, state],
            device=model.device,
        )
    if model.spring_count:
        wp.launch(
            _update_springs,
            model.spring_count,
            inputs=[pos, model.spring_indices, model.spring_rest_length, model.spring_stiffness, state],
            device=model.device,
        )
    if model.edge_count:
        wp.launch(
            _update_bends,
            model.edge_count,
            inputs=[
                pos,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_rest_length,
                model.edge_bending_properties,
                state,
            ],
            device=model.device,
        )


def reset_particle_elasticity_alm(model, world_mask: wp.array | None, state: ParticleElasticityAlmState):
    """Invalidate selected histories in place for reseeding after user pose edits."""
    if not state.enabled:
        return
    count = max(model.tet_count, model.spring_count, model.edge_count)
    if count:
        wp.launch(
            _reset_history,
            count,
            inputs=[
                world_mask,
                world_mask is None,
                model.world_count,
                model.particle_world,
                model.tet_indices,
                model.spring_indices,
                model.edge_indices,
                state,
            ],
            device=model.device,
        )
