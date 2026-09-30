# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only float64 cloth forces and experiment-only ALM metric overrides."""

import warp as wp

from newton._src.solvers.vbd.particle_alm_kernels import (
    ParticleElasticityAlmState,
    _bounded_rho,
    _bounded_rho_ratio,
    _particle_mobility,
    _triangle_deformation,
    particle_alm_hinge_geometry,
    particle_alm_triangle_geometry,
)


@wp.kernel
def triangle_metrics(
    q: wp.array[wp.vec3],
    ids: wp.array2d[int],
    poses: wp.array[wp.mat22],
    areas: wp.array[float],
    materials: wp.array2d[float],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    dt: float,
    floor: float,
    state: ParticleElasticityAlmState,
):
    t = wp.tid()
    f0, f1 = _triangle_deformation(t, q, ids, poses[t])
    norm, _area, g0, g1 = particle_alm_triangle_geometry(f0, f1)
    mn = float(0.0)
    ma = float(0.0)
    for v in range(3):
        w = -(poses[t][0] + poses[t][1])
        if v > 0:
            w = poses[t][v - 1]
        gn = (w[0] * f0 + w[1] * f1) / wp.max(norm, 1.0e-10)
        ga = w[0] * g0 + w[1] * g1
        mobility = _particle_mobility(ids[t, v], inv_mass, flags)
        mn += mobility * wp.dot(gn, gn)
        ma += mobility * wp.dot(ga, ga)
    # Native preparation has already seeded/retired rows. Only change rho.
    if state.tri_rho_stretch[t] > 0.0:
        inertia = _bounded_rho_ratio(state.rho_scale, 1.0, dt, dt, mn, areas[t])
        state.tri_rho_stretch[t] = _bounded_rho(wp.max(inertia, floor * materials[t, 0]))
    if state.tri_rho_area[t] > 0.0:
        inertia = _bounded_rho_ratio(state.rho_scale, 1.0, dt, dt, ma, areas[t])
        k = materials[t, 0] + materials[t, 1]
        state.tri_rho_area[t] = _bounded_rho(wp.max(inertia, floor * k))


@wp.kernel
def bend_metrics(
    q: wp.array[wp.vec3],
    ids: wp.array2d[int],
    rest_length: wp.array[float],
    properties: wp.array2d[float],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    dt: float,
    floor: float,
    state: ParticleElasticityAlmState,
):
    e = wp.tid()
    if state.bend_rho[e] <= 0.0:
        return
    i0, i1, i2, i3 = ids[e, 0], ids[e, 1], ids[e, 2], ids[e, 3]
    _theta, g0, g1, g2, g3, _valid = particle_alm_hinge_geometry(q[i0], q[i1], q[i2], q[i3])
    mobility = (
        _particle_mobility(i0, inv_mass, flags) * wp.dot(g0, g0)
        + _particle_mobility(i1, inv_mass, flags) * wp.dot(g1, g1)
        + _particle_mobility(i2, inv_mass, flags) * wp.dot(g2, g2)
        + _particle_mobility(i3, inv_mass, flags) * wp.dot(g3, g3)
    )
    inertia = _bounded_rho_ratio(state.rho_scale, 1.0, dt, dt, mobility, 1.0)
    k = properties[e, 0] * rest_length[e]
    state.bend_rho[e] = _bounded_rho(wp.max(inertia, floor * k))


@wp.func
def transmitted(k: wp.float64, rho: float, history: float, c: wp.float64):
    r = wp.float64(rho)
    if k <= wp.float64(0.0) or r <= wp.float64(0.0):
        return wp.float64(0.0)
    return k / (k + r) * (r * c + wp.float64(history))


@wp.kernel
def triangle_forces(
    q: wp.array[wp.vec3],
    q0: wp.array[wp.vec3],
    ids: wp.array2d[int],
    poses: wp.array[wp.mat22],
    areas: wp.array[float],
    materials: wp.array2d[float],
    dt: wp.float64,
    state: ParticleElasticityAlmState,
    elastic: wp.array[wp.vec3d],
    damping: wp.array[wp.vec3d],
    gap: wp.array[wp.vec3d],
):
    t = wp.tid()
    p = wp.mat22d(poses[t])
    e0 = wp.vec3d(q[ids[t, 1]]) - wp.vec3d(q[ids[t, 0]])
    e1 = wp.vec3d(q[ids[t, 2]]) - wp.vec3d(q[ids[t, 0]])
    a0 = wp.vec3d(q0[ids[t, 1]]) - wp.vec3d(q0[ids[t, 0]])
    a1 = wp.vec3d(q0[ids[t, 2]]) - wp.vec3d(q0[ids[t, 0]])
    f0 = e0 * p[0, 0] + e1 * p[1, 0]
    f1 = e0 * p[0, 1] + e1 * p[1, 1]
    h0 = a0 * p[0, 0] + a1 * p[1, 0]
    h1 = a0 * p[0, 1] + a1 * p[1, 1]
    aa, bb, ab = wp.dot(f0, f0), wp.dot(f1, f1), wp.dot(f0, f1)
    norm = wp.sqrt(wp.max(aa + bb, wp.float64(1.0e-20)))
    j = wp.sqrt(wp.max(aa * bb - ab * ab, wp.float64(1.0e-20)))
    g0, g1 = (bb * f0 - ab * f1) / j, (aa * f1 - ab * f0) / j
    mu = wp.float64(materials[t, 0])
    k = wp.float64(materials[t, 0] + materials[t, 1])
    ca = j - wp.float64(1.0) - mu / wp.max(k, wp.float64(1.0e-6))
    p0, p1 = mu * f0 + k * ca * g0, mu * f1 + k * ca * g1
    d00, d11, d01 = aa - wp.dot(h0, h0), bb - wp.dot(h1, h1), ab - wp.dot(h0, h1)
    pd0 = wp.float64(2.0) * wp.float64(materials[t, 2]) / dt * (d00 * f0 + d01 * f1)
    pd1 = wp.float64(2.0) * wp.float64(materials[t, 2]) / dt * (d01 * f0 + d11 * f1)
    gp0, gp1 = wp.vec3d(0.0), wp.vec3d(0.0)
    if state.enabled != 0:
        tn = transmitted(mu, state.tri_rho_stretch[t], state.tri_lambda_stretch[t], norm)
        ta = transmitted(k, state.tri_rho_area[t], state.tri_lambda_area[t], ca)
        gp0 = (tn / norm) * f0 + ta * g0 - p0
        gp1 = (tn / norm) * f1 + ta * g1 - p1
    for v in range(3):
        w = -(p[0] + p[1])
        if v > 0:
            w = p[v - 1]
        area = wp.float64(areas[t])
        wp.atomic_add(elastic, ids[t, v], -area * (w[0] * p0 + w[1] * p1))
        wp.atomic_add(damping, ids[t, v], -area * (w[0] * pd0 + w[1] * pd1))
        wp.atomic_add(gap, ids[t, v], -area * (w[0] * gp0 + w[1] * gp1))


@wp.func
def hinge(q: wp.array[wp.vec3], i0: int, i1: int, i2: int, i3: int):
    x0, x1, x2, x3 = wp.vec3d(q[i0]), wp.vec3d(q[i1]), wp.vec3d(q[i2]), wp.vec3d(q[i3])
    edge = x3 - x2
    n0, n1 = wp.cross(x2 - x0, x3 - x0), wp.cross(x3 - x1, x2 - x1)
    le, l0, l1 = wp.length(edge), wp.length(n0), wp.length(n1)
    if le < wp.float64(1.0e-6) or l0 < wp.float64(1.0e-6) or l1 < wp.float64(1.0e-6):
        return wp.float64(0.0), wp.vec3d(0.0), wp.vec3d(0.0), wp.vec3d(0.0), wp.vec3d(0.0), 0
    theta = wp.atan2(wp.dot(wp.cross(n0 / l0, n1 / l1), edge / le), wp.dot(n0 / l0, n1 / l1))
    g0, g1 = -le * n0 / (l0 * l0), -le * n1 / (l1 * l1)
    g2 = -(wp.dot(x3 - x0, edge) * g0 + wp.dot(x3 - x1, edge) * g1) / (le * le)
    g3 = -(g0 + g1 + g2)
    return theta, g0, g1, g2, g3, 1


@wp.kernel
def bend_forces(
    q: wp.array[wp.vec3],
    q0: wp.array[wp.vec3],
    ids: wp.array2d[int],
    rest_angle: wp.array[float],
    rest_length: wp.array[float],
    properties: wp.array2d[float],
    dt: wp.float64,
    state: ParticleElasticityAlmState,
    elastic: wp.array[wp.vec3d],
    damping: wp.array[wp.vec3d],
    gap: wp.array[wp.vec3d],
):
    e = wp.tid()
    if ids[e, 0] < 0 or ids[e, 1] < 0 or ids[e, 2] < 0 or ids[e, 3] < 0:
        return
    theta, g0, g1, g2, g3, valid = hinge(q, ids[e, 0], ids[e, 1], ids[e, 2], ids[e, 3])
    old_theta, _h0, _h1, _h2, _h3, old_valid = hinge(q0, ids[e, 0], ids[e, 1], ids[e, 2], ids[e, 3])
    if valid == 0:
        return
    k = wp.float64(properties[e, 0]) * wp.float64(rest_length[e])
    c = theta - wp.float64(rest_angle[e])
    moment = k * c
    delta = theta - old_theta
    pi = wp.float64(3.141592653589793)
    if delta > pi:
        delta -= wp.float64(2.0) * pi
    elif delta < -pi:
        delta += wp.float64(2.0) * pi
    damp = wp.float64(0.0)
    if old_valid != 0:
        damp = wp.float64(properties[e, 1]) * wp.float64(rest_length[e]) / dt * delta
    dm = wp.float64(0.0)
    if state.enabled != 0:
        dm = transmitted(k, state.bend_rho[e], state.bend_lambda[e], c) - moment
    for v in range(4):
        g = g0
        if v == 1:
            g = g1
        elif v == 2:
            g = g2
        elif v == 3:
            g = g3
        wp.atomic_add(elastic, ids[e, v], -moment * g)
        wp.atomic_add(damping, ids[e, v], -damp * g)
        wp.atomic_add(gap, ids[e, v], -dm * g)


@wp.kernel
def record_row(
    q: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    mass: wp.array[float],
    flags: wp.array[int],
    elastic: wp.array[wp.vec3d],
    damping: wp.array[wp.vec3d],
    contact: wp.array[wp.vec3],
    gap: wp.array[wp.vec3d],
    dt: wp.float64,
    iteration: int,
    rows: wp.array2d[wp.float64],
):
    # Serial deterministic reduction: the force assembly is parallel.
    for v in range(q.shape[0]):
        if (flags[v] & 1) != 0 and mass[v] > 0.0:
            fi = wp.float64(mass[v]) / (dt * dt) * (wp.vec3d(target[v]) - wp.vec3d(q[v]))
            fc = wp.vec3d(contact[v])
            r = elastic[v] + damping[v] + fc + fi
            ra = r + gap[v]
            dx = wp.vec3d(q[v]) - wp.vec3d(previous[v])
            rows[iteration, 0] += wp.dot(r, r)
            rows[iteration, 1] += wp.dot(ra, ra)
            rows[iteration, 2] += wp.dot(gap[v], gap[v])
            rows[iteration, 3] += wp.dot(elastic[v], elastic[v])
            rows[iteration, 4] += wp.dot(damping[v], damping[v])
            rows[iteration, 5] += wp.dot(fc, fc)
            rows[iteration, 6] += wp.dot(fi, fi)
            rows[iteration, 7] += wp.float64(1.0)
            rows[iteration, 8] = wp.max(rows[iteration, 8], wp.length(r))
            rows[iteration, 9] += wp.dot(dx, dx)
