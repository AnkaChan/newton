# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent float64 evaluation of the native VBD tet step objective."""

import numpy as np
import warp as wp


@wp.func
def deformation(q: wp.array[wp.vec3], ids: wp.vec4i, inverse_rest: wp.mat33d):
    a = wp.vec3d(q[ids[0]])
    return wp.matrix_from_cols(wp.vec3d(q[ids[1]]) - a, wp.vec3d(q[ids[2]]) - a, wp.vec3d(q[ids[3]]) - a) * inverse_rest


@wp.func
def cofactor(f: wp.mat33d):
    a = wp.vec3d(f[0, 0], f[1, 0], f[2, 0])
    b = wp.vec3d(f[0, 1], f[1, 1], f[2, 1])
    c = wp.vec3d(f[0, 2], f[1, 2], f[2, 2])
    return wp.matrix_from_cols(wp.cross(b, c), wp.cross(c, a), wp.cross(a, b))


@wp.kernel
def tet_forces(
    q: wp.array[wp.vec3],
    q0: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    dt: wp.float64,
    elastic: wp.array[wp.vec3d],
    damping: wp.array[wp.vec3d],
    initial_elastic: wp.array[wp.vec3d],
    determinants: wp.array[wp.float64],
):
    t = wp.tid()
    ids = wp.vec4i(indices[t, 0], indices[t, 1], indices[t, 2], indices[t, 3])
    b = wp.mat33d(poses[t])
    f = deformation(q, ids, b)
    f0 = deformation(q0, ids, b)
    mu = wp.float64(materials[t, 0])
    kp = mu + wp.float64(materials[t, 1])
    kd = wp.float64(materials[t, 2])
    volume = wp.float64(1.0) / (wp.float64(6.0) * wp.determinant(b))
    j = wp.determinant(f)
    stress = mu * f + (kp * (j - wp.float64(1.0)) - mu) * cofactor(f)
    stress0 = mu * f0 + (kp * (wp.determinant(f0) - wp.float64(1.0)) - mu) * cofactor(f0)
    damp_stress = wp.float64(2.0) * kd / dt * f * (wp.transpose(f) * f - wp.transpose(f0) * f0)
    for k in range(4):
        w = wp.vec3d()
        if k == 0:
            w = -wp.vec3d(b[0, 0] + b[1, 0] + b[2, 0], b[0, 1] + b[1, 1] + b[2, 1], b[0, 2] + b[1, 2] + b[2, 2])
        else:
            w = wp.vec3d(b[k - 1, 0], b[k - 1, 1], b[k - 1, 2])
        wp.atomic_add(elastic, ids[k], -volume * stress * w)
        wp.atomic_add(damping, ids[k], -volume * damp_stress * w)
        wp.atomic_add(initial_elastic, ids[k], -volume * stress0 * w)
    determinants[t] = j


@wp.kernel
def vertex_metrics(
    q: wp.array[wp.vec3],
    q0: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    mass: wp.array[float],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    elastic: wp.array[wp.vec3d],
    damping: wp.array[wp.vec3d],
    initial_elastic: wp.array[wp.vec3d],
    dt: wp.float64,
    metrics: wp.array2d[wp.float64],
):
    v = wp.tid()
    for k in range(9):
        metrics[v, k] = wp.float64(0.0)
    if (flags[v] & 1) != 0 and inv_mass[v] > 0.0:
        inertial = wp.float64(mass[v]) / (dt * dt) * (wp.vec3d(target[v]) - wp.vec3d(q[v]))
        initial_inertial = wp.float64(mass[v]) / (dt * dt) * (wp.vec3d(target[v]) - wp.vec3d(q0[v]))
        r = elastic[v] + damping[v] + inertial
        r0 = initial_elastic[v] + initial_inertial
        metrics[v, 0] = wp.dot(r, r)
        metrics[v, 1] = wp.dot(r0, r0)
        metrics[v, 2] = wp.dot(elastic[v], elastic[v])
        metrics[v, 3] = wp.dot(damping[v], damping[v])
        metrics[v, 4] = wp.dot(inertial, inertial)
        metrics[v, 5] = wp.float64(1.0)
        metrics[v, 6] = wp.length(r)
        metrics[v, 7] = wp.length(wp.vec3d(velocity[v]))
        metrics[v, 8] = wp.dot(wp.vec3d(q[v]) - wp.vec3d(q0[v]), wp.vec3d(q[v]) - wp.vec3d(q0[v]))


@wp.kernel
def finish_metrics(
    metrics: wp.array2d[wp.float64],
    determinants: wp.array[wp.float64],
    contacts: wp.array[int],
    counter: wp.array[int],
    rows: wp.array2d[wp.float64],
):
    # One deterministic serial reduction per step; the tet assembly is parallel.
    row = counter[0]
    if row < rows.shape[0]:
        for k in range(11):
            rows[row, k] = wp.float64(0.0)
        for v in range(metrics.shape[0]):
            for k in range(6):
                rows[row, k] += metrics[v, k]
            rows[row, 6] = wp.max(rows[row, 6], metrics[v, 6])
            rows[row, 7] = wp.max(rows[row, 7], metrics[v, 7])
            rows[row, 8] += metrics[v, 8]
        j_min = wp.float64(1.0e30)
        for t in range(determinants.shape[0]):
            j_min = wp.min(j_min, determinants[t])
        rows[row, 9] = j_min
        if contacts:
            rows[row, 10] = wp.float64(contacts[0])
    counter[0] = row + 1


def numpy_forces(q, q0, indices, poses, materials, dt):
    """Float64 reference, also used to finite-difference the scalar objective."""
    q, q0, poses, materials = [np.asarray(a, dtype=np.float64) for a in (q, q0, poses, materials)]
    ds = np.stack([q[indices[:, k]] - q[indices[:, 0]] for k in (1, 2, 3)], axis=2)
    ds0 = np.stack([q0[indices[:, k]] - q0[indices[:, 0]] for k in (1, 2, 3)], axis=2)
    f, f0 = ds @ poses, ds0 @ poses
    j = np.linalg.det(f)
    cof = np.stack(
        [np.cross(f[:, :, 1], f[:, :, 2]), np.cross(f[:, :, 2], f[:, :, 0]), np.cross(f[:, :, 0], f[:, :, 1])], axis=2
    )
    mu, kp, kd = materials[:, 0], materials[:, 0] + materials[:, 1], materials[:, 2]
    volume = 1 / (6 * np.linalg.det(poses))
    delta_c = np.swapaxes(f, 1, 2) @ f - np.swapaxes(f0, 1, 2) @ f0
    p = mu[:, None, None] * f + (kp * (j - 1) - mu)[:, None, None] * cof
    pd = (2 * kd / dt)[:, None, None] * (f @ delta_c)
    w = np.concatenate([-poses.sum(axis=1)[:, None], poses], axis=1)
    fe, fd = np.zeros_like(q), np.zeros_like(q)
    for k in range(4):
        np.add.at(fe, indices[:, k], -volume[:, None] * np.einsum("eij,ej->ei", p, w[:, k]))
        np.add.at(fd, indices[:, k], -volume[:, None] * np.einsum("eij,ej->ei", pd, w[:, k]))
    energy = np.sum(
        volume
        * (
            0.5 * mu * (np.sum(f * f, axis=(1, 2)) - 3)
            - mu * (j - 1)
            + 0.5 * kp * (j - 1) ** 2
            + kd / (2 * dt) * np.sum(delta_c * delta_c, axis=(1, 2))
        )
    )
    return fe, fd, float(energy)
