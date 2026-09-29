# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent energy-gradient and native-force checks for the bag residual."""

import numpy as np
import warp as wp
from cloth_residual import bend_forces, triangle_forces

import newton
from newton._src.solvers.vbd.particle_alm_kernels import ParticleElasticityAlmState
from newton._src.solvers.vbd.particle_vbd_kernels import (
    evaluate_dihedral_angle_based_bending_force_hessian,
    evaluate_neo_hookean_membrane_force_hessian,
)


@wp.kernel
def native_forces(
    q: wp.array[wp.vec3],
    q0: wp.array[wp.vec3],
    ids: wp.array2d[int],
    poses: wp.array[wp.mat22],
    areas: wp.array[float],
    materials: wp.array2d[float],
    edges: wp.array2d[int],
    angles: wp.array[float],
    lengths: wp.array[float],
    props: wp.array2d[float],
    dt: float,
    output: wp.array[wp.vec3],
):
    v = wp.tid()
    force = wp.vec3(0.0)
    for t in range(ids.shape[0]):
        for c in range(3):
            if ids[t, c] == v:
                f, _h = evaluate_neo_hookean_membrane_force_hessian(
                    t, c, q, q0, ids, poses[t], areas[t], materials[t, 0], materials[t, 1], materials[t, 2], dt
                )
                force += f
    for e in range(edges.shape[0]):
        for c in range(4):
            if edges[e, c] == v:
                f, _h = evaluate_dihedral_angle_based_bending_force_hessian(
                    e, c, q, q0, edges, angles, lengths, props[e, 0], props[e, 1], dt
                )
                force += f
    output[v] = force


def validate(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rest = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
    for point in rest:
        builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=1.0)
    for tri in ((0, 2, 3), (1, 3, 2)):
        builder.add_triangle(*tri, tri_ke=12.0, tri_ka=18.0, tri_kd=0.1)
    builder.add_edge(0, 1, 2, 3, rest=0.0, edge_ke=20.0, edge_kd=0.02)
    model = builder.finalize(device=device)
    q = rest + np.array([[0.1, 0.2, 0.1], [0.0, -0.1, 0.3], [0.02, 0.0, 0.03], [0.1, 0.05, -0.1]], np.float32)
    points = wp.array(q, dtype=wp.vec3, device=device)
    elastic = wp.zeros(4, dtype=wp.vec3d, device=device)
    damping = wp.zeros_like(elastic)
    gap = wp.zeros_like(elastic)
    dt = float(np.float32(0.01))
    empty = ParticleElasticityAlmState()
    wp.launch(
        triangle_forces,
        2,
        inputs=[
            points,
            model.particle_q,
            model.tri_indices,
            model.tri_poses,
            model.tri_areas,
            model.tri_materials,
            wp.float64(dt),
            empty,
            elastic,
            damping,
            gap,
        ],
        device=device,
    )
    wp.launch(
        bend_forces,
        1,
        inputs=[
            points,
            model.particle_q,
            model.edge_indices,
            model.edge_rest_angle,
            model.edge_rest_length,
            model.edge_bending_properties,
            wp.float64(dt),
            empty,
            elastic,
            damping,
            gap,
        ],
        device=device,
    )
    actual = elastic.numpy() + damping.numpy()
    native = wp.zeros(4, dtype=wp.vec3, device=device)
    wp.launch(
        native_forces,
        4,
        inputs=[
            points,
            model.particle_q,
            model.tri_indices,
            model.tri_poses,
            model.tri_areas,
            model.tri_materials,
            model.edge_indices,
            model.edge_rest_angle,
            model.edge_rest_length,
            model.edge_bending_properties,
            dt,
            native,
        ],
        device=device,
    )
    ids, poses, areas = model.tri_indices.numpy(), model.tri_poses.numpy().astype(float), model.tri_areas.numpy()

    def energy(x):
        value = 0.0
        for t, indices in enumerate(ids):
            f = np.column_stack((x[indices[1]] - x[indices[0]], x[indices[2]] - x[indices[0]])) @ poses[t]
            f0 = np.column_stack((rest[indices[1]] - rest[indices[0]], rest[indices[2]] - rest[indices[0]])) @ poses[t]
            j = np.linalg.norm(np.cross(f[:, 0], f[:, 1]))
            dc = f.T @ f - f0.T @ f0
            value += areas[t] * (
                6.0 * (np.sum(f * f) - 2.0)
                - 12.0 * (j - 1.0)
                + 15.0 * (j - 1.0) ** 2
                + 0.1 / (2 * dt) * np.sum(dc * dc)
            )
        n0 = np.cross(x[2] - x[0], x[3] - x[0])
        n0 /= np.linalg.norm(n0)
        n1 = np.cross(x[3] - x[1], x[2] - x[1])
        n1 /= np.linalg.norm(n1)
        edge = x[3] - x[2]
        edge /= np.linalg.norm(edge)
        angle = np.arctan2(np.dot(np.cross(n0, n1), edge), np.dot(n0, n1))
        return value + 10.0 * angle**2 + 0.01 / dt * angle**2

    expected = np.zeros((4, 3))
    x = q.astype(float)
    for v in range(4):
        for c in range(3):
            delta = np.zeros_like(x)
            delta[v, c] = 1.0e-5
            expected[v, c] = -(energy(x + delta) - energy(x - delta)) / (2.0e-5)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-7, atol=2.0e-7)
    np.testing.assert_allclose(actual, native.numpy(), rtol=3.0e-5, atol=3.0e-5)
    assert np.count_nonzero(gap.numpy()) == 0
    return {
        "energy_gradient_relative_error": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
        "native_force_relative_error": float(np.linalg.norm(actual - native.numpy()) / np.linalg.norm(actual)),
    }


if __name__ == "__main__":
    wp.init()
    print(validate(wp.get_device()))
