import numpy as np
import warp as wp

from newton.core.builder import ModelBuilder
from newton.solvers.vbd.solver_vbd import evaluate_volumetric_neo_hooken_force_and_hessian_4_vertices, mat43, vec9


@wp.func
def assemble_tet_vertex_force(
    dE_dF: vec9,
    m1: float,
    m2: float,
    m3: float,
):
    f = wp.vec3(
        -(dE_dF[0] * m1 + dE_dF[3] * m2 + dE_dF[6] * m3),
        -(dE_dF[1] * m1 + dE_dF[4] * m2 + dE_dF[7] * m3),
        -(dE_dF[2] * m1 + dE_dF[5] * m2 + dE_dF[8] * m3),
    )

    return f


@wp.kernel
def compute_neo_hookean_energy_and_force_and_hessian(
    # inputs
    tet_id: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    f1, f2, f3, f4, h1, h2, h3, h4 = evaluate_volumetric_neo_hooken_force_and_hessian_4_vertices(
        tet_id,
        pos,  # dont need damping
        pos,
        tet_indices,
        tet_poses,
        tet_materials,
        dt,
    )

    particle_forces[tet_indices[tet_id, 0]] = f1
    particle_forces[tet_indices[tet_id, 1]] = f2
    particle_forces[tet_indices[tet_id, 2]] = f3
    particle_forces[tet_indices[tet_id, 3]] = f4

    particle_hessians[tet_indices[tet_id, 0]] = h1
    particle_hessians[tet_indices[tet_id, 1]] = h2
    particle_hessians[tet_indices[tet_id, 2]] = h3
    particle_hessians[tet_indices[tet_id, 3]] = h4


@wp.kernel
def compute_neo_hookean_energy_and_force(
    # inputs
    tet_id: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    # outputs: particle force and hessian
    tet_energy: wp.array(dtype=float),
    particle_forces: wp.array(dtype=float),
):
    v0_idx = tet_indices[tet_id, 0]
    v1_idx = tet_indices[tet_id, 1]
    v2_idx = tet_indices[tet_id, 2]
    v3_idx = tet_indices[tet_id, 3]

    mu = tet_materials[tet_id, 0]
    lmbd = tet_materials[tet_id, 1]

    v0 = pos[v0_idx]
    v1 = pos[v1_idx]
    v2 = pos[v2_idx]
    v3 = pos[v3_idx]

    Dm_inv = tet_poses[tet_id]
    rest_volume = 1.0 / (wp.determinant(Dm_inv) * 6.0)

    diff_1 = v1 - v0
    diff_2 = v2 - v0
    diff_3 = v3 - v0
    Ds = wp.mat33(
        diff_1[0],
        diff_2[0],
        diff_3[0],
        diff_1[1],
        diff_2[1],
        diff_3[1],
        diff_1[2],
        diff_2[2],
        diff_3[2],
    )

    F = Ds * Dm_inv

    a = 1.0 + mu / lmbd

    det_F = wp.determinant(F)

    E = rest_volume * 0.5 * (mu * (wp.trace(F * wp.transpose(F)) - 3.0) + lmbd * (det_F - a) * (det_F - a))
    tet_energy[tet_id] = E

    F1_1 = F[0, 0]
    F2_1 = F[1, 0]
    F3_1 = F[2, 0]
    F1_2 = F[0, 1]
    F2_2 = F[1, 1]
    F3_2 = F[2, 1]
    F1_3 = F[0, 2]
    F2_3 = F[1, 2]
    F3_3 = F[2, 2]

    dPhi_D_dF = vec9(
        F1_1,
        F2_1,
        F3_1,
        F1_2,
        F2_2,
        F3_2,
        F1_3,
        F2_3,
        F3_3,
    )

    ddetF_dF = vec9(
        F2_2 * F3_3 - F2_3 * F3_2,
        F1_3 * F3_2 - F1_2 * F3_3,
        F1_2 * F2_3 - F1_3 * F2_2,
        F2_3 * F3_1 - F2_1 * F3_3,
        F1_1 * F3_3 - F1_3 * F3_1,
        F1_3 * F2_1 - F1_1 * F2_3,
        F2_1 * F3_2 - F2_2 * F3_1,
        F1_2 * F3_1 - F1_1 * F3_2,
        F1_1 * F2_2 - F1_2 * F2_1,
    )

    k = det_F - a
    dPhi_D_dF = dPhi_D_dF * mu
    dPhi_H_dF = ddetF_dF * lmbd * k

    dE_dF = (dPhi_D_dF + dPhi_H_dF) * rest_volume

    Dm_inv_1_1 = Dm_inv[0, 0]
    Dm_inv_2_1 = Dm_inv[1, 0]
    Dm_inv_3_1 = Dm_inv[2, 0]
    Dm_inv_1_2 = Dm_inv[0, 1]
    Dm_inv_2_2 = Dm_inv[1, 1]
    Dm_inv_3_2 = Dm_inv[2, 1]
    Dm_inv_1_3 = Dm_inv[0, 2]
    Dm_inv_2_3 = Dm_inv[1, 2]
    Dm_inv_3_3 = Dm_inv[2, 2]

    ms = mat43(
        -Dm_inv_1_1 - Dm_inv_2_1 - Dm_inv_3_1,
        -Dm_inv_1_2 - Dm_inv_2_2 - Dm_inv_3_2,
        -Dm_inv_1_3 - Dm_inv_2_3 - Dm_inv_3_3,
        Dm_inv_1_1,
        Dm_inv_1_2,
        Dm_inv_1_3,
        Dm_inv_2_1,
        Dm_inv_2_2,
        Dm_inv_2_3,
        Dm_inv_3_1,
        Dm_inv_3_2,
        Dm_inv_3_3,
    )

    for v_counter in range(4):
        f = assemble_tet_vertex_force(dE_dF, ms[v_counter, 0], ms[v_counter, 1], ms[v_counter, 2])
        particle_forces[tet_indices[tet_id, v_counter] * 3 + 0] = f[0]
        particle_forces[tet_indices[tet_id, v_counter] * 3 + 1] = f[1]
        particle_forces[tet_indices[tet_id, v_counter] * 3 + 2] = f[2]


def test_tet_energy():
    rng = np.random.default_rng(seed=42)

    for _test in range(100):
        builder = ModelBuilder()

        vertices = [wp.vec3(rng.standard_normal((3,))) for _ in range(4)]

        p = np.array(vertices[0])
        q = np.array(vertices[1])
        r = np.array(vertices[2])
        s = np.array(vertices[3])

        qp = q - p
        rp = r - p
        sp = s - p

        Dm = np.array((qp, rp, sp)).T
        volume = np.linalg.det(Dm) / 6.0

        if volume < 0:
            vertices = [
                vertices[1],
                vertices[0],
                vertices[2],
                vertices[3],
            ]

        tet_indices = [0, 1, 2, 3]

        builder.add_soft_mesh(
            vertices=vertices,
            indices=tet_indices,
            rot=wp.quat_identity(),
            pos=wp.vec3(0.0),
            vel=wp.vec3(0.0),
            density=1000.0,
            scale=1.0,
            k_mu=rng.standard_normal(),
            k_lambda=rng.standard_normal(),
            k_damp=0.0,
        )
        dt = 0.001666

        model = builder.finalize(requires_grad=True)
        tet_energy = wp.zeros(1, dtype=float, requires_grad=True)
        particle_forces = wp.zeros(12, dtype=float, requires_grad=True)
        particle_hessian = wp.zeros(4, dtype=wp.mat33, requires_grad=False)

        state = model.state(requires_grad=True)
        state.particle_q.assign(state.particle_q.numpy() + rng.standard_normal((4, 3)))

        with wp.Tape() as tape:
            wp.launch(
                dim=1,
                kernel=compute_neo_hookean_energy_and_force,
                inputs=[
                    0,
                    dt,
                    state.particle_q,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    tet_energy,
                    particle_forces,
                ],
            )

        tape.backward(tet_energy)

        particle_force_auto_diff = -state.particle_q.grad.numpy()
        particle_forces_analytical_1 = particle_forces.numpy().copy().reshape(4, -1)
        assert (np.isclose(particle_force_auto_diff, particle_forces_analytical_1, rtol=1.0e-4, atol=0.1)).all()

        # calculate hessians using auto diff
        particle_hessian_auto_diff = np.zeros((4, 3, 3), dtype=np.float32)

        def onehot(i, ndim):
            x = np.zeros(ndim, dtype=np.float32)
            x[i] = 1.0
            return wp.array(
                x,
            )

        for v_counter in range(4):
            for dim in range(3):
                tape.zero()
                tape.backward(grads={particle_forces: onehot(v_counter * 3 + dim, 12)})
                # force is the negative gradient so the hessian is the negative jacobian of it
                particle_hessian_auto_diff[v_counter, dim, :] = -state.particle_q.grad.numpy()[v_counter, :]

        particle_forces_vec3 = wp.zeros_like(state.particle_q)
        wp.launch(
            dim=1,
            kernel=compute_neo_hookean_energy_and_force_and_hessian,
            inputs=[
                0,
                dt,
                state.particle_q,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                particle_forces_vec3,
                particle_hessian,
            ],
        )
        particle_forces_analytical_2 = particle_forces_vec3.numpy()
        particle_hessian_analytical = particle_hessian.numpy()
        assert (np.isclose(particle_forces_analytical_2, particle_forces_analytical_1, rtol=1.0e-4, atol=0.1)).all()

        for i in range(4):
            print(
                "autodiff hessian:\n",
                particle_hessian_auto_diff[i],
                "\nanalytical hessian: \n",
                particle_hessian_analytical[i],
            )


if __name__ == "__main__":
    test_tet_energy()
