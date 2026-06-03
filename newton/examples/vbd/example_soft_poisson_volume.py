# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Poisson Volume
#
# Validates the volumetric (Poisson) response of the Neo-Hookean material in
# the small-deformation regime, where linear elasticity is exact. A short
# beam is pinned at the left face and stretched a small amount (~2%) at the
# right, with its sides free to contract. For a uniaxial stress the relative
# volume change is
#
#     dV/V = (1 - 2*nu) * eps_axial,   nu = lambda / (2*(lambda + mu))
#
# i.e. the volume change is governed entirely by Poisson's ratio: nu -> 0.5 is
# incompressible (no volume change), nu -> 0 stretches without lateral
# contraction. The test sweeps nu via k_lambda / k_mu and checks that the
# Poisson ratio implied by the measured volume change matches the material's
# nu across the whole compressibility range. Measurements use the central
# region of the beam, away from the laterally-constrained end faces.
#
# Command: python -m newton.examples vbd.example_soft_poisson_volume
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags

# Shared material/geometry. The stable Neo-Hookean material is calibrated to
# these linear Lame parameters, so nu = lambda / (2*(lambda + mu)).
_DENSITY = 1000.0
_K_MU = 1.0e4
_DIM_X = 16
_DIM_YZ = 4
_CELL = 0.05
_STRETCH = 1.02  # small (~2%) axial stretch, deep in the linear-elastic regime

# Poisson sweep: vary k_lambda (with k_mu fixed) to span compressible -> incompressible.
_LAMBDA_SWEEP = (1.0e3, 1.0e4, 3.0e4, 1.0e5, 5.0e5)  # nu ~ 0.045, 0.25, 0.375, 0.455, 0.49


def _poisson_ratio(k_mu: float, k_lambda: float) -> float:
    return k_lambda / (2.0 * (k_lambda + k_mu))


def _tet_volume(q: np.ndarray, tet_indices: np.ndarray) -> float:
    v0, v1, v2, v3 = q[tet_indices[:, 0]], q[tet_indices[:, 1]], q[tet_indices[:, 2]], q[tet_indices[:, 3]]
    vols = np.einsum("ij,ij->i", v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0
    return float(np.sum(np.abs(vols)))


def _run_poisson(
    k_mu: float, k_lambda: float, iterations: int, ramp_frames: int = 60, settle_frames: int = 120
) -> tuple[float, float]:
    """Stretch a beam ~5% axially with free lateral contraction; return (dV/V, eps_axial)
    measured over the central region (away from the laterally-fixed end faces)."""
    builder = newton.ModelBuilder()
    rest_x = _DIM_X * _CELL
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=_DIM_X,
        dim_y=_DIM_YZ,
        dim_z=_DIM_YZ,
        cell_x=_CELL,
        cell_y=_CELL,
        cell_z=_CELL,
        density=_DENSITY,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=1.0e-3,
        fix_left=True,
    )
    builder.color()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))

    q0 = model.particle_q.numpy()
    right_idx = np.where(np.abs(q0[:, 0] - rest_x) < 1e-6)[0]
    flags = model.particle_flags.numpy()
    for i in right_idx:
        flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags)

    solver = newton.solvers.SolverVBD(model=model, iterations=iterations, particle_enable_self_contact=False)
    s0, s1 = model.state(), model.state()
    ctrl, contacts = model.control(), model.contacts()
    dt = 1.0 / 60 / 6

    tet = model.tet_indices.numpy()
    centroid_x = q0[tet].mean(axis=1)[:, 0]
    central = (centroid_x > 0.3 * rest_x) & (centroid_x < 0.7 * rest_x)
    rest_vol_central = _tet_volume(q0, tet[central])

    # two central cross-sections to measure the central axial strain
    layer_x = np.unique(np.round(q0[:, 0], 6))
    xa = layer_x[np.argmin(np.abs(layer_x - 0.35 * rest_x))]
    xb = layer_x[np.argmin(np.abs(layer_x - 0.65 * rest_x))]
    a_idx = np.where(np.abs(q0[:, 0] - xa) < 1e-6)[0]
    b_idx = np.where(np.abs(q0[:, 0] - xb) < 1e-6)[0]
    rest_gap = xb - xa

    target_x = _STRETCH * rest_x
    for f in range(ramp_frames + settle_frames):
        r = min(f / max(ramp_frames - 1, 1), 1.0)
        q = s0.particle_q.numpy()
        q[right_idx, 0] = rest_x + r * (target_x - rest_x)
        s0.particle_q.assign(q)
        model.collide(s0, contacts)
        for _ in range(6):
            s0.clear_forces()
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0

    q = s0.particle_q.numpy()
    dv_over_v = _tet_volume(q, tet[central]) / rest_vol_central - 1.0
    eps_axial = (float(np.mean(q[b_idx, 0])) - float(np.mean(q[a_idx, 0]))) / rest_gap - 1.0
    return dv_over_v, eps_axial


class Example:
    # iterations scale with axial length so the stretch converges along the beam.
    ITERATIONS = 5 * _DIM_X
    # absolute tolerance on the implied Poisson ratio vs the material value.
    NU_TOLERANCE = 0.03

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = self.ITERATIONS
        self.sim_time = 0.0
        self.ramp_frames = 60
        self._frame_index = 0

        # Visual: a single mid-range Poisson beam (k_lambda = k_mu -> nu = 0.25) being stretched.
        builder = newton.ModelBuilder()
        self.rest_x = _DIM_X * _CELL
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=_DIM_X,
            dim_y=_DIM_YZ,
            dim_z=_DIM_YZ,
            cell_x=_CELL,
            cell_y=_CELL,
            cell_z=_CELL,
            density=_DENSITY,
            k_mu=_K_MU,
            k_lambda=_K_MU,
            k_damp=1.0e-3,
            fix_left=True,
        )
        builder.color()
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        q_np = self.model.particle_q.numpy()
        self.right_indices = np.where(np.abs(q_np[:, 0] - self.rest_x) < 1e-6)[0]
        flags = self.model.particle_flags.numpy()
        for i in self.right_indices:
            flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags)

        self.solver = newton.solvers.SolverVBD(
            model=self.model, iterations=self.iterations, particle_enable_self_contact=False
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _apply_stretch(self):
        r = min(self._frame_index / max(self.ramp_frames - 1, 1), 1.0)
        target_x = self.rest_x + r * (_STRETCH - 1.0) * self.rest_x
        q = self.state_0.particle_q.numpy()
        q[self.right_indices, 0] = target_x
        self.state_0.particle_q.assign(q)

    def step(self):
        self._apply_stretch()
        self.simulate()
        self.sim_time += self.frame_dt
        self._frame_index += 1

    def test_final(self):
        # Sweep Poisson's ratio and check the implied nu (from the measured volume
        # change) matches the material nu across the compressibility range.
        results = []
        for k_lambda in _LAMBDA_SWEEP:
            nu = _poisson_ratio(_K_MU, k_lambda)
            dv_over_v, eps = _run_poisson(_K_MU, k_lambda, iterations=self.ITERATIONS)
            nu_implied = 0.5 * (1.0 - dv_over_v / eps)
            results.append((nu, nu_implied, dv_over_v, eps))
            if abs(nu_implied - nu) > self.NU_TOLERANCE:
                raise ValueError(
                    f"Poisson volume mismatch at nu={nu:.3f}: implied {nu_implied:.3f} "
                    f"(dV/V={dv_over_v:+.4f}, eps={eps:.4f}, tol={self.NU_TOLERANCE})"
                )

        # Sanity: volume change must shrink monotonically as nu increases toward 0.5.
        dvs = [r[2] for r in results]
        if not all(dvs[i] > dvs[i + 1] for i in range(len(dvs) - 1)):
            raise ValueError("dV/V did not decrease monotonically with increasing Poisson ratio: " + str(dvs))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
