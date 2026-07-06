# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Genesis-style soft-transform-constraint coupling strategy."""

import unittest
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled.interface import CouplingInterface
from newton._src.solvers.coupled.solver_coupled_soft_constraint import (
    harvest_proxy_softconstraint_forces_kernel,
)
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import (
    SolverCoupled,
    SolverCoupledSoftConstraint,
)


def _spatial(force, torque):
    return np.array([*force, *torque], dtype=np.float64)


class TestSoftConstraintHarvestKernel(unittest.TestCase):
    """The harvest kernel must reproduce Genesis' reduced reaction (math doc section 4)."""

    def _run_kernel(self, target_q, solved_q, mass, inertia, mapping, dt):
        n = len(mapping)
        out = wp.zeros(max(max(mapping) + 1, 1), dtype=wp.spatial_vector, device="cpu")
        wp.launch(
            harvest_proxy_softconstraint_forces_kernel,
            dim=n,
            inputs=[
                float(dt),
                wp.array(mapping, dtype=int, device="cpu"),
                wp.array(target_q, dtype=wp.transform, device="cpu"),
                wp.array(solved_q, dtype=wp.transform, device="cpu"),
                wp.array(mass, dtype=float, device="cpu"),
                wp.array(inertia, dtype=wp.mat33, device="cpu"),
                out,
            ],
            device="cpu",
        )
        return out.numpy()

    def test_linear_reduced_force(self):
        # F = m / dt^2 * (p_solved - p_target), no rotation.
        dt = 0.02
        m = 7.0
        dp = np.array([0.01, -0.02, 0.03])
        target = [wp.transform(wp.vec3(0.5, 0.0, 0.3), wp.quat_identity())]
        solved = [wp.transform(wp.vec3(*(np.array([0.5, 0.0, 0.3]) + dp)), wp.quat_identity())]
        out = self._run_kernel(target, solved, [m], [wp.mat33(np.eye(3))], [0], dt)
        expected_f = m / dt**2 * dp
        np.testing.assert_allclose(out[0][:3], expected_f, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(out[0][3:], np.zeros(3), atol=1e-5)

    def test_rotational_reduced_torque(self):
        # tau = I_world / dt^2 * log(R_solved R_target^T), target at identity so
        # R_solved = exp(theta) and I_world = R_solved I R_solved^T.
        dt = 0.02
        I = np.diag([2.0, 3.0, 4.0])
        angle = 0.05
        axis = np.array([0.0, 0.0, 1.0])
        target = [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())]
        r_solved = wp.quat_from_axis_angle(wp.vec3(*axis), angle)
        solved = [wp.transform(wp.vec3(0.0, 0.0, 0.0), r_solved)]
        out = self._run_kernel(target, solved, [1.0], [wp.mat33(I)], [0], dt)
        rot = np.array(wp.quat_to_matrix(r_solved)).reshape(3, 3)
        theta = angle * axis
        i_world = rot @ I @ rot.T
        expected_tau = i_world @ theta / dt**2
        np.testing.assert_allclose(out[0][:3], np.zeros(3), atol=1e-5)
        np.testing.assert_allclose(out[0][3:], expected_tau, atol=1e-3)

    def test_unmapped_bodies_are_skipped(self):
        # global id -1 -> no contribution.
        dt = 0.02
        target = [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())] * 2
        solved = [wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())] * 2
        out = self._run_kernel(target, solved, [1.0, 1.0], [wp.mat33(np.eye(3))] * 2, [-1, 0], dt)
        # Only local id 1 (global 0) contributes; the kernel writes index 0.
        self.assertGreater(abs(out[0][0]), 0.0)


# ----------------------------------------------------------------------
# Framework integration with a fake destination solver
# ----------------------------------------------------------------------
@wp.kernel(enable_backward=False)
def _add_translation_kernel(delta: wp.vec3, q: wp.array[wp.transform]):
    i = wp.tid()
    t = q[i]
    q[i] = wp.transform(wp.transform_get_translation(t) + delta, wp.transform_get_rotation(t))


class _SoftConstraintProbeSolver(SolverBase, CouplingInterface):
    """Destination solver exposing an AVBD-like inertial target ``body_inertia_q``.

    Each step it records the commanded pose as the spring target and displaces
    the solved pose by a fixed deviation, so the soft-constraint harvest sees a
    known ``solved - target`` and the reduced reaction is analytic.
    """

    instances: ClassVar[list] = []
    DEVIATION = (0.01, -0.02, 0.03)

    def __init__(self, model):
        super().__init__(model)
        self.body_inertia_q = wp.clone(model.body_q)
        self.instances.append(self)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)
        # Spring target = commanded (synced) pose; solved pose = target + deviation.
        wp.copy(self.body_inertia_q, state_in.body_q)
        wp.launch(
            _add_translation_kernel,
            dim=state_out.body_q.shape[0],
            inputs=[wp.vec3(*self.DEVIATION), state_out.body_q],
            device=self.model.device,
        )


class _BodyForceRecorder(SolverBase, CouplingInterface):
    """Source solver that records the proxy feedback it receives in ``body_f``."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.recorded = []
        self.instances.append(self)

    def coupling_notify_input_state_update(self, state, flags, *, iteration_restart=False, dt=0.0):
        del flags, iteration_restart, dt
        if state.body_f is not None:
            self.recorded.append(state.body_f.numpy().copy())

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)


class TestSoftConstraintCoupler(unittest.TestCase):
    def _build(self, eta_p=100.0, eta_a=50.0, own_mass=0.5):
        _SoftConstraintProbeSolver.instances.clear()
        _BodyForceRecorder.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        # body 0: source (real finger); body 1: dst-owned; body 2: proxy of body 0.
        builder.add_body(mass=own_mass, inertia=wp.mat33(np.diag([0.2, 0.3, 0.4])))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=own_mass, inertia=wp.mat33(np.diag([0.2, 0.3, 0.4])))
        model = builder.finalize(device="cpu")
        coupled = SolverCoupledSoftConstraint(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_BodyForceRecorder, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_SoftConstraintProbeSolver, bodies=[1]),
            ],
            coupling=SolverCoupledProxy_Config(proxies=[("src", "dst", [0], [2])]),
            constraint_strength_translation=eta_p,
            constraint_strength_rotation=eta_a,
        )
        return model, coupled

    def test_installs_eta_scaled_proxy_mass(self):
        eta_p, own_mass = 100.0, 0.5
        _, coupled = self._build(eta_p=eta_p, own_mass=own_mass)
        view = coupled.view("dst")
        masses = view.body_mass.numpy()
        # The proxy body (global id 2) carries eta_p * own_mass.
        self.assertTrue(np.any(np.isclose(masses, eta_p * own_mass)), f"masses={masses}")

    def test_reaction_from_pose_deviation(self):
        eta_p, own_mass = 100.0, 0.5
        dt = 0.02
        model, coupled = self._build(eta_p=eta_p, own_mass=own_mass)
        state_0, state_1 = model.state(), model.state()
        # Run two steps: the harvested reaction is fed to the source on the next step.
        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)
        recorder = _BodyForceRecorder.instances[-1]
        self.assertTrue(recorder.recorded, "source never received force input")
        received = recorder.recorded[-1]  # (num_src_bodies, 6)
        dev = np.array(_SoftConstraintProbeSolver.DEVIATION)
        expected_f = (eta_p * own_mass) / dt**2 * dev
        np.testing.assert_allclose(received[0][:3], expected_f, rtol=1e-3, atol=1e-2)


def SolverCoupledProxy_Config(proxies):
    """Build a proxy Config from (src, dst, bodies, proxy_bodies) tuples."""
    from newton.solvers.experimental.coupled import SolverCoupledProxy

    return SolverCoupledProxy.Config(
        proxies=[
            SolverCoupledProxy.Proxy(source=s, destination=d, bodies=b, proxy_bodies=p) for (s, d, b, p) in proxies
        ]
    )


if __name__ == "__main__":
    unittest.main()
