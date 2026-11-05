# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The suture contact gate: gated particle-shape pairs produce no force and no DAT plane."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices
from newton.tests.utils.soft_contacts import evaluate_soft_contact_force_mask

CAPSULE_RADIUS = 2.0e-3
CAPSULE_HALF_HEIGHT = 1.0e-2
PARTICLE_RADIUS = 1.0e-3
DT = 1.0 / 60.0


def _scene(device, *, particle_x, second_capsule=False, soft_contact_gap=0.0, dat=False, velocity_x=0.0):
    """One particle beside a capsule that is a suture needle; optionally a second ungated capsule."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    SolverVBD.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    needle = builder.add_body(xform=wp.transform(wp.vec3(0.0), wp.quat_identity()))
    capsule = builder.add_shape_capsule(needle, radius=CAPSULE_RADIUS, half_height=CAPSULE_HALF_HEIGHT, cfg=cfg)
    builder.add_suture_needle(needle, capsule, centerline=[wp.vec3(0.0, 0.0, 0.01), wp.vec3(0.0, 0.0, -0.01)])
    if second_capsule:
        other = builder.add_body(xform=wp.transform(wp.vec3(2.0 * particle_x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_capsule(other, radius=CAPSULE_RADIUS, half_height=CAPSULE_HALF_HEIGHT, cfg=cfg)
    builder.add_particle(pos=wp.vec3(particle_x, 0.0, 0.0), vel=wp.vec3(velocity_x, 0.0, 0.0), mass=1.0e-3, radius=PARTICLE_RADIUS)
    builder.color()
    model = builder.finalize(skip_validation_inertia=True)
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=soft_contact_gap)
    contacts = pipeline.contacts()
    solver = SolverVBD(model, iterations=10, rigid_soft_enable_dat=dat, collision_pipeline=pipeline if dat else None)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    return model, pipeline, contacts, solver, state_0, state_1, control


def _step(model, pipeline, contacts, solver, state_0, state_1, control):
    if solver.collision_pipeline is not None:
        # The solver owns the pipeline (needed by rigid-soft DAT) and detects into solver.contacts.
        solver.step(state_0, state_1, control, None, DT)
    else:
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
    return state_1.particle_q.numpy()[0]


def test_gated_particle_receives_no_contact_force(test, device):
    with wp.ScopedDevice(device):
        for gated in (True, False):
            model, pipeline, contacts, solver, state_0, state_1, control = _scene(device, particle_x=2.5e-3)
            if gated:
                model.particle_contact_gate.assign(np.array([1], dtype=np.int32))
            pipeline.collide(state_0, contacts)
            test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 1)
            mask = evaluate_soft_contact_force_mask(model, state_0, contacts).numpy()
            test.assertEqual(bool(mask[0]), not gated)
            q = _step(model, pipeline, contacts, solver, state_0, state_1, control)
            if gated:
                np.testing.assert_allclose(q, [2.5e-3, 0.0, 0.0], atol=1.0e-9)
            else:
                test.assertGreater(q[0], 2.5e-3 + 1.0e-5)


def test_gate_is_per_pair(test, device):
    with wp.ScopedDevice(device):
        model, pipeline, contacts, solver, state_0, state_1, control = _scene(device, particle_x=2.5e-3, second_capsule=True)
        q_ungated = _step(model, pipeline, contacts, solver, state_0, state_1, control)
        test.assertLess(abs(q_ungated[0] - 2.5e-3), 1.0e-6)
        model, pipeline, contacts, solver, state_0, state_1, control = _scene(device, particle_x=2.5e-3, second_capsule=True)
        model.particle_contact_gate.assign(np.array([1], dtype=np.int32))
        q_gated = _step(model, pipeline, contacts, solver, state_0, state_1, control)
        test.assertLess(q_gated[0], 2.5e-3 - 1.0e-5)


def test_gated_pair_skips_dat(test, device):
    with wp.ScopedDevice(device):
        for gated in (True, False):
            model, pipeline, contacts, solver, state_0, state_1, control = _scene(
                device, particle_x=8.0e-3, soft_contact_gap=1.0e-2, dat=True, velocity_x=-1.0
            )
            if gated:
                model.particle_contact_gate.assign(np.array([1], dtype=np.int32))
            q = _step(model, pipeline, contacts, solver, state_0, state_1, control)
            if gated:
                test.assertLess(q[0], 0.0)
            else:
                test.assertGreaterEqual(q[0], CAPSULE_RADIUS - 1.0e-4)


devices = get_test_devices()


class TestSutureGate(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


add_function_test(TestSutureGate, "test_gated_particle_receives_no_contact_force", test_gated_particle_receives_no_contact_force, devices=devices)
add_function_test(TestSutureGate, "test_gate_is_per_pair", test_gate_is_per_pair, devices=devices)
add_function_test(TestSutureGate, "test_gated_pair_skips_dat", test_gated_pair_skips_dat, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
