# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for particle elasticity ALM in SolverVBD."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _add_tet(builder, *, mu=1000.0, lame=2000.0, pos=(0.0, 0.0, 0.0)):
    builder.add_soft_mesh(
        pos=wp.vec3(pos),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)],
        indices=[0, 1, 2, 3],
        density=24.0,
        k_mu=mu,
        k_lambda=lame,
        k_damp=0.0,
    )


def _tet_model(device, *, pinned=False, mu=1000.0, lame=2000.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _add_tet(builder, mu=mu, lame=lame)
    if pinned:
        for i in range(3):
            builder.particle_mass[i] = 0.0
    builder.color()
    return builder.finalize(device=device)


def _alm_solver(test, model, **kwargs):
    try:
        return newton.solvers.SolverVBD(model, particle_elasticity_alm=True, **kwargs)
    except TypeError as error:
        test.fail(f"Particle elasticity ALM must be available in SolverVBD: {error}")


def _seeded_rest(test, device):
    """Opposed tet stress memories must preserve a rotated rest pose."""
    model = _tet_model(device)
    angle = np.float32(0.6)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    initial = model.particle_q.numpy() @ rotation.T + np.array([0.3, 0.2, -0.1], dtype=np.float32)
    for deviatoric in (False, True):
        with test.subTest(deviatoric=deviatoric):
            solver = _alm_solver(test, model, iterations=2, particle_elasticity_alm_deviatoric=deviatoric)
            state, output = model.state(), model.state()
            state.particle_q.assign(initial)
            for _ in range(20):
                solver.step(state, output, None, None, 1.0 / 120.0)
                state, output = output, state
            np.testing.assert_allclose(state.particle_q.numpy(), initial, atol=3.0e-6, rtol=0.0)
            np.testing.assert_allclose(state.particle_qd.numpy(), 0.0, atol=4.0e-5, rtol=0.0)


def _loaded_tet(test, device):
    """Full and pressure-only ALM must reach the same implicit material response."""
    model = _tet_model(device, pinned=True)
    dt = 0.1
    # The apex has mass 1. Its constrained elastic stiffness is
    # V0*(mu + K_pressure) = (1000 + 3000)/6.
    expected_z = 1.0 - 1.0 / (1.0 / dt**2 + 4000.0 / 6.0)
    for deviatoric in (False, True):
        with test.subTest(deviatoric=deviatoric):
            solver = _alm_solver(test, model, iterations=80, particle_elasticity_alm_deviatoric=deviatoric)
            state, output = model.state(), model.state()
            force = np.zeros((4, 3), dtype=np.float32)
            force[3, 2] = -1.0
            state.particle_f.assign(force)
            solver.step(state, output, None, None, dt)
            np.testing.assert_allclose(output.particle_q.numpy()[:3], model.particle_q.numpy()[:3], atol=1.0e-7)
            test.assertAlmostEqual(float(output.particle_q.numpy()[3, 2]), expected_z, delta=3.0e-6)


def _loaded_spring(test, device):
    """Converge an ALM spring to its authored implicit stiffness."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), mass=0.0)
    builder.add_particle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), mass=1.0)
    builder.add_spring(0, 1, ke=1000.0, kd=0.0, control=0.0)
    builder.color()
    model = builder.finalize(device=device)
    solver = _alm_solver(test, model, iterations=30)
    state, output = model.state(), model.state()
    state.particle_f.assign(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    solver.step(state, output, None, None, 0.1)
    np.testing.assert_allclose(output.particle_q.numpy()[0], 0.0, atol=1.0e-7)
    test.assertAlmostEqual(float(output.particle_q.numpy()[1, 0]), 1.0 + 1.0 / 1100.0, delta=1.0e-6)


def _reset_selected_world(test, device):
    """Rebaseline selected tet histories from post-reset edits and retain other worlds."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for _ in range(2):
        builder.begin_world()
        _add_tet(builder)
        builder.end_world()
    _add_tet(builder)  # A global element must obey the mask's final entry.
    builder.color()
    model = builder.finalize(device=device)
    solvers = [_alm_solver(test, model, iterations=2, particle_elasticity_alm_deviatoric=True) for _ in range(3)]
    states = [model.state() for _ in range(3)]
    outputs = [model.state() for _ in range(3)]
    for solver, state, output in zip(solvers[:2], states[:2], outputs[:2], strict=True):
        q = model.particle_q.numpy().copy()
        q[3::4, 2] *= 0.8
        state.particle_q.assign(q)
        for _ in range(3):
            solver.step(state, output, None, None, 0.01)
    mask = wp.array([True, False, True], dtype=wp.bool, device=device)
    solvers[0].reset(states[0], world_mask=mask, flags=0)
    edited = model.particle_q.numpy().copy()
    edited[3::4, 2] *= 0.95
    for state in states:
        state.particle_q.assign(edited)
        state.particle_qd.zero_()
    for solver, state, output in zip(solvers, states, outputs, strict=True):
        solver.step(state, output, None, None, 0.01)
    selected = model.particle_world.numpy() != 1
    np.testing.assert_allclose(
        outputs[0].particle_q.numpy()[selected], outputs[2].particle_q.numpy()[selected], atol=2e-7
    )
    np.testing.assert_array_equal(outputs[0].particle_q.numpy()[~selected], outputs[1].particle_q.numpy()[~selected])
    test.assertGreater(
        np.max(np.abs(outputs[0].particle_q.numpy()[selected] - outputs[1].particle_q.numpy()[selected])), 1e-6
    )


def _captured_steps_and_reset(test, device):
    """Replay ALM steps and reset with stable addresses and changing device masks."""
    model = _tet_model(device, pinned=True)
    for deviatoric in (False, True):
        with test.subTest(deviatoric=deviatoric):
            eager = _alm_solver(test, model, iterations=3, particle_elasticity_alm_deviatoric=deviatoric)
            captured = _alm_solver(test, model, iterations=3, particle_elasticity_alm_deviatoric=deviatoric)
            eager_states = [model.state(), model.state()]
            graph_states = [model.state(), model.state()]
            force = np.zeros((4, 3), dtype=np.float32)
            force[3, 2] = -2.0
            mask = wp.zeros(model.world_count + 1, dtype=wp.bool, device=device)
            for solver, states in ((eager, eager_states), (captured, graph_states)):
                for state in states:
                    state.particle_f.assign(force)
                solver.step(states[0], states[1], None, None, 0.02)
                solver.reset(states[0])
            with wp.ScopedCapture(device=device) as capture:
                captured.reset(graph_states[0], world_mask=mask, flags=0)
                captured.step(graph_states[0], graph_states[1], None, None, 0.02)
                captured.step(graph_states[1], graph_states[0], None, None, 0.02)
            for replay in range(8):
                mask.assign(np.full(model.world_count + 1, replay == 4, dtype=bool))
                eager.reset(eager_states[0], world_mask=mask, flags=0)
                eager.step(eager_states[0], eager_states[1], None, None, 0.02)
                eager.step(eager_states[1], eager_states[0], None, None, 0.02)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(graph_states[0].particle_q.numpy(), eager_states[0].particle_q.numpy())
                np.testing.assert_array_equal(graph_states[0].particle_qd.numpy(), eager_states[0].particle_qd.numpy())


def _tile_matches_scalar(test, device):
    """Match scalar and tile ALM for tet and mixed membrane, hinge, and spring meshes."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    points = [(0.0, 1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    for point in points:
        builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=1.0)
    builder.add_triangle(0, 2, 3, tri_ke=50.0, tri_ka=100.0, tri_kd=0.0)
    builder.add_triangle(1, 3, 2, tri_ke=50.0, tri_ka=100.0, tri_kd=0.0)
    builder.add_edge(0, 1, 2, 3, edge_ke=100.0, edge_kd=0.0)
    builder.add_spring(0, 1, ke=100.0, kd=0.0, control=0.0)
    builder.color()
    for model in (_tet_model(device), builder.finalize(device=device)):
        for deviatoric in (False, True):
            results = []
            for tiled in (False, True):
                solver = _alm_solver(
                    test,
                    model,
                    iterations=5,
                    particle_enable_tile_solve=tiled,
                    particle_elasticity_alm_deviatoric=deviatoric,
                )
                state, output = model.state(), model.state()
                q = model.particle_q.numpy().copy()
                q[0, 2] += 0.1
                state.particle_q.assign(q)
                for _ in range(4):
                    solver.step(state, output, None, None, 0.02)
                    state, output = output, state
                results.append(state.particle_q.numpy())
            np.testing.assert_allclose(results[0], results[1], atol=2.0e-6, rtol=2.0e-6)


def _unsupported_restart(test, device):
    """Reject a coupling restart that would advance retained stress twice."""
    model = _tet_model(device)
    solver = _alm_solver(test, model)
    with test.assertRaisesRegex(ValueError, "repeated-interval proxy coupling"):
        solver.coupling_notify_input_state_update(model.state(), 0, iteration_restart=True, dt=0.01)


def _default_preserves_rotation(test, device):
    """Keep a rotated rest tet force-free when using the default pressure-only mode."""
    model = _tet_model(device)
    solver = _alm_solver(test, model, iterations=1)
    state, output = model.state(), model.state()
    solver.step(state, output, None, None, 0.01)
    theta = 0.2
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rotated = model.particle_q.numpy() @ rotation.T
    state.particle_q.assign(rotated)
    state.particle_qd.zero_()
    solver.step(state, output, None, None, 0.01)
    np.testing.assert_allclose(output.particle_q.numpy(), rotated, atol=2e-6, rtol=0.0)


def _legacy_dat_with_alm(test, device):
    """Stop a moving ALM tet before it crosses a fixed triangle using existing DAT."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for point in [(-2.0, -2.0, 0.0), (5.0, -2.0, 0.0), (-2.0, 5.0, 0.0)]:
        builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=0.0)
    builder.add_triangle(0, 1, 2, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0)
    _add_tet(builder, pos=(0.0, 0.0, 0.1))
    for i in range(3, 7):
        builder.particle_qd[i] = wp.vec3(0.0, 0.0, -5.0)
    builder.color()
    model = builder.finalize(device=device)
    solver = _alm_solver(
        test,
        model,
        iterations=5,
        particle_enable_self_contact=True,
        particle_self_contact_margin=0.03,
        particle_self_contact_gap=0.2,
    )
    state, output = model.state(), model.state()
    solver.step(state, output, None, None, 0.1)
    positions = output.particle_q.numpy()
    test.assertTrue(np.isfinite(positions).all())
    np.testing.assert_array_equal(positions[:3], model.particle_q.numpy()[:3])
    test.assertGreater(float(positions[3:, 2].min()), 0.0)
    test.assertLess(float(positions[3:, 2].min()), 0.1)


class TestSolverVBDElasticityALM(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestSolverVBDElasticityALM, "test_seeded_rest", _seeded_rest, devices=devices)
add_function_test(TestSolverVBDElasticityALM, "test_loaded_tet", _loaded_tet, devices=devices)
add_function_test(TestSolverVBDElasticityALM, "test_loaded_spring", _loaded_spring, devices=devices)
add_function_test(TestSolverVBDElasticityALM, "test_reset_selected_world", _reset_selected_world, devices=devices)
add_function_test(TestSolverVBDElasticityALM, "test_unsupported_restart", _unsupported_restart, devices=devices)
add_function_test(
    TestSolverVBDElasticityALM, "test_default_preserves_rotation", _default_preserves_rotation, devices=devices
)
add_function_test(TestSolverVBDElasticityALM, "test_legacy_dat_with_alm", _legacy_dat_with_alm, devices=devices)
cuda_devices = [device for device in devices if device.is_cuda]
add_function_test(
    TestSolverVBDElasticityALM, "test_captured_steps_and_reset", _captured_steps_and_reset, devices=cuda_devices
)
add_function_test(TestSolverVBDElasticityALM, "test_tile_matches_scalar", _tile_matches_scalar, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main()
