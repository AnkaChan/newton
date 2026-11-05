# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the ALM-DAT self-contact path of SolverVBD."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _two_layer_drop_model(device, layer_gap=0.012):
    """A fixed flat cloth at z = 1 with a smaller free cloth hovering above its interior."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cell = 0.05
    for layer, grid in enumerate((12, 8)):
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * grid * cell, -0.5 * grid * cell, 1.0 + layer * layer_gap),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=grid,
            dim_y=grid,
            cell_x=cell,
            cell_y=cell,
            mass=0.1,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-1,
            edge_ke=1.0e-2,
            particle_radius=0.005,
        )
    lower_count = 13 * 13
    for i in range(lower_count):
        builder.particle_mass[i] = 0.0
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e-2
    model.soft_contact_mu = 0.2
    return model, lower_count


def _solver(model, method, **kwargs):
    return newton.solvers.SolverVBD(
        model,
        iterations=4,
        particle_enable_self_contact=True,
        particle_self_contact_margin=0.02,
        particle_self_contact_gap=0.01,
        particle_self_contact_method=method,
        **kwargs,
    )


def test_self_contact_method_is_validated(test, device):
    """Reject unknown self-contact methods at construction."""
    model, _lower = _two_layer_drop_model(device)
    with test.assertRaises(ValueError):
        _solver(model, "bogus")


def _drop_lowest_heights(test, device, method, steps=90):
    """Step the drop scene and return the per-step lowest free-vertex height, asserting no penetration."""
    # start above the 0.03 query radius so the upper layer really falls onto the fixed one
    model, lower_count = _two_layer_drop_model(device, layer_gap=0.06)
    solver = _solver(model, method)
    state, output = model.state(), model.state()
    dt = 1.0 / 60.0
    lowest = []
    for _ in range(steps):
        solver.step(state, output, None, None, dt)
        state, output = output, state
        z = state.particle_q.numpy()[lower_count:, 2]
        lowest.append(float(z.min()))
        test.assertGreaterEqual(float(z.min()), 1.0 - 1.0e-6)
    return lowest


def test_alm_dat_drop_never_penetrates_the_fixed_layer(test, device):
    """Keep every free vertex above the fixed layer on every step while still letting it settle."""
    lowest = _drop_lowest_heights(test, device, "alm_dat")
    # it came down and rests within the penalty margin of the fixed layer
    test.assertLess(min(lowest), 1.0 + 0.025)
    test.assertLess(lowest[-1], 1.0 + 0.025)


def test_penalty_dat_drop_still_works(test, device):
    """Keep the default per-color DAT path intact on the same scene."""
    lowest = _drop_lowest_heights(test, device, "penalty_dat")
    test.assertLess(min(lowest), 1.0 + 0.025)
    test.assertLess(lowest[-1], 1.0 + 0.025)


def _pinch_model(device):
    """A free strip between a fixed bottom plate at z = 1 and a fixed top plate 3 cm above it."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cell = 0.05
    specs = ((10, 1.0, 0.0), (6, 1.015, 0.1), (10, 1.03, 0.0))  # (grid, z, mass)
    for grid, z, mass in specs:
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * grid * cell, -0.5 * grid * cell, z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=grid,
            dim_y=grid,
            cell_x=cell,
            cell_y=cell,
            mass=mass,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-1,
            edge_ke=1.0e-2,
            particle_radius=0.005,
        )
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e-2
    model.soft_contact_mu = 0.2
    counts = [(g + 1) ** 2 for g, _z, _m in specs]
    return model, counts


def test_alm_dat_pinched_strip_never_crosses_the_fixed_plate(test, device):
    """Keep a strip above the fixed bottom plate while the top plate is pressed 2 cm toward it."""
    model, (n_bottom, n_strip, _n_top) = _pinch_model(device)
    # the closing pinch needs more than the default average contact budget per element
    solver = _solver(model, "alm_dat", particle_vertex_contact_buffer_size=16, particle_edge_contact_buffer_size=64)
    state, output = model.state(), model.state()
    dt = 1.0 / 60.0
    top = slice(n_bottom + n_strip, None)
    base_top = model.particle_q.numpy()[top].copy()
    for step in range(120):
        q = state.particle_q.numpy()
        q[top] = base_top + np.array([0.0, 0.0, -0.02 * min(step, 60) / 60.0], dtype=np.float32)
        state.particle_q.assign(q)
        solver.step(state, output, None, None, dt)
        state, output = output, state
        test.assertTrue(solver.particle_dat_coverage_ok(), f"contact buffer overflowed at step {step}")
        z = state.particle_q.numpy()[n_bottom : n_bottom + n_strip, 2]
        test.assertGreaterEqual(float(z.min()), 1.0 - 1.0e-6, f"strip crossed the bottom plate at step {step}")


def test_alm_dat_reports_lost_coverage_on_buffer_overflow(test, device):
    """Flag the step as uncertified when the shared contact buffer drops pairs, and recover after growth."""
    model, _counts = _pinch_model(device)
    solver = _solver(model, "alm_dat", particle_vertex_contact_buffer_size=1, particle_edge_contact_buffer_size=1)
    state, output = model.state(), model.state()
    dt = 1.0 / 60.0
    solver.step(state, output, None, None, dt)
    test.assertFalse(solver.particle_dat_coverage_ok())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grew = solver.check_and_grow_self_contact_buffers()
    test.assertTrue(grew)
    solver.step(output, state, None, None, dt)
    test.assertTrue(solver.particle_dat_coverage_ok())


def test_alm_dat_step_is_graph_capturable(test, device):
    """Replay a captured alm_dat step and keep the free layer above the fixed one."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    model, lower_count = _two_layer_drop_model(device, layer_gap=0.06)
    solver = _solver(model, "alm_dat")
    state, output = model.state(), model.state()
    dt = 1.0 / 60.0
    # warm up kernels before capture
    solver.step(state, output, None, None, dt)
    solver.step(output, state, None, None, dt)
    with wp.ScopedCapture(device) as capture:
        solver.step(state, output, None, None, dt)
        solver.step(output, state, None, None, dt)
    for _ in range(40):
        wp.capture_launch(capture.graph)
    z = state.particle_q.numpy()[lower_count:, 2]
    test.assertTrue(np.all(np.isfinite(state.particle_q.numpy())))
    test.assertGreaterEqual(float(z.min()), 1.0 - 1.0e-6)
    test.assertLess(float(z.min()), 1.0 + 0.025)


devices = get_test_devices()


def _schedule_types():
    from newton._src.solvers.solver import SolverBase  # noqa: PLC0415

    return SolverBase.CollisionSlot, SolverBase.CollisionFrequencyType


def test_alm_dat_default_schedule_detects_twice(test, device):
    """alm_dat follows the host's default soft self-contact schedule: a second detection after initialization."""
    Slot, Freq = _schedule_types()
    model, _lower = _two_layer_drop_model(device)
    solver = _solver(model, "alm_dat")
    test.assertEqual(solver._default_collision_frequency_type(Slot.SOFT_SELF_CONTACT), Freq.PRE_POST_INIT)


def test_alm_dat_rejects_in_iteration_detection(test, device):
    """Only PRE_INIT and PRE_POST_INIT keep the certified anchor; ITERATIONS is refused at the step."""
    Slot, Freq = _schedule_types()
    model, _lower = _two_layer_drop_model(device)
    solver = _solver(model, "alm_dat")
    solver.set_collision_frequency(
        collision_frequency_type={Slot.SOFT_SELF_CONTACT: Freq.ITERATIONS},
        collision_frequency={Slot.SOFT_SELF_CONTACT: 1},
    )
    state0, state1, control = model.state(), model.state(), model.control()
    with test.assertRaises(ValueError):
        solver.step(state0, state1, control, None, 1.0 / 60.0)


def _free_sheet_model(device, speed):
    """One free cloth sheet without gravity, translating at ``speed`` along +z, no pairs within reach."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, speed),
        dim_x=6,
        dim_y=6,
        cell_x=0.05,
        cell_y=0.05,
        mass=0.1,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=1.0e-1,
        edge_ke=1.0e-2,
        particle_radius=0.005,
    )
    builder.color()
    return builder.finalize(device=device)


def test_alm_dat_second_detection_doubles_the_admitted_motion(test, device):
    """With one detection the committed motion per step is one budget; with two it is up to two budgets."""
    Slot, Freq = _schedule_types()
    dt = 1.0 / 60.0
    budget = 0.5 * 0.85 * (0.02 + 0.01)  # margin + gap of _solver, relaxation 0.85
    speed = 1.5 * budget / dt  # the free flight asks for one and a half budgets per step
    admitted = {}
    for schedule in (Freq.PRE_INIT, Freq.PRE_POST_INIT):
        model = _free_sheet_model(device, speed)
        solver = _solver(model, "alm_dat")
        solver.set_collision_frequency(collision_frequency_type={Slot.SOFT_SELF_CONTACT: schedule})
        state0, state1, control = model.state(), model.state(), model.control()
        q0 = state0.particle_q.numpy().copy()
        solver.step(state0, state1, control, None, dt)
        q1 = state1.particle_q.numpy()
        moved = np.linalg.norm(q1 - q0, axis=1)
        test.assertTrue(np.isfinite(moved).all())
        admitted[schedule] = float(np.median(moved))
    test.assertAlmostEqual(admitted[Freq.PRE_INIT], budget, delta=0.05 * budget)
    test.assertAlmostEqual(admitted[Freq.PRE_POST_INIT], 1.5 * budget, delta=0.05 * budget)


class TestSolverVBDAlmDat(unittest.TestCase):
    pass


add_function_test(
    TestSolverVBDAlmDat,
    "test_self_contact_method_is_validated",
    test_self_contact_method_is_validated,
    devices=devices,
)
add_function_test(
    TestSolverVBDAlmDat,
    "test_alm_dat_drop_never_penetrates_the_fixed_layer",
    test_alm_dat_drop_never_penetrates_the_fixed_layer,
    devices=devices,
)
add_function_test(
    TestSolverVBDAlmDat,
    "test_penalty_dat_drop_still_works",
    test_penalty_dat_drop_still_works,
    devices=devices,
)
add_function_test(
    TestSolverVBDAlmDat,
    "test_alm_dat_pinched_strip_never_crosses_the_fixed_plate",
    test_alm_dat_pinched_strip_never_crosses_the_fixed_plate,
    devices=devices,
)
add_function_test(
    TestSolverVBDAlmDat,
    "test_alm_dat_reports_lost_coverage_on_buffer_overflow",
    test_alm_dat_reports_lost_coverage_on_buffer_overflow,
    devices=devices,
)
add_function_test(
    TestSolverVBDAlmDat,
    "test_alm_dat_step_is_graph_capturable",
    test_alm_dat_step_is_graph_capturable,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)

add_function_test(
    TestSolverVBDAlmDat,
    "test_alm_dat_default_schedule_detects_twice",
    test_alm_dat_default_schedule_detects_twice,
    devices=devices,
)
add_function_test(
    TestSolverVBDAlmDat,
    "test_alm_dat_rejects_in_iteration_detection",
    test_alm_dat_rejects_in_iteration_detection,
    devices=devices,
)
add_function_test(
    TestSolverVBDAlmDat,
    "test_alm_dat_second_detection_doubles_the_admitted_motion",
    test_alm_dat_second_detection_doubles_the_admitted_motion,
    devices=devices,
)
