# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gauss-Newton reduced-coordinate projection."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.reduced_projection import ReducedCoordinateProjection
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


def _build_double_pendulum(device, joint_limits=None):
    """Build a 2-link revolute pendulum and return (model, state)."""
    builder = newton.ModelBuilder()

    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    b2 = builder.add_link(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )

    limit_kwargs = {}
    if joint_limits is not None:
        limit_kwargs = {"limit_lower": joint_limits[0], "limit_upper": joint_limits[1]}

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        **limit_kwargs,
    )
    j2 = builder.add_joint_revolute(
        parent=b1,
        child=b2,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        **limit_kwargs,
    )
    builder.add_articulation([j1, j2], label="pendulum")

    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    # Set non-trivial joint angles
    joint_q = state.joint_q.numpy()
    joint_q[0] = 0.5
    joint_q[1] = 0.3
    state.joint_q.assign(joint_q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    return model, state


def test_projection_recovers_fk_poses(test, device):
    """After FK → projection, joint_q and body_q should be consistent."""
    model, state = _build_double_pendulum(device)

    # Save ground truth
    joint_q_gt = state.joint_q.numpy().copy()
    body_q_gt = state.body_q.numpy().copy()

    # Project (should be near no-op on already-consistent state)
    projection = ReducedCoordinateProjection(model, gn_iterations=3)
    projection.project(state, dt=1.0)

    joint_q_proj = state.joint_q.numpy()
    body_q_proj = state.body_q.numpy()

    np.testing.assert_allclose(joint_q_proj, joint_q_gt, atol=1e-4)
    np.testing.assert_allclose(body_q_proj, body_q_gt, atol=1e-4)


def test_projection_corrects_perturbed_bodies(test, device):
    """Perturb body_q away from kinematic manifold, verify projection snaps back."""
    model, state = _build_double_pendulum(device)

    # Perturb body_q (break kinematic consistency)
    body_q_np = state.body_q.numpy().copy().reshape(-1, 7)
    body_q_np[1, 0] += 0.05  # shift second body's x position
    body_q_np[1, 1] += 0.03  # shift second body's y position
    state.body_q.assign(wp.array(body_q_np.flatten(), dtype=wp.transform, device=device))

    # Project
    projection = ReducedCoordinateProjection(model, gn_iterations=5)
    projection.project(state, dt=1.0)

    # After projection, body_q should satisfy FK exactly
    state_check = model.state()
    state_check.joint_q.assign(state.joint_q)
    newton.eval_fk(model, state_check.joint_q, state_check.joint_qd, state_check)

    body_q_proj = state.body_q.numpy().reshape(-1, 7)
    body_q_check = state_check.body_q.numpy().reshape(-1, 7)
    np.testing.assert_allclose(body_q_proj, body_q_check, atol=1e-5)


def test_projection_analytical_only(test, device):
    """With gn_iterations=0, projection uses eval_ik → eval_fk only."""
    model, state = _build_double_pendulum(device)

    # Perturb body_q
    body_q_np = state.body_q.numpy().copy().reshape(-1, 7)
    body_q_np[1, 0] += 0.05
    state.body_q.assign(wp.array(body_q_np.flatten(), dtype=wp.transform, device=device))

    # Analytical projection
    projection = ReducedCoordinateProjection(model, gn_iterations=0)
    projection.project(state, dt=1.0)

    # Body_q should be kinematically consistent (FK round-trip)
    state_check = model.state()
    state_check.joint_q.assign(state.joint_q)
    newton.eval_fk(model, state_check.joint_q, state_check.joint_qd, state_check)

    body_q_proj = state.body_q.numpy().reshape(-1, 7)
    body_q_check = state_check.body_q.numpy().reshape(-1, 7)
    np.testing.assert_allclose(body_q_proj, body_q_check, atol=1e-5)


def test_projection_multi_joint_chain(test, device):
    """Test projection on a 5-link revolute chain."""
    builder = newton.ModelBuilder()

    bodies = []
    joints = []
    n_links = 5
    for i in range(n_links):
        b = builder.add_link(
            xform=wp.transform(wp.vec3(float(i), 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        bodies.append(b)
        parent_body = bodies[i - 1] if i > 0 else -1
        parent_xform = (
            wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())
            if i > 0
            else wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        )
        j = builder.add_joint_revolute(
            parent=parent_body,
            child=b,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=parent_xform,
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        joints.append(j)

    builder.add_articulation(joints, label="chain")
    model = builder.finalize(device=device)
    state = model.state()

    # Set varied joint angles
    joint_q = state.joint_q.numpy()
    for i in range(n_links):
        joint_q[i] = 0.2 * (i + 1)
    state.joint_q.assign(joint_q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Perturb all bodies
    rng = np.random.default_rng(42)
    body_q_np = state.body_q.numpy().copy().reshape(-1, 7)
    body_q_np[:, 0] += rng.normal(0, 0.02, n_links)
    body_q_np[:, 1] += rng.normal(0, 0.02, n_links)
    state.body_q.assign(wp.array(body_q_np.flatten(), dtype=wp.transform, device=device))

    # Project
    projection = ReducedCoordinateProjection(model, gn_iterations=5)
    projection.project(state, dt=1.0)

    # Verify FK consistency
    state_check = model.state()
    state_check.joint_q.assign(state.joint_q)
    newton.eval_fk(model, state_check.joint_q, state_check.joint_qd, state_check)

    body_q_proj = state.body_q.numpy().reshape(-1, 7)
    body_q_check = state_check.body_q.numpy().reshape(-1, 7)
    np.testing.assert_allclose(body_q_proj, body_q_check, atol=1e-5)


def test_projection_skips_free_articulation(test, device):
    """FREE-joint articulations must keep their maximal body_q bit-exact."""
    builder = newton.ModelBuilder()

    # Managed articulation: single revolute link.
    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1], label="arm")

    # Unmanaged articulation: free-floating body.
    b2 = builder.add_link(
        xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    j2 = builder.add_joint_free(child=b2)
    builder.add_articulation([j2], label="floater")

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    projection = ReducedCoordinateProjection(model, gn_iterations=3)
    test.assertEqual(projection.managed_articulation_count, 1)

    # Perturb both bodies off their kinematic configuration.
    body_q_np = state.body_q.numpy().copy().reshape(-1, 7)
    body_q_np[0, 0] += 0.05
    body_q_np[1, 0] += 0.31
    body_q_np[1, 2] += 0.17
    state.body_q.assign(wp.array(body_q_np.flatten(), dtype=wp.transform, device=device))
    free_body_q_before = state.body_q.numpy().reshape(-1, 7)[1].copy()

    projection.project(state, dt=1.0)

    body_q_after = state.body_q.numpy().reshape(-1, 7)

    # The free body's maximal pose is untouched (bit-exact).
    np.testing.assert_array_equal(body_q_after[1], free_body_q_before)

    # The revolute link is projected back onto its manifold: its position must
    # lie on the joint's rotation circle through the origin (x-offset removed).
    state_check = model.state()
    state_check.joint_q.assign(state.joint_q)
    newton.eval_fk(model, state_check.joint_q, state_check.joint_qd, state_check)
    np.testing.assert_allclose(body_q_after[0], state_check.body_q.numpy().reshape(-1, 7)[0], atol=1e-5)


def test_projection_clamps_to_joint_limits(test, device):
    """Projected joint coordinates must respect finite joint limits."""
    model, state = _build_double_pendulum(device, joint_limits=(-0.4, 0.4))

    # The initial configuration (0.5, 0.3) already violates joint 0's upper
    # limit; the projection must clamp it back into range.
    projection = ReducedCoordinateProjection(model, gn_iterations=3)
    projection.project(state, dt=1.0)

    joint_q = state.joint_q.numpy()
    test.assertLessEqual(joint_q[0], 0.4 + 1e-6)
    test.assertGreaterEqual(joint_q[0], -0.4 - 1e-6)
    test.assertLessEqual(joint_q[1], 0.4 + 1e-6)

    # body_q must be FK-consistent with the clamped joint_q.
    state_check = model.state()
    state_check.joint_q.assign(state.joint_q)
    newton.eval_fk(model, state_check.joint_q, state_check.joint_qd, state_check)
    np.testing.assert_allclose(
        state.body_q.numpy().reshape(-1, 7),
        state_check.body_q.numpy().reshape(-1, 7),
        atol=1e-5,
    )


def test_projection_clamps_velocity(test, device):
    """Recovered joint velocities are clamped to max_joint_vel."""
    model, state = _build_double_pendulum(device)

    # Give the first body a large angular velocity about the joint axis.
    body_qd_np = state.body_qd.numpy().copy().reshape(-1, 6)
    body_qd_np[0, 5] = 100.0  # omega_z, world frame
    state.body_qd.assign(wp.array(body_qd_np.flatten(), dtype=wp.spatial_vector, device=device))

    projection = ReducedCoordinateProjection(model, gn_iterations=2, max_joint_vel=5.0)
    projection.project(state, dt=1.0 / 60.0)

    joint_qd = state.joint_qd.numpy()
    test.assertTrue(np.all(np.abs(joint_qd) <= 5.0 + 1e-6))


def test_projection_clamps_coord_delta(test, device):
    """The per-step coordinate change is limited to max_joint_vel * dt."""
    model, state = _build_double_pendulum(device)

    dt = 1.0 / 60.0
    max_vel = 5.0
    projection = ReducedCoordinateProjection(model, gn_iterations=3, max_joint_vel=max_vel)
    # joint_q_prev is the model's initial config (zeros); the state sits at
    # (0.5, 0.3), so the projection may move each coordinate at most
    # max_vel * dt from zero.
    projection.project(state, dt=dt)

    joint_q = state.joint_q.numpy()
    test.assertTrue(np.all(np.abs(joint_q) <= max_vel * dt + 1e-6))


def test_projection_graph_capture(test, device):
    """SolverVBD with RVBD enabled must be CUDA-graph capturable."""
    model, state_0 = _build_double_pendulum(device)
    state_1 = model.state()
    wp.copy(state_1.body_q, state_0.body_q)
    control = model.control()

    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        body_enable_reduced_solve=True,
        reduced_gn_iterations=2,
    )

    dt = 1.0 / 240.0

    # Warm up (module load + lazy buffer allocation) outside capture.
    solver.step(state_0, state_1, control, None, dt)
    solver.step(state_1, state_0, control, None, dt)

    with wp.ScopedCapture(device) as capture:
        solver.step(state_0, state_1, control, None, dt)
        solver.step(state_1, state_0, control, None, dt)

    for _ in range(3):
        wp.capture_launch(capture.graph)

    joint_q = state_0.joint_q.numpy()
    body_q = state_0.body_q.numpy()
    test.assertTrue(np.all(np.isfinite(joint_q)))
    test.assertTrue(np.all(np.isfinite(body_q)))

    # body_q stays FK-consistent with joint_q after graph replays.
    state_check = model.state()
    state_check.joint_q.assign(state_0.joint_q)
    newton.eval_fk(model, state_check.joint_q, state_check.joint_qd, state_check)
    np.testing.assert_allclose(
        body_q.reshape(-1, 7),
        state_check.body_q.numpy().reshape(-1, 7),
        atol=1e-5,
    )


class TestReducedProjection(unittest.TestCase):
    pass


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

add_function_test(
    TestReducedProjection,
    "test_projection_recovers_fk_poses",
    test_projection_recovers_fk_poses,
    devices=devices,
)
add_function_test(
    TestReducedProjection,
    "test_projection_corrects_perturbed_bodies",
    test_projection_corrects_perturbed_bodies,
    devices=devices,
)
add_function_test(
    TestReducedProjection,
    "test_projection_analytical_only",
    test_projection_analytical_only,
    devices=devices,
)
add_function_test(
    TestReducedProjection,
    "test_projection_multi_joint_chain",
    test_projection_multi_joint_chain,
    devices=devices,
)
add_function_test(
    TestReducedProjection,
    "test_projection_skips_free_articulation",
    test_projection_skips_free_articulation,
    devices=devices,
)
add_function_test(
    TestReducedProjection,
    "test_projection_clamps_to_joint_limits",
    test_projection_clamps_to_joint_limits,
    devices=devices,
)
add_function_test(
    TestReducedProjection,
    "test_projection_clamps_velocity",
    test_projection_clamps_velocity,
    devices=devices,
)
add_function_test(
    TestReducedProjection,
    "test_projection_clamps_coord_delta",
    test_projection_clamps_coord_delta,
    devices=devices,
)
add_function_test(
    TestReducedProjection,
    "test_projection_graph_capture",
    test_projection_graph_capture,
    devices=cuda_devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
