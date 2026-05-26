# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gauss-Newton reduced-coordinate projection."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.reduced_projection import project_to_reduced_coordinates
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_double_pendulum(device):
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

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    j2 = builder.add_joint_revolute(
        parent=b1,
        child=b2,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j1, j2], label="pendulum")

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
    project_to_reduced_coordinates(model, state, gn_iterations=3)

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
    project_to_reduced_coordinates(model, state, gn_iterations=5)

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
    project_to_reduced_coordinates(model, state, gn_iterations=0)

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
    project_to_reduced_coordinates(model, state, gn_iterations=5)

    # Verify FK consistency
    state_check = model.state()
    state_check.joint_q.assign(state.joint_q)
    newton.eval_fk(model, state_check.joint_q, state_check.joint_qd, state_check)

    body_q_proj = state.body_q.numpy().reshape(-1, 7)
    body_q_check = state_check.body_q.numpy().reshape(-1, 7)
    np.testing.assert_allclose(body_q_proj, body_q_check, atol=1e-5)


class TestReducedProjection(unittest.TestCase):
    pass


devices = get_test_devices()

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


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
