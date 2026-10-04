# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the slack-gap ALM self-contact row used by the alm_dat path."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.particle_dat_kernels import alm_contact_row
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _contact_row_kernel(
    gap: wp.array[float],
    lam: wp.array[float],
    stiffness: float,
    radius: float,
    rho: float,
    force: wp.array[float],
    hessian: wp.array[float],
):
    i = wp.tid()
    t, k = alm_contact_row(gap[i], lam[i], stiffness, radius, rho)
    force[i] = t
    hessian[i] = k


def _run_contact_row(device, gaps, lams, stiffness=10.0, radius=0.5, rho=30.0):
    n = len(gaps)
    force = wp.zeros(n, dtype=float, device=device)
    hessian = wp.zeros(n, dtype=float, device=device)
    wp.launch(
        _contact_row_kernel,
        dim=n,
        inputs=[
            wp.array(gaps, dtype=float, device=device),
            wp.array(lams, dtype=float, device=device),
            stiffness,
            radius,
            rho,
        ],
        outputs=[force, hessian],
        device=device,
    )
    return force.numpy(), hessian.numpy()


def test_contact_row_reproduces_the_cushion_law_at_its_fixed_point(test, device):
    """Return the target cushion force K (R - C) when the multiplier already equals it."""
    stiffness, radius, rho = 10.0, 0.5, 30.0
    gaps = [0.7, 0.3, 0.1, 0.0]
    lams = [max(0.0, stiffness * (radius - c)) for c in gaps]
    force, hessian = _run_contact_row(device, gaps, lams, stiffness, radius, rho)
    np.testing.assert_allclose(force, [0.0, 2.0, 4.0, 5.0], atol=1.0e-5)
    k_eff = stiffness * rho / (stiffness + rho)
    np.testing.assert_allclose(hessian, [0.0, k_eff, k_eff, k_eff], atol=1.0e-5)


def test_contact_row_switches_to_the_hard_wall_past_the_plane(test, device):
    """Grow the force by rho times the penetration once the slack is pinned at zero."""
    stiffness, radius, rho = 10.0, 0.5, 30.0
    # multiplier at the cushion's full load; a penetration of 0.1 puts the slack at zero
    force, hessian = _run_contact_row(device, [-0.1], [stiffness * radius], stiffness, radius, rho)
    np.testing.assert_allclose(force, [stiffness * radius + rho * 0.1], atol=1.0e-5)
    np.testing.assert_allclose(hessian, [rho], atol=1.0e-5)
    # the force is continuous across the switch: just inside the cushion with the same multiplier
    force_c, _ = _run_contact_row(device, [1.0e-6], [stiffness * radius], stiffness, radius, rho)
    test.assertAlmostEqual(float(force_c[0]), stiffness * radius, places=3)


def test_contact_row_releases_when_far_from_the_plane(test, device):
    """Transmit nothing and zero the multiplier when the pair is outside the cushion."""
    force, hessian = _run_contact_row(device, [0.6, 2.0], [0.0, 3.0], 10.0, 0.5, 30.0)
    np.testing.assert_allclose(force, [0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(hessian, [0.0, 0.0], atol=1.0e-6)


def _vertex_over_fixed_cloth(device, height):
    """A fixed 2x2 cloth patch at z = 1 with one free particle above its center."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cell = 0.1
    builder.add_cloth_grid(
        pos=wp.vec3(-cell, -cell, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        cell_x=cell,
        cell_y=cell,
        mass=0.0,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        edge_ke=1.0e-2,
        particle_radius=0.005,
    )
    builder.add_particle(pos=wp.vec3(0.01, 0.02, 1.0 + height), vel=wp.vec3(0.0), mass=0.1, radius=0.005)
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0
    return model


def test_resting_vertex_carries_its_weight_in_the_multiplier(test, device):
    """Hold a particle on a fixed patch at the cushion height R - m g / K with lambda = m g and no sinking."""
    model = _vertex_over_fixed_cloth(device, height=0.015)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
        particle_enable_self_contact=True,
        particle_self_contact_margin=0.02,
        particle_self_contact_gap=0.01,
        particle_self_contact_method="alm_dat",
    )
    state, output = model.state(), model.state()
    dt = 1.0 / 60.0
    weight = 0.1 * 9.81
    expected_height = 0.02 - weight / 1.0e3
    free = model.particle_count - 1
    heights = []
    for _ in range(240):
        solver.step(state, output, None, None, dt)
        state, output = output, state
        z = float(state.particle_q.numpy()[free, 2]) - 1.0
        test.assertGreaterEqual(z, -1.0e-6)
        heights.append(z)
    test.assertAlmostEqual(heights[-1], expected_height, delta=2.0e-4)
    # no sinking over the last two seconds
    test.assertAlmostEqual(heights[-1], heights[-120], delta=1.0e-4)
    lam = solver.particle_contact_alm_multipliers()
    test.assertAlmostEqual(float(lam.max()), weight, delta=0.05 * weight)


devices = get_test_devices()


class TestParticleContactAlm(unittest.TestCase):
    pass


add_function_test(
    TestParticleContactAlm,
    "test_contact_row_reproduces_the_cushion_law_at_its_fixed_point",
    test_contact_row_reproduces_the_cushion_law_at_its_fixed_point,
    devices=devices,
)
add_function_test(
    TestParticleContactAlm,
    "test_contact_row_switches_to_the_hard_wall_past_the_plane",
    test_contact_row_switches_to_the_hard_wall_past_the_plane,
    devices=devices,
)
add_function_test(
    TestParticleContactAlm,
    "test_contact_row_releases_when_far_from_the_plane",
    test_contact_row_releases_when_far_from_the_plane,
    devices=devices,
)
add_function_test(
    TestParticleContactAlm,
    "test_resting_vertex_carries_its_weight_in_the_multiplier",
    test_resting_vertex_carries_its_weight_in_the_multiplier,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
