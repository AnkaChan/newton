# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the slack-gap ALM self-contact row used by the alm_dat path."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.particle_dat_kernels import (
    alm_contact_row,
    alm_contact_row_smooth,
    alm_friction_row,
    cushion_stiffness,
)
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


@wp.kernel
def _friction_row_kernel(
    lambda_t: wp.array[wp.vec3],
    rel: wp.array[wp.vec3],
    n: wp.vec3,
    rho_t: float,
    radius: float,
    out_force: wp.array[wp.vec3],
    out_jacobian: wp.array[wp.mat33],
):
    i = wp.tid()
    force, jacobian = alm_friction_row(lambda_t[i], rho_t, rel[i], n, radius)
    out_force[i] = force
    out_jacobian[i] = jacobian


def _run_friction_row(device, lambda_t, rel, n=(0.0, 0.0, 1.0), rho_t=100.0, radius=1.0):
    count = len(lambda_t)
    out_force = wp.zeros(count, dtype=wp.vec3, device=device)
    out_jacobian = wp.zeros(count, dtype=wp.mat33, device=device)
    wp.launch(
        _friction_row_kernel,
        dim=count,
        inputs=[
            wp.array(lambda_t, dtype=wp.vec3, device=device),
            wp.array(rel, dtype=wp.vec3, device=device),
            wp.vec3(*n),
            rho_t,
            radius,
        ],
        outputs=[out_force, out_jacobian],
        device=device,
    )
    return out_force.numpy(), out_jacobian.numpy()


def test_friction_row_clamps_to_the_coulomb_disk(test, device):
    """Keep the tangential multiplier inside the Coulomb disk and drop its normal component."""
    lambda_t = [(0.2, 0.0, 0.5), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    rel = [(0.001, 0.0, 0.02), (0.02, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    force, jacobian = _run_friction_row(device, lambda_t, rel, rho_t=100.0, radius=1.0)
    # inside the disk: trial = P lambda_t + rho_t * P rel, normal parts removed
    np.testing.assert_allclose(force[0], [0.3, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(jacobian[0], 100.0 * np.diag([1.0, 1.0, 0.0]), atol=1e-5)
    # on the rim: length equals the radius along the trial direction, jacobian annihilates that direction
    np.testing.assert_allclose(force[1], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(jacobian[1] @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, 0.0], atol=1e-5)
    test.assertGreater(jacobian[1][1, 1], 0.0)
    # no slip and no history: nothing
    np.testing.assert_allclose(force[2], [0.0, 0.0, 0.0], atol=1e-7)
    # oversized history is clamped to the disk even with no slip
    np.testing.assert_allclose(force[3], [1.0, 0.0, 0.0], atol=1e-6)
    # released: zero radius returns no force and no stiffness
    force0, jacobian0 = _run_friction_row(device, [(0.5, 0.0, 0.0)], [(0.1, 0.0, 0.0)], radius=0.0)
    np.testing.assert_allclose(force0[0], [0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(jacobian0[0], np.zeros((3, 3)), atol=1e-7)


def test_friction_row_jacobian_matches_finite_differences(test, device):
    """Match the friction force's derivative with respect to the slip in both regimes, away from the rim."""
    rng = np.random.default_rng(1)
    n = np.array([0.0, 0.6, 0.8])
    h = 1e-4
    for radius, scale in ((1.0, 0.002), (0.05, 0.02)):  # inside the disk, then on the rim
        lambda_t = [tuple(rng.normal(size=3) * 0.01)]
        rel0 = rng.normal(size=3) * scale
        _base, jacobian = _run_friction_row(device, lambda_t, [tuple(rel0)], n=tuple(n), rho_t=100.0, radius=radius)
        numeric = np.zeros((3, 3))
        for k in range(3):
            d = np.zeros(3)
            d[k] = h
            plus, _ = _run_friction_row(device, lambda_t, [tuple(rel0 + d)], n=tuple(n), rho_t=100.0, radius=radius)
            minus, _ = _run_friction_row(device, lambda_t, [tuple(rel0 - d)], n=tuple(n), rho_t=100.0, radius=radius)
            numeric[:, k] = (plus[0] - minus[0]) / (2.0 * h)
        np.testing.assert_allclose(jacobian[0], numeric, atol=2e-2 * 100.0, rtol=2e-2)
        test.assertGreaterEqual(np.linalg.eigvalsh(0.5 * (jacobian[0] + jacobian[0].T)).min(), -1e-4)


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


def test_overloaded_vertex_rests_on_the_hard_wall_above_the_plane(test, device):
    """Hold a vertex heavier than the cushion can carry at the hard gap, with the multiplier equal to its weight."""
    model = _vertex_over_fixed_cloth(device, height=0.015)
    model.particle_mass.assign(np.where(model.particle_mass.numpy() > 0.0, 1.0, 0.0).astype(np.float32))
    model.particle_inv_mass.assign(np.where(model.particle_mass.numpy() > 0.0, 1.0, 0.0).astype(np.float32))
    model.soft_contact_ke = 100.0  # cushion capacity K (R - hard_gap) = 1.5 N, weight 9.81 N
    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
        particle_enable_self_contact=True,
        particle_self_contact_margin=0.02,
        particle_self_contact_gap=0.01,
        particle_self_contact_method="alm_dat",
        particle_contact_alm_hard_gap=0.005,
    )
    state, output = model.state(), model.state()
    free = model.particle_count - 1
    heights = []
    for _ in range(240):
        solver.step(state, output, None, None, 1.0 / 60.0)
        state, output = output, state
        heights.append(float(state.particle_q.numpy()[free, 2]) - 1.0)
        test.assertGreaterEqual(heights[-1], -1.0e-6)
    # rests at the hard gap (within the wall's compliance 1/rho), not on the surface, and does not sink
    test.assertAlmostEqual(heights[-1], 0.005, delta=5.0e-4)
    test.assertAlmostEqual(heights[-1], heights[-120], delta=1.0e-4)
    lam = solver.particle_contact_alm_multipliers()
    test.assertAlmostEqual(float(lam.max()), 9.81, delta=0.5)


def _landing(device, damping, height=0.03, steps=40):
    """Drop a vertex onto the fixed patch; return (max upward speed after landing, lowest height)."""
    model = _vertex_over_fixed_cloth(device, height=height)
    model.soft_contact_kd = damping
    model.soft_contact_mu = 0.5
    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
        particle_enable_self_contact=True,
        particle_self_contact_margin=0.02,
        particle_self_contact_gap=0.01,
        particle_self_contact_method="alm_dat",
    )
    state, output = model.state(), model.state()
    free = model.particle_count - 1
    landed = False
    rebound = 0.0
    lowest = height
    for _ in range(steps):
        solver.step(state, output, None, None, 1.0 / 60.0)
        state, output = output, state
        z = float(state.particle_q.numpy()[free, 2]) - 1.0
        vz = float(state.particle_qd.numpy()[free, 2])
        lowest = min(lowest, z)
        if z < 0.02:
            landed = True
        if landed:
            rebound = max(rebound, vz)
        if z < -1.0e-6:
            raise AssertionError("vertex crossed the fixed patch")
    return rebound, lowest


def test_contact_damping_reduces_rebound(test, device):
    """Land a dropped vertex with far less rebound and shallower cushion penetration when the gap rate is damped."""
    undamped_rebound, undamped_lowest = _landing(device, 0.0)
    damped_rebound, damped_lowest = _landing(device, 30.0)
    test.assertGreater(undamped_rebound, 0.03)
    test.assertLess(damped_rebound, 0.3 * undamped_rebound)
    test.assertGreater(damped_lowest, undamped_lowest)


def _patch_on_incline(device, angle):
    """A free 6x6 cloth on a fixed 12x12 plate, both tilted about the x axis by ``angle``."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cell = 0.05
    for grid, z, mass in ((12, 1.0, 0.0), (6, 1.019, 0.02)):
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
            tri_kd=1.0e-2,
            edge_ke=1.0e-3,
            particle_radius=0.005,
        )
    c, s = np.cos(angle), np.sin(angle)
    for i in range(builder.particle_count):
        x, y, z = builder.particle_q[i]
        y, z = y * c - (z - 1.0) * s, (z - 1.0) * c + y * s + 1.0
        builder.particle_q[i] = (x, y, z)
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 0.0
    return model, 13 * 13


def _incline_drift(device, mu, steps=120):
    angle = np.radians(10.0)
    model, fixed_count = _patch_on_incline(device, angle)
    model.soft_contact_mu = mu
    solver = newton.solvers.SolverVBD(
        model,
        iterations=6,
        particle_enable_self_contact=True,
        particle_self_contact_margin=0.02,
        particle_self_contact_gap=0.01,
        particle_self_contact_method="alm_dat",
        particle_vertex_contact_buffer_size=16,
        particle_edge_contact_buffer_size=64,
    )
    state, output = model.state(), model.state()
    start = model.particle_q.numpy()[fixed_count:].mean(axis=0)
    plate_normal = np.array([0.0, -np.sin(angle), np.cos(angle)])
    for _ in range(steps):
        solver.step(state, output, None, None, 1.0 / 60.0)
        state, output = output, state
    q = state.particle_q.numpy()[fixed_count:]
    heights = (q - np.array([0.0, 0.0, 1.0])) @ plate_normal
    return float(np.linalg.norm(q.mean(axis=0) - start)), float(heights.min())


def test_patch_sticks_on_an_incline_with_friction(test, device):
    """Hold a patch on a 10 degree incline with mu = 0.5 and let it slide away with mu = 0."""
    drift_friction, height_friction = _incline_drift(device, 0.5)
    drift_free, _height_free = _incline_drift(device, 0.0)
    test.assertLess(drift_friction, 5.0e-3)
    test.assertGreater(drift_free, 5.0e-2)
    test.assertGreaterEqual(height_friction, -1.0e-6)


def _patch_on_fixed_plate(device):
    """A free 6x6 cloth resting inside the margin above a fixed, level 12x12 cloth."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cell = 0.05
    for grid, z, mass in ((12, 1.0, 0.0), (6, 1.019, 0.02)):
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
            tri_kd=1.0e-2,
            edge_ke=1.0e-3,
            particle_radius=0.005,
        )
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0
    return model, 13 * 13


def test_resting_patch_does_not_drift_sideways(test, device):
    """Keep a frictionless patch resting on a level plate in place: contact forces must act along the contact direction."""
    model, fixed_count = _patch_on_fixed_plate(device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=6,
        particle_enable_self_contact=True,
        particle_self_contact_margin=0.02,
        particle_self_contact_gap=0.01,
        particle_self_contact_method="alm_dat",
        particle_vertex_contact_buffer_size=16,
        particle_edge_contact_buffer_size=64,
    )
    state, output = model.state(), model.state()
    start = model.particle_q.numpy()[fixed_count:, :2].mean(axis=0)
    centers = []
    for _ in range(240):
        solver.step(state, output, None, None, 1.0 / 60.0)
        state, output = output, state
        centers.append(state.particle_q.numpy()[fixed_count:, :2].mean(axis=0))
    # the triangulated plate biases the first settle by a few millimetres on every path;
    # what must not happen is a sustained lateral push, so the drift has to stop growing
    total = float(np.linalg.norm(centers[-1] - start))
    late = float(np.linalg.norm(centers[-1] - centers[119]))
    test.assertLess(total, 1.0e-2)
    test.assertLess(late, 1.0e-3)
    test.assertGreaterEqual(float(state.particle_q.numpy()[fixed_count:, 2].min()), 1.0 - 1.0e-6)


devices = get_test_devices()


@wp.kernel
def _smooth_row_kernel(
    gap: wp.array[float],
    lam: wp.array[float],
    stiffness: float,
    radius: float,
    rho: float,
    band: float,
    wall_ratio: float,
    force: wp.array[float],
    hessian: wp.array[float],
    k_cushion: wp.array[float],
):
    i = wp.tid()
    k_c = cushion_stiffness(gap[i], stiffness, rho, band, wall_ratio)
    t, k = alm_contact_row_smooth(gap[i], lam[i], stiffness, radius, rho, band, wall_ratio)
    force[i] = t
    hessian[i] = k
    k_cushion[i] = k_c


def _run_smooth_row(device, gaps, lams, stiffness, radius, rho, band, wall_ratio):
    n = len(gaps)
    force = wp.zeros(n, dtype=float, device=device)
    hessian = wp.zeros(n, dtype=float, device=device)
    k_cushion = wp.zeros(n, dtype=float, device=device)
    wp.launch(
        _smooth_row_kernel,
        dim=n,
        inputs=[
            wp.array(gaps, dtype=float, device=device),
            wp.array(lams, dtype=float, device=device),
            stiffness,
            radius,
            rho,
            band,
            wall_ratio,
        ],
        outputs=[force, hessian, k_cushion],
        device=device,
    )
    return force.numpy(), hessian.numpy(), k_cushion.numpy()


def test_smooth_transition_raises_the_stiffness_toward_the_wall_without_a_jump(test, device):
    """With a transition band the cushion stiffness rises from K to the wall stiffness along a smoothstep,
    the in-sweep stiffness stays below rho, and the force is continuous across the band and the wall."""
    stiffness, radius, rho, band, wall_ratio = 10.0, 0.5, 30.0, 0.2, 9.0
    gaps = np.linspace(-0.05, 0.45, 101)
    lam = stiffness * radius  # the cushion's full load, so the slack pins at the wall for gap <= 0
    force, hessian, k_c = _run_smooth_row(device, gaps, [lam] * len(gaps), stiffness, radius, rho, band, wall_ratio)
    # outside the band the authored stiffness; at the wall the wall stiffness
    np.testing.assert_allclose(k_c[gaps >= band], stiffness, atol=1.0e-6)
    np.testing.assert_allclose(k_c[gaps <= 0.0], wall_ratio * rho, atol=1.0e-4)
    # monotone in the band and never decreasing toward the wall
    inside = (gaps > 0.0) & (gaps < band)
    test.assertTrue(np.all(np.diff(k_c[inside]) <= 1.0e-6))
    # in-sweep stiffness bounded by the metric, and above the authored cushion value once the
    # slack sits inside the band (the stiffness is evaluated at the slack, not at the gap)
    k_eff_authored = stiffness * rho / (stiffness + rho)
    test.assertTrue(np.all(hessian <= rho + 1.0e-6))
    test.assertGreater(float(hessian[gaps <= 0.0].min()), k_eff_authored)
    # a multiplier above the full band force pins the slack at the wall: hard regime, stiffness rho
    force_hard, hessian_hard, _ = _run_smooth_row(
        device,
        [-0.01],
        [stiffness * radius + (wall_ratio * rho - stiffness) * band],
        stiffness,
        radius,
        rho,
        band,
        wall_ratio,
    )
    np.testing.assert_allclose(hessian_hard, [rho], atol=1.0e-5)
    np.testing.assert_allclose(
        force_hard, [stiffness * radius + (wall_ratio * rho - stiffness) * band + rho * 0.01], rtol=1.0e-5
    )
    # force continuous: the largest jump between neighbouring samples is small compared with the force scale
    jumps = np.abs(np.diff(force))
    test.assertLess(float(jumps.max()), 0.2 * stiffness * radius)
    # the smooth law never pushes less than the piecewise law at the same gap and multiplier
    force_piecewise, _, _ = _run_smooth_row(device, gaps, [lam] * len(gaps), stiffness, radius, rho, 0.0, wall_ratio)
    test.assertTrue(np.all(force + 1.0e-6 >= force_piecewise))
    # band off reproduces the piecewise cushion stiffness everywhere
    _, _, k_off = _run_smooth_row(device, gaps, [lam] * len(gaps), stiffness, radius, rho, 0.0, wall_ratio)
    np.testing.assert_allclose(k_off, stiffness, atol=1.0e-6)
    # at the fixed point (multiplier equal to the cushion force) the transmitted force is the integral law:
    # K (R - C) outside the band, and K R + (K_wall - K) band / 2 at the wall
    wall = wall_ratio * rho
    fixed_gaps = np.array([0.4, 0.3, band, 0.0])
    expected = np.array(
        [
            stiffness * (radius - 0.4),
            stiffness * (radius - 0.3),
            stiffness * (radius - band),
            stiffness * radius + (wall - stiffness) * band / 2,
        ]
    )
    force_fp, _, _ = _run_smooth_row(device, fixed_gaps, expected, stiffness, radius, rho, band, wall_ratio)
    np.testing.assert_allclose(force_fp, expected, rtol=1.0e-4, atol=1.0e-5)


def test_overloaded_vertex_rests_inside_the_transition_band(test, device):
    """With the smooth law a vertex heavier than the cushion rests above the wall, inside the band,
    with the multiplier equal to its weight and without sinking."""
    model = _vertex_over_fixed_cloth(device, height=0.015)
    model.particle_mass.assign(np.where(model.particle_mass.numpy() > 0.0, 1.0, 0.0).astype(np.float32))
    model.particle_inv_mass.assign(np.where(model.particle_mass.numpy() > 0.0, 1.0, 0.0).astype(np.float32))
    model.soft_contact_ke = 100.0  # cushion capacity K (R - hard_gap) = 1.5 N, weight 9.81 N
    hard_gap, band = 0.005, 0.004
    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
        particle_enable_self_contact=True,
        particle_self_contact_margin=0.02,
        particle_self_contact_gap=0.01,
        particle_self_contact_method="alm_dat",
        particle_contact_alm_hard_gap=hard_gap,
        particle_contact_alm_transition_band=band,
    )
    state, output = model.state(), model.state()
    dt = 1.0 / 60.0
    weight = 9.81
    free = model.particle_count - 1
    heights = []
    for _ in range(240):
        solver.step(state, output, None, None, dt)
        state, output = output, state
        z = float(state.particle_q.numpy()[free, 2]) - 1.0
        test.assertGreaterEqual(z, -1.0e-6)
        heights.append(z)
    # rests above the wall and inside the band
    test.assertGreater(heights[-1], hard_gap - 1.0e-4)
    test.assertLess(heights[-1], hard_gap + band + 1.0e-4)
    # no sinking over the last two seconds
    test.assertAlmostEqual(heights[-1], heights[-120], delta=1.0e-4)
    lam = solver.particle_contact_alm_multipliers()
    test.assertAlmostEqual(float(lam.max()), weight, delta=0.05 * weight)


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
add_function_test(
    TestParticleContactAlm,
    "test_overloaded_vertex_rests_on_the_hard_wall_above_the_plane",
    test_overloaded_vertex_rests_on_the_hard_wall_above_the_plane,
    devices=devices,
)
add_function_test(
    TestParticleContactAlm,
    "test_resting_patch_does_not_drift_sideways",
    test_resting_patch_does_not_drift_sideways,
    devices=devices,
)
add_function_test(
    TestParticleContactAlm,
    "test_friction_row_clamps_to_the_coulomb_disk",
    test_friction_row_clamps_to_the_coulomb_disk,
    devices=devices,
)
add_function_test(
    TestParticleContactAlm,
    "test_friction_row_jacobian_matches_finite_differences",
    test_friction_row_jacobian_matches_finite_differences,
    devices=devices,
)
add_function_test(
    TestParticleContactAlm,
    "test_contact_damping_reduces_rebound",
    test_contact_damping_reduces_rebound,
    devices=devices,
)
add_function_test(
    TestParticleContactAlm,
    "test_patch_sticks_on_an_incline_with_friction",
    test_patch_sticks_on_an_incline_with_friction,
    devices=devices,
)


add_function_test(
    TestParticleContactAlm,
    "test_smooth_transition_raises_the_stiffness_toward_the_wall_without_a_jump",
    test_smooth_transition_raises_the_stiffness_toward_the_wall_without_a_jump,
    devices=devices,
)
add_function_test(
    TestParticleContactAlm,
    "test_overloaded_vertex_rests_inside_the_transition_band",
    test_overloaded_vertex_rests_inside_the_transition_band,
    devices=devices,
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
