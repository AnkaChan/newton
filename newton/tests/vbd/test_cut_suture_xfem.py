# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The suture on tissue cut with the XFEM-type cut: entry at a wound face, tet walk over fragment rows, exit at a wound face."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices
from newton.tests.vbd.test_cut import (
    CELL,
    SUBSTEPS,
    THROUGH_FRAMES,
    TOP,
    XFEM_TETS,
    _xfem_scene,
)

TIP_RADIUS = 4.0e-4
TIP_HALF_HEIGHT = 1.0e-3
NEEDLE_LENGTH = 0.03
SPACING = 1.0e-3
# Each fixed end moves this far from the wound, so the two level-zero surfaces of the cut are 4 mm apart
# and the wound faces 5 mm: more than the gate radius of the needle (2.9 mm), so the gate of the first
# tunnel does not reach the other wound face.
OPEN_SHIFT = 2.0e-3
OPEN_FRAMES = 20
SETTLE_FRAMES = 10
# The needle is straight. Its tip goes down at 25 degrees from the horizontal in the plane y = 2.2 mm: in
# through the top face of side A at x = 4 mm, out through the wound face of side A, across the open wound,
# in through the wound face of side B and out through the bottom face of side B.
PASS_ANGLE = math.radians(25.0)
PASS_Y = 2.2e-3
PASS_ENTRY_X = 4.0e-3
PASS_LEAD = 3.0e-3
PASS_STEP = 1.0e-4
PASS_LENGTH = PASS_LEAD + TOP / math.sin(PASS_ANGLE) + 2.0e-3
PUNCTURE_FORCE = 1.0e-3
PUNCTURE_COS = 0.3
NEEDLE_PARK = (0.0, 0.0, 0.05)


def _add_needle(builder):
    """A straight kinematic needle; the body origin is the tip and the needle extends along local +z."""
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    needle = builder.add_body(xform=wp.transform(wp.vec3(*NEEDLE_PARK), wp.quat_identity()))
    tip = builder.add_shape_capsule(
        needle,
        xform=wp.transform(wp.vec3(0.0, 0.0, TIP_HALF_HEIGHT), wp.quat_identity()),
        radius=TIP_RADIUS,
        half_height=TIP_HALF_HEIGHT,
        cfg=cfg,
    )
    shaft_half = 0.5 * (NEEDLE_LENGTH - 2.0 * TIP_HALF_HEIGHT)
    builder.add_shape_capsule(
        needle,
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * TIP_HALF_HEIGHT + shaft_half), wp.quat_identity()),
        radius=TIP_RADIUS,
        half_height=shaft_half,
        cfg=cfg,
    )
    builder.add_suture_needle(
        needle, tip, centerline=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, NEEDLE_LENGTH)], max_tunnels=4
    )
    return needle


def _suture_scene(device, **kwargs):
    """The block of test_cut.py with the blade plane in the middle of a cell column, and a parked needle."""
    scene = _xfem_scene(device, before_color=_add_needle, suture_point_spacing=SPACING, **kwargs)
    scene.needle = scene.extra
    # The puncture attributes of the tissue only: the spare particles keep the registered defaults, so a
    # wound particle has the values of the tissue only when the cut copies them.
    for attribute, value in (
        (scene.model.vbd.suture_puncture_force, PUNCTURE_FORCE),
        (scene.model.vbd.suture_puncture_cos, PUNCTURE_COS),
    ):
        values = attribute.numpy()
        values[: scene.tissue_count] = value
        attribute.assign(values)
    return scene


def _pass_direction():
    return np.array([math.cos(PASS_ANGLE), 0.0, -math.sin(PASS_ANGLE)])


def _needle_pose(travel):
    """Pose of the needle with its tip ``travel`` along the pass, which starts PASS_LEAD before the top face."""
    direction = _pass_direction()
    entry = np.array([PASS_ENTRY_X, PASS_Y, TOP])
    tip = entry + (travel - PASS_LEAD) * direction
    # Local +z points from the tip back along the needle: a turn about +y by the pass angle minus 90 degrees.
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), PASS_ANGLE - 0.5 * math.pi)
    return wp.transform(wp.vec3(*tip), rotation)


def _run_pass(scene, length=PASS_LENGTH):
    """Move the needle tip along the pass, one PASS_STEP per substep."""
    steps = int(length / PASS_STEP)
    for step in range(steps):
        if step % SUBSTEPS == 0:
            scene.solver.rebuild_bvh(scene.state_0)
        body_q = scene.state_0.body_q.numpy()
        body_q[scene.needle] = _needle_pose((step + 1) * PASS_STEP)
        scene.state_0.body_q.assign(body_q)
        scene.substep()


def _tunnels(scene):
    """State, tet rows, weights and world places of the tunnel points of every tunnel of needle 0."""
    solver = scene.solver
    q = scene.state_0.particle_q.numpy().astype(np.float64)
    tets = scene.model.tet_indices.numpy().reshape(-1, 4)
    states = solver._suture_tunnel_state.numpy()
    tunnels = []
    for tunnel in range(int(solver._suture_tunnel_count.numpy()[0])):
        rows, weights = solver.suture_tunnel_points(0, tunnel)
        places = np.einsum("pk,pkc->pc", weights.astype(np.float64), q[tets[rows]])
        tunnels.append({"state": int(states[tunnel]), "rows": rows, "weights": weights, "places": places})
    return tunnels


def _signed_levels(scene, rows, weights):
    """Level value at each tunnel point, with the sign turned so that the side of its fragment row is positive [m].

    The value is ``inf`` for a point in a tet that is not cut.
    """
    x = scene.solver._xfem
    side = x["tet_side"].numpy()[rows]
    level = x["particle_level"].numpy()[x["tet_vertex"].numpy()[rows]].astype(np.float64)
    value = (weights.astype(np.float64) * level).sum(axis=1)
    return np.where(side == 1, -value, np.where(side == 2, value, np.inf))


def test_xfem_suture_pass_across_the_wound(test, device):
    """Verify one needle pass across a wound of the XFEM-type cut: two through tunnels in the fragments of their side."""
    scene = _suture_scene(device)
    solver = scene.solver
    x = solver._xfem
    scene.run(THROUGH_FRAMES)
    test.assertEqual(solver.cut_report()["cut_tets"], XFEM_TETS)
    test.assertEqual(sum(solver.cut_report()["overflow"]), 0)
    for i in range(OPEN_FRAMES):
        scene.set_holds(OPEN_SHIFT * (i + 1) / OPEN_FRAMES)
        scene.step_frame(scene.travel)
    for _ in range(SETTLE_FRAMES):
        scene.step_frame(scene.travel)
    # The cut copies the puncture attributes of the tissue to the wound particles.
    wound = x["wound_edge"].numpy() >= 0
    wound_side = x["wound_side"].numpy()
    test.assertTrue((scene.model.vbd.suture_puncture_force.numpy()[wound] == np.float32(PUNCTURE_FORCE)).all())
    test.assertTrue((scene.model.vbd.suture_puncture_cos.numpy()[wound] == np.float32(PUNCTURE_COS)).all())
    test.assertEqual(int(solver._suture_tunnel_count.numpy()[0]), 0)

    _run_pass(scene)

    q = scene.state_0.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertFalse(solver.suture_point_overflow())
    tunnels = _tunnels(scene)
    test.assertEqual(len(tunnels), 2)
    tet_side = x["tet_side"].numpy()
    tet_first = x["tet_first"].numpy()
    first, second = tunnels
    for tunnel in tunnels:
        test.assertEqual(tunnel["state"], 2)  # through
        test.assertGreaterEqual(len(tunnel["rows"]), 3)
        # Every tunnel point is in the material of its row: inside the tet shape and on the row's side.
        test.assertGreaterEqual(float(tunnel["weights"].min()), -1.0e-6)
        np.testing.assert_allclose(tunnel["weights"].sum(axis=1), 1.0, atol=1.0e-5)
        test.assertGreaterEqual(float(_signed_levels(scene, tunnel["rows"], tunnel["weights"]).min()), -1.0e-6)
        test.assertTrue((tet_side[tunnel["rows"]] >= 0).all())  # never a spare tet that is not in use
    # The first tunnel is in side A and leaves through the wound face: its last point is in a fragment row
    # of side A, on the level-zero surface of that row.
    test.assertTrue((tet_side[first["rows"]] <= 1).all())
    test.assertEqual(int(tet_side[first["rows"][-1]]), 1)
    test.assertLess(abs(float(_signed_levels(scene, first["rows"][-1:], first["weights"][-1:])[0])), 1.0e-5)
    # The second tunnel enters at the wound face of side B: its first point is in a fragment row of side B,
    # which is a spare tet, and all its points are in side B.
    test.assertEqual(int(tet_side[second["rows"][0]]), 2)
    test.assertGreaterEqual(int(second["rows"][0]), scene.tissue_tets)
    test.assertEqual(int(tet_side[tet_first[second["rows"][0]]]), 1)
    test.assertTrue(np.isin(tet_side[second["rows"]], (0, 2)).all())
    # The places of the points: side A is left of the open wound, side B right of it, and every point is
    # on the straight needle. A point in the fragment row of the wrong side would be 4 mm off.
    middle = scene.start[0]
    test.assertLess(float(first["places"][:, 0].max()), middle - OPEN_SHIFT + 1.0e-4)
    test.assertGreater(float(second["places"][:, 0].min()), middle + OPEN_SHIFT - 1.0e-4)
    direction = _pass_direction()
    tip = np.array(wp.transform_get_translation(wp.transform(*scene.state_0.body_q.numpy()[scene.needle])))
    for tunnel in tunnels:
        offset = tunnel["places"] - tip
        offset -= np.outer(offset @ direction, direction)
        test.assertLess(float(np.linalg.norm(offset, axis=1).max()), 5.0e-4)
    # The contact gate: wound particles of both wound faces near the tunnels carry the needle bit, next to
    # the blade bit that the cut gave them.
    gate = scene.model.particle_contact_gate.numpy()
    needle_bit = int(scene.model.suture_needle_group.numpy()[0])
    for side in (0, 1):
        test.assertGreater(int(((gate[wound & (wound_side == side)] & needle_bit) != 0).sum()), 0)
    test.assertEqual(solver.cut_report()["blocked_by_tunnel"], 0)


def test_xfem_no_cut_through_a_tunnel(test, device):
    """Verify the guard of a cut after a suture: a tet that holds a tunnel point is not cut."""
    scene = _suture_scene(device)
    solver = scene.solver
    x = solver._xfem
    # A tunnel point in a tet of the top cell layer that the blade plane crosses, with a row too soft to
    # move the tissue toward the parked needle.
    rest = scene.model.particle_q.numpy()
    tets = scene.model.tet_indices.numpy().reshape(-1, 4)[: scene.tissue_tets]
    centers = rest[tets].mean(axis=1)
    side_b = rest[tets][:, :, 0] >= scene.start[0]
    crossed = side_b.any(axis=1) & ~side_b.all(axis=1)
    top_layer = centers[:, 2] > TOP - CELL
    held = int(np.flatnonzero(crossed & top_layer)[0])
    solver.suture_set_points(0, 0, [held], [[0.25, 0.25, 0.25, 0.25]], state=scene.state_0, stiffness=1.0e-6)
    scene.run(THROUGH_FRAMES)
    report = solver.cut_report()
    tet_side = x["tet_side"].numpy()
    test.assertEqual(int(tet_side[held]), 0)
    test.assertGreater(report["blocked_by_tunnel"], 0)
    test.assertGreater(report["cut_tets"], 0)
    test.assertLess(report["cut_tets"], XFEM_TETS)
    test.assertEqual(sum(report["overflow"]), 0)
    test.assertTrue(np.isfinite(scene.state_0.particle_q.numpy()).all())
    # The tunnel point is where it was put.
    rows, weights = solver.suture_tunnel_points(0, 0)
    test.assertEqual(rows.tolist(), [held])
    np.testing.assert_allclose(weights, [[0.25, 0.25, 0.25, 0.25]])
    # A reset undoes the cut and clears every tunnel.
    scene.solver.reset(scene.state_0)
    test.assertEqual(int(solver._suture_tunnel_count.numpy()[0]), 0)
    test.assertTrue((solver._suture_point_tet.numpy() == -1).all())
    test.assertEqual(solver.cut_report()["cut_tets"], 0)
    test.assertEqual(solver.cut_report()["blocked_by_tunnel"], 0)


devices = get_test_devices()
cuda_devices = [device for device in devices if wp.get_device(device).is_cuda]


class TestCutSutureXfem(NewtonTestCase):
    pass


# The tests run the whole cut first; they run on CUDA only, as the long tests of test_cut.py do.
for fn in (test_xfem_suture_pass_across_the_wound, test_xfem_no_cut_through_a_tunnel):
    add_function_test(TestCutSutureXfem, fn.__name__, fn, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
