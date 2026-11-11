# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cutting in SolverVBD: builder capacity, solver rules, the cut along tet faces, the XFEM-type cut, capture and reset."""

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.cut_xfem_kernels import XFEM_EDGE_FRONT, XFEM_ROW_WOUND, tet_side_a_share
from newton.exceptions import NewtonWarning
from newton.solvers import SolverVBD
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices

CELL = 3.0e-3
DIMS = (6, 2, 2)
TOP = DIMS[2] * CELL
BLADE_RADIUS = 4.5e-4
FPS = 60
SUBSTEPS = 10
DT = 1.0 / FPS / SUBSTEPS
# Edges shorter than the square root of this value act as points in edge-edge self contact.
EDGE_PARALLEL_EPSILON = 1.0e-7


def _builder_with_block(*, hold=True):
    """A 18 x 6 x 6 mm tissue block; with ``hold`` the two end layers along x are fixed."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 1.0e-3
    SolverVBD.register_custom_attributes(builder)
    E, nu = 1.0e5, 0.45
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=DIMS[0],
        dim_y=DIMS[1],
        dim_z=DIMS[2],
        cell_x=CELL,
        cell_y=CELL,
        cell_z=CELL,
        density=1000.0,
        k_mu=E / (2.0 * (1.0 + nu)),
        k_lambda=E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        k_damp=0.0,
        particle_radius=5.0e-4,
    )
    if hold:
        length = DIMS[0] * CELL
        for i in range(builder.particle_count):
            x = builder.particle_q[i][0]
            if x <= 1.0e-9 or x >= length - 1.0e-9:
                builder.particle_mass[i] = 0.0
                builder.particle_flags[i] = int(builder.particle_flags[i]) & ~int(newton.ParticleFlags.ACTIVE)
    return builder


def _add_blade(builder, yaw_deg=0.0, offset=0.0):
    """A capsule along y above the middle grid plane (moved by ``offset`` along x), on a kinematic holder with a D6 drive."""
    plane_x = (DIMS[0] // 2) * CELL + offset
    start_z = TOP + BLADE_RADIUS + 1.0e-3
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(yaw_deg))
    xf = wp.transform(wp.vec3(plane_x, 0.5 * DIMS[1] * CELL, start_z), rotation)
    holder = builder.add_body(xform=xf)
    blade = builder.add_body(xform=xf)
    cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3)
    z_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
    edge = builder.add_shape_capsule(
        blade,
        xform=wp.transform(wp.vec3(0.0), z_to_y),
        radius=BLADE_RADIUS,
        half_height=0.5 * DIMS[1] * CELL + 2.0e-3,
        cfg=cfg,
    )
    # The body of the blade: a thin plate without contact. Its mass keeps the body inertia well conditioned.
    plate = newton.ModelBuilder.ShapeConfig(density=7800.0, has_shape_collision=False, has_particle_collision=False)
    builder.add_shape_box(
        blade,
        xform=wp.transform(wp.vec3(0.0, 0.0, 3.0e-3), wp.quat_identity()),
        hx=1.5e-4,
        hy=0.5 * DIMS[1] * CELL + 2.0e-3,
        hz=3.0e-3,
        cfg=plate,
    )
    linear = [
        newton.ModelBuilder.JointDofConfig(axis=axis, target_ke=1.0e4, target_kd=1.0e2)
        for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
    ]
    angular = [
        newton.ModelBuilder.JointDofConfig(axis=axis, target_ke=10.0, target_kd=0.1)
        for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
    ]
    builder.add_joint_d6(holder, blade, linear_axes=linear, angular_axes=angular)
    builder.add_cut_blade(blade, edge, cut_direction=(0.0, 0.0, -1.0))
    return holder, blade, edge, np.array([plane_x, 0.5 * DIMS[1] * CELL, start_z]), rotation


def _add_capacity(builder, **kwargs):
    options = {
        "max_vertices": 8,
        "max_faces": 16,
        "park_position": (0.0, 0.05, -0.03),
        "park_spacing": 3.0e-3,
        "particle_radius": 5.0e-4,
    }
    options.update(kwargs)
    builder.add_cut_capacity(**options)


def _solver(model, **kwargs):
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=1.0e-3, enable_rigid_soft_full_surface_contact=True)
    options = {
        "iterations": 10,
        "rigid_soft_enable_dat": True,
        "particle_enable_self_contact": True,
        "particle_self_contact_margin": 1.0e-3,
        "particle_self_contact_gap": 1.0e-3,
        "particle_topological_contact_filter_threshold": 1,
        "particle_edge_parallel_epsilon": EDGE_PARALLEL_EPSILON,
        "collision_pipeline": pipeline,
    }
    options.update(kwargs)
    return SolverVBD(model, **options)


class _Scene:
    """The block, a blade and the cut capacity, finalized on ``device``."""

    def __init__(
        self, device, *, cut_force=0.5, yaw_deg=0.0, offset=0.0, capacity=None, capture=False, **solver_kwargs
    ):
        builder = _builder_with_block()
        self.tissue_count = builder.particle_count
        self.tissue_tets = builder.tet_count
        self.yaw_deg = yaw_deg
        # The two fixed end layers; set_holds() moves them.
        q = np.array([[v[0], v[1], v[2]] for v in builder.particle_q], dtype=np.float64)
        self.hold_a = np.flatnonzero(q[:, 0] <= 1.0e-9)
        self.hold_b = np.flatnonzero(q[:, 0] >= DIMS[0] * CELL - 1.0e-9)
        self.hold_rest_a = q[self.hold_a].copy()
        self.hold_rest_b = q[self.hold_b].copy()
        self.holder, self.blade, self.edge, self.start, self.rotation = _add_blade(builder, yaw_deg, offset)
        builder.color()
        _add_capacity(builder, **(capacity or {}))
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = 5.0e2
        self.model.soft_contact_kd = 5.0
        self.model.soft_contact_mu = 0.3
        self.model.shape_material_ke.fill_(5.0e2)
        self.model.shape_material_kd.fill_(5.0)
        force = self.model.vbd.cut_force.numpy()
        force[: self.tissue_count] = cut_force
        self.model.vbd.cut_force.assign(force)
        self.solver = _solver(self.model, **solver_kwargs)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.frame = 0
        self.graph = None
        if capture:
            # One frame captured in a CUDA graph; every later frame replays it.
            with wp.ScopedCapture(device=device) as cap:
                self.frame_body()
            self.graph = cap.graph
            self.frame = 1

    def set_holder(self, travel):
        body_q = self.state_0.body_q.numpy()
        position = self.start - np.array([0.0, 0.0, travel])
        body_q[self.holder] = wp.transform(wp.vec3(*position), self.rotation)
        self.state_0.body_q.assign(body_q)

    def set_holds(self, shift):
        """Move the fixed end layers: side A by ``-shift`` along x and side B by ``+shift`` (a negative value presses the halves together)."""
        q = self.state_0.particle_q.numpy()
        q[self.hold_a] = self.hold_rest_a + np.array([-shift, 0.0, 0.0])
        q[self.hold_b] = self.hold_rest_b + np.array([shift, 0.0, 0.0])
        self.state_0.particle_q.assign(q)

    def frame_body(self):
        """One frame: rebuild the self-contact BVH, then the substeps."""
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(SUBSTEPS):
            self.substep()

    def corner_positions(self):
        """Particle positions read through the tet corners, so that the indices of copies do not matter."""
        tets = self.model.tet_indices.numpy().reshape(-1, 4)[: self.tissue_tets]
        return self.state_0.particle_q.numpy()[tets]

    def lip_gaps(self):
        """Distance along +x from each original of the cut plane to each of its copies."""
        rest = self.model.particle_q.numpy()
        q = self.state_0.particle_q.numpy()
        tets = self.model.tet_indices.numpy().reshape(-1, 4)[: self.tissue_tets]
        copies = np.unique(tets[tets >= self.tissue_count])
        originals = np.arange(self.tissue_count)
        gaps = []
        for copy in copies:
            original = originals[int(np.argmin(np.linalg.norm(rest[originals] - rest[copy], axis=1)))]
            gaps.append(float(q[copy, 0] - q[original, 0]))
        return np.array(gaps)

    def substep(self):
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, None, DT)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step_frame(self, travel):
        """One frame with the holder ``travel`` below its start."""
        self.travel = travel
        self.set_holder(travel)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.frame_body()
        self.frame += 1

    def run(self, frames, *, speed=4.0e-3, max_travel=None):
        """Lower the holder at ``speed`` for ``frames`` frames."""
        for _ in range(frames):
            travel = speed * self.frame / FPS
            if max_travel is not None:
                travel = min(travel, max_travel)
            self.step_frame(travel)

    def parts(self):
        """Tet counts of the connected parts of the tissue."""
        tets = self.model.tet_indices.numpy().reshape(-1, 4)[: self.tissue_tets]
        parent = np.arange(self.model.particle_count)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for tet in tets:
            root = find(int(tet[0]))
            for v in tet[1:]:
                parent[find(int(v))] = root
        counts = {}
        for tet in tets:
            root = find(int(tet[0]))
            counts[root] = counts.get(root, 0) + 1
        return sorted(counts.values())


THROUGH_TRAVEL = TOP + 2.0 * BLADE_RADIUS + 3.0e-3
THROUGH_FRAMES = int(THROUGH_TRAVEL / 4.0e-3 * FPS) + 10
PLANE_FACES = 2 * DIMS[1] * DIMS[2]
PLANE_VERTICES = (DIMS[1] + 1) * (DIMS[2] + 1)


def _check_consistent_topology(test, scene):
    """Every tet corner has exactly one tet record, on the particle it names; every wound row names live particles."""
    model = scene.model
    solver = scene.solver
    tets = model.tet_indices.numpy().reshape(-1, 4)
    pairs = solver.particle_adjacency.v_adj_tets.numpy().reshape(-1, 2)
    offsets = solver.particle_adjacency.v_adj_tets_offsets.numpy()
    idle = solver._cut_idle_tet
    seen = np.zeros((model.tet_count, 4), dtype=np.int32)
    for particle in range(model.particle_count):
        for tet, corner in pairs[offsets[particle] // 2 : offsets[particle + 1] // 2]:
            if tet == idle:
                continue
            test.assertEqual(int(tets[tet, corner]), particle)
            seen[tet, corner] += 1
    test.assertTrue((seen[:idle] == 1).all())
    flags = model.particle_flags.numpy()
    mass = model.particle_mass.numpy()
    used = np.unique(tets[: scene.tissue_tets])
    spare_used = used[used >= scene.tissue_count]
    test.assertTrue(((flags[spare_used] & int(newton.ParticleFlags.ACTIVE)) != 0).all())
    test.assertTrue((mass[spare_used] > 0.0).all())
    capacity = model.cut_capacity
    tri = model.tri_indices.numpy().reshape(-1, 3)
    taken = int(solver._cut_tri_taken.numpy()[0])
    start = capacity["tri_spare_start"]
    wound = tri[start : start + taken]
    test.assertTrue(np.isin(wound, used).all())
    unused = tri[start + taken :]
    test.assertTrue((unused[:, 0] == unused[:, 1]).all() and (unused[:, 1] == unused[:, 2]).all())
    # Every open tet face (skin or cut) of the tissue has exactly one triangle row with its particles, and
    # every edge row lies on an open face.
    neighbors = solver._tet_neighbors.numpy()[: scene.tissue_tets]
    faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    open_faces = {}
    open_edges = set()
    for tet, face in zip(*np.nonzero(neighbors < 0), strict=True):
        ids = tuple(sorted(int(tets[tet, c]) for c in faces[face]))
        open_faces[ids] = open_faces.get(ids, 0) + 1
        open_edges.update({(ids[0], ids[1]), (ids[0], ids[2]), (ids[1], ids[2])})
    rows = {}
    for row in tri:
        if row[0] != row[1]:
            ids = tuple(sorted(int(i) for i in row))
            rows[ids] = rows.get(ids, 0) + 1
    test.assertEqual(rows, open_faces)
    edges = model.edge_indices.numpy().reshape(-1, 4)
    for edge in edges:
        if edge[2] != edge[3]:
            test.assertIn((min(int(edge[2]), int(edge[3])), max(int(edge[2]), int(edge[3]))), open_edges)


def test_cut_capacity_layout(test, device):
    """Verify the particles, rows, colors and guards that add_cut_capacity() makes."""
    builder = _builder_with_block()
    tissue_count = builder.particle_count
    tet_count = builder.tet_count
    tri_count = len(builder.tri_indices)
    edge_count = len(builder.edge_indices)
    with test.assertRaises(RuntimeError):
        _add_capacity(builder)  # no coloring yet
    builder.color()
    colors = len(builder.particle_color_groups)
    _add_capacity(builder, max_vertices=8, max_faces=16)
    test.assertEqual(len(builder.particle_color_groups), 2 * colors)
    test.assertEqual(builder.particle_count, tissue_count + colors * 8 + 32 + 48 + 4)
    test.assertEqual(builder.tet_count, tet_count + 1)
    test.assertEqual(len(builder.tri_indices), tri_count + 32)
    test.assertEqual(len(builder.edge_indices), edge_count + 48)
    for color in range(colors):
        test.assertEqual(len(builder.particle_color_groups[colors + color]), 8 if color < colors - 1 else 8 + 84)
    grouped = np.concatenate(builder.particle_color_groups)
    test.assertEqual(len(np.unique(grouped)), builder.particle_count)
    test.assertEqual(len(grouped), builder.particle_count)
    with test.assertRaises(RuntimeError):
        _add_capacity(builder)
    with test.assertRaises(RuntimeError):
        builder.color()
    with test.assertRaises(RuntimeError):
        builder.set_coloring(builder.particle_color_groups)
    with test.assertRaises(NotImplementedError):
        newton.ModelBuilder().add_builder(builder)

    model = builder.finalize(device=device)
    capacity = model.cut_capacity
    test.assertEqual(capacity["color_count"], colors)
    test.assertEqual(capacity["idle_tet"], tet_count)
    flags = model.particle_flags.numpy()[tissue_count:]
    test.assertTrue((flags == 0).all())
    test.assertTrue((model.particle_mass.numpy()[tissue_count:] == 0.0).all())
    test.assertTrue((model.tet_materials.numpy()[tet_count, :2] == 0.0).all())
    spare_tris = model.tri_indices.numpy().reshape(-1, 3)[tri_count:]
    test.assertTrue((spare_tris[:, 0] == spare_tris[:, 1]).all() and (spare_tris[:, 1] == spare_tris[:, 2]).all())
    test.assertTrue(np.isinf(model.vbd.cut_force.numpy()).all())


def test_cut_blade_validation(test, device):
    """Verify that add_cut_blade() rejects wrong shapes and directions and normalizes the direction."""
    builder = _builder_with_block()
    body = builder.add_body(xform=wp.transform_identity())
    capsule = builder.add_shape_capsule(body, radius=1.0e-3, half_height=5.0e-3)
    box = builder.add_shape_box(body, hx=1.0e-3, hy=1.0e-3, hz=1.0e-3)
    with test.assertRaises(ValueError):
        builder.add_cut_blade(body, box, cut_direction=(1.0, 0.0, 0.0))
    with test.assertRaises(ValueError):
        builder.add_cut_blade(body, capsule, cut_direction=(0.0, 0.0, 1.0))  # parallel to the capsule axis
    with test.assertRaises(IndexError):
        builder.add_cut_blade(body + 5, capsule, cut_direction=(1.0, 0.0, 0.0))
    blade = builder.add_cut_blade(body, capsule, cut_direction=(1.0, 0.0, 1.0))
    test.assertEqual(blade, 0)
    direction = np.array(builder._cut["blade_direction"][0])
    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1.0e-6)
    with test.assertRaises(ValueError):
        builder.add_cut_blade(body, capsule, cut_direction=(1.0, 0.0, 0.0))  # the shape already has a blade


def test_solver_rules(test, device):
    """Verify that the solver constructor rejects settings a cuttable model cannot run with."""
    # A blade without cut capacity.
    builder = _builder_with_block()
    _add_blade(builder)
    builder.color()
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(ValueError, "needs cut capacity"):
        _solver(model)

    def scene_model():
        b = _builder_with_block()
        _add_blade(b)
        b.color()
        _add_capacity(b)
        m = b.finalize(device=device)
        force = m.vbd.cut_force.numpy()
        force[:] = 0.5
        m.vbd.cut_force.assign(force)
        return m

    with test.assertRaisesRegex(ValueError, "filter_threshold"):
        _solver(scene_model(), particle_topological_contact_filter_threshold=2)
    with test.assertRaisesRegex(ValueError, "particle_enable_self_contact"):
        _solver(scene_model(), particle_enable_self_contact=False)
    with test.assertRaisesRegex(ValueError, "rigid_soft_enable_dat"):
        _solver(scene_model(), rigid_soft_enable_dat=False)
    with test.assertRaisesRegex(ValueError, "exclusion_radius"):
        _solver(scene_model(), particle_rest_shape_contact_exclusion_radius=1.0e-3)
    model = scene_model()
    with test.assertRaisesRegex(ValueError, "full_surface_contact"):
        SolverVBD(
            model,
            rigid_soft_enable_dat=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=1.0e-3,
            particle_self_contact_gap=1.0e-3,
            particle_topological_contact_filter_threshold=1,
            particle_edge_parallel_epsilon=1.0e-7,
            collision_pipeline=newton.CollisionPipeline(model, soft_contact_gap=1.0e-3),
        )
    # A parking lattice closer than the self-contact query distance.
    b = _builder_with_block()
    b.color()
    _add_capacity(b, park_spacing=1.0e-3)
    with test.assertRaisesRegex(ValueError, "Parking particles"):
        _solver(b.finalize(device=device))
    # A parking lattice on a shape that collides with particles.
    b = _builder_with_block()
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.05, -0.03), wp.quat_identity()), hx=0.01, hy=0.01, hz=0.01)
    b.color()
    _add_capacity(b)
    with test.assertRaisesRegex(ValueError, "parking lattice"):
        _solver(b.finalize(device=device))
    # Spare particles with another radius than the cuttable tissue.
    b = _builder_with_block()
    b.color()
    _add_capacity(b, particle_radius=4.0e-4)
    m = b.finalize(device=device)
    force = m.vbd.cut_force.numpy()
    force[:] = 0.5
    m.vbd.cut_force.assign(force)
    with test.assertRaisesRegex(ValueError, "radius"):
        _solver(m)


def test_capacity_does_not_change_the_uncut_motion(test, device):
    """Spare particles, spare rows and copy colors leave the motion of the uncut tissue as it is."""

    def run(with_capacity):
        builder = _builder_with_block()
        tissue_count = builder.particle_count
        for i in range(tissue_count):
            builder.particle_qd[i] = wp.vec3(0.0, 0.0, -0.05 * math.sin(40.0 * builder.particle_q[i][0]))
        builder.color()
        if with_capacity:
            _add_capacity(builder)
        model = builder.finalize(device=device)
        solver = _solver(model)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(20):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
        return state_0.particle_q.numpy()[:tissue_count]

    plain = run(False)
    with_capacity = run(True)
    test.assertTrue(np.isfinite(with_capacity).all())
    np.testing.assert_allclose(with_capacity, plain, atol=2.0e-6)


def _check_through_cut(test, scene):
    """Assert a complete cut on the middle grid plane with consistent topology."""
    report = scene.solver.cut_report()
    test.assertEqual(report["cut_faces"], PLANE_FACES)
    test.assertEqual(sum(report["copies_per_pool"]), PLANE_VERTICES)
    test.assertEqual(report["triangle_rows_taken"], 2 * PLANE_FACES)
    test.assertEqual(report["edge_rows_taken"], 3 * PLANE_FACES)
    test.assertEqual(sum(report["overflow"]), 0)
    test.assertEqual(report["unsafe"], 0)
    half = scene.tissue_tets // 2
    test.assertEqual(scene.parts(), [half, half])
    test.assertTrue(np.isfinite(scene.state_0.particle_q.numpy()).all())
    _check_consistent_topology(test, scene)


def test_gate_arm_cuts_through_row_by_row(test, device):
    """Verify that the gate arm cuts one tet row at a time, first a partial cut and then a through cut."""
    scene = _Scene(device, cut_gate_wound=True)
    mass_before = float(scene.model.particle_mass.numpy().sum())
    scene.run(60)
    # A partial cut: the first tet row only. The body is one part and the cut front is not split.
    report = scene.solver.cut_report()
    test.assertEqual(report["cut_faces"], 2 * DIMS[1])
    test.assertEqual(sum(report["copies_per_pool"]), DIMS[1] + 1)
    test.assertEqual(sum(report["overflow"]), 0)
    test.assertEqual(report["unsafe"], 0)
    test.assertEqual(scene.parts(), [scene.tissue_tets])
    _check_consistent_topology(test, scene)
    test.assertTrue((scene.lip_gaps() > 0.0).all())
    scene.run(THROUGH_FRAMES - 60)
    _check_through_cut(test, scene)
    test.assertAlmostEqual(
        float(scene.model.particle_mass.numpy().sum()), mass_before, delta=1.0e-6 * mass_before + 1e-12
    )
    # Every copy stays on the +x side of its original: the two wound faces did not cross.
    gaps = scene.lip_gaps()
    test.assertEqual(len(gaps), PLANE_VERTICES)
    test.assertTrue((gaps > 0.0).all())


def test_wound_arm_cuts_through(test, device):
    """Verify that the wound arm cuts the block through without a gate."""
    scene = _Scene(device, cut_press_wound=True)
    scene.run(THROUGH_FRAMES)
    _check_through_cut(test, scene)


def test_designed_rule_cuts_the_first_row(test, device):
    """Verify the advance rule as designed (both forks off): the first tet row is cut and the state stays valid."""
    scene = _Scene(device)
    scene.run(60)
    report = scene.solver.cut_report()
    test.assertGreaterEqual(report["cut_faces"], 2 * DIMS[1])
    test.assertEqual(sum(report["overflow"]), 0)
    test.assertEqual(report["unsafe"], 0)
    test.assertTrue(np.isfinite(scene.state_0.particle_q.numpy()).all())
    _check_consistent_topology(test, scene)
    test.assertTrue((scene.lip_gaps() > 0.0).all())


def test_stepped_cut_first_advance(test, device):
    """Verify a cut with the blade 15 degrees off the grid: the first advance gives a stepped partial cut."""
    scene = _Scene(device, yaw_deg=15.0, cut_gate_wound=True)
    for _ in range(80):
        scene.run(1)
        if scene.solver.cut_report()["cut_faces"] > 0:
            break
    report = scene.solver.cut_report()
    test.assertGreater(report["cut_faces"], 0)
    test.assertEqual(sum(report["overflow"]), 0)
    test.assertEqual(report["unsafe"], 0)
    test.assertEqual(scene.parts(), [scene.tissue_tets])
    test.assertTrue(np.isfinite(scene.state_0.particle_q.numpy()).all())
    _check_consistent_topology(test, scene)


def test_no_cut_below_the_cut_force(test, device):
    """Verify that a blade force below vbd.cut_force cuts nothing."""
    scene = _Scene(device, cut_force=50.0, cut_gate_wound=True)
    scene.run(60, max_travel=2.5e-3)
    report = scene.solver.cut_report()
    test.assertEqual(report["cut_faces"], 0)
    test.assertEqual(sum(report["copies_per_pool"]), 0)
    test.assertGreater(report["blade_force"][0], 0.2)
    test.assertEqual(scene.parts(), [scene.tissue_tets])


def test_armed_cut_on_first_contact(test, device):
    """Verify that cut_armed cuts at the first contact with force, whatever the cut force."""
    scene = _Scene(device, cut_force=50.0, cut_armed=True, cut_gate_wound=True)
    scene.run(THROUGH_FRAMES)
    _check_through_cut(test, scene)


def test_reset_undoes_the_cut(test, device):
    """Verify that a particle reset restores every array a cut edits and that the block can be cut again."""
    scene = _Scene(device, cut_gate_wound=True)
    model = scene.model
    before = {
        "tet_indices": model.tet_indices.numpy().copy(),
        "tri_indices": model.tri_indices.numpy().copy(),
        "edge_indices": model.edge_indices.numpy().copy(),
        "particle_mass": model.particle_mass.numpy().copy(),
        "particle_flags": model.particle_flags.numpy().copy(),
        "particle_q": model.particle_q.numpy().copy(),
        "v_adj_tets": scene.solver.particle_adjacency.v_adj_tets.numpy().copy(),
    }
    scene.run(THROUGH_FRAMES)
    test.assertEqual(scene.parts(), [scene.tissue_tets // 2, scene.tissue_tets // 2])
    scene.solver.reset(scene.state_0)
    np.testing.assert_array_equal(model.tet_indices.numpy(), before["tet_indices"])
    np.testing.assert_array_equal(model.tri_indices.numpy(), before["tri_indices"])
    np.testing.assert_array_equal(model.edge_indices.numpy(), before["edge_indices"])
    np.testing.assert_array_equal(model.particle_mass.numpy(), before["particle_mass"])
    np.testing.assert_array_equal(model.particle_flags.numpy(), before["particle_flags"])
    np.testing.assert_array_equal(model.particle_q.numpy(), before["particle_q"])
    np.testing.assert_array_equal(scene.solver.particle_adjacency.v_adj_tets.numpy(), before["v_adj_tets"])
    test.assertEqual(scene.solver.cut_report()["cut_faces"], 0)
    test.assertEqual(scene.parts(), [scene.tissue_tets])
    # The block can be cut again after the reset.
    scene.state_0.body_qd.zero_()
    body_q = scene.state_0.body_q.numpy()
    body_q[scene.blade] = wp.transform(wp.vec3(*scene.start), wp.quat_identity())
    scene.state_0.body_q.assign(body_q)
    scene.frame = 0
    scene.run(THROUGH_FRAMES)
    _check_through_cut(test, scene)


def test_graph_capture_matches_uncaptured(test, device):
    """Verify that a captured frame replays the cut: positions right after the first cut advance, then the whole cut."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("CUDA graph capture needs a CUDA device")

    def run(capture):
        scene = _Scene(device, cut_force=50.0, cut_armed=True, cut_gate_wound=True)
        graph = None
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                scene.frame_body()
            graph = cap.graph
        else:
            scene.frame_body()
        scene.frame = 1
        first_advance = None
        settled = None
        while scene.frame < THROUGH_FRAMES:
            scene.set_holder(4.0e-3 * scene.frame / FPS)
            if graph is not None:
                wp.capture_launch(graph)
            else:
                scene.frame_body()
            scene.frame += 1
            if first_advance is None and scene.solver.cut_report()["cut_faces"] > 0:
                first_advance = scene.frame
            if first_advance is not None and scene.frame == first_advance + 20:
                settled = scene.corner_positions().copy()
        return scene, first_advance, settled

    plain, plain_first, plain_settled = run(False)
    captured, captured_first, captured_settled = run(True)
    # The first cut advance is replayed from the graph in the same frame. The armed cut fires at the first
    # contact with any force, and the substep of that contact differs from run to run (two plain runs differ by
    # 0.2 mm in that frame). Twenty frames later two runs agree to about 1e-8 m, so the positions are compared there.
    test.assertEqual(captured_first, plain_first)
    test.assertLess(float(np.abs(captured_settled - plain_settled).max()), 1.0e-6)
    _check_through_cut(test, captured)
    _check_through_cut(test, plain)
    np.testing.assert_array_equal(captured.solver._cut_pool_taken.numpy(), plain.solver._cut_pool_taken.numpy())
    np.testing.assert_array_equal(captured.solver._tet_neighbors.numpy(), plain.solver._tet_neighbors.numpy())
    # Run-to-run differences of the GPU contact sums grow over the whole cut; the end state stays close.
    test.assertLess(float(np.abs(captured.corner_positions() - plain.corner_positions()).max()), 1.0e-3)


# =====================================================================
# The XFEM-type cut (cut_method="xfem")
# =====================================================================

# The blade plane is in the middle of a cell column: it crosses the five tets of every cell of the column.
XFEM_OFFSET = 0.5 * CELL
XFEM_TETS = 5 * DIMS[1] * DIMS[2]
XFEM_EDGES = (DIMS[1] + 1) * (DIMS[2] + 1) + DIMS[1] * (DIMS[2] + 1) + (DIMS[1] + 1) * DIMS[2]
XFEM_VERTICES = 2 * (DIMS[1] + 1) * (DIMS[2] + 1)
# Per cell, four corner tets with one corner alone on a side (one wound triangle per side) and one
# centre tet with two corners on each side (two wound triangles per side).
XFEM_WOUND_TRIANGLES = 12 * DIMS[1] * DIMS[2]
XFEM_CAPACITY = {"max_vertices": 24, "max_faces": 72, "max_edges": 300, "max_tets": 24}
# How far the two fixed ends move toward each other when the wound is pressed closed [m]: past the start
# gap (1 mm), so that the wound faces pass through each other when they have no contact.
XFEM_PRESS = 1.5e-3
XFEM_PRESS_FRAMES = 30
XFEM_HOLD_FRAMES = 30


def _xfem_scene(device, **kwargs):
    """The block with the blade plane in the middle of a cell column, the gate arm and the XFEM-type cut."""
    options = {
        "offset": XFEM_OFFSET,
        "capacity": XFEM_CAPACITY,
        "cut_method": "xfem",
        "cut_gate_wound": True,
        # The small rows of the wound fill the self-contact lists faster than the tissue skin does.
        "particle_vertex_contact_buffer_size": 64,
        "particle_edge_contact_buffer_size": 128,
    }
    options.update(kwargs)
    with warnings.catch_warnings():
        # The scenes run where the solver warns about the limit of the blade force (cut force 0.5 N, one
        # contact record 0.25 N). test_xfem_capacity_and_rules checks the warning itself.
        warnings.filterwarnings("ignore", message=".*largest vbd.cut_force.*", category=NewtonWarning)
        return _Scene(device, **options)


def _xfem_plane_normal(scene):
    yaw = math.radians(scene.yaw_deg)
    return np.array([math.cos(yaw), math.sin(yaw), 0.0])


def _xfem_rest_levels(scene):
    """Signed distance of every tissue vertex of the rest shape from the blade plane of the scene [m], and the tissue tets."""
    rest = scene.solver._cut_saved["particle_q"].numpy().astype(np.float64)
    tets = scene.solver._cut_saved["tet_indices"].numpy().reshape(-1, 4)[: scene.tissue_tets]
    level = (rest - np.array([scene.start[0], scene.start[1], 0.0])) @ _xfem_plane_normal(scene)
    return level, tets


def _xfem_nominal_tets(scene):
    """Tissue tets of the rest shape that the blade plane of the scene crosses."""
    level, tets = _xfem_rest_levels(scene)
    side = level[tets] >= 0.0
    return np.flatnonzero(side.any(axis=1) & ~side.all(axis=1))


def _xfem_expected(scene):
    """Counts of a through cut, from the rest shape and the blade plane of the scene.

    ``limited_edges`` are the crossed mesh edges that are too short along the plane normal to hold
    the start gap, ``clamped_vertices`` the vertices of crossed tets nearer to the plane than the
    level clamp.
    """
    solver = scene.solver
    level, tets = _xfem_rest_levels(scene)
    crossed = tets[_xfem_nominal_tets(scene)]
    side_b = (level[crossed] >= 0.0).sum(axis=1)
    pairs = np.array(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    edges = np.unique(np.sort(crossed[:, pairs], axis=2).reshape(-1, 2), axis=0)
    edges = edges[(level[edges[:, 0]] >= 0.0) != (level[edges[:, 1]] >= 0.0)]
    span = np.abs(level[edges[:, 1]] - level[edges[:, 0]])
    vertices = np.unique(crossed)
    return {
        "tets": len(crossed),
        "edges": len(edges),
        "vertices": len(vertices),
        "wound_triangles": int(np.where(side_b == 2, 4, 2).sum()),
        "limited_edges": int((span * (1.0 - 2.0 * solver.cut_xfem_min_fraction) < solver._cut_start_gap).sum()),
        "clamped_vertices": int((np.abs(level[vertices]) < solver._xfem_level_min).sum()),
    }


XFEM_EXPECTED = {
    "tets": XFEM_TETS,
    "edges": XFEM_EDGES,
    "vertices": XFEM_VERTICES,
    "wound_triangles": XFEM_WOUND_TRIANGLES,
    "limited_edges": 0,
    "clamped_vertices": 0,
}


def _xfem_parts(scene):
    """Counts of tets and fragment rows in each connected part of the tissue."""
    tet_side = scene.solver._xfem["tet_side"].numpy()
    tets = scene.model.tet_indices.numpy().reshape(-1, 4)
    rows = np.flatnonzero(tet_side >= 0)
    parent = np.arange(scene.model.particle_count)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in rows:
        root = find(int(tets[row, 0]))
        for v in tets[row, 1:]:
            parent[find(int(v))] = root
    counts = {}
    for row in rows:
        root = find(int(tets[row, 0]))
        counts[root] = counts.get(root, 0) + 1
    return sorted(counts.values())


def _xfem_live_front_rows(scene):
    """The two particles of every front row that is in use, shape [count, 2]."""
    edges = scene.model.edge_indices.numpy().reshape(-1, 4)[:, 2:]
    front = edges[scene.solver._xfem["edge_kind"].numpy() == XFEM_EDGE_FRONT]
    return front[front[:, 0] != front[:, 1]]


def _closest_point_on_triangle(p, a, b, c):
    """Closest point to ``p`` on the triangle ``(a, b, c)`` (Ericson, Real-Time Collision Detection, 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
    return a + (ab * vb + ac * vc) / (va + vb + vc)


def _xfem_wound_triangles(scene):
    """The wound triangle rows in use: their particles, shape [count, 3], and the side of each row."""
    x = scene.solver._xfem
    tri = scene.model.tri_indices.numpy().reshape(-1, 3)
    flags = scene.model.particle_flags.numpy()
    rows = np.flatnonzero((x["tri_kind"].numpy() == XFEM_ROW_WOUND) & (flags[tri[:, 0]] != 0))
    return tri[rows].copy(), x["tri_sub"].numpy()[rows] // 2


def _xfem_wound_measures(scene, wound_triangles):
    """Pass-through measures of the two wound faces [m].

    ``gap_min`` is the smallest distance along the blade plane normal from the side A wound particle of
    a cut mesh edge to its side B wound particle. ``face_distance`` is the smallest signed distance of
    a wound particle from the wound face of the other side: the distance to the nearest of
    ``wound_triangles`` (particles and side of each row) of that side, negative when the particle is
    behind it. A value of zero or less means that the wound faces passed through each other.
    """
    x = scene.solver._xfem
    q = scene.state_0.particle_q.numpy().astype(np.float64)
    edge_wound = x["edge_wound"].numpy()
    edge_wound = edge_wound[edge_wound[:, 0] >= 0]
    gap = (q[edge_wound[:, 1]] - q[edge_wound[:, 0]]) @ _xfem_plane_normal(scene)
    triangles, triangle_side = wound_triangles
    wound_side = x["wound_side"].numpy()
    face_distance = math.inf
    for side in (0, 1):
        corners = q[triangles[triangle_side == side]]
        for point in q[wound_side == 1 - side]:
            nearest = math.inf
            signed = 0.0
            for a, b, c in corners:
                offset = point - _closest_point_on_triangle(point, a, b, c)
                distance = float(np.linalg.norm(offset))
                if distance < nearest:
                    nearest = distance
                    signed = distance if offset @ np.cross(b - a, c - a) >= 0.0 else -distance
            face_distance = min(face_distance, signed)
    return {"gap_min": float(gap.min()), "gap_max": float(gap.max()), "face_distance": face_distance}


def _xfem_park_wound_rows(scene):
    """Park every triangle row and edge row that names a wound particle (test only).

    The wound faces then have no collision rows, as if their self contact were switched off. The rows
    stay parked while no cut advance writes them again.
    """
    model = scene.model
    capacity = model.cut_capacity
    wound = scene.solver._xfem["wound_edge"].numpy() >= 0
    park = model.cut_row_park_particle.numpy()
    tri = model.tri_indices.numpy().reshape(-1, 3).copy()
    for row in np.flatnonzero(wound[tri].any(axis=1)):
        tri[row] = park[max(row - capacity["tri_spare_start"], 0)]
    model.tri_indices.assign(tri)
    edge = model.edge_indices.numpy().reshape(-1, 4).copy()
    for row in np.flatnonzero(wound[edge[:, 2:]].any(axis=1)):
        edge[row, 2:] = park[capacity["tri_spare_count"] + max(row - capacity["edge_spare_start"], 0)]
    model.edge_indices.assign(edge)


def _check_xfem_topology(test, scene):
    """Tet records, fragment rows, virtual corners, wound particles and surface rows are consistent."""
    model = scene.model
    solver = scene.solver
    x = solver._xfem
    capacity = model.cut_capacity
    tets = model.tet_indices.numpy().reshape(-1, 4)
    tet_side = x["tet_side"].numpy()
    tet_twin = x["tet_twin"].numpy()
    flags = model.particle_flags.numpy()
    mass = model.particle_mass.numpy()
    ghost = x["particle_ghost"].numpy() != 0
    wound = x["wound_edge"].numpy() >= 0
    # Every corner of a row in use has exactly one tet record, on the particle it names.
    pairs = solver.particle_adjacency.v_adj_tets.numpy().reshape(-1, 2)
    offsets = solver.particle_adjacency.v_adj_tets_offsets.numpy()
    owner = np.repeat(np.arange(model.particle_count), (offsets[1:] - offsets[:-1]) // 2)
    live = pairs[:, 0] != solver._cut_idle_tet
    np.testing.assert_array_equal(tets[pairs[live, 0], pairs[live, 1]], owner[live])
    seen = np.zeros((model.tet_count, 4), dtype=np.int32)
    np.add.at(seen, (pairs[live, 0], pairs[live, 1]), 1)
    test.assertTrue((seen[tet_side >= 0] == 1).all())
    test.assertTrue((seen[tet_side < 0] == 0).all())
    # Fragment rows come in twins with the same rest pose and the two shares of the tet stiffness.
    first = np.flatnonzero(tet_side == 1)
    test.assertEqual(int((tet_side == 2).sum()), len(first))
    np.testing.assert_array_equal(tet_twin[tet_twin[first]], first)
    test.assertTrue((tet_side[tet_twin[first]] == 2).all())
    poses = model.tet_poses.numpy()
    np.testing.assert_array_equal(poses[tet_twin[first]], poses[first])
    materials = model.tet_materials.numpy()
    saved = solver._cut_saved["tet_materials"].numpy()
    share = x["tet_share"].numpy()
    np.testing.assert_allclose(share[first] + share[tet_twin[first]], 1.0, atol=1.0e-6)
    # The edge parameter limit bounds the share of a fragment from below.
    smallest_share = solver.cut_xfem_min_fraction**3
    if len(first) > 0:
        test.assertGreaterEqual(float(share[first].min()), smallest_share - 1.0e-6)
        test.assertGreaterEqual(float(share[tet_twin[first]].min()), smallest_share - 1.0e-6)
    np.testing.assert_allclose(materials[first] + materials[tet_twin[first]], saved[first], rtol=1.0e-5)
    # A virtual corner and a wound particle are solved particles.
    for particles in (np.flatnonzero(ghost), np.flatnonzero(wound)):
        test.assertTrue(((flags[particles] & int(newton.ParticleFlags.ACTIVE)) != 0).all())
        test.assertTrue((mass[particles] > 0.0).all())
    test.assertFalse((ghost & wound).any())
    # A spare in use is a virtual corner or a wound particle: this method makes no copy that is material.
    spare_count = capacity["spare_count"]
    pools = np.concatenate([np.arange(start, start + spare_count) for start in model.cut_spare_start.numpy()])
    active_spares = pools[(flags[pools] & int(newton.ParticleFlags.ACTIVE)) != 0]
    test.assertTrue((ghost | wound)[active_spares].all())
    # Every crossed edge of a cut tet has its two wound particles: no cut tet is without wound rows.
    test.assertEqual(int((x["edge_cut"].numpy() == 2).sum()), 0)
    edge_wound = x["edge_wound"].numpy()
    test.assertTrue((edge_wound[x["edge_cut"].numpy() == 1] >= 0).all())
    # A tie target is inside [t_min, 1 - t_min] of its edge.
    tie_s = x["tie_s"].numpy()[wound]
    if len(tie_s) > 0:
        test.assertGreaterEqual(float(tie_s.min()), solver.cut_xfem_min_fraction - 1.0e-6)
        test.assertLessEqual(float(tie_s.max()), 1.0 - solver.cut_xfem_min_fraction + 1.0e-6)
    # A wound particle never has the color of a corner of its tie.
    colors = model.particle_colors.numpy()
    owner_row = x["edge_owner"].numpy()
    owner_local = x["edge_owner_local"].numpy()
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    wound_edge = x["wound_edge"].numpy()
    wound_side = x["wound_side"].numpy()
    for p in np.flatnonzero(wound):
        row = owner_row[wound_edge[p], wound_side[p]]
        a, b = edge_pairs[owner_local[wound_edge[p], wound_side[p]]]
        test.assertNotEqual(colors[p], colors[tets[row, a]])
        test.assertNotEqual(colors[p], colors[tets[row, b]])
    # A surface row is all parked or all live and names no virtual corner. No edge of a live row is so
    # short that it acts as a point in edge-edge self contact.
    parked = np.zeros(model.particle_count, dtype=bool)
    parked[capacity["particle_start"] : capacity["particle_end"]] = True
    parked &= (flags & int(newton.ParticleFlags.ACTIVE)) == 0
    tri = model.tri_indices.numpy().reshape(-1, 3)
    edge = model.edge_indices.numpy().reshape(-1, 4)[:, 2:]
    tri_parked = parked[tri].sum(axis=1)
    edge_parked = parked[edge].sum(axis=1)
    test.assertTrue(((tri_parked == 0) | (tri_parked == 3)).all())
    test.assertTrue(((edge_parked == 0) | (edge_parked == 2)).all())
    live_tri = tri[tri_parked == 0]
    live_edge = edge[edge_parked == 0]
    test.assertFalse(ghost[live_tri].any())
    test.assertFalse(ghost[live_edge].any())
    q = scene.state_0.particle_q.numpy()
    corners = q[live_tri]
    shortest = math.sqrt(EDGE_PARALLEL_EPSILON)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        test.assertGreater(float(np.linalg.norm(corners[:, i] - corners[:, j], axis=1).min()), shortest)
    test.assertGreater(float(np.linalg.norm(q[live_edge[:, 0]] - q[live_edge[:, 1]], axis=1).min()), shortest)
    # The normal of a wound triangle points from the side of its fragment to the far side (rest shape).
    rest = model.particle_q.numpy()
    plane_normal = _xfem_plane_normal(scene)
    kind = x["tri_kind"].numpy()
    sub = x["tri_sub"].numpy()
    wound_rows = np.flatnonzero((kind == XFEM_ROW_WOUND) & (tri_parked == 0))
    a, b, c = (rest[tri[wound_rows, k]] for k in range(3))
    toward_b = np.cross(b - a, c - a) @ plane_normal
    test.assertTrue((toward_b[sub[wound_rows] // 2 == 0] > 0.0).all())
    test.assertTrue((toward_b[sub[wound_rows] // 2 == 1] < 0.0).all())
    return len(wound_rows)


def _check_xfem_through_cut(test, scene, mass_before, expected=None):
    """Assert that every tet that the blade plane crosses is cut once, that the tissue has two parts and that mass stays.

    ``expected`` holds the counts of the cut (see :func:`_xfem_expected`); the default is the cut in
    the middle of a cell column.
    """
    expected = XFEM_EXPECTED if expected is None else expected
    solver = scene.solver
    report = solver.cut_report()
    x = solver._xfem
    cut_rows = np.flatnonzero(x["tet_side"].numpy()[: scene.tissue_tets] == 1)
    np.testing.assert_array_equal(cut_rows, _xfem_nominal_tets(scene))
    test.assertEqual(report["method"], "xfem")
    test.assertEqual(report["cut_tets"], expected["tets"])
    test.assertEqual(report["spare_tets_taken"], expected["tets"])
    test.assertEqual(len(np.unique(x["tet_slot"].numpy()[cut_rows])), expected["tets"])
    test.assertEqual(report["virtual_corners"], expected["vertices"])
    test.assertEqual(report["wound_particles"], 2 * expected["edges"])
    test.assertEqual(sum(report["copies_per_pool"]), expected["vertices"] + 2 * expected["edges"])
    # No spare is lost: the counter of every pool equals the spares of the pool that are in use.
    test.assertEqual(report["copies_per_pool"], solver._cut_pool_taken.numpy().tolist())
    test.assertEqual(sum(report["overflow"]), 0)
    test.assertEqual(len(_xfem_parts(scene)), 2)
    test.assertEqual(sum(_xfem_parts(scene)), scene.tissue_tets + expected["tets"])
    test.assertTrue(np.isfinite(scene.state_0.particle_q.numpy()).all())
    mass_after = float(scene.model.particle_mass.numpy().sum())
    test.assertAlmostEqual(mass_after, mass_before, delta=1.0e-6 * mass_before)
    wound_rows = _check_xfem_topology(test, scene)
    # Two wound triangles (one per side) for a tet with one corner alone, four for a tet with two and two.
    test.assertEqual(wound_rows, expected["wound_triangles"])
    # The start gap. The two tie targets of a cut mesh edge are (s_b - s_a) of the edge apart. With the
    # span of the edge along the blade plane normal, from the raw signed distances of its two vertices,
    # that is the start gap in metres. An edge that is too short along the normal holds
    # (1 - 2 * t_min) of its span, and cut_report() counts it.
    edge_wound = x["edge_wound"].numpy()
    has_wound = edge_wound[:, 0] >= 0
    limited = x["edge_gap_limited"].numpy()[has_wound] != 0
    mesh_edge = solver._xfem_mesh_edge.numpy()[has_wound]
    edge_wound = edge_wound[has_wound]
    raw = x["particle_level_raw"].numpy().astype(np.float64)
    span = np.abs(raw[mesh_edge[:, 1]] - raw[mesh_edge[:, 0]])
    tie_s = x["tie_s"].numpy().astype(np.float64)
    held = (tie_s[edge_wound[:, 1]] - tie_s[edge_wound[:, 0]]) * span
    start_gap = solver._cut_start_gap
    range_limit = 1.0 - 2.0 * solver.cut_xfem_min_fraction
    test.assertEqual(report["start_gap_limited_edges"], expected["limited_edges"])
    test.assertEqual(int(limited.sum()), expected["limited_edges"])
    np.testing.assert_allclose(held[~limited], start_gap, rtol=1.0e-3)
    np.testing.assert_allclose(held[limited], range_limit * span[limited], rtol=1.0e-3)
    test.assertTrue((held[limited] < start_gap).all())
    # The same gap from positions: the cut writes the rest position of a wound particle at its tie target.
    # The level values are taken under the blade, at the advance that first needs each vertex, so the rest
    # shape differs from them by the strain and the motion of the tissue (measured: 0.74 to 1.16 of the
    # start gap in these scenes).
    plane_normal = _xfem_plane_normal(scene)
    rest = scene.model.particle_q.numpy()
    rest_gap = (rest[edge_wound[:, 1]] - rest[edge_wound[:, 0]]) @ plane_normal
    test.assertGreater(float(rest_gap[~limited].min()), 0.6 * start_gap)
    test.assertLess(float(rest_gap[~limited].max()), 1.4 * start_gap)
    test.assertGreater(float(rest_gap.min()), 0.0)
    # The two wound particles of one mesh edge stay apart along the blade plane normal: the wound faces
    # did not cross. The tie holds every wound particle near its target.
    q = scene.state_0.particle_q.numpy()
    gap = (q[edge_wound[:, 1]] - q[edge_wound[:, 0]]) @ plane_normal
    test.assertGreater(float(gap.min()), 0.0)
    test.assertLess(report["tie_residual_max"], 1.0e-4)
    # No front edge is left: every front row is parked and every wound particle has the gate bit of the cut.
    test.assertEqual(len(_xfem_live_front_rows(scene)), 0)
    gate = scene.model.particle_contact_gate.numpy()
    test.assertTrue(((gate[x["wound_edge"].numpy() >= 0] >> 30) & 1 == 1).all())
    return report


@wp.kernel
def _evaluate_side_a_share(levels: wp.array[wp.vec4], t_min: float, shares: wp.array[float]):
    i = wp.tid()
    shares[i] = tet_side_a_share(levels[i], t_min)


def test_xfem_material_share(test, device):
    """Verify the side A volume share of a crossed tet against analytic values and a Monte Carlo estimate."""
    rng = np.random.default_rng(7)
    cases = [
        (-1.0, 1.0, 1.0, 1.0),  # one corner alone on side A, cut at the edge middles: 1/8
        (1.0, -1.0, -1.0, -1.0),  # one corner alone on side B: 7/8
        (-1.0, -1.0, 1.0, 1.0),  # two and two, cut at the edge middles: 1/2
        (-1.0, 3.0, 3.0, 3.0),  # one corner alone, edge parameter 1/4: 1/64
        (2.0, -1.0, 0.5, -3.0),
        (-0.3, 1.2, -2.0, 0.7),
        (0.4, 0.9, -1.5, -0.2),
        (-2.5, 0.1, 0.2, 1.0),
        (1.0, 1.0, 1.0, 1.0),  # not crossed, side B
        (-1.0, -1.0, -1.0, -1.0),  # not crossed, side A
    ]
    levels = np.array(cases, dtype=np.float32)
    shares = wp.zeros(len(cases), dtype=float, device=device)
    wp.launch(
        _evaluate_side_a_share,
        dim=len(cases),
        inputs=[wp.array(levels, dtype=wp.vec4, device=device), 1.0e-6],
        outputs=[shares],
        device=device,
    )
    shares = shares.numpy()
    np.testing.assert_allclose(shares[:4], [1.0 / 8.0, 7.0 / 8.0, 0.5, 1.0 / 64.0], atol=1.0e-6)
    np.testing.assert_allclose(shares[8:], [0.0, 1.0], atol=1.0e-6)
    # Uniform points in the tet have Dirichlet(1, 1, 1, 1) barycentric weights; the level value is linear in them.
    weights = rng.dirichlet(np.ones(4), size=400000)
    estimate = (weights @ levels.astype(np.float64).T < 0.0).mean(axis=0)
    np.testing.assert_allclose(shares, estimate, atol=4.0e-3)
    # The edge parameter limit keeps the share of a fragment above zero.
    limited = wp.zeros(2, dtype=float, device=device)
    wp.launch(
        _evaluate_side_a_share,
        dim=2,
        inputs=[
            wp.array(np.array([(-1.0e-6, 1.0, 1.0, 1.0), (1.0e-6, -1.0, -1.0, -1.0)]), dtype=wp.vec4, device=device),
            0.2,
        ],
        outputs=[limited],
        device=device,
    )
    np.testing.assert_allclose(limited.numpy(), [0.2**3, 1.0 - 0.2**3], atol=1.0e-6)


def test_xfem_capacity_and_rules(test, device):
    """Verify the spare tets of add_cut_capacity() and the solver rules and warnings of the XFEM-type cut."""
    builder = _builder_with_block()
    tet_count = builder.tet_count
    builder.color()
    _add_capacity(builder, max_tets=5)
    test.assertEqual(builder.tet_count, tet_count + 6)
    model = builder.finalize(device=device)
    capacity = model.cut_capacity
    test.assertEqual(capacity["tet_spare_start"], tet_count)
    test.assertEqual(capacity["tet_spare_count"], 5)
    test.assertEqual(capacity["idle_tet"], tet_count + 5)
    test.assertTrue((model.tet_materials.numpy()[tet_count:] == 0.0).all())
    spare_tets = model.tet_indices.numpy().reshape(-1, 4)[tet_count:]
    test.assertTrue((spare_tets == spare_tets[-1]).all())  # all on the corners of the idle tet

    def scene_model(*, blade=True, cut_force=0.5, needle=False, **capacity_options):
        b = _builder_with_block()
        if blade:
            _add_blade(b, offset=XFEM_OFFSET)
        if needle:
            # A straight needle far above the block and away from the parking lattice.
            body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
            tip = b.add_shape_capsule(body, radius=4.0e-4, half_height=5.0e-3)
            b.add_suture_needle(body, tip, centerline=[(0.0, 0.0, -5.0e-3), (0.0, 0.0, 5.0e-3)])
        b.color()
        _add_capacity(b, **capacity_options)
        m = b.finalize(device=device)
        m.soft_contact_ke = 5.0e2
        m.shape_material_ke.fill_(5.0e2)
        if cut_force is not None:
            force = m.vbd.cut_force.numpy()
            force[:] = cut_force
            m.vbd.cut_force.assign(force)
        return m

    # The armed rule does not read the cut force, so these solvers give no warning about its limit.
    xfem = {"cut_method": "xfem", "cut_armed": True}
    with test.assertRaisesRegex(ValueError, "cut_method"):
        _solver(scene_model(**XFEM_CAPACITY), cut_method="planes")
    with test.assertRaisesRegex(ValueError, "max_tets"):
        _solver(scene_model(), **xfem)
    with test.assertRaisesRegex(ValueError, "max_faces"):
        _solver(scene_model(max_tets=24, max_faces=16, max_edges=300), **xfem)
    with test.assertRaisesRegex(ValueError, "max_edges"):
        _solver(scene_model(max_tets=24, max_faces=72, max_edges=100), **xfem)
    with test.assertRaisesRegex(ValueError, "copy_colors"):
        _solver(scene_model(**XFEM_CAPACITY, copy_colors=False), **xfem)
    with test.assertRaisesRegex(ValueError, "cut_xfem_min_fraction"):
        _solver(scene_model(**XFEM_CAPACITY), **xfem, cut_xfem_min_fraction=0.5)
    with test.assertRaisesRegex(ValueError, "cut_tie_stiffness"):
        _solver(scene_model(**XFEM_CAPACITY), **xfem, cut_tie_stiffness=0.0)
    # Switches of the cut along tet faces: one is an error, the others give a warning.
    with test.assertRaisesRegex(ValueError, "cut_press_wound"):
        _solver(scene_model(**XFEM_CAPACITY), **xfem, cut_press_wound=True)
    with test.assertWarnsRegex(NewtonWarning, "cut_surface_edges belongs"):
        _solver(scene_model(**XFEM_CAPACITY), **xfem, cut_surface_edges=False)
    with test.assertWarnsRegex(NewtonWarning, "cut_keep_side_labels, cut_gap_direction belong"):
        _solver(scene_model(**XFEM_CAPACITY), **xfem, cut_keep_side_labels=False, cut_gap_direction="normal_sum")
    # A front row needs an edge row beyond the ten per spare tet.
    with test.assertRaisesRegex(ValueError, "cut_xfem_front_rows"):
        _solver(scene_model(max_tets=24, max_faces=72, max_edges=240), **xfem, cut_xfem_front_rows=True)
    # The method does nothing on a model without cut capacity.
    plain = _builder_with_block()
    plain.color()
    with test.assertWarnsRegex(NewtonWarning, "no cut capacity"):
        _solver(plain.finalize(device=device), cut_method="xfem")
    # The suture does not know fragments yet.
    with test.assertWarnsRegex(NewtonWarning, "suture needles"):
        _solver(scene_model(needle=True, **XFEM_CAPACITY), **xfem)
    # The blade force has an upper limit with the penalty law: 500 N/m times 0.5 mm per contact record.
    with test.assertWarnsRegex(NewtonWarning, r"largest vbd.cut_force \(0.5 N\).*\(0.25 N"):
        _solver(scene_model(**XFEM_CAPACITY), cut_method="xfem")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # these solvers must not warn
        _solver(scene_model(cut_force=0.2, **XFEM_CAPACITY), cut_method="xfem")
        _solver(scene_model(**XFEM_CAPACITY), cut_method="xfem", rigid_soft_contact_use_log_barrier=True)
        _solver(scene_model(**XFEM_CAPACITY), cut_method="faces")
        # Spare tets do not change the tables of the cut along tet faces.
        solver = _solver(scene_model(**XFEM_CAPACITY))
        neighbors = solver._tet_neighbors.numpy()
        test.assertTrue((neighbors[tet_count:] == -1).all())
        test.assertTrue((neighbors[:tet_count] < tet_count).all())
        solver = _solver(scene_model(**XFEM_CAPACITY), **xfem, cut_gate_wound=True)
    report = solver.cut_report()
    test.assertEqual(report["cut_tets"], 0)
    test.assertEqual(report["wound_particles"], 0)
    test.assertEqual(report["start_gap_limited_edges"], 0)

    # The tables do not depend on vbd.cut_force at construction: tissue that a scene makes cuttable later
    # has its free tet records, and a reset keeps the cut force of the tissue.
    def records(solver):
        offsets = solver.particle_adjacency.v_adj_tets_offsets.numpy()
        return (offsets[1:] - offsets[:-1]) // 2

    late_model = scene_model(blade=False, cut_force=None, **XFEM_CAPACITY)
    late = _solver(late_model, **xfem)
    np.testing.assert_array_equal(records(late), records(solver))
    test.assertEqual(late._xfem_level_range, 0.0)  # no blade
    force = late_model.vbd.cut_force.numpy()
    tissue = capacity["particle_start"]
    force[:tissue] = 0.3
    force[tissue:] = 7.0  # a cut writes the cut force of the spares it takes
    late_model.vbd.cut_force.assign(force)
    state = late_model.state()
    late.reset(state)
    force = late_model.vbd.cut_force.numpy()
    test.assertTrue((force[:tissue] == np.float32(0.3)).all())
    test.assertTrue(np.isinf(force[tissue:]).all())


def test_xfem_first_advance(test, device):
    """Verify the first advance of the XFEM-type cut: a partial cut with a consistent state."""
    scene = _xfem_scene(device, cut_force=50.0, cut_armed=True)
    mass_before = float(scene.model.particle_mass.numpy().sum())
    for _ in range(40):
        scene.run(1)
        if scene.solver.cut_report()["cut_tets"] > 0:
            break
    # The blade stays for 15 frames. The new wound rows must not push on their own side: without the
    # contact filter of the cut the tie residual reached 7 mm and the tissue 760 mm/s in these frames.
    largest_residual = 0.0
    largest_speed = 0.0
    for _ in range(15):
        scene.run(1, max_travel=scene.travel)
        largest_residual = max(largest_residual, scene.solver.cut_report()["tie_residual_max"])
        active = (scene.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        speed = np.linalg.norm(scene.state_0.particle_qd.numpy(), axis=1)
        largest_speed = max(largest_speed, float(speed[active].max()))
    test.assertLess(largest_residual, 0.5e-3)
    test.assertLess(largest_speed, 0.5)
    report = scene.solver.cut_report()
    test.assertGreater(report["cut_tets"], 0)
    test.assertLess(report["cut_tets"], XFEM_TETS)
    test.assertGreater(report["wound_particles"], 0)
    test.assertEqual(sum(report["overflow"]), 0)
    test.assertEqual(len(_xfem_parts(scene)), 1)  # a partial cut: the vertices at the cut front are not split
    test.assertTrue(np.isfinite(scene.state_0.particle_q.numpy()).all())
    test.assertAlmostEqual(float(scene.model.particle_mass.numpy().sum()), mass_before, delta=1.0e-6 * mass_before)
    _check_xfem_topology(test, scene)
    cut_rows = np.flatnonzero(scene.solver._xfem["tet_side"].numpy()[: scene.tissue_tets] == 1)
    test.assertTrue(np.isin(cut_rows, _xfem_nominal_tets(scene)).all())
    # At the cut front, the wound particles of a front edge have no gate bit of the cut, so the blade
    # rests on them; every other wound particle has the bit.
    x = scene.solver._xfem
    wound_edge = x["wound_edge"].numpy()
    wound = np.flatnonzero(wound_edge >= 0)
    on_front = x["edge_front"].numpy()[wound_edge[wound]] != 0
    gate = (scene.model.particle_contact_gate.numpy()[wound] >> 30) & 1
    test.assertTrue(on_front.any() and (~on_front).any())
    np.testing.assert_array_equal(gate, np.where(on_front, 0, 1))
    test.assertEqual(len(_xfem_live_front_rows(scene)), 0)  # the front row fork is off


def test_xfem_front_rows_fork(test, device):
    """Verify the front row fork: an edge row joins the two wound particles of every front edge."""
    scene = _xfem_scene(device, cut_force=50.0, cut_armed=True, cut_xfem_front_rows=True)
    for _ in range(40):
        scene.run(1)
        if scene.solver.cut_report()["cut_tets"] > 0:
            break
    scene.run(2)
    test.assertEqual(sum(scene.solver.cut_report()["overflow"]), 0)
    _check_xfem_topology(test, scene)
    x = scene.solver._xfem
    front = _xfem_live_front_rows(scene)
    wound_edge = x["wound_edge"].numpy()
    wound_side = x["wound_side"].numpy()
    edge_front = x["edge_front"].numpy()
    test.assertGreater(len(front), 0)
    np.testing.assert_array_equal(wound_edge[front[:, 0]], wound_edge[front[:, 1]])
    test.assertTrue((wound_side[front[:, 0]] == 0).all() and (wound_side[front[:, 1]] == 1).all())
    test.assertTrue((edge_front[wound_edge[front[:, 0]]] == 1).all())
    # One row for each front edge that has wound particles.
    has_wound = x["edge_wound"].numpy()[:, 0] >= 0
    test.assertEqual(len(front), int((has_wound & (edge_front != 0)).sum()))


def test_xfem_refused_advance(test, device):
    """Verify that a cut advance that does not fit the cut capacity leaves its tets whole and loses no spare."""
    # Six spare particles per pool hold the first cut advances only (the through cut needs 60 spares).
    scene = _xfem_scene(device, cut_force=50.0, cut_armed=True, capacity={**XFEM_CAPACITY, "max_vertices": 6})
    mass_before = float(scene.model.particle_mass.numpy().sum())
    solver = scene.solver
    counts = ("cut_tets", "virtual_corners", "wound_particles", "spare_tets_taken", "copies_per_pool")
    # A refused advance: the overflow of the pools grows and nothing is cut. (The overflow also grows
    # when a virtual corner finds its pool empty; that advance still cuts its tets.)
    refused_at = solver.cut_report()
    for _ in range(80):
        previous = refused_at
        scene.run(1)
        refused_at = solver.cut_report()
        if refused_at["overflow_pools"] > previous["overflow_pools"] and all(
            refused_at[key] == previous[key] for key in counts
        ):
            break
    scene.run(2, max_travel=scene.travel)  # the blade stays; every cut update asks again and is refused
    report = solver.cut_report()
    test.assertGreater(refused_at["overflow_pools"], 0)
    test.assertGreater(report["overflow_pools"], refused_at["overflow_pools"])
    test.assertEqual(report["overflow_tet_records"], 0)
    # Nothing changed after the first refused advance, and the state is that of a partial cut.
    for key in counts:
        test.assertEqual(report[key], refused_at[key], key)
    test.assertGreater(report["cut_tets"], 0)
    test.assertLess(report["cut_tets"], XFEM_TETS)
    _check_xfem_topology(test, scene)  # includes: every crossed edge of a cut tet has its wound particles
    test.assertAlmostEqual(float(scene.model.particle_mass.numpy().sum()), mass_before, delta=1.0e-6 * mass_before)
    # No spare is lost: the spares in use are the spares that the pool counters gave out.
    spare_count = scene.model.cut_capacity["spare_count"]
    taken = np.minimum(solver._cut_pool_taken.numpy(), spare_count)
    test.assertEqual(report["copies_per_pool"], taken.tolist())
    test.assertEqual(sum(report["copies_per_pool"]), report["virtual_corners"] + report["wound_particles"])


def test_xfem_cuts_through_mid_cell(test, device):
    """Verify a through cut with the blade plane in the middle of a cell column, and the gate of the wound."""
    scene = _xfem_scene(device, capture=True)
    test.assertEqual(_xfem_expected(scene), XFEM_EXPECTED)
    mass_before = float(scene.model.particle_mass.numpy().sum())
    scene.run(THROUGH_FRAMES)
    _check_xfem_through_cut(test, scene, mass_before)
    # The stroke that made the cut has ended: the blade edge is below the block. Every wound particle and
    # every vertex of a cut tet carries the number of that stroke, and the stroke number has moved on, so
    # a later cut gets other side tags.
    x = scene.solver._xfem
    stroke = x["particle_stroke"].numpy()
    test.assertTrue((stroke[x["wound_edge"].numpy() >= 0] == 0).all())
    test.assertTrue((stroke[x["particle_level_set"].numpy() == 2] == 0).all())
    test.assertTrue((stroke[x["particle_level_set"].numpy() == 0] == -1).any())
    test.assertGreater(int(x["stroke_number"].numpy()[0]), 0)
    # The blade goes back up into the wound, to the middle of the block. Its edge is then nearer to the
    # wound particles of both wound faces than the two radii, and no particle of the tissue is at the
    # blade plane. The wound particles carry the gate bit, so the blade passes without contact force:
    # it follows its holder and does not push the wound faces apart.
    wound_triangles = _xfem_wound_triangles(scene)
    before = _xfem_wound_measures(scene, wound_triangles)
    travel = scene.travel
    middle = scene.start[2] - 0.5 * TOP
    for i in range(30):
        scene.step_frame(travel + (middle - travel) * (i + 1) / 30)
    for _ in range(10):
        scene.step_frame(middle)
    q = scene.state_0.particle_q.numpy()
    body_q = scene.state_0.body_q.numpy()
    wound = np.flatnonzero(scene.solver._xfem["wound_edge"].numpy() >= 0)
    offset = q[wound] - body_q[scene.blade, :3]
    offset[:, 1] = 0.0  # the blade edge is along y
    nearest = float(np.linalg.norm(offset, axis=1).min())
    test.assertLess(nearest, BLADE_RADIUS + 5.0e-4)  # the edge overlaps the wound particles
    test.assertLess(float(np.abs(body_q[scene.blade, :3] - body_q[scene.holder, :3]).max()), 5.0e-5)
    test.assertEqual(scene.solver.cut_report()["blade_force"][0], 0.0)
    after = _xfem_wound_measures(scene, wound_triangles)
    test.assertLess(abs(after["gap_max"] - before["gap_max"]), 1.0e-5)
    test.assertLess(abs(after["gap_min"] - before["gap_min"]), 1.0e-5)


def test_xfem_cuts_through_turned_plane(test, device):
    """Verify a through cut with the blade turned 10 degrees about the vertical axis."""
    scene = _xfem_scene(device, yaw_deg=10.0, capture=True)
    test.assertEqual(_xfem_expected(scene), XFEM_EXPECTED)  # the plane stays inside one cell column
    mass_before = float(scene.model.particle_mass.numpy().sum())
    scene.run(THROUGH_FRAMES)
    _check_xfem_through_cut(test, scene, mass_before)


# A blade plane that passes two vertex columns at 0.39 mm, one on each side: the level values of these
# vertices are clamped, the mesh edges between the two columns are too short along the plane normal for
# the start gap, and the tie targets on the edges from these vertices are moved to the edge limit.
XFEM_NEAR_YAW_DEG = 15.0
XFEM_NEAR_OFFSET = 0.4e-3
XFEM_NEAR_CAPACITY = {"max_vertices": 40, "max_faces": 120, "max_edges": 520, "max_tets": 40}


def test_xfem_cuts_through_plane_near_vertices(test, device):
    """Verify a through cut with a turned blade plane that passes near vertices: level clamp and edge limits."""
    scene = _xfem_scene(
        device, yaw_deg=XFEM_NEAR_YAW_DEG, offset=XFEM_NEAR_OFFSET, capacity=XFEM_NEAR_CAPACITY, capture=True
    )
    solver = scene.solver
    x = solver._xfem
    expected = _xfem_expected(scene)
    test.assertGreater(expected["clamped_vertices"], 0)
    test.assertGreater(expected["limited_edges"], 0)
    test.assertGreater(expected["tets"], XFEM_TETS)
    mass_before = float(scene.model.particle_mass.numpy().sum())
    largest_speed = 0.0
    for _ in range(THROUGH_FRAMES):
        scene.run(1)
        active = (scene.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        speed = np.linalg.norm(scene.state_0.particle_qd.numpy(), axis=1)
        largest_speed = max(largest_speed, float(speed[active].max()))
    report = _check_xfem_through_cut(test, scene, mass_before, expected)
    test.assertEqual(report["start_gap_limited_edges"], expected["limited_edges"])
    # The level clamp acted: stored values at the clamp, with the sign of the raw signed distance.
    level_min = solver._xfem_level_min
    level = x["particle_level"].numpy()
    raw = x["particle_level_raw"].numpy()
    vertices = np.unique(x["tet_vertex"].numpy()[x["tet_side"].numpy() == 1])
    clamped = vertices[np.abs(raw[vertices]) < level_min]
    test.assertEqual(len(clamped), expected["clamped_vertices"])
    np.testing.assert_allclose(np.abs(level[clamped]), level_min, rtol=1.0e-5)
    test.assertTrue((np.sign(level[clamped]) == np.sign(raw[clamped])).all())
    np.testing.assert_array_equal(level[np.setdiff1d(vertices, clamped)], raw[np.setdiff1d(vertices, clamped)])
    # The edge limits acted: tie targets at t_min and at 1 - t_min.
    t_min = solver.cut_xfem_min_fraction
    tie_s = x["tie_s"].numpy()[x["wound_edge"].numpy() >= 0]
    test.assertTrue(np.isclose(tie_s, t_min, atol=1.0e-6).any())
    test.assertTrue(np.isclose(tie_s, 1.0 - t_min, atol=1.0e-6).any())
    # A limited edge has both tie targets at the limits.
    edge_wound = x["edge_wound"].numpy()
    limited = np.flatnonzero((edge_wound[:, 0] >= 0) & (x["edge_gap_limited"].numpy() != 0))
    np.testing.assert_allclose(x["tie_s"].numpy()[edge_wound[limited, 0]], t_min, atol=1.0e-6)
    np.testing.assert_allclose(x["tie_s"].numpy()[edge_wound[limited, 1]], 1.0 - t_min, atol=1.0e-6)
    # The cut stays calm: no particle is thrown.
    test.assertLess(largest_speed, 1.0)


def test_xfem_pressed_wound_holds(test, device):
    """Verify the self contact of the two wound faces under load, and that the check fails without the wound rows."""
    measures = {}
    for wound_contact in (True, False):
        scene = _xfem_scene(device, capture=True)
        scene.run(THROUGH_FRAMES)
        test.assertEqual(scene.solver.cut_report()["cut_tets"], XFEM_TETS)
        wound_triangles = _xfem_wound_triangles(scene)
        test.assertEqual(len(wound_triangles[0]), XFEM_WOUND_TRIANGLES)
        if not wound_contact:
            _xfem_park_wound_rows(scene)
        # The fixed ends push the halves together, past the start gap; the blade stays below the block.
        travel = scene.travel
        residual = 0.0
        for i in range(XFEM_PRESS_FRAMES + XFEM_HOLD_FRAMES):
            scene.set_holds(-0.5 * XFEM_PRESS * min((i + 1) / XFEM_PRESS_FRAMES, 1.0))
            scene.step_frame(travel)
            residual = max(residual, scene.solver.cut_report()["tie_residual_max"])
        test.assertTrue(np.isfinite(scene.state_0.particle_q.numpy()).all())
        measures[wound_contact] = _xfem_wound_measures(scene, wound_triangles)
        measures[wound_contact]["tie_residual"] = residual
        if wound_contact:
            _check_xfem_topology(test, scene)
    held, free = measures[True], measures[False]
    start_gap = 1.0e-3
    test.assertGreater(XFEM_PRESS, start_gap)
    # With the wound rows the two wound faces rest on each other, about one contact margin apart
    # (measured: 0.966 mm and 0.983 mm).
    test.assertGreater(held["gap_min"], 0.9 * start_gap)
    test.assertGreater(held["face_distance"], 0.9 * start_gap)
    # The tie residual under load is the force on a wound particle divided by the tie stiffness
    # (measured: 0.051 mm; the bound is two times that).
    test.assertLess(held["tie_residual"], 0.1e-3)
    # Without the wound rows the same push takes the wound faces through each other: the check can fail.
    test.assertLess(free["gap_min"], 0.0)
    test.assertLess(free["face_distance"], 0.0)


def test_xfem_reset_undoes_the_cut(test, device):
    """Verify that a particle reset restores every array that the XFEM-type cut edits."""
    scene = _xfem_scene(device)
    model = scene.model
    solver = scene.solver

    def snapshot():
        arrays = {
            "tet_indices": model.tet_indices,
            "tet_poses": model.tet_poses,
            "tet_materials": model.tet_materials,
            "tri_indices": model.tri_indices,
            "edge_indices": model.edge_indices,
            "particle_mass": model.particle_mass,
            "particle_inv_mass": model.particle_inv_mass,
            "particle_flags": model.particle_flags,
            "particle_q": model.particle_q,
            "particle_contact_gate": model.particle_contact_gate,
            "cut_force": model.vbd.cut_force,
            "v_adj_tets": solver.particle_adjacency.v_adj_tets,
            "tri_source": solver._cut_tri_source,
            "edge_source": solver._cut_edge_source,
        }
        arrays.update({f"xfem_{name}": array for name, array in solver._xfem.items()})
        return {name: array.numpy().copy() for name, array in arrays.items()}

    before = snapshot()
    mass_before = float(model.particle_mass.numpy().sum())
    for _ in range(60):
        scene.run(1)
        if scene.solver.cut_report()["cut_tets"] >= XFEM_TETS // 2:
            break
    test.assertGreater(scene.solver.cut_report()["cut_tets"], 0)
    test.assertFalse(np.array_equal(model.tet_indices.numpy(), before["tet_indices"]))
    solver.reset(scene.state_0)
    after = snapshot()
    for name, values in before.items():
        np.testing.assert_array_equal(after[name], values, err_msg=name)
    report = solver.cut_report()
    test.assertEqual(report["cut_tets"], 0)
    test.assertEqual(report["wound_particles"], 0)
    test.assertEqual(report["virtual_corners"], 0)
    test.assertEqual(_xfem_parts(scene), [scene.tissue_tets])
    if not wp.get_device(device).is_cuda:
        return
    # The block can be cut again after the reset.
    scene.state_0.body_qd.zero_()
    body_q = scene.state_0.body_q.numpy()
    body_q[scene.blade] = wp.transform(wp.vec3(*scene.start), wp.quat_identity())
    scene.state_0.body_q.assign(body_q)
    scene.frame = 0
    scene.run(THROUGH_FRAMES)
    _check_xfem_through_cut(test, scene, mass_before)


def test_xfem_graph_capture_matches_uncaptured(test, device):
    """Verify that a captured frame replays the XFEM-type cut: the same cut counts as plain steps."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("CUDA graph capture needs a CUDA device")
    reports = []
    for capture in (False, True):
        scene = _xfem_scene(device, capture=capture)
        mass_before = float(scene.model.particle_mass.numpy().sum())
        scene.run(THROUGH_FRAMES)
        reports.append((_check_xfem_through_cut(test, scene, mass_before), scene))
    (plain, plain_scene), (captured, captured_scene) = reports
    for key in (
        "cut_tets",
        "virtual_corners",
        "wound_particles",
        "spare_tets_taken",
        "triangle_rows_taken",
        "edge_rows_taken",
        "start_gap_limited_edges",
    ):
        test.assertEqual(captured[key], plain[key], key)
    np.testing.assert_array_equal(
        captured_scene.solver._xfem["tet_side"].numpy()[: plain_scene.tissue_tets],
        plain_scene.solver._xfem["tet_side"].numpy()[: plain_scene.tissue_tets],
    )
    test.assertEqual(sorted(captured["copies_per_pool"]), sorted(plain["copies_per_pool"]))
    # Run-to-run differences of the GPU contact sums grow over the whole cut; the end state stays close
    # (measured: 0.004 to 0.013 mm between two runs, captured or not).
    test.assertLess(float(np.abs(captured_scene.corner_positions() - plain_scene.corner_positions()).max()), 1.0e-4)


devices = get_test_devices()
cuda_devices = [device for device in devices if wp.get_device(device).is_cuda]


class TestCut(NewtonTestCase):
    pass


# Builder only: no device.
add_function_test(TestCut, "test_cut_blade_validation", test_cut_blade_validation, devices=None)
for fn in (
    test_cut_capacity_layout,
    test_solver_rules,
    test_capacity_does_not_change_the_uncut_motion,
    test_gate_arm_cuts_through_row_by_row,
    test_no_cut_below_the_cut_force,
    test_reset_undoes_the_cut,
):
    add_function_test(TestCut, fn.__name__, fn, devices=devices)
# The other arms and modes repeat the cut; they run on CUDA only to keep the CPU run short.
for fn in (
    test_wound_arm_cuts_through,
    test_designed_rule_cuts_the_first_row,
    test_stepped_cut_first_advance,
    test_armed_cut_on_first_contact,
    test_graph_capture_matches_uncaptured,
):
    add_function_test(TestCut, fn.__name__, fn, devices=cuda_devices)
# The XFEM-type cut.
for fn in (
    test_xfem_material_share,
    test_xfem_capacity_and_rules,
    test_xfem_first_advance,
    test_xfem_front_rows_fork,
    test_xfem_reset_undoes_the_cut,
):
    add_function_test(TestCut, fn.__name__, fn, devices=devices)
for fn in (
    test_xfem_refused_advance,
    test_xfem_cuts_through_mid_cell,
    test_xfem_cuts_through_turned_plane,
    test_xfem_cuts_through_plane_near_vertices,
    test_xfem_pressed_wound_holds,
    test_xfem_graph_capture_matches_uncaptured,
):
    add_function_test(TestCut, fn.__name__, fn, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
