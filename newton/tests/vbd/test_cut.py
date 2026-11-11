# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cutting in SolverVBD: builder capacity, solver rules, the cut along tet faces, capture and reset."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices

CELL = 3.0e-3
DIMS = (6, 2, 2)
TOP = DIMS[2] * CELL
BLADE_RADIUS = 4.5e-4
FPS = 60
SUBSTEPS = 10
DT = 1.0 / FPS / SUBSTEPS


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


def _add_blade(builder, yaw_deg=0.0):
    """A capsule along y above the middle grid plane, on a kinematic holder with a D6 drive."""
    plane_x = (DIMS[0] // 2) * CELL
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
        "particle_edge_parallel_epsilon": 1.0e-7,
        "collision_pipeline": pipeline,
    }
    options.update(kwargs)
    return SolverVBD(model, **options)


class _Scene:
    """The block, a blade and the cut capacity, finalized on ``device``."""

    def __init__(self, device, *, cut_force=0.5, yaw_deg=0.0, **solver_kwargs):
        builder = _builder_with_block()
        self.tissue_count = builder.particle_count
        self.tissue_tets = builder.tet_count
        self.holder, self.blade, self.edge, self.start, self.rotation = _add_blade(builder, yaw_deg)
        builder.color()
        _add_capacity(builder)
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

    def set_holder(self, travel):
        body_q = self.state_0.body_q.numpy()
        position = self.start - np.array([0.0, 0.0, travel])
        body_q[self.holder] = wp.transform(wp.vec3(*position), self.rotation)
        self.state_0.body_q.assign(body_q)

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

    def run(self, frames, *, speed=4.0e-3, max_travel=None):
        """Lower the holder at ``speed`` for ``frames`` frames."""
        for _ in range(frames):
            travel = speed * self.frame / FPS
            if max_travel is not None:
                travel = min(travel, max_travel)
            self.set_holder(travel)
            self.frame_body()
            self.frame += 1

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
