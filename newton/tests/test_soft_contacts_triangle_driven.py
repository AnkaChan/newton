# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the water-tight triangle-driven rigid-soft contact path.

Covers the Stage 1 schema (Section A), rigid ``Mesh`` ownership + Model
flat-pack (Section B), per-shape contact schemes (Section C), end-to-end
flag behaviour (Section D), ownership dedup (Section E) and negative cases
(Section F) from the test plan.  Per AGENTS.md these use ``unittest``.
"""

import inspect
import unittest
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.sim import contacts as contacts_mod
from newton._src.sim.collide import CollisionPipeline
from newton._src.sim.contacts import (
    SOFT_CONTACT_KIND_EDGE,
    SOFT_CONTACT_KIND_FACE,
    Contacts,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

MARGIN = 0.01
KIND_EDGE = int(SOFT_CONTACT_KIND_EDGE)  # 2
KIND_FACE = int(SOFT_CONTACT_KIND_FACE)  # 3


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


def _popcount(x: int) -> int:
    return bin(int(x) & 0xFF).count("1")


# Unit cube centred at the origin, 8 verts at +/-0.5, triangulated as 12 tris
# (2 per face).  A closed manifold so every edge is shared by exactly 2 tris
# and every vertex by several -- exercises rigid-side ownership dedup.
_CUBE_V = [
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
]
_CUBE_I = [
    0,
    2,
    1,
    0,
    3,
    2,  # -z
    4,
    5,
    6,
    4,
    6,
    7,  # +z
    0,
    4,
    7,
    0,
    7,
    3,  # -x
    1,
    2,
    6,
    1,
    6,
    5,  # +x
    0,
    1,
    5,
    0,
    5,
    4,  # -y
    3,
    7,
    6,
    3,
    6,
    2,  # +y
]


def _cube_mesh() -> newton.Mesh:
    return newton.Mesh(_CUBE_V, _CUBE_I)


def _soft_tri_model(device, verts, indices=(0, 1, 2), *, particle_radius=0.0, shapes=()):
    """Build a model with the given rigid ``shapes`` and one soft mesh.

    ``shapes`` is a sequence of callables ``add(builder) -> shape_id`` invoked
    in order, so rigid shape indices are 0, 1, ...  The soft mesh is placed at
    the world coordinates given by ``verts`` (pos=0, rot=identity, scale=1) and
    its triangle adjacency is built by ``finalize()``.
    """
    builder = newton.ModelBuilder()
    for add_shape in shapes:
        add_shape(builder)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts],
        indices=list(indices),
        density=1.0,
        particle_radius=particle_radius,
    )
    model = builder.finalize(device=device)
    return model, model.state()


def _collide(model, state, *, water_tight, margin=MARGIN, soft_max=512):
    pipeline = CollisionPipeline(model, soft_contact_max=soft_max, soft_contact_margin=margin)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, enable_water_tight_rigid_soft_contact=water_tight)
    return contacts


def _counts(contacts):
    cnt = contacts.soft_contact_count.numpy()
    return int(cnt[0]), int(cnt[1])


def _legacy_records(contacts):
    """Particle-range records [0, soft_contact_count[0]) as a list of dicts."""
    n0, _ = _counts(contacts)
    particle = contacts.soft_contact_particle.numpy()
    shape = contacts.soft_contact_shape.numpy()
    body_pos = contacts.soft_contact_body_pos.numpy()
    normal = contacts.soft_contact_normal.numpy()
    out = []
    for i in range(n0):
        out.append(
            {
                "particle": int(particle[i]),
                "shape": int(shape[i]),
                "body_pos": tuple(float(x) for x in body_pos[i]),
                "normal": tuple(float(x) for x in normal[i]),
            }
        )
    return out


def _tri_records(contacts):
    """E/F-range records.  New-only fields read at local [0, n1); shared fields
    read at [n0, n0 + n1).  Sorted for assertion stability."""
    n0, n1 = _counts(contacts)
    smax = contacts.soft_contact_max
    primitive = contacts.soft_contact_primitive.numpy()
    kind = contacts.soft_contact_kind.numpy()
    bary = contacts.soft_contact_barycentric.numpy()
    shape = contacts.soft_contact_shape.numpy()
    body_pos = contacts.soft_contact_body_pos.numpy()
    normal = contacts.soft_contact_normal.numpy()
    out = []
    for j in range(n1):
        idx = n0 + j
        if idx >= smax:
            break
        out.append(
            {
                "soft_tri": int(primitive[j]),
                "kind": int(kind[j]),
                "bary": tuple(float(x) for x in bary[j]),
                "shape": int(shape[idx]),
                "body_pos": tuple(float(x) for x in body_pos[idx]),
                "normal": tuple(float(x) for x in normal[idx]),
            }
        )
    out.sort(
        key=lambda r: (r["kind"], round(r["body_pos"][0], 4), round(r["body_pos"][1], 4), round(r["body_pos"][2], 4))
    )
    return out


def _edge_records(records):
    return [r for r in records if r["kind"] == KIND_EDGE]


def _face_records(records):
    return [r for r in records if r["kind"] == KIND_FACE]


def _assert_vec(test, got, want, tol=1e-3, msg=""):
    for g, w in zip(got, want, strict=True):
        test.assertAlmostEqual(g, w, delta=tol, msg=f"{msg}: got {got}, want {want}")


# ---------------------------------------------------------------------------
# Section A. Schema
# ---------------------------------------------------------------------------


class TestSoftContactSchema(unittest.TestCase):
    def _contacts(self, **kwargs):
        return Contacts(16, 16, device="cpu", **kwargs)

    def test_contacts_legacy_fields_unchanged(self):
        c = self._contacts()
        for name in (
            "soft_contact_particle",
            "soft_contact_shape",
            "soft_contact_body_pos",
            "soft_contact_body_vel",
            "soft_contact_normal",
            "soft_contact_tids",
        ):
            arr = getattr(c, name)
            self.assertIsNotNone(arr, name)
            self.assertEqual(arr.shape[0], 16, name)

    def test_contacts_new_fields_present(self):
        c = self._contacts()
        self.assertEqual(c.soft_contact_primitive.dtype, wp.int32)
        self.assertEqual(c.soft_contact_kind.dtype, wp.uint8)
        self.assertEqual(c.soft_contact_barycentric.dtype, wp.vec3)
        for name in ("soft_contact_primitive", "soft_contact_kind", "soft_contact_barycentric"):
            self.assertEqual(getattr(c, name).shape[0], 16, name)

    def test_soft_contact_count_is_length_two(self):
        c = self._contacts()
        self.assertEqual(c.soft_contact_count.shape, (2,))
        self.assertEqual(c.soft_contact_count.dtype, wp.int32)

    def test_contact_counters_extended_to_three(self):
        c = self._contacts()
        self.assertEqual(c.contact_counters.shape, (3,))
        # rigid count slices [0:1], soft count slices [1:3] of the same buffer.
        self.assertEqual(c.rigid_contact_count.ptr, c.contact_counters.ptr)
        self.assertEqual(c.soft_contact_count.shape[0], 2)

    def test_no_soft_contact_tri_max_keyword(self):
        params = inspect.signature(Contacts.__init__).parameters
        self.assertNotIn("soft_contact_tri_max", params)
        # Still constructs with the documented keywords.
        Contacts(8, 8, device="cpu")

    def test_clear_resets_both_counter_slots(self):
        c = self._contacts()
        c.contact_counters.assign(np.array([3, 5, 7], dtype=np.int32))
        c.clear()
        self.assertTrue((c.soft_contact_count.numpy() == [0, 0]).all())

    def test_clear_buffers_resets_new_fields(self):
        c = self._contacts(clear_buffers=True)
        c.soft_contact_primitive.fill_(5)
        c.soft_contact_kind.fill_(KIND_FACE)
        c.soft_contact_barycentric.fill_(wp.vec3(1.0, 2.0, 3.0))
        c.clear()
        self.assertTrue((c.soft_contact_primitive.numpy() == -1).all())
        self.assertTrue((c.soft_contact_kind.numpy() == 0).all())
        self.assertTrue((c.soft_contact_barycentric.numpy() == 0.0).all())

    def test_soft_contact_kind_constants(self):
        self.assertEqual(KIND_EDGE, 2)
        self.assertEqual(KIND_FACE, 3)
        self.assertFalse(hasattr(contacts_mod, "SOFT_CONTACT_KIND_VERTEX"))


# ---------------------------------------------------------------------------
# Section B. Rigid Mesh ownership numpy properties
# ---------------------------------------------------------------------------


class TestRigidMeshOwnership(unittest.TestCase):
    TET_V: ClassVar = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    TET_I: ClassVar = [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2]
    QUAD_V: ClassVar = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    QUAD_I: ClassVar = [0, 1, 2, 2, 1, 3]

    def test_mesh_tri_edges_shape(self):
        m = newton.Mesh(self.TET_V, self.TET_I)
        self.assertEqual(m.tri_edges.shape, (4, 3))
        n_edges = len(m.edges)
        self.assertTrue(((0 <= m.tri_edges) & (m.tri_edges < n_edges)).all())

    def test_mesh_vertex_owner_each_vertex_has_owner(self):
        m = newton.Mesh(self.TET_V, self.TET_I)
        owner = m.vertex_owner_tri
        self.assertEqual(owner.shape, (4,))
        self.assertTrue((owner >= 0).all())

    def test_mesh_tri_feature_owner_flag_bits_set(self):
        m = newton.Mesh(self.TET_V, self.TET_I)
        flags = m.tri_feature_owner_flag
        # 4 vertices + 6 edges, each owned exactly once.
        total = sum(_popcount(f) for f in flags)
        self.assertEqual(total, 4 + 6)

    def test_mesh_open_vertex_ownership_count(self):
        m = newton.Mesh(self.QUAD_V, self.QUAD_I, compute_inertia=False)
        flags = m.tri_feature_owner_flag
        vbits = sum(_popcount(int(f) & 0b111) for f in flags)
        self.assertEqual(vbits, 4)

    def test_mesh_open_edge_ownership_count(self):
        m = newton.Mesh(self.QUAD_V, self.QUAD_I, compute_inertia=False)
        flags = m.tri_feature_owner_flag
        ebits = sum(_popcount((int(f) >> 3) & 0b111) for f in flags)
        self.assertEqual(ebits, 5)

    def test_mesh_empty_returns_empty_arrays(self):
        m = newton.Mesh([], [])
        self.assertEqual(m.tri_edges.shape, (0, 3))
        self.assertEqual(m.vertex_owner_tri.shape, (0,))
        self.assertEqual(m.tri_feature_owner_flag.shape, (0,))

    def test_mesh_setter_invalidates_cache(self):
        m = newton.Mesh(self.TET_V, self.TET_I)
        flags_before = m.tri_feature_owner_flag
        m.indices = np.array(self.QUAD_I, dtype=np.int32)
        flags_after = m.tri_feature_owner_flag
        # New topology -> recomputed (different object, different shape).
        self.assertIsNot(flags_before, flags_after)
        self.assertEqual(flags_after.shape, (2,))


# ---------------------------------------------------------------------------
# Section B.4 Model-level flat-pack
# ---------------------------------------------------------------------------


class TestModelMeshFlatPack(unittest.TestCase):
    def _finalize(self, shapes, device="cpu"):
        builder = newton.ModelBuilder()
        for add_shape in shapes:
            add_shape(builder)
        # A soft mesh so the ownership arrays are actually built/used.
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 5.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(0.0, 0.0, 5.0), wp.vec3(1.0, 0.0, 5.0), wp.vec3(0.0, 1.0, 5.0)],
            indices=[0, 1, 2],
            density=1.0,
            particle_radius=0.0,
        )
        return builder.finalize(device=device)

    def test_model_shape_mesh_ownership_fields_built(self):
        model = self._finalize([lambda b: b.add_shape_mesh(body=-1, mesh=_cube_mesh(), scale=wp.vec3(1.0, 1.0, 1.0))])
        for name in (
            "shape_mesh_tri_edges",
            "shape_mesh_vertex_owner_tri",
            "shape_mesh_tri_feature_owner_flag",
            "shape_mesh_ownership_range",
        ):
            self.assertIsNotNone(getattr(model, name), name)

    def test_model_shape_mesh_ownership_range_for_mesh(self):
        mesh = newton.Mesh(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
        )
        model = self._finalize([lambda b: b.add_shape_mesh(body=-1, mesh=mesh, scale=wp.vec3(1.0, 1.0, 1.0))])
        rng = model.shape_mesh_ownership_range.numpy()
        self.assertEqual(list(rng[0]), [0, 0, 0])

    def test_model_shape_mesh_ownership_range_for_non_mesh(self):
        model = self._finalize(
            [
                lambda b: b.add_shape_mesh(body=-1, mesh=_cube_mesh(), scale=wp.vec3(1.0, 1.0, 1.0)),
                lambda b: b.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5),
            ]
        )
        rng = model.shape_mesh_ownership_range.numpy()
        self.assertEqual(list(rng[1]), [-1, -1, -1])

    def test_model_shape_mesh_flat_pack_offsets(self):
        tet = newton.Mesh(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
        )
        quad = newton.Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [0, 1, 2, 2, 1, 3], compute_inertia=False)
        model = self._finalize(
            [
                lambda b: b.add_shape_mesh(body=-1, mesh=tet, scale=wp.vec3(1.0, 1.0, 1.0)),
                lambda b: b.add_shape_mesh(body=-1, mesh=quad, scale=wp.vec3(1.0, 1.0, 1.0)),
            ]
        )
        rng = model.shape_mesh_ownership_range.numpy()
        self.assertEqual(list(rng[0]), [0, 0, 0])
        # tet contributes 4 tris / 4 verts / 6 edges.
        self.assertEqual(list(rng[1]), [4, 4, 6])

    def test_model_shape_mesh_skips_non_mesh_geo(self):
        model = self._finalize(
            [
                lambda b: b.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5),
                lambda b: b.add_shape_sphere(body=-1, radius=0.3),
                lambda b: b.add_shape_capsule(body=-1, radius=0.2, half_height=0.5),
            ]
        )
        rng = model.shape_mesh_ownership_range.numpy()
        for i in range(3):
            self.assertEqual(list(rng[i]), [-1, -1, -1])


def _test_flat_pack_stable_pointers(test, device):
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=_cube_mesh(), scale=wp.vec3(1.0, 1.0, 1.0))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(0.6, 0.0, 0.0), wp.vec3(0.0, 0.6, 0.0), wp.vec3(0.0, 0.0, 0.6)],
        indices=[0, 1, 2],
        density=1.0,
        particle_radius=0.0,
    )
    model = builder.finalize(device=device)
    state = model.state()
    ptr0 = model.shape_mesh_tri_edges.ptr
    pipeline = CollisionPipeline(model, soft_contact_max=128, soft_contact_margin=MARGIN)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, enable_water_tight_rigid_soft_contact=True)
    pipeline.collide(state, contacts, enable_water_tight_rigid_soft_contact=True)
    test.assertEqual(model.shape_mesh_tri_edges.ptr, ptr0)


add_function_test(
    TestModelMeshFlatPack,
    "test_model_shape_mesh_stable_pointers",
    _test_flat_pack_stable_pointers,
    devices=get_cuda_test_devices(),
)


# ---------------------------------------------------------------------------
# Section C / D / E / F. Contact generation (GPU)
# ---------------------------------------------------------------------------


class TestTriangleDrivenContacts(unittest.TestCase):
    pass


# --- Box --------------------------------------------------------------------


def _box(b):
    return b.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)


def _test_box_legacy_face(test, device):
    # C2.1: single soft V just outside +X face; other two far away.
    model, state = _soft_tri_model(
        device,
        [(0.505, 0.0, 0.0), (3.0, 2.0, 0.0), (3.0, -2.0, 0.0)],
        shapes=(_box,),
    )
    contacts = _collide(model, state, water_tight=True)
    legacy = _legacy_records(contacts)
    test.assertEqual(len(legacy), 1)
    _assert_vec(test, legacy[0]["body_pos"], (0.5, 0.0, 0.0), tol=2e-3, msg="box face body_pos")
    test.assertGreater(abs(legacy[0]["normal"][0]), 0.9)
    # Mid-face point is far from every corner/edge -> no E/F.
    test.assertEqual(len(_tri_records(contacts)), 0)


def _test_box_edge_gap(test, device):
    # C2.4 [water-tight gap]: flat tri straddles the +X/+Y vertical box edge,
    # all soft V > margin from the box.  Legacy SDF sees nothing.
    model, state = _soft_tri_model(
        device,
        [(0.6, 0.405, 0.0), (0.405, 0.6, 0.0), (1.5, 1.5, 0.0)],
        shapes=(_box,),
    )
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    edges = _edge_records(_tri_records(contacts))
    test.assertEqual(len(edges), 1)
    _assert_vec(test, edges[0]["body_pos"], (0.5, 0.5, 0.0), tol=2e-3, msg="box edge body_pos")
    n = edges[0]["normal"]
    _assert_vec(test, (n[0], n[1]), (0.7071, 0.7071), tol=5e-2, msg="box edge normal")


def _test_box_corner_face(test, device):
    # C2.5: small tri swept over the (0.5,0.5,0.5) corner, perpendicular to the
    # body diagonal.  Corner is the closest rigid feature.
    p = np.array([0.5, 0.5, 0.5])
    d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    g = p + 0.006 * d
    u1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    u2 = np.cross(d, u1)
    r = 0.03
    verts = [tuple(g + r * (np.cos(t) * u1 + np.sin(t) * u2)) for t in (0.0, 2.094395, 4.18879)]
    model, state = _soft_tri_model(device, verts, shapes=(_box,))
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    faces = _face_records(_tri_records(contacts))
    test.assertEqual(len(faces), 1)
    _assert_vec(test, faces[0]["body_pos"], (0.5, 0.5, 0.5), tol=2e-3, msg="box corner body_pos")
    _assert_vec(test, faces[0]["bary"], (1.0 / 3, 1.0 / 3, 1.0 / 3), tol=5e-2, msg="box corner bary")


def _test_box_penetrating(test, device):
    # C2.6: tri entirely inside the box -> three legacy particle records.
    model, state = _soft_tri_model(
        device,
        [(0.1, 0.0, 0.0), (-0.1, 0.1, 0.0), (-0.1, -0.1, 0.0)],
        shapes=(_box,),
    )
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 3)


# --- Sphere -----------------------------------------------------------------


def _sphere(b):
    return b.add_shape_sphere(body=-1, radius=0.3)


def _test_sphere_legacy(test, device):
    # C4.1: soft V just outside the sphere surface.
    model, state = _soft_tri_model(
        device,
        [(0.305, 0.0, 0.0), (5.0, 5.0, 0.0), (5.0, 6.0, 0.0)],
        shapes=(_sphere,),
    )
    contacts = _collide(model, state, water_tight=True)
    legacy = _legacy_records(contacts)
    test.assertEqual(len(legacy), 1)
    _assert_vec(test, legacy[0]["body_pos"], (0.3, 0.0, 0.0), tol=3e-3, msg="sphere body_pos")


def _test_sphere_gap_face(test, device):
    # C4.2 [water-tight gap]: tri straddles the sphere in the z=0.1 plane, all
    # soft V well outside radius+margin, the centre projects inside the tri.
    model, state = _soft_tri_model(
        device,
        [(0.6, 0.0, 0.1), (-0.3, 0.5196, 0.1), (-0.3, -0.5196, 0.1)],
        shapes=(_sphere,),
    )
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    faces = _face_records(_tri_records(contacts))
    test.assertEqual(len(faces), 1)
    _assert_vec(test, faces[0]["body_pos"], (0.0, 0.0, 0.3), tol=3e-3, msg="sphere gap body_pos")
    _assert_vec(test, faces[0]["bary"], (1.0 / 3, 1.0 / 3, 1.0 / 3), tol=5e-2, msg="sphere gap bary")


# --- Capsule ----------------------------------------------------------------


def _capsule(b):
    return b.add_shape_capsule(body=-1, radius=0.2, half_height=0.5)


def _test_capsule_legacy(test, device):
    # C3.1: soft V just outside the cylindrical region.
    model, state = _soft_tri_model(
        device,
        [(0.205, 0.0, 0.0), (5.0, 5.0, 0.0), (5.0, 6.0, 0.0)],
        shapes=(_capsule,),
    )
    contacts = _collide(model, state, water_tight=True)
    legacy = _legacy_records(contacts)
    test.assertEqual(len(legacy), 1)
    _assert_vec(test, legacy[0]["body_pos"], (0.2, 0.0, 0.0), tol=3e-3, msg="capsule body_pos")


def _test_capsule_edge_gap(test, device):
    # C3.3 [water-tight gap]: soft owned edge crosses near the capsule axis,
    # all soft V at radial distance > radius+margin.
    model, state = _soft_tri_model(
        device,
        [(-0.5, 0.05, 0.0), (0.5, 0.05, 0.0), (0.0, 1.0, 0.0)],
        shapes=(_capsule,),
    )
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    edges = _edge_records(_tri_records(contacts))
    test.assertEqual(len(edges), 1)
    _assert_vec(test, edges[0]["body_pos"], (0.0, 0.2, 0.0), tol=2e-3, msg="capsule edge body_pos")
    _assert_vec(test, edges[0]["normal"], (0.0, 1.0, 0.0), tol=2e-2, msg="capsule edge normal")


# --- Cylinder ---------------------------------------------------------------


def _cylinder(b):
    return b.add_shape_cylinder(body=-1, radius=0.3, half_height=0.5)


def _test_cylinder_cap_face_no_edge(test, device):
    # C6.2 [water-tight gap]: large horizontal tri just above the top cap
    # (z=0.5), centroid over the axis.  The SDF face minimizer is the cap point
    # (0,0,0.5).  Vertices (radial 0.7) and edges (inradius 0.35 > cap radius
    # 0.3) stay clear of the cap -> no legacy, no EDGE records.
    model, state = _soft_tri_model(
        device,
        [(0.7, 0.0, 0.505), (-0.35, 0.606, 0.505), (-0.35, -0.606, 0.505)],
        shapes=(_cylinder,),
    )
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    records = _tri_records(contacts)
    test.assertEqual(len(_edge_records(records)), 0)
    faces = _face_records(records)
    test.assertEqual(len(faces), 1)
    _assert_vec(test, faces[0]["body_pos"], (0.0, 0.0, 0.5), tol=3e-3, msg="cylinder cap body_pos")
    _assert_vec(test, faces[0]["normal"], (0.0, 0.0, 1.0), tol=2e-2, msg="cylinder cap normal")


# --- Cone -------------------------------------------------------------------


def _cone(b):
    return b.add_shape_cone(body=-1, radius=0.3, half_height=0.5)


def _test_cone_apex_face(test, device):
    # C7.2: small tri above the apex; apex (0,0,0.5) is the closest feature.
    g = np.array([0.0, 0.0, 0.505])
    r = 0.05
    verts = [tuple(g + r * np.array([np.cos(t), np.sin(t), 0.0])) for t in (0.0, 2.094395, 4.18879)]
    model, state = _soft_tri_model(device, verts, shapes=(_cone,))
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    faces = _face_records(_tri_records(contacts))
    test.assertEqual(len(faces), 1)
    _assert_vec(test, faces[0]["body_pos"], (0.0, 0.0, 0.5), tol=2e-3, msg="cone apex body_pos")
    _assert_vec(test, faces[0]["bary"], (1.0 / 3, 1.0 / 3, 1.0 / 3), tol=5e-2, msg="cone apex bary")


def _test_cone_base_face(test, device):
    # C7.4 [water-tight gap]: large horizontal tri just below the base cap
    # (z=-0.5), centroid over the axis.  A single smooth phi field yields ONE
    # face minimizer per triangle -> the base point (0,0,-0.5).  Vertices
    # (radial 0.7) and edges (inradius 0.35 > base radius 0.3) stay clear of the
    # base -> no legacy, no EDGE records.
    model, state = _soft_tri_model(
        device,
        [(0.7, 0.0, -0.505), (-0.35, 0.606, -0.505), (-0.35, -0.606, -0.505)],
        shapes=(_cone,),
    )
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    records = _tri_records(contacts)
    test.assertEqual(len(_edge_records(records)), 0)
    faces = _face_records(records)
    test.assertEqual(len(faces), 1)
    _assert_vec(test, faces[0]["body_pos"], (0.0, 0.0, -0.5), tol=3e-3, msg="cone base body_pos")
    _assert_vec(test, faces[0]["normal"], (0.0, 0.0, -1.0), tol=2e-2, msg="cone base normal")


# --- Mesh -------------------------------------------------------------------


def _mesh_cube(b):
    return b.add_shape_mesh(body=-1, mesh=_cube_mesh(), scale=wp.vec3(1.0, 1.0, 1.0))


def _test_mesh_corner_face(test, device):
    # C1.2 / E.1: soft tri sweeps over the cube corner (a shared rigid vertex).
    # Owner gating -> exactly one FACE despite many tris sharing the corner.
    p = np.array([0.5, 0.5, 0.5])
    d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    g = p + 0.006 * d
    u1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    u2 = np.cross(d, u1)
    r = 0.03
    verts = [tuple(g + r * (np.cos(t) * u1 + np.sin(t) * u2)) for t in (0.0, 2.094395, 4.18879)]
    model, state = _soft_tri_model(device, verts, shapes=(_mesh_cube,))
    contacts = _collide(model, state, water_tight=True)
    faces = _face_records(_tri_records(contacts))
    test.assertEqual(len(faces), 1)
    _assert_vec(test, faces[0]["body_pos"], (0.5, 0.5, 0.5), tol=2e-3, msg="mesh corner body_pos")


def _test_mesh_edge_gap(test, device):
    # C1.4 [water-tight gap]: soft tri straddles a cube edge; owner gating ->
    # exactly one EDGE despite the edge being shared by two rigid tris.
    model, state = _soft_tri_model(
        device,
        [(0.6, 0.405, 0.0), (0.405, 0.6, 0.0), (1.5, 1.5, 0.0)],
        shapes=(_mesh_cube,),
    )
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    edges = _edge_records(_tri_records(contacts))
    test.assertEqual(len(edges), 1)
    _assert_vec(test, edges[0]["body_pos"], (0.5, 0.5, 0.0), tol=2e-3, msg="mesh edge body_pos")


# --- Section E. Dedup -------------------------------------------------------


def _test_dedup_soft_shared_edge(test, device):
    # E.3: two soft tris share an edge; only the owner tri emits the E x E.
    # Shared edge (v1,v2) straddles the cube/box vertical edge at (0.5,0.5,0).
    verts = [
        (1.5, 1.5, 0.3),  # v0 (tri 0 apex, far)
        (0.6, 0.405, 0.0),  # v1 (shared edge endpoint)
        (0.405, 0.6, 0.0),  # v2 (shared edge endpoint)
        (1.5, 1.5, -0.3),  # v3 (tri 1 apex, far)
    ]
    indices = [0, 1, 2, 2, 1, 3]
    model, state = _soft_tri_model(device, verts, indices=indices, shapes=(_box,))
    contacts = _collide(model, state, water_tight=True)
    edges = _edge_records(_tri_records(contacts))
    test.assertEqual(len(edges), 1)
    _assert_vec(test, edges[0]["body_pos"], (0.5, 0.5, 0.0), tol=2e-3, msg="shared-edge dedup body_pos")


# --- Section D. End-to-end --------------------------------------------------


def _grid_box_model(device, soft_max=512):
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.6, -0.6, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=5,
        dim_y=5,
        cell_x=0.3,
        cell_y=0.3,
        mass=1.0,
    )
    model = builder.finalize(device=device)
    pipeline = CollisionPipeline(model, soft_contact_max=soft_max, soft_contact_margin=MARGIN)
    return model, model.state(), pipeline


def _test_flag_off_slot1_zero(test, device):
    _model, state, pipeline = _grid_box_model(device)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, enable_water_tight_rigid_soft_contact=False)
    n0, n1 = _counts(contacts)
    test.assertGreater(n0, 0)
    test.assertEqual(n1, 0)


def _test_flag_off_matches_flag_on_particle_range(test, device):
    _model, state, pipeline = _grid_box_model(device)
    off = pipeline.contacts()
    pipeline.collide(state, off, enable_water_tight_rigid_soft_contact=False)
    off_map = {r["particle"]: r["body_pos"] for r in _legacy_records(off)}

    on = pipeline.contacts()
    pipeline.collide(state, on, enable_water_tight_rigid_soft_contact=True)
    on_map = {r["particle"]: r["body_pos"] for r in _legacy_records(on)}

    test.assertEqual(set(off_map), set(on_map))
    for p, off_pos in off_map.items():
        _assert_vec(test, on_map[p], off_pos, tol=1e-6, msg=f"particle {p} body_pos drift")


def _test_flag_on_smoke_box(test, device):
    _model, state, pipeline = _grid_box_model(device)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, enable_water_tight_rigid_soft_contact=True)
    n0, n1 = _counts(contacts)
    test.assertGreater(n0, 0)
    test.assertGreater(n1, 0)
    records = _tri_records(contacts)
    test.assertEqual(len(records), n1)
    test.assertGreaterEqual(len(_edge_records(records)), 1)
    # Every E/F record carries a valid kind. This grid+box scene drives only
    # EDGE contacts -- the box corners project outside the grid-tri interiors,
    # so no FACE contacts are expected.
    test.assertTrue(all(r["kind"] in (KIND_EDGE, KIND_FACE) for r in records))


def _test_overflow_does_not_corrupt(test, device):
    _model, state, pipeline = _grid_box_model(device, soft_max=4)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, enable_water_tight_rigid_soft_contact=True)
    n0, n1 = _counts(contacts)
    test.assertGreaterEqual(n0, 0)
    test.assertGreaterEqual(n1, 0)
    # The grid+box scene attempts more than 4 emissions: overflow is visible in
    # the counters without any crash or negative index.
    test.assertGreater(n0 + n1, 4)


# --- Section F. Negative ----------------------------------------------------


def _test_soft_v_out_of_margin(test, device):
    # F.1: soft V well beyond margin -> no particle record.
    model, state = _soft_tri_model(
        device,
        [(0.6, 0.0, 0.0), (3.0, 2.0, 0.0), (3.0, -2.0, 0.0)],
        shapes=(_box,),
    )
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)


def _test_disabled_shape_skipped(test, device):
    # F.2: shape with COLLIDE_PARTICLES cleared -> no records (either range).
    def _disabled_box(b):
        cfg = b.default_shape_cfg.copy()
        cfg.has_particle_collision = False
        return b.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)

    # Same geometry as the box edge-gap case, which would emit an EDGE if active.
    model, state = _soft_tri_model(
        device,
        [(0.6, 0.405, 0.0), (0.405, 0.6, 0.0), (1.5, 1.5, 0.0)],
        shapes=(_disabled_box,),
    )
    contacts = _collide(model, state, water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    test.assertEqual(len(_tri_records(contacts)), 0)


def _test_different_world_ids_skipped(test, device):
    # F.3: box in world 0, soft tri in world 1, geometrically overlapping.
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    builder.end_world()
    builder.begin_world()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(0.6, 0.405, 0.0), wp.vec3(0.405, 0.6, 0.0), wp.vec3(1.5, 1.5, 0.0)],
        indices=[0, 1, 2],
        density=1.0,
        particle_radius=0.0,
    )
    builder.end_world()
    model = builder.finalize(device=device)
    contacts = _collide(model, model.state(), water_tight=True)
    test.assertEqual(len(_legacy_records(contacts)), 0)
    test.assertEqual(len(_tri_records(contacts)), 0)


def _test_no_soft_mesh_adjacency_falls_back(test, device):
    # F.4: particles registered without triangle topology -> flag-on == flag-off.
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    builder.add_particle(pos=wp.vec3(0.505, 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
    builder.add_particle(pos=wp.vec3(0.0, 0.505, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
    model = builder.finalize(device=device)
    # Loose particles carry no triangle topology, so the triangle-driven kernel
    # no-ops (tri_count == 0) and the water-tight path matches the legacy path.
    test.assertEqual(model.tri_count, 0)
    state = model.state()

    pipeline = CollisionPipeline(model, soft_contact_max=128, soft_contact_margin=MARGIN)
    off = pipeline.contacts()
    pipeline.collide(state, off, enable_water_tight_rigid_soft_contact=False)
    on = pipeline.contacts()
    pipeline.collide(state, on, enable_water_tight_rigid_soft_contact=True)
    test.assertEqual(_counts(off), _counts(on))
    test.assertEqual(_counts(on)[1], 0)


_CONTACT_TESTS = [
    ("test_box_legacy_face", _test_box_legacy_face),
    ("test_box_edge_gap", _test_box_edge_gap),
    ("test_box_corner_face", _test_box_corner_face),
    ("test_box_penetrating", _test_box_penetrating),
    ("test_sphere_legacy", _test_sphere_legacy),
    ("test_sphere_gap_face", _test_sphere_gap_face),
    ("test_capsule_legacy", _test_capsule_legacy),
    ("test_capsule_edge_gap", _test_capsule_edge_gap),
    ("test_cylinder_cap_face_no_edge", _test_cylinder_cap_face_no_edge),
    ("test_cone_apex_face", _test_cone_apex_face),
    ("test_cone_base_face", _test_cone_base_face),
    ("test_mesh_corner_face", _test_mesh_corner_face),
    ("test_mesh_edge_gap", _test_mesh_edge_gap),
    ("test_dedup_soft_shared_edge", _test_dedup_soft_shared_edge),
    ("test_flag_off_slot1_zero", _test_flag_off_slot1_zero),
    ("test_flag_off_matches_flag_on_particle_range", _test_flag_off_matches_flag_on_particle_range),
    ("test_flag_on_smoke_box", _test_flag_on_smoke_box),
    ("test_overflow_does_not_corrupt", _test_overflow_does_not_corrupt),
    ("test_soft_v_out_of_margin", _test_soft_v_out_of_margin),
    ("test_disabled_shape_skipped", _test_disabled_shape_skipped),
    ("test_different_world_ids_skipped", _test_different_world_ids_skipped),
    ("test_no_soft_mesh_adjacency_falls_back", _test_no_soft_mesh_adjacency_falls_back),
]

for _name, _func in _CONTACT_TESTS:
    add_function_test(TestTriangleDrivenContacts, _name, _func, devices=get_cuda_test_devices())


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
