# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Bag Lift
#
# A box-shaped cloth bag (open top) is suspended by its top-edge vertices.
# Rigid bodies of all supported shape types (sphere, box, capsule,
# cylinder, cone, mesh bear) are placed inside.  After settling, the
# pinned vertices are raised upward; the test validates that the contents
# are lifted with the bag.
#
# Command: python -m newton.tests.test_bag_lift
#
###########################################################################

import os
import time
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags
from newton.tests.unittest_utils import add_function_test, get_test_devices

SHAPE_NAMES = ["mesh", "cone", "sphere", "box", "capsule", "cylinder"]
SHAPE_SIZE = 0.012
SHAPE_MARGIN = 0.003
MESH_MARGIN = 0.008
BAG_WIDTH = 0.12
BAG_DEPTH = 0.07
BAG_HEIGHT = 0.24
BAG_RES = 10
BAG_ELEVATION = 0.30
LIFT_SPEED = 0.10
SETTLE_FRAMES = 120
LIFT_FRAMES = 180
TOTAL_FRAMES = SETTLE_FRAMES + LIFT_FRAMES


def _generate_box_bag(half_x, half_y, height, res, z_base):
    """Generate a box-shaped bag (5 faces, open top) as a single merged mesh."""
    cell_x = 2.0 * half_x / res
    cell_y = 2.0 * half_y / res
    cell_z = height / res

    vertex_map = {}
    vertices = []
    faces = []

    def get_or_add_vertex(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append([x, y, z])
        return vertex_map[key]

    def add_quad(v00, v10, v01, v11):
        faces.extend([v00, v10, v01])
        faces.extend([v10, v11, v01])

    # Bottom face
    for i in range(res):
        for j in range(res):
            x0, x1 = -half_x + i * cell_x, -half_x + (i + 1) * cell_x
            y0, y1 = -half_y + j * cell_y, -half_y + (j + 1) * cell_y
            add_quad(
                get_or_add_vertex(x0, y0, z_base),
                get_or_add_vertex(x1, y0, z_base),
                get_or_add_vertex(x0, y1, z_base),
                get_or_add_vertex(x1, y1, z_base),
            )

    # Side walls
    sides = [
        lambda i, j: (-half_x + i * cell_x, -half_y, z_base + j * cell_z, cell_x, 0, cell_z, 0),
        lambda i, j: (-half_x + i * cell_x, half_y, z_base + j * cell_z, cell_x, 0, cell_z, 1),
        lambda i, j: (-half_x, -half_y + i * cell_y, z_base + j * cell_z, 0, cell_y, cell_z, 2),
        lambda i, j: (half_x, -half_y + i * cell_y, z_base + j * cell_z, 0, cell_y, cell_z, 3),
    ]
    for side_fn in sides:
        for i in range(res):
            for j in range(res):
                x0, y0, z0, dx, dy, dz, side = side_fn(i, j)
                if side == 0:
                    add_quad(
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0 + dx, y0, z0),
                        get_or_add_vertex(x0, y0, z0 + dz),
                        get_or_add_vertex(x0 + dx, y0, z0 + dz),
                    )
                elif side == 1:
                    add_quad(
                        get_or_add_vertex(x0 + dx, y0, z0),
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0 + dx, y0, z0 + dz),
                        get_or_add_vertex(x0, y0, z0 + dz),
                    )
                elif side == 2:
                    add_quad(
                        get_or_add_vertex(x0, y0 + dy, z0),
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0, y0 + dy, z0 + dz),
                        get_or_add_vertex(x0, y0, z0 + dz),
                    )
                elif side == 3:
                    add_quad(
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0, y0 + dy, z0),
                        get_or_add_vertex(x0, y0, z0 + dz),
                        get_or_add_vertex(x0, y0 + dy, z0 + dz),
                    )

    return np.array(vertices, dtype=np.float32), faces


def _load_bear_mesh(target_size=SHAPE_SIZE):
    from pxr import Usd, UsdGeom

    bear_path = os.path.join(newton.examples.get_asset_directory(), "bear.usd")
    stage = Usd.Stage.Open(bear_path)
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))

    points = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    center = (points.max(axis=0) + points.min(axis=0)) / 2.0
    points -= center
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    points *= (target_size * 2.0) / extent

    return points, indices.tolist()


@wp.kernel
def lift_pinned_vertices(
    pinned_indices: wp.array[wp.int32],
    original_positions: wp.array[wp.vec3],
    dz: float,
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    tid = wp.tid()
    vi = pinned_indices[tid]
    p = original_positions[tid]
    new_p = wp.vec3(p[0], p[1], p[2] + dz)
    pos_0[vi] = new_p
    pos_1[vi] = new_p


def build_model(builder, seed=42):
    rng = np.random.default_rng(seed)

    bag_verts, bag_faces = _generate_box_bag(BAG_WIDTH / 2, BAG_DEPTH / 2, BAG_HEIGHT, BAG_RES, BAG_ELEVATION)

    particle_radius = 0.003
    bag_start_particle = len(builder.particle_q)

    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=bag_verts.tolist(),
        indices=bag_faces,
        density=0.08,
        tri_ke=2e6,
        tri_ka=2e6,
        tri_kd=1e2,
        edge_ke=50.0,
        edge_kd=5e-1,
        particle_radius=particle_radius,
    )

    bag_end_particle = len(builder.particle_q)

    # Top-edge vertices
    z_top = BAG_ELEVATION + BAG_HEIGHT
    top_mask = np.abs(bag_verts[:, 2] - z_top) < 0.001
    top_global_indices = np.where(top_mask)[0] + bag_start_particle

    # Rigid bodies
    r = SHAPE_SIZE
    interior_x = BAG_WIDTH / 2 - r * 1.5
    interior_y = BAG_DEPTH / 2 - r * 1.5
    body_indices = []
    shape_indices = []
    positions = []

    bear_mesh = None

    for i in range(len(SHAPE_NAMES)):
        if SHAPE_NAMES[i] == "mesh":
            positions.append((0.0, 0.0))
        else:
            for _ in range(200):
                x = rng.uniform(-interior_x, interior_x)
                y = rng.uniform(-interior_y, interior_y)
                ok = all(np.sqrt((x - px) ** 2 + (y - py) ** 2) >= r * 2 for px, py in positions)
                if ok:
                    positions.append((x, y))
                    break
            else:
                positions.append((x, y))

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 100.0
    cfg.ke = 1e3
    cfg.kd = 1e-1
    cfg.mu = 0.5
    cfg.has_particle_collision = True
    cfg.margin = SHAPE_MARGIN

    for i, name in enumerate(SHAPE_NAMES):
        px, py = positions[i]
        drop_z = BAG_ELEVATION + 0.04 + i * 0.03

        body = builder.add_body(xform=wp.transform(wp.vec3(px, py, drop_z), wp.quat_identity()))
        body_indices.append(body)
        shape_idx = len(builder.shape_type)

        if name == "sphere":
            builder.add_shape_sphere(body, radius=r, cfg=cfg)
        elif name == "box":
            builder.add_shape_box(body, hx=r, hy=r, hz=r, cfg=cfg)
        elif name == "capsule":
            builder.add_shape_capsule(body, radius=r * 0.7, half_height=r, cfg=cfg)
        elif name == "cylinder":
            builder.add_shape_cylinder(body, radius=r, half_height=r * 0.5, cfg=cfg)
        elif name == "cone":
            builder.add_shape_cone(body, radius=r, half_height=r, cfg=cfg)
        elif name == "mesh":
            if bear_mesh is None:
                bear_pts, bear_idx = _load_bear_mesh()
                bear_mesh = newton.Mesh(bear_pts, np.array(bear_idx, dtype=np.int32))
            mesh_cfg = newton.ModelBuilder.ShapeConfig()
            mesh_cfg.density = cfg.density
            mesh_cfg.ke = cfg.ke
            mesh_cfg.kd = cfg.kd
            mesh_cfg.mu = cfg.mu
            mesh_cfg.has_particle_collision = True
            mesh_cfg.margin = MESH_MARGIN
            builder.add_shape_mesh(body, mesh=bear_mesh, cfg=mesh_cfg)
        shape_indices.append(shape_idx)

    builder.color(include_bending=True)

    return {
        "bag_particle_count": bag_end_particle - bag_start_particle,
        "top_global_indices": top_global_indices,
        "body_indices": body_indices,
        "shape_indices": shape_indices,
        "particle_radius": particle_radius,
    }


def setup_sim(builder, info):
    model = builder.finalize()
    model.soft_contact_ke = 1e3
    model.soft_contact_kd = 1e0
    model.soft_contact_mu = 0.8

    top_idx = info["top_global_indices"]
    flags = model.particle_flags.numpy()
    for vi in top_idx:
        flags[vi] = flags[vi] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags, dtype=wp.int32)

    pq = model.state().particle_q.numpy()
    pinned_indices = wp.array(top_idx.astype(np.int32), dtype=wp.int32)
    pinned_original = wp.array(pq[top_idx].copy(), dtype=wp.vec3)

    pr = info["particle_radius"]
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=15,
        rigid_body_contact_buffer_size=512,
        rigid_body_particle_contact_buffer_size=512,
        particle_enable_self_contact=True,
        particle_self_contact_radius=pr * 2.0,
        particle_self_contact_margin=pr * 3.0,
        particle_topological_contact_filter_threshold=1,
    )

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=pr + MESH_MARGIN + 0.01)

    return model, solver, pipeline, pinned_indices, pinned_original


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0

        seed = getattr(args, "seed", 42)
        builder = newton.ModelBuilder(gravity=-9.8)
        print("Building model...", flush=True)
        self.info = build_model(builder, seed=seed)
        print("Setting up solver...", flush=True)
        self.model, self.solver, self.pipeline, self.pinned_indices, self.pinned_original = setup_sim(
            builder, self.info
        )
        print("Ready.", flush=True)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.3, -0.4, 0.5), -30.0, -60.0)

    def step(self):
        self.frame += 1
        if self.frame == 1:
            print("Simulating...", flush=True)
        dz = 0.0
        if self.frame > SETTLE_FRAMES:
            dz = LIFT_SPEED * (self.frame - SETTLE_FRAMES) * self.frame_dt

        t0 = time.perf_counter()
        for _ in range(self.sim_substeps):
            wp.launch(
                lift_pinned_vertices,
                dim=self.pinned_indices.shape[0],
                inputs=[self.pinned_indices, self.pinned_original, dz],
                outputs=[self.state_0.particle_q, self.state_1.particle_q],
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0
        print(f"frame {self.frame:4d}  step {step_ms:7.1f} ms", flush=True)

        self.sim_time += self.frame_dt

    def simulate(self, num_frames=TOTAL_FRAMES):
        for _ in range(num_frames):
            self.step()

    def render(self):
        if self.viewer and hasattr(self.viewer, "render"):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.render(self.state_0)
            self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_indices = self.info["body_indices"]

        lift_dist = LIFT_SPEED * LIFT_FRAMES * self.frame_dt
        threshold = BAG_ELEVATION + lift_dist * 0.2

        lifted = 0
        for bi in body_indices:
            z = body_q[bi][2]
            if not np.isnan(z) and z > threshold:
                lifted += 1

        assert lifted >= len(body_indices) - 1, (
            f"Only {lifted}/{len(body_indices)} rigid bodies lifted above {threshold:.3f}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=42)
        return parser


# ---------------------------------------------------------------------------
# Unit test
# ---------------------------------------------------------------------------


class BagLiftSim:
    def __init__(self, device):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder(gravity=-9.8)
            self.info = build_model(builder, seed=42)
            self.model, self.solver, self.pipeline, self.pinned_indices, self.pinned_original = setup_sim(
                builder, self.info
            )
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            self.contacts = self.pipeline.contacts()
            self.frame_dt = 1.0 / 60
            self.sim_dt = self.frame_dt / 5

    def run(self):
        for frame in range(TOTAL_FRAMES):
            dz = 0.0
            if frame > SETTLE_FRAMES:
                dz = LIFT_SPEED * (frame - SETTLE_FRAMES) * self.frame_dt

            for _ in range(5):
                wp.launch(
                    lift_pinned_vertices,
                    dim=self.pinned_indices.shape[0],
                    inputs=[self.pinned_indices, self.pinned_original, dz],
                    outputs=[self.state_0.particle_q, self.state_1.particle_q],
                )
                self.state_0.clear_forces()
                self.pipeline.collide(self.state_0, self.contacts)
                self.solver.step(
                    self.state_0,
                    self.state_1,
                    self.control,
                    self.contacts,
                    self.sim_dt,
                )
                self.state_0, self.state_1 = self.state_1, self.state_0


class TestBagLift(unittest.TestCase):
    pass


def test_bag_lift(test, device):
    sim = BagLiftSim(device)
    sim.run()

    body_q = sim.state_0.body_q.numpy()
    body_indices = sim.info["body_indices"]
    lift_dist = LIFT_SPEED * LIFT_FRAMES * sim.frame_dt
    threshold = BAG_ELEVATION + lift_dist * 0.2

    lifted = 0
    for bi in body_indices:
        z = body_q[bi][2]
        if not np.isnan(z) and z > threshold:
            lifted += 1

    test.assertGreaterEqual(
        lifted,
        len(body_indices) - 1,
        f"Only {lifted}/{len(body_indices)} rigid bodies lifted above {threshold:.3f}",
    )


devices = [d for d in get_test_devices(mode="basic") if "cuda" in str(d)]
add_function_test(TestBagLift, "test_bag_lift", test_bag_lift, devices=devices, check_output=False)

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        sys.argv.remove("--test")
        unittest.main()
    else:
        parser = Example.create_parser()
        viewer, args = newton.examples.init(parser)
        example = Example(viewer, args)
        newton.examples.run(example, args)
