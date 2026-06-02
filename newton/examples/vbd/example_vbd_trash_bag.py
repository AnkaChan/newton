# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Trash Bag
#
# Rigid "apples" (spheres) are dropped into a procedural drawstring trash
# bag and the bag is cinched shut by its drawstring.  Two thin-shell cloth
# meshes are loaded from vbd/asset/:
#   - trash_bag.obj      the bag shell, whose top hem folds OUTWARD into two
#                        tunnels (the colored "stripe" band)
#   - trash_bag_rope.obj the drawstring "tie", a single-layer cloth ribbon
#                        that follows the bag contour through both tunnels and
#                        out the four side holes as two handles
# trash_bag_layout.json supplies the 38 tunnel-closure spring pairs (flap edge
# <-> wall, keeping each tunnel shut around the rope while the mesh stays
# manifold) and the rope handle vertex indices.
#
# The two drawstring handles (the exposed left/right rope ends) are pinned.
# After the apples settle, the handles are pulled UP and APART: the nearly
# inextensible rope loop collapses in the perpendicular (depth) direction,
# gathering the mouth shut around the apples -- the "rope to tie".
#
# Contact / cloth parameters follow example_vbd_apple_bag.py.
#
# Commands:
#   python newton/examples/vbd/example_vbd_trash_bag.py --num-frames 360
#   python newton/examples/vbd/example_vbd_trash_bag.py --viewer null --num-frames 360 --test
###########################################################################

from __future__ import annotations

import json
import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags

ASSET = os.path.join(os.path.dirname(__file__), "asset")
BAG_OBJ = os.path.join(ASSET, "trash_bag.obj")
ROPE_OBJ = os.path.join(ASSET, "trash_bag_rope.obj")
LAYOUT_JSON = os.path.join(ASSET, "trash_bag_layout.json")

PARAMS = {
    # --- apples (rigid spheres) ---
    "num_apples": 5,
    "apple_radius": 0.034,
    "apple_margin": 0.005,
    "apple_density": 1000.0,
    "apple_ke": 5.0e5,
    "apple_kd": 5.0e1,
    "apple_mu": 0.5,
    # --- bag cloth (floppy plastic) ---
    "particle_radius": 0.004,
    "cloth_density": 0.08,
    "cloth_tri_ke": 2.0e4,
    "cloth_tri_ka": 2.0e3,
    "cloth_tri_kd": 1.0e2,
    "cloth_edge_ke": 0.05,  # low bending -> floppy, wrinkly plastic
    "cloth_edge_kd": 0.001,
    # --- rope cloth (the tie): stiff/inextensible so pulling collapses the loop ---
    "rope_density": 0.2,
    "rope_tri_ke": 3.0e5,
    "rope_tri_ka": 1.0e4,
    "rope_tri_kd": 1.0e2,
    "rope_edge_ke": 0.2,
    "rope_edge_kd": 0.01,
    # --- springs ---
    "closure_ke": 2.0e3,  # tunnel closure: flap free edge <-> wall
    "closure_kd": 1.0e0,
    "tether_ke": 6.0e3,  # rope tunnel verts <-> nearest bag wall vert (hangs bag on the tie)
    "tether_kd": 1.0e0,
    # --- contacts ---
    "soft_contact_ke": 5.0e5,
    "soft_contact_kd": 1.0e1,
    "soft_contact_mu": 1.0,
    "soft_contact_creation_margin": 0.012,
    "rigid_body_particle_contact_buffer_size": 4096,
    "rigid_body_contact_buffer_size": 1024,
    "rigid_contact_hard": True,
    # --- solver / time ---
    "fps": 60,
    "sim_substeps": 10,
    "solver_iterations": 12,
    "gravity": -9.8,
    "vertical_axis": 2,
    # --- pin + cinch ---
    "settle_frames": 150,
    "cinch_frames": 200,
    "cinch_up": 0.26,  # how far the handles rise [m]
    "cinch_together": 0.13,  # how far each handle pulls INWARD in x [m] (gathers the neck shut)
    "cinch_ramp": 1.4,  # ease the cinch in over this many seconds
    # --- view (fixed for the whole clip; framed for full motion z in [0, ~0.62]) ---
    "camera_pos": (0.52, -0.98, 0.58),
    "camera_target": (0.0, 0.0, 0.3),
    "camera_fov": 45.0,
    "draw_wireframe": True,
    "initial_paused": False,
    "seed": 42,
}


def _pitch_yaw(pos, target):
    d = np.array(target, dtype=np.float64) - np.array(pos, dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-9
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(d[2])))))
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    return pitch, yaw


def _load_obj(path):
    """Load a triangle-mesh OBJ preserving vertex order (so JSON indices stay valid)."""
    vertices = []
    faces = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.extend([idx[0] - 1, idx[k] - 1, idx[k + 1] - 1])
    return np.array(vertices, dtype=np.float32), faces


@wp.kernel
def move_pinned_vertices(
    pinned_indices: wp.array[wp.int32],
    original_positions: wp.array[wp.vec3],
    offset: wp.vec3,
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    tid = wp.tid()
    vi = pinned_indices[tid]
    new_p = original_positions[tid] + offset
    pos_0[vi] = new_p
    pos_1[vi] = new_p


def build_model(builder, params, seed):
    rng = np.random.default_rng(seed)
    layout = json.load(open(LAYOUT_JSON))
    pr = params["particle_radius"]

    # --- bag shell ---
    bag_verts, bag_faces = _load_obj(BAG_OBJ)
    bag_start = len(builder.particle_q)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=bag_verts.tolist(),
        indices=bag_faces,
        density=params["cloth_density"],
        tri_ke=params["cloth_tri_ke"],
        tri_ka=params["cloth_tri_ka"],
        tri_kd=params["cloth_tri_kd"],
        edge_ke=params["cloth_edge_ke"],
        edge_kd=params["cloth_edge_kd"],
        particle_radius=pr,
    )

    # --- drawstring tie (ribbon) ---
    rope_verts, rope_faces = _load_obj(ROPE_OBJ)
    rope_start = len(builder.particle_q)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=rope_verts.tolist(),
        indices=rope_faces,
        density=params["rope_density"],
        tri_ke=params["rope_tri_ke"],
        tri_ka=params["rope_tri_ka"],
        tri_kd=params["rope_tri_kd"],
        edge_ke=params["rope_edge_ke"],
        edge_kd=params["rope_edge_kd"],
        particle_radius=pr,
    )

    # --- tunnel-closure springs: flap free edge <-> wall (bag-local indices) ---
    for i, j in layout["tunnel_spring_pairs"]:
        builder.add_spring(bag_start + i, bag_start + j, params["closure_ke"], params["closure_kd"], 0.0)

    # --- tethers: rope tunnel verts <-> nearest bag wall vert (hang the bag on the tie) ---
    rope_labels = layout["rope"]["vertex_labels"]
    tunnel_rope_local = [i for i, lbl in enumerate(rope_labels) if lbl in ("front_tunnel", "back_tunnel")]
    for i in tunnel_rope_local:
        d = np.linalg.norm(bag_verts - rope_verts[i], axis=1)
        j = int(np.argmin(d))
        builder.add_spring(rope_start + i, bag_start + j, params["tether_ke"], params["tether_kd"], 0.0)

    # --- pinned drawstring handles (the two exposed side ends) ---
    hv = layout["rope"]["handle_vertex_indices"]
    right_idx = np.array([rope_start + i for i in hv["right"]], dtype=np.int32)
    left_idx = np.array([rope_start + i for i in hv["left"]], dtype=np.int32)

    # --- rigid apples dropped inside ---
    r = params["apple_radius"]
    z_floor = float(bag_verts[:, 2].min())
    half_x = 0.5 * float(bag_verts[:, 0].max() - bag_verts[:, 0].min()) - r - 0.02
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = params["apple_density"]
    cfg.ke = params["apple_ke"]
    cfg.kd = params["apple_kd"]
    cfg.mu = params["apple_mu"]
    cfg.has_particle_collision = True
    cfg.margin = params["apple_margin"]

    body_indices = []
    n = params["num_apples"]
    xs = (np.arange(n) - (n - 1) * 0.5) * (2.0 * r + 0.01)
    for i in range(n):
        px = float(np.clip(xs[i], -half_x, half_x))
        py = float(rng.uniform(-0.012, 0.012))
        pz = z_floor + r + 0.03 + (i % 2) * 0.05 + 0.10  # staggered, drop into the bottom
        body = builder.add_body(xform=wp.transform(wp.vec3(px, py, pz), wp.quat_identity()), label=f"apple_{i}")
        body_indices.append(body)
        builder.add_shape_sphere(body, radius=r, cfg=cfg)

    builder.color(include_bending=True)

    return {
        "bag_start": bag_start,
        "rope_start": rope_start,
        "bag_count": rope_start - bag_start,
        "rope_count": len(rope_verts),
        "right_idx": right_idx,
        "left_idx": left_idx,
        "body_indices": body_indices,
        "z_floor": z_floor,
        "num_tunnel_springs": len(layout["tunnel_spring_pairs"]),
        "num_tethers": len(tunnel_rope_local),
    }


def setup_sim(builder, info, params):
    model = builder.finalize()
    model.soft_contact_ke = params["soft_contact_ke"]
    model.soft_contact_kd = params["soft_contact_kd"]
    model.soft_contact_mu = params["soft_contact_mu"]

    pin_idx = np.concatenate([info["right_idx"], info["left_idx"]])
    flags = model.particle_flags.numpy()
    for vi in pin_idx:
        flags[vi] = flags[vi] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags, dtype=wp.int32)

    pq = model.state().particle_q.numpy()
    right = wp.array(info["right_idx"], dtype=wp.int32)
    left = wp.array(info["left_idx"], dtype=wp.int32)
    right_orig = wp.array(pq[info["right_idx"]].copy(), dtype=wp.vec3)
    left_orig = wp.array(pq[info["left_idx"]].copy(), dtype=wp.vec3)

    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=params["solver_iterations"],
        rigid_body_particle_contact_buffer_size=params["rigid_body_particle_contact_buffer_size"],
        rigid_body_contact_buffer_size=params["rigid_body_contact_buffer_size"],
        particle_enable_self_contact=False,
        rigid_contact_hard=params["rigid_contact_hard"],
    )
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=params["soft_contact_creation_margin"]
    )
    return model, solver, pipeline, (right, left, right_orig, left_orig)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.sim_time = 0.0
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0

        seed = getattr(args, "seed", self.params["seed"])
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self.info = build_model(builder, self.params, seed=seed)
        self.model, self.solver, self.pipeline, pins = setup_sim(builder, self.info, self.params)
        self.right, self.left, self.right_orig, self.left_orig = pins

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()

        print(
            f"[trash_bag] bag verts {self.info['bag_count']}  rope verts {self.info['rope_count']}  "
            f"tunnel springs {self.info['num_tunnel_springs']}  tethers {self.info['num_tethers']}  "
            f"apples {len(self.info['body_indices'])}"
        )

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = self.params["draw_wireframe"]
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = self.params["initial_paused"]
        if hasattr(self.viewer, "set_camera"):
            pitch, yaw = _pitch_yaw(self.params["camera_pos"], self.params["camera_target"])
            self.viewer.set_camera(wp.vec3(*self.params["camera_pos"]), pitch, yaw)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

    def _cinch(self):
        """Offsets for the right and left pinned handles."""
        if self.frame <= self.params["settle_frames"]:
            return wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)
        t_c = (self.frame - self.params["settle_frames"]) * self.frame_dt
        ramp = min(1.0, t_c / self.params["cinch_ramp"])
        up = self.params["cinch_up"] * ramp
        tg = self.params["cinch_together"] * ramp
        # gather the neck: right handle pulls inward (-x), left inward (+x), both lift up
        return wp.vec3(-tg, 0.0, up), wp.vec3(tg, 0.0, up)

    def simulate(self):
        off_r, off_l = self._cinch()
        for _ in range(self.sim_substeps):
            wp.launch(
                move_pinned_vertices,
                dim=self.right.shape[0],
                inputs=[self.right, self.right_orig, off_r],
                outputs=[self.state_0.particle_q, self.state_1.particle_q],
            )
            wp.launch(
                move_pinned_vertices,
                dim=self.left.shape[0],
                inputs=[self.left, self.left_orig, off_l],
                outputs=[self.state_0.particle_q, self.state_1.particle_q],
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pq = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(pq)), "Cloth positions contain non-finite values"
        body_q = self.state_0.body_q.numpy()
        apple_pos = body_q[self.info["body_indices"]][:, :3]
        assert np.all(np.isfinite(apple_pos)), "Apple positions contain non-finite values"
        # apples stay roughly within the loaded/lifted bag (no escape / explosion)
        assert np.all(np.abs(apple_pos[:, 0]) < 0.4), "An apple escaped in x"
        assert np.all(np.abs(apple_pos[:, 1]) < 0.3), "An apple escaped in y"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=PARAMS["seed"])
        parser.set_defaults(num_frames=PARAMS["settle_frames"] + PARAMS["cinch_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
