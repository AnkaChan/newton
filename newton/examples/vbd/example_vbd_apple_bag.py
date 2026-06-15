# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Apple Bag
#
# Rigid "apples" (spheres) are placed inside a procedural supermarket
# carrier bag -- a single welded thin-shell cloth mesh with two side
# handles, loaded from vbd/asset/supermarket_bag.obj.  The handle tops are
# pinned, so the bag hangs open as if held by its handles.  After the
# apples drop and settle, the pinned handles wiggle in x and y,
# swinging the loaded bag from different directions so the thin plastic shell
# deforms and wrinkles.
#
# A single VBD solver integrates the cloth bag and the rigid apples with
# two-way coupling.  Contact / cloth parameters follow
# example_vbd_bag_franka_pickup_two_way_obj_franka.py.
#
# Commands:
#   # headless validation (no window):
#   python newton/examples/vbd/example_vbd_apple_bag.py --viewer null --num-frames 420 --test
#   # interactive (OpenGL window):
#   python newton/examples/vbd/example_vbd_apple_bag.py
#
###########################################################################

from __future__ import annotations

import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags

BAG_OBJ = os.path.join(os.path.dirname(__file__), "asset", "supermarket_bag.obj")

PARAMS = {
    # --- apples (rigid spheres) ---
    "num_apples": 5,
    "apple_radius": 0.036,
    "apple_margin": 0.005,
    "apple_density": 1000.0,  # ~0.2 kg per apple at r=0.036
    "apple_ke": 5.0e5,
    "apple_kd": 5.0e1,
    "apple_mu": 0.5,
    "apple_drop_offset": 0.045,  # start this far above the rest layer so they drop in
    # --- cloth (plastic bag) ---
    "particle_radius": 0.004,
    "cloth_density": 0.08,
    "cloth_tri_ke": 2.0e4,
    "cloth_tri_ka": 2.0e3,
    "cloth_tri_kd": 1.0e-3,
    "cloth_edge_ke": 0.001,  # low bending -> floppy plastic that wrinkles (high tri_ke keeps it from stretching)
    "cloth_edge_kd": 0.01,
    # --- contacts ---
    "soft_contact_ke": 5.0e5,
    "soft_contact_kd": 1.0e0,
    "soft_contact_mu": 0.2,
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
    # --- pin + wiggle ---
    "pin_band": 0.03,  # pin bag vertices within this many m of the topmost vertex
    "settle_frames": 150,  # let the apples drop and settle before wiggling
    "wiggle_amplitude": 0.085,  # left<->right travel of the pinned handles [m]
    "wiggle_freq": 0.55,  # wiggle frequency [Hz]
    "wiggle_y_amplitude": 0.055,  # front<->back travel of the pinned handles [m]
    "wiggle_y_freq": 0.37,  # y-axis wiggle frequency [Hz]
    "wiggle_y_phase": 0.5 * math.pi,  # phase offset from the x wiggle [rad]
    "wiggle_bob": 0.035,  # vertical bob of the pinned handles [m] (adds bounce/jostle)
    "wiggle_bob_freq": 1.1,  # bob frequency [Hz] (~2x swing -> lively shake)
    "wiggle_ramp": 0.6,  # ease the wiggle in over this many seconds
    "wiggle_axis": 0,  # 0 = x (left<->right)
    # --- view --- 3/4 elevated front view (wireframe shows the apples sloshing inside)
    "camera_pos": (0.22, -1.0, 0.48),
    "camera_target": (0.0, 0.0, 0.14),
    "camera_fov": 45.0,
    "draw_wireframe": True,
    "initial_paused": False,
    "seed": 42,
}


def _pitch_yaw(pos, target):
    """Pitch/yaw (deg) to look from pos toward target, Z-up convention (see camera.py)."""
    d = np.array(target, dtype=np.float64) - np.array(pos, dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-9
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(d[2])))))
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    return pitch, yaw


def _load_obj(path):
    """Load a triangle-mesh OBJ as (vertices [V,3] float32, faces flat int list)."""
    vertices = []
    faces = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                # OBJ is 1-indexed; fan-triangulate any polygon into triangles.
                for k in range(1, len(idx) - 1):
                    faces.extend([idx[0] - 1, idx[k] - 1, idx[k + 1] - 1])
    if not vertices or not faces:
        raise ValueError(f"OBJ has no geometry: {path}")
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


def _apple_layout(num, r, margin, half_x, half_y, z_floor, drop_offset, rng):
    """Lay apples out in x-rows stacked in layers, centered, with a small drop."""
    half_x_in = max(r, half_x - r - margin)
    spacing = 2.0 * r + 0.022
    per_layer = max(1, int((2.0 * half_x_in) / spacing) + 1)
    layer_gap = 2.0 * r + 0.008
    base_z = z_floor + r + 0.012

    positions = []
    for k in range(num):
        layer = k // per_layer
        col = k % per_layer
        n_here = min(per_layer, num - layer * per_layer)
        xs = (np.arange(n_here) - (n_here - 1) * 0.5) * spacing
        stagger = 0.5 * spacing if (layer % 2 == 1) else 0.0
        x = float(xs[col] + stagger)
        x = float(np.clip(x, -half_x_in, half_x_in))
        y = float(rng.uniform(-0.4, 0.4) * max(0.0, half_y - r - margin))
        z = base_z + layer * layer_gap + drop_offset
        positions.append((x, y, z))
    return positions


def build_model(builder, params, seed):
    rng = np.random.default_rng(seed)

    bag_verts, bag_faces = _load_obj(BAG_OBJ)
    va = params["vertical_axis"]

    pr = params["particle_radius"]
    bag_start_particle = len(builder.particle_q)

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

    bag_end_particle = len(builder.particle_q)

    # Pin the handle tops: bag vertices within pin_band of the topmost vertex.
    z_top = float(bag_verts[:, va].max())
    z_floor = float(bag_verts[:, va].min())
    pin_mask = bag_verts[:, va] >= z_top - params["pin_band"]
    top_global_indices = np.where(pin_mask)[0] + bag_start_particle

    half_x = 0.5 * float(bag_verts[:, 0].max() - bag_verts[:, 0].min())
    half_y = 0.5 * float(bag_verts[:, 1].max() - bag_verts[:, 1].min())

    # Rigid apples
    r = params["apple_radius"]
    positions = _apple_layout(
        params["num_apples"], r, params["apple_margin"], half_x, half_y, z_floor, params["apple_drop_offset"], rng
    )

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = params["apple_density"]
    cfg.ke = params["apple_ke"]
    cfg.kd = params["apple_kd"]
    cfg.mu = params["apple_mu"]
    cfg.has_particle_collision = True
    cfg.margin = params["apple_margin"]

    body_indices = []
    shape_indices = []
    for i, (px, py, pz) in enumerate(positions):
        body = builder.add_body(xform=wp.transform(wp.vec3(px, py, pz), wp.quat_identity()), label=f"apple_{i}")
        body_indices.append(body)
        shape_indices.append(len(builder.shape_type))
        builder.add_shape_sphere(body, radius=r, cfg=cfg)

    builder.color(include_bending=True)

    return {
        "bag_particle_count": bag_end_particle - bag_start_particle,
        "top_global_indices": top_global_indices,
        "body_indices": body_indices,
        "shape_indices": shape_indices,
        "particle_radius": pr,
        "z_floor": z_floor,
        "z_top": z_top,
        "half_width": half_x,
        "half_depth": half_y,
    }


def setup_sim(builder, info, params):
    model = builder.finalize()
    model.soft_contact_ke = params["soft_contact_ke"]
    model.soft_contact_kd = params["soft_contact_kd"]
    model.soft_contact_mu = params["soft_contact_mu"]

    top_idx = info["top_global_indices"]
    flags = model.particle_flags.numpy()
    for vi in top_idx:
        flags[vi] = flags[vi] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags, dtype=wp.int32)

    pq = model.state().particle_q.numpy()
    pinned_indices = wp.array(top_idx.astype(np.int32), dtype=wp.int32)
    pinned_original = wp.array(pq[top_idx].copy(), dtype=wp.vec3)

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

    return model, solver, pipeline, pinned_indices, pinned_original


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
        self.model, self.solver, self.pipeline, self.pinned_indices, self.pinned_original = setup_sim(
            builder, self.info, self.params
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()

        print(
            f"[apple_bag] bag particles: {self.info['bag_particle_count']}  "
            f"pinned handle-top verts: {len(self.info['top_global_indices'])}  "
            f"apples: {len(self.info['body_indices'])}"
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

    def _wiggle_offset(self):
        if self.frame <= self.params["settle_frames"]:
            return wp.vec3(0.0, 0.0, 0.0)
        t_w = (self.frame - self.params["settle_frames"]) * self.frame_dt
        ramp = min(1.0, t_w / self.params["wiggle_ramp"])
        dx = self.params["wiggle_amplitude"] * ramp * math.sin(2.0 * math.pi * self.params["wiggle_freq"] * t_w)
        dy = (
            self.params["wiggle_y_amplitude"]
            * ramp
            * math.sin(2.0 * math.pi * self.params["wiggle_y_freq"] * t_w + self.params["wiggle_y_phase"])
        )
        dz = self.params["wiggle_bob"] * ramp * math.sin(2.0 * math.pi * self.params["wiggle_bob_freq"] * t_w)
        off = [0.0, 0.0, 0.0]
        off[self.params["wiggle_axis"]] += dx
        off[1] += dy
        off[self.params["vertical_axis"]] += dz
        return wp.vec3(*off)

    def simulate(self):
        offset = self._wiggle_offset()
        for _ in range(self.sim_substeps):
            wp.launch(
                move_pinned_vertices,
                dim=self.pinned_indices.shape[0],
                inputs=[self.pinned_indices, self.pinned_original, offset],
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
        assert np.all(np.isfinite(pq)), "Bag particle positions contain non-finite values"

        body_q = self.state_0.body_q.numpy()
        body_indices = self.info["body_indices"]
        apple_pos = body_q[body_indices][:, :3]
        assert np.all(np.isfinite(apple_pos)), "Apple positions contain non-finite values"

        va = self.params["vertical_axis"]
        az = apple_pos[:, va]
        assert np.all(az > self.info["z_floor"] - 0.06), f"An apple fell through the bag bottom: min z {az.min():.3f}"
        assert np.all(az < self.info["z_top"] + 0.02), f"An apple is above the handles: max z {az.max():.3f}"

        motion_limits = [0.0, 0.0, 0.0]
        motion_limits[self.params["wiggle_axis"]] += self.params["wiggle_amplitude"]
        motion_limits[1] += self.params["wiggle_y_amplitude"]
        x_lim = self.info["half_width"] + motion_limits[0] + self.params["apple_radius"] + 0.08
        y_lim = self.info["half_depth"] + motion_limits[1] + self.params["apple_radius"] + 0.06
        assert np.all(np.abs(apple_pos[:, 0]) < x_lim), "An apple escaped the bag in x"
        assert np.all(np.abs(apple_pos[:, 1]) < y_lim), "An apple escaped the bag in y"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=PARAMS["seed"])
        parser.set_defaults(num_frames=PARAMS["settle_frames"] + 270)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
