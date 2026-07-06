# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Trash Bag H1 Full (unified AVBD/VBD)
#
# The full trash-day demo, extending example_vbd_trash_bag_h1.py: instead
# of the trash spawning inside the bag, the spheres start on a side table
# to the H1's right.  The right hand picks each sphere up, carries it over
# the bag mouth and drops it in.  Once all trash is deposited, both hands
# grab the drawstring handles, cinch the mouth shut and lift the loaded
# bag out of the can (the sequence of the base example).
#
# A held sphere is carried kinematically: its inverse mass is zeroed and
# its transform follows the right hand (the solver is told via
# notify_model_changed); on release the mass is restored and it falls.
#
# Commands:
#   python newton/examples/vbd/example_vbd_trash_bag_h1_full.py
#   python newton/examples/vbd/example_vbd_trash_bag_h1_full.py --viewer gl --headless \
#       --record-video trash_bag_h1_full.mp4
#   python newton/examples/vbd/example_vbd_trash_bag_h1_full.py --viewer null --test
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd import example_vbd_trash_bag_h1 as base

_TRANSFER_SEGMENTS = ("fetch", "pick", "haul", "drop")

FULL_PARAMS = dict(base.PARAMS)
FULL_PARAMS.update(
    {
        # --- trash starts on a side table at the robot's right ---
        "num_trash": 3,
        # margin 0 like the tablecloth tableware: a 5 mm rigid-contact margin
        # on a 34 mm sphere resting on a box produces explosive contact kicks
        "trash_margin": 0.0,
        "trash_table_x": -0.48,
        "trash_table_y": -0.36,
        "trash_table_top_z": 1.10,
        "trash_table_half_x": 0.11,
        "trash_table_half_y": 0.09,
        "trash_spacing": 0.07,  # sphere spacing along x on the table
        # --- per-sphere transfer segments (seconds) ---
        "initial_settle_time": 1.2,
        "fetch_time": 0.9,  # right hand from rest to above the sphere
        "pick_time": 0.5,  # descend + curl fingers; attach at the end
        "haul_time": 1.0,  # carry the sphere over the bag mouth
        "drop_time": 0.6,  # release above the mouth, fingers open
        # --- transfer geometry ---
        "ball_hover_height": 0.12,  # hover above a sphere before picking
        "ball_grasp_z_offset": 0.005,  # pinch point relative to the sphere center
        "drop_x": -0.22,  # release point: over the bag mouth center
        "drop_y": 0.0,
        "drop_z": 1.40,
        "drop_release_fraction": 0.4,  # release this far into the drop segment
        # finger curl while carrying a sphere
        "ball_hook_fraction": 0.45,
        "ball_hold_fraction": 0.85,
    }
)
# The base sequence's "settle" window covers the initial settle plus every
# per-sphere transfer; the bag-grasp phases follow unchanged after it.
FULL_PARAMS["settle_time"] = FULL_PARAMS["initial_settle_time"] + FULL_PARAMS["num_trash"] * sum(
    FULL_PARAMS[f"{segment}_time"] for segment in _TRANSFER_SEGMENTS
)


class Example(base.Example):
    DEFAULT_PARAMS = FULL_PARAMS

    def __init__(self, viewer, args):
        self.held_ball = None
        self._ball_local = None
        self._delivered = None
        super().__init__(viewer, args)
        self._delivered = [False] * len(self.trash_bodies)
        self._transfer_cycle = sum(self.params[f"{segment}_time"] for segment in _TRANSFER_SEGMENTS)

    # ------------------------------------------------------------ scene ---
    def _build_trash(self, builder, seed):
        """Side table with the trash spheres resting on top, right of the H1."""
        p = self.params
        rng = np.random.default_rng(seed)
        top = p["trash_table_top_z"]
        table_cfg = newton.ModelBuilder.ShapeConfig(
            ke=p["shape_ke"],
            kd=p["shape_kd"],
            mu=p["shape_mu"],
            gap=p["rigid_contact_gap"],
            has_particle_collision=False,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(p["trash_table_x"], p["trash_table_y"], 0.5 * top),
                wp.quat_identity(),
            ),
            hx=p["trash_table_half_x"],
            hy=p["trash_table_half_y"],
            hz=0.5 * top,
            cfg=table_cfg,
            color=wp.vec3(0.50, 0.38, 0.26),
            label="trash_table",
        )

        r = p["trash_radius"]
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = p["trash_density"]
        cfg.ke = p["trash_ke"]
        cfg.kd = p["trash_kd"]
        cfg.mu = p["trash_mu"]
        cfg.has_particle_collision = True
        cfg.margin = p["trash_margin"]

        colors = [
            wp.vec3(0.85, 0.30, 0.20),
            wp.vec3(0.90, 0.70, 0.20),
            wp.vec3(0.30, 0.55, 0.85),
        ]
        body_indices = []
        ball_shapes = []
        self.ball_homes = []
        n = p["num_trash"]
        for i in range(n):
            px = p["trash_table_x"] + p["trash_spacing"] * (i - 0.5 * (n - 1))
            py = p["trash_table_y"] + float(rng.uniform(-0.005, 0.005))
            pz = top + r + 0.002
            body = builder.add_body(xform=wp.transform(wp.vec3(px, py, pz), wp.quat_identity()), label=f"trash_{i}")
            body_indices.append(body)
            shape = builder.add_shape_sphere(body, radius=r, cfg=cfg, color=colors[i % len(colors)])
            # The spheres are picked and carried kinematically; rigid contact
            # against the (infinite-mass) robot would only punt them off the
            # table while the fingers close around them.
            for robot_shape in self._robot_rigid_shapes:
                builder.add_shape_collision_filter_pair(robot_shape, shape)
            # A kinematically carried sphere sweeping past its neighbours would
            # punt them off the table, so spheres ignore each other too.
            for other in ball_shapes:
                builder.add_shape_collision_filter_pair(other, shape)
            ball_shapes.append(shape)
            self.ball_homes.append(np.array([px, py, pz], dtype=np.float32))
        return body_indices

    # ------------------------------------------------------- trajectory ---
    def _hand_targets(self, t):
        p = self.params
        if t >= self._phase_ends["settle"]:
            return self._bag_grasp_targets(t)

        left = self.rest_positions[0]
        right = self.rest_positions[1]
        thumb = 0.0
        index = 0.0
        other = 0.0

        t_transfer = t - p["initial_settle_time"]
        if t_transfer < 0.0:
            self.phase = "settle"
            return np.stack([left, right]), 0.0, 0.0, 0.0, 0.0, 0.0

        ball = min(int(t_transfer // self._transfer_cycle), len(self.trash_bodies) - 1)
        t_seg = t_transfer - ball * self._transfer_cycle
        home = self.ball_homes[ball]
        above = home + np.array([0.0, 0.0, p["ball_hover_height"]], dtype=np.float32)
        grasp = home + np.array([0.0, 0.0, p["ball_grasp_z_offset"]], dtype=np.float32)
        drop = np.array([p["drop_x"], p["drop_y"], p["drop_z"]], dtype=np.float32)

        fetch_end = p["fetch_time"]
        pick_end = fetch_end + p["pick_time"]
        haul_end = pick_end + p["haul_time"]

        if t_seg < fetch_end:
            self.phase = f"fetch_{ball}"
            u = base._smoothstep(t_seg / p["fetch_time"])
            start = self.rest_positions[1] if ball == 0 else drop
            right = start * (1.0 - u) + above * u
            other = p["ball_hook_fraction"] * u
            index = p["ball_hook_fraction"] * u
        elif t_seg < pick_end:
            self.phase = f"pick_{ball}"
            u = base._smoothstep((t_seg - fetch_end) / p["pick_time"])
            right = above * (1.0 - u) + grasp * u
            other = p["ball_hook_fraction"] + (p["ball_hold_fraction"] - p["ball_hook_fraction"]) * u
            index = other
            thumb = p["ball_hold_fraction"] * u
        elif t_seg < haul_end:
            self.phase = f"haul_{ball}"
            if self.held_ball is None and not self._delivered[ball]:
                self._attach_ball(self.trash_bodies[ball])
            u = base._smoothstep((t_seg - pick_end) / p["haul_time"])
            # rise quickly, then travel: keeps the sphere clear of the rim and
            # the pinned drawstring handle it passes over
            u_z = base._smoothstep(min(1.0, 3.0 * (t_seg - pick_end) / p["haul_time"]))
            right = grasp * (1.0 - u) + drop * u
            right[2] = grasp[2] * (1.0 - u_z) + drop[2] * u_z
            other = p["ball_hold_fraction"]
            index = other
            thumb = p["ball_hold_fraction"]
        else:
            self.phase = f"drop_{ball}"
            u = (t_seg - haul_end) / p["drop_time"]
            right = drop.copy()
            if u >= p["drop_release_fraction"] and self.held_ball is not None:
                self._release_ball(ball)
            if self._delivered[ball]:
                open_u = base._smoothstep(
                    (u - p["drop_release_fraction"]) / max(1e-6, 1.0 - p["drop_release_fraction"])
                )
                hold = p["ball_hold_fraction"]
                other = hold * (1.0 - open_u) + p["ball_hook_fraction"] * open_u
                index = other
                thumb = hold * (1.0 - open_u)
            else:
                other = p["ball_hold_fraction"]
                index = other
                thumb = p["ball_hold_fraction"]

        # left hand rests; right-hand fractions apply to the shared "other"
        positions = np.stack([left, np.asarray(right, dtype=np.float32)])
        return positions, 0.0, thumb, 0.0, index, other

    # ------------------------------------------------- sphere attachment ---
    def _attach_ball(self, ball):
        inv_mass = self.model.body_inv_mass.numpy()
        inv_inertia = self.model.body_inv_inertia.numpy()
        self._saved_ball_inv = (inv_mass[ball].copy(), inv_inertia[ball].copy())
        inv_mass[ball] = 0.0
        inv_inertia[ball] = 0.0
        self.model.body_inv_mass.assign(inv_mass)
        self.model.body_inv_inertia.assign(inv_inertia)
        self.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

        body_q = self.state_0.body_q.numpy()
        hand_tf = wp.transform(*body_q[self.hand_bodies[1]])
        ball_tf = wp.transform(*body_q[ball])
        self._ball_local = wp.transform_multiply(wp.transform_inverse(hand_tf), ball_tf)
        self.held_ball = ball
        pinch = np.asarray(wp.transform_point(hand_tf, self.hand_offsets[1]))
        offset = float(np.linalg.norm(pinch - body_q[ball][:3]))
        print(
            f"[trash_bag_h1_full] picked sphere (body {ball}) at t={self.sim_time:.2f}s  "
            f"pinch-to-ball offset {offset:.3f}m",
            flush=True,
        )

    def _release_ball(self, ball_slot):
        ball = self.held_ball
        inv_mass = self.model.body_inv_mass.numpy()
        inv_inertia = self.model.body_inv_inertia.numpy()
        inv_mass[ball], inv_inertia[ball] = self._saved_ball_inv
        self.model.body_inv_mass.assign(inv_mass)
        self.model.body_inv_inertia.assign(inv_inertia)
        self.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        self.held_ball = None
        self._delivered[ball_slot] = True
        print(f"[trash_bag_h1_full] dropped sphere (body {ball}) at t={self.sim_time:.2f}s", flush=True)

    def _drive_robot_kinematic(self):
        super()._drive_robot_kinematic()
        if self.held_ball is None:
            return
        body_q = self.state_0.body_q.numpy()
        hand_tf = wp.transform(*body_q[self.hand_bodies[1]])
        ball_tf = wp.transform_multiply(hand_tf, self._ball_local)
        pos = wp.transform_get_translation(ball_tf)
        rot = wp.transform_get_rotation(ball_tf)
        ball_row = np.array([pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]], dtype=np.float32)
        for state in (self.state_0, self.state_1):
            bq = state.body_q.numpy()
            bqd = state.body_qd.numpy()
            bq[self.held_ball] = ball_row
            bqd[self.held_ball] = bqd[self.hand_bodies[1]]
            state.body_q.assign(bq)
            state.body_qd.assign(bqd)

    # ------------------------------------------------------------- test ---
    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Cloth state contains non-finite values"
        assert np.all(np.isfinite(body_q)), "Rigid state contains non-finite values"
        assert self.attached, "The H1 never grasped the drawstring handles"
        assert all(self._delivered), "The H1 did not deposit every trash sphere"
        assert self.held_ball is None, "The H1 is still holding a trash sphere"

        bag = particle_q[self.bag_info["bag_start"] : self.bag_info["bag_start"] + self.bag_info["bag_count"]]
        assert float(bag[:, 2].min()) > self.params["pedestal_top_z"] + 0.05, "Bag was not lifted off the can floor"

        trash = body_q[self.trash_bodies][:, :3]
        bag_centroid = bag.mean(axis=0)
        assert np.all(np.linalg.norm(trash - bag_centroid, axis=1) < 0.6), "Trash escaped the bag"
        assert float(trash[:, 2].min()) > 0.2, "Trash fell to the ground"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
