# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Native jacket manipulation with two original coupled Unitree H1 hands."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
from experiments.collision_benchmark.h1_gripper import H1HandDriver, add_h1_hand
from experiments.collision_benchmark.jacket_asset import load_native_jacket
from experiments.collision_benchmark.jacket_diagnostics import ClothStrain
from experiments.collision_benchmark.manipulation import tick
from newton.solvers import SolverVBD

SCENE_IDS = ["j3_h1_jacket"]
_PINCH_RECIPES = [
    {
        "side": "left",
        "vertex": 33404,
        "roll": 75,
        "open": {
            "controls": [1.25, 0.008686668469377504, 1.0, 0, 0, 0],
            "physical_mesh_gap_m": 0.030000014855481916,
            "requested_mesh_gap_m": 0.03,
            "grip_center_root_m": [0.12362329018475814, -0.07825451495165134, 0.02839300183166217],
            "grip_normal_root": [0.7337062376240556, 0.6731982637483205, -0.09208286788430997],
            "surface_points_root_m": [
                [0.11261769117061743, -0.08835249390821846, 0.02977424553389451],
                [0.13462888919889884, -0.06815653599508421, 0.02701175812942983],
            ],
        },
        "closed": {
            "controls": [1.25, 0.17689289119915097, 1.0, 0, 0, 0],
            "physical_mesh_gap_m": 0.004799966768924005,
            "requested_mesh_gap_m": 0.0048,
            "grip_center_root_m": [0.1371606863638369, -0.07150989443932612, 0.02853556095637639],
            "grip_normal_root": [0.07409915184291248, 0.9663979875394891, -0.24613867102872364],
            "surface_points_root_m": [
                [0.1369828496306112, -0.07382923355219842, 0.029126289677118885],
                [0.13733852309706263, -0.06919055532645384, 0.027944832235633894],
            ],
        },
        "rotation_xyzw": [0.7944934961779014, 0.17309151324639055, 0.5816160645473168, -0.023284459057084747],
        "normal": [0.08600780993453117, -0.9477347270879378, 0.3072483420652052],
        "initial_offset": 0.004,
    },
    {
        "side": "right",
        "vertex": 31949,
        "roll": -70,
        "open": {
            "controls": [1.25, 0.009059418357859632, 1.0, 0, 0, 0],
            "physical_mesh_gap_m": 0.02999997695551713,
            "requested_mesh_gap_m": 0.03,
            "grip_center_root_m": [0.1231038363683342, 0.07832773053487563, 0.026401426443010707],
            "grip_normal_root": [0.7204390682293661, -0.6931843275509851, -0.02151829469291947],
            "surface_points_root_m": [
                [0.1122972586459666, 0.08872548746110323, 0.026724200615465512],
                [0.1339104140907018, 0.06792997360864804, 0.0260786522705559],
            ],
        },
        "closed": {
            "controls": [1.25, 0.17951209771354962, 1.0, 0, 0, 0],
            "physical_mesh_gap_m": 0.004799982604040795,
            "requested_mesh_gap_m": 0.0048,
            "grip_center_root_m": [0.13648802515004896, 0.07162676678012178, 0.025569974587941836],
            "grip_normal_root": [0.0831279716728119, -0.9950249745219021, 0.0549093835628487],
            "surface_points_root_m": [
                [0.1362885187410796, 0.07401481806426741, 0.025438192544991696],
                [0.1366875315590183, 0.06923871549597614, 0.025701756630891973],
            ],
        },
        "rotation_xyzw": [0.17490933357081812, 0.8170581262792321, -0.08793959584326423, -0.5423000744917947],
        "normal": [-0.26899182491253515, -0.8845385200126215, 0.38109710671179997],
        "initial_offset": 0.0025,
    },
]


class H1JacketScene:
    """Prescribe original hand articulations; every jacket vertex stays free."""

    def __init__(self, *, device=None, substeps=10, iterations=100, settle_only=False, **kwargs):
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 2 or substeps > 10 or substeps % 2:
            raise ValueError("substeps must be an even integer from 2 through 10")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self.scene_id = "j3_h1_jacket"
        self.title = "J3 | Native jacket and two original H1 hands"
        self.description = (
            "Native Style3D jacket at its previous scale, with two original Unitree H1 hands. "
            "The upper garment starts over a visible table edge and settles freely before contact acquisition. "
            "Original coupled finger joints and detached wrist roots are prescribed; all cloth vertices remain dynamic."
        )
        self.asset_ids = ["h1_jacket_native", "unitree_h1_original_left_hand", "unitree_h1_original_right_hand"]
        self.frame_dt, self.substeps, self.iterations = 1 / 30, substeps, iterations
        self.dt = self.frame_dt / self.substeps
        self.time, self.duration = 0.0, 21.5
        self.table_top, self.radius = 0.25, 0.0015
        self.table_center, self.table_half = (0.0, 0.3, 0.12), (0.75, 0.70, 0.13)
        self.camera_pos, self.camera_target = (1.65, -2.10, 1.65), (0, -0.20, 0.65)
        self.camera_fov = 48.0
        self.mesh = load_native_jacket()
        mesh = self.mesh
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        self.scale = 0.6 / max(hi - lo)
        b = newton.ModelBuilder(gravity=(0, 0, -9.81))
        b.default_shape_cfg.ke, b.default_shape_cfg.kd = 50000.0, 0.1
        b.default_shape_cfg.mu, b.default_shape_cfg.gap = 1.5, 0.005
        b.default_shape_cfg.margin = 0.001
        b.default_shape_cfg.configure_sdf(force_sdf=True, max_resolution=64)
        self.authored_rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi) * wp.quat_from_axis_angle(
            wp.vec3(1, 0, 0), -np.pi / 2
        )
        self.authored_position = (
            float((lo[0] + hi[0]) * 0.5 * self.scale),
            0.05 + float(lo[2] * self.scale),
            0.258 + float(hi[1] * self.scale),
        )
        b.add_cloth_mesh(
            vertices=mesh.vertices.tolist(),
            indices=mesh.triangles.ravel().tolist(),
            pos=self.authored_position,
            rot=self.authored_rotation,
            vel=(0, 0, 0),
            scale=self.scale,
            density=0.22,
            particle_radius=self.radius,
            tri_ke=1800.0,
            tri_ka=1800.0,
            tri_kd=0.025,
            edge_ke=0.08,
            edge_kd=0.0005,
            color=tuple(mesh.metadata["color_rgb"]),
        )
        table_cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=0.35, ke=50000, kd=0.1, gap=0.005, margin=0)
        self.table_shape = b.add_shape_box(
            -1,
            xform=wp.transform(self.table_center, wp.quat_identity()),
            hx=self.table_half[0],
            hy=self.table_half[1],
            hz=self.table_half[2],
            cfg=table_cfg,
            color=(0.48, 0.51, 0.55),
            label="Original finite jacket support table",
        )
        self.hands = [
            add_h1_hand(b, side=recipe["side"], root_pos=(2 + i, 2, 2), controls=recipe["open"]["controls"])
            for i, recipe in enumerate(_PINCH_RECIPES)
        ]
        b.add_ground_plane()
        b.color()
        self.model = b.finalize(device=device)
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 50000.0, 0.1, 0.35
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.002,
            particle_self_contact_gap=0.001,
            particle_rest_shape_contact_exclusion_radius=0.005,
            particle_topological_contact_filter_threshold=1,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=64,
            friction_epsilon=0.001,
            integrate_with_external_rigid_solver=True,
            rigid_compliant_alm=True,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=0.005, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self.state0, self.state1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.strain = ClothStrain(self.model)
        self.authored_state = self.strain.sample(self.state.particle_q.numpy())
        self.drivers = [
            H1HandDriver(
                self.model,
                hand,
                times=[0, 30],
                positions=[(2 + i, 2, 2)] * 2,
                rotations=(0, 0, 0, 1),
                controls=[_PINCH_RECIPES[i]["open"]["controls"]] * 2,
            )
            for i, hand in enumerate(self.hands)
        ]
        for driver in self.drivers:
            driver.update(self.clock, self.state0, self.state1)
        self.settle_history, self.history, self.hand_audits = [], [], []
        state_dir = os.environ.get("NEWTON_JACKET_STATE_DIR")
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.state_dir / "rest-mesh.npz",
                positions=self.model.particle_q.numpy(),
                triangles=self.model.tri_indices.numpy(),
                triangle_rest_inverse=self.model.tri_poses.numpy(),
                particle_mass=self.model.particle_mass.numpy(),
            )
        self._capture()
        for frame in range(60):
            wp.capture_launch(self.graph)
            if frame % 3 == 2:
                q = self.state.particle_q.numpy()
                if not np.isfinite(q).all():
                    raise ValueError("Native jacket free settling produced a non-finite state")
                self.settle_history.append({"time_s": (frame + 1) * self.frame_dt, **self.strain.sample(q)})
        q = self.state.particle_q.numpy()
        self.initial = self.strain.sample(q)
        if self.state_dir:
            np.savez_compressed(self.state_dir / "settled-state.npz", positions=q, triangles=mesh.triangles)
            (self.state_dir / "settle-metrics.json").write_text(json.dumps(self.settle_history, indent=2))
        self.clock.zero_()
        self.grasp = []
        if settle_only:
            self.description += " Diagnostic mode: parked hands and free settling only."
            self._capture()
            return
        self._configure_grips(q)
        self._capture()

    @property
    def state(self):
        return self.state0

    @property
    def phase(self):
        for end, name in [
            (1.5, "close"),
            (2.0, "acquire"),
            (3.0, "proof lift"),
            (3.5, "proof hold"),
            (6.0, "lift"),
            (7.5, "carry"),
            (9.5, "hold"),
            (12.5, "place"),
            (14.0, "open thumb"),
            (18.5, "straighten index"),
            (20.0, "withdraw"),
            (22.0, "released observation"),
        ]:
            if self.time < end:
                return name
        return "done"

    def _configure_grips(self, q):
        # The two material identities were selected from the freely settled
        # exposed collar/shoulder region. Retarget safe frames on this run.
        triangles = self.mesh.triangles
        adjacent = [set() for _ in q]
        for a, b, c in triangles:
            adjacent[a].update((int(b), int(c)))
            adjacent[b].update((int(a), int(c)))
            adjacent[c].update((int(a), int(b)))
        face_normals = np.cross(q[triangles[:, 1]] - q[triangles[:, 0]], q[triangles[:, 2]] - q[triangles[:, 0]])
        vertex_normals = np.zeros_like(q)
        for k in range(3):
            np.add.at(vertex_normals, triangles[:, k], face_normals)
        vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1)[:, None], 1e-12)
        self.times = [0, 1.5, 2.0, 3.0, 3.5, 6.0, 7.5, 9.5, 12.5, 14.0, 18.5, 20.0, 21.5]
        self.drivers = []
        for hand, recipe in zip(self.hands, _PINCH_RECIPES, strict=True):
            index = recipe["vertex"]
            connected = {index}
            connected.update(adjacent[index])
            for j in list(connected):
                connected.update(adjacent[j])
            connected = np.asarray(sorted(connected))
            normal = vertex_normals[connected].mean(0)
            normal /= np.linalg.norm(normal)
            if normal[1] > 0:
                normal = -normal
            normal = normal.astype(float)
            normal /= np.linalg.norm(normal)
            observed_normal = normal.copy()
            # A small change in a corrugated patch normal can swing a whole
            # hand into the table. Keep the measured safe wrist frame; update
            # the material position and audit the complete new settled state.
            rotation = wp.quat(*recipe["rotation_xyzw"])
            matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3).astype(float)
            normal = np.asarray(recipe["normal"], dtype=float)
            target = q[index].astype(float)
            root = target - matrix @ np.asarray(recipe["closed"]["grip_center_root_m"])
            lifted = root + np.array([0, 0, 1.05])
            carried = lifted + np.array([0, 0.25, 0])
            placed = carried - [0, 0, target[2] + 1.05 - 0.42]
            withdrawn = placed + np.array([0, -0.35, 0.10])
            positions = [
                root + normal * recipe["initial_offset"],
                root,
                root,
                root + np.array([0, 0, 0.05]),
                root + np.array([0, 0, 0.05]),
                lifted,
                carried,
                carried,
                placed,
                placed,
                placed,
                withdrawn,
                withdrawn,
            ]
            opened, closed = recipe["open"]["controls"], recipe["closed"]["controls"]
            straight = list(opened)
            straight[2] = 0.0
            controls = [opened] + [closed] * 8 + [opened, straight, straight, straight]
            driver = H1HandDriver(
                self.model, hand, times=self.times, positions=positions, rotations=rotation, controls=controls
            )
            self.drivers.append(driver)
            driver.update(self.clock, self.state0, self.state1)
            audit = driver.geometry_audit(
                self.state,
                radii=self.radius,
                ground_z=0,
                box_center=self.table_center,
                box_half=self.table_half,
                full_surface_contact=True,
            )
            if audit["minimum_shell_to_robot_distance_m"] < 0.001 - 1e-6 or audit["vertices_inside_robot_solids"]:
                raise ValueError(
                    f"Initial {hand.side} H1 hand intersects native jacket shell plus hand margin: {audit}"
                )
            if audit["minimum_conservative_box_separation_m"] < 0.002:
                raise ValueError(f"Initial {hand.side} H1 hand lacks finite table clearance: {audit}")
            delta = q - target
            normal_coordinates = delta @ normal
            tangent_radius = np.sqrt(np.maximum(np.sum(delta * delta, axis=1) - normal_coordinates**2, 0))
            patch = (tangent_radius < 0.008) & (np.abs(normal_coordinates) < 0.012)
            self.grasp.append(
                {
                    "side": hand.side,
                    "vertex": index,
                    "source_vertex_m": self.mesh.vertices[index].tolist(),
                    "target_after_settle_m": target.tolist(),
                    "connected_patch_vertices": connected.tolist(),
                    "connected_patch_normal": observed_normal.tolist(),
                    "command_contact_normal": normal.tolist(),
                    "normal_mismatch_degrees": float(np.degrees(np.arccos(np.clip(observed_normal @ normal, -1, 1)))),
                    "hand_roll_degrees": recipe["roll"],
                    "hand_rotation_xyzw": [float(v) for v in rotation],
                    "initial_open_physical_gap_m": recipe["open"]["physical_mesh_gap_m"],
                    "closed_physical_gap_m": recipe["closed"]["physical_mesh_gap_m"],
                    "closed_local_contact_center_m": recipe["closed"]["grip_center_root_m"],
                    "open_controls_rad": opened,
                    "closed_controls_rad": closed,
                    "nominal_total_shell_plus_margin_compression_m": 0.0002,
                    "initial_outward_hand_offset_m": recipe["initial_offset"],
                    "pinch_disk_radius_m": 0.008,
                    "pinch_disk_vertices": int(patch.sum()),
                    "pinch_disk_material_normal_span_m": float(np.ptp(normal_coordinates[patch])),
                    "initial_geometry_audit": audit,
                    "root_positions_m": [p.tolist() for p in positions],
                    "times_s": self.times,
                    "acquisition_scope": "Starts in an audited open pose after free settle; only original coupled thumb closure and the recorded inward wrist motion attempt cloth acquisition. Parking-to-open transfer is outside the test.",
                }
            )
        # Sample the entire planned hand path against the unchanged finite table.
        self.path_audits = []
        pair_gaps = []
        for time in np.linspace(0, self.duration, 216):
            self.clock.assign([time])
            clouds = []
            for driver in self.drivers:
                driver.update(self.clock, self.state0, self.state1)
                audit = driver.geometry_audit(
                    self.state,
                    points=np.empty((0, 3)),
                    ground_z=0,
                    box_center=self.table_center,
                    box_half=self.table_half,
                )
                if (
                    audit["minimum_conservative_box_separation_m"] < 0.002
                    or audit["hand_ground_clearance_including_margin_m"] < 0.002
                ):
                    raise ValueError(f"H1 planned path fails table/ground clearance at {time}: {audit}")
                self.path_audits.append({"time_s": float(time), "side": driver.spec.side, **audit})
                clouds.append(np.concatenate([v for v, _ in driver.world_collision_meshes(self.state).values()]))
            gap = float(clouds[1][:, 0].min() - clouds[0][:, 0].max() - 0.002)
            if gap < 0:
                raise ValueError("Planned H1 pair lacks conservative X separation including margins")
            pair_gaps.append(gap)
        self.minimum_pair_x_gap = min(pair_gaps)
        self.clock.zero_()
        for driver in self.drivers:
            driver.update(self.clock, self.state0, self.state1)

    def _capture(self):
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        self.solver.rebuild_bvh(self.state0)
        for _ in range(self.substeps):
            for driver in self.drivers:
                driver.update(self.clock, self.state0, self.state1, dt=self.dt)
            self.state0.clear_forces()
            self.state1.clear_forces()
            self.pipeline.collide(self.state0, self.contacts)
            self.solver.step(self.state0, self.state1, self.control, self.contacts, self.dt)
            self.state0, self.state1 = self.state1, self.state0
            wp.launch(tick, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def step(self):
        wp.capture_launch(self.graph)
        self.time += self.frame_dt
        q = self.state.particle_q.numpy()
        count = min(int(self.contacts.soft_contact_count.numpy()[0]), len(self.contacts.soft_contact_shape))
        shapes = self.contacts.soft_contact_shape.numpy()[:count]
        contact_counts = {
            hand.side: {
                "all_hand": int(np.isin(shapes, hand.shape_ids).sum()),
                "thumb_distal": int((shapes == hand.shape_ids[5]).sum()),
                "index_distal": int((shapes == hand.shape_ids[7]).sum()),
            }
            for hand in self.hands
        }
        self.history.append(
            {
                "time_s": self.time,
                "phase": self.phase,
                "detected_rigid_soft_contacts": contact_counts,
                **self.strain.sample(q),
            }
        )
        if round(self.time / self.frame_dt) % 15 == 0:
            self.hand_audits.append(
                {
                    "time_s": self.time,
                    "hands": [
                        {
                            "side": d.spec.side,
                            **d.geometry_audit(
                                self.state,
                                radii=self.radius,
                                ground_z=0,
                                box_center=self.table_center,
                                box_half=self.table_half,
                                full_surface_contact=True,
                            ),
                        }
                        for d in self.drivers
                    ],
                }
            )
        if self.state_dir and round(self.time / self.frame_dt) % 15 == 0:
            np.savez_compressed(self.state_dir / f"state-{self.time:.2f}s.npz", positions=q)

    def metrics(self):
        hold = [row for row in self.history if 7.5 - 1e-9 <= row["time_s"] <= 9.5 + 1e-9]
        return {
            "physics_parameters": {
                "frame_dt_s": self.frame_dt,
                "substeps": self.substeps,
                "substep_dt_s": self.dt,
                "iterations": self.iterations,
                "density_kg_m2": 0.22,
                "membrane_stretch_coefficient_N_m": 1800.0,
                "membrane_area_coefficient_N_m": 1800.0,
                "membrane_damping": 0.025,
                "bending_coefficient_N": 0.08,
                "bending_damping": 0.0005,
                "bending_rest_angles": "authored 3D source geometry",
                "particle_radius_m": self.radius,
                "self_contact_enabled": True,
                "self_contact_margin_m": 0.002,
                "self_contact_gap_m": 0.001,
                "self_contact_rest_exclusion_m": 0.005,
                "self_contact_topological_rings": 1,
                "friction_robot": 1.5,
                "friction_table": 0.35,
                "friction_soft_self": 0.35,
                "contact_stiffness_N_m": 50000.0,
                "contact_damping": 0.1,
                "rigid_soft_detection_gap_m": 0.005,
                "robot_shape_margin_m": 0.001,
                "ground_shape_margin_m": 0.001,
                "table_shape_margin_m": 0.0,
                "full_surface_rigid_soft_contact": True,
                "robot_mesh_sdf_max_resolution": 64,
                "mesh_vertices": len(self.mesh.vertices),
                "mesh_triangles": len(self.mesh.triangles),
                "mesh_scale": self.scale,
            },
            "boundary_conditions": "Two original H1 detached wrist roots and coupled finger joints prescribed; every cloth vertex has positive mass. No attachments, pins or hidden supports.",
            "pinned_particle_count": int(np.count_nonzero(self.model.particle_inv_mass.numpy() == 0)),
            "free_particle_count": int(np.count_nonzero(self.model.particle_inv_mass.numpy() > 0)),
            "zero_mass_particle_count": int(np.count_nonzero(self.model.particle_inv_mass.numpy() == 0)),
            "source_asset": self.mesh.metadata,
            "robot_assets": [hand.provenance for hand in self.hands],
            "robot_internal_geometry_validation": {
                "original_collision_shapes_per_hand": 14,
                "original_moving_joints_per_hand": 12,
                "all_nonadjacent_original_meshes_sweep_inside_vertices": 0,
                "minimum_left_physical_mesh_separation_m": 0.0023086839355528355,
                "minimum_right_physical_mesh_separation_m": 0.0013412347761914134,
                "sweep_samples": 41,
                "audit_reference": "smoke-artifacts/h1-hand-validation/safe-h1-full-sweep-audit.json",
                "exact_constant_thumb_uncurl_audit_reference": "smoke-artifacts/h1-hand-validation/safe-h1-open-index-sweeps.json",
                "scope": "Normal-signed original mesh vertex sweep across closure, reopening and index straightening. Same-body and direct-parent pairs excluded. Original palm/pinky topology limits sign interpretation; source geometry is unmodified.",
                "self_contact_mode": "Prescribed hand bodies have no internal rigid collision response; sampled original nonadjacent meshes are physically separated. Numerical margins can overlap near the original basal thumb/palm interface.",
                "supersedes": "h1-jacket-diagnostics/invalid-interdigit-full-v1",
            },
            "cloth_scale": self.scale,
            "cloth_mass_kg": float(self.strain.mass.sum()),
            "initial_rigid_placement": {
                "position_m": self.authored_position,
                "rotation_xyzw": [float(x) for x in self.authored_rotation],
                "reason": "Upper garment overhangs original finite table front for physical underside access",
            },
            "table_center_m": self.table_center,
            "table_half_extent_m": self.table_half,
            "grasp_recipe": self.grasp,
            "authored_initial_strain": self.authored_state,
            "free_settle_trajectory": self.settle_history,
            "post_settle_initial_strain": self.initial,
            "hold_window_s": [7.5, 9.5],
            "hold_min_shell_clearance_m": min(
                (r["min_z_m"] - self.table_top - self.radius for r in hold), default=None
            ),
            "hold_max_principal_stretch": max((r["principal_stretch_max"] for r in hold), default=None),
            "hold_p95_principal_stretch_max": max((r["principal_stretch_p95"] for r in hold), default=None),
            "hold_center_rise_m": min(
                (r["center_of_mass_m"][2] - self.initial["center_of_mass_m"][2] for r in hold), default=None
            ),
            "contact_record_semantics": "Raw detected full-surface rigid-soft contacts include candidates within detection gap; counts alone do not measure force or prove retention.",
            "trajectory": self.history,
            "robot_geometry_audits": self.hand_audits,
            "planned_path_audits": getattr(self, "path_audits", []),
            "minimum_planned_hand_pair_x_gap_including_margins_m": getattr(self, "minimum_pair_x_gap", None),
            "qualification": "Smoke only; observed contact, strain and release diagnostics do not certify physical fidelity or actuator capacity.",
        }


def build_scene(scene_id="j3_h1_jacket", **kwargs):
    if scene_id not in SCENE_IDS:
        raise ValueError(f"Unknown H1 jacket scene {scene_id}")
    return H1JacketScene(**kwargs)
