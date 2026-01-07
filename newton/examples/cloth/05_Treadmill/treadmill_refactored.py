# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Treadmill Cloth Simulation (Refactored)
#
# A rolled cloth mesh that unrolls as the inner seam rotates.
# Uses the base Simulator class from M01_Simulator.py.
#
###########################################################################

import math
import sys
from datetime import datetime
from os.path import join

import numpy as np
import polyscope as ps
import warp as wp

from newton import ParticleFlags

sys.path.append("..")
from newton.examples.cloth.M01_Simulator import Simulator


@wp.kernel
def apply_rotation(
    angular_speed: float,
    dt: float,
    t: float,
    q0: wp.array(dtype=wp.vec3),
    fixed: wp.array(dtype=wp.int64),
    q1: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    particle_index = fixed[i]
    c0 = math.cos(-angular_speed * (t - dt))
    s0 = math.sin(-angular_speed * (t - dt))
    c1 = math.cos(angular_speed * t)
    s1 = math.sin(angular_speed * t)
    x0, y0, z0 = q0[particle_index][0], q0[particle_index][1], q0[particle_index][2]
    q0[particle_index][0] = c0 * x0 + s0 * z0
    q0[particle_index][1] = y0
    q0[particle_index][2] = -s0 * x0 + c0 * z0
    x0, y0, z0 = q0[particle_index][0], q0[particle_index][1], q0[particle_index][2]
    q0[particle_index][0] = c1 * x0 + s1 * z0
    q0[particle_index][1] = y0
    q0[particle_index][2] = -s1 * x0 + c1 * z0
    q1[particle_index] = q0[particle_index]


@wp.kernel
def rotate_cylinder(
    angular_speed: float,
    dt: float,
    t: float,
    center_x: float,
    center_z: float,
    q0: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int64),
    q1: wp.array(dtype=wp.vec3),
):
    """Rotate cylinder vertices around their center axis."""
    i = wp.tid()
    particle_index = indices[i]
    c0 = math.cos(-angular_speed * (t - dt))
    s0 = math.sin(-angular_speed * (t - dt))
    c1 = math.cos(angular_speed * t)
    s1 = math.sin(angular_speed * t)

    # Translate to center, rotate, translate back
    x0 = q0[particle_index][0] - center_x
    y0 = q0[particle_index][1]
    z0 = q0[particle_index][2] - center_z

    # Undo previous rotation
    rx = c0 * x0 + s0 * z0
    rz = -s0 * x0 + c0 * z0

    # Apply new rotation
    x1 = c1 * rx + s1 * rz
    z1 = -s1 * rx + c1 * rz

    # Translate back
    q0[particle_index][0] = x1 + center_x
    q0[particle_index][1] = y0
    q0[particle_index][2] = z1 + center_z
    q1[particle_index] = q0[particle_index]


def rolled_cloth_mesh(
    length=500.0,
    width=100.0,
    nu=200,
    nv=15,
    inner_radius=10.0,
    thickness=0.4,
    target_x=None,
    target_y=None,
    extension_segments=10,
):
    """
    Create a rolled cloth mesh with optional extension to a target point.

    Args:
        target_x, target_y: Target position in local coords (before rotation).
                           If provided, extension goes directly to this point.
        extension_segments: Number of rows for extension
    """
    verts = []
    faces = []

    # Create the spiral part
    for i in range(nu):
        u = length * i / (nu - 1)
        theta = u / inner_radius
        r = inner_radius + (thickness / (2.0 * np.pi)) * theta

        for j in range(nv):
            v = width * j / (nv - 1)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = v
            verts.append([x, y, z])

    # Get outer edge position
    last_theta = length / inner_radius
    last_r = inner_radius + (thickness / (2.0 * np.pi)) * last_theta
    outer_x = last_r * np.cos(last_theta)
    outer_y = last_r * np.sin(last_theta)

    # Add extension rows if target is provided
    ext_rows = 0
    if target_x is not None and target_y is not None:
        # Direction from outer edge to target
        dx = target_x - outer_x
        dy = target_y - outer_y
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > 1.0:
            ext_rows = extension_segments
            for i in range(1, ext_rows + 1):
                t = i / ext_rows
                ext_x = outer_x + t * dx
                ext_y = outer_y + t * dy

                for j in range(nv):
                    v = width * j / (nv - 1)
                    verts.append([ext_x, ext_y, v])

    total_rows = nu + ext_rows

    def idx(i, j):
        return i * nv + j

    for i in range(total_rows - 1):
        for j in range(nv - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            faces.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32), nu, ext_rows


def cylinder_mesh(radius=9.5, height=120.0, segments=64, caps=False):
    """Create a cylinder mesh.

    Args:
        radius: Cylinder radius
        height: Cylinder height
        segments: Number of segments around the circumference
        caps: If True, add top and bottom caps; if False, side walls only
    """
    verts = []
    faces = []

    y_bottom = -height * 0.5
    y_top = height * 0.5

    if caps:
        # Efficient vertex layout for capped cylinder
        # Bottom ring, then top ring
        for i in range(segments):
            t = 2 * math.pi * i / segments
            x, z = radius * math.cos(t), radius * math.sin(t)
            verts.append([x, y_bottom, z])
        for i in range(segments):
            t = 2 * math.pi * i / segments
            x, z = radius * math.cos(t), radius * math.sin(t)
            verts.append([x, y_top, z])

        # Side faces
        for i in range(segments):
            i_next = (i + 1) % segments
            b0, b1 = i, i_next
            t0, t1 = segments + i, segments + i_next
            faces.append([b0, b1, t1])
            faces.append([b0, t1, t0])

        # Cap centers
        bottom_center = len(verts)
        verts.append([0.0, y_bottom, 0.0])
        top_center = len(verts)
        verts.append([0.0, y_top, 0.0])

        # Bottom cap (fan)
        for i in range(segments):
            i_next = (i + 1) % segments
            faces.append([bottom_center, i_next, i])

        # Top cap (fan)
        for i in range(segments):
            i_next = (i + 1) % segments
            faces.append([top_center, segments + i, segments + i_next])
    else:
        # Original layout: 4 verts per segment
        for i in range(segments):
            t0 = 2 * math.pi * i / segments
            t1 = 2 * math.pi * (i + 1) / segments

            x0, z0 = radius * math.cos(t0), radius * math.sin(t0)
            x1, z1 = radius * math.cos(t1), radius * math.sin(t1)

            base = len(verts)

            verts += [
                [x0, y_bottom, z0],
                [x1, y_bottom, z1],
                [x1, y_top, z1],
                [x0, y_top, z0],
            ]

            faces += [
                [base + 0, base + 1, base + 2],
                [base + 0, base + 2, base + 3],
            ]

    return (
        np.array(verts, np.float32),
        np.array(faces, np.int32),
    )


def save_ply(verts, faces, filepath):
    """Save a mesh to PLY format."""
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    with open(filepath, "w") as f:
        f.write("\n".join(header) + "\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"{len(face)} {' '.join(map(str, face))}\n")


class TreadmillSimulator(Simulator):
    """
    Treadmill simulation using the refactored Simulator base class.
    """

    def custom_init(self):
        """Add rolled cloth and cylinder meshes."""
        # Cloth parameters
        self.cloth_thickness = self.config.get("cloth_thickness", 0.4)
        self.nv = 15  # vertices per row

        # Cylinder properties - cylinders are now further apart
        self.cyl1_radius = 9.9
        self.cyl2_radius = 14.9
        self.cyl1_center = (-27.2, 7.4)  # (X, Z)
        self.cyl2_center = (40.0, 0.0)  # (X, Z) - moved further right

        # Cloth position offset
        cloth_offset_x = self.cyl1_center[0]  # -27.2
        cloth_offset_z = self.cyl1_center[1]  # 7.4

        # Calculate target position for extension (cylinder 2's left side)
        # in LOCAL coordinates (before 90° rotation around X)
        # World target: (cyl2_x - radius - offset, cyl2_z)
        # Local coords: local_x = world_x - cloth_offset_x, local_y = world_z - cloth_offset_z
        self_contact_radius = self.config.get("self_contact_radius", 0.4)
        attach_offset = self.cloth_thickness + self_contact_radius
        target_world_x = self.cyl2_center[0] - self.cyl2_radius - attach_offset
        target_world_z = self.cyl2_center[1]

        target_local_x = target_world_x - cloth_offset_x
        target_local_y = target_world_z - cloth_offset_z

        # Cloth spiral parameters (more length = more wraps)
        cloth_length = self.config.get("cloth_length", 800.0)  # Increased for more wraps
        cloth_nu = self.config.get("cloth_nu", 300)  # More rows for denser mesh

        # Generate cloth mesh with extension going directly to target
        self.cloth_verts, self.cloth_faces, self.spiral_rows, self.ext_rows = rolled_cloth_mesh(
            length=cloth_length,
            nu=cloth_nu,
            thickness=self.cloth_thickness,
            target_x=target_local_x,
            target_y=target_local_y,
            extension_segments=20,
        )
        self.cloth_faces_flat = self.cloth_faces.reshape(-1)
        self.num_cloth_verts = len(self.cloth_verts)
        self.total_rows = self.spiral_rows + self.ext_rows

        # Generate cylinder meshes
        cylinder_caps = self.config.get("cylinder_caps", False)
        self.cyl1_verts, self.cyl1_faces = cylinder_mesh(radius=self.cyl1_radius, caps=cylinder_caps)
        self.cyl2_verts, self.cyl2_faces = cylinder_mesh(radius=self.cyl2_radius, caps=cylinder_caps)
        self.num_cyl1_verts = len(self.cyl1_verts)
        self.num_cyl2_verts = len(self.cyl2_verts)

        # Add cloth mesh
        self.builder.add_cloth_mesh(
            pos=wp.vec3(-27.2, 100.0, 7.4),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2),
            scale=1.0,
            vertices=self.cloth_verts,
            indices=self.cloth_faces_flat,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
            particle_radius=0.5,
        )

        # Add first cylinder
        self.builder.add_cloth_mesh(
            pos=wp.vec3(self.cyl1_center[0], 50.0, self.cyl1_center[1]),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
            scale=1.0,
            vertices=self.cyl1_verts,
            indices=self.cyl1_faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
        )

        # Add second cylinder
        self.builder.add_cloth_mesh(
            pos=wp.vec3(self.cyl2_center[0], 50.0, self.cyl2_center[1]),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
            scale=1.0,
            vertices=self.cyl2_verts,
            indices=self.cyl2_faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
        )

        # Add ground plane
        self.builder.add_ground_plane()

        # Rotation parameters - match linear velocity at surface
        # v = omega * r, so for same v: omega2 = omega1 * r1 / r2
        self.angular_speed = -4 * np.pi  # rad/sec (base speed for cloth)
        linear_velocity = abs(self.angular_speed) * self.cyl1_radius
        self.angular_speed_cyl1 = -linear_velocity / self.cyl1_radius  # = angular_speed
        self.angular_speed_cyl2 = -linear_velocity / self.cyl2_radius  # slower due to larger radius
        self.spin_duration = 15.0  # seconds

    def custom_finalize(self):
        """Fix outer edge of cloth to cylinder 2 and set up cylinder rotation."""
        # Outer edge = last row (end of extension), attached to cylinder 2's leftmost line
        last_row = self.total_rows - 1
        self.fixed_point_indices = [last_row * self.nv + i for i in range(self.nv)]

        # Position the outer edge at cylinder 2's leftmost line
        # This avoids penetration by placing cloth on the surface facing the spiral
        # Offset = cloth thickness + self_contact_radius to allow air gap
        positions = self.model.particle_q.numpy()
        self_contact_radius = self.config.get("self_contact_radius", 0.4)
        attach_offset = self.cloth_thickness + self_contact_radius
        left_x = self.cyl2_center[0] - self.cyl2_radius - attach_offset
        for idx in self.fixed_point_indices:
            positions[idx][0] = left_x
            positions[idx][2] = self.cyl2_center[1]  # Align Z with cylinder center
        self.model.particle_q = wp.array(positions, dtype=wp.vec3)

        # Also update state_0 positions
        state_positions = self.state_0.particle_q.numpy()
        for idx in self.fixed_point_indices:
            state_positions[idx][0] = left_x
            state_positions[idx][2] = self.cyl2_center[1]
        self.state_0.particle_q = wp.array(state_positions, dtype=wp.vec3)

        # Fix the outer edge vertices (kinematic, attached to cylinder 2)
        if len(self.fixed_point_indices):
            flags = self.model.particle_flags.numpy()
            for fixed_vertex_id in self.fixed_point_indices:
                flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE
            self.model.particle_flags = wp.array(flags)

        self.fixed_point_indices = wp.array(self.fixed_point_indices)

        # Store cylinder vertex indices for rotation
        cyl1_start = self.num_cloth_verts
        cyl1_end = cyl1_start + self.num_cyl1_verts
        cyl2_start = cyl1_end
        cyl2_end = cyl2_start + self.num_cyl2_verts

        self.cyl1_indices = wp.array(list(range(cyl1_start, cyl1_end)), dtype=wp.int64)
        self.cyl2_indices = wp.array(list(range(cyl2_start, cyl2_end)), dtype=wp.int64)

        # Make all cylinder vertices static (kinematic, not simulated)
        flags = self.model.particle_flags.numpy()
        for id in range(self.num_cloth_verts, len(self.builder.particle_q)):
            flags[id] = flags[id] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        # Disable collision detection (matches original)
        self.contacts = None

    def init_polyscope(self):
        """Override to match original polyscope initialization."""
        ps.set_ground_plane_height(0.0)
        ps.init()
        # Note: NOT calling ps.set_up_dir() to match original

        # Register keyboard callback for pause/unpause
        ps.set_user_callback(self.keyboard_callback)

    def run_step(self):
        """Execute one frame of simulation with rotation applied."""
        self.solver.rebuild_bvh(self.state_0)

        for _ in range(self.num_substeps):
            self.state_0.clear_forces()

            # Apply rotation during spin duration
            if self.sim_time < self.spin_duration:
                # Rotate cloth outer edge (attached to cylinder 2's left side)
                wp.launch(
                    kernel=rotate_cylinder,
                    dim=len(self.fixed_point_indices),
                    inputs=[
                        self.angular_speed_cyl2,  # Same speed as cylinder 2
                        self.dt,
                        self.sim_time,
                        self.cyl2_center[0],  # Rotate around cylinder 2's center
                        self.cyl2_center[1],
                        self.state_0.particle_q,
                        self.fixed_point_indices,
                        self.state_1.particle_q,
                    ],
                )

                # Rotate cylinder 1 (around its center, matching surface velocity)
                wp.launch(
                    kernel=rotate_cylinder,
                    dim=len(self.cyl1_indices),
                    inputs=[
                        self.angular_speed_cyl1,
                        self.dt,
                        self.sim_time,
                        self.cyl1_center[0],
                        self.cyl1_center[1],
                        self.state_0.particle_q,
                        self.cyl1_indices,
                        self.state_1.particle_q,
                    ],
                )

                # Rotate cylinder 2 (around its center, slower due to larger radius)
                wp.launch(
                    kernel=rotate_cylinder,
                    dim=len(self.cyl2_indices),
                    inputs=[
                        self.angular_speed_cyl2,
                        self.dt,
                        self.sim_time,
                        self.cyl2_center[0],
                        self.cyl2_center[1],
                        self.state_0.particle_q,
                        self.cyl2_indices,
                        self.state_1.particle_q,
                    ],
                )

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def setup_polyscope_meshes(self):
        """Set up separate meshes for cloth and cylinders."""
        if not self.do_rendering:
            return

        all_verts = self.model.particle_q.numpy()

        # Cloth mesh
        cloth_start = 0
        cloth_end = self.num_cloth_verts
        self.register_ps_mesh(
            name="RolledCloth",
            vertices=all_verts[cloth_start:cloth_end],
            faces=self.cloth_faces,
            vertex_indices=slice(cloth_start, cloth_end),
        )

        # Cylinder 1 mesh
        cyl1_start = cloth_end
        cyl1_end = cyl1_start + self.num_cyl1_verts
        self.register_ps_mesh(
            name="Cylinder1",
            vertices=all_verts[cyl1_start:cyl1_end],
            faces=self.cyl1_faces,
            vertex_indices=slice(cyl1_start, cyl1_end),
        )

        # Cylinder 2 mesh
        cyl2_start = cyl1_end
        cyl2_end = cyl2_start + self.num_cyl2_verts
        self.register_ps_mesh(
            name="Cylinder2",
            vertices=all_verts[cyl2_start:cyl2_end],
            faces=self.cyl2_faces,
            vertex_indices=slice(cyl2_start, cyl2_end),
        )

        # Wider view for extended cloth between cylinders
        ps.look_at((10.0, 300.0, 50.0), (10.0, 0.0, 0.0))

    def save_initial_meshes(self):
        """Save initial meshes for cloth and cylinders."""
        if self.output_path is None:
            return

        all_verts = self.model.particle_q.numpy()

        # Save cloth
        cloth_verts = all_verts[: self.num_cloth_verts]
        save_ply(cloth_verts, self.cloth_faces, join(self.output_path, "initial_cloth.ply"))

        # Save cylinder 1
        cyl1_start = self.num_cloth_verts
        cyl1_end = cyl1_start + self.num_cyl1_verts
        cyl1_verts = all_verts[cyl1_start:cyl1_end]
        save_ply(cyl1_verts, self.cyl1_faces, join(self.output_path, "initial_cylinder1.ply"))

        # Save cylinder 2
        cyl2_start = cyl1_end
        cyl2_end = cyl2_start + self.num_cyl2_verts
        cyl2_verts = all_verts[cyl2_start:cyl2_end]
        save_ply(cyl2_verts, self.cyl2_faces, join(self.output_path, "initial_cylinder2.ply"))

        print(f"Initial meshes saved to: {self.output_path}")


if __name__ == "__main__":
    # Configuration
    truncation_mode = 0
    iterations = 10
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_treadmill_trunc{truncation_mode}_iter{iterations}_{timestamp}"

    config = {
        "name": "treadmill_cloth",
        "fps": 60,
        "sim_substeps": 10,
        "sim_num_frames": 1000,
        "iterations": iterations,
        "truncation_mode": truncation_mode,
        "up_axis": "y",
        "gravity": 0,  # No gravity in this simulation
        "cloth_length": 800.0,  # Length of cloth spiral (more = more wraps)
        "cloth_nu": 300,  # Number of rows along cloth length
        "cloth_thickness": 0.4,  # Thickness of rolled cloth mesh
        "cylinder_caps": False,  # Whether to add top/bottom caps to cylinders
        "handle_self_contact": True,
        "self_contact_radius": 0.4,  # attach_offset = cloth_thickness + self_contact_radius
        "self_contact_margin": 0.6,
        "topological_contact_filter_threshold": 1,
        "soft_contact_ke": 1.0e5,
        "soft_contact_kd": 1.0e-5,
        "soft_contact_mu": 0.1,
        "output_path": output_dir,
        "output_ext": "npy",
        "write_output": False,
        "write_video": False,
        "do_rendering": True,
        "has_ground": False,
        "is_initially_paused": False,
    }

    sim = TreadmillSimulator(config)
    sim.finalize()
    sim.simulate()
