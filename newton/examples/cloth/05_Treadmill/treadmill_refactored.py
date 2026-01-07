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
):
    verts = []
    faces = []

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

    def idx(i, j):
        return i * nv + j

    for i in range(nu - 1):
        for j in range(nv - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            faces.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def cylinder_mesh(radius=9.5, height=120.0, segments=64):
    verts = []
    faces = []

    for i in range(segments):
        t0 = 2 * math.pi * i / segments
        t1 = 2 * math.pi * (i + 1) / segments

        x0, z0 = radius * math.cos(t0), radius * math.sin(t0)
        x1, z1 = radius * math.cos(t1), radius * math.sin(t1)

        y0 = -height * 0.5
        y1 = height * 0.5

        base = len(verts)

        verts += [
            [x0, y0, z0],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z0],
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
        # Generate cloth mesh
        self.cloth_verts, self.cloth_faces = rolled_cloth_mesh()
        self.cloth_faces_flat = self.cloth_faces.reshape(-1)
        self.num_cloth_verts = len(self.cloth_verts)
        self.nv = 15  # vertices per row

        # Cylinder properties
        self.cyl1_radius = 9.9
        self.cyl2_radius = 14.9
        self.cyl1_center = (-27.2, 7.4)  # (X, Z)
        self.cyl2_center = (0.0, 0.0)  # (X, Z)

        # Generate cylinder meshes
        self.cyl1_verts, self.cyl1_faces = cylinder_mesh(radius=self.cyl1_radius)
        self.cyl2_verts, self.cyl2_faces = cylinder_mesh(radius=self.cyl2_radius)
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
        self.angular_speed = -np.pi  # rad/sec (base speed for cloth)
        linear_velocity = abs(self.angular_speed) * self.cyl1_radius
        self.angular_speed_cyl1 = -linear_velocity / self.cyl1_radius  # = angular_speed
        self.angular_speed_cyl2 = -linear_velocity / self.cyl2_radius  # slower due to larger radius
        self.spin_duration = 10.0  # seconds

    def custom_finalize(self):
        """Fix inner seam of cloth and set up cylinder rotation."""
        # Inner seam = kinematic rotation handle (last row of vertices)
        self.fixed_point_indices = [199 * self.nv + i for i in range(self.nv)]

        # Fix the inner seam vertices
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

    def run_step(self):
        """Execute one frame of simulation with rotation applied."""
        self.solver.rebuild_bvh(self.state_0)

        for _ in range(self.num_substeps):
            self.state_0.clear_forces()

            # Apply rotation during spin duration
            if self.sim_time < self.spin_duration:
                # Rotate cloth fixed points (around origin)
                wp.launch(
                    kernel=apply_rotation,
                    dim=len(self.fixed_point_indices),
                    inputs=[
                        self.angular_speed,
                        self.dt,
                        self.sim_time,
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

        ps.look_at((-20.0, 250.0, 20.0), (-0.1, 0.0, 0.1))

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
    truncation_mode = 1
    iterations = 10
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_treadmill_trunc{truncation_mode}_iter{iterations}_{timestamp}"

    config = {
        "name": "treadmill_cloth",
        "fps": 60,
        "sim_substeps": 10,
        "sim_num_frames": 500,
        "iterations": iterations,
        "truncation_mode": truncation_mode,
        "up_axis": "y",
        "gravity": 0,  # No gravity in this simulation
        "handle_self_contact": True,
        "self_contact_radius": 0.4,
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
    }

    sim = TreadmillSimulator(config)
    sim.finalize()
    sim.simulate()
