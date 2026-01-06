# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Unroll Cloth Simulation (Refactored)
#
# This simulation demonstrates a rolled cloth unrolling onto a prism
# collider using the VBD solver with self-contact handling.
#
# Refactored to use M01_Simulator base class.
###########################################################################

import math
import os
from datetime import datetime
from os.path import join

import numpy as np
import warp as wp

from newton import ParticleFlags
from newton.examples.cloth.M01_Simulator import Simulator, default_config, read_obj

# =============================================================================
# Mesh Generation
# =============================================================================


def rolled_cloth_mesh(length=500.0, width=100.0, nu=200, nv=15, inner_radius=10.0, thickness=0.4):
    """
    Generate a rolled cloth mesh (spiral shape).

    Args:
        length: Length of the cloth when unrolled.
        width: Width of the cloth.
        nu: Number of vertices along the length.
        nv: Number of vertices along the width.
        inner_radius: Inner radius of the roll.
        thickness: Thickness between layers of the roll.

    Returns:
        Tuple of (vertices, faces) as numpy arrays.
    """
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

    return np.array(verts), np.array(faces)


def save_ply(filename, verts, faces):
    """Save a mesh to a PLY file."""
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# =============================================================================
# Configuration
# =============================================================================

example_config = {
    **default_config,  # Start with defaults
    "name": "unroll_cloth",
    # Simulation timing
    "fps": 60,
    "sim_substeps": 20,
    "sim_num_frames": 250,
    "iterations": 5,
    "bvh_rebuild_frames": 1,
    # Solver settings
    "use_cuda_graph": True,
    "handle_self_contact": True,
    "self_contact_radius": 0.4,
    "self_contact_margin": 0.6,
    "topological_contact_filter_threshold": 2,
    "truncation_mode": 1,
    # Global physics settings
    "up_axis": "y",
    "gravity": -1000.0,
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 1.0e-5,
    "soft_contact_mu": 0.1,
    # Ground plane
    "has_ground": True,
    "ground_height": 0.0,
    # Visualization
    "output_path": r"D:\Data\DAT_Sim\unroll_cloth",  # Directory to save output files
    "output_ext": "npy",  # "ply", "usd", or "npy" (npy saves only positions, initial meshes saved as ply)
    "do_rendering": True,
    "show_ground_plane": True,
    # Output settings
    "write_output": True,
    "write_video": True,
}


# =============================================================================
# Unroll Cloth Simulator
# =============================================================================


class UnrollClothSimulator(Simulator):
    """
    Cloth unrolling simulation using M01_Simulator base class.

    Simulates a rolled cloth unrolling onto a prism-shaped collider.
    """

    def __init__(self, config: dict):
        # Track mesh info
        self.num_collider_verts = 0
        self.collider_faces = None
        self.cloth_faces = None

        # Get script directory for loading prism.obj
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        super().__init__(config)

    def custom_init(self):
        """Add collider and cloth meshes to the builder."""
        # Generate rolled cloth mesh
        cloth_verts, self.cloth_faces = rolled_cloth_mesh()

        # Load collider mesh
        prism_path = os.path.join(self.script_dir, "prism.obj")
        vs_collider, fs_collider = read_obj(prism_path)
        self.num_collider_verts = len(vs_collider)
        self.collider_faces = np.array(fs_collider)

        # Add collider mesh as static geometry
        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 100.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -math.pi / 2),
            scale=wp.vec3(80.0, 100.0, 100.0),
            vertices=[wp.vec3(v) for v in vs_collider],
            indices=np.array(fs_collider).reshape(-1),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
        )

        # Make collider static
        if self.builder.particle_count > 0:
            for i in range(self.num_collider_verts):
                self.builder.particle_mass[i] = 0.0
                self.builder.particle_flags[i] &= ~ParticleFlags.ACTIVE

        # Add cloth mesh
        self.builder.add_cloth_mesh(
            pos=wp.vec3(50.0, 180.0, -40.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.0),
            scale=1.0,
            vertices=cloth_verts,
            indices=self.cloth_faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
            particle_radius=0.5,
        )

        # Store fixed point indices (end vertices of rolled cloth)
        # nu=200, nv=15, so last row is at index 199 * 15 = 2985 to 2999
        self.fixed_point_indices = [self.num_collider_verts + 15 * 199 + i for i in range(15)]

    def custom_finalize(self):
        """Set up fixed particles after model finalization."""
        # Fix the end vertices of the cloth
        if self.fixed_point_indices:
            flags = self.model.particle_flags.numpy()
            for fixed_vertex_id in self.fixed_point_indices:
                flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE
            self.model.particle_flags = wp.array(flags)

    def setup_polyscope_meshes(self):
        """Set up separate polyscope meshes for collider and cloth."""
        if not self.do_rendering:
            return

        all_verts = self.model.particle_q.numpy()

        # Register collider mesh
        collider_verts = all_verts[: self.num_collider_verts]
        self.register_ps_mesh(
            name="Collider",
            vertices=collider_verts,
            faces=self.collider_faces,
            vertex_indices=slice(0, self.num_collider_verts),
            color=(0.3, 0.3, 0.3),
        )

        # Register cloth mesh
        cloth_verts = all_verts[self.num_collider_verts :]
        self.register_ps_mesh(
            name="Cloth",
            vertices=cloth_verts,
            faces=self.cloth_faces,
            vertex_indices=slice(self.num_collider_verts, None),
            color=(0.8, 0.4, 0.4),
        )

    def save_initial_meshes(self):
        """Save initial mesh topology as separate PLY files for collider and cloth."""
        if self.output_path is None:
            return

        all_verts = self.model.particle_q.numpy()

        # Save collider mesh
        collider_verts = all_verts[: self.num_collider_verts]
        collider_file = join(self.output_path, "initial_collider.ply")
        save_ply(collider_file, collider_verts, self.collider_faces)
        print(f"Initial collider mesh saved to: {collider_file}")

        # Save cloth mesh
        cloth_verts = all_verts[self.num_collider_verts :]
        cloth_file = join(self.output_path, "initial_cloth.ply")
        save_ply(cloth_file, cloth_verts, self.cloth_faces)
        print(f"Initial cloth mesh saved to: {cloth_file}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    wp.clear_kernel_cache()

    # Generate and save the rolled cloth mesh for reference
    verts, faces = rolled_cloth_mesh()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_ply(os.path.join(script_dir, "rolled_cloth.ply"), verts, faces)

    # Create output subfolder with truncation mode, iterations, and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    truncation_mode = example_config["truncation_mode"]
    iterations = example_config["iterations"]
    base_output_path = example_config["output_path"]
    output_dir = os.path.join(base_output_path, f"truncation_{truncation_mode}_iter_{iterations}_{timestamp}")
    example_config["output_path"] = output_dir
    # Note: output directory is created automatically by M01_Simulator.finalize()

    print(f"Output directory: {output_dir}")

    # Create simulator and run
    sim = UnrollClothSimulator(example_config)
    sim.finalize()
    sim.simulate()

    print("\nSimulation complete!")
