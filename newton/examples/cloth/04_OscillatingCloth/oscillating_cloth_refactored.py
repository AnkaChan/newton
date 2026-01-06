# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Oscillating Cloth Simulation (Refactored)
#
# A demo that first stretches a cloth, then oscillates one of its edges.
# Three cloth layers are stacked and simulated together.
#
# Refactored to use M01_Simulator base class.
###########################################################################

import math
import os
from datetime import datetime
from os.path import join

import numpy as np
import polyscope as ps
import warp as wp
import warp.examples
from pxr import Usd, UsdGeom

from newton import ParticleFlags
from newton.examples.cloth.M01_Simulator import Simulator, default_config

# =============================================================================
# Motion Kernels
# =============================================================================


@wp.kernel
def left_edge_motion(
    q0: wp.array(dtype=wp.vec3),
    left_edge: wp.array(dtype=wp.int64),
    dx: wp.vec3,
    q1: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    particle_index = left_edge[i]
    q1[particle_index] = q0[particle_index] + dx
    q0[particle_index] = q1[particle_index]


@wp.kernel
def right_edge_motion(
    q0: wp.array(dtype=wp.vec3),
    right_edge: wp.array(dtype=wp.int64),
    t: float,
    dt: float,
    freq: float,
    amp: float,
    pull_duration: float,
    q1: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    particle_index = right_edge[i]
    dx = wp.vec3(0.0, amp * wp.sin(freq * (t - pull_duration)), 0.0)
    dx_prev = wp.vec3(0.0, amp * wp.sin(freq * (t - dt - pull_duration)), 0.0)
    q1[particle_index] = q0[particle_index] + dx - dx_prev
    q0[particle_index] = q1[particle_index]


# =============================================================================
# Configuration
# =============================================================================

example_config = {
    **default_config,  # Start with defaults
    "name": "oscillating_cloth",
    # Simulation timing
    "fps": 60,
    "sim_substeps": 5,
    "sim_num_frames": 900,
    "iterations": 10,
    "bvh_rebuild_frames": 1,
    # Solver settings
    "use_cuda_graph": False,  # Can't use CUDA graph due to conditional logic
    "handle_self_contact": True,
    "self_contact_radius": 0.2,
    "self_contact_margin": 0.3,
    "topological_contact_filter_threshold": 1,
    "truncation_mode": 1,
    # Global physics settings
    "up_axis": "y",
    "gravity": 0.0,  # No gravity for this demo
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 1.0e-5,
    "soft_contact_mu": 0.1,
    # Ground plane
    "has_ground": False,
    "ground_height": 0.0,
    # Visualization
    "do_rendering": True,
    "show_ground_plane": True,
    # Output settings
    "output_path": r"D:\Data\DAT_Sim\oscillating_cloth",
    "output_ext": "npy",
    "write_output": True,
    "write_video": True,
    # Oscillation-specific parameters
    "pull_speed": 20.0,
    "pull_duration": 0.0,  # Set to > 0 to have a pull phase before oscillation
    "osc_amp": 5.0,
    "osc_freq": 8.0 * math.pi,
    "cloth_offset": 0.2,  # Vertical offset between cloth layers
}


# =============================================================================
# Oscillating Cloth Simulator
# =============================================================================


class OscillatingClothSimulator(Simulator):
    """
    Oscillating cloth simulation using M01_Simulator base class.

    Simulates three stacked cloth layers with one edge fixed and the
    opposite edge oscillating sinusoidally.
    """

    def __init__(self, config: dict):
        # Store oscillation-specific config
        self.pull_speed = config.get("pull_speed", 20.0)
        self.pull_duration = config.get("pull_duration", 0.0)
        self.osc_amp = config.get("osc_amp", 5.0)
        self.osc_freq = config.get("osc_freq", 8.0 * math.pi)
        self.cloth_offset = config.get("cloth_offset", 0.4)
        self.cloth_size = 50

        # Track mesh info
        self.num_verts_per_cloth = 0
        self.cloth_faces = None

        super().__init__(config)

    def custom_init(self):
        """Add three cloth meshes to the builder."""
        # Load cloth mesh from USD
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth/cloth"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        self.num_verts_per_cloth = len(mesh_points)
        self.cloth_faces = mesh_indices.reshape(-1, 3)

        vertices = [wp.vec3(v) for v in mesh_points]

        # Add three cloth meshes at different heights (same parameters)
        for i in range(3):
            y_offset = 30.0 + i * self.cloth_offset
            self.builder.add_cloth_mesh(
                pos=wp.vec3(0.0, y_offset, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
                scale=1.0,
                vertices=vertices,
                indices=mesh_indices,
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.02,
                tri_ke=1.0e5,
                tri_ka=1.0e5,
                tri_kd=0.0,
                edge_ke=10,
                edge_kd=0.0,
                particle_radius=0.5,
            )

        # Compute edge indices for all three cloths
        left_edge_single = [self.cloth_size - 1 + i * self.cloth_size for i in range(self.cloth_size)]
        right_edge_single = [i * self.cloth_size for i in range(self.cloth_size)]

        # Extend to all three cloths
        self.left_edge_indices = []
        self.right_edge_indices = []
        for cloth_idx in range(3):
            offset = cloth_idx * self.num_verts_per_cloth
            self.left_edge_indices.extend([idx + offset for idx in left_edge_single])
            self.right_edge_indices.extend([idx + offset for idx in right_edge_single])

    def custom_finalize(self):
        """Set up fixed edges after model finalization."""
        # Fix left edge vertices
        if self.left_edge_indices:
            flags = self.model.particle_flags.numpy()
            for idx in self.left_edge_indices:
                flags[idx] = flags[idx] & ~ParticleFlags.ACTIVE
            self.model.particle_flags = wp.array(flags)

        # Fix right edge vertices
        if self.right_edge_indices:
            flags = self.model.particle_flags.numpy()
            for idx in self.right_edge_indices:
                flags[idx] = flags[idx] & ~ParticleFlags.ACTIVE
            self.model.particle_flags = wp.array(flags)

        # Create warp arrays for edge indices
        self.left_edge = wp.array(self.left_edge_indices, dtype=wp.int64)
        self.right_edge = wp.array(self.right_edge_indices, dtype=wp.int64)

    def run_step(self):
        """Override to add edge motion logic to the simulation step."""
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()

            # Apply motion based on simulation phase
            if self.sim_time < self.pull_duration:
                # Pull phase: move left edge
                wp.launch(
                    kernel=left_edge_motion,
                    inputs=[
                        self.state_0.particle_q,
                        self.left_edge,
                        wp.vec3(self.pull_speed * self.dt, 0.0, 0.0),
                    ],
                    outputs=[self.state_1.particle_q],
                    dim=len(self.left_edge),
                )
            else:
                # Oscillation phase: oscillate right edge
                wp.launch(
                    kernel=right_edge_motion,
                    inputs=[
                        self.state_0.particle_q,
                        self.right_edge,
                        self.sim_time,
                        self.dt,
                        self.osc_freq,
                        self.osc_amp,
                        self.pull_duration,
                    ],
                    outputs=[self.state_1.particle_q],
                    dim=len(self.right_edge),
                )

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def setup_polyscope_meshes(self):
        """Set up separate polyscope meshes for each cloth layer."""
        if not self.do_rendering:
            return

        ps.look_at((0, 80, 200), (0, 0, 0))

        all_verts = self.model.particle_q.numpy()
        n = self.num_verts_per_cloth

        # Register each cloth layer
        colors = [
            (0.8, 0.3, 0.3),  # Cloth 1: Red
            (0.3, 0.8, 0.3),  # Cloth 2: Green
            (0.3, 0.3, 0.8),  # Cloth 3: Blue
        ]

        for i in range(3):
            start_idx = i * n
            end_idx = (i + 1) * n
            cloth_verts = all_verts[start_idx:end_idx]

            self.register_ps_mesh(
                name=f"Cloth{i + 1}",
                vertices=cloth_verts,
                faces=self.cloth_faces,
                vertex_indices=slice(start_idx, end_idx),
                color=colors[i],
            )

    def save_initial_meshes(self):
        """Save initial mesh topology as separate PLY files for each cloth."""
        if self.output_path is None:
            return

        all_verts = self.model.particle_q.numpy()
        n = self.num_verts_per_cloth

        for i in range(3):
            start_idx = i * n
            end_idx = (i + 1) * n
            cloth_verts = all_verts[start_idx:end_idx]

            out_file = join(self.output_path, f"initial_cloth{i + 1}.ply")
            self._save_ply(out_file, cloth_verts, self.cloth_faces)
            print(f"Initial cloth {i + 1} mesh saved to: {out_file}")

    def _save_ply(self, filename, verts, faces):
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
# Main
# =============================================================================

if __name__ == "__main__":
    # wp.clear_kernel_cache()

    # Create output subfolder with truncation mode, iterations, and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    truncation_mode = example_config["truncation_mode"]
    iterations = example_config["iterations"]
    base_output_path = example_config["output_path"]
    output_dir = os.path.join(base_output_path, f"truncation_{truncation_mode}_iter_{iterations}_{timestamp}")
    example_config["output_path"] = output_dir
    # Note: output directory is created automatically by M01_Simulator.finalize()

    print(f"Output directory: {output_dir}")
    print(f"Oscillation: amp={example_config['osc_amp']}, freq={example_config['osc_freq']:.2f} rad/s")

    # Create simulator and run
    sim = OscillatingClothSimulator(example_config)
    sim.finalize()
    sim.simulate()

    print("\nSimulation complete!")
