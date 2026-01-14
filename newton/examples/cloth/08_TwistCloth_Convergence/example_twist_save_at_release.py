# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Cloth Twist - Save at Release Point
#
# This script runs the cloth twist simulation until just before release
# and saves the recovery state for convergence evaluation.
#
# The cloth is twisted for rot_end_time seconds, then the recovery state
# is saved at the exact frame when release would occur.
###########################################################################

import itertools
import math
import os
from datetime import datetime

import numpy as np
import warp as wp

from newton import ParticleFlags
from newton.examples.cloth.M01_Simulator import Simulator, default_config

# =============================================================================
# Mesh Generation
# =============================================================================


def generate_cloth_mesh(Nx, Ny, size_x=2.0, size_y=2.0, position=(0, 0)):
    """
    Generate a cloth mesh with cross tessellation.

    Args:
        Nx: Number of vertices along X axis
        Ny: Number of vertices along Y axis
        size_x: Physical size of the cloth along X axis
        size_y: Physical size of the cloth along Y axis
        position: Center position (x, y) of the cloth

    Returns:
        vertices: List of vertex positions (x, y, z)
        faces: List of face indices (triangles)
    """
    X = np.linspace(-0.5 * size_x + position[0], 0.5 * size_x + position[0], Nx)
    Y = np.linspace(-0.5 * size_y + position[1], 0.5 * size_y + position[1], Ny)

    X, Y = np.meshgrid(X, Y, indexing="ij")
    Z = np.zeros((Nx, Ny))

    vertices = []
    for i, j in itertools.product(range(Nx), range(Ny)):
        vertices.append((X[i, j], Z[i, j], Y[i, j]))

    faces = []
    for i, j in itertools.product(range(0, Nx - 1), range(0, Ny - 1)):
        vId = j + i * Ny

        if (j + i) % 2:
            faces.append((vId, vId + Ny + 1, vId + 1))
            faces.append((vId, vId + Ny, vId + Ny + 1))
        else:
            faces.append((vId, vId + Ny, vId + 1))
            faces.append((vId + Ny, vId + Ny + 1, vId + 1))

    return vertices, faces


# =============================================================================
# Rotation Kernels (copied from original)
# =============================================================================


@wp.kernel
def initialize_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot_centers: wp.array(dtype=wp.vec3),
    rot_axes: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    # output
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    v_index = vertex_indices_to_rot[wp.tid()]

    p = pos[v_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]
    op = p - rot_center

    root = wp.dot(op, rot_axis) * rot_axis

    root_to_p = p - root

    roots[tid] = root
    roots_to_ps[tid] = root_to_p

    if tid == 0:
        t[0] = 0.0


@wp.kernel
def apply_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    rot_axes: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    angular_velocity: float,
    dt: float,
    end_time: float,
    # output
    pos_0: wp.array(dtype=wp.vec3),
    pos_1: wp.array(dtype=wp.vec3),
):
    cur_t = t[0]
    if cur_t > end_time:
        return

    tid = wp.tid()
    v_index = vertex_indices_to_rot[wp.tid()]

    rot_axis = rot_axes[tid]

    ux = rot_axis[0]
    uy = rot_axis[1]
    uz = rot_axis[2]

    theta = cur_t * angular_velocity

    R = wp.mat33(
        wp.cos(theta) + ux * ux * (1.0 - wp.cos(theta)),
        ux * uy * (1.0 - wp.cos(theta)) - uz * wp.sin(theta),
        ux * uz * (1.0 - wp.cos(theta)) + uy * wp.sin(theta),
        uy * ux * (1.0 - wp.cos(theta)) + uz * wp.sin(theta),
        wp.cos(theta) + uy * uy * (1.0 - wp.cos(theta)),
        uy * uz * (1.0 - wp.cos(theta)) - ux * wp.sin(theta),
        uz * ux * (1.0 - wp.cos(theta)) - uy * wp.sin(theta),
        uz * uy * (1.0 - wp.cos(theta)) + ux * wp.sin(theta),
        wp.cos(theta) + uz * uz * (1.0 - wp.cos(theta)),
    )

    root = roots[tid]
    root_to_p = roots_to_ps[tid]
    root_to_p_rot = R * root_to_p
    p_rot = root + root_to_p_rot

    pos_0[v_index] = p_rot
    pos_1[v_index] = p_rot

    if tid == 0:
        t[0] = cur_t + dt


# =============================================================================
# Configuration
# =============================================================================

example_config = {
    **default_config,  # Start with defaults
    "name": "cloth_twist_convergence",
    # Simulation timing
    "fps": 60,
    "sim_substeps": 10,
    "sim_num_frames": 601,  # Run until frame 600 (release point)
    "iterations": 10,
    "bvh_rebuild_frames": 10,
    # Solver settings
    "use_cuda_graph": True,
    "handle_self_contact": True,
    "use_tile_solve": False,
    "self_contact_radius": 0.25,
    "self_contact_margin": 0.30,
    "topological_contact_filter_threshold": 2,
    "rest_shape_contact_exclusion_radius": 0.0,
    "vertex_collision_buffer_pre_alloc": 64,
    "edge_collision_buffer_pre_alloc": 128,
    "collision_buffer_resize_frames": 5,
    "collision_buffer_growth_ratio": 1.5,
    "collision_detection_interval": 5,
    "include_bending": True,
    # Global physics settings
    "up_axis": "y",
    "gravity": -0.0,  # No gravity during twist
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 0.0,
    "soft_contact_mu": 0.2,
    # Visualization
    "do_rendering": True,
    "show_ground_plane": False,
    "is_initially_paused": False,
    # Ground plane
    "has_ground": False,
    "ground_height": 0.0,
    # Cloth mesh resolution and size
    "cloth_Nx": 100,  # Number of vertices along X axis
    "cloth_Ny": 100,  # Number of vertices along Y axis
    "cloth_size_x": 100.0,  # Physical size along X axis (same as original USD mesh)
    "cloth_size_y": 100.0,  # Physical size along Y axis
    # Cloth material properties
    "cloth_density": 0.02,
    "cloth_tri_ke": 1.0e5,
    "cloth_tri_ka": 1.0e5,
    "cloth_tri_kd": 0.0,
    "cloth_edge_ke": 1,
    "cloth_edge_kd": 1e-3,
    # Twist-specific parameters
    "rot_angular_velocity": math.pi / 3,
    "rot_end_time": 10.0,  # Twist for 10 seconds
    "truncation_mode": 1,
    # outputs
    "output_path": r"D:\Data\DAT_Sim\cloth_twist_convergence",
    "output_ext": "npy",
    "write_output": False,
    "write_video": False,
    "recovery_state_save_steps": -1,  # Disabled, we'll save manually at the end
}


# =============================================================================
# Twist Cloth Simulator (no release)
# =============================================================================


class TwistClothSimulator(Simulator):
    """
    Cloth twist simulation that runs until the release point and saves state.
    This version does NOT release the cloth - it just saves the twisted state.
    """

    def __init__(self, config: dict):
        # Store twist-specific config
        self.rot_angular_velocity = config.get("rot_angular_velocity", math.pi / 3)
        self.rot_end_time = config.get("rot_end_time", 10.0)
        # Cloth resolution (Nx x Ny vertices)
        self.cloth_Nx = config.get("cloth_Nx", 50)
        self.cloth_Ny = config.get("cloth_Ny", 50)

        super().__init__(config)

    def custom_init(self):
        """Add the cloth mesh to the builder."""
        # Generate cloth mesh with configurable resolution
        Nx = self.cloth_Nx
        Ny = self.cloth_Ny
        size_x = self.config.get("cloth_size_x", 50.0)
        size_y = self.config.get("cloth_size_y", 50.0)

        mesh_points, mesh_faces = generate_cloth_mesh(Nx, Ny, size_x, size_y)

        # Convert faces to flat indices
        mesh_indices = np.array(mesh_faces, dtype=np.int32).flatten()

        # Store faces for output saving
        self.faces = np.array(mesh_faces, dtype=np.int32)

        vertices = [wp.vec3(v) for v in mesh_points]

        # Add cloth to builder using config values
        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2),
            scale=1,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=self.config.get("cloth_density", 0.02),
            tri_ke=self.config.get("cloth_tri_ke", 1.0e5),
            tri_ka=self.config.get("cloth_tri_ka", 1.0e5),
            tri_kd=self.config.get("cloth_tri_kd", 0.0),
            edge_ke=self.config.get("cloth_edge_ke", 10.0),
            edge_kd=self.config.get("cloth_edge_kd", 1e-3),
        )

    def custom_finalize(self):
        """Set up rotation logic after model finalization."""
        # Identify vertices to rotate (left and right edges of cloth)
        # Mesh is generated with vertices indexed as: index = i*Ny + j
        # where i is column (X direction), j is row (Z direction)
        # After 90° rotation around Z, cloth is in YZ plane with:
        #   - Y varies with original X (column index i)
        #   - Z varies with original Z (row index j)
        # We pin the LEFT (i=0) and RIGHT (i=Nx-1) vertical edges
        Nx = self.cloth_Nx
        Ny = self.cloth_Ny
        # Left edge: i=0, all j → indices 0 to Ny-1
        self.left_side = list(range(Ny))
        # Right edge: i=Nx-1, all j → indices (Nx-1)*Ny to Nx*Ny-1
        self.right_side = list(range((Nx - 1) * Ny, Nx * Ny))
        rot_point_indices = self.left_side + self.right_side

        # Fix rotation points
        if len(rot_point_indices):
            flags = self.model.particle_flags.numpy()
            for fixed_vertex_id in rot_point_indices:
                flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE

            self.model.particle_flags = wp.array(flags)

        # Set up rotation axes
        rot_axes = [[0, 1, 0]] * len(self.right_side) + [[0, -1, 0]] * len(self.left_side)

        self.rot_point_indices = wp.array(rot_point_indices, dtype=int)
        self.t = wp.zeros((1,), dtype=float)
        self.rot_centers = wp.zeros(len(rot_point_indices), dtype=wp.vec3)
        self.rot_axes = wp.array(rot_axes, dtype=wp.vec3)

        self.roots = wp.zeros_like(self.rot_centers)
        self.roots_to_ps = wp.zeros_like(self.rot_centers)

        # Initialize rotation
        wp.launch(
            kernel=initialize_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[
                self.rot_point_indices,
                self.state_0.particle_q,
                self.rot_centers,
                self.rot_axes,
                self.t,
            ],
            outputs=[
                self.roots,
                self.roots_to_ps,
            ],
        )

    def run_step(self):
        """Override to add rotation logic to the simulation step."""
        # Run substeps
        for _ in range(self.num_substeps):
            # Run collision detection
            if self.model.shape_count:
                self.contacts = self.model.collide(self.state_0)

            self.state_0.clear_forces()

            # Apply rotation to fixed vertices
            wp.launch(
                kernel=apply_rotation,
                dim=self.rot_point_indices.shape[0],
                inputs=[
                    self.rot_point_indices,
                    self.rot_axes,
                    self.roots,
                    self.roots_to_ps,
                    self.t,
                    self.rot_angular_velocity,
                    self.dt,
                    self.rot_end_time,
                ],
                outputs=[
                    self.state_0.particle_q,
                    self.state_1.particle_q,
                ],
            )

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0


# =============================================================================
# Main
# =============================================================================

def save_config(config: dict, output_dir: str):
    """Save the configuration to a JSON file for reproducibility."""
    import json
    config_path = os.path.join(output_dir, "run_config.json")
    
    # Convert non-JSON-serializable values
    config_to_save = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            config_to_save[k] = v
        else:
            config_to_save[k] = str(v)  # Convert to string for non-serializable types
    
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    return config_path


if __name__ == "__main__":
    # wp.clear_kernel_cache()

    # Create output subfolder with resolution, truncation mode, iterations, and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    truncation_mode = example_config["truncation_mode"]
    iterations = example_config["iterations"]
    cloth_Nx = example_config["cloth_Nx"]
    cloth_Ny = example_config["cloth_Ny"]
    base_output_path = example_config["output_path"]
    output_dir = os.path.join(
        base_output_path, f"res_{cloth_Nx}x{cloth_Ny}_truncation_{truncation_mode}_iter_{iterations}_{timestamp}"
    )
    example_config["output_path"] = output_dir

    # Calculate release frame
    fps = example_config["fps"]
    rot_end_time = example_config["rot_end_time"]
    release_frame = int(fps * rot_end_time)  # Frame 600 for 60fps * 10s

    print(f"Output directory: {output_dir}")
    print(f"Cloth resolution: {cloth_Nx} x {cloth_Ny} = {cloth_Nx * cloth_Ny} vertices")
    print(f"Rotation: {example_config['rot_angular_velocity']:.3f} rad/s for {example_config['rot_end_time']:.1f}s")
    print(f"Will save recovery state at frame {release_frame} (release point)")

    # Create simulator and run
    sim = TwistClothSimulator(example_config)
    sim.finalize()
    
    # Create output directory and save config BEFORE simulation
    os.makedirs(output_dir, exist_ok=True)
    save_config(example_config, output_dir)
    
    sim.simulate()

    # Save recovery state at the release point
    sim.save_recovery_state(release_frame)
    print(f"\nRecovery state saved at frame {release_frame}")
    print(f"Recovery file: {os.path.join(output_dir, f'recovery_state_{release_frame:06d}.npz')}")

    # Also save the initial mesh for reference
    sim.save_initial_meshes()

    print("\nSimulation complete! Recovery state saved for convergence evaluation.")
