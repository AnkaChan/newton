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
# Example Cloth Twist (Refactored)
#
# This simulation demonstrates twisting an FEM cloth model using the VBD
# solver, showcasing its ability to handle complex self-contacts while
# ensuring it remains intersection-free.
#
# Refactored to use M01_Simulator base class.
###########################################################################

import math
import os
from datetime import datetime

import numpy as np
import warp as wp
import warp.examples
from pxr import Usd, UsdGeom

import newton
from newton import ParticleFlags
from newton.examples.cloth.M01_Simulator import Simulator, default_config


# =============================================================================
# Rotation Kernels
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
    "name": "cloth_twist",
    # Simulation timing
    "fps": 60,
    "sim_substeps": 10,
    "sim_num_frames": 1500,
    "iterations": 10,
    "bvh_rebuild_frames": 10,
    # Solver settings
    "use_cuda_graph": True,
    "handle_self_contact": True,
    "use_tile_solve": False,
    "self_contact_radius": 0.2,
    "self_contact_margin": 0.35,
    "topological_contact_filter_threshold": 1,
    "rest_shape_contact_exclusion_radius": 0.0,
    "vertex_collision_buffer_pre_alloc": 128,
    "edge_collision_buffer_pre_alloc": 256,
    "collision_buffer_resize_frames": -1,
    "include_bending": False,
    # Global physics settings
    "up_axis": "y",
    "gravity": 0.0,
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 0.0,
    "soft_contact_mu": 0.2,
    # Output settings
    "output_path": None,  # Will be set based on timestamp
    "output_ext": "ply",
    "write_output": True,
    "write_video": True,
    "recovery_state_save_steps": -1,
    # Visualization
    "do_rendering": True,
    "show_ground_plane": False,
    "is_initially_paused": False,
    # Ground plane
    "has_ground": False,
    "ground_height": 0.0,
    # Twist-specific parameters
    "rot_angular_velocity": math.pi / 3,
    "rot_end_time": 10.0,
    "truncation_mode": 1,
    "collision_detection_interval": 5,
}


# =============================================================================
# Twist Cloth Simulator
# =============================================================================


class TwistClothSimulator(Simulator):
    """
    Cloth twist simulation using M01_Simulator base class.
    """

    def __init__(self, config: dict):
        # Store twist-specific config
        self.rot_angular_velocity = config.get("rot_angular_velocity", math.pi / 3)
        self.rot_end_time = config.get("rot_end_time", 10.0)
        self.cloth_size = 50

        super().__init__(config)

    def custom_init(self):
        """Add the cloth mesh to the builder."""
        # Load cloth mesh from USD
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth/cloth"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        vertices = [wp.vec3(v) for v in mesh_points]

        # Add cloth to builder
        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2),
            scale=1,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=0,
            edge_ke=1e-3,
            edge_kd=0,
        )

    def custom_finalize(self):
        """Set up rotation logic after model finalization."""
        # Identify vertices to rotate (left and right sides of cloth)
        self.left_side = [self.cloth_size - 1 + i * self.cloth_size for i in range(self.cloth_size)]
        self.right_side = [i * self.cloth_size for i in range(self.cloth_size)]
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

    def step(self):
        """Override step to release right side after rotation ends."""
        super().step()

        # Release right side after rotation time
        if self.sim_time >= self.rot_end_time:
            # Only do this once when we cross the threshold
            if hasattr(self, "_right_side_released"):
                return
            self._right_side_released = True

            flags = self.model.particle_flags.numpy()
            for fixed_vertex_id in self.right_side:
                flags[fixed_vertex_id] = flags[fixed_vertex_id] | ParticleFlags.ACTIVE

            self.model.particle_flags = wp.array(flags)
            print(f"Released right side at t={self.sim_time:.2f}s")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    wp.clear_kernel_cache()

    # Set output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "Output",
        f"twist_{timestamp}",
    )
    example_config["output_path"] = output_dir

    print(f"Output directory: {output_dir}")
    print(f"Rotation: {example_config['rot_angular_velocity']:.3f} rad/s for {example_config['rot_end_time']:.1f}s")

    # Create simulator and run
    sim = TwistClothSimulator(example_config)
    sim.finalize()
    sim.simulate()

    print("\nSimulation complete!")

