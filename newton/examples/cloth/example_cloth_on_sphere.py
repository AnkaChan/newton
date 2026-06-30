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
# Example Cloth on Sphere
#
# This simulation demonstrates a sheet of cloth falling onto a static sphere.
# The cloth drapes over the sphere under gravity, showcasing cloth-rigid body
# collision handling with realistic draping behavior.
#
# Cloth dimensions:
# - Size: 1.0m x 1.0m
# - Resolution: 32x32 cells
#
# Static sphere:
# - Radius: 0.25m
# - Position: centered at origin, elevated above ground
#
# Command: uv run -m newton.examples multiphysics.example_cloth_on_sphere
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0

        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 10

        # Cloth parameters
        self.cloth_size = 1.0  # m (1m x 1m cloth)
        self.dim_x = 32  # cells along x
        self.dim_y = 32  # cells along y
        self.cell_x = self.cloth_size / self.dim_x
        self.cell_y = self.cloth_size / self.dim_y

        # Sphere parameters
        self.sphere_radius = 0.25  # m
        self.sphere_height = 0.5  # m - height of sphere center above ground

        # Cloth drop parameters
        # Drop cloth above the sphere so it falls and drapes over it
        self.cloth_drop_height = self.sphere_height + self.sphere_radius + 0.3  # m

        # Build the model
        builder = newton.ModelBuilder(gravity=-9.8)  # m/s²

        # Add a static sphere for cloth to drape over
        body_sphere = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.sphere_height),
                q=wp.quat_identity(),
            ),
            key="sphere",
        )
        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = 0.0  # Static body (infinite mass)
        sphere_cfg.ke = 1.0e4  # Contact stiffness
        sphere_cfg.kd = 1.0e-1  # Contact damping
        sphere_cfg.mu = 0.5  # Friction
        builder.add_shape_sphere(body_sphere, radius=self.sphere_radius, cfg=sphere_cfg)

        # Cloth material properties
        # tri_ke/tri_ka: in-plane stretch stiffness
        # edge_ke: bending stiffness
        tri_ke = 1.0e3  # Stretch stiffness
        tri_ka = 1.0e3  # Shear stiffness
        tri_kd = 1.0e-1  # Damping
        edge_ke = 1.0e0  # Bending stiffness (low for soft draping)
        edge_kd = 1.0e-2  # Bending damping

        # Particle mass and radius
        particle_mass = 0.1  # kg per particle
        particle_radius = 0.01  # m

        # Add cloth grid
        # Center the cloth above the sphere
        cloth_pos_x = -self.cloth_size / 2
        cloth_pos_y = -self.cloth_size / 2
        cloth_pos_z = self.cloth_drop_height

        builder.add_cloth_grid(
            pos=wp.vec3(cloth_pos_x, cloth_pos_y, cloth_pos_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            mass=particle_mass,
            fix_left=False,
            fix_right=False,
            fix_top=False,
            fix_bottom=False,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
            particle_radius=particle_radius,
        )

        # Add ground plane
        builder.add_ground_plane()

        # Color the mesh for VBD solver (include bending constraints)
        builder.color(include_bending=True)

        # Finalize model
        self.model = builder.finalize()

        # Contact parameters for cloth-sphere and cloth-ground interactions
        self.model.soft_contact_ke = 1.0e4  # Contact stiffness
        self.model.soft_contact_kd = 1.0e-1  # Contact damping
        self.model.soft_contact_mu = 0.5  # Friction coefficient

        # Create VBD solver with self-contact enabled
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.01,  # m
            particle_self_contact_margin=0.015,  # m
        )

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline for sphere and ground contact
        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            broad_phase_mode=newton.BroadPhaseMode.NXN,
            soft_contact_margin=0.02,  # m
        )
        self.contacts = self.collision_pipeline.collide(self.model, self.state_0)

        self.viewer.set_model(self.model)

        # Set camera to view the draping
        self.viewer.set_camera(
            pos=wp.vec3(2.0, -2.0, 1.5),
            pitch=-20.0,
            yaw=135.0,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 60.0

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply viewer forces (for interactive manipulation)
            self.viewer.apply_forces(self.state_0)

            # Collision detection
            self.contacts = self.collision_pipeline.collide(self.model, self.state_0)

            # Solver step
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify simulation reached a valid end state."""
        # Test that cloth particles have settled (low velocity)
        newton.examples.test_particle_state(
            self.state_0,
            "cloth particles have come close to rest",
            lambda q, qd: max(abs(qd)) < 0.5,  # m/s
        )

        # Test that cloth particles are above ground
        newton.examples.test_particle_state(
            self.state_0,
            "cloth particles are above the ground",
            lambda q, qd: q[2] > -0.01,  # Allow small tolerance
        )

        # Test that cloth particles are within a reasonable volume
        # The cloth should drape around the sphere and touch the ground
        p_lower = wp.vec3(-1.0, -1.0, -0.01)  # m
        p_upper = wp.vec3(1.0, 1.0, 1.5)  # m
        newton.examples.test_particle_state(
            self.state_0,
            "cloth particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

        # Test that spring/edge lengths haven't stretched too much from rest length
        if self.model.spring_count > 0:
            positions = self.state_0.particle_q.numpy()
            spring_indices = self.model.spring_indices.numpy().reshape(-1, 2)
            rest_lengths = self.model.spring_rest_length.numpy()

            max_stretch_ratio = 0.0
            for i, (v0, v1) in enumerate(spring_indices):
                current_length = np.linalg.norm(positions[v0] - positions[v1])
                stretch_ratio = abs(current_length - rest_lengths[i]) / rest_lengths[i]
                max_stretch_ratio = max(max_stretch_ratio, stretch_ratio)

            # Allow up to 20% stretch/compression
            assert max_stretch_ratio < 0.2, (
                f"edges stretched too much from rest length: max stretch ratio = {max_stretch_ratio:.2%}"
            )


if __name__ == "__main__":
    # Create parser with base arguments
    parser = newton.examples.create_parser()

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
