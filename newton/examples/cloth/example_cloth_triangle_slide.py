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
# Example Triangle Slide
#
# Drop a single triangle onto a 45-degree tilted plane and measure the
# time it takes to slide to the ground.
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

# =============================================================================
# EXPERIMENT PARAMETERS - All configurable parameters collected here
# =============================================================================

# Triangle parameters (in meters)
TRIANGLE_EDGE_LENGTH = 1.0  # Edge length of equilateral triangle (m)

# Plane parameters
PLANE_TILT_ANGLE_DEG = 45.0  # Tilt angle from horizontal (degrees)

# Drop parameters
DROP_HEIGHT = 5.0  # Height above ground to start (m)
DROP_OFFSET_ABOVE_PLANE = 0.01  # Small gap above the plane surface (m)

# Physics parameters
CLOTH_DENSITY = 0.02  # Density per area (kg/m^2)
TRI_KE = 5.0e1  # Triangle stretch stiffness
TRI_KA = 5.0e1  # Triangle area stiffness
TRI_KD = 1.0e-1  # Triangle damping

# Contact parameters
CONTACT_KE = 1.0e2  # Contact stiffness
CONTACT_KD = 1.0e0  # Contact damping

# friction parameters
CONTACT_MU = 0.5  # Friction coefficient for cloth particles
PLANE_MU = 0.5  # Friction coefficient of the tilted plane (adjustable)
GROUND_MU = 0.5  # Friction coefficient of the ground plane

# CONTACT_MU = 1.0  # Friction coefficient for cloth particles
# PLANE_MU =1.0  # Friction coefficient of the tilted plane (adjustable)
# GROUND_MU = 1.0  # Friction coefficient of the ground plane

# Simulation parameters
FPS = 60
SIM_SUBSTEPS = 10
SOLVER_ITERATIONS = 10

# Measurement parameters
GROUND_THRESHOLD = 0.05  # Consider triangle at ground when z < this value (m)
REST_VELOCITY_THRESHOLD = 0.01  # Consider at rest when speed < this value (m/s)

# Camera parameters
CAMERA_DISTANCE = 5.0  # Distance from the scene (m)
CAMERA_PITCH = -20.0  # Camera pitch angle (degrees, negative = looking down)
CAMERA_YAW = 90.0  # Camera yaw angle (degrees)
ENABLE_VSYNC = True  # Lock to display refresh rate (typically 60fps)

# Particle visualization
PARTICLE_RADIUS = 0.05  # Radius for particle visualization (m)

# Collision parameters
SOFT_CONTACT_MARGIN = 0.1  # Contact margin for collision detection (m)

# =============================================================================
# DERIVED PARAMETERS
# =============================================================================

# Convert tilt angle to radians
PLANE_TILT_ANGLE_RAD = math.radians(PLANE_TILT_ANGLE_DEG)

# Compute plane normal (tilted from vertical in X direction)
# Normal points at angle from +Z toward +X
PLANE_NORMAL_X = math.sin(PLANE_TILT_ANGLE_RAD)
PLANE_NORMAL_Z = math.cos(PLANE_TILT_ANGLE_RAD)

# Compute triangle vertices (equilateral triangle in XY plane, centered at origin)
# For equilateral triangle with edge length a:
# - Height h = a * sqrt(3) / 2
# - Circumradius R = a / sqrt(3)
TRIANGLE_HEIGHT = TRIANGLE_EDGE_LENGTH * math.sqrt(3) / 2
TRIANGLE_CIRCUMRADIUS = TRIANGLE_EDGE_LENGTH / math.sqrt(3)

# Vertices centered at origin in XY plane
TRIANGLE_VERTICES = [
    wp.vec3(0.0, TRIANGLE_CIRCUMRADIUS, 0.0),  # Top vertex
    wp.vec3(-TRIANGLE_EDGE_LENGTH / 2, -TRIANGLE_HEIGHT / 3, 0.0),  # Bottom left
    wp.vec3(TRIANGLE_EDGE_LENGTH / 2, -TRIANGLE_HEIGHT / 3, 0.0),  # Bottom right
]
TRIANGLE_INDICES = [0, 1, 2]

# Starting position: at DROP_HEIGHT, positioned on the tilted plane
# For plane with normal (nx, 0, nz) passing through origin: nx*x + nz*z = 0
# At z = DROP_HEIGHT, x = -DROP_HEIGHT * nz / nx (for normal pointing up-right)
# But we want the plane to be below the drop point, so we offset
START_X = -DROP_HEIGHT * PLANE_NORMAL_Z / PLANE_NORMAL_X + TRIANGLE_EDGE_LENGTH * 0.55
START_Z = DROP_HEIGHT + DROP_OFFSET_ABOVE_PLANE
START_POSITION = wp.vec3(START_X, 0.0, START_Z)


class Example:
    def __init__(self, viewer):
        # Simulation timing
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = SOLVER_ITERATIONS

        self.viewer = viewer

        # Measurement state
        self.slide_complete = False
        self.slide_time = None
        self.start_time = 0.0

        # Ground sliding measurement
        self.reached_ground = False
        self.ground_arrival_time = None
        self.ground_arrival_position = None
        self.rest_complete = False
        self.rest_time = None
        self.rest_position = None
        self.ground_slide_distance = None

        # Build the model
        builder = newton.ModelBuilder()

        # Configure contact parameters
        builder.default_shape_cfg.ke = CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = CONTACT_MU

        # Add the triangle cloth mesh
        builder.add_cloth_mesh(
            pos=START_POSITION,
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=TRIANGLE_VERTICES,
            indices=TRIANGLE_INDICES,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=CLOTH_DENSITY,
            tri_ke=TRI_KE,
            tri_ka=TRI_KA,
            tri_kd=TRI_KD,
            edge_ke=0.0,  # No bending for single triangle
            edge_kd=0.0,
            particle_radius=PARTICLE_RADIUS,
        )

        builder.color(include_bending=False)

        # Add 45-degree tilted plane with specific friction
        # Plane equation: nx*x + nz*z = 0 (passes through origin)
        plane_normal = (PLANE_NORMAL_X, 0.0, PLANE_NORMAL_Z, 0.0)
        tilted_plane_cfg = newton.ModelBuilder.ShapeConfig(
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=PLANE_MU,
        )
        builder.add_shape_plane(
            plane=plane_normal,
            width=0.0,  # Infinite plane
            length=0.0,
            cfg=tilted_plane_cfg,
            key="tilted_plane",
        )

        # Add ground plane at z=0 (to catch the triangle at the bottom)
        ground_plane_cfg = newton.ModelBuilder.ShapeConfig(
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=GROUND_MU,
        )
        builder.add_ground_plane(cfg=ground_plane_cfg, key="ground_plane")

        self.model = builder.finalize()
        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = CONTACT_KD
        self.model.soft_contact_mu = CONTACT_MU

        # Create VBD solver
        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            particle_enable_self_contact=False,
        )

        # Collision pipeline
        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            broad_phase_mode=newton.BroadPhaseMode.NXN,
            soft_contact_margin=SOFT_CONTACT_MARGIN,
        )

        # Initialize states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.collide(self.model, self.state_0)

        self.viewer.set_model(self.model)

        # Enable VSync to lock to display refresh rate (typically 60fps)
        if ENABLE_VSYNC and hasattr(self.viewer, "vsync"):
            self.viewer.vsync = True

        # Set up camera to view the triangle at start position
        # Camera positioned to the side (-Y direction), looking at the triangle
        camera_pos = wp.vec3(START_X + 1.0, -CAMERA_DISTANCE, START_Z + 1.0)
        self.viewer.set_camera(pos=camera_pos, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)

        # Print experiment setup
        print("=" * 60)
        print("Triangle Slide Experiment")
        print("=" * 60)
        print(f"Triangle edge length: {TRIANGLE_EDGE_LENGTH} m")
        print(f"Plane tilt angle: {PLANE_TILT_ANGLE_DEG} degrees")
        print(f"Drop height: {DROP_HEIGHT} m")
        print(f"Start position: ({START_X:.3f}, 0.0, {START_Z:.3f}) m")
        print(f"Tilted plane friction (mu): {PLANE_MU}")
        print(f"Ground plane friction (mu): {GROUND_MU}")
        print(f"Cloth contact friction (mu): {CONTACT_MU}")
        print("=" * 60)

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
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.collision_pipeline.collide(self.model, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

        # Measure slide time
        self._check_slide_complete()

    def _check_slide_complete(self):
        """Check if the triangle has reached the ground and come to rest."""
        # Get particle positions and velocities
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()

        # Compute average position and velocity
        avg_pos = particle_q.mean(axis=0)
        avg_z = avg_pos[2]
        avg_velocity = particle_qd.mean(axis=0)
        speed = (avg_velocity[0] ** 2 + avg_velocity[1] ** 2 + avg_velocity[2] ** 2) ** 0.5

        # Check if triangle reached ground level
        if not self.reached_ground and avg_z < GROUND_THRESHOLD:
            self.reached_ground = True
            self.ground_arrival_time = self.sim_time
            self.ground_arrival_position = avg_pos.copy()
            print("=" * 60)
            print("REACHED GROUND!")
            print(f"Time to reach ground: {self.ground_arrival_time:.3f} seconds")
            print(f"Position at ground arrival: ({avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.4f}) m")
            print("=" * 60)

        # Check if triangle has come to rest on the ground
        if self.reached_ground and not self.rest_complete and speed < REST_VELOCITY_THRESHOLD:
            self.rest_complete = True
            self.rest_time = self.sim_time
            self.rest_position = avg_pos.copy()

            # Calculate ground slide distance (XY distance from arrival to rest)
            dx = self.rest_position[0] - self.ground_arrival_position[0]
            dy = self.rest_position[1] - self.ground_arrival_position[1]
            self.ground_slide_distance = (dx**2 + dy**2) ** 0.5

            # Also mark slide_complete for backward compatibility
            self.slide_complete = True
            self.slide_time = self.rest_time

            print("=" * 60)
            print("TRIANGLE AT REST!")
            print(f"Time to come to rest: {self.rest_time:.3f} seconds")
            print(f"Time sliding on ground: {self.rest_time - self.ground_arrival_time:.3f} seconds")
            print(f"Final position: ({avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.4f}) m")
            print(f"Distance slid on ground: {self.ground_slide_distance:.4f} m")
            print("=" * 60)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Verify triangle reached the ground
        particle_q = self.state_0.particle_q.numpy()
        z_positions = particle_q[:, 2]
        avg_z = z_positions.mean()

        if not self.slide_complete:
            print(f"Warning: Triangle did not reach ground. Current avg Z: {avg_z:.4f} m")

        newton.examples.test_particle_state(
            self.state_0,
            "particles have come close to a rest",
            lambda q, qd: max(abs(qd)) < 0.5,
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)
