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
# Example: Crusher Simulation
#
# Two counter-rotating gear cylinders crush a soft body (bunny).
# The rollers are rigid bodies controlled externally via
# integrate_with_external_rigid_solver=True.
#
# Command:
#   Newton viewer:   python -m newton.examples.mutlphysics.example_crusher_simulation
#   Polyscope:       python -m newton.examples.mutlphysics.example_crusher_simulation --viewer polyscope
#
###########################################################################

import os

import numpy as np
import warp as wp

import newton
import newton.examples


def load_vtk_tet_mesh(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a tetrahedral mesh from a VTK file.

    Args:
        filepath: Path to the VTK file.

    Returns:
        Tuple of (vertices, tet_indices) as numpy arrays.
    """
    vertices = []
    tet_indices = []

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    num_points = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("POINTS"):
            parts = line.split()
            num_points = int(parts[1])
            i += 1
            while len(vertices) < num_points and i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1
            continue

        elif line.startswith("CELLS"):
            parts = line.split()
            num_cells = int(parts[1])
            i += 1
            while len(tet_indices) < num_cells and i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 5 and parts[0] == "4":
                    tet_indices.append([int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])])
                i += 1
            continue

        i += 1

    vertices = np.array(vertices, dtype=np.float32)
    tet_indices = np.array(tet_indices, dtype=np.int32)

    return vertices, tet_indices


def create_gear_cylinder_mesh(
    inner_radius: float = 1.0,
    outer_radius: float = 1.5,
    length: float = 10.0,
    num_teeth: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a cylinder with gear teeth on its surface (extruded gear profile).
    The teeth run along the length of the cylinder.

    Args:
        inner_radius: Radius to the valleys between teeth.
        outer_radius: Radius to the tooth tips.
        length: Length of the cylinder.
        num_teeth: Number of teeth around the circumference.

    Returns:
        Tuple of (vertices, faces) as numpy arrays.
    """
    vertices = []
    faces = []

    half_len = length / 2.0

    # Create gear profile (cross-section) - thick teeth, narrow valleys
    profile = []
    for i in range(num_teeth):
        tooth_angle = 2 * np.pi / num_teeth
        base_angle = i * tooth_angle

        # Valley (narrow)
        profile.append((base_angle, inner_radius))
        # Rising edge (quick rise)
        profile.append((base_angle + tooth_angle * 0.08, inner_radius))
        # Tooth tip start (wide tooth)
        profile.append((base_angle + tooth_angle * 0.15, outer_radius))
        # Tooth tip end (wide tooth)
        profile.append((base_angle + tooth_angle * 0.85, outer_radius))
        # Falling edge (quick fall)
        profile.append((base_angle + tooth_angle * 0.92, inner_radius))

    num_profile = len(profile)

    # Create vertices for left and right ends
    left_start = len(vertices)
    for angle, radius in profile:
        y = radius * np.cos(angle)
        z = radius * np.sin(angle)
        vertices.append([-half_len, y, z])

    right_start = len(vertices)
    for angle, radius in profile:
        y = radius * np.cos(angle)
        z = radius * np.sin(angle)
        vertices.append([half_len, y, z])

    # Side faces (connect left and right profiles)
    for i in range(num_profile):
        i_next = (i + 1) % num_profile
        faces.append([left_start + i, left_start + i_next, right_start + i_next])
        faces.append([left_start + i, right_start + i_next, right_start + i])

    # End caps - left (center point + fan)
    left_center = len(vertices)
    vertices.append([-half_len, 0, 0])
    for i in range(num_profile):
        i_next = (i + 1) % num_profile
        faces.append([left_center, left_start + i_next, left_start + i])

    # End caps - right
    right_center = len(vertices)
    vertices.append([half_len, 0, 0])
    for i in range(num_profile):
        i_next = (i + 1) % num_profile
        faces.append([right_center, right_start + i, right_start + i_next])

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    return vertices, faces


# =============================================================================
# Configuration - Single Source of Truth (all units in meters, kg, seconds)
# =============================================================================
config = {
    # Viewer: "newton" or "polyscope"
    "viewer": "polyscope",
    # Timing
    "fps": 60,
    "sim_substeps": 10,
    "iterations": 10,
    # Physics
    "up_axis": "z",
    "gravity": -9.81,  # m/s²
    "ground_height": -1.5,  # m
    "include_bending": False,
    # Roller parameters (m)
    "roller_inner_radius": 0.36,  # m
    "roller_outer_radius": 0.40,  # m
    "roller_length": 1.6,  # m
    "roller_num_teeth": 16,
    "roller_gap": 0.08,  # m
    "roller_rotation_speed": 1.0,  # rad/s
    "roller_ke": 1.0e6,
    "roller_kd": 1e-7,
    "roller_mu": 0.2,
    # Soft body
    "softbody_vtk_path": "Armadilo_15K.1.vtk",
    "softbody_position": (0.0, 0.0, 1.0),  # m
    "softbody_rotation_axis": (1.0, 0.0, 0.0),  # rotate around X
    "softbody_rotation_angle": 1.5708,  # 90 degrees in radians (π/2)
    "softbody_scale": 1.,  # m
    "softbody_density": 1000.0,  # kg/m³
    "softbody_k_mu": 1.0e5,
    "softbody_k_lambda": 1.0e6,
    "softbody_k_damp": 1e-7,
    "softbody_particle_radius": 0.005,  # m
    # Contact parameters
    "soft_contact_ke": 1.0e6,
    "soft_contact_kd": 1e-7,
    "soft_contact_mu": 0.2,
    "soft_contact_max": 64 * 1024,
    "soft_contact_margin": 0.01,  # m
    # VBD Solver - self contact
    "particle_enable_self_contact": True,
    "particle_self_contact_radius": 0.005,  # m
    "particle_self_contact_margin": 0.006,  # m
    "particle_topological_contact_filter_threshold": 1,
    "particle_rest_shape_contact_exclusion_radius": 0.02,  # m
    "particle_enable_tile_solve": True,
    "particle_vertex_contact_buffer_size": 16,
    "particle_edge_contact_buffer_size": 32,
    # Output
    "output_path": r"D:\Data\DAT_Sim\Crusher",
    "write_output": True,
    "sim_num_frames": 600,
}


class Example:
    """
    Crusher simulation with two counter-rotating gear rollers crushing a soft body.
    Supports both Newton's built-in viewer and Polyscope for visualization.
    """

    def __init__(self, viewer=None, cfg: dict = None):
        self.cfg = cfg if cfg is not None else config
        self.viewer = viewer
        self.sim_time = 0.0
        self.use_polyscope = self.cfg.get("viewer", "newton") == "polyscope"
        self.paused = False
        self.frame_count = 0

        # Output setup with timestamped subfolder
        self.write_output = self.cfg.get("write_output", False)
        base_output_path = self.cfg.get("output_path", "")
        if self.write_output and base_output_path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_path = os.path.join(base_output_path, timestamp)
            os.makedirs(self.output_path, exist_ok=True)
            print(f"Output will be saved to: {self.output_path}")
        else:
            self.output_path = ""

        # Timing
        self.fps = self.cfg["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.cfg["sim_substeps"]
        self.iterations = self.cfg["iterations"]
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Roller parameters
        inner_radius = self.cfg["roller_inner_radius"]
        outer_radius = self.cfg["roller_outer_radius"]
        roller_length = self.cfg["roller_length"]
        num_teeth = self.cfg["roller_num_teeth"]
        roller_gap = self.cfg["roller_gap"]

        # Calculate roller separation (distance between centers)
        self.roller_separation = 2 * outer_radius + roller_gap

        # Rotation state
        self.roller_angle1 = 0.0
        self.roller_angle2 = 0.0
        self.rotation_speed = self.cfg["roller_rotation_speed"]

        # Build model
        builder = newton.ModelBuilder(up_axis=self.cfg["up_axis"])
        builder.gravity = self.cfg["gravity"]

        # Add ground plane
        builder.add_ground_plane(height=self.cfg["ground_height"])

        # Create gear cylinder mesh
        roller_verts, roller_faces = create_gear_cylinder_mesh(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            length=roller_length,
            num_teeth=num_teeth,
        )

        # Store for polyscope visualization
        self.roller_verts = roller_verts
        self.roller_faces = roller_faces

        # Create Newton mesh for rollers
        roller_mesh = newton.Mesh(
            vertices=roller_verts,
            indices=roller_faces.flatten(),
        )

        # Roller shape config
        roller_cfg = newton.ModelBuilder.ShapeConfig()
        roller_cfg.ke = self.cfg["roller_ke"]
        roller_cfg.kd = self.cfg["roller_kd"]
        roller_cfg.mu = self.cfg["roller_mu"]
        roller_cfg.density = 0.0  # Kinematic (infinite mass)

        # Add roller 1 (bottom, at -Y)
        self.body_roller1 = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, -self.roller_separation / 2, 0.0),
                q=wp.quat_identity(),
            ),
            key="roller_1",
        )
        builder.add_shape_mesh(
            self.body_roller1,
            mesh=roller_mesh,
            cfg=roller_cfg,
        )

        # Add roller 2 (top, at +Y)
        self.body_roller2 = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, self.roller_separation / 2, 0.0),
                q=wp.quat_identity(),
            ),
            key="roller_2",
        )
        builder.add_shape_mesh(
            self.body_roller2,
            mesh=roller_mesh,
            cfg=roller_cfg,
        )

        # Load soft body mesh
        softbody_path = self.cfg["softbody_vtk_path"]
        if not os.path.isabs(softbody_path):
            softbody_path = os.path.join(os.path.dirname(__file__), softbody_path)
        softbody_verts, softbody_tets = load_vtk_tet_mesh(softbody_path)
        print(f"Loaded soft body: {len(softbody_verts)} vertices, {len(softbody_tets)} tetrahedra")

        # Print bounding box
        bbox_min = softbody_verts.min(axis=0)
        bbox_max = softbody_verts.max(axis=0)
        bbox_size = bbox_max - bbox_min
        scale = self.cfg["softbody_scale"]
        pos = self.cfg["softbody_position"]
        print(f"  Raw bbox: min={bbox_min}, max={bbox_max}, size={bbox_size}")
        print(f"  Scaled bbox: min={bbox_min * scale + pos}, max={bbox_max * scale + pos}, size={bbox_size * scale}")

        # Convert to format expected by add_soft_mesh
        softbody_vertices = [(v[0], v[1], v[2]) for v in softbody_verts]
        softbody_indices = softbody_tets.flatten().tolist()

        # Add soft body
        pos = self.cfg["softbody_position"]
        rot_axis = self.cfg.get("softbody_rotation_axis", (0.0, 0.0, 1.0))
        rot_angle = self.cfg.get("softbody_rotation_angle", 0.0)
        rot = wp.quat_from_axis_angle(wp.vec3(rot_axis[0], rot_axis[1], rot_axis[2]), rot_angle)
        builder.add_soft_mesh(
            pos=wp.vec3(pos[0], pos[1], pos[2]),
            rot=rot,
            scale=self.cfg["softbody_scale"],
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=softbody_vertices,
            indices=softbody_indices,
            density=self.cfg["softbody_density"],
            k_mu=self.cfg["softbody_k_mu"],
            k_lambda=self.cfg["softbody_k_lambda"],
            k_damp=self.cfg["softbody_k_damp"],
            particle_radius=self.cfg["softbody_particle_radius"],
        )

        # Color the mesh for VBD solver
        builder.color(include_bending=self.cfg["include_bending"])

        # Finalize the model
        self.model = builder.finalize()

        # Store triangle faces for visualization
        self.tri_faces = self.model.tri_indices.numpy()

        # Set contact parameters
        self.model.soft_contact_ke = self.cfg["soft_contact_ke"]
        self.model.soft_contact_kd = self.cfg["soft_contact_kd"]
        self.model.soft_contact_mu = self.cfg["soft_contact_mu"]
        self.model.soft_contact_max = self.cfg["soft_contact_max"]

        # Create the VBD solver with external rigid body integration
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=self.cfg["particle_enable_self_contact"],
            particle_self_contact_radius=self.cfg["particle_self_contact_radius"],
            particle_self_contact_margin=self.cfg["particle_self_contact_margin"],
            particle_topological_contact_filter_threshold=self.cfg["particle_topological_contact_filter_threshold"],
            particle_rest_shape_contact_exclusion_radius=self.cfg["particle_rest_shape_contact_exclusion_radius"],
            particle_enable_tile_solve=self.cfg["particle_enable_tile_solve"],
            particle_vertex_contact_buffer_size=self.cfg["particle_vertex_contact_buffer_size"],
            particle_edge_contact_buffer_size=self.cfg["particle_edge_contact_buffer_size"],
        )

        # Create simulation states and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Set up visualization
        if self.use_polyscope:
            self._setup_polyscope()
        else:
            self.viewer.set_model(self.model)

        # Save static mesh data and initial state
        if self.write_output and self.output_path:
            # Static mesh data
            np.save(os.path.join(self.output_path, "roller_verts.npy"), self.roller_verts)
            np.save(os.path.join(self.output_path, "roller_faces.npy"), self.roller_faces)
            np.save(os.path.join(self.output_path, "tri_faces.npy"), self.tri_faces)
            # Initial particle positions
            np.save(os.path.join(self.output_path, "particles_initial.npy"), self.state_0.particle_q.numpy())
            np.save(os.path.join(self.output_path, "body_q_initial.npy"), self.state_0.body_q.numpy())
            # Metadata (fps, num_frames, etc.)
            metadata = {
                "fps": self.fps,
                "sim_substeps": self.sim_substeps,
                "sim_num_frames": self.cfg.get("sim_num_frames", 600),
                "frame_dt": self.frame_dt,
                "num_particles": self.model.particle_count,
                "num_bodies": self.model.body_count,
            }
            np.save(os.path.join(self.output_path, "metadata.npy"), metadata)
            print(f"Saved static mesh data and initial state to {self.output_path}")

    def _setup_polyscope(self):
        """Initialize Polyscope visualization."""
        import polyscope as ps
        import polyscope.imgui as psim

        self.ps = ps
        self.psim = psim

        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height_factor(self.cfg["ground_height"])

        # Register soft body mesh
        verts = self.state_0.particle_q.numpy()
        self.ps_softbody = ps.register_surface_mesh("SoftBody", verts, self.tri_faces)
        self.ps_softbody.set_color((0.8, 0.5, 0.3))
        self.ps_softbody.set_smooth_shade(True)

        # Register roller meshes
        body_q = self.state_0.body_q.numpy()

        transform1 = body_q[self.body_roller1]
        roller1_world = self._transform_vertices(self.roller_verts, transform1[:3], transform1[3:7])
        self.ps_roller1 = ps.register_surface_mesh("Roller_1", roller1_world, self.roller_faces)
        self.ps_roller1.set_color((0.5, 0.5, 0.55))
        self.ps_roller1.set_smooth_shade(False)

        transform2 = body_q[self.body_roller2]
        roller2_world = self._transform_vertices(self.roller_verts, transform2[:3], transform2[3:7])
        self.ps_roller2 = ps.register_surface_mesh("Roller_2", roller2_world, self.roller_faces)
        self.ps_roller2.set_color((0.5, 0.5, 0.55))
        self.ps_roller2.set_smooth_shade(False)

    def _transform_vertices(self, verts, pos, quat):
        """Transform vertices by quaternion rotation and translation."""
        qx, qy, qz, qw = quat

        # Build rotation matrix from quaternion
        rot_mat = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )

        transformed = verts @ rot_mat.T + pos
        return transformed.astype(np.float32)

    def update_roller_transforms(self):
        """Update roller positions based on rotation angles.

        For correct friction with integrate_with_external_rigid_solver=True:
        - state_0.body_q = previous state (where body WAS)
        - state_1.body_q = current state (where body IS NOW)
        The solver computes body velocity from (state_1 - state_0) / dt.
        """

        # Create quaternion for rotation around X axis
        def quat_from_angle_x(angle):
            half_angle = angle / 2.0
            return np.array([np.sin(half_angle), 0.0, 0.0, np.cos(half_angle)], dtype=np.float32)

        # Set state_0 to PREVIOUS transforms (current angles before update)
        body_q_prev = self.state_0.body_q.numpy()
        q1_prev = quat_from_angle_x(self.roller_angle1)
        body_q_prev[self.body_roller1, :3] = [0.0, -self.roller_separation / 2, 0.0]
        body_q_prev[self.body_roller1, 3:7] = q1_prev
        q2_prev = quat_from_angle_x(self.roller_angle2)
        body_q_prev[self.body_roller2, :3] = [0.0, self.roller_separation / 2, 0.0]
        body_q_prev[self.body_roller2, 3:7] = q2_prev
        self.state_0.body_q.assign(body_q_prev)

        # Update angles (counter-rotating)
        self.roller_angle1 -= self.rotation_speed * self.sim_dt
        self.roller_angle2 += self.rotation_speed * self.sim_dt

        # Set state_1 to CURRENT transforms (new angles after update)
        body_q_curr = self.state_1.body_q.numpy()
        q1_curr = quat_from_angle_x(self.roller_angle1)
        body_q_curr[self.body_roller1, :3] = [0.0, -self.roller_separation / 2, 0.0]
        body_q_curr[self.body_roller1, 3:7] = q1_curr
        q2_curr = quat_from_angle_x(self.roller_angle2)
        body_q_curr[self.body_roller2, :3] = [0.0, self.roller_separation / 2, 0.0]
        body_q_curr[self.body_roller2, 3:7] = q2_curr
        self.state_1.body_q.assign(body_q_curr)

    def step(self):
        """Execute one frame with external rigid body updates."""
        if self.paused:
            return

        for _ in range(self.sim_substeps):
            # Update roller rotations
            self.update_roller_transforms()

            # Rebuild BVH after body transforms change
            self.solver.rebuild_bvh(self.state_0)

            # Run collision detection
            if self.model.shape_count:
                self.contacts = self.model.collide(
                    self.state_0,
                    soft_contact_margin=self.cfg["soft_contact_margin"],
                )

            self.state_0.clear_forces()

            # Apply viewer forces (for interactive dragging) - Newton viewer only
            if not self.use_polyscope:
                self.viewer.apply_forces(self.state_0)

            # Step simulation
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

        # Save output
        if self.write_output:
            self.save_output()
        self.frame_count += 1

    def save_output(self):
        """Save particle positions and roller transforms to npy files."""
        if not self.output_path:
            return

        # Save particle positions
        particle_q = self.state_0.particle_q.numpy()
        particle_file = os.path.join(self.output_path, f"particles_{self.frame_count:06d}.npy")
        np.save(particle_file, particle_q)

        # Save roller transforms (body_q contains [px, py, pz, qx, qy, qz, qw] for each body)
        body_q = self.state_0.body_q.numpy()
        body_file = os.path.join(self.output_path, f"body_q_{self.frame_count:06d}.npy")
        np.save(body_file, body_q)


        if self.frame_count % 60 == 0:
            print(f"Saved frame {self.frame_count}, time={self.sim_time:.3f}s")

    def render(self):
        """Render current state."""
        if self.use_polyscope:
            self._render_polyscope()
        else:
            self._render_newton()

    def _render_newton(self):
        """Render using Newton's viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _render_polyscope(self):
        """Render using Polyscope."""
        # Update soft body mesh
        verts = self.state_0.particle_q.numpy()
        self.ps_softbody.update_vertex_positions(verts)

        # Update roller meshes
        body_q = self.state_0.body_q.numpy()

        transform1 = body_q[self.body_roller1]
        roller1_world = self._transform_vertices(self.roller_verts, transform1[:3], transform1[3:7])
        self.ps_roller1.update_vertex_positions(roller1_world)

        transform2 = body_q[self.body_roller2]
        roller2_world = self._transform_vertices(self.roller_verts, transform2[:3], transform2[3:7])
        self.ps_roller2.update_vertex_positions(roller2_world)

    def test_final(self):
        """Test that simulation completed reasonably."""
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] > -1.0,
        )


def run_polyscope(example):
    """Run simulation loop with Polyscope."""
    import polyscope as ps
    import polyscope.imgui as psim

    max_frames = example.cfg.get("sim_num_frames", 600)

    def callback():
        # UI controls
        changed, example.paused = psim.Checkbox("Paused", example.paused)

        psim.TextUnformatted(f"Frame: {example.frame_count} / {max_frames}")
        psim.TextUnformatted(f"Time: {example.sim_time:.3f}s")

        # Get particle positions for info display
        verts = example.state_0.particle_q.numpy()
        psim.TextUnformatted(f"Min Z: {verts[:, 2].min():.4f}")
        psim.TextUnformatted(f"Max Z: {verts[:, 2].max():.4f}")

        # Check if simulation is complete
        if example.frame_count >= max_frames:
            example.paused = True
            psim.TextUnformatted("SIMULATION COMPLETE")
            return

        # Step and render
        example.step()
        example.render()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    if config["viewer"] == "polyscope":
        # Use Polyscope
        example = Example(cfg=config)
        run_polyscope(example)
    else:
        # Use Newton's viewer
        parser = newton.examples.create_parser()
        viewer, args = newton.examples.init(parser)
        example = Example(viewer=viewer, cfg=config)
        newton.examples.run(example, args)
