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
# Debug Softbody with Bunny
#
# This simulation uses the bunny mesh for debugging softbody physics.
# Uses polyscope for visualization. Units: meters.
#
# Command: python -m newton.examples.mutlphysics.debug_softbody_hanging
#
###########################################################################

import os

import numpy as np
import polyscope as ps
import warp as wp

import newton


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


class DebugSoftbody:
    def __init__(self):
        self.sim_time = 0.0
        self.frame_count = 0
        self.is_paused = False

        # Timing parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Soft body parameters (meters)
        self.softbody_scale = 1.0
        self.softbody_position = (0.0, 0.0, 1.0)  # m - drop height
        self.softbody_density = 1000.0  # kg/m³
        self.softbody_k_mu = 1.0e5
        self.softbody_k_lambda = 1.0e6
        self.softbody_k_damp = 1e-6
        self.softbody_particle_radius = 0.01  # m

        # Contact parameters
        self.soft_contact_ke = 1.0e5
        self.soft_contact_kd = 1e-5
        self.soft_contact_mu = 0.2
        self.soft_contact_margin = 0.01  # m

        # Ground plane
        self.ground_height = -0.5  # m

        # Gravity (m/s²)
        self.gravity = -9.81

        # Load bunny mesh
        softbody_path = os.path.join(os.path.dirname(__file__), "bunny_small.vtk")
        softbody_verts, softbody_tets = load_vtk_tet_mesh(softbody_path)
        print(f"Loaded soft body: {len(softbody_verts)} vertices, {len(softbody_tets)} tetrahedra")

        # Convert to format expected by add_soft_mesh
        softbody_vertices = [(v[0], v[1], v[2]) for v in softbody_verts]
        softbody_indices = softbody_tets.flatten().tolist()

        builder = newton.ModelBuilder(up_axis="z", gravity=self.gravity)

        # Add ground plane at specified height
        builder.add_ground_plane(height=self.ground_height)

        builder.add_soft_mesh(
            pos=wp.vec3(self.softbody_position[0], self.softbody_position[1], self.softbody_position[2]),
            rot=wp.quat_identity(),
            scale=self.softbody_scale,
            vel=wp.vec3(0.0),
            vertices=softbody_vertices,
            indices=softbody_indices,
            density=self.softbody_density,
            k_mu=self.softbody_k_mu,
            k_lambda=self.softbody_k_lambda,
            k_damp=self.softbody_k_damp,
            particle_radius=self.softbody_particle_radius,
        )
        builder.color(include_bending=False)

        self.model = builder.finalize()

        # Set contact parameters
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.soft_contact_mu

        # Print particle info for debugging
        particle_mass = self.model.particle_mass.numpy()
        print(f"Particle count: {self.model.particle_count}")
        print(f"Particle mass range: {particle_mass.min():.6f} to {particle_mass.max():.6f}")
        print(f"Total mass: {particle_mass.sum():.6f}")

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.soft_contact_margin)

        # Get triangle faces for visualization
        self.faces = self.model.tri_indices.numpy()

        # Setup polyscope
        self.setup_polyscope()

    def setup_polyscope(self):
        """Initialize polyscope and register meshes."""
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height_factor(self.ground_height)

        # Register soft body mesh
        verts = self.state_0.particle_q.numpy()
        self.ps_mesh = ps.register_surface_mesh("SoftBody", verts, self.faces)
        self.ps_mesh.set_color((0.8, 0.5, 0.3))  # Orange-ish
        self.ps_mesh.set_smooth_shade(True)

        # Add ground plane visualization
        ground_size = 2.0  # m
        ground_verts = np.array([
            [-ground_size, -ground_size, self.ground_height],
            [ground_size, -ground_size, self.ground_height],
            [ground_size, ground_size, self.ground_height],
            [-ground_size, ground_size, self.ground_height],
        ], dtype=np.float32)
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.ps_ground = ps.register_surface_mesh("Ground", ground_verts, ground_faces)
        self.ps_ground.set_color((0.5, 0.5, 0.5))
        self.ps_ground.set_transparency(0.5)

    def run_step(self):
        """Execute one frame of simulation."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.soft_contact_margin)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt
        self.frame_count += 1

    def update_visualization(self):
        """Update polyscope meshes with current state."""
        verts = self.state_0.particle_q.numpy()
        self.ps_mesh.update_vertex_positions(verts)

    def callback(self):
        """Polyscope callback for each frame."""
        import polyscope.imgui as psim

        # Control panel
        changed, self.is_paused = psim.Checkbox("Paused", self.is_paused)

        psim.TextUnformatted(f"Frame: {self.frame_count}")
        psim.TextUnformatted(f"Time: {self.sim_time:.3f}s")

        # Get particle positions for debug info
        verts = self.state_0.particle_q.numpy()
        psim.TextUnformatted(f"Min Z: {verts[:, 2].min():.3f} m")
        psim.TextUnformatted(f"Max Z: {verts[:, 2].max():.3f} m")

        if not self.is_paused:
            self.run_step()
            self.update_visualization()

    def simulate(self):
        """Run the simulation with polyscope visualization."""
        ps.set_user_callback(self.callback)
        ps.show()


def main():
    """Main entry point."""
    sim = DebugSoftbody()
    sim.simulate()


if __name__ == "__main__":
    main()
