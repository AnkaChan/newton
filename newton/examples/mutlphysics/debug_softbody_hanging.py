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


# Debug config - simple cube mesh for fast iteration
config_debug = {
    # Timing parameters
    "fps": 60,
    "sim_substeps": 10,
    # Soft body mesh (None = use built-in cube)
    "softbody_vtk_path": None,
    "softbody_scale": 0.2,
    "softbody_position": (0.0, 0.0, 0.5),  # m - drop height
    # Soft body material (meters, kg)
    "softbody_density": 1000.0,  # kg/m³
    "softbody_k_mu": 1.0e4,
    "softbody_k_lambda": 1.0e4,
    "softbody_k_damp": 1e-4,
    "softbody_particle_radius": 0.02,  # m
    # Contact parameters
    "soft_contact_ke": 1.0e4,
    "soft_contact_kd": 1e-3,
    "soft_contact_mu": 0.5,
    "soft_contact_margin": 0.02,  # m
    # Ground plane
    "ground_height": 0.0,  # m
    # Gravity (m/s²)
    "gravity": -9.81,
    # Model builder
    "up_axis": "z",
    "include_bending": False,
    # VBD Solver parameters
    "iterations": 5,
    "particle_enable_self_contact": False,
    "particle_enable_tile_solve": False,
    # Visualization
    "ground_size": 2.0,  # m
}

# Full config - bunny mesh
config_bunny = {
    # Timing parameters
    "fps": 60,
    "sim_substeps": 10,
    # Soft body mesh
    "softbody_vtk_path": "bunny_small.vtk",
    "softbody_scale": 1.0,
    "softbody_position": (0.0, 0.0, 1.0),  # m - drop height
    # Soft body material (meters, kg)
    "softbody_density": 1000.0,  # kg/m³
    "softbody_k_mu": 1.0e5,
    "softbody_k_lambda": 1.0e5,
    "softbody_k_damp": 1e-6,
    "softbody_particle_radius": 0.01,  # m
    # Contact parameters
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 1e-5,
    "soft_contact_mu": 0.2,
    "soft_contact_margin": 0.01,  # m
    # Ground plane
    "ground_height": -0.5,  # m
    # Gravity (m/s²)
    "gravity": -9.81,
    # Model builder
    "up_axis": "z",
    "include_bending": False,
    # VBD Solver parameters
    "iterations": 10,
    "particle_enable_self_contact": False,
    "particle_enable_tile_solve": False,
    # Visualization
    "ground_size": 2.0,  # m
}

# Select which config to use
config = config_bunny  # Change to config_debug for simple cube mesh


# Built-in simple cube tet mesh (18 particles, 20 tets)
CUBE_PARTICLES = [
    (0.0, 0.0, 0.0),  # 0
    (1.0, 0.0, 0.0),  # 1
    (2.0, 0.0, 0.0),  # 2
    (0.0, 1.0, 0.0),  # 3
    (1.0, 1.0, 0.0),  # 4
    (2.0, 1.0, 0.0),  # 5
    (0.0, 2.0, 0.0),  # 6
    (1.0, 2.0, 0.0),  # 7
    (2.0, 2.0, 0.0),  # 8
    (0.0, 0.0, 1.0),  # 9
    (1.0, 0.0, 1.0),  # 10
    (2.0, 0.0, 1.0),  # 11
    (0.0, 1.0, 1.0),  # 12
    (1.0, 1.0, 1.0),  # 13
    (2.0, 1.0, 1.0),  # 14
    (0.0, 2.0, 1.0),  # 15
    (1.0, 2.0, 1.0),  # 16
    (2.0, 2.0, 1.0),  # 17
]

CUBE_TET_INDICES = np.array(
    [
        [0, 1, 3, 9],
        [1, 4, 3, 13],
        [1, 3, 9, 13],
        [3, 9, 13, 12],
        [1, 9, 10, 13],
        [1, 2, 4, 10],
        [2, 5, 4, 14],
        [2, 4, 10, 14],
        [4, 10, 14, 13],
        [2, 10, 11, 14],
        [3, 4, 6, 12],
        [4, 7, 6, 16],
        [4, 6, 12, 16],
        [6, 12, 16, 15],
        [4, 12, 13, 16],
        [4, 5, 7, 13],
        [5, 8, 7, 17],
        [5, 7, 13, 17],
        [7, 13, 17, 16],
        [5, 13, 14, 17],
    ],
    dtype=np.int32,
)


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
    def __init__(self, cfg: dict):
        self.config = cfg
        self.sim_time = 0.0
        self.frame_count = 0
        self.is_paused = False

        # Timing
        self.fps = cfg["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = cfg["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Load mesh (VTK file or built-in cube)
        softbody_path = cfg["softbody_vtk_path"]
        if softbody_path is None:
            # Use built-in cube mesh
            softbody_vertices = CUBE_PARTICLES
            softbody_indices = CUBE_TET_INDICES.flatten().tolist()
            print(f"Using built-in cube: {len(CUBE_PARTICLES)} vertices, {len(CUBE_TET_INDICES)} tetrahedra")
        else:
            if not os.path.isabs(softbody_path):
                softbody_path = os.path.join(os.path.dirname(__file__), softbody_path)
            softbody_verts, softbody_tets = load_vtk_tet_mesh(softbody_path)
            print(f"Loaded soft body: {len(softbody_verts)} vertices, {len(softbody_tets)} tetrahedra")
            softbody_vertices = [(v[0], v[1], v[2]) for v in softbody_verts]
            softbody_indices = softbody_tets.flatten().tolist()

        builder = newton.ModelBuilder(up_axis=cfg["up_axis"], gravity=cfg["gravity"])

        # Add ground plane
        builder.add_ground_plane(height=cfg["ground_height"])

        # Add soft body
        pos = cfg["softbody_position"]
        builder.add_soft_mesh(
            pos=wp.vec3(pos[0], pos[1], pos[2]),
            rot=wp.quat_identity(),
            scale=cfg["softbody_scale"],
            vel=wp.vec3(0.0),
            vertices=softbody_vertices,
            indices=softbody_indices,
            density=cfg["softbody_density"],
            k_mu=cfg["softbody_k_mu"],
            k_lambda=cfg["softbody_k_lambda"],
            k_damp=cfg["softbody_k_damp"],
            particle_radius=cfg["softbody_particle_radius"],
        )
        builder.color(include_bending=cfg["include_bending"])

        self.model = builder.finalize()

        # Set contact parameters
        self.model.soft_contact_ke = cfg["soft_contact_ke"]
        self.model.soft_contact_kd = cfg["soft_contact_kd"]
        self.model.soft_contact_mu = cfg["soft_contact_mu"]

        # Print particle info for debugging
        particle_mass = self.model.particle_mass.numpy()
        print(f"Particle count: {self.model.particle_count}")
        print(f"Particle mass range: {particle_mass.min():.6f} to {particle_mass.max():.6f}")
        print(f"Total mass: {particle_mass.sum():.6f}")

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=cfg["iterations"],
            particle_enable_self_contact=cfg["particle_enable_self_contact"],
            particle_enable_tile_solve=cfg["particle_enable_tile_solve"],
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, soft_contact_margin=cfg["soft_contact_margin"])

        # Get triangle faces for visualization
        self.faces = self.model.tri_indices.numpy()

        # Setup polyscope
        self.setup_polyscope()

    def setup_polyscope(self):
        """Initialize polyscope and register meshes."""
        cfg = self.config
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height_factor(cfg["ground_height"])

        # Register soft body mesh
        verts = self.state_0.particle_q.numpy()
        self.ps_mesh = ps.register_surface_mesh("SoftBody", verts, self.faces)
        self.ps_mesh.set_color((0.8, 0.5, 0.3))  # Orange-ish
        self.ps_mesh.set_smooth_shade(True)

        # Add ground plane visualization
        ground_size = cfg["ground_size"]
        ground_height = cfg["ground_height"]
        ground_verts = np.array([
            [-ground_size, -ground_size, ground_height],
            [ground_size, -ground_size, ground_height],
            [ground_size, ground_size, ground_height],
            [-ground_size, ground_size, ground_height],
        ], dtype=np.float32)
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.ps_ground = ps.register_surface_mesh("Ground", ground_verts, ground_faces)
        self.ps_ground.set_color((0.5, 0.5, 0.5))
        self.ps_ground.set_transparency(0.5)

    def run_step(self):
        """Execute one frame of simulation."""
        cfg = self.config
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.contacts = self.model.collide(self.state_0, soft_contact_margin=cfg["soft_contact_margin"])
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
    sim = DebugSoftbody(config)
    sim.simulate()


if __name__ == "__main__":
    main()
