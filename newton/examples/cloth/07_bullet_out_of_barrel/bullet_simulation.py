# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Bullet Out of Barrel Demo 🎉

A soft body bullet being pushed out of a rifled barrel by a massive force!
Everything in centimeters.
"""

import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent.parent))
from M01_Simulator import Simulator, default_config, get_config_value


@wp.kernel
def apply_force_to_particles(
    particle_f: wp.array(dtype=wp.vec3),
    force_indices: wp.array(dtype=wp.int32),
    force_magnitudes: wp.array(dtype=wp.float32),
    force_direction: wp.vec3,
):
    """Apply a force to specified particle indices, proportional to their volume."""
    tid = wp.tid()
    idx = force_indices[tid]
    force = force_direction * force_magnitudes[tid]
    wp.atomic_add(particle_f, idx, force)


def load_tetmesh_npz(filepath):
    """Load tetrahedral mesh from NPZ file."""
    data = np.load(filepath)
    vertices = data["vertices"]
    tetrahedra = data["tetrahedra"]
    print(f"Loaded tetmesh: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra")
    return vertices, tetrahedra


def compute_tet_volume(v0, v1, v2, v3):
    """Compute volume of a tetrahedron given 4 vertices."""
    # Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
    a = v1 - v0
    b = v2 - v0
    c = v3 - v0
    return abs(np.dot(a, np.cross(b, c))) / 6.0


def compute_vertex_volumes(vertices, tetrahedra):
    """Compute the volume associated with each vertex (1/4 of adjacent tet volumes)."""
    vertex_volumes = np.zeros(len(vertices))

    for tet in tetrahedra:
        v0, v1, v2, v3 = vertices[tet[0]], vertices[tet[1]], vertices[tet[2]], vertices[tet[3]]
        tet_volume = compute_tet_volume(v0, v1, v2, v3)
        # Each vertex gets 1/4 of the tet volume
        for idx in tet:
            vertex_volumes[idx] += tet_volume / 4.0

    return vertex_volumes


def load_ply_mesh(filepath):
    """Load surface mesh from PLY file."""
    vertices = []
    faces = []

    with open(filepath) as f:
        line = f.readline().strip()
        assert line == "ply", f"Not a PLY file: {filepath}"

        vertex_count = 0
        face_count = 0
        in_header = True

        while in_header:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line == "end_header":
                in_header = False

        for _ in range(vertex_count):
            line = f.readline().strip()
            coords = [float(x) for x in line.split()[:3]]
            vertices.append(coords)

        for _ in range(face_count):
            line = f.readline().strip()
            parts = [int(x) for x in line.split()]
            n = parts[0]
            face_indices = parts[1 : n + 1]
            if n == 3:
                faces.append(face_indices)
            elif n == 4:
                faces.append([face_indices[0], face_indices[1], face_indices[2]])
                faces.append([face_indices[0], face_indices[2], face_indices[3]])

    print(f"Loaded PLY: {len(vertices)} vertices, {len(faces)} faces")
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


# =============================================================================
# Bullet Simulation
# =============================================================================


class BulletSimulator(Simulator):
    """
    Simulation of a soft bullet being pushed out of a rifled barrel.
    """

    def __init__(self, config: dict | None = None):
        self.script_dir = Path(__file__).parent

        # Helper to get config values with defaults
        def cfg(key, default=None):
            val = get_config_value(config, key) if config else default
            return val if val is not None else default

        # Store bullet params for later use
        self.bullet_density = cfg("density", 11.34)
        self.bullet_k_mu = cfg("k_mu", 1.0e6)
        self.bullet_k_lambda = cfg("k_lambda", 1.0e6)
        self.bullet_k_damp = cfg("k_damp", 1e-4)
        self.particle_radius = cfg("particle_radius", 0.1)
        self.force_magnitude = cfg("force_magnitude", 1e2)
        self.force_direction = cfg("force_direction", [0.0, 0.0, 1.0])
        self.bottom_threshold_cm = cfg("bottom_threshold_cm", 0.5)
        self.bullet_color = tuple(cfg("color", [0.8, 0.6, 0.2]))

        # Load bullet tetmesh
        bullet_scale = cfg("scale", 100.0)
        tetmesh_file = cfg("tetmesh_file", "bullet_tetmesh.npz")
        tetmesh_path = self.script_dir / tetmesh_file
        self.bullet_verts, self.bullet_tets = load_tetmesh_npz(tetmesh_path)
        self.bullet_verts = self.bullet_verts * bullet_scale

        # Load barrel mesh
        barrel_file = cfg("mesh_file", "rifled_barrel.ply")
        barrel_path = self.script_dir / barrel_file
        self.barrel_verts, self.barrel_faces = load_ply_mesh(barrel_path)
        self.barrel_verts = self.barrel_verts * bullet_scale  # Use same scale

        self.bullet_vert_count = len(self.bullet_verts)

        # Store visualization config
        self.camera_position = tuple(cfg("camera_position", [0.0, -50.0, 0.0]))
        self.camera_target = tuple(cfg("camera_target", [0.0, 0.0, 15.0]))

        # Call parent init
        super().__init__(config)

    def custom_init(self):
        """Add bullet soft body and barrel as fixed cloth mesh."""
        from newton import ParticleFlags

        # Set particle radius for collision detection
        self.builder.default_particle_radius = self.particle_radius

        # Add bullet as soft body (using config params)
        self.builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=self.bullet_verts.tolist(),
            indices=self.bullet_tets.flatten().tolist(),
            density=self.bullet_density,
            k_mu=self.bullet_k_mu,
            k_lambda=self.bullet_k_lambda,
            k_damp=self.bullet_k_damp,
        )

        # Track barrel vertex start index
        self.barrel_vert_start = len(self.builder.particle_q)

        # Add barrel as cloth mesh (will be fixed in place)
        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=self.barrel_verts.tolist(),
            indices=self.barrel_faces.flatten().tolist(),
            density=1.0,  # Doesn't matter since we fix all vertices
        )

        # Fix all barrel vertices (set mass=0, remove ACTIVE flag)
        barrel_vert_end = len(self.builder.particle_q)
        for i in range(self.barrel_vert_start, barrel_vert_end):
            self.builder.particle_mass[i] = 0.0
            self.builder.particle_flags[i] &= ~ParticleFlags.ACTIVE

        self.barrel_vert_count = barrel_vert_end - self.barrel_vert_start
        print(f"Barrel: {self.barrel_vert_count} vertices (all fixed)")

    def custom_finalize(self):
        """Setup after model is built."""
        # Get bullet vertex indices for force application
        # Bottom vertices are those with smallest z
        bullet_positions = self.model.particle_q.numpy()[: self.bullet_vert_count]
        z_coords = bullet_positions[:, 2]
        z_min = z_coords.min()
        z_max = z_coords.max()
        z_threshold = z_min + self.bottom_threshold_cm

        bottom_indices = np.where(z_coords < z_threshold)[0].astype(np.int32)
        self.bottom_vertex_indices = wp.array(bottom_indices, dtype=wp.int32)

        # Compute per-vertex volumes for force weighting
        vertex_volumes = compute_vertex_volumes(bullet_positions, self.bullet_tets)
        bottom_volumes = vertex_volumes[bottom_indices]

        # Normalize volumes so total force equals force_magnitude * total_bottom_volume
        # Force per vertex = force_magnitude * (vertex_volume / total_volume) * total_volume
        #                  = force_magnitude * vertex_volume
        # This means force_magnitude is effectively "pressure" (force per unit volume)
        force_magnitudes = (self.force_magnitude * bottom_volumes).astype(np.float32)
        self.force_magnitudes = wp.array(force_magnitudes, dtype=wp.float32)

        # Store force direction as unit vector
        fd = self.force_direction
        fd_norm = np.linalg.norm(fd)
        self.force_direction_vec = wp.vec3(fd[0] / fd_norm, fd[1] / fd_norm, fd[2] / fd_norm)

        total_volume = bottom_volumes.sum()
        total_force = force_magnitudes.sum()

        print(f"Bullet Z range: {z_min:.2f} to {z_max:.2f} cm (height: {z_max - z_min:.2f} cm)")
        print(f"Bottom threshold: z < {z_threshold:.2f} cm")
        print(f"Found {len(bottom_indices)} / {self.bullet_vert_count} vertices for force application")
        print(f"Bottom volume: {total_volume:.4f} cm³")
        print(f"Force magnitude (pressure): {self.force_magnitude}")
        print(f"Total force: {total_force:.2f} dynes")

        # Store for visualization
        self.bullet_surface_faces = None
        if hasattr(self, "bullet_tets"):
            # Extract surface faces from tetrahedra
            face_count = {}
            for tet in self.bullet_tets:
                faces_of_tet = [
                    (tet[0], tet[2], tet[1]),
                    (tet[0], tet[1], tet[3]),
                    (tet[0], tet[3], tet[2]),
                    (tet[1], tet[2], tet[3]),
                ]
                for face in faces_of_tet:
                    key = tuple(sorted(face))
                    if key not in face_count:
                        face_count[key] = [0, face]
                    face_count[key][0] += 1

            self.bullet_surface_faces = np.array(
                [face for key, (count, face) in face_count.items() if count == 1], dtype=np.int32
            )
            print(f"Extracted {len(self.bullet_surface_faces)} surface faces")

    def apply_bullet_force(self):
        """Apply force to bottom of bullet, proportional to vertex volume (GPU kernel)."""
        wp.launch(
            kernel=apply_force_to_particles,
            dim=len(self.bottom_vertex_indices),
            inputs=[
                self.state_0.particle_f,
                self.bottom_vertex_indices,
                self.force_magnitudes,
                self.force_direction_vec,
            ],
        )

    def run_step(self):
        """Override run_step to apply force at start of each substep."""
        for _ in range(self.num_substeps):
            # Run collision detection
            if self.model.shape_count:
                self.contacts = self.model.collide(self.state_0)

            # Clear forces and apply our bullet force
            self.state_0.clear_forces()
            self.apply_bullet_force()

            # Run solver
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def setup_polyscope_meshes(self):
        """Register meshes for visualization."""
        if not self.do_rendering:
            return

        import polyscope as ps

        # Register barrel as surface mesh (transparent)
        # Get barrel vertices from particle positions
        all_verts = self.model.particle_q.numpy()
        barrel_verts = all_verts[self.barrel_vert_start : self.barrel_vert_start + self.barrel_vert_count]
        # Remap barrel faces to local indices (0-based)
        barrel_faces_local = self.barrel_faces.copy()
        barrel_mesh = ps.register_surface_mesh("Barrel", barrel_verts, barrel_faces_local)
        barrel_mesh.set_transparency(0.18)

        # Register bullet
        if self.bullet_surface_faces is not None:
            bullet_verts = self.model.particle_q.numpy()[: self.bullet_vert_count]
            self.ps_bullet = ps.register_surface_mesh("Bullet", bullet_verts, self.bullet_surface_faces)
            self.ps_bullet.set_color(self.bullet_color)

    def update_ps_meshes(self):
        """Update bullet mesh positions."""
        if not self.do_rendering:
            return

        bullet_verts = self.state_0.particle_q.numpy()[: self.bullet_vert_count]
        if hasattr(self, "ps_bullet"):
            self.ps_bullet.update_vertex_positions(bullet_verts)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import polyscope as ps

    import newton

    # Configuration - all parameters in one place
    config = {
        **default_config,
        "name": "bullet_out_of_barrel",
        # Simulation timing
        "up_axis": "z",
        "gravity": -980.0,
        "fps": 1000,
        "sim_substeps": 5,
        "iterations": 10,
        "sim_num_frames": 5000,
        # Solver
        "use_tile_solve": True,
        "use_cuda_graph": True,
        # Contact
        "handle_self_contact": True,
        "self_contact_radius": 0.1,  # Collision detection radius in cm
        "self_contact_margin": 0.15,  # Should be > self_contact_radius
        "topological_contact_filter_threshold":1  ,
        "rest_shape_contact_exclusion_radius": 0.2,
        "soft_contact_ke": 1.0e7,
        "soft_contact_kd": 1e-6,
        "soft_contact_mu": 0.3,
        # Ground
        "has_ground": False,
        "show_ground_plane": False,
        # Output
        "output_path": "bullet_frames",
        "write_output": False,
        "write_video": False,
        # Visualization
        "do_rendering": True,
        "is_initially_paused": False,
        "camera_position": [0.0, -50.0, 0.0],
        "camera_target": [0.0, 0.0, 15.0],
        # Bullet parameters
        "tetmesh_file": "bullet_tetmesh.npz",
        "scale": 100.0,
        "density": 11.34,
        "k_mu": 1.0e7,
        "k_lambda": 1.0e7,
        "k_damp": 1e-4,
        "particle_radius": 0.10 ,  # Collision radius in cm
        "force_magnitude": 2e7,
        "force_direction": [0.0, 0.0, 1.0],
        "bottom_threshold_cm": 0.5,  # Increased to capture more bottom vertices
        "color": [0.8, 0.6, 0.2],
        # Barrel parameters
        "mesh_file": "rifled_barrel.ply",
    }

    print("=" * 60)
    print("🔫 BULLET OUT OF BARREL SIMULATION 🔫")
    print("=" * 60)
    print("\nLoading meshes...")

    # Create simulator
    sim = BulletSimulator(config)
    sim.finalize()

    # Set camera from config
    ps.look_at(sim.camera_position, sim.camera_target)

    print("\n[INFO] Press SPACE to start simulation!")
    print("[INFO] Watch the bullet get pushed out! 🚀")

    try:
        sim.simulate()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        ps.shutdown()
        print("[INFO] Simulation finished! 🎉")
