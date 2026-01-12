# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Bullet Out of Barrel Demo - Collision Shape Version 🎉

A soft body bullet being pushed out of a rifled barrel.
Barrel is a static collision shape (particle-shape collision).
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
# Bullet Simulation with Collision Shape Barrel
# =============================================================================


class BulletSimulatorShape(Simulator):
    """
    Simulation of a soft bullet being pushed out of a rifled barrel.
    Barrel is a static collision shape (faster, simpler collision).
    """

    def __init__(self, config: dict | None = None):
        self.script_dir = Path(__file__).parent

        # Helper to get config values with defaults
        def cfg(key, default=None):
            val = get_config_value(config, key) if config else default
            return val if val is not None else default

        # Velocity tracking
        self.prev_centroid = None
        self.sim_fps = cfg("fps", 60000)  # Simulation FPS (for velocity calc)
        self.video_fps = cfg("video_fps", 60)  # Playback FPS for video
        self.frame_count = 0

        # Store bullet params for later use
        self.bullet_density = cfg("density", 11.34)
        self.bullet_k_mu = cfg("k_mu", 1.0e6)
        self.bullet_k_lambda = cfg("k_lambda", 1.0e6)
        self.bullet_k_damp = cfg("k_damp", 1e-4)
        self.particle_radius = cfg("particle_radius", 0.1)
        self.shape_friction = cfg("shape_friction", 0.05)  # Barrel friction
        self.force_magnitude = cfg("force_magnitude", 1e2)
        self.force_direction = cfg("force_direction", [0.0, 0.0, 1.0])
        self.bottom_threshold_cm = cfg("bottom_threshold_cm", 0.5)
        self.bullet_color = tuple(cfg("color", [0.8, 0.6, 0.2]))

        # Load bullet tetmesh
        bullet_scale = cfg("bullet_scale", 100.0)
        tetmesh_file = cfg("tetmesh_file", "bullet_tetmesh.npz")
        tetmesh_path = self.script_dir / tetmesh_file
        self.bullet_verts, self.bullet_tets = load_tetmesh_npz(tetmesh_path)
        self.bullet_verts = self.bullet_verts * bullet_scale

        # Load barrel mesh (separate scale)
        barrel_scale = cfg("barrel_scale", 100.0)
        barrel_file = cfg("mesh_file", "rifled_barrel.ply")
        barrel_path = self.script_dir / barrel_file
        self.barrel_verts, self.barrel_faces = load_ply_mesh(barrel_path)
        self.barrel_verts = self.barrel_verts * barrel_scale

        self.bullet_vert_count = len(self.bullet_verts)

        # Store visualization config
        self.camera_position = tuple(cfg("camera_position", [0.0, -50.0, 0.0]))
        self.camera_target = tuple(cfg("camera_target", [0.0, 0.0, 15.0]))

        # Call parent init
        super().__init__(config)

    def custom_init(self):
        """Add bullet soft body and barrel as static collision shape."""
        import newton

        # Set particle radius for collision detection
        self.builder.default_particle_radius = self.particle_radius

        # Add bullet as soft body
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

        # Add barrel as static collision mesh (body=-1 means static/world)
        barrel_mesh = newton.Mesh(self.barrel_verts, self.barrel_faces.flatten())
        
        # Configure barrel shape with low friction
        barrel_cfg = newton.ModelBuilder.ShapeConfig(
            mu=self.shape_friction,  # Low friction for smooth sliding
        )
        self.builder.add_shape_mesh(
            body=-1,
            mesh=barrel_mesh,
            cfg=barrel_cfg,
        )
        print(f"Barrel: {len(self.barrel_verts)} vertices as collision shape (mu={self.shape_friction})")

    def custom_finalize(self):
        """Setup after model is built."""
        # Get bullet vertex indices for force application
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

        # Force per vertex = pressure * vertex_volume
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

        # Extract surface faces for visualization
        self.bullet_surface_faces = None
        if hasattr(self, "bullet_tets"):
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
        """Apply force to bottom of bullet, proportional to vertex volume."""
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
            # Run collision detection (particle-shape collision)
            if self.model.shape_count:
                self.contacts = self.model.collide(self.state_0)

            # Clear forces and apply our bullet force
            self.state_0.clear_forces()
            self.apply_bullet_force()

            # Run solver
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Override step to print velocity after each frame."""
        result = super().step()
        if result:
            # Compute and print velocity every 10 frames
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                self.print_velocity()
        return result

    def print_velocity(self):
        """Compute and print bullet velocity in m/s."""
        bullet_positions = self.state_0.particle_q.numpy()[: self.bullet_vert_count]
        current_centroid = bullet_positions.mean(axis=0)

        speed_m_s = 0.0
        if self.prev_centroid is not None:
            # Displacement in cm over 10 frames
            displacement_cm = current_centroid - self.prev_centroid
            time_s = 10.0 / self.sim_fps  # Use simulation FPS, not video FPS
            velocity_cm_s = displacement_cm / time_s
            velocity_m_s = velocity_cm_s / 100.0  # cm to m
            speed_m_s = np.linalg.norm(velocity_m_s)
        
        print(f"Frame {self.frame_count:4d}: z={current_centroid[2]:6.1f} cm, speed={speed_m_s:6.1f} m/s", flush=True)
        self.prev_centroid = current_centroid.copy()

    def simulate(self):
        """Override simulate to use video_fps for video output."""
        # Temporarily set fps to video_fps for video writer
        original_fps = self.fps
        video_fps = getattr(self, 'video_fps', 60)
        self.fps = video_fps
        
        # Call parent simulate
        super().simulate()
        
        # Restore original fps
        self.fps = original_fps

    def setup_polyscope_meshes(self):
        """Register meshes for visualization."""
        if not self.do_rendering:
            return

        import polyscope as ps

        # Register barrel as surface mesh (transparent) - use original vertices
        barrel_mesh = ps.register_surface_mesh("Barrel", self.barrel_verts, self.barrel_faces)
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
        "name": "bullet_out_of_barrel_shape",
        # Simulation timing
        "up_axis": "z",
        "gravity": -0.0,
        # "fps": 60000,
        # "sim_substeps": 30,
        # "sim_num_frames": 120,
        "fps": 600000,
        "sim_substeps": 3,
        "sim_num_frames": 1200,
        "iterations": 20,
        # Solver
        "use_tile_solve": True,
        "use_cuda_graph": True,
        # Contact (no self-contact needed for collision shape approach)
        "handle_self_contact": False,
        "soft_contact_ke": 2.0e10,
        "soft_contact_kd": 0,
        "soft_contact_mu": 0.0 ,  # Low friction for smooth sliding
        # Ground
        "has_ground": False,
        "show_ground_plane": False,
        # Output (subfolder will be created with FPS)
        "output_base_path": r"D:\Data\DAT_Sim\bullet_out_of_barrel",
        "output_ext": "npy",  # Save positions only (faster than ply)
        "write_output": True,
        "write_video": True,
        "video_fps": 60,  # Playback FPS for video (separate from sim fps)
        # Visualization
        "do_rendering": True,
        "is_initially_paused": False,
        "camera_position": [0.0, -50.0, 0.0],
        "camera_target": [0.0, 0.0, 15.0],
        # Bullet parameters
        "tetmesh_file": "bullet_tetmesh.npz",

        "bullet_scale": 100.0,  # Bullet scale (separate from barrel)
        "density": 11.34,
        "k_mu": 5.0e9,
        "k_lambda": 2.0e10,
        "k_damp": 1e-12 ,
        "particle_radius": 0.1,
        "force_magnitude": 2e9,
        "force_direction": [0.0, 0.0, 1.0], 
        "bottom_threshold_cm": 0.5,
        "color": [0.8, 0.6, 0.2],  
        # Barrel parameters
        "mesh_file": "rifled_barrel.ply",
        "barrel_scale": 100.0,  # Barrel scale (separate from bullet)
        "shape_friction": 0.0,  # Some friction so rifling can grip and spin the bullet
    }

    # Create output subfolder with FPS
    fps = config["fps"]
    base_path = config.get("output_base_path", "bullet_frames")
    config["output_path"] = f"{base_path}/fps_{fps}"

    print("=" * 60)
    print("🔫 BULLET OUT OF BARREL (Collision Shape) 🔫")
    print("=" * 60)
    print(f"\nOutput path: {config['output_path']}")
    print("Loading meshes...")

    # Create simulator
    sim = BulletSimulatorShape(config)
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
