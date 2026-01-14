# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Falling Gift Demo

Four stacked soft body blocks with two cloth straps wrapped around them.
Refactored to use the M01_Simulator utility for simulation management.
"""

import numpy as np
import warp as wp

from os.path import join

import sys
from pathlib import Path

# Add parent directory to path for M01_Simulator import
sys.path.insert(0, str(Path(__file__).parent.parent))
from M01_Simulator import Simulator, default_config


# =============================================================================
# Geometry Helpers
# =============================================================================

def cloth_loop_around_box(
    hx=1.6,            # half-size in X (box width / 2)
    hz=2.0,            # half-size in Z (box height / 2)
    width=0.25,        # strap width (along Y)
    center_y=0.0,      # Y position of the strap center
    nu=120,            # resolution along loop
    nv=6,              # resolution across strap width
):
    """
    Vertical closed cloth loop wrapped around a cuboid.
    Loop lies in X-Z plane, strap width is along Y.
    Z is up.
    """
    verts = []
    faces = []

    # Rectangle perimeter length
    P = 4.0 * (hx + hz)

    for i in range(nu):
        s = (i / nu) * P

        # Walk rectangle in X–Z plane (counter-clockwise)
        if s < 2 * hx:
            x = -hx + s
            z = -hz
        elif s < 2 * hx + 2 * hz:
            x = hx
            z = -hz + (s - 2 * hx)
        elif s < 4 * hx + 2 * hz:
            x = hx - (s - (2 * hx + 2 * hz))
            z = hz
        else:
            x = -hx
            z = hz - (s - (4 * hx + 2 * hz))

        for j in range(nv):
            v = (j / (nv - 1) - 0.5) * width
            y = center_y + v
            verts.append([x, y, z])

    def idx(i, j):
        return (i % nu) * nv + j

    # Triangulation
    for i in range(nu):
        for j in range(nv - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            faces.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    return (
        np.array(verts, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )


PYRAMID_TET_INDICES = np.array(
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

PYRAMID_PARTICLES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (2.0, 1.0, 0.0),
    (0.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
    (2.0, 2.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (2.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0),
    (0.0, 2.0, 1.0),
    (1.0, 2.0, 1.0),
    (2.0, 2.0, 1.0),
]

# Number of vertices per soft body block
VERTS_PER_BLOCK = len(PYRAMID_PARTICLES)  # 18


# =============================================================================
# Falling Gift Simulator
# =============================================================================

class FallingGiftSimulator(Simulator):
    """
    Simulation of four stacked soft body blocks with two cloth straps.
    """

    def __init__(self, config: dict | None = None):
        # Store geometry for later use
        self.base_height = 30.0
        self.spacing = 1.01  # small gap to avoid initial penetration
        
        # Generate cloth geometry
        self.strap1_verts, self.strap1_faces = cloth_loop_around_box(
            hx=1.01, hz=2.02, width=0.6
        )
        self.strap2_verts, self.strap2_faces = cloth_loop_around_box(
            hx=1.015, hz=2.025, width=0.6
        )
        
        self.strap1_count = len(self.strap1_verts)
        self.strap2_count = len(self.strap2_verts)
        
        # Call parent init (this calls custom_init)
        super().__init__(config)

    def custom_init(self):
        """Add soft body blocks and cloth straps to the simulation."""
        
        # Add 4 stacked soft body blocks
        for i in range(4):
            self.builder.add_soft_mesh(
                pos=wp.vec3(0.0, 0.0, self.base_height + i * self.spacing),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=PYRAMID_PARTICLES,
                indices=PYRAMID_TET_INDICES.flatten().tolist(),
                density=100,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=1e-5,
            )
        
        # Add first cloth strap
        self.builder.add_cloth_mesh(
            pos=wp.vec3(1.0, 1.0, self.base_height + 1.5 * self.spacing + 0.5),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=self.strap1_verts,
            indices=self.strap1_faces.flatten().tolist(),
            density=0.02,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-5,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=0.05,
        )
        
        # Add second cloth strap (rotated 90 degrees)
        self.builder.add_cloth_mesh(
            pos=wp.vec3(1.0, 1.0, self.base_height + 1.5 * self.spacing + 0.5),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=self.strap2_verts,
            indices=self.strap2_faces.flatten().tolist(),
            density=0.02,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-5,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=0.05,
        )

    def custom_finalize(self):
        """Extract face indices for each mesh type."""
        # Get box faces (first portion of tri_indices, excluding cloth)
        all_faces = self.model.tri_indices.numpy()
        cloth_face_count = len(self.strap1_faces) + len(self.strap2_faces)
        box_faces_all = all_faces[:len(all_faces) - cloth_face_count]
        
        # Each box has the same faces (just first quarter)
        self.box_faces = box_faces_all[:len(box_faces_all) // 4]
        
        # Vertex layout:
        # [0:18]    - Box 1
        # [18:36]   - Box 2
        # [36:54]   - Box 3
        # [54:72]   - Box 4
        # [72:72+strap1_count] - Strap 1
        # [72+strap1_count:]   - Strap 2
        self.box_start = 0
        self.strap1_start = 4 * VERTS_PER_BLOCK
        self.strap2_start = self.strap1_start + self.strap1_count

    def setup_polyscope_meshes(self):
        """Register individual meshes for visualization."""
        if not self.do_rendering:
            return
        
        import polyscope as ps
        
        all_verts = self.model.particle_q.numpy()
        
        # Register boxes as volume meshes
        self.ps_box1 = ps.register_volume_mesh(
            "Box1", all_verts[0:18], tets=PYRAMID_TET_INDICES
        )
        self.ps_box2 = ps.register_volume_mesh(
            "Box2", all_verts[18:36], tets=PYRAMID_TET_INDICES
        )
        self.ps_box3 = ps.register_volume_mesh(
            "Box3", all_verts[36:54], tets=PYRAMID_TET_INDICES
        )
        self.ps_box4 = ps.register_volume_mesh(
            "Box4", all_verts[54:72], tets=PYRAMID_TET_INDICES
        )
        
        # Register cloth straps as surface meshes
        self.register_ps_mesh(
            name="Strap1",
            vertices=all_verts[self.strap1_start:self.strap1_start + self.strap1_count],
            faces=self.strap1_faces,
            vertex_indices=slice(self.strap1_start, self.strap1_start + self.strap1_count),
            color=(1.0, 0.0, 0.0),
        )
        self.register_ps_mesh(
            name="Strap2",
            vertices=all_verts[self.strap2_start:],
            faces=self.strap2_faces,
            vertex_indices=slice(self.strap2_start, None),
            color=(1.0, 0.0, 0.0),
        )
        
        # Set box colors
        box_color = (0.0, 0.2, 0.125)
        self.ps_box1.set_color(box_color)
        self.ps_box2.set_color(box_color)
        self.ps_box3.set_color(box_color)
        self.ps_box4.set_color(box_color)

    def update_ps_meshes(self):
        """Update all meshes with current positions."""
        all_verts = self.state_0.particle_q.numpy()
        
        # Update boxes
        self.ps_box1.update_vertex_positions(all_verts[0:18])
        self.ps_box2.update_vertex_positions(all_verts[18:36])
        self.ps_box3.update_vertex_positions(all_verts[36:54])
        self.ps_box4.update_vertex_positions(all_verts[54:72])
        
        # Update cloth straps (handled by parent class via ps_meshes registry)
        super().update_ps_meshes()

    def save_initial_meshes(self):
        """Save initial mesh topology for each component separately."""
        if self.output_path is None:
            return
        
        all_verts = self.model.particle_q.numpy()
        
        def write_ply(path, vertices, faces):
            with open(path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                for v in vertices:
                    f.write(f"{v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        # Save each mesh separately
        write_ply(join(self.output_path, "initial_cloth1.ply"),
                  all_verts[self.strap1_start:self.strap1_start + self.strap1_count],
                  self.strap1_faces)
        write_ply(join(self.output_path, "initial_cloth2.ply"),
                  all_verts[self.strap2_start:],
                  self.strap2_faces)
        write_ply(join(self.output_path, "initial_box1.ply"), all_verts[0:18], self.box_faces)
        write_ply(join(self.output_path, "initial_box2.ply"), all_verts[18:36], self.box_faces)
        write_ply(join(self.output_path, "initial_box3.ply"), all_verts[36:54], self.box_faces)
        write_ply(join(self.output_path, "initial_box4.ply"), all_verts[54:72], self.box_faces)
        
        print(f"Initial meshes saved to: {self.output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import polyscope as ps
    
    # Configuration
    config = {
        **default_config,
        "name": "falling_gift",
        "up_axis": "z",
        "gravity": -10,
        "fps": 60,
        "sim_substeps": 10,
        "iterations": 15,
        "sim_num_frames": 800,
        # Self-contact settings
        "handle_self_contact": True,
        "self_contact_radius": 0.04,
        "self_contact_margin": 0.06,
        "topological_contact_filter_threshold": 1,
        "truncation_mode": 1,
        # Contact physics
        "soft_contact_ke": 1.0e5,
        "soft_contact_kd": 1e-5,
        "soft_contact_mu": 1.0,
        # Ground
        "has_ground": True,
        "ground_height": 0.0,
        "show_ground_plane": True,
        # Output
        "output_path": "ply_frames",
        "output_ext": "npy",
        "write_output": False,
        "write_video": False,
        # Visualization
        "do_rendering": True,
        "is_initially_paused": False,
    }
    
    # Create and run simulator
    sim = FallingGiftSimulator(config)
    sim.finalize()
    
    # Set camera
    ps.look_at((-10.0, 0.0, 20.0), (0.0, 0.0, 0.0))
    
    print("[INFO] Simulation starting...")
    
    try:
        sim.simulate()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        ps.shutdown()
        print("[INFO] Simulation finished")
