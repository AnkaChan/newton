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
# Example Multiphysics Drop
#
# This simulation demonstrates multiple physics types interacting:
# - A volumetric soft body (pyramid tet mesh)
# - A rigid body box
# - A cloth sheet
#
# All objects drop onto the cloth under gravity, showcasing coupled
# soft body, rigid body, and cloth simulation.
#
# Command: python -m newton.examples.mutlphysics.example_multiphysics_drop
#
###########################################################################

import os

import numpy as np
import polyscope as ps
import warp as wp

import newton
from newton.examples.cloth.M01_Simulator import Simulator, get_config_value


def load_vtk_tet_mesh(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a tetrahedral mesh from a VTK file.

    Args:
        filepath: Path to the VTK file.

    Returns:
        Tuple of (vertices, tet_indices) as numpy arrays.
        - vertices: Nx3 array of vertex positions
        - tet_indices: Mx4 array of tetrahedron vertex indices
    """
    vertices = []
    tet_indices = []

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    num_points = 0
    num_cells = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("POINTS"):
            parts = line.split()
            num_points = int(parts[1])
            i += 1
            # Read vertices
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
            # Read cells (tetrahedra)
            while len(tet_indices) < num_cells and i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 5 and parts[0] == "4":  # Tetrahedron has 4 vertices
                    tet_indices.append([int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])])
                i += 1
            continue

        i += 1

    vertices = np.array(vertices, dtype=np.float32)
    tet_indices = np.array(tet_indices, dtype=np.int32)

    return vertices, tet_indices


def create_gear_mesh(
    num_teeth: int = 12,
    outer_radius: float = 1.0,
    inner_radius: float = 0.6,
    tooth_height: float = 0.25,
    thickness: float = 0.3,
    hole_radius: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a 3D gear mesh with teeth.

    Args:
        num_teeth: Number of teeth around the gear.
        outer_radius: Radius to the base of the teeth.
        inner_radius: Radius of the gear body (between hole and teeth).
        tooth_height: Height of each tooth (added to outer_radius).
        thickness: Thickness of the gear (z-axis).
        hole_radius: Radius of the center hole.

    Returns:
        Tuple of (vertices, faces) as numpy arrays.
        - vertices: Nx3 array of vertex positions
        - faces: Mx3 array of triangle indices
    """
    vertices = []
    faces = []

    half_thickness = thickness / 2.0
    tooth_outer_radius = outer_radius + tooth_height

    # Angular width of each tooth and gap
    tooth_angle = np.pi / num_teeth  # Half the angular period
    tooth_top_ratio = 0.4  # Ratio of tooth top width to tooth base

    # Create vertices for top and bottom faces
    # Structure: for each tooth, we have points at:
    # - hole radius (inner edge)
    # - inner radius (gear body)
    # - outer radius (tooth base)
    # - tooth outer radius (tooth tip)

    def add_ring_vertices(z: float, radii: list, angles: list) -> int:
        """Add a ring of vertices at given z, returns starting index."""
        start_idx = len(vertices)
        for angle in angles:
            for r in radii:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                vertices.append([x, y, z])
        return start_idx

    # Generate angles for each tooth segment
    angles_per_tooth = []
    for i in range(num_teeth):
        base_angle = i * 2 * np.pi / num_teeth

        # Tooth profile angles (symmetric around base_angle + tooth_angle/2)
        tooth_center = base_angle + tooth_angle / 2

        # Gap before tooth
        angles_per_tooth.append(base_angle)  # Start of gap

        # Tooth base start
        tooth_base_start = tooth_center - tooth_angle * 0.45
        angles_per_tooth.append(tooth_base_start)

        # Tooth top start
        tooth_top_start = tooth_center - tooth_angle * tooth_top_ratio / 2
        angles_per_tooth.append(tooth_top_start)

        # Tooth top end
        tooth_top_end = tooth_center + tooth_angle * tooth_top_ratio / 2
        angles_per_tooth.append(tooth_top_end)

        # Tooth base end
        tooth_base_end = tooth_center + tooth_angle * 0.45
        angles_per_tooth.append(tooth_base_end)

        # Gap after tooth (will be start of next)

    # Simplified approach: create gear profile as a polygon and extrude
    # Generate the 2D gear profile
    profile_points = []
    for i in range(num_teeth):
        base_angle = i * 2 * np.pi / num_teeth
        tooth_center = base_angle + tooth_angle

        # Points around each tooth (going counterclockwise)
        # 1. Gap (at outer_radius)
        gap_start = base_angle
        profile_points.append((outer_radius * np.cos(gap_start), outer_radius * np.sin(gap_start)))

        # 2. Tooth base start
        tooth_start = base_angle + tooth_angle * 0.3
        profile_points.append((outer_radius * np.cos(tooth_start), outer_radius * np.sin(tooth_start)))

        # 3. Tooth tip start (with slight inward angle for realistic tooth shape)
        tip_start = base_angle + tooth_angle * 0.4
        profile_points.append(
            (tooth_outer_radius * np.cos(tip_start), tooth_outer_radius * np.sin(tip_start))
        )

        # 4. Tooth tip end
        tip_end = base_angle + tooth_angle * 1.6
        profile_points.append((tooth_outer_radius * np.cos(tip_end), tooth_outer_radius * np.sin(tip_end)))

        # 5. Tooth base end
        tooth_end = base_angle + tooth_angle * 1.7
        profile_points.append((outer_radius * np.cos(tooth_end), outer_radius * np.sin(tooth_end)))

    num_profile = len(profile_points)

    # Create hole profile (simple circle)
    num_hole_segments = num_teeth * 2
    hole_angles = np.linspace(0, 2 * np.pi, num_hole_segments, endpoint=False)
    hole_points = [(hole_radius * np.cos(a), hole_radius * np.sin(a)) for a in hole_angles]

    # Build 3D vertices
    # Top face outer profile
    top_outer_start = len(vertices)
    for px, py in profile_points:
        vertices.append([px, py, half_thickness])

    # Top face hole profile
    top_hole_start = len(vertices)
    for px, py in hole_points:
        vertices.append([px, py, half_thickness])

    # Bottom face outer profile
    bot_outer_start = len(vertices)
    for px, py in profile_points:
        vertices.append([px, py, -half_thickness])

    # Bottom face hole profile
    bot_hole_start = len(vertices)
    for px, py in hole_points:
        vertices.append([px, py, -half_thickness])

    # Center vertices for top and bottom (for fan triangulation)
    # We'll use a ring approach instead for the hole

    # Create faces
    # Top face - triangulate between outer profile and hole
    # Use a simple approach: connect outer to hole with triangles
    # For simplicity, create a center point and fan triangulate

    top_center_idx = len(vertices)
    vertices.append([0, 0, half_thickness])
    bot_center_idx = len(vertices)
    vertices.append([0, 0, -half_thickness])

    # Top face: fan from center to hole, then connect hole to outer
    # Actually, let's triangulate the top face properly

    # Top face outer ring to center (simplified - just outer ring)
    for i in range(num_profile):
        i_next = (i + 1) % num_profile
        # Triangle from center to outer edge
        faces.append([top_center_idx, top_outer_start + i, top_outer_start + i_next])

    # Bottom face (reversed winding)
    for i in range(num_profile):
        i_next = (i + 1) % num_profile
        faces.append([bot_center_idx, bot_outer_start + i_next, bot_outer_start + i])

    # Side faces (connect top and bottom outer profiles)
    for i in range(num_profile):
        i_next = (i + 1) % num_profile
        # Two triangles per quad
        faces.append([top_outer_start + i, bot_outer_start + i, bot_outer_start + i_next])
        faces.append([top_outer_start + i, bot_outer_start + i_next, top_outer_start + i_next])

    # Inner hole faces (connect top and bottom hole profiles)
    for i in range(num_hole_segments):
        i_next = (i + 1) % num_hole_segments
        # Two triangles per quad (reversed winding for inner surface)
        faces.append([top_hole_start + i, top_hole_start + i_next, bot_hole_start + i_next])
        faces.append([top_hole_start + i, bot_hole_start + i_next, bot_hole_start + i])

    # Connect hole to center on top and bottom
    for i in range(num_hole_segments):
        i_next = (i + 1) % num_hole_segments
        # Top - from hole to center (reversed to face outward)
        faces.append([top_center_idx, top_hole_start + i_next, top_hole_start + i])
        # Bottom - from hole to center
        faces.append([bot_center_idx, bot_hole_start + i, bot_hole_start + i_next])

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    return vertices, faces


# Pyramid tet mesh data (2x2x1 grid of particles forming 20 tets)
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

# Configuration dict - single source of truth for all simulation parameters
config = {
    "name": "multiphysics_drop",
    # Simulation timing
    "fps": 60,
    "sim_substeps": 20,
    "sim_num_frames": 300,
    "iterations": 5,
    # Solver settings
    "use_cuda_graph": True,
    "handle_self_contact": True,
    "use_tile_solve": True,
    "self_contact_radius": 0.3,  # cm
    "self_contact_margin": 0.5,  # cm
    "include_bending": True,
    "topological_contact_filter_threshold": 2,
    "rest_shape_contact_exclusion_radius": 1.5,
    # Physics (using centimeters)
    "up_axis": "z",
    "gravity": -980.0,  # cm/s²
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 1e-5,
    "soft_contact_mu": 0.2,
    # Ground plane
    "has_ground": True,
    "ground_height": 0.0,
    # Shared soft body material parameters
    "softbody_density": 0.0003,  # g/cm³ (same as 500 kg/m³)
    "softbody_k_mu": 5.0e4,
    "softbody_k_lambda": 5.0e4,
    "softbody_k_damp": 1e-9,
    "softbody_particle_radius": 1,
    # Soft body hippo (mesh-specific)
    "hippo_enabled": True,
    "hippo_vtk_path": "hippo.vtk",
    "hippo_pos": (30.0, -30.0, 250.0),  # cm
    "hippo_scale": 50.0,  # cm
    # Soft body bunny (mesh-specific)
    "bunny_enabled": True,
    "bunny_vtk_path": "bunny_small.vtk",
    "bunny_pos": (-30.0, 30.0, 250.0),  # cm
    "bunny_scale": 5.0,  # cm
    # Rigid body box parameters
    "box_enabled": True,
    "box_pos": (-30.0, -30.0, 220.0),  # cm
    "box_half_extents": (15.0, 15.0, 15.0),  # cm (30cm cube)
    "box_density": 0.0005,  # g/cm³ (same as 500 kg/m³)
    "box_ke": 1.0e5,
    "box_kd": 1e-7,
    "box_mu": 0.5,
    # Rigid body gear parameters
    "gear_pos": (30.0, 30.0, 280.0),  # cm
    "gear_scale": 15.0,  # cm
    "gear_num_teeth": 12,
    "gear_outer_radius": 1.0,  # relative (scaled by gear_scale)
    "gear_inner_radius": 0.6,
    "gear_tooth_height": 0.25,
    "gear_thickness": 0.3,
    "gear_hole_radius": 0.15,
    "gear_density": 0.0008,  # g/cm³ (same as 800 kg/m³)
    "gear_ke": 1.0e5,
    "gear_kd": 1e-3,
    "gear_mu": 0.5,
    # Cloth parameters
    "cloth_pos": (-100.0, -100.0, 100.0),  # cm
    "cloth_dim_x": 80,
    "cloth_dim_y": 80,
    "cloth_cell_x": 3.0,  # cm (200cm total width)
    "cloth_cell_y": 3.0,  # cm
    "cloth_mass": 0.05,  # g per particle
    "cloth_tri_ke": 5e5,
    "cloth_tri_ka": 5e5,
    "cloth_tri_kd": 1e-5,
    "cloth_edge_ke": 0.01,
    "cloth_edge_kd": 1e-2,
    "cloth_particle_radius": 1.0,  # cm
    "cloth_fix_left": True,
    "cloth_fix_right": True,
    # Output settings
    "output_path": r"D:\Data\DAT_Sim\multiphysics_drop",  # Parent directory
    "experiment_name": "run",  # Folder name prefix
    "output_timestamp": True,  # Append timestamp to experiment folder
    "write_output": False,
    "write_video": False,
    # Visualization
    "do_rendering": True,
    "show_ground_plane": True,
}


class MultiphysicsDropSimulator(Simulator):
    """
    Simulator for multiphysics interaction: soft bodies + rigid bodies + cloth.

    Inherits from the base Simulator class and adds:
    - A soft body hippo (loaded from VTK file)
    - A soft body bunny (loaded from VTK file)
    - A rigid body box
    - A rigid body gear
    - A cloth grid with fixed corners
    """

    def custom_init(self):
        """Add soft bodies, rigid bodies, and cloth to the simulation."""
        # Helper to read from config or fall back to default
        def cfg(key):
            return get_config_value(self.config, key)

        # Add soft body hippo (from VTK file)
        if cfg("hippo_enabled"):
            vtk_path = cfg("hippo_vtk_path")
            if not os.path.isabs(vtk_path):
                vtk_path = os.path.join(os.path.dirname(__file__), vtk_path)

            hippo_verts, hippo_tets = load_vtk_tet_mesh(vtk_path)
            print(f"Loaded hippo mesh: {len(hippo_verts)} vertices, {len(hippo_tets)} tetrahedra")

            hippo_vertices = [(v[0], v[1], v[2]) for v in hippo_verts]

            hippo_pos = cfg("hippo_pos")
            self.builder.add_soft_mesh(
                pos=wp.vec3(hippo_pos[0], hippo_pos[1], hippo_pos[2]),
                rot=wp.quat_identity(),
                scale=cfg("hippo_scale"),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=hippo_vertices,
                indices=hippo_tets.flatten().tolist(),
                density=cfg("softbody_density"),
                k_mu=cfg("softbody_k_mu"),
                k_lambda=cfg("softbody_k_lambda"),
                k_damp=cfg("softbody_k_damp"),
                particle_radius=cfg("softbody_particle_radius")
            )

        # Add soft body bunny (from VTK file)
        if cfg("bunny_enabled"):
            vtk_path = cfg("bunny_vtk_path")
            if not os.path.isabs(vtk_path):
                vtk_path = os.path.join(os.path.dirname(__file__), vtk_path)

            bunny_verts, bunny_tets = load_vtk_tet_mesh(vtk_path)
            print(f"Loaded bunny mesh: {len(bunny_verts)} vertices, {len(bunny_tets)} tetrahedra")

            bunny_vertices = [(v[0], v[1], v[2]) for v in bunny_verts]

            bunny_pos = cfg("bunny_pos")
            self.builder.add_soft_mesh(
                pos=wp.vec3(bunny_pos[0], bunny_pos[1], bunny_pos[2]),
                rot=wp.quat_identity(),
                scale=cfg("bunny_scale"),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=bunny_vertices,
                indices=bunny_tets.flatten().tolist(),
                density=cfg("softbody_density"),
                k_mu=cfg("softbody_k_mu"),
                k_lambda=cfg("softbody_k_lambda"),
                k_damp=cfg("softbody_k_damp"),
                particle_radius=cfg("softbody_particle_radius")
            )

        # Track particle count before adding cloth (for visualization separation)
        self.softbody_particle_count = self.builder.particle_count

        # Add rigid body box
        box_pos = cfg("box_pos")
        box_half = cfg("box_half_extents")
        body_box = self.builder.add_body(
            xform=wp.transform(
                p=wp.vec3(box_pos[0], box_pos[1], box_pos[2]),
                q=wp.quat_identity(),
            ),
            key="box",
        )
        # Create ShapeConfig for the box
        box_shape_cfg = newton.ModelBuilder.ShapeConfig()
        box_shape_cfg.density = cfg("box_density")
        box_shape_cfg.ke = cfg("box_ke")
        box_shape_cfg.kd = cfg("box_kd")
        box_shape_cfg.mu = cfg("box_mu")

        self.builder.add_shape_box(
            body_box,
            hx=box_half[0],
            hy=box_half[1],
            hz=box_half[2],
            cfg=box_shape_cfg,
        )

        # Store body index for visualization
        self.box_body_index = body_box

        # Add rigid body gear
        gear_pos = cfg("gear_pos")
        gear_scale = cfg("gear_scale")

        # Generate gear mesh
        gear_verts, gear_faces = create_gear_mesh(
            num_teeth=cfg("gear_num_teeth"),
            outer_radius=cfg("gear_outer_radius"),
            inner_radius=cfg("gear_inner_radius"),
            tooth_height=cfg("gear_tooth_height"),
            thickness=cfg("gear_thickness"),
            hole_radius=cfg("gear_hole_radius"),
        )

        # Scale the gear mesh
        gear_verts = gear_verts * gear_scale

        # Create a Newton Mesh object for the gear
        gear_mesh = newton.Mesh(
            vertices=gear_verts,
            indices=gear_faces.flatten(),
        )

        # Add gear body
        body_gear = self.builder.add_body(
            xform=wp.transform(
                p=wp.vec3(gear_pos[0], gear_pos[1], gear_pos[2]),
                q=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2),  # Rotate to lay flat
            ),
            key="gear",
        )

        # Create ShapeConfig for the gear
        gear_shape_cfg = newton.ModelBuilder.ShapeConfig()
        gear_shape_cfg.density = cfg("gear_density")
        gear_shape_cfg.ke = cfg("gear_ke")
        gear_shape_cfg.kd = cfg("gear_kd")
        gear_shape_cfg.mu = cfg("gear_mu")

        self.builder.add_shape_mesh(
            body_gear,
            mesh=gear_mesh,
            cfg=gear_shape_cfg,
        )

        # Store gear info for visualization
        self.gear_body_index = body_gear
        self.gear_verts_local = gear_verts
        self.gear_faces = gear_faces

        # Add cloth grid
        cloth_pos = cfg("cloth_pos")
        self.builder.add_cloth_grid(
            pos=wp.vec3(cloth_pos[0], cloth_pos[1], cloth_pos[2]),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            fix_left=cfg("cloth_fix_left"),
            fix_right=cfg("cloth_fix_right"),
            dim_x=cfg("cloth_dim_x"),
            dim_y=cfg("cloth_dim_y"),
            cell_x=cfg("cloth_cell_x"),
            cell_y=cfg("cloth_cell_y"),
            mass=cfg("cloth_mass"),
            tri_ke=cfg("cloth_tri_ke"),
            tri_ka=cfg("cloth_tri_ka"),
            tri_kd=cfg("cloth_tri_kd"),
            edge_ke=cfg("cloth_edge_ke"),
            edge_kd=cfg("cloth_edge_kd"),
            particle_radius=cfg("cloth_particle_radius"),
        )

    def custom_finalize(self):
        """Store rigid body box info for visualization."""
        # Get box half-extents from config for visualization
        def cfg(key):
            return get_config_value(self.config, key)

        self.box_half_extents = cfg("box_half_extents")

    def setup_polyscope_meshes(self):
        """Override to set up separate meshes for cloth and add box visualization."""
        if not self.do_rendering:
            return

        # Register cloth mesh (particles only)
        verts = self.model.particle_q.numpy()
        self.register_ps_mesh(
            name="Simulation",
            vertices=verts,
            faces=self.faces,
            vertex_indices=None,
            color=(0.2, 0.6, 0.9),  # Light blue for cloth
        )

        # Register rigid body box as a separate mesh
        self._register_box_mesh()

        # Register rigid body gear as a separate mesh
        self._register_gear_mesh()

    def _register_box_mesh(self):
        """Create and register a polyscope mesh for the rigid body box."""
        # Create box vertices (unit cube centered at origin, scaled by half-extents)
        hx, hy, hz = self.box_half_extents

        # 8 vertices of a box
        box_verts = np.array(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],
            ],
            dtype=np.float32,
        )

        # 12 triangles (2 per face)
        box_faces = np.array(
            [
                # Bottom face
                [0, 2, 1],
                [0, 3, 2],
                # Top face
                [4, 5, 6],
                [4, 6, 7],
                # Front face
                [0, 1, 5],
                [0, 5, 4],
                # Back face
                [2, 3, 7],
                [2, 7, 6],
                # Left face
                [0, 4, 7],
                [0, 7, 3],
                # Right face
                [1, 2, 6],
                [1, 6, 5],
            ],
            dtype=np.int32,
        )

        # Get initial box position from body state
        body_q = self.state_0.body_q.numpy()
        if len(body_q) > 0:
            # body_q contains transforms: [px, py, pz, qx, qy, qz, qw]
            pos = body_q[self.box_body_index][:3]
            transformed_verts = box_verts + pos

            ps_mesh = ps.register_surface_mesh("RigidBox", transformed_verts, box_faces)
            ps_mesh.set_color((0.9, 0.4, 0.2))  # Orange for rigid body
            ps_mesh.set_smooth_shade(False)  # Flat shading for box

            # Store for updates
            self.box_verts_local = box_verts
            self.box_faces = box_faces
            self.ps_box_mesh = ps_mesh

    def _register_gear_mesh(self):
        """Create and register a polyscope mesh for the rigid body gear."""
        # Get initial gear position from body state
        body_q = self.state_0.body_q.numpy()
        if len(body_q) > self.gear_body_index:
            # body_q contains transforms: [px, py, pz, qx, qy, qz, qw]
            transform = body_q[self.gear_body_index]
            pos = transform[:3]
            quat = transform[3:7]

            # Transform gear vertices
            transformed_verts = self._transform_vertices(self.gear_verts_local, pos, quat)

            ps_mesh = ps.register_surface_mesh("RigidGear", transformed_verts, self.gear_faces)
            ps_mesh.set_color((0.7, 0.7, 0.2))  # Gold/yellow for gear
            ps_mesh.set_smooth_shade(False)  # Flat shading for gear teeth

            self.ps_gear_mesh = ps_mesh

    def update_ps_meshes(self):
        """Override to also update the rigid body meshes (box and gear)."""
        # Update particle-based meshes (cloth, softbody)
        super().update_ps_meshes()

        if self.model.body_count > 0:
            body_q = self.state_0.body_q.numpy()

            # Update rigid body box
            if hasattr(self, "ps_box_mesh") and len(body_q) > self.box_body_index:
                transform = body_q[self.box_body_index]
                pos = transform[:3]
                quat = transform[3:7]  # qx, qy, qz, qw
                transformed_verts = self._transform_vertices(self.box_verts_local, pos, quat)
                self.ps_box_mesh.update_vertex_positions(transformed_verts)

            # Update rigid body gear
            if hasattr(self, "ps_gear_mesh") and len(body_q) > self.gear_body_index:
                transform = body_q[self.gear_body_index]
                pos = transform[:3]
                quat = transform[3:7]
                transformed_verts = self._transform_vertices(self.gear_verts_local, pos, quat)
                self.ps_gear_mesh.update_vertex_positions(transformed_verts)

    def _transform_vertices(self, verts, pos, quat):
        """Transform vertices by quaternion rotation and translation."""
        # quat is [qx, qy, qz, qw]
        qx, qy, qz, qw = quat

        # Build rotation matrix from quaternion
        rot_mat = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )

        # Apply rotation and translation
        transformed = verts @ rot_mat.T + pos
        return transformed.astype(np.float32)


def main():
    """Main entry point for running the example."""
    sim = MultiphysicsDropSimulator(config)
    sim.finalize()
    sim.simulate()


if __name__ == "__main__":
    main()
