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

import math
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

    # Build 3D vertices
    # Top face outer profile
    top_outer_start = len(vertices)
    for px, py in profile_points:
        vertices.append([px, py, half_thickness])

    # Bottom face outer profile
    bot_outer_start = len(vertices)
    for px, py in profile_points:
        vertices.append([px, py, -half_thickness])

    # Center vertices for top and bottom (for fan triangulation)
    top_center_idx = len(vertices)
    vertices.append([0, 0, half_thickness])
    bot_center_idx = len(vertices)
    vertices.append([0, 0, -half_thickness])

    # Top face: fan from center to outer edge
    for i in range(num_profile):
        i_next = (i + 1) % num_profile
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
    # Ground plane (created manually with friction)
    "has_ground": False,  # We create ground manually with friction
    "ground_height": 0.0,
    "ground_ke": 1.0e5,
    "ground_kd": 1e-5,
    "ground_mu": 0.5,  # Ground friction
    # Shared soft body material parameters
    "softbody_density": 0.0003,  # g/cm³ (same as 500 kg/m³)
    "softbody_k_mu": 5.0e4,
    "softbody_k_lambda": 5.0e4,
    "softbody_k_damp": 1e-9,
    "softbody_particle_radius": 1,
    # Mass drop configuration (3 layers, 4 objects each, shuffled)
    "mass_drop_seed": 42,  # Random seed for reproducibility
    "mass_drop_layers": 5,
    "mass_drop_objects_per_layer": 4,  # hippo, bunny, box, gear
    "mass_drop_base_height": 140.0,  # cm
    "mass_drop_layer_spacing": 50.0,  # cm between layers
    "mass_drop_grid_spacing": 70.0,  # cm between objects in layer
    # Soft body meshes
    "hippo_vtk_path": "hippo.vtk",
    "hippo_scale": 50.0,  # cm
    "bunny_vtk_path": "bunny_small.vtk",
    "bunny_scale": 5.0,  # cm
    # Rigid body contact margin
    "rigid_contact_margin": 1.0,  # cm
    # Rigid body box parameters
    "box_half_extents": (12.0, 12.0, 12.0),  # cm
    "box_density": 0.0005,
    "box_ke": 1.0e5,
    "box_kd": 1e-7,
    "box_mu": 0.5,
    # Rigid body gear parameters
    "gear_scale": 15.0,  # cm
    "gear_num_teeth": 12,
    "gear_outer_radius": 1.0,  # relative (scaled by gear_scale)
    "gear_inner_radius": 0.6,
    "gear_tooth_height": 0.25,
    "gear_thickness": 0.6,
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
    "experiment_name": "4x5",  # NxM: N objects per layer, M layers
    "output_timestamp": True,  # Append timestamp to experiment folder
    "output_ext": "npy",  # Output format
    "write_output": True,
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

        # Add ground plane with friction
        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.ke = cfg("ground_ke")
        ground_cfg.kd = cfg("ground_kd")
        ground_cfg.mu = cfg("ground_mu")
        self.builder.add_ground_plane(height=cfg("ground_height"), cfg=ground_cfg)

        # Set random seed for reproducibility
        np.random.seed(cfg("mass_drop_seed"))

        # Load soft body meshes
        hippo_path = cfg("hippo_vtk_path")
        if not os.path.isabs(hippo_path):
            hippo_path = os.path.join(os.path.dirname(__file__), hippo_path)
        hippo_verts, hippo_tets = load_vtk_tet_mesh(hippo_path)
        hippo_vertices = [(v[0], v[1], v[2]) for v in hippo_verts]
        hippo_indices = hippo_tets.flatten().tolist()
        print(f"Loaded hippo mesh: {len(hippo_verts)} vertices, {len(hippo_tets)} tetrahedra")

        bunny_path = cfg("bunny_vtk_path")
        if not os.path.isabs(bunny_path):
            bunny_path = os.path.join(os.path.dirname(__file__), bunny_path)
        bunny_verts, bunny_tets = load_vtk_tet_mesh(bunny_path)
        bunny_vertices = [(v[0], v[1], v[2]) for v in bunny_verts]
        bunny_indices = bunny_tets.flatten().tolist()
        print(f"Loaded bunny mesh: {len(bunny_verts)} vertices, {len(bunny_tets)} tetrahedra")

        # Generate gear mesh
        gear_verts_base, gear_faces_base = create_gear_mesh(
            num_teeth=cfg("gear_num_teeth"),
            outer_radius=cfg("gear_outer_radius"),
            inner_radius=cfg("gear_inner_radius"),
            tooth_height=cfg("gear_tooth_height"),
            thickness=cfg("gear_thickness"),
            hole_radius=cfg("gear_hole_radius"),
        )
        gear_verts_scaled = gear_verts_base * cfg("gear_scale")

        # Mass drop configuration
        num_layers = cfg("mass_drop_layers")
        base_height = cfg("mass_drop_base_height")
        layer_spacing = cfg("mass_drop_layer_spacing")
        grid_spacing = cfg("mass_drop_grid_spacing")

        # Calculate cloth center for grid placement
        cloth_pos = cfg("cloth_pos")
        cloth_dim_x = cfg("cloth_dim_x")
        cloth_dim_y = cfg("cloth_dim_y")
        cloth_cell_x = cfg("cloth_cell_x")
        cloth_cell_y = cfg("cloth_cell_y")
        grid_center_x = cloth_pos[0] + (cloth_dim_x * cloth_cell_x) / 2.0
        grid_center_y = cloth_pos[1] + (cloth_dim_y * cloth_cell_y) / 2.0

        # Track rigid bodies for visualization
        self.mass_drop_boxes = []
        self.mass_drop_gears = []
        # Track soft bodies with their particle ranges and mesh data
        self.mass_drop_softbodies = []  # [(start_idx, num_verts, type_name, local_verts, tet_indices)]

        # Object types: 0=hippo, 1=bunny, 2=box, 3=gear
        object_types = [0, 1, 2, 3]
        total_objects = 0

        for layer in range(num_layers):
            z_height = base_height + layer * layer_spacing

            # Shuffle object order for this layer
            shuffled_types = object_types.copy()
            np.random.shuffle(shuffled_types)

            # 2x2 grid positions for 4 objects
            positions = [
                (grid_center_x - grid_spacing / 2, grid_center_y - grid_spacing / 2),
                (grid_center_x + grid_spacing / 2, grid_center_y - grid_spacing / 2),
                (grid_center_x - grid_spacing / 2, grid_center_y + grid_spacing / 2),
                (grid_center_x + grid_spacing / 2, grid_center_y + grid_spacing / 2),
            ]

            for i, obj_type in enumerate(shuffled_types):
                x_pos, y_pos = positions[i]
                # Add small jitter
                x_pos += np.random.uniform(-5.0, 5.0)
                y_pos += np.random.uniform(-5.0, 5.0)

                if obj_type == 0:
                    # Hippo (soft body)
                    start_idx = self.builder.particle_count
                    self.builder.add_soft_mesh(
                        pos=wp.vec3(x_pos, y_pos, z_height),
                        rot=wp.quat_identity(),
                        scale=cfg("hippo_scale"),
                        vel=wp.vec3(0.0, 0.0, 0.0),
                        vertices=hippo_vertices,
                        indices=hippo_indices,
                        density=cfg("softbody_density"),
                        k_mu=cfg("softbody_k_mu"),
                        k_lambda=cfg("softbody_k_lambda"),
                        k_damp=cfg("softbody_k_damp"),
                        particle_radius=cfg("softbody_particle_radius"),
                    )
                    num_verts = self.builder.particle_count - start_idx
                    self.mass_drop_softbodies.append((
                        start_idx, num_verts, "hippo",
                        np.array(hippo_vertices, dtype=np.float32),
                        np.array(hippo_tets, dtype=np.int32),
                    ))
                elif obj_type == 1:
                    # Bunny (soft body)
                    start_idx = self.builder.particle_count
                    self.builder.add_soft_mesh(
                        pos=wp.vec3(x_pos, y_pos, z_height),
                        rot=wp.quat_identity(),
                        scale=cfg("bunny_scale"),
                        vel=wp.vec3(0.0, 0.0, 0.0),
                        vertices=bunny_vertices,
                        indices=bunny_indices,
                        density=cfg("softbody_density"),
                        k_mu=cfg("softbody_k_mu"),
                        k_lambda=cfg("softbody_k_lambda"),
                        k_damp=cfg("softbody_k_damp"),
                        particle_radius=cfg("softbody_particle_radius"),
                    )
                    num_verts = self.builder.particle_count - start_idx
                    self.mass_drop_softbodies.append((
                        start_idx, num_verts, "bunny",
                        np.array(bunny_vertices, dtype=np.float32),
                        np.array(bunny_tets, dtype=np.int32),
                    ))
                elif obj_type == 2:
                    # Box (rigid body)
                    box_half = cfg("box_half_extents")
                    body_box = self.builder.add_body(
                        xform=wp.transform(
                            p=wp.vec3(x_pos, y_pos, z_height),
                            q=wp.quat_identity(),
                        ),
                        key=f"box_{total_objects}",
                    )
                    box_shape_cfg = newton.ModelBuilder.ShapeConfig()
                    box_shape_cfg.density = cfg("box_density")
                    box_shape_cfg.ke = cfg("box_ke")
                    box_shape_cfg.kd = cfg("box_kd")
                    box_shape_cfg.mu = cfg("box_mu")
                    box_shape_cfg.contact_margin = cfg("rigid_contact_margin")
                    self.builder.add_shape_box(
                        body_box,
                        hx=box_half[0],
                        hy=box_half[1],
                        hz=box_half[2],
                        cfg=box_shape_cfg,
                    )
                    # Store for visualization
                    hx, hy, hz = box_half
                    box_verts_local = np.array([
                        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
                    ], dtype=np.float32)
                    box_faces_local = np.array([
                        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
                    ], dtype=np.int32)
                    self.mass_drop_boxes.append((body_box, box_verts_local, box_faces_local))
                else:
                    # Gear (rigid body)
                    gear_mesh = newton.Mesh(
                        vertices=gear_verts_scaled,
                        indices=gear_faces_base.flatten(),
                    )
                    body_gear = self.builder.add_body(
                        xform=wp.transform(
                            p=wp.vec3(x_pos, y_pos, z_height),
                            q=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2),
                        ),
                        key=f"gear_{total_objects}",
                    )
                    gear_shape_cfg = newton.ModelBuilder.ShapeConfig()
                    gear_shape_cfg.density = cfg("gear_density")
                    gear_shape_cfg.ke = cfg("gear_ke")
                    gear_shape_cfg.kd = cfg("gear_kd")
                    gear_shape_cfg.mu = cfg("gear_mu")
                    gear_shape_cfg.contact_margin = cfg("rigid_contact_margin")
                    self.builder.add_shape_mesh(
                        body_gear,
                        mesh=gear_mesh,
                        cfg=gear_shape_cfg,
                    )
                    self.mass_drop_gears.append((body_gear, gear_verts_scaled.copy(), gear_faces_base))

                total_objects += 1

        print(f"Created {total_objects} objects in {num_layers} layers (shuffled)")

        # Track particle count before adding cloth
        self.softbody_particle_count = self.builder.particle_count
        self.cloth_particle_start = self.builder.particle_count  # Same as softbody_particle_count

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
        """Store rigid body info and save initial meshes."""
        self._save_initial_meshes()

    def _save_initial_meshes(self):
        """Save all initial meshes separately with their transformations."""
        def cfg(key):
            return get_config_value(self.config, key)

        mesh_dir = os.path.join(self.output_path, "initial_meshes")
        os.makedirs(mesh_dir, exist_ok=True)

        # Get initial states
        particle_q = self.model.particle_q.numpy()
        body_q = self.state_0.body_q.numpy() if self.model.body_count > 0 else np.array([])

        # Track mesh info for metadata
        mesh_info = {
            "soft_bodies": [],
            "rigid_bodies": [],
            "cloth": None,
        }

        # Save each soft body mesh separately
        for idx, (start_idx, num_verts, type_name, local_verts, tet_indices) in enumerate(self.mass_drop_softbodies):
            # Get world-space vertices for this soft body
            world_verts = particle_q[start_idx:start_idx + num_verts]

            # Get faces for this soft body (faces that use these vertex indices)
            softbody_faces = []
            for face in self.faces:
                if all(start_idx <= f_idx < start_idx + num_verts for f_idx in face):
                    # Remap to local indices
                    softbody_faces.append([f_idx - start_idx for f_idx in face])

            if len(softbody_faces) > 0:
                softbody_faces = np.array(softbody_faces, dtype=np.int32)
                name = f"{type_name}_{idx}"

                self._save_ply(os.path.join(mesh_dir, f"{name}.ply"), world_verts, softbody_faces)
                np.save(os.path.join(mesh_dir, f"{name}_vertices_local.npy"), local_verts)
                np.save(os.path.join(mesh_dir, f"{name}_vertices_world.npy"), world_verts)
                np.save(os.path.join(mesh_dir, f"{name}_faces.npy"), softbody_faces)
                np.save(os.path.join(mesh_dir, f"{name}_tet_indices.npy"), tet_indices)

                mesh_info["soft_bodies"].append({
                    "name": name,
                    "type": type_name,
                    "particle_start": start_idx,
                    "num_vertices": num_verts,
                    "num_faces": len(softbody_faces),
                })

        # Save cloth mesh
        if self.softbody_particle_count < len(particle_q):
            cloth_verts = particle_q[self.softbody_particle_count:]
            cloth_faces = []
            for face in self.faces:
                if all(idx >= self.softbody_particle_count for idx in face):
                    # Remap indices to cloth-local
                    cloth_faces.append([idx - self.softbody_particle_count for idx in face])
            if len(cloth_faces) > 0:
                cloth_faces = np.array(cloth_faces)
                self._save_ply(os.path.join(mesh_dir, "cloth.ply"), cloth_verts, cloth_faces)
                np.save(os.path.join(mesh_dir, "cloth_vertices.npy"), cloth_verts)
                np.save(os.path.join(mesh_dir, "cloth_faces.npy"), cloth_faces)
                mesh_info["cloth"] = {
                    "name": "cloth",
                    "particle_start": self.cloth_particle_start,
                    "num_vertices": len(cloth_verts),
                    "num_faces": len(cloth_faces),
                }

        # Save rigid body boxes with transforms
        for i, (body_idx, local_verts, faces) in enumerate(self.mass_drop_boxes):
            if body_idx < len(body_q):
                transform = body_q[body_idx]
                pos = transform[:3]
                quat = transform[3:7]
                world_verts = self._transform_vertices(local_verts, pos, quat)

                name = f"box_{i}"
                self._save_ply(os.path.join(mesh_dir, f"{name}.ply"), world_verts, faces)
                np.save(os.path.join(mesh_dir, f"{name}_vertices_local.npy"), local_verts)
                np.save(os.path.join(mesh_dir, f"{name}_vertices_world.npy"), world_verts)
                np.save(os.path.join(mesh_dir, f"{name}_faces.npy"), faces)
                np.save(os.path.join(mesh_dir, f"{name}_transform.npy"), transform)

                mesh_info["rigid_bodies"].append({
                    "name": name,
                    "type": "box",
                    "body_idx": body_idx,
                    "position": pos.tolist(),
                    "quaternion": quat.tolist(),  # [qx, qy, qz, qw]
                })

        # Save rigid body gears with transforms
        for i, (body_idx, local_verts, faces) in enumerate(self.mass_drop_gears):
            if body_idx < len(body_q):
                transform = body_q[body_idx]
                pos = transform[:3]
                quat = transform[3:7]
                world_verts = self._transform_vertices(local_verts, pos, quat)

                name = f"gear_{i}"
                self._save_ply(os.path.join(mesh_dir, f"{name}.ply"), world_verts, faces)
                np.save(os.path.join(mesh_dir, f"{name}_vertices_local.npy"), local_verts)
                np.save(os.path.join(mesh_dir, f"{name}_vertices_world.npy"), world_verts)
                np.save(os.path.join(mesh_dir, f"{name}_faces.npy"), faces)
                np.save(os.path.join(mesh_dir, f"{name}_transform.npy"), transform)

                mesh_info["rigid_bodies"].append({
                    "name": name,
                    "type": "gear",
                    "body_idx": body_idx,
                    "position": pos.tolist(),
                    "quaternion": quat.tolist(),
                })

        # Save mesh info as JSON
        import json
        with open(os.path.join(mesh_dir, "mesh_info.json"), "w") as f:
            json.dump(mesh_info, f, indent=2)

        print(f"Saved initial meshes to {mesh_dir}/")
        print(f"  - Soft bodies: {len(mesh_info['soft_bodies'])} meshes")
        print(f"  - Rigid bodies: {len(mesh_info['rigid_bodies'])} meshes")
        print(f"  - Cloth: {'yes' if mesh_info['cloth'] else 'no'}")

    def _save_ply(self, filepath, vertices, faces):
        """Save mesh to PLY format."""
        with open(filepath, "w") as f:
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

    def setup_polyscope_meshes(self):
        """Override to set up separate meshes for cloth and rigid bodies."""
        if not self.do_rendering:
            return

        # Register cloth/softbody mesh (particles only)
        verts = self.model.particle_q.numpy()
        self.register_ps_mesh(
            name="Simulation",
            vertices=verts,
            faces=self.faces,
            vertex_indices=None,
            color=(0.2, 0.6, 0.9),  # Light blue
        )

        # Register all rigid body meshes
        self._register_mass_drop_meshes()

    def _register_mass_drop_meshes(self):
        """Register polyscope meshes for all mass drop rigid bodies."""
        body_q = self.state_0.body_q.numpy()
        if len(body_q) == 0:
            return

        # Register all boxes
        self.ps_box_meshes = []
        for i, (body_idx, local_verts, faces) in enumerate(self.mass_drop_boxes):
            if body_idx < len(body_q):
                transform = body_q[body_idx]
                pos = transform[:3]
                quat = transform[3:7]
                transformed_verts = self._transform_vertices(local_verts, pos, quat)

                ps_mesh = ps.register_surface_mesh(f"Box_{i}", transformed_verts, faces)
                ps_mesh.set_color((0.9, 0.4, 0.2))  # Orange
                ps_mesh.set_smooth_shade(False)
                self.ps_box_meshes.append((body_idx, local_verts, faces, ps_mesh))

        # Register all gears
        self.ps_gear_meshes = []
        for i, (body_idx, local_verts, faces) in enumerate(self.mass_drop_gears):
            if body_idx < len(body_q):
                transform = body_q[body_idx]
                pos = transform[:3]
                quat = transform[3:7]
                transformed_verts = self._transform_vertices(local_verts, pos, quat)

                ps_mesh = ps.register_surface_mesh(f"Gear_{i}", transformed_verts, faces)
                ps_mesh.set_color((0.7, 0.7, 0.2))  # Gold
                ps_mesh.set_smooth_shade(False)
                self.ps_gear_meshes.append((body_idx, local_verts, faces, ps_mesh))

    def update_ps_meshes(self):
        """Override to also update all rigid body meshes."""
        # Update particle-based meshes (cloth, softbody)
        super().update_ps_meshes()

        if self.model.body_count > 0:
            body_q = self.state_0.body_q.numpy()

            # Update all boxes
            if hasattr(self, "ps_box_meshes"):
                for body_idx, local_verts, faces, ps_mesh in self.ps_box_meshes:
                    if body_idx < len(body_q):
                        transform = body_q[body_idx]
                        pos = transform[:3]
                        quat = transform[3:7]
                        transformed_verts = self._transform_vertices(local_verts, pos, quat)
                        ps_mesh.update_vertex_positions(transformed_verts)

            # Update all gears
            if hasattr(self, "ps_gear_meshes"):
                for body_idx, local_verts, faces, ps_mesh in self.ps_gear_meshes:
                    if body_idx < len(body_q):
                        transform = body_q[body_idx]
                        pos = transform[:3]
                        quat = transform[3:7]
                        transformed_verts = self._transform_vertices(local_verts, pos, quat)
                        ps_mesh.update_vertex_positions(transformed_verts)

    def save_output(self, frame_id):
        """Override to also save rigid body transforms."""
        # Call parent to save particle positions
        super().save_output(frame_id)

        # Also save body_q for rigid bodies (if we have any)
        if self.model.body_count > 0 and self.output_ext == "npy":
            body_q = self.state_0.body_q.numpy()
            out_file = os.path.join(self.output_path, f"body_q_{frame_id:06d}.npy")
            np.save(out_file, body_q)

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
