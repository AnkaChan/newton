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
# - Volumetric soft bodies (hippo and bunny from VTK files)
# - Rigid bodies (box and gear)
# - A cloth sheet
#
# All objects drop onto the cloth under gravity, showcasing coupled
# soft body, rigid body, and cloth simulation.
#
# Command: python -m newton.examples.multiphysics.example_multiphysics_drop
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

    return np.array(vertices, dtype=np.float32), np.array(tet_indices, dtype=np.int32)


def create_gear_mesh(
    num_teeth: int = 12,
    outer_radius: float = 1.0,
    tooth_height: float = 0.25,
    thickness: float = 0.3,
    hole_radius: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 3D gear mesh with teeth."""
    vertices = []
    faces = []

    half_thickness = thickness / 2.0
    tooth_outer_radius = outer_radius + tooth_height
    tooth_angle = np.pi / num_teeth

    # Generate the 2D gear profile
    profile_points = []
    for i in range(num_teeth):
        base_angle = i * 2 * np.pi / num_teeth

        # Gap
        gap_start = base_angle
        profile_points.append((outer_radius * np.cos(gap_start), outer_radius * np.sin(gap_start)))

        # Tooth base start
        tooth_start = base_angle + tooth_angle * 0.3
        profile_points.append((outer_radius * np.cos(tooth_start), outer_radius * np.sin(tooth_start)))

        # Tooth tip start
        tip_start = base_angle + tooth_angle * 0.4
        profile_points.append((tooth_outer_radius * np.cos(tip_start), tooth_outer_radius * np.sin(tip_start)))

        # Tooth tip end
        tip_end = base_angle + tooth_angle * 1.6
        profile_points.append((tooth_outer_radius * np.cos(tip_end), tooth_outer_radius * np.sin(tip_end)))

        # Tooth base end
        tooth_end = base_angle + tooth_angle * 1.7
        profile_points.append((outer_radius * np.cos(tooth_end), outer_radius * np.sin(tooth_end)))

    num_profile = len(profile_points)

    # Create hole profile
    num_hole_segments = num_teeth * 2
    hole_angles = np.linspace(0, 2 * np.pi, num_hole_segments, endpoint=False)
    hole_points = [(hole_radius * np.cos(a), hole_radius * np.sin(a)) for a in hole_angles]

    # Build 3D vertices
    top_outer_start = len(vertices)
    for px, py in profile_points:
        vertices.append([px, py, half_thickness])

    top_hole_start = len(vertices)
    for px, py in hole_points:
        vertices.append([px, py, half_thickness])

    bot_outer_start = len(vertices)
    for px, py in profile_points:
        vertices.append([px, py, -half_thickness])

    bot_hole_start = len(vertices)
    for px, py in hole_points:
        vertices.append([px, py, -half_thickness])

    top_center_idx = len(vertices)
    vertices.append([0, 0, half_thickness])
    bot_center_idx = len(vertices)
    vertices.append([0, 0, -half_thickness])

    # Create faces
    for i in range(num_profile):
        i_next = (i + 1) % num_profile
        faces.append([top_center_idx, top_outer_start + i, top_outer_start + i_next])

    for i in range(num_profile):
        i_next = (i + 1) % num_profile
        faces.append([bot_center_idx, bot_outer_start + i_next, bot_outer_start + i])

    for i in range(num_profile):
        i_next = (i + 1) % num_profile
        faces.append([top_outer_start + i, bot_outer_start + i, bot_outer_start + i_next])
        faces.append([top_outer_start + i, bot_outer_start + i_next, top_outer_start + i_next])

    for i in range(num_hole_segments):
        i_next = (i + 1) % num_hole_segments
        faces.append([top_hole_start + i, top_hole_start + i_next, bot_hole_start + i_next])
        faces.append([top_hole_start + i, bot_hole_start + i_next, bot_hole_start + i])

    for i in range(num_hole_segments):
        i_next = (i + 1) % num_hole_segments
        faces.append([top_center_idx, top_hole_start + i_next, top_hole_start + i])
        faces.append([bot_center_idx, bot_hole_start + i, bot_hole_start + i_next])

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0

        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 5

        # Build model
        builder = newton.ModelBuilder(gravity=-980.0)  # cm/s²

        # Add soft body hippo
        hippo_path = os.path.join(os.path.dirname(__file__), "hippo.vtk")
        if os.path.exists(hippo_path):
            hippo_verts, hippo_tets = load_vtk_tet_mesh(hippo_path)
            hippo_vertices = [(v[0], v[1], v[2]) for v in hippo_verts]
            builder.add_soft_mesh(
                pos=wp.vec3(30.0, -30.0, 250.0),
                rot=wp.quat_identity(),
                scale=50.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=hippo_vertices,
                indices=hippo_tets.flatten().tolist(),
                density=0.0003,
                k_mu=5.0e4,
                k_lambda=5.0e4,
                k_damp=1e-9,
                particle_radius=1.0,
            )

        # Add soft body bunny
        bunny_path = os.path.join(os.path.dirname(__file__), "bunny_small.vtk")
        if os.path.exists(bunny_path):
            bunny_verts, bunny_tets = load_vtk_tet_mesh(bunny_path)
            bunny_vertices = [(v[0], v[1], v[2]) for v in bunny_verts]
            builder.add_soft_mesh(
                pos=wp.vec3(-30.0, 30.0, 250.0),
                rot=wp.quat_identity(),
                scale=5.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=bunny_vertices,
                indices=bunny_tets.flatten().tolist(),
                density=0.0003,
                k_mu=5.0e4,
                k_lambda=5.0e4,
                k_damp=1e-9,
                particle_radius=1.0,
            )

        # Add rigid body box
        body_box = builder.add_body(
            xform=wp.transform(p=wp.vec3(-30.0, -30.0, 220.0), q=wp.quat_identity()),
            key="box",
        )
        box_cfg = newton.ModelBuilder.ShapeConfig()
        box_cfg.density = 0.0005
        box_cfg.ke = 1.0e5
        box_cfg.kd = 1e-7
        box_cfg.mu = 0.5
        builder.add_shape_box(body_box, hx=15.0, hy=15.0, hz=15.0, cfg=box_cfg)

        # Add rigid body gear
        gear_verts, gear_faces = create_gear_mesh(
            num_teeth=12,
            outer_radius=1.0,
            tooth_height=0.25,
            thickness=0.3,
            hole_radius=0.15,
        )
        gear_scale = 15.0
        gear_verts = gear_verts * gear_scale
        gear_mesh = newton.Mesh(vertices=gear_verts, indices=gear_faces.flatten())

        body_gear = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(30.0, 30.0, 280.0),
                q=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2),
            ),
            key="gear",
        )
        gear_cfg = newton.ModelBuilder.ShapeConfig()
        gear_cfg.density = 0.0008
        gear_cfg.ke = 1.0e5
        gear_cfg.kd = 1e-3
        gear_cfg.mu = 0.5
        builder.add_shape_mesh(body_gear, mesh=gear_mesh, cfg=gear_cfg)

        # Add cloth grid
        builder.add_cloth_grid(
            pos=wp.vec3(-100.0, -100.0, 100.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            fix_left=True,
            fix_right=True,
            dim_x=80,
            dim_y=80,
            cell_x=3.0,
            cell_y=3.0,
            mass=0.05,
            tri_ke=5e5,
            tri_ka=5e5,
            tri_kd=1e-5,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=1.0,
        )

        # Add ground plane
        builder.add_ground_plane()

        # Color the meshes
        builder.color(include_bending=True)

        # Finalize model
        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1e-5
        self.model.soft_contact_mu = 0.2

        # Create solver
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.3,
            particle_self_contact_margin=0.5,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=1.5,
        )

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Disable collision detection
        self.contacts = None

        self.viewer.set_model(self.model)

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


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
