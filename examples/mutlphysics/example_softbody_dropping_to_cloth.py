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
# Example Softbody Dropping to Cloth
#
# This simulation demonstrates a volumetric soft body (pyramid tet mesh)
# dropping onto a cloth grid. Uses the Simulator base class for consistent
# simulation infrastructure.
#
# Command: python -m newton.examples.mutlphysics.example_softbody_dropping_to_cloth
#
###########################################################################

import numpy as np
import warp as wp

from newton.examples.cloth.M01_Simulator import Simulator

# Pyramid tet mesh data (same as newton.tests.test_softbody.py)
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

# Configuration dict - single source of truth for all parameters
config = {
    "name": "softbody_dropping_to_cloth",
    # Simulation timing
    "fps": 60,
    "sim_substeps": 10,
    "sim_num_frames": 1000,
    "iterations": 10,
    # Solver settings
    "use_cuda_graph": True,
    "handle_self_contact": True,
    "use_tile_solve": True,
    "self_contact_radius": 0.01,
    "self_contact_margin": 0.02,
    "include_bending": True,
    # Physics (using meters)
    "up_axis": "z",
    "gravity": -9.8,
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 1e-5,
    "soft_contact_mu": 1.0,
    # Ground plane
    "has_ground": True,
    "ground_height": 0.0,
    # Soft body parameters
    "softbody_pos": (0.0, 0.0, 2.0),
    "softbody_scale": 0.2,
    "softbody_density": 1.0e3,
    "softbody_k_mu": 1.0e5,
    "softbody_k_lambda": 1.0e5,
    "softbody_k_damp": 1e-5,
    # Cloth grid parameters
    "cloth_pos": (-1.0, -1.0, 1.0),
    "cloth_dim_x": 40,
    "cloth_dim_y": 40,
    "cloth_cell_x": 0.05,
    "cloth_cell_y": 0.05,
    "cloth_mass": 0.0005,
    "cloth_tri_ke": 1e5,
    "cloth_tri_ka": 1e5,
    "cloth_tri_kd": 1e-5,
    "cloth_edge_ke": 0.01,
    "cloth_edge_kd": 1e-2,
    "cloth_particle_radius": 0.05,
    # Visualization
    "do_rendering": True,
    "show_ground_plane": True,
    # Output (disabled by default)
    "write_output": False,
    "write_video": False,
}


class SoftbodyDroppingToClothSimulator(Simulator):
    """
    Simulator for a soft body dropping onto a cloth grid.

    Demonstrates multi-physics interaction between volumetric soft body
    and thin shell cloth using the VBD solver.
    """

    def custom_init(self):
        """Add soft mesh and cloth grid to the simulation."""
        cfg = self.config

        # Add soft body (pyramid tet mesh)
        self.builder.add_soft_mesh(
            pos=wp.vec3(*cfg["softbody_pos"]),
            rot=wp.quat_identity(),
            scale=cfg["softbody_scale"],
            vel=wp.vec3(0.0),
            vertices=PYRAMID_PARTICLES,
            indices=PYRAMID_TET_INDICES.flatten().tolist(),
            density=cfg["softbody_density"],
            k_mu=cfg["softbody_k_mu"],
            k_lambda=cfg["softbody_k_lambda"],
            k_damp=cfg["softbody_k_damp"],
        )

        # Add cloth grid (fixed on left and right edges)
        self.builder.add_cloth_grid(
            pos=wp.vec3(*cfg["cloth_pos"]),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            fix_left=True,
            fix_right=True,
            dim_x=cfg["cloth_dim_x"],
            dim_y=cfg["cloth_dim_y"],
            cell_x=cfg["cloth_cell_x"],
            cell_y=cfg["cloth_cell_y"],
            mass=cfg["cloth_mass"],
            tri_ke=cfg["cloth_tri_ke"],
            tri_ka=cfg["cloth_tri_ka"],
            tri_kd=cfg["cloth_tri_kd"],
            edge_ke=cfg["cloth_edge_ke"],
            edge_kd=cfg["cloth_edge_kd"],
            particle_radius=cfg["cloth_particle_radius"],
        )


if __name__ == "__main__":
    sim = SoftbodyDroppingToClothSimulator(config)
    sim.finalize()
    sim.simulate()
