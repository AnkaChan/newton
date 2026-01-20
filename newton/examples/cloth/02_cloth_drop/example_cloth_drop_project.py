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
# Example Cloth Drop
#
# This simulation demonstrates dropping a cloth onto a cylinder collider
# using the Simulator base class from M01_Simulator.
#
###########################################################################

import itertools
import os

import numpy as np
import polyscope as ps
import tqdm
import warp as wp

import newton
from newton import ParticleFlags
from newton.examples.cloth.M01_Simulator import Simulator, default_config, get_config_value, read_obj

# =============================================================================
# Configuration
# =============================================================================

example_config = {
    **default_config,  # Start with defaults
    # Simulation timing
    "fps": 60,
    "sim_substeps": 20,
    "sim_num_frames": 1200,
    "iterations": 40,
    "bvh_rebuild_frames": 1,
    # Solver settings
    "use_cuda_graph": True,
    "handle_self_contact": True,
    "use_tile_solve": True,
    "self_contact_radius": 0.25,
    "self_contact_margin": 0.35,
    "topological_contact_filter_threshold": 1,
    "rest_shape_contact_exclusion_radius": 0.0,
    # Collision buffer settings - start small, let resize grow as needed
    # Based on 100-layer analysis: 99th percentile ~32 vertex, ~64 edge collisions
    "vertex_collision_buffer_pre_alloc": 32,  # Start conservative, will grow
    "edge_collision_buffer_pre_alloc": 40,  # Start conservative, will grow
    "collision_buffer_resize_frames": 1,  # Check and resize every 5 frames
    "collision_buffer_growth_ratio": 1.5,  # 50% headroom when growing
    "collision_detection_interval": 5,
    # Global physics settings
    "up_axis": "y",
    "gravity": -980.0,
    "soft_contact_ke": 5e4,
    "soft_contact_kd": 1e-7,
    "soft_contact_mu": 0.1,
    # Visualization
    "do_rendering": True,
    "show_ground_plane": True,
    "is_initially_paused": True,
    "zero_velocity_duration": 0.1,
    # Ground plane
    "has_ground": True,
    "ground_height": -30.0,
    # Static shape colliders (proper collision shapes, not cloth meshes)
    "static_colliders": {
        "cylinder": {
            "type": "cylinder",
            "position": (0.0, 20.0, 0.0),
            # No rotation needed - mesh is already vertical (Y-up), so rotate shape to match
            # Cylinder axis is Z in local frame; rotate -90° around X to align with Y-up mesh
            "rotation_axis": (0.0, 0.0, 1.0),
            "rotation_angle": -1.5708,  # -pi/2 radians = -90 degrees (Z->Y)
            "radius": 30.0,
            "half_height": 30.0,
            # Shape material properties
            "mu": 0.1,  # Friction coefficient
            # Optional: mesh for visualization (no rotation applied to mesh)
            "vis_mesh_path": "CylinderCollider_tri.obj",
            "vis_mesh_scale": 30.0,
        },
    },
    # Legacy mesh-based colliders (cloth meshes with zero mass) - now empty
    "colliders": {},
    # Cloths configuration (dict of name -> cloth params)
    "cloths": {
        "main_cloth": {
            "resolution": (60, 80),  # (Nx, Ny) vertices
            "size": (120.0, 160.0),  # Physical size (size_x, size_y)
            "position": (20.0, 66.0, 0.0),
            "rotation_axis": (1.0, 0.0, 0.0),
            "rotation_angle": 0.0,
            "scale": 1.0,
            "density": 0.02,
            "particle_radius": 0.5,  # Particle radius for collision handling
            "tri_ke": 1e4,
            "tri_ka": 1e4,
            "tri_kd": 1e-5,
            "edge_ke": 10.0,
            "edge_kd": 1e-1,
            # Multi-layer settings
            "num_layers": 100,  # Number of cloth layers
            "layer_spacing": 0.25,  # Distance between layers (in up_axis direction)
        },
    },
    # outputs
    "output_path": r"D:\Data\DAT_Sim",  # Directory to save output files
    "output_ext": "npy",  # "ply" "npy" or "usd"
    # "output_ext": "ply",  # "ply" "npy" or "usd"
    "write_output": True,
    "write_video": True,
    "recovery_state_save_steps": 100,
}


# =============================================================================
# Helper Functions
# =============================================================================


def generate_cloth_mesh(Nx, Ny, size_x=2.0, size_y=2.0, position=(0, 0)):
    """
    Generate a cloth mesh with cross tessellation.

    Args:
        Nx: Number of vertices along X axis
        Ny: Number of vertices along Y axis
        size_x: Physical size of the cloth along X axis
        size_y: Physical size of the cloth along Y axis
        position: Center position (x, y) of the cloth

    Returns:
        vertices: List of vertex positions (x, y, z)
        faces: List of face indices (triangles)
    """
    X = np.linspace(-0.5 * size_x + position[0], 0.5 * size_x + position[0], Nx)
    Y = np.linspace(-0.5 * size_y + position[1], 0.5 * size_y + position[1], Ny)

    X, Y = np.meshgrid(X, Y, indexing="ij")
    Z = np.zeros((Nx, Ny))

    vertices = []
    for i, j in itertools.product(range(Nx), range(Ny)):
        vertices.append((X[i, j], Z[i, j], Y[i, j]))

    faces = []
    for i, j in itertools.product(range(0, Nx - 1), range(0, Ny - 1)):
        vId = j + i * Ny

        if (j + i) % 2:
            faces.append((vId, vId + Ny + 1, vId + 1))
            faces.append((vId, vId + Ny, vId + Ny + 1))
        else:
            faces.append((vId, vId + Ny, vId + 1))
            faces.append((vId + Ny, vId + Ny + 1, vId + 1))

    return vertices, faces


# =============================================================================
# Cloth Drop Simulator
# =============================================================================


class ClothDropSimulator(Simulator):
    """
    Simulator that drops a cloth onto a cylinder collider.
    """

    def custom_init(self):
        """Add the collider and cloth meshes to the simulation."""

        def cfg(key):
            return get_config_value(self.config, key)

        # Track mesh info for separate polyscope meshes
        # Each entry: {"name": str, "vertex_start": int, "vertex_count": int, "faces": array}
        self._mesh_info: list = []
        self._collider_names: list = []
        self._cloth_names: list = []
        # Store static collider visualization data (vertices and faces for polyscope only)
        self._static_collider_vis: list = []

        # --- Add Static Shape Colliders ---
        static_colliders = cfg("static_colliders") or {}
        for name, sc_cfg in static_colliders.items():
            collider_type = sc_cfg["type"]
            position = wp.vec3(sc_cfg["position"])
            rotation_axis = wp.vec3(sc_cfg.get("rotation_axis", (0.0, 0.0, 1.0)))
            rotation_angle = sc_cfg.get("rotation_angle", 0.0)
            xform = wp.transform(position, wp.quat_from_axis_angle(rotation_axis, rotation_angle))

            # Create ShapeConfig with material properties
            shape_cfg = newton.ModelBuilder.ShapeConfig(
                mu=sc_cfg.get("mu", 0.5),  # Friction coefficient
            )

            # Add the static shape (body=-1 means static/world-attached)
            if collider_type == "cylinder":
                self.builder.add_shape_cylinder(
                    body=-1,
                    xform=xform,
                    radius=sc_cfg["radius"],
                    half_height=sc_cfg["half_height"],
                    cfg=shape_cfg,
                )
            elif collider_type == "box":
                self.builder.add_shape_box(
                    body=-1,
                    xform=xform,
                    hx=sc_cfg["hx"],
                    hy=sc_cfg["hy"],
                    hz=sc_cfg["hz"],
                    cfg=shape_cfg,
                )
            elif collider_type == "sphere":
                self.builder.add_shape_sphere(
                    body=-1,
                    xform=xform,
                    radius=sc_cfg["radius"],
                    cfg=shape_cfg,
                )
            elif collider_type == "capsule":
                self.builder.add_shape_capsule(
                    body=-1,
                    xform=xform,
                    radius=sc_cfg["radius"],
                    half_height=sc_cfg["half_height"],
                    cfg=shape_cfg,
                )

            # Load visualization mesh if provided (mesh stays as-is, no rotation)
            vis_mesh_path = sc_cfg.get("vis_mesh_path")
            if vis_mesh_path:
                if not os.path.isabs(vis_mesh_path):
                    vis_mesh_path = os.path.join(os.path.dirname(__file__), vis_mesh_path)
                vs_vis, fs_vis = read_obj(vis_mesh_path)
                vis_scale = sc_cfg.get("vis_mesh_scale", 1.0)

                # Transform vertices to world space (scale and translate only)
                vs_vis = np.array(vs_vis) * vis_scale
                pos_np = np.array([float(position[0]), float(position[1]), float(position[2])])
                vs_vis = vs_vis + pos_np

                self._static_collider_vis.append(
                    {
                        "name": f"static_collider_{name}",
                        "vertices": vs_vis,
                        "faces": np.array(fs_vis),
                    }
                )
            self._collider_names.append(f"static_collider_{name}")

        # --- Add Legacy Mesh Colliders (cloth meshes with zero mass) ---
        colliders = cfg("colliders") or {}
        for name, collider_cfg in colliders.items():
            collider_path = collider_cfg["mesh_path"]

            # Handle relative path
            if not os.path.isabs(collider_path):
                collider_path = os.path.join(os.path.dirname(__file__), collider_path)

            vs_collider, fs_collider = read_obj(collider_path)
            vertex_start = self.builder.particle_count
            num_verts = len(vs_collider)

            self.builder.add_cloth_mesh(
                pos=wp.vec3(collider_cfg["position"]),
                rot=wp.quat_from_axis_angle(
                    wp.vec3(collider_cfg["rotation_axis"]),
                    collider_cfg["rotation_angle"],
                ),
                scale=collider_cfg["scale"],
                vertices=[wp.vec3(v) for v in vs_collider],
                indices=list(itertools.chain(*fs_collider)),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=collider_cfg["density"],
                tri_ke=collider_cfg["tri_ke"],
                tri_ka=collider_cfg["tri_ka"],
                tri_kd=collider_cfg["tri_kd"],
                edge_ke=collider_cfg["edge_ke"],
                edge_kd=collider_cfg["edge_kd"],
            )

            # Store mesh info (faces are local indices, will be used directly)
            self._mesh_info.append(
                {
                    "name": f"collider_{name}",
                    "vertex_start": vertex_start,
                    "vertex_count": num_verts,
                    "faces": np.array(fs_collider),  # Local indices (0-based)
                    "is_collider": True,
                }
            )
            self._collider_names.append(f"collider_{name}")

            # Make collider static (zero mass, inactive)
            for i in range(vertex_start, vertex_start + num_verts):
                self.builder.particle_mass[i] = 0.0
                self.builder.particle_flags[i] &= ~ParticleFlags.ACTIVE

        # --- Add Cloths ---
        cloths = cfg("cloths") or {}
        for name, cloth_cfg in cloths.items():
            Nx, Ny = cloth_cfg["resolution"]
            size_x, size_y = cloth_cfg["size"]
            base_position = cloth_cfg["position"]
            num_layers = cloth_cfg.get("num_layers", 1)
            layer_spacing = cloth_cfg.get("layer_spacing", 5.0)

            # Generate cloth mesh once (reused for all layers)
            vertices_cloth, faces_cloth = generate_cloth_mesh(Nx, Ny, size_x, size_y)
            faces_cloth = np.array(faces_cloth)

            # Determine which axis to offset for layers based on up_axis
            up_axis = cfg("up_axis")
            axis_index = {"x": 0, "y": 1, "z": 2}.get(up_axis, 1)

            # Create each layer
            for layer_idx in tqdm.tqdm(range(num_layers), desc="Creating cloth layers"):
                # Compute layer position (offset along up axis)
                layer_offset = layer_idx * layer_spacing
                layer_position = list(base_position)
                layer_position[axis_index] += layer_offset
                layer_position = tuple(layer_position)

                vertex_start = self.builder.particle_count
                num_verts = len(vertices_cloth)

                self.builder.add_cloth_mesh(
                    pos=wp.vec3(layer_position),
                    rot=wp.quat_from_axis_angle(
                        wp.vec3(cloth_cfg["rotation_axis"]),
                        cloth_cfg["rotation_angle"],
                    ),
                    scale=cloth_cfg["scale"],
                    vertices=[wp.vec3(v) for v in vertices_cloth],
                    indices=faces_cloth.reshape(-1),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    density=cloth_cfg["density"],
                    particle_radius=cloth_cfg.get("particle_radius"),
                    tri_ke=cloth_cfg["tri_ke"],
                    tri_ka=cloth_cfg["tri_ka"],
                    tri_kd=cloth_cfg["tri_kd"],
                    edge_ke=cloth_cfg["edge_ke"],
                    edge_kd=cloth_cfg["edge_kd"],
                )

                # Store mesh info (faces are local indices)
                layer_name = f"cloth_{name}" if num_layers == 1 else f"cloth_{name}_layer{layer_idx}"
                self._mesh_info.append(
                    {
                        "name": layer_name,
                        "vertex_start": vertex_start,
                        "vertex_count": num_verts,
                        "faces": faces_cloth.copy(),  # Local indices (0-based)
                        "is_collider": False,
                        "layer_idx": layer_idx,
                    }
                )
                self._cloth_names.append(layer_name)

    def custom_finalize(self):
        """Store combined faces for file output."""
        # Build combined faces array for file output (with global indices)
        face_arrays = []
        for info in self._mesh_info:
            global_faces = info["faces"] + info["vertex_start"]
            face_arrays.append(global_faces)
        if face_arrays:
            self.faces = np.vstack(face_arrays)
        else:
            self.faces = np.array([], dtype=np.int32).reshape(0, 3)

    def setup_polyscope_meshes(self):
        """Register each collider and cloth as a separate polyscope mesh."""
        if not self.do_rendering:
            return

        all_verts = self.model.particle_q.numpy()

        # Color palette for cloth layers
        cloth_colors = [
            (0.2, 0.6, 0.9),  # Blue
            (0.9, 0.4, 0.3),  # Red-orange
            (0.3, 0.8, 0.4),  # Green
            (0.8, 0.5, 0.9),  # Purple
            (0.9, 0.7, 0.2),  # Yellow-orange
            (0.4, 0.8, 0.8),  # Cyan
        ]

        for info in self._mesh_info:
            name = info["name"]
            v_start = info["vertex_start"]
            v_count = info["vertex_count"]
            faces = info["faces"]  # Local indices
            is_collider = info["is_collider"]

            # Extract vertices for this mesh
            vertex_slice = slice(v_start, v_start + v_count)
            verts = all_verts[vertex_slice]

            # Register with different colors for colliders vs cloths
            if is_collider:
                color = (0.6, 0.6, 0.6)  # Gray for colliders
            else:
                # Cycle through colors for different layers
                layer_idx = info.get("layer_idx", 0)
                color = cloth_colors[layer_idx % len(cloth_colors)]

            self.register_ps_mesh(
                name=name,
                vertices=verts,
                faces=faces,
                vertex_indices=vertex_slice,
                color=color,
                smooth_shade=True,
            )

        # Register static collider visualization meshes (these don't update with simulation)
        for vis_info in self._static_collider_vis:
            ps.register_surface_mesh(
                vis_info["name"],
                vis_info["vertices"],
                vis_info["faces"],
                smooth_shade=True,
                color=(0.6, 0.6, 0.6),  # Gray for colliders
            ).set_transparency(0.3)

        # Apply any additional polyscope configuration
        self.configure_polyscope()

    def configure_polyscope(self):
        """
        Configure polyscope mesh appearance. Override to customize.
        """
        # Make colliders slightly transparent
        for name in self._collider_names:
            self.configure_ps_mesh(name, transparency=0.3)

    def step(self):
        """Execute one frame with zero velocity at start."""

        def cfg(key):
            return get_config_value(self.config, key)

        # Zero out velocity at the beginning of simulation
        zero_vel_duration = cfg("zero_velocity_duration") if "zero_velocity_duration" in (self.config or {}) else 0.1
        if self.sim_time < zero_vel_duration:
            self.state_0.particle_qd.zero_()

        # Call parent step
        super().step()

    def save_initial_meshes(self):
        """Save initial mesh topology as separate PLY files, numbered by order in particle_q."""
        if self.output_path is None:
            return

        all_verts = self.model.particle_q.numpy()

        for mesh_idx, info in enumerate(self._mesh_info):
            name = info["name"]
            v_start = info["vertex_start"]
            v_count = info["vertex_count"]
            faces = info["faces"]  # Local indices (0-based)

            # Extract vertices for this mesh
            verts = all_verts[v_start : v_start + v_count]

            # Save as PLY with numeric index and name
            out_file = os.path.join(self.output_path, f"initial_mesh_{mesh_idx:03d}_{name}.ply")

            header = [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(verts)}",
                "property float x",
                "property float y",
                "property float z",
                f"element face {len(faces)}",
                "property list uchar int vertex_indices",
                "end_header",
            ]

            with open(out_file, "w") as ply_file:
                ply_file.write("\n".join(header) + "\n")
                for vertex in verts:
                    ply_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
                for face in faces:
                    ply_file.write(f"{len(face)} {' '.join(map(str, face))}\n")

            print(f"Initial mesh {mesh_idx} saved: {out_file}")


# =============================================================================
# Main
# =============================================================================


def save_config(config: dict, output_path: str):
    """Save the run configuration to a JSON file."""
    # fmt: off
    import json  # noqa: PLC0415
    from datetime import datetime  # noqa: PLC0415
    # fmt: on

    config_file = os.path.join(output_path, "run_config.json")

    # Convert config to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable_config = make_serializable(config)

    # Add metadata
    serializable_config["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "output_path": output_path,
    }

    with open(config_file, "w") as f:
        json.dump(serializable_config, f, indent=2)

    print(f"Config saved to: {config_file}")


if __name__ == "__main__":
    from datetime import datetime

    # wp.clear_kernel_cache()

    # Create output folder with date/time and layer count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_layers = example_config["cloths"]["main_cloth"]["num_layers"]
    subfolder = f"{timestamp}"
    example_config["output_path"] = os.path.join(example_config["output_path"], f"{num_layers}_layers", subfolder)

    # Create output directory
    os.makedirs(example_config["output_path"], exist_ok=True)

    # Save the run config
    save_config(example_config, example_config["output_path"])

    # Create and run the simulation
    sim = ClothDropSimulator(example_config)
    sim.finalize()
    sim.simulate()
