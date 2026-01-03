# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Reusable Cloth Simulator
#
# A reusable simulator class for FEM cloth simulation using Newton's
# SolverVBD. Supports collision with static meshes and self-contact.
#
###########################################################################

import os
from os.path import join

import cv2
import numpy as np
import polyscope as ps
import tqdm
import warp as wp

import newton
from newton import ParticleFlags

default_config = {
    "name": "default_cloth_sim",
    # Simulation timing
    "fps": 60,
    "sim_substeps": 10,
    "sim_num_frames": 1000,
    "iterations": 10,
    "bvh_rebuild_frames": 1,
    # Solver settings (newton.solvers.SolverVBD parameters)
    "use_cuda_graph": False,
    "handle_self_contact": True,
    "use_tile_solve": True,
    "self_contact_radius": 0.2,
    "self_contact_margin": 0.3,
    "topological_contact_filter_threshold": 1,
    "rest_shape_contact_exclusion_radius": 0.0,
    "vertex_collision_buffer_pre_alloc": 64,
    "edge_collision_buffer_pre_alloc": 128,
    "include_bending": False,  # Include bending edges in coloring
    # Global physics settings (newton.Model parameters)
    "up_axis": "y",
    "gravity": -980.0,
    "soft_contact_ke": 2e4,  # Contact stiffness
    "soft_contact_kd": 1e-5,  # Contact damping
    "soft_contact_mu": 0.2,  # Friction coefficient
    # Output settings
    "output_path": None,  # Directory to save output files
    "output_ext": "ply",  # "ply" or "usd"
    "write_output": False,
    "write_video": False,
    # Visualization
    "do_rendering": True,
    "show_ground_plane": False,
    "is_initially_paused": True,
}


def get_config_value(config, key):
    """
    Get a config value: use config[key] if it exists, otherwise fall back to default_config[key].
    """
    if config is not None and key in config:
        return config[key]
    return default_config[key]


class Simulator:
    """
    A reusable cloth simulator using Newton's SolverVBD.

    This class provides a flexible framework for cloth simulation with support for:
    - Collision with static meshes
    - Self-contact handling
    - Multiple output formats (PLY, USD)
    - Polyscope visualization
    - CUDA graph acceleration

    Subclass this and override `custom_init()` and `custom_finalize()` to add
    cloth meshes and colliders to the simulation.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize the simulator with configuration parameters.

        Args:
            config: Configuration dictionary. Missing keys will use defaults from `default_config`.
        """
        self.config = config

        # Helper to read from config or fall back to default
        def cfg(key):
            return get_config_value(config, key)

        # Simulation timing
        self.fps = cfg("fps")
        self.frame_dt = 1.0 / self.fps
        self.num_substeps = cfg("sim_substeps")
        self.iterations = cfg("iterations")
        self.sim_num_frames = cfg("sim_num_frames")
        self.rebuild_frames = cfg("bvh_rebuild_frames")

        # Solver settings
        self.use_cuda_graph = cfg("use_cuda_graph")
        self.handle_self_contact = cfg("handle_self_contact")

        # Physics
        self.up_axis = cfg("up_axis")
        self.gravity = cfg("gravity")

        # Output settings
        self.output_path = cfg("output_path")
        self.output_ext = cfg("output_ext")
        self.write_output = cfg("write_output")
        self.write_video = cfg("write_video")

        # Visualization
        self.do_rendering = cfg("do_rendering")
        self.show_ground_plane = cfg("show_ground_plane")

        # Runtime state
        self.sim_time = 0.0
        self.profiler = {}
        self.frame_times = []
        self.graph = None

        # Initialize polyscope
        ps.init()
        if self.show_ground_plane:
            ps.set_ground_plane_mode("shadow_only")
        else:
            ps.set_ground_plane_mode("none")
        ps.set_up_dir(self.up_axis + "_up")

        # Create Newton model builder
        self.builder = newton.ModelBuilder(up_axis=self.up_axis, gravity=self.gravity)

        # Allow subclasses to add meshes
        self.custom_init()

    def custom_init(self):
        """Override this method to add cloth meshes and colliders to self.builder."""
        pass

    def finalize(self, fixed_particles=None):
        """
        Finalize the model and create the solver.

        Args:
            fixed_particles: Optional list of particle indices to fix in place.
        """

        # Helper to read from config or fall back to default
        def cfg(key):
            return get_config_value(self.config, key)

        # Color the mesh for VBD solver
        self.builder.color(include_bending=cfg("include_bending"))

        # Finalize the model
        self.model = self.builder.finalize()

        # Set contact parameters
        self.model.soft_contact_ke = cfg("soft_contact_ke")
        self.model.soft_contact_kd = cfg("soft_contact_kd")
        self.model.soft_contact_mu = cfg("soft_contact_mu")

        # Create output directory if needed
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)

        # Compute timestep
        self.dt = self.frame_dt / self.num_substeps

        # Fix specified particles
        if fixed_particles is not None:
            self.set_points_fixed(fixed_particles)

        # Create the VBD solver
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            handle_self_contact=cfg("handle_self_contact"),
            self_contact_radius=cfg("self_contact_radius"),
            self_contact_margin=cfg("self_contact_margin"),
            topological_contact_filter_threshold=cfg("topological_contact_filter_threshold"),
            rest_shape_contact_exclusion_radius=cfg("rest_shape_contact_exclusion_radius"),
            use_tile_solve=cfg("use_tile_solve"),
            vertex_collision_buffer_pre_alloc=cfg("vertex_collision_buffer_pre_alloc"),
            edge_collision_buffer_pre_alloc=cfg("edge_collision_buffer_pre_alloc"),
        )

        # Create simulation states and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Get triangle indices for visualization
        self.faces = self.model.tri_indices.numpy()

        # Set up polyscope visualization
        if self.do_rendering:
            self.verts_for_vis = self.model.particle_q.numpy().copy()
            self.ps_vis_mesh = ps.register_surface_mesh("Sim", self.verts_for_vis, self.faces)

        # Allow subclasses to do additional setup
        self.custom_finalize()

        # Capture CUDA graph if enabled
        self.use_cuda_graph = self.use_cuda_graph and wp.get_device().is_cuda
        if self.use_cuda_graph:
            self.graph = self.capture_graph()

    def custom_finalize(self):
        """Override this method for additional setup after model finalization."""
        pass

    def capture_graph(self):
        """Capture a CUDA graph of one simulation step for accelerated execution."""
        with wp.ScopedCapture() as capture:
            self.run_step()
        return capture.graph

    def run_step(self):
        """Execute one frame of simulation (all substeps)."""
        # Run collision detection
        self.contacts = self.model.collide(self.state_0)

        # Rebuild BVH for self-contact if needed
        if self.handle_self_contact:
            self.solver.rebuild_bvh(self.state_0)

        # Run substeps
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Execute one frame, using CUDA graph if available."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.run_step()
        self.sim_time += self.frame_dt

    def simulate(self):
        """Run the full simulation loop."""
        vid_out = None
        if self.write_video and self.output_path:
            out_video_file = join(self.output_path, "video.mp4")
            pixels = ps.screenshot_to_buffer(False)
            vid_out = cv2.VideoWriter(
                out_video_file,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (pixels.shape[1], pixels.shape[0]),
                isColor=True,
            )

        for frame_id in tqdm.tqdm(range(self.sim_num_frames)):
            self.step()

            # Rebuild BVH periodically (outside of CUDA graph)
            if not self.use_cuda_graph and self.rebuild_frames and frame_id % self.rebuild_frames == 0:
                self.rebuild_bvh()

            if self.write_output:
                self.save_output(frame_id)

            if self.do_rendering:
                self.render()
                if vid_out is not None:
                    pixels = ps.screenshot_to_buffer(False)
                    vid_out.write(pixels[:, :, [2, 1, 0]])

        if vid_out is not None:
            vid_out.release()

    def save_output(self, frame_id):
        """Save the current frame to a file."""
        if self.output_path is None:
            return

        if self.output_ext == "ply":
            out_file = join(self.output_path, f"frame_{frame_id:06d}.ply")
            self.save_ply(self.state_0, out_file)
        elif self.output_ext == "usd":
            out_file = join(self.output_path, f"frame_{frame_id:06d}.usd")
            self.save_usd(self.state_0, out_file)

    def set_points_fixed(self, fixed_particles):
        """
        Fix specified particles in place (zero mass, inactive).

        Args:
            fixed_particles: List of particle indices to fix.
        """
        if not fixed_particles:
            return

        # Set mass to zero and remove ACTIVE flag
        for v_id in fixed_particles:
            self.builder.particle_mass[v_id] = 0.0
            self.builder.particle_flags[v_id] &= ~ParticleFlags.ACTIVE

    def rebuild_bvh(self):
        """Rebuild the BVH for self-contact detection."""
        if self.handle_self_contact:
            self.solver.rebuild_bvh(self.state_0)

    def render(self):
        """Update the polyscope visualization."""
        self.verts_for_vis = self.state_0.particle_q.numpy()
        self.ps_vis_mesh.update_vertex_positions(self.verts_for_vis)
        ps.frame_tick()

    def save_ply(self, state, filename):
        """
        Save the current state to a PLY file.

        Args:
            state: The simulation state containing particle positions.
            filename: Output file path.
        """
        vertices = state.particle_q.numpy()
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(self.faces)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]

        with open(filename, "w") as ply_file:
            ply_file.write("\n".join(header) + "\n")

            for vertex in vertices:
                ply_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

            for face in self.faces:
                ply_file.write(f"{len(face)} {' '.join(map(str, face))}\n")

    def save_usd(self, state, filename):
        """
        Save the current state to a USD file.

        Args:
            state: The simulation state containing particle positions.
            filename: Output file path.
        """
        # USD export requires pxr library
        try:
            # fmt: off
            from pxr import Usd, UsdGeom  # noqa: PLC0415
            # fmt: on

            stage = Usd.Stage.CreateNew(filename)
            mesh_prim = UsdGeom.Mesh.Define(stage, "/ClothMesh")

            vertices = state.particle_q.numpy()
            mesh_prim.GetPointsAttr().Set(vertices.tolist())
            mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(self.faces))
            mesh_prim.GetFaceVertexIndicesAttr().Set(self.faces.flatten().tolist())

            stage.Save()
        except ImportError:
            print("Warning: pxr (USD) library not available. Falling back to PLY.")
            self.save_ply(state, filename.replace(".usd", ".ply"))

    def load_state(self, filepath):
        """
        Load particle positions from a PLY file.

        Args:
            filepath: Path to the PLY file.
        """
        vertices = []
        with open(filepath) as f:
            in_header = True
            vertex_count = 0
            for raw_line in f:
                stripped = raw_line.strip()
                if in_header:
                    if stripped.startswith("element vertex"):
                        vertex_count = int(stripped.split()[-1])
                    elif stripped == "end_header":
                        in_header = False
                else:
                    if len(vertices) < vertex_count:
                        parts = stripped.split()
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

        if vertices:
            vertices_np = np.array(vertices, dtype=np.float32)
            self.state_0.particle_q.assign(vertices_np)
            self.state_1.particle_q.assign(vertices_np)


def read_obj(filepath):
    """
    Read vertices and faces from an OBJ file.

    Args:
        filepath: Path to the OBJ file.

    Returns:
        Tuple of (vertices, faces) where vertices is a list of (x, y, z) tuples
        and faces is a list of triangle index tuples.
    """
    vertices = []
    faces = []

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "v":
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                # Handle various face formats: v, v/vt, v/vt/vn, v//vn
                face_indices = []
                for p in parts[1:]:
                    idx = p.split("/")[0]
                    face_indices.append(int(idx) - 1)  # OBJ is 1-indexed
                faces.append(tuple(face_indices))

    return vertices, faces
