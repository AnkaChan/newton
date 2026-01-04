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
    "collision_buffer_resize_frames": -1,  # Check and resize collision buffers every N frames (<0 to disable)
    "collision_buffer_growth_ratio": 1.5,  # Growth ratio for collision buffer resize (1.5 = 50% headroom)
    "include_bending": False,  # Include bending edges in coloring
    # Global physics settings (newton.Model parameters)
    "up_axis": "y",
    "gravity": -980.0,
    "soft_contact_ke": 2e4,  # Contact stiffness
    "soft_contact_kd": 1e-5,  # Contact damping
    "soft_contact_mu": 0.2,  # Friction coefficient
    # Output settings
    "output_path": None,  # Directory to save output files
    "output_ext": "ply",  # "ply", "usd", or "npy" (npy saves only positions, initial meshes saved as ply)
    "write_output": False,
    "write_video": False,
    "recovery_state_save_steps": -1,  # Save recovery state every N frames (<0 to disable)
    # Visualization
    "do_rendering": True,
    "show_ground_plane": False,
    "is_initially_paused": True,
    # Ground plane
    "has_ground": False,  # Add ground collision plane to simulation
    "ground_height": 0.0,  # Height of the ground plane
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
        self.collision_buffer_resize_frames = cfg("collision_buffer_resize_frames")
        self.collision_buffer_growth_ratio = cfg("collision_buffer_growth_ratio")

        # Physics
        self.up_axis = cfg("up_axis")
        self.gravity = cfg("gravity")

        # Output settings
        self.output_path = cfg("output_path")
        self.output_ext = cfg("output_ext")
        self.write_output = cfg("write_output")
        self.write_video = cfg("write_video")
        self.recovery_state_save_steps = cfg("recovery_state_save_steps")

        # Visualization
        self.do_rendering = cfg("do_rendering")
        self.show_ground_plane = cfg("show_ground_plane")

        # Ground plane
        self.has_ground = cfg("has_ground")
        self.ground_height = cfg("ground_height")

        # Runtime state
        self.sim_time = 0.0
        self.profiler = {}
        self.frame_times = []
        self.graph = None

        # Polyscope mesh registry: name -> {"mesh": ps_mesh, "vertex_indices": slice or array, "faces": array}
        self.ps_meshes: dict = {}

        # Track whether initial meshes have been saved (for npy output mode)
        self._initial_meshes_saved = False

        # Initialize polyscope
        self.init_polyscope()

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

        # Add ground plane if enabled
        if self.has_ground:
            self.builder.add_ground_plane(self.ground_height)

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

        # Allow subclasses to do additional setup (before polyscope setup)
        self.custom_finalize()

        # Set up polyscope visualization
        self.setup_polyscope_meshes()

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

        # Run substeps
        for _ in range(self.num_substeps):
            # Run collision detection (check shape_count, not body_count, since ground plane has no body)
            if self.model.shape_count:
                self.contacts = self.model.collide(self.state_0)

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
        screenshot_dir = None
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
            # Create screenshot subfolder
            screenshot_dir = join(self.output_path, "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)

        try:
            for frame_id in tqdm.tqdm(range(self.sim_num_frames)):
                self.step()

                # Rebuild BVH periodically (outside of CUDA graph)
                if self.rebuild_frames and frame_id % self.rebuild_frames == 0:
                    self.rebuild_bvh()

                # Resize collision buffers periodically if enabled
                if self.collision_buffer_resize_frames > 0 and frame_id % self.collision_buffer_resize_frames == 0:
                    self.resize_collision_buffers()

                if self.write_output:
                    self.save_output(frame_id)

                # Save recovery state periodically
                if (
                    self.recovery_state_save_steps > 0
                    and self.output_path
                    and frame_id % self.recovery_state_save_steps == 0
                ):
                    self.save_recovery_state(frame_id)

                if self.do_rendering:
                    self.render()
                    if vid_out is not None:
                        pixels = ps.screenshot_to_buffer(False)
                        vid_out.write(pixels[:, :, [2, 1, 0]])
                        # Save screenshot as PNG
                        screenshot_file = join(screenshot_dir, f"frame_{frame_id:06d}.png")
                        cv2.imwrite(screenshot_file, pixels[:, :, [2, 1, 0]])
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user. Saving video...")
        finally:
            if vid_out is not None:
                vid_out.release()
                print(f"Video saved to: {join(self.output_path, 'video.mp4')}")
                print(f"Screenshots saved to: {screenshot_dir}")

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
        elif self.output_ext == "npy":
            # Save initial meshes once (for topology reference)
            if not self._initial_meshes_saved:
                self.save_initial_meshes()
                self._initial_meshes_saved = True
            # Save only positions as npy
            out_file = join(self.output_path, f"frame_{frame_id:06d}.npy")
            self.save_npy(self.state_0, out_file)

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

    def resize_collision_buffers(self, shrink_to_fit: bool = False) -> bool:
        """
        Resize collision buffers based on actual collision counts.

        If buffers are resized and CUDA graph is enabled, the graph will be recaptured
        since it references the old buffer memory addresses.

        Args:
            shrink_to_fit: If True, also shrink oversized buffers. Default is False.

        Returns:
            True if any buffer was resized, False otherwise.
        """
        if not self.handle_self_contact:
            return False

        resized = self.solver.resize_collision_buffers(
            shrink_to_fit=shrink_to_fit,
            growth_ratio=self.collision_buffer_growth_ratio,
        )

        if resized:
            # Recapture CUDA graph since buffer memory addresses changed
            if self.use_cuda_graph and self.graph is not None:
                print("Collision buffers resized, recapturing CUDA graph...")
                self.graph = self.capture_graph()

        return resized

    # =========================================================================
    # Polyscope Visualization Methods (Override these for custom rendering)
    # =========================================================================

    def init_polyscope(self):
        """
        Initialize polyscope. Override to customize polyscope settings.
        """
        ps.init()
        ps.set_up_dir(self.up_axis + "_up")

        if self.show_ground_plane or self.has_ground:
            # Set ground plane to match simulation
            ps.set_ground_plane_mode("tile_reflection")
            ps.set_ground_plane_height(self.ground_height)
        else:
            ps.set_ground_plane_mode("none")

    def setup_polyscope_meshes(self):
        """
        Set up polyscope meshes for visualization. Override to register custom meshes.

        By default, registers a single mesh with all particles and faces.
        Subclasses can override to register separate meshes for colliders and cloths.
        """
        if not self.do_rendering:
            return

        verts = self.model.particle_q.numpy()
        self.register_ps_mesh(
            name="Sim",
            vertices=verts,
            faces=self.faces,
            vertex_indices=None,  # Use all vertices
        )

    def register_ps_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_indices: np.ndarray | slice | None = None,
        color: tuple | None = None,
        edge_width: float = 0.0,
        smooth_shade: bool = True,
        enabled: bool = True,
    ):
        """
        Register a polyscope surface mesh.

        Args:
            name: Unique name for the mesh.
            vertices: Vertex positions (Nx3 array).
            faces: Face indices (Mx3 array), indices into vertices array.
            vertex_indices: Indices into the global particle array for updates.
                           If None, uses all particles. Can be a slice or array.
            color: Optional RGB color tuple (0-1 range).
            edge_width: Edge line width (0 to disable).
            smooth_shade: Enable smooth shading.
            enabled: Whether the mesh is initially visible.
        """
        ps_mesh = ps.register_surface_mesh(name, vertices, faces, enabled=enabled)

        if smooth_shade:
            ps_mesh.set_smooth_shade(True)
        if color is not None:
            ps_mesh.set_color(color)
        if edge_width > 0:
            ps_mesh.set_edge_width(edge_width)

        self.ps_meshes[name] = {
            "mesh": ps_mesh,
            "vertex_indices": vertex_indices,
            "faces": faces,
        }

        return ps_mesh

    def configure_ps_mesh(self, name: str, **kwargs):
        """
        Configure an existing polyscope mesh.

        Args:
            name: Name of the registered mesh.
            **kwargs: Properties to set (color, edge_width, smooth_shade, enabled, transparency, etc.)
        """
        if name not in self.ps_meshes:
            return

        ps_mesh = self.ps_meshes[name]["mesh"]

        if "color" in kwargs:
            ps_mesh.set_color(kwargs["color"])
        if "edge_width" in kwargs:
            ps_mesh.set_edge_width(kwargs["edge_width"])
        if "smooth_shade" in kwargs:
            ps_mesh.set_smooth_shade(kwargs["smooth_shade"])
        if "enabled" in kwargs:
            ps_mesh.set_enabled(kwargs["enabled"])
        if "transparency" in kwargs:
            ps_mesh.set_transparency(kwargs["transparency"])
        if "material" in kwargs:
            ps_mesh.set_material(kwargs["material"])

    def update_ps_meshes(self):
        """
        Update all registered polyscope meshes with current particle positions.
        Override to customize mesh updates.
        """
        all_verts = self.state_0.particle_q.numpy()

        for _name, mesh_info in self.ps_meshes.items():
            ps_mesh = mesh_info["mesh"]
            vertex_indices = mesh_info["vertex_indices"]

            if vertex_indices is None:
                # Use all vertices
                verts = all_verts
            elif isinstance(vertex_indices, slice):
                verts = all_verts[vertex_indices]
            else:
                verts = all_verts[vertex_indices]

            ps_mesh.update_vertex_positions(verts)

    def render(self):
        """Update the polyscope visualization."""
        self.update_ps_meshes()
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

    def save_npy(self, state, filename):
        """
        Save only particle positions to a NPY file.

        This is more efficient than PLY when only positions change and topology is fixed.

        Args:
            state: The simulation state containing particle positions.
            filename: Output file path.
        """
        vertices = state.particle_q.numpy()
        np.save(filename, vertices)

    def save_initial_meshes(self):
        """
        Save initial mesh topology as PLY files.

        Override this method in subclasses to save colliders and cloths separately.
        By default, saves a single combined mesh with all particles and faces.

        This is called once when using npy output format to save the initial topology.
        """
        if self.output_path is None:
            return

        # Save combined mesh as initial reference
        vertices = self.model.particle_q.numpy()
        out_file = join(self.output_path, "initial_mesh.ply")

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

        with open(out_file, "w") as ply_file:
            ply_file.write("\n".join(header) + "\n")
            for vertex in vertices:
                ply_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            for face in self.faces:
                ply_file.write(f"{len(face)} {' '.join(map(str, face))}\n")

        print(f"Initial mesh saved to: {out_file}")

    def save_recovery_state(self, frame_id: int):
        """
        Save a recovery state to an npz file.

        The recovery state contains:
        - particle_q: Particle positions
        - particle_qd: Particle velocities
        - frame_id: Current frame number
        - sim_time: Current simulation time

        Args:
            frame_id: Current frame number.
        """
        if self.output_path is None:
            return

        recovery_file = join(self.output_path, f"recovery_state_{frame_id:06d}.npz")

        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()

        np.savez(
            recovery_file,
            particle_q=particle_q,
            particle_qd=particle_qd,
            frame_id=frame_id,
            sim_time=self.sim_time,
        )

    def load_recovery_state(self, filepath: str) -> int:
        """
        Load a recovery state from an npz file and restore the simulation.

        Args:
            filepath: Path to the recovery state npz file.

        Returns:
            The frame_id from the recovery state.
        """
        data = np.load(filepath)

        particle_q = data["particle_q"]
        particle_qd = data["particle_qd"]
        frame_id = int(data["frame_id"])
        sim_time = float(data["sim_time"])

        # Restore state
        self.state_0.particle_q.assign(particle_q)
        self.state_0.particle_qd.assign(particle_qd)
        self.state_1.particle_q.assign(particle_q)
        self.state_1.particle_qd.assign(particle_qd)
        self.sim_time = sim_time

        return frame_id

    def get_latest_recovery_state(self) -> str | None:
        """
        Find the latest recovery state file in the output directory.

        Returns:
            Path to the latest recovery state file, or None if not found.
        """
        if self.output_path is None:
            return None

        import glob  # noqa: PLC0415

        pattern = join(self.output_path, "recovery_state_*.npz")
        files = glob.glob(pattern)

        if not files:
            return None

        # Sort by frame number (extracted from filename)
        files.sort(key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]))
        return files[-1]

    def simulate_from_recovery(self, recovery_filepath: str | None = None):
        """
        Resume simulation from a recovery state file.

        Args:
            recovery_filepath: Path to the recovery state npz file.
                              If None, uses the latest recovery state in output_path.
        """
        # Find recovery file if not specified
        if recovery_filepath is None:
            recovery_filepath = self.get_latest_recovery_state()
            if recovery_filepath is None:
                print("No recovery state found. Starting from beginning.")
                self.simulate()
                return

        # Load the recovery state
        start_frame = self.load_recovery_state(recovery_filepath)
        print(f"Resuming simulation from frame {start_frame}, time {self.sim_time:.4f}")

        vid_out = None
        screenshot_dir = None
        if self.write_video and self.output_path:
            out_video_file = join(self.output_path, "video_resumed.mp4")
            pixels = ps.screenshot_to_buffer(False)
            vid_out = cv2.VideoWriter(
                out_video_file,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (pixels.shape[1], pixels.shape[0]),
                isColor=True,
            )
            # Create screenshot subfolder (reuse same folder)
            screenshot_dir = join(self.output_path, "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)

        try:
            for frame_id in tqdm.tqdm(range(start_frame + 1, self.sim_num_frames)):
                self.step()

                # Rebuild BVH periodically (outside of CUDA graph)
                if not self.use_cuda_graph and self.rebuild_frames and frame_id % self.rebuild_frames == 0:
                    self.rebuild_bvh()

                # Resize collision buffers periodically if enabled
                if self.collision_buffer_resize_frames > 0 and frame_id % self.collision_buffer_resize_frames == 0:
                    self.resize_collision_buffers()

                if self.write_output:
                    self.save_output(frame_id)

                # Save recovery state periodically
                if (
                    self.recovery_state_save_steps > 0
                    and self.output_path
                    and frame_id % self.recovery_state_save_steps == 0
                ):
                    self.save_recovery_state(frame_id)

                if self.do_rendering:
                    self.render()
                    if vid_out is not None:
                        pixels = ps.screenshot_to_buffer(False)
                        vid_out.write(pixels[:, :, [2, 1, 0]])
                        # Save screenshot as PNG
                        screenshot_file = join(screenshot_dir, f"frame_{frame_id:06d}.png")
                        cv2.imwrite(screenshot_file, pixels[:, :, [2, 1, 0]])
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user. Saving video...")
        finally:
            if vid_out is not None:
                vid_out.release()
                print(f"Video saved to: {join(self.output_path, 'video_resumed.mp4')}")
                print(f"Screenshots saved to: {screenshot_dir}")

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
