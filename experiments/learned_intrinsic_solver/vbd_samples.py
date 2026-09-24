# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record seeded cantilever initializations with Newton's unmodified VBD solver.

Experimental, standalone data-generation driver. The independent cell fields
from ``augment_grid`` are projected onto shared corners only at initialization.
All elastic rest data is constructed from the canonical, undeformed cuboid.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .data import VoxelGridData, augment_grid, generate_cuboid

CELL_COUNTS = (10, 10, 40)
CELL_SIZE = 0.025
DEFORMATION_AMPLITUDE = 0.15
VELOCITY_AMPLITUDE = 0.75
DURATION = 10.0
FPS = 30
SUBSTEPS = 10
ITERATIONS = 20
YOUNG_MODULUS = 5.0e5
POISSON_RATIO = 0.3
DENSITY = 1000.0
DAMPING = 100.0
ROTATION = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
TRANSLATION = np.array([0.0, -0.125, 1.125])
CAMERA_POSITION = (1.3, -1.65, 1.5)
CAMERA_TARGET = (0.45, 0.0, 0.65)
CAMERA_FOV = 43.0
DEFAULT_OUTPUT = Path(__file__).parent / "generated" / "vbd_10x10x40"


class CornerProjector:
    """Fit all 12 cell edges to target deformation, with the z=0 end fixed.

    Minimize sum over cells/edges of ||u_j-u_i-h(F-I)e_axis||^2.
    Shared edges retain their multiplicity; all edge rows have equal weight.
    Unlike a single center-gradient sample, this objective has no hourglass
    nullspace. Eliminating the clamped degrees of freedom fixes translation.
    """

    def __init__(self, rest: VoxelGridData):
        from scipy import sparse
        from scipy.sparse.linalg import factorized

        self.rest = rest
        local = np.indices((2, 2, 2)).reshape(3, -1).T
        starts, ends, axes = [], [], []
        for axis in range(3):
            for start in np.flatnonzero(local[:, axis] == 0):
                end_position = local[start].copy()
                end_position[axis] = 1
                end = np.flatnonzero(np.all(local == end_position, axis=1))[0]
                starts.append(start)
                ends.append(end)
                axes.append(axis)
        self.axes = np.asarray(axes)
        self.edge_start = rest.cell_corner_indices[:, starts].reshape(-1)
        self.edge_end = rest.cell_corner_indices[:, ends].reshape(-1)
        self.fixed = np.isclose(rest.corner_rest_positions[:, 2], 0.0)
        self.free = ~self.fixed
        row = np.repeat(np.arange(len(self.edge_start)), 2)
        col = np.column_stack((self.edge_start, self.edge_end)).reshape(-1)
        value = np.tile((-1.0, 1.0), len(self.edge_start))
        self.incidence = sparse.csr_matrix((value, (row, col)), shape=(len(self.edge_start), len(self.free)))
        self.free_incidence = self.incidence[:, self.free]
        self.solve = factorized((self.free_incidence.T @ self.free_incidence).tocsc())

    def project(self, augmented: VoxelGridData) -> tuple[np.ndarray, np.ndarray, float]:
        """Return material-space positions, velocities, and edge residual [m]."""
        targets = (
            self.rest.cell_size * (augmented.cell_deformation - np.eye(3))[:, :, self.axes].transpose(0, 2, 1)
        ).reshape(-1, 3)
        displacement = np.zeros_like(self.rest.corner_rest_positions)
        displacement[self.free] = self.solve(self.free_incidence.T @ targets)
        residual = self.incidence @ displacement - targets

        velocity = np.zeros_like(displacement)
        multiplicity = np.zeros(len(displacement))
        for corner in range(8):
            indices = self.rest.cell_corner_indices[:, corner]
            np.add.at(velocity, indices, augmented.cell_velocity)
            np.add.at(multiplicity, indices, 1)
        velocity /= multiplicity[:, None]
        velocity[self.fixed] = 0.0
        return self.rest.corner_rest_positions + displacement, velocity, float(np.sqrt(np.mean(residual**2)))


def _ordering_map(counts: tuple[int, int, int]) -> np.ndarray:
    """Map helper (z-fast) indices into Newton's (x-fast) particle ordering."""
    coordinates = np.indices(tuple(count + 1 for count in counts)).reshape(3, -1).T
    return coordinates[:, 0] + (counts[0] + 1) * (coordinates[:, 1] + (counts[1] + 1) * coordinates[:, 2])


def _tet_determinants(positions: np.ndarray, tets: np.ndarray) -> np.ndarray:
    vertices = positions[tets]
    return np.linalg.det(np.stack([vertices[:, axis] - vertices[:, 0] for axis in (1, 2, 3)], axis=-1))


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "environment": {
            "newton_version": version("newton"),
            "warp_version": version("warp-lang"),
            "numpy_version": np.__version__,
            "scipy_version": version("scipy"),
            "pyglet_version": version("pyglet"),
            "newton_git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
            ).strip(),
        },
        "grid": {
            "cell_counts": list(CELL_COUNTS),
            "cell_size": CELL_SIZE,
            "cell_count": 4000,
            "particle_count": 4961,
            "tet_count": 20000,
            "canonical_extents_m": [0.25, 0.25, 1.0],
        },
        "simulation": {
            "duration_seconds": DURATION,
            "frame_rate": FPS,
            "solver": "Newton SolverVBD",
            "renderer": "Newton ViewerGL",
            "substeps_per_frame": SUBSTEPS,
            "timestep_seconds": 1.0 / (FPS * SUBSTEPS),
            "iterations_per_substep": ITERATIONS,
            "particle_enable_tile_solve": True,
            "gravity_m_per_s2": [0.0, 0.0, -9.81],
            "self_contact": False,
            "ground_contact": False,
            "material": {
                "young_modulus_pa": YOUNG_MODULUS,
                "poisson_ratio": POISSON_RATIO,
                "density_kg_per_m3": DENSITY,
                "damping_pa_s": DAMPING,
                "lame_mu_pa": YOUNG_MODULUS / (2 * (1 + POISSON_RATIO)),
                "lame_lambda_pa": YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO)),
            },
            "clamp": {
                "material_face": "z=0",
                "particle_count": 121,
                "position": "canonical rest positions",
                "inverse_mass": 0.0,
                "velocity": [0.0, 0.0, 0.0],
            },
            "mass_discretization": "Tetrahedral rest-volume lumping, rho*V/4 to each vertex; clamp mass set to zero",
            "world_from_material_rotation": ROTATION.tolist(),
            "world_translation_m": TRANSLATION.tolist(),
            "camera": {
                "position": CAMERA_POSITION,
                "target": CAMERA_TARGET,
                "fov_degrees": CAMERA_FOV,
                "width": 1280,
                "height": 720,
                "fixed_for_all_frames_and_samples": True,
            },
            "video_frames": 300,
            "video_frame_times": "1/30, 2/30, ..., 10 seconds; initial state stored as poster",
            "physical_inertia": "Unmodified Newton VBD implicit-Euler inertial term",
        },
        "augmentation": {
            "generator": "experiments.learned_intrinsic_solver.data.augment_grid",
            "deformation_amplitude": DEFORMATION_AMPLITUDE,
            "velocity_amplitude_m_per_s": VELOCITY_AMPLITUDE,
            "projection": "Equal-weight least squares of all 12 directed cell edges; fixed material z=0 end",
            "velocity_projection": "Average incident cell velocities at each shared corner, then zero the clamp",
            "rest_shape": "Canonical cuboid; augmented positions are assigned only to initial states",
            "orientation_policy": "Halve initial displacement only if any initial tet volume ratio <= 0.15; record scale",
            "numpy_version": np.__version__,
            "reproducibility": "Explicit PCG64 seed for every sample; repeat-seed cell fields and CPU projection verified exact",
            "gpu_trajectory_bitwise_determinism": "Not promised",
        },
        "samples": [],
    }


def _build_simulation(seed: int):
    import warp as wp  # noqa: PLC0415 - keep projection tests independent of GPU initialization

    import newton  # noqa: PLC0415
    import newton.solvers  # noqa: PLC0415

    rest = generate_cuboid(CELL_COUNTS, cell_size=CELL_SIZE)
    projector = CornerProjector(rest)
    kwargs = {"deformation_amplitude": DEFORMATION_AMPLITUDE, "velocity_amplitude": VELOCITY_AMPLITUDE, "seed": seed}
    sample = augment_grid(rest, **kwargs)
    initial_material, velocity_material, residual = projector.project(sample)
    repeated = augment_grid(rest, **kwargs)
    repeated_positions, repeated_velocities, _ = projector.project(repeated)
    reproducible = all(
        (
            np.array_equal(sample.cell_deformation, repeated.cell_deformation),
            np.array_equal(sample.cell_velocity, repeated.cell_velocity),
            np.array_equal(initial_material, repeated_positions),
            np.array_equal(velocity_material, repeated_velocities),
        )
    )
    if not reproducible:
        raise RuntimeError("Repeated seed did not reproduce initialization")

    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(*TRANSLATION),
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2),
        vel=wp.vec3(0.0),
        dim_x=CELL_COUNTS[0],
        dim_y=CELL_COUNTS[1],
        dim_z=CELL_COUNTS[2],
        cell_x=CELL_SIZE,
        cell_y=CELL_SIZE,
        cell_z=CELL_SIZE,
        density=DENSITY,
        k_mu=YOUNG_MODULUS / (2 * (1 + POISSON_RATIO)),
        k_lambda=YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO)),
        k_damp=DAMPING,
        add_surface_mesh_edges=False,
        color=wp.vec3(0.13, 0.56, 0.78),
    )
    mapping = _ordering_map(CELL_COUNTS)
    fixed = np.zeros(len(mapping), dtype=bool)
    fixed[mapping] = projector.fixed
    canonical = np.asarray(builder.particle_q, dtype=np.float64)
    tets = np.asarray(builder.tet_indices, dtype=np.int64)
    rest_determinants = _tet_determinants(canonical, tets)
    if not np.all(rest_determinants > 0):
        raise RuntimeError("Canonical tetrahedra have invalid orientations")
    # Replace the grid helper's uniform vertex masses by physical volume lumping.
    mass = np.zeros(len(mapping))
    for corner in range(4):
        np.add.at(mass, tets[:, corner], DENSITY * rest_determinants / 24.0)
    total_rest_mass = float(mass.sum())
    mass[fixed] = 0.0
    builder.particle_mass = mass.tolist()
    builder.color()
    model = builder.finalize()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=ITERATIONS,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=True,
    )
    states = [model.state(), model.state()]
    control = model.control()
    displacement = initial_material - rest.corner_rest_positions
    initial_world = np.empty_like(canonical)
    scale = 1.0
    while True:
        initial_world[mapping] = (rest.corner_rest_positions + scale * displacement) @ ROTATION.T + TRANSLATION
        initial_world[fixed] = canonical[fixed]
        ratios = _tet_determinants(initial_world, tets) / rest_determinants
        if np.all(ratios > 0.15):
            break
        scale *= 0.5
        if scale < 1e-6:
            raise RuntimeError("Could not obtain positively oriented initial tetrahedra")
    initial_velocity = np.empty_like(canonical)
    initial_velocity[mapping] = velocity_material @ ROTATION.T
    initial_velocity[fixed] = 0.0
    for state in states:
        state.particle_q.assign(initial_world.astype(np.float32))
        state.particle_qd.assign(initial_velocity.astype(np.float32))
    # Store exactly the float32 values sent to the solver, including the clamp.
    initial_world = states[0].particle_q.numpy()
    initial_velocity = states[0].particle_qd.numpy()
    initial_ratio = _tet_determinants(initial_world.astype(np.float64), tets) / rest_determinants
    archive = {
        "seed": seed,
        "deformation_amplitude": DEFORMATION_AMPLITUDE,
        "velocity_amplitude": VELOCITY_AMPLITUDE,
        "cell_counts": CELL_COUNTS,
        "cell_size": CELL_SIZE,
        "cell_deformation": sample.cell_deformation,
        "cell_velocity": sample.cell_velocity,
        "corner_rest_positions_material": rest.corner_rest_positions,
        "particle_rest_positions_world": canonical,
        "particle_initial_positions_world": initial_world,
        "particle_initial_velocities_world": initial_velocity,
        "helper_to_newton_index": mapping,
        "cell_corner_indices_helper": rest.cell_corner_indices,
        "tet_indices_newton": tets,
        "clamped_particle_indices": np.flatnonzero(fixed),
        "deformation_displacement_scale": scale,
        "world_from_material_rotation": ROTATION,
        "world_translation": TRANSLATION,
    }
    metrics = {
        "repeat_seed_exact": reproducible,
        "initialization_sha256": hashlib.sha256(initial_world.tobytes() + initial_velocity.tobytes()).hexdigest(),
        "initial_tet_min_volume_ratio": float(initial_ratio.min()),
        "initial_tet_max_volume_ratio": float(initial_ratio.max()),
        "initial_edge_projection_rms_m": residual,
        "deformation_displacement_scale": scale,
        "displacement_scale_reason": "No scaling needed"
        if scale == 1
        else "Avoid inverted/near-degenerate initial tets",
        "total_canonical_mass_kg": total_rest_mass,
        "initial_displacement_rms_m": float(np.sqrt(np.mean((initial_world - canonical) ** 2))),
        "initial_speed_max_m_per_s": float(np.linalg.norm(initial_velocity, axis=1).max()),
    }
    return model, solver, states, control, archive, metrics


def _record_one(seed: int, output: Path) -> dict:
    import warp as wp  # noqa: PLC0415 - keep projection tests independent of GPU initialization
    from PIL import Image

    capture_tools = Path(os.environ.get("AI_LOGS", "/home/horde/Code/AI-Docs/AI-Logs")) / "Newton" / "tools"
    sys.path.insert(0, str(capture_tools))
    from newton_capture import Capture  # noqa: PLC0415 - external capture helper path resolved above
    from newton_capture._video import VideoWriter  # noqa: PLC0415

    start = time.perf_counter()
    for folder in ("initial", "final", "videos", "images", "metrics"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    name = f"sample_{seed:02d}"
    model, solver, states, control, archive, metrics = _build_simulation(seed)
    canonical = archive["particle_rest_positions_world"]
    fixed = archive["clamped_particle_indices"]
    tets = archive["tet_indices_newton"]
    rest_determinants = _tet_determinants(canonical, tets)
    np.savez_compressed(output / "initial" / f"{name}.npz", **archive)

    def simulate():
        for _ in range(SUBSTEPS):
            states[0].clear_forces()
            solver.step(states[0], states[1], control, None, 1.0 / (FPS * SUBSTEPS))
            states[0], states[1] = states[1], states[0]

    assert SUBSTEPS % 2 == 0, "Captured graph must leave double-buffer identities unchanged"
    with wp.ScopedCapture(device=model.device) as capture:
        simulate()
    graph = capture.graph
    trajectory_min = np.full(3, np.inf)
    trajectory_max = np.full(3, -np.inf)
    clamp_drift = 0.0
    speed_max = 0.0
    min_volume_ratio = np.inf
    traces = []
    frame_std_min = np.inf
    free_end = np.flatnonzero(canonical[:, 0] > 0.999)
    triangles = model.tri_indices.numpy()
    mesh_edges = np.unique(
        np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0
    )
    with Capture(
        out_dir=str(output),
        width=1280,
        height=720,
        camera_pos=CAMERA_POSITION,
        camera_target=CAMERA_TARGET,
        camera_fov=CAMERA_FOV,
    ) as cap:
        viewer = cap._get_viewer(model)
        viewer.show_particles = False
        viewer.show_ui = False
        viewer.renderer.draw_wireframe = False
        viewer.renderer.line_width = 0.65
        cap._apply_camera(viewer)
        clamp_points = wp.array(canonical[fixed].astype(np.float32), dtype=wp.vec3, device=model.device)
        clamp_colors = wp.full(len(fixed), wp.vec3(0.97, 0.47, 0.12), dtype=wp.vec3, device=model.device)
        edge_starts = wp.empty(len(mesh_edges), dtype=wp.vec3, device=model.device)
        edge_ends = wp.empty_like(edge_starts)
        edge_colors = wp.full(len(mesh_edges), wp.vec3(0.025, 0.13, 0.18), dtype=wp.vec3, device=model.device)

        def render(sim_time):
            cap._apply_camera(viewer)
            viewer.begin_frame(sim_time)
            viewer.log_state(states[0])
            render_positions = states[0].particle_q.numpy()
            triangle_positions = render_positions[triangles]
            face_normals = np.cross(
                triangle_positions[:, 1] - triangle_positions[:, 0], triangle_positions[:, 2] - triangle_positions[:, 0]
            )
            normals = np.zeros_like(render_positions)
            for corner in range(3):
                np.add.at(normals, triangles[:, corner], face_normals)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            # A small outward display offset keeps mesh edges above the filled surface.
            edge_positions = render_positions + 0.0006 * normals
            edge_starts.assign(edge_positions[mesh_edges[:, 0]])
            edge_ends.assign(edge_positions[mesh_edges[:, 1]])
            viewer.log_lines("surface_edges", edge_starts, edge_ends, edge_colors)
            viewer.log_points("clamped_end", clamp_points, radii=0.006, colors=clamp_colors)
            viewer.end_frame()
            return viewer.get_frame().numpy()

        initial_frame = render(0.0)
        if initial_frame.std() < 3:
            raise RuntimeError("Initial renderer output is black or uniform")
        Image.fromarray(initial_frame).save(output / "images" / f"{name}.png")
        with VideoWriter(str(output / "videos" / f"{name}.mp4"), fps=FPS) as writer:
            for frame_index in range(round(DURATION * FPS)):
                wp.capture_launch(graph)
                positions = states[0].particle_q.numpy()
                velocities = states[0].particle_qd.numpy()
                if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                    raise RuntimeError(f"Nonfinite state at frame {frame_index + 1}")
                trajectory_min = np.minimum(trajectory_min, positions.min(axis=0))
                trajectory_max = np.maximum(trajectory_max, positions.max(axis=0))
                clamp_drift = max(
                    clamp_drift, float(np.max(np.linalg.norm(positions[fixed] - canonical[fixed], axis=1)))
                )
                speed_max = max(speed_max, float(np.linalg.norm(velocities, axis=1).max()))
                ratios = _tet_determinants(positions.astype(np.float64), tets) / rest_determinants
                min_volume_ratio = min(min_volume_ratio, float(ratios.min()))
                if clamp_drift > 1e-6 or ratios.min() <= 0 or np.abs(positions).max() > 5:
                    raise RuntimeError(
                        f"Invalid trajectory at frame {frame_index + 1}: clamp={clamp_drift}, tet={ratios.min()}"
                    )
                physical_time = (frame_index + 1) / FPS
                traces.append([physical_time, *positions[free_end].mean(axis=0), float(ratios.min())])
                frame = render(physical_time)
                frame_std_min = min(frame_std_min, float(frame.std()))
                if frame.std() < 3:
                    raise RuntimeError(f"Black/uniform render at frame {frame_index + 1}")
                writer.write_frame(frame)
                if frame_index in (29, 149, 299):
                    Image.fromarray(frame).save(output / "images" / f"{name}_t{physical_time:g}.png")
                    print(
                        f"seed={seed} time={physical_time:g}s elapsed={time.perf_counter() - start:.1f}s "
                        f"min_tet_ratio={min_volume_ratio:.4f} clamp_drift={clamp_drift:.2e}",
                        flush=True,
                    )

    np.savez_compressed(
        output / "final" / f"{name}.npz",
        particle_positions_world=positions,
        particle_velocities_world=velocities,
        trace_columns=["time_s", "tip_x", "tip_y", "tip_z", "tet_ratio_min"],
        traces=np.asarray(traces),
    )
    metrics.update(
        {
            "physical_duration_seconds": DURATION,
            "frame_count": round(DURATION * FPS),
            "simulation_substeps": round(DURATION * FPS) * SUBSTEPS,
            "finite_all_frames": True,
            "clamp_drift_max_m": clamp_drift,
            "trajectory_tet_min_volume_ratio": min_volume_ratio,
            "trajectory_speed_max_m_per_s": speed_max,
            "trajectory_bounds_min_m": trajectory_min.tolist(),
            "trajectory_bounds_max_m": trajectory_max.tolist(),
            "frame_pixel_std_min": frame_std_min,
            "wall_seconds": time.perf_counter() - start,
            "final_tip_center_world_m": positions[free_end].mean(axis=0).tolist(),
        }
    )
    entry = {
        "id": seed,
        "seed": seed,
        "status": "complete",
        "video": f"videos/{name}.mp4",
        "poster": f"images/{name}.png",
        "initial_state": f"initial/{name}.npz",
        "final_state": f"final/{name}.npz",
        "metrics": metrics,
    }
    _atomic_json(output / "metrics" / f"{name}.json", entry)
    print(json.dumps({"completed_seed": seed, "wall_seconds": metrics["wall_seconds"]}), flush=True)
    return entry


def _refresh_manifest(output: Path) -> None:
    manifest = _manifest()
    manifest["samples"] = sorted(
        (json.loads(path.read_text()) for path in (output / "metrics").glob("sample_*.json")),
        key=lambda sample: sample["id"],
    )
    _atomic_json(output / "manifest.json", manifest)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Run exactly one seed in this process.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed for a batch.")
    parser.add_argument("--count", type=int, default=20, help="Number of consecutive explicitly seeded cases.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", help="Skip cases with completed metrics and video files.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        _record_one(args.seed, args.output)
        _refresh_manifest(args.output)
        return
    if args.count < 1 or args.seed_start < 0:
        parser.error("count must be positive and seeds nonnegative")
    _refresh_manifest(args.output)
    # One GL context per child avoids black frames from repeated context creation.
    for seed in range(args.seed_start, args.seed_start + args.count):
        name = f"sample_{seed:02d}"
        if (
            args.resume
            and (args.output / "metrics" / f"{name}.json").exists()
            and (args.output / "videos" / f"{name}.mp4").exists()
        ):
            continue
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.learned_intrinsic_solver.vbd_samples",
                "--seed",
                str(seed),
                "--output",
                str(args.output),
            ],
            check=True,
        )
        _refresh_manifest(args.output)


if __name__ == "__main__":
    _main()
