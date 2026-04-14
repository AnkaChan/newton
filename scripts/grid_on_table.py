#!/usr/bin/env python
"""Minimal reproducible example for VBD contact instability study.

Simulates NxN cloth grids resting on a table. Adjustable parameters let you
study how grid resolution, layer count, stiffness ratios, and damping affect
oscillation amplitude.

Usage (viewer):
    uv run --extra examples python scripts/grid_on_table.py --layers 2 --grid-n 4

Usage (headless data collection):
    uv run --extra examples python scripts/grid_on_table.py --headless --frames 120
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import warp as wp


def find_settled_substep(positions, window=20, threshold_fraction=0.01):
    """Find the substep where horizontal motion has settled.

    Computes max horizontal displacement per substep over a sliding window.
    Returns the first substep where it drops below threshold_fraction of
    the peak value. Falls back to 50% if never settles.
    """
    n = positions.shape[0]
    if n < window * 2:
        return n // 2

    # Max horizontal speed per substep (across all particles)
    xy_delta = np.diff(positions[:, :, :2], axis=0)  # [n-1, N, 2]
    max_xy_speed = np.sqrt((xy_delta**2).sum(axis=-1)).max(axis=1)  # [n-1]

    # Smooth with sliding window
    kernel = np.ones(window) / window
    smoothed = np.convolve(max_xy_speed, kernel, mode="valid")

    peak = smoothed.max()
    if peak < 1e-10:
        return 0  # no horizontal motion at all

    threshold = peak * threshold_fraction
    settled_indices = np.where(smoothed < threshold)[0]
    if len(settled_indices) > 0:
        return int(settled_indices[0]) + window
    # Never fully settled — use last 25%
    return int(n * 0.75)


class Example:
    def __init__(
        self,
        viewer,
        args=None,
        grid_n: int = 1,
        grid_ny: int | None = None,
        layers: int = 1,
        contact_ke: float = 1.0e4,
        tri_ke: float = 1.0e4,
        tri_ka: float = 1.0e4,
        contact_kd: float = 1.0e-2,
        tri_kd: float = 1.5e-6,
        edge_ke: float = 5.0,
        edge_kd: float = 1.0e-2,
        iterations: int = 5,
        substeps: int = 10,
        seed: int = 42,
        enable_self_contact: bool = True,
        particle_radius: float = 0.8,
        self_contact_radius: float = 0.2,
        self_contact_margin: float = 0.2,
        density: float = 0.02,
        fold: bool = False,
        tube: bool = False,
        step_ratio: float = 1.0,
        material: str = "stvk",
    ):
        self.fold = fold
        self.tube = tube
        self.grid_n = grid_n
        self.grid_ny = grid_ny if grid_ny is not None else grid_n
        self.layers = layers
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        cell_size = 1.0  # 1 cm (cm scale)

        builder = newton.ModelBuilder(gravity=-981.0)

        # Ground plane with friction
        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.ke = contact_ke
        ground_cfg.kd = contact_kd
        ground_cfg.mu = 1.5
        builder.add_ground_plane(cfg=ground_cfg)

        rng = np.random.default_rng(seed)

        # Stack layers with perturbation (cm scale)
        fold_gap = particle_radius * 2 + 0.1 if fold and grid_n >= 2 else 0.0
        tube_diameter = 0.0
        if tube and grid_n >= 3:
            tube_radius = (grid_n * cell_size) / (2.0 * np.pi)
            tube_diameter = 2.0 * tube_radius
        layer_spacing = 0.5 + fold_gap + tube_diameter
        base_z = particle_radius + 0.1
        if tube and grid_n >= 3:
            base_z = tube_radius + particle_radius + 0.1  # center above ground

        # Compute mass from density (g/cm^2)
        cell_area = cell_size * cell_size
        mass_per_particle = density * cell_area

        self.layer_vertex_ranges = []
        for layer_i in range(layers):
            z = base_z + layer_i * layer_spacing
            # Small random perturbation (cm scale)
            dx = rng.uniform(-0.2, 0.2)
            dy = rng.uniform(-0.2, 0.2)
            angle = rng.uniform(-0.1, 0.1)  # ~6 degrees max

            start_idx = builder.particle_count

            builder.add_cloth_grid(
                pos=wp.vec3(dx, dy, z),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=grid_n,
                dim_y=self.grid_ny,
                cell_x=cell_size,
                cell_y=cell_size,
                mass=mass_per_particle,
                tri_ke=tri_ke,
                tri_ka=tri_ka,
                tri_kd=tri_kd,
                edge_ke=edge_ke,
                edge_kd=edge_kd,
                particle_radius=particle_radius,
            )

            end_idx = builder.particle_count
            self.layer_vertex_ranges.append((start_idx, end_idx))

        # Fold each layer in half along x if requested
        if fold and grid_n >= 2:
            fold_gap = particle_radius * 2 + 0.1  # gap at fold to avoid interpenetration
            for start_idx, end_idx in self.layer_vertex_ranges:
                positions = []
                for i in range(start_idx, end_idx):
                    positions.append(list(builder.particle_q[i]))
                positions = np.array(positions)
                x_mid = (positions[:, 0].min() + positions[:, 0].max()) / 2.0
                for i in range(start_idx, end_idx):
                    px, py, pz = builder.particle_q[i]
                    if px > x_mid + 0.01:  # right half folds over
                        new_x = 2.0 * x_mid - px  # reflect
                        new_z = pz + fold_gap  # place on top
                        builder.particle_q[i] = (new_x, py, new_z)

        # Roll each layer into a tube (cylinder along y-axis) if requested
        if tube and grid_n >= 3:
            for start_idx, end_idx in self.layer_vertex_ranges:
                positions = []
                for i in range(start_idx, end_idx):
                    positions.append(list(builder.particle_q[i]))
                positions = np.array(positions)
                x_min = positions[:, 0].min()
                x_max = positions[:, 0].max()
                x_span = x_max - x_min
                x_center = (x_min + x_max) / 2.0
                r = x_span / (2.0 * np.pi)
                z_base = positions[:, 2].mean()  # layer's base z
                for i in range(start_idx, end_idx):
                    px, py, pz = builder.particle_q[i]
                    # Map x position to angle around circle
                    theta = (px - x_min) / x_span * 2.0 * np.pi
                    new_x = x_center + r * np.sin(theta)
                    new_z = z_base + r * (1.0 - np.cos(theta))  # bottom at z_base
                    builder.particle_q[i] = (new_x, py, new_z)

        builder.color(include_bending=True)
        self.model = builder.finalize()

        # Contact material
        self.model.soft_contact_ke = contact_ke
        self.model.soft_contact_kd = contact_kd
        self.model.soft_contact_mu = 0.5

        # Solver
        self.solver = SolverVBD(
            self.model,
            iterations=iterations,
            step_ratio=step_ratio,
            particle_enable_self_contact=enable_self_contact and (layers > 1 or fold or tube),
            particle_self_contact_radius=self_contact_radius,
            particle_self_contact_margin=self_contact_margin,
            particle_enable_tile_solve=False,
            particle_tri_material_model=material,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Zero rest angles for bending
        self.model.edge_rest_angle.zero_()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(61.0, 6.0, 27.0), pitch=-22.4, yaw=-175.1)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 27.0
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.shading_style = "studio"
            self.viewer.renderer.draw_edges = True

        # Trajectory recording (per-substep)
        self.positions_history = []

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            # Record per-substep
            self.positions_history.append(self.state_0.particle_q.numpy().copy())

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] > -0.01,
        )

    def report(self):
        """Compute and print instability metrics."""
        if len(self.positions_history) < 4:
            print("Not enough frames to analyze.")
            return

        positions = np.array(self.positions_history)  # [substeps, N, 3]
        start = find_settled_substep(positions)
        steady = positions[start:]

        z = steady[:, :, 2]
        z_std = np.std(z, axis=0)
        z_range = np.ptp(z, axis=0)

        n_substeps = positions.shape[0]
        print(f"\n{'=' * 60}")
        print(f"Instability Report ({n_substeps} substeps, settled at substep {start})")
        print(f"Grid: {self.grid_n}x{self.grid_n}, Layers: {self.layers}, Fold: {self.fold}")
        print(f"Particles: {positions.shape[1]}")
        print(f"{'=' * 60}")
        print(f"Z-position std  — max: {z_std.max():.4f} cm, mean: {z_std.mean():.4f} cm")
        print(f"Z-position range — max: {z_range.max():.4f} cm, mean: {z_range.mean():.4f} cm")

        for i, (s, e) in enumerate(self.layer_vertex_ranges):
            layer_std = z_std[s:e]
            layer_range = z_range[s:e]
            print(f"  Layer {i}: std max={layer_std.max():.6f}, range max={layer_range.max():.6f}")

        top5 = np.argsort(z_std)[-5:][::-1]
        print("\nTop 5 unstable vertices:")
        for v in top5:
            print(f"  v{v}: std={z_std[v]:.6f}, range={z_range[v]:.6f}, z_mean={np.mean(z[:, v]):.6f}")

        return {
            "positions": positions,
            "z_std": z_std,
            "z_range": z_range,
            "settled_substep": start,
        }


def create_parser():
    parser = newton.examples.create_parser()
    parser.add_argument("--grid-n", type=int, default=1, help="Grid cells per side")
    parser.add_argument("--layers", type=int, default=1, help="Number of stacked layers")
    parser.add_argument("--contact-ke", type=float, default=1e4, help="Contact stiffness")
    parser.add_argument("--tri-ke", type=float, default=1e4, help="Elastic stiffness")
    parser.add_argument("--tri-ka", type=float, default=1e4, help="Area stiffness")
    parser.add_argument("--contact-kd", type=float, default=None, help="Contact damping (auto-scaled by damping mode)")
    parser.add_argument("--tri-kd", type=float, default=None, help="Elastic damping (auto-scaled by damping mode)")
    parser.add_argument("--edge-ke", type=float, default=5.0, help="Bending stiffness")
    parser.add_argument("--edge-kd", type=float, default=None, help="Bending damping (auto-scaled by damping mode)")
    parser.add_argument("--iterations", type=int, default=5, help="VBD iterations")
    parser.add_argument("--substeps", type=int, default=10, help="Substeps per frame")
    parser.add_argument("--frames", type=int, default=120, help="Total frames")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--no-self-contact", action="store_true", help="Disable self-contact")
    parser.add_argument("--fold", action="store_true", help="Fold grid in half along x")
    parser.add_argument("--tube", action="store_true", help="Roll grid into a tube (cylinder along y)")
    parser.add_argument("--grid-ny", type=int, default=None, help="Grid cells in y (defaults to grid-n)")
    parser.add_argument("--particle-radius", type=float, default=0.8, help="Particle radius (cm)")
    parser.add_argument("--density", type=float, default=0.02, help="Cloth area density (g/cm^2)")
    parser.add_argument("--step-ratio", type=float, default=1.0, help="VBD step ratio gamma (default 1.0)")
    parser.add_argument(
        "--material",
        type=str,
        default="stvk",
        choices=["stvk", "neohookean"],
        help="Triangle material model (default stvk)",
    )
    parser.add_argument(
        "--rayleigh-damping",
        action="store_true",
        help="Use Rayleigh damping instead of absolute (default is absolute)",
    )
    return parser


def _resolve_damping_defaults(args):
    """Fill in damping defaults based on the active damping mode."""
    if args.rayleigh_damping:
        # Rayleigh: effective = kd * ke
        if args.contact_kd is None:
            args.contact_kd = 1e-2
        if args.tri_kd is None:
            args.tri_kd = 1.5e-6
        if args.edge_kd is None:
            args.edge_kd = 1e-2
    else:
        # Absolute: effective = kd  (scale Rayleigh defaults by ke)
        if args.contact_kd is None:
            args.contact_kd = 1e-2 * args.contact_ke  # 100
        if args.tri_kd is None:
            args.tri_kd = 1.5e-6 * args.tri_ke  # 0.015
        if args.edge_kd is None:
            args.edge_kd = 1e-2 * args.edge_ke  # 0.05


if __name__ == "__main__":
    # Phase 1: pre-parse damping mode flag before importing newton so the
    # compile-time constant is set before any kernel compilation.
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--rayleigh-damping", action="store_true")
    _pre_args, _ = _pre_parser.parse_known_args()
    use_absolute = not _pre_args.rayleigh_damping

    import newton._src.solvers.vbd.particle_vbd_kernels as _pvk
    import newton._src.solvers.vbd.rigid_vbd_kernels as _rvk

    _damping_tag = "absolute" if use_absolute else "rayleigh"
    wp.config.kernel_cache_dir = os.path.join(
        wp.config.kernel_cache_dir or os.path.expanduser("~/.cache/warp"),
        f"damping_{_damping_tag}",
    )
    if use_absolute:
        # Patch both modules — `from ... import` copies the name binding.
        _rvk._DAMPING_ABSOLUTE = True
        _pvk._DAMPING_ABSOLUTE = True
        print("*** Damping mode: ABSOLUTE ***")
    else:
        _rvk._DAMPING_ABSOLUTE = False
        _pvk._DAMPING_ABSOLUTE = False
        print("*** Damping mode: RAYLEIGH ***")

    # Phase 2: now safe to import newton and parse full args.
    import newton
    import newton.examples
    from newton.solvers import SolverVBD

    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    _resolve_damping_defaults(args)

    example = Example(
        viewer=viewer,
        args=args,
        grid_n=args.grid_n,
        grid_ny=args.grid_ny,
        layers=args.layers,
        contact_ke=args.contact_ke,
        tri_ke=args.tri_ke,
        tri_ka=args.tri_ka,
        contact_kd=args.contact_kd,
        tri_kd=args.tri_kd,
        edge_ke=args.edge_ke,
        edge_kd=args.edge_kd,
        iterations=args.iterations,
        substeps=args.substeps,
        seed=args.seed,
        enable_self_contact=not args.no_self_contact,
        particle_radius=args.particle_radius,
        density=args.density,
        fold=args.fold,
        tube=args.tube,
        step_ratio=args.step_ratio,
        material=args.material,
    )

    newton.examples.run(example, args)

    # Print report after simulation
    example.report()
