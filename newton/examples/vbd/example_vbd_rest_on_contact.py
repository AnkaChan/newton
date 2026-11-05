# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Rest On Contact
#
# Several square cloth sheets start a short distance apart above a pinned
# cloth table and settle under gravity. Every contact in the scene (sheet
# on sheet and sheet on table) goes through the soft self-contact pipeline,
# so the same model runs on the default penalty contact with per-color
# Divide-and-Truncate ("penalty_dat") and on the stored-plane ALM contact
# path ("alm_dat"), and the resting-contact behavior of the two paths can
# be compared directly. The table has no thickness of its own: the cloth's
# self-contact margin sets every resting gap.
#
# Units are centimeters and grams.
#
# Command: python -m newton.examples vbd_rest_on_contact
#          python -m newton.examples vbd_rest_on_contact --contact-method alm_dat
#          python -m newton.examples vbd_rest_on_contact --contact-mu 0.0
#          python -m newton.examples vbd_rest_on_contact --contact-method alm_dat --dat-truncation-iterations 5 \
#              --layers 5 --layer-spacing 0.25 --first-layer-height 0.25 --initial-velocity -100 --substeps 5
#
###########################################################################

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd._viewer import set_viewer_camera


def _option(args, name: str, default):
    """Read an optional command-line value, falling back when the example is launched without it."""
    value = getattr(args, name, None) if args is not None else None
    return default if value is None else value


class Example:
    CELL_SIZE = 1.0
    """Edge length of one cloth cell [cm]."""
    DENSITY = 0.02
    """Area density of the sheets [g/cm^2]."""
    SELF_CONTACT_MARGIN = 0.2
    """Interaction distance of every self-contact pair [cm]."""
    SELF_CONTACT_GAP = 0.1
    """Extra detection reach beyond the margin [cm]."""
    LAYER_SPACING = 0.5
    """Initial vertical spacing between consecutive sheets [cm]."""
    TABLE_COLOR = (0.78, 0.78, 0.80)
    SHEET_COLORS = ((0.93, 0.52, 0.25), (0.33, 0.70, 0.86), (0.55, 0.78, 0.35), (0.87, 0.42, 0.60))

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = int(_option(args, "fps", 60))
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(_option(args, "substeps", 10))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = int(_option(args, "iterations", 5))

        self.contact_method = str(_option(args, "contact_method", "penalty_dat"))
        self.contact_mu = float(_option(args, "contact_mu", 0.5))
        self.dat_truncation_iterations = int(_option(args, "dat_truncation_iterations", 2))
        self.layers = int(_option(args, "layers", 4))
        self.grid_n = int(_option(args, "grid_n", 10))
        self.layer_spacing = float(_option(args, "layer_spacing", self.LAYER_SPACING))
        self.first_layer_height = float(_option(args, "first_layer_height", self.SELF_CONTACT_MARGIN + 0.1))
        self.initial_velocity = float(_option(args, "initial_velocity", 0.0))
        self.with_table = bool(_option(args, "table", True))
        self.fold = bool(_option(args, "fold", False))
        self.fold_gap = float(_option(args, "fold_gap", 0.3))
        self.fold_clean = bool(_option(args, "fold_clean", False))
        self.contact_gap = float(_option(args, "contact_gap", self.SELF_CONTACT_GAP))
        self.contact_buffer_scale = int(_option(args, "contact_buffer_scale", 1))
        self.dat_plane_update_interval = int(_option(args, "dat_plane_update_interval", 1))
        seed = int(_option(args, "seed", 42))

        cell = self.CELL_SIZE
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -981.0))

        # the table: a pinned cloth sheet three times the size of a stacked sheet (optional:
        # without it the sheets fall freely, which isolates the damping of the contact method)
        if self.with_table:
            table_cells = 3 * self.grid_n
            half_table = 0.5 * table_cells * cell
            builder.add_cloth_grid(
                pos=wp.vec3(-half_table, -half_table, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=table_cells,
                dim_y=table_cells,
                cell_x=cell,
                cell_y=cell,
                mass=0.0,
                tri_ke=1.0e4,
                tri_ka=1.0e4,
                tri_kd=1.5e-2,
                edge_ke=5.0,
                edge_kd=5.0e-2,
                particle_radius=0.2,
                color=self.TABLE_COLOR,
            )
        self.table_particle_count = builder.particle_count

        # the stacked sheets, each slightly shifted and rotated so no two are aligned
        rng = np.random.default_rng(seed)
        half = 0.5 * self.grid_n * cell
        base_z = self.first_layer_height
        self.layer_ranges = []
        self.layer_placements = []
        for layer in range(self.layers):
            dx, dy = (float(v) for v in rng.uniform(-0.2, 0.2, size=2))
            angle = float(rng.uniform(-0.1, 0.1))
            self.layer_placements.append((dx, dy, angle))
            start = builder.particle_count
            builder.add_cloth_grid(
                pos=wp.vec3(-half + dx, -half + dy, base_z + layer * self.layer_spacing),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
                vel=wp.vec3(0.0, 0.0, self.initial_velocity),
                dim_x=self.grid_n,
                dim_y=self.grid_n,
                cell_x=cell,
                cell_y=cell,
                mass=self.DENSITY * cell * cell,
                tri_ke=1.0e4,
                tri_ka=1.0e4,
                tri_kd=1.5e-2,
                edge_ke=5.0,
                edge_kd=5.0e-2,
                particle_radius=0.2,
                color=self.SHEET_COLORS[layer % len(self.SHEET_COLORS)],
            )
            self.layer_ranges.append((start, builder.particle_count))

        if self.fold and self.fold_clean:
            # fold every sheet along its own middle grid column: reflect in the sheet's local
            # frame, so the crease follows a mesh column and no triangle is crushed
            for (start, end), (dx, dy, angle) in zip(self.layer_ranges, self.layer_placements, strict=True):
                ox, oy = -half + dx, -half + dy
                c, s = math.cos(angle), math.sin(angle)
                for i in range(start, end):
                    px, py, pz = builder.particle_q[i]
                    lx = c * (px - ox) + s * (py - oy)
                    ly = -s * (px - ox) + c * (py - oy)
                    if lx > half + 1.0e-3:
                        lx = 2.0 * half - lx
                        builder.particle_q[i] = (ox + c * lx - s * ly, oy + s * lx + c * ly, pz + self.fold_gap)
        elif self.fold:
            # fold every sheet in half along world x: the right half is reflected over the sheet's
            # mid-line and lifted by the fold gap, so each sheet becomes a double layer joined
            # at the crease (the same construction as the original grid_on_table script; because
            # the sheets are rotated, the crease cuts across mesh columns and crushes the
            # triangles it passes through)
            for start, end in self.layer_ranges:
                xs = np.array([builder.particle_q[i][0] for i in range(start, end)])
                x_mid = 0.5 * (xs.min() + xs.max())
                for i in range(start, end):
                    px, py, pz = builder.particle_q[i]
                    if px > x_mid + 1.0e-3:
                        builder.particle_q[i] = (2.0 * x_mid - px, py, pz + self.fold_gap)

        builder.color(include_bending=True)
        self.model = builder.finalize()

        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = self.contact_mu

        solver_options = {
            "particle_vertex_contact_buffer_size": 8 * self.contact_buffer_scale,
            "particle_edge_contact_buffer_size": 16 * self.contact_buffer_scale,
        }
        if self.contact_method == "alm_dat":
            solver_options["particle_dat_truncation_iterations"] = self.dat_truncation_iterations
            solver_options["particle_dat_plane_update_interval"] = self.dat_plane_update_interval
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=self.SELF_CONTACT_MARGIN,
            particle_self_contact_gap=self.contact_gap,
            particle_self_contact_method=self.contact_method,
            **solver_options,
        )
        self.sheet_masses = self.model.particle_mass.numpy()[self.table_particle_count :]

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if self.with_table:
            set_viewer_camera(self.viewer, pos=wp.vec3(26.0, -22.0, 15.0), target=wp.vec3(0.0, 0.0, 0.4), fov=26.0)
        else:
            # free fall: a static camera framing about a metre of travel below the start
            set_viewer_camera(self.viewer, pos=wp.vec3(120.0, -140.0, -30.0), target=wp.vec3(0.0, 0.0, -37.0), fov=30.0)
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None and hasattr(renderer, "draw_edges"):
            renderer.draw_edges = True

        # per-frame heights, speeds and kinetic energy of the free sheets, for the rest report
        self.sheet_heights = []
        self.sheet_max_speeds = []
        self.sheet_kinetic_energies = []
        self.sheet_mean_velocities = []

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

        q = self.state_0.particle_q.numpy()[self.table_particle_count :]
        qd = self.state_0.particle_qd.numpy()[self.table_particle_count :]
        self.sheet_heights.append(q[:, 2].copy())
        self.sheet_max_speeds.append(float(np.linalg.norm(qd, axis=1).max()))
        self.sheet_kinetic_energies.append(float(0.5 * (self.sheet_masses * (qd * qd).sum(axis=1)).sum()))
        self.sheet_mean_velocities.append(float(qd[:, 2].mean()))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        free = list(range(self.table_particle_count, self.model.particle_count))
        if self.with_table:
            newton.examples.test_particle_state(
                self.state_0,
                "sheets stay above the table",
                lambda q, qd: q[2] > -1.0e-3,
                indices=free,
            )
        if self.contact_method == "alm_dat" and not self.solver.particle_dat_coverage_ok():
            raise ValueError("alm_dat self-contact buffers overflowed: some pairs had no stored plane")

    def report(self, window_frames: int = 60) -> dict:
        """Print and return how still the sheets are over the last ``window_frames`` frames."""
        if not self.sheet_heights:
            return {}
        heights = np.array(self.sheet_heights[-window_frames:])
        speeds = np.array(self.sheet_max_speeds[-window_frames:])
        z_std = heights.std(axis=0)
        z_range = np.ptp(heights, axis=0)
        result = {
            "contact_method": self.contact_method,
            "frames": len(self.sheet_heights),
            "window_frames": int(heights.shape[0]),
            "max_speed_cm_s": float(speeds.max()),
            "layers": [],
        }
        print(f"rest report ({self.contact_method}, last {heights.shape[0]} frames)")
        print(f"  largest sheet speed: {speeds.max():.4f} cm/s")
        for layer, (start, end) in enumerate(self.layer_ranges):
            s = slice(start - self.table_particle_count, end - self.table_particle_count)
            entry = {
                "mean_height_cm": float(heights[:, s].mean()),
                "z_std_max_cm": float(z_std[s].max()),
                "z_range_max_cm": float(z_range[s].max()),
            }
            result["layers"].append(entry)
            print(
                f"  sheet {layer}: mean height {entry['mean_height_cm']:.4f} cm, "
                f"z std max {entry['z_std_max_cm']:.6f} cm, z range max {entry['z_range_max_cm']:.6f} cm"
            )
        return result


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--contact-method",
        type=str,
        default="penalty_dat",
        choices=["penalty_dat", "alm_dat"],
        help="Soft self-contact method of SolverVBD.",
    )
    parser.add_argument(
        "--contact-mu",
        type=float,
        default=0.5,
        help="Self-contact friction coefficient.",
    )
    parser.add_argument(
        "--dat-truncation-iterations",
        type=int,
        default=2,
        help="Final truncation rounds of the alm_dat path (particle_dat_truncation_iterations).",
    )
    parser.add_argument(
        "--dat-plane-update-interval",
        type=int,
        default=1,
        help="alm_dat: truncate and rebuild the stored planes every this many iterations (0 disables).",
    )
    parser.add_argument(
        "--contact-buffer-scale",
        type=int,
        default=1,
        help="Multiplier on the default self-contact buffer sizes (raise it when many sheets lie within the detection radius).",
    )
    parser.add_argument("--layers", type=int, default=4, help="Number of stacked sheets.")
    parser.add_argument(
        "--layer-spacing", type=float, default=0.5, help="Initial vertical spacing between sheets [cm]."
    )
    parser.add_argument(
        "--first-layer-height", type=float, default=0.3, help="Initial height of the lowest sheet above the table [cm]."
    )
    parser.add_argument(
        "--initial-velocity", type=float, default=0.0, help="Initial vertical velocity of every sheet [cm/s]."
    )
    parser.add_argument(
        "--fold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fold every sheet in half along x before stacking (double layers joined at the crease).",
    )
    parser.add_argument(
        "--fold-gap", type=float, default=0.3, help="Gap between the two halves of a folded sheet [cm]."
    )
    parser.add_argument(
        "--fold-clean",
        action="store_true",
        help="Fold along each sheet's own middle mesh column (no crushed crease triangles) instead of world x.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Frames per second (dt = 1 / (fps * substeps)).")
    parser.add_argument(
        "--contact-gap", type=float, default=0.1, help="Extra self-contact detection reach beyond the margin [cm]."
    )
    parser.add_argument(
        "--table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the pinned cloth table (--no-table lets the sheets fall freely).",
    )
    parser.add_argument("--grid-n", type=int, default=10, help="Cells per side of each sheet.")
    parser.add_argument("--substeps", type=int, default=10, help="Substeps per frame.")
    parser.add_argument("--iterations", type=int, default=5, help="VBD iterations per substep.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the sheet perturbations.")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.report()
