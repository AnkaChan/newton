#!/usr/bin/env python
"""Study the effect of VBD step ratio (gamma) on convergence and cloth behavior.

Experiments:
  1. Residual vs timestep for different gammas (cloth hanging + grid_on_table)
  2. Residual vs iteration at a near-rest contact state
  3. Cloth sag measurement (softness test)

Usage:
    uv run --extra examples python scripts/study_step_ratio.py --exp 1
    uv run --extra examples python scripts/study_step_ratio.py --exp 2
    uv run --extra examples python scripts/study_step_ratio.py --exp 3
    uv run --extra examples python scripts/study_step_ratio.py --exp all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import warp as wp

# Patch damping mode before importing newton
import newton._src.solvers.vbd.rigid_vbd_kernels as _rvk
import newton._src.solvers.vbd.particle_vbd_kernels as _pvk

_rvk._DAMPING_ABSOLUTE = True
_pvk._DAMPING_ABSOLUTE = True
wp.config.kernel_cache_dir = os.path.join(
    wp.config.kernel_cache_dir or os.path.expanduser("~/.cache/warp"),
    "damping_absolute",
)

import newton
import newton.examples
from newton.solvers import SolverVBD

OUT_DIR = os.path.expanduser("~/Desktop/scripts/step_ratio_study")
os.makedirs(OUT_DIR, exist_ok=True)

GAMMAS = [0.3, 0.5, 0.7, 0.9, 1.0]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def compute_residual(solver):
    """Return RMS force norm from the solver's last iteration."""
    fnorm_sq = solver.force_norm_sq.numpy()
    return np.sqrt(fnorm_sq.mean())


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------

def build_cloth_hanging(gamma, iterations=5, substeps=10, width=32, height=16):
    """Cloth fixed on one side, no ground, hanging under gravity."""
    builder = newton.ModelBuilder()
    # No ground plane!

    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 4.0),
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=width,
        dim_y=height,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        fix_left=True,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=1.0e-1,
        edge_ke=1.0e1,
        edge_kd=0.0,
        particle_radius=0.05,
    )
    builder.color(include_bending=True)
    model = builder.finalize()
    model.soft_contact_ke = 1.0e2
    model.soft_contact_kd = 1.0e0
    model.soft_contact_mu = 1.0
    model.edge_rest_angle.zero_()

    solver = SolverVBD(
        model,
        iterations=iterations,
        step_ratio=gamma,
        particle_enable_tile_solve=False,
    )

    return model, solver


def build_grid_on_table(gamma, iterations=5, substeps=10, grid_n=20, grid_ny=10, layers=3):
    """Tube on table (grid_on_table.py --tube --layers N)."""
    cell_size = 1.0
    contact_ke = 1e4
    tri_ke = 1e4
    tri_ka = 1e4
    edge_ke = 5.0
    density = 0.02
    particle_radius = 0.8

    # Absolute damping defaults
    contact_kd = 1e-2 * contact_ke
    tri_kd = 1.5e-6 * tri_ke
    edge_kd = 1e-2 * edge_ke

    builder = newton.ModelBuilder(gravity=-981.0)

    ground_cfg = builder.default_shape_cfg.copy()
    ground_cfg.ke = contact_ke
    ground_cfg.kd = contact_kd
    ground_cfg.mu = 1.5
    builder.add_ground_plane(cfg=ground_cfg)

    rng = np.random.default_rng(42)
    tube_radius = (grid_n * cell_size) / (2.0 * np.pi)
    layer_spacing = 0.5 + 2.0 * tube_radius
    base_z = tube_radius + particle_radius + 0.1
    cell_area = cell_size * cell_size
    mass_per_particle = density * cell_area

    layer_vertex_ranges = []
    for layer_i in range(layers):
        z = base_z + layer_i * layer_spacing
        dx = rng.uniform(-0.2, 0.2)
        dy = rng.uniform(-0.2, 0.2)
        angle = rng.uniform(-0.1, 0.1)

        start_idx = builder.particle_count
        builder.add_cloth_grid(
            pos=wp.vec3(dx, dy, z),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=grid_n, dim_y=grid_ny,
            cell_x=cell_size, cell_y=cell_size,
            mass=mass_per_particle,
            tri_ke=tri_ke, tri_ka=tri_ka, tri_kd=tri_kd,
            edge_ke=edge_ke, edge_kd=edge_kd,
            particle_radius=particle_radius,
        )
        end_idx = builder.particle_count
        layer_vertex_ranges.append((start_idx, end_idx))

    # Roll into tubes
    for start_idx, end_idx in layer_vertex_ranges:
        positions = np.array([list(builder.particle_q[i]) for i in range(start_idx, end_idx)])
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        x_span = x_max - x_min
        x_center = (x_min + x_max) / 2.0
        r = x_span / (2.0 * np.pi)
        z_base = positions[:, 2].mean()
        for i in range(start_idx, end_idx):
            px, py, pz = builder.particle_q[i]
            theta = (px - x_min) / x_span * 2.0 * np.pi
            new_x = x_center + r * np.sin(theta)
            new_z = z_base + r * (1.0 - np.cos(theta))
            builder.particle_q[i] = (new_x, py, new_z)

    builder.color(include_bending=True)
    model = builder.finalize()
    model.soft_contact_ke = contact_ke
    model.soft_contact_kd = contact_kd
    model.soft_contact_mu = 0.5

    solver = SolverVBD(
        model,
        iterations=iterations,
        step_ratio=gamma,
        particle_enable_self_contact=(layers > 1),
        particle_self_contact_radius=0.2,
        particle_self_contact_margin=0.2,
        particle_enable_tile_solve=False,
    )
    model.edge_rest_angle.zero_()

    return model, solver


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(model, solver, frames, substeps=10, fps=60,
                   record_residual=True, record_positions=False):
    """Run simulation, return per-substep residuals and optionally positions."""
    dt = 1.0 / (fps * substeps)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    residuals = []
    positions = []

    for frame in range(frames):
        for sub in range(substeps):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

            if record_residual:
                residuals.append(compute_residual(solver))
            if record_positions:
                positions.append(state_0.particle_q.numpy().copy())

    result = {"state": state_0, "state_alt": state_1, "control": control}
    if record_residual:
        result["residuals"] = np.array(residuals)
    if record_positions:
        result["positions"] = np.array(positions)
    return result


# ---------------------------------------------------------------------------
# Experiment 1: Residual vs timestep
# ---------------------------------------------------------------------------

def experiment_1():
    """Residual vs timestep for different gammas on two scenes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = {
        "cloth_hanging": {
            "builder": build_cloth_hanging,
            "frames": 120,
            "substeps": 10,
            "title": "Cloth Hanging (no ground)",
        },
        "grid_on_table": {
            "builder": build_grid_on_table,
            "frames": 60,
            "substeps": 10,
            "title": "Tubes on Table (3 layers)",
        },
    }

    for scene_name, scene_cfg in scenes.items():
        print(f"\n{'='*60}")
        print(f"Experiment 1 — {scene_cfg['title']}")
        print(f"{'='*60}")

        all_residuals = {}
        for gi, gamma in enumerate(GAMMAS):
            print(f"  gamma={gamma} ...", end=" ", flush=True)
            t0 = time.time()
            model, solver = scene_cfg["builder"](gamma, iterations=5, substeps=scene_cfg["substeps"])
            result = run_simulation(
                model, solver,
                frames=scene_cfg["frames"],
                substeps=scene_cfg["substeps"],
                record_residual=True,
            )
            elapsed = time.time() - t0
            res = result["residuals"]
            all_residuals[gamma] = res
            print(f"done ({elapsed:.1f}s), final residual={res[-1]:.6e}")

        # Save data
        np.savez_compressed(
            os.path.join(OUT_DIR, f"exp1_{scene_name}.npz"),
            gammas=np.array(GAMMAS),
            **{f"residual_{g}": all_residuals[g] for g in GAMMAS},
        )

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for gi, gamma in enumerate(GAMMAS):
            res = all_residuals[gamma]
            ax.semilogy(res, color=COLORS[gi], label=f"γ={gamma}", alpha=0.85, linewidth=1.2)

        ax.set_xlabel("Substep")
        ax.set_ylabel("RMS Force Residual (log)")
        ax.set_title(f"Exp 1: Residual vs Timestep — {scene_cfg['title']}\n(5 iterations per substep)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"exp1_{scene_name}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: exp1_{scene_name}.png")


# ---------------------------------------------------------------------------
# Experiment 2: Residual vs iteration (single near-rest state)
# ---------------------------------------------------------------------------

def evaluate_consistent_residual(solver, model, state_in, state_out, control, contacts, dt, iter_num):
    """Run one evaluation-only iteration (step_ratio=0) to get a consistent residual.

    Saves and restores positions and displacements so the evaluation is non-destructive.
    """
    # Save state that the eval pass might disturb
    saved_q = state_in.particle_q.numpy().copy()
    saved_disp = solver.particle_displacements.numpy().copy()
    saved_pos_prev = solver.pos_prev_collision_detection.numpy().copy()
    saved_ratio = solver.step_ratio

    solver.step_ratio = 0.0
    solver._solve_particle_iteration(state_in, state_out, contacts, dt, iter_num)
    res = compute_residual(solver)

    # Restore
    solver.step_ratio = saved_ratio
    wp.copy(state_in.particle_q, wp.array(saved_q, dtype=wp.vec3, device=solver.device))
    wp.copy(solver.particle_displacements, wp.array(saved_disp, dtype=wp.vec3, device=solver.device))
    wp.copy(solver.pos_prev_collision_detection, wp.array(saved_pos_prev, dtype=wp.vec3, device=solver.device))
    return res


def experiment_2():
    """Residual vs iteration at a near-rest contact state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print("Experiment 2 — Residual vs Iteration (near-rest state)")
    print(f"{'='*60}")

    # Warm up to near-rest with gamma=1.0 (baseline)
    # Substep 400 (frame 40) is validated as near-rest: KE≈11, tubes settled on table
    warmup_frames = 40
    warmup_substeps = 10
    print(f"  Warming up for {warmup_frames} frames ({warmup_frames * warmup_substeps} substeps)...")

    model, solver = build_grid_on_table(gamma=1.0, iterations=5, substeps=warmup_substeps)
    result = run_simulation(model, solver, frames=warmup_frames, substeps=warmup_substeps,
                            record_residual=False, record_positions=False)

    # Save the warmed-up state
    base_state_q = result["state"].particle_q.numpy().copy()
    base_state_qd = result["state"].particle_qd.numpy().copy()
    print(f"  Warmup done. Testing per-iteration convergence...")

    # For each gamma: restore state, run 1 substep with many iterations, track residual per iteration
    max_iters = 50
    fps = 60
    dt = 1.0 / (fps * warmup_substeps)

    all_residuals = {}
    for gi, gamma in enumerate(GAMMAS):
        print(f"  gamma={gamma} ...", end=" ", flush=True)

        # Rebuild solver with high iterations
        _, solver = build_grid_on_table(gamma=gamma, iterations=1, substeps=warmup_substeps)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Restore warmed-up state
        wp.copy(state_0.particle_q, wp.array(base_state_q, dtype=wp.vec3, device=solver.device))
        wp.copy(state_0.particle_qd, wp.array(base_state_qd, dtype=wp.vec3, device=solver.device))

        # Manual iteration loop for fine-grained residual tracking
        state_0.clear_forces()
        contacts = model.collide(state_0)
        solver._initialize_rigid_bodies(state_0, control, contacts, dt, True)
        solver._initialize_particles(state_0, state_1, dt)

        # Evaluate initial residual (before any iteration)
        iter_residuals = [evaluate_consistent_residual(
            solver, model, state_0, state_1, control, contacts, dt, 0
        )]

        for it in range(max_iters):
            solver._solve_rigid_body_iteration(state_0, state_1, control, contacts, dt)
            solver._solve_particle_iteration(state_0, state_1, contacts, dt, it)
            # Consistent residual: eval-only pass with step_ratio=0
            iter_residuals.append(evaluate_consistent_residual(
                solver, model, state_0, state_1, control, contacts, dt, it + 1
            ))

        solver._finalize_rigid_bodies(state_1, dt)
        solver._finalize_particles(state_1, dt)

        all_residuals[gamma] = np.array(iter_residuals)
        print(f"done, before={iter_residuals[0]:.4e}, after50={iter_residuals[-1]:.4e}, "
              f"ratio={iter_residuals[-1]/iter_residuals[0]:.4f}")

    # Save data
    np.savez_compressed(
        os.path.join(OUT_DIR, "exp2_per_iteration.npz"),
        gammas=np.array(GAMMAS),
        **{f"residual_{g}": all_residuals[g] for g in GAMMAS},
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for gi, gamma in enumerate(GAMMAS):
        res = all_residuals[gamma]
        ax.semilogy(range(len(res)), res, color=COLORS[gi],
                     label=f"γ={gamma}", linewidth=1.5, marker=".", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMS Force Residual (log)")
    ax.set_title("Per-Iteration Convergence\n(consistent eval, step_ratio=0 probe)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot convergence ratio (residual[k] / residual[0])
    ax = axes[1]
    for gi, gamma in enumerate(GAMMAS):
        res = all_residuals[gamma]
        ratio = res / res[0]
        ax.semilogy(range(len(ratio)), ratio, color=COLORS[gi],
                     label=f"γ={gamma}", linewidth=1.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="no improvement")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual / Initial Residual")
    ax.set_title("Normalized Convergence Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 2: Tubes on Table (3 layers), Near-Rest Contact State", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp2_per_iteration.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp2_per_iteration.png")


# ---------------------------------------------------------------------------
# Experiment 3: Cloth sag measurement
# ---------------------------------------------------------------------------

def experiment_3():
    """Measure cloth sag (vertical extent) for different gammas.

    Uses the z-centroid of free particles averaged over the final portion
    of the simulation to get a stable equilibrium measurement despite
    pendulum oscillation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print("Experiment 3 — Cloth Sag (softness test)")
    print(f"{'='*60}")

    frames = 600  # 10 seconds at 60fps — enough for pendulum to damp
    substeps = 10
    avg_window = 500  # average over last 500 substeps for equilibrium measurement

    all_lowest_z = {}
    all_centroid_z = {}
    all_lowest_z_over_time = {}
    all_centroid_z_over_time = {}

    for gi, gamma in enumerate(GAMMAS):
        print(f"  gamma={gamma} ...", end=" ", flush=True)
        t0 = time.time()

        model, solver = build_cloth_hanging(gamma, iterations=5, substeps=substeps,
                                             width=32, height=16)
        result = run_simulation(
            model, solver, frames=frames, substeps=substeps,
            record_residual=False, record_positions=True,
        )

        positions = result["positions"]  # [substeps*frames, N, 3]
        # Identify free particles (not fixed)
        pos_range = np.ptp(positions[:, :, 2], axis=0)
        free_mask = pos_range > 1e-6

        # Metric 1: lowest z of free particles over time
        lowest_z_series = np.array([positions[t, free_mask, 2].min() for t in range(len(positions))])
        # Metric 2: z-centroid of free particles over time
        centroid_z_series = np.array([positions[t, free_mask, 2].mean() for t in range(len(positions))])

        # Equilibrium values: time-averaged over last window
        eq_lowest = lowest_z_series[-avg_window:].mean()
        eq_centroid = centroid_z_series[-avg_window:].mean()

        all_lowest_z[gamma] = eq_lowest
        all_centroid_z[gamma] = eq_centroid
        all_lowest_z_over_time[gamma] = lowest_z_series
        all_centroid_z_over_time[gamma] = centroid_z_series

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s), eq lowest_z={eq_lowest:.4f}, eq centroid_z={eq_centroid:.4f}")

    # Save data
    np.savez_compressed(
        os.path.join(OUT_DIR, "exp3_cloth_sag.npz"),
        gammas=np.array(GAMMAS),
        eq_lowest_z=np.array([all_lowest_z[g] for g in GAMMAS]),
        eq_centroid_z=np.array([all_centroid_z[g] for g in GAMMAS]),
        **{f"lowest_z_time_{g}": all_lowest_z_over_time[g] for g in GAMMAS},
        **{f"centroid_z_time_{g}": all_centroid_z_over_time[g] for g in GAMMAS},
    )

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Bar chart of equilibrium z-centroid
    ax = axes[0, 0]
    vals = [all_centroid_z[g] for g in GAMMAS]
    bars = ax.bar(range(len(GAMMAS)), vals, color=COLORS)
    ax.set_xticks(range(len(GAMMAS)))
    ax.set_xticklabels([f"γ={g}" for g in GAMMAS])
    ax.set_ylabel("Z-Centroid of Free Particles [m]")
    ax.set_title("Equilibrium Sag (z-centroid, lower = more sag)")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # Top-right: Bar chart of equilibrium lowest z
    ax = axes[0, 1]
    vals = [all_lowest_z[g] for g in GAMMAS]
    bars = ax.bar(range(len(GAMMAS)), vals, color=COLORS)
    ax.set_xticks(range(len(GAMMAS)))
    ax.set_xticklabels([f"γ={g}" for g in GAMMAS])
    ax.set_ylabel("Lowest Z of Free Particles [m]")
    ax.set_title("Equilibrium Lowest Point (lower = more sag)")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # Bottom-left: Z-centroid over time
    ax = axes[1, 0]
    for gi, gamma in enumerate(GAMMAS):
        ax.plot(all_centroid_z_over_time[gamma], color=COLORS[gi],
                label=f"γ={gamma}", alpha=0.85, linewidth=1.0)
    ax.set_xlabel("Substep")
    ax.set_ylabel("Z-Centroid [m]")
    ax.set_title("Z-Centroid Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Lowest z over time
    ax = axes[1, 1]
    for gi, gamma in enumerate(GAMMAS):
        ax.plot(all_lowest_z_over_time[gamma], color=COLORS[gi],
                label=f"γ={gamma}", alpha=0.85, linewidth=1.0)
    ax.set_xlabel("Substep")
    ax.set_ylabel("Lowest Z [m]")
    ax.set_title("Lowest Point Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 3: Cloth Sag — 32x16 cloth, one side fixed, no ground\n"
                 f"(5 iters, {frames} frames, avg over last {avg_window} substeps)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp3_cloth_sag.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp3_cloth_sag.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Study step ratio effect on VBD convergence")
    parser.add_argument("--exp", type=str, default="all",
                        help="Which experiment to run: 1, 2, 3, or all")
    args = parser.parse_args()

    wp.init()

    experiments = {
        "1": experiment_1,
        "2": experiment_2,
        "3": experiment_3,
    }

    if args.exp == "all":
        for name, func in experiments.items():
            func()
    elif args.exp in experiments:
        experiments[args.exp]()
    else:
        print(f"Unknown experiment: {args.exp}. Use 1, 2, 3, or all.")
        sys.exit(1)

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
