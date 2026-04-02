#!/usr/bin/env python
"""Analyze oscillating vertices for grid_n=10, layers=5, folded.

Runs with DebugRecorder to capture per-iteration forces, hessians, contacts,
then identifies unstable vertices and prints what's changing.

Usage:
    CUDA_VISIBLE_DEVICES=3 uv run --extra examples python scripts/analyze_grid_instability.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton.solvers import SolverVBD
from newton._src.solvers.vbd.debug_recorder import DebugRecorder

sys.path.insert(0, os.path.dirname(__file__))
from grid_on_table import find_settled_substep

# ---------- Config ----------
GRID_N = 10
LAYERS = 5
FOLD = True
FRAMES = 100
SUBSTEPS = 10
ITERATIONS = 5
FPS = 60
TOP_K = 5  # vertices to analyze

# T-shirt regime params
CONTACT_KE = 1e4
TRI_KE = 1e4
TRI_KA = 1e4
CONTACT_KD = 1e-2
TRI_KD = 1.5e-6
EDGE_KE = 5.0
EDGE_KD = 1e-2
PARTICLE_RADIUS = 0.8
SELF_CONTACT_RADIUS = 0.2
SELF_CONTACT_MARGIN = 0.2
DENSITY = 0.02
SEED = 42


def build_scene():
    cell_size = 1.0
    builder = newton.ModelBuilder(gravity=-981.0)

    ground_cfg = builder.default_shape_cfg.copy()
    ground_cfg.ke = CONTACT_KE
    ground_cfg.kd = CONTACT_KD
    ground_cfg.mu = 1.5
    builder.add_ground_plane(cfg=ground_cfg)

    rng = np.random.default_rng(SEED)
    fold_gap = PARTICLE_RADIUS * 2 + 0.1
    layer_spacing = 0.5 + fold_gap
    base_z = PARTICLE_RADIUS + 0.1

    cell_area = cell_size * cell_size
    mass_per_particle = DENSITY * cell_area

    layer_ranges = []
    for layer_i in range(LAYERS):
        z = base_z + layer_i * layer_spacing
        dx = rng.uniform(-0.2, 0.2)
        dy = rng.uniform(-0.2, 0.2)
        angle = rng.uniform(-0.1, 0.1)

        start_idx = builder.particle_count
        builder.add_cloth_grid(
            pos=wp.vec3(dx, dy, z),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=GRID_N, dim_y=GRID_N,
            cell_x=cell_size, cell_y=cell_size,
            mass=mass_per_particle,
            tri_ke=TRI_KE, tri_ka=TRI_KA, tri_kd=TRI_KD,
            edge_ke=EDGE_KE, edge_kd=EDGE_KD,
            particle_radius=PARTICLE_RADIUS,
        )
        end_idx = builder.particle_count
        layer_ranges.append((start_idx, end_idx))

    # Fold
    for start_idx, end_idx in layer_ranges:
        positions = []
        for i in range(start_idx, end_idx):
            positions.append(list(builder.particle_q[i]))
        positions = np.array(positions)
        x_mid = (positions[:, 0].min() + positions[:, 0].max()) / 2.0
        for i in range(start_idx, end_idx):
            px, py, pz = builder.particle_q[i]
            if px > x_mid + 0.01:
                builder.particle_q[i] = (2.0 * x_mid - px, py, pz + fold_gap)

    builder.color(include_bending=True)
    model = builder.finalize()

    model.soft_contact_ke = CONTACT_KE
    model.soft_contact_kd = CONTACT_KD
    model.soft_contact_mu = 0.5

    solver = SolverVBD(
        model,
        iterations=ITERATIONS,
        particle_enable_self_contact=True,
        particle_self_contact_radius=SELF_CONTACT_RADIUS,
        particle_self_contact_margin=SELF_CONTACT_MARGIN,
        particle_enable_tile_solve=False,
    )

    model.edge_rest_angle.zero_()

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    return model, solver, state_0, state_1, control, contacts, layer_ranges


def main():
    wp.init()

    model, solver, state_0, state_1, control, contacts, layer_ranges = build_scene()

    total_substeps = FRAMES * SUBSTEPS
    print(f"Particles: {model.particle_count}, Substeps: {total_substeps}, "
          f"Iterations: {total_substeps * ITERATIONS}")

    # Set up recorder
    recorder = DebugRecorder(
        model,
        max_substeps=total_substeps,
        iterations=ITERATIONS,
        device=str(model.device),
    )
    solver.debug_recorder = recorder

    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SUBSTEPS

    # Also collect per-substep positions on CPU
    positions_history = []

    t0 = time.time()
    for frame in range(FRAMES):
        for _ in range(SUBSTEPS):
            state_0.clear_forces()
            state_1.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
            positions_history.append(state_0.particle_q.numpy().copy())

        if (frame + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {frame + 1}/{FRAMES} ({elapsed:.1f}s)")

    wp.synchronize()
    print(f"Simulation done in {time.time() - t0:.1f}s")
    print(f"Recorded {recorder.substeps_recorded} substeps, "
          f"{recorder.iterations_recorded} iterations")

    # --- Find unstable vertices ---
    positions = np.array(positions_history)  # [S, N, 3]
    settled = find_settled_substep(positions)
    print(f"Settled at substep {settled}")

    steady_z = positions[settled:, :, 2]
    z_std = np.std(steady_z, axis=0)
    top_verts = np.argsort(z_std)[-TOP_K:][::-1]

    print(f"\nTop {TOP_K} unstable vertices:")
    for v in top_verts:
        layer_idx = -1
        for li, (s, e) in enumerate(layer_ranges):
            if s <= v < e:
                layer_idx = li
                break
        print(f"  v{v} (layer {layer_idx}): z_std={z_std[v]:.4f} cm, "
              f"z_range={np.ptp(steady_z[:, v]):.4f} cm")

    # --- Export debug data ---
    data = recorder.to_dict()
    N = model.particle_count
    S = recorder.substeps_recorded
    I = recorder.iterations_recorded

    # Per-iteration data: [I, N, ...] — reshape to [S, iters, N, ...]
    iters = ITERATIONS
    assert I == S * iters, f"I={I}, S={S}, iters={iters}"

    pre_forces = data["pre_solve_forces"].reshape(S, iters, N, 3)
    pre_hessians = data["pre_solve_hessians"].reshape(S, iters, N, 3, 3)
    f_inertia = data["f_inertia"].reshape(S, iters, N, 3)
    f_elastic = data["f_elastic"].reshape(S, iters, N, 3)
    f_bending = data["f_bending"].reshape(S, iters, N, 3)
    displacements = data["displacements"].reshape(S, iters, N, 3)
    truncation_ts = data["truncation_ts"].reshape(S, iters, N)

    vt_counts = data["vt_contact_counts"]  # [S, N]
    ee_counts = data["ee_contact_counts"]  # [S, E]
    tv_counts = data["tv_contact_counts"]  # [S, T]

    velocities = data["velocities"]  # [S, N, 3]
    velocities_end = data["velocities_end"]  # [S, N, 3]

    # --- Analyze each unstable vertex ---
    for v in top_verts:
        layer_idx = -1
        for li, (s, e) in enumerate(layer_ranges):
            if s <= v < e:
                layer_idx = li
                break

        print(f"\n{'='*70}")
        print(f"VERTEX {v} (layer {layer_idx}) — z_std={z_std[v]:.4f} cm")
        print(f"{'='*70}")

        # Use steady-state range for analysis
        ss = settled  # start substep

        # 1. Position trajectory
        z_traj = positions[ss:, v, 2]
        print(f"\nZ trajectory (settled): min={z_traj.min():.4f}, max={z_traj.max():.4f}, "
              f"mean={z_traj.mean():.4f}")

        # 2. Velocity
        vel = velocities[ss:, v, :]
        vel_end = velocities_end[ss:, v, :]
        vel_z = vel[:, 2]
        vel_end_z = vel_end[:, 2]
        print(f"\nVelocity Z (start of substep): "
              f"std={vel_z.std():.4f}, range=[{vel_z.min():.4f}, {vel_z.max():.4f}]")
        print(f"Velocity Z (end of substep):   "
              f"std={vel_end_z.std():.4f}, range=[{vel_end_z.min():.4f}, {vel_end_z.max():.4f}]")

        # 3. Force components (last iteration of each substep, during steady state)
        f_in = f_inertia[ss:, -1, v, :]  # [substeps, 3]
        f_el = f_elastic[ss:, -1, v, :]
        f_bn = f_bending[ss:, -1, v, :]
        f_pre = pre_forces[ss:, -1, v, :]  # contact + spring pre-solve

        f_in_mag = np.linalg.norm(f_in, axis=-1)
        f_el_mag = np.linalg.norm(f_el, axis=-1)
        f_bn_mag = np.linalg.norm(f_bn, axis=-1)
        f_pre_mag = np.linalg.norm(f_pre, axis=-1)

        print(f"\nForce magnitudes (last iter, steady state):")
        print(f"  Inertia:  mean={f_in_mag.mean():.2f}, std={f_in_mag.std():.2f}, "
              f"range=[{f_in_mag.min():.2f}, {f_in_mag.max():.2f}]")
        print(f"  Elastic:  mean={f_el_mag.mean():.2f}, std={f_el_mag.std():.2f}, "
              f"range=[{f_el_mag.min():.2f}, {f_el_mag.max():.2f}]")
        print(f"  Bending:  mean={f_bn_mag.mean():.2f}, std={f_bn_mag.std():.2f}, "
              f"range=[{f_bn_mag.min():.2f}, {f_bn_mag.max():.2f}]")
        print(f"  PreSolve: mean={f_pre_mag.mean():.2f}, std={f_pre_mag.std():.2f}, "
              f"range=[{f_pre_mag.min():.2f}, {f_pre_mag.max():.2f}]")

        # Z-components specifically
        print(f"\nForce Z-components (last iter, steady state):")
        print(f"  Inertia Z:  mean={f_in[:, 2].mean():.2f}, std={f_in[:, 2].std():.2f}")
        print(f"  Elastic Z:  mean={f_el[:, 2].mean():.2f}, std={f_el[:, 2].std():.2f}")
        print(f"  Bending Z:  mean={f_bn[:, 2].mean():.2f}, std={f_bn[:, 2].std():.2f}")
        print(f"  PreSolve Z: mean={f_pre[:, 2].mean():.2f}, std={f_pre[:, 2].std():.2f}")

        # Total force and cancellation
        f_total = f_in + f_el + f_bn + f_pre
        f_total_mag = np.linalg.norm(f_total, axis=-1)
        sum_component_mags = f_in_mag + f_el_mag + f_bn_mag + f_pre_mag
        cancel_ratio = np.where(f_total_mag > 1e-10,
                                sum_component_mags / f_total_mag, 0.0)
        print(f"\n  Total force mag: mean={f_total_mag.mean():.2f}, std={f_total_mag.std():.2f}")
        print(f"  Cancellation ratio: mean={cancel_ratio.mean():.1f}, "
              f"max={cancel_ratio.max():.1f}")

        # 4. Hessian analysis
        H = pre_hessians[ss:, -1, v, :, :]  # [substeps, 3, 3]
        # Trace (sum of diagonal) as a simple stiffness measure
        H_trace = H[:, 0, 0] + H[:, 1, 1] + H[:, 2, 2]
        H_zz = H[:, 2, 2]
        print(f"\nHessian (pre-solve, last iter):")
        print(f"  Trace:   mean={H_trace.mean():.2f}, std={H_trace.std():.2f}")
        print(f"  H_zz:    mean={H_zz.mean():.2f}, std={H_zz.std():.2f}, "
              f"range=[{H_zz.min():.2f}, {H_zz.max():.2f}]")

        # 5. Contact counts
        vt = vt_counts[ss:, v]
        print(f"\nSelf-contact VT count: "
              f"mean={vt.mean():.1f}, range=[{vt.min()}, {vt.max()}], "
              f"nonzero={np.count_nonzero(vt)}/{len(vt)} substeps")

        # Count changes (chattering)
        vt_changes = np.sum(np.diff(vt) != 0)
        print(f"  VT count changes: {vt_changes}/{len(vt)-1} substeps")

        # 6. Displacements
        dx = displacements[ss:, -1, v, :]
        dx_mag = np.linalg.norm(dx, axis=-1)
        print(f"\nDisplacement (last iter): "
              f"mag mean={dx_mag.mean():.6f}, std={dx_mag.std():.6f}")
        print(f"  Z: mean={dx[:, 2].mean():.6f}, std={dx[:, 2].std():.6f}")

        # 7. Truncation
        ts = truncation_ts[ss:, -1, v]
        truncated = np.sum(ts < 1.0)
        print(f"\nTruncation: {truncated}/{len(ts)} substeps truncated, "
              f"mean t={ts.mean():.4f}")

        # 8. Cross-iteration convergence within a substep
        # Pick a few representative substeps and show iteration-by-iteration
        sample_substeps = [ss, ss + len(z_traj)//4, ss + len(z_traj)//2]
        print(f"\nIteration-by-iteration forces (sample substeps):")
        for sub_s in sample_substeps:
            if sub_s >= S:
                continue
            print(f"  Substep {sub_s}:")
            for it in range(iters):
                fi = f_inertia[sub_s, it, v, :]
                fe = f_elastic[sub_s, it, v, :]
                fb = f_bending[sub_s, it, v, :]
                fp = pre_forces[sub_s, it, v, :]
                ft = fi + fe + fb + fp
                dx_it = displacements[sub_s, it, v, :]
                print(f"    iter {it}: f_total_z={ft[2]:+.2f}, "
                      f"|f_total|={np.linalg.norm(ft):.2f}, "
                      f"f_inertia_z={fi[2]:+.2f}, f_elastic_z={fe[2]:+.2f}, "
                      f"f_pre_z={fp[2]:+.2f}, dx_z={dx_it[2]:+.6f}")

    # --- Plot trajectories of top vertices ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(TOP_K, 3, figsize=(18, 4 * TOP_K))
    if TOP_K == 1:
        axes = axes[np.newaxis, :]

    for row, v in enumerate(top_verts):
        # Col 0: Z trajectory
        axes[row, 0].plot(positions[:, v, 2], linewidth=0.5)
        axes[row, 0].axvline(settled, color="red", linestyle="--", alpha=0.5, label="settled")
        axes[row, 0].set_ylabel(f"v{v} Z (cm)")
        axes[row, 0].set_title(f"Z position — vertex {v}")
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)

        # Col 1: Force Z components (last iter)
        axes[row, 1].plot(f_inertia[:, -1, v, 2], label="inertia", linewidth=0.5)
        axes[row, 1].plot(f_elastic[:, -1, v, 2], label="elastic", linewidth=0.5)
        axes[row, 1].plot(pre_forces[:, -1, v, 2], label="pre_solve", linewidth=0.5)
        axes[row, 1].set_ylabel("Force Z")
        axes[row, 1].set_title(f"Force Z components — vertex {v}")
        axes[row, 1].legend(fontsize=7)
        axes[row, 1].grid(True, alpha=0.3)

        # Col 2: VT contact count + Hessian Hzz
        ax2a = axes[row, 2]
        ax2b = ax2a.twinx()
        ax2a.plot(vt_counts[:, v], color="tab:green", linewidth=0.5, label="VT count")
        ax2b.plot(pre_hessians[:, -1, v, 2, 2], color="tab:orange", linewidth=0.5, label="H_zz")
        ax2a.set_ylabel("VT contacts", color="tab:green")
        ax2b.set_ylabel("H_zz", color="tab:orange")
        ax2a.set_title(f"Contacts & Hessian — vertex {v}")
        ax2a.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Substep")
    axes[-1, 1].set_xlabel("Substep")
    axes[-1, 2].set_xlabel("Substep")

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "grid_instability_analysis.png")
    out_path = os.path.abspath(out_path)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
