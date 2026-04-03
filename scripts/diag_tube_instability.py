#!/usr/bin/env python
"""Diagnostic script for tube cloth instability.

Runs grid_on_table with --grid-n 20 --grid-ny 10 --layers 2 --tube
using DebugRecorder to capture per-substep and per-iteration data,
then analyzes and reports the top unstable vertices with force/contact breakdown.

Usage:
    CUDA_VISIBLE_DEVICES=4 uv run --extra examples python scripts/diag_tube_instability.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverVBD
from newton._src.solvers.vbd.debug_recorder import DebugRecorder

# ---------- parameters ----------
GRID_N = 20
GRID_NY = 10
LAYERS = 2
NUM_FRAMES = 60
SIM_SUBSTEPS = 10
ITERATIONS = 5
FPS = 60

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

MAX_SUBSTEPS = NUM_FRAMES * SIM_SUBSTEPS
TOP_K = 10  # number of unstable vertices to analyze

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "debug_tube")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)


def build_tube_scene():
    """Build tube scene matching grid_on_table.py --tube parameters."""
    cell_size = 1.0
    builder = newton.ModelBuilder(gravity=-981.0)

    ground_cfg = builder.default_shape_cfg.copy()
    ground_cfg.ke = CONTACT_KE
    ground_cfg.kd = CONTACT_KD
    ground_cfg.mu = 1.5
    builder.add_ground_plane(cfg=ground_cfg)

    rng = np.random.default_rng(42)

    tube_radius = (GRID_N * cell_size) / (2.0 * np.pi)
    tube_diameter = 2.0 * tube_radius
    layer_spacing = 0.5 + tube_diameter
    base_z = tube_radius + PARTICLE_RADIUS + 0.1

    cell_area = cell_size * cell_size
    mass_per_particle = DENSITY * cell_area

    layer_vertex_ranges = []
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
            dim_x=GRID_N,
            dim_y=GRID_NY,
            cell_x=cell_size,
            cell_y=cell_size,
            mass=mass_per_particle,
            tri_ke=TRI_KE,
            tri_ka=TRI_KA,
            tri_kd=TRI_KD,
            edge_ke=EDGE_KE,
            edge_kd=EDGE_KD,
            particle_radius=PARTICLE_RADIUS,
        )

        end_idx = builder.particle_count
        layer_vertex_ranges.append((start_idx, end_idx))

    # Roll into tubes
    for start_idx, end_idx in layer_vertex_ranges:
        positions = []
        for i in range(start_idx, end_idx):
            positions.append(list(builder.particle_q[i]))
        positions = np.array(positions)
        x_min = positions[:, 0].min()
        x_max = positions[:, 0].max()
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

    model.soft_contact_ke = CONTACT_KE
    model.soft_contact_kd = CONTACT_KD
    model.soft_contact_mu = 0.5

    solver = SolverVBD(
        model,
        iterations=ITERATIONS,
        particle_enable_self_contact=True,
        particle_self_contact_radius=SELF_CONTACT_RADIUS,
        particle_self_contact_margin=SELF_CONTACT_MARGIN,
        particle_enable_tile_solve=False,  # required for debug force output
    )

    model.edge_rest_angle.zero_()

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    return model, state_0, state_1, control, contacts, solver, layer_vertex_ranges


def find_unstable_vertices(positions, top_k=TOP_K):
    """Rank vertices by z-position std in steady state."""
    n_substeps = positions.shape[0]
    # Use second half as steady state
    start = n_substeps // 2
    steady = positions[start:]
    z = steady[:, :, 2]
    z_std = np.std(z, axis=0)
    top_indices = np.argsort(z_std)[-top_k:][::-1]
    return top_indices, z_std


def analyze_vertex(v, data, z_std, layer_vertex_ranges):
    """Print detailed analysis for a single vertex."""
    S = data["substeps_recorded"]
    iters = data["iterations_per_substep"]
    I = data["iterations_recorded"]
    N = data["particle_count"]

    # Determine layer
    layer = -1
    for li, (s, e) in enumerate(layer_vertex_ranges):
        if s <= v < e:
            layer = li
            break

    print(f"\n{'='*70}")
    print(f"VERTEX {v} (layer {layer}, z_std={z_std[v]:.6f})")
    print(f"{'='*70}")

    # Position trajectory
    pos = data["positions"][:, v, :]  # [S, 3]
    z = pos[:, 2]
    print(f"\n  Position Z: mean={z.mean():.4f}, std={z.std():.6f}, "
          f"range={np.ptp(z):.6f}, min={z.min():.4f}, max={z.max():.4f}")

    # Velocity
    vel_start = data["velocities"][:, v, :]  # [S, 3]
    vel_end = data["velocities_end"][:, v, :]
    vel_mag = np.linalg.norm(vel_start, axis=1)
    print(f"  Velocity |v|: mean={vel_mag.mean():.4f}, max={vel_mag.max():.4f}")
    vz = vel_start[:, 2]
    sign_changes = np.sum(np.diff(np.sign(vz)) != 0)
    print(f"  Velocity Z sign changes: {sign_changes}/{S}")

    # Force components — last iteration of each substep
    last_iter_indices = np.arange(iters - 1, I, iters)[:S]

    f_inertia = data["f_inertia"][last_iter_indices, v, :]
    f_elastic = data["f_elastic"][last_iter_indices, v, :]
    f_bending = data["f_bending"][last_iter_indices, v, :]
    f_contact = data["pre_solve_forces"][last_iter_indices, v, :]

    mag_inertia = np.linalg.norm(f_inertia, axis=1)
    mag_elastic = np.linalg.norm(f_elastic, axis=1)
    mag_bending = np.linalg.norm(f_bending, axis=1)
    mag_contact = np.linalg.norm(f_contact, axis=1)

    f_net = f_inertia + f_elastic + f_bending + f_contact
    mag_net = np.linalg.norm(f_net, axis=1)
    mag_sum = mag_inertia + mag_elastic + mag_bending + mag_contact
    cancellation = np.where(mag_net > 1e-10, mag_sum / mag_net, 0.0)

    print(f"\n  Forces (last iter of each substep):")
    print(f"  {'Component':<12} {'Mean|f|':<12} {'Max|f|':<12} {'Std|f|':<12}")
    for name, mag in [("Inertia", mag_inertia), ("Elastic", mag_elastic),
                       ("Bending", mag_bending), ("Contact", mag_contact),
                       ("Net", mag_net)]:
        print(f"  {name:<12} {mag.mean():<12.4f} {mag.max():<12.4f} {mag.std():<12.4f}")
    print(f"  Cancellation ratio: mean={cancellation.mean():.2f}, max={cancellation.max():.2f}")

    # Force Z components
    print(f"\n  Force Z components (last iter, mean over substeps):")
    print(f"    Inertia_Z:  mean={f_inertia[:, 2].mean():+.4f}, std={f_inertia[:, 2].std():.4f}")
    print(f"    Elastic_Z:  mean={f_elastic[:, 2].mean():+.4f}, std={f_elastic[:, 2].std():.4f}")
    print(f"    Bending_Z:  mean={f_bending[:, 2].mean():+.4f}, std={f_bending[:, 2].std():.4f}")
    print(f"    Contact_Z:  mean={f_contact[:, 2].mean():+.4f}, std={f_contact[:, 2].std():.4f}")

    # Per-iteration convergence (for a few representative substeps)
    print(f"\n  Per-iteration displacement convergence (sample substeps):")
    sample_substeps = [0, S // 4, S // 2, 3 * S // 4, S - 1]
    for ss in sample_substeps:
        iter_start = ss * iters
        iter_end = iter_start + iters
        if iter_end > I:
            continue
        disp = data["displacements"][iter_start:iter_end, v, :]
        disp_mag = np.linalg.norm(disp, axis=1)
        disp_str = " → ".join(f"{d:.6f}" for d in disp_mag)
        ratio = disp_mag[-1] / disp_mag[0] if disp_mag[0] > 1e-15 else float('inf')
        print(f"    substep {ss:4d}: {disp_str}  (ratio={ratio:.3f})")

    # Hessian
    H = data["pre_solve_hessians"][last_iter_indices, v, :, :]  # [S, 3, 3]
    H_zz = H[:, 2, 2]
    H_trace = np.trace(H, axis1=1, axis2=2)
    print(f"\n  Hessian (pre-solve, last iter):")
    print(f"    H_zz:   mean={H_zz.mean():.2f}, max={H_zz.max():.2f}, min={H_zz.min():.2f}")
    print(f"    Trace:  mean={H_trace.mean():.2f}, max={H_trace.max():.2f}")
    # Condition number
    eigvals = np.linalg.eigvalsh(H)
    max_eig = eigvals[:, -1]
    min_eig = eigvals[:, 0]
    cond = np.where(min_eig > 1e-10, max_eig / min_eig, float('inf'))
    print(f"    Cond:   mean={np.mean(cond[np.isfinite(cond)]):.2f}, max={np.max(cond[np.isfinite(cond)]):.2f}")
    neg_count = np.sum(min_eig < 0)
    print(f"    Negative eigenvalues: {neg_count}/{S} substeps")

    # VT contact
    vt = data["vt_contact_counts"][:, v]
    vt_changes = np.sum(np.diff(vt > 0) != 0)
    print(f"\n  VT self-contact:")
    print(f"    Count: mean={vt.mean():.2f}, max={vt.max()}, min={vt.min()}")
    print(f"    State changes (on/off): {vt_changes}")
    print(f"    Always in contact: {bool(vt.min() > 0)}")
    print(f"    Never in contact: {bool(vt.max() == 0)}")

    if vt.max() > 0:
        vt_dist = data["vt_min_dist"][:, v]
        in_contact = vt > 0
        if in_contact.any():
            print(f"    Min dist when in contact: mean={vt_dist[in_contact].mean():.6f}, "
                  f"min={vt_dist[in_contact].min():.6f}")

    # Truncation
    trunc = data["truncation_ts"][:, v]
    active = trunc < 1.0 - 1e-6
    print(f"\n  Truncation: active in {active.sum()}/{I} iterations")

    return {
        "vertex": v,
        "layer": layer,
        "z_std": z_std[v],
        "z_range": np.ptp(z),
        "max_vel": vel_mag.max(),
        "mean_elastic": mag_elastic.mean(),
        "mean_contact": mag_contact.mean(),
        "mean_inertia": mag_inertia.mean(),
        "cancellation_mean": cancellation.mean(),
        "vt_changes": vt_changes,
        "convergence_issue": cancellation.mean() > 5.0,
    }


def plot_results(data, unstable_verts, z_std, layer_vertex_ranges):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = data["substeps_recorded"]
    iters = data["iterations_per_substep"]
    I = data["iterations_recorded"]

    n_verts = min(5, len(unstable_verts))
    fig, axes = plt.subplots(n_verts, 4, figsize=(20, 4 * n_verts))
    if n_verts == 1:
        axes = axes[np.newaxis, :]

    for row, v in enumerate(unstable_verts[:n_verts]):
        layer = -1
        for li, (s, e) in enumerate(layer_vertex_ranges):
            if s <= v < e:
                layer = li
                break

        last_iter_indices = np.arange(iters - 1, I, iters)[:S]
        substep_range = np.arange(S)

        # Col 0: Z position trajectory
        ax = axes[row, 0]
        z = data["positions"][:, v, 2]
        ax.plot(substep_range, z, linewidth=0.5)
        ax.set_title(f"v{v} (L{layer}) Z position")
        ax.set_xlabel("substep")
        ax.set_ylabel("z (cm)")

        # Col 1: Force Z components
        ax = axes[row, 1]
        f_inertia_z = data["f_inertia"][last_iter_indices, v, 2]
        f_elastic_z = data["f_elastic"][last_iter_indices, v, 2]
        f_bending_z = data["f_bending"][last_iter_indices, v, 2]
        f_contact_z = data["pre_solve_forces"][last_iter_indices, v, 2]
        ax.plot(substep_range, f_inertia_z, label="inertia", alpha=0.7, linewidth=0.5)
        ax.plot(substep_range, f_elastic_z, label="elastic", alpha=0.7, linewidth=0.5)
        ax.plot(substep_range, f_bending_z, label="bending", alpha=0.7, linewidth=0.5)
        ax.plot(substep_range, f_contact_z, label="contact", alpha=0.7, linewidth=0.5)
        ax.set_title(f"v{v} Force Z")
        ax.set_xlabel("substep")
        ax.legend(fontsize=6)

        # Col 2: VT contact count + velocity magnitude
        ax = axes[row, 2]
        vt = data["vt_contact_counts"][:, v]
        ax.plot(substep_range, vt, 'r-', linewidth=0.5, label="VT count")
        ax.set_ylabel("VT count", color='r')
        ax2 = ax.twinx()
        vel_mag = np.linalg.norm(data["velocities"][:, v, :], axis=1)
        ax2.plot(substep_range, vel_mag, 'b-', linewidth=0.5, label="|vel|")
        ax2.set_ylabel("|vel|", color='b')
        ax.set_title(f"v{v} Contacts & Velocity")
        ax.set_xlabel("substep")

        # Col 3: Per-iteration displacement convergence (sample substeps)
        ax = axes[row, 3]
        sample_substeps = np.linspace(0, S - 1, min(10, S), dtype=int)
        for ss in sample_substeps:
            iter_start = ss * iters
            iter_end = iter_start + iters
            if iter_end > I:
                continue
            disp = data["displacements"][iter_start:iter_end, v, :]
            disp_mag = np.linalg.norm(disp, axis=1)
            ax.plot(range(iters), disp_mag, alpha=0.3, linewidth=0.5)
        ax.set_title(f"v{v} Displacement per iter")
        ax.set_xlabel("iteration")
        ax.set_ylabel("|disp|")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "tube_instability_analysis.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close()


def main():
    wp.init()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Building tube scene: grid={GRID_N}x{GRID_NY}, layers={LAYERS}")
    model, state_0, state_1, control, contacts, solver, layer_ranges = build_tube_scene()

    N = model.particle_count
    print(f"Particles: {N}, Edges: {model.edge_count}, Triangles: {model.tri_count}")
    for i, (s, e) in enumerate(layer_ranges):
        print(f"  Layer {i}: vertices [{s}, {e}) = {e - s} particles")

    # Set up recorder
    recorder = DebugRecorder(model, max_substeps=MAX_SUBSTEPS, iterations=ITERATIONS, device=str(model.device))
    solver.debug_recorder = recorder

    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SIM_SUBSTEPS

    # Also record positions per substep (for trajectory analysis)
    positions_history = []

    t0 = time.time()
    for frame in range(NUM_FRAMES):
        for _ in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
            positions_history.append(state_0.particle_q.numpy().copy())

        elapsed = time.time() - t0
        substeps_done = (frame + 1) * SIM_SUBSTEPS
        if (frame + 1) % 10 == 0 or frame == 0:
            print(f"  Frame {frame + 1}/{NUM_FRAMES} | {substeps_done}/{MAX_SUBSTEPS} substeps | {elapsed:.1f}s")

    total = time.time() - t0
    print(f"\nSimulation: {total:.1f}s ({MAX_SUBSTEPS} substeps, {recorder.iterations_recorded} iterations)")

    # Export recorder data
    print("Exporting debug data...")
    data = recorder.to_dict()
    data["tri_indices"] = model.tri_indices.numpy().copy()
    data["edge_indices"] = model.edge_indices.numpy().copy()

    positions = np.array(positions_history)
    data["positions_history"] = positions

    npz_path = os.path.join(OUTPUT_DIR, "debug_tube_instability.npz")
    np.savez_compressed(npz_path, **data)
    file_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"Saved {npz_path} ({file_mb:.1f} MB)")

    # Find unstable vertices
    unstable_verts, z_std = find_unstable_vertices(positions, TOP_K)

    print(f"\n{'='*70}")
    print(f"TOP-{TOP_K} UNSTABLE VERTICES (by z-position std, steady state)")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Vtx':<8} {'Z_std':<12} {'Layer':<8}")
    for rank, v in enumerate(unstable_verts):
        layer = -1
        for li, (s, e) in enumerate(layer_ranges):
            if s <= v < e:
                layer = li
                break
        print(f"{rank + 1:<6} {v:<8} {z_std[v]:<12.6f} {layer:<8}")

    # Detailed analysis for each
    summaries = []
    for v in unstable_verts:
        summary = analyze_vertex(v, data, z_std, layer_ranges)
        summaries.append(summary)

    # Classification
    print(f"\n{'='*70}")
    print("INSTABILITY CLASSIFICATION")
    print(f"{'='*70}")
    for s in summaries:
        v = s["vertex"]
        mechanisms = []
        if s["mean_elastic"] > 10 * s["mean_inertia"] and s["vt_changes"] == 0:
            mechanisms.append("elastic-coupling (off-diagonal)")
        if s["vt_changes"] > 20:
            mechanisms.append("contact-chattering")
        if s["mean_contact"] > 10 * s["mean_inertia"]:
            mechanisms.append("contact-force-dominance")
        if s["cancellation_mean"] > 10:
            mechanisms.append("high-cancellation")
        if not mechanisms:
            mechanisms.append("mild")
        print(f"  v{v} (L{s['layer']}): {', '.join(mechanisms)}")

    # Generate plots
    try:
        plot_results(data, unstable_verts, z_std, layer_ranges)
    except Exception as e:
        print(f"Plot generation failed: {e}")

    # Save text report
    report_path = os.path.join(OUTPUT_DIR, "tube_instability_report.txt")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
