#!/usr/bin/env python
"""Analyze saved tube instability data."""

from __future__ import annotations

import os
import sys

import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "debug_tube")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
NPZ_PATH = os.path.join(OUTPUT_DIR, "debug_tube_instability.npz")

TOP_K = 10
LAYER_RANGES = [(0, 231), (231, 462)]


def find_unstable_vertices(positions, top_k=TOP_K):
    n_substeps = positions.shape[0]
    start = n_substeps // 2
    steady = positions[start:]
    z = steady[:, :, 2]
    z_std = np.std(z, axis=0)
    top_indices = np.argsort(z_std)[-top_k:][::-1]
    return top_indices, z_std


def analyze_vertex(v, data, z_std):
    S = int(data["substeps_recorded"])
    iters = int(data["iterations_per_substep"])
    I = int(data["iterations_recorded"])

    layer = -1
    for li, (s, e) in enumerate(LAYER_RANGES):
        if s <= v < e:
            layer = li
            break

    print(f"\n{'='*70}")
    print(f"VERTEX {v} (layer {layer}, z_std={z_std[v]:.6f})")
    print(f"{'='*70}")

    pos = data["positions"][:, v, :]
    z = pos[:, 2]
    print(f"\n  Position Z: mean={z.mean():.4f}, std={z.std():.6f}, "
          f"range={np.ptp(z):.6f}, min={z.min():.4f}, max={z.max():.4f}")

    vel_start = data["velocities"][:, v, :]
    vel_mag = np.linalg.norm(vel_start, axis=1)
    print(f"  Velocity |v|: mean={vel_mag.mean():.4f}, max={vel_mag.max():.4f}")
    vz = vel_start[:, 2]
    sign_changes = np.sum(np.diff(np.sign(vz)) != 0)
    print(f"  Velocity Z sign changes: {sign_changes}/{S}")

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

    print(f"\n  Force Z components (last iter, mean over substeps):")
    print(f"    Inertia_Z:  mean={f_inertia[:, 2].mean():+.4f}, std={f_inertia[:, 2].std():.4f}")
    print(f"    Elastic_Z:  mean={f_elastic[:, 2].mean():+.4f}, std={f_elastic[:, 2].std():.4f}")
    print(f"    Bending_Z:  mean={f_bending[:, 2].mean():+.4f}, std={f_bending[:, 2].std():.4f}")
    print(f"    Contact_Z:  mean={f_contact[:, 2].mean():+.4f}, std={f_contact[:, 2].std():.4f}")

    # Per-iteration convergence
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

    # All-iteration force evolution for a single bad substep
    worst_substep = np.argmax(np.linalg.norm(data["velocities"][:, v, :], axis=1))
    print(f"\n  Force evolution at worst substep {worst_substep} (all {iters} iterations):")
    for it in range(iters):
        gi = worst_substep * iters + it
        if gi >= I:
            break
        fi = data["f_inertia"][gi, v, :]
        fe = data["f_elastic"][gi, v, :]
        fb = data["f_bending"][gi, v, :]
        fc = data["pre_solve_forces"][gi, v, :]
        fn = fi + fe + fb + fc
        print(f"    iter {it}: inertia_z={fi[2]:+9.2f}, elastic_z={fe[2]:+9.2f}, "
              f"bending_z={fb[2]:+9.2f}, contact_z={fc[2]:+9.2f}, net_z={fn[2]:+9.2f}")

    # Hessian
    H = data["pre_solve_hessians"][last_iter_indices, v, :, :]
    H_zz = H[:, 2, 2]
    H_trace = np.trace(H, axis1=1, axis2=2)
    print(f"\n  Hessian (pre-solve, last iter):")
    print(f"    H_zz:   mean={H_zz.mean():.2f}, max={H_zz.max():.2f}, min={H_zz.min():.2f}")
    print(f"    Trace:  mean={H_trace.mean():.2f}, max={H_trace.max():.2f}")
    eigvals = np.linalg.eigvalsh(H)
    max_eig = eigvals[:, -1]
    min_eig = eigvals[:, 0]
    cond = np.where(min_eig > 1e-10, max_eig / min_eig, float('inf'))
    finite_cond = cond[np.isfinite(cond)]
    if len(finite_cond) > 0:
        print(f"    Cond:   mean={np.mean(finite_cond):.2f}, max={np.max(finite_cond):.2f}")
    neg_count = np.sum(min_eig < 0)
    print(f"    Negative eigenvalues: {neg_count}/{S} substeps")

    # VT contact
    vt = data["vt_contact_counts"][:, v]
    vt_changes = int(np.sum(np.diff(vt > 0) != 0))
    print(f"\n  VT self-contact:")
    print(f"    Count: mean={vt.mean():.2f}, max={vt.max()}, min={vt.min()}")
    print(f"    State changes (on/off): {vt_changes}")
    if vt.max() > 0:
        vt_dist = data["vt_min_dist"][:, v]
        in_contact = vt > 0
        if in_contact.any():
            print(f"    Min dist when in contact: mean={vt_dist[in_contact].mean():.6f}, "
                  f"min={vt_dist[in_contact].min():.6f}")

    # Body (ground) contact — approximate from body_contact_count
    body_ct = data["body_contact_count"]
    print(f"\n  Body contact (global): mean={body_ct.mean():.1f}, max={body_ct.max()}")

    trunc = data["truncation_ts"][:, v]
    active = trunc < 1.0 - 1e-6
    print(f"  Truncation: active in {active.sum()}/{I} iterations")

    return {
        "vertex": v, "layer": layer, "z_std": z_std[v],
        "mean_elastic": mag_elastic.mean(), "mean_contact": mag_contact.mean(),
        "mean_inertia": mag_inertia.mean(), "cancellation_mean": cancellation.mean(),
        "vt_changes": vt_changes, "max_vel": vel_mag.max(),
    }


def plot_results(data, unstable_verts, z_std):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = int(data["substeps_recorded"])
    iters = int(data["iterations_per_substep"])
    I = int(data["iterations_recorded"])

    n_verts = min(5, len(unstable_verts))
    fig, axes = plt.subplots(n_verts, 4, figsize=(20, 4 * n_verts))
    if n_verts == 1:
        axes = axes[np.newaxis, :]

    for row, v in enumerate(unstable_verts[:n_verts]):
        layer = -1
        for li, (s, e) in enumerate(LAYER_RANGES):
            if s <= v < e:
                layer = li
                break

        last_iter_indices = np.arange(iters - 1, I, iters)[:S]
        substep_range = np.arange(S)

        ax = axes[row, 0]
        z = data["positions"][:, v, 2]
        ax.plot(substep_range, z, linewidth=0.5)
        ax.set_title(f"v{v} (L{layer}) Z position")
        ax.set_xlabel("substep")
        ax.set_ylabel("z (cm)")

        ax = axes[row, 1]
        ax.plot(substep_range, data["f_inertia"][last_iter_indices, v, 2], label="inertia", alpha=0.7, linewidth=0.5)
        ax.plot(substep_range, data["f_elastic"][last_iter_indices, v, 2], label="elastic", alpha=0.7, linewidth=0.5)
        ax.plot(substep_range, data["f_bending"][last_iter_indices, v, 2], label="bending", alpha=0.7, linewidth=0.5)
        ax.plot(substep_range, data["pre_solve_forces"][last_iter_indices, v, 2], label="contact", alpha=0.7, linewidth=0.5)
        ax.set_title(f"v{v} Force Z")
        ax.set_xlabel("substep")
        ax.legend(fontsize=6)

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
    print(f"Loading {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    S = int(data["substeps_recorded"])
    I = int(data["iterations_recorded"])
    N = int(data["particle_count"])
    print(f"  Substeps: {S}, Iterations: {I}, Particles: {N}")

    positions = data["positions_history"]
    unstable_verts, z_std = find_unstable_vertices(positions, TOP_K)

    print(f"\n{'='*70}")
    print(f"TOP-{TOP_K} UNSTABLE VERTICES")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Vtx':<8} {'Z_std':<12} {'Layer':<8}")
    for rank, v in enumerate(unstable_verts):
        layer = -1
        for li, (s, e) in enumerate(LAYER_RANGES):
            if s <= v < e:
                layer = li
                break
        print(f"{rank + 1:<6} {v:<8} {z_std[v]:<12.6f} {layer:<8}")

    summaries = []
    for v in unstable_verts:
        s = analyze_vertex(v, data, z_std)
        summaries.append(s)

    print(f"\n{'='*70}")
    print("INSTABILITY CLASSIFICATION")
    print(f"{'='*70}")
    for s in summaries:
        v = s["vertex"]
        mechanisms = []
        if s["mean_elastic"] > 10 * s["mean_inertia"] and s["vt_changes"] == 0:
            mechanisms.append("elastic-coupling")
        if s["vt_changes"] > 20:
            mechanisms.append("contact-chattering")
        if s["mean_contact"] > 10 * s["mean_inertia"]:
            mechanisms.append("contact-dominance")
        if s["cancellation_mean"] > 10:
            mechanisms.append("high-cancellation")
        if not mechanisms:
            mechanisms.append("mild")
        print(f"  v{v} (L{s['layer']}): {', '.join(mechanisms)}")

    try:
        plot_results(data, unstable_verts, z_std)
    except Exception as e:
        print(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()
