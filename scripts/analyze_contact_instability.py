#!/usr/bin/env python
"""Analyze recorded VBD debug data to identify and diagnose contact instability.

Loads the .npz output from diag_contact_instability.py and produces:
1. Identification of unstable vertices (velocity oscillation / growth)
2. Force component analysis for unstable vertices
3. Hessian conditioning analysis
4. Contact chattering detection
5. Float32 cancellation ratio check
"""

from __future__ import annotations

import os
import sys

import numpy as np

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "debug_contact_instability.npz"
)
DATA_PATH = os.path.abspath(DATA_PATH)


def load_data(path):
    print(f"Loading {path}...")
    d = dict(np.load(path, allow_pickle=True))
    print(f"  Substeps: {d['substeps_recorded']}")
    print(f"  Iterations: {d['iterations_recorded']}")
    print(f"  Particles: {d['particle_count']}")
    print(f"  Edges: {d['edge_count']}")
    print(f"  Triangles: {d['tri_count']}")
    return d


# ======================================================================
# Phase 1: Identify unstable vertices
# ======================================================================

def identify_unstable_vertices(d, top_n=20):
    """Find vertices with largest velocity oscillation or growth."""
    vel = d["velocities"]  # [S, N, 3]
    vel_end = d["velocities_end"]  # [S, N, 3]
    S, N, _ = vel.shape

    # Metric 1: max velocity magnitude across all substeps
    vel_mag = np.linalg.norm(vel_end, axis=2)  # [S, N]
    max_vel = np.max(vel_mag, axis=0)  # [N]

    # Metric 2: velocity growth (final vs initial substep magnitudes)
    # Use a sliding window to detect growth
    window = min(50, S // 2)
    vel_early = np.mean(vel_mag[:window], axis=0)  # [N]
    vel_late = np.mean(vel_mag[-window:], axis=0)  # [N]
    vel_growth = vel_late - vel_early  # [N]

    # Metric 3: velocity oscillation (sign changes in z-component)
    vel_z = vel_end[:, :, 2]  # [S, N]
    sign_changes = np.sum(np.diff(np.sign(vel_z), axis=0) != 0, axis=0)  # [N]

    # Metric 4: velocity variance (high variance = oscillation)
    vel_var = np.var(vel_mag, axis=0)  # [N]

    # Combined score: weighted combination
    # Normalize each metric to [0, 1]
    def norm01(x):
        mn, mx = np.min(x), np.max(x)
        return (x - mn) / (mx - mn + 1e-12)

    score = (
        0.3 * norm01(max_vel)
        + 0.3 * norm01(vel_growth)
        + 0.2 * norm01(sign_changes.astype(float))
        + 0.2 * norm01(vel_var)
    )

    top_idx = np.argsort(score)[-top_n:][::-1]

    print("\n" + "=" * 80)
    print("UNSTABLE VERTICES (top-20 by combined instability score)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Vtx':<6} {'Score':<8} {'MaxVel':<10} {'VelGrowth':<12} {'SignChg':<9} {'VelVar':<10}")
    print("-" * 60)
    for rank, idx in enumerate(top_idx):
        print(
            f"{rank+1:<5} {idx:<6} {score[idx]:<8.4f} {max_vel[idx]:<10.2f} "
            f"{vel_growth[idx]:<12.2f} {sign_changes[idx]:<9d} {vel_var[idx]:<10.2f}"
        )

    return top_idx, {
        "max_vel": max_vel,
        "vel_growth": vel_growth,
        "sign_changes": sign_changes,
        "vel_var": vel_var,
        "score": score,
    }


# ======================================================================
# Phase 2: Force component analysis
# ======================================================================

def analyze_forces(d, unstable_vtx):
    """Analyze force component magnitudes and ratios for unstable vertices."""
    f_inertia = d["f_inertia"]  # [I, N, 3]
    f_elastic = d["f_elastic"]  # [I, N, 3]
    f_bending = d["f_bending"]  # [I, N, 3]
    f_contact = d["pre_solve_forces"]  # [I, N, 3] (body + spring + self-contact)
    I_total, N, _ = f_inertia.shape
    iters_per_sub = d["iterations_per_substep"]

    print("\n" + "=" * 80)
    print("FORCE COMPONENT ANALYSIS (last iteration of each substep)")
    print("=" * 80)

    # Use last iteration of each substep for steady-state force balance
    # Iteration indices for last-iteration-of-substep
    last_iter_idx = np.arange(iters_per_sub - 1, I_total, iters_per_sub)

    for vtx in unstable_vtx[:10]:
        # Extract force vectors for this vertex at last iterations
        fi = f_inertia[last_iter_idx, vtx]  # [S, 3]
        fe = f_elastic[last_iter_idx, vtx]  # [S, 3]
        fb = f_bending[last_iter_idx, vtx]  # [S, 3]
        fc = f_contact[last_iter_idx, vtx]  # [S, 3]

        fi_mag = np.linalg.norm(fi, axis=1)
        fe_mag = np.linalg.norm(fe, axis=1)
        fb_mag = np.linalg.norm(fb, axis=1)
        fc_mag = np.linalg.norm(fc, axis=1)

        # Total force and cancellation
        f_total = fi + fe + fb + fc
        ft_mag = np.linalg.norm(f_total, axis=1)
        sum_components = fi_mag + fe_mag + fb_mag + fc_mag
        cancellation_ratio = sum_components / (ft_mag + 1e-12)

        # Scale ratio (max / min of nonzero components)
        all_mags = np.stack([fi_mag, fe_mag, fb_mag, fc_mag], axis=1)
        nonzero_mask = all_mags > 1e-10
        # Per-substep scale ratio
        scale_ratios = []
        for s in range(len(last_iter_idx)):
            nonzero = all_mags[s][nonzero_mask[s]]
            if len(nonzero) >= 2:
                scale_ratios.append(np.max(nonzero) / np.min(nonzero))
            else:
                scale_ratios.append(1.0)
        scale_ratios = np.array(scale_ratios)

        print(f"\n--- Vertex {vtx} ---")
        print(f"  {'Component':<12} {'Mean|f|':<12} {'Max|f|':<12} {'Std|f|':<12}")
        print(f"  {'Inertia':<12} {np.mean(fi_mag):<12.4f} {np.max(fi_mag):<12.4f} {np.std(fi_mag):<12.4f}")
        print(f"  {'Elastic':<12} {np.mean(fe_mag):<12.4f} {np.max(fe_mag):<12.4f} {np.std(fe_mag):<12.4f}")
        print(f"  {'Bending':<12} {np.mean(fb_mag):<12.4f} {np.max(fb_mag):<12.4f} {np.std(fb_mag):<12.4f}")
        print(f"  {'Contact':<12} {np.mean(fc_mag):<12.4f} {np.max(fc_mag):<12.4f} {np.std(fc_mag):<12.4f}")
        print(f"  {'Net':<12} {np.mean(ft_mag):<12.4f} {np.max(ft_mag):<12.4f} {np.std(ft_mag):<12.4f}")
        print(f"  Cancellation ratio: mean={np.mean(cancellation_ratio):.2f}, max={np.max(cancellation_ratio):.2f}")
        print(f"  Scale ratio: mean={np.mean(scale_ratios):.2f}, max={np.max(scale_ratios):.2f}")
        if np.max(scale_ratios) > 1e5:
            print(f"  *** WARNING: Scale ratio > 1e5 — float32 precision suspect! ***")
        if np.max(cancellation_ratio) > 100:
            print(f"  *** WARNING: High cancellation — large forces nearly cancel ***")


# ======================================================================
# Phase 3: Hessian conditioning analysis
# ======================================================================

def analyze_hessians(d, unstable_vtx):
    """Analyze per-vertex hessian eigenvalues and condition numbers."""
    hessians = d["pre_solve_hessians"]  # [I, N, 3, 3]
    I_total, N, _, _ = hessians.shape
    iters_per_sub = int(d["iterations_per_substep"])
    last_iter_idx = np.arange(iters_per_sub - 1, I_total, iters_per_sub)

    print("\n" + "=" * 80)
    print("HESSIAN CONDITIONING (pre-solve hessians = contact + spring component)")
    print("=" * 80)

    for vtx in unstable_vtx[:10]:
        H = hessians[last_iter_idx, vtx]  # [S, 3, 3]
        # Eigenvalues per substep
        eigvals = np.linalg.eigvalsh(H)  # [S, 3]
        # Condition number = max/min eigenvalue
        min_eig = np.min(np.abs(eigvals), axis=1)
        max_eig = np.max(np.abs(eigvals), axis=1)
        cond = max_eig / (min_eig + 1e-12)

        # Check for negative eigenvalues (non-PD hessian)
        has_negative = np.any(eigvals < -1e-8, axis=1)
        neg_count = np.sum(has_negative)

        # Frobenius norm
        h_norm = np.linalg.norm(H.reshape(-1, 9), axis=1)

        print(f"\n--- Vertex {vtx} ---")
        print(f"  |H|_F: mean={np.mean(h_norm):.4f}, max={np.max(h_norm):.4f}")
        print(f"  Eigenvalues: min={np.min(eigvals):.6f}, max={np.max(eigvals):.6f}")
        print(f"  Condition number: mean={np.mean(cond):.2f}, max={np.max(cond):.2f}")
        print(f"  Substeps with negative eigenvalues: {neg_count}/{len(last_iter_idx)}")
        if np.max(cond) > 1e6:
            print(f"  *** WARNING: Hessian very ill-conditioned (cond > 1e6) ***")
        if neg_count > 0:
            print(f"  *** WARNING: Non-positive-definite hessian detected ***")


# ======================================================================
# Phase 4: Contact chattering detection
# ======================================================================

def analyze_contact_chattering(d, unstable_vtx):
    """Check if unstable vertices have flickering contact states."""
    vt_counts = d["vt_contact_counts"]  # [S, N]
    S, N = vt_counts.shape

    print("\n" + "=" * 80)
    print("CONTACT CHATTERING (VT contact count changes over substeps)")
    print("=" * 80)

    for vtx in unstable_vtx[:10]:
        cts = vt_counts[:, vtx]
        # Contact state changes (on→off or off→on)
        has_contact = cts > 0
        state_changes = np.sum(np.diff(has_contact.astype(int)) != 0)

        # Contact count variance
        ct_var = np.var(cts)
        ct_mean = np.mean(cts)
        ct_max = np.max(cts)

        print(f"\n--- Vertex {vtx} ---")
        print(f"  Contact count: mean={ct_mean:.2f}, max={ct_max}, var={ct_var:.2f}")
        print(f"  Contact state changes (on/off): {state_changes}")
        print(f"  Always in contact: {np.all(has_contact)}")
        print(f"  Never in contact: {np.all(~has_contact)}")
        if state_changes > S * 0.1:
            print(f"  *** WARNING: Contact chattering — {state_changes} state changes in {S} substeps ***")


# ======================================================================
# Phase 5: Displacement and truncation analysis
# ======================================================================

def analyze_displacements(d, unstable_vtx):
    """Analyze displacement magnitudes and truncation activity."""
    disp = d["displacements"]  # [I, N, 3]
    trunc = d["truncation_ts"]  # [I, N]
    I_total, N, _ = disp.shape
    iters_per_sub = int(d["iterations_per_substep"])

    print("\n" + "=" * 80)
    print("DISPLACEMENT & TRUNCATION ANALYSIS")
    print("=" * 80)

    for vtx in unstable_vtx[:10]:
        d_vtx = disp[:, vtx]  # [I, 3]
        t_vtx = trunc[:, vtx]  # [I]

        d_mag = np.linalg.norm(d_vtx, axis=1)

        # Per-substep: check convergence across iterations
        d_per_substep = d_mag.reshape(-1, iters_per_sub)  # [S, iters]
        # Convergence: ratio of last to first iteration displacement
        convergence_ratio = d_per_substep[:, -1] / (d_per_substep[:, 0] + 1e-12)

        # Truncation activity
        t_per_substep = t_vtx.reshape(-1, iters_per_sub)
        truncated = t_per_substep < 0.999
        truncated_count = np.sum(truncated)
        truncated_substeps = np.any(truncated, axis=1).sum()

        # Displacement sign alternation (oscillation in z)
        dz = d_vtx[:, 2]
        dz_sign_changes = np.sum(np.diff(np.sign(dz)) != 0)

        print(f"\n--- Vertex {vtx} ---")
        print(f"  |displacement|: mean={np.mean(d_mag):.6f}, max={np.max(d_mag):.6f}")
        print(f"  Convergence ratio (last/first iter): mean={np.mean(convergence_ratio):.4f}, max={np.max(convergence_ratio):.4f}")
        print(f"  Truncation active: {truncated_count}/{I_total} iterations, {truncated_substeps}/{I_total//iters_per_sub} substeps")
        if truncated_count > 0:
            print(f"  Min truncation_t: {np.min(t_vtx[t_vtx < 0.999]):.6f}")
        print(f"  Displacement z sign changes: {dz_sign_changes}")
        if np.mean(convergence_ratio) > 0.8:
            print(f"  *** WARNING: Poor convergence — displacement not decreasing across iterations ***")
        if truncated_substeps > I_total // iters_per_sub * 0.3:
            print(f"  *** WARNING: Truncation very active — may be causing residual oscillation ***")


# ======================================================================
# Phase 6: Body contact + self-contact interference
# ======================================================================

def analyze_body_contact_interference(d, unstable_vtx):
    """Check if unstable vertices have both body and self-contacts."""
    vt_counts = d["vt_contact_counts"]  # [S, N]
    body_ct = d["body_contact_count"]  # [S]
    S, N = vt_counts.shape

    print("\n" + "=" * 80)
    print("BODY CONTACT COUNT PER SUBSTEP")
    print("=" * 80)
    print(f"  Mean: {np.mean(body_ct):.1f}, Max: {np.max(body_ct)}, Min: {np.min(body_ct)}")
    print(f"  Substeps with body contacts: {np.sum(body_ct > 0)}/{S}")


# ======================================================================
# Phase 7: Velocity trajectory for unstable vertices
# ======================================================================

def print_velocity_trajectory(d, unstable_vtx):
    """Print velocity magnitude trajectory for top unstable vertices."""
    vel_end = d["velocities_end"]  # [S, N, 3]
    S, N, _ = vel_end.shape

    print("\n" + "=" * 80)
    print("VELOCITY TRAJECTORY (every 10 substeps, top-5 unstable vertices)")
    print("=" * 80)

    step_indices = list(range(0, S, 10))
    header = f"{'Substep':<10}" + "".join(f"{'v' + str(vtx):<14}" for vtx in unstable_vtx[:5])
    print(header)
    print("-" * len(header))
    for s in step_indices:
        row = f"{s:<10}"
        for vtx in unstable_vtx[:5]:
            vm = np.linalg.norm(vel_end[s, vtx])
            row += f"{vm:<14.4f}"
        print(row)


# ======================================================================
# Phase 8: Global statistics
# ======================================================================

def global_statistics(d):
    """Print global simulation statistics."""
    vel_end = d["velocities_end"]  # [S, N, 3]
    S, N, _ = vel_end.shape
    vel_mag = np.linalg.norm(vel_end, axis=2)  # [S, N]

    print("\n" + "=" * 80)
    print("GLOBAL STATISTICS")
    print("=" * 80)
    print(f"  Particles: {N}, Substeps: {S}")
    print(f"  Max velocity across all particles/substeps: {np.max(vel_mag):.4f}")
    print(f"  Mean velocity (last substep): {np.mean(vel_mag[-1]):.4f}")
    print(f"  Particles with max|vel| > 50: {np.sum(np.max(vel_mag, axis=0) > 50)}")
    print(f"  Particles with max|vel| > 100: {np.sum(np.max(vel_mag, axis=0) > 100)}")
    print(f"  Particles with max|vel| > 200: {np.sum(np.max(vel_mag, axis=0) > 200)}")

    # Velocity growth: compare first 50 vs last 50 substeps
    early = np.mean(vel_mag[:50], axis=0)
    late = np.mean(vel_mag[-50:], axis=0)
    growing = np.sum(late > early * 2)
    print(f"  Particles with velocity doubling (late vs early): {growing}")


# ======================================================================
# Main
# ======================================================================

def main():
    d = load_data(DATA_PATH)
    global_statistics(d)
    top_vtx, metrics = identify_unstable_vertices(d, top_n=20)
    print_velocity_trajectory(d, top_vtx)
    analyze_forces(d, top_vtx)
    analyze_hessians(d, top_vtx)
    analyze_contact_chattering(d, top_vtx)
    analyze_displacements(d, top_vtx)
    analyze_body_contact_interference(d, top_vtx)

    # Save analysis results
    out_path = DATA_PATH.replace(".npz", "_analysis.txt")
    print(f"\nAnalysis complete. Re-run with redirect to save: python {__file__} > {out_path}")


if __name__ == "__main__":
    main()
