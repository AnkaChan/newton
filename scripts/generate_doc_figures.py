#!/usr/bin/env python
"""Generate focused figures for the VBD instability design doc."""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "debug_tube")
NPZ_PATH = os.path.join(OUTPUT_DIR, "debug_tube_instability.npz")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LAYER_RANGES = [(0, 231), (231, 462)]

print(f"Loading {NPZ_PATH}...")
data = np.load(NPZ_PATH)
S = int(data["substeps_recorded"])
iters = int(data["iterations_per_substep"])
I = int(data["iterations_recorded"])
N = int(data["particle_count"])
positions = data["positions_history"]

# Steady-state z_std
start = S // 2
steady = positions[start:]
z_std = np.std(steady[:, :, 2], axis=0)
top10 = np.argsort(z_std)[-10:][::-1]

# Representative vertices
V_TOP = 454    # worst, layer 1, no contact
V_GROUND = 138 # layer 0, ground contact chattering
V_MILD = 453   # layer 1, pure elastic, zero contact

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# =========================================================================
# Figure 1: Z-position heatmap — all vertices over time
# =========================================================================
print("Fig 1: Z-position heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for li, (s, e) in enumerate(LAYER_RANGES):
    ax = axes[li]
    z_data = positions[:, s:e, 2].T  # [n_verts, substeps]
    im = ax.imshow(z_data, aspect="auto", cmap="RdYlBu_r",
                   extent=[0, S, e - 1 - s, 0], interpolation="nearest")
    ax.set_xlabel("Substep")
    ax.set_ylabel("Vertex index (within layer)")
    ax.set_title(f"Layer {li} — Z position over time")
    plt.colorbar(im, ax=ax, label="z (cm)")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_z_heatmap.png"))
plt.close()

# =========================================================================
# Figure 2: Z-position trajectories of top-5 unstable vertices
# =========================================================================
print("Fig 2: Z trajectories...")
fig, ax = plt.subplots(figsize=(12, 5))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, v in enumerate(top10[:5]):
    layer = 0 if v < 231 else 1
    z = positions[:, v, 2]
    ax.plot(np.arange(S), z, linewidth=0.8, color=colors[i],
            label=f"v{v} (L{layer}, std={z_std[v]:.4f})")
ax.set_xlabel("Substep")
ax.set_ylabel("Z position (cm)")
ax.set_title("Z-position trajectories — top 5 unstable vertices")
ax.legend()
ax.axvline(x=S // 2, color="gray", linestyle="--", alpha=0.5, label="steady-state cutoff")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_z_trajectories.png"))
plt.close()

# =========================================================================
# Figure 3: Force component breakdown — v454 (no contact) vs v138 (contact)
# =========================================================================
print("Fig 3: Force breakdown...")
last_iter_idx = np.arange(iters - 1, I, iters)[:S]
substep_range = np.arange(S)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax_i, (v, label) in enumerate([(V_TOP, f"v{V_TOP} (Layer 1, no contact)"),
                                     (V_GROUND, f"v{V_GROUND} (Layer 0, ground contact)")]):
    ax = axes[ax_i]
    fi_z = data["f_inertia"][last_iter_idx, v, 2]
    fe_z = data["f_elastic"][last_iter_idx, v, 2]
    fb_z = data["f_bending"][last_iter_idx, v, 2]
    fc_z = data["pre_solve_forces"][last_iter_idx, v, 2]
    fn_z = fi_z + fe_z + fb_z + fc_z

    ax.plot(substep_range, fi_z, label="Inertia Z", alpha=0.8, linewidth=0.7)
    ax.plot(substep_range, fe_z, label="Elastic Z", alpha=0.8, linewidth=0.7)
    ax.plot(substep_range, fb_z, label="Bending Z", alpha=0.8, linewidth=0.7)
    ax.plot(substep_range, fc_z, label="Contact Z", alpha=0.8, linewidth=0.7)
    ax.plot(substep_range, fn_z, label="Net Z", color="black", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("Force Z component")
    ax.set_title(label)
    ax.legend(loc="upper right")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)

axes[-1].set_xlabel("Substep")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_force_breakdown.png"))
plt.close()

# =========================================================================
# Figure 4: Per-iteration displacement convergence
# =========================================================================
print("Fig 4: Displacement convergence...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax_i, (v, label) in enumerate([
    (V_TOP, f"v{V_TOP} (worst, L1)"),
    (V_GROUND, f"v{V_GROUND} (ground contact, L0)"),
    (V_MILD, f"v{V_MILD} (elastic only, L1)"),
]):
    ax = axes[ax_i]
    # Plot displacement per iteration for many substeps
    sample = np.linspace(S // 2, S - 1, 30, dtype=int)  # steady-state only
    ratios = []
    for ss in sample:
        iter_start = ss * iters
        iter_end = iter_start + iters
        if iter_end > I:
            continue
        disp = data["displacements"][iter_start:iter_end, v, :]
        disp_mag = np.linalg.norm(disp, axis=1)
        ax.plot(range(iters), disp_mag, alpha=0.15, linewidth=0.8, color="steelblue")
        if disp_mag[0] > 1e-15:
            ratios.append(disp_mag[-1] / disp_mag[0])

    mean_ratio = np.mean(ratios) if ratios else 0
    ax.set_xlabel("VBD Iteration")
    ax.set_ylabel("|displacement|")
    ax.set_title(f"{label}\nmean ratio = {mean_ratio:.2f}")
    ax.set_xticks(range(iters))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_displacement_convergence.png"))
plt.close()

# =========================================================================
# Figure 5: Force evolution within a single substep (iteration by iteration)
# =========================================================================
print("Fig 5: Force evolution within substep...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_i, (v, label) in enumerate([(V_TOP, f"v{V_TOP}"), (V_GROUND, f"v{V_GROUND}")]):
    ax = axes[ax_i]
    # Find worst substep by velocity
    vel_mag = np.linalg.norm(data["velocities"][:, v, :], axis=1)
    worst_ss = np.argmax(vel_mag)

    iter_range = range(iters)
    fi_z, fe_z, fb_z, fc_z, fn_z = [], [], [], [], []
    for it in range(iters):
        gi = worst_ss * iters + it
        if gi >= I:
            break
        fi = data["f_inertia"][gi, v, 2]
        fe = data["f_elastic"][gi, v, 2]
        fb = data["f_bending"][gi, v, 2]
        fc = data["pre_solve_forces"][gi, v, 2]
        fi_z.append(fi)
        fe_z.append(fe)
        fb_z.append(fb)
        fc_z.append(fc)
        fn_z.append(fi + fe + fb + fc)

    x = np.arange(len(fi_z))
    width = 0.18
    ax.bar(x - 1.5 * width, fi_z, width, label="Inertia Z", alpha=0.8)
    ax.bar(x - 0.5 * width, fe_z, width, label="Elastic Z", alpha=0.8)
    ax.bar(x + 0.5 * width, fb_z, width, label="Bending Z", alpha=0.8)
    ax.bar(x + 1.5 * width, fc_z, width, label="Contact Z", alpha=0.8)
    ax.plot(x, fn_z, "ko-", markersize=5, linewidth=1.5, label="Net Z")
    ax.set_xlabel("VBD Iteration")
    ax.set_ylabel("Force Z")
    ax.set_title(f"{label} — worst substep {worst_ss}")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xticks(x)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_force_per_iteration.png"))
plt.close()

# =========================================================================
# Figure 6: VT contact count vs velocity for v138 (chattering vertex)
# =========================================================================
print("Fig 6: Contact chattering...")
fig, ax1 = plt.subplots(figsize=(12, 4))

v = V_GROUND
vt = data["vt_contact_counts"][:, v]
vel_mag = np.linalg.norm(data["velocities"][:, v, :], axis=1)

color1, color2 = "tab:red", "tab:blue"
ax1.bar(substep_range, vt, color=color1, alpha=0.6, width=1.0, label="VT contact count")
ax1.set_ylabel("VT contact count", color=color1)
ax1.set_xlabel("Substep")
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
ax2.plot(substep_range, vel_mag, color=color2, linewidth=0.6, alpha=0.8, label="|velocity|")
ax2.set_ylabel("|velocity| (cm/s)", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

ax1.set_title(f"v{v} (Layer 0) — Contact chattering vs velocity")
fig.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_contact_chattering.png"))
plt.close()

# =========================================================================
# Figure 7: Instability map — z_std per vertex, colored by layer
# =========================================================================
print("Fig 7: Instability map...")
fig, ax = plt.subplots(figsize=(12, 4))

for li, (s, e) in enumerate(LAYER_RANGES):
    verts = np.arange(s, e)
    ax.bar(verts, z_std[s:e], width=1.0, alpha=0.7, label=f"Layer {li}")

# Mark top-5
for v in top10[:5]:
    ax.annotate(f"v{v}", (v, z_std[v]), fontsize=8, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                xytext=(v, z_std[v] + 0.01))

ax.set_xlabel("Vertex index")
ax.set_ylabel("Z-position std (cm)")
ax.set_title("Instability map — steady-state Z oscillation per vertex")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig7_instability_map.png"))
plt.close()

# =========================================================================
# Figure 8: Cancellation ratio histogram
# =========================================================================
print("Fig 8: Cancellation ratio...")
last_iter_idx = np.arange(iters - 1, I, iters)[:S]

cancel_per_vert = []
for v in range(N):
    fi = np.linalg.norm(data["f_inertia"][last_iter_idx, v, :], axis=1)
    fe = np.linalg.norm(data["f_elastic"][last_iter_idx, v, :], axis=1)
    fb = np.linalg.norm(data["f_bending"][last_iter_idx, v, :], axis=1)
    fc = np.linalg.norm(data["pre_solve_forces"][last_iter_idx, v, :], axis=1)
    fn = np.linalg.norm(
        data["f_inertia"][last_iter_idx, v, :] +
        data["f_elastic"][last_iter_idx, v, :] +
        data["f_bending"][last_iter_idx, v, :] +
        data["pre_solve_forces"][last_iter_idx, v, :], axis=1)
    s = fi + fe + fb + fc
    c = np.where(fn > 1e-10, s / fn, 0.0)
    cancel_per_vert.append(c.mean())

cancel_per_vert = np.array(cancel_per_vert)

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(cancel_per_vert, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
ax.axvline(x=5, color="red", linestyle="--", label="threshold = 5")
ax.set_xlabel("Mean cancellation ratio")
ax.set_ylabel("Number of vertices")
ax.set_title("Force cancellation ratio distribution\n(sum of |components| / |net force|)")
ax.legend()
n_high = np.sum(cancel_per_vert > 5)
ax.annotate(f"{n_high}/{N} vertices > 5", xy=(0.7, 0.85), xycoords="axes fraction", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig8_cancellation_histogram.png"))
plt.close()

# =========================================================================
# Figure 9: Displacement ratio map (all vertices)
# =========================================================================
print("Fig 9: Displacement ratio map...")
disp_ratios = np.zeros(N)
for v in range(N):
    ratios = []
    for ss in range(S // 2, S):
        iter_start = ss * iters
        iter_end = iter_start + iters
        if iter_end > I:
            break
        disp = data["displacements"][iter_start:iter_end, v, :]
        disp_mag = np.linalg.norm(disp, axis=1)
        if disp_mag[0] > 1e-15:
            ratios.append(disp_mag[-1] / disp_mag[0])
    if ratios:
        disp_ratios[v] = np.mean(ratios)

fig, ax = plt.subplots(figsize=(12, 4))
for li, (s, e) in enumerate(LAYER_RANGES):
    verts = np.arange(s, e)
    ax.bar(verts, disp_ratios[s:e], width=1.0, alpha=0.7, label=f"Layer {li}")
ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="ratio = 1 (no convergence)")
ax.set_xlabel("Vertex index")
ax.set_ylabel("Mean displacement ratio (last/first iter)")
ax.set_title("VBD iteration convergence — displacement ratio per vertex\n(< 1 = converging, > 1 = diverging)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig9_displacement_ratio_map.png"))
plt.close()

print(f"\nAll figures saved to {FIG_DIR}/")
print("Files:", sorted(os.listdir(FIG_DIR)))
