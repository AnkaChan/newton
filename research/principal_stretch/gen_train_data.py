# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Generate diverse force-driven trajectories for self-supervised training.

For each trajectory:
  - Same 8x4x4 hanging soft grid (fix_left=True).
  - Random rotation applied to gravity (norm preserved at 9.8 m/s^2).
  - Random body force applied uniformly to all unpinned tets, sampled
    fresh each trajectory and held constant.
  - Optionally a single point poke on a random unpinned vertex.

Records (x, F_ext) per frame and dumps to data/train.npz with the same
topology arrays used by the recovery solver.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .recorder import StretchRecorder
from .run_forward import build_model, find_pinned_indices


def random_unit_vector(rng):
    v = rng.standard_normal(3)
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trajs", type=int, default=200)
    parser.add_argument("--frames-per-traj", type=int, default=20)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wp.init()
    rng = np.random.default_rng(args.seed)

    fps = 60
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / args.substeps

    # Topology is identical across trajectories; build once for shape info.
    model, builder = build_model()
    pinned = find_pinned_indices(builder)
    rest_q = np.asarray(builder.particle_q, dtype=np.float32)
    tet_indices_np = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)
    tet_poses_np = model.tet_poses.numpy().reshape(-1, 3, 3).astype(np.float32)
    particle_mass = np.asarray(builder.particle_mass, dtype=np.float32)

    # Per-tet material (homogeneous soft_grid -> broadcast).
    tet_materials = model.tet_materials.numpy().reshape(-1, 3)  # [k_mu, k_lambda, k_damp]
    mu_per_tet = tet_materials[:, 0].astype(np.float32)
    lam_per_tet = tet_materials[:, 1].astype(np.float32)

    n_verts = rest_q.shape[0]
    n_tets = tet_indices_np.shape[0]
    n_frames_per = args.frames_per_traj
    n_total = args.num_trajs * n_frames_per

    x_log = np.zeros((n_total, n_verts, 3), dtype=np.float32)
    f_ext_log = np.zeros((n_total, n_verts, 3), dtype=np.float32)
    F_log = np.zeros((n_total, n_tets, 3, 3), dtype=np.float32)
    S_log = np.zeros((n_total, n_tets, 3, 3), dtype=np.float32)
    traj_start = np.zeros(args.num_trajs, dtype=np.int32)

    g_mag = 9.8
    t0 = time.time()
    write_idx = 0
    for traj in range(args.num_trajs):
        traj_start[traj] = write_idx

        # Rebuild the model so we can change gravity (set on model.gravity).
        model_t, builder_t = build_model()
        solver = SolverVBD(model=model_t, iterations=args.iterations,
                           particle_enable_self_contact=False,
                           particle_enable_tile_solve=False)

        # Keep gravity at the default (Newton: -9.81 along Z). Variation comes
        # from body force + poke. Recording per-traj gravity to support varying
        # it can be added once basic training works.
        gravity_vec = np.asarray(model_t.gravity.numpy()[0], dtype=np.float32)

        # Random body force (per-vertex constant), magnitude in [0, 30] N total /n_verts.
        body_f_mag = rng.uniform(0.0, 25.0)
        body_f_dir = random_unit_vector(rng)
        body_force = (body_f_dir * body_f_mag).astype(np.float32)

        # Random point poke: choose a non-pinned vertex, magnitude up to 50 N, applied for first half.
        unpinned = np.where(particle_mass > 0)[0]
        poke_vert = int(rng.choice(unpinned))
        poke_force = (random_unit_vector(rng) * rng.uniform(0.0, 50.0)).astype(np.float32)
        poke_end_frame = rng.integers(2, n_frames_per // 2 + 1)

        state_0 = model_t.state()
        state_1 = model_t.state()
        control = model_t.control()
        contacts = model_t.contacts()
        recorder = StretchRecorder(model_t)

        f_ext_np = np.zeros((n_verts, 3), dtype=np.float32)
        f_ext_np[:] = body_force[None, :]  # body force on all (including pinned, but they have mass=0 so don't move)

        try:
            for f in range(n_frames_per):
                # Compose f_ext for this frame.
                f_ext_frame = f_ext_np.copy()
                if f < poke_end_frame:
                    f_ext_frame[poke_vert] += poke_force

                f_ext_wp = wp.array(f_ext_frame, dtype=wp.vec3, device=model_t.device)

                for _ in range(args.substeps):
                    state_0.clear_forces()
                    # Add external force.
                    state_0.particle_f.assign(f_ext_frame)
                    model_t.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    state_0, state_1 = state_1, state_0

                snap = recorder.capture(state_0)
                x_log[write_idx] = snap["x"]
                f_ext_log[write_idx] = f_ext_frame
                F_log[write_idx] = snap["F"].reshape(-1, 3, 3)
                S_log[write_idx] = snap["S"].reshape(-1, 3, 3)
                write_idx += 1
        except Exception as e:
            print(f"  traj {traj} failed at frame {f}: {e}; truncating")
            # Truncate to actual frames captured.
            write_idx = traj_start[traj]
            continue

        if traj % 10 == 0:
            elapsed = time.time() - t0
            print(f"  traj {traj+1}/{args.num_trajs}  elapsed={elapsed:.1f}s  written={write_idx}")

    print(f"total wrote {write_idx} frames across {args.num_trajs} trajectories in {time.time()-t0:.1f}s")

    # Trim arrays
    x_log = x_log[:write_idx]
    f_ext_log = f_ext_log[:write_idx]
    F_log = F_log[:write_idx]
    S_log = S_log[:write_idx]

    # Default Newton gravity at the time of recording (constant across trajectories).
    default_g = np.asarray(model.gravity.numpy()[0], dtype=np.float32)

    np.savez_compressed(
        out_path,
        rest_q=rest_q,
        tet_indices=tet_indices_np,
        tet_poses=tet_poses_np,
        pinned_indices=pinned,
        particle_mass=particle_mass,
        mu_per_tet=mu_per_tet,
        lam_per_tet=lam_per_tet,
        gravity=default_g,
        x=x_log,
        f_ext=f_ext_log,
        F=F_log,
        S=S_log,
        traj_start=traj_start,
        frames_per_traj=n_frames_per,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
