# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Forward FEM run that records per-tet (x, F, R, S) for the recovery test.

Single hanging soft tet grid, VBD/StVK, 120 frames. Dumps a single .npz with
stacked arrays across all frames plus the topology (tet_indices, tet_poses,
rest_positions, pinned_indices).
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


def build_model(dim_x=8, dim_y=4, dim_z=4, cell=0.1, k_mu=1.0e5, k_lambda=1.0e5, k_damp=1e-3):
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 1.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_x,
        dim_y=dim_y,
        dim_z=dim_z,
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        density=1.0e3,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=k_damp,
        fix_left=True,
    )
    builder.color()
    model = builder.finalize()
    return model, builder


def find_pinned_indices(builder) -> np.ndarray:
    # add_soft_grid pins via mass = 0 (kinematic).
    mass = np.asarray(builder.particle_mass, dtype=np.float64)
    pinned = np.where(mass == 0.0)[0]
    return pinned.astype(np.int64)


def sanity_check(frame_idx, F, R, S, max_F_err=1e-4, max_sym_err=1e-5):
    F = F.reshape(-1, 3, 3)
    R = R.reshape(-1, 3, 3)
    S = S.reshape(-1, 3, 3)
    det = np.linalg.det(F)
    assert (det > 0).all(), f"frame {frame_idx}: {int((det <= 0).sum())} inverted tets"
    F_hat = np.einsum("eij,ejk->eik", R, S)
    err = np.linalg.norm(F - F_hat, axis=(1, 2)).max()
    assert err < max_F_err, f"frame {frame_idx}: F!=R*S round-trip err={err:.3e}"
    sym = np.linalg.norm(S - np.transpose(S, (0, 2, 1)), axis=(1, 2)).max()
    assert sym < max_sym_err, f"frame {frame_idx}: S not symmetric, err={sym:.3e}"
    R_orth = np.einsum("eij,ekj->eik", R, R)
    eye = np.broadcast_to(np.eye(3), R.shape)
    orth_err = np.linalg.norm(R_orth - eye, axis=(1, 2)).max()
    assert orth_err < 1e-4, f"frame {frame_idx}: R not orthogonal, err={orth_err:.3e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dim-x", type=int, default=8)
    parser.add_argument("--dim-y", type=int, default=4)
    parser.add_argument("--dim-z", type=int, default=4)
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            pathlib.Path(__file__).resolve().parents[2] / "research_data" / "principal_stretch" / "forward_run.npz"
        ),
    )
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wp.init()

    model, builder = build_model(dim_x=args.dim_x, dim_y=args.dim_y, dim_z=args.dim_z)
    pinned = find_pinned_indices(builder)
    rest_q = np.asarray(builder.particle_q, dtype=np.float32)
    tet_indices_np = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)
    tet_poses_np = model.tet_poses.numpy().reshape(-1, 3, 3).astype(np.float32)

    solver = SolverVBD(
        model=model,
        iterations=args.iterations,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    recorder = StretchRecorder(model)

    fps = 60
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / args.substeps

    n_frames = args.num_frames
    n_tets = model.tet_count
    n_verts = rest_q.shape[0]

    x_log = np.zeros((n_frames, n_verts, 3), dtype=np.float32)
    F_log = np.zeros((n_frames, n_tets, 3, 3), dtype=np.float32)
    R_log = np.zeros((n_frames, n_tets, 3, 3), dtype=np.float32)
    S_log = np.zeros((n_frames, n_tets, 3, 3), dtype=np.float32)

    t0 = time.time()
    for f in range(n_frames):
        for _ in range(args.substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
        snap = recorder.capture(state_0)
        x_log[f] = snap["x"]
        F_log[f] = snap["F"].reshape(-1, 3, 3)
        R_log[f] = snap["R"].reshape(-1, 3, 3)
        S_log[f] = snap["S"].reshape(-1, 3, 3)
        sanity_check(f, F_log[f], R_log[f], S_log[f])
        if f % 10 == 0:
            print(
                f"  frame {f:3d}/{n_frames}  det(F) range=[{np.linalg.det(F_log[f]).min():.3f}, {np.linalg.det(F_log[f]).max():.3f}]"
            )

    dt = time.time() - t0
    print(f"forward run done in {dt:.1f}s ({n_frames} frames, {n_verts} verts, {n_tets} tets)")
    print(f"writing {out_path}")

    np.savez_compressed(
        out_path,
        rest_q=rest_q,
        tet_indices=tet_indices_np,
        tet_poses=tet_poses_np,
        pinned_indices=pinned,
        x=x_log,
        F=F_log,
        R=R_log,
        S=S_log,
    )
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
