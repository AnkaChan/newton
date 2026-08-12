# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Train the stretch predictor through the batched differentiable decoder.

Supersedes the original per-sample-loop trainer and folds in the repairs from
the 2026-07-28 takeover review:

- **Batched decoder** — one ``solve()`` call per rollout step for the whole
  batch (the mesh, and therefore the Cholesky factor, is shared).
- **Inertial warm start** (``--warm inertial``) — decode from ``2 x_t - x_prev``
  instead of ``x_t``.  Local-global converges at ~0.98/iter, so the warm start
  dominates the output; measured 10x on the decoder floor.
- **Residual parameterisation** (``--residual``) — ``S* = S_t + delta`` instead
  of ``S* = I + delta``, so the zero-initialised network starts at "stretch
  unchanged" rather than "rest shape".
- **Position supervision** (``--loss pos``) — mass-weighted squared error
  against the reference trajectory.  The self-supervised incremental potential
  (``--loss phys``) is retained as an ablation; note it is one backward-Euler
  step at ``--dt`` while the reference data is 10 substeps at ``dt/10``, which
  caps its fidelity (review §2.3).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

from . import torch_solver as ts
from .graph_transformer import GraphTransformerConfig
from .potentials import incremental_potential_batched
from .predictor import PREDICTOR_KINDS, build_stretch_predictor
from .torch_solver import compute_S_from_x, inertial_predictor


def vert_to_tet_pin_flag(pinned: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pin_set = {int(v) for v in pinned}
    flag = np.zeros(tets.shape[0], dtype=np.float32)
    for t in range(tets.shape[0]):
        if any(int(tets[t, k]) in pin_set for k in range(4)):
            flag[t] = 1.0
    return flag


def build_windows(traj_start: np.ndarray, n_total: int, k_max: int) -> np.ndarray:
    """(i_prev, i_t, room) triples; room = GT frames available after i_t."""
    windows = []
    n_traj = traj_start.size
    for traj in range(n_traj):
        s = int(traj_start[traj])
        e = int(traj_start[traj + 1]) if traj + 1 < n_traj else n_total
        for t_off in range(1, (e - s) - 1):
            i_t = s + t_off
            room = (e - 1) - i_t
            if room >= 1:
                windows.append((i_t - 1, i_t, min(room, k_max)))
    return np.array(windows, dtype=np.int64)


def select_training_windows(windows: np.ndarray, limit: int) -> np.ndarray:
    """Select a deterministic, dataset-spanning subset for overfit studies."""
    if limit <= 0 or limit >= len(windows):
        return windows
    # Cover the complete trajectory corpus rather than taking the first few
    # temporally adjacent windows.  Integer arithmetic keeps the choice stable
    # across NumPy versions and produces unique indices while limit < N.
    index = np.arange(limit, dtype=np.int64) * len(windows) // limit
    return windows[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--solver-iters", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-rollout", type=int, default=4)
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=0,
        help="train on a deterministic dataset-spanning subset (0 uses every window)",
    )
    parser.add_argument("--curriculum-frac", type=float, default=0.5)
    parser.add_argument("--init-ckpt", type=str, default=None)
    parser.add_argument("--loss", choices=("pos", "phys"), default="pos")
    parser.add_argument("--predictor", choices=PREDICTOR_KINDS, default="mlp")
    parser.add_argument("--residual", action="store_true", help="predict S* = S_t + delta instead of S* = I + delta")
    parser.add_argument(
        "--warm", choices=("inertial", "prev"), default="inertial", help="decoder warm start: inertial predictor or x_t"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="MGN-style input noise: perturb (x_prev, x_t) at window start "
        "so the model learns to contract off-manifold states",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=1,
        help="alternating network<->decoder blocks per step; decoder iterations are split evenly across blocks",
    )
    parser.add_argument(
        "--phys-weight",
        type=float,
        default=0.0,
        help="with --loss pos: add this weight of the incremental potential as an off-manifold regulariser",
    )
    parser.add_argument("--gt-hidden", type=int, default=64, help="graph-transformer hidden width")
    parser.add_argument("--gt-heads", type=int, default=4, help="graph-transformer attention heads")
    parser.add_argument("--gt-levels", type=int, default=5, help="maximum topology-coarsening levels")
    parser.add_argument("--gt-cluster-size", type=int, default=8, help="target children per coarse node")
    parser.add_argument("--gt-dropout", type=float, default=0.0)
    parser.add_argument("--gt-max-delta", type=float, default=0.35, help="maximum Hencky update Frobenius norm")
    args = parser.parse_args()

    if args.predictor == "graph-transformer":
        if args.blocks != 1:
            raise ValueError("the multiresolution graph transformer already has global context; use --blocks 1")
        # Its output is always exp(log(U_t) + delta_H), independent of the
        # legacy flat predictor's absolute/residual switch.
        args.residual = True

    # The seed is part of the experiment contract: it controls initialization,
    # input noise, dropout, and window sampling.  ``manual_seed`` covers both
    # CPU and CUDA generators; the explicit CUDA call also covers later-created
    # devices in multi-GPU experiment runners.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32  # network dtype; decoder runs fp64

    data = np.load(args.train)
    rest_q = data["rest_q"]
    tets_np = data["tet_indices"]
    x_all = data["x"].astype(np.float64)
    n_total = x_all.shape[0]

    solver = ts.build_solver(
        rest_q, tets_np, data["tet_poses"], data["pinned_indices"], device=device, dtype=torch.float64
    )
    pin_flag = torch.as_tensor(vert_to_tet_pin_flag(data["pinned_indices"], tets_np), dtype=dtype, device=device)
    mass = torch.as_tensor(data["particle_mass"], dtype=torch.float64, device=device)
    mu32 = torch.as_tensor(data["mu_per_tet"], dtype=dtype, device=device)
    lam32 = torch.as_tensor(data["lam_per_tet"], dtype=dtype, device=device)
    mu64, lam64 = mu32.double(), lam32.double()
    volume = solver.w.double()
    gravity64 = torch.as_tensor(data["gravity"], dtype=torch.float64, device=device)
    pinned_targets = torch.as_tensor(rest_q[data["pinned_indices"]], dtype=torch.float64, device=device)

    windows = build_windows(data["traj_start"], n_total, args.max_rollout)
    windows = select_training_windows(windows, args.limit_windows)
    if args.batch > len(windows):
        raise ValueError(f"batch size {args.batch} exceeds the {len(windows)} selected training windows")
    print(f"{len(windows)} training windows, K_max={args.max_rollout}, batch={args.batch}")

    x_gpu = torch.as_tensor(x_all, dtype=torch.float64, device=device)
    f_ext_gpu = torch.as_tensor(data["f_ext"], dtype=torch.float64, device=device)

    graph_config = GraphTransformerConfig(
        hidden_dim=args.gt_hidden,
        num_heads=args.gt_heads,
        n_levels=args.gt_levels,
        cluster_size=args.gt_cluster_size,
        dropout=args.gt_dropout,
        max_hencky_update=args.gt_max_delta,
        dt=args.dt,
    )
    predictor = build_stretch_predictor(
        args.predictor,
        rest_q,
        tets_np,
        device,
        dtype,
        residual=args.residual,
        graph_config=graph_config,
    )
    print(f"predictor={predictor.kind} config={predictor.checkpoint_config()}")
    parameter_count = sum(parameter.numel() for parameter in predictor.parameters())
    print(f"trainable parameters: {parameter_count:,}")
    if predictor.kind == "graph-transformer":
        level_sizes = [
            predictor.model._level_buffer("adjacency", level).shape[0] for level in range(predictor.model.n_levels + 1)
        ]
        print(f"topology hierarchy: {' -> '.join(str(size) for size in level_sizes)} nodes")
    # The graph transformer reconstructs Hencky strain from positions.  Avoid
    # materializing the unused full-trajectory stretch tensor on the GPU
    # (about 4.6 GiB for the 4k training set).
    S_gpu = (
        None
        if predictor.kind == "graph-transformer"
        else torch.as_tensor(data["S"], dtype=torch.float64, device=device)
    )
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        predictor.model.load_state_dict(ckpt["state_dict"])
        print(f"loaded init weights from {args.init_ckpt}")
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=1e-5)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rng = np.random.default_rng(args.seed)
    pin_b = pinned_targets[None].expand(args.batch, -1, -1)
    curriculum_end = max(1, int(args.steps * args.curriculum_frac))

    log = []
    t0 = time.time()
    for step in range(args.steps):
        if step < curriculum_end:
            k_target = 1 + int((args.max_rollout - 1) * step / curriculum_end)
        else:
            k_target = args.max_rollout

        # Prefer windows with enough room for the full target rollout.
        ok = np.where(windows[:, 2] >= k_target)[0]
        if ok.size >= args.batch:
            idx = rng.choice(ok, size=args.batch, replace=False)
            k_roll = k_target
        else:
            idx = rng.choice(len(windows), size=args.batch, replace=False)
            k_roll = int(windows[idx, 2].min())
        b = windows[idx]
        i_t0 = torch.as_tensor(b[:, 1], dtype=torch.int64, device=device)

        x_prev = x_gpu[b[:, 0]]  # (B, V, 3)
        x_t = x_gpu[b[:, 1]]
        if args.noise_std > 0.0:
            # Perturb the input state (targets stay clean) so training visits the
            # off-manifold states rollout inevitably produces.
            x_prev = x_prev + args.noise_std * torch.randn_like(x_prev)
            x_t = x_t + args.noise_std * torch.randn_like(x_t)
            x_prev = x_prev.clone()
            x_t = x_t.clone()
            x_prev[:, solver.pinned] = pinned_targets
            x_t[:, solver.pinned] = pinned_targets
            S_prev = compute_S_from_x(solver, x_prev)
            S_now = compute_S_from_x(solver, x_t)
        else:
            if S_gpu is None:
                S_prev = compute_S_from_x(solver, x_prev)
                S_now = compute_S_from_x(solver, x_t)
            else:
                S_prev = S_gpu[b[:, 0]]
                S_now = S_gpu[b[:, 1]]

        opt.zero_grad()
        loss_total = torch.zeros((), dtype=torch.float64, device=device)
        x_pred = None
        for k in range(k_roll):
            f_ext = f_ext_gpu[i_t0 + k]  # (B, V, 3)

            if args.warm == "inertial":
                x0 = inertial_predictor(solver, x_t, x_prev, pin_b)
            else:
                x0 = x_t
            # Alternating network <-> decoder blocks (PoissonNet-style).  Each
            # block's global solve propagates the previous block's local
            # prediction across the whole mesh, so B blocks give the *network*
            # B global hops of receptive field at matched total decoder cost.
            iters_per_block = max(1, args.solver_iters // args.blocks)
            x_next = x0
            S_cur = S_now
            for _b in range(args.blocks):
                S_star = predictor(
                    solver,
                    x_t,
                    x_prev,
                    f_ext,
                    gravity64,
                    mu32,
                    lam32,
                    pin_flag,
                    S_cur,
                    S_prev,
                )
                x_next = ts.solve(solver, S_star.double(), pin_b, x_init=x_next, n_iters=iters_per_block)
                if _b + 1 < args.blocks:
                    S_cur = compute_S_from_x(solver, x_next)

            if args.loss == "pos":
                diff = x_next - x_gpu[i_t0 + k + 1]
                loss_total = loss_total + (mass[None, :, None] * diff * diff).sum()
                if args.phys_weight > 0.0:
                    loss_total = loss_total + args.phys_weight * incremental_potential_batched(
                        x_next=x_next,
                        x_t=x_t,
                        x_prev=x_prev,
                        mass=mass,
                        gravity=gravity64,
                        f_ext=f_ext,
                        tets=solver.tets,
                        J=solver.J,
                        mu=mu64,
                        lam=lam64,
                        volume=volume,
                        dt=args.dt,
                    )
            else:
                loss_total = loss_total + incremental_potential_batched(
                    x_next=x_next,
                    x_t=x_t,
                    x_prev=x_prev,
                    mass=mass,
                    gravity=gravity64,
                    f_ext=f_ext,
                    tets=solver.tets,
                    J=solver.J,
                    mu=mu64,
                    lam=lam64,
                    volume=volume,
                    dt=args.dt,
                )

            S_prev = S_now
            S_now = compute_S_from_x(solver, x_next)
            x_prev = x_t
            x_t = x_next
            x_pred = x_next

        (loss_total / (args.batch * k_roll)).backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                pos_err = (x_pred - x_gpu[i_t0 + k_roll]).norm(dim=-1).mean().item()
            mean_loss = loss_total.item() / (args.batch * k_roll)
            elapsed = time.time() - t0
            print(f"step {step:5d}  K={k_roll}  L={mean_loss:+.4e}  pos_err={pos_err:.4e}  {elapsed:.1f}s", flush=True)
            log.append({"step": step, "K": k_roll, "loss": mean_loss, "pos_err": pos_err})

    train_seconds = time.time() - t0
    runtime = {"train_seconds": train_seconds, "parameter_count": parameter_count}
    if device.type == "cuda":
        runtime.update(
            {
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "device_name": torch.cuda.get_device_name(device),
            }
        )
        print(
            "peak CUDA memory: "
            f"{runtime['peak_cuda_allocated_bytes'] / 2**30:.2f} GiB allocated, "
            f"{runtime['peak_cuda_reserved_bytes'] / 2**30:.2f} GiB reserved"
        )
    print(f"training done in {train_seconds:.1f}s")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": predictor.model.state_dict(),
            "predictor_config": predictor.checkpoint_config(),
            "log": log,
            "args": vars(args),
            "runtime": runtime,
            "torch_version": str(torch.__version__),
        },
        out,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
