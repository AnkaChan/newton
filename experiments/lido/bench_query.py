# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inference speed of one optimizer query at batch 1 on the canonical beam: eager, eager with capacity pairs,
CUDA-graph replay, replay with the compiled cell-graph layer; all back to back in one process.

    python -m experiments.lido.bench_query --plane --profile
    python -m experiments.lido.bench_query --plane --voxel-hole 3 --fusion mg   # the beam with a through-hole
"""

from __future__ import annotations

import argparse
import time

import torch

from . import csr_attention, scenes
from .batch import Batch
from .capture import CapturedQuery
from .config import TrainConfig
from .fusion import Fusion
from .grid import Grid
from .network import Net
from .rollout import COMPILE_MODE
from .step import Step
from .units import material_from_si


def beam_with_hole(cell_counts, radius: float) -> torch.Tensor:
    """Occupancy of the beam with a through-hole of the given radius (cells) along z through the section centre."""
    nx, ny, nz = cell_counts
    x, y, _ = torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij")
    return ~(((x + 0.5 - nx / 2) ** 2 + (y + 0.5 - ny / 2) ** 2) <= radius**2)


def make_batch(cfg: TrainConfig, device, plane: bool, points: bool, hole: float = 0.0) -> Batch:
    if hole > 0:
        grid = Grid.from_voxels(beam_with_hole(cfg.cell_counts, hole), cfg.pins, device)
    else:
        grid = Grid.build(cfg.cell_counts, cfg.pins, device)
    b = Batch.build([grid], device)
    b.material = material_from_si(
        E=1e5,
        nu=0.3,
        rho=1000.0,
        eta=100.0,
        gravity=cfg.gravity,
        h=cfg.cell_size,
        dt=cfg.time_step,
        cell_count=grid.C,
        sample_count=grid.S,
        kappa=100.0 if plane or points else 0.0,
        beta=0.1,
        mu_f=0.3,
        device=device,
    )
    spec = {}
    if plane:
        spec.update(plane_present=True, plane_height=-0.4 * cfg.cell_size)  # 0.4 h under the rest bottom: in contact
    if points:
        spec.update(points=[[0.1, -0.02, 0.5], [0.15, 0.27, 0.3]], normals=[[0, 1, 0], [0, -1, 0]], radii=[0.05, 0.05])
    b.scene = scenes.scene_from_spec(spec, cfg.cell_size, device)
    b.X = grid.rest.clone()
    b.V = torch.zeros_like(b.X)
    b.X_prev = b.X.clone()
    b.x = b.X.clone()
    return b


def time_blocks(fn, blocks: int, per_block: int, device) -> list[float]:
    """Milliseconds per call for `blocks` timed blocks (the first block is warm-up and dropped)."""
    times = []
    for i in range(blocks + 1):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(per_block):
            fn()
        torch.cuda.synchronize(device)
        if i > 0:
            times.append((time.perf_counter() - t0) / per_block * 1e3)
    return times


def time_interleaved(fns: dict, rounds: int, per_block: int, device) -> dict:
    """Round-robin over the callables: each round times one block per callable, so contention drift hits all
    of them alike. Returns name -> (median, min) ms per call."""
    times = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            times[name].extend(time_blocks(fn, 1, per_block, device))
    return {name: (sorted(ts)[len(ts) // 2], min(ts)) for name, ts in times.items()}


def profile_replay(captured: CapturedQuery, n: int, rows: int) -> tuple[str, int]:
    """(profiler table sorted by CUDA time, number of CUDA kernels per replay)."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            captured.replay()
        torch.cuda.synchronize()
    averages = prof.key_averages()
    kernels = sum(e.count for e in averages if e.device_type == torch.autograd.DeviceType.CUDA) // n
    return averages.table(sort_by="cuda_time_total", row_limit=rows), kernels


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="LIDO query speed: eager / capacity pairs / captured / compiled")
    p.add_argument("--plane", action="store_true", help="ground plane 0.4 h under the rest bottom (contact active)")
    p.add_argument("--points", action="store_true", help="two static points near the beam")
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--per-block", type=int, default=10)
    p.add_argument("--no-compile", action="store_true", help="skip the compiled variants")
    p.add_argument("--profile", action="store_true", help="torch.profiler table of the captured replay")
    p.add_argument("--profile-rows", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--voxel-hole", type=float, default=0.0, help="voxel beam with a z through-hole of this radius")
    p.add_argument("--fusion", default="auto", help="fusion solver: auto, kron, mg, sparse, dense")
    p.add_argument("--edge-module", default="a02", choices=["a02", "pair"], help="TrainConfig.edge_module")
    a = p.parse_args(argv)
    device = torch.device("cuda:0")
    torch.manual_seed(a.seed)
    cfg = TrainConfig(edge_module=a.edge_module)
    net = Net.from_config(cfg).to(device).eval()
    with torch.no_grad():
        for prm in net.parameters():
            torch.nn.init.normal_(prm, std=0.02)
    sel = torch.ones(1, dtype=torch.bool, device=device)
    results = {}

    def report(name, fn):
        mean, best = time_blocks(fn, a.blocks, a.per_block, device)
        results[name] = (mean, best)
        print(f"{name:64s} {mean:7.2f} ms/query  (best block {best:6.2f})", flush=True)

    def show(name, med, best):
        print(f"{name:58s} median {med:7.2f} ms/query   best block {best:6.2f}", flush=True)

    with torch.no_grad():
        eager = make_batch(cfg, device, a.plane, a.points, a.voxel_hole)
        step_e = Step(net, Fusion(a.fusion))
        step_e.prepare(eager, sel)
        cap = make_batch(cfg, device, a.plane, a.points, a.voxel_hole)
        step_c = Step(net, Fusion(a.fusion), pair_capacity=True)
        step_c.prepare(cap, sel)
        shape = f"voxel beam, hole radius {a.voxel_hole:g}" if a.voxel_hole > 0 else "canonical beam"
        print(
            f"{shape} {cfg.cell_counts}: {eager.C} cells, {eager.edges.shape[1]} edges, {eager.grids[0].Pf} free corners, "
            f"fusion {type(step_c.fusion.factor(cap.grids[0])).__name__}, {eager.pairs.count} pairs (capacity rows "
            f"{cap.pairs.count}); edge module {cfg.edge_module}; {a.blocks} rounds x {a.per_block} queries",
            flush=True,
        )
        eager_fns = {
            "eager (compacted pairs)": lambda: step_e.commit(eager, step_e.query(eager)),
            "eager (capacity pairs, no host sync)": lambda: step_c.commit(cap, step_c.query(cap)),
        }
        graphs = {"captured": CapturedQuery(step_c, cap)}
        if not a.no_compile:
            variants = (
                ("captured, compiled layer + edge encoder (graph break at attention)", False, False, False),
                ("captured, compiled layer + edge encoder (attention custom op)", True, True, False),
                ("captured, compiled layer + edge encoder + feature chains (custom op)", True, True, True),
            )
            for name, custom_op, fullgraph, feats in variants:
                csr_attention.CUSTOM_OP = custom_op
                torch._dynamo.reset()
                t0 = time.perf_counter()
                net.compile_layers(edge_encoder=True, fullgraph=fullgraph, mode=COMPILE_MODE)
                if feats:
                    step_c.compile_features(mode=COMPILE_MODE)
                graphs[name] = CapturedQuery(step_c, cap)  # the warm-up compiles
                torch.cuda.synchronize(device)
                print(f"{name}: compile + capture took {time.perf_counter() - t0:.0f} s", flush=True)
            eager_fns["eager (capacity pairs), everything compiled"] = lambda: step_c.commit(cap, step_c.query(cap))
        for name, (med, best) in time_interleaved(eager_fns, a.blocks, a.per_block, device).items():
            show(name, med, best)
        for name, (med, best) in time_interleaved(
            {n: g.replay for n, g in graphs.items()}, a.blocks, a.per_block, device
        ).items():
            show(name, med, best)
            results[name] = (med, best)
        if a.profile:
            for name in (next(iter(graphs)), list(graphs)[-1]):
                table, kernels = profile_replay(graphs[name], a.per_block, a.profile_rows)
                print(f"\n{name}: {kernels} CUDA kernels per replay\n{table}", flush=True)
    return results


if __name__ == "__main__":
    main()
