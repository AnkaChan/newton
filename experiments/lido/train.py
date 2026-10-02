# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Trainer (design spec 7): fixed-state epochs, one AdamW update per batch, cosine LR by epoch, validation,
selection and checkpoints at every epoch boundary, report files for the existing dashboard.

    torchrun --nproc_per_node 4 -m experiments.lido.train --config cfg.json --run-dir generated/run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

from . import contact
from . import report as rep
from .augment import Augmenter
from .config import TrainConfig
from .fusion import Fusion
from .grid import GridCache
from .jobs import growth_stage
from .network import Net
from .runner import make_runner
from .step import Step
from .units import force_scale
from .validation import (
    local_objective,
    validate_cheap,
    validate_cheap_v5,
    validate_full_horizon,
    validate_full_horizon_v5,
)


def cosine_lr(epoch: int, cfg: TrainConfig) -> float:
    if cfg.max_epochs <= 1:
        return cfg.learning_rate
    t = (epoch - 1) / (cfg.max_epochs - 1)
    return cfg.lr_final + 0.5 * (cfg.learning_rate - cfg.lr_final) * (1 + math.cos(math.pi * t))


def step_cap(cfg: TrainConfig, advanced_epochs: int) -> float:
    """Per-cell step cap for the next epoch: max_step_size x (start + (1 - start) x min(1, advanced / ramp))."""
    if cfg.step_cap_ramp_epochs <= 0:
        return cfg.max_step_size
    frac = min(1.0, advanced_epochs / cfg.step_cap_ramp_epochs)
    return cfg.max_step_size * (cfg.step_cap_start + (1.0 - cfg.step_cap_start) * frac)


STEP_CAP_STATE = {"advanced": 0, "last_metric": None}  # ramp progress, saved in checkpoints


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def save_checkpoint(path: Path, net, opt, epoch: int, updates: int, best: dict | None, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "network_state": net.state_dict(),
            "optimizer_state": opt.state_dict(),
            "epoch": epoch,
            "updates_done": updates,
            "best_selection": best,
            "config": cfg.to_dict(),
            "feature_schema_version": cfg.feature_schema_version,
            "git_sha": git_sha(),
            "step_cap_state": dict(STEP_CAP_STATE),
        },
        tmp,
    )
    os.replace(tmp, path)


def unwrap(net):
    return net.module if hasattr(net, "module") else net


def train(
    cfg: TrainConfig,
    run_dir: str,
    resume: str | None = None,
    rank: int = 0,
    world: int = 1,
    max_updates: int | None = None,
):
    device = torch.device(cfg.device if world == 1 else f"cuda:{rank}")
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    run_dir = Path(run_dir)
    torch.set_num_threads(max(1, cfg.cpu_threads))
    torch.manual_seed(cfg.seed + rank)
    v5 = cfg.scene_mode == "v5"
    net = Net.from_config(cfg).to(device)
    if cfg.compile_network and device.type == "cuda":
        # body mode: static shapes per grid; v5: symbolic shapes, one graph for every scene (section 11)
        net.compile_layers(dynamic=v5)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    start_epoch, updates_done, best = 1, 0, None
    initialized_from = None
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["network_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start_epoch, updates_done, best = ck["epoch"] + 1, ck["updates_done"], ck.get("best_selection")
        STEP_CAP_STATE.update(ck.get("step_cap_state", {"advanced": cfg.step_cap_ramp_epochs, "last_metric": None}))
        initialized_from = {"checkpoint": str(resume), "completed_epochs": ck["epoch"], "best_selection": best}
    model = net
    if world > 1:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
        model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device.index], broadcast_buffers=False)
    grids = GridCache(device)
    grid = grids.get(cfg.cell_counts, cfg.pins) if not v5 else None
    aug = Augmenter(device)
    step = Step(model, Fusion(batched=v5), aug, noise_range=cfg.candidate_noise_range)  # v5: one padded Kron solve
    runner = make_runner(cfg, step, aug, grids, rank, world, device, cfg.seed)
    report = rep.RunReport(run_dir, cfg, world, git_sha(), initialized_from) if rank == 0 else None
    if report and resume and (run_dir / "report.json").exists():
        old = json.loads((run_dir / "report.json").read_text())  # a resume in the same run dir keeps its history
        for key in ("epochs", "updates", "best_selection_history"):
            report.report[key] = old.get(key, [])
        report.report["completed_epochs"] = len(report.report["epochs"])
        report.report["completed_updates"] = old.get("completed_updates", updates_done)
    if report:
        report.set_parameter_count(sum(p.numel() for p in net.parameters()))
        if best:
            report.set_best_selection(best, start_epoch - 1)
        report.set_status("running", phase="training")
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        t_epoch = time.perf_counter()
        lr = cosine_lr(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        cap = step_cap(cfg, STEP_CAP_STATE["advanced"])
        unwrap(net).step_cap.fill_(cap)
        runner.start_epoch(epoch)
        stage, k_max, h_max = growth_stage(epoch, cfg)
        losses, gnorms, steps_mean, steps_min, steps_max, residuals, pens, realized = [], [], [], [], [], [], [], []
        n_updates = runner.U if max_updates is None else min(runner.U, max_updates)
        for update in range(n_updates):
            t0 = time.perf_counter()
            batch = runner.batch
            out = step.query(batch)
            loss_vec = local_objective(
                out.E_after, out.E_before, batch.material.floor, cfg.energy_increase_weight, cfg.bounded_increase
            )
            mask = batch.active & torch.isfinite(loss_vec)
            loss = loss_vec.masked_fill(~mask, 0.0).sum() / mask.sum().clamp_min(1)  # NaN rows must not poison the mean
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.gradient_clip_norm)
            if torch.isfinite(gn):
                opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                residuals.append(
                    float((out.residual * force_scale(batch.material))[mask].mean())
                    if bool(mask.any())
                    else float("nan")
                )
                realized.append(int(batch.pairs.count > 0))
                if v5 and bool(mask.any()):
                    pens.append(float(contact.penetration(batch, out.cand_after.detach()).max()))
            runner.commit(out)
            updates_done += 1
            lv, gv = float(loss.detach()), float(gn)
            losses.append(lv)
            gnorms.append(gv)
            sm = out.step
            steps_mean.append(float(sm.mean()))
            steps_min.append(float(sm.min()))
            steps_max.append(float(sm.max()))
            if report and (update % cfg.log_every == 0 or update == n_updates - 1):
                report.log_update(
                    epoch,
                    updates_done,
                    lv,
                    gv,
                    lr,
                    steps_mean[-1],
                    runner.active_count,
                    runner.resets,
                    time.perf_counter() - t0,
                )
        if world > 1:
            torch.distributed.barrier()
        if rank == 0:
            unwrap(step.net).eval()
            K_full, H_full = min(k_max, cfg.validation_full_iterations), h_max  # stage caps, as in the v4 campaign
            if v5:
                cheap_samples = validate_cheap_v5(step, cfg, grids, aug, device, cfg.seed, epoch)
                full_samples, full_seconds = validate_full_horizon_v5(
                    step, cfg, grids, aug, device, cfg.seed, K_full, H_full
                )
            else:
                cheap_samples = validate_cheap(step, cfg, aug, grid, device, cfg.seed)
                full_samples, full_seconds = validate_full_horizon(
                    step, cfg, aug, grid, device, cfg.seed, K_full, H_full
                )
            cheap = rep.summarize_cheap_validation(cheap_samples, cfg.validation_iterations)
            full = rep.summarize_full_horizon(full_samples, K_full, H_full, full_seconds)
            unwrap(step.net).train()
            selection = full["selection"] if cfg.selection_source == "full_horizon" else cheap["selection"]
            candidate = {
                "metric": selection["metric"],
                "eligible": selection["eligible"],
                "epoch": epoch,
                "source": cfg.selection_source,
                "iterations": K_full,
                "physical_steps": H_full,
                "final_energy_joule": full["final_energy_joule"],
                "final_max_penetration_r": full["final_max_penetration_r"],
            }
            metric = selection.get("metric")
            improved = metric is not None and (
                STEP_CAP_STATE["last_metric"] is None or metric <= STEP_CAP_STATE["last_metric"]
            )
            if not cfg.step_cap_gate_on_validation or improved:
                STEP_CAP_STATE["advanced"] += 1  # the cap ramps only while the selection metric does not get worse
            if metric is not None:
                STEP_CAP_STATE["last_metric"] = metric
            if rep.selection_better(candidate, best):
                best = candidate
                report.set_best_selection(best, epoch)
                save_checkpoint(
                    run_dir / "checkpoints" / "best_validation.pt", net, opt, epoch, updates_done, best, cfg
                )
            finite_losses = [v for v in losses if math.isfinite(v)]
            query_count = runner.queries if v5 else n_updates * cfg.batch_size
            record = rep.build_epoch_record(
                epoch=epoch,
                loss=sum(finite_losses) / max(1, len(finite_losses)),
                query_count=query_count,
                updates=n_updates,
                seconds=time.perf_counter() - t_epoch,
                lr=lr,
                grad_norm_mean=sum(gnorms) / max(1, len(gnorms)),
                grad_norm_max=max(gnorms) if gnorms else 0.0,
                step_mean=sum(steps_mean) / max(1, len(steps_mean)),
                step_min=min(steps_min) if steps_min else 0.0,
                step_max=max(steps_max) if steps_max else 0.0,
                tie_cell_count=0,
                mean_force_residual_n=sum(r for r in residuals if math.isfinite(r))
                / max(1, sum(1 for r in residuals if math.isfinite(r))),
                resets=runner.resets,
                failures=[f.__dict__ for f in runner.failures],
                regime={
                    "stage": stage,
                    "k_max": k_max,
                    "h_max": h_max,
                    "queries": query_count,
                    "filler_queries": runner.idle_updates if v5 else 0,
                    "updates": n_updates,
                    "step_cap": cap,
                    **(
                        {"pinned_fraction": runner.mix.pinned_fraction, "resting_fraction": runner.mix.resting_fraction}
                        if v5
                        else {}
                    ),
                },
                available_K=[k for k in cfg.K_values if k <= k_max],
                available_H=list(range(1, h_max + 1)),
                contact_scene_fraction=runner.contact_scenes / max(1, runner.loaded_jobs),
                contact_realized_fraction=sum(realized) / max(1, len(realized)),
                contact_max_penetration_r=max((p for p in pens), default=0.0),
                material_histograms=None,
                validation=cheap,
                full_horizon_validation=full,
                rank_diagnostics=None,
                scene_regime=runner.epoch_summary() if v5 else None,
            )
            report.log_epoch(record)
            save_checkpoint(run_dir / "checkpoints" / "latest.pt", net, opt, epoch, updates_done, best, cfg)
            if epoch % cfg.checkpoint_interval == 0:
                save_checkpoint(
                    run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt", net, opt, epoch, updates_done, best, cfg
                )
        if world > 1:
            torch.distributed.barrier()
    if report:
        report.set_status("completed", phase="done")
    if world > 1:
        torch.distributed.destroy_process_group()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-updates", type=int, default=None)
    args = ap.parse_args()
    cfg = TrainConfig.load(args.config)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    train(cfg, args.run_dir, args.resume, rank, world, args.max_updates)


if __name__ == "__main__":
    main()
