# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental detached mixed-pool trainer; no retained intermediate dataset.

One update evaluates one proposal per ready trajectory. Epochs count global
optimizer queries, not unique trajectories. CPU preparation overlaps other
ready queries; differentiable CPU fusion still synchronizes each proposal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .material_sampling import MaterialRanges
from .mixed_validation import validate as _validate
from .mixed_validation import validation_chunk as _validation_chunk  # noqa: F401 -- Keep the existing test seam.

__all__ = ["MixedTrainConfig", "local_objective", "run_training"]


@dataclass(frozen=True)
class MixedTrainConfig:
    """Experimental V2 settings. Material is independently sampled per reset."""

    cell_counts: tuple[int, int, int] = (10, 10, 40)
    cell_size: float = 0.025
    time_step: float = 1 / 300
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    hidden_dim: int = 128
    edge_hidden_dim: int = 64
    num_heads: int = 4
    hops: tuple[int, ...] = (1,)
    query_chunk_size: int = 128
    max_step_size: float = 0.05
    learning_rate: float = 1e-4
    energy_increase_weight: float = 1.0
    batch_size: int = 16
    pool_multiplier: int = 4
    queries_per_epoch: int = 8192
    max_epochs: int = 500
    stage_epochs: int = 10
    stage_patience: int = 2
    stage_descent_rate: float = 0.9
    candidate_probabilities: tuple[float, ...] = (0.5, 0.35, 0.1, 0.05)
    iteration_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    physical_step_counts: tuple[int, ...] = (8, 16, 32, 64, 128)
    validation_count: int = 512
    validation_iterations: int = 100
    validation_physical_steps: int = 8
    validation_physical_iterations: int = 2
    checkpoint_interval: int = 5
    early_stopping: bool = True
    youngs_modulus_range: tuple[float, float] = (1e3, 1e6)
    poissons_ratio_range: tuple[float, float] = (0.2, 0.49)
    density_range: tuple[float, float] = (100.0, 10000.0)
    strength_range: tuple[float, float] = (0.02, 0.1)
    velocity_dt_range: tuple[float, float] = (0.0, 0.1)
    perturbation_scale_range: tuple[float, float] = (0.0, 1.0)
    seed: int = 73
    device: str = "cuda"
    cpu_threads: int = 2
    preparation_workers: int = 2
    verbose: bool = True

    def __post_init__(self):
        """Reject invalid budgets and physical ranges before creating outputs."""
        for name in (
            "cell_counts",
            "gravity",
            "hops",
            "iteration_counts",
            "physical_step_counts",
            "youngs_modulus_range",
            "poissons_ratio_range",
            "density_range",
            "strength_range",
            "velocity_dt_range",
            "perturbation_scale_range",
            "candidate_probabilities",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in (
            "hidden_dim",
            "edge_hidden_dim",
            "num_heads",
            "query_chunk_size",
            "batch_size",
            "pool_multiplier",
            "queries_per_epoch",
            "max_epochs",
            "stage_epochs",
            "validation_count",
            "validation_iterations",
            "validation_physical_steps",
            "validation_physical_iterations",
            "checkpoint_interval",
            "cpu_threads",
            "preparation_workers",
            "stage_patience",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("cell_counts", "hops", "iteration_counts", "physical_step_counts"):
            values = getattr(self, name)
            if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integers")
        if len(self.cell_counts) != 3 or self.hidden_dim % self.num_heads or self.pool_multiplier < 2:
            raise ValueError("invalid grid, attention heads, or pool multiplier (minimum 2)")
        if max(self.iteration_counts) > 32 or max(self.physical_step_counts) > 128:
            raise ValueError("V2 budgets are limited to K <= 32 and H <= 128")
        if 1 not in self.iteration_counts or min(self.physical_step_counts) > 8:
            raise ValueError("initial curriculum requires K=1 and an H <= 8")
        for name in ("cell_size", "time_step", "max_step_size", "learning_rate"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.energy_increase_weight) or self.energy_increase_weight < 0:
            raise ValueError("energy_increase_weight must be finite and nonnegative")
        if not 0 <= self.stage_descent_rate <= 1:
            raise ValueError("stage_descent_rate must lie in [0,1]")
        if (
            len(self.candidate_probabilities) != 4
            or any(not math.isfinite(v) or v < 0 for v in self.candidate_probabilities)
            or not math.isclose(sum(self.candidate_probabilities), 1.0)
        ):
            raise ValueError("candidate_probabilities must contain four nonnegative probabilities summing to one")
        if len(self.gravity) != 3 or not all(math.isfinite(v) for v in self.gravity):
            raise ValueError("gravity must be a finite three-vector")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("strength_range", "velocity_dt_range", "perturbation_scale_range"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 <= bounds[0] <= bounds[1]:
                raise ValueError(f"invalid {name}")
        self.material_ranges()

    def material_ranges(self):
        """Return the independent E, nu, rho sampling bounds."""
        return MaterialRanges(self.youngs_modulus_range, self.poissons_ratio_range, self.density_range)


def local_objective(after, initial, previous, *, increase_weight=1.0):
    """Return per-member local losses; gradients reach only the new proposal."""
    import torch

    initial, previous = initial.detach(), previous.detach()
    scale = initial.clamp_min(1.0)
    return (after - initial) / scale + increase_weight * torch.relu(after - previous) / scale


class _TrajectoryFactory:
    """Prepare detached CPU trajectories without accessing network parameters."""

    def __init__(self, step, rest, config, *, rank, validation=False):
        self.step, self.rest, self.config = step, rest, config
        self.prefix = f"{'validation' if validation else 'train'}-{rank}"
        self.master_seed = config.seed + (1000000007 if validation else 0)
        self.seed_parity = int(validation)
        # Preparation workers use CPU topology without synchronizing CUDA.
        self.fixed_indices = step.fixed_indices.detach().cpu().clone()

    def reset(self, seed):
        import torch

        from .initial_state import InitialStateAugmenter  # noqa: PLC0415 -- Optional training boundary.

        c = self.config
        initial = InitialStateAugmenter(
            self.rest,
            master_seed=self.master_seed,
            time_step=c.time_step,
            material_ranges=c.material_ranges(),
            strength_range=c.strength_range,
            velocity_dt_range=c.velocity_dt_range,
            perturbation_scale_range=c.perturbation_scale_range,
        ).reset(2 * seed + self.seed_parity)
        key = f"{self.prefix}-{seed}"
        specification = asdict(initial.material)
        self.step.register_context(key, **specification)
        try:
            payload = self.step.prepare(key, torch.from_numpy(initial.positions), torch.from_numpy(initial.velocities))
            payload.update(context_spec=specification, metadata=initial.metadata, seed=seed, physical_age=0)
            return self._candidate(payload)
        except BaseException:
            self.step.discard_context(key)
            raise

    def advance(self, payload):
        from .train_epochs import _cpu  # noqa: PLC0415 -- Optional training boundary.

        payload = _cpu(payload)
        prepared = self.step.advance(payload)
        for key in ("context_spec", "metadata", "seed"):
            prepared[key] = payload[key]
        prepared["physical_age"] = payload["physical_age"] + 1
        return self._candidate(prepared)

    def retire(self, payload):
        self.step.discard_context(payload["context_id"])

    def _candidate(self, payload):
        """Use configurable initializer probabilities; never repair a learned output."""
        import torch

        from .multiscale import generate_multiscale, screen_geometry  # noqa: PLC0415 -- Optional training boundary.

        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, payload["seed"], payload["physical_age"], 911])
        )
        draw = rng.random()
        index = min(int(np.searchsorted(np.cumsum(self.config.candidate_probabilities), draw, side="right")), 3)
        mode = ("inertial", "noisy_inertial", "previous", "rigid")[index]
        fixed = self.fixed_indices
        base = payload["inertial_prediction"].clone()
        base[fixed] = payload["fixed_positions"]

        def valid(candidate):
            return (
                bool(torch.isfinite(candidate).all())
                and min(screen_geometry(self.rest, candidate.numpy()).values()) > 0
            )

        base_fallback = not valid(base)
        if base_fallback:
            base = payload["physical_positions"].clone()
        candidate = (
            payload["physical_positions"].clone()
            if mode == "previous"
            else payload["candidate"].clone()
            if mode == "rigid"
            else base.clone()
        )
        halvings = 0
        if mode == "noisy_inertial":
            sample = generate_multiscale(self.rest, seed=int(rng.integers(2**32)), strength=0.1)
            noise = sample.positions - self.rest.corner_rest_positions
            rms = float(np.sqrt(np.mean(np.sum(noise**2, axis=-1))))
            noise *= float(rng.uniform(0.01, 0.1) * self.config.cell_size) / rms if rms else 0
            noise = torch.from_numpy(noise.astype(np.float32))
            noise[fixed] = 0
            candidate = base + noise
            while not valid(candidate) and halvings < 32:
                halvings += 1
                candidate = base + (0.5**halvings) * noise
        candidate[fixed] = payload["fixed_positions"]
        fallback = not valid(candidate)
        if fallback:
            candidate = payload["physical_positions"].clone()
        if not valid(candidate):
            raise ValueError("physical state is invalid before candidate initialization")
        payload.update(
            candidate=candidate.detach(),
            candidate_mode=mode,
            initializer_halvings=halvings,
            initializer_fallback=bool(fallback or base_fallback),
        )
        return payload


def _batch(records, device):
    import torch

    payloads = [getattr(record, "payload", record) for record in records]
    values = {
        name: torch.stack([p[name].detach().to(device) for p in payloads])
        for name in ("candidate", "inertial_prediction", "fixed_positions")
    }
    values["context_ids"] = tuple(p["context_id"] for p in payloads)
    return values


def _checked_forward(module, step, batch):
    import torch

    result = module(
        batch["candidate"], batch["inertial_prediction"], batch["context_ids"], fixed_positions=batch["fixed_positions"]
    )
    if not torch.isfinite(result.loss.total).all() or not torch.isfinite(result.positions).all():
        raise ValueError("nonfinite learned proposal; trajectory retained as a failure")
    if not torch.equal(result.positions[:, step.fixed_indices], batch["fixed_positions"]):
        raise ValueError("learned proposal moved prescribed corners")
    with torch.no_grad():
        corners = result.positions[:, step.cell_corner_indices]
        center = torch.einsum("bcki,kj->bcij", corners - corners[:, :, :1], step.center_gradients)
        singular = torch.linalg.svdvals(center)
        threshold = 4 * torch.finfo(result.positions.dtype).eps * singular[..., 0].clamp_min(1)
        if (torch.linalg.det(center) <= 0).any() or (singular[..., -1] <= threshold).any():
            raise ValueError("learned proposal has an invalid or singular center deformation")
    return result


def _write_report(output, report):
    from .train_smoke import _atomic_json  # noqa: PLC0415 -- Optional training boundary.

    _atomic_json(output / "report.json", report)
    for name, rows, columns in (
        ("updates", report["updates"], ("update", "epoch", "loss", "before_joule", "after_joule")),
        ("epochs", report["epochs"], ("epoch", "loss", "query_count", "seconds")),
    ):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        (output / f"{name}.csv").write_text(buffer.getvalue())
    rows = report["epochs"]
    values = [r["loss"] for r in rows]
    lower, upper = (min(values), max(values)) if values else (-1, 1)
    width = max(upper - lower, 1e-6)
    points = " ".join(
        f"{50 + 700 * i / max(len(values) - 1, 1):.2f},{250 - 200 * (v - lower) / width:.2f}"
        for i, v in enumerate(values)
    )
    (output / "loss_curve.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300" viewBox="0 0 800 300">'
        '<rect width="800" height="300" fill="white"/><text x="35" y="22">Mean local training loss vs epoch</text>'
        f'<text x="5" y="55">{upper:.3g}</text><text x="5" y="255">{lower:.3g}</text>'
        f'<polyline points="{points}" fill="none" stroke="#2468b0" stroke-width="2"/></svg>'
    )
    (output / "index.html").write_text("""<!doctype html><meta charset="utf-8"><title>V2 mixed-pool training</title>
<style>body{font:16px system-ui;max-width:1000px;margin:3em auto;padding:0 1em}canvas{width:100%;height:320px}pre{white-space:pre-wrap}</style>
<h1>V2 mixed-pool training</h1><p>One detached proposal per member, one Adam update per batch.
Epochs count queries across all ranks, not unique initial states. Training curves mix different solver ages.
Validation weights and initial seeds are fixed throughout each evaluated trajectory.</p>
<img src="loss_curve.svg" style="width:100%"><h2>Validation relative energy</h2>
<p>Latest epoch: mean (blue), median (green), maximum (red), through the configured optimizer iterations.
Optimizer failures invalidate curves from the failed iteration onward; physical rollout failures and near-zero energies are reported separately.</p><canvas id="curve" width="1000" height="320"></canvas>
<pre id="status"></pre><a href="report.json">Full metrics</a> · <a href="updates.csv">Update losses</a> · <a href="epochs.csv">Epoch losses</a>
<script>fetch('report.json').then(r=>r.json()).then(r=>{let e=r.epochs.at(-1);document.querySelector('#status').textContent=JSON.stringify({status:r.status,updates:r.completed_updates,epoch:e},null,2);if(!e)return;
let rows=e.validation.relative_energy,c=document.querySelector('canvas').getContext('2d'),v=rows.flatMap(r=>[r.mean,r.median,r.max]).filter(Number.isFinite),hi=Math.max(1,...v),lo=Math.min(0,...v);
c.fillText('Relative energy',10,15);[['mean','#2468b0'],['median','#268b52'],['max','#b92d36']].forEach(([k,color])=>{c.beginPath();c.strokeStyle=color;let open=false;rows.forEach((r,i)=>{if(r[k]===null){open=false;return}let x=40+i*920/Math.max(1,rows.length-1),y=280-(r[k]-lo)*250/(hi-lo);if(open)c.lineTo(x,y);else c.moveTo(x,y);open=true});c.stroke()})})</script>""")


def run_training(output: Path, config: MixedTrainConfig, *, resume: Path | None = None):
    """Run an explicitly requested V2 campaign or a bounded verification run.

    Checkpoints restore the same rank count, curriculum, pool queues and Adam
    sequence. Native factors are rebuilt; they are never serialized.
    """
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from .curriculum import MixedCurriculum  # noqa: PLC0415 -- Optional training boundary.
    from .data import generate_cuboid  # noqa: PLC0415 -- Optional training boundary.
    from .mixed_physics import MixedHexSolverStep  # noqa: PLC0415 -- Optional training boundary.
    from .network import IntrinsicSolverNetwork  # noqa: PLC0415 -- Optional training boundary.
    from .train_epochs import _all_ranks_ok, _atomic_torch, _cpu  # noqa: PLC0415 -- Optional training boundary.
    from .training_schedule import PlateauController  # noqa: PLC0415 -- Optional training boundary.
    from .trajectory_pool import ActiveTrajectoryPool  # noqa: PLC0415 -- Optional training boundary.

    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if config.queries_per_epoch % (config.batch_size * world_size):
        raise ValueError("queries_per_epoch must be divisible by batch_size * world_size")
    output = Path(output).resolve()
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if saved:
        if saved.get("format") != "mixed_pool_v2" or saved["world_size"] != world_size:
            raise ValueError("incompatible checkpoint format or rank count")
        allowed = {"max_epochs", "verbose", "early_stopping"}
        if any(saved["config"].get(k) != v for k, v in asdict(config).items() if k not in allowed):
            raise ValueError("resume configuration differs from saved physical/training configuration")
        if config.max_epochs < saved["report"]["completed_epochs"]:
            raise ValueError("max_epochs precedes the checkpoint")
    elif (output / "report.json").exists() or (output / "checkpoints").exists():
        raise FileExistsError("use a fresh output directory or an explicit resume checkpoint")
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(config.device)
    if device.type == "cuda":
        if world_size > 1 and (torch.cuda.device_count() != 1 or int(os.environ.get("LOCAL_RANK", "0")) != 0):
            raise ValueError(
                "distributed training requires one exclusively claimed CUDA device per rank; use launch_training"
            )
        torch.cuda.set_device(0)
    owned_group = world_size > 1 and not dist.is_initialized()
    if owned_group:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    random.seed(config.seed + rank)
    np.random.seed((config.seed + rank) % 2**32)  # noqa: NPY002 -- Preserve process RNG in checkpoints.
    torch.manual_seed(config.seed)
    rest = generate_cuboid(config.cell_counts, cell_size=config.cell_size)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
    network = IntrinsicSolverNetwork(
        config.cell_counts,
        38,
        hidden_dim=config.hidden_dim,
        edge_hidden_dim=config.edge_hidden_dim,
        num_heads=config.num_heads,
        hops=config.hops,
        max_step_size=config.max_step_size,
        query_chunk_size=config.query_chunk_size,
    ).to(device)
    step = MixedHexSolverStep(rest, fixed, network=network, time_step=config.time_step, gravity=config.gravity).to(
        device
    )
    factory = _TrajectoryFactory(step, rest, config, rank=rank)
    validation_factory = _TrajectoryFactory(step, rest, config, rank=rank, validation=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    controller = PlateauController(
        config.learning_rate, min_epochs=1, max_epochs=config.max_epochs, min_lr=min(1e-6, config.learning_rate)
    )
    curriculum = MixedCurriculum(
        config.iteration_counts,
        config.physical_step_counts,
        min_stage_epochs=config.stage_epochs,
        patience=config.stage_patience,
        min_descent_rate=config.stage_descent_rate,
    )
    pool = None
    try:
        if saved:
            network.load_state_dict(saved["network_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            controller.load_state_dict(saved["controller_state"])
            curriculum.load_state_dict(saved["curriculum_state"])
            state = saved["rank_states"][rank]
            for key, spec in state["context_specs"].items():
                step.register_context(key, **spec)
            pool = ActiveTrajectoryPool.from_state_dict(
                state["pool"],
                reset=factory.reset,
                advance=factory.advance,
                retire=factory.retire,
                workers=config.preparation_workers,
            )
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])  # noqa: NPY002 -- Preserve process RNG in checkpoints.
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"], device)
            report = saved["report"]
            report["config"] = asdict(config)
        else:
            counts = curriculum.available_counts
            pool = ActiveTrajectoryPool(
                capacity=config.pool_multiplier * config.batch_size,
                batch_size=config.batch_size,
                reset=factory.reset,
                advance=factory.advance,
                retire=factory.retire,
                iteration_counts=counts[0],
                physical_step_counts=counts[1],
                seed=config.seed + rank * 100000000,
                workers=config.preparation_workers,
            )
            report = {
                "format": "mixed_pool_v2",
                "config": asdict(config),
                "world_size": world_size,
                "parameter_count": sum(p.numel() for p in network.parameters()),
                "completed_updates": 0,
                "completed_epochs": 0,
                "epochs": [],
                "updates": [],
                "status": "running",
            }
        module = (
            DistributedDataParallel(step, device_ids=[0] if device.type == "cuda" else None, broadcast_buffers=False)
            if world_size > 1
            else step
        )
        if rank == 0:
            (output / "checkpoints").mkdir(parents=True, exist_ok=True)

        def checkpoint(name):
            checkpoint_error = None
            try:
                pool_state = pool.state_dict()
            except Exception as caught:
                checkpoint_error = repr(caught)
            failures = _all_ranks_ok(checkpoint_error, device, world_size)
            if failures:
                raise RuntimeError(f"checkpoint preparation failed: {failures}")
            local = {
                "pool": pool_state,
                "context_specs": step.context_specs,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),  # noqa: NPY002 -- Preserve process RNG in checkpoints.
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "parameter_sha256": hashlib.sha256(
                    b"".join(p.detach().cpu().numpy().tobytes() for p in network.parameters())
                ).hexdigest(),
            }
            states = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(states, local)
            else:
                states[0] = local
            if rank == 0:
                _atomic_torch(
                    output / "checkpoints" / name,
                    {
                        "format": "mixed_pool_v2",
                        "config": asdict(config),
                        "world_size": world_size,
                        "network_state": _cpu(network.state_dict()),
                        "optimizer_state": _cpu(optimizer.state_dict()),
                        "controller_state": controller.state_dict(),
                        "curriculum_state": curriculum.state_dict(),
                        "rank_states": states,
                        "report": report,
                    },
                )

        if not saved:
            checkpoint("initial.pt")
        for epoch in range(report["completed_epochs"] + 1, config.max_epochs + 1):
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            counts = curriculum.available_counts
            pool.set_available_counts(*counts)
            totals = Counter()
            budgets, ages, steps, materials, perturbations = Counter(), Counter(), Counter(), [], []
            timings = Counter()
            for _ in range(config.queries_per_epoch // (config.batch_size * world_size)):
                began = time.perf_counter()
                error, records = None, None
                try:
                    records = pool.take_batch()
                    batch = _batch(records, device)
                    with torch.no_grad():
                        previous = step.energy(
                            batch["candidate"], batch["inertial_prediction"], batch["context_ids"]
                        ).total
                    initial = torch.stack(
                        [
                            r.payload.get("energy_initial", previous[i]).to(device).detach()
                            for i, r in enumerate(records)
                        ]
                    )
                    if not torch.isfinite(initial).all() or not torch.isfinite(previous).all():
                        raise ValueError("nonfinite input energy")
                except Exception as caught:
                    error = repr(caught)
                failures = _all_ranks_ok(error, device, world_size)
                if failures:
                    raise RuntimeError(f"mixed input preparation failed: {failures}")
                timings["prepare_wait_and_transfer_seconds"] += time.perf_counter() - began
                optimizer.zero_grad(set_to_none=True)
                began = time.perf_counter()
                error = None
                try:
                    with torch.autocast(device_type=device.type, enabled=False):
                        result = _checked_forward(module, step, batch)
                        losses = local_objective(
                            result.loss.total, initial, previous, increase_weight=config.energy_increase_weight
                        )
                        loss = losses.mean()
                except Exception as caught:
                    error = repr(caught)
                failures = _all_ranks_ok(error, device, world_size)
                if failures:
                    raise RuntimeError(f"learned proposal failed: {failures}")
                timings["forward_seconds"] += time.perf_counter() - began
                began = time.perf_counter()
                loss.backward()
                gradient_error = (
                    None
                    if all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
                    else "nonfinite gradient"
                )
                failures = _all_ranks_ok(gradient_error, device, world_size)
                if failures:
                    raise RuntimeError(f"backward failed: {failures}")
                optimizer.step()
                timings["backward_and_adam_seconds"] += time.perf_counter() - began
                after = result.loss.total.detach()
                values = torch.stack((losses.detach().sum(), previous.sum(), after.sum())).double()
                if world_size > 1:
                    dist.all_reduce(values)
                values = values.cpu().tolist()
                query_count = config.batch_size * world_size
                report["completed_updates"] += 1
                report["updates"].append(
                    {
                        "update": report["completed_updates"],
                        "epoch": epoch,
                        "loss": values[0] / query_count,
                        "before_joule": values[1] / query_count,
                        "after_joule": values[2] / query_count,
                    }
                )
                totals.update(loss=values[0], query_count=query_count)
                for i, record in enumerate(records):
                    budgets[f"{record.iteration_budget}/{record.step_budget}"] += 1
                    ages[str(record.inner_iteration)] += 1
                    steps[str(record.physical_step)] += 1
                    materials.append(
                        {**record.payload["context_spec"], **record.payload["metadata"]["material_parameters"]}
                    )
                    perturbations.append(record.payload["metadata"]["perturbation_scale"])
                    record.payload.update(
                        candidate=result.positions[i].detach(),
                        energy_initial=initial[i].detach(),
                        energy_previous=after[i].detach(),
                    )
                pool.finish_batch(records)
                del result, loss, losses, records, batch, initial, previous, after
            validation = _validate(step, validation_factory, config, device, rank, world_size)
            curriculum_decision = curriculum.observe(validation)
            decision = controller.observe(
                epoch, validation, allow_early_stop=config.early_stopping and epoch >= max(30, 2 * config.stage_epochs)
            )
            for group in optimizer.param_groups:
                group["lr"] = decision["learning_rate"]
            diagnostics = {
                "rank": rank,
                "budgets": dict(budgets),
                "inner_ages": dict(ages),
                "physical_ages": dict(steps),
                "pool": dict(pool.stats),
                "timings": dict(timings),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "material_histograms": {},
                "perturbation_histogram": np.histogram(perturbations, bins=np.linspace(0, 1, 11))[0].tolist(),
            }
            for name, bounds, logarithmic in (
                ("youngs_modulus", config.youngs_modulus_range, True),
                ("poissons_ratio", config.poissons_ratio_range, False),
                ("density", config.density_range, True),
            ):
                edges = np.geomspace(*bounds, 11) if logarithmic else np.linspace(*bounds, 11)
                # Constant configured materials still need nonzero histogram bins.
                if bounds[0] == bounds[1]:
                    edges = np.linspace(bounds[0] - 0.5, bounds[1] + 0.5, 11)
                diagnostics["material_histograms"][name] = {
                    "edges": edges.tolist(),
                    "counts": np.histogram([m[name] for m in materials], bins=edges)[0].tolist(),
                }
            rank_diagnostics = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(rank_diagnostics, diagnostics)
            else:
                rank_diagnostics[0] = diagnostics
            row = {
                "epoch": epoch,
                "loss": totals["loss"] / totals["query_count"],
                "query_count": totals["query_count"],
                "seconds": time.perf_counter() - epoch_start,
                "available_K": counts[0],
                "available_H": counts[1],
                "rank_0_budgets": dict(budgets),
                "rank_0_inner_ages": dict(ages),
                "rank_0_physical_ages": dict(steps),
                "rank_0_material_ranges": {
                    k: [min(m[k] for m in materials), max(m[k] for m in materials)] for k in materials[0]
                },
                "rank_0_perturbation_scale": [min(perturbations), max(perturbations)],
                "rank_0_timings": dict(timings),
                "validation": validation,
                "learning_rate": decision["learning_rate"],
                "curriculum": curriculum_decision,
                "rank_0_pool": dict(pool.stats),
                "rank_0_peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "rank_diagnostics": rank_diagnostics,
            }
            report["epochs"].append(row)
            report.update(completed_epochs=epoch, status=decision["status"])
            checkpoint("latest.pt")
            if controller.bad_epochs == 0:
                checkpoint("best_validation.pt")
            if epoch % config.checkpoint_interval == 0:
                checkpoint(f"epoch_{epoch:04d}.pt")
            if rank == 0:
                _write_report(output, report)
                if config.verbose:
                    print(
                        f"epoch {epoch}: loss={row['loss']:.6g}, K={counts[0]}, H={counts[1]}, validation failures={validation['failed_count']}",
                        flush=True,
                    )
            if decision["stop"]:
                break
        checkpoint("final.pt")
        return report
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "error": repr(error),
            "rank": rank,
            "config": asdict(config),
            "completed_updates": locals().get("report", {}).get("completed_updates", 0),
            "inputs": _cpu([vars(record) for record in (locals().get("records") or [])]),
            "active_records": _cpu([vars(record) for record in pool.records]) if pool is not None else [],
            "context_specs": step.context_specs,
            "network_state": _cpu(network.state_dict()),
            "optimizer_state": _cpu(optimizer.state_dict()),
        }
        _atomic_torch(output / f"failure_rank_{rank}.pt", failure)
        if rank == 0:
            (output / "failure.json").write_text(
                json.dumps(
                    {"error": repr(error), "completed_updates": locals().get("report", {}).get("completed_updates", 0)},
                    indent=2,
                )
            )
            if "report" in locals():
                report["status"] = "failed"
                _write_report(output, report)
        raise
    finally:
        error_in_flight = sys.exc_info()[0] is not None
        try:
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    if not error_in_flight:
                        raise
        finally:
            try:
                step.close()
            finally:
                if owned_group:
                    dist.destroy_process_group()


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--config", type=Path, help="JSON overrides for MixedTrainConfig; resume starts from saved config"
    )
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    values = {}
    if args.resume:
        import torch

        values = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
    if args.config:
        values.update(json.loads(args.config.read_text()))
    if args.max_epochs is not None:
        values["max_epochs"] = args.max_epochs
    if args.device is not None:
        values["device"] = args.device
    run_training(args.output, MixedTrainConfig(**values), resume=args.resume)


if __name__ == "__main__":
    _main()
