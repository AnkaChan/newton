# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train a bounded float32 single-update learned hex optimizer experiment.

Experimental: this trains independent implicit-Euler optimization queries,
not physical rollouts. Original physical Y stays fixed while candidate shapes
vary. A learned invalid proposal stops training; it is never repaired.
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import generate_cuboid
from .multiscale import generate_multiscale, interpolate_control_grid

__all__ = ["TrainSmokeConfig", "run_training"]


@dataclass(frozen=True)
class TrainSmokeConfig:
    """Experimental immutable training settings; lengths are meters and times seconds."""

    updates: int = 64
    cell_counts: tuple[int, int, int] = (10, 10, 40)
    cell_size: float = 0.025
    lame_lambda: float = 288461.53846
    lame_mu: float = 192307.69231
    density: float = 1000.0
    time_step: float = 1 / 300
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    learning_rate: float = 1e-4
    max_step_size: float = 0.05
    hidden_dim: int = 128
    edge_hidden_dim: int = 64
    num_heads: int = 4
    hops: tuple[int, ...] = (1, 1, 1)
    query_chunk_size: int = 128
    train_count: int = 16
    validation_count: int = 8
    train_seed_start: int = 0
    validation_seed_start: int = 10000
    seed: int = 73
    validation_interval: int = 8
    checkpoint_interval: int = 16
    cpu_threads: int = 1
    device: str = "cuda"
    verbose: bool = True

    def __post_init__(self):
        """Validate the disjoint dataset pools and positive numerical settings."""
        for name in ("cell_counts", "hops", "gravity"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in (
            "updates",
            "hidden_dim",
            "edge_hidden_dim",
            "num_heads",
            "query_chunk_size",
            "train_count",
            "validation_count",
            "validation_interval",
            "checkpoint_interval",
            "cpu_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "train_seed_start", "validation_seed_start"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if len(self.cell_counts) != 3 or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in self.cell_counts
        ):
            raise ValueError("cell_counts must have three positive integers")
        if not self.hops or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in self.hops):
            raise ValueError("hops must contain positive integers")
        for name in ("cell_size", "lame_mu", "density", "time_step", "learning_rate", "max_step_size"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.lame_lambda) or self.lame_lambda < 0:
            raise ValueError("lame_lambda must be finite and nonnegative")
        if len(self.gravity) != 3 or not all(math.isfinite(value) for value in self.gravity):
            raise ValueError("gravity must be a finite three-vector")
        if set(self.train_seeds) & set(self.validation_seeds):
            raise ValueError("training and validation physical seeds must be disjoint")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

    @property
    def train_seeds(self):
        """Return the fixed physical training seed pool."""
        return tuple(range(self.train_seed_start, self.train_seed_start + self.train_count))

    @property
    def validation_seeds(self):
        """Return the disjoint fixed physical validation seed pool."""
        return tuple(range(self.validation_seed_start, self.validation_seed_start + self.validation_count))


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _screen(problem, positions):
    import torch

    with torch.no_grad():
        if not torch.isfinite(positions).all().item():
            return {"valid": False, "min_gauss_j": None, "min_center_j": None, "min_center_singular_ratio": None}
        corners = positions[:, problem.optimizer.energy.cell_corner_indices]
        relative = corners - corners[:, :, :1]
        gauss = torch.einsum("bcki,qkj->bcqij", relative, problem.optimizer.energy.shape_gradients)
        center = torch.einsum("bcki,kj->bcij", relative, problem.optimizer.center_gradients)
        gauss_j = torch.linalg.det(gauss)
        center_j = torch.linalg.det(center)
        singular = torch.linalg.svdvals(center)
        ratio = singular[..., -1] / singular[..., 0].clamp_min(1)
        valid = (
            torch.isfinite(gauss_j).all()
            & torch.isfinite(center_j).all()
            & (gauss_j > 0).all()
            & (center_j > 0).all()
            & (ratio > 4 * torch.finfo(positions.dtype).eps).all()
        )
        return {
            "valid": bool(valid.item()),
            "min_gauss_j": float(gauss_j.min()),
            "min_center_j": float(center_j.min()),
            "min_center_singular_ratio": float(ratio.min()),
        }


def _rms(values):
    return float(np.sqrt(np.mean(np.sum(np.asarray(values, dtype=np.float64) ** 2, axis=-1))))


class _Sampler:
    """Cache fixed physical problems while sampling only optimizer candidates."""

    def __init__(self, config, rest, model, solver, saved=None):
        import torch

        self.config, self.rest, self.solver = config, rest, solver
        self.rng = np.random.default_rng(np.random.SeedSequence([config.seed, 811]))
        self.order, self.cursor = [], 0
        self.physical, self.problems, self.validation = {}, {}, []
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
        state = model.state()
        for seed in (*config.train_seeds, *config.validation_seeds):
            record = copy.deepcopy(saved["physical_samples"][seed]) if saved else self._physical(seed, fixed)
            state.particle_q.assign(record["positions"])
            state.particle_qd.assign(record["velocity"])
            state.particle_f.zero_()
            problem = solver.prepare_problem(state, config.time_step)
            screen = _screen(problem, problem.previous_positions)
            if not screen["valid"]:
                raise ValueError(f"Physical seed {seed} is invalid after float32 quantization: {screen}")
            record["screen"] = screen
            self.physical[seed], self.problems[seed] = record, problem
        if saved:
            for record in saved["validation_candidates"]:
                device = self.problems[record["metadata"]["physical_seed"]].previous_positions.device
                self.validation.append(
                    {
                        "positions": torch.from_numpy(record["positions"].copy()).to(device),
                        "metadata": copy.deepcopy(record["metadata"]),
                    }
                )
        else:
            for seed in config.validation_seeds:
                rng = np.random.default_rng(np.random.SeedSequence([config.seed, seed, 919]))
                candidate, metadata = self.candidate(seed, rng)
                self.validation.append({"positions": candidate, "metadata": metadata})
        # Do this after deterministic physical/cache reconstruction.
        if saved:
            self.load_state_dict(saved["sampler_state"])

    def _physical(self, seed, fixed):
        config, rest = self.config, self.rest
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, seed, 701]))
        strength = float(rng.uniform(0.02, 0.1))
        sample = generate_multiscale(rest, seed=seed, strength=strength)
        positions = sample.positions.astype(np.float32)
        positions[fixed] = rest.corner_rest_positions[fixed].astype(np.float32)
        counts = tuple(min(n + 1, cap) for n, cap in zip(rest.cell_counts, (3, 3, 5), strict=True))
        controls = rng.uniform(-1, 1, size=(*counts, 3))
        controls[:, :, 0] = 0
        velocity = interpolate_control_grid(
            controls,
            rest.corner_rest_positions,
            origin=rest.corner_rest_positions[0],
            extent=np.asarray(rest.cell_counts) * rest.cell_size,
        )
        velocity[fixed] = 0
        target_displacement_rms = float(rng.uniform(0, 0.1) * rest.cell_size)
        norm = _rms(velocity)
        velocity *= target_displacement_rms / (config.time_step * norm) if norm else 0
        velocity = velocity.astype(np.float32)
        velocity[fixed] = 0
        return {
            "positions": positions,
            "velocity": velocity,
            "metadata": {
                "physical_seed": seed,
                "strength": strength,
                "augmentation_scale": sample.effective_scale,
                "velocity_dt_rms_m": _rms(config.time_step * velocity),
                "requested_velocity_dt_rms_m": target_displacement_rms,
            },
        }

    def candidate(self, seed, rng):
        import torch

        config, problem = self.config, self.problems[seed]
        value = float(rng.random())
        mode = (
            "inertial" if value < 0.5 else "noisy_inertial" if value < 0.85 else "previous" if value < 0.95 else "rigid"
        )
        noise_seed = int(rng.integers(0, 2**32))
        requested_noise_rms = float(rng.uniform(0.01, 0.1) * config.cell_size)
        base = problem.inertial_prediction.clone()
        base[:, problem.fixed_indices] = problem.fixed_positions
        base_fallback = not _screen(problem, base)["valid"]
        if base_fallback:
            base = problem.previous_positions.clone()
        candidate = base.clone()
        halvings, scale = 0, 1.0
        fallback = False
        if mode == "previous":
            candidate = problem.previous_positions.clone()
        elif mode == "rigid":
            with torch.no_grad():
                candidate = self.solver.initialize_candidate(problem).detach()
            if not _screen(problem, candidate)["valid"]:
                candidate = problem.previous_positions.clone()
                fallback = True
        elif mode == "noisy_inertial":
            sample = generate_multiscale(self.rest, seed=noise_seed, strength=0.1)
            noise = sample.positions - self.rest.corner_rest_positions
            norm = _rms(noise)
            noise *= requested_noise_rms / norm if norm else 0
            noise = torch.from_numpy(noise.astype(np.float32))[None].to(base.device)
            noise[:, problem.fixed_indices] = 0
            while True:
                candidate = base + scale * noise
                candidate[:, problem.fixed_indices] = problem.fixed_positions
                if _screen(problem, candidate)["valid"]:
                    break
                halvings += 1
                scale *= 0.5
                if halvings == 32:
                    candidate, scale, fallback = base.clone(), 0.0, True
                    break
        candidate[:, problem.fixed_indices] = problem.fixed_positions
        screen = _screen(problem, candidate)
        if not screen["valid"]:
            raise ValueError(f"Initializer for physical seed {seed} remains invalid: {screen}")
        metadata = {
            "physical_seed": seed,
            "candidate_mode": mode,
            "noise_seed": noise_seed,
            "requested_noise_rms_m": requested_noise_rms if mode == "noisy_inertial" else 0,
            "noise_scale": scale if mode == "noisy_inertial" else 0,
            "noise_halvings": halvings,
            "inertial_base_fallback": base_fallback,
            "initializer_fallback": fallback,
            "candidate_from_previous_rms_m": _rms((candidate - problem.previous_positions).detach().cpu().numpy()[0]),
            **screen,
        }
        return candidate.detach(), metadata

    def next_training(self):
        if self.cursor == len(self.order):
            self.order = [int(seed) for seed in self.rng.permutation(self.config.train_seeds)]
            self.cursor = 0
        seed = self.order[self.cursor]
        self.cursor += 1
        candidate, metadata = self.candidate(seed, self.rng)
        return self.problems[seed], candidate, metadata

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "order": self.order.copy(), "cursor": self.cursor}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.order, self.cursor = list(state["order"]), int(state["cursor"])

    def saved_validation(self):
        return [
            {
                "positions": entry["positions"].detach().cpu().numpy().copy(),
                "metadata": copy.deepcopy(entry["metadata"]),
            }
            for entry in self.validation
        ]


def _metrics(before, after):
    return {
        "energy_before_joule": before,
        "energy_after_joule": after,
        "normalized_loss": (after - before) / max(before, 1.0),
        "energy_ratio": after / before if before > 0 else None,
        "descent": after < before,
    }


def _evaluate(solver, sampler, updates):
    import torch

    cases = []
    solver.network.eval()
    with torch.no_grad():
        for entry in sampler.validation:
            metadata, candidate = entry["metadata"], entry["positions"]
            problem = sampler.problems[metadata["physical_seed"]]
            record = {**metadata}
            try:
                before = float(problem.objective(candidate).total.item())
                proposal = solver.propose_update(candidate, problem)
                after = float(proposal.loss.total.item())
                screen = _screen(problem, proposal.positions)
                if not math.isfinite(after) or not screen["valid"]:
                    raise ValueError("Invalid learned validation output")
                record.update(status="complete", **_metrics(before, after))
            except (ValueError, RuntimeError) as exc:
                record.update(status="failed", error=str(exc))
            cases.append(record)
    valid = [case for case in cases if case["status"] == "complete"]
    result = {
        "optimizer_updates": updates,
        "cases": cases,
        "valid_count": len(valid),
        "failed_count": len(cases) - len(valid),
    }
    for target, source in (
        ("mean_normalized_loss", "normalized_loss"),
        ("mean_before_joule", "energy_before_joule"),
        ("mean_after_joule", "energy_after_joule"),
    ):
        result[target] = float(np.mean([case[source] for case in valid])) if valid else None
    ratios = [case["energy_ratio"] for case in valid if case["energy_ratio"] is not None]
    result["mean_energy_ratio"] = float(np.mean(ratios)) if ratios else None
    result["descent_rate"] = sum(case.get("descent", False) for case in cases) / len(cases)
    solver.network.train()
    return result


def _write_csv(path, records):
    keys = list(dict.fromkeys(key for record in records for key in record if key != "cases"))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: value for key, value in record.items() if key != "cases"} for record in records)


def _write_report(output, report):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _atomic_json(output / "report.json", report)
    _write_csv(output / "training.csv", report["training"])
    _write_csv(output / "validation.csv", report["validation"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    training = [entry for entry in report["training"] if entry["status"] == "complete"]
    validation = [entry for entry in report["validation"] if entry["mean_normalized_loss"] is not None]
    axes[0].plot(
        [row["optimizer_updates"] for row in training],
        [row["normalized_loss"] for row in training],
        color="#7ca8bc",
        alpha=0.7,
        label="Training query (before optimizer.step)",
    )
    axes[0].plot(
        [row["optimizer_updates"] for row in validation],
        [row["mean_normalized_loss"] for row in validation],
        "o-",
        color="#c7793c",
        label="Fixed held-out queries (after update)",
    )
    axes[0].axhline(0, color="#8999a4", linewidth=0.7)
    axes[0].set(xlabel="Completed parameter updates", ylabel="(E after - E before) / max(E before, 1 J)")
    axes[0].legend(fontsize=8)
    for key, label, color in (
        ("mean_before_joule", "Before learned query", "#7ca8bc"),
        ("mean_after_joule", "After learned query", "#c7793c"),
    ):
        axes[1].plot(
            [row["optimizer_updates"] for row in validation],
            [row[key] for row in validation],
            "o-",
            color=color,
            label=label,
        )
    axes[1].set(xlabel="Completed parameter updates", ylabel="Held-out mean physical energy (J)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "loss_curve.png", dpi=150)
    fig.savefig(output / "loss_curve.svg")
    plt.close(fig)
    links = " ".join(
        f'<a href="checkpoints/{html.escape(path.name)}">{html.escape(path.name)}</a>'
        for path in sorted((output / "checkpoints").glob("*.pt"))
    )
    last = report["validation"][-1] if report["validation"] else {}
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Learned hex optimizer training smoke run</title><style>body{{font:16px/1.5 system-ui;margin:2em auto;max-width:1100px;padding:0 1em;color:#203b4c}}img{{width:100%}}a{{margin-right:1em}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f7;padding:1em}}small{{color:#597181}}</style>
<h1>Single-update learned hex optimizer</h1><p>Status: <b>{html.escape(report["status"])}</b> · {report["optimizer_updates"]} actual Adam updates · {report["failure_count"]} training failures.</p>
<p>Float32 training on {html.escape(report["device"])}. The network, features, loss, and Adam use that device; the current SciPy sparse factorization and solves plus native model sampling use CPU. Each physical problem keeps its original implicit-Euler target Y. Candidate initialization varies; learned outputs receive no line search or repair. These independent optimization queries are not simulation rollouts.</p>
<img src="loss_curve.png" alt="Training and held-out normalized energy improvements and held-out physical energy">
<p>Lower normalized loss means more improvement. The training curve evaluates a fresh query using parameters before that row's optimizer update; validation uses fixed held-out queries after the indicated updates. The initial checkpoint has zero parameter updates. Invalid validation cases are counted separately, not included as successes.</p>
<h2>Latest fixed validation</h2><pre>{html.escape(json.dumps({key: value for key, value in last.items() if key != "cases"}, indent=2))}</pre>
<p><a href="training.csv">Training CSV</a><a href="validation.csv">Validation CSV</a><a href="report.json">Full report</a><a href="loss_curve.svg">SVG curve</a><a href="config.json">Configuration</a></p>
<h2>Checkpoints</h2><p>{links}</p><small>Checkpoints include weights, Adam state, completed updates, full configuration, sample identities and arrays, fixed validation candidates, and NumPy/Torch RNG state. Load only trusted PyTorch checkpoint files.</small>
<h2>Configuration</h2><pre>{html.escape(json.dumps(report["config"], indent=2))}</pre></html>"""
    (output / "index.html").write_text(page)


def run_training(output: Path, config: TrainSmokeConfig, *, resume: Path | None = None) -> dict:
    """Run actual Adam updates and persist resumable checkpoints and curves.

    Resume permits changing only the requested final update count and verbosity.
    Reconstruct caches before restoring RNG. Failure checkpoints are diagnostic;
    start a separately configured run to address an invalid learned proposal.
    """
    import torch

    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("training device must be cpu or cuda")
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    previous_threads = torch.get_num_threads()
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.set_num_threads(config.cpu_threads)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(config.seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(config.seed)
            with torch.autocast(device_type=device.type, enabled=False):
                return _run_training(Path(output), config, resume=resume)
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
        torch.set_num_threads(previous_threads)


def _run_training(output, config, *, resume):
    import torch

    from .network import IntrinsicSolverNetwork  # noqa: PLC0415 - Optional training boundary.
    from .newton_model import build_newton_hex_model  # noqa: PLC0415
    from .newton_solver import SolverLearnedIntrinsic  # noqa: PLC0415

    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if saved:
        if saved["status"] == "failed":
            raise ValueError("A failure checkpoint is diagnostic; begin a separately configured run")
        current, prior = asdict(config), saved["config"].copy()
        for key in ("updates", "verbose"):
            current.pop(key)
            prior.pop(key)
        if current != prior:
            raise ValueError("resume configuration differs beyond updates/verbosity")
        if config.updates < saved["optimizer_updates"]:
            raise ValueError("requested final updates precede checkpoint's completed updates")
    rest = generate_cuboid(config.cell_counts, cell_size=config.cell_size)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
    model = build_newton_hex_model(
        rest,
        fixed,
        lame_lambda=config.lame_lambda,
        lame_mu=config.lame_mu,
        density=config.density,
        gravity=config.gravity,
    )
    network = IntrinsicSolverNetwork(
        config.cell_counts,
        38,
        hidden_dim=config.hidden_dim,
        edge_hidden_dim=config.edge_hidden_dim,
        num_heads=config.num_heads,
        hops=config.hops,
        max_step_size=config.max_step_size,
        query_chunk_size=config.query_chunk_size,
    ).to(config.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    solver = SolverLearnedIntrinsic(model, network=network, iterations=1)
    sampler = _Sampler(config, rest, model, solver, saved=saved)
    completed = 0
    training, validation = [], []
    best = math.inf
    if saved:
        network.load_state_dict(saved["network_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        completed = int(saved["optimizer_updates"])
        training, validation = copy.deepcopy(saved["training"]), copy.deepcopy(saved["validation"])
        best = saved["best_validation_loss"]
        torch.set_rng_state(saved["torch_rng_state"])
        if saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device=config.device)
    status, failure = "running", None
    start = time.perf_counter()
    elapsed_prior = saved.get("elapsed_seconds", 0) if saved else 0
    _atomic_json(output / "config.json", asdict(config))

    def checkpoint(name):
        payload = {
            "schema_version": 1,
            "status": status,
            "optimizer_updates": completed,
            "config": asdict(config),
            "network_state": network.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "sampler_state": sampler.state_dict(),
            "cuda_rng_state": torch.cuda.get_rng_state(config.device)
            if torch.device(config.device).type == "cuda"
            else None,
            "physical_samples": sampler.physical,
            "validation_candidates": sampler.saved_validation(),
            "train_seeds": list(config.train_seeds),
            "validation_seeds": list(config.validation_seeds),
            "training": training,
            "validation": validation,
            "best_validation_loss": best,
            "failure": failure,
            "elapsed_seconds": elapsed_prior + time.perf_counter() - start,
        }
        path = output / "checkpoints" / name
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def report():
        return {
            "status": status,
            "optimizer_updates": completed,
            "requested_updates": config.updates,
            "failure_count": int(failure is not None),
            "failure": failure,
            "config": asdict(config),
            "train_seeds": list(config.train_seeds),
            "validation_seeds": list(config.validation_seeds),
            "training": training,
            "validation": validation,
            "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
            "elapsed_seconds": elapsed_prior + time.perf_counter() - start,
            "working_dtype": "float32",
            "device": str(next(network.parameters()).device),
            "tf32_enabled": False,
            "autocast_enabled": False,
            "gpu_name": torch.cuda.get_device_name(config.device)
            if torch.device(config.device).type == "cuda"
            else None,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(config.device)
            if torch.device(config.device).type == "cuda"
            else 0,
            "optimizer_iterations_per_query": 1,
            "compute_partition": "Network/features/energy/Adam on selected Torch device; SciPy sparse forward/adjoint and native model sampling on CPU",
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "candidate_probabilities": {"inertial": 0.5, "noisy_inertial": 0.35, "previous": 0.1, "rigid": 0.05},
            "normalization": "(E_after - E_before.detach()) / max(E_before.detach(), 1 joule)",
            "physical_samples": [
                record["metadata"] | {"screen": record["screen"]} for record in sampler.physical.values()
            ],
            "initialization_policy": "Check Gauss J and center polar validity in float32; pin-correct Y, fallback to Xn if invalid; halve only initializer noise. Invalid rigid initializer also falls back to Xn and is recorded.",
            "failure_policy": "Stop and save on invalid/nonfinite learned training output or gradient. No learned-output repair. Validation failures are counted separately.",
        }

    def validate():
        nonlocal best
        value = _evaluate(solver, sampler, completed)
        validation.append(value)
        if value["failed_count"] == 0 and value["mean_normalized_loss"] < best:
            best = value["mean_normalized_loss"]
            checkpoint("best_validation.pt")
        if config.verbose:
            print(
                f"validation updates={completed} loss={value['mean_normalized_loss']} failures={value['failed_count']}",
                flush=True,
            )

    if not saved:
        validate()
        checkpoint("initial.pt")
    else:
        checkpoint("resume_start.pt")
    _write_report(output, report())
    network.train()
    while completed < config.updates:
        row = {"attempt": completed + 1, "optimizer_updates": completed}
        try:
            problem, candidate, metadata = sampler.next_training()
            row.update(metadata)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                before = problem.objective(candidate).total.detach()
            proposal = solver.propose_update(candidate, problem)
            screen = _screen(problem, proposal.positions)
            if not screen["valid"] or not torch.isfinite(proposal.loss.total).all().item():
                raise ValueError(f"Invalid learned training output: {screen}")
            normalized = (proposal.loss.total - before) / before.clamp_min(1.0)
            normalized.mean().backward()
            gradients = [parameter.grad for parameter in network.parameters() if parameter.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
                raise ValueError("Nonfinite or absent training parameter gradients")
            gradient_norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients)).item()
            if not math.isfinite(gradient_norm):
                raise ValueError("Nonfinite training gradient norm")
            row.update(_metrics(float(before.item()), float(proposal.loss.total.detach().item())))
            row["gradient_norm"] = gradient_norm
            row["nonzero_gradient_tensors"] = sum(bool((gradient != 0).any()) for gradient in gradients)
            optimizer.step()
            completed += 1
            row["optimizer_updates"] = completed
            if not all(torch.isfinite(parameter).all().item() for parameter in network.parameters()):
                raise ValueError("Optimizer produced nonfinite parameters")
            row.update(status="complete", elapsed_seconds=elapsed_prior + time.perf_counter() - start)
            training.append(row)
            _write_csv(output / "training.csv", training)
            if config.verbose:
                print(
                    f"update {completed}/{config.updates} seed={row['physical_seed']} mode={row['candidate_mode']} loss={row['normalized_loss']:.6g} E={row['energy_before_joule']:.6g}->{row['energy_after_joule']:.6g} J",
                    flush=True,
                )
            if completed % config.validation_interval == 0 or completed == config.updates:
                validate()
                _write_report(output, report())
            if completed % config.checkpoint_interval == 0:
                checkpoint(f"step_{completed:04d}.pt")
        except (ValueError, RuntimeError) as exc:
            row.update(status="failed", error=str(exc), optimizer_updates=completed)
            training.append(row)
            failure, status = row, "failed"
            checkpoint("failure.pt")
            break
    if status != "failed":
        status = "complete"
    checkpoint("final.pt")
    result = report()
    _write_report(output, result)
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated" / "training_smoke")
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-step-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--cell-counts", nargs=3, type=int, default=(10, 10, 40))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--edge-hidden-dim", type=int, default=64)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--validation-interval", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", default="cuda", help="Training device; cuda by default, cpu for small tests")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    values = vars(args).copy()
    output, resume = values.pop("output"), values.pop("resume")
    result = run_training(output, TrainSmokeConfig(**values), resume=resume)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "optimizer_updates", "failure_count", "elapsed_seconds")}, indent=2
        )
    )
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
