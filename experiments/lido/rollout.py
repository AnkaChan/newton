# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stand-alone rollout of one hex beam through `Step` (design spec 3, ROLLOUT), without Newton.

SI in and out; the batch holds the body in normalised units (h = mu = dt = 1) with positions relative to the
object's origin. The npz written by `save_npz` carries the keys the old `render_learned.py` reads
(positions, times, rest_positions, fixed_indices, cell_corner_indices, optional contact_* arrays).

    python -m experiments.lido.rollout --checkpoint ckpt.pt --steps 60 --iterations 8 --out beam.npz
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import torch

from . import contact as _contact
from . import physics, scenes
from .batch import Batch
from .capture import CapturedQuery
from .config import TrainConfig
from .fusion import Fusion
from .grid import Grid
from .network import Net
from .step import Step
from .units import energy_scale, force_scale, material_from_si

DEFAULT_SCENARIO = {
    "cell_counts": (10, 10, 40),
    "h": 0.025,
    "dt": 1.0 / 300.0,
    "pins": "zmin_face",
    "E": 1e5,
    "nu": 0.3,
    "rho": 1000.0,
    "eta": 100.0,
    "gravity": (0.0, -9.81, 0.0),
    "origin": (0.0, 0.0, 0.0),
    "contact": {},  # SceneSpec.contact: plane_present/normal/height, points, normals, radii, kappa, beta, mu_f
}

NPZ_KEYS = (
    "positions",
    "velocities",
    "times",
    "rest_positions",
    "fixed_indices",
    "cell_corner_indices",
    "energy_joule",
    "residual_n",
    "penetration_r",
    "h",
    "cell_counts",
    "dt",
    "iterations",
    "seconds_per_query",
)
CONTACT_KEYS = (
    "contact_plane_present",
    "contact_plane_point",
    "contact_plane_normal",
    "contact_point_positions",
    "contact_point_normals",
    "contact_point_radii",
)


def load_checkpoint(checkpoint: str | dict | None) -> dict | None:
    """A checkpoint dict (`network_state`, `config`, trainer keys) from a path or dict; None passes through."""
    if checkpoint is None or isinstance(checkpoint, dict):
        return checkpoint
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def load_network(
    checkpoint: str | dict | None, cfg: TrainConfig | None = None, overrides: dict | None = None, device="cpu"
) -> tuple[Net, TrainConfig]:
    """Net and TrainConfig from a checkpoint (or a fresh zero-init-headed Net when `checkpoint` is None).

    `cfg` replaces the checkpoint's config; `overrides` are applied on top of whichever config is used.
    """
    ck = load_checkpoint(checkpoint)
    if cfg is None:
        cfg = TrainConfig.from_dict(ck["config"]) if ck is not None and "config" in ck else TrainConfig()
    if overrides:
        cfg = TrainConfig.from_dict({**cfg.to_dict(), **overrides})
    net = Net.from_config(cfg)
    if ck is not None:
        net.load_state_dict(ck["network_state"])
    return net.to(device).eval(), cfg


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


COMPILE_MODE = "max-autotune-no-cudagraphs"  # the whole query is captured as one graph; inductor adds none


def compile_inference(net: Net, step: Step, fullgraph: bool = True) -> None:
    """torch.compile the static-shape chains of the rollout query: the cell-graph layer (one graph, the Warp
    attention being a custom op without autograd), the edge encoder and the node / edge feature chains."""
    net.compile_layers(edge_encoder=True, fullgraph=fullgraph, mode=COMPILE_MODE)
    step.compile_features(mode=COMPILE_MODE)


def rollout(
    checkpoint,
    cfg_overrides: dict,
    scenario: dict,
    steps: int,
    iterations: int,
    device="cuda",
    capture: bool = False,
    compile_network: bool = False,
    translation: str = "implicit_contact",
) -> dict:
    """Roll one beam `steps` physical steps with K = `iterations` queries each. Returns SI arrays (see NPZ_KEYS).

    `capture`: record one query + commit as a CUDA graph after the first prepare and replay it K times per step
    (pairs in the capacity layout; falls back to eager with a warning if the capture fails). `compile_network`:
    torch.compile the cell-graph layer for the static shapes (fuses the per-edge elementwise chains).
    `translation`: centroid target of an unpinned body (`pins="none"`), "picard" or "implicit_contact" (step.py).
    """
    sc = {**DEFAULT_SCENARIO, **(scenario or {})}
    device = torch.device(device)
    net, cfg = load_network(checkpoint, overrides=cfg_overrides, device=device)
    h, dt, K = float(sc["h"]), float(sc["dt"]), int(iterations)
    contact_spec = dict(sc.get("contact") or {})
    grid = Grid.build(sc["cell_counts"], sc["pins"], device)
    batch = Batch.build([grid], device)
    batch.material = material_from_si(
        E=sc["E"],
        nu=sc["nu"],
        rho=sc["rho"],
        eta=sc["eta"],
        gravity=sc["gravity"],
        h=h,
        dt=dt,
        cell_count=grid.C,
        sample_count=grid.S,
        kappa=contact_spec.get("kappa", 0.0),
        beta=contact_spec.get("beta", 0.0),
        mu_f=contact_spec.get("mu_f", 0.0),
        friction_epsilon=cfg.contact_friction_epsilon,
        floor_scale=cfg.energy_floor_scale,
        device=device,
    )
    batch.scene = scenes.scene_from_spec(contact_spec, h, device)
    origin = torch.tensor(sc["origin"], dtype=torch.float32, device=device)
    batch.origin = origin[None].clone()
    batch.X = grid.rest.clone()
    batch.V = torch.zeros_like(batch.X)
    batch.X_prev = batch.X.clone()
    batch.x = batch.X.clone()
    capture = bool(capture) and device.type == "cuda"
    step = Step(net, Fusion(), pair_capacity=capture, translation=translation)
    if compile_network:
        compile_inference(net, step)
    sel = torch.ones(1, dtype=torch.bool, device=device)

    P = grid.P
    positions = torch.empty(steps + 1, P, 3, device=device)
    velocities = torch.empty(steps + 1, P, 3, device=device)
    energy_joule = np.zeros(steps)
    residual_n = np.zeros(steps)
    penetration_r = np.zeros(steps)
    query_seconds = []
    e_scale, f_scale = energy_scale(batch.material), force_scale(batch.material)
    captured = None
    with torch.no_grad():
        step.prepare(batch, sel)
        if capture:
            try:
                captured = CapturedQuery(step, batch)
            except Exception as e:  # any capture failure means: run eagerly
                warnings.warn(f"rollout: CUDA-graph capture failed ({e!r}); running eagerly", stacklevel=2)
        positions[0] = origin + h * batch.X
        velocities[0] = batch.V * (h / dt)
        for s in range(steps):
            if captured is not None:
                captured.sync()  # prepare / advance rebound the candidate, the step constants and the pairs
            for _ in range(K):
                _sync(device)
                t0 = time.perf_counter()
                if captured is not None:
                    captured.replay()
                else:
                    step.commit(batch, step.query(batch))
                _sync(device)
                query_seconds.append(time.perf_counter() - t0)
            energy_joule[s] = (batch.E * e_scale).item()
            residual_n[s] = (physics.residual(batch.gX, batch) * f_scale).item()
            penetration_r[s] = _contact.penetration(batch, batch.x).item()
            step.advance(batch, sel)
            positions[s + 1] = origin + h * batch.X
            velocities[s + 1] = batch.V * (h / dt)
    timed = query_seconds[K:] if steps > 1 else query_seconds  # the first step pays warm-up (kernel loads, factor)
    seconds_per_query = float(np.mean(timed)) if timed else float("nan")

    result = {
        "positions": positions.cpu().numpy(),
        "velocities": velocities.cpu().numpy(),
        "times": np.arange(steps + 1, dtype=np.float64) * dt,
        "rest_positions": (origin + h * grid.rest).cpu().numpy(),
        "fixed_indices": grid.pinned.cpu().numpy().astype(np.int64),
        "cell_corner_indices": grid.cells.cpu().numpy().astype(np.int64),
        "energy_joule": energy_joule,
        "residual_n": residual_n,
        "penetration_r": penetration_r,
        "h": np.float64(h),
        "cell_counts": np.asarray(grid.cell_counts, dtype=np.int64),
        "dt": np.float64(dt),
        "iterations": np.int64(K),
        "seconds_per_query": np.float64(seconds_per_query),
        "seconds_per_step": np.float64(seconds_per_query * K),
        "captured": captured is not None,
        "compiled": bool(compile_network),
        "config": cfg.to_dict(),
        "scenario": sc,
    }
    result.update(_contact_arrays(contact_spec, origin.cpu().numpy()))
    return result


def _contact_arrays(spec: dict, origin: np.ndarray) -> dict:
    """The six renderer contact keys (SI world) when the scenario has a plane or static points; empty otherwise."""
    points = np.asarray(spec.get("points", []), dtype=np.float32).reshape(-1, 3)
    present = bool(spec.get("plane_present", False))
    if not present and points.shape[0] == 0:
        return {}
    n = np.asarray(spec.get("plane_normal", scenes.PLANE_NORMAL), dtype=np.float32)
    return {
        "contact_plane_present": np.asarray([present], dtype=np.bool_),
        "contact_plane_point": (origin + n * float(spec.get("plane_height", 0.0))).astype(np.float32),
        "contact_plane_normal": n,
        "contact_point_positions": (points + origin).astype(np.float32),
        "contact_point_normals": np.asarray(spec.get("normals", []), dtype=np.float32).reshape(-1, 3),
        "contact_point_radii": np.asarray(spec.get("radii", []), dtype=np.float32).reshape(-1),
    }


def save_npz(result: dict, path: str) -> None:
    """Write the array entries of a rollout result (NPZ_KEYS, seconds_per_step and any contact_* keys)."""
    keys = [*NPZ_KEYS, "seconds_per_step", *[k for k in CONTACT_KEYS if k in result]]
    np.savez(path, **{k: result[k] for k in keys})


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="LIDO rollout of one hex beam (no Newton)")
    p.add_argument("--checkpoint", default=None, help="checkpoint path; omitted = fresh network (candidate stays at Y)")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--out", default=None, help="npz path readable by render_learned.py")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cell-counts", type=int, nargs=3, default=list(DEFAULT_SCENARIO["cell_counts"]))
    p.add_argument("--h", type=float, default=DEFAULT_SCENARIO["h"])
    p.add_argument("--dt", type=float, default=DEFAULT_SCENARIO["dt"])
    p.add_argument("--E", type=float, default=DEFAULT_SCENARIO["E"])
    p.add_argument("--nu", type=float, default=DEFAULT_SCENARIO["nu"])
    p.add_argument("--rho", type=float, default=DEFAULT_SCENARIO["rho"])
    p.add_argument("--eta", type=float, default=DEFAULT_SCENARIO["eta"])
    p.add_argument("--gravity", type=float, nargs=3, default=list(DEFAULT_SCENARIO["gravity"]))
    p.add_argument(
        "--plane-height", type=float, default=None, help="ground plane y below the rest origin [m]; omitted = no plane"
    )
    p.add_argument("--kappa", type=float, default=100.0, help="contact stiffness ratio ke / (E h)")
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--mu-f", type=float, default=0.0)
    p.add_argument("--capture", action="store_true", help="replay one CUDA-graph-captured query K times per step")
    p.add_argument("--compile-network", action="store_true", help="torch.compile the cell-graph layer")
    p.add_argument("--pins", default=DEFAULT_SCENARIO["pins"], choices=["zmin_face", "none"])
    p.add_argument(
        "--translation",
        default="implicit_contact",
        choices=["picard", "implicit_contact"],
        help="free-body centroid target",
    )
    a = p.parse_args(argv)
    scenario = {
        "cell_counts": tuple(a.cell_counts),
        "h": a.h,
        "dt": a.dt,
        "pins": a.pins,
        "E": a.E,
        "nu": a.nu,
        "rho": a.rho,
        "eta": a.eta,
        "gravity": tuple(a.gravity),
    }
    if a.plane_height is not None:
        scenario["contact"] = {
            "plane_present": True,
            "plane_height": a.plane_height,
            "kappa": a.kappa,
            "beta": a.beta,
            "mu_f": a.mu_f,
        }
    result = rollout(
        a.checkpoint, {}, scenario, a.steps, a.iterations, a.device, a.capture, a.compile_network, a.translation
    )
    mode = ("captured" if result["captured"] else "eager") + (", compiled" if a.compile_network else "")
    print(
        f"steps {a.steps} K {a.iterations} device {a.device} ({mode}): {result['seconds_per_query'] * 1e3:.2f} ms/query, "
        f"{result['seconds_per_step'] * 1e3:.1f} ms/step; final E {result['energy_joule'][-1]:.4g} J, "
        f"residual {result['residual_n'][-1]:.4g} N, penetration {result['penetration_r'][-1]:.3f} r"
    )
    if a.out:
        save_npz(result, a.out)
        print(f"wrote {a.out}")
    return result


if __name__ == "__main__":
    main()
