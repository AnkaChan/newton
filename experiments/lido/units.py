# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SI <-> normalised units (h = dt = 1, mu = the body's own or a scene's common reference modulus) and the material
record (normalisation note of 2026-09-27; common normaliser per scene, section 11 of the design spec)."""

from __future__ import annotations

import math

import torch

from .structs import Material

EPS32 = float(torch.finfo(torch.float32).eps)


def lame(E: float, nu: float) -> tuple[float, float]:
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def material_from_si(
    *,
    E: float,
    nu: float,
    rho: float,
    eta: float,
    gravity,
    h: float,
    dt: float,
    cell_count: int,
    sample_count: int,
    kappa: float = 0.0,
    beta: float = 0.0,
    mu_f: float = 0.0,
    friction_epsilon: float = 0.01,
    floor_scale: float = 1.0,
    physical_floor: bool = True,
    device="cpu",
    mu_ref: float | None = None,
    dtype=torch.float32,
) -> Material:
    """One object's material in normalised units. `kappa` is the contact stiffness ratio ke / (E h).

    `mu_ref` (Pa) selects the unit of energy mu_ref h^3 shared by the bodies of one scene (`Material.mu_norm`;
    `reference_modulus` gives the geometric mean of a scene's moduli): the contact constants and the energy floor
    are divided by it and `mu_scale` = mu / mu_ref carries the body's elastic, damping and inertia terms into that
    unit; `lam`, `rho`, `eta` stay the body's own dimensionless groups. Without it (body mode) the body's own mu is
    the unit and every field is as before (mu_scale = 1). `dtype` of the tensors (float64 for the CPU reference
    paths: the values are then exact, not float32 roundings cast up).
    """
    mu, lam = lame(E, nu)
    mu_n = mu if mu_ref is None else float(mu_ref)
    ke = kappa * E * h  # N/m
    kd = beta * ke * dt
    g = [float(v) for v in gravity]
    volume = cell_count * h**3
    r = 0.5 * h
    floor_si = (
        floor_scale * EPS32 * (volume * (lam + 2.0 * mu + eta / dt + rho * h**2 / dt**2) + ke * r**2 * sample_count)
    )
    if physical_floor:
        # the loss scale of a body at rest: the work of lifting it by one cell (Anka, 2026-10-02), so a bad proposal on
        # a resting body is measured against something the body can do, not against float32 roundoff
        g_norm = sum(v * v for v in g) ** 0.5
        floor_si = max(floor_si, rho * volume * g_norm * h)
    scale = mu_n * h**3
    si = {
        "E": E,
        "nu": nu,
        "rho": rho,
        "eta": eta,
        "gravity": g,
        "h": h,
        "dt": dt,
        "mu": mu,
        "lam": lam,
        "ke": ke,
        "kd": kd,
        "mu_f": mu_f,
        "kappa": kappa,
        "beta": beta,
        "floor": floor_si,
        "physical_floor": physical_floor,
        "energy_scale": scale,
        "mu_norm": mu_n,
        "mu_scale": mu / mu_n,
        "friction_epsilon": friction_epsilon,
    }
    t = lambda v: torch.tensor([float(v)], dtype=dtype, device=device)  # noqa: E731
    return Material(
        lam=t(lam / mu),
        rho=t(rho * h**2 / (mu * dt**2)),
        eta=t(eta / (mu * dt)),
        g=torch.tensor([[gi * dt**2 / h for gi in g]], dtype=dtype, device=device),
        ke=t(ke / (mu_n * h)),
        kd=t(kd / (mu_n * h * dt)),
        mu_f=t(mu_f),
        kappa=t(kappa),
        beta=t(beta),
        friction_eps=t(friction_epsilon * dt / h),
        floor=t(floor_si / scale),
        h=t(h),
        dt=t(dt),
        mu=t(mu),
        si=[si],
        mu_scale=t(mu / mu_n),
        mu_norm=t(mu_n),
    )


def reference_modulus(materials_si) -> float:
    """The common unit modulus of a scene: the geometric mean of the bodies' shear moduli (dicts with E, nu)."""
    logs = [math.log(lame(float(m["E"]), float(m["nu"]))[0]) for m in materials_si]
    return math.exp(sum(logs) / len(logs))


def conditioning(m: Material) -> torch.Tensor:
    """The 7 dimensionless FiLM channels [O,7]: the body's own groups (lam, rho, eta are stored in them, whatever
    the scene's unit modulus)."""
    return torch.stack(
        [
            torch.log1p(m.lam),
            torch.log(m.rho),
            torch.log1p(m.g.norm(dim=-1)),
            torch.log1p(m.eta),
            torch.log1p(m.kappa),
            m.beta,
            m.mu_f,
        ],
        -1,
    )


def unit_rho(m: Material) -> torch.Tensor:
    """rho in the object's unit of energy [O]: mu_scale times the body's own group rho h^2 / (mu dt^2)."""
    return m.rho * m.mu_scale.to(m.rho.dtype)


def energy_scale(m: Material) -> torch.Tensor:
    """mu_norm h^3 per object [O]: normalised energy times this is joules."""
    return m.mu_norm * m.h**3


def force_scale(m: Material) -> torch.Tensor:
    """mu_norm h^2 per object [O]: normalised gradient times this is newtons."""
    return m.mu_norm * m.h**2


def log_uniform(rng: torch.Generator, lo: float, hi: float, device="cpu") -> float:
    u = torch.rand((), generator=rng, device=device).item()
    return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
