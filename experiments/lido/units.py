# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SI <-> normalised units (h = mu = dt = 1) and the material record (normalisation note of 2026-09-27)."""

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
    device="cpu",
) -> Material:
    """One object's material in normalised units. `kappa` is the contact stiffness ratio ke / (E h)."""
    mu, lam = lame(E, nu)
    ke = kappa * E * h  # N/m
    kd = beta * ke * dt
    g = [float(v) for v in gravity]
    volume = cell_count * h**3
    r = 0.5 * h
    floor_si = (
        floor_scale * EPS32 * (volume * (lam + 2.0 * mu + eta / dt + rho * h**2 / dt**2) + ke * r**2 * sample_count)
    )
    scale = mu * h**3
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
        "energy_scale": scale,
        "friction_epsilon": friction_epsilon,
    }
    t = lambda v: torch.tensor([float(v)], dtype=torch.float32, device=device)  # noqa: E731
    return Material(
        lam=t(lam / mu),
        rho=t(rho * h**2 / (mu * dt**2)),
        eta=t(eta / (mu * dt)),
        g=torch.tensor([[gi * dt**2 / h for gi in g]], dtype=torch.float32, device=device),
        ke=t(ke / (mu * h)),
        kd=t(kd / (mu * h * dt)),
        mu_f=t(mu_f),
        kappa=t(kappa),
        beta=t(beta),
        friction_eps=t(friction_epsilon * dt / h),
        floor=t(floor_si / scale),
        h=t(h),
        dt=t(dt),
        mu=t(mu),
        si=[si],
    )


def conditioning(m: Material) -> torch.Tensor:
    """The 7 dimensionless FiLM channels [O,7]."""
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


def energy_scale(m: Material) -> torch.Tensor:
    """mu h^3 per object [O]: normalised energy times this is joules."""
    return m.mu * m.h**3


def force_scale(m: Material) -> torch.Tensor:
    """mu h^2 per object [O]: normalised gradient times this is newtons."""
    return m.mu * m.h**2


def log_uniform(rng: torch.Generator, lo: float, hi: float, device="cpu") -> float:
    u = torch.rand((), generator=rng, device=device).item()
    return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
