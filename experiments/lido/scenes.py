# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact scenes (contact note section 7 with its amendments; normalised-units note): a ground plane and static
points per object, sampled in SI and stored in the SceneSpec; converted to the object's cell units for the solver.

SI positions in the spec are relative to the object's origin (the rest corner (0,0,0)): normalised = SI / h.
Gravity acts along -y, so the plane is the y-plane n = (0, 1, 0) at `plane_height` metres below the rest
y-minimum; its normalised offset is plane_d = plane_height / h (signed distance of x is y - plane_d).
"""

from __future__ import annotations

import math

import torch

from . import hex as hx
from .contact import R_SAMPLE
from .structs import ContactScene
from .units import log_uniform

PLANE_NORMAL = (0.0, 1.0, 0.0)
SHELL_XZ = 0.10  # m: the point box extends the rest body by this in x and z
SHELL_Y = 0.35  # m: and by this in -y
NORMAL_COS = math.cos(math.radians(60.0))  # the point normal must oppose the nearest face normal within 60 deg
MAX_NORMAL_DRAWS = 8
MAX_POSITION_DRAWS = 1000


def _uniform(rng: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand((), generator=rng).item()


def contact_stiffness_floor(rho: float, g: float, h: float, cell_count: int, n_face: int, penetration_max) -> float:
    """ke >= m g / (n_face d_max): m = rho C h^3, d_max = penetration_max r (note section 7, amendment 2026-09-29)."""
    if penetration_max is None:
        return 0.0
    return rho * cell_count * h**3 * g / (n_face * penetration_max * R_SAMPLE * h)


def _rest_candidate(p: torch.Tensor, n: torch.Tensor, r_p: float, xs: torch.Tensor) -> bool:
    """Would any rest sample be a detection candidate of the disc in the widened band -r <= gap < r + h?"""
    diff = xs - p
    gap = diff @ n
    lateral = (diff - gap[:, None] * n).norm(dim=-1)
    return bool(((lateral < r_p) & (gap >= -R_SAMPLE) & (gap < R_SAMPLE + 1.0)).any())  # note: -r <= gap < r + h


def _sample_points(rng: torch.Generator, grid, h: float, n: int, radius_range) -> tuple[list, list, list]:
    """n static points in cell units with the note's rejection rules; returns positions, normals, radii (cell units)."""
    nx, ny, nz = grid.cell_counts
    rest = grid.rest.to(torch.float64).cpu()
    xs = rest[grid.samples.corners.cpu()].mean(1)  # rest sample centres
    nf = hx.FACE_NORMALS[grid.samples.face.cpu()]  # outward rest normals
    lo = torch.tensor([-SHELL_XZ / h, -SHELL_Y / h, -SHELL_XZ / h], dtype=torch.float64)
    hi = torch.tensor([nx + SHELL_XZ / h, float(ny), nz + SHELL_XZ / h], dtype=torch.float64)
    centre = torch.tensor([nx, ny, nz], dtype=torch.float64) / 2
    grown = torch.tensor([nx + 1.0, ny + 1.0, nz + 1.0], dtype=torch.float64)  # rest box grown by one cell
    points, normals, radii = [], [], []
    for _ in range(n):
        for _draw in range(MAX_POSITION_DRAWS):
            p = lo + (hi - lo) * torch.rand(3, generator=rng, dtype=torch.float64)
            if bool(((p > -1.0) & (p < grown)).all()):
                continue  # inside the rest box grown by one cell
            if p[2] < 1.0:
                continue  # in front of or on the clamped face
            r_p = _uniform(rng, *radius_range)  # units of h
            n_face = nf[(xs - p).norm(dim=-1).argmin()]
            for _ndraw in range(MAX_NORMAL_DRAWS):
                nrm = torch.randn(3, generator=rng, dtype=torch.float64)
                nrm = nrm / nrm.norm()
                if nrm @ (centre - p) < 0:
                    nrm = -nrm  # point at the body
                if nrm @ n_face <= -NORMAL_COS:
                    break
            else:
                continue
            if _rest_candidate(p, nrm, r_p, xs):
                continue
            points.append(p.tolist())
            normals.append(nrm.tolist())
            radii.append(r_p)
            break
        else:
            raise RuntimeError(f"no admissible static point after {MAX_POSITION_DRAWS} draws")
    return points, normals, radii


def sample_contact_spec(rng: torch.Generator, cfg, material_si: dict, grid) -> dict:
    """One object's contact scene in SI (JSON-serialisable): plane, static points, kappa (floored), beta, mu_f.

    Draw order: plane, point count, kappa, beta, mu_f from `rng`; the points from a child generator seeded by
    `rng`, so their rejections do not disturb the other draws. `kappa` is the effective ke / (E h) after the load
    floor; pass it to `material_from_si`.
    """
    if not cfg.contact:
        return {}
    h, E, rho = float(material_si["h"]), float(material_si["E"]), float(material_si["rho"])
    g = math.sqrt(sum(float(v) ** 2 for v in material_si["gravity"]))
    plane_present = bool(torch.rand((), generator=rng).item() < cfg.contact_plane_probability)
    plane_height = _uniform(rng, *cfg.contact_plane_height_range)
    count = int(torch.randint(0, int(cfg.contact_max_points) + 1, (), generator=rng))
    kappa_drawn = log_uniform(rng, *cfg.contact_kappa_range)
    beta = _uniform(rng, *cfg.contact_beta_range)
    mu_f = _uniform(rng, *cfg.contact_mu_range)
    child = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (), generator=rng)))
    points, normals, radii = _sample_points(child, grid, h, count, cfg.contact_point_radius_range)

    n_face = int(torch.bincount(grid.samples.face, minlength=6).max())
    ke_floor = contact_stiffness_floor(rho, g, h, grid.C, n_face, cfg.contact_static_penetration_max)
    ke = max(kappa_drawn * E * h, ke_floor)
    return {
        "plane_present": plane_present,
        "plane_normal": list(PLANE_NORMAL),
        "plane_height": float(plane_height),
        "points": [[v * h for v in p] for p in points],
        "normals": normals,
        "radii": [r * h for r in radii],
        "kappa": float(ke / (E * h)),
        "kappa_drawn": float(kappa_drawn),
        "ke": float(ke),
        "ke_floor": float(ke_floor),
        "floor_bound": bool(ke_floor > kappa_drawn * E * h),
        "beta": float(beta),
        "mu_f": float(mu_f),
    }


def scene_from_spec(spec_contact: dict, h: float, device, dtype=torch.float32) -> ContactScene:
    """One object's ContactScene in its normalised units (SI lengths divided by h); point_offsets = [0, n]."""
    t = lambda v, shape: torch.tensor(v, dtype=dtype, device=device).reshape(shape)  # noqa: E731
    points = t(spec_contact.get("points", []), (-1, 3)) / h
    n = points.shape[0]
    return ContactScene(
        plane_n=t(spec_contact.get("plane_normal", PLANE_NORMAL), (1, 3)),
        plane_d=t([spec_contact.get("plane_height", 0.0) / h], (1,)),
        plane_present=torch.tensor([bool(spec_contact.get("plane_present", False))], device=device),
        points=points,
        normals=t(spec_contact.get("normals", []), (-1, 3)),
        radii=t(spec_contact.get("radii", []), (-1,)) / h,
        point_offsets=torch.tensor([0, n], dtype=torch.int64, device=device),
    )


def cat_scenes(scenes: list[ContactScene]) -> ContactScene:
    """Concatenate per-object scenes and rebuild point_offsets."""
    counts = torch.stack([s.point_offsets[-1] for s in scenes])
    offsets = torch.zeros(len(scenes) + 1, dtype=torch.int64, device=counts.device)
    offsets[1:] = counts.cumsum(0)
    return ContactScene(
        plane_n=torch.cat([s.plane_n for s in scenes]),
        plane_d=torch.cat([s.plane_d for s in scenes]),
        plane_present=torch.cat([s.plane_present for s in scenes]),
        points=torch.cat([s.points for s in scenes]),
        normals=torch.cat([s.normals for s in scenes]),
        radii=torch.cat([s.radii for s in scenes]),
        point_offsets=offsets,
        faces=torch.cat([s.faces for s in scenes]),
    )


def scenes_for_objects(specs: list[dict], hs: list[float], device, dtype=torch.float32) -> ContactScene:
    """The batch ContactScene from the objects' contact specs and cell sizes."""
    return cat_scenes([scene_from_spec(s, h, device, dtype) for s, h in zip(specs, hs, strict=True)])
