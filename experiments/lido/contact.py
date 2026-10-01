# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact (contact note 2026-09-27; design spec 1b "Contact layout"): brute-force detection once per physical step
on the step-start shape, Newton's contact law as a differentiable energy, the 19-channel tokens, penetration metric,
and the rigid-translation quantities of the free-body centroid target (total contact force, active-pair stiffness;
derivation note section 7).

Everything is in each object's normalised units (h = dt = mu = 1): the sample radius is r = 0.5, velocities are
cells per step, the friction band is `material.friction_eps` (= friction_epsilon dt / h). For a pair with sample
position x, partner point p, partner normal n (pointing at the body), gap = (x - p) . n and the sample radius r
(the partner radius r_p is only the lateral extent of a static disc, contact note section 5): d = r - gap is the
penetration depth, delta = x - anchor the step displacement.
"""

from __future__ import annotations

import torch

from . import contact_kernel
from .structs import Pairs

Tensor = torch.Tensor
USE_WARP = True  # float32 CUDA contact energy through the Warp kernel when the pairs are in the capacity layout

R_SAMPLE = 0.5  # sample radius in cell units (r = 0.5 h)
M_PAIR = 4  # nearest static points kept per sample
TOKEN_DIM = 19
RADIUS_CHANNEL_CAP = 10.0  # cap of the r_p / r token channel


def sample_positions(batch, x: Tensor) -> Tensor:
    """Exposed face centres [S,3]: mean of the four corners."""
    return x[batch.sample_corners].mean(1)


def sample_normals(batch, x: Tensor) -> Tensor:
    """Outward face normals [S,3] from the diagonals (corners counter-clockwise); degenerate faces use the rest normal."""
    c = x[batch.sample_corners]
    n = torch.linalg.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1])
    norm = n.norm(dim=-1, keepdim=True)
    rest = batch.hc.face_normals[batch.sample_face].to(x.dtype)
    return torch.where(norm > 1e-12, n / norm.clamp_min(1e-12), rest)


def detect(batch, X: Tensor, V: Tensor, capacity: bool = False) -> Pairs:
    """Pairs on the step-start shape X with velocity margin from V (note section 4 with its amendments).

    Candidates per sample: the object's plane if present, plus the nearest M_PAIR static points of the object.
    A partner is kept when its normal opposes the face normal (n_q . n_s < 0) and the surface distance
    gap - r is below margin = r + |v_s|; a static point also needs lateral distance < r_p and gap >= -r
    (one-sided disc). Rows come out sorted by owning cell, sample, partner (plane first, then point id).

    `capacity=True` (design spec 1b "CUDA graphs") keeps every candidate slot instead of compacting: S (1 + k)
    rows in the same sample-major, slot-minor order with `valid` marking the detected ones, so the shapes are
    static per scene; `token_offsets` is then the static capacity CSR (the valid counts per cell are data) and the
    within-cell token attention pairs come precomputed from the layout (cached on the batch).
    """
    ok_all, sample_t, point_t, normal_t, radius_t, kind_t, xs = _candidates(batch, X, V)
    cols = ok_all.shape[1]
    dev = X.device
    sample = sample_t.reshape(-1)
    if capacity:
        layout = capacity_layout(batch, cols)
        return Pairs(
            token_offsets=layout[1],
            sample=sample,
            cell=batch.sample_cell[sample],
            obj=batch.sample_obj[sample],
            partner_point=point_t.reshape(-1, 3),
            partner_normal=normal_t.reshape(-1, 3),
            kind=kind_t.reshape(-1),
            radius=radius_t.reshape(-1),
            anchor=xs[sample],
            valid=ok_all.reshape(-1),
            padded=True,
            attn_pairs=layout[2],
            attn_offsets=layout[3],
        )
    sel = ok_all.reshape(-1)  # row-major (sample, partner): samples are sorted by cell, so rows are too
    sample = sample[sel]
    cell = batch.sample_cell[sample]
    offsets = torch.zeros(batch.C + 1, dtype=torch.int64, device=dev)
    offsets[1:] = torch.bincount(cell, minlength=batch.C).cumsum(0)
    return Pairs(
        token_offsets=offsets,
        sample=sample,
        cell=cell,
        obj=batch.sample_obj[sample],
        partner_point=point_t.reshape(-1, 3)[sel],
        partner_normal=normal_t.reshape(-1, 3)[sel],
        kind=kind_t.reshape(-1)[sel],
        radius=radius_t.reshape(-1)[sel],
        anchor=xs[sample],
        valid=torch.ones(sample.shape[0], dtype=torch.bool, device=dev),
    )


def capacity_layout(batch, cols: int) -> tuple:
    """(cols, token_offsets [C+1], attn_pairs [2,P], attn_offsets [S cols + 1]) of the capacity layout: sample s
    owns rows [s cols, (s + 1) cols), so each cell's token range is its samples' rows. Cached on the batch."""
    if batch.pair_layout is None or batch.pair_layout[0] != cols:
        from .network import within_range_pairs

        dev = batch.sample_cell.device
        per_cell = torch.bincount(batch.sample_cell, minlength=batch.C) * cols
        offsets = torch.zeros(batch.C + 1, dtype=torch.int64, device=dev)
        offsets[1:] = per_cell.cumsum(0)
        pairs, pair_offsets = within_range_pairs(offsets)
        batch.pair_layout = (cols, offsets, pairs, pair_offsets)
    return batch.pair_layout


def _candidates(batch, X: Tensor, V: Tensor):
    """The candidate slot grid [S, 1 + k]: plane slot first, then the k = min(M_PAIR, Npts) nearest points.

    Returns ok [S,cols] bool, sample [S,cols], partner point [S,cols,3], normal [S,cols,3], radius [S,cols],
    kind [S,cols] and the sample positions xs [S,3]. Dropped point slots hold the last point's data, ok = False.
    """
    scene = batch.scene
    dev, dtype = X.device, X.dtype
    xs = sample_positions(batch, X)
    ns = sample_normals(batch, X)
    margin = R_SAMPLE + V[batch.sample_corners].mean(1).norm(dim=-1)  # r + |v_s| dt, dt = 1
    obj = batch.sample_obj
    S = xs.shape[0]

    # plane (kind 0): partner point is the foot point, radius r
    pn = scene.plane_n[obj].to(dtype)
    gap_p = (xs * pn).sum(-1) - scene.plane_d[obj].to(dtype)
    ok_p = scene.plane_present[obj] & (gap_p - R_SAMPLE < margin) & ((pn * ns).sum(-1) < 0)
    foot = xs - gap_p[:, None] * pn

    # static points (kind 1): brute force [S, Npts] within the sample's object, nearest M_PAIR kept
    npts = scene.points.shape[0]
    k = min(M_PAIR, npts)
    if k > 0:
        pts, pnrm, prad = scene.points.to(dtype), scene.normals.to(dtype), scene.radii.to(dtype)
        counts = scene.point_offsets[1:] - scene.point_offsets[:-1]
        pobj = torch.repeat_interleave(torch.arange(batch.O, device=dev), counts)
        diff = xs[:, None, :] - pts[None]  # [S,Npts,3]
        gap = (diff * pnrm[None]).sum(-1)
        lateral = (diff - gap[..., None] * pnrm[None]).norm(dim=-1)
        ok = (
            (obj[:, None] == pobj[None])
            & (lateral < prad[None])
            & (gap >= -R_SAMPLE)
            & (gap - R_SAMPLE < margin[:, None])
            & ((pnrm[None] * ns[:, None]).sum(-1) < 0)
        )
        dist = diff.norm(dim=-1).masked_fill(~ok, float("inf"))
        dist_k, idx_k = dist.topk(k, dim=1, largest=False)  # [S,k]
        idx_k = idx_k.masked_fill(~torch.isfinite(dist_k), npts).sort(dim=1).values  # by point id, dropped last
        ok_k = idx_k < npts
        idx_k = idx_k.clamp_max(npts - 1)
        ok_all = torch.cat([ok_p[:, None], ok_k], 1)
        point_t = torch.cat([foot[:, None], pts[idx_k]], 1)
        normal_t = torch.cat([pn[:, None], pnrm[idx_k]], 1)
        radius_t = torch.cat([torch.full((S, 1), R_SAMPLE, dtype=dtype, device=dev), prad[idx_k]], 1)
    else:
        ok_all, point_t, normal_t = ok_p[:, None], foot[:, None], pn[:, None]
        radius_t = torch.full((S, 1), R_SAMPLE, dtype=dtype, device=dev)
    cols = 1 + k
    kind_t = torch.cat(
        [torch.zeros(S, 1, dtype=torch.int64, device=dev), torch.ones(S, k, dtype=torch.int64, device=dev)], 1
    )
    sample_t = torch.arange(S, device=dev)[:, None].expand(S, cols)
    return ok_all, sample_t, point_t, normal_t, radius_t, kind_t, xs


def _geometry(batch, x: Tensor, pairs: Pairs):
    """Sample position, partner normal, gap and r_total per pair at x."""
    xs = x[batch.sample_corners[pairs.sample]].mean(1)
    n = pairs.partner_normal
    gap = ((xs - pairs.partner_point) * n).sum(-1)
    r_total = torch.full_like(gap, R_SAMPLE)  # d = r - gap for both kinds (note section 5)
    return xs, n, gap, r_total


def _f0(y: Tensor, eps: Tensor) -> Tensor:
    """IPC smooth friction potential: -y^3/(3 eps^2) + y^2/eps + eps/3 for y < eps, y otherwise."""
    return torch.where(y < eps, -(y**3) / (3.0 * eps**2) + y**2 / eps + eps / 3.0, y)


def pair_energies(batch, x: Tensor, pairs: Pairs, load: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Normal, damping and friction energies per pair [Q] (note section 5, Newton's `_compute_body_particle_contact_force`).

    E_n = ke/2 relu(d)^2; E_d = kd/2 relu(-n . delta)^2 while penetrating; E_f = mu f_n f0(|u|) with the detached
    normal load f_n = ke relu(d) (or the given `load` [Q], for tests that freeze it), tangential slip
    u = delta - (n . delta) n and band eps = friction_eps.
    """
    xs, n, gap, r_total = _geometry(batch, x, pairs)
    m, o = batch.material, pairs.obj
    ke, kd, mu_f, eps = m.ke[o], m.kd[o], m.mu_f[o], m.friction_eps[o]
    d = r_total - gap
    pen = torch.relu(d)
    active = (d > 0).to(x.dtype)
    delta = xs - pairs.anchor
    vn = (n * delta).sum(-1)
    E_n = 0.5 * ke * pen**2
    E_d = 0.5 * kd * torch.relu(-vn) ** 2 * active
    u = delta - vn[:, None] * n
    y = (u * u).sum(-1).clamp_min(1e-24).sqrt()  # zero gradient at u = 0
    f_n = (ke * pen).detach() if load is None else load
    E_f = mu_f * f_n * _f0(y, eps)
    return E_n, E_d, E_f


def contact_energy(batch, x: Tensor) -> Tensor:
    """Contact energy per object [O] in mu h^3 units; zeros without pairs. Differentiable in x."""
    pairs = batch.pairs
    out = torch.zeros(batch.O, dtype=x.dtype, device=x.device)
    if pairs is None or pairs.count == 0:
        return out
    if USE_WARP and x.is_cuda and x.dtype == torch.float32 and pairs.padded:
        return contact_kernel.contact_energy_warp(batch, x, R_SAMPLE)
    E_n, E_d, E_f = pair_energies(batch, x, pairs)
    return out.index_add(0, pairs.obj, (E_n + E_d + E_f).masked_fill(~pairs.valid, 0.0))


def contact_force(batch, x: Tensor) -> Tensor:
    """Total contact force per object F_con(x) = -sum_i grad_{x_i} E_con(x) [O,3] (eq. 1.9), detached.

    The gradient is taken with autograd through `contact_energy` (the torch pair energies or the Warp pair kernel);
    the sum over the object's corners equals the sum over its pairs of -dE/dx_s since each pair's gradient is split
    equally over the four face corners."""
    pairs = batch.pairs
    out = torch.zeros(batch.O, 3, dtype=x.dtype, device=x.device)
    if pairs is None or pairs.count == 0:
        return out
    with torch.enable_grad():
        xg = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(contact_energy(batch, xg).sum(), xg)
    return out.index_add_(0, batch.corner_obj, -g)


def active_stiffness(batch, x: Tensor) -> tuple[Tensor, Tensor]:
    """(sum_active ke [O], sum_active ke n n^T [O,3,3]) over the penetrating pairs (d = r - gap > 0) at x: minus the
    derivative of the normal contact force with respect to a rigid translation (eq. 7.13) and the contact part of
    the translation residual's Jacobian (eq. 7.14). Detached."""
    pairs = batch.pairs
    O = batch.O
    ke_sum = torch.zeros(O, dtype=x.dtype, device=x.device)
    H = torch.zeros(O, 3, 3, dtype=x.dtype, device=x.device)
    if pairs is None or pairs.count == 0:
        return ke_sum, H
    _, n, gap, r_total = _geometry(batch, x.detach(), pairs)
    active = ((r_total - gap > 0) & pairs.valid).to(x.dtype)
    ke = batch.material.ke[pairs.obj].to(x.dtype) * active
    ke_sum = ke_sum.index_add_(0, pairs.obj, ke)
    H = H.index_add_(0, pairs.obj, ke[:, None, None] * n[:, :, None] * n[:, None, :])
    return ke_sum, H


def contact_tokens(batch, x: Tensor, R: Tensor) -> Tensor:
    """One 19-channel token per pair in the owning cell's frozen frame R[cell] (note section 6).

    Channels: sample position (3), partner point (3), partner normal (3), all relative to the cell centre in the
    cell frame; gap / r; approach rate -(n . delta) / r; r_p / r (capped); log1p(kappa); beta; mu_f; kind one-hot
    (plane, point, self); self flag.
    """
    pairs = batch.pairs
    Q = pairs.count
    if Q == 0:
        return torch.zeros(0, TOKEN_DIM, dtype=x.dtype, device=x.device)
    xs, n, gap, _ = _geometry(batch, x, pairs)
    centre = x[batch.cells[pairs.cell]].mean(1)
    Rt = R[pairs.cell].transpose(-1, -2).to(x.dtype)
    loc = lambda v: torch.einsum("qab,qb->qa", Rt, v)  # noqa: E731
    m, o = batch.material, pairs.obj
    approach = -(n * (xs - pairs.anchor)).sum(-1) / R_SAMPLE
    return torch.cat(
        [
            loc(xs - centre),
            loc(pairs.partner_point - centre),
            loc(n),
            (gap / R_SAMPLE)[:, None],
            approach[:, None],
            (pairs.radius / R_SAMPLE).clamp_max(RADIUS_CHANNEL_CAP)[:, None],
            torch.log1p(m.kappa[o])[:, None],  # log1p(kappa), kappa = ke / (E h)
            m.beta[o][:, None],  # beta = kd / (ke dt)
            m.mu_f[o][:, None],
            torch.zeros(Q, 3, dtype=x.dtype, device=x.device).scatter_(1, pairs.kind[:, None], 1.0),  # kind one-hot
            torch.zeros(Q, 1, dtype=x.dtype, device=x.device),
        ],
        -1,
    )


def penetration(batch, x: Tensor) -> Tensor:
    """Maximum penetration depth over each object's pairs in units of r [O]; zero without pairs."""
    pairs = batch.pairs
    out = torch.zeros(batch.O, dtype=x.dtype, device=x.device)
    if pairs is None or pairs.count == 0:
        return out
    _, _, gap, r_total = _geometry(batch, x, pairs)
    pen = (torch.relu(r_total - gap) / R_SAMPLE).masked_fill(~pairs.valid, 0.0)
    return out.scatter_reduce(0, pairs.obj, pen, reduce="amax", include_self=True)
