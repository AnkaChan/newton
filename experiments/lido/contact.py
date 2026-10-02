# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact (contact note 2026-09-27; design spec 1b "Contact layout"; section 11 for body-body contact): detection
once per physical step on the step-start shape, Newton's contact law as a differentiable energy, the 19-channel
tokens, penetration metric, and the rigid-translation quantities of the free-body centroid target (total contact
force, active-pair stiffness; derivation note section 7).

Everything is in each object's normalised units (h = dt = mu = 1): the sample radius is r = 0.5, velocities are
cells per step, the friction band is `material.friction_eps` (= friction_epsilon dt / h). For a pair with sample
position x, partner point p, partner normal n (pointing at the body), gap = (x - p) . n and the sample radius r
(the partner radius r_p is only the lateral extent of a static disc, contact note section 5): d = r - gap is the
penetration depth, delta the step displacement that drives damping and friction.

Static partners (kind 0 plane, kind 1 disc) are frozen with the pair list: p, n are the detection values and
delta = x - anchor with the anchor the sample position at step start.

Body pairs (kind 2, `batch.body_contact`; one world frame, every body a partner of every other). Detection builds a
`wp.Mesh` of every object's exposed faces (two triangles per quad from `batch.sample_corners`, vertices the object's
corners at X, refitted every step) and queries each sample against the other objects' meshes with
`wp.mesh_query_point_sign_parity`, so the closest face and the inside / outside sign stay robust under penetration.
A kept pair records the partner body and the global sample id of its face; at every query the geometry is rebuilt
from the partner's current corners c_i (so the energy's gradient reaches both bodies):

    w     = barycentric weights of the closest point to x on the quad (c_0, c_1, c_2, c_3) = triangles (0, 1, 2),
            (0, 2, 3); held constant in the gradient (the partner's material point under the sample)
    p     = sum_i w_i c_i                          n = unit (c_2 - c_0) x (c_3 - c_1) (outward, as `sample_normals`)
    delta = (x - anchor) - sum_i w_i (c_i - C_i)   the slip of the sample relative to that material point since step
                                                   start (C_i the partner's corners at X)

Holding w is exact for planar faces (the closest point slides tangentially), makes the forces on the two bodies equal
and opposite (the partner corners receive -w_i times the sample's gradient, plus the gradient through n, which sums
to zero over the four corners), and lets two bodies that move together see no friction or damping between them.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
import warp as wp

from . import contact_kernel
from .structs import Pairs

Tensor = torch.Tensor
USE_WARP = True  # float32 CUDA contact energy through the Warp kernel when the pairs are in the capacity layout
USE_WARP_GEOMETRY = True  # ... and the pair geometry of the tokens, stiffness and penetration (`_geometry`)

R_SAMPLE = 0.5  # sample radius in cell units (r = 0.5 h)
M_PAIR = 4  # nearest partners kept per sample besides the plane (static points and other bodies' faces)
TOKEN_DIM = 19
RADIUS_CHANNEL_CAP = 10.0  # cap of the r_p / r token channel
BODY_SIGN_RAYS = 3  # rays of the sign-parity classification per mesh query (odd; Warp's default is 1)
KIND_BODY = 2  # the reserved slot of the kind one-hot
FACE_TIE_TOL = 1e-5  # faces whose closest points are this close (relative) count as ties in the detection


def sample_positions(batch, x: Tensor) -> Tensor:
    """Exposed face centres [S,3]: mean of the four corners."""
    return x[batch.sample_corners].mean(1)


def quad_normals(c: Tensor, rest: Tensor) -> Tensor:
    """Outward unit normals [...,3] of the quads with corners c [...,4,3] (counter-clockwise) from the diagonals;
    degenerate quads (|cross| <= 1e-12) take the rest normal `rest` [...,3]."""
    n = torch.linalg.cross(c[..., 2, :] - c[..., 0, :], c[..., 3, :] - c[..., 1, :])
    norm = n.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-12, n / norm.clamp_min(1e-12), rest)


def sample_normals(batch, x: Tensor) -> Tensor:
    """Outward face normals [S,3] from the diagonals (corners counter-clockwise); degenerate faces use the rest normal."""
    return quad_normals(x[batch.sample_corners], batch.hc.face_normals[batch.sample_face].to(x.dtype))


# --------------------------------------------------------------------------------------------- closest points
def _safe_div(num: Tensor, den: Tensor) -> Tensor:
    return num / torch.where(den == 0, torch.ones_like(den), den)


def closest_point_on_triangle(p: Tensor, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Barycentric weights [...,3] of the closest point to p on the triangles (a, b, c), all [...,3] (Ericson,
    Real-Time Collision Detection 5.1.5: vertex, edge and face regions in that priority)."""
    ab, ac = b - a, c - a
    d1, d2 = ((ab * (p - a)).sum(-1), (ac * (p - a)).sum(-1))
    d3, d4 = ((ab * (p - b)).sum(-1), (ac * (p - b)).sum(-1))
    d5, d6 = ((ab * (p - c)).sum(-1), (ac * (p - c)).sum(-1))
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    one, zero = torch.ones_like(d1), torch.zeros_like(d1)
    # face region
    denom = _safe_div(one, va + vb + vc)
    v, w = vb * denom, vc * denom
    out = torch.stack([1.0 - v - w, v, w], -1)
    # edge BC
    w = _safe_div(d4 - d3, (d4 - d3) + (d5 - d6))
    out = torch.where(
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], torch.stack([zero, 1.0 - w, w], -1), out
    )
    # edge AC
    w = _safe_div(d2, d2 - d6)
    out = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], torch.stack([1.0 - w, zero, w], -1), out)
    # vertex C
    out = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], torch.stack([zero, zero, one], -1), out)
    # edge AB
    v = _safe_div(d1, d1 - d3)
    out = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], torch.stack([1.0 - v, v, zero], -1), out)
    # vertex B
    out = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], torch.stack([zero, one, zero], -1), out)
    # vertex A
    out = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], torch.stack([one, zero, zero], -1), out)
    return out


def closest_point_on_quad(p: Tensor, c0: Tensor, c1: Tensor, c2: Tensor, c3: Tensor) -> tuple[Tensor, Tensor]:
    """Closest point to p on the quad (c0, c1, c2, c3) = triangles (c0, c1, c2) and (c0, c2, c3), all [...,3]: the
    point and its weights over the four corners [...,4] (one of them zero); the nearer triangle, the first on ties."""
    w1 = closest_point_on_triangle(p, c0, c1, c2)
    w2 = closest_point_on_triangle(p, c0, c2, c3)
    z = torch.zeros_like(w1[..., :1])
    w1 = torch.cat([w1, z], -1)
    w2 = torch.cat([w2[..., :1], z, w2[..., 1:]], -1)
    c = torch.stack([c0, c1, c2, c3], -2)  # [...,4,3]
    p1 = (w1[..., None] * c).sum(-2)
    p2 = (w2[..., None] * c).sum(-2)
    first = (((p - p1) ** 2).sum(-1) <= ((p - p2) ** 2).sum(-1))[..., None]
    return torch.where(first, p1, p2), torch.where(first, w1, w2)


# --------------------------------------------------------------------------------------------------- meshes
def _warp_scope(device: torch.device):
    """Run Warp work (mesh build, refit, launches) on the torch stream of a CUDA device."""
    if device.type != "cuda":
        return contextlib.nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))


class BodyMeshes:
    """One `wp.Mesh` per object over its exposed faces: two triangles per quad (c0, c1, c2), (c0, c2, c3) in local
    corner ids, so triangle t of object o is the face of global sample `sample_off[o] + t // 2`; the vertices alias
    the object's rows of one float32 copy of the corners, refreshed by `refresh` (points + BVH refit)."""

    def __init__(self, batch, X: Tensor):
        wp.init()
        self.device = X.device
        self.N = int(X.shape[0])
        self.points = X.detach().to(torch.float32).contiguous().clone()  # [N,3]
        self.face_neighbours = face_neighbours(batch)  # [S,W] global sample ids of the edge-adjacent faces, -1 padded
        self.indices = []
        self.meshes = []
        with _warp_scope(self.device):
            for o in range(batch.O):
                c0, c1 = int(batch.corner_off[o]), int(batch.corner_off[o + 1])
                s0, s1 = int(batch.sample_off[o]), int(batch.sample_off[o + 1])
                quads = batch.sample_corners[s0:s1] - c0  # [S_o,4] local corner ids
                tris = torch.stack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], 1).reshape(-1).to(torch.int32)
                tris = tris.contiguous()
                self.indices.append(tris)
                self.meshes.append(
                    wp.Mesh(wp.from_torch(self.points[c0:c1], dtype=wp.vec3), wp.from_torch(tris, dtype=wp.int32))
                )
            ids = np.array([m.id for m in self.meshes], dtype=np.uint64)
            self.ids = wp.array(ids, dtype=wp.uint64, device=wp.device_from_torch(self.device))

    def refresh(self, X: Tensor) -> None:
        self.points.copy_(X.detach())
        with _warp_scope(self.device):
            for m in self.meshes:
                m.refit()


def face_neighbours(batch) -> Tensor:
    """For every exposed face (sample) the faces of the same body sharing one of its four edges [S,W], W = 4 times
    the largest edge multiplicity (2 on a manifold surface), padded with -1. Objects share no corners, so corner
    pairs identify edges globally."""
    sc = batch.sample_corners
    S, dev = sc.shape[0], sc.device
    if S == 0:
        return torch.zeros(0, 4, dtype=torch.int64, device=dev)
    e = torch.stack([sc, sc.roll(-1, dims=1)], -1)  # [S,4,2] the edges (c_i, c_{i+1})
    key = (e.min(-1).values * int(batch.N) + e.max(-1).values).reshape(-1)  # [4S]
    uniq, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
    order = inv.argsort(stable=True)
    start = counts.cumsum(0) - counts
    rank = torch.arange(4 * S, device=dev) - start[inv[order]]
    width = int(counts.max())
    table = torch.full((uniq.shape[0], width), -1, dtype=torch.int64, device=dev)
    table[inv[order], rank] = order // 4  # the faces on each edge
    nb = table[inv].view(S, 4 * width)  # [S,4W]: the faces on the sample's four edges, itself included
    own = torch.arange(S, device=dev)[:, None]
    return nb.masked_fill(nb == own, -1)


def body_meshes(batch, X: Tensor) -> BodyMeshes:
    """The batch's meshes at X: built when absent (after `Batch.relayout`), refreshed otherwise."""
    m = batch.meshes
    if m is None or m.N != X.shape[0] or m.device != X.device:
        batch.meshes = m = BodyMeshes(batch, X)
    else:
        m.refresh(X)
    return m


def object_bounds(batch, X: Tensor) -> tuple[Tensor, Tensor]:
    """Per-object bounding boxes of the corners [O,3], [O,3] (float32)."""
    x = X.detach().to(torch.float32)
    idx = batch.corner_obj[:, None].expand(-1, 3)
    lo = torch.full((batch.O, 3), float("inf"), dtype=torch.float32, device=X.device)
    hi = torch.full((batch.O, 3), float("-inf"), dtype=torch.float32, device=X.device)
    lo.scatter_reduce_(0, idx, x, reduce="amin")
    hi.scatter_reduce_(0, idx, x, reduce="amax")
    return lo, hi


# ------------------------------------------------------------------------------------------------- detection
def detect(batch, X: Tensor, V: Tensor, capacity: bool = False) -> Pairs:
    """Pairs on the step-start shape X with velocity margin from V (note section 4 with its amendments).

    Candidates per sample: the object's plane if present, plus the nearest M_PAIR of {the object's static points,
    the other bodies' surfaces (when `batch.body_contact`)}. A partner is kept when its normal opposes the face
    normal (n_q . n_s < 0) and the surface distance gap - r is below margin = r + |v_s| for the static partners and
    r + |v_s - v_f| (the relative speed to the partner face) for a body; a static point also needs lateral
    distance < r_p and gap >= -r (one-sided disc). For a body the gap is the signed distance to its surface
    mesh at X (negative inside, `wp.mesh_query_point_sign_parity`, queried within the largest margin + r), the partner
    normal the closest face's normal at X, the partner point the closest point, and the closest point's lateral offset
    must be below r (`body_candidates`). Rows come out sorted by owning cell, sample, partner (plane, then point
    ids, then bodies by object id).

    `capacity=True` (design spec 1b "CUDA graphs") keeps every candidate slot instead of compacting: S (1 + k)
    rows in the same sample-major, slot-minor order with `valid` marking the detected ones, so the shapes are
    static per scene; `token_offsets` is then the static capacity CSR (the valid counts per cell are data) and the
    within-cell token attention pairs come precomputed from the layout (cached on the batch).
    """
    cand = _candidates(batch, X, V)
    ok_all, sample_t = cand["ok"], cand["sample"]
    cols = ok_all.shape[1]
    dev = X.device
    sample = sample_t.reshape(-1)
    fields = {
        "partner_point": cand["point"].reshape(-1, 3),
        "partner_normal": cand["normal"].reshape(-1, 3),
        "kind": cand["kind"].reshape(-1),
        "radius": cand["radius"].reshape(-1),
        "partner_body": cand["body"].reshape(-1),
        "partner_face": cand["face"].reshape(-1),
    }
    if capacity:
        layout = capacity_layout(batch, cols)
        return Pairs(
            token_offsets=layout[1],
            sample=sample,
            cell=batch.sample_cell[sample],
            obj=batch.sample_obj[sample],
            anchor=cand["xs"][sample],
            valid=ok_all.reshape(-1),
            padded=True,
            attn_pairs=layout[2],
            attn_offsets=layout[3],
            **fields,
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
        anchor=cand["xs"][sample],
        valid=torch.ones(sample.shape[0], dtype=torch.bool, device=dev),
        **{k: v[sel] for k, v in fields.items()},
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


def body_candidates(batch, X: Tensor, xs: Tensor, ns: Tensor, vs: Tensor) -> tuple:
    """Candidates of every sample against every other object's surface mesh at X: keep [S,O], signed surface
    distance [S,O], partner face (global sample id, -1 without a result) [S,O], closest point [S,O,3].

    The mesh query (one thread per sample and other object, skipped outside the object's bounding box grown by the
    reach) returns the closest face within the largest possible margin + r (ties between edge-adjacent faces resolved
    towards the normal opposing the sample's, `FACE_TIE_TOL`); a candidate is kept when its face normal at X opposes
    the sample normal, gap - r < margin with gap the signed surface distance (always true inside the partner) and
    margin = r + |v_s - v_f| the RELATIVE speed of the sample and the partner face (`vs` [S,3] the sample velocities;
    the face velocity is the mean of its corner velocities: two bodies moving together do not approach, a body at
    rest sees a body coming at it; the static partners use |v_s| in `_candidates`), and the closest point's lateral
    offset from the sample's normal line is below r (the static disc's rule with r_p = r: a sample overhanging an
    edge by more than its radius does not touch the face)."""
    meshes = body_meshes(batch, X)
    dev, dtype = X.device, X.dtype
    S, O = xs.shape[0], batch.O
    lo, hi = object_bounds(batch, X)
    ok = torch.zeros(S, O, dtype=torch.bool, device=dev)
    dist = torch.zeros(S, O, dtype=torch.float32, device=dev)
    face = torch.full((S, O), -1, dtype=torch.int64, device=dev)
    point = torch.zeros(S, O, 3, dtype=torch.float32, device=dev)
    speed = vs.norm(dim=-1)
    reach = 2.0 * R_SAMPLE + speed + speed.max()  # margin + r for any partner: |v_s - v_f| <= |v_s| + max |v|
    with _warp_scope(dev):
        contact_kernel.launch_body_query(
            xs.detach().to(torch.float32).contiguous(),
            batch.sample_obj,
            reach.detach().to(torch.float32).contiguous(),
            meshes.ids,
            lo,
            hi,
            batch.sample_off,
            ok,
            dist,
            face,
            point,
            BODY_SIGN_RAYS,
        )
    dist, point = dist.to(dtype), point.to(dtype)
    hit = ok.nonzero(as_tuple=True)
    if hit[0].numel() > 0:
        # the closest point may lie on an edge shared with a neighbouring face (the BVH returns either): among the
        # found face and its edge neighbours, those within tolerance of the smallest distance are ties, and the one
        # whose normal opposes the sample normal best is taken; point and distance recomputed in the batch's dtype
        s_idx = hit[0]
        f0 = face[hit]
        faces = torch.cat([f0[:, None], meshes.face_neighbours[f0]], 1)  # [K,1+W]
        present = faces >= 0
        fc = faces.clamp_min(0)
        c = X[batch.sample_corners[fc]]  # [K,1+W,4,3]
        ps = xs[s_idx][:, None, :].expand(-1, faces.shape[1], 3)
        pt, _ = closest_point_on_quad(ps, c[..., 0, :], c[..., 1, :], c[..., 2, :], c[..., 3, :])
        df = (pt - ps).norm(dim=-1).masked_fill(~present, float("inf"))
        dmin = df.min(1).values
        tie = df <= dmin[:, None] + FACE_TIE_TOL * (1.0 + dmin[:, None])
        dots = (ns[fc] * ns[s_idx][:, None, :]).sum(-1).masked_fill(~tie, float("inf"))
        best = dots.argmin(1)
        rows = torch.arange(f0.shape[0], device=dev)
        face[hit] = faces[rows, best]
        point[hit] = pt[rows, best]
        dist[hit] = torch.where(dist[hit] < 0, -dmin, dmin)
    fc = face.clamp_min(0)
    nf = ns[fc]  # [S,O,3] partner face normals at X
    opposing = (nf * ns[:, None, :]).sum(-1) < 0
    rel = xs[:, None, :] - point
    gap = (rel * nf).sum(-1)
    lateral = (rel - gap[..., None] * nf).norm(dim=-1)  # offset of the closest point from the sample's normal line
    margin = R_SAMPLE + (vs[:, None, :] - vs[fc]).norm(dim=-1)  # r + the relative speed to the partner face
    keep = ok & opposing & (dist - R_SAMPLE < margin) & (lateral < R_SAMPLE)
    return keep, dist, face, point


def _candidates(batch, X: Tensor, V: Tensor) -> dict:
    """The candidate slot grid [S, 1 + k]: plane slot first, then the k = min(M_PAIR, Npts + O - 1) nearest of the
    static points (brute force within the sample's object) and the other bodies' faces (mesh queries), ordered by
    point id then partner object id. Dropped slots hold the last candidate's data with ok = False.

    Returns a dict: ok [S,cols] bool, sample [S,cols], point [S,cols,3], normal [S,cols,3], radius [S,cols], kind
    [S,cols], body [S,cols] (partner object, -1), face [S,cols] (partner face sample id, -1) and the sample positions
    xs [S,3].
    """
    scene = batch.scene
    dev, dtype = X.device, X.dtype
    xs = sample_positions(batch, X)
    ns = sample_normals(batch, X)
    vs = V[batch.sample_corners].mean(1)  # sample velocities (cells per step)
    margin = R_SAMPLE + vs.norm(dim=-1)  # r + |v_s| dt, dt = 1: the static partners' margin
    obj = batch.sample_obj
    S = xs.shape[0]
    rows = torch.arange(S, device=dev)[:, None]

    # plane (kind 0): partner point is the foot point, radius r
    pn = scene.plane_n[obj].to(dtype)
    gap_p = (xs * pn).sum(-1) - scene.plane_d[obj].to(dtype)
    ok_p = scene.plane_present[obj] & (gap_p - R_SAMPLE < margin) & ((pn * ns).sum(-1) < 0)
    foot = xs - gap_p[:, None] * pn

    # static points (kind 1): brute force [S, Npts] within the sample's object
    npts = scene.points.shape[0]
    if npts > 0:
        pts, pnrm, prad = scene.points.to(dtype), scene.normals.to(dtype), scene.radii.to(dtype)
        counts = scene.point_offsets[1:] - scene.point_offsets[:-1]
        pobj = torch.repeat_interleave(torch.arange(batch.O, device=dev), counts)
        diff = xs[:, None, :] - pts[None]  # [S,Npts,3]
        gap = (diff * pnrm[None]).sum(-1)
        lateral = (diff - gap[..., None] * pnrm[None]).norm(dim=-1)
        ok_pt = (
            (obj[:, None] == pobj[None])
            & (lateral < prad[None])
            & (gap >= -R_SAMPLE)
            & (gap - R_SAMPLE < margin[:, None])
            & ((pnrm[None] * ns[:, None]).sum(-1) < 0)
        )
        dist_pt = diff.norm(dim=-1)
    else:
        ok_pt = torch.zeros(S, 0, dtype=torch.bool, device=dev)
        dist_pt = torch.zeros(S, 0, dtype=dtype, device=dev)

    # other bodies (kind 2): the signed surface distance ranks them against the points
    nbody = batch.O if (batch.body_contact and batch.O > 1) else 0
    if nbody > 0:
        ok_b, dist_b, face_b, point_b = body_candidates(batch, X, xs, ns, vs)
    else:
        ok_b = torch.zeros(S, 0, dtype=torch.bool, device=dev)
        dist_b = torch.zeros(S, 0, dtype=dtype, device=dev)

    total = npts + nbody  # candidate columns (a body's own column never fires)
    k = min(M_PAIR, npts + max(nbody - 1, 0))
    full = lambda value, dt=dtype: torch.full((S, k), value, dtype=dt, device=dev)  # noqa: E731
    if k > 0:
        ok = torch.cat([ok_pt, ok_b], 1)
        dist = torch.cat([dist_pt, dist_b], 1).masked_fill(~ok, float("inf"))
        dist_k, idx_k = dist.topk(k, dim=1, largest=False)  # [S,k]
        idx_k = idx_k.masked_fill(~torch.isfinite(dist_k), total).sort(dim=1).values  # by candidate id, dropped last
        ok_k = idx_k < total
        idx_k = idx_k.clamp_max(total - 1)
        is_body = idx_k >= npts
        point_k, normal_k, radius_k = (
            full(0.0)[..., None].expand(S, k, 3),
            full(0.0)[..., None].expand(S, k, 3),
            full(0.0),
        )
        body_k, face_k = full(-1, torch.int64), full(-1, torch.int64)
        if npts > 0:
            ip = idx_k.clamp_max(npts - 1)
            point_k, normal_k, radius_k = pts[ip], pnrm[ip], prad[ip]
        if nbody > 0:
            ib = (idx_k - npts).clamp_min(0)
            face_sel = face_b[rows, ib]
            point_k = torch.where(is_body[..., None], point_b[rows, ib], point_k)
            normal_k = torch.where(is_body[..., None], ns[face_sel.clamp_min(0)], normal_k)
            radius_k = torch.where(is_body, full(R_SAMPLE), radius_k)
            body_k = torch.where(is_body, ib, body_k)
            face_k = torch.where(is_body, face_sel, face_k)
        kind_k = torch.where(is_body, full(KIND_BODY, torch.int64), full(1, torch.int64))
        ok_all = torch.cat([ok_p[:, None], ok_k], 1)
        point_t = torch.cat([foot[:, None], point_k], 1)
        normal_t = torch.cat([pn[:, None], normal_k], 1)
        radius_t = torch.cat([torch.full((S, 1), R_SAMPLE, dtype=dtype, device=dev), radius_k], 1)
        kind_t = torch.cat([torch.zeros(S, 1, dtype=torch.int64, device=dev), kind_k], 1)
        body_t = torch.cat([torch.full((S, 1), -1, dtype=torch.int64, device=dev), body_k], 1)
        face_t = torch.cat([torch.full((S, 1), -1, dtype=torch.int64, device=dev), face_k], 1)
    else:
        ok_all, point_t, normal_t = ok_p[:, None], foot[:, None], pn[:, None]
        radius_t = torch.full((S, 1), R_SAMPLE, dtype=dtype, device=dev)
        kind_t = torch.zeros(S, 1, dtype=torch.int64, device=dev)
        body_t = torch.full((S, 1), -1, dtype=torch.int64, device=dev)
        face_t = body_t.clone()
    cols = 1 + k
    sample_t = torch.arange(S, device=dev)[:, None].expand(S, cols)
    return {
        "ok": ok_all,
        "sample": sample_t,
        "point": point_t,
        "normal": normal_t,
        "radius": radius_t,
        "kind": kind_t,
        "body": body_t,
        "face": face_t,
        "xs": xs,
    }


# ---------------------------------------------------------------------------------------------- pair geometry
def _body_geometry(batch, x: Tensor, pairs: Pairs, xs: Tensor, p: Tensor, n: Tensor, delta: Tensor) -> tuple:
    """Kind-2 rows at x (module docstring): partner point from the held closest-point weights on the partner's
    current corners, the quad's normal, and the slip relative to the partner's material point; other rows pass
    through. Differentiable in x through the corners (weights detached)."""
    body = (pairs.kind == KIND_BODY)[:, None]
    face = pairs.partner_face.clamp_min(0)
    corners = batch.sample_corners[face]  # [Q,4]
    c = x[corners]  # [Q,4,3]
    rest = batch.hc.face_normals[batch.sample_face[face]].to(x.dtype)
    n_b = quad_normals(c, rest)
    with torch.no_grad():
        _, w = closest_point_on_quad(xs.detach(), *(c[:, i].detach() for i in range(4)))
    p_b = (w[..., None] * c).sum(1)
    moved = (w[..., None] * (c - batch.X[corners])).sum(1)  # the material point's displacement since step start
    return torch.where(body, p_b, p), torch.where(body, n_b, n), torch.where(body, delta - moved, delta)


def _geometry(batch, x: Tensor, pairs: Pairs):
    """Per pair at x: sample position xs, partner point p, partner normal n, gap = (xs - p) . n, r_total and the
    step displacement delta (relative to the partner's material point for body pairs). Capacity-layout rows on
    float32 CUDA without a graph on x go through the Warp kernel (`contact_kernel.pair_geometry_warp`: one launch
    over the 4-5 rows per sample instead of a torch chain over all of them, padded rows zero)."""
    if (
        USE_WARP
        and USE_WARP_GEOMETRY
        and x.is_cuda
        and x.dtype == torch.float32
        and pairs.padded
        and not x.requires_grad
    ):
        xs, p, n, gap, delta = contact_kernel.pair_geometry_warp(batch, x, pairs)
        return xs, p, n, gap, torch.full_like(gap, R_SAMPLE), delta
    xs = x[batch.sample_corners[pairs.sample]].mean(1)
    p, n = pairs.partner_point, pairs.partner_normal
    delta = xs - pairs.anchor
    if batch.body_contact:
        p, n, delta = _body_geometry(batch, x, pairs, xs, p, n, delta)
    gap = ((xs - p) * n).sum(-1)
    r_total = torch.full_like(gap, R_SAMPLE)  # d = r - gap for every kind (note section 5)
    return xs, p, n, gap, r_total, delta


def _f0(y: Tensor, eps: Tensor) -> Tensor:
    """IPC smooth friction potential: -y^3/(3 eps^2) + y^2/eps + eps/3 for y < eps, y otherwise."""
    return torch.where(y < eps, -(y**3) / (3.0 * eps**2) + y**2 / eps + eps / 3.0, y)


def pair_energies(batch, x: Tensor, pairs: Pairs, load: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Normal, damping and friction energies per pair [Q] (note section 5, Newton's `_compute_body_particle_contact_force`).

    E_n = ke/2 relu(d)^2; E_d = kd/2 relu(-n . delta)^2 while penetrating; E_f = mu f_n f0(|u|) with the detached
    normal load f_n = ke relu(d) (or the given `load` [Q], for tests that freeze it), tangential slip
    u = delta - (n . delta) n and band eps = friction_eps.
    """
    _, _, n, gap, r_total, delta = _geometry(batch, x, pairs)
    m, o = batch.material, pairs.obj
    ke, kd, mu_f, eps = m.ke[o], m.kd[o], m.mu_f[o], m.friction_eps[o]
    d = r_total - gap
    pen = torch.relu(d)
    active = (d > 0).to(x.dtype)
    vn = (n * delta).sum(-1)
    E_n = 0.5 * ke * pen**2
    E_d = 0.5 * kd * torch.relu(-vn) ** 2 * active
    u = delta - vn[:, None] * n
    y = (u * u).sum(-1).clamp_min(1e-24).sqrt()  # zero gradient at u = 0
    f_n = (ke * pen).detach() if load is None else load
    E_f = mu_f * f_n * _f0(y, eps)
    return E_n, E_d, E_f


def contact_energy(batch, x: Tensor) -> Tensor:
    """Contact energy per object [O] in mu h^3 units (a pair's energy goes to the sample's object); zeros without
    pairs. Differentiable in x, body pairs included."""
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
    equally over the four face corners, plus the reaction of the body pairs in which the object is the partner."""
    pairs = batch.pairs
    out = torch.zeros(batch.O, 3, dtype=x.dtype, device=x.device)
    if pairs is None or pairs.count == 0:
        return out
    with torch.enable_grad():
        xg = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(contact_energy(batch, xg).sum(), xg)
    return out.index_add_(0, batch.corner_obj, -g)


def pair_stiffness(batch, x: Tensor, pairs: Pairs) -> tuple[Tensor, Tensor]:
    """Per pair at x: (ke_active [Q], K [Q,3,3]) with K the Hessian of the pair's energy with respect to a rigid
    translation of the sample relative to its partner, the normal load held as in the energy:

        K = ke n n^T [d > 0] + kd n n^T [d > 0, vn < 0] + mu_f f_n ((f0'' - f0'/y) u u^T / y^2 + (f0'/y) (I - n n^T))

    (f_n = ke relu(d); inside the friction band f0'/y = 2/eps - y/eps^2 and f0'' = 2/eps - 2y/eps^2, outside 1/y and
    0, so a sticking pair resists tangential motion with 2 mu_f f_n / eps). Zero for invalid rows. Detached."""
    _, _, n, gap, r_total, delta = _geometry(batch, x.detach(), pairs)
    m, o = batch.material, pairs.obj
    ke, kd, mu_f, eps = (t[o].to(x.dtype) for t in (m.ke, m.kd, m.mu_f, m.friction_eps))
    d = r_total - gap
    active = ((d > 0) & pairs.valid).to(x.dtype)
    vn = (n * delta).sum(-1)
    u = delta - vn[:, None] * n
    y = (u * u).sum(-1).clamp_min(1e-24).sqrt()
    inside = y < eps
    f1_y = torch.where(inside, 2.0 / eps - y / eps**2, 1.0 / y)  # f0'(y) / y
    f2 = torch.where(inside, 2.0 / eps - 2.0 * y / eps**2, torch.zeros_like(y))  # f0''(y)
    f_n = mu_f * ke * torch.relu(d) * active
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    nn = n[:, :, None] * n[:, None, :]
    uu = u[:, :, None] * u[:, None, :] / (y * y)[:, None, None]
    normal = ke * active + kd * active * (vn < 0).to(x.dtype)
    K = normal[:, None, None] * nn + f_n[:, None, None] * (
        (f2 - f1_y)[:, None, None] * uu + f1_y[:, None, None] * (eye - nn)
    )
    return ke * active, K


def active_stiffness(batch, x: Tensor) -> tuple[Tensor, Tensor]:
    """(sum_active ke [O], H [O,3,3]) over the pairs at x: the Picard constant's numerator (eq. 7.13, the normal
    stiffness of the penetrating pairs, d = r - gap > 0) and the contact part of the translation residual's Jacobian
    (eq. 7.14), the sum of the pair stiffnesses `pair_stiffness` (normal, damping while approaching, friction
    curvature with the held load). A body pair stiffens both bodies' translations. Detached."""
    pairs = batch.pairs
    O = batch.O
    ke_sum = torch.zeros(O, dtype=x.dtype, device=x.device)
    H = torch.zeros(O, 3, 3, dtype=x.dtype, device=x.device)
    if pairs is None or pairs.count == 0:
        return ke_sum, H
    ke, K = pair_stiffness(batch, x, pairs)
    ke_sum = ke_sum.index_add_(0, pairs.obj, ke)
    H = H.index_add_(0, pairs.obj, K)
    if batch.body_contact:
        partner = pairs.partner_body
        has = (partner >= 0).to(x.dtype)
        ke_sum = ke_sum.index_add_(0, partner.clamp_min(0), ke * has)
        H = H.index_add_(0, partner.clamp_min(0), K * has[:, None, None])
    return ke_sum, H


def translation_hessian(batch, x: Tensor) -> tuple[Tensor, Tensor]:
    """(sum_active ke [O] as `active_stiffness`, J [O,O,3,3]): the contact part of the Jacobian of the translation
    residuals (eq. 7.14) with respect to the bodies' rigid translations at x. Block (o, o) = the sum of
    `pair_stiffness` over the pairs o owns or partners (= H of `active_stiffness`), block (o, o') = minus the sum
    over the pairs between o and o'. Symmetric positive semidefinite; `Step.centroid_target` adds M_tot I on the
    diagonal for the coupled Newton step on the centroids of a scene with body pairs. Detached."""
    pairs = batch.pairs
    O = batch.O
    ke_sum = torch.zeros(O, dtype=x.dtype, device=x.device)
    J = torch.zeros(O * O, 3, 3, dtype=x.dtype, device=x.device)
    if pairs is None or pairs.count == 0:
        return ke_sum, J.view(O, O, 3, 3)
    ke, K = pair_stiffness(batch, x, pairs)
    o = pairs.obj
    ke_sum = ke_sum.index_add_(0, o, ke)
    J = J.index_add_(0, o * O + o, K)
    if batch.body_contact:
        partner = pairs.partner_body
        has = (partner >= 0).to(x.dtype)
        q = partner.clamp_min(0)
        ke_sum = ke_sum.index_add_(0, q, ke * has)
        has = has[:, None, None]
        J = J.index_add_(0, q * O + q, K * has)
        J = J.index_add_(0, o * O + q, -K * has)
        J = J.index_add_(0, q * O + o, -K * has)
    return ke_sum, J.view(O, O, 3, 3)


def contact_tokens(batch, x: Tensor, R: Tensor) -> Tensor:
    """One 19-channel token per pair in the owning cell's frozen frame R[cell] (note section 6).

    Channels: sample position (3), partner point (3), partner normal (3), all relative to the cell centre in the
    cell frame; gap / r; approach rate -(n . delta) / r; r_p / r (capped; 1 for planes and body faces); log1p(kappa);
    beta; mu_f; kind one-hot (plane, point, other body); self flag.
    """
    pairs = batch.pairs
    Q = pairs.count
    if Q == 0:
        return torch.zeros(0, TOKEN_DIM, dtype=x.dtype, device=x.device)
    xs, p, n, gap, _, delta = _geometry(batch, x, pairs)
    centre = x[batch.cells[pairs.cell]].mean(1)
    Rt = R[pairs.cell].transpose(-1, -2).to(x.dtype)
    loc = lambda v: torch.einsum("qab,qb->qa", Rt, v)  # noqa: E731
    m, o = batch.material, pairs.obj
    approach = -(n * delta).sum(-1) / R_SAMPLE
    return torch.cat(
        [
            loc(xs - centre),
            loc(p - centre),
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
    _, _, _, gap, r_total, _ = _geometry(batch, x, pairs)
    pen = (torch.relu(r_total - gap) / R_SAMPLE).masked_fill(~pairs.valid, 0.0)
    return out.scatter_reduce(0, pairs.obj, pen, reduce="amax", include_self=True)


def kind_penetration(batch, x: Tensor) -> Tensor:
    """Maximum penetration depth in units of r per pair kind [3] (plane, static point, other body) over the batch's
    valid pairs at x; zeros without pairs (the v5 validation's plane and inter-body penetration)."""
    pairs = batch.pairs
    out = torch.zeros(3, dtype=x.dtype, device=x.device)
    if pairs is None or pairs.count == 0:
        return out
    _, _, _, gap, r_total, _ = _geometry(batch, x, pairs)
    pen = (torch.relu(r_total - gap) / R_SAMPLE).masked_fill(~pairs.valid, 0.0)
    return out.scatter_reduce(0, pairs.kind, pen, reduce="amax", include_self=True)
