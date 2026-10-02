# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact energy of every pair and its analytic gradient in one float32 Warp kernel (the inference path of
`contact.contact_energy`; the torch `contact.pair_energies` stays the reference and the training path), plus the
mesh-query kernel of the body-body detection (`contact.body_candidates`).

One thread per pair row of the capacity layout (`valid` mask, padded rows do nothing). With sample position
xs = mean of the 4 face corners, partner normal n, gap = (xs - p) . n, d = r - gap, delta the step displacement,
vn = n . delta, u = delta - vn n, y = |u| (floored at 1e-12 as the torch path's clamp of |u|^2 at 1e-24):

    E_n = ke/2 relu(d)^2                 dE_n/dxs = -ke relu(d) n
    E_d = kd/2 relu(-vn)^2 [d > 0]       dE_d/dxs = -kd relu(-vn) n [d > 0]
    E_f = mu_f f_n f0(y)                 dE_f/dxs = (I - n n^T) mu_f f_n f0'(y) u / y    (f_n = ke relu(d) held)

with the IPC smooth friction potential f0(y) = -y^3/(3 eps^2) + y^2/eps + eps/3 for y < eps, y otherwise, and
Newton's constant normal load f_n in the friction term (the torch path detaches it). Static partners (kinds 0, 1)
have the detection's p, n and delta = xs - anchor.

Body pairs (kind 2, `partner_face >= 0`; contact.py module docstring) rebuild the geometry from the partner's current
corners c_i and their step-start values C_i: w = closest-point weights on the quad (held), p = sum w_i c_i,
n = unit (c_2 - c_0) x (c_3 - c_1) (rest normal if degenerate), delta = (xs - anchor) - sum w_i (c_i - C_i). The
partner corners receive -w_i times the sample gradient (E depends on xs - p only) plus the gradient through n:
dE/dn = -ke relu(d) (xs - p) - kd relu(-vn) delta [d > 0] - (mu_f f_n f0'(y) / y) vn u, pulled back through the
normalisation and the cross product (dE/da = b x g_m, dE/db = g_m x a with m = a x b, g_m = (I - n n^T) dE/dn / |m|).

The pair energy goes to the owning object (atomic add into [O]); the gradients either come out per pair ([Q,3] for
the sample, [Q,4,3] for the partner corners; the torch.autograd.Function used by `contact_energy`) or are scattered
straight onto the corners (sample corners with weight 1/4, partner corners as computed; atomic adds into [N,3], the
fused inference pass of `energy_kernel.energy_and_grad_warp`).
"""

from __future__ import annotations

import torch
import warp as wp

Tensor = torch.Tensor


@wp.func
def triangle_closest_weights(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    """Barycentric weights of the closest point to p on the triangle (a, b, c) (Ericson 5.1.5)."""
    ab = b - a
    ac = c - a
    d1 = wp.dot(ab, p - a)
    d2 = wp.dot(ac, p - a)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)
    d3 = wp.dot(ab, p - b)
    d4 = wp.dot(ac, p - b)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)
    d5 = wp.dot(ab, p - c)
    d6 = wp.dot(ac, p - c)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3(0.0, 1.0 - w, w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def quad_closest_weights(p: wp.vec3, c0: wp.vec3, c1: wp.vec3, c2: wp.vec3, c3: wp.vec3) -> wp.vec4:
    """Weights over the four corners of the closest point to p on the quad = triangles (0, 1, 2), (0, 2, 3); the
    nearer triangle, the first on ties (as `contact.closest_point_on_quad`)."""
    w1 = triangle_closest_weights(p, c0, c1, c2)
    w2 = triangle_closest_weights(p, c0, c2, c3)
    p1 = w1[0] * c0 + w1[1] * c1 + w1[2] * c2
    p2 = w2[0] * c0 + w2[1] * c2 + w2[2] * c3
    if wp.length_sq(p - p1) <= wp.length_sq(p - p2):
        return wp.vec4(w1[0], w1[1], w1[2], 0.0)
    return wp.vec4(w2[0], 0.0, w2[1], w2[2])


@wp.kernel
def contact_pair_kernel(
    x: wp.array[wp.vec3],
    X: wp.array[wp.vec3],  # step-start corners (the partner material point's anchor)
    sample_corners: wp.array2d[wp.int64],  # [S,4]
    sample_face: wp.array[wp.int64],  # [S]
    face_normals: wp.array[wp.vec3],  # [6] rest normals (degenerate partner quads)
    sample: wp.array[wp.int64],  # [Q]
    obj: wp.array[wp.int64],  # [Q]
    partner_point: wp.array[wp.vec3],  # [Q]
    partner_normal: wp.array[wp.vec3],  # [Q]
    partner_face: wp.array[wp.int64],  # [Q] global sample id of the partner face, -1 for static partners
    anchor: wp.array[wp.vec3],  # [Q]
    valid: wp.array[wp.bool],  # [Q]
    ke: wp.array[float],  # [O]
    kd: wp.array[float],
    mu_f: wp.array[float],
    friction_eps: wp.array[float],
    r_sample: float,
    scatter: int,  # 0: grad_pair[q], grad_partner[q]; 1: grad_x[corner] += (sample 1/4 each, partner as computed)
    energy_obj: wp.array[float],  # [O], accumulated
    grad_pair: wp.array[wp.vec3],  # [Q] (scatter 0)
    grad_partner: wp.array2d[wp.vec3],  # [Q,4] (scatter 0)
    grad_x: wp.array[wp.vec3],  # [N] (scatter 1)
):
    q = wp.tid()
    if not valid[q]:
        if scatter == 0:
            grad_pair[q] = wp.vec3(0.0)
            for i in range(4):
                grad_partner[q, i] = wp.vec3(0.0)
        return
    s = int(sample[q])
    c0 = int(sample_corners[s, 0])
    c1 = int(sample_corners[s, 1])
    c2 = int(sample_corners[s, 2])
    c3 = int(sample_corners[s, 3])
    xs = 0.25 * (x[c0] + x[c1] + x[c2] + x[c3])
    o = int(obj[q])
    n = partner_normal[q]
    p = partner_point[q]
    delta = xs - anchor[q]
    # body pair: geometry from the partner's current corners
    pf = int(partner_face[q])
    body = pf >= 0
    d0 = int(0)
    d1 = int(0)
    d2 = int(0)
    d3 = int(0)
    w = wp.vec4(0.0)
    da = wp.vec3(0.0)
    db = wp.vec3(0.0)
    mlen = float(0.0)
    if body:
        d0 = int(sample_corners[pf, 0])
        d1 = int(sample_corners[pf, 1])
        d2 = int(sample_corners[pf, 2])
        d3 = int(sample_corners[pf, 3])
        e0 = x[d0]
        e1 = x[d1]
        e2 = x[d2]
        e3 = x[d3]
        w = quad_closest_weights(xs, e0, e1, e2, e3)
        p = w[0] * e0 + w[1] * e1 + w[2] * e2 + w[3] * e3
        da = e2 - e0
        db = e3 - e1
        m = wp.cross(da, db)
        mlen = wp.length(m)
        if mlen > 1e-12:
            n = m / mlen
        else:
            n = face_normals[int(sample_face[pf])]
        delta = delta - (w[0] * (e0 - X[d0]) + w[1] * (e1 - X[d1]) + w[2] * (e2 - X[d2]) + w[3] * (e3 - X[d3]))
    gap = wp.dot(xs - p, n)
    d = r_sample - gap
    pen = wp.max(d, 0.0)
    vn = wp.dot(n, delta)
    ke_o = ke[o]
    # normal
    E = 0.5 * ke_o * pen * pen
    g = (-ke_o * pen) * n
    g_n = (-ke_o * pen) * (xs - p)
    # damping, gated on penetration
    if d > 0.0:
        approach = wp.max(-vn, 0.0)
        E += 0.5 * kd[o] * approach * approach
        g += (-kd[o] * approach) * n
        g_n += (-kd[o] * approach) * delta
    # friction with the held normal load
    u = delta - vn * n
    uu = wp.dot(u, u)
    y = wp.sqrt(wp.max(uu, 1e-24))
    eps = friction_eps[o]
    load = mu_f[o] * ke_o * pen
    if y < eps:
        E += load * (-(y * y * y) / (3.0 * eps * eps) + y * y / eps + eps / 3.0)
        df = load * (-(y * y) / (eps * eps) + 2.0 * y / eps)
    else:
        E += load * y
        df = load
    if uu >= 1e-24:  # the torch path's clamp_min passes no gradient below the floor
        gu = (df / y) * u
        g += gu - wp.dot(n, gu) * n
        g_n += (-(df / y) * vn) * u
    wp.atomic_add(energy_obj, o, E)
    # partner corners: -w_i g (E depends on xs - p only) plus the gradient through the normal
    gp0 = wp.vec3(0.0)
    gp1 = wp.vec3(0.0)
    gp2 = wp.vec3(0.0)
    gp3 = wp.vec3(0.0)
    if body:
        gp0 = -w[0] * g
        gp1 = -w[1] * g
        gp2 = -w[2] * g
        gp3 = -w[3] * g
        if mlen > 1e-12:
            g_m = (g_n - wp.dot(n, g_n) * n) / mlen
            ta = wp.cross(db, g_m)  # dE/da, a = c2 - c0
            tb = wp.cross(g_m, da)  # dE/db, b = c3 - c1
            gp2 += ta
            gp0 -= ta
            gp3 += tb
            gp1 -= tb
    if scatter == 0:
        grad_pair[q] = g
        grad_partner[q, 0] = gp0
        grad_partner[q, 1] = gp1
        grad_partner[q, 2] = gp2
        grad_partner[q, 3] = gp3
    else:
        gc = 0.25 * g
        wp.atomic_add(grad_x, c0, gc)
        wp.atomic_add(grad_x, c1, gc)
        wp.atomic_add(grad_x, c2, gc)
        wp.atomic_add(grad_x, c3, gc)
        if body:
            wp.atomic_add(grad_x, d0, gp0)
            wp.atomic_add(grad_x, d1, gp1)
            wp.atomic_add(grad_x, d2, gp2)
            wp.atomic_add(grad_x, d3, gp3)


@wp.kernel
def pair_geometry_kernel(
    x: wp.array[wp.vec3],
    X: wp.array[wp.vec3],
    sample_corners: wp.array2d[wp.int64],  # [S,4]
    sample_face: wp.array[wp.int64],  # [S]
    face_normals: wp.array[wp.vec3],  # [6]
    sample: wp.array[wp.int64],  # [Q]
    partner_point: wp.array[wp.vec3],  # [Q]
    partner_normal: wp.array[wp.vec3],  # [Q]
    partner_face: wp.array[wp.int64],  # [Q]
    anchor: wp.array[wp.vec3],  # [Q]
    valid: wp.array[wp.bool],  # [Q]
    xs_out: wp.array[wp.vec3],  # [Q]
    p_out: wp.array[wp.vec3],
    n_out: wp.array[wp.vec3],
    gap_out: wp.array[float],
    delta_out: wp.array[wp.vec3],
):
    """The pair geometry of `contact._geometry` per row (sample position, partner point and normal, gap, step
    displacement; body pairs from the partner's current corners as `contact_pair_kernel`), zeros on padded rows:
    the capacity layout evaluates the tokens, the translation stiffness and the penetration over 4-5 rows per
    sample of which a few per cent are valid, and the torch chain over all of them cost 3-4 ms per evaluation."""
    q = wp.tid()
    if not valid[q]:
        xs_out[q] = wp.vec3(0.0)
        p_out[q] = wp.vec3(0.0)
        n_out[q] = wp.vec3(0.0)
        gap_out[q] = 0.0
        delta_out[q] = wp.vec3(0.0)
        return
    s = int(sample[q])
    xs = 0.25 * (
        x[int(sample_corners[s, 0])]
        + x[int(sample_corners[s, 1])]
        + x[int(sample_corners[s, 2])]
        + x[int(sample_corners[s, 3])]
    )
    n = partner_normal[q]
    p = partner_point[q]
    delta = xs - anchor[q]
    pf = int(partner_face[q])
    if pf >= 0:
        d0 = int(sample_corners[pf, 0])
        d1 = int(sample_corners[pf, 1])
        d2 = int(sample_corners[pf, 2])
        d3 = int(sample_corners[pf, 3])
        e0 = x[d0]
        e1 = x[d1]
        e2 = x[d2]
        e3 = x[d3]
        w = quad_closest_weights(xs, e0, e1, e2, e3)
        p = w[0] * e0 + w[1] * e1 + w[2] * e2 + w[3] * e3
        m = wp.cross(e2 - e0, e3 - e1)
        mlen = wp.length(m)
        if mlen > 1e-12:
            n = m / mlen
        else:
            n = face_normals[int(sample_face[pf])]
        delta = delta - (w[0] * (e0 - X[d0]) + w[1] * (e1 - X[d1]) + w[2] * (e2 - X[d2]) + w[3] * (e3 - X[d3]))
    xs_out[q] = xs
    p_out[q] = p
    n_out[q] = n
    gap_out[q] = wp.dot(xs - p, n)
    delta_out[q] = delta


@wp.kernel
def body_query_kernel(
    xs: wp.array[wp.vec3],  # [S] sample positions at X
    sample_obj: wp.array[wp.int64],  # [S]
    reach: wp.array[float],  # [S] query bound (margin + r)
    mesh_ids: wp.array[wp.uint64],  # [O]
    lo: wp.array[wp.vec3],  # [O] bounding boxes of the objects' corners
    hi: wp.array[wp.vec3],
    sample_off: wp.array[wp.int64],  # [O+1]
    n_rays: int,
    ok: wp.array2d[wp.bool],  # [S,O]
    dist: wp.array2d[float],  # [S,O] signed distance to the surface (negative inside)
    face: wp.array2d[wp.int64],  # [S,O] global sample id of the closest face
    point: wp.array2d[wp.vec3],  # [S,O] closest point
):
    """One thread per (sample, other object): the closest point on the object's surface mesh within `reach`, its
    face and the inside / outside sign by ray parity; nothing for the sample's own object or outside the object's
    box grown by the reach."""
    s, o = wp.tid()
    if int(sample_obj[s]) == o:
        return
    p = xs[s]
    r = reach[s]
    lo_o = lo[o]
    hi_o = hi[o]
    if p[0] < lo_o[0] - r or p[1] < lo_o[1] - r or p[2] < lo_o[2] - r:
        return
    if p[0] > hi_o[0] + r or p[1] > hi_o[1] + r or p[2] > hi_o[2] + r:
        return
    res = wp.mesh_query_point_sign_parity(mesh_ids[o], p, r, n_rays, 0.1)
    if not res.result:
        return
    cp = wp.mesh_eval_position(mesh_ids[o], res.face, res.u, res.v)
    d = wp.length(cp - p)
    if res.sign < 0.0:
        d = -d
    ok[s, o] = True
    dist[s, o] = d
    face[s, o] = sample_off[o] + wp.int64(res.face // 2)
    point[s, o] = cp


def _array(t: Tensor | None, dtype):
    return None if t is None else wp.from_torch(t.contiguous(), dtype=dtype, requires_grad=False, return_ctype=True)


def _stream(x: Tensor):
    return wp.stream_from_torch(torch.cuda.current_stream(x.device)) if x.is_cuda else None


def launch_body_query(xs, sample_obj, reach, mesh_ids, lo, hi, sample_off, ok, dist, face, point, n_rays) -> None:
    """Fill the [S,O] candidate arrays of `contact.body_candidates` (float32 / int64 tensors on one device)."""
    S, O = ok.shape
    if S == 0 or O == 0:
        return
    wp.init()
    inputs = [
        _array(xs, wp.vec3),
        _array(sample_obj, wp.int64),
        _array(reach, wp.float32),
        mesh_ids,
        _array(lo, wp.vec3),
        _array(hi, wp.vec3),
        _array(sample_off, wp.int64),
        int(n_rays),
        _array(ok, wp.bool),
        _array(dist, wp.float32),
        _array(face, wp.int64),
        _array(point, wp.vec3),
    ]
    wp.launch(body_query_kernel, dim=(S, O), inputs=inputs, device=wp.device_from_torch(xs.device), stream=_stream(xs))


def launch_contact(
    batch,
    x: Tensor,
    pairs,
    r_sample: float,
    energy_obj: Tensor,
    grad_pair: Tensor | None,
    grad_x: Tensor | None,
    grad_partner: Tensor | None = None,
) -> None:
    """Accumulate the pair energies into `energy_obj` [O] and write the gradients per pair (`grad_pair` [Q,3] for
    the sample, `grad_partner` [Q,4,3] for the partner corners of body pairs) or scatter them onto the corners
    (`grad_x` [N,3], accumulated). x, the pair fields and the material are float32 / int64 CUDA tensors;
    `r_sample` is the sample radius (`contact.R_SAMPLE`); the partner anchors read `batch.X`."""
    Q = pairs.count
    if Q == 0:
        return
    wp.init()
    m = batch.material
    face_normals = batch.hc.face_normals.to(torch.float32)  # kept alive until the launch (a view on float32 batches)
    inputs = [
        _array(x, wp.vec3),
        _array(batch.X, wp.vec3),
        _array(batch.sample_corners, wp.int64),
        _array(batch.sample_face, wp.int64),
        _array(face_normals, wp.vec3),
        _array(pairs.sample, wp.int64),
        _array(pairs.obj, wp.int64),
        _array(pairs.partner_point, wp.vec3),
        _array(pairs.partner_normal, wp.vec3),
        _array(pairs.partner_face, wp.int64),
        _array(pairs.anchor, wp.vec3),
        _array(pairs.valid, wp.bool),
        _array(m.ke, wp.float32),
        _array(m.kd, wp.float32),
        _array(m.mu_f, wp.float32),
        _array(m.friction_eps, wp.float32),
        float(r_sample),
        int(grad_pair is None),
        _array(energy_obj, wp.float32),
        _array(grad_pair, wp.vec3),
        _array(grad_partner, wp.vec3),
        _array(grad_x, wp.vec3),
    ]
    wp.launch(contact_pair_kernel, dim=Q, inputs=inputs, stream=_stream(x))


def pair_geometry_warp(batch, x: Tensor, pairs) -> tuple:
    """(xs, p, n, gap, delta) of every pair row at x through `pair_geometry_kernel` (float32 CUDA, capacity
    layout; detached): the Warp counterpart of `contact._geometry` without r_total."""
    Q = pairs.count
    x = x.detach().contiguous()
    out = torch.zeros(Q, 13, dtype=torch.float32, device=x.device)
    xs, p, n, gap, delta = out[:, 0:3], out[:, 3:6], out[:, 6:9], out[:, 9], out[:, 10:13]
    xs, p, n, gap, delta = (t.contiguous() for t in (xs, p, n, gap, delta))  # separate buffers for the kernel
    if Q == 0:
        return xs, p, n, gap, delta
    wp.init()
    face_normals = batch.hc.face_normals.to(torch.float32)  # kept alive until the launch
    inputs = [
        _array(x, wp.vec3),
        _array(batch.X, wp.vec3),
        _array(batch.sample_corners, wp.int64),
        _array(batch.sample_face, wp.int64),
        _array(face_normals, wp.vec3),
        _array(pairs.sample, wp.int64),
        _array(pairs.partner_point, wp.vec3),
        _array(pairs.partner_normal, wp.vec3),
        _array(pairs.partner_face, wp.int64),
        _array(pairs.anchor, wp.vec3),
        _array(pairs.valid, wp.bool),
        _array(xs, wp.vec3),
        _array(p, wp.vec3),
        _array(n, wp.vec3),
        _array(gap, wp.float32),
        _array(delta, wp.vec3),
    ]
    wp.launch(pair_geometry_kernel, dim=Q, inputs=inputs, stream=_stream(x))
    return xs, p, n, gap, delta


class ContactEnergy(torch.autograd.Function):
    """E [O] = contact energy per object; backward scatters the saved per-pair gradients, scaled by the incoming
    per-object gradient, onto the 4 sample corners (1/4 each) and the 4 partner corners of body pairs. Only x
    carries a gradient."""

    @staticmethod
    def forward(ctx, x, batch, pairs, r_sample):
        x = x.contiguous()
        O = batch.O
        energy = torch.zeros(O, dtype=x.dtype, device=x.device)
        grad_pair = torch.empty(pairs.count, 3, dtype=x.dtype, device=x.device)
        grad_partner = torch.empty(pairs.count, 4, 3, dtype=x.dtype, device=x.device)
        launch_contact(batch, x, pairs, r_sample, energy, grad_pair, None, grad_partner)
        partner_corners = batch.sample_corners[pairs.partner_face.clamp_min(0)]
        ctx.save_for_backward(pairs.obj, batch.sample_corners[pairs.sample], grad_pair, partner_corners, grad_partner)
        ctx.N = x.shape[0]
        ctx.body = bool(batch.body_contact)
        return energy

    @staticmethod
    def backward(ctx, grad_out):
        obj, corners, grad_pair, partner_corners, grad_partner = ctx.saved_tensors
        scale = grad_out[obj][:, None]  # [Q,1]
        per_corner = (0.25 * scale) * grad_pair  # [Q,3]
        grad_x = torch.zeros(ctx.N, 3, dtype=grad_pair.dtype, device=grad_pair.device)
        grad_x.index_add_(0, corners.reshape(-1), per_corner[:, None, :].expand(-1, 4, 3).reshape(-1, 3))
        if ctx.body:  # static partners have zero partner gradients
            grad_x.index_add_(0, partner_corners.reshape(-1), (scale[:, None, :] * grad_partner).reshape(-1, 3))
        return grad_x, None, None, None


def contact_energy_warp(batch, x: Tensor, r_sample: float) -> Tensor:
    """Contact energy per object [O] through the kernel (float32 CUDA, capacity-layout pairs), differentiable in x."""
    return ContactEnergy.apply(x, batch, batch.pairs, r_sample)
