# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact energy of every pair and its analytic gradient in one float32 Warp kernel (the inference path of
`contact.contact_energy`; the torch `contact.pair_energies` stays the reference and the training path).

One thread per pair row of the capacity layout (`valid` mask, padded rows do nothing). With sample position
xs = mean of the 4 face corners, partner normal n, gap = (xs - p) . n, d = r - gap, delta = xs - anchor,
vn = n . delta, u = delta - vn n, y = |u| (floored at 1e-12 as the torch path's clamp of |u|^2 at 1e-24):

    E_n = ke/2 relu(d)^2                 dE_n/dxs = -ke relu(d) n
    E_d = kd/2 relu(-vn)^2 [d > 0]       dE_d/dxs = -kd relu(-vn) n [d > 0]
    E_f = mu_f f_n f0(y)                 dE_f/dxs = (I - n n^T) mu_f f_n f0'(y) u / y    (f_n = ke relu(d) held)

with the IPC smooth friction potential f0(y) = -y^3/(3 eps^2) + y^2/eps + eps/3 for y < eps, y otherwise, and
Newton's constant normal load f_n in the friction term (the torch path detaches it). The pair energy goes to the
owning object (atomic add into [O]); the sample gradient either comes out per pair ([Q,3], for the
torch.autograd.Function used by `contact_energy`) or is scattered straight onto the 4 corners with weight 1/4
(atomic adds into [N,3], the fused inference pass of `energy_kernel.energy_and_grad_warp`).
"""

from __future__ import annotations

import torch
import warp as wp

Tensor = torch.Tensor


@wp.kernel
def contact_pair_kernel(
    x: wp.array[wp.vec3],
    sample_corners: wp.array2d[wp.int64],  # [S,4]
    sample: wp.array[wp.int64],  # [Q]
    obj: wp.array[wp.int64],  # [Q]
    partner_point: wp.array[wp.vec3],  # [Q]
    partner_normal: wp.array[wp.vec3],  # [Q]
    anchor: wp.array[wp.vec3],  # [Q]
    valid: wp.array[wp.bool],  # [Q]
    ke: wp.array[float],  # [O]
    kd: wp.array[float],
    mu_f: wp.array[float],
    friction_eps: wp.array[float],
    r_sample: float,
    scatter: int,  # 0: grad_pair[q] = dE/dxs; 1: grad_x[corner] += dE/dxs / 4 for the 4 corners
    energy_obj: wp.array[float],  # [O], accumulated
    grad_pair: wp.array[wp.vec3],  # [Q] (scatter 0)
    grad_x: wp.array[wp.vec3],  # [N] (scatter 1)
):
    q = wp.tid()
    if not valid[q]:
        if scatter == 0:
            grad_pair[q] = wp.vec3(0.0)
        return
    s = int(sample[q])
    c0 = int(sample_corners[s, 0])
    c1 = int(sample_corners[s, 1])
    c2 = int(sample_corners[s, 2])
    c3 = int(sample_corners[s, 3])
    xs = 0.25 * (x[c0] + x[c1] + x[c2] + x[c3])
    n = partner_normal[q]
    o = int(obj[q])
    gap = wp.dot(xs - partner_point[q], n)
    d = r_sample - gap
    pen = wp.max(d, 0.0)
    delta = xs - anchor[q]
    vn = wp.dot(n, delta)
    ke_o = ke[o]
    # normal
    E = 0.5 * ke_o * pen * pen
    g = (-ke_o * pen) * n
    # damping, gated on penetration
    if d > 0.0:
        approach = wp.max(-vn, 0.0)
        E += 0.5 * kd[o] * approach * approach
        g += (-kd[o] * approach) * n
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
    wp.atomic_add(energy_obj, o, E)
    if scatter == 0:
        grad_pair[q] = g
    else:
        gc = 0.25 * g
        wp.atomic_add(grad_x, c0, gc)
        wp.atomic_add(grad_x, c1, gc)
        wp.atomic_add(grad_x, c2, gc)
        wp.atomic_add(grad_x, c3, gc)


def _array(t: Tensor | None, dtype):
    return None if t is None else wp.from_torch(t.contiguous(), dtype=dtype, requires_grad=False, return_ctype=True)


def launch_contact(
    batch, x: Tensor, pairs, r_sample: float, energy_obj: Tensor, grad_pair: Tensor | None, grad_x: Tensor | None
) -> None:
    """Accumulate the pair energies into `energy_obj` [O] and write the sample gradient per pair (`grad_pair`
    [Q,3]) or scatter it onto the corners (`grad_x` [N,3], accumulated). x, the pair fields and the material
    are float32 / int64 CUDA tensors; `r_sample` is the sample radius (`contact.R_SAMPLE`)."""
    Q = pairs.count
    if Q == 0:
        return
    wp.init()
    m = batch.material
    inputs = [
        _array(x, wp.vec3),
        _array(batch.sample_corners, wp.int64),
        _array(pairs.sample, wp.int64),
        _array(pairs.obj, wp.int64),
        _array(pairs.partner_point, wp.vec3),
        _array(pairs.partner_normal, wp.vec3),
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
        _array(grad_x, wp.vec3),
    ]
    stream = wp.stream_from_torch(torch.cuda.current_stream(x.device))
    wp.launch(contact_pair_kernel, dim=Q, inputs=inputs, stream=stream)


class ContactEnergy(torch.autograd.Function):
    """E [O] = contact energy per object; backward scatters the saved per-pair sample gradient, scaled by the
    incoming per-object gradient, onto the 4 face corners (1/4 each). Only x carries a gradient."""

    @staticmethod
    def forward(ctx, x, batch, pairs, r_sample):
        x = x.contiguous()
        O = batch.O
        energy = torch.zeros(O, dtype=x.dtype, device=x.device)
        grad_pair = torch.empty(pairs.count, 3, dtype=x.dtype, device=x.device)
        launch_contact(batch, x, pairs, r_sample, energy, grad_pair, None)
        ctx.save_for_backward(pairs.obj, batch.sample_corners[pairs.sample], grad_pair)
        ctx.N = x.shape[0]
        return energy

    @staticmethod
    def backward(ctx, grad_out):
        obj, corners, grad_pair = ctx.saved_tensors
        per_corner = (0.25 * grad_out[obj])[:, None] * grad_pair  # [Q,3]
        grad_x = torch.zeros(ctx.N, 3, dtype=grad_pair.dtype, device=grad_pair.device)
        grad_x.index_add_(0, corners.reshape(-1), per_corner[:, None, :].expand(-1, 4, 3).reshape(-1, 3))
        return grad_x, None, None, None


def contact_energy_warp(batch, x: Tensor, r_sample: float) -> Tensor:
    """Contact energy per object [O] through the kernel (float32 CUDA, capacity-layout pairs), differentiable in x."""
    return ContactEnergy.apply(x, batch, batch.pairs, r_sample)
