# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The solver step (design spec 4.1, 4.3): prepare (step-constant tier), query (one learned proposal fused on
the fixed objective, one energy pass), advance (after K queries), commit (carry the query's results).

All tensors live in normalised units on the batch's device. prepare / advance take a boolean object mask and
are whole-batch tensor ops: recomputing a step-constant quantity for an unchanged object reproduces it exactly,
so only the candidate needs the mask.

Unpinned bodies (derivation note section 7; decision of 2026-10-01). The fusion cannot move the centroid of a
body without pins (7.1), so every query gives `Fusion.fuse` a centroid target for the free objects, recomputed
from the contact force of the current candidate: the rigid semi-implicit step c_rig(x_k) of eq. 7.11 (Picard
iteration on the implicit translation equation 7.9, contraction constant sum_active ke / M_tot by 7.13;
`translation="picard"`) or the semismooth Newton step on the centroid of eq. 7.15 (`"implicit_contact"`, the default
since Anka's decision of 2026-10-01: the Picard constant 7.13 exceeds 1 over most of the campaign's contact range) with the active-pair
contact Hessian (`translation="implicit_contact"`: the contact force implicit in the translation; same fixed point,
no contraction condition). With dt = 1 and the corner masses rho m_i:
c_rig = c_n + cdot_n + g + F_con(x_k) / M_tot. `prepare` stores c_n and cdot_n for the step, checks the Picard
constant (7.13) at the step's candidate and warns once per object when it exceeds 1; `query` writes the constant of
its candidate into `batch.picard_constant`.
"""

from __future__ import annotations

import warnings

import torch

from . import contact, physics
from . import hex as hx
from .features import edge_features, features, node_features
from .frames import frames
from .fusion import Fusion
from .grid import reference_rotation
from .structs import QueryOutput

Tensor = torch.Tensor


TRANSLATIONS = ("picard", "implicit_contact")


def _unwrap(net):
    return net.module if hasattr(net, "module") else net  # DistributedDataParallel


def _solve3(A: Tensor, b: Tensor) -> Tensor:
    """x = A^-1 b for a batch of 3x3 matrices [O,3,3] and vectors [O,3] by the adjugate (no library call, no host
    synchronisation; A is symmetric positive definite here)."""
    c0 = torch.linalg.cross(A[:, :, 1], A[:, :, 2])  # cofactor columns
    c1 = torch.linalg.cross(A[:, :, 2], A[:, :, 0])
    c2 = torch.linalg.cross(A[:, :, 0], A[:, :, 1])
    det = (A[:, :, 0] * c0).sum(-1, keepdim=True)
    adj_t = torch.stack([c0, c1, c2], 1)  # [O,3,3]: row a = cofactor column a, so adj_t @ b = adj(A) b
    return torch.einsum("oab,ob->oa", adj_t, b) / det


def translation_residual(batch, x: Tensor, F_con: Tensor | None = None) -> Tensor:
    """r_tr(c(x)) of eq. 7.14 per object [O,3], dt = 1: M_tot (c(x) - c_n - cdot_n - g) - F_con(x), detached. Zero
    at the translation component of stationarity (7.9); meaningful for unpinned objects (requires `prepare`)."""
    x = x.detach()
    if F_con is None:
        F_con = contact.contact_force(batch, x)
    M_tot = physics.total_mass(batch)[:, None]
    return M_tot * (physics.centroid(batch, x) - batch.c_n - batch.cdot_n - batch.material.g) - F_con


class Step:
    def __init__(
        self,
        net,
        fusion: Fusion | None = None,
        augmenter=None,
        noise_prob: float = 0.5,
        noise_range=(0.01, 0.10),
        pair_capacity: bool = False,
        translation: str = "implicit_contact",
    ):
        if translation not in TRANSLATIONS:
            raise ValueError(f"translation must be one of {TRANSLATIONS}, got {translation!r}")
        self.net = net
        self.fusion = fusion or Fusion()
        self.aug = augmenter
        self.noise_prob = noise_prob
        self.noise_range = noise_range
        self.pair_capacity = pair_capacity  # static pair buffers with a valid mask (capture.py); training: compacted
        self.translation = (
            translation  # centroid target of unpinned objects: "implicit_contact" (7.15, default) or "picard" (7.11)
        )
        self.node_features = node_features
        self.edge_features = edge_features
        # the "pair" edge module reads the per-edge sender block; the a02 module does not need it built
        self.sender = getattr(_unwrap(net), "edge_module", "a02") == "pair"
        self._picard_warned: Tensor | None = None  # [O] bool, one warning per object slot

    def compile_features(self, **kwargs) -> Step:
        """torch.compile the per-cell and per-edge feature chains (static shapes; the rollout path)."""
        self.node_features = torch.compile(node_features, dynamic=False, fullgraph=True, **kwargs)
        self.edge_features = torch.compile(edge_features, dynamic=False, fullgraph=True, **kwargs)
        return self

    # ------------------------------------------------------------------ query
    def query(self, batch) -> QueryOutput:
        x = batch.x
        m_c, F_c = physics.modes_and_center(x, batch)
        R = frames(F_c.detach().float(), batch.R_ref[batch.cell_obj].float()).to(x.dtype)  # float32 Warp kernel
        g_m = self.fusion.project_gradient(batch, batch.gX)
        feats = features(batch, x, R, m_c, F_c, g_m, self.node_features, self.edge_features, self.sender)
        out = self.net(feats, batch.edges, batch.edge_offsets, batch.cell_obj)
        dm_world = torch.einsum("cab,cvb->cva", R, out.step[:, None, None] * out.corr)
        c_t, picard = self.centroid_target(batch, x) if batch.any_free else (None, None)
        d = self.fusion.fuse(batch, hx.modes_to_gauss(dm_world, batch.hc), centroid_target=c_t)
        cand = x + d
        E_after, gX_after = physics.energy_and_grad(batch, cand)
        achieved = hx.modes(d.detach()[batch.cells], batch.hc)
        return QueryOutput(
            E_before=batch.E,
            E_after=E_after,
            gX_after=gX_after,
            cand_after=cand,
            dm_world=achieved,
            g_world=g_m.detach(),
            residual=physics.residual(batch.gX, batch),
            inverted=physics.inverted_cells(F_c.detach(), batch),
            step=out.step.detach(),
            picard_constant=picard,
        )

    @torch.no_grad()
    def centroid_target(self, batch, x: Tensor) -> tuple[Tensor, Tensor]:
        """(c_t [O,3], Picard constant [O]) at the candidate x for the unpinned objects (pinned objects get their
        own centroid, which `fuse` ignores). Picard: c_rig(x_k) of eq. 7.11. Implicit contact: the Newton step on
        the centroid, c(x_k) + dc with (M_tot I + sum_active ke n n^T) dc = -r_tr(c(x_k)), eq. 7.15; with body pairs
        (`batch.body_contact`) the Newton step is taken jointly on all free bodies' centroids with the coupled
        translational contact Hessian (a 3O x 3O dense solve; `contact.translation_hessian`). Detached; the
        Picard constant sum_active ke / M_tot (7.13) is also written into `batch.picard_constant`."""
        x = x.detach()
        free = batch.free_objects
        F_con = contact.contact_force(batch, x)
        coupled_bodies = batch.body_contact and batch.O > 1
        if coupled_bodies:
            ke_sum, J = contact.translation_hessian(batch, x)
        else:
            ke_sum, H = contact.active_stiffness(batch, x)
        M_tot = physics.total_mass(batch)
        c_x = physics.centroid(batch, x)
        c_rig = batch.c_n + batch.cdot_n + batch.material.g + F_con / M_tot[:, None]
        if self.translation == "picard":
            c_t = c_rig
        elif coupled_bodies:
            # body pairs couple the bodies' translations: one Newton step on all centroids with the full
            # translational contact Hessian (contact.translation_hessian), bodies that are not free held fixed
            r_tr = M_tot[:, None] * (c_x - c_rig)
            r_tr = r_tr.masked_fill(~free[:, None], 0.0)
            O = batch.O
            coupled = (free[:, None] & free[None, :])[:, :, None, None]
            J = torch.where(coupled, J, torch.zeros_like(J))
            eye3 = torch.eye(3, dtype=x.dtype, device=x.device)
            J = J + (torch.eye(O, dtype=x.dtype, device=x.device) * M_tot[:, None])[:, :, None, None] * eye3
            dc, _ = torch.linalg.solve_ex(J.permute(0, 2, 1, 3).reshape(3 * O, 3 * O), r_tr.reshape(3 * O, 1))
            c_t = c_x - dc.view(O, 3)
        else:
            r_tr = M_tot[:, None] * (c_x - c_rig)  # eq. 7.14 at c(x_k)
            A = M_tot[:, None, None] * torch.eye(3, dtype=x.dtype, device=x.device)[None] + H
            c_t = c_x - _solve3(A, r_tr)
        c_t = torch.where(free[:, None], c_t, c_x)
        picard = (ke_sum / M_tot).masked_fill(~free, 0.0)
        batch.picard_constant.copy_(picard)
        return c_t, picard

    # ----------------------------------------------------------------- commit
    @torch.no_grad()
    def commit(self, batch, out: QueryOutput) -> None:
        batch.x.copy_(out.cand_after.detach())
        batch.E.copy_(out.E_after.detach())
        batch.gX.copy_(out.gX_after)
        batch.hist_grad.copy_(out.g_world)
        batch.hist_update.copy_(out.dm_world)
        batch.hist_valid.fill_(True)

    # ---------------------------------------------------------------- prepare
    @torch.no_grad()
    def prepare(self, batch, sel: Tensor, gens=None, perturb: Tensor | None = None) -> None:
        """Start of a physical step for the objects in `sel` [O] bool: Y, step-constant modes and damping anchor,
        candidate (inertial or perturbed), detection, the one fresh energy pass. `gens`: the candidate-noise
        stream, a list of per-object generators (body mode: one object at a time, a host synchronisation per
        object) or ONE generator for the whole batch (v5 scenes: `_perturb_batched`, every grid group in one call,
        no host synchronisation; the draw order is the batch's group order, then per group the use flags, the
        RMS values and the octaves of all its objects)."""
        rows = sel[batch.corner_obj]
        Y = batch.X + batch.V + batch.material.g[batch.corner_obj]
        batch.Y = torch.where(batch.pinned[:, None], batch.X, Y)
        batch.m_Y = hx.modes(batch.Y[batch.cells], batch.hc)
        m_prev, _F_prev = physics.modes_and_center(batch.X, batch)
        batch.m_prev = m_prev
        F_prev_q = hx.gauss_deformation(batch.X[batch.cells], batch.hc)
        batch.C_prev = hx.mat3_tn(F_prev_q, F_prev_q)
        batch.R_ref = reference_rotation(batch.X, batch.ref_corners)
        batch.c_n = physics.centroid(batch, batch.X)  # step constants of the free-body centroid target (7.11)
        batch.cdot_n = physics.centroid(batch, batch.V)
        cand = batch.Y.clone()
        if self.aug is not None and isinstance(gens, torch.Generator):
            self._perturb_batched(batch, sel, cand, gens, perturb)
        elif self.aug is not None and gens is not None:
            for o in sel.nonzero().flatten().tolist():
                gen = gens[o]
                use_noise = (
                    perturb[o].item()
                    if perturb is not None
                    else (torch.rand((), generator=gen, device=gen.device).item() < self.noise_prob)
                )
                if use_noise:
                    lo, hi = self.noise_range
                    rng = torch.rand((), generator=gen, device=gen.device).item()
                    sl = slice(int(batch.corner_off[o]), int(batch.corner_off[o + 1]))
                    grid = batch.grids[o]
                    noise = self.aug.candidate_noise(grid, gen, lo + rng * (hi - lo), lo + rng * (hi - lo))
                    noise = noise.to(cand.dtype)  # the float32 field; exact cast, the add below did it implicitly
                    if grid.pinned.numel() == 0:
                        # a free body's candidate keeps the inertial centroid c(Y) = c_n + cdot_n + g (7.6): remove
                        # the mass-weighted mean of the noise (rho is one value per object, so the lumped masses weight)
                        w = grid.mass.to(noise.dtype)
                        noise = noise - (w[:, None] * noise).sum(0) / w.sum()
                    cand[sl] += noise
        batch.x = torch.where(rows[:, None], cand, batch.x)
        batch.pairs = contact.detect(batch, batch.X, batch.V, capacity=self.pair_capacity)
        E, gX = physics.energy_and_grad(batch, batch.x)
        batch.E, batch.gX = E, gX
        if batch.any_free:
            self._check_picard(batch)

    def _perturb_batched(self, batch, sel: Tensor, cand: Tensor, gen: torch.Generator, perturb: Tensor | None) -> None:
        """Candidate noise for the selected objects of the whole batch in a handful of launches (`prepare`): per
        object a use flag (probability `noise_prob`, or `perturb`) and an RMS U(noise_range), then the multiscale
        fields of all objects from the one generator (`Augmenter.candidate_noise_all`); a free body's noise loses
        its mass-weighted mean so the candidate keeps the inertial centroid (7.6). In place on `cand`."""
        lo, hi = self.noise_range
        dev, O = gen.device, batch.O
        use = torch.rand(O, generator=gen, device=dev) < self.noise_prob
        rng = torch.rand(O, generator=gen, device=dev)
        if perturb is not None:
            use = perturb.to(dev)
        rms = lo + rng * (hi - lo)
        noise = self.aug.candidate_noise_all(batch, gen, rms).to(device=cand.device, dtype=cand.dtype)
        obj = batch.corner_obj
        m = batch.mass.to(noise.dtype)  # rho is one value per object: the lumped masses weight the mean
        mean = torch.zeros(O, 3, dtype=noise.dtype, device=cand.device).index_add_(0, obj, m[:, None] * noise)
        mean = mean / torch.zeros(O, dtype=noise.dtype, device=cand.device).index_add_(0, obj, m)[:, None]
        noise = torch.where(batch.free_objects[obj, None], noise - mean[obj], noise)
        active = (use.to(cand.device) & sel)[obj, None].to(noise.dtype)
        cand.add_(noise * active)

    def _check_picard(self, batch) -> None:
        """The Picard constant sum_active ke / M_tot (7.13) at the step's candidate, into `batch.picard_constant`;
        with `translation="picard"` one warning per object slot the first time it exceeds 1 (the rigid target of
        7.11 then does not contract on its own; `translation="implicit_contact"` removes the condition, Proposition
        7.2, so no warning: a v5 scene of 150 bodies exceeds 1 on many of them at every landing). One host sync per
        prepare, only for batches with free bodies and the Picard translation."""
        ke_sum, _ = contact.active_stiffness(batch, batch.x)
        picard = (ke_sum / physics.total_mass(batch)).masked_fill(~batch.free_objects, 0.0)
        batch.picard_constant.copy_(picard)
        if self.translation != "picard":
            return
        if self._picard_warned is None or self._picard_warned.shape != picard.shape:
            self._picard_warned = torch.zeros_like(batch.free_objects)
        bad = (picard > 1.0) & ~self._picard_warned
        if bool(bad.any()):
            for o, value in zip(bad.nonzero().flatten().tolist(), picard[bad].tolist(), strict=True):
                warnings.warn(
                    f"Step: object {o} has Picard constant {value:.3g} > 1 (sum of active contact stiffness over "
                    f"the total mass, derivation eq. 7.13): the rigid centroid target (translation='picard') does "
                    f"not contract; consider translation='implicit_contact'",
                    stacklevel=3,
                )
            self._picard_warned |= bad

    # ---------------------------------------------------------------- advance
    @torch.no_grad()
    def advance(
        self,
        batch,
        sel: Tensor,
        gens=None,
        prescribed: Tensor | None = None,
        prescribed_velocity: Tensor | None = None,
    ) -> None:
        """After K queries: the candidate becomes the new position, V = X - X_prev (dt = 1), then prepare."""
        rows = sel[batch.corner_obj][:, None]
        X_new = torch.where(rows, batch.x, batch.X)
        X_prev_new = torch.where(rows, batch.X, batch.X_prev)
        V_new = torch.where(rows, X_new - X_prev_new, batch.V)
        pin = batch.pinned[:, None] & rows
        if prescribed is not None:
            X_new = torch.where(pin, prescribed, X_new)
        V_new = torch.where(
            pin, prescribed_velocity if prescribed_velocity is not None else torch.zeros_like(V_new), V_new
        )
        batch.X, batch.X_prev, batch.V = X_new, X_prev_new, V_new
        batch.x = torch.where(pin, X_new, batch.x)
        self.prepare(batch, sel, gens)
