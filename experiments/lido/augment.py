# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multiscale augmenter (design spec 1b "Augmenter", 6.3): smooth random vector fields on the corners.

Everything is in normalised units: positions in cells, velocities in cells per step. Noise is drawn from a
`torch.Generator` living on the augmenter's device, so equal seeds give equal fields.

Trilinear mapping. For wavelength `w` (cells) the coarse lattice has nodes `j` (integer triples) at rest position
`w * j`, with `M_a = n_a // w + 2` nodes along axis `a`: the lattice covers the box [0, n_a] plus one coarse cell of
padding. A corner at rest position `p` (cell units, 0 <= p_a <= n_a) has lattice coordinate `u = p / w`, base node
`i0 = floor(u)` and fraction `f = u - i0`; its value is the trilinear blend of the eight nodes `i0 + {0,1}^3` with
weights `prod_a (1 - f_a or f_a)`. Since `i0_a + 1 <= n_a // w + 1 = M_a - 1` every node exists (no clamping).
"""

from __future__ import annotations

import torch

from .grid import Grid
from .structs import SceneSpec

Tensor = torch.Tensor


# Displacement RMS = perturbation_scale * strength * DISPLACEMENT_SCALE cells. The previous generator (strength times the
# shortest beam side with 1.5-power octave weights and volume backtracking) measured 0.445 h RMS at strength 0.1 on the
# canonical beam, so the same ranges give the same deformation magnitudes. "strength x beam length" (the first parity
# attempt) gave 10-40 times larger deformations and deep initial contact penetration.
DISPLACEMENT_SCALE = 4.45


class Augmenter:
    def __init__(self, device):
        self.device = torch.device(device)
        self._tables: dict[tuple, tuple[Tensor, Tensor, tuple]] = {}

    # ------------------------------------------------------------------ octaves
    @staticmethod
    def wavelengths(grid: Grid) -> list[int]:
        """2, 4, 8, ... cells up to the longest side."""
        out, w = [], 2
        while w <= max(grid.cell_counts):
            out.append(w)
            w *= 2
        return out

    def _table(self, grid: Grid, w: int) -> tuple[Tensor, Tensor, tuple]:
        """(idx [P,8] flat lattice node ids, wts [P,8] trilinear weights, lattice shape) for wavelength w."""
        key = (grid.key, w)
        if key not in self._tables:
            shape = tuple(n // w + 2 for n in grid.cell_counts)
            u = grid.rest / w
            i0 = u.floor()
            f = u - i0
            i0 = i0.to(torch.int64)
            offs = torch.tensor([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], device=grid.rest.device)
            node = i0[:, None, :] + offs[None, :, :]  # [P,8,3]
            idx = (node[..., 0] * shape[1] + node[..., 1]) * shape[2] + node[..., 2]
            frac_hi = offs.to(f.dtype)[None]  # [1,8,3]
            wts = (frac_hi * f[:, None, :] + (1 - frac_hi) * (1 - f[:, None, :])).prod(-1)  # [P,8]
            self._tables[key] = (idx, wts, shape)
        return self._tables[key]

    def _raw(self, grid: Grid, gens: list) -> Tensor:
        """Unnormalised multiscale field [n,P,3]: sum over octaves of wavelength x interpolated unit lattice noise.

        One generator per state (draw order per generator: the octaves in increasing wavelength); the trilinear
        gather is batched over the states.
        """
        out = torch.zeros(len(gens), grid.P, 3, device=self.device)
        for w in self.wavelengths(grid):
            idx, wts, shape = self._table(grid, w)
            noise = torch.stack(
                [torch.randn(*shape, 3, generator=g, device=self.device) for g in gens]
            )  # [n,Mx,My,Mz,3]
            vals = noise.reshape(len(gens), -1, 3)[:, idx]  # [n,P,8,3]
            out += w * (vals * wts[None, :, :, None]).sum(2)
        return out

    def octave(self, grid: Grid, gen: torch.Generator, wavelength: int) -> Tensor:
        """One octave [P,3]: unit-variance lattice noise at the given wavelength, trilinearly interpolated."""
        idx, wts, shape = self._table(grid, wavelength)
        noise = torch.randn(*shape, 3, generator=gen, device=self.device).reshape(-1, 3)
        return (noise[idx] * wts[:, :, None]).sum(1)

    @staticmethod
    def _rescale(raw: Tensor, rms: Tensor) -> Tensor:
        """Scale each state's field [n,P,3] to the requested RMS (over corners of the vector norm) [n]."""
        cur = raw.double().pow(2).sum(-1).mean(-1).sqrt().clamp_min(1e-300)
        return raw * (rms.double() / cur).to(raw.dtype)[:, None, None]

    # ------------------------------------------------------------------ public
    def field(self, grid: Grid, gen: torch.Generator, rms: float) -> Tensor:
        """Smooth random vector field on the corners [P,3] with the requested RMS (cell units)."""
        rms_t = torch.tensor([float(rms)], device=self.device)
        return self._rescale(self._raw(grid, [gen]), rms_t)[0]

    def initial_state(self, grid: Grid, spec: SceneSpec, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """(X, V) [P,3] in cell units / cells per step; pinned corners at rest with zero velocity."""
        X, V = self.initial_states([grid], [spec], [gen])
        return X[0], V[0]

    def initial_states(self, grids: list, specs: list, gens: list) -> tuple[list, list]:
        """Batched `initial_state` over states; states sharing a grid are interpolated together."""
        groups: dict[tuple, list[int]] = {}
        for i, g in enumerate(grids):
            groups.setdefault(g.key, []).append(i)
        Xs, Vs = [None] * len(grids), [None] * len(grids)
        for members in groups.values():
            grid = grids[members[0]]
            d_rms = torch.tensor(
                [specs[i].perturbation_scale * specs[i].strength * DISPLACEMENT_SCALE for i in members],
                device=self.device,
            )
            v_rms = torch.tensor(
                [specs[i].perturbation_scale * specs[i].velocity_dt for i in members], device=self.device
            )
            member_gens = [gens[i] for i in members]
            X = grid.rest[None] + self._rescale(self._raw(grid, member_gens), d_rms)
            V = self._rescale(self._raw(grid, member_gens), v_rms)
            X[:, grid.pinned] = grid.rest[grid.pinned]
            V[:, grid.pinned] = 0.0
            for k, i in enumerate(members):
                Xs[i], Vs[i] = X[k], V[k]
        return Xs, Vs

    def candidate_noise(self, grid: Grid, gen: torch.Generator, lo: float = 0.01, hi: float = 0.10) -> Tensor:
        """The same field at RMS U(lo, hi) cells (1-10 % of h by default), pinned rows zero."""
        u = torch.rand((), generator=gen, device=self.device)
        rms = (lo + (hi - lo) * u).reshape(1)
        noise = self._rescale(self._raw(grid, [gen]), rms)[0]
        return noise.masked_fill(grid.pinned_mask[:, None], 0.0)
