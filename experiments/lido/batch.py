# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The flat batch: every object served this update, concatenated and sorted (design spec 4.2, decision "Batch layout").

Slot i (object i) owns fixed row ranges of every persistent buffer. Index tensors are rebuilt only when
an object's grid changes; everything else is written in place by prepare / advance / load / commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .hex import MODE_COUNT, HexConstants
from .structs import ContactScene, Group, Material, Pairs

Tensor = torch.Tensor


def _offsets(counts: list[int], device) -> Tensor:
    out = torch.zeros(len(counts) + 1, dtype=torch.int64, device=device)
    if counts:
        out[1:] = torch.tensor(counts, dtype=torch.int64, device=device).cumsum(0)
    return out


def _empty_pairs(C: int, device) -> Pairs:
    z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
    return Pairs(
        token_offsets=z(C + 1, dtype=torch.int64),
        sample=z(0, dtype=torch.int64),
        cell=z(0, dtype=torch.int64),
        obj=z(0, dtype=torch.int64),
        partner_point=z(0, 3),
        partner_normal=z(0, 3),
        kind=z(0, dtype=torch.int64),
        radius=z(0),
        anchor=z(0, 3),
        valid=z(0, dtype=torch.bool),
        partner_body=z(0, dtype=torch.int64),
        partner_face=z(0, dtype=torch.int64),
    )


def empty_scene(O: int, device) -> ContactScene:
    return ContactScene(
        plane_n=torch.zeros(O, 3, device=device),
        plane_d=torch.zeros(O, device=device),
        plane_present=torch.zeros(O, dtype=torch.bool, device=device),
        points=torch.zeros(0, 3, device=device),
        normals=torch.zeros(0, 3, device=device),
        radii=torch.zeros(0, device=device),
        point_offsets=torch.zeros(O + 1, dtype=torch.int64, device=device),
    )


@dataclass
class Batch:
    grids: list
    hc: HexConstants
    device: torch.device
    # layout (rebuilt when a grid changes)
    corner_off: Tensor = None  # [O+1]
    cell_off: Tensor = None  # [O+1]
    edge_off: Tensor = None  # [O+1]
    sample_off: Tensor = None  # [O+1]
    cells: Tensor = None  # [C,8] global corner ids
    edges: Tensor = None  # [2,E] global cell ids, sorted by dst
    edge_offsets: Tensor = None  # [C+1]
    corner_obj: Tensor = None  # [N]
    cell_obj: Tensor = None  # [C]
    pinned: Tensor = None  # [N] bool
    mass: Tensor = None  # [N]
    flags: Tensor = None  # [C,14]
    edge_rest: Tensor = None  # [E,3]
    rest: Tensor = None  # [N,3] rest positions (cell units)
    sample_cell: Tensor = None  # [S] global cell
    sample_face: Tensor = None  # [S]
    sample_corners: Tensor = None  # [S,4] global corner ids
    sample_obj: Tensor = None  # [S]
    ref_corners: Tensor = None  # [O,3] global corner ids
    free_objects: Tensor = None  # [O] bool: objects without pinned corners (free bodies, derivation section 7)
    any_free: bool = False
    groups: list = field(default_factory=list)
    # per object
    material: Material = None
    scene: ContactScene = None
    origin: Tensor = None  # [O,3] SI origin of the object's cell coordinates
    # state
    X: Tensor = None
    V: Tensor = None
    X_prev: Tensor = None
    Y: Tensor = None
    x: Tensor = None  # candidate
    E: Tensor = None  # [O]
    gX: Tensor = None  # [N,3]
    hist_grad: Tensor = None  # [C,7,3]
    hist_update: Tensor = None
    hist_valid: Tensor = None  # [O] bool
    active: Tensor = None  # [O] bool
    picard_constant: Tensor = None  # [O] sum_active ke / M_tot at the last query's candidate (free objects, else 0)
    # step-constant
    m_Y: Tensor = None
    m_prev: Tensor = None
    C_prev: Tensor = None  # [C,8,3,3]
    R_ref: Tensor = None  # [O,3,3]
    c_n: Tensor = None  # [O,3] mass-weighted centroid of X (eq. 1.10)
    cdot_n: Tensor = None  # [O,3] centroid velocity c(V)
    pairs: Pairs = None
    pair_layout: tuple = None  # capacity layout cache of contact.detect(capacity=True), reset by relayout
    # body-body contact (v5, design spec section 11): every body is a partner of every other body in one world frame
    body_contact: bool = False  # contact.detect also queries the other bodies' surface meshes
    meshes: object = None  # contact.BodyMeshes: one wp.Mesh per object over its exposed faces, rebuilt per step
    static_mesh: object = None  # contact.StaticMesh: one wp.Mesh over the scene's static faces, built once per scene
    fusion_cache: dict = (
        None  # fusion.BatchedKron | BatchedSparse | False per dtype (the multi-group solve), reset by relayout
    )
    noise_cache: object = None  # augment.Augmenter's padded-lattice tables of the whole batch, reset by relayout
    # job-constant network cache
    film: tuple = None
    # sizes, cached at layout time (reading them must not synchronise: the query is CUDA-graph captured)
    _N: int = 0
    _C: int = 0
    _S: int = 0

    @property
    def O(self) -> int:  # noqa: E743 (object count, used package-wide)
        return len(self.grids)

    @property
    def N(self) -> int:
        return self._N

    @property
    def C(self) -> int:
        return self._C

    @property
    def S(self) -> int:
        return self._S

    @staticmethod
    def build(grids: list, device, dtype=torch.float32) -> Batch:
        device = torch.device(device)
        b = Batch(grids=list(grids), hc=HexConstants.get(device, dtype), device=device)
        b._layout()
        O, N, C = b.O, b.N, b.C
        z = lambda *shape, dt=dtype: torch.zeros(*shape, dtype=dt, device=device)  # noqa: E731
        b.X, b.V, b.X_prev, b.Y, b.x, b.gX = (z(N, 3) for _ in range(6))
        b.E = z(O)
        b.hist_grad, b.hist_update = z(C, MODE_COUNT, 3), z(C, MODE_COUNT, 3)
        b.hist_valid = z(O, dt=torch.bool)
        b.active = torch.ones(O, dtype=torch.bool, device=device)
        b.m_Y, b.m_prev = z(C, MODE_COUNT, 3), z(C, MODE_COUNT, 3)
        b.C_prev = z(C, 8, 3, 3)
        b.R_ref = torch.eye(3, device=device, dtype=dtype).expand(O, 3, 3).clone()
        b.c_n, b.cdot_n = z(O, 3), z(O, 3)
        b.picard_constant = z(O)
        b.pairs = _empty_pairs(C, device)
        b.scene = empty_scene(O, device)
        b.origin = z(O, 3)
        b.material = None
        return b

    def _layout(self) -> None:
        dev = self.device
        grids = self.grids
        self.corner_off = _offsets([g.P for g in grids], dev)
        self.cell_off = _offsets([g.C for g in grids], dev)
        self.edge_off = _offsets([g.E for g in grids], dev)
        self.sample_off = _offsets([g.S for g in grids], dev)
        self._N, self._C, self._S = sum(g.P for g in grids), sum(g.C for g in grids), sum(g.S for g in grids)
        cells, edges, edge_counts, s_cell, s_corners = [], [], [], [], []
        for o, g in enumerate(grids):
            cells.append(g.cells + self.corner_off[o])
            edges.append(g.edges + self.cell_off[o])
            edge_counts.append(g.edge_offsets[1:] - g.edge_offsets[:-1])
            s_cell.append(g.samples.cell + self.cell_off[o])
            s_corners.append(g.samples.corners + self.corner_off[o])
        cat = lambda parts, dim=0: torch.cat(parts, dim) if parts else torch.zeros(0, device=dev)  # noqa: E731
        self.cells = cat(cells)
        self.edges = cat(edges, 1)
        counts = cat(edge_counts)
        self.edge_offsets = torch.zeros(self.C + 1, dtype=torch.int64, device=dev)
        self.edge_offsets[1:] = counts.cumsum(0)
        self.corner_obj = torch.repeat_interleave(
            torch.arange(self.O, device=dev), self.corner_off[1:] - self.corner_off[:-1]
        )
        self.cell_obj = torch.repeat_interleave(
            torch.arange(self.O, device=dev), self.cell_off[1:] - self.cell_off[:-1]
        )
        self.pinned = cat([g.pinned_mask for g in grids])
        self.mass = cat([g.mass for g in grids])
        self.rest = cat([g.rest for g in grids])
        self.flags = cat([torch.cat([g.exposed, g.fixed_flags], 1).to(self.hc.Gq.dtype) for g in grids])
        self.edge_rest = cat([g.edge_rest for g in grids])
        self.sample_cell = cat(s_cell)
        self.sample_face = cat([g.samples.face for g in grids])
        self.sample_corners = cat(s_corners)
        self.sample_obj = torch.repeat_interleave(
            torch.arange(self.O, device=dev), self.sample_off[1:] - self.sample_off[:-1]
        )
        self.ref_corners = torch.stack([g.ref_corners + self.corner_off[o] for o, g in enumerate(grids)])
        self.free_objects = torch.tensor([g.pinned.numel() == 0 for g in grids], dtype=torch.bool, device=dev)
        self.any_free = bool(self.free_objects.any())
        by_key: dict[tuple, list[int]] = {}
        for o, g in enumerate(grids):
            by_key.setdefault(g.key, []).append(o)
        self.groups = []
        for _key, objs in by_key.items():
            g = grids[objs[0]]
            corner_idx = torch.cat([torch.arange(self.corner_off[o], self.corner_off[o + 1], device=dev) for o in objs])
            cell_idx = torch.cat([torch.arange(self.cell_off[o], self.cell_off[o + 1], device=dev) for o in objs])
            self.groups.append(Group(g, torch.tensor(objs, device=dev), corner_idx, cell_idx))

    def corner_rows(self, objects: Tensor) -> Tensor:
        """Boolean mask [N] of the corner rows owned by the given objects (bool mask [O])."""
        return objects[self.corner_obj]

    def cell_rows(self, objects: Tensor) -> Tensor:
        return objects[self.cell_obj]

    def gather_cells(self, x: Tensor) -> Tensor:
        return x[self.cells]  # [C,8,3]

    def relayout(self, new_grids: list) -> None:
        """Change some objects' grids, keeping the state rows of unchanged objects."""
        old = self
        keep = [o for o in range(self.O) if new_grids[o].key == old.grids[o].key]
        state = {}
        for name in ("X", "V", "X_prev", "Y", "x", "gX"):
            state[name] = [getattr(old, name)[old.corner_off[o] : old.corner_off[o + 1]] for o in range(old.O)]
        for name in ("hist_grad", "hist_update", "m_Y", "m_prev", "C_prev"):
            state[name] = [getattr(old, name)[old.cell_off[o] : old.cell_off[o + 1]] for o in range(old.O)]
        self.grids = list(new_grids)
        self._layout()
        dtype = self.hc.Gq.dtype
        for name, parts in state.items():
            shape = parts[0].shape[1:]
            new = torch.zeros(
                self.N if name in ("X", "V", "X_prev", "Y", "x", "gX") else self.C,
                *shape,
                dtype=dtype,
                device=self.device,
            )
            off = self.corner_off if name in ("X", "V", "X_prev", "Y", "x", "gX") else self.cell_off
            for o in keep:
                new[off[o] : off[o + 1]] = parts[o]
            setattr(self, name, new)
        self.pairs = _empty_pairs(self.C, self.device)
        self.pair_layout = None
        self.meshes = None
        self.static_mesh = None
        self.fusion_cache = None
        self.noise_cache = None
        self.film = None
