# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Data structures shared by the solver, network, data generator and trainer (design spec sections 4.2, 5.2, 6.2).

Every batch is flat and sorted: objects are contiguous, edges are sorted by destination cell, contact
pairs are sorted by owning cell, so every grouping is a CSR offsets vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class Job:
    seed: int
    K: int
    H: int


@dataclass
class FaceSamples:
    """Exposed face centres of a grid (contact samples, radius 0.5 h)."""

    cell: Tensor  # [S] local cell id
    face: Tensor  # [S] face id 0..5
    corners: Tensor  # [S,4] local corner ids, counter-clockwise around the outward normal


@dataclass
class Material:
    """Per-object material in normalised units (h = mu = dt = 1); fields stacked over objects [O]."""

    lam: Tensor  # lambda / mu
    rho: Tensor  # rho h^2 / (mu dt^2)
    eta: Tensor  # eta / (mu dt)
    g: Tensor  # [O,3] g dt^2 / h
    ke: Tensor  # contact stiffness ke / (mu h)
    kd: Tensor  # contact damping kd / (mu h dt)
    mu_f: Tensor  # friction coefficient
    kappa: Tensor  # ke / (E h)
    beta: Tensor  # kd / (ke dt)
    friction_eps: Tensor  # friction_epsilon dt / h (velocity band in cell units per step)
    floor: Tensor  # energy floor in mu h^3
    h: Tensor  # cell size [m]
    dt: Tensor  # time step [s]
    mu: Tensor  # shear modulus [Pa]
    si: list  # per-object dicts of the SI parameters (records)

    @property
    def count(self) -> int:
        return int(self.lam.shape[0])

    def __getitem__(self, idx) -> Material:
        idx_list = idx if isinstance(idx, list) else torch.as_tensor(idx).reshape(-1).tolist()
        t = torch.as_tensor(idx_list, device=self.lam.device, dtype=torch.long)
        return Material(
            *(getattr(self, f)[t] for f in _MATERIAL_TENSORS),
            si=[self.si[i] for i in idx_list],
        )

    @staticmethod
    def cat(parts: list[Material]) -> Material:
        return Material(
            *(torch.cat([getattr(p, f) for p in parts]) for f in _MATERIAL_TENSORS),
            si=[d for p in parts for d in p.si],
        )

    def write(self, idx: Tensor, src: Material) -> None:
        for f in _MATERIAL_TENSORS:
            getattr(self, f)[idx] = getattr(src, f)
        for i, d in zip(idx.tolist(), src.si, strict=True):
            self.si[i] = d


_MATERIAL_TENSORS = (
    "lam",
    "rho",
    "eta",
    "g",
    "ke",
    "kd",
    "mu_f",
    "kappa",
    "beta",
    "friction_eps",
    "floor",
    "h",
    "dt",
    "mu",
)


@dataclass
class ContactScene:
    """Static contact partners of every object, in each object's normalised coordinates; concatenated over objects."""

    plane_n: Tensor  # [O,3] unit normal (pointing out of the obstacle, towards the body)
    plane_d: Tensor  # [O]   plane offset: signed distance of x to the plane is n . x - d
    plane_present: Tensor  # [O] bool
    points: Tensor  # [Npts,3]
    normals: Tensor  # [Npts,3]
    radii: Tensor  # [Npts]
    point_offsets: Tensor  # [O+1] CSR by object


@dataclass
class Pairs:
    """Flat list of detected contact pairs, frozen per physical step, sorted by owning cell, face sample, partner."""

    token_offsets: Tensor  # [C+1] CSR by owning cell
    sample: Tensor  # [Q] global sample id
    cell: Tensor  # [Q] global owning cell id
    obj: Tensor  # [Q] object id
    partner_point: Tensor  # [Q,3]
    partner_normal: Tensor  # [Q,3]
    kind: Tensor  # [Q] 0 plane, 1 static point
    radius: Tensor  # [Q] partner radius
    anchor: Tensor  # [Q,3] sample position at step start (friction anchor)
    valid: Tensor  # [Q] bool
    # capacity layout (design spec 1b "CUDA graphs"): Q = S (1 + k) slots in a fixed sample-major order, padded
    # rows carry valid = False; token_offsets is then the static capacity CSR and the token attention pairs are
    # precomputed once per layout
    padded: bool = False
    attn_pairs: Tensor | None = None  # [2,P] within-cell token pairs (src, dst) sorted by dst
    attn_offsets: Tensor | None = None  # [Q+1]

    @property
    def count(self) -> int:
        return int(self.sample.shape[0])


@dataclass
class Group:
    """Objects of a batch that share one grid: solved together by the fusion."""

    grid: Grid
    objects: Tensor  # [n] object ids
    corner_idx: Tensor  # [n*P] global corner ids, object-major
    cell_idx: Tensor  # [n*C] global cell ids, object-major


@dataclass
class QueryOutput:
    E_before: Tensor  # [O]
    E_after: Tensor  # [O]
    gX_after: Tensor  # [N,3] detached
    cand_after: Tensor  # [N,3] attached to the graph
    dm_world: Tensor  # [C,7,3] achieved mode update in world axes (detached)
    g_world: Tensor  # [C,7,3] projected gradient in world axes (detached)
    residual: Tensor  # [O] free-corner gradient norm at the query's input candidate
    inverted: Tensor  # [O] number of cells with det F_centre <= 0 at the input candidate
    step: Tensor = None  # [C] per-cell step size (detached)
    picard_constant: Tensor = None  # [O] sum_active ke / M_tot at the input candidate (unpinned objects; eq. 7.13)


@dataclass
class Features:
    node: Tensor  # [C,142] everything but the 17 contact channels (appended by the encoder)
    edge_attr: Tensor  # [E,24]
    cond: Tensor  # [O,7]
    tokens: Tensor  # [Q,19]
    token_offsets: Tensor  # [C+1]
    schema: int = 6
    # capacity-mode tokens (padded rows present): validity, owning cell and the static attention pair list
    token_valid: Tensor | None = None  # [Q] bool
    token_cell: Tensor | None = None  # [Q]
    token_attn: tuple | None = None  # (pairs [2,P], pair_offsets [Q+1])
    edge_sender: Tensor | None = None  # [E,21] sender modes in the receiver's frame, R_i^T m_j ("pair" edge module)


@dataclass
class NetOutput:
    corr: Tensor  # [C,7,3]
    step: Tensor  # [C]


@dataclass
class FailureRecord:
    seed: int
    K: int
    H: int
    k: int
    h: int
    kind: str
    epoch: int
    update: int


@dataclass
class SceneSpec:
    """Everything drawn from one state's seed; logged with the run. SI units."""

    seed: int
    cell_counts: tuple
    h: float
    dt: float
    pins: str
    E: float
    nu: float
    rho: float
    eta: float
    gravity: tuple
    perturbation_scale: float
    strength: float
    velocity_dt: float
    contact: dict = field(
        default_factory=dict
    )  # plane_present, plane_height, points [n,3], normals, radii, kappa, beta, mu_f
