# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grid topology of one hex block in cell units (design spec 4.2): corners, cells, pins, radius-one edge list
sorted by destination with CSR offsets, exposed faces, fixed-corner flags, face samples, tie-break corners.

Two constructors share the layout: `Grid.build` for a full box of cells and `Grid.from_voxels` for an arbitrary
voxel subset of a lattice with pins anywhere (design spec section 10, "Canonical-cube meshes of arbitrary shape").
Both number cells and corners in lattice order (x slowest, z fastest) and use the same local corner order."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import torch

from . import hex as hx
from .structs import FaceSamples

Tensor = torch.Tensor

# pin patterns of a box grid: one lattice face clamped ("<axis><min|max>_face") or no pins
FACE_PINS = ("xmin_face", "xmax_face", "ymin_face", "ymax_face", "zmin_face", "zmax_face")
BOX_PINS = ("none", *FACE_PINS)


def face_pin(pins: str) -> tuple[int, str]:
    """(axis 0 | 1 | 2, "min" | "max") of a face pin name; ValueError for anything else."""
    if pins not in FACE_PINS:
        raise ValueError(f"unknown pin pattern {pins!r}: expected one of {BOX_PINS}")
    return "xyz".index(pins[0]), pins[1:4]


def face_pin_mask(corner_lattice: Tensor, cell_counts, pins: str) -> Tensor:
    """[P] bool: the corners on the lattice face `pins` (coordinate 0 for a min face, n_axis for a max face)."""
    axis, side = face_pin(pins)
    return corner_lattice[:, axis] == (0 if side == "min" else int(cell_counts[axis]))


def lattice_face_mask(cell_counts, pins: str) -> Tensor:
    """[nx+1, ny+1, nz+1] bool: every lattice corner of the face `pins` (coordinate 0 for a min face, n_axis for a
    max face), the pin mask `Grid.from_voxels` takes for a voxel body clamped at that face (it keeps the present
    corners of the mask; v6 scenes, 2026-10-02)."""
    axis, side = face_pin(pins)
    counts = tuple(int(v) for v in cell_counts)
    mask = torch.zeros(counts[0] + 1, counts[1] + 1, counts[2] + 1, dtype=torch.bool)
    index: list = [slice(None)] * 3
    index[axis] = 0 if side == "min" else counts[axis]
    mask[tuple(index)] = True
    return mask


def corner_index(ix: Tensor, iy: Tensor, iz: Tensor, counts) -> Tensor:
    _nx, ny, nz = counts
    return (ix * (ny + 1) + iy) * (nz + 1) + iz


@dataclass
class Grid:
    key: tuple
    cell_counts: tuple
    rest: Tensor  # [P,3] float32, cell units, corner (ix,iy,iz) at (ix,iy,iz)
    cells: Tensor  # [C,8] int64, local order 000..111 z fastest
    free: Tensor  # [Pf] int64
    pinned: Tensor  # [Pp] int64
    pinned_mask: Tensor  # [P] bool
    edges: Tensor  # [2,E] int64 (src, dst) sorted by dst, self edges included
    edge_offsets: Tensor  # [C+1] int64
    edge_rest: Tensor  # [E,3] float32 rest centre offset src - dst (cell units)
    exposed: Tensor  # [C,6] bool
    fixed_flags: Tensor  # [C,8] bool
    mass: Tensor  # [P] float32 lumped mass in rho h^3 units (cells sharing the corner / 8)
    samples: FaceSamples
    ref_corners: Tensor  # [3] int64 corners defining the reference frame for the tie-break
    device: torch.device
    voxel_index: Tensor = None  # [C,3] int64 lattice coordinates of each cell
    corner_lattice: Tensor = None  # [P,3] int64 lattice coordinates of each corner

    @property
    def P(self) -> int:
        return int(self.rest.shape[0])

    @property
    def C(self) -> int:
        return int(self.cells.shape[0])

    @property
    def E(self) -> int:
        return int(self.edges.shape[1])

    @property
    def S(self) -> int:
        return int(self.samples.cell.shape[0])

    @property
    def Pf(self) -> int:
        return int(self.free.shape[0])

    @property
    def kind(self) -> str:
        """ "box" for `Grid.build` grids (key = (nx, ny, nz, pins)), "voxel" for `Grid.from_voxels` grids."""
        return "voxel" if self.key[0] == "voxel" else "box"

    @property
    def pins(self) -> str:
        """Pin pattern: one of `BOX_PINS` ("none" or a clamped lattice face such as "zmin_face") or, for voxel grids
        with an explicit mask, "mask"."""
        return self.key[2] if self.kind == "voxel" else self.key[3]

    @staticmethod
    def build(cell_counts, pins: str = "zmin_face", device="cpu") -> Grid:
        """Full box of nx x ny x nz cells with `pins` one of `BOX_PINS`: "none" (a free body) or one of the six
        lattice faces clamped ("xmin_face", ..., "zmax_face"; the corners with that coordinate at 0 or n_axis)."""
        nx, ny, nz = (int(v) for v in cell_counts)
        device = torch.device(device)
        ix, iy, iz = torch.meshgrid(torch.arange(nx + 1), torch.arange(ny + 1), torch.arange(nz + 1), indexing="ij")
        rest = torch.stack([ix, iy, iz], -1).reshape(-1, 3)  # z fastest
        P = rest.shape[0]
        cx, cy, cz = torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij")
        cbase = torch.stack([cx, cy, cz], -1).reshape(-1, 3)  # [C,3]
        C = cbase.shape[0]
        corner = cbase[:, None, :] + hx.CORNER_OFFSETS[None, :, :]  # [C,8,3]
        cells = corner_index(corner[..., 0], corner[..., 1], corner[..., 2], (nx, ny, nz))

        if pins == "none":
            pinned_mask = torch.zeros(P, dtype=torch.bool)
        else:
            pinned_mask = face_pin_mask(rest, (nx, ny, nz), pins)
        pinned = pinned_mask.nonzero().flatten()
        free = (~pinned_mask).nonzero().flatten()

        # radius-one directed edges (src -> dst), grouped by dst in increasing order => sorted by dst
        cell_id = torch.full((nx + 2, ny + 2, nz + 2), -1, dtype=torch.int64)
        cell_id[1:-1, 1:-1, 1:-1] = torch.arange(C).reshape(nx, ny, nz)
        offsets = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])  # [27,3]
        nb = cbase[:, None, :] + 1 + offsets[None, :, :]  # [C,27,3] padded indices
        src = cell_id[nb[..., 0], nb[..., 1], nb[..., 2]]  # [C,27], -1 where absent
        dst = torch.arange(C)[:, None].expand(C, 27)
        keep = src >= 0
        edges = torch.stack([src[keep], dst[keep]])  # sorted by dst because rows are dst-major
        counts = keep.sum(1)
        edge_offsets = torch.zeros(C + 1, dtype=torch.int64)
        edge_offsets[1:] = counts.cumsum(0)
        edge_rest = (cbase[edges[0]] - cbase[edges[1]]).to(torch.float32)

        face_nb = cbase[:, None, :] + 1 + (hx.FACE_NORMALS.to(torch.int64))[None, :, :]  # [C,6,3]
        exposed = cell_id[face_nb[..., 0], face_nb[..., 1], face_nb[..., 2]] < 0
        fixed_flags = pinned_mask[cells]
        mass = torch.zeros(P).index_add_(0, cells.reshape(-1), torch.full((C * 8,), 1.0 / 8.0))

        s_cell, s_face = exposed.nonzero(as_tuple=True)
        s_corners = (
            cells[s_cell][:, None, :].expand(-1, 4, 8).gather(2, hx.FACE_CORNERS[s_face][:, :, None]).squeeze(-1)
        )
        # sorted by cell then face already (nonzero is row-major)
        samples = FaceSamples(cell=s_cell.to(device), face=s_face.to(device), corners=s_corners.to(device))

        # tie-break reference corners on the pinned face: p0 = (0,0,0), p1 = farthest pinned corner (nx,ny,0),
        # p2 = the corner maximising |(p1 - p0) x (p2 - p0)| = (0,ny,0) for the z-min face (and, by convention, for
        # a free body); the other five faces take the same rule through `reference_corners`
        if pins in ("zmin_face", "none"):
            ci = lambda a, b, c: corner_index(torch.tensor(a), torch.tensor(b), torch.tensor(c), (nx, ny, nz))  # noqa: E731
            ref = torch.tensor([ci(0, 0, 0), ci(nx, ny, 0), ci(0, ny, 0)])
        else:
            ref = reference_corners(rest, pinned, cells)
        return Grid(
            key=(nx, ny, nz, pins),
            cell_counts=(nx, ny, nz),
            rest=rest.to(torch.float32).to(device),
            cells=cells.to(device),
            free=free.to(device),
            pinned=pinned.to(device),
            pinned_mask=pinned_mask.to(device),
            edges=edges.to(device),
            edge_offsets=edge_offsets.to(device),
            edge_rest=edge_rest.to(device),
            exposed=exposed.to(device),
            fixed_flags=fixed_flags.to(device),
            mass=mass.to(device),
            samples=samples,
            ref_corners=ref.to(device),
            device=device,
            voxel_index=cbase.to(device),
            corner_lattice=rest.to(device),
        )

    @staticmethod
    def from_voxels(occupancy: Tensor, pins="zmin_face", device="cpu") -> Grid:
        """Grid of the occupied voxels of a lattice.

        occupancy [nx, ny, nz] bool: the cells are the occupied voxels (unit cubes) in lattice order; the corners are the
        lattice corners touched by at least one occupied voxel, numbered in lattice order (x slowest, z fastest).
        pins: "zmin_face" (present corners with iz == 0; error if there are none), a bool mask over the lattice corners
        [nx+1, ny+1, nz+1] (restricted to the present corners), or None / "none" for no pins.
        Edges join present cells at Chebyshev distance <= 1 (self edges included), sorted by destination; a face is
        exposed when its neighbour voxel is absent. On full occupancy every field equals `Grid.build`'s.
        """
        occ = occupancy.detach().to("cpu", torch.bool)
        if occ.dim() != 3:
            raise ValueError("occupancy must be a [nx, ny, nz] bool tensor")
        nx, ny, nz = (int(v) for v in occ.shape)
        device = torch.device(device)
        cbase = occ.nonzero()  # [C,3] lattice order
        C = int(cbase.shape[0])
        if C == 0:
            raise ValueError("occupancy has no voxels")
        corner_full = cbase[:, None, :] + hx.CORNER_OFFSETS[None, :, :]  # [C,8,3]
        full_id = corner_index(corner_full[..., 0], corner_full[..., 1], corner_full[..., 2], (nx, ny, nz))
        P_full = (nx + 1) * (ny + 1) * (nz + 1)
        present = torch.zeros(P_full, dtype=torch.bool)
        present[full_id.reshape(-1)] = True
        pos = torch.full((P_full,), -1, dtype=torch.int64)
        pos[present] = torch.arange(int(present.sum()))
        cells = pos[full_id]  # [C,8]
        lx, ly, lz = torch.meshgrid(torch.arange(nx + 1), torch.arange(ny + 1), torch.arange(nz + 1), indexing="ij")
        corner_lattice = torch.stack([lx, ly, lz], -1).reshape(-1, 3)[present]  # [P,3]
        P = int(corner_lattice.shape[0])

        if isinstance(pins, str) and pins == "zmin_face":
            pinned_mask = corner_lattice[:, 2] == 0
            if not bool(pinned_mask.any()):
                raise ValueError("pins='zmin_face' but no present corner lies on the z-min plane")
            mode = "zmin_face"
            mask_bytes = b""
        elif pins is None or (isinstance(pins, str) and pins == "none"):
            pinned_mask = torch.zeros(P, dtype=torch.bool)
            mode = "none"
            mask_bytes = b""
        elif isinstance(pins, Tensor):
            mask = pins.detach().to("cpu", torch.bool)
            if tuple(mask.shape) != (nx + 1, ny + 1, nz + 1):
                raise ValueError(f"pin mask must have the lattice corner shape {(nx + 1, ny + 1, nz + 1)}")
            pinned_mask = mask.reshape(-1)[present]
            mode = "mask"
            mask_bytes = mask.numpy().tobytes()
        else:
            raise ValueError(f"unknown pin pattern {pins!r}")
        pinned = pinned_mask.nonzero().flatten()
        free = (~pinned_mask).nonzero().flatten()

        cell_id = torch.full((nx + 2, ny + 2, nz + 2), -1, dtype=torch.int64)
        cell_id[cbase[:, 0] + 1, cbase[:, 1] + 1, cbase[:, 2] + 1] = torch.arange(C)
        offsets = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])  # [27,3]
        nb = cbase[:, None, :] + 1 + offsets[None, :, :]
        src = cell_id[nb[..., 0], nb[..., 1], nb[..., 2]]  # [C,27], -1 where absent
        dst = torch.arange(C)[:, None].expand(C, 27)
        keep = src >= 0
        edges = torch.stack([src[keep], dst[keep]])  # sorted by dst (rows are dst-major)
        edge_offsets = torch.zeros(C + 1, dtype=torch.int64)
        edge_offsets[1:] = keep.sum(1).cumsum(0)
        edge_rest = (cbase[edges[0]] - cbase[edges[1]]).to(torch.float32)

        face_nb = cbase[:, None, :] + 1 + (hx.FACE_NORMALS.to(torch.int64))[None, :, :]
        exposed = cell_id[face_nb[..., 0], face_nb[..., 1], face_nb[..., 2]] < 0
        fixed_flags = pinned_mask[cells]
        mass = torch.zeros(P).index_add_(0, cells.reshape(-1), torch.full((C * 8,), 1.0 / 8.0))

        s_cell, s_face = exposed.nonzero(as_tuple=True)
        s_corners = (
            cells[s_cell][:, None, :].expand(-1, 4, 8).gather(2, hx.FACE_CORNERS[s_face][:, :, None]).squeeze(-1)
        )
        samples = FaceSamples(cell=s_cell.to(device), face=s_face.to(device), corners=s_corners.to(device))

        ref = reference_corners(corner_lattice, pinned, cells)
        digest = hashlib.sha1(
            bytes(str((nx, ny, nz)), "ascii") + occ.numpy().tobytes() + bytes(mode, "ascii") + mask_bytes
        ).hexdigest()
        return Grid(
            key=("voxel", digest, mode),
            cell_counts=(nx, ny, nz),
            rest=corner_lattice.to(torch.float32).to(device),
            cells=cells.to(device),
            free=free.to(device),
            pinned=pinned.to(device),
            pinned_mask=pinned_mask.to(device),
            edges=edges.to(device),
            edge_offsets=edge_offsets.to(device),
            edge_rest=edge_rest.to(device),
            exposed=exposed.to(device),
            fixed_flags=fixed_flags.to(device),
            mass=mass.to(device),
            samples=samples,
            ref_corners=ref.to(device),
            device=device,
            voxel_index=cbase.to(device),
            corner_lattice=corner_lattice.to(device),
        )


def reference_corners(corner_lattice: Tensor, pinned: Tensor, cells: Tensor) -> Tensor:
    """Tie-break reference corners [3] from the pinned set (the `Grid.build` rule generalised).

    p0 = the lexicographically smallest pinned corner, p1 = the pinned corner farthest from p0, p2 = the pinned corner
    maximising |(p1 - p0) x (p2 - p0)| (first in lattice order on ties). With fewer than three non-collinear pinned
    corners the three corners (0,0,0), (1,0,0), (0,1,0) of the first cell are used instead; this is noted with a
    warning when any corner is pinned (with no pins at all it is the only choice).
    """
    lat = corner_lattice.to(torch.float64)
    if pinned.numel() >= 3:
        pts = lat[pinned]
        p0 = pinned[0]  # pinned is sorted, corners are numbered in lattice order
        d = pts - lat[p0]
        p1 = pinned[d.norm(dim=1).argmax()]
        cross = torch.linalg.cross((lat[p1] - lat[p0])[None, :].expand_as(d), d).norm(dim=1)
        if cross.max() > 1e-9:
            return torch.stack([p0, p1, pinned[cross.argmax()]])
    if pinned.numel() > 0:
        warnings.warn(
            "Grid.from_voxels: fewer than three non-collinear pinned corners; the reference frame uses three corners "
            "of the first cell",
            stacklevel=3,
        )
    return cells[0, [0, 4, 2]].clone()


class GridCache:
    """Grids by shape and pins on one device: `get` the box grids (key (nx, ny, nz, pins)), `get_voxel` the grids of
    the voxel bodies of the v6 scenes (key (shape_id, pins))."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.grids: dict[tuple, Grid] = {}

    def get(self, cell_counts, pins: str = "zmin_face") -> Grid:
        key = (*(int(v) for v in cell_counts), pins)
        if key not in self.grids:
            self.grids[key] = Grid.build(cell_counts, pins, self.device)
        return self.grids[key]

    def get_voxel(self, shape_id: int, occupancy, pins: str = "none") -> Grid:
        """The grid of a voxel body (v6 scenes, 2026-10-02): `Grid.from_voxels` of the bool occupancy [nx, ny, nz]
        (numpy or tensor) with `pins` "none" or one of `FACE_PINS`, handed to the constructor as the corner mask of
        that lattice face (`lattice_face_mask`: the present corners with that coordinate at 0 or n are clamped).
        Keyed by (shape_id, pins): the ids of a run's shape library (`shapes.ShapeLibrary`) are stable, so a cache
        serves one library; a hit whose lattice or cell count disagrees with the occupancy raises."""
        key = (int(shape_id), str(pins))
        occ = torch.as_tensor(occupancy).to("cpu", torch.bool)
        if key not in self.grids:
            mask = None if pins == "none" else lattice_face_mask(occ.shape, pins)
            self.grids[key] = Grid.from_voxels(occ, mask, self.device)
        grid = self.grids[key]
        if grid.cell_counts != tuple(int(v) for v in occ.shape) or grid.C != int(occ.sum()):
            raise ValueError(f"GridCache: shape {shape_id} with pins {pins!r} was cached from a different occupancy")
        return grid


def reference_rotation(x: Tensor, ref_corners: Tensor) -> Tensor:
    """Reference frame for the frame tie-break from the current positions of the three reference corners.

    x [P,3] or [n,P,3] -> R_ref [3,3] or [n,3,3]: e1 = normalize(x1 - x0), n = normalize(e1 x (x2 - x0)),
    e2 = n x e1, columns [e1 e2 n]. Rotates with the body.
    """
    p = x[..., ref_corners, :]
    e1 = p[..., 1, :] - p[..., 0, :]
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n = torch.linalg.cross(e1, p[..., 2, :] - p[..., 0, :])
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    e2 = torch.linalg.cross(n, e1)
    return torch.stack([e1, e2, n], -1)
