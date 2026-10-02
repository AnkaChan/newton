# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Arbitrary voxel shapes by coarse-to-fine sampling (v6 scenes, Anka 2026-10-02).

A shape is a face-connected set of unit voxels on a lattice, built in levels: a coarse lattice of 2-4 voxels per
axis is seeded with one voxel and grown by adding random face neighbours until a target fill is reached (level 0),
then refined once or twice, each refinement subdividing every voxel into 2 x 2 x 2 and flipping boundary voxels at
random (surface voxels removed, empty voxels touching the surface added, with probabilities that shrink with the
level), keeping the largest face-connected component. The result ranges from bars, L and T shapes and slabs to
blobs with notches and bumps; `Grid.from_voxels` turns the occupancy into a solver grid.

`ShapeLibrary` (v6 scenes, Anka 2026-10-02) is the fixed list of shapes a training run draws its bodies from: `size`
shapes sampled once from one seed with cell counts in `cell_range`, the index of a shape being its stable id across
scenes, epochs and ranks; `shape_library` builds a library at most once per process (a module-level cache keyed by
(size, seed, cell_range); the default of 2048 shapes takes about 13 s on one CPU core).
"""

from __future__ import annotations

from collections import deque

import numpy as np

FACE_NEIGHBOURS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])


def largest_component(occ: np.ndarray) -> np.ndarray:
    """The largest face-connected component of a bool occupancy [nx, ny, nz] (empty input stays empty)."""
    occ = np.asarray(occ, dtype=bool)
    seen = np.zeros_like(occ)
    best, best_size = np.zeros_like(occ), 0
    for start in zip(*np.nonzero(occ & ~seen), strict=True):
        if seen[start]:
            continue
        comp = np.zeros_like(occ)
        queue = deque([start])
        seen[start] = comp[start] = True
        size = 1
        while queue:
            p = queue.popleft()
            for d in FACE_NEIGHBOURS:
                q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                if all(0 <= q[i] < occ.shape[i] for i in range(3)) and occ[q] and not seen[q]:
                    seen[q] = comp[q] = True
                    size += 1
                    queue.append(q)
        if size > best_size:
            best, best_size = comp, size
    return best


def surface_mask(occ: np.ndarray) -> np.ndarray:
    """Occupied voxels with at least one empty (or outside) face neighbour."""
    occ = np.asarray(occ, dtype=bool)
    padded = np.pad(occ, 1)
    full = np.ones_like(occ)
    for d in FACE_NEIGHBOURS:
        shifted = padded[
            1 + d[0] : 1 + d[0] + occ.shape[0], 1 + d[1] : 1 + d[1] + occ.shape[1], 1 + d[2] : 1 + d[2] + occ.shape[2]
        ]
        full &= shifted
    return occ & ~full


def shell_mask(occ: np.ndarray) -> np.ndarray:
    """Empty voxels with at least one occupied face neighbour."""
    occ = np.asarray(occ, dtype=bool)
    padded = np.pad(occ, 1)
    touch = np.zeros_like(occ)
    for d in FACE_NEIGHBOURS:
        shifted = padded[
            1 + d[0] : 1 + d[0] + occ.shape[0], 1 + d[1] : 1 + d[1] + occ.shape[1], 1 + d[2] : 1 + d[2] + occ.shape[2]
        ]
        touch |= shifted
    return touch & ~occ


def grow_coarse(rng: np.random.Generator, shape: tuple, count: int) -> np.ndarray:
    """A face-connected set of `count` voxels on a lattice of `shape`, grown from a random seed voxel by adding
    random face neighbours of the current set (uniform over the candidate neighbours)."""
    occ = np.zeros(shape, dtype=bool)
    seed = tuple(int(rng.integers(0, s)) for s in shape)
    occ[seed] = True
    while occ.sum() < min(count, occ.size):
        candidates = np.argwhere(shell_mask(occ))
        pick = candidates[rng.integers(0, len(candidates))]
        occ[tuple(pick)] = True
    return occ


def neighbour_count(occ: np.ndarray) -> np.ndarray:
    """Occupied face neighbours of every voxel [nx, ny, nz] (0-6)."""
    occ = np.asarray(occ, dtype=bool)
    padded = np.pad(occ, 1).astype(np.int8)
    count = np.zeros(occ.shape, dtype=np.int8)
    for d in FACE_NEIGHBOURS:
        count += padded[
            1 + d[0] : 1 + d[0] + occ.shape[0], 1 + d[1] : 1 + d[1] + occ.shape[1], 1 + d[2] : 1 + d[2] + occ.shape[2]
        ]
    return count


def smooth(occ: np.ndarray, prune_below: int = 2, fill_above: int = 4, rounds: int = 4) -> np.ndarray:
    """Remove spikes (occupied voxels with fewer than `prune_below` occupied face neighbours) and fill pits (empty
    voxels with more than `fill_above` occupied face neighbours), a few rounds."""
    occ = np.asarray(occ, dtype=bool).copy()
    for _ in range(rounds):
        n = neighbour_count(occ)
        changed = (occ & (n < prune_below)) | (~occ & (n > fill_above))
        if not changed.any():
            break
        occ = occ ^ changed
    return occ


def refine(rng: np.random.Generator, occ: np.ndarray, p_remove: float, p_add: float, block: int = 1) -> np.ndarray:
    """Subdivide every voxel into 2 x 2 x 2, flip boundary voxels at random in blocks of `block` fine voxels per
    axis (the same draw for a whole block, so the surface detail has the block's scale), smooth, and keep the
    largest face-connected component."""
    fine = np.repeat(np.repeat(np.repeat(occ, 2, 0), 2, 1), 2, 2)
    fine = np.pad(fine, block)  # room for added voxels on the outside
    draws = rng.random(tuple(-(-n // block) for n in fine.shape))
    draws = np.repeat(np.repeat(np.repeat(draws, block, 0), block, 1), block, 2)[
        : fine.shape[0], : fine.shape[1], : fine.shape[2]
    ]
    remove = surface_mask(fine) & (draws < p_remove)
    add = shell_mask(fine) & (draws > 1.0 - p_add)
    fine = (fine & ~remove) | add
    fine = smooth(fine)
    fine = largest_component(fine)
    return trim(fine)


def trim(occ: np.ndarray) -> np.ndarray:
    """The occupancy cropped to its bounding box (an empty occupancy comes back with shape (0, 0, 0))."""
    idx = np.argwhere(occ)
    if idx.shape[0] == 0:
        return occ[:0, :0, :0]
    lo, hi = idx.min(0), idx.max(0) + 1
    return occ[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]


def sample_voxel_shape(
    rng: np.random.Generator,
    coarse_sides=(1, 4),
    levels=(1, 2),
    fill=(0.4, 0.9),
    flips=((0.25, 0.15), (0.10, 0.06)),
    cell_range=(27, 1728),
    max_tries: int = 50,
) -> np.ndarray:
    """One shape: bool occupancy [nx, ny, nz] with a cell count inside `cell_range` (resampled otherwise).

    coarse_sides: U{lo..hi} coarse voxels per axis, independently (at most one axis of 1: slabs and bars, never
    a line); levels: U{lo..hi} refinements (each x2); fill: U(lo, hi) fraction of the coarse lattice occupied;
    flips[level] = (p_remove, p_add) of the boundary flips at that refinement level, drawn per block of 2 fine
    voxels at the first level and per voxel afterwards (the last entry repeats for deeper levels).
    """
    for _ in range(max_tries):
        shape = tuple(int(rng.integers(coarse_sides[0], coarse_sides[1] + 1)) for _ in range(3))
        if sum(1 for n in shape if n == 1) > 1:
            continue
        volume = int(np.prod(shape))
        count = max(2, int(round(float(rng.uniform(*fill)) * volume)))
        occ = grow_coarse(rng, shape, count)
        n_levels = int(rng.integers(levels[0], levels[1] + 1))
        for level in range(n_levels):
            p_remove, p_add = flips[min(level, len(flips) - 1)]
            occ = refine(rng, occ, p_remove, p_add, block=2 if level == 0 else 1)
            if not occ.any():  # the smoothing can prune a thin shape away entirely (seed 2026 of the default
                break  # library hit it, 2026-10-02): a failed try, drawn again
        cells = int(occ.sum())
        if cell_range[0] <= cells <= cell_range[1]:
            return occ
    raise RuntimeError(f"no shape within {cell_range} cells after {max_tries} tries")


def exposed_faces(occ: np.ndarray) -> list:
    """Quads [4,3] (lattice units) of the exposed voxel faces, for drawing."""
    occ = np.asarray(occ, dtype=bool)
    padded = np.pad(occ, 1)
    quads = []
    corners = {
        0: [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
        1: [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
        2: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        3: [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
        4: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        5: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    }
    for f, d in enumerate(FACE_NEIGHBOURS):
        neighbour = padded[
            1 + d[0] : 1 + d[0] + occ.shape[0], 1 + d[1] : 1 + d[1] + occ.shape[1], 1 + d[2] : 1 + d[2] + occ.shape[2]
        ]
        for v in np.argwhere(occ & ~neighbour):
            quads.append(v[None, :] + np.asarray(corners[f]))
    return quads


def face_counts(occ: np.ndarray) -> np.ndarray:
    """[6] int: exposed voxel faces per direction (the order of `FACE_NEIGHBOURS`: -x, +x, -y, +y, -z, +z), the
    occupied voxels whose neighbour in that direction is empty or outside. On a full box the direction's face area;
    the largest entry is the load floor's n_face of a voxel body (`scenes_v5`)."""
    occ = np.asarray(occ, dtype=bool)
    padded = np.pad(occ, 1)
    counts = np.zeros(len(FACE_NEIGHBOURS), dtype=np.int64)
    for f, d in enumerate(FACE_NEIGHBOURS):
        neighbour = padded[
            1 + d[0] : 1 + d[0] + occ.shape[0], 1 + d[1] : 1 + d[1] + occ.shape[1], 1 + d[2] : 1 + d[2] + occ.shape[2]
        ]
        counts[f] = int((occ & ~neighbour).sum())
    return counts


class ShapeLibrary:
    """The fixed shapes of a run (module docstring): `shapes[i]` is a bool occupancy [nx, ny, nz] with id i, drawn in
    order from `np.random.default_rng(seed)` by `sample_voxel_shape(cell_range=cell_range)`, so equal (size, seed,
    cell_range) give equal libraries on every process; `cells[i]` is the occupied count."""

    def __init__(self, size: int, seed: int, cell_range=(27, 1728)):
        self.size = int(size)
        self.seed = int(seed)
        self.cell_range = (int(cell_range[0]), int(cell_range[1]))
        if self.size <= 0:
            raise ValueError(f"shape library size must be positive, got {size}")
        rng = np.random.default_rng(self.seed)
        self.shapes = [sample_voxel_shape(rng, cell_range=self.cell_range) for _ in range(self.size)]
        self.cells = np.array([int(occ.sum()) for occ in self.shapes], dtype=np.int64)

    @property
    def key(self) -> tuple:
        """(size, seed, cell_range): what names the library (`SceneV5.shape_library` records the same three)."""
        return (self.size, self.seed, self.cell_range)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, shape_id: int) -> np.ndarray:
        return self.shapes[int(shape_id)]


_LIBRARIES: dict[tuple, ShapeLibrary] = {}


def shape_library(size: int, seed: int, cell_range=(27, 1728)) -> ShapeLibrary:
    """The process's library for (size, seed, cell_range), built on the first call and cached."""
    key = (int(size), int(seed), (int(cell_range[0]), int(cell_range[1])))
    if key not in _LIBRARIES:
        _LIBRARIES[key] = ShapeLibrary(*key)
    return _LIBRARIES[key]
