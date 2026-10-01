# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One solver query (`Step.query` + `Step.commit`) recorded as a CUDA graph and replayed (design spec 1b "CUDA
graphs", section 3 ROLLOUT).

Everything the query reads lives in static buffers: the batch's candidate, energy, gradient and history, the
step-constant tensors (Y, m_Y, m_prev, C_prev, R_ref), the material and the contact pairs in the capacity layout
(`contact.detect(capacity=True)`: S (1 + k) rows with a valid mask, so the shapes are fixed per scene). The
tensors the batch holds at capture time become those buffers; `sync()` copies whatever `prepare` / `advance`
rebound since into them and points the batch back at them, so the eager step code needs no change. Replay
leaves the batch exactly as an eager query + commit would (within float32 noise of the atomics).
"""

from __future__ import annotations

import torch

from .structs import _MATERIAL_TENSORS, Material, Pairs

Tensor = torch.Tensor

STATE = (
    "x",
    "E",
    "gX",
    "hist_grad",
    "hist_update",
    "hist_valid",
    "picard_constant",
    "Y",
    "m_Y",
    "m_prev",
    "C_prev",
    "R_ref",
    "c_n",
    "cdot_n",
)
LAYOUT = (
    "cells",
    "edges",
    "edge_offsets",
    "corner_obj",
    "cell_obj",
    "pinned",
    "mass",
    "flags",
    "edge_rest",
    "sample_cell",
    "sample_face",
    "sample_corners",
    "sample_obj",
    "ref_corners",
)
PAIR_FIELDS = (
    "token_offsets",
    "sample",
    "cell",
    "obj",
    "partner_point",
    "partner_normal",
    "kind",
    "radius",
    "anchor",
    "valid",
    "attn_pairs",
    "attn_offsets",
)
COMMITTED = ("x", "E", "gX", "hist_grad", "hist_update", "hist_valid", "picard_constant")  # what warm-up queries change


class CapturedQuery:
    """Captures `step.query(batch)` + `step.commit(batch, out)` once and replays it.

    Requires a CUDA batch whose pairs are in the capacity layout (`Step(pair_capacity=True)` before `prepare`).
    `out` holds the static QueryOutput of the recorded query; after `replay()` it carries the latest results.
    """

    def __init__(self, step, batch, warmup: int = 3):
        if batch.device.type != "cuda":
            raise ValueError("CUDA-graph capture needs a CUDA batch")
        if not batch.pairs.padded:
            raise ValueError("capture needs capacity-mode pairs: Step(pair_capacity=True) before prepare")
        self.step = step
        self.batch = batch
        self.layout = {name: getattr(batch, name) for name in LAYOUT}
        self.state = {name: getattr(batch, name) for name in STATE}
        self.pairs: Pairs = batch.pairs
        self.material: Material = batch.material
        self.graph = torch.cuda.CUDAGraph()
        self.out = None
        self._capture(warmup)

    def _capture(self, warmup: int) -> None:
        step, batch = self.step, self.batch
        device = batch.device
        saved = {name: getattr(batch, name).clone() for name in COMMITTED}
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(warmup):  # module loads, fusion factors, cuBLAS workspaces, torch.compile
                step.commit(batch, step.query(batch))
            for name, value in saved.items():
                getattr(batch, name).copy_(value)
        torch.cuda.current_stream(device).wait_stream(side)
        self.sync()  # warm-up must not have rebound anything
        with torch.cuda.graph(self.graph, stream=side), torch.no_grad():
            self.out = step.query(batch)
            step.commit(batch, self.out)
        # capture records without executing: the batch state is the one restored above

    def sync(self) -> None:
        """Copy tensors that prepare / advance / a material rebuild rebound into the static buffers."""
        batch = self.batch
        for name, ref in self.layout.items():
            if getattr(batch, name) is not ref:
                raise RuntimeError(f"batch layout changed ({name}): capture the query again")
        for name, buf in self.state.items():
            cur = getattr(batch, name)
            if cur is not buf:
                _copy_into(buf, cur, name)
                setattr(batch, name, buf)
        if batch.pairs is not self.pairs:
            if not batch.pairs.padded:
                raise RuntimeError("pairs are not in the capacity layout: keep Step.pair_capacity on")
            for name in PAIR_FIELDS:
                _copy_into(getattr(self.pairs, name), getattr(batch.pairs, name), f"pairs.{name}")
            batch.pairs = self.pairs
        if batch.material is not self.material:
            for name in _MATERIAL_TENSORS:
                _copy_into(getattr(self.material, name), getattr(batch.material, name), f"material.{name}")
            self.material.si = list(batch.material.si)
            batch.material = self.material

    def replay(self):
        """One query + commit on the current stream; returns the static QueryOutput."""
        self.graph.replay()
        return self.out


def _copy_into(buf: Tensor, src: Tensor, name: str) -> None:
    if buf.shape != src.shape or buf.dtype != src.dtype:
        raise RuntimeError(
            f"{name}: static buffer {tuple(buf.shape)} {buf.dtype} cannot take {tuple(src.shape)} {src.dtype}; "
            "capture the query again"
        )
    buf.copy_(src)
