# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Per-frame stretch capture from a Newton tet model+state."""

from __future__ import annotations

import warp as wp

from .kernels import compute_F_polar


class StretchRecorder:
    def __init__(self, model):
        if model.tet_count == 0:
            raise ValueError("StretchRecorder requires a tet mesh in the model.")
        self.tet_count = model.tet_count
        self.model = model
        device = model.device
        self.F = wp.zeros(self.tet_count, dtype=wp.mat33, device=device)
        self.R = wp.zeros(self.tet_count, dtype=wp.mat33, device=device)
        self.S = wp.zeros(self.tet_count, dtype=wp.mat33, device=device)

    def capture(self, state) -> dict:
        wp.launch(
            compute_F_polar,
            dim=self.tet_count,
            inputs=[
                state.particle_q,
                self.model.tet_indices,
                self.model.tet_poses,
            ],
            outputs=[self.F, self.R, self.S],
            device=self.model.device,
        )
        return {
            "x": state.particle_q.numpy().copy(),
            "F": self.F.numpy().copy(),
            "R": self.R.numpy().copy(),
            "S": self.S.numpy().copy(),
        }
