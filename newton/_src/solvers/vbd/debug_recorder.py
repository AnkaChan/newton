# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Debug recorder for VBD solver instrumentation.

Pre-allocates GPU buffers and uses Warp kernels to snapshot solver internals
at each substep and iteration. Design is graph-capture-compatible (all buffers
pre-allocated, no CPU-GPU sync during recording).
"""

from __future__ import annotations

import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
# Snapshot kernels (GPU → GPU copies at a given buffer offset)
# ---------------------------------------------------------------------------


@wp.kernel
def _snapshot_vec3(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
    offset: int,
    count: int,
):
    """Copy src[tid] → dst[offset + tid] for tid in [0, count)."""
    tid = wp.tid()
    if tid < count:
        dst[offset + tid] = src[tid]


@wp.kernel
def _snapshot_mat33(
    src: wp.array(dtype=wp.mat33),
    dst: wp.array(dtype=wp.mat33),
    offset: int,
    count: int,
):
    tid = wp.tid()
    if tid < count:
        dst[offset + tid] = src[tid]


@wp.kernel
def _snapshot_float(
    src: wp.array(dtype=float),
    dst: wp.array(dtype=float),
    offset: int,
    count: int,
):
    tid = wp.tid()
    if tid < count:
        dst[offset + tid] = src[tid]


@wp.kernel
def _snapshot_int(
    src: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.int32),
    offset: int,
    count: int,
):
    tid = wp.tid()
    if tid < count:
        dst[offset + tid] = src[tid]


@wp.kernel
def _snapshot_scalar_to_array(
    src: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.int32),
    idx: int,
):
    """Copy src[0] → dst[idx]."""
    dst[idx] = src[0]


class DebugRecorder:
    """Pre-allocated GPU recording buffers for VBD contact debugging.

    Records two levels of data:
    - **Per-substep**: positions, velocities, inertia targets, contact counts/distances
    - **Per-iteration**: force components (inertia, elastic, bending, contact),
      hessians, displacements, truncation factors

    Usage::

        recorder = DebugRecorder(model, max_substeps=600, iterations=5, device=device)
        solver.debug_recorder = recorder
        # ... run simulation ...
        data = recorder.to_dict()
        np.savez_compressed("debug_data.npz", **data)
    """

    def __init__(
        self,
        model,
        max_substeps: int,
        iterations: int,
        device: str = "cuda:0",
    ):
        self.device = device
        self.particle_count = model.particle_count
        self.edge_count = model.edge_count
        self.tri_count = model.tri_count
        self.max_substeps = max_substeps
        self.iterations = iterations
        self.max_total_iters = max_substeps * iterations

        N = self.particle_count
        E = self.edge_count
        T = self.tri_count
        S = max_substeps
        I = self.max_total_iters

        # Tracking counters (Python-side; graph-capture would use device-side)
        self._substep_idx = 0
        self._iter_idx = 0  # global iteration counter across all substeps

        # ---- Per-substep buffers [S * N], [S * E], [S * T] ----

        # Positions & velocities at substep start
        self.positions = wp.zeros(S * N, dtype=wp.vec3, device=device)
        self.velocities = wp.zeros(S * N, dtype=wp.vec3, device=device)
        self.inertia_targets = wp.zeros(S * N, dtype=wp.vec3, device=device)

        # Self-contact: VT per vertex, EE per edge, TV per triangle
        self.vt_contact_counts = wp.zeros(S * N, dtype=wp.int32, device=device)
        self.vt_min_dist = wp.zeros(S * N, dtype=float, device=device)
        self.ee_contact_counts = wp.zeros(S * E, dtype=wp.int32, device=device)
        self.ee_min_dist = wp.zeros(S * E, dtype=float, device=device)
        self.tv_contact_counts = wp.zeros(S * T, dtype=wp.int32, device=device)
        self.tv_min_dist = wp.zeros(S * T, dtype=float, device=device)

        # Body-particle contact count per substep
        self.body_contact_count = wp.zeros(S, dtype=wp.int32, device=device)

        # Final velocities after substep
        self.velocities_end = wp.zeros(S * N, dtype=wp.vec3, device=device)

        # ---- Per-iteration buffers [I * N] ----

        # Pre-solve forces (body contact + spring + self-contact from buffer)
        self.pre_solve_forces = wp.zeros(I * N, dtype=wp.vec3, device=device)
        self.pre_solve_hessians = wp.zeros(I * N, dtype=wp.mat33, device=device)

        # Force components from inside solve_elasticity
        self.f_inertia = wp.zeros(I * N, dtype=wp.vec3, device=device)
        self.f_elastic = wp.zeros(I * N, dtype=wp.vec3, device=device)
        self.f_bending = wp.zeros(I * N, dtype=wp.vec3, device=device)

        # Displacements and truncation
        self.displacements = wp.zeros(I * N, dtype=wp.vec3, device=device)
        self.truncation_ts = wp.zeros(I * N, dtype=float, device=device)

    # -------------------------------------------------------------------
    # Recording methods (called by the solver at specific points)
    # -------------------------------------------------------------------

    def record_substep_start(self, state_in, solver):
        """Snapshot positions, velocities, inertia after forward_step."""
        s = self._substep_idx
        if s >= self.max_substeps:
            return
        N = self.particle_count
        off = s * N

        wp.launch(_snapshot_vec3, dim=N, inputs=[solver.particle_q_prev, self.positions, off, N], device=self.device)
        wp.launch(_snapshot_vec3, dim=N, inputs=[state_in.particle_qd, self.velocities, off, N], device=self.device)
        wp.launch(_snapshot_vec3, dim=N, inputs=[solver.inertia, self.inertia_targets, off, N], device=self.device)

    def record_contacts(self, solver, contacts):
        """Snapshot contact counts/distances after collision detection."""
        s = self._substep_idx
        if s >= self.max_substeps:
            return
        N = self.particle_count
        E = self.edge_count
        T = self.tri_count

        # Self-contact data from trimesh_collision_detector
        if solver.particle_enable_self_contact:
            det = solver.trimesh_collision_detector

            # VT contacts per vertex
            wp.launch(
                _snapshot_int, dim=N,
                inputs=[det.vertex_colliding_triangles_count, self.vt_contact_counts, s * N, N],
                device=self.device,
            )
            wp.launch(
                _snapshot_float, dim=N,
                inputs=[det.vertex_colliding_triangles_min_dist, self.vt_min_dist, s * N, N],
                device=self.device,
            )

            # EE contacts per edge
            wp.launch(
                _snapshot_int, dim=E,
                inputs=[det.edge_colliding_edges_count, self.ee_contact_counts, s * E, E],
                device=self.device,
            )
            wp.launch(
                _snapshot_float, dim=E,
                inputs=[det.edge_colliding_edges_min_dist, self.ee_min_dist, s * E, E],
                device=self.device,
            )

            # TV contacts per triangle
            if det.triangle_colliding_vertices_count is not None:
                wp.launch(
                    _snapshot_int, dim=T,
                    inputs=[det.triangle_colliding_vertices_count, self.tv_contact_counts, s * T, T],
                    device=self.device,
                )
            if det.triangle_colliding_vertices_min_dist is not None:
                wp.launch(
                    _snapshot_float, dim=T,
                    inputs=[det.triangle_colliding_vertices_min_dist, self.tv_min_dist, s * T, T],
                    device=self.device,
                )

        # Body-particle contact count
        if contacts is not None:
            wp.launch(
                _snapshot_scalar_to_array, dim=1,
                inputs=[contacts.soft_contact_count, self.body_contact_count, s],
                device=self.device,
            )

    def record_iteration(self, solver, state_in):
        """Snapshot forces, hessians, displacements after all colors of one iteration."""
        i = self._iter_idx
        if i >= self.max_total_iters:
            return
        N = self.particle_count
        off = i * N

        # Pre-solve forces/hessians (contact + spring from buffer)
        wp.launch(_snapshot_vec3, dim=N, inputs=[solver.particle_forces, self.pre_solve_forces, off, N], device=self.device)
        wp.launch(_snapshot_mat33, dim=N, inputs=[solver.particle_hessians, self.pre_solve_hessians, off, N], device=self.device)

        # Force components from solve_elasticity debug output
        if hasattr(solver, 'debug_f_inertia'):
            wp.launch(_snapshot_vec3, dim=N, inputs=[solver.debug_f_inertia, self.f_inertia, off, N], device=self.device)
            wp.launch(_snapshot_vec3, dim=N, inputs=[solver.debug_f_elastic, self.f_elastic, off, N], device=self.device)
            wp.launch(_snapshot_vec3, dim=N, inputs=[solver.debug_f_bending, self.f_bending, off, N], device=self.device)

        # Displacements and truncation
        wp.launch(_snapshot_vec3, dim=N, inputs=[solver.particle_displacements, self.displacements, off, N], device=self.device)
        wp.launch(_snapshot_float, dim=N, inputs=[solver.truncation_ts, self.truncation_ts, off, N], device=self.device)

    def record_substep_end(self, state_out):
        """Snapshot final velocities after finalization."""
        s = self._substep_idx
        if s >= self.max_substeps:
            return
        N = self.particle_count
        off = s * N
        wp.launch(_snapshot_vec3, dim=N, inputs=[state_out.particle_qd, self.velocities_end, off, N], device=self.device)

    def advance_iteration(self):
        self._iter_idx += 1

    def advance_substep(self):
        self._substep_idx += 1

    @property
    def substeps_recorded(self):
        return self._substep_idx

    @property
    def iterations_recorded(self):
        return self._iter_idx

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert all recorded data to a numpy dict for saving with np.savez."""
        S = self._substep_idx
        I = self._iter_idx
        N = self.particle_count
        E = self.edge_count
        T = self.tri_count

        def _to_np(arr, total):
            """Copy to CPU and return numpy array."""
            return arr.numpy()[:total]

        d = {
            "particle_count": N,
            "edge_count": E,
            "tri_count": T,
            "substeps_recorded": S,
            "iterations_recorded": I,
            "iterations_per_substep": self.iterations,
            # Per-substep [S, N] or [S, E] or [S, T]
            "positions": _to_np(self.positions, S * N).reshape(S, N, 3),
            "velocities": _to_np(self.velocities, S * N).reshape(S, N, 3),
            "velocities_end": _to_np(self.velocities_end, S * N).reshape(S, N, 3),
            "inertia_targets": _to_np(self.inertia_targets, S * N).reshape(S, N, 3),
            "vt_contact_counts": _to_np(self.vt_contact_counts, S * N).reshape(S, N),
            "vt_min_dist": _to_np(self.vt_min_dist, S * N).reshape(S, N),
            "ee_contact_counts": _to_np(self.ee_contact_counts, S * E).reshape(S, E),
            "ee_min_dist": _to_np(self.ee_min_dist, S * E).reshape(S, E),
            "tv_contact_counts": _to_np(self.tv_contact_counts, S * T).reshape(S, T),
            "tv_min_dist": _to_np(self.tv_min_dist, S * T).reshape(S, T),
            "body_contact_count": _to_np(self.body_contact_count, S),
            # Per-iteration [I, N]
            "pre_solve_forces": _to_np(self.pre_solve_forces, I * N).reshape(I, N, 3),
            "pre_solve_hessians": _to_np(self.pre_solve_hessians, I * N).reshape(I, N, 3, 3),
            "f_inertia": _to_np(self.f_inertia, I * N).reshape(I, N, 3),
            "f_elastic": _to_np(self.f_elastic, I * N).reshape(I, N, 3),
            "f_bending": _to_np(self.f_bending, I * N).reshape(I, N, 3),
            "displacements": _to_np(self.displacements, I * N).reshape(I, N, 3),
            "truncation_ts": _to_np(self.truncation_ts, I * N).reshape(I, N),
        }
        return d
