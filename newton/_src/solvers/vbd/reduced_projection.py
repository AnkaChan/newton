# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gauss-Newton projection from maximal to reduced coordinates.

After VBD's maximal-coordinate rigid-body solve, this module projects body
poses onto the kinematic manifold defined by the articulation's joint
structure. The projection runs entirely on the device (all fixed-dimension
kernel launches), so it is CUDA-graph capturable.

Pipeline per :meth:`ReducedCoordinateProjection.project` call:

1. Save the maximal ``body_q`` as the projection target.
2. ``eval_ik`` warm-starts ``joint_q``/``joint_qd`` from the maximal state.
3. Managed joint coordinates are sanitized (non-finite -> previous value)
   and clamped to their joint limits.
4. A fixed number of Gauss-Newton iterations refine ``joint_q``: masked FK,
   spatial Jacobian, then a per-articulation damped normal-equations solve.
5. The per-step coordinate change is clamped to ``max_joint_vel * dt`` and
   the recovered joint velocity to ``max_joint_vel``.
6. A final masked FK writes the projected ``body_q``/``body_qd``.

Only *managed* articulations participate: those whose joints are all
REVOLUTE, PRISMATIC, or FIXED (one coordinate per DOF). Articulations
containing BALL/FREE/D6/DISTANCE joints keep their maximal-solve result
untouched; their ``joint_q``/``joint_qd`` are still refreshed by ``eval_ik``
so downstream consumers see a consistent reduced state.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

from ...sim.articulation import eval_fk, eval_ik, eval_jacobian
from ...sim.enums import JointType
from ...sim.model import Model
from ...sim.state import State

_VERSION = "rvbd_warp_v1"

# Joint types for which n_coords == n_dofs and a DOF-space delta maps directly
# onto the coordinate array. BALL/FREE/DISTANCE use quaternion coordinates and
# D6 may mix them; those need an exp-map update and are excluded.
_SIMPLE_JOINT_TYPES = (int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.FIXED))


@wp.kernel
def sanitize_and_clamp_coords(
    coord_is_managed: wp.array[wp.int32],
    coord_limit_lower: wp.array[float],
    coord_limit_upper: wp.array[float],
    joint_q_prev: wp.array[float],
    # outputs
    joint_q: wp.array[float],
):
    """Reset non-finite managed coordinates to their previous value and clamp to joint limits."""
    tid = wp.tid()
    if coord_is_managed[tid] == 0:
        return

    q = joint_q[tid]
    if not wp.isfinite(q):
        q = joint_q_prev[tid]
    joint_q[tid] = wp.clamp(q, coord_limit_lower[tid], coord_limit_upper[tid])


@wp.kernel
def clamp_coord_delta(
    coord_is_managed: wp.array[wp.int32],
    joint_q_prev: wp.array[float],
    max_delta: float,
    # outputs
    joint_q: wp.array[float],
):
    """Clamp the per-step change of managed coordinates to ``max_delta``."""
    tid = wp.tid()
    if coord_is_managed[tid] == 0:
        return

    delta = joint_q[tid] - joint_q_prev[tid]
    if not wp.isfinite(delta):
        delta = 0.0
    joint_q[tid] = joint_q_prev[tid] + wp.clamp(delta, -max_delta, max_delta)


@wp.kernel
def clamp_dof_velocity(
    dof_is_managed: wp.array[wp.int32],
    max_vel: float,
    # outputs
    joint_qd: wp.array[float],
):
    """Reset non-finite managed joint velocities to zero and clamp to ``max_vel``."""
    tid = wp.tid()
    if dof_is_managed[tid] == 0:
        return

    qd = joint_qd[tid]
    if not wp.isfinite(qd):
        qd = 0.0
    joint_qd[tid] = wp.clamp(qd, -max_vel, max_vel)


@wp.kernel
def gauss_newton_solve_articulation(
    articulation_mask: wp.array[bool],
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    body_q_fk: wp.array[wp.transform],
    body_q_target: wp.array[wp.transform],
    J: wp.array3d[float],
    coord_limit_lower: wp.array[float],
    coord_limit_upper: wp.array[float],
    damping: float,
    # outputs
    joint_q: wp.array[float],
    JtJ: wp.array3d[float],
    rhs: wp.array2d[float],
):
    """One damped Gauss-Newton step per managed articulation.

    Accumulates the normal equations ``(J^T J + damping I) dq = -J^T r`` from
    the 6-per-link pose residual between the current FK poses and the maximal
    (target) poses, solves them by in-place Cholesky in the global scratch
    buffers, and applies the limit-clamped coordinate update.
    """
    art = wp.tid()
    if not articulation_mask[art]:
        return

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]
    n_links = joint_end - joint_start

    dof_start = joint_qd_start[joint_start]
    n_dofs = joint_qd_start[joint_end] - dof_start
    if n_dofs == 0:
        return

    q_start = joint_q_start[joint_start]

    for a in range(n_dofs):
        rhs[art, a] = 0.0
        for b in range(n_dofs):
            JtJ[art, a, b] = 0.0

    # Accumulate J^T J and -J^T r over the 6 residual rows of each link.
    for li in range(n_links):
        child = joint_child[joint_start + li]

        X_fk = body_q_fk[child]
        X_t = body_q_target[child]

        r_pos = wp.transform_get_translation(X_fk) - wp.transform_get_translation(X_t)

        q_err = wp.quat_inverse(wp.transform_get_rotation(X_t)) * wp.transform_get_rotation(X_fk)
        if q_err[3] < 0.0:
            q_err = -q_err
        r_rot = 2.0 * wp.vec3(q_err[0], q_err[1], q_err[2])

        row = li * 6
        for a in range(n_dofs):
            Ja = wp.spatial_vector(
                J[art, row + 0, a],
                J[art, row + 1, a],
                J[art, row + 2, a],
                J[art, row + 3, a],
                J[art, row + 4, a],
                J[art, row + 5, a],
            )
            rhs[art, a] -= wp.dot(wp.spatial_top(Ja), r_pos) + wp.dot(wp.spatial_bottom(Ja), r_rot)
            for b in range(a, n_dofs):
                acc = float(0.0)
                for k in range(6):
                    acc += J[art, row + k, a] * J[art, row + k, b]
                JtJ[art, a, b] += acc

    # Symmetrize and damp the diagonal.
    for a in range(n_dofs):
        JtJ[art, a, a] += damping
        for b in range(a + 1, n_dofs):
            JtJ[art, b, a] = JtJ[art, a, b]

    # In-place Cholesky factorization (lower triangle).
    for k in range(n_dofs):
        pivot = JtJ[art, k, k]
        for j in range(k):
            pivot -= JtJ[art, k, j] * JtJ[art, k, j]
        if not wp.isfinite(pivot) or pivot <= 0.0:
            return
        pivot = wp.sqrt(pivot)
        JtJ[art, k, k] = pivot
        for i in range(k + 1, n_dofs):
            s = JtJ[art, i, k]
            for j in range(k):
                s -= JtJ[art, i, j] * JtJ[art, k, j]
            JtJ[art, i, k] = s / pivot

    # Forward substitution: L y = rhs.
    for i in range(n_dofs):
        s = rhs[art, i]
        for j in range(i):
            s -= JtJ[art, i, j] * rhs[art, j]
        rhs[art, i] = s / JtJ[art, i, i]

    # Backward substitution: L^T dq = y.
    for ii in range(n_dofs):
        i = n_dofs - 1 - ii
        s = rhs[art, i]
        for j in range(i + 1, n_dofs):
            s -= JtJ[art, j, i] * rhs[art, j]
        rhs[art, i] = s / JtJ[art, i, i]

    # Reject the whole update if any component is non-finite.
    for a in range(n_dofs):
        if not wp.isfinite(rhs[art, a]):
            return

    # Apply the limit-clamped coordinate update (managed joints have
    # n_coords == n_dofs, so coordinate a aligns with DOF a).
    for a in range(n_dofs):
        c = q_start + a
        joint_q[c] = wp.clamp(joint_q[c] + rhs[art, a], coord_limit_lower[c], coord_limit_upper[c])


class ReducedCoordinateProjection:
    """Device-side reduced-coordinate projection for :class:`SolverVBD`.

    Precomputes the static articulation topology (which articulations,
    coordinates, and DOFs are managed) and all scratch buffers at
    construction, so :meth:`project` issues only fixed-dimension kernel
    launches and is CUDA-graph capturable.

    Args:
        model: The model containing articulation definitions.
        gn_iterations: Number of Gauss-Newton iterations (0 = analytical IK
            projection only).
        damping: Levenberg-Marquardt damping for the normal equations.
        max_joint_vel: Maximum joint velocity [rad/s or m/s]. Clamps both the
            per-step coordinate change (``max_joint_vel * dt``) and the
            recovered joint velocity.
    """

    def __init__(
        self,
        model: Model,
        gn_iterations: int = 3,
        damping: float = 1e-6,
        max_joint_vel: float = 20.0,
    ):
        self.model = model
        self.device = model.device
        self.gn_iterations = gn_iterations
        self.damping = damping
        self.max_joint_vel = max_joint_vel

        art_start_np = model.articulation_start.numpy()
        joint_type_np = model.joint_type.numpy()
        joint_q_start_np = model.joint_q_start.numpy()
        joint_qd_start_np = model.joint_qd_start.numpy()

        total_coords = model.joint_coord_count
        total_dofs = model.joint_dof_count

        art_mask_np = np.zeros(model.articulation_count, dtype=bool)
        coord_managed_np = np.zeros(total_coords, dtype=np.int32)
        dof_managed_np = np.zeros(total_dofs, dtype=np.int32)
        coord_lo_np = np.full(total_coords, -np.inf, dtype=np.float32)
        coord_hi_np = np.full(total_coords, np.inf, dtype=np.float32)

        limit_lo_np = model.joint_limit_lower.numpy() if model.joint_limit_lower is not None else None
        limit_hi_np = model.joint_limit_upper.numpy() if model.joint_limit_upper is not None else None

        for art_idx in range(model.articulation_count):
            joint_start = int(art_start_np[art_idx])
            joint_end = int(art_start_np[art_idx + 1])
            if joint_end <= joint_start:
                continue
            if not all(int(joint_type_np[j]) in _SIMPLE_JOINT_TYPES for j in range(joint_start, joint_end)):
                continue

            art_mask_np[art_idx] = True
            q_start = int(joint_q_start_np[joint_start])
            d_start = int(joint_qd_start_np[joint_start])
            n_dofs = int(joint_qd_start_np[joint_end]) - d_start
            coord_managed_np[q_start : q_start + n_dofs] = 1
            dof_managed_np[d_start : d_start + n_dofs] = 1
            if limit_lo_np is not None and limit_hi_np is not None:
                lo = limit_lo_np[d_start : d_start + n_dofs]
                hi = limit_hi_np[d_start : d_start + n_dofs]
                coord_lo_np[q_start : q_start + n_dofs] = np.where(np.isfinite(lo), lo, -np.inf)
                coord_hi_np[q_start : q_start + n_dofs] = np.where(np.isfinite(hi), hi, np.inf)

        self.managed_articulation_count = int(art_mask_np.sum())

        if self.managed_articulation_count == 0:
            return

        with wp.ScopedDevice(self.device):
            self.articulation_mask = wp.array(art_mask_np, dtype=bool)
            self.coord_is_managed = wp.array(coord_managed_np, dtype=wp.int32)
            self.dof_is_managed = wp.array(dof_managed_np, dtype=wp.int32)
            self.coord_limit_lower = wp.array(coord_lo_np, dtype=float)
            self.coord_limit_upper = wp.array(coord_hi_np, dtype=float)

            # Previous projected joint coordinates (for the per-step delta clamp).
            self.joint_q_prev = wp.clone(model.joint_q)

            # Scratch buffers reused across steps.
            max_links = model.max_joints_per_articulation
            max_dofs = model.max_dofs_per_articulation
            self.body_q_target = wp.zeros(model.body_count, dtype=wp.transform)
            self.J = wp.zeros((model.articulation_count, max_links * 6, max_dofs), dtype=float)
            self.joint_S_s = wp.zeros(total_dofs, dtype=wp.spatial_vector)
            self.JtJ = wp.zeros((model.articulation_count, max_dofs, max_dofs), dtype=float)
            self.rhs = wp.zeros((model.articulation_count, max_dofs), dtype=float)

        if os.environ.get("NEWTON_RVBD_VERBOSE"):
            print(f"[reduced_projection] version: {_VERSION}, managed articulations: {self.managed_articulation_count}")

    def project(self, state: State, dt: float) -> None:
        """Project the maximal ``state.body_q`` onto the kinematic manifold.

        Overwrites managed articulations' ``body_q``/``body_qd`` with the
        FK-consistent projected result and refreshes ``joint_q``/``joint_qd``
        for all articulations. Unmanaged articulations keep their
        maximal-solve body state.

        Args:
            state: State to project in place.
            dt: Timestep [s].
        """
        if self.managed_articulation_count == 0:
            return

        model = self.model

        # Save the maximal result as the projection target.
        wp.copy(self.body_q_target, state.body_q)

        # Warm-start joint coordinates from the maximal state. Unmasked so
        # unmanaged (e.g. FREE-joint) articulations also get a consistent
        # reduced state for downstream consumers.
        eval_ik(model, state, state.joint_q, state.joint_qd)

        wp.launch(
            kernel=sanitize_and_clamp_coords,
            dim=model.joint_coord_count,
            inputs=[
                self.coord_is_managed,
                self.coord_limit_lower,
                self.coord_limit_upper,
                self.joint_q_prev,
            ],
            outputs=[state.joint_q],
            device=self.device,
        )

        for _ in range(self.gn_iterations):
            eval_fk(model, state.joint_q, state.joint_qd, state, mask=self.articulation_mask)
            eval_jacobian(model, state, J=self.J, joint_S_s=self.joint_S_s, mask=self.articulation_mask)
            wp.launch(
                kernel=gauss_newton_solve_articulation,
                dim=model.articulation_count,
                inputs=[
                    self.articulation_mask,
                    model.articulation_start,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    state.body_q,
                    self.body_q_target,
                    self.J,
                    self.coord_limit_lower,
                    self.coord_limit_upper,
                    self.damping,
                ],
                outputs=[state.joint_q, self.JtJ, self.rhs],
                device=self.device,
            )

        # Keep the projection correction local: clamp the per-step coordinate
        # change, then clamp the recovered joint velocity. The velocity keeps
        # eval_ik's tangent-space interpretation of the maximal body_qd; it is
        # NOT recomputed from the position correction (BDF1), which under
        # high-stiffness PD drives turns projection corrections into
        # explosive drive forces.
        wp.launch(
            kernel=clamp_coord_delta,
            dim=model.joint_coord_count,
            inputs=[self.coord_is_managed, self.joint_q_prev, self.max_joint_vel * dt],
            outputs=[state.joint_q],
            device=self.device,
        )
        wp.launch(
            kernel=clamp_dof_velocity,
            dim=model.joint_dof_count,
            inputs=[self.dof_is_managed, self.max_joint_vel],
            outputs=[state.joint_qd],
            device=self.device,
        )

        # Final FK writes the projected body_q/body_qd for managed
        # articulations only; unmanaged ones keep the maximal-solve result.
        eval_fk(model, state.joint_q, state.joint_qd, state, mask=self.articulation_mask)

        wp.copy(self.joint_q_prev, state.joint_q)
