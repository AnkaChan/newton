# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gauss-Newton projection from maximal to reduced coordinates.

After AVBD's maximal-coordinate solve, this module projects body poses onto
the kinematic manifold defined by the articulation's joint structure.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim.articulation import eval_fk, eval_ik, eval_jacobian
from ...sim.enums import JointType
from ...sim.model import Model
from ...sim.state import State

# Joint types for which n_coords == n_dofs and the DOF delta maps directly into
# the coord array. BALL/FREE/DISTANCE use quaternion parametrization, and D6 may
# also mix unit-quaternion components; those need a proper exp-map update and
# are excluded from the per-DOF coord write below.
_SIMPLE_JOINT_TYPES = {int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.FIXED)}


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (x, y, z, w) convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _compute_body_residual(
    body_q_fk: np.ndarray,
    body_q_target: np.ndarray,
    body_indices: list[int],
    n_links: int,
) -> np.ndarray:
    """Compute 6-DOF residual per link: [pos_err(3), rot_err(3)].

    Args:
        body_q_fk: FK-predicted transforms, shape (n_bodies, 7).
        body_q_target: AVBD maximal transforms, shape (n_bodies, 7).
        body_indices: Global body index for each link in the articulation.
        n_links: Number of links in the articulation.

    Returns:
        Residual vector of shape (n_links * 6,).
    """
    residual = np.zeros(n_links * 6)
    for i, body_idx in enumerate(body_indices):
        # Position error
        pos_fk = body_q_fk[body_idx, :3]
        pos_target = body_q_target[body_idx, :3]
        residual[i * 6 : i * 6 + 3] = pos_fk - pos_target

        # Orientation error: q_err = q_fk^{-1} * q_target → rotation vector
        q_fk = body_q_fk[body_idx, 3:]  # (x, y, z, w)
        q_target = body_q_target[body_idx, 3:]
        q_fk_inv = np.array([-q_fk[0], -q_fk[1], -q_fk[2], q_fk[3]])
        q_err = _quat_multiply(q_fk_inv, q_target)

        # Shortest path
        if q_err[3] < 0.0:
            q_err = -q_err

        residual[i * 6 + 3 : i * 6 + 6] = 2.0 * q_err[:3]

    return residual


def project_to_reduced_coordinates(
    model: Model,
    state: State,
    joint_q_prev: wp.array,
    dt: float,
    gn_iterations: int = 3,
    damping: float = 1e-6,
    max_joint_vel: float = 20.0,
) -> None:
    """Project maximal body_q onto the reduced-coordinate manifold.

    After the GN solve finds the closest on-manifold joint_q, the joint
    velocity is derived via BDF1 and clamped to ``max_joint_vel`` to prevent
    explosion from large projection corrections.  FK then produces consistent
    ``body_q`` and ``body_qd``.

    Args:
        model: The model containing articulation definitions.
        state: The state to project (body_q is read as the AVBD target, then
            overwritten with the FK-projected result).
        joint_q_prev: Previous step's projected joint coordinates (used for
            BDF1 velocity).
        dt: Timestep [s].
        gn_iterations: Number of Gauss-Newton iterations (0 = analytical IK
            projection only).
        damping: Levenberg-Marquardt damping for the normal equations.
        max_joint_vel: Maximum joint velocity [rad/s or m/s] for clamping.
    """
    if model.articulation_count == 0 or model.body_count == 0:
        return

    # --- Pre-fetch model topology (CPU) ---
    art_start_np = model.articulation_start.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_type_np = model.joint_type.numpy()
    joint_qd_start_np = model.joint_qd_start.numpy()
    joint_q_start_np = model.joint_q_start.numpy()
    total_coords = int(state.joint_q.shape[0])
    total_dofs = int(state.joint_qd.shape[0])

    # --- Identify articulations whose every joint is 1-coord-per-DOF
    # (PRISMATIC/REVOLUTE/FIXED). Articulations containing BALL/FREE/D6/DISTANCE
    # are left untouched: their joint_q includes quaternion components that
    # cannot be safely updated by a DOF-space delta or per-coord clamp. ---
    managed_arts: list[int] = []
    managed_coord_mask = np.zeros(total_coords, dtype=bool)
    managed_dof_mask = np.zeros(total_dofs, dtype=bool)
    for art_idx in range(model.articulation_count):
        joint_start = int(art_start_np[art_idx])
        joint_end = int(art_start_np[art_idx + 1])
        if joint_end <= joint_start:
            continue
        if not all(int(joint_type_np[j]) in _SIMPLE_JOINT_TYPES for j in range(joint_start, joint_end)):
            continue
        managed_arts.append(art_idx)
        q_start = int(joint_q_start_np[joint_start])
        q_end = int(joint_q_start_np[joint_end])
        d_start = int(joint_qd_start_np[joint_start])
        d_end = int(joint_qd_start_np[joint_end])
        managed_coord_mask[q_start:q_end] = True
        managed_dof_mask[d_start:d_end] = True

    if not managed_arts:
        return

    # --- Save AVBD maximal result as projection target ---
    body_q_target = wp.clone(state.body_q)

    # --- Warm-start joint_q via analytical per-joint IK ---
    eval_ik(model, state, state.joint_q, state.joint_qd)

    if gn_iterations > 0:
        body_q_target_np = body_q_target.numpy().reshape(-1, 7)

        # --- Gauss-Newton iterations ---
        for _k in range(gn_iterations):
            # FK from current joint_q → state.body_q
            eval_fk(model, state.joint_q, state.joint_qd, state)

            # Jacobian (GPU kernel, then pull to CPU)
            J_wp = eval_jacobian(model, state)
            J_np = J_wp.numpy()  # (art_count, max_links*6, max_dofs)

            # Pull current body_q and joint_q to CPU
            body_q_fk_np = state.body_q.numpy().reshape(-1, 7)
            joint_q_np = state.joint_q.numpy().copy()

            # Solve per managed articulation
            for art_idx in managed_arts:
                joint_start = int(art_start_np[art_idx])
                joint_end = int(art_start_np[art_idx + 1])
                n_links = joint_end - joint_start

                # Body indices for this articulation's links
                body_indices = [int(joint_child_np[j]) for j in range(joint_start, joint_end)]

                # Articulation DOF range
                dof_start = int(joint_qd_start_np[joint_start])
                dof_end = int(joint_qd_start_np[joint_end])
                n_dofs = dof_end - dof_start

                if n_dofs == 0:
                    continue

                # Residual: FK vs AVBD target
                r = _compute_body_residual(body_q_fk_np, body_q_target_np, body_indices, n_links)

                # Extract this articulation's Jacobian block
                J_art = J_np[art_idx, : n_links * 6, :n_dofs]

                # Normal equations: (J^T J + λI) Δq = -J^T r
                JtJ = J_art.T @ J_art + damping * np.eye(n_dofs)
                Jtr = J_art.T @ r
                delta_q = np.linalg.solve(JtJ, -Jtr)

                # Map DOF delta to coordinate update (safe: managed joints have
                # n_coords == n_dofs).
                q_start = int(joint_q_start_np[joint_start])
                joint_q_np[q_start : q_start + n_dofs] += delta_q

            # Push updated joint_q back to GPU
            state.joint_q.assign(wp.array(joint_q_np, dtype=float, device=state.joint_q.device))

    # --- Clamp joint_q change to enforce velocity limit (managed coords only) ---
    joint_q_np = state.joint_q.numpy().copy()
    joint_q_prev_np = joint_q_prev.numpy()
    max_dq = max_joint_vel * dt
    delta = joint_q_np - joint_q_prev_np
    delta_clamped = np.where(managed_coord_mask, np.clip(delta, -max_dq, max_dq), delta)
    joint_q_clamped = joint_q_prev_np + delta_clamped
    # Preserve unmanaged coords exactly as VBD/IK left them.
    joint_q_clamped = np.where(managed_coord_mask, joint_q_clamped, joint_q_np)
    state.joint_q.assign(wp.array(joint_q_clamped, dtype=float, device=state.joint_q.device))

    # --- BDF1 velocity for managed DOFs; leave others as VBD/IK set them ---
    # For managed (PRISMATIC/REVOLUTE/FIXED) joints, n_coords == n_dofs so the
    # coord-mask positions align 1:1 with the dof-mask positions.
    if dt > 0.0 and managed_dof_mask.any():
        joint_qd_np = state.joint_qd.numpy().copy()
        managed_delta_per_dof = delta_clamped[managed_coord_mask]
        joint_qd_np[managed_dof_mask] = managed_delta_per_dof / dt
        state.joint_qd.assign(wp.array(joint_qd_np, dtype=float, device=state.joint_qd.device))

    # --- Final FK → body_q and body_qd ---
    eval_fk(model, state.joint_q, state.joint_qd, state)
