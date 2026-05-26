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
from ...sim.model import Model
from ...sim.state import State


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (x, y, z, w) convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


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
    gn_iterations: int = 3,
    damping: float = 1e-6,
) -> None:
    """Project maximal body_q onto the reduced-coordinate manifold.

    Modifies ``state.body_q``, ``state.joint_q``, and ``state.joint_qd`` in
    place.  Call this after the AVBD iterations and **before** the velocity
    finalize so that ``body_qd`` is derived from the projected poses.

    Args:
        model: The model containing articulation definitions.
        state: The state to project (body_q is read as the AVBD target, then
            overwritten with the FK-projected result).
        gn_iterations: Number of Gauss-Newton iterations (0 = analytical IK
            projection only).
        damping: Levenberg-Marquardt damping for the normal equations.
    """
    if model.articulation_count == 0 or model.body_count == 0:
        return

    # --- Save AVBD maximal result as projection target ---
    body_q_target = wp.clone(state.body_q)

    # --- Warm-start joint_q via analytical per-joint IK ---
    eval_ik(model, state, state.joint_q, state.joint_qd)

    if gn_iterations == 0:
        # Pure analytical projection: FK from IK-derived joint_q
        eval_fk(model, state.joint_q, state.joint_qd, state)
        return

    # --- Pre-fetch model topology (CPU) ---
    art_start_np = model.articulation_start.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_qd_start_np = model.joint_qd_start.numpy()
    joint_q_start_np = model.joint_q_start.numpy()
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

        # Solve per articulation
        for art_idx in range(model.articulation_count):
            joint_start = art_start_np[art_idx]
            joint_end = art_start_np[art_idx + 1]
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

            # Map DOF delta to coordinate update.
            # For revolute/prismatic joints (Franka), coord and DOF indices
            # are 1:1.  Ball/free joints would need a separate mapping.
            q_start = int(joint_q_start_np[joint_start])
            joint_q_np[q_start : q_start + n_dofs] += delta_q

        # Push updated joint_q back to GPU
        state.joint_q.assign(wp.array(joint_q_np, dtype=float, device=state.joint_q.device))

    # --- Final FK from converged joint_q ---
    eval_fk(model, state.joint_q, state.joint_qd, state)
