# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Device-resident matrix-free correction primitives for MG-VBD research.

This module is a Warp implementation of the frozen stable-Neo-Hookean
Gauss-Newton operator in :mod:`research.principal_stretch.correction_gpu`.
It is intentionally research-only and is not integrated with
:class:`newton.solvers.SolverVBD`.

The elastic gather is deterministic: every free vertex owns one sorted CSR
row of ``(tet, local corner)`` incidences and accumulates that row in a single
thread.  No floating-point atomics are used.  Dirichlet degrees of freedom are
eliminated from every Krylov vector; pinned vertices can only enter through
the frozen position buffer used to evaluate deformation gradients.

The PCG launcher owns all buffers and always launches the same iteration
schedule.  Algebraic convergence and failures are device-side masks, never
host-side convergence branches.  Host records are diagnostic synchronization
points after execution.  Timings from this module are not performance
evidence until an integrated, captured benchmark establishes that separately.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import numbers
from collections.abc import Sequence

import numpy as np
import warp as wp

from .correction_gpu import MatrixFreeStableNHOperator

KERNEL_VERSION = "mg-vbd-warp-operator-v1"
CONTRACT_ID = "mg-vbd-warp-fixed-pcg-research-v1"


def _current_operator_arrays_sha256(
    *,
    tets: np.ndarray,
    shape_gradients: np.ndarray,
    volumes: np.ndarray,
    mass: np.ndarray,
    mu: np.ndarray,
    lam: np.ndarray,
    free: np.ndarray,
    cofactors: np.ndarray,
    dt: float,
) -> str:
    """Hash canonical arrays defining one free-space Gauss--Newton operator."""
    digest = hashlib.sha256()

    def add(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    add(b"matrix-free-stable-nh-current-gn-operator-v1")
    for name, value in (
        ("tets", tets),
        ("shape_gradients", shape_gradients),
        ("volumes", volumes),
        ("mass", mass),
        ("mu", mu),
        ("lam", lam),
        ("free", free),
        ("cofactors", cofactors),
    ):
        array = np.asarray(value)
        add(name.encode("utf-8"))
        add(array.dtype.str.encode("ascii"))
        add(repr(array.shape).encode("ascii"))
        add(np.ascontiguousarray(array).tobytes())
    add(b"dt")
    add(np.float64(dt).tobytes())
    return digest.hexdigest()


def _current_operator_sha256(operator: MatrixFreeStableNHOperator) -> str:
    """Hash the exact frozen NumPy free-space Gauss--Newton operator."""
    return _current_operator_arrays_sha256(
        tets=operator.tets,
        shape_gradients=operator.shape_gradients,
        volumes=operator.volumes,
        mass=operator.mass,
        mu=operator.mu,
        lam=operator.lam,
        free=operator.free,
        cofactors=operator.cofactors,
        dt=operator.dt,
    )


def _operator_preconditioner_binding_sha256(
    current_operator_sha256: str,
    preconditioner_identity: str,
    static_preconditioner_sha256: str | None,
) -> str:
    """Bind the current matrix-free operator to one exact preconditioner."""
    digest = hashlib.sha256()
    for value in (
        "warp-fixed-pcg-operator-preconditioner-binding-v1",
        current_operator_sha256,
        preconditioner_identity,
        "none" if static_preconditioner_sha256 is None else static_preconditioner_sha256,
    ):
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


@wp.func
def _outer3(left: wp.vec3d, right: wp.vec3d) -> wp.mat33d:
    return wp.mat33d(
        left[0] * right[0],
        left[0] * right[1],
        left[0] * right[2],
        left[1] * right[0],
        left[1] * right[1],
        left[1] * right[2],
        left[2] * right[0],
        left[2] * right[1],
        left[2] * right[2],
    )


@wp.func
def _zero_matrix() -> wp.mat33d:
    zero = wp.float64(0.0)
    return wp.mat33d(zero, zero, zero, zero, zero, zero, zero, zero, zero)


@wp.func
def _scaled_identity(scale: wp.float64) -> wp.mat33d:
    zero = wp.float64(0.0)
    return wp.mat33d(scale, zero, zero, zero, scale, zero, zero, zero, scale)


@wp.func
def _cofactor3(matrix: wp.mat33d) -> wp.mat33d:
    return wp.mat33d(
        matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1],
        matrix[1, 2] * matrix[2, 0] - matrix[1, 0] * matrix[2, 2],
        matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0],
        matrix[0, 2] * matrix[2, 1] - matrix[0, 1] * matrix[2, 2],
        matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0],
        matrix[0, 1] * matrix[2, 0] - matrix[0, 0] * matrix[2, 1],
        matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1],
        matrix[0, 2] * matrix[1, 0] - matrix[0, 0] * matrix[1, 2],
        matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0],
    )


@wp.func
def _determinant3(matrix: wp.mat33d) -> wp.float64:
    return (
        matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


@wp.func
def _double_dot(left: wp.mat33d, right: wp.mat33d) -> wp.float64:
    value = wp.float64(0.0)
    for row in range(3):
        for column in range(3):
            value += left[row, column] * right[row, column]
    return value


@wp.func
def _finite_vec3(value: wp.vec3d) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _finite_mat33(value: wp.mat33d) -> bool:
    finite = bool(True)
    for row in range(3):
        for column in range(3):
            finite = finite and wp.isfinite(value[row, column])
    return finite


@wp.func
def _inverse_spd3(block: wp.mat33d) -> wp.mat33d:
    cofactor = _cofactor3(block)
    determinant = _determinant3(block)
    inverse_determinant = wp.float64(1.0) / determinant
    return wp.mat33d(
        cofactor[0, 0] * inverse_determinant,
        cofactor[1, 0] * inverse_determinant,
        cofactor[2, 0] * inverse_determinant,
        cofactor[0, 1] * inverse_determinant,
        cofactor[1, 1] * inverse_determinant,
        cofactor[2, 1] * inverse_determinant,
        cofactor[0, 2] * inverse_determinant,
        cofactor[1, 2] * inverse_determinant,
        cofactor[2, 2] * inverse_determinant,
    )


@wp.kernel(enable_backward=False)
def _evaluate_geometry(
    positions: wp.array[wp.vec3d],
    tets: wp.array[int],
    shape_gradients: wp.array[wp.vec3d],
    mu: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    deformation_gradients: wp.array[wp.mat33d],
    cofactors: wp.array[wp.mat33d],
    determinants: wp.array[wp.float64],
    first_piola: wp.array[wp.mat33d],
):
    tet = wp.tid()
    deformation = _zero_matrix()
    for corner in range(4):
        entry = 4 * tet + corner
        deformation += _outer3(positions[tets[entry]], shape_gradients[entry])
    cofactor = _cofactor3(deformation)
    determinant = _determinant3(deformation)
    alpha = wp.float64(1.0) + mu[tet] / wp.max(lam[tet], wp.float64(1.0e-6))
    piola = mu[tet] * deformation + lam[tet] * (determinant - alpha) * cofactor
    deformation_gradients[tet] = deformation
    cofactors[tet] = cofactor
    determinants[tet] = determinant
    first_piola[tet] = piola


@wp.kernel(enable_backward=False)
def _gather_gradient(
    positions: wp.array[wp.vec3d],
    inertial_target: wp.array[wp.vec3d],
    mass: wp.array[wp.float64],
    free: wp.array[int],
    incidence_offsets: wp.array[int],
    incidence_tets: wp.array[int],
    incidence_corners: wp.array[int],
    shape_gradients: wp.array[wp.vec3d],
    volumes: wp.array[wp.float64],
    first_piola: wp.array[wp.mat33d],
    inverse_dt_squared: wp.float64,
    scale: wp.float64,
    output: wp.array[wp.vec3d],
):
    free_index = wp.tid()
    vertex = free[free_index]
    value = mass[vertex] * inverse_dt_squared * (positions[vertex] - inertial_target[vertex])
    start = incidence_offsets[free_index]
    end = incidence_offsets[free_index + 1]
    for cursor in range(start, end):
        tet = incidence_tets[cursor]
        corner = incidence_corners[cursor]
        value += volumes[tet] * (first_piola[tet] * shape_gradients[4 * tet + corner])
    output[free_index] = scale * value


@wp.kernel(enable_backward=False)
def _apply_tet_operator(
    direction: wp.array[wp.vec3d],
    tets: wp.array[int],
    vertex_to_free: wp.array[int],
    shape_gradients: wp.array[wp.vec3d],
    cofactors: wp.array[wp.mat33d],
    mu: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    delta_piola: wp.array[wp.mat33d],
):
    tet = wp.tid()
    delta_deformation = _zero_matrix()
    for corner in range(4):
        entry = 4 * tet + corner
        free_index = vertex_to_free[tets[entry]]
        if free_index >= 0:
            delta_deformation += _outer3(direction[free_index], shape_gradients[entry])
    determinant_direction = _double_dot(cofactors[tet], delta_deformation)
    delta_piola[tet] = mu[tet] * delta_deformation + lam[tet] * determinant_direction * cofactors[tet]


@wp.kernel(enable_backward=False)
def _gather_operator_product(
    direction: wp.array[wp.vec3d],
    mass: wp.array[wp.float64],
    free: wp.array[int],
    incidence_offsets: wp.array[int],
    incidence_tets: wp.array[int],
    incidence_corners: wp.array[int],
    shape_gradients: wp.array[wp.vec3d],
    volumes: wp.array[wp.float64],
    delta_piola: wp.array[wp.mat33d],
    inverse_dt_squared: wp.float64,
    output: wp.array[wp.vec3d],
):
    free_index = wp.tid()
    vertex = free[free_index]
    value = mass[vertex] * inverse_dt_squared * direction[free_index]
    start = incidence_offsets[free_index]
    end = incidence_offsets[free_index + 1]
    for cursor in range(start, end):
        tet = incidence_tets[cursor]
        corner = incidence_corners[cursor]
        value += volumes[tet] * (delta_piola[tet] * shape_gradients[4 * tet + corner])
    output[free_index] = value


@wp.kernel(enable_backward=False)
def _gather_block_diagonal(
    mass: wp.array[wp.float64],
    free: wp.array[int],
    incidence_offsets: wp.array[int],
    incidence_tets: wp.array[int],
    incidence_corners: wp.array[int],
    shape_gradients: wp.array[wp.vec3d],
    volumes: wp.array[wp.float64],
    cofactors: wp.array[wp.mat33d],
    mu: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    inverse_dt_squared: wp.float64,
    output: wp.array[wp.mat33d],
):
    free_index = wp.tid()
    vertex = free[free_index]
    block = _scaled_identity(mass[vertex] * inverse_dt_squared)
    start = incidence_offsets[free_index]
    end = incidence_offsets[free_index + 1]
    for cursor in range(start, end):
        tet = incidence_tets[cursor]
        corner = incidence_corners[cursor]
        shape = shape_gradients[4 * tet + corner]
        cofactor_shape = cofactors[tet] * shape
        local_identity = mu[tet] * wp.dot(shape, shape)
        block += volumes[tet] * (_scaled_identity(local_identity) + lam[tet] * _outer3(cofactor_shape, cofactor_shape))
    output[free_index] = block


@wp.kernel(enable_backward=False)
def _invert_block_diagonal(
    blocks: wp.array[wp.mat33d],
    inverse: wp.array[wp.mat33d],
    valid: wp.array[int],
):
    index = wp.tid()
    block = blocks[index]
    leading_one = block[0, 0]
    leading_two = block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0]
    determinant = _determinant3(block)
    if (
        _finite_mat33(block)
        and wp.isfinite(leading_two)
        and wp.isfinite(determinant)
        and leading_one > wp.float64(0.0)
        and leading_two > wp.float64(0.0)
        and determinant > wp.float64(0.0)
    ):
        inverse[index] = _inverse_spd3(block)
        valid[index] = 1
    else:
        inverse[index] = _zero_matrix()
        valid[index] = 0


_STATUS_ACTIVE = 0
_STATUS_COMPLETED = 1
_STATUS_ZERO_RHS = 2
_STATUS_CONVERGED = 3
_STATUS_NONFINITE_RHS = 4
_STATUS_INVALID_PRECONDITIONER = 5
_STATUS_NONFINITE_PRECONDITIONER = 6
_STATUS_NONPOSITIVE_PRECONDITIONER = 7
_STATUS_NONFINITE_OPERATOR = 8
_STATUS_NONPOSITIVE_CURVATURE = 9
_STATUS_NONFINITE_UPDATE = 10
_STATUS_NONFINITE_TRUE_RESIDUAL = 11

_STATUS_NAMES = {
    _STATUS_ACTIVE: "active",
    _STATUS_COMPLETED: "completed",
    _STATUS_ZERO_RHS: "zero_rhs",
    _STATUS_CONVERGED: "converged",
    _STATUS_NONFINITE_RHS: "nonfinite_rhs",
    _STATUS_INVALID_PRECONDITIONER: "invalid_preconditioner",
    _STATUS_NONFINITE_PRECONDITIONER: "nonfinite_preconditioner",
    _STATUS_NONPOSITIVE_PRECONDITIONER: "nonpositive_preconditioner",
    _STATUS_NONFINITE_OPERATOR: "nonfinite_operator",
    _STATUS_NONPOSITIVE_CURVATURE: "nonpositive_curvature",
    _STATUS_NONFINITE_UPDATE: "nonfinite_update",
    _STATUS_NONFINITE_TRUE_RESIDUAL: "nonfinite_true_residual",
}


@wp.kernel(enable_backward=False)
def _clear_trace(
    curvature: wp.array[wp.float64],
    step_size: wp.array[wp.float64],
    residual_squared: wp.array[wp.float64],
    conjugacy: wp.array[wp.float64],
    status: wp.array[int],
):
    index = wp.tid()
    curvature[index] = wp.float64(0.0)
    step_size[index] = wp.float64(0.0)
    residual_squared[index] = wp.float64(0.0)
    conjugacy[index] = wp.float64(0.0)
    status[index] = _STATUS_ACTIVE


@wp.kernel(enable_backward=False)
def _initialize_pcg_vectors(
    rhs: wp.array[wp.vec3d],
    preconditioner_inverse: wp.array[wp.mat33d],
    solution: wp.array[wp.vec3d],
    residual: wp.array[wp.vec3d],
    preconditioned: wp.array[wp.vec3d],
    direction: wp.array[wp.vec3d],
    preconditioner_valid: wp.array[int],
    direction_valid: wp.array[int],
):
    index = wp.tid()
    value = rhs[index]
    if not _finite_vec3(value):
        value = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    transformed = preconditioner_inverse[index] * value
    transformed_valid = _finite_vec3(transformed)
    if not transformed_valid:
        transformed = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    solution[index] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    residual[index] = value
    preconditioned[index] = transformed
    direction[index] = transformed
    preconditioner_valid[index] = int(transformed_valid)
    direction_valid[index] = int(transformed_valid)


@wp.kernel(enable_backward=False)
def _initialize_unpreconditioned_pcg_vectors(
    rhs: wp.array[wp.vec3d],
    solution: wp.array[wp.vec3d],
    residual: wp.array[wp.vec3d],
    preconditioned: wp.array[wp.vec3d],
    direction: wp.array[wp.vec3d],
):
    index = wp.tid()
    value = rhs[index]
    if not _finite_vec3(value):
        value = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    zero = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    solution[index] = zero
    residual[index] = value
    preconditioned[index] = zero
    direction[index] = zero


@wp.kernel(enable_backward=False)
def _validate_initial_device_preconditioner(
    preconditioned: wp.array[wp.vec3d],
    preconditioner_valid: wp.array[int],
):
    index = wp.tid()
    value = preconditioned[index]
    valid = _finite_vec3(value)
    if not valid:
        preconditioned[index] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    preconditioner_valid[index] = int(valid)


@wp.kernel(enable_backward=False)
def _initialize_direction_from_preconditioner(
    preconditioned: wp.array[wp.vec3d],
    preconditioner_valid: wp.array[int],
    direction: wp.array[wp.vec3d],
    direction_valid: wp.array[int],
):
    index = wp.tid()
    direction[index] = preconditioned[index]
    direction_valid[index] = preconditioner_valid[index]


@wp.kernel(enable_backward=False)
def _validate_device_preconditioner(
    state_status: wp.array[int],
    preconditioned: wp.array[wp.vec3d],
    preconditioner_valid: wp.array[int],
):
    index = wp.tid()
    if state_status[0] != _STATUS_ACTIVE:
        preconditioned[index] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        preconditioner_valid[index] = 1
        return
    value = preconditioned[index]
    valid = _finite_vec3(value)
    if not valid:
        preconditioned[index] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    preconditioner_valid[index] = int(valid)


@wp.kernel(enable_backward=False)
def _initialize_pcg_state(
    rhs: wp.array[wp.vec3d],
    residual: wp.array[wp.vec3d],
    preconditioned: wp.array[wp.vec3d],
    block_valid: wp.array[int],
    preconditioner_valid: wp.array[int],
    state_status: wp.array[int],
    completed_iterations: wp.array[int],
    rho: wp.array[wp.float64],
    rhs_squared: wp.array[wp.float64],
    recursive_residual_squared: wp.array[wp.float64],
    true_residual_squared: wp.array[wp.float64],
):
    if wp.tid() != 0:
        return
    status = int(_STATUS_ACTIVE)
    rhs_norm_squared = wp.float64(0.0)
    rho_value = wp.float64(0.0)
    count = rhs.shape[0]
    for index in range(count):
        raw_rhs = rhs[index]
        if not _finite_vec3(raw_rhs):
            status = _STATUS_NONFINITE_RHS
        if block_valid[index] == 0 and status == _STATUS_ACTIVE:
            status = _STATUS_INVALID_PRECONDITIONER
        if preconditioner_valid[index] == 0 and status == _STATUS_ACTIVE:
            status = _STATUS_NONFINITE_PRECONDITIONER
        rhs_norm_squared += wp.dot(raw_rhs, raw_rhs)
        rho_value += wp.dot(residual[index], preconditioned[index])
    if status == _STATUS_ACTIVE and (not wp.isfinite(rhs_norm_squared) or not wp.isfinite(rho_value)):
        status = _STATUS_NONFINITE_RHS
    if status == _STATUS_ACTIVE and rhs_norm_squared == wp.float64(0.0):
        status = _STATUS_ZERO_RHS
    if status == _STATUS_ACTIVE and rho_value <= wp.float64(0.0):
        status = _STATUS_NONPOSITIVE_PRECONDITIONER
    state_status[0] = status
    completed_iterations[0] = 0
    rho[0] = rho_value
    rhs_squared[0] = rhs_norm_squared
    recursive_residual_squared[0] = rhs_norm_squared
    true_residual_squared[0] = rhs_norm_squared


@wp.kernel(enable_backward=False)
def _compute_pcg_step(
    direction: wp.array[wp.vec3d],
    operator_direction: wp.array[wp.vec3d],
    direction_valid: wp.array[int],
    state_status: wp.array[int],
    rho: wp.array[wp.float64],
    iteration: int,
    trace_curvature: wp.array[wp.float64],
    trace_step_size: wp.array[wp.float64],
    trace_status: wp.array[int],
):
    if wp.tid() != 0:
        return
    status = state_status[0]
    if status != _STATUS_ACTIVE:
        trace_status[iteration] = status
        return
    curvature = wp.float64(0.0)
    count = direction.shape[0]
    for index in range(count):
        if direction_valid[index] == 0:
            status = _STATUS_NONFINITE_UPDATE
        left = direction[index]
        right = operator_direction[index]
        if not _finite_vec3(left) or not _finite_vec3(right):
            status = _STATUS_NONFINITE_OPERATOR
        curvature += wp.dot(left, right)
    trace_curvature[iteration] = curvature
    if status == _STATUS_ACTIVE and not wp.isfinite(curvature):
        status = _STATUS_NONFINITE_OPERATOR
    if status == _STATUS_ACTIVE and curvature <= wp.float64(0.0):
        status = _STATUS_NONPOSITIVE_CURVATURE
    if status == _STATUS_ACTIVE:
        step = rho[0] / curvature
        if not wp.isfinite(step) or step <= wp.float64(0.0):
            status = _STATUS_NONFINITE_UPDATE
        else:
            trace_step_size[iteration] = step
    state_status[0] = status
    trace_status[iteration] = status


@wp.kernel(enable_backward=False)
def _update_solution_residual(
    direction: wp.array[wp.vec3d],
    operator_direction: wp.array[wp.vec3d],
    state_status: wp.array[int],
    iteration: int,
    trace_step_size: wp.array[wp.float64],
    solution: wp.array[wp.vec3d],
    residual: wp.array[wp.vec3d],
    update_valid: wp.array[int],
):
    index = wp.tid()
    if state_status[0] != _STATUS_ACTIVE:
        update_valid[index] = 1
        return
    step = trace_step_size[iteration]
    candidate_solution = solution[index] + step * direction[index]
    candidate_residual = residual[index] - step * operator_direction[index]
    valid = _finite_vec3(candidate_solution) and _finite_vec3(candidate_residual)
    if valid:
        solution[index] = candidate_solution
        residual[index] = candidate_residual
    update_valid[index] = int(valid)


@wp.kernel(enable_backward=False)
def _apply_block_preconditioner(
    residual: wp.array[wp.vec3d],
    preconditioner_inverse: wp.array[wp.mat33d],
    state_status: wp.array[int],
    preconditioned: wp.array[wp.vec3d],
    preconditioner_valid: wp.array[int],
):
    index = wp.tid()
    if state_status[0] != _STATUS_ACTIVE:
        preconditioned[index] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        preconditioner_valid[index] = 1
        return
    value = preconditioner_inverse[index] * residual[index]
    valid = _finite_vec3(value)
    if not valid:
        value = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    preconditioned[index] = value
    preconditioner_valid[index] = int(valid)


@wp.kernel(enable_backward=False)
def _compute_pcg_conjugacy(
    solution: wp.array[wp.vec3d],
    residual: wp.array[wp.vec3d],
    preconditioned: wp.array[wp.vec3d],
    update_valid: wp.array[int],
    preconditioner_valid: wp.array[int],
    state_status: wp.array[int],
    completed_iterations: wp.array[int],
    rho: wp.array[wp.float64],
    recursive_residual_squared: wp.array[wp.float64],
    iteration: int,
    trace_residual_squared: wp.array[wp.float64],
    trace_conjugacy: wp.array[wp.float64],
    trace_status: wp.array[int],
):
    if wp.tid() != 0:
        return
    status = state_status[0]
    if status != _STATUS_ACTIVE:
        trace_status[iteration] = status
        return
    residual_norm_squared = wp.float64(0.0)
    count = residual.shape[0]
    for index in range(count):
        if update_valid[index] == 0:
            status = _STATUS_NONFINITE_UPDATE
        if not _finite_vec3(solution[index]) or not _finite_vec3(residual[index]):
            status = _STATUS_NONFINITE_UPDATE
        residual_norm_squared += wp.dot(residual[index], residual[index])
    trace_residual_squared[iteration] = residual_norm_squared
    recursive_residual_squared[0] = residual_norm_squared
    if status == _STATUS_ACTIVE and not wp.isfinite(residual_norm_squared):
        status = _STATUS_NONFINITE_UPDATE
    if status == _STATUS_ACTIVE:
        # The x/r update is complete before the next preconditioner is
        # consumed. Preserve that completed work even if this application or
        # its subsequent r.T z reduction fails.
        completed_iterations[0] = completed_iterations[0] + 1
        if residual_norm_squared == wp.float64(0.0):
            status = _STATUS_CONVERGED
            rho[0] = wp.float64(0.0)
        else:
            conjugacy = wp.float64(0.0)
            for index in range(count):
                if preconditioner_valid[index] == 0 or not _finite_vec3(preconditioned[index]):
                    status = _STATUS_NONFINITE_PRECONDITIONER
            rho_new = wp.float64(0.0)
            if status == _STATUS_ACTIVE:
                for index in range(count):
                    rho_new += wp.dot(residual[index], preconditioned[index])
            if status == _STATUS_ACTIVE and not wp.isfinite(rho_new):
                status = _STATUS_NONFINITE_UPDATE
            if status == _STATUS_ACTIVE and rho_new <= wp.float64(0.0):
                status = _STATUS_NONPOSITIVE_PRECONDITIONER
            if status == _STATUS_ACTIVE:
                conjugacy = rho_new / rho[0]
                if not wp.isfinite(conjugacy) or conjugacy < wp.float64(0.0):
                    status = _STATUS_NONFINITE_UPDATE
            if status == _STATUS_ACTIVE:
                trace_conjugacy[iteration] = conjugacy
                rho[0] = rho_new
    state_status[0] = status
    trace_status[iteration] = status


@wp.kernel(enable_backward=False)
def _finalize_pcg_iteration(
    solution: wp.array[wp.vec3d],
    residual: wp.array[wp.vec3d],
    update_valid: wp.array[int],
    state_status: wp.array[int],
    completed_iterations: wp.array[int],
    recursive_residual_squared: wp.array[wp.float64],
    iteration: int,
    trace_residual_squared: wp.array[wp.float64],
    trace_status: wp.array[int],
):
    if wp.tid() != 0:
        return
    status = state_status[0]
    if status != _STATUS_ACTIVE:
        trace_status[iteration] = status
        return
    residual_norm_squared = wp.float64(0.0)
    count = residual.shape[0]
    for index in range(count):
        if update_valid[index] == 0:
            status = _STATUS_NONFINITE_UPDATE
        if not _finite_vec3(solution[index]) or not _finite_vec3(residual[index]):
            status = _STATUS_NONFINITE_UPDATE
        residual_norm_squared += wp.dot(residual[index], residual[index])
    trace_residual_squared[iteration] = residual_norm_squared
    recursive_residual_squared[0] = residual_norm_squared
    if status == _STATUS_ACTIVE and not wp.isfinite(residual_norm_squared):
        status = _STATUS_NONFINITE_UPDATE
    if status == _STATUS_ACTIVE:
        completed_iterations[0] = completed_iterations[0] + 1
        status = _STATUS_COMPLETED
    state_status[0] = status
    trace_status[iteration] = status


@wp.kernel(enable_backward=False)
def _update_pcg_direction(
    preconditioned: wp.array[wp.vec3d],
    state_status: wp.array[int],
    iteration: int,
    trace_conjugacy: wp.array[wp.float64],
    direction: wp.array[wp.vec3d],
    direction_valid: wp.array[int],
):
    index = wp.tid()
    if state_status[0] != _STATUS_ACTIVE:
        direction[index] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        direction_valid[index] = 1
        return
    value = preconditioned[index] + trace_conjugacy[iteration] * direction[index]
    valid = _finite_vec3(value)
    if not valid:
        value = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    direction[index] = value
    direction_valid[index] = int(valid)


@wp.kernel(enable_backward=False)
def _verify_true_residual(
    rhs: wp.array[wp.vec3d],
    operator_solution: wp.array[wp.vec3d],
    state_status: wp.array[int],
    true_residual_squared: wp.array[wp.float64],
):
    if wp.tid() != 0:
        return
    value = wp.float64(0.0)
    finite = bool(True)
    count = rhs.shape[0]
    for index in range(count):
        residual = rhs[index] - operator_solution[index]
        finite = finite and _finite_vec3(residual)
        value += wp.dot(residual, residual)
    if not finite or not wp.isfinite(value):
        if (
            state_status[0] == _STATUS_ACTIVE
            or state_status[0] == _STATUS_COMPLETED
            or state_status[0] == _STATUS_ZERO_RHS
            or state_status[0] == _STATUS_CONVERGED
        ):
            state_status[0] = _STATUS_NONFINITE_TRUE_RESIDUAL
        true_residual_squared[0] = wp.float64(-1.0)
    else:
        true_residual_squared[0] = value
        if state_status[0] == _STATUS_ACTIVE:
            state_status[0] = _STATUS_COMPLETED


def _immutable_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Return a C-contiguous NumPy array backed by immutable bytes."""
    owned = np.asarray(value, dtype=dtype, order="C")
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def _build_sorted_free_incidence(
    tets: np.ndarray,
    free: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic free-vertex CSR rows sorted by tet and corner."""
    lookup = {int(vertex): free_index for free_index, vertex in enumerate(free)}
    rows: list[list[tuple[int, int]]] = [[] for _ in range(free.size)]
    for tet_index, tet in enumerate(tets):
        for corner, vertex in enumerate(tet):
            free_index = lookup.get(int(vertex))
            if free_index is not None:
                rows[free_index].append((tet_index, corner))
    offsets = np.zeros(free.size + 1, dtype=np.int32)
    entries: list[tuple[int, int]] = []
    for free_index, row in enumerate(rows):
        row.sort()
        entries.extend(row)
        offsets[free_index + 1] = len(entries)
    incidence_tets = np.asarray([entry[0] for entry in entries], dtype=np.int32)
    incidence_corners = np.asarray([entry[1] for entry in entries], dtype=np.int32)
    return offsets, incidence_tets, incidence_corners


class WarpMatrixFreeWorkspace:
    """Persistent temporary storage for one matrix-free operator application."""

    def __init__(self, operator: WarpMatrixFreeStableNHOperator):
        self._operator_identity = id(operator)
        self.delta_piola = wp.empty(operator.n_tets, dtype=wp.mat33d, device=operator.device)


class WarpMatrixFreeStableNHOperator:
    """Frozen device-resident exact-gradient / PSD-Gauss-Newton operator.

    Instances are constructed from the independent NumPy oracle so validation
    and objective conventions have one authoritative source.  Call
    :meth:`launch_refresh_geometry` after changing ``positions`` outside a
    capture.  All Krylov-facing arrays have exactly ``n_free`` entries.
    """

    def __init__(self, oracle: MatrixFreeStableNHOperator, *, device: str = "cpu"):
        if not isinstance(oracle, MatrixFreeStableNHOperator):
            raise TypeError("oracle must be a MatrixFreeStableNHOperator")
        self.device = wp.get_device(device)
        self.n_vertices = oracle.n_vertices
        self.n_tets = int(oracle.tets.shape[0])
        self.n_free = int(oracle.free.size)
        self.n_free_dofs = oracle.n_free_dofs
        self.dt = float(oracle.dt)
        self.inverse_dt_squared = float(1.0 / (oracle.dt * oracle.dt))
        self.current_operator_sha256 = _current_operator_sha256(oracle)

        tets = np.asarray(oracle.tets, dtype=np.int32)
        free = np.asarray(oracle.free, dtype=np.int32)
        vertex_to_free = np.full(self.n_vertices, -1, dtype=np.int32)
        vertex_to_free[free] = np.arange(self.n_free, dtype=np.int32)
        offsets, incidence_tets, incidence_corners = _build_sorted_free_incidence(tets, free)
        self.incidence_offsets_host = _immutable_array(offsets, np.dtype(np.int32))
        self.incidence_tets_host = _immutable_array(incidence_tets, np.dtype(np.int32))
        self.incidence_corners_host = _immutable_array(incidence_corners, np.dtype(np.int32))
        self.free_host = _immutable_array(free, np.dtype(np.int32))
        self.vertex_to_free_host = _immutable_array(vertex_to_free, np.dtype(np.int32))

        self.positions = wp.array(oracle.positions, dtype=wp.vec3d, device=self.device)
        self.tets = wp.array(tets.reshape(-1), dtype=wp.int32, device=self.device)
        self.shape_gradients = wp.array(
            np.asarray(oracle.shape_gradients).reshape(-1, 3), dtype=wp.vec3d, device=self.device
        )
        self.volumes = wp.array(oracle.volumes, dtype=wp.float64, device=self.device)
        self.mass = wp.array(oracle.mass, dtype=wp.float64, device=self.device)
        self.mu = wp.array(oracle.mu, dtype=wp.float64, device=self.device)
        self.lam = wp.array(oracle.lam, dtype=wp.float64, device=self.device)
        self.inertial_target = wp.array(oracle.inertial_target, dtype=wp.vec3d, device=self.device)
        self.free = wp.array(free, dtype=wp.int32, device=self.device)
        self.vertex_to_free = wp.array(vertex_to_free, dtype=wp.int32, device=self.device)
        self.incidence_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)
        self.incidence_tets = wp.array(incidence_tets, dtype=wp.int32, device=self.device)
        self.incidence_corners = wp.array(incidence_corners, dtype=wp.int32, device=self.device)
        self.deformation_gradients = wp.empty(self.n_tets, dtype=wp.mat33d, device=self.device)
        self.cofactors = wp.empty(self.n_tets, dtype=wp.mat33d, device=self.device)
        self.determinants = wp.empty(self.n_tets, dtype=wp.float64, device=self.device)
        self.first_piola = wp.empty(self.n_tets, dtype=wp.mat33d, device=self.device)
        self.launch_refresh_geometry()

    @classmethod
    def from_oracle(
        cls,
        oracle: MatrixFreeStableNHOperator,
        *,
        device: str = "cpu",
    ) -> WarpMatrixFreeStableNHOperator:
        """Create a device snapshot from the validated NumPy oracle."""
        return cls(oracle, device=device)

    def create_apply_workspace(self) -> WarpMatrixFreeWorkspace:
        """Allocate reusable tet-local operator storage."""
        return WarpMatrixFreeWorkspace(self)

    def record_current_operator_sha256(self) -> str:
        """Synchronously hash the exact device arrays defining current ``A``."""
        result = _current_operator_arrays_sha256(
            tets=np.asarray(self.tets.numpy(), dtype=np.int64).reshape(self.n_tets, 4),
            shape_gradients=np.asarray(self.shape_gradients.numpy(), dtype=np.float64).reshape(self.n_tets, 4, 3),
            volumes=np.asarray(self.volumes.numpy(), dtype=np.float64),
            mass=np.asarray(self.mass.numpy(), dtype=np.float64),
            mu=np.asarray(self.mu.numpy(), dtype=np.float64),
            lam=np.asarray(self.lam.numpy(), dtype=np.float64),
            free=np.asarray(self.free.numpy(), dtype=np.int64),
            cofactors=np.asarray(self.cofactors.numpy(), dtype=np.float64),
            dt=self.dt,
        )
        self.current_operator_sha256 = result
        return result

    def _validate_vector(self, vector: wp.array[wp.vec3d], name: str) -> None:
        if vector.device != self.device or vector.dtype != wp.vec3d or vector.shape != (self.n_free,):
            raise ValueError(f"{name} must be a vec3d array of shape ({self.n_free},) on {self.device}")

    def launch_refresh_geometry(self) -> None:
        """Refresh deformation, cofactor, determinant, and exact Piola buffers."""
        wp.launch(
            _evaluate_geometry,
            dim=self.n_tets,
            inputs=[
                self.positions,
                self.tets,
                self.shape_gradients,
                self.mu,
                self.lam,
                self.deformation_gradients,
                self.cofactors,
                self.determinants,
                self.first_piola,
            ],
            device=self.device,
        )

    def launch_gradient(self, output: wp.array[wp.vec3d], *, scale: float = 1.0) -> None:
        """Launch the exact free gradient gather into ``output``."""
        self._validate_vector(output, "output")
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        wp.launch(
            _gather_gradient,
            dim=self.n_free,
            inputs=[
                self.positions,
                self.inertial_target,
                self.mass,
                self.free,
                self.incidence_offsets,
                self.incidence_tets,
                self.incidence_corners,
                self.shape_gradients,
                self.volumes,
                self.first_piola,
                self.inverse_dt_squared,
                float(scale),
                output,
            ],
            device=self.device,
        )

    def launch_apply(
        self,
        direction: wp.array[wp.vec3d],
        output: wp.array[wp.vec3d],
        workspace: WarpMatrixFreeWorkspace,
    ) -> None:
        """Launch one deterministic matrix-free Gauss-Newton product."""
        self._validate_vector(direction, "direction")
        self._validate_vector(output, "output")
        if not isinstance(workspace, WarpMatrixFreeWorkspace) or workspace._operator_identity != id(self):
            raise ValueError("workspace belongs to a different operator")
        wp.launch(
            _apply_tet_operator,
            dim=self.n_tets,
            inputs=[
                direction,
                self.tets,
                self.vertex_to_free,
                self.shape_gradients,
                self.cofactors,
                self.mu,
                self.lam,
                workspace.delta_piola,
            ],
            device=self.device,
        )
        wp.launch(
            _gather_operator_product,
            dim=self.n_free,
            inputs=[
                direction,
                self.mass,
                self.free,
                self.incidence_offsets,
                self.incidence_tets,
                self.incidence_corners,
                self.shape_gradients,
                self.volumes,
                workspace.delta_piola,
                self.inverse_dt_squared,
                output,
            ],
            device=self.device,
        )

    def launch_block_diagonal(self, output: wp.array[wp.mat33d]) -> None:
        """Launch the exact free-vertex 3x3 block-diagonal gather."""
        if output.device != self.device or output.dtype != wp.mat33d or output.shape != (self.n_free,):
            raise ValueError(f"output must be a mat33d array of shape ({self.n_free},) on {self.device}")
        wp.launch(
            _gather_block_diagonal,
            dim=self.n_free,
            inputs=[
                self.mass,
                self.free,
                self.incidence_offsets,
                self.incidence_tets,
                self.incidence_corners,
                self.shape_gradients,
                self.volumes,
                self.cofactors,
                self.mu,
                self.lam,
                self.inverse_dt_squared,
                output,
            ],
            device=self.device,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class WarpDevicePreconditionerApplication:
    """Retained exact evidence for one device preconditioner application."""

    application_index: int
    preconditioner_identity: str
    static_preconditioner_sha256: str
    device_snapshot_sha256: str
    input_sha256: str
    output_sha256: str
    algebraic_work_sha256: str
    rhs_count: int
    level_visits: tuple[int, ...]
    matrix_block_products: int
    smoother_block_solves: int
    restriction_block_products: int
    prolongation_block_products: int
    coarsest_factor_solves: int
    scheduled_kernel_launches: int
    output_finite: bool
    capture_replay: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.application_index, bool)
            or not isinstance(self.application_index, numbers.Integral)
            or self.application_index < 0
        ):
            raise ValueError("application_index must be a non-negative integer")
        if type(self.preconditioner_identity) is not str or not self.preconditioner_identity:
            raise ValueError("preconditioner_identity must be a non-empty exact string")
        for name in (
            "static_preconditioner_sha256",
            "device_snapshot_sha256",
            "input_sha256",
            "output_sha256",
            "algebraic_work_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "rhs_count",
            "matrix_block_products",
            "smoother_block_solves",
            "restriction_block_products",
            "prolongation_block_products",
            "coarsest_factor_solves",
            "scheduled_kernel_launches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.rhs_count < 1 or self.scheduled_kernel_launches < 1:
            raise ValueError("application RHS count and scheduled kernel work must be positive")
        object.__setattr__(self, "level_visits", tuple(self.level_visits))
        if not self.level_visits or any(
            isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0
            for value in self.level_visits
        ):
            raise ValueError("level_visits must contain non-negative integer counts")
        if not isinstance(self.output_finite, bool) or not isinstance(self.capture_replay, bool):
            raise TypeError("device preconditioner flags must be bools")

    def deterministic_record(self) -> dict[str, object]:
        """Serialize immutable application identity and exact work."""
        return dataclasses.asdict(self)


class WarpDevicePreconditioner:
    """Typed allocation-free device preconditioner boundary.

    Implementations own immutable device data and allocate one persistent
    workspace per scheduled application before :meth:`WarpFixedPCGWorkspace.launch`.
    Launch and record methods are deliberately separate: the former may only
    enqueue device work, while the latter is called after replay and may
    synchronize to retain diagnostic evidence.
    """

    device: object
    vector_count: int
    free_vertices_host: np.ndarray
    preconditioner_identity: str
    static_preconditioner_sha256: str
    device_snapshot_sha256: str
    application_kernel_launches: int

    def create_application_workspace(self) -> object:
        """Allocate persistent buffers for one independently retained apply."""
        raise NotImplementedError

    def launch_apply(
        self,
        rhs: wp.array[wp.vec3d],
        output: wp.array[wp.vec3d],
        workspace: object,
    ) -> None:
        """Enqueue one application without allocation or host reads."""
        raise NotImplementedError

    def record_application(
        self,
        application_index: int,
        workspace: object,
        *,
        capture_replay: bool,
    ) -> WarpDevicePreconditionerApplication:
        """Synchronously retain one completed application's exact evidence."""
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class WarpFixedPCGWork:
    """Exact scheduled primitive work for one launcher execution."""

    geometry_evaluations: int
    preconditioner_builds: int
    operator_applications: int
    residual_verification_applications: int
    preconditioner_applications: int
    scalar_reductions: int
    kernel_launches: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{field.name} must be a non-negative integer")


@dataclasses.dataclass(frozen=True)
class WarpFixedPCGIteration:
    """One device trace slot in the fixed PCG schedule."""

    iteration: int
    active_update_completed: bool
    status: str
    residual_norm: float | None
    direction_curvature: float | None
    step_size: float | None
    conjugacy: float | None

    def __post_init__(self) -> None:
        values = (self.residual_norm, self.direction_curvature, self.step_size, self.conjugacy)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("present PCG trace scalars must be finite")
        nonnegative = (self.residual_norm, self.step_size, self.conjugacy)
        if any(value is not None and value < 0.0 for value in nonnegative):
            raise ValueError("present PCG trace norms and coefficients must be non-negative")


@dataclasses.dataclass(frozen=True, eq=False)
class WarpFixedPCGRecord:
    """Post-execution diagnostic record for the research Warp primitive."""

    solution: np.ndarray
    success: bool
    reason: str
    requested_iterations: int
    completed_iterations: int
    rhs_norm: float | None
    recursive_residual_norm: float | None
    true_residual_norm: float | None
    trace: tuple[WarpFixedPCGIteration, ...]
    work: WarpFixedPCGWork
    preconditioner_identity: str
    capture_replay: bool
    current_operator_sha256: str
    static_preconditioner_sha256: str | None
    operator_preconditioner_binding_sha256: str
    preconditioner_evidence: tuple[WarpDevicePreconditionerApplication, ...]
    contract_id: str = CONTRACT_ID
    research_only: bool = True
    performance_evidence: bool = False

    def __post_init__(self) -> None:
        solution = _immutable_array(np.asarray(self.solution, dtype=np.float64), np.dtype(np.float64))
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "trace", tuple(self.trace))
        object.__setattr__(self, "preconditioner_evidence", tuple(self.preconditioner_evidence))
        if self.reason not in set(_STATUS_NAMES.values()) - {"active"}:
            raise ValueError(f"unknown terminal PCG reason: {self.reason}")
        if self.requested_iterations < 1 or not 0 <= self.completed_iterations <= self.requested_iterations:
            raise ValueError("PCG iteration counts are inconsistent")
        if len(self.trace) != self.requested_iterations:
            raise ValueError("fixed PCG trace must contain every scheduled iteration")
        if self.success != (self.reason in ("completed", "zero_rhs", "converged")):
            raise ValueError("success flag disagrees with terminal reason")
        if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in self.norms):
            raise ValueError("present PCG norms must be finite and non-negative")
        if self.success and any(value is None for value in self.norms):
            raise ValueError("successful PCG must record all finite norms")
        if type(self.preconditioner_identity) is not str or not self.preconditioner_identity:
            raise ValueError("preconditioner_identity must be a non-empty exact string")
        for name in ("current_operator_sha256", "operator_preconditioner_binding_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.static_preconditioner_sha256 is not None:
            value = self.static_preconditioner_sha256
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("static_preconditioner_sha256 must be a lowercase SHA-256 digest")
            if len(self.preconditioner_evidence) != self.work.preconditioner_applications:
                raise ValueError("device preconditioner evidence must retain every scheduled application")
        elif self.preconditioner_evidence:
            raise ValueError("block preconditioners cannot carry device application evidence")
        for application_index, evidence in enumerate(self.preconditioner_evidence):
            if evidence.application_index != application_index:
                raise ValueError("device preconditioner evidence indices must be contiguous")
            if evidence.preconditioner_identity != self.preconditioner_identity:
                raise ValueError("device application and PCG preconditioner identities disagree")
            if evidence.static_preconditioner_sha256 != self.static_preconditioner_sha256:
                raise ValueError("device application and PCG static identities disagree")
        expected_binding = _operator_preconditioner_binding_sha256(
            self.current_operator_sha256,
            self.preconditioner_identity,
            self.static_preconditioner_sha256,
        )
        if self.operator_preconditioner_binding_sha256 != expected_binding:
            raise ValueError("operator_preconditioner_binding_sha256 does not bind the recorded identities")
        if not isinstance(self.capture_replay, bool):
            raise TypeError("capture_replay must be a bool")
        if not self.research_only or self.performance_evidence:
            raise ValueError("this research primitive cannot claim performance evidence")

    @property
    def norms(self) -> tuple[float | None, float | None, float | None]:
        """Recorded RHS, recursive-residual, and true-residual norms."""
        return self.rhs_norm, self.recursive_residual_norm, self.true_residual_norm

    def deterministic_record(self) -> dict[str, object]:
        """Serialize deterministic status and work without timing claims."""
        return {
            "contract_id": self.contract_id,
            "research_only": self.research_only,
            "performance_evidence": self.performance_evidence,
            "capture_replay": self.capture_replay,
            "success": self.success,
            "reason": self.reason,
            "preconditioner_identity": self.preconditioner_identity,
            "current_operator_sha256": self.current_operator_sha256,
            "static_preconditioner_sha256": self.static_preconditioner_sha256,
            "operator_preconditioner_binding_sha256": self.operator_preconditioner_binding_sha256,
            "requested_iterations": self.requested_iterations,
            "completed_iterations": self.completed_iterations,
            "rhs_norm": self.rhs_norm,
            "recursive_residual_norm": self.recursive_residual_norm,
            "true_residual_norm": self.true_residual_norm,
            "work": dataclasses.asdict(self.work),
            "preconditioner_evidence": [item.deterministic_record() for item in self.preconditioner_evidence],
            "trace": [dataclasses.asdict(item) for item in self.trace],
        }


class WarpFixedPCGWorkspace:
    """Persistent fixed-buffer, fixed-launch PCG schedule.

    :meth:`launch` performs no allocations, synchronization, or device-to-host
    reads.  Every call schedules all requested iterations even after a
    device-side failure or exact convergence.  :meth:`record` is the explicit
    post-execution synchronization point and must stay outside graph capture.
    The legacy block path retains its ``K + 1`` masked applications. Typed
    device preconditioners schedule exactly ``K`` applications: one initial
    application and one after each nonfinal update.  Each device application
    has an independent persistent workspace so its evidence is retained.
    Neither schedule is a GPU-performance claim.
    """

    def __init__(
        self,
        operator: WarpMatrixFreeStableNHOperator,
        iterations: int,
        *,
        external_preconditioner_inverse: np.ndarray | Sequence[Sequence[Sequence[float]]] | None = None,
        preconditioner_identity: str | None = None,
        device_preconditioner: WarpDevicePreconditioner | None = None,
    ):
        if not isinstance(operator, WarpMatrixFreeStableNHOperator):
            raise TypeError("operator must be a WarpMatrixFreeStableNHOperator")
        if isinstance(iterations, bool) or not isinstance(iterations, numbers.Integral) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self.operator = operator
        self.iterations = int(iterations)
        self.apply_workspace = operator.create_apply_workspace()
        device = operator.device
        count = operator.n_free
        self.rhs = wp.empty(count, dtype=wp.vec3d, device=device)
        self.solution = wp.empty(count, dtype=wp.vec3d, device=device)
        self.residual = wp.empty(count, dtype=wp.vec3d, device=device)
        self.preconditioned = wp.empty(count, dtype=wp.vec3d, device=device)
        self.direction = wp.empty(count, dtype=wp.vec3d, device=device)
        self.operator_direction = wp.empty(count, dtype=wp.vec3d, device=device)
        self.operator_solution = wp.empty(count, dtype=wp.vec3d, device=device)
        self.block_diagonal = wp.empty(count, dtype=wp.mat33d, device=device)
        self.preconditioner_inverse = wp.empty(count, dtype=wp.mat33d, device=device)
        self.block_valid = wp.empty(count, dtype=wp.int32, device=device)
        self.preconditioner_valid = wp.empty(count, dtype=wp.int32, device=device)
        self.update_valid = wp.empty(count, dtype=wp.int32, device=device)
        self.direction_valid = wp.empty(count, dtype=wp.int32, device=device)
        self.state_status = wp.empty(1, dtype=wp.int32, device=device)
        self.completed_iterations = wp.empty(1, dtype=wp.int32, device=device)
        self.rho = wp.empty(1, dtype=wp.float64, device=device)
        self.rhs_squared = wp.empty(1, dtype=wp.float64, device=device)
        self.recursive_residual_squared = wp.empty(1, dtype=wp.float64, device=device)
        self.true_residual_squared = wp.empty(1, dtype=wp.float64, device=device)
        self.trace_curvature = wp.empty(self.iterations, dtype=wp.float64, device=device)
        self.trace_step_size = wp.empty(self.iterations, dtype=wp.float64, device=device)
        self.trace_residual_squared = wp.empty(self.iterations, dtype=wp.float64, device=device)
        self.trace_conjugacy = wp.empty(self.iterations, dtype=wp.float64, device=device)
        self.trace_status = wp.empty(self.iterations, dtype=wp.int32, device=device)

        if device_preconditioner is not None and external_preconditioner_inverse is not None:
            raise ValueError("device_preconditioner and external_preconditioner_inverse are mutually exclusive")
        if device_preconditioner is not None:
            if not isinstance(device_preconditioner, WarpDevicePreconditioner):
                raise TypeError("device_preconditioner must implement WarpDevicePreconditioner")
            if preconditioner_identity not in (None, device_preconditioner.preconditioner_identity):
                raise ValueError("a device preconditioner cannot be relabelled")
            if device_preconditioner.device != device:
                raise ValueError("device preconditioner and operator must reside on the same device")
            if device_preconditioner.vector_count != count:
                raise ValueError("device preconditioner size does not match the free operator")
            if not np.array_equal(device_preconditioner.free_vertices_host, operator.free_host):
                raise ValueError("device preconditioner free-vertex order does not match the operator")
            if device_preconditioner.application_kernel_launches < 1:
                raise ValueError("device preconditioner application work must be positive")
            self.device_preconditioner = device_preconditioner
            self.device_preconditioner_workspaces = tuple(
                device_preconditioner.create_application_workspace() for _ in range(self.iterations)
            )
            self._build_preconditioner = False
            self.preconditioner_identity = device_preconditioner.preconditioner_identity
            self.static_preconditioner_sha256 = device_preconditioner.static_preconditioner_sha256
            self.block_valid.assign(np.ones(count, dtype=np.int32))
        else:
            self.device_preconditioner = None
            self.device_preconditioner_workspaces = ()
            self._build_preconditioner = external_preconditioner_inverse is None
            self.static_preconditioner_sha256 = None
        if self.device_preconditioner is None and self._build_preconditioner:
            if preconditioner_identity not in (None, "block-jacobi-3x3-warp-v1"):
                raise ValueError("the built block-Jacobi preconditioner cannot be relabelled")
            self.preconditioner_identity = "block-jacobi-3x3-warp-v1"
        elif self.device_preconditioner is None:
            if type(preconditioner_identity) is not str or not preconditioner_identity:
                raise ValueError("an external preconditioner requires a non-empty exact identity")
            inverse = np.asarray(external_preconditioner_inverse, dtype=np.float64)
            if inverse.shape != (count, 3, 3) or not np.isfinite(inverse).all():
                raise ValueError(f"external_preconditioner_inverse must have finite shape ({count}, 3, 3)")
            if not np.array_equal(inverse, np.swapaxes(inverse, 1, 2)):
                raise ValueError("external_preconditioner_inverse must be exactly symmetric")
            for block_index, block in enumerate(inverse):
                try:
                    np.linalg.cholesky(block)
                except np.linalg.LinAlgError as error:
                    raise ValueError(
                        f"external_preconditioner_inverse block {block_index} must be positive definite"
                    ) from error
            self.preconditioner_inverse.assign(inverse)
            self.block_valid.assign(np.ones(count, dtype=np.int32))
            self.preconditioner_identity = preconditioner_identity

    @property
    def work(self) -> WarpFixedPCGWork:
        """Exact scheduled work, including masked launches after failure."""
        if self.device_preconditioner is not None:
            preconditioner_launches = self.device_preconditioner.application_kernel_launches
            # Initial vector/trace/state setup contributes five kernels around
            # the first preconditioner apply. Each nonfinal PCG update adds
            # seven non-preconditioner kernels, the final update adds five,
            # and true-residual verification adds three.
            kernel_launches = preconditioner_launches + 13 + (self.iterations - 1) * (preconditioner_launches + 7)
            return WarpFixedPCGWork(
                geometry_evaluations=0,
                preconditioner_builds=0,
                operator_applications=self.iterations + 1,
                residual_verification_applications=1,
                preconditioner_applications=self.iterations,
                scalar_reductions=2 * self.iterations + 2,
                kernel_launches=kernel_launches,
            )
        preconditioner_build_launches = 2 if self._build_preconditioner else 0
        return WarpFixedPCGWork(
            geometry_evaluations=0,
            preconditioner_builds=int(self._build_preconditioner),
            operator_applications=self.iterations + 1,
            residual_verification_applications=1,
            preconditioner_applications=self.iterations + 1,
            scalar_reductions=2 * self.iterations + 2,
            kernel_launches=preconditioner_build_launches + 7 * self.iterations + 6,
        )

    def set_rhs(self, rhs: np.ndarray | Sequence[float]) -> None:
        """Copy a finite or nonfinite host RHS into the persistent device buffer."""
        values = np.asarray(rhs, dtype=np.float64)
        if values.shape not in ((self.operator.n_free, 3), (self.operator.n_free_dofs,)):
            raise ValueError(
                f"rhs must have shape ({self.operator.n_free}, 3) or ({self.operator.n_free_dofs},), got {values.shape}"
            )
        self.rhs.assign(values.reshape(-1, 3))

    def launch(self) -> None:
        """Launch the complete allocation-free fixed PCG schedule."""
        if self.device_preconditioner is not None:
            self._launch_device_preconditioned()
            return
        device = self.operator.device
        count = self.operator.n_free
        if self._build_preconditioner:
            self.operator.launch_block_diagonal(self.block_diagonal)
            wp.launch(
                _invert_block_diagonal,
                dim=count,
                inputs=[self.block_diagonal, self.preconditioner_inverse, self.block_valid],
                device=device,
            )
        wp.launch(
            _clear_trace,
            dim=self.iterations,
            inputs=[
                self.trace_curvature,
                self.trace_step_size,
                self.trace_residual_squared,
                self.trace_conjugacy,
                self.trace_status,
            ],
            device=device,
        )
        wp.launch(
            _initialize_pcg_vectors,
            dim=count,
            inputs=[
                self.rhs,
                self.preconditioner_inverse,
                self.solution,
                self.residual,
                self.preconditioned,
                self.direction,
                self.preconditioner_valid,
                self.direction_valid,
            ],
            device=device,
        )
        wp.launch(
            _initialize_pcg_state,
            dim=1,
            inputs=[
                self.rhs,
                self.residual,
                self.preconditioned,
                self.block_valid,
                self.preconditioner_valid,
                self.state_status,
                self.completed_iterations,
                self.rho,
                self.rhs_squared,
                self.recursive_residual_squared,
                self.true_residual_squared,
            ],
            device=device,
        )
        for iteration in range(self.iterations):
            self.operator.launch_apply(self.direction, self.operator_direction, self.apply_workspace)
            wp.launch(
                _compute_pcg_step,
                dim=1,
                inputs=[
                    self.direction,
                    self.operator_direction,
                    self.direction_valid,
                    self.state_status,
                    self.rho,
                    iteration,
                    self.trace_curvature,
                    self.trace_step_size,
                    self.trace_status,
                ],
                device=device,
            )
            wp.launch(
                _update_solution_residual,
                dim=count,
                inputs=[
                    self.direction,
                    self.operator_direction,
                    self.state_status,
                    iteration,
                    self.trace_step_size,
                    self.solution,
                    self.residual,
                    self.update_valid,
                ],
                device=device,
            )
            wp.launch(
                _apply_block_preconditioner,
                dim=count,
                inputs=[
                    self.residual,
                    self.preconditioner_inverse,
                    self.state_status,
                    self.preconditioned,
                    self.preconditioner_valid,
                ],
                device=device,
            )
            wp.launch(
                _compute_pcg_conjugacy,
                dim=1,
                inputs=[
                    self.solution,
                    self.residual,
                    self.preconditioned,
                    self.update_valid,
                    self.preconditioner_valid,
                    self.state_status,
                    self.completed_iterations,
                    self.rho,
                    self.recursive_residual_squared,
                    iteration,
                    self.trace_residual_squared,
                    self.trace_conjugacy,
                    self.trace_status,
                ],
                device=device,
            )
            wp.launch(
                _update_pcg_direction,
                dim=count,
                inputs=[
                    self.preconditioned,
                    self.state_status,
                    iteration,
                    self.trace_conjugacy,
                    self.direction,
                    self.direction_valid,
                ],
                device=device,
            )
        self.operator.launch_apply(self.solution, self.operator_solution, self.apply_workspace)
        wp.launch(
            _verify_true_residual,
            dim=1,
            inputs=[self.rhs, self.operator_solution, self.state_status, self.true_residual_squared],
            device=device,
        )

    def _launch_device_preconditioned(self) -> None:
        """Launch fixed PCG with exactly one typed device apply per Krylov step."""
        preconditioner = self.device_preconditioner
        if preconditioner is None:
            raise RuntimeError("typed device preconditioner is not configured")
        device = self.operator.device
        count = self.operator.n_free
        wp.launch(
            _clear_trace,
            dim=self.iterations,
            inputs=[
                self.trace_curvature,
                self.trace_step_size,
                self.trace_residual_squared,
                self.trace_conjugacy,
                self.trace_status,
            ],
            device=device,
        )
        wp.launch(
            _initialize_unpreconditioned_pcg_vectors,
            dim=count,
            inputs=[self.rhs, self.solution, self.residual, self.preconditioned, self.direction],
            device=device,
        )
        preconditioner.launch_apply(
            self.residual,
            self.preconditioned,
            self.device_preconditioner_workspaces[0],
        )
        wp.launch(
            _validate_initial_device_preconditioner,
            dim=count,
            inputs=[self.preconditioned, self.preconditioner_valid],
            device=device,
        )
        wp.launch(
            _initialize_direction_from_preconditioner,
            dim=count,
            inputs=[self.preconditioned, self.preconditioner_valid, self.direction, self.direction_valid],
            device=device,
        )
        wp.launch(
            _initialize_pcg_state,
            dim=1,
            inputs=[
                self.rhs,
                self.residual,
                self.preconditioned,
                self.block_valid,
                self.preconditioner_valid,
                self.state_status,
                self.completed_iterations,
                self.rho,
                self.rhs_squared,
                self.recursive_residual_squared,
                self.true_residual_squared,
            ],
            device=device,
        )
        for iteration in range(self.iterations):
            self.operator.launch_apply(self.direction, self.operator_direction, self.apply_workspace)
            wp.launch(
                _compute_pcg_step,
                dim=1,
                inputs=[
                    self.direction,
                    self.operator_direction,
                    self.direction_valid,
                    self.state_status,
                    self.rho,
                    iteration,
                    self.trace_curvature,
                    self.trace_step_size,
                    self.trace_status,
                ],
                device=device,
            )
            wp.launch(
                _update_solution_residual,
                dim=count,
                inputs=[
                    self.direction,
                    self.operator_direction,
                    self.state_status,
                    iteration,
                    self.trace_step_size,
                    self.solution,
                    self.residual,
                    self.update_valid,
                ],
                device=device,
            )
            if iteration + 1 == self.iterations:
                wp.launch(
                    _finalize_pcg_iteration,
                    dim=1,
                    inputs=[
                        self.solution,
                        self.residual,
                        self.update_valid,
                        self.state_status,
                        self.completed_iterations,
                        self.recursive_residual_squared,
                        iteration,
                        self.trace_residual_squared,
                        self.trace_status,
                    ],
                    device=device,
                )
                continue
            preconditioner.launch_apply(
                self.residual,
                self.preconditioned,
                self.device_preconditioner_workspaces[iteration + 1],
            )
            wp.launch(
                _validate_device_preconditioner,
                dim=count,
                inputs=[self.state_status, self.preconditioned, self.preconditioner_valid],
                device=device,
            )
            wp.launch(
                _compute_pcg_conjugacy,
                dim=1,
                inputs=[
                    self.solution,
                    self.residual,
                    self.preconditioned,
                    self.update_valid,
                    self.preconditioner_valid,
                    self.state_status,
                    self.completed_iterations,
                    self.rho,
                    self.recursive_residual_squared,
                    iteration,
                    self.trace_residual_squared,
                    self.trace_conjugacy,
                    self.trace_status,
                ],
                device=device,
            )
            wp.launch(
                _update_pcg_direction,
                dim=count,
                inputs=[
                    self.preconditioned,
                    self.state_status,
                    iteration,
                    self.trace_conjugacy,
                    self.direction,
                    self.direction_valid,
                ],
                device=device,
            )
        self.operator.launch_apply(self.solution, self.operator_solution, self.apply_workspace)
        wp.launch(
            _verify_true_residual,
            dim=1,
            inputs=[self.rhs, self.operator_solution, self.state_status, self.true_residual_squared],
            device=device,
        )

    def record(self, *, capture_replay: bool = False) -> WarpFixedPCGRecord:
        """Synchronously materialize a diagnostic record after execution."""
        if not isinstance(capture_replay, bool):
            raise TypeError("capture_replay must be a bool")
        status_code = int(self.state_status.numpy()[0])
        if status_code not in _STATUS_NAMES or status_code == _STATUS_ACTIVE:
            raise RuntimeError(f"device returned invalid terminal PCG status {status_code}")
        completed = int(self.completed_iterations.numpy()[0])
        trace_status = self.trace_status.numpy()
        curvature = self.trace_curvature.numpy()
        step_size = self.trace_step_size.numpy()
        residual_squared = self.trace_residual_squared.numpy()
        conjugacy = self.trace_conjugacy.numpy()
        trace = tuple(
            WarpFixedPCGIteration(
                iteration=index,
                active_update_completed=index < completed,
                status=_STATUS_NAMES[int(trace_status[index])],
                residual_norm=(
                    math.sqrt(float(residual_squared[index]))
                    if index < completed
                    and math.isfinite(float(residual_squared[index]))
                    and float(residual_squared[index]) >= 0.0
                    else None
                ),
                direction_curvature=(
                    float(curvature[index]) if index < completed and math.isfinite(float(curvature[index])) else None
                ),
                step_size=(
                    float(step_size[index]) if index < completed and math.isfinite(float(step_size[index])) else None
                ),
                conjugacy=(
                    float(conjugacy[index])
                    if index < completed
                    and (self.device_preconditioner is None or index + 1 < self.iterations)
                    and int(trace_status[index]) == _STATUS_ACTIVE
                    and math.isfinite(float(conjugacy[index]))
                    else None
                ),
            )
            for index in range(self.iterations)
        )
        rhs_squared = float(self.rhs_squared.numpy()[0])
        recursive_squared = float(self.recursive_residual_squared.numpy()[0])
        true_squared = float(self.true_residual_squared.numpy()[0])
        reason = _STATUS_NAMES[status_code]

        def norm_from_squared(value: float) -> float | None:
            return math.sqrt(value) if math.isfinite(value) and value >= 0.0 else None

        preconditioner_evidence = (
            tuple(
                self.device_preconditioner.record_application(
                    application_index,
                    workspace,
                    capture_replay=capture_replay,
                )
                for application_index, workspace in enumerate(self.device_preconditioner_workspaces)
            )
            if self.device_preconditioner is not None
            else ()
        )
        current_operator_sha256 = self.operator.record_current_operator_sha256()
        binding_sha256 = _operator_preconditioner_binding_sha256(
            current_operator_sha256,
            self.preconditioner_identity,
            self.static_preconditioner_sha256,
        )

        return WarpFixedPCGRecord(
            solution=self.solution.numpy(),
            success=reason in ("completed", "zero_rhs", "converged"),
            reason=reason,
            requested_iterations=self.iterations,
            completed_iterations=completed,
            rhs_norm=norm_from_squared(rhs_squared),
            recursive_residual_norm=norm_from_squared(recursive_squared),
            true_residual_norm=norm_from_squared(true_squared),
            trace=trace,
            work=self.work,
            preconditioner_identity=self.preconditioner_identity,
            capture_replay=capture_replay,
            current_operator_sha256=current_operator_sha256,
            static_preconditioner_sha256=self.static_preconditioner_sha256,
            operator_preconditioner_binding_sha256=binding_sha256,
            preconditioner_evidence=preconditioner_evidence,
        )
