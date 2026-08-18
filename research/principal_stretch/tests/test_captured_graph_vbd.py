# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for captured direct multiplicative-graph VBD."""

from __future__ import annotations

import ctypes
import dataclasses
import inspect
import json
import os
import subprocess
import sys
import unittest
import weakref
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import warp as wp

from research.principal_stretch.captured_graph_vbd import (
    CONTRACT_ID,
    FINALIZE_GATE_BLOCK_DIM,
    FINALIZE_GATE_COLLECTIVE_VERSION,
    FINALIZE_GATE_OWNER_ROLES,
    FINALIZE_GATE_OWNER_THREADS,
    FINALIZE_GATE_ROUTE,
    FIRST_CYCLE_PUBLICATION_ROLE,
    FUSED_GATHER_KERNEL_VERSION,
    OUTER_CORRECTIONS,
    OUTER_KERNEL_VERSION,
    OUTER_SCHEDULE_SHA256,
    OUTER_SCHEDULE_VERSION,
    SECOND_CYCLE_PUBLICATION_ROLE,
    V_CYCLES_PER_OUTER,
    CapturedDirectGraphVBD,
    CapturedGraphVBDEndpoint,
    CapturedGraphVBDTiming,
    _finalize_gate,
    _fused_vertex_outer_terms,
    _hash_parts,
    _initialize_from_k1,
)
from research.principal_stretch.captured_mg_vbd import (
    _commit_candidate,
    _copy_positions,
    _directional_terms,
    _tet_gate_terms,
    _vertex_gate_terms,
    _write_endpoint,
)
from research.principal_stretch.captured_vbd_baseline import CONTRACT_ID as VBD_BASELINE_CONTRACT_ID
from research.principal_stretch.correction_gpu import (
    MatrixFreeStableNHOperator,
    minimum_determinant_on_segment,
)
from research.principal_stretch.correction_gpu_warp import (
    SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
    WarpMatrixFreeWorkspace,
)
from research.principal_stretch.correction_graph_vbd import DirectGraphVBDConfig
from research.principal_stretch.correction_multigrid import apply_v_cycle
from research.principal_stretch.correction_multigrid_warp_scalar_fused import (
    EXTERNAL_SHARED_PUBLICATION_ROUTE as V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE,
)
from research.principal_stretch.correction_multigrid_warp_scalar_fused import (
    KERNEL_VERSION as V_CYCLE_KERNEL_VERSION,
)
from research.principal_stretch.correction_multigrid_warp_scalar_fused import (
    PUBLICATION_VERSION as V_CYCLE_PUBLICATION_VERSION,
)
from research.principal_stretch.correction_multigrid_warp_scalar_fused import (
    SCHEDULE_VERSION as V_CYCLE_SCHEDULE_VERSION,
)
from research.principal_stretch.correction_multigrid_warp_scalar_fused import (
    STANDALONE_PUBLICATION_ROUTE as V_CYCLE_STANDALONE_PUBLICATION_ROUTE,
)
from research.principal_stretch.correction_multigrid_warp_scalar_fused import (
    WarpScalarFusedStaticMultigridHierarchy,
    WarpScalarFusedVCyclePhysicalWork,
    WarpScalarFusedVCycleRecord,
    WarpScalarFusedVCycleWorkspace,
    _copy_scalar_to_vec3,
)
from research.principal_stretch.solver_benchmark import (
    build_common_problem,
    build_structured_cantilever_scene,
    evaluate_common_state,
)
from research.principal_stretch.solver_scenes import build_stretch_scene


@wp.func
def _oracle_finite_vec(value: wp.vec3d) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.kernel(enable_backward=False)
def _oracle_add_vectors(
    left: wp.array[wp.vec3d],
    right: wp.array[wp.vec3d],
    output: wp.array[wp.vec3d],
):
    index = wp.tid()
    output[index] = left[index] + right[index]


@wp.kernel(enable_backward=False)
def _oracle_mask_rhs(active: wp.array[int], rhs: wp.array[wp.vec3d]):
    index = wp.tid()
    if active[0] == 0:
        rhs[index] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


@wp.kernel(enable_backward=False)
def _oracle_subtract_vectors(
    left: wp.array[wp.vec3d],
    right: wp.array[wp.vec3d],
    output: wp.array[wp.vec3d],
):
    index = wp.tid()
    output[index] = left[index] - right[index]


@wp.kernel(enable_backward=False)
def _oracle_build_candidate(
    current: wp.array[wp.vec3d],
    vertex_to_free: wp.array[int],
    direction: wp.array[wp.vec3d],
    active: wp.array[int],
    candidate: wp.array[wp.vec3d],
    proposal_finite: wp.array[int],
):
    vertex = wp.tid()
    value = current[vertex]
    valid = bool(True)
    free_index = vertex_to_free[vertex]
    if active[0] != 0 and free_index >= 0:
        proposed = value + direction[free_index]
        valid = _oracle_finite_vec(proposed)
        if valid:
            publishable = wp.vec3(
                wp.float32(proposed[0]),
                wp.float32(proposed[1]),
                wp.float32(proposed[2]),
            )
            value = wp.vec3d(
                wp.float64(publishable[0]),
                wp.float64(publishable[1]),
                wp.float64(publishable[2]),
            )
    candidate[vertex] = value
    proposal_finite[vertex] = int(valid)


@wp.kernel(enable_backward=False)
def _serial_finalize_gate_oracle(
    outer_index: int,
    current_inertia: wp.array[wp.float64],
    candidate_inertia: wp.array[wp.float64],
    current_elastic: wp.array[wp.float64],
    candidate_elastic: wp.array[wp.float64],
    directional_terms: wp.array[wp.float64],
    candidate_determinants: wp.array[wp.float64],
    segment_minima: wp.array[wp.float64],
    proposal_finite: wp.array[int],
    vertex_finite: wp.array[int],
    tet_finite: wp.array[int],
    minimum_determinant: wp.float64,
    armijo: wp.float64,
    active: wp.array[int],
    accepted: wp.array[int],
    reasons: wp.array[int],
    initial_objective: wp.array[wp.float64],
    candidate_objective: wp.array[wp.float64],
    directional_derivative: wp.array[wp.float64],
    minimum_segment_determinant: wp.array[wp.float64],
):
    """Frozen f502169 serial gate used only as an independent CUDA oracle."""
    if wp.tid() != 0:
        return
    accepted[outer_index] = 0
    initial_objective[outer_index] = wp.float64(0.0)
    candidate_objective[outer_index] = wp.float64(0.0)
    directional_derivative[outer_index] = wp.float64(0.0)
    minimum_segment_determinant[outer_index] = wp.float64(0.0)
    if active[0] == 0:
        reasons[outer_index] = 2
        return

    start_objective = wp.float64(0.0)
    end_objective = wp.float64(0.0)
    derivative = wp.float64(0.0)
    all_finite = bool(True)
    for vertex in range(current_inertia.shape[0]):
        start_objective += current_inertia[vertex]
        end_objective += candidate_inertia[vertex]
        all_finite = all_finite and proposal_finite[vertex] != 0 and vertex_finite[vertex] != 0
    for index in range(directional_terms.shape[0]):
        derivative += directional_terms[index]
    minimum_segment = wp.float64(1.0e300)
    minimum_candidate = wp.float64(1.0e300)
    for tet in range(current_elastic.shape[0]):
        start_objective += current_elastic[tet]
        end_objective += candidate_elastic[tet]
        minimum_segment = wp.min(minimum_segment, segment_minima[tet])
        minimum_candidate = wp.min(minimum_candidate, candidate_determinants[tet])
        all_finite = all_finite and tet_finite[tet] != 0

    if (
        not all_finite
        or not wp.isfinite(start_objective)
        or not wp.isfinite(end_objective)
        or not wp.isfinite(derivative)
        or not wp.isfinite(minimum_segment)
    ):
        reasons[outer_index] = 3
        active[0] = 0
        return

    initial_objective[outer_index] = start_objective
    candidate_objective[outer_index] = end_objective
    directional_derivative[outer_index] = derivative
    minimum_segment_determinant[outer_index] = minimum_segment
    if derivative >= wp.float64(0.0):
        reasons[outer_index] = 4
        active[0] = 0
    elif minimum_candidate <= minimum_determinant or minimum_segment <= minimum_determinant:
        reasons[outer_index] = 5
        active[0] = 0
    elif not (end_objective < start_objective and end_objective <= start_objective + armijo * derivative):
        reasons[outer_index] = 6
        active[0] = 0
    else:
        accepted[outer_index] = 1
        reasons[outer_index] = 1


def _cpu_ordered_gate_reference(
    current_inertia: np.ndarray,
    candidate_inertia: np.ndarray,
    current_elastic: np.ndarray,
    candidate_elastic: np.ndarray,
    directional_terms: np.ndarray,
    candidate_determinants: np.ndarray,
    segment_minima: np.ndarray,
    proposal_finite: np.ndarray,
    vertex_finite: np.ndarray,
    tet_finite: np.ndarray,
    *,
    minimum_determinant: float,
    armijo: float,
    active: int,
) -> tuple[int, int, int, np.float64, np.float64, np.float64, np.float64]:
    """Replay the committed scalar recurrence without vectorized reductions."""
    if active == 0:
        return 0, 0, 2, np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)

    start_objective = np.float64(0.0)
    end_objective = np.float64(0.0)
    all_finite = True
    for vertex in range(current_inertia.shape[0]):
        start_objective = np.float64(start_objective + current_inertia[vertex])
        end_objective = np.float64(end_objective + candidate_inertia[vertex])
        all_finite = all_finite and proposal_finite[vertex] != 0 and vertex_finite[vertex] != 0
    derivative = np.float64(0.0)
    for value in directional_terms:
        derivative = np.float64(derivative + value)
    minimum_segment = np.float64(1.0e300)
    minimum_candidate = np.float64(1.0e300)
    for tet in range(current_elastic.shape[0]):
        start_objective = np.float64(start_objective + current_elastic[tet])
        end_objective = np.float64(end_objective + candidate_elastic[tet])
        minimum_segment = np.float64(min(minimum_segment, segment_minima[tet]))
        minimum_candidate = np.float64(min(minimum_candidate, candidate_determinants[tet]))
        all_finite = all_finite and tet_finite[tet] != 0
    if not all_finite or not all(
        np.isfinite(value) for value in (start_objective, end_objective, derivative, minimum_segment)
    ):
        return 0, 0, 3, np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)
    if derivative >= np.float64(0.0):
        reason = 4
    elif minimum_candidate <= minimum_determinant or minimum_segment <= minimum_determinant:
        reason = 5
    elif not (end_objective < start_objective and end_objective <= start_objective + np.float64(armijo) * derivative):
        reason = 6
    else:
        return 1, 1, 1, start_objective, end_objective, derivative, minimum_segment
    return 0, 0, reason, start_objective, end_objective, derivative, minimum_segment


def _gate_case_arrays(
    case: dict[str, object], device: wp.context.Device
) -> tuple[list[object], dict[str, wp.array[Any]]]:
    """Allocate one gate case with poisoned non-target output slots."""
    f64_names = (
        "current_inertia",
        "candidate_inertia",
        "current_elastic",
        "candidate_elastic",
        "directional_terms",
        "candidate_determinants",
        "segment_minima",
    )
    i32_names = ("proposal_finite", "vertex_finite", "tet_finite")
    inputs: dict[str, wp.array[Any]] = {
        name: wp.array(np.asarray(case[name], dtype=np.float64), dtype=wp.float64, device=device) for name in f64_names
    }
    inputs.update(
        {name: wp.array(np.asarray(case[name], dtype=np.int32), dtype=wp.int32, device=device) for name in i32_names}
    )
    outputs = {
        "active": wp.array(np.array([case.get("active", 1)], dtype=np.int32), dtype=wp.int32, device=device),
        "accepted": wp.array(np.array([71, 72, 73, 74], dtype=np.int32), dtype=wp.int32, device=device),
        "reasons": wp.array(np.array([81, 82, 83, 84], dtype=np.int32), dtype=wp.int32, device=device),
        "initial_objective": wp.array(
            np.array([11.0, -0.0, 13.0, 14.0], dtype=np.float64), dtype=wp.float64, device=device
        ),
        "candidate_objective": wp.array(
            np.array([21.0, -0.0, 23.0, 24.0], dtype=np.float64), dtype=wp.float64, device=device
        ),
        "directional_derivative": wp.array(
            np.array([31.0, -0.0, 33.0, 34.0], dtype=np.float64), dtype=wp.float64, device=device
        ),
        "minimum_segment_determinant": wp.array(
            np.array([41.0, -0.0, 43.0, 44.0], dtype=np.float64), dtype=wp.float64, device=device
        ),
    }
    launch_inputs: list[object] = [
        1,
        inputs["current_inertia"],
        inputs["candidate_inertia"],
        inputs["current_elastic"],
        inputs["candidate_elastic"],
        inputs["directional_terms"],
        inputs["candidate_determinants"],
        inputs["segment_minima"],
        inputs["proposal_finite"],
        inputs["vertex_finite"],
        inputs["tet_finite"],
        float(case.get("minimum_determinant", 0.0)),
        float(case.get("armijo", 1.0e-4)),
        outputs["active"],
        outputs["accepted"],
        outputs["reasons"],
        outputs["initial_objective"],
        outputs["candidate_objective"],
        outputs["directional_derivative"],
        outputs["minimum_segment_determinant"],
    ]
    return launch_inputs, outputs


def _run_gate_case(
    kernel: wp.context.Kernel,
    case: dict[str, object],
    device: wp.context.Device,
    *,
    collective: bool,
) -> tuple[dict[str, tuple[str, tuple[int, ...], bytes]], dict[str, np.ndarray]]:
    launch_inputs, outputs = _gate_case_arrays(case, device)
    if collective:
        wp.launch(
            kernel,
            dim=FINALIZE_GATE_BLOCK_DIM,
            inputs=launch_inputs,
            block_dim=FINALIZE_GATE_BLOCK_DIM,
            device=device,
        )
    else:
        wp.launch(kernel, dim=1, inputs=launch_inputs, device=device)
    patterns = {name: _device_bit_pattern(value) for name, value in outputs.items()}
    host = {name: np.asarray(value.numpy()) for name, value in outputs.items()}
    return patterns, host


def _tiny_scene():
    return build_structured_cantilever_scene(
        dimensions=(2, 2, 1),
        dt=1.0 / 16.0,
        gravity=(0.1, -0.2, -2.0),
        total_tip_force=(4.0, -3.0, -6.0),
        initial_velocity=(0.03, -0.02, 0.01),
        name="captured-direct-graph-vbd-tiny",
    )


def _enqueue_unfused_outer_oracle(solver: CapturedDirectGraphVBD):
    """Enqueue the exact pre-fusion six-launch outer schedule for comparison."""
    module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
    binding = module._lookup_workspace_owners(solver)
    solver._validate_workspace_owner_bindings(binding)
    public = binding.public_vbd
    direct = binding.direct
    scene = binding.scene
    operator = binding.operator
    device_hierarchy = binding.device.scalar
    lane = public.k1_lane
    public.baseline._enqueue_reset_and_step(lane)
    wp.launch(
        _initialize_from_k1,
        dim=scene.n_vertices,
        inputs=[
            lane.state_out.particle_q,
            direct.canonical_positions,
            operator.vertex_to_free,
            operator.positions,
            direct.candidate,
            direct.proposal_finite,
            direct.active,
            direct.accepted,
            direct.reasons,
        ],
        device=solver.device,
    )
    for outer_index, workspace in enumerate(binding.outer):
        wp.launch(
            _copy_positions,
            dim=scene.n_vertices,
            inputs=[operator.positions, direct.outer_start_positions[outer_index]],
            device=solver.device,
        )
        operator.launch_refresh_geometry()
        operator.launch_gradient(workspace.rhs, scale=-1.0)
        wp.launch(_oracle_mask_rhs, dim=operator.n_free, inputs=[direct.active, workspace.rhs], device=solver.device)
        device_hierarchy.launch_apply(
            workspace.rhs,
            workspace.first_correction,
            workspace.first_cycle.workspace,
        )
        operator.launch_apply(
            workspace.first_correction,
            workspace.operator_product_after_first,
            workspace.operator_apply,
        )
        wp.launch(
            _oracle_subtract_vectors,
            dim=operator.n_free,
            inputs=[workspace.rhs, workspace.operator_product_after_first, workspace.residual_after_first],
            device=solver.device,
        )
        device_hierarchy.launch_apply_core(
            workspace.residual_after_first,
            workspace.second_cycle.workspace,
        )
        wp.launch(
            _copy_scalar_to_vec3,
            dim=operator.n_free,
            inputs=[workspace.second_cycle.final_scalar_correction, workspace.second_correction],
            device=solver.device,
        )
        wp.launch(
            _oracle_add_vectors,
            dim=operator.n_free,
            inputs=[workspace.first_correction, workspace.second_correction, workspace.direction],
            device=solver.device,
        )
        wp.launch(
            _oracle_build_candidate,
            dim=scene.n_vertices,
            inputs=[
                operator.positions,
                operator.vertex_to_free,
                workspace.direction,
                direct.active,
                direct.candidate,
                direct.proposal_finite,
            ],
            device=solver.device,
        )
        wp.launch(
            _copy_positions,
            dim=scene.n_vertices,
            inputs=[direct.candidate, direct.outer_candidate_positions[outer_index]],
            device=solver.device,
        )
        wp.launch(
            _vertex_gate_terms,
            dim=scene.n_vertices,
            inputs=[
                operator.positions,
                direct.candidate,
                operator.inertial_target,
                operator.mass,
                operator.inverse_dt_squared,
                direct.current_inertia,
                direct.candidate_inertia,
                direct.vertex_finite,
            ],
            device=solver.device,
        )
        wp.launch(
            _directional_terms,
            dim=operator.n_free,
            inputs=[
                workspace.rhs,
                operator.free,
                operator.positions,
                direct.candidate,
                direct.directional_terms,
            ],
            device=solver.device,
        )
        wp.launch(
            _tet_gate_terms,
            dim=scene.n_tets,
            inputs=[
                operator.deformation_gradients,
                operator.cofactors,
                operator.determinants,
                direct.candidate,
                operator.tets,
                operator.shape_gradients,
                operator.volumes,
                operator.mu,
                operator.lam,
                direct.current_elastic,
                direct.candidate_elastic,
                direct.candidate_determinants,
                direct.segment_minima,
                direct.tet_finite,
            ],
            device=solver.device,
        )
        wp.launch(
            _finalize_gate,
            dim=FINALIZE_GATE_BLOCK_DIM,
            inputs=[
                outer_index,
                direct.current_inertia,
                direct.candidate_inertia,
                direct.current_elastic,
                direct.candidate_elastic,
                direct.directional_terms,
                direct.candidate_determinants,
                direct.segment_minima,
                direct.proposal_finite,
                direct.vertex_finite,
                direct.tet_finite,
                binding.config.minimum_determinant,
                binding.config.armijo,
                direct.active,
                direct.accepted,
                direct.reasons,
                direct.initial_objectives,
                direct.candidate_objectives,
                direct.directional_derivatives,
                direct.minimum_segment_determinants,
            ],
            block_dim=FINALIZE_GATE_BLOCK_DIM,
            device=solver.device,
        )
        wp.launch(
            _commit_candidate,
            dim=scene.n_vertices,
            inputs=[outer_index, direct.candidate, direct.accepted, operator.positions],
            device=solver.device,
        )
    wp.launch(
        _write_endpoint,
        dim=scene.n_vertices,
        inputs=[
            operator.positions,
            direct.x_current,
            operator.vertex_to_free,
            1.0 / scene.dt,
            direct.final_positions,
            direct.final_velocities,
        ],
        device=solver.device,
    )
    return binding


def _enqueue_committed_190_route(solver: CapturedDirectGraphVBD):
    """Enqueue the exact committed 190-node route as a test-only oracle."""
    module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
    binding = module._lookup_workspace_owners(solver)
    first_outputs = {id(outer.first_cycle.workspace): outer.first_correction for outer in binding.outer}
    original_core = WarpScalarFusedStaticMultigridHierarchy.launch_apply_core

    def launch_committed_cycle(hierarchy, rhs, workspace):
        output = first_outputs.get(id(workspace))
        if output is not None:
            hierarchy.launch_apply(rhs, output, workspace)
        else:
            original_core(hierarchy, rhs, workspace)

    def launch_committed_apply(
        operator,
        _direction_scalar,
        published_direction,
        rhs,
        product,
        residual,
        workspace,
    ):
        operator.launch_apply_residual(published_direction, rhs, product, residual, workspace)

    with (
        mock.patch.object(
            WarpScalarFusedStaticMultigridHierarchy,
            "launch_apply_core",
            launch_committed_cycle,
        ),
        mock.patch.object(
            module.WarpMatrixFreeStableNHOperator,
            "launch_apply_residual_scalar_direction",
            launch_committed_apply,
        ),
    ):
        solver._enqueue_integrated(binding)
    return binding


def _device_bit_pattern(array: wp.array[Any]) -> tuple[str, tuple[int, ...], bytes]:
    host = np.ascontiguousarray(array.numpy())
    return host.dtype.str, host.shape, host.tobytes()


def _integrated_bit_patterns(binding) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    direct = binding.direct
    arrays = {
        "operator.positions": binding.operator.positions,
        "candidate": direct.candidate,
        "proposal_finite": direct.proposal_finite,
        "final_positions": direct.final_positions,
        "final_velocities": direct.final_velocities,
        "active": direct.active,
        "accepted": direct.accepted,
        "reasons": direct.reasons,
        "current_inertia": direct.current_inertia,
        "candidate_inertia": direct.candidate_inertia,
        "vertex_finite": direct.vertex_finite,
        "current_elastic": direct.current_elastic,
        "candidate_elastic": direct.candidate_elastic,
        "candidate_determinants": direct.candidate_determinants,
        "segment_minima": direct.segment_minima,
        "tet_finite": direct.tet_finite,
        "directional_terms": direct.directional_terms,
        "initial_objectives": direct.initial_objectives,
        "candidate_objectives": direct.candidate_objectives,
        "directional_derivatives": direct.directional_derivatives,
        "minimum_segment_determinants": direct.minimum_segment_determinants,
    }
    for outer_index, workspace in enumerate(binding.outer):
        arrays[f"outer_{outer_index}.start"] = direct.outer_start_positions[outer_index]
        arrays[f"outer_{outer_index}.candidate"] = direct.outer_candidate_positions[outer_index]
        for field_name in (
            "rhs",
            "first_correction",
            "operator_product_after_first",
            "residual_after_first",
            "second_correction",
            "direction",
        ):
            arrays[f"outer_{outer_index}.{field_name}"] = getattr(workspace, field_name)
    return {name: _device_bit_pattern(array) for name, array in arrays.items()}


def _assert_float32_device_reconstruction(
    testcase: unittest.TestCase,
    solver: CapturedDirectGraphVBD,
    endpoint,
) -> None:
    """Independently replay current-A/two-B work and fp32 publication on CPU."""
    current = endpoint.outer_start_positions[0].copy()
    active = True
    for outer_index in range(OUTER_CORRECTIONS):
        work = endpoint.outer_work[outer_index]
        np.testing.assert_array_equal(endpoint.outer_start_positions[outer_index], current)
        operator = MatrixFreeStableNHOperator.from_problem(solver.problem, current)
        rhs = -operator.gradient_free().reshape(-1, 3)
        if not active:
            rhs.fill(0.0)
        first = apply_v_cycle(solver.hierarchy, rhs.reshape(-1)).correction.reshape(-1, 3)
        product = operator.apply_free(first.reshape(-1)).reshape(-1, 3)
        residual = rhs - product
        second = apply_v_cycle(solver.hierarchy, residual.reshape(-1)).correction.reshape(-1, 3)
        direction = first + second

        np.testing.assert_allclose(work.rhs, rhs, rtol=2.0e-12, atol=2.0e-13)
        np.testing.assert_allclose(work.first_correction, first, rtol=3.0e-12, atol=3.0e-13)
        np.testing.assert_allclose(work.operator_product_after_first, product, rtol=4.0e-12, atol=4.0e-13)
        np.testing.assert_allclose(work.residual_after_first, residual, rtol=4.0e-12, atol=4.0e-13)
        np.testing.assert_allclose(work.second_correction, second, rtol=4.0e-12, atol=4.0e-13)
        np.testing.assert_allclose(work.direction, direction, rtol=5.0e-12, atol=5.0e-13)

        candidate = current.copy()
        if active:
            candidate[operator.free] = (
                (current[operator.free] + direction.reshape(-1, 3)).astype(np.float32).astype(np.float64)
            )
        np.testing.assert_array_equal(endpoint.outer_candidate_positions[outer_index], candidate)
        if not active:
            testcase.assertEqual(endpoint.reasons[outer_index], "masked-after-rejection")
            continue

        candidate_operator = MatrixFreeStableNHOperator.from_problem(solver.problem, candidate)
        actual_step = candidate[operator.free] - current[operator.free]
        derivative = float(np.vdot(operator.gradient_free(), actual_step.reshape(-1)))
        segment = minimum_determinant_on_segment(operator, candidate_operator).determinant
        accepted = bool(
            derivative < 0.0
            and candidate_operator.objective() < operator.objective()
            and candidate_operator.objective() <= operator.objective() + solver.config.armijo * derivative
            and candidate_operator.minimum_determinant > solver.config.minimum_determinant
            and segment > solver.config.minimum_determinant
        )
        testcase.assertEqual(endpoint.accepted[outer_index], accepted)
        testcase.assertAlmostEqual(endpoint.initial_objectives[outer_index], operator.objective(), delta=2.0e-14)
        testcase.assertAlmostEqual(
            endpoint.candidate_objectives[outer_index], candidate_operator.objective(), delta=2.0e-14
        )
        testcase.assertAlmostEqual(endpoint.directional_derivatives[outer_index], derivative, delta=5.0e-12)
        testcase.assertAlmostEqual(endpoint.segment_minimum_determinants[outer_index], segment, delta=2.0e-11)
        if accepted:
            current = candidate
        else:
            active = False

    np.testing.assert_array_equal(endpoint.positions, current.astype(np.float32).astype(np.float64))


class TestCapturedDirectGraphVBD(unittest.TestCase):
    def test_source_is_public_research_only_and_contains_no_krylov_solver(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        source = inspect.getsource(module)
        self.assertNotIn("newton._src", source)
        self.assertNotIn("from newton import _src", source)
        self.assertNotIn("WarpFixedPCG", source)
        self.assertIn("b - A(x) B b", source)
        self.assertIn("end_objective < start_objective", source)
        self.assertIn("minimum_segment <= minimum_determinant", source)
        self.assertIn("WarpScalarFusedStaticMultigridHierarchy", source)
        self.assertNotIn("level_product", source)
        self.assertNotIn("def _mask_rhs", source)
        self.assertNotIn("def _subtract_vectors", source)
        self.assertEqual(source.count("operator.launch_gradient_masked("), 1)
        self.assertEqual(source.count("operator.launch_apply_residual("), 0)
        self.assertEqual(source.count("operator.launch_apply_residual_scalar_direction("), 1)
        self.assertIn('"captured-direct-graph-vbd-graph-identity-v7"', source)

    def test_four_warp_gate_contract_and_cpu_ordered_reference_are_frozen(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        gate_source = inspect.getsource(_finalize_gate.func)
        enqueue_source = inspect.getsource(CapturedDirectGraphVBD._enqueue_integrated)
        self.assertEqual(FINALIZE_GATE_ROUTE, "cuda-one-block-four-warp-ordered-fp64-v1")
        self.assertEqual(FINALIZE_GATE_BLOCK_DIM, 128)
        self.assertEqual(FINALIZE_GATE_OWNER_THREADS, (0, 32, 64, 96))
        self.assertEqual(
            FINALIZE_GATE_OWNER_ROLES,
            (
                "ordered-objective-pair",
                "ordered-directional-derivative",
                "ordered-determinant-minima-pair",
                "ordered-finite-flags",
            ),
        )
        self.assertEqual(
            FINALIZE_GATE_COLLECTIVE_VERSION,
            "shared-tile-vec2d-float64-vec2d-int32-broadcasts-v1",
        )
        self.assertEqual(OUTER_SCHEDULE_SHA256, "9cbe82532dc76e292d0b34df0ba483c53b4eac4ba82fb7536f272dfbc753e8d3")
        self.assertEqual(gate_source.count("wp.tile_from_thread("), 4)
        self.assertEqual(gate_source.count('storage="shared"'), 4)
        self.assertEqual(gate_source.count("wp.tile_extract("), 4)
        self.assertLess(gate_source.rindex("wp.tile_extract("), gate_source.index("if lane != 0:"))
        self.assertNotIn("tile_sum", gate_source)
        self.assertNotIn("atomic_", gate_source)
        self.assertNotIn("_FINALIZE_", gate_source)
        for owner in (0, 32, 64, 96):
            self.assertIn(f"if lane == {owner}:", gate_source)
            self.assertIn(f"thread_idx={owner}", gate_source)
        for name in (
            "_FINALIZE_OBJECTIVE_THREAD",
            "_FINALIZE_DERIVATIVE_THREAD",
            "_FINALIZE_MINIMA_THREAD",
            "_FINALIZE_FINITE_THREAD",
        ):
            self.assertNotIn(name, vars(module))
        self.assertIn("dim=FINALIZE_GATE_BLOCK_DIM", enqueue_source)
        self.assertIn("block_dim=FINALIZE_GATE_BLOCK_DIM", enqueue_source)
        self.assertFalse(
            any("finalize" in name or "gate" in name for name in module._DirectCorrectionOwnerBinding._fields)
        )

        case = {
            "current_inertia": np.array([1.0e16]),
            "candidate_inertia": np.array([0.0]),
            "current_elastic": np.array([-1.0e16, 1.0]),
            "candidate_elastic": np.array([0.0, 0.0]),
            "directional_terms": np.array([1.0e16, -1.0e16, -1.0]),
            "candidate_determinants": np.array([1.0, 2.0]),
            "segment_minima": np.array([3.0, 1.0]),
            "proposal_finite": np.array([1]),
            "vertex_finite": np.array([1]),
            "tet_finite": np.array([1, 1]),
        }
        result = _cpu_ordered_gate_reference(**case, minimum_determinant=0.0, armijo=1.0e-4, active=1)
        self.assertEqual(result[:3], (1, 1, 1))
        self.assertEqual(result[3:], (1.0, 0.0, -1.0, 1.0))
        split_start = np.float64(np.sum(case["current_inertia"], dtype=np.float64)) + np.float64(
            np.sum(case["current_elastic"], dtype=np.float64)
        )
        self.assertNotEqual(split_start.view(np.uint64), result[3].view(np.uint64))

        masked = _cpu_ordered_gate_reference(
            **{
                name: np.full_like(value, np.nan, dtype=np.float64) if value.dtype.kind == "f" else value
                for name, value in case.items()
            },
            minimum_determinant=0.0,
            armijo=1.0e-4,
            active=0,
        )
        self.assertEqual(masked[:3], (0, 0, 2))
        self.assertEqual(np.asarray(masked[3:]).view(np.uint64).tolist(), [0, 0, 0, 0])

    def test_fresh_process_private_owner_injection_cannot_change_prejit_module_hash(self):
        private_names = (
            "_FINALIZE_OBJECTIVE_THREAD",
            "_FINALIZE_DERIVATIVE_THREAD",
            "_FINALIZE_MINIMA_THREAD",
            "_FINALIZE_FINITE_THREAD",
        )
        script_prefix = f"""
import importlib
module = importlib.import_module("research.principal_stretch.captured_graph_vbd")
private_names = {private_names!r}
assert not any(name in vars(module) for name in private_names)
"""
        hashes = []
        for inject in (False, True):
            injection = "\nfor name in private_names:\n    setattr(module, name, 0)\n" if inject else "\n"
            script = (
                script_prefix
                + injection
                + 'print("FINALIZE_GATE_MODULE_SHA256=" + module._finalize_gate.module.get_module_hash().hex())\n'
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.getcwd(),
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            marker = "FINALIZE_GATE_MODULE_SHA256="
            matches = [line.removeprefix(marker) for line in completed.stdout.splitlines() if line.startswith(marker)]
            self.assertEqual(len(matches), 1, completed.stdout + completed.stderr)
            hashes.append(matches[0])
        self.assertEqual(hashes[0], hashes[1])

    def test_contract_rejects_cpu_and_nondefault_coarse_bound(self):
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            CapturedDirectGraphVBD(_tiny_scene(), device="cpu")
        with self.assertRaisesRegex(ValueError, "coarse_node_limit=4"):
            CapturedDirectGraphVBD(
                _tiny_scene(),
                device="cpu",
                config=DirectGraphVBDConfig(coarse_node_limit=1),
            )

    def test_diagnostic_timing_is_balanced_and_cannot_claim_performance(self):
        common = {
            "graph_seconds": (1.0e-3, 1.1e-3),
            "k4_seconds": (2.0e-4, 2.1e-4),
            "random_seed": 17,
            "device": "cuda:0",
            "contract_id": CONTRACT_ID,
            "scene_sha256": "0" * 64,
            "objective_instance_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "static_hierarchy_sha256": "3" * 64,
            "persistent_device_sha256": "4" * 64,
            "graph_identity_sha256": "5" * 64,
            "k4_graph_identity_sha256": "6" * 64,
            "comparator_contract_id": VBD_BASELINE_CONTRACT_ID,
            "fused_gather_kernel_version": FUSED_GATHER_KERNEL_VERSION,
            "scalar_direction_apply_kernel_version": SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
            "v_cycle_kernel_version": V_CYCLE_KERNEL_VERSION,
            "v_cycle_schedule_version": V_CYCLE_SCHEDULE_VERSION,
            "v_cycle_schedule_sha256": "7" * 64,
            "v_cycle_core_schedule_sha256": "8" * 64,
            "v_cycle_publication_version": V_CYCLE_PUBLICATION_VERSION,
            "v_cycle_standalone_publication_route": V_CYCLE_STANDALONE_PUBLICATION_ROUTE,
            "v_cycle_external_shared_publication_route": V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE,
            "first_cycle_publication_role": FIRST_CYCLE_PUBLICATION_ROLE,
            "second_cycle_publication_role": SECOND_CYCLE_PUBLICATION_ROLE,
            "v_cycle_kernel_launches": 20,
            "v_cycle_core_kernel_launches": 19,
            "outer_kernel_version": OUTER_KERNEL_VERSION,
            "outer_schedule_version": OUTER_SCHEDULE_VERSION,
            "outer_schedule_sha256": OUTER_SCHEDULE_SHA256,
            "finalize_gate_route": FINALIZE_GATE_ROUTE,
            "finalize_gate_block_dim": FINALIZE_GATE_BLOCK_DIM,
            "finalize_gate_owner_threads": FINALIZE_GATE_OWNER_THREADS,
            "finalize_gate_owner_roles": FINALIZE_GATE_OWNER_ROLES,
            "finalize_gate_collective_version": FINALIZE_GATE_COLLECTIVE_VERSION,
            "correction_kernel_launches": 186,
        }
        with self.assertRaisesRegex(ValueError, "equal AB and BA"):
            CapturedGraphVBDTiming(pair_orders=("AB", "AB"), warmup_replays=1, **common)
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            CapturedGraphVBDTiming(
                pair_orders=("AB", "BA"),
                warmup_replays=1,
                performance_evidence=True,
                **common,
            )

        timing = CapturedGraphVBDTiming(pair_orders=("AB", "BA"), warmup_replays=1, **common)
        attacks = (
            (
                {
                    "performance_evidence": True,
                    "graph_identity_sha256": "f" * 64,
                },
                "diagnostic-only",
            ),
            ({"fused_gather_kernel_version": FUSED_GATHER_KERNEL_VERSION + "-forged"}, "kernel and publication"),
            (
                {"scalar_direction_apply_kernel_version": SCALAR_DIRECTION_APPLY_KERNEL_VERSION + "-forged"},
                "kernel and publication",
            ),
            ({"v_cycle_publication_version": V_CYCLE_PUBLICATION_VERSION + "-forged"}, "kernel and publication"),
            ({"first_cycle_publication_role": FIRST_CYCLE_PUBLICATION_ROLE + "-forged"}, "kernel and publication"),
            ({"second_cycle_publication_role": SECOND_CYCLE_PUBLICATION_ROLE + "-forged"}, "kernel and publication"),
            ({"finalize_gate_route": FINALIZE_GATE_ROUTE + "-forged"}, "finalize gate route"),
            ({"finalize_gate_block_dim": 96}, "finalize gate block dimension"),
            ({"finalize_gate_owner_threads": (0, 1, 2, 3)}, "finalize gate owner threads"),
            (
                {"finalize_gate_owner_roles": (*FINALIZE_GATE_OWNER_ROLES[:-1], "forged")},
                "finalize gate owner roles",
            ),
            (
                {"finalize_gate_collective_version": FINALIZE_GATE_COLLECTIVE_VERSION + "-forged"},
                "finalize gate collective version",
            ),
            ({"correction_kernel_launches": 194}, "dual-core captured schedule"),
            ({"graph_identity_sha256": "forged"}, "lowercase SHA-256"),
            ({"graph_seconds": (-1.0e-3, 1.1e-3)}, "finite and positive"),
            ({"pair_orders": ["AB"]}, "same positive even length"),
        )
        for mutations, message in attacks:
            with self.subTest(mutations=mutations):
                originals = {name: getattr(timing, name) for name in mutations}
                try:
                    for name, value in mutations.items():
                        object.__setattr__(timing, name, value)
                    with self.assertRaisesRegex(ValueError, message):
                        timing.deterministic_record()
                finally:
                    for name, value in originals.items():
                        object.__setattr__(timing, name, value)

        object.__setattr__(timing, "pair_orders", ["AB", "BA"])
        object.__setattr__(timing, "graph_seconds", [1.0e-3, 1.1e-3])
        try:
            canonical = timing.deterministic_record()
            self.assertEqual(canonical["pair_orders"], ["AB", "BA"])
            self.assertEqual(canonical["graph_seconds"], [1.0e-3, 1.1e-3])
            self.assertEqual(canonical["finalize_gate_route"], FINALIZE_GATE_ROUTE)
            self.assertEqual(canonical["finalize_gate_block_dim"], FINALIZE_GATE_BLOCK_DIM)
            self.assertEqual(canonical["finalize_gate_owner_threads"], list(FINALIZE_GATE_OWNER_THREADS))
            self.assertEqual(canonical["finalize_gate_owner_roles"], list(FINALIZE_GATE_OWNER_ROLES))
            self.assertEqual(canonical["finalize_gate_collective_version"], FINALIZE_GATE_COLLECTIVE_VERSION)
            self.assertEqual(
                canonical["measurement_authentication"],
                "schema-validated-not-content-authenticated-v1",
            )
            self.assertFalse(canonical["solver_issued_authentication"])
        finally:
            object.__setattr__(timing, "pair_orders", ("AB", "BA"))
            object.__setattr__(timing, "graph_seconds", (1.0e-3, 1.1e-3))

        object.__setattr__(timing, "graph_identity_sha256", "f" * 64)
        object.__setattr__(timing, "graph_seconds", (2.0e-3, 2.1e-3))
        try:
            coherent_diagnostic = timing.deterministic_record()
            self.assertEqual(coherent_diagnostic["graph_identity_sha256"], "f" * 64)
            self.assertEqual(coherent_diagnostic["graph_seconds"], [2.0e-3, 2.1e-3])
            self.assertFalse(coherent_diagnostic["performance_evidence"])
            self.assertFalse(coherent_diagnostic["solver_issued_authentication"])
            self.assertAlmostEqual(timing.graph_median_seconds, 2.05e-3)
        finally:
            object.__setattr__(timing, "graph_identity_sha256", "5" * 64)
            object.__setattr__(timing, "graph_seconds", (1.0e-3, 1.1e-3))

        object.__setattr__(timing, "graph_seconds", (-1.0e-3, 1.1e-3))
        try:
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                _ = timing.graph_median_seconds
        finally:
            object.__setattr__(timing, "graph_seconds", (1.0e-3, 1.1e-3))

        object.__setattr__(timing, "k4_seconds", (2.0e-4, float("nan")))
        try:
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                _ = timing.k4_median_seconds
        finally:
            object.__setattr__(timing, "k4_seconds", (2.0e-4, 2.1e-4))

    def test_public_endpoint_constructor_cannot_mint_unvalidated_evidence(self):
        fake_positions = tuple(np.full((1, 3), float(index), dtype=np.float64) for index in range(4))
        with self.assertRaisesRegex(TypeError, "_validation_context"):
            CapturedGraphVBDEndpoint(
                scene_sha256="0" * 64,
                objective_instance_sha256="1" * 64,
                static_hierarchy_sha256="2" * 64,
                config_sha256="3" * 64,
                k1_endpoint_sha256="4" * 64,
                k1_position_sha256="5" * 64,
                k1_velocity_sha256="6" * 64,
                k1_pristine_state_sha256="7" * 64,
                persistent_device_sha256="8" * 64,
                graph_identity_sha256="9" * 64,
                fused_gather_kernel_version=FUSED_GATHER_KERNEL_VERSION,
                scalar_direction_apply_kernel_version=SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
                v_cycle_publication_version=V_CYCLE_PUBLICATION_VERSION,
                v_cycle_standalone_publication_route=V_CYCLE_STANDALONE_PUBLICATION_ROUTE,
                v_cycle_external_shared_publication_route=V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE,
                first_cycle_publication_role=FIRST_CYCLE_PUBLICATION_ROLE,
                second_cycle_publication_role=SECOND_CYCLE_PUBLICATION_ROLE,
                outer_kernel_version=OUTER_KERNEL_VERSION,
                outer_schedule_version=OUTER_SCHEDULE_VERSION,
                outer_schedule_sha256=OUTER_SCHEDULE_SHA256,
                finalize_gate_route=FINALIZE_GATE_ROUTE,
                finalize_gate_block_dim=FINALIZE_GATE_BLOCK_DIM,
                finalize_gate_owner_threads=FINALIZE_GATE_OWNER_THREADS,
                finalize_gate_owner_roles=FINALIZE_GATE_OWNER_ROLES,
                finalize_gate_collective_version=FINALIZE_GATE_COLLECTIVE_VERSION,
                armijo=1.0e-4,
                minimum_determinant=0.0,
                free_vertices=np.array([0]),
                positions=np.full((1, 3), 4.0),
                velocities=np.zeros((1, 3)),
                accepted=(True,) * 4,
                reasons=("accepted",) * 4,
                initial_objectives=(1.0,) * 4,
                candidate_objectives=(0.5,) * 4,
                directional_derivatives=(-0.1,) * 4,
                segment_minimum_determinants=(1.0,) * 4,
                outer_start_positions=fake_positions,
                outer_candidate_positions=tuple(value + 1.0 for value in fake_positions),
                outer_work=(),
                graph_replay=True,
            )


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestFinalizeGateFourWarpCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.device = wp.get_device("cuda:0")

    @staticmethod
    def _base_case(n_vertices: int = 3, n_tets: int = 3, n_directional: int = 3) -> dict[str, object]:
        return {
            "current_inertia": np.full(n_vertices, 2.0, dtype=np.float64),
            "candidate_inertia": np.full(n_vertices, 0.5, dtype=np.float64),
            "current_elastic": np.full(n_tets, 3.0, dtype=np.float64),
            "candidate_elastic": np.full(n_tets, 1.0, dtype=np.float64),
            "directional_terms": np.full(n_directional, -1.0, dtype=np.float64),
            "candidate_determinants": np.full(n_tets, 2.0, dtype=np.float64),
            "segment_minima": np.full(n_tets, 1.0, dtype=np.float64),
            "proposal_finite": np.ones(n_vertices, dtype=np.int32),
            "vertex_finite": np.ones(n_vertices, dtype=np.int32),
            "tet_finite": np.ones(n_tets, dtype=np.int32),
            "minimum_determinant": 0.0,
            "armijo": 1.0e-4,
            "active": 1,
        }

    def _assert_serial_bitwise_equal(self, case: dict[str, object]) -> dict[str, np.ndarray]:
        serial_patterns, serial_host = _run_gate_case(
            _serial_finalize_gate_oracle,
            case,
            self.device,
            collective=False,
        )
        collective_patterns, collective_host = _run_gate_case(
            _finalize_gate,
            case,
            self.device,
            collective=True,
        )
        self.assertEqual(serial_patterns.keys(), collective_patterns.keys())
        for name in serial_patterns:
            with self.subTest(output=name):
                self.assertEqual(serial_patterns[name], collective_patterns[name])
                np.testing.assert_array_equal(serial_host[name], collective_host[name])
        return collective_host

    def test_exact_recurrences_and_all_gate_decisions_match_frozen_serial(self):
        ordered = self._base_case(n_vertices=1, n_tets=2, n_directional=3)
        ordered.update(
            {
                "current_inertia": np.array([1.0e16]),
                "candidate_inertia": np.array([0.0]),
                "current_elastic": np.array([-1.0e16, 1.0]),
                "candidate_elastic": np.array([0.0, 0.0]),
                "directional_terms": np.array([1.0e16, -1.0e16, -1.0]),
                "candidate_determinants": np.array([2.0, 1.0]),
                "segment_minima": np.array([3.0, 1.0]),
            }
        )
        decision_cases: list[tuple[str, dict[str, object], int]] = [("accepted-ordered", ordered, 1)]

        nonfinite = self._base_case()
        nonfinite["vertex_finite"] = np.array([1, 0, 1], dtype=np.int32)
        decision_cases.append(("nonfinite", nonfinite, 3))

        non_descent = self._base_case(n_directional=1)
        non_descent["directional_terms"] = np.array([-0.0])
        decision_cases.append(("non-descent-signed-zero", non_descent, 4))

        segment = self._base_case(n_tets=1)
        segment["segment_minima"] = np.array([-0.0])
        segment["candidate_determinants"] = np.array([1.0])
        decision_cases.append(("segment-signed-zero", segment, 5))

        candidate_minimum = self._base_case(n_tets=1)
        candidate_minimum["segment_minima"] = np.array([1.0])
        candidate_minimum["candidate_determinants"] = np.array([-0.0])
        decision_cases.append(("candidate-minimum-signed-zero", candidate_minimum, 5))

        threshold = self._base_case(n_tets=1)
        threshold["candidate_determinants"] = np.array([0.25])
        threshold["segment_minima"] = np.array([0.5])
        threshold["minimum_determinant"] = 0.25
        decision_cases.append(("candidate-threshold-equality", threshold, 5))

        objective = self._base_case()
        objective["candidate_inertia"] = np.array(objective["current_inertia"], copy=True)
        objective["candidate_elastic"] = np.array(objective["current_elastic"], copy=True)
        decision_cases.append(("objective-equality", objective, 6))

        masked = self._base_case()
        for array_name in (
            "current_inertia",
            "candidate_inertia",
            "current_elastic",
            "candidate_elastic",
            "directional_terms",
            "candidate_determinants",
            "segment_minima",
        ):
            masked[array_name] = np.full_like(masked[array_name], np.nan)
        masked["proposal_finite"] = np.zeros(3, dtype=np.int32)
        masked["vertex_finite"] = np.zeros(3, dtype=np.int32)
        masked["tet_finite"] = np.zeros(3, dtype=np.int32)
        masked["active"] = 0
        decision_cases.append(("masked-neutral-collectives", masked, 2))

        for name, case, expected_reason in decision_cases:
            with self.subTest(case=name):
                result = self._assert_serial_bitwise_equal(case)
                self.assertEqual(int(result["reasons"][1]), expected_reason)
                if expected_reason == 2:
                    for output_name in (
                        "initial_objective",
                        "candidate_objective",
                        "directional_derivative",
                        "minimum_segment_determinant",
                    ):
                        self.assertEqual(result[output_name][1].view(np.uint64), 0)

    def test_nan_infinity_flags_and_minimum_order_match_frozen_serial_bitwise(self):
        adversarial: list[tuple[str, dict[str, object]]] = []
        numeric_names = (
            "current_inertia",
            "candidate_inertia",
            "current_elastic",
            "candidate_elastic",
            "directional_terms",
            "candidate_determinants",
            "segment_minima",
        )
        for array_name in numeric_names:
            for label, value in (("nan", np.nan), ("positive-inf", np.inf), ("negative-inf", -np.inf)):
                case = self._base_case()
                values = np.array(case[array_name], copy=True)
                values[1] = value
                case[array_name] = values
                adversarial.append((f"{array_name}-{label}", case))
        for array_name in ("proposal_finite", "vertex_finite", "tet_finite"):
            for index in (0, 1, 2):
                case = self._base_case()
                values = np.array(case[array_name], copy=True)
                values[index] = 0
                case[array_name] = values
                adversarial.append((f"{array_name}-zero-{index}", case))
        for array_name in ("candidate_determinants", "segment_minima"):
            for index in (0, 1, 2):
                case = self._base_case()
                values = np.array([0.75, 0.5, 0.25], dtype=np.float64)
                values[index] = np.nan
                case[array_name] = values
                adversarial.append((f"{array_name}-nan-order-{index}", case))

        for name, case in adversarial:
            with self.subTest(case=name):
                self._assert_serial_bitwise_equal(case)

    def test_armijo_and_both_minimum_thresholds_match_at_adjacent_float64_values(self):
        armijo_bound = np.float64(1.75)
        armijo_values = (
            ("below", np.nextafter(armijo_bound, -np.inf), 1),
            ("equal", armijo_bound, 1),
            ("above", np.nextafter(armijo_bound, np.inf), 6),
        )
        for label, end_objective, expected_reason in armijo_values:
            case = self._base_case(n_vertices=1, n_tets=1, n_directional=1)
            case.update(
                {
                    "current_inertia": np.array([2.0]),
                    "current_elastic": np.array([0.0]),
                    "candidate_inertia": np.array([end_objective]),
                    "candidate_elastic": np.array([0.0]),
                    "directional_terms": np.array([-1.0]),
                    "armijo": 0.25,
                }
            )
            with self.subTest(armijo=label):
                result = self._assert_serial_bitwise_equal(case)
                self.assertEqual(int(result["reasons"][1]), expected_reason)
                self.assertEqual(result["candidate_objective"][1].view(np.uint64), end_objective.view(np.uint64))

        threshold = np.float64(0.25)
        minimum_values = (
            ("below", np.nextafter(threshold, -np.inf), 5),
            ("equal", threshold, 5),
            ("above", np.nextafter(threshold, np.inf), 1),
        )
        for array_name in ("candidate_determinants", "segment_minima"):
            for label, value, expected_reason in minimum_values:
                case = self._base_case(n_vertices=1, n_tets=1, n_directional=1)
                case["minimum_determinant"] = threshold
                case[array_name] = np.array([value])
                with self.subTest(minimum=array_name, relation=label):
                    result = self._assert_serial_bitwise_equal(case)
                    self.assertEqual(int(result["reasons"][1]), expected_reason)
                    if array_name == "segment_minima":
                        self.assertEqual(
                            result["minimum_segment_determinant"][1].view(np.uint64),
                            value.view(np.uint64),
                        )

    def test_arbitrary_unbalanced_domain_sizes_match_frozen_serial_bitwise(self):
        sizes = (
            (0, 0, 0),
            (1, 97, 33),
            (31, 1, 96),
            (32, 33, 95),
            (33, 32, 97),
            (95, 96, 31),
            (96, 95, 32),
            (97, 31, 95),
        )
        for n_vertices, n_tets, n_directional in sizes:
            with self.subTest(vertices=n_vertices, tets=n_tets, directional=n_directional):
                self._assert_serial_bitwise_equal(self._base_case(n_vertices, n_tets, n_directional))


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestCapturedDirectGraphVBDTinyCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.scene = _tiny_scene()
        cls.solver = CapturedDirectGraphVBD(cls.scene, device="cuda:0")
        cls.solver.capture_graphs(warmup_replays=1)
        cls.endpoint = cls.solver.run(graph_replay=True)

    @staticmethod
    def _owner_boundaries(solver: CapturedDirectGraphVBD):
        return (
            ("run", lambda: solver.run(graph_replay=True)),
            ("run_k4", lambda: solver.run_k4(graph_replay=True)),
            ("timing", lambda: solver.benchmark_paired(pair_count=2, warmup_replays=1)),
            ("serialization", solver.deterministic_record),
            ("recapture", lambda: solver.capture_graphs(warmup_replays=1)),
            ("poison", lambda: solver.poison(seed=4817)),
        )

    def test_float32_device_semantics_match_independent_cpu_reconstruction(self):
        _assert_float32_device_reconstruction(self, self.solver, self.endpoint)

    def test_exact_two_cycle_work_and_launch_count_are_retained(self):
        endpoint = self.endpoint
        core_launches = self.solver.device_hierarchy.core_kernel_launches
        expected_linear = 5 + 2 * core_launches
        expected_outer = expected_linear + 3
        expected_total = 2 + OUTER_CORRECTIONS * expected_outer
        self.assertEqual(endpoint.total_v_cycle_count, OUTER_CORRECTIONS * V_CYCLES_PER_OUTER)
        self.assertEqual(endpoint.correction_kernel_launches, expected_total)
        self.assertEqual(self.solver.correction_kernel_launches, expected_total)
        self.assertTrue(endpoint.exact_work_completed)
        for outer_index, work in enumerate(endpoint.outer_work):
            with self.subTest(outer=outer_index):
                self.assertEqual(work.outer_index, outer_index)
                self.assertEqual(work.linear_kernel_launches, expected_linear)
                self.assertTrue(work.exact_work_completed)
                self.assertEqual(len(work.v_cycles), 2)
                for record in work.v_cycles:
                    self.assertIs(type(record), WarpScalarFusedVCycleRecord)
                    self.assertIs(type(record.physical_work), WarpScalarFusedVCyclePhysicalWork)
                    expected_cycle_launches = core_launches
                    expected_route = V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE
                    expected_schedule = self.solver.device_hierarchy.core_schedule_sha256
                    self.assertEqual(record.scheduled_kernel_launches, expected_cycle_launches)
                    self.assertEqual(record.schedule_version, V_CYCLE_SCHEDULE_VERSION)
                    self.assertEqual(record.work.hierarchy_sha256, endpoint.static_hierarchy_sha256)
                    self.assertEqual(record.work.rhs_count, 1)
                    self.assertEqual(record.work.coarsest_factor_solves, 1)
                    physical = record.physical_work
                    self.assertEqual(
                        physical.matrix_block_products_executed + physical.matrix_block_products_elided_zero_start,
                        record.work.matrix_block_products,
                    )
                    self.assertEqual(physical.scheduled_kernel_launches, expected_cycle_launches)
                    self.assertEqual(physical.core_kernel_launches, core_launches)
                    self.assertEqual(physical.publication_kernel_launches, 0)
                    self.assertEqual(physical.publication_version, V_CYCLE_PUBLICATION_VERSION)
                    self.assertEqual(physical.publication_route, expected_route)
                    self.assertEqual(physical.root_ingress_zero_start_fusions, 1)
                    self.assertEqual(physical.schedule_sha256, expected_schedule)
        schedule = self.solver.deterministic_record()
        self.assertEqual(schedule["contract_id"], CONTRACT_ID)
        self.assertEqual(schedule["krylov_iterations"], 0)
        self.assertEqual(schedule["v_cycles"], 8)
        self.assertEqual(schedule["linear_prefix_kernel_launches_per_outer"], expected_linear - 1)
        self.assertEqual(schedule["fused_gather_kernel_launches_per_outer"], 2)
        self.assertEqual(schedule["fused_vertex_kernel_launches_per_outer"], 1)
        self.assertEqual(schedule["finalize_gate_kernel_launches_per_outer"], 1)
        self.assertEqual(schedule["outer_kernel_launches_per_outer"], expected_outer)
        self.assertEqual(schedule["correction_kernel_launches_excluding_public_k1"], expected_total)
        self.assertEqual(schedule["graph_identity_schema"], "captured-direct-graph-vbd-graph-identity-v7")
        self.assertEqual(schedule["fused_gather_kernel_version"], FUSED_GATHER_KERNEL_VERSION)
        self.assertEqual(
            schedule["scalar_direction_apply_kernel_version"],
            SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
        )
        self.assertEqual(schedule["first_cycle_publication_role"], FIRST_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(schedule["second_cycle_publication_role"], SECOND_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(schedule["first_cycle_kernel_launches"], core_launches)
        self.assertEqual(schedule["second_cycle_kernel_launches"], core_launches)
        self.assertEqual(schedule["v_cycle_first_publication_kernel_launches"], 0)
        self.assertEqual(schedule["v_cycle_second_publication_kernel_launches"], 0)
        self.assertEqual(schedule["outer_kernel_version"], OUTER_KERNEL_VERSION)
        self.assertEqual(schedule["outer_schedule_version"], OUTER_SCHEDULE_VERSION)
        self.assertEqual(schedule["outer_schedule_sha256"], OUTER_SCHEDULE_SHA256)
        self.assertEqual(schedule["finalize_gate_route"], FINALIZE_GATE_ROUTE)
        self.assertEqual(schedule["finalize_gate_block_dim"], FINALIZE_GATE_BLOCK_DIM)
        self.assertEqual(schedule["finalize_gate_owner_threads"], list(FINALIZE_GATE_OWNER_THREADS))
        self.assertEqual(schedule["finalize_gate_owner_roles"], list(FINALIZE_GATE_OWNER_ROLES))
        self.assertEqual(schedule["finalize_gate_collective_version"], FINALIZE_GATE_COLLECTIVE_VERSION)
        self.assertEqual(schedule["v_cycle_schedule_version"], V_CYCLE_SCHEDULE_VERSION)
        self.assertEqual(schedule["v_cycle_publication_version"], V_CYCLE_PUBLICATION_VERSION)
        self.assertEqual(schedule["v_cycle_standalone_publication_route"], V_CYCLE_STANDALONE_PUBLICATION_ROUTE)
        self.assertEqual(
            schedule["v_cycle_external_shared_publication_route"],
            V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE,
        )
        self.assertEqual(schedule["v_cycle_root_ingress_zero_start_fusions"], 1)
        self.assertEqual(schedule["workspace_owner_identity_sha256"], self.solver._workspace_owner_identity_sha256())
        self.assertFalse(schedule["performance_evidence"])
        json.dumps(schedule, allow_nan=False)
        endpoint_record = endpoint.deterministic_record()
        self.assertEqual(endpoint_record["fused_gather_kernel_version"], FUSED_GATHER_KERNEL_VERSION)
        self.assertEqual(
            endpoint_record["scalar_direction_apply_kernel_version"],
            SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
        )
        self.assertEqual(endpoint_record["first_cycle_publication_role"], FIRST_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(endpoint_record["second_cycle_publication_role"], SECOND_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(endpoint_record["v_cycle_publication_version"], V_CYCLE_PUBLICATION_VERSION)
        self.assertEqual(endpoint_record["outer_kernel_version"], OUTER_KERNEL_VERSION)
        self.assertEqual(endpoint_record["outer_schedule_version"], OUTER_SCHEDULE_VERSION)
        self.assertEqual(endpoint_record["outer_schedule_sha256"], OUTER_SCHEDULE_SHA256)
        self.assertEqual(endpoint_record["finalize_gate_route"], FINALIZE_GATE_ROUTE)
        self.assertEqual(endpoint_record["finalize_gate_block_dim"], FINALIZE_GATE_BLOCK_DIM)
        self.assertEqual(endpoint_record["finalize_gate_owner_threads"], list(FINALIZE_GATE_OWNER_THREADS))
        self.assertEqual(endpoint_record["finalize_gate_owner_roles"], list(FINALIZE_GATE_OWNER_ROLES))
        self.assertEqual(endpoint_record["finalize_gate_collective_version"], FINALIZE_GATE_COLLECTIVE_VERSION)
        for outer_record in endpoint_record["outer_work"]:
            self.assertEqual(outer_record["fused_gather_kernel_version"], FUSED_GATHER_KERNEL_VERSION)
            self.assertEqual(
                outer_record["scalar_direction_apply_kernel_version"],
                SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
            )
            self.assertEqual(outer_record["v_cycle_publication_version"], V_CYCLE_PUBLICATION_VERSION)
            self.assertEqual(outer_record["first_cycle_publication_route"], FIRST_CYCLE_PUBLICATION_ROLE)
            self.assertEqual(outer_record["second_cycle_publication_route"], SECOND_CYCLE_PUBLICATION_ROLE)
            self.assertEqual(outer_record["outer_kernel_version"], OUTER_KERNEL_VERSION)
            self.assertEqual(outer_record["outer_schedule_version"], OUTER_SCHEDULE_VERSION)
            self.assertEqual(outer_record["outer_schedule_sha256"], OUTER_SCHEDULE_SHA256)
            self.assertEqual(outer_record["finalize_gate_route"], FINALIZE_GATE_ROUTE)
            self.assertEqual(outer_record["finalize_gate_block_dim"], FINALIZE_GATE_BLOCK_DIM)
            self.assertEqual(outer_record["finalize_gate_owner_threads"], list(FINALIZE_GATE_OWNER_THREADS))
            self.assertEqual(outer_record["finalize_gate_owner_roles"], list(FINALIZE_GATE_OWNER_ROLES))
            self.assertEqual(outer_record["finalize_gate_collective_version"], FINALIZE_GATE_COLLECTIVE_VERSION)
        json.dumps(endpoint_record, allow_nan=False)

    def test_fused_integrated_schedule_is_bitwise_equal_to_unfused_oracle(self):
        fused = CapturedDirectGraphVBD(self.scene, device="cuda:0")
        oracle = CapturedDirectGraphVBD(self.scene, device="cuda:0")
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        fused_binding = module._lookup_workspace_owners(fused)
        fused._enqueue_integrated(fused_binding)
        oracle_binding = _enqueue_unfused_outer_oracle(oracle)
        fused_patterns = _integrated_bit_patterns(fused_binding)
        oracle_patterns = _integrated_bit_patterns(oracle_binding)
        self.assertEqual(fused_patterns.keys(), oracle_patterns.keys())
        for name in fused_patterns:
            with self.subTest(array=name):
                self.assertEqual(fused_patterns[name], oracle_patterns[name])

    def test_fused_gathers_match_legacy_oracle_after_sticky_rejection(self):
        config = DirectGraphVBDConfig(minimum_determinant=2.0)
        fused = CapturedDirectGraphVBD(self.scene, device="cuda:0", config=config)
        oracle = CapturedDirectGraphVBD(self.scene, device="cuda:0", config=config)
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        fused_binding = module._lookup_workspace_owners(fused)
        fused._enqueue_integrated(fused_binding)
        oracle_binding = _enqueue_unfused_outer_oracle(oracle)
        fused_patterns = _integrated_bit_patterns(fused_binding)
        oracle_patterns = _integrated_bit_patterns(oracle_binding)
        self.assertEqual(fused_patterns.keys(), oracle_patterns.keys())
        for name in fused_patterns:
            with self.subTest(array=name):
                self.assertEqual(fused_patterns[name], oracle_patterns[name])

        np.testing.assert_array_equal(fused_binding.direct.accepted.numpy(), 0)
        np.testing.assert_array_equal(
            fused_binding.direct.reasons.numpy(),
            np.array([5, 2, 2, 2], dtype=np.int32),
        )
        for workspace in fused_binding.outer[1:]:
            for field_name in (
                "rhs",
                "first_correction",
                "operator_product_after_first",
                "residual_after_first",
                "second_correction",
                "direction",
            ):
                bit_pattern = _device_bit_pattern(getattr(workspace, field_name))[2]
                self.assertEqual(bit_pattern, bytes(len(bit_pattern)))

    def test_186_route_is_bitwise_equal_to_exact_committed_190_route(self):
        for label, config in (
            ("accepted", None),
            ("sticky-rejection", DirectGraphVBDConfig(minimum_determinant=2.0)),
        ):
            with self.subTest(route=label):
                fused = CapturedDirectGraphVBD(self.scene, device="cuda:0", config=config)
                committed = CapturedDirectGraphVBD(self.scene, device="cuda:0", config=config)
                module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
                fused_binding = module._lookup_workspace_owners(fused)
                fused._enqueue_integrated(fused_binding)
                committed_binding = _enqueue_committed_190_route(committed)
                fused_patterns = _integrated_bit_patterns(fused_binding)
                committed_patterns = _integrated_bit_patterns(committed_binding)
                self.assertEqual(fused_patterns.keys(), committed_patterns.keys())
                for name in fused_patterns:
                    with self.subTest(route=label, array=name):
                        self.assertEqual(fused_patterns[name], committed_patterns[name])

    def test_fused_vertex_kernel_matches_unfused_edge_semantics_bitwise(self):
        device = self.solver.device
        current_host = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, -2.0, 3.0],
                [2.0, 0.25, -0.5],
                [np.nan, 1.0, 2.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float64,
        )
        vertex_to_free_host = np.array([-1, 0, 1, -1, 2, 3], dtype=np.int32)
        first_host = np.array(
            [
                [2.0**-24, 0.0, 0.0],
                [1.0e100, 0.0, 0.0],
                [np.inf, 0.0, 0.0],
                [np.nan, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        second_host = np.zeros_like(first_host)
        rhs_host = np.array(
            [[1.0, -2.0, 3.0], [-1.0, 0.5, 2.0], [3.0, 4.0, 5.0], [2.0, -1.0, 0.25]],
            dtype=np.float64,
        )
        inertial_target_host = np.zeros_like(current_host)
        mass_host = np.arange(1, current_host.shape[0] + 1, dtype=np.float64)
        current = wp.array(current_host, dtype=wp.vec3d, device=device)
        vertex_to_free = wp.array(vertex_to_free_host, dtype=wp.int32, device=device)
        first = wp.array(first_host, dtype=wp.vec3d, device=device)
        second_scalar = wp.array(second_host.reshape(-1), dtype=wp.float64, device=device)
        rhs = wp.array(rhs_host, dtype=wp.vec3d, device=device)
        inertial_target = wp.array(inertial_target_host, dtype=wp.vec3d, device=device)
        mass = wp.array(mass_host, dtype=wp.float64, device=device)
        free_vertices = wp.array(np.array([1, 2, 4, 5]), dtype=wp.int32, device=device)
        n_vertices = current_host.shape[0]
        n_free = first_host.shape[0]

        def outputs():
            return {
                "second_correction": wp.empty(n_free, dtype=wp.vec3d, device=device),
                "direction": wp.empty(n_free, dtype=wp.vec3d, device=device),
                "outer_start": wp.empty(n_vertices, dtype=wp.vec3d, device=device),
                "candidate": wp.empty(n_vertices, dtype=wp.vec3d, device=device),
                "outer_candidate": wp.empty(n_vertices, dtype=wp.vec3d, device=device),
                "proposal_finite": wp.empty(n_vertices, dtype=wp.int32, device=device),
                "current_inertia": wp.empty(n_vertices, dtype=wp.float64, device=device),
                "candidate_inertia": wp.empty(n_vertices, dtype=wp.float64, device=device),
                "vertex_finite": wp.empty(n_vertices, dtype=wp.int32, device=device),
                "directional_terms": wp.empty(n_free, dtype=wp.float64, device=device),
            }

        for active_value in (1, 0):
            with self.subTest(active=active_value):
                active = wp.array([active_value], dtype=wp.int32, device=device)
                fused = outputs()
                oracle = outputs()
                wp.launch(
                    _copy_positions,
                    dim=n_vertices,
                    inputs=[current, oracle["outer_start"]],
                    device=device,
                )
                wp.launch(
                    _copy_scalar_to_vec3,
                    dim=n_free,
                    inputs=[second_scalar, oracle["second_correction"]],
                    device=device,
                )
                wp.launch(
                    _oracle_add_vectors,
                    dim=n_free,
                    inputs=[first, oracle["second_correction"], oracle["direction"]],
                    device=device,
                )
                wp.launch(
                    _oracle_build_candidate,
                    dim=n_vertices,
                    inputs=[
                        current,
                        vertex_to_free,
                        oracle["direction"],
                        active,
                        oracle["candidate"],
                        oracle["proposal_finite"],
                    ],
                    device=device,
                )
                wp.launch(
                    _copy_positions,
                    dim=n_vertices,
                    inputs=[oracle["candidate"], oracle["outer_candidate"]],
                    device=device,
                )
                wp.launch(
                    _vertex_gate_terms,
                    dim=n_vertices,
                    inputs=[
                        current,
                        oracle["candidate"],
                        inertial_target,
                        mass,
                        9.0,
                        oracle["current_inertia"],
                        oracle["candidate_inertia"],
                        oracle["vertex_finite"],
                    ],
                    device=device,
                )
                wp.launch(
                    _directional_terms,
                    dim=n_free,
                    inputs=[rhs, free_vertices, current, oracle["candidate"], oracle["directional_terms"]],
                    device=device,
                )
                wp.launch(
                    _fused_vertex_outer_terms,
                    dim=n_vertices,
                    inputs=[
                        current,
                        vertex_to_free,
                        first,
                        second_scalar,
                        fused["second_correction"],
                        rhs,
                        fused["direction"],
                        active,
                        inertial_target,
                        mass,
                        9.0,
                        fused["outer_start"],
                        fused["candidate"],
                        fused["outer_candidate"],
                        fused["proposal_finite"],
                        fused["current_inertia"],
                        fused["candidate_inertia"],
                        fused["vertex_finite"],
                        fused["directional_terms"],
                    ],
                    device=device,
                )
                for name in fused:
                    with self.subTest(active=active_value, array=name):
                        self.assertEqual(_device_bit_pattern(fused[name]), _device_bit_pattern(oracle[name]))

                candidate = np.asarray(fused["candidate"].numpy(), dtype=np.float64)
                proposal_finite = np.asarray(fused["proposal_finite"].numpy(), dtype=np.int32)
                vertex_finite = np.asarray(fused["vertex_finite"].numpy(), dtype=np.int32)
                np.testing.assert_array_equal(candidate[[0, 3]], current_host[[0, 3]])
                if active_value:
                    self.assertEqual(candidate[1, 0], 1.0)
                    self.assertTrue(np.isinf(candidate[2, 0]))
                    np.testing.assert_array_equal(candidate[[4, 5]], current_host[[4, 5]])
                    np.testing.assert_array_equal(proposal_finite, np.array([1, 1, 1, 1, 0, 0]))
                    np.testing.assert_array_equal(vertex_finite, np.array([1, 1, 0, 0, 1, 1]))
                else:
                    self.assertEqual(candidate.view(np.uint64).tobytes(), current_host.view(np.uint64).tobytes())
                    np.testing.assert_array_equal(proposal_finite, np.ones(n_vertices, dtype=np.int32))
                    np.testing.assert_array_equal(vertex_finite, np.array([1, 1, 1, 0, 1, 1]))

    def test_coordinated_outer_schedule_version_and_hash_tamper_fails_closed(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        original_globals = (
            module.FUSED_GATHER_KERNEL_VERSION,
            module.SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
            module.FIRST_CYCLE_PUBLICATION_ROLE,
            module.SECOND_CYCLE_PUBLICATION_ROLE,
            module.OUTER_KERNEL_VERSION,
            module.OUTER_SCHEDULE_VERSION,
            module.OUTER_SCHEDULE_SHA256,
            module.FINALIZE_GATE_ROUTE,
            module.FINALIZE_GATE_BLOCK_DIM,
            module.FINALIZE_GATE_OWNER_THREADS,
            module.FINALIZE_GATE_OWNER_ROLES,
            module.FINALIZE_GATE_COLLECTIVE_VERSION,
        )
        original_facades = (
            self.solver._fused_gather_kernel_version_bound,
            self.solver._scalar_direction_apply_kernel_version_bound,
            self.solver._first_cycle_publication_role_bound,
            self.solver._second_cycle_publication_role_bound,
            self.solver._outer_kernel_version_bound,
            self.solver._outer_schedule_version_bound,
            self.solver._outer_schedule_sha256_bound,
            self.solver._finalize_gate_route_bound,
            self.solver._finalize_gate_block_dim_bound,
            self.solver._finalize_gate_owner_threads_bound,
            self.solver._finalize_gate_owner_roles_bound,
            self.solver._finalize_gate_collective_version_bound,
        )
        marker = np.asarray(self.solver.final_positions.numpy(), dtype=np.float32).copy()
        for boundary_name, operation in self._owner_boundaries(self.solver):
            with self.subTest(boundary=boundary_name):
                try:
                    module.FUSED_GATHER_KERNEL_VERSION = original_globals[0] + "-forged"
                    module.SCALAR_DIRECTION_APPLY_KERNEL_VERSION = original_globals[1] + "-forged"
                    module.FIRST_CYCLE_PUBLICATION_ROLE = original_globals[2] + "-forged"
                    module.SECOND_CYCLE_PUBLICATION_ROLE = original_globals[3] + "-forged"
                    module.OUTER_KERNEL_VERSION = original_globals[4] + "-forged"
                    module.OUTER_SCHEDULE_VERSION = original_globals[5] + "-forged"
                    module.FINALIZE_GATE_ROUTE = original_globals[7] + "-forged"
                    module.FINALIZE_GATE_BLOCK_DIM = 96
                    module.FINALIZE_GATE_OWNER_THREADS = (0, 1, 2, 3)
                    module.FINALIZE_GATE_OWNER_ROLES = tuple(role + "-forged" for role in original_globals[10])
                    module.FINALIZE_GATE_COLLECTIVE_VERSION = original_globals[11] + "-forged"
                    forged_sha256 = module._derive_outer_schedule_sha256(
                        module.OUTER_KERNEL_VERSION,
                        module.FUSED_GATHER_KERNEL_VERSION,
                        module.SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
                        module.V_CYCLE_PUBLICATION_VERSION,
                        module.V_CYCLE_STANDALONE_PUBLICATION_ROUTE,
                        module.V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE,
                        module.FIRST_CYCLE_PUBLICATION_ROLE,
                        module.SECOND_CYCLE_PUBLICATION_ROLE,
                        module.FINALIZE_GATE_ROUTE,
                        module.FINALIZE_GATE_BLOCK_DIM,
                        module.FINALIZE_GATE_OWNER_THREADS,
                        module.FINALIZE_GATE_OWNER_ROLES,
                        module.FINALIZE_GATE_COLLECTIVE_VERSION,
                        module.OUTER_SCHEDULE_VERSION,
                    )
                    module.OUTER_SCHEDULE_SHA256 = forged_sha256
                    self.solver._fused_gather_kernel_version_bound = module.FUSED_GATHER_KERNEL_VERSION
                    self.solver._scalar_direction_apply_kernel_version_bound = (
                        module.SCALAR_DIRECTION_APPLY_KERNEL_VERSION
                    )
                    self.solver._first_cycle_publication_role_bound = module.FIRST_CYCLE_PUBLICATION_ROLE
                    self.solver._second_cycle_publication_role_bound = module.SECOND_CYCLE_PUBLICATION_ROLE
                    self.solver._outer_kernel_version_bound = module.OUTER_KERNEL_VERSION
                    self.solver._outer_schedule_version_bound = module.OUTER_SCHEDULE_VERSION
                    self.solver._outer_schedule_sha256_bound = forged_sha256
                    self.solver._finalize_gate_route_bound = module.FINALIZE_GATE_ROUTE
                    self.solver._finalize_gate_block_dim_bound = module.FINALIZE_GATE_BLOCK_DIM
                    self.solver._finalize_gate_owner_threads_bound = module.FINALIZE_GATE_OWNER_THREADS
                    self.solver._finalize_gate_owner_roles_bound = module.FINALIZE_GATE_OWNER_ROLES
                    self.solver._finalize_gate_collective_version_bound = module.FINALIZE_GATE_COLLECTIVE_VERSION
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "finalize gate construction claims|outer kernel or schedule construction claim",
                    ):
                        operation()
                    np.testing.assert_array_equal(
                        np.asarray(self.solver.final_positions.numpy(), dtype=np.float32),
                        marker,
                    )
                finally:
                    (
                        module.FUSED_GATHER_KERNEL_VERSION,
                        module.SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
                        module.FIRST_CYCLE_PUBLICATION_ROLE,
                        module.SECOND_CYCLE_PUBLICATION_ROLE,
                        module.OUTER_KERNEL_VERSION,
                        module.OUTER_SCHEDULE_VERSION,
                        module.OUTER_SCHEDULE_SHA256,
                        module.FINALIZE_GATE_ROUTE,
                        module.FINALIZE_GATE_BLOCK_DIM,
                        module.FINALIZE_GATE_OWNER_THREADS,
                        module.FINALIZE_GATE_OWNER_ROLES,
                        module.FINALIZE_GATE_COLLECTIVE_VERSION,
                    ) = original_globals
                    (
                        self.solver._fused_gather_kernel_version_bound,
                        self.solver._scalar_direction_apply_kernel_version_bound,
                        self.solver._first_cycle_publication_role_bound,
                        self.solver._second_cycle_publication_role_bound,
                        self.solver._outer_kernel_version_bound,
                        self.solver._outer_schedule_version_bound,
                        self.solver._outer_schedule_sha256_bound,
                        self.solver._finalize_gate_route_bound,
                        self.solver._finalize_gate_block_dim_bound,
                        self.solver._finalize_gate_owner_threads_bound,
                        self.solver._finalize_gate_owner_roles_bound,
                        self.solver._finalize_gate_collective_version_bound,
                    ) = original_facades

    def test_finalize_gate_claims_owner_context_and_scratch_free_bindings_fail_closed(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        gate_array_names = {
            "active",
            "accepted",
            "reasons",
            "current_inertia",
            "candidate_inertia",
            "vertex_finite",
            "current_elastic",
            "candidate_elastic",
            "candidate_determinants",
            "segment_minima",
            "tet_finite",
            "directional_terms",
            "initial_objectives",
            "candidate_objectives",
            "directional_derivatives",
            "minimum_segment_determinants",
        }
        self.assertTrue(gate_array_names.issubset(binding.direct._fields))
        self.assertFalse(any("finalize_gate" in name for name, _array in self.solver._persistent_input_arrays(binding)))
        for name in gate_array_names:
            with self.subTest(pointer=name):
                self.assertIs(getattr(self.solver, name), getattr(binding.direct, name))
                self.assertEqual(int(getattr(self.solver, name).ptr), int(getattr(binding.direct, name).ptr))

        claim_attacks = (
            ("finalize_gate_route", FINALIZE_GATE_ROUTE + "-forged"),
            ("finalize_gate_block_dim", 96),
            ("finalize_gate_owner_threads", (0, 32, 64, 95)),
            ("finalize_gate_owner_roles", (*FINALIZE_GATE_OWNER_ROLES[:-1], "forged")),
            ("finalize_gate_collective_version", FINALIZE_GATE_COLLECTIVE_VERSION + "-forged"),
            ("finalize_gate_owner_threads", (*FINALIZE_GATE_OWNER_THREADS,)),
            ("finalize_gate_owner_roles", (*FINALIZE_GATE_OWNER_ROLES,)),
        )
        for field_name, value in claim_attacks:
            with self.subTest(claim=field_name, value=value):
                forged = binding._replace(claims=binding.claims._replace(**{field_name: value}))
                with self.assertRaisesRegex(
                    RuntimeError,
                    "finalize gate construction claims|outer kernel or schedule construction claim",
                ):
                    self.solver._validate_workspace_owner_bindings(forged)

        for field_name in ("_finalize_gate_owner_threads_bound", "_finalize_gate_owner_roles_bound"):
            original = getattr(self.solver, field_name)
            try:
                setattr(self.solver, field_name, (*original,))
                with self.assertRaisesRegex(RuntimeError, "outer kernel or schedule construction claim"):
                    self.solver.deterministic_record()
            finally:
                setattr(self.solver, field_name, original)

        context = self.endpoint._validation_context
        context_attacks = (
            ("finalize_gate_route", FINALIZE_GATE_ROUTE + "-forged"),
            ("finalize_gate_block_dim", 96),
            ("finalize_gate_owner_threads", (0, 32, 64, 95)),
            (
                "finalize_gate_owner_roles",
                (*FINALIZE_GATE_OWNER_ROLES[:-1], "forged"),
            ),
            (
                "finalize_gate_collective_version",
                FINALIZE_GATE_COLLECTIVE_VERSION + "-forged",
            ),
        )
        for field_name, value in context_attacks:
            original = getattr(context, field_name)
            with self.subTest(context=field_name):
                try:
                    object.__setattr__(context, field_name, value)
                    with self.assertRaisesRegex(ValueError, "validation context fields"):
                        self.endpoint.deterministic_record()
                    with self.assertRaisesRegex(ValueError, "validation context fields"):
                        self.endpoint.outer_work[0].deterministic_record()
                finally:
                    object.__setattr__(context, field_name, original)

    def test_endpoint_and_outer_serializers_revalidate_context_and_content_hashes(self):
        context = self.endpoint._validation_context
        original_context_schedule = (
            context.fused_gather_kernel_version,
            context.scalar_direction_apply_kernel_version,
            context.v_cycle_publication_version,
            context.v_cycle_standalone_publication_route,
            context.v_cycle_external_shared_publication_route,
            context.first_cycle_publication_role,
            context.second_cycle_publication_role,
            context.outer_kernel_version,
            context.outer_schedule_version,
            context.outer_schedule_sha256,
        )
        forged_gather_version = original_context_schedule[0] + "-forged"
        forged_scalar_apply_version = original_context_schedule[1] + "-forged"
        forged_publication_version = original_context_schedule[2] + "-forged"
        forged_standalone_route = original_context_schedule[3] + "-forged"
        forged_external_route = original_context_schedule[4] + "-forged"
        forged_first_role = original_context_schedule[5] + "-forged"
        forged_second_role = original_context_schedule[6] + "-forged"
        forged_kernel_version = original_context_schedule[7] + "-forged"
        forged_schedule_version = original_context_schedule[8] + "-forged"
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        forged_sha256 = module._derive_outer_schedule_sha256(
            forged_kernel_version,
            forged_gather_version,
            forged_scalar_apply_version,
            forged_publication_version,
            forged_standalone_route,
            forged_external_route,
            forged_first_role,
            forged_second_role,
            context.finalize_gate_route,
            context.finalize_gate_block_dim,
            context.finalize_gate_owner_threads,
            context.finalize_gate_owner_roles,
            context.finalize_gate_collective_version,
            forged_schedule_version,
        )
        serializers = (
            ("endpoint", self.endpoint.deterministic_record),
            ("outer-work", self.endpoint.outer_work[0].deterministic_record),
        )
        for name, serializer in serializers:
            with self.subTest(context=name):
                try:
                    object.__setattr__(context, "fused_gather_kernel_version", forged_gather_version)
                    object.__setattr__(
                        context,
                        "scalar_direction_apply_kernel_version",
                        forged_scalar_apply_version,
                    )
                    object.__setattr__(context, "v_cycle_publication_version", forged_publication_version)
                    object.__setattr__(context, "v_cycle_standalone_publication_route", forged_standalone_route)
                    object.__setattr__(context, "v_cycle_external_shared_publication_route", forged_external_route)
                    object.__setattr__(context, "first_cycle_publication_role", forged_first_role)
                    object.__setattr__(context, "second_cycle_publication_role", forged_second_role)
                    object.__setattr__(context, "outer_kernel_version", forged_kernel_version)
                    object.__setattr__(context, "outer_schedule_version", forged_schedule_version)
                    object.__setattr__(context, "outer_schedule_sha256", forged_sha256)
                    with self.assertRaisesRegex(ValueError, "validation context fields"):
                        serializer()
                finally:
                    object.__setattr__(context, "fused_gather_kernel_version", original_context_schedule[0])
                    object.__setattr__(
                        context,
                        "scalar_direction_apply_kernel_version",
                        original_context_schedule[1],
                    )
                    object.__setattr__(context, "v_cycle_publication_version", original_context_schedule[2])
                    object.__setattr__(context, "v_cycle_standalone_publication_route", original_context_schedule[3])
                    object.__setattr__(
                        context, "v_cycle_external_shared_publication_route", original_context_schedule[4]
                    )
                    object.__setattr__(context, "first_cycle_publication_role", original_context_schedule[5])
                    object.__setattr__(context, "second_cycle_publication_role", original_context_schedule[6])
                    object.__setattr__(context, "outer_kernel_version", original_context_schedule[7])
                    object.__setattr__(context, "outer_schedule_version", original_context_schedule[8])
                    object.__setattr__(context, "outer_schedule_sha256", original_context_schedule[9])

        outer = self.endpoint.outer_work[0]
        original_outer_hash = outer.content_sha256
        try:
            object.__setattr__(outer, "content_sha256", "a" * 64)
            with self.assertRaisesRegex(ValueError, "outer work content hash is not canonical"):
                outer.deterministic_record()
        finally:
            object.__setattr__(outer, "content_sha256", original_outer_hash)

        original_endpoint_hash = self.endpoint.endpoint_sha256
        try:
            object.__setattr__(self.endpoint, "endpoint_sha256", "b" * 64)
            with self.assertRaisesRegex(ValueError, "endpoint content hash is not canonical"):
                self.endpoint.deterministic_record()
        finally:
            object.__setattr__(self.endpoint, "endpoint_sha256", original_endpoint_hash)

    def test_changed_poison_replay_restores_endpoint_and_all_linear_evidence(self):
        expected = self.endpoint
        for replay_index, seed in enumerate((1701, 99831, 42)):
            with self.subTest(replay=replay_index):
                self.solver.poison(seed=seed)
                actual = self.solver.run(graph_replay=True)
                self.assertEqual(actual.endpoint_sha256, expected.endpoint_sha256)
                np.testing.assert_array_equal(actual.positions, expected.positions)
                np.testing.assert_array_equal(actual.velocities, expected.velocities)
                self.assertEqual(actual.accepted, expected.accepted)
                self.assertEqual(actual.reasons, expected.reasons)
                for actual_work, expected_work in zip(actual.outer_work, expected.outer_work, strict=True):
                    self.assertEqual(actual_work.content_sha256, expected_work.content_sha256)
                    np.testing.assert_array_equal(actual_work.rhs, expected_work.rhs)
                    np.testing.assert_array_equal(actual_work.first_correction, expected_work.first_correction)
                    np.testing.assert_array_equal(
                        actual_work.operator_product_after_first,
                        expected_work.operator_product_after_first,
                    )
                    np.testing.assert_array_equal(actual_work.residual_after_first, expected_work.residual_after_first)
                    np.testing.assert_array_equal(actual_work.second_correction, expected_work.second_correction)
                    np.testing.assert_array_equal(actual_work.direction, expected_work.direction)

    def test_coordinated_rehash_cannot_forge_candidate_or_negative_vcycle_work(self):
        candidates = list(self.endpoint.outer_candidate_positions)
        candidates[0] = self.endpoint.outer_start_positions[0]
        with self.assertRaisesRegex(ValueError, "float32-publishable"):
            dataclasses.replace(self.endpoint, outer_candidate_positions=tuple(candidates))

        outer = self.endpoint.outer_work[0]
        record = outer.v_cycles[0]
        bad_work = dataclasses.replace(record.work, restriction_block_products=-7)
        work_sha256 = _hash_parts(
            "v-cycle-work-record-v1",
            (
                ("hierarchy_sha256", bad_work.hierarchy_sha256),
                ("rhs_sha256", bad_work.rhs_sha256),
                ("result_sha256", bad_work.result_sha256),
                ("rhs_count", bad_work.rhs_count),
                ("level_visits", np.asarray(bad_work.level_visits, dtype=np.int64)),
                ("matrix_block_products", bad_work.matrix_block_products),
                ("smoother_block_solves", bad_work.smoother_block_solves),
                ("restriction_block_products", bad_work.restriction_block_products),
                ("prolongation_block_products", bad_work.prolongation_block_products),
                ("coarsest_factor_solves", bad_work.coarsest_factor_solves),
            ),
        )
        bad_work = dataclasses.replace(bad_work, content_sha256=work_sha256)
        record_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-result-v4",
            (
                ("device_snapshot_sha256", record.device_snapshot_sha256),
                ("static_device_content_sha256", record.static_device_content_sha256),
                ("schedule_sha256", record.schedule_sha256),
                ("work_sha256", work_sha256),
                ("physical_work_sha256", record.physical_work.content_sha256),
                ("scheduled_kernel_launches", record.scheduled_kernel_launches),
                ("capture_replay", record.capture_replay),
            ),
        )
        bad_record = dataclasses.replace(record, work=bad_work, content_sha256=record_sha256)
        with self.assertRaisesRegex(ValueError, "canonical fixed work|non-negative"):
            dataclasses.replace(outer, v_cycles=(bad_record, outer.v_cycles[1]))

        physical = record.physical_work
        bad_executed = physical.matrix_block_products_executed + 1
        bad_elided = physical.matrix_block_products_elided_zero_start - 1
        physical_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-physical-work-v4",
            (
                ("hierarchy_sha256", physical.hierarchy_sha256),
                ("schedule_sha256", physical.schedule_sha256),
                ("rhs_sha256", physical.rhs_sha256),
                ("result_sha256", physical.result_sha256),
                ("matrix_block_products_executed", bad_executed),
                ("matrix_block_products_elided_zero_start", bad_elided),
                ("zero_start_block_solves", physical.zero_start_block_solves),
                (
                    "root_ingress_zero_start_fusions",
                    physical.root_ingress_zero_start_fusions,
                ),
                ("out_of_place_jacobi_block_solves", physical.out_of_place_jacobi_block_solves),
                ("matrix_kernel_launches", physical.matrix_kernel_launches),
                ("jacobi_kernel_launches", physical.jacobi_kernel_launches),
                ("core_kernel_launches", physical.core_kernel_launches),
                ("publication_kernel_launches", physical.publication_kernel_launches),
                ("publication_version", physical.publication_version),
                ("publication_route", physical.publication_route),
                ("scheduled_kernel_launches", physical.scheduled_kernel_launches),
            ),
        )
        bad_physical = dataclasses.replace(
            physical,
            matrix_block_products_executed=bad_executed,
            matrix_block_products_elided_zero_start=bad_elided,
            content_sha256=physical_sha256,
        )
        physical_record_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-result-v4",
            (
                ("device_snapshot_sha256", record.device_snapshot_sha256),
                ("static_device_content_sha256", record.static_device_content_sha256),
                ("schedule_sha256", record.schedule_sha256),
                ("work_sha256", record.work.content_sha256),
                ("physical_work_sha256", physical_sha256),
                ("scheduled_kernel_launches", record.scheduled_kernel_launches),
                ("capture_replay", record.capture_replay),
            ),
        )
        bad_physical_record = dataclasses.replace(
            record,
            physical_work=bad_physical,
            content_sha256=physical_record_sha256,
        )
        with self.assertRaisesRegex(ValueError, "physical matrix_block_products_executed"):
            dataclasses.replace(outer, v_cycles=(bad_physical_record, outer.v_cycles[1]))

        original_root_fusions = physical.root_ingress_zero_start_fusions
        original_physical_sha256 = physical.content_sha256
        original_record_sha256 = record.content_sha256
        forged_root_fusions = 0
        forged_physical_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-physical-work-v4",
            (
                ("hierarchy_sha256", physical.hierarchy_sha256),
                ("schedule_sha256", physical.schedule_sha256),
                ("rhs_sha256", physical.rhs_sha256),
                ("result_sha256", physical.result_sha256),
                ("matrix_block_products_executed", physical.matrix_block_products_executed),
                (
                    "matrix_block_products_elided_zero_start",
                    physical.matrix_block_products_elided_zero_start,
                ),
                ("zero_start_block_solves", physical.zero_start_block_solves),
                ("root_ingress_zero_start_fusions", forged_root_fusions),
                ("out_of_place_jacobi_block_solves", physical.out_of_place_jacobi_block_solves),
                ("matrix_kernel_launches", physical.matrix_kernel_launches),
                ("jacobi_kernel_launches", physical.jacobi_kernel_launches),
                ("core_kernel_launches", physical.core_kernel_launches),
                ("publication_kernel_launches", physical.publication_kernel_launches),
                ("publication_version", physical.publication_version),
                ("publication_route", physical.publication_route),
                ("scheduled_kernel_launches", physical.scheduled_kernel_launches),
            ),
        )
        forged_record_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-result-v4",
            (
                ("device_snapshot_sha256", record.device_snapshot_sha256),
                ("static_device_content_sha256", record.static_device_content_sha256),
                ("schedule_sha256", record.schedule_sha256),
                ("work_sha256", record.work.content_sha256),
                ("physical_work_sha256", forged_physical_sha256),
                ("scheduled_kernel_launches", record.scheduled_kernel_launches),
                ("capture_replay", record.capture_replay),
            ),
        )
        try:
            object.__setattr__(physical, "root_ingress_zero_start_fusions", forged_root_fusions)
            object.__setattr__(physical, "content_sha256", forged_physical_sha256)
            object.__setattr__(record, "content_sha256", forged_record_sha256)
            with self.assertRaisesRegex(ValueError, "physical root_ingress_zero_start_fusions"):
                dataclasses.replace(outer)
        finally:
            object.__setattr__(physical, "root_ingress_zero_start_fusions", original_root_fusions)
            object.__setattr__(physical, "content_sha256", original_physical_sha256)
            object.__setattr__(record, "content_sha256", original_record_sha256)

        original_schedule_version = record.schedule_version
        try:
            object.__setattr__(record, "schedule_version", original_schedule_version + "-forged")
            with self.assertRaisesRegex(ValueError, "invalid V-cycle policy provenance"):
                dataclasses.replace(outer)
        finally:
            object.__setattr__(record, "schedule_version", original_schedule_version)

        for field_name, value in (("research_only", 1), ("performance_evidence", 0)):
            with self.subTest(policy_field=field_name):
                bad_policy_record = dataclasses.replace(record, **{field_name: value})
                with self.assertRaisesRegex(ValueError, "invalid V-cycle policy provenance"):
                    dataclasses.replace(outer, v_cycles=(bad_policy_record, outer.v_cycles[1]))

        provenance_forgeries = (
            ({"scene_sha256": "a" * 64}, "canonical retained scene"),
            ({"objective_instance_sha256": "b" * 64}, "canonical retained scene"),
            ({"config_sha256": "c" * 64}, "exact captured configuration"),
            ({"k1_endpoint_sha256": "d" * 64}, "K1 hashes"),
            ({"persistent_device_sha256": "e" * 64}, "persistent device identity"),
            ({"graph_identity_sha256": "f" * 64}, "graph identity"),
            ({"free_vertices": self.endpoint.free_vertices[::-1]}, "canonical problem ordering"),
        )
        for replacement, message in provenance_forgeries:
            with self.subTest(field=next(iter(replacement))):
                with self.assertRaisesRegex(ValueError, message):
                    dataclasses.replace(self.endpoint, **replacement)

        cloned_context = dataclasses.replace(
            self.endpoint._validation_context,
            persistent_device_sha256="a" * 64,
            graph_identity_sha256="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "exact live solver-issued object"):
            tuple(
                dataclasses.replace(
                    work,
                    persistent_device_sha256="a" * 64,
                    _validation_context=cloned_context,
                )
                for work in self.endpoint.outer_work
            )

        context = self.endpoint._validation_context
        original_schedule = context.v_cycle_schedule_sha256
        original_core_schedule = context.v_cycle_core_schedule_sha256
        original_static = context.v_cycle_static_device_content_sha256
        original_snapshot = context.v_cycle_device_snapshot_sha256
        original_core_snapshot = context.v_cycle_core_device_snapshot_sha256
        original_launches = context.v_cycle_kernel_launches
        original_core_launches = context.v_cycle_core_kernel_launches
        original_publication_version = context.v_cycle_publication_version
        original_root_fusions = context.v_cycle_root_ingress_zero_start_fusions
        original_capture_binding = context.capture_binding
        try:
            object.__setattr__(context, "v_cycle_schedule_sha256", "c" * 64)
            with self.assertRaisesRegex(ValueError, "scalar-fused schedule identity"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_schedule_sha256", original_schedule)
            object.__setattr__(context, "v_cycle_core_schedule_sha256", "f" * 64)
            with self.assertRaisesRegex(ValueError, "core schedule identity"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_core_schedule_sha256", original_core_schedule)
            object.__setattr__(context, "v_cycle_static_device_content_sha256", "d" * 64)
            with self.assertRaisesRegex(ValueError, "scalar-fused static content identity"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_static_device_content_sha256", original_static)
            object.__setattr__(context, "v_cycle_device_snapshot_sha256", "e" * 64)
            with self.assertRaisesRegex(ValueError, "scalar-fused device snapshot identity"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_device_snapshot_sha256", original_snapshot)
            object.__setattr__(context, "v_cycle_core_device_snapshot_sha256", "a" * 64)
            with self.assertRaisesRegex(ValueError, "core device snapshot identity"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_core_device_snapshot_sha256", original_core_snapshot)
            object.__setattr__(context, "v_cycle_kernel_launches", original_launches + 1)
            with self.assertRaisesRegex(ValueError, "V-cycle launch count"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_kernel_launches", original_launches)
            object.__setattr__(context, "v_cycle_core_kernel_launches", original_core_launches + 1)
            with self.assertRaisesRegex(ValueError, "full/core launch counts|core launch count"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_core_kernel_launches", original_core_launches)
            object.__setattr__(context, "v_cycle_publication_version", original_publication_version + "-forged")
            with self.assertRaisesRegex(ValueError, "outer kernel schedule|publication provenance"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_publication_version", original_publication_version)
            object.__setattr__(context, "v_cycle_root_ingress_zero_start_fusions", 0)
            with self.assertRaisesRegex(ValueError, "root ingress fusion count"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_root_ingress_zero_start_fusions", original_root_fusions)
            object.__setattr__(context, "capture_binding", None)
            with self.assertRaisesRegex(ValueError, "fields do not match their solver-issued receipt"):
                dataclasses.replace(self.endpoint)
        finally:
            object.__setattr__(context, "v_cycle_schedule_sha256", original_schedule)
            object.__setattr__(context, "v_cycle_core_schedule_sha256", original_core_schedule)
            object.__setattr__(context, "v_cycle_static_device_content_sha256", original_static)
            object.__setattr__(context, "v_cycle_device_snapshot_sha256", original_snapshot)
            object.__setattr__(context, "v_cycle_core_device_snapshot_sha256", original_core_snapshot)
            object.__setattr__(context, "v_cycle_kernel_launches", original_launches)
            object.__setattr__(context, "v_cycle_core_kernel_launches", original_core_launches)
            object.__setattr__(context, "v_cycle_publication_version", original_publication_version)
            object.__setattr__(context, "v_cycle_root_ingress_zero_start_fusions", original_root_fusions)
            object.__setattr__(context, "capture_binding", original_capture_binding)

    def test_nested_vcycle_signed_zero_correction_tamper_is_rejected(self):
        solver = CapturedDirectGraphVBD(
            self.scene,
            device="cuda:0",
            config=DirectGraphVBDConfig(minimum_determinant=2.0),
        )
        solver.capture_graphs(warmup_replays=1)
        endpoint = solver.run(graph_replay=True)
        outer = endpoint.outer_work[1]

        for cycle_index, record in enumerate(outer.v_cycles):
            with self.subTest(cycle=cycle_index):
                correction = np.array(record.correction, copy=True)
                flat = correction.reshape(-1)
                zero_indices = np.flatnonzero(flat == 0.0)
                self.assertGreater(zero_indices.size, 0)
                index = int(zero_indices[0])
                flat[index] = np.copysign(0.0, 1.0 if np.signbit(flat[index]) else -1.0)
                self.assertTrue(np.array_equal(correction, record.correction))
                self.assertNotEqual(correction.tobytes(), record.correction.tobytes())

                forged_record = dataclasses.replace(record)
                object.__setattr__(forged_record, "correction", correction)
                forged_cycles = list(outer.v_cycles)
                forged_cycles[cycle_index] = forged_record
                with self.assertRaisesRegex(ValueError, "retained correction bytes"):
                    dataclasses.replace(outer, v_cycles=tuple(forged_cycles))

    def test_pristine_k1_reset_source_tamper_fails_before_execution(self):
        source = self.solver.baseline.pristine_input.particle_qd
        pristine = np.asarray(source.numpy(), dtype=np.float32)
        try:
            source.assign(np.full_like(pristine, 2.0))
            with self.assertRaisesRegex(RuntimeError, "pristine input state"):
                self.solver.run(graph_replay=True)
        finally:
            source.assign(pristine)
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_static_device_hierarchy_tamper_fails_before_execution(self):
        values = self.solver.device_hierarchy.levels[0].matrix_values
        pristine = np.asarray(values.numpy(), dtype=np.float64)
        try:
            values.assign(np.zeros_like(pristine))
            with self.assertRaisesRegex(RuntimeError, "hierarchy.level_0.matrix_values"):
                self.solver.run(graph_replay=True)
        finally:
            values.assign(pristine)
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_scalar_fused_owner_and_identity_labels_fail_closed(self):
        self.assertIs(type(self.solver.device_hierarchy), WarpScalarFusedStaticMultigridHierarchy)
        self.assertIs(self.solver.device_hierarchy.source_hierarchy, self.solver.source_device_hierarchy)
        fields = (
            "_schedule_sha256",
            "_static_device_content_sha256",
            "_device_snapshot_sha256",
        )
        for field in fields:
            with self.subTest(field=field):
                original = getattr(self.solver.device_hierarchy, field)
                try:
                    setattr(self.solver.device_hierarchy, field, "a" * 64)
                    with self.assertRaisesRegex(RuntimeError, "workspace identity|scalar-fused hierarchy"):
                        self.solver.deterministic_record()
                finally:
                    setattr(self.solver.device_hierarchy, field, original)

        original_source_snapshot = self.solver.source_device_hierarchy.device_snapshot_sha256
        try:
            self.solver.source_device_hierarchy.device_snapshot_sha256 = "b" * 64
            with self.assertRaisesRegex(RuntimeError, "hierarchy metadata|scalar-fused hierarchy"):
                self.solver.deterministic_record()
        finally:
            self.solver.source_device_hierarchy.device_snapshot_sha256 = original_source_snapshot

        original_wrapper = self.solver.device_hierarchy
        replacement = WarpScalarFusedStaticMultigridHierarchy.from_device_hierarchy(self.solver.source_device_hierarchy)
        try:
            self.solver.device_hierarchy = replacement
            with self.assertRaisesRegex(RuntimeError, "hierarchy owner object|workspace owner object"):
                self.solver.run(graph_replay=True)
        finally:
            self.solver.device_hierarchy = original_wrapper

    def test_scalar_fused_alt_buffers_are_pointer_bound_and_named(self):
        signatures = dict(self.solver._persistent_array_signatures())
        self.assertTrue(any("level_correction_alt" in name for name in signatures))
        self.assertFalse(any("level_product" in name for name in signatures))
        cycle = self.solver.workspaces[0].first_cycle
        self.assertIs(type(cycle), WarpScalarFusedVCycleWorkspace)
        original = cycle.level_correction_alt
        replacement = wp.empty(
            original[0].shape,
            dtype=wp.float64,
            device=self.solver.device,
        )
        try:
            cycle.level_correction_alt = (replacement, *original[1:])
            with self.assertRaisesRegex(
                RuntimeError,
                "workspace persistent array pointers|allocation or pointer|level_correction_alt container|final_scalar_correction owner",
            ):
                self.solver.deterministic_record()
        finally:
            cycle.level_correction_alt = original

    def test_shared_publications_reject_partial_alias_pointer_and_cycle_role_swap(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        outer_binding = binding.outer[0]
        for cycle_name, output_name in (("first_cycle", "first_correction"), ("second_cycle", "second_correction")):
            final_scalar = getattr(outer_binding, cycle_name).final_scalar_correction
            aliased_output = wp.array(
                ptr=int(final_scalar.ptr) + 8,
                dtype=wp.vec3d,
                shape=(self.solver.operator.n_free,),
                device=self.solver.device,
                copy=False,
            )
            with self.subTest(cycle=cycle_name):
                with self.assertRaisesRegex(
                    RuntimeError, f"{cycle_name.replace('_', '-')} final scalar aliases {output_name}"
                ):
                    self.solver._validate_external_publication_aliases(
                        outer_binding._replace(**{output_name: aliased_output}),
                        name="forged outer",
                    )

        first_scalar = outer_binding.first_cycle.final_scalar_correction
        aliased_vector = wp.array(
            ptr=int(first_scalar.ptr) + 8,
            dtype=wp.vec3d,
            shape=(self.solver.operator.n_free,),
            device=self.solver.device,
            copy=False,
        )
        for field_name in ("rhs", "operator_product_after_first", "residual_after_first"):
            with self.subTest(first_scalar_alias=field_name):
                with self.assertRaisesRegex(RuntimeError, f"first-cycle final scalar aliases {field_name}"):
                    self.solver._validate_external_publication_aliases(
                        outer_binding._replace(**{field_name: aliased_vector}),
                        name="forged outer",
                    )

        aliased_delta_piola = wp.array(
            ptr=int(first_scalar.ptr) + 8,
            dtype=wp.mat33d,
            shape=(self.solver.operator.n_tets,),
            device=self.solver.device,
            copy=False,
        )
        with self.assertRaisesRegex(RuntimeError, "first-cycle final scalar aliases operator_apply.delta_piola"):
            self.solver._validate_external_publication_aliases(
                outer_binding._replace(operator_apply_delta_piola=aliased_delta_piola),
                name="forged outer",
            )

        forged_first_cycle = outer_binding.first_cycle._replace(
            final_scalar_pointer=outer_binding.first_cycle.final_scalar_pointer + 8
        )
        with self.assertRaisesRegex(RuntimeError, "scalar-fused workspace owner object changed"):
            self.solver._validate_cycle_workspace_owner_binding(
                outer_binding.first_cycle.workspace,
                forged_first_cycle,
                name="forged first cycle",
            )

        outer = self.endpoint.outer_work[0]
        cycle_workspaces = (
            self.solver.workspaces[0].first_cycle,
            self.solver.workspaces[0].second_cycle,
        )
        for cycle_index, cycle_workspace in enumerate(cycle_workspaces):
            full_record = cycle_workspace.record_internal_application(capture_replay=True)
            forged_cycles = list(outer.v_cycles)
            forged_cycles[cycle_index] = full_record
            with self.subTest(full_publication_route=cycle_index):
                with self.assertRaisesRegex(ValueError, "scheduled launch count is stale"):
                    dataclasses.replace(outer, v_cycles=tuple(forged_cycles))
        with self.assertRaisesRegex(ValueError, "linear_kernel_launches"):
            dataclasses.replace(outer, linear_kernel_launches=outer.linear_kernel_launches + 1)
        with self.assertRaisesRegex(ValueError, "retained correction does not exactly bind"):
            dataclasses.replace(outer, v_cycles=(outer.v_cycles[1], outer.v_cycles[0]))
        with self.assertRaisesRegex(ValueError, "kernel schedule identity"):
            dataclasses.replace(
                outer,
                first_cycle_publication_route=SECOND_CYCLE_PUBLICATION_ROLE,
                second_cycle_publication_route=FIRST_CYCLE_PUBLICATION_ROLE,
            )

    def test_workspace_container_swap_fails_at_every_public_boundary(self):
        original = self.solver.workspaces
        for name, operation in self._owner_boundaries(self.solver):
            with self.subTest(boundary=name):
                try:
                    self.solver.workspaces = list(original)
                    with self.assertRaisesRegex(RuntimeError, "workspace tuple container"):
                        operation()
                finally:
                    self.solver.workspaces = original

    def test_outer_position_container_attacks_fail_at_every_public_boundary(self):
        class StatefulPositionSequence:
            def __init__(self, canonical, redirected):
                self.canonical = canonical
                self.redirected = redirected
                self.reads = 0

            def __len__(self):
                self.reads += 1
                return len(self.canonical)

            def __iter__(self):
                self.reads += 1
                return iter(self.canonical if self.reads < 4 else self.redirected)

            def __getitem__(self, index):
                self.reads += 1
                values = self.canonical if self.reads < 4 else self.redirected
                return values[index]

        for field_name in ("outer_start_positions", "outer_candidate_positions"):
            original = getattr(self.solver, field_name)
            alternate = wp.empty(original[0].shape, dtype=wp.vec3d, device=self.solver.device)
            redirected = (alternate, *original[1:])
            attacks = (
                ("list", lambda values=original: list(values)),
                ("shallow-tuple", lambda values=original: tuple(iter(values))),
                (
                    "stateful",
                    lambda canonical=original, alternate_values=redirected: StatefulPositionSequence(
                        canonical,
                        alternate_values,
                    ),
                ),
            )
            for attack_name, make_attack in attacks:
                for boundary_name, operation in self._owner_boundaries(self.solver):
                    with self.subTest(field=field_name, attack=attack_name, boundary=boundary_name):
                        attack = make_attack()
                        try:
                            setattr(self.solver, field_name, attack)
                            with self.assertRaisesRegex(RuntimeError, f"{field_name} tuple container or order"):
                                operation()
                            if isinstance(attack, StatefulPositionSequence):
                                self.assertEqual(attack.reads, 0)
                        finally:
                            setattr(self.solver, field_name, original)

    def test_particle_color_group_container_attacks_fail_for_k1_k4_and_all_boundaries(self):
        class StatefulColorGroups(list):
            def __init__(self, canonical, redirected):
                super().__init__(canonical)
                self.canonical = canonical
                self.redirected = redirected
                self.reads = 0

            def __len__(self):
                self.reads += 1
                return super().__len__()

            def __iter__(self):
                self.reads += 1
                return iter(self.canonical if self.reads < 4 else self.redirected)

            def __getitem__(self, index):
                self.reads += 1
                values = self.canonical if self.reads < 4 else self.redirected
                return values[index]

        model = self.solver.baseline.model
        original = model.particle_color_groups
        alternate = wp.array(
            np.asarray(original[0].numpy(), dtype=np.int32)[::-1].copy(),
            dtype=wp.int32,
            device=self.solver.device,
        )
        redirected = [alternate, *original[1:]]
        attacks = (
            ("shallow-list", lambda: list(original)),
            ("stateful-list", lambda: StatefulColorGroups(original, redirected)),
        )
        for attack_name, make_attack in attacks:
            for boundary_name, operation in self._owner_boundaries(self.solver):
                with self.subTest(attack=attack_name, boundary=boundary_name):
                    attack = make_attack()
                    try:
                        model.particle_color_groups = attack
                        with self.assertRaisesRegex(RuntimeError, "particle color-group list or order"):
                            operation()
                        if isinstance(attack, StatefulColorGroups):
                            self.assertEqual(attack.reads, 0)
                    finally:
                        model.particle_color_groups = original

    def test_coordinated_hierarchy_level_container_attacks_fail_at_all_boundaries(self):
        class StatefulLevels:
            def __init__(self, canonical, redirected):
                self.canonical = canonical
                self.redirected = redirected
                self.reads = 0

            def __len__(self):
                self.reads += 1
                return len(self.canonical)

            def __iter__(self):
                self.reads += 1
                return iter(self.canonical if self.reads < 5 else self.redirected)

            def __getitem__(self, index):
                self.reads += 1
                values = self.canonical if self.reads < 5 else self.redirected
                return values[index]

        source = self.solver.source_device_hierarchy
        scalar = self.solver.device_hierarchy
        original = source.levels
        alternate_values = wp.empty(
            original[0].matrix_values.shape,
            dtype=wp.float64,
            device=self.solver.device,
        )
        redirected_level = dataclasses.replace(original[0], matrix_values=alternate_values)
        redirected = (redirected_level, *original[1:])
        attacks = (
            ("list", lambda: list(original)),
            ("shallow-tuple", lambda: tuple(iter(original))),
            ("stateful", lambda: StatefulLevels(original, redirected)),
        )
        for attack_name, make_attack in attacks:
            for boundary_name, operation in self._owner_boundaries(self.solver):
                with self.subTest(attack=attack_name, boundary=boundary_name):
                    attack = make_attack()
                    try:
                        source.levels = attack
                        scalar._levels = attack
                        with self.assertRaisesRegex(RuntimeError, "hierarchy levels tuple container"):
                            operation()
                        if isinstance(attack, StatefulLevels):
                            self.assertEqual(attack.reads, 0)
                    finally:
                        source.levels = original
                        scalar._levels = original

    def test_each_hierarchy_level_array_owner_is_construction_bound(self):
        fields = (
            "row_offsets",
            "column_indices",
            "matrix_values",
            "inverse_diagonal",
            "aggregate",
            "prolongation_blocks",
            "member_offsets",
            "member_fine_nodes",
        )
        for level_index, level in enumerate(self.solver.source_device_hierarchy.levels):
            for field_name in fields:
                original = getattr(level, field_name)
                if original is None:
                    continue
                with self.subTest(level=level_index, field=field_name):
                    replacement = wp.empty(
                        original.shape,
                        dtype=original.dtype,
                        device=self.solver.device,
                    )
                    try:
                        object.__setattr__(level, field_name, replacement)
                        with self.assertRaisesRegex(RuntimeError, f"level {level_index} {field_name} owner"):
                            self.solver.deterministic_record()
                    finally:
                        object.__setattr__(level, field_name, original)

    def test_outer_proxy_and_shallow_clone_cannot_redirect_recapture(self):
        original_container = self.solver.workspaces
        original = original_container[0]
        alternate = wp.empty(
            original.first_correction.shape,
            dtype=wp.vec3d,
            device=self.solver.device,
        )

        class StatefulOuterProxy:
            def __init__(self, canonical, redirected):
                self.canonical = canonical
                self.redirected = redirected
                self.first_correction_reads = 0

            def __getattr__(self, name):
                if name == "first_correction":
                    self.first_correction_reads += 1
                    if self.first_correction_reads > 2:
                        return self.redirected
                return getattr(self.canonical, name)

        proxy = StatefulOuterProxy(original, alternate)
        graph = self.solver.graph
        k4_graph = self.solver.k4_graph
        try:
            self.solver.workspaces = (proxy, *original_container[1:])
            with self.assertRaisesRegex(RuntimeError, "workspace tuple container"):
                self.solver.capture_graphs(warmup_replays=1)
            self.assertEqual(proxy.first_correction_reads, 0)
            self.assertIs(self.solver.graph, graph)
            self.assertIs(self.solver.k4_graph, k4_graph)
        finally:
            self.solver.workspaces = original_container

        shallow_outer = dataclasses.replace(original)
        try:
            self.solver.workspaces = (shallow_outer, *original_container[1:])
            with self.assertRaisesRegex(RuntimeError, "workspace tuple container"):
                self.solver.deterministic_record()
        finally:
            self.solver.workspaces = original_container

    def test_stateful_registry_dispatch_cannot_redirect_any_public_boundary(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        solver = CapturedDirectGraphVBD(self.scene, device="cuda:0")
        solver.capture_graphs(warmup_replays=1)
        expected_positions = solver.run(graph_replay=True).positions
        original_lookup = module._lookup_workspace_owners
        canonical = original_lookup(solver)
        alternate = wp.empty(
            canonical.outer[0].first_correction.shape,
            dtype=wp.vec3d,
            device=solver.device,
        )
        sentinel = np.full((alternate.shape[0], 3), 734.25, dtype=np.float64)
        alternate.assign(sentinel)
        registry = next(
            cell.cell_contents
            for cell in original_lookup.__closure__ or ()
            if isinstance(cell.cell_contents, weakref.WeakKeyDictionary)
        )

        for dispatch_kind in ("module-global", "closure-registry-get"):
            for boundary_name, operation in self._owner_boundaries(solver):
                with self.subTest(dispatch=dispatch_kind, boundary=boundary_name):
                    canonical = original_lookup(solver)
                    forged_outer = canonical.outer[0]._replace(first_correction=alternate)
                    forged = canonical._replace(outer=(forged_outer, *canonical.outer[1:]))
                    calls = 0

                    def stateful_lookup(
                        owner,
                        default=None,
                        canonical_binding=canonical,
                        forged_binding=forged,
                    ):
                        nonlocal calls
                        calls += 1
                        if owner is not solver:
                            return original_lookup(owner)
                        return canonical_binding if calls == 1 else forged_binding

                    alternate.assign(sentinel)
                    try:
                        if dispatch_kind == "module-global":
                            module._lookup_workspace_owners = stateful_lookup
                        else:
                            registry.get = stateful_lookup
                        operation()
                        self.assertEqual(calls, 1)
                        np.testing.assert_array_equal(np.asarray(alternate.numpy(), dtype=np.float64), sentinel)
                    finally:
                        if dispatch_kind == "module-global":
                            module._lookup_workspace_owners = original_lookup
                        elif "get" in vars(registry):
                            del registry.get
                    if boundary_name == "recapture":
                        published = original_lookup(solver)
                        self.assertIsNot(published, canonical)
                        self.assertIs(published.persistent, canonical.persistent)
                        self.assertIs(published.claims, canonical.claims)
                        self.assertEqual(published.capture.generation, canonical.capture.generation + 1)
        np.testing.assert_array_equal(solver.run(graph_replay=True).positions, expected_positions)

    def test_cycle_owner_and_all_tuple_container_clones_fail_closed(self):
        cycles = tuple(
            cycle for workspace in self.solver.workspaces for cycle in (workspace.first_cycle, workspace.second_cycle)
        )
        for cycle_index, cycle in enumerate(cycles):
            for field_name in (
                "level_rhs",
                "level_correction",
                "level_correction_alt",
                "level_residual",
                "_persistent_arrays",
                "_persistent_pointers",
            ):
                with self.subTest(cycle=cycle_index, container=field_name):
                    original = getattr(cycle, field_name)
                    try:
                        setattr(cycle, field_name, list(original))
                        with self.assertRaisesRegex(RuntimeError, f"{field_name} container"):
                            self.solver.deterministic_record()
                    finally:
                        setattr(cycle, field_name, original)

        original_cycle = self.solver.workspaces[0].first_cycle
        shallow_cycle = object.__new__(WarpScalarFusedVCycleWorkspace)
        for slot in WarpScalarFusedVCycleWorkspace.__slots__:
            setattr(shallow_cycle, slot, getattr(original_cycle, slot))
        try:
            self.solver.workspaces[0].first_cycle = shallow_cycle
            with self.assertRaisesRegex(RuntimeError, "scalar-fused workspace owner object"):
                self.solver.run(graph_replay=True)
        finally:
            self.solver.workspaces[0].first_cycle = original_cycle

        foreign_cycle = SimpleNamespace(
            **{slot: getattr(original_cycle, slot) for slot in WarpScalarFusedVCycleWorkspace.__slots__}
        )
        try:
            self.solver.workspaces[0].first_cycle = foreign_cycle
            with self.assertRaisesRegex(RuntimeError, "scalar-fused workspace owner object"):
                self.solver.run_k4(graph_replay=True)
        finally:
            self.solver.workspaces[0].first_cycle = original_cycle

    def test_cycle_arrays_and_operator_apply_owner_clones_fail_closed(self):
        workspace = self.solver.workspaces[0]
        cycle = workspace.first_cycle
        for field_name in ("rhs", "correction", "coarse_intermediate"):
            with self.subTest(cycle_array=field_name):
                original = getattr(cycle, field_name)
                replacement = wp.empty(
                    original.shape,
                    dtype=original.dtype,
                    device=self.solver.device,
                )
                try:
                    setattr(cycle, field_name, replacement)
                    with self.assertRaisesRegex(RuntimeError, f"{field_name} owner"):
                        self.solver.deterministic_record()
                finally:
                    setattr(cycle, field_name, original)

        original_apply = workspace.operator_apply
        shallow_apply = object.__new__(WarpMatrixFreeWorkspace)
        shallow_apply.__dict__.update(original_apply.__dict__)
        try:
            workspace.operator_apply = shallow_apply
            with self.assertRaisesRegex(RuntimeError, "matrix-free apply workspace owner"):
                self.solver.benchmark_paired(pair_count=2, warmup_replays=1)
        finally:
            workspace.operator_apply = original_apply

        foreign_apply = SimpleNamespace(**original_apply.__dict__)
        try:
            workspace.operator_apply = foreign_apply
            with self.assertRaisesRegex(RuntimeError, "matrix-free apply workspace owner"):
                self.solver.capture_graphs(warmup_replays=1)
        finally:
            workspace.operator_apply = original_apply

    def test_operator_and_endpoint_source_tamper_fail_before_execution(self):
        cases = (
            ("mass", self.solver.operator.mass, "operator.mass"),
            ("canonical positions", self.solver.canonical_positions, "canonical_positions"),
            ("x current", self.solver.x_current, "x_current"),
        )
        for name, array, message in cases:
            with self.subTest(source=name):
                pristine = np.asarray(array.numpy()).copy()
                try:
                    array.assign(pristine + 1.0)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.solver.run(graph_replay=True)
                finally:
                    array.assign(pristine)

    def test_every_direct_correction_array_owner_is_construction_bound(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver).direct
        for field_name in binding._fields:
            if field_name in ("outer_start_positions", "outer_candidate_positions"):
                continue
            with self.subTest(field=field_name):
                original = getattr(self.solver, field_name)
                replacement = wp.empty(original.shape, dtype=original.dtype, device=self.solver.device)
                try:
                    setattr(self.solver, field_name, replacement)
                    with self.assertRaisesRegex(RuntimeError, f"direct correction {field_name} owner"):
                        self.solver.deterministic_record()
                finally:
                    setattr(self.solver, field_name, original)

    def test_public_model_and_config_tamper_fail_before_execution(self):
        model_mass = self.solver.baseline.model.particle_mass
        pristine_mass = np.asarray(model_mass.numpy()).copy()
        try:
            model_mass.assign(pristine_mass + 1.0)
            with self.assertRaisesRegex(RuntimeError, "public static model"):
                self.solver.run(graph_replay=True)
        finally:
            model_mass.assign(pristine_mass)

        pristine_config = self.solver.config
        try:
            self.solver.config = dataclasses.replace(pristine_config, armijo=2.0e-4)
            with self.assertRaisesRegex(RuntimeError, "scene or configuration identity"):
                self.solver.deterministic_record()
        finally:
            self.solver.config = pristine_config

    def test_persistent_array_pointer_replacement_fails_before_execution(self):
        pristine_array = self.solver.operator.mass
        replacement = wp.array(
            np.asarray(pristine_array.numpy(), dtype=np.float64),
            dtype=wp.float64,
            device=self.solver.device,
        )
        try:
            self.solver.operator.mass = replacement
            with self.assertRaisesRegex(RuntimeError, "allocation or pointer"):
                self.solver.run(graph_replay=True)
        finally:
            self.solver.operator.mass = pristine_array

    def test_complete_persistent_array_binding_defeats_recomputed_cache_at_all_boundaries(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        persistent = binding.persistent
        self.assertIsNotNone(persistent)
        self.assertEqual(len(persistent.arrays), 384)
        self.assertEqual(len(persistent.signatures), 384)
        self.assertIs(self.solver._persistent_array_identity, persistent.signatures)
        for (array_name, array), (signature_name, _signature) in zip(
            persistent.arrays,
            persistent.signatures,
            strict=True,
        ):
            self.assertEqual(array_name, signature_name)
            self.assertIs(dict(self.solver._persistent_input_arrays(binding))[array_name], array)

        original_identity = self.solver._persistent_array_identity
        for iterations in (1, 4):
            lane_solver = self.solver.baseline._lane(iterations).solver
            original = lane_solver.particle_displacements
            alternate = wp.empty(original.shape, dtype=original.dtype, device=self.solver.device)
            for boundary_name, operation in self._owner_boundaries(self.solver):
                with self.subTest(iterations=iterations, boundary=boundary_name):
                    try:
                        lane_solver.particle_displacements = alternate
                        self.solver._persistent_array_identity = self.solver._persistent_array_signatures(binding)
                        with self.assertRaisesRegex(
                            RuntimeError,
                            rf"persistent array owner lane_{iterations}\.solver\.particle_displacements",
                        ):
                            operation()
                    finally:
                        lane_solver.particle_displacements = original
                        self.solver._persistent_array_identity = original_identity
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_each_lane_adjacency_content_tamper_fails_before_execution(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                adjacency = self.solver.baseline._lane(iterations).solver.particle_adjacency
                source = adjacency.v_adj_tets
                pristine = np.asarray(source.numpy(), dtype=np.int32).copy()
                tampered = pristine.copy()
                tampered[0] = np.int32(tampered[0] + 1)
                try:
                    source.assign(tampered)
                    operation = self.solver.run if iterations == 1 else self.solver.run_k4
                    with self.assertRaisesRegex(RuntimeError, "adjacency"):
                        operation(graph_replay=True)
                finally:
                    source.assign(pristine)
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_lane_snapshot_names_every_required_adjacency_and_scratch_array(self):
        signatures = dict(self.solver._persistent_array_signatures())
        adjacency_names = (
            "v_adj_faces",
            "v_adj_faces_offsets",
            "v_adj_edges",
            "v_adj_edges_offsets",
            "v_adj_springs",
            "v_adj_springs_offsets",
            "v_adj_tets",
            "v_adj_tets_offsets",
        )
        scratch_names = (
            "particle_q_prev",
            "inertia",
            "particle_displacements",
            "pos_prev_collision_detection",
            "truncation_ts",
            "particle_forces",
            "particle_hessians",
        )
        for iterations in (1, 4):
            for name in scratch_names:
                self.assertIn(f"lane_{iterations}.solver.{name}", signatures)
            for name in adjacency_names:
                key = f"lane_{iterations}.solver.particle_adjacency.{name}"
                self.assertIn(key, signatures)
                array = getattr(self.solver.baseline._lane(iterations).solver.particle_adjacency, name)
                if array.size == 0:
                    self.assertIsNone(array.ptr)
                    self.assertEqual(signatures[key][1], 0)

    def test_each_lane_adjacency_pointer_tamper_blocks_serialization(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                adjacency = self.solver.baseline._lane(iterations).solver.particle_adjacency
                source = adjacency.v_adj_tets
                replacement = wp.array(
                    np.asarray(source.numpy(), dtype=np.int32),
                    dtype=wp.int32,
                    device=self.solver.device,
                )
                try:
                    adjacency.v_adj_tets = replacement
                    with self.assertRaisesRegex(RuntimeError, "allocation or pointer"):
                        self.solver.deterministic_record()
                finally:
                    adjacency.v_adj_tets = source
        json.dumps(self.solver.deterministic_record(), allow_nan=False)

    def test_each_lane_adjacency_cached_ctype_data_tamper_fails_closed(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                adjacency = self.solver.baseline._lane(iterations).solver.particle_adjacency
                descriptor = adjacency._ctype.v_adj_tets
                original_data = descriptor.data
                try:
                    descriptor.data = int(adjacency.v_adj_faces.ptr)
                    with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                        if iterations == 1:
                            self.solver.run(graph_replay=True)
                        else:
                            self.solver.capture_graphs(warmup_replays=1)
                    if iterations == 4:
                        with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                            self.solver.benchmark_paired(pair_count=2, warmup_replays=1)
                        with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                            self.solver.deterministic_record()
                finally:
                    descriptor.data = original_data

    def test_each_lane_adjacency_cached_ctype_shape_tamper_fails_closed(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                descriptor = self.solver.baseline._lane(iterations).solver.particle_adjacency._ctype.v_adj_tets
                original_shape = descriptor.shape[0]
                try:
                    descriptor.shape[0] = original_shape + 1
                    with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                        if iterations == 1:
                            self.solver.deterministic_record()
                        else:
                            self.solver.run_k4(graph_replay=True)
                finally:
                    descriptor.shape[0] = original_shape

    def test_each_lane_adjacency_cached_ctype_stride_tamper_fails_closed(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                descriptor = self.solver.baseline._lane(iterations).solver.particle_adjacency._ctype.v_adj_tets
                original_stride = descriptor.strides[0]
                try:
                    descriptor.strides[0] = original_stride + 4
                    with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                        if iterations == 1:
                            self.solver.run(graph_replay=True)
                        else:
                            self.solver.benchmark_paired(pair_count=2, warmup_replays=1)
                finally:
                    descriptor.strides[0] = original_stride

    def test_direct_array_fake_same_layout_cached_ctype_fails_closed(self):
        source = self.solver.operator.mass
        original = source.__ctype__()

        class FakeArrayDescriptor(ctypes.Structure):
            _fields_ = type(original)._fields_

        fake = FakeArrayDescriptor()
        fake.data = original.data
        fake.grad = original.grad
        fake.ndim = original.ndim
        for index in range(4):
            fake.shape[index] = original.shape[index]
            fake.strides[index] = original.strides[index]
        try:
            source.ctype = fake
            with self.assertRaisesRegex(RuntimeError, "type or field layout"):
                self.solver.deterministic_record()
        finally:
            source.ctype = original

    def test_each_lane_scratch_pointer_tamper_fails_run_and_timing(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                lane_solver = self.solver.baseline._lane(iterations).solver
                source = lane_solver.particle_forces
                replacement = wp.array(
                    np.asarray(source.numpy(), dtype=np.float32),
                    dtype=wp.vec3,
                    device=self.solver.device,
                )
                try:
                    lane_solver.particle_forces = replacement
                    if iterations == 1:
                        with self.assertRaisesRegex(RuntimeError, "allocation or pointer"):
                            self.solver.run(graph_replay=True)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "allocation or pointer"):
                            self.solver.benchmark_paired(pair_count=2, warmup_replays=1)
                finally:
                    lane_solver.particle_forces = source

    def test_k4_iteration_schedule_tamper_fails_before_recapture(self):
        k4_solver = self.solver.baseline._lane(4).solver
        original_iterations = k4_solver.iterations
        try:
            k4_solver.iterations = 1
            with self.assertRaisesRegex(RuntimeError, "K4 solver iteration schedule"):
                self.solver.capture_graphs(warmup_replays=1)
            with self.assertRaisesRegex(RuntimeError, "K4 solver iteration schedule"):
                self.solver.run_k4(graph_replay=True)
        finally:
            k4_solver.iterations = original_iterations
        self.assertEqual(
            self.solver.run_k4(graph_replay=True).endpoint_sha256,
            self.solver._construction_k4.endpoint_sha256,
        )

    def test_record_path_requires_an_unconsumed_private_execution(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        self.assertFalse(hasattr(self.solver, "record"))
        self.assertFalse(hasattr(module, "_EXECUTION_TOKEN"))
        self.solver._pending_execution = (False, 999)
        try:
            with self.assertRaisesRegex(RuntimeError, "exact solver-issued execution receipt"):
                self.solver._record(execution_receipt=object())
        finally:
            del self.solver._pending_execution
        graph = self.solver.graph
        try:
            self.solver.graph = self.solver.k4_graph
            with self.assertRaisesRegex(RuntimeError, "captured graph object"):
                self.solver.run(graph_replay=True)
        finally:
            self.solver.graph = graph

    def test_solver_graph_identity_labels_cannot_be_reassigned(self):
        fields = (
            "_uncaptured_graph_identity_sha256",
            "graph_identity_sha256",
            "k4_graph_identity_sha256",
        )
        for field in fields:
            with self.subTest(field=field):
                original = getattr(self.solver, field)
                try:
                    setattr(self.solver, field, "a" * 64)
                    with self.assertRaisesRegex(RuntimeError, "identity label"):
                        self.solver.deterministic_record()
                finally:
                    setattr(self.solver, field, original)

    def test_graph_pair_generation_and_recomputed_facades_fail_at_all_execution_boundaries(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        capture_binding = binding.capture
        self.assertIsNotNone(capture_binding)
        with wp.ScopedCapture(device=self.solver.device) as clone_capture:
            self.solver._enqueue_integrated(binding)
        graph_clone = clone_capture.graph
        self.assertIsNot(graph_clone, capture_binding.graph)

        original_facade = (
            self.solver.graph,
            self.solver.k4_graph,
            self.solver._captured_graph_object_identity,
            self.solver._capture_generation,
            self.solver.graph_identity_sha256,
            self.solver.k4_graph_identity_sha256,
        )

        def install_forged_facade(graph, k4_graph, generation):
            self.solver.graph = graph
            self.solver.k4_graph = k4_graph
            self.solver._captured_graph_object_identity = (id(graph), id(k4_graph))
            self.solver._capture_generation = generation
            self.solver.graph_identity_sha256 = self.solver._derive_graph_identity(
                captured=True,
                comparator=False,
                owner_binding=binding,
                graph=graph,
                generation=generation,
            )
            self.solver.k4_graph_identity_sha256 = self.solver._derive_graph_identity(
                captured=True,
                comparator=True,
                owner_binding=binding,
                graph=k4_graph,
                generation=generation,
            )

        def install_fresh_equal_object_identity():
            install_forged_facade(
                capture_binding.graph,
                capture_binding.k4_graph,
                capture_binding.generation,
            )
            self.solver._captured_graph_object_identity = (
                capture_binding.object_identity[0],
                capture_binding.object_identity[1],
            )

        attacks = (
            (
                "integrated-to-k4",
                lambda: install_forged_facade(
                    capture_binding.k4_graph,
                    capture_binding.k4_graph,
                    capture_binding.generation,
                ),
            ),
            (
                "swap-integrated-k4",
                lambda: install_forged_facade(
                    capture_binding.k4_graph,
                    capture_binding.graph,
                    capture_binding.generation,
                ),
            ),
            (
                "integrated-clone",
                lambda: install_forged_facade(
                    graph_clone,
                    capture_binding.k4_graph,
                    capture_binding.generation,
                ),
            ),
            (
                "generation-and-labels",
                lambda: install_forged_facade(
                    capture_binding.graph,
                    capture_binding.k4_graph,
                    capture_binding.generation + 1,
                ),
            ),
            ("fresh-equal-object-identity", install_fresh_equal_object_identity),
        )
        boundaries = (
            ("run", lambda: self.solver.run(graph_replay=True)),
            ("run_k4", lambda: self.solver.run_k4(graph_replay=True)),
            ("timing", lambda: self.solver.benchmark_paired(pair_count=2, warmup_replays=1)),
            ("serialization", self.solver.deterministic_record),
        )
        for attack_name, install_attack in attacks:
            for boundary_name, operation in boundaries:
                with self.subTest(attack=attack_name, boundary=boundary_name):
                    try:
                        install_attack()
                        with self.assertRaisesRegex(RuntimeError, "captured graph object facade or generation"):
                            operation()
                    finally:
                        (
                            self.solver.graph,
                            self.solver.k4_graph,
                            self.solver._captured_graph_object_identity,
                            self.solver._capture_generation,
                            self.solver.graph_identity_sha256,
                            self.solver.k4_graph_identity_sha256,
                        ) = original_facade
        self.assertIs(module._lookup_workspace_owners(self.solver), binding)
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_native_graph_exec_swaps_fail_at_all_launch_and_evidence_boundaries(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        capture_binding = binding.capture
        self.assertIsNotNone(capture_binding)
        graph = capture_binding.graph
        k4_graph = capture_binding.k4_graph
        graph_exec = capture_binding.graph_native.graph_exec
        k4_graph_exec = capture_binding.k4_graph_native.graph_exec
        self.assertIs(graph.graph_exec, graph_exec)
        self.assertIs(k4_graph.graph_exec, k4_graph_exec)
        self.assertNotEqual(graph_exec.value, k4_graph_exec.value)

        attacks = (
            ("integrated-to-k4-exec", lambda: setattr(graph, "graph_exec", k4_graph_exec)),
            ("k4-to-integrated-exec", lambda: setattr(k4_graph, "graph_exec", graph_exec)),
            (
                "swap-both-execs",
                lambda: (
                    setattr(graph, "graph_exec", k4_graph_exec),
                    setattr(k4_graph, "graph_exec", graph_exec),
                ),
            ),
        )
        boundaries = (
            ("run", lambda: self.solver.run(graph_replay=True)),
            ("run_k4", lambda: self.solver.run_k4(graph_replay=True)),
            ("timing", lambda: self.solver.benchmark_paired(pair_count=2, warmup_replays=1)),
            ("serialization", self.solver.deterministic_record),
            ("record", lambda: dataclasses.replace(self.endpoint)),
            ("recapture", lambda: self.solver.capture_graphs(warmup_replays=1)),
        )
        marker = np.asarray(self.solver.final_positions.numpy(), dtype=np.float32).copy()
        for attack_name, install_attack in attacks:
            for boundary_name, operation in boundaries:
                with self.subTest(attack=attack_name, boundary=boundary_name):
                    try:
                        install_attack()
                        self.solver.graph_identity_sha256 = self.solver._graph_identity(
                            captured=True,
                            comparator=False,
                            owner_binding=binding,
                        )
                        self.solver.k4_graph_identity_sha256 = self.solver._graph_identity(
                            captured=True,
                            comparator=True,
                            owner_binding=binding,
                        )
                        with self.assertRaisesRegex(RuntimeError, "native graph_exec owner or handle value"):
                            operation()
                        np.testing.assert_array_equal(
                            np.asarray(self.solver.final_positions.numpy(), dtype=np.float32),
                            marker,
                        )
                    finally:
                        graph.graph_exec = graph_exec
                        k4_graph.graph_exec = k4_graph_exec

    def test_all_native_graph_and_replay_stream_fields_are_bound(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        capture_binding = binding.capture
        self.assertIsNotNone(capture_binding)
        graph_native = capture_binding.graph_native
        graph = capture_binding.graph
        replay = binding.replay_stream
        cases = (
            (
                "graph-handle-object",
                graph,
                "graph",
                capture_binding.k4_graph_native.graph_handle,
                "native graph owner or handle value",
            ),
            ("capture-id", graph, "capture_id", graph.capture_id + 1, "capture ID"),
            ("module-execs", graph, "module_execs", set(graph.module_execs), "module-exec set"),
            (
                "stream-handle",
                replay.stream,
                "cuda_stream",
                int(self.solver.device.stream.cuda_stream),
                "replay stream owner or native handle",
            ),
        )
        for name, owner, field_name, replacement, message in cases:
            with self.subTest(field=name):
                original = getattr(owner, field_name)
                try:
                    setattr(owner, field_name, replacement)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.solver.deterministic_record()
                finally:
                    setattr(owner, field_name, original)

        original_exec_value = graph_native.graph_exec.value
        try:
            graph_native.graph_exec.value = capture_binding.k4_graph_native.graph_exec_value
            with self.assertRaisesRegex(RuntimeError, "native graph_exec owner or handle value"):
                self.solver.run(graph_replay=True)
        finally:
            graph_native.graph_exec.value = original_exec_value

        original_context = self.solver.device._context
        try:
            self.solver.device._context = original_context + 1
            with self.assertRaisesRegex(RuntimeError, "replay device or context owner"):
                self.solver.deterministic_record()
        finally:
            self.solver.device._context = original_context

    def test_segment_rejection_is_sticky_fail_closed_but_keeps_fixed_work(self):
        solver = CapturedDirectGraphVBD(
            self.scene,
            device="cuda:0",
            config=DirectGraphVBDConfig(minimum_determinant=2.0),
        )
        solver.capture_graphs(warmup_replays=1)
        endpoint = solver.run(graph_replay=True)
        self.assertEqual(endpoint.accepted, (False, False, False, False))
        self.assertEqual(
            endpoint.reasons,
            ("segment-inversion", "masked-after-rejection", "masked-after-rejection", "masked-after-rejection"),
        )
        np.testing.assert_array_equal(endpoint.positions, endpoint.outer_start_positions[0])
        self.assertEqual(endpoint.initial_objectives[1:], (0.0, 0.0, 0.0))
        self.assertEqual(endpoint.candidate_objectives[1:], (0.0, 0.0, 0.0))
        rejected_state = endpoint.outer_start_positions[0]
        for outer_index, work in enumerate(endpoint.outer_work[1:], start=1):
            np.testing.assert_array_equal(endpoint.outer_start_positions[outer_index], rejected_state)
            np.testing.assert_array_equal(endpoint.outer_candidate_positions[outer_index], rejected_state)
            for name in (
                "rhs",
                "first_correction",
                "operator_product_after_first",
                "residual_after_first",
                "second_correction",
                "direction",
            ):
                np.testing.assert_array_equal(getattr(work, name), 0.0)
        self.assertTrue(endpoint.exact_work_completed)

        masked = endpoint.outer_work[1]
        with self.assertRaisesRegex(ValueError, "exact solver-issued schedule slot"):
            dataclasses.replace(masked, outer_index=0)
        slot_zero = endpoint.outer_work[0]
        with self.assertRaisesRegex(ValueError, "exact solver-issued schedule slot"):
            dataclasses.replace(
                masked,
                outer_index=0,
                start_position_sha256=slot_zero.start_position_sha256,
                current_operator_sha256=slot_zero.current_operator_sha256,
                rhs=slot_zero.rhs,
                first_correction=slot_zero.first_correction,
                operator_product_after_first=slot_zero.operator_product_after_first,
                residual_after_first=slot_zero.residual_after_first,
                second_correction=slot_zero.second_correction,
                direction=slot_zero.direction,
                v_cycles=slot_zero.v_cycles,
                accepted=slot_zero.accepted,
                reason=slot_zero.reason,
                _validation_operator=slot_zero._validation_operator,
            )

    def test_timing_boundary_is_balanced_diagnostic_only(self):
        timing = self.solver.benchmark_paired(pair_count=4, warmup_replays=2, random_seed=4817)
        self.assertEqual(timing.pair_orders.count("AB"), 2)
        self.assertEqual(timing.pair_orders.count("BA"), 2)
        self.assertGreater(timing.graph_median_seconds, 0.0)
        self.assertGreater(timing.k4_median_seconds, 0.0)
        self.assertTrue(timing.integrated_direct_graph)
        self.assertFalse(timing.setup_included)
        self.assertFalse(timing.transfers_included)
        self.assertFalse(timing.performance_evidence)
        self.assertEqual(timing.contract_id, CONTRACT_ID)
        self.assertEqual(timing.scene_sha256, self.solver.scene_sha256)
        self.assertEqual(timing.config_sha256, self.solver.config_sha256)
        self.assertEqual(timing.persistent_device_sha256, self.solver._persistent_device_sha256)
        self.assertEqual(timing.graph_identity_sha256, self.solver.graph_identity_sha256)
        self.assertEqual(timing.k4_graph_identity_sha256, self.solver.k4_graph_identity_sha256)
        record = timing.deterministic_record()
        self.assertEqual(record["fused_gather_kernel_version"], FUSED_GATHER_KERNEL_VERSION)
        self.assertEqual(record["scalar_direction_apply_kernel_version"], SCALAR_DIRECTION_APPLY_KERNEL_VERSION)
        self.assertEqual(record["first_cycle_publication_role"], FIRST_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(record["second_cycle_publication_role"], SECOND_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(record["v_cycle_publication_version"], V_CYCLE_PUBLICATION_VERSION)
        self.assertEqual(record["correction_kernel_launches"], self.solver.correction_kernel_launches)
        json.dumps(record, allow_nan=False)

    def test_endpoint_evidence_is_bytes_backed_and_immutable(self):
        with self.assertRaises(ValueError):
            self.endpoint.positions.setflags(write=True)
        with self.assertRaises(ValueError):
            self.endpoint.outer_work[0].direction.setflags(write=True)
        self.assertEqual(len(self.endpoint.outer_start_position_sha256s), OUTER_CORRECTIONS)
        self.assertEqual(len(self.endpoint.outer_candidate_position_sha256s), OUTER_CORRECTIONS)
        self.assertEqual(len(set(self.endpoint.current_operator_sha256s)), OUTER_CORRECTIONS)


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestCapturedDirectGraphVBDDefaultStretchCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.scene = build_stretch_scene()
        cls.problem = build_common_problem(cls.scene)
        cls.solver = CapturedDirectGraphVBD(cls.scene, device="cuda:0")
        cls.solver.capture_graphs(warmup_replays=1)
        cls.endpoint = cls.solver.run(graph_replay=True)
        cls.k4 = cls.solver.run_k4(graph_replay=True)

    def test_real_default_stretch_accepts_four_current_operator_corrections(self):
        endpoint = self.endpoint
        self.assertEqual(endpoint.accepted, (True,) * OUTER_CORRECTIONS)
        self.assertEqual(endpoint.reasons, ("accepted",) * OUTER_CORRECTIONS)
        self.assertEqual(len(set(endpoint.current_operator_sha256s)), OUTER_CORRECTIONS)
        for index in range(OUTER_CORRECTIONS):
            self.assertLess(endpoint.candidate_objectives[index], endpoint.initial_objectives[index])
            self.assertLessEqual(
                endpoint.candidate_objectives[index],
                endpoint.initial_objectives[index]
                + self.solver.config.armijo * endpoint.directional_derivatives[index],
            )
            self.assertGreater(endpoint.segment_minimum_determinants[index], 0.0)

    def test_real_default_stretch_scalar_fused_schedule_and_physical_work_are_exact(self):
        self.assertEqual(self.solver.device_hierarchy.scheduled_kernel_launches, 20)
        self.assertEqual(self.solver.device_hierarchy.core_kernel_launches, 19)
        self.assertEqual(self.solver.linear_prefix_kernel_launches_per_outer, 42)
        self.assertEqual(self.solver.linear_kernel_launches_per_outer, 43)
        self.assertEqual(self.solver.outer_kernel_launches_per_outer, 46)
        self.assertEqual(self.solver.correction_kernel_launches, 186)
        for outer in self.endpoint.outer_work:
            self.assertEqual(outer.linear_kernel_launches, 43)
            for record in outer.v_cycles:
                self.assertEqual(record.work.matrix_block_products, 5058)
                self.assertEqual(record.physical_work.matrix_block_products_executed, 3372)
                self.assertEqual(record.physical_work.matrix_block_products_elided_zero_start, 1686)
                self.assertEqual(record.physical_work.zero_start_block_solves, 184)
                self.assertEqual(record.physical_work.out_of_place_jacobi_block_solves, 184)
                self.assertEqual(record.physical_work.matrix_kernel_launches, 6)
                self.assertEqual(record.physical_work.jacobi_kernel_launches, 6)
                self.assertEqual(record.physical_work.root_ingress_zero_start_fusions, 1)
                self.assertEqual(record.physical_work.scheduled_kernel_launches, 19)
                self.assertEqual(record.physical_work.core_kernel_launches, 19)
                self.assertEqual(record.physical_work.publication_kernel_launches, 0)
                self.assertEqual(record.physical_work.publication_route, V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE)
                self.assertEqual(record.schedule_version, V_CYCLE_SCHEDULE_VERSION)
                self.assertFalse(record.performance_evidence)
        schedule = self.solver.deterministic_record()
        self.assertEqual(schedule["v_cycle_kernel_launches"], 20)
        self.assertEqual(schedule["v_cycle_core_kernel_launches"], 19)
        self.assertEqual(schedule["v_cycle_root_ingress_zero_start_fusions"], 1)
        self.assertEqual(schedule["linear_prefix_kernel_launches_per_outer"], 42)
        self.assertEqual(schedule["fused_gather_kernel_launches_per_outer"], 2)
        self.assertEqual(schedule["fused_vertex_kernel_launches_per_outer"], 1)
        self.assertEqual(schedule["finalize_gate_kernel_launches_per_outer"], 1)
        self.assertEqual(schedule["linear_kernel_launches_per_outer"], 43)
        self.assertEqual(schedule["outer_kernel_launches_per_outer"], 46)
        self.assertEqual(schedule["correction_kernel_launches_excluding_public_k1"], 186)
        self.assertEqual(schedule["fused_gather_kernel_version"], FUSED_GATHER_KERNEL_VERSION)
        self.assertEqual(
            schedule["scalar_direction_apply_kernel_version"],
            SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
        )
        self.assertEqual(schedule["first_cycle_publication_role"], FIRST_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(schedule["second_cycle_publication_role"], SECOND_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(schedule["outer_kernel_version"], OUTER_KERNEL_VERSION)
        self.assertEqual(schedule["outer_schedule_version"], OUTER_SCHEDULE_VERSION)
        self.assertEqual(schedule["outer_schedule_sha256"], OUTER_SCHEDULE_SHA256)
        self.assertEqual(schedule["finalize_gate_route"], FINALIZE_GATE_ROUTE)
        self.assertEqual(schedule["finalize_gate_block_dim"], FINALIZE_GATE_BLOCK_DIM)
        self.assertEqual(schedule["finalize_gate_owner_threads"], list(FINALIZE_GATE_OWNER_THREADS))
        self.assertEqual(schedule["finalize_gate_owner_roles"], list(FINALIZE_GATE_OWNER_ROLES))
        self.assertEqual(schedule["finalize_gate_collective_version"], FINALIZE_GATE_COLLECTIVE_VERSION)
        self.assertEqual(schedule["v_cycle_schedule_version"], V_CYCLE_SCHEDULE_VERSION)
        self.assertEqual(schedule["v_cycle_publication_version"], V_CYCLE_PUBLICATION_VERSION)
        self.assertEqual(schedule["v_cycle_canonical_matrix_block_products"], 5058)
        self.assertEqual(schedule["v_cycle_matrix_block_products_executed"], 3372)
        self.assertEqual(schedule["v_cycle_matrix_block_products_elided_zero_start"], 1686)
        self.assertEqual(schedule["v_cycle_zero_start_block_solves"], 184)
        self.assertEqual(schedule["v_cycle_out_of_place_jacobi_block_solves"], 184)
        self.assertEqual(schedule["v_cycle_matrix_kernel_launches"], 6)
        self.assertEqual(schedule["v_cycle_jacobi_kernel_launches"], 6)

    def test_real_default_stretch_is_safe_exactly_pinned_and_better_than_k4(self):
        endpoint = self.endpoint
        np.testing.assert_array_equal(endpoint.positions[self.scene.pinned_indices], self.scene.pin_targets)
        np.testing.assert_array_equal(endpoint.velocities[self.scene.pinned_indices], 0.0)
        expected_velocity = (
            ((endpoint.positions - self.scene.x_current) * np.float64(1.0 / self.scene.dt))
            .astype(np.float32)
            .astype(np.float64)
        )
        expected_velocity[self.scene.pinned_indices] = 0.0
        np.testing.assert_array_equal(endpoint.velocities, expected_velocity)
        metrics = evaluate_common_state(self.problem, endpoint.positions)
        k4_metrics = evaluate_common_state(self.problem, self.k4.positions)
        self.assertEqual(metrics.inverted_tet_fraction, 0.0)
        self.assertGreater(metrics.determinant_min, 0.0)
        self.assertLess(metrics.relative_residual, k4_metrics.relative_residual)
        self.assertFalse(self.k4.integrated_mg)

    def test_real_default_stretch_paired_timing_is_diagnostic(self):
        timing = self.solver.benchmark_paired(pair_count=4, warmup_replays=2, random_seed=9901)
        record = timing.deterministic_record()
        self.assertGreater(timing.graph_median_seconds, 0.0)
        self.assertGreater(timing.k4_median_seconds, 0.0)
        self.assertFalse(timing.performance_evidence)
        self.assertEqual(record["fused_gather_kernel_version"], FUSED_GATHER_KERNEL_VERSION)
        self.assertEqual(record["scalar_direction_apply_kernel_version"], SCALAR_DIRECTION_APPLY_KERNEL_VERSION)
        self.assertEqual(record["first_cycle_publication_role"], FIRST_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(record["second_cycle_publication_role"], SECOND_CYCLE_PUBLICATION_ROLE)
        self.assertEqual(record["v_cycle_publication_version"], V_CYCLE_PUBLICATION_VERSION)
        self.assertEqual(record["finalize_gate_route"], FINALIZE_GATE_ROUTE)
        self.assertEqual(record["finalize_gate_block_dim"], FINALIZE_GATE_BLOCK_DIM)
        self.assertEqual(record["finalize_gate_owner_threads"], list(FINALIZE_GATE_OWNER_THREADS))
        self.assertEqual(record["finalize_gate_owner_roles"], list(FINALIZE_GATE_OWNER_ROLES))
        self.assertEqual(record["finalize_gate_collective_version"], FINALIZE_GATE_COLLECTIVE_VERSION)
        self.assertEqual(record["correction_kernel_launches"], 186)
        print("CAPTURED_DIRECT_GRAPH_VBD_TIMING=" + json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
