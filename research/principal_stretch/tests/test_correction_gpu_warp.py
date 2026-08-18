# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the research-only Warp matrix-free correction primitives."""

from __future__ import annotations

import ctypes
import inspect
import json
import os
import unittest
from unittest import mock

import numpy as np
import torch
import warp as wp

from research.principal_stretch import correction_gpu_warp as warp_operator_module
from research.principal_stretch import correction_multigrid_warp_scalar_fused as scalar_fused_module
from research.principal_stretch.correction_gpu import MatrixFreeStableNHOperator, solve_fixed_pcg
from research.principal_stretch.correction_gpu_warp import (
    CONTRACT_ID,
    FUSED_GATHER_KERNEL_VERSION,
    KERNEL_VERSION,
    ROOT_SEEDED_GATHER_KERNEL_VERSION,
    SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
    WarpFixedPCGWorkspace,
    WarpMatrixFreeStableNHOperator,
)
from research.principal_stretch.newton_baseline import NewtonProblem, build_newton_problem
from research.principal_stretch.solver_benchmark import build_common_problem
from research.principal_stretch.solver_scenes import build_stretch_scene


@wp.kernel(enable_backward=False)
def _mask_vector(active: wp.array[int], vector: wp.array[wp.vec3d]):
    index = wp.tid()
    if active[0] == 0:
        vector[index] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


@wp.kernel(enable_backward=False)
def _subtract_vectors(
    left: wp.array[wp.vec3d],
    right: wp.array[wp.vec3d],
    output: wp.array[wp.vec3d],
):
    index = wp.tid()
    output[index] = left[index] - right[index]


@wp.kernel(enable_backward=False)
def _publish_scalar_direction(source: wp.array[wp.float64], destination: wp.array[wp.vec3d]):
    index = wp.tid()
    destination[index] = wp.vec3d(source[3 * index], source[3 * index + 1], source[3 * index + 2])


def _shared_vertex_problem() -> NewtonProblem:
    rest = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.4, 0.2, 0.8),
        ),
        dtype=np.float64,
    )
    # Deliberately non-monotone corner ownership stresses sorted shared-vertex
    # gathers independently of the free-vertex ordering.
    tets = np.array(((1, 2, 3, 4), (0, 1, 2, 3), (2, 1, 4, 5)), dtype=np.int64)
    poses = []
    for tet in tets:
        corners = rest[tet]
        rest_matrix = np.stack(
            (corners[1] - corners[0], corners[2] - corners[0], corners[3] - corners[0]),
            axis=1,
        )
        poses.append(np.linalg.inv(rest_matrix))
    return build_newton_problem(
        rest,
        tets,
        np.stack(poses),
        np.array((0.8, 1.1, 1.4, 0.9, 1.3, 1.0), dtype=np.float64),
        np.array((13.0, 29.0, 47.0), dtype=np.float64),
        np.array((41.0, 73.0, 101.0), dtype=np.float64),
        0.061,
        pinned_indices=np.array((0, 4), dtype=np.int64),
        pin_targets=rest[[0, 4]],
        inertial_target=rest
        + np.array(
            (
                (0.0, 0.0, 0.0),
                (0.01, -0.02, 0.005),
                (-0.02, 0.01, 0.015),
                (0.005, 0.012, -0.007),
                (0.0, 0.0, 0.0),
                (-0.014, 0.006, 0.009),
            ),
            dtype=np.float64,
        ),
    )


def _deformed_positions(problem: NewtonProblem) -> np.ndarray:
    positions = problem.rest_q.numpy().copy()
    positions += np.array(
        (
            (0.0, 0.0, 0.0),
            (0.07, 0.03, -0.01),
            (-0.02, -0.05, 0.04),
            (0.03, -0.01, 0.08),
            (0.0, 0.0, 0.0),
            (-0.04, 0.06, 0.02),
        ),
        dtype=np.float64,
    )
    return positions


def _oracle_and_device(device: str) -> tuple[MatrixFreeStableNHOperator, WarpMatrixFreeStableNHOperator]:
    problem = _shared_vertex_problem()
    oracle = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))
    return oracle, WarpMatrixFreeStableNHOperator.from_oracle(oracle, device=device)


def _diagonal_oracle_and_device(
    vector_count: int,
    device: str,
) -> tuple[MatrixFreeStableNHOperator, WarpMatrixFreeStableNHOperator]:
    """Build an inertia-only SPD operator with an exact requested row count."""
    positions = np.zeros((vector_count, 3), dtype=np.float64)
    oracle = MatrixFreeStableNHOperator(
        positions=positions,
        tets=np.array(((0, 1, 2, 3),), dtype=np.int64),
        shape_gradients=np.zeros((1, 4, 3), dtype=np.float64),
        volumes=np.ones(1, dtype=np.float64),
        mass=np.linspace(0.75, 1.75, vector_count, dtype=np.float64),
        mu=np.zeros(1, dtype=np.float64),
        lam=np.zeros(1, dtype=np.float64),
        inertial_target=positions,
        pinned=np.empty(0, dtype=np.int64),
        free=np.arange(vector_count, dtype=np.int64),
        pin_targets=np.empty((0, 3), dtype=np.float64),
        dt=0.2,
    )
    return oracle, WarpMatrixFreeStableNHOperator.from_oracle(oracle, device=device)


def _default_stretch_oracle_and_device(
    device: str,
) -> tuple[MatrixFreeStableNHOperator, WarpMatrixFreeStableNHOperator]:
    scene = build_stretch_scene()
    problem = build_common_problem(scene)
    positions = np.array(scene.x_current, dtype=np.float64, copy=True)
    positions[scene.pinned_indices] = scene.pin_targets
    oracle = MatrixFreeStableNHOperator.from_problem(problem, positions)
    return oracle, WarpMatrixFreeStableNHOperator.from_oracle(oracle, device=device)


def _identity_block_preconditioner(vector_count: int) -> np.ndarray:
    return np.repeat(np.eye(3, dtype=np.float64)[None], vector_count, axis=0)


def _assert_bitwise_equal(test: unittest.TestCase, actual: np.ndarray, expected: np.ndarray) -> None:
    actual_array = np.ascontiguousarray(actual)
    expected_array = np.ascontiguousarray(expected)
    test.assertEqual(actual_array.dtype, expected_array.dtype)
    test.assertEqual(actual_array.shape, expected_array.shape)
    np.testing.assert_array_equal(actual_array.view(np.uint8), expected_array.view(np.uint8))


def _writable_layout_view(device, dtype, length: int, layout: str) -> tuple[wp.array | None, wp.array]:
    element_size = wp.types.type_size_in_bytes(dtype)
    alignment = ctypes.alignment(dtype._type_)
    if layout == "null-pointer":
        return None, wp.array(
            ptr=0,
            dtype=dtype,
            shape=(length,),
            strides=(element_size,),
            device=device,
            copy=False,
        )
    if layout == "zero-stride":
        raw = wp.zeros(element_size + 2 * alignment, dtype=wp.uint8, device=device)
        pointer = int(raw.ptr) + (-int(raw.ptr) % alignment)
        return raw, wp.array(
            ptr=pointer,
            dtype=dtype,
            shape=(length,),
            strides=(0,),
            device=device,
            copy=False,
        )
    if layout in ("misaligned-stride-positive", "misaligned-stride-negative"):
        step = element_size + 1
    elif layout in ("aligned-gapped-positive", "aligned-gapped-negative"):
        step = element_size + alignment
    elif layout in ("misaligned-pointer-positive", "misaligned-pointer-negative"):
        step = element_size
    else:
        raise ValueError(f"unknown test layout: {layout}")

    span = step * (length - 1)
    anchor = ((span + alignment - 1) // alignment) * alignment
    raw = wp.zeros(anchor + element_size + 2 * alignment, dtype=wp.uint8, device=device)
    aligned_base = int(raw.ptr) + (-int(raw.ptr) % alignment)
    if layout.endswith("positive"):
        pointer = aligned_base
        stride = step
    else:
        pointer = aligned_base + (anchor if "misaligned-stride" in layout else span)
        stride = -step
    if "misaligned-pointer" in layout:
        pointer += 1
    view = wp.array(
        ptr=pointer,
        dtype=dtype,
        shape=(length,),
        strides=(stride,),
        device=device,
        copy=False,
    )
    return raw, view


def _seeded_output_arrays(operator: WarpMatrixFreeStableNHOperator):
    return {
        "external_rhs": wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device),
        "scalar_rhs": wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device),
        "root_primary_correction": wp.empty(
            operator.n_free_dofs,
            dtype=wp.float64,
            device=operator.device,
        ),
    }


def _assert_seeded_writable_layout_contract(
    test: unittest.TestCase,
    operator: WarpMatrixFreeStableNHOperator,
) -> None:
    active = wp.ones(1, dtype=wp.int32, device=operator.device)
    inverse = wp.ones(9 * operator.n_free, dtype=wp.float64, device=operator.device)
    direction = wp.array(
        np.linspace(-0.5, 0.7, operator.n_free_dofs),
        dtype=wp.float64,
        device=operator.device,
    )
    rhs = wp.ones(operator.n_free, dtype=wp.vec3d, device=operator.device)
    target_specs = {
        "external_rhs": (wp.vec3d, operator.n_free),
        "scalar_rhs": (wp.float64, operator.n_free_dofs),
        "root_primary_correction": (wp.float64, operator.n_free_dofs),
    }
    apply_target_specs = {
        "published_direction": (wp.vec3d, operator.n_free),
        "product": (wp.vec3d, operator.n_free),
        **target_specs,
        "workspace_delta_piola": (wp.mat33d, operator.n_tets),
    }
    misaligned_layouts = (
        "misaligned-stride-positive",
        "misaligned-stride-negative",
        "misaligned-pointer-positive",
        "misaligned-pointer-negative",
    )
    rejected_layouts = (*misaligned_layouts, "null-pointer")
    aligned_layouts = ("aligned-gapped-positive", "aligned-gapped-negative")

    for target_name, (dtype, length) in target_specs.items():
        for layout in (*rejected_layouts, *aligned_layouts):
            with test.subTest(api="gradient", target=target_name, layout=layout):
                raw, target = _writable_layout_view(operator.device, dtype, length, layout)
                outputs = _seeded_output_arrays(operator)
                outputs[target_name] = target
                tracked = [
                    *([] if raw is None else [raw]),
                    *(array for name, array in outputs.items() if name != target_name),
                ]
                before = [array.numpy().tobytes() for array in tracked]
                with mock.patch.object(warp_operator_module.wp, "launch") as launch:
                    if layout in rejected_layouts:
                        message = (
                            "non-null data pointer"
                            if layout == "null-pointer"
                            else "naturally aligned pointer and stride"
                        )
                        with test.assertRaisesRegex(ValueError, message):
                            operator.launch_gradient_masked_seed_root_zero_start(
                                outputs["external_rhs"],
                                active,
                                inverse,
                                0.5,
                                outputs["scalar_rhs"],
                                outputs["root_primary_correction"],
                                scale=-1.0,
                            )
                        test.assertEqual(launch.call_count, 0)
                    else:
                        operator.launch_gradient_masked_seed_root_zero_start(
                            outputs["external_rhs"],
                            active,
                            inverse,
                            0.5,
                            outputs["scalar_rhs"],
                            outputs["root_primary_correction"],
                            scale=-1.0,
                        )
                        test.assertEqual(launch.call_count, 1)
                test.assertEqual([array.numpy().tobytes() for array in tracked], before)

    for target_name, (dtype, length) in apply_target_specs.items():
        for layout in (*rejected_layouts, *aligned_layouts):
            with test.subTest(api="apply", target=target_name, layout=layout):
                raw, target = _writable_layout_view(operator.device, dtype, length, layout)
                outputs = _seeded_output_arrays(operator)
                published = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                workspace = operator.create_apply_workspace()
                if target_name in outputs:
                    outputs[target_name] = target
                elif target_name == "published_direction":
                    published = target
                elif target_name == "product":
                    product = target
                else:
                    workspace.delta_piola = target
                writable = {
                    "published_direction": published,
                    "product": product,
                    **outputs,
                    "workspace_delta_piola": workspace.delta_piola,
                }
                tracked = [
                    *([] if raw is None else [raw]),
                    *(array for name, array in writable.items() if name != target_name),
                ]
                before = [array.numpy().tobytes() for array in tracked]
                with mock.patch.object(warp_operator_module.wp, "launch") as launch:
                    if layout in rejected_layouts:
                        message = (
                            "non-null data pointer"
                            if layout == "null-pointer"
                            else "naturally aligned pointer and stride"
                        )
                        with test.assertRaisesRegex(ValueError, message):
                            operator.launch_apply_residual_scalar_direction_seed_root_zero_start(
                                direction,
                                published,
                                rhs,
                                product,
                                outputs["external_rhs"],
                                workspace,
                                inverse,
                                0.5,
                                outputs["scalar_rhs"],
                                outputs["root_primary_correction"],
                            )
                        test.assertEqual(launch.call_count, 0)
                    else:
                        operator.launch_apply_residual_scalar_direction_seed_root_zero_start(
                            direction,
                            published,
                            rhs,
                            product,
                            outputs["external_rhs"],
                            workspace,
                            inverse,
                            0.5,
                            outputs["scalar_rhs"],
                            outputs["root_primary_correction"],
                        )
                        test.assertEqual(launch.call_count, 2)
                test.assertEqual([array.numpy().tobytes() for array in tracked], before)


def _assert_seeded_read_layout_contract(
    test: unittest.TestCase,
    operator: WarpMatrixFreeStableNHOperator,
) -> None:
    invalid_layouts = (
        "misaligned-stride-positive",
        "misaligned-stride-negative",
        "misaligned-pointer-positive",
        "misaligned-pointer-negative",
        "null-pointer",
    )
    valid_layouts = (
        "zero-stride",
        "aligned-gapped-positive",
        "aligned-gapped-negative",
    )

    for source_name, (dtype, length) in {
        "active": (wp.int32, 1),
        "root_inverse_diagonal": (wp.float64, 9 * operator.n_free),
    }.items():
        for layout in (*invalid_layouts, *valid_layouts):
            with test.subTest(api="gradient", source=source_name, layout=layout):
                _raw, source = _writable_layout_view(operator.device, dtype, length, layout)
                active = wp.ones(1, dtype=wp.int32, device=operator.device)
                inverse = wp.ones(9 * operator.n_free, dtype=wp.float64, device=operator.device)
                if source_name == "active":
                    active = source
                else:
                    inverse = source
                outputs = _seeded_output_arrays(operator)
                with mock.patch.object(warp_operator_module.wp, "launch") as launch:
                    if layout in invalid_layouts:
                        message = (
                            "non-null data pointer"
                            if layout == "null-pointer"
                            else "naturally aligned pointer and stride"
                        )
                        with test.assertRaisesRegex(ValueError, message):
                            operator.launch_gradient_masked_seed_root_zero_start(
                                outputs["external_rhs"],
                                active,
                                inverse,
                                0.5,
                                outputs["scalar_rhs"],
                                outputs["root_primary_correction"],
                            )
                        test.assertEqual(launch.call_count, 0)
                    else:
                        operator.launch_gradient_masked_seed_root_zero_start(
                            outputs["external_rhs"],
                            active,
                            inverse,
                            0.5,
                            outputs["scalar_rhs"],
                            outputs["root_primary_correction"],
                        )
                        test.assertEqual(launch.call_count, 1)

    apply_source_specs = {
        "direction_scalar": (wp.float64, operator.n_free_dofs),
        "rhs": (wp.vec3d, operator.n_free),
        "root_inverse_diagonal": (wp.float64, 9 * operator.n_free),
    }
    for source_name, (dtype, length) in apply_source_specs.items():
        for layout in (*invalid_layouts, *valid_layouts):
            with test.subTest(api="apply", source=source_name, layout=layout):
                _raw, source = _writable_layout_view(operator.device, dtype, length, layout)
                direction = wp.ones(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
                rhs = wp.ones(operator.n_free, dtype=wp.vec3d, device=operator.device)
                inverse = wp.ones(9 * operator.n_free, dtype=wp.float64, device=operator.device)
                if source_name == "direction_scalar":
                    direction = source
                elif source_name == "rhs":
                    rhs = source
                else:
                    inverse = source
                outputs = _seeded_output_arrays(operator)
                published = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                workspace = operator.create_apply_workspace()
                with mock.patch.object(warp_operator_module.wp, "launch") as launch:
                    if layout in invalid_layouts:
                        message = (
                            "non-null data pointer"
                            if layout == "null-pointer"
                            else "naturally aligned pointer and stride"
                        )
                        with test.assertRaisesRegex(ValueError, message):
                            operator.launch_apply_residual_scalar_direction_seed_root_zero_start(
                                direction,
                                published,
                                rhs,
                                product,
                                outputs["external_rhs"],
                                workspace,
                                inverse,
                                0.5,
                                outputs["scalar_rhs"],
                                outputs["root_primary_correction"],
                            )
                        test.assertEqual(launch.call_count, 0)
                    else:
                        operator.launch_apply_residual_scalar_direction_seed_root_zero_start(
                            direction,
                            published,
                            rhs,
                            product,
                            outputs["external_rhs"],
                            workspace,
                            inverse,
                            0.5,
                            outputs["scalar_rhs"],
                            outputs["root_primary_correction"],
                        )
                        test.assertEqual(launch.call_count, 2)


def _owned_overlap_target_view(
    operator: WarpMatrixFreeStableNHOperator,
    owned_array: wp.array,
    target_dtype,
    target_length: int,
    overlap: str,
) -> wp.array:
    alignment = ctypes.alignment(target_dtype._type_)
    pointer = int(owned_array.ptr)
    if overlap == "partial":
        pointer += alignment
        owned_span = operator._array_memory_span(
            owned_array,
            wp.types.type_size_in_bytes(owned_array.dtype),
        )
        if pointer >= owned_span[1]:
            raise AssertionError("owned test array is too small for an aligned partial-overlap view")
    elif overlap != "exact":
        raise ValueError(f"unknown overlap: {overlap}")
    return wp.array(
        ptr=pointer,
        dtype=target_dtype,
        shape=(target_length,),
        strides=(wp.types.type_size_in_bytes(target_dtype),),
        device=operator.device,
        copy=False,
    )


def _assert_seeded_targets_reject_operator_owned_storage(
    test: unittest.TestCase,
    operator: WarpMatrixFreeStableNHOperator,
) -> None:
    expected_owned_names = (
        "positions",
        "tets",
        "shape_gradients",
        "volumes",
        "mass",
        "mu",
        "lam",
        "inertial_target",
        "free",
        "vertex_to_free",
        "incidence_offsets",
        "incidence_tets",
        "incidence_corners",
        "deformation_gradients",
        "cofactors",
        "determinants",
        "first_piola",
    )
    owned_arrays = operator._owned_device_arrays()
    test.assertEqual(tuple(name for name, _array in owned_arrays), expected_owned_names)
    test.assertEqual(len(owned_arrays), 17)

    active = wp.ones(1, dtype=wp.int32, device=operator.device)
    inverse = wp.ones(9 * operator.n_free, dtype=wp.float64, device=operator.device)
    gradient_target_specs = {
        "external_rhs": ("external_rhs", wp.vec3d, operator.n_free),
        "scalar_rhs": ("scalar_rhs", wp.float64, operator.n_free_dofs),
        "root_primary_correction": ("root_primary_correction", wp.float64, operator.n_free_dofs),
    }
    for target_key, (target_name, dtype, length) in gradient_target_specs.items():
        for owned_name, owned_array in owned_arrays:
            for overlap in ("exact", "partial"):
                with test.subTest(api="gradient", target=target_name, owned=owned_name, overlap=overlap):
                    target = _owned_overlap_target_view(operator, owned_array, dtype, length, overlap)
                    outputs = _seeded_output_arrays(operator)
                    outputs[target_key] = target
                    with mock.patch.object(warp_operator_module.wp, "launch") as launch:
                        with test.assertRaisesRegex(
                            ValueError,
                            f"{target_name} must not overlap operator-owned {owned_name}",
                        ):
                            operator.launch_gradient_masked_seed_root_zero_start(
                                outputs["external_rhs"],
                                active,
                                inverse,
                                0.5,
                                outputs["scalar_rhs"],
                                outputs["root_primary_correction"],
                            )
                    test.assertEqual(launch.call_count, 0)

    direction = wp.ones(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
    rhs = wp.ones(operator.n_free, dtype=wp.vec3d, device=operator.device)
    apply_target_specs = {
        "published_direction": ("published_direction", wp.vec3d, operator.n_free),
        "product": ("product", wp.vec3d, operator.n_free),
        "external_rhs": ("external_rhs", wp.vec3d, operator.n_free),
        "workspace_delta_piola": ("workspace delta_piola", wp.mat33d, operator.n_tets),
        "scalar_rhs": ("scalar_rhs", wp.float64, operator.n_free_dofs),
        "root_primary_correction": ("root_primary_correction", wp.float64, operator.n_free_dofs),
    }
    for target_key, (target_name, dtype, length) in apply_target_specs.items():
        for owned_name, owned_array in owned_arrays:
            for overlap in ("exact", "partial"):
                with test.subTest(api="apply", target=target_name, owned=owned_name, overlap=overlap):
                    target = _owned_overlap_target_view(operator, owned_array, dtype, length, overlap)
                    outputs = _seeded_output_arrays(operator)
                    published = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                    product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                    workspace = operator.create_apply_workspace()
                    if target_key in outputs:
                        outputs[target_key] = target
                    elif target_key == "published_direction":
                        published = target
                    elif target_key == "product":
                        product = target
                    else:
                        workspace.delta_piola = target
                    with mock.patch.object(warp_operator_module.wp, "launch") as launch:
                        with test.assertRaisesRegex(
                            ValueError,
                            f"{target_name} must not overlap operator-owned {owned_name}",
                        ):
                            operator.launch_apply_residual_scalar_direction_seed_root_zero_start(
                                direction,
                                published,
                                rhs,
                                product,
                                outputs["external_rhs"],
                                workspace,
                                inverse,
                                0.5,
                                outputs["scalar_rhs"],
                                outputs["root_primary_correction"],
                            )
                    test.assertEqual(launch.call_count, 0)


class TestWarpMatrixFreeStableNHOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_dtype(torch.float64)
        cls.oracle, cls.operator = _oracle_and_device("cpu")

    def test_kernel_version_is_explicit(self):
        self.assertEqual(KERNEL_VERSION, "mg-vbd-warp-operator-v2-tiled-pcg")
        self.assertEqual(FUSED_GATHER_KERNEL_VERSION, "mg-vbd-warp-fused-gather-v1")
        self.assertEqual(ROOT_SEEDED_GATHER_KERNEL_VERSION, "mg-vbd-warp-root-seeded-gather-v1")
        self.assertEqual(
            SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
            "mg-vbd-warp-scalar-direction-apply-residual-v1",
        )
        self.assertEqual(CONTRACT_ID, "mg-vbd-warp-fixed-pcg-research-v2")

        gradient_source = inspect.getsource(warp_operator_module._gather_gradient_masked)
        apply_source = inspect.getsource(warp_operator_module._gather_operator_product_residual)
        self.assertLess(gradient_source.index("for cursor in range"), gradient_source.index("if active[0] == 0"))
        self.assertNotIn("atomic", gradient_source)
        self.assertNotIn("atomic", apply_source)

        seeded_gradient_source = inspect.getsource(warp_operator_module._gather_gradient_masked_seed_root_zero_start)
        seeded_apply_source = inspect.getsource(
            warp_operator_module._gather_operator_product_residual_seed_root_zero_start
        )
        for source in (seeded_gradient_source, seeded_apply_source):
            self.assertNotIn("atomic", source)
            self.assertIn("transformed = wp.float64(0.0)", source)
            self.assertIn("for local_column in range(3)", source)
        self.assertLess(
            seeded_gradient_source.index("value = scale * value"), seeded_gradient_source.index("if active[0] == 0")
        )
        self.assertLess(
            seeded_gradient_source.index("external_rhs[free_index] = value"),
            seeded_gradient_source.index("scalar_rhs[scalar_base]"),
        )
        self.assertLess(
            seeded_apply_source.index("product[free_index] = value"),
            seeded_apply_source.index("residual_value = rhs[free_index] - value"),
        )

        scalar_tet_source = inspect.getsource(warp_operator_module._apply_tet_operator_publish_scalar_direction)
        scalar_gather_source = inspect.getsource(warp_operator_module._gather_operator_product_residual)
        self.assertNotIn("atomic", scalar_tet_source)
        self.assertNotIn("atomic", scalar_gather_source)
        self.assertIn("published_direction[free_index] = wp.vec3d", scalar_tet_source)
        self.assertIn("if owner < n_tets", scalar_tet_source)

    def test_sorted_gather_and_exact_free_elimination(self):
        operator = self.operator
        oracle = self.oracle
        np.testing.assert_array_equal(operator.free_host, oracle.free)
        expected_lookup = np.full(oracle.n_vertices, -1, dtype=np.int32)
        expected_lookup[oracle.free] = np.arange(oracle.free.size, dtype=np.int32)
        np.testing.assert_array_equal(operator.vertex_to_free_host, expected_lookup)
        np.testing.assert_array_equal(operator.vertex_to_free_host[oracle.pinned], -1)

        expected_entries = 0
        for free_index, vertex in enumerate(oracle.free):
            start = int(operator.incidence_offsets_host[free_index])
            end = int(operator.incidence_offsets_host[free_index + 1])
            pairs = list(
                zip(
                    operator.incidence_tets_host[start:end].tolist(),
                    operator.incidence_corners_host[start:end].tolist(),
                    strict=True,
                )
            )
            self.assertEqual(pairs, sorted(pairs))
            expected = sorted(
                (tet, corner)
                for tet, corners in enumerate(oracle.tets)
                for corner, candidate in enumerate(corners)
                if int(candidate) == int(vertex)
            )
            self.assertEqual(pairs, expected)
            expected_entries += len(expected)
        self.assertEqual(int(operator.incidence_offsets_host[-1]), expected_entries)
        self.assertNotIn("wp.atomic", inspect.getsource(__import__(operator.__module__, fromlist=["*"])))

        full_sized = wp.zeros(oracle.n_vertices, dtype=wp.vec3d, device="cpu")
        output = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        with self.assertRaisesRegex(ValueError, "direction must be a vec3d array"):
            operator.launch_apply(full_sized, output, operator.create_apply_workspace())

    def test_gradient_action_diagonal_and_geometry_match_oracle(self):
        operator = self.operator
        oracle = self.oracle
        gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        operator.launch_gradient(gradient)
        np.testing.assert_allclose(gradient.numpy().reshape(-1), oracle.gradient_free(), rtol=2.0e-14, atol=8.0e-14)
        np.testing.assert_allclose(
            operator.deformation_gradients.numpy(), oracle.deformation_gradients, rtol=2.0e-14, atol=8.0e-14
        )
        np.testing.assert_allclose(operator.cofactors.numpy(), oracle.cofactors, rtol=3.0e-14, atol=8.0e-14)
        np.testing.assert_allclose(operator.determinants.numpy(), oracle.determinants, rtol=3.0e-14, atol=8.0e-14)

        direction_host = np.random.default_rng(817).normal(size=(operator.n_free, 3))
        direction = wp.array(direction_host, dtype=wp.vec3d, device="cpu")
        product = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        operator.launch_apply(direction, product, operator.create_apply_workspace())
        np.testing.assert_allclose(
            product.numpy().reshape(-1), oracle.apply_free(direction_host), rtol=3.0e-14, atol=2.0e-13
        )

        diagonal = wp.empty(operator.n_free, dtype=wp.mat33d, device="cpu")
        operator.launch_block_diagonal(diagonal)
        np.testing.assert_allclose(diagonal.numpy(), oracle.block_diagonal(), rtol=3.0e-14, atol=1.0e-13)

    def test_repeated_gathers_are_bitwise_deterministic(self):
        operator = self.operator
        direction_host = np.random.default_rng(823).normal(size=(operator.n_free, 3))
        direction = wp.array(direction_host, dtype=wp.vec3d, device="cpu")
        gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        product = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        diagonal = wp.empty(operator.n_free, dtype=wp.mat33d, device="cpu")
        workspace = operator.create_apply_workspace()

        snapshots = []
        for _ in range(3):
            operator.launch_gradient(gradient)
            operator.launch_apply(direction, product, workspace)
            operator.launch_block_diagonal(diagonal)
            snapshots.append((gradient.numpy(), product.numpy(), diagonal.numpy()))
        for current in snapshots[1:]:
            for expected, actual in zip(snapshots[0], current, strict=True):
                np.testing.assert_array_equal(actual, expected)

    def test_fused_gathers_match_unfused_random_inputs_bitwise(self):
        cases = (
            ("shared_vertex", *_oracle_and_device("cpu")),
            ("default_stretch", *_default_stretch_oracle_and_device("cpu")),
        )
        for case_index, (name, _oracle, operator) in enumerate(cases):
            with self.subTest(name=name):
                generator = np.random.default_rng(1800 + case_index)
                direction_host = generator.normal(size=(operator.n_free, 3))
                rhs_host = generator.normal(size=(operator.n_free, 3))
                direction = wp.array(direction_host, dtype=wp.vec3d, device=operator.device)
                rhs = wp.array(rhs_host, dtype=wp.vec3d, device=operator.device)
                active = wp.array([1], dtype=wp.int32, device=operator.device)
                legacy_gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                fused_gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                legacy_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                legacy_residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                fused_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                fused_residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                legacy_workspace = operator.create_apply_workspace()
                fused_workspace = operator.create_apply_workspace()
                pointers = tuple(
                    int(array.ptr)
                    for array in (
                        direction,
                        rhs,
                        active,
                        legacy_gradient,
                        fused_gradient,
                        legacy_product,
                        legacy_residual,
                        fused_product,
                        fused_residual,
                        legacy_workspace.delta_piola,
                        fused_workspace.delta_piola,
                    )
                )

                for scale in (-1.0, 0.375):
                    for active_value in (1, 0):
                        with self.subTest(name=name, scale=scale, active=active_value):
                            active.assign(np.array([active_value], dtype=np.int32))
                            fused_gradient.assign(np.full((operator.n_free, 3), np.nan, dtype=np.float64))
                            operator.launch_gradient(legacy_gradient, scale=scale)
                            wp.launch(
                                _mask_vector,
                                dim=operator.n_free,
                                inputs=[active, legacy_gradient],
                                device=operator.device,
                            )
                            operator.launch_gradient_masked(fused_gradient, active, scale=scale)
                            actual = fused_gradient.numpy()
                            expected = legacy_gradient.numpy()
                            _assert_bitwise_equal(self, actual, expected)
                            if active_value == 0:
                                np.testing.assert_array_equal(actual.view(np.uint64), 0)

                fused_product.assign(np.full((operator.n_free, 3), np.nan, dtype=np.float64))
                fused_residual.assign(np.full((operator.n_free, 3), np.nan, dtype=np.float64))
                operator.launch_apply(direction, legacy_product, legacy_workspace)
                wp.launch(
                    _subtract_vectors,
                    dim=operator.n_free,
                    inputs=[rhs, legacy_product, legacy_residual],
                    device=operator.device,
                )
                operator.launch_apply_residual(direction, rhs, fused_product, fused_residual, fused_workspace)
                _assert_bitwise_equal(self, fused_product.numpy(), legacy_product.numpy())
                _assert_bitwise_equal(self, fused_residual.numpy(), legacy_residual.numpy())
                self.assertEqual(
                    pointers,
                    tuple(
                        int(array.ptr)
                        for array in (
                            direction,
                            rhs,
                            active,
                            legacy_gradient,
                            fused_gradient,
                            legacy_product,
                            legacy_residual,
                            fused_product,
                            fused_residual,
                            legacy_workspace.delta_piola,
                            fused_workspace.delta_piola,
                        )
                    ),
                )

    def test_fused_gathers_preserve_nonfinite_edge_semantics(self):
        _oracle, operator = _oracle_and_device("cpu")
        first_piola = operator.first_piola.numpy()
        first_piola[0, 0, 0] = np.nan
        operator.first_piola.assign(first_piola)
        active = wp.array([1], dtype=wp.int32, device=operator.device)
        legacy_gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        fused_gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        operator.launch_gradient(legacy_gradient, scale=-1.0)
        operator.launch_gradient_masked(fused_gradient, active, scale=-1.0)
        expected_gradient = legacy_gradient.numpy()
        actual_gradient = fused_gradient.numpy()
        self.assertFalse(np.isfinite(expected_gradient).all())
        _assert_bitwise_equal(self, actual_gradient, expected_gradient)

        active.assign(np.array([0], dtype=np.int32))
        fused_gradient.assign(np.full((operator.n_free, 3), np.nan, dtype=np.float64))
        operator.launch_gradient_masked(fused_gradient, active, scale=-1.0)
        np.testing.assert_array_equal(fused_gradient.numpy().view(np.uint64), 0)

        direction_host = np.random.default_rng(1811).normal(size=(operator.n_free, 3))
        rhs_host = np.random.default_rng(1817).normal(size=(operator.n_free, 3))
        direction_host[0, 1] = np.nan
        rhs_host[-1, 2] = np.inf
        direction = wp.array(direction_host, dtype=wp.vec3d, device=operator.device)
        rhs = wp.array(rhs_host, dtype=wp.vec3d, device=operator.device)
        legacy_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        legacy_residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        fused_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        fused_residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        operator.launch_apply(direction, legacy_product, operator.create_apply_workspace())
        wp.launch(
            _subtract_vectors,
            dim=operator.n_free,
            inputs=[rhs, legacy_product, legacy_residual],
            device=operator.device,
        )
        operator.launch_apply_residual(
            direction,
            rhs,
            fused_product,
            fused_residual,
            operator.create_apply_workspace(),
        )
        self.assertFalse(np.isfinite(legacy_product.numpy()).all())
        self.assertFalse(np.isfinite(legacy_residual.numpy()).all())
        _assert_bitwise_equal(self, fused_product.numpy(), legacy_product.numpy())
        _assert_bitwise_equal(self, fused_residual.numpy(), legacy_residual.numpy())

    def test_scalar_direction_apply_matches_standalone_publication_bitwise(self):
        cases = (
            ("shared_vertex", *_oracle_and_device("cpu")),
            ("default_stretch", *_default_stretch_oracle_and_device("cpu")),
        )
        for case_index, (name, _oracle, operator) in enumerate(cases):
            generator = np.random.default_rng(1910 + case_index)
            random_direction = generator.normal(size=(operator.n_free, 3))
            edge_direction = np.array(random_direction, copy=True)
            edge_direction[0] = np.array((-0.0, 0.0, np.nan), dtype=np.float64)
            edge_direction[-1, 1] = np.inf
            rhs_host = generator.normal(size=(operator.n_free, 3))
            rhs_host[-1, 2] = -np.inf
            for edge_label, direction_host in (("finite", random_direction), ("edge", edge_direction)):
                with self.subTest(name=name, edge=edge_label):
                    direction_scalar = wp.array(
                        direction_host.reshape(-1),
                        dtype=wp.float64,
                        device=operator.device,
                    )
                    rhs = wp.array(rhs_host, dtype=wp.vec3d, device=operator.device)
                    legacy_direction = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                    legacy_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                    legacy_residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                    published_direction = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                    fused_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                    fused_residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                    legacy_workspace = operator.create_apply_workspace()
                    fused_workspace = operator.create_apply_workspace()

                    wp.launch(
                        _publish_scalar_direction,
                        dim=operator.n_free,
                        inputs=[direction_scalar, legacy_direction],
                        device=operator.device,
                    )
                    operator.launch_apply_residual(
                        legacy_direction,
                        rhs,
                        legacy_product,
                        legacy_residual,
                        legacy_workspace,
                    )
                    published_direction.assign(np.full((operator.n_free, 3), 41.0, dtype=np.float64))
                    operator.launch_apply_residual_scalar_direction(
                        direction_scalar,
                        published_direction,
                        rhs,
                        fused_product,
                        fused_residual,
                        fused_workspace,
                    )

                    _assert_bitwise_equal(self, published_direction.numpy(), legacy_direction.numpy())
                    _assert_bitwise_equal(self, fused_product.numpy(), legacy_product.numpy())
                    _assert_bitwise_equal(self, fused_residual.numpy(), legacy_residual.numpy())
                    _assert_bitwise_equal(
                        self,
                        fused_workspace.delta_piola.numpy(),
                        legacy_workspace.delta_piola.numpy(),
                    )

    def test_root_seeded_producers_are_bitwise_old_route_for_masks_signed_zero_and_nonfinite(self):
        for case_index, (name, _oracle, operator) in enumerate(
            (
                ("shared_vertex", *_oracle_and_device("cpu")),
                ("default_stretch", *_default_stretch_oracle_and_device("cpu")),
            )
        ):
            with self.subTest(problem=name):
                generator = np.random.default_rng(1931 + case_index)
                inverse_host = generator.normal(size=9 * operator.n_free)
                inverse = wp.array(inverse_host, dtype=wp.float64, device=operator.device)
                omega = 0.713
                active = wp.ones(1, dtype=wp.int32, device=operator.device)
                for active_value in (1, 0):
                    with self.subTest(problem=name, producer="gradient", active=active_value):
                        active.assign(np.array([active_value], dtype=np.int32))
                        old_external = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                        old_scalar = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
                        old_primary = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
                        new_external = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                        new_scalar = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
                        new_primary = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
                        operator.launch_gradient_masked(old_external, active, scale=-1.0)
                        wp.launch(
                            scalar_fused_module._fused_root_ingress_zero_start_scalar_jacobi,
                            dim=operator.n_free_dofs,
                            inputs=[old_external, inverse, omega, old_scalar, old_primary],
                            device=operator.device,
                        )
                        operator.launch_gradient_masked_seed_root_zero_start(
                            new_external,
                            active,
                            inverse,
                            omega,
                            new_scalar,
                            new_primary,
                            scale=-1.0,
                        )
                        for actual, expected in (
                            (new_external, old_external),
                            (new_scalar, old_scalar),
                            (new_primary, old_primary),
                        ):
                            _assert_bitwise_equal(self, actual.numpy(), expected.numpy())
                        if active_value == 0:
                            np.testing.assert_array_equal(new_external.numpy().view(np.uint64), 0)
                            np.testing.assert_array_equal(new_scalar.numpy().view(np.uint64), 0)

                direction_host = generator.normal(size=(operator.n_free, 3))
                rhs_host = generator.normal(size=(operator.n_free, 3))
                direction_host.reshape(-1)[:4] = (-0.0, 0.0, np.nan, np.inf)
                rhs_host.reshape(-1)[-4:] = (-0.0, 0.0, -np.inf, np.nan)
                direction_scalar = wp.array(direction_host.reshape(-1), dtype=wp.float64, device=operator.device)
                rhs = wp.array(rhs_host, dtype=wp.vec3d, device=operator.device)
                old_published = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                old_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                old_external = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                old_scalar = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
                old_primary = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
                old_workspace = operator.create_apply_workspace()
                operator.launch_apply_residual_scalar_direction(
                    direction_scalar,
                    old_published,
                    rhs,
                    old_product,
                    old_external,
                    old_workspace,
                )
                wp.launch(
                    scalar_fused_module._fused_root_ingress_zero_start_scalar_jacobi,
                    dim=operator.n_free_dofs,
                    inputs=[old_external, inverse, omega, old_scalar, old_primary],
                    device=operator.device,
                )

                new_published = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                new_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                new_external = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
                new_scalar = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
                new_primary = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
                new_workspace = operator.create_apply_workspace()
                operator.launch_apply_residual_scalar_direction_seed_root_zero_start(
                    direction_scalar,
                    new_published,
                    rhs,
                    new_product,
                    new_external,
                    new_workspace,
                    inverse,
                    omega,
                    new_scalar,
                    new_primary,
                )
                for actual, expected in (
                    (new_published, old_published),
                    (new_product, old_product),
                    (new_external, old_external),
                    (new_scalar, old_scalar),
                    (new_primary, old_primary),
                    (new_workspace.delta_piola, old_workspace.delta_piola),
                ):
                    _assert_bitwise_equal(self, actual.numpy(), expected.numpy())

    def test_root_seeded_producer_validation_rejects_aliases_before_launch(self):
        operator = self.operator
        n_free = operator.n_free
        n_dofs = operator.n_free_dofs
        active = wp.ones(1, dtype=wp.int32, device=operator.device)
        external = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
        scalar_rhs = wp.empty(n_dofs, dtype=wp.float64, device=operator.device)
        primary = wp.empty(n_dofs, dtype=wp.float64, device=operator.device)
        inverse = wp.empty(9 * n_free, dtype=wp.float64, device=operator.device)

        invalid_gradient_calls = (
            (external, active, inverse, 0.7, scalar_rhs, scalar_rhs),
            (
                external,
                active,
                wp.array(ptr=scalar_rhs.ptr, dtype=wp.float64, shape=(9 * n_free,), device=operator.device, copy=False),
                0.7,
                scalar_rhs,
                primary,
            ),
            (
                external,
                wp.array(ptr=external.ptr, dtype=wp.int32, shape=(1,), device=operator.device, copy=False),
                inverse,
                0.7,
                scalar_rhs,
                primary,
            ),
            (external, active, inverse, np.nan, scalar_rhs, primary),
        )
        for arguments in invalid_gradient_calls:
            with self.subTest(
                arguments=tuple(int(value.ptr) if isinstance(value, wp.array) else value for value in arguments)
            ):
                with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
                    with self.assertRaises(ValueError):
                        operator.launch_gradient_masked_seed_root_zero_start(*arguments)
                self.assertEqual(launch.call_count, 0)

        direction_scalar = wp.empty(n_dofs, dtype=wp.float64, device=operator.device)
        published = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
        rhs = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
        product = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
        workspace = operator.create_apply_workspace()
        invalid_apply_calls = (
            (direction_scalar, published, rhs, product, rhs, workspace, inverse, 0.7, scalar_rhs, primary),
            (direction_scalar, published, rhs, product, external, workspace, inverse, 0.7, direction_scalar, primary),
            (direction_scalar, published, rhs, product, external, workspace, inverse, 0.7, scalar_rhs, scalar_rhs),
            (
                direction_scalar,
                published,
                rhs,
                product,
                external,
                workspace,
                wp.array(ptr=product.ptr, dtype=wp.float64, shape=(9 * n_free,), device=operator.device, copy=False),
                0.7,
                scalar_rhs,
                primary,
            ),
        )
        for arguments in invalid_apply_calls:
            with self.subTest(
                alias=tuple(
                    int(value.ptr) if isinstance(value, wp.array) else type(value).__name__ for value in arguments
                )
            ):
                with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
                    with self.assertRaises(ValueError):
                        operator.launch_apply_residual_scalar_direction_seed_root_zero_start(*arguments)
                self.assertEqual(launch.call_count, 0)

    def test_root_seeded_producer_writable_layout_preflight(self):
        _assert_seeded_writable_layout_contract(self, self.operator)

    def test_root_seeded_producer_read_layout_preflight(self):
        _assert_seeded_read_layout_contract(self, self.operator)

    def test_root_seeded_targets_reject_operator_owned_storage_before_launch(self):
        _assert_seeded_targets_reject_operator_owned_storage(self, self.operator)

    def test_fused_gather_validation_rejects_invalid_arrays_and_output_aliases(self):
        operator = self.operator
        n_free = operator.n_free
        direction = wp.zeros(n_free, dtype=wp.vec3d, device=operator.device)
        rhs = wp.zeros(n_free, dtype=wp.vec3d, device=operator.device)
        product = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
        residual = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
        workspace = operator.create_apply_workspace()
        output = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)

        for invalid_active in (
            wp.zeros(1, dtype=wp.int64, device=operator.device),
            wp.zeros(1, dtype=wp.float32, device=operator.device),
            wp.zeros(2, dtype=wp.int32, device=operator.device),
        ):
            with self.subTest(dtype=invalid_active.dtype, shape=invalid_active.shape):
                with self.assertRaisesRegex(ValueError, "active must be an int32 array"):
                    operator.launch_gradient_masked(output, invalid_active)
        with self.assertRaisesRegex(ValueError, "active must be an int32 array"):
            operator.launch_gradient_masked(output, [1])
        aliased_active = wp.array(
            ptr=output.ptr,
            dtype=wp.int32,
            shape=(1,),
            device=operator.device,
            copy=False,
        )
        with self.assertRaisesRegex(ValueError, "active and output must not alias"):
            operator.launch_gradient_masked(output, aliased_active)
        for scale in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "scale must be finite"):
                operator.launch_gradient_masked(output, wp.ones(1, dtype=wp.int32, device=operator.device), scale=scale)

        wrong_type = wp.empty(n_free, dtype=wp.vec3f, device=operator.device)
        wrong_shape = wp.empty(n_free + 1, dtype=wp.vec3d, device=operator.device)
        for name, arguments in (
            ("direction", (wrong_type, rhs, product, residual, workspace)),
            ("rhs", (direction, wrong_shape, product, residual, workspace)),
            ("product", (direction, rhs, wrong_shape, residual, workspace)),
            ("residual", (direction, rhs, product, wrong_shape, workspace)),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f"{name} must be a vec3d array"):
                    operator.launch_apply_residual(*arguments)

        _other_oracle, other_operator = _oracle_and_device("cpu")
        with self.assertRaisesRegex(ValueError, "workspace belongs to a different operator"):
            operator.launch_apply_residual(direction, rhs, product, residual, other_operator.create_apply_workspace())

        for left_name, right_name in (("rhs", "product"), ("rhs", "residual"), ("product", "residual")):
            arrays = {"rhs": rhs, "product": product, "residual": residual}
            arrays[right_name] = arrays[left_name]
            with self.subTest(alias=f"{left_name}-{right_name}"):
                with self.assertRaisesRegex(ValueError, f"{left_name} and {right_name} must not alias"):
                    operator.launch_apply_residual(
                        direction,
                        arrays["rhs"],
                        arrays["product"],
                        arrays["residual"],
                        workspace,
                    )

        overlap_storage = wp.empty(n_free + 1, dtype=wp.vec3d, device=operator.device)
        overlapping_rhs = overlap_storage[:n_free]
        overlapping_product = overlap_storage[1:]
        with self.assertRaisesRegex(ValueError, "rhs and product must not alias"):
            operator.launch_apply_residual(direction, overlapping_rhs, overlapping_product, residual, workspace)

        for output_name in ("product", "residual"):
            overlap_storage = wp.empty(n_free + 1, dtype=wp.vec3d, device=operator.device)
            overlapping_direction = overlap_storage[:n_free]
            overlapping_output = overlap_storage[1:]
            output_arrays = {"product": product, "residual": residual}
            output_arrays[output_name] = overlapping_output
            with self.subTest(partial_direction_alias=output_name):
                with self.assertRaisesRegex(
                    ValueError,
                    f"direction and {output_name} must not partially alias",
                ):
                    operator.launch_apply_residual(
                        overlapping_direction,
                        rhs,
                        output_arrays["product"],
                        output_arrays["residual"],
                        workspace,
                    )

        invalid_workspace = operator.create_apply_workspace()
        invalid_workspace.delta_piola = wp.empty(operator.n_tets, dtype=wp.vec3d, device=operator.device)
        with self.assertRaisesRegex(ValueError, "workspace delta_piola must be a mat33d array"):
            operator.launch_apply_residual(direction, rhs, product, residual, invalid_workspace)

        for vector_name, vector in (
            ("direction", direction),
            ("rhs", rhs),
            ("product", product),
            ("residual", residual),
        ):
            aliased_workspace = operator.create_apply_workspace()
            aliased_workspace.delta_piola = wp.array(
                ptr=vector.ptr,
                dtype=wp.mat33d,
                shape=(operator.n_tets,),
                device=operator.device,
                copy=False,
            )
            with self.subTest(workspace_alias=vector_name):
                with self.assertRaisesRegex(ValueError, f"{vector_name} must not alias workspace delta_piola"):
                    operator.launch_apply_residual(direction, rhs, product, residual, aliased_workspace)

    def test_scalar_direction_apply_rejects_every_alias_before_launch(self):
        operator = self.operator
        n_free = operator.n_free
        direction_scalar = wp.zeros(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
        published = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
        rhs = wp.zeros(n_free, dtype=wp.vec3d, device=operator.device)
        product = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
        residual = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
        workspace = operator.create_apply_workspace()

        for invalid in (
            wp.empty(operator.n_free_dofs, dtype=wp.float32, device=operator.device),
            wp.empty(operator.n_free_dofs + 1, dtype=wp.float64, device=operator.device),
            wp.empty((operator.n_free, 3), dtype=wp.float64, device=operator.device),
        ):
            with self.subTest(dtype=invalid.dtype, shape=invalid.shape):
                with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
                    with self.assertRaisesRegex(ValueError, "direction_scalar must be a float64 array"):
                        operator.launch_apply_residual_scalar_direction(
                            invalid,
                            published,
                            rhs,
                            product,
                            residual,
                            workspace,
                        )
                self.assertEqual(launch.call_count, 0)

        named_vectors = {
            "published_direction": published,
            "rhs": rhs,
            "product": product,
            "residual": residual,
        }
        vector_names = tuple(named_vectors)
        vector_element_size = wp.types.type_size_in_bytes(wp.vec3d)
        for vector_name in ("published_direction", "product", "residual"):
            for layout_name, stride in (("zero", 0), ("sub-element", vector_element_size - 1)):
                backing = wp.empty(n_free + 1, dtype=wp.vec3d, device=operator.device)
                overlapping_output = wp.array(
                    ptr=backing.ptr,
                    dtype=wp.vec3d,
                    shape=(n_free,),
                    strides=(stride,),
                    device=operator.device,
                    copy=False,
                )
                arguments = dict(named_vectors)
                arguments[vector_name] = overlapping_output
                with self.subTest(writable_layout=vector_name, layout=layout_name):
                    with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
                        with self.assertRaisesRegex(
                            ValueError,
                            f"{vector_name} must have a non-overlapping writable 1-D layout",
                        ):
                            operator.launch_apply_residual_scalar_direction(
                                direction_scalar,
                                arguments["published_direction"],
                                arguments["rhs"],
                                arguments["product"],
                                arguments["residual"],
                                workspace,
                            )
                    self.assertEqual(launch.call_count, 0)

        delta_element_size = wp.types.type_size_in_bytes(wp.mat33d)
        for layout_name, stride in (("zero", 0), ("sub-element", delta_element_size - 1)):
            backing = wp.empty(operator.n_tets + 1, dtype=wp.mat33d, device=operator.device)
            overlapping_delta = wp.array(
                ptr=backing.ptr,
                dtype=wp.mat33d,
                shape=(operator.n_tets,),
                strides=(stride,),
                device=operator.device,
                copy=False,
            )
            overlapping_workspace = operator.create_apply_workspace()
            overlapping_workspace.delta_piola = overlapping_delta
            with self.subTest(writable_layout="workspace delta_piola", layout=layout_name):
                with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
                    with self.assertRaisesRegex(
                        ValueError,
                        "workspace delta_piola must have a non-overlapping writable 1-D layout",
                    ):
                        operator.launch_apply_residual_scalar_direction(
                            direction_scalar,
                            published,
                            rhs,
                            product,
                            residual,
                            overlapping_workspace,
                        )
                self.assertEqual(launch.call_count, 0)

        for left_index, left_name in enumerate(vector_names):
            for right_name in vector_names[left_index + 1 :]:
                arguments = dict(named_vectors)
                arguments[right_name] = arguments[left_name]
                with self.subTest(alias=f"{left_name}-{right_name}"):
                    with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
                        with self.assertRaisesRegex(ValueError, f"{left_name} and {right_name} must not alias"):
                            operator.launch_apply_residual_scalar_direction(
                                direction_scalar,
                                arguments["published_direction"],
                                arguments["rhs"],
                                arguments["product"],
                                arguments["residual"],
                                workspace,
                            )
                    self.assertEqual(launch.call_count, 0)

                overlap_storage = wp.empty(n_free + 1, dtype=wp.vec3d, device=operator.device)
                arguments = dict(named_vectors)
                arguments[left_name] = overlap_storage[:n_free]
                arguments[right_name] = overlap_storage[1:]
                with self.subTest(partial_alias=f"{left_name}-{right_name}"):
                    with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
                        with self.assertRaisesRegex(ValueError, f"{left_name} and {right_name} must not alias"):
                            operator.launch_apply_residual_scalar_direction(
                                direction_scalar,
                                arguments["published_direction"],
                                arguments["rhs"],
                                arguments["product"],
                                arguments["residual"],
                                workspace,
                            )
                    self.assertEqual(launch.call_count, 0)

        for vector_name in vector_names:
            storage = wp.empty(operator.n_free_dofs + 1, dtype=wp.float64, device=operator.device)
            scalar_view = storage[: operator.n_free_dofs]
            overlapping_vector = wp.array(
                ptr=int(storage.ptr) + wp.types.type_size_in_bytes(wp.float64),
                dtype=wp.vec3d,
                shape=(n_free,),
                device=operator.device,
                copy=False,
            )
            arguments = dict(named_vectors)
            arguments[vector_name] = overlapping_vector
            with self.subTest(scalar_alias=vector_name):
                with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
                    with self.assertRaisesRegex(ValueError, f"direction_scalar and {vector_name} must not alias"):
                        operator.launch_apply_residual_scalar_direction(
                            scalar_view,
                            arguments["published_direction"],
                            arguments["rhs"],
                            arguments["product"],
                            arguments["residual"],
                            workspace,
                        )
                self.assertEqual(launch.call_count, 0)

        aliased_workspace = operator.create_apply_workspace()
        aliased_workspace.delta_piola = wp.array(
            ptr=direction_scalar.ptr,
            dtype=wp.mat33d,
            shape=(operator.n_tets,),
            device=operator.device,
            copy=False,
        )
        with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(ValueError, "direction_scalar must not alias workspace delta_piola"):
                operator.launch_apply_residual_scalar_direction(
                    direction_scalar,
                    published,
                    rhs,
                    product,
                    residual,
                    aliased_workspace,
                )
        self.assertEqual(launch.call_count, 0)

        for vector_name, vector in named_vectors.items():
            arguments = dict(named_vectors)
            aliased_workspace = operator.create_apply_workspace()
            aliased_workspace.delta_piola = wp.array(
                ptr=int(vector.ptr) + wp.types.type_size_in_bytes(wp.float64),
                dtype=wp.mat33d,
                shape=(operator.n_tets,),
                device=operator.device,
                copy=False,
            )
            with self.subTest(workspace_partial_alias=vector_name):
                with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
                    with self.assertRaisesRegex(ValueError, f"{vector_name} must not alias workspace delta_piola"):
                        operator.launch_apply_residual_scalar_direction(
                            direction_scalar,
                            arguments["published_direction"],
                            arguments["rhs"],
                            arguments["product"],
                            arguments["residual"],
                            aliased_workspace,
                        )
                self.assertEqual(launch.call_count, 0)

        invalid_workspace = operator.create_apply_workspace()
        invalid_workspace.delta_piola = wp.empty(operator.n_tets, dtype=wp.vec3d, device=operator.device)
        with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(ValueError, "workspace delta_piola must be a mat33d array"):
                operator.launch_apply_residual_scalar_direction(
                    direction_scalar,
                    published,
                    rhs,
                    product,
                    residual,
                    invalid_workspace,
                )
        self.assertEqual(launch.call_count, 0)

        _other_oracle, other_operator = _oracle_and_device("cpu")
        with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(ValueError, "workspace belongs to a different operator"):
                operator.launch_apply_residual_scalar_direction(
                    direction_scalar,
                    published,
                    rhs,
                    product,
                    residual,
                    other_operator.create_apply_workspace(),
                )
        self.assertEqual(launch.call_count, 0)

    def test_direction_aliases_are_safe_for_fused_apply(self):
        operator = self.operator
        n_free = operator.n_free
        direction_host = np.random.default_rng(1823).normal(size=(n_free, 3))
        rhs_host = np.random.default_rng(1831).normal(size=(n_free, 3))
        for alias in ("rhs", "product", "residual"):
            with self.subTest(alias=alias):
                effective_rhs = direction_host if alias == "rhs" else rhs_host
                expected_product = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
                expected_residual = wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
                reference_direction = wp.array(direction_host, dtype=wp.vec3d, device=operator.device)
                reference_rhs = wp.array(effective_rhs, dtype=wp.vec3d, device=operator.device)
                operator.launch_apply(reference_direction, expected_product, operator.create_apply_workspace())
                wp.launch(
                    _subtract_vectors,
                    dim=n_free,
                    inputs=[reference_rhs, expected_product, expected_residual],
                    device=operator.device,
                )

                direction = wp.array(direction_host, dtype=wp.vec3d, device=operator.device)
                rhs = direction if alias == "rhs" else wp.array(rhs_host, dtype=wp.vec3d, device=operator.device)
                product = direction if alias == "product" else wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
                residual = (
                    direction if alias == "residual" else wp.empty(n_free, dtype=wp.vec3d, device=operator.device)
                )
                operator.launch_apply_residual(direction, rhs, product, residual, operator.create_apply_workspace())
                _assert_bitwise_equal(self, product.numpy(), expected_product.numpy())
                _assert_bitwise_equal(self, residual.numpy(), expected_residual.numpy())

    def test_fused_methods_have_exact_kernel_launch_counts(self):
        operator = self.operator
        direction = wp.zeros(operator.n_free, dtype=wp.vec3d, device=operator.device)
        rhs = wp.zeros(operator.n_free, dtype=wp.vec3d, device=operator.device)
        product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        output = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        active = wp.ones(1, dtype=wp.int32, device=operator.device)
        workspace = operator.create_apply_workspace()
        operator.launch_gradient_masked(output, active)
        operator.launch_apply_residual(direction, rhs, product, residual, workspace)
        direction_scalar = wp.zeros(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
        published_direction = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        root_inverse = wp.ones(9 * operator.n_free, dtype=wp.float64, device=operator.device)
        scalar_rhs = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
        root_primary = wp.empty(operator.n_free_dofs, dtype=wp.float64, device=operator.device)
        operator.launch_apply_residual_scalar_direction(
            direction_scalar,
            published_direction,
            rhs,
            product,
            residual,
            workspace,
        )

        with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
            operator.launch_gradient_masked(output, active)
        self.assertEqual(launch.call_count, 1)
        self.assertIs(launch.call_args.args[0], warp_operator_module._gather_gradient_masked)

        with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
            operator.launch_apply_residual(direction, rhs, product, residual, workspace)
        self.assertEqual(launch.call_count, 2)
        self.assertIs(launch.call_args_list[0].args[0], warp_operator_module._apply_tet_operator)
        self.assertIs(launch.call_args_list[1].args[0], warp_operator_module._gather_operator_product_residual)

        with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
            operator.launch_apply_residual_scalar_direction(
                direction_scalar,
                published_direction,
                rhs,
                product,
                residual,
                workspace,
            )
        self.assertEqual(launch.call_count, 2)
        self.assertIs(
            launch.call_args_list[0].args[0],
            warp_operator_module._apply_tet_operator_publish_scalar_direction,
        )
        self.assertIs(
            launch.call_args_list[1].args[0],
            warp_operator_module._gather_operator_product_residual,
        )

        with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
            operator.launch_gradient_masked_seed_root_zero_start(
                output,
                active,
                root_inverse,
                0.5,
                scalar_rhs,
                root_primary,
            )
        self.assertEqual(launch.call_count, 1)
        self.assertIs(launch.call_args.args[0], warp_operator_module._gather_gradient_masked_seed_root_zero_start)

        with mock.patch.object(warp_operator_module.wp, "launch", wraps=wp.launch) as launch:
            operator.launch_apply_residual_scalar_direction_seed_root_zero_start(
                direction_scalar,
                published_direction,
                rhs,
                product,
                residual,
                workspace,
                root_inverse,
                0.5,
                scalar_rhs,
                root_primary,
            )
        self.assertEqual(launch.call_count, 2)
        self.assertIs(
            launch.call_args_list[0].args[0],
            warp_operator_module._apply_tet_operator_publish_scalar_direction,
        )
        self.assertIs(
            launch.call_args_list[1].args[0],
            warp_operator_module._gather_operator_product_residual_seed_root_zero_start,
        )


class TestWarpFixedPCG(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_dtype(torch.float64)
        cls.oracle, cls.operator = _oracle_and_device("cpu")
        cls.rhs = -cls.oracle.gradient_free()

    def test_fixed_pcg_matches_numpy_oracle_and_reports_work(self):
        iterations = 4
        expected = solve_fixed_pcg(self.oracle, self.rhs, iterations)
        workspace = WarpFixedPCGWorkspace(self.operator, iterations)
        pointers_before = {
            name: int(getattr(workspace, name).ptr)
            for name in (
                "rhs",
                "solution",
                "residual",
                "preconditioned",
                "direction",
                "operator_direction",
                "operator_solution",
                "block_diagonal",
                "preconditioner_inverse",
                "state_status",
                "rho",
                "trace_curvature",
                "reduction_partial_first",
                "reduction_partial_second",
                "reduction_partial_flag_first",
                "reduction_partial_flag_second",
            )
        }
        workspace.set_rhs(self.rhs)
        workspace.launch()
        result = workspace.record()
        pointers_after = {name: int(getattr(workspace, name).ptr) for name in pointers_before}

        self.assertTrue(result.success, result.deterministic_record())
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.completed_iterations, iterations)
        self.assertEqual(result.requested_iterations, iterations)
        self.assertEqual(len(result.trace), iterations)
        self.assertEqual(result.preconditioner_identity, "block-jacobi-3x3-warp-v1")
        self.assertFalse(result.capture_replay)
        self.assertTrue(result.research_only)
        self.assertFalse(result.performance_evidence)
        self.assertEqual(pointers_after, pointers_before)
        self.assertEqual(result.work.preconditioner_builds, 1)
        self.assertEqual(result.work.operator_applications, iterations + 1)
        self.assertEqual(result.work.residual_verification_applications, 1)
        self.assertEqual(result.work.preconditioner_applications, iterations + 1)
        self.assertEqual(result.work.scalar_reductions, 2 * iterations + 2)
        self.assertEqual(result.work.reduction_stages, 2)
        self.assertEqual(result.work.reduction_block_size, 256)
        self.assertEqual(result.work.reduction_tile_count, 1)
        self.assertEqual(result.work.reduction_kernel_launches, 2 * result.work.scalar_reductions)
        self.assertEqual(result.work.kernel_launches, 2 + 7 * iterations + 6 + result.work.scalar_reductions)
        np.testing.assert_allclose(result.solution.reshape(-1), expected.solution, rtol=3.0e-13, atol=3.0e-14)
        self.assertAlmostEqual(result.true_residual_norm, expected.true_residual_norm, places=13)

    def test_tiled_reductions_cover_rows_around_block_boundary(self):
        iterations = 4
        for vector_count in (255, 256, 257):
            with self.subTest(vector_count=vector_count):
                oracle, operator = _diagonal_oracle_and_device(vector_count, "cpu")
                rhs = np.random.default_rng(900 + vector_count).normal(size=oracle.n_free_dofs)
                expected = solve_fixed_pcg(
                    oracle,
                    rhs,
                    iterations,
                    preconditioner=lambda value: value,
                    preconditioner_identity="test-identity-tiled-reduction-v1",
                )
                workspace = WarpFixedPCGWorkspace(
                    operator,
                    iterations,
                    external_preconditioner_inverse=_identity_block_preconditioner(vector_count),
                    preconditioner_identity="test-identity-tiled-reduction-v1",
                )
                partial_names = (
                    "reduction_partial_first",
                    "reduction_partial_second",
                    "reduction_partial_flag_first",
                    "reduction_partial_flag_second",
                )
                pointers = tuple(int(getattr(workspace, name).ptr) for name in partial_names)
                snapshots = []
                for _ in range(2):
                    workspace.set_rhs(rhs)
                    workspace.launch()
                    snapshots.append(workspace.record())

                actual = snapshots[0]
                self.assertTrue(actual.success, actual.deterministic_record())
                self.assertEqual(actual.reason, "completed")
                self.assertEqual(actual.work.reduction_block_size, 256)
                self.assertEqual(actual.work.reduction_tile_count, (vector_count + 255) // 256)
                self.assertEqual(workspace.reduction_padded_count, actual.work.reduction_tile_count * 256)
                self.assertEqual(actual.work.reduction_kernel_launches, 2 * actual.work.scalar_reductions)
                np.testing.assert_allclose(actual.solution.reshape(-1), expected.solution, rtol=8.0e-13, atol=8.0e-14)
                np.testing.assert_array_equal(snapshots[1].solution, actual.solution)
                self.assertEqual(snapshots[1].deterministic_record(), actual.deterministic_record())
                self.assertEqual(pointers, tuple(int(getattr(workspace, name).ptr) for name in partial_names))

    def test_nonfinite_rhs_in_second_tile_preserves_primary_status(self):
        vector_count = 257
        oracle, operator = _diagonal_oracle_and_device(vector_count, "cpu")
        rhs = np.random.default_rng(1171).normal(size=oracle.n_free_dofs)
        rhs[3 * 256 + 1] = np.nan
        workspace = WarpFixedPCGWorkspace(
            operator,
            3,
            external_preconditioner_inverse=_identity_block_preconditioner(vector_count),
            preconditioner_identity="test-identity-tiled-nonfinite-v1",
        )
        workspace.set_rhs(rhs)
        workspace.launch()
        result = workspace.record()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonfinite_rhs")
        self.assertEqual(result.completed_iterations, 0)
        self.assertIsNone(result.rhs_norm)
        self.assertEqual(result.work.reduction_tile_count, 2)
        self.assertEqual(result.work.reduction_kernel_launches, 2 * result.work.scalar_reductions)
        self.assertEqual([item.status for item in result.trace], [result.reason] * 3)
        json.dumps(result.deterministic_record(), allow_nan=False)

    def test_breakdown_is_masked_without_shortening_schedule(self):
        iterations = 3
        identity_inverse = np.repeat(np.eye(3, dtype=np.float64)[None], self.operator.n_free, axis=0)
        workspace = WarpFixedPCGWorkspace(
            self.operator,
            iterations,
            external_preconditioner_inverse=identity_inverse,
            preconditioner_identity="test-zero-block-preconditioner-v1",
        )
        # A post-validation device corruption emulates runtime memory failure;
        # the PCG schedule must fail closed without a host-side short circuit.
        workspace.preconditioner_inverse.assign(np.zeros_like(identity_inverse))
        workspace.set_rhs(self.rhs)
        workspace.launch()
        result = workspace.record()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonpositive_preconditioner")
        self.assertEqual(result.completed_iterations, 0)
        self.assertEqual(len(result.trace), iterations)
        self.assertEqual([item.status for item in result.trace], [result.reason] * iterations)
        self.assertTrue(all(not item.active_update_completed for item in result.trace))
        np.testing.assert_array_equal(result.solution, 0.0)
        self.assertEqual(result.work.preconditioner_builds, 0)
        self.assertEqual(result.work.operator_applications, iterations + 1)
        self.assertEqual(result.work.preconditioner_applications, iterations + 1)
        self.assertEqual(result.work.scalar_reductions, 2 * iterations + 2)
        self.assertEqual(result.work.reduction_stages, 2)
        self.assertEqual(result.work.reduction_kernel_launches, 2 * result.work.scalar_reductions)
        self.assertEqual(result.work.kernel_launches, 7 * iterations + 6 + result.work.scalar_reductions)

    def test_external_preconditioner_requires_exact_symmetric_positive_definite_blocks(self):
        blocks = np.repeat(np.eye(3, dtype=np.float64)[None], self.operator.n_free, axis=0)
        nonsymmetric = blocks.copy()
        nonsymmetric[0, 0, 1] = 0.25
        with self.assertRaisesRegex(ValueError, "exactly symmetric"):
            WarpFixedPCGWorkspace(
                self.operator,
                2,
                external_preconditioner_inverse=nonsymmetric,
                preconditioner_identity="test-nonsymmetric-v1",
            )
        indefinite = blocks.copy()
        indefinite[0, 0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive definite"):
            WarpFixedPCGWorkspace(
                self.operator,
                2,
                external_preconditioner_inverse=indefinite,
                preconditioner_identity="test-indefinite-v1",
            )

    def test_zero_rhs_is_successfully_masked_at_fixed_work(self):
        iterations = 5
        workspace = WarpFixedPCGWorkspace(self.operator, iterations)
        workspace.set_rhs(np.zeros(self.operator.n_free_dofs, dtype=np.float64))
        workspace.launch()
        result = workspace.record()

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "zero_rhs")
        self.assertEqual(result.completed_iterations, 0)
        self.assertEqual(len(result.trace), iterations)
        self.assertEqual(result.true_residual_norm, 0.0)
        self.assertEqual(result.work.operator_applications, iterations + 1)
        self.assertEqual(result.work.preconditioner_applications, iterations + 1)
        np.testing.assert_array_equal(result.solution, 0.0)

    def test_nonfinite_rhs_fails_closed_and_preserves_primary_reason(self):
        iterations = 2
        for nonfinite in (np.nan, np.inf, -np.inf):
            with self.subTest(nonfinite=nonfinite):
                rhs = self.rhs.copy()
                rhs[3] = nonfinite
                workspace = WarpFixedPCGWorkspace(self.operator, iterations)
                workspace.set_rhs(rhs)
                workspace.launch()
                result = workspace.record()

                self.assertFalse(result.success)
                self.assertEqual(result.reason, "nonfinite_rhs")
                self.assertEqual(result.completed_iterations, 0)
                self.assertIsNone(result.rhs_norm)
                self.assertIsNone(result.true_residual_norm)
                self.assertEqual([item.status for item in result.trace], [result.reason] * iterations)
                np.testing.assert_array_equal(result.solution, 0.0)
                self.assertEqual(result.work.operator_applications, iterations + 1)
                json.dumps(result.deterministic_record(), allow_nan=False)

    def test_nonfinite_preconditioner_corruption_is_finite_json_safe(self):
        iterations = 2
        identity_inverse = np.repeat(np.eye(3, dtype=np.float64)[None], self.operator.n_free, axis=0)
        workspace = WarpFixedPCGWorkspace(
            self.operator,
            iterations,
            external_preconditioner_inverse=identity_inverse,
            preconditioner_identity="test-corrupted-block-preconditioner-v1",
        )
        corrupted = identity_inverse.copy()
        corrupted[0, 0, 0] = np.nan
        workspace.preconditioner_inverse.assign(corrupted)
        workspace.set_rhs(self.rhs)
        workspace.launch()
        result = workspace.record()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonfinite_preconditioner")
        self.assertEqual(result.completed_iterations, 0)
        self.assertTrue(all(item.residual_norm is None for item in result.trace))
        json.dumps(result.deterministic_record(), allow_nan=False)

    def test_nonfinite_operator_corruption_is_finite_json_safe(self):
        oracle, operator = _oracle_and_device("cpu")
        corrupted = operator.cofactors.numpy()
        corrupted[0, 0, 0] = np.nan
        operator.cofactors.assign(corrupted)
        identity_inverse = np.repeat(np.eye(3, dtype=np.float64)[None], operator.n_free, axis=0)
        workspace = WarpFixedPCGWorkspace(
            operator,
            2,
            external_preconditioner_inverse=identity_inverse,
            preconditioner_identity="test-identity-block-preconditioner-v1",
        )
        workspace.set_rhs(-oracle.gradient_free())
        workspace.launch()
        result = workspace.record()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonfinite_operator")
        self.assertEqual(result.completed_iterations, 0)
        self.assertIsNone(result.true_residual_norm)
        self.assertTrue(all(item.direction_curvature is None for item in result.trace))
        json.dumps(result.deterministic_record(), allow_nan=False)

    def test_repeat_launch_reuses_buffers_and_is_bitwise_deterministic(self):
        workspace = WarpFixedPCGWorkspace(self.operator, 4)
        pointers = (
            int(workspace.solution.ptr),
            int(workspace.direction.ptr),
            int(workspace.apply_workspace.delta_piola.ptr),
        )
        solutions = []
        records = []
        for _ in range(3):
            workspace.set_rhs(self.rhs)
            workspace.launch()
            record = workspace.record()
            solutions.append(record.solution)
            records.append(record.deterministic_record())
        self.assertEqual(
            pointers,
            (int(workspace.solution.ptr), int(workspace.direction.ptr), int(workspace.apply_workspace.delta_piola.ptr)),
        )
        np.testing.assert_array_equal(solutions[1], solutions[0])
        np.testing.assert_array_equal(solutions[2], solutions[0])
        self.assertEqual(records[1], records[0])
        self.assertEqual(records[2], records[0])


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestWarpFusedGatherCudaCapture(unittest.TestCase):
    def test_root_seeded_producer_writable_layout_preflight(self):
        if wp.get_cuda_device_count() < 1:
            self.skipTest("no claimed CUDA device is visible")
        _oracle, operator = _oracle_and_device("cuda:0")
        _assert_seeded_writable_layout_contract(self, operator)

    def test_scalar_direction_apply_capture_replays_and_overwrites_poison(self):
        if wp.get_cuda_device_count() < 1:
            self.skipTest("no claimed CUDA device is visible")
        torch.set_default_dtype(torch.float64)
        _oracle, operator = _oracle_and_device("cuda:0")
        generator = np.random.default_rng(1931)
        direction_scalar = wp.array(
            generator.normal(size=operator.n_free_dofs),
            dtype=wp.float64,
            device=operator.device,
        )
        rhs = wp.array(generator.normal(size=(operator.n_free, 3)), dtype=wp.vec3d, device=operator.device)
        published = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        workspace = operator.create_apply_workspace()
        operator.launch_apply_residual_scalar_direction(
            direction_scalar,
            published,
            rhs,
            product,
            residual,
            workspace,
        )
        pointers = tuple(
            int(value.ptr) for value in (direction_scalar, rhs, published, product, residual, workspace.delta_piola)
        )
        with wp.ScopedCapture(device=operator.device) as capture:
            operator.launch_apply_residual_scalar_direction(
                direction_scalar,
                published,
                rhs,
                product,
                residual,
                workspace,
            )

        snapshots = []
        for replay_index in range(3):
            if replay_index == 1:
                direction_host = np.zeros(operator.n_free_dofs, dtype=np.float64)
                direction_host[0] = -0.0
            elif replay_index == 2:
                direction_host = generator.normal(size=operator.n_free_dofs)
                direction_host[:6] = (-0.0, 0.0, np.nextafter(0.0, 1.0), np.nan, np.inf, -np.inf)
            else:
                direction_host = generator.normal(size=operator.n_free_dofs) * (replay_index + 1.0)
            rhs_host = generator.normal(size=(operator.n_free, 3))
            if replay_index == 2:
                rhs_host[-1] = (np.nan, np.inf, -np.inf)
            direction_scalar.assign(direction_host)
            rhs.assign(rhs_host)
            poison = np.full((operator.n_free, 3), 91.0, dtype=np.float64)
            published.assign(poison)
            product.assign(poison)
            residual.assign(poison)
            workspace.delta_piola.assign(np.full((operator.n_tets, 3, 3), -73.0, dtype=np.float64))
            wp.capture_launch(capture.graph)

            legacy_direction = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
            legacy_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
            legacy_residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
            legacy_workspace = operator.create_apply_workspace()
            wp.launch(
                _publish_scalar_direction,
                dim=operator.n_free,
                inputs=[direction_scalar, legacy_direction],
                device=operator.device,
            )
            operator.launch_apply_residual(
                legacy_direction,
                rhs,
                legacy_product,
                legacy_residual,
                legacy_workspace,
            )
            _assert_bitwise_equal(self, published.numpy(), legacy_direction.numpy())
            _assert_bitwise_equal(self, product.numpy(), legacy_product.numpy())
            _assert_bitwise_equal(self, residual.numpy(), legacy_residual.numpy())
            _assert_bitwise_equal(self, workspace.delta_piola.numpy(), legacy_workspace.delta_piola.numpy())
            snapshots.append(published.numpy())

        self.assertNotEqual(snapshots[0].tobytes(), snapshots[1].tobytes())
        self.assertNotEqual(snapshots[1].tobytes(), snapshots[2].tobytes())
        self.assertEqual(
            pointers,
            tuple(
                int(value.ptr) for value in (direction_scalar, rhs, published, product, residual, workspace.delta_piola)
            ),
        )

    def test_capture_replays_changed_inputs_after_output_poisoning(self):
        if wp.get_cuda_device_count() < 1:
            self.skipTest("no claimed CUDA device is visible")
        torch.set_default_dtype(torch.float64)
        _oracle, operator = _oracle_and_device("cuda:0")
        generator = np.random.default_rng(1847)
        direction = wp.array(generator.normal(size=(operator.n_free, 3)), dtype=wp.vec3d, device=operator.device)
        rhs = wp.array(generator.normal(size=(operator.n_free, 3)), dtype=wp.vec3d, device=operator.device)
        active = wp.ones(1, dtype=wp.int32, device=operator.device)
        gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        workspace = operator.create_apply_workspace()

        operator.launch_gradient_masked(gradient, active, scale=-1.0)
        operator.launch_apply_residual(direction, rhs, product, residual, workspace)
        pointers = tuple(
            int(array.ptr) for array in (direction, rhs, active, gradient, product, residual, workspace.delta_piola)
        )
        with wp.ScopedCapture(device=operator.device) as capture:
            operator.launch_gradient_masked(gradient, active, scale=-1.0)
            operator.launch_apply_residual(direction, rhs, product, residual, workspace)

        snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for replay_index, active_value in enumerate((0, 1, 1)):
            direction_host = generator.normal(size=(operator.n_free, 3)) * (replay_index + 1.5)
            rhs_host = generator.normal(size=(operator.n_free, 3)) - replay_index * 0.25
            direction.assign(direction_host)
            rhs.assign(rhs_host)
            active.assign(np.array([active_value], dtype=np.int32))
            poison = np.full((operator.n_free, 3), np.nan, dtype=np.float64)
            gradient.assign(poison)
            product.assign(poison)
            residual.assign(poison)
            wp.capture_launch(capture.graph)
            captured = (gradient.numpy(), product.numpy(), residual.numpy())

            expected_gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
            expected_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
            expected_residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
            operator.launch_gradient(expected_gradient, scale=-1.0)
            wp.launch(
                _mask_vector,
                dim=operator.n_free,
                inputs=[active, expected_gradient],
                device=operator.device,
            )
            operator.launch_apply(direction, expected_product, operator.create_apply_workspace())
            wp.launch(
                _subtract_vectors,
                dim=operator.n_free,
                inputs=[rhs, expected_product, expected_residual],
                device=operator.device,
            )
            _assert_bitwise_equal(self, captured[0], expected_gradient.numpy())
            _assert_bitwise_equal(self, captured[1], expected_product.numpy())
            _assert_bitwise_equal(self, captured[2], expected_residual.numpy())
            if active_value == 0:
                np.testing.assert_array_equal(captured[0].view(np.uint64), 0)
            snapshots.append(captured)

        self.assertNotEqual(snapshots[0][1].tobytes(), snapshots[1][1].tobytes())
        self.assertNotEqual(snapshots[1][1].tobytes(), snapshots[2][1].tobytes())
        self.assertEqual(
            pointers,
            tuple(
                int(array.ptr) for array in (direction, rhs, active, gradient, product, residual, workspace.delta_piola)
            ),
        )

    def test_device_validation_rejects_cpu_arrays_for_cuda_operator(self):
        if wp.get_cuda_device_count() < 1:
            self.skipTest("no claimed CUDA device is visible")
        _oracle, operator = _oracle_and_device("cuda:0")
        cuda_output = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        cuda_active = wp.ones(1, dtype=wp.int32, device=operator.device)
        cuda_direction = wp.zeros(operator.n_free, dtype=wp.vec3d, device=operator.device)
        cuda_rhs = wp.zeros(operator.n_free, dtype=wp.vec3d, device=operator.device)
        cuda_product = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        cuda_residual = wp.empty(operator.n_free, dtype=wp.vec3d, device=operator.device)
        workspace = operator.create_apply_workspace()
        cpu_vector = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        cpu_active = wp.ones(1, dtype=wp.int32, device="cpu")

        with self.assertRaisesRegex(ValueError, "output must be a vec3d array"):
            operator.launch_gradient_masked(cpu_vector, cuda_active)
        with self.assertRaisesRegex(ValueError, "active must be an int32 array"):
            operator.launch_gradient_masked(cuda_output, cpu_active)
        for name, arguments in (
            ("direction", (cpu_vector, cuda_rhs, cuda_product, cuda_residual, workspace)),
            ("rhs", (cuda_direction, cpu_vector, cuda_product, cuda_residual, workspace)),
            ("product", (cuda_direction, cuda_rhs, cpu_vector, cuda_residual, workspace)),
            ("residual", (cuda_direction, cuda_rhs, cuda_product, cpu_vector, workspace)),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f"{name} must be a vec3d array"):
                    operator.launch_apply_residual(*arguments)


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestWarpFixedPCGCudaCapture(unittest.TestCase):
    def test_fixed_schedule_replays_changed_rhs_across_tile_boundary(self):
        if wp.get_cuda_device_count() < 1:
            self.skipTest("no claimed CUDA device is visible")
        torch.set_default_dtype(torch.float64)
        vector_count = 257
        oracle, operator = _diagonal_oracle_and_device(vector_count, "cuda:0")
        rhs_a = np.random.default_rng(1201).normal(size=oracle.n_free_dofs)
        rhs_b = np.random.default_rng(1207).normal(size=oracle.n_free_dofs)
        preconditioner_identity = "test-identity-tiled-capture-v1"
        expected_b = solve_fixed_pcg(
            oracle,
            rhs_b,
            4,
            preconditioner=lambda value: value,
            preconditioner_identity=preconditioner_identity,
        )
        workspace = WarpFixedPCGWorkspace(
            operator,
            4,
            external_preconditioner_inverse=_identity_block_preconditioner(vector_count),
            preconditioner_identity=preconditioner_identity,
        )
        partial_names = (
            "reduction_partial_first",
            "reduction_partial_second",
            "reduction_partial_flag_first",
            "reduction_partial_flag_second",
        )
        pointers = tuple(int(getattr(workspace, name).ptr) for name in partial_names)

        # Warm compilation and allocations before capture.  The fixed launcher
        # itself performs neither allocation nor host synchronization.
        workspace.set_rhs(rhs_a)
        workspace.launch()
        warm = workspace.record()
        workspace.set_rhs(rhs_b)
        workspace.launch()
        direct_b = workspace.record()
        workspace.set_rhs(rhs_a)
        with wp.ScopedCapture(device=operator.device) as capture:
            workspace.launch()
        workspace.set_rhs(rhs_b)
        wp.capture_launch(capture.graph)
        captured = workspace.record(capture_replay=True)
        wp.capture_launch(capture.graph)
        repeated = workspace.record(capture_replay=True)

        self.assertTrue(captured.success, captured.deterministic_record())
        self.assertTrue(captured.capture_replay)
        self.assertTrue(captured.research_only)
        self.assertFalse(captured.performance_evidence)
        self.assertEqual(captured.work, direct_b.work)
        self.assertEqual(captured.work.reduction_tile_count, 2)
        self.assertEqual(captured.work.reduction_block_size, 256)
        self.assertEqual(captured.work.reduction_stages, 2)
        self.assertEqual(captured.work.reduction_kernel_launches, 2 * captured.work.scalar_reductions)
        self.assertEqual(pointers, tuple(int(getattr(workspace, name).ptr) for name in partial_names))
        np.testing.assert_array_equal(captured.solution, direct_b.solution)
        np.testing.assert_array_equal(repeated.solution, captured.solution)
        self.assertEqual(repeated.deterministic_record(), captured.deterministic_record())
        self.assertNotEqual(captured.solution.tobytes(), warm.solution.tobytes())
        np.testing.assert_allclose(captured.solution.reshape(-1), expected_b.solution, rtol=8.0e-13, atol=8.0e-14)
        self.assertAlmostEqual(captured.true_residual_norm, direct_b.true_residual_norm, places=13)


if __name__ == "__main__":
    unittest.main()
