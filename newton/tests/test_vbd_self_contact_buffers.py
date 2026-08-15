# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for VBD self-contact collision-buffer consumption."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    accumulate_contact_force_and_hessian,
    accumulate_self_contact_force_and_hessian,
    apply_planar_truncation_parallel_by_collision,
)
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionInfo
from newton._src.solvers.vbd.vbd_coupling_kernels import _harvest_vbd_proxy_particle_self_contact_forces_kernel
from newton.tests.unittest_utils import add_function_test, get_test_devices


_VERSION = "self_contact_count_bounds_test_v1"
print(f"[test_vbd_self_contact_buffers] version: {_VERSION}")


@wp.kernel
def _reset_self_contact_outputs(
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
    truncation: wp.array[float],
    harvested_forces: wp.array[wp.vec3],
    combined_forces: wp.array[wp.vec3],
    combined_hessians: wp.array[wp.mat33],
):
    """Reset outputs shared by the direct and captured consumer checks."""
    particle = wp.tid()
    forces[particle] = wp.vec3(0.0)
    hessians[particle] = wp.mat33(0.0)
    truncation[particle] = 1.0
    combined_forces[particle] = wp.vec3(0.0)
    combined_hessians[particle] = wp.mat33(0.0)
    if particle < harvested_forces.shape[0]:
        harvested_forces[particle] = wp.vec3(0.0)


def _make_self_contact_buffer_data(device):
    particle_count = 4
    capacity = 2

    positions = wp.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.05],
        ],
        dtype=wp.vec3,
        device=device,
    )
    triangle_indices = wp.array([[0, 1, 2]], dtype=wp.int32, ndim=2, device=device)
    edge_indices = wp.array([[0, 0, 0, 1]], dtype=wp.int32, ndim=2, device=device)

    vertex_records_np = np.full(2 * particle_count * capacity, -1, dtype=np.int32)
    query_row_offset = (particle_count - 1) * capacity
    vertex_records_np[2 * query_row_offset : 2 * (query_row_offset + capacity)] = [3, 0, 3, 0]

    collision_info = TriMeshCollisionInfo()
    vertex_records = wp.array(vertex_records_np, dtype=wp.int32, device=device)
    collision_info.vertex_colliding_triangles = vertex_records
    collision_info.vertex_colliding_triangles_offsets = wp.array(
        np.arange(particle_count + 1, dtype=np.int32) * capacity, dtype=wp.int32, device=device
    )
    collision_info.vertex_colliding_triangles_buffer_sizes = wp.full(
        particle_count, capacity, dtype=wp.int32, device=device
    )
    vertex_counts = wp.zeros(particle_count, dtype=wp.int32, device=device)
    collision_info.vertex_colliding_triangles_count = vertex_counts
    collision_info.vertex_colliding_triangles_min_dist = wp.zeros(particle_count, dtype=float, device=device)

    collision_info.triangle_colliding_vertices = wp.empty(0, dtype=wp.int32, device=device)
    collision_info.triangle_colliding_vertices_offsets = wp.array([0], dtype=wp.int32, device=device)
    collision_info.triangle_colliding_vertices_buffer_sizes = wp.empty(0, dtype=wp.int32, device=device)
    collision_info.triangle_colliding_vertices_count = wp.empty(0, dtype=wp.int32, device=device)
    collision_info.triangle_colliding_vertices_min_dist = wp.zeros(1, dtype=float, device=device)

    collision_info.edge_colliding_edges = wp.empty(0, dtype=wp.int32, device=device)
    collision_info.edge_colliding_edges_offsets = wp.array([0, 0], dtype=wp.int32, device=device)
    collision_info.edge_colliding_edges_buffer_sizes = wp.zeros(1, dtype=wp.int32, device=device)
    collision_info.edge_colliding_edges_count = wp.zeros(1, dtype=wp.int32, device=device)
    collision_info.edge_colliding_edges_min_dist = wp.zeros(1, dtype=float, device=device)

    collision_info_array = wp.array([collision_info], dtype=TriMeshCollisionInfo, device=device)
    displacements_np = np.zeros((particle_count, 3), dtype=np.float32)
    displacements_np[-1, 2] = -0.1

    return {
        "particle_count": particle_count,
        "capacity": capacity,
        "positions": positions,
        "particle_colors": wp.zeros(particle_count, dtype=wp.int32, device=device),
        "triangle_indices": triangle_indices,
        "edge_indices": edge_indices,
        "collision_info": collision_info_array,
        "collision_info_host": collision_info,
        "vertex_records": vertex_records,
        "vertex_counts": vertex_counts,
        "displacements": wp.array(displacements_np, dtype=wp.vec3, device=device),
        "particle_to_proxy": wp.array([-1, -1, -1, 0], dtype=wp.int32, device=device),
        "particle_flags": wp.array(
            [
                int(newton.ParticleFlags.ACTIVE),
                int(newton.ParticleFlags.ACTIVE),
                int(newton.ParticleFlags.ACTIVE),
                int(newton.ParticleFlags.PROXY),
            ],
            dtype=wp.int32,
            device=device,
        ),
        "particle_inv_mass": wp.ones(particle_count, dtype=float, device=device),
        "particle_radius": wp.zeros(particle_count, dtype=float, device=device),
        "body_contact_count": wp.zeros(1, dtype=wp.int32, device=device),
        "empty_int": wp.empty(0, dtype=wp.int32, device=device),
        "empty_float": wp.empty(0, dtype=float, device=device),
        "empty_vec3": wp.empty(0, dtype=wp.vec3, device=device),
        "empty_transform": wp.empty(0, dtype=wp.transform, device=device),
        "empty_spatial": wp.empty(0, dtype=wp.spatial_vector, device=device),
    }


def _make_self_contact_outputs(data, device):
    particle_count = data["particle_count"]
    return {
        "forces": wp.zeros(particle_count, dtype=wp.vec3, device=device),
        "hessians": wp.zeros(particle_count, dtype=wp.mat33, device=device),
        "truncation": wp.ones(particle_count, dtype=float, device=device),
        "harvested_forces": wp.zeros(1, dtype=wp.vec3, device=device),
        "combined_forces": wp.zeros(particle_count, dtype=wp.vec3, device=device),
        "combined_hessians": wp.zeros(particle_count, dtype=wp.mat33, device=device),
    }


def _launch_self_contact_consumers(data, outputs, device):
    particle_count = data["particle_count"]
    launch_dim = particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE

    wp.launch(
        _reset_self_contact_outputs,
        dim=particle_count,
        inputs=[
            outputs["forces"],
            outputs["hessians"],
            outputs["truncation"],
            outputs["harvested_forces"],
            outputs["combined_forces"],
            outputs["combined_hessians"],
        ],
        device=device,
    )
    self_contact_inputs = [
        0.01,
        0,
        data["positions"],
        data["positions"],
        data["particle_colors"],
        data["triangle_indices"],
        data["edge_indices"],
        data["collision_info"],
        0.1,
        10.0,
        0.0,
        0.0,
        0.01,
        1.0e-5,
    ]
    wp.launch(
        accumulate_self_contact_force_and_hessian,
        dim=launch_dim,
        inputs=self_contact_inputs,
        outputs=[outputs["forces"], outputs["hessians"]],
        device=device,
    )
    wp.launch(
        apply_planar_truncation_parallel_by_collision,
        dim=launch_dim,
        inputs=[
            data["positions"],
            data["displacements"],
            data["triangle_indices"],
            data["edge_indices"],
            data["collision_info"],
            1.0e-6,
            0.85,
        ],
        outputs=[outputs["truncation"]],
        device=device,
    )
    wp.launch(
        _harvest_vbd_proxy_particle_self_contact_forces_kernel,
        dim=launch_dim,
        inputs=[
            0.01,
            data["particle_to_proxy"],
            data["positions"],
            data["positions"],
            data["particle_flags"],
            data["particle_inv_mass"],
            int(newton.ParticleFlags.ACTIVE),
            int(newton.ParticleFlags.PROXY),
            data["triangle_indices"],
            data["edge_indices"],
            data["collision_info"],
            0.1,
            10.0,
            0.0,
            0.0,
            0.01,
            1.0e-5,
        ],
        outputs=[outputs["harvested_forces"]],
        device=device,
    )
    wp.launch(
        accumulate_contact_force_and_hessian,
        dim=launch_dim,
        inputs=[
            *self_contact_inputs,
            data["particle_radius"],
            data["empty_int"],
            data["body_contact_count"],
            0,
            data["empty_float"],
            data["empty_int"],
            data["empty_transform"],
            data["empty_transform"],
            data["empty_spatial"],
            data["empty_vec3"],
            data["empty_int"],
            data["empty_vec3"],
            data["empty_vec3"],
            data["empty_vec3"],
            data["empty_float"],
        ],
        outputs=[outputs["combined_forces"], outputs["combined_hessians"]],
        device=device,
    )


def _snapshot_self_contact_outputs(outputs):
    return {name: value.numpy().copy() for name, value in outputs.items()}


def _assert_self_contact_count_contract(test, results):
    zero = results[0]
    partial = results[1]
    exact = results[2]
    overflow = results[3]

    np.testing.assert_array_equal(zero["forces"], 0.0)
    np.testing.assert_array_equal(zero["hessians"], 0.0)
    np.testing.assert_array_equal(zero["combined_forces"], 0.0)
    np.testing.assert_array_equal(zero["combined_hessians"], 0.0)
    np.testing.assert_array_equal(zero["harvested_forces"], 0.0)
    np.testing.assert_array_equal(zero["truncation"], 1.0)

    test.assertGreater(float(np.max(np.abs(partial["forces"]))), 0.0)
    test.assertGreater(float(np.max(np.abs(partial["hessians"]))), 0.0)
    test.assertGreater(float(np.max(np.abs(partial["harvested_forces"]))), 0.0)
    test.assertLess(float(partial["truncation"][-1]), 1.0)

    np.testing.assert_allclose(partial["combined_forces"], partial["forces"], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(partial["combined_hessians"], partial["hessians"], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(exact["forces"], 2.0 * partial["forces"], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(exact["hessians"], 2.0 * partial["hessians"], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        exact["harvested_forces"], 2.0 * partial["harvested_forces"], rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(exact["truncation"], partial["truncation"], rtol=0.0, atol=0.0)

    for name in results[2]:
        np.testing.assert_allclose(overflow[name], exact[name], rtol=0.0, atol=0.0)


def test_self_contact_consumers_ignore_stale_tails(test, device):
    """Ignore stale records beyond each row's clamped active count."""
    with wp.ScopedDevice(device):
        data = _make_self_contact_buffer_data(device)
        outputs = _make_self_contact_outputs(data, device)
        results = {}
        for raw_count in (0, 1, 2, 3):
            counts = np.zeros(data["particle_count"], dtype=np.int32)
            counts[-1] = raw_count
            data["vertex_counts"].assign(counts)
            _launch_self_contact_consumers(data, outputs, device)
            results[raw_count] = _snapshot_self_contact_outputs(outputs)

        _assert_self_contact_count_contract(test, results)


def test_self_contact_consumers_capture_replays_device_counts(test, device):
    """Read changing clamped row counts when replaying a captured graph."""
    with wp.ScopedDevice(device):
        data = _make_self_contact_buffer_data(device)
        outputs = _make_self_contact_outputs(data, device)

        _launch_self_contact_consumers(data, outputs, device)
        outputs["forces"].numpy()
        with wp.ScopedCapture(device=device) as capture:
            _launch_self_contact_consumers(data, outputs, device)
        graph = capture.graph
        test.assertIsNotNone(graph)

        results = {}
        for raw_count in (0, 1, 2, 3):
            counts = np.zeros(data["particle_count"], dtype=np.int32)
            counts[-1] = raw_count
            data["vertex_counts"].assign(counts)
            wp.capture_launch(graph)
            results[raw_count] = _snapshot_self_contact_outputs(outputs)

        _assert_self_contact_count_contract(test, results)


class TestVBDSelfContactBuffers(unittest.TestCase):
    """Test segmented VBD self-contact buffers."""

    pass


devices = get_test_devices()
cuda_devices = [device for device in devices if device.is_cuda]

add_function_test(
    TestVBDSelfContactBuffers,
    "test_self_contact_consumers_ignore_stale_tails",
    test_self_contact_consumers_ignore_stale_tails,
    devices=devices,
)
add_function_test(
    TestVBDSelfContactBuffers,
    "test_self_contact_consumers_capture_replays_device_counts",
    test_self_contact_consumers_capture_replays_device_counts,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
