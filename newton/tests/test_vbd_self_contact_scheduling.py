# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify that wider VBD self-contact rows preserve contact contributions."""

import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import particle_vbd_kernels as kernels
from newton._src.solvers.vbd import vbd_coupling_kernels
from newton._src.solvers.vbd.tri_mesh_collision import (
    TriMeshCollisionInfo,
    get_edge_colliding_edges_count,
    get_vertex_colliding_triangles_count,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

_VERSION = "self_contact_scheduling_test_v2"
print(f"[test_vbd_self_contact_scheduling] version: {_VERSION}")


@wp.func
def _record_contribution(
    slot: int,
    vertex: int,
    color: int,
    colors: wp.array[int],
    force: wp.vec3,
    hessian: wp.mat33,
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
    destinations: wp.array[int],
    hits: wp.array[int],
):
    if colors[vertex] == color:
        forces[slot] = force
        hessians[slot] = hessian
        destinations[slot] = vertex
        wp.atomic_add(hits, slot, 1)


def _make_contact_recorder(lanes):
    threads = wp.constant(lanes)

    @wp.kernel
    def record(
        color: int,
        previous: wp.array[wp.vec3],
        positions: wp.array[wp.vec3],
        colors: wp.array[int],
        triangles: wp.array2d[int],
        edges: wp.array2d[int],
        infos: wp.array[TriMeshCollisionInfo],
        forces: wp.array[wp.vec3],
        hessians: wp.array[wp.mat33],
        destinations: wp.array[int],
        hits: wp.array[int],
    ):
        owner = wp.tid() // threads
        lane = wp.tid() % threads
        info = infos[0]
        if owner < info.edge_colliding_edges_count.shape[0]:
            count = get_edge_colliding_edges_count(info, owner)
            offset = info.edge_colliding_edges_offsets[owner]
            index = lane
            while index < count:
                other = info.edge_colliding_edges[2 * (offset + index) + 1]
                if other >= 0:
                    has_contact, f0, f1, h0, h1 = kernels.evaluate_edge_edge_contact_2_vertices(
                        owner, other, positions, previous, edges, 0.1, 10.0, 0.3, 0.2, 0.01, 0.01, 1e-5
                    )
                    if has_contact:
                        slot = 2 * (offset + index)
                        _record_contribution(
                            slot, edges[owner, 2], color, colors, f0, h0, forces, hessians, destinations, hits
                        )
                        _record_contribution(
                            slot + 1, edges[owner, 3], color, colors, f1, h1, forces, hessians, destinations, hits
                        )
                index += threads
        if owner < info.vertex_colliding_triangles_count.shape[0]:
            count = get_vertex_colliding_triangles_count(info, owner)
            offset = info.vertex_colliding_triangles_offsets[owner]
            index = lane
            while index < count:
                triangle = info.vertex_colliding_triangles[2 * (offset + index) + 1]
                if triangle >= 0:
                    has_contact, f0, f1, f2, f3, h0, h1, h2, h3 = (
                        kernels.evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                            owner, triangle, positions, previous, triangles, 0.1, 10.0, 0.3, 0.2, 0.01, 0.01
                        )
                    )
                    if has_contact:
                        slot = info.edge_colliding_edges.shape[0] + 4 * (offset + index)
                        _record_contribution(
                            slot, triangles[triangle, 0], color, colors, f0, h0, forces, hessians, destinations, hits
                        )
                        _record_contribution(
                            slot + 1,
                            triangles[triangle, 1],
                            color,
                            colors,
                            f1,
                            h1,
                            forces,
                            hessians,
                            destinations,
                            hits,
                        )
                        _record_contribution(
                            slot + 2,
                            triangles[triangle, 2],
                            color,
                            colors,
                            f2,
                            h2,
                            forces,
                            hessians,
                            destinations,
                            hits,
                        )
                        _record_contribution(
                            slot + 3, owner, color, colors, f3, h3, forces, hessians, destinations, hits
                        )
                index += threads

    return record


_record_four = _make_contact_recorder(4)
_record_eight = _make_contact_recorder(8)


def _fixture(device):
    positions = []
    triangles = []
    edges = []
    vt_owners = []
    ee_owners = []
    distances = (0.0, 0.05, float(np.nextafter(np.float32(0.1), np.float32(0.0))), 0.1, 0.12, 0.05)
    for case in range(24):
        offset = len(positions)
        distance = distances[(case // 2) % len(distances)]
        if case % 2 == 0:
            vertices = [[0, 0, distance], [-1, -1, 0], [1, -1, 0], [0, 1, 0]]
            vt_owners.append((offset, len(triangles)))
            triangles.append([offset + 1, offset + 2, offset + 3])
        else:
            vertices = [[-1, 0, 0], [1, 0, 0], [0, -1, distance], [0, 1, distance]]
            if case in (11, 23):
                vertices = [[0, 0, 0], [0, 0, 0], [distance, 0, 0], [distance, 0, 0]]
            first = len(edges)
            ee_owners.extend(((first, first + 1), (first + 1, first)))
            edges.extend(([0, 0, offset, offset + 1], [0, 0, offset + 2, offset + 3]))
        positions.extend(vertices)
    positions = np.array(positions, dtype=np.float32)
    previous = positions.copy()
    previous[::4] += [0.01, 0.005, 0.015]
    previous[1::4] += [-0.005, 0.01, 0.005]
    displacements = np.random.default_rng(421).normal(0.0, 0.1, positions.shape).astype(np.float32)
    colors = np.tile([0, 0, 1, 2], len(positions) // 4).astype(np.int32)
    info = TriMeshCollisionInfo()
    row_data = {}
    for name, owners, row_count in (
        ("vertex_colliding_triangles", vt_owners, len(positions)),
        ("edge_colliding_edges", ee_owners, len(edges)),
    ):
        capacities = np.zeros(row_count, dtype=np.int32)
        for index, (owner, _) in enumerate(owners):
            capacities[owner] = (0, 5, 9, 13)[index % 4]
        offsets = np.zeros(row_count + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(capacities)
        records = np.full((int(offsets[-1]), 2), -1, dtype=np.int32)
        for owner, other in owners:
            records[offsets[owner] : offsets[owner + 1]] = [owner, other]
        setattr(info, name, wp.array(records.ravel(), dtype=int, device=device))
        setattr(info, name + "_offsets", wp.array(offsets, dtype=int, device=device))
        setattr(info, name + "_buffer_sizes", wp.array(capacities, dtype=int, device=device))
        setattr(info, name + "_count", wp.zeros(row_count, dtype=int, device=device))
        row_data[name] = (owners, capacities)
    return {
        "positions": wp.array(positions, dtype=wp.vec3, device=device),
        "previous": wp.array(previous, dtype=wp.vec3, device=device),
        "displacements": wp.array(displacements, dtype=wp.vec3, device=device),
        "colors": wp.array(colors, dtype=int, device=device),
        "triangles": wp.array(triangles, dtype=int, ndim=2, device=device),
        "edges": wp.array(edges, dtype=int, ndim=2, device=device),
        "info": info,
        "info_array": wp.array([info], dtype=TriMeshCollisionInfo, device=device),
        "rows": row_data,
    }


def _set_counts(data, phase):
    for name, (owners, capacities) in data["rows"].items():
        counts = np.zeros(len(capacities), dtype=np.int32)
        for index, (owner, _) in enumerate(owners):
            counts[owner] = (0, 1, 8, 13, 17)[(index + phase) % 5]
        getattr(data["info"], name + "_count").assign(counts)


def _outputs(data, device):
    count = data["positions"].size
    info = data["info"]
    record_count = info.edge_colliding_edges.size + 2 * info.vertex_colliding_triangles.size
    return {
        "forces": wp.zeros(count, dtype=wp.vec3, device=device),
        "hessians": wp.zeros(count, dtype=wp.mat33, device=device),
        "truncation": wp.ones(count, dtype=float, device=device),
        "record_forces": wp.zeros(record_count, dtype=wp.vec3, device=device),
        "record_hessians": wp.zeros(record_count, dtype=wp.mat33, device=device),
        "destinations": wp.full(record_count, -1, dtype=int, device=device),
        "hits": wp.zeros(record_count, dtype=int, device=device),
    }


def _launch(data, outputs, color, wide, device, launch_lanes=None, *, reset_bounds=False):
    lanes = 8 if wide else 4
    if launch_lanes is None:
        launch_lanes = lanes
    dimension = max(data["positions"].size, data["edges"].shape[0]) * launch_lanes
    for name, array in outputs.items():
        array.fill_(1.0 if name == "truncation" else -1 if name == "destinations" else 0)
    force = (
        kernels._accumulate_self_contact_force_and_hessian_wide
        if wide
        else kernels.accumulate_self_contact_force_and_hessian
    )
    force_outputs = [outputs["forces"], outputs["hessians"]]
    if reset_bounds:
        # A missing reset must fail even for empty rows that never contribute a minimum.
        outputs["truncation"].fill_(0.0)
        force = kernels._accumulate_self_contact_force_and_hessian_wide_with_truncation_reset
        force_outputs.append(outputs["truncation"])
    truncation = (
        kernels._apply_planar_truncation_parallel_by_collision_wide
        if wide
        else kernels.apply_planar_truncation_parallel_by_collision
    )
    common = [data[name] for name in ("previous", "positions", "colors", "triangles", "edges", "info_array")]
    wp.launch(
        force,
        dimension,
        [0.01, color, *common, 0.1, 10.0, 0.3, 0.2, 0.01, 1e-5],
        force_outputs,
        device=device,
        block_dim=128 if wide else 256,
    )
    wp.launch(
        truncation,
        dimension,
        [data["positions"], data["displacements"], data["triangles"], data["edges"], data["info_array"], 1e-5, 0.9],
        [outputs["truncation"]],
        device=device,
        block_dim=64 if wide else 256,
    )
    wp.launch(
        _record_eight if wide else _record_four,
        dimension,
        [
            color,
            *common,
            outputs["record_forces"],
            outputs["record_hessians"],
            outputs["destinations"],
            outputs["hits"],
        ],
        device=device,
        block_dim=128 if wide else 256,
    )


def _assert_outputs(test, reference, candidate):
    snapshots = {name: array.numpy() for name, array in reference.items()}
    for name, expected in snapshots.items():
        with test.subTest(output=name):
            actual = candidate[name].numpy()
            np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(expected.view(np.uint8), actual.view(np.uint8), err_msg="stored bits")
            test.assertTrue(np.isfinite(expected).all())
    hits = snapshots["hits"]
    test.assertTrue(np.all((hits == 0) | (hits == 1)))
    test.assertGreater(int(hits.sum()), 0)
    # All contacts on a given destination have identical geometry and values, so
    # the production atomics have an exact, order-independent serial oracle.
    forces = np.zeros_like(snapshots["forces"])
    hessians = np.zeros_like(snapshots["hessians"])
    for slot in np.flatnonzero(hits):
        destination = snapshots["destinations"][slot]
        forces[destination] += snapshots["record_forces"][slot]
        hessians[destination] += snapshots["record_hessians"][slot]
    np.testing.assert_array_equal(forces, snapshots["forces"])
    np.testing.assert_array_equal(hessians, snapshots["hessians"])


def test_self_contact_scheduling(test, device):
    """Preserve contact values, multiplicity and minima across four- and eight-lane rows."""
    data = _fixture(device)
    reference = _outputs(data, device)
    candidate = _outputs(data, device)
    for phase in (0, 2, 4):
        _set_counts(data, phase)
        for color in (0, 1, 2):
            _launch(data, reference, color, False, device)
            _launch(data, candidate, color, True, device)
            with test.subTest(phase=phase, color=color):
                _assert_outputs(test, reference, candidate)


def test_self_contact_reset_scheduling(test, device):
    """Preserve contributions and final minima when the force pass resets poisoned bounds."""
    data = _fixture(device)
    reference = _outputs(data, device)
    candidate = _outputs(data, device)
    for phase in range(5):
        _set_counts(data, phase)
        for color in range(3):
            _launch(data, reference, color, True, device)
            _launch(data, candidate, color, True, device, reset_bounds=True)
            with test.subTest(phase=phase, color=color):
                _assert_outputs(test, reference, candidate)


def test_self_contact_scheduling_graph(test, device):
    """Read changed active counts when graphs replay wider rows and fused bound resets."""
    data = _fixture(device)
    reference = _outputs(data, device)
    candidate = _outputs(data, device)
    for reset_bounds in (False, True):
        _set_counts(data, 0)
        _launch(data, reference, 0, False, device)
        _launch(data, candidate, 0, True, device, reset_bounds=reset_bounds)
        with wp.ScopedCapture(device=device) as capture:
            _launch(data, reference, 0, False, device)
            _launch(data, candidate, 0, True, device, reset_bounds=reset_bounds)
        for phase in (4, 1, 3, 0):
            _set_counts(data, phase)
            wp.capture_launch(capture.graph)
            with test.subTest(phase=phase, reset_bounds=reset_bounds):
                _assert_outputs(test, reference, candidate)


def _check_solver_scheduling(device):
    test = unittest.TestCase()
    device = wp.get_device(device)
    builder = newton.ModelBuilder(gravity=0.0)
    for height in (0.0, 0.05):
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.01, 0.0, 0.1 if height == 0.0 else -0.1),
            dim_x=1,
            dim_y=1,
            cell_x=0.2,
            cell_y=0.2,
            mass=1.0,
        )
    builder.particle_flags[0] |= newton.ParticleFlags.PROXY
    builder.color()
    model = builder.finalize(device=device)
    primitive_count = max(model.particle_count, model.edge_count)
    crossing_displacements = np.zeros((model.particle_count, 3), dtype=np.float32)
    crossing_displacements[:, 2] = np.where(model.particle_q.numpy()[:, 2] < 0.025, 0.2, -0.2)
    mapping = wp.array(np.arange(model.particle_count, dtype=np.int32), device=device)
    proxy_forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
    modes = wp.DeterministicMode
    cases = (
        (modes.NOT_GUARANTEED, None),
        (modes.NOT_GUARANTEED, modes.RUN_TO_RUN),
        (modes.RUN_TO_RUN, None),
        (modes.NOT_GUARANTEED, modes.GPU_TO_GPU),
        (modes.GPU_TO_GPU, None),
        (modes.GPU_TO_GPU, modes.NOT_GUARANTEED),
    )
    solvers = []
    states = []
    for inherited, explicit in cases:
        wp.config.deterministic = inherited
        solvers.append(
            newton.solvers.SolverVBD(
                model,
                iterations=2,
                particle_enable_self_contact=True,
                particle_enable_tile_solve=False,
                particle_self_contact_radius=0.1,
                particle_self_contact_margin=0.15,
                particle_vertex_contact_buffer_size=13,
                particle_edge_contact_buffer_size=13,
                deterministic=explicit,
            )
        )
        states.append([model.state(), model.state()])

    force_kernels = (
        kernels.accumulate_self_contact_force_and_hessian,
        kernels._accumulate_self_contact_force_and_hessian_wide_with_truncation_reset,
    )
    truncation_kernels = (
        kernels.apply_planar_truncation_parallel_by_collision,
        kernels._apply_planar_truncation_parallel_by_collision_wide,
    )
    harvest_kernel = vbd_coupling_kernels._harvest_vbd_proxy_particle_self_contact_forces_kernel
    launch = wp.launch
    for order in ((0, 1, 3, 2, 4, 5), (5, 4, 2, 3, 1, 0)):
        for index in order:
            solver = solvers[index]
            inherited, explicit = cases[index]
            mode = inherited if explicit is None else explicit
            wide = device.is_cuda and mode == modes.NOT_GUARANTEED
            lanes = 8 if wide else 4
            calls = {"force": 0, "truncation": 0, "harvest": 0}
            computed_bounds = []

            def record_launch(
                kernel,
                *args,
                wide=wide,
                lanes=lanes,
                calls=calls,
                mode=mode,
                solver=solver,
                computed_bounds=computed_bounds,
                **kwargs,
            ):
                if kernel in (*force_kernels, kernels._accumulate_self_contact_force_and_hessian_wide):
                    kind, expected_kernel, block = "force", force_kernels[wide], 128 if wide else 256
                    test.assertEqual(len(kwargs["outputs"]), 3 if wide else 2)
                    if wide:
                        test.assertIs(kwargs["outputs"][-1], solver.truncation_ts)
                elif kernel in truncation_kernels:
                    kind, expected_kernel, block = "truncation", truncation_kernels[wide], 64 if wide else 256
                    np.testing.assert_array_equal(solver.truncation_ts.numpy(), np.ones(model.particle_count))
                elif kernel is harvest_kernel:
                    kind, expected_kernel, block = "harvest", harvest_kernel, None
                else:
                    return launch(kernel, *args, **kwargs)
                calls[kind] += 1
                test.assertIs(kernel, expected_kernel)
                test.assertEqual(kwargs["dim"], primitive_count * (4 if kind == "harvest" else lanes))
                if block is not None:
                    test.assertEqual(kwargs["block_dim"], block)
                options = wp.get_module_options(module=vbd_coupling_kernels if kind == "harvest" else kernels)
                test.assertEqual(options["deterministic"], mode)
                test.assertEqual(options["deterministic_max_records"] > 0, mode != modes.NOT_GUARANTEED)
                result = launch(kernel, *args, **kwargs)
                if kind == "truncation":
                    computed_bounds.append(solver.truncation_ts.numpy())
                return result

            state_in, state_out = states[index]
            with test.subTest(device=str(device), explicit=explicit, inherited=inherited):
                test.assertEqual(solver._self_contact_evaluation_launch_size, primitive_count * lanes)
                test.assertEqual(solver.particle_self_contact_evaluation_kernel_launch_size, primitive_count * 4)
                # Another solver has changed shared module options since this instance was initialized.
                with (
                    mock.patch.object(wp, "launch", side_effect=record_launch),
                    mock.patch.object(solver.truncation_ts, "fill_", wraps=solver.truncation_ts.fill_) as bound_fills,
                ):
                    solver.step(state_in, state_out, None, None, 1.0 / 240.0)
                    expected_fills = 1 if wide else 1 + solver.iterations * len(model.particle_color_groups)
                    test.assertEqual(bound_fills.call_args_list, [mock.call(1.0)] * expected_fills)
                    np.testing.assert_array_equal(
                        computed_bounds[-1].view(np.uint8), solver.truncation_ts.numpy().view(np.uint8)
                    )
                    solver.coupling_harvest_proxy_particle_forces(
                        mapping,
                        proxy_forces,
                        particle_qd_before=state_in.particle_qd,
                        state=state_in,
                        state_out=state_out,
                        contacts=None,
                        dt=1.0 / 240.0,
                    )
                    # A direct call must initialize bounds even for a solver whose force pass can reset them.
                    previous_displacements = solver.particle_displacements.numpy()
                    # Drive the two layers toward one another so retained minima cannot all equal one.
                    solver.particle_displacements.assign(crossing_displacements)
                    solver.truncation_ts.fill_(0.0)
                    bound_fills.reset_mock()
                    solver._penetration_free_truncation()
                    bound_fills.assert_called_once_with(1.0)
                    test.assertTrue(np.any(computed_bounds[-1] < 1.0))
                    np.testing.assert_array_equal(
                        computed_bounds[-1].view(np.uint8), solver.truncation_ts.numpy().view(np.uint8)
                    )
                    solver.particle_displacements.assign(previous_displacements)
                test.assertTrue(all(count > 0 for count in calls.values()), calls)
                test.assertGreater(np.count_nonzero(proxy_forces.numpy()), 0)
                test.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
            states[index].reverse()
        for explicit_index, inherited_index in ((1, 2), (3, 4)):
            for field in ("particle_q", "particle_qd"):
                np.testing.assert_array_equal(
                    getattr(states[explicit_index][0], field).numpy(),
                    getattr(states[inherited_index][0], field).numpy(),
                    err_msg=f"explicit and inherited deterministic {field}",
                )


def test_solver_scheduling(test, device):
    """Preserve deterministic and proxy traversal while selecting wider ordinary CUDA launches."""
    # Deterministic kernel compilation and global module options persist for the process lifetime.
    code = (
        "from newton.tests.test_vbd_self_contact_scheduling import _check_solver_scheduling; "
        f"_check_solver_scheduling({str(device)!r})"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, check=False)
    test.assertEqual(result.returncode, 0, f"Solver scheduling subprocess failed:\n{result.stdout}\n{result.stderr}")


class TestVBDSelfContactScheduling(unittest.TestCase):
    """Check that contact scheduling preserves solver contributions."""


devices = get_test_devices()
add_function_test(
    TestVBDSelfContactScheduling, "test_self_contact_scheduling", test_self_contact_scheduling, devices=devices
)
add_function_test(
    TestVBDSelfContactScheduling,
    "test_self_contact_reset_scheduling",
    test_self_contact_reset_scheduling,
    devices=devices,
)
add_function_test(
    TestVBDSelfContactScheduling,
    "test_self_contact_scheduling_graph",
    test_self_contact_scheduling_graph,
    devices=[device for device in devices if device.is_cuda],
)
add_function_test(TestVBDSelfContactScheduling, "test_solver_scheduling", test_solver_scheduling, devices=devices)


if __name__ == "__main__":
    unittest.main()
