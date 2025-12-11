import unittest

import numpy as np
import warp as wp

from newton._src.sim.builder import ModelBuilder
from newton._src.solvers.vbd.solver_vbd import SolverVBD
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_model_with_soft_mesh(vertices: list[tuple[float, float, float]], tets: np.ndarray):
    """Use add_soft_mesh (full builder path) to create a soft-body model."""
    builder = ModelBuilder()
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=vertices,
        indices=tets.flatten().tolist(),
        density=1.0,
        k_mu=1.0,
        k_lambda=1.0,
        k_damp=0.0,
    )
    builder.color()
    return builder.finalize(device="cpu")


def _expected_tet_adjacency(particle_count: int, tet_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate adjacency exactly as the kernels do: append (tet_id, local_order) per vertex."""
    buckets = [[] for _ in range(particle_count)]
    for tet_id, tet in enumerate(tet_indices):
        for local_order, v in enumerate(tet):
            buckets[int(v)].append((tet_id, local_order))
    offsets = [0]
    flat = []
    for b in buckets:
        offsets.append(offsets[-1] + 2 * len(b))
        for tet_id, order in b:
            flat.extend([tet_id, order])
    return np.array(offsets, dtype=np.int32), np.array(flat, dtype=np.int32)


def _assert_adjacency_matches_tets(test, adjacency, tet_indices: np.ndarray):
    """Check each recorded (tet_id, local_order) really maps back to the vertex being visited."""
    offsets = adjacency.v_adj_tets_offsets.numpy()
    flat = adjacency.v_adj_tets.numpy()
    particle_count = len(offsets) - 1
    for v in range(particle_count):
        start, end = offsets[v], offsets[v + 1]
        entries = flat[start:end].reshape(-1, 2)
        for tet_id, local_order in entries:
            test.assertTrue(
                tet_indices[tet_id, local_order] == v, f"vertex {v} mismatch tet {tet_id} order {local_order}"
            )


class TestSoftBody(unittest.TestCase):
    pass


def test_tet_adjacency_single_tet(test, device, solver):
    tet_indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    particles = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    model = _build_model_with_soft_mesh(particles, tet_indices)

    solver = SolverVBD()
    solver.model = model
    solver.device = model.device

    adjacency = solver.compute_force_element_adjacency(model)

    exp_offsets, exp_flat = _expected_tet_adjacency(4, tet_indices)
    np.testing.assert_array_equal(adjacency.v_adj_tets_offsets.numpy(), exp_offsets)
    np.testing.assert_array_equal(adjacency.v_adj_tets.numpy(), exp_flat)
    _assert_adjacency_matches_tets(adjacency, tet_indices)


def test_tet_adjacency_complex_pyramid(test, device):
    # Pyramid-like fan around apex 4 with a quadrilateral base (0,1,2,3) split into four tets:
    # (0,1,2,4), (0,2,3,4), (0,3,1,4), (1,3,2,4)
    # fmt: off
    tet_indices = np.array(
        [
            [0, 1, 3, 9],
            [1, 4, 3, 13],
            [1, 3, 9, 13],
            [3, 9, 13, 12],
            [1, 9, 10, 13],

            [1, 2, 4, 10],
            [2, 5, 4, 14],
            [2, 4, 10, 14],
            [4, 10, 14, 13],
            [2, 10, 11, 14],

            [3, 4, 6, 12],
            [4, 7, 6, 16],
            [4, 6, 12, 16],
            [6, 12, 16, 15],
            [4, 12, 13, 16],

            [4, 5, 7, 13],
            [5, 8, 7, 17],
            [5, 7, 13, 17],
            [7, 13, 17, 16],
            [5, 13, 14, 17],
        ],
        dtype=np.int32,
    )

    particles = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (2.0, 0.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (1.0, 1.0, 0.0),  # 4
        (2.0, 1.0, 0.0),  # 5
        (0.0, 2.0, 0.0),  # 6
        (1.0, 2.0, 0.0),  # 7
        (2.0, 2.0, 0.0),  # 8

        (0.0, 0.0, 1.0),  # 9
        (1.0, 0.0, 1.0),  # 10
        (2.0, 0.0, 1.0),  # 11
        (0.0, 1.0, 1.0),  # 12
        (1.0, 1.0, 1.0),  # 13
        (2.0, 1.0, 1.0),  # 14
        (0.0, 2.0, 1.0),  # 15
        (1.0, 2.0, 1.0),  # 16
        (2.0, 2.0, 1.0),  # 17
    ]
    # fmt: on

    model = _build_model_with_soft_mesh(particles, tet_indices)

    solver = SolverVBD(model)

    adjacency = solver.compute_force_element_adjacency(model)

    exp_offsets, exp_flat = _expected_tet_adjacency(5, tet_indices)
    np.testing.assert_array_equal(adjacency.v_adj_tets_offsets.numpy(), exp_offsets)
    np.testing.assert_array_equal(adjacency.v_adj_tets.numpy(), exp_flat)
    _assert_adjacency_matches_tets(test, adjacency, tet_indices)


devices = get_test_devices(mode="basic")
add_function_test(
    TestSoftBody, "test_tet_adjacency_complex_pyramid", test_tet_adjacency_complex_pyramid, devices=devices
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
