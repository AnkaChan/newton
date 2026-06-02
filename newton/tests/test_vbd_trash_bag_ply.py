# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from newton.examples.vbd import example_vbd_trash_bag


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def numpy(self):
        return self.values.copy()

    def zero_(self):
        self.values.fill(0)


class _FakeModel:
    def __init__(self, particle_count, tri_indices=None, edge_indices=None):
        self.particle_count = particle_count
        self.particle_flags = _Array(np.ones(particle_count, dtype=np.int32))
        self.tri_indices = _Array(
            np.empty((0, 3), dtype=np.int32) if tri_indices is None else np.asarray(tri_indices, dtype=np.int32)
        )
        self.edge_indices = _Array(
            np.empty((0, 4), dtype=np.int32) if edge_indices is None else np.asarray(edge_indices, dtype=np.int32)
        )

    def state(self):
        state = mock.Mock()
        state.particle_q = _Array(np.zeros((self.particle_count, 3), dtype=np.float32))
        return state

    def control(self):
        return mock.Mock()


class _FakeBuilder:
    def __init__(self, particle_count=0, tri_indices=None, edge_indices=None):
        self.particle_q = [None] * particle_count
        self.springs = []
        self.spring_rest_length = []
        self.body_count = 0
        self.tri_indices = tri_indices
        self.edge_indices = edge_indices

    def add_cloth_mesh(self, *, vertices, **kwargs):
        del kwargs
        self.particle_q.extend([None] * len(vertices))

    def add_spring(self, i, j, ke, kd, control):
        self.springs.append((i, j, ke, kd, control))
        self.spring_rest_length.append(0.014)

    def add_body(self, **kwargs):
        del kwargs
        body = self.body_count
        self.body_count += 1
        return body

    def add_shape_sphere(self, *args, **kwargs):
        del args, kwargs

    def color(self, **kwargs):
        del kwargs

    def finalize(self):
        return _FakeModel(len(self.particle_q), self.tri_indices, self.edge_indices)


class TestVbdTrashBagPlyExport(unittest.TestCase):
    def test_parser_adds_ply_export_options(self):
        parser = example_vbd_trash_bag.Example.create_parser()

        args = parser.parse_args(["--export-ply", "--ply-output-dir", "mesh_frames"])

        self.assertTrue(args.export_ply)
        self.assertEqual(args.ply_output_dir, "mesh_frames")

    def test_export_ply_frame_combines_bag_and_rope_meshes(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        info = {
            "bag_start": 0,
            "bag_count": 3,
            "rope_start": 3,
            "rope_count": 3,
            "bag_faces": np.array([[0, 1, 2]], dtype=np.int32),
            "rope_faces": np.array([[0, 1, 2]], dtype=np.int32),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = example_vbd_trash_bag._export_ply_frame(tmp_dir, 12, positions, info)

            self.assertEqual(path, os.path.join(tmp_dir, "trash_bag_000012.ply"))
            with open(path, encoding="utf-8") as file:
                lines = file.read().splitlines()

        self.assertEqual(
            lines,
            [
                "ply",
                "format ascii 1.0",
                "element vertex 6",
                "property float x",
                "property float y",
                "property float z",
                "element face 2",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "1 0 1",
                "0 1 1",
                "3 0 1 2",
                "3 3 4 5",
            ],
        )


class TestVbdTrashBagContacts(unittest.TestCase):
    def test_builds_tunnel_seam_collision_filters_for_vt_tv_and_edges(self):
        model = _FakeModel(
            particle_count=14,
            tri_indices=np.array(
                [
                    [0, 1, 4],
                    [2, 3, 5],
                    [8, 9, 12],
                    [10, 11, 13],
                ],
                dtype=np.int32,
            ),
            edge_indices=np.array(
                [
                    [-1, -1, 0, 1],
                    [-1, -1, 2, 3],
                    [-1, -1, 8, 9],
                    [-1, -1, 10, 11],
                    [-1, -1, 0, 6],
                    [-1, -1, 2, 7],
                ],
                dtype=np.int32,
            ),
        )

        vertex_filter, edge_filter = example_vbd_trash_bag._build_tunnel_seam_contact_filters(
            model,
            np.array(
                [
                    [0, 2],
                    [1, 3],
                    [8, 10],
                    [9, 11],
                ],
                dtype=np.int32,
            ),
        )

        self.assertIn(1, vertex_filter[0])
        self.assertIn(1, vertex_filter[1])
        self.assertIn(0, vertex_filter[2])
        self.assertIn(0, vertex_filter[3])
        self.assertIn(3, vertex_filter[8])
        self.assertIn(2, vertex_filter[10])
        self.assertIn(1, edge_filter[0])
        self.assertIn(0, edge_filter[1])
        self.assertIn(3, edge_filter[2])
        self.assertIn(2, edge_filter[3])
        self.assertIn(5, edge_filter[4])
        self.assertIn(4, edge_filter[5])

    def test_tunnel_seam_collision_filters_do_not_exclude_rope_primitives(self):
        rope_start = 8
        model = _FakeModel(
            particle_count=12,
            tri_indices=np.array(
                [
                    [0, 1, 4],
                    [2, 3, 5],
                    [rope_start, rope_start + 1, rope_start + 2],
                ],
                dtype=np.int32,
            ),
            edge_indices=np.array(
                [
                    [-1, -1, 0, 1],
                    [-1, -1, 2, 3],
                    [-1, -1, rope_start, rope_start + 1],
                ],
                dtype=np.int32,
            ),
        )

        vertex_filter, edge_filter = example_vbd_trash_bag._build_tunnel_seam_contact_filters(
            model,
            np.array(
                [
                    [0, 2],
                    [1, 3],
                ],
                dtype=np.int32,
            ),
        )
        tri_indices = model.tri_indices.numpy()
        edge_indices = model.edge_indices.numpy()

        self.assertFalse(any(vertex >= rope_start for vertex in vertex_filter))
        self.assertFalse(
            any(
                np.any(tri_indices[triangle] >= rope_start)
                for triangles in vertex_filter.values()
                for triangle in triangles
            )
        )
        self.assertFalse(any(np.any(edge_indices[edge_id, 2:4] >= rope_start) for edge_id in edge_filter))
        self.assertFalse(
            any(np.any(edge_indices[edge_id, 2:4] >= rope_start) for edges in edge_filter.values() for edge_id in edges)
        )

    def test_particle_self_contact_radius_matches_cloth_collision_scale(self):
        params = example_vbd_trash_bag.PARAMS

        self.assertGreaterEqual(params["particle_self_contact_radius"], params["particle_radius"])
        self.assertGreaterEqual(params["particle_self_contact_margin"], 2.0 * params["particle_self_contact_radius"])

    def test_build_model_keeps_springs_within_bag_mesh(self):
        builder = _FakeBuilder()

        info = example_vbd_trash_bag.build_model(builder, example_vbd_trash_bag.PARAMS, seed=42)

        self.assertEqual(len(builder.springs), info["num_tunnel_springs"])
        self.assertEqual(info["num_tethers"], 0)
        for i, j, _, _, _ in builder.springs:
            self.assertGreaterEqual(i, info["bag_start"])
            self.assertLess(i, info["rope_start"])
            self.assertGreaterEqual(j, info["bag_start"])
            self.assertLess(j, info["rope_start"])

    def test_build_model_sets_tunnel_springs_to_stitched_rest_length(self):
        builder = _FakeBuilder()

        info = example_vbd_trash_bag.build_model(builder, example_vbd_trash_bag.PARAMS, seed=42)

        self.assertEqual(len(builder.spring_rest_length), info["num_tunnel_springs"])
        self.assertTrue(builder.spring_rest_length)
        self.assertTrue(
            all(
                rest_length == example_vbd_trash_bag.PARAMS["closure_rest_length"]
                for rest_length in builder.spring_rest_length
            )
        )

    def test_setup_sim_enables_particle_self_contact(self):
        captured_solver_kwargs = {}

        class _FakeSolver:
            def __init__(self, **kwargs):
                captured_solver_kwargs.update(kwargs)

        class _FakePipeline:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def contacts(self):
                return mock.Mock()

        info = {
            "right_idx": np.array([0], dtype=np.int32),
            "left_idx": np.array([1], dtype=np.int32),
            "tunnel_spring_pairs": np.array(
                [
                    [0, 2],
                    [1, 3],
                    [8, 10],
                    [9, 11],
                ],
                dtype=np.int32,
            ),
        }
        builder = _FakeBuilder(
            particle_count=14,
            tri_indices=np.array(
                [
                    [0, 1, 4],
                    [2, 3, 5],
                    [8, 9, 12],
                    [10, 11, 13],
                ],
                dtype=np.int32,
            ),
            edge_indices=np.array(
                [
                    [-1, -1, 0, 1],
                    [-1, -1, 2, 3],
                    [-1, -1, 8, 9],
                    [-1, -1, 10, 11],
                ],
                dtype=np.int32,
            ),
        )
        with (
            mock.patch.object(example_vbd_trash_bag.newton.solvers, "SolverVBD", _FakeSolver),
            mock.patch.object(example_vbd_trash_bag.newton, "CollisionPipeline", _FakePipeline),
        ):
            example_vbd_trash_bag.setup_sim(builder, info, example_vbd_trash_bag.PARAMS)

        self.assertTrue(captured_solver_kwargs["particle_enable_self_contact"])
        self.assertEqual(
            captured_solver_kwargs["particle_self_contact_radius"],
            example_vbd_trash_bag.PARAMS["particle_self_contact_radius"],
        )
        self.assertEqual(
            captured_solver_kwargs["particle_self_contact_margin"],
            example_vbd_trash_bag.PARAMS["particle_self_contact_margin"],
        )
        self.assertGreater(
            captured_solver_kwargs["particle_self_contact_margin"],
            captured_solver_kwargs["particle_self_contact_radius"],
        )
        self.assertIn(1, captured_solver_kwargs["particle_external_vertex_contact_filtering_map"][0])
        self.assertIn(1, captured_solver_kwargs["particle_external_edge_contact_filtering_map"][0])


class TestVbdTrashBagPreroll(unittest.TestCase):
    def test_preroll_runs_hidden_frames_with_velocity_kill_enabled(self):
        events = []
        state = types.SimpleNamespace(
            particle_qd=mock.Mock(),
            body_qd=mock.Mock(),
        )
        example = types.SimpleNamespace(
            params={"preroll_frames": 10},
            state_0=state,
            frame=3,
            sim_time=1.25,
        )

        def simulate(*, zero_velocities_each_step=False):
            events.append(zero_velocities_each_step)

        example.simulate = simulate

        example_vbd_trash_bag.Example._preroll(example)

        self.assertEqual(events, [True] * 10)
        self.assertEqual(example.frame, 3)
        self.assertEqual(example.sim_time, 1.25)

    def test_simulate_zeros_velocities_after_each_substep_when_requested(self):
        state_a = types.SimpleNamespace(
            particle_q=object(),
            particle_qd=mock.Mock(),
            body_qd=mock.Mock(),
            clear_forces=mock.Mock(),
        )
        state_b = types.SimpleNamespace(
            particle_q=object(),
            particle_qd=mock.Mock(),
            body_qd=mock.Mock(),
            clear_forces=mock.Mock(),
        )
        example = types.SimpleNamespace(
            sim_substeps=3,
            right=types.SimpleNamespace(shape=(0,)),
            left=types.SimpleNamespace(shape=(0,)),
            right_orig=object(),
            left_orig=object(),
            state_0=state_a,
            state_1=state_b,
            viewer=mock.Mock(),
            pipeline=mock.Mock(),
            solver=mock.Mock(),
            control=object(),
            contacts=object(),
            sim_dt=1.0 / 60.0,
        )
        example._cinch = lambda: (object(), object())
        example._zero_velocities = types.MethodType(example_vbd_trash_bag.Example._zero_velocities, example)

        with mock.patch.object(example_vbd_trash_bag.wp, "launch"):
            example_vbd_trash_bag.Example.simulate(example, zero_velocities_each_step=True)

        self.assertEqual(state_a.particle_qd.zero_.call_count + state_b.particle_qd.zero_.call_count, 3)
        self.assertEqual(state_a.body_qd.zero_.call_count + state_b.body_qd.zero_.call_count, 3)


if __name__ == "__main__":
    unittest.main()
