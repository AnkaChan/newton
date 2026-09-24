# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check cuboid graph geometry and cell-frame transport on the CPU."""

import importlib.util
import unittest

from experiments.learned_intrinsic_solver.network_geometry import build_edge_features, build_grid_neighborhood

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is an optional dependency")


class TestGridNeighborhood(unittest.TestCase):
    def test_hop_one_uses_literal_z_fast_neighbors(self):
        """Find face, edge, and corner neighbors without wrapping across grid bounds."""
        import torch

        indices, valid = build_grid_neighborhood((3, 3, 5), 1)
        self.assertEqual(indices.shape, (45, 27))
        self.assertEqual(indices.dtype, torch.long)
        self.assertEqual(valid.dtype, torch.bool)
        self.assertEqual(sorted(indices[0, valid[0]].tolist()), [0, 1, 5, 6, 15, 16, 20, 21])
        self.assertEqual(
            sorted(indices[22, valid[22]].tolist()),
            [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23, 26, 27, 28, 31, 32, 33, 36, 37, 38, 41, 42, 43],
        )
        torch.testing.assert_close(indices[:, 0], torch.arange(45))
        self.assertTrue(valid[:, 0].all())
        self.assertTrue((indices[~valid] == 0).all())

    def test_hop_two_is_an_exact_shell(self):
        """Include distance-two cells while excluding distance-one neighbors."""
        indices, valid = build_grid_neighborhood((5, 5, 5), 2)
        self.assertEqual(indices.shape, (125, 99))
        self.assertEqual(
            sorted(indices[0, valid[0]].tolist()),
            [0, 2, 7, 10, 11, 12, 27, 32, 35, 36, 37, 50, 51, 52, 55, 56, 57, 60, 61, 62],
        )
        excluded_hop_one = {
            31,
            32,
            33,
            36,
            37,
            38,
            41,
            42,
            43,
            56,
            57,
            58,
            61,
            63,
            66,
            67,
            68,
            81,
            82,
            83,
            86,
            87,
            88,
            91,
            92,
            93,
        }
        self.assertEqual(sorted(indices[62, valid[62]].tolist()), sorted(set(range(125)) - excluded_hop_one))

    def test_hop_four_keeps_the_full_shell(self):
        """Retain all 386 distance-four neighbors and mask the corner boundary."""
        import torch

        indices, valid = build_grid_neighborhood((9, 9, 9), 4)
        self.assertEqual(indices.shape, (729, 387))
        self.assertEqual(int(valid[364].sum()), 387)
        self.assertEqual(len(set(indices[364].tolist())), 387)
        corner_ids = set(indices[0, valid[0]].tolist())
        self.assertEqual(len(corner_ids), 62)
        self.assertTrue({0, 4, 36, 40, 324, 328, 360, 364}.issubset(corner_ids))
        self.assertNotIn(273, corner_ids)
        neighbor = indices[364, 1:]
        coordinates = torch.stack((neighbor // 81, neighbor // 9 % 9, neighbor % 9), dim=-1)
        self.assertTrue(((coordinates - 4).abs().amax(dim=-1) == 4).all())

    def test_singleton_retains_only_self(self):
        """Preserve a valid self slot when every shell neighbor is outside."""
        for hop, size in ((1, 27), (2, 99), (4, 387)):
            with self.subTest(hop=hop):
                indices, valid = build_grid_neighborhood((1, 1, 1), hop, device="cpu")
                self.assertEqual(indices.shape, (1, size))
                self.assertEqual(int(valid.sum()), 1)
                self.assertTrue(valid[0, 0])
                self.assertTrue((indices == 0).all())

    def test_reject_invalid_counts_and_hops(self):
        """Reject dimensions and hop distances that cannot define a cuboid graph."""
        for counts in ((1, 0, 2), (1, -1, 2), (1, 1.5, 2), (1, True, 2), (1, 2)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                build_grid_neighborhood(counts, 1)
        for hop in (0, -1, 1.5, True):
            with self.subTest(hop=hop), self.assertRaises(ValueError):
                build_grid_neighborhood((1, 1, 1), hop)


class TestEdgeFeatures(unittest.TestCase):
    def setUp(self):
        import torch

        self.rest = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float32)
        self.current = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]]], dtype=torch.float32)
        self.frames = torch.tensor(
            [
                [
                    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        )
        self.axes = torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.5, 0.25], [0.0, 0.25, 2.0]],
                    [[2.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 4.0]],
                ]
            ],
            dtype=torch.float32,
        )
        self.indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.mask = torch.ones((2, 2), dtype=torch.bool)

    def _features(self, **overrides):
        arguments = {
            "rest_centers": self.rest,
            "current_centers": self.current,
            "frames": self.frames,
            "local_axes": self.axes,
            "cell_size": 2.0,
            "neighbor_indices": self.indices,
            "neighbor_mask": self.mask,
        }
        arguments.update(overrides)
        return build_edge_features(**arguments)

    def test_known_receiver_frame_and_axis_transport(self):
        """Match hand-computed offsets, relative frames, and transported columns."""
        import torch

        features = self._features()
        self.assertEqual(features.shape, (1, 2, 2, 24))
        self.assertEqual(features.dtype, torch.float32)
        expected = torch.tensor(
            [
                0.0,
                0.0,
                1.0,
                2.0,
                -1.0,
                3.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.5,
                3.0,
                0.0,
                0.0,
                0.0,
                -4.0,
                -2.0,
                -0.5,
                0.0,
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(features[0, 0, 1], expected, rtol=0, atol=0)

    def test_self_keeps_identity_frame_and_own_axes(self):
        """Represent self as actual geometry instead of an all-zero edge."""
        import torch

        features = self._features()
        for cell in (0, 1):
            with self.subTest(cell=cell):
                torch.testing.assert_close(features[0, cell, 0, :6], torch.zeros(6), rtol=0, atol=0)
                torch.testing.assert_close(features[0, cell, 0, 6:15], torch.eye(3).flatten(), rtol=0, atol=0)
                torch.testing.assert_close(features[0, cell, 0, 15:], self.axes[0, cell].flatten(), rtol=0, atol=0)

    def test_global_rigid_transform_leaves_two_batch_features_unchanged(self):
        """Cancel a common world rotation and translation in every batch item."""
        import torch

        current = torch.cat((self.current, self.current * 1.25 + 0.4), dim=0)
        frames = self.frames.expand(2, -1, -1, -1).clone()
        axes = self.axes.expand(2, -1, -1, -1).clone()
        rotation = torch.tensor([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
        translation = torch.tensor([4.2, -1.3, 8.0])
        before = self._features(current_centers=current, frames=frames, local_axes=axes)
        after = self._features(
            current_centers=current @ rotation.T + translation,
            frames=rotation @ frames,
            local_axes=axes,
        )
        self.assertEqual(before.shape, (2, 2, 2, 24))
        torch.testing.assert_close(after, before, rtol=2e-6, atol=2e-6)

    def test_masked_sentinels_are_safe_and_zero(self):
        """Mask invalid IDs before gathering and return zero features for padding."""
        import torch

        indices = torch.tensor([[0, 1, -1, 999], [1, 0, 777, -7]], dtype=torch.long)
        mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
        features = self._features(neighbor_indices=indices, neighbor_mask=mask)
        torch.testing.assert_close(features[:, :, :2], self._features(), rtol=0, atol=0)
        torch.testing.assert_close(features[:, :, 2:], torch.zeros((1, 2, 2, 24)), rtol=0, atol=0)

    def test_detach_frames_but_keep_geometry_gradients(self):
        """Block frame derivatives while retaining position and local-axis paths."""
        import torch

        rest = self.rest.clone().requires_grad_()
        current = self.current.clone().requires_grad_()
        frames = self.frames.clone().requires_grad_()
        axes = self.axes.clone().requires_grad_()
        features = self._features(rest_centers=rest, current_centers=current, frames=frames, local_axes=axes)
        features[0, 0, 1].square().sum().backward()
        self.assertIsNone(frames.grad)
        for name, tensor in (("rest", rest), ("current", current), ("axes", axes)):
            with self.subTest(name=name):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(float(tensor.grad.abs().sum()), 0.0)

    def test_reject_incompatible_shapes_and_dtypes(self):
        """Reject ambiguous batches, malformed masks, and silent dtype promotion."""
        import torch

        bad_shapes = (
            {"rest_centers": self.rest[None]},
            {"current_centers": self.current[0]},
            {"frames": self.frames[:, :1]},
            {"local_axes": self.axes[..., :2]},
            {"neighbor_indices": self.indices[:, :1]},
            {"neighbor_mask": self.mask[None]},
        )
        for override in bad_shapes:
            with self.subTest(override=tuple(override)), self.assertRaises(ValueError):
                self._features(**override)
        bad_dtypes = (
            {"rest_centers": self.rest.to(torch.int32)},
            {"current_centers": self.current.to(torch.float64)},
            {"neighbor_indices": self.indices.to(torch.int32)},
            {"neighbor_mask": self.mask.to(torch.int64)},
        )
        for override in bad_dtypes:
            with self.subTest(override=tuple(override)), self.assertRaises(TypeError):
                self._features(**override)
        for size in (0.0, -2.0, float("inf"), float("nan"), True):
            with self.subTest(cell_size=size), self.assertRaises(ValueError):
                self._features(cell_size=size)


if __name__ == "__main__":
    unittest.main()
