# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the no-ball deck's actual geometry and fixed supports on CPU."""

import unittest

import numpy as np

import newton
from experiments.collision_benchmark.poker_cards_scene import build_scene


class PokerCardsSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build one full authored deck for the CPU checks."""
        cls.scene = build_scene("s4_poker_cards_no_ball", device="cpu")

    def test_free_card_geometry_matches_authored_seed(self):
        """Check every card against the source dimensions and seeded offsets."""
        model = self.scene.model
        self.assertEqual(model.particle_count, 52 * 35)
        self.assertEqual(model.tri_count, 52 * 48)
        self.assertTrue(np.all(model.particle_inv_mass.numpy() > 0.0))
        self.assertTrue(np.all(model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)))
        np.testing.assert_allclose(model.particle_mass.numpy(), 0.0018 / 35, rtol=1e-7)
        np.testing.assert_allclose(model.particle_radius.numpy(), 0.003, rtol=1e-7)

        # Independent planar rotation, retaining the source grid-origin placement.
        grid = np.array([(x * 0.0635 / 4, y * 0.0889 / 6) for y in range(7) for x in range(5)])
        rng = np.random.default_rng(42)
        expected = []
        for card in range(52):
            offset_x, offset_y = rng.uniform(-0.005, 0.005, size=2)
            angle = rng.uniform(-0.1, 0.1)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            xy = grid @ rotation.T + [-0.0635 / 2 + offset_x, -0.0889 / 2 + offset_y]
            expected.append(np.column_stack((xy, np.full(35, 0.25 + card * 0.001))))
        np.testing.assert_allclose(model.particle_q.numpy(), np.concatenate(expected), rtol=0, atol=3e-8)

        triangles = model.tri_indices.numpy().reshape(52, 48, 3)
        self.assertTrue(np.all(triangles // 35 == np.arange(52)[:, None, None]))
        for card, faces in enumerate(triangles):
            np.testing.assert_array_equal(faces - 35 * card, triangles[0])
        np.testing.assert_array_equal(model.tri_materials.numpy()[:, :3], np.tile([1e4, 1e4, 1.0], (2496, 1)))

    def test_only_world_fixed_supports_remain_fixed_during_steps(self):
        """Exclude any hidden sphere or moving deck through physical stepping."""
        scene = self.scene
        model = scene.model
        self.assertEqual(model.body_count, 0)
        self.assertEqual(model.shape_count, 2)
        np.testing.assert_array_equal(model.shape_type.numpy(), [int(newton.GeoType.BOX), int(newton.GeoType.PLANE)])
        np.testing.assert_array_equal(model.shape_body.numpy(), [-1, -1])
        before_transform = model.shape_transform.numpy().copy()
        before_scale = model.shape_scale.numpy().copy()
        np.testing.assert_allclose(before_transform[0, :3], [0, 0, 0.1], rtol=0, atol=2e-9)
        np.testing.assert_allclose(before_scale[0], [0.1, 0.1, 0.1], rtol=0, atol=2e-9)
        q0 = scene.state.particle_q.numpy().copy()
        for _ in range(2):
            scene.step()
        self.assertEqual(scene.time, 2.0 / 60)
        self.assertTrue(np.isfinite(scene.state.particle_q.numpy()).all())
        self.assertFalse(np.array_equal(scene.state.particle_q.numpy(), q0))
        np.testing.assert_array_equal(model.shape_transform.numpy(), before_transform)
        np.testing.assert_array_equal(model.shape_scale.numpy(), before_scale)


if __name__ == "__main__":
    unittest.main()
