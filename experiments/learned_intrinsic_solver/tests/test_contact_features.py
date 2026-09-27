# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the schema-4 contact token layout, per-cell capping and padding on the CPU."""

import importlib.util
import math
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is an optional dependency")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.contact_features import RADIUS_RATIO_CAP, build_contact_tokens
from experiments.learned_intrinsic_solver.features import CONTACT_TOKEN_DIM

CELL_SIZE = 0.5
RADIUS = 0.1
KAPPA, BETA, MU = 3.0, 0.25, 0.4


def _rotation_z_90() -> torch.Tensor:
    """Proper rotation by +90 degrees about z with world columns."""
    return torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class TestBuildContactTokens(unittest.TestCase):
    def setUp(self):
        """Hand-build two cells, three surface samples and four pairs (one masked with poisoned rows)."""
        self.frames = torch.stack((torch.eye(3), _rotation_z_90()))[None]
        self.centers = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        self.face_cell_index = torch.tensor([0, 1, 1])
        self.samples = torch.tensor([[[0.1, -0.25, 0.0], [1.25, 0.0, 0.0], [1.0, 0.0, 0.25]]])
        self.starts = torch.tensor([[[0.1, -0.20, 0.0], [1.3, 0.0, 0.0], [1.0, 0.0, 0.25]]])
        nan = float("nan")
        self.pairs = {
            "sample_index": torch.tensor([[0, 1, -1, 2]]),
            "kind": torch.tensor([[0, 1, 7, 1]]),
            "partner_point": torch.tensor([[[0.1, -0.3, 0.0], [1.4, 0.0, 0.0], [nan, nan, nan], [1.0, 0.0, 0.3]]]),
            "partner_normal": torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -1.0]]]),
            "partner_radius": torch.tensor([[1e9, 0.2, nan, 0.05]]),
            "pair_mask": torch.tensor([[True, True, False, True]]),
        }
        self.coefficients = {
            "contact_kappa": torch.tensor([KAPPA]),
            "contact_beta": torch.tensor([BETA]),
            "contact_mu": torch.tensor([MU]),
        }

    def _build(self, tokens_per_cell=2, **overrides):
        arguments = {
            "frames": self.frames,
            "cell_centers": self.centers,
            "cell_size": CELL_SIZE,
            "radius": RADIUS,
            "face_cell_index": self.face_cell_index,
            "sample_positions": self.samples,
            "sample_start_positions": self.starts,
            **self.pairs,
            **self.coefficients,
            "tokens_per_cell": tokens_per_cell,
        }
        arguments.update(overrides)
        return build_contact_tokens(**arguments)

    def test_layout_in_owner_cell_frames(self):
        """Write every channel in the owning cell's frame with the documented signs and scalings."""
        tokens, mask = self._build(tokens_per_cell=2)
        self.assertEqual(tokens.shape, (1, 2, 2, CONTACT_TOKEN_DIM))
        self.assertEqual(mask.shape, (1, 2, 2))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.tolist(), [[[True, False], [True, True]]])
        self.assertTrue(torch.isfinite(tokens).all())
        shared = [math.log1p(KAPPA), BETA, MU]
        # Pair 0: sample 0 against the plane, owner cell 0 with the identity frame; approaching at 0.5 r per step.
        plane = tokens[0, 0, 0]
        torch.testing.assert_close(plane[0:3], torch.tensor([0.2, -0.5, 0.0]))
        torch.testing.assert_close(plane[3:6], torch.tensor([0.2, -0.6, 0.0]))
        torch.testing.assert_close(plane[6:9], torch.tensor([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(plane[9].item(), 0.5, places=5)
        self.assertAlmostEqual(plane[10].item(), 0.5, places=5)
        self.assertEqual(plane[11].item(), RADIUS_RATIO_CAP)
        torch.testing.assert_close(plane[12:15], torch.tensor(shared))
        self.assertEqual(plane[15:18].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(plane[18].item(), 0.0)
        # Pair 1: sample 1 against a point, owner cell 1 rotated by 90 degrees; separating at -0.5 r per step.
        point = tokens[0, 1, 0]
        torch.testing.assert_close(point[0:3], torch.tensor([0.0, -0.5, 0.0]))
        torch.testing.assert_close(point[3:6], torch.tensor([0.0, -0.8, 0.0]))
        torch.testing.assert_close(point[6:9], torch.tensor([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(point[9].item(), 1.5, places=5)
        self.assertAlmostEqual(point[10].item(), -0.5, places=5)
        self.assertAlmostEqual(point[11].item(), 2.0, places=5)
        torch.testing.assert_close(point[12:15], torch.tensor(shared))
        self.assertEqual(point[15:18].tolist(), [0.0, 1.0, 0.0])
        # Pair 3: sample 2 against a small point below its own radius, resting.
        small = tokens[0, 1, 1]
        torch.testing.assert_close(small[0:3], torch.tensor([0.0, 0.0, 0.5]))
        torch.testing.assert_close(small[3:6], torch.tensor([0.0, 0.0, 0.6]))
        torch.testing.assert_close(small[6:9], torch.tensor([0.0, 0.0, -1.0]))
        self.assertAlmostEqual(small[9].item(), 0.5, places=5)
        self.assertEqual(small[10].item(), 0.0)
        self.assertAlmostEqual(small[11].item(), 0.5, places=5)
        self.assertEqual(small[15:18].tolist(), [0.0, 1.0, 0.0])
        # Padding is exactly zero.
        self.assertEqual(torch.count_nonzero(tokens[0, 0, 1]).item(), 0)

    def test_cap_keeps_first_pairs_in_pair_order(self):
        """Keep the first tokens_per_cell pairs of a cell in pair order and drop the rest deterministically."""
        full, _ = self._build(tokens_per_cell=2)
        capped, capped_mask = self._build(tokens_per_cell=1)
        self.assertEqual(capped.shape, (1, 2, 1, CONTACT_TOKEN_DIM))
        self.assertEqual(capped_mask.tolist(), [[[True], [True]]])
        torch.testing.assert_close(capped[0, 1, 0], full[0, 1, 0], rtol=0, atol=0)
        torch.testing.assert_close(capped[0, 0, 0], full[0, 0, 0], rtol=0, atol=0)
        again, again_mask = self._build(tokens_per_cell=1)
        torch.testing.assert_close(again, capped, rtol=0, atol=0)
        self.assertEqual(again_mask.tolist(), capped_mask.tolist())
        # Reordering the two cell-1 pairs changes which one survives the cap.
        order = torch.tensor([0, 3, 2, 1])
        swapped = {name: value[:, order] for name, value in self.pairs.items()}
        reordered, _ = self._build(tokens_per_cell=1, **swapped)
        torch.testing.assert_close(reordered[0, 1, 0], full[0, 1, 1], rtol=0, atol=0)
        wide, wide_mask = self._build(tokens_per_cell=5)
        self.assertEqual(wide_mask.sum().item(), 3)
        torch.testing.assert_close(wide[:, :, :2], full, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(wide[0, :, 2:]).item(), 0)

    def test_empty_pairs_and_all_masked_give_zero_tokens(self):
        """Return zero tokens with a False mask for Q = 0 and for a batch without valid pairs."""
        empty = {
            "sample_index": torch.zeros((1, 0), dtype=torch.int64),
            "kind": torch.zeros((1, 0), dtype=torch.int64),
            "partner_point": torch.zeros((1, 0, 3)),
            "partner_normal": torch.zeros((1, 0, 3)),
            "partner_radius": torch.zeros((1, 0)),
            "pair_mask": torch.zeros((1, 0), dtype=torch.bool),
        }
        tokens, mask = self._build(tokens_per_cell=3, **empty)
        self.assertEqual(tokens.shape, (1, 2, 3, CONTACT_TOKEN_DIM))
        self.assertEqual(torch.count_nonzero(tokens).item(), 0)
        self.assertFalse(mask.any())
        masked, masked_mask = self._build(pair_mask=torch.zeros((1, 4), dtype=torch.bool))
        self.assertEqual(torch.count_nonzero(masked).item(), 0)
        self.assertFalse(masked_mask.any())

    def test_batch_members_are_independent_and_outputs_detached(self):
        """Handle per-object validity independently and never carry gradients."""
        pairs = {name: torch.cat((value, value)) for name, value in self.pairs.items()}
        pairs["pair_mask"] = torch.tensor([[True, True, False, True], [False, True, False, False]])
        frames = torch.cat((self.frames, self.frames)).requires_grad_()
        samples = torch.cat((self.samples, self.samples)).requires_grad_()
        coefficients = {name: torch.cat((value, value)) for name, value in self.coefficients.items()}
        tokens, mask = self._build(
            frames=frames,
            cell_centers=torch.cat((self.centers, self.centers)),
            sample_positions=samples,
            sample_start_positions=torch.cat((self.starts, self.starts)),
            **pairs,
            **coefficients,
        )
        self.assertFalse(tokens.requires_grad)
        self.assertFalse(mask.requires_grad)
        single, single_mask = self._build()
        torch.testing.assert_close(tokens[0], single[0], rtol=0, atol=0)
        self.assertEqual(mask[0].tolist(), single_mask[0].tolist())
        self.assertEqual(mask[1].tolist(), [[False, False], [True, False]])
        torch.testing.assert_close(tokens[1, 1, 0], single[0, 1, 0], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(tokens[1, 0]).item(), 0)

    def test_rejects_invalid_inputs(self):
        """Refuse bad slot counts, kinds, sample indices, owner indices, dtypes and shapes."""
        with self.assertRaises(ValueError):
            self._build(tokens_per_cell=0)
        with self.assertRaises(ValueError):
            self._build(tokens_per_cell=True)
        with self.assertRaises(ValueError):
            self._build(kind=torch.tensor([[0, 3, 0, 1]]))
        with self.assertRaises(ValueError):
            self._build(sample_index=torch.tensor([[0, 3, -1, 2]]))
        with self.assertRaises(ValueError):
            self._build(face_cell_index=torch.tensor([0, 2, 1]))
        with self.assertRaises(TypeError):
            self._build(face_cell_index=self.face_cell_index.int())
        with self.assertRaises(TypeError):
            self._build(pair_mask=self.pairs["pair_mask"].float())
        with self.assertRaises(TypeError):
            self._build(sample_index=self.pairs["sample_index"].float())
        with self.assertRaises(ValueError):
            self._build(partner_point=self.pairs["partner_point"][:, :3])
        with self.assertRaises(ValueError):
            self._build(cell_centers=self.centers[:, :1])
        with self.assertRaises(ValueError):
            self._build(contact_kappa=torch.tensor([KAPPA, KAPPA]))
        with self.assertRaises(ValueError):
            self._build(radius=0.0)
        with self.assertRaises(ValueError):
            self._build(cell_size=math.nan)
        with self.assertRaises(TypeError):
            self._build(sample_positions=self.samples.double())


if __name__ == "__main__":
    unittest.main()
