# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for contact partner sampling and brute-force pair detection."""

import importlib.util
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is an optional dependency")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import contact_scene
from experiments.learned_intrinsic_solver.contact_geometry import exposed_face_samples, sample_points
from experiments.learned_intrinsic_solver.contact_scene import (
    KIND_PLANE,
    KIND_POINT,
    PLANE_PARTNER_RADIUS,
    POINT_BOX_DEPTH_MARGIN,
    POINT_BOX_LATERAL_MARGIN,
    POINT_CLEARANCE_CELLS,
    POINT_NORMAL_MAX_ANGLE,
    ContactPairs,
    ContactPartners,
    contact_stiffness_floor,
    detect_contacts,
    sample_contact_partners,
)

CELL_SIZE = 0.025
TIME_STEP = 1.0 / 300.0
RADIUS = 0.5 * CELL_SIZE
YOUNGS_MODULUS = 1e5


def _sample(seed: int = 0, **overrides) -> ContactPartners:
    rest = generate_cuboid((2, 2, 3), cell_size=CELL_SIZE)
    kwargs = {
        "master_seed": 11,
        "youngs_modulus": YOUNGS_MODULUS,
        "cell_size": CELL_SIZE,
        "time_step": TIME_STEP,
        **overrides,
    }
    kwargs.setdefault("seed", seed)
    return sample_contact_partners(rest, **kwargs)


def _points_partners(positions, normals, radii, *, plane: bool = False, plane_y: float = 0.0) -> ContactPartners:
    return ContactPartners(
        plane_present=plane,
        plane_point=torch.tensor([0.0, plane_y, 0.0]),
        plane_normal=torch.tensor([0.0, 1.0, 0.0]),
        point_positions=torch.tensor(positions, dtype=torch.float32).reshape(-1, 3),
        point_normals=torch.tensor(normals, dtype=torch.float32).reshape(-1, 3),
        point_radii=torch.tensor(radii, dtype=torch.float32).reshape(-1),
        ke=1.0,
        kd=0.0,
        mu=0.0,
    )


def _detect(positions, velocities=None, partners=None, **kwargs) -> ContactPairs:
    positions = torch.tensor(positions, dtype=torch.float32).reshape(-1, 3)
    if velocities is None:
        velocities = torch.zeros_like(positions)
    else:
        velocities = torch.tensor(velocities, dtype=torch.float32).reshape(-1, 3)
    if partners is None:
        partners = _points_partners([], [], [], plane=True)
    kwargs.setdefault("radius", RADIUS)
    kwargs.setdefault("time_step", TIME_STEP)
    return detect_contacts(positions, velocities, partners, **kwargs)


class TestContactPartners(unittest.TestCase):
    def test_contact_free_has_no_partners(self):
        """Build the contact-free record with no plane, no points and zero coefficients."""
        partners = ContactPartners.contact_free()
        self.assertFalse(partners.plane_present)
        self.assertEqual(partners.point_count, 0)
        self.assertEqual(partners.point_positions.shape, (0, 3))
        self.assertEqual(partners.point_normals.shape, (0, 3))
        self.assertEqual(partners.point_radii.shape, (0,))
        self.assertEqual((partners.ke, partners.kd, partners.mu), (0.0, 0.0, 0.0))
        torch.testing.assert_close(partners.plane_normal, torch.tensor([0.0, 1.0, 0.0]))
        for tensor in (partners.plane_point, partners.plane_normal, partners.point_positions, partners.point_radii):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tensor.device.type, "cpu")

    def test_dict_round_trip_is_json_serializable_and_exact(self):
        """Serialize sampled partners to plain JSON and rebuild an identical record."""
        partners = _sample(3)
        self.assertGreater(partners.point_count, 0)
        data = json.loads(json.dumps(partners.to_dict()))
        restored = ContactPartners.from_dict(data)
        self.assertEqual(restored.plane_present, partners.plane_present)
        self.assertEqual(restored.point_count, partners.point_count)
        for name in ("plane_point", "plane_normal", "point_positions", "point_normals", "point_radii"):
            self.assertTrue(torch.equal(getattr(restored, name), getattr(partners, name)), name)
            self.assertEqual(getattr(restored, name).dtype, torch.float32)
        self.assertEqual((restored.ke, restored.kd, restored.mu), (partners.ke, partners.kd, partners.mu))
        self.assertEqual((restored.ke_floor, restored.floor_bound), (partners.ke_floor, partners.floor_bound))
        # E = 1 Pa keeps kappa E h far below the 245 N/m floor of this 12-cell body at rho = 5e3.
        floored = _sample(3, youngs_modulus=1.0, density=5e3, gravity_magnitude=9.81, static_penetration_max=0.5)
        self.assertTrue(floored.floor_bound)
        rebuilt = ContactPartners.from_dict(json.loads(json.dumps(floored.to_dict())))
        self.assertEqual((rebuilt.ke, rebuilt.ke_floor, rebuilt.floor_bound), (floored.ke, floored.ke_floor, True))
        # Payloads written before the floor lack its fields and rebuild with the floor disabled.
        legacy = {name: value for name, value in data.items() if name not in ("ke_floor", "floor_bound")}
        self.assertEqual(set(legacy), set(data) - {"ke_floor", "floor_bound"})
        without_floor = ContactPartners.from_dict(legacy)
        self.assertEqual(
            (without_floor.ke, without_floor.ke_floor, without_floor.floor_bound), (partners.ke, 0.0, False)
        )
        empty = ContactPartners.from_dict(ContactPartners.contact_free().to_dict())
        self.assertEqual(empty.point_count, 0)
        self.assertEqual(empty.point_positions.shape, (0, 3))
        with self.assertRaises(ValueError):
            ContactPartners.from_dict({"plane_present": True})

    def test_rejects_invalid_floor_record(self):
        """Raise ValueError for a negative floor, a non-bool flag or a bound flag whose ke differs from the floor."""
        base = ContactPartners.contact_free().to_dict()
        for overrides in (
            {"ke_floor": -1.0},
            {"ke_floor": math.nan},
            {"floor_bound": "yes"},
            {"floor_bound": 1},
            {"ke": 2.0, "ke_floor": 3.0, "floor_bound": True},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                ContactPartners(**{**base, **overrides})
        bound = ContactPartners(**{**base, "ke": 3.0, "ke_floor": 3.0, "floor_bound": np.bool_(True)})
        self.assertIs(bound.floor_bound, True)

    def test_rejects_invalid_fields(self):
        """Raise ValueError on inconsistent shapes, non-finite values and bad scalars."""
        good = _sample(5).to_dict()
        bad_cases = {
            "plane_point": [0.0, 0.0],
            "plane_normal": [0.0, 2.0, 0.0],
            "point_normals": good["point_normals"][:-1],
            "point_radii": [0.0] * len(good["point_radii"]),
            "point_positions": [[math.nan, 0.0, 0.0]] * len(good["point_positions"]),
            "ke": -1.0,
            "kd": math.inf,
            "mu": "0.5",
            "plane_present": 1,
        }
        for name, value in bad_cases.items():
            with self.subTest(field=name), self.assertRaises(ValueError):
                ContactPartners.from_dict({**good, name: value})


class TestSampleContactPartners(unittest.TestCase):
    def test_same_seeds_repeat_and_either_seed_changes_the_scene(self):
        """Reproduce a scene from (master_seed, seed) and change it with either seed."""
        first, repeated = _sample(7), _sample(7)
        self.assertEqual(first.to_dict(), repeated.to_dict())
        other_seed = _sample(8)
        other_master = _sample(7, master_seed=12)
        for other in (other_seed, other_master):
            self.assertNotEqual(first.ke, other.ke)
            self.assertNotEqual(first.mu, other.mu)
            self.assertNotEqual(first.to_dict(), other.to_dict())

    def test_realized_ranges_over_many_seeds(self):
        """Keep every sampled quantity inside its documented range on a small grid."""
        rest = generate_cuboid((2, 2, 3), cell_size=CELL_SIZE)
        lower = rest.corner_rest_positions.min(axis=0)
        upper = rest.corner_rest_positions.max(axis=0)
        center = 0.5 * (lower + upper)
        box_lower = lower - np.array([POINT_BOX_LATERAL_MARGIN, POINT_BOX_DEPTH_MARGIN, POINT_BOX_LATERAL_MARGIN])
        box_lower[2] = lower[2] + CELL_SIZE
        box_upper = upper + np.array([POINT_BOX_LATERAL_MARGIN, 0.0, POINT_BOX_LATERAL_MARGIN])
        clearance_lower = lower - POINT_CLEARANCE_CELLS * CELL_SIZE
        clearance_upper = upper + POINT_CLEARANCE_CELLS * CELL_SIZE
        seen_plane = seen_no_plane = seen_points = False
        for seed in range(150):
            partners = _sample(seed)
            kappa = partners.ke / (YOUNGS_MODULUS * CELL_SIZE)
            self.assertTrue(0.1 <= kappa <= 10.0, kappa)
            beta = partners.kd / (partners.ke * TIME_STEP)
            self.assertTrue(0.0 <= beta <= 1.0, beta)
            self.assertTrue(0.0 <= partners.mu <= 1.0, partners.mu)
            torch.testing.assert_close(partners.plane_normal, torch.tensor([0.0, 1.0, 0.0]))
            height = float(partners.plane_point[1]) - lower[1]
            self.assertTrue(-0.35 - 1e-6 <= height <= -0.02 + 1e-6, height)
            self.assertAlmostEqual(float(partners.plane_point[0]), center[0], places=6)
            self.assertAlmostEqual(float(partners.plane_point[2]), center[2], places=6)
            seen_plane |= partners.plane_present
            seen_no_plane |= not partners.plane_present
            self.assertTrue(0 <= partners.point_count <= 64)
            if partners.point_count:
                seen_points = True
                positions = partners.point_positions.double().numpy()
                normals = partners.point_normals.double().numpy()
                radii = partners.point_radii.double().numpy()
                self.assertTrue(np.all(positions >= box_lower - 1e-6) and np.all(positions <= box_upper + 1e-6))
                # No point inside the rest body or within one cell of it, and none in front of the clamped face.
                inside = np.all((positions > clearance_lower) & (positions < clearance_upper), axis=1)
                self.assertFalse(inside.any())
                self.assertTrue(np.all(positions[:, 2] >= lower[2] + CELL_SIZE - 1e-9))
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
                self.assertTrue(np.all(np.einsum("ij,ij->i", normals, center[None] - positions) >= 0.0))
                self.assertTrue(np.all(radii >= 0.5 * CELL_SIZE - 1e-9) and np.all(radii <= 2.0 * CELL_SIZE + 1e-9))
        self.assertTrue(seen_plane and seen_no_plane and seen_points)

    def test_coefficients_do_not_depend_on_plane_or_points(self):
        """Sample ke, kd and mu identically whether or not the plane and points are drawn."""
        with_scene = _sample(4)
        bare = _sample(4, plane_probability=0.0, max_points=0)
        self.assertFalse(bare.plane_present)
        self.assertEqual(bare.point_count, 0)
        self.assertEqual((bare.ke, bare.kd, bare.mu), (with_scene.ke, with_scene.kd, with_scene.mu))
        self.assertGreater(bare.ke, 0.0)

    def test_placement_failure_raises_after_the_attempt_budget(self):
        """Raise ValueError when no normal can satisfy the cone within the position attempt budget."""
        with (
            patch.object(contact_scene, "POINT_NORMAL_MAX_ANGLE", 0.0),
            patch.object(contact_scene, "POINT_PLACEMENT_ATTEMPTS", 20),
            self.assertRaisesRegex(ValueError, "could not place static point"),
        ):
            _sample(3)

    def test_rejects_invalid_arguments(self):
        """Raise ValueError for bad seeds, scalars, probabilities and ranges."""
        for overrides in (
            {"master_seed": -1},
            {"seed": 1.5},
            {"youngs_modulus": 0.0},
            {"cell_size": math.nan},
            {"time_step": -TIME_STEP},
            {"plane_probability": 1.5},
            {"plane_height_range": (0.0, -1.0)},
            {"max_points": -1},
            {"point_radius_range": (0.0, 1.0)},
            {"kappa_range": (0.0, 1.0)},
            {"beta_range": (-0.1, 1.0)},
            {"mu_range": (0.5,)},
            # A cell at least as large as the depth margin leaves no room for points around the body.
            {"cell_size": POINT_BOX_DEPTH_MARGIN},
            # The load-based floor needs the body's density and the gravity magnitude, all in range.
            {"static_penetration_max": 0.5},
            {"static_penetration_max": 0.5, "density": 1e3},
            {"static_penetration_max": 0.5, "gravity_magnitude": 9.81},
            {"static_penetration_max": 0.0, "density": 1e3, "gravity_magnitude": 9.81},
            {"static_penetration_max": True, "density": 1e3, "gravity_magnitude": 9.81},
            {"density": 0.0},
            {"gravity_magnitude": -9.81},
            {"sample_radius": 0.0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                _sample(**overrides)
        # Density and gravity alone leave the floor disabled.
        unfloored = _sample(density=1e3, gravity_magnitude=9.81)
        self.assertEqual((unfloored.ke_floor, unfloored.floor_bound), (0.0, False))
        self.assertEqual(unfloored.to_dict(), _sample().to_dict())
        # Without points the clearance does not matter.
        self.assertEqual(_sample(cell_size=POINT_BOX_DEPTH_MARGIN, max_points=0).point_count, 0)
        with self.assertRaises(ValueError):
            sample_contact_partners(
                None, master_seed=0, seed=0, youngs_modulus=1.0, cell_size=CELL_SIZE, time_step=TIME_STEP
            )

    def test_canonical_grid_plane_fraction_and_point_count(self):
        """Report the realized plane fraction and mean point count on the canonical grid."""
        rest = generate_cuboid((10, 10, 40), cell_size=CELL_SIZE)
        planes = 0
        counts = []
        for seed in range(200):
            partners = sample_contact_partners(
                rest,
                master_seed=2026,
                seed=seed,
                youngs_modulus=YOUNGS_MODULUS,
                cell_size=CELL_SIZE,
                time_step=TIME_STEP,
            )
            planes += int(partners.plane_present)
            counts.append(partners.point_count)
        plane_fraction = planes / 200
        mean_points = float(np.mean(counts))
        print(
            f"\ncontact scenes (200 seeds, 10x10x40): plane fraction {plane_fraction:.3f}, mean points {mean_points:.1f}"
        )
        self.assertTrue(0.7 <= plane_fraction <= 0.9, plane_fraction)
        self.assertTrue(20.0 <= mean_points <= 44.0, mean_points)

    def test_canonical_grid_points_clear_the_rest_body_and_keep_the_coefficient_stream(self):
        """Place every static point clear of the rest shape, facing its nearest face, behind the clamped end.

        Over 200 canonical (10, 10, 40) scenes with E = 1e5: no point is a
        detection candidate of any rest surface sample within the widened band
        ``gap < r + h`` (a downward sample velocity of ``(h - r) / dt`` turns
        the detector's ``2 r + |v| dt`` band into exactly ``r + h``), every
        point normal opposes the outward normal of its nearest rest sample
        within 60 degrees, no point lies below ``z = h`` and the coefficient,
        plane and count draws reproduce the documented ``SeedSequence`` order,
        so the point rejection loop shifts none of them.
        """
        rest = generate_cuboid((10, 10, 40), cell_size=CELL_SIZE)
        faces = exposed_face_samples(rest)
        positions = torch.from_numpy(rest.corner_rest_positions.astype(np.float32))[None]
        samples = sample_points(positions, faces.corners)[0]
        lower = rest.corner_rest_positions.min(axis=0)
        widening = torch.zeros_like(samples)
        widening[:, 1] = -(CELL_SIZE - RADIUS) / TIME_STEP
        max_cos = -math.cos(math.radians(POINT_NORMAL_MAX_ANGLE))
        scenes_with_candidates, counts = 0, []
        for seed in range(200):
            partners = sample_contact_partners(
                rest,
                master_seed=2026,
                seed=seed,
                youngs_modulus=YOUNGS_MODULUS,
                cell_size=CELL_SIZE,
                time_step=TIME_STEP,
            )
            counts.append(partners.point_count)
            rng = np.random.default_rng(np.random.SeedSequence([2026, seed, 2203]))
            kappa = math.exp(rng.uniform(math.log(0.1), math.log(10.0)))
            beta = float(rng.uniform(0.0, 1.0))
            mu = float(rng.uniform(0.0, 1.0))
            plane_present = bool(rng.random() < 0.8)
            plane_height = float(rng.uniform(-0.35, -0.02))
            count = int(rng.integers(0, 64, endpoint=True))
            self.assertEqual(partners.ke, kappa * YOUNGS_MODULUS * CELL_SIZE, seed)
            self.assertEqual(partners.kd, beta * partners.ke * TIME_STEP, seed)
            self.assertEqual(partners.mu, mu, seed)
            self.assertEqual(partners.plane_present, plane_present, seed)
            self.assertAlmostEqual(float(partners.plane_point[1]) - lower[1], plane_height, places=6)
            self.assertEqual(partners.point_count, count, seed)
            if not count:
                continue
            pairs = detect_contacts(samples, widening, partners, radius=RADIUS, time_step=TIME_STEP)
            point_pairs = int((pairs.kind == KIND_POINT).sum())
            scenes_with_candidates += int(point_pairs > 0)
            self.assertEqual(point_pairs, 0, seed)
            points = partners.point_positions.double()
            nearest = torch.argmin(torch.cdist(points, samples.double()), dim=1)
            cosines = (partners.point_normals.double() * faces.rest_normals.double()[nearest]).sum(dim=1)
            self.assertTrue(bool((cosines <= max_cos + 1e-6).all()), (seed, float(cosines.max())))
            self.assertTrue(bool((points[:, 2] >= lower[2] + CELL_SIZE - 1e-9).all()), seed)
        print(
            f"\ncontact scenes (200 seeds, 10x10x40): scenes with a rest candidate {scenes_with_candidates / 200:.3f}, "
            f"mean points {float(np.mean(counts)):.1f}"
        )
        self.assertTrue(20.0 <= float(np.mean(counts)) <= 44.0)


class TestContactStiffnessFloor(unittest.TestCase):
    def test_formula_on_hand_built_grids(self):
        """Return rho V g / (n_face d_max) with n_face the largest exposed-face count of one material face."""
        # (1, 2, 3) cells: six faces each on -x and +x, three on -y/+y, two on -z/+z.
        rest = generate_cuboid((1, 2, 3), cell_size=CELL_SIZE)
        volume = 6 * CELL_SIZE**3
        floor = contact_stiffness_floor(
            rest, density=2e3, gravity_magnitude=10.0, static_penetration_max=0.25, sample_radius=RADIUS
        )
        self.assertAlmostEqual(floor, 2e3 * volume * 10.0 / (6 * 0.25 * RADIUS), places=9)
        # Zero gravity means no load and no floor; the radius and the cap scale it inversely.
        self.assertEqual(
            contact_stiffness_floor(
                rest, density=2e3, gravity_magnitude=0.0, static_penetration_max=0.25, sample_radius=RADIUS
            ),
            0.0,
        )
        self.assertAlmostEqual(
            contact_stiffness_floor(
                rest, density=2e3, gravity_magnitude=10.0, static_penetration_max=0.5, sample_radius=2 * RADIUS
            ),
            floor / 4,
            places=9,
        )
        for kwargs in (
            {"density": 0.0},
            {"density": -1.0},
            {"gravity_magnitude": -1.0},
            {"gravity_magnitude": math.inf},
            {"static_penetration_max": 0.0},
            {"sample_radius": 0.0},
            {"sample_radius": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                contact_stiffness_floor(
                    rest,
                    **{
                        "density": 2e3,
                        "gravity_magnitude": 10.0,
                        "static_penetration_max": 0.25,
                        "sample_radius": RADIUS,
                        **kwargs,
                    },
                )
        with self.assertRaises(ValueError):
            contact_stiffness_floor(
                None, density=2e3, gravity_magnitude=10.0, static_penetration_max=0.25, sample_radius=RADIUS
            )

    def test_canonical_beam_floor_binds_heavy_soft_bodies_and_not_light_stiff_ones(self):
        """Raise ke to 2452 N/m for E = 1e3, rho = 1e4, g = 9.81 when kappa E h is lower; never for E = 1e5, rho = 1e3.

        The 10x10x40 beam has 4000 cells of 0.025 m (0.0625 m^3): at rho = 1e4
        it weighs 6131 N, spread over the 400 samples of its bottom face, and
        may sink at most d_max = 0.5 r = 6.25 mm, so ke >= 6131 / 2.5 = 2452
        N/m. With kappa in [10, 1000] and E = 1e3 the sampled ke = 25 kappa
        lies below it for kappa < 98, about half the scenes. At E = 1e5 and
        rho = 1e3 the floor is 245 N/m against ke >= 25000 N/m.
        """
        rest = generate_cuboid((10, 10, 40), cell_size=CELL_SIZE)
        floor = contact_stiffness_floor(
            rest, density=1e4, gravity_magnitude=9.81, static_penetration_max=0.5, sample_radius=RADIUS
        )
        self.assertAlmostEqual(floor, 625 * 9.81 / (400 * 0.5 * RADIUS), places=9)
        self.assertAlmostEqual(floor, 2452.5, places=9)
        common = {
            "master_seed": 2026,
            "cell_size": CELL_SIZE,
            "time_step": TIME_STEP,
            "kappa_range": (10.0, 1000.0),
        }
        bound = 0
        for seed in range(40):
            plain = sample_contact_partners(rest, seed=seed, youngs_modulus=1e3, **common)
            heavy = sample_contact_partners(
                rest,
                seed=seed,
                youngs_modulus=1e3,
                density=1e4,
                gravity_magnitude=9.81,
                static_penetration_max=0.5,
                **common,
            )
            # The floor changes neither the draws nor the scene, only ke and the kd built from it.
            self.assertEqual(heavy.mu, plain.mu, seed)
            self.assertEqual(heavy.plane_present, plain.plane_present, seed)
            self.assertTrue(torch.equal(heavy.plane_point, plain.plane_point), seed)
            self.assertTrue(torch.equal(heavy.point_positions, plain.point_positions), seed)
            self.assertEqual(heavy.ke_floor, floor, seed)
            beta = plain.kd / (plain.ke * TIME_STEP)
            self.assertAlmostEqual(heavy.kd, beta * heavy.ke * TIME_STEP, places=9)
            if plain.ke < floor:
                bound += 1
                self.assertTrue(heavy.floor_bound, seed)
                self.assertEqual(heavy.ke, floor, seed)
            else:
                self.assertFalse(heavy.floor_bound, seed)
                self.assertEqual(heavy.ke, plain.ke, seed)
            light = sample_contact_partners(
                rest,
                seed=seed,
                youngs_modulus=1e5,
                density=1e3,
                gravity_magnitude=9.81,
                static_penetration_max=0.5,
                **common,
            )
            self.assertAlmostEqual(light.ke_floor, 245.25, places=9)
            self.assertFalse(light.floor_bound, seed)
            self.assertGreaterEqual(light.ke, 10.0 * 1e5 * CELL_SIZE)
        print(f"\ncontact stiffness floor (40 seeds, 10x10x40, E = 1e3, rho = 1e4): bound in {bound} scenes")
        self.assertTrue(8 <= bound <= 32, bound)


class TestContactPairs(unittest.TestCase):
    def test_empty_shapes_and_dtypes(self):
        """Return zero-row tensors with the contract dtypes."""
        pairs = ContactPairs.empty()
        self.assertEqual(pairs.sample_index.shape, (0,))
        self.assertEqual(pairs.partner_index.dtype, torch.int64)
        self.assertEqual(pairs.kind.dtype, torch.int64)
        self.assertEqual(pairs.partner_point.shape, (0, 3))
        self.assertEqual(pairs.partner_normal.shape, (0, 3))
        self.assertEqual(pairs.partner_radius.shape, (0,))
        self.assertEqual(pairs.partner_point.dtype, torch.float32)


class TestDetectContacts(unittest.TestCase):
    def test_plane_threshold_is_radius_plus_margin(self):
        """Accept a resting sample just inside gap = radius + margin and reject one just outside."""
        threshold = 2.0 * RADIUS
        pairs = _detect([[0.0, threshold - 1e-6, 0.0], [0.0, threshold + 1e-6, 0.0]])
        self.assertEqual(pairs.sample_index.tolist(), [0])
        self.assertEqual(pairs.kind.tolist(), [KIND_PLANE])
        self.assertEqual(pairs.partner_index.tolist(), [-1])

    def test_margin_grows_with_velocity(self):
        """Admit a sample beyond the resting band once |v| dt widens the margin."""
        gap = 2.0 * RADIUS + 1e-3
        resting = _detect([[0.0, gap, 0.0]])
        self.assertEqual(resting.sample_index.numel(), 0)
        moving = _detect([[0.0, gap, 0.0]], velocities=[[0.0, -1.0, 0.0]])
        self.assertEqual(moving.sample_index.tolist(), [0])
        slow = _detect([[0.0, gap, 0.0]], velocities=[[0.0, -0.1, 0.0]])
        self.assertEqual(slow.sample_index.numel(), 0)

    def test_plane_pair_carries_foot_point_and_normal(self):
        """Report the foot point on the plane, the (0, 1, 0) normal, kind 0 and index -1."""
        partners = _points_partners([], [], [], plane=True, plane_y=-0.5)
        pairs = _detect([[0.3, -0.49, -0.2]], partners=partners)
        self.assertEqual(pairs.kind.tolist(), [KIND_PLANE])
        self.assertEqual(pairs.partner_index.tolist(), [-1])
        torch.testing.assert_close(pairs.partner_point, torch.tensor([[0.3, -0.5, -0.2]]))
        torch.testing.assert_close(pairs.partner_normal, torch.tensor([[0.0, 1.0, 0.0]]))
        torch.testing.assert_close(pairs.partner_radius, torch.tensor([PLANE_PARTNER_RADIUS], dtype=torch.float32))
        self.assertEqual(pairs.partner_point.dtype, torch.float32)
        self.assertEqual(pairs.sample_index.dtype, torch.int64)

    def test_point_outside_lateral_disk_is_excluded(self):
        """Exclude a point whose lateral offset exceeds r_p even when the gap is small."""
        partners = _points_partners([[0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [0.02])
        outside = _detect([[0.03, 0.005, 0.0]], partners=partners)
        self.assertEqual(outside.sample_index.numel(), 0)
        inside = _detect([[0.01, 0.005, 0.0]], partners=partners)
        self.assertEqual(inside.kind.tolist(), [KIND_POINT])
        self.assertEqual(inside.partner_index.tolist(), [0])
        torch.testing.assert_close(inside.partner_point, torch.tensor([[0.0, 0.0, 0.0]]))
        torch.testing.assert_close(inside.partner_radius, torch.tensor([0.02]))
        far_gap = _detect([[0.01, 0.03, 0.0]], partners=partners)
        self.assertEqual(far_gap.sample_index.numel(), 0)

    def test_point_disk_is_one_sided_with_sample_radius_thickness(self):
        """Keep a sample up to r behind the disk and drop one farther behind; the plane keeps everything below."""
        partners = _points_partners([[0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [2.0 * CELL_SIZE], plane=True, plane_y=0.0)
        behind = _detect([[0.0, -RADIUS + 1e-6, 0.0], [0.0, -RADIUS - 1e-6, 0.0], [0.0, -0.3, 0.0]], partners=partners)
        rows = list(zip(behind.sample_index.tolist(), behind.kind.tolist(), strict=True))
        self.assertIn((0, KIND_POINT), rows)
        self.assertNotIn((1, KIND_POINT), rows)
        self.assertNotIn((2, KIND_POINT), rows)
        self.assertEqual(
            [row for row in rows if row[1] == KIND_PLANE], [(0, KIND_PLANE), (1, KIND_PLANE), (2, KIND_PLANE)]
        )
        # The bound is the sample radius, not the lateral radius r_p.
        wide = _detect(
            [[0.0, -1.5 * RADIUS, 0.0]], partners=_points_partners([[0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [1.0])
        )
        self.assertEqual(wide.sample_index.numel(), 0)

    def test_keeps_only_the_nearest_points_by_gap(self):
        """Keep the four smallest gaps among six candidates and order them by gap."""
        depths = [0.004, 0.001, 0.006, 0.002, 0.005, 0.003]
        partners = _points_partners([[0.0, -depth, 0.0] for depth in depths], [[0.0, 1.0, 0.0]] * 6, [RADIUS] * 6)
        pairs = _detect([[0.0, 0.0, 0.0]], partners=partners)
        self.assertEqual(pairs.partner_index.tolist(), [1, 3, 5, 0])
        self.assertEqual(pairs.kind.tolist(), [KIND_POINT] * 4)
        self.assertEqual(pairs.sample_index.tolist(), [0] * 4)
        two = _detect([[0.0, 0.0, 0.0]], partners=partners, max_pairs_per_sample=2)
        self.assertEqual(two.partner_index.tolist(), [1, 3])
        none = _detect([[0.0, 0.0, 0.0]], partners=partners, max_pairs_per_sample=0)
        self.assertEqual(none.sample_index.numel(), 0)

    def test_empty_when_nothing_is_near(self):
        """Return the empty pair list for far samples, absent partners and zero samples."""
        partners = _points_partners([[0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [RADIUS], plane=True)
        far = _detect([[1.0, 1.0, 1.0]], partners=partners)
        self.assertEqual(far.sample_index.numel(), 0)
        free = _detect([[0.0, 0.0, 0.0]], partners=ContactPartners.contact_free())
        self.assertEqual(free.sample_index.numel(), 0)
        empty = detect_contacts(torch.zeros(0, 3), torch.zeros(0, 3), partners, radius=RADIUS, time_step=TIME_STEP)
        self.assertEqual(empty.partner_point.shape, (0, 3))

    def test_ordering_is_by_sample_kind_gap_and_repeatable(self):
        """Order rows by sample, then kind, then gap and reproduce them across calls."""
        partners = _points_partners(
            [[0.0, -0.002, 0.0], [0.0, -0.001, 0.0], [0.1, -0.003, 0.0]],
            [[0.0, 1.0, 0.0]] * 3,
            [RADIUS] * 3,
            plane=True,
            plane_y=-0.01,
        )
        positions = [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
        first = _detect(positions, partners=partners)
        second = _detect(positions, partners=partners)
        for field_first, field_second in zip(first, second, strict=True):
            self.assertTrue(torch.equal(field_first, field_second))
        self.assertEqual(first.sample_index.tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(first.kind.tolist(), [KIND_PLANE, KIND_POINT, KIND_PLANE, KIND_POINT, KIND_POINT])
        self.assertEqual(first.partner_index.tolist(), [-1, 2, -1, 1, 0])
        torch.testing.assert_close(first.partner_point[0], torch.tensor([0.1, -0.01, 0.0]))
        torch.testing.assert_close(first.partner_point[2], torch.tensor([0.0, -0.01, 0.0]))

    def test_pair_count_is_bounded(self):
        """Return at most S * (1 + max_pairs_per_sample) rows in a dense scene."""
        sample_count, point_count = 5, 12
        positions = [[0.0, 0.0, 0.01 * index] for index in range(sample_count)]
        points = [[0.0, -0.001 * (k + 1), 0.0] for k in range(point_count)]
        partners = _points_partners(points, [[0.0, 1.0, 0.0]] * point_count, [1.0] * point_count, plane=True)
        pairs = _detect(positions, partners=partners)
        self.assertEqual(pairs.sample_index.numel(), sample_count * (1 + 4))
        counts = torch.bincount(pairs.sample_index, minlength=sample_count)
        self.assertEqual(counts.tolist(), [5] * sample_count)
        self.assertEqual(int((pairs.kind == KIND_PLANE).sum()), sample_count)

    def test_sample_normals_drop_partners_that_do_not_oppose_the_face(self):
        """Keep plane and point pairs on faces that oppose the partner normal and drop side and far faces."""
        partners = _points_partners([[0.0, 0.0, 0.1]], [[0.0, 1.0, 0.0]], [0.02], plane=True, plane_y=0.0)
        positions = [[0.0, 0.01, 0.0], [0.0, 0.01, 0.0], [0.0, 0.01, 0.0], [0.005, 0.01, 0.1], [0.005, 0.01, 0.1]]
        normals = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        unfiltered = _detect(positions, partners=partners)
        self.assertEqual(unfiltered.sample_index.tolist(), [0, 1, 2, 3, 3, 4, 4])
        filtered = _detect(positions, partners=partners, sample_normals=torch.tensor(normals))
        rows = list(zip(filtered.sample_index.tolist(), filtered.kind.tolist(), strict=True))
        # Bottom faces (0, 3) keep the plane, side (1, 4) and top (2) faces lose it; only the bottom face at the
        # point keeps the point pair, the side face there loses it.
        self.assertEqual(rows, [(0, KIND_PLANE), (3, KIND_PLANE), (3, KIND_POINT)])
        for field_filtered, field_unfiltered in zip(filtered, unfiltered, strict=True):
            self.assertEqual(field_filtered.dtype, field_unfiltered.dtype)
        # A grazing normal (exactly perpendicular) is dropped; a slight opposition is kept.
        grazing = _detect(positions[:1], partners=partners, sample_normals=torch.tensor([[1.0, 0.0, 0.0]]))
        self.assertEqual(grazing.sample_index.numel(), 0)
        tilted = _detect(positions[:1], partners=partners, sample_normals=torch.tensor([[0.99, -0.01, 0.0]]))
        self.assertEqual(tilted.kind.tolist(), [KIND_PLANE])
        # Filtered candidates do not occupy the per-sample point slots.
        depths = [0.004, 0.001, 0.006, 0.002, 0.005, 0.003]
        stack = _points_partners(
            [[0.0, -depth, 0.0] for depth in depths],
            [[0.0, 1.0, 0.0]] * 3 + [[0.0, -1.0, 0.0]] * 3,
            [RADIUS] * 6,
        )
        kept = _detect([[0.0, 0.0, 0.0]], partners=stack, sample_normals=torch.tensor([[0.0, -1.0, 0.0]]))
        self.assertEqual(kept.partner_index.tolist(), [1, 0, 2])
        for bad in (torch.zeros(4, 3), torch.zeros(5, 2), torch.full((5, 3), math.nan), [[0.0, 1.0, 0.0]] * 5):
            with self.subTest(bad=type(bad)), self.assertRaises(ValueError):
                _detect(positions, partners=partners, sample_normals=bad)

    def test_rejects_invalid_inputs(self):
        """Raise ValueError for bad shapes, non-finite positions and invalid scalars."""
        partners = ContactPartners.contact_free()
        good = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            detect_contacts(torch.zeros(2, 2), good, partners, radius=RADIUS, time_step=TIME_STEP)
        with self.assertRaises(ValueError):
            detect_contacts(good, torch.zeros(3, 3), partners, radius=RADIUS, time_step=TIME_STEP)
        with self.assertRaises(ValueError):
            detect_contacts(torch.full((2, 3), math.nan), good, partners, radius=RADIUS, time_step=TIME_STEP)
        with self.assertRaises(ValueError):
            detect_contacts(good, good, partners, radius=0.0, time_step=TIME_STEP)
        with self.assertRaises(ValueError):
            detect_contacts(good, good, partners, radius=RADIUS, time_step=math.inf)
        with self.assertRaises(ValueError):
            detect_contacts(good, good, partners, radius=RADIUS, time_step=TIME_STEP, max_pairs_per_sample=-1)
        with self.assertRaises(ValueError):
            detect_contacts(good, good, None, radius=RADIUS, time_step=TIME_STEP)


if __name__ == "__main__":
    unittest.main()
