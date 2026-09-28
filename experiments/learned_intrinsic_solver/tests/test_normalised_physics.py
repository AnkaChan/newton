# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that the cell-unit evaluation inside ``MixedHexSolverStep`` reproduces the SI objective and update.

:class:`mixed_physics.MixedHexSolverStep` keeps its SI API but evaluates the
objective, the fusion and the network inputs in the cell units of the
normalisation law (``notes/ideas/idea-normalize-cells.md``,
``tests/test_scaling_invariance.py``). Four properties pin that down:

1. ``energy`` equals the SI :class:`hex_energy.HexImplicitEulerLoss` plus the
   SI :func:`contact_energy.contact_energy`, term by term, for several
   materials with active contact, damping and gravity.
2. ``forward`` returns the fused positions of a direct SI-space fusion of the
   same network increment (unit fusion weights, SI grid), its force residual
   is the SI gradient norm of the free corners and its energy is the SI
   objective at the fused positions.
3. Two physical scenes related by the law (``h = 0.025`` versus ``h = 0.05``
   with ``rho / 4``, ``2 g``, ``2 ke``, ``2 kd``, ``2 friction_epsilon`` at
   the same ``dt``) produce identical network inputs (frames, axes, state,
   edge features, contact tokens, conditioning) and fused positions that
   differ exactly by the factor 2.
4. Registering contexts builds no fusion factor: one unit-weight factor built
   by the constructor is shared by every context.
5. The deployment path, :class:`solver_step.LearnedHexSolverStep`, is scale-free
   as well: it evaluates in SI but expresses its gradient feature in ``mu h^3``,
   so the two scenes of item 3 produce identical network inputs (the log RMS
   included) and positions that differ by the factor 2 through that step too.

Everything runs in float32 on the CPU with float64 references where the SI
quantity is recomputed independently.
"""

import importlib.util
import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features, mixed_physics
from experiments.learned_intrinsic_solver.contact_energy import contact_energy
from experiments.learned_intrinsic_solver.contact_geometry import sample_points
from experiments.learned_intrinsic_solver.contact_scene import ContactPartners
from experiments.learned_intrinsic_solver.fusion import HexFusion
from experiments.learned_intrinsic_solver.hex_energy import HexImplicitEulerLoss
from experiments.learned_intrinsic_solver.input_assembly import OptimizerHistory
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep
from experiments.learned_intrinsic_solver.train_mixed import _batch

CELL_COUNTS = (2, 2, 3)
CELL_SIZE = 0.025
TIME_STEP = 1.0 / 300.0
GRAVITY = (0.0, -9.81, 0.0)
FRICTION_EPSILON = 1e-2
CONTACT_KAPPA, CONTACT_BETA, CONTACT_FRICTION = 1.0, 0.5, 0.3
RTOL = 1e-5
"""Relative tolerance of the float32 step against the float64 SI reference (observed: below 1e-6)."""


def _lame(youngs: float, poisson: float) -> tuple[float, float]:
    """Return the Lamé pair of a Young's modulus and Poisson ratio."""
    return youngs * poisson / ((1 + poisson) * (1 - 2 * poisson)), youngs / (2 * (1 + poisson))


def _material(youngs: float, poisson: float, density: float, damping: float) -> dict[str, float]:
    lam, mu = _lame(youngs, poisson)
    return {"lame_lambda": lam, "lame_mu": mu, "density": density, "damping": damping}


def _floor(height: float, youngs: float, cell_size: float, time_step: float) -> ContactPartners:
    """Return a ground plane at ``height`` with ke = kappa E h and kd = beta ke dt."""
    ke = CONTACT_KAPPA * youngs * cell_size
    return ContactPartners(
        plane_present=True,
        plane_point=torch.tensor([0.0, height, 0.0]),
        plane_normal=torch.tensor([0.0, 1.0, 0.0]),
        point_positions=torch.zeros((0, 3)),
        point_normals=torch.zeros((0, 3)),
        point_radii=torch.zeros((0,)),
        ke=ke,
        kd=CONTACT_BETA * ke * time_step,
        mu=CONTACT_FRICTION,
    )


def _network(seed: int, *, contact_tokens: bool, target_modes: int = 3) -> IntrinsicSolverNetwork:
    """Return a small revised-schema network with nonzero heads so the update depends on every input."""
    torch.manual_seed(seed)
    network = IntrinsicSolverNetwork(
        CELL_COUNTS,
        features.state_feature_dim(target_modes),
        target_modes=target_modes,
        conditioning_dim=features.CONDITIONING_DIM,
        hidden_dim=16,
        edge_hidden_dim=8,
        contact_tokens=contact_tokens,
    )
    with torch.no_grad():
        network.correction_head.weight.normal_(std=0.003)
        network.step_head.weight.normal_(std=0.01)
        for layer in network.layers:
            layer.film.weight.normal_(std=0.01)
        if contact_tokens:
            network.contact_encoder.pool_projection.weight.normal_(std=0.1)
    return network


def _deviation(actual, expected) -> float:
    """Return ``max|actual - expected| / max|expected|`` in float64."""
    actual = torch.as_tensor(actual).detach().double()
    expected = torch.as_tensor(expected).detach().double()
    return ((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-300)).item()


class TestNormalisedPhysics(unittest.TestCase):
    """SI equivalence and scale invariance of the normalised mixed step on a (2, 2, 3) clamped grid."""

    def setUp(self):
        self.rest = generate_cuboid(CELL_COUNTS, cell_size=CELL_SIZE)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.generator = torch.Generator().manual_seed(7)

    def _step(
        self, rest, network, *, time_step=TIME_STEP, gravity=GRAVITY, friction_epsilon=FRICTION_EPSILON, target_modes=3
    ):
        step = MixedHexSolverStep(
            rest,
            self.fixed,
            network=network,
            time_step=time_step,
            gravity=gravity,
            contact_friction_epsilon=friction_epsilon,
            target_modes=target_modes,
        )
        self.addCleanup(step.close)
        return step

    def _state(self, rest, *, drift: float):
        """Return SI start positions (rest plus 1 % h smooth-ish noise) and a velocity drifting down at ``drift`` h / dt."""
        h, dt = rest.cell_size, TIME_STEP
        positions = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)
        noise = 0.01 * h * torch.randn(positions.shape, generator=self.generator)
        velocity = 0.05 * h / dt * torch.randn(positions.shape, generator=self.generator)
        velocity[:, 1] -= drift * h / dt
        noise[self.fixed] = 0
        velocity[self.fixed] = 0
        return positions + noise, velocity

    def _reference_terms(self, step, material, floor, candidate, batch):
        """Return the SI float64 objective at ``candidate``: HexImplicitEulerLoss terms and the contact energy."""
        loss = HexImplicitEulerLoss(
            self.rest,
            material["lame_lambda"],
            material["lame_mu"],
            material["density"],
            step.time_step,
            damping=material["damping"],
            dtype=torch.float64,
        )
        terms = loss(
            candidate.double(),
            batch["inertial_prediction"].double(),
            previous_positions=batch["physical_positions"].double(),
        )
        contact = batch["contact"]
        corners = step.face_samples.corners
        contact_term = contact_energy(
            sample_points(candidate.double(), corners),
            sample_points(batch["physical_positions"].double(), corners),
            contact["sample_index"],
            contact["partner_point"].double(),
            contact["partner_normal"].double(),
            contact["mask"],
            radius=step.contact_radius,
            ke=floor.ke,
            kd=floor.kd,
            mu=floor.mu,
            time_step=step.time_step,
            friction_epsilon=step.contact_friction_epsilon,
        )
        return terms, contact_term

    def test_energy_matches_the_si_objective_term_by_term(self):
        """step.energy (float32, cell units inside) equals the float64 SI loss plus SI contact energy to 1e-5."""
        step = self._step(self.rest, _network(1, contact_tokens=False))
        worst = dict.fromkeys(("elastic", "inertia", "damping", "contact", "total"), 0.0)
        cases = list(itertools.product((1e3, 1e6), (0.3, 0.45), ((100.0, 0.0), (5000.0, 300.0))))
        for index, (youngs, poisson, (density, damping)) in enumerate(cases):
            name = f"material{index}"
            material = _material(youngs, poisson, density, damping)
            floor = _floor(-0.3 * CELL_SIZE, youngs, CELL_SIZE, TIME_STEP)
            step.register_context(name, **material, contact=floor)
            positions, velocity = self._state(self.rest, drift=0.03)
            payload = step.prepare(name, positions, velocity)
            self.assertGreater(payload["contact_sample_index"].numel(), 0, "the floor is in contact")
            batch = _batch([payload], torch.device("cpu"))
            candidate = batch["candidate"] + 0.02 * CELL_SIZE * torch.randn(
                batch["candidate"].shape, generator=self.generator
            )
            terms = step.energy(
                candidate,
                batch["inertial_prediction"],
                batch["context_ids"],
                previous_positions=batch["physical_positions"],
                contact=batch["contact"],
            )
            reference, contact_term = self._reference_terms(step, material, floor, candidate, batch)
            expected = {
                "elastic": reference.elastic,
                "inertia": reference.inertia,
                "damping": reference.damping,
                "contact": contact_term,
                "total": reference.total + contact_term,
            }
            for term, value in expected.items():
                actual = getattr(terms, term)
                if term == "damping" and damping == 0.0:
                    self.assertEqual(actual.item(), 0.0)
                    continue
                self.assertGreater(value.item(), 0.0, f"{name} {term} is exercised")
                worst[term] = max(worst[term], _deviation(actual, value))
        print(f"energy versus the SI float64 objective, max relative deviation per term: {worst}")
        for term, value in worst.items():
            self.assertLessEqual(value, RTOL, term)

    def test_forward_matches_a_direct_si_fusion_and_gradient(self):
        """The fused positions equal an SI-grid fusion of the network increment; the residual is the SI gradient norm.

        Checked for the three-mode and the seven-mode step, each against an
        SI-grid fusion with the same mode count.
        """
        for target_modes in (3, 7):
            with self.subTest(target_modes=target_modes):
                self._check_forward_matches_a_direct_si_fusion(target_modes)

    def _check_forward_matches_a_direct_si_fusion(self, target_modes: int):
        network = _network(2, contact_tokens=True, target_modes=target_modes)
        step = self._step(self.rest, network, target_modes=target_modes)
        material = _material(1e4, 0.3, 1000.0, 10.0)
        floor = _floor(-0.3 * CELL_SIZE, 1e4, CELL_SIZE, TIME_STEP)
        step.register_context("beam", **material, contact=floor)
        positions, velocity = self._state(self.rest, drift=0.03)
        batch = _batch([step.prepare("beam", positions, velocity)], torch.device("cpu"))
        candidate = batch["candidate"] + 0.02 * CELL_SIZE * torch.randn(
            batch["candidate"].shape, generator=self.generator
        )
        query = {"previous_positions": batch["physical_positions"], "contact": batch["contact"]}
        inputs = step.prepare_inputs(candidate, batch["inertial_prediction"], batch["context_ids"], **query)
        output = step(
            candidate,
            batch["inertial_prediction"],
            batch["context_ids"],
            fixed_positions=batch["fixed_positions"],
            **query,
        )
        # The same network increment fused on the SI grid with uniform weights (the weights' scale is immaterial).
        prediction = network(
            inputs.local_axes,
            inputs.state_features,
            inputs.edge_features,
            inputs.conditioning,
            contact_tokens=inputs.contact_tokens,
            contact_mask=inputs.contact_mask,
        )
        increment = inputs.frames @ (prediction.local_target_axes - inputs.local_axes)
        self.assertEqual(increment.shape[-1], target_modes)
        physical_fusion = HexFusion(
            self.rest,
            self.fixed,
            cell_weights=torch.full((len(self.rest.cell_corner_indices),), 3.0),
            target_modes=target_modes,
        )
        expected = physical_fusion.fuse(candidate, increment.detach(), batch["fixed_positions"])
        deviation = (output.positions.detach() - expected).abs().max().item() / CELL_SIZE
        self.assertGreater((expected - candidate).abs().max().item(), 1e-4 * CELL_SIZE, "the update is not trivial")
        self.assertTrue(torch.equal(output.positions[:, self.fixed], batch["fixed_positions"]), "pins are exact")
        # The residual [N] is the SI gradient norm of the free corners at the pre-update candidate.
        variable = candidate.clone().requires_grad_(True)
        total = step.energy(variable, batch["inertial_prediction"], batch["context_ids"], **query).total.sum()
        gradient = torch.autograd.grad(total, variable)[0]
        gradient[:, self.fixed] = 0
        residual = _deviation(output.force_residual_norm, gradient.flatten(1).norm(dim=1))
        # The energy of the output is the SI objective at the fused positions.
        reference, contact_term = self._reference_terms(step, material, floor, output.positions.detach(), batch)
        energy = _deviation(output.loss.total, reference.total + contact_term)
        self.assertGreater(output.loss.contact.item(), 0.0, "contact is active at the fused positions")
        print(
            f"forward ({target_modes} modes): positions / h {deviation:.2e}, residual {residual:.2e}, energy {energy:.2e}"
        )
        self.assertLessEqual(deviation, RTOL)
        self.assertLessEqual(residual, RTOL)
        self.assertLessEqual(energy, RTOL)

    def test_two_scales_give_identical_inputs_and_positions_scaled_by_two(self):
        """h = 0.025 and h = 0.05 with rho / 4, 2 g, 2 ke, 2 kd, 2 friction_epsilon: same inputs, positions x 2."""
        scale = 2.0
        network = _network(3, contact_tokens=True)
        material = _material(1e4, 0.3, 1000.0, 10.0)
        fine = self._step(self.rest, network)
        coarse_rest = generate_cuboid(CELL_COUNTS, cell_size=scale * CELL_SIZE)
        coarse = self._step(
            coarse_rest,
            network,
            gravity=tuple(scale * component for component in GRAVITY),
            friction_epsilon=scale * FRICTION_EPSILON,
        )
        fine.register_context("beam", **material, contact=_floor(-0.3 * CELL_SIZE, 1e4, CELL_SIZE, TIME_STEP))
        coarse.register_context(
            "beam",
            **{**material, "density": material["density"] / scale**2},
            contact=_floor(-0.3 * scale * CELL_SIZE, 1e4, scale * CELL_SIZE, TIME_STEP),
        )
        positions, velocity = self._state(self.rest, drift=0.03)
        fine_batch = _batch([fine.prepare("beam", positions, velocity)], torch.device("cpu"))
        coarse_batch = _batch([coarse.prepare("beam", scale * positions, scale * velocity)], torch.device("cpu"))
        self.assertGreater(fine_batch["contact"]["sample_index"].shape[1], 0, "contact is active")
        self.assertEqual(
            fine_batch["contact"]["sample_index"].tolist(), coarse_batch["contact"]["sample_index"].tolist()
        )
        deviation = {}
        for name in ("candidate", "inertial_prediction", "fixed_positions"):
            deviation[name] = (coarse_batch[name] - scale * fine_batch[name]).abs().max().item() / (scale * CELL_SIZE)
        noise = 0.02 * CELL_SIZE * torch.randn(fine_batch["candidate"].shape, generator=self.generator)
        fine_candidate = fine_batch["candidate"] + noise
        coarse_candidate = coarse_batch["candidate"] + scale * noise

        def run(step, batch, candidate):
            query = {"previous_positions": batch["physical_positions"], "contact": batch["contact"]}
            inputs = step.prepare_inputs(candidate, batch["inertial_prediction"], batch["context_ids"], **query)
            output = step(
                candidate,
                batch["inertial_prediction"],
                batch["context_ids"],
                fixed_positions=batch["fixed_positions"],
                **query,
            )
            return inputs, output

        fine_inputs, fine_output = run(fine, fine_batch, fine_candidate)
        coarse_inputs, coarse_output = run(coarse, coarse_batch, coarse_candidate)
        for name in ("frames", "local_axes", "state_features", "conditioning", "contact_tokens", "axis_gradient_world"):
            deviation[name] = (getattr(fine_inputs, name) - getattr(coarse_inputs, name)).abs().max().item()
        # The position gradient is reported in newtons and scales with mu h^2 (the same mu here).
        deviation["position_gradient_over_h2"] = _deviation(
            coarse_inputs.position_gradient, scale**2 * fine_inputs.position_gradient
        )
        self.assertTrue(torch.equal(fine_inputs.contact_mask, coarse_inputs.contact_mask))
        self.assertGreater(fine_inputs.contact_mask.sum().item(), 0)
        for hop, edges in fine_inputs.edge_features.items():
            deviation[f"edge_features_hop{hop}"] = (edges - coarse_inputs.edge_features[hop]).abs().max().item()
        deviation["positions_over_h"] = (coarse_output.positions - scale * fine_output.positions).abs().max().item() / (
            scale * CELL_SIZE
        )
        deviation["energy"] = _deviation(coarse_output.loss.total, scale**3 * fine_output.loss.total)
        deviation["force_residual"] = _deviation(
            coarse_output.force_residual_norm, scale**2 * fine_output.force_residual_norm
        )
        deviation["penetration"] = _deviation(
            coarse_output.contact_max_penetration, fine_output.contact_max_penetration
        )
        exact = torch.equal(fine_inputs.state_features, coarse_inputs.state_features) and torch.equal(
            coarse_output.positions, scale * fine_output.positions
        )
        print(f"two scales (h and 2 h): max deviations {deviation}; bit-identical inputs and positions: {exact}")
        self.assertGreater((fine_output.positions - fine_candidate).abs().max().item(), 1e-4 * CELL_SIZE)
        for name, value in deviation.items():
            self.assertLessEqual(value, 1e-6, name)

    def test_single_material_step_is_scale_free_too(self):
        """LearnedHexSolverStep at h and 2 h (rho / 4, 2 g, 2 eta, same dt): identical inputs, positions x 2.

        The deployment path evaluates in SI; its log RMS would be shifted by
        ``log 8`` between the two scenes if the gradient feature were left in
        joules, and a mixed-trained network would receive different inputs.
        """
        scale = 2.0
        network = _network(5, contact_tokens=False)
        material = _material(1e4, 0.3, 1000.0, 10.0)
        coarse_rest = generate_cuboid(CELL_COUNTS, cell_size=scale * CELL_SIZE)
        fine = LearnedHexSolverStep(
            self.rest, self.fixed, **material, time_step=TIME_STEP, gravity=GRAVITY, network=network
        )
        coarse = LearnedHexSolverStep(
            coarse_rest,
            self.fixed,
            **{**material, "density": material["density"] / scale**2},
            time_step=TIME_STEP,
            gravity=tuple(scale * component for component in GRAVITY),
            network=network,
        )
        self.assertEqual(coarse.energy_unit, scale**3 * fine.energy_unit)
        previous, velocity = self._state(self.rest, drift=0.03)
        previous, velocity = previous[None], velocity[None]
        prediction = previous + TIME_STEP * velocity + TIME_STEP**2 * torch.tensor(GRAVITY)
        candidate = previous + 0.02 * CELL_SIZE * torch.randn(previous.shape, generator=self.generator)
        candidate[:, self.fixed] = previous[:, self.fixed]
        pins = previous[:, self.fixed]
        history = None
        deviation = {}
        for _ in range(2):
            fine_inputs = fine.prepare_inputs(candidate, prediction, previous_positions=previous, history=history)
            coarse_inputs = coarse.prepare_inputs(
                scale * candidate, scale * prediction, previous_positions=scale * previous, history=history
            )
            fine_output = fine(
                candidate, prediction, previous_positions=previous, fixed_positions=pins, history=history
            )
            coarse_output = coarse(
                scale * candidate,
                scale * prediction,
                previous_positions=scale * previous,
                fixed_positions=scale * pins,
                history=history,
            )
            for name in ("frames", "local_axes", "state_features", "conditioning", "axis_gradient_world"):
                deviation[name] = max(
                    deviation.get(name, 0.0),
                    (getattr(fine_inputs, name) - getattr(coarse_inputs, name)).abs().max().item(),
                )
            for hop, edges in fine_inputs.edge_features.items():
                deviation[f"edge_features_hop{hop}"] = (edges - coarse_inputs.edge_features[hop]).abs().max().item()
            deviation["position_gradient_over_h2"] = _deviation(
                coarse_inputs.position_gradient, scale**2 * fine_inputs.position_gradient
            )
            deviation["positions_over_h"] = (
                coarse_output.positions - scale * fine_output.positions
            ).abs().max().item() / (scale * CELL_SIZE)
            deviation["energy"] = _deviation(coarse_output.loss.total, scale**3 * fine_output.loss.total)
            deviation["force_residual"] = _deviation(
                coarse_output.force_residual_norm, scale**2 * fine_output.force_residual_norm
            )
            self.assertGreater((fine_output.positions - candidate).abs().max().item(), 1e-4 * CELL_SIZE)
            # The second round carries this query's history (shared verbatim: both steps use mu h^3 units).
            history = OptimizerHistory(
                fine_output.axis_gradient_world, fine_output.achieved_axis_update_world, torch.tensor([True])
            )
            candidate = fine_output.positions.detach()
        self.assertEqual(fine_inputs.state_features[..., -1].unique().tolist(), [1.0], "history was consumed")
        column = features.MATRIX_FEATURE_DIM + features.BOUNDARY_DIM
        self.assertTrue(torch.isfinite(fine_inputs.state_features[..., column]).all())
        print(f"single-material step at h and 2 h (SI shift would be log 8 = {math.log(8):.3f}): {deviation}")
        for name, value in deviation.items():
            self.assertLessEqual(value, 1e-6, name)

    def test_register_context_shares_one_fusion_factor(self):
        """Ten registrations build no factor: the constructor's unit-weight factor is shared by every context."""
        network = _network(4, contact_tokens=False)
        with patch.object(mixed_physics, "HexFusion", wraps=HexFusion) as counted:
            step = self._step(self.rest, network)
            self.assertEqual(counted.call_count, 1)
            factor = step._fusion
            for index in range(10):
                step.register_context(f"material{index}", **_material(1e3 * (index + 1), 0.3, 500.0 + index, 0.0))
            self.assertEqual(counted.call_count, 1)
        self.assertIs(step._fusion, factor)
        self.assertIsInstance(factor, HexFusion)
        # Two materials a decade apart evaluate through the same factor and still produce distinct, finite updates.
        positions = torch.tensor(self.rest.corner_rest_positions, dtype=torch.float32)[None].repeat(2, 1, 1)
        positions[:, :, 0] += 0.05 * positions[:, :, 2].square()
        ids = ("material0", "material9")
        target = positions + torch.tensor([0.0, -0.001, 0.0])
        output = step(positions, target, ids, previous_positions=positions, fixed_positions=positions[:, self.fixed])
        self.assertTrue(torch.isfinite(output.positions).all())
        self.assertFalse(torch.allclose(output.loss.total[0], output.loss.total[1]))


if __name__ == "__main__":
    unittest.main()
