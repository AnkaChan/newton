# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the penalty contact energy against Newton's VBD contact force law."""

import importlib.util
import math
import unittest

import numpy as np
import warp as wp

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.contact_energy import contact_energy, contact_penetration
from newton._src.solvers.vbd.rigid_vbd_kernels import _compute_body_particle_contact_force

RADIUS = 0.0125
TIME_STEP = 1.0 / 300.0
FRICTION_EPSILON = 1e-2
EPS_U = FRICTION_EPSILON * TIME_STEP
KE = 2000.0
KD = 0.4 * KE * TIME_STEP
MU = 0.6


@wp.kernel
def _newton_contact_force_oracle(
    distance: wp.array[float],
    normals: wp.array[wp.vec3],
    translations: wp.array[wp.vec3],
    radius: float,
    ke: float,
    kd: float,
    mu: float,
    friction_epsilon: float,
    dt: float,
    forces: wp.array[wp.vec3],
):
    pair = wp.tid()
    force, _hessian = _compute_body_particle_contact_force(
        distance[pair], radius, normals[pair], translations[pair], ke, kd, mu, friction_epsilon, dt, False
    )
    forces[pair] = force


def _smoothing(y: float, eps_u: float = EPS_U) -> float:
    """Reference IPC ``f0`` used by the finite-difference correction."""
    if y < eps_u:
        return -(y**3) / (3.0 * eps_u**2) + y**2 / eps_u + eps_u / 3.0
    return y


def _frame(normal):
    """Return a unit normal and two unit tangents in float64."""
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent_a = np.cross(normal, [1.0, 0.0, 0.0])
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(normal, tangent_a)
    return normal, tangent_a, tangent_b


def _energy(positions, start, index, points, normals, mask, *, ke=KE, kd=KD, mu=MU, dtype=None):
    """Evaluate ``contact_energy`` with the module constants and per-batch scalar parameters."""
    dtype = dtype or positions.dtype
    batch = positions.shape[0]
    return contact_energy(
        positions,
        start,
        index,
        points,
        normals,
        mask,
        radius=RADIUS,
        ke=torch.full((batch,), ke, dtype=dtype),
        kd=torch.full((batch,), kd, dtype=dtype),
        mu=torch.full((batch,), mu, dtype=dtype),
        time_step=TIME_STEP,
        friction_epsilon=FRICTION_EPSILON,
    )


class TestContactEnergyOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_negative_gradient_matches_newton_force(self):
        """Match Newton's body-particle force for approaching, separating, sliding and sticking pairs."""
        normal, tangent_a, tangent_b = _frame([0.2, 0.9, -0.3])
        point = np.array([0.05, -0.3, 0.4])
        # (depth, normal translation, tangential translation) per case.
        cases = [
            (0.003, -0.0008, 1.0e-5 * tangent_a),  # penetrating, approaching, sticking
            (0.002, 0.0005, 5.0e-4 * tangent_a),  # penetrating, separating, sliding
            (0.004, -0.0003, 2.0e-3 * tangent_a + 1.0e-3 * tangent_b),  # sliding while approaching
            (0.001, 0.0002, 1.5e-5 * tangent_a - 1.0e-5 * tangent_b),  # sticking while separating
        ]
        positions = np.stack([point + (RADIUS - depth) * normal + 0.02 * tangent_b for depth, _, _ in cases])
        translations = np.stack([dn * normal + slip for _, dn, slip in cases])
        x = torch.tensor(positions, dtype=torch.float32)[None].requires_grad_()
        x0 = torch.tensor(positions - translations, dtype=torch.float32)[None]
        count = len(cases)
        normals = torch.tensor(normal, dtype=torch.float32).expand(count, 3).contiguous()
        points = torch.tensor(point, dtype=torch.float32).expand(count, 3).contiguous()
        index = torch.arange(count)[None]
        energy = _energy(x, x0, index, points[None], normals[None], torch.ones(1, count, dtype=torch.bool))
        force = -torch.autograd.grad(energy.sum(), x)[0][0].numpy()

        distance = ((x.detach()[0] - points) * normals).sum(-1).numpy()
        translation32 = (x.detach()[0] - x0[0]).numpy()
        forces = wp.zeros(count, dtype=wp.vec3, device="cpu")
        wp.launch(
            _newton_contact_force_oracle,
            dim=count,
            inputs=[
                wp.array(distance, dtype=float, device="cpu"),
                wp.array(normals.numpy(), dtype=wp.vec3, device="cpu"),
                wp.array(translation32, dtype=wp.vec3, device="cpu"),
                RADIUS,
                KE,
                KD,
                MU,
                FRICTION_EPSILON,
                TIME_STEP,
            ],
            outputs=[forces],
            device="cpu",
        )
        expected = forces.numpy()
        self.assertTrue(np.all(np.linalg.norm(expected, axis=1) > 1.0))
        np.testing.assert_allclose(force, expected, rtol=1e-4, atol=1e-6)
        # Damping only alters the two approaching cases.
        undamped = _energy(x, x0, index, points[None], normals[None], torch.ones(1, count, dtype=torch.bool), kd=0.0)
        undamped_force = -torch.autograd.grad(undamped.sum(), x)[0][0].numpy()
        np.testing.assert_allclose(undamped_force[[1, 3]], expected[[1, 3]], rtol=1e-4, atol=1e-6)
        self.assertGreater(np.abs(undamped_force[[0, 2]] - expected[[0, 2]]).max(), 0.1)


class TestContactEnergy(unittest.TestCase):
    def _scene(self, dtype=torch.float64):
        """Three samples, two partners, one masked pair; all valid pairs penetrate."""
        normal_a, tangent_a, tangent_b = _frame([0.1, 1.0, 0.2])
        normal_b, tangent_c, _ = _frame([-0.3, 0.8, 0.1])
        point_a = np.array([0.0, -0.2, 0.3])
        position_0 = point_a + (RADIUS - 0.003) * normal_a + 0.01 * tangent_a
        # Partner b is placed so that sample 0 also penetrates it by 1.5 mm.
        point_b = position_0 - (RADIUS - 0.0015) * normal_b + 0.02 * tangent_c
        positions = np.stack(
            [
                position_0,
                point_a + (RADIUS - 0.002) * normal_a - 0.02 * tangent_b,
                point_b + (RADIUS - 0.004) * normal_b + 0.015 * tangent_c,
            ]
        )
        translations = np.stack(
            [
                -0.0006 * normal_a + 2.0e-3 * tangent_a,  # approaching, sliding
                0.0004 * normal_a + 1.2e-5 * tangent_b,  # separating, sticking
                -0.0002 * normal_b + 1.5e-5 * tangent_c + 4.0e-4 * tangent_b,  # approaching, sliding
            ]
        )
        # Sample 0 touches both partners; sample 1 partner a; sample 2 partner b; last row masked.
        index = torch.tensor([[0, 0, 1, 2, -1]])
        points = torch.tensor(np.stack([point_a, point_b, point_a, point_b, np.full(3, np.nan)]), dtype=dtype)[None]
        normals = torch.tensor(np.stack([normal_a, normal_b, normal_a, normal_b, np.full(3, np.nan)]), dtype=dtype)[
            None
        ]
        mask = torch.tensor([[True, True, True, True, False]])
        x = torch.tensor(positions, dtype=dtype)[None]
        x0 = torch.tensor(positions - translations, dtype=dtype)[None]
        return x, x0, index, points, normals, mask

    def test_finite_differences_float64(self):
        """Match central differences of the total energy, adding the detached normal-load term."""
        x, x0, index, points, normals, mask = self._scene()
        x.requires_grad_()
        x0.requires_grad_()
        energy = _energy(x, x0, index, points, normals, mask)
        grad_x, grad_x0 = torch.autograd.grad(energy.sum(), (x, x0))
        self.assertTrue(torch.isfinite(grad_x).all() and torch.isfinite(grad_x0).all())

        def total(current, start):
            return _energy(current, start, index, points, normals, mask).sum().item()

        # The cubic friction band makes central differences O(step^2 / eps_u^2); keep the step small.
        step = 1e-8
        fd_x = torch.zeros_like(x)
        fd_x0 = torch.zeros_like(x0)
        for flat in range(x.numel()):
            for target, store in ((x, fd_x), (x0, fd_x0)):
                plus = target.detach().clone()
                minus = target.detach().clone()
                plus.view(-1)[flat] += step
                minus.view(-1)[flat] -= step
                if target is x:
                    store.view(-1)[flat] = (total(plus, x0.detach()) - total(minus, x0.detach())) / (2 * step)
                else:
                    store.view(-1)[flat] = (total(x.detach(), plus) - total(x.detach(), minus)) / (2 * step)
        # The anchor gradient is exact: the detached load does not depend on x0.
        torch.testing.assert_close(grad_x0, fd_x0, rtol=1e-5, atol=1e-7)
        # The position gradient omits mu * f0(|u|) * d f_n / dx = -mu ke f0(|u|) n per pair by design.
        correction = torch.zeros_like(x)
        with torch.no_grad():
            for pair in range(index.shape[1]):
                if not mask[0, pair]:
                    continue
                sample = index[0, pair].item()
                current = x[0, sample]
                start = x0[0, sample]
                normal = normals[0, pair]
                depth = RADIUS - torch.dot(current - points[0, pair], normal).item()
                self.assertGreater(depth, 0.0)
                translation = current - start
                slip = translation - torch.dot(normal, translation) * normal
                correction[0, sample] += -MU * KE * _smoothing(slip.norm().item()) * normal
        torch.testing.assert_close(grad_x + correction, fd_x, rtol=1e-5, atol=1e-7)
        # Without friction, gradcheck verifies the position derivative of normal and damping terms.
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda current, start: _energy(current, start, index, points, normals, mask, mu=0.0),
                (x, x0),
                eps=1e-7,
                atol=1e-8,
                rtol=1e-6,
            )
        )

    def test_no_penetration_gives_zero_energy_and_gradient(self):
        """Return exactly zero energy and gradient when every valid pair is separated, whatever its motion."""
        x, x0, index, points, normals, mask = self._scene()
        # Lift each sample along its partner normals so all gaps exceed the radius.
        lifted = x.clone()
        for pair in range(index.shape[1]):
            if mask[0, pair]:
                lifted[0, index[0, pair]] += 0.01 * normals[0, pair]
        lifted.requires_grad_()
        start = (lifted.detach() - (x - x0)).requires_grad_()
        energy = _energy(lifted, start, index, points, normals, mask)
        self.assertEqual(energy.tolist(), [0.0])
        grad_x, grad_x0 = torch.autograd.grad(energy.sum(), (lifted, start))
        self.assertEqual(grad_x.abs().max().item(), 0.0)
        self.assertEqual(grad_x0.abs().max().item(), 0.0)
        penetration = contact_penetration(lifted, index, points, normals, mask, radius=RADIUS)
        self.assertEqual(penetration.abs().max().item(), 0.0)

    def test_masked_pairs_with_nan_payload_contribute_nothing(self):
        """Ignore masked rows completely, including NaN payload and -1 indices."""
        x, x0, index, points, normals, mask = self._scene()
        x.requires_grad_()
        reference = _energy(x, x0, index[:, :4], points[:, :4], normals[:, :4], mask[:, :4])
        reference_grad = torch.autograd.grad(reference.sum(), x)[0]
        self.assertTrue(torch.isnan(points[0, 4]).all())
        padded = _energy(x, x0, index, points, normals, mask)
        padded_grad = torch.autograd.grad(padded.sum(), x)[0]
        torch.testing.assert_close(padded, reference)
        torch.testing.assert_close(padded_grad, reference_grad)
        # A fully masked pair list yields zero energy for a member even when its rows are NaN.
        empty = _energy(x, x0, index, points, normals, torch.zeros_like(mask))
        self.assertEqual(empty.tolist(), [0.0])
        penetration = contact_penetration(x, index, points, normals, mask, radius=RADIUS)
        self.assertEqual(penetration[0, 4].item(), 0.0)
        self.assertTrue(torch.isfinite(penetration).all())
        np.testing.assert_allclose(penetration[0, :4].detach().numpy(), [0.003, 0.0015, 0.002, 0.004], atol=1e-9)

    def test_zero_slip_has_finite_gradient(self):
        """Keep the friction gradient finite and zero when the sample has not moved."""
        x, _, index, points, normals, mask = self._scene(dtype=torch.float32)
        x.requires_grad_()
        energy = _energy(x, x.detach().clone(), index, points, normals, mask)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        self.assertTrue(torch.isfinite(grad).all())
        depth = contact_penetration(x.detach(), index, points, normals, mask, radius=RADIUS)
        expected = 0.5 * KE * depth.square().sum() + MU * KE * depth.sum() * EPS_U / 3.0
        torch.testing.assert_close(energy.sum(), expected)
        # Only the normal penalty force remains.
        expected_force = torch.zeros_like(grad)
        for pair in range(index.shape[1]):
            if mask[0, pair]:
                expected_force[0, index[0, pair]] += KE * depth[0, pair] * normals[0, pair]
        torch.testing.assert_close(-grad, expected_force, rtol=1e-5, atol=1e-6)

    def test_damping_only_while_approaching_and_penetrating(self):
        """Add the damping energy only for approaching, penetrating pairs."""
        normal, tangent, _ = _frame([0.0, 1.0, 0.0])
        point = np.zeros(3)
        depth = 0.002
        position = point + (RADIUS - depth) * normal
        index = torch.tensor([[0]])
        points = torch.tensor(point, dtype=torch.float64)[None, None]
        normals = torch.tensor(normal, dtype=torch.float64)[None, None]
        mask = torch.ones(1, 1, dtype=torch.bool)
        x = torch.tensor(position, dtype=torch.float64)[None, None]
        for normal_motion in (-0.0007, 0.0007):
            x0 = torch.tensor(position - normal_motion * normal - 3e-4 * tangent, dtype=torch.float64)[None, None]
            damped = _energy(x, x0, index, points, normals, mask, mu=0.0)
            undamped = _energy(x, x0, index, points, normals, mask, mu=0.0, kd=0.0)
            expected = KD / (2 * TIME_STEP) * normal_motion**2 if normal_motion < 0 else 0.0
            self.assertAlmostEqual((damped - undamped).item(), expected, places=12)
        # Approaching without penetration: no damping at all.
        outside = torch.tensor(point + (RADIUS + 0.001) * normal, dtype=torch.float64)[None, None]
        x0 = outside + 0.0007 * normals
        self.assertEqual(_energy(outside, x0, index, points, normals, mask).item(), 0.0)

    def test_friction_matches_ipc_derivative(self):
        """Reproduce the IPC f1(y)/y factor inside the smoothing band and 1/y outside."""
        normal, tangent_a, tangent_b = _frame([0.3, 1.0, -0.2])
        point = np.array([0.02, -0.1, 0.05])
        depth = 0.0025
        position = point + (RADIUS - depth) * normal
        direction = (0.6 * tangent_a - 0.8 * tangent_b) / math.hypot(0.6, 0.8)
        index = torch.tensor([[0]])
        points = torch.tensor(point, dtype=torch.float64)[None, None]
        normals = torch.tensor(normal, dtype=torch.float64)[None, None]
        mask = torch.ones(1, 1, dtype=torch.bool)
        load = KE * depth
        for slip in (0.4 * EPS_U, 5.0 * EPS_U):
            x = torch.tensor(position, dtype=torch.float64)[None, None].requires_grad_()
            x0 = torch.tensor(position - slip * direction, dtype=torch.float64)[None, None]
            energy = _energy(x, x0, index, points, normals, mask, kd=0.0)
            force = -torch.autograd.grad(energy.sum(), x)[0][0, 0].numpy()
            factor = (-slip / EPS_U + 2.0) / EPS_U if slip < EPS_U else 1.0 / slip
            expected = load * normal - MU * load * factor * slip * direction
            np.testing.assert_allclose(force, expected, rtol=1e-9, atol=1e-12)
            tangential = force - np.dot(force, normal) * normal
            self.assertAlmostEqual(np.linalg.norm(tangential), MU * load * min(1.0, factor * slip), places=10)
        # The smoothing is C1 at the band edge: crossing it changes the energy by mu f_n dy only.
        slips = (EPS_U * (1 - 1e-7), EPS_U * (1 + 1e-7))
        energies = []
        forces = []
        for slip in slips:
            x = torch.tensor(position, dtype=torch.float64)[None, None].requires_grad_()
            x0 = torch.tensor(position - slip * direction, dtype=torch.float64)[None, None]
            energy = _energy(x, x0, index, points, normals, mask, kd=0.0)
            energies.append(energy.item())
            forces.append(torch.autograd.grad(energy.sum(), x)[0].numpy())
        expected_change = MU * load * (slips[1] - slips[0])
        self.assertAlmostEqual(energies[1] - energies[0], expected_change, delta=1e-3 * expected_change)
        np.testing.assert_allclose(forces[0], forces[1], rtol=1e-6)

    def test_batch_matches_per_object_evaluation(self):
        """Evaluate two objects with different pair counts together and separately."""
        x, x0, index, points, normals, mask = self._scene()
        # Object A keeps all four pairs; object B keeps one pair with a different material.
        index_b = torch.tensor([[2, -1, -1, -1, -1]])
        mask_b = torch.tensor([[True, False, False, False, False]])
        points_b = torch.full_like(points, float("nan"))
        normals_b = torch.full_like(normals, float("nan"))
        points_b[0, 0] = points[0, 3]
        normals_b[0, 0] = normals[0, 3]
        shift = torch.tensor([0.001, -0.0005, 0.0007], dtype=torch.float64)
        x_b = x + shift
        x0_b = x0 + 0.5 * shift
        ke = torch.tensor([KE, 0.5 * KE], dtype=torch.float64)
        kd = torch.tensor([KD, 2.0 * KD], dtype=torch.float64)
        mu = torch.tensor([MU, 0.25], dtype=torch.float64)

        batched_x = torch.cat([x, x_b]).requires_grad_()
        batched_x0 = torch.cat([x0, x0_b])
        batched = contact_energy(
            batched_x,
            batched_x0,
            torch.cat([index, index_b]),
            torch.cat([points, points_b]),
            torch.cat([normals, normals_b]),
            torch.cat([mask, mask_b]),
            radius=RADIUS,
            ke=ke,
            kd=kd,
            mu=mu,
            time_step=TIME_STEP,
        )
        batched_grad = torch.autograd.grad(batched.sum(), batched_x)[0]
        self.assertEqual(tuple(batched.shape), (2,))
        self.assertTrue((batched > 0).all())
        singles = []
        grads = []
        objects = (
            (x, x0, index, points, normals, mask),
            (x_b, x0_b, index_b[:, :1], points_b[:, :1], normals_b[:, :1], mask_b[:, :1]),
        )
        for member, (xs, x0s, idx, pts, nrm, msk) in enumerate(objects):
            leaf = xs.detach().clone().requires_grad_()
            single = contact_energy(
                leaf,
                x0s,
                idx,
                pts,
                nrm,
                msk,
                radius=RADIUS,
                ke=ke[member : member + 1],
                kd=kd[member : member + 1],
                mu=mu[member : member + 1],
                time_step=TIME_STEP,
            )
            singles.append(single)
            grads.append(torch.autograd.grad(single.sum(), leaf)[0])
        torch.testing.assert_close(batched, torch.cat(singles))
        torch.testing.assert_close(batched_grad, torch.cat(grads))
        self.assertNotAlmostEqual(batched[0].item(), batched[1].item())

    def test_rejects_bad_inputs(self):
        """Raise ValueError for malformed shapes, out-of-range valid indices and bad scalars."""
        x, x0, index, points, normals, mask = self._scene()
        with self.assertRaises(ValueError):
            _energy(x[..., :2], x0[..., :2], index, points, normals, mask)
        with self.assertRaises(ValueError):
            _energy(x, x0[:, :2], index, points, normals, mask)
        with self.assertRaises(ValueError):
            _energy(x, x0, index[:, :3], points, normals, mask)
        with self.assertRaises(ValueError):
            _energy(x, x0, index, points, normals, mask.float())
        with self.assertRaises(ValueError):
            _energy(x, x0, index, points[..., :2], normals, mask)
        bad_index = index.clone()
        bad_index[0, 0] = 3
        with self.assertRaises(ValueError):
            _energy(x, x0, bad_index, points, normals, mask)
        with self.assertRaises(ValueError):
            contact_energy(x, x0, index, points, normals, mask, radius=RADIUS, ke=1.0, kd=0.0, mu=0.0, time_step=0.0)
        with self.assertRaises(ValueError):
            contact_energy(
                x, x0, index, points, normals, mask, radius=-1.0, ke=1.0, kd=0.0, mu=0.0, time_step=TIME_STEP
            )
        with self.assertRaises(ValueError):
            contact_energy(
                x,
                x0,
                index,
                points,
                normals,
                mask,
                radius=RADIUS,
                ke=torch.ones(3),
                kd=0.0,
                mu=0.0,
                time_step=TIME_STEP,
            )
        # Scalar parameters broadcast over the batch.
        scalar = contact_energy(
            x, x0, index, points, normals, mask, radius=RADIUS, ke=KE, kd=KD, mu=MU, time_step=TIME_STEP
        )
        torch.testing.assert_close(scalar, _energy(x, x0, index, points, normals, mask))


if __name__ == "__main__":
    unittest.main()
