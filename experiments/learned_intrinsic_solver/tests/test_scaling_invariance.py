# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the cell-normalisation law of ``notes/ideas/idea-normalize-cells.md``.

A physical scene (cell size h, time step dt, Lamé lambda and mu, density rho,
viscosity eta, gravity g, contact ke, kd and friction mu_f, positions X and
velocities V) and its normalised copy (h' = dt' = mu' = 1, lambda' = lambda /
mu, rho' = rho h^2 / (mu dt^2), eta' = eta / (mu dt), g' = g dt^2 / h, ke' =
ke / (mu h), kd' = kd / (mu h dt), the same mu_f, X' = X / h, V' = V dt / h)
must satisfy exact identities: every energy term scales with mu h^3, position
gradients (forces) with mu h^2, lengths (positions, inertial prediction, fused
positions, penetration depth) with h, velocities with h / dt, and a whole
implicit-Euler trajectory produced by the same network-free minimiser is the
same up to the factor h, hence has the same deformation gradients.

One parameter the note does not list must transform as well: the IPC friction
band of :func:`contact_energy.contact_energy` is ``eps_u = friction_epsilon *
dt`` (a length), so ``friction_epsilon`` is a velocity and scales as
``friction_epsilon' = friction_epsilon dt / h``. The contact test shows the
identity breaks for slips inside the band when it is left at its SI value.

The plain-energy, gradient, contact and fusion identities are checked in
float64 to 1e-9. :class:`mixed_physics.MixedHexSolverStep` (energy floor,
rigid candidate, contact detection, network inputs, trajectory) is
float32-only and is checked to 1e-4 relative to the natural scale of each
quantity (h for lengths, h / dt for velocities, one for deformation
gradients). "The same optimizer" shares the minimiser, not the iterates:
``torch.optim.LBFGS`` scales its first trial step by ``1 / |g|_1`` and the
fixed-step polish uses an absolute unit step, neither of which is covariant
under X = h X', so the two float32 minimisations follow different paths and
only agree to the precision each reaches. Strong-Wolfe L-BFGS stalls when
energy differences drop below ``eps32 * E_total`` (about 1e-4 h here), so the
trajectory test appends a fixed-step L-BFGS polish that works from gradient
differences alone and reaches about 1e-6 h. The float64 one-step minimiser of
the negative test has no such limit and agrees to 1e-6.

Since :class:`mixed_physics.MixedHexSolverStep` evaluates its objective, fusion
and network inputs in cell units (``tests/test_normalised_physics.py`` checks
the SI equivalence), every network input coincides between the two spaces,
including the state scalar ``log_gradient_rms`` and all conditioning channels
of :func:`features.conditioning_channels`, which are the dimensionless groups
of the law. Absolute constants that never enter the energy are not covered:
the static-point sampling box margins of :mod:`contact_scene` (0.10 m and
0.35 m), its plane partner radius placeholder (1e9 m, saturating the token
channel in both spaces), ``features.RMS_FLOOR`` (1e-12, now in cell units) and
the 1e-12 degeneracy guards of the geometry helpers.
"""

import importlib.util
import itertools
import math
import unittest
from dataclasses import dataclass, replace

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.contact_energy import contact_energy, contact_penetration
from experiments.learned_intrinsic_solver.contact_geometry import exposed_face_samples, sample_normals, sample_points
from experiments.learned_intrinsic_solver.contact_scene import ContactPartners, detect_contacts
from experiments.learned_intrinsic_solver.fusion import HexFusion
from experiments.learned_intrinsic_solver.hex_energy import HexImplicitEulerLoss, make_inertial_prediction
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.train_mixed import _batch

CELL_COUNTS = (2, 2, 3)
CELL_SIZE = 0.025
TIME_STEP = 1.0 / 300.0
GRAVITY = (0.0, -9.81, 0.0)
FRICTION_EPSILON = 1e-2
"""Default IPC band velocity of the mixed step [m/s]: ``eps_u = FRICTION_EPSILON * dt`` metres."""

YOUNGS_MODULI = (1e3, 1e6)
POISSON_RATIOS = (0.3, 0.45)
DENSITIES = (100.0, 5000.0)
VISCOSITIES = (0.0, 300.0)
CONTACT_KAPPA, CONTACT_BETA, CONTACT_FRICTION = 1.0, 0.5, 0.3

FLOAT64_RTOL = 1e-9
"""Bound on the max-norm relative deviation of the float64 identities."""

FLOAT32_RTOL = 1e-4
"""Bound on the max-norm relative deviation of the float32 mixed-step identities."""


@dataclass(frozen=True)
class Scene:
    """Parameters of one scene, in SI units or in cell units after :meth:`normalised`."""

    cell_size: float
    time_step: float
    lame_lambda: float
    lame_mu: float
    density: float
    damping: float
    gravity: tuple[float, float, float]
    contact_ke: float
    contact_kd: float
    contact_mu: float
    friction_epsilon: float

    @property
    def energy_scale(self) -> float:
        """Return mu h^3, the factor between physical and normalised energies."""
        return self.lame_mu * self.cell_size**3

    @property
    def force_scale(self) -> float:
        """Return mu h^2, the factor between physical and normalised position gradients."""
        return self.lame_mu * self.cell_size**2

    @property
    def velocity_scale(self) -> float:
        """Return h / dt, the factor between physical and normalised velocities."""
        return self.cell_size / self.time_step

    @property
    def youngs_modulus(self) -> float:
        lam, mu = self.lame_lambda, self.lame_mu
        return mu * (3 * lam + 2 * mu) / (lam + mu)

    def normalised(self) -> "Scene":
        """Return the copy with h = dt = mu = 1 and every other quantity transformed by the law."""
        h, dt, mu = self.cell_size, self.time_step, self.lame_mu
        return Scene(
            cell_size=1.0,
            time_step=1.0,
            lame_lambda=self.lame_lambda / mu,
            lame_mu=1.0,
            density=self.density * h**2 / (mu * dt**2),
            damping=self.damping / (mu * dt),
            gravity=tuple(component * dt**2 / h for component in self.gravity),
            contact_ke=self.contact_ke / (mu * h),
            contact_kd=self.contact_kd / (mu * h * dt),
            contact_mu=self.contact_mu,
            friction_epsilon=self.friction_epsilon * dt / h,
        )

    def partially_normalised(self) -> "Scene":
        """Return the copy with h = 1 and mu = 1 only; rho, g, eta and dt keep their SI values."""
        return replace(self, cell_size=1.0, lame_lambda=self.lame_lambda / self.lame_mu, lame_mu=1.0)

    def groups(self) -> dict[str, float]:
        """Return the dimensionless groups of the note computed from the raw parameters."""
        lam, mu, h, dt = self.lame_lambda, self.lame_mu, self.cell_size, self.time_step
        ke = self.contact_ke
        return {
            "poisson": lam / mu,
            "Lambda": self.density * h**2 / (mu * dt**2),
            "Gamma": math.sqrt(sum(component**2 for component in self.gravity)) * dt**2 / h,
            "Xi": self.damping / (mu * dt),
            "kappa": ke / (self.youngs_modulus * h),
            "beta": self.contact_kd / (ke * dt) if ke > 0 else 0.0,
            "friction": self.contact_mu,
        }


def _scene(youngs, poisson, density, damping) -> Scene:
    """Return an SI scene with the Lamé pair of (E, nu) and contact ke = kappa E h, kd = beta ke dt."""
    lam = youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = youngs / (2 * (1 + poisson))
    ke = CONTACT_KAPPA * youngs * CELL_SIZE
    return Scene(
        cell_size=CELL_SIZE,
        time_step=TIME_STEP,
        lame_lambda=lam,
        lame_mu=mu,
        density=density,
        damping=damping,
        gravity=GRAVITY,
        contact_ke=ke,
        contact_kd=CONTACT_BETA * ke * TIME_STEP,
        contact_mu=CONTACT_FRICTION,
        friction_epsilon=FRICTION_EPSILON,
    )


def _all_scenes() -> list[Scene]:
    return [_scene(*combo) for combo in itertools.product(YOUNGS_MODULI, POISSON_RATIOS, DENSITIES, VISCOSITIES)]


def _deviation(actual, expected) -> float:
    """Return ``max|actual - expected| / max|expected|``, the scale-free max-norm relative deviation."""
    actual = torch.as_tensor(actual).detach().to(torch.float64)
    expected = torch.as_tensor(expected).detach().to(torch.float64)
    scale = expected.abs().max().clamp_min(torch.finfo(torch.float64).tiny)
    return ((actual - expected).abs().max() / scale).item()


def _smooth_field(rest, generator, *, amplitude: float, dtype, waves=(1, 2, 4)) -> torch.Tensor:
    """Return a smooth random vector field on the rest corners, shape [1, P, 3], entries of about ``amplitude``.

    The field is a sum of sinusoids over three wavelengths with random wave
    vectors, phases and component weights; the amplitude decays with the wave
    number so the field is dominated by its smooth part.
    """
    positions = torch.tensor(rest.corner_rest_positions, dtype=dtype)
    lower, upper = positions.min(0).values, positions.max(0).values
    unit = (positions - lower) / (upper - lower)
    field = torch.zeros_like(positions)
    for wave in waves:
        directions = torch.randn((3, 3), generator=generator, dtype=dtype)
        phases = 2 * math.pi * torch.rand(3, generator=generator, dtype=dtype)
        weights = torch.randn(3, generator=generator, dtype=dtype)
        field = field + (amplitude / wave) * weights * torch.sin(2 * math.pi * wave * (unit @ directions) + phases)
    return field[None]


def _deformed_state(rest, generator, *, amplitude: float, dtype, rotate: bool = True) -> torch.Tensor:
    """Return the rest grid under a random rigid rotation about its centre plus smooth noise, shape [1, P, 3]."""
    positions = torch.tensor(rest.corner_rest_positions, dtype=dtype)
    if rotate:
        rotation, upper = torch.linalg.qr(torch.randn((3, 3), generator=generator, dtype=dtype))
        rotation = rotation * torch.sign(torch.diagonal(upper))
        if torch.linalg.det(rotation) < 0:
            rotation = rotation.clone()
            rotation[:, 0] = -rotation[:, 0]
        center = positions.mean(0)
        positions = center + (positions - center) @ rotation.T
    return positions[None] + _smooth_field(rest, generator, amplitude=amplitude, dtype=dtype)


def _floor_partners(scene: Scene, *, height: float) -> ContactPartners:
    """Return partners with only a ground plane at ``height`` (in the scene's length unit) and the scene's ke, kd, mu."""
    return ContactPartners(
        plane_present=True,
        plane_point=torch.tensor([0.0, height, 0.0]),
        plane_normal=torch.tensor([0.0, 1.0, 0.0]),
        point_positions=torch.zeros((0, 3)),
        point_normals=torch.zeros((0, 3)),
        point_radii=torch.zeros((0,)),
        ke=scene.contact_ke,
        kd=scene.contact_kd,
        mu=scene.contact_mu,
    )


def _network(cell_counts, *, contact_tokens: bool = False, target_modes: int = 3) -> IntrinsicSolverNetwork:
    """Return a tiny revised-schema network; the mixed step needs one even when forward is never called."""
    return IntrinsicSolverNetwork(
        cell_counts,
        features.state_feature_dim(target_modes),
        target_modes=target_modes,
        conditioning_dim=features.CONDITIONING_DIM,
        hidden_dim=16,
        edge_hidden_dim=8,
        contact_tokens=contact_tokens,
    )


def _state_column(name: str, target_modes: int = 3) -> int:
    """Return the column of the trailing scalar ``name`` in the packed state features of ``target_modes`` modes."""
    matrix_width = len(features.MATRIX_BLOCKS) * features.target_dim(target_modes)
    return matrix_width + features.BOUNDARY_DIM + features.SCALAR_FEATURES.index(name)


def _pairs(payload) -> list[tuple[int, int]]:
    """Return the frozen (sample, kind) pairs of a prepared payload, sorted."""
    return sorted(zip(payload["contact_sample_index"].tolist(), payload["contact_kind"].tolist(), strict=True))


def _minimise(step: MixedHexSolverStep, payload: dict, free: torch.Tensor, *, iterations: int, polish: int):
    """Minimise the physical objective over the free corners with plain L-BFGS from the rigid candidate.

    The same network-free procedure runs in both spaces. Phase one runs
    ``iterations`` strong-Wolfe L-BFGS iterations with no gradient or change
    tolerance. In float32 that phase stalls once energy differences along the
    line search fall below ``eps32 * E_total`` (about 1e-4 h in position, see
    the probe discussion in the module docstring), so phase two runs
    ``polish`` fixed-unit-step L-BFGS iterations that use only gradient
    differences and converge to the much lower gradient-noise floor. Returns
    the positions [P, 3], the collated batch, the max-norm gradient at the
    solution and the total energies after each phase.
    """
    batch = _batch([payload], torch.device("cpu"))
    candidate = batch["candidate"]
    variable = candidate[0, free].clone().requires_grad_(True)

    def assemble():
        positions = candidate.clone()
        positions[0, free] = variable
        return positions

    def objective():
        return step.energy(
            assemble(),
            batch["inertial_prediction"],
            batch["context_ids"],
            previous_positions=batch["physical_positions"],
            contact=batch["contact"],
        ).total.sum()

    def run(count, line_search):
        optimizer = torch.optim.LBFGS(
            [variable],
            lr=1.0,
            max_iter=count,
            tolerance_grad=0.0,
            tolerance_change=0.0,
            history_size=count,
            line_search_fn=line_search,
        )

        def closure():
            optimizer.zero_grad()
            total = objective()
            total.backward()
            return total

        optimizer.step(closure)
        return objective().item()

    wolfe_energy = run(iterations, "strong_wolfe")
    polished_energy = run(polish, None)
    gradient = torch.autograd.grad(objective(), variable)[0]
    with torch.no_grad():
        positions = assemble()[0].clone()
    return positions, batch, gradient.abs().max().item(), (wolfe_energy, polished_energy)


def _minimise_float64(loss: HexImplicitEulerLoss, prediction, previous, free, *, iterations: int = 300):
    """Return the float64 implicit-Euler minimiser [1, P, 3] over ``free`` corners and its max-norm gradient.

    Starts from the inertial prediction with the pinned corners held at their
    predicted (= previous) positions and runs strong-Wolfe L-BFGS with no
    tolerances; in float64 the line search does not stall at the level that
    matters here (1e-6 in F).
    """
    variable = prediction[0, free].clone().requires_grad_(True)

    def assemble():
        positions = prediction.clone()
        positions[0, free] = variable
        return positions

    def objective():
        return loss(assemble(), prediction, previous_positions=previous).total.sum()

    optimizer = torch.optim.LBFGS(
        [variable],
        lr=1.0,
        max_iter=iterations,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        history_size=iterations,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        total = objective()
        total.backward()
        return total

    optimizer.step(closure)
    gradient = torch.autograd.grad(objective(), variable)[0]
    return assemble().detach(), gradient.abs().max().item()


class TestScalingInvariance(unittest.TestCase):
    """Check the identities between a physical scene and its normalised copy on a (2, 2, 3) grid."""

    def setUp(self):
        self.rest = generate_cuboid(CELL_COUNTS, cell_size=CELL_SIZE)
        self.unit_rest = generate_cuboid(CELL_COUNTS, cell_size=1.0)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.generator = torch.Generator().manual_seed(2026)

    def _mixed_steps(self, scene: Scene, *, contact_tokens: bool = False, target_modes: int = 3):
        """Return (physical, normalised) mixed steps sharing one network, each with a "beam" floor-contact context.

        Gravity and the friction band are constructor constants of the step,
        so the normalised copy is built with g' = g dt^2 / h and
        friction_epsilon' = friction_epsilon dt / h; the floor lies 0.3 h under
        the rest bottom face in both spaces. ``target_modes`` selects the
        three- or seven-mode step and network.
        """
        unit = scene.normalised()
        network = _network(CELL_COUNTS, contact_tokens=contact_tokens, target_modes=target_modes)
        steps = []
        for rest, material, height in ((self.rest, scene, -0.3 * scene.cell_size), (self.unit_rest, unit, -0.3)):
            step = MixedHexSolverStep(
                rest,
                self.fixed,
                network=network,
                time_step=material.time_step,
                gravity=material.gravity,
                contact_friction_epsilon=material.friction_epsilon,
                target_modes=target_modes,
            )
            self.addCleanup(step.close)
            step.register_context(
                "beam",
                lame_lambda=material.lame_lambda,
                lame_mu=material.lame_mu,
                density=material.density,
                damping=material.damping,
                contact=_floor_partners(material, height=height),
            )
            steps.append(step)
        return tuple(steps)

    def _free_corners(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pinned mask [P], free corner indices) for the clamped z = 0 face."""
        pinned = torch.zeros(len(self.rest.corner_rest_positions), dtype=torch.bool)
        pinned[self.fixed] = True
        return pinned, torch.nonzero(~pinned).flatten()

    def _initial_state(self, pinned: torch.Tensor):
        """Return SI positions and velocities [P, 3] float64: rest + 1 % h smooth noise, 5 % h / dt drifting down."""
        h, dt = CELL_SIZE, TIME_STEP
        displacement = _smooth_field(self.rest, self.generator, amplitude=0.01 * h, dtype=torch.float64)[0]
        velocity = _smooth_field(self.rest, self.generator, amplitude=0.05 * h / dt, dtype=torch.float64)[0]
        velocity[:, 1] -= 0.03 * h / dt
        displacement[pinned] = 0.0
        velocity[pinned] = 0.0
        return torch.tensor(self.rest.corner_rest_positions, dtype=torch.float64) + displacement, velocity

    def _states(self, *, rotate: bool = True):
        """Return (previous, current, velocity) in SI, each [1, P, 3] float64.

        The previous positions are a rotated rest grid with 3 % h smooth
        noise, the current positions add a 1 % h smooth increment and the
        velocity is a smooth field of about 0.05 h / dt.
        """
        previous = _deformed_state(
            self.rest, self.generator, amplitude=0.03 * CELL_SIZE, dtype=torch.float64, rotate=rotate
        )
        current = previous + _smooth_field(self.rest, self.generator, amplitude=0.01 * CELL_SIZE, dtype=torch.float64)
        velocity = _smooth_field(self.rest, self.generator, amplitude=0.05 * CELL_SIZE / TIME_STEP, dtype=torch.float64)
        return previous, current, velocity

    @staticmethod
    def _losses(scene: Scene, rest, unit_rest):
        """Return the float64 implicit-Euler losses of a scene and of its normalised copy."""
        unit = scene.normalised()
        physical = HexImplicitEulerLoss(
            rest,
            scene.lame_lambda,
            scene.lame_mu,
            scene.density,
            scene.time_step,
            damping=scene.damping,
            dtype=torch.float64,
        )
        normalised = HexImplicitEulerLoss(
            unit_rest,
            unit.lame_lambda,
            unit.lame_mu,
            unit.density,
            unit.time_step,
            damping=unit.damping,
            dtype=torch.float64,
        )
        return physical, normalised

    @staticmethod
    def _predictions(scene: Scene, previous, velocity):
        """Return Y in SI and Y' in cell units from make_inertial_prediction applied in each space."""
        unit = scene.normalised()
        h, dt = scene.cell_size, scene.time_step
        prediction = make_inertial_prediction(
            previous, velocity, dt, explicit_acceleration=torch.tensor(scene.gravity, dtype=torch.float64)
        )
        unit_prediction = make_inertial_prediction(
            previous / h,
            velocity * dt / h,
            unit.time_step,
            explicit_acceleration=torch.tensor(unit.gravity, dtype=torch.float64),
        )
        return prediction, unit_prediction

    def test_energy_terms_scale_with_mu_h3(self):
        """Elastic, inertia, damping and total energies satisfy E = mu h^3 E', and Y = h Y', for 16 materials."""
        worst = dict.fromkeys(("prediction", "total", "elastic", "inertia", "damping"), 0.0)
        for scene in _all_scenes():
            physical, normalised = self._losses(scene, self.rest, self.unit_rest)
            h = scene.cell_size
            for _ in range(2):
                previous, current, velocity = self._states()
                prediction, unit_prediction = self._predictions(scene, previous, velocity)
                worst["prediction"] = max(worst["prediction"], _deviation(prediction, h * unit_prediction))
                terms = physical(current, prediction, previous_positions=previous)
                unit_terms = normalised(current / h, unit_prediction, previous_positions=previous / h)
                for name in ("total", "elastic", "inertia", "damping"):
                    expected = scene.energy_scale * getattr(unit_terms, name)
                    worst[name] = max(worst[name], _deviation(getattr(terms, name), expected))
                # Every compared term is non-trivial (damping only when eta > 0).
                self.assertGreater(terms.elastic.item(), 0.0)
                self.assertGreater(terms.inertia.item(), 0.0)
                self.assertEqual(terms.damping.item() > 0.0, scene.damping > 0.0)
        print(f"energy terms, max relative deviation: {worst}")
        for name, value in worst.items():
            self.assertLessEqual(value, FLOAT64_RTOL, name)

    def test_force_gradient_scales_with_mu_h2(self):
        """The autograd position gradient of the total satisfies dE/dX = mu h^2 dE'/dX' for 16 materials."""
        worst = 0.0
        for scene in _all_scenes():
            physical, normalised = self._losses(scene, self.rest, self.unit_rest)
            h = scene.cell_size
            previous, current, velocity = self._states()
            prediction, unit_prediction = self._predictions(scene, previous, velocity)
            positions = current.clone().requires_grad_(True)
            gradient = torch.autograd.grad(
                physical(positions, prediction, previous_positions=previous).total.sum(), positions
            )[0]
            unit_positions = (current / h).clone().requires_grad_(True)
            unit_gradient = torch.autograd.grad(
                normalised(unit_positions, unit_prediction, previous_positions=previous / h).total.sum(), unit_positions
            )[0]
            self.assertGreater(gradient.abs().max().item(), 0.0)
            worst = max(worst, _deviation(gradient, scene.force_scale * unit_gradient))
        print(f"force gradient, max relative deviation: {worst:.3e}")
        self.assertLessEqual(worst, FLOAT64_RTOL)

    def test_contact_energy_and_penetration_scale(self):
        """Penalty contact energy scales with mu h^3 and penetration depth with h on hand-built floor and disk pairs.

        Two batch members share the pairs. The first moves 3 % h towards the
        floor with slips far outside the IPC friction band, so its normal,
        damping (``approach = relu(-n . (x - x_start)) > 0``) and sliding
        friction terms are all nonzero, which the test proves by dropping kd
        and mu_f in turn. The second barely moves, so its slips lie deep inside
        the band. Two negative checks show the transformations the identity
        needs: kd' = kd / (mu h dt) (dropping the 1 / dt breaks the first
        member) and friction_epsilon' = friction_epsilon dt / h (the band
        length ``eps_u = friction_epsilon * dt`` must scale with h; leaving it
        at its SI value breaks the in-band member).
        """
        scene = _scene(1e3, 0.3, 100.0, 0.0)
        unit = scene.normalised()
        h, dt, mu = scene.cell_size, scene.time_step, scene.lame_mu
        radius = 0.5 * h
        faces = exposed_face_samples(self.rest)
        start = _deformed_state(self.rest, self.generator, amplitude=0.02 * h, dtype=torch.float64, rotate=False)
        start = start.expand(2, -1, -1)
        increments = torch.cat(
            [
                _smooth_field(self.rest, self.generator, amplitude=0.02 * h, dtype=torch.float64),
                _smooth_field(self.rest, self.generator, amplitude=1e-4 * h, dtype=torch.float64),
            ]
        )
        # A uniform tangential shift keeps every slip of the first member well outside the friction band, and
        # a uniform downward shift makes its penetrating samples approach the floor so the kd term is active.
        increments[0, :, 0] += 0.05 * h
        increments[0, :, 1] -= 0.03 * h
        current = start + increments
        samples_start = sample_points(start, faces.corners)
        samples = sample_points(current, faces.corners)

        # Floor 0.3 h below the rest bottom (samples at y ~ 0, so depth d = r - gap ~ 0.2 h) plus one tilted disk.
        plane_point = torch.tensor([0.0, -0.3 * h, 0.0], dtype=torch.float64)
        plane_normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        bottom = torch.nonzero(faces.face_index == 2).flatten()
        sides = torch.nonzero(faces.face_index <= 1).flatten()[:2]
        disk_sample = bottom[1]
        disk_point = samples_start[0, disk_sample] + torch.tensor([0.1 * h, -0.4 * h, -0.05 * h], dtype=torch.float64)
        disk_normal = torch.tensor([0.2, 1.0, 0.1], dtype=torch.float64)
        disk_normal = disk_normal / disk_normal.norm()
        index = torch.cat([bottom, sides, disk_sample[None], torch.zeros(1, dtype=torch.int64)])
        plane_rows = len(bottom) + len(sides)
        foot = samples_start[:, index[:plane_rows]]
        foot = foot - ((foot - plane_point) @ plane_normal)[..., None] * plane_normal
        points = torch.cat([foot, disk_point.expand(2, 1, 3), torch.zeros((2, 1, 3), dtype=torch.float64)], dim=1)
        normals = torch.cat(
            [
                plane_normal.expand(2, plane_rows, 3),
                disk_normal.expand(2, 1, 3),
                torch.zeros((2, 1, 3), dtype=torch.float64),
            ],
            dim=1,
        )
        index = index[None].expand(2, -1)
        mask = torch.ones(index.shape, dtype=torch.bool)
        mask[:, -1] = False

        def energy(positions, starts, scene_):
            return contact_energy(
                positions,
                starts,
                index,
                points / (h / scene_.cell_size),
                normals,
                mask,
                radius=radius / (h / scene_.cell_size),
                ke=scene_.contact_ke,
                kd=scene_.contact_kd,
                mu=scene_.contact_mu,
                time_step=scene_.time_step,
                friction_epsilon=scene_.friction_epsilon,
            )

        physical = energy(samples, samples_start, scene)
        normalised = energy(samples / h, samples_start / h, unit)
        depth = contact_penetration(samples, index, points, normals, mask, radius=radius)
        unit_depth = contact_penetration(samples / h, index, points / h, normals, mask, radius=radius / h)
        self.assertTrue((physical > 0).all())
        self.assertTrue((depth[:, : len(bottom)] > 0).all(), "floor pairs under the bottom face penetrate")
        self.assertTrue((depth[:, len(bottom) : plane_rows] == 0).all(), "side faces do not reach the floor")
        self.assertGreater(depth[0, plane_rows].item(), 0.0, "the disk pair penetrates")
        deviation = {
            "energy": _deviation(physical, scene.energy_scale * normalised),
            "depth": _deviation(depth, h * unit_depth),
        }
        # Every branch of contact_energy is active for the approaching member: dropping kd or mu_f lowers its energy.
        translation = (samples - samples_start)[:, index[0, : len(bottom)]]
        approach = -(translation[0] @ plane_normal)
        self.assertGreater(approach.min().item(), 0.0, "every floor pair of the first member approaches the floor")
        fractions = {}
        for name, field in (("kd", "contact_kd"), ("mu", "contact_mu")):
            without = energy(samples, samples_start, replace(scene, **{field: 0.0}))
            fractions[name] = ((physical - without) / physical).tolist()
            self.assertLess(without[0].item(), physical[0].item(), f"the {name} term of the first member is active")
        self.assertGreater(fractions["kd"][0], 2e-3, "the damping term carries a measurable share of the energy")
        self.assertLess(fractions["kd"][1], 1e-6, "the barely moving member has a negligible damping term")
        print(f"contact energy fractions removed by kd = 0 and mu_f = 0 per member: {fractions}")
        # Negative check: kd' = kd / (mu h) (no 1 / dt) breaks the identity for the approaching member; the barely
        # moving member cannot see the wrong exponent, which is why an approaching member is needed at all.
        wrong_kd = energy(samples / h, samples_start / h, replace(unit, contact_kd=scene.contact_kd / (mu * h)))
        deviation["kd_without_dt_approaching"] = _deviation(physical[0], scene.energy_scale * wrong_kd[0])
        self.assertLess(_deviation(physical[1], scene.energy_scale * wrong_kd[1]), 1e-6)
        # The in-band member's slips lie inside eps_u = friction_epsilon dt; the other's far outside.
        slip = translation - (translation @ plane_normal)[..., None] * plane_normal
        band = scene.friction_epsilon * dt
        self.assertGreater(slip[0].norm(dim=-1).min().item(), 5 * band)
        self.assertLess(slip[1].norm(dim=-1).max().item(), 0.2 * band)
        unscaled = energy(samples / h, samples_start / h, replace(unit, friction_epsilon=scene.friction_epsilon))
        deviation["unscaled_friction_epsilon_in_band"] = _deviation(physical[1], scene.energy_scale * unscaled[1])
        print(f"contact, max relative deviation: {deviation}")
        self.assertLessEqual(deviation["energy"], FLOAT64_RTOL)
        self.assertLessEqual(deviation["depth"], FLOAT64_RTOL)
        self.assertGreater(deviation["kd_without_dt_approaching"], 1e-3)
        self.assertGreater(deviation["unscaled_friction_epsilon_in_band"], 1e-3)

        # Detection on the step-start shape finds the same pairs in both spaces.
        velocity = _smooth_field(self.rest, self.generator, amplitude=0.05 * h / dt, dtype=torch.float64)
        sample_velocity = sample_points(velocity, faces.corners)[0]
        face_normals = sample_normals(start[:1], faces.corners, faces.rest_normals)[0]
        partners = ContactPartners(
            plane_present=True,
            plane_point=plane_point,
            plane_normal=plane_normal,
            point_positions=disk_point[None],
            point_normals=disk_normal[None],
            point_radii=torch.tensor([1.5 * h]),
            ke=scene.contact_ke,
            kd=scene.contact_kd,
            mu=scene.contact_mu,
        )
        unit_partners = ContactPartners(
            plane_present=True,
            plane_point=plane_point / h,
            plane_normal=plane_normal,
            point_positions=disk_point[None] / h,
            point_normals=disk_normal[None],
            point_radii=torch.tensor([1.5]),
            ke=unit.contact_ke,
            kd=unit.contact_kd,
            mu=unit.contact_mu,
        )
        detected = detect_contacts(
            samples_start[0], sample_velocity, partners, radius=radius, time_step=dt, sample_normals=face_normals
        )
        unit_detected = detect_contacts(
            samples_start[0] / h,
            sample_velocity * dt / h,
            unit_partners,
            radius=radius / h,
            time_step=unit.time_step,
            sample_normals=face_normals,
        )
        self.assertGreater(detected.sample_index.numel(), len(bottom))
        for name in ("sample_index", "partner_index", "kind"):
            self.assertEqual(getattr(detected, name).tolist(), getattr(unit_detected, name).tolist(), name)
        self.assertLessEqual(_deviation(detected.partner_point, h * unit_detected.partner_point), 1e-6)

    def test_fusion_is_scale_equivariant(self):
        """HexFusion gives X_fused = h X'_fused for the same relative axis increments and the same weight pattern.

        The identity the law needs is geometric: the gradient operator scales
        as 1 / h and the increments are dimensionless, so the weighted
        least-squares fit of ``grad(X_fused - X) = D`` scales with h. The mu
        h^3 factor of the physical weights is not observable here (and
        therefore not validated): a weighted least-squares minimiser is
        invariant under any uniform rescaling of its weights, which the test
        demonstrates with a factor 7.3. What does matter is the relative
        weight pattern across cells, so the two spaces use the same non-uniform
        pattern and the test checks that a different pattern changes the fit.
        """
        worst = 0.0
        cell_count = len(self.rest.cell_corner_indices)
        pattern = 1 + 0.5 * torch.sin(torch.arange(cell_count, dtype=torch.float64))
        for scene in (_scene(1e3, 0.3, 100.0, 0.0), _scene(1e6, 0.45, 5000.0, 300.0)):
            unit = scene.normalised()
            h = scene.cell_size

            def stiffness(material: Scene) -> float:
                return material.lame_mu * (3 - material.lame_mu / (material.lame_lambda + material.lame_mu))

            weights = stiffness(scene) * h**3 * pattern
            unit_weights = stiffness(unit) * pattern
            fusion = HexFusion(self.rest, self.fixed, cell_weights=weights, dtype=torch.float64)
            unit_fusion = HexFusion(self.unit_rest, self.fixed, cell_weights=unit_weights, dtype=torch.float64)
            base = _deformed_state(self.rest, self.generator, amplitude=0.03 * h, dtype=torch.float64)
            increment = 0.05 * torch.randn((1, cell_count, 3, 3), generator=self.generator, dtype=torch.float64)
            prescribed = base[:, self.fixed] + 0.01 * h * torch.randn(
                (1, len(self.fixed), 3), generator=self.generator, dtype=torch.float64
            )
            fused = fusion.fuse(base, increment, prescribed)
            unit_fused = unit_fusion.fuse(base / h, increment, prescribed / h)
            self.assertGreater(_deviation(fused, base), 1e-3, "the increment moves the corners")
            worst = max(worst, _deviation(fused, h * unit_fused))
            # A uniform rescaling of the weights leaves the weighted least-squares minimiser unchanged ...
            rescaled = HexFusion(self.unit_rest, self.fixed, cell_weights=7.3 * unit_weights, dtype=torch.float64)
            self.assertLessEqual(_deviation(rescaled.fuse(base / h, increment, prescribed / h), unit_fused), 1e-12)
            # ... while a different relative pattern does not, so the pattern comparison above is not vacuous.
            uniform = HexFusion(self.unit_rest, self.fixed, cell_weights=unit_weights / pattern, dtype=torch.float64)
            self.assertGreater(_deviation(uniform.fuse(base / h, increment, prescribed / h), unit_fused), 1e-3)
        print(f"fusion, max relative deviation: {worst:.3e}")
        self.assertLessEqual(worst, FLOAT64_RTOL)

    def test_dimensionless_groups_algebra_self_check(self):
        """The note's groups computed from raw SI parameters equal those of the normalised parameters.

        This exercises only the :class:`Scene` algebra of this file
        (``normalised`` against ``groups``), not any production module; it
        guards the test's own transformation table, while the production
        exponents are covered by the energy, contact and trajectory tests.
        """
        for scene in _all_scenes():
            unit = scene.normalised()
            self.assertEqual((unit.cell_size, unit.time_step, unit.lame_mu), (1.0, 1.0, 1.0))
            for name, value in scene.groups().items():
                self.assertTrue(math.isclose(value, unit.groups()[name], rel_tol=1e-12, abs_tol=0.0), name)
            self.assertFalse(math.isclose(scene.groups()["Lambda"], scene.partially_normalised().groups()["Lambda"]))

    def test_energy_floor_and_conditioning_channels(self):
        """MixedHexSolverStep.energy_floor scales with mu h^3 and every conditioning channel is invariant."""
        scenes = [
            _scene(1e3, 0.3, 100.0, 0.0),
            _scene(1e6, 0.45, 5000.0, 300.0),
            _scene(1e3, 0.45, 5000.0, 300.0),
            _scene(1e6, 0.3, 100.0, 0.0),
        ]
        units = [scene.normalised() for scene in scenes]
        network = _network(CELL_COUNTS)
        physical = MixedHexSolverStep(
            self.rest,
            self.fixed,
            network=network,
            time_step=TIME_STEP,
            gravity=GRAVITY,
            contact_friction_epsilon=FRICTION_EPSILON,
        )
        normalised = MixedHexSolverStep(
            self.unit_rest,
            self.fixed,
            network=network,
            time_step=units[0].time_step,
            gravity=units[0].gravity,
            contact_friction_epsilon=units[0].friction_epsilon,
        )
        self.addCleanup(physical.close)
        self.addCleanup(normalised.close)
        ids = tuple(f"material{index}" for index in range(len(scenes)))
        for name, scene, unit in zip(ids, scenes, units, strict=True):
            for step, material, height in ((physical, scene, -0.3 * CELL_SIZE), (normalised, unit, -0.3)):
                step.register_context(
                    name,
                    lame_lambda=material.lame_lambda,
                    lame_mu=material.lame_mu,
                    density=material.density,
                    damping=material.damping,
                    contact=_floor_partners(material, height=height),
                )
        floor = physical.energy_floor(ids)
        unit_floor = normalised.energy_floor(ids)
        scale = torch.tensor([scene.energy_scale for scene in scenes])
        deviation = _deviation(floor, scale * unit_floor)
        print(f"energy floor (float32 output), max relative deviation: {deviation:.3e}")
        self.assertTrue((floor > 0).all())
        self.assertLessEqual(deviation, 1e-6)

        def channels(materials, cell_size, time_step):
            column = lambda attribute: torch.tensor([getattr(m, attribute) for m in materials], dtype=torch.float64)  # noqa: E731
            return features.conditioning_channels(
                column("lame_lambda"),
                column("lame_mu"),
                column("density"),
                column("damping"),
                cell_size,
                time_step,
                materials[0].gravity,
                contact_ke=column("contact_ke"),
                contact_kd=column("contact_kd"),
                contact_mu=column("contact_mu"),
            )

        conditioning = channels(scenes, CELL_SIZE, TIME_STEP)
        unit_conditioning = channels(units, 1.0, 1.0)
        # Every channel is a dimensionless group of the law, so the two spaces agree column by column.
        for column, name in enumerate(features.CONDITIONING_CHANNELS):
            torch.testing.assert_close(
                conditioning[:, column], unit_conditioning[:, column], rtol=1e-9, atol=1e-12, msg=name
            )
        self.assertTrue((conditioning[:, :5] != 0).any(dim=0).all(), "every material and contact group is exercised")
        # A scene rescaled in h alone (rho, g, dt unchanged) is a different problem and gets different channels.
        self.assertFalse(torch.allclose(channels(scenes, 2 * CELL_SIZE, TIME_STEP), conditioning))

    def test_network_inputs_are_invariant(self):
        """Every network input of MixedHexSolverStep.prepare_inputs is scale-free, ``log_gradient_rms`` included.

        The step evaluates the objective and the fusion in cell units, so the
        state scalar ``log_gradient_rms`` is the log RMS of the normalised
        axis gradient and coincides between the two spaces, as do frames,
        local axes, the five matrix blocks, boundary flags, edge features,
        contact tokens and all conditioning channels. ``axis_gradient_world``
        is reported in cell units and coincides too; ``position_gradient`` is
        reported in newtons and scales with ``mu h^2`` like the force
        residual. The test also checks the float32 mixed-step objective
        term by term at a common state, ``step.energy(X, ...) = mu h^3
        step'.energy(X / h, ...)``, which the trajectory test only implies at
        two independently converged minimisers.
        """
        for target_modes in (3, 7):
            with self.subTest(target_modes=target_modes):
                self._check_network_inputs_are_invariant(target_modes)

    def _check_network_inputs_are_invariant(self, target_modes: int):
        scene = _scene(1e4, 0.3, 1000.0, 10.0)
        h, dt = scene.cell_size, scene.time_step
        width = features.target_dim(target_modes)
        physical, normalised = self._mixed_steps(scene, contact_tokens=True, target_modes=target_modes)
        pinned, _ = self._free_corners()
        positions, velocity = self._initial_state(pinned)
        payload = physical.prepare("beam", positions.float(), velocity.float())
        unit_payload = normalised.prepare("beam", (positions / h).float(), (velocity * dt / h).float())
        self.assertGreater(len(_pairs(payload)), 0, "contact is active")
        self.assertEqual(_pairs(payload), _pairs(unit_payload))
        batch = _batch([payload], torch.device("cpu"))
        unit_batch = _batch([unit_payload], torch.device("cpu"))

        def inputs(step, collated):
            return step.prepare_inputs(
                collated["candidate"],
                collated["inertial_prediction"],
                collated["context_ids"],
                previous_positions=collated["physical_positions"],
                contact=collated["contact"],
            )

        physical_inputs, unit_inputs = inputs(physical, batch), inputs(normalised, unit_batch)
        self.assertEqual(physical_inputs.local_axes.shape[-1], target_modes)
        self.assertEqual(physical_inputs.state_features.shape[-1], features.state_feature_dim(target_modes))
        deviation = {
            "frames": _deviation(physical_inputs.frames, unit_inputs.frames),
            "local_axes": _deviation(physical_inputs.local_axes, unit_inputs.local_axes),
            "position_gradient": _deviation(
                physical_inputs.position_gradient, scene.force_scale * unit_inputs.position_gradient
            ),
            "axis_gradient_world": _deviation(physical_inputs.axis_gradient_world, unit_inputs.axis_gradient_world),
        }
        state, unit_state = physical_inputs.state_features, unit_inputs.state_features
        # The matrix blocks are dimensionless with natural scale one (F differences, RMS-normalised gradients), so
        # they are compared absolutely, as the trajectory test compares F.
        for block, name in enumerate(features.MATRIX_BLOCKS):
            columns = slice(width * block, width * block + width)
            deviation[name] = (state[..., columns] - unit_state[..., columns]).abs().max().item()
        matrix_width = len(features.MATRIX_BLOCKS) * width
        boundary = slice(matrix_width, matrix_width + features.BOUNDARY_DIM)
        self.assertTrue(torch.equal(state[..., boundary], unit_state[..., boundary]), "boundary flags")
        history = _state_column("history_valid", target_modes)
        self.assertTrue(torch.equal(state[..., history], unit_state[..., history]), "history flag")
        for hop, edges in physical_inputs.edge_features.items():
            deviation[f"edge_features_hop{hop}"] = _deviation(edges, unit_inputs.edge_features[hop])
        self.assertTrue(torch.equal(physical_inputs.contact_mask, unit_inputs.contact_mask), "contact token mask")
        self.assertGreater(physical_inputs.contact_mask.sum().item(), 0, "contact tokens are present")
        tokens = torch.where(physical_inputs.contact_mask[..., None], physical_inputs.contact_tokens, 0.0)
        unit_tokens = torch.where(unit_inputs.contact_mask[..., None], unit_inputs.contact_tokens, 0.0)
        deviation["contact_tokens"] = _deviation(tokens, unit_tokens)
        # The gradient block is non-trivial, so its agreement is a statement about the RMS normalisation.
        gradient_block = width * features.MATRIX_BLOCKS.index("current_axis_gradient")
        self.assertGreater(unit_state[..., gradient_block : gradient_block + width].abs().max().item(), 1.0)
        print(f"network inputs ({target_modes} modes), max deviation (relative, matrix blocks absolute): {deviation}")
        for name, value in deviation.items():
            self.assertLessEqual(value, FLOAT32_RTOL, name)

        # The former hidden absolute scale: the log RMS is now dimensionless and coincides. Its value is far from
        # log(mu h^3) apart, so the agreement is not accidental.
        column = _state_column("log_gradient_rms", target_modes)
        log_rms, unit_log_rms = state[..., column], unit_state[..., column]
        self.assertEqual(log_rms.unique().numel(), 1, "log_gradient_rms is broadcast per object")
        offset = (log_rms[0, 0] - unit_log_rms[0, 0]).item()
        print(
            f"log_gradient_rms: physical {log_rms[0, 0].item():+.6f}, normalised {unit_log_rms[0, 0].item():+.6f}, "
            f"offset {offset:+.2e} (log(mu h^3) = {math.log(scene.energy_scale):+.6f} would be the SI offset)"
        )
        self.assertLessEqual(abs(offset), FLOAT32_RTOL)
        self.assertGreater(abs(math.log(scene.energy_scale)), 1.0, "an SI log RMS would be far off for this scene")
        # Every conditioning channel coincides (the floor test checks the function on several materials).
        torch.testing.assert_close(
            physical_inputs.conditioning[0, 0], unit_inputs.conditioning[0, 0], rtol=0, atol=1e-6
        )

        # Float32 objective of the mixed step at a common state (the rigid candidate plus 2 % h smooth noise).
        common = batch["candidate"] + _smooth_field(self.rest, self.generator, amplitude=0.02 * h, dtype=torch.float32)
        terms = physical.energy(
            common,
            batch["inertial_prediction"],
            batch["context_ids"],
            previous_positions=batch["physical_positions"],
            contact=batch["contact"],
        )
        unit_terms = normalised.energy(
            common / h,
            batch["inertial_prediction"] / h,
            unit_batch["context_ids"],
            previous_positions=batch["physical_positions"] / h,
            contact=unit_batch["contact"],
        )
        common_state = {}
        for name in ("total", "elastic", "inertia", "damping", "contact"):
            self.assertGreater(getattr(terms, name).item(), 0.0, name)
            common_state[name] = _deviation(getattr(terms, name), scene.energy_scale * getattr(unit_terms, name))
        print(f"mixed-step float32 energy at a common state, max relative deviation: {common_state}")
        for name, value in common_state.items():
            self.assertLessEqual(value, FLOAT32_RTOL, name)

    def test_plain_optimizer_trajectory_is_identical_up_to_h(self):
        """Three implicit-Euler steps solved by the same plain L-BFGS in both spaces agree up to h, h / dt and in F.

        Contact is active throughout (floor 0.3 h under the bottom face, ke =
        E h), damping is positive and gravity acts. The rigid candidate, the
        inertial prediction and the frozen contact pairs of every step are
        compared as well. The mixed step is float32-only, hence the 1e-4
        tolerance relative to h (lengths), h / dt (velocities) and 1 (F); with
        the two-phase minimiser of :func:`_minimise` the observed deviations
        are about 1e-6 to 1e-5, and without the polish phase they reach the
        1e-4 float32 line-search stall. Only the minimiser is shared between
        the spaces: the L-BFGS iterates are not scale-covariant (module
        docstring) and are not compared. The seven-mode step shares the
        unit-grid factor and fuses a seven-mode zero increment in ``prepare``,
        so the same identities hold for it.
        """
        for target_modes in (3, 7):
            with self.subTest(target_modes=target_modes):
                self._check_plain_optimizer_trajectory(target_modes)

    def _check_plain_optimizer_trajectory(self, target_modes: int):
        scene = _scene(1e4, 0.3, 1000.0, 10.0)
        h, dt = scene.cell_size, scene.time_step
        physical, normalised = self._mixed_steps(scene, target_modes=target_modes)
        pinned, free = self._free_corners()
        positions, velocity = self._initial_state(pinned)
        payload = physical.prepare("beam", positions.float(), velocity.float())
        unit_payload = normalised.prepare("beam", (positions / h).float(), (velocity * dt / h).float())
        for step_index in range(3):
            self.assertGreater(len(_pairs(payload)), 0, "contact is active")
            self.assertEqual(_pairs(payload), _pairs(unit_payload))
            deviation = {
                "pairs_point": _deviation(payload["contact_partner_point"], h * unit_payload["contact_partner_point"]),
                "candidate": _deviation(payload["candidate"], h * unit_payload["candidate"]),
                "inertial_prediction": _deviation(
                    payload["inertial_prediction"], h * unit_payload["inertial_prediction"]
                ),
            }
            solved, batch, residual, energies = _minimise(physical, payload, free, iterations=200, polish=50)
            unit_solved, unit_batch, unit_residual, unit_energies = _minimise(
                normalised, unit_payload, free, iterations=200, polish=50
            )
            for label, (wolfe, polished) in (("physical", energies), ("normalised", unit_energies)):
                self.assertLessEqual(polished, wolfe * (1 + 1e-6), f"{label} polish phase did not increase the energy")
            self.assertLess(residual, 1e-4 * scene.force_scale, "physical minimisation converged")
            self.assertLess(unit_residual, 1e-4, "normalised minimisation converged")
            terms = physical.energy(
                solved[None],
                batch["inertial_prediction"],
                batch["context_ids"],
                previous_positions=batch["physical_positions"],
                contact=batch["contact"],
            )
            unit_terms = normalised.energy(
                unit_solved[None],
                unit_batch["inertial_prediction"],
                unit_batch["context_ids"],
                previous_positions=unit_batch["physical_positions"],
                contact=unit_batch["contact"],
            )
            self.assertGreater(terms.contact.item(), 0.0, "contact energy is active at the solution")
            deviation["positions_over_h"] = (solved.double() - h * unit_solved.double()).abs().max().item() / h
            deviation["energy"] = _deviation(terms.total, scene.energy_scale * unit_terms.total)
            deformation = features.center_deformation(
                solved[None], physical.cell_corner_indices, physical.center_gradients
            )
            unit_deformation = features.center_deformation(
                unit_solved[None], normalised.cell_corner_indices, normalised.center_gradients
            )
            deviation["deformation"] = (deformation.double() - unit_deformation.double()).abs().max().item()
            payload = dict(payload, candidate=solved)
            unit_payload = dict(unit_payload, candidate=unit_solved)
            payload = physical.advance(payload)
            unit_payload = normalised.advance(unit_payload)
            deviation["velocity_over_h_dt"] = (
                payload["velocities"].double() - scene.velocity_scale * unit_payload["velocities"].double()
            ).abs().max().item() / scene.velocity_scale
            print(
                f"trajectory step {step_index} ({target_modes} modes): max deviations {deviation}; "
                f"final |grad|_max / (mu h^2) = {residual / scene.force_scale:.2e} physical, "
                f"{unit_residual:.2e} normalised"
            )
            # Step 0 starts from inputs that are identical up to h; later steps inherit the float32 convergence
            # error of the previous minimisation through the committed positions and velocities.
            inherited = 1e-5 if step_index == 0 else FLOAT32_RTOL
            for name in ("pairs_point", "candidate", "inertial_prediction"):
                self.assertLessEqual(deviation[name], inherited, f"step {step_index} {name}")
            for name in ("positions_over_h", "deformation", "velocity_over_h_dt", "energy"):
                self.assertLessEqual(deviation[name], FLOAT32_RTOL, f"step {step_index} {name}")

    @staticmethod
    def _partial_loss(scene: Scene, unit_rest) -> HexImplicitEulerLoss:
        """Return the float64 loss of the partially normalised copy: h = mu = 1 with rho, eta and dt in SI."""
        partial = scene.partially_normalised()
        return HexImplicitEulerLoss(
            unit_rest,
            partial.lame_lambda,
            partial.lame_mu,
            partial.density,
            partial.time_step,
            damping=partial.damping,
            dtype=torch.float64,
        )

    def test_normalising_only_h_and_mu_breaks_the_identity(self):
        """Leaving rho, g, eta and dt in SI while setting h = mu = 1 keeps the elastic term but not the total.

        The total-energy deviation saturates at 1 for every state (the
        mis-scaled inertia exceeds the true energy by a factor of order mu /
        h^2), so a second part states the note's claim literally: one
        implicit-Euler step solved in float64 by the same L-BFGS gives a
        different beam, ``F`` differing by more than 1e-2, whereas the fully
        normalised copy reproduces ``F`` to 1e-6.
        """
        deviations = []
        elastic_worst = 0.0
        for scene in _all_scenes():
            partial = scene.partially_normalised()
            physical, _ = self._losses(scene, self.rest, self.unit_rest)
            broken = self._partial_loss(scene, self.unit_rest)
            h = scene.cell_size
            previous, current, velocity = self._states()
            gravity = torch.tensor(scene.gravity, dtype=torch.float64)
            prediction = make_inertial_prediction(previous, velocity, scene.time_step, explicit_acceleration=gravity)
            # Lengths are divided by h; time, density, gravity and viscosity keep their SI values.
            partial_prediction = make_inertial_prediction(
                previous / h, velocity / h, partial.time_step, explicit_acceleration=gravity
            )
            terms = physical(current, prediction, previous_positions=previous)
            partial_terms = broken(current / h, partial_prediction, previous_positions=previous / h)
            deviations.append(_deviation(terms.total, scene.energy_scale * partial_terms.total))
            elastic_worst = max(elastic_worst, _deviation(terms.elastic, scene.energy_scale * partial_terms.elastic))
        print(
            f"partial normalisation: total-energy relative deviation min {min(deviations):.3e}, max {max(deviations):.3e}"
        )
        self.assertGreater(max(deviations), 0.1)
        self.assertGreater(sum(value > 0.1 for value in deviations), len(deviations) // 2)
        self.assertLessEqual(elastic_worst, FLOAT64_RTOL, "the elastic term alone still scales with mu h^3")

        # One float64 implicit-Euler step (no contact): the fully normalised minimiser has the same F, the
        # partially normalised one a different beam.
        pinned, free = self._free_corners()
        cells = torch.tensor(self.rest.cell_corner_indices, dtype=torch.long)
        signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float64)
        worst = {"positions_over_h": 0.0, "deformation": 0.0}
        broken_worst = {"positions_over_h": math.inf, "deformation": math.inf}
        for scene in (_scene(1e4, 0.3, 1000.0, 10.0), _scene(1e3, 0.3, 100.0, 0.0), _scene(1e6, 0.45, 5000.0, 300.0)):
            unit, partial = scene.normalised(), scene.partially_normalised()
            h, dt = scene.cell_size, scene.time_step
            physical, normalised = self._losses(scene, self.rest, self.unit_rest)
            broken = self._partial_loss(scene, self.unit_rest)
            positions, velocity = self._initial_state(pinned)
            positions, velocity = positions[None], velocity[None]
            gravity = torch.tensor(scene.gravity, dtype=torch.float64)
            prediction = make_inertial_prediction(positions, velocity, dt, explicit_acceleration=gravity)
            unit_prediction = make_inertial_prediction(
                positions / h,
                velocity * dt / h,
                unit.time_step,
                explicit_acceleration=torch.tensor(unit.gravity, dtype=torch.float64),
            )
            partial_prediction = make_inertial_prediction(
                positions / h, velocity / h, partial.time_step, explicit_acceleration=gravity
            )
            solved, residual = _minimise_float64(physical, prediction, positions, free)
            unit_solved, unit_residual = _minimise_float64(normalised, unit_prediction, positions / h, free)
            partial_solved, partial_residual = _minimise_float64(broken, partial_prediction, positions / h, free)
            self.assertLess(residual / scene.force_scale, 1e-5)
            self.assertLess(unit_residual, 1e-5)
            self.assertLess(partial_residual / partial.force_scale, 0.1, "the broken problem converged as well")
            deformation = features.center_deformation(solved, cells, signs / (4 * h))
            unit_deformation = features.center_deformation(unit_solved, cells, signs / 4)
            partial_deformation = features.center_deformation(partial_solved, cells, signs / 4)
            self.assertGreater((deformation - torch.eye(3, dtype=torch.float64)).abs().max().item(), 1e-2, "strain")
            worst["positions_over_h"] = max(
                worst["positions_over_h"], (solved - h * unit_solved).abs().max().item() / h
            )
            worst["deformation"] = max(worst["deformation"], (deformation - unit_deformation).abs().max().item())
            broken_worst["positions_over_h"] = min(
                broken_worst["positions_over_h"], (solved - h * partial_solved).abs().max().item() / h
            )
            broken_worst["deformation"] = min(
                broken_worst["deformation"], (deformation - partial_deformation).abs().max().item()
            )
        print(
            f"float64 one-step minimiser: max deviation {worst} fully normalised, min deviation {broken_worst} partial"
        )
        for name, value in worst.items():
            self.assertLessEqual(value, 1e-6, name)
        for name, value in broken_worst.items():
            self.assertGreater(value, 1e-2, name)


if __name__ == "__main__":
    unittest.main()
