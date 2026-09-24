# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental Newton solver interface for learned hexahedral updates.

The native Model/State own physical particles. A separate scratch Newton rigid
body supplies a frozen pose proposal using Newton's public integrator. Torch
retains network/fusion gradients within this step; committing Warp State arrays
is a detached boundary, so this module does not differentiate trajectories
through Newton states or through the rigid predictor. Native setup is CPU
float32; learned geometry, network, and loss run on the network's CPU or CUDA
device, with the fixed sparse factorization and adjoint solve on CPU.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase

from .data import generate_cuboid
from .hex_energy import HexLossTerms, make_inertial_prediction
from .network import IntrinsicSolverNetwork
from .rigid_predictor import RigidPosePredictor, RigidPrediction
from .solver_step import LearnedHexSolverStep

if TYPE_CHECKING:
    import torch

__all__ = ["LearnedHexProblem", "LearnedNewtonStepResult", "LearnedOptimizerUpdate", "SolverLearnedIntrinsic"]


class LearnedHexProblem(NamedTuple):
    """Experimental fixed objective for all optimizer queries in one physical step.

    previous_positions and inertial_prediction are [1,P,3] [m]. Prescribed
    positions are [1,K,3] [m], indexed by fixed_indices [K]. The frozen rigid
    prediction only initializes fusion. time_step is [s]. The optimizer owns
    this problem's material, mass, dt, and fusion context while sharing the
    trainable network. Preparing another problem does not change this one.

    Treat the captured tensors and optimizer buffers as read-only snapshots.
    """

    previous_positions: torch.Tensor
    inertial_prediction: torch.Tensor
    fixed_positions: torch.Tensor
    fixed_indices: torch.Tensor
    rigid_prediction: RigidPrediction
    time_step: float
    optimizer: LearnedHexSolverStep

    def objective(self, positions: torch.Tensor) -> HexLossTerms:
        """Evaluate implicit-Euler energy [J] at a candidate [1,P,3] [m]."""
        return self.optimizer.energy(positions, self.inertial_prediction, previous_positions=self.previous_positions)


class LearnedOptimizerUpdate(NamedTuple):
    """Experimental single optimizer proposal against a fixed physical objective.

    current_positions, positions, and direction are [1,P,3] [m], with
    direction = positions - current_positions and zero direction at pins.
    Local target axes and corrections are dimensionless [1,C,3,3]; frames
    are frozen [1,C,3,3] rotations. step_size is dimensionless [1]. loss
    is the post-update energy [J]. An untrained proposal need not descend.
    """

    current_positions: torch.Tensor
    positions: torch.Tensor
    direction: torch.Tensor
    local_target_axes: torch.Tensor
    axis_correction: torch.Tensor
    step_size: torch.Tensor
    frames: torch.Tensor
    loss: HexLossTerms


class LearnedNewtonStepResult(NamedTuple):
    """Experimental differentiable result and diagnostics from one physical step.

    Positions and inertial prediction are [m], velocities [m/s], all [1,P,3].
    The rigid prediction is frozen. Each entry in updates is an inner optimizer
    iterate using the same physical Y; the final entry supplies the loss [J].
    """

    positions: torch.Tensor
    velocities: torch.Tensor
    inertial_prediction: torch.Tensor
    rigid_prediction: RigidPrediction
    updates: tuple[LearnedOptimizerUpdate, ...]
    initial_positions: torch.Tensor

    @property
    def loss(self) -> HexLossTerms:
        """Return the final candidate's unchanged implicit-Euler energy [J]."""
        return self.updates[-1].loss


class SolverLearnedIntrinsic(SolverBase):
    """Advance a native Newton hex particle model using learned fusion proposals.

    Experimental: accepts the single-object CPU model produced by
    newton_model.build_newton_hex_model. The model owns canonical geometry,
    materials, positive physical masses, and stationary boundary flags. Its
    states own the current shared corners, velocities, and external forces.
    Forces must exclude gravity, which is read from model.gravity.

    The rigid target and physical Y are each computed once from state_in.
    Rigid-guided fusion initializes a candidate once; repeated learned queries
    then refine it against the same physical objective. Each query can also
    accept an arbitrary feasible candidate for that prepared problem.
    Prescribed corners override the rigid proposal. The predictor does not
    solve boundary reactions or contacts. The clamped fusion currently supports
    no free-body gauge, and populated contacts are rejected.

    step() writes detached final corners/velocities to a separate Newton State.
    Its last_result retains the Torch graph for training network parameters.
    This is a learned proposal, not a converged implicit physical solve: no
    line search, convergence test, contact response, or inversion repair is
    supplied. Use a trained network before interpreting rollouts physically.

    Args:
        model: CPU float32 Newton model with learned_intrinsic hex attributes.
        network: Optional existing baseline-compatible Torch network.
        iterations: Positive number of learned optimization iterations per dt.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        network: IntrinsicSolverNetwork | None = None,
        iterations: int = 5,
    ):
        super().__init__(model)
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self.iterations = iterations
        self.network = network
        self.last_result: LearnedNewtonStepResult | None = None
        self.learned_step: LearnedHexSolverStep | None = None
        self._time_step: float | None = None
        self._configure_model()

    def _configure_model(self) -> None:
        import torch

        model = self.model
        if not model.device.is_cpu:
            raise ValueError("SolverLearnedIntrinsic currently requires a CPU model")
        if any(
            (
                model.body_count,
                model.joint_count,
                model.tet_count,
                model.tri_count,
                model.shape_count,
                model.spring_count,
                model.edge_count,
                model.muscle_count,
            )
        ):
            raise ValueError("the current learned solver supports only its single hex particle object")
        if not hasattr(model, "learned_intrinsic"):
            raise ValueError("model must be created with build_newton_hex_model")
        metadata = model.learned_intrinsic
        counts = tuple(int(value) for value in metadata.cell_counts.numpy()[0])
        h = float(metadata.cell_size.numpy()[0])
        stored_rest = metadata.rest_positions.numpy()
        self._rest = generate_cuboid(counts, cell_size=h, origin=tuple(float(v) for v in stored_rest[0]))
        expected = self._rest.corner_rest_positions.astype(np.float32)
        tolerance = 8 * np.finfo(np.float32).eps * max(h, float(np.max(np.abs(expected))))
        if stored_rest.shape != expected.shape or not np.allclose(stored_rest, expected, rtol=0, atol=tolerance):
            raise ValueError("model rest positions must match the canonical cuboid metadata")
        if not np.array_equal(metadata.cell_corner_indices.numpy(), self._rest.cell_corner_indices):
            raise ValueError("model cell_corner_indices must use the canonical z-fast ordering")
        if model.particle_count != len(expected):
            raise ValueError("model particle count must match the hex corners")
        fixed = metadata.fixed.numpy()
        if fixed.shape != (model.particle_count,):
            raise ValueError("fixed metadata must have one flag per corner")
        self._fixed = np.flatnonzero(fixed)
        if not len(self._fixed):
            raise ValueError("at least one fixed corner is required by the current fusion layer")
        active = (model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        if not np.array_equal(active, ~fixed):
            raise ValueError("Newton ACTIVE flags and learned fixed flags must agree")
        worlds = np.unique(model.particle_world.numpy())
        if len(worlds) != 1:
            raise ValueError("the current solver supports one particle world")
        self._world = int(worlds[0])
        self._lame_lambda = metadata.lame_lambda.numpy().copy()
        self._lame_mu = metadata.lame_mu.numpy().copy()
        self._density = metadata.density.numpy().copy()
        self._damping = (
            metadata.damping.numpy().copy() if hasattr(metadata, "damping") else np.zeros_like(self._lame_mu)
        )
        if (
            self._damping.shape != self._lame_mu.shape
            or not np.isfinite(self._damping).all()
            or (self._damping < 0).any()
        ):
            raise ValueError("model damping must be finite and nonnegative with one value per hex")
        self._mass = torch.from_numpy(model.particle_mass.numpy().copy())
        if self._mass.dtype != torch.float32 or not torch.isfinite(self._mass).all() or (self._mass <= 0).any():
            raise ValueError("all physical corner masses must remain positive float32, including pins")
        self._fixed_tensor = torch.from_numpy(self._fixed)
        self.rigid_predictor = RigidPosePredictor(self._mass, gravity=tuple(self._gravity().tolist()))
        if self.network is None:
            has_damping = bool((self._damping > 0).any())
            self.network = IntrinsicSolverNetwork(
                counts, 86 if has_damping else 38, conditioning_dim=6 if has_damping else 5
            )
        if self.network.cell_counts != counts:
            raise ValueError("network cell counts must match the model")
        self.learned_step = None
        self._time_step = None
        self.last_result = None

    def _gravity(self) -> np.ndarray:
        gravity = self.model.gravity.numpy()[self._world].copy()
        if not np.isfinite(gravity).all():
            raise ValueError("model gravity must be finite")
        return gravity

    def _step_for_dt(self, dt: float) -> LearnedHexSolverStep:
        import torch

        device = next(self.network.parameters()).device
        if self.learned_step is None or dt != self._time_step or self.learned_step.rest_positions.device != device:
            step = LearnedHexSolverStep(
                self._rest,
                self._fixed,
                lame_lambda=self._lame_lambda,
                lame_mu=self._lame_mu,
                density=self._density,
                damping=self._damping,
                time_step=dt,
                network=self.network,
            )
            mass = self._mass.to(device)
            if not torch.allclose(step.energy.lumped_mass, mass, rtol=2e-5, atol=0):
                raise ValueError("Newton particle masses must match the hex density and rest volume")
            # Native Model masses are authoritative; metadata round-trips can
            # round h differently by one float32 ulp.
            step.energy.lumped_mass.copy_(mass)
            self.learned_step = step
            self._time_step = dt
        return self.learned_step

    def _check_state(self, state: newton.State) -> None:
        for name in ("particle_q", "particle_qd", "particle_f"):
            value = getattr(state, name, None)
            if (
                value is None
                or value.shape != (self.model.particle_count,)
                or value.device != self.model.device
                or value.dtype != wp.vec3
            ):
                raise ValueError(f"state.{name} must match the model's CPU float32 corner array")
        if state.body_count:
            raise ValueError("rigid bodies in the physical state are not supported by this prototype")

    @staticmethod
    def _check_contacts(contacts: newton.Contacts | None) -> None:
        if contacts is not None:
            for name in ("rigid_contact_count", "soft_contact_count"):
                count = getattr(contacts, name, None)
                if count is not None and np.any(count.numpy()):
                    raise NotImplementedError("contact response is not implemented in the learned hex solver")

    def prepare_problem(
        self,
        state_in: newton.State,
        dt: float,
        *,
        control: newton.Control | None = None,
        contacts: newton.Contacts | None = None,
    ) -> LearnedHexProblem:
        """Snapshot one physical timestep objective without evaluating the network.

        Args:
            state_in: Current native corner positions, velocities, external forces.
            dt: Positive physical time step [s].
            control: Standard Newton Control; this particle-only model has no
                actuators. Applied corner forces belong in state_in.particle_f.
            contacts: None or an empty contact buffer. Nonempty contacts fail.

        Returns:
            Frozen physical inputs and the optimizer context for this dt. Its
            objective always uses the original implicit-Euler Y, independently
            of the supplied optimizer candidate or rigid initialization.
        """
        import torch

        self.last_result = None
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        dt = float(dt)
        self._check_state(state_in)
        self._check_contacts(contacts)
        if control is not None and not isinstance(control, newton.Control):
            raise TypeError("control must be a Newton Control or None")
        step = self._step_for_dt(dt)
        # Snapshot: wp.to_torch aliases Warp storage, which a later simulation
        # step may overwrite before this loss has been differentiated.
        x = wp.to_torch(state_in.particle_q).detach().clone()
        velocity = wp.to_torch(state_in.particle_qd).detach().clone()
        forces = wp.to_torch(state_in.particle_f).detach().clone()
        gravity = torch.from_numpy(self._gravity())
        self.rigid_predictor.model.set_gravity(tuple(gravity.tolist()))
        rigid = self.rigid_predictor.predict(x, velocity, forces, dt)
        device = step.rest_positions.device
        x, velocity, forces, gravity = (value.to(device) for value in (x, velocity, forces, gravity))
        rigid = RigidPrediction(*(value.to(device) for value in rigid))
        inertial = make_inertial_prediction(
            x[None], velocity[None], dt, explicit_acceleration=gravity + forces / step.energy.lumped_mass[:, None]
        )
        return LearnedHexProblem(
            x[None],
            inertial,
            x[step.fixed_indices][None],
            step.fixed_indices.clone(),
            rigid,
            dt,
            step,
        )

    def _check_problem(self, problem: LearnedHexProblem) -> None:
        if not isinstance(problem, LearnedHexProblem) or problem.optimizer.network is not self.network:
            raise ValueError("problem must use this solver's network")

    def initialize_candidate(self, problem: LearnedHexProblem) -> torch.Tensor:
        """Fuse the one-time rigid target with prescribed corners, returning [1,P,3] [m].

        This initialization has no learned correction and does not replace Y.
        Call only once per solve; propose_update never applies the rigid map.
        """
        self._check_problem(problem)
        rigid = problem.rigid_prediction
        base = (
            problem.previous_positions @ rigid.rigid_delta_rotation.transpose(-1, -2)
            + rigid.rigid_delta_translation[:, None, :]
        )
        count = len(problem.optimizer.energy.cell_corner_indices)
        zero_increment = base.new_zeros((1, count, 3, 3))
        return problem.optimizer.fusion.fuse(base, zero_increment, problem.fixed_positions)

    def propose_update(
        self,
        candidate: torch.Tensor,
        problem: LearnedHexProblem,
        *,
        frames: torch.Tensor | None = None,
    ) -> LearnedOptimizerUpdate:
        """Query the shared learned optimizer at any feasible current candidate.

        candidate is [1,P,3] float32 [m] on the network device, with prescribed corners already
        satisfied exactly. Geometry/features are recomputed from this candidate;
        optional frames [1,C,3,3] replay frozen rotations for derivative checks.
        Returns the actual fused displacement, without advancing physical time
        or altering this problem. No descent or accepted-step guarantee exists.
        """
        import torch

        self._check_problem(problem)
        reference = problem.previous_positions
        if (
            not isinstance(candidate, torch.Tensor)
            or candidate.shape != reference.shape
            or candidate.dtype != reference.dtype
            or candidate.device != reference.device
            or not torch.isfinite(candidate).all()
        ):
            raise ValueError("candidate must be finite float32 [1,P,3] matching the problem device")
        if not torch.equal(candidate[:, problem.fixed_indices], problem.fixed_positions):
            raise ValueError("candidate must satisfy the problem's prescribed corners exactly")
        update = problem.optimizer(
            candidate,
            problem.inertial_prediction,
            previous_positions=problem.previous_positions,
            fixed_positions=problem.fixed_positions,
            frames=frames,
        )
        return LearnedOptimizerUpdate(
            candidate,
            update.positions,
            update.positions - candidate,
            update.local_target_axes,
            update.axis_correction,
            update.step_size,
            update.frames,
            update.loss,
        )

    def solve(
        self,
        problem: LearnedHexProblem,
        *,
        initial_positions: torch.Tensor | None = None,
        iterations: int | None = None,
    ) -> LearnedNewtonStepResult:
        """Unroll repeated optimizer queries with gradients through every update.

        initial_positions optionally supplies any feasible [1,P,3] candidate
        [m]; otherwise rigid-guided fusion initializes it. iterations overrides
        the configured positive count for this solve. The same network, Y, dt,
        materials, and prescribed positions are used throughout. Candidates
        remain attached to the graph; each iteration's polar frames are frozen.
        """
        import torch

        self.last_result = None
        self._check_problem(problem)
        count = self.iterations if iterations is None else iterations
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("iterations must be a positive integer")
        initial = self.initialize_candidate(problem) if initial_positions is None else initial_positions
        current = initial
        updates = []
        for _ in range(count):
            update = self.propose_update(current, problem)
            current = update.positions
            updates.append(update)
        updated_velocity = (current - problem.previous_positions) / problem.time_step
        free = torch.ones(current.shape[1], dtype=torch.bool, device=current.device)
        free[problem.fixed_indices] = False
        updated_velocity = torch.where(free[None, :, None], updated_velocity, 0)
        result = LearnedNewtonStepResult(
            current,
            updated_velocity,
            problem.inertial_prediction,
            problem.rigid_prediction,
            tuple(updates),
            initial,
        )
        self.last_result = result
        return result

    def predict(
        self,
        state_in: newton.State,
        dt: float,
        *,
        control: newton.Control | None = None,
        contacts: newton.Contacts | None = None,
    ) -> LearnedNewtonStepResult:
        """Prepare and solve one physical step without modifying native State arrays.

        Return all differentiable optimizer updates and retain them in
        last_result. Physical inputs are snapshotted once by prepare_problem.
        """
        return self.solve(self.prepare_problem(state_in, dt, control=control, contacts=contacts))

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Write one final learned step into a separate native Newton State.

        The output force buffer is untouched, following ordinary solver usage;
        clear/populate forces on the next input state before its physical step.
        Network gradients remain available through last_result.loss.total.
        """
        if state_in is state_out:
            raise ValueError("use separate input and output Newton States")
        self._check_state(state_out)
        result = self.predict(state_in, dt, control=control, contacts=contacts)
        state_out.particle_q.assign(result.positions.detach().cpu().numpy()[0])
        state_out.particle_qd.assign(result.velocities.detach().cpu().numpy()[0])

    def notify_model_changed(self, flags: newton.ModelFlags | int) -> None:
        """Reload masses, materials, geometry, and pins while retaining the network.

        Call after editing custom model metadata or masses/flags. Any supplied
        category refreshes this small experimental solver. Gravity is read live
        even without notification. A changed topology must match the network.
        """
        self._configure_model()
