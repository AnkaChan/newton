# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental frozen rigid guidance using Newton's public body integrator."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase

if TYPE_CHECKING:
    import torch

__all__ = ["RigidPosePredictor", "RigidPrediction"]


class RigidPrediction(NamedTuple):
    """Frozen float32 CPU snapshots of one proxy integration.

    Experimental: the first two fields map current world points to proposed
    world points as ``x_new = rotation @ x + translation``. Their shapes are
    [1, 3, 3] and [1, 3] to match the fusion interface. Remaining vector fields
    have shape [3], and matrix fields have shape [3, 3]. Current momentum and
    velocity diagnostics precede the optional impulse application.
    """

    rigid_delta_rotation: torch.Tensor
    """Dimensionless current-to-predicted rotation, shape [1, 3, 3]."""
    rigid_delta_translation: torch.Tensor
    """Current-to-predicted translation [m], shape [1, 3]."""
    center_of_mass: torch.Tensor
    """Current world mass center [m]."""
    center_of_mass_velocity: torch.Tensor
    """Current mass-center velocity [m/s]."""
    angular_momentum: torch.Tensor
    """Current angular momentum about the mass center [kg*m^2/s]."""
    inertia: torch.Tensor
    """Current inertia about the mass center in world coordinates [kg*m^2]."""
    predicted_position: torch.Tensor
    """Predicted world mass center [m]."""
    predicted_rotation: torch.Tensor
    """Dimensionless predicted proxy rotation from its world-aligned input frame."""
    predicted_linear_velocity: torch.Tensor
    """Predicted mass-center velocity [m/s]."""
    predicted_angular_velocity: torch.Tensor
    """Predicted world angular velocity [rad/s]."""


class RigidPosePredictor:
    """Predict fusion guidance with a separate one-body Newton model and states.

    Experimental: CPU float32 only, one object, and no differentiation through
    this predictor. Corner masses remain fixed while each call recomputes the
    center, momentum, and inertia from current deformed geometry. A nonsingular
    inertia tensor is required; collinear or collapsed geometry is rejected.

    Every call resets the scratch body at the current mass center with identity
    orientation and zero body-frame COM. Its initial spin is the instantaneous
    rigid projection of the supplied corner velocities. Newton's public
    ``SolverBase.integrate_bodies`` performs the actual semi-implicit update,
    including its gyroscopic term. The predictor does not advance a physical
    particle state or accumulate an independent proxy trajectory.

    ``model`` is the scratch public Newton Model. Call ``model.set_gravity``
    to update its gravity. ``solver`` is the public SolverBase instance used
    for integration. Optional impulses act once, at the start of the proxy
    interval; they are not also included in its force wrench. This module does
    not detect contact or define restitution/friction laws.

    Args:
        corner_masses: Fixed nonnegative lumped masses [kg], shape [P], with a
            finite positive total. Static array-like values are stored as a
            detached CPU float32 copy.
        gravity: Gravity vector [m/s^2], shape [3], used by Newton exactly once.
    """

    def __init__(self, corner_masses, *, gravity=(0.0, 0.0, -9.81)):
        import torch

        if isinstance(corner_masses, torch.Tensor) and corner_masses.device.type != "cpu":
            raise ValueError("corner_masses must be on the CPU")
        masses = torch.as_tensor(corner_masses, dtype=torch.float32, device="cpu").detach().clone()
        if masses.ndim != 1 or masses.numel() == 0 or not torch.isfinite(masses).all() or (masses < 0).any():
            raise ValueError("corner_masses must be a finite nonnegative vector")
        total_mass = masses.sum()
        if not torch.isfinite(total_mass) or total_mass <= 0:
            raise ValueError("corner_masses must have a finite positive total")
        if isinstance(gravity, torch.Tensor) and gravity.device.type != "cpu":
            raise ValueError("gravity must be on the CPU")
        gravity_tensor = torch.as_tensor(gravity, dtype=torch.float32, device="cpu").detach().clone()
        if gravity_tensor.shape != (3,) or not torch.isfinite(gravity_tensor).all():
            raise ValueError("gravity must be a finite vector of shape [3]")
        self._masses = masses
        self._total_mass = total_mass
        self.corner_count = masses.numel()

        builder = newton.ModelBuilder(gravity=tuple(gravity_tensor.tolist()))
        builder.add_body(
            xform=wp.transform_identity(),
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3, dtype=np.float32)),
            mass=float(total_mass),
            label="learned_intrinsic_rigid_proxy",
            lock_inertia=True,
        )
        self.model = builder.finalize(device="cpu", requires_grad=False)
        self.solver = SolverBase(self.model)
        self._state_in = self.model.state(requires_grad=False)
        self._state_out = self.model.state(requires_grad=False)

    @property
    def corner_masses(self) -> torch.Tensor:
        """Return a detached copy of the fixed physical masses [kg], shape [P]."""
        return self._masses.clone()

    @staticmethod
    def _check_tensor(name, tensor, shape):
        import torch

        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be a float32 Torch tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on the CPU")
        if tensor.shape != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite with shape {list(shape)}")

    def predict(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        dt: float,
        *,
        linear_impulse: torch.Tensor | None = None,
        angular_impulse: torch.Tensor | None = None,
    ) -> RigidPrediction:
        """Return rigid pose guidance without mutating the physical inputs.

        Args:
            positions: Current world corner positions [m], shape [P, 3].
            velocities: Current world corner velocities [m/s], shape [P, 3].
            forces: External corner forces [N], shape [P, 3]. Exclude gravity
                already supplied to the model and the separately supplied
                impulses. Forces are summed into a force/torque wrench.
            dt: Positive finite predictor interval [s].
            linear_impulse: Optional net world impulse [kg*m/s], shape [3].
            angular_impulse: Optional angular impulse about the current mass
                center [kg*m^2/s], shape [3].

        Returns:
            Detached CPU float32 tensors describing the rigid map and proxy
            diagnostics. Earlier returned snapshots remain independent of
            subsequent calls.

        Raises:
            TypeError: If dynamic input tensors are not float32.
            ValueError: If shapes, devices, values, timestep, or inertia are invalid.
        """
        import torch

        for name, tensor in (("positions", positions), ("velocities", velocities), ("forces", forces)):
            self._check_tensor(name, tensor, (self.corner_count, 3))
        if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number")
        for name, tensor in (("linear_impulse", linear_impulse), ("angular_impulse", angular_impulse)):
            if tensor is not None:
                self._check_tensor(name, tensor, (3,))

        with torch.no_grad():
            center = (self._masses[:, None] * positions).sum(dim=0) / self._total_mass
            center_velocity = (self._masses[:, None] * velocities).sum(dim=0) / self._total_mass
            relative = positions - center
            momentum = self._masses[:, None] * (velocities - center_velocity)
            angular_momentum = torch.linalg.cross(relative, momentum).sum(dim=0)

            # Sum positive diagonal terms directly to avoid trace subtraction.
            x, y, z = relative.unbind(dim=1)
            xx = (self._masses * (y.square() + z.square())).sum()
            yy = (self._masses * (x.square() + z.square())).sum()
            zz = (self._masses * (x.square() + y.square())).sum()
            xy = -(self._masses * x * y).sum()
            xz = -(self._masses * x * z).sum()
            yz = -(self._masses * y * z).sum()
            inertia = torch.stack((torch.stack((xx, xy, xz)), torch.stack((xy, yy, yz)), torch.stack((xz, yz, zz))))
            cholesky, status = torch.linalg.cholesky_ex(inertia)
            if int(status) != 0 or not torch.isfinite(inertia).all():
                raise ValueError("current mass distribution must have a nonsingular positive-definite inertia")
            inverse_inertia = torch.cholesky_inverse(cholesky)
            initial_velocity = center_velocity.clone()
            initial_angular_momentum = angular_momentum.clone()
            if linear_impulse is not None:
                initial_velocity += linear_impulse / self._total_mass
            if angular_impulse is not None:
                initial_angular_momentum += angular_impulse
            initial_omega = inverse_inertia @ initial_angular_momentum
            net_force = forces.sum(dim=0)
            net_torque = torch.linalg.cross(relative, forces).sum(dim=0)

            self.model.body_com.assign(np.zeros((1, 3), dtype=np.float32))
            self.model.body_inertia.assign(inertia.numpy()[None])
            self.model.body_inv_inertia.assign(inverse_inertia.numpy()[None])
            pose = np.zeros((1, 7), dtype=np.float32)
            pose[0, :3] = center.numpy()
            pose[0, 6] = 1.0
            self._state_in.body_q.assign(pose)
            self._state_in.body_qd.assign(torch.cat((initial_velocity, initial_omega))[None].numpy())
            self._state_in.body_f.assign(torch.cat((net_force, net_torque))[None].numpy())
            self.solver.integrate_bodies(self.model, self._state_in, self._state_out, float(dt), angular_damping=0.0)

            predicted_pose = self._state_out.body_q.numpy()[0].copy()
            predicted_twist = self._state_out.body_qd.numpy()[0].copy()
            rotation_array = (
                np.asarray(wp.quat_to_matrix(wp.quat(*predicted_pose[3:].tolist())), dtype=np.float32)
                .reshape(3, 3)
                .copy()
            )
            rotation = torch.from_numpy(rotation_array)
            predicted_position = torch.from_numpy(predicted_pose[:3])
            translation = predicted_position - rotation @ center
            return RigidPrediction(
                rigid_delta_rotation=rotation[None],
                rigid_delta_translation=translation[None],
                center_of_mass=center,
                center_of_mass_velocity=center_velocity,
                angular_momentum=angular_momentum,
                inertia=inertia,
                predicted_position=predicted_position,
                predicted_rotation=rotation,
                predicted_linear_velocity=torch.from_numpy(predicted_twist[:3]),
                predicted_angular_velocity=torch.from_numpy(predicted_twist[3:]),
            )
