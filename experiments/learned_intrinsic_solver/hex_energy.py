# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental full-quadrature hexahedral implicit-Euler physical objective.

This opt-in PyTorch module may change without compatibility guarantees. It uses
8-node trilinear cubic rest hexahedra and full 2x2x2 Gauss integration, without
tetrahedralization or reduced-integration stabilization. Its elastic law is
Newton VBD's stable Neo-Hookean density with the sampled Lamé parameters mapped
as mu_NH = mu and lambda_NH = lambda + mu. The density is a polynomial in the
deformation gradient, so it stays finite through collapse (J = 0) and inversion
(J < 0); no quadrature point is rejected for its Jacobian sign.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from torch import nn  # noqa: TID253 - Explicit optional module needs the nn.Module base class.

from .damping import damping_metric_difference
from .data import VoxelGridData

if TYPE_CHECKING:
    import torch

__all__ = [
    "HexImplicitEulerLoss",
    "HexLossTerms",
    "HexQuadrature",
    "hex_gauss_quadrature",
    "make_inertial_prediction",
    "stable_neo_hookean_density",
]


class HexQuadrature(NamedTuple):
    """Store NumPy quadrature for a canonical cubic hexahedron.

    Corner and quadrature indices follow x/y/z order with z varying fastest.
    Gradients are material derivatives [1/m], shape [8,8,3]. Weights include
    the rest Jacobian determinant [m^3], shape [8]. Values have shape [8,8];
    points lie in the dimensionless reference cube [-1,1]^3, shape [8,3].
    """

    shape_gradients: np.ndarray
    weights: np.ndarray
    shape_values: np.ndarray
    points: np.ndarray


def hex_gauss_quadrature(cell_size: float, *, dtype: np.dtype = np.float32) -> HexQuadrature:
    """Construct the full eight-point rule for an axis-aligned rest cube.

    Args:
        cell_size: Positive cubic rest edge length [m].
        dtype: NumPy float32 or float64 dtype for all returned arrays.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("quadrature dtype must be float32 or float64")
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be finite and positive")
    scalar = dtype.type
    signs = (2 * np.indices((2, 2, 2)).reshape(3, -1).T - 1).astype(dtype)
    points = signs / np.sqrt(scalar(3))
    factors = scalar(1) + points[:, None, :] * signs[None, :, :]
    shape_values = np.prod(factors, axis=-1, dtype=dtype) / scalar(8)
    gradients = np.empty((8, 8, 3), dtype=dtype)
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        gradients[:, :, axis] = (
            signs[None, :, axis] * np.prod(factors[:, :, other], axis=-1, dtype=dtype) / scalar(4 * cell_size)
        )
    weights = np.full(8, scalar(cell_size) ** 3 / scalar(8), dtype=dtype)
    return HexQuadrature(gradients, weights, shape_values, points)


class HexLossTerms(NamedTuple):
    """Return physical energies [J], each shape [B].

    Experimental. ``damping=None`` supports legacy three-argument construction;
    :class:`HexImplicitEulerLoss` always returns a damping tensor, including zero.
    """

    total: torch.Tensor
    elastic: torch.Tensor
    inertia: torch.Tensor
    damping: torch.Tensor | None = None


def _time_step_tensor(time_step, reference):
    import torch

    result = torch.as_tensor(time_step, dtype=reference.dtype, device=reference.device)
    if result.ndim != 0 or not torch.isfinite(result).item() or result.item() <= 0:
        raise ValueError("time_step must be a finite positive scalar")
    return result


def make_inertial_prediction(
    previous_positions: torch.Tensor,
    previous_velocity: torch.Tensor,
    time_step: float | torch.Tensor,
    *,
    explicit_acceleration: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the unchanged physical prediction Y = Xn + dt Vn + dt^2 a.

    Experimental. This helper has no dependence on a rigid fusion target.
    Explicit acceleration may include gravity and must not also be counted as
    a separate potential in this objective. Gradients through the supplied
    states and acceleration are preserved.

    Args:
        previous_positions: Previous shared corners [m], shape [B,P,3].
        previous_velocity: Previous velocities [m/s], same shape/dtype/device.
        time_step: Positive time interval [s].
        explicit_acceleration: Explicit acceleration [m/s^2], broadcastable to
            the positions, with the same dtype/device. None means zero.
    """
    import torch

    if previous_positions.ndim != 3 or previous_positions.shape[-1] != 3:
        raise ValueError("previous_positions must have shape [B,P,3]")
    if previous_positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("physical geometry must use float32 or float64")
    if previous_velocity.shape != previous_positions.shape:
        raise ValueError("previous_velocity must have the same shape as previous_positions")
    if previous_velocity.dtype != previous_positions.dtype or previous_velocity.device != previous_positions.device:
        raise TypeError("previous_velocity must match the positions dtype and device")
    if not torch.isfinite(previous_positions).all().item() or not torch.isfinite(previous_velocity).all().item():
        raise ValueError("previous positions and velocity must be finite")
    step = _time_step_tensor(time_step, previous_positions)
    prediction = previous_positions + step * previous_velocity
    if explicit_acceleration is not None:
        if (
            explicit_acceleration.dtype != previous_positions.dtype
            or explicit_acceleration.device != previous_positions.device
        ):
            raise TypeError("explicit_acceleration must match the positions dtype and device")
        if not torch.isfinite(explicit_acceleration).all().item():
            raise ValueError("explicit_acceleration must be finite")
        if torch.broadcast_shapes(explicit_acceleration.shape, previous_positions.shape) != previous_positions.shape:
            raise ValueError("explicit_acceleration must broadcast to the positions shape")
        prediction = prediction + step.square() * explicit_acceleration
    return prediction


def stable_neo_hookean_density(deformation: torch.Tensor, lame_mu, lame_lambda) -> torch.Tensor:
    """Return the stable Neo-Hookean density psi(F) [J/m^3] over the leading dims of F.

    Experimental. This is Newton VBD's stable Neo-Hookean law with the sampled
    Lamé parameters mapped as mu_NH = mu and lambda_NH = lambda + mu, written in
    its rest-zero form (psi(I) = 0 and P(I) = 0). With J = det(F):

    psi = mu/2 (||F||_F^2 - 3) - mu (J - 1) + (lambda + mu)/2 (J - 1)^2,
    P = dpsi/dF = mu F + ((lambda + mu)(J - 1) - mu) cof(F).

    The value is evaluated from H = F - I using the exact identities
    ||F||_F^2 - 3 = 2 tr(H) + ||H||_F^2 and J - 1 = tr(H) + s2(H) + s3(H),
    where s2 is the sum of the three principal 2x2 minors of H and s3 = det(H)
    from cofactors. The mu tr(H) terms cancel algebraically, leaving
    psi = mu/2 ||H||_F^2 - mu (s2 + s3) + (lambda + mu)/2 (tr(H) + s2 + s3)^2.
    This is algebraically identical to the naive formula and reduces float32
    cancellation near rest (generated/verification/energy_floor_calibration/
    SUMMARY.md). No inverse or logarithm is used, so the density and its
    autograd stress are finite for every finite F, including J <= 0 and F = 0.
    Nonfinite deformation entries propagate; callers validate their inputs.

    Args:
        deformation: Deformation gradients F, shape [...,3,3], float32/float64.
        lame_mu: Shear modulus [Pa], scalar or tensor broadcastable to the
            leading shape ``deformation.shape[:-2]`` (for example ``[1,C,1]``
            against ``[B,C,Q,3,3]``). Tensors must share F's dtype and device.
        lame_lambda: First Lamé parameter [Pa], broadcast like ``lame_mu``.

    Returns:
        Density psi with shape ``deformation.shape[:-2]``.

    Raises:
        ValueError: F does not end in 3x3 or a material does not broadcast to
            its leading shape.
        TypeError: F is not a float32/float64 tensor, or a material tensor
            differs from F in dtype or device.
    """
    import torch

    if not isinstance(deformation, torch.Tensor) or deformation.dtype not in (torch.float32, torch.float64):
        raise TypeError("deformation must be a float32 or float64 tensor")
    if deformation.ndim < 2 or deformation.shape[-2:] != (3, 3):
        raise ValueError("deformation must have shape [...,3,3]")
    leading = deformation.shape[:-2]
    materials = []
    for value, name in ((lame_mu, "lame_mu"), (lame_lambda, "lame_lambda")):
        if isinstance(value, torch.Tensor):
            if value.dtype != deformation.dtype or value.device != deformation.device:
                raise TypeError(f"{name} must match the deformation dtype and device")
            material = value
        else:
            material = torch.as_tensor(value, dtype=deformation.dtype, device=deformation.device)
        try:
            broadcast = torch.broadcast_shapes(material.shape, leading)
        except RuntimeError as error:
            raise ValueError(f"{name} must broadcast to the leading deformation shape {tuple(leading)}") from error
        if broadcast != leading:
            raise ValueError(f"{name} must broadcast to the leading deformation shape {tuple(leading)}")
        materials.append(material)
    mu, lam = materials
    increment = deformation - torch.eye(3, dtype=deformation.dtype, device=deformation.device)
    h00, h01, h02 = increment[..., 0, 0], increment[..., 0, 1], increment[..., 0, 2]
    h10, h11, h12 = increment[..., 1, 0], increment[..., 1, 1], increment[..., 1, 2]
    h20, h21, h22 = increment[..., 2, 0], increment[..., 2, 1], increment[..., 2, 2]
    trace = h00 + h11 + h22
    # First-row cofactors of H give det(H) without an inverse.
    cofactor_00 = h11 * h22 - h12 * h21
    cofactor_01 = h12 * h20 - h10 * h22
    cofactor_02 = h10 * h21 - h11 * h20
    principal_minors = cofactor_00 + (h00 * h22 - h02 * h20) + (h00 * h11 - h01 * h10)
    determinant = h00 * cofactor_00 + h01 * cofactor_01 + h02 * cofactor_02
    higher_order = principal_minors + determinant
    jacobian_minus_one = trace + higher_order
    squared_norm = increment.square().sum(dim=(-1, -2))
    return 0.5 * mu * squared_norm - mu * higher_order + 0.5 * (lam + mu) * jacobian_minus_one.square()


class HexImplicitEulerLoss(nn.Module):
    """Evaluate physical inertia, elasticity, and VBD metric damping.

    Experimental. This module assumes the canonical cubic rest grid produced
    by generate_cuboid. It registers constant topology, material, quadrature,
    and lumped mass buffers; move the module with .to(device) as needed.
    Inputs and buffers must share dtype and device. No force or material
    parameter is inferred from a rigid fusion target.

    psi(F) = mu/2 (tr(F^T F)-3) - mu (J-1) + (lambda+mu)/2 (J-1)^2,
    J = det(F), is Newton VBD's stable Neo-Hookean density with the Lamé
    mapping mu_NH = mu, lambda_NH = lambda + mu in rest-zero form; see
    :func:`stable_neo_hookean_density`. It is finite through collapse and
    inversion, so no quadrature Jacobian sign is rejected.
    Total = sum_q psi(F_q) w_q + sum_v m_v |X_v-Y_v|^2/(2 dt^2)
    + sum_q damping*w_q*||F_q^T F_q - F_n,q^T F_n,q||_F^2/(2 dt).
    The final term matches Newton VBD solid damping, uses all nine metric
    entries, and vanishes under finite rigid motion of the previous geometry.
    A cell contributes density*cell_size^3/8 to each shared-corner mass.
    Material parameters are scalar or shape [cell_count], and fixed at module
    construction. No boundary constraints are applied by this loss.
    Lambda must be nonnegative so the volumetric stiffness lambda + mu stays
    positive and the law remains bounded below. Zero lambda is valid; the
    mu-dependent volumetric terms are retained.

    Args:
        rest: Canonical cuboid topology and rest geometry [m].
        lame_lambda: Nonnegative first Lamé parameter [Pa], scalar or per cell.
        lame_mu: Positive shear modulus [Pa], scalar or per cell.
        density: Positive rest density [kg/m^3], scalar or per cell.
        time_step: Positive scalar time interval [s].
        damping: Nonnegative viscosity [Pa s], scalar or per cell. Zero disables
            damping and permits legacy checkpoints lacking this buffer to load
            strictly. Such checkpoints cannot load into a damped configuration.
        dtype: PyTorch float32 (default when None) or float64.
    """

    def __init__(
        self,
        rest: VoxelGridData,
        lame_lambda,
        lame_mu,
        density,
        time_step: float,
        *,
        damping=0.0,
        dtype: torch.dtype | None = None,
    ):
        import torch

        super().__init__()
        dtype = torch.float32 if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("loss dtype must be float32 or float64")
        counts = tuple(int(count) + 1 for count in rest.cell_counts)
        expected = np.indices(counts).reshape(3, -1).T * rest.cell_size + rest.corner_rest_positions[0]
        if rest.corner_rest_positions.shape != expected.shape or not np.allclose(
            rest.corner_rest_positions, expected, rtol=0, atol=1e-12
        ):
            raise ValueError("rest positions must be the canonical axis-aligned cubic grid")
        corners = torch.tensor(rest.cell_corner_indices, dtype=torch.int64)
        if corners.ndim != 2 or corners.shape[1] != 8:
            raise ValueError("rest cells must have eight corner indices")
        self.particle_count = len(rest.corner_rest_positions)
        self.cell_count = len(corners)
        self.register_buffer("cell_corner_indices", corners)
        rule = hex_gauss_quadrature(rest.cell_size, dtype=np.float32 if dtype == torch.float32 else np.float64)
        self.register_buffer("shape_gradients", torch.tensor(rule.shape_gradients, dtype=dtype))
        self.register_buffer("quadrature_weights", torch.tensor(rule.weights, dtype=dtype))

        def material(value, name, lower, *, lower_inclusive=False):
            tensor = torch.as_tensor(value, dtype=dtype, device="cpu").detach().clone()
            if tensor.ndim == 0:
                tensor = tensor.expand(self.cell_count).clone()
            if tensor.shape != (self.cell_count,):
                raise ValueError(f"{name} must be scalar or shape [cell_count]")
            lower_valid = tensor >= lower if lower_inclusive else tensor > lower
            if not torch.isfinite(tensor).all().item() or not lower_valid.all().item():
                relation = "at least" if lower_inclusive else "greater than"
                raise ValueError(f"{name} must be finite and {relation} {lower}")
            self.register_buffer(name, tensor)
            return tensor

        material(lame_lambda, "lame_lambda", 0, lower_inclusive=True)
        mu = material(lame_mu, "lame_mu", 0)
        rho = material(density, "density", 0)
        material(damping, "damping", 0, lower_inclusive=True)
        self.register_buffer("time_step", _time_step_tensor(time_step, mu).detach().clone())
        cell_mass_eighth = rho * self.quadrature_weights.sum() / 8
        mass = torch.zeros(self.particle_count, dtype=dtype)
        mass.index_add_(0, corners.reshape(-1), cell_mass_eighth[:, None].expand(-1, 8).reshape(-1))
        self.register_buffer("lumped_mass", mass)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        damping_key = prefix + "damping"
        if damping_key not in state_dict:
            if (self.damping != 0).any().item():
                error_msgs.append(f"Missing {damping_key}: a damped energy requires an explicit damping buffer")
            else:
                # Old undamped checkpoints predate this physical material field.
                state_dict[damping_key] = self.damping.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self,
        positions: torch.Tensor,
        inertial_prediction: torch.Tensor,
        *,
        previous_positions: torch.Tensor | None = None,
    ) -> HexLossTerms:
        """Return per-object energies for positions and the physical Y [m].

        Args:
            positions: Current shared corners [m], shape [B,P,3].
            inertial_prediction: Unmodified physical predictor [m], same shape.
            previous_positions: Physical substep start [m], same shape, dtype
                and device. Required when damping is positive. Keep this anchor
                unchanged through inner solver iterations; gradients through it
                are preserved for differentiable physical rollouts.

        Collapsed or inverted quadrature points (J <= 0) are accepted; the
        stable Neo-Hookean density is finite for every finite deformation.

        Raises:
            ValueError: An input is malformed or nonfinite.
            TypeError: Input dtype/device differs from the module buffers.
        """
        import torch

        if positions.ndim != 3 or positions.shape[1:] != (self.particle_count, 3):
            raise ValueError("positions must have shape [B,particle_count,3]")
        if inertial_prediction.shape != positions.shape:
            raise ValueError("inertial_prediction must have the same shape as positions")
        damping_enabled = (self.damping > 0).any().item()
        if previous_positions is None and damping_enabled:
            raise ValueError("previous_positions is required when damping is positive")
        inputs = [(positions, "positions"), (inertial_prediction, "inertial_prediction")]
        if previous_positions is not None:
            if previous_positions.shape != positions.shape:
                raise ValueError("previous_positions must have the same shape as positions")
            inputs.append((previous_positions, "previous_positions"))
        for value, name in inputs:
            if value.dtype != self.lumped_mass.dtype or value.device != self.lumped_mass.device:
                raise TypeError(f"{name} must match the module buffers dtype and device")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite")
        corners = positions[:, self.cell_corner_indices]
        # Relative coordinates reduce cancellation from global translation;
        # partition of unity makes this the same material gradient.
        deformation = torch.einsum("bcki,qkj->bcqij", corners - corners[:, :, :1], self.shape_gradients)
        density = stable_neo_hookean_density(deformation, self.lame_mu[None, :, None], self.lame_lambda[None, :, None])
        elastic = (density * self.quadrature_weights[None, None]).sum(dim=(1, 2))
        inertia = (
            0.5
            * (self.lumped_mass[None, :, None] * (positions - inertial_prediction).square()).sum(dim=(1, 2))
            / self.time_step.square()
        )
        total = elastic + inertia
        damping = torch.zeros_like(elastic)
        if damping_enabled:
            metric_difference = damping_metric_difference(
                positions, previous_positions, self.cell_corner_indices, self.shape_gradients
            )
            damping = (
                self.damping[None, :, None]
                * self.quadrature_weights[None, None]
                * metric_difference.square().sum(dim=(-1, -2))
            ).sum(dim=(1, 2)) / (2 * self.time_step)
            total = total + damping
        return HexLossTerms(total, elastic, inertia, damping)
