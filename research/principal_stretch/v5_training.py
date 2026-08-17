# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Training primitives for the iterative principal-stretch v5 architecture.

This module deliberately contains losses and label construction, not an
optimizer or a dataset policy.  Its contracts are small enough to serialize
verbatim in a future schema-v5 checkpoint:

* labels use the same explicit ``F = A exp(H)`` representation as the model;
* compatible-state losses score already-projected positions and their induced
  deformation gradients;
* physics supervision is signed common-objective potential excess, evaluated
  with first derivatives only.

Every normalization floor is a fixed configuration value.  Nothing is fitted
from the evaluated candidate, and a candidate below the accepted reference's
potential remains a negative excess rather than being silently clamped away.
"""

from __future__ import annotations

import dataclasses
import math
import numbers

import torch

from .spd_log import so3_log_axial, spd_floor, sym_exp, sym_log
from .v5_objective import CommonObjectiveContext, common_objective_components


def _finite_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_non_negative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _execution_scalar(
    reference: torch.Tensor,
    name: str,
    value: float,
    *,
    strictly_positive: bool,
) -> torch.Tensor:
    """Materialize one config scalar without changing its registered meaning."""
    result = reference.new_tensor(value)
    if not bool(torch.isfinite(result).item()):
        raise ValueError(f"{name} must remain finite in execution dtype {reference.dtype}")
    if strictly_positive:
        if not bool((result > 0.0).item()):
            raise ValueError(f"{name} must remain strictly positive in execution dtype {reference.dtype}")
    elif bool((result < 0.0).item()):
        raise ValueError(f"{name} must remain non-negative in execution dtype {reference.dtype}")
    return result


def _require_finite_positive_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()) or bool((value <= 0.0).any().item()):
        raise ValueError(f"{name} must be finite and strictly positive in the execution dtype")


def _require_finite_tensors(name: str, values: tuple[torch.Tensor, ...]) -> None:
    if not all(bool(torch.isfinite(value).all().item()) for value in values):
        raise ValueError(f"{name} produced a non-finite value in the execution dtype")


@dataclasses.dataclass(frozen=True)
class PrincipalStretchLabelConfig:
    """Validity and reconstruction contract for v5 representation labels."""

    max_hencky_update: float = 0.35
    max_rotation_update: float = 0.75
    minimum_principal_stretch: float = 0.05
    maximum_rotation_branch_angle: float = 3.0
    reconstruction_relative_tolerance: float = 1.0e-6
    reconstruction_absolute_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in (
            "max_hencky_update",
            "max_rotation_update",
            "minimum_principal_stretch",
            "maximum_rotation_branch_angle",
            "reconstruction_relative_tolerance",
            "reconstruction_absolute_tolerance",
        ):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))
        if self.maximum_rotation_branch_angle > 3.0:
            raise ValueError("maximum_rotation_branch_angle must not exceed the validated 3.0 rad SO(3) branch")
        if self.max_rotation_update >= self.maximum_rotation_branch_angle:
            raise ValueError("max_rotation_update must be smaller than maximum_rotation_branch_angle")


@dataclasses.dataclass(frozen=True)
class PrincipalStretchLabelDiagnostics:
    """Scalar validity evidence recorded while constructing one label field."""

    minimum_determinant: float
    minimum_observed_principal_stretch: float
    floor_would_activate: bool
    maximum_rotation_angle: float
    near_pi_branch: bool
    maximum_hencky_update_norm: float
    maximum_rotation_update_norm: float
    maximum_hencky_cap_ratio: float
    maximum_rotation_cap_ratio: float
    maximum_reconstruction_absolute_error: float
    maximum_reconstruction_relative_error: float


@dataclasses.dataclass(frozen=True)
class PrincipalStretchLabels:
    """Exact material-frame v5 labels and their checked reconstruction."""

    H0: torch.Tensor
    A0: torch.Tensor
    H_target: torch.Tensor
    A_target: torch.Tensor
    delta_H: torch.Tensor
    omega: torch.Tensor
    reconstructed_F: torch.Tensor
    diagnostics: PrincipalStretchLabelDiagnostics

    @property
    def H_star(self) -> torch.Tensor:
        """Accepted-reference Hencky tensor (``H*``)."""
        return self.H_target

    @property
    def A_star(self) -> torch.Tensor:
        """Accepted-reference proper frame (``A*``)."""
        return self.A_target


@dataclasses.dataclass(frozen=True)
class RepresentationLossConfig:
    """Cap-normalized representation supervision weights."""

    max_hencky_update: float = 0.35
    max_rotation_update: float = 0.75
    hencky_weight: float = 1.0
    rotation_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_hencky_update", _finite_positive("max_hencky_update", self.max_hencky_update))
        object.__setattr__(
            self, "max_rotation_update", _finite_positive("max_rotation_update", self.max_rotation_update)
        )
        object.__setattr__(self, "hencky_weight", _finite_non_negative("hencky_weight", self.hencky_weight))
        object.__setattr__(self, "rotation_weight", _finite_non_negative("rotation_weight", self.rotation_weight))
        if self.hencky_weight == 0.0 and self.rotation_weight == 0.0:
            raise ValueError("at least one representation loss weight must be positive")


@dataclasses.dataclass(frozen=True)
class RepresentationLoss:
    """Scalar cap-normalized representation loss components."""

    total: torch.Tensor
    hencky: torch.Tensor
    rotation: torch.Tensor


@dataclasses.dataclass(frozen=True)
class CompatibleStateLossConfig:
    """Physical normalization and weights for already-compatible states."""

    characteristic_length_m: float = 1.0
    position_denominator_floor_kg_m2: float = 1.0e-16
    deformation_denominator_floor_m3: float = 1.0e-16
    position_weight: float = 1.0
    deformation_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "characteristic_length_m",
            "position_denominator_floor_kg_m2",
            "deformation_denominator_floor_m3",
        ):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))
        object.__setattr__(self, "position_weight", _finite_non_negative("position_weight", self.position_weight))
        object.__setattr__(
            self,
            "deformation_weight",
            _finite_non_negative("deformation_weight", self.deformation_weight),
        )
        if self.position_weight == 0.0 and self.deformation_weight == 0.0:
            raise ValueError("at least one compatible-state loss weight must be positive")


@dataclasses.dataclass(frozen=True)
class CompatibleStateLoss:
    """Mass/volume-normalized loss for one compatible projected state."""

    total: torch.Tensor
    position: torch.Tensor
    projected_F: torch.Tensor
    per_sample_total: torch.Tensor
    per_sample_position: torch.Tensor
    per_sample_projected_F: torch.Tensor
    position_denominator: torch.Tensor
    deformation_denominator: torch.Tensor
    position_floor_active: bool
    deformation_floor_active: bool


@dataclasses.dataclass(frozen=True)
class PotentialExcessLossConfig:
    """Signed potential-excess normalization fixed by a physical energy floor."""

    denominator_floor_joules: float = 1.0e-12
    negative_baseline_tolerance_joules: float = 1.0e-12
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "denominator_floor_joules",
            _finite_positive("denominator_floor_joules", self.denominator_floor_joules),
        )
        object.__setattr__(
            self,
            "negative_baseline_tolerance_joules",
            _finite_non_negative(
                "negative_baseline_tolerance_joules",
                self.negative_baseline_tolerance_joules,
            ),
        )
        object.__setattr__(self, "weight", _finite_positive("weight", self.weight))


@dataclasses.dataclass(frozen=True)
class PotentialExcessLoss:
    """Signed common-objective excess and its explicit baseline scale."""

    total: torch.Tensor
    excess: torch.Tensor
    per_sample_total: torch.Tensor
    per_sample_excess: torch.Tensor
    predicted_potential: torch.Tensor
    accepted_reference_potential: torch.Tensor
    baseline_potential: torch.Tensor
    baseline_excess: torch.Tensor
    denominator: torch.Tensor
    denominator_floor_active: bool
    negative_excess: bool
    negative_baseline_excess: bool


def _validate_deformation_fields(initial_F: torch.Tensor, target_F: torch.Tensor) -> None:
    if not isinstance(initial_F, torch.Tensor) or not isinstance(target_F, torch.Tensor):
        raise TypeError("initial_F and target_F must be torch.Tensor instances")
    if initial_F.shape != target_F.shape:
        raise ValueError("initial_F and target_F must have the same shape")
    if initial_F.ndim < 2 or initial_F.shape[-2:] != (3, 3):
        raise ValueError(f"deformation fields must end in (3, 3), got {tuple(initial_F.shape)}")
    if initial_F.numel() == 0:
        raise ValueError("deformation fields must not be empty")
    if initial_F.device != target_F.device or initial_F.dtype != target_F.dtype:
        raise ValueError("initial_F and target_F must share one device and dtype")
    if not initial_F.is_floating_point():
        raise ValueError("deformation fields must have a floating dtype")
    if not torch.isfinite(initial_F).all() or not torch.isfinite(target_F).all():
        raise ValueError("deformation fields must be finite")


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(vector[..., 0])
    return torch.stack(
        (
            torch.stack((zero, -vector[..., 2], vector[..., 1]), dim=-1),
            torch.stack((vector[..., 2], zero, -vector[..., 0]), dim=-1),
            torch.stack((-vector[..., 1], vector[..., 0], zero), dim=-1),
        ),
        dim=-2,
    )


def _principal_stretch(F: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvalsh(F.transpose(-1, -2) @ F)
    return torch.sqrt(torch.clamp_min(eigenvalues, 0.0))


def _matrix_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    absolute = float((actual - expected).abs().amax().item())
    scale = max(float(expected.abs().amax().item()), torch.finfo(expected.dtype).tiny)
    return absolute, absolute / scale


def build_principal_stretch_labels(
    initial_F: torch.Tensor,
    target_F: torch.Tensor,
    config: PrincipalStretchLabelConfig,
) -> PrincipalStretchLabels:
    """Construct exact v5 ``delta_H`` and principal-branch ``omega`` labels.

    ``initial_F`` and ``target_F`` may carry arbitrary leading dimensions.  A
    label is accepted only if both fields are positively oriented and above
    the registered principal-stretch floor, its generators lie strictly
    inside the model's radial caps, and explicit reconstruction satisfies the
    registered tolerance.  The spectral floor therefore never changes an
    accepted label.
    """
    if not isinstance(config, PrincipalStretchLabelConfig):
        raise TypeError("config must be a PrincipalStretchLabelConfig")
    _validate_deformation_fields(initial_F, target_F)

    for name in (
        "max_hencky_update",
        "max_rotation_update",
        "minimum_principal_stretch",
        "maximum_rotation_branch_angle",
        "reconstruction_relative_tolerance",
        "reconstruction_absolute_tolerance",
    ):
        _execution_scalar(initial_F, name, getattr(config, name), strictly_positive=True)
    floor_squared = config.minimum_principal_stretch * config.minimum_principal_stretch
    _execution_scalar(
        initial_F,
        "right_cauchy_green_eigenvalue_floor",
        floor_squared,
        strictly_positive=True,
    )

    source = initial_F.detach()
    target = target_F.detach()
    determinants = torch.stack((torch.linalg.det(source), torch.linalg.det(target)), dim=0)
    if not torch.isfinite(determinants).all():
        raise ValueError("representation-label determinants must be finite")
    minimum_determinant = float(determinants.amin().item())
    if minimum_determinant <= 0.0:
        raise ValueError("representation labels require a strictly positive determinant")

    stretches = torch.stack((_principal_stretch(source), _principal_stretch(target)), dim=0)
    if not torch.isfinite(stretches).all():
        raise ValueError("representation-label principal stretches must be finite")
    minimum_stretch = float(stretches.amin().item())
    floor_would_activate = minimum_stretch < config.minimum_principal_stretch
    if floor_would_activate:
        raise ValueError("representation label would activate the configured principal-stretch floor")

    H0 = 0.5 * sym_log(spd_floor(source.transpose(-1, -2) @ source, lam_min=floor_squared))
    H_target = 0.5 * sym_log(spd_floor(target.transpose(-1, -2) @ target, lam_min=floor_squared))
    A0 = source @ sym_exp(-H0)
    A_target = target @ sym_exp(-H_target)
    if not all(torch.isfinite(value).all() for value in (H0, H_target, A0, A_target)):
        raise ValueError("principal-stretch decomposition must be finite")
    relative_rotation = A0.transpose(-1, -2) @ A_target
    cosine = torch.clamp(0.5 * (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0), -1.0, 1.0)
    rotation_angle = torch.acos(cosine)
    if not torch.isfinite(rotation_angle).all():
        raise ValueError("principal SO(3) branch diagnostics must be finite")
    maximum_rotation_angle = float(rotation_angle.amax().item())
    near_pi_branch = maximum_rotation_angle >= config.maximum_rotation_branch_angle
    if near_pi_branch:
        raise ValueError("relative rotation is outside the configured principal SO(3) branch")

    delta_H = H_target - H0
    omega = so3_log_axial(relative_rotation)
    if not torch.isfinite(delta_H).all() or not torch.isfinite(omega).all():
        raise ValueError("principal-stretch labels must be finite")
    hencky_norm = torch.linalg.matrix_norm(delta_H, ord="fro", dim=(-2, -1))
    rotation_norm = torch.linalg.vector_norm(omega, dim=-1)
    maximum_hencky_norm = float(hencky_norm.amax().item())
    maximum_rotation_norm = float(rotation_norm.amax().item())
    if maximum_hencky_norm >= config.max_hencky_update:
        raise ValueError("Hencky label exceeds the strict radial model cap")
    if maximum_rotation_norm >= config.max_rotation_update:
        raise ValueError("rotation label exceeds the strict radial model cap")

    reconstructed = A0 @ torch.matrix_exp(_skew(omega)) @ sym_exp(H0 + delta_H)
    if not torch.isfinite(reconstructed).all():
        raise ValueError("principal-stretch label reconstruction must be finite")
    reconstruction_absolute, reconstruction_relative = _matrix_relative_error(reconstructed, target)
    scale = float(target.abs().amax().item())
    tolerance = config.reconstruction_absolute_tolerance + config.reconstruction_relative_tolerance * scale
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("label reconstruction tolerance must remain finite and positive in execution")
    if reconstruction_absolute > tolerance:
        raise ValueError(
            "principal-stretch label reconstruction missed its registered tolerance: "
            f"absolute={reconstruction_absolute:.6e}, tolerance={tolerance:.6e}"
        )

    diagnostics = PrincipalStretchLabelDiagnostics(
        minimum_determinant=minimum_determinant,
        minimum_observed_principal_stretch=minimum_stretch,
        floor_would_activate=floor_would_activate,
        maximum_rotation_angle=maximum_rotation_angle,
        near_pi_branch=near_pi_branch,
        maximum_hencky_update_norm=maximum_hencky_norm,
        maximum_rotation_update_norm=maximum_rotation_norm,
        maximum_hencky_cap_ratio=maximum_hencky_norm / config.max_hencky_update,
        maximum_rotation_cap_ratio=maximum_rotation_norm / config.max_rotation_update,
        maximum_reconstruction_absolute_error=reconstruction_absolute,
        maximum_reconstruction_relative_error=reconstruction_relative,
    )
    return PrincipalStretchLabels(H0, A0, H_target, A_target, delta_H, omega, reconstructed, diagnostics)


def _validate_loss_tensor(name: str, value: torch.Tensor, suffix: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < len(suffix) or value.shape[-len(suffix) :] != suffix:
        raise ValueError(f"{name} must end in {suffix}, got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _validate_symmetric(name: str, value: torch.Tensor) -> None:
    tolerance = 1.0e-10 if value.dtype == torch.float64 else 1.0e-5
    if bool(((value - value.transpose(-1, -2)).abs().amax() > tolerance).item()):
        raise ValueError(f"{name} must be symmetric")


def principal_stretch_representation_loss(
    predicted_delta_H: torch.Tensor,
    predicted_omega: torch.Tensor,
    target_delta_H: torch.Tensor,
    target_omega: torch.Tensor,
    config: RepresentationLossConfig,
    *,
    volume: torch.Tensor,
) -> RepresentationLoss:
    """Return rest-volume-weighted, cap-normalized errors for both v5 heads."""
    if not isinstance(config, RepresentationLossConfig):
        raise TypeError("config must be a RepresentationLossConfig")
    for name, value, suffix in (
        ("predicted_delta_H", predicted_delta_H, (3, 3)),
        ("target_delta_H", target_delta_H, (3, 3)),
        ("predicted_omega", predicted_omega, (3,)),
        ("target_omega", target_omega, (3,)),
    ):
        _validate_loss_tensor(name, value, suffix)
    if predicted_delta_H.shape != target_delta_H.shape or predicted_omega.shape != target_omega.shape:
        raise ValueError("predicted and target representation fields must have matching shapes")
    if predicted_delta_H.shape[:-2] != predicted_omega.shape[:-1]:
        raise ValueError("delta_H and omega must have matching leading shapes")
    if predicted_delta_H.ndim < 3:
        raise ValueError("representation fields must contain an explicit tetrahedron axis")
    tensors = (target_delta_H, predicted_omega, target_omega)
    if any(value.device != predicted_delta_H.device or value.dtype != predicted_delta_H.dtype for value in tensors):
        raise ValueError("all representation fields must share one device and dtype")
    _validate_symmetric("predicted_delta_H", predicted_delta_H)
    _validate_symmetric("target_delta_H", target_delta_H)
    if not isinstance(volume, torch.Tensor) or volume.shape != (predicted_delta_H.shape[-3],):
        raise ValueError("volume must have shape (T,) matching the representation fields")
    if volume.device != predicted_delta_H.device or volume.dtype != predicted_delta_H.dtype:
        raise ValueError("volume and representation fields must share one device and dtype")
    if not torch.isfinite(volume).all() or (volume <= 0.0).any():
        raise ValueError("volume must be finite and strictly positive")

    target_delta_H = target_delta_H.detach()
    target_omega = target_omega.detach()

    max_hencky_update = _execution_scalar(
        predicted_delta_H,
        "max_hencky_update",
        config.max_hencky_update,
        strictly_positive=True,
    )
    max_rotation_update = _execution_scalar(
        predicted_delta_H,
        "max_rotation_update",
        config.max_rotation_update,
        strictly_positive=True,
    )
    hencky_weight = _execution_scalar(
        predicted_delta_H,
        "hencky_weight",
        config.hencky_weight,
        strictly_positive=config.hencky_weight > 0.0,
    )
    rotation_weight = _execution_scalar(
        predicted_delta_H,
        "rotation_weight",
        config.rotation_weight,
        strictly_positive=config.rotation_weight > 0.0,
    )
    hencky_denominator = max_hencky_update.square()
    rotation_denominator = max_rotation_update.square()
    _require_finite_positive_tensor("Hencky cap-squared denominator", hencky_denominator)
    _require_finite_positive_tensor("rotation cap-squared denominator", rotation_denominator)

    predicted_h_norm = torch.linalg.matrix_norm(predicted_delta_H, ord="fro", dim=(-2, -1))
    predicted_rotation_norm = torch.linalg.vector_norm(predicted_omega, dim=-1)
    target_h_norm = torch.linalg.matrix_norm(target_delta_H, ord="fro", dim=(-2, -1))
    target_rotation_norm = torch.linalg.vector_norm(target_omega, dim=-1)
    if (predicted_h_norm >= max_hencky_update).any():
        raise ValueError("predicted Hencky update exceeds the strict radial model cap")
    if (predicted_rotation_norm >= max_rotation_update).any():
        raise ValueError("predicted rotation update exceeds the strict radial model cap")
    if (target_h_norm >= max_hencky_update).any():
        raise ValueError("target Hencky label exceeds the strict radial model cap")
    if (target_rotation_norm >= max_rotation_update).any():
        raise ValueError("target rotation label exceeds the strict radial model cap")

    volume_sum = volume.sum()
    _require_finite_positive_tensor("representation volume sum", volume_sum)
    normalized_volume = volume / volume_sum
    hencky_per_tet = (predicted_delta_H - target_delta_H).square().sum(dim=(-2, -1))
    hencky = (hencky_per_tet * normalized_volume).sum(dim=-1).mean()
    hencky = hencky / hencky_denominator
    rotation_per_tet = (predicted_omega - target_omega).square().sum(dim=-1)
    rotation = (rotation_per_tet * normalized_volume).sum(dim=-1).mean()
    rotation = rotation / rotation_denominator
    total = hencky_weight * hencky + rotation_weight * rotation
    _require_finite_tensors("representation loss", (hencky, rotation, total))
    return RepresentationLoss(total=total, hencky=hencky, rotation=rotation)


def _validate_mesh_fields(
    positions: torch.Tensor,
    reference: torch.Tensor,
    tets: torch.Tensor,
    J: torch.Tensor,
    volume: torch.Tensor,
    mass: torch.Tensor,
    pinned: torch.Tensor,
) -> None:
    _validate_loss_tensor("positions", positions, (3,))
    _validate_loss_tensor("reference_positions", reference, (3,))
    if positions.shape != reference.shape:
        raise ValueError("positions and reference_positions must have the same shape")
    if positions.ndim < 2:
        raise ValueError("positions must have shape (..., V, 3)")
    n_vertices = int(positions.shape[-2])
    if n_vertices == 0:
        raise ValueError("compatible-state loss requires at least one vertex")
    if tets.ndim != 2 or tets.shape[1] != 4 or tets.dtype != torch.int64:
        raise ValueError("tets must have shape (T, 4) and dtype torch.int64")
    n_tets = int(tets.shape[0])
    if n_tets == 0:
        raise ValueError("compatible-state loss requires at least one tetrahedron")
    expected_shapes = {"J": (n_tets, 4, 3), "volume": (n_tets,), "mass": (n_vertices,)}
    for name, value in (("J", J), ("volume", volume), ("mass", mass)):
        if value.shape != expected_shapes[name]:
            raise ValueError(f"{name} must have shape {expected_shapes[name]}")
        if not value.is_floating_point() or value.dtype != positions.dtype:
            raise ValueError("positions and floating mesh tensors must share one floating dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if pinned.ndim != 1 or pinned.dtype != torch.int64:
        raise ValueError("pinned must have shape (P,) and dtype torch.int64")
    tensors = (reference, tets, J, volume, mass, pinned)
    if any(value.device != positions.device for value in tensors):
        raise ValueError("all compatible-state tensors must share one device")
    if (tets < 0).any() or (tets >= n_vertices).any():
        raise ValueError("tets contains an out-of-range vertex")
    if (pinned < 0).any() or (pinned >= n_vertices).any():
        raise ValueError("pinned contains an out-of-range vertex")
    if torch.unique(pinned).numel() != pinned.numel():
        raise ValueError("pinned must not contain duplicates")
    if (volume <= 0.0).any():
        raise ValueError("volume must be strictly positive")
    if (mass < 0.0).any():
        raise ValueError("mass must be non-negative")


def _deformation_gradient(positions: torch.Tensor, tets: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tac,...tad->...tdc", J, positions[..., tets, :])


def _require_positive_deformation(name: str, F: torch.Tensor) -> None:
    determinant = torch.linalg.det(F)
    if not torch.isfinite(determinant).all():
        raise ValueError(f"{name} deformation determinant must be finite")
    if (determinant <= 0.0).any():
        raise ValueError(f"{name} deformation must have a strictly positive determinant")


def compatible_state_loss(
    projected_positions: torch.Tensor,
    reference_positions: torch.Tensor,
    *,
    tets: torch.Tensor,
    J: torch.Tensor,
    volume: torch.Tensor,
    mass: torch.Tensor,
    pinned: torch.Tensor,
    config: CompatibleStateLossConfig,
) -> CompatibleStateLoss:
    """Score an already-projected compatible state with physical weighting.

    This function intentionally does not run compatibility projection.  Its
    position argument is the differentiable output of that layer, so the
    induced deformation gradients are globally compatible by construction.
    Pinned coordinates must match the accepted reference bit-for-bit.
    """
    if not isinstance(config, CompatibleStateLossConfig):
        raise TypeError("config must be a CompatibleStateLossConfig")
    _validate_mesh_fields(projected_positions, reference_positions, tets, J, volume, mass, pinned)
    reference_positions = reference_positions.detach()
    if pinned.numel() and not torch.equal(projected_positions[..., pinned, :], reference_positions[..., pinned, :]):
        raise ValueError("projected state does not preserve exact pinned coordinates")

    free_mask = torch.ones(projected_positions.shape[-2], dtype=torch.bool, device=projected_positions.device)
    free_mask[pinned] = False
    if not free_mask.any():
        raise ValueError("compatible-state loss requires positive mass on at least one free vertex")
    free_mass = mass[free_mask].sum()
    _require_finite_positive_tensor("free-vertex mass sum", free_mass)

    characteristic_length = _execution_scalar(
        projected_positions,
        "characteristic_length_m",
        config.characteristic_length_m,
        strictly_positive=True,
    )
    position_floor = _execution_scalar(
        projected_positions,
        "position_denominator_floor_kg_m2",
        config.position_denominator_floor_kg_m2,
        strictly_positive=True,
    )
    deformation_floor = _execution_scalar(
        projected_positions,
        "deformation_denominator_floor_m3",
        config.deformation_denominator_floor_m3,
        strictly_positive=True,
    )
    position_weight = _execution_scalar(
        projected_positions,
        "position_weight",
        config.position_weight,
        strictly_positive=config.position_weight > 0.0,
    )
    deformation_weight = _execution_scalar(
        projected_positions,
        "deformation_weight",
        config.deformation_weight,
        strictly_positive=config.deformation_weight > 0.0,
    )

    predicted_F = _deformation_gradient(projected_positions, tets, J)
    reference_F = _deformation_gradient(reference_positions, tets, J)
    _require_positive_deformation("projected", predicted_F)
    _require_positive_deformation("reference", reference_F)

    difference = projected_positions[..., free_mask, :] - reference_positions[..., free_mask, :]
    position_numerator = (mass[free_mask, None] * difference.square()).sum(dim=(-2, -1))
    characteristic_length_squared = characteristic_length.square()
    _require_finite_positive_tensor("characteristic-length squared", characteristic_length_squared)
    raw_position_denominator = free_mass * characteristic_length_squared
    _require_finite_positive_tensor("raw position denominator", raw_position_denominator)
    position_denominator = torch.maximum(raw_position_denominator, position_floor)
    _require_finite_positive_tensor("position denominator", position_denominator)
    per_sample_position = position_numerator / position_denominator

    deformation_numerator = (volume * (predicted_F - reference_F).square().sum(dim=(-2, -1))).sum(dim=-1)
    volume_sum = volume.sum()
    _require_finite_positive_tensor("compatible-state volume sum", volume_sum)
    raw_deformation_denominator = projected_positions.new_tensor(9.0) * volume_sum
    _require_finite_positive_tensor("raw deformation denominator", raw_deformation_denominator)
    deformation_denominator = torch.maximum(raw_deformation_denominator, deformation_floor)
    _require_finite_positive_tensor("deformation denominator", deformation_denominator)
    per_sample_deformation = deformation_numerator / deformation_denominator
    per_sample_total = position_weight * per_sample_position + deformation_weight * per_sample_deformation
    total = per_sample_total.mean()
    position = per_sample_position.mean()
    projected_F_loss = per_sample_deformation.mean()
    _require_finite_tensors(
        "compatible-state loss",
        (
            position_numerator,
            deformation_numerator,
            per_sample_position,
            per_sample_deformation,
            per_sample_total,
            total,
            position,
            projected_F_loss,
        ),
    )
    return CompatibleStateLoss(
        total=total,
        position=position,
        projected_F=projected_F_loss,
        per_sample_total=per_sample_total,
        per_sample_position=per_sample_position,
        per_sample_projected_F=per_sample_deformation,
        position_denominator=position_denominator,
        deformation_denominator=deformation_denominator,
        position_floor_active=bool((raw_position_denominator < position_floor).item()),
        deformation_floor_active=bool((raw_deformation_denominator < deformation_floor).item()),
    )


def _validate_common_positions(
    context: CommonObjectiveContext,
    predicted_positions: torch.Tensor,
    accepted_reference_positions: torch.Tensor,
    baseline_positions: torch.Tensor,
) -> None:
    if not isinstance(context, CommonObjectiveContext):
        raise TypeError("context must be a CommonObjectiveContext")
    fields = (
        ("predicted_positions", predicted_positions),
        ("accepted_reference_positions", accepted_reference_positions),
        ("baseline_positions", baseline_positions),
    )
    for name, value in fields:
        _validate_loss_tensor(name, value, (context.n_vertices, 3))
    if (
        predicted_positions.shape != accepted_reference_positions.shape
        or predicted_positions.shape != baseline_positions.shape
    ):
        raise ValueError("predicted, accepted-reference, and baseline positions must have the same shape")
    for _name, value in fields:
        if value.device != context.device or value.dtype != context.dtype:
            raise ValueError("common-potential positions and context must share one device and dtype")
    pinned = context.pinned
    if pinned.numel():
        reference_pin = accepted_reference_positions[..., pinned, :]
        if not torch.equal(predicted_positions[..., pinned, :], reference_pin) or not torch.equal(
            baseline_positions[..., pinned, :], reference_pin
        ):
            raise ValueError("common-potential states must preserve exact pinned coordinates")
    for name, value in fields:
        _require_positive_deformation(name, _deformation_gradient(value, context.tets, context.J))


def common_potential_excess_loss(
    context: CommonObjectiveContext,
    predicted_positions: torch.Tensor,
    accepted_reference_positions: torch.Tensor,
    baseline_positions: torch.Tensor,
    config: PotentialExcessLossConfig,
) -> PotentialExcessLoss:
    """Return signed common-objective excess with a fixed denominator floor.

    The denominator is ``max(abs(Phi_baseline - Phi_reference), floor)``.  Its
    floor is an authenticated configuration value and both baseline/reference
    branches are detached.  Only ``Phi_predicted`` carries an autograd path, so
    optimizing this loss requires the common objective's first derivative and
    never constructs a residual-squared Hessian path.  A negative predicted
    excess is retained and reported. Leading axes represent multiple
    candidates for the *same* physical objective. Use
    :func:`common_potential_excess_loss_batch` when transitions have distinct
    physical contexts.
    """
    if not isinstance(config, PotentialExcessLossConfig):
        raise TypeError("config must be a PotentialExcessLossConfig")
    _validate_common_positions(context, predicted_positions, accepted_reference_positions, baseline_positions)

    negative_tolerance = _execution_scalar(
        predicted_positions,
        "negative_baseline_tolerance_joules",
        config.negative_baseline_tolerance_joules,
        strictly_positive=config.negative_baseline_tolerance_joules > 0.0,
    )
    floor = _execution_scalar(
        predicted_positions,
        "denominator_floor_joules",
        config.denominator_floor_joules,
        strictly_positive=True,
    )
    weight = _execution_scalar(
        predicted_positions,
        "weight",
        config.weight,
        strictly_positive=True,
    )

    reference = accepted_reference_positions.detach()
    baseline = baseline_positions.detach()
    predicted_potential = common_objective_components(context, predicted_positions)["total"]
    reference_potential = common_objective_components(context, reference)["total"]
    baseline_potential = common_objective_components(context, baseline)["total"]
    if not all(torch.isfinite(value).all() for value in (predicted_potential, reference_potential, baseline_potential)):
        raise ValueError("common-potential evaluation produced a non-finite value")
    excess = predicted_potential - reference_potential
    baseline_excess = baseline_potential - reference_potential
    _require_finite_tensors("common-potential excess", (excess, baseline_excess))
    if (baseline_excess < -negative_tolerance).any():
        raise ValueError("accepted reference has higher potential than the registered baseline beyond tolerance")
    denominator = torch.maximum(baseline_excess.abs(), floor)
    _require_finite_positive_tensor("potential-excess denominator", denominator)
    per_sample_total = weight * excess / denominator
    total = per_sample_total.mean()
    mean_excess = excess.mean()
    _require_finite_tensors(
        "common-potential loss",
        (per_sample_total, total, mean_excess),
    )
    return PotentialExcessLoss(
        total=total,
        excess=mean_excess,
        per_sample_total=per_sample_total,
        per_sample_excess=excess,
        predicted_potential=predicted_potential,
        accepted_reference_potential=reference_potential,
        baseline_potential=baseline_potential,
        baseline_excess=baseline_excess,
        denominator=denominator,
        denominator_floor_active=bool((baseline_excess.abs() < floor).any().item()),
        negative_excess=bool((excess < 0.0).any().item()),
        negative_baseline_excess=bool((baseline_excess < 0.0).any().item()),
    )


def common_potential_excess_loss_batch(
    contexts: tuple[CommonObjectiveContext, ...],
    predicted_positions: torch.Tensor,
    accepted_reference_positions: torch.Tensor,
    baseline_positions: torch.Tensor,
    config: PotentialExcessLossConfig,
) -> PotentialExcessLoss:
    """Evaluate one distinct common objective per training sample.

    Current v5 objective contexts intentionally bind one unbatched physical
    timestep. This helper routes a topology-homogeneous tensor batch through
    its per-sample contexts, preventing different inertial targets, masses, or
    materials from being silently scored under the first sample's objective.
    """
    if not isinstance(contexts, tuple) or not contexts:
        raise ValueError("contexts must be a non-empty tuple")
    if predicted_positions.ndim != 3:
        raise ValueError("batched common-potential positions must have shape (B, V, 3)")
    if (
        accepted_reference_positions.shape != predicted_positions.shape
        or baseline_positions.shape != predicted_positions.shape
    ):
        raise ValueError("batched predicted, reference, and baseline positions must have identical shapes")
    if len(contexts) != predicted_positions.shape[0]:
        raise ValueError("contexts must contain exactly one physical objective per batch sample")

    individual = tuple(
        common_potential_excess_loss(
            context,
            predicted_positions[index],
            accepted_reference_positions[index],
            baseline_positions[index],
            config,
        )
        for index, context in enumerate(contexts)
    )
    per_sample_total = torch.stack(tuple(result.total for result in individual))
    per_sample_excess = torch.stack(tuple(result.excess for result in individual))
    predicted_potential = torch.stack(tuple(result.predicted_potential for result in individual))
    reference_potential = torch.stack(tuple(result.accepted_reference_potential for result in individual))
    baseline_potential = torch.stack(tuple(result.baseline_potential for result in individual))
    baseline_excess = torch.stack(tuple(result.baseline_excess for result in individual))
    denominator = torch.stack(tuple(result.denominator for result in individual))
    return PotentialExcessLoss(
        total=per_sample_total.mean(),
        excess=per_sample_excess.mean(),
        per_sample_total=per_sample_total,
        per_sample_excess=per_sample_excess,
        predicted_potential=predicted_potential,
        accepted_reference_potential=reference_potential,
        baseline_potential=baseline_potential,
        baseline_excess=baseline_excess,
        denominator=denominator,
        denominator_floor_active=any(result.denominator_floor_active for result in individual),
        negative_excess=any(result.negative_excess for result in individual),
        negative_baseline_excess=any(result.negative_baseline_excess for result in individual),
    )


__all__ = [
    "CompatibleStateLoss",
    "CompatibleStateLossConfig",
    "PotentialExcessLoss",
    "PotentialExcessLossConfig",
    "PrincipalStretchLabelConfig",
    "PrincipalStretchLabelDiagnostics",
    "PrincipalStretchLabels",
    "RepresentationLoss",
    "RepresentationLossConfig",
    "build_principal_stretch_labels",
    "common_potential_excess_loss",
    "common_potential_excess_loss_batch",
    "compatible_state_loss",
    "principal_stretch_representation_loss",
]
