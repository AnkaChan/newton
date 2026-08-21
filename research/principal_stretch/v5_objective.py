# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Device-resident common objective and residual for iterative v5 models.

The residual is evaluated analytically.  In particular, it does not call
``torch.autograd.grad`` and therefore can be detached before it is used as a
recurrent model feature without constructing a second-derivative path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers

import torch

from .torch_solver import (
    OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT,
    operator_volume_sha256,
)

_OBJECTIVE_TENSOR_FIELDS = ("tets", "J", "volume", "mass", "mu", "lam", "inertial_target", "pinned")


def _require_execution_finite(
    name: str,
    value: torch.Tensor,
    *,
    positive: bool = False,
) -> None:
    """Validate a derived value after execution-dtype materialization."""
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite in the execution dtype")
    if positive and not (value > 0.0).all():
        raise ValueError(f"{name} must be positive in the execution dtype")


def _require_finite_output(name: str, value: torch.Tensor) -> None:
    """Reject a non-finite public result at the authenticated boundary."""
    if not torch.isfinite(value.detach()).all():
        raise RuntimeError(f"{name} produced a non-finite value")


def _objective_digest(context: CommonObjectiveContext) -> str:
    bound_volume = object.__getattribute__(context, "operator_volume_policy") is not None
    digest = hashlib.sha256(
        b"pr2901-v5-common-objective-context-v3\0" if bound_volume else b"pr2901-v5-common-objective-context-v2\0"
    )
    if bound_volume:
        binding_metadata = json.dumps(
            {
                "operator_geometry_sha256": object.__getattribute__(context, "operator_geometry_sha256"),
                "operator_volume_policy": object.__getattribute__(context, "operator_volume_policy"),
                "operator_volume_sha256": object.__getattribute__(context, "operator_volume_sha256"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(binding_metadata).to_bytes(8, "big"))
        digest.update(binding_metadata)
    scalar_metadata = json.dumps(
        {
            "dt_float_hex": object.__getattribute__(context, "dt").hex(),
            "inverse_dt_squared_float_hex": object.__getattribute__(context, "_inverse_dt_squared").hex(),
            "residual_scale_float_hex": object.__getattribute__(context, "residual_scale").hex(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(scalar_metadata).to_bytes(8, "big"))
    digest.update(scalar_metadata)
    for name in _OBJECTIVE_TENSOR_FIELDS:
        value = object.__getattribute__(context, name).detach().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = value.view(torch.uint8).cpu().numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class CommonObjectiveContext:
    """Immutable tensors for one stable-Neo-Hookean implicit objective.

    All tensor inputs are detached and cloned on their existing device. The
    public tensor attributes return clones, so ordinary caller mutation cannot
    change the owned problem. Construction and every external trust-boundary
    validation copy canonical bytes to the host for authentication. The
    iterative solver performs those checks at solve boundaries and around
    every external constraint-hook call, not inside every residual evaluation;
    this cold per-step cost must still be reported. Candidate positions may be
    unbatched or have arbitrary
    leading batch dimensions. The normalized-residual force scale is derived
    exactly from total rest volume, material coefficients, free mass, and
    ``dt``; callers cannot choose it. Construction also proves that the
    inverse timestep square, normalization, stable-NH ``alpha``, and static
    inertial/material products remain finite after conversion to the actual
    execution dtype.

    Args:
        tets: Tet vertex indices, shape ``[T, 4]``.
        J: Shape-function gradients [1/m], shape ``[T, 4, 3]``.
        volume: Rest volumes [m^3], shape ``[T]``.
        mass: Lumped vertex masses [kg], shape ``[V]``.
        mu: Per-tet first material coefficient [Pa], shape ``[T]``.
        lam: Per-tet second material coefficient [Pa], shape ``[T]``.
        inertial_target: Force-shifted inertial target positions [m], shape
            ``[V, 3]``.
        pinned: Dirichlet vertex indices, shape ``[P]``.
        dt: Implicit substep duration [s].
        operator_geometry_sha256: Optional authenticated source-operator
            identity for a portable volume binding.
        operator_volume_policy: Optional portable volume policy. This and both
            SHA-256 fields must be supplied together.
        operator_volume_sha256: Optional canonical volume identity.
    """

    tets: torch.Tensor
    J: torch.Tensor
    volume: torch.Tensor
    mass: torch.Tensor
    mu: torch.Tensor
    lam: torch.Tensor
    inertial_target: torch.Tensor
    pinned: torch.Tensor
    dt: float
    operator_geometry_sha256: str | None = None
    operator_volume_policy: str | None = None
    operator_volume_sha256: str | None = None
    residual_scale: float = dataclasses.field(init=False)
    _inverse_dt_squared: float = dataclasses.field(init=False, repr=False)
    common_objective_sha256: str = dataclasses.field(init=False)
    _sealed: bool = dataclasses.field(init=False, repr=False, default=False)

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in _OBJECTIVE_TENSOR_FIELDS and object.__getattribute__(self, "_sealed"):
            return value.clone()
        return value

    def __post_init__(self) -> None:
        volume_binding = (
            self.operator_geometry_sha256,
            self.operator_volume_policy,
            self.operator_volume_sha256,
        )
        bound_volume = all(value is not None for value in volume_binding)
        if any(value is not None for value in volume_binding) and not bound_volume:
            raise ValueError("operator geometry and volume identity fields must all be provided together")
        if bound_volume:
            for name, value in (
                ("operator_geometry_sha256", self.operator_geometry_sha256),
                ("operator_volume_sha256", self.operator_volume_sha256),
            ):
                if (
                    type(value) is not str
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            if self.operator_volume_policy != OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT:
                raise ValueError("operator volume policy is not the registered portable policy")
        tensor_fields = ("tets", "J", "volume", "mass", "mu", "lam", "inertial_target", "pinned")
        for name in tensor_fields:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        if self.inertial_target.ndim != 2 or self.inertial_target.shape[1] != 3:
            raise ValueError(f"inertial_target must have shape (V, 3), got {tuple(self.inertial_target.shape)}")
        n_vertices = int(self.inertial_target.shape[0])
        if n_vertices == 0:
            raise ValueError("at least one vertex is required")
        if self.tets.ndim != 2 or self.tets.shape[1] != 4:
            raise ValueError(f"tets must have shape (T, 4), got {tuple(self.tets.shape)}")
        n_tets = int(self.tets.shape[0])
        if n_tets == 0:
            raise ValueError("at least one tetrahedron is required")
        expected_shapes = {
            "J": (n_tets, 4, 3),
            "volume": (n_tets,),
            "mass": (n_vertices,),
            "mu": (n_tets,),
            "lam": (n_tets,),
        }
        for name, expected in expected_shapes.items():
            actual = tuple(getattr(self, name).shape)
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}, got {actual}")
        if self.pinned.ndim != 1:
            raise ValueError(f"pinned must have shape (P,), got {tuple(self.pinned.shape)}")
        if self.tets.dtype != torch.int64:
            raise ValueError(f"tets must have dtype torch.int64, got {self.tets.dtype}")
        if self.pinned.dtype != torch.int64:
            raise ValueError(f"pinned must have dtype torch.int64, got {self.pinned.dtype}")

        device = self.inertial_target.device
        dtype = self.inertial_target.dtype
        if not self.inertial_target.is_floating_point():
            raise ValueError("inertial_target must have a floating dtype")
        if any(getattr(self, name).device != device for name in tensor_fields):
            raise ValueError("all common-objective tensors must be on the same device")
        for name in ("J", "volume", "mass", "mu", "lam"):
            value = getattr(self, name)
            if not value.is_floating_point() or value.dtype != dtype:
                raise ValueError("all floating common-objective tensors must share one floating dtype")
        if bound_volume and dtype != torch.float64:
            raise ValueError("portable operator-volume binding requires torch.float64 execution")

        for name in ("J", "volume", "mass", "mu", "lam", "inertial_target"):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite")
        if (self.volume <= 0.0).any():
            raise ValueError("volume must be strictly positive")
        if (self.mass < 0.0).any():
            raise ValueError("mass must be non-negative")
        if (self.mu < 0.0).any() or (self.lam < 0.0).any():
            raise ValueError("stable Neo-Hookean material coefficients must be non-negative")
        active = (self.mu > 0.0) | (self.lam > 0.0)
        if (active & (self.lam <= 0.0)).any():
            raise ValueError("stable Neo-Hookean lambda must be positive on active tets")
        if (self.tets < 0).any() or (self.tets >= n_vertices).any():
            raise ValueError("tets contains an out-of-range vertex")
        if self.pinned.numel() > 0:
            if (self.pinned < 0).any() or (self.pinned >= n_vertices).any():
                raise ValueError("pinned contains an out-of-range vertex")
            if torch.unique(self.pinned).numel() != self.pinned.numel():
                raise ValueError("pinned must not contain duplicates")

        if isinstance(self.dt, bool) or not isinstance(self.dt, numbers.Real):
            raise TypeError("dt must be a real number")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {self.dt}")
        dt_squared = float(self.dt) * float(self.dt)
        if not math.isfinite(dt_squared) or dt_squared == 0.0:
            raise ValueError("dt must have a finite representable inverse square")
        inverse_dt_squared = 1.0 / dt_squared
        if not math.isfinite(inverse_dt_squared) or inverse_dt_squared <= 0.0:
            raise ValueError("dt must have a finite representable inverse square")
        # Own the tensors so later mutation of constructor inputs cannot change
        # the bound physical problem.  These copies remain on the input device.
        for name in tensor_fields:
            object.__setattr__(self, name, getattr(self, name).detach().clone())
        object.__setattr__(self, "dt", float(self.dt))
        if bound_volume:
            actual_volume_sha256 = operator_volume_sha256(
                self.operator_geometry_sha256,
                self.volume.detach().contiguous().cpu().numpy(),
                policy=self.operator_volume_policy,
            )
            if actual_volume_sha256 != self.operator_volume_sha256:
                raise ValueError("common-objective operator-volume SHA-256 verification failed")

        inverse_dt_squared_execution = torch.tensor(inverse_dt_squared, dtype=dtype, device=device)
        _require_execution_finite(
            "inverse timestep square",
            inverse_dt_squared_execution,
            positive=True,
        )
        inertial_weight = self.mass * inverse_dt_squared_execution
        _require_execution_finite("mass-times-inverse-timestep-square weights", inertial_weight)
        _require_execution_finite(
            "summed mass-times-inverse-timestep-square weight",
            inertial_weight.sum(),
        )

        lambda_floor = torch.tensor(1.0e-6, dtype=dtype, device=device)
        _require_execution_finite("stable Neo-Hookean lambda floor", lambda_floor, positive=True)
        alpha = 1.0 + self.mu / torch.clamp_min(self.lam, lambda_floor)
        _require_execution_finite("stable Neo-Hookean alpha", alpha, positive=True)
        volume_mu = self.volume * self.mu
        volume_lam = self.volume * self.lam
        lambda_alpha = self.lam * alpha
        lambda_alpha_squared = self.lam * alpha.square()
        volume_lambda_alpha_squared = self.volume * lambda_alpha_squared
        for name, value in (
            ("volume-times-mu coefficients", volume_mu),
            ("volume-times-lambda coefficients", volume_lam),
            ("lambda-times-alpha coefficients", lambda_alpha),
            ("lambda-times-alpha-squared coefficients", lambda_alpha_squared),
            ("volume-times-lambda-times-alpha-squared coefficients", volume_lambda_alpha_squared),
            ("summed volume-times-mu coefficient", volume_mu.sum()),
            ("summed volume-times-lambda-times-alpha-squared coefficient", volume_lambda_alpha_squared.sum()),
        ):
            _require_execution_finite(name, value)

        pinned_mask = torch.zeros(n_vertices, dtype=torch.bool, device=device)
        pinned_mask[self.pinned] = True
        if pinned_mask.all():
            raise ValueError("at least one free vertex is required")
        if (self.mass[~pinned_mask] <= 0.0).any():
            raise ValueError("every free vertex must have positive mass")
        # Derive the normalization in CPU float64 during authenticated setup so
        # the checkpoint identity is independent of the execution device and
        # model dtype. This transfer is already part of the documented cold
        # context-authentication cost.
        volume64 = self.volume.detach().to(device="cpu", dtype=torch.float64)
        mu64 = self.mu.detach().to(device="cpu", dtype=torch.float64)
        lam64 = self.lam.detach().to(device="cpu", dtype=torch.float64)
        mass64 = self.mass.detach().to(device="cpu", dtype=torch.float64)
        pinned64 = self.pinned.detach().to(device="cpu")
        free_mask64 = torch.ones(n_vertices, dtype=torch.bool)
        free_mask64[pinned64] = False
        characteristic_length = volume64.sum().pow(1.0 / 3.0)
        material_force_scale = (volume64 * (mu64 + lam64)).sum() / characteristic_length
        inertial_force_scale = mass64[free_mask64].sum() * characteristic_length / (self.dt * self.dt)
        residual_scale = float(torch.maximum(material_force_scale, inertial_force_scale).clamp_min(1.0e-12))
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("derived residual scale must be finite and positive")
        residual_scale_execution = torch.tensor(residual_scale, dtype=dtype, device=device)
        _require_execution_finite(
            "derived residual scale",
            residual_scale_execution,
            positive=True,
        )
        object.__setattr__(self, "residual_scale", residual_scale)
        object.__setattr__(self, "_inverse_dt_squared", inverse_dt_squared)
        object.__setattr__(self, "common_objective_sha256", _objective_digest(self))
        object.__setattr__(self, "_sealed", True)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the bound objective."""
        return int(object.__getattribute__(self, "mass").shape[0])

    @property
    def n_tets(self) -> int:
        """Number of tetrahedra in the bound objective."""
        return int(object.__getattribute__(self, "tets").shape[0])

    @property
    def device(self) -> torch.device:
        """Device holding every context tensor."""
        return object.__getattribute__(self, "inertial_target").device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype shared by the objective tensors."""
        return object.__getattribute__(self, "inertial_target").dtype

    def _owned_tensor(self, name: str) -> torch.Tensor:
        """Return a zero-copy owned tensor for internal solver code."""
        if name not in _OBJECTIVE_TENSOR_FIELDS:
            raise KeyError(name)
        return object.__getattribute__(self, name)

    def validate_immutable(self) -> None:
        """Reauthenticate the sealed context against its canonical bytes."""
        if not self._sealed:
            raise RuntimeError("common-objective context is not sealed")
        volume_binding = tuple(
            object.__getattribute__(self, name)
            for name in ("operator_geometry_sha256", "operator_volume_policy", "operator_volume_sha256")
        )
        if any(value is not None for value in volume_binding):
            if not all(value is not None for value in volume_binding):
                raise RuntimeError("common-objective portable volume binding changed after authentication")
            geometry_sha256, policy, expected_volume_sha256 = volume_binding
            if policy != OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT:
                raise RuntimeError("common-objective portable volume policy changed after authentication")
            try:
                actual_volume_sha256 = operator_volume_sha256(
                    geometry_sha256,
                    object.__getattribute__(self, "volume").detach().contiguous().cpu().numpy(),
                    policy=policy,
                )
            except ValueError as exc:
                raise RuntimeError("common-objective portable volume binding changed after authentication") from exc
            if actual_volume_sha256 != expected_volume_sha256:
                raise RuntimeError("common-objective portable volume identity changed after authentication")
        expected_inverse = 1.0 / (self.dt * self.dt)
        if self._inverse_dt_squared != expected_inverse:
            raise RuntimeError("common-objective derived timestep state changed after authentication")
        if _objective_digest(self) != self.common_objective_sha256:
            raise RuntimeError("common-objective context changed after authentication")

    def _validate_sealed(self) -> None:
        """Check construction state without repeating cold byte authentication."""
        if not self._sealed:
            raise RuntimeError("common-objective context is not sealed")


def _validate_positions(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
    *,
    authenticate_context: bool,
) -> None:
    if not isinstance(context, CommonObjectiveContext):
        raise TypeError("context must be a CommonObjectiveContext")
    if authenticate_context:
        context.validate_immutable()
    else:
        context._validate_sealed()
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim < 2 or positions.shape[-2:] != (context.n_vertices, 3):
        raise ValueError(f"positions must have shape (..., {context.n_vertices}, 3), got {tuple(positions.shape)}")
    if positions.device != context.device:
        raise ValueError("positions and context must be on the same device")
    if positions.dtype != context.dtype:
        raise ValueError("positions and context must share one floating dtype")


def _deformation_gradient(context: CommonObjectiveContext, positions: torch.Tensor) -> torch.Tensor:
    tets = context._owned_tensor("tets")
    J = context._owned_tensor("J")
    x_tet = positions[..., tets, :]
    return torch.einsum("tac,...tad->...tdc", J, x_tet)


def _determinant_3x3(matrix: torch.Tensor) -> torch.Tensor:
    """Polynomial determinant, finite at flat and inverted tetrahedra."""
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 0, 2]
    d = matrix[..., 1, 0]
    e = matrix[..., 1, 1]
    f = matrix[..., 1, 2]
    g = matrix[..., 2, 0]
    h = matrix[..., 2, 1]
    i = matrix[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _cofactor_3x3(matrix: torch.Tensor) -> torch.Tensor:
    """Derivative of the polynomial determinant with respect to ``matrix``."""
    row_0 = torch.linalg.cross(matrix[..., 1, :], matrix[..., 2, :], dim=-1)
    row_1 = torch.linalg.cross(matrix[..., 2, :], matrix[..., 0, :], dim=-1)
    row_2 = torch.linalg.cross(matrix[..., 0, :], matrix[..., 1, :], dim=-1)
    return torch.stack((row_0, row_1, row_2), dim=-2)


def _common_objective_components(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
    *,
    authenticate_context: bool,
) -> dict[str, torch.Tensor]:
    """Implement batched or unbatched common-objective components.

    Args:
        context: Fixed physical problem data.
        positions: Candidate positions [m], shape ``[..., V, 3]``.

    Returns:
        ``total``, ``inertia``, and ``elastic`` tensors with shape
        ``positions.shape[:-2]``.  Unbatched components are scalar tensors.
    """
    _validate_positions(context, positions, authenticate_context=authenticate_context)
    inertial_target = context._owned_tensor("inertial_target")
    mass = context._owned_tensor("mass")
    volume = context._owned_tensor("volume")
    delta = positions - inertial_target
    inertia = 0.5 * context._inverse_dt_squared * (mass[:, None] * delta.square()).sum(dim=(-2, -1))

    deformation_gradient = _deformation_gradient(context, positions)
    determinant = _determinant_3x3(deformation_gradient)
    frobenius_squared = deformation_gradient.square().sum(dim=(-2, -1))
    mu = context._owned_tensor("mu")
    lam = context._owned_tensor("lam")
    active = (mu > 0.0) | (lam > 0.0)
    alpha = 1.0 + mu / torch.clamp_min(lam, 1.0e-6)
    density = 0.5 * mu * (frobenius_squared - 3.0)
    density = density + 0.5 * lam * (determinant - alpha).square()
    density = torch.where(active, density, torch.zeros_like(density))
    elastic = (volume * density).sum(dim=-1)
    return {"total": inertia + elastic, "inertia": inertia, "elastic": elastic}


def common_objective_components(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate common-objective components after authenticating the context.

    Args:
        context: Fixed physical problem data.
        positions: Candidate positions [m], shape ``[..., V, 3]``.

    Returns:
        ``total``, ``inertia``, and ``elastic`` tensors with shape
        ``positions.shape[:-2]``. Unbatched components are scalar tensors.

    Raises:
        RuntimeError: If any returned component is non-finite.
    """
    components = _common_objective_components(context, positions, authenticate_context=True)
    for name, value in components.items():
        _require_finite_output(f"common-objective {name}", value)
    return components


def _common_objective_components_trusted(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate components inside a caller-authenticated execution scope.

    This hot helper deliberately performs no host-visible finite check. Its
    authenticated iterative/corrector callers must reject non-finite results
    before committing a state.
    """
    return _common_objective_components(context, positions, authenticate_context=False)


def _common_objective_residual(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
    *,
    normalize: bool = False,
    detach: bool = False,
    authenticate_context: bool,
) -> torch.Tensor:
    """Implement the exact analytic common-objective position residual.

    The free-vertex residual is

    ``m / dt^2 * (x - y) + scatter(V * P @ J)``,

    where ``P = mu * F + lam * (det(F) - alpha) * cof(F)``.  Pinned rows are
    replaced with exact zeros.  ``detach=True`` detaches positions before any
    arithmetic, making the result suitable as a recurrent feature without a
    Hessian-through-residual path.

    Args:
        context: Fixed physical problem data.
        positions: Candidate positions [m], shape ``[..., V, 3]``.
        normalize: Divide the residual by the context's fixed force scale.
        detach: Return a feature detached from the position autograd graph.

    Returns:
        Position residuals [N], or dimensionless normalized residuals, with
        the same shape as ``positions``.
    """
    _validate_positions(context, positions, authenticate_context=authenticate_context)
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a bool")
    if not isinstance(detach, bool):
        raise TypeError("detach must be a bool")

    candidate = positions.detach() if detach else positions
    deformation_gradient = _deformation_gradient(context, candidate)
    determinant = _determinant_3x3(deformation_gradient)
    cofactor = _cofactor_3x3(deformation_gradient)
    mu = context._owned_tensor("mu")
    lam = context._owned_tensor("lam")
    J = context._owned_tensor("J")
    volume = context._owned_tensor("volume")
    mass = context._owned_tensor("mass")
    inertial_target = context._owned_tensor("inertial_target")
    tets = context._owned_tensor("tets")
    pinned = context._owned_tensor("pinned")
    active = (mu > 0.0) | (lam > 0.0)
    alpha = 1.0 + mu / torch.clamp_min(lam, 1.0e-6)
    first_piola = mu[:, None, None] * deformation_gradient
    first_piola = first_piola + (lam * (determinant - alpha))[..., None, None] * cofactor
    first_piola = torch.where(active[:, None, None], first_piola, torch.zeros_like(first_piola))

    tet_contribution = torch.einsum("...tdc,tac->...tad", first_piola, J)
    tet_contribution = tet_contribution * volume[:, None, None]
    flat_contribution = tet_contribution.reshape(*candidate.shape[:-2], -1, 3)
    inertia = context._inverse_dt_squared * mass[:, None] * (candidate - inertial_target)
    residual = inertia.index_add(-2, tets.reshape(-1), flat_contribution)
    residual = residual.index_fill(-2, pinned, 0.0)
    if normalize:
        residual = residual / context.residual_scale
    return residual


def common_objective_residual(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
    *,
    normalize: bool = False,
    detach: bool = False,
) -> torch.Tensor:
    """Evaluate the exact analytic residual after authenticating the context.

    The free-vertex residual is

    ``m / dt^2 * (x - y) + scatter(V * P @ J)``,

    where ``P = mu * F + lam * (det(F) - alpha) * cof(F)``. Pinned rows are
    exact zeros. ``detach=True`` prevents a Hessian-through-residual path.

    Raises:
        RuntimeError: If the returned residual contains a non-finite value.
    """
    residual = _common_objective_residual(
        context,
        positions,
        normalize=normalize,
        detach=detach,
        authenticate_context=True,
    )
    _require_finite_output("common-objective residual", residual)
    return residual


def _common_objective_residual_trusted(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
    *,
    normalize: bool = False,
    detach: bool = False,
) -> torch.Tensor:
    """Evaluate a residual inside a caller-authenticated execution scope.

    This hot helper deliberately performs no host-visible finite check. Its
    authenticated iterative/corrector callers must reject non-finite results
    before committing a state.
    """
    return _common_objective_residual(
        context,
        positions,
        normalize=normalize,
        detach=detach,
        authenticate_context=False,
    )


__all__ = ["CommonObjectiveContext", "common_objective_components", "common_objective_residual"]
