# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Retain physical step starts and replay them with a supplied current network.

The disk record stores X and V before rigid-guided candidate construction, not
the candidate or Y, plus the detached optimizer history carried into that
step (None at a trajectory start). Replay recomputes candidate and Y from the
retained physical state, forces, gravity, material, mass, and timestep, and
seeds the inner unroll with the retained history. A synchronous pre-step
callback also retains the input to a learned solve that subsequently fails.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch  # noqa: TID253 -- Optional experimental tensor/serialization adapter.

from .data import generate_cuboid
from .disk_replay import DiskReplayStore, ReplayState
from .input_assembly import OptimizerHistory
from .network import IntrinsicSolverNetwork
from .newton_model import build_newton_hex_model
from .newton_solver import SolverLearnedIntrinsic
from .physical_rollout import PhysicalRollout, PhysicalStep, PhysicalWindow
from .unrolled_solver import UnrolledHexSolver

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["build_replay_rollout", "physical_context", "replay_state", "retain_windows"]


def _cpu_array(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy().copy()


def physical_context(rollout: PhysicalRollout) -> dict:
    """Return deduplicable physical context from the exact active solver buffers.

    Float32 rest geometry is stored alongside canonical cell size and origin;
    material arrays and native lumped masses are stored separately because
    density-derived mass can differ by float32 roundtrip through Model metadata.
    Network parameters and architecture are intentionally absent: a retained
    physical state can be re-solved with a different current network.
    Schema 2 includes absolute per-cell damping [Pa*s].
    """
    step = rollout.unrolled.step
    rest = rollout.solver._rest
    return {
        "metadata": {
            "schema_version": 2,
            "cell_counts": list(rest.cell_counts),
            "cell_size": float(rest.cell_size),
            "origin": [float(value) for value in rest.corner_rest_positions[0]],
            "time_step": rollout.time_step,
        },
        "arrays": {
            "rest_positions": _cpu_array(step.rest_positions),
            "cell_corner_indices": _cpu_array(step.cell_corner_indices),
            "fixed_indices": _cpu_array(step.fixed_indices),
            "lame_lambda": _cpu_array(step.energy.lame_lambda),
            "lame_mu": _cpu_array(step.energy.lame_mu),
            "density": _cpu_array(step.energy.density),
            "damping": _cpu_array(step.energy.damping),
            "lumped_mass": _cpu_array(step.energy.lumped_mass),
        },
    }


def _context_damping(context: Mapping) -> np.ndarray:
    """Read explicit viscosity, interpreting legacy context schema 1 as zero."""
    metadata, arrays = context["metadata"], context["arrays"]
    version = metadata.get("schema_version")
    if version not in (1, 2):
        raise ValueError("unsupported physical replay context schema")
    if version == 2 and "damping" not in arrays:
        raise ValueError("physical replay context schema 2 requires damping")
    material = np.asarray(arrays["lame_mu"])
    damping = np.asarray(arrays.get("damping", np.zeros_like(material)))
    if (
        damping.shape != material.shape
        or damping.dtype.kind not in "fiu"
        or not np.isfinite(damping).all()
        or (damping < 0).any()
    ):
        raise ValueError("replay damping must be finite and nonnegative with one value per cell")
    if version == 1 and (damping != 0).any():
        raise ValueError("legacy replay context schema 1 only supports zero damping")
    return damping


def _verify_context(rollout: PhysicalRollout, context: Mapping) -> None:
    expected = physical_context(rollout)
    metadata = context["metadata"]
    arrays = {**context["arrays"], "damping": _context_damping(context)}
    for key, value in expected["metadata"].items():
        if key == "schema_version":
            continue
        if metadata.get(key) != value:
            raise ValueError(f"replay context {key} differs from the supplied PhysicalRollout")
    for key, value in expected["arrays"].items():
        if key not in arrays or not np.array_equal(np.asarray(arrays[key]), value):
            raise ValueError(f"replay context {key} differs from the supplied PhysicalRollout")


def retain_windows(
    rollout: PhysicalRollout,
    store: DiskReplayStore,
    positions: Tensor,
    velocities: Tensor,
    *,
    trajectory_ids: Sequence[str],
    physical_steps: int,
    iterations: int = 1,
    gradient_window: int = 1,
    forces: Tensor | None = None,
    fixed_positions: Tensor | None = None,
    step_offset: int = 0,
    provenance: Mapping | Callable[[int], Mapping] | None = None,
    trajectory_metadata: Sequence[Mapping] | None = None,
    history: OptimizerHistory | None = None,
) -> Iterator[PhysicalWindow]:
    """Yield ordinary windows while durably retaining each member before solve.

    ``trajectory_ids`` identify batch members across physical steps. They must
    be stable and distinct within a store; ``step_offset`` supports successive
    calls on the same trajectory. The caller supplies source provenance, such
    as checkpoint hash, epoch/update, and physical seed. A callable provenance
    is evaluated with the absolute step index at each pre-step callback, after
    the caller may have updated weights between yielded windows. It only sees detached copies
    and the persisted arrays are detached CPU snapshots. ``history`` seeds the
    optimizer history when continuing a trajectory across successive calls;
    pass the previous call's last ``steps[-1].history``.
    """
    if not isinstance(rollout, PhysicalRollout):
        raise TypeError("rollout must be a PhysicalRollout")
    if len(trajectory_ids) != positions.shape[0] or len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("trajectory_ids must be distinct and match the batch size")
    if any(not isinstance(value, str) or not value for value in trajectory_ids):
        raise ValueError("trajectory_ids must be nonempty strings")
    if isinstance(step_offset, bool) or not isinstance(step_offset, int) or step_offset < 0:
        raise ValueError("step_offset must be a nonnegative integer")
    if trajectory_metadata is not None and len(trajectory_metadata) != len(trajectory_ids):
        raise ValueError("trajectory_metadata must match the batch size")
    per_member = (
        [dict(item) for item in trajectory_metadata]
        if trajectory_metadata is not None
        else [{} for _ in trajectory_ids]
    )
    payload = physical_context(rollout)
    context_id = store.register_context(**payload)

    def save_step(
        index: int,
        x: Tensor,
        v: Tensor,
        f: Tensor,
        pins: Tensor,
        gravity: Tensor,
        history: OptimizerHistory | None,
    ) -> None:
        snapshots = []
        current_provenance = provenance(step_offset + index) if callable(provenance) else provenance
        shared = dict(current_provenance or {})
        for member, trajectory_id in enumerate(trajectory_ids):
            retained_history = None
            if history is not None and bool(history.valid[member]):
                retained_history = {
                    "axis_gradient_world": _cpu_array(history.axis_gradient_world[member]),
                    "axis_update_world": _cpu_array(history.axis_update_world[member]),
                }
            metadata = {**shared, **per_member[member]}
            metadata.update(
                {
                    "gravity": [float(value) for value in gravity.cpu()],
                    "iterations": iterations,
                    "detach_energy_target": rollout.detach_energy_target,
                    "energy_increase_weight": rollout.unrolled.energy_increase_weight,
                }
            )
            snapshots.append(
                ReplayState(
                    context_id=context_id,
                    trajectory_id=trajectory_id,
                    step_index=step_offset + index,
                    positions=x[member],
                    velocities=v[member],
                    forces=f[member],
                    fixed_positions=pins[member],
                    metadata=metadata,
                    history=retained_history,
                )
            )
        store.append(snapshots)

    windows = rollout.windows(
        positions,
        velocities,
        physical_steps=physical_steps,
        iterations=iterations,
        gradient_window=gradient_window,
        forces=forces,
        fixed_positions=fixed_positions,
        history=history,
        on_step_start=save_step,
    )
    del positions, velocities, forces, fixed_positions
    yield from windows


def build_replay_rollout(
    context: Mapping,
    current_network: IntrinsicSolverNetwork,
    *,
    gravity: Sequence[float],
    max_iterations: int = 32,
    energy_increase_weight: float = 1.0,
    detach_energy_target: bool = True,
    checkpoint_activations: bool = False,
) -> PhysicalRollout:
    """Build native physical metadata and a fresh solver around current weights.

    ``current_network`` is supplied by the caller and is never loaded from the
    source record. Gravity is required explicitly: pass the selected retained
    state's ``metadata['gravity']`` for a faithful physical replay. Loss policy
    may deliberately differ from the source; source policy is in each state.
    The saved native masses override rounding differences in density-derived
    mass after checking that they are physically consistent.
    Schema 1 contexts without damping rebuild zero viscosity; schema 2 requires
    an explicit per-cell coefficient and never supplies a missing default.
    """
    metadata, arrays = context["metadata"], context["arrays"]
    damping = _context_damping(context)
    rest = generate_cuboid(
        tuple(metadata["cell_counts"]),
        cell_size=float(metadata["cell_size"]),
        origin=tuple(metadata["origin"]),
    )
    if not np.array_equal(rest.corner_rest_positions.astype(np.float32), np.asarray(arrays["rest_positions"])):
        raise ValueError("saved rest positions do not match canonical model geometry")
    if not np.array_equal(rest.cell_corner_indices, np.asarray(arrays["cell_corner_indices"])):
        raise ValueError("saved cell topology does not match canonical model geometry")
    model = build_newton_hex_model(
        rest,
        np.asarray(arrays["fixed_indices"], dtype=np.int64),
        lame_lambda=np.asarray(arrays["lame_lambda"], dtype=np.float32),
        lame_mu=np.asarray(arrays["lame_mu"], dtype=np.float32),
        density=np.asarray(arrays["density"], dtype=np.float32),
        damping=np.asarray(damping, dtype=np.float32),
        gravity=tuple(gravity),
    )
    saved_mass = np.asarray(arrays["lumped_mass"], dtype=np.float32)
    if not np.allclose(model.particle_mass.numpy(), saved_mass, rtol=2e-5, atol=0):
        raise ValueError("saved physical masses do not match material and rest geometry")
    model.particle_mass.assign(saved_mass)
    model.particle_inv_mass.assign(np.reciprocal(saved_mass))
    solver = SolverLearnedIntrinsic(model, network=current_network, iterations=1)
    step = solver._step_for_dt(float(metadata["time_step"]))
    unrolled = UnrolledHexSolver(
        step,
        max_iterations=max_iterations,
        energy_increase_weight=energy_increase_weight,
        checkpoint_activations=checkpoint_activations,
    )
    result = PhysicalRollout(
        solver,
        unrolled,
        time_step=float(metadata["time_step"]),
        detach_energy_target=detach_energy_target,
    )
    _verify_context(result, context)
    return result


def replay_state(
    rollout: PhysicalRollout,
    context: Mapping,
    state: ReplayState,
    *,
    iterations: int | None = None,
) -> PhysicalStep:
    """Recompute one saved physical solve using the supplied network's weights.

    The retained optimizer history (None at a trajectory start) seeds the inner
    unroll exactly as it did when the state was recorded.
    """
    if context.get("context_id") != state.context_id:
        raise ValueError("replay state context_id does not match the supplied context")
    _verify_context(rollout, context)
    saved_gravity = np.asarray(state.metadata["gravity"], dtype=np.float32)
    if not np.array_equal(rollout.solver._gravity(), saved_gravity):
        raise ValueError("replay gravity differs from retained physical step")
    count = state.metadata["iterations"] if iterations is None else iterations
    reference = rollout.unrolled.step.rest_positions
    kwargs = {"dtype": reference.dtype, "device": reference.device}
    x = torch.as_tensor(np.asarray(state.positions).copy(), **kwargs)[None]
    v = torch.as_tensor(np.asarray(state.velocities).copy(), **kwargs)[None]
    f = (
        torch.zeros_like(x)
        if state.forces is None
        else torch.as_tensor(np.asarray(state.forces).copy(), **kwargs)[None]
    )
    fixed = context["arrays"]["fixed_indices"]
    pin_values = context["arrays"]["rest_positions"][fixed] if state.fixed_positions is None else state.fixed_positions
    pins = torch.as_tensor(np.asarray(pin_values).copy(), **kwargs)[None]
    history = None
    if state.history is not None:
        history = OptimizerHistory(
            torch.as_tensor(np.asarray(state.history["axis_gradient_world"]).copy(), **kwargs)[None],
            torch.as_tensor(np.asarray(state.history["axis_update_world"]).copy(), **kwargs)[None],
            torch.ones(1, dtype=torch.bool, device=reference.device),
        )
    window = next(
        rollout.windows(
            x,
            v,
            physical_steps=1,
            iterations=count,
            gradient_window=1,
            forces=f,
            fixed_positions=pins,
            history=history,
        )
    )
    return window.steps[0]
