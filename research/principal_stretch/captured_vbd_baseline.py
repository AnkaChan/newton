# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Captured public-API SolverVBD baseline for MG-VBD research.

This module deliberately measures only a pristine, contact-free particle
substep.  It owns separate ``SolverVBD`` instances and CUDA graphs for K=1
and K=4.  Each graph restores every public particle state array before
calling :meth:`newton.solvers.SolverVBD.step`, so graph replay cannot silently
turn a convergence comparison into a multi-timestep rollout.

The result is a diagnostic baseline.  It contains no multigrid correction and
must not be presented as integrated MG-VBD performance evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import statistics
from collections.abc import Iterable, Sequence

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .solver_benchmark import TetBenchmarkScene, VBDRunResult, run_vbd

CONTRACT_ID = "captured-public-solver-vbd-baseline-v1"
ITERATION_BUDGETS = (1, 4)

MUTATION_HAZARDS = (
    "SolverVBD mutates state_in.particle_q during forward prediction and color sweeps",
    "SolverVBD overwrites state_out.particle_q and state_out.particle_qd",
    "state_in.particle_qd and state_in.particle_f define the inertial target",
    "state_out.particle_f is not an endpoint output and must not retain poison data",
    "K1 and K4 require separate solver-owned scratch and graph state",
    "rigid, contact, and self-contact histories are outside this public reset contract",
)


def _immutable_array(value: np.ndarray | Iterable[float], dtype: np.dtype | type) -> np.ndarray:
    """Return a C-contiguous array backed by immutable bytes."""
    source = np.array(value, dtype=dtype, order="C", copy=True)
    return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(source.shape)


def _array_sha256(value: np.ndarray) -> str:
    """Hash an array using the benchmark manifest's dtype/shape convention."""
    array = np.asarray(value)
    dtype = array.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    contiguous = np.ascontiguousarray(array, dtype=canonical_dtype)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _named_arrays_sha256(tag: str, arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    """Hash a typed sequence of named arrays without ambiguous concatenation."""
    digest = hashlib.sha256()

    def add(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    add(tag.encode("utf-8"))
    for name, array in arrays:
        add(name.encode("utf-8"))
        add(_array_sha256(np.asarray(array)).encode("ascii"))
    return digest.hexdigest()


def _build_public_model(scene: TetBenchmarkScene, device: wp.Device) -> newton.Model:
    """Rebuild one scene using public Newton construction APIs only."""
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particles(
        pos=[wp.vec3(*position) for position in scene.rest_q],
        vel=[wp.vec3(*velocity) for velocity in scene.velocity],
        mass=scene.mass.tolist(),
        flags=scene.particle_flags.tolist(),
    )
    for tet, material in zip(scene.tet_indices, scene.tet_materials, strict=True):
        volume = builder.add_tetrahedron(
            int(tet[0]),
            int(tet[1]),
            int(tet[2]),
            int(tet[3]),
            float(material[0]),
            float(material[1]),
            float(material[2]),
        )
        if volume <= 0.0:
            raise RuntimeError("scene reconstruction produced an inverted rest tetrahedron")
    for triangle, material in zip(scene.tri_indices, scene.tri_materials, strict=True):
        area = builder.add_triangle(
            int(triangle[0]),
            int(triangle[1]),
            int(triangle[2]),
            float(material[0]),
            float(material[1]),
            float(material[2]),
            float(material[3]),
            float(material[4]),
        )
        if area <= 0.0:
            raise RuntimeError("scene reconstruction produced a degenerate boundary triangle")
    groups = [
        scene.color_group_particles[scene.color_group_offsets[index] : scene.color_group_offsets[index + 1]].astype(
            np.int32
        )
        for index in range(scene.color_group_offsets.size - 1)
    ]
    builder.set_coloring(groups)
    model = builder.finalize(device=device)
    model.set_gravity(scene.gravity)

    checks = (
        ("rest positions", model.particle_q.numpy(), scene.rest_q),
        ("masses", model.particle_mass.numpy(), scene.mass),
        ("inverse masses", model.particle_inv_mass.numpy(), scene.particle_inv_mass),
        ("tet topology", model.tet_indices.numpy().reshape(-1, 4), scene.tet_indices),
        ("tet inverse rest poses", model.tet_poses.numpy().reshape(-1, 3, 3), scene.tet_poses),
        ("tet materials", model.tet_materials.numpy().reshape(-1, 3), scene.tet_materials),
        ("particle flags", model.particle_flags.numpy(), scene.particle_flags),
    )
    for name, actual, expected in checks:
        if not np.array_equal(np.asarray(actual).astype(np.asarray(expected).dtype), expected):
            raise RuntimeError(f"public model reconstruction changed {name}")
    actual_triangles = (
        np.empty((0, 3), dtype=np.int64) if model.tri_indices is None else model.tri_indices.numpy().reshape(-1, 3)
    )
    if not np.array_equal(actual_triangles, scene.tri_indices):
        raise RuntimeError("public model reconstruction changed boundary-triangle topology")
    if scene.n_triangles:
        triangle_checks = (
            ("boundary-triangle inverse rest poses", model.tri_poses.numpy().reshape(-1, 2, 2), scene.tri_poses),
            ("boundary-triangle materials", model.tri_materials.numpy().reshape(-1, 5), scene.tri_materials),
            ("boundary-triangle areas", model.tri_areas.numpy(), scene.tri_areas),
        )
        for name, actual, expected in triangle_checks:
            if not np.array_equal(np.asarray(actual).astype(np.asarray(expected).dtype), expected):
                raise RuntimeError(f"public model reconstruction changed {name}")
    expected_groups = [
        scene.color_group_particles[scene.color_group_offsets[index] : scene.color_group_offsets[index + 1]]
        for index in range(scene.color_group_offsets.size - 1)
    ]
    if len(model.particle_color_groups) != len(expected_groups) or any(
        not np.array_equal(actual.numpy(), expected)
        for actual, expected in zip(model.particle_color_groups, expected_groups, strict=True)
    ):
        raise RuntimeError("public model reconstruction changed particle coloring")
    if model.body_count or model.shape_count or model.spring_count or model.edge_count or model.joint_count:
        raise ValueError("captured baseline supports contact-free particle-only models")
    return model


def _make_pristine_input(model: newton.Model, scene: TetBenchmarkScene) -> newton.State:
    """Create the immutable-by-contract physical input state."""
    state = model.state()
    positions = scene.x_current.copy()
    positions[scene.pinned_indices] = scene.pin_targets
    state.clear_forces()
    state.particle_q.assign(wp.array(positions.astype(np.float32), dtype=wp.vec3, device=model.device))
    state.particle_qd.assign(wp.array(scene.velocity.astype(np.float32), dtype=wp.vec3, device=model.device))
    state.particle_f.assign(wp.array(scene.external_force.astype(np.float32), dtype=wp.vec3, device=model.device))
    return state


def _make_pristine_output(model: newton.Model) -> newton.State:
    """Create the exact fresh output state used by uncaptured ``run_vbd``."""
    state = model.state()
    state.clear_forces()
    return state


def _copy_particle_state(destination: newton.State, source: newton.State) -> None:
    """Enqueue the complete public particle-state reset used inside graphs."""
    wp.copy(destination.particle_q, source.particle_q)
    wp.copy(destination.particle_qd, source.particle_qd)
    wp.copy(destination.particle_f, source.particle_f)


def _minimum_determinant(scene: TetBenchmarkScene, positions: np.ndarray) -> float:
    """Compute the minimum deformation determinant without solver internals."""
    corners = np.asarray(positions, dtype=np.float64)[scene.tet_indices]
    edge_matrix = np.stack(
        (corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]),
        axis=2,
    )
    deformation_gradient = edge_matrix @ scene.tet_poses
    determinants = np.linalg.det(deformation_gradient)
    if not np.isfinite(determinants).all():
        raise FloatingPointError("VBD endpoint produced a nonfinite deformation determinant")
    return float(np.min(determinants))


def _public_model_sha256(model: newton.Model) -> str:
    """Synchronously hash the public static model arrays consumed by VBD."""
    triangles = (
        np.empty((0, 3), dtype=np.int32) if model.tri_indices is None else model.tri_indices.numpy().reshape(-1, 3)
    )
    triangle_poses = (
        np.empty((0, 2, 2), dtype=np.float32) if model.tri_poses is None else model.tri_poses.numpy().reshape(-1, 2, 2)
    )
    triangle_materials = (
        np.empty((0, 5), dtype=np.float32)
        if model.tri_materials is None
        else model.tri_materials.numpy().reshape(-1, 5)
    )
    triangle_areas = np.empty((0,), dtype=np.float32) if model.tri_areas is None else model.tri_areas.numpy()
    color_groups = tuple(group.numpy() for group in model.particle_color_groups)
    return _named_arrays_sha256(
        "captured-public-vbd-model-v1",
        (
            ("particle_q", model.particle_q.numpy()),
            ("particle_qd", model.particle_qd.numpy()),
            ("particle_mass", model.particle_mass.numpy()),
            ("particle_inv_mass", model.particle_inv_mass.numpy()),
            ("particle_flags", model.particle_flags.numpy()),
            ("tet_indices", model.tet_indices.numpy().reshape(-1, 4)),
            ("tet_poses", model.tet_poses.numpy().reshape(-1, 3, 3)),
            ("tet_materials", model.tet_materials.numpy().reshape(-1, 3)),
            ("tri_indices", triangles),
            ("tri_poses", triangle_poses),
            ("tri_materials", triangle_materials),
            ("tri_areas", triangle_areas),
            *((f"particle_color_group_{index}", group) for index, group in enumerate(color_groups)),
            ("gravity", model.gravity.numpy()),
        ),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class CapturedVBDEndpoint:
    """One synchronized public SolverVBD endpoint and reset evidence."""

    positions: np.ndarray
    velocities: np.ndarray
    iterations: int
    device: str
    graph_replay: bool
    pristine_state_sha256: str
    position_sha256: str
    position_fp32_sha256: str
    velocity_sha256: str
    endpoint_sha256: str
    max_pin_error_m: float
    minimum_determinant: float
    contract_id: str = CONTRACT_ID
    research_only: bool = True
    diagnostic_baseline: bool = True
    integrated_mg: bool = False
    performance_evidence: bool = False

    def __post_init__(self) -> None:
        positions = _immutable_array(self.positions, np.float64)
        velocities = _immutable_array(self.velocities, np.float64)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions and velocities must have matching (N, 3) shapes")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("captured endpoint must be finite")
        if self.iterations not in ITERATION_BUDGETS:
            raise ValueError("captured endpoint must use the registered K1 or K4 budget")
        if self.max_pin_error_m != 0.0:
            raise ValueError("captured endpoint must preserve pins exactly")
        if not math.isfinite(self.minimum_determinant) or self.minimum_determinant <= 0.0:
            raise ValueError("captured endpoint must not contain an inverted tetrahedron")
        for name in (
            "pristine_state_sha256",
            "position_sha256",
            "position_fp32_sha256",
            "velocity_sha256",
            "endpoint_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.research_only or not self.diagnostic_baseline or self.integrated_mg or self.performance_evidence:
            raise ValueError("this record cannot claim integrated MG-VBD performance")

    def deterministic_record(self) -> dict[str, object]:
        """Return finite JSON-shaped endpoint evidence."""
        return {
            "contract_id": self.contract_id,
            "research_only": self.research_only,
            "diagnostic_baseline": self.diagnostic_baseline,
            "integrated_mg": self.integrated_mg,
            "performance_evidence": self.performance_evidence,
            "iterations": self.iterations,
            "device": self.device,
            "graph_replay": self.graph_replay,
            "pristine_state_sha256": self.pristine_state_sha256,
            "position_sha256": self.position_sha256,
            "position_fp32_sha256": self.position_fp32_sha256,
            "velocity_sha256": self.velocity_sha256,
            "endpoint_sha256": self.endpoint_sha256,
            "max_pin_error_m": self.max_pin_error_m,
            "minimum_determinant": self.minimum_determinant,
        }


@dataclasses.dataclass(frozen=True)
class CapturedVBDTiming:
    """Paired CUDA-event timings for the two diagnostic baseline graphs."""

    pair_orders: tuple[str, ...]
    k1_seconds: tuple[float, ...]
    k4_seconds: tuple[float, ...]
    warmup_replays: int
    random_seed: int
    device: str
    contract_id: str = CONTRACT_ID
    setup_included: bool = False
    transfers_included: bool = False
    integrated_mg: bool = False
    performance_evidence: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_orders", tuple(self.pair_orders))
        object.__setattr__(self, "k1_seconds", tuple(self.k1_seconds))
        object.__setattr__(self, "k4_seconds", tuple(self.k4_seconds))
        count = len(self.pair_orders)
        if count < 2 or len(self.k1_seconds) != count or len(self.k4_seconds) != count:
            raise ValueError("paired timing arrays must have the same length of at least two")
        if any(order not in ("AB", "BA") for order in self.pair_orders):
            raise ValueError("pair orders must be AB or BA")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.k1_seconds + self.k4_seconds):
            raise ValueError("CUDA-event timings must be finite and positive")
        if self.warmup_replays < 1:
            raise ValueError("at least one warmup replay is required")
        if self.setup_included or self.transfers_included or self.integrated_mg or self.performance_evidence:
            raise ValueError("diagnostic timing cannot claim integrated MG-VBD performance")

    @property
    def k1_median_seconds(self) -> float:
        """Median captured K1 reset-plus-step time [s]."""
        return statistics.median(self.k1_seconds)

    @property
    def k4_median_seconds(self) -> float:
        """Median captured K4 reset-plus-step time [s]."""
        return statistics.median(self.k4_seconds)

    def deterministic_record(self) -> dict[str, object]:
        """Serialize the diagnostic paired timing result."""
        return {
            "contract_id": self.contract_id,
            "diagnostic_baseline": True,
            "integrated_mg": self.integrated_mg,
            "performance_evidence": self.performance_evidence,
            "setup_included": self.setup_included,
            "transfers_included": self.transfers_included,
            "device": self.device,
            "warmup_replays": self.warmup_replays,
            "random_seed": self.random_seed,
            "pair_orders": list(self.pair_orders),
            "k1_seconds": list(self.k1_seconds),
            "k4_seconds": list(self.k4_seconds),
            "k1_median_seconds": self.k1_median_seconds,
            "k4_median_seconds": self.k4_median_seconds,
        }


@dataclasses.dataclass(slots=True)
class _VBDLane:
    iterations: int
    solver: SolverVBD
    state_in: newton.State
    state_out: newton.State
    graph: object | None = None
    completed_launches: int = 0


class CapturedPublicVBDBaseline:
    """Persistent public-API K1/K4 SolverVBD baseline.

    The supported scope is intentionally narrow: tetrahedral particles,
    optional zero-energy boundary triangles, no contacts, no self-contact,
    no springs, and no rigid bodies.  Those exclusions make every cross-step
    history either public particle state (explicitly reset) or solver scratch
    that ``SolverVBD.step`` overwrites.  Unsupported histories fail closed.
    """

    def __init__(
        self,
        scene: TetBenchmarkScene,
        *,
        device: str = "cpu",
        tile_solve: bool = False,
    ):
        if not isinstance(scene, TetBenchmarkScene):
            raise TypeError("scene must be a TetBenchmarkScene")
        if not isinstance(tile_solve, bool):
            raise TypeError("tile_solve must be a bool")
        self.scene = scene
        self.device = wp.get_device(device)
        if tile_solve and self.device.is_cuda and scene.n_triangles == 0:
            raise ValueError("CUDA tile solve requires retained boundary triangles")
        self.tile_solve = tile_solve
        self.model = _build_public_model(scene, self.device)
        self.control = self.model.control()
        self.model_sha256 = _public_model_sha256(self.model)
        self.pristine_input = _make_pristine_input(self.model, scene)
        self.pristine_output = _make_pristine_output(self.model)
        self.pristine_state_sha256 = self._record_pristine_state_sha256()
        self.scene_sha256 = str(scene.manifest()["scene_sha256"])
        self._lanes: dict[int, _VBDLane] = {}
        for iterations in ITERATION_BUDGETS:
            self._lanes[iterations] = _VBDLane(
                iterations=iterations,
                solver=SolverVBD(
                    self.model,
                    iterations=iterations,
                    particle_enable_self_contact=False,
                    particle_enable_tile_solve=tile_solve,
                ),
                state_in=self.model.state(),
                state_out=self.model.state(),
            )

    def _lane(self, iterations: int) -> _VBDLane:
        if isinstance(iterations, bool) or not isinstance(iterations, numbers.Integral):
            raise ValueError("iterations must be K1 or K4")
        try:
            return self._lanes[int(iterations)]
        except KeyError as error:
            raise ValueError("iterations must be K1 or K4") from error

    def _record_pristine_state_sha256(self) -> str:
        """Synchronously hash every persistent graph-reset source array."""
        return _named_arrays_sha256(
            "captured-vbd-pristine-particle-state-v1",
            (
                ("input_q", self.pristine_input.particle_q.numpy()),
                ("input_qd", self.pristine_input.particle_qd.numpy()),
                ("input_f", self.pristine_input.particle_f.numpy()),
                ("output_q", self.pristine_output.particle_q.numpy()),
                ("output_qd", self.pristine_output.particle_qd.numpy()),
                ("output_f", self.pristine_output.particle_f.numpy()),
            ),
        )

    def _enqueue_reset_and_step(self, lane: _VBDLane) -> None:
        _copy_particle_state(lane.state_in, self.pristine_input)
        _copy_particle_state(lane.state_out, self.pristine_output)
        lane.solver.step(lane.state_in, lane.state_out, self.control, None, self.scene.dt)

    def capture_graphs(self, *, warmup_replays: int = 1) -> None:
        """Warm and capture separate reset-plus-step graphs for K1 and K4."""
        if not self.device.is_cuda:
            raise RuntimeError("CUDA graph capture requires a CUDA device")
        if isinstance(warmup_replays, bool) or not isinstance(warmup_replays, numbers.Integral) or warmup_replays < 1:
            raise ValueError("warmup_replays must be a positive integer")
        for lane in self._lanes.values():
            for _ in range(int(warmup_replays)):
                self._enqueue_reset_and_step(lane)
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                self._enqueue_reset_and_step(lane)
            lane.graph = capture.graph

    def run(self, iterations: int, *, graph_replay: bool = False) -> CapturedVBDEndpoint:
        """Run one pristine lane and synchronously record its endpoint."""
        lane = self._lane(iterations)
        if graph_replay:
            if lane.graph is None:
                raise RuntimeError("capture_graphs() must complete before graph replay")
            wp.capture_launch(lane.graph)
        else:
            self._enqueue_reset_and_step(lane)
        lane.completed_launches += 1
        return self.record(iterations, graph_replay=graph_replay)

    def record(self, iterations: int, *, graph_replay: bool) -> CapturedVBDEndpoint:
        """Synchronize and retain a previously launched lane endpoint."""
        lane = self._lane(iterations)
        if lane.completed_launches < 1:
            raise RuntimeError("the requested lane has not executed")
        if not isinstance(graph_replay, bool):
            raise TypeError("graph_replay must be a bool")
        if _public_model_sha256(self.model) != self.model_sha256:
            raise RuntimeError("public static model changed after baseline construction")
        current_pristine_sha256 = self._record_pristine_state_sha256()
        if current_pristine_sha256 != self.pristine_state_sha256:
            raise RuntimeError("persistent pristine input state was mutated")
        positions_fp32 = np.asarray(lane.state_out.particle_q.numpy(), dtype=np.float32)
        velocities_fp32 = np.asarray(lane.state_out.particle_qd.numpy(), dtype=np.float32)
        input_positions_fp32 = np.asarray(lane.state_in.particle_q.numpy(), dtype=np.float32)
        input_velocities_fp32 = np.asarray(lane.state_in.particle_qd.numpy(), dtype=np.float32)
        input_force = np.asarray(lane.state_in.particle_f.numpy(), dtype=np.float32)
        pristine_velocity = np.asarray(self.pristine_input.particle_qd.numpy(), dtype=np.float32)
        pristine_force = np.asarray(self.pristine_input.particle_f.numpy(), dtype=np.float32)
        output_force = np.asarray(lane.state_out.particle_f.numpy(), dtype=np.float32)
        pristine_output_force = np.asarray(self.pristine_output.particle_f.numpy(), dtype=np.float32)
        if not np.array_equal(input_positions_fp32, positions_fp32):
            raise RuntimeError("SolverVBD input/output endpoint positions disagree")
        if not np.array_equal(input_velocities_fp32, pristine_velocity):
            raise RuntimeError("state_in.particle_qd retained data across reset-plus-step")
        if not np.array_equal(input_force, pristine_force):
            raise RuntimeError("state_in.particle_f retained data across reset-plus-step")
        if not np.array_equal(output_force, pristine_output_force):
            raise RuntimeError("state_out.particle_f retained data across reset-plus-step")
        positions = positions_fp32.astype(np.float64)
        velocities = velocities_fp32.astype(np.float64)
        pin_error = (
            float(np.max(np.linalg.norm(positions[self.scene.pinned_indices] - self.scene.pin_targets, axis=1)))
            if self.scene.pinned_indices.size
            else 0.0
        )
        position_sha256 = _array_sha256(positions)
        position_fp32_sha256 = _array_sha256(positions_fp32)
        velocity_sha256 = _array_sha256(velocities)
        endpoint_sha256 = _named_arrays_sha256(
            "captured-public-vbd-endpoint-v1",
            (("positions", positions), ("velocities", velocities)),
        )
        return CapturedVBDEndpoint(
            positions=positions,
            velocities=velocities,
            iterations=int(iterations),
            device=str(self.device),
            graph_replay=graph_replay,
            pristine_state_sha256=self.pristine_state_sha256,
            position_sha256=position_sha256,
            position_fp32_sha256=position_fp32_sha256,
            velocity_sha256=velocity_sha256,
            endpoint_sha256=endpoint_sha256,
            max_pin_error_m=pin_error,
            minimum_determinant=_minimum_determinant(self.scene, positions),
        )

    def poison_lane(self, iterations: int, *, seed: int) -> None:
        """Overwrite public lane state to test that the next reset is complete."""
        lane = self._lane(iterations)
        rng = np.random.default_rng(seed)
        count = self.scene.n_vertices
        for state in (lane.state_in, lane.state_out):
            state.particle_q.assign(rng.normal(loc=11.0, scale=3.0, size=(count, 3)).astype(np.float32))
            state.particle_qd.assign(rng.normal(loc=-7.0, scale=2.0, size=(count, 3)).astype(np.float32))
            state.particle_f.assign(rng.normal(loc=5.0, scale=4.0, size=(count, 3)).astype(np.float32))

    def validate_against_run_vbd(
        self,
        iterations: int,
        *,
        graph_replay: bool,
    ) -> tuple[CapturedVBDEndpoint, VBDRunResult]:
        """Run this baseline and require bitwise agreement with public ``run_vbd``."""
        endpoint = self.run(iterations, graph_replay=graph_replay)
        reference = run_vbd(
            self.scene,
            int(iterations),
            device=str(self.device),
            tile_solve=self.tile_solve,
            warmup=False,
            repeats=1,
        )
        if not np.array_equal(endpoint.positions, reference.positions):
            raise RuntimeError("captured baseline positions differ from uncaptured public run_vbd")
        if not np.array_equal(endpoint.velocities, reference.velocities):
            raise RuntimeError("captured baseline velocities differ from uncaptured public run_vbd")
        if endpoint.position_sha256 != reference.result_state_sha256:
            raise RuntimeError("captured and uncaptured public position hashes differ")
        reference_fp32_sha256 = _array_sha256(reference.positions.astype(np.float32))
        if endpoint.position_fp32_sha256 != reference_fp32_sha256:
            raise RuntimeError("captured and uncaptured public fp32 position hashes differ")
        return endpoint, reference

    def benchmark_paired(
        self,
        *,
        pair_count: int = 10,
        warmup_replays: int = 5,
        random_seed: int = 20260817,
    ) -> CapturedVBDTiming:
        """Measure captured K1/K4 graphs with balanced randomized AB/BA order."""
        if not self.device.is_cuda:
            raise RuntimeError("paired CUDA-event timing requires a CUDA device")
        if any(lane.graph is None for lane in self._lanes.values()):
            raise RuntimeError("capture_graphs() must complete before timing")
        for name, value, minimum in (
            ("pair_count", pair_count, 2),
            ("warmup_replays", warmup_replays, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(random_seed, bool) or not isinstance(random_seed, numbers.Integral):
            raise ValueError("random_seed must be an integer")

        for _ in range(int(warmup_replays)):
            wp.capture_launch(self._lanes[1].graph)
            wp.capture_launch(self._lanes[4].graph)
        wp.synchronize_device(self.device)

        orders = ["AB" if index % 2 == 0 else "BA" for index in range(int(pair_count))]
        np.random.default_rng(int(random_seed)).shuffle(orders)
        event_pairs = {
            iterations: [
                (wp.Event(self.device, enable_timing=True), wp.Event(self.device, enable_timing=True)) for _ in orders
            ]
            for iterations in ITERATION_BUDGETS
        }
        for pair_index, order in enumerate(orders):
            iteration_order = (1, 4) if order == "AB" else (4, 1)
            for iterations in iteration_order:
                begin, end = event_pairs[iterations][pair_index]
                wp.record_event(begin)
                wp.capture_launch(self._lanes[iterations].graph)
                wp.record_event(end)
                self._lanes[iterations].completed_launches += 1
        last_iterations = 4 if orders[-1] == "AB" else 1
        wp.synchronize_event(event_pairs[last_iterations][-1][1])

        seconds = {
            iterations: tuple(
                float(wp.get_event_elapsed_time(begin, end, synchronize=False)) * 1.0e-3
                for begin, end in event_pairs[iterations]
            )
            for iterations in ITERATION_BUDGETS
        }
        return CapturedVBDTiming(
            pair_orders=tuple(orders),
            k1_seconds=seconds[1],
            k4_seconds=seconds[4],
            warmup_replays=int(warmup_replays),
            random_seed=int(random_seed),
            device=str(self.device),
        )

    def deterministic_record(self) -> dict[str, object]:
        """Serialize scope, immutable identity, and reset hazards."""
        return {
            "contract_id": CONTRACT_ID,
            "research_only": True,
            "diagnostic_baseline": True,
            "integrated_mg": False,
            "performance_evidence": False,
            "scene_sha256": self.scene_sha256,
            "model_sha256": self.model_sha256,
            "pristine_state_sha256": self.pristine_state_sha256,
            "device": str(self.device),
            "iteration_budgets": list(ITERATION_BUDGETS),
            "requested_tile_solve": self.tile_solve,
            "effective_tile_solve": bool(self.tile_solve and self.device.is_cuda),
            "control_scope": "persistent read-only empty particle control",
            "reset_arrays": [
                "state_in.particle_q",
                "state_in.particle_qd",
                "state_in.particle_f",
                "state_out.particle_q",
                "state_out.particle_qd",
                "state_out.particle_f",
            ],
            "mutation_hazards": list(MUTATION_HAZARDS),
            "unsupported_histories": ["rigid", "contacts", "particle self-contact"],
        }
