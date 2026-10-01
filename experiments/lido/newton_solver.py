# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LIDO inside Newton (design spec 1b "Newton API", 4.3 `SolverLIDO`): hex bodies as corner particles and a
`SolverBase` that advances them with the learned solver.

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    body = add_hex_body(builder, (10, 10, 40), 0.025, pos=(0, 0.5, 0), material=dict(E=1e5, nu=0.3, rho=1000, eta=100),
                        contact=dict(kappa=100, beta=0.0, mu_f=0.3))
    builder.add_ground_plane()
    model = builder.finalize(device)
    attach_bodies(model, builder)                  # one-liner after finalize: model.lido_bodies = builder.lido_bodies
    solver = SolverLIDO(model, "ckpt.pt", iterations=8)
    solver.step(state_in, state_out, None, None, dt)

Conventions. Inside the solver every body is normalised (h = mu = dt = 1): X' = (q - origin) / h in world axes
(the energy is rotation invariant, so a rotated body is just a rotated X'), V' = v dt / h. Pinned corners are the
particles with zero inverse mass (mass 0 in `add_hex_body`); their `state_in.particle_q / particle_qd` is the
prescribed position and velocity for the step (kinematic pins). A body added with `pins="none"` has every particle
mass positive and is a free body (derivation note section 7): its centroid follows the rigid semi-implicit target of
the current contact force at every query. Contact partners are the model's infinite static
ground plane (shape type PLANE, body -1, zero extent), converted into each body's units, plus optional SI static
points; a body only feels them when it was given a `contact` dict (ke = kappa E h).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch
import warp as wp

import newton
from newton.solvers import SolverBase

from . import scenes
from .batch import Batch
from .capture import CapturedQuery
from .config import TrainConfig
from .fusion import Fusion
from .grid import Grid, GridCache
from .rollout import compile_inference, load_network
from .step import Step
from .structs import ContactScene, Material
from .units import material_from_si

Tensor = torch.Tensor
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


@dataclass
class HexBody:
    """One registered hex block: its particle range in the model and the data the solver needs (SI)."""

    index: int
    key: str | None
    particle_start: int
    particle_count: int
    cell_counts: tuple
    h: float
    pins: str
    material: dict  # E, nu, rho, eta
    contact: dict | None  # kappa, beta, mu_f
    origin: tuple  # world position of the rest corner (0, 0, 0) [m]
    rot: tuple  # quaternion (x, y, z, w) applied to the rest lattice

    @property
    def grid_key(self) -> tuple:
        return (*self.cell_counts, self.pins)


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def add_hex_body(
    builder,
    cell_counts,
    h: float,
    pos=(0.0, 0.0, 0.0),
    rot=IDENTITY_QUAT,
    material: dict | None = None,
    pins: str = "zmin_face",
    contact: dict | None = None,
    key: str | None = None,
) -> int:
    """Add the corner particles of one hex block (world = pos + rot (h rest), pinned corners with mass 0) and
    register it in `builder.lido_bodies`. Returns the body index. `material`: dict(E, nu, rho, eta)."""
    if material is None or not {"E", "nu", "rho", "eta"} <= set(material):
        raise ValueError("material must be a dict with E, nu, rho, eta")
    grid = Grid.build(cell_counts, pins, "cpu")
    rest = grid.rest.numpy().astype(np.float64) * float(h)
    world = np.asarray(pos, dtype=np.float64) + rest @ quat_to_matrix(rot).T
    mass = float(material["rho"]) * float(h) ** 3 * grid.mass.numpy().astype(np.float64)
    mass[grid.pinned_mask.numpy()] = 0.0
    start = builder.particle_count
    builder.add_particles(
        [tuple(float(v) for v in p) for p in world],
        [(0.0, 0.0, 0.0)] * grid.P,
        [float(m) for m in mass],
        radius=[0.5 * float(h)] * grid.P,
    )
    bodies = getattr(builder, "lido_bodies", None)
    if bodies is None:
        bodies = builder.lido_bodies = []
    body = HexBody(
        index=len(bodies),
        key=key,
        particle_start=start,
        particle_count=grid.P,
        cell_counts=tuple(int(v) for v in cell_counts),
        h=float(h),
        pins=pins,
        material={k: float(material[k]) for k in ("E", "nu", "rho", "eta")},
        contact=None if contact is None else {k: float(contact.get(k, 0.0)) for k in ("kappa", "beta", "mu_f")},
        origin=tuple(float(v) for v in pos),
        rot=tuple(float(v) for v in rot),
    )
    bodies.append(body)
    return body.index


def attach_bodies(model, builder) -> list:
    """Copy the body registry from the builder onto the finalised model (`model.lido_bodies`)."""
    model.lido_bodies = list(getattr(builder, "lido_bodies", []))
    return model.lido_bodies


def ground_plane_from_model(model) -> tuple[np.ndarray, np.ndarray] | None:
    """(point, unit normal) of the first infinite static plane shape (PLANE, body -1, zero extents), or None.
    Newton's plane normal is the local +z axis of the shape transform."""
    if getattr(model, "shape_count", 0) == 0 or model.shape_type is None:
        return None
    types = model.shape_type.numpy()
    bodies = model.shape_body.numpy()
    scale = model.shape_scale.numpy()
    xform = model.shape_transform.numpy()
    for i in range(len(types)):
        infinite_static_plane = (
            int(types[i]) == int(newton.GeoType.PLANE)
            and int(bodies[i]) == -1
            and scale[i, 0] == 0.0
            and scale[i, 1] == 0.0
        )
        if infinite_static_plane:
            point = np.asarray(xform[i, :3], dtype=np.float64)
            normal = quat_to_matrix(xform[i, 3:7]) @ np.array([0.0, 0.0, 1.0])
            return point, normal / np.linalg.norm(normal)
    return None


class SolverLIDO(SolverBase):
    """Learned intrinsic solver for the hex bodies registered with `add_hex_body`.

    One Grid per distinct (cell_counts, pins), one flat Batch over all bodies, a Material per body (built at the
    first `step` from its dt, rebuilt when dt or gravity changes), a ContactScene per body in its own units.
    Each step runs K = `iterations` queries of the network and advances; the batch keeps the trajectory between
    steps. Free particles whose `state_in` values differ from what the solver wrote last (first call, a user
    reset) re-initialise X and V from `state_in`.

    `capture` (default: on CUDA) records the query as a CUDA graph at the first step and replays it (the bodies
    are fixed, so every shape is static); a failed capture falls back to eager queries with a warning.
    `compile_network` torch.compiles the cell-graph layer for the static shapes. `translation` selects the centroid
    target of bodies without pins (`pins="none"` in `add_hex_body`): "picard" or "implicit_contact" (step.py).
    """

    def __init__(
        self,
        model,
        checkpoint: str | dict | None = None,
        iterations: int = 8,
        device=None,
        static_points: dict | None = None,
        ground: bool | None = None,
        cfg: TrainConfig | None = None,
        capture: bool | None = None,
        compile_network: bool = False,
        translation: str = "implicit_contact",
    ):
        super().__init__(model=model)
        bodies = getattr(model, "lido_bodies", None)
        if not bodies:
            raise ValueError(
                "model has no LIDO bodies: call add_hex_body(builder, ...) and attach_bodies(model, builder)"
            )
        self.bodies: list[HexBody] = list(bodies)
        self.K = int(iterations)
        self.model_device = torch.device(str(model.device))
        self.torch_device = torch.device(device) if device is not None else self.model_device
        self.net, self.cfg = load_network(checkpoint, cfg, device=self.torch_device)
        self.capture = (self.torch_device.type == "cuda") if capture is None else bool(capture)
        self.captured: CapturedQuery | None = None
        self.lido = Step(self.net, Fusion(), pair_capacity=self.capture, translation=translation)
        if compile_network:
            compile_inference(self.net, self.lido)
        self.grid_cache = GridCache(self.torch_device)
        grids = [self.grid_cache.get(b.cell_counts, b.pins) for b in self.bodies]
        self.batch = Batch.build(grids, self.torch_device)
        dev = self.torch_device
        self.particle_idx = torch.cat(
            [torch.arange(b.particle_start, b.particle_start + b.particle_count, device=dev) for b in self.bodies]
        )
        self.batch.origin = torch.tensor([b.origin for b in self.bodies], dtype=torch.float32, device=dev)
        self.h = torch.tensor([b.h for b in self.bodies], dtype=torch.float32, device=dev)
        self._h_row = self.h[self.batch.corner_obj][:, None]
        self._origin_row = self.batch.origin[self.batch.corner_obj]
        self.all_objects = torch.ones(self.batch.O, dtype=torch.bool, device=dev)
        self.free_rows = ~self.batch.pinned

        inv_mass = wp.to_torch(model.particle_inv_mass).to(dev)[self.particle_idx]
        if not torch.equal(inv_mass == 0, self.batch.pinned):
            raise ValueError("pinned corners (inverse mass 0) do not match the bodies' pin pattern")

        self.gravity = self._gravity_per_body()
        plane = ground_plane_from_model(model) if ground is not False else None
        if ground is True and plane is None:
            raise ValueError("ground=True but the model has no infinite static plane shape")
        self.plane = plane
        self.static_points = static_points
        self.batch.scene = self._build_scene()
        self._dt: float | None = None
        self._q_written: Tensor | None = None
        self._qd_written: Tensor | None = None

    # ----------------------------------------------------------------- setup
    def _gravity_per_body(self) -> np.ndarray:
        g = self.model.gravity.numpy()  # [world_count + 1, 3]; the last row is the global world -1
        worlds = self.model.particle_world.numpy()
        out = np.zeros((len(self.bodies), 3), dtype=np.float64)
        for i, b in enumerate(self.bodies):
            w = int(worlds[b.particle_start])
            out[i] = g[w if w >= 0 else -1]
        return out

    def _build_material(self, dt: float) -> Material:
        parts = []
        for i, b in enumerate(self.bodies):
            grid = self.batch.grids[i]
            c = b.contact or {}
            parts.append(
                material_from_si(
                    E=b.material["E"],
                    nu=b.material["nu"],
                    rho=b.material["rho"],
                    eta=b.material["eta"],
                    gravity=self.gravity[i],
                    h=b.h,
                    dt=dt,
                    cell_count=grid.C,
                    sample_count=grid.S,
                    kappa=c.get("kappa", 0.0),
                    beta=c.get("beta", 0.0),
                    mu_f=c.get("mu_f", 0.0),
                    friction_epsilon=self.cfg.contact_friction_epsilon,
                    floor_scale=self.cfg.energy_floor_scale,
                    device=self.torch_device,
                )
            )
        return Material.cat(parts)

    def _build_scene(self) -> ContactScene:
        dev = self.torch_device
        pts = nrm = rad = None
        if self.static_points is not None:
            pts = np.asarray(self.static_points["points"], dtype=np.float64).reshape(-1, 3)
            nrm = np.asarray(self.static_points["normals"], dtype=np.float64).reshape(-1, 3)
            rad = np.asarray(self.static_points["radii"], dtype=np.float64).reshape(-1)
        parts = []
        for b in self.bodies:
            o = np.asarray(b.origin, dtype=np.float64)
            if self.plane is not None:
                p, n = self.plane
                plane_n, plane_d, present = n, float(n @ (p - o)) / b.h, True
            else:
                plane_n, plane_d, present = np.array(scenes.PLANE_NORMAL), 0.0, False
            t = lambda v, shape: torch.tensor(np.asarray(v, dtype=np.float32), device=dev).reshape(shape)  # noqa: E731
            npts = 0 if pts is None else pts.shape[0]
            parts.append(
                ContactScene(
                    plane_n=t(plane_n, (1, 3)),
                    plane_d=t([plane_d], (1,)),
                    plane_present=torch.tensor([present], device=dev),
                    points=t((pts - o) / b.h if npts else np.zeros((0, 3)), (-1, 3)),
                    normals=t(nrm if npts else np.zeros((0, 3)), (-1, 3)),
                    radii=t(rad / b.h if npts else np.zeros(0), (-1,)),
                    point_offsets=torch.tensor([0, npts], dtype=torch.int64, device=dev),
                )
            )
        return scenes.cat_scenes(parts)

    @property
    def groups(self) -> list:
        return self.batch.groups

    # ------------------------------------------------------------------ Newton
    def notify_model_changed(self, flags) -> None:
        if int(flags) & int(newton.ModelFlags.MODEL_PROPERTIES):
            self.gravity = self._gravity_per_body()
            self._dt = None  # rebuild the material (and re-read the state) at the next step

    def reset(self, state, world_mask=None, flags=None) -> None:
        super().reset(state, world_mask, flags)
        self._q_written = None  # the next step re-initialises X and V from state_in

    def step(self, state_in, state_out, control, contacts, dt: float) -> None:
        dt = float(dt)
        model = self.model
        if state_out is not state_in:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
            if model.body_count:
                wp.copy(state_out.body_q, state_in.body_q)
                wp.copy(state_out.body_qd, state_in.body_qd)
        self._torch_waits_for_warp()
        q_all = wp.to_torch(state_in.particle_q)
        qd_all = wp.to_torch(state_in.particle_qd)
        idx = self.particle_idx
        q_b = q_all.to(self.torch_device)[idx]
        qd_b = qd_all.to(self.torch_device)[idx]
        with torch.no_grad():
            if self._dt != dt:
                self.batch.material = self._build_material(dt)
                self._dt = dt
                self._q_written = None
            X_in = (q_b - self._origin_row) / self._h_row
            V_in = qd_b * (dt / self._h_row)
            if not self._matches_written(q_b, qd_b):
                self.batch.X = X_in.clone()
                self.batch.V = V_in.clone()
                self.batch.X_prev = X_in - V_in
                self.batch.x = X_in.clone()
                self.lido.prepare(self.batch, self.all_objects)
            batch = self.batch
            if self.capture and self.captured is None:
                try:
                    self.captured = CapturedQuery(self.lido, batch)
                except Exception as e:  # any capture failure means: run eagerly
                    warnings.warn(f"SolverLIDO: CUDA-graph capture failed ({e!r}); running eagerly", stacklevel=2)
                    self.capture = False
            if self.captured is not None:
                self.captured.sync()  # the prescribed pins, step constants and pairs of this step
                for _ in range(self.K):
                    self.captured.replay()
            else:
                for _ in range(self.K):
                    out = self.lido.query(batch)
                    self.lido.commit(batch, out)
            self.lido.advance(batch, self.all_objects, prescribed=X_in, prescribed_velocity=V_in)
            pin = batch.pinned[:, None]
            q_out = torch.where(pin, q_b, self._origin_row + self._h_row * batch.X)
            qd_out = torch.where(pin, qd_b, batch.V * (self._h_row / dt))
        wp.to_torch(state_out.particle_q)[idx.to(self.model_device)] = q_out.to(self.model_device)
        wp.to_torch(state_out.particle_qd)[idx.to(self.model_device)] = qd_out.to(self.model_device)
        self._q_written, self._qd_written = q_out, qd_out
        self._warp_waits_for_torch()

    def _matches_written(self, q_b: Tensor, qd_b: Tensor) -> bool:
        if self._q_written is None:
            return False
        f = self.free_rows
        return bool(torch.equal(q_b[f], self._q_written[f]) and torch.equal(qd_b[f], self._qd_written[f]))

    def _torch_waits_for_warp(self) -> None:
        if self.model_device.type == "cuda":
            warp_stream = wp.stream_to_torch(wp.get_stream(self.model.device))
            torch.cuda.current_stream(self.model_device).wait_stream(warp_stream)

    def _warp_waits_for_torch(self) -> None:
        if self.model_device.type == "cuda":
            torch_stream = wp.stream_from_torch(torch.cuda.current_stream(self.model_device))
            wp.get_stream(self.model.device).wait_stream(torch_stream)

    def describe(self) -> list[dict]:
        """Per-body records (SI) for logging."""
        return [asdict(b) for b in self.bodies]
