# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The authored 52-card cloth deck with fixed supports and no moving ball."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

__all__ = ["PokerCardsScene", "build_scene"]

_SCENE_ID = "s4_poker_cards_no_ball"
_CARD_COLORS = (
    (68 / 255, 119 / 255, 170 / 255),
    (102 / 255, 204 / 255, 238 / 255),
    (34 / 255, 136 / 255, 51 / 255),
    (204 / 255, 187 / 255, 68 / 255),
    (238 / 255, 102 / 255, 119 / 255),
    (170 / 255, 51 / 255, 119 / 255),
    (238 / 255, 153 / 255, 51 / 255),
    (0 / 255, 153 / 255, 136 / 255),
)


class PokerCardsScene:
    """Let all cards settle freely, then observe passive motion for 60 seconds.

    Geometry, masses, materials, seed and collision settings come from
    ``newton/examples/cloth/example_cloth_poker_cards.py``. The ball is absent.
    The source's massless platform body becomes an equivalent world-fixed box.
    The benchmark uses ten substeps instead of the source's twenty.
    """

    def __init__(self, name: str, *, device=None, use_cuda_graph=False, substeps=10, iterations=10):
        if name != _SCENE_ID:
            raise ValueError(f"Unknown poker-card scene: {name}")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or not 1 <= substeps <= 10:
            raise ValueError("substeps must be an integer from 1 to 10")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if use_cuda_graph and substeps % 2:
            raise ValueError("CUDA graph reuse requires an even number of double-buffer swaps")

        self.scene_id = name
        self.time = 0.0
        self._frame = 0
        self.frame_dt = 1.0 / 60.0
        self.substeps = substeps
        self.iterations = iterations
        self.dt = self.frame_dt / self.substeps
        self.duration = 65.0
        self.title = "S4 · poker cards without the ball"
        self.description = (
            "52 free cloth cards settle on a fixed cube and ground. "
            "After a 5 s settling allowance, passive observation lasts 60 s. "
            "There is no ball, prescribed motion, pinned card vertex or interactive force."
        )
        self.camera_pos = (0.5, -0.5, 0.3)
        pitch, yaw = np.deg2rad([-15.0, 140.0])
        direction = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
        self.camera_target = tuple(np.asarray(self.camera_pos) + direction)
        self.camera_fov = 70.0
        self.objects = [(35 * card, 35 * (card + 1)) for card in range(52)]

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        cube_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5e6, kd=1e4, mu=0.1)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=cube_cfg,
        )
        rng = np.random.default_rng(42)
        for card in range(52):
            offset_x = rng.uniform(-0.005, 0.005)
            offset_y = rng.uniform(-0.005, 0.005)
            angle = rng.uniform(-0.1, 0.1)
            builder.add_cloth_grid(
                pos=wp.vec3(-0.0635 / 2 + offset_x, -0.0889 / 2 + offset_y, 0.25 + card * 0.001),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
                vel=wp.vec3(0.0),
                dim_x=4,
                dim_y=6,
                cell_x=0.0635 / 4,
                cell_y=0.0889 / 6,
                mass=0.0018 / 35,
                fix_left=False,
                fix_right=False,
                fix_top=False,
                fix_bottom=False,
                tri_ke=1e4,
                tri_ka=1e4,
                tri_kd=1.0,
                edge_ke=100.0,
                edge_kd=1.0,
                particle_radius=0.003,
                color=_CARD_COLORS[card % len(_CARD_COLORS)],
            )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=1e5, kd=100.0, mu=0.3))
        builder.color(include_bending=True)
        self.model = builder.finalize(device=device)
        self.initial_positions = self.model.particle_q.numpy().copy()
        self.model.soft_contact_ke = 1e5
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_mu = 0.3
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.001,
            particle_self_contact_gap=0.0005,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.0,
            rigid_body_particle_contact_buffer_size=1024,
            enable_fast_math=False,
        )
        self._state0 = self.model.state()
        self._state1 = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=0.005, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self._graph = None
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            self._graph = capture.graph

    @property
    def state(self):
        """Return the current state after double-buffer swaps."""
        return self._state0

    @property
    def phase(self):
        """Name the observation schedule without claiming that cards settled."""
        return "initial settling allowance" if self.time < 5.0 else "stationary observation"

    def _simulate(self):
        for _ in range(self.substeps):
            self._state0.clear_forces()
            self.pipeline.collide(self._state0, self.contacts)
            self.solver.step(self._state0, self._state1, self.control, self.contacts, self.dt)
            self._state0, self._state1 = self._state1, self._state0

    def step(self):
        """Advance exactly one display frame using the declared substeps."""
        if self._graph is None:
            self._simulate()
        else:
            wp.capture_launch(self._graph)
        self._frame += 1
        self.time = self._frame * self.frame_dt

    def render(self, viewer):
        """Render the source card colors and both supports without extra forces."""
        viewer.begin_frame(self.time)
        viewer.log_state(self.state)
        viewer.end_frame()

    def metrics(self):
        """Describe the procedural deck and authored versus active contact values."""
        return {
            "scene_id": self.scene_id,
            "time_s": self.time,
            "phase": self.phase,
            "source_example": "newton/examples/cloth/example_cloth_poker_cards.py",
            "card_count": 52,
            "seed": 42,
            "card_size_m": [0.0635, 0.0889],
            "card_mass_kg": 0.0018,
            "particle_count": self.model.particle_count,
            "triangle_count": self.model.tri_count,
            "particle_radius_m": 0.003,
            "body_count": self.model.body_count,
            "shape_count": self.model.shape_count,
            "boundary_conditions": "All card vertices free; world-fixed cube and ground; no ball or motion driver.",
            "platform_representation": "World box at (0, 0, 0.1) m with 0.1 m half-extents replaces source massless body.",
            "source_substeps": 20,
            "frame_dt_s": self.frame_dt,
            "substeps": self.substeps,
            "iterations": self.iterations,
            "stationary_observation_interval_s": [5.0, 65.0],
            "authored_shape_contact_ke": [5e6, 1e5],
            "authored_soft_contact_ke": 1e5,
            "physics_parameters": {
                "gravity_m_s2": [0.0, 0.0, -9.8],
                "triangle_ke_ka_kd": [1e4, 1e4, 1.0],
                "bending_ke_kd": [100.0, 1.0],
                "self_contact_margin_m": 0.001,
                "self_contact_gap_m": 0.0005,
                "soft_contact_gap_m": 0.005,
                "soft_contact_ke": float(self.model.soft_contact_ke),
                "soft_contact_kd": float(self.model.soft_contact_kd),
                "soft_contact_mu": float(self.model.soft_contact_mu),
                "shape_contact_ke": self.model.shape_material_ke.numpy().tolist(),
                "shape_contact_kd": self.model.shape_material_kd.numpy().tolist(),
                "shape_contact_mu": self.model.shape_material_mu.numpy().tolist(),
            },
        }


def build_scene(name: str, *, device=None, use_cuda_graph=False, substeps=10, iterations=10):
    """Build the unforced poker deck on the requested Warp device."""
    return PokerCardsScene(name, device=device, use_cuda_graph=use_cuda_graph, substeps=substeps, iterations=iterations)
