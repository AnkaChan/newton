# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle pinning system for the GL viewer.

Toggle pin mode with 'P'. Middle-click to pin/unpin particles.
Pinned particles become kinematic (zero mass, zero velocity).
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from ..geometry.raycast import ray_intersect_particle_sphere


@wp.kernel
def _raycast_particles_kernel(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    ray_start: wp.vec3,
    ray_dir: wp.vec3,
    lock: wp.array[wp.int32],
    min_dist: wp.array[float],
    min_index: wp.array[wp.int32],
):
    """Raycast against all particles, find closest hit."""
    i = wp.tid()
    if not (particle_flags[i] & 1):  # ParticleFlags.ACTIVE
        # Still allow picking inactive (pinned) particles for unpinning
        pass

    dist = ray_intersect_particle_sphere(ray_start, ray_dir, particle_q[i], particle_radius[i])
    if dist < 0.0:
        return

    # Spinlock for thread-safe min update
    while wp.atomic_cas(lock, 0, 0, 1) != 0:
        pass
    wp.atomic_thread_fence()

    if dist < min_dist[0]:
        min_dist[0] = dist
        min_index[0] = i

    wp.atomic_thread_fence()
    wp.atomic_store(lock, 0, 0)


@wp.kernel
def _set_pinned_color(
    pinned: wp.array[wp.int32],
    base_color: wp.vec3,
    pin_color: wp.vec3,
    colors: wp.array[wp.vec3],
):
    i = wp.tid()
    if pinned[i]:
        colors[i] = pin_color
    else:
        colors[i] = base_color


class ParticlePinning:
    """Interactive particle pinning for the GL viewer.

    When pin mode is active, middle-click toggles pin state on particles.
    Pinned particles are made kinematic by zeroing their mass and velocity.
    """

    def __init__(self, model: newton.Model):
        self.model = model
        self.device = model.device
        self.active = False  # pin mode on/off

        N = model.particle_count
        if N == 0:
            return

        # Track which particles are pinned
        self.pinned = wp.zeros(N, dtype=wp.int32, device=self.device)

        # Store original masses for unpinning
        self.original_mass = model.particle_mass.numpy().copy()

        # Raycast scratch buffers
        self.min_dist = wp.array([1.0e10], dtype=float, device=self.device)
        self.min_index = wp.array([-1], dtype=wp.int32, device=self.device)
        self.lock = wp.array([0], dtype=wp.int32, device=self.device)

        # Colors
        self._base_color = wp.vec3(0.7, 0.6, 0.4)
        self._pin_color = wp.vec3(1.0, 0.2, 0.2)
        self._colors = wp.full(N, value=self._base_color, device=self.device)
        self._colors_dirty = False

        self._pin_count = 0

        # Dragging state
        self._dragging_particle = -1
        self._drag_depth = 0.0

    def toggle_mode(self):
        """Toggle pin mode on/off."""
        self.active = not self.active
        return self.active

    def try_pick(self, state: newton.State, ray_start: wp.vec3, ray_dir: wp.vec3) -> int:
        """Raycast to find closest particle. Returns particle index or -1."""
        N = self.model.particle_count
        if N == 0:
            return -1

        self.min_dist.fill_(1.0e10)
        self.min_index.fill_(-1)
        self.lock.zero_()

        wp.launch(
            _raycast_particles_kernel,
            dim=N,
            inputs=[
                state.particle_q,
                self.model.particle_radius,
                self.model.particle_flags,
                ray_start,
                ray_dir,
                self.lock,
                self.min_dist,
                self.min_index,
            ],
            device=self.device,
        )
        wp.synchronize()

        idx = int(self.min_index.numpy()[0])
        dist = float(self.min_dist.numpy()[0])
        if idx >= 0 and dist < 1.0e10:
            self._drag_depth = dist
            return idx
        return -1

    def toggle_pin(self, particle_idx: int, state: newton.State):
        """Toggle pin state of a particle."""
        if particle_idx < 0 or particle_idx >= self.model.particle_count:
            return

        pinned_np = self.pinned.numpy()
        mass_np = self.model.particle_mass.numpy()
        flags_np = self.model.particle_flags.numpy()

        if pinned_np[particle_idx]:
            # Unpin: restore mass, set active
            pinned_np[particle_idx] = 0
            self._pin_count -= 1
            mass_np[particle_idx] = self.original_mass[particle_idx]
            flags_np[particle_idx] |= 1  # ACTIVE
            # Zero velocity on unpin to prevent explosion
            qd = state.particle_qd.numpy()
            qd[particle_idx] = [0.0, 0.0, 0.0]
            state.particle_qd.assign(wp.array(qd, dtype=wp.vec3, device=self.device))
        else:
            # Pin: zero mass, clear active, zero velocity
            pinned_np[particle_idx] = 1
            self._pin_count += 1
            mass_np[particle_idx] = 0.0
            flags_np[particle_idx] &= ~1  # clear ACTIVE
            qd = state.particle_qd.numpy()
            qd[particle_idx] = [0.0, 0.0, 0.0]
            state.particle_qd.assign(wp.array(qd, dtype=wp.vec3, device=self.device))

        self.pinned.assign(wp.array(pinned_np, dtype=wp.int32, device=self.device))
        self.model.particle_mass.assign(wp.array(mass_np, dtype=float, device=self.device))
        self.model.particle_flags.assign(wp.array(flags_np, dtype=wp.int32, device=self.device))
        self._colors_dirty = True

    def start_drag(self, particle_idx: int):
        """Begin dragging a pinned particle."""
        pinned_np = self.pinned.numpy()
        if particle_idx >= 0 and pinned_np[particle_idx]:
            self._dragging_particle = particle_idx

    def update_drag(self, state: newton.State, ray_start: wp.vec3, ray_dir: wp.vec3):
        """Move dragged particle to ray position at stored depth."""
        idx = self._dragging_particle
        if idx < 0:
            return

        # Project ray to drag depth
        target = wp.vec3(
            ray_start[0] + ray_dir[0] * self._drag_depth,
            ray_start[1] + ray_dir[1] * self._drag_depth,
            ray_start[2] + ray_dir[2] * self._drag_depth,
        )

        q = state.particle_q.numpy()
        q[idx] = [float(target[0]), float(target[1]), float(target[2])]
        state.particle_q.assign(wp.array(q, dtype=wp.vec3, device=self.device))

    def release_drag(self):
        """Stop dragging."""
        self._dragging_particle = -1

    def is_dragging(self) -> bool:
        return self._dragging_particle >= 0

    def has_pinned(self) -> bool:
        """Return True if any particles are currently pinned."""
        return self._pin_count > 0

    def get_colors(self) -> wp.array | None:
        """Return particle colors if any are pinned, else None."""
        N = self.model.particle_count
        if N == 0:
            return None
        if not self._colors_dirty and not self.has_pinned():
            return None

        wp.launch(
            _set_pinned_color,
            dim=N,
            inputs=[self.pinned, self._base_color, self._pin_color, self._colors],
            device=self.device,
        )
        self._colors_dirty = False
        return self._colors
