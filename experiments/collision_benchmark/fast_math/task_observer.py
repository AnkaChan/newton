# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cheap sampled task evidence; no contact solve, force estimate or fidelity score.

A positive distance between conservative bounds establishes separation at that
sample. Bounds overlap does not establish contact, acquisition or retention.
"""

from __future__ import annotations

import itertools

import numpy as np


def array(value):
    if value is None:
        return np.empty(0)
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def rotate(quat, points):
    quat = np.asarray(quat, dtype=float)
    xyz = quat[..., :3]
    twice = 2 * np.cross(xyz, points)
    return points + quat[..., 3:4] * twice + np.cross(xyz, twice)


def corners(lo, hi):
    return np.asarray(list(itertools.product(*zip(lo, hi, strict=True))), dtype=float)


def separation(a, b):
    """Euclidean distance of two closed AABBs; zero means inconclusive overlap."""
    return float(np.linalg.norm(np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0)))


class TaskObserver:
    """Sample once per display frame, called after every physical substep."""

    def __init__(self, scene):
        self.scene = scene
        self.raw = getattr(scene, "raw_scene", scene)
        self.model = scene.model
        self.protocol = scene.protocol
        self.rows = []
        self.warnings = []
        self.next_frame = 0
        self.dt = float(scene.frame_dt)
        self.enabled = self.protocol.get("family") == "Robot manipulation"
        self.shape_body = array(self.model.shape_body).astype(int)
        self.shape_type = array(self.model.shape_type).astype(int)
        self.shape_scale = array(self.model.shape_scale)
        self.shape_xform = array(self.model.shape_transform)
        self.shape_margin = array(getattr(self.model, "shape_margin", None))
        self.radius = array(self.model.particle_radius)
        self.mass = array(self.model.particle_mass)
        self.shape_corners = {}
        self.hands = []
        self.components = []
        self.cable_shapes = []
        self.supports = []
        if not self.enabled:
            return
        specs = []
        for name in ("specs", "hands"):
            candidate = getattr(self.raw, name, None)
            if candidate:
                specs = list(candidate)
                break
        if not specs and hasattr(self.raw, "hand"):
            specs = [self.raw.hand]
        for spec in specs:
            ids = list(map(int, spec.shape_ids))
            self.hands.append({"label": getattr(spec, "label", str(len(self.hands))), "shape_ids": ids})
        if not self.hands and self.protocol.get("case_id") == "r2_franka_duck":
            labels = list(self.model.body_label)
            ids = [
                i
                for i, body in enumerate(self.shape_body)
                if body >= 0 and any(name in labels[body].lower() for name in ("hand", "finger"))
            ]
            if ids:
                self.hands.append({"label": "original_franka_hand_by_body_label", "shape_ids": ids})
        if not self.hands:
            self.warnings.append("No original robot shape group was identified; separation is unavailable.")
        for hand in self.hands:
            for shape in hand["shape_ids"]:
                self._shape_box(shape)
        bag = getattr(self.raw, "bag", None)
        if bag is not None and hasattr(bag, "particle_start"):
            self.components.append(("bag", int(bag.particle_start), int(bag.particle_end)))
            for i, asset in enumerate(getattr(self.raw, "payloads", [])):
                self.components.append((f"payload_{i}", int(asset.particle_start), int(asset.particle_end)))
        elif len(self.radius):
            self.components.append(("whole_particle_object", 0, len(self.radius)))
        cable_bodies = self.protocol.get("cable_body_indices", getattr(self.raw, "cable_bodies", []))
        self.cable_shapes = np.flatnonzero(np.isin(self.shape_body, cable_bodies)).tolist()
        for shape in self.cable_shapes:
            self._shape_box(shape)
        for surface in self.protocol.get("penetration_surfaces", []):
            if surface["kind"] == "plane" and np.array_equal(surface["normal"], [0, 0, 1]):
                z = float(surface["offset_m"])
            elif surface["kind"] == "box" and np.array_equal(surface.get("rotation_xyzw", [0, 0, 0, 1]), [0, 0, 0, 1]):
                z = float(surface["center_m"][2] + surface["half_extents_m"][2])
            else:
                continue
            self.supports.append({"label": surface.get("label", "support"), "top_z_m": z})
        self.sample(scene.state, 0.0, scene.phase)

    def _shape_box(self, shape):
        kind, scale = self.shape_type[shape], self.shape_scale[shape]
        if kind in (8, 10):
            source = self.model.shape_source[shape]
            vertices = array(source.vertices) * scale
            lo, hi = vertices.min(0), vertices.max(0)
        elif kind == 3:
            hi = np.repeat(scale[0], 3)
            lo = -hi
        elif kind == 4:
            hi = np.asarray([scale[0], scale[0], scale[0] + scale[1]])
            lo = -hi
        elif kind in (6, 9):
            hi = np.asarray([scale[0], scale[0], scale[1]])
            lo = -hi
        elif kind in (5, 7):
            hi, lo = np.abs(scale), -np.abs(scale)
        else:
            raise ValueError(f"Task bounds do not support robot/cable shape {shape}, type {kind}")
        # Shape-margin expansion bounds the contact offset too. Detection gap is excluded.
        margin = float(self.shape_margin[shape]) if len(self.shape_margin) else 0.0
        local = corners(lo - margin, hi + margin)
        xf = self.shape_xform[shape]
        self.shape_corners[shape] = rotate(xf[3:], local) + xf[:3]

    def _bounds(self, shapes, poses):
        points = []
        for shape in shapes:
            local = self.shape_corners[shape]
            body = self.shape_body[shape]
            points.append(local if body < 0 else rotate(poses[body, 3:], local) + poses[body, :3])
        q = np.concatenate(points)
        return q.min(0), q.max(0)

    def sample(self, state, time_s, phase):
        if not self.enabled or float(time_s) + 1e-9 < self.next_frame * self.dt:
            return
        self.next_frame = int(round(float(time_s) / self.dt)) + 1
        q, poses = array(state.particle_q).astype(float), array(state.body_q).astype(float)
        row = {
            "time_s": float(time_s),
            "phase": str(phase),
            "finite": bool(np.isfinite(q).all() and np.isfinite(poses).all()),
        }
        if not row["finite"]:
            self.rows.append(row)
            return
        hands = [self._bounds(hand["shape_ids"], poses) for hand in self.hands]
        row["hand_bounds_m"] = [[lo.tolist(), hi.tolist()] for lo, hi in hands]
        components = []
        all_lo, all_hi = [], []
        for label, start, end in self.components:
            points, radius = q[start:end], self.radius[start:end, None]
            lo, hi = (points - radius).min(0), (points + radius).max(0)
            masses = self.mass[start:end]
            com = np.sum(points * masses[:, None], axis=0) / masses.sum()
            components.append(
                {
                    "label": label,
                    "bounds_m": [lo.tolist(), hi.tolist()],
                    "center_of_mass_m": com.tolist(),
                    "minimum_shell_z_m": float(lo[2]),
                    "support_plane_clearance_m": [float(lo[2] - s["top_z_m"]) for s in self.supports],
                }
            )
            all_lo.append(lo)
            all_hi.append(hi)
        if self.cable_shapes:
            lo, hi = self._bounds(self.cable_shapes, poses)
            components.append(
                {
                    "label": "whole_cable_arrangement",
                    "bounds_m": [lo.tolist(), hi.tolist()],
                    "minimum_shell_z_lower_bound_m": float(lo[2]),
                    "support_plane_clearance_lower_bound_m": [float(lo[2] - s["top_z_m"]) for s in self.supports],
                }
            )
            all_lo.append(lo)
            all_hi.append(hi)
        row["components"] = components
        if all_lo:
            whole = np.min(all_lo, axis=0), np.max(all_hi, axis=0)
            row["whole_object_bounds_m"] = [whole[0].tolist(), whole[1].tolist()]
            gaps = [separation(whole, hand) for hand in hands]
            row["hand_separation_lower_bound_m"] = gaps
            row["fully_separate_from_all_hands"] = bool(min(gaps) > 0) if gaps else None
        self.rows.append(row)

    def result(self):
        start, end = self.protocol.get("hold_start"), self.protocol.get("hold_end")
        hold = [r for r in self.rows if start is not None and end is not None and start <= r["time_s"] <= end]
        separated = [r for r in hold if r.get("fully_separate_from_all_hands") is True]
        suffix = []
        for row in reversed(self.rows):
            if not row.get("fully_separate_from_all_hands"):
                break
            suffix.append(row)
        suffix.reverse()
        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "sample_interval_s": self.dt,
            "method": "Conservative whole-object shell AABB versus whole original-hand collision AABB, expanded by shape margin; detection gap excluded. Positive distance establishes geometric separation at the sampled state. Zero is inconclusive, not contact.",
            "hold_interval_s": [start, end],
            "hold_sample_count": len(hold),
            "hold_fully_separated_sample_count": len(separated),
            "first_proven_hold_separation_s": separated[0]["time_s"] if separated else None,
            "final_fully_separated": self.rows[-1].get("fully_separate_from_all_hands") if self.rows else None,
            "final_separation_suffix_start_s": suffix[0]["time_s"] if suffix else None,
            "final_separation_suffix_observed_duration_s": suffix[-1]["time_s"] - suffix[0]["time_s"] if suffix else 0,
            "hands": self.hands,
            "support_planes": self.supports,
            "warnings": self.warnings,
            "scope_limits": [
                "Sampled separation evidence, not continuous retention or acquisition qualification.",
                "No force, contact membership, interface slip, target-placement tolerance or fidelity pass is inferred.",
                "Support-plane height is a geometric task observable; negative height outside a finite table is not penetration.",
                "Cable bounds are conservative; particle shell bounds use all vertices and their radii.",
                "AABB overlap can hide separation. Absence of a positive bound cannot establish retention.",
            ],
            "rows": self.rows,
        }
