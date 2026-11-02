# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adapt prescribed benchmark hands to upstream solver-owned rigid-soft DAT.

Kinematic poses remain prescribed. DAT constrains the soft/dynamic solve, not
continuous kinematic hand sweeps. No native solver or material equation changes.
"""

from __future__ import annotations

import copy

import warp as wp

import newton
from newton.solvers import SolverBase

__all__ = [
    "bind_solver",
    "configuration",
    "configure",
    "disable",
    "native_options",
    "owned_pipeline",
    "prepare_kinematic_endpoints",
    "record_contact_counts",
    "record_self_contact_counts",
]

_ORIGINAL_PIPELINE = newton.CollisionPipeline
_CONFIG = None
_OWNERS = {}
_GAPS = {
    "m2_franka_pouch": 0.003,
    "m3_franka_tote_empty": 0.01,
    "r5_franka_dress": 0.005,
    "r5_franka_patch_contact_v2": 0.005,
    "r5_allegro_towel": 0.005,
    "r5_franka_towel_contact_v2": 0.005,
    "r5_franka_trousers": 0.005,
    "j3_h1_jacket": 0.005,
    "r7_h1_towel": 0.005,
}


def configure(case_id):
    """Select one audited recipe before any scene construction."""
    global _CONFIG  # noqa: PLW0603
    if case_id not in _GAPS:
        raise ValueError(f"No audited native DAT pipeline settings for {case_id}")
    proposed = {
        "case_id": case_id,
        "particle_elasticity": "standard VBD",
        "rigid_soft_enable_dat": True,
        "integrate_with_external_rigid_solver": False,
        "enable_fast_math": False,
        "pipeline": {"soft_contact_gap": _GAPS[case_id], "enable_rigid_soft_full_surface_contact": True},
        "collision_schedule": {"RIGID": "PRE_POST_INIT", "SOFT_SELF_CONTACT": "PRE_POST_INIT"},
        "mesh_contact_filter": "Merged upstream OGC Eq. 14 wireframe edge-edge filtering; native full-surface solver selects force rows",
        "motion_bridge": "Copy prescribed state_out(t+dt) kinematic endpoints into state_in before native step",
        "history_scope": "Native first/reset step rebaselines at its input endpoint; later steps retain prior accepted pose",
        "guarantee_scope": "Kinematic motion is not DAT-truncated; no continuous prescribed-hand sweep guarantee",
        "contact_latch_scope": "Global rigid/full-surface counts at every owned detection, including preparation",
        "self_contact_latch_scope": "Raw vertex/edge/optional triangle row counts and capacity excess at every native self detection; sticky across preparation",
        "capacity_scope_limit": "Per-body adjacency and other internal broad/narrow-phase scratch capacities are not latched",
    }
    if _CONFIG is not None and proposed != _CONFIG:
        raise ValueError("Use a separate process for another native DAT case")
    _CONFIG = proposed
    newton.CollisionPipeline = _scene_pipeline


def disable():
    """Restore the public pipeline factory after isolated CPU tests."""
    global _CONFIG  # noqa: PLW0603
    newton.CollisionPipeline = _ORIGINAL_PIPELINE
    _CONFIG = None
    _OWNERS.clear()


def configuration():
    """Return a serializable copy suitable for provenance and cache identity."""
    return copy.deepcopy(_CONFIG)


def native_options(mode):
    """Use public upstream options and reject unavailable ALM elasticity."""
    if mode:
        raise ValueError("The upstream DAT benchmark supports standard VBD elasticity only")
    return {
        "rigid_compliant_alm": False,
        "rigid_contact_hard": False,
        "rigid_soft_enable_dat": True,
        "integrate_with_external_rigid_solver": False,
        "enable_fast_math": False,
        "collision_frequency_type": {
            SolverBase.CollisionSlot.RIGID: SolverBase.CollisionFrequencyType.PRE_POST_INIT,
            SolverBase.CollisionSlot.SOFT_SELF_CONTACT: SolverBase.CollisionFrequencyType.PRE_POST_INIT,
        },
    }


def owned_pipeline(model):
    """Construct detection before the native solver allocates DAT references."""
    if _CONFIG is None:
        raise RuntimeError("Configure the native DAT case before constructing its solver")
    key = id(model)
    if key not in _OWNERS:
        _OWNERS[key] = {"model": model, "pipeline": _OwnedPipeline(model, **_CONFIG["pipeline"]), "solver": None}
    return _OWNERS[key]["pipeline"]


def bind_solver(model, solver):
    """Publish the native contacts buffer to the scene-facing pipeline view."""
    _OWNERS[id(model)]["solver"] = solver


@wp.kernel
def _latch_contact_counts(counters: wp.array[wp.int32], latch: wp.array[wp.int32]):
    for index in range(4):
        latch[index] = wp.max(latch[index], counters[index])
    latch[4] = latch[4] + 1


def record_contact_counts(counters, latch):
    """Latch counts after each detection without host reads during capture."""
    wp.launch(_latch_contact_counts, dim=1, inputs=[counters], outputs=[latch], device=latch.device)


@wp.kernel
def _latch_row_capacity(
    counts: wp.array[wp.int32], capacities: wp.array[wp.int32], family: int, latch: wp.array[wp.int32]
):
    row = wp.tid()
    wp.atomic_max(latch, family, counts[row])
    wp.atomic_max(latch, family + 3, wp.max(0, counts[row] - capacities[row]))


@wp.kernel
def _increment_self_detection(latch: wp.array[wp.int32]):
    latch[6] = latch[6] + 1


def record_self_contact_counts(info, latch):
    """Preserve self-contact row peaks and positive capacity excess on device."""
    for family, name in enumerate(
        ("vertex_colliding_triangles", "edge_colliding_edges", "triangle_colliding_vertices")
    ):
        counts = getattr(info, name + "_count")
        capacities = getattr(info, name + "_buffer_sizes")
        if counts is not None and counts.size:
            wp.launch(
                _latch_row_capacity,
                dim=counts.size,
                inputs=[counts, capacities, family],
                outputs=[latch],
                device=latch.device,
            )
    wp.launch(_increment_self_detection, dim=1, outputs=[latch], device=latch.device)


class _OwnedPipeline(_ORIGINAL_PIPELINE):
    """Keep capacity evidence from every native detection, not only the last."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Peaks: rigid rows, soft rows, reduction/source overflow; then call count.
        self.dat_contact_latch = wp.zeros(5, dtype=wp.int32, device=model.device)
        # Three row-family peaks, their positive capacity excess, then call count.
        self.dat_self_contact_latch = wp.zeros(7, dtype=wp.int32, device=model.device)

    def collide(self, state, contacts, **kwargs):
        """Run unchanged native detection and preserve its capacity counters."""
        super().collide(state, contacts, **kwargs)
        record_contact_counts(contacts.contact_counters, self.dat_contact_latch)

    def _detect_soft_self_contact(self, particle_q, contacts):
        """Latch every native self-detection pass, including separate DAT passes."""
        super()._detect_soft_self_contact(particle_q, contacts)
        record_self_contact_counts(contacts.soft_self_contact_data, self.dat_self_contact_latch)


class _ScenePipelineView:
    """Suppress the recipe's obsolete external collision pass."""

    def __init__(self, owner):
        self.owner = owner

    def contacts(self):
        """Use the native solver's current owned contact storage."""
        solver = self.owner["solver"]
        if solver is None:
            raise RuntimeError("Construct the native DAT solver before requesting scene contacts")
        return solver.contacts

    def collide(self, state, contacts):
        """Detection runs inside the native solver at its configured instants."""

    def __getattr__(self, name):
        return getattr(self.owner["pipeline"], name)


def _scene_pipeline(model, **kwargs):
    """Check the original recipe's detection options without allocating a duplicate."""
    if _CONFIG is None or kwargs != _CONFIG["pipeline"]:
        raise ValueError(f"Scene pipeline options differ from the audited native DAT settings: {kwargs}")
    owned_pipeline(model)
    return _ScenePipelineView(_OWNERS[id(model)])


@wp.kernel
def _copy_kinematic_endpoints(
    inverse_mass: wp.array[float],
    body_flags: wp.array[wp.int32],
    endpoint_q: wp.array[wp.transform],
    endpoint_qd: wp.array[wp.spatial_vector],
    input_q: wp.array[wp.transform],
    input_qd: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    if inverse_mass[body] == 0.0 or (body_flags[body] & int(newton.BodyFlags.KINEMATIC)) != 0:
        input_q[body] = endpoint_q[body]
        input_qd[body] = endpoint_qd[body]


def prepare_kinematic_endpoints(model, state_in, state_out):
    """Preserve endpoint FK while leaving all dynamic body rows untouched."""
    if model.body_count:
        wp.launch(
            _copy_kinematic_endpoints,
            dim=model.body_count,
            inputs=[model.body_inv_mass, model.body_flags, state_out.body_q, state_out.body_qd],
            outputs=[state_in.body_q, state_in.body_qd],
            device=model.device,
        )
