# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Observed task evidence without inferred qualification or interface slip.

``criterion_results`` is a pure frame/protocol reduction for offline reports.
Display-frame observations can show displacement or loss, but cannot certify
an unobserved substep or replace the agreed contact and stability criteria.
"""

from __future__ import annotations

import inspect
import math

import numpy as np

__all__ = ["criterion_results", "sample_final_geometry"]


def _finite_number(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _center(frame):
    observation = frame.get("recipe_observation", {})
    candidates = [
        (frame, "particle_center_of_mass_m", "mass-weighted complete particle system"),
        (observation, "center_of_mass_m", "recipe mass-weighted cloth center"),
        (observation, "selected_center_m", "selected cable segment-center mean"),
        (observation, "cable_center_m", "complete cable segment-center mean"),
        (observation, "bag_center_m", "bag particle mean; payloads reported separately"),
        (frame, "particle_mean_m", "unweighted particle mean"),
        (observation, "center_m", "recipe material-point mean"),
    ]
    for row, key, basis in candidates:
        if key in row:
            value = np.asarray(row[key], dtype=float)
            if value.shape == (3,) and np.isfinite(value).all():
                return value, basis
    return None, None


def _reference_height(protocol):
    if _finite_number(protocol.get("support_reference_height_m")):
        return float(protocol["support_reference_height_m"])
    heights = []
    for surface in protocol.get("penetration_surfaces", []):
        if surface["kind"] == "plane" and np.allclose(surface["normal"], [0, 0, 1]):
            heights.append(float(surface["offset_m"]))
        if surface["kind"] == "box" and np.allclose(surface.get("rotation_xyzw", [0, 0, 0, 1]), [0, 0, 0, 1]):
            heights.append(float(surface["center_m"][2] + surface["half_extents_m"][2]))
    return max(heights) if heights else None


def _height_clearance(frame, support_height):
    observation = frame.get("recipe_observation", {})
    if support_height is not None and _finite_number(frame.get("minimum_shell_z_m")):
        return float(frame["minimum_shell_z_m"]) - support_height, "complete particle shell minus support reference"
    for key in ("minimum_surface_clearance_m", "minimum_capsule_surface_clearance_m", "bag_surface_clearance_m"):
        if _finite_number(observation.get(key)):
            return float(observation[key]), key
    if support_height is not None:
        for row, key in (
            (frame, "minimum_shell_z_m"),
            (observation, "selected_minimum_capsule_surface_z_m"),
            (observation, "minimum_surface_z_m"),
            (observation, "min_surface_z_m"),
        ):
            if _finite_number(row.get(key)):
                return float(row[key]) - support_height, key + " minus support reference"
    return None, None


def _window(frames, start, end, protocol):
    cadence = float(protocol["frame_dt_s"])
    selected = [row for row in frames if start - 1e-9 <= row["time_s"] <= end + 1e-9]
    observed_end = float(frames[-1]["time_s"]) if frames else None
    gaps = np.diff([row["time_s"] for row in selected])
    covered = bool(
        selected
        and observed_end >= end - 1e-9
        and selected[0]["time_s"] <= start + cadence + 1e-8
        and selected[-1]["time_s"] >= end - cadence - 1e-8
        and (not len(gaps) or max(gaps) <= cadence + 1e-8)
    )
    reference = _reference_height(protocol)
    clearances = [(row["time_s"], *_height_clearance(row, reference)) for row in selected]
    clearances = [row for row in clearances if row[1] is not None]
    centers = [(row["time_s"], *_center(row)) for row in selected]
    centers = [row for row in centers if row[1] is not None]
    result = {
        "requested_start_s": float(start),
        "requested_end_s": float(end),
        "observed_interval_complete_at_frame_cadence": covered,
        "sample_count": len(selected),
        "first_sample_s": selected[0]["time_s"] if selected else None,
        "last_sample_s": selected[-1]["time_s"] if selected else None,
        "sample_period_s": cadence,
        "support_reference_height_m": reference,
        "minimum_sampled_shell_height_above_support_m": None,
        "positive_height_at_every_observed_frame": None,
        "height_interpretation": "Positive height proves sampled material is above the reference plane. Nonpositive height alone does not prove contact with a finite support or grasp loss.",
    }
    if clearances:
        witness = min(clearances, key=lambda row: row[1])
        result.update(
            minimum_sampled_shell_height_above_support_m=witness[1],
            minimum_height_time_s=witness[0],
            clearance_source=witness[2],
            positive_height_at_every_observed_frame=all(row[1] > 0 for row in clearances),
            clearance_sample_count=len(clearances),
        )
    if centers:
        origin = centers[0][1]
        result.update(
            center_basis=centers[0][2],
            center_change_m=(centers[-1][1] - origin).tolist(),
            maximum_center_excursion_m=max(float(np.linalg.norm(row[1] - origin)) for row in centers),
            center_motion_is_interface_slip=False,
        )
    payload_rows = [row.get("recipe_observation", {}).get("payloads") for row in selected]
    payload_rows = [row for row in payload_rows if row is not None]
    if payload_rows:
        counts = {len(row) for row in payload_rows}
        result["payload_count_consistent"] = len(counts) == 1
        result["payloads"] = []
        if len(counts) == 1:
            for index in range(len(payload_rows[0])):
                values = [row[index].get("surface_clearance_m") for row in payload_rows]
                values = [float(value) for value in values if _finite_number(value)]
                result["payloads"].append(
                    {"index": index, "minimum_sampled_shell_height_above_support_m": min(values) if values else None}
                )
    return result


def _base(applicable, missing):
    return {
        "applicable": bool(applicable),
        "status": "incomplete" if applicable else "not_applicable",
        "scope": "Observed task geometry at display frames; no force-closure, contact-slip or material-fidelity pass inferred.",
        "measurements": {},
        "missing_evidence": missing if applicable else [],
    }


def criterion_results(frames, scene=None, protocol=None):
    """Return the four task-category rows using saved frame data only.

    ``scene`` is accepted for the runner's common interface. Offline callers
    may omit it and supply ``protocol``. No simulation or geometry query is
    performed here. Remaining four categories belong to ComparisonObserver.
    """
    if protocol is None:
        if scene is None:
            raise ValueError("A declared comparison protocol is required")
        protocol = scene.protocol
    frames = sorted(frames, key=lambda row: row["time_s"])
    applicable = protocol.get("category_applicability") or protocol["applicability"]
    results = {
        name: _base(
            applicable[name], ["Declared task acceptance limits and qualified grip/contact evidence are missing."]
        )
        for name in ("grasp", "hold", "transport", "place_release")
    }
    start, end = protocol.get("hold_start"), protocol.get("hold_end")
    results["grasp"]["measurements"] = {
        "prepared_initial_contact": bool(protocol.get("initial_hold_prepared")),
        "acquisition_claimed_by_recipe": bool(applicable["grasp"]),
        "stationary_hold_command_start_s": start,
        "acquisition_boundary_qualified": False,
    }
    if applicable["hold"] and start is not None:
        checkpoints = []
        for elapsed in (10.0, 60.0):
            checkpoint = _window(frames, start, start + elapsed, protocol)
            checkpoint.update(
                checkpoint_elapsed_s=elapsed, checkpoint_within_declared_hold=start + elapsed <= end + 1e-9
            )
            checkpoints.append(checkpoint)
        results["hold"]["measurements"] = {
            "hold_start_s": start,
            "hold_end_s": end,
            "checkpoints": checkpoints,
            "interface_slip_m": None,
            "continuous_grasp_loss_time_s": None,
            "continuous_retention_qualified": False,
        }
        results["hold"]["missing_evidence"] = [
            "Material-interface slip and its model-derived allowance are not measured by center motion.",
            "Frame geometry does not certify continuous retention between samples or a qualified acquisition boundary.",
            "Required numerical acceptance limits remain unset.",
        ]
    if applicable["transport"]:
        results["transport"]["measurements"] = {
            "intervals": [
                _window(frames, interval["start_s"], interval["end_s"], protocol)
                for interval in protocol.get("transport_intervals", [])
            ],
            "command_tracking_error_m": None,
            "load_retention_qualified": False,
        }
    if applicable["place_release"]:
        last = frames[-1] if frames else {}
        results["place_release"]["measurements"] = {
            "intervals": [
                _window(frames, interval["start_s"], interval["end_s"], protocol)
                for interval in protocol.get("place_release_intervals", [])
            ],
            "final_time_s": last.get("time_s"),
            "full_commanded_cycle_observed": bool(
                last and last["time_s"] >= protocol["duration_s"] - protocol["frame_dt_s"]
            ),
            "final_hand_geometry": last.get("final_hand_geometry"),
            "released_on_intended_support_qualified": False,
        }
        results["place_release"]["missing_evidence"] = [
            "Final sampled hand separation is distinct from correct placement and full surface separation through withdrawal.",
            "Placement region, settling and release acceptance limits remain unset.",
        ]
    return results


def sample_final_geometry(scene):
    """Read final original-hand geometry once; never move material or bodies.

    This is point/shell sampling against original colliders. It is not a full
    triangle intersection proof. Detected contact counts alone are not used.
    """
    raw = getattr(scene, "raw_scene", scene)
    drivers = list(getattr(raw, "drivers", []))
    if hasattr(raw, "driver"):
        drivers.append(raw.driver)
    result = {"time_s": float(scene.time), "method": "material points against original hand solids", "hands": []}
    if not scene.model.particle_count and hasattr(raw, "_geometry_audit"):
        result.update(
            method="original recipe capsule-axis samples with physical radius against original hand solids",
            recipe_capsule_geometry=raw._geometry_audit(),
            limitation="Sampled capsule distance, not continuous surface intersection proof.",
        )
        return result
    if not drivers or not scene.model.particle_count:
        result["unavailable_reason"] = (
            "No particle-to-original-hand driver audit; cable task uses recipe capsule samples."
        )
        return result
    q, radius = scene.state.particle_q.numpy(), scene.model.particle_radius.numpy()
    margins = scene.model.shape_margin.numpy()
    for index, driver in enumerate(drivers):
        parameters = inspect.signature(driver.geometry_audit).parameters
        kwargs = {"points": q, "radii": radius}
        if "full_surface_contact" in parameters:
            kwargs["full_surface_contact"] = True
        audit = driver.geometry_audit(scene.state, **kwargs)
        gap = audit.get("minimum_shell_to_robot_distance_m", audit.get("minimum_shell_to_hand_distance_m"))
        shape_ids = list(driver.spec.shape_ids)
        margin = float(np.max(margins[shape_ids])) if shape_ids else 0.0
        result["hands"].append(
            {
                "hand_index": index,
                "minimum_sampled_shell_to_original_solid_m": gap,
                "maximum_original_hand_shape_margin_m": margin,
                "sampled_shell_beyond_all_hand_margins": None if gap is None else bool(gap > margin),
                "original_geometry_audit": audit,
            }
        )
    return result
