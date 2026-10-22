# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exploratory VBD jaw manipulation of a coarsened cached Style3D jacket.

Source geometry is BSD-3-Clause. This module does not transfer Style3D's
material model. The prepared mesh retains three boundary loops and one
connected component. Thin triangles remain a declared discretization limit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from experiments.collision_benchmark.assets import AssetMesh, validate_asset
from experiments.collision_benchmark.manipulation import JawScene, drive

__all__ = ["JacketScene", "build_scene", "prepare_jacket_asset"]

_ASSETS = Path(__file__).with_name("assets")


def _boundary_topology(mesh):
    triangles = mesh.triangles
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    adjacency = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    pending, loops = set(adjacency), 0
    while pending:
        stack = [pending.pop()]
        loops += 1
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in pending:
                    pending.remove(neighbour)
                    stack.append(neighbour)
    return {
        "euler_characteristic": len(mesh.vertices) - len(unique) + len(triangles),
        "boundary_loops": loops,
        "boundary_vertices_all_degree_two": all(len(neighbours) == 2 for neighbours in adjacency.values()),
    }


def prepare_jacket_asset():
    """Prepare a local derivative and retain its full source notice and hashes.

    This offline preparation uses the locally installed optional simplification
    tool. Running the prepared simulation only needs the NPZ and metadata.
    """
    from importlib.metadata import version  # noqa: PLC0415

    import trimesh
    from scipy.spatial import KDTree

    from experiments.collision_benchmark.assets import (  # noqa: PLC0415
        _cached_mesh,
        _complete,
        _preview,
        cached_inventory,
    )

    record = next(r for r in cached_inventory() if r["name"] == "h1_jacket")
    source = _cached_mesh(record)
    original = trimesh.Trimesh(vertices=source.vertices, faces=source.triangles, process=False)
    simplified = original.simplify_quadric_decimation(face_count=1600, aggression=3)
    mesh = _complete(
        "cached_jacket_smoke",
        simplified.vertices,
        simplified.faces,
        metadata={
            "source": record["source"],
            "source_name": "h1_jacket",
            "source_revision": record["source_revision"],
            "source_tree_sha256": record["source_tree_sha256"],
            "license": "BSD-3-Clause",
            "license_notice": record["license_notice"],
            "generator_revision": "cached-jacket-quadric-v1",
            "source_entrypoint": record["entrypoint"],
            "source_vertex_count": len(source.vertices),
            "source_triangle_count": len(source.triangles),
            "preparation": {
                "operation": "quadric edge collapse in the authored rest geometry",
                "fast_simplification_version": version("fast-simplification"),
                "trimesh_version": version("trimesh"),
                "target_faces": 1600,
                "aggression": 3,
                "max_distance_to_source_vertex_m": float(KDTree(source.vertices).query(simplified.vertices)[0].max()),
            },
            "color_rgb": (0.68, 0.27, 0.18),
            "preview_pose": "coarsened source rest geometry",
            "material_mapping": "Synthetic Newton triangle/bending coefficients. Style3D material data not transferred.",
            "limitations": "Thin triangles remain after decimation; exploratory smoke only, not a qualified fidelity case.",
        },
    )
    checks = validate_asset(mesh)
    topology = _boundary_topology(mesh)
    original_topology = _boundary_topology(source)
    if not checks["valid"] or topology != original_topology:
        raise ValueError(f"Jacket preparation changed topology: {checks}, {original_topology}, {topology}")
    mesh.metadata.update(validation=checks, topology=topology, source_topology=original_topology)
    np.savez_compressed(_ASSETS / "cached_jacket_smoke.npz", vertices=mesh.vertices, triangles=mesh.triangles)
    mesh.metadata["npz_sha256"] = hashlib.sha256((_ASSETS / "cached_jacket_smoke.npz").read_bytes()).hexdigest()
    mesh.metadata["npz"] = "cached_jacket_smoke.npz"
    mesh.metadata["preview"] = "cached_jacket_smoke.png"
    (_ASSETS / "cached_jacket_smoke.metadata.json").write_text(json.dumps(mesh.metadata, indent=2) + "\n")
    (_ASSETS / "STYLE3D-LICENSE.txt").write_text(record["license_notice"] + "\n")
    _preview(mesh, _ASSETS / "cached_jacket_smoke.png")
    print(
        json.dumps(
            {
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.triangles),
                "validation": checks,
                "topology": topology,
            }
        )
    )
    return mesh


class JacketScene(JawScene):
    """Apply the common contact-only jaw recipe to the prepared cached jacket."""

    def __init__(self, *, device=None, **kwargs):
        data = np.load(_ASSETS / "cached_jacket_smoke.npz")
        metadata = json.loads((_ASSETS / "cached_jacket_smoke.metadata.json").read_text())
        mesh = AssetMesh("cached_jacket_smoke", data["vertices"], data["triangles"], metadata=metadata)
        if not validate_asset(mesh)["valid"]:
            raise ValueError("Prepared jacket mesh failed geometry validation")
        super().__init__("r1_jacket", device=device, cloth_mesh_override=mesh, use_cuda_graph=False, **kwargs)
        # Settle under gravity with the jaws parked away from the garment. Only
        # rigid boundary poses change; cloth positions come from solver steps.
        self.base = (2.0, 2.0, 1.0)
        for _ in range(round(2.0 / self.frame_dt)):
            super().step()
        q = self.state.particle_q.numpy()
        grasp_vertex = int(np.argmax(q[:, 2]))
        target = q[grasp_vertex]
        triangles = q[mesh.triangles]
        lo, hi = triangles.min(axis=1), triangles.max(axis=1)
        near = (
            (hi[:, 0] >= target[0] - 0.063)
            & (lo[:, 0] <= target[0] + 0.063)
            & (hi[:, 2] >= target[2] - 0.028)
            & (lo[:, 2] <= target[2] + 0.028)
        )
        local = triangles[near].reshape(-1, 3)
        lower_y, upper_y = float(local[:, 1].min()), float(local[:, 1].max())
        self.axis = 1
        self.half = 0.045  # Existing jaw geometry's half-extent on Y.
        self.base = (float(target[0]), 0.5 * (lower_y + upper_y), float(target[2]))
        self.open_gap = max(0.035, 0.5 * (upper_y - lower_y) + 0.006)
        if self.base[2] - 0.025 < 0.253:
            raise ValueError("The settled jacket has no raised fold clear of the table for side gripping")
        overlaps = 0
        for sign in (-1, 1):
            center = np.asarray(self.base) + np.asarray([0, sign * (self.half + self.open_gap), 0])
            overlaps += int(np.sum(np.all(np.abs(q - center) < np.asarray([0.06, 0.045, 0.025]) + 0.003, axis=1)))
        if overlaps:
            raise ValueError(f"Settled jacket has {overlaps} initial particle/jaw overlaps")
        self.settle_grasp = {
            "settle_duration_s": 2.0,
            "grasp_vertex": grasp_vertex,
            "grasp_vertex_position_m": target.tolist(),
            "jaw_base_m": list(self.base),
            "local_y_span_m": upper_y - lower_y,
            "initial_particle_jaw_overlap_count": overlaps,
            "table_jaw_clearance_m": self.base[2] - 0.025 - 0.25,
            "recipe": "Settle freely, select highest fold vertex, side-pinch its local footprint.",
        }
        self.time = 0.0
        self.clock.zero_()
        self.history.clear()
        self.initial_q = q.copy()
        wp.launch(
            drive,
            dim=2,
            inputs=[
                self.clock,
                self.jaws,
                wp.vec3(*self.base),
                self.half,
                self.open_gap,
                self.closed_gap,
                self.axis,
                self.dt,
                self.state0.body_q,
                self.state1.body_q,
                self.state0.body_qd,
                self.state1.body_qd,
            ],
            device=self.model.device,
        )
        self.camera_pos = (1.35, -1.75, 1.35)
        self.camera_target = (0.10, 0.35, 0.62)
        self.camera_fov = 52.0
        self.scene_id = "j1_cached_jacket"
        self.title = "J1 | Cached jacket: contact-only jaw pick, carry, release"
        self.description = (
            "Coarsened Style3D h1_jacket source geometry with its three openings preserved. "
            "Prescribed rigid jaws drive contact; every jacket vertex is free. "
            "VBD computes jacket motion with synthetic Newton cloth coefficients. "
            "After two seconds of free settling, the jaws side-pinch the highest fold. "
            "Thin triangles remain: this is exploratory smoke evidence."
        )
        self.asset_ids = ["h1_jacket_coarsened", "G0_generated_jaws"]
        self.jacket_metadata = metadata

    def metrics(self):
        result = super().metrics()
        result["source_asset"] = self.jacket_metadata
        result["settle_grasp_recipe"] = self.settle_grasp
        return result


def build_scene(scene_id, **kwargs):
    """Build the native-hand scene or retain the archived fixture recipe."""
    if scene_id == "j2_franka_jacket":
        from experiments.collision_benchmark.jacket_franka_scene import FrankaJacketScene  # noqa: PLC0415

        return FrankaJacketScene(**kwargs)
    if scene_id != "j1_cached_jacket":
        raise ValueError(f"Unknown jacket smoke scene {scene_id}")
    return JacketScene(**kwargs)


if __name__ == "__main__":
    prepare_jacket_asset()
