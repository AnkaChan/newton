# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Reproduce free-settling jacket strain without robot acquisition or pins."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import warp as wp

from experiments.collision_benchmark.assets import AssetMesh, _cached_mesh, cached_inventory, validate_asset
from experiments.collision_benchmark.jacket_diagnostics import ClothStrain, mesh_statistics
from experiments.collision_benchmark.manipulation import JawScene
from newton.solvers import SolverVBD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=("coarse", "source"), default="coarse")
    parser.add_argument("--flat-bending", action="store_true")
    parser.add_argument("--contact", choices=("original", "off", "reference", "margin-only"), default="original")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/collision-benchmark-warp-jacket-fix")
    if args.mesh == "coarse":
        data = np.load(Path(__file__).with_name("assets") / "cached_jacket_smoke.npz")
        mesh = AssetMesh("jacket", data["vertices"], data["triangles"])
    else:
        mesh = _cached_mesh(next(r for r in cached_inventory() if r["name"] == "h1_jacket"))
    scene = JawScene("r1_jacket", cloth_mesh_override=mesh, use_cuda_graph=False, device="cuda:0")
    scene.base = (2.0, 2.0, 1.0)
    if args.contact != "original":
        scene.solver = SolverVBD(
            scene.model,
            iterations=10,
            particle_enable_self_contact=args.contact != "off",
            particle_self_contact_margin=0.002,
            particle_self_contact_gap=0.006 if args.contact == "margin-only" else 0.0,
            particle_rest_shape_contact_exclusion_radius=0.0 if args.contact == "margin-only" else 0.005,
            particle_topological_contact_filter_threshold=2 if args.contact == "margin-only" else 1,
            particle_vertex_contact_buffer_size=64,
            friction_epsilon=0.001,
            integrate_with_external_rigid_solver=False,
            rigid_compliant_alm=True,
        )
    if args.flat_bending:
        scene.model.edge_rest_angle.zero_()
    contact_audit = {}
    if args.contact != "off":
        detector = scene.solver.trimesh_collision_detector
        detector.refit(scene.state.particle_q)
        detector.vertex_triangle_collision_detection(
            scene.solver.particle_self_contact_margin,
            min_query_radius=scene.solver.particle_rest_shape_contact_exclusion_radius,
            min_distance_filtering_ref_pos=scene.model.particle_q,
        )
        detector.edge_edge_collision_detection(
            scene.solver.particle_self_contact_margin,
            min_query_radius=scene.solver.particle_rest_shape_contact_exclusion_radius,
            min_distance_filtering_ref_pos=scene.model.particle_q,
        )
        info = detector.collision_info
        contact_audit = {
            "query_radius_m": scene.solver.particle_self_contact_margin,
            "method": "Newton self-contact detector queried at the force interaction radius in the imported rest state, before stepping.",
            "vertex_triangle_candidates": int(info.vertex_colliding_triangles_count.numpy().sum()),
            "edge_edge_candidates": int(info.edge_colliding_edges_count.numpy().sum()),
        }
    strain = ClothStrain(scene.model)
    samples = [{"time_s": 0.0, **strain.sample(scene.state.particle_q.numpy())}]
    with wp.ScopedCapture(device=scene.model.device) as capture:
        scene.simulate()
    scene.graph = capture.graph
    for i in range(round(args.duration / scene.frame_dt)):
        scene.step()
        q = scene.state.particle_q.numpy()
        if not np.isfinite(q).all():
            samples.append(
                {"time_s": scene.time, "finite": False, "nonfinite_vertices": int(np.sum(~np.isfinite(q).all(axis=1)))}
            )
            break
        if (i + 1) % 3 == 0:
            samples.append({"time_s": scene.time, **strain.sample(q)})
    result = {
        "method": "Common JawScene material/contact/step recipe, both jaws parked remotely throughout. No cloth pins.",
        "mesh": args.mesh,
        "flat_bending_rest_angles": args.flat_bending,
        "self_contact_recipe": args.contact,
        "geometry": mesh_statistics(mesh.vertices, mesh.triangles),
        "geometry_validation": validate_asset(mesh),
        "particle_mass_kg": float(strain.mass.sum()),
        "initial_interaction_candidate_audit": contact_audit,
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.out.with_suffix(".npz"), positions=scene.state.particle_q.numpy(), triangles=mesh.triangles
    )
    print(json.dumps(samples[-1]), flush=True)


if __name__ == "__main__":
    main()
