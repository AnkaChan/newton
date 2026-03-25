#!/usr/bin/env python
"""Diagnostic script for VBD contact instability.

Runs the cloth_franka_from_state example with instrumentation to record
per-substep and per-iteration VBD data for debugging contact instability.

Usage:
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/diag_contact_instability.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverVBD
from newton._src.solvers.vbd.debug_recorder import DebugRecorder

# ---------- parameters ----------
NUM_FRAMES = 60
SIM_SUBSTEPS = 10
ITERATIONS = 5
FPS = 60
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "debug_contact_instability.npz"
)
OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)

MAX_SUBSTEPS = NUM_FRAMES * SIM_SUBSTEPS


def build_scene():
    """Build the scene identical to example_cloth_franka_from_state."""
    # Parameters (cm scale)
    cloth_particle_radius = 0.8
    cloth_body_contact_margin = 0.8
    particle_self_contact_radius = 0.2
    particle_self_contact_margin = 0.2

    soft_contact_ke = 1e4
    soft_contact_kd = 1e-2

    robot_contact_ke = 5e4
    robot_contact_kd = 1e-3
    robot_contact_mu = 1.5

    self_contact_friction = 0.25

    tri_ke = 1e4
    tri_ka = 1e4
    tri_kd = 1.5e-6

    bending_ke = 5
    bending_kd = 1e-2

    scene = ModelBuilder(gravity=-981.0)

    # Robot articulation
    franka = ModelBuilder()
    asset_path = newton.utils.download_asset("franka_emika_panda")
    franka.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform((-50.0, -50.0, -10.0), wp.quat_identity()),
        floating=False,
        scale=100,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )
    franka.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
    scene.add_world(franka)

    # Table (cm)
    table_shape_idx = scene.shape_count
    scene.add_shape_box(
        -1,
        wp.transform(wp.vec3(0.0, -50.0, 10.0), wp.quat_identity()),
        hx=40.0, hy=40.0, hz=10.0,
    )

    # T-shirt
    usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
    usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
    shirt_mesh = newton.usd.get_mesh(usd_prim)
    vertices = [wp.vec3(v) for v in shirt_mesh.vertices]

    scene.add_cloth_mesh(
        vertices=vertices,
        indices=shirt_mesh.indices,
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
        pos=wp.vec3(0.0, 70.0, 30.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.02,
        scale=1.0,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        edge_ke=bending_ke,
        edge_kd=bending_kd,
        particle_radius=cloth_particle_radius,
    )

    scene.color()
    scene.add_ground_plane()
    model = scene.finalize(requires_grad=False)

    # Material setup
    model.soft_contact_ke = soft_contact_ke
    model.soft_contact_kd = soft_contact_kd
    model.soft_contact_mu = self_contact_friction

    shape_ke = model.shape_material_ke.numpy()
    shape_kd = model.shape_material_kd.numpy()
    shape_mu = model.shape_material_mu.numpy()
    shape_ke[...] = robot_contact_ke
    shape_kd[...] = robot_contact_kd
    shape_mu[...] = robot_contact_mu
    model.shape_material_ke = wp.array(shape_ke, dtype=model.shape_material_ke.dtype, device=model.device)
    model.shape_material_kd = wp.array(shape_kd, dtype=model.shape_material_kd.dtype, device=model.device)
    model.shape_material_mu = wp.array(shape_mu, dtype=model.shape_material_mu.dtype, device=model.device)

    # States
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # Collision pipeline
    collision_pipeline = newton.CollisionPipeline(
        model, soft_contact_margin=cloth_body_contact_margin
    )
    contacts = collision_pipeline.contacts()

    # Load saved state
    state_path = os.path.join(
        os.path.dirname(newton.examples.__file__), "cloth", "cloth_state_2.npz"
    )
    saved = np.load(state_path)
    state_0.particle_q.assign(wp.array(saved["particle_q"], dtype=wp.vec3))
    state_0.particle_qd.assign(wp.array(saved["particle_qd"], dtype=wp.vec3))
    if "joint_q" in saved:
        state_0.joint_q.assign(wp.array(saved["joint_q"], dtype=float))
    if "joint_qd" in saved:
        state_0.joint_qd.assign(wp.array(saved["joint_qd"], dtype=float))
    sim_time = float(saved["sim_time"])
    print(f"Loaded state from {state_path} (time={sim_time:.3f}s)")

    eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # Zero rest angles (same as example)
    model.edge_rest_angle.zero_()

    # Create solver with tile solve disabled (for debug force output)
    cloth_solver = SolverVBD(
        model,
        iterations=ITERATIONS,
        integrate_with_external_rigid_solver=True,
        particle_self_contact_radius=particle_self_contact_radius,
        particle_self_contact_margin=particle_self_contact_margin,
        particle_topological_contact_filter_threshold=1,
        particle_rest_shape_contact_exclusion_radius=0.5,
        particle_enable_self_contact=True,
        particle_vertex_contact_buffer_size=16,
        particle_edge_contact_buffer_size=20,
        particle_collision_detection_interval=-1,
        rigid_contact_k_start=soft_contact_ke,
        particle_enable_tile_solve=False,  # Force non-tile for debug output
    )

    gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

    return (
        model, state_0, state_1, control, collision_pipeline, contacts,
        cloth_solver, gravity_earth, sim_time
    )


def main():
    wp.init()
    print(f"Warp device: {wp.get_device()}")
    print(f"Recording {NUM_FRAMES} frames ({MAX_SUBSTEPS} substeps, {MAX_SUBSTEPS * ITERATIONS} iterations)")

    (model, state_0, state_1, control, collision_pipeline, contacts,
     cloth_solver, gravity_earth, sim_time) = build_scene()

    print(f"Particle count: {model.particle_count}")
    print(f"Edge count: {model.edge_count}")
    print(f"Triangle count: {model.tri_count}")

    # Set up recorder
    recorder = DebugRecorder(
        model,
        max_substeps=MAX_SUBSTEPS,
        iterations=ITERATIONS,
        device=str(model.device),
    )
    cloth_solver.debug_recorder = recorder
    print(f"DebugRecorder allocated (device={model.device})")

    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SIM_SUBSTEPS

    # Also record tri_indices and edge_indices for analysis
    tri_indices = model.tri_indices.numpy().copy()
    edge_indices = model.edge_indices.numpy().copy()

    t0 = time.time()
    for frame in range(NUM_FRAMES):
        cloth_solver.rebuild_bvh(state_0)
        for _step in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            state_1.clear_forces()

            # Keep robot frozen
            wp.copy(state_1.body_q, state_0.body_q)
            wp.copy(state_1.joint_q, state_0.joint_q)

            model.gravity.assign(gravity_earth)

            collision_pipeline.collide(state_0, contacts)
            cloth_solver.step(state_0, state_1, control, contacts, sim_dt)

            state_0, state_1 = state_1, state_0
            sim_time += sim_dt

        elapsed = time.time() - t0
        substeps_done = (frame + 1) * SIM_SUBSTEPS
        rate = substeps_done / elapsed
        print(
            f"  Frame {frame + 1}/{NUM_FRAMES} | "
            f"substeps: {substeps_done}/{MAX_SUBSTEPS} | "
            f"{rate:.1f} substeps/s | "
            f"elapsed: {elapsed:.1f}s"
        )

    wp.synchronize()
    total_time = time.time() - t0
    print(f"\nSimulation done in {total_time:.1f}s")
    print(f"Recorded {recorder.substeps_recorded} substeps, {recorder.iterations_recorded} iterations")

    # Export
    print(f"Exporting to {OUTPUT_PATH}...")
    data = recorder.to_dict()
    # Add mesh topology for analysis
    data["tri_indices"] = tri_indices
    data["edge_indices"] = edge_indices
    np.savez_compressed(OUTPUT_PATH, **data)
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Saved {OUTPUT_PATH} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
