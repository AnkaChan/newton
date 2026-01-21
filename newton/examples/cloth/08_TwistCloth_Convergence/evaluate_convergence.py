# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Convergence Evaluation Script
#
# This script loads a recovery state and evaluates the convergence behavior
# of different truncation modes by tracking force residuals after each
# VBD iteration.
#
# Truncation modes:
#   0: Isotropic (conservative bound)
#   1: Planar (DAT - Divide and Truncate)
#   2: CCD (Continuous Collision Detection - global min t)
#   3: Dykstra (Hessian-weighted Dykstra alternating projection)
###########################################################################

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import warp as wp
import matplotlib.pyplot as plt

# Import from the save_at_release script
from example_twist_save_at_release import TwistClothSimulator, example_config as default_config

# Import kernels from the VBD solver module (internal)
import newton._src.solvers.vbd.solver_vbd as vbd_module
from newton._src.solvers.vbd.solver_vbd import *

from newton import ParticleFlags
from newton.solvers import SolverVBD

@wp.kernel
def solve_elasticity_tile_measure_force(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    adjacency: RigidForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    particle_displacements: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        if thread_idx == 0:
            particle_displacements[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # elastic force and hessian
    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    batch_counter = wp.int32(0)

    if tri_indices:
        # loop through all the adjacent triangles using whole block
        while batch_counter + thread_idx < num_adj_faces:
            adj_tri_counter = thread_idx + batch_counter
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            # elastic force and hessian
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, adj_tri_counter)

            # fmt: off
            if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "particle: %d | num_adj_faces: %d | ",
                    particle_index,
                    get_vertex_num_adjacent_faces(particle_index, adjacency),
                )
                wp.printf("i_face: %d | face id: %d | v_order: %d | ", adj_tri_counter, tri_index, vertex_order)
                wp.printf(
                    "face: %d %d %d\n",
                    tri_indices[tri_index, 0],
                    tri_indices[tri_index, 1],
                    tri_indices[tri_index, 2],
                )
            # fmt: on

            if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
                f_tri, h_tri = evaluate_stvk_force_hessian(
                    tri_index,
                    vertex_order,
                    pos,
                    pos_prev,
                    tri_indices,
                    tri_poses[tri_index],
                    tri_areas[tri_index],
                    tri_materials[tri_index, 0],
                    tri_materials[tri_index, 1],
                    tri_materials[tri_index, 2],
                    dt,
                )

                f += f_tri
                h += h_tri

    if edge_indices:
        batch_counter = wp.int32(0)
        num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_index)
        while batch_counter + thread_idx < num_adj_edges:
            adj_edge_counter = batch_counter + thread_idx
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
                adjacency, particle_index, adj_edge_counter
            )
            if edge_bending_properties[nei_edge_index, 0] > 0.0:
                f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                    nei_edge_index,
                    vertex_order_on_edge,
                    pos,
                    pos_prev,
                    edge_indices,
                    edge_rest_angles,
                    edge_rest_length,
                    edge_bending_properties[nei_edge_index, 0],
                    edge_bending_properties[nei_edge_index, 1],
                    dt,
                )

                f += f_edge
                h += h_edge

    if tet_indices:
        # solve tet elasticity
        batch_counter = wp.int32(0)
        num_adj_tets = get_vertex_num_adjacent_tets(adjacency, particle_index)
        while batch_counter + thread_idx < num_adj_tets:
            adj_tet_counter = batch_counter + thread_idx
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(
                adjacency, particle_index, adj_tet_counter
            )
            if tet_materials[nei_tet_index, 0] > 0.0 or tet_materials[nei_tet_index, 1] > 0.0:
                f_tet, h_tet = evaluate_volumetric_neo_hooken_force_and_hessian(
                    nei_tet_index,
                    vertex_order_on_tet,
                    pos_prev,
                    pos,
                    tet_indices,
                    tet_poses[nei_tet_index],
                    tet_materials[nei_tet_index, 0],
                    tet_materials[nei_tet_index, 1],
                    tet_materials[nei_tet_index, 2],
                    dt,
                )

                f += f_tet
                h += h_tet

    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)

    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    if thread_idx == 0:
        h_total = (
            h_total
            + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
            + particle_hessians[particle_index]
        )
        if abs(wp.determinant(h_total)) > 1e-8:
            h_inv = wp.inverse(h_total)
            f_total = (
                f_total
                + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
                + particle_forces[particle_index]
            )
            particle_displacements[particle_index] = particle_displacements[particle_index] + h_inv * f_total
            particle_forces[particle_index] = f_total



def load_run_config(recovery_state_dir: str) -> dict:
    """
    Load the run configuration from the recovery state directory.
    
    Args:
        recovery_state_dir: Directory containing the recovery state and config
        
    Returns:
        dict: The loaded configuration, merged with defaults
    """
    config_path = os.path.join(recovery_state_dir, "run_config.json")
    
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        
        # Merge with defaults (saved config takes precedence)
        config = {**default_config, **saved_config}
        print(f"  Loaded {len(saved_config)} config values from file")
        return config
    else:
        print(f"Warning: No config file found at {config_path}")
        print("  Using default config instead")
        return default_config.copy()


def compute_force_residual(solver: SolverVBD, state, dt: float) -> dict:
    """
    Compute the force residual for all particles.
    
    The force residual measures how far the system is from equilibrium.
    A lower residual means better convergence.
    
    Returns:
        dict with various residual metrics
    """
    model = solver.model
    
    # Get flags to identify active particles
    flags = model.particle_flags.numpy()
    
    # Read forces directly from solver.particle_forces (already computed by the solver)
    forces = solver.particle_forces.numpy()
    
    # Only consider active particles
    active_mask = (flags & int(ParticleFlags.ACTIVE)) != 0
    active_forces = forces[active_mask]
    
    # Compute per-vertex force norms
    force_norms = np.linalg.norm(active_forces, axis=1)
    
    # Return per-vertex average as main metric
    return {
        "mean_norm": float(np.mean(force_norms)) if len(force_norms) > 0 else 0.0,  # Per-vertex average
        "l2_norm": float(np.sqrt(np.sum(force_norms**2))),
        "linf_norm": float(np.max(force_norms)) if len(force_norms) > 0 else 0.0,
        "num_active": int(np.sum(active_mask)),
    }


def run_vbd_iterations_with_residuals(
    sim: TwistClothSimulator,
    solver: SolverVBD,
    dt: float,
    num_iterations: int,
    truncation_mode: int,
    collision_detection_interval: int = -1,  # -1 = no redo, 0 = once at start, N = every N iterations
) -> tuple:
    """
    Run VBD iterations and collect force residuals after each iteration.
    
    This manually steps through the VBD solve to capture per-iteration data.
    
    Args:
        sim: The simulator
        solver: The VBD solver
        dt: Time step
        num_iterations: Number of iterations to run
        truncation_mode: 0=isometric, 1=planar, 2=CCD
        
    Returns:
        Tuple of (residuals, timing_info):
          - residuals: List of residual dicts, one per iteration
          - timing_info: Dict with timing measurements
    """
    model = solver.model
    state_in = sim.state_0
    
    # Save original mode
    original_mode = solver.truncation_mode
    
    residuals = []
    
    # =========================================================================
    # Initialization phase: ALWAYS use mode 1 (Planar) for fair comparison
    # =========================================================================
    solver.truncation_mode = 1  # Use Planar for initialization
    
    # Collision detection BEFORE forward_step
    solver.collision_detection_penetration_free(state_in)
    
    # Forward step (initialization)
    wp.launch(
        kernel=vbd_module.forward_step,
        inputs=[
            dt,
            model.gravity,
            solver.particle_q_prev,
            state_in.particle_q, 
            state_in.particle_qd,
            model.particle_inv_mass,
            state_in.particle_f,
            model.particle_flags,
        ],
        outputs=[
            solver.inertia,
            solver.particle_displacements,
        ],
        dim=model.particle_count,
        device=solver.device,
    )
    
    # Initial truncation (after forward_step) - using mode 1
    solver.penetration_free_truncation(state_in.particle_q)
    
    # NOTE: We don't record init residual here because particle_forces hasn't been
    # computed yet - forces are only populated inside the iteration loop.
    
    # =========================================================================
    # Now switch to the actual truncation mode for iterations
    # =========================================================================
    solver.truncation_mode = truncation_mode
    
    # Mode 3 (Dykstra): Initialize projection arrays if not already present
    if truncation_mode == 3:
        if not hasattr(solver, 'project_t_vt') or solver.project_t_vt is None:
            solver.project_t_vt = wp.zeros(
                dtype=float, 
                shape=(len(solver.trimesh_collision_detector.collision_info.vertex_colliding_triangles) // 2,), 
                device=solver.device
            )
            solver.project_t_ee = wp.zeros(
                dtype=float, 
                shape=solver.trimesh_collision_detector.collision_info.edge_colliding_edges.shape, 
                device=solver.device
            )
            solver.project_t_tv = wp.zeros(
                dtype=float, 
                shape=(len(solver.trimesh_collision_detector.collision_info.triangle_colliding_vertices) * 3,), 
                device=solver.device
            )
            solver.dis_out = wp.zeros_like(solver.particle_displacements, device=solver.device)
    
    # =========================================================================
    # VBD iterations (timed)
    # =========================================================================
    wp.synchronize()  # Ensure all initialization work is complete before timing
    iteration_start_time = time.perf_counter()
    
    for _iter in range(num_iterations):
        # Collision detection at appropriate intervals (controlled by collision_detection_interval)
        if collision_detection_interval >= 1 and _iter % collision_detection_interval == 0:
            solver.collision_detection_penetration_free(state_in)
        
        # Clear forces and hessians
        solver.particle_forces.zero_()
        solver.particle_hessians.zero_()
        
        # Process all colors in this iteration
        for color in range(len(model.particle_color_groups)):
            particle_ids_in_color = model.particle_color_groups[color]
            
            # Accumulate self-contact forces
            if solver.handle_self_contact:
                wp.launch(
                    kernel=vbd_module.accumulate_self_contact_force_and_hessian,
                    dim=solver.self_contact_evaluation_kernel_launch_size,
                    inputs=[
                        dt,
                        color,
                        solver.particle_q_prev,
                        state_in.particle_q,
                        model.particle_colors,
                        model.tri_indices,
                        model.edge_indices,
                        solver.trimesh_collision_info,
                        solver.self_contact_radius,
                        model.soft_contact_ke,
                        model.soft_contact_kd,
                        model.soft_contact_mu,
                        solver.friction_epsilon,
                        solver.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    ],
                    outputs=[solver.particle_forces, solver.particle_hessians],
                    device=solver.device,
                    max_blocks=model.device.sm_count,
                )
            
            # Solve elasticity (using tile version for better performance)
            wp.launch(
                kernel=solve_elasticity_tile_measure_force,
                dim=particle_ids_in_color.size * vbd_module.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                block_dim=vbd_module.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                inputs=[
                    dt,
                    particle_ids_in_color,
                    solver.particle_q_prev,
                    state_in.particle_q,
                    model.particle_mass,
                    solver.inertia,
                    model.particle_flags,
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_materials,
                    model.tri_areas,
                    model.edge_indices,
                    model.edge_rest_angle,
                    model.edge_rest_length,
                    model.edge_bending_properties,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    solver.adjacency,
                    solver.particle_forces,
                    solver.particle_hessians,
                ],
                outputs=[
                    solver.particle_displacements,
                ],
                device=solver.device,
            )
            
            # Mode 3 (Dykstra): Apply Hessian-weighted Dykstra projection before truncation
            if truncation_mode == 3:
                # Zero out Dykstra projection arrays for fresh state
                solver.project_t_vt.zero_()
                solver.project_t_ee.zero_()
                solver.project_t_tv.zero_()
                
                wp.launch(
                    kernel=vbd_module.hessian_dykstra_projection,
                    inputs=[
                        solver.dykstra_iterations,
                        solver.pos_prev_collision_detection,
                        solver.particle_displacements,
                        solver.particle_hessians,
                        model.tri_indices,
                        model.edge_indices,
                        solver.adjacency,
                        solver.trimesh_collision_info,
                        1e-7,  # parallel_eps
                        solver.project_t_vt,
                        solver.project_t_ee,
                        solver.project_t_tv,
                    ],
                    outputs=[
                        solver.dis_out,
                    ],
                    dim=model.particle_count,
                    device=solver.device,
                )
                # Copy Dykstra output back to particle_displacements for truncation
                wp.copy(solver.particle_displacements, solver.dis_out)
            
            # Apply truncation after each color
            solver.penetration_free_truncation(state_in.particle_q)
        
        # Record residual after this iteration (all colors processed)
        wp.synchronize()
        iter_residual = compute_force_residual(solver, state_in, dt)
        iter_residual["iteration"] = _iter + 1  # 1-indexed
        residuals.append(iter_residual)
    
    # Final synchronize and record total iteration time
    wp.synchronize()
    iteration_end_time = time.perf_counter()
    total_iteration_time = iteration_end_time - iteration_start_time
    
    # Compute timing statistics
    timing_info = {
        "total_time_sec": total_iteration_time,
        "num_iterations": num_iterations,
        "time_per_iteration_ms": (total_iteration_time / num_iterations) * 1000,
        "iterations_per_second": num_iterations / total_iteration_time if total_iteration_time > 0 else 0,
    }
    
    # Restore original truncation mode
    solver.truncation_mode = original_mode
    
    return residuals, timing_info


def run_convergence_comparison(
    sim: TwistClothSimulator,
    solver: SolverVBD,
    dt: float,
    num_iterations: int,
    recovery_state_path: str,
    collision_detection_interval: int = -1,
) -> dict:
    """
    Run convergence comparison for all three truncation modes.
    
    For each mode, we:
    1. Reset to the recovery state
    2. Run VBD iterations
    3. Collect force residuals
    
    Returns:
        Dict with results for each mode
    """
    results = {}
    mode_names = {0: "Isotropic", 1: "Planar", 2: "CCD", 3: "Dykstra"}
    
    for mode in [0, 1, 2, 3]:  # Include all truncation modes
        print(f"\n{'='*60}")
        print(f"Running convergence test: Mode {mode} ({mode_names[mode]})")
        print(f"{'='*60}")
        
        # Reload the recovery state to reset
        frame_id = sim.load_recovery_state(recovery_state_path)
        
        # Release the right side (same as what happens at the release point)
        flags = sim.model.particle_flags.numpy()
        for fixed_vertex_id in sim.right_side:
            flags[fixed_vertex_id] = flags[fixed_vertex_id] | ParticleFlags.ACTIVE
        flags_new = wp.array(flags, dtype=wp.uint32, device=sim.model.particle_flags.device)
        wp.copy(sim.model.particle_flags, flags_new)
        
        # Set rotation time past end_time
        sim.t.fill_(sim.rot_end_time + 1.0)
        
        # Run iterations and collect residuals
        residuals, timing_info = run_vbd_iterations_with_residuals(
            sim, solver, dt, num_iterations, truncation_mode=mode,
            collision_detection_interval=collision_detection_interval,
        )
        
        results[mode_names[mode]] = {
            "mode": mode,
            "residuals": residuals,
            "timing": timing_info,
        }
        
        # Print summary (per-vertex average force)
        print(f"  Initial avg force: {residuals[0]['mean_norm']:.6e}")
        print(f"  Final avg force:   {residuals[-1]['mean_norm']:.6e}")
        reduction = residuals[0]['mean_norm'] / residuals[-1]['mean_norm'] if residuals[-1]['mean_norm'] > 0 else float('inf')
        print(f"  Reduction factor:  {reduction:.2f}x")
        print(f"  Total time:        {timing_info['total_time_sec']:.3f}s")
        print(f"  Time per iter:     {timing_info['time_per_iteration_ms']:.3f}ms")
    
    return results


def plot_convergence(results: dict, output_path: str = None):
    """
    Plot convergence curves for all truncation modes.
    """
    plt.figure(figsize=(12, 8))
    
    colors = {"Isotropic": "blue", "Planar": "green", "CCD": "red", "Dykstra": "purple"}
    markers = {"Isotropic": "o", "Planar": "s", "CCD": "^", "Dykstra": "d"}
    
    for mode_name, data in results.items():
        residuals = data["residuals"]
        iterations = [r["iteration"] for r in residuals]
        mean_norms = [r["mean_norm"] for r in residuals]
        
        plt.semilogy(
            iterations, mean_norms,
            label=mode_name,
            color=colors.get(mode_name, "black"),
            marker=markers.get(mode_name, "x"),
            markersize=4,
            linewidth=1.5,
        )
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Force Residual (Per-Vertex Average)", fontsize=12)
    plt.title("VBD Convergence: Per-Vertex Average Force vs Iteration", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def main():
    # Hardcoded recovery state path
    recovery_state_dir = r"D:\Data\DAT_Sim\cloth_twist_convergence\res_100x100_truncation_1_iter_10_20260113_225325"
    recovery_state_path = os.path.join(recovery_state_dir, "recovery_state_000600.npz")

    if not os.path.exists(recovery_state_path):
        print(f"Error: Recovery state file not found: {recovery_state_path}")
        print("Please update the path to a valid recovery state file.")
        sys.exit(1)

    print(f"Loading recovery state from: {recovery_state_path}")

    # Load the run config from the recovery state directory
    loaded_config = load_run_config(recovery_state_dir)

    # Configuration
    num_iterations = 500  # Number of VBD iterations to run
    collision_detection_interval = 20  # -1 = no redo during iterations, N = redo every N iterations
    
    # Create config for evaluation (use loaded config as base)
    eval_config = {
        **loaded_config,
        "do_rendering": False,
        "write_output": False,
        "write_video": False,
        "use_cuda_graph": False,
        "sim_num_frames": 1,
    }

    # Create simulator
    sim = TwistClothSimulator(eval_config)
    sim.finalize()

    # Load recovery state
    frame_id = sim.load_recovery_state(recovery_state_path)
    print(f"Loaded state from frame {frame_id}, sim_time={sim.sim_time:.4f}s")

    # Get solver and dt
    solver = sim.solver
    dt = sim.dt

    print(f"\nConfiguration:")
    print(f"  dt = {dt}")
    print(f"  num_iterations = {num_iterations}")
    print(f"  collision_detection_interval = {collision_detection_interval}")
    print(f"  Total particles: {solver.model.particle_count}")
    print(f"  Total colors: {len(solver.model.particle_color_groups)}")

    # Run convergence comparison
    results = run_convergence_comparison(
        sim, solver, dt, num_iterations, recovery_state_path,
        collision_detection_interval=collision_detection_interval,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(recovery_state_path)
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"convergence_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Save error curves as CSV - one file per mode for easy plotting
    for mode_name, data in results.items():
        csv_path = os.path.join(output_dir, f"convergence_{mode_name}_{timestamp}.csv")
        residuals = data["residuals"]
        with open(csv_path, "w") as f:
            f.write("iteration,mean_norm,l2_norm,linf_norm\n")
            for r in residuals:
                f.write(f"{r['iteration']},{r['mean_norm']:.6e},{r['l2_norm']:.6e},{r['linf_norm']:.6e}\n")
        print(f"  {mode_name} CSV saved to: {csv_path}")
    
    # Save as NumPy arrays - one file per mode
    for mode_name, data in results.items():
        npz_path = os.path.join(output_dir, f"convergence_{mode_name}_{timestamp}.npz")
        residuals = data["residuals"]
        np.savez(
            npz_path,
            iterations=np.array([r["iteration"] for r in residuals]),
            mean_norm=np.array([r["mean_norm"] for r in residuals]),
            l2_norm=np.array([r["l2_norm"] for r in residuals]),
            linf_norm=np.array([r["linf_norm"] for r in residuals]),
        )
        print(f"  {mode_name} NPZ saved to: {npz_path}")
    
    # Plot convergence
    plot_path = os.path.join(output_dir, f"convergence_plot_{timestamp}.pdf")
    plot_convergence(results, plot_path)
    
    # Print final comparison table
    print("\n" + "="*90)
    print("CONVERGENCE SUMMARY (Per-Vertex Average Force)")
    print("="*90)
    print(f"{'Mode':<15} {'Initial Avg':>15} {'Final Avg':>15} {'Reduction':>12} {'Time (s)':>12} {'ms/iter':>12}")
    print("-"*90)
    
    for mode_name, data in results.items():
        residuals = data["residuals"]
        timing = data["timing"]
        init_avg = residuals[0]["mean_norm"]
        final_avg = residuals[-1]["mean_norm"]
        reduction = init_avg / final_avg if final_avg > 0 else float('inf')
        print(f"{mode_name:<15} {init_avg:>15.6e} {final_avg:>15.6e} {reduction:>11.2f}x {timing['total_time_sec']:>11.3f}s {timing['time_per_iteration_ms']:>11.3f}")
    
    print("="*90)
    
    # Print timing comparison
    print("\nTIMING COMPARISON")
    print("-"*50)
    mode_times = {name: data["timing"]["total_time_sec"] for name, data in results.items()}
    baseline_mode = "Planar"
    if baseline_mode in mode_times:
        baseline_time = mode_times[baseline_mode]
        for mode_name, total_time in mode_times.items():
            speedup = baseline_time / total_time if total_time > 0 else float('inf')
            print(f"  {mode_name:<12}: {total_time:.3f}s ({speedup:.2f}x vs {baseline_mode})")
    print("-"*50)


if __name__ == "__main__":
    main()
