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
#   0: Isometric (conservative bound)
#   1: Planar (DAT - Divide and Truncate)
#   2: CCD (Continuous Collision Detection - global min t)
###########################################################################

import os
import sys
import json
from datetime import datetime

import numpy as np
import warp as wp
import matplotlib.pyplot as plt

# Import from the save_at_release script
from example_twist_save_at_release import TwistClothSimulator, example_config

# Import kernels from the VBD solver module (internal)
import newton._src.solvers.vbd.solver_vbd as vbd_module
from newton import ParticleFlags
from newton.solvers import SolverVBD


def compute_force_residual(solver: SolverVBD, state, dt: float) -> dict:
    """
    Compute the force residual for all particles.
    
    The force residual measures how far the system is from equilibrium.
    A lower residual means better convergence.
    
    Returns:
        dict with various residual metrics
    """
    model = solver.model
    
    # Get current state
    positions = state.particle_q.numpy()
    velocities = state.particle_qd.numpy()
    masses = model.particle_mass.numpy()
    flags = model.particle_flags.numpy()
    
    # Compute force residual as: ||m * a|| where a = (v_new - v_old) / dt
    # For VBD, we can approximate this from the displacement
    displacements = solver.particle_displacements.numpy()
    
    # Force = mass * displacement / dt^2 (from Newton's 2nd law in implicit form)
    forces = masses[:, np.newaxis] * displacements / (dt * dt)
    
    # Only consider active particles
    active_mask = (flags & int(ParticleFlags.ACTIVE)) != 0
    active_forces = forces[active_mask]
    
    # Compute various norms
    force_norms = np.linalg.norm(active_forces, axis=1)
    
    return {
        "l2_norm": float(np.sqrt(np.sum(force_norms**2))),
        "linf_norm": float(np.max(force_norms)) if len(force_norms) > 0 else 0.0,
        "mean_norm": float(np.mean(force_norms)) if len(force_norms) > 0 else 0.0,
        "num_active": int(np.sum(active_mask)),
    }


def run_vbd_iterations_with_residuals(
    sim: TwistClothSimulator,
    solver: SolverVBD,
    dt: float,
    num_iterations: int,
    truncation_mode: int,
) -> list:
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
        List of residual dicts, one per iteration
    """
    model = solver.model
    state_in = sim.state_0
    
    # Set truncation mode
    original_mode = solver.truncation_mode
    solver.truncation_mode = truncation_mode
    
    residuals = []
    
    # =========================================================================
    # Collision detection BEFORE forward_step
    # =========================================================================
    solver.collision_detection_penetration_free(state_in)
    
    # =========================================================================
    # Forward step (initialization)
    # =========================================================================
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
    
    # =========================================================================
    # Initial truncation (after forward_step)
    # =========================================================================
    solver.penetration_free_truncation(state_in.particle_q)
    
    # Record initial residual (after initialization)
    wp.synchronize()
    init_residual = compute_force_residual(solver, state_in, dt)
    init_residual["iteration"] = 0
    init_residual["phase"] = "init"
    residuals.append(init_residual)
    
    # =========================================================================
    # VBD iterations
    # =========================================================================
    for _iter in range(num_iterations):
        # Collision detection at appropriate intervals
        if (solver.collision_detection_interval == 0 and _iter == 0) or (
            solver.collision_detection_interval >= 1 and _iter % solver.collision_detection_interval == 0
        ):
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
            
            # Solve elasticity
            wp.launch(
                kernel=vbd_module.solve_elasticity,
                dim=particle_ids_in_color.size,
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
            
            # Apply truncation after each color
            solver.penetration_free_truncation(state_in.particle_q)
        
        # Record residual after this iteration (all colors processed)
        wp.synchronize()
        iter_residual = compute_force_residual(solver, state_in, dt)
        iter_residual["iteration"] = _iter + 1
        iter_residual["phase"] = "solve"
        residuals.append(iter_residual)
    
    # Restore original truncation mode
    solver.truncation_mode = original_mode
    
    return residuals


def run_convergence_comparison(
    sim: TwistClothSimulator,
    solver: SolverVBD,
    dt: float,
    num_iterations: int,
    recovery_state_path: str,
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
    mode_names = {0: "Isometric", 1: "Planar", 2: "CCD"}
    
    for mode in [0, 1, 2]:
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
        residuals = run_vbd_iterations_with_residuals(
            sim, solver, dt, num_iterations, truncation_mode=mode
        )
        
        results[mode_names[mode]] = {
            "mode": mode,
            "residuals": residuals,
        }
        
        # Print summary
        print(f"  Initial L2 residual: {residuals[0]['l2_norm']:.6e}")
        print(f"  Final L2 residual:   {residuals[-1]['l2_norm']:.6e}")
        reduction = residuals[0]['l2_norm'] / residuals[-1]['l2_norm'] if residuals[-1]['l2_norm'] > 0 else float('inf')
        print(f"  Reduction factor:    {reduction:.2f}x")
    
    return results


def plot_convergence(results: dict, output_path: str = None):
    """
    Plot convergence curves for all truncation modes.
    """
    plt.figure(figsize=(12, 8))
    
    colors = {"Isometric": "blue", "Planar": "green", "CCD": "red"}
    markers = {"Isometric": "o", "Planar": "s", "CCD": "^"}
    
    for mode_name, data in results.items():
        residuals = data["residuals"]
        iterations = [r["iteration"] for r in residuals]
        l2_norms = [r["l2_norm"] for r in residuals]
        
        plt.semilogy(
            iterations, l2_norms,
            label=mode_name,
            color=colors.get(mode_name, "black"),
            marker=markers.get(mode_name, "x"),
            markersize=4,
            linewidth=1.5,
        )
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Force Residual (L2 Norm)", fontsize=12)
    plt.title("VBD Convergence: Force Residual vs Iteration", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def main():
    # Hardcoded recovery state path
    recovery_state_dir = r"C:\Data\DAT_Sim\cloth_twist_convergence\res_100x100_truncation_1_iter_10_20260113_151401"
    recovery_state_path = os.path.join(recovery_state_dir, "recovery_state_000600.npz")

    if not os.path.exists(recovery_state_path):
        print(f"Error: Recovery state file not found: {recovery_state_path}")
        print("Please update the path to a valid recovery state file.")
        sys.exit(1)

    print(f"Loading recovery state from: {recovery_state_path}")

    # Configuration
    num_iterations = 20  # Number of VBD iterations to run
    
    # Create config for evaluation
    eval_config = {
        **example_config,
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
    print(f"  Total particles: {solver.model.particle_count}")
    print(f"  Total colors: {len(solver.model.particle_color_groups)}")

    # Run convergence comparison
    results = run_convergence_comparison(
        sim, solver, dt, num_iterations, recovery_state_path
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(recovery_state_path)
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"convergence_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Plot convergence
    plot_path = os.path.join(output_dir, f"convergence_plot_{timestamp}.png")
    plot_convergence(results, plot_path)
    
    # Print final comparison table
    print("\n" + "="*70)
    print("CONVERGENCE SUMMARY")
    print("="*70)
    print(f"{'Mode':<15} {'Initial L2':>15} {'Final L2':>15} {'Reduction':>12}")
    print("-"*70)
    
    for mode_name, data in results.items():
        residuals = data["residuals"]
        init_l2 = residuals[0]["l2_norm"]
        final_l2 = residuals[-1]["l2_norm"]
        reduction = init_l2 / final_l2 if final_l2 > 0 else float('inf')
        print(f"{mode_name:<15} {init_l2:>15.6e} {final_l2:>15.6e} {reduction:>11.2f}x")
    
    print("="*70)


if __name__ == "__main__":
    main()
