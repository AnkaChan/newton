# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Convergence Evaluation Script
#
# This script loads a recovery state and evaluates displacement before and
# after truncation for mode 0 (isometric) and mode 1 (planar) on the first
# color group only.
###########################################################################

import os
import sys

import numpy as np
import warp as wp

# Import from the save_at_release script
from example_twist_save_at_release import TwistClothSimulator, example_config

# Import kernels from the VBD solver module (internal)
import newton._src.solvers.vbd.solver_vbd as vbd_module
from newton import ParticleFlags
from newton.solvers import SolverVBD


def evaluate_one_iteration_displacement(
    sim: TwistClothSimulator,
    solver: SolverVBD,
    dt: float,
):
    """
    Run one VBD iteration for the first color only and capture displacement
    before and after truncation (mode 0 and mode 1).

    This follows the EXACT same flow as simulate_one_step_no_tile:
    1. Collision detection (before forward_step)
    2. Forward step (initialization)
    3. Truncation (after forward_step - initialization truncation)
    4. Collision detection (at iter 0 if collision_detection_interval == 0)
    5. Clear forces/hessians
    6. For color 0: accumulate forces, solve elasticity, then truncation

    Returns:
        dict with:
            - displacement_before_truncation: displacement after solve_elasticity
            - displacement_after_truncation_mode0: after isometric truncation
            - displacement_after_truncation_mode1: after planar truncation
            - first_color_particle_ids: indices of particles in first color
    """
    model = solver.model
    state_in = sim.state_0

    # =========================================================================
    # Step 1: Collision detection BEFORE forward_step (same as original)
    # =========================================================================
    solver.collision_detection_penetration_free(state_in)

    # =========================================================================
    # Step 2: Forward step (initialization)
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
    # Step 3: Evaluate INITIALIZATION truncation (after forward_step)
    # =========================================================================
    wp.synchronize()
    init_disp_before = solver.particle_displacements.numpy().copy()

    # Compute conservative bounds for mode 0
    wp.launch(
        kernel=vbd_module.compute_particle_conservative_bound,
        inputs=[
            solver.conservative_bound_relaxation,
            solver.self_contact_margin,
            solver.adjacency,
            solver.trimesh_collision_detector.collision_info,
        ],
        outputs=[
            solver.particle_conservative_bounds,
        ],
        dim=model.particle_count,
        device=solver.device,
    )

    # Mode 0 (isometric) for initialization
    init_disp_mode0 = wp.zeros_like(solver.particle_displacements)
    wp.copy(init_disp_mode0, solver.particle_displacements)
    init_pos_out_mode0 = wp.zeros_like(solver.particle_displacements)
    wp.launch(
        kernel=vbd_module.apply_conservative_bound_truncation,
        inputs=[
            init_disp_mode0,
            solver.pos_prev_collision_detection,
            solver.particle_conservative_bounds,
            init_pos_out_mode0,
        ],
        dim=model.particle_count,
        device=solver.device,
    )
    wp.synchronize()
    init_disp_after_mode0 = init_disp_mode0.numpy().copy()

    # Mode 1 (planar) for initialization
    init_disp_mode1 = wp.zeros_like(solver.particle_displacements)
    wp.copy(init_disp_mode1, solver.particle_displacements)
    init_truncation_ts = wp.zeros(model.particle_count, dtype=float, device=solver.device)
    init_truncation_ts.fill_(1.0)
    wp.launch(
        kernel=vbd_module.apply_planar_truncation_parallel_by_collision,
        inputs=[
            solver.pos_prev_collision_detection,
            init_disp_mode1,
            model.tri_indices,
            model.edge_indices,
            solver.trimesh_collision_info,
            solver.trimesh_collision_detector.edge_edge_parallel_epsilon,
            solver.conservative_bound_relaxation * 2,
        ],
        outputs=[
            init_truncation_ts,
        ],
        dim=solver.self_contact_evaluation_kernel_launch_size,
        device=solver.device,
    )
    wp.launch(
        kernel=vbd_module.apply_truncation_ts,
        dim=model.particle_count,
        inputs=[
            solver.pos_prev_collision_detection,
            init_disp_mode1,
            init_truncation_ts,
            init_disp_mode1,
            None,
            solver.self_contact_margin * solver.conservative_bound_relaxation,
        ],
        device=solver.device,
    )
    wp.synchronize()
    init_disp_after_mode1 = init_disp_mode1.numpy().copy()

    # Now apply actual truncation (using mode 1 as the simulation uses)
    solver.penetration_free_truncation(state_in.particle_q)

    # =========================================================================
    # Step 4: Collision detection at iteration 0 (if collision_detection_interval == 0)
    # =========================================================================
    _iter = 0
    if (solver.collision_detection_interval == 0 and _iter == 0) or (
        solver.collision_detection_interval >= 1 and _iter % solver.collision_detection_interval == 0
    ):
        solver.collision_detection_penetration_free(state_in)

    # =========================================================================
    # Step 5: Clear forces and hessians
    # =========================================================================
    solver.particle_forces.zero_()
    solver.particle_hessians.zero_()

    # =========================================================================
    # Step 6: Process color 0 only
    # =========================================================================
    color = 0
    first_color_particle_ids = model.particle_color_groups[color]

    # Accumulate self-contact forces for color 0
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

    # =========================================================================
    # Step 7: Solve elasticity for color 0
    # =========================================================================
    wp.launch(
        kernel=vbd_module.solve_elasticity,
        dim=first_color_particle_ids.size,
        inputs=[
            dt,
            first_color_particle_ids,
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

    # =========================================================================
    # Step 8: Save displacement BEFORE truncation
    # =========================================================================
    wp.synchronize()
    displacement_before_truncation = solver.particle_displacements.numpy().copy()

    # =========================================================================
    # Step 9: Compute conservative bounds for mode 0 comparison
    # (This is normally only done when truncation_mode == 0 in collision_detection)
    # =========================================================================
    wp.launch(
        kernel=vbd_module.compute_particle_conservative_bound,
        inputs=[
            solver.conservative_bound_relaxation,
            solver.self_contact_margin,
            solver.adjacency,
            solver.trimesh_collision_detector.collision_info,
        ],
        outputs=[
            solver.particle_conservative_bounds,
        ],
        dim=model.particle_count,
        device=solver.device,
    )

    # =========================================================================
    # Step 10: Apply truncation mode 0 (isometric) - save to separate buffer
    # =========================================================================
    disp_mode0 = wp.zeros_like(solver.particle_displacements)
    wp.copy(disp_mode0, solver.particle_displacements)
    pos_out_mode0 = wp.zeros_like(solver.particle_displacements)  # dummy output for positions

    wp.launch(
        kernel=vbd_module.apply_conservative_bound_truncation,
        inputs=[
            disp_mode0,  # particle_displacements (modified in-place)
            solver.pos_prev_collision_detection,  # pos_prev_collision_detection
            solver.particle_conservative_bounds,  # particle_conservative_bounds
            pos_out_mode0,  # particle_q_out (required, will be written)
        ],
        dim=model.particle_count,
        device=solver.device,
    )
    wp.synchronize()
    displacement_after_truncation_mode0 = disp_mode0.numpy().copy()

    # =========================================================================
    # Step 11: Apply truncation mode 1 (planar) - save to separate buffer
    # =========================================================================
    disp_mode1 = wp.zeros_like(solver.particle_displacements)
    wp.copy(disp_mode1, solver.particle_displacements)

    truncation_ts_mode1 = wp.zeros(model.particle_count, dtype=float, device=solver.device)
    truncation_ts_mode1.fill_(1.0)

    wp.launch(
        kernel=vbd_module.apply_planar_truncation_parallel_by_collision,
        inputs=[
            solver.pos_prev_collision_detection,
            disp_mode1,
            model.tri_indices,
            model.edge_indices,
            solver.trimesh_collision_info,
            solver.trimesh_collision_detector.edge_edge_parallel_epsilon,
            solver.conservative_bound_relaxation * 2,
        ],
        outputs=[
            truncation_ts_mode1,
        ],
        dim=solver.self_contact_evaluation_kernel_launch_size,
        device=solver.device,
    )

    wp.launch(
        kernel=vbd_module.apply_truncation_ts,
        dim=model.particle_count,
        inputs=[
            solver.pos_prev_collision_detection,
            disp_mode1,
            truncation_ts_mode1,
            disp_mode1,  # output to same buffer
            None,  # don't update positions (this kernel checks for None)
            solver.self_contact_margin * solver.conservative_bound_relaxation,
        ],
        device=solver.device,
    )
    wp.synchronize()
    displacement_after_truncation_mode1 = disp_mode1.numpy().copy()

    # Get first color particle indices
    first_color_ids = first_color_particle_ids.numpy()

    return {
        # Initialization step (forward_step) truncation
        "init_disp_before": init_disp_before,
        "init_disp_after_mode0": init_disp_after_mode0,
        "init_disp_after_mode1": init_disp_after_mode1,
        "init_truncation_ts_mode1": init_truncation_ts.numpy(),
        # Solve step (solve_elasticity) truncation
        "solve_disp_before": displacement_before_truncation,
        "solve_disp_after_mode0": displacement_after_truncation_mode0,
        "solve_disp_after_mode1": displacement_after_truncation_mode1,
        "solve_truncation_ts_mode1": truncation_ts_mode1.numpy(),
        # Particle info
        "first_color_particle_ids": first_color_ids,
    }


def compute_displacement_stats(displacements: np.ndarray, particle_ids: np.ndarray):
    """Compute statistics on displacement magnitudes for given particle IDs."""
    disp_subset = displacements[particle_ids]
    magnitudes = np.linalg.norm(disp_subset, axis=1)

    return {
        "mean": np.mean(magnitudes),
        "max": np.max(magnitudes),
        "min": np.min(magnitudes),
        "std": np.std(magnitudes),
        "median": np.median(magnitudes),
        "count": len(magnitudes),
        "num_nonzero": np.sum(magnitudes > 1e-10),
    }


def main():
    # Hardcoded recovery state path
    recovery_state_dir = r"C:\Data\DAT_Sim\cloth_twist_convergence\res_100x100_truncation_1_iter_10_20260113_151401"
    recovery_state_path = os.path.join(recovery_state_dir, "recovery_state_000600.npz")

    if not os.path.exists(recovery_state_path):
        print(f"Error: Recovery state file not found: {recovery_state_path}")
        sys.exit(1)

    print(f"Loading recovery state from: {recovery_state_path}")

    # Create config for evaluation (with rendering)
    eval_config = {
        **example_config,
        "do_rendering": True,  # Enable rendering
        "write_output": False,
        "write_video": False,
        "use_cuda_graph": False,  # Disable CUDA graph for custom step
        "sim_num_frames": 1,  # We only need one step
    }

    # Create simulator
    sim = TwistClothSimulator(eval_config)
    sim.finalize()

    # Load recovery state
    frame_id = sim.load_recovery_state(recovery_state_path)
    print(f"Loaded state from frame {frame_id}, sim_time={sim.sim_time:.4f}s")

    # Release the right side (same as what happens at the release point in the original)
    # This re-enables the ACTIVE flag for right_side vertices
    flags = sim.model.particle_flags.numpy()
    for fixed_vertex_id in sim.right_side:
        flags[fixed_vertex_id] = flags[fixed_vertex_id] | ParticleFlags.ACTIVE
    flags_new = wp.array(flags, dtype=wp.uint32, device=sim.model.particle_flags.device)
    wp.copy(sim.model.particle_flags, flags_new)
    print(f"Released right side ({len(sim.right_side)} vertices)")

    # Set rotation time past end_time so the rotation kernel won't apply anymore
    # (The apply_rotation kernel checks: if cur_t > end_time: return)
    sim.t.fill_(sim.rot_end_time + 1.0)
    print(f"Set rotation time to {sim.rot_end_time + 1.0} (past end_time={sim.rot_end_time})")

    # Get the solver
    solver = sim.solver

    # Compute dt
    dt = sim.dt

    # Print average initial velocity
    initial_velocities = sim.state_0.particle_qd.numpy()
    avg_velocity = np.mean(np.linalg.norm(initial_velocities, axis=1))
    print(f"\nInitial average velocity magnitude: {avg_velocity:.6f}")

    # Run multiple simulation steps to let velocity propagate (VBD is a local solver)
    # Use truncation mode 1 (planar) for these steps
    # num_warmup_steps = 100
    num_warmup_steps = 1
    print(f"\nRunning {num_warmup_steps} simulation steps (truncation mode 1) to build up velocity...")
    solver.truncation_mode = 1
    for step_i in range(num_warmup_steps):
        sim.run_step()
        # Render each frame
        # sim.render()
        if (step_i + 1) % 1 == 0:
            wp.synchronize()
            velocities = sim.state_0.particle_qd.numpy()
            avg_vel = np.mean(np.linalg.norm(velocities, axis=1))
            print(f"  Step {step_i + 1}/{num_warmup_steps}: avg velocity = {avg_vel:.6f}")
    wp.synchronize()

    # Print velocity after warmup
    velocities_after_warmup = sim.state_0.particle_qd.numpy()
    avg_velocity_after = np.mean(np.linalg.norm(velocities_after_warmup, axis=1))
    print(f"Average velocity after {num_warmup_steps} steps: {avg_velocity_after:.6f}")

    print("\nNow evaluating one iteration for first color (color 0)...")
    print(f"  dt = {dt}")
    print(f"  First color has {solver.model.particle_color_groups[0].size} particles")
    print(f"  Total particles: {solver.model.particle_count}")
    print(f"  Total colors: {len(solver.model.particle_color_groups)}")

    # Evaluate one iteration
    results = evaluate_one_iteration_displacement(sim, solver, dt)

    # Compute statistics for first color particles only
    first_color_ids = results["first_color_particle_ids"]

    def print_truncation_stats(
        name: str,
        disp_before: np.ndarray,
        disp_after_mode0: np.ndarray,
        disp_after_mode1: np.ndarray,
        particle_ids: np.ndarray,
    ):
        """Print truncation statistics for a given step."""
        print(f"\n{'=' * 60}")
        print(f"{name} TRUNCATION STATISTICS (First Color Only)")
        print(f"{'=' * 60}")

        print("\n--- Before Truncation ---")
        stats_before = compute_displacement_stats(disp_before, particle_ids)
        for k, v in stats_before.items():
            print(f"  {k}: {v}")

        print("\n--- After Truncation Mode 0 (Isometric) ---")
        stats_mode0 = compute_displacement_stats(disp_after_mode0, particle_ids)
        for k, v in stats_mode0.items():
            print(f"  {k}: {v}")

        print("\n--- After Truncation Mode 1 (Planar) ---")
        stats_mode1 = compute_displacement_stats(disp_after_mode1, particle_ids)
        for k, v in stats_mode1.items():
            print(f"  {k}: {v}")

        # Compute truncation ratios
        disp_b = disp_before[particle_ids]
        disp_m0 = disp_after_mode0[particle_ids]
        disp_m1 = disp_after_mode1[particle_ids]

        mag_b = np.linalg.norm(disp_b, axis=1)
        mag_m0 = np.linalg.norm(disp_m0, axis=1)
        mag_m1 = np.linalg.norm(disp_m1, axis=1)

        # Use a meaningful threshold to filter out near-zero displacements
        # Use 1% of mean displacement as threshold
        disp_threshold = max(1e-6, 0.01 * np.mean(mag_b))
        significant_mask = mag_b > disp_threshold
        num_significant = np.sum(significant_mask)

        print(
            f"\n--- Filtering: {num_significant}/{len(mag_b)} particles have significant displacement (>{disp_threshold:.2e}) ---"
        )

        if np.any(significant_mask):
            ratio_m0 = mag_m0[significant_mask] / mag_b[significant_mask]
            ratio_m1 = mag_m1[significant_mask] / mag_b[significant_mask]

            print("\n--- Truncation Ratios (after/before, significant displacements only) ---")
            print(
                f"  Mode 0 (Isometric): mean={np.mean(ratio_m0):.4f}, min={np.min(ratio_m0):.4f}, max={np.max(ratio_m0):.4f}"
            )
            print(
                f"  Mode 1 (Planar):    mean={np.mean(ratio_m1):.4f}, min={np.min(ratio_m1):.4f}, max={np.max(ratio_m1):.4f}"
            )

            # Count how many were truncated
            truncated_m0 = np.sum(ratio_m0 < 0.999)
            truncated_m1 = np.sum(ratio_m1 < 0.999)
            print(
                f"\n  Mode 0: {truncated_m0}/{len(ratio_m0)} particles truncated ({100 * truncated_m0 / len(ratio_m0):.1f}%)"
            )
            print(
                f"  Mode 1: {truncated_m1}/{len(ratio_m1)} particles truncated ({100 * truncated_m1 / len(ratio_m1):.1f}%)"
            )

            # Improvement ratio: mode1 / mode0 (how much more displacement mode1 preserves)
            # Only compute for particles where mode0 has significant displacement
            mode0_significant = mag_m0[significant_mask] > disp_threshold
            if np.any(mode0_significant):
                improvement_ratio = (
                    mag_m1[significant_mask][mode0_significant] / mag_m0[significant_mask][mode0_significant]
                )

                print(
                    f"\n--- Mode 1 Improvement over Mode 0 ({np.sum(mode0_significant)} particles with significant mode0 disp) ---"
                )
                print(f"  Mean:   {np.mean(improvement_ratio):.4f}")
                print(f"  Median: {np.median(improvement_ratio):.4f}")
                print(f"  Min:    {np.min(improvement_ratio):.4f}")
                print(f"  Max:    {np.max(improvement_ratio):.4f}")
                print("  Percentiles:")
                for p in [5, 10, 25, 50, 75, 90, 95, 99]:
                    print(f"    {p}th: {np.percentile(improvement_ratio, p):.4f}")

                # Count particles where mode1 > mode0 (improvement)
                improved = np.sum(improvement_ratio > 1.001)
                same = np.sum((improvement_ratio >= 0.999) & (improvement_ratio <= 1.001))
                worse = np.sum(improvement_ratio < 0.999)
                print(f"\n  Improved (mode1 > mode0): {improved} ({100 * improved / len(improvement_ratio):.1f}%)")
                print(f"  Same:                     {same} ({100 * same / len(improvement_ratio):.1f}%)")
                print(f"  Worse (mode1 < mode0):    {worse} ({100 * worse / len(improvement_ratio):.1f}%)")

    # Print stats for INITIALIZATION step (forward_step truncation)
    print_truncation_stats(
        "INITIALIZATION (forward_step)",
        results["init_disp_before"],
        results["init_disp_after_mode0"],
        results["init_disp_after_mode1"],
        first_color_ids,
    )

    # Print stats for SOLVE step (solve_elasticity truncation)
    print_truncation_stats(
        "SOLVE (solve_elasticity)",
        results["solve_disp_before"],
        results["solve_disp_after_mode0"],
        results["solve_disp_after_mode1"],
        first_color_ids,
    )

    # Save results to file
    output_dir = os.path.dirname(recovery_state_path)
    output_file = os.path.join(output_dir, "convergence_evaluation.npz")
    np.savez(
        output_file,
        # Initialization step
        init_disp_before=results["init_disp_before"],
        init_disp_after_mode0=results["init_disp_after_mode0"],
        init_disp_after_mode1=results["init_disp_after_mode1"],
        init_truncation_ts_mode1=results["init_truncation_ts_mode1"],
        # Solve step
        solve_disp_before=results["solve_disp_before"],
        solve_disp_after_mode0=results["solve_disp_after_mode0"],
        solve_disp_after_mode1=results["solve_disp_after_mode1"],
        solve_truncation_ts_mode1=results["solve_truncation_ts_mode1"],
        # Particle info
        first_color_particle_ids=first_color_ids,
    )
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
