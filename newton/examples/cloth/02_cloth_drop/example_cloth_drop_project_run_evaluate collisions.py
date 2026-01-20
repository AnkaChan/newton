# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Collision Evaluation Script
#
# This script loads a recovery state, runs one collision detection,
# and evaluates collision statistics (histogram and percentiles).
#
###########################################################################

import os

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from example_cloth_drop_project import ClothDropSimulator, example_config


def analyze_collisions(vertex_counts: np.ndarray, edge_counts: np.ndarray, output_path: str | None = None):
    """
    Analyze collision counts and display histograms and percentile statistics.

    Args:
        vertex_counts: Array of collision counts per vertex
        edge_counts: Array of collision counts per edge
        output_path: Optional path to save the analysis results
    """
    print("\n" + "=" * 70)
    print("COLLISION ANALYSIS REPORT")
    print("=" * 70)

    # =========================================================================
    # Vertex Collision Statistics
    # =========================================================================
    print("\n--- VERTEX-TRIANGLE COLLISIONS ---")
    print(f"Total vertices: {len(vertex_counts)}")
    print(f"Vertices with collisions: {np.sum(vertex_counts > 0)}")
    print(f"Total collision pairs: {np.sum(vertex_counts)}")
    print(f"Max collisions per vertex: {np.max(vertex_counts)}")
    print(f"Mean collisions per vertex: {np.mean(vertex_counts):.4f}")
    print(f"Std dev: {np.std(vertex_counts):.4f}")

    # Percentiles for vertices
    percentiles = [50, 75, 90, 95, 99, 99.9, 100]
    print("\nVertex Collision Percentiles:")
    vertex_percentile_values = {}
    for p in percentiles:
        val = np.percentile(vertex_counts, p)
        vertex_percentile_values[p] = val
        print(f"  {p:5.1f}th percentile: {val:.0f} collisions")

    # =========================================================================
    # Edge Collision Statistics
    # =========================================================================
    print("\n--- EDGE-EDGE COLLISIONS ---")
    print(f"Total edges: {len(edge_counts)}")
    print(f"Edges with collisions: {np.sum(edge_counts > 0)}")
    print(f"Total collision pairs: {np.sum(edge_counts)}")
    print(f"Max collisions per edge: {np.max(edge_counts)}")
    print(f"Mean collisions per edge: {np.mean(edge_counts):.4f}")
    print(f"Std dev: {np.std(edge_counts):.4f}")

    # Percentiles for edges
    print("\nEdge Collision Percentiles:")
    edge_percentile_values = {}
    for p in percentiles:
        val = np.percentile(edge_counts, p)
        edge_percentile_values[p] = val
        print(f"  {p:5.1f}th percentile: {val:.0f} collisions")

    # =========================================================================
    # Create Histograms
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Collision Count Distribution Analysis", fontsize=14, fontweight="bold")

    # Vertex collision histogram (linear scale)
    ax1 = axes[0, 0]
    max_v = int(np.max(vertex_counts))
    bins_v = np.arange(0, max_v + 2) - 0.5  # Centered bins
    ax1.hist(vertex_counts, bins=bins_v, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Number of Collisions per Vertex")
    ax1.set_ylabel("Count")
    ax1.set_title("Vertex-Triangle Collision Distribution")
    ax1.axvline(np.mean(vertex_counts), color="red", linestyle="--", label=f"Mean: {np.mean(vertex_counts):.2f}")
    ax1.axvline(
        np.percentile(vertex_counts, 95),
        color="orange",
        linestyle="--",
        label=f"95th: {np.percentile(vertex_counts, 95):.0f}",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Vertex collision histogram (log scale for y-axis)
    ax2 = axes[0, 1]
    ax2.hist(vertex_counts, bins=bins_v, color="steelblue", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Number of Collisions per Vertex")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Vertex-Triangle Collision Distribution (Log Scale)")
    ax2.set_yscale("log")
    ax2.axvline(np.mean(vertex_counts), color="red", linestyle="--", label=f"Mean: {np.mean(vertex_counts):.2f}")
    ax2.axvline(
        np.percentile(vertex_counts, 95),
        color="orange",
        linestyle="--",
        label=f"95th: {np.percentile(vertex_counts, 95):.0f}",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Edge collision histogram (linear scale)
    ax3 = axes[1, 0]
    max_e = int(np.max(edge_counts))
    bins_e = np.arange(0, max_e + 2) - 0.5
    ax3.hist(edge_counts, bins=bins_e, color="darkorange", edgecolor="black", alpha=0.7)
    ax3.set_xlabel("Number of Collisions per Edge")
    ax3.set_ylabel("Count")
    ax3.set_title("Edge-Edge Collision Distribution")
    ax3.axvline(np.mean(edge_counts), color="red", linestyle="--", label=f"Mean: {np.mean(edge_counts):.2f}")
    ax3.axvline(
        np.percentile(edge_counts, 95),
        color="purple",
        linestyle="--",
        label=f"95th: {np.percentile(edge_counts, 95):.0f}",
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Edge collision histogram (log scale for y-axis)
    ax4 = axes[1, 1]
    ax4.hist(edge_counts, bins=bins_e, color="darkorange", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("Number of Collisions per Edge")
    ax4.set_ylabel("Count (log scale)")
    ax4.set_title("Edge-Edge Collision Distribution (Log Scale)")
    ax4.set_yscale("log")
    ax4.axvline(np.mean(edge_counts), color="red", linestyle="--", label=f"Mean: {np.mean(edge_counts):.2f}")
    ax4.axvline(
        np.percentile(edge_counts, 95),
        color="purple",
        linestyle="--",
        label=f"95th: {np.percentile(edge_counts, 95):.0f}",
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the figure if output path is provided
    if output_path:
        fig_path = os.path.join(output_path, "collision_analysis.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nHistogram saved to: {fig_path}")

    plt.show()

    # =========================================================================
    # Create Cumulative Distribution Plot
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Cumulative Distribution of Collision Counts", fontsize=14, fontweight="bold")

    # Vertex CDF
    ax_v = axes2[0]
    sorted_v = np.sort(vertex_counts)
    cdf_v = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax_v.plot(sorted_v, cdf_v, color="steelblue", linewidth=2)
    ax_v.set_xlabel("Number of Collisions per Vertex")
    ax_v.set_ylabel("Cumulative Probability")
    ax_v.set_title("Vertex-Triangle Collision CDF")
    ax_v.grid(True, alpha=0.3)
    ax_v.axhline(0.95, color="orange", linestyle="--", alpha=0.7, label="95th percentile")
    ax_v.axhline(0.99, color="red", linestyle="--", alpha=0.7, label="99th percentile")
    ax_v.legend()

    # Edge CDF
    ax_e = axes2[1]
    sorted_e = np.sort(edge_counts)
    cdf_e = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    ax_e.plot(sorted_e, cdf_e, color="darkorange", linewidth=2)
    ax_e.set_xlabel("Number of Collisions per Edge")
    ax_e.set_ylabel("Cumulative Probability")
    ax_e.set_title("Edge-Edge Collision CDF")
    ax_e.grid(True, alpha=0.3)
    ax_e.axhline(0.95, color="purple", linestyle="--", alpha=0.7, label="95th percentile")
    ax_e.axhline(0.99, color="red", linestyle="--", alpha=0.7, label="99th percentile")
    ax_e.legend()

    plt.tight_layout()

    if output_path:
        fig2_path = os.path.join(output_path, "collision_cdf.png")
        plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
        print(f"CDF plot saved to: {fig2_path}")

    plt.show()

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} {'Vertex':<15} {'Edge':<15}")
    print("-" * 70)
    print(f"{'Total primitives':<35} {len(vertex_counts):<15} {len(edge_counts):<15}")
    print(f"{'Primitives with collisions':<35} {np.sum(vertex_counts > 0):<15} {np.sum(edge_counts > 0):<15}")
    print(f"{'Total collision pairs':<35} {np.sum(vertex_counts):<15} {np.sum(edge_counts):<15}")
    print(f"{'Max collisions':<35} {np.max(vertex_counts):<15} {np.max(edge_counts):<15}")
    print(f"{'Mean collisions':<35} {np.mean(vertex_counts):<15.4f} {np.mean(edge_counts):<15.4f}")
    print(f"{'95th percentile':<35} {vertex_percentile_values[95]:<15.0f} {edge_percentile_values[95]:<15.0f}")
    print(f"{'99th percentile':<35} {vertex_percentile_values[99]:<15.0f} {edge_percentile_values[99]:<15.0f}")
    print("=" * 70)

    return {
        "vertex": {
            "counts": vertex_counts,
            "percentiles": vertex_percentile_values,
        },
        "edge": {
            "counts": edge_counts,
            "percentiles": edge_percentile_values,
        },
    }


def run_collision_evaluation(recovery_state_path: str, config: dict | None = None):
    """
    Load a recovery state, run one collision detection, and analyze results.

    Args:
        recovery_state_path: Path to the recovery state .npz file
        config: Optional config dict (uses example_config if not provided)
    """
    if config is None:
        config = example_config.copy()

    # Disable rendering and output for evaluation
    config["do_rendering"] = False
    config["write_output"] = False
    config["write_video"] = False
    config["use_cuda_graph"] = False  # Need to disable for single-step evaluation

    print(f"Loading recovery state from: {recovery_state_path}")

    # Create simulator
    sim = ClothDropSimulator(config)
    sim.finalize()

    # Load the recovery state
    frame_id = sim.load_recovery_state(recovery_state_path)
    print(f"Loaded state from frame {frame_id}, sim_time = {sim.sim_time:.4f}")

    # Rebuild BVH with loaded positions
    print("Rebuilding BVH...")
    sim.rebuild_bvh()

    # Access the collision detector
    collision_detector = sim.solver.trimesh_collision_detector

    # Refit BVH with current positions
    print("Refitting collision detector...")
    collision_detector.refit(sim.state_0.particle_q)

    # Run vertex-triangle collision detection
    print("Running vertex-triangle collision detection...")
    collision_detector.vertex_triangle_collision_detection(
        max_query_radius=sim.solver.self_contact_margin,
        min_query_radius=sim.solver.rest_shape_contact_exclusion_radius,
        min_distance_filtering_ref_pos=sim.solver.rest_shape,
    )

    # Run edge-edge collision detection
    print("Running edge-edge collision detection...")
    collision_detector.edge_edge_collision_detection(
        max_query_radius=sim.solver.self_contact_margin,
        min_query_radius=sim.solver.rest_shape_contact_exclusion_radius,
        min_distance_filtering_ref_pos=sim.solver.rest_shape,
    )

    # Synchronize to ensure GPU operations are complete
    wp.synchronize()

    # Get collision counts from GPU
    vertex_collision_counts = collision_detector.vertex_colliding_triangles_count.numpy()
    edge_collision_counts = collision_detector.edge_colliding_edges_count.numpy()

    print("\nCollision detection complete!")
    print(f"  Vertex count: {len(vertex_collision_counts)}")
    print(f"  Edge count: {len(edge_collision_counts)}")

    # Get output path from recovery state directory
    output_path = os.path.dirname(recovery_state_path)

    # Analyze and visualize
    results = analyze_collisions(vertex_collision_counts, edge_collision_counts, output_path)

    # Save raw data
    if output_path:
        np.savez(
            os.path.join(output_path, "collision_counts.npz"),
            vertex_collision_counts=vertex_collision_counts,
            edge_collision_counts=edge_collision_counts,
            frame_id=frame_id,
            sim_time=sim.sim_time,
        )
        print(f"\nRaw collision data saved to: {os.path.join(output_path, 'collision_counts.npz')}")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # HARDCODE YOUR RECOVERY STATE PATH HERE
    # =========================================================================
    recovery_path = r"D:\Data\DAT_Sim\100_layers\20260103_232647\recovery_state_000100.npz"
    # =========================================================================

    if not os.path.exists(recovery_path):
        print(f"Error: Recovery state not found: {recovery_path}")
        exit(1)

    # Run evaluation
    run_collision_evaluation(recovery_path)
