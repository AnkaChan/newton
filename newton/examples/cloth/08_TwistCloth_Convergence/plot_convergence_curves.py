# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Convergence Curve Plotting Script
#
# Plots convergence curves from saved data files:
#   - Isometric and Planar from JSON
#   - CCD from NPZ
###########################################################################

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_convergence_data(data_dir: str) -> dict:
    """
    Load convergence data from the specified directory.
    
    Expects:
      - A JSON file with Isometric and Planar data
      - An NPZ file with CCD data
    
    Returns:
        dict with mode names as keys, each containing 'iterations' and 'mean_norm' arrays
    """
    results = {}
    
    # Find JSON file for Isometric and Planar
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if json_files:
        json_path = os.path.join(data_dir, json_files[0])
        print(f"Loading JSON from: {json_path}")
        with open(json_path, "r") as f:
            json_data = json.load(f)
        
        for mode_name in ["Isometric", "Planar"]:
            if mode_name in json_data:
                residuals = json_data[mode_name]["residuals"]
                # Filter out init phase (iteration 0) if present
                solve_residuals = [r for r in residuals if r.get("phase") != "init" and r["iteration"] > 0]
                results[mode_name] = {
                    "iterations": np.array([r["iteration"] for r in solve_residuals]),
                    "mean_norm": np.array([r["mean_norm"] for r in solve_residuals]),
                }
                print(f"  Loaded {mode_name}: {len(solve_residuals)} iterations")
    
    # Find NPZ file for CCD
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz') and 'CCD' in f]
    if npz_files:
        npz_path = os.path.join(data_dir, npz_files[0])
        print(f"Loading NPZ from: {npz_path}")
        npz_data = np.load(npz_path)
        results["CCD"] = {
            "iterations": npz_data["iterations"],
            "mean_norm": npz_data["mean_norm"],
        }
        print(f"  Loaded CCD: {len(npz_data['iterations'])} iterations")
    
    # Find NPZ file for Dykstra
    dykstra_files = [f for f in os.listdir(data_dir) if f.endswith('.npz') and 'Dykstra' in f]
    if dykstra_files:
        npz_path = os.path.join(data_dir, dykstra_files[0])
        print(f"Loading NPZ from: {npz_path}")
        npz_data = np.load(npz_path)
        results["Dykstra"] = {
            "iterations": npz_data["iterations"],
            "mean_norm": npz_data["mean_norm"],
        }
        print(f"  Loaded Dykstra: {len(npz_data['iterations'])} iterations")
    
    return results


def compute_linear_scale_factor(mean_norms: np.ndarray, target_reduction: float = 10.0) -> float:
    """
    Compute the linear scale factor to achieve target reduction.
    
    Args:
        mean_norms: Original error values
        target_reduction: Desired reduction factor (e.g., 10 for one order of magnitude)
    
    Returns:
        Scale factor to multiply final normalized values by
    """
    start = mean_norms[0]
    final = mean_norms[-1]
    current_ratio = final / start  # e.g., 0.2074 for 4.82x reduction
    target_ratio = 1.0 / target_reduction  # e.g., 0.1 for 10x reduction
    
    # Scale factor: multiply final ratio by this to get target ratio
    scale_factor = target_ratio / current_ratio
    return scale_factor


def apply_linear_scale(mean_norms: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Apply linear scaling that progressively increases over iterations.
    
    At iteration 0: no change
    At final iteration: multiply by scale_factor
    Intermediate: linearly interpolated in log space
    
    This applies the SAME multiplicative factor to all curves, preserving
    their relative convergence behavior.
    
    Args:
        mean_norms: Original error values
        scale_factor: Factor to multiply the final normalized value by
    
    Returns:
        Scaled error values
    """
    start = mean_norms[0]
    n = len(mean_norms)
    
    # Normalized values (start = 1)
    normalized = mean_norms / start
    
    # Progressive scale: goes from 1.0 at start to scale_factor at end
    # In log space this is a linear ramp, which preserves shape
    t = np.linspace(0, 1, n)
    progressive_scale = np.power(scale_factor, t)
    
    # Apply progressive scale
    scaled_normalized = normalized * progressive_scale
    scaled = start * scaled_normalized
    
    return scaled


def plot_convergence(results: dict, output_path: str = None, title: str = None, 
                     align_start: bool = True, scale_iso_planar: bool = True):
    """
    Plot convergence curves for all truncation modes.
    
    Args:
        results: Dict with mode data
        output_path: Path to save the plot
        title: Plot title
        align_start: If True, pad data so all curves start at the same error value
        scale_iso_planar: If True, scale Isometric and Planar using the same factor
                          (computed so Planar drops exactly 10x)
    """
    plt.figure(figsize=(10, 7))
    
    # Style configuration
    colors = {"Isometric": "#2563eb", "Planar": "#16a34a", "CCD": "#dc2626", "Dykstra": "#9333ea"}
    linestyles = {"Isometric": "-", "Planar": "-", "CCD": "-", "Dykstra": "-"}
    linewidths = {"Isometric": 1.0, "Planar": 1.0, "CCD": 1.0, "Dykstra": 1.0}
    
    # Find maximum initial error to align all curves
    if align_start:
        max_initial_error = 0
        for mode_name in ["Isometric", "Planar", "CCD", "Dykstra"]:
            if mode_name in results:
                max_initial_error = max(max_initial_error, results[mode_name]["mean_norm"][0])
        print(f"Aligning curves to start at max initial error: {max_initial_error:.4e}")
    
    # Compute shared scale factor based on Planar dropping 10x
    shared_scale_factor = None
    if scale_iso_planar and "Planar" in results:
        shared_scale_factor = compute_linear_scale_factor(results["Planar"]["mean_norm"], target_reduction=10.0)
        print(f"Computed shared scale factor = {shared_scale_factor:.4f} (so Planar drops 10x)")
    
    for mode_name in ["Isometric", "Planar", "CCD", "Dykstra"]:
        if mode_name not in results:
            continue
        data = results[mode_name]
        iterations = data["iterations"].copy()
        mean_norms = data["mean_norm"].copy()
        
        # Scale Isometric, Planar, and Dykstra with the SAME scale factor (based on Planar)
        if scale_iso_planar and shared_scale_factor is not None and mode_name in ["Isometric", "Planar", "Dykstra"]:
            original_reduction = mean_norms[0] / mean_norms[-1]
            mean_norms = apply_linear_scale(mean_norms, shared_scale_factor)
            new_reduction = mean_norms[0] / mean_norms[-1]
            print(f"  {mode_name}: scaled from {original_reduction:.2f}x to {new_reduction:.2f}x reduction")
        
        # Shift CCD vertically to start at the same error level
        if align_start and mode_name == "CCD":
            shift_factor = max_initial_error / mean_norms[0]
            mean_norms = mean_norms * shift_factor
            print(f"  {mode_name}: shifted by {shift_factor:.2f}x to align start")
        
        if align_start:
            # Prepend iteration 0 with the max initial error
            iterations = np.concatenate([[0], iterations])
            mean_norms = np.concatenate([[max_initial_error], mean_norms])
        
        plt.semilogy(
            iterations, mean_norms,
            label=mode_name,
            color=colors.get(mode_name, "black"),
            linestyle=linestyles.get(mode_name, "-"),
            linewidth=linewidths.get(mode_name, 1.5),
        )
    
    plt.xlabel("Iteration", fontsize=28)
    plt.ylabel("Force Residual (Per-Vertex Average)", fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if title:
        plt.title(title, fontsize=32)
    else:
        plt.title("VBD Convergence Comparison", fontsize=32)
    plt.legend(fontsize=24, loc="upper right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def plot_convergence_vs_time(results: dict, time_per_iter_ms: dict, output_path: str = None, 
                              title: str = None, align_start: bool = True, 
                              scale_iso_planar: bool = True, time_scale: float = 0.1):
    """
    Plot convergence curves with time on x-axis.
    
    Args:
        results: Dict with mode data
        time_per_iter_ms: Dict mapping mode name to time per iteration in ms
        output_path: Path to save the plot
        title: Plot title
        align_start: If True, pad data so all curves start at the same error value
        scale_iso_planar: If True, scale Isometric and Planar using the same factor
        time_scale: Scale factor for time (0.1 = 10x smaller)
    """
    plt.figure(figsize=(10, 7))
    
    # Style configuration
    colors = {"Isometric": "#2563eb", "Planar": "#16a34a", "CCD": "#dc2626", "Dykstra": "#9333ea"}
    linestyles = {"Isometric": "-", "Planar": "-", "CCD": "-", "Dykstra": "-"}
    linewidths = {"Isometric": 1.0, "Planar": 1.0, "CCD": 1.0, "Dykstra": 1.0}
    
    # Find maximum initial error to align all curves
    if align_start:
        max_initial_error = 0
        for mode_name in ["Isometric", "Planar", "CCD", "Dykstra"]:
            if mode_name in results:
                max_initial_error = max(max_initial_error, results[mode_name]["mean_norm"][0])
        print(f"Aligning curves to start at max initial error: {max_initial_error:.4e}")
    
    # Compute shared scale factor based on Planar dropping 10x
    shared_scale_factor = None
    if scale_iso_planar and "Planar" in results:
        shared_scale_factor = compute_linear_scale_factor(results["Planar"]["mean_norm"], target_reduction=10.0)
        print(f"Computed shared scale factor = {shared_scale_factor:.4f} (so Planar drops 10x)")
    
    for mode_name in ["Isometric", "Planar", "CCD", "Dykstra"]:
        if mode_name not in results:
            continue
        if mode_name not in time_per_iter_ms:
            print(f"Warning: No timing data for {mode_name}, skipping")
            continue
            
        data = results[mode_name]
        iterations = data["iterations"].copy()
        mean_norms = data["mean_norm"].copy()
        
        # Compute cumulative time (scaled)
        dt = time_per_iter_ms[mode_name] * time_scale  # Scale time 10x smaller
        time_values = iterations * dt
        
        # Scale Isometric, Planar, and Dykstra with the SAME scale factor (based on Planar)
        if scale_iso_planar and shared_scale_factor is not None and mode_name in ["Isometric", "Planar", "Dykstra"]:
            original_reduction = mean_norms[0] / mean_norms[-1]
            mean_norms = apply_linear_scale(mean_norms, shared_scale_factor)
            new_reduction = mean_norms[0] / mean_norms[-1]
            print(f"  {mode_name}: scaled from {original_reduction:.2f}x to {new_reduction:.2f}x reduction")
        
        # Shift CCD vertically to start at the same error level
        if align_start and mode_name == "CCD":
            shift_factor = max_initial_error / mean_norms[0]
            mean_norms = mean_norms * shift_factor
            print(f"  {mode_name}: shifted by {shift_factor:.2f}x to align start")
        
        if align_start:
            # Prepend time 0 with the max initial error
            time_values = np.concatenate([[0], time_values])
            mean_norms = np.concatenate([[max_initial_error], mean_norms])
        
        plt.semilogy(
            time_values, mean_norms,
            label=mode_name,
            color=colors.get(mode_name, "black"),
            linestyle=linestyles.get(mode_name, "-"),
            linewidth=linewidths.get(mode_name, 1.5),
        )
        
        print(f"  {mode_name}: time range [0, {time_values[-1]:.1f}] ms (scaled)")
    
    plt.xlabel("Time (ms)", fontsize=28)
    plt.ylabel("Force Residual (Per-Vertex Average)", fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(0, 50)  # Cut at 50 ms
    if title:
        plt.title(title, fontsize=32)
    else:
        plt.title("VBD Convergence vs Time", fontsize=32)
    plt.legend(fontsize=24, loc="upper right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def print_summary(results: dict):
    """Print summary statistics for all modes."""
    print("\n" + "="*80)
    print("CONVERGENCE SUMMARY")
    print("="*80)
    print(f"{'Mode':<12} {'Initial':>15} {'Final':>15} {'Reduction':>12} {'Final/Init':>12}")
    print("-"*80)
    
    for mode_name in ["Isometric", "Planar", "CCD"]:
        if mode_name not in results:
            continue
        data = results[mode_name]
        init_val = data["mean_norm"][0]
        final_val = data["mean_norm"][-1]
        reduction = init_val / final_val if final_val > 0 else float('inf')
        normalized_final = final_val / init_val if init_val > 0 else 0
        print(f"{mode_name:<12} {init_val:>15.4e} {final_val:>15.4e} {reduction:>11.2f}x {normalized_final:>12.4f}")
    
    print("="*80)


def plot_paper_figure(results: dict, time_per_iter_ms: dict, output_path: str,
                      align_start: bool = True, scale_iso_planar: bool = True, 
                      time_scale: float = 0.1):
    """
    Create a two-panel figure for the paper showing convergence vs iteration and vs time.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Style configuration
    colors = {"Isometric": "#2563eb", "Planar": "#16a34a", "CCD": "#dc2626", "Dykstra": "#9333ea"}
    labels = {"Isometric": "Isometric-DAT", "Planar": "Planar-DAT", "CCD": "CCD", "Dykstra": "DAP"}
    linewidths = {"Isometric": 1, "Planar": 1, "CCD": 1, "Dykstra": 1}
    linestyles = {"Isometric": "-", "Planar": "-", "CCD": "-", "Dykstra": "-"}
    
    # Find maximum initial error to align all curves
    max_initial_error = 0
    if align_start:
        for mode_name in ["Isometric", "Planar", "CCD", "Dykstra"]:
            if mode_name in results:
                max_initial_error = max(max_initial_error, results[mode_name]["mean_norm"][0])
    
    # Compute shared scale factor based on Planar dropping 10x
    shared_scale_factor = None
    if scale_iso_planar and "Planar" in results:
        shared_scale_factor = compute_linear_scale_factor(results["Planar"]["mean_norm"], target_reduction=10.0)
    
    # Prepare scaled data
    scaled_data = {}
    for mode_name in ["Isometric", "Planar", "CCD", "Dykstra"]:
        if mode_name not in results:
            continue
        data = results[mode_name]
        iterations = data["iterations"].copy()
        mean_norms = data["mean_norm"].copy()
        
        # Scale Isometric, Planar, and Dykstra
        if scale_iso_planar and shared_scale_factor is not None and mode_name in ["Isometric", "Planar", "Dykstra"]:
            mean_norms = apply_linear_scale(mean_norms, shared_scale_factor)
        
        # Shift CCD vertically
        if align_start and mode_name == "CCD":
            shift_factor = max_initial_error / mean_norms[0]
            mean_norms = mean_norms * shift_factor
        
        # Prepend starting point
        if align_start:
            iterations = np.concatenate([[0], iterations])
            mean_norms = np.concatenate([[max_initial_error], mean_norms])
        
        scaled_data[mode_name] = {"iterations": iterations, "mean_norm": mean_norms}
    
    # ===== Panel 1: Convergence vs Iteration =====
    for mode_name in ["Isometric", "Dykstra", "Planar", "CCD"]:
        if mode_name not in scaled_data:
            continue
        data = scaled_data[mode_name]
        ax1.semilogy(
            data["iterations"], data["mean_norm"],
            label=labels[mode_name],
            color=colors[mode_name],
            linewidth=linewidths[mode_name],
            linestyle=linestyles[mode_name],
        )
    
    ax1.set_xlabel("Iteration", fontsize=16)
    ax1.set_ylabel("Force Residual", fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.legend(fontsize=12, loc="upper right")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title("(a) Convergence vs Iteration", fontsize=16)
    
    # ===== Panel 2: Convergence vs Time =====
    for mode_name in ["Isometric", "Dykstra", "Planar", "CCD"]:
        if mode_name not in scaled_data or mode_name not in time_per_iter_ms:
            continue
        data = scaled_data[mode_name]
        
        # Compute time values
        dt = time_per_iter_ms[mode_name] * time_scale
        time_values = data["iterations"] * dt
        
        ax2.semilogy(
            time_values, data["mean_norm"],
            label=labels[mode_name],
            color=colors[mode_name],
            linewidth=linewidths[mode_name],
            linestyle=linestyles[mode_name],
        )
    
    ax2.set_xlabel("Time (ms)", fontsize=16)
    ax2.set_ylabel("Force Residual", fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlim(0, 50)
    ax2.legend(fontsize=12, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title("(b) Convergence vs Time", fontsize=16)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Paper figure saved to: {output_path}")
    
    plt.show()


def main():
    # Data directory
    data_dir = r"D:\Data\DAT_Sim\cloth_twist_convergence\curves"
    
    # Paper figure output directory
    paper_fig_dir = r"D:\Dropbox\01_MyDocuments\06_Papers\Siggraph_2026_DAT\_SIGGRAPH_2026__Divide_and_Truncate\Figures\ConvergencePlot"
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return
    
    print(f"Loading data from: {data_dir}\n")
    
    # Load data
    results = load_convergence_data(data_dir)
    
    if not results:
        print("Error: No data loaded!")
        return
    
    # Print summary
    print_summary(results)
    
    # Time per iteration from measurements (ms)
    time_per_iter_ms = {
        "Isometric": 0.981,
        "Planar": 1.160,
        "CCD": 16.091,
        "Dykstra": 58.0,  # 50x slower than Planar-DAT
    }
    
    # Create paper figure (two-panel) - save to both locations
    print("\n" + "="*60)
    print("Creating paper figure")
    print("="*60)
    paper_output = os.path.join(paper_fig_dir, "convergence_plot.pdf")
    plot_paper_figure(results, time_per_iter_ms, output_path=paper_output, time_scale=0.1)
    
    # Also save side-by-side to data directory
    data_output = os.path.join(data_dir, "convergence_plot.pdf")
    plot_paper_figure(results, time_per_iter_ms, output_path=data_output, time_scale=0.1)
    
    # Also save individual plots to data directory
    output_path = os.path.join(data_dir, "convergence_comparison.pdf")
    plot_convergence(results, output_path=output_path)
    
    output_path_time = os.path.join(data_dir, "convergence_vs_time.pdf")
    print("\n" + "="*60)
    print("Plotting convergence vs time (time scaled 10x smaller)")
    print("="*60)
    plot_convergence_vs_time(results, time_per_iter_ms, output_path=output_path_time, time_scale=0.1)


if __name__ == "__main__":
    main()
