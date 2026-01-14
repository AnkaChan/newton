"""
Random Mesh Truncation Test

Generates random triangle meshes with 3000 faces and evaluates two truncation schemes:
1. Isometric Truncation - conservative bound-based truncation (TruncatorOffset)
2. Planar Truncation - planar division-based truncation (TruncatorPlanar)

All parameters are defined in the config dict for reproducibility.
"""

import numpy as np
import warp as wp
import json
import time
import os

from newton import ModelBuilder
from newton._src.solvers.vbd.tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshContinuousCollisionDetector,
)
from newton._src.solvers.vbd.solver_vbd import SolverVBD

from M03_Evaluation import TruncationAnalyzer, check_intersection
from M02_TruncationBenchMark import (
    RandomMeshGenerator,
    TruncatorPlanar,
    Truncator,
)
from newton._src.solvers.vbd.solver_vbd import compute_particle_conservative_bound


def save_ply(filepath: str, vertices: np.ndarray, triangles: np.ndarray):
    """Save mesh as PLY file."""
    num_vertices = len(vertices)
    num_faces = len(triangles)
    
    with open(filepath, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Vertices
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Faces
        for t in triangles:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def save_sample_data(
    output_dir: str,
    mesh_idx: int,
    disp_idx: int,
    vertices: np.ndarray,
    triangles: np.ndarray,
    original_displacement: np.ndarray,
    isometric_displacement: np.ndarray,
    planar_displacement: np.ndarray,
    evaluation: dict,
    ccd_displacement: np.ndarray = None,
):
    """
    Save sample data to disk.
    
    - Mesh: .ply (standard 3D format)
    - Displacements: .npz (efficient binary format)
    - Evaluation: .json (human readable)
    """
    # Create mesh directory if this is a new mesh
    mesh_dir = os.path.join(output_dir, f"mesh_{mesh_idx:04d}")
    os.makedirs(mesh_dir, exist_ok=True)
    
    # Save mesh as PLY only once per mesh (when disp_idx == 0)
    mesh_file = os.path.join(mesh_dir, "mesh.ply")
    if not os.path.exists(mesh_file):
        save_ply(mesh_file, vertices, triangles)
    
    # Save displacements for this sample
    disp_file = os.path.join(mesh_dir, f"disp_{disp_idx:04d}.npz")
    save_dict = {
        "original": original_displacement,
        "isometric": isometric_displacement,
        "planar": planar_displacement,
    }
    if ccd_displacement is not None:
        save_dict["ccd"] = ccd_displacement
    np.savez_compressed(disp_file, **save_dict)
    
    # Save evaluation as JSON
    eval_file = os.path.join(mesh_dir, f"eval_{disp_idx:04d}.json")
    with open(eval_file, "w") as f:
        json.dump(evaluation, f, indent=2)


class TruncatorIsometric(Truncator):
    """Isometric truncation using conservative bounds - fixed version."""
    
    @wp.kernel
    def _apply_isometric_truncation(
        positions: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        particle_conservative_bounds: wp.array(dtype=float),
        displacement_out: wp.array(dtype=wp.vec3),
    ):
        particle_idx = wp.tid()
        
        displacement = displacement_in[particle_idx]
        conservative_bound = particle_conservative_bounds[particle_idx]
        
        displacement_norm = wp.length(displacement)
        if displacement_norm > conservative_bound and conservative_bound > 1e-6:
            displacement = displacement * (conservative_bound / displacement_norm)
        
        displacement_out[particle_idx] = displacement
    
    def truncate(self, positions, displacements):
        self.positions = wp.array(positions, dtype=wp.vec3)
        self.displacements_in = wp.array(displacements, dtype=wp.vec3)
        self.displacements_out = wp.zeros_like(self.displacements_in)
        self.particle_conservative_bounds = wp.zeros(displacements.shape[0], dtype=float)
        
        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.bound_relaxation,
                self.collision_query_radius,
                self.adjacency,
                self.collision_detector.collision_info,
            ],
            outputs=[self.particle_conservative_bounds],
            dim=self.model.particle_count,
        )
        
        wp.launch(
            kernel=self._apply_isometric_truncation,
            inputs=[
                self.positions,
                self.displacements_in,
                self.particle_conservative_bounds,
            ],
            outputs=[self.displacements_out],
            dim=self.model.particle_count,
        )
        
        return self.displacements_out.numpy()


class TruncatorCCD(Truncator):
    """
    CCD (Continuous Collision Detection) truncation.
    
    Uses GLOBAL truncation: finds the earliest collision time across ALL 
    vertex-triangle and edge-edge pairs, then scales ALL displacements by
    this single factor.
    
    This is the most conservative approach - if ANY primitive collides at time t,
    ALL vertices are scaled to stop at time t.
    """
    
    def __init__(self, collision_query_radius: float, bound_relaxation: float, safety_margin: float = 0.99):
        super().__init__(collision_query_radius, bound_relaxation)
        self.safety_margin = safety_margin
        self.ccd_detector = None
    
    def analyze_shape(self, collision_detector: TriMeshCollisionDetector, adjacency):
        """Override to also create the CCD detector."""
        super().analyze_shape(collision_detector, adjacency)
        # CCD detector will be created in truncate() with actual positions/displacements
    
    def truncate(self, positions, displacements):
        """
        Truncate displacements using global CCD.
        
        1. Create CCD detector with current positions and displacements
        2. Run V-T and E-E CCD detection
        3. Find GLOBAL minimum collision time
        4. Scale ALL displacements by this time
        """
        # Create warp arrays
        vertex_positions = wp.array(positions.astype(np.float32), dtype=wp.vec3)
        vertex_displacements = wp.array(displacements.astype(np.float32), dtype=wp.vec3)
        
        # Create CCD detector
        self.ccd_detector = TriMeshContinuousCollisionDetector(
            self.collision_detector,
            vertex_positions,
            vertex_displacements,
        )
        
        # Run V-T CCD
        self.ccd_detector.detect_vertex_triangle_ccd()
        vt_times = self.ccd_detector.vertex_collision_times.numpy()
        
        # Run E-E CCD
        self.ccd_detector.detect_edge_edge_ccd()
        ee_times = self.ccd_detector.edge_collision_times.numpy()
        
        # Find GLOBAL minimum collision time
        min_vt = vt_times.min() if len(vt_times) > 0 else 1.0
        min_ee = ee_times.min() if len(ee_times) > 0 else 1.0
        global_min_t = min(min_vt, min_ee)
        
        # Apply safety margin
        global_t = global_min_t * self.safety_margin
        
        # Clamp to [0, 1]
        global_t = max(0.0, min(1.0, global_t))
        
        # Scale ALL displacements by global_t
        truncated_displacements = displacements * global_t
        
        # Store for debugging
        self.last_global_t = global_t
        self.last_min_vt = min_vt
        self.last_min_ee = min_ee
        
        return truncated_displacements


# ==============================================================================
# Config Dict - Single Source of Truth
# ==============================================================================

config = {
    # Identity
    "name": "random_mesh_truncation_test",
    
    # Random Mesh Generation
    "num_triangles": 3000,           # Number of triangles (faces) in the mesh
    "world_size": 10.0,              # Bounding box size for mesh generation
    "target_edge": 0.25,             # Target edge length for triangles
    "edge_tol_low": 0.1,             # Lower tolerance for edge length acceptance
    "edge_tol_high": 3.0,            # Upper tolerance for edge length acceptance
    "k_neighbors": 16,               # K nearest neighbors for mesh generation
    "displacement_scale": 0.1,       # Scale of random displacements
    "shrink_ratio": 0.95,            # Shrink triangles to avoid self-intersection
    "displacements_per_mesh": 10,   # Number of displacement samples per mesh (100 per mesh × 10 meshes)
    "mesh_seed": 42,                 # Random seed for reproducibility
    "fail_limit": 100_000,           # Max failed attempts before giving up on mesh
    
    # Dataset
    "data_set_size": 100,           # Total samples: 10 meshes × 100 displacements = 1000
    
    # Solver/Collision settings (used by SolverVBD)
    "self_contact_radius": 0.2,                           # Collision detection radius
    "self_contact_margin": 0.4,                           # Collision detection margin (should be > radius)
    "penetration_free_conservative_bound_relaxation": 0.42,  # Relaxation factor for truncation
    "vertex_collision_buffer_pre_alloc": 64,              # Collision buffer pre-allocation
    "edge_collision_buffer_pre_alloc": 128,               # Edge collision buffer
    "topological_contact_filter_threshold": 2,            # Topological filter threshold
    
    # CCD-specific settings
    "ccd_safety_margin": 0.99,                            # Safety margin for CCD (slightly less than t to avoid edge cases)
    
    # Evaluation
    "evaluate_initial_state": True,  # Check if initial mesh is intersection-free
    "plot_intersections": False,     # Visualize intersections with polyscope
    
    # Output
    "print_per_sample": False,       # Print results for each sample (disabled for large runs)
    "output_json": False,            # Output results as JSON (disabled for large runs)
    
    # Data saving
    "save_data": True,               # Save mesh, displacements, and evaluations
    "output_dir": "",                # Will be set dynamically below
}

# Build output directory name with displacement_scale
config["output_dir"] = rf"D:\Data\DAT_Sim\Truncation_Scheme_Evaluation\Random_disp{config['displacement_scale']}"


def run_truncation_test(vs, ts, displacements, truncator_type: str, cfg: dict):
    """
    Run truncation using the Truncator classes from M02_TruncationBenchMark.py.
    
    Args:
        vs: Vertex positions (N, 3)
        ts: Triangle indices (F, 3)
        displacements: Random displacements (N, 3)
        truncator_type: "isometric" or "planar"
        cfg: Config dict
        
    Returns:
        displacements_truncated: Truncated displacements (N, 3)
        truncation_time: Time spent in truncation
        collision_detector: TriMeshCollisionDetector for intersection checking
    """
    # Build newton model
    builder = ModelBuilder()
    vertices = [wp.vec3(vs[i, :]) for i in range(vs.shape[0])]
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vertices=vertices,
        indices=ts.reshape(-1),
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.02,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=2.0e-6,
        edge_ke=10,
    )
    builder.color()
    model = builder.finalize()
    
    # Create VBD solver for adjacency info
    vbd_integrator = SolverVBD(model)
    
    # Create collision detector
    collision_detector = TriMeshCollisionDetector(
        model,
        vertex_collision_buffer_pre_alloc=cfg["vertex_collision_buffer_pre_alloc"],
        triangle_collision_buffer_pre_alloc=cfg["vertex_collision_buffer_pre_alloc"],
        edge_collision_buffer_pre_alloc=cfg["edge_collision_buffer_pre_alloc"],
        triangle_triangle_collision_buffer_pre_alloc=128,
        record_triangle_contacting_vertices=True,
    )
    
    # Create truncator based on type
    if truncator_type == "isometric":
        truncator = TruncatorIsometric(
            collision_query_radius=cfg["self_contact_margin"],
            bound_relaxation=cfg["penetration_free_conservative_bound_relaxation"],
        )
    elif truncator_type == "planar":
        truncator = TruncatorPlanar(
            collision_query_radius=cfg["self_contact_margin"],
            bound_relaxation=cfg["penetration_free_conservative_bound_relaxation"],
        )
    elif truncator_type == "ccd":
        truncator = TruncatorCCD(
            collision_query_radius=cfg["self_contact_margin"],
            bound_relaxation=cfg["penetration_free_conservative_bound_relaxation"],
            safety_margin=cfg.get("ccd_safety_margin", 0.99),
        )
    else:
        raise ValueError(f"Unknown truncator type: {truncator_type}")
    
    # Analyze shape (sets up collision detection)
    truncator.analyze_shape(collision_detector, vbd_integrator.adjacency)
    
    # Time the truncation
    wp.synchronize()
    start_time = time.perf_counter()
    
    displacements_truncated = truncator.truncate(vs, displacements)
    
    wp.synchronize()
    end_time = time.perf_counter()
    truncation_time = end_time - start_time
    
    return displacements_truncated, truncation_time, collision_detector


def run_vertex_comparison(mesh_gen, cfg: dict):
    """
    Run all truncation methods on the same meshes and compare vertex-by-vertex.
    
    Compares: Isometric, Planar, and CCD (global minimum t).
    Returns per-vertex improvement ratios.
    """
    # Planar vs Isometric improvement
    all_planar_vs_iso_ratios = []
    all_planar_norms = []
    all_isometric_norms = []
    
    # Planar vs CCD (how much Planar improves over CCD)
    all_ccd_norms = []
    all_planar_vs_ccd_ratios = []
    all_ccd_global_t = []
    
    # For full evaluation reports
    all_isometric_reports = []
    all_planar_reports = []
    all_ccd_reports = []
    
    # Reset the mesh generator
    mesh_gen.data_counter = 0
    mesh_gen.rng = np.random.default_rng(cfg["mesh_seed"])
    
    # Setup output directory if saving
    save_data = cfg.get("save_data", False)
    output_dir = cfg.get("output_dir", "")
    if save_data and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save config
        config_file = os.path.join(output_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"\nSaving data to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("VERTEX-BY-VERTEX COMPARISON: Isometric vs Planar vs CCD")
    print("=" * 80)
    
    evaluator = TruncationAnalyzer()
    displacements_per_mesh = cfg.get("displacements_per_mesh", 1)
    
    for sample_idx in range(cfg["data_set_size"]):
        vs, ts, displacements = mesh_gen.get_mesh()
        mesh_gen.data_counter += 1
        
        # Calculate mesh and displacement indices
        mesh_idx = sample_idx // displacements_per_mesh
        disp_idx = sample_idx % displacements_per_mesh
        
        try:
            # Run all three truncation methods on the SAME mesh and displacements
            disp_iso, time_iso, _ = run_truncation_test(vs, ts, displacements, "isometric", cfg)
            disp_planar, time_planar, _ = run_truncation_test(vs, ts, displacements, "planar", cfg)
            disp_ccd, time_ccd, _ = run_truncation_test(vs, ts, displacements, "ccd", cfg)
            
            # Compute norms per vertex
            norms_iso = np.linalg.norm(disp_iso, axis=1)
            norms_planar = np.linalg.norm(disp_planar, axis=1)
            norms_ccd = np.linalg.norm(disp_ccd, axis=1)
            norms_original = np.linalg.norm(displacements, axis=1)
            
            eps = 1e-10
            
            # --- Planar vs Isometric ---
            valid_mask_iso = norms_iso > eps
            planar_vs_iso = np.zeros_like(norms_iso)
            if valid_mask_iso.sum() > 0:
                planar_vs_iso[valid_mask_iso] = (norms_planar[valid_mask_iso] - norms_iso[valid_mask_iso]) / norms_iso[valid_mask_iso]
                all_planar_vs_iso_ratios.extend(planar_vs_iso[valid_mask_iso].tolist())
                all_planar_norms.extend(norms_planar[valid_mask_iso].tolist())
                all_isometric_norms.extend(norms_iso[valid_mask_iso].tolist())
            
            # --- Planar vs CCD (how much Planar improves over CCD) ---
            valid_mask_ccd = norms_ccd > eps
            planar_vs_ccd = np.zeros_like(norms_ccd)
            if valid_mask_ccd.sum() > 0:
                planar_vs_ccd[valid_mask_ccd] = (norms_planar[valid_mask_ccd] - norms_ccd[valid_mask_ccd]) / norms_ccd[valid_mask_ccd]
                all_planar_vs_ccd_ratios.extend(planar_vs_ccd[valid_mask_ccd].tolist())
            
            # Track CCD norms
            all_ccd_norms.extend(norms_ccd.tolist())
            
            # Track cases where iso zeroed but planar didn't
            iso_zero_planar_not = (norms_iso <= eps) & (norms_planar > eps)
            
            mean_planar_vs_iso = planar_vs_iso[valid_mask_iso].mean() * 100 if valid_mask_iso.sum() > 0 else 0.0
            mean_planar_vs_ccd = planar_vs_ccd[valid_mask_ccd].mean() * 100 if valid_mask_ccd.sum() > 0 else 0.0
            
            # Build evaluation dict
            evaluation = {
                "sample_idx": sample_idx,
                "mesh_idx": mesh_idx,
                "disp_idx": disp_idx,
                "num_vertices": len(vs),
                "num_triangles": len(ts),
                "isometric": {
                    "time_ms": time_iso * 1000,
                    "mean_norm": float(norms_iso.mean()),
                    "preservation_ratio": float(norms_iso.mean() / norms_original.mean()) if norms_original.mean() > eps else 0.0,
                },
                "planar": {
                    "time_ms": time_planar * 1000,
                    "mean_norm": float(norms_planar.mean()),
                    "preservation_ratio": float(norms_planar.mean() / norms_original.mean()) if norms_original.mean() > eps else 0.0,
                },
                "ccd": {
                    "time_ms": time_ccd * 1000,
                    "mean_norm": float(norms_ccd.mean()),
                    "preservation_ratio": float(norms_ccd.mean() / norms_original.mean()) if norms_original.mean() > eps else 0.0,
                },
                "comparison": {
                    "planar_vs_iso_pct": float(mean_planar_vs_iso),
                    "planar_vs_ccd_pct": float(mean_planar_vs_ccd),
                    "fraction_planar_better_than_iso": float((planar_vs_iso[valid_mask_iso] > 0).mean()) if valid_mask_iso.sum() > 0 else 0.0,
                    "fraction_planar_better_than_ccd": float((planar_vs_ccd[valid_mask_ccd] > 0).mean()) if valid_mask_ccd.sum() > 0 else 0.0,
                    "iso_zeroed_planar_kept": int(iso_zero_planar_not.sum()),
                },
            }
            
            # Run full TruncationAnalyzer evaluation for all methods
            iso_report = evaluator.analyze(vs, ts, displacements, disp_iso, print_results=False)
            planar_report = evaluator.analyze(vs, ts, displacements, disp_planar, print_results=False)
            ccd_report = evaluator.analyze(vs, ts, displacements, disp_ccd, print_results=False)
            all_isometric_reports.append(iso_report)
            all_planar_reports.append(planar_report)
            all_ccd_reports.append(ccd_report)
            
            # Save data if enabled
            if save_data and output_dir:
                save_sample_data(
                    output_dir=output_dir,
                    mesh_idx=mesh_idx,
                    disp_idx=disp_idx,
                    vertices=vs.astype(np.float32),
                    triangles=ts.astype(np.int32),
                    original_displacement=displacements.astype(np.float32),
                    isometric_displacement=disp_iso.astype(np.float32),
                    planar_displacement=disp_planar.astype(np.float32),
                    evaluation=evaluation,
                    ccd_displacement=disp_ccd.astype(np.float32),
                )
            
            # Print progress
            if sample_idx % 10 == 0 or cfg.get("print_per_sample", False):
                print(f"[{sample_idx+1}/{cfg['data_set_size']}] Mesh {mesh_idx}, Disp {disp_idx}: "
                      f"iso={norms_iso.mean():.4f}, planar={norms_planar.mean():.4f}, ccd={norms_ccd.mean():.4f}, "
                      f"planar_vs_iso={mean_planar_vs_iso:+.1f}%, planar_vs_ccd={mean_planar_vs_ccd:+.1f}%")
            
        except Exception as e:
            print(f"Sample {sample_idx}: Error - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate statistics
    if all_planar_vs_iso_ratios:
        planar_vs_iso_arr = np.array(all_planar_vs_iso_ratios)
        planar_vs_ccd_arr = np.array(all_planar_vs_ccd_ratios) if all_planar_vs_ccd_ratios else np.array([])
        
        print("\n" + "-" * 80)
        print("VERTEX-BY-VERTEX IMPROVEMENT STATISTICS")
        print("-" * 80)
        print(f"  Total vertices compared: {len(planar_vs_iso_arr)}")
        
        # --- Planar vs Isometric ---
        print(f"\n  === Planar vs Isometric ===")
        print(f"  Improvement Ratio (planar_norm - iso_norm) / iso_norm:")
        print(f"    Mean:   {planar_vs_iso_arr.mean()*100:+.2f}%")
        print(f"    Median: {np.median(planar_vs_iso_arr)*100:+.2f}%")
        print(f"    Max:    {planar_vs_iso_arr.max()*100:+.2f}%")
        print(f"  Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"    {p}th: {np.percentile(planar_vs_iso_arr, p)*100:+.2f}%")
        print(f"  Fraction where Planar > Isometric: {(planar_vs_iso_arr > 0).mean()*100:.2f}%")
        
        # --- Planar vs CCD (how much Planar improves over CCD) ---
        if len(planar_vs_ccd_arr) > 0:
            print(f"\n  === Planar vs CCD (Planar's improvement over CCD) ===")
            print(f"  Improvement Ratio (planar_norm - ccd_norm) / ccd_norm:")
            print(f"    Mean:   {planar_vs_ccd_arr.mean()*100:+.2f}%")
            print(f"    Median: {np.median(planar_vs_ccd_arr)*100:+.2f}%")
            print(f"    Max:    {planar_vs_ccd_arr.max()*100:+.2f}%")
            print(f"  Percentiles:")
            for p in [25, 50, 75, 90, 95, 99]:
                print(f"    {p}th: {np.percentile(planar_vs_ccd_arr, p)*100:+.2f}%")
            print(f"  Fraction where Planar > CCD: {(planar_vs_ccd_arr > 0).mean()*100:.2f}%")
        
        print("-" * 80)
        
        # Aggregate full evaluation reports
        def aggregate_reports(reports):
            """Aggregate a list of TruncationAnalyzer reports into mean values."""
            if not reports:
                return {}
            keys = reports[0].keys()
            aggregated = {}
            for key in keys:
                values = [r[key] for r in reports if key in r and isinstance(r[key], (int, float))]
                if values:
                    aggregated[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
            return aggregated
        
        iso_aggregated = aggregate_reports(all_isometric_reports)
        planar_aggregated = aggregate_reports(all_planar_reports)
        ccd_aggregated = aggregate_reports(all_ccd_reports)
        
        # Print full evaluation summary
        print("\n" + "=" * 80)
        print("FULL EVALUATION SUMMARY (Averaged over all samples)")
        print("=" * 80)
        
        print("\n--- ISOMETRIC TRUNCATION ---")
        for key, stats in iso_aggregated.items():
            print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        print("\n--- PLANAR TRUNCATION ---")
        for key, stats in planar_aggregated.items():
            print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        print("\n--- CCD TRUNCATION (Global min t) ---")
        for key, stats in ccd_aggregated.items():
            print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        # Print side-by-side comparison table for key metrics
        print("\n" + "=" * 80)
        print("COMPARISON TABLE (Key Metrics)")
        print("=" * 80)
        key_metrics = ["preservation_ratio", "fraction_unchanged", "fraction_zeroed"]
        print(f"\n{'Metric':<30} {'Isometric':>12} {'Planar':>12} {'CCD':>12}")
        print("-" * 70)
        for metric in key_metrics:
            iso_val = iso_aggregated.get(metric, {}).get("mean", 0)
            planar_val = planar_aggregated.get(metric, {}).get("mean", 0)
            ccd_val = ccd_aggregated.get(metric, {}).get("mean", 0)
            print(f"{metric:<30} {iso_val:>12.4f} {planar_val:>12.4f} {ccd_val:>12.4f}")
        
        print("=" * 80)
        
        summary = {
            "vertex_comparison": {
                "planar_vs_iso": {
                    "mean": float(planar_vs_iso_arr.mean()),
                    "median": float(np.median(planar_vs_iso_arr)),
                    "max": float(planar_vs_iso_arr.max()),
                    "percentiles": {str(p): float(np.percentile(planar_vs_iso_arr, p)) for p in [25, 50, 75, 90, 95, 99]},
                    "fraction_planar_better": float((planar_vs_iso_arr > 0).mean()),
                },
                "planar_vs_ccd": {
                    "mean": float(planar_vs_ccd_arr.mean()) if len(planar_vs_ccd_arr) > 0 else None,
                    "median": float(np.median(planar_vs_ccd_arr)) if len(planar_vs_ccd_arr) > 0 else None,
                    "max": float(planar_vs_ccd_arr.max()) if len(planar_vs_ccd_arr) > 0 else None,
                    "percentiles": {str(p): float(np.percentile(planar_vs_ccd_arr, p)) for p in [25, 50, 75, 90, 95, 99]} if len(planar_vs_ccd_arr) > 0 else {},
                    "fraction_planar_better": float((planar_vs_ccd_arr > 0).mean()) if len(planar_vs_ccd_arr) > 0 else None,
                },
                "total_vertices": len(planar_vs_iso_arr),
            },
            "full_evaluation": {
                "isometric": iso_aggregated,
                "planar": planar_aggregated,
                "ccd": ccd_aggregated,
            },
            "total_samples": cfg["data_set_size"],
            "num_meshes": cfg["data_set_size"] // cfg.get("displacements_per_mesh", 1),
            "displacements_per_mesh": cfg.get("displacements_per_mesh", 1),
        }
        
        # Save summary if saving is enabled
        if save_data and output_dir:
            summary_file = os.path.join(output_dir, "summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {summary_file}")
        
        return summary
    
    return None


def run_single_truncation_test(mesh_gen, truncator_type: str, cfg: dict, truncator_name: str):
    """
    Run truncation test with a single truncator type.
    
    Args:
        mesh_gen: RandomMeshGenerator instance
        truncator_type: "isometric" or "planar"
        cfg: Config dict
        truncator_name: Name for logging
        
    Returns:
        reports: List of per-sample evaluation reports
        timing: Total time spent in truncation
    """
    evaluator = TruncationAnalyzer()
    reports = []
    total_time = 0.0
    
    # Reset the mesh generator for fair comparison
    mesh_gen.data_counter = 0
    mesh_gen.rng = np.random.default_rng(cfg["mesh_seed"])
    
    for sample_idx in range(cfg["data_set_size"]):
        vs, ts, displacements = mesh_gen.get_mesh()
        mesh_gen.data_counter += 1
        
        # Run truncation
        try:
            displacements_truncated, truncation_time, collision_detector = run_truncation_test(
                vs, ts, displacements, truncator_type, cfg
            )
            total_time += truncation_time
        except Exception as e:
            print(f"[{truncator_name}] Sample {sample_idx}: Error during truncation: {e}")
            continue
        
        # Check initial state
        if cfg["evaluate_initial_state"]:
            is_intersection_free = check_intersection(
                collision_detector, vs, None, plot=cfg["plot_intersections"]
            )
            if is_intersection_free:
                if cfg["print_per_sample"]:
                    print(f"[{truncator_name}] Sample {sample_idx}: Initial mesh is intersection-free.")
            else:
                print(f"[{truncator_name}] Sample {sample_idx}: Initial mesh has intersections! Skipping.")
                continue
        
        # Check for intersections after truncation
        is_safe = check_intersection(
            collision_detector, vs, displacements_truncated, plot=cfg["plot_intersections"]
        )
        if not is_safe:
            print(f"[{truncator_name}] Sample {sample_idx}: ERROR! Intersection after truncation!")
        
        # Evaluate truncation quality
        report = evaluator.analyze(
            vs, ts, displacements, displacements_truncated, print_results=False
        )
        report["sample_idx"] = sample_idx
        report["intersection_free_after"] = is_safe
        reports.append(report)
        
        if cfg["print_per_sample"]:
            print(f"[{truncator_name}] Sample {sample_idx}: "
                  f"preservation_ratio={report['preservation_ratio_stats']['mean']:.4f}, "
                  f"fraction_unchanged={report['fraction_unchanged']:.4f}, "
                  f"time={truncation_time*1000:.2f}ms, "
                  f"safe={is_safe}")
    
    return reports, total_time, evaluator


def compare_truncation_schemes(cfg: dict):
    """
    Compare three truncation schemes: Isometric, Planar, and CCD (global min t).
    """
    print("=" * 80)
    print(f"Random Mesh Truncation Test: {cfg['name']}")
    print(f"  Triangles: {cfg['num_triangles']}, Samples: {cfg['data_set_size']}")
    print(f"  Schemes: Isometric, Planar (DAT), CCD (Global min t)")
    print("=" * 80)
    
    # Create mesh generator (shared between all truncation modes for fair comparison)
    mesh_gen = RandomMeshGenerator(
        data_set_size=cfg["data_set_size"],
        num_triangles=cfg["num_triangles"],
        world_size=cfg["world_size"],
        target_edge=cfg["target_edge"],
        edge_tol_low=cfg["edge_tol_low"],
        edge_tol_high=cfg["edge_tol_high"],
        k_neighbors=cfg["k_neighbors"],
        displacement_scale=cfg["displacement_scale"],
        shrink_ratio=cfg["shrink_ratio"],
        displacements_per_mesh=cfg["displacements_per_mesh"],
        seed=cfg["mesh_seed"],
        fail_limit=cfg["fail_limit"],
    )
    
    results = {}
    
    # Test Isometric Truncation (TruncatorOffset)
    print("\n" + "-" * 40)
    print("Testing: Isometric Truncation")
    print("  Conservative bound-based truncation")
    print("-" * 40)
    reports_isometric, time_isometric, evaluator = run_single_truncation_test(
        mesh_gen, truncator_type="isometric", cfg=cfg, truncator_name="Isometric"
    )
    summary_isometric = evaluator.combine_reports(reports_isometric) if reports_isometric else {}
    summary_isometric["total_time_sec"] = time_isometric
    summary_isometric["avg_time_per_sample_ms"] = (time_isometric / len(reports_isometric) * 1000) if reports_isometric else 0
    results["isometric"] = {
        "summary": summary_isometric,
        "per_sample": reports_isometric,
    }
    
    # Test Planar Truncation (TruncatorPlanar)
    print("\n" + "-" * 40)
    print("Testing: Planar Truncation (DAT)")
    print("  Planar division-based truncation")
    print("-" * 40)
    reports_planar, time_planar, evaluator = run_single_truncation_test(
        mesh_gen, truncator_type="planar", cfg=cfg, truncator_name="Planar"
    )
    summary_planar = evaluator.combine_reports(reports_planar) if reports_planar else {}
    summary_planar["total_time_sec"] = time_planar
    summary_planar["avg_time_per_sample_ms"] = (time_planar / len(reports_planar) * 1000) if reports_planar else 0
    results["planar"] = {
        "summary": summary_planar,
        "per_sample": reports_planar,
    }
    
    # Test CCD Truncation (Global min t)
    print("\n" + "-" * 40)
    print("Testing: CCD Truncation (Global min t)")
    print("  Continuous collision detection - scales ALL vertices by earliest collision time")
    print("-" * 40)
    reports_ccd, time_ccd, evaluator = run_single_truncation_test(
        mesh_gen, truncator_type="ccd", cfg=cfg, truncator_name="CCD"
    )
    summary_ccd = evaluator.combine_reports(reports_ccd) if reports_ccd else {}
    summary_ccd["total_time_sec"] = time_ccd
    summary_ccd["avg_time_per_sample_ms"] = (time_ccd / len(reports_ccd) * 1000) if reports_ccd else 0
    results["ccd"] = {
        "summary": summary_ccd,
        "per_sample": reports_ccd,
    }
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (3 Schemes)")
    print("=" * 80)
    
    # Metrics with "higher_is_better" flag:
    # True = higher value is better (e.g., preservation ratio)
    # False = lower value is better (e.g., fraction zeroed, angle deviation)
    metrics_to_compare = [
        ("Preservation Ratio (mean)", "preservation_ratio_stats", "mean", True),
        ("Fraction Unchanged", "fraction_unchanged", None, True),
        ("Fraction Zeroed", "fraction_zeroed", None, False),
        ("Global Preservation (sum)", "global_preservation_sum", None, True),
        ("Direction Angle (mean deg)", "direction_angle_deg_stats", "mean", False),
        ("Avg Time (ms)", "avg_time_per_sample_ms", None, False),
    ]
    
    print(f"\n{'Metric':<30} {'Isometric':>12} {'Planar':>12} {'CCD':>12}")
    print("-" * 70)
    
    for metric_name, key, subkey, higher_is_better in metrics_to_compare:
        if subkey:
            val_iso = summary_isometric.get(key, {}).get(subkey, None)
            val_planar = summary_planar.get(key, {}).get(subkey, None)
            val_ccd = summary_ccd.get(key, {}).get(subkey, None)
        else:
            val_iso = summary_isometric.get(key, None)
            val_planar = summary_planar.get(key, None)
            val_ccd = summary_ccd.get(key, None)
        
        # Format values
        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v) if v is not None else "N/A"
        
        print(f"{metric_name:<30} {fmt(val_iso):>12} {fmt(val_planar):>12} {fmt(val_ccd):>12}")
    
    # Additional interpretation
    print("\n" + "-" * 70)
    print("Notes:")
    print("  - Isometric: Per-vertex conservative bound (safe but restrictive)")
    print("  - Planar:    Per-vertex planar division (Divide & Truncate)")
    print("  - CCD:       Global min collision time (scales ALL vertices uniformly)")
    print("  - Higher preservation = more motion preserved")
    print("  - CCD is the most conservative (global scale)")
    
    # Run vertex-by-vertex comparison (includes CCD now)
    vertex_comparison = run_vertex_comparison(mesh_gen, cfg)
    results["vertex_comparison"] = vertex_comparison
    
    # Output full results as JSON
    if cfg["output_json"]:
        full_output = {
            "config": cfg,
            "results": {
                "isometric": {"summary": summary_isometric},
                "planar": {"summary": summary_planar},
                "ccd": {"summary": summary_ccd},
                "vertex_comparison": vertex_comparison,
            }
        }
        print("\n" + "-" * 40)
        print("Full JSON Output:")
        print("-" * 40)
        print(json.dumps(full_output, indent=2, default=str))
    
    return results


if __name__ == "__main__":
    wp.init()
    compare_truncation_schemes(config)
