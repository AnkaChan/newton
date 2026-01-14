import numpy as np
import warp as wp

from newton import ModelBuilder
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton._src.solvers.vbd.solver_vbd import *

import numpy as np
import warp as wp

from newton import ModelBuilder
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton._src.solvers.vbd.solver_vbd import *
import polyscope as ps

def check_intersection(collision_detector: TriMeshCollisionDetector, vertices, displacement, plot=True):
    if isinstance(vertices, wp.array):
        vertices = vertices.numpy()

    if displacement is not None:
        vertices_new = vertices + displacement
    else:
        vertices_new = vertices

    collision_detector.refit(wp.array(vertices_new, dtype=wp.vec3))

    ts = collision_detector.model.tri_indices.numpy()

    collision_detector.triangle_triangle_intersection_detection()
    num_intersections = collision_detector.triangle_intersecting_triangles_count.numpy()
    if num_intersections.sum():
        # Print the colliding triangle pairs and plot only colliding triangles with polyscope

        # Get the offsets and flattened pairs array
        offsets = collision_detector.triangle_intersecting_triangles_offsets.numpy()
        pairs_flat = collision_detector.triangle_intersecting_triangles.numpy()
        num_tris = offsets.shape[0] - 1

        print("Colliding triangle pairs (by triangle):")
        colliding_tris = set()

        for tri in range(num_tris):
            if num_intersections[tri]:
                offset = offsets[tri]
                colliding_tris.add(tri)
                for intersection in range(num_intersections[tri]):
                    t_intersect = pairs_flat[offset + intersection]
                    print("Pair: ", tri, " - ", t_intersect)
                    colliding_tris.add(t_intersect)

        if plot:
            # Plot only the colliding triangles with polyscope
            try:
                import polyscope as ps
                ps.init()
                ps.remove_all_structures()
                if colliding_tris:
                    colliding_tris = sorted(list(colliding_tris))
                    vs_collide = vertices_new
                    ts_collide = ts[colliding_tris]
                    ps.register_surface_mesh("colliding triangles after", vs_collide, ts_collide)
                    if displacement is not None:
                        ps.register_surface_mesh("colliding triangles before", vertices, ts_collide)
                    ps.show()
                else:
                    print("No colliding triangles to plot.")
            except ImportError:
                print("polyscope not installed; skipping visualization.")

            return False
    else:
        return True

import numpy as np

class TruncationAnalyzer:
    def __init__(self, eps=1e-12, equal_tol=1e-12, min_dir_magnitude=1e-6):
        self.eps = float(eps)
        self.equal_tol = float(equal_tol)
        # below this magnitude, directions are considered unreliable
        self.min_dir_magnitude = float(min_dir_magnitude)



    @staticmethod
    def _mag(x):
        return np.linalg.norm(x, axis=1)

    @staticmethod
    def _safe_div(a, b, eps):
        return a / (b + eps)

    def analyze(self,
                vs,                      # (N,3) float
                ts,                      # (F,3) int (unused, but kept for completeness)
                disp_before,             # (N,3) float64
                disp_after,              # (N,3) float64
                intersection_checker=None,  # optional: callable(vertices: (N,3)) -> bool or int,
                print_results=True
                ):
        vs = np.asarray(vs, dtype=np.float64)
        d0 = np.asarray(disp_before, dtype=np.float64)
        d1 = np.asarray(disp_after,  dtype=np.float64)

        assert vs.shape == d0.shape == d1.shape and vs.shape[1] == 3, "shape mismatch"

        # magnitudes
        m0 = self._mag(d0)
        m1 = self._mag(d1)

        # basic stats helper
        def stats(arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                return {
                    "mean": 0.0,
                    "median": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "std": 0.0,
                }
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
            }

        # per-vertex preservation ratio (length)
        pres_ratio = self._safe_div(m1, m0, self.eps)

        # direction cosine (per-vertex)
        cos = np.einsum("ij,ij->i", d0, d1)
        cos = self._safe_div(cos, (m0 * m1), self.eps)
        # clamp numerical noise
        cos = np.clip(cos, -1.0, 1.0)

        # only trust direction where both magnitudes are sufficiently large
        dir_valid = (m0 > self.min_dir_magnitude) & (m1 > self.min_dir_magnitude)
        if np.any(dir_valid):
            cos_valid = cos[dir_valid]
            # deviation angle in degrees between disp_before and disp_after
            angles_deg = np.degrees(np.arccos(cos_valid))
        else:
            cos_valid = np.array([], dtype=np.float64)
            angles_deg = np.array([], dtype=np.float64)

        # exact/zero stats
        unchanged = (np.linalg.norm(d1 - d0, axis=1) <= self.equal_tol)
        zeroed = (m1 <= self.equal_tol)

        # global preservation (two ways)
        global_pres_sum = float(self._safe_div(np.sum(m1), np.sum(m0), self.eps))
        global_pres_l2  = float(self._safe_div(np.linalg.norm(m1), np.linalg.norm(m0), self.eps))

        # optional intersection checks
        inter_before = None
        inter_after = None
        if callable(intersection_checker):
            try:
                inter_before = intersection_checker(vs + d0)
            except Exception:
                inter_before = None
            try:
                inter_after = intersection_checker(vs + d1)
            except Exception:
                inter_after = None

        report = {
            "num_vertices": int(vs.shape[0]),
            "num_triangles": int(ts.shape[0]) if ts is not None and ts.size else 0,

            "disp_before_stats": stats(m0),
            "disp_after_stats": stats(m1),

            "preservation_ratio_stats": stats(pres_ratio),

            # direction analysis (only where displacement is "large enough")
            "direction_magnitude_threshold": float(self.min_dir_magnitude),
            "direction_num_valid": int(dir_valid.sum()),
            "direction_cosine_stats": stats(cos_valid),
            "direction_angle_deg_stats": stats(angles_deg),

            "fraction_unchanged": float(np.mean(unchanged)) if unchanged.size else 0.0,
            "fraction_zeroed": float(np.mean(zeroed)) if zeroed.size else 0.0,

            "global_preservation_sum": global_pres_sum,   # sum ||d'||
            "global_preservation_l2":  global_pres_l2,    # || ||d'|| ||_2

            "intersections_before": int(inter_before) if isinstance(inter_before, (bool, np.bool_, int)) else inter_before,
            "intersections_after":  int(inter_after)  if isinstance(inter_after, (bool, np.bool_, int)) else inter_after,
        }

        if print_results:
            print(report)

        return report

    def combine_stats(self, stats_list, counts):
        """Combine multiple stats dicts with per-item weights."""
        counts = np.asarray(counts, dtype=float)
        N = np.sum(counts)

        if N == 0:
            return dict(mean=0.0, median=0.0, min=0.0, max=0.0, std=0.0)

        means = np.array([s["mean"] for s in stats_list])
        medians = np.array([s["median"] for s in stats_list])
        mins = np.array([s["min"] for s in stats_list])
        maxs = np.array([s["max"] for s in stats_list])
        stds = np.array([s["std"] for s in stats_list])

        # weighted mean
        w_mean = np.sum(means * counts) / N

        # weighted median (approx)
        # sort by median
        order = np.argsort(medians)
        sorted_meds = medians[order]
        sorted_counts = counts[order]
        cum = np.cumsum(sorted_counts)
        w_median = sorted_meds[np.searchsorted(cum, N * 0.5)]

        # min/max across all sets
        w_min = np.min(mins)
        w_max = np.max(maxs)

        # weighted variance
        # var = E[x^2] - (E[x])^2
        # E[x^2] = (sum_i n_i*(std_i^2 + mean_i^2)) / N
        second_moment = np.sum(counts * (stds ** 2 + means ** 2)) / N
        w_var = max(0.0, second_moment - w_mean ** 2)
        w_std = np.sqrt(w_var)

        return dict(
            mean=float(w_mean),
            median=float(w_median),
            min=float(w_min),
            max=float(w_max),
            std=float(w_std),
        )

    def combine_reports(self, reports):
        """Combine multiple truncation reports weighted by num_vertices"""
        counts = [r["num_vertices"] for r in reports]
        N = sum(counts)

        def combine_field(field):
            return self.combine_stats([r[field] for r in reports], counts)

        result = {
            "num_meshes": len(reports),
            "total_vertices": N,

            "disp_before_stats": combine_field("disp_before_stats"),
            "disp_after_stats": combine_field("disp_after_stats"),
            "preservation_ratio_stats": combine_field("preservation_ratio_stats"),
            "direction_cosine_stats": combine_field("direction_cosine_stats"),
            "direction_angle_deg_stats": combine_field("direction_angle_deg_stats"),

            "fraction_unchanged":
                float(sum(r["fraction_unchanged"] * r["num_vertices"]
                          for r in reports) / N) if N > 0 else 0.0,

            "fraction_zeroed":
                float(sum(r["fraction_zeroed"] * r["num_vertices"]
                          for r in reports) / N) if N > 0 else 0.0,

            # global preservation measures must be weighted
            "global_preservation_sum":
                float(sum(r["global_preservation_sum"] * r["num_vertices"]
                          for r in reports) / N) if N > 0 else 0.0,

            "global_preservation_l2":
                float(sum(r["global_preservation_l2"] * r["num_vertices"]
                          for r in reports) / N) if N > 0 else 0.0,

            # intersections: OR combined
            "any_intersection_before":
                any(r["intersections_before"] for r in reports),

            "any_intersection_after":
                any(r["intersections_after"] for r in reports),
        }

        return result