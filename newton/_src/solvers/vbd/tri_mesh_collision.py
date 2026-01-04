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

import numpy as np
import warp as wp

from ...geometry.kernels import (
    compute_edge_aabbs,
    compute_tri_aabbs,
    edge_colliding_edges_detection_kernel,
    init_triangle_collision_data_kernel,
    triangle_triangle_collision_detection_kernel,
    vertex_triangle_collision_detection_kernel,
)
from ...sim import Model


@wp.struct
class TriMeshCollisionInfo:
    # size: 2 x sum(vertex_colliding_triangles_buffer_sizes)
    # every two elements records the vertex index and a triangle index it collides to
    vertex_colliding_triangles: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_min_dist: wp.array(dtype=float)

    triangle_colliding_vertices: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_offsets: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_buffer_sizes: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_min_dist: wp.array(dtype=float)

    # size: 2 x sum(edge_colliding_edges_buffer_sizes)
    # every two elements records the edge index and an edge index it collides to
    edge_colliding_edges: wp.array(dtype=wp.int32)
    edge_colliding_edges_offsets: wp.array(dtype=wp.int32)
    edge_colliding_edges_buffer_sizes: wp.array(dtype=wp.int32)
    edge_colliding_edges_count: wp.array(dtype=wp.int32)
    edge_colliding_edges_min_dist: wp.array(dtype=float)


@wp.func
def get_vertex_colliding_triangles_count(col_info: TriMeshCollisionInfo, v: int):
    return wp.min(col_info.vertex_colliding_triangles_count[v], col_info.vertex_colliding_triangles_buffer_sizes[v])


@wp.func
def get_vertex_colliding_triangles(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision) + 1]


@wp.func
def get_vertex_collision_buffer_vertex_index(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision)]


@wp.func
def get_triangle_colliding_vertices_count(col_info: TriMeshCollisionInfo, tri: int):
    return wp.min(
        col_info.triangle_colliding_vertices_count[tri], col_info.triangle_colliding_vertices_buffer_sizes[tri]
    )


@wp.func
def get_triangle_colliding_vertices(col_info: TriMeshCollisionInfo, tri: int, i_collision: int):
    offset = col_info.triangle_colliding_vertices_offsets[tri]
    return col_info.triangle_colliding_vertices[offset + i_collision]


@wp.func
def get_edge_colliding_edges_count(col_info: TriMeshCollisionInfo, e: int):
    return wp.min(col_info.edge_colliding_edges_count[e], col_info.edge_colliding_edges_buffer_sizes[e])


@wp.func
def get_edge_colliding_edges(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision) + 1]


@wp.func
def get_edge_collision_buffer_edge_index(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision)]


class TriMeshCollisionDetector:
    def __init__(
        self,
        model: Model,
        record_triangle_contacting_vertices=False,
        vertex_positions=None,
        vertex_collision_buffer_pre_alloc=8,
        vertex_collision_buffer_max_alloc=256,
        vertex_triangle_filtering_list=None,
        vertex_triangle_filtering_list_offsets=None,
        triangle_collision_buffer_pre_alloc=16,
        triangle_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=8,
        edge_collision_buffer_max_alloc=256,
        edge_filtering_list=None,
        edge_filtering_list_offsets=None,
        triangle_triangle_collision_buffer_pre_alloc=8,
        triangle_triangle_collision_buffer_max_alloc=256,
        edge_edge_parallel_epsilon=1e-5,
        collision_detection_block_size=16,
    ):
        self.model = model
        self.record_triangle_contacting_vertices = record_triangle_contacting_vertices
        self.vertex_positions = model.particle_q if vertex_positions is None else vertex_positions
        self.device = model.device
        self.vertex_collision_buffer_pre_alloc = vertex_collision_buffer_pre_alloc
        self.vertex_collision_buffer_max_alloc = vertex_collision_buffer_max_alloc
        self.triangle_collision_buffer_pre_alloc = triangle_collision_buffer_pre_alloc
        self.triangle_collision_buffer_max_alloc = triangle_collision_buffer_max_alloc
        self.edge_collision_buffer_pre_alloc = edge_collision_buffer_pre_alloc
        self.edge_collision_buffer_max_alloc = edge_collision_buffer_max_alloc
        self.triangle_triangle_collision_buffer_pre_alloc = triangle_triangle_collision_buffer_pre_alloc
        self.triangle_triangle_collision_buffer_max_alloc = triangle_triangle_collision_buffer_max_alloc

        self.vertex_triangle_filtering_list = vertex_triangle_filtering_list
        self.vertex_triangle_filtering_list_offsets = vertex_triangle_filtering_list_offsets

        self.edge_filtering_list = edge_filtering_list
        self.edge_filtering_list_offsets = edge_filtering_list_offsets

        self.edge_edge_parallel_epsilon = edge_edge_parallel_epsilon

        self.collision_detection_block_size = collision_detection_block_size

        self.lower_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=model.tri_count,
            device=model.device,
        )

        self.bvh_tris = wp.Bvh(self.lower_bounds_tris, self.upper_bounds_tris)

        # collision detections results

        # vertex collision buffers
        self.vertex_colliding_triangles = wp.zeros(
            shape=(2 * model.particle_count * self.vertex_collision_buffer_pre_alloc,),
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_count = wp.array(
            shape=(model.particle_count,), dtype=wp.int32, device=self.device
        )
        self.vertex_colliding_triangles_min_dist = wp.array(
            shape=(model.particle_count,), dtype=float, device=self.device
        )
        self.vertex_colliding_triangles_buffer_sizes = wp.full(
            shape=(model.particle_count,),
            value=self.vertex_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_offsets = wp.array(
            shape=(model.particle_count + 1,), dtype=wp.int32, device=self.device
        )
        self.compute_collision_buffer_offsets(
            self.vertex_colliding_triangles_buffer_sizes, self.vertex_colliding_triangles_offsets
        )

        if record_triangle_contacting_vertices:
            # triangle collision buffers
            self.triangle_colliding_vertices = wp.zeros(
                shape=(model.tri_count * self.triangle_collision_buffer_pre_alloc,), dtype=wp.int32, device=self.device
            )
            self.triangle_colliding_vertices_count = wp.zeros(
                shape=(model.tri_count,), dtype=wp.int32, device=self.device
            )
            self.triangle_colliding_vertices_buffer_sizes = wp.full(
                shape=(model.tri_count,),
                value=self.triangle_collision_buffer_pre_alloc,
                dtype=wp.int32,
                device=self.device,
            )

            self.triangle_colliding_vertices_offsets = wp.array(
                shape=(model.tri_count + 1,), dtype=wp.int32, device=self.device
            )
            self.compute_collision_buffer_offsets(
                self.triangle_colliding_vertices_buffer_sizes, self.triangle_colliding_vertices_offsets
            )
        else:
            self.triangle_colliding_vertices = None
            self.triangle_colliding_vertices_count = None
            self.triangle_colliding_vertices_buffer_sizes = None
            self.triangle_colliding_vertices_offsets = None

        # this is need regardless of whether we record triangle contacting vertices
        self.triangle_colliding_vertices_min_dist = wp.array(shape=(model.tri_count,), dtype=float, device=self.device)

        # edge collision buffers
        self.edge_colliding_edges = wp.zeros(
            shape=(2 * model.edge_count * self.edge_collision_buffer_pre_alloc,), dtype=wp.int32, device=self.device
        )
        self.edge_colliding_edges_count = wp.zeros(shape=(model.edge_count,), dtype=wp.int32, device=self.device)
        self.edge_colliding_edges_buffer_sizes = wp.full(
            shape=(model.edge_count,),
            value=self.edge_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.edge_colliding_edges_offsets = wp.array(shape=(model.edge_count + 1,), dtype=wp.int32, device=self.device)
        self.compute_collision_buffer_offsets(self.edge_colliding_edges_buffer_sizes, self.edge_colliding_edges_offsets)
        self.edge_colliding_edges_min_dist = wp.array(shape=(model.edge_count,), dtype=float, device=self.device)

        self.lower_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=model.edge_count,
            device=model.device,
        )

        self.bvh_edges = wp.Bvh(self.lower_bounds_edges, self.upper_bounds_edges)

        self.resize_flags = wp.zeros(shape=(4,), dtype=wp.int32, device=self.device)

        self.collision_info = self.get_collision_data()

        # data for triangle-triangle intersection; they will only be initialized on demand, as triangle-triangle intersection is not needed for simulation
        self.triangle_intersecting_triangles = None
        self.triangle_intersecting_triangles_count = None
        self.triangle_intersecting_triangles_offsets = None

    def set_collision_filter_list(
        self,
        vertex_triangle_filtering_list,
        vertex_triangle_filtering_list_offsets,
        edge_filtering_list,
        edge_filtering_list_offsets,
    ):
        self.vertex_triangle_filtering_list = vertex_triangle_filtering_list
        self.vertex_triangle_filtering_list_offsets = vertex_triangle_filtering_list_offsets

        self.edge_filtering_list = edge_filtering_list
        self.edge_filtering_list_offsets = edge_filtering_list_offsets

    def get_collision_data(self):
        collision_info = TriMeshCollisionInfo()

        collision_info.vertex_colliding_triangles = self.vertex_colliding_triangles
        collision_info.vertex_colliding_triangles_offsets = self.vertex_colliding_triangles_offsets
        collision_info.vertex_colliding_triangles_buffer_sizes = self.vertex_colliding_triangles_buffer_sizes
        collision_info.vertex_colliding_triangles_count = self.vertex_colliding_triangles_count
        collision_info.vertex_colliding_triangles_min_dist = self.vertex_colliding_triangles_min_dist

        if self.record_triangle_contacting_vertices:
            collision_info.triangle_colliding_vertices = self.triangle_colliding_vertices
            collision_info.triangle_colliding_vertices_offsets = self.triangle_colliding_vertices_offsets
            collision_info.triangle_colliding_vertices_buffer_sizes = self.triangle_colliding_vertices_buffer_sizes
            collision_info.triangle_colliding_vertices_count = self.triangle_colliding_vertices_count

        collision_info.triangle_colliding_vertices_min_dist = self.triangle_colliding_vertices_min_dist

        collision_info.edge_colliding_edges = self.edge_colliding_edges
        collision_info.edge_colliding_edges_offsets = self.edge_colliding_edges_offsets
        collision_info.edge_colliding_edges_buffer_sizes = self.edge_colliding_edges_buffer_sizes
        collision_info.edge_colliding_edges_count = self.edge_colliding_edges_count
        collision_info.edge_colliding_edges_min_dist = self.edge_colliding_edges_min_dist

        return collision_info

    def compute_collision_buffer_offsets(
        self, buffer_sizes: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32)
    ):
        assert offsets.size == buffer_sizes.size + 1
        offsets_np = np.empty(shape=(offsets.size,), dtype=np.int32)
        offsets_np[1:] = np.cumsum(buffer_sizes.numpy())[:]
        offsets_np[0] = 0

        offsets.assign(offsets_np)

    # Maximum total buffer size (int32 max, accounting for 2x factor in some buffers)
    MAX_BUFFER_TOTAL_SIZE = (2**31 - 1) // 2

    @staticmethod
    def _compute_aligned_buffer_sizes(
        counts: np.ndarray, pre_alloc: int, max_alloc: int, growth_ratio: float = 1.0
    ) -> np.ndarray:
        """
        Compute buffer sizes aligned to 4, clamped between pre_alloc and max_alloc.

        Args:
            counts: Actual collision counts per primitive.
            pre_alloc: Minimum buffer size per primitive.
            max_alloc: Maximum buffer size per primitive.
            growth_ratio: Multiplier for counts to provide headroom (e.g., 1.5 = 50% extra).

        Returns:
            Buffer sizes rounded up to next multiple of 4, clamped to [pre_alloc, max_alloc].
        """
        # Apply growth ratio
        grown = (counts * growth_ratio).astype(np.int32)
        # Round up to next multiple of 4
        aligned = ((grown + 3) // 4) * 4
        # Clamp to valid range
        return np.clip(aligned, pre_alloc, max_alloc).astype(np.int32)

    def resize_collision_buffer_to_fit(self, shrink_to_fit: bool = False, growth_ratio: float = 1.5) -> bool:
        """
        Resize collision buffers based on actual collision counts.

        This function analyzes the collision counts from the last detection pass and
        resizes buffers that overflowed (or shrinks oversized buffers if shrink_to_fit=True).

        Buffer sizes are:
        - Multiplied by growth_ratio to provide headroom
        - Rounded up to the next multiple of 4 for memory alignment
        - Clamped between pre_alloc and max_alloc settings

        Args:
            shrink_to_fit: If True, also shrink buffers that are larger than needed.
                          If False (default), only grow buffers that overflowed.
            growth_ratio: Multiplier for collision counts to provide headroom and reduce
                         resize frequency. Default is 1.5 (50% extra space).
                         Set to 1.0 for exact fit (no headroom).

        Returns:
            True if any buffer was resized, False otherwise.
        """
        flags = self.resize_flags.numpy()
        any_resized = False

        # === Vertex-Triangle Buffer (flag index 0) ===
        if flags[0] or shrink_to_fit:
            resized = self._resize_vertex_triangle_buffer(shrink_to_fit, growth_ratio)
            any_resized = any_resized or resized

        # === Triangle-Vertex Buffer (flag index 1) ===
        if self.record_triangle_contacting_vertices and (flags[1] or shrink_to_fit):
            resized = self._resize_triangle_vertex_buffer(shrink_to_fit, growth_ratio)
            any_resized = any_resized or resized

        # === Edge-Edge Buffer (flag index 2) ===
        if flags[2] or shrink_to_fit:
            resized = self._resize_edge_edge_buffer(shrink_to_fit, growth_ratio)
            any_resized = any_resized or resized

        # === Triangle-Triangle Buffer (flag index 3) ===
        if self.triangle_intersecting_triangles is not None and (flags[3] or shrink_to_fit):
            resized = self._resize_triangle_triangle_buffer(shrink_to_fit, growth_ratio)
            any_resized = any_resized or resized

        # Reset resize flags
        if any_resized:
            self.resize_flags.zero_()
            # Update collision_info struct with new buffer references
            self.collision_info = self.get_collision_data()

        return any_resized

    def _resize_vertex_triangle_buffer(self, shrink_to_fit: bool, growth_ratio: float) -> bool:
        """Resize vertex-triangle collision buffer. Returns True if resized."""
        counts = self.vertex_colliding_triangles_count.numpy()
        current_sizes = self.vertex_colliding_triangles_buffer_sizes.numpy()

        new_sizes = self._compute_aligned_buffer_sizes(
            counts, self.vertex_collision_buffer_pre_alloc, self.vertex_collision_buffer_max_alloc, growth_ratio
        )

        # Determine if resize is needed
        if shrink_to_fit:
            needs_resize = not np.array_equal(new_sizes, current_sizes)
        else:
            # Only grow: resize if any new_size > current_size
            needs_resize = np.any(new_sizes > current_sizes)

        if not needs_resize:
            return False

        # When only growing, take max of current and new
        if not shrink_to_fit:
            new_sizes = np.maximum(new_sizes, current_sizes)

        # Check total size limit (int32 max, with 2x factor for buffer layout)
        total_size = int(np.sum(new_sizes))
        if total_size > self.MAX_BUFFER_TOTAL_SIZE:
            print(f"Warning: vertex-triangle buffer resize skipped, total size {total_size} exceeds limit")
            return False

        # Reallocate buffer
        self.vertex_colliding_triangles = wp.zeros(shape=(2 * total_size,), dtype=wp.int32, device=self.device)
        self.vertex_colliding_triangles_buffer_sizes = wp.array(new_sizes, dtype=wp.int32, device=self.device)
        self.compute_collision_buffer_offsets(
            self.vertex_colliding_triangles_buffer_sizes, self.vertex_colliding_triangles_offsets
        )

        return True

    def _resize_triangle_vertex_buffer(self, shrink_to_fit: bool, growth_ratio: float) -> bool:
        """Resize triangle-vertex collision buffer. Returns True if resized."""
        if self.triangle_colliding_vertices_count is None or self.triangle_colliding_vertices_buffer_sizes is None:
            return False

        counts = self.triangle_colliding_vertices_count.numpy()
        current_sizes = self.triangle_colliding_vertices_buffer_sizes.numpy()

        new_sizes = self._compute_aligned_buffer_sizes(
            counts, self.triangle_collision_buffer_pre_alloc, self.triangle_collision_buffer_max_alloc, growth_ratio
        )

        if shrink_to_fit:
            needs_resize = not np.array_equal(new_sizes, current_sizes)
        else:
            needs_resize = np.any(new_sizes > current_sizes)

        if not needs_resize:
            return False

        if not shrink_to_fit:
            new_sizes = np.maximum(new_sizes, current_sizes)

        # Check total size limit (int32 max)
        total_size = int(np.sum(new_sizes))
        if total_size > self.MAX_BUFFER_TOTAL_SIZE:
            print(f"Warning: triangle-vertex buffer resize skipped, total size {total_size} exceeds limit")
            return False

        self.triangle_colliding_vertices = wp.zeros(shape=(total_size,), dtype=wp.int32, device=self.device)
        self.triangle_colliding_vertices_buffer_sizes = wp.array(new_sizes, dtype=wp.int32, device=self.device)
        self.compute_collision_buffer_offsets(
            self.triangle_colliding_vertices_buffer_sizes, self.triangle_colliding_vertices_offsets
        )

        return True

    def _resize_edge_edge_buffer(self, shrink_to_fit: bool, growth_ratio: float) -> bool:
        """Resize edge-edge collision buffer. Returns True if resized."""
        counts = self.edge_colliding_edges_count.numpy()
        current_sizes = self.edge_colliding_edges_buffer_sizes.numpy()

        new_sizes = self._compute_aligned_buffer_sizes(
            counts, self.edge_collision_buffer_pre_alloc, self.edge_collision_buffer_max_alloc, growth_ratio
        )

        if shrink_to_fit:
            needs_resize = not np.array_equal(new_sizes, current_sizes)
        else:
            needs_resize = np.any(new_sizes > current_sizes)

        if not needs_resize:
            return False

        if not shrink_to_fit:
            new_sizes = np.maximum(new_sizes, current_sizes)

        # Check total size limit (int32 max, with 2x factor for buffer layout)
        total_size = int(np.sum(new_sizes))
        if total_size > self.MAX_BUFFER_TOTAL_SIZE:
            print(f"Warning: edge-edge buffer resize skipped, total size {total_size} exceeds limit")
            return False

        self.edge_colliding_edges = wp.zeros(shape=(2 * total_size,), dtype=wp.int32, device=self.device)
        self.edge_colliding_edges_buffer_sizes = wp.array(new_sizes, dtype=wp.int32, device=self.device)
        self.compute_collision_buffer_offsets(self.edge_colliding_edges_buffer_sizes, self.edge_colliding_edges_offsets)

        return True

    def _resize_triangle_triangle_buffer(self, shrink_to_fit: bool, growth_ratio: float) -> bool:
        """Resize triangle-triangle intersection buffer. Returns True if resized."""
        if self.triangle_intersecting_triangles_count is None or self.triangle_intersecting_triangles_offsets is None:
            return False

        counts = self.triangle_intersecting_triangles_count.numpy()

        # Compute new sizes
        new_sizes = self._compute_aligned_buffer_sizes(
            counts,
            self.triangle_triangle_collision_buffer_pre_alloc,
            self.triangle_triangle_collision_buffer_max_alloc,
            growth_ratio,
        )

        # Get current sizes from offsets (since we don't store per-primitive sizes for tri-tri)
        offsets = self.triangle_intersecting_triangles_offsets.numpy()
        current_sizes = np.diff(offsets)

        if shrink_to_fit:
            needs_resize = not np.array_equal(new_sizes, current_sizes)
        else:
            needs_resize = np.any(new_sizes > current_sizes)

        if not needs_resize:
            return False

        if not shrink_to_fit:
            new_sizes = np.maximum(new_sizes, current_sizes)

        # Check total size limit (int32 max)
        total_size = int(np.sum(new_sizes))
        if total_size > self.MAX_BUFFER_TOTAL_SIZE:
            print(f"Warning: triangle-triangle buffer resize skipped, total size {total_size} exceeds limit")
            return False

        self.triangle_intersecting_triangles = wp.zeros(shape=(total_size,), dtype=wp.int32, device=self.device)

        # Recompute offsets
        new_offsets = np.zeros((len(new_sizes) + 1,), dtype=np.int32)
        new_offsets[1:] = np.cumsum(new_sizes)
        self.triangle_intersecting_triangles_offsets = wp.array(new_offsets, dtype=wp.int32, device=self.device)

        return True

    def rebuild(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = new_pos

        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[
                self.vertex_positions,
                self.model.tri_indices,
            ],
            outputs=[self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris.rebuild()

        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices],
            outputs=[self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.rebuild()

    def refit(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = new_pos

        self.refit_triangles()
        self.refit_edges()

    def refit_triangles(self):
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, self.model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris.refit()

    def refit_edges(self):
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.refit()

    def vertex_triangle_collision_detection(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        self.vertex_colliding_triangles.fill_(-1)

        if self.record_triangle_contacting_vertices:
            wp.launch(
                kernel=init_triangle_collision_data_kernel,
                inputs=[
                    max_query_radius,
                ],
                outputs=[
                    self.triangle_colliding_vertices_count,
                    self.triangle_colliding_vertices_min_dist,
                    self.resize_flags,
                ],
                dim=self.model.tri_count,
                device=self.model.device,
            )
        else:
            self.triangle_colliding_vertices_min_dist.fill_(max_query_radius)

        wp.launch(
            kernel=vertex_triangle_collision_detection_kernel,
            inputs=[
                max_query_radius,
                min_query_radius,
                self.bvh_tris.id,
                self.vertex_positions,
                self.model.tri_indices,
                self.vertex_colliding_triangles_offsets,
                self.vertex_colliding_triangles_buffer_sizes,
                self.triangle_colliding_vertices_offsets,
                self.triangle_colliding_vertices_buffer_sizes,
                self.vertex_triangle_filtering_list,
                self.vertex_triangle_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
            ],
            outputs=[
                self.vertex_colliding_triangles,
                self.vertex_colliding_triangles_count,
                self.vertex_colliding_triangles_min_dist,
                self.triangle_colliding_vertices,
                self.triangle_colliding_vertices_count,
                self.triangle_colliding_vertices_min_dist,
                self.resize_flags,
            ],
            dim=self.model.particle_count,
            device=self.model.device,
            block_dim=self.collision_detection_block_size,
        )

    def edge_edge_collision_detection(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        self.edge_colliding_edges.fill_(-1)
        wp.launch(
            kernel=edge_colliding_edges_detection_kernel,
            inputs=[
                max_query_radius,
                min_query_radius,
                self.bvh_edges.id,
                self.vertex_positions,
                self.model.edge_indices,
                self.edge_colliding_edges_offsets,
                self.edge_colliding_edges_buffer_sizes,
                self.edge_edge_parallel_epsilon,
                self.edge_filtering_list,
                self.edge_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
            ],
            outputs=[
                self.edge_colliding_edges,
                self.edge_colliding_edges_count,
                self.edge_colliding_edges_min_dist,
                self.resize_flags,
            ],
            dim=self.model.edge_count,
            device=self.model.device,
            block_dim=self.collision_detection_block_size,
        )

    def triangle_triangle_intersection_detection(self):
        if self.triangle_intersecting_triangles is None:
            self.triangle_intersecting_triangles = wp.zeros(
                shape=(self.model.tri_count * self.triangle_triangle_collision_buffer_pre_alloc,),
                dtype=wp.int32,
                device=self.device,
            )

        if self.triangle_intersecting_triangles_count is None:
            self.triangle_intersecting_triangles_count = wp.array(
                shape=(self.model.tri_count,), dtype=wp.int32, device=self.device
            )

        if self.triangle_intersecting_triangles_offsets is None:
            buffer_sizes = np.full((self.model.tri_count,), self.triangle_triangle_collision_buffer_pre_alloc)
            offsets = np.zeros((self.model.tri_count + 1,), dtype=np.int32)
            offsets[1:] = np.cumsum(buffer_sizes)

            self.triangle_intersecting_triangles_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=triangle_triangle_collision_detection_kernel,
            inputs=[
                self.bvh_tris.id,
                self.vertex_positions,
                self.model.tri_indices,
                self.triangle_intersecting_triangles_offsets,
            ],
            outputs=[
                self.triangle_intersecting_triangles,
                self.triangle_intersecting_triangles_count,
                self.resize_flags,
            ],
            dim=self.model.tri_count,
            device=self.model.device,
        )
