# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ==================================================================================
# sim utils
# ==================================================================================
from ._src.sim.graph_coloring import color_graph, plot_graph

__all__ = [
    "color_graph",
    "plot_graph",
]

# ==================================================================================
# mesh utils
# ==================================================================================
from ._src.geometry.utils import remesh_mesh
from ._src.utils.mesh import solidify_mesh


class MeshAdjacency:
    """Deprecated triangle-mesh edge adjacency helper.

    Use :attr:`newton.Model.soft_mesh_adjacency` for simulation adjacency data.
    """

    @dataclass(slots=True)
    class Edge:
        v0: int
        v1: int
        o0: int
        o1: int
        f0: int
        f1: int

    def __init__(self, indices: Sequence[Sequence[int]] | np.ndarray):
        warnings.warn(
            "newton.utils.MeshAdjacency is deprecated; use Model.soft_mesh_adjacency for simulation data.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.edges: dict[tuple[int, int], MeshAdjacency.Edge] = {}
        self.indices = indices

        for tri_id, tri in enumerate(np.asarray(indices, dtype=np.int32).reshape(-1, 3)):
            self.add_edge(int(tri[0]), int(tri[1]), int(tri[2]), tri_id)
            self.add_edge(int(tri[1]), int(tri[2]), int(tri[0]), tri_id)
            self.add_edge(int(tri[2]), int(tri[0]), int(tri[1]), tri_id)

    def add_edge(self, i0: int, i1: int, o: int, f: int):
        key = (min(i0, i1), max(i0, i1))
        if key in self.edges:
            edge = self.edges[key]
            if edge.f1 != -1:
                warnings.warn("Detected non-manifold edge", stacklevel=2)
                return
            edge.o1 = o
            edge.f1 = f
        else:
            edge = MeshAdjacency.Edge(i0, i1, o, -1, f, -1)

        self.edges[key] = edge


__all__ += [
    "MeshAdjacency",
    "remesh_mesh",
    "solidify_mesh",
]

# ==================================================================================
# render utils
# ==================================================================================
from ._src.utils.render import (  # noqa: E402
    bourke_color_map,
)

__all__ += [
    "bourke_color_map",
]

# ==================================================================================
# cable utils
# ==================================================================================
from ._src.utils.cable import (  # noqa: E402
    create_cable_stiffness_from_elastic_moduli,
    create_parallel_transport_cable_quaternions,
    create_straight_cable_points,
    create_straight_cable_points_and_quaternions,
)

__all__ += [
    "create_cable_stiffness_from_elastic_moduli",
    "create_parallel_transport_cable_quaternions",
    "create_straight_cable_points",
    "create_straight_cable_points_and_quaternions",
]

# ==================================================================================
# world utils
# ==================================================================================
from ._src.utils import compute_world_offsets  # noqa: E402

__all__ += [
    "compute_world_offsets",
]

# ==================================================================================
# asset management
# ==================================================================================
from ._src.utils.download_assets import download_asset  # noqa: E402

__all__ += [
    "download_asset",
]

# ==================================================================================
# run benchmark
# ==================================================================================

from ._src.utils.benchmark import EventTracer, event_scope, run_benchmark  # noqa: E402

__all__ += [
    "EventTracer",
    "event_scope",
    "run_benchmark",
]

# ==================================================================================
# import utils
# ==================================================================================

from ._src.utils.import_utils import string_to_warp  # noqa: E402

__all__ += [
    "string_to_warp",
]

# ==================================================================================
# texture utils
# ==================================================================================

from ._src.utils.texture import load_texture, normalize_texture  # noqa: E402

__all__ += [
    "load_texture",
    "normalize_texture",
]
