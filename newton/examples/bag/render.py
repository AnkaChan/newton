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

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def render_bag_meshes(
    viewer: Any,
    *,
    sim_time: float,
    viz_state: Any,
    full_positions: Any,
    full_indices: Any,
    proxy_positions: Any,
    proxy_indices: Any,
    render_proxy_overlay: Callable[[bool], None] | None = None,
) -> None:
    """Render the hi-res bag or proxy from viewer cloth/collision flags."""
    proxy_mode = bool(viewer.show_collision or viewer.show_triangles)

    show_triangles = viewer.show_triangles
    viewer.show_triangles = False
    viewer.begin_frame(sim_time)
    viewer.log_state(viz_state)
    viewer.show_triangles = show_triangles

    viewer.log_mesh(
        "/bag",
        full_positions,
        full_indices,
        backface_culling=False,
        hidden=proxy_mode,
        alpha=0.5,
    )

    viewer.log_mesh(
        "/bag_proxy",
        proxy_positions,
        proxy_indices,
        backface_culling=False,
        hidden=not proxy_mode,
    )

    if render_proxy_overlay is not None:
        render_proxy_overlay(proxy_mode)

    viewer.end_frame()
