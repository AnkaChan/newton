# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
# Example Table Cloth (XPBD)
#
# XPBD variant of :mod:`example_table_cloth_vbd`. The scene, assets, replay
# plumbing, pile hull sizing, and rendering path are shared with the VBD
# example; only the cloth constraints and solver are changed to XPBD.
#
# Command: python -m newton.examples table_cloth_xpbd
#
###########################################################################

from __future__ import annotations

import newton
import newton.examples

from newton.examples.bag.capture import (
    add_capture_arguments as _add_capture_arguments,
)
from newton.examples.bag.capture import (
    finalize_capture as _finalize_capture,
)
from newton.examples.bag.example_table_cloth_vbd import (
    _FPS,
    Example as _TableClothVBDExample,
)

_XPBD_ITERS = 10
# The folded mesh starts with near-contact layers and short edges. XPBD's
# spring projection becomes unstable here with the 1e3 stiffness used by the
# simple hanging-cloth grid example, so start much softer.
_XPBD_SPRING_KE = 1.0e1
_XPBD_SPRING_KD = 0.0


class Example(_TableClothVBDExample):
    """Run the table-cloth scene with :class:`newton.solvers.SolverXPBD`.

    XPBD does not solve the triangular FEM constraints used by VBD, so the
    cloth mesh adds stretch/shear springs over the triangle adjacency graph.
    Bending still uses the mesh's initial folded dihedrals as the rest pose.
    """

    _log_prefix = "[table_cloth_xpbd]"

    def _cloth_mesh_solver_kwargs(self):
        return {
            "add_springs": True,
            "spring_ke": _XPBD_SPRING_KE,
            "spring_kd": _XPBD_SPRING_KD,
        }

    def _finalize_builder_for_solver(self, builder):
        # XPBD does not need the VBD graph-coloring pass.
        return

    def _create_solver(self):
        # The folded cloth starts with many nearby layers. VBD has
        # topological/rest-distance filters for cloth self-contact; XPBD's
        # particle-particle contact is unfiltered, so disable that hash-grid
        # path while keeping particle-shape contact against the table, robot,
        # and pile through the CollisionPipeline.
        self.model.particle_grid = None
        self.model.particle_max_radius = 0.0
        return newton.solvers.SolverXPBD(
            self.model,
            iterations=_XPBD_ITERS,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    _add_capture_arguments(
        parser,
        replay_help="Capture rendered frames and auto-build a replay video",
        capture_fps_default=_FPS,
    )
    parser.add_argument(
        "--no-pile",
        action="store_true",
        help=(
            "Skip adding the rigid cloth-pile body and its convex-hull "
            "collider"
        ),
    )
    parser.add_argument(
        "--show-record",
        action="store_true",
        help=(
            "Render the recorded cloth (red-orange overlay) on top of our "
            "simulated cloth. Off by default; can also be toggled at runtime "
            "via the 'Show record' checkbox in the HUD."
        ),
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        save_mp4=getattr(args, "save_mp4", None),
        capture_replay=bool(args.capture_replay),
        capture_frames=int(args.capture_frames),
        capture_fps=int(args.capture_fps),
        capture_dir=str(args.capture_dir),
        capture_format=str(args.capture_format),
        no_pile=bool(args.no_pile),
        show_record=bool(args.show_record),
    )

    if hasattr(example, "gui") and hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(
            lambda ui, ex=example: ex.gui(ui),
            position="side",
        )

    while viewer.is_running() and not getattr(example, "capture_done", False):
        example.step()
        example.render()

    if args.test:
        example.test_final()

    _finalize_capture(example)
