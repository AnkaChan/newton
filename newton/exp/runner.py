# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment runner: assemble a (scene, solver, controller) and run it.

The runner is the only object handed to ``newton.examples.run``. It owns model
assembly order, the IK controller, the substep loop / CUDA-graph capture, reset,
rendering and ``test_final``. The three swept axes are pulled from registries by
``--scene`` / ``--solver`` / ``--control``.

Timing matches IsaacLab: a control update (IK solve + target write) happens once
per env step, followed by ``decimation`` physics steps of ``num_substeps`` VBD
sub-steps each, at a sub-step dt of ``(1/60)/num_substeps``.
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples

from .controllers import CONTROLLERS, make_controller
from .ik import IKController
from .robots import gripper_body_ids
from .scenes import SCENES, SceneHandles, make_scene
from .solvers import SOLVERS, make_solver

_BASE_FPS = 60.0


class _FrameRecorder:
    """Capture GL viewer frames to an MP4 via imageio-ffmpeg (opt-in, ``--record``).

    Frames come from :meth:`ViewerGL.get_frame` as an RGB ``(H, W, 3)`` uint8 array
    (top-left origin, so no flip is needed). The writer is opened lazily on the first
    frame (once the size is known) and finalized by :meth:`close`.
    """

    def __init__(self, path: str, fps: float):
        self.path = path
        self.fps = max(1, int(round(fps)))
        self.count = 0
        self._writer = None

    def capture(self, viewer) -> None:
        img = viewer.get_frame().numpy()
        # libx264 needs even dimensions; crop the odd last row/column if any.
        h, w = img.shape[:2]
        img = img[: h - (h % 2), : w - (w % 2)]
        if self._writer is None:
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(self.path, fps=self.fps, macro_block_size=1)
        self._writer.append_data(img)
        self.count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            print(f"[record] wrote {self.count} frames -> {self.path}")


class Experiment:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.use_graph = bool(args.graph_capture)

        self.scene = make_scene(args.scene, args)
        self.strategy = make_solver(args.solver, args)
        # Let the strategy consult scene-provided physics overrides (materials /
        # solver kwargs) during model assembly and solver construction.
        self.strategy.scene = self.scene
        self.home_pos, self.home_quat = self.scene.home_pose()

        # Timing (IsaacLab: sim.dt = 1/60, num_substeps VBD sub-steps per sim
        # step, decimation sim steps per control/env step).
        self.num_substeps = max(1, int(args.substeps))
        self.decimation = max(1, int(self.strategy.decimation))
        self.substeps_per_step = self.num_substeps * self.decimation
        self.sim_dt = (1.0 / _BASE_FPS) / self.num_substeps
        self.frame_dt = self.decimation / _BASE_FPS

        self._build_model()
        self._build_ik()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = self.strategy.make_collision_pipeline(self.model, water_tight=self.args.water_tight)
        self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)
        self.viewer.show_triangles = True
        cam = self.scene.camera()
        if cam is not None and isinstance(self.viewer, newton.viewer.ViewerGL):
            pos, pitch, yaw, look_at = cam
            self.viewer.set_camera(pos=pos, pitch=pitch, yaw=yaw)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(look_at)

        # Optional MP4 recording of the GL viewer (opt-in via --record).
        self._recorder = None
        self._record_max = int(getattr(args, "num_frames", 0) or 0)
        if getattr(args, "record", None):
            if isinstance(self.viewer, newton.viewer.ViewerGL):
                self._recorder = _FrameRecorder(args.record, fps=1.0 / self.frame_dt)
            else:
                print(f"[record] --record requires --viewer gl (got {type(self.viewer).__name__}); disabled")

        # The model was built at the zero (rest) config so the solver's
        # joint_rest_angle is zero (matching IsaacLab). Spawn the arm at the
        # default config on the STATE, not the model build config.
        spawn_q = np.asarray(self.scene.robot_init_q(), dtype=np.float32)
        n = min(spawn_q.shape[0], self.model.joint_coord_count)
        for state in (self.state_0, self.state_1):
            jq = state.joint_q.numpy()
            jq[:n] = spawn_q[:n]
            state.joint_q.assign(jq)
            state.joint_qd.zero_()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.strategy.sync_initial(self.state_0)

        self.rest_particle_q = wp.clone(self.state_0.particle_q)
        self.init_joint_q = np.array(self.state_0.joint_q.numpy(), copy=True)

        self.controller = make_controller(args.control, args, self)
        self.graph = None
        # Stage a sensible control target before graph capture so the capture
        # warm-up drives toward home (not toward the zero config).
        self._stage_home_target()
        # One-time solver warm-up. SolverVBD's AVBD joint penalties ramp from
        # k_start (<< ke) across frames and are never re-seeded, so a cold solver
        # leaves the arm's joints nearly free and it collapses under gravity on
        # the very first frame (dragging the cloth/cube with it). Step a few
        # times to ramp the penalties, then restore the clean spawn state:
        # reset() keeps the warmed penalties (its reset_internal is a no-op for
        # the monolithic VBD solver).
        warmup_steps = int(getattr(self.strategy, "warmup_steps", 0))
        for _ in range(warmup_steps):
            self.step()
        if warmup_steps:
            self.reset()
            self._stage_home_target()
        self.capture()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def _build_ik(self):
        # IsaacLab solves IK on the full simulation model and re-seeds from the
        # measured joint state each control update (_apply_ik_action). We do the
        # same when the IK target link exists in the full model (it does for the
        # Panda / collapse=False). Otherwise (FR3 collapse=True merges the hand)
        # we fall back to a robot-only IK model with a persisted warm-start.
        labels = list(self.model.body_label)
        self._ik_on_full = any(lbl.endswith(self.scene.ik_link_label) for lbl in labels)
        if self._ik_on_full:
            ik_model = self.model
        else:
            ik_builder = newton.ModelBuilder(gravity=-9.81)
            self.scene.build_robot(ik_builder, collapse_fixed_joints=False)
            ik_model = ik_builder.finalize()
        self.ik = IKController(
            ik_model,
            self.scene.ik_link_label,
            self.scene.ik_link_offset,
            self.home_pos,
            self.home_quat,
            iters=self.scene.ik_iters,
            joint_limit_weight=self.scene.ik_joint_limit_weight,
        )
        self.ik.seed(self.scene.robot_init_q())
        # Scratch for re-seeding the IK from the measured joint state (full-model path).
        self._meas_q = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        self._meas_qd = wp.zeros(self.model.joint_dof_count, dtype=float, device=self.model.device)

    def _build_model(self):
        builder = newton.ModelBuilder(gravity=-9.81)
        builder.rigid_gap = 0.01
        self.strategy.register_attributes(builder)

        robot_bodies, robot_joints, robot_shapes = self.scene.build_robot(
            builder, collapse_fixed_joints=self.strategy.collapse_fixed_joints
        )
        self.strategy.configure_robot(builder, robot_bodies, robot_joints)
        static_shapes = self.scene.add_static(builder)
        self.strategy.filter_collisions(builder, robot_shapes, static_shapes)
        self.scene.add_deformables(builder)

        builder.color()
        if self.args.water_tight:
            # Opt in to volume SDFs for the rigid meshes (e.g. the gripper) that
            # will collide with the cloth; finalize() then builds them. Analytic
            # primitives use their closed-form SDF and are skipped.
            builder.enable_rigid_mesh_sdfs()
        self.model = builder.finalize()
        self.device = self.model.device
        self.strategy.apply_materials(self.model)

        handles = SceneHandles(
            robot_bodies=robot_bodies,
            robot_joints=robot_joints,
            robot_shapes=robot_shapes,
            static_shapes=static_shapes,
            gripper_bodies=gripper_body_ids(self.model, robot_bodies),
            particle_count=self.model.particle_count,
        )
        self.handles = handles
        self.strategy.post_finalize(self.model, handles)
        self.solver = self.strategy.build_solver(self.model, handles, self.args)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        for state in (self.state_0, self.state_1):
            state.joint_q.assign(self.init_joint_q)
            state.joint_qd.zero_()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            wp.copy(state.particle_q, self.rest_particle_q)
            state.particle_qd.zero_()
            state.clear_forces()
            if getattr(state, "body_qd", None) is not None:
                state.body_qd.zero_()

        self.control.clear()
        self.ik.seed(self.init_joint_q)
        wp.copy(self.control.joint_target_q, self.ik.joint_q, count=self.ik.n_coords)

        self.strategy.reset_internal(self.state_0, self.device)
        self.controller.reset()
        self.sim_time = 0.0
        if self.device.is_cuda:
            wp.synchronize_device()

    def _stage_home_target(self):
        """Solve IK to the home pose and write it as the control target."""
        self.ik.set_target(
            wp.vec3(*[float(x) for x in self.home_pos]),
            wp.vec4(*[float(x) for x in self.home_quat]),
        )
        self.ik.solve()
        self.ik.write_control(self.control)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if self.use_graph and self.device.is_cuda:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        # One NewtonManager._simulate_physics_only: rebuild_bvh -> collide once ->
        # num_substeps solver steps with clear_forces AFTER each step+swap.
        self.strategy.pre_substeps(self.solver, self.state_0)  # VBD rebuild_bvh
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.num_substeps):
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.strategy.post_step(self.model, self.state_1)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.state_0.clear_forces()  # IsaacLab clears AFTER step+swap

    def step(self):
        # IsaacLab DirectRLEnv: one _pre_physics_step, then `decimation` x
        # (_apply_action -> sim.step). _apply_action solves IK (re-seeded from the
        # measured joint state) and writes joint targets; sim.step runs one
        # _simulate_physics_only. We mirror that exactly here.
        pos, quat, finger = self.controller.update(self.sim_time, self.frame_dt)
        # Print the commanded IK target position every 60 simulation steps
        # (mirrors IsaacLab _apply_ik_action). This is the single point where
        # the per-step target is produced, before it is fed to the IK solve.
        self._ik_print_count = getattr(self, "_ik_print_count", 0)
        if self._ik_print_count % 60 == 0:
            print(
                f"[IK Target] step={self._ik_print_count} t={self.sim_time:6.2f}s  "
                f"pos=({float(pos[0]):.4f}, {float(pos[1]):.4f}, {float(pos[2]):.4f})  "
                f"quat=({float(quat[0]):.4f}, {float(quat[1]):.4f}, "
                f"{float(quat[2]):.4f}, {float(quat[3]):.4f})"
            )
        self._ik_print_count += 1
        if self.controller.consume_reset():
            self.reset()
        for _ in range(self.decimation):
            if self._ik_on_full:
                # Re-seed IK from the measured joint state (IsaacLab _apply_ik_action).
                newton.eval_ik(self.model, self.state_0, self._meas_q, self._meas_qd)
                wp.copy(self.ik.joint_q, self._meas_q, count=self.ik.n_coords)
            self.ik.set_target(pos, quat)
            self.ik.set_finger(finger)
            self.ik.solve()
            self.ik.write_control(self.control)
            if self.graph is not None:
                with wp.ScopedDevice(self.device):
                    wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.controller.render_overlay(self.viewer)
        self.viewer.end_frame()
        if self._recorder is not None:
            self._recorder.capture(self.viewer)
            if self._record_max and self._recorder.count >= self._record_max:
                # GL's is_running() only tracks the window; signal the event loop to
                # exit so run() finishes and main()'s finally finalizes the MP4.
                try:
                    self.viewer.renderer.app.event_loop.has_exit = True
                except Exception:
                    pass

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(body_q)), "Body positions contain NaN or inf"
        assert np.all(np.isfinite(body_qd)), "Body velocities contain NaN or inf"
        particle_q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Particle positions contain NaN or inf"
        lo = np.min(particle_q, axis=0)
        hi = np.max(particle_q, axis=0)
        bbox = float(np.linalg.norm(hi - lo))
        assert bbox < 5.0, f"Cloth bounding box exploded: {bbox:.2f} m"
        assert lo[2] > -0.1, f"Cloth tunneled below ground: z_min={lo[2]:.4f} m"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser(scene_cls, solver_cls, controller_cls):
    parser = newton.examples.create_parser()
    newton.examples.add_coupled_view_args(parser)
    parser.add_argument("--scene", type=str, default="shirt_pick", choices=sorted(SCENES), help="Scene to run.")
    parser.add_argument("--solver", type=str, default="proxy", choices=sorted(SOLVERS), help="Coupling strategy.")
    parser.add_argument("--control", type=str, default="state_machine", choices=sorted(CONTROLLERS), help="Controller.")
    parser.add_argument("--substeps", type=int, default=10, help="VBD sub-steps per physics step.")
    parser.add_argument(
        "--water-tight",
        action="store_true",
        dest="water_tight",
        default=False,
        help="Enable water-tight rigid-soft contacts (adds edge/face cloth-mesh contacts so cloth cannot "
        "tunnel between the gripper's mesh surface samples). Builds volume SDFs for the rigid meshes.",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="OUT.mp4",
        help="Record the GL viewer to an MP4 at this path (requires --viewer gl). With --num-frames "
        "set, the run stops and finalizes the video after that many frames; otherwise it records until "
        "you close the window.",
    )
    parser.add_argument(
        "--no-graph-capture", action="store_false", dest="graph_capture", default=True, help="Disable CUDA graph capture."
    )
    scene_cls.add_args(parser)
    solver_cls.add_args(parser)
    controller_cls.add_args(parser)
    return parser


def main(num_frames=600):
    # Resolve the selected components first so only their args are registered
    # (avoids cross-component flag collisions).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--scene", default="shirt_pick")
    pre.add_argument("--solver", default="proxy")
    pre.add_argument("--control", default="state_machine")
    known, _ = pre.parse_known_args()

    scene_cls = SCENES.get(known.scene) or next(iter(SCENES.values()))
    solver_cls = SOLVERS.get(known.solver) or next(iter(SOLVERS.values()))
    controller_cls = CONTROLLERS.get(known.control) or next(iter(CONTROLLERS.values()))

    parser = build_parser(scene_cls, solver_cls, controller_cls)
    parser.set_defaults(num_frames=num_frames)
    viewer, args = newton.examples.init(parser)
    experiment = Experiment(viewer, args)
    try:
        newton.examples.run(experiment, args)
    finally:
        recorder = getattr(experiment, "_recorder", None)
        if recorder is not None:
            recorder.close()
