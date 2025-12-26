import math
import os
import numpy as np
import polyscope as ps
import warp as wp
import cv2
import warp.examples
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Example:
    def __init__(self, viewer, video_path: str | None = None):

        # ---------------- Simulation parameters ----------------
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 4
        self.use_cuda_graph = False
        self.viewer = viewer

        # ---------------- Progress tracking ----------------
        self.frame_idx = 0
        self.print_every = 100

        # ---------------- Load cloth mesh ----------------
        usd_stage = Usd.Stage.Open(
            os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd")
        )
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth/cloth"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        vertices = [wp.vec3(v) for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        # ---------------- Build Newton scene ----------------
        scene = newton.ModelBuilder(up_axis="Y", gravity=-100)

        # Cloth 1
        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 55.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi / 2.0),
            scale=1.0,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-1,
            edge_ke=1e-3,
            edge_kd=0.0,
        )

        # Cloth 2
        offset = 0.3
        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 55.0, offset),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi / 2.0),
            scale=1.0,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-1,
            edge_ke=1e-3,
            edge_kd=0.0,
        )

        scene.add_ground_plane()
        scene.color()

        self.model = scene.finalize()

        # ---------------- Solver ----------------
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            handle_self_contact=True,
            self_contact_radius=0.3,
            self_contact_margin=0.8,
            topological_contact_filter_threshold=1,
            truncation_mode=1,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # ---------------- Polyscope ----------------
        ps.init()
        num_particles = len(vertices)

        self.ps_vis_mesh1 = ps.register_surface_mesh(
            "Cloth1", self.state_0.particle_q.numpy()[:num_particles], self.faces
        )
        self.ps_vis_mesh2 = ps.register_surface_mesh(
            "Cloth2", self.state_0.particle_q.numpy()[num_particles:], self.faces
        )

        ps.set_ground_plane_height(-1)

        # ---------------- Video ----------------
        self.video_path = video_path
        self.video_writer = None

    # ---------------------------------------------------------

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        self.solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_idx += 1

        if self.frame_idx % self.print_every == 0:
            print(f"[INFO] Frame {self.frame_idx}")

    def render(self):
        verts = self.state_0.particle_q.numpy()
        n = verts.shape[0] // 2

        self.ps_vis_mesh1.update_vertex_positions(verts[:n])
        self.ps_vis_mesh2.update_vertex_positions(verts[n:])

        ps.frame_tick()

        if self.video_path:
            frame = ps.screenshot_to_buffer()
            if frame is None:
                return

            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

            if frame.shape[-1] == 4:
                frame = frame[..., :3]

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w, _ = frame_bgr.shape

            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(
                    self.video_path, fourcc, self.fps, (w, h)
                )
                print(f"[INFO] Recording video to {self.video_path}")
            if self.frame_idx == 300:
                example.video_writer.release()
                print("[INFO] Video file finalized")
                exit()
        

            self.video_writer.write(frame_bgr)

    def test(self):
        pass


# =============================================================

if __name__ == "__main__":
    wp.clear_kernel_cache()

    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    parser.set_defaults(viewer="null")
    parser.add_argument(
        "--video-output", type=str, default=None,
        help="Optional path to write MP4"
    )

    viewer, args = newton.examples.init(parser)

    example = Example(
        newton.viewer.ViewerNull(num_frames=args.num_frames),
        video_path=args.video_output
    )

    print("[INFO] Simulation started")

    try:
        newton.examples.run(example, args)
    finally:
        if example.video_writer is not None:
            example.video_writer.release()
            print("[INFO] Video file finalized")

        ps.shutdown()
        print("[INFO] Simulation finished")
