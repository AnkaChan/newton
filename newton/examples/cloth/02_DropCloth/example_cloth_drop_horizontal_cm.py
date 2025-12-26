import math
import os
import numpy as np
import polyscope as ps
import warp as wp
import warp.examples
import cv2
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Example:
    def __init__(self, viewer, video_path: str | None = None):

        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 10
        self.use_cuda_graph = False
        self.viewer = viewer

    
        self.frame_idx = 0
        self.print_every = 50
        self.ply_dir = "ply_frames"
        os.makedirs(self.ply_dir, exist_ok=True)    

       
        usd_stage = Usd.Stage.Open(
            os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd")
        )
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth/cloth"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        vertices = [wp.vec3(v) for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        
        scene = newton.ModelBuilder(up_axis="Y", gravity=-1000)

        # Cloth 1
        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 200.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
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
        offset = 0.4
        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 200.0 + offset, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
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
        # cloth 3
        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 200.0 + 2 * offset, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
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

      
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e2
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

       
        ps.init()
        ps.look_at((0, 250, -150), (0,0,0))
        num_particles = len(vertices)

        self.ps_vis_mesh1 = ps.register_surface_mesh(
            "Cloth1", self.state_0.particle_q.numpy()[:num_particles], self.faces
        )
        self.ps_vis_mesh2 = ps.register_surface_mesh(
            "Cloth2", self.state_0.particle_q.numpy()[num_particles: 2 * num_particles], self.faces
        )
        self.ps_vis_mesh3 = ps.register_surface_mesh(
            "Cloth3", self.state_0.particle_q.numpy()[2 * num_particles: 3 * num_particles], self.faces
        )

        ps.set_ground_plane_height(-20)

        
        self.video_path = video_path
        self.video_writer = None
    def write_ply(self, path, vertices, faces):
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

   

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
        max_frames = 180
        verts = self.state_0.particle_q.numpy()
        n = verts.shape[0] // 3

        self.ps_vis_mesh1.update_vertex_positions(verts[:n])
        self.ps_vis_mesh2.update_vertex_positions(verts[n:2 * n])
        self.ps_vis_mesh3.update_vertex_positions(verts[2 * n:3 * n])

        ps.frame_tick()
        frame_str = f"{self.frame_idx:06d}"

        ply1 = os.path.join(self.ply_dir, f"cloth1_{frame_str}.ply")
        ply2 = os.path.join(self.ply_dir, f"cloth2_{frame_str}.ply")
        ply3 = os.path.join(self.ply_dir, f"cloth3_{frame_str}.ply")
        cloth1 = verts[:n]
        cloth2 = verts[n:2*n]
        cloth3 = verts[2*n:3*n]
        
        self.write_ply(ply1, cloth1, self.faces)
        self.write_ply(ply2, cloth2, self.faces)
        self.write_ply(ply3, cloth3, self.faces)
        

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
            if self.frame_idx == max_frames:
                example.video_writer.release()
                print("[INFO] Video file finalized")
                exit()
            self.video_writer.write(frame_bgr)
        elif self.frame_idx == max_frames:
            print("[INFO] Simulation Complete")
            exit()

            

    def test(self):
        pass



if __name__ == "__main__":
    wp.clear_kernel_cache()

    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=180)
    parser.set_defaults(viewer="null")
    parser.add_argument(
        "--video-output", type=str, default=None,
        help="Optional path to write MP4"
    )

    viewer, args = newton.examples.init(parser)

    example = Example(
        newton.viewer.ViewerNull(num_frames=args.num_frames),
        video_path=args.video_output,
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
