import math
import os
import numpy as np
import polyscope as ps
import warp as wp
import warp.examples
import cv2

import newton
import newton.examples
from newton import ParticleFlags



@wp.kernel
def apply_rotation(
    angular_speed: float,
    dt: float, 
    t: float,
    q0: wp.array(dtype=wp.vec3),
    fixed: wp.array(dtype=wp.int64),
    q1: wp.array(dtype=wp.vec3)
):
    i = wp.tid()
    particle_index = fixed[i]  
    c0 = math.cos(-angular_speed * (t - dt))
    s0 = math.sin(-angular_speed * (t - dt))
    c1 = math.cos(angular_speed * t)
    s1 = math.sin(angular_speed * t)
    x0, y0, z0 = q0[particle_index][0], q0[particle_index][1], q0[particle_index][2]
    q0[particle_index][0] =  c0 * x0 + s0 * z0
    q0[particle_index][1] =  y0
    q0[particle_index][2] = -s0 * x0 + c0 * z0
    x0, y0, z0 = q0[particle_index][0], q0[particle_index][1], q0[particle_index][2]
    q0[particle_index][0] =  c1 * x0 + s1 * z0
    q0[particle_index][1] =  y0
    q0[particle_index][2] = -s1 * x0 + c1 * z0
    q1[particle_index] = q0[particle_index]
        

def rolled_cloth_mesh(
    length=500.0,
    width=100.0,
    nu=200,
    nv=15,
    inner_radius=10.0,
    thickness=0.4
):
    verts = []
    faces = []

    for i in range(nu):
        u = length * i / (nu - 1)
        theta = u / inner_radius
        r = inner_radius + (thickness / (2.0 * np.pi)) * theta

        for j in range(nv):
            v = width * j / (nv - 1)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = v
            verts.append([x, y, z])

    def idx(i, j):
        return i * nv + j

    for i in range(nu - 1):
        for j in range(nv - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            faces.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

def cylinder_mesh(radius=9.5, height=120.0, segments=64):
    verts = []
    faces = []

    for i in range(segments):
        t0 = 2 * math.pi * i / segments
        t1 = 2 * math.pi * (i + 1) / segments

        x0, z0 = radius * math.cos(t0), radius * math.sin(t0)
        x1, z1 = radius * math.cos(t1), radius * math.sin(t1)

        y0 = -height * 0.5
        y1 =  height * 0.5

        base = len(verts)

        verts += [
            [x0, y0, z0],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z0],
        ]

        faces += [
            [base + 0, base + 1, base + 2],
            [base + 0, base + 2, base + 3],
        ]

    return (
        np.array(verts, np.float32),
        np.array(faces, np.int32),
    )



class Example:
    def __init__(self, viewer, video_path=None):

        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 10
        self.use_cuda_graph = False
        self.viewer = viewer

        self.frame_idx = 0
        self.print_every = 50

        

        verts, faces_tri = rolled_cloth_mesh()
        faces_flat = faces_tri.reshape(-1)

        self.faces = faces_tri
        self.num_verts = len(verts)

        verts_c, faces_c = cylinder_mesh(radius=9.9)
        verts_cc, faces_cc = cylinder_mesh(radius=14.9)

        

        scene = newton.ModelBuilder(up_axis="Y", gravity=0)

        scene.add_cloth_mesh(
            pos=wp.vec3(-27.2, 100.0, 7.4),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2),
            scale=1.0,
            vertices=verts,
            indices=faces_flat,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
            particle_radius=0.5,
        )
        
        scene.add_cloth_mesh(
            pos=wp.vec3(-27.2, 50.0, 7.4),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
            scale=1.0,
            vertices=verts_c,
            indices=faces_c.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
        )
        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 50.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
            scale=1.0,
            vertices=verts_cc,
            indices=faces_cc.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
        )
        scene.add_ground_plane()
        scene.color()

        self.model = scene.finalize()
        # --------------------------------------------------------
        # Inner seam = kinematic rotation handle
        # (UNCHANGED vertex choice)
        # --------------------------------------------------------

        self.fixed_point_indices = [199 * 15 + i for i in range(15)]
        if len(self.fixed_point_indices):
            flags = self.model.particle_flags.numpy()
            for fixed_vertex_id in self.fixed_point_indices:
                flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE

            self.model.particle_flags = wp.array(flags)
        self.fixed_point_indices = wp.array(self.fixed_point_indices)

        
        flags = self.model.particle_flags.numpy()
        for id in range(len(verts), len(scene.particle_q)):
            flags[id] = flags[id] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        # --------------------------------------------------------
        # Contacts (UNCHANGED)
        # --------------------------------------------------------

        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-5
        self.model.soft_contact_mu = 0.1

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            handle_self_contact=True,
            self_contact_radius=0.4,
            self_contact_margin=0.6,
            topological_contact_filter_threshold=1,
            truncation_mode=1,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.viewer.set_model(self.model)

        

        self.rest_q = self.state_0.particle_q.numpy().copy()

        self.angular_speed = -np.pi   # rad/sec
        self.spin_duration = 10.0   # seconds (sudden stop)

        

        self.capture()

        
        ps.set_ground_plane_height(0.0)
        ps.init()
        ps.look_at((-20.0, 250.0, 20.0), (-0.1, 0.0, 0.1))
        self.verts = verts
        self.faces_c = faces_c
        self.verts_c = verts_c
        self.faces_cc = faces_cc
        self.ps_mesh1 = ps.register_surface_mesh(
            "RolledCloth",
            self.state_0.particle_q.numpy()[0:len(verts)],
            self.faces,
        )
        self.ps_mesh2 = ps.register_surface_mesh(
            "Cylinder1",
            self.state_0.particle_q.numpy()[len(verts):len(verts)+len(verts_c)],
            faces_c,
        )
        self.ps_mesh3 = ps.register_surface_mesh(
            "Cylinder2",
            self.state_0.particle_q.numpy()[len(verts)+len(verts_c):],
            faces_cc,
        )

        self.video_path = video_path
        self.video_writer = None

    

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    

    def simulate(self):
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            #self.contacts = self.model.collide(self.state_0)
            
            if self.sim_time < self.spin_duration:
                wp.launch(
                    kernel=apply_rotation,
                    dim= len(self.fixed_point_indices),
                    inputs=[
                        self.angular_speed,
                        self.sim_dt,
                        self.sim_time,
                        self.state_0.particle_q,
                        self.fixed_point_indices,
                        self.state_1.particle_q,
                    ],
                )
            
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

  

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.frame_idx += 1

        if self.frame_idx % self.print_every == 0:
            print(f"[INFO] Frame {self.frame_idx}")

    

    def render(self):
        max_frames = 500
        self.ps_mesh1.update_vertex_positions(
            self.state_0.particle_q.numpy()[0:len(self.verts)]
        )
        self.ps_mesh2.update_vertex_positions(
            self.state_0.particle_q.numpy()[len(self.verts):len(self.verts)+len(self.verts_c)]
        )
        self.ps_mesh3.update_vertex_positions(
            self.state_0.particle_q.numpy()[len(self.verts)+len(self.verts_c):]
        )
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
            if self.frame_idx == max_frames:
                example.video_writer.release()
                print("[INFO] Video file finalized")
                exit()
            self.video_writer.write(frame_bgr)
        elif self.frame_idx == max_frames:
            print("[INFO] Simulation Complete")
            exit()

    # ------------------------------------------------------------

    def test(self):
        pass




if __name__ == "__main__":

    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=250)
    parser.set_defaults(viewer="null")
    parser.add_argument("--video-output", type=str, default=None)

    viewer, args = newton.examples.init(parser)

    example = Example(
        newton.viewer.ViewerNull(num_frames=args.num_frames),
        video_path=args.video_output,
    )

    print("[INFO] Simulation started")

    try:
        newton.examples.run(example, args)
    finally:
        ps.shutdown()
        print("[INFO] Simulation finished")
