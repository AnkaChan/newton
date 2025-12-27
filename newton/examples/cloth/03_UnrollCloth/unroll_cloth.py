import numpy as np

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
            faces.append([idx(i, j), idx(i+1, j), idx(i, j+1)])
            faces.append([idx(i+1, j), idx(i+1, j+1), idx(i, j+1)])

    return np.array(verts), np.array(faces)

verts, faces = rolled_cloth_mesh()

def save_ply(filename, verts, faces):
    with open(filename, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

       
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

       
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

save_ply("rolled_cloth.ply", verts, faces)

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
from newton import ParticleFlags


def readObj(vt_path, idMinus1=True, convertFacesToOnlyPos=False):
    vts = []
    fs = []
    vns = []
    vs = []
    with open(vt_path) as objFile:
        lines = objFile.readlines()
        for line in lines:
            l = line.split(" ")
            if "" in l:
                l.remove("")
            if l[0] == "vt":
                assert len(l) == 3
                u = float(l[1])
                v = float(l[2].split("\n")[0])
                vts.append([u, v])
            elif l[0] == "vn":
                assert len(l) == 4
                vns.append([float(l[1]), float(l[2]), float(l[3])])
            elif l[0] == "v":
                assert len(l) == 4 or len(l) == 7  # 7 means vertex has color
                vs.append([float(l[1]), float(l[2]), float(l[3])])
            elif l[0] == "f":
                fs_curr = []
                for i in range(len(l) - 1):
                    fi = l[i + 1].split("/")
                    fi[-1] = fi[-1].split("\n")[0]
                    if idMinus1:
                        f = [int(fi[i]) - 1 for i in range(len(fi))]
                    else:
                        f = [int(fi[i]) for i in range(len(fi))]
                    if convertFacesToOnlyPos:
                        f = f[0]
                    fs_curr.append(f)
                fs.append(fs_curr)
        objFile.close()
    return vs, vns, vts, fs



class Example:
    def __init__(self, viewer, video_path: str | None = None):

    
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 5
        # self.use_cuda_graph = False
        self.use_cuda_graph = True
        self.viewer = viewer
        
        # Pause control
        self.is_paused = False

      
        self.frame_idx = 0
        self.print_every = 50
        self.ply_dir = "ply_frames"
        #os.makedirs(self.ply_dir, exist_ok=True)    

        
        verts, faces = rolled_cloth_mesh()
       
        scene = newton.ModelBuilder(up_axis="Y", gravity=-1000)

        # Load collider mesh
        vs_collider, _, __, fs_collider = readObj("prism.obj", convertFacesToOnlyPos=True)
        num_collider_verts = len(vs_collider)

        # Add collider mesh as static geometry
        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 100.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -math.pi/2),
            scale=wp.vec3(80.0, 100.0, 100.),
            vertices=[wp.vec3(v) for v in vs_collider],
            indices=np.array(fs_collider).reshape(-1),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
        )

        # Make collider static
        if scene.particle_count > 0:
            for i in range(num_collider_verts):
                scene.particle_mass[i] = 0.0
                scene.particle_flags[i] &= ~ParticleFlags.ACTIVE

        # Cloth 1
        scene.add_cloth_mesh(
            pos=wp.vec3(50.0, 180, -40.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.0),
            scale=1.0,
            vertices=verts,
            indices=faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke = 1.0e5,
            tri_ka = 1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
            particle_radius=0.5
        )
        scene.add_ground_plane()
        scene.color()
        self.model = scene.finalize()

        # Store faces and vertex counts for visualization
        self.num_collider_verts = num_collider_verts
        self.collider_faces = np.array(fs_collider)
        self.cloth_faces = faces
        self.faces = np.vstack([self.collider_faces, num_collider_verts + self.cloth_faces])

        fixed_point_indices = [num_collider_verts + 15 * 199 + i for i in range(15)]
        if len(fixed_point_indices):
            flags = self.model.particle_flags.numpy()
            for fixed_vertex_id in fixed_point_indices:
                flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE

            self.model.particle_flags = wp.array(flags)
    

        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-5
        self.model.soft_contact_mu = 0.1

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            handle_self_contact=True,
            self_contact_radius=0.4,
            self_contact_margin=0.6,
            topological_contact_filter_threshold=2,
            truncation_mode=0,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)


        self.viewer.set_model(self.model)

        # Capture CUDA graph
        self.capture()
        
        ps.init()
        #ps.look_at((0, 250, -150), (0,0,0))

        # Register collider mesh
        all_verts = self.state_0.particle_q.numpy()
        collider_verts = all_verts[:self.num_collider_verts]
        self.ps_collider_mesh = ps.register_surface_mesh(
            "Collider", collider_verts, self.collider_faces
        )
        self.ps_collider_mesh.set_color((0.3, 0.3, 0.3))  # Gray color
        
        # Register cloth mesh
        cloth_verts = all_verts[self.num_collider_verts:]
        self.ps_cloth_mesh = ps.register_surface_mesh(
            "Cloth", cloth_verts, self.cloth_faces
        )
        self.ps_cloth_mesh.set_color((0.8, 0.4, 0.4))  # Reddish color

        ps.set_ground_plane_height(0)
        
        # Register keyboard callback for pause/unpause
        ps.set_user_callback(self.keyboard_callback)

        
        self.video_path = video_path
        self.video_writer = None
    
    def keyboard_callback(self):
        """Callback function for keyboard input"""
        if ps.imgui.IsKeyPressed(ps.imgui.GetKeyIndex(ps.imgui.ImGuiKey_Space)):
            self.is_paused = not self.is_paused
            if self.is_paused:
                print("Simulation PAUSED (press Space to resume)")
            else:
                print("Simulation RESUMED")
    
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
    
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
        self.solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not self.is_paused:
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            self.sim_time += self.frame_dt
            self.frame_idx += 1

            if self.frame_idx % self.print_every == 0:
                print(f"[INFO] Frame {self.frame_idx}")

    def render(self):
        max_frames = self.viewer.num_frames
        all_verts = self.state_0.particle_q.numpy()
        
        # Update collider mesh
        collider_verts = all_verts[:self.num_collider_verts]
        self.ps_collider_mesh.update_vertex_positions(collider_verts)
        
        # Update cloth mesh
        cloth_verts = all_verts[self.num_collider_verts:]
        self.ps_cloth_mesh.update_vertex_positions(cloth_verts)
        
        ps.frame_tick()
        frame_str = f"{self.frame_idx:06d}"
        '''
        ply1 = os.path.join(self.ply_dir, f"cloth1_{frame_str}.ply")
        cloth1 = verts
        
        self.write_ply(ply1, cloth1, self.faces)
        '''

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

# =============================================================

if __name__ == "__main__":
    # wp.clear_kernel_cache()

    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=250)
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

    print("[INFO] Simulation started PAUSED (press Space to begin)")

    try:
        newton.examples.run(example, args)
    finally:
        if example.video_writer is not None:
            example.video_writer.release()
            print("[INFO] Video file finalized")

        ps.shutdown()
        print("[INFO] Simulation finished")
