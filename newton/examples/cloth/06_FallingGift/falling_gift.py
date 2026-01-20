# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton
import newton.examples
import cv2
import numpy as np
import polyscope as ps
import os

def cloth_loop_around_box(
    hx=1.6,            # half-size in X (box width / 2)
    hz=2.0,            # half-size in Z (box height / 2)
    width=0.25,        # strap width (along Y)
    center_y=0.0,      # Y position of the strap center
    nu=240,            # resolution along loop
    nv=6,              # resolution across strap width
):
    """
    Vertical closed cloth loop wrapped around a cuboid.
    Loop lies in X-Z plane, strap width is along Y.
    Z is up.
    """

    verts = []
    faces = []

    # Rectangle perimeter length
    P = 4.0 * (hx + hz)

    for i in range(nu):
        s = (i / nu) * P

        # Walk rectangle in X–Z plane (counter-clockwise)
        if s < 2 * hx:
            x = -hx + s
            z = -hz
        elif s < 2 * hx + 2 * hz:
            x = hx
            z = -hz + (s - 2 * hx)
        elif s < 4 * hx + 2 * hz:
            x = hx - (s - (2 * hx + 2 * hz))
            z = hz
        else:
            x = -hx
            z = hz - (s - (4 * hx + 2 * hz))

        for j in range(nv):
            v = (j / (nv - 1) - 0.5) * width
            y = center_y + v
            verts.append([x, y, z])

    def idx(i, j):
        return (i % nu) * nv + j

    # Triangulation
    for i in range(nu):
        for j in range(nv - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            faces.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    return (
        np.array(verts, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )


PYRAMID_TET_INDICES = np.array(
    [
        [0, 1, 3, 9],
        [1, 4, 3, 13],
        [1, 3, 9, 13],
        [3, 9, 13, 12],
        [1, 9, 10, 13],
        [1, 2, 4, 10],
        [2, 5, 4, 14],
        [2, 4, 10, 14],
        [4, 10, 14, 13],
        [2, 10, 11, 14],
        [3, 4, 6, 12],
        [4, 7, 6, 16],
        [4, 6, 12, 16],
        [6, 12, 16, 15],
        [4, 12, 13, 16],
        [4, 5, 7, 13],
        [5, 8, 7, 17],
        [5, 7, 13, 17],
        [7, 13, 17, 16],
        [5, 13, 14, 17],
    ],
    dtype=np.int32,
)

PYRAMID_PARTICLES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (2.0, 1.0, 0.0),
    (0.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
    (2.0, 2.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (2.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0),
    (0.0, 2.0, 1.0),
    (1.0, 2.0, 1.0),
    (2.0, 2.0, 1.0),
]


class Example:
    def __init__(self, viewer, video_path: str | None = None):
        self.viewer = viewer
      

        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.iterations = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.frame_idx = 0
        self.print_every = 10
        self.ply_dir = "ply_frames"
        os.makedirs(self.ply_dir, exist_ok=True)   

        builder = newton.ModelBuilder(up_axis="Z", gravity=-10)
        builder.add_ground_plane()

        # ----------------------------------------------------
        # FOUR STACKED SOFT BODY BLOCKS
        # ----------------------------------------------------
        base_height = 20.0
        spacing = 1.01  # small gap to avoid initial penetration

        for i in range(4):
            builder.add_soft_mesh(
                pos=wp.vec3(0.0, 0.0, base_height + i * spacing),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=PYRAMID_PARTICLES,
                indices=PYRAMID_TET_INDICES.flatten().tolist(),
                density=100,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                #k_damp=1e-5,
                k_damp=0.0,
            )
        
        # ----------------------------------------------------
        # CLOTH STRAP AROUND THE STACK
        # ----------------------------------------------------
        
        strap_verts, strap_faces = cloth_loop_around_box(
            hx = 1.01,
            hz = 2.02,
            width=0.6,
        )
        
        builder.add_cloth_mesh(
            pos=wp.vec3(1.0, 1.0, base_height + 1.5 * spacing + 0.5),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=strap_verts,
            indices=strap_faces.flatten().tolist(),
            density=0.2,
            tri_ke=1e5,
            tri_ka=1e5,
            #tri_kd=1e-5,
            tri_kd=0.0,
            #edge_ke=0.01,
            #edge_kd=1e-2,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.03,
        )
        strap2_verts, strap2_faces = cloth_loop_around_box(
            hx = 1.015,
            hz = 2.025,
            width=0.6,
        )
        builder.add_cloth_mesh(
            pos=wp.vec3(1.0, 1.0, base_height + 1.5 * spacing + 0.5),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi/2),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=strap2_verts,
            indices=strap2_faces.flatten().tolist(),
            density=0.2,
            tri_ke=1e5,
            tri_ka=1e5,
            #tri_kd=1e-5,
            tri_kd=0.0,
            #edge_ke=0.01,
            #edge_kd=1e-2,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.03,
        )
        # ----------------------------------------------------
        # CLOTH
        # ----------------------------------------------------
        '''
        builder.add_cloth_grid(
            pos=wp.vec3(-1.0, -1.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            fix_left=True,
            fix_right=True,
            dim_x=40,
            dim_y=40,
            cell_x=0.25,
            cell_y=0.25,
            mass=0.0025,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-5,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=0.05,
        )
        '''
        builder.color(include_bending=True)

        self.model = builder.finalize()

        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1e-5
        self.model.soft_contact_mu = 0.2

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            self_contact_radius=0.024,
            self_contact_margin=0.037,
            topological_contact_filter_threshold=1,
            handle_self_contact=False,
            truncation_mode=1,
            #dykstra_iterations=3
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        
        self.box_faces = self.model.tri_indices.numpy()[0:len(self.model.tri_indices) - 2 * len(strap_faces)]
        self.box_faces = self.box_faces[0:len(self.box_faces) // 4]
        self.strap_faces = strap_faces
        self.strap2_faces = strap2_faces
        self.capture()
        ps.init()
        ps.set_up_dir("z_up")
        ps.look_at((-40.0, 1.0, 10.0), (0.0,1.0,10.0))
        self.ps_cloth1_mesh = ps.register_surface_mesh(
            "Cloth1", strap_verts, strap_faces
        )
        self.ps_cloth2_mesh = ps.register_surface_mesh(
            "Cloth2", strap2_verts, strap2_faces
        )
        all_verts = self.model.particle_q.numpy()
        self.ps_box1 = ps.register_volume_mesh(
            "Box1", all_verts[0:18], tets=PYRAMID_TET_INDICES
        )
        self.ps_box2 = ps.register_volume_mesh(
            "Box2", all_verts[18:36], tets=PYRAMID_TET_INDICES
        )
        self.ps_box3 = ps.register_volume_mesh(
            "Box3", all_verts[36:54], tets=PYRAMID_TET_INDICES
        )
        self.ps_box4 = ps.register_volume_mesh(
            "Box4", all_verts[54:72], tets=PYRAMID_TET_INDICES
        )
        ps.set_ground_plane_height(0)
        self.ps_cloth1_mesh.set_color((1.0, 0.0, 0.0))  
        self.ps_cloth2_mesh.set_color((1.0, 0.0, 0.0))
        self.ps_box1.set_color((0.0, 0.2, 0.125))
        self.ps_box2.set_color((0.0, 0.2, 0.125))
        self.ps_box3.set_color((0.0, 0.2, 0.125))
        self.ps_box4.set_color((0.0, 0.2, 0.125))
        self.video_path = video_path
        self.video_writer = None
        self.strap1_count = len(strap_verts)
    

    def capture(self):
        self.graph = None
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
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
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
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        max_frames = self.viewer.num_frames
        all_verts = self.state_0.particle_q.numpy()
        
        self.ps_box1.update_vertex_positions(all_verts[0:18])
        self.ps_box2.update_vertex_positions(all_verts[18:36])
        self.ps_box3.update_vertex_positions(all_verts[36:54])
        self.ps_box4.update_vertex_positions(all_verts[54:72])
        self.ps_cloth1_mesh.update_vertex_positions(all_verts[72:72 + self.strap1_count])
        self.ps_cloth2_mesh.update_vertex_positions(all_verts[72 + self.strap1_count:])

        
        ps.frame_tick()
        frame_str = f"{self.frame_idx:06d}"
        
        ply1 = os.path.join(self.ply_dir, f"cloth1_{frame_str}.ply")
        ply2 = os.path.join(self.ply_dir, f"cloth2_{frame_str}.ply")
        ply3 = os.path.join(self.ply_dir, f"box1_{frame_str}.ply")
        ply4 = os.path.join(self.ply_dir, f"box2_{frame_str}.ply")
        ply5 = os.path.join(self.ply_dir, f"box3_{frame_str}.ply")
        ply6 = os.path.join(self.ply_dir, f"box4_{frame_str}.ply")
        
        self.frame_idx += 1
        self.write_ply(ply1, all_verts[72:72 + self.strap1_count], self.strap_faces)
        self.write_ply(ply2, all_verts[72 + self.strap1_count:], self.strap2_faces)
        self.write_ply(ply3, all_verts[0:18], self.box_faces)
        self.write_ply(ply4, all_verts[18:36], self.box_faces)
        self.write_ply(ply5, all_verts[36:54], self.box_faces)
        self.write_ply(ply6, all_verts[54:72], self.box_faces)
        
        if self.frame_idx % self.print_every == 0:
            print("Completed:", self.frame_idx, "frames.")
        
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

if __name__ == "__main__":
    # wp.clear_kernel_cache()

    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=400)
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
