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

###########################################################################
# Example Cloth Twist
#
# This simulation demonstrates twisting an FEM cloth model using the VBD
# solver, showcasing its ability to handle complex self-contacts while
# ensuring it remains intersection-free.
#
# Command: python -m newton.examples cloth_twist
#
###########################################################################


import itertools

import numpy as np
import polyscope as ps
import warp as wp
import warp.examples

import newton
import newton.examples
from newton import ParticleFlags

# from S01_WalkThrough import *


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
                    # fi = '{}/{}/{}'.format(fi[0], fi[1], fi[2].split('\n')[0])
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


def get_top_vertices(
    verts,
    axis="z",
    thresh=1e-3,
):
    """
    verts: (N, 3) numpy array
    axis: 'x', 'y', or 'z' — which direction is considered 'top'
    thresh: tolerance below the max value to include vertices

    Returns:
        idx_top: indices of vertices near the top
    """
    verts = np.asarray(verts)
    axis_id = {"x": 0, "y": 1, "z": 2}[axis.lower()]

    vals = verts[:, axis_id]
    vmax = np.max(vals)
    idx_top = np.where(vals >= vmax - thresh)[0]
    return idx_top


@wp.kernel
def initialize_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot_centers: wp.array(dtype=wp.vec3),
    rot_axes: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    # output
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    v_index = vertex_indices_to_rot[wp.tid()]

    p = pos[v_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]
    op = p - rot_center

    root = wp.dot(op, rot_axis) * rot_axis

    root_to_p = p - root

    roots[tid] = root
    roots_to_ps[tid] = root_to_p

    if tid == 0:
        t[0] = 0.0


@wp.kernel
def apply_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    rot_axes: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    angular_velocity: float,
    dt: float,
    end_time: float,
    # output
    pos_0: wp.array(dtype=wp.vec3),
    pos_1: wp.array(dtype=wp.vec3),
):
    cur_t = t[0]
    if cur_t > end_time:
        return

    tid = wp.tid()
    v_index = vertex_indices_to_rot[wp.tid()]

    rot_axis = rot_axes[tid]

    ux = rot_axis[0]
    uy = rot_axis[1]
    uz = rot_axis[2]

    theta = cur_t * angular_velocity

    R = wp.mat33(
        wp.cos(theta) + ux * ux * (1.0 - wp.cos(theta)),
        ux * uy * (1.0 - wp.cos(theta)) - uz * wp.sin(theta),
        ux * uz * (1.0 - wp.cos(theta)) + uy * wp.sin(theta),
        uy * ux * (1.0 - wp.cos(theta)) + uz * wp.sin(theta),
        wp.cos(theta) + uy * uy * (1.0 - wp.cos(theta)),
        uy * uz * (1.0 - wp.cos(theta)) - ux * wp.sin(theta),
        uz * ux * (1.0 - wp.cos(theta)) - uy * wp.sin(theta),
        uz * uy * (1.0 - wp.cos(theta)) + ux * wp.sin(theta),
        wp.cos(theta) + uz * uz * (1.0 - wp.cos(theta)),
    )

    root = roots[tid]
    root_to_p = roots_to_ps[tid]
    root_to_p_rot = R * root_to_p
    p_rot = root + root_to_p_rot

    pos_0[v_index] = p_rot
    pos_1[v_index] = p_rot

    if tid == 0:
        t[0] = cur_t + dt


def generate_cloth_mesh(Nx, Ny, size_x=2.0, size_y=2.0, position=(0, 0)):
    """
    Generate a cloth mesh with cross tessellation.

    Args:
        Nx: Number of vertices along X axis
        Ny: Number of vertices along Y axis
        size_x: Physical size of the cloth along X axis
        size_y: Physical size of the cloth along Y axis
        position: Center position (x, y) of the cloth

    Returns:
        vertices: List of vertex positions (x, y, z)
        faces: List of face indices (triangles)
        uvs: List of UV coordinates
    """
    # Generate grid coordinates
    X = np.linspace(-0.5 * size_x + position[0], 0.5 * size_x + position[0], Nx)
    Y = np.linspace(-0.5 * size_y + position[1], 0.5 * size_y + position[1], Ny)

    U = np.linspace(0, 1, Nx, endpoint=True)
    V = np.linspace(0, 1, Ny, endpoint=True)

    X, Y = np.meshgrid(X, Y, indexing="ij")
    U, V = np.meshgrid(U, V, indexing="ij")

    # Generate Z coordinates (vertical gradient)
    # Z = []
    # for i in range(Nx):
    #     Z.append(np.linspace(0, size_y, Ny))
    Z = np.zeros((Nx, Ny))

    # Create vertex list
    vertices = []
    uvs = []
    for i, j in itertools.product(range(Nx), range(Ny)):
        vertices.append((X[i, j], Z[i, j], Y[i, j]))
        uvs.append((U[i, j], V[i, j]))

    # Generate faces with cross tessellation
    faces = []
    for i, j in itertools.product(range(0, Nx - 1), range(0, Ny - 1)):
        vId = j + i * Ny

        if (j + i) % 2:
            # Diagonal one way
            faces.append((vId, vId + Ny + 1, vId + 1))
            faces.append((vId, vId + Ny, vId + Ny + 1))
        else:
            # Diagonal the other way
            faces.append((vId, vId + Ny, vId + 1))
            faces.append((vId + Ny, vId + Ny + 1, vId + 1))

    return vertices, faces, uvs


all_configs = {
    "cloth_hang": {
        # Simulation timing
        "fps": 60,
        "sim_substeps": 10,
        "iterations": 10,
        "bvh_rebuild_frames": 1,
        # Solver settings
        "use_cuda_graph": True,
        "handle_self_contact": True,
        "use_tile_solve": True,
        "self_contact_radius": 0.2,
        "self_contact_margin": 0.3,
        "topological_contact_filter_threshold": 1,
        "rest_shape_contact_exclusion_radius": 0.0,
        "vertex_collision_buffer_pre_alloc": 64,
        "edge_collision_buffer_pre_alloc": 128,
        # Collider
        "input_collider_mesh": r"CylinderCollider_tri.obj",
        # "input_mesh": r'D:\Data\GTC2026_01\12_17\clothStrandCCollisionRestGeo1p5vC1.obj',
        "collider_position": (0.0, 20.0, 0.0),
        "collider_rotation": (1.0, 0.0, 0.0, 0.0),  # (axis, angle)
        # Physics parameters
        "resolution": (60, 80),
        "shape": (120, 160),
        "up_axis": "y",
        "gravity": -980,
        "cloth_density": 0.02,
        "tri_ke": 1e4,
        "tri_ka": 1e4,
        "tri_kd": 1e-5,
        "edge_ke": 10,
        "edge_kd": 1e-2,
        "collision_stiffness": 2e4,
        "collision_kd": 1e-5,
        "collision_mu": 0.2,
        # Transform
        "cloth_init_position": (0.0, 80.0, 0),
        "cloth_init_rotation": (1.0, 0.0, 0.0, 0.0),
        # Visualization
        "show_ground_plane": False,
        "is_paused": True,
        # Zero velocity duration
        "zero_velocity_duration": 0.1,
    }
}


class Example:
    def __init__(self, viewer):
        global example_cfg

        example_cfg = all_configs["cloth_hang"]

        # setup simulation parameters first
        self.fps = example_cfg["fps"]
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = example_cfg["sim_substeps"]  # must be an even number when using CUDA Graph
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = example_cfg["iterations"]
        # the BVH used by SolverVBD will be rebuilt every self.bvh_rebuild_frames
        # When the simulated object deforms significantly, simply refitting the BVH   can lead to deterioration of the BVH's
        # quality, in this case we need to completely rebuild the tree to achieve better query efficiency.
        self.bvh_rebuild_frames = example_cfg["bvh_rebuild_frames"]

        # collision parameters
        self.use_cuda_graph = example_cfg["use_cuda_graph"]

        # Pause control
        self.is_paused = example_cfg["is_paused"]

        # save a reference to the viewer
        self.viewer = viewer

        faces = []
        scene = newton.ModelBuilder(up_axis=example_cfg["up_axis"], gravity=example_cfg["gravity"])

        vs_collider, _, __, fs_collider = readObj(example_cfg["input_collider_mesh"], convertFacesToOnlyPos=True)
        num_collider_verts = len(vs_collider)

        scene.add_cloth_mesh(
            pos=example_cfg["collider_position"],
            rot=wp.quat_from_axis_angle(
                wp.vec3f(example_cfg["collider_rotation"][:3]), example_cfg["collider_rotation"][3]
            ),
            scale=30,
            vertices=[wp.vec3(v) for v in vs_collider],
            indices=np.array(fs_collider).reshape(-1),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=example_cfg["cloth_density"],
            tri_ke=example_cfg["tri_ke"],
            tri_ka=example_cfg["tri_ka"],
            tri_kd=example_cfg["tri_kd"],
            edge_ke=example_cfg["edge_ke"],
            edge_kd=example_cfg["edge_kd"],
        )
        faces.append(np.array(fs_collider))

        if scene.particle_count > 0:
            for i in range(num_collider_verts):
                scene.particle_mass[i] = 0.0
                scene.particle_flags[i] &= ~ParticleFlags.ACTIVE

        vertices_cloth, faces_cloth, uvs = generate_cloth_mesh(*example_cfg["resolution"], *example_cfg["shape"])

        vertices_cloth = [wp.vec3(v) for v in vertices_cloth]
        faces_cloth = np.array(faces_cloth)

        for i in range(1):
            faces.append(i * len(faces) + num_collider_verts + faces_cloth)
            scene.add_cloth_mesh(
                pos=example_cfg["cloth_init_position"],
                rot=wp.quat_from_axis_angle(
                    wp.vec3f(example_cfg["cloth_init_rotation"][:3]), example_cfg["cloth_init_rotation"][3]
                ),
                scale=1,
                vertices=vertices_cloth,
                indices=faces_cloth.reshape(-1),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=example_cfg["cloth_density"],
                tri_ke=example_cfg["tri_ke"],
                tri_ka=example_cfg["tri_ka"],
                tri_kd=example_cfg["tri_kd"],
                edge_ke=example_cfg["edge_ke"],
                edge_kd=example_cfg["edge_kd"],
            )

        self.faces = np.vstack(faces)

        scene.color()
        self.model = scene.finalize()
        self.model.soft_contact_ke = example_cfg["collision_stiffness"]
        self.model.soft_contact_kd = example_cfg["collision_kd"]
        self.model.soft_contact_mu = example_cfg["collision_mu"]

        color_groups_all = np.concatenate([a.numpy() for a in self.model.particle_color_groups])
        color_groups_all_sorted = np.unique(np.sort(color_groups_all))
        assert color_groups_all_sorted[0] == 0
        assert color_groups_all_sorted[-1] == self.model.particle_count - 1

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            handle_self_contact=example_cfg["handle_self_contact"],
            self_contact_radius=example_cfg["self_contact_radius"],
            self_contact_margin=example_cfg["self_contact_margin"],
            topological_contact_filter_threshold=example_cfg["topological_contact_filter_threshold"],
            rest_shape_contact_exclusion_radius=example_cfg["rest_shape_contact_exclusion_radius"],
            use_tile_solve=example_cfg["use_tile_solve"],
            vertex_collision_buffer_pre_alloc=example_cfg["vertex_collision_buffer_pre_alloc"],
            edge_collision_buffer_pre_alloc=example_cfg["edge_collision_buffer_pre_alloc"],
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # put graph capture into it's own function
        self.capture()

        ps.init()
        self.ps_vis_mesh = ps.register_surface_mesh("Sim", self.state_0.particle_q.numpy(), self.faces)
        ps.set_up_dir(example_cfg["up_axis"] + "_up")
        ps.set_ground_plane_mode("none")

        # Register keyboard callback for pause/unpause
        ps.set_user_callback(self.keyboard_callback)

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

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not self.is_paused:
            if self.sim_time < example_cfg.get("zero_velocity_duration", 2.0):
                self.state_0.particle_qd.zero_()

            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

            self.sim_time += self.frame_dt

    def render(self):
        # if self.viewer is None:
        #     return
        #
        # # Begin frame with time
        # self.viewer.begin_frame(self.sim_time)
        #
        # # Render model-driven content (ground plane)
        # self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

        self.verts_for_vis = self.state_0.particle_q.numpy()
        # print(self.verts_for_vis)
        self.ps_vis_mesh.update_vertex_positions(self.verts_for_vis)
        ps.frame_tick()

    def test(self):
        p_lower = wp.vec3(-0.6, -0.9, -0.6)
        p_upper = wp.vec3(0.6, 0.9, 0.6)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 1.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    wp.clear_kernel_cache()
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=12000)
    parser.set_defaults(viewer="null")

    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(newton.viewer.ViewerNull(num_frames=args.num_frames))

    newton.examples.run(example, args)
