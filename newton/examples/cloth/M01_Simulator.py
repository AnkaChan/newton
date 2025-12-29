# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import json, tqdm
###########################################################################
# Example Sim Cloth
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the wp.sim.ModelBuilder().
#
###########################################################################

import math
import os
from enum import Enum

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
from warp.sim.model import (
    PARTICLE_FLAG_ACTIVE,
)
from PyToolKit.Graphics.IO import *
from PyToolKit.Utility.Path import *

import itertools
import cv2
import polyscope as ps


example_config = {
        "name": "example_01",
        # Simulation timing
        "fps": 60,
        "sim_substeps": 10,
        "iterations": 10,
        "bvh_rebuild_frames": 1,

        # Solver settings
        # "use_cuda_graph": True,
        "use_cuda_graph": False,
        "handle_self_contact": True,
        "use_tile_solve": True,
        "self_contact_radius": 0.2,
        "self_contact_margin": 0.3,
        "topological_contact_filter_threshold": 1,
        "rest_shape_contact_exclusion_radius": 0.0,
        "vertex_collision_buffer_pre_alloc": 64,
        "edge_collision_buffer_pre_alloc": 128,

        # Collider
        "colliders":[
            "cylinder":{
                "input_collider_mesh": r"CylinderCollider_tri.obj",
                "collider_position": (0.0, 20.0, 0.0),
                "collider_rotation": (1.0, 0.0, 0.0, 0.0),  # (axis, angle)
            },
        ],
        "cloth":[
            "cloth_01":{
                "resolution": (60, 80),
                "shape": (120, 160),
                "cloth_density": 0.02,
                "tri_ke": 1e4,
                "tri_ka": 1e4,
                "tri_kd": 1e-5,
                "edge_ke": 10,
                "edge_kd": 1e-2,
                "cloth_init_position": (0.0, 80.0, 0),
                "cloth_init_rotation": (1.0, 0.0, 0.0, 0.0),
            },
        ]

        # Global settings
        "up_axis": "y",
        "gravity": -980,
        "collision_stiffness": 2e4,
        "collision_kd": 1e-5,
        "collision_mu": 0.2,

        # Visualization
        "show_gui": True,
        "is_intially_paused": True,
        "show_ground_plane": False,
}

class Simulator:
    def __init__(
            self,

    ):
        self.rebuild_frames = rebuild_frames
        self.output_ext = output_ext
        self.frame_dt = frame_dt
        self.num_substeps = num_substeps
        self.iterations = iterations

        self.sim_num_frames = sim_num_frames
        self.sim_time = 0.0
        self.write_output = write_output
        self.do_rendering = do_rendering
        # self.do_rendering = False
        self.write_video = write_video

        self.profiler = {}
        self.frame_times = []

        self.outPath = output_path
        self.stage_path = stage_path

        self.input_scale_factor = input_scale_factor

        self.use_cuda_graph = use_cuda_graph

        ps.init()
        ps.set_ground_plane_mode('none')

        self.builder = wp.sim.ModelBuilder()

        self.custom_init()

    def custom_init(self):
        pass

    def finalize(self,
                 fixed_particles = None,
                 collision_stiffness=1e5,
                 soft_contact_kd=1.0e-6,
                 friction_mu=0.2,
                 contact_query_margin=0.5,
                 contact_radius=0.3,
                 handle_self_contact=True,
                 has_ground=False,
                 use_cuda_graph=True
            ):
        self.builder.color(include_bending=False)
        # self.builder.color(include_bending=True)
        self.model = self.builder.finalize()
        self.model.gravity = wp.vec3(0,-1000.0,0)
        self.model.ground = has_ground

        self.model.soft_contact_ke = collision_stiffness
        self.model.soft_contact_kd = soft_contact_kd
        self.model.soft_contact_mu = friction_mu
        self.contact_query_margin = contact_query_margin
        self.contact_radius = contact_radius

        self.rot_end_time = 10

        # self.outPath = r"D:\Data\Sims\Cloth_VBD_rot_gauss_newton"
        if self.outPath is not None:
            os.makedirs(self.outPath, exist_ok=True)

        self.dt = self.frame_dt / self.num_substeps

        if fixed_particles is not None:
            self.set_points_fixed(self.model, fixed_particles)

        # self up contact query and contact detection distances
        self.model.soft_contact_radius = self.contact_radius
        self.model.soft_contact_margin = self.contact_query_margin


        self.integrator = wp.sim.VBDIntegrator(self.model, self.iterations,
                                               handle_self_contact=handle_self_contact,
                                                vertex_collision_buffer_pre_alloc=256,
                                                edge_collision_buffer_pre_alloc=256,
                                                triangle_collision_buffer_pre_alloc=256,
                                               )
        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.state_for_render = self.model.state()

        self.faces = self.model.tri_indices.numpy()
        if self.do_rendering:
            self.verts_for_vis = self.model.particle_q.numpy().copy()
            self.ps_vis_mesh = ps.register_surface_mesh('Sim', self.verts_for_vis, self.faces)

        self.custom_finalize()

        self.use_cuda_graph = use_cuda_graph and self.model.device == "cuda"
        if self.use_cuda_graph:
            self.graph = self.capture_graph()

        if self.write_output and self.output_ext == 'usd' and self.stage_path is not None :
            stage_path = join(self.outPath, self.stage_path)
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=0.01)
        else:
            self.renderer = None


    def capture_graph(self):
        with wp.ScopedCapture() as capture:
            self.runStep()
        return capture.graph

    def custom_finalize(self):
        pass

    def simulate(self):
        if self.write_video:
            out_fid_file = join(self.outPath, "video.mp4")
            pixels = ps.screenshot_to_buffer(False)
            vid_out = cv2.VideoWriter(out_fid_file, cv2.VideoWriter_fourcc(*"h264"), 60,
                                      (pixels.shape[1], pixels.shape[0]),
                                      isColor=True)

        for frame_id in tqdm.tqdm(range(0, self.sim_num_frames)):
            # with wp.ScopedTimer("frame", cuda_filter=wp.TIMING_ALL, dict=self.profiler) as timer:
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.runStep()
            # wp.synchronize()

            # self.frame_times.append(timer.timing_results[0].elapsed)

            if self.rebuild_frames is not None and frame_id % self.rebuild_frames == 0:
                self.rebuild_bvh()

            self.sim_time = self.sim_time + self.frame_dt

            if self.write_output:
                self.save_output(frame_id)
            if self.do_rendering:
                self.render()
                if self.write_video:
                    pixels = ps.screenshot_to_buffer(False)
                    vid_out.write(pixels[:,:,[2,1,0]])
        if self.write_video:
            vid_out.release()

        if self.renderer:
            self.renderer.save()

    def save_output(self, frame_id):
        if self.output_ext == 'ply':
            out_file = os.path.join(self.outPath, "A" + str(frame_id).zfill(6) + '.' + self.output_ext)
            self.savePly(self.state0, out_file)
        elif self.output_ext == 'usd' and self.renderer is not None:
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state0)
            self.renderer.end_frame()

    def set_points_fixed(self, model, fixed_particles):
        if len(fixed_particles):
            flags = model.particle_flags.numpy()
            for fixed_v_id in fixed_particles:
                flags[fixed_v_id] = wp.uint32(int(flags[fixed_v_id]) & ~int(PARTICLE_FLAG_ACTIVE))

            model.particle_flags = wp.array(flags, device=model.device)

    def runStep(self):
        wp.sim.collide(self.model, self.state0)

        for step in range(self.num_substeps):
            self.integrator.simulate(self.model, self.state0, self.state1, self.dt, None)
            (self.state0, self.state1) = (self.state1, self.state0)

        # wp.synchronize()

    def step(self):
        if self.use_cuda_graph  and self.device != "cpu":
            wp.capture_launch(self.graph)
        else:
            self.simulate()


    def render(self):
        self.verts_for_vis = self.state0.particle_q.numpy()
        self.ps_vis_mesh.update_vertex_positions(self.verts_for_vis)
        ps.frame_tick()

    def add_usd_mesh(self, usd_path, model_path_in_usd,
                     tri_ke=1e5, tri_ka=1e5, tri_kd=1e-6, bending_ke=1e5, bending_kd=1e-6,
                     pos=wp.vec3(0.0, 0.0, 0.0),
                     scale=1.0
                     ):
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(model_path_in_usd))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        vertices = [wp.vec3(v) for v in mesh_points]
        faces = mesh_indices.reshape(-1, 3)

        self.builder.add_cloth_mesh(
            pos=pos,
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=scale,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            edge_ke=bending_ke,
            edge_kd=bending_kd,
        )

    def add_obj_mesh(self, obj_path, coloring,
            tri_ke=1e5, tri_ka=1e5, tri_kd=1e-6, bending_ke=10, bending_kd=1e-6,
            pos = wp.vec3(0.0, 0.0, 0.0),
            scale=1.0,
            rot_axis = wp.vec3(1.0, 0.0, 0.0),
            rot_angle = 0.0,
            particle_radius = 0.5,
                     ):
        vs, vns, vts, faces = readObj(obj_path, convertFacesToOnlyPos=True)
        vertices = [wp.vec3(v) * self.input_scale_factor for v in vs]
        fs_flatten = list(itertools.chain(*faces))

        # print(vs)
        # print(fs_flatten)

        self.builder.add_cloth_mesh(
            pos=pos,
            rot=wp.quat_from_axis_angle(rot_axis, rot_angle),
            scale=scale,
            vertices=vertices,
            indices=fs_flatten,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            edge_ke=bending_ke,
            edge_kd=bending_kd,
            particle_radius=particle_radius
        )

    def savePly(self, state, filename):
        """
        Save the vertices and faces to a PLY file.

        Parameters:
        vertices (list of tuples): List of (x, y, z) coordinates.
        faces (list of tuples): List of faces, each face is a tuple of vertex indices.
        filename (str): The name of the output PLY file.
        """
        vertices = state.particle_q.numpy()
        # Header
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(self.faces)}",
            "property list uchar int vertex_indices",
            "end_header"
        ]

        # Write to file
        with open(filename, 'w') as ply_file:
            # Write header
            ply_file.write("\n".join(header) + "\n")

            # Write vertex data
            for vertex in vertices:
                ply_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write face data
            for face in self.faces:
                ply_file.write(f"{len(face)} {' '.join(map(str, face))}\n")

    def recover(self, initial_state_path, frame_id=None):
        path, stem, ext = filePart(initial_state_path)

        if ext == "ply":
            mesh_initial_state = PLYMesh(initial_state_path)
            self.state0.particle_q.assign(mesh_initial_state.vertices)
            self.state1.particle_q.assign(mesh_initial_state.vertices)
        else:
            raise ValueError('Unsupported simulator format: ' + ext)

        if frame_id is not None:
            self.sim_time = self.dt * (frame_id + 1)


    def rebuild_bvh(self):
        # if self.integrator.handle_self_contact:
        #     self.integrator.trimesh_collision_detector.rebuild(self.state0.particle_q)
        #     if self.use_cuda_graph:
        #         self.graph = self.capture_graph()
        #     #
        pass
