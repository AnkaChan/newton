import math
import os

import numpy as np
import polyscope as ps
import warp as wp
import warp.examples
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
       
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 4
        self.bvh_rebuild_frames = 10

        self.use_cuda_graph = False
        self.viewer = viewer

        # load cloth mesh
        usd_stage = Usd.Stage.Open(
            os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd")
        )
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth/cloth"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        vertices = [wp.vec3(v) for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

  
        scene = newton.ModelBuilder(up_axis="Y", gravity=-9.81)

        # Add first cloth
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
        # add second cloth mesh
        offset = 0.2 # distance between two cloths
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

        # Add ground plane
        scene.add_ground_plane()

        # Optional coloring
        scene.color()

        # finalize model
        self.model = scene.finalize()

        # contact parameters
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0

        # solver
        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            handle_self_contact=True,
            self_contact_radius=0.2,
            self_contact_margin=0.35,
            topological_contact_filter_threshold=1,
            truncation_mode=0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

     
        self.capture()

        ps.init()
        num_particles = len(vertices)
        self.ps_vis_mesh1 = ps.register_surface_mesh(
            "Cloth1", self.state_0.particle_q.numpy()[:num_particles], self.faces
        )
        self.ps_vis_mesh2 = ps.register_surface_mesh(
            "Cloth2", self.state_0.particle_q.numpy()[num_particles:], self.faces
        )

    
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
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        verts = self.state_0.particle_q.numpy()
        num_particles = verts.shape[0] // 2
        self.ps_vis_mesh1.update_vertex_positions(verts[:num_particles])
        self.ps_vis_mesh2.update_vertex_positions(verts[num_particles:])
        ps.frame_tick()

    def test(self):
        pass


if __name__ == "__main__":
    wp.clear_kernel_cache()
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    parser.set_defaults(viewer="null")

    viewer, args = newton.examples.init(parser)

    example = Example(newton.viewer.ViewerNull(num_frames=args.num_frames))
    newton.examples.run(example, args)