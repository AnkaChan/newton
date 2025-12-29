from newton._src.utils.sim_usd_gtc import (
    Simulator,
    parse_xform
)

import math
import os

import numpy as np
import polyscope as ps
import warp as wp
import warp.examples
from pxr import Usd, UsdGeom

import newton
import newton.examples
from newton import ParticleFlags
import tqdm
from pathlib import Path
from os.path import join

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

def writeObj(vs, vns, vts, fs, outFile, withMtl=False, textureFile=None, convertToMM=False, vIdAdd1=True):
    # write new
    with open(outFile, 'w+') as f:
        fp = Path(outFile)
        outMtlFile = join(str(fp.parent), fp.stem + '.mtl')
        if withMtl:
            f.write('mtllib ./' + fp.stem + '.mtl\n')
            with open(outMtlFile, 'w') as fMtl:
                mtlStr = '''newmtl material_0
    Ka 0.200000 0.200000 0.200000
    Kd 1.000000 1.000000 1.000000
    Ks 1.000000 1.000000 1.000000
    Tr 1.000000
    illum 2
    Ns 0.000000
    map_Kd '''
                assert textureFile is not None
                mtlStr += textureFile
                fMtl.write(mtlStr)

        for i, v in enumerate(vs):

            if convertToMM:
                v[0] = 1000 * v[0]
                v[1] = 1000 * v[1]
                v[2] = 1000 * v[2]
            if len(v) == 3:
                f.write('v {:f} {:f} {:f}\n'.format(v[0], v[1], v[2]))
            elif len(v) == 6:
                f.write('v {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(v[0], v[1], v[2], v[3], v[4], v[5]))

        for i, v in enumerate(vns):
            vn = vns[i]
            f.write('vn {:f} {:f} {:f}\n'.format(vn[0], vn[1], vn[2]))

        for vt in vts:
            f.write('vt {:f} {:f}\n'.format(vt[0], vt[1]))

        if withMtl:
            f.write('usemtl material_0\n')
        for iF in range(len(fs)):
            # if facesToPreserve is not None and iF not in facesToPreserve:
            #     continue
            f.write('f')
            if vIdAdd1:
                for fis in fs[iF]:
                    if  isinstance(fis, list):
                        f.write(' {}'.format('/'.join([str(fi+1) for fi in fis])))
                    else:
                        f.write(' {}'.format(fis+1))

            else:
                for fis in fs[iF]:
                    if isinstance(fis, list):
                        f.write(' {}'.format( '/'.join([str(fi) for fi in fis])))
                    else:
                        f.write(' {}'.format(fis))
            f.write('\n')
        f.close()


class SimulatorClothDroid(Simulator):
    def __init__(self, input_path, output_path, num_frames):
        super().__init__(input_path, output_path, num_frames)

    def _setup_model_attributes(self):
        super()._setup_model_attributes()

    def _setup_integrator(self):
        pass

    def _collect_animated_colliders(self, builder, path_body_map):
        super()._collect_animated_colliders(builder, path_body_map)
        global example_cfg

        # SET UP CLOTH HERE, since it's the last function before .finalize() call
        rest_shape_data = []
        example_cfg["cloth_mesh_info"] = []  # Store vertex ranges for each mesh
        vertex_offset = 0
        
        for i, cloth_path in enumerate(example_cfg["input_cloth"]):

            usd_geom = UsdGeom.Mesh(self.in_stage.GetPrimAtPath(cloth_path))

            mesh_points = np.array(usd_geom.GetPointsAttr().Get())
            vertices = [wp.vec3(v) for v in mesh_points]
            mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

            transform = parse_xform(usd_geom)
            # Extract position and rotation
            position = wp.transform_get_translation(transform)  # wp.vec3
            rotation = wp.transform_get_rotation(transform)

            # Store for OBJ export (apply transform)
            # transformed_points = np.array([wp.transform_point(transform, wp.vec3(*p)) for p in mesh_points])
            rest_shape_data.append({
                "vertices": np.array(vertices),
                "indices": mesh_indices,
                "path": cloth_path
            })
            
            # Store mesh info for separate USD export
            mesh_name = Path(cloth_path).stem
            example_cfg["cloth_mesh_info"].append({
                "path": cloth_path,
                "name": mesh_name,
                "vertex_start": vertex_offset,
                "vertex_count": len(vertices),
                "indices": mesh_indices,
            })
            vertex_offset += len(vertices)

            self.builder.add_cloth_mesh(
            pos=position,
            rot=rotation,
            scale=1,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=example_cfg["cloth_density"],
            tri_ke=example_cfg["tri_ke"],
            tri_ka=example_cfg["tri_ka"],
            tri_kd=example_cfg["tri_kd"],
            edge_ke=example_cfg["edge_ke"],
            edge_kd=example_cfg["edge_kd"],
            particle_radius=example_cfg["particle_radius"]
        )

        # Save rest shape meshes separately if requested
        if example_cfg.get("save_rest_shape_obj", False):
            obj_dir = example_cfg.get("output_path", ".")
            os.makedirs(obj_dir, exist_ok=True)
            
            # Save individual meshes
            for data in rest_shape_data:
                # Extract filename from USD path (last component)
                path_stem = Path(data["path"]).stem
                rest_obj_path = os.path.join(obj_dir, f"{path_stem}_rest.obj")
                
                faces = data["indices"].reshape(-1, 3).tolist()
                writeObj(data["vertices"].tolist(), [], [], faces, rest_obj_path, withMtl=False, vIdAdd1=True)
                print(f"Saved rest shape to {rest_obj_path}")
            
            # Also save the full combined mesh
            if len(rest_shape_data) > 0:
                all_vertices = []
                all_faces = []
                vertex_offset = 0
                
                for data in rest_shape_data:
                    all_vertices.extend(data["vertices"].tolist())
                    offset_indices = data["indices"] + vertex_offset
                    all_faces.extend(offset_indices.reshape(-1, 3).tolist())
                    vertex_offset += len(data["vertices"])
                
                full_rest_path = os.path.join(obj_dir, "full_rest.obj")
                writeObj(all_vertices, [], [], all_faces, full_rest_path, withMtl=False, vIdAdd1=True)
                print(f"Saved full rest shape to {full_rest_path}")

        init_positions = []
        init_shape_data = []
        
        for i, cloth_rest_path in enumerate(example_cfg["init_states"]):
            usd_geom_initial_shape = UsdGeom.Mesh(self.in_stage.GetPrimAtPath(cloth_rest_path))
            mesh_points_initial_org = np.array(usd_geom_initial_shape.GetPointsAttr().Get())
            mesh_indices_initial = np.array(usd_geom_initial_shape.GetFaceVertexIndicesAttr().Get())
            transform_initial_shape = parse_xform(usd_geom_initial_shape)

            # Apply transform_initial_shape to mesh_points_initial
            mesh_points_initial = np.array(
                [wp.transform_point(transform_initial_shape, wp.vec3(*p)) for p in mesh_points_initial_org]
            )

            init_positions.append(mesh_points_initial)
            init_shape_data.append({
                "vertices": mesh_points_initial,
                "indices": mesh_indices_initial,
                "path": cloth_rest_path
            })

        # Concatenate all init_positions into a single array
        if len(init_positions) > 0:
            example_cfg["init_positions"] = np.concatenate(init_positions, axis=0)

        # Save initial shape meshes separately if requested
        if example_cfg.get("save_initial_shape_obj", False):
            obj_dir = example_cfg.get("output_path", ".")
            os.makedirs(obj_dir, exist_ok=True)
            
            # Save individual meshes
            for data in init_shape_data:
                # Extract filename from USD path (last component)
                path_stem = Path(data["path"]).stem
                initial_obj_path = os.path.join(obj_dir, f"{path_stem}_initial.obj")
                
                faces = data["indices"].reshape(-1, 3).tolist()
                writeObj(data["vertices"].tolist(), [], [], faces, initial_obj_path, withMtl=False, vIdAdd1=True)
                print(f"Saved initial shape to {initial_obj_path}")
            
            # Also save the full combined mesh
            if len(init_shape_data) > 0:
                all_vertices = []
                all_faces = []
                vertex_offset = 0
                
                for data in init_shape_data:
                    all_vertices.extend(data["vertices"].tolist())
                    offset_indices = data["indices"] + vertex_offset
                    all_faces.extend(offset_indices.reshape(-1, 3).tolist())
                    vertex_offset += len(data["vertices"])
                
                full_initial_path = os.path.join(obj_dir, "full_initial.obj")
                writeObj(all_vertices, [], [], all_faces, full_initial_path, withMtl=False, vIdAdd1=True)
                print(f"Saved full initial shape to {full_initial_path}")

        self.setup_fixed_points()
        self.builder.color()

    def setup_fixed_points(self):
        # Select fixed vertices from the concatenated init_positions array
        init_positions = example_cfg["init_positions"]
        fixed_vertices = []
        
        if example_cfg["fixed_points_scheme"]["name"] == "top":
            fixed_vertices = get_top_vertices(
                init_positions, "z", thresh=example_cfg["fixed_points_scheme"]["threshold"]
            )
        elif (
                isinstance(example_cfg["fixed_points_scheme"], dict)
                and example_cfg["fixed_points_scheme"].get("name") == "box"
        ):
            boxes = example_cfg["fixed_points_scheme"].get("boxes", [])

            for box in boxes:
                min_x, min_y, min_z, max_x, max_y, max_z = box
                mask = (
                    (init_positions[:, 0] >= min_x)
                    & (init_positions[:, 0] <= max_x)
                    & (init_positions[:, 1] >= min_y)
                    & (init_positions[:, 1] <= max_y)
                    & (init_positions[:, 2] >= min_z)
                    & (init_positions[:, 2] <= max_z)
                )
                idx_in_box = np.where(mask)[0]
                fixed_vertices.extend(idx_in_box.tolist())
        
        fixed_vertices = np.unique(fixed_vertices)

        for fixed_v_id in fixed_vertices:
            self.builder.particle_flags[fixed_v_id] = self.builder.particle_flags[fixed_v_id] & ~ParticleFlags.ACTIVE


    def _setup_solver_attributes(self):
        super()._setup_solver_attributes()
        self.sim_substeps = example_cfg["sim_substeps"]

def create_stage_from_path(input_path) -> Usd.Stage:
    stage = Usd.Stage.Open(input_path, Usd.Stage.LoadAll)
    flattened = stage.Flatten()
    out_stage = Usd.Stage.Open(flattened.identifier)
    return out_stage

all_configs = {
    "walk_through": {
        # in & out
        "output_path": r"D:\Data\GTC2026_01\12_17",
        "input_cloth" : [
            "/World/ClothStrands1p50CollisionRest_01/geo/clothStrandACollisionRestGeo1p5vC1",
            "/World/ClothStrands1p50CollisionRest_01/geo/clothStrandBCollisionRestGeo1p5vC1",
            "/World/ClothStrands1p50CollisionRest_01/geo/clothStrandCCollisionRestGeo1p5vC1",
            # "/World/ClothStrands1p50CollisionRest_01/geo/clothStrandACollisionRestGeo1p5vB2",
            # "/World/ClothStrands1p50CollisionRest_01/geo/clothStrandBCollisionRestGeo1p5vB2",
            # "/World/ClothStrands1p50CollisionRest_01/geo/clothStrandCCollisionRestGeo1p5vB2",
        ],
        "init_states" : [
            "/World/ClothStrands_01/geo/clothStrandACollisionGeo1p5vC1",
            "/World/ClothStrands_01/geo/clothStrandBCollisionGeo1p5vC1",
            "/World/ClothStrands_01/geo/clothStrandCCollisionGeo1p5vC1",
        ],

        "init_positions" : None,
        "fixed_points_scheme":  {
            "name" : "top",
            "threshold": 0.3,
        },
        "camera_cfg" : {
            "pos": wp.vec3(9.17, 14.59, 1.14),  # Position
            "pitch": -3.2,  # Pitch in degrees
            "yaw": -335.6,
        },
        
        # Simulation timing
        "fps": 60,
        "sim_substeps": 20,
        "iterations": 5,
        "bvh_rebuild_frames": 1,
        
        # Physics parameters
        "cloth_density": 1.0,
        "tri_ke": 500.0,
        "tri_ka": 500.0,
        "tri_kd": 1e-5,
        "edge_ke": 1e-2,
        "edge_kd": 1e-3,
        "particle_radius": 0.03,

        # Collision parameters
        "collision_stiffness": 1e3,
        "collision_kd": 0,
        "collision_mu": 0.2,
        "collision_query_margin": 0.35,
        "collision_filter_threshold": 2,
        "vertex_collision_buffer_pre_alloc": 128,
        "edge_collision_buffer_pre_alloc": 256,
        
        # SolverVBD parameters
        "handle_self_contact": True,
        "use_tile_solve": True,
        "self_contact_radius": 0.003,
        "self_contact_margin": 0.008,
        "topological_contact_filter_threshold": 2,
        "rest_shape_contact_exclusion_radius": 0.006,
        "solver_vertex_collision_buffer_pre_alloc": 64,
        "solver_edge_collision_buffer_pre_alloc": 128,
        
        # CUDA Graph
        "use_cuda_graph": True,
        # "use_cuda_graph": False,

        # sim
        # run
        "preroll_frames": 800,
        "load_preroll_state": False,
        "save_preroll_state": True,
        # obj export
        "save_rest_shape_obj": True,
        "save_initial_shape_obj": True,
        "start_time": 0.0,
        "save_usd": True

    }
}




class Example:
    def __init__(self, input_path, output_path, viewer, args):

        global example_cfg

        example_cfg = all_configs["walk_through"]
        
        # setup simulation parameters first
        self.fps = example_cfg["fps"]
        self.frame_dt = 1.0 / self.fps
        self.num_frames = args.num_frames

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = example_cfg["sim_substeps"]  # must be an even number when using CUDA Graph
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = example_cfg["iterations"]
        # the BVH used by SolverVBD will be rebuilt every self.bvh_rebuild_frames
        # When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        # quality, in this case we need to completely rebuild the tree to achieve better query efficiency.
        self.bvh_rebuild_frames = example_cfg["bvh_rebuild_frames"]

        # collision parameters
        self.collision_stiffness = example_cfg["collision_stiffness"]
        self.collision_kd = example_cfg["collision_kd"]
        self.collision_mu = example_cfg["collision_mu"]
        self.collision_query_margin = example_cfg["collision_query_margin"]
        self.collision_filter_threshold = example_cfg["collision_filter_threshold"]
        self.vertex_collision_buffer_pre_alloc = example_cfg["vertex_collision_buffer_pre_alloc"]
        self.edge_collision_buffer_pre_alloc = example_cfg["edge_collision_buffer_pre_alloc"]

        self.use_cuda_graph = example_cfg["use_cuda_graph"]

        # save a reference to the viewer
        self.viewer = viewer

        self.usd_sim = SimulatorClothDroid(input_path, output_path, args.num_frames)

        self.model = self.usd_sim.model
        self.faces = self.model.tri_indices.numpy()

        self.model.soft_contact_ke = self.collision_stiffness
        self.model.soft_contact_kd = self.collision_kd
        self.model.soft_contact_mu = self.collision_mu

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.state_0.particle_q.assign(example_cfg["init_positions"])
        self.state_1.particle_q.assign(example_cfg["init_positions"])

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            handle_self_contact=example_cfg["handle_self_contact"],
            use_tile_solve=example_cfg["use_tile_solve"],
            self_contact_radius=example_cfg["self_contact_radius"],
            self_contact_margin=example_cfg["self_contact_margin"],
            topological_contact_filter_threshold=example_cfg["topological_contact_filter_threshold"],
            rest_shape_contact_exclusion_radius=example_cfg["rest_shape_contact_exclusion_radius"],
            vertex_collision_buffer_pre_alloc=example_cfg["solver_vertex_collision_buffer_pre_alloc"],
            edge_collision_buffer_pre_alloc=example_cfg["solver_edge_collision_buffer_pre_alloc"],
        )

        self.viewer.set_model(self.model)
        if "camera_cfg" in example_cfg:
            self.viewer.set_camera(
                pos=example_cfg["camera_cfg"]["pos"],  # Position
                pitch=example_cfg["camera_cfg"]["pitch"],  # Pitch in degrees
                yaw=example_cfg["camera_cfg"]["yaw"],  # Yaw in degrees
            )

        # put graph capture into it's own function
        self.capture()

        if example_cfg["preroll_frames"]>0:
            self.run_preroll(example_cfg["output_path"])

        self.usd_sim.time_step_wp.fill_(int(example_cfg["start_time"])* (example_cfg["fps"]*example_cfg["sim_substeps"]))

        # ps.init()
        # self.ps_vis_mesh = ps.register_surface_mesh("Sim", self.state_0.particle_q.numpy(), self.faces)

        if example_cfg["save_usd"]:
            # Create separate ViewerUSD and model for each cloth mesh
            self.viewer_usds = []
            for mesh_info in example_cfg["cloth_mesh_info"]:
                mesh_name = mesh_info["name"]
                output_path = join(example_cfg["output_path"], f"{mesh_name}.usd")
                viewer_usd = newton.viewer.ViewerUSD(output_path=output_path, num_frames=self.num_frames)
                
                # Create a lightweight model for this mesh
                mesh_builder = newton.ModelBuilder(up_axis='z', gravity=-9.80)
                
                # Get initial vertices for this mesh
                v_start = mesh_info["vertex_start"]
                v_count = mesh_info["vertex_count"]
                init_particles = example_cfg["init_positions"][v_start:v_start+v_count]
                vertices = [wp.vec3(p) for p in init_particles]
                
                # Add cloth mesh with re-indexed indices (starting from 0)
                mesh_builder.add_cloth_mesh(
                    pos=wp.vec3(0.),
                    rot=wp.quat_identity(),
                    scale=1,
                    vertices=vertices,
                    indices=mesh_info["indices"],
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    density=example_cfg["cloth_density"],
                    tri_ke=example_cfg["tri_ke"],
                    tri_ka=example_cfg["tri_ka"],
                    tri_kd=example_cfg["tri_kd"],
                    edge_ke=example_cfg["edge_ke"],
                    edge_kd=example_cfg["edge_kd"],
                    particle_radius=example_cfg["particle_radius"]
                )
                
                mesh_model = mesh_builder.finalize()
                viewer_usd.set_model(mesh_model)
                
                self.viewer_usds.append({
                    "viewer": viewer_usd,
                    "model": mesh_model,
                    "mesh_info": mesh_info
                })
        else:
            self.viewer_usds = []

        self.frame = 0

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        self.preroll_graph = None
        if wp.get_device().is_cuda and self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                for _ in range(self.sim_substeps):
                    self.contacts = self.model.collide(self.state_0)
                    self.state_0.clear_forces()
                    self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                    # swap states
                    self.state_0, self.state_1 = self.state_1, self.state_0

            self.use_cuda_graph = capture.graph

    def _update_animated_colliders(self):
        wp.launch(
            self.usd_sim._update_animated_colliders_kernel,
            dim=len(self.usd_sim.animated_colliders_body_ids),
            inputs=[
                self.usd_sim.time_step_wp,
                self.usd_sim.collider_body_q,
                self.usd_sim.collider_body_qd,
                self.usd_sim.animated_colliders_joint_q_start_wp,
                self.usd_sim.animated_colliders_joint_qd_start_wp,
                self.usd_sim.animated_colliders_body_ids_wp,
            ],
            outputs=[self.state_0.body_q, self.state_0.body_qd, self.state_0.joint_q, self.state_0.joint_qd],
        )

        self.usd_sim._advance_substep_time()

    def run_preroll(self, output_path):
        preroll_frames = example_cfg.get("preroll_frames", 0)
        preroll_state_path = os.path.join(output_path, "preroll.npy")
        load_preroll_state = example_cfg.get("load_preroll_state", False)
        # If not explicitly provided, deduce preroll state path from the output path
        # ps.init()
        # ps.set_up_dir("z_up")
        # self.ps_vis_mesh = ps.register_surface_mesh("Sim", self.state_0.particle_q.numpy(), self.faces)

        if load_preroll_state and preroll_state_path is not None:
            preroll_state = np.load(preroll_state_path, allow_pickle=True).item()
            self.state_0.particle_q.assign(preroll_state["particle_q"])
            self.state_1.particle_q.assign(preroll_state["particle_q"])
        elif preroll_frames > 0:

            state = self.state_0
            for frame in tqdm.tqdm(range(preroll_frames), desc="Preroll Frames"):
                if self.preroll_graph:
                    wp.capture_launch(self.preroll_graph)
                else:
                    for substep in range(self.sim_substeps):
                        self.contacts = self.model.collide(self.state_0)

                        self.state_0.clear_forces()

                        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

                        # swap states
                        (self.state_0, self.state_1) = (self.state_1, self.state_0)

                        if frame < example_cfg.get("preroll_zero_velocity_ratio", 0.1) * preroll_frames:
                            self.state_0.particle_qd.zero_()
                            self.state_1.particle_qd.zero_()
                        # else:
                        #     self.state_0.particle_qd.assign(self.state_0.particle_qd * self.run_cfg.get("preroll_velocity_damping_ratio", 0.99))
                        #     self.state_1.particle_qd.assign(self.state_1.particle_qd * self.run_cfg.get("preroll_velocity_damping_ratio", 0.99))

                self.viewer.begin_frame(self.sim_time)
                self.viewer.log_state(self.state_0)

                self.viewer.end_frame()

                # self.verts_for_vis = self.state_0.particle_q.numpy()
                # # print(self.verts_for_vis)
                # self.ps_vis_mesh.update_vertex_positions(self.verts_for_vis)
                # ps.frame_tick()

            state = self.state_0  # assuming self.simulate() advances self.state_0

            # Save the last frame's state
            last_frame = {
                "particle_q": np.array(state.particle_q.numpy()),
                "particle_qd": np.array(state.particle_qd.numpy()),
            }
            if example_cfg['save_preroll_state']:
                np.save(preroll_state_path, last_frame)

    def simulate(self):
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.contacts = self.model.collide(self.state_0)
            self._update_animated_colliders()
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()



    def render(self):
        if self.viewer is None:
            return

        # Begin frame with time
        self.viewer.begin_frame(self.sim_time)

        # Render model-driven content (ground plane)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

        # Render each cloth mesh separately to its own USD file
        if len(self.viewer_usds) > 0:
            all_particles = self.state_0.particle_q.numpy()
            all_velocities = self.state_0.particle_qd.numpy()
            
            for viewer_info in self.viewer_usds:
                viewer_usd = viewer_info["viewer"]
                mesh_model = viewer_info["model"]
                mesh_info = viewer_info["mesh_info"]
                
                # Extract particles for this mesh
                v_start = mesh_info["vertex_start"]
                v_count = mesh_info["vertex_count"]
                v_end = v_start + v_count
                
                # Create a state for this mesh's model
                mesh_state = mesh_model.state()
                mesh_particles = all_particles[v_start:v_end]
                mesh_velocities = all_velocities[v_start:v_end]
                
                mesh_state.particle_q.assign(mesh_particles)
                mesh_state.particle_qd.assign(mesh_velocities)
                
                viewer_usd.begin_frame(self.sim_time)
                viewer_usd.log_state(mesh_state)
                viewer_usd.end_frame()

        self.frame += 1
        self.sim_time += self.frame_dt

        if self.frame > self.num_frames:
            for viewer_info in self.viewer_usds:
                viewer_info["viewer"].close()

            self.viewer.close()


        # self.verts_for_vis = self.state_0.particle_q.numpy()
        # # print(self.verts_for_vis)
        # self.ps_vis_mesh.update_vertex_positions(self.verts_for_vis)
        # ps.frame_tick()

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
    # wp.clear_kernel_cache()
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=2200)
    parser.set_defaults(viewer="null")

    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
    )


    viewer, args = newton.examples.init(parser)

    # Create example and run

    example = Example(args.input_path, args.output_path, newton.viewer.ViewerGL(), args)
    # example = Example(args.input_path, args.output_path, newton.viewer.ViewerNull(), args)

    newton.examples.run(example, args)