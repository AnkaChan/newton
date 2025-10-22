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
# Newton script to run simulations on USD files
#
# Simulates the stage of the input USD file as described by the USD Physics
# definitions.
#
###########################################################################

import inspect
import os
from enum import Enum
from os.path import join
from pathlib import Path
from typing import ClassVar

import numpy as np
import tqdm
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics

import newton
import newton.examples
from newton import ParticleFlags
from newton._src.utils.import_usd import parse_usd
from newton._src.utils.schema_resolver import (
    Attribute,
    PrimType,
    SchemaResolver,
    SchemaResolverNewton,
    _ResolverManager,
)
from newton._src.utils.update_usd import UpdateUsd


def writeObj(vs, vns, vts, fs, outFile, withMtl=False, textureFile=None, convertToMM=False, vIdAdd1=True):
    # write new
    with open(outFile, "w+") as f:
        fp = Path(outFile)
        outMtlFile = join(str(fp.parent), fp.stem + ".mtl")
        if withMtl:
            f.write("mtllib ./" + fp.stem + ".mtl\n")
            with open(outMtlFile, "w") as fMtl:
                mtlStr = """newmtl material_0
    Ka 0.200000 0.200000 0.200000
    Kd 1.000000 1.000000 1.000000
    Ks 1.000000 1.000000 1.000000
    Tr 1.000000
    illum 2
    Ns 0.000000
    map_Kd """
                assert textureFile is not None
                mtlStr += textureFile
                fMtl.write(mtlStr)

        for i, v in enumerate(vs):
            if convertToMM:
                v[0] = 1000 * v[0]
                v[1] = 1000 * v[1]
                v[2] = 1000 * v[2]
            if len(v) == 3:
                f.write(f"v {v[0]:f} {v[1]:f} {v[2]:f}\n")
            elif len(v) == 6:
                f.write(f"v {v[0]:f} {v[1]:f} {v[2]:f} {v[3]:f} {v[4]:f} {v[5]:f}\n")
        if vns:
            for i, v in enumerate(vns):
                vn = vns[i]
                f.write(f"vn {vn[0]:f} {vn[1]:f} {vn[2]:f}\n")

        if vns:
            for vt in vts:
                f.write(f"vt {vt[0]:f} {vt[1]:f}\n")

        if withMtl:
            f.write("usemtl material_0\n")
        for iF in range(len(fs)):
            # if facesToPreserve is not None and iF not in facesToPreserve:
            #     continue
            f.write("f")
            if vIdAdd1:
                for fis in fs[iF]:
                    if isinstance(fis, list):
                        f.write(" {}".format("/".join([str(fi + 1) for fi in fis])))
                    else:
                        f.write(f" {fis + 1}")

            else:
                for fis in fs[iF]:
                    if isinstance(fis, list):
                        f.write(" {}".format("/".join([str(fi) for fi in fis])))
                    else:
                        f.write(f" {fis}")
            f.write("\n")
        f.close()


run_cfgs = {
    "test_040": {
        "camera_cfg": {
            "pos": wp.vec3(11.52, 4.85, 1.68),  # Position
            "pitch": -0.8,  # Pitch in degrees
            "yaw": 72.4,
        },
        "cloth_cfg": {
            "path": "/World/RopeNetACollisionRest_01/geo/ropeNetAcollision",
            #   elasticity
            "tri_ke": 1e2,
            "tri_ka": 1e2,
            "tri_kd": 1.5e-6,
            "bending_ke": 1e-3,
            "bending_kd": 1e-3,
            "particle_radius": 0.02,
            # "fixed_particles" : [23100, 22959]
        },
        "viewer_type": "usd",
        # "viewer_type": "gl",
    },
    # "scene1": {
    #     "camera_cfg": {
    #         "pos": wp.vec3(19.82, 11.22, 1.41),  # Position
    #         "pitch": -3.2,  # Pitch in degrees
    #         "yaw": 97.6,
    #     },
    #     "initial_time": 0.0,
    #     "preroll_frames": 1500,
    #     "preroll_zero_velocity_ratio": 0.1,
    #     "load_preroll_state": False,
    #     # "load_preroll_state": True,
    #     "cloth_cfg": {
    #         "path": "/World/ClothModuleC_01/geo/clothModuleCbCollisionGeo05K",
    #         # "path": "/World/ClothModuleC5kCollisionRest_01/geo/clothModuleCbCollisionRestGeo05K",
    #         "rest_path": "/World/ClothModuleC5kCollisionRest_01/geo/clothModuleCbCollisionRestGeo05K",
    #         #   elasticity
    #         "tri_ke": 5e2,
    #         "tri_ka": 5e2,
    #         "tri_kd": 1e-7,
    #         "bending_ke": 1e-2,
    #         "bending_kd": 1e-8,
    #         "particle_radius": 0.04,
    #         "additional_translation": [0,0,-0.05]
    #         # "fixed_particles" : [23100, 22959]
    #     },
    #     "additional_collider": [
    #
    #     ],
    #     "save_usd": True,
    #     "save_rest_and_init_state": True,
    #     "fixed_points_scheme": "top",
    #     # "viewer_type": "gl",
    # },
    "sceneB": {
        "camera_cfg": {
            "pos": wp.vec3(19.82, 11.22, 1.41),  # Position
            "pitch": -3.2,  # Pitch in degrees
            "yaw": 97.6,
        },
        "initial_time": 18.0,
        "preroll_frames": 500,
        "preroll_zero_velocity_ratio": 0.1,
        # "load_preroll_state": False,
        "load_preroll_state": True,
        "cloth_cfg": {
            "path": "/World/ClothModuleC_01/geo/clothModuleCbCollisionGeo1p12",
            # "path": "/World/ClothModuleC5kCollisionRest_01/geo/clothModuleCbCollisionRestGeo05K",
            "rest_path": "/World/ClothModuleC_01_Rest/geo/clothModuleCbCollisionRestGeo1p12",
            #   elasticity
            "tri_ke": 5e2,
            "tri_ka": 5e2,
            "tri_kd": 1e-7,
            "bending_ke": 1e-2,
            "bending_kd": 1e-7,
            "particle_radius": 0.03,
            "density": 1.0,
            "additional_translation": [0, 0, -0.05],
            # "fixed_particles" : [23100, 22959]
        },
        "additional_collider": [],
        "save_usd": True,
        "save_rest_and_init_state": True,
        "fixed_points_scheme": {
           "name": "top",
            "threshold": 0.1,
        },
        "substeps": 20,
        "iterations": 20,
        "collision_detection_interval": 10,
        "self_contact_rest_filter_radius": 0.02,
        "self_contact_radius": 0.005,
        "self_contact_margin": 0.025,
        "handle_self_contact": True,
        "soft_contact_ke": 1e3,
        "soft_contact_kd": 1e-3,
        "soft_contact_mu": 0.
    },
    "sceneA": {
        "camera_cfg": {
            "pos": wp.vec3(19.82, 11.22, 1.41),  # Position
            "pitch": -3.2,  # Pitch in degrees
            "yaw": 97.6,
        },
        "initial_time": 14.0,
        "preroll_frames": 500,
        "self_collision_off_frame" : 1000,
        "preroll_zero_velocity_ratio": 0.1,
        # "load_preroll_state": False,
        "load_preroll_state": True,
        "cloth_cfg": {
            # "rest_path": "/World/RopeNetA1p12CollisionRest_01/geo/ropeNetACollisionRest1p12",
            # "path": "/World/RopeNetA_01/geo/ropeNetACollision1p12",
            "rest_path": "/World/RopeNetA1p50CollisionRest_01/geo/ropeNetACollisionRest1p50",
            "path": "/World/RopeNetA_01/geo/ropeNetACollision1p50",
            #   elasticity
            "density": 1,
            "tri_ke": 5e2,
            "tri_ka": 5e2,
            "tri_kd": 1e-8,
            "bending_ke": 1e-2,
            "bending_kd": 1e-8,
            "particle_radius": 0.03,
            # "fixed_particles" : [23100, 22959]
        },
        "additional_collider": [
            # "/World/TerrainCollision_01/geo/collision/staircol01"
        ],
        "save_usd": True,
        "fixed_points_scheme": {
            "name": "box",
            "boxes": [
                [a - 0.05 for a in [19.57, 17.09, 2.60]] + [a + 0.05 for a in [19.57, 17.09, 2.60]],
                [a - 0.05 for a in [20.19, 17.49, 2.05]] + [a + 0.05 for a in [20.19, 17.49, 2.05]],
            ],
        },
        # "viewer_type": "gl",
        "save_rest_and_init_state": True,
        "substeps": 20,
        "iterations": 20,
        "collision_detection_interval": 10,
        "self_contact_rest_filter_radius": 0.02,
        "self_contact_radius": 0.005,
        "self_contact_margin": 0.02,
        "handle_self_contact": True,
        "soft_contact_ke": 1e3,
        "soft_contact_kd": 1e-3,
        "soft_contact_mu": 0.

    },
    "sceneC": {
        "camera_cfg": {
            "pos": wp.vec3(19.82, 11.22, 1.41),  # Position
            "pitch": -3.2,  # Pitch in degrees
            "yaw": 97.6,
        },
        "initial_time": 22.0,
        "preroll_frames": 500,
        # "self_collision_off_frame" : 1000,
        "preroll_zero_velocity_ratio": 0.1,
        # "load_preroll_state": False,
        "load_preroll_state": True,
        "cloth_cfg": {
            "path": "/World/ClothModuleD_01/geo/clothModuleDCollisionGeo1p50",
            #   elasticity
            "density": 1,
            "tri_ke": 5e2,
            "tri_ka": 5e2,
            "tri_kd": 1e-8,
            "bending_ke": 1e-2,
            "bending_kd": 1e-8,
            "particle_radius": 0.035,
            # "fixed_particles" : [23100, 22959]
        },
        "additional_collider": [
            # "/World/TerrainCollision_01/geo/collision/staircol01"
        ],
        "save_usd": True,
         "fixed_points_scheme": {
           "name": "top",
            "threshold": 0.3,
        },
        # "viewer_type": "gl",
        "save_rest_and_init_state": True,
        "substeps": 20,
        "iterations": 20,
        "collision_detection_interval": 10,
        "self_contact_rest_filter_radius": 0.02,
        "self_contact_radius": 0.005,
        "self_contact_margin": 0.02,
        "handle_self_contact": True,
        "soft_contact_ke": 3e3,
        "soft_contact_kd": 1e-3,
        "soft_contact_mu": 0.

    },
}

# D:\Data\GTC2025DC_Demo\Inputs\SceneB\20251017_to_sim_inSimClothB_01_physics.usd -n 1800 -i vbd
# D:\Data\GTC2025DC_Demo\Inputs\SceneB\20251017_to_sim_inSimClothB_01_physics.usd -n 1800 -i vbd
# D:\Data\GTC2025DC_Demo\Inputs\SceneB\1021\20251021_to_sim_tdSimClothB_01_physics.usd
run_cfg = run_cfgs["sceneB"]
"""
Comments:
- [x] the cloth look too light:
    1, I increased the density of the cloth
    2, Reduce the time step
- [] the cloth keeps going up after the robot strucks on it
    1, added a bit extra bending damping (not working)
    2, reduce the density for a bit  (not working)
    3, reduce collision stiffness (looks like works but cause penetration)
    4, more iterations (will through it even higher)
    5, even higher damping on bending? (not significant)
    6. making it even heavier?
        a 5x too much
        b 2x
    7, damp collision in both ways
"""

#
# D:\Data\GTC2025DC_Demo\Inputs\SceneA\1021\20251021_to_sim_tdSimClothA_01_physics.usd
# run_cfg = run_cfgs["sceneA"]


# ClothA: omniverse://creative3d.ov.nvidia.com/Projects/CreativeRealtime3D/Projects/GTC_DC2025_DisneyDroidDemo/shot/tdSim/tdSimClothA//pub/sim/handoff/20251021_to_sim_tdSimClothA_01.usd
# ClothB: omniverse://creative3d.ov.nvidia.com/Projects/CreativeRealtime3D/Projects/GTC_DC2025_DisneyDroidDemo/shot/tdSim/tdSimClothB//pub/sim/handoff/20251021_to_sim_tdSimClothB_01.usd
# ClothC: omniverse://creative3d.ov.nvidia.com/Projects/CreativeRealtime3D/Projects/GTC_DC2025_DisneyDroidDemo/shot/tdSim/tdSimClothC//pub/sim/handoff/20251021_to_sim_tdSimClothC_02.usd (edited)
#
#
# D:\Data\GTC2025DC_Demo\Inputs\SceneC\1021\20251021_to_sim_tdSimClothC_02.usd
run_cfg = run_cfgs["sceneC"]


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
    vmin = np.min(vals)
    idx_top = np.where(vals <= vmin + thresh)[0]
    return idx_top


def parse_xform(prim, time=Usd.TimeCode.Default()):
    from pxr import UsdGeom

    xform = UsdGeom.Xform(prim)
    mat = np.array(xform.ComputeLocalToWorldTransform(time), dtype=np.float32)
    rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
    pos = mat[3, :3]
    return wp.transform(pos, rot)
    # cache = UsdGeom.XformCache(time)
    # world_xform = cache.GetLocalToWorldTransform(prim)


def _build_solver_args_from_resolver(
    resolver_mgr, prim, prim_type, solver_cls, defaults: dict[str, object] | None = None
):
    defaults = defaults or {}
    sig = inspect.signature(solver_cls.__init__)
    solver_args = {}
    for name, param in sig.parameters.items():
        # skip self, model, and var positional/keyword args: *args/**kwargs
        if name in ("self", "model") or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        value = resolver_mgr.get_value(prim, prim_type, name, defaults.get(name))
        if value is not None:
            solver_args[name] = value
    return solver_args


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"
    MJWARP = "mjwarp"
    COUPLED_MPM = "cmpm"

    def __str__(self):
        return self.value


class SchemaResolverSimUsd(SchemaResolver):
    name: ClassVar[str] = "sim_usd"
    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {
            # model attributes
            "joint_attach_kd": [Attribute("newton:joint_attach_kd", 2718.0)],
            "joint_attach_ke": [Attribute("newton:joint_attach_ke", 2718.0)],
            "soft_contact_ke": [Attribute("newton:soft_contact_ke", 2.0e2)],
            "soft_contact_kd": [Attribute("newton:soft_contact_kd", 1.0e-2)],
            # solver attributes
            "fps": [Attribute("newton:fps", 60)],
            "sim_substeps": [Attribute("newton:substeps", run_cfg["substeps"])],
            "integrator_type": [Attribute("newton:integrator", "xpbd")],
            "integrator_iterations": [Attribute("newton:integrator_iterations", run_cfg["iterations"])],
            "collide_on_substeps": [Attribute("newton:collide_on_substeps", True)],
        },
        PrimType.BODY: {
            "kinematic_collider": [Attribute("physics:kinematicEnabled", False)],
        },
        PrimType.SHAPE: {
            "kinematic_collider": [Attribute("physics:kinematicEnabled", False)],
        },
    }


class SchemaResolverEuler(SchemaResolver):
    name: ClassVar[str] = "euler"
    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {
            "angular_damping": [Attribute("newton:euler:angular_damping", 2718.0)],
            "friction_smoothing": [Attribute("newton:euler:friction_smoothing", 2718.0)],
        },
    }


class SchemaResolverVBD(SchemaResolver):
    name: ClassVar[str] = "vbd"
    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {
            "friction_epsilon": [Attribute("newton:vbd:friction_epsilon", 2718.0)],
            "handle_self_contact": [Attribute("newton:vbd:handle_self_contact", run_cfg["handle_self_contact"])],
            "self_contact_radius": [Attribute("newton:vbd:self_contact_radius", run_cfg["self_contact_radius"])],
            "self_contact_margin": [Attribute("newton:vbd:self_contact_margin", run_cfg["self_contact_margin"])],
            "self_contact_rest_filter_radius": [Attribute("newton:vbd:self_contact_rest_filter_radius", run_cfg["self_contact_rest_filter_radius"])],
            "collision_detection_interval": [
                Attribute("newton:vbd:collision_detection_interval", run_cfg["collision_detection_interval"])
            ],
            "integrate_with_external_rigid_solver": [
                Attribute("newton:vbd:integrate_with_external_rigid_solver", True)
            ],
        },
    }


class SchemaResolverXPBD(SchemaResolver):
    name: ClassVar[str] = "xpbd"
    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {
            "soft_body_relaxation": [Attribute("newton:xpbd:soft_body_relaxation", 0.9)],
            "soft_contact_relaxation": [Attribute("newton:xpbd:soft_contact_relaxation", 0.9)],
            "joint_linear_relaxation": [Attribute("newton:xpbd:joint_linear_relaxation", 0.7)],
            "joint_angular_relaxation": [Attribute("newton:xpbd:joint_angular_relaxation", 0.4)],
            "rigid_contact_relaxation": [Attribute("newton:xpbd:rigid_contact_relaxation", 0.8)],
            "rigid_contact_con_weighting": [Attribute("newton:xpbd:rigid_contact_con_weighting", True)],
            "angular_damping": [Attribute("newton:xpbd:angular_damping", 0.0)],
            "enable_restitution": [Attribute("newton:xpbd:enable_restitution", False)],
        },
    }


class SchemaResolverMJWarp(SchemaResolver):
    name: ClassVar[str] = "mjwarp"
    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {
            "use_mujoco_cpu": [Attribute("newton:mjwarp:use_mujoco_cpu", False)],
            "solver": [Attribute("newton:mjwarp:solver", "newton")],
            "integrator": [Attribute("newton:mjwarp:integrator", "euler")],
            "iterations": [Attribute("newton:mjwarp:iterations", 100)],
            "ls_iterations": [Attribute("newton:mjwarp:ls_iterations", 5)],
            "save_to_mjcf": [Attribute("newton:mjwarp:save_to_mjcf", "sim_usd_mjcf.xml")],
            "contact_stiffness_time_const": [Attribute("newton:mjwarp:contact_stiffness_time_const", 0.02)],
            "ncon_per_env": [Attribute("newton:mjwarp:ncon_per_env", 150)],
            "njmax": [Attribute("newton:mjwarp:njmax", 16)],
        },
    }


class SchemaResolverCoupledMPM(SchemaResolver):
    name: ClassVar[str] = "cmpm"
    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {},
    }


class SchemaResolverMPM(SchemaResolver):
    name: ClassVar[str] = "mpm"
    mapping: ClassVar[dict[PrimType, dict[str, list[Attribute]]]] = {
        PrimType.SCENE: {
            "voxel_size": [Attribute("newton:mpm:voxel_size", False)],
            "grid_type": [Attribute("newton:mpm:grid_type", "sparse")],
            "max_iterations": [Attribute("newton:mpm:max_iterations", 250)],
        },
    }


class CoupledMPMIntegrator(newton.solvers.SolverBase):
    """Integrator for coupled MPM and rigid body solvers."""

    def __init__(self, model: newton.Model, **kwargs):
        super().__init__(model)

        rigid_solver_kwargs = {}

        mpm_options = newton.solvers.SolverImplicitMPM.Options()
        for key, value in kwargs.items():
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, value)
            else:
                rigid_solver_kwargs[key] = value

        mpm_model = self._build_mpm_model(model, mpm_options)

        self.mpm_solver = newton.solvers.SolverImplicitMPM(mpm_model, mpm_options)

        self.mpm_state_0 = mpm_model.model.state()
        self.mpm_state_1 = mpm_model.model.state()

        self.mpm_solver.enrich_state(self.mpm_state_0)
        self.mpm_solver.enrich_state(self.mpm_state_1)

        self.rigid_solver = newton.solvers.SolverXPBD(model, **rigid_solver_kwargs)

        self._initialized = False

        self.particle_render_colors = wp.full(
            mpm_model.model.particle_count, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=mpm_model.model.device
        )

    def step(
        self,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):
        if not self._initialized:
            # not required for MuJoCo, but required for other solvers
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state_0)
            self.sand_body_forces = wp.zeros_like(state_0.body_f)
            self._update_collider_meshes(state_0, state_0, dt)

            self.collider_impulses = None
            self.collider_impulse_pos = None
            self.collider_impulse_ids = None
            # self.collect_collider_impulses(self.mpm_state_0)
            self._initialized = True

        # wp.launch(
        #     compute_body_forces,
        #     dim=self.collider_impulse_ids.shape[0],
        #     inputs=[
        #         self.frame_dt,
        #         self.collider_impulse_ids,
        #         self.collider_impulses,
        #         self.collider_impulse_pos,
        #         self.collider_body_id,
        #         self.state_0.body_q,
        #         self.model.body_com,
        #         self.state_0.body_f,
        #     ],
        # )
        self.sand_body_forces.assign(state_0.body_f)
        self.rigid_solver.step(state_0, state_1, control, contacts, dt)

        self._update_collider_meshes(state_0, state_1, dt)

        self.mpm_solver.step(self.mpm_state_0, self.mpm_state_1, None, None, dt)
        self.mpm_solver.project_outside(self.mpm_state_1, self.mpm_state_1, dt)
        # self.collect_collider_impulses(self.mpm_state_1)

        self.mpm_state_0, self.mpm_state_1 = self.mpm_state_1, self.mpm_state_0

    def _build_mpm_model(self, model, mpm_options):
        sand_builder = newton.ModelBuilder()
        self._add_particles(sand_builder)
        sand_model = sand_builder.finalize()

        mpm_model = newton.solvers.SolverImplicitMPM.Model(sand_model, mpm_options)

        self._setup_mpm_collider(model, mpm_model)

        return mpm_model

    def _setup_mpm_collider(self, model: newton.Model, mpm_model: newton.solvers.SolverImplicitMPM.Model):
        collider_body_shapes = {}

        shape_body = model.shape_body.numpy()

        # build body_id -> shapes map
        for k in range(model.shape_count):
            src = model.shape_source[k]
            if src is not None:
                if shape_body[k] not in collider_body_shapes:
                    collider_body_shapes[shape_body[k]] = []

                collider_body_shapes[shape_body[k]].append(k)

        # merge meshes for each body
        collider_meshes = []
        collider_ids = []
        collider_shape_ids = []
        collider_body_id = []

        for body, shapes in collider_body_shapes.items():
            collider_points, collider_indices, collider_v_shape_ids = self._merge_meshes(
                [self.model.shape_source[m].vertices for m in shapes],
                [self.model.shape_source[m].indices for m in shapes],
                [self.model.shape_scale.numpy()[m] for m in shapes],
                shapes,
            )

            collider_mesh = wp.Mesh(wp.clone(collider_points), collider_indices, wp.zeros_like(collider_points))
            nv = collider_points.shape[0]

            collider_meshes.append(collider_mesh)
            collider_ids.append(
                np.hstack(
                    (
                        np.full(nv, len(collider_ids)).reshape(-1, 1),
                        np.arange(nv).reshape(-1, 1),
                    )
                )
            )
            collider_body_id.append(body)
            collider_shape_ids.append(collider_v_shape_ids)

        self.collider_meshes = collider_meshes
        self.collider_mesh_ids = wp.array([mesh.id for mesh in collider_meshes], dtype=wp.uint64)
        self.collider_ids = wp.array(np.vstack(collider_ids), dtype=wp.vec2i)
        self.collider_shape_ids = wp.array(np.concatenate(collider_shape_ids), dtype=int)
        self.collider_body_id = wp.array(collider_body_id, dtype=int)
        self.collider_rest_points = wp.array(
            np.vstack([mesh.points.numpy() for mesh in collider_meshes]), dtype=wp.vec3
        )

        body_masses = self.model.body_mass.numpy()
        mpm_model.setup_collider(
            colliders=self.collider_meshes,
            # collider_masses=[body_masses[body_id] if body_id >= 0 else 1.0e15 for body_id in collider_body_id],
            collider_friction=[0.5 for _ in collider_body_id],
            collider_adhesion=[0.0 for _ in collider_body_id],
        )

    def _add_particles(self, sand_builder: newton.ModelBuilder):
        # ------------------------------------------
        # Add sand bed (2m x 2m x 0.5m) above ground
        # ------------------------------------------
        voxel_size = 0.05  # 5 cm
        particles_per_cell = 3.0
        density = 2500.0

        bed_lo = np.array([3.0, -1.0, 0.0])
        bed_hi = np.array([4.0, 1.0, 0.5])
        bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int)

        # spawn particles on a jittered grid
        Nx, Ny, Nz = bed_res
        px = np.linspace(bed_lo[0], bed_hi[0], Nx + 1)
        py = np.linspace(bed_lo[1], bed_hi[1], Ny + 1)
        pz = np.linspace(bed_lo[2], bed_hi[2], Nz + 1)
        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_volume) * density)

        rng = np.random.default_rng()
        points += 2.0 * radius * (rng.random(points.shape) - 0.5)
        vel = np.zeros_like(points)

        sand_builder.particle_q = points
        sand_builder.particle_qd = vel
        sand_builder.particle_mass = np.full(points.shape[0], mass)
        sand_builder.particle_radius = np.full(points.shape[0], radius)
        sand_builder.particle_flags = np.ones(points.shape[0], dtype=int)

    @staticmethod
    def _merge_meshes(
        points: list[np.array],
        indices: list[np.array],
        scales: list[np.array],
        shape_ids: list[int],
    ):
        pt_count = np.array([len(pts) for pts in points])
        offsets = np.cumsum(pt_count) - pt_count

        mesh_id = np.repeat(np.arange(len(points), dtype=int), repeats=pt_count)

        merged_points = np.vstack([pts * scale for pts, scale in zip(points, scales, strict=False)])

        merged_indices = np.concatenate([idx + offsets[k] for k, idx in enumerate(indices)])
        # merged_shape_ids = np.concatenate([np.full(len(idx), shape_ids[k]) for k, idx in enumerate(indices)])

        return (
            wp.array(merged_points, dtype=wp.vec3),
            wp.array(merged_indices, dtype=int),
            np.array(shape_ids)[mesh_id],
        )

    def collect_collider_impulses(self, mpm_state):
        if self.collider_impulses is None:
            self.collider_impulses, self.collider_impulse_pos, self.collider_impulse_ids = (
                self.mpm_solver.collect_collider_impulses(mpm_state)
            )
        else:
            collider_impulses, collider_impulse_pos, collider_impulse_ids = self.mpm_solver.collect_collider_impulses(
                mpm_state
            )
            self.collider_impulses.assign(collider_impulses)
            self.collider_impulse_pos.assign(collider_impulse_pos)
            self.collider_impulse_ids.assign(collider_impulse_ids)

    def _update_collider_meshes(self, state_cur, state_next, dt):
        wp.launch(
            update_collider_coms,
            dim=self.collider_body_id.shape[0],
            inputs=[
                self.collider_body_id,
                state_next.body_q,
                # self.ref_q,
                self.model.body_inv_inertia,
                self.model.body_com,
                self.mpm_solver.mpm_model.collider_inv_inertia,
                self.mpm_solver.mpm_model.collider_coms,
            ],
        )
        wp.launch(
            update_collider_meshes,
            dim=self.collider_rest_points.shape[0],
            inputs=[
                self.collider_ids,
                self.collider_mesh_ids,
                self.collider_rest_points,
                self.collider_shape_ids,
                self.model.shape_transform,
                self.model.shape_body,
                state_cur.body_q,
                state_next.body_q,
                state_next.body_qd,
                dt,
                self.sand_body_forces,
                self.model.body_inv_inertia,
                self.model.body_com,
                self.model.body_inv_mass,
            ],
        )

        for mesh in self.collider_meshes:
            mesh.refit()


@wp.kernel
def update_collider_meshes(
    collider_id: wp.array(dtype=wp.vec2i),
    collider_meshes: wp.array(dtype=wp.uint64),
    src_points: wp.array(dtype=wp.vec3),
    src_shape: wp.array(dtype=int),
    shape_transforms: wp.array(dtype=wp.transform),
    shape_body_id: wp.array(dtype=int),
    body_q_cur: wp.array(dtype=wp.transform),
    body_q_next: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    dt: float,
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_coms: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
):
    v = wp.tid()

    cid = collider_id[v][0]
    cv = collider_id[v][1]

    res_mesh = collider_meshes[cid]
    res = wp.mesh_get(res_mesh)

    shape_id = src_shape[v]
    p = wp.transform_point(shape_transforms[shape_id], src_points[v])

    body_id = shape_body_id[shape_id]

    # Remove previously applied force
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q_next[body_id])

    dw = dt * body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f))
    delta_v += wp.quat_rotate(r, wp.cross(dw, p - body_coms[body_id]))

    # (body_inv_mass[body_id] > 0.0)

    # q_new, qd_new = integrate_rigid_body(
    #     q,
    #     body_f.dtype(0.0),
    #     -f,
    #     body_coms[body_id],
    #     wp.mat33(0.0),
    #     body_inv_mass[body_id],
    #     body_inv_inertia[body_id],
    #     wp.vec3(0.0),
    #     0.0,
    #     dt,
    # )

    next_p = wp.transform_point(body_q_next[body_id], p)

    vel = wp.spatial_top(body_qd[body_id]) + wp.cross(
        wp.spatial_bottom(body_qd[body_id]), wp.quat_rotate(r, p - body_coms[body_id])
    )
    res.velocities[cv] = vel - delta_v
    res.points[cv] = next_p

    # cur_p = wp.transform_point(body_q_cur[body_id], p)  # res.points[cv] + dt * res.velocities[cv]
    # res.velocities[cv] = (next_p - cur_p) / dt - delta_v * IR
    # res.points[cv] = cur_p


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        f_world = collider_impulses[i] / dt

        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def update_collider_coms(
    body_id: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_coms: wp.array(dtype=wp.vec3),
    collider_inv_inertia: wp.array(dtype=wp.mat33),
    collider_coms: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    body_index = body_id[i]

    X_wb = body_q[body_index]
    X_com = body_coms[body_index]

    collider_coms[i] = wp.transform_point(X_wb, X_com)
    R = wp.quat_to_matrix(wp.transform_get_rotation(X_wb))
    collider_inv_inertia[i] = R @ body_inv_inertia[body_index] @ wp.transpose(R)


@wp.kernel
def extract_rotation_and_scales(
    transform: wp.array(dtype=wp.mat33),
    rotation: wp.array(dtype=wp.vec4h),
    scales: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    A = transform[i]

    Q = wp.mat33()
    R = wp.mat33()
    wp.qr3(A, Q, R)

    q = wp.quat_from_matrix(Q)

    rotation[i] = wp.vec4h(wp.vec4(q[0], q[1], q[2], q[3]))
    scales[i] = wp.vec3(wp.length(R[0]), wp.length(R[1]), wp.length(R[2]))


def extract_particle_rotation_and_scales(state: newton.State):
    particle_transform_rotation = wp.empty(state.particle_count, dtype=wp.vec4h)
    particle_transform_scales = wp.empty(state.particle_count, dtype=wp.vec3)

    wp.launch(
        extract_rotation_and_scales,
        dim=state.particle_count,
        inputs=[
            state.particle_transform,
            particle_transform_rotation,
            particle_transform_scales,
        ],
    )

    return particle_transform_rotation, particle_transform_scales


class Simulator:
    # TODO: make logic for the case when attributes can be specified in multiple places
    #       eg: fps specified on the stage or physxScene:timeStepsPerSecond for substeps

    MODEL_ATTRIBUTES = {
        "newton:joint_attach_kd": "joint_attach_kd",
        "newton:joint_attach_ke": "joint_attach_ke",
        "newton:soft_contact_kd": "soft_contact_kd",
        "newton:soft_contact_ke": "soft_contact_ke",
    }
    SOLVER_ATTRIBUTES = {
        "newton:collide_on_substeps": "collide_on_substeps",
        "newton:fps": "fps",
        "newton:integrator": "integrator_type",
        "newton:integrator_iterations": "integrator_iterations",
        "newton:substeps": "substeps",
    }
    INTEGRATOR_ATTRIBUTES = {
        IntegratorType.EULER: {
            "newton:euler:angular_damping": "angular_damping",
            "newton:euler:friction_smoothing": "friction_smoothing",
        },
        IntegratorType.VBD: {"newton:vbd:friction_epsilon": "friction_epsilon"},
        IntegratorType.XPBD: {
            "newton:xpbd:soft_body_relaxation": "soft_body_relaxation",
            "newton:xpbd:soft_contact_relaxation": "soft_contact_relaxation",
            "newton:xpbd:joint_linear_relaxation": "joint_linear_relaxation",
            "newton:xpbd:joint_angular_relaxation": "joint_angular_relaxation",
            "newton:xpbd:rigid_contact_relaxation": "rigid_contact_relaxation",
            "newton:xpbd:rigid_contact_con_weighting": "rigid_contact_con_weighting",
            "newton:xpbd:angular_damping": "angular_damping",
            "newton:xpbd:enable_restitution": "enable_restitution",
        },
        IntegratorType.MJWARP: {
            "newton:mjwarp:use_mujoco_cpu": "use_mujoco_cpu",
            "newton:mjwarp:solver": "solver",
            "newton:mjwarp:integrator": "integrator",
            "newton:mjwarp:iterations": "iterations",
            "newton:mjwarp:ls_iterations": "ls_iterations",
            "newton:mjwarp:save_to_mjcf": "save_to_mjcf",
            "newton:mjwarp:contact_stiffness_time_const": "contact_stiffness_time_const",
        },
    }
    MODEL_ATTRIBUTES_KEYS = MODEL_ATTRIBUTES.keys()
    SOLVER_ATTRIBUTES_KEYS = SOLVER_ATTRIBUTES.keys()

    def __init__(self, input_path, output_path, integrator: IntegratorType | None = None):
        def create_stage_from_path(input_path) -> Usd.Stage:
            stage = Usd.Stage.Open(input_path, Usd.Stage.LoadAll)
            flattened = stage.Flatten()
            out_stage = Usd.Stage.Open(flattened.identifier)
            return out_stage

        self.output_folder = os.path.dirname(output_path)

        self.sim_time = 0.0
        self.profiler = {}

        self.in_stage = create_stage_from_path(input_path)

        builder = newton.ModelBuilder()
        builder.up_axis = newton.Axis.Z
        builder.default_shape_cfg.density = 1.0
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.mu = 0.1
        results = parse_usd(
            builder,
            self.in_stage,
            invert_rotations=True,
            collapse_fixed_joints=True,
        )
        # See what's actually in the stage
        print("All Mesh prims in the stage:")
        for prim in self.in_stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                print(f"  {prim.GetPath()}")

        if run_cfg["cloth_cfg"].get("path", None) is not None:
            rest_shape_path = run_cfg["cloth_cfg"].get("rest_path", None)
            has_rest_shape = rest_shape_path is not None
            rest_shape_path = rest_shape_path if has_rest_shape else run_cfg["cloth_cfg"]["path"]

            usd_geom = UsdGeom.Mesh(self.in_stage.GetPrimAtPath(rest_shape_path))
            mesh_points = np.array(usd_geom.GetPointsAttr().Get())
            mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

            if run_cfg.get("save_rest_and_init_state", False):
                writeObj(
                    mesh_points, None, None, mesh_indices.reshape(-1, 3), join(self.output_folder, "rest_state.obj")
                )

            vertices = [wp.vec3(v) for v in mesh_points]
            transform = parse_xform(usd_geom)
            # Extract position and rotation
            position = wp.transform_get_translation(transform)  # wp.vec3
            rotation = wp.transform_get_rotation(transform)

            if has_rest_shape:
                usd_geom_initial_shape = UsdGeom.Mesh(self.in_stage.GetPrimAtPath(run_cfg["cloth_cfg"]["path"]))
                mesh_points_initial_org = np.array(usd_geom_initial_shape.GetPointsAttr().Get())
                transform_initial_shape = parse_xform(usd_geom_initial_shape)

                # Apply transform_initial_shape to mesh_points_initial
                mesh_points_initial = np.array(
                    [wp.transform_point(transform_initial_shape, wp.vec3(*p)) for p in mesh_points_initial_org]
                )
            else:
                mesh_points_initial_org = mesh_points
                mesh_points_initial = np.array(
                    [wp.transform_point(transform, wp.vec3(*p)) for p in mesh_points_initial_org]
                )

            if run_cfg.get("save_rest_and_init_state", False):
                writeObj(
                    mesh_points, None, None, mesh_indices.reshape(-1, 3), join(self.output_folder, "init_state.obj")
                )

            if run_cfg["fixed_points_scheme"]["name"] == "top":
                fixed_vertices = get_top_vertices(mesh_points_initial_org, "y", thresh=run_cfg["fixed_points_scheme"]["threshold"])
            elif (
                isinstance(run_cfg.get("fixed_points_scheme"), dict)
                and run_cfg["fixed_points_scheme"].get("name") == "box"
            ):
                # Implement box selection for fixed vertices
                fixed_vertices = []
                boxes = run_cfg["fixed_points_scheme"].get("boxes", [])
                # Each box: [min_x, min_y, min_z, max_x, max_y, max_z]
                mesh_points_arr = np.array(mesh_points_initial)

                def save_obj(filename, vertices):
                    with open(filename, "w") as f:
                        for v in vertices:
                            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

                save_obj("mesh_points_arr.obj", mesh_points_arr)
                for box in boxes:
                    min_x, min_y, min_z, max_x, max_y, max_z = box
                    mask = (
                        (mesh_points_arr[:, 0] >= min_x)
                        & (mesh_points_arr[:, 0] <= max_x)
                        & (mesh_points_arr[:, 1] >= min_y)
                        & (mesh_points_arr[:, 1] <= max_y)
                        & (mesh_points_arr[:, 2] >= min_z)
                        & (mesh_points_arr[:, 2] <= max_z)
                    )
                    idx_in_box = np.where(mask)[0]
                    fixed_vertices.extend(idx_in_box.tolist())
                # Remove duplicates if boxes overlap
                fixed_vertices = np.unique(fixed_vertices)

            builder.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=rotation,
                pos=position,
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=run_cfg["cloth_cfg"]["density"],
                scale=1.0,
                tri_ke=run_cfg["cloth_cfg"]["tri_ke"],
                tri_ka=run_cfg["cloth_cfg"]["tri_ka"],
                tri_kd=run_cfg["cloth_cfg"]["tri_kd"],
                edge_ke=run_cfg["cloth_cfg"]["bending_ke"],
                edge_kd=run_cfg["cloth_cfg"]["bending_kd"],
                particle_radius=run_cfg["cloth_cfg"]["particle_radius"],
            )

            # if run_cfg["cloth_cfg"].get("fixed_particles", None) is not None:
            #     fixed_particles = run_cfg["cloth_cfg"].get("fixed_particles", None)
            #     for fixed_v_id in fixed_particles:
            #         builder.particle_flags[fixed_v_id] = builder.particle_flags[fixed_v_id] & ~ParticleFlags.ACTIVE

            for fixed_v_id in fixed_vertices:
                builder.particle_flags[fixed_v_id] = builder.particle_flags[fixed_v_id] & ~ParticleFlags.ACTIVE

        self.R = _ResolverManager([SchemaResolverSimUsd(), SchemaResolverNewton(), SchemaResolverMJWarp()])
        self.physics_prim = next(iter([prim for prim in self.in_stage.Traverse() if prim.IsA(UsdPhysics.Scene)]), None)

        self.path_body_map = results["path_body_map"]

        self._setup_solver_attributes()
        if integrator:
            self.integrator_type = integrator

        self._collect_animated_colliders(builder, results["path_body_map"], results["path_shape_map"])
        if self.integrator_type == IntegratorType.VBD:
            builder.color()

        print("average edge length: ", np.mean(builder.edge_rest_length))

        # INSERT_YOUR_CODE
        # Add additional static collider meshes from run_cfg if provided
        additional_colliders = run_cfg.get("additional_collider", [])
        for collider_path in additional_colliders:
            # Find the prim for the collider path
            prim = self.in_stage.GetPrimAtPath(collider_path)
            if prim and prim.IsValid():
                mesh = UsdGeom.Mesh(prim)
                if not mesh:
                    continue  # Not a mesh, skip

                # Get mesh points and faces
                points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
                face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
                face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())

                # Only support triangles/quads, but best effort: triangulate faces if needed.
                # If all faces are triangles, proceed, else skip (could add robust triangulation here)
                if not np.all(face_vertex_counts == 3):
                    # Simple conversion to triangles for quads (for e.g.)
                    tris = []
                    idx = 0
                    for n in face_vertex_counts:
                        if n == 3:
                            tris.append(face_vertex_indices[idx : idx + 3])
                        elif n == 4:
                            f = face_vertex_indices[idx : idx + 4]
                            tris.append([f[0], f[1], f[2]])
                            tris.append([f[0], f[2], f[3]])
                        else:
                            # skip faces that are not tri/quad
                            pass
                        idx += n
                    mesh_indices = np.array(tris, dtype=np.int32).reshape(-1, 3)
                else:
                    mesh_indices = face_vertex_indices.reshape(-1, 3)

                # Use identity for transform, could later do more (respect local Xform?)
                builder.add_shape_mesh(
                    body=-1,
                    xform=parse_xform(prim),
                    mesh=newton.Mesh(points, indices=mesh_indices),
                    scale=wp.vec3(
                        *np.array(prim.GetAttribute("xformOp:scale").Get() or [1.0, 1.0, 1.0], dtype=np.float32)
                    ),
                )

        # builder.shape_scale[0] = wp.vec3(1, 1, 1)

        self.model = builder.finalize()
        self.model.soft_contact_ke = run_cfg["soft_contact_ke"]
        self.model.soft_contact_kd = run_cfg["soft_contact_kd"]
        self.model.soft_contact_mu = run_cfg["soft_contact_mu"]
        self.builder_results = results

        self.path_body_map = self.builder_results["path_body_map"]
        self.path_shape_map = results["path_shape_map"]
        self.body_path_map = {idx: path for path, idx in self.path_body_map.items()}
        self.shape_path_map = {idx: path for path, idx in self.path_shape_map.items()}
        collapse_results = self.builder_results["collapse_results"]
        self.path_body_relative_transform = self.builder_results["path_body_relative_transform"]
        if collapse_results:
            self.body_remap = collapse_results["body_remap"]
            self.body_merged_parent = collapse_results["body_merged_parent"]
            self.body_merged_transform = collapse_results["body_merged_transform"]
        else:
            self.body_remap = None
            self.body_merged_parent = None
            self.body_merged_transform = None

        self._setup_model_attributes()
        self._setup_integrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=self.rigid_contact_margin)

        if run_cfg["cloth_cfg"].get("path", None) is not None and has_rest_shape:
            self.state_0.particle_q.assign(mesh_points_initial)
            self.state_1.particle_q.assign(mesh_points_initial)

        # NB: body_q will be modified, so initial state will be slightly altered
        if self.model.joint_count:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0, mask=None)

        self.use_cuda_graph = True  # wp.get_device().is_cuda
        # self.use_cuda_graph = False  # wp.get_device().is_cuda
        self.is_mujoco_cpu_mode = self.integrator_type == IntegratorType.MJWARP and self.R.get_value(
            self.physics_prim, PrimType.SCENE, "use_mujoco_cpu", False
        )
        if self.use_cuda_graph and not self.is_mujoco_cpu_mode:
            with wp.ScopedCapture() as capture:
                self.run_substep()
            self.graph_even_step = capture.graph
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

            with wp.ScopedCapture() as capture:
                self.run_substep()
            self.graph_odd_step = capture.graph
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

        self.usd_updater = UpdateUsd(
            stage=output_path,
            source_stage=input_path,
            path_body_relative_transform=self.path_body_relative_transform,
            path_body_map=self.path_body_map,
            builder_results=self.builder_results,
            up_axis="Z",
        )
        self.usd_updater.configure_body_mapping(
            path_body_map=self.path_body_map,
            path_body_relative_transform=self.path_body_relative_transform,
            builder_results=self.builder_results,
        )

        self.DEBUG = True
        if self.DEBUG:
            if run_cfg["save_usd"]:
                self.viewer_usd = newton.viewer.ViewerUSD(
                    output_path=input_path.replace(".usd", "_sim_v.usd"), num_frames=None
                )
                self.viewer_usd.set_model(self.model)
            else:
                self.viewer_usd = None

            self.viewer_gl = newton.viewer.ViewerGL()
            self.viewer_gl.set_model(self.model)

            if run_cfg.get("camera_cfg", None) is not None:
                self.viewer_gl.set_camera(
                    pos=run_cfg["camera_cfg"]["pos"],  # Position
                    pitch=run_cfg["camera_cfg"]["pitch"],  # Pitch in degrees
                    yaw=run_cfg["camera_cfg"]["yaw"],  # Yaw in degrees
                )

        self.sim_time = run_cfg["initial_time"]
        self.run_preroll(output_path)

    def run_preroll(self, output_path):
        preroll_frames = run_cfg.get("preroll_frames", 0)
        out_p = Path(output_path)
        preroll_state_path = str(out_p.parent / f"{out_p.stem}.preroll.npy")
        load_preroll_state = run_cfg.get("load_preroll_state", False)
        # If not explicitly provided, deduce preroll state path from the output path

        if load_preroll_state and preroll_state_path is not None:
            import numpy as np

            preroll_state = np.load(preroll_state_path, allow_pickle=True).item()
            self.state_0.particle_q.assign(preroll_state["particle_q"])
            self.state_1.particle_q.assign(preroll_state["particle_q"])
        elif preroll_frames > 0:
            import numpy as np

            state = self.state_0
            for frame in tqdm.tqdm(range(preroll_frames), desc="Preroll Frames"):
                for substep in range(self.sim_substeps):
                    self.state_0.clear_forces()
                    if self.use_cuda_graph:
                        if substep % self.sim_substeps:
                            wp.capture_launch(self.graph_even_step)
                        else:
                            wp.capture_launch(self.graph_odd_step)
                    self.state_0, self.state_1 = self.state_1, self.state_0

                    if frame < run_cfg.get("preroll_zero_velocity_ratio", 0.1) * preroll_frames:
                        self.state_0.particle_qd.zero_()
                        self.state_1.particle_qd.zero_()
                    # else:
                    #     self.state_0.particle_qd.assign(self.state_0.particle_qd * run_cfg.get("preroll_velocity_damping_ratio", 0.99))
                    #     self.state_1.particle_qd.assign(self.state_1.particle_qd * run_cfg.get("preroll_velocity_damping_ratio", 0.99))

                self.viewer_gl.begin_frame(self.sim_time)
                self.viewer_gl.log_state(self.state_0)

                self.viewer_gl.end_frame()

            state = self.state_0  # assuming self.simulate() advances self.state_0

            # Save the last frame's state
            last_frame = {
                "particle_q": np.array(state.particle_q.numpy()),
                "particle_qd": np.array(state.particle_qd.numpy()),
            }
            np.save(preroll_state_path, last_frame)

    def _setup_solver_attributes(self):
        """Apply scene attributes parsed from the stage to self."""

        self.fps = self.R.get_value(self.physics_prim, PrimType.SCENE, "fps")
        self.sim_substeps = self.R.get_value(self.physics_prim, PrimType.SCENE, "sim_substeps")
        self.integrator_type = self.R.get_value(self.physics_prim, PrimType.SCENE, "integrator_type")
        self.integrator_iterations = self.R.get_value(self.physics_prim, PrimType.SCENE, "integrator_iterations")
        self.collide_on_substeps = self.R.get_value(self.physics_prim, PrimType.SCENE, "collide_on_substeps")

        # Derived/computed attributes that depend on the above
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.integrator_type = IntegratorType(self.integrator_type)
        self.rigid_contact_margin = self.R.get_value(self.physics_prim, PrimType.SCENE, "contact_margin", 0.1)

    def _setup_model_attributes(self):
        """Apply scene attributes parsed from the stage to the model."""

        # Defaults
        # TODO: set self.model.ground from the resolver manager
        self.model.ground = False
        self.model.joint_attach_kd = self.R.get_value(self.physics_prim, PrimType.SCENE, "joint_attach_kd")
        self.model.joint_attach_ke = self.R.get_value(self.physics_prim, PrimType.SCENE, "joint_attach_ke")
        self.model.soft_contact_kd = self.R.get_value(self.physics_prim, PrimType.SCENE, "soft_contact_kd")
        self.model.soft_contact_ke = self.R.get_value(self.physics_prim, PrimType.SCENE, "soft_contact_ke")

    def _setup_integrator(self):
        """Set up the integrator, and apply attributes parsed from the stage."""

        if self.integrator_type == IntegratorType.XPBD:
            res = SchemaResolverXPBD()
            R = _ResolverManager([res])
            solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.SolverXPBD,
                defaults={"iterations": self.integrator_iterations},
            )
            self.integrator = newton.solvers.SolverXPBD(self.model, **solver_args)

        elif self.integrator_type == IntegratorType.MJWARP:
            res = SchemaResolverMJWarp()
            R = _ResolverManager([res])
            solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.SolverMuJoCo,
                defaults={"iterations": self.integrator_iterations},
            )
            self.integrator = newton.solvers.SolverMuJoCo(self.model, **solver_args)

        elif self.integrator_type == IntegratorType.COUPLED_MPM:
            res = SchemaResolverCoupledMPM()
            R = _ResolverManager([SchemaResolverMPM(), SchemaResolverXPBD()])
            solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=CoupledMPMIntegrator,
                defaults={"iterations": self.integrator_iterations},
            )
            self.integrator = CoupledMPMIntegrator(self.model, **solver_args)
        else:  # VBD
            res = SchemaResolverVBD()
            R = _ResolverManager([res])
            solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.SolverVBD,
                defaults={"iterations": self.integrator_iterations},
            )
            self.integrator = newton.solvers.SolverVBD(self.model, **solver_args)

        # Iterate resolver-defined keys (these are your internal integrator attribute names)
        var_map = res.mapping.get(PrimType.SCENE, {})
        for key in var_map.keys():
            value = R.get_value(self.physics_prim, PrimType.SCENE, key)
            if value is not None and hasattr(self.integrator, key):
                setattr(self.integrator, key, value)

    def _collect_animated_colliders(self, builder, path_body_map, path_shape_map):
        """
        Go through the builder mass array and set the inverse mass and inertia to 0 for kinematic bodies.
        """
        self.animated_colliders_body_ids = []
        self.animated_colliders_paths = []
        R = _ResolverManager([SchemaResolverSimUsd()])
        for path, body_id in path_body_map.items():
            kinematic_collider = R.get_value(self.in_stage.GetPrimAtPath(path), PrimType.BODY, "kinematic_collider")
            if kinematic_collider:
                self.animated_colliders_body_ids.append(body_id)
                self.animated_colliders_paths.append(path)
                # Mujoco requires nonzero inertia
                # if self.integrator_type != IntegratorType.MJWARP:
                builder.body_mass[body_id] = 9999999.0
                builder.body_inv_mass[body_id] = 0.00000001
                builder.body_inertia[body_id] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                builder.body_inv_inertia[body_id] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    @wp.kernel
    def _update_animated_colliders_kernel(
        dt: float,
        usd_transforms: wp.array(dtype=wp.mat44),
        usd_transforms_next: wp.array(dtype=wp.mat44),
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
    ):
        i = wp.tid()
        xform = wp.transpose(usd_transforms[i])
        xform_next = wp.transpose(usd_transforms_next[i])

        pos, R, _s = wp.transform_decompose(xform)
        pos_next, R_next, _s_next = wp.transform_decompose(xform_next)

        axis, angle = wp.quat_to_axis_angle(R_next * wp.quat_inverse(R))

        body_q[i] = wp.transform(pos, R)
        body_qd[i] = wp.spatial_vector(
            (pos_next - pos) / dt,
            axis * angle / dt,
        )

    def _update_animated_colliders(self, substep: int = 0):
        collider_prims = [self.in_stage.GetPrimAtPath(p) for p in self.animated_colliders_paths]
        collider_body_ids = wp.array(self.animated_colliders_body_ids, dtype=int)

        time = self.fps * (self.sim_time + self.sim_dt * float(substep))
        time_next = time + self.fps * self.sim_dt

        xform_cache = UsdGeom.XformCache(time)
        usd_transforms = wp.array(
            [xform_cache.GetLocalToWorldTransform(prim) for prim in collider_prims], dtype=wp.mat44
        )
        xform_cache.SetTime(time_next)
        usd_transforms_next = wp.array(
            [xform_cache.GetLocalToWorldTransform(prim) for prim in collider_prims], dtype=wp.mat44
        )

        if self.integrator_type == IntegratorType.VBD:
            state_out = self.state_1
        else:
            state_out = self.state_0

        delta_time = (time_next - time) / self.fps
        if self.integrator_type != IntegratorType.MJWARP:
            wp.launch(
                self._update_animated_colliders_kernel,
                dim=len(collider_prims),
                inputs=[delta_time, usd_transforms, usd_transforms_next, state_out.body_q, state_out.body_qd],
            )
        else:
            body_q_np = state_out.body_q.numpy()
            body_qd_np = state_out.body_qd.numpy()
            for i in self.animated_colliders_body_ids:
                path = self.animated_colliders_paths[i]
                prim = self.in_stage.GetPrimAtPath(path)
                wp_xform = parse_xform(prim, time)
                wp_xform_next = parse_xform(prim, time_next)
                vel = wp.vec3(wp_xform_next[0:3] - wp_xform[0:3]) / delta_time
                # TODO: WARNING: we are not computing the angular velocity correctly
                ang = wp.vec3(0.0, 0.0, 0.0)
                body_q_np[i] = wp_xform
                body_qd_np[i] = wp.spatial_vector(vel[0], vel[1], vel[2], ang[0], ang[1], ang[2])
            state_out.joint_q.assign(body_q_np)
            state_out.joint_qd.assign(body_qd_np)

    def simulate(self):
        if not self.collide_on_substeps:
            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=self.rigid_contact_margin)

        for substep in range(self.sim_substeps):
            self._update_animated_colliders(substep)
            if self.use_cuda_graph:
                if substep % self.sim_substeps:
                    wp.capture_launch(self.graph_even_step)
                else:
                    wp.capture_launch(self.graph_odd_step)
            else:
                self.run_substep()

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

        if run_cfg.get("self_collision_off_frame", -1) > 0 and self.integrator.penetration_free_init:
            self_collision_off_time = (run_cfg.get("self_collision_off_frame", -1)) / self.fps
            if self.sim_time > self_collision_off_time:
                self.integrator.penetration_free_init = False

                if self.use_cuda_graph and not self.is_mujoco_cpu_mode:
                    with wp.ScopedCapture() as capture:
                        self.run_substep()
                    self.graph_even_step = capture.graph
                    (self.state_0, self.state_1) = (self.state_1, self.state_0)

                    with wp.ScopedCapture() as capture:
                        self.run_substep()
                    self.graph_odd_step = capture.graph
                    (self.state_0, self.state_1) = (self.state_1, self.state_0)




    def run_substep(self):
        if self.collide_on_substeps:
            self.contacts = self.model.collide(
                self.state_0,
                rigid_contact_margin=self.rigid_contact_margin,
                soft_contact_margin=run_cfg["cloth_cfg"]["particle_radius"],
            )

        self.state_0.clear_forces()
        self.integrator.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

    def step(self):
        with wp.ScopedTimer("step", dict=self.profiler):
            self.simulate()
        self.sim_time += self.frame_dt
        print(f"sim_time = {self.sim_time}")

    def render(self):
        with wp.ScopedTimer("render", dict=self.profiler):
            # self.usd_updater.begin_frame(self.sim_time)
            # self.usd_updater.update_usd(self.state_0)
            # if self.integrator_type == IntegratorType.COUPLED_MPM:
            #     rot, scale = extract_particle_rotation_and_scales(self.integrator.mpm_state_0)
            #     self.usd_updater.render_points(
            #         path="/particles",
            #         points=self.integrator.mpm_state_0.particle_q,
            #         rotations=rot,
            #         scales=scale,
            #         radius=float(self.integrator.mpm_solver.mpm_model.model.particle_radius.numpy()[0]),
            #     )
            #
            # self.usd_updater.end_frame()

            if self.DEBUG:
                self.viewer_gl.begin_frame(self.sim_time)
                self.viewer_gl.log_state(self.state_0)
                self.viewer_gl.end_frame()

                if self.viewer_usd is not None:
                    if self.integrator_type == IntegratorType.COUPLED_MPM:
                        self.viewer_usd.log_points(
                            "sand",
                            points=self.integrator.mpm_state_0.particle_q,
                            radii=self.integrator.mpm_solver.mpm_model.model.particle_radius,
                            colors=self.integrator.particle_render_colors,
                            hidden=False,
                        )

                    self.viewer_usd.begin_frame(self.sim_time)
                    self.viewer_usd.log_state(self.state_0)
                    self.viewer_usd.end_frame()

    def save(self):
        self.viewer_gl.close()
        if self.viewer_usd is not None:
            self.viewer_usd.close()

        self.usd_updater.close()


def print_time_profiler(simulator):
    frame_times = simulator.profiler["step"]
    render_times = simulator.profiler["render"]
    print(f"\nAverage frame sim time: {sum(frame_times) / len(frame_times):.2f} ms")
    print(f"\nAverage frame render time: {sum(render_times) / len(render_times):.2f} ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "stage_path",
        help="Path to the input USD file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output USD file.",
    )
    parser.add_argument("-d", "--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("-n", "--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument(
        "-i",
        "--integrator",
        help="Type of integrator",
        type=IntegratorType,
        choices=list(IntegratorType),
        default=None,
    )

    args = parser.parse_known_args()[0]

    if not args.output:
        path = Path(args.stage_path)
        base_path = path.parent / "output"
        base_path.mkdir(parents=True, exist_ok=True)
        args.output = str(base_path / path.name)
        print(f'Output path not specified (-o flag). Writing to "{args.output}".')

    with wp.ScopedDevice(args.device):
        simulator = Simulator(input_path=args.stage_path, output_path=args.output, integrator=args.integrator)

        for i in range(args.num_frames):
            print(f"frame {i}")
            simulator.step()
            simulator.render()

        print_time_profiler(simulator)

        simulator.save()
