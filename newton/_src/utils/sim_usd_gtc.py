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
from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics

import newton
import newton.examples
from newton._src.utils.import_usd import parse_usd
from newton._src.utils.schema_resolver import (
    Attribute,
    PrimType,
    SchemaResolver,
    SchemaResolverNewton,
    SchemaResolverPhysx,
    _ResolverManager,
)
from newton._src.utils.update_usd import UpdateUsd


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
            "soft_contact_ke": [Attribute("newton:soft_contact_ke", 1.0e4)],
            "soft_contact_kd": [Attribute("newton:soft_contact_kd", 1.0e2)],
            # solver attributes
            "fps": [Attribute("newton:fps", 60)],
            "sim_substeps": [Attribute("newton:substeps", 8)],
            "integrator_type": [Attribute("newton:integrator", "xpbd")],
            "integrator_iterations": [Attribute("newton:integrator_iterations", 10)],
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
            "njmax": [Attribute("newton:mjwarp:njmax", 152)],
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
            "voxel_size": [Attribute("newton:mpm:voxel_size", 0.05)],
            "grid_type": [Attribute("newton:mpm:grid_type", "sparse")],
            "max_iterations": [Attribute("newton:mpm:max_iterations", 250)],
        },
    }


class CoupledMPMIntegrator(newton.solvers.SolverBase):
    """Integrator for coupled MPM and rigid body solvers."""

    def __init__(self, model: newton.Model, particles_dict: dict[str, np.ndarray], **kwargs):
        super().__init__(model)

        self.particles_dict = particles_dict

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
            self.sand_body_forces = wp.zeros_like(state_0.body_f)
            self.collider_impulses = None
            self.collider_impulse_pos = None
            self.collider_impulse_ids = None
            self.collect_collider_impulses(self.mpm_state_0)
            self._initialized = True

        has_compliance_bodies = self.mpm_state_0.body_q is not None

        if has_compliance_bodies:
            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    state_0.body_q,
                    self.model.body_com,
                    self.model.body_mass,
                    state_0.body_f,
                ],
            )

        self.sand_body_forces.assign(state_0.body_f)
        self.rigid_solver.step(state_0, state_1, control, contacts, dt)

        # Subtract previously applied impulses from body velocities
        if has_compliance_bodies:
            wp.launch(
                substract_body_force,
                dim=self.mpm_state_0.body_q.shape,
                inputs=[
                    dt,
                    state_1.body_q,
                    state_1.body_qd,
                    self.sand_body_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.mpm_state_0.body_q,
                    self.mpm_state_0.body_qd,
                ],
            )

        self.mpm_solver.step(self.mpm_state_0, self.mpm_state_1, None, None, dt)

        self.collect_collider_impulses(self.mpm_state_1)

        # self.mpm_state_1.body_q.assign(state_1.body_q)
        # self.mpm_state_1.body_qd.assign(state_1.body_qd)
        # self.mpm_solver.project_outside(self.mpm_state_1, self.mpm_state_1, dt)

        self.mpm_state_0, self.mpm_state_1 = self.mpm_state_1, self.mpm_state_0

    def _build_mpm_model(self, model, mpm_options):
        sand_builder = newton.ModelBuilder()
        self._add_particles(sand_builder, mpm_options.voxel_size)
        sand_model = sand_builder.finalize()

        # basic particle material params
        sand_model.particle_mu = 0.48
        sand_model.particle_ke = 1.0e15
        sand_model.particle_kd = 0.0
        sand_model.particle_adhesion = 0.0
        sand_model.particle_cohesion = 0.0

        mpm_model = newton.solvers.SolverImplicitMPM.Model(sand_model, mpm_options)

        self._setup_mpm_collider(model, mpm_model)

        return mpm_model

    def _setup_mpm_collider(self, model: newton.Model, mpm_model: newton.solvers.SolverImplicitMPM.Model):
        collider_body_id = np.arange(-1, model.body_count)

        # terrain_id =-1
        # collider_body_id = [terrain_id]
        # print(self.model.body_key[terrain_id[0]])

        mpm_model.setup_collider(
            model=self.model,
            collider_body_ids=collider_body_id,
            collider_friction=[0.5 for _ in collider_body_id],
            collider_adhesion=[0.0 for _ in collider_body_id],
            # body_mass=wp.zeros_like(self.model.body_mass),  # so that the bodies are considered as kinematic,
            ground_height=-10.0,
        )

        self.collider_body_id = wp.array(collider_body_id, dtype=int)

    def _add_particles(self, sand_builder: newton.ModelBuilder, voxel_size: float):
        density = 2500

        if self.particles_dict:
            psize = voxel_size / 2.0
            radius = float(psize * 0.5)
            mass = float(psize**3 * density)

            points = np.vstack([pts for pts in self.particles_dict.values()])
        else:
            # ------------------------------------------
            # Add sand bed (1m x 2m x 0.5m) above ground
            # ------------------------------------------
            particles_per_cell = 3.0

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

        sand_builder.particle_q = points
        sand_builder.particle_qd = np.zeros_like(points)
        sand_builder.particle_mass = np.full(points.shape[0], mass)
        sand_builder.particle_radius = np.full(points.shape[0], radius)
        sand_builder.particle_flags = np.ones(points.shape[0], dtype=int)

    def collect_collider_impulses(self, mpm_state):
        self.collider_impulses, self.collider_impulse_pos, self.collider_impulse_ids = (
            self.mpm_solver.collect_collider_impulses(mpm_state)
        )


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index >= 0:
            m = body_mass[body_index]

            f_world = collider_impulses[i] / dt
            max_f = 0.0 * m
            if wp.length(f_world) > max_f:
                f_world = f_world * max_f / wp.length(f_world)

            X_wb = body_q[body_index]
            X_com = body_com[body_index]
            r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
            wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def substract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    body_id = wp.tid()

    # Remove previously applied force
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])

    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


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

    def __init__(self, input_path, output_path, integrator: Optional[IntegratorType] = None):
        def create_stage_from_path(input_path) -> Usd.Stage:
            stage = Usd.Stage.Open(input_path, Usd.Stage.LoadAll)
            flattened = stage.Flatten()
            out_stage = Usd.Stage.Open(flattened.identifier)
            return out_stage

        self.sim_time = 0.0
        self.profiler = {}

        self.in_stage = create_stage_from_path(input_path)

        builder = newton.ModelBuilder()
        builder.up_axis = newton.Axis.Z
        builder.default_shape_cfg.density = 1.0
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        results = parse_usd(
            builder,
            self.in_stage,
            invert_rotations=True,
            collapse_fixed_joints=True,
        )
        self.R = _ResolverManager([SchemaResolverSimUsd(), SchemaResolverNewton(), SchemaResolverMJWarp()])
        self.physics_prim = next(iter([prim for prim in self.in_stage.Traverse() if prim.IsA(UsdPhysics.Scene)]), None)

        self.path_body_map = results["path_body_map"]

        self._setup_solver_attributes()
        if integrator:
            self.integrator_type = integrator

        self._collect_animated_colliders(builder, results["path_body_map"], results["path_shape_map"])
        if self.integrator_type == IntegratorType.VBD:
            builder.color()
        self.model = builder.finalize()
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

        # NB: body_q will be modified, so initial state will be slightly altered
        if self.model.joint_count:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0, mask=None)

        self._update_animated_colliders()
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=self.rigid_contact_margin)

        self.use_cuda_graph = False  # wp.get_device().is_cuda
        self.is_mujoco_cpu_mode = self.integrator_type == IntegratorType.MJWARP and self.R.get_value(
            self.physics_prim, PrimType.SCENE, "use_mujoco_cpu", False
        )
        if self.use_cuda_graph and not self.is_mujoco_cpu_mode:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

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
            self.viewer = newton.viewer.ViewerGL()
            self.viewer.set_model(self.model)


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
        self.rigid_contact_margin = self.R.get_value(self.physics_prim, PrimType.SCENE, "contact_margin", 0.01)

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
                solver_cls=newton.solvers.SolverXPBD,
                defaults={"iterations": self.integrator_iterations},
            )
            mpm_solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.SolverImplicitMPM.Options,
                defaults={},
            )

            particles_dict = {}
            for prim in self.in_stage.Traverse():
                if prim.GetTypeName() == "PointInstancer":
                    pi = UsdGeom.PointInstancer(prim)
                    particles_dict[prim.GetPath()] = np.array(pi.GetPositionsAttr().Get())

            self.integrator = CoupledMPMIntegrator(self.model, particles_dict, **solver_args, **mpm_solver_args)
        else:  # VBD
            res = SchemaResolverVBD()
            R = _ResolverManager([res])
            solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.VBDIntegrator,
                defaults={"iterations": self.integrator_iterations},
            )
            self.integrator = newton.solvers.VBDIntegrator(self.model, **solver_args)

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
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inv_inertia[body_id] = wp.mat33(0.0)

                self.animated_colliders_body_ids.append(body_id)
                self.animated_colliders_paths.append(path)
                # Mujoco requires nonzero inertia
                if self.integrator_type == IntegratorType.MJWARP:
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
        collider_prims = [
            self.in_stage.GetPrimAtPath(self.animated_colliders_paths[i]) for i in self.animated_colliders_body_ids
        ]
        time = self.fps * (self.sim_time + self.sim_dt / self.sim_substeps * float(substep))
        time_next = self.fps * (self.sim_time + self.frame_dt)

        xform_cache = UsdGeom.XformCache(time)
        usd_transforms = wp.array(
            [xform_cache.GetLocalToWorldTransform(prim) for prim in collider_prims], dtype=wp.mat44
        )
        xform_cache.SetTime(time_next)
        usd_transforms_next = wp.array(
            [xform_cache.GetLocalToWorldTransform(prim) for prim in collider_prims], dtype=wp.mat44
        )

        delta_time = (time_next - time) / self.fps
        if self.integrator_type != IntegratorType.MJWARP:
            wp.launch(
                self._update_animated_colliders_kernel,
                dim=len(collider_prims),
                inputs=[delta_time, usd_transforms, usd_transforms_next, self.state_0.body_q, self.state_0.body_qd],
            )
        else:
            body_q_np = self.state_0.body_q.numpy()
            body_qd_np = self.state_0.body_qd.numpy()
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
            self.state_0.joint_q.assign(body_q_np)
            self.state_0.joint_qd.assign(body_qd_np)

    def simulate(self):
        if not self.collide_on_substeps:
            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=self.rigid_contact_margin)

        for substep in range(self.sim_substeps):
            self._update_animated_colliders(substep)

            if self.collide_on_substeps:
                self.contacts = self.model.collide(self.state_0, rigid_contact_margin=self.rigid_contact_margin)

            self.state_0.clear_forces()
            self.integrator.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step", dict=self.profiler):
            if self.use_cuda_graph and not self.is_mujoco_cpu_mode:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt
        print(f"sim_time = {self.sim_time}")

    def render(self):
        with wp.ScopedTimer("render", dict=self.profiler):
            self.usd_updater.begin_frame(self.sim_time)
            self.usd_updater.update_usd(self.state_0)
            if self.integrator_type == IntegratorType.COUPLED_MPM:
                rot, scale = extract_particle_rotation_and_scales(self.integrator.mpm_state_0)
                self.usd_updater.render_points(
                    path="/particles",
                    points=self.integrator.mpm_state_0.particle_q,
                    rotations=rot,
                    scales=scale,
                    radius=float(self.integrator.mpm_solver.mpm_model.model.particle_radius.numpy()[0]),
                )

            self.usd_updater.end_frame()

            if self.DEBUG:
                self.viewer.begin_frame(self.sim_time)
                self.viewer.log_state(self.state_0)
                self.viewer.log_contacts(self.contacts, self.state_0)
                if self.integrator_type == IntegratorType.COUPLED_MPM:
                    self.viewer.log_points(
                        "sand",
                        points=self.integrator.mpm_state_0.particle_q,
                        radii=self.integrator.mpm_solver.mpm_model.model.particle_radius,
                        colors=self.integrator.particle_render_colors,
                        hidden=False,
                    )
                    impulses, pos, cid = self.integrator.mpm_solver.collect_collider_impulses(
                        self.integrator.mpm_state_0
                    )
                    self.viewer.log_lines(
                        "impulses",
                        starts=pos,
                        ends=pos + impulses,
                        colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
                    )
                self.viewer.end_frame()

    def save(self):
        self.usd_updater.close()


def print_time_profiler(simulator):
    frame_times = simulator.profiler["step"]
    render_times = simulator.profiler["render"]
    if frame_times:
        print("\nAverage frame sim time: {:.2f} ms".format(sum(frame_times) / len(frame_times)))
    if render_times:
        print("\nAverage frame render time: {:.2f} ms".format(sum(render_times) / len(render_times)))


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

        i = 0
        while i < args.num_frames:
            print(f"frame {i}")
            if simulator.DEBUG and not simulator.viewer.is_paused():
                simulator.step()
            simulator.render()
            if simulator.DEBUG and not simulator.viewer.is_paused():
                i += 1

        print_time_profiler(simulator)

        simulator.save()
