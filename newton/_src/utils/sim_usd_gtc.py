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

from enum import Enum
from pathlib import Path
from typing import Optional, ClassVar
import inspect

from pxr import Usd, UsdPhysics

import numpy as np
import warp as wp
import newton
from newton._src.utils.import_usd import parse_usd
from newton._src.utils.update_usd import UpdateUsd
from newton._src.utils.schema_resolver import (
    Attribute,
    PrimType,
    SchemaResolver,
    SchemaResolverNewton,
    SchemaResolverPhysx,
    _ResolverManager,
)


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
            "sim_substeps": [Attribute("newton:substeps", 32)],
            "integrator_type": [Attribute("newton:integrator", "xpbd")],
            "integrator_iterations": [Attribute("newton:integrator_iterations", 100)],
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
        },
    }


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
        results = parse_usd(
            builder,
            self.in_stage,
            invert_rotations=True,
            collapse_fixed_joints=True,
        )
        self.R = _ResolverManager([SchemaResolverSimUsd(), SchemaResolverNewton(), SchemaResolverMJWarp()])
        self.physics_prim = next(iter([prim for prim in self.in_stage.Traverse() if prim.IsA(UsdPhysics.Scene)]), None)

        # self._prepare_kinematic_bodies(builder, results["path_body_map"], results["path_shape_map"])
        self.path_body_map = results["path_body_map"]
        self.path_shape_map = results["path_shape_map"]
        self.body_path_map = {idx: path for path, idx in self.path_body_map.items()}
        self.shape_path_map = {idx: path for path, idx in self.path_shape_map.items()}

        self._setup_solver_attributes()
        if integrator:
            self.integrator_type = integrator

        if self.integrator_type == IntegratorType.VBD:
            builder.color()
        self.model = builder.finalize()
        self.builder_results = results

        self.path_body_map = self.builder_results["path_body_map"]
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

        self.use_cuda_graph = wp.get_device().is_cuda
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

        self.DEBUG = False
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

    def _prepare_kinematic_bodies(self, builder, path_body_map, path_shape_map):
        """
        Go through the builder mass array and set the inverse mass and inertia to 0 for kinematic bodies.
        """
        print(f"==== process_kinematic_bodies ====")
        R = _ResolverManager([SchemaResolverSimUsd()])
        for path, body_id in path_body_map.items():
            kinematic_collider = R.get_value(self.in_stage.GetPrimAtPath(path), PrimType.BODY, "kinematic_collider")
            print(f"kinematic_collider = {kinematic_collider} for path {path}")
            if kinematic_collider:
                print(f"builder.body_mass[body_id] orig = {builder.body_mass[body_id]}")
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inv_inertia[body_id] = wp.mat33(0.0)
        print(f"==== end process_kinematic_bodies ====")

    def simulate(self):
        if not self.collide_on_substeps:
            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=self.rigid_contact_margin)

        for substep in range(self.sim_substeps):
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
            self.usd_updater.end_frame()

            if self.DEBUG:
                self.viewer.begin_frame(self.sim_time)
                self.viewer.log_state(self.state_0)
                self.viewer.end_frame()

    def save(self):
        self.usd_updater.close()


def print_time_profiler(simulator):
    frame_times = simulator.profiler["step"]
    render_times = simulator.profiler["render"]
    print("\nAverage frame sim time: {:.2f} ms".format(sum(frame_times) / len(frame_times)))
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

        for i in range(args.num_frames):
            print(f"frame {i}")
            simulator.step()
            simulator.render()

        print_time_profiler(simulator)

        simulator.save()
