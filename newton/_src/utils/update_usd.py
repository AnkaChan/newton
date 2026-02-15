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


from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from pxr import Sdf, Usd
    from ..core.types import AxisType


class UpdateUsd:
    def __init__(
        self,
        stage: str | Usd.Stage,
        source_stage: str | Usd.Stage | None = None,
        scaling: float = 1.0,
        fps: int = 60,
        up_axis: AxisType | None = None,
        path_body_map: dict | None = None,
        path_body_relative_transform: dict | None = None,
        builder_results: dict | None = None,
        global_offset: wp.vec3 | None = None,
    ):
        """
        Construct a UpdateUsd object.

        Args:
            model (newton.Model): The Newton physics model to render.
            stage (str | Usd.Stage): The USD stage to render to. This is the output stage.
            source_stage (str | Usd.Stage, optional): The USD stage to use as a source for the output stage.
            scaling (float, optional): Scaling factor for the rendered objects. Defaults to 1.0.
            fps (int, optional): Frames per second for the animation. Defaults to 60.
            up_axis (newton.AxisType, optional): Up axis for the scene. If None, uses model's up axis. Defaults to None.
            show_joints (bool, optional): Whether to show joint visualizations.  Defaults to False.
            path_body_map (dict, optional): A dictionary mapping prim paths to body IDs.
            path_body_relative_transform (dict, optional): A dictionary mapping prim paths to relative transformations.
            builder_results (dict, optional): A dictionary containing builder results.
            global_offset (wp.vec3, optional): A global offset to apply to the stage.
        """
        if source_stage:
            if path_body_map is None:
                raise ValueError("path_body_map must be set if you are providing a source_stage")
            if path_body_relative_transform is None:
                raise ValueError("path_body_relative_transform must be set if you are providing a source_stage")
            if builder_results is None:
                raise ValueError("builder_results must be set if you are providing a source_stage")

        self.source_stage = source_stage
        self.global_offset: wp.vec3 = global_offset if global_offset is not None else wp.vec3(0.0, 0.0, 0.0)
        if not self.source_stage:
            raise ValueError("source_stage must be set")
        else:
            self.stage_path = stage
            self.stage = self._create_output_stage(self.source_stage, stage)
            self.fps = fps
            self.scaling = scaling
            self.up_axis = up_axis
            self.path_body_map = path_body_map
            self.path_body_relative_transform = path_body_relative_transform
            self.builder_results = builder_results
            self.particle_layer_info = {}
            self._prepare_output_stage()
            self._precompute_parents_xform_inverses()

    def _prepare_output_stage(self):
        """
        Set USD parameters on the output stage to match the simulation settings.

        Must be called after _apply_solver_attributes!"""
        from pxr import Sdf  # noqa: PLC0415

        if self.path_body_map is None:
            raise ValueError("self.path_body_map must be set before calling _prepare_output_stage")

        self.stage.SetStartTimeCode(0.0)
        self.stage.SetEndTimeCode(0.0)
        # NB: this is now coming from warp:fps, but timeCodesPerSecond is a good source too
        self.stage.SetTimeCodesPerSecond(self.fps)

        for prim_path in self.path_body_map.keys():
            prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
            self._xform_to_tqs(prim)

    def _precompute_parents_xform_inverses(self):
        """
        Convention: prefix c is for **current** prim.
        Prefix p is for **parent** prim.
        """
        from pxr import Sdf, Usd  # noqa: PLC0415

        if self.path_body_map is None:
            raise ValueError("self.path_body_map must be set before calling _precompute_parents_xform_inverses")

        self.parent_translates = {}
        self.parent_inv_Rs = {}
        self.parent_inv_Rns = {}

        with wp.ScopedTimer("prep_parents_xform"):
            time = Usd.TimeCode.Default()
            for prim_path in self.path_body_map.keys():
                current_prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
                parent_path = str(current_prim.GetParent().GetPath())

                if parent_path not in self.parent_translates:
                    (
                        self.parent_translates[parent_path],
                        self.parent_inv_Rs[parent_path],
                        self.parent_inv_Rns[parent_path],
                    ) = self._compute_parents_inverses(prim_path, time)

    def begin_frame(self, time):
        """
        Begin a new frame at the given simulation time.

        Parameters:
            time (float): The simulation time for the new frame.
        """
        # super().begin_frame(time)
        self.time = round(time * self.fps)
        self.stage.SetEndTimeCode(self.time)

    def end_frame(self):
        """
        End the current frame.

        This method is a placeholder for any end-of-frame logic required by the backend.
        """
        pass

    def attach_source_stage(self, source_stage: str | Usd.Stage, output_stage: str | Usd.Stage):
        """
        Attach a USD stage created from a source stage (flattened copy) as the viewer's stage.

        """
        if isinstance(output_stage, str):
            stage = Usd.Stage.Open(source_stage, Usd.Stage.LoadAll)
            flattened = stage.Flatten()
            temp_stage = Usd.Stage.Open(flattened.identifier)
            exported = temp_stage.ExportToString()

            new_stage = Usd.Stage.CreateNew(output_stage)
            new_stage.GetRootLayer().ImportFromString(exported)
            self.stage = new_stage
        elif isinstance(output_stage, Usd.Stage):
            self.stage = output_stage
        else:
            raise ValueError("output_stage must be a string or a Usd.Stage")

        # carry over fps and basic settings
        self.stage.SetFramesPerSecond(self.fps)
        self.stage.SetStartTimeCode(0)
        return self.stage

    #    def create_output_stage_from_source(self, source_stage: str | Usd.Stage, output_stage: str | Usd.Stage) -> Usd.Stage:
    #        """
    #        Create and return a new stage from a source, without attaching it. Provided for convenience.
    #        """
    #        if isinstance(output_stage, str):
    #            stage = Usd.Stage.Open(source_stage, Usd.Stage.LoadAll)
    #            flattened = stage.Flatten()
    #            temp_stage = Usd.Stage.Open(flattened.identifier)
    #            exported = temp_stage.ExportToString()
    #
    #            new_stage = Usd.Stage.CreateNew(output_stage)
    #            new_stage.GetRootLayer().ImportFromString(exported)
    #            return new_stage
    #        elif isinstance(output_stage, Usd.Stage):
    #            return output_stage
    #        else:
    #            raise ValueError("output_stage must be a string or a Usd.Stage")
    #
    def configure_body_mapping(
        self,
        path_body_map: dict,
        path_body_relative_transform: dict,
        builder_results: dict,
    ):
        """
        Configure mapping from USD prim paths to body indices and precompute parent inverses.
        """
        if path_body_map is None:
            raise ValueError("path_body_map must be set for configure_body_mapping")
        if path_body_relative_transform is None:
            raise ValueError("path_body_relative_transform must be set for configure_body_mapping")
        if builder_results is None:
            raise ValueError("builder_results must be set for configure_body_mapping")

        self.path_body_map = path_body_map
        self.path_body_relative_transform = path_body_relative_transform
        self.builder_results = builder_results

        # Stage-wide prep to align xform stacks and cache parent inverses
        self._prepare_output_stage_for_mapping()
        self._precompute_parents_xform_inverses()

    def update_usd(self, state):
        """
        Write transforms of USD prims as time-sampled animation in USD using the current simulation state.

        Args:
            state (newton.State): The simulation state to render.
        """
        from pxr import Sdf  # noqa: PLC0415

        if self.path_body_map is None:
            raise ValueError("configure_body_mapping must be called before update_stage_from_state")

        # body_q is either a warp array (preferred) or numpy; ensure numpy here
        body_q = state.body_q.numpy()
        if self.global_offset is not None:
            body_q[:, :3] -= self.global_offset

        # Use a change block for efficient time-sampled updates
        with Sdf.ChangeBlock():
            for prim_path, body_id in self.path_body_map.items():
                full_xform = wp.transform(*body_q[body_id])

                # apply relative xform if any
                rel_xform = self.path_body_relative_transform.get(prim_path)
                if rel_xform:
                    full_xform = wp.mul(full_xform, rel_xform)

                # convert to local space relative to parent (if required)
                full_xform = self._apply_parents_inverse_xform(wp.transform(*full_xform), prim_path, body_q)

                # set xform ops at current frame index
                self._update_usd_prim_xform(prim_path, full_xform)

    def render_points(
        self,
        path: str,
        points: wp.array,
        rotations: wp.array | None = None,
        scales: wp.array | None = None,
        radius: float | None = None,
    ):
        from pxr import UsdGeom

        stage = self.stage
        time = self.time

        instancer_path = path
        instancer = UsdGeom.PointInstancer.Get(stage, instancer_path)
        if not instancer:
            # UsdGeom.Xform.Define(stage, root_path)
            instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
            instancer_sphere = UsdGeom.Sphere.Define(stage, instancer.GetPath().AppendChild("sphere"))
            instancer_sphere.GetRadiusAttr().Set(radius)
            # instancer_sphere = UsdGeom.Cube.Define(self.stage, instancer.GetPath().AppendChild("sphere"))
            # instancer_sphere.GetSizeAttr().Set(radius*2.0)

            instancer.CreatePrototypesRel().SetTargets([instancer_sphere.GetPath()])
            instancer.CreateProtoIndicesAttr().Set([0] * len(points))

        instancer.GetPositionsAttr().Set(points.numpy() - self.global_offset, time)

        if rotations is not None:
            instancer.GetOrientationsAttr().Set(rotations.numpy(), time)

        if scales is not None:
            instancer.GetScalesAttr().Set(scales.numpy(), time)

    def _apply_parents_inverse_xform(self, full_xform: wp.transform, prim_path: str, body_q: wp.array) -> wp.transform:
        """
        Transformation in Warp sim consists of translation and pure rotation: trnslt and quat.
        Transformations of bodies are stored in body_q in simulation state.
        For sim_usd, trnslt is computed directly from PhysicsUtils by the function GetRigidBodyTransformation
        in parseUtils.cpp:
            const GfMatrix4d mat = UsdGeomXformable(bodyPrim).ComputeLocalToWorldTransform(UsdTimeCode::Default());
            const GfTransform tr(mat);
            const GfVec3d pos = tr.GetTranslation();
            const GfQuatd rot = tr.GetRotation().GetQuat();
        In import_nvusd, we set trnslt = pos and quat = fromgfquat(rot), where fromgfquat has the following logic:
        wp.normalize(wp.quat(*gfquat.imaginary, gfquat.real)).

        For trnslt, we have:
            warp_trnslt = xform.ComputeLocalToWorldTransform().GetTranslation().
        But in USD space:
            xform.ComputeLocalToworldTransform() = xform.GetLocalTransform() * xform.ComputeParentToWorldTransform()
            Prim_LTW_USD = Prim_Local_USD * Parent_LTW_USD,
        or in Warp space, we work with transpose and arrive at:
            Prim_LTW = Parent_LTW * Prim_Local
            warp_trnslt = p_Rot * prim_trnslt + p_trnslt,
            i.e.
            prim_trnslt = p_inv_Rot * (warp_trnslt - p_trnslt).

        For rotation, we have:
            rot = tr.GetRotation().GetQuat();
            warp_quat = wp.normalize(wp.quat(rot)).
        However, rot is already normalized, so we don't actually need to renormalize it.
        So in Warp space,
            warp_Rot = p_Rot * diag(1/s_x, 1/s_y, 1/s_z) * prim_Rot, so
            prim_Rot = wp.inv(p_Rot * diag(1/s_x, 1/s_y, 1/s_z)) * warp_Rot

        Both p_inv_Rot and wp.inv(p_Rot * diag(1/s_x, 1/s_y, 1/s_z)) do not change during sim, so they are computed in __init__.
        """
        from pxr import Sdf  # noqa: PLC0415

        current_prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
        parent_path = str(current_prim.GetParent().GetPath())

        if parent_path in self.path_body_map:
            parent_xform = wp.transform(*body_q[self.path_body_map[parent_path]])
            return wp.transform_inverse(parent_xform) * full_xform

        parent_translate = self.parent_translates[parent_path]
        parent_inv_Rot = self.parent_inv_Rs[parent_path]
        parent_inv_Rot_n = self.parent_inv_Rns[parent_path]

        warp_translate = wp.transform_get_translation(full_xform)
        warp_quat = wp.transform_get_rotation(full_xform)

        prim_translate = parent_inv_Rot * (warp_translate - wp.vec3(parent_translate))
        prim_quat = parent_inv_Rot_n * warp_quat

        return wp.transform(prim_translate, prim_quat)

    def _update_usd_prim_xform(self, prim_path: str, warp_xform: wp.transform):
        from pxr import Gf, Sdf, UsdGeom  # noqa: PLC0415

        prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))

        pos = tuple(map(float, warp_xform[0:3]))
        rot = tuple(map(float, warp_xform[3:7]))

        xform = UsdGeom.Xform(prim)
        xform_ops = xform.GetOrderedXformOps()

        # Detect precision from the first xform op (translation)
        if xform_ops and xform_ops[0].GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            # Use double precision types
            if pos is not None:
                xform_ops[0].Set(Gf.Vec3d(pos[0], pos[1], pos[2]), self.time)
            if rot is not None:
                xform_ops[1].Set(Gf.Quatd(rot[3], rot[0], rot[1], rot[2]), self.time)
        else:
            # Use float precision types
            if pos is not None:
                xform_ops[0].Set(Gf.Vec3f(pos[0], pos[1], pos[2]), self.time)
            if rot is not None:
                xform_ops[1].Set(Gf.Quatf(rot[3], rot[0], rot[1], rot[2]), self.time)

    # Note: if _compute_parents_inverses turns to be too slow, then we should consider using a UsdGeomXformCache as described here:
    # https://openusd.org/release/api/class_usd_geom_imageable.html#a4313664fa692f724da56cc254bce70fc
    def _compute_parents_inverses(self, prim_path: str, time: Usd.TimeCode):
        from pxr import Gf, Sdf, UsdGeom  # noqa: PLC0415

        prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
        xform = UsdGeom.Xform(prim)

        parent_world = Gf.Matrix4f(xform.ComputeParentToWorldTransform(time))
        Rpw = wp.mat33(parent_world.ExtractRotationMatrix().GetTranspose())
        (_, _, s, _, translate_parent_world, _) = parent_world.Factor()

        transpose_Rpwn = wp.mat33(
            Rpw[0, 0] / s[0],
            Rpw[1, 0] / s[0],
            Rpw[2, 0] / s[0],
            Rpw[0, 1] / s[1],
            Rpw[1, 1] / s[1],
            Rpw[2, 1] / s[1],
            Rpw[0, 2] / s[2],
            Rpw[1, 2] / s[2],
            Rpw[2, 2] / s[2],
        )
        inv_Rpwn = wp.quat_from_matrix(transpose_Rpwn)
        inv_Rpw = wp.inverse(Rpw)

        return translate_parent_world, inv_Rpw, inv_Rpwn

    def _precompute_parents_xform_inverses(self):
        """
        Convention: prefix c is for **current** prim.
        Prefix p is for **parent** prim.
        """
        from pxr import Sdf, Usd  # noqa: PLC0415

        if self.path_body_map is None:
            raise ValueError("path_body_map must be set before calling _precompute_parents_xform_inverses")

        self.parent_translates = {}
        self.parent_inv_Rs = {}
        self.parent_inv_Rns = {}

        time = Usd.TimeCode.Default()
        for prim_path in self.path_body_map.keys():
            current_prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
            parent_path = str(current_prim.GetParent().GetPath())

            if parent_path not in self.parent_translates:
                (
                    self.parent_translates[parent_path],
                    self.parent_inv_Rs[parent_path],
                    self.parent_inv_Rns[parent_path],
                ) = self._compute_parents_inverses(prim_path, time)

    def _prepare_output_stage_for_mapping(self):
        from pxr import Sdf  # noqa: PLC0415

        if self.path_body_map is None:
            raise ValueError("path_body_map must be set before calling _prepare_output_stage_for_mapping")

        # Reset time codes to align with viewer timeline
        self.stage.SetStartTimeCode(0.0)
        # keep existing end-time; it will be extended in begin_frame
        self.stage.SetTimeCodesPerSecond(self.fps)

        for prim_path in self.path_body_map.keys():
            prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
            self._xform_to_tqs(prim)

    def _create_output_stage(self, source_stage: str | Usd.Stage, output_stage: str | Usd.Stage) -> Usd.Stage:
        from pxr import Usd  # noqa: PLC0415

        if isinstance(output_stage, str):
            source_stage = Usd.Stage.Open(source_stage, Usd.Stage.LoadAll)
            flattened = source_stage.Flatten()
            stage = Usd.Stage.Open(flattened.identifier)
            exported = stage.ExportToString()

            output_stage = Usd.Stage.CreateNew(output_stage)
            output_stage.GetRootLayer().ImportFromString(exported)
            return output_stage
        elif isinstance(output_stage, Usd.Stage):
            return output_stage
        else:
            raise ValueError("output_stage must be a string or a Usd.Stage")

    def close(self):
        """
        Finalize and save the USD stage.

        This should be called when all logging is complete to ensure the USD file is written.
        """
        try:
            self.stage.Save()
            self.stage = None
            print("USD stage saved successfully")
        except Exception as e:
            print("Failed to save USD stage:", e)
            return False

    @staticmethod
    def _xform_to_tqs(prim: Usd.Prim, time: Usd.TimeCode | None = None):
        """
        Update the transformation stack of a primitive to translate/orient/scale format.

        The original transformation stack is assumed to be a rigid transformation.
        The precision (float/double) is detected from existing transform ops.
        """
        from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415

        if time is None:
            time = Usd.TimeCode.Default()

        _tqs_op_order = [UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.TypeScale]

        xform = UsdGeom.Xform(prim)
        xform_ops = xform.GetOrderedXformOps()

        # Detect precision from existing transform ops
        # Prefer double if any op uses double, otherwise use float
        detected_precision = UsdGeom.XformOp.PrecisionFloat
        if xform_ops:
            for op in xform_ops:
                if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                    detected_precision = UsdGeom.XformOp.PrecisionDouble
                    break

        _tqs_op_precision = [detected_precision, detected_precision, detected_precision]

        # if the order, type, and precision of the transformation is already in our canonical form, then there's no need to change anything.
        if _tqs_op_order == [op.GetOpType() for op in xform_ops] and _tqs_op_precision == [
            op.GetPrecision() for op in xform_ops
        ]:
            return

        # this assumes no skewing
        # NB: the rotation coming from Factor is the result of solving an eigenvalue problem. We found wrong answer with non-identity scaling.
        m_lcl = xform.GetLocalTransformation(time)
        (_, _, scale, _, translation, _) = m_lcl.Factor()

        # Use appropriate type based on detected precision
        if detected_precision == UsdGeom.XformOp.PrecisionDouble:
            t = Gf.Vec3d(translation)
            q = Gf.Quatd(m_lcl.ExtractRotationQuat())
            s = Gf.Vec3d(scale)
        else:
            t = Gf.Vec3f(translation)
            q = Gf.Quatf(m_lcl.ExtractRotationQuat())
            s = Gf.Vec3f(scale)

        # need to reset the transform
        for op in xform_ops:
            attr = op.GetAttr()
            prim.RemoveProperty(attr.GetName())

        xform.ClearXformOpOrder()
        xform.AddTranslateOp(precision=detected_precision).Set(t)
        xform.AddOrientOp(precision=detected_precision).Set(q)
        xform.AddScaleOp(precision=detected_precision).Set(s)

    @staticmethod
    def _stitch_time_samples(stages, instancer_path, name: str):
        from pxr import UsdGeom  # noqa: PLC0415

        prim_path = f"{instancer_path}_split/{name}"

        main_stage = stages[0]
        instancer = UsdGeom.PointInstancer.Get(main_stage, prim_path)

        for stage in stages[1:]:
            other_instancer = UsdGeom.PointInstancer.Get(stage, prim_path)
            for time in other_instancer.GetPositionsAttr().GetTimeSamples():
                instancer.GetPositionsAttr().Set(other_instancer.GetPositionsAttr().Get(time), time)
                instancer.GetOrientationsAttr().Set(other_instancer.GetOrientationsAttr().Get(time), time)

    @dataclass
    class ParticleLayerInfo:
        particle_layer_names: list[str]
        particle_layer_count: int
        frames_per_layer: int
        particles_per_layer: int

        particle_layers: list = field(default_factory=list)
        cur_layer_num: int = -1
        num_written_frames: int = 0

        def get_layer_name(self, chunk: int):
            num = self.cur_layer_num
            suffix = "" if num == 0 else f"_{num}"
            return self.particle_layer_names[chunk] + suffix

        def get_layer_path(self, stage_path, instancer_path: str, chunk: int):
            return f"{os.path.splitext(stage_path)[0]}_{str(instancer_path).replace('/', '_')}_{self.get_layer_name(chunk)}.usdc"

    def _renew_particle_sublayers(
        self,
        instancer_path: str,
    ):
        from pxr import Sdf  # noqa: PLC0415

        particle_layer_info = self.particle_layer_info[instancer_path]
        particle_layer_info.cur_layer_num += 1

        out_root_layer = self.stage.GetRootLayer()

        # Remove sublayer path for this instancer
        instancer_path_key = str(instancer_path).replace("/", "_")
        sublayer_paths = [path for path in out_root_layer.subLayerPaths if instancer_path_key not in path]
        out_root_layer.subLayerPaths.clear()
        for path in sublayer_paths:
            out_root_layer.subLayerPaths.append(path)

        particle_layers = [
            Sdf.Layer.New(
                Sdf.FileFormat.FindByExtension("usdc"),
                particle_layer_info.get_layer_path(self.stage_path, instancer_path, chunk),
            )
            for chunk in range(particle_layer_info.particle_layer_count)
        ]
        for layer in particle_layers:
            layer.timeCodesPerSecond = self.fps
            out_root_layer.subLayerPaths.append(os.path.join(".", os.path.basename(layer.identifier)))

        particle_layer_info.particle_layers = particle_layers

        return particle_layers

    def create_particle_sublayers(
        self, instancer_path, particle_count, frame_count, has_orientations=True, has_scales=False
    ):
        out_point_count = particle_count
        bytes_per_point = 12
        if has_orientations:
            bytes_per_point += 8
        if has_scales:
            bytes_per_point += 12

        # 6.4GB files
        particles_per_layer = min(particle_count, int(6.4e9 // bytes_per_point / frame_count))
        # 24GB working memory
        frames_per_layer = int(2.4e10 // bytes_per_point / out_point_count)

        particle_layer_count = (out_point_count + particles_per_layer - 1) // particles_per_layer
        particle_layer_names = [f"particles_{i:04d}" for i in range(particle_layer_count)]

        print(
            f"Splitting scheme for instancer {instancer_path}: {particles_per_layer} particles per layer ({particle_layer_count} layers), {frames_per_layer} frames per layer"
        )

        self.particle_layer_info[instancer_path] = UpdateUsd.ParticleLayerInfo(
            particle_layer_names=particle_layer_names,
            particle_layer_count=particle_layer_count,
            frames_per_layer=frames_per_layer,
            particles_per_layer=particles_per_layer,
        )

        self._renew_particle_sublayers(instancer_path)

    def _split_instancer(self, path: str, split_path: str, slc: slice):
        from pxr import UsdGeom  # noqa: PLC0415

        split_instancer = UsdGeom.PointInstancer.Get(self.stage, split_path)
        if split_instancer:
            return

        instancer = UsdGeom.PointInstancer.Get(self.stage, path)
        if not instancer:
            return

        instancer_split = UsdGeom.PointInstancer.Define(self.stage, split_path)
        instancer_split.CreatePrototypesRel().SetTargets(instancer.GetPrototypesRel().GetTargets())
        instancer_split.CreateProtoIndicesAttr().Set(instancer.GetProtoIndicesAttr().Get()[slc])
        instancer_split.CreateScalesAttr().Set(instancer.GetScalesAttr().Get()[slc])
        instancer_split.CreateOrientationsAttr().Set(instancer.GetOrientationsAttr().Get()[slc])
        instancer_split.CreatePositionsAttr().Set(instancer.GetPositionsAttr().Get()[slc])

    def render_points_split(
        self,
        path: str,
        points: wp.array,
        rotations: wp.array | None = None,
        scales: wp.array | None = None,
        radius: float | None = None,
    ):
        from pxr import Usd  # noqa: PLC0415

        layer_info = self.particle_layer_info[path]

        frame = layer_info.num_written_frames
        layer_info.num_written_frames += 1

        # Renew layers if required
        num = frame // layer_info.frames_per_layer
        if num != layer_info.cur_layer_num:
            print(f"Switching layers from {layer_info.cur_layer_num} to {num}...")

            try:
                import psutil
                memory_info = psutil.Process().memory_info()
                print(f"Memory usage before switching layers: {memory_info.rss / (1024 * 1024):.2f} MB")
            except ImportError:
                pass
            self.stage.Save()
            self.stage = Usd.Stage.Open(self.stage_path)

            self._renew_particle_sublayers(path)

            try:
                import psutil
                memory_info = psutil.Process().memory_info()
                print(f"Memory usage after re-creating layers: {memory_info.rss / (1024 * 1024):.2f} MB")
            except ImportError:
                pass

        prev_edit_target = self.stage.GetEditTarget()
        for k in range(layer_info.particle_layer_count):
            slc = slice(
                k * layer_info.particles_per_layer, min((k + 1) * layer_info.particles_per_layer, points.shape[0])
            )
            self.stage.SetEditTarget(Usd.EditTarget(layer_info.particle_layers[k]))

            self._split_instancer(path, f"{path}_split/{layer_info.particle_layer_names[k]}", slc)

            self.render_points(
                f"{path}_split/{layer_info.particle_layer_names[k]}",
                points[slc],
                rotations[slc] if rotations is not None else None,
                scales[slc] if scales is not None else None,
                radius=radius,
            )
        self.stage.SetEditTarget(prev_edit_target)

    def stitch_time_samples(self, instancer_path: str):
        from pxr import Usd  # noqa: PLC0415

        layer_info = self.particle_layer_info[instancer_path]
        del layer_info.particle_layers
        cur_layer_num = layer_info.cur_layer_num
        particle_layer_count = len(layer_info.particle_layer_names)

        if cur_layer_num > 0:
            print(f"Stitching layers {cur_layer_num}...", flush=True)

            for chunk in range(particle_layer_count):
                name = layer_info.particle_layer_names[chunk]

                stages = []
                for num in range(cur_layer_num + 1):
                    layer_info.cur_layer_num = num
                    stages.append(Usd.Stage.Open(layer_info.get_layer_path(self.stage_path, instancer_path, chunk)))

                UpdateUsd._stitch_time_samples(stages, instancer_path, name)
                stages[0].Save()

                for stage in stages[1:]:
                    os.remove(stage.GetRootLayer().identifier)

                del stages
                try:
                    import psutil
                    memory_info = psutil.Process().memory_info()
                    print(f"Memory usage after stitching {name}: {memory_info.rss / (1024 * 1024):.2f} MB")
                except ImportError:
                    pass

            out_stage = Usd.Stage.Open(self.stage_path)
            out_root_layer = out_stage.GetRootLayer()

            try:
                import psutil
                memory_info = psutil.Process().memory_info()
                print(f"Memory usage after re-opening: {memory_info.rss / (1024 * 1024):.2f} MB")
            except ImportError:
                pass

            instancer_path_key = str(instancer_path).replace("/", "_")
            sublayer_paths = [path for path in out_root_layer.subLayerPaths if instancer_path_key not in path]
            out_root_layer.subLayerPaths.clear()
            for path in sublayer_paths:
                out_root_layer.subLayerPaths.append(path)

            layer_info.cur_layer_num = 0
            for chunk in range(layer_info.particle_layer_count):
                out_root_layer.subLayerPaths.append(layer_info.get_layer_path(self.stage_path, instancer_path, chunk))

            out_stage.Save()