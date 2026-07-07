# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Proxy-coupling strategy: MuJoCo arm + VBD cloth via SolverCoupledProxy.

MuJoCo Newton solves the rigid arm (IsaacLab-style gains, effort limits,
armature); VBD solves the cloth plus the static shapes. The gripper bodies are
exposed to VBD as virtual proxies so the cloth detects them as contacts and
feeds lagged impulses back to the arm.

Matches IsaacLab Isaac-Pick-Proxy-Cloth-Direct-v0: gravity is enabled with no
gravity compensation (the actuator PD holds the arm against gravity), and the
MuJoCo solver uses the IsaacLab MJWarpSolverCfg defaults.
"""

from __future__ import annotations

import warp as wp

import newton
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

from .base import SolverStrategy
from . import register

# IsaacLab FRANKA_PANDA_PROXY_CFG gains: moderate-stiffness MuJoCo position
# drive (arm 4000/400), stiff fingers (4e4/400), uniform 1000 N effort limit.
ARM_STIFFNESS = 4000.0
ARM_DAMPING = 400.0
FINGER_STIFFNESS = 4.0e4
FINGER_DAMPING = 400.0
EFFORT_LIMIT = 1000.0


@register
class ProxyCouplingStrategy(SolverStrategy):
    key = "proxy"
    collapse_fixed_joints = False
    uses_mujoco = True
    decimation = 1  # IsaacLab Isaac-Pick-Proxy-Cloth-Direct-v0

    def register_attributes(self, builder):
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

    def configure_robot(self, builder, robot_bodies, robot_joints):
        gains = {
            "arm_stiffness": ARM_STIFFNESS,
            "arm_damping": ARM_DAMPING,
            "finger_stiffness": FINGER_STIFFNESS,
            "finger_damping": FINGER_DAMPING,
        }
        gains.update(self.scene_robot_gains())  # scene overrides win
        builder.joint_target_ke[:7] = [gains["arm_stiffness"]] * 7
        builder.joint_target_kd[:7] = [gains["arm_damping"]] * 7
        builder.joint_target_ke[7:9] = [gains["finger_stiffness"]] * 2
        builder.joint_target_kd[7:9] = [gains["finger_damping"]] * 2
        builder.joint_effort_limit[:9] = [EFFORT_LIMIT] * 9
        builder.joint_armature[:7] = [1.0e-3] * 7
        builder.joint_armature[7:9] = [0.0, 0.0]
        # No gravity compensation: IsaacLab leaves mujoco:gravcomp at 0 and lets
        # the actuator PD hold the arm against gravity.

    def filter_collisions(self, builder, robot_shapes, static_shapes):
        pass  # arm and statics live in different solver entries; no filter needed.

    def apply_materials(self, model):
        mats = {
            "shape_material_ke": 4.0e4,
            "shape_material_kd": 1.0e-5,
            "shape_material_mu": 5.0,
            "soft_contact_ke": 1.0e4,
            "soft_contact_kd": 1.0e-2,
            "soft_contact_mu": 1.5,
        }
        mats.update(self.scene_materials())  # scene overrides win
        model.shape_material_ke.fill_(mats["shape_material_ke"])
        model.shape_material_kd.fill_(mats["shape_material_kd"])
        model.shape_material_mu.fill_(mats["shape_material_mu"])
        model.soft_contact_ke = mats["soft_contact_ke"]
        model.soft_contact_kd = mats["soft_contact_kd"]
        model.soft_contact_mu = mats["soft_contact_mu"]

    def post_finalize(self, model, handles):
        self._handles = handles

    def _build_entries_and_coupling(self, model, handles, args):
        """Assemble the MuJoCo+VBD entries and the proxy coupling config.

        Shared by the proxy and soft-constraint strategies, which differ only in
        the coupled-solver class they instantiate (see ``build_solver``).
        """
        vbd_kwargs = dict(
            iterations=int(args.vbd_iterations),
            particle_enable_self_contact=True,
            particle_self_contact_radius=2.0e-3,
            particle_self_contact_margin=2.0e-3,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.0,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
        )
        vbd_kwargs.update(self.scene_solver_overrides())  # scene overrides win
        entries = [
            SolverCoupled.Entry(
                name="mjc",
                # IsaacLab MJWarpSolverCfg: defaults + proxy overrides.
                solver=lambda v: SolverMuJoCo(
                    model=v,
                    solver="newton",
                    integrator="implicitfast",
                    cone="elliptic",
                    impratio=1.0,
                    tolerance=1.0e-6,
                    iterations=int(args.mujoco_iterations),
                    ls_iterations=int(args.mujoco_ls_iterations),
                    ls_parallel=True,
                    ccd_iterations=35,
                    use_mujoco_contacts=True,
                    njmax=300,
                    nconmax=None,
                ),
                bodies=handles.robot_bodies,
                joints=handles.robot_joints,
                shapes=handles.robot_shapes,
            ),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, **vbd_kwargs),
                particles=list(range(model.particle_count)),
                shapes=handles.static_shapes,
            ),
        ]
        coupling = SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="mjc",
                    destination="vbd",
                    bodies=handles.gripper_bodies,
                    mass_scale=float(args.mass_scale),
                    mode=args.coupling_mode,
                    collision_pipeline=lambda m: newton.CollisionPipeline(m, broad_phase="explicit"),
                    collide_interval=1,
                ),
            ],
            iterations=int(args.proxy_iterations),
        )
        return entries, coupling

    def build_solver(self, model, handles, args):
        entries, coupling = self._build_entries_and_coupling(model, handles, args)
        self.solver = SolverCoupledProxy(model=model, entries=entries, coupling=coupling)
        return self.solver

    def sync_initial(self, state):
        self.solver.sync_entry_states(state)

    def post_step(self, model, state_out):
        newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)

    def reset_internal(self, state, device):
        solver = self.solver
        for attr in ("_proxy_mappings", "_proxy_particle_mappings"):
            for mapping in getattr(solver, attr, None) or ():
                for arr_name in ("coupling_forces", "proxy_qd_before"):
                    arr = getattr(mapping, arr_name, None)
                    if arr is not None:
                        arr.zero_()
        for config in (getattr(solver, "_proxy_collision_configs", None) or {}).values():
            if hasattr(config, "collide_counter"):
                config.collide_counter = 0
            contacts = getattr(config, "contacts", None)
            if contacts is not None and hasattr(contacts, "clear"):
                contacts.clear(bump_generation=True)
        for entry in (getattr(solver, "_entries", None) or {}).values():
            for entry_state in (getattr(entry, "state_0", None), getattr(entry, "state_1", None)):
                if entry_state is None:
                    continue
                for arr_name in ("body_f", "particle_f", "body_qdd"):
                    arr = getattr(entry_state, arr_name, None)
                    if arr is not None:
                        arr.zero_()
            sub = getattr(entry, "solver", None)
            mjw_model = getattr(sub, "mjw_model", None) if sub is not None else None
            mjw_data = getattr(sub, "mjw_data", None) if sub is not None else None
            if mjw_model is not None and mjw_data is not None:
                try:
                    import mujoco_warp as _mjw

                    nworld = int(getattr(mjw_data, "nworld", 1))
                    _mjw.reset_data(mjw_model, mjw_data, wp.ones(nworld, dtype=wp.bool, device=device))
                except Exception as exc:  # noqa: BLE001
                    print(f"[proxy] mujoco_warp.reset_data failed: {exc}")
        solver.sync_entry_states(state)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--vbd-iterations", type=int, default=20, help="VBD iterations per substep (proxy).")
        parser.add_argument("--proxy-iterations", type=int, default=1, help="Proxy relaxation passes per substep.")
        parser.add_argument("--mass-scale", type=float, default=5.0, help="Proxy body mass scale in VBD.")
        parser.add_argument(
            "--coupling-mode", type=str, choices=["lagged", "staggered"], default="lagged", help="Proxy transfer mode."
        )
        parser.add_argument("--mujoco-iterations", type=int, default=100, help="MuJoCo iterations (proxy).")
        parser.add_argument("--mujoco-ls-iterations", type=int, default=20, help="MuJoCo line-search iterations.")
