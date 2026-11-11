# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
import weakref
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from ...core.types import override
from ...exceptions import NewtonDeprecationWarning, NewtonWarning
from ...geometry import GeoType, ParticleFlags, ShapeFlags
from ...geometry.soft_contacts_mesh import _make_soft_contact_filter_data
from ...geometry.tri_mesh_collision import (
    TriMeshCollisionDetector,
)
from ...sim import (
    BodyFlags,
    CollisionPipeline,
    Contacts,
    Control,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    StateFlags,
    TendonGuideFlags,
)
from ...sim.collide import _count_soft_particle_rigid_contact_pairs
from ...sim.joint_mimic import has_supported_joint_mimics
from ...utils import is_graph_capture_allocation_enabled
from ...utils.mesh import MeshAdjacencyData, build_vertex_adjacency_with_warp
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from ..tendon_kernels import solve_tendon_material, update_tendon_attachments
from ..tendon_state import TendonStateMixin
from ..xpbd import kernels as xpbd_kernels
from ..xpbd.kernels import apply_joint_forces, project_joint_mimics
from . import particle_vbd_kernels, rigid_vbd_kernels, vbd_coupling_kernels
from .cut_kernels import (
    CUT_GATE_BIT,
    CUT_GHOST_GATE_BIT,
    accumulate_cut_blade_force,
    apply_start_gap_offset,
    begin_cut_update,
    build_cut_row_sources,
    build_padded_tet_adjacency,
    clear_cut_gate_bit,
    compute_start_gap_offset,
    decide_cut_advance,
    mark_cut_faces,
    mark_pressed_vertices,
    refresh_edge_rows,
    refresh_triangle_rows,
    split_cut_vertices,
    take_surface_rows,
)
from .cut_xfem_kernels import (
    XFEM_EDGE_WOUND,
    XFEM_LEVEL_UNSET,
    XFEM_NO_ROW,
    XFEM_ROW_FIXED,
    XFEM_ROW_PARKED,
    XFEM_ROW_SKIN,
    XFEM_ROW_WOUND,
    XFEM_WOUND_EDGES_PER_TET,
    XFEM_WOUND_TRIS_PER_TET,
    accumulate_cut_tie_force_and_hessian,
    apply_wound_mass_take,
    build_cut_mesh_edges,
    check_cut_demand,
    count_cut_edge_demand,
    count_cut_edge_row_demand,
    count_cut_triangle_row_demand,
    cut_crossed_tets,
    filter_cut_self_contacts,
    finish_wound_particles,
    make_wound_particles,
    mark_crossed_tets,
    mark_cut_front_edges,
    set_cut_edge_parameters,
    set_cut_level_values,
    split_cut_tet_vertices,
    take_cut_skin_edge_rows,
    take_cut_skin_triangle_rows,
    update_cut_stroke,
    update_duals_cut_ties,
    write_cut_edge_rows,
    write_cut_triangle_rows,
)
from .particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
    # Topological filtering helper functions
    accumulate_body_particle_attachment_force_and_hessian,
    accumulate_self_contact_force_and_hessian,
    accumulate_spring_force_and_hessian,
    # Planar DAT (Divide and Truncate) kernels
    apply_planar_truncation_parallel_by_collision,
    apply_truncation_ts,
    build_particle_body_contact_adjacency_active,
    # Solver kernels (particle VBD)
    forward_step,
    gather_particle_body_contact_force_and_hessian,
    make_solve_elasticity_tile,
    reset_particle_state,
    scatter_particle_body_contact_force_and_hessian,
    solve_elasticity,
    step_particle_body_attachment_lambda_rho,
    update_velocity,
)
from .rigid_vbd_kernels import (
    _NUM_CONTACT_THREADS_PER_BODY,
    CONTACT_HISTORY_NO_BUFFER,
    RigidContactHistory,
    RigidForceElementAdjacencyInfo,
    _count_body_particle_attachments_per_body,
    _fill_body_particle_attachments_per_body,
    accumulate_body_body_contacts_per_body,
    accumulate_body_particle_attachments_per_body,
    accumulate_body_particle_contacts_per_body,
    apply_body_truncation_ts,
    apply_rigid_soft_truncation,
    build_body_body_contact_lists,
    build_body_particle_contact_lists,
    check_contact_overflow,
    compute_body_body_contact_forces,
    compute_body_particle_contact_forces,
    compute_rigid_contact_forces,
    compute_rod_dahl_parameters,
    create_solve_rigid_body,
    forward_step_rigid_bodies,
    init_body_body_contact_materials,
    init_body_body_contacts_alm,
    init_body_particle_contacts,
    init_rod_rest_bend_twist,
    refresh_body_structural_k,
    refresh_joint_material_params,
    reset_particle_body_attachment_state,
    reset_rigid_state,
    select_variant_bounding_radius,
    snapshot_body_body_contact_history,
    step_body_body_contact_C0_lambda,
    step_joint_C0_lambda_rho,
    update_body_velocity,
    update_duals_body_body_contacts,
    update_duals_body_particle_contacts,
    update_duals_joint,
    update_duals_particle_body_attachments,
    update_rod_dahl_state,
)
from .suture_kernels import (
    SuturePointData,
    SutureRodData,
    accumulate_suture_row_force_and_hessian,
    accumulate_suture_rows_per_body,
    accumulate_suture_tip_force,
    begin_suture_update,
    build_tet_neighbors,
    initialize_suture_points,
    refresh_suture_points,
    update_duals_suture_points,
    update_suture_gate,
    update_suture_needle_state,
)
from .tendon_kernels import (
    TendonForceElementAdjacencyInfo,
    snapshot_tendon_segment_length_reference,
    update_tendon_segment_diagnostics,
)
from .vbd_coupling_kernels import (
    _harvest_vbd_body_particle_contact_forces_on_proxy_bodies_kernel,
    _harvest_vbd_proxy_particle_body_contact_forces_kernel,
    _harvest_vbd_proxy_particle_self_contact_forces_kernel,
    _harvest_vbd_proxy_wrenches_kernel,
    _update_vbd_body_input_state_kernel,
)

logger = logging.getLogger(__name__)

__all__ = ["SolverVBD"]

_PARTICLE_CONTACT_GATHER_BLOCK_DIM = 128

# ``NewtonVbdSceneAPI`` settings: constructor argument -> (dtype, default, USD attribute name).
# Read by both ``register_custom_attributes`` and ``SolverVBD.__init__``.
# Per-particle tissue parameters of the suturing feature: grip force per unit tunnel length [N/m],
# tunnel friction coefficient, puncture force [N] and the cosine of the puncture angle.
_SUTURE_PARTICLE_ATTRIBUTES = (
    ("suture_grip", 1.0),
    ("suture_mu", 0.3),
    ("suture_puncture_force", 0.5),
    ("suture_puncture_cos", 0.5),
    ("suture_stiffness", float("inf")),
)

# Per-particle tissue parameter of the cutting feature: the blade force [N] that cuts the tissue at this
# particle. Infinite (the default) marks tissue that cannot be cut.
_CUT_PARTICLE_ATTRIBUTES = (("cut_force", float("inf")),)

# The six edges of a tet as local vertex pairs, for the mean edge length of the spacing rule.
_TET_EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# The XFEM-type cut warns at construction when the largest cut force is above this multiple of the
# largest force of one contact record of the blade edge with the penalty law (contact stiffness
# times collision radius). Measured with a record limit of 0.25 N and a cut force of 0.5 N: two
# head-on records held the blade at 0.478 N (1.9 record limits) and the cut stopped; with the blade
# turned 10 degrees the blade stayed at 0.38 to 0.40 N (1.5 to 1.6 record limits) in 1 of 11 runs.
# So a cut force above one record limit needs several records at the cut front and can stop.
_XFEM_BLADE_RECORD_LIMIT_FACTOR = 1.0

_SCENE_ATTRIBUTES = {
    "rigid_contact_history": (wp.bool, False, "newton:vbd:rigid:contactHistory"),
    "rigid_joint_linear_ke": (wp.float32, 1.0e5, "newton:vbd:rigid:jointLinearStiffness"),
    "rigid_joint_angular_ke": (wp.float32, 1.0e5, "newton:vbd:rigid:jointAngularStiffness"),
    "rigid_joint_linear_kd": (wp.float32, 0.0, "newton:vbd:rigid:jointLinearDamping"),
    "rigid_joint_angular_kd": (wp.float32, 0.0, "newton:vbd:rigid:jointAngularDamping"),
}


def _select_rigid_block_dim(launch_dim: int, device: wp.Device) -> int:
    """Expose more blocks on small rigid launches, allowing sub-warp blocks.

    Target as many blocks as SMs, with a four-thread minimum and a 256-thread
    maximum. This empirical tradeoff does not guarantee occupancy.
    """
    if not device.is_cuda:
        return 256

    target_blocks = device.sm_count
    block_dim = 4
    while block_dim < 256:
        candidate = 2 * block_dim
        if (launch_dim + candidate - 1) // candidate < target_blocks:
            break
        block_dim = candidate
    return block_dim


def _particle_contact_scatter_worker_count(soft_contact_max: int, particle_count: int, device) -> int:
    """Return the host-static worker grid for the per-record body-particle contact accumulation.

    Enough workers to fill the device (two 256-thread blocks per SM) or one per particle, whichever
    is larger, but never more than the contact capacity. Each worker strides over the active contact
    prefix, so the launch size is fixed for CUDA-graph capture while the work follows the active count.
    """
    sm_count = device.sm_count if device.is_cuda else 1
    return max(1, min(soft_contact_max, max(2 * 256 * sm_count, particle_count)))


def _is_tet_only_elasticity_model(model: Model) -> bool:
    """Return whether the model's active element materials are tetrahedral only."""
    if model.tet_count == 0:
        return False

    if model.tri_count > 0:
        tri_materials = model.tri_materials.numpy()
        if np.any((tri_materials[:, 0] > 0.0) | (tri_materials[:, 1] > 0.0)):
            return False

    if model.edge_count > 0:
        edge_bending_properties = model.edge_bending_properties.numpy()
        if np.any(edge_bending_properties[:, 0] > 0.0):
            return False

    return True


def _validate_compliant_alm_material_coefficient(
    values: Any,
    name: str,
    *,
    allow_negative: bool = False,
) -> None:
    """Require material coefficients representable by solver float storage."""
    values = np.asarray(values)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    if not allow_negative and np.any(values < 0.0):
        raise ValueError(f"{name} must contain nonnegative values")
    if np.any(values > np.finfo(np.float32).max):
        raise ValueError(f"{name} must not exceed the float32 maximum")


def _rigid_lambda_retention(alpha: float, gamma: float, use_compliant_alm: bool) -> float:
    """Resolve cross-step multiplier retention without coupling ALM state to stabilization.

    Compliant ALM uses ``gamma`` because alpha only defines the stabilized
    residual. The legacy path keeps its historical ``alpha * gamma`` policy.
    """
    return gamma if use_compliant_alm else alpha * gamma


class SolverVBD(TendonStateMixin, SolverBase, CouplingInterface):
    """An implicit solver using Vertex Block Descent (VBD) for particles and Augmented VBD (AVBD) for rigid bodies.

    .. experimental::
        SolverVBD's public API and behavior may change without prior notice.

    Tuning guide: :ref:`Tuning VBD`
    (https://newton-physics.github.io/newton/stable/concepts/simulation_tuning_vbd.html)
    covers cloth, soft bodies, rigid bodies, cables, and their contacts:

    - Parameter effects: the constructor arguments that matter most for tuning,
      the model parameters, and the time step, substeps and iterations.
    - Contact handling: interaction distance, detection slack, query radius and motion
      budget; what DAT does to motion; how to set margin and gap.
    - A debug protocol: the jitter pipeline and the explosion checklist, each in order.
    - Special topics with symptom, cause and fix: the stiffness ratio (contact against
      elasticity, membrane against bending), a fine mesh converging slowly, DAT damping,
      bad assets, a smaller time step making things worse, self-contact tunneling.
    - Parameter combinations from the shipped examples, and rules of thumb.
    - Rigid bodies and cables: joint gains, cable materials, contact history and
      capacity, model updates, and reset.

    The math behind the advice (step objective, block solve, rigid bodies and joints, the
    energy terms, DAT) is on the solver page :doc:`/solvers/vbd`.

    This unified solver supports:
        - Particle simulation (cloth, soft bodies) using the VBD algorithm
        - Rigid body simulation (joints, contacts) using the AVBD algorithm
        - Coupled particle-rigid body systems
        - Compliant and hard rigid-body-to-particle attachments

    For rigid bodies, two paths are supported:

    - **Compliant ALM** (``rigid_compliant_alm=True``, default): one
      finite-material formulation for structural joints, drives, limits, and
      body-body contacts. Authored finite stiffness controls physical compliance,
      while ``SolverVBD`` selects an internal ALM metric ``rho`` for numerical
      conditioning. For a material row, they combine as
      ``k_eff = k*rho/(k+rho)``, and a multiplier ``lambda`` carries accumulated
      reaction. The per-slot joint **hard/soft** API (authored via
      ``model.vbd.joint_is_hard`` or :meth:`set_joint_constraint_mode`) is
      deprecated. ``rigid_contact_hard`` is a separate legacy-contact control and
      has no formulation effect under compliant ALM. Both distinctions will be
      removed with the legacy path.
    - **Legacy AVBD** (``rigid_compliant_alm=False``, deprecated): penalty stiffness
      that is fixed by default (``rigid_avbd_beta=0``) or ramped per iteration from
      ``k_start`` seeds, where non-rod joint slots default to hard mode (augmented
      Lagrangian with persistent lambda and C0 stabilization) and rod stretch,
      shear, bend, and twist default to soft (penalty-based). Deprecated as of
      Newton 1.6 and will be removed in a future release; select it explicitly
      with ``rigid_compliant_alm=False``.

    Joint limitations:
        - Supported joint types: BALL, FIXED, FREE, REVOLUTE, PRISMATIC, D6, ROD.
          DISTANCE joints are not supported.
        - :attr:`~newton.Model.joint_enabled` is supported for all joint types and
          is read live. After changing enable flags, call
          :meth:`notify_model_changed` with
          :attr:`~newton.ModelFlags.JOINT_PROPERTIES` to refresh derived contact
          conditioning. Structural-slot material (``rigid_joint_linear_ke``/
          ``rigid_joint_angular_ke``), constraint layout, and non-ROD drive/limit
          rest-angle offsets are captured at construction; rebuild ``SolverVBD``
          after changing them.
        - :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd` are supported
          for REVOLUTE, PRISMATIC, D6 (as drives), and ROD (as stretch, shear,
          bend, and twist stiffness and damping).
          VBD interprets ``kd`` as absolute damping in physical units. ROD values are cached in
          solver-owned buffers at construction; after editing them call
          :meth:`notify_model_changed` with
          :attr:`~newton.ModelFlags.JOINT_DOF_PROPERTIES` to refresh that cache. REVOLUTE,
          PRISMATIC, and D6 drive/limit coefficients are gathered live from the model on every
          solve and need no notification -- except on the deprecated legacy AVBD path, whose
          cached penalty state the same flag refreshes.
        - :attr:`~newton.Model.joint_limit_lower`/:attr:`~newton.Model.joint_limit_upper` and
          :attr:`~newton.Model.joint_limit_ke`/:attr:`~newton.Model.joint_limit_kd` are supported
          for REVOLUTE, PRISMATIC, and D6 joints.
        - Angular winding is retained for REVOLUTE joints and D6 joints with
          exactly one angular DOF. Tracking requires less than ``pi`` radians
          of relative rotation per solver step, including prediction and
          solver corrections.
        - :attr:`~newton.Control.joint_f` (feedforward forces) is supported,
          including six-component world-frame wrenches for ROD joints.
        - Not supported: :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_velocity_limit`,
          :attr:`~newton.Model.joint_target_mode`, equality constraints, and the deprecated sparse mimic constraints.
        - Joint-owned mimic relationships are supported for PRISMATIC, REVOLUTE, and D6 joints.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Tendon limitations:
        - Static routes and isolated dynamic ROLLER guides are supported. Zero
          segment compliance is approximated as ``1.0e-8`` m/N.
        - Call :meth:`newton.ModelBuilder.color` after adding tendons so that
          segment endpoints receive different rigid-body colors.

    Particle-body attachment limitations:
        - Attachments are translational and constrain one particle to a body-local point.
        - Attachments are compliant-ALM rows with a limit-row penalty: a finite
          ``stiffness`` keeps its physical spring offset under load, while infinite
          stiffness is the hard limit of the same row and ignores ``damping``.
          Multipliers are retained across steps with ``rigid_avbd_gamma``. Finite
          iterations can leave residual error, especially for conflicting constraints.
        - Both endpoints must be integrated by this solver. Attachments are not supported
          with ``integrate_with_external_rigid_solver=True``.

        See :ref:`Particle-body attachments` for authoring and cross-solver behavior.

    Buffer sizing:
        Body-body contact state is pre-allocated from ``model.rigid_contact_max`` when a
        :class:`~newton.CollisionPipeline` has already published it and this solver owns the
        rigid system. Body-particle contact state is pre-sized from a world-aware
        particle-shape pair count, which excludes the
        ``enable_rigid_soft_full_surface_contact`` edge/face headroom. Both grow from
        ``Contacts`` on the first :meth:`step`, and the rigid contact force outputs grow in
        :meth:`collect_rigid_contact_forces`. During graph capture,
        ordinary lazy resizing is supported on CPU and on CUDA with Warp's
        stream-ordered memory pool enabled; otherwise the solver raises with
        guidance to pre-size before capture. Rigid contact history is
        cross-replay-persistent state, so it must always be allocated before
        capture regardless of the device's allocation-during-capture support --
        allocating it inside a graph records a `wp.zeros` fill that wipes the
        warm-start buffers on every replay. With ``rigid_contact_history=True``,
        construct :class:`~newton.CollisionPipeline` before ``SolverVBD``, or run
        one uncaptured solver step before capture.

    Contact force observables:
        Request :attr:`~newton.solvers.SolverBase.ObservableKind.CONTACT_F` with :meth:`observables`
        and pass the container to :meth:`step` to evaluate one wrench per body-body contact
        (when this solver integrates the rigid bodies) and per rigid-soft contact record --
        particle, edge, and face -- at the final configuration. The step writes directly into
        :attr:`~newton.solvers.SolverBase.Observables.contact_f` without changing the simulation result.
        Unused rows, including rigid rows owned by an external rigid solver, are zero.
        Soft self-contact forces are not reported. See :ref:`vbd_contact_forces` for the wrench
        convention and a usage guide.

    References:
        - Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block Descent. ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages.
          https://doi.org/10.1145/3658179
        - Chris Giles, Elie Diaz, and Cem Yuksel. 2025. Augmented Vertex Block Descent. ACM Trans. Graph. 44, 4, Article 90 (August 2025), 12 pages.
          https://doi.org/10.1145/3731195
        - Anka H. Chen, Jerry Hsu, Youssef Ayman, and Miles Macklin. 2026. Divide and Truncate: A Penetration and Inversion Free Framework for Coupled Multi-physics System. In ACM SIGGRAPH 2026 Conference Papers.
          https://doi.org/10.1145/3799902.3811143
        - Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel. 2025. Offset Geometric Contact. ACM Trans. Graph. 44, 4.
          https://doi.org/10.1145/3731205

    Note:
        `SolverVBD` requires coloring for each system it solves:

        - Particle coloring: :attr:`newton.Model.particle_color_groups` (required if particles are present)
        - Rigid body coloring: :attr:`newton.Model.body_color_groups` (required if rigid bodies are integrated by VBD)

        Call :meth:`newton.ModelBuilder.color` to automatically color both particles and rigid bodies.

        For non-ROD rigid joints, VBD uses ``model.body_q`` as the structural
        rest pose and reads ``model.joint_q`` for drive/limit rest-angle
        offsets. The body transforms must match the joint angles at solver
        creation time (see example below).

        For ROD joints, SolverVBD captures structural-rest bend and twist from
        ``model.joint_rest_q`` when constructed. After changing these Model-owned
        rest rotations, call :meth:`notify_model_changed` with
        :attr:`~newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES`; the broader
        :attr:`~newton.ModelFlags.JOINT_DOF_PROPERTIES` also refreshes it.

        For CUDA graph capture, the recommended construction order is
        ``CollisionPipeline`` -> ``Contacts`` -> ``SolverVBD``, all before capture.

    Example
    -------

    .. code-block:: python

        # Automatically color both particles and rigid bodies
        builder.color()

        model = builder.finalize()

        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        solver = newton.solvers.SolverVBD(model)

        # Initialize states and control
        state_in = model.state()
        state_out = model.state()
        control = model.control()

        # Simulation loop
        for i in range(100):
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    supports_collision_pipeline = True

    SUPPORTED_OBSERVABLES = frozenset({SolverBase.ObservableKind.CONTACT_F})

    class JointSlot:
        """Named constraint slot indices for :meth:`set_joint_constraint_mode`.

        Structural constraint slots by joint type:
          - ROD: STRETCH=0, SHEAR=1, BEND=2, TWIST=3
          - BALL: LINEAR=0 only
          - FIXED/REVOLUTE/PRISMATIC/D6: LINEAR=0, ANGULAR=1

        STRETCH/SHEAR/BEND/TWIST name SolverVBD's four-slot constraint layout
        and apply only to :attr:`~newton.JointType.ROD`. Only structural
        slots are named here; per-DOF drive/limit slots (slot 2+ on non-rod
        joints) are not.

        These are solver constraint slots, not offsets into
        :attr:`newton.Model.joint_target_ke` or
        :attr:`newton.Model.joint_target_kd`. Rod material coefficients start at
        :attr:`newton.Model.joint_qd_start` in the order
        ``[shear_x, shear_y, stretch_z, bend_x, bend_y, twist_z]``; see
        :ref:`Rod joints`.
        """

        # Non-rod structural slots.
        LINEAR = 0
        ANGULAR = 1
        # Rod structural slots (all four are linear/angular rod constraints; they
        # are not the non-rod LINEAR/ANGULAR despite STRETCH sharing index 0).
        STRETCH = 0
        SHEAR = 1
        BEND = 2
        TWIST = 3

    def __init__(
        self,
        model: Model,
        *,
        # Common parameters
        iterations: int | None = None,
        friction_epsilon: float = 1e-2,
        dat_conservative_bound_relaxation: float = 0.85,
        integrate_with_external_rigid_solver: bool = False,
        # Particle parameters
        particle_enable_self_contact: bool = False,
        particle_self_contact_radius: float | None = None,
        particle_self_contact_margin: float | None = None,
        particle_self_contact_gap: float | None = None,
        particle_conservative_bound_relaxation: float | None = None,
        particle_vertex_contact_buffer_size: int = 32,
        particle_edge_contact_buffer_size: int = 64,
        particle_collision_detection_interval: int | None = None,
        particle_edge_parallel_epsilon: float = 1e-5,
        particle_enable_tile_solve: bool = True,
        particle_topological_contact_filter_threshold: int = 2,
        particle_rest_shape_contact_exclusion_radius: float = 0.0,
        particle_external_vertex_contact_filtering_map: dict | None = None,
        particle_external_edge_contact_filtering_map: dict | None = None,
        # Rigid body - constraint formulation and stabilization
        rigid_compliant_alm: bool = True,
        rigid_avbd_alpha: float | None = None,  # Shared alpha override; None uses mode defaults
        rigid_avbd_joint_alpha: float | None = None,  # Joint alpha override
        rigid_avbd_contact_alpha: float | None = None,  # Body-body contact alpha override
        rigid_avbd_beta: float | None = None,  # Legacy AVBD penalty ramp rate per iteration
        rigid_avbd_linear_beta: float | None = None,  # Legacy linear beta override
        rigid_avbd_angular_beta: float | None = None,  # Legacy angular beta override
        rigid_avbd_gamma: float = 0.999,  # Per-step decay for persisted lambda (and legacy penalty k)
        # Rigid body - contacts
        rigid_contact_hard: bool | None = None,  # Legacy body-body contact hard/soft mode
        rigid_contact_history: bool | None = None,  # Body-body contact numeric warm-start
        rigid_contact_k_start: float | None = None,  # Legacy AVBD contact penalty ramp seed
        rigid_body_contact_buffer_size: int = 64,  # Per-body body-body contact list capacity
        rigid_body_particle_contact_buffer_size: int = 256,  # Per-body soft-contact list capacity (particle + edge/face)
        rigid_soft_contact_use_log_barrier: bool = False,  # Use particle-style normal log barrier
        # Rigid body - joints
        rigid_joint_linear_ke: float | None = None,  # Structural linear joint stiffness
        rigid_joint_angular_ke: float | None = None,  # Structural angular joint stiffness
        rigid_joint_linear_k_start: float | None = None,  # Legacy AVBD linear joint penalty ramp seed
        rigid_joint_angular_k_start: float | None = None,  # Legacy AVBD angular joint penalty ramp seed
        rigid_joint_linear_kd: float | None = None,  # Absolute damping for non-rod linear joint constraints
        rigid_joint_angular_kd: float | None = None,  # Absolute damping for non-rod angular joint constraints
        # Tendons
        tendon_max_sweeps: int = 256,
        tendon_settle_tol: float = 1.0e-3,
        tendon_activation_tol: float = 2.0e-3,
        # Rigid body - penetration-free DAT truncation
        rigid_soft_enable_dat: bool = False,  # Truncate rigid pose updates against per-contact division planes
        rigid_soft_dat_use_interval_arithmetic: bool = False,  # Experimental interval verification of rigid trajectories
        # Suturing (experimental): suture needles authored with ModelBuilder.add_suture_needle
        suture_tunnel_update_interval: int = 1,  # Substeps between needle state updates
        suture_point_spacing: float
        | None = None,  # Tunnel point spacing [m]; None = clamp(0.5 x mean tet edge, 0.5 mm, 2 mm)
        suture_gate_slack: float = 1.0e-3,  # Extra gate radius beyond needle radius + margin + gap + particle radius [m]
        suture_armed: bool = False,  # True: the first tip contact punctures; False: force and angle test
        suture_closest_search_window: int = 1,  # Rod segments searched on each side of the cached segment per refresh
        suture_friction_transverse_weight: float = 0.0,  # Weight of the transverse multiplier in the friction cap (1 = contact law)
        suture_smooth_tangent: bool = False,  # Continuous rod tangent for the rows (bisector at joints) instead of the segment direction
        # Cutting (experimental): cut blades authored with ModelBuilder.add_cut_blade and add_cut_capacity
        cut_armed: bool = False,  # True: the first blade edge contact with force cuts; False: force test
        cut_update_interval: int = 1,  # Substeps between two cut updates
        cut_start_gap: float
        | None = None,  # Distance between the two wound faces right after a split [m]; None = self-contact margin
        cut_contact_range: str | float = "detection",  # "detection", "force" or a value [m]
        cut_keep_side_labels: bool = True,  # Keep the side label of a tet for the stroke
        cut_gap_direction: str = "largest_least_opening",  # or "normal_sum"
        cut_mass_rule: str = "rest_volume",  # or "full_copy"
        cut_surface_edges: bool = True,  # Wound faces get edge rows
        cut_press_wound: bool = False,  # Fork: wound particles inside the blade span count as pressed
        cut_gate_wound: bool = False,  # Fork: the blade has no contact with split vertices and copies
        cut_method: str = "faces",  # "faces": along tet faces; "xfem": the XFEM-type cut, through the tets where the blade is
        cut_tie_stiffness: float | None = None,  # Tie stiffness of a wound particle [N/m]; None = 2 x soft_contact_ke
        cut_tie_dual: bool = False,  # Multiplier update of the tie, once per iteration
        cut_xfem_min_fraction: float = 0.15,  # Smallest distance of a wound particle from an edge end, as an edge share
        cut_wound_mass_fraction: float = 0.05,  # Mass share that the new wound particles take from a tie corner
        cut_xfem_contact_filter: bool = True,  # No self contact between near rows of the same side of a cut
        cut_xfem_front_rows: bool = False,  # Fork: an edge row between the two wound particles of a front edge
        deterministic: wp.DeterministicMode | None = None,
        enable_fast_math: bool = True,
        collision_pipeline: CollisionPipeline | None = None,
        collision_frequency: Mapping[SolverBase.CollisionSlot, int] | None = None,
        collision_frequency_type: Mapping[SolverBase.CollisionSlot, SolverBase.CollisionFrequencyType] | None = None,
    ):
        """
        Args:
            model: The `Model` object used to initialize the integrator. Must be identical to the `Model` object passed
                to the `step` function.

            iterations: Number of VBD iterations per step. If ``None``, use the registered
                ``newton:maxSolverIterations`` scene value, or ``10`` if absent or ``-1``.
                Explicit values override the scene.
            friction_epsilon: Threshold to smooth small relative velocities in friction computation (used for both particle
                and rigid body contacts).
            dat_conservative_bound_relaxation: Relaxation factor in (0, 1) shared by every Divide-and-Truncate
                (DAT) truncation: soft self-contact (``particle_enable_self_contact``) and rigid-soft
                (``rigid_soft_enable_dat``). It scales every truncation scalar, and each side's motion
                budget between detections is 0.5 x relaxation x the detection query radius.
            integrate_with_external_rigid_solver: Indicator for coupled rigid body-cloth simulation. When set to `True`,
                the solver assumes rigid bodies are integrated by an external solver (one-way coupling).

            particle_enable_self_contact: Whether to enable self-contact detection for particles.
                Requires an active soft self-contact collision schedule: ``CollisionFrequencyType.NONE``
                for the ``SOFT_SELF_CONTACT`` slot raises at :meth:`step`, because the self-contact
                trajectory reference is refreshed only when detection runs.
            particle_self_contact_radius: Deprecated; use ``particle_self_contact_margin`` +
                ``particle_self_contact_gap`` instead. When set, the legacy interpretation applies
                exactly: radius = interaction distance and ``particle_self_contact_margin`` = detection
                query radius.
            particle_self_contact_margin: Self-contact interaction distance [m] — the surface offset at
                which vertex-triangle and edge-edge pairs start to interact. Defaults to 0.2.
                (Legacy meaning — the detection query radius — applies only while the deprecated
                ``particle_self_contact_radius`` is set or when ``particle_self_contact_gap`` is omitted.)
            particle_self_contact_gap: Additional detection-only distance [m]; self-contact detection
                queries use ``margin + gap``, mirroring the ``ShapeConfig.margin`` / ``gap`` convention.
                Defaults to 0. Give it ~0.5-1x the margin of slack to avoid missing contacts.
            particle_conservative_bound_relaxation: Deprecated; use ``dat_conservative_bound_relaxation``
                instead. When set, it overrides ``dat_conservative_bound_relaxation``.

                .. deprecated:: 1.7
            particle_vertex_contact_buffer_size: Preallocation size for each vertex's vertex-triangle collision
                buffer. Pairs beyond this capacity are silently dropped during detection.
            particle_edge_contact_buffer_size: Preallocation size for each edge's edge-edge collision buffer. Pairs
                beyond this capacity are silently dropped during detection.
            particle_collision_detection_interval: Deprecated; use the self-contact slot of
                ``collision_frequency`` / ``collision_frequency_type`` instead.
                Controls how frequently particle self-contact detection is applied
                during the simulation. If set to a value < 0, collision detection is only performed once before the
                initialization step. If set to 0, collision detection is applied twice: once before and once immediately
                after initialization. If set to a value `n` >= 1, collision detection is applied before every `n` VBD
                iterations.
            particle_edge_parallel_epsilon: Threshold to detect near-parallel edges in edge-edge collision handling.
            particle_enable_tile_solve: Whether to accelerate the particle solver using tile API.
                The tiled kernel is specialized once at construction from the model's element
                materials (e.g. a tetrahedra-only model compiles without triangle/edge code paths);
                rebuild the solver after changing triangle or edge stiffness.
            particle_topological_contact_filter_threshold: Maximum topological distance (measured in rings) under which candidate
                self-contacts are discarded. Set to a higher value to tolerate contacts between more closely connected mesh
                elements. Only used when `particle_enable_self_contact` is `True`. Note that setting this to a value larger than 3 will
                result in a significant increase in computation time.
            particle_rest_shape_contact_exclusion_radius: Additional world-space distance threshold for filtering topologically close
                primitives. Candidate contacts with a rest separation shorter than this value are ignored. The distance is
                evaluated in the rest configuration conveyed by `model.particle_q`. Only used when `particle_enable_self_contact` is `True`.
            particle_external_vertex_contact_filtering_map: Optional dictionary used to exclude additional vertex-triangle pairs during
                contact generation. Keys must be vertex primitive ids (integers), and each value must be a `list` or
                `set` containing the triangle primitives to be filtered out. Only used when `particle_enable_self_contact` is `True`.
            particle_external_edge_contact_filtering_map: Optional dictionary used to exclude additional edge-edge pairs during contact
                generation. Keys must be edge primitive ids (integers), and each value must be a `list` or `set`
                containing the edges to be filtered out. Only used when `particle_enable_self_contact` is `True`.

            rigid_compliant_alm: Unified compliant-ALM mode for body-body contacts,
                structural joints, drives, and limits. This is the default path.
                Defaults to ``True``. Pass ``False`` to keep the deprecated legacy
                penalty/AVBD path during its migration window. Finite authored
                coefficients define the material response, while ``SolverVBD`` selects
                ``rho`` internally for numerical conditioning. Values used with legacy
                hard constraints may require retuning for the desired deformation.
                Values must be finite and representable in float32; infinity is unsupported.

                Explicit non-``None`` values for deprecated beta, hard-contact, or
                penalty-seed controls emit a :class:`~newton.exceptions.NewtonDeprecationWarning`.

                .. deprecated:: 1.6
                    The legacy path selected by ``False`` will be removed in a future
                    release. Explicit ``False`` emits a
                    :class:`~newton.exceptions.NewtonDeprecationWarning` for all models.
                    Use the default compliant ALM path instead.
            rigid_avbd_alpha: C0 stabilization strength (``C_stab = C - alpha * C0``). Range: [0, 1].
                Controls both joints and body-body contacts when neither class-specific
                override (``rigid_avbd_joint_alpha`` / ``rigid_avbd_contact_alpha``) is set.
                ``None`` leaves mode defaults: compliant ALM uses ``0.0`` for both
                constraint classes (the raw material residual), while the legacy path
                uses ``0.95``.
            rigid_avbd_joint_alpha: Joint-specific alpha override. ``None`` falls back to
                ``rigid_avbd_alpha`` when set, otherwise to ``0.0`` under compliant ALM or
                ``0.95`` on the legacy path.
            rigid_avbd_contact_alpha: Body-body contact alpha override. ``None`` falls back
                to ``rigid_avbd_alpha`` when set, otherwise ``0.0`` under compliant ALM (authored
                contact stiffness applies to the raw residual) or ``0.95`` on the legacy path.
                Under compliant ALM, alpha is stabilization only; retention is set separately by
                ``rigid_avbd_gamma``.
            rigid_avbd_beta: Legacy AVBD penalty ramp rate per iteration. ``None`` (default)
                uses ``0``, disabling ramping (fixed-k). Set to e.g. ``1e5`` for ramping. Used for both
                linear and angular constraints unless overridden. Does not tune the
                internal compliant-ALM ``rho`` for converted rigid rows. Note: linear
                (meters) and angular (radians) constraints have different units, so the
                overrides should be used for production tuning.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated for all uses. Body-particle contacts
                    continue to honor this control during migration; keep the effective beta
                    at ``0`` (the default behavior) and author fixed material stiffness instead.
            rigid_avbd_linear_beta: Legacy linear beta override for linear constraints
                (meters). ``None`` (default) uses ``rigid_avbd_beta``. Does not tune
                compliant-ALM ``rho``.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated for all uses. Body-particle contacts
                    continue to honor this control during migration; keep the effective beta
                    at ``0`` (the default behavior) and author fixed material stiffness instead.
            rigid_avbd_angular_beta: Legacy angular beta override for angular constraints
                (radians). ``None`` (default) uses ``rigid_avbd_beta``. Does not tune
                compliant-ALM ``rho``.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated. Keep the effective beta at ``0`` (the
                    default behavior) and author fixed material stiffness instead.
            rigid_avbd_gamma: Per-step decay factor for penalty k and persisted lambda.
                Compliant ALM joints, particle-body attachments, and validated contacts
                retain lambda by ``gamma``; the legacy path retains joint and contact
                lambda by ``alpha * gamma``. Lower values discard history faster.
            rigid_contact_hard: Legacy body-body contact hard/soft mode. With
                ``rigid_compliant_alm=True``, contacts use the ALM path. With
                ``rigid_compliant_alm=False``, ``True`` selects legacy hard AVBD contact
                and ``False`` selects legacy penalty-only contact. ``None`` (default) uses ``True``.

                .. deprecated:: 1.6
                    Use the default compliant ALM path and author finite contact stiffness.
            rigid_contact_history: Whether to persist body-body numeric contact state
                across steps using ``Contacts.rigid_contact_match_index``. Compliant ALM
                restores the normal multiplier for matched rows. With latest
                matching it also restores projected tangential multipliers as a
                numerical warm start; with sticky matching, tangential memory is
                represented by the collision pipeline's replayed material anchor.
                Legacy hard contacts restore the full multiplier; legacy soft contacts
                restore penalty k only. History is restored only from the contact set
                the previous step solved: a step on the same contacts (no collision
                pass in between) restores each contact from itself, and the match
                indices are used only when
                ``Contacts.rigid_contact_match_generation`` reports that they refer
                to that set. Contacts start cold after a collision pass into another
                buffer, two passes between steps, or with another buffer. Contact
                geometry remains owned by the collision pipeline. Requires ``CollisionPipeline(contact_matching="latest")`` or ``"sticky"``.
                Ignored when ``integrate_with_external_rigid_solver=True`` or
                ``model.body_count == 0``. During graph capture, construct the
                collision pipeline before ``SolverVBD`` so history is pre-allocated,
                or run one uncaptured solver step before capture.
                ``None`` (default) reads ``model.vbd.rigid_contact_history`` when
                registered, otherwise uses ``False``.
            rigid_contact_k_start: Body-body and body-particle contact penalty seed for
                legacy AVBD ramping [N/m]. Used when ``rigid_avbd_linear_beta`` (or
                ``rigid_avbd_beta`` fallback) is greater than zero. When the linear beta
                is 0, k is fixed at the contact stiffness regardless of this value.
                ``None`` (default) uses ``100.0``.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated for all uses. Body-particle contacts
                    continue to honor this control during migration; keep the effective beta
                    at ``0`` (the default behavior) and author fixed contact stiffness instead.
            rigid_body_contact_buffer_size: Max body-body contacts per rigid body for per-body contact lists.
            rigid_body_particle_contact_buffer_size: Capacity of the per-body soft-contact adjacency
                list, including currently ineligible mesh candidates. If a body's list overflows, force
                evaluation and coupling feedback scan all stored soft contacts for that body instead.
                This preserves reactions but can increase evaluation cost.
            rigid_soft_contact_use_log_barrier: Whether body-particle normal contacts use the same C2
                log-barrier law as particle self-contact. When ``False`` (default), they use the legacy quadratic
                penalty law.
            rigid_joint_linear_ke: Material stiffness for non-rod structural linear joint slots [N/m].
                ``None`` (default) reads ``model.vbd.rigid_joint_linear_ke`` when
                registered, otherwise uses ``1.0e5``. A scene-authored value must be finite and
                non-negative, and applies with and without ``rigid_compliant_alm``.
            rigid_joint_angular_ke: Material stiffness for non-rod structural angular joint slots [N·m/rad].
                ``None`` (default) reads ``model.vbd.rigid_joint_angular_ke`` when
                registered, otherwise uses ``1.0e5``. A scene-authored value must be finite and
                non-negative, and applies with and without ``rigid_compliant_alm``.
            rigid_joint_linear_k_start: Linear penalty seed for legacy AVBD ramping [N/m]. Used when
                ``rigid_avbd_linear_beta`` (or ``rigid_avbd_beta`` fallback) is greater than zero.
                When the linear beta is 0, k is fixed at the joint stiffness regardless of this value.
                ``None`` (default) uses ``100.0``.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated. Keep the effective beta at ``0`` (the
                    default behavior) and author fixed joint stiffness instead.
            rigid_joint_angular_k_start: Angular penalty seed for legacy AVBD ramping [N·m/rad]. Used when
                ``rigid_avbd_angular_beta`` (or ``rigid_avbd_beta`` fallback) is greater than zero.
                When the angular beta is 0, k is fixed at the joint stiffness regardless of this value.
                ``None`` (default) uses ``10.0``.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated. Keep the effective beta at ``0`` (the
                    default behavior) and author fixed joint stiffness instead.
            rigid_joint_linear_kd: Damping coefficient for non-rod linear joint constraints [N·s/m].
                ``None`` (default) reads ``model.vbd.rigid_joint_linear_kd`` when
                registered, otherwise uses ``0.0``. Negative explicit values are clamped to 0; a
                scene-authored value must be finite and non-negative. Scene values apply with and
                without ``rigid_compliant_alm``.
            rigid_joint_angular_kd: Damping coefficient for non-rod angular joint constraints [N·m·s/rad].
                ``None`` (default) reads ``model.vbd.rigid_joint_angular_kd`` when
                registered, otherwise uses ``0.0``. Negative explicit values are clamped to 0; a
                scene-authored value must be finite and non-negative. Scene values apply with and
                without ``rigid_compliant_alm``.
            rigid_soft_enable_dat: Whether to apply Divide-and-Truncate (DAT) penetration-free
                truncation to rigid-soft contacts, symmetric to particle self-contact truncation.
                Each rigid-soft contact reported by the owned collision pipeline defines a division
                plane at its detection-time reference configuration; rigid pose updates and particle
                displacements are jointly truncated so neither side crosses it. Rigid rotation makes
                trajectories curved, so crossing times are found by sampling and bisection,
                optionally certified with interval arithmetic. Full-surface mesh contacts use
                complete VT/TV/EE primitives through Contacts.soft_contact_rigid_indices;
                every corner stays on its assigned side of a certified separating plane.
                Requires a solver-owned pipeline (``collision_pipeline=``); each side's motion budget
                between detections is 0.5 x ``dat_conservative_bound_relaxation`` x the minimum
                rigid-soft query radius,
                where a query radius is the pipeline gap plus the soft feature radius and rigid
                shape margin. With
                ``collision_frequency_type`` AUTO, the rigid slot resolves to ``PRE_POST_INIT``
                (detect before and right after initialization), matching the self-contact slot;
                raise the detection frequency (``ITERATIONS``) to widen the per-step motion budget
                for fast bodies. Kinematic bodies move outside the solver and are not truncated.
                The motion budget is fixed at construction from the pipeline's
                ``soft_contact_gap`` and the model's particle radii and shape margins; later
                changes to those values are not reflected. While enabled, the isotropic budget
                bounds the per-detection motion of every dynamic body and particle in the
                model, not only those near soft geometry.

                Full-surface mesh coverage includes every mesh edge and nearby VT/TV/EE pair,
                including rows ineligible for contact forces. It requires initially separated
                geometry; topology, nonzero mesh scales, radii, and motion bounds must remain unchanged. Analytic
                and penetration-recovery contacts retain their point-anchor behavior; this
                is not a general initial-penetration recovery method. When no separating
                plane can be certified, the affected primitives freeze. Contact overflow
                makes DAT coverage incomplete; stop the simulation and increase contact
                capacity if an overflow is reported.
                Sampling and bisection alone can miss a rigid trajectory that crosses and
                returns between samples; enable the experimental interval path to certify
                the complete rigid trajectory.
            rigid_soft_dat_use_interval_arithmetic: Selector for rigid DAT trajectory
                truncation. ``False`` uses sampling and bisection, plus an interval
                derivative bound that releases rows starting inside the empty band only
                when their signed distance is certified nonincreasing; ``True``
                additionally certifies the whole sampled prefix with interval arithmetic
                to detect crossings between sample points. Only used when
                ``rigid_soft_enable_dat`` is ``True``.

                .. experimental::
                    The Stage-2 prefix certification is experimental and may change or be
                    removed; the derivative bound is part of the default path.
            deterministic: Opt-in determinism for this solver's atomic-emitting
                kernel modules. Pass a :class:`warp.DeterministicMode`, or
                ``None`` (default) to inherit the current
                ``wp.config.deterministic`` mode. Body-particle contact forces
                are accumulated per contact record with atomics in
                ``NOT_GUARANTEED`` mode and gathered per particle without
                atomics in every other mode.
            enable_fast_math: Compile the particle kernels, the rigid body
                solve, and the joint dual update with Warp's ``fast_math``
                (approximate division and square root). Results change at the
                floating-point rounding level. Ignored on CPU.
            tendon_max_sweeps: Maximum capstan material-relaxation sweeps per VBD iteration.
            tendon_settle_tol: Relative tension-change tolerance for stopping capstan material relaxation.
            tendon_activation_tol: Relative radius gap that inactive dynamic rollers must cross before activation.
            suture_tunnel_update_interval: Substeps between two updates of the suture needle state (puncture
                test, tet walk, tunnel point creation, contact gate). ``1`` updates every substep.
            suture_point_spacing: Distance of needle tip travel between two tunnel points [m]. ``None``
                uses half the mean tet edge length of the model, clamped to [0.5 mm, 2 mm].
            suture_gate_slack: Slack added to the contact gate radius [m]. The gate radius is the needle
                group's largest capsule radius plus its largest shape margin plus the pipeline's soft
                contact gap plus the largest particle radius plus this slack.
            suture_armed: ``True`` opens a tunnel at the first tip contact with tissue. ``False`` requires
                the tip's summed contact normal force to exceed ``vbd.suture_puncture_force`` while the
                tip direction is within the angle given by ``vbd.suture_puncture_cos`` of the inward
                surface normal.
            suture_closest_search_window: Number of rod segments searched on each side of a tunnel
                point's cached segment when its closest place on the rod curve is refreshed.
            suture_friction_transverse_weight: Weight ``w`` of the transverse multiplier in the tunnel
                friction cap ``mu * (w * |lambda_perp| + g * ds)``. ``1`` is the body-body contact law
                (friction bounded by the normal force); ``0`` (default) bounds friction by the tissue grip
                alone. A positive weight feeds tissue shear back into the cap and can drag tissue along a
                straight needle; see the solver page.
            suture_smooth_tangent: ``True`` gives every tunnel point a tangent that is continuous along
                the rod curve (the bisector of the two capsules at a joint, interpolated along a capsule), so
                the transverse plane of its row does not jump when the closest place crosses a kinked joint.
                ``False`` (default) uses the direction of the closest capsule.
            cut_armed: ``True`` cuts at the first contact of a blade edge that carries force; ``False``
                (default) cuts when the blade force reaches ``vbd.cut_force``.

                .. experimental::

                    Cutting (all ``cut_*`` arguments, :meth:`~newton.ModelBuilder.add_cut_blade` and
                    :meth:`~newton.ModelBuilder.add_cut_capacity`) is experimental and can change
                    without a deprecation period. See the *Cutting* section of the solver page.
            cut_update_interval: Substeps between two cut updates (blade force test, cut faces, split).
            cut_start_gap: Distance between the two wound faces right after a split [m]. ``None`` uses
                the self-contact margin. Each new particle moves half of it into its own material.
                With ``cut_method="xfem"`` it is the distance along the blade plane normal between the
                two wound particles of a crossed mesh edge, at their tie targets. An edge that spans
                less than ``cut_start_gap / (1 - 2 * cut_xfem_min_fraction)`` along that normal
                cannot hold it and gets ``1 - 2 * cut_xfem_min_fraction`` of its span;
                :meth:`cut_report` counts these edges (``start_gap_limited_edges``).
            cut_contact_range: Distance from the blade edge axis within which a tissue vertex counts as
                pressed. ``"detection"`` (default) is the edge radius plus the shape margin, the particle
                radius and the pipeline's ``soft_contact_gap``; ``"force"`` leaves out the gap; a number
                is the distance itself [m].
            cut_keep_side_labels: ``True`` (default) keeps the side of the blade plane that a tet was
                labeled with until the blade edge has no contact record; ``False`` computes the side
                again at every cut update.
            cut_gap_direction: Direction of the start gap. ``"largest_least_opening"`` (default) makes
                the smallest opening over the cut faces of a particle as large as possible;
                ``"normal_sum"`` uses the area-weighted sum of their normals.
            cut_mass_rule: ``"rest_volume"`` (default) shares the mass of a split vertex between the
                original and its copies by the rest volume of their tets; ``"full_copy"`` gives every
                copy the full mass.
            cut_surface_edges: ``True`` (default) gives the wound faces edge rows, so they take part in
                edge-edge self contact; ``False`` leaves vertex-triangle contact only.
            cut_press_wound: ``True`` also counts the wound particles under the blade edge as pressed,
                so every cut update with enough blade force cuts the next tet row from the cut front.
                A fork of the design (the blade rests on the two wound lips); default ``False``.
            cut_gate_wound: ``True`` switches off the contact between the blade body and the split
                vertices and their copies, through the contact gate of the suture feature, so the blade
                edge goes down to the cut front and the cut follows it. A fork of the design; default
                ``False``. With ``cut_method="xfem"`` the gated particles are the virtual corners and
                the wound particles that are not on a front edge.
            cut_method: ``"faces"`` (default) cuts along the existing tet faces nearest to the blade
                plane. ``"xfem"`` is the XFEM-type cut: it cuts through the tets where the blade plane
                is. A cut tet becomes two fragments: its own line of the tet arrays keeps the material
                on side A and a spare tet (``add_cut_capacity(max_tets=...)``) takes the material on
                side B. Far-side corners become virtual corners, and wound particles on the crossed
                mesh edges carry the collision triangles of the two wound faces.

                Rules and limits of ``"xfem"``:

                - ``cut_keep_side_labels``, ``cut_gap_direction``, ``cut_surface_edges`` and
                  ``cut_press_wound`` belong to ``"faces"``. ``cut_press_wound=True`` raises an error;
                  a non-default value of the other three gives a warning and is not used.
                - A cut advance is all or nothing: when the cut capacity cannot hold the spare tets,
                  the wound particles and the extra surface rows of the advance, no tet is cut in
                  that cut update and :meth:`cut_report` counts the overflow.
                - The suture does not know fragments yet. A model with suture needles gives a
                  warning; keep the needles away from cut tissue.
                - ``vbd.cut_force`` is read at every cut update, so a scene can change it after
                  construction, also from not cuttable to cuttable, and :meth:`reset` keeps the
                  values of the tissue. The rules for a cuttable particle (not the end of a spring,
                  the radius of the spare particles) are checked at construction only, for the
                  particles that are cuttable then.
                - Near a vertex that is closer to the blade plane than
                  ``0.5 * cut_start_gap / (1 - 2 * cut_xfem_min_fraction)`` the cut surface is not on
                  the blade plane: the side, the crossing points and the material shares come from a
                  level value that is kept that far from zero, so no fragment is a sliver.
                - The blade force has an upper limit when the rigid-soft contact uses the penalty law
                  (``rigid_soft_contact_use_log_barrier=False``). The truncation stops the blade edge
                  at zero distance from the tissue, where one contact record carries the contact
                  stiffness times its collision radius (particle radius plus shape margin). At the cut
                  front few records carry the blade. If their limit is below ``vbd.cut_force``, the
                  cut stops. The solver warns at construction when the largest ``vbd.cut_force`` is
                  above that limit of one contact record of the blade edge. The sound remedy is
                  ``rigid_soft_contact_use_log_barrier=True``: its force grows without that limit as
                  the distance goes to zero. A lower cut force or a stiffer blade contact also moves
                  the scene away from the limit.
            cut_tie_stiffness: Stiffness of the tie that holds a wound particle at its point on its
                mesh edge [N/m] (``cut_method="xfem"``). ``None`` uses two times
                ``model.soft_contact_ke`` at solver construction.
            cut_tie_dual: ``True`` updates a multiplier of every tie once per iteration
                (``dual += stiffness * residual``). ``False`` (default) gives a plain spring, whose
                residual is the force on the wound particle divided by the stiffness. The multiplier
                has no bound. In the test scenes it grew at every iteration on wound particles that
                contact held off their tie target (most on front edges under the blade), and the
                tissue became unstable.
            cut_xfem_min_fraction: Lower limit of the edge parameter of a wound particle and of its
                distance from the far end, as a share of the mesh edge, in (0, 0.5). It keeps wound
                particles off the tet corners and keeps the material share of a fragment above zero.
                Keep this share of the shortest tet edge above the square root of
                ``particle_edge_parallel_epsilon``.
            cut_wound_mass_fraction: Share of the mass of a tie corner that the wound particles made in
                one cut advance take from it, in (0, 1). A wound particle needs mass to be solved;
                the total particle mass stays.
            cut_xfem_contact_filter: ``True`` (default) removes two kinds of self-contact pairs after
                each detection (``cut_method="xfem"``): a pair whose particles all belong to the same
                side of the same cut (wound particles of that side, and vertices of cut tets on that
                side) and whose two parts are nearer than the self-contact detection radius (margin
                plus gap) in the rest shape, and a pair of a wound particle and a row that names a
                corner of its tie. The wound face of a fragment lies inside the material of its own
                side, so without the filter it pushes on the corners of its own fragment. A pair of
                one side that is far apart in the rest shape stays, so a part that folds onto
                itself has contact. ``False`` keeps all pairs.
            cut_xfem_front_rows: ``True`` gives every front edge (a crossed edge that a tet that is
                not cut still holds) an edge row between its two wound particles
                (``cut_method="xfem"``), and the blade edge rests on these rows. ``False`` (default):
                the blade edge meets only the two wound particles of a front edge, which are half a
                start gap to each side of the blade plane. A fork of the design. The rows take no
                part in self contact. Each crossed mesh edge needs one spare edge row for it
                (``max_edges`` above ten per spare tet). The limit of the blade force that
                ``cut_method`` describes holds with both settings.

            collision_pipeline: Optional :class:`~newton.CollisionPipeline`
                owned by this solver. When given, the solver allocates its own
                contacts buffer (:attr:`contacts`), seeds the pipeline's
                self-contact configuration from the ``particle_self_contact_*``
                parameters, and runs rigid collision detection itself per the
                schedule below; ``step()`` must then receive ``contacts=None``.
            collision_frequency: Iteration frequencies keyed by
                :class:`SolverBase.CollisionSlot`; only used by ``ITERATIONS``
                slots (before iterations k, 2k, and so on).
            collision_frequency_type: In-step detection points keyed by
                :class:`SolverBase.CollisionSlot`; runtime-changeable via
                :meth:`SolverBase.set_collision_frequency`. A slot whose DAT family is
                active may not be ``NONE`` (``RIGID`` with ``rigid_soft_enable_dat``,
                ``SOFT_SELF_CONTACT`` with ``particle_enable_self_contact``); :meth:`step`
                raises otherwise.

        Note:
            - The `integrate_with_external_rigid_solver` argument enables one-way coupling between rigid body and soft body
              solvers. If set to True, the rigid states should be integrated externally, with `state_in` passed to `step`
              representing the previous rigid state and `state_out` representing the current one. Frictional forces are
              computed accordingly.
            - `particle_vertex_contact_buffer_size`, `particle_edge_contact_buffer_size`, `rigid_body_contact_buffer_size`,
              and `rigid_body_particle_contact_buffer_size` are fixed and will not be dynamically resized during runtime.
              Setting them too small may result in undetected collisions (particles) or contact overflow (rigid body
              contacts). Body-particle list overflow uses the complete stored soft-contact buffer as
              a fallback; it cannot recover records lost to overflow of that collision buffer itself.
              Setting them excessively large may increase memory usage and degrade performance.
            - Dahl hysteresis friction for rod angular response is controlled by custom model attributes
              ``model.vbd.dahl_eps_max`` and ``model.vbd.dahl_tau``. Register them with
              ``SolverVBD.register_custom_attributes`` before building the model. Dahl friction is
              enabled only when positive Dahl parameters are authored.

        """
        integrates_rigid_bodies = model.body_count > 0 and not integrate_with_external_rigid_solver
        vbd_attrs = getattr(model, "vbd", None)

        def resolve_scene_attribute(name: str, explicit_value: Any) -> tuple[Any, bool]:
            """Resolve an explicit option before its model custom attribute and default.

            Returns the value and whether it came from the scene (``model.vbd``).
            """
            if explicit_value is not None:
                return explicit_value, False
            model_value = getattr(vbd_attrs, name, None)
            if model_value is None:
                return _SCENE_ATTRIBUTES[name][1], False
            value = model_value.numpy()[0].item()
            usd_name = _SCENE_ATTRIBUTES[name][2]
            if name != "rigid_contact_history" and not (np.isfinite(value) and value >= 0.0):
                raise ValueError(
                    f"Scene-authored {usd_name} (model.vbd.{name}) must be finite and >= 0, got {value}. "
                    f"Fix the scene value or pass {name}= explicitly to SolverVBD."
                )
            return value, True

        if iterations is None:
            scene_iterations = getattr(vbd_attrs, "max_solver_iterations", None)
            iterations = int(scene_iterations.numpy()[0]) if scene_iterations is not None else -1
            if iterations == -1:
                iterations = 10
            elif iterations <= 0:
                raise ValueError(
                    "Scene-authored newton:maxSolverIterations must be positive or -1 (use the default), "
                    f"got {iterations}. Fix the scene value or pass iterations= explicitly to SolverVBD."
                )

        rigid_contact_history, history_from_scene = resolve_scene_attribute(
            "rigid_contact_history", rigid_contact_history
        )
        rigid_contact_history = bool(rigid_contact_history)
        if history_from_scene and not integrates_rigid_bodies:
            # Contact history only applies to rigid bodies integrated by VBD; a scene-authored
            # value must not make body-free or externally integrated models fail.
            rigid_contact_history = False
        rigid_joint_linear_ke = float(resolve_scene_attribute("rigid_joint_linear_ke", rigid_joint_linear_ke)[0])
        rigid_joint_angular_ke = float(resolve_scene_attribute("rigid_joint_angular_ke", rigid_joint_angular_ke)[0])
        rigid_joint_linear_kd = float(resolve_scene_attribute("rigid_joint_linear_kd", rigid_joint_linear_kd)[0])
        rigid_joint_angular_kd = float(resolve_scene_attribute("rigid_joint_angular_kd", rigid_joint_angular_kd)[0])

        if rigid_compliant_alm is None:
            raise TypeError("rigid_compliant_alm must be True or False; omit it to use the default.")

        legacy_controls = [
            name
            for name, value in (
                ("rigid_avbd_beta", rigid_avbd_beta),
                ("rigid_avbd_linear_beta", rigid_avbd_linear_beta),
                ("rigid_avbd_angular_beta", rigid_avbd_angular_beta),
                ("rigid_contact_hard", rigid_contact_hard),
                ("rigid_contact_k_start", rigid_contact_k_start),
                ("rigid_joint_linear_k_start", rigid_joint_linear_k_start),
                ("rigid_joint_angular_k_start", rigid_joint_angular_k_start),
            )
            if value is not None
        ]
        rigid_avbd_beta = 0.0 if rigid_avbd_beta is None else rigid_avbd_beta
        rigid_contact_hard = True if rigid_contact_hard is None else rigid_contact_hard
        rigid_contact_k_start = 1.0e2 if rigid_contact_k_start is None else rigid_contact_k_start
        rigid_joint_linear_k_start = 1.0e2 if rigid_joint_linear_k_start is None else rigid_joint_linear_k_start
        rigid_joint_angular_k_start = 1.0e1 if rigid_joint_angular_k_start is None else rigid_joint_angular_k_start

        if rigid_avbd_beta < 0:
            raise ValueError(f"rigid_avbd_beta must be >= 0, got {rigid_avbd_beta}")
        rigid_avbd_linear_beta = rigid_avbd_linear_beta if rigid_avbd_linear_beta is not None else rigid_avbd_beta
        rigid_avbd_angular_beta = rigid_avbd_angular_beta if rigid_avbd_angular_beta is not None else rigid_avbd_beta

        # Self-contact geometry: margin/gap scheme (margin = interaction distance,
        # detection query radius = margin + gap). The legacy radius/margin pair is
        # deprecated; its presence selects the legacy interpretation exactly.
        if particle_self_contact_radius is not None:
            # Validate before warning: under warnings-as-errors (CI) a warning
            # emitted first would mask the ValueError with a DeprecationWarning.
            if particle_self_contact_gap is not None:
                raise ValueError(
                    "particle_self_contact_gap cannot be combined with the deprecated "
                    "particle_self_contact_radius; migrate to margin + gap."
                )
            _sc_margin = particle_self_contact_radius
            _legacy_query = particle_self_contact_margin if particle_self_contact_margin is not None else 0.2
            _sc_gap = _legacy_query - _sc_margin
            if _sc_gap < 0.0:
                raise ValueError(
                    "particle_self_contact_margin is smaller than particle_self_contact_radius, this will result in missing contacts and cause instability.\n"
                    "It is advisable to make particle_self_contact_margin 1.5-2 times larger than particle_self_contact_radius."
                )
            warnings.warn(
                "particle_self_contact_radius is deprecated; use particle_self_contact_margin "
                "(interaction distance) and particle_self_contact_gap (extra detection reach, "
                "query radius = margin + gap) instead. With radius set, "
                "particle_self_contact_margin keeps its legacy meaning (detection query radius).",
                NewtonDeprecationWarning,
                stacklevel=3,
            )
        else:
            if particle_self_contact_margin is not None and particle_self_contact_gap is None:
                _sc_gap = particle_self_contact_margin - 0.2
                if _sc_gap < 0.0:
                    if particle_enable_self_contact and model.particle_count > 0:
                        raise ValueError(
                            "particle_self_contact_margin is smaller than the legacy interaction radius 0.2; "
                            "this would miss contacts. Pass particle_self_contact_gap explicitly to use the "
                            "new margin + gap convention."
                        )
                    _sc_gap = 0.0
                warnings.warn(
                    "particle_self_contact_margin without particle_self_contact_gap retains its deprecated "
                    "meaning as the detection query radius; pass particle_self_contact_gap explicitly "
                    "to use particle_self_contact_margin as the interaction distance.",
                    NewtonDeprecationWarning,
                    stacklevel=3,
                )
                _sc_margin = 0.2
            else:
                _sc_margin = particle_self_contact_margin if particle_self_contact_margin is not None else 0.2
                _sc_gap = particle_self_contact_gap if particle_self_contact_gap is not None else 0.0
            if _sc_gap < 0.0:
                raise ValueError(f"particle_self_contact_gap must be >= 0, got {_sc_gap}")

        if model.attachment_particle_body_count > 0 and integrate_with_external_rigid_solver:
            raise ValueError(
                "Particle-body attachments require SolverVBD to integrate both endpoints; "
                "integrate_with_external_rigid_solver=True is not supported."
            )

        if particle_collision_detection_interval is not None:
            if (
                collision_frequency_type is not None
                and SolverBase.CollisionSlot.SOFT_SELF_CONTACT in collision_frequency_type
                and SolverBase.CollisionFrequencyType(
                    collision_frequency_type[SolverBase.CollisionSlot.SOFT_SELF_CONTACT]
                )
                != SolverBase.CollisionFrequencyType.AUTO
            ):
                raise ValueError(
                    "set either the deprecated particle_collision_detection_interval or the "
                    "self-contact slot of collision_frequency_type, not both"
                )
            warnings.warn(
                "particle_collision_detection_interval is deprecated; use the self-contact slot of "
                "collision_frequency / collision_frequency_type instead (PRE_INIT ~ interval < 0, "
                "PRE_POST_INIT ~ interval == 0, ITERATIONS ~ interval >= 1).",
                NewtonDeprecationWarning,
                stacklevel=3,
            )
        self._deprecated_particle_interval = particle_collision_detection_interval

        # Shared DAT relaxation; the deprecated particle-only name overrides it with a warning.
        if particle_conservative_bound_relaxation is not None:
            dat_conservative_bound_relaxation = particle_conservative_bound_relaxation
        # Validate before warning so warnings-as-errors cannot mask the ValueError.
        if not (0.0 < dat_conservative_bound_relaxation < 1.0):
            raise ValueError(
                f"dat_conservative_bound_relaxation must be in (0, 1), got {dat_conservative_bound_relaxation}"
            )
        if particle_conservative_bound_relaxation is not None:
            warnings.warn(
                "particle_conservative_bound_relaxation is deprecated and overrides "
                "dat_conservative_bound_relaxation; pass only dat_conservative_bound_relaxation.",
                NewtonDeprecationWarning,
                stacklevel=3,
            )
        self.dat_conservative_bound_relaxation = dat_conservative_bound_relaxation
        # Set before super().__init__: _default_collision_frequency_type (AUTO
        # resolution) reads them as soon as the base class is constructed.
        self.particle_enable_self_contact = particle_enable_self_contact
        self.rigid_soft_enable_dat = rigid_soft_enable_dat

        # With an owned pipeline, seed its self-contact configuration from the solver's
        # parameters before the base class allocates the owned Contacts buffer.
        if collision_pipeline is not None and particle_enable_self_contact:
            collision_pipeline.init_soft_self_contact(
                margin=_sc_margin,
                gap=_sc_gap,
                rest_shape_exclusion_radius=particle_rest_shape_contact_exclusion_radius,
                vertex_buffer_pre_alloc=particle_vertex_contact_buffer_size,
                edge_buffer_pre_alloc=particle_edge_contact_buffer_size,
                edge_edge_parallel_epsilon=particle_edge_parallel_epsilon,
                topological_filter_threshold=particle_topological_contact_filter_threshold,
                external_vertex_filter_map=particle_external_vertex_contact_filtering_map,
                external_edge_filter_map=particle_external_edge_contact_filtering_map,
            )

        super().__init__(
            model,
            collision_pipeline=collision_pipeline,
            collision_frequency=collision_frequency,
            collision_frequency_type=collision_frequency_type,
        )
        # The solver's warm-start setting is authoritative for an owned pipeline:
        # lambda/penalty restores ride the pipeline's matched contact indices, so
        # history without matching silently cold-starts every refresh (k times per
        # step under rigid ITERATIONS). Matching is fixed at pipeline construction,
        # so surface the mismatch instead of repairing it.
        if (
            collision_pipeline is not None
            and rigid_contact_history
            and collision_pipeline.contact_matching == "disabled"
        ):
            raise ValueError(
                "SolverVBD(rigid_contact_history=True) with an owned pipeline requires contact "
                "matching for the warm-start restore; construct the pipeline with "
                "contact_matching='latest' (or 'sticky' for persistent friction anchors). "
                "Alternatively, set rigid_contact_history=False."
                + (
                    " The value came from the scene (model.vbd.rigid_contact_history, "
                    "newton:vbd:rigid:contactHistory); passing rigid_contact_history=False overrides it."
                    if history_from_scene
                    else ""
                )
            )
        # Per-step schedule cache; refreshed at every step() so runtime
        # set_collision_frequency() changes take effect at the next step.
        self._self_contact_mode_this_step, self._self_contact_freq_this_step = self._resolve_self_contact_schedule()
        effective_deterministic = deterministic if deterministic is not None else wp.config.deterministic
        # Body-particle contact accumulation: per-record atomics over the active prefix by default;
        # deterministic modes use the per-particle gather, which needs no atomics.
        self._particle_contact_use_gather = effective_deterministic != wp.DeterministicMode.NOT_GUARANTEED
        particle_deterministic_max_records = 0
        coupling_deterministic_max_records = 0
        if effective_deterministic != wp.DeterministicMode.NOT_GUARANTEED:
            if model.attachment_particle_body_count > 0:
                attachment_particles = model.attachment_particle_body_particle.numpy()
                attachment_records = int(np.bincount(attachment_particles, minlength=model.particle_count).max())
                particle_deterministic_max_records = max(
                    particle_deterministic_max_records,
                    attachment_records,
                )
            if particle_enable_self_contact:
                edge_iterations = (
                    particle_edge_contact_buffer_size + NUM_THREADS_PER_COLLISION_PRIMITIVE - 1
                ) // NUM_THREADS_PER_COLLISION_PRIMITIVE
                vertex_iterations = (
                    particle_vertex_contact_buffer_size + NUM_THREADS_PER_COLLISION_PRIMITIVE - 1
                ) // NUM_THREADS_PER_COLLISION_PRIMITIVE
                truncation_records = 4 * (edge_iterations + vertex_iterations)
                force_records = 2 * edge_iterations + 4 * vertex_iterations
                if model.shape_count > 0:
                    force_records += 1
                particle_deterministic_max_records = max(
                    particle_deterministic_max_records,
                    truncation_records,
                    force_records,
                )
                coupling_deterministic_max_records = 2 * edge_iterations + 3 * vertex_iterations
        if model.particle_count > 0:
            self._set_module_options(
                {
                    "deterministic": effective_deterministic,
                    "deterministic_max_records": particle_deterministic_max_records,
                    "fast_math": enable_fast_math,
                },
                module=particle_vbd_kernels,
            )
        self._set_module_options(
            {
                "deterministic": effective_deterministic,
                "deterministic_max_records": coupling_deterministic_max_records,
            },
            module=vbd_coupling_kernels,
        )

        options = {"deterministic": effective_deterministic, "deterministic_max_records": 0}
        self._solve_rigid_body_kernel = create_solve_rigid_body(model.tendon_count > 0)
        if integrates_rigid_bodies:
            rigid_modules = (
                rigid_vbd_kernels,
                accumulate_body_body_contacts_per_body.module,
                compute_rigid_contact_forces.module,
                update_duals_body_body_contacts.module,
            )
            for module in rigid_modules:
                self._set_module_options(options, module=module)
            for module in (self._solve_rigid_body_kernel.module, update_duals_joint.module):
                self._set_module_options(options | {"fast_math": enable_fast_math}, module=module)
        if model.joint_count > 0:
            self._set_module_options(
                {"deterministic": effective_deterministic, "deterministic_max_records": 0},
                module=xpbd_kernels,
            )

        # Common parameters
        self.iterations = iterations
        self.friction_epsilon = friction_epsilon
        self.rigid_soft_contact_use_log_barrier = bool(rigid_soft_contact_use_log_barrier)
        self._joint_mode_deprecation_warned = False

        self.tendon_max_sweeps = tendon_max_sweeps
        self.tendon_settle_tol = tendon_settle_tol
        self.tendon_activation_tol = tendon_activation_tol
        # Rigid integration mode: when True, rigid bodies are integrated by an external
        # solver (one-way coupling). SolverVBD will not move rigid bodies, but can still
        # participate in particle-rigid interaction on the particle side.
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver
        self._integrates_rigid_bodies = integrates_rigid_bodies

        # Initialize particle system
        self._init_particle_system(
            model,
            particle_enable_self_contact,
            _sc_margin,
            _sc_gap,
            particle_vertex_contact_buffer_size,
            particle_edge_contact_buffer_size,
            particle_collision_detection_interval,
            particle_edge_parallel_epsilon,
            particle_enable_tile_solve,
            particle_topological_contact_filter_threshold,
            particle_rest_shape_contact_exclusion_radius,
            particle_external_vertex_contact_filtering_map,
            particle_external_edge_contact_filtering_map,
            enable_fast_math,
        )

        # Routed tendon geometry and material state are shared with XPBD; VBD
        # evaluates their force and Hessian contributions natively.
        self._init_tendon_state(model)

        # Initialize rigid body system and rigid-particle (body-particle) interaction state
        self._init_rigid_system(
            model,
            rigid_compliant_alm,
            rigid_avbd_alpha,
            rigid_avbd_linear_beta,
            rigid_avbd_angular_beta,
            rigid_avbd_gamma,
            rigid_avbd_joint_alpha,
            rigid_avbd_contact_alpha,
            rigid_contact_hard,
            rigid_contact_history,
            rigid_contact_k_start,
            rigid_body_contact_buffer_size,
            rigid_body_particle_contact_buffer_size,
            rigid_joint_linear_ke,
            rigid_joint_angular_ke,
            rigid_joint_linear_k_start,
            rigid_joint_angular_k_start,
            rigid_joint_linear_kd,
            rigid_joint_angular_kd,
        )

        attachment_count = model.attachment_particle_body_count
        self._attachment_particle_body_lambda = wp.zeros(attachment_count, dtype=wp.vec3, device=self.device)
        self._attachment_particle_body_rho = wp.zeros(attachment_count, dtype=float, device=self.device)

        self._has_joint_mimics = self._integrates_rigid_bodies and has_supported_joint_mimics(model, "SolverVBD")
        self._mimic_body_deltas = None
        if self._has_joint_mimics:
            for kernel in (xpbd_kernels.solve_joint_mimics, xpbd_kernels.apply_joint_mimic_deltas):
                self._set_module_options(options, module=kernel.module)
            self._mimic_body_deltas = wp.zeros_like(model.body_qd)

        # Initialize rigid penetration-free (DAT) truncation state
        self._init_rigid_penetration_free(model, rigid_soft_dat_use_interval_arithmetic)

        # Initialize cutting state (shared tet-neighbor table, padded vertex-to-tet table, row sources)
        self._init_cut_system(
            model,
            _sc_margin,
            _sc_gap,
            effective_deterministic,
            particle_enable_self_contact,
            particle_topological_contact_filter_threshold,
            particle_rest_shape_contact_exclusion_radius,
            particle_edge_parallel_epsilon,
            particle_external_vertex_contact_filtering_map,
            particle_external_edge_contact_filtering_map,
            cut_armed,
            cut_update_interval,
            cut_start_gap,
            cut_contact_range,
            cut_keep_side_labels,
            cut_gap_direction,
            cut_mass_rule,
            cut_surface_edges,
            cut_press_wound,
            cut_gate_wound,
            cut_method,
            cut_tie_stiffness,
            cut_tie_dual,
            cut_xfem_min_fraction,
            cut_wound_mass_fraction,
            cut_xfem_contact_filter,
            cut_xfem_front_rows,
        )

        # Initialize suturing state (tunnel points, needle state machine, contact gate)
        self._init_suture_system(
            model,
            suture_tunnel_update_interval,
            suture_point_spacing,
            suture_gate_slack,
            suture_armed,
            suture_closest_search_window,
            suture_friction_transverse_weight,
            suture_smooth_tangent,
        )

        # Controls whether the next step() refreshes contact state derived from
        # the Contacts buffer or reuses the current rigid/body-particle contact state.
        # Defaults to True and is reset to True when consumed by step().
        self._update_rigid_history = True

        # Warn after validation so warnings-as-errors do not mask invalid inputs.
        if not self.rigid_compliant_alm:
            legacy_controls.insert(0, "rigid_compliant_alm=False")
        if legacy_controls:
            warnings.warn(
                f"VBD options {', '.join(legacy_controls)} are deprecated as of Newton 1.6 "
                "and will be removed in a future release. "
                "Omit these options and author fixed finite material stiffness.",
                NewtonDeprecationWarning,
                stacklevel=2,
            )

    def _init_particle_system(
        self,
        model: Model,
        particle_enable_self_contact: bool,
        particle_self_contact_margin: float,
        particle_self_contact_gap: float,
        particle_vertex_contact_buffer_size: int,
        particle_edge_contact_buffer_size: int,
        particle_collision_detection_interval: int,
        particle_edge_parallel_epsilon: float,
        particle_enable_tile_solve: bool,
        particle_topological_contact_filter_threshold: int,
        particle_rest_shape_contact_exclusion_radius: float,
        particle_external_vertex_contact_filtering_map: dict | None,
        particle_external_edge_contact_filtering_map: dict | None,
        enable_fast_math: bool,
    ):
        """Initialize particle-specific data structures and settings."""
        # Early exit if no particles
        if model.particle_count == 0:
            return

        self.particle_collision_detection_interval = particle_collision_detection_interval
        self.particle_topological_contact_filter_threshold = particle_topological_contact_filter_threshold
        self.particle_rest_shape_contact_exclusion_radius = particle_rest_shape_contact_exclusion_radius

        # Particle state storage
        self.particle_q_prev = wp.zeros_like(
            model.particle_q, device=self.device
        )  # per-substep previous q (for velocity)
        # Particle pose at the start of the coupled entry step, paired with
        # ``_coupling_body_q_prev_snapshot`` by the coupling harvest hooks.
        self._coupling_particle_q_prev_snapshot = wp.clone(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)  # inertial target positions

        # Particle adjacency info: reuse the shared device copy built once at finalize (the VBD
        # solver and the collision pipeline both use it, so it is uploaded only once).
        if self.model.soft_mesh_adjacency_device is None:
            raise ValueError("model.soft_mesh_adjacency_device is missing; finalize the model with ModelBuilder.")
        self.particle_adjacency = self.model.soft_mesh_adjacency_device

        # Self-contact settings (particle_enable_self_contact is set pre-super)
        self.particle_self_contact_margin = particle_self_contact_margin
        self.particle_self_contact_gap = particle_self_contact_gap
        # Detection query radius; margin is the interaction distance (surface offset).
        self._self_contact_query_radius = particle_self_contact_margin + particle_self_contact_gap
        self.particle_q_rest = model.particle_q

        # Tile solve settings
        if model.device.is_cpu and particle_enable_tile_solve and wp.config.log_level <= wp.LOG_DEBUG:
            logger.info("Tiled solve requires model.device='cuda'. Tiled solve is disabled.")

        self.use_particle_tile_solve = particle_enable_tile_solve and model.device.is_cuda
        # One tiled elasticity kernel, specialized at code generation from static model data.
        # Element materials are static solver inputs; rebuild the solver after changing triangle
        # or edge stiffness so the specialization is recomputed.
        self._tiled_elasticity_kernel = None
        self._tiled_elasticity_particles_per_block = 1
        if self.use_particle_tile_solve:
            include_tets = model.tet_count > 0
            include_triangles = not _is_tet_only_elasticity_model(model)
            two_particles_per_warp = not include_tets
            self._tiled_elasticity_kernel = make_solve_elasticity_tile(
                include_triangles, include_tets, two_particles_per_warp
            )
            self._set_module_options({"fast_math": enable_fast_math}, module=self._tiled_elasticity_kernel.module)
            self._tiled_elasticity_particles_per_block = 2 if two_particles_per_warp else 1
        if particle_enable_self_contact:
            self.particle_conservative_bounds = wp.zeros((model.particle_count,), dtype=float, device=self.device)
            self._self_contact_edge_edge_parallel_epsilon = particle_edge_parallel_epsilon

            if self.collision_pipeline is not None:
                collision_info = self._pipeline_contacts.soft_self_contact_data
            else:
                self.trimesh_collision_detector = TriMeshCollisionDetector(
                    self.model,
                    init_collision_info=True,
                    vertex_collision_buffer_pre_alloc=particle_vertex_contact_buffer_size,
                    edge_collision_buffer_pre_alloc=particle_edge_contact_buffer_size,
                    edge_edge_parallel_epsilon=particle_edge_parallel_epsilon,
                    topological_contact_filter_threshold=particle_topological_contact_filter_threshold,
                    external_vertex_triangle_filtering_map=particle_external_vertex_contact_filtering_map,
                    external_edge_edge_filtering_map=particle_external_edge_contact_filtering_map,
                )
                collision_info = self.trimesh_collision_detector.collision_info

            self.trimesh_collision_info = collision_info

            self.particle_self_contact_evaluation_kernel_launch_size = max(
                self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                self.model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
            )
        else:
            self.particle_self_contact_evaluation_kernel_launch_size = None

        # Particle force and hessian storage
        self.particle_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.particle_hessians = wp.zeros(self.model.particle_count, dtype=wp.mat33, device=self.device)

        # Validation
        if len(self.model.particle_color_groups) == 0:
            raise ValueError(
                "model.particle_color_groups is empty! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

        self.pos_prev_collision_detection = wp.zeros_like(model.particle_q, device=self.device)
        self.particle_displacements = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.truncation_ts = wp.zeros(self.model.particle_count, dtype=float, device=self.device)

    def _init_rigid_system(
        self,
        model: Model,
        rigid_compliant_alm: bool,
        rigid_avbd_alpha: float | None,
        rigid_avbd_linear_beta: float,
        rigid_avbd_angular_beta: float,
        rigid_avbd_gamma: float,
        rigid_avbd_joint_alpha: float | None,
        rigid_avbd_contact_alpha: float | None,
        rigid_contact_hard: bool,
        rigid_contact_history: bool,
        rigid_contact_k_start: float,
        rigid_body_contact_buffer_size: int,
        rigid_body_particle_contact_buffer_size: int,
        rigid_joint_linear_ke: float,
        rigid_joint_angular_ke: float,
        rigid_joint_linear_k_start: float,
        rigid_joint_angular_k_start: float,
        rigid_joint_linear_kd: float,
        rigid_joint_angular_kd: float,
    ) -> None:
        """Initialize rigid-body VBD data structures and settings.

        This includes:
          - Rigid-only joint/contact state (legacy AVBD and compliant ALM)
          - Shared interaction state for body-particle (rigid-particle) soft contacts
        """
        # Validate rigid constraint parameters
        if rigid_avbd_alpha is not None and not (0.0 <= rigid_avbd_alpha <= 1.0):
            raise ValueError(f"rigid_avbd_alpha must be in [0, 1], got {rigid_avbd_alpha}")
        if rigid_avbd_joint_alpha is not None and not (0.0 <= rigid_avbd_joint_alpha <= 1.0):
            raise ValueError(f"rigid_avbd_joint_alpha must be in [0, 1], got {rigid_avbd_joint_alpha}")
        if rigid_avbd_contact_alpha is not None and not (0.0 <= rigid_avbd_contact_alpha <= 1.0):
            raise ValueError(f"rigid_avbd_contact_alpha must be in [0, 1], got {rigid_avbd_contact_alpha}")
        if rigid_avbd_linear_beta < 0:
            raise ValueError(f"rigid_avbd_linear_beta must be >= 0, got {rigid_avbd_linear_beta}")
        if rigid_avbd_angular_beta < 0:
            raise ValueError(f"rigid_avbd_angular_beta must be >= 0, got {rigid_avbd_angular_beta}")
        if not (0.0 <= rigid_avbd_gamma <= 1.0):
            raise ValueError(f"rigid_avbd_gamma must be in [0, 1], got {rigid_avbd_gamma}")
        if rigid_contact_k_start < 0:
            raise ValueError(f"rigid_contact_k_start must be >= 0, got {rigid_contact_k_start}")
        if rigid_joint_linear_k_start < 0:
            raise ValueError(f"rigid_joint_linear_k_start must be >= 0, got {rigid_joint_linear_k_start}")
        if rigid_joint_angular_k_start < 0:
            raise ValueError(f"rigid_joint_angular_k_start must be >= 0, got {rigid_joint_angular_k_start}")
        if rigid_joint_linear_ke < 0:
            raise ValueError(f"rigid_joint_linear_ke must be >= 0, got {rigid_joint_linear_ke}")
        if rigid_joint_angular_ke < 0:
            raise ValueError(f"rigid_joint_angular_ke must be >= 0, got {rigid_joint_angular_ke}")
        if rigid_compliant_alm:
            _validate_compliant_alm_material_coefficient(rigid_joint_linear_ke, "rigid_joint_linear_ke")
            _validate_compliant_alm_material_coefficient(rigid_joint_angular_ke, "rigid_joint_angular_ke")
            _validate_compliant_alm_material_coefficient(
                rigid_joint_linear_kd, "rigid_joint_linear_kd", allow_negative=True
            )
            _validate_compliant_alm_material_coefficient(
                rigid_joint_angular_kd, "rigid_joint_angular_kd", allow_negative=True
            )
        self.rigid_avbd_gamma = rigid_avbd_gamma
        self.rigid_contact_k_start_value = -1.0 if rigid_avbd_linear_beta == 0.0 else float(rigid_contact_k_start)
        self.rigid_compliant_alm = bool(rigid_compliant_alm)
        self._validate_contact_materials()
        if self.rigid_compliant_alm:
            self._validate_compliant_joint_dof_materials()

        self.rigid_joint_linear_k_start = rigid_joint_linear_k_start if rigid_avbd_linear_beta > 0.0 else None
        self.rigid_joint_angular_k_start = rigid_joint_angular_k_start if rigid_avbd_angular_beta > 0.0 else None
        # Joint alpha precedence: joint override > shared override > mode default.
        # Compliant ALM defaults to the raw material residual (alpha=0).
        if rigid_avbd_joint_alpha is not None:
            self.rigid_joint_alpha = rigid_avbd_joint_alpha
        elif rigid_avbd_alpha is not None:
            self.rigid_joint_alpha = rigid_avbd_alpha
        elif rigid_compliant_alm:
            self.rigid_joint_alpha = 0.0
        else:
            self.rigid_joint_alpha = 0.95
        self.rigid_linear_beta = rigid_avbd_linear_beta
        self.rigid_angular_beta = rigid_avbd_angular_beta
        self.rigid_contact_hard = int(rigid_contact_hard)
        self.rigid_contact_history = rigid_contact_history
        # Contact alpha precedence: contact override > shared override > mode default.
        # Under compliant ALM it controls only residual stabilization.
        if rigid_avbd_contact_alpha is not None:
            self.rigid_contact_alpha = rigid_avbd_contact_alpha
        elif rigid_avbd_alpha is not None:
            self.rigid_contact_alpha = rigid_avbd_alpha
        else:
            self.rigid_contact_alpha = 0.0 if rigid_compliant_alm else 0.95

        # Joint constraint stiffness and damping for non-rod structural joints
        self.rigid_joint_linear_ke = rigid_joint_linear_ke
        self.rigid_joint_angular_ke = rigid_joint_angular_ke
        self.rigid_joint_linear_kd = max(0.0, rigid_joint_linear_kd)
        self.rigid_joint_angular_kd = max(0.0, rigid_joint_angular_kd)

        # -------------------------------------------------------------
        # Rigid-only solver state (used when SolverVBD integrates bodies)
        # -------------------------------------------------------------
        if self._integrates_rigid_bodies:
            # The first step's State establishes pose history; reset marks selected
            # worlds for a new baseline. Final slot: entities without a world.
            history_mask_size = model.world_count + 1
            self._rigid_pose_rebaseline_mask = wp.ones(history_mask_size, dtype=wp.bool, device=self.device)
            # Contact-reset state is consumed only by the warm-start refresh, so
            # allocate it (and let the reset kernel write it) only when enabled.
            if self.rigid_contact_history:
                self._contact_history_reset_mask = wp.zeros(history_mask_size, dtype=wp.bool, device=self.device)
                self._contact_history_reset_pending = wp.zeros(1, dtype=wp.int32, device=self.device)
            else:
                self._contact_history_reset_mask = None
                self._contact_history_reset_pending = None

            # Deterministic fallbacks for inspection before the first step overwrites them.
            self.body_q_prev = wp.clone(model.body_q, device=self.device)
            self._coupling_body_q_prev_snapshot = wp.clone(model.body_q, device=self.device)
            self.body_inertia_q = wp.zeros_like(model.body_q, device=self.device)  # inertial target poses

            # Adjacency and dimensions
            joint_constraint_dim = self._init_joint_constraint_layout()
            self.rigid_adjacency = self._compute_rigid_force_element_adjacency(model, joint_constraint_dim)
            self.tendon_adjacency = self._compute_tendon_force_element_adjacency(model).to(self.device)
            (
                self.body_particle_attachment_offsets,
                self.body_particle_attachment_indices,
            ) = self._compute_body_particle_attachment_adjacency(model)

            # Pack two vec3 and three mat33 fields for one clear.
            body_count = model.body_count
            self._body_contact_scratch = wp.zeros((11, body_count, 3), dtype=float, device=self.device)
            self.body_forces = self._body_contact_scratch[0].view(wp.vec3)
            self.body_torques = self._body_contact_scratch[1].view(wp.vec3)
            # Hessian slices store consecutive per-body matrices, not scratch[row, body] matrix rows.
            self.body_hessian_ll = self._body_contact_scratch[2:5].reshape((body_count, 3, 3)).view(wp.mat33)
            self.body_hessian_al = self._body_contact_scratch[5:8].reshape((body_count, 3, 3)).view(wp.mat33)
            self.body_hessian_aa = self._body_contact_scratch[8:11].reshape((body_count, 3, 3)).view(wp.mat33)

            # Persistent scratch for joint_f accumulation
            self._body_f_for_integration = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=self.device)

            # Per-body contact lists (CSR-like: per-body counts + flat index array).
            # Tight: pre_alloc = 0 when the contact source is absent (no shapes / no particles).
            bb_pre_alloc = rigid_body_contact_buffer_size if model.shape_count > 0 else 0
            self.body_body_contact_buffer_pre_alloc = bb_pre_alloc
            self.body_body_contact_counts = wp.zeros(model.body_count, dtype=wp.int32, device=self.device)
            self.body_body_contact_indices = wp.zeros(
                model.body_count * bb_pre_alloc, dtype=wp.int32, device=self.device
            )
            self.body_body_contact_overflow_max = wp.zeros(1, dtype=wp.int32, device=self.device)

            bp_pre_alloc = (
                rigid_body_particle_contact_buffer_size if model.shape_count > 0 and model.particle_count > 0 else 0
            )
            self.body_particle_contact_buffer_pre_alloc = bp_pre_alloc
            # Large cloths need more lanes to accumulate their contacts onto each rigid body.
            self._body_particle_contact_threads = (
                128 if self.device.is_cuda and model.particle_count // max(model.body_count, 1) >= 4096 else 4
            )
            self.body_particle_contact_counts = wp.zeros(model.body_count, dtype=wp.int32, device=self.device)
            self.body_particle_contact_indices = wp.zeros(
                model.body_count * bp_pre_alloc, dtype=wp.int32, device=self.device
            )
            self.body_particle_contact_overflow_max = wp.zeros(1, dtype=wp.int32, device=self.device)

            # Launch dimensions are fixed for the solver's lifetime.
            self._rigid_body_block_dim = _select_rigid_block_dim(model.body_count, self.device)
            self._rigid_joint_block_dim = _select_rigid_block_dim(model.joint_count, self.device)
            self._rigid_color_block_dims = tuple(
                (
                    _select_rigid_block_dim(group.size, self.device),
                    _select_rigid_block_dim(group.size * _NUM_CONTACT_THREADS_PER_BODY, self.device),
                    _select_rigid_block_dim(group.size * self._body_particle_contact_threads, self.device),
                )
                for group in model.body_color_groups
            )

            # Joint penalty state and material data.
            (
                self.joint_penalty_k,
                self.joint_penalty_k_min,
                self.joint_material_k,
                self.joint_rho,
                self.joint_penalty_kd,
                self.joint_is_hard,
            ) = self._init_joint_penalty_k()
            self._init_structural_k()
            self.joint_rest_angle = self._init_joint_rest_angle()
            self.joint_angle_prev = wp.clone(self.joint_rest_angle)

            # Body-body contact state (pre-allocated in __init__ when possible, resized on first step otherwise).
            self.body_body_contact_penalty_k = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_normal_rho = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_material_ke = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_material_kd = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_material_mu = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_tangent_rho = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_lambda = wp.zeros(0, dtype=wp.vec3, device=self.device)
            self.body_body_contact_C0 = wp.zeros(0, dtype=wp.vec3, device=self.device)

            # Rigid contact warm-start buffers.
            self._prev_contact_lambda = None
            self._prev_contact_penalty_k = None
            self._prev_contact_normal = None
            # ``[buffer id, contact generation]`` of the snapshotted contact set, on the
            # device so captured graphs replay it. Generations count collision passes
            # per buffer, so each buffer gets its own positive id.
            self._prev_contact_frame = None
            self._contact_history_buffer_ids: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
            self._last_contact_history_buffer_id = 0  # CONTACT_HISTORY_NO_BUFFER; real ids start at 1

            # Joint augmented-Lagrangian state (vec3, per-joint, bilateral)
            self.joint_lambda_lin = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_lambda_ang = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C0_lin = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C0_ang = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            # Shared directional support; drive and limit derive separate rho policies.
            self.joint_drive_limit_support = wp.zeros(model.joint_dof_count, dtype=float, device=self.device)
            # Bilateral drive dual, cleared whenever the drive row stops existing.
            self.joint_drive_lambda = wp.zeros(model.joint_dof_count, dtype=float, device=self.device)
            # Unilateral limit dual, held separately from the drive so neither can
            # destroy the other's reaction. Sign encodes the bound: >0 upper, <0 lower.
            self.joint_limit_lambda = wp.zeros(model.joint_dof_count, dtype=float, device=self.device)

            # Dahl friction state (rod angular hysteresis, persistent across timesteps)
            self.joint_sigma_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_kappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_dkappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Pre-computed Dahl parameters (frozen during iterations, updated per timestep)
            self.joint_sigma_start = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C_fric = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Dahl friction: registered custom attributes are inert until enabled by positive values.
            vbd_attrs: Any = getattr(model, "vbd", None)
            has_dahl = (
                model.joint_count > 0
                and vbd_attrs is not None
                and hasattr(vbd_attrs, "dahl_eps_max")
                and hasattr(vbd_attrs, "dahl_tau")
            )
            if has_dahl:
                self.joint_dahl_eps_max = vbd_attrs.dahl_eps_max
                self.joint_dahl_tau = vbd_attrs.dahl_tau
                dahl_eps_max = self._to_numpy(self.joint_dahl_eps_max, dtype=float)
                dahl_tau = self._to_numpy(self.joint_dahl_tau, dtype=float)
                self.enable_dahl_friction = bool(np.any((dahl_eps_max > 0.0) & (dahl_tau > 0.0)))
            else:
                self.joint_dahl_eps_max = wp.zeros(model.joint_count, dtype=float, device=self.device)
                self.joint_dahl_tau = wp.zeros(model.joint_count, dtype=float, device=self.device)
                self.enable_dahl_friction = False

            # Per-joint DER rest invariants, refreshed at init and on model change
            # (see _refresh_rod_rest_bend_twist_cache): the parent-local rest
            # curvature binormal (bend) and the rest transported-material twist.
            # Rod joints use local +Z as the material tangent (a SolverVBD convention).
            self.joint_rod_rest_kb_local = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_rod_rest_twist = wp.zeros(model.joint_count, dtype=float, device=self.device)
            self._refresh_rod_rest_bend_twist_cache()

        # -------------------------------------------------------------
        # Body-particle interaction shared state.
        # -------------------------------------------------------------
        self.body_particle_contact_penalty_k = wp.zeros(0, dtype=float, device=self.device)
        self.body_particle_contact_material_ke = wp.zeros(0, dtype=float, device=self.device)
        self.body_particle_contact_material_kd = wp.zeros(0, dtype=float, device=self.device)
        self.body_particle_contact_material_mu = wp.zeros(0, dtype=float, device=self.device)
        self._particle_contact_head = wp.full(model.particle_count, -1, dtype=wp.int32, device=self.device)
        self._particle_contact_next = wp.empty(0, dtype=wp.int32, device=self.device)
        self._particle_contact_adjacency_initialized = False
        self._particle_contact_worker_count = 0
        # Zero-length body poses for static-shape contact kernels when State.body_q is absent.
        self._empty_body_q = wp.empty(0, dtype=wp.transform, device=self.device)
        if model.particle_count > 0 and model.shape_count > 0:
            # Not shape_count * particle_count: that counts cross-world pairs, so it is quadratic in
            # world count and can exceed Warp's int32 array shape limit. A hint only -- the first step
            # grows this to contacts.soft_contact_max, raising if capture cannot allocate.
            self._init_body_particle_contact_state(_count_soft_particle_rigid_contact_pairs(model))

        # Kinematic body support: create effective inv_mass / inv_inertia arrays
        # with kinematic bodies zeroed out.
        self._init_kinematic_state()

        # Pre-allocate body-body contact buffers when the contact capacity is
        # already known; otherwise lazy allocation handles the first step.
        rcm = getattr(model, "rigid_contact_max", 0) or 0
        if rcm > 0 and self._integrates_rigid_bodies:
            self._init_body_body_contact_state(rcm)
            if (
                self.rigid_contact_history
                or self._resolved_collision_frequency_type(SolverBase.CollisionSlot.RIGID)
                == SolverBase.CollisionFrequencyType.ITERATIONS
            ):
                self._init_rigid_contact_warmstart(rcm)

        # Persistent contact-query outputs; per-contact arrays grow on demand.
        self._rigid_contact_body0 = wp.full(0, -1, dtype=wp.int32, device=self.device)
        self._rigid_contact_body1 = wp.full(0, -1, dtype=wp.int32, device=self.device)
        self._rigid_contact_point0_world = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self._rigid_contact_point1_world = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self._rigid_contact_zero_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._rigid_contact_zero_force = wp.zeros(0, dtype=wp.vec3, device=self.device)

        # Validation
        has_body_coloring = len(model.body_color_groups) > 0

        if self._integrates_rigid_bodies and not has_body_coloring:
            raise ValueError(
                "model.body_color_groups is empty but rigid bodies are present! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh solver caches affected by the specified model-property flags."""
        self._apply_module_options()
        if flags & ModelFlags.TENDON_PROPERTIES:
            self._refresh_tendon_properties()
        refresh_structural_k = (
            bool(
                flags
                & (
                    ModelFlags.JOINT_PROPERTIES
                    | ModelFlags.JOINT_DOF_PROPERTIES
                    | ModelFlags.JOINT_DOF_FORCE_PROPERTIES
                )
            )
            and self._integrates_rigid_bodies
            and self.model.joint_count > 0
        )
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()
        if flags & ModelFlags.SHAPE_TYPE_AND_SOURCE:
            self._select_rigid_dat_variant_radius()
        if (
            flags & (ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            and self._integrates_rigid_bodies
            and self.model.joint_count > 0
        ):
            self._validate_rod_material_axes()
            if self.rigid_compliant_alm:
                self._validate_compliant_joint_dof_materials()
            # Must run before _refresh_structural_k() below: that summary reads
            # joint_material_k as an input, so a stale material_k here would
            # produce a stale body_structural_k that is harder to spot.
            self._refresh_joint_material_params()
        if refresh_structural_k:
            self._refresh_structural_k()
        if flags & (ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES):
            self._refresh_rod_rest_bend_twist_cache()

    @override
    def coupling_supports_inertial_property_refresh(self) -> bool:
        return True

    @override
    def coupling_supports_full_surface_soft_contacts(self) -> bool:
        return True

    def coupling_notify_input_state_update(
        self,
        state: State,
        flags: StateFlags | int,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """Convert input body pose updates into VBD-compatible history updates.

        Also snapshots the input particle pose, which the coupling harvest hooks
        pair with the body pose history over the whole coupled entry step (all substeps).
        """
        self._apply_module_options()
        flags = int(flags)

        if (
            flags & StateFlags.PARTICLE_Q
            and dt > 0.0
            and not iteration_restart
            and state.particle_q is not None
            and self.model.particle_count > 0
        ):
            # Restarted iterations restore the same input particle pose, so they keep the pass-0 snapshot.
            wp.copy(dest=self._coupling_particle_q_prev_snapshot, src=state.particle_q)

        if (
            not (flags & StateFlags.BODY_Q)
            or state.body_q is None
            or state.body_qd is None
            or not self._integrates_rigid_bodies
        ):
            return

        if dt <= 0.0:
            # A reset distributes state before its world mask selects histories.
            if not iteration_restart:
                wp.copy(dest=self.body_q_prev, src=state.body_q)
            return

        if iteration_restart:
            # Restore the beginning-of-iteration history after a previous solve advanced it.
            wp.copy(dest=self.body_q_prev, src=self._coupling_body_q_prev_snapshot)

        wp.launch(
            _update_vbd_body_input_state_kernel,
            dim=self.model.body_count,
            inputs=[
                float(dt),
                self.model.body_flags,
                int(BodyFlags.KINEMATIC),
                self.model.body_world,
                self._rigid_pose_rebaseline_mask,
                state.body_q,
                self.body_q_prev,
                state.body_qd,
            ],
            device=self.device,
        )

        if not iteration_restart:
            # Snapshot pass-0 history so restarted iterations restore the same baseline.
            wp.copy(dest=self._coupling_body_q_prev_snapshot, src=self.body_q_prev)

    def coupling_prepare_proxy_contacts(
        self,
        state: State,
        contacts: Contacts | None,
        *,
        contacts_freshly_detected: bool = False,
    ) -> Contacts | None:
        """Update rigid history cadence for proxy contacts."""
        # Do not call super(); we can keep proxy-proxy collisions as we
        # are using a custom force harvesting hook
        self.set_rigid_history_update(bool(contacts_freshly_detected))
        return contacts

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Harvest contact-only proxy-body wrenches.

        VBD deliberately does not rely on the default momentum harvest here.
        The generic proxy path filters proxy-vs-proxy and proxy-vs-static rigid
        contacts so harvested momentum only reflects coupling-relevant
        interactions. VBD relaxes that restriction because allowing some proxy
        interaction inside the destination solve can strengthen the coupled
        solve. Those extra interactions still must not feed back through the
        coupling interface, so VBD harvests explicit contact forces instead of
        inferring feedback from total proxy momentum change.
        """
        self._apply_module_options()
        if not self._integrates_rigid_bodies:
            super().coupling_harvest_proxy_wrenches(
                body_local_to_proxy_global,
                out_body_f,
                body_qd_before=body_qd_before,
                state=state,
                state_out=state_out,
                contacts=contacts,
                dt=dt,
            )
            return

        out_body_f.zero_()
        if contacts is None:
            return

        # Both pose histories span the whole coupled entry step, matching the coupled ``dt``;
        # ``particle_q_prev`` and ``body_q_prev`` cover only the last substep of a sub-stepped entry.
        body_q_prev = self._coupling_body_q_prev_snapshot

        if contacts.rigid_contact_max > 0:
            body0, body1, point0, point1, force_on_body1, rigid_contact_count = self.collect_rigid_contact_forces(
                state_out.body_q,
                body_q_prev,
                contacts,
                dt,
            )
            wp.launch(
                _harvest_vbd_proxy_wrenches_kernel,
                dim=contacts.rigid_contact_max,
                inputs=[
                    rigid_contact_count,
                    body0,
                    body1,
                    point0,
                    point1,
                    force_on_body1,
                    self.model.body_inv_mass,
                    self.model.body_flags,
                    body_local_to_proxy_global,
                    int(BodyFlags.PROXY),
                    self.model.body_com,
                    state_out.body_q,
                    out_body_f,
                ],
                device=self.device,
            )

        if contacts.soft_contact_max > 0 and self.body_particle_contact_penalty_k.shape[0] >= contacts.soft_contact_max:
            # Use the solve's candidate list and force law, evaluating eligibility at
            # the feedback state. Retain the same per-body capacity limit as the solve.
            wp.launch(
                _harvest_vbd_body_particle_contact_forces_on_proxy_bodies_kernel,
                dim=self.model.body_count * _NUM_CONTACT_THREADS_PER_BODY,
                inputs=[
                    float(dt),
                    body_local_to_proxy_global,
                    state_out.particle_q,
                    self._coupling_particle_q_prev_snapshot,
                    self.model.particle_radius,
                    state_out.body_q,
                    body_q_prev,
                    state_out.body_qd,
                    self.model.body_com,
                    float(self.friction_epsilon),
                    self.rigid_soft_contact_use_log_barrier,
                    self.body_particle_contact_penalty_k,
                    self.body_particle_contact_material_kd,
                    self.body_particle_contact_material_mu,
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self.model.shape_margin,
                    self.model.shape_body,
                    self.body_particle_contact_buffer_pre_alloc,
                    self.body_particle_contact_counts,
                    self.body_particle_contact_indices,
                    _make_soft_contact_filter_data(self.model, contacts, state_out.particle_q, state_out.body_q),
                    out_body_f,
                ],
                device=self.device,
            )

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        particle_qd_before: wp.array[wp.vec3],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Harvest contact-only proxy-particle forces.

        As for proxy-body harvest, this stays contact-based because VBD allows
        some proxy interaction inside the destination solve for stronger
        coupling, but those proxy-only interactions should not appear as
        feedback forces on the source side.
        """
        self._apply_module_options()
        del particle_qd_before
        out_particle_f.zero_()
        if self.model.particle_count == 0 or particle_local_to_proxy_global.shape[0] == 0:
            return

        if contacts is not None:
            contacts._assert_particle_only_soft_contacts("SolverVBD proxy-particle coupling")

        if (
            contacts is not None
            and contacts.soft_contact_max > 0
            and contacts.soft_contact_count is not None
            and contacts.soft_contact_particle is not None
            and contacts.soft_contact_shape is not None
            and self.body_particle_contact_penalty_k.shape[0] >= contacts.soft_contact_max
        ):
            # Contacts are evaluated at the solved poses in ``state_out``, with histories at the
            # coupled entry step start pose, matching the coupled ``dt``.
            if self.integrate_with_external_rigid_solver:
                body_q_for_particles = state_out.body_q
                body_q_prev_for_particles = state.body_q
                body_qd_for_particles = state_out.body_qd
            else:
                body_q_for_particles = state_out.body_q
                body_q_prev_for_particles = self._coupling_body_q_prev_snapshot if self.model.body_count > 0 else None
                body_qd_for_particles = state_out.body_qd

            wp.launch(
                _harvest_vbd_proxy_particle_body_contact_forces_kernel,
                dim=contacts.soft_contact_max,
                inputs=[
                    float(dt),
                    particle_local_to_proxy_global,
                    state_out.particle_q,
                    self._coupling_particle_q_prev_snapshot,
                    self.model.particle_flags,
                    self.model.particle_inv_mass,
                    int(ParticleFlags.ACTIVE),
                    int(ParticleFlags.PROXY),
                    self.friction_epsilon,
                    self.rigid_soft_contact_use_log_barrier,
                    self.model.particle_radius,
                    contacts.soft_contact_count,
                    _make_soft_contact_filter_data(self.model, contacts, state_out.particle_q, body_q_for_particles),
                    contacts.soft_contact_particle,
                    self.body_particle_contact_penalty_k,
                    self.body_particle_contact_material_kd,
                    self.body_particle_contact_material_mu,
                    self.model.shape_body,
                    self.model.body_flags,
                    self.model.body_inv_mass,
                    int(BodyFlags.PROXY),
                    body_q_for_particles,
                    body_q_prev_for_particles,
                    body_qd_for_particles,
                    self.model.body_com,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self.model.shape_margin,
                    out_particle_f,
                ],
                device=self.device,
            )

        if self.particle_enable_self_contact:
            wp.launch(
                _harvest_vbd_proxy_particle_self_contact_forces_kernel,
                dim=self.particle_self_contact_evaluation_kernel_launch_size,
                inputs=[
                    float(dt),
                    particle_local_to_proxy_global,
                    self._coupling_particle_q_prev_snapshot,
                    state_out.particle_q,
                    self.model.particle_flags,
                    self.model.particle_inv_mass,
                    int(ParticleFlags.ACTIVE),
                    int(ParticleFlags.PROXY),
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.trimesh_collision_info,
                    self.particle_self_contact_margin,
                    self.model.soft_contact_ke,
                    self.model.soft_contact_kd,
                    self.model.soft_contact_mu,
                    self.friction_epsilon,
                    self._self_contact_edge_edge_parallel_epsilon,
                    out_particle_f,
                ],
                device=self.device,
                max_blocks=self.model.device.sm_count,
            )

    # =====================================================
    # Initialization Helper Methods
    # =====================================================

    def _init_body_body_contact_state(self, rigid_contact_max: int) -> None:
        """Allocate body-body contact state arrays sized to the given contact buffer capacity."""
        self.body_body_contact_penalty_k = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_normal_rho = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_material_ke = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_material_kd = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_material_mu = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_tangent_rho = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_lambda = wp.zeros(rigid_contact_max, dtype=wp.vec3, device=self.device)
        self.body_body_contact_C0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, device=self.device)

    def _validate_contact_materials(self) -> None:
        """Validate shape materials for particle contacts and compliant rigid contacts."""
        if self.model.shape_count == 0:
            return
        if self.model.particle_count == 0 and not (self._integrates_rigid_bodies and self.rigid_compliant_alm):
            return
        for attribute in ("shape_material_ke", "shape_material_kd", "shape_material_mu"):
            values = self._to_numpy(getattr(self.model, attribute), dtype=float)
            _validate_compliant_alm_material_coefficient(values, f"model.{attribute}")

    def _validate_compliant_joint_dof_materials(self) -> None:
        """Validate drive/limit coefficients consumed live by compliant ALM.

        Skip when an external rigid solver owns the bodies: VBD does not evaluate
        joint drive/limit rows in that coupling mode.
        """
        if not self._integrates_rigid_bodies or self.model.joint_count == 0:
            return
        for attribute in ("joint_target_ke", "joint_target_kd", "joint_limit_ke", "joint_limit_kd"):
            values = self._to_numpy(getattr(self.model, attribute), dtype=float)
            _validate_compliant_alm_material_coefficient(values, f"model.{attribute}")

    def _validate_rod_material_axes(self) -> None:
        """Require the duplicated transverse ROD material axes to remain isotropic."""
        if not self.model._has_rod_joints:  # pyright: ignore[reportPrivateUsage]
            return
        joint_type = self._to_numpy(self.model.joint_type, dtype=int)
        if not np.any(joint_type == JointType.ROD):
            return

        joint_qd_start = self._to_numpy(self.model.joint_qd_start, dtype=int)
        joint_dof_dim = self._to_numpy(self.model.joint_dof_dim, dtype=int)
        joint_target_ke = self._to_numpy(self.model.joint_target_ke, dtype=float)
        joint_target_kd = self._to_numpy(self.model.joint_target_kd, dtype=float)
        for joint in np.flatnonzero(joint_type == JointType.ROD):
            dof_start = int(joint_qd_start[joint])
            linear_count = int(joint_dof_dim[joint, 0])
            angular_count = int(joint_dof_dim[joint, 1])
            angular_dof = dof_start + linear_count
            if linear_count != 3 or angular_count != 3:
                raise RuntimeError(
                    "SolverVBD requires JointType.ROD to have three linear and three angular DOFs; "
                    f"joint {joint} has dimensions {(linear_count, angular_count)}."
                )
            if (
                joint_target_ke[dof_start] != joint_target_ke[dof_start + 1]
                or joint_target_kd[dof_start] != joint_target_kd[dof_start + 1]
                or joint_target_ke[angular_dof] != joint_target_ke[angular_dof + 1]
                or joint_target_kd[angular_dof] != joint_target_kd[angular_dof + 1]
            ):
                raise ValueError(
                    "SolverVBD requires isotropic ROD shear and bend coefficients about the local +Z "
                    f"material tangent; X and Y stiffness/damping entries differ for joint {joint}."
                )

    def _init_body_particle_contact_state(self, soft_contact_max: int) -> None:
        """Allocate body-particle material arrays sized to the given soft contact capacity."""
        self.body_particle_contact_penalty_k = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self.body_particle_contact_material_ke = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self.body_particle_contact_material_kd = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self.body_particle_contact_material_mu = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self._particle_contact_next = wp.empty(3 * soft_contact_max, dtype=wp.int32, device=self.device)
        self._particle_contact_adjacency_initialized = False
        self._particle_contact_worker_count = _particle_contact_scatter_worker_count(
            soft_contact_max, self.model.particle_count, self.device
        )

    def _init_rigid_contact_warmstart(self, rigid_contact_max: int) -> None:
        """Allocate fresh contact-history buffers."""
        cap = rigid_contact_max
        self._prev_contact_lambda = wp.zeros(cap, dtype=wp.vec3, device=self.device)
        self._prev_contact_penalty_k = wp.zeros(cap, dtype=float, device=self.device)
        self._prev_contact_normal = wp.zeros(cap, dtype=wp.vec3, device=self.device)
        # Fresh history holds no contact set, so every contact starts cold.
        self._prev_contact_frame = wp.full(2, CONTACT_HISTORY_NO_BUFFER, dtype=wp.int32, device=self.device)

    def _contact_history_buffer_id(self, contacts: Contacts) -> int:
        """Return the positive id rigid contact history uses for a contact buffer.

        Ids are never reused, so a replaced buffer cannot alias the snapshotted set.
        """
        buffer_id = self._contact_history_buffer_ids.get(contacts)
        if buffer_id is None:
            self._last_contact_history_buffer_id += 1
            buffer_id = self._last_contact_history_buffer_id
            self._contact_history_buffer_ids[contacts] = buffer_id
        return buffer_id

    def _raise_if_capturing_resize(self, name: str, current: int, required: int) -> None:
        if self.device.is_capturing and not is_graph_capture_allocation_enabled(self.device):
            raise RuntimeError(
                f"SolverVBD {name} buffer needs to grow from {current} to {required} "
                "during graph capture, but allocation during capture is not enabled on this device. "
                "Run one uncaptured step (or force-collection pass) before capture so the contact "
                "buffers are sized for the scene, or enable Warp's stream-ordered memory pool on this device. "
                "Rigid buffers can also be pre-sized by constructing CollisionPipeline before SolverVBD, "
                "which publishes model.rigid_contact_max; there is no equivalent for body-particle contacts."
            )

    def _refresh_rod_rest_bend_twist_cache(self) -> None:
        """Recompute Rod bend/twist rest invariants from Model rest coordinates."""
        # The cache is only allocated when SolverVBD integrates the rigid system
        # (see _init_rigid_system); skip when bodies are handled externally.
        if not self._integrates_rigid_bodies or self.model.joint_count == 0:
            return

        if not self.model._has_rod_joints:  # pyright: ignore[reportPrivateUsage]
            return
        joint_type_np = self._to_numpy(self.model.joint_type, dtype=np.int32)
        if not np.any(joint_type_np == int(JointType.ROD)):
            return

        wp.launch(
            kernel=init_rod_rest_bend_twist,
            dim=self.model.joint_count,
            inputs=[
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_rest_q,
            ],
            outputs=[
                self.joint_rod_rest_kb_local,
                self.joint_rod_rest_twist,
            ],
            device=self.device,
        )

    @staticmethod
    def _to_numpy(arr, dtype=None):
        """Transfer a Warp array to CPU and return as numpy, optionally casting dtype."""
        cpu = arr.to("cpu")
        result = cpu.numpy() if hasattr(cpu, "numpy") else np.asarray(cpu)
        return result if dtype is None else result.astype(dtype, copy=False)

    def _init_joint_constraint_layout(self) -> np.ndarray:
        """Initialize VBD-owned joint constraint indexing.

        VBD indexes scalar constraint components for structural joint penalties,
        compliant-ALM rho, and drive/limit penalty slots:
          - ROD:   4 scalars (stretch, shear, bend, twist)
          - BALL:  1 scalar (isotropic linear anchor-coincidence)
          - FIXED: 2 scalars (isotropic linear + isotropic angular)
          - REVOLUTE:  3 scalars (isotropic linear + 2-DOF perpendicular angular + angular drive/limit)
          - PRISMATIC: 3 scalars (2-DOF perpendicular linear + isotropic angular + linear drive/limit)
          - D6:   2 + lin_count + ang_count scalars (projected linear + projected angular + per-DOF drive/limit)
          - FREE:  0 scalars (not a constraint)

        Drive and limit for each free DOF share one AVBD slot (mutually exclusive at runtime).

        Any other joint type will raise NotImplementedError.

        Returns:
            Host constraint dimensions for adjacency construction without a device round trip.
        """
        n_j = self.model.joint_count
        with wp.ScopedDevice("cpu"):
            jt = self._to_numpy(self.model.joint_type, dtype=int)
            jdof_dim = self._to_numpy(self.model.joint_dof_dim, dtype=int)

            dim_np = np.zeros((n_j,), dtype=np.int32)
            for j in range(n_j):
                if jt[j] == JointType.ROD:
                    lin_count = int(jdof_dim[j, 0])
                    ang_count = int(jdof_dim[j, 1])
                    if lin_count != 3 or ang_count != 3:
                        raise RuntimeError(
                            "SolverVBD rigid joints: JointType.ROD requires three linear and three angular "
                            "DOF entries emitted by the rod builder APIs "
                            f"(got linear={lin_count}, angular={ang_count}) "
                            f"for joint {j}."
                        )
                    dim_np[j] = 4
                elif jt[j] == JointType.BALL:
                    dim_np[j] = 1
                elif jt[j] == JointType.FIXED:
                    dim_np[j] = 2
                elif jt[j] == JointType.REVOLUTE:
                    dim_np[j] = 3
                elif jt[j] == JointType.PRISMATIC:
                    dim_np[j] = 3
                elif jt[j] == JointType.D6:
                    dim_np[j] = 2 + int(jdof_dim[j, 0]) + int(jdof_dim[j, 1])
                else:
                    if jt[j] != JointType.FREE:
                        raise NotImplementedError(
                            f"SolverVBD rigid joints: JointType.{JointType(jt[j]).name} is not implemented yet "
                            "(only ROD, BALL, FIXED, REVOLUTE, PRISMATIC, and D6 are supported)."
                        )
                    dim_np[j] = 0

            start_np = np.zeros((n_j,), dtype=np.int32)
            c = 0
            for j in range(n_j):
                start_np[j] = np.int32(c)
                c += int(dim_np[j])

            self.joint_constraint_count = int(c)
            self.joint_constraint_dim = wp.array(dim_np, dtype=wp.int32, device=self.device)
            self.joint_constraint_start = wp.array(start_np, dtype=wp.int32, device=self.device)
        return dim_np

    def _init_joint_penalty_k(self):
        """Build initial joint penalty state on CPU and upload to solver device.

        Returns:
            (joint_penalty_k, joint_penalty_k_min, joint_material_k, joint_rho,
            joint_penalty_kd, joint_is_hard) tuple:
              - joint_penalty_k:       mutable legacy solver penalty per constraint scalar.
              - joint_penalty_k_min:   floor for the mutable legacy solver penalty; refreshed
                                       by ``notify_model_changed(JOINT_DOF_PROPERTIES)``.
              - joint_material_k:      material stiffness (= slot-specific ke); refreshed
                                       by ``notify_model_changed(JOINT_DOF_PROPERTIES)``.
              - joint_rho:             zeroed solver-owned storage; compliant ALM fills
                                       structural slots automatically each step.
              - joint_penalty_kd:      damping coefficient per constraint scalar.
              - joint_is_hard:         hard/soft flag per constraint scalar (1 = hard, 0 = soft).
        """
        if (
            not hasattr(self, "joint_constraint_start")
            or not hasattr(self, "joint_constraint_dim")
            or not hasattr(self, "joint_constraint_count")
        ):
            raise RuntimeError(
                "SolverVBD joint constraint layout is not initialized. "
                "Call SolverVBD._init_joint_constraint_layout() before _init_joint_penalty_k()."
            )

        if self.joint_constraint_count < 0:
            raise RuntimeError(
                f"SolverVBD joint constraint layout is invalid: joint_constraint_count={self.joint_constraint_count!r}"
            )

        constraint_count = self.joint_constraint_count
        legacy_lin_k_start = self.rigid_joint_linear_k_start
        legacy_ang_k_start = self.rigid_joint_angular_k_start

        with wp.ScopedDevice("cpu"):
            joint_material_k_np = np.zeros((constraint_count,), dtype=float)
            joint_k_init_np = np.zeros((constraint_count,), dtype=float)
            joint_rho_np = np.zeros((constraint_count,), dtype=float)
            joint_kd_np = np.zeros((constraint_count,), dtype=float)
            is_hard_np = np.zeros((constraint_count,), dtype=np.int32)

            jt = self._to_numpy(self.model.joint_type, dtype=int)
            jdofs = self._to_numpy(self.model.joint_qd_start, dtype=int)
            jtarget_ke = self._to_numpy(self.model.joint_target_ke, dtype=float)
            jtarget_kd = self._to_numpy(self.model.joint_target_kd, dtype=float)
            jlimit_ke = self._to_numpy(self.model.joint_limit_ke, dtype=float)
            jdof_dim = self._to_numpy(self.model.joint_dof_dim, dtype=int)
            jc_start = self._to_numpy(self.joint_constraint_start, dtype=np.int32)

            # TODO: Complete the Newton 1.6 deprecation by removing per-slot
            # hard/soft attribute compatibility with the legacy non-ALM path.
            # Per-joint hard/soft mode from model attribute (default=1, hard).
            vbd_attrs: Any = getattr(self.model, "vbd", None)
            if vbd_attrs is not None and hasattr(vbd_attrs, "joint_is_hard"):
                j_is_hard = self._to_numpy(vbd_attrs.joint_is_hard, dtype=np.int32)
                if not np.all((j_is_hard == 0) | (j_is_hard == 1)):
                    raise ValueError("model.vbd.joint_is_hard values must be 0 (soft) or 1 (hard).")
                # Authoring a soft slot is deliberate use of the deprecated hard/soft
                # distinction (the default is 1=hard, so all-hard is indistinguishable
                # from the untouched default and stays silent).
                if np.any(j_is_hard == 0):
                    warnings.warn(
                        "model.vbd.joint_is_hard (per-slot joint hard/soft mode) is deprecated as of "
                        "Newton 1.6 and will be removed with the legacy path. Under compliant ALM it has "
                        "no solver-mode effect; legacy AVBD still honors it during the migration window.",
                        NewtonDeprecationWarning,
                        # Reaches the constructor call site through _init_rigid_system.
                        stacklevel=4,
                    )
            else:
                j_is_hard = np.ones(self.model.joint_count, dtype=np.int32)

            structural_linear_ke = self.rigid_joint_linear_ke
            structural_angular_ke = self.rigid_joint_angular_ke

            def _penalty_k_init(material_k: float, seed: float | None) -> float:
                # Initial penalty k for any slot (structural or drive/limit): full material_k,
                # or a ramp-capped seed when AVBD ramping is enabled (seed is not None).
                return float(material_k) if seed is None else min(float(seed), float(material_k))

            n_j = self.model.joint_count
            for j in range(n_j):
                if jt[j] == JointType.ROD:
                    c0 = int(jc_start[j])
                    dof0 = int(jdofs[j])
                    linear_count = int(jdof_dim[j, 0])
                    angular_count = int(jdof_dim[j, 1])
                    angular_dof = dof0 + linear_count
                    dof_end = angular_dof + angular_count
                    if (
                        dof0 < 0
                        or linear_count != 3
                        or angular_count != 3
                        or dof_end > len(jtarget_ke)
                        or dof_end > len(jtarget_kd)
                    ):
                        raise RuntimeError(
                            "SolverVBD _init_joint_penalty_k: JointType.ROD requires three linear and three "
                            "angular DOF entries in model.joint_target_ke/kd. "
                            f"Got joint_index={j}, joint_qd_start={dof0}, "
                            f"len(joint_target_ke)={len(jtarget_ke)}, len(joint_target_kd)={len(jtarget_kd)}."
                        )
                    if (
                        jtarget_ke[dof0] != jtarget_ke[dof0 + 1]
                        or jtarget_kd[dof0] != jtarget_kd[dof0 + 1]
                        or jtarget_ke[angular_dof] != jtarget_ke[angular_dof + 1]
                        or jtarget_kd[angular_dof] != jtarget_kd[angular_dof + 1]
                    ):
                        raise ValueError(
                            "SolverVBD requires isotropic ROD shear and bend coefficients about the local +Z "
                            f"material tangent; X and Y stiffness/damping entries differ for joint {j}."
                        )
                    stretch_slot = c0
                    shear_slot = c0 + 1
                    bend_slot = c0 + 2
                    twist_slot = c0 + 3

                    # Rod anchors use local +Z as the material tangent, so Z carries
                    # stretch/twist. Shear and bend are isotropic about that tangent,
                    # so the validated X entry represents both transverse axes.
                    shear_dof = dof0
                    stretch_dof = dof0 + 2
                    bend_dof = angular_dof
                    twist_dof = angular_dof + 2

                    ke_stretch = jtarget_ke[stretch_dof]
                    ke_shear = jtarget_ke[shear_dof]
                    ke_bend = jtarget_ke[bend_dof]
                    ke_twist = jtarget_ke[twist_dof]

                    joint_material_k_np[stretch_slot] = ke_stretch
                    joint_material_k_np[shear_slot] = ke_shear
                    joint_material_k_np[bend_slot] = ke_bend
                    joint_material_k_np[twist_slot] = ke_twist

                    joint_k_init_np[stretch_slot] = _penalty_k_init(ke_stretch, legacy_lin_k_start)
                    joint_k_init_np[shear_slot] = _penalty_k_init(ke_shear, legacy_lin_k_start)
                    joint_k_init_np[bend_slot] = _penalty_k_init(ke_bend, legacy_ang_k_start)
                    joint_k_init_np[twist_slot] = _penalty_k_init(ke_twist, legacy_ang_k_start)

                    joint_kd_np[stretch_slot] = jtarget_kd[stretch_dof]
                    joint_kd_np[shear_slot] = jtarget_kd[shear_dof]
                    joint_kd_np[bend_slot] = jtarget_kd[bend_dof]
                    joint_kd_np[twist_slot] = jtarget_kd[twist_dof]
                elif jt[j] == JointType.BALL:
                    c0 = int(jc_start[j])
                    hard = int(j_is_hard[j])
                    joint_material_k_np[c0] = structural_linear_ke
                    joint_k_init_np[c0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                    joint_kd_np[c0] = self.rigid_joint_linear_kd
                    is_hard_np[c0] = hard
                elif jt[j] == JointType.FIXED:
                    c0 = int(jc_start[j])
                    hard = int(j_is_hard[j])
                    joint_material_k_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = hard
                    joint_material_k_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = _penalty_k_init(structural_angular_ke, legacy_ang_k_start)
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = hard
                elif jt[j] == JointType.REVOLUTE:
                    c0 = int(jc_start[j])
                    hard = int(j_is_hard[j])
                    joint_material_k_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = hard
                    joint_material_k_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = _penalty_k_init(structural_angular_ke, legacy_ang_k_start)
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = hard
                    dof0 = int(jdofs[j])
                    dl_material_k = max(float(jtarget_ke[dof0]), float(jlimit_ke[dof0]))
                    dl_seed = _penalty_k_init(dl_material_k, legacy_ang_k_start)
                    joint_material_k_np[c0 + 2] = dl_material_k
                    joint_k_init_np[c0 + 2] = dl_seed
                    joint_kd_np[c0 + 2] = 0.0
                elif jt[j] == JointType.PRISMATIC:
                    c0 = int(jc_start[j])
                    hard = int(j_is_hard[j])
                    joint_material_k_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = hard
                    joint_material_k_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = _penalty_k_init(structural_angular_ke, legacy_ang_k_start)
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = hard
                    dof0 = int(jdofs[j])
                    dl_material_k = max(float(jtarget_ke[dof0]), float(jlimit_ke[dof0]))
                    dl_seed = _penalty_k_init(dl_material_k, legacy_lin_k_start)
                    joint_material_k_np[c0 + 2] = dl_material_k
                    joint_k_init_np[c0 + 2] = dl_seed
                    joint_kd_np[c0 + 2] = 0.0
                elif jt[j] == JointType.D6:
                    c0 = int(jc_start[j])
                    dof0 = int(jdofs[j])
                    lc = int(jdof_dim[j, 0])
                    ac = int(jdof_dim[j, 1])
                    hard = int(j_is_hard[j])
                    if lc < 3:
                        joint_material_k_np[c0 + 0] = structural_linear_ke
                        joint_k_init_np[c0 + 0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                        joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                        is_hard_np[c0 + 0] = hard
                    if ac < 3:
                        joint_material_k_np[c0 + 1] = structural_angular_ke
                        joint_k_init_np[c0 + 1] = _penalty_k_init(structural_angular_ke, legacy_ang_k_start)
                        joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                        is_hard_np[c0 + 1] = hard
                    for li in range(lc):
                        dof_idx = dof0 + li
                        slot = c0 + 2 + li
                        dl_material_k = max(float(jtarget_ke[dof_idx]), float(jlimit_ke[dof_idx]))
                        dl_seed = _penalty_k_init(dl_material_k, legacy_lin_k_start)
                        joint_material_k_np[slot] = dl_material_k
                        joint_k_init_np[slot] = dl_seed
                        joint_kd_np[slot] = 0.0
                    for ai in range(ac):
                        dof_idx = dof0 + lc + ai
                        slot = c0 + 2 + lc + ai
                        dl_material_k = max(float(jtarget_ke[dof_idx]), float(jlimit_ke[dof_idx]))
                        dl_seed = _penalty_k_init(dl_material_k, legacy_ang_k_start)
                        joint_material_k_np[slot] = dl_material_k
                        joint_k_init_np[slot] = dl_seed
                        joint_kd_np[slot] = 0.0
                else:
                    pass

            k = wp.array(joint_k_init_np, dtype=float, device=self.device)
            k_min = wp.array(joint_k_init_np.copy(), dtype=float, device=self.device)
            material_k = wp.array(joint_material_k_np, dtype=float, device=self.device)
            rho = wp.array(joint_rho_np, dtype=float, device=self.device)
            kd = wp.array(joint_kd_np, dtype=float, device=self.device)
            is_hard = wp.array(is_hard_np, dtype=wp.int32, device=self.device)
            return k, k_min, material_k, rho, kd, is_hard

    def _init_structural_k(self) -> None:
        """Initialize the per-body structural stiffness summary from joint state.

        ``body_structural_k[b]`` is the max enabled linear-joint stiffness
        anchored on body ``b`` (rod joints use ``max(stretch, shear)``). Contact
        conditioning augments each dynamic endpoint's inertial scale with its
        own summary before combining endpoints.
        Direction- and chain-blind by design: it bounds neighborhood stiffness to
        condition rho and never enters a force law.
        Topology and the non-rod structural constants are construction-time state. The summary
        is refreshed in place after a notified joint-enable change, and after a
        ``JOINT_DOF_PROPERTIES`` change to rod stretch/shear stiffness, which feeds it.
        """
        self.body_structural_k = wp.empty(self.model.body_count, dtype=float, device=self.device)
        self._refresh_structural_k()

    def _refresh_structural_k(self) -> None:
        """Refresh the structural summary in place, without reallocating it.

        Depends on joint-enable flags and on rod stretch/shear ``joint_material_k``, so it is
        rerun for both ``JOINT_PROPERTIES`` and ``JOINT_DOF_PROPERTIES``.
        """
        self.body_structural_k.zero_()
        if self.model.joint_count == 0:
            return
        wp.launch(
            kernel=refresh_body_structural_k,
            dim=self.model.joint_count,
            inputs=[
                self.model.joint_type,
                self.model.joint_enabled,
                self.model.joint_parent,
                self.model.joint_child,
                self.joint_constraint_start,
                self.joint_constraint_dim,
                self.joint_material_k,
            ],
            outputs=[self.body_structural_k],
            device=self.device,
        )

    def _refresh_joint_material_params(self) -> None:
        """Refresh ``joint_target_ke``/``joint_target_kd``-derived material stiffness and damping.

        Stiffness covers ROD and the drive/limit slot(s) of REVOLUTE, PRISMATIC, and D6; its
        penalty and floor are reseeded alongside. Only legacy AVBD reads those non-ROD slots --
        compliant ALM gathers the same coefficients live from the model each solve. Not covered:
        BALL, FIXED, and REVOLUTE/PRISMATIC/D6's structural slots, which come from the
        solver-wide ``rigid_joint_linear_ke``/``rigid_joint_angular_ke`` constants.

        Damping is ROD-only -- other joint types read drive damping live from
        ``joint_target_kd`` at step time rather than caching it here.

        Stiffness slots are reseeded only where the effective stiffness changed, so untouched
        slots keep any legacy AVBD ramp they have accumulated.
        """
        if self.model.joint_count == 0:
            return
        wp.launch(
            kernel=refresh_joint_material_params,
            dim=self.model.joint_count,
            inputs=[
                self.model.joint_type,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.joint_constraint_start,
                self.model.joint_target_ke,
                self.model.joint_target_kd,
                self.model.joint_limit_ke,
                self.rigid_joint_linear_k_start if self.rigid_joint_linear_k_start is not None else -1.0,
                self.rigid_joint_angular_k_start if self.rigid_joint_angular_k_start is not None else -1.0,
            ],
            outputs=[
                self.joint_material_k,
                self.joint_penalty_k,
                self.joint_penalty_k_min,
                self.joint_penalty_kd,
            ],
            device=self.device,
        )

    def _init_joint_rest_angle(self):
        """Compute per-DOF rest-pose joint angles from ``model.joint_q``.

        VBD computes angular joint angles via ``kappa`` (rotation vector relative to
        the rest pose stored in ``model.body_q``). After ``eval_fk(model, ..., model)``,
        the rest pose encodes the initial joint configuration, so ``kappa = 0`` at the
        initial angles. Drive targets and limits, however, are specified in absolute
        joint coordinates. This array stores the rest-pose angle offset per DOF so that
        ``theta_abs = theta + joint_rest_angle[dof_idx]`` converts rest-relative
        ``theta`` back to absolute coordinates for drive/limit comparison.

        REVOLUTE joints and D6 joints with exactly one angular DOF additionally
        select the equivalent absolute angle nearest ``joint_angle_prev``.
        D6 joints with multiple angular DOFs measure absolute coordinates and need no offset.

        Only angular DOFs of REVOLUTE and D6 joints need nonzero entries. Linear DOFs
        (PRISMATIC, D6 linear) use absolute geometric measurements (``d_along``) and
        are unaffected - their entries are left at 0.
        """
        dof_count = self.model.joint_dof_count
        rest_angle_np = np.zeros(dof_count, dtype=float)

        with wp.ScopedDevice("cpu"):
            jt = self._to_numpy(self.model.joint_type, dtype=int)
            jq = self._to_numpy(self.model.joint_q, dtype=float)
            jq_start = self._to_numpy(self.model.joint_q_start, dtype=int)
            jqd_start = self._to_numpy(self.model.joint_qd_start, dtype=int)
            jdof_dim = self._to_numpy(self.model.joint_dof_dim, dtype=int)

            for j in range(self.model.joint_count):
                if jt[j] == JointType.REVOLUTE:
                    q_start = int(jq_start[j])
                    qd_start = int(jqd_start[j])
                    rest_angle_np[qd_start] = float(jq[q_start])
                elif jt[j] == JointType.D6:
                    q_start = int(jq_start[j])
                    qd_start = int(jqd_start[j])
                    lin_count = int(jdof_dim[j, 0])
                    ang_count = int(jdof_dim[j, 1])
                    if ang_count == 1:
                        rest_angle_np[qd_start + lin_count] = float(jq[q_start + lin_count])

        return wp.array(rest_angle_np, dtype=float, device=self.device)

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register SolverVBD custom Model attributes.

        Currently registers:
          - ``vbd:max_solver_iterations`` for ``NewtonSceneAPI.newton:maxSolverIterations``.
          - ``vbd:rigid_contact_history``, ``vbd:rigid_joint_linear_ke``, ``vbd:rigid_joint_angular_ke``,
            ``vbd:rigid_joint_linear_kd`` and ``vbd:rigid_joint_angular_kd`` for the ``NewtonVbdSceneAPI``
            solver settings, authored in USD as ``newton:vbd:rigid:contactHistory``,
            ``newton:vbd:rigid:jointLinearStiffness``, ``...jointAngularStiffness``,
            ``...jointLinearDamping`` and ``...jointAngularDamping``. ``SolverVBD`` uses them as
            defaults: explicit constructor argument > ``model.vbd.*`` > built-in default. Scene-authored
            stiffness and damping must be finite and non-negative.
          - ``vbd:joint_is_hard`` for per-joint hard/soft constraint mode (non-rod joints)
          - ``vbd:dahl_eps_max`` and ``vbd:dahl_tau`` for optional rod angular Dahl friction
          - ``vbd:suture_grip``, ``vbd:suture_mu``, ``vbd:suture_puncture_force``,
            ``vbd:suture_puncture_cos`` and ``vbd:suture_stiffness`` per particle for the suturing
            feature (tissue grip force per unit tunnel length [N/m], tunnel friction coefficient,
            puncture force [N], cosine of the puncture angle, tunnel row stiffness [N/m]); defaults
            1.0, 0.3, 0.5, 0.5 and infinity (a hard row). A tunnel point takes the smallest
            stiffness of its tet's four vertices.
          - ``vbd:cut_force`` per particle for the cutting feature: the blade force [N] along the
            cutting direction that cuts the tissue at this particle; default infinity (a value at or
            above 1e30 marks tissue that cannot be cut).

        Attributes are declared in the ``vbd`` namespace so they can be authored
        in scenes. The Dahl and ``joint_is_hard`` attributes are authored in USD as
        ``newton:vbd:<attr>``; the solver settings above use the names listed there.

        Dahl rod friction is enabled per joint only where both
        ``model.vbd.dahl_eps_max`` and ``model.vbd.dahl_tau`` are authored
        positive; the attributes default to zero.

        Args:
            builder: Model builder to register attributes on.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="max_solver_iterations",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                namespace="vbd",
                usd_attribute_name="newton:maxSolverIterations",
            )
        )

        for name, (dtype, default, usd_attribute_name) in _SCENE_ATTRIBUTES.items():
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=Model.AttributeFrequency.ONCE,
                    assignment=Model.AttributeAssignment.MODEL,
                    dtype=dtype,
                    default=default,
                    namespace="vbd",
                    usd_attribute_name=usd_attribute_name,
                )
            )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dahl_eps_max",
                frequency=Model.AttributeFrequency.JOINT,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="vbd",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dahl_tau",
                frequency=Model.AttributeFrequency.JOINT,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="vbd",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_is_hard",
                frequency=Model.AttributeFrequency.JOINT,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=1,
                namespace="vbd",
            )
        )
        for name, default in (*_SUTURE_PARTICLE_ATTRIBUTES, *_CUT_PARTICLE_ATTRIBUTES):
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=Model.AttributeFrequency.PARTICLE,
                    assignment=Model.AttributeAssignment.MODEL,
                    dtype=wp.float32,
                    default=default,
                    namespace="vbd",
                )
            )

    # =====================================================
    # Adjacency Building Methods
    # =====================================================

    def _compute_particle_force_element_adjacency(self):
        if self.model.soft_mesh_adjacency is None:
            raise ValueError("model.soft_mesh_adjacency is missing; finalize the model with ModelBuilder.")
        return self.model.soft_mesh_adjacency.init_vertex_adjacency(self.model.particle_count)

    def _compute_rigid_force_element_adjacency(self, model, joint_constraint_dim: np.ndarray):
        """Build body-to-constraint CSR adjacency in ascending joint order."""
        adjacency = RigidForceElementAdjacencyInfo()
        if self.joint_constraint_count == 0:
            adjacency.body_adj_joints = wp.empty(0, dtype=wp.int32, device=self.device)
            adjacency.body_adj_joints_offsets = wp.zeros(model.body_count + 1, dtype=wp.int32, device=self.device)
            return adjacency
        # FREE joints have no constraints; their applied forces are handled separately.
        # Keep disabled constraints here because joint_enabled can change between steps.
        constrained_joints = np.flatnonzero(joint_constraint_dim).astype(np.int32)
        body_ids = np.column_stack((model.joint_parent.numpy(), model.joint_child.numpy()))[constrained_joints].ravel()
        joint_ids = np.repeat(constrained_joints, 2)
        valid = body_ids >= 0  # Exclude world endpoints.
        body_ids, joint_ids = body_ids[valid], joint_ids[valid]
        # A stable sort preserves ascending joint order within each body.
        order = np.argsort(body_ids, kind="stable")
        offsets = np.zeros(model.body_count + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(np.bincount(body_ids, minlength=model.body_count))
        adjacency.body_adj_joints = wp.array(joint_ids[order], dtype=wp.int32, device=self.device)
        adjacency.body_adj_joints_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)
        return adjacency

    def _compute_tendon_force_element_adjacency(self, model: Model) -> TendonForceElementAdjacencyInfo:
        """Build CSR adjacency for authored and possible dynamic bypass segments."""
        adjacency = TendonForceElementAdjacencyInfo()
        body_segments = [set() for _ in range(model.body_count)]
        body_colors = np.full(model.body_count, -1, dtype=np.int32)
        for color, group in enumerate(model.body_color_groups):
            body_colors[group.numpy()] = color

        if model.tendon_segment_count > 0:
            guide_bodies = model.tendon_guide_body.numpy()
            guide_flags = model.tendon_guide_flags.numpy()
            segment_left_guides = self.tendon_segment_guide_l.numpy()
            guide_left_segments = self.tendon_guide_seg_left.numpy()
            for segment, left_guide in enumerate(segment_left_guides):
                body_l = int(guide_bodies[left_guide])
                body_r = int(guide_bodies[left_guide + 1])
                if (
                    body_l >= 0
                    and body_r >= 0
                    and body_l != body_r
                    and len(model.body_color_groups) > 0
                    and body_colors[body_l] == body_colors[body_r]
                ):
                    raise ValueError(
                        "model.body_color_groups does not separate tendon segment endpoints; "
                        "call ModelBuilder.color() after adding tendons"
                    )
                if body_l >= 0:
                    body_segments[body_l].add(segment)
                if body_r >= 0:
                    body_segments[body_r].add(segment)

                right_guide = left_guide + 1
                # Deactivation reuses the left authored segment for the bypass to the following guide.
                if (guide_flags[right_guide] & int(TendonGuideFlags.DYNAMIC)) != 0 and guide_left_segments[
                    right_guide
                ] == segment:
                    bypass_body = int(guide_bodies[right_guide + 1])
                    if (
                        bypass_body >= 0
                        and body_l >= 0
                        and bypass_body != body_l
                        and len(model.body_color_groups) > 0
                        and body_colors[bypass_body] == body_colors[body_l]
                    ):
                        raise ValueError(
                            "model.body_color_groups does not separate dynamic tendon bypass endpoints; "
                            "call ModelBuilder.color() after adding tendons"
                        )
                    if bypass_body >= 0:
                        body_segments[bypass_body].add(segment)

        offsets = np.zeros(model.body_count + 1, dtype=np.int32)
        for body, segments in enumerate(body_segments):
            offsets[body + 1] = offsets[body] + len(segments)

        flat_segments = np.asarray(
            [segment for segments in body_segments for segment in sorted(segments)], dtype=np.int32
        )
        with wp.ScopedDevice("cpu"):
            adjacency.body_adj_segments = wp.array(flat_segments, dtype=wp.int32)
            adjacency.body_adj_segments_offsets = wp.array(offsets, dtype=wp.int32)
        return adjacency

    def _compute_body_particle_attachment_adjacency(self, model: Model) -> tuple[wp.array, wp.array]:
        """Build CSR adjacency from rigid bodies to particle-body attachments."""
        if model.attachment_particle_body_count == 0:
            return (
                wp.zeros(model.body_count + 1, dtype=wp.int32, device=self.device),
                wp.empty(0, dtype=wp.int32, device=self.device),
            )

        indices, offsets = build_vertex_adjacency_with_warp(
            model.attachment_particle_body_body.to("cpu"),
            model.body_count,
            count_kernel=_count_body_particle_attachments_per_body,
            fill_kernel=_fill_body_particle_attachments_per_body,
            values_per_entry=1,
        )
        return (
            wp.array(offsets, dtype=wp.int32, device=self.device),
            wp.array(indices, dtype=wp.int32, device=self.device),
        )

    # =====================================================
    # Main Solver Methods
    # =====================================================

    def _initialize_particle_body_attachments(self, state: State, dt: float) -> None:
        """Refresh attachment ALM metrics and retain multipliers once per step."""
        model = self.model
        wp.launch(
            kernel=step_particle_body_attachment_lambda_rho,
            dim=model.attachment_particle_body_count,
            inputs=[
                dt,
                self.rigid_avbd_gamma,
                model.attachment_particle_body_body,
                model.attachment_particle_body_particle,
                model.attachment_particle_body_stiffness,
                model.attachment_particle_body_enabled,
                state.particle_q,
                self.particle_q_prev,
                model.particle_inv_mass,
                model.particle_flags,
                self.body_inv_mass_effective,
                self.particle_adjacency,
                model.tri_indices,
                model.tri_poses,
                model.tri_areas,
                model.tri_materials,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_rest_length,
                model.edge_bending_properties,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
            ],
            outputs=[self._attachment_particle_body_rho, self._attachment_particle_body_lambda],
            device=self.device,
        )

    def set_rigid_history_update(self, update: bool):
        """Set whether the next step() should update rigid solver history.

        When True (default), the step refreshes rigid contact state from the
        provided ``Contacts`` buffer: rebuilds per-body contact lists, initializes
        penalty_k/lambda/C0, and restores warm-start state when contact history
        is enabled (from each contact's own state when ``Contacts`` was not
        collided since the previous step, otherwise through
        ``Contacts.rigid_contact_match_index``).
        When False, the step reuses the current rigid contact lists and records
        without reinitializing them. Pass the same contact result/buffers
        used by the previous refresh; do not run collision into the contacts
        buffer between refreshes. Passing newly collided contacts while update is
        disabled can mismatch stale per-body contact lists with current contact
        rows. For the same reason, do not change a body's solvability (mass or
        kinematic flag) while update is disabled: the per-body lists depend on
        effective inverse mass and are not rebuilt until the next refresh.
        The per-particle contact adjacency is frozen with the same contract: its
        node ids index the contact buffer of the previous refresh, so passing a
        re-collided or smaller-capacity buffer while update is disabled reads
        the wrong records (or out of bounds).

        Joint constraint maintenance (C0 snapshot, lambda retention/decay, and
        automatic rho refresh) runs every step regardless of this flag via
        step_joint_C0_lambda_rho().
        Rigid contact history snapshotting also runs every step when enabled.
        Per-step contact maintenance (C0, conditioning, penalty, and lambda
        retention) also runs on the reused contact records.

        This setting applies only to the next call to :meth:`step` and is then
        reset to True.  Useful for substepping where collision detection frequency
        differs from the simulation step frequency.

        Args:
            update: If True, update rigid solver state. If False, reuse previous.
        """
        self._update_rigid_history = update

    # TODO: Complete the Newton 1.6 deprecation by removing this compatibility
    # method with the legacy non-ALM path.
    def set_joint_constraint_mode(self, joint_index: int, hard: bool, slot: int | None = None):
        """Set legacy hard/soft mode for a joint's structural slots at runtime.

        .. deprecated:: 1.6
            Per-slot joint hard/soft mode is deprecated. Under compliant ALM (the
            default) all structural slots use the unified scheme, so this
            has no solver-mode effect; it will be removed with the legacy path.

        Non-rod structural slots are LINEAR (slot 0) and ANGULAR (slot 1).
        Builder-created rod joints expose STRETCH (slot 0), SHEAR
        (slot 1), BEND (slot 2), and TWIST (slot 3). Other drive/limit slots
        are always soft and cannot be set to hard.

        By default, rod stretch, shear, bend, and twist slots are soft, while
        non-rod structural slots are hard.

        For non-rod joints, hard/soft mode can also be authored per joint at
        build time via the ``vbd:joint_is_hard`` custom attribute, avoiding a
        runtime :meth:`set_joint_constraint_mode` call::

            SolverVBD.register_custom_attributes(builder)  # before adding joints
            builder.add_joint_fixed(..., custom_attributes={"vbd:joint_is_hard": 0})
            model = builder.finalize()
            solver = SolverVBD(model, rigid_compliant_alm=False)

        Args:
            joint_index: Index of the joint to modify.
            hard: In legacy mode, True selects hard AL mode and False selects
                soft penalty mode. Has no solver-mode effect under compliant ALM.
            slot: Specific slot index to set. If None, sets all structural slots.
                Use JointSlot.LINEAR / JointSlot.ANGULAR for non-rod joints,
                or JointSlot.STRETCH / JointSlot.SHEAR / JointSlot.BEND /
                JointSlot.TWIST for rod joints.

        Raises:
            ValueError: If the joint index is out of range or the slot is not a
                structural slot for this joint.
        """
        if not self._joint_mode_deprecation_warned:
            warnings.warn(
                "SolverVBD.set_joint_constraint_mode (per-slot joint hard/soft mode) is deprecated as of "
                "Newton 1.6 and will be removed with the legacy path. Under compliant ALM it has no "
                "solver-mode effect; legacy AVBD still honors it during the migration window.",
                NewtonDeprecationWarning,
                stacklevel=2,
            )
            self._joint_mode_deprecation_warned = True
        n_j = self.model.joint_count
        if joint_index < 0 or joint_index >= n_j:
            raise ValueError(f"joint_index={joint_index} out of range [0, {n_j}).")

        with wp.ScopedDevice("cpu"):
            c_start_np = self._to_numpy(self.joint_constraint_start, dtype=np.int32)
            c_dim_np = self._to_numpy(self.joint_constraint_dim, dtype=np.int32)
            is_hard_np = self._to_numpy(self.joint_is_hard, dtype=np.int32)
            joint_type_np = self._to_numpy(self.model.joint_type, dtype=np.int32)

            c0 = int(c_start_np[joint_index])
            cdim = int(c_dim_np[joint_index])
            joint_type = int(joint_type_np[joint_index])
            structural_count = cdim if joint_type == int(JointType.ROD) else min(cdim, 2)
            val = 1 if hard else 0

            if slot is not None:
                if slot < 0 or slot >= structural_count:
                    if structural_count == 0:
                        names = "no structural slots"
                    elif joint_type == int(JointType.ROD):
                        names = "STRETCH=0, SHEAR=1, BEND=2, TWIST=3"
                    elif structural_count == 1:
                        names = "LINEAR=0"
                    else:
                        names = "LINEAR=0, ANGULAR=1"
                    raise ValueError(
                        f"Cannot set hard mode on slot={slot}: this joint has "
                        f"{structural_count} structural slot(s) ({names})."
                    )
                is_hard_np[c0 + slot] = val
            else:
                for s in range(structural_count):
                    is_hard_np[c0 + s] = val

            # Mutate in place: a rebuilt wp.array would orphan pointers captured
            # in existing CUDA graphs, silently ignoring the mode change on replay.
            self.joint_is_hard.assign(is_hard_np)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Execute one simulation timestep using VBD (particles) and AVBD (rigid bodies).

        The solver follows a 3-phase structure:
        1. Initialize: Forward integrate particles and rigid bodies, detect collisions, initialize contact state
        2. Iterate: Interleave particle and rigid-body VBD iterations
        3. Finalize: Update velocities and persistent state (Dahl friction)

        To control rigid body substepping behavior, call set_rigid_history_update().
        When True (default), the step rebuilds rigid contact lists, re-initializes
        rigid contact state (penalty_k, lambda, C0), and restores from history if enabled.
        When False, reuses previous rigid contact state. The flag is reset to True when consumed.

        Args:
            state_in: Input state.
            state_out: Output state.
            control: Control inputs.
            contacts: Contact data produced by :meth:`~newton.CollisionPipeline.collide` (rigid-rigid and
                rigid-particle contacts), allocated with :meth:`~newton.CollisionPipeline.contacts`.
                If None, rigid contact handling is skipped. Note that particle self-contact (if enabled) does not
                depend on this argument.
            dt: Time step size.
            observables: Optional solver observable arrays allocated by :meth:`observables`.
                When ``CONTACT_F`` is requested, evaluate one wrench per body-body and rigid-soft
                contact record at the final configuration. See :ref:`vbd_contact_forces`.

        Raises:
            RuntimeError: If required rigid contact-matching data is unavailable, or contact-history storage would
                need to be allocated or grown during graph capture.
        """
        contacts = self._resolve_step_contacts(contacts)
        self.validate_observables(observables, contacts)
        self._require_unreduced_contacts(contacts)
        self._apply_module_options()
        update_rigid = self._update_rigid_history
        self._update_rigid_history = True

        _Frequency = SolverBase.CollisionFrequencyType
        self._self_contact_mode_this_step, self._self_contact_freq_this_step = self._resolve_self_contact_schedule()
        self._rigid_mode_this_step = _Frequency.NONE
        self._rigid_freq_this_step = 1
        if self.collision_pipeline is not None:
            rigid_slot = SolverBase.CollisionSlot.RIGID
            self._rigid_mode_this_step = self._resolved_collision_frequency_type(rigid_slot)
            self._rigid_freq_this_step = self._collision_frequency[rigid_slot]

        self._validate_dat_collision_schedules()

        if self._pre_initialization_detection(state_in, dt):
            update_rigid = True

        if control is None:
            control = self.model.control(clone_variables=False)

        # Must run BEFORE _initialize_rigid_bodies: the dynamic routing activation
        # decision has to see the accepted pose, but the inertial predictor inside
        # _initialize_rigid_bodies overwrites state_in.body_q in place.
        if self.tendon_segment_lambda is not None:
            self._load_tendon_history(state_in)
            self._snapshot_tendon_step_state()
            self._update_tendon_guide_active(self.model, state_in.body_q)
            self._prepare_tendon_route(self.model, state_in.body_q, 1.0e-8)
            self._rebaseline_tendon_geometry(state_in.body_q)
            self.tendon_segment_material_tension.zero_()
            if self.iterations == 0 or self.integrate_with_external_rigid_solver:
                self.tendon_segment_lambda.zero_()

        self._initialize_rigid_bodies(state_in, control, contacts, dt, update_rigid)
        self._initialize_particles(state_in, state_out, contacts, dt)
        if self.model.attachment_particle_body_count > 0:
            self._initialize_particle_body_attachments(state_in, dt)
        if self.suture_needle_count > 0:
            self._initialize_suture_points(state_in, dt)

        rigid_due, soft_due = self._collision_detection_due((_Frequency.PRE_POST_INIT,))
        # With contact history, the post-initialization pass carries the state just
        # restored, like the in-iteration passes; its match indices refer to the
        # pre-initialization pass, not to the previous step's snapshot.
        self._mid_step_detection(
            state_in,
            state_out,
            contacts,
            dt,
            rigid_due=rigid_due,
            soft_due=soft_due,
            preserve_history=self.rigid_contact_history,
        )

        for iter_num in range(self.iterations):
            rigid_due = self.collision_pipeline is not None and self._rigid_collision_is_due(iter_num)
            soft_due = self.particle_enable_self_contact and self._self_contact_is_due(iter_num)
            self._mid_step_detection(
                state_in, state_out, contacts, dt, rigid_due=rigid_due, soft_due=soft_due, preserve_history=True
            )
            # Material can redistribute between fixed guides even without any rigid bodies.
            if self.tendon_segment_lambda is not None and not self.integrate_with_external_rigid_solver:
                self._update_tendon_routing(state_in, dt, False)
            if self.suture_needle_count > 0:
                self._refresh_suture_points(state_in)
            self._solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)
            self._solve_particle_iteration(state_in, state_out, contacts, dt)
            # The tie residual is read after the last iteration; the multiplier needs every iteration.
            if self._cut_xfem and (self.cut_tie_dual or iter_num == self.iterations - 1):
                self._update_duals_cut_ties(state_in)
            if self.suture_needle_count > 0:
                self._update_duals_suture_points(state_in)
            if self.model.attachment_particle_body_count > 0:
                wp.launch(
                    kernel=update_duals_particle_body_attachments,
                    dim=self.model.attachment_particle_body_count,
                    inputs=[
                        self.model.attachment_particle_body_body,
                        self.model.attachment_particle_body_particle,
                        self.model.attachment_particle_body_body_point,
                        self.model.attachment_particle_body_stiffness,
                        self.model.attachment_particle_body_enabled,
                        state_in.particle_q,
                        state_in.body_q,
                        self._attachment_particle_body_rho,
                    ],
                    outputs=[self._attachment_particle_body_lambda],
                    device=self.device,
                )

        update_tendon_diagnostics = (
            self.iterations > 0
            and self.tendon_segment_lambda is not None
            and not self.integrate_with_external_rigid_solver
        )
        if update_tendon_diagnostics:
            # The final body update changes tendon geometry after the last
            # in-iteration material solve. Accept its route state before
            # reporting tension or carrying rest lengths into the next step.
            # This accepted state is also where the unsupported-wrap
            # diagnostic is meaningful (mid-iteration predictor overshoot is
            # not), so report it here.
            self._update_tendon_routing(state_in, dt, True)
            self._snapshot_tendon_segment_length_reference(state_in.body_q, dt)

        # Opt-in contact force export: evaluate at the final iterate while the pose history the
        # iterations used is still intact (finalization advances it below).
        if observables is not None and observables.is_requested(SolverBase.ObservableKind.CONTACT_F):
            self._export_contact_forces(state_in, state_out, contacts, dt, observables.contact_f)

        # Snapshot solved rigid contact state for next-frame warm-start.
        self._snapshot_rigid_contact_history(contacts)
        self._finalize_rigid_bodies(state_in, state_out, dt)
        if update_tendon_diagnostics:
            self._update_tendon_segment_diagnostics(state_out, dt)
        if self.suture_needle_count > 0:
            self._update_suture_needles(state_in, contacts)
        self._finalize_particles(state_out, dt)
        self._save_tendon_history(state_out)
        if self._cut_blade_count > 0:
            self._update_cut_blades(state_out, contacts)

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset rigid solver history and optional body and particle state for selected worlds.

        Body fields selected by *flags* are copied from the model defaults.
        Joint penalty is restored to its minimum; joint C0 and AVBD dual history
        is zeroed immediately. Pose and enabled-rod friction history (curvature,
        stress, and increment) are rebaselined together from the next :meth:`step`
        input pose, after any intervening state edits or forward kinematics.
        Mutable tendon material and routing state is restored to its initialized
        values regardless of *flags*.
        REVOLUTE and one-axis-D6 winding re-anchors to the pose-equivalent angle
        nearest the authored rest coordinate; a pose alone cannot restore an
        independently intended multi-turn coordinate.
        Selected-world contact warm-start is cold-started when fresh rigid contacts
        are next processed. Internal rigid history is reset regardless of *flags*.
        When an external solver integrates the bodies, reset performs no rigid
        mutation; ``state`` and ``world_mask`` validation and particle reset still
        apply, but body State arrays are not accessed or validated.

        ``BODY_Q`` / ``BODY_QD`` copy ``model.body_q`` / ``model.body_qd`` into
        *state*; they do not restore a previously supplied state. A requested field
        is skipped if its *state* array is ``None``. If your initial pose differs
        from the model defaults, pass ``flags=0`` and author the pose any time
        before the next step; reset then preserves it and only clears VBD history.
        ``JOINT_Q`` / ``JOINT_QD`` are ignored (VBD uses maximal ``body_q`` /
        ``body_qd``); to reset from joint coordinates, run :func:`~newton.eval_fk`
        after reset so the resulting ``body_q`` supersedes reset's model copy.

        ``PARTICLE_Q`` / ``PARTICLE_QD`` likewise copy ``model.particle_q`` /
        ``model.particle_qd`` into *state* for particles in the selected worlds,
        using the same masking as the body fields (``world_mask=None`` also
        restores global ``world == -1`` particles; an explicit mask restores
        globals only through its final entry). One path covers both cloth and
        volumetric (tet) soft bodies, and it runs even when an external solver
        integrates the bodies or the model has none. A requested particle field is
        skipped if its *state* array is ``None``. Particle and body-particle solver
        history is intentionally left untouched except for retained attachment
        multipliers, which are cleared in selected worlds.
        ``particle_q_prev`` is rebaselined
        from the incoming state at the start of the next :meth:`step`, self-contact
        and body-particle contacts rebuild per step, and tet/cloth elasticity is
        stateless, so no particle history cold-start is required. Reset does not
        refresh the particle self-contact BVH; the next :meth:`step` refits it from
        the incoming positions. After a large reset displacement, call
        :meth:`rebuild_bvh` to restore acceleration-structure quality. Both reset
        and :meth:`rebuild_bvh` are graph-capturable, so either may run inside a
        captured episode-reset graph.

        Reset does not run collision detection, and :meth:`step` consumes the
        supplied contacts rather than rerunning
        :meth:`~newton.CollisionPipeline.collide`. After moving bodies or
        particles, regenerate contacts so stale soft contacts are not reused, and
        let the next :meth:`step` refresh rigid contact state. The next
        rigid :meth:`step` consumes the pose and rod rebaseline even when
        ``contacts=None``, so author the final pose (or run :func:`~newton.eval_fk`)
        before stepping; contact invalidation instead waits for a fresh refresh.
        VBD cold-starts its numeric contact state for reset-selected worlds.
        Frame-to-frame correspondence and sticky contact geometry remain owned by
        :class:`~newton.CollisionPipeline`; after a discontinuous episode reset,
        call :meth:`~newton.CollisionPipeline.reset_contact_matching` with the
        same world mask to discard that history. Reset does not change
        ``set_rigid_history_update()``; leave rigid history refresh enabled for the
        next contact-bearing step. Reusing contacts
        (``set_rigid_history_update(False)``) is unsupported only while contact
        invalidation is still pending.

        Args:
            state: The simulation state to reset (modified in place).
            world_mask: One-dimensional Warp boolean mask on the solver device.
                Shape ``(world_count + 1,)``, with the final entry selecting
                entities in global world ``-1``. ``None`` selects all local and
                global entities.
            flags: :class:`~newton.StateFlags` (or ``int``) selecting which body
                and particle fields to copy from the model defaults. VBD honors
                :attr:`~newton.StateFlags.BODY_Q`,
                :attr:`~newton.StateFlags.BODY_QD`,
                :attr:`~newton.StateFlags.PARTICLE_Q`, and
                :attr:`~newton.StateFlags.PARTICLE_QD`; ``None`` requests all flags.
        """
        if state is None:
            raise ValueError("'state' argument is required.")
        model = self.model
        world_mask = self._normalize_reset_world_mask(world_mask)

        flags_value = int(StateFlags.ALL if flags is None else flags)

        # Only requested BODY flags reach the launch as actionable arrays; everything
        # else stays None so an unrequested (possibly wrong-device) State array never
        # binds, and a supplied array is itself the kernel's reset signal.
        body_q = None
        body_qd = None
        if self._integrates_rigid_bodies:
            if flags_value & int(StateFlags.BODY_Q) and state.body_q is not None:
                if state.body_q.device != self.device:
                    raise ValueError(
                        f"state.body_q is on device {state.body_q.device}, expected solver device {self.device}."
                    )
                body_q = state.body_q
            if flags_value & int(StateFlags.BODY_QD) and state.body_qd is not None:
                if state.body_qd.device != self.device:
                    raise ValueError(
                        f"state.body_qd is on device {state.body_qd.device}, expected solver device {self.device}."
                    )
                body_qd = state.body_qd

        # Particle state reset mirrors the rigid path: only a requested, on-device
        # State array binds, and a bound array is itself the kernel's per-field
        # reset signal. This runs before the rigid early-return so particle-only
        # models and external-rigid coupling still restore deformables.
        if model.particle_count > 0:
            particle_q = None
            particle_qd = None
            if flags_value & int(StateFlags.PARTICLE_Q) and state.particle_q is not None:
                if state.particle_q.device != self.device:
                    raise ValueError(
                        f"state.particle_q is on device {state.particle_q.device}, "
                        f"expected solver device {self.device}."
                    )
                particle_q = state.particle_q
            if flags_value & int(StateFlags.PARTICLE_QD) and state.particle_qd is not None:
                if state.particle_qd.device != self.device:
                    raise ValueError(
                        f"state.particle_qd is on device {state.particle_qd.device}, "
                        f"expected solver device {self.device}."
                    )
                particle_qd = state.particle_qd
            # A cut edits model arrays (tet corners, masses, surface rows). A particle position reset
            # undoes every cut first, for the whole model only.
            if self._cut_has_capacity and particle_q is not None:
                if world_mask is not None:
                    raise NotImplementedError("A model with cut capacity can be reset for the whole model only")
                self._reset_cut_system()
            if particle_q is not None or particle_qd is not None:
                wp.launch(
                    kernel=reset_particle_state,
                    dim=model.particle_count,
                    inputs=[
                        world_mask,
                        world_mask is None,
                        model.world_count,
                        model.particle_world,
                        model.particle_q,
                        model.particle_qd,
                    ],
                    outputs=[particle_q, particle_qd],
                    device=self.device,
                )

        self._reset_tendon_state(state, world_mask)

        if model.attachment_particle_body_count > 0:
            wp.launch(
                kernel=reset_particle_body_attachment_state,
                dim=model.attachment_particle_body_count,
                inputs=[world_mask, world_mask is None, model.world_count, model.attachment_particle_body_world],
                outputs=[self._attachment_particle_body_lambda],
                device=self.device,
            )

        if not self._integrates_rigid_bodies:
            return

        # Entity-parallel launch over the widest lane: world slots (+1 global),
        # joints, and bodies (bodies only when copying model-default body state).
        reset_dim = max(
            model.world_count + 1,
            model.joint_count,
            model.body_count if (body_q is not None or body_qd is not None) else 0,
        )
        wp.launch(
            kernel=reset_rigid_state,
            dim=reset_dim,
            inputs=[
                world_mask,
                world_mask is None,
                model.world_count,
                model.body_world,
                model.joint_world,
                self.joint_constraint_start,
                self.joint_constraint_dim,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.body_q,
                model.body_qd,
                self.joint_penalty_k_min,
            ],
            outputs=[
                body_q,
                body_qd,
                self.joint_penalty_k,
                self.joint_C0_lin,
                self.joint_C0_ang,
                self.joint_lambda_lin,
                self.joint_lambda_ang,
                self.joint_drive_lambda,
                self.joint_limit_lambda,
                self._rigid_pose_rebaseline_mask,
                self._contact_history_reset_mask,
                self._contact_history_reset_pending,
            ],
            device=self.device,
        )

    def _snapshot_rigid_contact_history(self, contacts: Contacts | None, *, force: bool = False):
        """Snapshot solved contact state for persistent or in-step matched restoration."""
        if (not self.rigid_contact_history and not force) or contacts is None:
            return

        if not self._integrates_rigid_bodies:
            return

        contact_launch_dim = contacts.rigid_contact_max
        if contact_launch_dim == 0:
            return

        if self._prev_contact_lambda is None or self._prev_contact_lambda.shape[0] < contact_launch_dim:
            history_cap = 0 if self._prev_contact_lambda is None else self._prev_contact_lambda.shape[0]
            self._raise_if_capturing_resize("rigid contact history", history_cap, contact_launch_dim)
            self._init_rigid_contact_warmstart(contact_launch_dim)

        # Snapshot solved contact rows for the next step's warm-start.
        wp.launch(
            kernel=snapshot_body_body_contact_history,
            dim=contact_launch_dim,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_normal,
                self.body_body_contact_lambda,
                self.body_body_contact_penalty_k,
                contacts.contact_generation,
                self._contact_history_buffer_id(contacts),
            ],
            outputs=[
                self._prev_contact_lambda,
                self._prev_contact_penalty_k,
                self._prev_contact_normal,
                self._prev_contact_frame,
            ],
            device=self.device,
        )

    def _init_rigid_penetration_free(self, model: Model, rigid_soft_dat_use_interval_arithmetic: bool):
        """Initialize rigid-soft DAT state and motion budgets.

        A particle-only model can still collide with static rigid shapes, so the
        zero-body case retains the particle-side DAT state with empty body arrays.
        """
        self.rigid_soft_dat_use_interval_arithmetic = rigid_soft_dat_use_interval_arithmetic

        if not self.rigid_soft_enable_dat:
            return

        if self.collision_pipeline is None:
            raise ValueError(
                "rigid_soft_enable_dat requires a solver-owned collision pipeline "
                "(pass collision_pipeline=CollisionPipeline(...) to SolverVBD): the DAT reference poses "
                "must be snapshotted at the exact detection instants the solver drives."
            )
        has_rigid_soft_queries = self.collision_pipeline._has_rigid_soft_queries
        if self.integrate_with_external_rigid_solver:
            raise ValueError("rigid_soft_enable_dat is not supported with an external rigid solver.")

        self.body_truncation_ts = wp.zeros(model.body_count, dtype=float, device=self.device)
        # Reference poses at the last rigid collision detection (mirror of
        # pos_prev_collision_detection): rigid pose updates accumulate from here and the
        # per-body motion budget below is measured from here.
        self.body_q_prev_collision_detection = wp.clone(model.body_q, device=self.device)

        self._init_rigid_dat_body_bounding_radius(model)

        # Every SDF pass prunes at soft_contact_gap + shape_margin + feature radius, and every
        # feature radius is at least the model's minimum particle radius, so the minimum particle
        # radius plus the minimum shape margin is a lower bound on every rigid-soft query radius.
        # Do not filter mutable ACTIVE/COLLIDE_PARTICLES flags: the pipeline's candidate tables
        # deliberately remain valid when those flags are enabled after construction.
        rigid_soft_query_radius_min = np.inf
        if has_rigid_soft_queries and model.particle_count > 0 and model.shape_count > 0:
            particle_radius_min = float(np.min(model.particle_radius.numpy()))
            shape_margin_min = float(np.min(model.shape_margin.numpy()))
            # A variant row can take a variant shape with a smaller margin after construction.
            variant_margin = model._variant_shape_geometry.get("shape_margin")
            if variant_margin is not None and variant_margin.shape[0] > 0:
                shape_margin_min = min(shape_margin_min, float(np.min(variant_margin.numpy())))
            rigid_soft_query_radius_min = (
                float(self.collision_pipeline.soft_contact_gap) + particle_radius_min + shape_margin_min
            )
            if not np.isfinite(rigid_soft_query_radius_min) or rigid_soft_query_radius_min <= 0.0:
                raise ValueError(
                    "rigid_soft_enable_dat requires a positive minimum rigid-soft query "
                    "radius (soft_contact_gap + minimum particle radius + minimum shape margin)"
                )

        self._rigid_soft_query_radius_min = rigid_soft_query_radius_min
        if np.isfinite(rigid_soft_query_radius_min):
            rigid_soft_max_displacement = 0.5 * self.dat_conservative_bound_relaxation * rigid_soft_query_radius_min
        else:
            rigid_soft_max_displacement = wp.inf
        self._rigid_dat_particle_max_displacement = rigid_soft_max_displacement
        self._rigid_dat_body_max_displacement = wp.full(
            model.body_count, rigid_soft_max_displacement, dtype=float, device=self.device
        )

    def _init_cut_system(
        self,
        model: Model,
        margin: float,
        gap: float,
        deterministic,
        enable_self_contact: bool,
        filter_threshold: int,
        exclusion_radius: float,
        edge_parallel_epsilon: float,
        external_vertex_filter_map: dict | None,
        external_edge_filter_map: dict | None,
        armed: bool,
        update_interval: int,
        start_gap: float | None,
        contact_range: str | float,
        keep_side_labels: bool,
        gap_direction: str,
        mass_rule: str,
        surface_edges: bool,
        press_wound: bool,
        gate_wound: bool,
        method: str,
        tie_stiffness: float | None,
        tie_dual: bool,
        min_fraction: float,
        wound_mass_fraction: float,
        contact_filter: bool,
        front_rows: bool,
    ) -> None:
        """Check the settings a cuttable model needs and allocate every array of the cut once.

        Builds the tet-neighbor table that the cut shares with the suture, the solver-owned
        vertex-to-tet table with room for copies, and the tet sources of the skin rows. With
        ``method == "xfem"`` it also builds the tables of the XFEM-type cut.
        """
        if method not in ("faces", "xfem"):
            raise ValueError(f"Unknown cut_method {method!r}")
        self.cut_method = method
        self._cut_xfem = False
        self.cut_xfem_contact_filter = bool(contact_filter)
        self.cut_xfem_front_rows = bool(front_rows)
        self._tet_neighbors = None
        self._cut_blade_count = int(getattr(model, "cut_blade_count", 0) or 0)
        capacity = getattr(model, "cut_capacity", None)
        self._cut_has_capacity = capacity is not None
        self._cut_idle_tet = int(capacity["idle_tet"]) if capacity is not None else -1
        # Tissue tets come first; the spare tets of the cut capacity and the idle tet follow them.
        self._cut_first_tet_count = int(capacity["tet_spare_start"]) if capacity is not None else model.tet_count
        if self._cut_blade_count > 0 and capacity is None:
            raise ValueError("A model with a cut blade needs cut capacity; call ModelBuilder.add_cut_capacity()")
        if capacity is None:
            if method == "xfem":
                warnings.warn(
                    'cut_method="xfem" has no effect: the model has no cut capacity '
                    "(ModelBuilder.add_cut_capacity) and no cut blade",
                    NewtonWarning,
                    stacklevel=3,
                )
            return

        # ---- rules of a model with cut capacity ----
        if model.particle_world_start is not None and model.world_count > 1:
            raise NotImplementedError("Cutting supports a single world for now")
        if not _is_tet_only_elasticity_model(model):
            raise ValueError("A model with cut capacity must have no triangle or edge stiffness")
        if deterministic != wp.DeterministicMode.NOT_GUARANTEED:
            raise ValueError(
                "Cutting takes spare particles through atomic counters; a Warp deterministic mode is not supported"
            )
        vbd_attrs = getattr(model, "vbd", None)
        if vbd_attrs is None or not hasattr(vbd_attrs, "cut_force"):
            raise ValueError(
                "Cutting needs the SolverVBD custom attributes; call "
                "SolverVBD.register_custom_attributes(builder) before adding particles"
            )
        if not enable_self_contact:
            raise ValueError("A model with cut capacity needs particle_enable_self_contact=True")
        if filter_threshold > 1:
            raise ValueError("A model with cut capacity needs particle_topological_contact_filter_threshold <= 1")
        if exclusion_radius != 0.0:
            raise ValueError("A model with cut capacity needs particle_rest_shape_contact_exclusion_radius == 0")
        if external_vertex_filter_map is not None or external_edge_filter_map is not None:
            raise ValueError("A model with cut capacity does not support external contact filter maps")
        if update_interval < 1:
            raise ValueError("cut_update_interval must be at least 1")
        if gap_direction not in ("largest_least_opening", "normal_sum"):
            raise ValueError(f"Unknown cut_gap_direction {gap_direction!r}")
        if mass_rule not in ("rest_volume", "full_copy"):
            raise ValueError(f"Unknown cut_mass_rule {mass_rule!r}")
        if method == "xfem":
            # Switches of the cut along tet faces: the XFEM-type cut does not read them.
            if press_wound:
                raise ValueError(
                    'cut_press_wound=True belongs to cut_method="faces"; cut_method="xfem" has the plain rule and '
                    "the gate rule (cut_gate_wound) only"
                )
            unused = [
                name
                for name, value, default in (
                    ("cut_keep_side_labels", bool(keep_side_labels), True),
                    ("cut_gap_direction", gap_direction, "largest_least_opening"),
                    ("cut_surface_edges", bool(surface_edges), True),
                )
                if value != default
            ]
            if unused:
                warnings.warn(
                    f'{", ".join(unused)} belong{"s" if len(unused) == 1 else ""} to cut_method="faces" and '
                    f'{"is" if len(unused) == 1 else "are"} not used with cut_method="xfem"',
                    NewtonWarning,
                    stacklevel=3,
                )

        device = self.device
        particle_count = model.particle_count
        color_count = int(capacity["color_count"])
        spare_count = int(capacity["spare_count"])
        cut_force = vbd_attrs.cut_force.numpy()
        cuttable = np.isfinite(cut_force) & (cut_force < 1.0e30)
        rest_q = model.particle_q.numpy()
        radius = model.particle_radius.numpy()
        flags = model.particle_flags.numpy()
        parked = np.arange(int(capacity["particle_start"]), int(capacity["particle_end"]))
        spare_start = model.cut_spare_start.numpy()
        spare_particles = np.concatenate([np.arange(start, start + spare_count) for start in spare_start])
        query = float(margin) + float(gap)
        is_parked = np.zeros(particle_count, dtype=bool)
        is_parked[parked] = True
        cuttable = cuttable & ~is_parked  # a scene may fill vbd.cut_force for every particle
        if int(getattr(model, "suture_needle_count", 0) or 0) > 30:
            raise ValueError("A model with cut capacity supports at most 30 suture needles (one gate bit is the cut's)")
        if model.spring_count > 0 and cuttable[model.spring_indices.numpy().reshape(-1)].any():
            raise ValueError("A cuttable particle must not be the end of a spring")
        # The copy color layout as add_cut_capacity() made it.
        copy_colors = bool(capacity["copy_colors"])
        particle_colors = model.particle_colors.numpy()
        if len(model.particle_color_groups) != (2 if copy_colors else 1) * color_count:
            raise ValueError("The particle color groups were changed after add_cut_capacity()")
        for color, start in enumerate(spare_start):
            expected = color_count + color if copy_colors else color
            if (particle_colors[start : start + spare_count] != expected).any():
                raise ValueError("The particle color groups were changed after add_cut_capacity()")
        if len(parked) > 1:
            parked_q = rest_q[parked]
            closest = np.inf
            for start in range(0, len(parked), 1024):
                d = np.linalg.norm(parked_q[start : start + 1024, None, :] - parked_q[None, :, :], axis=2)
                rows = np.arange(d.shape[0])
                d[rows, start + rows] = np.inf
                closest = min(closest, float(d.min()))
            if closest <= query:
                raise ValueError(
                    f"Parking particles are {closest * 1e3:.2f} mm apart; the spacing must exceed the "
                    f"self-contact margin plus gap ({query * 1e3:.2f} mm)"
                )
        active = np.flatnonzero((flags & int(ParticleFlags.ACTIVE)) != 0)
        if len(active) > 0 and len(parked) > 0:
            lower = rest_q[parked].min(axis=0) - query
            upper = rest_q[parked].max(axis=0) + query
            near = active[np.all((rest_q[active] >= lower) & (rest_q[active] <= upper), axis=1)]
            if len(near) > 0:
                raise ValueError("An active particle starts inside the parking lattice of the cut capacity")
        if not np.allclose(radius[parked], radius[parked][0]) or (
            cuttable.any() and not np.allclose(radius[cuttable], radius[parked][0])
        ):
            raise ValueError("Spare and parking particles must have the radius of every cuttable particle")
        # At the start no shape that collides with particles is within its rigid-soft detection distance
        # of a parked particle: edge and face contact records also see inactive particles.
        if model.shape_count > 0 and len(parked) > 0:
            soft_gap = float(getattr(self.collision_pipeline, "soft_contact_gap", 0.0) or 0.0)
            shape_flags = model.shape_flags.numpy()
            shape_type = model.shape_type.numpy()
            shape_scale = model.shape_scale.numpy()
            shape_xform = model.shape_transform.numpy()
            shape_body = model.shape_body.numpy()
            shape_bound = model.shape_collision_radius.numpy()
            margins = (
                model.shape_margin.numpy()
                if model.shape_margin is not None and model.shape_margin.shape[0] > 0
                else np.zeros(model.shape_count, dtype=np.float32)
            )
            body_q = model.body_q.numpy() if model.body_count > 0 else None
            parked_q = rest_q[parked].astype(np.float64)
            for shape in np.flatnonzero((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0):
                xform = wp.transform(*shape_xform[shape])
                if shape_body[shape] >= 0:
                    xform = wp.transform_multiply(wp.transform(*body_q[shape_body[shape]]), xform)
                rotation = np.array(wp.quat_to_matrix(wp.transform_get_rotation(xform)), dtype=np.float64).reshape(3, 3)
                local = (parked_q - np.array(wp.transform_get_translation(xform), dtype=np.float64)) @ rotation
                scale = shape_scale[shape].astype(np.float64)
                kind = int(shape_type[shape])
                if kind == int(GeoType.BOX):
                    outside = np.abs(local) - scale
                    distance = np.linalg.norm(np.maximum(outside, 0.0), axis=1) + np.minimum(outside.max(axis=1), 0.0)
                elif kind == int(GeoType.CAPSULE):
                    axis_point = np.zeros_like(local)
                    axis_point[:, 2] = np.clip(local[:, 2], -scale[1], scale[1])
                    distance = np.linalg.norm(local - axis_point, axis=1) - scale[0]
                elif kind == int(GeoType.SPHERE):
                    distance = np.linalg.norm(local, axis=1) - scale[0]
                elif kind == int(GeoType.PLANE):
                    distance = local[:, 2]
                else:
                    distance = np.linalg.norm(local, axis=1) - float(shape_bound[shape])  # bounding sphere
                reach = float(radius[parked].max()) + float(margins[shape]) + soft_gap
                if float(distance.min()) <= reach:
                    raise ValueError(
                        f"Shape {int(shape)} starts within its contact detection distance of the parking lattice of "
                        "the cut capacity; move the lattice (park_position) away from every shape"
                    )

        tet_indices = model.tet_indices.numpy().reshape(-1, 4)
        cuttable_tets = cuttable[tet_indices].all(axis=1)
        self._cut_start_gap = float(margin if start_gap is None else start_gap)
        if cuttable_tets.any():
            x = rest_q[tet_indices[cuttable_tets]]
            edges = np.concatenate([np.linalg.norm(x[:, a] - x[:, b], axis=1) for a, b in _TET_EDGE_PAIRS])
            if edge_parallel_epsilon >= float(edges.min()) ** 2:
                warnings.warn(
                    f"particle_edge_parallel_epsilon ({edge_parallel_epsilon:g}) is not below the squared shortest "
                    f"edge of a cuttable tet ({float(edges.min()) ** 2:g} m^2): such edges act as points in "
                    "edge-edge self contact",
                    NewtonWarning,
                    stacklevel=3,
                )
            volume = (
                np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])) / 6.0
            )
            areas = [
                0.5 * np.linalg.norm(np.cross(x[:, b] - x[:, a], x[:, c] - x[:, a]), axis=1)
                for a, b, c in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
            ]
            height = float((3.0 * volume / np.maximum(np.max(areas, axis=0), 1.0e-30)).min())
            if self._cut_start_gap > height:
                warnings.warn(
                    f"cut_start_gap ({self._cut_start_gap * 1e3:.2f} mm) is larger than the smallest height of a "
                    f"cuttable tet ({height * 1e3:.2f} mm): a tet at the wound can start inverted",
                    NewtonWarning,
                    stacklevel=3,
                )

        # ---- rules of a model with a blade ----
        blade_count = self._cut_blade_count
        if blade_count > 1:
            raise NotImplementedError("Cutting supports one cut blade for now")
        if blade_count > 0:
            if self.collision_pipeline is None or not getattr(
                self.collision_pipeline, "enable_rigid_soft_full_surface_contact", False
            ):
                raise ValueError(
                    "A cut blade needs a solver-owned CollisionPipeline with enable_rigid_soft_full_surface_contact=True"
                )
            if not self.rigid_soft_enable_dat:
                raise ValueError("A cut blade needs rigid_soft_enable_dat=True")
            if getattr(self.collision_pipeline, "_soft_mesh_contact_data", None) is not None:
                raise NotImplementedError(
                    "A model with a cut blade supports box and capsule shapes in contact with particles; "
                    "mesh shapes read surface adjacency that a cut does not update"
                )
            blade_body = model.cut_blade_body.numpy()
            if (model.body_inv_mass.numpy()[blade_body] <= 0.0).any() or (
                (model.body_flags.numpy()[blade_body] & int(BodyFlags.KINEMATIC)) != 0
            ).any():
                raise ValueError("The body of a cut blade must be dynamic")
            if not cuttable.any():
                warnings.warn(
                    "The model has a cut blade but no particle with a finite vbd.cut_force at solver construction; "
                    "fill vbd.cut_force before creating SolverVBD",
                    NewtonWarning,
                    stacklevel=3,
                )

        # ---- shared tet-neighbor table ----
        # The spare tets share the corners of the idle tet; they have no neighbors.
        tet_count = tet_indices.shape[0]
        first_tet_count = self._cut_first_tet_count
        spare_tet_count = int(capacity["tet_spare_count"])
        neighbors = np.full((tet_count, 4), -1, dtype=np.int32)
        neighbors[:first_tet_count] = build_tet_neighbors(tet_indices[:first_tet_count]).reshape(-1, 4)
        self._tet_neighbors = wp.array(neighbors, dtype=wp.int32, device=device)
        self._cut_xfem = method == "xfem"

        # ---- solver-owned vertex-to-tet table with room for copies ----
        # The XFEM-type cut gives free tet records to every particle of the tissue tets, cuttable at
        # construction or not: its kernels read vbd.cut_force at every cut update, so a scene can make
        # tissue cuttable later.
        shared = model.soft_mesh_adjacency_device
        tissue_particles = np.setdiff1d(np.unique(tet_indices[:first_tet_count]), parked)
        pairs, offsets, max_tets = build_padded_tet_adjacency(
            shared.v_adj_tets.numpy(),
            shared.v_adj_tets_offsets.numpy(),
            spare_particles,
            self._cut_idle_tet,
            dropped_tets=np.arange(first_tet_count, first_tet_count + spare_tet_count),
            doubled_particles=tissue_particles if self._cut_xfem else None,
        )
        adjacency = MeshAdjacencyData()
        adjacency.edge_tri_indices = shared.edge_tri_indices
        adjacency.tri_edge_indices = shared.tri_edge_indices
        adjacency.v_adj_tris = shared.v_adj_tris
        adjacency.v_adj_tris_offsets = shared.v_adj_tris_offsets
        adjacency.v_adj_edges = shared.v_adj_edges
        adjacency.v_adj_edges_offsets = shared.v_adj_edges_offsets
        adjacency.v_adj_springs = shared.v_adj_springs
        adjacency.v_adj_springs_offsets = shared.v_adj_springs_offsets
        adjacency.v_adj_tets = wp.array(pairs, dtype=wp.int32, device=device)
        adjacency.v_adj_tets_offsets = wp.array(offsets, dtype=wp.int32, device=device)
        self.particle_adjacency = adjacency
        self._cut_max_tets_per_vertex = max_tets

        # ---- settings ----
        self.cut_armed = bool(armed)
        self.cut_update_interval = int(update_interval)
        self.cut_keep_side_labels = bool(keep_side_labels)
        self.cut_gap_direction = gap_direction
        self.cut_mass_rule = mass_rule
        self.cut_surface_edges = bool(surface_edges)
        self.cut_press_wound = bool(press_wound)
        self.cut_gate_wound = bool(gate_wound)

        # ---- blade force (chapter 2) ----
        self._cut_substep = wp.zeros(1, dtype=wp.int32, device=device)
        self._cut_due = wp.zeros(1, dtype=wp.int32, device=device)
        self._cut_update_number = wp.zeros(1, dtype=wp.int32, device=device)
        blades = max(blade_count, 1)
        self._cut_blade_force = wp.zeros(blades, dtype=float, device=device)
        self._cut_blade_record_key = wp.zeros(blades, dtype=wp.int64, device=device)
        self._cut_contact_count = wp.zeros(blades, dtype=wp.int32, device=device)
        self._cut_advance = wp.zeros(blades, dtype=wp.int32, device=device)
        shape_blade = np.full(model.shape_count, -1, dtype=np.int32)
        ranges = np.zeros(blades, dtype=np.float32)
        if blade_count > 0:
            edge_shapes = model.cut_blade_edge_shape.numpy()
            shape_scale = model.shape_scale.numpy()
            shape_margin = (
                model.shape_margin.numpy()
                if model.shape_margin is not None and model.shape_margin.shape[0] > 0
                else np.zeros(model.shape_count, dtype=np.float32)
            )
            soft_gap = float(getattr(self.collision_pipeline, "soft_contact_gap", 0.0) or 0.0)
            particle_radius = float(radius[cuttable].max()) if cuttable.any() else float(radius[parked].max())
            for blade in range(blade_count):
                shape = int(edge_shapes[blade])
                shape_blade[shape] = blade
                force_range = float(shape_scale[shape][0]) + float(shape_margin[shape]) + particle_radius
                if contact_range == "detection":
                    ranges[blade] = force_range + soft_gap
                elif contact_range == "force":
                    ranges[blade] = force_range
                elif isinstance(contact_range, (int, float)) and contact_range > 0.0:
                    ranges[blade] = float(contact_range)
                else:
                    raise ValueError(
                        f"cut_contact_range must be 'detection', 'force' or a positive value, got {contact_range!r}"
                    )
            # Fork: with cut_gate_wound the blade has no contact with split vertices and copies (the wound
            # lips). The bit is written for both switch values, so a second solver on the same model sees its own.
            shape_gate = model.shape_contact_gate.numpy()
            shape_body = model.shape_body.numpy()
            for body in model.cut_blade_body.numpy():
                if self.cut_gate_wound:
                    shape_gate[shape_body == body] |= np.int32(CUT_GATE_BIT)
                else:
                    shape_gate[shape_body == body] &= np.int32(~CUT_GATE_BIT)
            model.shape_contact_gate.assign(shape_gate)
        self._cut_shape_blade = wp.array(shape_blade, dtype=wp.int32, device=device)
        self._cut_contact_range = wp.array(ranges, dtype=wp.float32, device=device)

        # ---- cut faces (chapter 3) ----
        self._cut_particle_pressed = wp.zeros(particle_count, dtype=wp.int32, device=device)
        self._cut_particle_wound = wp.zeros(particle_count, dtype=wp.int32, device=device)
        self._cut_tet_side = wp.zeros(tet_count, dtype=wp.int32, device=device)
        self._cut_face_side = wp.zeros((tet_count, 4), dtype=wp.int32, device=device)
        self._cut_face_stamp = wp.full((tet_count, 4), -1, dtype=wp.int32, device=device)
        self._cut_particle_dirty = wp.zeros(particle_count, dtype=wp.int32, device=device)

        # ---- split and start gap (chapter 4) ----
        self._cut_tet_record_group = wp.full(max(len(pairs) // 2, 1), -1, dtype=wp.int32, device=device)
        self._cut_pool_taken = wp.zeros(color_count, dtype=wp.int32, device=device)
        # Failed takes: one entry per pool, then triangle rows, then edge rows; the XFEM-type cut adds
        # spare tets and tet records.
        self._cut_overflow = wp.zeros(color_count + (4 if self._cut_xfem else 2), dtype=wp.int32, device=device)
        self._cut_unsafe = wp.zeros(1, dtype=wp.int32, device=device)
        attached = np.zeros(particle_count, dtype=np.int32)
        if model.attachment_particle_body_count > 0:
            attached[model.attachment_particle_body_particle.numpy()] = 1
        self._cut_particle_attached = wp.array(attached, dtype=wp.int32, device=device)
        self._cut_particle_new = wp.zeros(particle_count, dtype=wp.int32, device=device)
        self._cut_offset = wp.zeros(particle_count, dtype=wp.vec3, device=device)

        # ---- surface rows (chapter 5) ----
        tri_indices = (
            model.tri_indices.numpy().reshape(-1, 3) if model.tri_count > 0 else np.zeros((0, 3), dtype=np.int32)
        )
        edge_indices = (
            model.edge_indices.numpy().reshape(-1, 4) if model.edge_count > 0 else np.zeros((0, 4), dtype=np.int32)
        )
        tri_source, edge_source, first_face_row, first_edge_row = build_cut_row_sources(
            tet_indices[:first_tet_count], neighbors[:first_tet_count], tri_indices, edge_indices
        )
        tet_face_row = np.full((tet_count, 4), -1, dtype=np.int32)
        tet_face_row[:first_tet_count] = first_face_row
        tet_edge_row = np.full((tet_count, 6), -1, dtype=np.int32)
        tet_edge_row[:first_tet_count] = first_edge_row
        self._cut_tri_source = wp.array(tri_source, dtype=wp.int32, device=device)
        self._cut_edge_source = wp.array(edge_source, dtype=wp.int32, device=device)
        self._cut_tet_face_row = wp.array(tet_face_row, dtype=wp.int32, device=device)
        self._cut_tet_edge_row = wp.array(tet_edge_row, dtype=wp.int32, device=device)
        self._cut_tri_taken = wp.zeros(1, dtype=wp.int32, device=device)
        self._cut_edge_taken = wp.zeros(1, dtype=wp.int32, device=device)
        self._cut_empty_indices = wp.zeros(1, dtype=wp.vec3i, device=device)
        self._cut_empty_vec3 = wp.zeros(1, dtype=wp.vec3, device=device)

        # ---- arrays that a full particle reset restores ----
        self._cut_saved = {
            "tet_indices": wp.clone(model.tet_indices),
            "v_adj_tets": wp.clone(adjacency.v_adj_tets),
            "particle_mass": wp.clone(model.particle_mass),
            "particle_inv_mass": wp.clone(model.particle_inv_mass),
            "particle_flags": wp.clone(model.particle_flags),
            "particle_q": wp.clone(model.particle_q),
            "tri_indices": wp.clone(model.tri_indices),
            "edge_indices": wp.clone(model.edge_indices),
            "tet_neighbors": wp.clone(self._tet_neighbors),
            "tri_source": wp.clone(self._cut_tri_source),
            "edge_source": wp.clone(self._cut_edge_source),
            "tet_face_row": wp.clone(self._cut_tet_face_row),
            "tet_edge_row": wp.clone(self._cut_tet_edge_row),
        }

        if self._cut_xfem:
            # Edge lengths of all tissue tets, not only of the tets that are cuttable now: the range of the
            # level values must hold for tissue that a scene makes cuttable after construction.
            longest_edge = 0.0
            shortest_edge = 0.0
            if first_tet_count > 0:
                x = rest_q[tet_indices[:first_tet_count]]
                lengths = np.concatenate([np.linalg.norm(x[:, a] - x[:, b], axis=1) for a, b in _TET_EDGE_PAIRS])
                longest_edge = float(lengths.max())
                shortest_edge = float(lengths.min())
            cut_force_max = float(cut_force[cuttable].max()) if cuttable.any() else 0.0
            self._init_cut_xfem_system(
                model,
                tet_indices,
                tri_source,
                edge_source,
                float(ranges[0]) + longest_edge if blade_count > 0 else 0.0,
                shortest_edge,
                edge_parallel_epsilon,
                len(pairs) // 2,
                tie_stiffness,
                tie_dual,
                min_fraction,
                wound_mass_fraction,
                cut_force_max,
                float(radius[parked].max()),
            )

    def _init_cut_xfem_system(
        self,
        model: Model,
        tet_indices: np.ndarray,
        tri_source: np.ndarray,
        edge_source: np.ndarray,
        level_range: float,
        shortest_edge: float,
        edge_parallel_epsilon: float,
        record_count: int,
        tie_stiffness: float | None,
        tie_dual: bool,
        min_fraction: float,
        wound_mass_fraction: float,
        cut_force_max: float,
        particle_radius: float,
    ) -> None:
        """Check the settings of the XFEM-type cut and allocate its tables once.

        Host tables: the mesh edges of the tissue tets and the mesh edges of the tets and of the
        surface rows. Device arrays: level values, edge parameters, wound particles and their ties,
        fragment rows, ghosts and the kinds of the surface rows. Every array that a cut edits is
        cloned for :meth:`_reset_cut_system`.

        Args:
            cut_force_max: Largest finite ``vbd.cut_force`` at construction [N]; 0 when none.
            particle_radius: Radius of the spare particles, which is the radius of every cuttable
                particle [m].
        """
        device = self.device
        capacity = model.cut_capacity
        particle_count = model.particle_count
        tet_count = model.tet_count
        first_tet_count = self._cut_first_tet_count
        spare_tet_count = int(capacity["tet_spare_count"])
        tri_spare_start = int(capacity["tri_spare_start"])
        tri_spare_count = int(capacity["tri_spare_count"])
        edge_spare_start = int(capacity["edge_spare_start"])
        edge_spare_count = int(capacity["edge_spare_count"])
        wound_tri_count = XFEM_WOUND_TRIS_PER_TET * spare_tet_count
        wound_edge_count = XFEM_WOUND_EDGES_PER_TET * spare_tet_count
        if not capacity["copy_colors"]:
            raise ValueError('cut_method="xfem" needs add_cut_capacity(copy_colors=True)')
        if spare_tet_count < 1:
            raise ValueError('cut_method="xfem" needs spare tets; call add_cut_capacity(max_tets=...)')
        if wound_tri_count > tri_spare_count:
            raise ValueError(
                f'cut_method="xfem" needs four spare triangle rows per spare tet ({wound_tri_count}); '
                f"the cut capacity has {tri_spare_count}; raise max_faces"
            )
        if wound_edge_count > edge_spare_count:
            raise ValueError(
                f'cut_method="xfem" needs ten spare edge rows per spare tet ({wound_edge_count}); '
                f"the cut capacity has {edge_spare_count}; raise max_edges"
            )
        if self.cut_xfem_front_rows and wound_edge_count >= edge_spare_count:
            raise ValueError(
                "cut_xfem_front_rows=True needs one spare edge row per crossed mesh edge beyond the ten per "
                f"spare tet ({wound_edge_count}); the cut capacity has {edge_spare_count}; raise max_edges"
            )
        if not (0.0 < min_fraction < 0.5):
            raise ValueError("cut_xfem_min_fraction must be in (0, 0.5)")
        if not (0.0 < wound_mass_fraction < 1.0):
            raise ValueError("cut_wound_mass_fraction must be in (0, 1)")
        if tie_stiffness is None:
            tie_stiffness = 2.0 * float(model.soft_contact_ke)
        if not (tie_stiffness > 0.0):
            raise ValueError("cut_tie_stiffness must be positive")
        if shortest_edge > 0.0 and (min_fraction * shortest_edge) ** 2 <= edge_parallel_epsilon:
            warnings.warn(
                f"cut_xfem_min_fraction ({min_fraction:g}) times the shortest edge of a tissue tet "
                f"({shortest_edge * 1e3:.2f} mm) is not above the square root of particle_edge_parallel_epsilon "
                f"({edge_parallel_epsilon**0.5 * 1e3:.2f} mm): a short part of a cut edge acts as a point in "
                "edge-edge self contact",
                NewtonWarning,
                stacklevel=4,
            )
        if int(getattr(model, "suture_needle_count", 0) or 0) > 0:
            warnings.warn(
                'cut_method="xfem" on a model with suture needles: the suture does not know fragments yet. A '
                "tunnel through a cut tet binds to fragment A only, and no tunnel opens at a wound face. Keep "
                "the needles away from cut tissue.",
                NewtonWarning,
                stacklevel=4,
            )
        if (
            self._cut_blade_count > 0
            and cut_force_max > 0.0
            and not self.cut_armed
            and not self.rigid_soft_contact_use_log_barrier
        ):
            # The penalty law gives one contact record of the blade edge at most the contact stiffness times
            # its collision radius, at zero distance, where the rigid-soft truncation stops the blade edge.
            shape = int(model.cut_blade_edge_shape.numpy()[0])
            margin = float(model.shape_margin.numpy()[shape]) if model.shape_margin is not None else 0.0
            stiffness = 0.5 * (float(model.soft_contact_ke) + float(model.shape_material_ke.numpy()[shape]))
            record_limit = stiffness * (particle_radius + margin)
            if cut_force_max > _XFEM_BLADE_RECORD_LIMIT_FACTOR * record_limit:
                warnings.warn(
                    f'cut_method="xfem": the largest vbd.cut_force ({cut_force_max:g} N) is above '
                    f"{_XFEM_BLADE_RECORD_LIMIT_FACTOR:g} times the largest force of one contact record of the "
                    f"blade edge with the penalty law ({record_limit:g} N = contact stiffness {stiffness:g} N/m "
                    f"times collision radius {(particle_radius + margin) * 1e3:g} mm). At the cut front few "
                    "records carry the blade, so the cut can stop. Use rigid_soft_contact_use_log_barrier=True, "
                    "a lower cut force or a stiffer blade contact.",
                    NewtonWarning,
                    stacklevel=4,
                )
        self.cut_tie_stiffness = float(tie_stiffness)
        self.cut_tie_dual = bool(tie_dual)
        self.cut_xfem_min_fraction = float(min_fraction)
        self.cut_wound_mass_fraction = float(wound_mass_fraction)
        self._xfem_level_range = float(level_range)
        # The level clamp: smallest distance of a stored level value from zero. It bounds the material
        # share of a fragment from below. It is the distance at which an edge along the blade plane
        # normal, with its other end as far away or farther, still holds two tie targets one start gap
        # apart inside [t_min, 1 - t_min]. The start gap itself comes from the raw signed distances
        # (set_cut_edge_parameters), not from this value.
        self._xfem_level_min = 0.5 * self._cut_start_gap / (1.0 - 2.0 * float(min_fraction))
        self._xfem_spare_particle_count = int(capacity["color_count"]) * int(capacity["spare_count"])
        self._xfem_wound_tri_count = wound_tri_count
        self._xfem_wound_edge_count = wound_edge_count

        # A ghost has no contact with any shape: every shape carries the ghost bit of the contact gate.
        shape_gate = model.shape_contact_gate.numpy()
        shape_gate |= np.int32(CUT_GHOST_GATE_BIT)
        model.shape_contact_gate.assign(shape_gate)

        mesh_edge, tet_mesh_edge, tri_mesh_edge, edge_row_mesh_edge = build_cut_mesh_edges(
            tet_indices, first_tet_count, tri_source, edge_source
        )
        if mesh_edge.shape[0] == 0:
            mesh_edge = np.zeros((1, 2), dtype=np.int32)
        mesh_edge_count = mesh_edge.shape[0]
        self._xfem_mesh_edge = wp.array(mesh_edge, dtype=wp.int32, device=device)  # [E, 2] smaller vertex first
        self._xfem_mesh_edge_count = mesh_edge_count

        tet_side = np.full(tet_count, -1, dtype=np.int32)
        tet_side[:first_tet_count] = 0
        tet_first = np.full(tet_count, -1, dtype=np.int32)
        tet_first[:first_tet_count] = np.arange(first_tet_count)
        tet_share = np.zeros(tet_count, dtype=np.float32)
        tet_share[:first_tet_count] = 1.0

        tri_kind = np.full(model.tri_count, XFEM_ROW_PARKED, dtype=np.int32)
        tri_kind[:tri_spare_start] = np.where(tri_source[:tri_spare_start, 0] >= 0, XFEM_ROW_SKIN, XFEM_ROW_FIXED)
        tri_sub = np.zeros(model.tri_count, dtype=np.int32)
        tri_slot = np.full(model.tri_count, -1, dtype=np.int32)
        wound_rows = np.arange(wound_tri_count)
        tri_kind[tri_spare_start + wound_rows] = XFEM_ROW_WOUND
        tri_sub[tri_spare_start + wound_rows] = wound_rows % XFEM_WOUND_TRIS_PER_TET
        tri_slot[tri_spare_start + wound_rows] = wound_rows // XFEM_WOUND_TRIS_PER_TET

        edge_kind = np.full(model.edge_count, XFEM_ROW_PARKED, dtype=np.int32)
        edge_kind[:edge_spare_start] = np.where(edge_source[:edge_spare_start, 0] >= 0, XFEM_ROW_SKIN, XFEM_ROW_FIXED)
        edge_sub = np.zeros(model.edge_count, dtype=np.int32)
        edge_slot = np.full(model.edge_count, -1, dtype=np.int32)
        wound_rows = np.arange(wound_edge_count)
        edge_kind[edge_spare_start + wound_rows] = XFEM_EDGE_WOUND
        edge_sub[edge_spare_start + wound_rows] = wound_rows % XFEM_WOUND_EDGES_PER_TET
        edge_slot[edge_spare_start + wound_rows] = wound_rows // XFEM_WOUND_EDGES_PER_TET

        def int_array(values):
            return wp.array(np.ascontiguousarray(values, dtype=np.int32), dtype=wp.int32, device=device)

        def full(shape, value, dtype=wp.int32):
            return wp.full(shape, value, dtype=dtype, device=device)

        # Run-time state of the XFEM-type cut, by name. Shapes: N particles, E mesh edges, T lines of the
        # tet arrays (tissue tets, spare tets, idle tet), F triangle rows, R edge rows, S cut slots,
        # C particle colors.
        color_count = int(capacity["color_count"])
        self._xfem = {
            "particle_level": full(particle_count, XFEM_LEVEL_UNSET, float),  # [N] level value, clamped [m]
            "particle_level_raw": full(particle_count, XFEM_LEVEL_UNSET, float),  # [N] signed distance itself [m]
            "particle_level_set": full(particle_count, 0),  # [N] 0 none, 1 for the stroke, 2 stays
            "particle_stroke": full(particle_count, -1),  # [N] stroke of a level value or of a wound particle
            "stroke_number": full(1, 0),  # number of the current stroke
            "stroke_open": full(1, 0),  # 1 while the blade edge has a contact record
            "particle_ghost": full(particle_count, 0),  # [N] 1 for a virtual corner
            "particle_dirty": full(particle_count, 0),  # [N] corner of a tet cut in this advance
            "edge_t": full(mesh_edge_count, 0.0, float),  # [E] edge parameter of a crossed edge
            "edge_half_gap_t": full(mesh_edge_count, 0.0, float),  # [E] step of it from the crossing to a tie target
            "edge_gap_limited": full(mesh_edge_count, 0),  # [E] 1 when the edge cannot hold a full start gap
            "edge_pending": full(mesh_edge_count, 0),  # [E] crossed edge of a tet that this advance wants to cut
            "tet_pending": full(tet_count, 0),  # [T] tet that this advance wants to cut
            "demand": full(color_count + 3, 0),  # [C + 3] needs of the advance: pools, triangle rows, edge rows, tets
            "wound_base": full(color_count, 0),  # [C] first spare of each pool for the wound particles of the advance
            "wound_next": full(color_count, 0),  # [C] wound particles of the advance taken from each pool
            "commit": full(1, 0),  # 1 when the advance of this cut update fits the capacity
            "edge_wound": full((mesh_edge_count, 2), -1),  # [E, 2] wound particle of side A, of side B
            "edge_owner": full((mesh_edge_count, 2), XFEM_NO_ROW),  # [E, 2] smallest fragment row per side
            "edge_owner_local": full((mesh_edge_count, 2), 0),  # [E, 2] local edge in the owner row
            "edge_front": full(mesh_edge_count, 0),  # [E] 1 when a tet that is not cut holds the edge
            "edge_cut": full(mesh_edge_count, 0),  # [E] 1 crossed edge of a cut tet, 2 without wound particles
            "tet_side": int_array(tet_side),  # [T] 0 not cut, 1 fragment A, 2 fragment B, -1 not in use
            "tet_twin": full(tet_count, -1),  # [T] the other fragment row of the same tet
            "tet_first": int_array(tet_first),  # [T] the tissue tet of a fragment row
            "tet_slot": full(tet_count, -1),  # [T] cut slot of a cut tet
            "tet_stamp": full(tet_count, -1),  # [T] cut update number of the cut
            "tet_share": wp.array(tet_share, dtype=float, device=device),  # [T] material share of the fragment row
            "tet_vertex": wp.clone(model.tet_indices.reshape((tet_count, 4))),  # [T, 4] vertices before any cut
            "tet_mesh_edge": int_array(tet_mesh_edge),  # [T, 6] mesh edge of each local tet edge
            "slot_tet": full(spare_tet_count, -1),  # [S] tissue tet of a cut slot
            "slot_taken": full(1, 0),  # cut slots taken
            "wound_edge": full(particle_count, -1),  # [N] mesh edge of a wound particle
            "wound_side": full(particle_count, -1),  # [N] side of a wound particle
            "wound_new": full(particle_count, 0),  # [N] 1 until the wound particle has its mass
            "tie_s": full(particle_count, 0.0, float),  # [N] place of the tie target on the edge
            "tie_dual": wp.zeros(particle_count, dtype=wp.vec3, device=device),  # [N] tie multiplier [N]
            "tie_residual": full(particle_count, 0.0, float),  # [N] distance from the tie target [m]
            "tie_count": full(particle_count, 0),  # [N] new ties of a corner in this advance
            "mass_taken": full(particle_count, 0.0, float),  # [N] mass that new wound particles took [kg]
            "tri_mesh_edge": int_array(tri_mesh_edge),  # [F, 3] mesh edge between row corners k and k + 1
            "tri_kind": int_array(tri_kind),  # [F] kind of the row
            "tri_sub": int_array(tri_sub),  # [F] part of a clipped skin triangle, or of a wound polygon
            "tri_slot": int_array(tri_slot),  # [F] cut slot of a wound row
            "tri_extra": full(model.tri_count, -1),  # [F] first extra row of a clipped skin triangle
            "tri_extra_taken": full(1, 0),  # extra triangle rows taken
            "edge_row_mesh_edge": int_array(edge_row_mesh_edge),  # [R] mesh edge of a skin edge row
            "edge_kind": int_array(edge_kind),  # [R] kind of the row
            "edge_sub": int_array(edge_sub),  # [R] part of the wound segments of a cut slot
            "edge_slot": int_array(edge_slot),  # [R] cut slot of a wound row
            "edge_extra": full(model.edge_count, -1),  # [R] extra row of a cut skin edge
            "edge_extra_taken": full(1, 0),  # extra edge rows taken
        }
        self._xfem_twin_group = wp.full(max(record_count, 1), -1, dtype=wp.int32, device=device)
        # Arrays that both tie kernels read to find the two corners and the place of each tie.
        x = self._xfem
        self._xfem_tie_inputs = [
            x["edge_owner"],
            x["edge_owner_local"],
            model.tet_indices.reshape((tet_count, 4)),
            x["tet_vertex"],
            x["particle_level"],
            x["wound_edge"],
            x["wound_side"],
            x["tie_s"],
        ]
        self._xfem_saved = {name: wp.clone(array) for name, array in self._xfem.items()}
        self._cut_saved["tet_poses"] = wp.clone(model.tet_poses)
        self._cut_saved["tet_materials"] = wp.clone(model.tet_materials)
        # A cut writes vbd.cut_force of the spares it takes. The values of the tissue are the scene's.
        self._cut_saved["cut_force"] = wp.clone(model.vbd.cut_force)

    def _reset_cut_system(self) -> None:
        """Undo every cut: restore the arrays saved at construction and clear the cut state."""
        model = self.model
        saved = self._cut_saved
        wp.copy(model.tet_indices, saved["tet_indices"])
        wp.copy(self.particle_adjacency.v_adj_tets, saved["v_adj_tets"])
        wp.copy(model.particle_mass, saved["particle_mass"])
        wp.copy(model.particle_inv_mass, saved["particle_inv_mass"])
        wp.copy(model.particle_flags, saved["particle_flags"])
        wp.copy(model.particle_q, saved["particle_q"])
        wp.copy(model.tri_indices, saved["tri_indices"])
        wp.copy(model.edge_indices, saved["edge_indices"])
        wp.copy(self._tet_neighbors, saved["tet_neighbors"])
        wp.copy(self._cut_tri_source, saved["tri_source"])
        wp.copy(self._cut_edge_source, saved["edge_source"])
        wp.copy(self._cut_tet_face_row, saved["tet_face_row"])
        wp.copy(self._cut_tet_edge_row, saved["tet_edge_row"])
        self._cut_face_stamp.fill_(-1)
        for array in (
            self._cut_substep,
            self._cut_due,
            self._cut_update_number,
            self._cut_blade_force,
            self._cut_contact_count,
            self._cut_advance,
            self._cut_particle_pressed,
            self._cut_particle_wound,
            self._cut_tet_side,
            self._cut_face_side,
            self._cut_particle_dirty,
            self._cut_pool_taken,
            self._cut_overflow,
            self._cut_unsafe,
            self._cut_particle_new,
            self._cut_tri_taken,
            self._cut_edge_taken,
        ):
            array.zero_()
        self._cut_blade_record_key.zero_()
        gate_bits = CUT_GATE_BIT
        if self._cut_xfem:
            wp.copy(model.tet_poses, saved["tet_poses"])
            wp.copy(model.tet_materials, saved["tet_materials"])
            # Only the cut force of the spare and parking particles: a scene can change the cut force of
            # the tissue after construction, and a reset keeps that.
            first_parked = int(model.cut_capacity["particle_start"])
            wp.copy(
                model.vbd.cut_force,
                saved["cut_force"],
                dest_offset=first_parked,
                src_offset=first_parked,
                count=int(model.cut_capacity["particle_end"]) - first_parked,
            )
            for name, array in self._xfem.items():
                wp.copy(array, self._xfem_saved[name])
            gate_bits = CUT_GATE_BIT | CUT_GHOST_GATE_BIT
        wp.launch(
            kernel=clear_cut_gate_bit,
            dim=model.particle_count,
            inputs=[gate_bits],
            outputs=[model.particle_contact_gate],
            device=self.device,
        )

    def _update_cut_blades(self, state: State, contacts: Contacts | None) -> None:
        """Last call of a substep: blade force, cut faces, vertex split, start gap and surface rows.

        Kernels only, with fixed launch sizes and no readback, so a captured graph can replay it.
        """
        model = self.model
        device = self.device
        tet_count = model.tet_count
        tets = model.tet_indices.reshape((tet_count, 4))
        capacity = model.cut_capacity
        vbd_attrs = model.vbd
        due = self._cut_due
        blade = [model.cut_blade_body, model.cut_blade_edge_shape, model.cut_blade_direction]
        wp.launch(
            kernel=begin_cut_update,
            dim=1,
            inputs=[
                self.cut_update_interval,
                self._cut_substep,
                due,
                self._cut_update_number,
                self._cut_blade_force,
                self._cut_blade_record_key,
                self._cut_contact_count,
            ],
            device=device,
        )
        if contacts is not None and contacts.soft_contact_max > 0:
            wp.launch(
                kernel=accumulate_cut_blade_force,
                dim=contacts.soft_contact_max,
                inputs=[
                    due,
                    self._cut_shape_blade,
                    model.cut_blade_body,
                    model.cut_blade_direction,
                    _make_soft_contact_filter_data(model, contacts, state.particle_q, state.body_q),
                    contacts.soft_contact_count,
                    contacts.soft_contact_max,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_barycentric,
                    model.shape_body,
                    model.shape_margin,
                    state.body_q,
                    state.particle_q,
                    model.particle_radius,
                    vbd_attrs.cut_force,
                    self.body_particle_contact_penalty_k,
                    self.rigid_soft_contact_use_log_barrier,
                ],
                outputs=[self._cut_blade_force, self._cut_blade_record_key, self._cut_contact_count],
                device=device,
            )
        wp.launch(
            kernel=decide_cut_advance,
            dim=self._cut_blade_count,
            inputs=[
                due,
                1 if self.cut_armed else 0,
                contacts.soft_contact_indices if contacts is not None else self._cut_empty_indices,
                contacts.soft_contact_barycentric if contacts is not None else self._cut_empty_vec3,
                vbd_attrs.cut_force,
                self._cut_blade_force,
                self._cut_blade_record_key,
            ],
            outputs=[self._cut_advance],
            device=device,
        )
        if self._cut_xfem:
            self._update_cut_xfem(state)
            return
        wp.launch(
            kernel=mark_pressed_vertices,
            dim=model.particle_count,
            inputs=[
                due,
                1 if self.cut_press_wound else 0,
                *blade,
                self._cut_contact_range,
                self._cut_contact_count,
                state.body_q,
                model.shape_transform,
                model.shape_scale,
                state.particle_q,
                model.particle_flags,
                vbd_attrs.cut_force,
                self._cut_particle_wound,
            ],
            outputs=[self._cut_particle_pressed],
            device=device,
        )
        wp.launch(
            kernel=mark_cut_faces,
            dim=4 * tet_count,
            inputs=[
                due,
                1 if self.cut_keep_side_labels else 0,
                *blade,
                self._cut_contact_count,
                self._cut_advance,
                self._cut_update_number,
                state.body_q,
                model.shape_transform,
                model.shape_scale,
                state.particle_q,
                vbd_attrs.cut_force,
                tets,
                self._cut_particle_pressed,
            ],
            outputs=[
                self._tet_neighbors,
                self._cut_tet_side,
                self._cut_face_side,
                self._cut_face_stamp,
                self._cut_particle_dirty,
                self._cut_particle_wound,
            ],
            device=device,
        )
        wp.launch(
            kernel=split_cut_vertices,
            dim=model.particle_count,
            inputs=[
                due,
                self._cut_idle_tet,
                int(capacity["color_count"]),
                int(capacity["spare_count"]),
                1 if self.cut_mass_rule == "full_copy" else 0,
                CUT_GATE_BIT,
                model.cut_spare_start,
                model.particle_colors,
                self._cut_particle_attached,
                self._tet_neighbors,
                self._cut_face_side,
                model.tet_poses,
                self.particle_adjacency,
                self._cut_tet_record_group,
            ],
            outputs=[
                tets,
                state.particle_q,
                state.particle_qd,
                model.particle_q,
                model.particle_mass,
                model.particle_inv_mass,
                model.particle_flags,
                vbd_attrs.cut_force,
                vbd_attrs.suture_grip,
                vbd_attrs.suture_mu,
                vbd_attrs.suture_puncture_force,
                vbd_attrs.suture_puncture_cos,
                vbd_attrs.suture_stiffness,
                model.particle_contact_gate,
                self._cut_particle_wound,
                self._cut_pool_taken,
                self._cut_overflow,
                self._cut_particle_dirty,
                self._cut_particle_new,
            ],
            device=device,
        )
        wp.launch(
            kernel=compute_start_gap_offset,
            dim=model.particle_count,
            inputs=[
                due,
                self._cut_start_gap,
                1 if self.cut_gap_direction == "normal_sum" else 0,
                self._cut_idle_tet,
                self.particle_adjacency,
                tets,
                self._tet_neighbors,
                state.particle_q,
                self._cut_particle_new,
            ],
            outputs=[self._cut_offset, self._cut_unsafe],
            device=device,
        )
        wp.launch(
            kernel=apply_start_gap_offset,
            dim=model.particle_count,
            inputs=[due, self._cut_particle_new, self._cut_offset],
            outputs=[state.particle_q],
            device=device,
        )
        wp.launch(
            kernel=take_surface_rows,
            dim=4 * tet_count,
            inputs=[
                due,
                1 if self.cut_surface_edges else 0,
                self._cut_max_tets_per_vertex,
                int(capacity["color_count"]),
                int(capacity["tri_spare_start"]),
                int(capacity["tri_spare_count"]),
                int(capacity["edge_spare_start"]),
                int(capacity["edge_spare_count"]),
                self._cut_update_number,
                tets,
                self._tet_neighbors,
                self._cut_face_stamp,
            ],
            outputs=[
                self._cut_tri_taken,
                self._cut_edge_taken,
                self._cut_overflow,
                self._cut_tri_source,
                self._cut_edge_source,
                self._cut_tet_face_row,
                self._cut_tet_edge_row,
            ],
            device=device,
        )
        if model.tri_count > 0:
            wp.launch(
                kernel=refresh_triangle_rows,
                dim=model.tri_count,
                inputs=[
                    due,
                    int(capacity["tri_spare_start"]),
                    model.cut_row_park_particle,
                    self._cut_tri_source,
                    tets,
                    self._tet_neighbors,
                    self._cut_tet_face_row,
                ],
                outputs=[model.tri_indices],
                device=device,
            )
        if model.edge_count > 0:
            wp.launch(
                kernel=refresh_edge_rows,
                dim=model.edge_count,
                inputs=[due, self._cut_edge_source, tets],
                outputs=[model.edge_indices],
                device=device,
            )

    def _update_cut_xfem(self, state: State) -> None:
        """The cut update of the XFEM-type cut, after the blade force and the advance test.

        In order: stroke number, level values, edge parameters, the tets that the advance wants to
        cut and what they need, the capacity check, fragments, virtual corners, wound particles (and
        their front rows), front edges, gate and mass of the wound particles, surface rows. From the
        fragments on, the kernels read ``commit`` in the place of the advance flag: it is 1 only when
        the capacity holds the whole advance. Kernels only, with fixed launch sizes and no readback,
        so a captured graph can replay it.
        """
        model = self.model
        device = self.device
        capacity = model.cut_capacity
        vbd_attrs = model.vbd
        x = self._xfem
        tet_count = model.tet_count
        tets = model.tet_indices.reshape((tet_count, 4))
        first_tet_count = self._cut_first_tet_count
        color_count = int(capacity["color_count"])
        spare_count = int(capacity["spare_count"])
        first_spare = int(capacity["particle_start"])
        tri_spare_start = int(capacity["tri_spare_start"])
        tri_spare_count = int(capacity["tri_spare_count"])
        edge_spare_start = int(capacity["edge_spare_start"])
        edge_spare_count = int(capacity["edge_spare_count"])
        due = self._cut_due
        advance = self._cut_advance
        blade = [model.cut_blade_body, model.cut_blade_edge_shape, model.cut_blade_direction]
        blade_pose = [state.body_q, model.shape_transform, model.shape_scale]
        suture_attrs = [
            vbd_attrs.suture_grip,
            vbd_attrs.suture_mu,
            vbd_attrs.suture_puncture_force,
            vbd_attrs.suture_puncture_cos,
            vbd_attrs.suture_stiffness,
        ]
        front_rows = 1 if self.cut_xfem_front_rows else 0
        extra_tri_start = tri_spare_start + self._xfem_wound_tri_count
        extra_tri_count = tri_spare_count - self._xfem_wound_tri_count
        extra_edge_start = edge_spare_start + self._xfem_wound_edge_count
        extra_edge_count = edge_spare_count - self._xfem_wound_edge_count
        commit = x["commit"]
        wp.launch(
            kernel=update_cut_stroke,
            dim=1,
            inputs=[due, self._cut_contact_count],
            outputs=[x["stroke_open"], x["stroke_number"]],
            device=device,
        )
        wp.launch(
            kernel=set_cut_level_values,
            dim=model.particle_count,
            inputs=[
                due,
                advance,
                self._cut_contact_count,
                x["stroke_number"],
                self._xfem_level_range,
                self._xfem_level_min,
                *blade,
                *blade_pose,
                state.particle_q,
                vbd_attrs.cut_force,
                x["wound_edge"],
            ],
            outputs=[x["particle_level"], x["particle_level_raw"], x["particle_level_set"], x["particle_stroke"]],
            device=device,
        )
        wp.launch(
            kernel=set_cut_edge_parameters,
            dim=self._xfem_mesh_edge_count,
            inputs=[
                due,
                advance,
                self._cut_start_gap,
                self.cut_xfem_min_fraction,
                self._xfem_mesh_edge,
                x["particle_level"],
                x["particle_level_raw"],
                x["particle_level_set"],
                x["edge_wound"],
            ],
            outputs=[x["edge_t"], x["edge_half_gap_t"], x["edge_gap_limited"], x["edge_pending"]],
            device=device,
        )
        # What the advance wants to cut and what that needs: spare tets, spare particles for the wound
        # particles, extra surface rows. The advance goes on only when all of it fits.
        wp.launch(
            kernel=mark_crossed_tets,
            dim=first_tet_count,
            inputs=[
                due,
                advance,
                color_count + 2,
                *blade,
                self._cut_contact_range,
                *blade_pose,
                state.particle_q,
                vbd_attrs.cut_force,
                x["particle_level"],
                x["particle_level_set"],
                x["edge_t"],
                tets,
                x["tet_vertex"],
                x["tet_mesh_edge"],
                x["tet_side"],
            ],
            outputs=[x["tet_pending"], x["edge_pending"], x["demand"]],
            device=device,
        )
        wp.launch(
            kernel=count_cut_edge_demand,
            dim=self._xfem_mesh_edge_count,
            inputs=[
                due,
                advance,
                front_rows,
                color_count + 1,
                model.particle_colors,
                self._xfem_mesh_edge,
                x["edge_pending"],
                x["edge_wound"],
                x["edge_cut"],
            ],
            outputs=[x["demand"]],
            device=device,
        )
        if model.tri_count > 0:
            wp.launch(
                kernel=count_cut_triangle_row_demand,
                dim=model.tri_count,
                inputs=[
                    due,
                    advance,
                    tri_spare_start,
                    color_count,
                    x["tet_pending"],
                    x["tet_vertex"],
                    x["particle_level"],
                    self._cut_tri_source,
                    x["tri_kind"],
                    x["tri_extra"],
                ],
                outputs=[x["demand"]],
                device=device,
            )
        if model.edge_count > 0:
            wp.launch(
                kernel=count_cut_edge_row_demand,
                dim=model.edge_count,
                inputs=[
                    due,
                    advance,
                    edge_spare_start,
                    color_count + 1,
                    x["edge_pending"],
                    x["edge_wound"],
                    x["edge_cut"],
                    x["edge_row_mesh_edge"],
                    x["edge_kind"],
                    x["edge_extra"],
                ],
                outputs=[x["demand"]],
                device=device,
            )
        wp.launch(
            kernel=check_cut_demand,
            dim=1,
            inputs=[
                due,
                advance,
                color_count,
                spare_count,
                extra_tri_count,
                extra_edge_count,
                int(capacity["tet_spare_count"]),
                x["tri_extra_taken"],
                x["edge_extra_taken"],
                x["slot_taken"],
            ],
            outputs=[
                self._cut_pool_taken,
                x["demand"],
                x["wound_base"],
                x["wound_next"],
                self._cut_overflow,
                commit,
            ],
            device=device,
        )
        wp.launch(
            kernel=cut_crossed_tets,
            dim=first_tet_count,
            inputs=[
                due,
                commit,
                self._cut_update_number,
                first_tet_count,
                int(capacity["tet_spare_count"]),
                1 if self.cut_mass_rule == "full_copy" else 0,
                self.cut_xfem_min_fraction,
                color_count + 2,
                x["particle_level"],
            ],
            outputs=[
                x["tet_pending"],
                tets,
                x["tet_vertex"],
                x["tet_mesh_edge"],
                model.tet_poses,
                model.tet_materials,
                x["tet_side"],
                x["tet_twin"],
                x["tet_first"],
                x["tet_slot"],
                x["tet_stamp"],
                x["tet_share"],
                x["slot_tet"],
                x["slot_taken"],
                self._cut_overflow,
                x["edge_cut"],
                x["edge_owner"],
                x["particle_level_set"],
                x["particle_dirty"],
            ],
            device=device,
        )
        wp.launch(
            kernel=split_cut_tet_vertices,
            dim=model.particle_count,
            inputs=[
                due,
                self._cut_update_number,
                self._cut_idle_tet,
                color_count,
                spare_count,
                1 if self.cut_mass_rule == "full_copy" else 0,
                CUT_GATE_BIT,
                CUT_GHOST_GATE_BIT,
                color_count + 3,
                model.cut_spare_start,
                model.particle_colors,
                self._cut_particle_attached,
                self._tet_neighbors,
                x["tet_side"],
                x["tet_twin"],
                x["tet_first"],
                x["tet_stamp"],
                x["tet_share"],
                x["tet_vertex"],
                model.tet_poses,
                self.particle_adjacency,
                self._cut_tet_record_group,
                self._xfem_twin_group,
            ],
            outputs=[
                tets,
                state.particle_q,
                state.particle_qd,
                model.particle_q,
                model.particle_mass,
                model.particle_inv_mass,
                model.particle_flags,
                vbd_attrs.cut_force,
                *suture_attrs,
                model.particle_contact_gate,
                x["particle_ghost"],
                x["particle_level"],
                x["particle_level_raw"],
                x["particle_level_set"],
                x["particle_stroke"],
                self._cut_pool_taken,
                self._cut_overflow,
                x["particle_dirty"],
            ],
            device=device,
        )
        wp.launch(
            kernel=make_wound_particles,
            dim=self._xfem_mesh_edge_count,
            inputs=[
                due,
                commit,
                spare_count,
                self.cut_xfem_min_fraction,
                CUT_GATE_BIT | CUT_GHOST_GATE_BIT,
                front_rows,
                extra_edge_start,
                extra_edge_count,
                color_count + 1,
                model.cut_spare_start,
                model.particle_colors,
                self._xfem_mesh_edge,
                x["edge_t"],
                x["edge_half_gap_t"],
                x["edge_owner"],
                x["tet_mesh_edge"],
                x["tet_vertex"],
                tets,
                x["particle_level"],
                x["stroke_number"],
                x["wound_base"],
            ],
            outputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_q,
                vbd_attrs.cut_force,
                *suture_attrs,
                model.particle_contact_gate,
                x["edge_cut"],
                x["edge_owner_local"],
                x["edge_wound"],
                x["edge_front"],
                x["wound_edge"],
                x["wound_side"],
                x["wound_new"],
                x["tie_s"],
                x["tie_dual"],
                x["tie_count"],
                x["particle_stroke"],
                x["wound_next"],
                x["edge_row_mesh_edge"],
                x["edge_kind"],
                x["edge_extra_taken"],
                self._cut_overflow,
            ],
            device=device,
        )
        wp.launch(
            kernel=mark_cut_front_edges,
            dim=first_tet_count,
            inputs=[due, commit, vbd_attrs.cut_force, tets, x["tet_mesh_edge"], x["tet_side"]],
            outputs=[x["edge_front"]],
            device=device,
        )
        wp.launch(
            kernel=finish_wound_particles,
            dim=self._xfem_spare_particle_count,
            inputs=[
                due,
                commit,
                first_spare,
                CUT_GATE_BIT,
                self.cut_wound_mass_fraction,
                x["edge_owner"],
                x["edge_owner_local"],
                x["edge_front"],
                tets,
                x["tet_vertex"],
                x["particle_level"],
                x["wound_edge"],
                x["wound_side"],
                x["tie_count"],
            ],
            outputs=[
                model.particle_mass,
                model.particle_inv_mass,
                model.particle_flags,
                model.particle_contact_gate,
                x["wound_new"],
                x["mass_taken"],
            ],
            device=device,
        )
        wp.launch(
            kernel=apply_wound_mass_take,
            dim=model.particle_count,
            inputs=[due, commit],
            outputs=[x["tie_count"], x["mass_taken"], model.particle_mass, model.particle_inv_mass],
            device=device,
        )
        if model.tri_count > 0:
            wp.launch(
                kernel=take_cut_skin_triangle_rows,
                dim=model.tri_count,
                inputs=[
                    due,
                    commit,
                    tri_spare_start,
                    extra_tri_start,
                    extra_tri_count,
                    color_count,
                    x["tet_side"],
                    x["tet_vertex"],
                    x["particle_level"],
                ],
                outputs=[
                    self._cut_tri_source,
                    x["tri_mesh_edge"],
                    x["tri_kind"],
                    x["tri_sub"],
                    x["tri_extra"],
                    x["tri_extra_taken"],
                    self._cut_overflow,
                ],
                device=device,
            )
            wp.launch(
                kernel=write_cut_triangle_rows,
                dim=model.tri_count,
                inputs=[
                    due,
                    commit,
                    tri_spare_start,
                    model.cut_row_park_particle,
                    tets,
                    x["tet_vertex"],
                    x["tet_mesh_edge"],
                    x["tet_side"],
                    x["tet_twin"],
                    x["slot_tet"],
                    x["particle_level"],
                    x["edge_wound"],
                    self._cut_tri_source,
                    x["tri_mesh_edge"],
                    x["tri_kind"],
                    x["tri_sub"],
                    x["tri_slot"],
                ],
                outputs=[model.tri_indices],
                device=device,
            )
        if model.edge_count > 0:
            wp.launch(
                kernel=take_cut_skin_edge_rows,
                dim=model.edge_count,
                inputs=[
                    due,
                    commit,
                    edge_spare_start,
                    extra_edge_start,
                    extra_edge_count,
                    color_count + 1,
                    x["edge_wound"],
                ],
                outputs=[
                    self._cut_edge_source,
                    x["edge_row_mesh_edge"],
                    x["edge_kind"],
                    x["edge_extra"],
                    x["edge_extra_taken"],
                    self._cut_overflow,
                ],
                device=device,
            )
            wp.launch(
                kernel=write_cut_edge_rows,
                dim=model.edge_count,
                inputs=[
                    due,
                    commit,
                    edge_spare_start,
                    tri_spare_count,
                    model.cut_row_park_particle,
                    tets,
                    x["tet_vertex"],
                    x["tet_mesh_edge"],
                    self._tet_neighbors,
                    x["tet_side"],
                    x["tet_twin"],
                    x["slot_tet"],
                    x["particle_level"],
                    x["edge_wound"],
                    x["edge_front"],
                    self._cut_edge_source,
                    x["edge_row_mesh_edge"],
                    x["edge_kind"],
                    x["edge_sub"],
                    x["edge_slot"],
                ],
                outputs=[model.edge_indices],
                device=device,
            )

    def _update_duals_cut_ties(self, state: State) -> None:
        """The multiplier update of every tie (with ``cut_tie_dual``) and its residual."""
        wp.launch(
            kernel=update_duals_cut_ties,
            dim=self._xfem_spare_particle_count,
            inputs=[
                int(self.model.cut_capacity["particle_start"]),
                self.cut_tie_stiffness,
                1 if self.cut_tie_dual else 0,
                state.particle_q,
                *self._xfem_tie_inputs,
            ],
            outputs=[self._xfem["tie_dual"], self._xfem["tie_residual"]],
            device=self.device,
        )

    def _cut_gate_keep_mask(self) -> int:
        """Bits of the particle contact gate that belong to the cut; the suture gate update keeps them."""
        if not self._cut_has_capacity:
            return 0
        if self._cut_xfem:
            return CUT_GATE_BIT | CUT_GHOST_GATE_BIT
        return CUT_GATE_BIT

    def cut_report(self) -> dict:
        """Read the cut counters on the host.

        .. experimental::

            Cutting is experimental; the keys of this report can change.

        Do not call this inside a captured graph: it reads device arrays back.

        Returns:
            An empty dict for a model without cut capacity. Otherwise a dict with:
            ``cut_faces`` (number of cut tet faces), ``copies_per_pool`` (copies made from the spare
            particles of each original particle color), ``triangle_rows_taken`` and ``edge_rows_taken``
            (spare surface rows in use), ``overflow`` (failed takes: one entry per pool, then triangle
            rows, then edge rows), ``unsafe`` (new particles whose start gap direction could not open
            every cut face at them), ``blade_force`` (blade force of the last cut update [N], per
            blade) and ``updates`` (number of cut updates so far).

            With ``cut_method="xfem"`` the dict has: ``method``, ``cut_tets``, ``virtual_corners``
            (ghost particles), ``wound_particles``, ``copies_per_pool`` (spare particles of each
            pool that are in use, virtual corners and wound particles together, counted from the
            particle arrays), ``spare_tets_taken``, ``triangle_rows_taken`` and ``edge_rows_taken``
            (four wound triangle rows and ten wound edge rows per cut tet, plus the extra rows of
            clipped skin triangles, of cut skin edges and of front edges),
            ``start_gap_limited_edges`` (cut mesh edges that are too short along the blade plane
            normal to hold ``cut_start_gap`` between their two wound particles),
            ``tie_residual_max`` (largest distance of a wound particle from its tie target at the
            last iteration [m]), ``blade_force`` and ``updates``.

            ``overflow`` has one entry per pool, then the triangle rows, the edge rows, the spare
            tets and the tet records, with the sums ``overflow_pools``,
            ``overflow_triangle_rows``, ``overflow_edge_rows``, ``overflow_spare_tets`` and
            ``overflow_tet_records``. A cut advance is all or nothing: an entry of a pool, of the
            rows or of the spare tets grows by one for every cut update whose advance was refused
            because that capacity could not hold it. The tets of a refused advance stay whole, and
            the count grows at every later cut update while the blade presses on them, so the
            entries count refused cut updates, not tets. A pool entry also grows by one for every
            virtual corner that found its pool empty; that group stays on its vertex, so the two
            sides stay joined there. ``overflow_tet_records`` counts fragment rows that found no
            free tet record.
        """
        if not self._cut_has_capacity:
            return {}
        capacity = self.model.cut_capacity
        if self._cut_xfem:
            x = self._xfem
            color_count = int(capacity["color_count"])
            spare_count = int(capacity["spare_count"])
            wound = x["wound_edge"].numpy() >= 0
            ghost = x["particle_ghost"].numpy() != 0
            residual = x["tie_residual"].numpy()
            tet_side = x["tet_side"].numpy()
            extra_tris = int(capacity["tri_spare_count"]) - self._xfem_wound_tri_count
            extra_edges = int(capacity["edge_spare_count"]) - self._xfem_wound_edge_count
            overflow = self._cut_overflow.numpy()
            # A spare in use is a wound particle or a virtual corner. The counter of a pool also counts
            # failed takes, so the particle arrays give the number.
            in_use = wound | ghost
            copies_per_pool = [
                int(in_use[start : start + spare_count].sum()) for start in self.model.cut_spare_start.numpy()
            ]
            has_wound = x["edge_wound"].numpy()[:, 0] >= 0
            return {
                "method": "xfem",
                "cut_tets": int((tet_side == 1).sum()),
                "virtual_corners": int(ghost.sum()),
                "wound_particles": int(wound.sum()),
                "copies_per_pool": copies_per_pool,
                "spare_tets_taken": int((tet_side == 2).sum()),
                "triangle_rows_taken": int((tet_side == 1).sum()) * XFEM_WOUND_TRIS_PER_TET
                + min(int(x["tri_extra_taken"].numpy()[0]), extra_tris),
                "edge_rows_taken": int((tet_side == 1).sum()) * XFEM_WOUND_EDGES_PER_TET
                + min(int(x["edge_extra_taken"].numpy()[0]), extra_edges),
                "start_gap_limited_edges": int((has_wound & (x["edge_gap_limited"].numpy() != 0)).sum()),
                "overflow": overflow.tolist(),
                "overflow_pools": int(overflow[:color_count].sum()),
                "overflow_triangle_rows": int(overflow[color_count]),
                "overflow_edge_rows": int(overflow[color_count + 1]),
                "overflow_spare_tets": int(overflow[color_count + 2]),
                "overflow_tet_records": int(overflow[color_count + 3]),
                "tie_residual_max": float(residual[wound].max()) if wound.any() else 0.0,
                "blade_force": self._cut_blade_force.numpy().tolist(),
                "updates": int(self._cut_update_number.numpy()[0]),
            }
        stamps = self._cut_face_stamp.numpy()
        return {
            "cut_faces": int((stamps >= 0).sum()) // 2,
            "copies_per_pool": np.minimum(self._cut_pool_taken.numpy(), int(capacity["spare_count"])).tolist(),
            "triangle_rows_taken": min(int(self._cut_tri_taken.numpy()[0]), int(capacity["tri_spare_count"])),
            "edge_rows_taken": min(int(self._cut_edge_taken.numpy()[0]), int(capacity["edge_spare_count"])),
            "overflow": self._cut_overflow.numpy().tolist(),
            "unsafe": int(self._cut_unsafe.numpy()[0]),
            "blade_force": self._cut_blade_force.numpy().tolist(),
            "updates": int(self._cut_update_number.numpy()[0]),
        }

    def _init_suture_system(
        self,
        model: Model,
        tunnel_update_interval: int,
        point_spacing: float | None,
        gate_slack: float,
        armed: bool,
        closest_search_window: int,
        friction_transverse_weight: float,
        smooth_tangent: bool = False,
    ) -> None:
        """Allocate the tunnel point rows and the needle state of every suture needle.

        Capacities are fixed at construction: ``T`` tunnels of ``Q`` points per needle, with ``T`` and
        ``Q`` the largest authored capacities. The flat point index is ``(needle * T + tunnel) * Q + q``.
        """
        self.suture_needle_count = int(getattr(model, "suture_needle_count", 0) or 0)
        self.suture_tunnel_update_interval = int(tunnel_update_interval)
        self.suture_armed = bool(armed)
        self.suture_closest_search_window = int(closest_search_window)
        self.suture_friction_transverse_weight = float(friction_transverse_weight)
        self.suture_smooth_tangent = bool(smooth_tangent)
        self._suture_point_spacing = 0.0
        self._suture_gate_radius = np.zeros(0, dtype=np.float32)
        if self.suture_needle_count == 0:
            return
        if self.suture_tunnel_update_interval < 1:
            raise ValueError("suture_tunnel_update_interval must be at least 1")
        if self.suture_closest_search_window < 1:
            raise ValueError("suture_closest_search_window must be at least 1")
        if model.particle_world_start is not None and model.world_count > 1:
            raise NotImplementedError("Suture needles support a single world for now")

        device = self.device
        max_tunnels = model.suture_needle_max_tunnels.numpy()
        max_points = model.suture_needle_max_points.numpy()
        n = self.suture_needle_count
        tunnels = int(max_tunnels.max())
        points = int(max_points.max())
        self._suture_tunnel_capacity = tunnels
        self._suture_point_capacity = points
        total = n * tunnels * points

        # Tunnel point rows (chapter 1.5).
        self._suture_point_tet = wp.full(total, -1, dtype=wp.int32, device=device)
        self._suture_point_bary = wp.zeros(total, dtype=wp.vec4, device=device)
        self._suture_point_active = wp.zeros(total, dtype=wp.int32, device=device)
        self._suture_point_segment = wp.full(total, -1, dtype=wp.int32, device=device)
        self._suture_point_u = wp.zeros(total, dtype=wp.float32, device=device)
        self._suture_point_tangent = wp.zeros(total, dtype=wp.vec3, device=device)
        self._suture_point_segment_start = wp.full(total, -1, dtype=wp.int32, device=device)
        self._suture_point_u_start = wp.zeros(total, dtype=wp.float32, device=device)
        self._suture_point_lambda = wp.zeros(total, dtype=wp.vec3, device=device)
        self._suture_point_lambda_t = wp.zeros(total, dtype=wp.float32, device=device)
        self._suture_point_rho = wp.zeros(total, dtype=wp.float32, device=device)
        self._suture_point_rho_t = wp.zeros(total, dtype=wp.float32, device=device)
        self._suture_point_ds = wp.zeros(total, dtype=wp.float32, device=device)
        self._suture_point_stiffness = wp.full(total, float("inf"), dtype=wp.float32, device=device)
        self._suture_point_cap_normal = wp.zeros(total, dtype=wp.float32, device=device)

        # Needle and tunnel state (chapter 2.5).
        self._suture_needle_state = wp.zeros(n, dtype=wp.int32, device=device)
        self._suture_needle_tunnel = wp.full(n, -1, dtype=wp.int32, device=device)
        self._suture_tunnel_count = wp.zeros(n, dtype=wp.int32, device=device)
        self._suture_tunnel_state = wp.zeros(n * tunnels, dtype=wp.int32, device=device)
        self._suture_tunnel_point_count = wp.zeros(n * tunnels, dtype=wp.int32, device=device)
        self._suture_needle_tip_tet = wp.full(n, -1, dtype=wp.int32, device=device)
        self._suture_needle_last_tip = wp.zeros(n, dtype=wp.vec3, device=device)
        self._suture_needle_travel = wp.zeros(n, dtype=wp.float32, device=device)
        self._suture_needle_tip_force = wp.zeros(n, dtype=wp.float32, device=device)
        self._suture_needle_tip_record = wp.full(n, -1, dtype=wp.int32, device=device)
        self._suture_needle_tip_record_key = wp.zeros(n, dtype=wp.int64, device=device)
        self._suture_substep = wp.zeros(1, dtype=wp.int32, device=device)
        self._suture_point_overflow = wp.zeros(1, dtype=wp.int32, device=device)
        self._suture_due = wp.zeros(1, dtype=wp.int32, device=device)
        self._suture_needle_entered = wp.zeros(n, dtype=wp.int32, device=device)
        shape_to_needle = np.full(model.shape_count, -1, dtype=np.int32)
        tip_shapes = model.suture_needle_tip_shape.numpy()
        for needle in range(n):
            shape_to_needle[int(tip_shapes[needle])] = needle
        self._suture_shape_to_needle = wp.array(shape_to_needle, dtype=wp.int32, device=device)
        self._suture_empty_indices = wp.zeros(1, dtype=wp.vec3i, device=device)
        self._suture_empty_tets = wp.zeros((1, 4), dtype=wp.int32, device=device)
        self._suture_empty_normals = wp.zeros(1, dtype=wp.vec3, device=device)

        tet_indices = (
            model.tet_indices.numpy().reshape(-1, 4) if model.tet_count > 0 else np.zeros((0, 4), dtype=np.int32)
        )
        # One tet-neighbor table for the suture and the cut: a cut face is a boundary for the tet walk.
        if self._tet_neighbors is None:
            self._tet_neighbors = wp.array(
                build_tet_neighbors(tet_indices).reshape(-1, 4), dtype=wp.int32, device=device
            )
        self._suture_tet_neighbors = self._tet_neighbors

        # Tunnel point spacing: half the mean tet edge length, clamped to [0.5 mm, 2 mm]. A model without
        # tets cannot open a tunnel; it keeps the lower clamp as a placeholder.
        if point_spacing is None:
            if model.tet_count > 0:
                q = model.particle_q.numpy()
                # The spare tets and the idle tet of a cut capacity are not tissue; their edges have
                # the parking spacing.
                tissue_tets = tet_indices[: self._cut_first_tet_count] if self._cut_idle_tet >= 0 else tet_indices
                edges = np.concatenate(
                    [np.linalg.norm(q[tissue_tets[:, a]] - q[tissue_tets[:, b]], axis=1) for a, b in _TET_EDGE_PAIRS]
                )
                self._suture_point_spacing = float(np.clip(0.5 * float(edges.mean()), 0.5e-3, 2.0e-3))
            else:
                self._suture_point_spacing = 0.5e-3
        else:
            if not (point_spacing > 0.0):
                raise ValueError("suture_point_spacing must be positive")
            self._suture_point_spacing = float(point_spacing)

        # Gate radius per needle: largest capsule radius and shape margin of the group, the pipeline's
        # soft contact gap, the largest particle radius, plus the slack.
        shape_gate = model.shape_contact_gate.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_margin = (
            model.shape_margin.numpy() if model.shape_margin is not None and model.shape_margin.shape[0] > 0 else None
        )
        groups = model.suture_needle_group.numpy()
        gap = (
            float(getattr(self.collision_pipeline, "soft_contact_gap", 0.0) or 0.0)
            if self.collision_pipeline is not None
            else 0.0
        )
        particle_radius = float(model.particle_radius.numpy().max()) if model.particle_count > 0 else 0.0
        radii = np.zeros(n, dtype=np.float32)
        for needle in range(n):
            in_group = (shape_gate & int(groups[needle])) != 0
            capsule = in_group & (shape_type == int(GeoType.CAPSULE))
            needle_radius = float(shape_scale[capsule, 0].max()) if capsule.any() else 0.0
            margin = float(shape_margin[in_group].max()) if shape_margin is not None and in_group.any() else 0.0
            radii[needle] = needle_radius + margin + gap + particle_radius + float(gate_slack)
        self._suture_gate_radius = radii
        self._suture_gate_radius_wp = wp.array(radii, dtype=wp.float32, device=device)

        vbd_attrs = getattr(model, "vbd", None)
        if vbd_attrs is None or not hasattr(vbd_attrs, "suture_grip"):
            raise ValueError(
                "Suture needles need the SolverVBD custom attributes; call "
                "SolverVBD.register_custom_attributes(builder) before adding particles"
            )
        self._suture_grip = vbd_attrs.suture_grip
        self._suture_mu = vbd_attrs.suture_mu
        self._suture_puncture_force = vbd_attrs.suture_puncture_force
        self._suture_puncture_cos = vbd_attrs.suture_puncture_cos
        self._suture_stiffness = vbd_attrs.suture_stiffness

        rod = SutureRodData()
        rod.segment_body = model.suture_segment_body
        rod.segment_local_a = model.suture_segment_local_a
        rod.segment_local_b = model.suture_segment_local_b
        rod.needle_segment_start = model.suture_needle_segment_start
        rod.needle_body = model.suture_needle_body
        rod.needle_tip_local = model.suture_needle_tip_local
        rod.needle_group = model.suture_needle_group
        self._suture_rod = rod

        pts = SuturePointData()
        pts.tet = self._suture_point_tet
        pts.bary = self._suture_point_bary
        pts.active = self._suture_point_active
        pts.segment = self._suture_point_segment
        pts.u = self._suture_point_u
        pts.tangent = self._suture_point_tangent
        pts.segment_start = self._suture_point_segment_start
        pts.u_start = self._suture_point_u_start
        pts.lam = self._suture_point_lambda
        pts.lam_t = self._suture_point_lambda_t
        pts.rho = self._suture_point_rho
        pts.rho_t = self._suture_point_rho_t
        pts.ds = self._suture_point_ds
        pts.stiffness = self._suture_point_stiffness
        pts.cap_normal = self._suture_point_cap_normal
        pts.tunnel_capacity = tunnels
        pts.point_capacity = points
        pts.friction_transverse_weight = self.suture_friction_transverse_weight
        self._suture_points = pts
        self._suture_point_total = total

    def _suture_tet_indices(self) -> wp.array:
        """The tet index array viewed as ``[tet_count, 4]``."""
        if self.model.tet_count == 0:
            return self._suture_empty_tets
        return self.model.tet_indices.reshape((self.model.tet_count, 4))

    def _initialize_suture_points(self, state_in: State, dt: float) -> None:
        """Once per step: multiplier retention, step-start rod place and the ALM metrics of every row."""
        model = self.model
        wp.launch(
            kernel=initialize_suture_points,
            dim=self._suture_point_total,
            inputs=[
                dt,
                self.rigid_avbd_gamma,
                self._suture_points,
                self._suture_rod,
                self._suture_tet_indices(),
                state_in.particle_q,
                self.particle_q_prev,
                model.particle_inv_mass,
                model.particle_flags,
                self.particle_adjacency,
                model.tri_indices,
                model.tri_poses,
                model.tri_areas,
                model.tri_materials,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_rest_length,
                model.edge_bending_properties,
                model.tet_poses,
                model.tet_materials,
                state_in.body_q,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
                self.body_structural_k,
            ],
            device=self.device,
        )

    def _refresh_suture_points(self, state_in: State) -> None:
        """Every iteration: the closest place of each row on the rod curve and its tangent."""
        wp.launch(
            kernel=refresh_suture_points,
            dim=self._suture_point_total,
            inputs=[
                self.suture_closest_search_window,
                1 if self.suture_smooth_tangent else 0,
                self._suture_points,
                self._suture_rod,
                self._suture_tet_indices(),
                state_in.particle_q,
                state_in.body_q,
            ],
            device=self.device,
        )

    def _update_duals_suture_points(self, state_in: State) -> None:
        """After both passes of an iteration: advance the transverse and tangential multipliers."""
        wp.launch(
            kernel=update_duals_suture_points,
            dim=self._suture_point_total,
            inputs=[
                self._suture_points,
                self._suture_rod,
                self._suture_tet_indices(),
                state_in.particle_q,
                self.particle_q_prev,
                state_in.body_q,
                self.body_q_prev,
                self._suture_grip,
                self._suture_mu,
            ],
            device=self.device,
        )

    def suture_set_points(
        self,
        needle: int,
        tunnel: int,
        tets,
        barys,
        *,
        state: State,
        ds: float | None = None,
        stiffness: float | None = None,
    ) -> None:
        """Write tunnel points of one tunnel directly and refresh their rod places (tests and tools).

        The tunnel becomes a fixed (through) tunnel with every point active, so its rows act while
        the needle state machine leaves it alone. ``ds`` is the tunnel length each point represents;
        ``None`` uses the resolved point spacing. ``stiffness`` is the row stiffness of every point
        [N/m]; ``None`` takes the smallest ``vbd.suture_stiffness`` of each point's tet.
        """
        if self.suture_needle_count == 0:
            raise ValueError("The model has no suture needle")
        tets = [int(t) for t in tets]
        count = len(tets)
        if count > self._suture_point_capacity:
            raise ValueError(f"{count} points exceed the capacity {self._suture_point_capacity}")
        base = (needle * self._suture_tunnel_capacity + tunnel) * self._suture_point_capacity
        spacing = self._suture_point_spacing if ds is None else float(ds)

        def _write(array, values, dtype):
            host = array.numpy()
            host[base : base + count] = np.asarray(values, dtype=dtype)
            array.assign(host)

        _write(self._suture_point_tet, tets, np.int32)
        _write(self._suture_point_bary, [np.asarray(b, dtype=np.float32) for b in barys], np.float32)
        _write(self._suture_point_active, [1] * count, np.int32)
        _write(self._suture_point_segment, [-1] * count, np.int32)
        _write(self._suture_point_segment_start, [-1] * count, np.int32)
        _write(self._suture_point_u, [0.0] * count, np.float32)
        _write(self._suture_point_u_start, [0.0] * count, np.float32)
        _write(self._suture_point_lambda, [np.zeros(3, dtype=np.float32)] * count, np.float32)
        _write(self._suture_point_lambda_t, [0.0] * count, np.float32)
        _write(self._suture_point_ds, [spacing] * count, np.float32)
        if stiffness is None:
            attribute = self._suture_stiffness.numpy()
            tet_vertices = self.model.tet_indices.numpy().reshape(-1, 4)
            row_stiffness = [float(attribute[tet_vertices[t]].min()) for t in tets]
        else:
            row_stiffness = [float(stiffness)] * count
        _write(self._suture_point_stiffness, row_stiffness, np.float32)
        _write(self._suture_point_cap_normal, [0.0] * count, np.float32)

        tunnel_index = needle * self._suture_tunnel_capacity + tunnel
        host = self._suture_tunnel_state.numpy()
        host[tunnel_index] = 2
        self._suture_tunnel_state.assign(host)
        host = self._suture_tunnel_point_count.numpy()
        host[tunnel_index] = count
        self._suture_tunnel_point_count.assign(host)
        host = self._suture_tunnel_count.numpy()
        host[needle] = max(int(host[needle]), tunnel + 1)
        self._suture_tunnel_count.assign(host)
        self._refresh_suture_points(state)

    def _update_suture_needles(self, state: State, contacts: Contacts | None) -> None:
        """At the end of a substep: tip force, needle state machine and gate recompute (chapter 2)."""
        model = self.model
        wp.launch(
            kernel=begin_suture_update,
            dim=1,
            inputs=[self.suture_tunnel_update_interval, self._suture_substep, self._suture_due],
            device=self.device,
        )
        if contacts is not None and contacts.soft_contact_max > 0:
            wp.launch(
                kernel=accumulate_suture_tip_force,
                dim=contacts.soft_contact_max,
                inputs=[
                    self._suture_due,
                    self._suture_shape_to_needle,
                    _make_soft_contact_filter_data(model, contacts, state.particle_q, state.body_q),
                    contacts.soft_contact_count,
                    contacts.soft_contact_max,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_barycentric,
                    model.shape_body,
                    model.shape_margin,
                    state.body_q,
                    state.particle_q,
                    model.particle_radius,
                    self.body_particle_contact_penalty_k,
                    self.rigid_soft_contact_use_log_barrier,
                ],
                outputs=[self._suture_needle_tip_force, self._suture_needle_tip_record_key],
                device=self.device,
            )
        wp.launch(
            kernel=update_suture_needle_state,
            dim=self.suture_needle_count,
            inputs=[
                self._suture_due,
                1 if self.suture_armed else 0,
                self._suture_point_spacing,
                self._suture_points,
                self._suture_rod,
                self._suture_tet_indices(),
                self._suture_tet_neighbors,
                self.particle_adjacency,
                state.particle_q,
                state.body_q,
                self._suture_puncture_force,
                self._suture_puncture_cos,
                self._suture_stiffness,
                contacts.soft_contact_indices if contacts is not None else self._suture_empty_indices,
                contacts.soft_contact_normal if contacts is not None else self._suture_empty_normals,
                model.suture_needle_max_tunnels,
                model.suture_needle_max_points,
                self._suture_gate_radius_wp,
                self._suture_needle_state,
                self._suture_needle_tunnel,
                self._suture_needle_entered,
                self._suture_tunnel_count,
                self._suture_tunnel_state,
                self._suture_tunnel_point_count,
                self._suture_needle_tip_tet,
                self._suture_needle_last_tip,
                self._suture_needle_travel,
                self._suture_needle_tip_force,
                self._suture_needle_tip_record_key,
                self._suture_point_overflow,
                self._cut_idle_tet,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=update_suture_gate,
            dim=model.particle_count,
            inputs=[
                self._suture_due,
                self.suture_needle_count,
                self._suture_points,
                self._suture_rod,
                self._suture_tet_indices(),
                state.particle_q,
                state.body_q,
                self._suture_tunnel_state,
                self._suture_tunnel_point_count,
                self._suture_needle_state,
                self._suture_needle_tunnel,
                self._suture_gate_radius_wp,
                self._cut_gate_keep_mask(),
            ],
            outputs=[model.particle_contact_gate],
            device=self.device,
        )

    def suture_needle_state(self) -> np.ndarray:
        """Per-needle state: 0 when no tunnel is growing (the tip may puncture), 1 while a tunnel grows."""
        return self._suture_needle_state.numpy()

    def suture_tunnel_points(self, needle: int, tunnel: int) -> tuple[np.ndarray, np.ndarray]:
        """Tets and barycentric weights of the points of one tunnel, in creation order."""
        slot = needle * self._suture_tunnel_capacity + tunnel
        count = int(self._suture_tunnel_point_count.numpy()[slot])
        base = slot * self._suture_point_capacity
        tets = self._suture_point_tet.numpy()[base : base + count]
        barys = self._suture_point_bary.numpy()[base : base + count]
        return tets, barys

    def suture_point_overflow(self) -> bool:
        """Whether a tunnel reached its point capacity since construction."""
        return bool(self._suture_point_overflow.numpy()[0])

    def _init_rigid_dat_body_bounding_radius(self, model: Model):
        """Compute a conservative surface radius about each rigid body's COM.

        The radius makes ``|translation| + |rotation angle| * radius`` bound the
        displacement of every represented surface point between collision detections.
        A body with variants gets a radius per variant, about that variant's COM;
        :meth:`notify_model_changed` selects it on the device.
        """
        from ...geometry import GeoType  # noqa: PLC0415

        def shape_local_reach(
            geo: int, scale: wp.vec3, source: Any, collision_radius: float, com_shape: wp.vec3
        ) -> float:
            """Return a conservative upper bound on the shape's distance about the body COM."""
            if geo == GeoType.SPHERE:
                return wp.length(com_shape) + wp.abs(scale[0])
            if geo == GeoType.CAPSULE:
                h = scale[1]
                p0 = wp.vec3(0.0, 0.0, -h)
                p1 = wp.vec3(0.0, 0.0, h)
                return wp.max(wp.length(p0 - com_shape), wp.length(p1 - com_shape)) + wp.abs(scale[0])
            if geo == GeoType.BOX:
                reach = 0.0
                for sx in (-1.0, 1.0):
                    for sy in (-1.0, 1.0):
                        for sz in (-1.0, 1.0):
                            corner = wp.vec3(sx * scale[0], sy * scale[1], sz * scale[2])
                            reach = wp.max(reach, wp.length(corner - com_shape))
                return reach
            if geo == GeoType.ELLIPSOID:
                # Conservative bounding sphere (over-approximates; only tightens truncation).
                radius = wp.max(wp.abs(scale[0]), wp.max(wp.abs(scale[1]), wp.abs(scale[2])))
                return wp.length(com_shape) + radius
            if geo in (GeoType.CYLINDER, GeoType.CONE):
                radius, h = scale[0], scale[1]
                # A sphere about the shape origin avoids underestimating the continuous rim.
                return wp.length(com_shape) + wp.length(wp.vec2(radius, h))
            if geo in (GeoType.MESH, GeoType.CONVEX_MESH):
                verts = getattr(source, "vertices", None)
                if verts is not None and len(verts) > 0:
                    points_shape = np.asarray(verts, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
                    return float(
                        np.max(np.linalg.norm(points_shape - np.asarray(com_shape, dtype=np.float64)[None, :], axis=1))
                    )
            # Preserve a conservative fallback for geometry without explicit support points.
            return wp.length(com_shape) + collision_radius

        def max_reach(shapes: list[tuple], com: np.ndarray) -> float:
            """Largest reach about ``com`` of ``(type, scale, transform, source, collision radius)`` shapes."""
            result = 0.0
            for geo, scale, transform, source, collision_radius in shapes:
                X_bs = wp.transform(*transform)
                # com_shape is the body's Center of Mass in shape frame
                com_shape = wp.transform_point(wp.transform_inverse(X_bs), wp.vec3(*com))
                result = max(result, shape_local_reach(geo, wp.vec3(*scale), source, collision_radius, com_shape))
            return result

        # Collision flags are mutable, so cache every attached shape that could later be
        # enabled for rigid-soft queries. Variant rows are covered per variant below.
        shape_body = model.shape_body.numpy()
        variant_row = (
            model._shape_variant_row.numpy() if model._has_variant_rows else np.zeros(model.shape_count, dtype=bool)
        )
        body_shapes: list[list[tuple]] = [[] for _ in range(model.body_count)]
        shapes = list(
            zip(
                model.shape_type.numpy(),
                model.shape_scale.numpy(),
                model.shape_transform.numpy(),
                model.shape_source,
                model.shape_collision_radius.numpy(),
                strict=True,
            )
        )
        for shape_index, body_index in enumerate(shape_body.tolist()):
            if body_index >= 0 and not variant_row[shape_index]:
                body_shapes[body_index].append(shapes[shape_index])
        body_com = model.body_com.numpy()
        bounding_radius = np.array(
            [max_reach(body_shapes[body], body_com[body]) for body in range(model.body_count)], dtype=np.float32
        )
        self._rigid_dat_body_bounding_radius = wp.array(bounding_radius, dtype=float, device=self.device)

        # Per body with variants, the radius of each variant about that variant's COM. Bodies
        # replicated from one declaration share the variant shapes' part.
        self._rigid_dat_variant_radius_start = None
        self._rigid_dat_variant_radius = None
        if not model._has_body_variants:
            return
        geometry = model._variant_shape_geometry
        variant_shapes = list(
            zip(
                geometry["shape_type"].numpy(),
                geometry["shape_scale"].numpy(),
                geometry["shape_transform"].numpy(),
                model._variant_shape_source,
                geometry["shape_collision_radius"].numpy(),
                strict=True,
            )
        )
        row_shape = model._variant_row_shape.numpy()
        row_shape_start = model._variant_row_shape_start.numpy()
        variant_first = model._body_variant_first.numpy()
        variant_count = model._body_variant_count.numpy()
        inertia_start = model._body_variant_inertia_start.numpy()
        variant_com = model._variant_com.numpy()
        radius_start = np.full(model.body_count, -1, dtype=np.int32)
        radii: list[float] = []
        variant_reach: dict[tuple[int, int], float] = {}
        for body in np.flatnonzero(model._body_variant_set >= 0).tolist():
            radius_start[body] = len(radii)
            for variant in range(variant_count[body]):
                global_variant = variant_first[body] + variant
                k = inertia_start[body] + variant
                if (global_variant, k) not in variant_reach:
                    held = row_shape[row_shape_start[global_variant] : row_shape_start[global_variant + 1]]
                    variant_reach[global_variant, k] = max_reach(
                        [variant_shapes[q] for q in held.tolist() if q >= 0], variant_com[k]
                    )
                radii.append(max(variant_reach[global_variant, k], max_reach(body_shapes[body], variant_com[k])))
        self._rigid_dat_variant_radius_start = wp.array(radius_start, dtype=wp.int32, device=self.device)
        self._rigid_dat_variant_radius = wp.array(radii, dtype=float, device=self.device)
        self._select_rigid_dat_variant_radius()

    def _select_rigid_dat_variant_radius(self) -> None:
        """Set the DAT bounding radius of bodies with variants to that of their selected variant."""
        if getattr(self, "_rigid_dat_variant_radius", None) is None:
            return
        wp.launch(
            select_variant_bounding_radius,
            dim=self.model.body_count,
            inputs=[self.model.body_variant, self._rigid_dat_variant_radius_start, self._rigid_dat_variant_radius],
            outputs=[self._rigid_dat_body_bounding_radius],
            device=self.device,
        )

    def _reset_dat_references(self, state: State, *, reset_rigid_soft: bool, reset_particles: bool):
        """Reset selected DAT trajectory references at a collision checkpoint."""
        if reset_rigid_soft and self.model.body_count > 0:
            self.body_q_prev_collision_detection.assign(state.body_q)
        if reset_particles and self.model.particle_count > 0:
            self.pos_prev_collision_detection.assign(state.particle_q)
            self.particle_displacements.zero_()

    def _refresh_collision_sets(
        self,
        state: State,
        dt: float,
        *,
        run_rigid_collision: bool,
        run_soft_self_collision: bool,
    ):
        """Run selected collision detectors, then reset their DAT references."""
        if run_rigid_collision:
            self._run_rigid_collision(state, dt)
        if run_soft_self_collision:
            self._collision_detection_penetration_free(state, reset_reference=False)

        reset_rigid_soft = run_rigid_collision and self.rigid_soft_enable_dat
        reset_particles = reset_rigid_soft or (run_soft_self_collision and self.particle_enable_self_contact)
        if reset_rigid_soft or reset_particles:
            self._reset_dat_references(
                state,
                reset_rigid_soft=reset_rigid_soft,
                reset_particles=reset_particles,
            )

    def _rigid_penetration_free_truncation(self, state: State, contacts: Contacts | None):
        """Truncate accumulated rigid and particle updates against rigid-soft planes.

        Applied after the rigid bodies move (forward step and each body color of every
        AVBD iteration, so no color accumulates forces against a pose past its plane).
        The adaptive plane depends on both sides' accumulated motion, so both resulting
        truncation factors are applied immediately. The particle-phase joint pass
        (:meth:`_penetration_free_truncation`) re-enforces the planes after particles move.
        """
        if not self.rigid_soft_enable_dat or contacts is None or state.body_q is None:
            return

        if self.model.particle_count > 0:
            self.truncation_ts.fill_(1.0)
        self.body_truncation_ts.fill_(1.0)

        if self.model.particle_count > 0 and contacts.soft_contact_max > 0:
            wp.launch(
                kernel=apply_rigid_soft_truncation,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_rigid_indices,
                    self.model.particle_contact_gate,
                    self.model.shape_contact_gate,
                    self.model.shape_body,
                    self.model.shape_transform,
                    self.model.shape_scale,
                    self.model.shape_source_ptr,
                    self.pos_prev_collision_detection,
                    self.particle_displacements,
                    self.body_q_prev_collision_detection,
                    state.body_q,
                    self.model.body_com,
                    self.dat_conservative_bound_relaxation,
                    self.rigid_soft_dat_use_interval_arithmetic,
                ],
                outputs=[
                    self.truncation_ts,
                    self.body_truncation_ts,
                ],
                device=self.device,
            )

        if self.model.particle_count > 0:
            max_displacement = self._rigid_dat_particle_max_displacement
            if self.particle_enable_self_contact:
                max_displacement = min(
                    max_displacement,
                    self._self_contact_query_radius * self.dat_conservative_bound_relaxation * 0.5,
                )
            wp.launch(
                kernel=apply_truncation_ts,
                dim=self.model.particle_count,
                inputs=[
                    self.pos_prev_collision_detection,
                    self.particle_displacements,
                    self.truncation_ts,
                    max_displacement,
                ],
                outputs=[
                    self.particle_displacements,
                    state.particle_q,
                ],
                device=self.device,
            )

        wp.launch(
            kernel=apply_body_truncation_ts,
            dim=self.model.body_count,
            inputs=[
                self.body_q_prev_collision_detection,
                self.model.body_com,
                self.body_truncation_ts,
                self._rigid_dat_body_bounding_radius,
                self._rigid_dat_body_max_displacement,
            ],
            outputs=[
                state.body_q,
            ],
            device=self.device,
        )

    def _penetration_free_truncation(self, state: State, contacts: Contacts | None):
        """
        Truncate the accumulated particle and rigid updates in ``state`` in place.

        When rigid DAT is enabled and ``contacts`` are provided, additionally
        truncates the accumulated rigid pose updates (``state.body_q`` relative to
        ``self.body_q_prev_collision_detection``) against the rigid-soft division planes,
        in the same pass that truncates the soft side.
        """
        # Rigid-soft DAT also applies to world-attached shapes, whose state has no
        # body_q. Warp accepts that null array because static rows never index it;
        # a model with actual bodies, however, must provide their current poses.
        rigid_dat_active = self.rigid_soft_enable_dat and contacts is not None
        if rigid_dat_active and self.model.body_count > 0 and state.body_q is None:
            raise ValueError("Rigid-soft DAT requires body_q for a model with rigid bodies")

        # Max displacement between collision detections (0.5 * gamma * detection slack
        # per side); displacements beyond it degenerate to isotropic truncation.
        max_displacement = wp.inf
        if self.particle_enable_self_contact:
            max_displacement = self._self_contact_query_radius * self.dat_conservative_bound_relaxation * 0.5
        if rigid_dat_active:
            max_displacement = min(max_displacement, self._rigid_dat_particle_max_displacement)

        self.truncation_ts.fill_(1.0)

        if self.particle_enable_self_contact:
            ##  parallel by collision and atomic operation
            wp.launch(
                kernel=apply_planar_truncation_parallel_by_collision,
                inputs=[
                    self.pos_prev_collision_detection,  # pos_prev_collision_detection: wp.array[wp.vec3],
                    self.particle_displacements,  # particle_displacements: wp.array[wp.vec3],
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.trimesh_collision_info,
                    self.dat_conservative_bound_relaxation,
                ],
                outputs=[
                    self.truncation_ts,
                ],
                dim=self.particle_self_contact_evaluation_kernel_launch_size,
                device=self.device,
            )

        if rigid_dat_active:
            # Joint rigid-soft pass: both sides of every contact are constrained against
            # the same division plane within this launch.
            self.body_truncation_ts.fill_(1.0)
            wp.launch(
                kernel=apply_rigid_soft_truncation,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_rigid_indices,
                    self.model.particle_contact_gate,
                    self.model.shape_contact_gate,
                    self.model.shape_body,
                    self.model.shape_transform,
                    self.model.shape_scale,
                    self.model.shape_source_ptr,
                    self.pos_prev_collision_detection,
                    self.particle_displacements,
                    self.body_q_prev_collision_detection,
                    state.body_q,
                    self.model.body_com,
                    self.dat_conservative_bound_relaxation,
                    self.rigid_soft_dat_use_interval_arithmetic,
                ],
                outputs=[
                    self.truncation_ts,
                    self.body_truncation_ts,
                ],
                device=self.device,
            )

        wp.launch(
            kernel=apply_truncation_ts,
            dim=self.model.particle_count,
            inputs=[
                self.pos_prev_collision_detection,
                self.particle_displacements,
                self.truncation_ts,
                max_displacement,
            ],
            outputs=[
                self.particle_displacements,
                state.particle_q,
            ],
            device=self.device,
        )

        if rigid_dat_active and self.model.body_count > 0:
            wp.launch(
                kernel=apply_body_truncation_ts,
                dim=self.model.body_count,
                inputs=[
                    self.body_q_prev_collision_detection,
                    self.model.body_com,
                    self.body_truncation_ts,
                    self._rigid_dat_body_bounding_radius,
                    self._rigid_dat_body_max_displacement,
                ],
                outputs=[
                    state.body_q,
                ],
                device=self.device,
            )

    def _initialize_particles(self, state_in: State, state_out: State, contacts: Contacts | None, dt: float):
        """Initialize particle positions for the VBD iteration."""
        model = self.model

        # Early exit if no particles
        if model.particle_count == 0:
            return

        # Without DAT, use the step-start particle positions as a fixed reference for
        # all VBD iterations. particle_displacements accumulates the inertial displacement
        # and VBD corrections, so the current iterate is reference + displacement.
        # Active DAT schedules establish this reference at collision-detection checkpoints.
        if not self.rigid_soft_enable_dat and not self.particle_enable_self_contact:
            self.pos_prev_collision_detection.assign(state_in.particle_q)
            self.particle_displacements.zero_()

        wp.launch(
            kernel=forward_step,
            inputs=[
                dt,
                model.gravity,
                model.particle_world,
                self.particle_q_prev,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_inv_mass,
                state_in.particle_f,
                self.model.particle_flags,
            ],
            outputs=[
                self.inertia,
                self.particle_displacements,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        self._penetration_free_truncation(state_in, contacts)

    def _refresh_rigid_contact_state(
        self, contacts: Contacts | None, refresh: bool, *, restore_history: bool = False
    ) -> None:
        """Rebuild body-body contact lists and AVBD/ALM contact state from ``contacts``.

        The once-per-step prologue calls this from :meth:`_initialize_rigid_bodies`
        (inside its ``self._integrates_rigid_bodies`` guard); rigid ``ITERATIONS``
        mode also calls it mid-solve after re-detection, where matched warm-start
        restores the in-flight contact state onto the refreshed contact set.
        No-op when this solver does not integrate rigid bodies. Undersizing is
        folded into ``refresh`` by the caller (``body_undersized``)."""
        if not self._integrates_rigid_bodies:
            return
        model = self.model
        # Contact C0 + history restore BEFORE integration: body_q is the collide frame
        # for all bodies (dynamic and kinematic) at this point.
        if refresh:
            if contacts is None:
                self.body_body_contact_counts.zero_()
            else:
                contact_launch_dim = contacts.rigid_contact_max

                if self.body_body_contact_penalty_k.shape[0] < contact_launch_dim:
                    self._raise_if_capturing_resize(
                        "body-body contact state",
                        self.body_body_contact_penalty_k.shape[0],
                        contact_launch_dim,
                    )
                    self._init_body_body_contact_state(contact_launch_dim)

                # Build body-body contact lists
                self.body_body_contact_counts.zero_()
                self.body_body_contact_overflow_max.zero_()
                wp.launch(
                    kernel=build_body_body_contact_lists,
                    dim=contact_launch_dim,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        model.shape_body,
                        self.body_inv_mass_effective,
                        self.body_body_contact_buffer_pre_alloc,
                    ],
                    outputs=[
                        self.body_body_contact_counts,
                        self.body_body_contact_indices,
                        self.body_body_contact_overflow_max,
                    ],
                    device=self.device,
                )
                wp.launch(
                    kernel=check_contact_overflow,
                    dim=1,
                    inputs=[self.body_body_contact_overflow_max, self.body_body_contact_buffer_pre_alloc, 0],
                    device=self.device,
                )

                # Restore AVBD body-body contact state from history and pre-compute material properties
                if (self.rigid_contact_history or restore_history) and contact_launch_dim > 0:
                    if contacts.rigid_contact_match_index is None or contacts.contact_matching_mode not in (
                        "latest",
                        "sticky",
                    ):
                        raise RuntimeError(
                            "Restoring rigid contact state requires Contacts with valid "
                            "contact-matching provenance. Use "
                            'CollisionPipeline(contact_matching="latest") or '
                            'CollisionPipeline(contact_matching="sticky"). '
                            f"Got contact_matching_mode={contacts.contact_matching_mode!r}."
                        )

                    history_required = contact_launch_dim
                    if self._prev_contact_lambda is None or self._prev_contact_lambda.shape[0] < history_required:
                        history_cap = 0 if self._prev_contact_lambda is None else self._prev_contact_lambda.shape[0]
                        self._raise_if_capturing_resize("rigid contact history", history_cap, history_required)
                        self._init_rigid_contact_warmstart(history_required)

                    history = RigidContactHistory()
                    history.lambda_ = self._prev_contact_lambda
                    history.penalty_k = self._prev_contact_penalty_k
                    history.normal = self._prev_contact_normal
                    restore_compliant_tangent_warmstart = int(
                        self.rigid_compliant_alm and contacts.contact_matching_mode == "latest"
                    )

                    wp.launch(
                        kernel=init_body_body_contacts_alm,
                        dim=contact_launch_dim,
                        inputs=[
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            contacts.rigid_contact_normal,
                            model.shape_material_ke,
                            model.shape_material_kd,
                            model.shape_material_mu,
                            self.rigid_contact_hard,
                            self.rigid_compliant_alm,
                            restore_compliant_tangent_warmstart,
                            contacts.rigid_contact_match_index,
                            contacts.rigid_contact_match_generation,
                            contacts.contact_generation,
                            self._contact_history_buffer_id(contacts),
                            history,
                            self._prev_contact_frame,
                            self._contact_history_reset_pending,
                            self._contact_history_reset_mask,
                            model.shape_world,
                            model.shape_body,
                            model.body_world,
                            self.rigid_contact_k_start_value,
                        ],
                        outputs=[
                            self.body_body_contact_penalty_k,
                            self.body_body_contact_lambda,
                            self.body_body_contact_material_kd,
                            self.body_body_contact_material_mu,
                            self.body_body_contact_material_ke,
                        ],
                        device=self.device,
                    )
                elif contact_launch_dim > 0:
                    wp.launch(
                        kernel=init_body_body_contact_materials,
                        inputs=[
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_material_ke,
                            model.shape_material_kd,
                            model.shape_material_mu,
                            self.rigid_contact_k_start_value,
                        ],
                        outputs=[
                            self.body_body_contact_penalty_k,
                            self.body_body_contact_material_kd,
                            self.body_body_contact_material_mu,
                            self.body_body_contact_material_ke,
                        ],
                        dim=contact_launch_dim,
                        device=self.device,
                    )
                    self.body_body_contact_lambda.zero_()

                # A fresh refresh supersedes the prior contact rows, so consume the
                # pending reset (contact-reset state exists only with history on).
                if self.rigid_contact_history and contact_launch_dim > 0:
                    self._contact_history_reset_mask.zero_()
                    self._contact_history_reset_pending.zero_()

    def _refresh_body_particle_contact_state(self, contacts: Contacts | None, refresh: bool) -> None:
        """Rebuild body-particle contact lists and material state when needed."""
        model = self.model
        if (
            not refresh
            and model.particle_count > 0
            and contacts is not None
            and contacts.soft_contact_max > 0
            and (
                self.body_particle_contact_penalty_k.shape[0] < contacts.soft_contact_max
                or not self._particle_contact_adjacency_initialized
            )
        ):
            refresh = True

        if model.particle_count == 0 or not refresh or contacts is None:
            return

        if self._integrates_rigid_bodies:
            self.body_particle_contact_counts.zero_()
            self.body_particle_contact_overflow_max.zero_()
            wp.launch(
                kernel=build_body_particle_contact_lists,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_shape,
                    model.shape_body,
                    self.body_inv_mass_effective,
                    self.body_particle_contact_buffer_pre_alloc,
                ],
                outputs=[
                    self.body_particle_contact_counts,
                    self.body_particle_contact_indices,
                    self.body_particle_contact_overflow_max,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=check_contact_overflow,
                dim=1,
                inputs=[self.body_particle_contact_overflow_max, self.body_particle_contact_buffer_pre_alloc, 1],
                device=self.device,
            )

        soft_contact_launch_dim = contacts.soft_contact_max
        if self.body_particle_contact_penalty_k.shape[0] < soft_contact_launch_dim:
            self._raise_if_capturing_resize(
                "body-particle contact state",
                self.body_particle_contact_penalty_k.shape[0],
                soft_contact_launch_dim,
            )
            self._init_body_particle_contact_state(soft_contact_launch_dim)
        wp.launch(
            kernel=init_body_particle_contacts,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_shape,
                model.soft_contact_ke,
                model.soft_contact_kd,
                model.soft_contact_mu,
                model.shape_material_ke,
                model.shape_material_kd,
                model.shape_material_mu,
                self.rigid_contact_k_start_value,
            ],
            outputs=[
                self.body_particle_contact_penalty_k,
                self.body_particle_contact_material_kd,
                self.body_particle_contact_material_mu,
                self.body_particle_contact_material_ke,
            ],
            dim=soft_contact_launch_dim,
            device=self.device,
        )

        if model.particle_count > 0:
            if self._particle_contact_use_gather:
                self._particle_contact_head.fill_(-1)
                if contacts.soft_contact_max > 0:
                    wp.launch(
                        kernel=build_particle_body_contact_adjacency_active,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            contacts.soft_contact_indices,
                            contacts.soft_contact_count,
                            contacts.soft_contact_max,
                            self._particle_contact_head,
                            self._particle_contact_next,
                        ],
                        device=self.device,
                    )
            self._particle_contact_adjacency_initialized = True

    def _step_body_body_contact_frame(
        self,
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        dt: float,
        lambda_retention: float,
        penalty_decay: float,
    ) -> None:
        """Update rigid-contact C0, conditioning, penalty, and lambda state."""
        if not self._integrates_rigid_bodies or contacts.rigid_contact_max == 0:
            return
        model = self.model
        wp.launch(
            kernel=step_body_body_contact_C0_lambda,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                model.shape_body,
                model.body_flags,
                self.body_inv_mass_effective,
                self.body_inv_inertia_effective,
                model.body_com,
                self.body_structural_k,
                int(BodyFlags.PROXY),
                body_q,
                self.rigid_contact_hard,
                self.rigid_compliant_alm,
                lambda_retention,
                1.0 / (dt * dt),
                penalty_decay,
                self.body_body_contact_material_ke,
                self.rigid_contact_k_start_value,
            ],
            outputs=[
                self.body_body_contact_normal_rho,
                self.body_body_contact_penalty_k,
                self.body_body_contact_C0,
                self.body_body_contact_lambda,
                self.body_body_contact_tangent_rho,
            ],
            device=self.device,
        )

    def _initialize_rigid_bodies(
        self,
        state_in: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
        refresh: bool,
    ) -> None:
        """Initialize rigid body states for AVBD solver (pre-iteration phase).

        Performs forward integration and initializes contact-related AVBD state when contacts are provided.

        If contacts is None, rigid contact-related work is skipped:
        no per-body contact lists are built, and no contact state is initialized or restored.

        If control provides joint_f, per-DOF joint forces are mapped to body spatial
        wrenches and included in the forward integration (shifting the inertial target).

        The ``refresh`` input controls whether rigid contact lists and contact
        state are rebuilt. It may be promoted locally when contact state needs
        first-time allocation or resizing.

        """
        model = self.model
        rigid_capacity = contacts.rigid_contact_max if contacts is not None else 0
        # Rigid undersizing is folded into ``refresh`` here (the rigid helper
        # relies on the caller for this); particle undersizing is handled inside
        # _refresh_body_particle_contact_state, so no soft capacity is needed here.
        body_undersized = self._integrates_rigid_bodies and self.body_body_contact_penalty_k.shape[0] < rigid_capacity

        # Rigid contact history is cross-replay-persistent state: allocating it
        # during capture records a `wp.zeros` fill into the graph, which then
        # re-zeros the warm-start buffers on every replay -- silently
        # equivalent to `rigid_contact_history=False`. So this guard fires
        # unconditionally when capturing, regardless of the device's
        # allocation-during-capture support.
        if self.device.is_capturing and self._integrates_rigid_bodies and self.rigid_contact_history:
            history_capacity = 0 if self._prev_contact_lambda is None else self._prev_contact_lambda.shape[0]
            if history_capacity < rigid_capacity:
                raise RuntimeError(
                    "SolverVBD contact history must be allocated before graph capture. "
                    "Construct CollisionPipeline before SolverVBD, or run one uncaptured solver step before capture."
                )

        refresh = refresh or body_undersized

        # ---------------------------
        # Rigid-only initialization
        # ---------------------------
        if self._integrates_rigid_bodies:
            self._refresh_rigid_contact_state(contacts, refresh)

            # Per-step penalty decay, lambda retention, C0, and ALM auto-rho
            # (body_q is still collide frame here).
            if contacts is not None and contacts.rigid_contact_max > 0:
                contact_lambda_retention = _rigid_lambda_retention(
                    self.rigid_contact_alpha, self.rigid_avbd_gamma, self.rigid_compliant_alm
                )
                self._step_body_body_contact_frame(
                    contacts,
                    state_in.body_q,
                    dt,
                    contact_lambda_retention,
                    self.rigid_avbd_gamma,
                )

            # Accumulate joint_f into body wrenches (scratch buffer avoids mutating user state).
            body_f_for_integration = state_in.body_f
            if model.joint_count > 0 and control is not None and control.joint_f is not None:
                wp.copy(self._body_f_for_integration, state_in.body_f)
                body_f_for_integration = self._body_f_for_integration
                wp.launch(
                    kernel=apply_joint_forces,
                    dim=model.joint_count,
                    inputs=[
                        state_in.body_q,
                        model.body_com,
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_axis,
                        control.joint_f,
                        dt,
                    ],
                    outputs=[
                        body_f_for_integration,
                        None,  # joint_impulse: VBD does not populate body_parent_f
                    ],
                    device=self.device,
                )

            # Forward integrate rigid bodies (body_q modified in-place for dynamic bodies only).
            wp.launch(
                kernel=forward_step_rigid_bodies,
                inputs=[
                    dt,
                    model.gravity,
                    model.body_world,
                    self._rigid_pose_rebaseline_mask,
                    body_f_for_integration,
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    state_in.body_q,  # input/output
                    state_in.body_qd,  # input/output
                ],
                outputs=[
                    self.body_q_prev,  # rebaselined for flagged worlds (first step / reset)
                    self.body_inertia_q,
                ],
                dim=model.body_count,
                block_dim=self._rigid_body_block_dim,
                device=self.device,
            )

            # Truncate the forward step against the division planes before any
            # solve iterations run on the predicted poses.
            self._rigid_penetration_free_truncation(state_in, contacts)

            if self.joint_constraint_count > 0:
                # Per-step joint setup: penalty-k decay, C0 snapshot, lambda retention,
                # and automatic per-row rho refresh for compliant ALM.
                joint_lambda_retention = _rigid_lambda_retention(
                    self.rigid_joint_alpha, self.rigid_avbd_gamma, self.rigid_compliant_alm
                )
                wp.launch(
                    kernel=step_joint_C0_lambda_rho,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_world,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_axis,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        self.joint_rod_rest_kb_local,
                        self.joint_rod_rest_twist,
                        state_in.body_q,
                        self.body_q_prev,
                        model.body_q,
                        self.joint_rest_angle,
                        self._rigid_pose_rebaseline_mask,
                        self.joint_constraint_start,
                        self.joint_constraint_dim,
                        self.joint_is_hard,
                        self.rigid_compliant_alm,
                        joint_lambda_retention,
                        self.rigid_avbd_gamma,
                        self.joint_penalty_k_min,
                        self.joint_material_k,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_limit_ke,
                        1.0 / (dt * dt),
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                    ],
                    outputs=[
                        self.joint_penalty_k,
                        self.joint_C0_lin,
                        self.joint_C0_ang,
                        self.joint_lambda_lin,
                        self.joint_lambda_ang,
                        self.joint_rho,
                        self.joint_drive_limit_support,
                        self.joint_drive_lambda,
                        self.joint_limit_lambda,
                        self.joint_angle_prev,
                    ],
                    block_dim=self._rigid_joint_block_dim,
                    device=self.device,
                )

            # Compute rod bend/twist Dahl hysteresis parameters once per timestep.
            if self.enable_dahl_friction and model.joint_count > 0:
                wp.launch(
                    kernel=compute_rod_dahl_parameters,
                    inputs=[
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_world,
                        self._rigid_pose_rebaseline_mask,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        self.joint_constraint_start,
                        self.joint_penalty_k,
                        self.joint_material_k,
                        self.joint_is_hard,
                        self.rigid_compliant_alm,
                        self.joint_rod_rest_kb_local,
                        self.joint_rod_rest_twist,
                        self.body_q_prev,
                        self.joint_sigma_prev,
                        self.joint_kappa_prev,
                        self.joint_dkappa_prev,
                        self.joint_dahl_eps_max,
                        self.joint_dahl_tau,
                    ],
                    outputs=[
                        self.joint_sigma_start,
                        self.joint_C_fric,
                    ],
                    dim=model.joint_count,
                    block_dim=self._rigid_joint_block_dim,
                    device=self.device,
                )

            # The forward step and any enabled rod update have consumed the mask.
            self._rigid_pose_rebaseline_mask.zero_()

        self._refresh_body_particle_contact_state(contacts, refresh)

    def _solve_particle_iteration(self, state_in: State, state_out: State, contacts: Contacts | None, dt: float):
        """Solve one VBD iteration for particles."""
        model = self.model

        # Select rigid-body poses for particle-rigid contact evaluation
        if self.integrate_with_external_rigid_solver:
            body_q_for_particles = state_out.body_q
            body_q_prev_for_particles = state_in.body_q
            body_qd_for_particles = state_out.body_qd
        else:
            body_q_for_particles = state_in.body_q
            if model.body_count > 0:
                body_q_prev_for_particles = self.body_q_prev
            else:
                body_q_prev_for_particles = None
            body_qd_for_particles = state_in.body_qd

        # Early exit if no particles
        if model.particle_count == 0:
            return

        contact_filter = (
            _make_soft_contact_filter_data(model, contacts, state_in.particle_q, body_q_for_particles)
            if contacts is not None
            else None
        )

        # Zero out forces and hessians
        self.particle_forces.zero_()
        self.particle_hessians.zero_()

        # Iterate over color groups
        for color in range(len(self.model.particle_color_groups)):
            if model.attachment_particle_body_count > 0:
                wp.launch(
                    kernel=accumulate_body_particle_attachment_force_and_hessian,
                    dim=model.attachment_particle_body_count,
                    inputs=[
                        dt,
                        color,
                        self.particle_q_prev,
                        state_in.particle_q,
                        model.particle_colors,
                        body_q_for_particles,
                        body_q_prev_for_particles,
                        model.attachment_particle_body_body,
                        model.attachment_particle_body_particle,
                        model.attachment_particle_body_body_point,
                        model.attachment_particle_body_stiffness,
                        model.attachment_particle_body_damping,
                        model.attachment_particle_body_enabled,
                        self._attachment_particle_body_rho,
                        self._attachment_particle_body_lambda,
                    ],
                    outputs=[self.particle_forces, self.particle_hessians],
                    device=self.device,
                )

            if self.suture_needle_count > 0:
                wp.launch(
                    kernel=accumulate_suture_row_force_and_hessian,
                    dim=self._suture_point_total,
                    inputs=[
                        dt,
                        color,
                        model.particle_colors,
                        self._suture_points,
                        self._suture_rod,
                        self._suture_tet_indices(),
                        state_in.particle_q,
                        self.particle_q_prev,
                        body_q_for_particles,
                        body_q_prev_for_particles,
                        self._suture_grip,
                        self._suture_mu,
                        self.friction_epsilon,
                    ],
                    outputs=[self.particle_forces, self.particle_hessians],
                    device=self.device,
                )

            if self._cut_xfem:
                wp.launch(
                    kernel=accumulate_cut_tie_force_and_hessian,
                    dim=self._xfem_spare_particle_count,
                    inputs=[
                        color,
                        int(model.cut_capacity["particle_start"]),
                        self.cut_tie_stiffness,
                        model.particle_colors,
                        state_in.particle_q,
                        *self._xfem_tie_inputs,
                        self._xfem["tie_dual"],
                    ],
                    outputs=[self.particle_forces, self.particle_hessians],
                    device=self.device,
                )

            if contacts is not None and contacts.soft_contact_max > 0:
                contact_material_and_body_inputs = [
                    self.body_particle_contact_penalty_k,
                    self.body_particle_contact_material_kd,
                    self.body_particle_contact_material_mu,
                    model.shape_body,
                    body_q_for_particles,
                    body_q_prev_for_particles,
                    body_qd_for_particles,
                    model.body_com,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    model.shape_margin,
                    contacts.soft_contact_barycentric,
                ]
                if self._particle_contact_use_gather:
                    # Deterministic modes: one thread per colored particle, no output atomics.
                    wp.launch(
                        kernel=gather_particle_body_contact_force_and_hessian,
                        dim=self.model.particle_color_groups[color].size,
                        block_dim=_PARTICLE_CONTACT_GATHER_BLOCK_DIM,
                        inputs=[
                            dt,
                            self.model.particle_color_groups[color],
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.friction_epsilon,
                            self.rigid_soft_contact_use_log_barrier,
                            model.particle_radius,
                            contacts.soft_contact_indices,
                            self._particle_contact_head,
                            self._particle_contact_next,
                            *contact_material_and_body_inputs,
                            contact_filter,
                        ],
                        outputs=[
                            self.particle_forces,
                            self.particle_hessians,
                        ],
                        device=self.device,
                    )
                else:
                    # Default: one record per thread over the active prefix, atomics into the
                    # active color's corners. The worker grid is host-static (graph-safe).
                    wp.launch(
                        kernel=scatter_particle_body_contact_force_and_hessian,
                        dim=self._particle_contact_worker_count,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_colors,
                            self.friction_epsilon,
                            self.rigid_soft_contact_use_log_barrier,
                            model.particle_radius,
                            contacts.soft_contact_indices,
                            contacts.soft_contact_count,
                            contact_filter,
                            contacts.soft_contact_max,
                            self._particle_contact_worker_count,
                            *contact_material_and_body_inputs,
                        ],
                        outputs=[
                            self.particle_forces,
                            self.particle_hessians,
                        ],
                        device=self.device,
                    )

            if model.spring_count:
                wp.launch(
                    kernel=accumulate_spring_force_and_hessian,
                    inputs=[
                        dt,
                        color,
                        self.particle_q_prev,
                        state_in.particle_q,
                        self.model.particle_colors,
                        model.spring_count,
                        self.model.spring_indices,
                        self.model.spring_rest_length,
                        self.model.spring_stiffness,
                        self.model.spring_damping,
                    ],
                    outputs=[self.particle_forces, self.particle_hessians],
                    dim=model.spring_count,
                    device=self.device,
                )

            if self.particle_enable_self_contact:
                wp.launch(
                    kernel=accumulate_self_contact_force_and_hessian,
                    dim=self.particle_self_contact_evaluation_kernel_launch_size,
                    inputs=[
                        dt,
                        color,
                        self.particle_q_prev,
                        state_in.particle_q,
                        self.model.particle_colors,
                        self.model.tri_indices,
                        self.model.edge_indices,
                        # self-contact
                        self.trimesh_collision_info,
                        self.particle_self_contact_margin,
                        self.model.soft_contact_ke,
                        self.model.soft_contact_kd,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        self._self_contact_edge_edge_parallel_epsilon,
                    ],
                    outputs=[self.particle_forces, self.particle_hessians],
                    device=self.device,
                    max_blocks=self.model.device.sm_count,
                )
            if self.use_particle_tile_solve:
                particle_count_in_color = self.model.particle_color_groups[color].size
                per_block = self._tiled_elasticity_particles_per_block
                blocks = (particle_count_in_color + per_block - 1) // per_block
                wp.launch(
                    kernel=self._tiled_elasticity_kernel,
                    dim=blocks * per_block * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                    block_dim=per_block * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                    inputs=[
                        dt,
                        self.model.particle_color_groups[color],
                        self.particle_q_prev,
                        state_in.particle_q,
                        self.model.particle_mass,
                        self.inertia,
                        self.model.particle_flags,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.model.edge_indices,
                        self.model.edge_rest_angle,
                        self.model.edge_rest_length,
                        self.model.edge_bending_properties,
                        self.model.tet_indices,
                        self.model.tet_poses,
                        self.model.tet_materials,
                        self.particle_adjacency,
                        self.particle_forces,
                        self.particle_hessians,
                    ],
                    outputs=[
                        self.particle_displacements,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    kernel=solve_elasticity,
                    dim=self.model.particle_color_groups[color].size,
                    inputs=[
                        dt,
                        self.model.particle_color_groups[color],
                        self.particle_q_prev,
                        state_in.particle_q,
                        self.model.particle_mass,
                        self.inertia,
                        self.model.particle_flags,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.model.edge_indices,
                        self.model.edge_rest_angle,
                        self.model.edge_rest_length,
                        self.model.edge_bending_properties,
                        self.model.tet_indices,
                        self.model.tet_poses,
                        self.model.tet_materials,
                        self.particle_adjacency,
                        self.particle_forces,
                        self.particle_hessians,
                    ],
                    outputs=[
                        self.particle_displacements,
                    ],
                    device=self.device,
                )
            self._penetration_free_truncation(state_in, contacts)

        wp.copy(state_out.particle_q, state_in.particle_q)

    def _update_tendon_routing(self, state_in: State, dt: float, report_unsupported_wrap: bool) -> None:
        """Update VBD routed-tendon geometry and rolling rest transfer for this iteration."""
        model = self.model
        if model.tendon_segment_count == 0:
            return

        wp.launch(
            kernel=update_tendon_attachments,
            dim=model.tendon_segment_count,
            inputs=[
                state_in.body_q,
                model.tendon_start,
                self.tendon_guide_tendon,
                model.tendon_guide_body,
                model.tendon_guide_type,
                model.tendon_guide_flags,
                model.tendon_guide_radius,
                model.tendon_guide_orientation,
                model.tendon_guide_offset,
                model.tendon_guide_axis,
                self.tendon_segment_active,
                self.tendon_segment_active_guide_l,
                self.tendon_segment_active_guide_r,
                self.tendon_guide_active,
                self.tendon_guide_active_step,
                self.tendon_segment_attachment_l_local_step,
                self.tendon_segment_attachment_r_local_step,
                1,
            ],
            outputs=[
                self.tendon_segment_attachment_l,
                self.tendon_segment_attachment_r,
                self.tendon_segment_attachment_l_local,
                self.tendon_segment_attachment_r_local,
                self.tendon_segment_rolling_delta_l,
                self.tendon_segment_rolling_delta_r,
                self.tendon_segment_length,
            ],
            device=self.device,
        )

        self._update_tendon_cone_rows(model, state_in.body_q, report_unsupported_wrap)

        wp.launch(
            kernel=solve_tendon_material,
            dim=model.tendon_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                self.body_q_prev if model.body_count else None,
                model.body_com,
                model.tendon_start,
                model.tendon_guide_body,
                model.tendon_guide_type,
                model.tendon_guide_radius,
                model.tendon_guide_offset,
                model.tendon_guide_axis,
                self.tendon_segment_rest_length,
                self.tendon_segment_rest_length_step,
                self.tendon_segment_route_rest_length,
                self.tendon_segment_stretch,
                self.tendon_segment_damping_tension,
                self.tendon_segment_active,
                self.tendon_segment_active_guide_l,
                self.tendon_segment_active_guide_r,
                self.tendon_segment_active_compliance,
                self.tendon_segment_active_damping,
                self.tendon_guide_active,
                self.tendon_guide_active_step,
                self.tendon_guide_route_rest_length,
                self.tendon_segment_attachment_l,
                self.tendon_segment_attachment_r,
                self.tendon_segment_length,
                self.tendon_segment_attachment_l_local,
                self.tendon_segment_attachment_r_local,
                self.tendon_segment_rolling_delta_l,
                self.tendon_segment_rolling_delta_r,
                self.tendon_guide_cone_seg_l,
                self.tendon_guide_cone_seg_r,
                self.tendon_guide_cap_ratio,
                self.tendon_cone_sweep_count,
                1,
                dt,
                1,
                1,
                1,
                self.tendon_max_sweeps,
                self.tendon_settle_tol,
            ],
            device=self.device,
        )

    def _snapshot_tendon_segment_length_reference(self, body_q: wp.array[wp.transform], dt: float) -> None:
        """Preserve previous-pose segment lengths through rigid finalization."""
        model = self.model
        # Native VBD does not accumulate XPBD delta lambdas, so this existing
        # per-segment scratch can hold the damping reference until finalization.
        wp.launch(
            kernel=snapshot_tendon_segment_length_reference,
            dim=model.tendon_segment_count,
            inputs=[
                dt,
                body_q,
                self.body_q_prev if model.body_count else None,
                model.body_com,
                model.tendon_guide_body,
                model.tendon_guide_type,
                model.tendon_guide_offset,
                model.tendon_guide_axis,
                self.tendon_segment_attachment_l_local,
                self.tendon_segment_attachment_r_local,
                self.tendon_segment_active,
                self.tendon_segment_active_guide_l,
                self.tendon_segment_active_guide_r,
            ],
            outputs=[self.tendon_segment_delta_lambda],
            device=self.device,
        )

    def _update_tendon_segment_diagnostics(self, state_out: State, dt: float) -> None:
        """Update tendon geometry and damping from the accepted rigid pose."""
        model = self.model
        wp.launch(
            kernel=update_tendon_segment_diagnostics,
            dim=model.tendon_segment_count,
            inputs=[
                dt,
                state_out.body_q,
                model.tendon_guide_body,
                self.tendon_segment_attachment_l_local,
                self.tendon_segment_attachment_r_local,
                self.tendon_segment_rest_length,
                self.tendon_segment_active_compliance,
                self.tendon_segment_active_damping,
                self.tendon_segment_active,
                self.tendon_segment_active_guide_l,
                self.tendon_segment_active_guide_r,
                self.tendon_segment_delta_lambda,
            ],
            outputs=[
                self.tendon_segment_attachment_l,
                self.tendon_segment_attachment_r,
                self.tendon_segment_material_tension,
                self.tendon_segment_damping_tension,
                self.tendon_segment_lambda,
            ],
            device=self.device,
        )

    def _solve_rigid_body_iteration(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ):
        """Solve one rigid-body VBD iteration (per-iteration phase).

        Accumulates contact and joint forces/hessians, solves 6x6 rigid body systems per color,
        and updates AVBD penalty parameters (dual update).
        """
        model = self.model
        contact_filter = (
            _make_soft_contact_filter_data(
                model,
                contacts,
                state_in.particle_q,
                state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
            )
            if contacts is not None
            else None
        )
        # Body-particle soft contacts still need penalty updates when VBD skips rigid solves:
        # external rigid mode uses state_out.body_q, while static-shape contacts use _empty_body_q.
        skip_rigid_solve = not self._integrates_rigid_bodies
        if skip_rigid_solve:
            if model.particle_count > 0 and contacts is not None:
                body_q = state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q
                if body_q is None:
                    body_q = self._empty_body_q

                wp.launch(
                    kernel=update_duals_body_particle_contacts,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        contacts.soft_contact_count,
                        contact_filter,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_barycentric,
                        state_in.particle_q,
                        model.particle_radius,
                        model.shape_body,
                        model.shape_margin,
                        body_q,
                        self.body_particle_contact_material_ke,
                        self.rigid_linear_beta,
                        self.body_particle_contact_penalty_k,  # input/output
                    ],
                    device=self.device,
                )
            return

        # Zero out forces and hessians
        self._body_contact_scratch.zero_()

        body_color_groups = model.body_color_groups

        # Gauss-Seidel-style per-color updates
        for color in range(len(body_color_groups)):
            color_group = body_color_groups[color]
            solve_block_dim, body_contact_block_dim, particle_contact_block_dim = self._rigid_color_block_dims[color]

            if model.attachment_particle_body_count > 0:
                wp.launch(
                    kernel=accumulate_body_particle_attachments_per_body,
                    dim=color_group.size,
                    inputs=[
                        dt,
                        color_group,
                        state_in.particle_q,
                        self.particle_q_prev,
                        state_in.body_q,
                        self.body_q_prev,
                        model.body_com,
                        self.body_inv_mass_effective,
                        model.attachment_particle_body_particle,
                        model.attachment_particle_body_body_point,
                        model.attachment_particle_body_stiffness,
                        model.attachment_particle_body_damping,
                        model.attachment_particle_body_enabled,
                        self._attachment_particle_body_rho,
                        self._attachment_particle_body_lambda,
                        self.body_particle_attachment_offsets,
                        self.body_particle_attachment_indices,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

            if self.suture_needle_count > 0:
                wp.launch(
                    kernel=accumulate_suture_rows_per_body,
                    dim=color_group.size,
                    inputs=[
                        dt,
                        color_group,
                        self._suture_points,
                        self._suture_rod,
                        self._suture_tet_indices(),
                        state_in.particle_q,
                        self.particle_q_prev,
                        state_in.body_q,
                        self.body_q_prev,
                        model.body_com,
                        self.body_inv_mass_effective,
                        self._suture_grip,
                        self._suture_mu,
                        self.friction_epsilon,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

            # Accumulate body-particle contact forces/hessians for bodies in this color
            if model.particle_count > 0 and contacts is not None:
                wp.launch(
                    kernel=accumulate_body_particle_contacts_per_body,
                    dim=color_group.size * self._body_particle_contact_threads,
                    inputs=[
                        dt,
                        color_group,
                        self._body_particle_contact_threads,
                        state_in.particle_q,
                        self.particle_q_prev,
                        model.particle_radius,
                        self.body_q_prev,
                        state_in.body_q,
                        state_in.body_qd,
                        model.body_com,
                        self.body_inv_mass_effective,
                        model.shape_body,
                        self.friction_epsilon,
                        self.rigid_soft_contact_use_log_barrier,
                        self.body_particle_contact_penalty_k,
                        self.body_particle_contact_material_ke,
                        self.body_particle_contact_material_kd,
                        self.body_particle_contact_material_mu,
                        contacts.soft_contact_count,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_barycentric,
                        model.shape_margin,
                        self.body_particle_contact_buffer_pre_alloc,
                        self.body_particle_contact_counts,
                        self.body_particle_contact_indices,
                        contact_filter,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    block_dim=particle_contact_block_dim,
                    device=self.device,
                )

            # Accumulate body-body (rigid-rigid) contact forces and Hessians on bodies (per-body, per-color)
            if contacts is not None and model.shape_count > 0:
                wp.launch(
                    kernel=accumulate_body_body_contacts_per_body,
                    dim=color_group.size * _NUM_CONTACT_THREADS_PER_BODY,
                    inputs=[
                        dt,
                        color_group,
                        self.body_q_prev,
                        state_in.body_q,
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.friction_epsilon,
                        self.body_body_contact_penalty_k,
                        self.body_body_contact_normal_rho,
                        self.body_body_contact_material_ke,
                        self.body_body_contact_material_kd,
                        self.body_body_contact_material_mu,
                        self.body_body_contact_tangent_rho,
                        self.body_body_contact_lambda,
                        self.body_body_contact_C0,
                        self.rigid_contact_alpha,
                        self.rigid_contact_hard,
                        self.rigid_compliant_alm,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_surface_velocity,
                        contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        model.shape_body,
                        self.body_body_contact_buffer_pre_alloc,
                        self.body_body_contact_counts,
                        self.body_body_contact_indices,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    block_dim=body_contact_block_dim,
                    device=self.device,
                )

            wp.launch(
                kernel=self._solve_rigid_body_kernel,
                inputs=[
                    dt,
                    color_group,
                    state_in.body_q,
                    self.body_q_prev,
                    model.body_q,
                    model.body_mass,
                    self.body_inv_mass_effective,
                    model.body_inertia,
                    self.body_inertia_q,
                    model.body_com,
                    self.rigid_adjacency,
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    self.joint_rod_rest_kb_local,
                    self.joint_rod_rest_twist,
                    model.joint_qd_start,
                    model.joint_target_q_start,
                    self.joint_constraint_start,
                    self.joint_penalty_k,
                    self.joint_rho,
                    self.joint_material_k,
                    self.joint_penalty_kd,
                    self.joint_sigma_start,
                    self.joint_C_fric,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    control.joint_target_q,
                    control.joint_target_qd,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    model.joint_limit_kd,
                    self.joint_drive_limit_support,
                    self.joint_drive_lambda,
                    self.joint_limit_lambda,
                    self.joint_lambda_lin,
                    self.joint_lambda_ang,
                    self.joint_C0_lin,
                    self.joint_C0_ang,
                    self.joint_is_hard,
                    self.rigid_joint_alpha,
                    self.rigid_compliant_alm,
                    model.joint_dof_dim,
                    self.joint_rest_angle,
                    self.joint_angle_prev,
                    self.body_forces,
                    self.body_torques,
                    self.body_hessian_ll,
                    self.body_hessian_al,
                    self.body_hessian_aa,
                    self.tendon_adjacency,
                    model.tendon_guide_body,
                    model.tendon_guide_type,
                    model.tendon_guide_radius,
                    model.tendon_guide_mu,
                    model.tendon_guide_offset,
                    model.tendon_guide_axis,
                    self.tendon_guide_cone_seg_l,
                    self.tendon_guide_cone_seg_r,
                    self.tendon_segment_rest_length,
                    self.tendon_segment_attachment_l_local,
                    self.tendon_segment_attachment_r_local,
                    self.tendon_segment_active_compliance,
                    self.tendon_segment_active_damping,
                    self.tendon_segment_active,
                    self.tendon_segment_active_guide_l,
                    self.tendon_segment_active_guide_r,
                ],
                outputs=[
                    state_in.body_q,
                ],
                dim=color_group.size,
                block_dim=solve_block_dim,
                device=self.device,
            )
            # Truncate this color's pose updates before the next color accumulates
            # forces against them; the last color also feeds the dual updates.
            self._rigid_penetration_free_truncation(state_in, contacts)

        if self._has_joint_mimics:
            project_joint_mimics(
                model,
                state_in.body_q,
                state_in.body_qd,
                self.body_inv_mass_effective,
                self.body_inv_inertia_effective,
                self._mimic_body_deltas,
                dt,
            )

        if contacts is not None and contacts.rigid_contact_max > 0 and model.shape_count > 0:
            wp.launch(
                kernel=update_duals_body_body_contacts,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_surface_velocity,
                    contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    model.shape_body,
                    state_in.body_q,
                    self.body_q_prev,
                    dt,
                    self.body_body_contact_material_mu,
                    self.body_body_contact_C0,
                    self.rigid_contact_alpha,
                    self.rigid_contact_hard,
                    self.rigid_compliant_alm,
                    self.body_body_contact_material_ke,
                    self.body_body_contact_tangent_rho,
                    self.body_body_contact_normal_rho,
                    self.rigid_linear_beta,
                    self.body_body_contact_penalty_k,  # input/output
                    self.body_body_contact_lambda,  # input/output
                ],
                device=self.device,
            )
        if contacts is not None and model.particle_count > 0:
            wp.launch(
                kernel=update_duals_body_particle_contacts,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contact_filter,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_barycentric,
                    state_in.particle_q,
                    model.particle_radius,
                    model.shape_body,
                    model.shape_margin,
                    state_in.body_q,
                    self.body_particle_contact_material_ke,
                    self.rigid_linear_beta,
                    self.body_particle_contact_penalty_k,  # input/output
                ],
                device=self.device,
            )

        if self.joint_constraint_count > 0:
            wp.launch(
                kernel=update_duals_joint,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    self.joint_rod_rest_kb_local,
                    self.joint_rod_rest_twist,
                    model.joint_qd_start,
                    model.joint_target_q_start,
                    self.joint_constraint_start,
                    state_in.body_q,
                    self.body_q_prev,
                    model.body_q,
                    model.joint_dof_dim,
                    self.joint_C0_lin,
                    self.joint_C0_ang,
                    self.joint_is_hard,
                    self.rigid_joint_alpha,
                    self.joint_material_k,
                    self.joint_rho,
                    self.rigid_compliant_alm,
                    self.rigid_linear_beta,
                    self.rigid_angular_beta,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    control.joint_target_q,
                    control.joint_target_qd,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    model.joint_limit_kd,
                    self.joint_rest_angle,
                    self.joint_angle_prev,
                    self.joint_drive_limit_support,
                    dt,
                    self.joint_penalty_k,  # input/output
                    self.joint_lambda_lin,  # input/output
                    self.joint_lambda_ang,  # input/output
                    self.joint_drive_lambda,  # input/output
                    self.joint_limit_lambda,  # input/output
                ],
                block_dim=self._rigid_joint_block_dim,
                device=self.device,
            )

    def _export_contact_forces(
        self, state_in: State, state_out: State, contacts: Contacts, dt: float, contact_f: wp.array[wp.spatial_vector]
    ) -> None:
        """Evaluate requested contact wrenches directly into the observable array.

        Both evaluations read the final iterate together with the pose history the last iteration
        used, so they never change the solve. Must run before rigid finalization advances ``body_q_prev``.
        """
        # Clear unowned rigid rows and soft rows when no particles are present as well as padding.
        contact_f.zero_()
        self._export_body_body_contact_forces(state_in, contacts, dt, contact_f)
        if contacts.soft_contact_max > 0 and self.model.particle_count > 0:
            # Slicing creates a view, not a device allocation, including during graph capture.
            self._export_body_particle_contact_forces(
                state_in, state_out, contacts, dt, contact_f[contacts.rigid_contact_max :]
            )

    def _export_body_body_contact_forces(
        self, state_in: State, contacts: Contacts, dt: float, contact_f: wp.array[wp.spatial_vector]
    ) -> None:
        """Evaluate one wrench per body-body contact record when this solver integrates the bodies."""
        if not self._integrates_rigid_bodies:
            return

        model = self.model
        rigid_contact_max = contacts.rigid_contact_max
        if rigid_contact_max == 0:
            return

        # The rigid prologue sized the per-contact state to this buffer; the iterate poses live in
        # state_in.body_q and body_q_prev still holds the step-start history the iterations used.
        wp.launch(
            kernel=compute_body_body_contact_forces,
            dim=rigid_contact_max,
            inputs=[
                float(dt),
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_surface_velocity,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                model.shape_body,
                state_in.body_q,
                self.body_q_prev,
                model.body_com,
                self.body_body_contact_penalty_k,
                self.body_body_contact_normal_rho,
                self.body_body_contact_material_ke,
                self.body_body_contact_material_kd,
                self.body_body_contact_material_mu,
                self.body_body_contact_tangent_rho,
                self.body_body_contact_lambda,
                self.body_body_contact_C0,
                self.rigid_contact_alpha,
                self.rigid_contact_hard,
                self.rigid_compliant_alm,
                float(self.friction_epsilon),
            ],
            outputs=[contact_f],
            device=self.device,
        )

    def _export_body_particle_contact_forces(
        self, state_in: State, state_out: State, contacts: Contacts, dt: float, contact_f: wp.array[wp.spatial_vector]
    ) -> None:
        """Evaluate one wrench per body-particle contact record."""
        model = self.model
        soft_contact_max = contacts.soft_contact_max

        # Same body pose selection as _solve_particle_iteration: the particle side evaluated these
        # contacts against exactly these arrays in the last iteration.
        if self.integrate_with_external_rigid_solver:
            body_q = state_out.body_q
            body_q_prev = state_in.body_q
            body_qd = state_out.body_qd
        else:
            body_q = state_in.body_q
            body_q_prev = self.body_q_prev if model.body_count > 0 else None
            body_qd = state_in.body_qd

        wp.launch(
            kernel=compute_body_particle_contact_forces,
            dim=soft_contact_max,
            inputs=[
                dt,
                state_in.particle_q,
                self.particle_q_prev,
                model.particle_radius,
                model.shape_body,
                body_q,
                body_q_prev,
                body_qd,
                model.body_com,
                self.friction_epsilon,
                self.rigid_soft_contact_use_log_barrier,
                self.body_particle_contact_penalty_k,
                self.body_particle_contact_material_kd,
                self.body_particle_contact_material_mu,
                contacts.soft_contact_count,
                _make_soft_contact_filter_data(model, contacts, state_in.particle_q, body_q),
                contacts.soft_contact_indices,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_barycentric,
                model.shape_margin,
            ],
            outputs=[contact_f],
            device=self.device,
        )

    def collect_rigid_contact_forces(
        self,
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        contacts: Contacts | None,
        dt: float,
    ) -> tuple[
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.vec3],
        wp.array[wp.vec3],
        wp.array[wp.vec3],
        wp.array[wp.int32],
    ]:
        """Collect per-contact rigid contact forces and world-space application points.

        Args:
            body_q: Current body transforms (world frame),
                typically ``state_out.body_q`` after a ``step()`` call.
            body_q_prev: Effective previous-pose history used by the step (world frame).
                Snapshot ``solver.body_q_prev`` before :meth:`step` (it is advanced
                after the step). On a first or reset step, overwrite each rebaselined
                row with that step's input ``body_q`` so its reported force matches the
                solve. For externally integrated bodies, pass the external solver's
                previous transforms.
            contacts: Contact data buffers containing rigid
                contact geometry/material references. If None, the function
                returns default zero/sentinel outputs.
            dt: Time step size [s].

        Note:
            Call after collision generation and ``step()`` with the same
            ``Contacts`` buffer. If rigid contact state is absent or undersized,
            this returns sentinel/zero outputs without growing output buffers.
            Output buffers persist and grow on demand; they do not shrink, so
            iterate up to the returned ``rigid_contact_count`` rather than the
            array length.

        Returns:
            tuple[
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.vec3],
                wp.array[wp.vec3],
                wp.array[wp.vec3],
                wp.array[wp.int32],
            ]: Tuple of per-contact outputs:
                - body0: Body index for shape0, int32.
                - body1: Body index for shape1, int32.
                - point0_world: World-space contact point on body0, wp.vec3 [m].
                - point1_world: World-space contact point on body1, wp.vec3 [m].
                - force_on_body1: Contact force applied to body1 in world frame, wp.vec3 [N].
                - rigid_contact_count: Length-1 active rigid-contact count, int32.
        """
        max_contacts = contacts.rigid_contact_max if contacts is not None else 0

        missing_rigid_state = any(
            arr is None or arr.shape[0] < max_contacts
            for arr in (
                getattr(self, "body_body_contact_penalty_k", None),
                getattr(self, "body_body_contact_normal_rho", None),
                getattr(self, "body_body_contact_material_ke", None),
                getattr(self, "body_body_contact_material_kd", None),
                getattr(self, "body_body_contact_material_mu", None),
                getattr(self, "body_body_contact_tangent_rho", None),
                getattr(self, "body_body_contact_lambda", None),
                getattr(self, "body_body_contact_C0", None),
            )
        )
        no_contact_capacity = contacts is None or max_contacts == 0

        if no_contact_capacity or missing_rigid_state:
            if contacts is not None and contacts.rigid_contact_force is not None:
                contacts.rigid_contact_force.zero_()
            if self._rigid_contact_body0.shape[0] > 0:
                self._rigid_contact_body0.fill_(-1)
                self._rigid_contact_body1.fill_(-1)
                self._rigid_contact_point0_world.zero_()
                self._rigid_contact_point1_world.zero_()
            return (
                self._rigid_contact_body0,
                self._rigid_contact_body1,
                self._rigid_contact_point0_world,
                self._rigid_contact_point1_world,
                contacts.rigid_contact_force if contacts is not None else self._rigid_contact_zero_force,
                self._rigid_contact_zero_count,
            )

        # Type narrowing: remaining path requires a valid Contacts instance.
        assert contacts is not None

        output_capacity = self._rigid_contact_body0.shape[0]
        if output_capacity < max_contacts:
            self._raise_if_capturing_resize("rigid contact output", output_capacity, max_contacts)
            self._rigid_contact_body0 = wp.full(max_contacts, -1, dtype=wp.int32, device=self.device)
            self._rigid_contact_body1 = wp.full(max_contacts, -1, dtype=wp.int32, device=self.device)
            self._rigid_contact_point0_world = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._rigid_contact_point1_world = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)

        wp.launch(
            kernel=compute_rigid_contact_forces,
            dim=max_contacts,
            inputs=[
                float(dt),
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_surface_velocity,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.model.shape_body,
                body_q,
                body_q_prev,
                self.model.body_com,
                self.body_body_contact_penalty_k,
                self.body_body_contact_normal_rho,
                self.body_body_contact_material_ke,
                self.body_body_contact_material_kd,
                self.body_body_contact_material_mu,
                self.body_body_contact_tangent_rho,
                self.body_body_contact_lambda,
                self.body_body_contact_C0,
                self.rigid_contact_alpha,
                self.rigid_contact_hard,
                self.rigid_compliant_alm,
                float(self.friction_epsilon),
            ],
            outputs=[
                self._rigid_contact_body0,
                self._rigid_contact_body1,
                self._rigid_contact_point0_world,
                self._rigid_contact_point1_world,
                contacts.rigid_contact_force,
            ],
            device=self.device,
        )

        return (
            self._rigid_contact_body0,
            self._rigid_contact_body1,
            self._rigid_contact_point0_world,
            self._rigid_contact_point1_world,
            contacts.rigid_contact_force,
            contacts.rigid_contact_count,
        )

    def _finalize_particles(self, state_out: State, dt: float):
        """Finalize particle velocities after VBD iterations."""
        # Early exit if no particles
        if self.model.particle_count == 0:
            return

        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q, state_out.particle_qd],
            dim=self.model.particle_count,
            device=self.device,
        )

    def _finalize_rigid_bodies(self, state_in: State, state_out: State, dt: float):
        """Finalize rigid body velocities and Dahl friction state after VBD iterations (post-iteration phase).

        Updates rigid body velocities using BDF1 and updates Dahl hysteresis state for rod bend/twist.
        Also transfers the final body poses from state_in to state_out.
        """
        model = self.model

        # Early exit if no rigid bodies or rigid bodies are driven by an external solver
        if not self._integrates_rigid_bodies:
            return

        wp.launch(
            kernel=update_body_velocity,
            inputs=[
                dt,
                state_in.body_q,
                model.body_com,
            ],
            outputs=[self.body_q_prev, state_out.body_qd, state_in.body_qd, state_out.body_q],
            dim=model.body_count,
            block_dim=self._rigid_body_block_dim,
            device=self.device,
        )

        if self.enable_dahl_friction and model.joint_count > 0:
            wp.launch(
                kernel=update_rod_dahl_state,
                inputs=[
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    self.joint_constraint_start,
                    self.joint_penalty_k,
                    self.joint_material_k,
                    self.joint_is_hard,
                    self.rigid_compliant_alm,
                    self.joint_rod_rest_kb_local,
                    self.joint_rod_rest_twist,
                    state_out.body_q,
                    self.joint_dahl_eps_max,
                    self.joint_dahl_tau,
                    self.joint_sigma_prev,
                    self.joint_kappa_prev,
                    self.joint_dkappa_prev,
                ],
                dim=model.joint_count,
                block_dim=self._rigid_joint_block_dim,
                device=self.device,
            )

    def _default_collision_frequency_type(self, slot: SolverBase.CollisionSlot) -> SolverBase.CollisionFrequencyType:
        """Resolve ``AUTO``: the self-contact slot follows the legacy VBD behavior.

        With self-contact enabled, ``AUTO`` derives from the deprecated
        ``particle_collision_detection_interval`` when that was set
        (PRE_INIT ~ interval < 0, PRE_POST_INIT ~ interval == 0,
        ITERATIONS ~ interval >= 1) and defaults to the legacy behavior
        (``PRE_POST_INIT``) otherwise. The rigid slot uses the base default
        (PRE_INIT with an owned pipeline).
        """
        Frequency = SolverBase.CollisionFrequencyType
        if slot == SolverBase.CollisionSlot.SOFT_SELF_CONTACT and self.particle_enable_self_contact:
            interval = self._deprecated_particle_interval
            if interval is None or interval == 0:
                return Frequency.PRE_POST_INIT
            if interval < 0:
                return Frequency.PRE_INIT
            return Frequency.ITERATIONS
        if slot == SolverBase.CollisionSlot.RIGID and self.rigid_soft_enable_dat:
            # Symmetric to the self-contact slot: rigid DAT references reset at each
            # detection, so AUTO detects before and right after initialization.
            return Frequency.PRE_POST_INIT
        return super()._default_collision_frequency_type(slot)

    def _rigid_iterate_view(self, state_in: State, state_out: State) -> State:
        """A State aliasing the mid-solve iterate arrays for collision detection.

        No arrays are copied: internal rigid iterations update
        ``state_in.body_q`` in place, externally integrated rigid bodies are
        supplied through ``state_out``. Particle VBD updates ``state_in.particle_q``
        in place; ``state_out`` is only synchronized at the end of each iteration.
        """
        view = State()
        if self.integrate_with_external_rigid_solver:
            view.body_q = state_out.body_q
            view.body_qd = state_out.body_qd
        else:
            view.body_q = state_in.body_q
            view.body_qd = state_in.body_qd
        view.particle_q = state_in.particle_q
        view.particle_qd = state_in.particle_qd
        return view

    def _resolve_self_contact_schedule(self) -> tuple[SolverBase.CollisionFrequencyType, int]:
        """Resolve the soft self-contact slot to its effective mode and frequency."""
        Frequency = SolverBase.CollisionFrequencyType
        slot = SolverBase.CollisionSlot.SOFT_SELF_CONTACT
        mode = self._resolved_collision_frequency_type(slot)
        freq = self._collision_frequency[slot]
        interval = self._deprecated_particle_interval
        if self._collision_frequency_type[slot] == Frequency.AUTO and interval is not None and interval >= 1:
            freq = interval
        return mode, freq

    def _self_contact_is_due(self, iter_num: int) -> bool:
        """Return whether iterative soft self-contact detection is due before this iteration."""
        Frequency = SolverBase.CollisionFrequencyType
        return (
            self._self_contact_mode_this_step == Frequency.ITERATIONS
            and (iter_num + 1) % self._self_contact_freq_this_step == 0
        )

    def _rigid_collision_is_due(self, iter_num: int) -> bool:
        """Return whether iterative rigid collision detection is due before this iteration."""
        Frequency = SolverBase.CollisionFrequencyType
        return self._rigid_mode_this_step == Frequency.ITERATIONS and (iter_num + 1) % self._rigid_freq_this_step == 0

    def _collision_detection_due(self, phase_modes: tuple[SolverBase.CollisionFrequencyType, ...]) -> tuple[bool, bool]:
        """Return whether the rigid and the soft self-contact detectors are due for a schedule phase."""
        rigid_due = self.collision_pipeline is not None and self._rigid_mode_this_step in phase_modes
        soft_due = self.particle_enable_self_contact and self._self_contact_mode_this_step in phase_modes
        return rigid_due, soft_due

    def _pre_initialization_detection(self, state_in: State, dt: float) -> bool:
        """Run the detectors scheduled before initialization on the step's input state.

        Returns whether the rigid detector ran, in which case initialization must
        rebuild the rigid contact state from the refreshed contacts.
        """
        Frequency = SolverBase.CollisionFrequencyType
        rigid_due, soft_due = self._collision_detection_due(
            (Frequency.PRE_INIT, Frequency.PRE_POST_INIT, Frequency.ITERATIONS)
        )
        if rigid_due or soft_due:
            self._refresh_collision_sets(state_in, dt, run_rigid_collision=rigid_due, run_soft_self_collision=soft_due)
        return rigid_due

    def _mid_step_detection(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
        *,
        rigid_due: bool,
        soft_due: bool,
        preserve_history: bool,
    ) -> None:
        """Detect from the current rigid iterate and rebuild the rigid contact state.

        Shared by the post-initialization pass and the in-iteration passes. The latter,
        and the former when contact history is enabled, set ``preserve_history``: the
        current contact state is snapshotted before the pipeline refresh so the
        in-flight ALM multipliers survive it through contact matching.
        """
        if not (rigid_due or soft_due):
            return
        collision_state = self._rigid_iterate_view(state_in, state_out)
        if rigid_due and preserve_history:
            self._snapshot_rigid_contact_history(contacts, force=True)
        self._refresh_collision_sets(
            collision_state, dt, run_rigid_collision=rigid_due, run_soft_self_collision=soft_due
        )
        if rigid_due:
            self._refresh_rigid_contact_state(contacts, refresh=True, restore_history=preserve_history)
            self._step_body_body_contact_frame(contacts, collision_state.body_q, dt, 1.0, 1.0)
            self._refresh_body_particle_contact_state(contacts, refresh=True)

    def _validate_dat_collision_schedules(self) -> None:
        """Require active DAT families to have usable, mutually consistent schedules."""
        Frequency = SolverBase.CollisionFrequencyType
        if self.rigid_soft_enable_dat and self._rigid_mode_this_step == Frequency.NONE:
            raise ValueError(
                "rigid_soft_enable_dat requires an active rigid collision schedule: "
                "set the RIGID slot to PRE_INIT, PRE_POST_INIT, or ITERATIONS, not NONE."
            )
        if self.particle_enable_self_contact and self._self_contact_mode_this_step == Frequency.NONE:
            raise ValueError(
                "particle_enable_self_contact requires an active soft self-collision schedule: "
                "set the SOFT_SELF_CONTACT slot to PRE_INIT, PRE_POST_INIT, or ITERATIONS, not NONE."
            )
        schedules_match = self._rigid_mode_this_step == self._self_contact_mode_this_step and (
            self._rigid_mode_this_step != Frequency.ITERATIONS
            or self._rigid_freq_this_step == self._self_contact_freq_this_step
        )
        if self.rigid_soft_enable_dat and self.particle_enable_self_contact and not schedules_match:
            raise ValueError(
                "rigid-soft DAT and self-contact DAT share one motion reference and require "
                "equivalent RIGID and SOFT_SELF_CONTACT schedules (same type; same frequency for ITERATIONS)."
            )

    def _collision_detection_penetration_free(self, current_state: State, *, reset_reference: bool = True):
        # particle_displacements is based on pos_prev_collision_detection
        # so reset them every time we do collision detection
        if reset_reference:
            self.pos_prev_collision_detection.assign(current_state.particle_q)
            self.particle_displacements.zero_()

        if self.collision_pipeline is not None:
            self.collision_pipeline._detect_soft_self_contact(current_state.particle_q, self._pipeline_contacts)
        else:
            self.trimesh_collision_detector.refit(current_state.particle_q)
            self.trimesh_collision_detector.vertex_triangle_collision_detection(
                self._self_contact_query_radius,
                min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.particle_q_rest,
            )
            self.trimesh_collision_detector.edge_edge_collision_detection(
                self._self_contact_query_radius,
                min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.particle_q_rest,
            )
        if self._cut_xfem:
            # The XFEM-type cut removes self-contact pairs from the lists: a virtual corner is not
            # material, and the wound face of a fragment has no contact with the near rows of its own
            # side. "Near" is the detection radius, measured in the rest shape.
            x = self._xfem
            wp.launch(
                kernel=filter_cut_self_contacts,
                dim=max(self.model.particle_count, self.model.edge_count),
                inputs=[
                    1 if self.cut_xfem_contact_filter else 0,
                    1 if self.cut_xfem_front_rows else 0,
                    self._self_contact_query_radius,
                    x["edge_kind"],
                    x["particle_ghost"],
                    x["wound_edge"],
                    x["wound_side"],
                    x["particle_level"],
                    x["particle_level_set"],
                    x["particle_stroke"],
                    self.model.particle_q,
                    x["edge_owner"],
                    x["edge_owner_local"],
                    self.model.tet_indices.reshape((self.model.tet_count, 4)),
                    x["tet_vertex"],
                    self.model.tri_indices,
                    self.model.edge_indices,
                ],
                outputs=[self.trimesh_collision_info],
                device=self.device,
            )

    def rebuild_bvh(self, state: State):
        """This function will rebuild the BVHs used for detecting self-contacts using the input `state`.

        When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        quality. In these cases, rebuilding the entire tree is necessary to achieve better querying efficiency.

        Args:
            state:  The state whose particle positions (:attr:`~newton.State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.particle_enable_self_contact:
            if self.collision_pipeline is not None:
                self.collision_pipeline.refit_soft_self_contact_bvh(state.particle_q, rebuild=True)
            else:
                self.trimesh_collision_detector.rebuild(state.particle_q)
