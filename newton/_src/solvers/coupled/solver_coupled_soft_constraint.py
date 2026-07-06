# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Soft-transform-constraint coupled multi-solver simulations.

This port reproduces Genesis' ``two_way_soft_constraint`` cloth/rigid coupling
strategy on top of the lagged-impulse proxy machinery
(:class:`~newton.solvers.experimental.coupled.SolverCoupledProxy`). The rigid
solver (e.g. MuJoCo) drives a *massless* proxy of each coupled body inside the
destination cloth solver (VBD); the proxy is tied to the commanded pose by a
mass-weighted penalty spring, and the spring's reaction wrench is fed back to
the rigid solver each substep.

The mapping onto Newton's AVBD machinery rests on a single observation: AVBD's
per-body inertial term is *itself* a quadratic pose spring,

    f = (p_target - p_current) * m / dt^2 ,

pulling the body toward its inertial target ``body_inertia_q`` with stiffness
``m / dt^2``. Genesis' soft-transform constraint is the same quadratic penalty
with stiffness ``eta * m / dt^2`` (``eta`` the dimensionless strength ratio).
So installing the proxy's destination mass/inertia as ``eta_p * m`` /
``eta_a * I`` makes the AVBD inertial term reproduce Genesis' spring exactly,
and the spring residual at the solved pose,

    F   = m_proxy / dt^2 * (p_solved - p_target)        (Genesis kappa_p * m * dp)
    tau = I_world / dt^2 * log(R_solved R_target^T)      (Genesis kappa_a * I_world * theta)

is the reduced reaction of Genesis' ``_apply_abd_coupling_forces`` (math doc
section 4). Because the proxy is (near-)massless and quasi-static, that residual
also equals the cloth contact load on the proxy -- the "massless relay" result.

We harvest the deviation against the inertial *target* ``body_inertia_q`` rather
than the raw commanded aim, so the deviation is purely contact-induced
(velocity/gravity drift of the target is absorbed); for a massless proxy that
snaps to the aim absent contact, ``body_inertia_q == aim`` and the two agree,
matching Genesis' ``external_kinetic`` proxy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .solver_coupled import SolverCoupled
from .solver_coupled_proxy import SolverCoupledProxy

if TYPE_CHECKING:
    from ...sim import Model

__all__ = ["SolverCoupledSoftConstraint"]


@wp.kernel(enable_backward=False)
def harvest_proxy_softconstraint_forces_kernel(
    dt: float,
    body_local_to_proxy_global: wp.array[int],
    target_body_q: wp.array[wp.transform],
    solved_body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    out_coupling_forces: wp.array[wp.spatial_vector],
):
    """Reduced soft-transform reaction from the proxy pose deviation.

    ``target_body_q`` is the AVBD inertial target (the spring's target pose),
    ``solved_body_q`` the destination-solved proxy pose. ``body_mass`` /
    ``body_inertia`` are the destination proxy's installed (eta-scaled) values,
    so ``m / dt^2`` is Genesis' ``kappa_p * m`` and ``I_world / dt^2`` its
    ``kappa_a * I_world``. The wrench is expressed in the world frame at the
    body origin, matching ``State.body_f``.
    """
    local_id = wp.tid()
    global_id = body_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    inv_dt2 = 1.0 / (dt * dt)

    target = target_body_q[local_id]
    solved = solved_body_q[local_id]

    p_target = wp.transform_get_translation(target)
    p_solved = wp.transform_get_translation(solved)
    r_target = wp.transform_get_rotation(target)
    r_solved = wp.transform_get_rotation(solved)

    m = body_mass[local_id]
    i_body = body_inertia[local_id]

    # Linear reduced force: F = kappa_p * m * (p_solved - p_target).
    f = (p_solved - p_target) * (m * inv_dt2)

    # Rotation deviation log(R_solved R_target^T) as a world-frame rotvec.
    q_rel = wp.mul(r_solved, wp.quat_inverse(r_target))
    if q_rel[3] < 0.0:
        q_rel = wp.quat(-q_rel[0], -q_rel[1], -q_rel[2], -q_rel[3])
    axis, angle = wp.quat_to_axis_angle(q_rel)
    theta = axis * angle

    # tau = kappa_a * I_world * theta, with I_world = R_solved I_body R_solved^T.
    rot = wp.quat_to_matrix(r_solved)
    i_world = rot * i_body * wp.transpose(rot)
    tau = (i_world * theta) * inv_dt2

    wp.atomic_add(out_coupling_forces, global_id, wp.spatial_vector(f, tau))


class SolverCoupledSoftConstraint(SolverCoupledProxy):
    """Couple two solvers with Genesis-style soft-transform-constraint proxies.

    Identical orchestration to :class:`SolverCoupledProxy` (lagged source/proxy
    state exchange, contact filtering, relaxation), but the proxy is anchored to
    its commanded pose by a mass-weighted penalty spring and the reaction is the
    spring residual (Genesis ``two_way_soft_constraint``) rather than the
    destination momentum/contact harvest.

    Args:
        model: Coupled model.
        entries: Sub-solver entries (exactly as :class:`SolverCoupledProxy`).
        coupling: Proxy configuration.
        constraint_strength_translation: Genesis ``eta_p`` -- dimensionless
            translation spring-strength ratio; the proxy's destination mass is
            installed as ``eta_p * m_own``. Keep within ``[0, 100]``.
        constraint_strength_rotation: Genesis ``eta_a`` -- dimensionless
            rotation spring-strength ratio; the proxy's destination inertia is
            installed as ``eta_a * I_own``. Keep within ``[0, 100]``.
    """

    def __init__(
        self,
        model: Model,
        entries: Sequence[SolverCoupled.Entry],
        coupling: SolverCoupledProxy.Config,
        *,
        constraint_strength_translation: float = 100.0,
        constraint_strength_rotation: float = 100.0,
    ) -> None:
        # Set before super().__init__: the base constructor installs proxy
        # masses (via the overridden _apply_proxy_body_effective_masses).
        self._eta_p = float(constraint_strength_translation)
        self._eta_a = float(constraint_strength_rotation)
        super().__init__(model, entries, coupling)
        self._install_soft_constraint_harvest()

    def _apply_proxy_body_effective_masses(self) -> None:
        """Install the soft-constraint spring stiffness as the proxy mass.

        Genesis weights the spring by the body's *own* mass matrix scaled by
        ``eta`` (``M~ = eta_p M_cm + eta_a M_rot``), not by the articulated
        effective mass. We therefore read the body's own mass/inertia from the
        global model and install ``eta_p * m`` / ``eta_a * I`` into the
        destination proxy body. The AVBD inertial term then reproduces Genesis'
        spring of stiffness ``eta * m / dt^2``.
        """
        device = self.model.device
        body_mass = self.model.body_mass.numpy()
        body_inertia = self.model.body_inertia.numpy()

        for proxy in self._proxy_mappings:
            if proxy.src_ids is None or proxy.src_ids.shape[0] == 0:
                continue
            dst = self._entries[proxy.dst_name]
            global_ids = proxy.src_ids.numpy()  # global body ids (== Proxy.bodies)
            proxy_masses = wp.array(
                [self._eta_p * float(body_mass[g]) for g in global_ids],
                dtype=float,
                device=device,
            )
            proxy_inertias = wp.array(
                [wp.mat33(np.asarray(body_inertia[g], dtype=np.float32) * self._eta_a) for g in global_ids],
                dtype=wp.mat33,
                device=device,
            )
            self._apply_body_inertia_override(dst, proxy.proxy_ids_local, proxy_masses, proxy_inertias)

    def _install_soft_constraint_harvest(self) -> None:
        """Route the soft-transform reaction through each destination solver's
        ``coupling_harvest_proxy_wrenches`` hook.

        We deliberately do **not** modify :class:`SolverCoupledProxy`. The base
        ``_step_proxy`` already invokes ``dst.solver.coupling_harvest_proxy_wrenches``
        to populate ``coupling_forces`` (which it then feeds back to the source),
        so installing the soft-transform reaction on that hook -- the documented
        :class:`~newton._src.solvers.coupled.interface.CouplingInterface`
        extension point -- gives Genesis' spring-residual reaction with zero
        changes to the shared coupler. Destinations without an AVBD inertial
        target (``body_inertia_q``) keep their own harvest.
        """
        installed: set[int] = set()
        for proxy in self._proxy_mappings:
            entry = self._entries.get(proxy.dst_name)
            if entry is None or getattr(entry, "solver", None) is None:
                continue
            dst_solver = entry.solver
            if id(dst_solver) in installed:
                continue
            if getattr(dst_solver, "body_inertia_q", None) is None:
                continue  # non-AVBD destination: leave its default harvest in place
            installed.add(id(dst_solver))
            coupler = self

            def _soft_harvest(
                body_local_to_proxy_global,
                out_body_f,
                *,
                body_qd_before,
                state,
                state_out,
                contacts,
                dt,
                _dst=dst_solver,
            ):
                del body_qd_before, state, contacts
                coupler._harvest_soft_constraint_reaction(_dst, body_local_to_proxy_global, out_body_f, state_out, dt)

            # Instance-attribute override (plain function, not bound) shadowing the
            # solver's method only for the instances this coupler owns.
            dst_solver.coupling_harvest_proxy_wrenches = _soft_harvest

    def _harvest_soft_constraint_reaction(self, dst_solver, body_local_to_proxy_global, out_body_f, state_out, dt):
        """Spring residual ``m_proxy / dt^2 * (solved - target)`` (and rotational
        analog) into ``out_body_f``, with ``target`` the AVBD inertial target so
        the deviation is purely contact-induced."""
        out_body_f.zero_()
        wp.launch(
            harvest_proxy_softconstraint_forces_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                body_local_to_proxy_global,
                dst_solver.body_inertia_q,
                state_out.body_q,
                dst_solver.model.body_mass,
                dst_solver.model.body_inertia,
                out_body_f,
            ],
            device=self.model.device,
        )
