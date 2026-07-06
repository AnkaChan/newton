# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Soft-transform-constraint coupling strategy: MuJoCo arm + VBD cloth.

Genesis ``two_way_soft_constraint`` port. Identical scene/robot/material setup
to the lagged-impulse :class:`~newton.exp.solvers.proxy.ProxyCouplingStrategy`
(it subclasses it), but the gripper bodies are coupled into the VBD cloth solver
through :class:`SolverCoupledSoftConstraint`: each finger has a massless proxy
tied to its commanded pose by a mass-weighted penalty spring, and the spring
residual is fed back to the MuJoCo arm as the reaction wrench.

The two knobs are the dimensionless spring-strength ratios
``--constraint-strength-translation`` (eta_p) and
``--constraint-strength-rotation`` (eta_a), mirroring Genesis'
``IPCCouplerOptions(constraint_strength_{translation,rotation})``.
"""

from __future__ import annotations

from newton.solvers.experimental.coupled import SolverCoupledSoftConstraint

from .proxy import ProxyCouplingStrategy
from . import register


@register
class SoftConstraintCouplingStrategy(ProxyCouplingStrategy):
    key = "soft_constraint"

    def build_solver(self, model, handles, args):
        entries, coupling = self._build_entries_and_coupling(model, handles, args)
        self.solver = SolverCoupledSoftConstraint(
            model=model,
            entries=entries,
            coupling=coupling,
            constraint_strength_translation=float(args.constraint_strength_translation),
            constraint_strength_rotation=float(args.constraint_strength_rotation),
        )
        return self.solver

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument(
            "--constraint-strength-translation",
            type=float,
            default=100.0,
            help="Soft-constraint translation strength ratio eta_p (Genesis default 100).",
        )
        parser.add_argument(
            "--constraint-strength-rotation",
            type=float,
            default=100.0,
            help="Soft-constraint rotation strength ratio eta_a (Genesis default 100).",
        )
