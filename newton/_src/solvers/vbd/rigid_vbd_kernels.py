# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Rigid body VBD solver kernels and utilities.

This module contains all rigid body-specific kernels, device functions, data structures,
and constants for the VBD solver's rigid body domain (AVBD algorithm).

Organization:
- Constants: Solver parameters and thresholds
- Data structures: RigidForceElementAdjacencyInfo and related structs
- Device functions: Helper functions for rigid body dynamics
- Utility kernels: Adjacency building
- Pre-iteration kernels: Forward integration, warmstarting, Dahl parameter computation
- Iteration kernels: Contact accumulation, rigid body solve, dual updates
- Post-iteration kernels: Velocity updates, Dahl state updates
"""

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def forward_step_rigid(
    dt: float,
    gravity: wp.array[wp.vec3],
    body_q_in: wp.array[wp.transform],
    body_qd_in: wp.array[wp.spatial_vector],
    body_f_in: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    angular_damping: float,
    # outputs
    body_q_prev_out: wp.array[wp.transform],
    body_q_inertial_out: wp.array[wp.transform],
    body_q_out: wp.array[wp.transform],
    body_qd_out: wp.array[wp.spatial_vector],
):
    pass
