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
from newton._src.sim import BodyFlag

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
    tid = wp.tid()
    q = body_q_in[tid]
    qd = body_qd_in[tid]
    f = body_f_in[tid]
    com_local = body_com[tid]
    I_body = body_inertia[tid]
    inv_m = body_inv_mass[tid]
    inv_I_body = body_inv_inertia[tid]
    flags = body_flags[tid]
    world_idx = body_world[tid]
    g = gravity[wp.max(world_idx, 0)]

    if (flags & BodyFlags.KINEMATIC) != 0:
       body_q_inertial_out[tid]=q; body_q_out[tid]=q;
       body_qd_out[tid]=qd; return

    body_q_prev_out[tid] = q

    # linear velocity
    v = wp.spatial_top(qd)
    # angular velocity
    w = wp.spatial_bottom(qd)

    # linear force
    f_lin = wp.spatial_top(f)
    # torque
    tau = wp.spatial_bottom(f)

    # rotational part (quaternion)
    rot = wp.transform_get_rotation(q)
    # semi-implicit rotation integration
    w_b = wp.quat_rotate_inv(rot, w)
    tau_b = wp.quat_rotate_inv(rot, tau)
    w_new_b = w_b + dt * inv_I_body * (tau_b - wp.cross(w_b, I_body * w_b))
    w_new = wp.quat_rotate(rot, w_new_b)
    # integrate rotation
    rot_new = wp.normalize(rot + 0.5 * dt * wp.quat(w_new, 0) * rot)
    
    # linear part
    x_com     = x + wp.quat_rotate(rot, com_local)
    v_new = v + dt *(f_lin * inv_m + g)
    x_com_new = x_com + dt * v_new
    # x_com and body_frame origina does not overlap
    pos_new   = x_com_new - wp.quat_rotate(rot_new, com_local)

    q_new = wp.transform(pos_new, rot_new)

    body_q_inertial_out[tid] = q_new
    body_q_out[tid] = q_new
    body_qd_out[tid] = wp.spatial_vector(v_new, w_new)
