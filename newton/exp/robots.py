# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared robot-building helpers for experiments (Franka FR3).

A scene calls :func:`add_franka` to add the arm to both the simulation model
and the IK model, so their joint coordinates line up.

We use Newton's bundled FR3 arm. The FR3 is kinematically near-identical to the
classic Panda IsaacLab loads (joint limits and link lengths match, so the IK
solves to the same configuration); only the link inertials differ slightly. The
isaacsim Panda URDF was evaluated but gave a larger spawn-to-home transient (its
URDF inertials and a zero-mass link8 differ from IsaacLab's panda USD), so the
bundled FR3 is both cleaner and a closer behavioural match.
"""

from __future__ import annotations

import glob
import os

import warp as wp

import newton
import newton.utils


def _find_panda_usd() -> str | None:
    """Locate IsaacLab's Panda USD so the arm's mass/inertia match exactly.

    Opt-in via ``NEWTON_EXP_PANDA_USD=1`` (or a path). Loading it pulls in pxr at
    build time and, in testing, did not improve behaviour over the bundled FR3
    (the residual first-step transient is a stepping effect, not the robot), so
    the default stays on the pxr-free FR3.
    """
    env = os.environ.get("NEWTON_EXP_PANDA_USD", "")
    if not env:
        return None
    if env not in ("1", "true", "True") and os.path.isfile(env):
        return env
    roots = ["/tmp/Assets", os.path.expanduser("~/Assets"), os.environ.get("ISAAC_ASSETS_DIR", "")]
    for root in roots:
        if not root:
            continue
        hits = glob.glob(os.path.join(root, "**", "Robots", "FrankaEmika", "panda_instanceable.usd"), recursive=True)
        if hits:
            return hits[0]
    return None


_PANDA_USD = _find_panda_usd()

# Top-down gripper orientation: 180 deg about world x flips the hand z-axis to
# -z (approach straight down). Quaternion order is (qx, qy, qz, qw).
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)

# Finger joint targets [m] (per finger).
GRIP_OPEN = 0.04
GRIP_CLOSE = 0.0
GRIP_FORCE = 1500.0  # gripper effort limit [N] (IsaacLab panda_hand)
GRIP_STIFFNESS = 1000.0  # finger PD stiffness [N/m] (IsaacLab panda_hand)
GRIP_DAMPING = 100.0

# Number of actuated arm joints (excludes the two fingers).
ARM_DOF = 7

# Label suffix of the body the IK objective targets (the TCP is offset from it).
HAND_BODY_SUFFIX = "panda_hand" if _PANDA_USD else "fr3_hand"


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    world_idx = wp.tid()
    joint_q[world_idx, idx0] = finger_pos[world_idx]
    joint_q[world_idx, idx1] = finger_pos[world_idx]


def find_label_index(labels: list[str], suffix: str) -> int:
    for index, label in enumerate(labels):
        if label.endswith(suffix):
            return index
    raise ValueError(f"Could not find label ending in {suffix!r}")


def add_franka(builder, *, collapse_fixed_joints: bool = False, base_z: float = 0.0):
    """Add the Franka arm to ``builder`` at its rest (zero) joint configuration.

    The spawn configuration is applied later to the simulation state (not here),
    because the VBD solver captures ``joint_rest_angle`` / the rest pose from the
    build-time joint config -- it must stay at the zero/rest pose.

    Args:
        builder: Target :class:`newton.ModelBuilder`.
        collapse_fixed_joints: Whether to collapse fixed joints (monolithic VBD
            wants this; the IK and MuJoCo paths leave it off).
        base_z: Mount height [m] of the arm base.

    Returns:
        ``(body_ids, joint_ids, shape_ids)`` ranges the arm occupies.
    """
    body_start = builder.body_count
    joint_start = builder.joint_count
    shape_start = builder.shape_count
    if _PANDA_USD is not None:
        # Duplicate IsaacLab's cloner (newton_replicate): build the robot as a
        # prototype with the PhysX/Newton schema resolvers (so joint drives,
        # limits, armature, frames and mass/inertia parse identically) and
        # convex-hull collision meshes, then add_builder it. A plain add_usd
        # parses the physics schema generically and yields a different model.
        from pxr import Usd

        from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx  # noqa: PLC0415

        stage = Usd.Stage.Open(_PANDA_USD)
        proto = newton.ModelBuilder(up_axis="Z")
        proto.add_usd(
            stage,
            root_path="/panda",
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            collapse_fixed_joints=collapse_fixed_joints,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        proto.approximate_meshes("convex_hull", keep_visual_shapes=True)
        builder.add_builder(proto, xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()))
    else:
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
            collapse_fixed_joints=collapse_fixed_joints,
        )
    # NOTE: do NOT seed the spawn config into builder.joint_q here. The solver
    # captures joint_rest_angle from the build-time joint config; IsaacLab builds
    # at the zero config (rest=0) and spawns at the default config on the STATE.
    # Seeding the default here would set rest=default and destabilize the drive.
    return (
        list(range(body_start, builder.body_count)),
        list(range(joint_start, builder.joint_count)),
        list(range(shape_start, builder.shape_count)),
    )


def gripper_body_ids(model, robot_bodies: list[int]) -> list[int]:
    """Return the hand + finger body ids (exposed to VBD as proxies)."""
    return gripper_body_ids_from_labels(model.body_label, robot_bodies)


def gripper_body_ids_from_labels(body_labels: list[str], robot_bodies: list[int]) -> list[int]:
    """Return the hand + finger body ids from a label list (usable at builder time)."""
    ids = [b for b in robot_bodies if "hand" in body_labels[b] or "finger" in body_labels[b]]
    if not ids:
        raise RuntimeError("Could not locate Franka gripper bodies")
    return ids
