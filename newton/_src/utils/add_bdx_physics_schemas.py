import argparse

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def apply_collision_api(prim):
    type_name = str(prim.GetTypeName()).lower()

    if type_name in ("mesh", "capsule", "sphere", "box", "cylinder", "cone"):
        print(f"Applying CollisionAPI to {prim}")
        collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        collisionAPI.CreateCollisionEnabledAttr(True)

    for child in prim.GetChildren():
        apply_collision_api(child)


chains = [
    [
        "HangingLanternChainA_09",
        "HangingLanternChainA_02",
        "HangingLanternE_01",
    ],
    [
        "HangingLanternChainA_08",
        "HangingLanternChainA_05",
        "HangingLanternA_01",
    ],
    [
        "HangingLanternChainA_10",
        "HangingLanternChainA_03",
        "HangingLanternD_01",
    ],
    [
        "HangingLanternChainA_13",
        "HangingLanternChainA_06",
        "HangingLanternA_02",
    ],
    [
        "HangingLanternChainA_04",
        "HangingLanternA_03",
    ],
    [
        "HangingLanternChainA_12",
        "HangingLanternChainA_01",
        "HangingLanternC_01",
    ],
]

chain_link_length = 0.6


def parse_xform(prim):
    xform = UsdGeom.Xform(prim)
    mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
    rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
    pos = mat[3, :3]
    return wp.transform(pos, rot)


def add_lantern_joints(stage):
    world = stage.GetPrimAtPath("/World")

    for chain_idx, chain in enumerate(chains):
        # register links as rigid bodies
        bodies = []
        for i, link in enumerate(chain):
            chain_prim = world.GetPrimAtPath(f"/World/{link}")
            UsdPhysics.RigidBodyAPI.Apply(chain_prim)
            apply_collision_api(world.GetPrimAtPath(f"/World/{link}/geo"))
            bodies.append(chain_prim)

            if i > 0 and i < len(chain) - 1:
                # connect links with fixed joints
                fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"/World/fixed_joint_{chain_idx}_{i}")
                fixed_joint.GetBody0Rel().AddTarget(bodies[i - 1].GetPrimPath())
                fixed_joint.GetBody1Rel().AddTarget(bodies[i].GetPrimPath())
                fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, chain_link_length))

        UsdPhysics.ArticulationRootAPI.Apply(bodies[0])

        anchor_joint = UsdPhysics.SphericalJoint.Define(stage, f"/World/anchor_joint_{chain_idx}")
        anchor_joint.GetBody0Rel().AddTarget(bodies[0].GetPrimPath())
        anchor_joint.GetBody1Rel().AddTarget("/World")
        anchor_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, chain_link_length))

        lantern_joint = UsdPhysics.SphericalJoint.Define(stage, f"/World/lantern_joint_{chain_idx}")
        lantern_joint.GetBody0Rel().AddTarget(bodies[-2].GetPrimPath())
        lantern_joint.GetBody1Rel().AddTarget(bodies[-1].GetPrimPath())

        # figure out world pose of the lantern to compute the right offset from the last chain link
        lantern_xform = parse_xform(bodies[-1])
        chain_xform = parse_xform(bodies[-2])
        lantern_offset = lantern_xform.p - chain_xform.p
        lantern_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -lantern_offset[2]))

        print(f"Added lanterns for chain {chain_idx} [{', '.join(chain)}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    args = parser.parse_args()

    output_path = args.input_path.replace(".usd", "_physics.usd")

    stage = Usd.Stage.Open(args.input_path)

    for prim in stage.Traverse():
        if "proxy" in str(prim.GetPath()):
            continue
        path = str(prim.GetPath()).split("/")

        # ROBOT
        if any(name in path[-1] for name in ("HEAD", "HIP", "KNEE", "PELVIS", "NECK", "FOOT")):
            print(f"Applying RigidBodyAPI to {prim}")
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigidBodyAPI.CreateKinematicEnabledAttr(True)

            for child in prim.GetChildren():
                apply_collision_api(child)

        # TERRAIN (adjust)
        elif any(name in path[-1] for name in ("terrainMaincol",)):
            print(f"Applying CollisionAPI to {prim}")
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
            collisionAPI.CreateCollisionEnabledAttr(True)

        # RIGID BODIES (adjust)
        elif len(path) == 5 and any(name in path[-1] for name in ("gear", "piece", "piston")):
            print(f"Applying RigidBodyAPI and MassAPI to {prim}")
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            massAPI = UsdPhysics.MassAPI.Apply(prim)

            for child in prim.GetChildren():
                apply_collision_api(child)

    # check if lanterns are present
    if stage.GetPrimAtPath("/World/HangingLanternA_01") is not None:
        print("Adding lanterns")
        add_lantern_joints(stage)

    print(f"Saving to {output_path}")
    stage.Export(output_path)
