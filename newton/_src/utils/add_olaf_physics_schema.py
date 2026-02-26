import argparse

from pxr import Usd, UsdPhysics


def apply_collision_api(prim):
    type_name = str(prim.GetTypeName()).lower()

    if type_name in ("mesh", "capsule", "sphere", "box", "cylinder", "cone"):
        print(f"Applying CollisionAPI to {prim}")
        collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        collisionAPI.CreateCollisionEnabledAttr(True)

    for child in prim.GetChildren():
        apply_collision_api(child)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    args = parser.parse_args()

    output_path = args.input_path.replace(".usd", "_physics.usd")

    stage = Usd.Stage.Open(args.input_path)

    for prim in stage.Traverse():
        if "cobble" in str(prim.GetPath()):
            continue
        path = str(prim.GetPath()).split("/")

        # ROBOT
        if any(name in path[-1] for name in ("PELVIS", "HIP", "KNEE", "ANKLE", "FOOT", "NECK", "SHOULDER", "ARM", "JAW", "BROW", "EYE", "HEAD")):
        # if any(name in path[-1] for name in ("FOOT", "ANKLE", "PELVIS")):
            print(f"Applying RigidBodyAPI to {prim}")
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigidBodyAPI.CreateKinematicEnabledAttr(True)

            for child in prim.GetChildren():
                apply_collision_api(child)

        # TERRAIN (adjust)
        elif any(name in path[-1] for name in ("Plane",)):
            print(f"Applying CollisionAPI to {prim}")
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
            collisionAPI.CreateCollisionEnabledAttr(True)

        # RIGID BODIES (adjust)
        elif len(path) == 5 and any(name in path[-1] for name in ("stoolWoodB", )):
            print(f"prim.GetPath(): {str(prim.GetPath())}")
            print(f"Applying RigidBodyAPI and MassAPI to {prim}")
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            massAPI = UsdPhysics.MassAPI.Apply(prim)
            apply_collision_api(prim)

    print(f"Saving to {output_path}")
    stage.Export(output_path)
