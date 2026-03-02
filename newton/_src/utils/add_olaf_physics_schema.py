import argparse

from pxr import Usd, UsdGeom, UsdPhysics


def apply_collision_api(prim):
    type_name = str(prim.GetTypeName()).lower()

    if type_name in ("mesh", "capsule", "sphere", "box", "cylinder", "cone"):
        # print(f"Applying CollisionAPI to {prim}")
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

    # RIGID BODIES (adjust)
    olaf_test_v1 = ("stoolWoodB", "vaseG")
    olaf_mar1_v1 = ("colFruitBasket1", "colCartAxel", "colFruitBasket2", "colCartMid", "colCartTop", "colCartBase", "colBoxBag", "colWheels2", "colWheels1", "colWheels4", "colWheels3", "woodbeam1", "woodbeam2", "colFruitTop", "REDAPPLE_COL")
    # Match by prim name (last path component)
    olaf_mar1_v2 = ("REDAPPLE_COL", "colFruitBasket1", )
    # Match by exact full prim path
    rigid_body_exact_paths = {
        "/World/CartHero_01/geo/applesBasket",
        "/World/CartHero_01/geo/cart",
        "/World/AppleHero_01/geo",
        "/World/AppleHero_02/geo",
        "/World/AppleHero_03/geo",
    }

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if "cobble" in prim_path:
            continue
        path = prim_path.split("/")

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

        elif (
            prim_path in rigid_body_exact_paths
        ):
            print(f"Applying RigidBodyAPI and MassAPI to {prim}")
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            massAPI = UsdPhysics.MassAPI.Apply(prim)
            apply_collision_api(prim)

    print(f"Saving to {output_path}")
    stage.Export(output_path)
