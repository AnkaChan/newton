import argparse

from pxr import Usd, UsdPhysics

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
        if any(name in path[-1] for name in ("HEAD", "HIP", "KNEE", "PELVIS", "NECK", "FOOT")):
            print(f"Applying RigidBodyAPI to {prim}")
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigidBodyAPI.CreateKinematicEnabledAttr(True)

    print(f"Saving to {output_path}")
    stage.Export(output_path)
