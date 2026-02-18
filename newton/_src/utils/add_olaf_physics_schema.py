import argparse

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
        if any(name in path[-2] for name in ("PELVIS", "HIP", "KNEE", "ANKLE", "FOOT", "NECK", "SHOULDER", "ARM", "JAW", "BROW", "EYE", "HEAD")):
            print(f"Applying RigidBodyAPI to {prim}")
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigidBodyAPI.CreateKinematicEnabledAttr(True)

    print(f"Saving to {output_path}")
    stage.Export(output_path)