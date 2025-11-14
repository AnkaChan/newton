########################################################
#
# !!! Use this script with Kit's `python.bat/.sh` !!!
#
# Usage:
# From Isaac Sim 5.1's executable directory:
#     python.bat render.py <stage_path> -o <output_path> -n <num_frames>
#
# Eg: on Windows, when build from source, that directory is: `isaac_sim\_build\windows-x86_64\release`
#
# Only <stage_path> is required, it can be an `omniverse://` path or a local path. All other arguments have default values.
########################################################

import argparse
import os
import sys
from pathlib import Path

try:
    from isaacsim import SimulationApp
except ImportError:
    print("Are you running this with Kit's `python.bat/sh`?\nCheck the top of the file for instructions.")
    sys.exit(1)


def do_render(stage_path, output_path, start_time, num_frames, frame_rate):
    simulation_app = SimulationApp({"headless": True})

    import omni
    import omni.replicator.core as rep
    import omni.timeline
    from isaacsim.core.api import SimulationContext
    from pxr import Usd, UsdGeom

    # -----------------------------------------------------

    def get_cameras(stage: Usd.Stage) -> list[Usd.Prim]:
        """Return camera prims from the stage, **excluding** the default Kit cameras."""
        return [
            prim
            for prim in stage.Traverse()
            if prim.IsA(UsdGeom.Camera) and not prim.GetName().startswith("OmniverseKit")
        ]

    # -----------------------------------------------------

    success = omni.usd.get_context().open_stage(stage_path)

    if not success:
        print(f"Failed to open stage {stage_path}.")
        simulation_app.close()
        return 1

    stage = omni.usd.get_context().get_stage()
    simulation_app.update()

    camera_render_products = [
        rep.create.render_product(camera.GetPath(), (1920, 1080), name=camera.GetName())
        for camera in get_cameras(stage)
    ]

    simulation_context = SimulationContext(
        physics_dt=1.0 / frame_rate,
        rendering_dt=1.0 / frame_rate,
        stage_units_in_meters=1.0,
    )
    simulation_app.update()

    camera_writer = rep.WriterRegistry.get("BasicWriter")
    camera_writer.initialize(
        output_dir=output_path,
        rgb=True,
    )
    camera_writer.attach(camera_render_products)

    simulation_app.update()

    simulation_context.play()
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(start_time)

    for _ in range(num_frames):
        simulation_app.update()

    # cleanup and shutdown
    simulation_context.stop()
    simulation_app.close()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Render USD stage to images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python.bat/.sh lidar_sdg.py <stage_path> <output_path> <sensor_name> <num_frames> <sensor_frequency>

See the top of the file for more information.
        """,
    )
    parser.add_argument("stage_path", type=str, help="Path to the stage to record LiDAR data from")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output directory")
    parser.add_argument("-n", "--num_frames", type=int, default=1, help="Number of frames to record")
    parser.add_argument("-t", "--start_time", type=float, default=0.0, help="Start time")
    parser.add_argument("-f", "--frame_rate", type=int, default=60, help="Frame rate")
    parser.add_argument("-y", "--yes_to_all", action="store_true", help="Yes to all prompts")

    args = parser.parse_args()

    stage_path = args.stage_path
    if args.output_path:
        output_path = Path(args.output_path).resolve()
    else:
        sep = "/" if stage_path.startswith("omniverse://") else os.sep
        output_path = Path("./render/" + stage_path[stage_path.rfind(sep) + 1 : stage_path.rfind(".")]).resolve()

    if output_path.exists() and not args.yes_to_all:
        do_overwrite = input(f"Output directory {output_path} already exists. Overwrite? (y/n): ")
        if do_overwrite.lower() != "y":
            print("Aborting...")
            return 1
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    return do_render(
        stage_path,
        output_path,
        args.start_time,
        args.num_frames,
        args.frame_rate,
    )


if __name__ == "__main__":
    sys.exit(main())
