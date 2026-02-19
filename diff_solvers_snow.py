import subprocess

base_cmd = [
    "uv", "run",
    "--python", "3.10",
    "--with", "tqdm",
    "--with", "imageio",
    "--with", "pyglet",
    "--extra", "sim",
    "--extra", "importers",
]

simulations = {
    "lantern_mjwarp": [
        "/media/andre/data/dev/newton/data/gtcdc25/rbd/lantern_oct23/20251023_to_sim_tdSimLanterns_03_physics.usd",
        "-o", "/media/andre/data/dev/newton/data/gtcdc25/cleanup/lantern_oct23/20251215_to_sim_tdSimLanterns_03_simmed_mjwarp_main.usd",
        "-i", "mjwarp",
        "-n", "720",
        "-t", "55.83",
        "--load_visual_shapes", "true",
        "--use_mesh_approximation", "True",
    ],
    "animated_cube": [
        "/media/andre/data/dev/newton/data/gtcdc25/debug_oct10/animated_cube_colliding_tri.usda",
        "-o", "/media/andre/data/dev/newton/data/gtcdc25/cleanup/colliding_cube.usda",
        "--integrator", "xpbd",
        "-n", "100",
        "--load_visual_shapes", "false",
        "--use_mesh_approximation", "True",
    ],
    "vase": [
        "/media/andre/data/dev/newton/data/gtcdc25/rbd/Collected_20251221_to_sim_tdSimVase_02/20251221_to_sim_tdSimVase_02_noground_physics.usd",
        "-o", "/media/andre/data/dev/newton/data/gtcdc25/cleanup/20251221_to_sim_tdSimVase_02_xpbdsimmeddec24_v4_rerun.usd",
        "--integrator", "xpbd",
        "-n", "260",
        "-t", "23.00",
    ],
    "mpm_granular": [
        "/media/andre/data/dev/newton/data/gtcdc25/mpm/Collected_20251216_to_sim_tdSimSand_01/20251216_to_sim_tdSimSand_01_physics.usd",
        "-o", "/media/andre/data/dev/newton/data/gtcdc25/mpm/feb16/snow1.usd",
        "--integrator", "cmpm",
        "-n", "600",
        "-t", "31.00",
        "-f", "/media/andre/data/dev/newton/data/gtcdc25/mpm/feb16/snow1",
    ],
    "olaf": [
        "/media/andre/data/dev/newton/data/gtcdc25/mpm/Collected_20260218_to_sim_fx_rndsnow_01/20260218_to_sim_fx_rndsnow_01_physics.usd",
        "-o", "/media/andre/data/dev/newton/data/gtcdc25/mpm/feb18/olaf0.usd",
        "--integrator", "xpbd",
        "-n", "600",
        "-t", "0.00",
        "-f", "/media/andre/data/dev/newton/data/gtcdc25/mpm/feb18/olaf0",
    ],
}

if __name__ == "__main__":
    for name, script_args in simulations.items():
        if name != "olaf":
            continue
        cmd_pr = base_cmd + ["/media/andre/data/dev/newton_snow/newton/_src/utils/sim_usd_pr.py"] + script_args
        cmd_gtc = base_cmd + ["/media/andre/data/dev/newton_snow/newton/_src/utils/sim_usd_gtc.py"] + script_args

        print(f"\n{'='*60}")
        print(f"Simulation: {name}")
        print(f"{'='*60}")

        # print(f"Running PR version ({name})...")
        # subprocess.run(cmd_pr, check=True)

        print(f"Running GTC version ({name})...")
        subprocess.run(cmd_gtc, check=True)
