import json
import os
from os.path import join

from wedge_rendering import *
from wedge_sims import *

if __name__ == "__main__":


    loaded_configs = {}
    # base_config_path = 'Config/config_b1_14.json'

    base_dir_new = r"D:\Data\GTC2025DC_Demo\A_1024"
    base_config_path = 'Config/config_a_1024.json'
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            config = json.load(f)
        loaded_configs[base_config_path] = config

    # config["frames"] = 10
    # config["preroll_frames"] = 10

    do_rendering = False

    base_name = "bending_sweep"

    wedge_parameter_name = "bending_ke"
    wedge_parameters = [0.03, 0.6, 0.1, 0.01,]
    videos = []
    # for new_sweep_idx, parameter in enumerate(wedge_parameters):
    for new_sweep_idx in range(0, len(wedge_parameters)):
        parameter = wedge_parameters[new_sweep_idx]
        new_sweep_name = f"bending_sweep_{wedge_parameter_name}_{str(new_sweep_idx).zfill(3)}"

        sweep_folder = os.path.join(base_dir_new, new_sweep_name)

        # config["self_collision_off_frame"] = 1450
        config[wedge_parameter_name] = parameter
        config["cloth_cfg"]["bending_ke"] = 5e-2
        # config["cloth_cfg"]["bending_kd"] = 1e-7

        config["wedge_number"] = new_sweep_name
        config["text"] = wedge_parameter_name + ": " + str(parameter)

        run_cloth_sim_2(config, sweep_folder)

        if do_rendering:
            render_temp_dir = join(base_dir_new, "rendering")
            cfg = json.load(open(os.path.join(sweep_folder, "config.json")))
            rendering_folder = os.path.join(sweep_folder, r"rendering")

            render_usd(sweep_folder, render_temp_dir, rendering_folder, cfg)
            videos.append(make_videos_for_sweep(
                rendering_folder, cfg=cfg,
                banner_lines=[wedge_parameter_name + ": " + str(parameter)]
            ))



    # run_in_batches_of_4(videos, base_dir)

    # loaded_configs now contains the configs indexed by sweep_name
