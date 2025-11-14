import json
import os
from os.path import join

from wedge_rendering import *
from wedge_sims import *

if __name__ == "__main__":


    loaded_configs = {}
    base_dir_new = r"D:\Data\GTC2025DC_Demo\B1021_2"
    base_config_path = 'Config/config_b1_14.json'
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            config = json.load(f)
        loaded_configs[base_config_path] = config

    do_sim = False
    do_rendering = True


    # config["frames"] = 10
    # config["preroll_frames"] = 10

    wedge_parameter_name = "air_drag"
    wedge_parameters = [0.01, 0.1, 0.5, 1.0 ]
    videos = []
    # for new_sweep_idx, parameter in enumerate(wedge_parameters):
    for new_sweep_idx in range(0, len(wedge_parameters)):
        parameter = wedge_parameters[new_sweep_idx]
        new_sweep_name = f"sweep_{wedge_parameter_name}_{str(new_sweep_idx).zfill(3)}"

        sweep_folder = os.path.join(base_dir_new, new_sweep_name)

        # config["self_collision_off_frame"] = 1450
        config[wedge_parameter_name] = parameter
        # config["cloth_cfg"]["bending_kd"] = 1e-7

        config["wedge_number"] = new_sweep_name
        config["text"] = wedge_parameter_name + ": " + str(parameter)
        if do_sim:
            run_cloth_sim_2(config, sweep_folder)

        if do_rendering:
            render_temp_dir = join(base_dir_new, "rendering")
            cfg = json.load(open(os.path.join(sweep_folder, "config.json")))
            rendering_folder = os.path.join(sweep_folder, r"rendering")

            # render_usd(sweep_folder, render_temp_dir, rendering_folder, cfg)
            # make_videos_for_sweep(
            #     rendering_folder, cfg=cfg,
            #     banner_lines=[wedge_parameter_name + ": " + str(parameter)])
            videos.append(join(rendering_folder, "rendering.mp4"))

    run_in_batches_of_4(videos, base_dir_new)

    # loaded_configs now contains the configs indexed by sweep_name
