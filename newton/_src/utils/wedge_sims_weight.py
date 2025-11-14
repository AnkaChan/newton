import json
import os
from os.path import join

from wedge_rendering import *
from wedge_sims import *

if __name__ == "__main__":


    loaded_configs = {}
    base_dir_new = r"D:\Data\GTC2025DC_Demo\B1024\density_new_sim"
    base_config_path = 'Config/config_b_1024.json'

    wedge_base_name = "sweep_weight"

    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            config = json.load(f)
        loaded_configs[base_config_path] = config

    do_sim = True
    # do_sim = False
    do_rendering = True
    # do_rendering = False


    # config["frames"] = 10
    # config["preroll_frames"] = 10

    wedge_parameter_name = ["cloth_cfg", "density"]
    wedge_parameters = [ 5.0, 3.0, 2.0, 1.0, ]
    # config["cloth_cfg"]["bending_kd"] = 0.01
    config["body_friction_mu"] = 0.1

    videos = []
    # for new_sweep_idx, parameter in enumerate(wedge_parameters):
    for new_sweep_idx in range(0, 4):
        parameter = wedge_parameters[new_sweep_idx]
        new_sweep_name = wedge_base_name + f"_{str(new_sweep_idx).zfill(3)}"

        sweep_folder = os.path.join(base_dir_new, new_sweep_name)

        # config["self_collision_off_frame"] = 1450
        # config["cloth_cfg"]["bending_kd"] = 1e-7
        if  isinstance(wedge_parameter_name, list):
            config[wedge_parameter_name[0]][wedge_parameter_name[1]] = parameter
            config["text"] = wedge_parameter_name[1] + ": " + str(parameter)
        else:
            config[wedge_parameter_name] = parameter
            config["text"] = wedge_parameter_name + ": " + str(parameter)

        config["wedge_number"] = new_sweep_name
        if do_sim:
            run_cloth_sim_2(config, sweep_folder)

        rendering_folder = os.path.join(sweep_folder, r"rendering")
        if do_rendering:
            render_temp_dir = join(base_dir_new, "rendering")
            cfg = json.load(open(os.path.join(sweep_folder, "config.json")))

            render_usd(sweep_folder,
                       render_temp_dir,
                       rendering_folder,
                       sim_project_name="full_20251024_to_sim_tdSimClothB_01.usd",
                       dst_filename="20251024_to_sim_tdSimClothB_01_physics_sim_v.usd", )
            make_videos_for_sweep(
                rendering_folder, cfg=cfg,
                banner_lines=[config["text"]])
        videos.append(join(rendering_folder, "rendering.mp4"))

    run_in_batches_of_4(videos, base_dir_new)

    # loaded_configs now contains the configs indexed by sweep_name
