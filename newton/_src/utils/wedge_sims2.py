from wedge_sims import *
from wedge_rendering import *

from os.path import join, exists, dirname, abspath

import os
import json


if __name__ == '__main__':
    good_numbers = [4,6,8,9,10,14]

    loaded_configs = {}
    base_dir_new = r"D:\Data\GTC2025DC_Demo\B1021_2"

    for new_sweep_idx, sweep_idx in enumerate(good_numbers):
        sweep_name = f"sweep_{str(sweep_idx).zfill(3)}"
        config_path = os.path.join(OUTPUT_FOLDER, sweep_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            loaded_configs[sweep_name] = config
            print(f"Loaded {sweep_name} from {config_path}")
        else:
            print(f"Config not found for {sweep_name} at {config_path}")

        new_sweep_name = f"sweep_batch2_{str(new_sweep_idx).zfill(3)}"

        sweep_folder = os.path.join(base_dir_new, new_sweep_name)


        abbreviation_map = {
            "tri_ke": "t_ke",
            "tri_kd": "t_kd",
            "bending_ke": "b_ke",
            "bending_kd": "b_kd",
            "soft_contact_ke": "c_ke",
            "soft_contact_kd": "c_kd",
            "air_drag": "adrag",
        }

        config["self_collision_off_frame"] = 1450
        config["air_drag"] = 0.1
        config["cloth_cfg"]["bending_kd"] = 1e-8

        config["wedge_number"] = new_sweep_name
        config["text"] = describe_config(config, abbreviation_map)

        run_cloth_sim(config, sweep_folder)
        render_temp_dir = join(base_dir_new, "rendering")
        cfg = json.load(open(os.path.join(sweep_folder, "config.json")))
        rendering_folder = os.path.join(sweep_folder, r"rendering")
        render_usd(sweep_folder, render_temp_dir, rendering_folder, cfg)


    # loaded_configs now contains the configs indexed by sweep_name