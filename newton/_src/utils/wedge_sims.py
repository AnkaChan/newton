import json
import os
import subprocess
import time
from pathlib import Path

ROOT_PATH = "/media/andre/data/dev/newton/data/gtcdc25/rbd/lantern_oct23"
WEDGE_NUMBER = "wedge01"
PYTHON_PATH = "/home/andre/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh"
RENDER_SCRIPT_PATH = "/media/andre/data/dev/newton_gtc/newton/_src/utils/render_script.py"
OUTPUT_FOLDER = r"D:\Data\GTC2025DC_Demo\B1021"
start = 3

ref_config = {
    "camera_cfg": {
        "pos": (12.8, 18.74, 1.41),  # Position
        "pitch": -4.8,  # Pitch in degrees
        "yaw": -9.6,
    },
    "input_usd": r"D:\Data\GTC2025DC_Demo\Inputs\SceneB\1021\20251021_to_sim_tdSimClothB_01_physics.usd",
    "output_usd": r"test.usd",
    "initial_time": 16.0,
    "frames": 1500,
    "preroll_frames": 1000,
    "preroll_zero_velocity_ratio": 0.1,
    # "self_collision_off_frame": 1450,
    "load_preroll_state": False,
    # "load_preroll_state": True,
    "cloth_cfg": {
        "path": "/World/ClothModuleC_01/geo/clothModuleCbCollisionGeo1p12",
        # "path": "/World/ClothModuleC5kCollisionRest_01/geo/clothModuleCbCollisionRestGeo05K",
        "rest_path": "/World/ClothModuleC_01_Rest/geo/clothModuleCbCollisionRestGeo1p12",
        #   elasticity
        "tri_ke": 5e2,
        "tri_ka": 5e2,
        "tri_kd": 1e-5,
        "bending_ke": 5e-2,
        "bending_kd": 1e-7,
        "particle_radius": 0.03,
        "density": 2.0,
        "additional_translation": [0, 0, -0.05],
        # "fixed_particles" : [23100, 22959]
    },
    "additional_collider": [],
    "save_usd": True,
    "save_rest_and_init_state": True,
    "fixed_points_scheme": {
        "name": "top",
        "threshold": 0.1,
    },
    "substeps": 20,
    "iterations": 20,
    "collision_detection_interval": 10,
    "self_contact_rest_filter_radius": 0.02,
    "self_contact_radius": 0.008,
    "self_contact_margin": 0.025,
    "handle_self_contact": True,
    # "handle_self_contact": False,
    "soft_contact_ke": 3e2,
    "soft_contact_kd": 6e-3,
    "soft_contact_mu": 0.0,
}
ref_name = "B"


def run_cloth_sim(config, output_folder):
    import json
    import os

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the config as a json file in the output folder
    config_path = os.path.join(output_folder, "config.json")
    config["output_usd"] = os.path.join(
        output_folder, Path(config["input_usd"]).stem + "_" + config["wedge_number"] + ".usd"
    )

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Fill config parameter, if needed - for demonstration, let's say fill a 'run_id' if not present
    create_usd = [
        "python",
        "sim_usd_gtc.py",
        config["input_usd"],
        "-o",
        config["output_usd"],
        "-c",
        config_path,
        "-n",
        str(config["frames"]),
        "-i",
        "vbd",
        # "-t",
        # str(config["t"])
    ]
    start = time.time()
    subprocess.run(create_usd, check=False)
    end = time.time()


def run_cloth_sim_2(config, output_folder):
    import json
    import os

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the config as a json file in the output folder
    config_path = os.path.join(output_folder, "config.json")
    config["output_usd"] = os.path.join(
        output_folder, Path(config["input_usd"]).stem + "_" + config["wedge_number"] + ".usd"
    )

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Fill config parameter, if needed - for demonstration, let's say fill a 'run_id' if not present
    create_usd = [
        "python",
        "sim_usd_gtc_from_json.py",
        config["input_usd"],
        "-o",
        config["output_usd"],
        "-c",
        config_path,
        "-n",
        str(config["frames"]),
        "-i",
        "vbd",
        # "-t",
        # str(config["t"])
    ]
    start = time.time()
    subprocess.run(create_usd, check=False)
    end = time.time()


def get_wedges_configs():
    test_cases = {
        "bdx_lantern": {
            "input_usd": f"{ROOT_PATH}/20251023_to_sim_tdSimLanterns_03_physics.usd",
            "integrator": "mjwarp",
            "n": 500,
            "t": 53.33,
            "do_process": True,
            "text": "MJWarp",
            "frame_rate": 60,
        },
    }
    return test_cases


def describe_config(config, abbreviation_map):
    """
    Generate human-readable description for a config dict,
    using abbreviations found in abbreviation_map.
    """
    desc_items = []
    for k, abbr in abbreviation_map.items():
        val = config.get(k, None)
        if val is not None:
            # Try to format floats without trailing .0, with 2-3 sig figs if small
            if isinstance(val, float):
                pretty_val = f"{val:.3g}" if (abs(val) < 1e4 and abs(val) > 1e-3) else str(val)
            else:
                pretty_val = str(val)
            desc_items.append(f"{abbr}={pretty_val}")
    desc = "_".join(desc_items)
    return desc


def init_configs():
    cloth_cfg_parameters = {
        "tri_ke": [5e2],
        "tri_kd": [1e-5, 1e-6],
        "bending_ke": [
            5e-2,
            1e-1,
        ],
        "bending_kd": [1e-7],
    }

    other_parameters = {
        "soft_contact_ke": [3e2, 5e2],
        "soft_contact_kd": [6e-3, 2e-3],
    }

    # create an abbreviation map for those, keep the original wording
    abbreviation_map = {
        "tri_ke": "t_ke",
        "tri_kd": "t_kd",
        "bending_ke": "b_ke",
        "bending_kd": "b_kd",
        "soft_contact_ke": "c_ke",
        "soft_contact_kd": "c_kd",
    }

    # Generate modified configs based on ref_config and parameter sweeps from cloth_cfg_parameters and other_parameters

    import itertools
    from copy import deepcopy

    # Prepare result container for generated configs.
    generated_configs = {}

    # We'll base modifications on first (and possibly only) reference config:
    # Flatten cloth_cfg_parameters dictionary for ease of use:
    cloth_param_keys = list(cloth_cfg_parameters.keys())  # already proper keys
    cloth_param_values = [cloth_cfg_parameters[k] for k in cloth_param_keys]
    other_param_keys = list(other_parameters.keys())
    other_param_values = [other_parameters[k] for k in other_param_keys]

    # Cartesian product of all combinations
    all_param_combos = list(itertools.product(*cloth_param_values, *other_param_values))

    for idx, param_combo in enumerate(all_param_combos):
        # Compose config dict
        new_config = deepcopy(ref_config)

        # Map values back to parameters
        for i, key in enumerate(cloth_param_keys):
            if key == "tri_ke":
                new_config["cloth_cfg"]["tri_ke"] = param_combo[i]
                new_config["cloth_cfg"]["tri_ka"] = param_combo[i]
            else:
                new_config["cloth_cfg"][key] = param_combo[i]

        for j, key in enumerate(other_param_keys):
            new_config[key] = param_combo[len(cloth_param_keys) + j]

        # Set output file path to be unique by combination
        # replace keys with abbreviation only in setting name
        abbr_param_keys = [abbreviation_map.get(k, k) for k in cloth_param_keys + other_param_keys]
        settings_name = "_".join([f"{k}={v}" for k, v in zip(abbr_param_keys, param_combo, strict=False)])

        new_config["do_process"] = True
        new_config["text"] = settings_name
        new_config["wedge_number"] = "sweep_" + str(idx).zfill(3)
        generated_configs[f"{ref_name}_sweep_{idx}"] = new_config

    # Replace test_cases with all generated configs
    test_cases = generated_configs

    return test_cases


def create_usds(test_cases):
    configs = list(test_cases.values())
    for i_c in range(start, len(configs)):
        config = configs[i_c]
        if not config["do_process"]:
            continue

        run_cloth_sim(config, os.path.join(OUTPUT_FOLDER, config["wedge_number"]))


def render_usds(test_cases):
    for config in test_cases.values():
        if not config["do_process"]:
            continue

        render_usd = [
            PYTHON_PATH,
            RENDER_SCRIPT_PATH,
            config["output_usd"],  # positional argument: stage_path
            "-o",
            config["clips_basepath"],  # output path
            "-n",
            str(config["n"]),  # num_frames
            "-t",
            str(config["t"]),  # sim time
            "-f",
            str(config["frame_rate"]),  # frame rate
        ]
        subprocess.run(render_usd, check=False)


def clip_path_to_mp4_impl(clip_name, clip_dir, config, rgb_path, image_base_name, output_path):
    if not os.path.exists(output_path):
        cmd = [
            "ffmpeg",
            "-framerate",
            "60",
            "-i",
            f"{rgb_path}/{image_base_name}%04d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True)
    config["clips_plain"].append({"name": clip_name, "dir": clip_dir, "full_path": output_path})


def pngs_to_plain_clips(test_cases):
    for name, config in test_cases.items():
        image_base_name = "rgb_"

        if not config["do_process"]:
            continue

        clip_path = config["clips_basepath"]
        folders_or_pngs = os.listdir(clip_path)
        folders = [p for p in folders_or_pngs if os.path.isdir(os.path.join(clip_path, p))]

        config["clips_plain"] = []

        if not folders:
            clip_name = name
            clip_dir = clip_path
            output_path = f"{clip_path}/{clip_name}.mp4"
            clip_path_to_mp4_impl(clip_name, clip_dir, config, clip_path, image_base_name, output_path)
        else:
            for folder in folders:
                full_path = os.path.join(config["clips_basepath"], folder)
                if os.path.isdir(full_path):
                    rgb_path = os.path.join(full_path, "rgb")
                    clip_name = f"{name}_{folder}"
                    clip_dir = rgb_path
                    output_path = f"{rgb_path}/{clip_name}.mp4"
                    clip_path_to_mp4_impl(clip_name, clip_dir, config, rgb_path, image_base_name, output_path)


def clip_to_annotated_mp4_impl(input, text, dir, annotated_name, output_path, config, align):
    if align == "left":
        draw_text_config = f"drawtext=text={text}:fontcolor=white:fontsize=72:x=30:y=h-text_h-20"
    else:
        draw_text_config = f"drawtext=text={text}:fontcolor=white:fontsize=72:x=(w-text_w)/2:y=h-text_h-20"
    if not os.path.exists(output_path):
        cmd = ["ffmpeg", "-i", input, "-vf", draw_text_config, "-codec:a", "copy", output_path]
        subprocess.run(cmd, check=True)
    config["clips_annotated"].append({"name": annotated_name, "dir": dir, "full_path": output_path})


def plain_clips_to_annotated_clips(test_cases):
    for config in test_cases.values():
        if not config["do_process"]:
            continue

        config["clips_annotated"] = []

        for clip_config in config["clips_plain"]:
            annotated_name = f"{clip_config['name']}_annotated.mp4"
            output_path = f"{clip_config['dir']}/{annotated_name}"
            clip_to_annotated_mp4_impl(
                clip_config["full_path"],
                config["text"],
                clip_config["dir"],
                annotated_name,
                output_path,
                config,
                "center",
            )


def plain_clips_to_2x2_grid_impl(name, config):
    input0 = config["clips_plain"][0]["full_path"]
    input1 = config["clips_plain"][1]["full_path"]
    input2 = config["clips_plain"][2]["full_path"]
    input3 = config["clips_plain"][3]["full_path"]
    name_clip_2x2 = f"{name}_2x2_grid.mp4"
    full_path_clip_2x2 = f"{config['clips_basepath']}/{name_clip_2x2}"

    if not os.path.exists(full_path_clip_2x2):
        cmd = [
            "ffmpeg",
            "-i",
            input0,
            "-i",
            input1,
            "-i",
            input2,
            "-i",
            input3,
            "-filter_complex",
            "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            full_path_clip_2x2,
        ]
        subprocess.run(cmd, check=True)

    config["clips_2x2"] = [{"name": name_clip_2x2, "dir": config["clips_basepath"], "full_path": full_path_clip_2x2}]


def plain_clips_to_2x2_grid(test_cases):
    for name, config in test_cases.items():
        if not config["do_process"] or not len(config["clips_plain"]) == 4:
            continue

        plain_clips_to_2x2_grid_impl(name, config)


def annotate_2x2_grid(test_cases):
    for config in test_cases.values():
        if not config["do_process"] or not len(config["clips_plain"]) == 4:
            continue

        for clip_config in config["clips_2x2"]:
            annotated_name = f"{clip_config['name']}_annotated.mp4"
            output_path = f"{clip_config['dir']}/{annotated_name}"
            clip_to_annotated_mp4_impl(
                input=clip_config["full_path"],
                text=config["text"],
                dir=clip_config["dir"],
                annotated_name=annotated_name,
                output_path=output_path,
                config=config,
                align="left",
            )


def save_test_cases(test_cases, path):
    with open(path, "w") as f:
        json.dump(test_cases, f)


if __name__ == "__main__":
    test_cases = init_configs()
    create_usds(test_cases)
    # render_usds(test_cases)
    # pngs_to_plain_clips(test_cases)
    # plain_clips_to_annotated_clips(test_cases)
    # # plain_clips_to_2x2_grid(test_cases)
    # # annotate_2x2_grid(test_cases)
    # save_test_cases(test_cases, f"{ROOT_PATH}/clips_{WEDGE_NUMBER}/test_cases_{WEDGE_NUMBER}.json")
