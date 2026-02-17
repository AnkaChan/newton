import subprocess
import sys

wedges = [
    {"name": "wedge1", "initial_jp": 0.96},
    {"name": "wedge2", "initial_jp": 0.975},
    {"name": "wedge3", "initial_jp": 0.995},
]

base_dir = "/media/andre/data/dev/newton_snow/wedges"

for exp in wedges:
    frame_dir = f"{base_dir}/{exp['name']}"
    cmd = [
        "uv", "run", "-m", "newton.examples", "mpm_granular",
        "--collider", "wedge",
        "--save-video",
        "--max-frames", "200",
        "--frame-dir", frame_dir,
        "--initial-jp", str(exp["initial_jp"]),
    ]
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp['name']} (initial_jp={exp['initial_jp']})")
    print(f"Frame dir: {frame_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: Experiment {exp['name']} exited with code {result.returncode}")
    else:
        print(f"Experiment {exp['name']} completed successfully.")

print("\nAll wedges finished.")
