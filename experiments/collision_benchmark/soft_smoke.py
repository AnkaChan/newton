# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run numerical smoke checks for the soft-contact scene recipes."""

import argparse
import json
import time
from pathlib import Path

from .soft_scenes import SCENE_IDS, build_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="*", choices=SCENE_IDS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for scene_id in args.scenes or SCENE_IDS:
        started = time.perf_counter()
        scene = build_scene(scene_id)
        samples = [scene.metrics()]
        for frame in range(round(scene.duration / scene.frame_dt)):
            scene.step()
            if frame % max(1, round(0.25 / scene.frame_dt)) == 0:
                metrics = scene.metrics()
                samples.append(metrics)
                if not metrics["finite"]:
                    break
        samples.append(scene.metrics())
        report = {
            "scene_id": scene_id,
            "title": scene.title,
            "description": scene.description,
            "asset_ids": scene.asset_ids,
            "wall_time_s": time.perf_counter() - started,
            "all_finite": all(s["finite"] for s in samples),
            "samples": samples,
        }
        (args.out / f"{scene_id}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
        print(json.dumps(samples[-1]), flush=True)


if __name__ == "__main__":
    main()
