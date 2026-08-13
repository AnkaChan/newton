"""Capture structural VBD workload counters for an Isaac Lab task."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np


def _array_size(value) -> int:
    return 0 if value is None else int(value.shape[0])


def _model_counts(model) -> dict[str, int]:
    names = (
        "world_count",
        "body_count",
        "shape_count",
        "particle_count",
        "tri_count",
        "edge_count",
        "tet_count",
        "spring_count",
    )
    return {name: int(getattr(model, name, 0)) for name in names}


def _summary(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": int(array.max()),
    }


def run(argv: list[str]) -> None:
    import gymnasium as gym

    from isaaclab.app import launch_simulation
    from isaaclab.benchmark import stepping
    from isaaclab.benchmark.entrypoints.runtime import _parse_args
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    args, remaining = _parse_args(argv)
    env_cfg, _ = resolve_task_config(args.task, None)

    with launch_simulation(env_cfg, args):
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.device is not None:
            env_cfg.sim.device = args.device
        if args.seed is not None:
            env_cfg.seed = args.seed

        with contextlib.closing(gym.make(args.task, cfg=env_cfg)) as env:
            physics_manager = env.unwrapped.sim.physics_manager
            coupled_solver = physics_manager._solver
            if coupled_solver is None or not hasattr(coupled_solver, "solver"):
                raise RuntimeError("The selected task did not construct a coupled Newton solver")

            soft_solver = coupled_solver.solver("soft")
            collision_cfg = coupled_solver._proxy_collision_configs[("rigid", "soft")]
            pipeline = collision_cfg.pipeline
            contacts = collision_cfg.contacts
            if pipeline is None or contacts is None:
                raise RuntimeError("The rigid-to-soft proxy collision pipeline is unavailable")

            sample_count = max(1, args.warmup_steps)
            env.reset()
            contact_samples: list[dict[str, int]] = []
            for _ in range(sample_count):
                env.step(stepping.sample_random_actions(env))
                active = int(contacts.soft_contact_count.numpy()[0])
                stored = min(active, int(contacts.soft_contact_max))
                indices = contacts.soft_contact_indices.numpy()[:stored]
                populated = np.count_nonzero(indices >= 0, axis=1)
                contact_samples.append(
                    {
                        "active": active,
                        "stored": stored,
                        "particle": int(np.count_nonzero(populated == 1)),
                        "edge": int(np.count_nonzero(populated == 2)),
                        "face": int(np.count_nonzero(populated == 3)),
                    }
                )

            model = physics_manager.get_model()
            soft_model = soft_solver.model
            shape_type_values, shape_type_counts = np.unique(model.shape_type.numpy(), return_counts=True)
            color_sizes = [int(group.shape[0]) for group in soft_model.particle_color_groups]
            iterations = int(soft_solver.iterations)
            colors = len(color_sizes)
            capacity = int(contacts.soft_contact_max)
            physics_steps_per_env_step = int(env_cfg.decimation) * int(env_cfg.sim.physics.num_substeps)
            scans_per_physics_step = 1 + iterations + iterations * colors

            active_values = [sample["active"] for sample in contact_samples]
            output = {
                "task": args.task,
                "seed": args.seed,
                "num_envs": int(env.unwrapped.num_envs),
                "presets": remaining,
                "physics": {
                    "dt": float(env_cfg.sim.dt),
                    "decimation": int(env_cfg.decimation),
                    "num_substeps": int(env_cfg.sim.physics.num_substeps),
                    "physics_steps_per_environment_step": physics_steps_per_env_step,
                },
                "global_model": _model_counts(model),
                "soft_model_view": _model_counts(soft_model),
                "shape_type_histogram": {
                    str(int(shape_type)): int(count)
                    for shape_type, count in zip(shape_type_values, shape_type_counts, strict=True)
                },
                "vbd": {
                    "iterations": iterations,
                    "tile_solve": bool(soft_solver.use_particle_tile_solve),
                    "particle_color_count": colors,
                    "particle_color_sizes": color_sizes,
                },
                "soft_collision_candidates": {
                    "particle_shape_pairs": int(pipeline.soft_rigid_contact_pair_count),
                    "edge_shape_pairs": _array_size(pipeline.soft_edge_rigid_pairs),
                    "face_shape_pairs": _array_size(pipeline.soft_face_rigid_pairs),
                    "contact_capacity": capacity,
                },
                "soft_contacts_after_environment_steps": {
                    "samples": contact_samples,
                    "active_summary": _summary(active_values),
                    "capacity_utilization_at_median": float(np.median(active_values) / capacity) if capacity else 0.0,
                },
                "capacity_scan_model": {
                    "capacity_scans_per_physics_step": scans_per_physics_step,
                    "capacity_threads_per_physics_step": scans_per_physics_step * capacity,
                    "capacity_threads_per_environment_step": (
                        physics_steps_per_env_step * scans_per_physics_step * capacity
                    ),
                    "formula": "soft_contact_max * (1 init + iterations dual + iterations * colors accumulation)",
                },
            }

            output_dir = Path(args.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"workload_counters_{args.task}.json"
            output_file.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(output, indent=2))
            print(f"Wrote {output_file.resolve()}")


if __name__ == "__main__":
    run(sys.argv[1:])
