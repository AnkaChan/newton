"""Capture a deterministic-action Newton task trajectory for implementation A/B checks."""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _numpy(value) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.float32)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _policy_observation(observation) -> np.ndarray:
    if isinstance(observation, dict):
        observation = observation.get("policy", next(iter(observation.values())))
    return _numpy(observation)


def _topology_hash(model) -> str:
    digest = hashlib.sha256()
    for name in ("particle_q", "tri_indices", "edge_indices", "tet_indices", "particle_colors"):
        value = getattr(model, name, None)
        array = _numpy(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _source_hash(package_root: Path) -> str:
    """Hash Newton Python sources so dirty A/B candidates remain identifiable."""
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_head(package_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=package_root.parent,
        capture_output=True,
        check=False,
        text=True,
    )
    return completed.stdout.strip() or None


CONTACT_FIELD_NAMES = (
    "barycentric_x",
    "barycentric_y",
    "barycentric_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "body_pos_x",
    "body_pos_y",
    "body_pos_z",
    "body_vel_x",
    "body_vel_y",
    "body_vel_z",
    "penalty_k",
)


def _capture_contacts(contacts, penalty_k) -> tuple[int, np.ndarray, np.ndarray, int]:
    raw_count = int(contacts.soft_contact_count.numpy()[0])
    count = min(raw_count, int(contacts.soft_contact_max))
    if count == 0:
        return (
            raw_count,
            np.empty((0, 4), dtype=np.int32),
            np.empty((0, len(CONTACT_FIELD_NAMES)), dtype=np.float32),
            0,
        )

    corners = contacts.soft_contact_indices.numpy()[:count]
    shape = contacts.soft_contact_shape.numpy()[:count, None]
    keys = np.concatenate((shape, corners), axis=1)
    fields = np.concatenate(
        (
            contacts.soft_contact_barycentric.numpy()[:count],
            contacts.soft_contact_normal.numpy()[:count],
            contacts.soft_contact_body_pos.numpy()[:count],
            contacts.soft_contact_body_vel.numpy()[:count],
            penalty_k.numpy()[:count, None],
        ),
        axis=1,
    )

    # The collision kernels append contacts atomically. Canonicalize the records so
    # implementation changes that only alter atomic arrival order compare cleanly.
    # Float fields break ties for the unexpected case of duplicate feature keys.
    ordered_columns = [keys[:, i] for i in range(keys.shape[1])]
    ordered_columns.extend(fields[:, i] for i in range(fields.shape[1]))
    order = np.lexsort(tuple(reversed(ordered_columns)))
    keys = keys[order]
    fields = fields[order]
    duplicate_count = int(np.count_nonzero(np.all(keys[1:] == keys[:-1], axis=1)))
    return raw_count, keys, fields, duplicate_count


def run(argv: list[str]) -> None:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import newton
    from isaaclab.app import launch_simulation
    from isaaclab.benchmark.entrypoints.runtime import _parse_args
    from isaaclab_tasks.utils import resolve_task_config

    args, remaining = _parse_args(argv)
    if args.num_envs != 1:
        raise ValueError("Trajectory capture requires --num_envs 1")
    if args.seed is None:
        raise ValueError("Trajectory capture requires an explicit --seed")

    env_cfg, _ = resolve_task_config(args.task, None)
    with launch_simulation(env_cfg, args):
        env_cfg.scene.num_envs = 1
        if args.device is not None:
            env_cfg.sim.device = args.device
        env_cfg.seed = args.seed

        with contextlib.closing(gym.make(args.task, cfg=env_cfg)) as env:
            observation, _ = env.reset(seed=args.seed)
            unwrapped = env.unwrapped
            manager = unwrapped.sim.physics_manager
            model = manager.get_model()
            coupled = manager._solver
            soft_solver = coupled.solver("soft")
            collision_cfg = coupled._proxy_collision_configs[("rigid", "soft")]
            contacts = collision_cfg.contacts

            rng = np.random.default_rng(args.seed + 1729)
            action_count = unwrapped.single_action_space.shape[0]
            action_tape = rng.uniform(-1.0, 1.0, size=(args.num_steps, 1, action_count)).astype(np.float32)

            particle_q = []
            particle_qd = []
            body_q = []
            body_qd = []
            joint_q = []
            joint_qd = []
            observations = [_policy_observation(observation)]
            rewards = []
            terminated = []
            truncated = []
            contact_offsets = [0]
            contact_counts_raw = []
            contact_key_duplicate_counts = []
            contact_keys = []
            contact_fields = []

            def capture_state() -> None:
                state = manager.get_state_0()
                particle_q.append(_numpy(state.particle_q).copy())
                particle_qd.append(_numpy(state.particle_qd).copy())
                body_q.append(_numpy(state.body_q).copy())
                body_qd.append(_numpy(state.body_qd).copy())
                joint_q.append(_numpy(state.joint_q).copy())
                joint_qd.append(_numpy(state.joint_qd).copy())
                raw_count, keys, fields, duplicate_count = _capture_contacts(
                    contacts, soft_solver.body_particle_contact_penalty_k
                )
                contact_counts_raw.append(raw_count)
                contact_key_duplicate_counts.append(duplicate_count)
                contact_keys.append(keys)
                contact_fields.append(fields)
                contact_offsets.append(contact_offsets[-1] + len(keys))

            capture_state()
            for action_np in action_tape:
                action = torch.as_tensor(action_np, dtype=torch.float32, device=unwrapped.device)
                observation, reward, term, trunc, _ = env.step(action)
                observations.append(_policy_observation(observation))
                rewards.append(_numpy(reward))
                terminated.append(_numpy(term))
                truncated.append(_numpy(trunc))
                capture_state()

            output_dir = Path(args.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"trajectory_{args.task}.npz"
            np.savez_compressed(
                output_file,
                action_tape=action_tape,
                particle_q=np.stack(particle_q),
                particle_qd=np.stack(particle_qd),
                body_q=np.stack(body_q),
                body_qd=np.stack(body_qd),
                joint_q=np.stack(joint_q),
                joint_qd=np.stack(joint_qd),
                observations=np.stack(observations),
                rewards=np.stack(rewards),
                terminated=np.stack(terminated),
                truncated=np.stack(truncated),
                contact_offsets=np.asarray(contact_offsets, dtype=np.int64),
                contact_counts_raw=np.asarray(contact_counts_raw, dtype=np.int64),
                contact_key_duplicate_counts=np.asarray(contact_key_duplicate_counts, dtype=np.int64),
                contact_keys=np.concatenate(contact_keys) if contact_keys else np.empty((0, 4), dtype=np.int32),
                contact_fields=(
                    np.concatenate(contact_fields)
                    if contact_fields
                    else np.empty((0, len(CONTACT_FIELD_NAMES)), dtype=np.float32)
                ),
            )
            manifest = {
                "task": args.task,
                "seed": args.seed,
                "num_steps": args.num_steps,
                "newton_file": str(Path(newton.__file__).resolve()),
                "newton_version": newton.__version__,
                "newton_git_head": _git_head(Path(newton.__file__).resolve().parent),
                "newton_source_hash": _source_hash(Path(newton.__file__).resolve().parent),
                "topology_hash": _topology_hash(model),
                "particle_count": int(model.particle_count),
                "tri_count": int(model.tri_count),
                "edge_count": int(model.edge_count),
                "tet_count": int(model.tet_count),
                "body_count": int(model.body_count),
                "shape_count": int(model.shape_count),
                "soft_contact_max": int(contacts.soft_contact_max),
                "contact_field_names": CONTACT_FIELD_NAMES,
                "dt": float(env_cfg.sim.dt),
                "decimation": int(env_cfg.decimation),
                "num_substeps": int(env_cfg.sim.physics.num_substeps),
                "vbd_iterations": int(soft_solver.iterations),
                "vbd_tile_solve": bool(soft_solver.use_particle_tile_solve),
                "overrides": remaining,
                "output": str(output_file.resolve()),
            }
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    run(sys.argv[1:])
