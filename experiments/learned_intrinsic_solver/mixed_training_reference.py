# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental bounded verification of one mixed-pool Adam update.

Read initial and one-update checkpoints, concatenate the saved first dispatch
from every rank, and perform one independent single-process global-mean update.
No pool callbacks, distributed collectives, or training campaign are launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np

__all__ = ["verify_first_update"]


@contextmanager
def _execution_settings(device, cpu_threads):
    import torch

    old_threads = torch.get_num_threads()
    old_precision = torch.get_float32_matmul_precision()
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    try:
        torch.set_num_threads(cpu_threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        torch.set_num_threads(old_threads)
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _first_batches(initial, after):
    if initial.get("format") != "mixed_pool_v2" or after.get("format") != "mixed_pool_v2":
        raise ValueError("expected mixed_pool_v2 checkpoints")
    if initial["report"]["completed_updates"] != 0 or after["report"]["completed_updates"] != 1:
        raise ValueError("verification requires an initial checkpoint and exactly one update")
    if initial["optimizer_state"]["state"]:
        raise ValueError("initial checkpoint must precede the first Adam update")
    config, world_size = initial["config"], initial["world_size"]
    if world_size < 1 or after["world_size"] != world_size:
        raise ValueError("initial and after checkpoints must have the same positive rank count")
    if after["config"] != config:
        raise ValueError("initial and after checkpoint configurations must match")
    if len(initial["rank_states"]) != world_size or len(after["rank_states"]) != world_size:
        raise ValueError("checkpoint must contain one state per rank")
    batch_size = config["batch_size"]
    if config["queries_per_epoch"] != batch_size * world_size:
        raise ValueError("verification requires exactly one update per epoch")
    records, specifications, rank_sizes, dispatches = [], {}, [], []
    for rank, state in enumerate(initial["rank_states"]):
        pool = state["pool"]
        selected = pool["dispatch"][:batch_size]
        if pool["batch_size"] != batch_size or len(selected) != batch_size or len(set(selected)) != batch_size:
            raise ValueError(f"rank {rank} cannot supply a full batch of distinct records")
        by_id = {record["id"]: record for record in pool["records"]}
        for identity in selected:
            if identity not in by_id:
                raise ValueError(f"rank {rank} dispatch references a missing record")
            record = by_id[identity]
            if record["inner_iteration"] != 0 or record["physical_step"] != 0:
                raise ValueError("initial dispatch must precede each trajectory's first update")
            payload = record["payload"]
            context_id = payload["context_id"]
            if context_id in specifications:
                raise ValueError("global first batch must have distinct context identifiers")
            if context_id not in state["context_specs"]:
                raise ValueError("initial dispatch has no serialized physical context")
            if "energy_initial" in payload or "energy_previous" in payload:
                raise ValueError("initial dispatch must have no optimizer energy history")
            specifications[context_id] = state["context_specs"][context_id]
            records.append(payload)
        rank_sizes.append(len(selected))
        dispatches.append(selected)
    return records, specifications, rank_sizes, dispatches


def _digest(state, names):
    return hashlib.sha256(b"".join(state[name].detach().cpu().numpy().tobytes() for name in names)).hexdigest()


def _compare(pairs, *, atol, rtol, require_relative=False):
    import torch

    tensor_count = element_count = 0
    max_absolute = max_relative = squared_error = squared_reference = 0.0
    failed, finite = [], True
    for name, reference_tensor, saved_tensor in pairs:
        if reference_tensor.shape != saved_tensor.shape or reference_tensor.dtype != saved_tensor.dtype:
            raise ValueError(f"checkpoint tensor {name!r} has a different shape or dtype")
        reference, saved = reference_tensor.detach().cpu().double(), saved_tensor.detach().cpu().double()
        tensor_count += 1
        element_count += reference.numel()
        if not torch.isfinite(reference).all() or not torch.isfinite(saved).all():
            finite = False
            failed.append(name)
            continue
        difference = (saved - reference).abs()
        max_absolute = max(max_absolute, difference.max().item())
        max_relative = max(max_relative, (difference / reference.abs().clamp_min(1e-30)).max().item())
        squared_error += difference.square().sum().item()
        squared_reference += reference.square().sum().item()
        exact = name.endswith(".step")
        if not torch.allclose(saved, reference, atol=0 if exact else atol, rtol=0 if exact else rtol):
            failed.append(name)
    relative_l2 = math.sqrt(squared_error / max(squared_reference, 1e-60))
    return {
        "passed": finite and not failed and (not require_relative or relative_l2 <= rtol),
        "tensor_count": tensor_count,
        "element_count": element_count,
        "max_abs_error": max_absolute if finite else None,
        "max_relative_error": max_relative if finite else None,
        "relative_l2_error": relative_l2 if finite else None,
        "failed_tensors": failed,
        "atol": atol,
        "rtol": rtol,
    }


def verify_first_update(
    initial: str | Path,
    after: str | Path,
    *,
    device: str = "cpu",
    parameter_atol: float = 2e-6,
    parameter_rtol: float = 2e-5,
    moment_atol: float = 1e-7,
    moment_rtol: float = 5e-4,
) -> dict:
    """Compare a saved first Adam update with one concatenated global batch.

    Experimental. The checkpoints must use the same configuration, contain
    zero and one completed updates respectively, and allocate exactly B times
    world-size queries per epoch. Inputs come from the first B dispatch records
    in each initial rank pool. Only selected physical contexts are rebuilt.

    The network and physics run on the requested CPU or CUDA device in float32.
    The reference directly constructs the detached local objective and one Adam
    update, without invoking the trainer's update loop or distributed reduction.
    Equal full rank batches make the global sample mean the expected DDP mean.
    Moment comparisons include a relative L2 bound, preventing an absolute
    tolerance from hiding a gradient-sum versus gradient-mean error.

    Args:
        initial: Initial mixed_pool_v2 checkpoint path.
        after: Checkpoint path after exactly one optimizer update.
        device: Torch CPU or CUDA device for the independent reference.
        parameter_atol: Absolute parameter comparison tolerance.
        parameter_rtol: Relative parameter comparison tolerance.
        moment_atol: Absolute Adam-moment comparison tolerance.
        moment_rtol: Relative Adam-moment comparison tolerance and relative L2 bound.

    Returns:
        JSON-safe error statistics, batch identities, material diversity and
        saved rank fingerprint agreement. ``passed`` requires all comparisons,
        exact Adam step counters, rank agreement and heterogeneous materials
        when the global batch has more than one member.
    """
    import torch

    from .data import generate_cuboid  # noqa: PLC0415 -- Keep optional execution imports local.
    from .mixed_physics import MixedHexSolverStep  # noqa: PLC0415
    from .network import IntrinsicSolverNetwork  # noqa: PLC0415

    for value in (parameter_atol, parameter_rtol, moment_atol, moment_rtol):
        if not math.isfinite(value) or value < 0:
            raise ValueError("comparison tolerances must be finite and nonnegative")
    saved_initial = torch.load(initial, map_location="cpu", weights_only=False)
    saved_after = torch.load(after, map_location="cpu", weights_only=False)
    records, specifications, rank_sizes, dispatches = _first_batches(saved_initial, saved_after)
    config = saved_initial["config"]
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("reference device must be CPU or CUDA")
    with _execution_settings(target_device, config["cpu_threads"]):
        rest = generate_cuboid(tuple(config["cell_counts"]), cell_size=config["cell_size"])
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
        network = IntrinsicSolverNetwork(
            tuple(config["cell_counts"]),
            38,
            hidden_dim=config["hidden_dim"],
            edge_hidden_dim=config["edge_hidden_dim"],
            num_heads=config["num_heads"],
            hops=tuple(config["hops"]),
            max_step_size=config["max_step_size"],
            query_chunk_size=config["query_chunk_size"],
        ).to(device=target_device, dtype=torch.float32)
        network.load_state_dict(saved_initial["network_state"])
        network.train()
        names = tuple(name for name, _ in network.named_parameters())
        initial_hash = _digest(saved_initial["network_state"], names)
        after_hash = _digest(saved_after["network_state"], names)
        initial_agreement = all(state["parameter_sha256"] == initial_hash for state in saved_initial["rank_states"])
        rank_agreement = all(state["parameter_sha256"] == after_hash for state in saved_after["rank_states"])
        step = MixedHexSolverStep(
            rest, fixed, network=network, time_step=config["time_step"], gravity=config["gravity"]
        )
        try:
            for name, specification in specifications.items():
                step.register_context(name, **specification)
            context_ids = tuple(record["context_id"] for record in records)
            candidate, inertial, prescribed = (
                torch.stack([record[name] for record in records]).to(target_device)
                for name in ("candidate", "inertial_prediction", "fixed_positions")
            )
            optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
            with torch.no_grad():
                original_energy = step.energy(candidate, inertial, context_ids).total
            with torch.autocast(device_type=target_device.type, enabled=False):
                result = step(candidate, inertial, context_ids, fixed_positions=prescribed)
                scale = original_energy.clamp_min(1.0)
                # Independently state the first-update objective. E0 and previous
                # energy coincide here, and neither has an autograd history.
                local_loss = (result.loss.total - original_energy) / scale
                local_loss += config["energy_increase_weight"] * torch.relu(result.loss.total - original_energy) / scale
                loss = local_loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in network.parameters()
            ):
                raise ValueError("reference produced missing or nonfinite network gradients")
            gradient_norm = math.sqrt(
                sum(parameter.grad.detach().double().square().sum().item() for parameter in network.parameters())
            )
            optimizer.step()
            parameters = _compare(
                (
                    (name, parameter, saved_after["network_state"][name])
                    for name, parameter in network.named_parameters()
                ),
                atol=parameter_atol,
                rtol=parameter_rtol,
            )
            reference_optimizer = optimizer.state_dict()
            actual_optimizer = saved_after["optimizer_state"]
            if reference_optimizer["param_groups"] != actual_optimizer["param_groups"]:
                raise ValueError("after checkpoint has incompatible Adam parameter groups")
            if set(reference_optimizer["state"]) != set(actual_optimizer["state"]):
                raise ValueError("after checkpoint is missing Adam parameter states")
            pairs = []
            for parameter_id, parameter_name in zip(
                reference_optimizer["param_groups"][0]["params"], names, strict=True
            ):
                expected = reference_optimizer["state"][parameter_id]
                actual = actual_optimizer["state"][parameter_id]
                if set(expected) != set(actual):
                    raise ValueError(f"after checkpoint has incompatible Adam state for {parameter_name}")
                for field in expected:
                    pairs.append((f"{parameter_name}.{field}", expected[field], actual[field]))
            moments = _compare(pairs, atol=moment_atol, rtol=moment_rtol)
            # Step counters are dimensionless O(1) values; exclude them from
            # the moment L2 ratio so they cannot conceal small scaled gradients.
            moment_only = _compare(
                [pair for pair in pairs if not pair[0].endswith(".step")],
                atol=moment_atol,
                rtol=moment_rtol,
                require_relative=True,
            )
            moments["relative_l2_error"] = moment_only["relative_l2_error"]
            moments["passed"] = moments["passed"] and moment_only["passed"]
            material_count = len(
                {
                    tuple(spec[name] for name in ("lame_lambda", "lame_mu", "density"))
                    for spec in specifications.values()
                }
            )
            heterogeneous = material_count >= min(2, len(records))
            return {
                "passed": parameters["passed"]
                and moments["passed"]
                and initial_agreement
                and rank_agreement
                and heterogeneous,
                "reference": "single_process_concatenated_global_batch",
                "device": str(target_device),
                "world_size": saved_initial["world_size"],
                "rank_batch_sizes": rank_sizes,
                "global_batch_size": len(records),
                "rank_dispatch_ids": dispatches,
                "context_ids": list(context_ids),
                "distinct_material_count": material_count,
                "heterogeneous_materials": heterogeneous,
                "initial_rank_parameter_agreement": initial_agreement,
                "rank_parameter_agreement": rank_agreement,
                "saved_parameter_sha256": after_hash,
                "rank_parameter_sha256": [state["parameter_sha256"] for state in saved_after["rank_states"]],
                "reference_loss": float(loss.detach()),
                "reference_gradient_l2": gradient_norm,
                "parameters": parameters,
                "optimizer": moments,
            }
        finally:
            step.close()


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-atol", type=float, default=2e-6)
    parser.add_argument("--parameter-rtol", type=float, default=2e-5)
    parser.add_argument("--moment-atol", type=float, default=1e-7)
    parser.add_argument("--moment-rtol", type=float, default=5e-4)
    arguments = vars(parser.parse_args())
    output = arguments.pop("output")
    try:
        report = verify_first_update(**arguments)
    except Exception as error:
        report = {"passed": False, "error": repr(error)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
