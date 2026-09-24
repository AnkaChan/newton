# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental per-rank physical datasets for replayable learned-solver epochs."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .distributed_probe import collate_queries
from .train_smoke import TrainSmokeConfig, _Sampler

__all__ = ["EpochDataset"]


class _ProgressSampler(_Sampler):
    """Report the slow physical sampling phase without changing its distribution."""

    def __init__(self, config, rest, model, solver, *, rank):
        self._rank = rank
        self._generated = 0
        self._total = config.train_count + config.validation_count
        super().__init__(config, rest, model, solver)

    def _physical(self, seed, fixed):
        record = super()._physical(seed, fixed)
        self._generated += 1
        if self._generated % 64 == 0 or self._generated == self._total:
            print(f"rank={self._rank} physical_samples={self._generated}/{self._total}", flush=True)
        return record


def _rest_digest(rest):
    digest = hashlib.sha256()
    for array in (rest.corner_rest_positions, rest.cell_corner_indices):
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_create(path, payload):
    import torch

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class EpochDataset:
    """Cache one rank's physical problems and sample independent optimizer queries.

    Experimental: all seed pools are contiguous and disjoint. A rank's physical
    samples and fixed validation candidates live in an immutable ``rank_N.pt``
    file. Training candidates depend only on master seed, physical seed, and
    epoch, so changing batch size or query order does not change a candidate.
    """

    def __init__(
        self,
        config: TrainSmokeConfig,
        rest,
        model,
        solver,
        *,
        rank: int,
        world_size: int,
        dataset_dir: Path,
        resume: bool = False,
    ):
        import torch

        if isinstance(world_size, bool) or world_size not in (1, 2, 4):
            raise ValueError("world_size must be 1, 2, or 4")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if config.train_count % world_size or config.validation_count % world_size:
            raise ValueError("train_count and validation_count must be divisible by world_size")

        self.config = config
        self.rank = rank
        self.world_size = world_size
        local_train = config.train_count // world_size
        local_validation = config.validation_count // world_size
        local_config = replace(
            config,
            train_count=local_train,
            validation_count=local_validation,
            train_seed_start=config.train_seed_start + rank * local_train,
            validation_seed_start=config.validation_seed_start + rank * local_validation,
        )
        self.train_seeds = local_config.train_seeds
        self.validation_seeds = local_config.validation_seeds
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dataset_dir / f"rank_{rank}.pt"
        config_identity = asdict(config)
        for schedule_field in ("updates", "max_epochs", "early_stopping", "verbose"):
            config_identity.pop(schedule_field, None)
        identity = {
            "schema_version": 1,
            "rank": rank,
            "world_size": world_size,
            "config": config_identity,
            "rest_sha256": _rest_digest(rest),
            "train_seeds": self.train_seeds,
            "validation_seeds": self.validation_seeds,
        }
        if resume:
            if not self.path.is_file():
                raise FileNotFoundError(f"resume dataset is missing: {self.path}")
            saved = torch.load(self.path, map_location="cpu", weights_only=False)
            if saved.get("dataset_metadata", {}).get("world_size") != world_size:
                raise ValueError("resume dataset world_size differs")
            if saved.get("dataset_metadata") != identity:
                raise ValueError("resume dataset configuration or metadata differs")
            if set(saved.get("physical_samples", ())) != {*self.train_seeds, *self.validation_seeds}:
                raise ValueError("resume dataset physical seeds differ")
            if (
                tuple(entry["metadata"]["physical_seed"] for entry in saved.get("validation_candidates", ()))
                != self.validation_seeds
            ):
                raise ValueError("resume dataset validation seeds differ")
            self.sampler = _Sampler(local_config, rest, model, solver, saved=saved)
        else:
            if self.path.exists():
                raise FileExistsError(f"dataset already exists at {self.path}; use resume=True")
            self.sampler = _ProgressSampler(local_config, rest, model, solver, rank=rank)
            saved = {
                "dataset_metadata": identity,
                "sampler_state": self.sampler.state_dict(),
                "physical_samples": self.sampler.physical,
                "validation_candidates": self.sampler.saved_validation(),
            }
            _atomic_create(self.path, saved)
        self.step = solver.learned_step
        self.dataset_identity = _file_digest(self.path)

    def training_batches(self, epoch: int, batch_size: int):
        """Yield every local physical seed once in deterministic shuffled order."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        order_rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.rank, epoch, 1701]))
        order = order_rng.permutation(self.train_seeds)
        for offset in range(0, len(order), batch_size):
            queries = []
            for ordered_seed in order[offset : offset + batch_size]:
                seed = int(ordered_seed)
                rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, seed, epoch, 1709]))
                candidate, metadata = self.sampler.candidate(seed, rng)
                queries.append((self.sampler.problems[seed], candidate, metadata))
            yield collate_queries(queries)

    def validation_batches(self, batch_size: int):
        """Yield fixed held-out candidates against their original physical Y."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        entries = self.sampler.validation
        for offset in range(0, len(entries), batch_size):
            queries = [
                (self.sampler.problems[entry["metadata"]["physical_seed"]], entry["positions"], entry["metadata"])
                for entry in entries[offset : offset + batch_size]
            ]
            yield collate_queries(queries)

    def state_dict(self):
        """Return the small checkpoint link to the immutable rank dataset."""
        return {"dataset_identity": self.dataset_identity}
