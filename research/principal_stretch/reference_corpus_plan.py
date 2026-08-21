# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Freeze the bounded multi-asset reference-corpus plan.

This tool authenticates only the source meshes and writes one small JSON plan.
It does not generate initial states, run a dynamics solver, or write numeric
payloads.  Every selected mesh passes through the existing strict legacy VTK
loader under the registered 50,000-point and 50,000-tetrahedron caps.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import struct
import tempfile
from collections.abc import Mapping, Sequence

import numpy as np

from .build_hierarchy_state_preview import DEFAULT_BASE_SEED, load_legacy_vtk_tet_mesh

REFERENCE_CORPUS_PLAN_SCHEMA = "pss-reference-corpus-plan-v1"
REFERENCE_CORPUS_MAX_POINTS = 50_000
REFERENCE_CORPUS_MAX_TETS = 50_000
REFERENCE_CORPUS_BASE_SEED = DEFAULT_BASE_SEED
REFERENCE_CORPUS_SEED_CONTRACT = "pss-hierarchy-random-state-sample-seeds-v1"

_ROLE_ORDER = ("train", "validation", "confirmation")
_REQUESTED_DT_SECONDS = 1.0 / 300.0
_EXECUTION_DT_SECONDS = float(np.float32(_REQUESTED_DT_SECONDS))
_EXECUTION_DT_FLOAT32_BITS = f"0x{struct.unpack('<I', struct.pack('<f', _EXECUTION_DT_SECONDS))[0]:08x}"
_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")


def _positive_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in int")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must match {_IDENTIFIER_PATTERN.pattern!r}")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclasses.dataclass(frozen=True)
class CorpusSequenceCounts:
    """Role-specific sequence counts and the common rollout length."""

    train: int = 64
    validation: int = 16
    confirmation: int = 16
    steps_per_sequence: int = 8

    def __post_init__(self) -> None:
        for name in ("train", "validation", "confirmation", "steps_per_sequence"):
            _positive_exact_int(getattr(self, name), name)

    def for_role(self, role: str) -> int:
        """Return the sequence count for one registered role."""
        if role not in _ROLE_ORDER:
            raise ValueError(f"role must be exactly one of {_ROLE_ORDER}")
        return int(getattr(self, role))


_DEFAULT_SEQUENCE_COUNTS = CorpusSequenceCounts()


@dataclasses.dataclass(frozen=True)
class CorpusAssetSpec:
    """Expected identity, size, family, and role of one selected VTK asset."""

    asset_id: str
    source_filename: str
    source_sha256: str
    vertex_count: int
    tet_count: int
    family_id: str
    role: str

    def __post_init__(self) -> None:
        _identifier(self.asset_id, "asset_id")
        _identifier(self.family_id, "family_id")
        if type(self.source_filename) is not str or self.source_filename != f"{self.asset_id}.vtk":
            raise ValueError("source_filename must be exactly '<asset_id>.vtk'")
        if pathlib.PurePath(self.source_filename).name != self.source_filename:
            raise ValueError("source_filename must be a basename")
        _sha256(self.source_sha256, "source_sha256")
        _positive_exact_int(self.vertex_count, "vertex_count")
        _positive_exact_int(self.tet_count, "tet_count")
        if type(self.role) is not str or self.role not in _ROLE_ORDER:
            raise ValueError(f"role must be exactly one of {_ROLE_ORDER}")


def _asset(
    asset_id: str,
    source_sha256: str,
    vertex_count: int,
    tet_count: int,
    family_id: str,
    role: str,
) -> CorpusAssetSpec:
    return CorpusAssetSpec(
        asset_id=asset_id,
        source_filename=f"{asset_id}.vtk",
        source_sha256=source_sha256,
        vertex_count=vertex_count,
        tet_count=tet_count,
        family_id=family_id,
        role=role,
    )


FROZEN_CORPUS_ASSETS: tuple[CorpusAssetSpec, ...] = (
    _asset("bar", "0e91b9ca551df0c67c9452eebb8acedccc7f25bab2b450d6a5e4a75021ccc01d", 738, 1920, "bar", "train"),
    _asset("bar16660", "0a2d35e2286b4d4cd15335a8c1e32a93567fdba98282e006b6794a4d6beee1c6", 5642, 16600, "bar", "train"),
    _asset("bar4099", "29ed40a8291b1c6019ade28a3c603ec4e38c44cdc9706ed4952b280c02afe047", 1422, 4099, "bar", "train"),
    _asset("bar990", "35862004b8c5a28d4a660def5b28b50ed58d8e37cc9a3316395415403275d713", 361, 990, "bar", "train"),
    _asset(
        "cactus", "43c7dac8a6ebafc51dbdc234fa660fc6d98a3b736190a494d9b5bb2630aad7da", 5261, 17187, "cactus", "train"
    ),
    _asset("cube", "28e077af2183a29aee7987c6f4413b3154fa19352832dd2f97b7ea6c36d1e4ec", 792, 2911, "cube", "train"),
    _asset(
        "frog_reg", "ff33298a7b6c232eac6df18898bdd6945e8d0fb7c1de9d9ea907413a4f302f6e", 6834, 21909, "frog", "train"
    ),
    _asset("hippo", "630f0cba3eaac4cbb249635b2994ff9e7a7211a333356c2266e100ad7b496310", 2387, 8406, "hippo", "train"),
    _asset("longbar", "761d6c1275aa796aa1be7eca49169e72c71313a5f4bc2e0f1bcb68244de393a0", 574, 1631, "bar", "train"),
    _asset("longbar2", "e3df317e011ec5524a871030d1859cafeb0324279de910b3a604adf3d1748f3b", 3472, 10199, "bar", "train"),
    _asset("sphere", "9acd25b41ed2ef6e26857c7ba8500ba6130bdba5cedc4e94c43d25f3d045bf43", 889, 1772, "sphere", "train"),
    _asset(
        "sphere_uniform",
        "ebffbc10cc4e21abfaa9efd054801211622e4ba80f8196764e54414592f7e108",
        2340,
        10671,
        "sphere",
        "train",
    ),
    _asset(
        "squirrel_modified",
        "9a8769200f853933d57faa4cee46a9f6f4b31c679cccf539ce8e4911f7a96527",
        8395,
        23645,
        "squirrel",
        "train",
    ),
    _asset(
        "squirrel_small",
        "e21d68130fa497c868a79a1f860cfd3990d5a6d2156c810640c246e3b186cec4",
        2404,
        7262,
        "squirrel",
        "train",
    ),
    _asset(
        "squirrel_small_modified",
        "0c708ba7b29d960f49e7c57b231099d0b32ae25006c457b861171102741b6a09",
        2472,
        7561,
        "squirrel",
        "train",
    ),
    _asset("torus", "e8682cc4470eb1315dbf47285454d6de21d27b71ebb8167eb2c9531ec41b31dc", 442, 1396, "torus", "train"),
    _asset(
        "bunny", "078c14fb46e597eababd2e3b40f30a7b862ada59438b77b6574b7029ffe6d281", 6308, 26096, "bunny", "validation"
    ),
    _asset(
        "bunny_small",
        "f38a703fc9354b7c4a7c4af84f93c38af8338cec5b8329846233516c949bbe53",
        1839,
        5891,
        "bunny",
        "validation",
    ),
    _asset(
        "liver", "2c41dea644a26b333fc79f0294813e6f896e9aeb81a477883462cd55b51c017b", 10641, 40240, "liver", "validation"
    ),
    _asset(
        "ditto", "a957624aeee9bc2234e69bf262cfe2735da69d138e58b560144f51a48be55d56", 1454, 4140, "ditto", "confirmation"
    ),
    _asset(
        "super_thin_sheet",
        "8d8f896b465deaa0dac482c4aa21889d8f148f6d5be95cf6a0e1caaf65d397ca",
        13122,
        38400,
        "thin_sheet",
        "confirmation",
    ),
    _asset(
        "super_thin_sheet_2",
        "bacf7969751d4d7ea93ac25bde86f86e15b0d3d7cddc8cc8aaf50c29cc00a315",
        6562,
        19119,
        "thin_sheet",
        "confirmation",
    ),
    _asset(
        "thin_sheet",
        "f4c6ad058aff967014b94cac3ed2e7adfa62dd63e32f9a67293a5a59499c07ce",
        660,
        1932,
        "thin_sheet",
        "confirmation",
    ),
)


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_exact_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _validated_specs(asset_specs: Sequence[CorpusAssetSpec]) -> tuple[CorpusAssetSpec, ...]:
    if isinstance(asset_specs, (str, bytes)):
        raise TypeError("asset_specs must be a sequence of CorpusAssetSpec values")
    specs = tuple(asset_specs)
    if not specs:
        raise ValueError("asset_specs must contain at least one asset")
    if any(type(spec) is not CorpusAssetSpec for spec in specs):
        raise TypeError("asset_specs must contain exactly CorpusAssetSpec values")
    canonical = tuple(sorted(specs, key=lambda spec: (_ROLE_ORDER.index(spec.role), spec.asset_id)))
    if specs != canonical:
        raise ValueError("asset_specs must be in canonical order (role, asset_id)")
    if {spec.role for spec in specs} != set(_ROLE_ORDER):
        raise ValueError(f"asset_specs must populate exactly the roles {_ROLE_ORDER}")

    for name, values in (
        ("asset_id", [spec.asset_id for spec in specs]),
        ("source filename", [spec.source_filename for spec in specs]),
        ("source SHA-256", [spec.source_sha256 for spec in specs]),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {name} in asset_specs")

    family_roles: dict[str, set[str]] = {}
    for spec in specs:
        family_roles.setdefault(spec.family_id, set()).add(spec.role)
    leaked = sorted(family_id for family_id, roles in family_roles.items() if len(roles) != 1)
    if leaked:
        raise ValueError(f"family leakage across roles: {', '.join(leaked)}")
    return specs


def _protocol(counts: CorpusSequenceCounts) -> dict[str, object]:
    return {
        "requested_dt": {"numerator": 1, "denominator": 300},
        "requested_dt_seconds": _REQUESTED_DT_SECONDS,
        "execution_dt_seconds": _EXECUTION_DT_SECONDS,
        "execution_dt_float32_bits": _EXECUTION_DT_FLOAT32_BITS,
        "steps_per_sequence": counts.steps_per_sequence,
        "max_points": REFERENCE_CORPUS_MAX_POINTS,
        "max_tets": REFERENCE_CORPUS_MAX_TETS,
    }


def _seed_namespace() -> dict[str, object]:
    return {
        "contract": REFERENCE_CORPUS_SEED_CONTRACT,
        "base_seed": REFERENCE_CORPUS_BASE_SEED,
        "algorithm": "SHA-256 first four digest bytes interpreted as unsigned little-endian",
        "sample_zero_prefix": "{base_seed}:{asset_id}:{source_sha256}",
        "later_sample_prefix": "{base_seed}:{asset_id}:{source_sha256}:sample:{sample_index}",
        "digest_input": "{prefix}:{role}",
        "roles": ["deformation", "velocity"],
    }


def _asset_record(spec: CorpusAssetSpec, counts: CorpusSequenceCounts) -> dict[str, object]:
    sequence_count = counts.for_role(spec.role)
    return {
        "asset_id": spec.asset_id,
        "source_filename": spec.source_filename,
        "source_sha256": spec.source_sha256,
        "vertex_count": spec.vertex_count,
        "tet_count": spec.tet_count,
        "family_id": spec.family_id,
        "role": spec.role,
        "sequence_count": sequence_count,
        "transition_count": sequence_count * counts.steps_per_sequence,
    }


def _role_summaries(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for role in _ROLE_ORDER:
        selected = [record for record in records if record["role"] == role]
        summaries.append(
            {
                "role": role,
                "asset_count": len(selected),
                "sequence_count": sum(int(record["sequence_count"]) for record in selected),
                "transition_count": sum(int(record["transition_count"]) for record in selected),
                "vertex_count": sum(int(record["vertex_count"]) for record in selected),
                "tet_count": sum(int(record["tet_count"]) for record in selected),
            }
        )
    return summaries


def _unsigned_plan(specs: Sequence[CorpusAssetSpec], counts: CorpusSequenceCounts) -> dict[str, object]:
    records = [_asset_record(spec, counts) for spec in specs]
    return {
        "schema": REFERENCE_CORPUS_PLAN_SCHEMA,
        "protocol": _protocol(counts),
        "seed_namespace": _seed_namespace(),
        "sequence_counts": {role: counts.for_role(role) for role in _ROLE_ORDER},
        "assets": records,
        "role_summaries": _role_summaries(records),
        "totals": {
            "asset_count": len(records),
            "sequence_count": sum(int(record["sequence_count"]) for record in records),
            "transition_count": sum(int(record["transition_count"]) for record in records),
            "vertex_count": sum(int(record["vertex_count"]) for record in records),
            "tet_count": sum(int(record["tet_count"]) for record in records),
        },
    }


def validate_reference_corpus_plan(
    plan: Mapping[str, object],
    *,
    asset_specs: Sequence[CorpusAssetSpec] = FROZEN_CORPUS_ASSETS,
) -> None:
    """Validate the exact schema, split, ordering, aggregates, and self-hash.

    Args:
        plan: In-memory JSON-compatible plan.
        asset_specs: Expected ordered inventory.  The default is the frozen
            23-asset corpus; injection exists for hermetic tests.
    """
    specs = _validated_specs(asset_specs)
    if not isinstance(plan, Mapping):
        raise TypeError("plan must be a mapping")
    payload = dict(plan)
    expected_keys = {
        "schema",
        "protocol",
        "seed_namespace",
        "sequence_counts",
        "assets",
        "role_summaries",
        "totals",
        "plan_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError(f"plan keys must be exactly {tuple(sorted(expected_keys))}")
    declared_hash = _sha256(payload.pop("plan_sha256"), "plan_sha256")
    if _canonical_digest(payload) != declared_hash:
        raise ValueError("plan self-hash does not match plan_sha256")

    sequence_counts = payload.get("sequence_counts")
    if type(sequence_counts) is not dict or set(sequence_counts) != set(_ROLE_ORDER):
        raise ValueError(f"sequence_counts keys must be exactly {_ROLE_ORDER}")
    counts = CorpusSequenceCounts(
        train=sequence_counts["train"],
        validation=sequence_counts["validation"],
        confirmation=sequence_counts["confirmation"],
        steps_per_sequence=(
            payload["protocol"].get("steps_per_sequence") if type(payload.get("protocol")) is dict else None
        ),
    )

    assets = payload.get("assets")
    if type(assets) is not list:
        raise ValueError("assets must be a JSON list")
    asset_order = [record.get("asset_id") if type(record) is dict else None for record in assets]
    expected_order = [spec.asset_id for spec in specs]
    if asset_order != expected_order:
        raise ValueError("asset records must be in canonical order (role, asset_id)")

    expected = _unsigned_plan(specs, counts)
    if not _exact_json_equal(payload, expected):
        raise ValueError("plan does not exactly match the frozen corpus specification and aggregates")


def build_reference_corpus_plan(
    asset_dir: str | pathlib.Path,
    *,
    sequence_counts: CorpusSequenceCounts = _DEFAULT_SEQUENCE_COUNTS,
    asset_specs: Sequence[CorpusAssetSpec] = FROZEN_CORPUS_ASSETS,
) -> dict[str, object]:
    """Authenticate the frozen VTK inventory and build its canonical plan.

    Args:
        asset_dir: Directory containing each selected ``.vtk`` source.
        sequence_counts: Role-specific sequence counts and rollout length.
        asset_specs: Expected ordered inventory.  The default freezes the
            agreed 23-asset family-safe split; injection exists for tests.

    Returns:
        A relocation-independent JSON-compatible plan with a self-hash.
    """
    if type(sequence_counts) is not CorpusSequenceCounts:
        raise TypeError("sequence_counts must be exactly CorpusSequenceCounts")
    specs = _validated_specs(asset_specs)
    source_root = pathlib.Path(asset_dir)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    for spec in specs:
        source = source_root / spec.source_filename
        mesh = load_legacy_vtk_tet_mesh(
            source,
            max_points=REFERENCE_CORPUS_MAX_POINTS,
            max_tets=REFERENCE_CORPUS_MAX_TETS,
        )
        if mesh.source_sha256 != spec.source_sha256:
            raise ValueError(
                f"source SHA-256 drift for {spec.source_filename}: "
                f"expected {spec.source_sha256}, got {mesh.source_sha256}"
            )
        vertex_count = int(mesh.rest_positions.shape[0])
        tet_count = int(mesh.tet_indices.shape[0])
        if vertex_count != spec.vertex_count or tet_count != spec.tet_count:
            raise ValueError(
                f"source size drift for {spec.source_filename}: expected V/T "
                f"{spec.vertex_count}/{spec.tet_count}, got {vertex_count}/{tet_count}"
            )

    unsigned = _unsigned_plan(specs, sequence_counts)
    plan = dict(unsigned)
    plan["plan_sha256"] = _canonical_digest(unsigned)
    validate_reference_corpus_plan(plan, asset_specs=specs)
    return plan


def write_reference_corpus_plan(
    output: str | pathlib.Path,
    *,
    asset_dir: str | pathlib.Path,
    sequence_counts: CorpusSequenceCounts = _DEFAULT_SEQUENCE_COUNTS,
) -> dict[str, object]:
    """Atomically write one canonical plan without overwriting other bytes.

    Args:
        output: Destination JSON path.
        asset_dir: Directory containing the frozen VTK sources.
        sequence_counts: Role-specific sequence counts and rollout length.

    Returns:
        The same plan written to ``output``.
    """
    plan = build_reference_corpus_plan(
        asset_dir,
        sequence_counts=sequence_counts,
        asset_specs=FROZEN_CORPUS_ASSETS,
    )
    payload = (json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    destination = pathlib.Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return plan
        raise FileExistsError(f"refusing to overwrite non-identical corpus plan: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != payload:
                raise FileExistsError(f"refusing to overwrite non-identical corpus plan: {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)
    return plan


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=pathlib.Path, help="directory holding the frozen VTK corpus")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="canonical JSON plan destination")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    plan = write_reference_corpus_plan(args.output, asset_dir=args.asset_dir)
    totals = plan["totals"]
    print(f"Wrote {totals['asset_count']}-asset, {totals['transition_count']}-transition corpus plan to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
