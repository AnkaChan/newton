# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the frozen reference-corpus planning tool."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from ..reference_corpus_plan import (
    REFERENCE_CORPUS_BASE_SEED,
    REFERENCE_CORPUS_MAX_POINTS,
    REFERENCE_CORPUS_MAX_TETS,
    REFERENCE_CORPUS_PLAN_SCHEMA,
    CorpusAssetSpec,
    CorpusSequenceCounts,
    build_reference_corpus_plan,
    main,
    validate_reference_corpus_plan,
    write_reference_corpus_plan,
)

_ROLE_ORDER = ("train", "validation", "confirmation")


def _vtk_text(title: str, *, degenerate: bool = False) -> str:
    fourth_point = "1 1 0" if degenerate else "0 0 1"
    return f"""\
# vtk DataFile Version 3.0
{title}
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 double
0 0 0
1 0 0
0 1 0
{fourth_point}
CELLS 1 5
4 0 1 2 3
CELL_TYPES 1
10
"""


def _oversize_vtk_text(title: str) -> str:
    return f"""\
# vtk DataFile Version 3.0
{title}
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 double
0 0 0  1 0 0  0 1 0  0 0 1
CELLS 50001 250005
"""


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TestReferenceCorpusPlan(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = pathlib.Path(self.temporary_directory.name)
        self.asset_root = self.root / "assets"
        self.asset_root.mkdir()
        self.layout = (
            ("train_alpha", "train_shape", "train"),
            ("train_beta", "train_shape", "train"),
            ("validation_alpha", "validation_shape", "validation"),
            ("confirmation_alpha", "confirmation_shape", "confirmation"),
        )
        for asset_id, _, _ in self.layout:
            (self.asset_root / f"{asset_id}.vtk").write_text(_vtk_text(asset_id), encoding="ascii")
        self.specs = self._specs(self.asset_root)

    def _specs(self, root: pathlib.Path) -> tuple[CorpusAssetSpec, ...]:
        specs = []
        for asset_id, family_id, role in self.layout:
            source = root / f"{asset_id}.vtk"
            specs.append(
                CorpusAssetSpec(
                    asset_id=asset_id,
                    source_filename=source.name,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    vertex_count=4,
                    tet_count=1,
                    family_id=family_id,
                    role=role,
                )
            )
        return tuple(sorted(specs, key=lambda spec: (_ROLE_ORDER.index(spec.role), spec.asset_id)))

    def test_build_is_relocation_independent_and_has_frozen_scale(self) -> None:
        relocated = self.root / "relocated"
        relocated.mkdir()
        for asset_id, _, _ in self.layout:
            (relocated / f"{asset_id}.vtk").write_text(_vtk_text(asset_id), encoding="ascii")

        first = build_reference_corpus_plan(self.asset_root, asset_specs=self.specs)
        second = build_reference_corpus_plan(relocated, asset_specs=self.specs)

        self.assertEqual(first, second)
        self.assertEqual(first["schema"], REFERENCE_CORPUS_PLAN_SCHEMA)
        self.assertEqual(first["sequence_counts"], {"train": 64, "validation": 16, "confirmation": 16})
        self.assertEqual(
            first["protocol"],
            {
                "requested_dt": {"numerator": 1, "denominator": 300},
                "requested_dt_seconds": 1.0 / 300.0,
                "execution_dt_seconds": float(np.float32(1.0 / 300.0)),
                "execution_dt_float32_bits": "0x3b5a740e",
                "steps_per_sequence": 8,
                "max_points": REFERENCE_CORPUS_MAX_POINTS,
                "max_tets": REFERENCE_CORPUS_MAX_TETS,
            },
        )
        self.assertEqual(
            first["seed_namespace"],
            {
                "contract": "pss-hierarchy-random-state-sample-seeds-v1",
                "base_seed": REFERENCE_CORPUS_BASE_SEED,
                "algorithm": "SHA-256 first four digest bytes interpreted as unsigned little-endian",
                "sample_zero_prefix": "{base_seed}:{asset_id}:{source_sha256}",
                "later_sample_prefix": "{base_seed}:{asset_id}:{source_sha256}:sample:{sample_index}",
                "digest_input": "{prefix}:{role}",
                "roles": ["deformation", "velocity"],
            },
        )
        self.assertEqual(
            [record["asset_id"] for record in first["assets"]],
            ["train_alpha", "train_beta", "validation_alpha", "confirmation_alpha"],
        )
        self.assertEqual(
            first["role_summaries"],
            [
                {
                    "role": "train",
                    "asset_count": 2,
                    "sequence_count": 128,
                    "transition_count": 1024,
                    "vertex_count": 8,
                    "tet_count": 2,
                },
                {
                    "role": "validation",
                    "asset_count": 1,
                    "sequence_count": 16,
                    "transition_count": 128,
                    "vertex_count": 4,
                    "tet_count": 1,
                },
                {
                    "role": "confirmation",
                    "asset_count": 1,
                    "sequence_count": 16,
                    "transition_count": 128,
                    "vertex_count": 4,
                    "tet_count": 1,
                },
            ],
        )
        self.assertEqual(
            first["totals"],
            {
                "asset_count": 4,
                "sequence_count": 160,
                "transition_count": 1280,
                "vertex_count": 16,
                "tet_count": 4,
            },
        )
        unsigned = dict(first)
        self.assertEqual(unsigned.pop("plan_sha256"), _canonical_digest(unsigned))
        self.assertNotIn(str(self.root), json.dumps(first))
        self.assertIsNone(validate_reference_corpus_plan(first, asset_specs=self.specs))

    def test_role_counts_require_positive_exact_built_in_integers(self) -> None:
        self.assertEqual(
            CorpusSequenceCounts(),
            CorpusSequenceCounts(train=64, validation=16, confirmation=16, steps_per_sequence=8),
        )
        for field in ("train", "validation", "confirmation", "steps_per_sequence"):
            with self.subTest(field=field, value=True), self.assertRaisesRegex(TypeError, "built-in int"):
                CorpusSequenceCounts(**{field: True})
            with self.subTest(field=field, value=np.int64(1)), self.assertRaisesRegex(TypeError, "built-in int"):
                CorpusSequenceCounts(**{field: np.int64(1)})
            with self.subTest(field=field, value=0), self.assertRaisesRegex(ValueError, "positive"):
                CorpusSequenceCounts(**{field: 0})

    def test_missing_drift_degenerate_and_oversize_sources_fail_closed(self) -> None:
        missing_root = self.root / "missing"
        missing_root.mkdir()
        with self.assertRaises(FileNotFoundError):
            build_reference_corpus_plan(missing_root, asset_specs=self.specs)

        drift_path = self.asset_root / "train_alpha.vtk"
        drift_path.write_text(drift_path.read_text(encoding="ascii") + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "source SHA-256 drift"):
            build_reference_corpus_plan(self.asset_root, asset_specs=self.specs)
        drift_path.write_text(_vtk_text("train_alpha"), encoding="ascii")

        degenerate_root = self.root / "degenerate"
        degenerate_root.mkdir()
        for asset_id, _, _ in self.layout:
            (degenerate_root / f"{asset_id}.vtk").write_text(
                _vtk_text(asset_id, degenerate=asset_id == "train_alpha"), encoding="ascii"
            )
        degenerate_specs = self._specs(degenerate_root)
        with self.assertRaisesRegex(ValueError, "degenerate tetrahedron"):
            build_reference_corpus_plan(degenerate_root, asset_specs=degenerate_specs)

        oversize_path = self.asset_root / "train_alpha.vtk"
        oversize_path.write_text(_oversize_vtk_text("train_alpha"), encoding="ascii")
        oversize_specs = list(self._specs_for_current_sources(expected_tet_counts={"train_alpha": 50_001}))
        with self.assertRaisesRegex(ValueError, "exceeds cap 50000"):
            build_reference_corpus_plan(self.asset_root, asset_specs=oversize_specs)

    def _specs_for_current_sources(
        self, *, expected_tet_counts: dict[str, int] | None = None
    ) -> tuple[CorpusAssetSpec, ...]:
        expected_tet_counts = {} if expected_tet_counts is None else expected_tet_counts
        specs = []
        for asset_id, family_id, role in self.layout:
            source = self.asset_root / f"{asset_id}.vtk"
            specs.append(
                CorpusAssetSpec(
                    asset_id=asset_id,
                    source_filename=source.name,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    vertex_count=4,
                    tet_count=expected_tet_counts.get(asset_id, 1),
                    family_id=family_id,
                    role=role,
                )
            )
        return tuple(sorted(specs, key=lambda spec: (_ROLE_ORDER.index(spec.role), spec.asset_id)))

    def test_duplicate_hash_family_leakage_and_unordered_specs_are_rejected(self) -> None:
        duplicated_source = self.asset_root / "train_beta.vtk"
        duplicated_source.write_bytes((self.asset_root / "train_alpha.vtk").read_bytes())
        duplicate_hash_specs = self._specs_for_current_sources()
        with self.assertRaisesRegex(ValueError, "duplicate source SHA-256"):
            build_reference_corpus_plan(self.asset_root, asset_specs=duplicate_hash_specs)
        duplicated_source.write_text(_vtk_text("train_beta"), encoding="ascii")

        leaky = list(self.specs)
        validation_index = next(index for index, spec in enumerate(leaky) if spec.role == "validation")
        validation = leaky[validation_index]
        leaky[validation_index] = CorpusAssetSpec(
            asset_id=validation.asset_id,
            source_filename=validation.source_filename,
            source_sha256=validation.source_sha256,
            vertex_count=validation.vertex_count,
            tet_count=validation.tet_count,
            family_id="train_shape",
            role=validation.role,
        )
        with self.assertRaisesRegex(ValueError, "family leakage"):
            build_reference_corpus_plan(self.asset_root, asset_specs=leaky)

        unordered = (self.specs[1], self.specs[0], *self.specs[2:])
        with self.assertRaisesRegex(ValueError, "canonical order"):
            build_reference_corpus_plan(self.asset_root, asset_specs=unordered)

    def test_validation_rejects_resealed_unordered_records_and_bad_self_hash(self) -> None:
        plan = build_reference_corpus_plan(self.asset_root, asset_specs=self.specs)

        bad_hash = copy.deepcopy(plan)
        bad_hash["totals"]["transition_count"] += 1
        with self.assertRaisesRegex(ValueError, "self-hash"):
            validate_reference_corpus_plan(bad_hash, asset_specs=self.specs)

        unordered = copy.deepcopy(plan)
        unordered["assets"][0], unordered["assets"][1] = unordered["assets"][1], unordered["assets"][0]
        unsigned = dict(unordered)
        unsigned.pop("plan_sha256")
        unordered["plan_sha256"] = _canonical_digest(unsigned)
        with self.assertRaisesRegex(ValueError, "canonical order"):
            validate_reference_corpus_plan(unordered, asset_specs=self.specs)

        wrong_json_type = copy.deepcopy(plan)
        wrong_json_type["assets"][0]["tet_count"] = True
        unsigned = dict(wrong_json_type)
        unsigned.pop("plan_sha256")
        wrong_json_type["plan_sha256"] = _canonical_digest(unsigned)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_reference_corpus_plan(wrong_json_type, asset_specs=self.specs)

    def test_write_is_canonical_idempotent_and_refuses_nonidentical_overwrite(self) -> None:
        output = self.root / "plan" / "corpus.json"
        with mock.patch(
            "research.principal_stretch.reference_corpus_plan.FROZEN_CORPUS_ASSETS",
            self.specs,
        ):
            first = write_reference_corpus_plan(output, asset_dir=self.asset_root)
        first_bytes = output.read_bytes()
        self.assertEqual(
            first_bytes,
            (json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"),
        )
        with mock.patch(
            "research.principal_stretch.reference_corpus_plan.FROZEN_CORPUS_ASSETS",
            self.specs,
        ):
            self.assertEqual(write_reference_corpus_plan(output, asset_dir=self.asset_root), first)
        self.assertEqual(output.read_bytes(), first_bytes)

        output.write_text("different\n", encoding="utf-8")
        with (
            mock.patch(
                "research.principal_stretch.reference_corpus_plan.FROZEN_CORPUS_ASSETS",
                self.specs,
            ),
            self.assertRaisesRegex(FileExistsError, "non-identical"),
        ):
            write_reference_corpus_plan(output, asset_dir=self.asset_root)

    def test_cli_requires_only_asset_dir_and_output(self) -> None:
        output = self.root / "cli-plan.json"
        with mock.patch(
            "research.principal_stretch.reference_corpus_plan.FROZEN_CORPUS_ASSETS",
            self.specs,
        ):
            self.assertEqual(main(["--asset-dir", str(self.asset_root), "--output", str(output)]), 0)
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
