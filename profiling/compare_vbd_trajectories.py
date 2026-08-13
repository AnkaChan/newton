"""Compare two fixed-action VBD task trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TOLERANCES = {
    "particle_q": (1.0e-5, 1.0e-6),
    "particle_qd": (1.0e-5, 1.0e-5),
    "body_q": (1.0e-5, 1.0e-6),
    "body_qd": (1.0e-5, 1.0e-5),
    "joint_q": (1.0e-5, 1.0e-6),
    "joint_qd": (1.0e-5, 1.0e-5),
    "observations": (1.0e-5, 1.0e-6),
    "rewards": (1.0e-5, 1.0e-6),
    "contact_fields": (1.0e-5, 1.0e-6),
}


def _load(path: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    manifest_path = path.parent / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8")), np.load(path)


def _numeric_result(name: str, baseline: np.ndarray, candidate: np.ndarray) -> dict:
    rtol, atol = TOLERANCES[name]
    shape_equal = baseline.shape == candidate.shape
    finite = bool(np.isfinite(baseline).all() and np.isfinite(candidate).all())
    if not shape_equal:
        return {"pass": False, "shape_equal": False, "baseline_shape": baseline.shape, "candidate_shape": candidate.shape}
    difference = np.abs(candidate - baseline)
    finite_difference = difference[np.isfinite(difference)]
    return {
        "pass": finite and bool(np.allclose(candidate, baseline, rtol=rtol, atol=atol)),
        "shape_equal": True,
        "finite": finite,
        "rtol": rtol,
        "atol": atol,
        "max_abs": float(finite_difference.max(initial=0.0)),
        "rms": (
            float(np.sqrt(np.mean(np.square(finite_difference), dtype=np.float64)))
            if finite_difference.size
            else 0.0
        ),
        "p99_abs": float(np.percentile(finite_difference, 99.0)) if finite_difference.size else 0.0,
    }


def _exact_result(baseline: np.ndarray, candidate: np.ndarray) -> dict:
    shape_equal = baseline.shape == candidate.shape
    equal = shape_equal and bool(np.array_equal(baseline, candidate))
    result = {
        "pass": equal,
        "baseline_shape": baseline.shape,
        "candidate_shape": candidate.shape,
    }
    if shape_equal and not equal:
        mismatches = np.argwhere(baseline != candidate)
        result["first_mismatch"] = mismatches[0].tolist()
    return result


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_manifest, baseline = _load(args.baseline)
    candidate_manifest, candidate = _load(args.candidate)
    exact_names = (
        "action_tape",
        "terminated",
        "truncated",
        "contact_offsets",
        "contact_counts_raw",
        "contact_key_duplicate_counts",
        "contact_keys",
    )
    manifest_names = (
        "task",
        "seed",
        "num_steps",
        "topology_hash",
        "particle_count",
        "tri_count",
        "edge_count",
        "tet_count",
        "body_count",
        "shape_count",
        "soft_contact_max",
        "contact_field_names",
        "dt",
        "decimation",
        "num_substeps",
        "vbd_iterations",
        "vbd_tile_solve",
        "overrides",
    )
    report = {
        "baseline": baseline_manifest,
        "candidate": candidate_manifest,
        "manifest": {
            name: {
                "pass": baseline_manifest.get(name) == candidate_manifest.get(name),
                "baseline": baseline_manifest.get(name),
                "candidate": candidate_manifest.get(name),
            }
            for name in manifest_names
        },
        "exact": {
            name: _exact_result(baseline[name], candidate[name])
            for name in exact_names
        },
        "numeric": {
            name: _numeric_result(name, baseline[name], candidate[name]) for name in TOLERANCES
        },
    }
    report["pass"] = (
        all(value["pass"] for value in report["manifest"].values())
        and all(value["pass"] for value in report["exact"].values())
        and all(value["pass"] for value in report["numeric"].values())
    )
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
