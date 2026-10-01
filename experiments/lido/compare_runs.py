# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side epoch metrics of two training runs (parity gate b of the design spec).

python -m experiments.lido.compare_runs generated/training_v4_20260928 generated/lido_parity_20261001 --epochs 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _sel(d: dict | None, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def rows(report: dict, epochs: int) -> dict:
    out = {}
    for e in report.get("epochs", [])[:epochs]:
        out[e["epoch"]] = {
            "loss": e.get("loss"),
            "updates": _sel(e, "regime", "updates"),
            "seconds": e.get("seconds"),
            "residual_n": e.get("mean_force_residual_n"),
            "cheap_metric": _sel(e, "validation", "selection", "metric"),
            "cheap_survivors": _sel(e, "validation", "physical_survivors"),
            "full_metric": _sel(e, "full_horizon_validation", "selection", "metric"),
            "full_survivors": _sel(e, "full_horizon_validation", "physical_survivors"),
            "penetration": _sel(e, "full_horizon_validation", "final_max_penetration_r", "mean"),
        }
    return out


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args(argv)
    ref = rows(json.loads((Path(args.reference) / "report.json").read_text()), args.epochs)
    cand = rows(json.loads((Path(args.candidate) / "report.json").read_text()), args.epochs)
    keys = [
        "loss",
        "updates",
        "seconds",
        "residual_n",
        "cheap_metric",
        "cheap_survivors",
        "full_metric",
        "full_survivors",
        "penetration",
    ]
    print(f"{'epoch':>5} {'metric':>16} {'reference':>12} {'candidate':>12}")
    for epoch in sorted(set(ref) | set(cand)):
        for k in keys:
            print(f"{epoch:>5} {k:>16} {fmt(ref.get(epoch, {}).get(k)):>12} {fmt(cand.get(epoch, {}).get(k)):>12}")
        print()


if __name__ == "__main__":
    main()
