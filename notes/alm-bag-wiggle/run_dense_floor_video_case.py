# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one bag trajectory with rho = max(rho_inertia, f*k), f dimensionless."""

import argparse
from pathlib import Path

from run_case import ROOT
from run_rho_video_case import run

FLOORS = {"off": None}
FLOORS.update(
    {f"floor{value:g}".replace(".", "p"): float(value) for value in (90, 60, 30, 20, 10, 9, 6, 3, 2, 1, 0.6, 0.3, 0.1)}
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stiffness", type=int, required=True)
    parser.add_argument("--mode", choices=list(FLOORS), required=True)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--output", type=Path, default=ROOT / "results-dense-floor-video-sweep")
    parser.set_defaults(native_floor9=False)
    arguments = parser.parse_args()
    run(arguments, material_floor=FLOORS[arguments.mode], dense_floor_sweep=True)
