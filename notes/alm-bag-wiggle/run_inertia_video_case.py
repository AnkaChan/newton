# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one bag trajectory with native rho = coefficient * rho_inertia and no floor."""

import argparse
from pathlib import Path

from run_case import ROOT
from run_rho_video_case import run

SCALES = {"off": 1.0, "inertia1": 1.0, "inertia10": 10.0, "inertia100": 100.0, "inertia1000": 1000.0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stiffness", type=int, required=True)
    parser.add_argument("--mode", choices=list(SCALES), required=True)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--output", type=Path, default=ROOT / "results-inertia-video-sweep")
    parser.set_defaults(native_floor9=False)
    arguments = parser.parse_args()
    run(arguments, native_rho_scale=SCALES[arguments.mode])
