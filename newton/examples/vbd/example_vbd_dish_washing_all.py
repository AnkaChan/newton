# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Dish Washing H1 — Full Pile (unified AVBD/VBD)
#
# The all-dishes variant of ``example_vbd_dish_washing``: the fixed-base
# Unitree H1 works through the entire pile of rigid plates. For each plate it
# drags the flush top plate to the table edge so its rim overhangs, pinches
# it off the pile, carries it to a washing spot, rubs it with the soft sponge
# pad, then stacks it on a growing clean pile on the far side of the table.
#
# Same physics as the single-dish example (all grasps physical, water-tight
# rigid-soft SDF contact for the sponge, one SolverVBD advancing the H1 and
# plates with AVBD and the sponge with VBD). Only the counts and timing
# differ, so this script reuses that example's ``Example`` class with an
# overridden parameter set.
#
# Command: python -m newton.examples vbd_dish_washing_all
#
###########################################################################

from __future__ import annotations

import copy

import newton.examples
from newton.examples.vbd.example_vbd_dish_washing import PARAMS as _BASE_PARAMS
from newton.examples.vbd.example_vbd_dish_washing import Example as _DishWashingExample

PARAMS = copy.deepcopy(_BASE_PARAMS)
PARAMS.update(
    {
        # wash two dishes and stack them on the clean side. Both start in a row
        # on the robot's right, where the right arm grabs and (cross-body) places
        # reliably; a plate grabbed from nearer centre leaves the fixed-base arm
        # in a pose from which it can't place cross-body onto the table without
        # dropping the plate at the front-edge lip.
        "plate_count": 2,
        "wash_count": 2,
        "dirty_layout": "row",
        "dirty_pile_y": -0.24,
        "row_spacing": 0.12,
        # a full run through all three dishes is much longer
        "num_frames": 2900,
    }
)


class Example(_DishWashingExample):
    def __init__(self, viewer, args, params: dict | None = None):
        super().__init__(viewer, args, params=PARAMS if params is None else params)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
