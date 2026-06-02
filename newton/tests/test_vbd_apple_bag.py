# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton.examples.vbd import example_vbd_apple_bag


class TestVBDAppleBagMotion(unittest.TestCase):
    def test_wiggle_offset_includes_y_motion_after_settle(self):
        example = object.__new__(example_vbd_apple_bag.Example)
        example.params = example_vbd_apple_bag.PARAMS.copy()
        example.frame_dt = 1.0 / example.params["fps"]
        example.frame = example.params["settle_frames"] + round(0.5 * example.params["fps"])

        offset = example._wiggle_offset()

        self.assertGreater(abs(float(offset[1])), 1.0e-3)


if __name__ == "__main__":
    unittest.main()
