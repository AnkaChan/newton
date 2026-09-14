# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test component accounting for specialized self-contact kernels."""

import unittest

from analyze_cloth_franka_self_contact import _prefix_components


class TestSelfContactComponentNames(unittest.TestCase):
    """Check names emitted by Warp for original and factory kernels."""

    def test_factory_kernels_keep_their_component(self):
        """Include both specializations in the correct self-contact total."""
        cases = {
            "accumulate_self_contact_force_and_hessian_5aec0474_cuda_kernel_forward": {"force"},
            "_create_self_contact_force_kernel__locals__accumulate_self_contact_force_and_hessian_123_cuda_kernel_forward": {
                "force"
            },
            "_create_planar_truncation_kernel__locals__apply_planar_truncation_parallel_by_collision_456_cuda_kernel_forward": {
                "planar_truncation"
            },
            "unrelated_accumulate_self_contact_force_and_hessian_123_cuda_kernel_forward": set(),
            "_accumulate_self_contact_force_and_hessian_wide_with_truncation_reset_789_cuda_kernel_forward": {"force"},
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(_prefix_components(name), expected)


if __name__ == "__main__":
    unittest.main()
