# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.vbd import example_vbd_apple_bag, example_vbd_apple_bag_franka


def _assert_selected_handle_matches_config(test_case, bag_verts, selected, params):
    if params["handle_side"] == "left":
        test_case.assertTrue(np.all(bag_verts[selected, 0] < 0.0))
    elif params["handle_side"] == "right":
        test_case.assertTrue(np.all(bag_verts[selected, 0] > 0.0))
    else:
        test_case.fail(f"Unsupported handle_side: {params['handle_side']}")


class TestVBDAppleBagFranka(unittest.TestCase):
    def test_select_handle_top_indices_uses_one_side_handle(self):
        bag_verts, _ = example_vbd_apple_bag._load_obj(example_vbd_apple_bag.BAG_OBJ)

        selected = example_vbd_apple_bag_franka._select_handle_top_indices(
            bag_verts,
            bag_start_particle=0,
            params=example_vbd_apple_bag_franka.PARAMS,
        )

        self.assertGreater(selected.shape[0], 0)
        self.assertLess(selected.shape[0], example_vbd_apple_bag.PARAMS["pin_band"] * 10000)
        _assert_selected_handle_matches_config(
            self,
            bag_verts,
            selected,
            example_vbd_apple_bag_franka.PARAMS,
        )

    def test_bag_handles_are_not_pinned(self):
        params = example_vbd_apple_bag_franka.PARAMS

        self.assertNotIn("pin_ungripped_handle", params)
        self.assertFalse(hasattr(example_vbd_apple_bag_franka.Example, "_pin_handle_particles"))
        self.assertFalse(hasattr(example_vbd_apple_bag_franka.Example, "_attach_handle_particles"))
        self.assertFalse(hasattr(example_vbd_apple_bag_franka, "move_attached_vertices"))

        builder = newton.ModelBuilder(gravity=params["gravity"])
        info = example_vbd_apple_bag_franka.build_model(builder, params, seed=params["seed"])

        self.assertNotIn("support_global_indices", info)

    def test_gripped_attachment_uses_one_side_handle_only(self):
        params = example_vbd_apple_bag_franka.PARAMS
        builder = newton.ModelBuilder(gravity=params["gravity"])
        info = example_vbd_apple_bag_franka.build_model(builder, params, seed=params["seed"])

        bag_verts, _ = example_vbd_apple_bag._load_obj(example_vbd_apple_bag.BAG_OBJ)
        selected = example_vbd_apple_bag_franka._select_handle_top_indices(
            bag_verts,
            bag_start_particle=0,
            params=params,
        )

        np.testing.assert_array_equal(info["handle_global_indices"], selected)
        _assert_selected_handle_matches_config(self, bag_verts, selected, params)

    def test_build_model_adds_static_table_support(self):
        params = example_vbd_apple_bag_franka.PARAMS

        self.assertIn("table_label", params)

        builder = newton.ModelBuilder(gravity=params["gravity"])
        info = example_vbd_apple_bag_franka.build_model(builder, params, seed=params["seed"])

        self.assertIn(params["table_label"], builder.shape_label)
        table_shape = builder.shape_label.index(params["table_label"])
        self.assertEqual(builder.shape_body[table_shape], params["table_body"])
        self.assertEqual(info["table_shape_index"], table_shape)

    def test_franka_uses_original_gripper_without_extra_pads(self):
        self.assertFalse(example_vbd_apple_bag_franka.PARAMS["add_finger_pads"])
        self.assertNotIn("finger_pad_half_width_scale", example_vbd_apple_bag_franka.PARAMS)

    def test_gripper_stays_open_for_handle_hang(self):
        params = example_vbd_apple_bag_franka.PARAMS.copy()

        self.assertFalse(params["close_gripper"])
        for frame in (0, params["lift_start_frame"], params["lift_start_frame"] + params["lift_frames"]):
            frac = example_vbd_apple_bag_franka._gripper_frac_for_frame(params, frame)
            self.assertEqual(frac, params["gripper_open_frac"])

    def test_self_contact_uses_small_radius_and_margin(self):
        params = example_vbd_apple_bag_franka.PARAMS

        self.assertTrue(params["particle_enable_self_contact"])
        self.assertEqual(params["particle_self_contact_radius"], 0.003)
        self.assertEqual(params["particle_self_contact_margin"], 0.005)

    def test_hanger_offsets_one_open_finger_into_handle(self):
        params = example_vbd_apple_bag_franka.PARAMS.copy()

        offset = example_vbd_apple_bag_franka._franka_hand_offset_for_hanger(params)

        self.assertLess(float(offset[0]), 0.0)
        self.assertAlmostEqual(abs(float(offset[1])), params["gripper_open_gap"])
        self.assertLess(float(offset[params["vertical_axis"]]), 0.0)

    def test_open_finger_wiggle_stays_inside_handle_loop(self):
        params = example_vbd_apple_bag_franka.PARAMS
        bag_verts, _ = example_vbd_apple_bag._load_obj(example_vbd_apple_bag.BAG_OBJ)
        selected = example_vbd_apple_bag_franka._select_handle_top_indices(
            bag_verts,
            bag_start_particle=0,
            params=params,
        )

        x_span = float(np.ptp(bag_verts[selected, 0]))
        y_span = float(np.ptp(bag_verts[selected, 1]))

        self.assertLessEqual(params["wiggle_amplitude"], x_span)
        self.assertLessEqual(params["wiggle_y_amplitude"], 0.5 * y_span)

    def test_tool_rotation_makes_gripper_horizontal(self):
        rotation = example_vbd_apple_bag_franka._tool_rotation_from_params(example_vbd_apple_bag_franka.PARAMS)

        finger_direction = np.array(wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, -1.0)))
        opening_direction = np.array(wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0)))

        self.assertLess(abs(float(finger_direction[2])), 1.0e-5)
        self.assertLess(abs(float(opening_direction[2])), 1.0e-5)
        self.assertLess(float(finger_direction[0]), -0.9)

    def test_franka_target_tracks_handle_wiggle(self):
        params = example_vbd_apple_bag_franka.PARAMS.copy()
        handle_center = np.array([0.1, 0.0, 0.42], dtype=np.float32)
        frame_dt = 1.0 / params["fps"]
        frame = params["lift_start_frame"] + params["lift_frames"] + round(0.5 * params["fps"])

        target = example_vbd_apple_bag_franka._franka_target_from_handle_center(
            handle_center,
            params,
            frame,
            frame_dt,
        )

        rest_target = handle_center + np.array(params["franka_grip_offset"], dtype=np.float32)
        self.assertGreater(abs(float(target[0] - rest_target[0])), 1.0e-3)
        self.assertGreater(abs(float(target[1] - rest_target[1])), 1.0e-3)

    def test_franka_target_lifts_before_wiggle(self):
        params = example_vbd_apple_bag_franka.PARAMS.copy()
        handle_center = np.array([0.1, 0.0, 0.42], dtype=np.float32)
        frame_dt = 1.0 / params["fps"]
        frame = params["lift_start_frame"] + params["lift_frames"]

        target = example_vbd_apple_bag_franka._franka_target_from_handle_center(
            handle_center,
            params,
            frame,
            frame_dt,
        )

        rest_target = handle_center + np.array(params["franka_grip_offset"], dtype=np.float32)
        self.assertGreater(float(target[params["vertical_axis"]] - rest_target[params["vertical_axis"]]), 0.05)


if __name__ == "__main__":
    unittest.main()
