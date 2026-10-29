# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual H1 adjustment code with CPU command-array doubles.

Extracting the planner method avoids constructor physics or device initialization.
"""

import ast
import copy
import json
import types
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np

TREE = ast.parse(Path(__file__).with_name("comparison_h1.py").read_text())
COMMON = next(n for n in ast.walk(TREE) if isinstance(n, ast.ClassDef) and n.name == "CommonJacket")
METHOD = next(n for n in COMMON.body if isinstance(n, ast.FunctionDef) and n.name == "_configure_grips")


class Array:
    def __init__(self, values):
        self.values = np.asarray(values).copy()

    def numpy(self):
        return self.values.copy()

    def assign(self, values):
        self.values = np.asarray(values, dtype=self.values.dtype).copy()


class Driver:
    calls: ClassVar[list] = []

    def __init__(self, model, spec, **commands):
        self.commands = {k: np.asarray(v).copy() for k, v in commands.items()}
        self.side = spec.side
        Driver.calls.append(self)


class OriginalPlanner:
    def __init__(self, report, module, *, fail=False):
        self.model = types.SimpleNamespace(collision_benchmark_contact_policy=report)
        self.state = types.SimpleNamespace(particle_q=Array([[1.0, 2.0, 3.0]]), particle_qd=Array([[4.0, 5.0, 6.0]]))
        self.grasp = [{}, {}]
        self.description = "Original"
        self.module, self.fail = module, fail

    def _configure_grips(self, q):
        for side in ("left", "right"):
            self.module.H1HandDriver(
                self.model,
                types.SimpleNamespace(side=side),
                times=[0, 1.5, 2.0, 3.0, 3.5, 6.0, 7.5, 9.5, 12.5, 14.0, 18.5, 20.0, 21.5],
                positions=[v.copy() for v in np.arange(39, dtype=float).reshape(13, 3)],
                rotations=np.array([0.0, 0.0, 0.0, 1.0]),
                controls=np.arange(78, dtype=float).reshape(13, 6),
            )
        if self.fail:
            raise ValueError("Original geometry guard rejects the prepared start")


def build(report, fail=False):
    module = types.SimpleNamespace(H1HandDriver=Driver)
    environment = {
        "np": np,
        "module": module,
        "preparation": {},
        "OriginalPlanner": OriginalPlanner,
        "_snapshot": lambda scene, drivers: {"positions": scene.state.particle_q.numpy()},
    }
    cls = ast.ClassDef(
        name="Draft",
        bases=[ast.Name(id="OriginalPlanner", ctx=ast.Load())],
        keywords=[],
        body=[copy.deepcopy(METHOD)],
        decorator_list=[],
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "<portable-draft>", "exec"),
        environment,
    )
    return environment["Draft"](report, module, fail=fail), module


def report(ratio=10.0, changed=True):
    return {
        "changed": changed,
        "policy": {"name": "elasticity-cap", "version": 1, "max_ratio": ratio, "tet_policy": "numeric"},
    }


class JacketAdjustmentTests(unittest.TestCase):
    def setUp(self):
        Driver.calls = []

    def test_authored_path_is_unchanged(self):
        """Keep authored jacket commands unchanged when no cap is selected."""
        scene, module = build({"changed": False, "policy": {"name": "authored", "version": 1}})
        scene._configure_grips(scene.state.particle_q.numpy())
        self.assertIs(module.H1HandDriver, Driver)
        self.assertFalse(hasattr(scene, "initial_open_adjustment"))
        for driver in Driver.calls:
            np.testing.assert_array_equal(driver.commands["positions"], np.arange(39).reshape(13, 3))

    def test_cap10_changes_only_right_first_position(self):
        """Move only the first right wrist keyframe under the selected cap."""
        scene, module = build(report())
        scene._configure_grips(scene.state.particle_q.numpy())
        self.assertIs(module.H1HandDriver, Driver)
        left, right = Driver.calls
        expected = np.arange(39, dtype=float).reshape(13, 3)
        np.testing.assert_array_equal(left.commands["positions"], expected)
        np.testing.assert_array_equal(right.commands["positions"][1:], expected[1:])
        np.testing.assert_array_equal(
            right.commands["positions"][0],
            expected[0] + np.asarray(scene.initial_open_adjustment["right_initial_translation_world_m"]),
        )
        expected_commands = {
            "times": [0, 1.5, 2.0, 3.0, 3.5, 6.0, 7.5, 9.5, 12.5, 14.0, 18.5, 20.0, 21.5],
            "controls": np.arange(78, dtype=float).reshape(13, 6),
            "rotations": [0.0, 0.0, 0.0, 1.0],
        }
        for driver in (left, right):
            for name, authored in expected_commands.items():
                np.testing.assert_array_equal(driver.commands[name], authored)

    def test_unchanged_contact_and_other_caps_do_not_select_correction(self):
        """Leave commands unchanged for inactive or different contact caps."""
        for policy in (report(changed=False), report(ratio=5.0)):
            Driver.calls = []
            scene, _ = build(policy)
            scene._configure_grips(scene.state.particle_q.numpy())
            self.assertFalse(hasattr(scene, "initial_open_adjustment"))
            np.testing.assert_array_equal(Driver.calls[1].commands["positions"], np.arange(39).reshape(13, 3))

    def test_original_guard_propagates_and_driver_is_restored(self):
        """Propagate the original guard and restore the driver after rejection."""
        scene, module = build(report(), fail=True)
        with self.assertRaisesRegex(ValueError, "Original geometry guard"):
            scene._configure_grips(scene.state.particle_q.numpy())
        self.assertIs(module.H1HandDriver, Driver)
        self.assertFalse(hasattr(scene, "initial_open_adjustment"))

    def test_planning_does_not_modify_material(self):
        """Preserve all particle positions and velocities during planning."""
        scene, _ = build(report())
        q, qd = scene.state.particle_q.numpy(), scene.state.particle_qd.numpy()
        scene._configure_grips(q)
        np.testing.assert_array_equal(scene.state.particle_q.numpy(), q)
        np.testing.assert_array_equal(scene.state.particle_qd.numpy(), qd)

    def test_cache_metadata_is_serializable_and_restored(self):
        """Preserve correction metadata through the actual cache helpers."""
        scene, _ = build(report())
        scene._configure_grips(scene.state.particle_q.numpy())
        fields = next(
            n.value
            for n in TREE.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_PREPARATION_FIELDS" for t in n.targets)
        )
        self.assertIn("initial_open_adjustment", ast.literal_eval(fields))
        helpers = [
            copy.deepcopy(n)
            for n in TREE.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_jsonable", "_restore_python")
        ]
        env = {"np": np, "copy": copy}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "<actual-cache-helpers>", "exec"), env)
        serialized = json.loads(json.dumps(env["_jsonable"](scene.initial_open_adjustment)))
        restored = env["_restore_python"](serialized, scene.initial_open_adjustment)
        self.assertEqual(restored, scene.initial_open_adjustment)
        # The actual common-start dictionary must publish the restored field.
        entries = [
            (k.value, v)
            for n in ast.walk(TREE)
            if isinstance(n, ast.Dict)
            for k, v in zip(n.keys, n.values, strict=True)
            if isinstance(k, ast.Constant)
        ]
        expression = next(v for k, v in entries if k == "recipe_adjustment")
        value = eval(compile(ast.Expression(expression), "<actual-common-start-field>", "eval"), {"scene": scene})
        self.assertEqual(value, restored)


class TowelDriver:
    def __init__(self, model, spec, *, times, positions, rotations, controls):
        self.times = Array(np.asarray(times, dtype=np.float32))
        self.positions = Array(np.asarray(positions, dtype=np.float32))
        self.rotations = Array(np.asarray(rotations, dtype=np.float32))
        c = np.asarray(controls)
        expanded = c[:, [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]].copy()
        expanded[:, 2] *= 1.6
        expanded[:, 3] *= 2.4
        self.fingers = Array(expanded.astype(np.float32))

    def update(self, *_):
        pass

    def geometry_audit(self, *_args, **_kwargs):
        return {"vertices_inside_robot_solids": 0, "minimum_shell_to_robot_distance_m": 0.002}


def towel_snapshot(scene, *, drivers):
    values = {"particle_q": scene.state.particle_q.numpy(), "particle_qd": scene.state.particle_qd.numpy()}
    for i, driver in enumerate(scene.drivers):
        for name in ("times", "positions", "rotations", "fingers"):
            values[f"driver{i}.{name}"] = getattr(driver, name).numpy()
    return values


def towel_scene(policy=None, bad_path=False):
    model = types.SimpleNamespace(collision_benchmark_contact_policy=policy or report())
    scene = types.SimpleNamespace(
        model=model,
        drivers=[],
        specs=[],
        grasps=[{}, {}],
        description="Original",
        radius=0.003,
        state=types.SimpleNamespace(particle_q=Array([[1.0, 2.0, 3.0]]), particle_qd=Array([[4.0, 5.0, 6.0]])),
        clock=None,
        state0=None,
        state1=None,
    )
    for side, pitch in zip(("left", "right"), (0.1583178316635836, 0.1607516612533908), strict=True):
        spec = types.SimpleNamespace(
            side=side, lower=np.full(12, -2.0), upper=np.full(12, 2.0), velocity_limits=np.full(12, 0.5)
        )
        controls = np.tile([1.25, 0.00868667, 1.0, 0.0, 0.0, 0.0], (12, 1))
        controls[1:8, 1] = pitch
        driver = TowelDriver(
            model,
            spec,
            times=np.arange(12) * 4.0,
            positions=np.arange(36).reshape(12, 3),
            rotations=np.tile([0.0, 0.0, 0.0, 1.0], (12, 1)),
            controls=controls,
        )
        # An unchanged release mimic value must retain its stored bit pattern.
        driver.fingers.values[8, 2] = np.nextafter(driver.fingers.values[8, 2], np.float32(1))
        scene.drivers.append(driver)
        scene.specs.append(spec)
    scene._audit_planned_path = lambda: {
        "minimum_conservative_table_separation_m": -0.01 if bad_path else 0.003,
        "minimum_conservative_interhand_separation_m": 0.1,
    }
    return scene


def adjust_towel(scene):
    method = next((n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_adjust_towel_pinch"), None)
    if method is None:
        raise AssertionError("Portable towel correction is absent")
    env = {"np": np, "_snapshot": towel_snapshot}
    exec(compile(ast.Module(body=[copy.deepcopy(method)], type_ignores=[]), "<actual-towel-adjustment>", "exec"), env)
    env[method.name](scene)


class TowelAdjustmentTests(unittest.TestCase):
    def test_authored_commands_unchanged(self):
        """Keep every authored towel command bit unchanged without a cap."""
        scene = towel_scene({"changed": False, "policy": {"name": "authored", "version": 1}})
        before = towel_snapshot(scene, drivers=True)
        adjust_towel(scene)
        for key, value in towel_snapshot(scene, drivers=True).items():
            np.testing.assert_array_equal(value, before[key])
        self.assertFalse(hasattr(scene, "contact_recipe_adjustment"))

    def test_only_closed_thumb_and_mimic_bits_change(self):
        """Retain open and release bits while changing closed thumb commands."""
        scene = towel_scene()
        before = towel_snapshot(scene, drivers=True)
        adjust_towel(scene)
        after = towel_snapshot(scene, drivers=True)
        self.assertEqual(
            [k for k in before if not np.array_equal(before[k], after[k])], ["driver0.fingers", "driver1.fingers"]
        )
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:8, 1:4] = True
        for i in range(2):
            np.testing.assert_array_equal(after[f"driver{i}.fingers"][~mask], before[f"driver{i}.fingers"][~mask])
        self.assertIn("contact_recipe_adjustment", Path(__file__).with_name("comparison_h1.py").read_text())

    def test_original_guard_failure_propagates(self):
        """Reject a towel correction when the original path guard fails."""
        with self.assertRaisesRegex(ValueError, "Original full-path guards"):
            adjust_towel(towel_scene(bad_path=True))

    def test_other_cap_does_not_select(self):
        """Leave towel commands unchanged for a different contact cap."""
        scene = towel_scene(report(ratio=5.0))
        before = towel_snapshot(scene, drivers=True)
        adjust_towel(scene)
        for key, value in towel_snapshot(scene, drivers=True).items():
            np.testing.assert_array_equal(value, before[key])


if __name__ == "__main__":
    unittest.main()
