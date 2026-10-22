# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression checks for compiler scope, inputs and finite stopping."""

# ruff: noqa: PLC0415
# Import patched solver modules only within the scope check that uses them.

import json
import unittest

import warp as wp

from . import run_fast_math as adapter


class ScopeTests(unittest.TestCase):
    def test_particle_free_constructor_does_not_require_an_elasticity_tile(self):
        """Verify particle free constructor does not require an elasticity tile."""
        import newton

        wp.init()
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body=body, hx=0.05, hy=0.05, hz=0.05)
        builder.color()
        model = builder.finalize(device="cpu")
        scope = adapter.CompilerScope()
        for enabled in (False, True):
            scope.set_math(enabled)
            before = scope.snapshot()
            solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=False)
            scope.finish_constructor(solver, before)
            self.assertEqual(model.particle_count, 0)
            self.assertEqual(wp.get_module_options(scope.rigid_solve.module)["fast_math"], enabled)
            self.assertEqual(wp.get_module_options(scope.rigid_duals.module)["fast_math"], enabled)

    def test_auxiliary_observer_error_preserves_explicit_error_record(self):
        """Verify auxiliary observer error preserves explicit error record."""

        class BrokenObserver:
            def result(self):
                raise AttributeError("missing field")

        result = adapter.task_result(BrokenObserver())
        self.assertEqual(result["status"], "observer_error")
        self.assertIn("AttributeError", result["error"])
        self.assertNotIn("pass", result)

    def test_both_arms_use_same_kernel_bodies_and_only_pr_modules_change(self):
        """Verify both arms use same kernel bodies and only pr modules change."""
        from newton._src.solvers.vbd import (
            particle_alm_kernels,
            rigid_vbd_kernels,
            solver_vbd,
            vbd_coupling_kernels,
        )

        original_solve = solver_vbd.solve_rigid_body
        original_duals = solver_vbd.update_duals_joint
        scope = adapter.CompilerScope()
        json.dumps(scope.snapshot(), allow_nan=False)
        self.assertIs(scope.rigid_solve.func, original_solve.func)
        self.assertIs(scope.rigid_duals.func, original_duals.func)
        self.assertIsNot(scope.rigid_solve.module, original_solve.module)
        self.assertIsNot(scope.rigid_duals.module, scope.rigid_solve.module)
        tile = solver_vbd.make_solve_elasticity_tile(True, False, True, include_alm=True)
        modules = (scope.rigid_solve.module, scope.rigid_duals.module, tile.module)
        untouched = {
            m.__name__: wp.get_module_options(m).copy()
            for m in (rigid_vbd_kernels, particle_alm_kernels, vbd_coupling_kernels)
        }
        for enabled in (True, False):
            scope.set_math(enabled)
            self.assertEqual(
                wp.get_module_options(solver_vbd.particle_vbd_kernels)["fast_math"],
                enabled,
            )
            for module in modules:
                self.assertEqual(wp.get_module_options(module)["fast_math"], enabled)
                self.assertFalse(wp.get_module_options(module)["enable_backward"])
            self.assertIs(solver_vbd.solve_rigid_body.module, modules[0])
            self.assertIs(solver_vbd.update_duals_joint.module, modules[1])
            for m in (rigid_vbd_kernels, particle_alm_kernels, vbd_coupling_kernels):
                self.assertEqual(wp.get_module_options(m), untouched[m.__name__])

    def test_fingerprint_mismatch_fails_before_treatment(self):
        """Verify fingerprint mismatch fails before treatment."""
        adapter.require_fingerprint(
            {"sha256": "same", "fields": {"state": "a"}},
            {"sha256": "same", "fields": {"state": "a"}},
        )
        with self.assertRaisesRegex(AssertionError, "fingerprint"):
            adapter.require_fingerprint(
                {"sha256": "same", "fields": {"state": "a"}},
                {"sha256": "same", "fields": {"state": "b"}},
            )


if __name__ == "__main__":
    unittest.main()
