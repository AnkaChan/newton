# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit saved paired settings and probe ALM activation through CUDA graph replay."""

import json
from pathlib import Path

import numpy as np
import warp as wp
from run_case import ROOT, Bag


def main():
    out = ROOT / "results-triangle-bending"
    saved = []
    for stiffness in (1000, 10000, 100000, 1000000, 10000000):
        off, on = [json.loads((out / f"ke{stiffness}_{mode}.json").read_text()) for mode in ("off", "on")]
        assert off["alm"] == "off" and on["alm"] == "on"
        for key in ("params", "model_sha256", "newton_revision", "solver_steps", "iterations_per_step"):
            assert off[key] == on[key], (stiffness, key)
        stats = on["alm_stats"]
        saved.append(
            {
                "stiffness": stiffness,
                "paired_settings_match": True,
                "triangle_stretch_rho_over_k": {
                    key: value / stiffness for key, value in stats["tri_rho_stretch"].items()
                },
                "triangle_area_rho_over_k": {
                    key: value / (1.2 * stiffness) for key, value in stats["tri_rho_area"].items()
                },
            }
        )

    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    trajectories = {}
    modes = {}
    for mode in ("off", "on", "on_with_state_bypassed"):
        sim = Bag(1.0e7, mode != "off")
        assert sim.model.device.is_cuda and sim.solver.use_particle_tile_solve
        state = sim.solver._particle_elasticity_alm_state
        if mode == "on_with_state_bypassed":
            # Keep the ALM tile specialization and allocated buffers, but disable
            # their use before capture. This is a diagnostic negative control.
            state.enabled = 0
        sim.capture()
        points = [sim.state_0.particle_q.numpy()]
        first_history = None
        for frame in range(60):
            sim.step()
            points.append(sim.state_0.particle_q.numpy())
            if frame == 0 and mode == "on":
                first_history = state.tri_lambda_stretch.numpy()
        trajectories[mode] = np.asarray(points)
        assert np.isfinite(trajectories[mode]).all()
        info = {"state_enabled": int(state.enabled), "tile_solve": True, "frames": 60}
        if mode == "on":
            info["triangle_history_changes_after_first_frame"] = int(
                np.count_nonzero(state.tri_lambda_stretch.numpy() != first_history)
            )
            rows = (
                ("triangle_stretch", state.tri_rho_stretch.numpy(), sim.model.tri_materials.numpy()[:, 0]),
                ("triangle_area", state.tri_rho_area.numpy(), sim.model.tri_materials.numpy()[:, :2].sum(axis=1)),
                (
                    "bending",
                    state.bend_rho.numpy(),
                    sim.model.edge_bending_properties.numpy()[:, 0] * sim.model.edge_rest_length.numpy(),
                ),
            )
            for name, rho, stiffness in rows:
                active = (rho > 0) & (stiffness > 0)
                ratio = rho[active].astype(np.float64) / stiffness[active].astype(np.float64)
                info[name] = {
                    "active_rows": int(active.sum()),
                    "rho_over_k_min": float(ratio.min()),
                    "rho_over_k_max": float(ratio.max()),
                    "fraction_at_floor": float(np.mean(np.isclose(ratio, 9.0))),
                    "effective_row_stiffness_fraction_min": float(np.min(ratio / (1 + ratio))),
                }
        modes[mode] = info
        print(mode, json.dumps(info), flush=True)

    differences = {}
    for mode in ("on", "on_with_state_bypassed"):
        np.testing.assert_array_equal(trajectories["off"][0], trajectories[mode][0])
        delta = trajectories[mode].astype(np.float64) - trajectories["off"].astype(np.float64)
        differences[mode] = {
            "particle_coordinate_rms_difference_m": float(np.sqrt(np.mean(delta**2))),
            "particle_coordinate_max_difference_m": float(np.max(np.abs(delta))),
        }
    result = {
        "saved_runs": saved,
        "probe": {"stiffness": 1.0e7, "frames": 60, "substeps": 10, "iterations": 10, "modes": modes},
        "differences_from_off": differences,
        "scope": "Activation control on the first settling second; not a convergence or performance benchmark.",
    }
    path: Path = out / "activation-audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(differences, indent=2), flush=True)


if __name__ == "__main__":
    main()
