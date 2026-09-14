# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Embed exact source excerpts from the stage-1 commit into the HTML walkthrough."""

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "246fb405f6658528dc594b34780dbc14fce804a2"
VBD = "newton/_src/solvers/vbd/"
SOLVER = VBD + "solver_vbd.py"
STATE = VBD + "particle_alm_kernels.py"
PRIMAL = VBD + "particle_vbd_kernels.py"
RIGID = VBD + "rigid_vbd_kernels.py"
BEHAVIOR = "newton/tests/test_solver_vbd_alm.py"
UNIT = "newton/tests/test_particle_alm_kernels.py"
BENCHMARK = "notes/stage1_elasticity_alm_benchmark.py"


def source(path):
    return subprocess.check_output(["git", "show", f"{SNAPSHOT}:{path}"], cwd=ROOT, text=True)


def function_ranges(path, names):
    tree = ast.parse(source(path))
    result = []
    for name in names:
        matches = [
            node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {name} in {path}; found {len(matches)}")
        node = matches[0]
        start = min([node.lineno, *[decorator.lineno for decorator in node.decorator_list]])
        result.append((start, node.end_lineno))
    return result


def excerpt(path, label, *, names=(), ranges=()):
    lines = source(path).splitlines()
    selected = [*ranges, *function_ranges(path, names)]
    entries = []
    for start, end in selected:
        if entries:
            entries.append({"number": None, "text": "# Separate excerpt from the same file."})
        entries.extend({"number": index + 1, "text": lines[index]} for index in range(start - 1, end))
    return {
        "path": path,
        "label": label,
        "ranges": "; ".join(f"{start}-{end}" for start, end in selected),
        "lines": entries,
    }


def main():
    excerpts = {
        "constructor": excerpt(SOLVER, "Constructor setup", ranges=[(333, 342), (420, 442), (785, 794), (819, 826)]),
        "tile-select": excerpt(SOLVER, "Tile specialization", ranges=[(919, 940)]),
        "step": excerpt(SOLVER, "Simulation step", names=["step"]),
        "initialize": excerpt(SOLVER, "Particle initialization", names=["_initialize_particles"]),
        "iteration": excerpt(SOLVER, "Particle color sweep", names=["_solve_particle_iteration"]),
        "finalize": excerpt(SOLVER, "Particle velocity finalization", names=["_finalize_particles"]),
        "struct": excerpt(STATE, "Persistent state", names=["ParticleElasticityAlmState"]),
        "create": excerpt(STATE, "Validate and allocate history", names=["create_particle_elasticity_alm_state"]),
        "prepare": excerpt(STATE, "Preparation dispatch", names=["prepare_particle_elasticity_alm"]),
        "prepare-tets": excerpt(STATE, "Tet preparation", names=["_prepare_tets"]),
        "prepare-lines": excerpt(STATE, "Spring and hinge preparation", names=["_prepare_springs", "_prepare_bends"]),
        "algebra": excerpt(
            RIGID,
            "Stable ALM algebra reused by particles",
            names=["_compliant_alm_coefficients", "_alm_relaxed_ascent"],
        ),
        "tet-primal": excerpt(
            PRIMAL, "Tet stress, force, and Hessian", names=["evaluate_volumetric_neo_hookean_force_and_hessian_alm"]
        ),
        "spring-primal": excerpt(
            PRIMAL,
            "Spring evaluation and accumulation",
            names=["evaluate_spring_force_and_hessian_both_vertices_alm", "accumulate_spring_force_and_hessian"],
        ),
        "hinge-primal": excerpt(
            PRIMAL, "Hinge force and Hessian", names=["evaluate_dihedral_angle_based_bending_force_hessian_alm"]
        ),
        "scalar": excerpt(PRIMAL, "Scalar vertex solve", names=["solve_elasticity"]),
        "tile": excerpt(PRIMAL, "Generated tiled vertex solve", names=["make_solve_elasticity_tile"]),
        "dual-order": excerpt(SOLVER, "Placement outside the color loop", ranges=[(3506, 3515)]),
        "update": excerpt(
            STATE,
            "Dual dispatch and element updates",
            names=["update_particle_elasticity_alm", "_update_tets", "_update_springs", "_update_bends"],
        ),
        "reset-solver": excerpt(SOLVER, "Solver reset integration", ranges=[(2554, 2635)]),
        "reset-kernel": excerpt(STATE, "History reset", names=["_reset_history", "reset_particle_elasticity_alm"]),
        "restart": excerpt(SOLVER, "Coupling restart guard", ranges=[(1353, 1367)]),
        "rotation-probe": excerpt(BENCHMARK, "Rotation diagnostic", names=["rotating_history"]),
        "capture-test": excerpt(BEHAVIOR, "Captured tet steps and reset", names=["_captured_steps_and_reset"]),
        "test-state": excerpt(
            UNIT,
            "Seed and metric tests",
            names=["test_rest_seed_and_stress_metrics", "test_skew_tet_metrics_use_inverse_rows"],
        ),
        "test-load": excerpt(BEHAVIOR, "Loaded equilibrium tests", names=["_loaded_tet", "_loaded_spring"]),
        "test-reset": excerpt(BEHAVIOR, "Selected-world reset behavior", names=["_reset_selected_world"]),
        "test-tile": excerpt(BEHAVIOR, "Scalar and tile parity", names=["_tile_matches_scalar"]),
        "test-primal": excerpt(
            UNIT,
            "Primal curvature and high-stiffness tests",
            names=["test_tet_primal_high_stiffness", "test_spring_primal_curvature"],
        ),
        "test-dat": excerpt(BEHAVIOR, "ALM with existing DAT", names=["_legacy_dat_with_alm"]),
    }
    path = ROOT / "notes/alm-code-walkthrough/index.html"
    html = path.read_text()
    payload = json.dumps(excerpts, ensure_ascii=False).replace("<", "\\u003c")
    html, count = re.subn(
        r'(<script id="source-data" type="application/json">).*?(</script>)',
        lambda match: match[1] + payload + match[2],
        html,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Expected one source-data script")
    requested = set(re.findall(r'data-source="([^"]+)"', html))
    if missing := requested - excerpts.keys():
        raise ValueError(f"Missing source excerpts: {missing}")
    path.write_text(html)
    print(f"Embedded {len(excerpts)} excerpts from {SNAPSHOT[:8]} in {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
