# Fast cut in SolverVBD: implementation notes

Date: 2026-11-10 (PT). Written after Anka's order "go work until you get the demo ready".
The design is `design.md` in this folder. This file says what was built, how to run it, what was measured, and where the code left the design.

## 1. What exists

- Worktree: `/home/horde/Code/Graphics/newton-working-copies/vbd-fast-cut-impl`
- Branch: `ankac/vbd-fast-cut-impl`, one local commit `45df5b9e3`, not pushed. Base: the suture branch `ankac/vbd-suture-dev` at `70de8cf35` (not merged with main).
- Python: the venv of the suture worktree (Warp 1.17.0), with `PYTHONPATH` set to the new worktree.

| File | What |
|---|---|
| `newton/_src/solvers/vbd/cut_kernels.py` (new) | all cut kernels and two host helpers |
| `newton/_src/sim/builder.py` | `add_cut_blade`, `add_cut_capacity`, upload in `finalize`, guards in `color`, `set_coloring` and the builder merge |
| `newton/_src/sim/model.py` | the `cut_*` model attributes |
| `newton/_src/solvers/vbd/solver_vbd.py` | ten `cut_*` arguments, `vbd.cut_force`, `_init_cut_system`, `_update_cut_blades` (last call of `step`), `_reset_cut_system`, `cut_report` |
| `newton/_src/solvers/vbd/suture_kernels.py` | the entry search skips the idle tet; the gate update keeps the blade bit |
| `newton/_src/geometry/soft_contacts_mesh.py` | the contact gate filter is also on for a model with a blade |
| `newton/_src/viewer/viewer.py` | triangle indices are uploaded every frame when the model has cut capacity |
| `newton/examples/vbd/example_vbd_cut_suture.py` (new) | the demo |
| `newton/tests/vbd/test_cut.py` (new) | 10 tests, each for CPU and CUDA |
| `newton/tests/test_examples.py`, `docs/solvers/vbd.rst` | example test entry; section "Cutting" |

## 2. How to run

```bash
export TMPDIR=/tmp/horde-tmpdir CUDA_VISIBLE_DEVICES=1 WARP_CACHE_PATH=/tmp/horde-tmpdir/warp-cache-fastcut-impl
export PYTHONPATH=/home/horde/Code/Graphics/newton-working-copies/vbd-fast-cut-impl PYTHONDONTWRITEBYTECODE=1
PY=/home/horde/Code/Graphics/newton-working-copies/vbd-suture/.venv/bin/python
cd /home/horde/Code/Graphics/newton-working-copies/vbd-fast-cut-impl

# the demo with its final test, no window (2040 frames, about 6 minutes on an L40)
$PY -m newton.examples vbd_cut_suture --viewer null --test --quiet

# the other two arms of the advance fork (section 4)
$PY -m newton.examples vbd_cut_suture --viewer null --cut-arm wound --num-frames 400
$PY -m newton.examples vbd_cut_suture --viewer null --cut-arm design --num-frames 400

# with numbers printed every 60 frames
$PY /home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/vbd-fast-cut-design/impl/run_demo.py --every 60 --cut-arm gate --final-test

# a video (Xvfb is started by the recorder; GPU 2)
$PY /home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/vbd-fast-cut-design/demo/record_example.py \
    --module newton.examples.vbd.example_vbd_cut_suture --frames 2040 --every 2 --fps 30 \
    --camera-pos -0.075 -0.185 0.085 --camera-target 0 0 0.006 --out <folder>

# the unit tests
$PY -c "import warp as wp; wp.config.quiet = True; import unittest; unittest.main(module='newton.tests.vbd.test_cut', argv=['x', '-v'])"
```

The interactive window: the same command without `--viewer null` on a machine with a display.

## 3. Results

The demo `newton/examples/vbd/example_vbd_cut_suture.py` with the gate arm (section 4), 3 stitches, 2040 frames, graph capture, L40.
Two full runs without a viewer passed `test_final` (logs `impl/logs/d2_gate_full.log` and `impl/logs/d4_gate_full.log`; the second has the final blade body).
With the reviewed code, `python -m newton.examples vbd_cut_suture --viewer null --test --quiet` passes with three stitches (2040 frames) and with `--stitches 1 --num-frames 1260`, the setup of the example test in the suite (`impl/logs/e2_three_stitches.log`, `impl/logs/e1_ci_one_stitch.log`).

| What | Measured |
|---|---|
| Model | 2367 particles (1595 tissue, 128 spares, 644 parking), 5601 tets, 1984 triangle rows, 2976 edge rows, 8 color groups |
| Cut | 80 cut faces (20 per tet row, at about 0.55, 1.3, 2.0 and 2.8 s), 55 copies (18, 10, 12, 15 per pool), 160 triangle rows and 240 edge rows taken, no overflow, no unsafe start gap |
| After the cut | two parts of 2800 tets each; particle mass unchanged; wound gap 4.00 mm after the gripper pull |
| Blade | blade force before each row at most the cut force (0.5 N); largest particle speed in the cut phase 16 mm/s; largest blade penetration 0.008 mm |
| Suture | 6 through tunnels (2 per stitch); largest needle penetration 0.014 mm; largest row residual 0.20 mm; wound gap 1.00 mm before the release and 1.16 mm at the end |
| Speed | 171 ms per frame (10 substeps, 20 iterations); 348 s for the run |

Videos (recorded with `demo/record_example.py`, oblique camera): see section 8.

Tests:

- `newton/tests/vbd/test_cut.py`: 18 test cases (12 functions; 6 run on CPU and CUDA, 5 on CUDA only, 1 without a device), all pass (`impl/logs/t5_unittest.log`, `t6_capture.log`, `t7_cpu.log`): builder layout and guards, solver rules, capacity leaves the uncut motion as it is, gate arm (partial cut, then through cut), wound arm, the designed rule, a stepped first advance, no cut below the cut force, armed cut, reset, graph capture.
- The 70 existing suture and VBD tests (`newton/tests/vbd/test_suture_*.py`, `test_vbd_current_mesh_filter.py`) pass with the cut code in place (`impl/logs/t4_suture_tests.log`).
- `newton/tests/test_example_spec.py` accepts the new example. It fails for the base example `vbd_suture` on the suture branch itself (its docstring has no line "Success criteria:"); this was so before my change.

## 4. The fork that P0 opened: how the cut goes on below the first tet row

P0 (`checks/p0/p0_notes.md`, reviewed in `checks/p0/p0_review.md`) measured that the blade does not enter the wound.
The wound is one start gap wide (1 mm). The blade edge with the particle radius needs 1.9 mm. So the blade rests on the two wound lips, and no vertex of the cut front comes within the contact range.
The rule of the design (a vertex is pressed only within the contact range of the blade edge) then has nothing to cut.

I built three arms and ran each on the demo scene (cut phase, 400 frames; logs `impl/logs/d1_gate_cut.log`, `d3_wound_cut.log`, `d3_design_cut.log`).

| Arm | Switch | What happens | Largest blade force | Largest particle speed |
|---|---|---|---|---|
| edge (the design) | both off | first row at 0.6 s; then the blade presses the block down; later rows are cut late and unevenly (20, 22, 36, 39, 42, 44, 46, 50, 80 faces) | 5.8 N | 757 mm/s |
| wound | `cut_press_wound=True` | all 80 faces within about 0.3 s of the first force test; then the blade wedges through the open wound | 3.2 N | 118 mm/s |
| gate | `cut_gate_wound=True` | one tet row each time the blade edge reaches the cut front | 0.5 N | 16 mm/s |

- The gate arm switches off the contact between the blade body and the split vertices and copies only. Contact with uncut tissue stays. It uses the contact gate of the suture branch with one more bit (bit 30).
- The gate arm departs from D5 ("no gate") for the wound lips. The wound arm keeps "no gate", but the cut runs far ahead of the blade edge.
- The demo uses the gate arm, because it is the only arm in which the cut follows the blade edge. This is my choice for the demo. It is not a decision. It is open item O1 in `decisions.md` and question 3.8.3 in `design.md`.

## 5. Where the code left the design

The list comes from an independent comparison of the code with `design.md` (`panel/impl-review-fidelity.md`, 38 items, each checked by a second reader).

Added, not in the design:

- `cut_gate_wound`, `cut_press_wound`, the per-particle wound flag `_cut_particle_wound`, the gate bit `CUT_GATE_BIT` (section 4).
- Two more changed places in existing suture code than the one the design names: `update_suture_gate` keeps the blade bit (one argument), and the gate filter in `soft_contacts_mesh.py` is also on for a model with a blade.
- `SolverVBD.cut_report()`, a host readback of the counters.
- The blade of the example has a thin steel plate above the edge capsule (no contact, 0.8 g). One thin capsule alone made Newton's inertia validation replace the blade inertia.
- The example sets `rigid_body_particle_contact_buffer_size=1024` (the blade touched 284 contact records; the default list holds 256).
- The example option `--cut-arm`.
- Checks that the design does not have: no spring on a cuttable particle; no mesh shape in contact with particles when the model has a blade; at most 30 suture needles.

Not built (the design lists them as fork switches):

- `cut_force_measure` (force per length), `cut_force_records` (cone, front), `cut_start_gap_in_rest`, `cut_take_mode` (ordered take), `cut_skip_empty_copy_colors`, and the third value `normal_sum_first` of `cut_gap_direction`.
- The ablation runs A1 to A15, except the three arms of section 4.

Different from the design:

- Base: the suture branch as it is, not merged with main (fork F6.4 is open). The two-sided contact damping of #5182 is not in this base.
- The blade lists and the capacity values are kept in one builder dict and one model dict (`model.cut_capacity`), not in builder lists with merge metadata. A builder with a blade or with capacity cannot be merged at all.
- "Not cuttable" is a cut force at or above 1e30, not only infinity.
- The start gap direction search holds at most 16 distinct face normals per particle; more counts as unsafe.
- `reset` raises for a world mask only when the flags select the particle positions.
- Rule 6 ("margin and gap are both given") is not checked; the solver gets the resolved values.
- The holder path of the example: after the cut the blade leaves below the block toward the camera side and parks out of view. It does not go back up through the wound.
- The parking lattice of the example sits 45 mm behind and 33 mm below the block, not 0.1 m to the side.
- The needle waits above the cut plane (z = 40 mm) during the cut. Nothing touches it.
- The captured-frame test compares positions within 1e-5 m right after the first cut advance and within 1e-3 m at the end of the whole cut, not 1e-6 m.
- The example test in the suite runs one stitch (1260 frames), as the suture example test does.

## 6. Limits and open points

- One world, one blade, box and capsule tools, one particle radius, no cut after the first puncture (design limits 1, 2, 18).
- The cut follows tet faces. A stepped cut was run for its first advance only (unit test). Whether the blade goes through a stepped wound is not run.
- Interior tissue has no surface rows. Below the skin the blade meets particles only. A blade that is off the particle planes by more than the force range meets no resistance there (inference from the contact code, not run).
- The blade force of the gate arm comes from the particles and edge rows of the cut front only.
- `cut_armed=True` cuts at the first contact with any force. The substep of that first contact is not the same in every run: two plain runs of the test scene differed by 0.2 mm in the frame of the first cut and agreed to 1e-8 m twenty frames later. The force test (not armed) crosses its threshold with a slope and does not show this.
- Copies are outside the particle range that `add_soft_grid` recorded for the body.
- Not done: merge with main, a README entry with a picture, the ablation runs, a performance measurement of the cut update alone, CPU runs of the demo.
- The implementation is not pushed. It needs Anka's word.

## 7. Review of the code

Four independent readers (kernels, wiring, design fidelity, example and tests), each followed by a second reader who tried to refute the findings: `panel/impl-review-*.md`.
81 findings were confirmed: 3 major and 78 minor. No finding is a wrong result of a cut kernel.

- Major, fixed: the viewer change broke viewers whose `log_mesh` has no `dynamic` argument, for models without cut; the start check of the parking lattice against shapes was missing; the radius rule compared with the largest cuttable radius only.
- Minor, fixed: reset order and the needle gate bits, the gate bit written for both switch values, the report counters after an overflow, the idle tet in the suture point spacing, argument docs, experimental markers, changelog fragment, stronger tests, and others.
- Minor, not fixed: the items listed in section 5 as "not built" and "different".

## 8. Videos

All in `/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/vbd-fast-cut-design/demo/`, 1280 x 720, 30 frames per second, every second frame, camera at (-75, -185, 85) mm looking at (0, 0, 6) mm.

| File | What |
|---|---|
| `final_gate_oblique/example_vbd_cut_suture.mp4` | the whole demo with the gate arm: cut, pull, three stitches, pull to close, release, hold (34 s) |
| `cut_wound_oblique/example_vbd_cut_suture.mp4` | the cut phase with the wound arm (the block splits at the first threshold) |
| `cut_design_oblique/example_vbd_cut_suture.mp4` | the cut phase with the rule as designed (the blade presses on the lips) |

The two comparison clips were recorded before the review fixes; the fixes do not change how the cut behaves in this scene.
Nothing is published. The files are local.
