# P0 wedge probe: review

Review of `p0_wedge_probe.py`, `p0_log.txt`, `p0_q1.csv`, `p0_q2.csv` and `p0_notes.md`. Date: 2026-11-10 (PT).

## 1. Verdict

| Question | Verdict | Reason in one line |
|---|---|---|
| Q1 (uncut block) | Trust | The scene follows the spec. Four independent force numbers agree within 0.02 N. Two more runs give the same numbers within 0.006 N. |
| Q2 (V wound) | Trust with limits | The main result holds in five runs: the blade rests on the two lips and no cut-front vertex comes closer than 3.29 mm. The limits: the first 20 frames are not a clean measurement, the travel at 0.5 N depends on how the force is read, and the class (i) share is a label, not a location. |

I found no setup error in the scene and no error in the force computation. I did not need a corrected copy of the script.

## 2. What I did

- Read the script, the log, both CSV files, the notes and the two earlier runs in `earlier_runs/`.
- Read main (commit 2134db9cd): the `SolverVBD` constructor, the rigid-soft force law, the contact record layout, the self-contact law and filter, the D6 drive, `add_soft_grid`.
- Run A: the unmodified script, both scenes, output to `review_rerun/` (168 s).
- Run B: `review_checks.py`, both scenes (168 s). It imports the `Probe` class of the script. The scene and all solver values are the same. It only adds read-outs: static scene checks, the force after every substep in a window of frames, the tet elastic forces on the host, the x force balance on the lips.
- GPU 1 only. About 6 minutes in total, with a 12 s test. Nothing installed. No Newton file changed. I set `PYTHONDONTWRITEBYTECODE=1`.

## 3. The scene against the spec

| Item | Spec | As built | How I checked | Result |
|---|---|---|---|---|
| Code base | main, 2134db9cd, GPU 1 | same; Warp 1.19.0, L40 | log head; `git status` clean | ok |
| Block | 12 x 10 x 4 cells of 3 mm, parity rule of `add_soft_grid` | 715 particles, 2400 tets | Run B: tets equal to a fresh `add_soft_grid` call, same order, same corner order; skin triangles equal as sets; tet volumes sum to 12960 mm^3 | ok |
| Tissue | density 1000, E 1e5, nu 0.45, no damping, radius 0.5 mm | k_mu 34482.8, k_lambda 310344.8, damping 0, radius 0.5 mm | `model.tet_materials`, log | ok |
| Split (Q2) | 11 top vertices on x = 18 get copies; tets with centroid x > 18 use them; cut front shared | 11 copies; 40 tets use a copy, all with centroid x > 18; 40 tets use an original, all with x < 18; no tet uses both; 40 wound triangles, 20 per side; cut front used on both sides | Run B | ok |
| Rest shape | zero gap | `tet_poses` equal the inverse edge matrices of the zero-gap shape (4.9e-7 relative) | log line "rest check"; copies and originals have equal rest positions | ok |
| Start gap | originals -0.5 mm, copies +0.5 mm, positions only | exactly -0.5000 and +0.5000 mm; no other particle moved; state equals model | Run B | ok |
| Rest-shape filter | off | radius 0.0 stored in the solver | solver attribute | ok |
| Hold | x <= 6 or x >= 30 mm fixed | 330 particles, mass 0 and inverse mass 0, exactly this set | Run B | ok |
| Gravity | off | (0, 0, 0) | model | ok |
| Blade | capsule r 0.45, half height 17, axis y, centered, density 7800 | scale (0.45, 17) mm, axis (0, 1, 0), center (18, 15, 13.45) mm, mass 0.1717 g | Run B | ok |
| Holder and joint | body without shape; D6 with six driven axes; ke 1e4, kd 1e2, 10, 0.1 | holder inverse mass 0; D6 from holder to blade; gains as given; target 0 | Run B, log | ok |
| Holder motion | once per frame, -z at 2 mm/s, then 0.5 s stop | 0.0333 mm per frame; deviation from the schedule 3e-6 mm; 30 frames of stop | CSV | ok |
| Pipeline | full surface on, soft_contact_gap 1 mm, owned by the solver | as given; other arguments default | Run B | ok |
| Solver | DAT on; self contact on, margin 1 mm, gap 1 mm; filter threshold 1; exclusion radius 0; parallel epsilon 1e-7; 20 iterations | every value is stored in the solver as given; no name was ignored | Run B, constructor at `solver_vbd.py:439-500` | ok |
| Contact values | ke 5e2, kd 5, mu 0.3, set as the suture example | model 500 / 5 / 0.3; shape 500 / 5 / 0.3; per-record stiffness 500 N/m in every frame | Run B, CSV | ok |
| Time | 60 fps, 10 substeps | frame 1/60 s, substep 1/600 s | Run B | ok |

The deviations that the notes list (section 3, items 1 to 9) are complete for the physics. Four small points are not in that list:

1. The probe wrote Python cache files (`__pycache__`, time stamp of its first run) into the Newton worktree. They are ignored by git. No tracked file changed.
2. The probe ran three full runs and one short test. The task allowed one probe. The notes say this openly.
3. The `time_s` column is one frame ahead of the holder schedule (time = (frame + 1) / 60, holder position from frame / 60). All results are given against holder travel, so no result changes.
4. The suture example sets `contact_matching="latest"` in some settings. The probe uses the default ("disabled"). The spec does not ask for it.

## 4. The force computation

- Law. Main uses `f = k * (radius - distance)` for a rigid-soft record with the log barrier off (`rigid_vbd_kernels.py:2111-2127`, called from `_eval_soft_ef_contact`, lines 2266-2374). The host copy in the script is the same.
- Contact point. Sum of weight times particle position over the corners. Same as main (lines 2305-2315).
- Collision radius. Largest particle radius of the corners plus the shape margin: 0.5 + 0 mm. Same as main (lines 2307-2336).
- Distance and sign. `dot(normal, point - body point)`, body point moved to world space with the blade pose. Same as main. The normal points from the blade to the tissue: every normal has a negative z part, and the signed sum equals the positive-part sum in every frame.
- Stiffness. Read from `solver.body_particle_contact_penalty_k`, which main indexes by the record index. It is 500 N/m in every frame (the mean of 500 and 500).
- Record selection. The script uses main's own function `_soft_contact_force_enabled`. It disabled no blade record.
- Classes. (i): all corners are lip vertices. (ii): the corner with the largest weight is a cut-front vertex. (iii): the rest. This follows the task text. See finding 3 for what the class (i) number means.

Cross checks of the vertical force on the blade:

| Frame | Host normal force along d [N] | Minus drive force [N] | Solver observable CONTACT_F, z [N] | Gripper reaction from tet elasticity [N] |
|---|---|---|---|---|
| Q1, 29 | 0.525 | 0.537 | 0.537 | 0.530 |
| Q1, 60 | 1.735 | 1.747 | 1.746 | 1.740 |
| Q1, 120 | 4.146 | 4.158 | 4.157 | 4.153 |
| Q1, 240 (stop) | 7.861 | 7.861 | 7.861 | 7.861 |
| Q2, 35 | 1.026 | 1.041 | 1.040 | 1.032 |
| Q2, 77 | 2.719 | 2.736 | 2.734 | 2.727 |
| Q2, 134 | 5.001 | 5.013 | 5.012 | 5.008 |
| Q2, 164 (stop) | 4.991 | 5.002 | 5.001 | 5.001 |

All values are from Run B. The first three columns are the probe's own read-outs. The last column is mine and uses no contact data. I copied main's stable Neo-Hookean tet law to numpy, computed the elastic force on every particle, and summed it over the 330 held particles. On the free particles away from the wound the sum of elastic force and blade force is below 0.001 N per particle, so the copy of the law is right and the state is quasi-static.

All four numbers agree within 0.02 N. The host number is the smallest while the holder moves, because it has no damping and no friction. At the stop all four agree within 0.01 N.

## 5. Sanity checks

- Zero before contact. Q1: exactly 0 N in frames 0 to 14. First force at holder travel 0.50 to 0.53 mm, where the axis is 0.95 mm above the top row. This is the force range of the design.
- Record counts. Q1: 11 particle records for the 11 vertices under the edge; 58 edge and 48 face records are the skin edges and triangles that touch these vertices. Q2: 22 particle records for the 22 lip vertices. Counts are the same in all runs.
- Start opening. 1.000 mm in `model.particle_q` and in the first state.
- Bound of one record. 500 N/m x 0.5 mm = 0.25 N. Largest record: 0.123 N (Q1), 0.040 N (Q2).
- Tissue stiffness. Q1 gives 1.33 N per mm of blade travel (frames 30 to 60). A hand estimate for a beam clamped at both ends (span 24 mm, 30 x 12 mm section, bending and shear) gives 1.3 N/mm. The size is right.
- Blade against holder. The lag equals force / 1e4 N/m in every quiet frame (0.786 mm at 7.86 N, 0.500 mm at 5.00 N). The blade does not tilt. Its x offset stays below 0.006 mm.
- Speed. Q1: at most 0.65 mm/s at a frame end. Q2: see finding 1.
- Inside a frame (Run B, quiet frames). The force along d grows from the first to the last substep by 0.03 to 0.04 N. It does not overshoot. The frame-end value is the largest value of the frame.
- No non-finite value in any run.

Repeat runs:

| | Probe run 3 (delivered) | Run A | Run B |
|---|---|---|---|
| Q1, force along d, largest difference to the delivered CSV | 0 | 0.005 N | 0.006 N |
| Q1, 0.5 N reached | frame 29 | frame 29 | frame 29 |
| Q2, largest difference to the delivered CSV from frame 24 on | 0 | 0.014 N | 0.012 N |
| Q2, first frame end with 0.5 N | frame 22 (0.505 N) | frame 22 (0.501 N) | frame 23 (0.495 N at frame 22) |
| Q2, smallest distance axis to cut front | 3.291 mm | 3.291 mm | 3.290 mm |
| Q2, class (ii) | 0 N | 0 N | 0 N |

## 6. Findings

No finding changes the main result. Findings 1 to 3 limit Q2.

1. Q2, frames 0 to 20: the wound shakes, and this is not a short start effect. In Run B the largest particle speed is between 27 and 505 mm/s after every substep of frames 3 to 20. The mean opening moves between 0.87 and 1.32 mm. In Run A it is the same (speeds up to 305 mm/s at frame ends until frame 19). The notes (6.4) describe a wound that comes to rest by frame 7 and shakes again at the first touch. That happened in probe runs 1 and 2 only, and there the second shaking also lasted until frame 19. In Run A and Run B the wound never came to rest before frame 20, where the blade load is about 0.4 N. In probe run 3 the shaking ended at frame 11. So the last frame with a particle speed of 5 mm/s or more is frame 11, 19, 19, 19 or 20, depending on the run. No Q2 number before frame 21 can be used. The cross-check numbers of the notes for these frames (0.10 N, 0.54 N) are also run dependent: Run A gives up to 1.4 N.
2. Q2, travel at 0.5 N: the number of the notes (0.73 mm, "the same frame in all three runs") is a frame-end number and it sits one or two frames after the shaking ends. In Run B the value at frame 22 is 0.495 N, so the frame-end crossing is at frame 23 (0.767 mm). After every substep (the design tests every substep) Run B crosses 0.5 N already inside frame 17, at holder travel 0.567 mm (0.531 N and 0.540 N in two substeps, then back to 0.06 N at the end of frame 18). So the travel at 0.5 N is 0.73 to 0.77 mm for a frame-end reading and can be 0.57 mm for a per-substep reading. The notes list "the force inside a frame is not read" as a limit but give no size.
3. Q2, class (i) share (23.0 % at 5.0 N): this number is right for the rule of the task, but it does not say where the force acts. All force acts at the 22 lip vertices: in Run B the blade normal force on the lip vertices is 4.998 N along z of 5.001 N at frame 134. A record of a top-face edge or triangle that touches a lip vertex has a corner outside the wound, so the rule puts it in class (iii). If "wound vertices" also means the cut-front vertices, class (i) is 60.7 %. The notes give both splits, so the data is complete. The plain answer is: lips 100 %, cut front 0 %.
4. Q2, why the lips do not open: the notes give this as an inference (6.3). Run B confirms it with numbers, see section 8. This closes a gap in the notes. It is not an error.
5. Q1 and Q2, the slope of force against travel belongs to the hold of the task (free span of 24 mm, nothing under the block). The notes say this (section 8, item 3). I repeat it because it also decides finding 4.
6. Process: three full runs instead of one, and cache files in the worktree (section 3). No effect on the numbers.

## 7. The numbers I consider right

Q1 (uncut block):

| Quantity | Value |
|---|---|
| Records per frame | 117: 11 particle, 58 edge, 48 face |
| Records with force | 111 (11 / 52 / 48) from the first force on; 105 (9 / 52 / 44) from frame 72 on |
| First force | holder travel 0.50 to 0.53 mm |
| 0.5 N | holder travel 0.967 mm (0.487 N at 0.933 mm, 0.525 N at 0.967 mm); blade travel 0.913 mm; also true per substep |
| 1.0 N | holder travel 1.400 mm |
| Slope | 1.17 N per mm of holder travel (1.33 N per mm of blade travel) at the start; 1.24 N per mm of holder travel at the end |
| End of travel (7.0 mm) | 7.85 N; 7.86 N during the stop |
| Does the edge pass? | No. There is no cut. 7.86 N is where the holder stopped, not a limit of the tissue. |
| Largest single record | 0.123 N |

Q2 (V wound):

| Quantity | Value |
|---|---|
| Does the blade enter the V? | No, in five runs. From frame 24 on the axis stays 0.73 to 0.80 mm above the lip row. |
| Opening at the top row | 1.015 to 1.018 mm from frame 24 on (five runs); single pairs 0.99 to 1.06 mm |
| Smallest distance axis to a cut-front vertex | 3.29 mm, at 5.0 N |
| Force when a cut-front vertex is within 1.95 mm | Does not exist in this scene |
| Force when a cut-front vertex is within 0.95 mm | Does not exist in this scene |
| Class (ii), cut front | 0 N in every frame of every run |
| Where the force acts | 100 % at the 22 lip vertices |
| Shares by the rule of the task at 5.0 N | (i) 23.0 %, (ii) 0 %, (iii) 77.0 % |
| 0.5 N, frame-end reading | holder travel 0.73 mm (four runs) or 0.77 mm (one run); nearest cut-front vertex 3.80 mm from the axis |
| 0.5 N, per-substep reading | can be earlier: 0.57 mm in Run B, during the shaking |
| 1.0 N | holder travel 1.167 mm in all runs; nearest cut-front vertex 3.73 mm from the axis |
| Slope | 1.21 N per mm of holder travel |
| End of travel (4.45 mm) | 5.00 N; axis 3.38 mm above the cut front; lag 0.50 mm |
| Stall | Against the wound: yes. Against the holder: no, the blade pushes the block down. |

## 8. Why the lips stay together (Run B)

Sum over the 11 lip vertices of one side, x direction. "Inward" means toward the cut plane. The remainder is what self contact between the two wound faces must give for balance. The two sides mirror each other within 0.006 N during the travel.

| Frame | Force along d [N] | Tet elastic force, inward [N] | Blade normal force, outward [N] | Remainder, outward [N] |
|---|---|---|---|---|
| 22 | 0.50 | 0.716 | 0.155 | 0.561 |
| 35 | 1.03 | 0.908 | 0.330 | 0.578 |
| 77 | 2.72 | 1.604 | 0.902 | 0.702 |
| 134 | 5.00 | 2.722 | 1.739 | 0.984 |

- The blade pushes each lip outward with 0.35 N per newton of blade force.
- The tissue pushes each lip inward with 0.45 N more per newton of blade force.
- So the two wound faces are pressed together harder as the load grows (0.56 N to 0.98 N per side). The lips sit at the self-contact margin (smallest vertex-triangle distance of a lip 0.98 to 1.04 mm, smallest edge-edge distance 0.95 mm) and cannot open.
- The remainder has the sign of a repulsion, as self contact needs. I did not compute the self-contact forces themselves.

My inference, not run: the inward push comes from the bending of the block between the held ends (the top face shortens, as the notes say). A block that is supported from below may give a different balance. P0 does not show it.

## 9. What I did not check

- The cause of the shaking in finding 1 (self contact at the margin, DAT, or the missing tet damping). I only measured it.
- The self-contact forces. Section 8 gives them as a remainder.
- How the solver forms the drive rate. I only checked that the drive force from the body state equals the solver's contact force at frame ends.
- Any other scene: a supported block, another start gap, another blade radius, a deeper wound, a cut under load.

## 10. Files of the review

All under `/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/vbd-fast-cut-design/checks/p0/`.

- `p0_review.md`: this file.
- `review_checks.py`: the extra read-outs (imports the probe, does not change it).
- `review_rerun/p0_q1.csv`, `review_rerun/p0_q2.csv`, `review_rerun/review_rerun_log.txt`: Run A, the unmodified script.
- `review_rerun/review_checks_log.txt`, `review_rerun/review_frames_q1.csv`, `review_rerun/review_frames_q2.csv`, `review_rerun/review_substeps_q1.csv`, `review_rerun/review_substeps_q2.csv`: Run B.
