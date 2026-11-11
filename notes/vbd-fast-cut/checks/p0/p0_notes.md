# P0 wedge probe: notes

Run P0 of the fast-cut design (decision D14, design.md question 2.8.1). Date: 2026-11-10 and 11 (UTC).
No cut code exists. No Newton source file was changed. The probe uses only what upstream main has.

## 1. Short result

- Q1 (uncut block). The force on the edge along the cutting direction d reaches 0.5 N at 0.97 mm of holder travel. The edge does not pass. The force grows to 7.86 N at 7 mm of holder travel and was still growing.
- Q2 (V wound). The blade does not enter the V. It rests on the two lips of the wound. The lips stay 1.015 to 1.016 mm apart from 0.5 N to 5.0 N. The force along d reaches 0.5 N at 0.73 mm of holder travel and 1.0 N at 1.17 mm.
- Q2. No cut-front vertex comes within the contact range (1.95 mm) of the blade axis. The smallest distance is 3.29 mm, at 5.0 N. So the two forces that the task asks for (at 1.95 mm and at 0.95 mm) do not exist in this run.
- Q2. Class (ii), records led by a cut-front vertex, gives 0 N in every frame. Every record with force has its contact point on one of the 22 lip vertices.

## 2. What was run

Command (one process, both scenes):

```
export TMPDIR=/tmp/horde-tmpdir
export CUDA_VISIBLE_DEVICES=1
export WARP_CACHE_PATH=/tmp/horde-tmpdir/warp-cache-fastcut-p0
export PYTHONPATH=/home/horde/Code/Graphics/newton-working-copies/vbd-fast-cut-design
cd /home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/vbd-fast-cut-design/checks/p0
/home/horde/Code/Graphics/newton-working-copies/alm-dat-main/.venv/bin/python p0_wedge_probe.py --scene both --out-dir . 2>&1 | tee p0_log.txt
```

Versions and device (from the head of `p0_log.txt`):

- newton 1.9.0.dev0, `newton.__file__` = `/home/horde/Code/Graphics/newton-working-copies/vbd-fast-cut-design/newton/__init__.py`, commit 2134db9cd0354019061d36a2762f886d59e88306.
- Warp 1.19.0 (CUDA Toolkit 12.9, driver 12.8), Python 3.12.13, numpy 2.5.3.
- Device `cuda:0` = NVIDIA L40, physical GPU 1. CUDA started. Nothing was installed.
- No viewer. No graph capture. 0.38 s per frame.

How often the probe ran. I state this because the task allows one probe.

| Run | What | Frames | Stored |
|---|---|---|---|
| smoke | both scenes, 3 frames each, to compile the kernels and test the script | 6 | no |
| 1 | both scenes, full | 241 + 165 | `earlier_runs/run1_p0_q1.csv`, `earlier_runs/run1_p0_q2.csv` |
| 2 | both scenes, full, more read-out columns | 241 + 165 | `earlier_runs/run2_p0_q1.csv`, `earlier_runs/run2_p0_q2.csv` |
| 3 | both scenes, full, more read-out columns | 241 + 165 | `p0_log.txt`, `p0_q1.csv`, `p0_q2.csv` |

The scene and all solver values were the same in every run. Only read-outs changed between the runs. All numbers below are from run 3 unless a run is named. GPU 1 was in use for about 15 minutes in total. About 6.5 minutes of that were the kernel compile of the smoke run. The stepping took about 8 minutes.

Run-to-run difference of the force along d over the three runs: at most 0.007 N in Q1 (all frames) and 0.021 N in Q2 from frame 20 on. In Q2 the first 20 frames differ between the runs (section 6.4).

## 3. The scene as built

Units SI. Lengths below in mm.

- Block: 12 x 10 x 4 cells of 3 mm (36 x 30 x 12), corner at the origin, top face at z = 12. Five tets per cell with the parity rule of `ModelBuilder.add_soft_grid`. Built in numpy and passed to `add_soft_mesh`.
- Q1: 715 particles, 2400 tets, 832 skin triangles, 1248 skin edges. Q2: 726 particles (11 copies), 2400 tets, 872 skin triangles, 1308 skin edges. 4 particle colors in both.
- Tissue: density 1000, E = 1e5 Pa, nu = 0.45 (k_mu 34482.8 Pa, k_lambda 310344.8 Pa), tet damping 0, particle radius 0.5.
- Cut plane x = 18. Q2: the 11 vertices (x = 18, z = 12) have copies. 40 tets with centroid x > 18 use the copies. 40 skin triangles lie on the plane (the two wound faces). The 11 vertices (x = 18, z = 9) are the cut front and are shared.
- Start gap, Q2: `add_soft_mesh` got the zero-gap positions. After that call the script moved each original by -0.5 in x and each copy by +0.5 in x in `builder.particle_q`. So `model.particle_q` and the state start with a gap of 1.000.
- Rest shape check (log line "rest check"): `model.tet_poses` equals the inverse edge matrices of the zero-gap positions within 4.9e-7 (relative). Against the initial positions the difference is 0.20. In Q2 80 tets start deformed; the largest entry of F - I is 0.167.
- `model.particle_q` is also the rest position of the rest-shape contact filter. That filter is off (radius 0).
- Hold: 330 particles with x <= 6 or x >= 30 have mass 0. The free particles weigh 7.56 g.
- Gravity: off, (0, 0, 0). The block is held and does not sag.
- Blade: one capsule, radius 0.45, half height 17, axis along y, center at x = 18, y = 15. Density 7800, mass 0.1717 g. Shape margin 0. It is a dynamic body.
- Holder: a body with no shape and mass 0, built with `add_body` as in the suture example. The host writes its pose once per frame, before the 10 substeps.
- Joint: `add_joint_d6(holder, blade)` with six driven axes. Linear ke 1e4, kd 1e2. Angular ke 10, kd 0.1. Target 0 on every axis.
- Holder path: start at z = 13.45 (capsule surface 1 mm above the rest top face). Down at 2 mm/s, so 0.0333 mm per frame. Q1: to z = 6.45 (capsule surface 6 mm below the rest top face), 7.00 mm of travel, reached in frame 210. Q2: to z = 9.00 (the rest height of the cut front), 4.45 mm of travel, reached in frame 134. Then 30 frames (0.5 s) of hold. Q1 has 241 frames, Q2 has 165.
- Pipeline: `CollisionPipeline(model, soft_contact_gap=1e-3, enable_rigid_soft_full_surface_contact=True)`, owned by the solver.
- Solver: `SolverVBD(model, iterations=20, rigid_soft_enable_dat=True, particle_enable_self_contact=True, particle_self_contact_margin=1e-3, particle_self_contact_gap=1e-3, particle_topological_contact_filter_threshold=1, particle_rest_shape_contact_exclusion_radius=0.0, particle_edge_parallel_epsilon=1e-7, rigid_body_particle_contact_buffer_size=1024, collision_pipeline=pipeline)`. Every other argument has its default on main.
- Contact values, set as the suture example sets them: `model.soft_contact_ke = 5e2`, `model.soft_contact_kd = 5`, `model.soft_contact_mu = 0.3`, `model.shape_material_ke` filled with 5e2, `model.shape_material_kd` filled with 5. The penalty stiffness that the solver holds per record is 500 N/m in every frame.
- 60 frames per second, 10 substeps, 20 iterations.
- Argument names: every name in the task exists on main with the same spelling. Nothing had to be renamed.

Deviations from the task text, and choices where the task text is open:

1. Mass. `add_soft_mesh` gives each particle a quarter of the volume of its tets (block 12.96 g before the hold). The suture example uses `add_soft_grid`, which gives each vertex the mass of one cell (19.3 g for this block). The task proposed `add_soft_mesh`.
2. Blade friction. The blade shape has mu 0.3 and the model has `soft_contact_mu` 0.3. Main mixes the two by the geometric mean, so the friction is 0.3. The suture needle has shape mu 0.2 (mixed 0.245). The task names 0.3.
3. Capacity. `rigid_body_particle_contact_buffer_size` is 1024 (default 256). This is a list size. The largest list in the runs was 218, so the default was enough and the value had no effect.
4. Blade inertia. `finalize` reported "Inertia validation corrected 1 bodies". The blade inertia diagonal changed from (1.71e-8, 1.73e-11, 1.71e-8) to (1.017e-6, 1.000e-6, 1.017e-6) kg m^2. The mass did not change. The blade tilt stayed below the resolution of the read-out (about 0.02 degrees) in all runs.
5. The script asks the solver for its contact force observable (`CONTACT_F`) in the last substep of each frame. Main documents this as a read-only output. It is used for the cross check only.
6. The force is read once per frame, after the tenth substep. The holder moves in one step of 0.0333 mm at the start of each frame, as the suture example does.
7. Q1 end of travel. I read "to 6 mm below the top face" as: the capsule surface is 6 mm below the rest top face.
8. Classes of Q2 as I read them. Top-row vertices: the 22 wound vertices at z = 12 (11 originals, 11 copies). Class (i): every corner of the record is a top-row vertex. Class (ii): the corner with the largest weight is a cut-front vertex. Class (iii): all other records. The CSV also splits (iii) into "wound mixed" (every corner is a top-row or cut-front vertex) and "other". A second split looks only at the corner with the largest weight (top row, cut front, other).
9. The folder `earlier_runs/` and the extra CSV columns are additions.

## 4. How the force is read

After each frame the script reads, from the Contacts object that the solver owns: record count, shape, particle indices, weights, body point, normal. From the solver it reads `body_particle_contact_penalty_k`.

For each record the host computes what `accumulate_suture_tip_force` computes (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:760-794`):

```
point    = sum of weight x particle position over the corners (a particle record: the particle position)
distance = dot(normal, point - body point in world space)
radius   = largest particle radius of the corners + shape margin        (0.5 mm here)
f        = k x (radius - distance) if radius - distance > 0, else 0     (k = 500 N/m; main rigid_vbd_kernels.py:2111, log barrier off)
f_d      = f x max(dot(normal, d), 0),  d = (0, 0, -1)
```

The normal points from the shape to the tissue. A record counts only if main's own selection function `_soft_contact_force_enabled` accepts it. It accepted every blade record in every frame.

CSV columns (62 per row). The main ones: `time_s`, `holder_z_mm`, `holder_travel_mm`, `blade_axis_z_mm`, `blade_lag_mm`, `axis_height_above_front_mm` (blade axis z minus the mean z of the 11 cut-front vertices), `min_dist_axis_to_front_vertex_mm`, `axis_height_above_top_row_mm`, `records_particle / _edge / _face`, `force_records_*`, `f_d_class_i_N`, `f_d_class_ii_N`, `f_d_class_iii_N`, `f_d_total_N`, `f_n_sum_N` (full normal force sum), `drive_fz_on_blade_N`, `solver_contact_fz_on_blade_N`, `wound_opening_top_mean_mm`, `max_particle_speed_mm_s`, `nonfinite`.

No run had a non-finite value.

## 5. Q1: capsule edge on the uncut block

"Top row" here is the 11 vertices of the grid line x = 18 on the top face. p / e / f = particle / edge / face records.

| frame | time [s] | holder travel [mm] | blade travel [mm] | blade lag [mm] | axis above the top row [mm] | records p / e / f | records with force p / e / f | force along d [N] | full normal force sum [N] | largest record [N] | drive force on the blade, z [N] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.017 | 0.000 | 0.000 | 0.000 | 1.450 | 11 / 58 / 48 | 0 / 0 / 0 | 0.000 | 0.000 | 0.0000 | 0.000 |
| 15 | 0.267 | 0.500 | 0.500 | 0.000 | 0.950 | 11 / 58 / 48 | 10 / 9 / 0 | 0.000 | 0.000 | 0.0000 | 0.000 |
| 16 | 0.283 | 0.533 | 0.530 | 0.003 | 0.949 | 11 / 58 / 48 | 11 / 52 / 48 | 0.032 | 0.032 | 0.0004 | -0.042 |
| 29 | 0.500 | 0.967 | 0.913 | 0.054 | 0.941 | 11 / 58 / 48 | 11 / 52 / 48 | 0.525 | 0.526 | 0.0063 | -0.537 |
| 42 | 0.717 | 1.400 | 1.297 | 0.103 | 0.935 | 11 / 58 / 48 | 11 / 52 / 48 | 1.030 | 1.034 | 0.0127 | -1.042 |
| 60 | 1.017 | 2.000 | 1.827 | 0.173 | 0.927 | 11 / 58 / 48 | 11 / 52 / 48 | 1.735 | 1.754 | 0.0222 | -1.747 |
| 90 | 1.517 | 3.000 | 2.707 | 0.293 | 0.918 | 11 / 58 / 48 | 9 / 52 / 44 | 2.930 | 3.018 | 0.0396 | -2.943 |
| 120 | 2.017 | 4.000 | 3.585 | 0.415 | 0.913 | 11 / 58 / 48 | 9 / 52 / 44 | 4.146 | 4.376 | 0.0585 | -4.158 |
| 150 | 2.517 | 5.000 | 4.462 | 0.538 | 0.908 | 11 / 58 / 48 | 9 / 52 / 44 | 5.373 | 5.849 | 0.0788 | -5.388 |
| 180 | 3.017 | 6.000 | 5.338 | 0.662 | 0.902 | 11 / 58 / 48 | 9 / 52 / 44 | 6.606 | 7.440 | 0.1001 | -6.627 |
| 210 | 3.517 | 7.000 | 6.214 | 0.786 | 0.895 | 11 / 58 / 48 | 9 / 52 / 44 | 7.851 | 9.164 | 0.1223 | -7.869 |
| 240 | 4.017 | 7.000 | 6.214 | 0.786 | 0.895 | 11 / 58 / 48 | 9 / 52 / 44 | 7.863 | 9.177 | 0.1228 | -7.861 |

Answers to Q1:

- Records. 117 records in every frame: 11 particle, 58 edge, 48 face. They exist from frame 0, because the detection range is 1.95 mm. From frame 16 on, 111 records carry force (11 / 52 / 48). From frame 72 on, 105 (9 / 52 / 44).
- Where they act. The corner with the largest weight is a top-row vertex in every record with force (largest weight between 0.74 and 1.00). So 11 vertices carry all the force, and each vertex is counted by about 10 records: the vertex, its edges and its triangles. Each record has its own penalty spring of 500 N/m.
- Start of force. Holder travel 0.50 to 0.53 mm. The axis is then 0.95 mm above the top row. This is the force range.
- 0.5 N. Between frame 28 (holder travel 0.933 mm, 0.487 N) and frame 29 (holder travel 0.967 mm, 0.525 N). Blade travel 0.913 mm. This is 0.43 mm of holder travel after the start of force. By kind at frame 29: particle 9 %, edge 49 %, face 42 %.
- 1.0 N. Frame 42, holder travel 1.400 mm.
- Largest force. The edge does not pass in this run. The force grows almost in a line: 1.17 N per mm of holder travel in frames 30 to 60, 1.24 N per mm in frames 120 to 210. At the end of the travel it is 7.85 N, in the hold 7.86 N (largest value 7.865 N). It was still growing when the holder stopped, so 7.86 N is not a limit of the tissue.
- Per record. The largest record carries 0.123 N. That is 49 % of the bound of design 2.3 (0.25 N = 500 N/m x 0.5 mm).
- What moves. The block bends between the held ends. The top row under the edge goes from z = 12.00 to 6.34 mm. The row at z = 9 goes to 4.10 mm. The blade is 0.786 mm behind the holder at the end (blade travel 6.214 mm).

## 6. Q2: capsule lowered into the V wound

"Top row" here is the 22 lip vertices (originals and copies). "Opening" is the mean x distance between each original and its copy.

| frame | time [s] | holder travel [mm] | blade axis z [mm] | blade lag [mm] | axis above the cut front [mm] | smallest distance axis to a cut-front vertex [mm] | axis above the top row [mm] | opening at the top row [mm] | (i) [N] | (ii) [N] | (iii) [N] | force along d, sum [N] | full normal force sum [N] | drive force on the blade, z [N] | largest particle speed [mm/s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.017 | 0.000 | 13.450 | 0.000 | 4.592 | 4.494 | 1.580 | 1.039 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 122.5 |
| 5 | 0.100 | 0.167 | 13.294 | 0.011 | 4.247 | 3.968 | 1.197 | 0.907 | 0.003 | 0.000 | 0.004 | 0.006 | 0.007 | -0.018 | 276.5 |
| 8 | 0.150 | 0.267 | 13.195 | 0.012 | 4.006 | 3.929 | 0.946 | 1.007 | 0.014 | 0.000 | 0.021 | 0.035 | 0.041 | -0.052 | 57.2 |
| 13 | 0.233 | 0.433 | 13.034 | 0.017 | 3.900 | 3.866 | 0.831 | 1.015 | 0.067 | 0.000 | 0.100 | 0.167 | 0.197 | -0.182 | 2.4 |
| 22 | 0.383 | 0.733 | 12.767 | 0.050 | 3.838 | 3.799 | 0.804 | 1.015 | 0.163 | 0.000 | 0.342 | 0.505 | 0.599 | -0.521 | 0.3 |
| 35 | 0.600 | 1.167 | 12.386 | 0.103 | 3.774 | 3.734 | 0.788 | 1.015 | 0.280 | 0.000 | 0.745 | 1.026 | 1.220 | -1.041 | 0.4 |
| 59 | 1.000 | 1.967 | 11.683 | 0.199 | 3.669 | 3.618 | 0.775 | 1.016 | 0.492 | 0.000 | 1.501 | 1.993 | 2.383 | -2.010 | 0.4 |
| 77 | 1.300 | 2.567 | 11.155 | 0.272 | 3.595 | 3.536 | 0.765 | 1.015 | 0.650 | 0.000 | 2.068 | 2.719 | 3.263 | -2.736 | 0.5 |
| 89 | 1.500 | 2.967 | 10.804 | 0.320 | 3.547 | 3.482 | 0.759 | 1.015 | 0.756 | 0.000 | 2.446 | 3.203 | 3.855 | -3.220 | 0.5 |
| 107 | 1.800 | 3.567 | 10.276 | 0.393 | 3.476 | 3.404 | 0.749 | 1.015 | 0.916 | 0.000 | 3.012 | 3.928 | 4.750 | -3.947 | 0.5 |
| 119 | 2.000 | 3.967 | 9.925 | 0.441 | 3.431 | 3.353 | 0.742 | 1.015 | 1.023 | 0.000 | 3.389 | 4.412 | 5.351 | -4.431 | 0.4 |
| 134 | 2.250 | 4.450 | 9.500 | 0.500 | 3.378 | 3.293 | 0.734 | 1.016 | 1.152 | 0.000 | 3.849 | 5.001 | 6.090 | -5.013 | 0.4 |
| 164 | 2.750 | 4.450 | 9.500 | 0.500 | 3.376 | 3.291 | 0.733 | 1.016 | 1.154 | 0.000 | 3.849 | 5.003 | 6.095 | -5.004 | 0.2 |

### 6.1 The requested numbers

- Force along d when the first cut-front vertex comes within 1.95 mm of the blade axis: not reached. No number.
- Force along d when it comes within 0.95 mm: not reached. No number.
- The smallest distance from the axis to a cut-front vertex in the whole run is 3.291 mm (last frame). The force along d is then 5.00 N.
- Share of the force along d at the end of the travel (frame 134, 5.00 N): class (i) 23.0 %, class (ii) 0.0 %, class (iii) 77.0 % (wound mixed 37.7 %, other 39.3 %).
- Share at 0.5 N (frame 22): class (i) 32.3 %, class (ii) 0.0 %, class (iii) 67.7 % (wound mixed 33.0 %, other 34.8 %).
- Class (ii) is exactly 0 N in all 165 frames.

### 6.2 What the blade does

- Before the touch the wound is at rest with an opening of 1.016 mm (single pairs 1.006 to 1.030). The lips stand at z = 12.32, which is 0.32 mm above the rest top face. The cut front is at z = 9.23. These values are from run 2, frame 7 (particle speed 0.3 mm/s). In run 3 the wound still shook in that frame.
- 0.5 N. Between frame 21 (holder travel 0.700 mm, 0.465 N) and frame 22 (0.733 mm, 0.505 N). The nearest cut-front vertex is then 3.80 mm from the axis. The frame number is the same in all three runs.
- 1.0 N. Frame 35, holder travel 1.167 mm. The nearest cut-front vertex is 3.73 mm from the axis.
- The force grows in a line: 1.21 N per mm of holder travel from frame 30 to the end. The uncut block of Q1 gives 1.17 to 1.24 N per mm.
- End of the travel (holder at z = 9.00): 5.00 N. The blade axis is at z = 9.50, 0.50 mm behind the holder. The cut front is at z = 6.12. So the axis is 3.38 mm above the cut front.
- The blade does not enter the wound. From frame 13 to the end the axis is 0.73 to 0.83 mm above the lip row. The lowest point of the capsule surface is 0.45 mm below the axis, so it stays 0.28 mm or more above the lip row.
- The opening does not change. From frame 13 to the end the mean opening is 1.015 to 1.016 mm. Single pairs are between 0.996 and 1.060 mm.
- The normal forces push each side outward: x component -1.74 N on the left side and +1.74 N on the right side at the end. The lips do not move apart under this push.
- All force is on the lips. In every frame the corner with the largest weight is a top-row vertex for 100 % of the force along d. The smallest "largest weight" over all records with force is 0.998. At the end 218 records carry force: 22 particle, 108 edge, 88 face. That is about 10 records per lip vertex. By kind: particle 10 %, edge 48 %, face 42 %.
- The largest record carries 0.040 N (16 % of the 0.25 N bound).

Does the blade stall?

- Against the wound: yes. The height of the axis above the lip row changes by 0.07 mm (0.804 mm at 0.5 N, 0.733 mm at 5.0 N) while the holder moves 3.7 mm.
- Against the holder: no. The blade follows the holder at 88 % of its speed and pushes the block down, as in Q1. The lag grows in a line to 0.50 mm. It does not change in the hold.

### 6.3 Reading for design question 2.8.1

Measured:

- The force test of 0.5 N passes while every cut-front vertex is 3.8 mm or more from the axis. 1 N is reached at 3.73 mm. So the size of the hand estimate holds: 1 N comes long before the edge is near the cut front.
- The source is not a squeeze inside the V. The blade rests on top of the two lips. The wound faces below the lips carry no blade force.
- The cut front never comes into the contact range, up to 5 N and up to the end of the holder path of the task.

My inference, not run:

- With the rule of design 2.3 a pressed vertex is a vertex within the contact range. In this run only the lips are pressed. Their faces on the blade plane are already cut. So I expect that the second row would not be cut in this scene, and that the cut would not lead the edge by 5 mm. It would stop after the first row.
- Hand calculation: the capsule and the particle radius together have a radius of 0.95 mm. Its axis can come within 1.95 mm of the corner of a V only if the half angle of the V is 29 degrees or more (sin = 0.95 / 1.95). For a face height of 3 mm that is an opening of about 3.3 mm at the lips. The start gap gives 1 mm (half angle 9.5 degrees).
- Why the lips stay together. In Q1 the top face is pulled toward the edge as the edge presses down: the width between the grid lines x = 15 and x = 21 on the top face shrinks from 6.00 to 5.38 mm. In Q2 the same width shrinks from 6.57 mm (run 2, before the touch) to 6.26 mm, while the opening stays 1.016 mm. So the tissue beside the wound is pressed together in x. I infer that this, and the pull of the zero-gap rest shape, hold the lips at the self-contact distance against the outward push of 1.74 N per side. The probe did not read the forces between the two wound faces.

### 6.4 The first 20 frames

- Q2 starts 0.5 mm away from its rest shape on 22 vertices and has no tet damping. In run 3 the wound shakes in frames 0 to 11: particle speed up to 439 mm/s, mean opening between 0.907 and 1.062 mm. From frame 14 on the largest speed is below 1 mm/s.
- In runs 1 and 2 the wound came to rest by frame 7 (speed below 1 mm/s, opening 1.016 mm). The blade touched the lips in frame 8. Then the wound shook again in frames 9 to 19: speed up to 281 mm/s (run 1) and 262 mm/s (run 2). The mean opening was between 0.88 and 1.18 mm in run 1 and between 0.94 and 1.08 mm in run 2. In frame 20 the largest speed is below 3 mm/s, from frame 21 on below 1 mm/s.
- So in two of three runs the first touch of the blade started about 0.18 s of shaking. In the third run the start shaking had not ended when the blade touched.
- The force along d in these frames differs by up to 0.28 N between the runs. From frame 20 on it differs by 0.021 N at most.
- Q1 has no such phase. Its largest particle speed is 0.65 mm/s.

## 7. Cross check

Three numbers for the vertical force on the blade:

- Drive: 1e4 N/m x (holder z - blade z) + 1e2 N s/m x (holder speed - blade speed), from the body poses and speeds of the state. This is the law that main applies to a D6 linear drive (`rigid_vbd_kernels.py`, `_evaluate_drive_axis` and its caller).
- Solver contact force: the sum over the blade records of the wrench that main's `CONTACT_F` observable reports. It holds the normal force, the contact damping and the friction.
- Host normal force: the sum of f x dot(normal, d) from section 4. Normal force only.

They agree:

| | Q1, all 225 frames with force | Q2, the 153 frames with force and particle speed below 5 mm/s | Q2, all 159 frames with force |
|---|---|---|---|
| largest abs(drive + solver contact force) | 0.0008 N | 0.0019 N | 0.10 N |
| largest abs(-drive - host normal force along d) | 0.025 N | 0.020 N | 0.052 N |
| largest abs(solver normal part sum - host full normal force sum) | 0.019 N | 0.020 N | 0.108 N |

Last frame of Q1: drive -7.8606 N, solver contact force +7.8607 N, host normal force along d 7.8627 N. Last frame of Q2: drive -5.0039 N, solver contact force +5.0038 N, host normal force along d 5.0029 N. The drive multiplier that the solver holds for the z axis has the same size (7.8606 and 5.0039).

The host normal force differs from the other two by up to 0.025 N in the quiet frames. Friction and contact damping are not in it. In the shaking frames of Q2 the three numbers differ more (up to 0.54 N in run 1).

In every frame the positive-part sum and the signed sum of the force along d are equal: no record has a normal that points up.

## 8. What the probe does NOT show

1. No cut. No row is split during the run. The probe does not show what the wound does when a row is cut under load, and it does not test the cut rule.
2. Nothing below the first row. The wound is one tet row deep. A deeper V is not tested.
3. One hold. The block is held by fixed particles 12 mm from the plane and nothing supports it from below. It bends by 3 mm (Q2) to 5.7 mm (Q1) under the edge. A block that lies on a table, or a longer block, is not tested. The force per mm of travel belongs to this hold.
4. One set of values: start gap 1 mm (equal to the self-contact margin), blade radius 0.45 mm, particle radius 0.5 mm, contact stiffness 500 N/m, speed 2 mm/s, friction 0.3, no tet damping. No smaller or larger start gap. No other radius.
5. The holder stops at z = 9 mm in Q2, as the task says. The probe does not show what happens if the holder goes lower. At the stop the force was 5.0 N and still growing in a line.
6. The forces between the two wound faces and the stress in the tissue are not read. The reason why the lips stay together (6.3) is an inference.
7. The force is read at the end of each frame. The design reads it in every substep. The force inside a frame, right after the holder step, is not read.
8. The first 0.2 to 0.33 s of Q2 are not a clean measurement (6.4). The probe does not separate the release of the start state from the first touch of the blade in run 3.
9. One record is not one contact. About 10 records share each pressed vertex. The sum is the real force on the blade (section 7), but a rule that looks at the strongest single record sees about a tenth of the force on a vertex.
10. Three runs on one GPU type. No CPU run. No graph capture. No Warp deterministic mode.
11. Gravity is off.

## 9. Files

All under `/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/vbd-fast-cut-design/checks/p0/`.

- `p0_wedge_probe.py`: the script. `--scene q1`, `--scene q2` or `--scene both`. `--max-frames N` for a short test. `--no-observable` to skip the solver's contact force observable.
- `p0_log.txt`: log of run 3. Head: versions, device, parameters. Per scene: the scene as built, one line per frame, a summary, and the list of records at the key frames (all 218 records of the last Q2 frame).
- `p0_q1.csv`, `p0_q2.csv`: one row per frame, run 3.
- `earlier_runs/`: the CSV files of runs 1 and 2 (fewer columns).
- `p0_notes.md`: this file.
