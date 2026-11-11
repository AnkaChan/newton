# Checks 2 and 3: referee comparison of implementation a and implementation b

Host-only check for decision D12. Pure geometry. No newton, no warp, no solver, no stepping, no GPU.
Date: 2026-11-10. Python 3.12.13, numpy 2.5.3.

## Files

Compared, not edited (all in `/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/vbd-fast-cut-design/checks`):

- a: `check2_3_impl_a.py`, `check2_3_impl_a.json`, `check2_3_impl_a.md`
- b: `check2_3_impl_b.py`, `check2_3_impl_b.json`, `check2_3_impl_b.md`

Written by the referee, same folder:

| file | what it does |
|---|---|
| `referee_compare.py` | Compares the two JSON files number by number (grid runs). Writes the tables of section 3. |
| `referee_checks.py` | Tests for the differences: re-run of b, the tie at the reach, the tie counter, the sign of n, the drift schedule. |
| `referee_impl_a_copy.py` | Copy of script a with test prints for the tie counter (lines marked REFEREE). Writes no file. |
| `referee_cross_mesh2.py` | Runs the second mesh of a through the replay of b, and the second mesh of b through the replay of a. |

Run (each under one minute):

```
export TMPDIR=/tmp/horde-tmpdir; export CUDA_VISIBLE_DEVICES=""; export PYTHONDONTWRITEBYTECODE=1
cd /home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/vbd-fast-cut-design/checks
PY=/home/horde/Code/Graphics/newton-working-copies/alm-dat-main/.venv/bin/python
$PY referee_compare.py
$PY referee_checks.py
$PY referee_impl_a_copy.py
$PY referee_cross_mesh2.py
```

Options: `referee_compare.py --tables PATH` also writes the tables of section 3 to PATH. `referee_checks.py --rerun-b PATH` also runs all of b with its JSON written to PATH.
The six files of a and b were not changed by the review (same time stamps before and after).

Source marks in this file: [a+b] = the number is in both JSON files and is the same. [a] or [b] = only one implementation has it. [ref] = a referee test. "My inference" = not tested.

## 1. Result in short

1. Grid mesh: 17 runs. I compared 37 numbers per run, 629 in all. 623 are the same. Openings and volume ratios agree to 2e-6 (b rounds them to 6 decimals). Step height and lead agree to 6e-5 mm (b rounds them to 4 decimals).
2. The task asks for 20 of these 37 numbers per run (340 in all). All 340 are the same.
3. The 6 numbers that differ are extra diagnostics: 1 "last pressed step" and 5 "tets with centroid on the plane". See D1 and D2.
4. No difference comes from a different reading of the cut rule. D1 is rounding at an exact tie in b. D2 to D4 are different definitions of extra numbers.
5. Both scripts reproduce their stored JSON files. I re-ran both with the output sent to a scratch path. a: identical. b: identical but for the run time.
6. The two implementers used different second meshes. I ran each mesh through the other implementation. All compared numbers are the same in the 6 runs.

## 2. Why I take the agreed numbers as right

- Two independent implementations agree. They differ in the face table, the label formula, the R2 solver and the tie tolerances.
- Both build the same mesh. The vertex array and the tet array are identical [ref]. Both follow the parity rule of `builder.py` lines 12529 to 12541. I read these lines and compared the ten tets.
- The interior face sets are identical (10336 faces). The area-weighted outward normals are identical [ref].
- Each script has a second code path for the runs without drift, and it matches. a: a one-shot cut in 27 runs [a]. b: a direct recount in 12 runs [b].
- Hand checks of case A fit. They are under the tables.
- At exact ties the agreed numbers follow the rule text ("<= reach", ">= 0").

Limits:

- For the drift runs no script has a second code path. There the evidence is the agreement of a and b alone.
- Both implementers took the same reading in the places of section 5. Two of them move numbers.

## 3. Agreed numbers, one table per blade case

Words in the tables:

- Pairs: (cut face, corner) pairs where the corner is split.
- Born crossed: opening <= 0.
- Cut-face corners not split: corners where the two tets of a cut face still hold the same particle.
- "u does not leave all own cut faces": the smallest (-m_f . u) over the own cut faces of the instance is <= 0. For R2 this means that no safe direction exists.
- Tets cut free: tets in components with fewer than 1 percent of all tets (fewer than 56 on the grid).
- Item 3 of the task (particles that join the two sides) is empty. The body is separated in every run.

Every cell of the 9 grid tables is the same text from a and from b (425 cells) [ref].

### 3.1 Case A, aligned, x0 = 42.00 mm (grid plane)

| number | no drift | drift, sticky labels | drift, labels not sticky |
|---|---|---|---|
| cut faces | 80 | 80 | 80 |
| split vertices | 55 | 55 | 55 |
| copies | 55 | 55 | 55 |
| split vertices with 3+ instances | 0 | 0 | 0 |
| most instances of one vertex | 2 | 2 | 2 |
| components | 2 | 2 | 2 |
| component sizes [tets] | 2800 / 2800 | 2800 / 2800 | 2800 / 2800 |
| tets cut free | 0 | 0 | 0 |
| separated | yes | yes | yes |
| step height [mm] | 0.000 | 0.000 | 0.000 |
| lead [mm] | 4.950 | 4.700 | 4.700 |
| start gap pairs | 240 | 240 | 240 |
| cut-face corners not split | 0 | 0 | 0 |
| R1 born crossed (opening <= 0) | 0 | 0 | 0 |
| R1 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R1 smallest opening [mm] | 1.000 | 1.000 | 1.000 |
| R1 smallest volume ratio | 0.833 | 0.833 | 0.833 |
| R1 inverted tets | 0 | 0 | 0 |
| R1 instances whose u does not leave all own cut faces | 0 of 110 | 0 of 110 | 0 of 110 |
| R2 born crossed (opening <= 0) | 0 | 0 | 0 |
| R2 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R2 smallest opening [mm] | 1.000 | 1.000 | 1.000 |
| R2 smallest volume ratio | 0.833 | 0.833 | 0.833 |
| R2 inverted tets | 0 | 0 | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 110 | 0 of 110 | 0 of 110 |

Reason: a and b agree in all three variants. Hand check: the plane x = 42 holds 10 x 4 cell faces with 2 triangles each = 80 faces, and 11 x 5 = 55 vertices. Each half holds 14 x 10 x 4 cells of 5 tets = 2800 tets.

Tie: the lead of 4.950 mm without drift hangs on an exact tie. At steps 8, 20, 32 and 44 the vertex row under the axis is at exactly 1.95 mm. The rule says "<= reach", so the row is pressed and 4.950 is right. With a reach that is 1e-9 mm smaller the lead is 4.700 mm [b, tie check]. With drift there is no tie.

### 3.2 Case A, aligned, x0 = 42.75 mm (quarter cell off)

| number | no drift | drift, sticky labels | drift, labels not sticky |
|---|---|---|---|
| cut faces | 80 | 80 | 80 |
| split vertices | 55 | 55 | 55 |
| copies | 55 | 55 | 55 |
| split vertices with 3+ instances | 0 | 0 | 0 |
| most instances of one vertex | 2 | 2 | 2 |
| components | 2 | 2 | 2 |
| component sizes [tets] | 2800 / 2800 | 2800 / 2800 | 2800 / 2800 |
| tets cut free | 0 | 0 | 0 |
| separated | yes | yes | yes |
| step height [mm] | 0.750 | 0.750 | 0.750 |
| lead [mm] | 4.700 | 4.700 | 4.700 |
| start gap pairs | 240 | 240 | 240 |
| cut-face corners not split | 0 | 0 | 0 |
| R1 born crossed (opening <= 0) | 0 | 0 | 0 |
| R1 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R1 smallest opening [mm] | 1.000 | 1.000 | 1.000 |
| R1 smallest volume ratio | 0.833 | 0.833 | 0.833 |
| R1 inverted tets | 0 | 0 | 0 |
| R1 instances whose u does not leave all own cut faces | 0 of 110 | 0 of 110 | 0 of 110 |
| R2 born crossed (opening <= 0) | 0 | 0 | 0 |
| R2 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R2 smallest opening [mm] | 1.000 | 1.000 | 1.000 |
| R2 smallest volume ratio | 0.833 | 0.833 | 0.833 |
| R2 inverted tets | 0 | 0 | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 110 | 0 of 110 | 0 of 110 |

Reason: a and b agree in all three variants.

Tie: 80 tets have the centroid exactly on the blade plane (two corner tets per cell of the column x = 42 to 45). The rule says ">= 0", so they are side B. The cut then follows the grid plane x = 42, and the step height is 42.75 - 42 = 0.75 mm. With drift along +n these tets stay on side B.

This result is fragile. With the plane 1e-6 mm further the 80 tets change side: parts 2880 / 2720, step height 2.25 mm, smallest opening 0.577 mm, smallest volume ratio 0.648 [b, tie check]. The other sign of n gives the same numbers [ref].

### 3.3 Case A, aligned, x0 = 43.50 mm (half a cell off)

| number | no drift | drift, sticky labels | drift, labels not sticky |
|---|---|---|---|
| cut faces | 80 | 80 | 80 |
| split vertices | 55 | 55 | 55 |
| copies | 55 | 55 | 55 |
| split vertices with 3+ instances | 0 | 0 | 0 |
| most instances of one vertex | 2 | 2 | 2 |
| components | 2 | 2 | 2 |
| component sizes [tets] | 2880 / 2720 | 2880 / 2720 | 2880 / 2720 |
| tets cut free | 0 | 0 | 0 |
| separated | yes | yes | yes |
| step height [mm] | 1.500 | 1.500 | 1.500 |
| lead [mm] | 4.200 | 4.450 | 4.450 |
| start gap pairs | 240 | 240 | 240 |
| cut-face corners not split | 0 | 0 | 0 |
| R1 born crossed (opening <= 0) | 0 | 0 | 0 |
| R1 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R1 smallest opening [mm] | 0.577 | 0.577 | 0.577 |
| R1 smallest volume ratio | 0.648 | 0.648 | 0.648 |
| R1 inverted tets | 0 | 0 | 0 |
| R1 instances whose u does not leave all own cut faces | 0 of 110 | 0 of 110 | 0 of 110 |
| R2 born crossed (opening <= 0) | 0 | 0 | 0 |
| R2 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R2 smallest opening [mm] | 0.577 | 0.577 | 0.577 |
| R2 smallest volume ratio | 0.648 | 0.648 | 0.648 |
| R2 inverted tets | 0 | 0 | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 110 | 0 of 110 | 0 of 110 |

Reason: a and b agree in all three variants. Hand check: side A holds 14 full columns (2800 tets) and the 80 corner tets with centroid x = 42.75 of the next column. This gives 2880.

Tie: 40 tets (the center tet of each cell of the column) have the centroid exactly on the plane. They are side B. With the plane 1e-6 mm further: parts 2920 / 2680 and smallest volume ratio 0.735. Step height and smallest opening stay [b, tie check].

Stroke end: no vertex is pressed between two vertex layers. The stroke ends inside the block 4 times without drift and 1 time with drift [a+b: a "label clears inside" 4 and 1, b "strokes" 5 and 2].

### 3.4 Case B, yaw 3 degrees

| number | no drift |
|---|---|
| cut faces | 80 |
| split vertices | 55 |
| copies | 55 |
| split vertices with 3+ instances | 0 |
| most instances of one vertex | 2 |
| components | 2 |
| component sizes [tets] | 2800 / 2800 |
| tets cut free | 0 |
| separated | yes |
| step height [mm] | 0.785 |
| lead [mm] | 4.950 |
| start gap pairs | 240 |
| cut-face corners not split | 0 |
| R1 born crossed (opening <= 0) | 0 |
| R1 opening 0 to 0.02 mm | 0 |
| R1 smallest opening [mm] | 1.000 |
| R1 smallest volume ratio | 0.833 |
| R1 inverted tets | 0 |
| R1 instances whose u does not leave all own cut faces | 0 of 110 |
| R2 born crossed (opening <= 0) | 0 |
| R2 opening 0 to 0.02 mm | 0 |
| R2 smallest opening [mm] | 1.000 |
| R2 smallest volume ratio | 0.833 |
| R2 inverted tets | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 110 |

Reason: a and b agree. No tet centroid is on the plane [b]. The lead of 4.950 mm hangs on the same tie as in 3.1: the vertices (42, 15, z) lie under the axis for every yaw angle. Without the tie it is 4.700 mm [b, tie check].

### 3.5 Case B, yaw 10 degrees

| number | no drift | drift, sticky labels | drift, labels not sticky |
|---|---|---|---|
| cut faces | 80 | 90 | 120 |
| split vertices | 55 | 61 | 63 |
| copies | 55 | 61 | 113 |
| split vertices with 3+ instances | 0 | 0 | 35 |
| most instances of one vertex | 2 | 2 | 5 |
| components | 2 | 2 | 14 |
| component sizes [tets] | 2800 / 2800 | 2816 / 2784 | 2821 / 2765 + 2 x 2 + 10 x 1 |
| tets cut free | 0 | 0 | 14 |
| separated | yes | yes | yes |
| step height [mm] | 2.084 | 2.605 | 2.954 |
| lead [mm] | 4.950 | 4.700 | 4.700 |
| start gap pairs | 240 | 270 | 338 |
| cut-face corners not split | 0 | 0 | 22 |
| R1 born crossed (opening <= 0) | 0 | 15 | 12 |
| R1 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R1 smallest opening [mm] | 0.174 | 0.000 | 0.000 |
| R1 smallest volume ratio | 0.695 | 0.695 | 0.415 |
| R1 inverted tets | 0 | 0 | 0 |
| R1 instances whose u does not leave all own cut faces | 0 of 110 | 22 of 122 | 50 of 176 |
| R2 born crossed (opening <= 0) | 0 | 0 | 4 |
| R2 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R2 smallest opening [mm] | 0.577 | 0.251 | -0.005 |
| R2 smallest volume ratio | 0.735 | 0.706 | 0.441 |
| R2 inverted tets | 0 | 0 | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 110 | 0 of 122 | 15 of 176 |

Reason: a and b agree in all three variants. The lead without drift has the tie of 3.1.

The run with labels that are not sticky has 14 components: two large parts, 2 pairs of tets and 10 single tets. All R1 crossed pairs have an opening of exactly zero. The 4 R2 crossed pairs are below zero (smallest -0.005 mm) [a+b].

### 3.6 Case B, yaw 15 degrees

| number | no drift |
|---|---|
| cut faces | 96 |
| split vertices | 65 |
| copies | 65 |
| split vertices with 3+ instances | 0 |
| most instances of one vertex | 2 |
| components | 2 |
| component sizes [tets] | 2800 / 2800 |
| tets cut free | 0 |
| separated | yes |
| step height [mm] | 2.329 |
| lead [mm] | 4.950 |
| start gap pairs | 288 |
| cut-face corners not split | 0 |
| R1 born crossed (opening <= 0) | 16 |
| R1 opening 0 to 0.02 mm | 0 |
| R1 smallest opening [mm] | 0.000 |
| R1 smallest volume ratio | 0.695 |
| R1 inverted tets | 0 |
| R1 instances whose u does not leave all own cut faces | 20 of 130 |
| R2 born crossed (opening <= 0) | 0 |
| R2 opening 0 to 0.02 mm | 0 |
| R2 smallest opening [mm] | 0.344 |
| R2 smallest volume ratio | 0.711 |
| R2 inverted tets | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 130 |

Reason: a and b agree. The lead has the tie of 3.1. All 16 R1 crossed pairs have an opening of exactly zero [a+b]. They lie on step faces with the normal along y [b].

### 3.7 Case C, lean 3 degrees

| number | no drift |
|---|---|
| cut faces | 80 |
| split vertices | 55 |
| copies | 55 |
| split vertices with 3+ instances | 0 |
| most instances of one vertex | 2 |
| components | 2 |
| component sizes [tets] | 2800 / 2800 |
| tets cut free | 0 |
| separated | yes |
| step height [mm] | 0.314 |
| lead [mm] | 4.708 |
| start gap pairs | 240 |
| cut-face corners not split | 0 |
| R1 born crossed (opening <= 0) | 0 |
| R1 opening 0 to 0.02 mm | 0 |
| R1 smallest opening [mm] | 1.000 |
| R1 smallest volume ratio | 0.833 |
| R1 inverted tets | 0 |
| R1 instances whose u does not leave all own cut faces | 0 of 110 |
| R2 born crossed (opening <= 0) | 0 |
| R2 opening 0 to 0.02 mm | 0 |
| R2 smallest opening [mm] | 1.000 |
| R2 smallest volume ratio | 0.833 |
| R2 inverted tets | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 110 |

Reason: a and b agree. No tie.

### 3.8 Case C, lean 10 degrees

| number | no drift |
|---|---|
| cut faces | 80 |
| split vertices | 55 |
| copies | 55 |
| split vertices with 3+ instances | 0 |
| most instances of one vertex | 2 |
| components | 2 |
| component sizes [tets] | 2800 / 2800 |
| tets cut free | 0 |
| separated | yes |
| step height [mm] | 1.913 |
| lead [mm] | 5.222 |
| start gap pairs | 240 |
| cut-face corners not split | 0 |
| R1 born crossed (opening <= 0) | 0 |
| R1 opening 0 to 0.02 mm | 0 |
| R1 smallest opening [mm] | 0.700 |
| R1 smallest volume ratio | 0.735 |
| R1 inverted tets | 0 |
| R1 instances whose u does not leave all own cut faces | 0 of 110 |
| R2 born crossed (opening <= 0) | 0 |
| R2 opening 0 to 0.02 mm | 0 |
| R2 smallest opening [mm] | 0.807 |
| R2 smallest volume ratio | 0.737 |
| R2 inverted tets | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 110 |

Reason: a and b agree. No tie.

### 3.9 Case C, lean 15 degrees

| number | no drift |
|---|---|
| cut faces | 100 |
| split vertices | 67 |
| copies | 67 |
| split vertices with 3+ instances | 0 |
| most instances of one vertex | 2 |
| components | 2 |
| component sizes [tets] | 2800 / 2800 |
| tets cut free | 0 |
| separated | yes |
| step height [mm] | 2.121 |
| lead [mm] | 5.234 |
| start gap pairs | 300 |
| cut-face corners not split | 0 |
| R1 born crossed (opening <= 0) | 20 |
| R1 opening 0 to 0.02 mm | 0 |
| R1 smallest opening [mm] | 0.000 |
| R1 smallest volume ratio | 0.648 |
| R1 inverted tets | 0 |
| R1 instances whose u does not leave all own cut faces | 24 of 134 |
| R2 born crossed (opening <= 0) | 0 |
| R2 opening 0 to 0.02 mm | 0 |
| R2 smallest opening [mm] | 0.344 |
| R2 smallest volume ratio | 0.660 |
| R2 inverted tets | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 134 |

Reason: a and b agree. No tie. All 20 R1 crossed pairs have an opening of exactly zero [a+b]. They lie on step faces with the normal along z [b].

### 3.10 Second meshes (not part of the a against b comparison)

The two meshes differ, so the stored numbers cannot be compared directly. I ran each mesh through the other implementation. All numbers of the two tables below are the same from both implementations [ref, `referee_cross_mesh2.py`].

Mesh of a: block of 12 x 6 x 4 cells of 3 mm, 165 interior vertices moved by up to 0.9 mm (seed 20261110), `scipy.spatial.Delaunay`. 455 vertices, 1982 tets. Edge length 1.69 to 5.67 mm. Smallest tet altitude 0.091 mm. Deviation from the task text: a removed 7 flat tets inside the block by an edge removal, not by a plain drop [a].

| number | A, x0 = 18 mm (middle) | B, yaw 10 degrees | C, lean 10 degrees |
|---|---|---|---|
| cut faces | 48 | 68 | 71 |
| split vertices | 35 | 48 | 48 |
| copies | 35 | 50 | 50 |
| split vertices with 3+ instances | 0 | 2 | 2 |
| most instances of one vertex | 2 | 3 | 3 |
| components | 2 | 2 | 2 |
| component sizes [tets] | 995 / 987 | 1000 / 982 | 997 / 985 |
| tets cut free | 0 | 0 | 0 |
| separated | yes | yes | yes |
| step height [mm] | 0.797 | 2.433 | 2.948 |
| lead [mm] | 5.466 | 5.387 | 5.568 |
| start gap pairs | 144 | 204 | 213 |
| cut-face corners not split | 0 | 0 | 0 |
| R1 born crossed (opening <= 0) | 0 | 13 | 13 |
| R1 opening 0 to 0.02 mm | 0 | 3 | 0 |
| R1 smallest opening [mm] | 0.891 | -0.549 | -0.391 |
| R1 smallest volume ratio | 0.686 | -3.297 | -0.713 |
| R1 inverted tets | 0 | 6 | 3 |
| R1 instances whose u does not leave all own cut faces | 0 of 70 | 26 of 98 | 24 of 98 |
| R2 born crossed (opening <= 0) | 0 | 0 | 0 |
| R2 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R2 smallest opening [mm] | 0.931 | 0.102 | 0.111 |
| R2 smallest volume ratio | 0.687 | -3.458 | -0.195 |
| R2 inverted tets | 0 | 5 | 1 |
| R2 instances whose u does not leave all own cut faces | 0 of 70 | 0 of 98 | 0 of 98 |

Mesh of b: `lido-p01-tetmesh/tet_library_v2/119.npz`, an fTetWild mesh of a Thingi10K model. It is a nearly solid slab, not tissue. Scaled by 3 mm per cell. 2056 vertices, 9438 tets. Box 84.08 x 31.73 x 15.07 mm. Edge length 0.81 to 8.37 mm. Smallest tet altitude 0.47 mm [b].

| number | A, x0 = 42.04 mm (middle) | B, yaw 10 degrees | C, lean 10 degrees |
|---|---|---|---|
| cut faces | 147 | 139 | 149 |
| split vertices | 93 | 95 | 93 |
| copies | 98 | 95 | 97 |
| split vertices with 3+ instances | 4 | 0 | 4 |
| most instances of one vertex | 4 | 2 | 3 |
| components | 2 | 2 | 2 |
| component sizes [tets] | 4881 / 4557 | 4890 / 4548 | 4884 / 4554 |
| tets cut free | 0 | 0 | 0 |
| separated | yes | yes | yes |
| step height [mm] | 2.700 | 2.585 | 2.569 |
| lead [mm] | 6.817 | 7.613 | 6.695 |
| start gap pairs | 441 | 417 | 447 |
| cut-face corners not split | 0 | 0 | 0 |
| R1 born crossed (opening <= 0) | 10 | 20 | 18 |
| R1 opening 0 to 0.02 mm | 2 | 3 | 4 |
| R1 smallest opening [mm] | -0.700 | -0.592 | -0.593 |
| R1 smallest volume ratio | 0.191 | -0.292 | 0.185 |
| R1 inverted tets | 0 | 4 | 0 |
| R1 instances whose u does not leave all own cut faces | 23 of 191 | 36 of 190 | 36 of 190 |
| R2 born crossed (opening <= 0) | 0 | 3 | 0 |
| R2 opening 0 to 0.02 mm | 0 | 0 | 0 |
| R2 smallest opening [mm] | 0.144 | -0.070 | 0.037 |
| R2 smallest volume ratio | 0.183 | -0.013 | 0.205 |
| R2 inverted tets | 0 | 3 | 0 |
| R2 instances whose u does not leave all own cut faces | 0 of 191 | 2 of 190 | 2 of 190 |

## 4. Differences and their cause

### D1. Last pressed step of A_x42.75 without drift: a 59, b 58

- Cause: rounding at an exact tie in b. At step 59 the vertex row z = 0 is 0.75 mm off the plane and 1.8 mm above the axis. The exact distance is sqrt(0.75^2 + 1.8^2) = 1.95 mm = reach. In floating point the distance is 1.9500000000000006 [ref].
- a tests "distance <= reach + 1e-9". b tests "distance <= reach" with no tolerance. So a says pressed and b says not pressed.
- Right: 59. The rule says "<= reach".
- Effect on the task numbers: none. b with a reach of 1.95 + 1e-9 gives step 59 and the same task numbers [ref].
- The same tie occurs at steps 11, 23, 35 and 47 for the row that leaves the reach. There b has 11 pressed vertices and a has 22 [ref]. Other vertices are pressed at these steps, so the stroke goes on. No cut depends on the leaving row: b with the larger reach gives the same task numbers.
- Kind: not a different reading. A small defect of b (no tie tolerance).
- Note: the tie of 3.1 (lead 4.950 mm) rounds the other way in b: the distance is 1.9499999999999993, so b says pressed. a and b agree there.

### D2. "Tets with centroid on the plane": 5 runs differ

| run | a (`label_ties_on_plane`) | b (`tets_on_plane_tie`) |
|---|---|---|
| A_x42.75, drift, sticky | 20 | 80 |
| A_x42.75, drift, not sticky | 0 | 80 |
| A_x43.50, no drift | 80 | 40 |
| A_x43.50, drift, sticky | 10 | 40 |
| A_x43.50, drift, not sticky | 0 | 40 |

- Cause: two definitions. b counts the tets whose rest centroid is on the plane. a counts events: each time a sticky label is set from a centroid that is on the plane at that step.
- A_x43.50 without drift: the 80 events of a are 40 tets, each counted twice. The stroke ends inside the block (steps 9, 21, 33, 45), the labels are cleared, and the labels are set again [ref, `referee_impl_a_copy.py`].
- With drift a counts only step 0, where the drift is zero (20 and 10 events). With labels that are not sticky a does not count at all.
- Right for "tets with the rest centroid on the plane": 0, 80 and 40 for x0 = 42, 42.75 and 43.5. This is the count of b. The events of a without drift come from the same number of tets: 80 and 40 [ref].
- Effect on the task numbers: none. It is a diagnostic.

### D3. Labels in the detail list of vertices with 3+ instances (B yaw 10, drift, labels not sticky)

- 12 vertices are listed by both. The tets per instance are the same in all 12. The labels per instance differ in 2 of 12: at (42, 15, 3) and (39, 18, 3) a prints [-1, 1] and b prints [-1] for the first instance.
- Cause: two definitions of the printed label when labels are not sticky. a prints the label at the last pressed step (step 59, drift 0.7375 mm). b prints the last label that was used at a candidate face. I rebuilt both lists from the state of b: 12 of 12 fit [ref].
- Effect on the task numbers: none.

### D4. Component size list

- a stores the 12 largest sizes. b stores up to 40. In the run with 14 components the list of a ends early. The count (14) and the tets cut free (14) are the same.

### D5. Tie tolerances

| test | a | b |
|---|---|---|
| pressed | distance <= reach + 1e-9 | distance <= reach |
| side B | n . (centroid - c) >= -1e-9 | >= 0 |
| born crossed | opening <= 1e-12 | opening <= 0 |
| u does not leave all own cut faces | <= 1e-12 | <= 1e-9 |

- Only D1 comes from this. All counts of crossed pairs are the same. On the grid every R1 crossed pair is an exact zero in both codes.

### D6. Test for "separated"

- a: the component with the most side-A tets differs from the component with the most side-B tets. b: the tets at the two far ends are in different components. a also stores the far-end test.
- All three tests say yes in all 17 grid runs. The second largest part has 2720 tets or more [ref].

### D7. One sentence in the notes of b

- The notes of b say that the nearest side-A centroid is 0.75 mm from the plane in the three A cases. This holds for x0 = 42 and x0 = 43.5. For x0 = 42.75 it is 1.5 mm [ref]. The notes of a have 1.5 mm. The conclusion (no label flips in case A) is the same.

### D8. Scope

- Second mesh: a built a jittered Delaunay block. b took a ready fTetWild mesh from the library. Neither is tissue. Both say so. See 3.10.
- Only a has: 30 sensitivity runs (drift schedule, split sets no labels, second mesh by drop only, five more seeds), the lead at pressed corners.
- Only b has: tie checks (plane moved by 1e-6 mm, reach smaller by 1e-9 mm), the number of strokes, the cut steps, the uncut faces between labels.
- Where these extras overlap they fit: a "label clears inside" + 1 = b "strokes" in all 17 runs [ref]. Both tested "the split sets no labels" on the four sticky drift runs: no number changes [a], [b]. Both tested the bridge report with a 12 mm blade and got 30 joining particles (notes only, not in the JSON files).

## 5. Readings that both took the same way

These are not differences. Two of them move numbers.

1. Sign of n. The rule text says n = e x d. The vectors of cases A, B and C give n = d x e. Both used the case vectors. I ran the code of b with n = e x d without drift [ref]: A_x42.00 does not change. A_x42.75 changes to parts 2880 / 2720, step height 2.25 mm, smallest opening 0.577 mm, smallest volume ratio 0.648. A_x43.50 changes to parts 2920 / 2680, smallest volume ratio 0.735. Yaw 10, yaw 15, lean 10 and lean 15 do not change. Not run: yaw 3, lean 3 and the drift runs. In a drift run the sign also sets the drift direction.
2. Drift schedule. Both let the drift grow over all steps of the motion: 0 at step 0, 0.8 mm at step 64. The last vertex is pressed at step 58 or 59, at a drift of 0.725 or 0.7375 mm. The nearest side-A centroid of case A is 0.75 mm from the plane (x0 = 42 and 43.5) or 1.5 mm (x0 = 42.75) [ref]. So no label of case A flips while a vertex is pressed. The margin is 0.0125 mm. Case A with this schedule does not test the sticky rule. a ran a second schedule (0.8 mm at the last pressed step). I ran it with the code of b: all 8 runs give the numbers of a [ref]. See finding 6.
3. Stroke end. Both took the rule as written: a step with no pressed vertex ends the stroke and clears the labels, also inside the block.
4. When a label is needed. For the two tets of each uncut interior face with a pressed corner, and for the tets around a particle that the split divides. The second part changes no number [a], [b].
5. Group that keeps the particle. Both take the lowest tet id among the groups with label -1 only, else among all groups. This changes particle ids only.
6. Stroke length in the lean cases. Both measure the 0.95 mm above the top face and the 3 mm below the bottom face along d. Steps: 65 (A, B), 65, 66 and 67 (C at 3, 10 and 15 degrees) [a+b].
7. Start gap. Both compute it once after the whole replay, on the final topology, from rest geometry.

## 6. Findings for the design

1. Does the rule separate the body? Yes. All 17 grid runs and all 6 second-mesh runs end with two large parts. No particle joins the two sides. Without drift every face between the two label sides is cut [b: 12 of 12 runs].

2. Does it cut tets free? Not with sticky labels, and not without drift: 0 tets in these 19 runs. With labels that are not sticky and drift it does: 14 tets at yaw 10 (10 single tets and 2 pairs) [a+b]. Case A shows 0 tets, but only because of the 0.0125 mm margin of section 5 item 2. With the full drift at the last pressed step, labels that are not sticky cut 10 tets free at x0 = 42 and at x0 = 43.5 [a, confirmed with the code of b].

3. How high are the steps? Grid, no drift: 0 mm on the grid plane, 0.75 mm a quarter cell off, 1.5 mm half a cell off. Yaw 3, 10, 15 degrees: 0.785, 2.084, 2.329 mm. Lean 3, 10, 15 degrees: 0.314, 1.913, 2.121 mm. Yaw 10 with drift: 2.605 mm (sticky), 2.954 mm (not sticky). Second meshes: 0.80 to 2.95 mm and 2.57 to 2.70 mm. So the wound is a stair with steps of up to about one cell (3 mm). The step is higher than the reach of 1.95 mm at yaw 10, yaw 15 and lean 15. The cut has 80 faces up to 10 degrees, 96 faces at yaw 15 and 100 at lean 15. The step height is fragile at a centroid tie: 0.75 mm or 2.25 mm for a plane shift of 1e-6 mm (3.2).

4. How far does the cut lead the edge? 4.2 to 5.23 mm ahead of the edge axis on the grid. Aligned and yaw: 4.20 to 4.95 mm. Lean: 4.71 to 5.23 mm. This is about the reach plus one cell: a face is cut when one corner is pressed, and its other corners are up to one cell deeper. At the pressed corners alone the lead is at most 1.95 mm [a]. Second meshes: 5.39 to 5.57 mm (edges up to 5.7 mm) and 6.70 to 7.61 mm (edges up to 8.4 mm). So the lead grows with the tet size.

5. Which start-gap rule gives fewer crossed corners? R2. It has fewer crossed pairs than R1 in every run where R1 has any (9 of 23 runs). In the other 14 runs both have none.
   - Grid, sticky labels or no drift: R1 has 16 crossed pairs at yaw 15, 20 at lean 15 and 15 at yaw 10 with drift. All have an opening of exactly zero, on step faces. R2 has none. Its smallest opening is 0.251 mm. The two rules are equal in case A, yaw 3 and lean 3.
   - Grid, labels not sticky: R1 12, R2 4 (smallest opening -0.005 mm).
   - Mesh of a: R1 0, 13, 13 crossed pairs (down to -0.549 mm). R2 0, 0, 0 (smallest opening 0.102 mm).
   - Mesh of b: R1 10, 20, 18 (down to -0.700 mm). R2 0, 3, 0 (smallest opening -0.070 mm). At yaw 10 two instances have no safe direction: every direction moves them toward one of their own cut faces. So R2 is better, but it is not a guarantee on a mesh that is not a grid.
   - Neither rule keeps thin tets valid at g = 1 mm. Inverted tets: mesh of a R1 6 and 3, R2 5 and 1. Mesh of b R1 4, R2 3. On the grid no tet inverts. The smallest volume ratio is 0.648 with sticky labels or without drift.

6. What do sticky labels buy under drift? One clean cut. Yaw 10 with a drift of 0.8 mm [a+b]:

   | number | sticky | not sticky |
   |---|---|---|
   | cut faces | 90 | 120 |
   | copies | 61 | 113 |
   | split vertices with 3+ instances | 0 | 35 (up to 5 instances) |
   | components | 2 | 14 |
   | tets cut free | 0 | 14 |
   | cut-face corners not split | 0 | 22 |
   | R2 born crossed | 0 | 4 |
   | R2 smallest volume ratio | 0.706 | 0.441 |

   Without sticky labels the labels flip under the blade and the rule cuts a second surface. With the full drift at the last pressed step case A shows the same at x0 = 42 and at x0 = 43.5: 100 cut faces in place of 80, 81 copies in place of 55, 11 vertices with 3+ instances, 7 components, 10 tets cut free. Sticky labels keep 80 faces, 55 copies and two parts [a, confirmed with the code of b]. At x0 = 42.75 nothing changes: the first label flips at a drift of 1.5 mm.

   Limit: sticky labels end with the stroke. Half a cell off, the stroke ends inside the block 4 times, because no vertex is pressed between two vertex layers [a+b]. In these runs the labels come back the same. My inference: with a drift that flips a label between two such strokes, the cut would behave as with labels that are not sticky.

7. Copies. On the grid there is one copy per split vertex with sticky labels or without drift (55 to 67 copies). On the meshes that are not a grid some vertices get 3 or 4 instances: 2 per tilted run on the mesh of a, 4 in two of three runs on the mesh of b.
