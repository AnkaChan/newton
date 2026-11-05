<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Catalog snapshot

- Source: `asv/collision_fidelity/cases.json` at `c63a3a24969f33cd86a7521e8d705664f2d29ed5`.
- Extracted on 2026-11-05. This is an inventory, not a pass report.

## Counts

| Family | Catalog cases | Supported cases |
|---|---:|---:|
| Robot manipulation | 30 | 29 |
| Soft contact | 13 | 13 |
| Cables | 2 | 2 |
| **Total** | **45** | **44** |

The duck case requires a second dynamics solver. Its asset and catalog entry remain.
Robot cable tasks belong to Robot manipulation. The two Cables cases are crossed
cables and cylinder wrap. Soft contact includes two analytic free-fall cases.

## Cases

| Case ID | Scene | Family | Support |
|---|---|---|---|
| `r5_franka_patch_contact_v2` | Franka grippers : cloth patch | Robot manipulation | supported |
| `r7_h1_towel` | Two H1 hands : towel | Robot manipulation | supported |
| `r7_h1_sheet` | Two H1 hands : sheet | Robot manipulation | supported |
| `r5_franka_trousers` | Franka grippers : trousers | Robot manipulation | supported |
| `r5_franka_dress` | Franka grippers : dress | Robot manipulation | supported |
| `r5_franka_tshirt_contact_v2` | Franka grippers : T-shirt | Robot manipulation | supported |
| `r5_franka_skirt_contact_v2` | Franka grippers : skirt | Robot manipulation | supported |
| `r5_franka_tote_contact_v2` | Franka gripper : tote handles | Robot manipulation | supported |
| `r5_franka_pouch_contact_v2` | Franka gripper : pouch edge | Robot manipulation | supported |
| `r7_h1_gusset_bag` | Two H1 hands : prepared gusset-bag rim | Robot manipulation | supported |
| `j3_h1_jacket` | Two H1 hands : native jacket | Robot manipulation | supported |
| `r6_franka_pad` | Franka gripper : soft pad | Robot manipulation | supported |
| `r6_allegro_block` | Allegro hand : soft block | Robot manipulation | supported |
| `r6_allegro_sphere` | Allegro hand : soft sphere | Robot manipulation | supported |
| `r6_franka_capsule` | Franka gripper : soft capsule | Robot manipulation | supported |
| `r6_franka_torus` | Franka gripper : soft torus | Robot manipulation | supported |
| `r6_franka_branched` | Franka gripper : branched solid | Robot manipulation | supported |
| `m3_franka_tote_empty` | Franka grippers : empty shopping bag | Robot manipulation | supported |
| `m3_franka_tote_2` | Franka grippers : shopping bag with two blocks | Robot manipulation | supported |
| `m3_franka_tote_2_spheres` | Franka grippers : shopping bag with two spheres | Robot manipulation | supported |
| `m3_franka_gusset_4` | Franka grippers : gusset bag with four payloads | Robot manipulation | supported |
| `m2_franka_pouch` | Franka grippers : prepared pouch | Robot manipulation | supported |
| `c6_crossed_cables_10substeps` | Crossed cables : sliding contact | Cables | supported |
| `c7_cylinder_wrap_10substeps` | Cable wrapped around a cylinder | Cables | supported |
| `c9_bundle_squeeze_withdraw` | Cable bundle : squeeze and withdraw | Robot manipulation | supported |
| `c9_pile_pickup` | Franka gripper : cable pile | Robot manipulation | supported |
| `s1_cloth_2_10substeps` | Cloth stack (2 layers) | Soft contact | supported |
| `s1_cloth_5_10substeps` | Cloth stack (5 layers) | Soft contact | supported |
| `s1_cloth_10_10substeps` | Cloth stack (10 layers) | Soft contact | supported |
| `s1_pads_2_10substeps` | Soft-pad stack (2 layers) | Soft contact | supported |
| `s1_pads_5_10substeps` | Soft-pad stack (5 layers) | Soft contact | supported |
| `s2_impact` | Soft solid impacting cloth | Soft contact | supported |
| `s2_lower` | Soft solid lowered onto cloth | Soft contact | supported |
| `s3_cloth_10substeps` | Free-falling cloth | Soft contact | supported |
| `s3_pads_10substeps` | Free-falling soft pad | Soft contact | supported |
| `s4_pads` | Soft pads : fixed sliding displacement | Soft contact | supported |
| `s5_pack_4` | Soft-solid packing (4 objects) | Soft contact | supported |
| `s5_pack_8` | Soft-solid packing (8 objects) | Soft contact | supported |
| `r5_franka_towel_contact_v2` | Franka grippers : towel | Robot manipulation | supported |
| `r2_franka_duck` | Franka arm : rubber duck | Robot manipulation | unsupported |
| `m2_franka_cable` | Franka gripper : cable | Robot manipulation | supported |
| `r5_allegro_towel` | Allegro hand : prepared towel | Robot manipulation | supported |
| `r3_allegro_ground_sphere` | Allegro hand : sphere on a table | Robot manipulation | supported |
| `m2_allegro_cable` | Allegro hand : cable | Robot manipulation | supported |
| `s4_poker_cards_no_ball` | Poker-card stack without a moving ball | Soft contact | supported |

## Source map

| Area | Source |
|---|---|
| Case records and loading | `asv/collision_fidelity/cases.json:1-102`; `asv/collision_fidelity/catalog.py:10-21` |
| Scene routing and unsupported recipe | `asv/collision_fidelity/scenes.py:11-59` |
| Asset sources and retained fixtures | `asv/collision_fidelity/assets/README.md:3-35` |
| Output contract and coverage | `asv/collision_fidelity/runner.py:17-68,144-290` |
| Current measurement definitions | `asv/collision_fidelity/metrics.py:110-131,377-445,447-551` |
| Current criterion reduction | `asv/collision_fidelity/tasks.py:195-256` |
| ASV cache and tracks | `asv/benchmarks/simulation/bench_collision_fidelity.py:23-126` |
| Shared semantic version | `asv/collision_fidelity/version.py:15-45` |
