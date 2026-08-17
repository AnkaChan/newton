# Cloth self-contact performance plan

Date: 2026-08-13

## Goal and acceptance criteria

Improve the standalone `newton.examples cloth_franka` workload on top of
Newton commit `2ba7ffd6` without changing its solver configuration or contact
semantics. The inherited IsaacLab `Isaac-Lift-Cloth-Franka` artifacts are not
the baseline for this task: that configuration leaves VBD particle self-contact
disabled and therefore does not exercise `TriMeshCollisionDetector`.

The task is complete only when repeatable measurements show both:

- at least 1.30x speedup for the combined triangle-mesh self-contact detection
  and self-contact force/Hessian accumulation portion; and
- at least 10% end-to-end environment-step throughput improvement over
  `2ba7ffd6`.

If those thresholds cannot be reached, exhaust the viable optimization
families below and preserve quantitative evidence for every attempted design.

## Measurement protocol

1. Verify the imported Newton package resolves to this worktree and print an
   explicit source version marker after every Warp kernel edit.
2. Keep the example configuration fixed: one shirt and Franka scene, ten
   simulation substeps per 60 Hz frame, five VBD iterations per substep,
   timestep `1/600 s`, self-contact radius and margin `0.2 cm`, vertex contact
   capacity 16, edge contact capacity 20, collision-detection interval `-1`,
   and the existing material/contact settings. Use the repo-native
   `FastExampleClothManipulation` 30-frame timed body as the end-to-end shape.
3. Establish the `2ba7ffd6` baseline before editing with:
   - process-isolated warm-up and steady-state runtime measurements;
   - a representative Nsight Systems trace covering environment steps;
   - self-contact workload counters and per-kernel CUDA durations; and
   - a short reference trajectory/contact capture for correctness comparison.
4. Use short repeat measurements while iterating. Re-run interleaved or
   process-isolated before/after measurements for every retained milestone.
5. Treat the combined self-contact time as the sum of triangle-mesh collision
   detection/refit/traversal/contact-buffer kernels and
   `accumulate_self_contact_force_and_hessian` work in the selected step.
   Report kernel time and environment-step wall time separately.

## Iterative optimization loop

### Detection path

Profile `TriMeshCollisionDetector` vertex-triangle and edge-edge stages,
including BVH construction/refit, candidate traversal, filtering, buffer
initialization, and contact emission. Test the highest-value semantics-
preserving options first:

- eliminate capacity-wide or topology-invariant work;
- improve launch geometry and use active-prefix/grid-stride traversal where it
  reduces empty threads without breaking CUDA graph capture;
- reduce buffer clears, temporary traffic, counters, and redundant atomics;
- improve candidate locality and branch coherence;
- cache topology-invariant exclusions or adjacency; and
- split or fuse stages only when measurements justify the extra launch or
  register cost.

### Force/Hessian accumulation path

Measure the current contact-buffer layout and per-particle launch efficiency.
Prototype, in descending expected value:

- active-particle/contact-prefix work reduction;
- compact adjacency or persistent topology-aware indexing;
- warp/cooperative evaluation when it reduces divergent serial traversals;
- more efficient scatter/gather ownership that preserves force and Hessian
  sums while reducing atomics and memory traffic; and
- safe stage fusion or specialization for the cloth-only contact mix.

### Keep/reject gate

For each hypothesis:

1. record the baseline and expected mechanism;
2. implement the smallest testable change;
3. clear the Warp cache after kernel edits and confirm the version marker;
4. run targeted unit tests and numerical/contact comparison;
5. measure repeated kernel and end-to-end timing;
6. keep and commit only a repeatable improvement; otherwise revert the
   experiment safely and log why it failed; and
7. re-profile the new bottleneck before selecting the next hypothesis.

## Correctness and completion

- Add or extend `unittest` coverage for any changed buffer/index/traversal
  invariant, including empty, capacity, overflow, graph replay, and repeated
  contact cases where applicable.
- Compare contact counts/payloads and simulated trajectories against the pinned
  baseline with tolerances that account only for existing nondeterministic CUDA
  atomic ordering.
- Run focused collision/VBD tests, then the relevant broader Newton test set,
  `git diff --check`, and `uvx pre-commit run -a`.
- Produce final process-isolated before/after results, a representative final
  Nsight Systems breakdown, and a report in the AI-Docs task directory.
- Commit every retained improvement with the required AnkaChen identity, push
  `ankac/cloth-franka-perf`, and do not open a pull request.

## Completion status

Completed on 2026-08-17 at source commit `a4c6a1a2`. Its tree is identical to
profiled pre-reword commit `c84f62fb`. The final isolated Nsight comparison
measured the full detector plus force/Hessian path at 186.279 to 133.168 ms per
30 frames (**1.399x**). The eight-block, 32-process ABBA/BAAB comparison
measured **1.135x** end-to-end throughput with a 95% block-bootstrap interval
of **[1.132x, 1.138x]**; all eight blocks favored the candidate. Both original
acceptance thresholds are therefore satisfied.
