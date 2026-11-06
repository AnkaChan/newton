<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Collision-fidelity benchmark overview

- Purpose: review how the benchmark builds scenes, measures contact, and reports results.
- Inputs: Newton at `32ef532f7ffc62ffa0688826583d72496df4604b`; [catalog digest](inputs/catalog.md); [requirements and decisions](decisions.md).
- Pseudo code: real names, plain conditions, explicit loops. Indentation marks function and control bodies only.
- Status: chapter 1 refreshed on 2026-11-06 after the small review fixes. This describes the existing draft implementation in [upstream PR #5144](https://github.com/newton-physics/newton/pull/5144). Full-run opt-in and lower-is-better ASV measures are approved but not implemented; Overall Pass rules remain open.

## 1.1 Intent

Newton already runs a workload once in `setup_cache` and shares its measurements across ASV tracks.
The nearest example is `asv/benchmarks/simulation/bench_quadruped_xpbd.py:51-75`.
Its tracks report speed, memory, and integration settings (`asv/benchmarks/benchmark_metric_tracks.py:54-97`).

This suite uses the same cache pattern for contact fidelity.
It selects one case, builds its inputs, calls public VBD, samples the state, and reduces the results.
The CLI and ASV share `run_case` (`asv/collision_fidelity/runner.py:159-307`).
Each case has one cache and seven scalar tracks (`asv/benchmarks/simulation/bench_collision_fidelity.py:56-131`).
Unfiltered ASV runs include these full workloads. There is no opt-in gate; `--quick` does not shorten their physics schedule (`asv/benchmarks/simulation/bench_collision_fidelity.py:4-10,24-35`).

## 1.2 Old and new

Here, **New** means the addition already present in the draft code.

| | Today (file:line) | New |
|---|---|---|
| Entry point | Cached timing workload, `asv/benchmarks/simulation/bench_quadruped_xpbd.py:56-75` | One shared runner for ASV and the CLI, `asv/collision_fidelity/runner.py:159-166`. |
| Measurements | Time, throughput, memory, `asv/benchmarks/benchmark_metric_tracks.py:54-97` | Jitter, RMS speed, fixed-support penetration, and coverage, `asv/benchmarks/simulation/bench_collision_fidelity.py:71-113`. |
| Physics | Ordinary Newton scenes | Public `SolverVBD` with DAT on for particle scenes, fast math off, and its owned `CollisionPipeline`, `asv/collision_fidelity/public_vbd.py:30-44`. |
| Selection | Workload class | 45 catalog cases; 44 supported and one explicit unsupported duck recipe, `notes/collision-fidelity/inputs/catalog.md:9-16`. |
| Assets | Newton's existing robot assets | Franka, Allegro, H1, procedural soft meshes, native jacket, and four H1 initial-state fixtures, `asv/collision_fidelity/assets/README.md:3-37`. |
| Result meaning | Timing tracks | Raw measurements, separate criteria, and an overall result, `asv/collision_fidelity/tasks.py:195-256`. |

The scene set contains 30 robot-manipulation cases, 13 soft-contact cases, and two cable-contact cases.
Robot cable tasks belong to the manipulation family; they are not counted twice.
The [catalog digest](inputs/catalog.md) lists every case and its support state (`notes/collision-fidelity/inputs/catalog.md:9-16`).
Generated meshes may differ slightly if the test mechanism stays the same (Anka, 2026-11-06; requirement R8). Tests check geometry structure and scale instead of cross-platform byte identity (`asv/tests/test_collision_fidelity_assets.py:41-122`).

## 1.3 Argument list

The shared entry point keeps the implemented signature (`asv/collision_fidelity/runner.py:159-166`):

```
run_case(
    case_id: str,                          # stable catalog ID; required
    *,                                    # remaining arguments are keyword-only
    profile: str = "full",                # full schedule or smoke, capped at 1 s
    device: str | None = None,             # None = cuda:0 when available, otherwise cpu
    duration: float | None = None,         # [s]; None = profile limit; given = earlier stop
    usd_path: str | Path | None = None,    # None = no USD recording
)
```

`case_id` stays positional, as in the current runner. It is not a new public Newton API.
The catalog supplies scene parameters. Explicit run arguments select device and duration.
The runner rejects a requested duration shorter than one physical substep, then checks the built scene's time step, iterations, and duration against the catalog (`asv/collision_fidelity/runner.py:175-185,229-237`).

`Timeline` extends a declared stationary hold to 60 seconds without slowing the other motion.
It preserves the frame interval and uses at most ten substeps (`asv/collision_fidelity/protocol.py:13-40`).
Smoke and early-stop runs retain the full required measurement windows (`asv/collision_fidelity/runner.py:183-191,256-263`).

Not arguments: material values, contact values, iteration count, and hand commands. These belong to the selected scene inputs (`asv/collision_fidelity/scenes.py:17-59`).
During scene construction, `create_solver` caps the authored contact coefficients at ten times the smallest declared elastic reference. For cloth and tetrahedral solids, this compares numerical values in Pa with contact coefficients in N/m; it is not a dimensionless physical stiffness ratio (`asv/collision_fidelity/public_vbd.py:23-30`; `asv/collision_fidelity/contact_stiffness.py:49-78,160-184`).

## 1.4 Input types

**Case record:** a JSON object, not a new Newton model type.

| Member | dtype, shape | Host counterpart | Meaning |
|---|---|---|---|
| `case_id`, `scene_id`, `runner` | `str` | Workload selection, `asv/benchmarks/simulation/bench_quadruped_xpbd.py:24-33` | Stable selection and scene factory. |
| `kwargs` | `dict` | Scene constructor arguments, `asv/collision_fidelity/scenes.py:50-51` | Authored parameters, including selected contact capacities. |
| `frame_dt_s`, `substeps`, `iterations` | `float`, `int`, `int` | Solver inputs, `asv/collision_fidelity/public_vbd.py:64-74` | Frame duration [s], physical steps per frame, iterations per step. |
| `hold_start`, `hold_end`, `stationary_windows` | optional `float`, optional `float`, `float[W,2]` | No timing-workload counterpart | Authored stage boundaries [s]; mapped by `asv/collision_fidelity/protocol.py:26-68`. |
| `category_applicability` | `dict[str,bool]` | No host counterpart | Which criteria apply; `asv/collision_fidelity/tasks.py:203-220`. |
| `support`, `assets` | `dict`, `dict` | Existing asset loader | Support reason and asset references; `asv/collision_fidelity/cases.json:84-102`. |

The runner checks the ID, profile, and requested duration before scene construction.
`Timeline` then checks substeps, frame time, duration, and stage boundaries (`asv/collision_fidelity/runner.py:175-185`; `asv/collision_fidelity/protocol.py:13-45`).
Unsupported cases return before importing the physics backend (`asv/collision_fidelity/runner.py:200-210`).

**Observed state:** the existing Newton state, read without solver hooks.
`P` is the particle count; `B` is the rigid-body count.

| Member | dtype, shape | Host counterpart | Meaning |
|---|---|---|---|
| `particle_q` | Warp `vec3[P]`, read as `[P,3]` | `newton/_src/sim/state.py:111-112` | World positions [m]. |
| `particle_qd` | Warp `vec3[P]`, read as `[P,3]` | `newton/_src/sim/state.py:114-115` | World velocities [m/s]. |
| `body_q` | Warp `transform[B]`, read as `[B,7]` | `newton/_src/sim/state.py:132-133` | Position and orientation of each rigid body. |
| `body_qd` | Warp `spatial_vector[B]`, read as `[B,6]` | `newton/_src/sim/state.py:135-139` | Linear COM velocity, then angular velocity. |

The observer uses every particle vertex and 14 fixed surface points per identified cable capsule.
It does not restrict Jitter to the grip region (`asv/collision_fidelity/metrics.py:163-166,232-285,480-489`).
Some scene boundaries are prescribed; the catalog does not promise that every point is free (`asv/collision_fidelity/public_vbd.py:49-54,82-91`).

## 1.5 Data structures on the model

The benchmark adds no solver state layout. It records settings on the model and stores observations on the host.
`SceneInputs.metrics()` exposes actual settings and counts (`asv/collision_fidelity/public_vbd.py:45-91`).

| Attribute | dtype, shape | Meaning |
|---|---|---|
| `collision_fidelity_solver_options` | `dict` on model | Actual public solver settings, `asv/collision_fidelity/public_vbd.py:45-48`. |
| `protocol` | `dict` in result | Physical schedule and measurement scope, `asv/collision_fidelity/runner.py:183-191`. |
| `execution` | `dict` in result | Steps completed, stop reason, and full-timeline coverage, `asv/collision_fidelity/runner.py:272-296`. |
| `measurements` | `dict` in result | Raw values, units, sample identities, and missing coverage, `asv/collision_fidelity/metrics.py:478-555`. |
| `task_observations` | `list[dict]`, one row per sampled frame | Conservative object and hand bounds, `asv/collision_fidelity/tasks.py:116-148`. |
| `criteria` | `dict[str,dict]` | Applicability, status, reason, and evidence, `asv/collision_fidelity/tasks.py:167-173`. |
| `provenance`, `inputs` | `dict`, `dict` | Tested Newton revision, device, versions, and scene settings, `asv/collision_fidelity/runner.py:238-252`. |

The result keeps three core measurements. Each has a narrow meaning.
Raw per-case JSON uses metres for distances. ASV and summaries convert them to millimetres (`asv/benchmarks/simulation/bench_collision_fidelity.py:71-89`; `asv/collision_fidelity/runner.py:40-66`).

| Measurement | Computation | Scope |
|---|---|---|
| **Jitter (mm)** | Position RMS after fitting and removing a linear trend | Maximum over points and complete 0.5 s windows; includes real deformation or swing, `asv/collision_fidelity/metrics.py:110-131,461-462`. |
| **RMS speed (m/s)** | Square root of the time-weighted mean squared speed | Same points and windows; retains mean motion, `asv/collision_fidelity/metrics.py:106-120`. |
| **Rigid-soft penetration: fixed supports (mm)** | Maximum negative particle-shell gap | Declared fixed planes and boxes; excludes robot, soft-soft, and self-contact penetration, `asv/collision_fidelity/metrics.py:381-405,532-544`. |

The two free-fall cases also compare system COM velocity with the analytic gravity solution.
They need no reference rollout (`asv/collision_fidelity/metrics.py:407-449`).

Grasp, Hold, Transport, and Place and release apply only where the case declares them.
A prepared hold excludes Grasp. Separated bounds can prove a sampled Hold failure; overlapping bounds cannot prove a successful hold (`asv/collision_fidelity/tasks.py:176-234`).
Missing evidence or acceptance limits stay **Incomplete**. Any required failure gives **Fail**; every required criterion must pass for **Pass** (`asv/collision_fidelity/tasks.py:215-255`).
No supported case can receive an overall Pass in this revision: required measurements or acceptance limits remain missing. Numerical completion is not a fidelity pass (`asv/collision_fidelity/metrics.py:496-535`; `asv/collision_fidelity/tasks.py:215-255`).
Numerical completion can describe a smoke prefix. `full_timeline_completed` records full coverage separately (`asv/collision_fidelity/tasks.py:235-243`; `asv/collision_fidelity/runner.py:292`).

The summary reports pass / evaluated and evaluated / applicable.
Motion means need every declared window. Penetration means need a full finite trajectory.
Each mean lists its contributing cases (`asv/collision_fidelity/runner.py:17-68`).
Raw JSON can retain useful partial values even when ASV emits NaN for missing coverage (`asv/benchmarks/simulation/bench_collision_fidelity.py:43-113`).
ASV skips unsupported execution and raises for `error` or `invalid`, including the reason and recorded traceback. `nonfinite` remains a numerical outcome, so its failure and coverage tracks remain available (`asv/benchmarks/simulation/bench_collision_fidelity.py:59-113`).

## 1.6 Pseudo code

This shows the numerical loop. Setup also checks inputs and records provenance.
The runner records setup and stepping exceptions around this loop. USD-close errors are recorded without discarding completed physics or an earlier failure (`asv/collision_fidelity/runner.py:125-132,227-271,293-300`).

```
run_case(case_id, *, profile, device, duration, usd_path):       # asv/collision_fidelity/runner.py:159
    cases = load_cases()                                       # read IDs and authored inputs
    timeline = Timeline(cases[case_id])                         # map authored stages to physical time
    if duration is given and shorter than timeline.dt:
        raise ValueError                                      # reject before scene construction
    result = initial result record from section 1.5
    if the catalog marks this case unsupported:
        result["execution"] = unsupported with the declared reason
        result["criteria"] = overall unsupported, not applicable
        return result                                         # includes the support reason
    scene = build_scene(case_id, device)                        # create the model, states, and public solver
    timeline.protocol = schedule with fixed supports and declared cable IDs
    observer = FidelityObserver(scene.model, timeline.protocol) # create the state observer
    observer.frame(scene.state, 0)                             # record the free-fall baseline when applicable
    frames = [sample_task(scene, 0, protocol=timeline.protocol)]
    execution = result["execution"]
    capacity = initial unavailable contact-count record
    stop = timeline.duration
    if profile == "smoke":
        stop = min(stop, 1.0)
    if duration is not None:
        stop = min(stop, duration)
    planned_steps = stop rounded up to a physical substep       # retain the full required windows
    for index in range(planned_steps):
        physical_start = index * timeline.dt
        state = scene.step_substep(timeline.command_time(physical_start))
        end = (index + 1) * timeline.dt
        execution.update(steps=index + 1, time_s=end)
        if _sample_capacity(scene.solver, capacity):
            execution.update(status="invalid")                # final exposed contact count overflow
            break
        observer.observe(state, end, timeline.dt)
        if not observer.all_samples_finite:
            execution.update(status="nonfinite")
            break
        if a display frame ends or this is the last planned step:
            frames.append(sample_task(scene, end, protocol=timeline.protocol))
    else:
        execution.update(status="completed", completed=True)
    execution["full_timeline_completed"] = all full-timeline steps completed
    result["measurements"] = observer.result()
    result["task_observations"] = frames
    result["criteria"] = criterion_results(frames, timeline.protocol, execution=execution, measurements=result["measurements"])
    return result
```

All names exist today: `load_cases` in `asv/collision_fidelity/catalog.py:10`; `Timeline` in `asv/collision_fidelity/protocol.py:10`; `build_scene` in `asv/collision_fidelity/scenes.py:11`; `FidelityObserver` in `asv/collision_fidelity/metrics.py:134`; `sample_task` and `criterion_results` in `asv/collision_fidelity/tasks.py:152,195`; `_sample_capacity` and `run_case` in `asv/collision_fidelity/runner.py:134,159`.

The cloth scene shows the substep contract: update prescribed hand commands, clear forces, call `solver.step`, swap states, and return the current state (`asv/collision_fidelity/cloth_robot_scenes.py:467-477`).
The solver owns collision passes. The benchmark does not replace them (`asv/collision_fidelity/public_vbd.py:23-44`).
The ASV cache calls the runner once; its tracks read this result (`asv/benchmarks/simulation/bench_collision_fidelity.py:24-113,127-131`).

## 1.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| Solver boundary | closed; Anka, chat, recorded 2026-11-04 | Standard VBD only; unsupported cases stay explicit. | None; requirement R1. |
| Measurement reference | closed; Anka, earlier chat | Self-measured state; simple analytic solutions allowed. | None; requirement R3. |
| Generated assets | closed; Anka, chat, 2026-11-06 | Preserve the mechanism; exact asset matching is not required. | None; requirement R8. |
| Run length | `profile` | `smoke`: at most 1 s; `full`: complete schedule and declared long hold. | Same selected case; coverage stays visible. |
| Full ASV workload | explicit opt-in approved; Anka, chat, 2026-11-06 | Unfiltered runs must skip full cases. Implementation and runtime measurement remain pending. | Full selected cases; no shortened hold presented as equivalent. |
| ASV status direction | closed; Anka, chat, 2026-11-06; implementation pending | Use failure rate, per-case failure flags, and missing coverage. Preserve unknown results and rate denominators; see D6 in decisions.md. | Same cached result; no new physical measurement or ASV extension. |
| Positive checks and limits | pending | Define positive task evidence and per-case or shared acceptance limits; neither is supplied by this review. | Existing measurements and declared task scope, with identical inputs. |

## 1.8 Open questions for the reviewer

1. Should the first PR retain explicit Incomplete results, or add the missing positive task checks and justified acceptance limits now?

Explicit opt-in and lower-is-better ASV measures are approved but not implemented. Overall Pass rules stay open. The small fixes landed at the input code head; [decisions](decisions.md) records their scope and the user's requirements.
