# Collision-fidelity benchmarks

This experimental suite contains the 44 current collision benchmark scenes.
It compares standard VBD with ALM **particle elasticity only**. Contact settings
stay fixed. It also tests the compilation scope of fast-math PR
[#4830](https://github.com/newton-physics/newton/pull/4830) on this same solver
version. The solver base is Newton `11702e64`; Warp is `1.17.0`.

The suite includes cloth and garments, empty and loaded bags, soft solids,
cables, soft-soft stacks, compression, impact, and analytic free fall.
Manipulation scenes use Franka, Allegro, or H1 hand assets. Some recipes still
fail manipulation. The suite preserves those failures.

[Measured results and videos](https://kanna.ankachen.com/artifacts/newton-fast-math-comparison/index.html)
are separate from this source package. Packaging changes make file paths and
asset downloads portable. They do not change the measured material, mesh,
contact, timestep, iteration, or motion settings. Earlier numerical results
refer to the archived source hashes, not this later packaging commit.

## Install

Use Linux, Python 3.12, an NVIDIA CUDA GPU, and a compatible NVIDIA driver.
The measured runs used L40 GPUs. Run these commands from the repository root:

```bash
git clone --branch ankac/collision-benchmark-compare-20261019 \
  https://github.com/AnkaChan/newton.git newton-collision-benchmark
cd newton-collision-benchmark
uv sync --python 3.12 --extra examples --locked
uv pip install -r experiments/collision_benchmark/requirements.txt
```

Use `uv run --no-sync` below so the recording packages remain installed.
Video capture needs OpenGL. On a Linux machine without a display, install
`xvfb` through the system package manager. Use `--no-video` when only numerical
measurements are needed. First use can download pinned robot asset revisions
and compile Warp kernels.

## Run one scene

Start with two soft pads. This scene runs for seven simulated seconds:

```bash
uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  s1_pads_2_10substeps --mode vbd --math off \
  --out experiments/collision_benchmark/outputs/pads-vbd
```

Run the same scene with ALM particle elasticity:

```bash
uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  s1_pads_2_10substeps --mode alm --math off \
  --out experiments/collision_benchmark/outputs/pads-alm
```

For the downward-facing hand that picks a ball from the ground, use
`r3_allegro_ground_sphere` as the case ID. For a Franka soft-pad grasp, use
`r6_franka_pad`. Use `--device cuda:1` to select another GPU.

The runner writes `results.json`, `frame-observations.json`, `progress.json`,
`video.mp4`, `poster.jpg`, and a contact sheet. With `--no-video`, it writes
the numerical files only. Each run needs a new output directory; the runner
refuses to replace an existing result.

The `fast_math` entry point also runs the standard baseline with `--math off`.
It includes the measured campaign's duck initialization repair and stops at
the first recorded nonfinite physical step. Use this entry point for the full
set. The lower-level `run_comparison` module predates those two safeguards.

Manipulation runs retain the measured 60-second hold. A quick setup check can
use `--duration 1 --no-video`. Such a short run is not a full benchmark result.
Keep default durations and hold times for full comparisons. Every scene uses
at most ten substeps. The saved result states the timestep and iteration count.

## Select a scene

Read [cases.json](cases.json) for all 44 IDs, settings, and metric applicability.
List the IDs without importing a simulation scene:

```bash
uv run --no-sync python -c \
  'from experiments.collision_benchmark.comparison_cases import load_cases; print("\n".join(load_cases()))'
```

Use `--help` on either runner for all options. Run each case and mode in a new
process, because the experiment installs solver hooks for that process.

To run all 44 scenes in both elasticity modes, use this sequential loop.
It continues after a failed scene and keeps that scene's failure record.
The full set can take hours; compilation and the 60-second holds add cost.

```bash
for case in $(uv run --no-sync python -c \
  'from experiments.collision_benchmark.comparison_cases import load_cases; print(" ".join(load_cases()))'); do
  for mode in vbd alm; do
    uv run --no-sync python -m experiments.collision_benchmark.fast_math \
      "$case" --mode "$mode" --math off \
      --out "experiments/collision_benchmark/outputs/all/$case/$mode" \
      || echo "Failed: $case $mode; inspect its results.json"
  done
done
```

## Compare fast math

Both commands use the same scene and solver source. The adapter changes only
the selected compiler flags. Preparation uses math off in both runs.

```bash
uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  s1_pads_2_10substeps --mode vbd --math off \
  --out experiments/collision_benchmark/outputs/pads-math-off

uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  s1_pads_2_10substeps --mode vbd --math on \
  --initial-reference experiments/collision_benchmark/outputs/pads-math-off/results.json \
  --out experiments/collision_benchmark/outputs/pads-math-on
```

`--initial-reference` checks the initialized state and model against the first
run. It does not use a reference trajectory or reference score. Use `--mode alm`
for the ALM elasticity pair. Use the same GPU and settings for both arms.
Alternate the arm order for repeated trials. The adapter verifies the pinned
Warp version and solver source before running. Benchmark-only child commits
are allowed; changed solver code requires a new explicit experiment baseline.

This tests the PR's compilation scope on the experimental ALM solver base.
It does not replay all changes in the newer upstream PR branch. For rod-only
scenes, the particle elasticity flag has no effect; selected rigid kernels can
still change with fast math.

## Read the measurements

- `measurements.finite` records nonfinite state and the first failure time.
- `measurements.stability` records J* and V* over complete fixed 0.5-second
  windows. J* is the worst local linearly detrended displacement RMS. V* is the
  worst local relative-speed RMS, including mean speed. Raw J* uses metres.
- `measurements.penetration` records nominal particle-shell gaps to declared
  static planes and boxes. This is not full robot, self-contact, or cable
  penetration coverage.
- `measurements.freefall` records analytic velocity error where applicable.
- `timing` records CUDA solver time. It excludes collision detection, drivers,
  observation, rendering, and setup.
- Fast-math runs also write conservative hand/object separation evidence in
  `task_measurements`. Full separation can prove contact loss during a required
  hold. Bounding-box overlap does not prove a successful grasp.

Lower jitter after a dropped object comes to rest is not a better hold.
Finite completion is not a fidelity pass. Limits remain unset for some
criteria; missing measurements are not zero. Compare the same complete finite
cases for each metric mean, and report failed runs separately. Repeated runs
can differ even with the same compiler flag.

## Source map and checks

- [run_comparison.py](run_comparison.py): baseline and ALM elasticity runner.
- [comparison_cases.py](comparison_cases.py): scene registry and hold timing.
- [comparison_metrics.py](comparison_metrics.py): numerical measurements.
- [fast_math](fast_math): compiler adapter and sampled task evidence.
- [assets](assets): generated meshes, prepared jacket, provenance and licenses.
- [recipes](recipes): fixed H1 grasp recipes used by the measured scenes.

Run the CPU checks from the repository root:

```bash
uv run --no-sync python -m unittest discover \
  -s experiments/collision_benchmark -p 'test_*.py'
```

The generated assets and existing failed or superseded recipe code remain
available. The 44-case registry selects the current comparison set.
