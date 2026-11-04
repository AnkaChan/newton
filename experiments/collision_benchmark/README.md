# Collision-fidelity benchmarks

This suite preserves the historical 44-scene benchmark and adds a synced-main
DAT/OGC campaign for nine reduced-stiffness cloth manipulation recipes.
The recorded benchmark commit is `9470121d2eeb995aa9488e4978d1b4abf92b0e1c`,
based on upstream main `351b9d9f521fbb9f8f729445a4bff929e8521197`.
It uses Warp `1.19.0+cu12`; Warp's Python version reports `1.19.0`.
Use the [synced reproduction guide](https://kanna.ankachen.com/artifacts/newton-dat-ogc-synced-main/evidence/reproduce.md)
for the exact source pin, verified downloads and selected nine-case commands.

The current registry also includes a passive 52-card stacking scene with no
moving ball. It is an added scene, separate from the recorded nine-case
manipulation comparison.

The current empty-shopping-tote recipe uses a 3 mm closed finger opening.
The recorded nine-case campaign used 4 mm. Use its pinned reproduction guide
to replay that older comparison.

## Synced-main DAT and OGC campaign

Use `run_dat_ogc` on this branch. The older `fast_math` entry point below
retains its historical physics guard and belongs to the
[original benchmark branch](https://github.com/AnkaChan/newton/tree/ankac/collision-benchmark-compare-20261019).
Current main does not provide that fork's ALM particle-elasticity option.

```bash
UV_PROJECT_ENVIRONMENT=.venv-dat-ogc uv sync --python 3.12 --extra examples --locked
uv pip install --python .venv-dat-ogc/bin/python \
  -r experiments/collision_benchmark/requirements.txt
```

Warp 1.19's default wheel requires a CUDA 13 driver. On a CUDA 12 driver,
install the official CUDA 12 build of the same release:

```bash
uv pip install --python .venv-dat-ogc/bin/python \
  'https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
```

Run a full native pouch attempt with its original 128-row self-contact storage:

```bash
UV_PROJECT_ENVIRONMENT=.venv-dat-ogc uv run --no-sync python \
  -m experiments.collision_benchmark.run_dat_ogc m2_franka_pouch \
  --out outputs/main-dat-ogc/m2_franka_pouch
```

The selected synced cohort uses the archived 512-row wrapper for pouch, patch,
dress and H1 towel. The direct command above does not select that override.
The reproduction guide also supplies the exact two H1 initial states.
A complete timeline does not establish successful retention or release.

The entry point fixes standard VBD elasticity, native fast math off, rigid-soft
DAT on, full-surface contacts, and the 10x contact-coefficient cap. It uses the
merged OGC edge-force rule from PR 4936. This does not implement every part of
OGC. It preserves the existing recipe's timestep, iterations, material, and
hand commands. Each result records the effective settings and exact sources.

Newton owns collision detection in this path. The adapter copies prescribed
kinematic hand endpoints into the native solver's input. DAT does not truncate
those hand motions or certify their continuous sweep. The first native step
initializes rigid pose history at its input endpoint. Native-main solver timing
includes owned collision detection and capacity latches. The previous
`3768a475` native-main campaign uses the same timing boundary. The historical
`11702e64` benchmark excluded external collision detection; its solver timings
do not have the same scope.

Use `--h1-initial-cache /path/to/common-start-directory` to import a verified
historical jacket or towel initial condition. The importer checks saved hashes,
rest geometry, body layout, and robot asset bytes. It keeps original files and
creates a new cache under the current solver identity. The towel's recorded
closed-thumb correction applies once. These arrays specify the initial scene;
they are never scoring references. Without this option, H1 uses a new native
preparation, which can differ from the saved campaign.

Use a new output directory for each run. `--duration 1 --no-video` is a setup
check only. Default durations preserve the full 60-second hold. The synced
campaign compares upstream `351b9d9f` with the selected `3768a475` attempts,
using the same Warp `1.19.0+cu12` build, capacities and recorded inputs.
Both already enable DAT and the merged OGC edge rule. This is an upstream
revision rerun, not an isolated DAT/OGC experiment. Comparisons with the
historical `11702e64` campaign also change Warp and the timing scope.

The runner stops on nonfinite state or detected contact-buffer overflow.
Sticky counters cover every owned global and soft-self detection pass.
They do not cover per-body adjacency buffers. These recipes prescribe all
robot bodies as kinematic; native adjacency excludes those bodies.

Check the current adapter on CPU:

```bash
UV_PROJECT_ENVIRONMENT=.venv-dat-ogc uv run --no-sync python -m unittest \
  experiments.collision_benchmark.test_main_dat_runtime \
  experiments.collision_benchmark.test_comparison_h1_import \
  experiments.collision_benchmark.test_comparison_h1_adjustments \
  experiments.collision_benchmark.test_contact_stiffness
```

The historical compiler-scope tests below require the original fork's ALM
module. They do not apply to this current-main entry point.

## Poker cards without the moving ball

Run the shipped card setup with its sphere body, collider and motion removed:

```bash
UV_PROJECT_ENVIRONMENT=.venv-dat-ogc uv run --no-sync python \
  -m experiments.collision_benchmark.run_dat_ogc s4_poker_cards_no_ball \
  --out outputs/main-dat-ogc/s4_poker_cards_no_ball
```

This scene keeps all 52 cards, seed 42, card mass and material, platform and
ground. The platform is a world-fixed box. Standard VBD uses fast math off,
DAT and the merged OGC edge rule. Settings are 60 Hz, ten substeps (dt 1/600 s),
and ten iterations. The original demo uses twenty substeps. The benchmark
keeps the ten-substep limit. The existing 10x contact cap lowers the platform
coefficient from 5,000,000 to 100,000. Soft and ground coefficients remain
100,000.

The default run lasts 65 s. It allows 0–5 s for initial drop and stacking,
then measures J* and V* over 120 half-second windows in [5, 65] s. This fixed
interval does not assert that the cards have settled. `--hold-seconds` does
not extend this passive scene. Static-support overlap covers the whole run.
It does not measure card-card penetration. Grasp, retention, transport and
release criteria do not apply. Do not include this scene in the manipulation
task-success denominator.

[Card scene video and measurements](https://kanna.ankachen.com/artifacts/newton-poker-cards-no-ball/index.html)

## Shopping bag grip

The empty tote (`m3_franka_tote_empty`) uses a 3 mm nominal closed opening,
reduced from 4 mm. The authored fingertip faces give an actual gap of 2.960 mm.
Both hands keep the same wrist poses and lift/carry/place motion. The 80 mm
release opening, bag mesh and material, friction, and solver settings stay
fixed. Other bag recipes keep their 4 mm opening.

```bash
UV_PROJECT_ENVIRONMENT=.venv-dat-ogc uv run --no-sync python \
  -m experiments.collision_benchmark.run_dat_ogc m3_franka_tote_empty \
  --out outputs/shopping-bag-grip
```

This runs the full 67.5 s schedule, including a 60 s hold. It uses ten
substeps at 30 Hz (dt 1/300 s), forty iterations and a 30,000 N/m contact
coefficient cap. Retention and placed-object release need separate checks;
final separation alone can also mean that the object fell earlier.

[Shopping bag video and measurements](https://kanna.ankachen.com/artifacts/newton-shopping-bag-grip/index.html)

## Historical benchmark configuration

The following subsections apply to the original benchmark branch linked above,
not this nine-case native-main campaign. Keep its ALM/compiler guards and
historical commands on that original branch.

The original branch compares standard VBD with ALM **particle elasticity only**.
Contact settings stay fixed. It also tests the compilation scope of fast-math PR
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

### Install

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

### Run one scene

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

### Reduce contact stiffness

Use this explicit configuration to cap contact stiffness at ten times the
declared elasticity reference. The default command without these flags keeps
the original settings so the published results remain reproducible.

```bash
uv run --no-sync python -m experiments.collision_benchmark.fast_math \
  m2_franka_pouch --mode vbd --math off \
  --contact-max-ratio 10 --contact-tet-policy numeric \
  --out outputs/contact-cap10/m2_franka_pouch
```

The cap only lowers values. It applies to `soft_contact_ke` and every
`shape_material_ke` before preparation and solver construction. Rigid-soft
contact uses their arithmetic mean, so changing only one side is insufficient.
Zero coefficients remain zero. Elastic materials, friction, damping, time
steps, and iteration counts stay unchanged.

The tested cap10/numeric configuration includes two recorded H1 pose corrections.
The jacket changes only the initial open right-wrist position. Its closed pinch
and later poses stay unchanged. The H1 towel closes its thumbs slightly further;
its open start, wrist path, timing, and release commands stay unchanged. Normal
fresh preparation and every original geometry and joint limit check still run.
The result identifies these changes in `initial_common_start.recipe_adjustment`.
Other cap values and commands without a cap keep the original H1 poses.
These two corrected scenes are not stiffness-only comparisons.

Cloth uses its positive membrane coefficients. Rods use their stretch/shear
coefficients; bending and robot actuator gains are excluded. Mixed scenes use
the smallest declared reference because soft contact has one global coefficient.
The original mesh, including zero-stiffness boundary triangles on tet solids,
does not change.

Tet elasticity is in Pa, whereas contact stiffness is in N/m. The `numeric`
option compares parameter numbers and explicitly records this unit difference.
It is not a physical stiffness ratio. The `mesh` option uses the dimensional
proxy `min(E) * median(unique rest tet edge length)`, with
`E = mu * (3 * lambda + 2 * mu) / (lambda + mu)`. This proxy is not an assembled
nodal stiffness or a proof that a grip will hold.

The result records the policy, reference terms, and original/effective contact
values in `effective_contact_materials`. H1 preparation uses a separate cache
for each policy and implementation. Original results and caches remain intact.
Ratios must be finite, positive, and at most ten. Use a fresh process and output
folder for each setting. A different policy must not reuse an old initial
fingerprint as a matched pair.

The cap is a configuration bound. Grip retention must be checked with the
actual acquisition, full hold, and release. The
[reduced-stiffness results and videos](https://kanna.ankachen.com/artifacts/newton-contact-stiffness-cap10/index.html)
cover the nine changed cloth scenes among the 15 previously completed
standard-VBD/math-off operations. All nine corrected recipes retain through
the observed 60-second hold. Six solid cases already fall below the numeric
cap and were not rerun. Prepared pinches do not establish acquisition.

Hold retention is separate from the complete operation. The reduced-stiffness
Franka dress remains caught during release. The empty tote releases, then falls
off the table. Large local stretch and static-support overlap also remain in
some cases. No overall fidelity pass or repeatability guarantee is assigned.
The report preserves the rejected jacket setup and the towel's original hold
failure beside the successful pose corrections.

### Select a scene

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

### Compare fast math

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

### Read the measurements

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

### Source map and checks

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
