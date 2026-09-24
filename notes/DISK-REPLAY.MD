# Retaining physical training states on disk

The archive keeps every collected physical timestep input. A replayed input is
solved again with the current network and the physical energy; an earlier
network prediction is never a ground-truth target. No training run is started
by these utilities, and the production epoch trainer is not yet connected to
the consecutive-step curriculum or this archive.

## Stored data

Each record contains float32 positions and velocities, its trajectory ID and
physical step index, prescribed corner positions, applied particle forces,
gravity, and source metadata. Zero particle forces take no array storage.
The input is committed **before** the learned solve, so an input that triggers
a solver failure is retained too. There is no success label implied by a row.

A shared physical context contains the canonical rest grid, cell topology,
fixed-corner indices, timestep, Lamé parameters, density, and exact native
particle masses. Identical contexts are stored once under a content hash;
context arrays are compressed. Network weights and computation graphs are
never archived. Contexts do not depend on the network architecture or training
loss policy. Source iteration count and loss settings are recorded separately.

Material sampling is deterministic: `sample_material(seed)` independently
samples lambda and mu log-uniformly from 1,000 to 1,000,000 Pa, and density
log-uniformly from 100 to 10,000 kg/m³. Sample once per object, then use that
material for the whole trajectory. A trajectory cannot switch physical
contexts halfway through; the store rejects such an append.

```python
from dataclasses import asdict
from experiments.learned_intrinsic_solver.material_sampling import sample_material
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model

material = sample_material(material_seed)
model = build_newton_hex_model(rest, fixed_indices, gravity=(0, -9.81, 0), **asdict(material))
```

Use one `rank_N.sqlite3` file per training rank. A whole batch of timestep
starts is one durable SQLite transaction. Existing records are never evicted
or overwritten. Reusing a trajectory ID and step index fails instead of
silently replacing data. Continue a trajectory with its original ID and a new
`step_offset`, or give a genuinely new trajectory a new ID. An interrupted
optimizer checkpoint does not automatically delete newer archived records.

## Retain a rollout

The existing `PhysicalRollout` still performs the solves. Wrap its window
iterator with `retain_windows`. It preserves the default one-physical-step
gradient boundary and does not change the original positions/velocities.

```python
from experiments.learned_intrinsic_solver.disk_replay import DiskReplayStore
from experiments.learned_intrinsic_solver.replay_rollout import retain_windows

# rollout is a PhysicalRollout; x and v are float32 [B, P, 3] tensors.
# Use distinct run/rank/trajectory IDs. The metadata callback is evaluated
# immediately before each solve, so it sees any preceding optimizer update.
with DiskReplayStore("generated/replay/run_001/rank_0.sqlite3") as archive:
    for window in retain_windows(
        rollout,
        archive,
        x,
        v,
        trajectory_ids=trajectory_ids,
        physical_steps=8,
        iterations=4,
        gradient_window=1,
        step_offset=next_physical_step,
        provenance=lambda index: {
            "optimizer_update": optimizer_update,
            "source_checkpoint": checkpoint_sha256,
        },
        trajectory_metadata=[{"physical_seed": seed} for seed in physical_seeds],
    ):
        # Existing caller decides whether/how to train this window.
        # Finish its backward pass before advancing to the next window.
        process_window(window)
```

`forces` may be `[B,P,3]` or a schedule `[physical_steps,B,P,3]`.
`fixed_positions` is optional and otherwise comes from the initial pinned
corners. The archive records the applied force and prescribed positions for
each step, not a pointer to a mutable schedule. It records timestep inputs;
the final output becomes another archived input when the trajectory advances
again. No inference or training is performed just by opening the store.

## Load and replay

```python
from experiments.learned_intrinsic_solver.disk_replay import DiskReplayDataset
from experiments.learned_intrinsic_solver.replay_rollout import (
    build_replay_rollout,
    replay_state,
)

with DiskReplayDataset(rank_paths) as dataset:
    state = dataset.sample(1, seed=123)[0]
    context = dataset.get_context(state.context_id)

# current_network is supplied by the caller, not loaded from the archive.
replay = build_replay_rollout(
    context,
    current_network,
    gravity=state.metadata["gravity"],
)
result = replay_state(replay, context, state, iterations=8)
# result.objective can be differentiated through the new solve.
```

Reconstruction restores the exact material and masses and recomputes the
inertial prediction and initial solver candidate from the saved physical X/V.
The supplied context ID, topology, timestep, material, masses and gravity are
checked. A state paired with another material is rejected. Updated network
weights and an explicitly chosen new loss policy are allowed. For efficiency,
callers can reuse a compatible replay solver across states; building a native
model for every sampled record is unnecessary.

The reader loads only requested state arrays. Its compact row-ID index uses
eight bytes per state, plus a small per-file index. Sampling is uniform and
seeded. Appends do not change the sampling population until `refresh()`.
Use `context_id=...` to restrict sampling to a compatible physical context.
Replay/live/fresh sampling ratios are a trainer decision, not a storage rule.

Open a separate reader in each process; SQLite connections are not shared
between DataLoader workers. Reopening a dataset captures all currently
committed rows. Exact sampling continuation across a trainer restart will
require saving its population snapshot and sampling state in the trainer
checkpoint; that integration is not part of these storage helpers.

## Durability and scope

Stores use SQLite WAL with FULL synchronization. A successful `append` has
committed; a duplicate or invalid batch leaves previous records intact.
Unknown database schemas are rejected. Use SQLite's backup API for a snapshot
while connections are open. For a standalone file copy, close all connections,
including readers; copying a database while omitting its active WAL can lose
committed records in the copy.

The raw X/V pair for 4,961 vertices occupies 119,064 bytes before database and
metadata overhead. Nonzero force arrays add 59,532 bytes. Contexts and pins
add storage too. Retention is unlimited by policy, but available disk space is
finite: write failures propagate to the caller rather than silently dropping
training states.

Disk retention accepts finite states; it does not make the solver inversion
tolerant. The current frame extraction and logarithmic Neo-Hookean energy
still reject inversions. Signed-axis frame handling and an energy continuation
through inversion remain design work.

Implementation: `experiments/learned_intrinsic_solver/disk_replay.py`,
`replay_rollout.py`, and `material_sampling.py`.
