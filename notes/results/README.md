# Research checkpoint results — 2026-09-24

These are copied measurements from the completed experiments, not results
from the planned 8,192-state training campaign. The current design and
interpretation are in [v1-DECISIONS.MD](../v1-DECISIONS.MD#22-current-design-and-research-checkpoint).

- [Float32 gradient checks](float32-gradients.json): physical loss, fusion adjoint,
  and frozen-frame derivatives. Reproduce with
  `python -m experiments.learned_intrinsic_solver.validate_gradients`.
- [GPU smoke-run configuration](smoke-training-config.json),
  [full report](smoke-training-report.json), [training CSV](smoke-training.csv),
  and [fixed validation CSV](smoke-validation.csv): 16 training states, eight
  validation states, 64 Adam updates, batch one, one solver iteration.
  Source: `experiments/learned_intrinsic_solver/train_smoke.py`.
- [Checkpoint reload check](smoke-checkpoint-reload.json): saved validation
  energies reproduced without further optimizer updates.
- [L40 batch capacity](batch-capacity.json): two full float32 steps per batch;
  batch 72 passed and 73 exhausted memory. Repeated smoke-run samples measure
  memory capacity, not generalization or multi-GPU throughput.

The experiment test suite passed all 123 tests, including CUDA tests, on this
checkpoint before commit:

```bash
source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh learned-intrinsic-commit-check occupy
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync python \
  -m unittest discover -s experiments/learned_intrinsic_solver/tests
```

Working precision is float32 with TF32 and AMP disabled. Sparse factorization
and its forward/adjoint solves use CPU SciPy; Torch work and Adam use CUDA.
Small float64 reference checks are numerical diagnostics, not the training
precision. CPU smoke/preflight results are not substituted for GPU results.

The full local smoke run, plots, and binary checkpoints remain in
`generated/training/lame_gpu_single_step_001/`. The
[browser report](https://ankachen.com/artifacts/learned-intrinsic-solver/training/index.html)
contains plots and checkpoint downloads; the
[project index](https://ankachen.com/artifacts/learned-intrinsic-solver/index.html)
links the other visualizations. Generated media, datasets, caches, and binary
model checkpoints are excluded from Git. Artifact links depend on the current
hosting deployment; the compact measurements above are preserved in this branch.
