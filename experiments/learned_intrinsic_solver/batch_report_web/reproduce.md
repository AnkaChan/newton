# Reproduce the 100 float32 round trips

Run from the Newton worktree root with the existing NumPy/SciPy/Matplotlib development environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m experiments.learned_intrinsic_solver.round_trip_batch --seed-start 0 --count 100
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m experiments.learned_intrinsic_solver.round_trip_batch_report
uv run --no-sync python -m unittest discover \
  -s experiments/learned_intrinsic_solver/tests -p test_round_trip_batch.py -v
```

Output defaults to `experiments/learned_intrinsic_solver/generated/round_trip_100/`.
The runner accepts `--output`, `--seed-start`, `--count`, `--tolerance`, and
`--resume`. A configuration digest prevents resuming results with mismatched
settings. Every seed is attempted; exceptions and nonconverged solves are
recorded rather than discarded. Report generation reads existing results and
does not rerun the batch. It performs one additional tolerance audit on the
worst sample, then caches that audit.

## Main experiment

- Seeds 0–99, canonical 10 by 10 by 40 grid, h=0.025 m, strength 0.5.
- Same automatic default multiscale hierarchy for every seed; full z-min face fixed.
- Existing deterministic augmentation is computed first, then positions are
  quantized to float32 at the encoding input boundary. This is explicit; the
  generator's random draws and original screening remain float64.
- Origins, center gradients F, polar frames R, local axes U, reconstruction
  matrix, RHS, Krylov vectors, x iterates, and final positions are float32.
- Polar SVD uses SciPy's native single-precision LAPACK SGESDD. NumPy's linalg
  SVD is avoided because it can internally compute in double precision.
- SciPy LSQR receives explicit float32 zero x0; omitting x0 would allocate a
  float64 solution. Matrix-vector products and output have dtype assertions.
  A test traces all observed x/u/v/w/dk arrays, confirming float32 iterations.
- SciPy may use Python double scalars for norm/condition/stopping bookkeeping.
  No float64 vector refinement or solve is used in the main pipeline.
- All 100 actual solves use atol=btol=1e-7, conlim=1e7, no damping or
  regularization, and at most 10,000 iterations per world coordinate.
- Decoding uses only origins, R/U, rest topology/dimensions, and explicit
  boundary positions. Original free corners and analytic null projections are
  never supplied to reconstruction.

## Error measurement

Analysis converts stored float32 arrays to float64 after the solve so that
measurement cancellation does not hide solver or output-rounding error.

Corner RMS means RMS Euclidean xyz error across all 4,961 shared corners.
The normalized equation residual is recomputed directly from the stored
float32 matrix, solver displacement, and RHS, accumulated in float64. Center
rows are divided by h; gradient and center row residuals are dimensionless.
This is distinct from gradient/center errors evaluated on the final rounded
positions, and from the corner recovery error.

The unresolved-mode diagnostic projects the error onto alternating xy layer
modes in float64 after reconstruction. The orthogonal remainder is reported
as row-space error. It includes float32 numerical effects; the report does not
assume that the full recovery error is explained by representation null modes.

`report.json` contains summary statistics and all case data; `metrics.csv`
contains one row per attempted seed. `samples/seed_NNN.json` records parameters,
precision, convergence, and errors. `states/seed_NNN.npz` stores exact float32
inputs/outputs and encoded quantities. `selected-cases/` uses the existing
round-trip viewer for worst, lower-middle, and best cases by RMS. Plots are
Matplotlib SVG files. These results concern only these sampled initial shapes;
they are not universal error bounds or simulation-accuracy measurements.
