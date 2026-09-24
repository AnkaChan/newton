# Reproduce the global-shape round trip

From the Newton worktree root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m experiments.learned_intrinsic_solver.round_trip
uv run --no-sync python -m unittest discover \
  -s experiments/learned_intrinsic_solver/tests -p test_round_trip.py -v
```

The default output is `experiments/learned_intrinsic_solver/generated/round_trip/`.
Use `--output PATH` to select a different directory. The default grid is
10 by 10 by 40 cells with spacing h=0.025 m. NumPy and the existing SciPy
development dependency are sufficient; no GPU or VBD simulation is used.

## Exact pipeline

1. Start with actual global shared-corner positions of a deformed grid.
2. Call the existing `compute_cell_frames(rest, positions)` encoder. For each
   cell it computes the origin (mean of all eight corners), the center gradient
   F (opposite-face centroid differences divided by h), its polar rotation R,
   and local axes U=R^T F.
3. Pass encoded R/U, optionally encoded centers, canonical rest topology and
   spacing, and the explicit positions of the material z-min fixed face to the
   decoder. No original free-corner positions are supplied to the solve.
4. Reconstruct F=R@U. Build the exact shared-corner linear measurement operator
   matching the encoder. Eliminate fixed corners and solve for free displacement
   from rest using undamped LSQR with a zero initial guess.
5. Compare the reconstructed positions with the original after solving.

Two decoders are reported: R/U plus the fixed end, and origins/R/U plus the fixed
end. The latter is the default visualization. Center residuals are divided by h
so both gradient and center rows have dimensionless residuals with equal weights.
Both use atol=btol=1e-13, no damping or smoothing, and a 20,000-iteration limit.

The system is rank deficient. Zero-start LSQR chooses the minimum free
displacement among solutions. This is a selection rule, not information that
was present in the stored frame representation. A small measurement residual
does not imply the original shared corners were uniquely recovered.

## Cases

- Current multiscale augmentation, seeds 0 and 7; default strength 0.5.
- Original independent-cell augmentation, seed 0, deformation amplitude 0.15,
  projected onto shared corners with the existing 12-edge objective. This uses
  material/global coordinates before the VBD gallery's rigid world transform.
- Identity canonical shape.
- A known affine shear/stretch that preserves the fixed face.
- A secondary invisible corner warp, delta_x=0.001*(-1)^(i+j)*k/nz. It preserves
  all stored center quantities and boundary positions but changes free corners.

All cases actually run the encoder and decoder; no reconstruction is hard-coded
to its expected answer. The affine case is retained even though minimum
displacement need not reproduce it exactly on an odd corner cross section.

## Outputs and metric definitions

`report.json` records separate corner, gradient, center and boundary errors,
polar factorization accuracy, LSQR stop codes and iteration counts, and a small
grid rank audit. Corner RMSE is the RMS Euclidean xyz distance over shared
corners. Gradient errors are dimensionless; position errors are in meters.

`data/CASE.npz` contains the canonical topology and rest positions, original
positions for comparison, encoded origins/R/U, explicitly supplied fixed-face
positions, and both reconstructed corner arrays. The decoded solver does not
use the archived original free positions. `source/round_trip.py` archives the
implementation used for the report.

The browser's error-gain slider affects visualization only. It draws
`original + gain*(recovered-original)` in the reconstructed panel; numeric
metrics and error colors always refer to the unscaled actual result.

The Three.js library and controls are copied from the existing inspector's
vendored assets, with their license. The page requires an HTTP server for ES
modules; the published artifact supplies one. The experiment leaves both the
existing frame inspector and VBD gallery untouched.
