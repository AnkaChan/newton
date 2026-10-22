# Recorded H1 recipe inputs

These small source and JSON snapshots preserve the four selected H1 setups.
They contain no recorded simulation state. `comparison_h1.py` loads each recipe
in its own module namespace. In particular, the sheet uses its recorded older
`h1_bag_scenes.py`; the bag uses its prepared body-rim wrapper.

`h1/manifest.json` keeps the original byte hashes and the current packaged byte
hashes. It also checks the original Python syntax trees and JSON values. Import
ordering and formatting may differ; the scene calculations, numerical values,
joint commands, guards, and phase schedules remain the same. Shared helpers
come from the portable benchmark package. Their current hashes are part of each
common-start cache identity. Original H1 asset bytes are verified after download
at the pinned revision.

The jacket, sheet, and towel prepare a shared initial state with standard
elasticity. The bag starts in its declared prepared contact. Both comparison
arms restore the same state buffers and command arrays, then use fresh solver
histories. The cache is an initial condition, not a reference solution. It is
stored outside this source directory; `NEWTON_COMPARE_H1_CACHE` overrides its
location.

After a formatter changes snapshot whitespace, refresh hashes from the checkout:

```bash
uv run --no-sync python -m experiments.collision_benchmark.recipes.update_manifest
```

This command refuses changed Python syntax trees or JSON values. It does not
replace the original provenance hashes. The Linux comparison runner loads the
snapshots through its adapter; do not execute `record-scene.py` directly.
