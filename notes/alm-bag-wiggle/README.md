# Pinned-bag ALM comparison

Open `results-triangle-bending/index.html` for the current comparison. The
earlier bending-only comparison remains in `results/index.html`. Each full MP4 has five consecutive six-second
chapters, each showing ALM off on the left and ALM on on the right.

Current particle ALM covers both triangle membrane and dihedral bending in this
bag; this scene has no springs or tetrahedra. See
[the triangle/bending derivation](../alm-triangle-bending.md). The geometric
stretch/shear/bend scores are deformation measures, not force residuals.

The May 26 scene file was untracked and is absent from this VM. `run_case.py`
reconstructs it using the parent bag geometry and contents from Newton commit
`be3e604b` (`example_vbd_bag_franka_pickup_two_way.py`) and the parameters and
pin-motion description saved in AI-Docs at
`AI-Logs/Newton/tasks/rigidity-bag-wiggle/2026-06-08-parse-notes.md`.
The archived parent in `sources/bag_parent.py` has only its USD import moved into
the loader to meet current lint rules. Its bag and contents builder is reused
without a ground or gripper. The exact original pin scheduling cannot be
verified; this reconstruction samples displacement and analytic velocity once
per frame and reapplies them before each substep. Both comparison modes use
this same schedule and the current solver branch.

`metrics.py` preserves the original metric functions from AI-Docs
`AI-Logs/Newton/tasks/rigidity-bag-wiggle/rigidity_trial.py`. Each sample includes
the rest state as frame zero. Seed 42, 60 fps, 360 frames, 10 substeps, 10 VBD
iterations, triangle damping 0.1, bending stiffness 200, and bending damping
0.02 are fixed. ALM uses rho scale 1.0 and retains history throughout each run.
The new comparison uses the material-based metric floor for triangle and bending
rows. The earlier comparison used inertia-only bending metrics and no triangle ALM.

Run from the isolated worktree with Warp 1.17.0, NumPy, usd-core, matplotlib,
pyglet, imageio, and imageio-ffmpeg installed in its environment:

```bash
source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh alm-bag-wiggle compete
export PYTHONPATH="$PWD"
export WARP_CACHE_PATH="$PWD/.cache/warp-alm-bag"
for stiffness in 1000 10000 100000 1000000 10000000; do
    for mode in off on; do
        uv run --no-sync python notes/alm-bag-wiggle/run_case.py \
            --stiffness "$stiffness" --alm "$mode"
    done
    uv run --no-sync python notes/alm-bag-wiggle/render_pair.py \
        --stiffness "$stiffness"
done
uv run --no-sync python notes/alm-bag-wiggle/build_review.py
```

Rendering replays saved states and does not rerun physics. All panels use the
same fixed camera and show the same simulation time. Videos, screenshots, and
compressed trajectories stay local and are ignored by Git. CSV, JSON, SVG, HTML,
and source files are retained in the branch. This is an equal-iteration study,
not an equal-time benchmark or an exact replay of the missing May fixture.
