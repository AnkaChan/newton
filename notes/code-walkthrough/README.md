# LIDO implementation walkthrough

The HTML explains the implemented mixed trainer in five chapters:

1. Transformer and solver network (`network.html`): node, edge and
   conditioning encoders, the masked graph-attention block with FiLM, the
   bounded nine-value correction head and the per-cell sigmoid step.
2. Energy and gradients (`energy.html`): the stable Neo-Hookean implicit-Euler
   objective with metric damping, its position gradient, and the LeCO loss with
   the material-aware energy floor.
3. Frames and assembly (`frames.html`): closest proper rotation frames with the
   clamped-face tie-break, local axes, the sparse least-squares assembly and
   its transposed backward solve.
4. Data augmentation (`data.html`): deterministic reset, material, shape and
   velocity sampling, candidate initializers and the trajectory pool.
5. Gradient input (`gradient.html`): how the energy gradient with respect to
   the corner positions is projected through the assembly adjoint, normalized
   and packed into the network input together with the optimizer history.

The page describes the implementation as it is and contains no
behavior-history section.

## Files

- `page.html` is the shell: navigation, overview, the "Follow one query"
  flow, the glossary, chapter placeholders (`@@NETWORK@@` ... `@@GRADIENT@@`)
  and the implementation map.
- `network.html`, `energy.html`, `frames.html`, `data.html`, `gradient.html`
  are the chapter fragments. Excerpt placeholders
  `<div data-code="file.py:START:END"></div>` and links of the form
  `source/<file>.py.html#L<line>` refer to exact current line numbers under
  `experiments/learned_intrinsic_solver/`.
- `main.js` holds the flow stages and reader tools; `style.css` the styling.
- `campaign-config.json` is a copy of `generated/training_v3_config.json` and
  supplies the documented dimensions and sampling ranges.
- `build.py` pins the Git revision (`REVISION`), renders excerpts and source
  pages, and validates every local link and line anchor.

## Rebuild

Run from the Newton worktree root:

```bash
uv run --no-sync python notes/code-walkthrough/build.py
```

Output is `generated/code-walkthrough/` (pass `--output DIR` for another
location). The main page embeds all explanations, styles, interaction code,
and source excerpts. The bundle also includes complete line-numbered source
snapshots, `snapshot.json` with source hashes, and a ZIP download. The builder
reads source from the pinned revision with `git show`, so the revision must be
present in the repository; building this document does not execute or change
training.

## Published location

https://ankachen.com/artifacts/learned-intrinsic-solver/code-walkthrough/index.html

To document a different implementation, review the prose, the excerpt ranges
and the source anchors alongside the code before changing the pinned revision
or the configuration copy.
