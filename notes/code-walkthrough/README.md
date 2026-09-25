# LIDO implementation walkthrough

The HTML explains the implemented mixed trainer in four chapters: network,
physical objective and gradients, local frames and differentiable assembly,
and deterministic data augmentation. It contains no behavior-history section.

Build from the Newton worktree root:

```bash
uv run --no-sync python notes/code-walkthrough/build.py
```

Output is `generated/code-walkthrough/`. The main page embeds all explanations,
styles, interaction code, and source excerpts. The bundle also includes complete
line-numbered source snapshots and a ZIP download. The builder checks every
local file link and line anchor. It reads source from the fixed Git revision in
`build.py`; the accompanying configuration supplies the documented dimensions
and sampling ranges. Building this document does not execute or change training.

Published location:
https://ankachen.com/artifacts/learned-intrinsic-solver/code-walkthrough/index.html

To document a different implementation, review the prose and excerpt ranges
alongside the code before changing the pinned revision or configuration.
