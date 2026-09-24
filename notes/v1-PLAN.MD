# V1 Plan: Learned Intrinsic Solver

**Historical V1 snapshot.** The architecture, backend, dataset and training
status below record the earlier baseline. Current V2 decisions and implemented
mixed-pool training are described in [v2-plan.md](v2-plan.md) and
[decision 39](v1-DECISIONS.MD#39-v2-mixed-pool-trainer-implementation--2026-09-24).
The new pipeline's verification is in progress; no new campaign has launched.

Build a learned optimizer for the original implicit-Euler problem of a
three-dimensional deformable solid. The current proof of concept is a pinned
10×10×40 grid of 0.025 m cubic hexahedra with canonical rest geometry. Its
physics is compressible logarithmic Neo-Hookean elasticity, integrated at
eight Gauss points per cell, plus the unchanged physical inertia. Lamé lambda,
Lamé mu, and density are explicit inputs. Contact and general free-body fusion
remain future work. [v1-DECISIONS.MD](v1-DECISIONS.MD#22-current-design-and-research-checkpoint)
records the complete current design, measurements, and next training settings.

Each occupied cell is a network node. Shared grid corners determine the global
shape. Compute each current cell's center deformation gradient from its
corners, extract its polar rotation as a local frame, and express the three
transformed unit material axes in that frame. Axis lengths retain stretch;
angles retain shear. These nine numbers describe an affine cell deformation,
not every possible eight-corner warping. Rest size is separate: cubic cells
use their edge length and fixed material topology. The network predicts
changed axes in the same input frame.

Inputs comprise nine axis components, 24 components of inertial_offset,
six exposed-face flags, eight fixed-corner flags, and five normalized
material/size/timestep scalars. Each inertial_offset is the original free
inertial prediction Y minus the current corner position, expressed in this
cell's frame and divided by rest size. It is a vector, not a distance.
Candidate perturbations never redefine Y. Three transformer blocks with
hops [1, 1, 1], width 128, and four heads attend to 26 neighboring cells plus
self. Missing slots are masked. Learned edge bias/value use relative offsets,
relative frames, and neighbor axes transported into the receiving frame;
there is no fixed geometric attention prior. FiLM supplies conditioning.
A bounded nine-component correction and one shared object step size define
the proposed local axes. The baseline has 865,942 trainable parameters.

Convert proposed axis changes into world coordinates and fit displacement
gradients at all eight Gauss points, with stiffness times rest-volume weights.
Add the solved displacement to current shared corners, enforcing pins exactly
by elimination. A zero correction preserves the current warped shape.
Repeating one axis change across the eight integration points makes fusion
well constrained for this pinned problem but does not give the network extra
outputs for independent corner-warping corrections. A residual energy floor
is possible. A separate semi-implicit rigid predictor reuses Newton rigid
integration, with inertia recomputed from the current mass distribution.
Its pose target guides fusion initialization only; the implicit-Euler Y and
physical inertia remain unchanged.

The same network can repeatedly update any valid pinned candidate for one
fixed physical problem. Recompute frames every iteration but freeze their
derivatives. Torch currently computes frames, features, energy, network, and
Adam on CUDA in float32, with TF32 and mixed precision disabled. A cached CPU
SciPy sparse factorization supplies fusion; a custom transpose-solve adjoint
propagates first-order gradients through fusion inputs. Newton Model/State
setup and the rigid predictor remain on CPU. A Warp frame implementation,
fully GPU sparse solve, and differentiation through physical State commits
are future work.

Train first with one learned iteration per query, minimizing normalized energy
change through fusion. The completed 64-update GPU smoke test verified the
pipeline but reduced energy on only four of eight held-out queries. The next
campaign, **recorded but not implemented or started**, uses 8,192 physical
training states, 512 fixed validation states, fresh candidate perturbations
each epoch, and a target batch of 16 per GPU on four GPUs. Monitor epoch loss,
raw energy, and descent rate; reduce learning rate on a plateau and save
resumable checkpoints. Start with a 200-epoch limit and permit plateau stopping
after at least 30 epochs under the detailed recorded rule. A poor flat curve
is stalled training, not successful convergence. Expand the unroll only after
reliable held-out descent and multi-iteration checks. There is no learned-output
repair or general descent guarantee.

**Future boundary work:** start with actual voxel solids. Later compare subgrid
occupancy with surface samples containing rest positions, normals, and areas.
A small transformer can compress variable sample sets into one fixed-size
geometry feature; compare with a shared point encoder and area-weighted pooling.
Use one main network for boundary and interior cells, with explicit masks,
material volume, and connectivity. Optimizer-gradient/history inputs are also
future experiments, not part of the current packed baseline.

The four-GPU computation path has now passed a separate bounded diagnostic,
including an independent gradient comparison and exact checkpoint continuation.
See decision 25 for measurements and the required NCCL setting on this VM.
