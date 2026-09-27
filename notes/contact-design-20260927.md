# LIDO contact handling, version 1: ground plane and artificial static contact points

Status: design approved in conversation on 2026-09-27; awaiting review of this
written spec before an implementation plan is drafted. Applies to the learned
intrinsic solver under `experiments/learned_intrinsic_solver/` at commit
`f4298870` and later.

## 1. Purpose and scope

The solver currently minimizes a contact-free implicit-Euler objective for one
clamped hexahedral body. Version 1 adds contact between the body's surface and
static partners so the learned optimizer can be trained and evaluated on
scenes with a floor and obstacles. It establishes the contact abstraction,
energy, network input and metrics that later versions reuse.

In scope:

- one unified partner representation (point, normal, radius, stiffness,
  damping, friction, kind, self flag) for every contact partner;
- body surface samples at exposed face centers;
- partners generated artificially per scene: one ground plane and a set of
  static contact points with normals;
- contact energy with Newton's contact force law (quadratic normal penalty,
  gap-rate damping, smooth isotropic friction) added to the physical objective;
- contact tokens processed by a small attention block and pooled into the
  owning cell's input;
- scene generation, validation metrics and tests.

Deferred, behind the same partner record: detection through Newton's
`CollisionPipeline` (soft particle, edge and face records against shapes),
self-contact through Newton's `TriMeshCollisionDetector`, several deformable
bodies, and the ablation of the contact attention block against pooled
hand-crafted features.

## 2. Decisions recorded from the brainstorm (2026-09-26 to 2026-09-27)

| Topic | Decision |
|---|---|
| Abstraction | Every partner is the same record; no distinction between obstacle, other body and self except the self flag and kind. |
| Body samples | One sample per exposed face: centroid and outward face normal from the four current corners, radius `r = 0.5 h`. |
| Partners in v1 | A ground plane plus artificial static contact points with normals; no Newton detection yet. |
| Network | Contact tokens per record, a small attention block per cell over its tokens, pooled into a fixed-width vector appended to the cell's state features. Cells remain the only output. |
| Energy | Newton's law: `ke/2 (r - gap)^2` penalty, gap-rate damping `kd`, IPC-style smooth isotropic friction with constant normal force. Cubic penalty and barrier were considered and not selected. |
| Detection cadence | Once per physical step on the step-start shape with a margin; the pair list, partner points and friction anchors are frozen for the K network calls of that step. |
| Scenes | Clamped beam with the existing augmentation plus a ground plane at random height and 0 to 64 static points near the beam; parameters sampled per scene from seeded streams. |
| Newton reuse | Formulas follow Newton's `_compute_body_particle_contact_force` and `compute_friction`; an oracle test checks the Torch energy gradient against the Warp force law so the later switch to Newton detection changes nothing in the energy. |

## 3. Geometry

### 3.1 Surface samples

`cell_exposed_faces` (data.py) gives the six faces per cell in material order
(-x, +x, -y, +y, -z, +z); local corner order is 000, 001, 010, 011, 100, 101,
110, 111 with z fastest. A new module `contact_geometry.py` provides:

- `exposed_face_corners(rest) -> (cell_index [S], face_index [S], corners [S, 4])`
  built once at construction; S is the sample count (about 1,400 for the
  10x10x40 grid). Corner order per face is fixed so that the rest-state normal
  points outward.
- `sample_points(positions [B, P, 3], corners [S, 4]) -> [B, S, 3]`: centroid,
  the mean of the four corners.
- `sample_normals(positions, corners) -> [B, S, 3]`: `normalize(d1 x d2)` with
  `d1 = x_2 - x_0`, `d2 = x_3 - x_1` (the two diagonals), sign fixed by the
  corner order; degenerate faces (norm below 1e-12) fall back to the rest normal.

Both are differentiable in the corner positions, so contact forces reach the
corners and, through the fusion adjoint, the axis coordinates.

### 3.2 Partner record

Dataclass `ContactPartners` stored per scene in the trajectory payload and per
context in `MixedHexSolverStep`:

| Field | Shape | Meaning |
|---|---|---|
| `plane_point`, `plane_normal` | [3], [3] | ground plane; `plane_normal` is `(0, 1, 0)` (gravity acts along -y) |
| `plane_present` | bool | some scenes have no floor so contact-free behavior stays in the training distribution |
| `point_positions` | [N, 3] | artificial static points, N in [0, 64] |
| `point_normals` | [N, 3] | unit normals |
| `point_radii` | [N] | lateral radius of influence `r_p` |
| `ke`, `kd`, `mu` | scalars | contact stiffness [N/m], damping [N s/m], friction coefficient |

`kind` is 0 for the plane, 1 for a static point; 2 is reserved for self. The
self flag is 0 for every v1 partner.

### 3.3 Pairs

A pair is (sample s, partner q, kind). Its gap is `gap = (x_s - p_q) . n_q`
where `p_q` is the plane's foot point of `x_s` for the plane and the static
point for kind 1. A static point acts as a small disk: a pair is a candidate
only when the lateral distance `| (x_s - p_q) - gap n_q | < r_p`.

Amendment 2026-09-27 (review of the implementation): the disk is one-sided
with thickness `r`. A static-point pair additionally requires `gap >= -r`, so a
sample whose sphere no longer reaches the disk plane from behind (a far face of
the body) is not pulled toward the point and a point pair starts at most `2 r`
deep. The plane keeps its unbounded half-space.

Second amendment 2026-09-27 (campaign diagnostic): a pair of either kind is a
candidate only if the partner normal opposes the sample's outward face normal,
`n_q . n_s < 0`. Without it, the deepest pairs at the initial candidate were
100 % static-point pairs with grazing normals (median `|cos(n_s, n_q)|` 0.29)
that no displacement of the face can resolve, and a floor within the search
band of a side face paired with that face although it cannot push on it.
`detect_contacts` takes the face normals as the optional `sample_normals`
argument; `MixedHexSolverStep.prepare` passes the step-start normals.

## 4. Detection

Performed once per physical step by `_TrajectoryFactory.reset` and `advance`
on the step-start positions `X_start` (after `MixedHexSolverStep.prepare`), on
the CPU preparation workers:

1. compute `x_s(X_start)` for all samples;
2. plane: candidate if `plane_present` and `gap < r + margin`;
3. points: brute-force distances `[S, N]`; candidate if lateral distance
   `< r_p` and `gap < r + margin`; keep the nearest `M_pair = 4` per sample;
4. write `contact_pairs` (sample index, partner index, kind) with at most
   `S * (1 + M_pair)` rows, padded with -1, plus `contact_partner_point [Q, 3]`
   and `contact_partner_normal [Q, 3]` for the Q kept pairs.

Second amendment 2026-09-27: step 1 also evaluates the face normals
`n_s(X_start)` (`contact_geometry.sample_normals`), and steps 2 and 3 keep a
candidate only when `n_q . n_s < 0` (section 3.3); dropped candidates do not
occupy the `M_pair` slots. The step-start normal is frozen with the pair list,
so a face that turns away during the inner iterations simply has zero energy.

`margin = r + |v_s| dt` where `v_s` is the sample's step-start velocity, so a fast
free end cannot cross the search band within one step (the tip reaches about
4 m/s after 0.4 s of free fall, 13 mm per step against r = 12.5 mm). The
friction anchor is the sample position at step start, recomputed from `physical_positions` in the payload, so nothing
else is stored. The pair list does not change between the K network calls of a
step; a pair that separates during the step simply has zero energy.

## 5. Contact energy

New module `contact_energy.py`, function
`contact_energy(positions, partners, pairs, physical_positions, dt) -> HexLossTerms-compatible tensor [B]`,
evaluated on the fused positions inside `MixedHexSolverStep._energy` and added
to `total`; `HexLossTerms` gains a `contact` field (default None for legacy
callers).

For each kept pair, with `x = x_s(X)`, `x_0 = x_s(X_start)`, partner point `p`
and normal `n`, `d = r - (x - p) . n` (penetration depth), and
`delta = x - x_0` (partner velocity is zero in v1):

- normal penalty: `E_n = ke/2 * relu(d)^2`;
- damping: `E_d = kd/(2 dt) * relu(-(n . delta))^2`, active only while
  approaching, matching Newton's damping force `-(kd/dt) (n . delta) n`;
- friction: `u = delta - (n . delta) n`, `f_n = ke * relu(d)` treated as a
  constant (detached), `eps_u = friction_epsilon * dt`, and
  `E_f = mu * f_n * f0(|u|)` with the IPC smoothing
  `f0(y) = -y^3/(3 eps_u^2) + y^2/eps_u + eps_u/3` for `y < eps_u` and
  `f0(y) = y` otherwise, whose derivative `f1(y)/y` equals Newton's
  `(-y/eps_u + 2)/eps_u` inside the smoothing band and `1/y` outside.

`friction_epsilon = 1e-2` (Newton's default). Units are joules. The energy is
finite for every finite configuration and needs no detection during the inner
iterations. It enters `E_before`, `E_after`, the projected gradient input and
the free-corner residual through the existing code paths.

The energy floor formula is unchanged in v1; if near-rest contact scenes show
loss saturation, add `ke r^2 * S` to the material-aware scale as a documented
follow-up.

Oracle test: a Warp kernel wraps Newton's `_compute_body_particle_contact_force`
for one record; the Torch gradient of `contact_energy` with respect to `x`
must match the returned force to float32 tolerance for penetrating, separating,
sliding and sticking cases.

## 6. Network input

Schema version 4 in `features.py`:

- `CONTACT_TOKEN_DIM = 19`: contact point in the owning cell frame (3), partner
  point in the cell frame (3), partner normal in the cell frame (3), gap / r (1),
  approach rate `-(n . (x - x_0)) / (r)` over the current step (1, so the network
  can decelerate before penetration), r_p / r (1), log ke, log(1 + kd), mu (3),
  kind one-hot (plane, point, self) (3), self flag (1). Positions are relative to
  the cell center and divided by h.
- `CONTACT_POOL_DIM = 16` appended to the state features together with one
  scalar `contact_count / M_cell`, so `STATE_FEATURE_DIM = 61 + 17 = 78`.
- Conditioning gains `log(ke h / E)`, `log(1 + kd)`, `mu` per object:
  `CONDITIONING_DIM = 9`.

Owning cell: the sample's cell. Tokens per cell are capped at
`M_cell = 24` (six faces times four pairs), padded and masked.

New module `contact_network.py`, `ContactEncoder(nn.Module)`:

1. token encoder `Linear(19, 64)`, SiLU, `Linear(64, 64)`;
2. one masked self-attention block over the cell's tokens (width 64, 2 heads,
   feed-forward 64 to 256 to 64, pre-LayerNorm), reusing the masking pattern of
   `IntrinsicTransformerLayer`;
3. masked mean and max pooling, concatenated (128), then `Linear(128, 16)`
   zero-initialized.

Cells without tokens produce the zero vector and count 0, so at initialization
and for contact-free cells the network is the schema-3 network with 17 zero
inputs. `IntrinsicSolverNetwork` gains a constructor flag `contact_tokens: bool`
and accepts the token tensor `[B, C, M_cell, 19]` with its mask; when the flag
is off the constructor is unchanged, so the existing tests keep running.

## 7. Scenes and sampling

`contact_scene.py`, seeded from `[master_seed, seed, 2203]` per trajectory:

- `plane_present` with probability 0.8; plane height `y0` uniform in
  `[-0.35, -0.02]` m relative to the beam's rest y-minimum (the free end sags
  under gravity along -y and reaches it in a fraction of scenes);
- `N` static points uniform in {0, ..., 64}, positions uniform in the box that
  extends the rest body by 0.10 m in x and z and by 0.35 m in -y, excluding the
  rest bounding box grown by one cell `h` (amendment 2026-09-27: the original
  box contained the body, so 95% of default scenes started with points inside
  the beam pairing with its far faces at about 25 r depth); normals uniform on
  the sphere, then flipped to point toward the beam's rest center; `r_p`
  uniform in `[0.5 h, 2 h]`;
- second amendment 2026-09-27: points are drawn one at a time from a spawned
  child of the scene seed (so `kappa`, `beta`, `mu`, the plane and the count
  keep their draws) and rejected while (a) any rest surface sample would be a
  detection candidate of the disk in the widened band `-r <= gap < r + h`,
  `r = 0.5 h` (the one-cell box clearance only kept the point outside the
  body; with random normals and `r_p` up to `2 h` the disk plane still cut the
  body in 43 % of default scenes, and 61 % of scenes had a rest candidate),
  (b) the normal does not oppose the outward normal of the nearest rest sample
  within 60°, `n . n_face <= -cos 60°` (redraw the normal up to eight times,
  then the position), or (c) `z < z_min + h`, in front of or on the clamped
  face where no free corner can move; 1000 position draws without success raise.
  Realized on the canonical grid over 200 seeds: 0 % of scenes with a rest
  candidate (was 61 %), mean point count unchanged at 32.3, 35 % of position
  draws rejected;
- `ke = kappa * E * h` with `kappa` log-uniform in `[0.1, 10]` (E is the
  sampled Young's modulus, so contact stiffness scales with the material);
  `kd = beta * ke * dt` with `beta` uniform in `[0, 1]`; `mu` uniform in
  `[0, 1]`.

Second amendment 2026-09-27 (trainer default, `MixedTrainConfig`): the plane
height range is `[-0.15, -0.005]` m. In 512 validation seeds of the first
campaign the floor never penetrated (the plane sat at least 0.02 m below the
rest y-minimum while the initial displacements are 0.05-0.4 h RMS), so the floor
part of the contact model was untested by the cheap validation; the shallower
range lets soft and moderately stiff beams reach the floor within the longest
training horizon (128 steps, 0.43 s). The sampler's own default keeps the
original range for reproducibility of earlier scenes.

Scene generation targets contact-rich data: the plane height and point box are
chosen so that, under the current augmentation and gravity, at least half of the
trajectories make contact within their horizon; the trainer records both the
scene fraction (`contact_scene_fraction`: plane or points present) and the
realized fraction (`contact_realized_fraction`: trajectories with at least one
detected pair in the epoch) so the balance can be tuned. All values are recorded
in the trajectory metadata and in checkpoints through the payload, as the
material is today. The ranges are provisional and are
listed as such in the plan's implementation record when implemented.

## 8. Training and validation

Training is unchanged: the LeCO per-update loss on the total objective, which
now includes contact. Detection runs on the preparation workers, so GPU work
per network call grows only by the contact energy and the contact encoder.

Validation adds, per iteration of the 100-iteration check and per physical
step of the cheap and full-horizon checks:

- maximum and mean penetration depth over active pairs, in units of `r`;
- number of active pairs and of cells with contacts;
- contact energy and the contact share of the free-corner residual norm.

Failed samples stay visible as today. The dashboard gains a penetration curve
next to the residual curve. Checkpoint selection keeps the residual metric; the
final maximum penetration is reported beside it and becomes an eligibility rule
(no sample deeper than r at the end of the full-horizon check) only after the
first contact campaign shows what values are attainable.

## 9. Tests

- Geometry: centroid and outward normal on the rest grid; gradients of
  `sample_points` and `sample_normals` against finite differences; degenerate
  face fallback.
- Detection: plane and point candidates on hand-built configurations; margin;
  the `M_pair` cap keeps the nearest; determinism across runs.
- Energy: the Warp oracle test of section 5; finite differences in float64;
  zero energy and zero gradient when no pair penetrates; damping only while
  approaching; friction potential matches the IPC `f1` derivative.
- Network: schema-4 widths; contact-free scenes give the same output as a
  schema-3 network with matching weights plus zero contact inputs at
  initialization; gradients reach the contact encoder; masks exclude padding.
- Physics sanity: a single free cell dropped onto the floor, solved by plain
  gradient descent on the total objective (no network), comes to rest with
  penetration below r and sliding stops under friction; the same scene solved
  by Newton's VBD with matching parameters gives the same rest height within
  float32 tolerance.
- Pipeline: a two-epoch CPU run on a tiny grid with a floor produces finite
  losses, records penetration metrics, resumes from a checkpoint with the scene
  parameters intact.

## 10. Follow-ups (not in v1)

1. Replace section 4 with Newton's `CollisionPipeline` soft-contact stream
   (particle, edge and face records against shapes) once the beam's surface
   triangles are added to the Newton model; the partner record and everything
   downstream stay the same.
2. Self-contact with Newton's `TriMeshCollisionDetector` and its log-barrier
   self-contact law, kind 2 with the self flag set.
3. Several deformable bodies per scene.
4. Ablation: contact attention block versus pooled hand-crafted features.
5. Contact-aware energy floor term.

## 11. Open points for review

None block implementation. Provisional numeric choices in sections 4 to 7
(`M_pair`, `M_cell`, token width, scene ranges, `kappa`, `beta`) are routine
and will be recorded when implemented; say so if any should be fixed
differently now.
