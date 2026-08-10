# Implementation Review — Core Code Walkthrough (2026-08-09)

Companion to `02_current_implementation.md` (which describes *what* exists);
this one shows the actual code of every load-bearing piece, with commentary
and review findings. All excerpts are verbatim from branch
`ankac/principal-stretch-dev` @ HEAD; line references follow each block.
Docstrings are elided from excerpts where they only repeat what the
commentary says.

Reading order: §1 decoder -> §2 polar gradients -> §3 network -> §4 training
step -> §5 data -> §6 review findings -> §7 test coverage.

---

## 1. The decoder (`torch_solver.py`)

### 1.1 Precomputation — one Laplacian, one Cholesky, forever

```python
def _build_J(Dm_inv: torch.Tensor) -> torch.Tensor:
    ...
    T = Dm_inv.shape[0]
    J = torch.zeros(T, 4, 3, dtype=Dm_inv.dtype, device=Dm_inv.device)
    J[:, 1, :] = Dm_inv[:, 0, :]
    J[:, 2, :] = Dm_inv[:, 1, :]
    J[:, 3, :] = Dm_inv[:, 2, :]
    J[:, 0, :] = -(J[:, 1, :] + J[:, 2, :] + J[:, 3, :])
    return J
```
(`torch_solver.py:28, 35-41`, docstring elided)

`J[t, a, c] = dF[t, :, c] / dx[tets[t, a], :]` — the per-tet shape-function
gradient. Row 0 is minus the sum of the others because F depends only on
edge vectors `x_a - x_0`. Everything downstream (F computation, RHS
assembly, Laplacian) is expressed through J, so the index convention is
stated once at the top of the file (`torch_solver.py:9-15`) and pinned by
the Warp cross-check test (§7).

```python
    det_inv = torch.linalg.det(Dm_inv)
    w = 1.0 / (6.0 * det_inv)
    if (w <= 0).any():
        raise ValueError("non-positive rest volumes — check tet orientation")

    J = _build_J(Dm_inv)  # (T, 4, 3)

    # Dense assembly of L on rest mesh: L = sum_e w_e * (J_e @ J_e^T) scattered.
    # K[t, a, b] = w[t] * sum_c J[t, a, c] * J[t, b, c]
    K = torch.einsum("tac,tbc->tab", J, J) * w[:, None, None]  # (T, 4, 4)
    L = torch.zeros(n_verts, n_verts, dtype=dtype, device=device)
    rows = tets[:, :, None].expand(-1, -1, 4)  # (T, 4, 4)
    cols = tets[:, None, :].expand(-1, 4, -1)
    L.index_put_((rows.reshape(-1), cols.reshape(-1)), K.reshape(-1), accumulate=True)

    mask = torch.ones(n_verts, dtype=torch.bool, device=device)
    mask[pinned] = False
    free = torch.where(mask)[0]

    L_ff = L[free][:, free]
    if tikhonov > 0.0:
        L_ff = L_ff + tikhonov * torch.eye(free.numel(), dtype=dtype, device=device)
    L_fp = L[free][:, pinned]
    L_ff_chol = torch.linalg.cholesky(L_ff)
```
(`torch_solver.py:109-132`)

Points worth noticing:

- `L = sum_e w_e J_e J_e^T` is the **rest-mesh** Laplacian of the ARAP
  energy — it does not depend on `x` or `S*`, which is the entire reason the
  global step is one pre-factored triangular solve instead of a fresh
  factorization per step.
- The free/pinned partition implements Dirichlet BCs by elimination:
  `L_ff x_f = rhs_f - L_fp x_p`. **All absolute-position information enters
  through `L_fp x_p`** — remove the pins and `L` is singular (translation
  nullspace). This is the exact spot the momentum-anchoring work (direction
  C, decided 2026-08-09) will modify: replace the pin partition with rank-6
  momentum constraints.
- `L` is dense `(V, V)` fp64 and Cholesky is dense: fine at 225-4225 verts
  (a 3900^2 factor is 122 MB), a hard wall around ~10k (finding R1).
- `tikhonov` defaults to 0 and no caller sets it — dead parameter (R12).

### 1.2 The local-global loop — the heart of the method

```python
    batch = S_target.shape[:-3]
    if x_init is None:
        x = state.rest_q.expand(*batch, -1, -1).clone()
    else:
        x = x_init.to(dtype=dtype, device=device).expand(*batch, -1, -1).clone()
    x[..., state.pinned, :] = pinned_targets

    bc_rhs = torch.einsum("fp,...pd->...fd", state.L_fp, pinned_targets)
    # index_add_ needs a flat leading dim; the mesh is shared so flatten/restore.
    flat_v = (-1, state.n_verts, 3)
    idx = state.tets.reshape(-1)

    for _ in range(n_iters):
        F = torch.einsum("tac,...tad->...tdc", state.J, x[..., state.tets, :])
        # Local step: R = polar(F S*^T).
        R = polar_rotation(F @ S_target.transpose(-1, -2))
        contrib = torch.einsum("...tdc,tac->...tad", R @ S_target, state.J) * state.w[:, None, None]
        rhs = torch.zeros(*batch, state.n_verts, 3, dtype=dtype, device=device)
        rhs.reshape(flat_v).index_add_(1, idx, contrib.reshape(-1, state.n_tets * 4, 3))
        # Global step: one solve against the pre-factored rest-mesh Laplacian.
        # Fold the batch into the RHS columns instead of broadcasting: a batched
        # B against an unbatched factor makes cholesky_solve materialise the
        # (F, F) factor per batch element (22 GB at 180 frames x 4k verts).
        b = rhs[..., state.free, :] - bc_rhs  # (*batch, F, 3)
        bf = b.reshape(-1, *b.shape[-2:])  # (B, F, 3)
        b_cols = bf.permute(1, 0, 2).reshape(bf.shape[1], -1)  # (F, B*3)
        x_cols = torch.cholesky_solve(b_cols, state.L_ff_chol)
        x_free = x_cols.reshape(-1, bf.shape[0], 3).permute(1, 0, 2).reshape(b.shape)
        x_new = x.clone()
        x_new[..., state.free, :] = x_free
        x_new[..., state.pinned, :] = pinned_targets
        x = x_new

    return x
```
(`torch_solver.py:212-246`)

This is block coordinate descent on
`E(x, R) = sum_e w_e/2 ||F_e(x) - R_e S*_e||_F^2`:

- **Local step** — for fixed `x`, the optimal `R_e` is the polar rotation of
  `F_e S*_e^T` (Procrustes, per tet, closed form). Differentiable through
  `polar_rotation` (§2), so gradients w.r.t. `S*` flow through both the
  rotation *and* the RHS.
- **Global step** — for fixed `R`, minimising over `x` is the linear system
  assembled by `index_add_` (adjoint of the F gather). One elliptic solve:
  positional information propagates **mesh-wide** in a single iteration.
  What converges slowly (~0.98/iter) is the *coupled* x-R fixed point, not
  information reach — this distinction was the takeover review's central
  correction, and is why the inertial warm start (below) dominates output
  quality at practical iteration counts.
- Everything batches over an arbitrary leading dim; the mesh (hence the
  Cholesky factor) is shared. The batch-fold in the global step is the
  2026-08-04 fix — before it, `cholesky_solve` silently materialised the
  factor per batch element.
- Note `x_new = x.clone()` inside the loop: keeps each iterate a distinct
  autograd node, so the unrolled chain back-propagates cleanly (this is the
  "differentiable decoder" — training's gradient path runs through every
  iteration).

### 1.3 State -> stretch: `compute_S_from_x`

```python
def compute_S_from_x(state: SolverState, x: torch.Tensor) -> torch.Tensor:
    ...
    x_tet = x[..., state.tets, :]  # (..., T, 4, 3)
    F = torch.einsum("tac,...tad->...tdc", state.J, x_tet)
    R = polar_rotation(F)
    S = R.transpose(-1, -2) @ F
    return 0.5 * (S + S.transpose(-1, -2))
```
(`torch_solver.py:150, 161-165`, docstring elided)

The network's input representation. `S = sym(R^T F)` is rotation-free by
construction; the explicit symmetrisation kills the ~1e-8 skew residue the
Newton iteration leaves. Used in three places: feature building, residual
base `S* = S_t + delta`, and recomputing state between rollout steps /
blocks — in the K>1 training chain this sits **inside** the autograd graph,
which is why the SVD-free polar mattered (SVD backward is ill-conditioned
whenever singular values coincide, i.e. near rest, i.e. most of the mesh).

### 1.4 The warm start — part of the method, not an optimisation

```python
def inertial_predictor(
    state: SolverState, x_t: torch.Tensor, x_prev: torch.Tensor, pinned_targets: torch.Tensor
) -> torch.Tensor:
    ...
    x0 = 2.0 * x_t - x_prev
    x0 = x0.clone()
    x0[..., state.pinned, :] = pinned_targets
    return x0
```
(`torch_solver.py:168-170, 179-182`, docstring elided)

Three lines, 10x on the decoder floor (1.05e-2 -> 1.02e-3 m at 10 iters,
measured with oracle stretches). Because the local-global iteration
contracts at ~0.98, the output at any practical budget is "warm start plus
a small correction" — so the warm start must already carry the inertial
motion. This is also where global rotation/translation *actually* comes
from in the current pipeline: the previous two frames. The decoder never
solves for a global transform; per-tet rotations re-emerge each iteration
from `polar(F S*^T)` evaluated at the current iterate.

---

## 2. Exact polar gradients (`polar.py`)

### 2.1 Forward — scaled Newton, no SVD on the hot path

```python
def polar_rotation_forward(M: torch.Tensor, iters: int = 6) -> torch.Tensor:
    ...
    with torch.no_grad():
        R = M.clone()
        for _ in range(iters):
            R_inv_t = _inv3(R).transpose(-1, -2)
            n_r = R.flatten(-2).norm(dim=-1, keepdim=True)[..., None]
            n_i = R_inv_t.flatten(-2).norm(dim=-1, keepdim=True)[..., None]
            gamma = torch.sqrt(n_i / n_r.clamp(min=1e-300))
            R = 0.5 * (gamma * R + R_inv_t / gamma)

        # Guard: Newton converges to a reflection when det M < 0, and to nothing
        # useful when M is singular.  Fall back to SVD on those elements only.
        bad = ~torch.isfinite(R).all(dim=(-2, -1)) | (torch.linalg.det(M) <= 0)
        if bad.any():
            R = R.clone()
            R[bad] = _svd_polar(M[bad])
    return R
```
(`polar.py:82, 84-99`, docstring elided)

Higham's scaled Newton `R <- (gamma R + gamma^-1 R^-T)/2`: pure batched
matmul/elementwise (the 3x3 inverse `_inv3` is the analytic adjugate), so
it is throughput-friendly where batched cuSOLVER SVD was the decoder's
dominant cost. Quadratic convergence; measured 5 iterations to fp64
round-off at 17:1 anisotropy. The `det <= 0` fallback never fires for
`M = F S*^T` with `det F > 0` and SPD `S*` — see finding R3 for the edge.

### 2.2 Backward — the Sylvester equation in closed form

```python
class _PolarRotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, iters):
        R = polar_rotation_forward(M, iters)
        ctx.save_for_backward(R, R.transpose(-1, -2) @ M)
        return R

    @staticmethod
    def backward(ctx, grad_R):
        R, S = ctx.saved_tensors
        S = 0.5 * (S + S.transpose(-1, -2))
        a = _axial(R.transpose(-1, -2) @ grad_R)
        eye = torch.eye(3, dtype=S.dtype, device=S.device).expand_as(S)
        tr_S = S.diagonal(dim1=-2, dim2=-1).sum(-1)
        K = tr_S[..., None, None] * eye - S
        b = torch.linalg.solve(K, a.unsqueeze(-1)).squeeze(-1)
        return 2.0 * R @ _cross_matrix(b), None
```
(`polar.py:122-138`)

The math (derivation in the module docstring, `polar.py:29-47`): with
`M = R S`, the rotation differential is `dR = R Omega` where the skew
`Omega` solves `Omega S + S Omega = R^T dM - dM^T R`. For 3x3 symmetric
`S`, the identity `[b]_x S + S [b]_x = [(tr(S) I - S) b]_x` collapses the
Sylvester equation to a single 3x3 solve on axial vectors. Key robustness
property: `K = tr(S) I - S` has eigenvalues `(s2+s3, s1+s3, s1+s2)` — sums
of principal stretches — so it is **positive definite for any physical
deformation**, including exactly at rest where the SVD backward blows up
(coincident singular values). Verified against `torch.autograd.gradcheck`
and finite differences across the deformation range (§7).

This replaced a surrogate that detached the SVD and re-routed gradients
through `M inv(sym(R0^T M))` — exact only at `S = I`, 22-27% Jacobian error
at 50% stretch. That surrogate is preserved in `diag_polar_grad.py` purely
as a measurement target.

---

## 3. The network (`model.py`) — deliberately minimal

```python
class StretchNet(nn.Module):
    """3-layer MLP per tet. No graph layers (neighbor info injected via mean pool)."""

    def __init__(self, in_dim: int = 28, hidden: int = 64, max_delta: float = 0.6):
        super().__init__()
        self.max_delta = max_delta
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),
        )
        # Zero-init last layer so the initial prediction is the base state
        # (identity in absolute mode, S_t in residual mode).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor, S_base: torch.Tensor | None = None) -> torch.Tensor:
        ...
        delta = self.max_delta * torch.tanh(self.net(feat))  # (..., T, 6)
        D = vec_to_sym(delta)
        if S_base is None:
            eye = torch.eye(3, dtype=feat.dtype, device=feat.device).expand_as(D)
            return eye + D
        return S_base + D
```
(`model.py:79-97, 109-114`, docstrings elided)

- ~6k parameters, shared across tets. The receptive field question
  (notes/01) lives entirely in what `feat` contains — the architecture
  itself sees one tet at a time.
- `0.6 * tanh(...)`: smooth output bound keeping `S*` in the region where
  `polar(F S*^T)` is well-posed. Near-identity in the small-signal regime
  (`tanh'(0) = 1`), saturating for extremes. The 0.6 is an inherited,
  unablated heuristic — several times the data's largest `|S - I|`
  component, well below the SPD-loss edge (R6).
- Zero-init + residual base means training starts at exactly "stretch
  unchanged", the correct dynamics prior; the network only ever learns a
  *correction* field.

Feature assembly (the non-local part is the last two entries):

```python
    feats = [
        sym_to_vec(S_t_c),  # 6
        sym_to_vec(S_prev_c),  # 6
        (gravity / G_SCALE).expand((*bT, 3)),  # 3
        f_ext_tet / F_SCALE,  # 3
        (mu / MAT_SCALE)[:, None].expand((*bT, 1)),  # 1
        (lam / MAT_SCALE)[:, None].expand((*bT, 1)),  # 1
        pin_flag[:, None].expand((*bT, 1)),  # 1
        sym_to_vec(S_n_c),  # 6
        (n_neigh / 4.0)[:, None].expand((*bT, 1)),  # 1
    ]
    return torch.cat(feats, dim=-1)  # (..., T, 28)
```
(`model.py:154-165`)

`S_n_c` is the mean over face-adjacent tets (<= 4, `build_face_adjacency`,
`model.py:52-76`) — a single mean-pool, i.e. a 1-ring receptive field. All
S inputs are identity-centred so "rest" is the zero vector. In the current
datasets gravity/mu/lam are constant -> 5 of 28 dims are dead inputs (R5).
The kNN test (notes/02 §8.6-7) measured what these 28 numbers can and
cannot determine.

---

## 4. The training step (`train.py`)

### 4.1 Window sampling, curriculum, input noise

```python
        if step < curriculum_end:
            k_target = 1 + int((args.max_rollout - 1) * step / curriculum_end)
        else:
            k_target = args.max_rollout
        ...
        x_prev = x_gpu[b[:, 0]]  # (B, V, 3)
        x_t = x_gpu[b[:, 1]]
        if args.noise_std > 0.0:
            # Perturb the input state (targets stay clean) so training visits the
            # off-manifold states rollout inevitably produces.
            x_prev = x_prev + args.noise_std * torch.randn_like(x_prev)
            x_t = x_t + args.noise_std * torch.randn_like(x_t)
            x_prev = x_prev.clone()
            x_t = x_t.clone()
            x_prev[:, solver.pinned] = pinned_targets
            x_t[:, solver.pinned] = pinned_targets
            S_prev = compute_S_from_x(solver, x_prev)
            S_now = compute_S_from_x(solver, x_t)
        else:
            S_prev = S_gpu[b[:, 0]]
            S_now = S_gpu[b[:, 1]]
```
(`train.py:148-151, 164-179`)

The noise is applied to *positions*, then S is recomputed — so the
perturbation is geometrically consistent (a noisy state, not a noisy
label). Targets stay clean: the model learns to contract back to the data
manifold. This plus the K-curriculum plus the potential regulariser is the
super-additive stability recipe (round 2); each alone is worth little.

### 4.2 The unrolled K-step chain with alternating blocks

```python
        for k in range(k_roll):
            f_ext = f_ext_gpu[i_t0 + k]  # (B, V, 3)

            if args.warm == "inertial":
                x0 = inertial_predictor(solver, x_t, x_prev, pin_b)
            else:
                x0 = x_t
            # Alternating network <-> decoder blocks (PoissonNet-style).  Each
            # block's global solve propagates the previous block's local
            # prediction across the whole mesh, so B blocks give the *network*
            # B global hops of receptive field at matched total decoder cost.
            S_prev_f = S_prev.to(dtype)
            iters_per_block = max(1, args.solver_iters // args.blocks)
            x_next = x0
            S_cur = S_now
            for _b in range(args.blocks):
                S_cur_f = S_cur.to(dtype)
                feat = build_features(
                    S_cur_f, S_prev_f, gravity32, f_ext.to(dtype), mu32, lam32, pin_flag, solver.tets, face_adj
                )
                S_star = net(feat, S_base=S_cur_f if args.residual else None)
                x_next = ts.solve(solver, S_star.double(), pin_b, x_init=x_next, n_iters=iters_per_block)
                if _b + 1 < args.blocks:
                    S_cur = compute_S_from_x(solver, x_next)
            ...
            S_prev = S_now
            S_now = compute_S_from_x(solver, x_next)
            x_prev = x_t
            x_t = x_next
```
(`train.py:184-207, 243-246`)

- The autoregressive chain is fully differentiable: step k's decoded
  `x_next` becomes step k+1's input state *without detach*, so gradients
  from frame k+K reach the prediction at frame k. This is what buys
  saturating (instead of compounding) rollout error — and what made the
  exact polar backward non-negotiable.
- The `--blocks` inner loop is the answer-so-far to notes/01: block b+1's
  features are built from `compute_S_from_x(x_next)` — the network gets to
  *see* the globally-solved consequence of its previous guess. Measured:
  2 blocks = -20% both metrics at equal decoder budget; 3 blocks help
  single-step, regress rollout.
- fp32/fp64 boundary is visible here: network and features in fp32,
  decoder and state in fp64, `.double()` / `.to(dtype)` at the seams (R7).

### 4.3 Loss

```python
            if args.loss == "pos":
                diff = x_next - x_gpu[i_t0 + k + 1]
                loss_total = loss_total + (mass[None, :, None] * diff * diff).sum()
                if args.phys_weight > 0.0:
                    loss_total = loss_total + args.phys_weight * incremental_potential_batched(
            ...
            else:
                loss_total = loss_total + incremental_potential_batched(
        ...
        (loss_total / (args.batch * k_roll)).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
```
(`train.py:209-213, 227-228, 249-250`, argument lists elided)

Mass-weighted position error against the reference frame — supervising
**decoded positions**, not S. The kNN experiment justified this after the
fact: the same predictions can be worse-than-persistence in S-Frobenius
while decoding 3x better, i.e. the decoder's pullback metric on S-space is
extremely anisotropic and the loss must be measured on the decoder's
output side.

The optional regulariser is one backward-Euler incremental potential:

```python
    inv_dt2 = 1.0 / (dt * dt)
    delta = x_next - 2.0 * x_t + x_prev  # (B, V, 3)
    L_inertia = 0.5 * inv_dt2 * (mass[None, :, None] * delta * delta).sum(dim=(-2, -1))

    x_tet = x_next[:, tets]  # (B, T, 4, 3)
    F = torch.einsum("tac,btad->btdc", J, x_tet)  # (B, T, 3, 3)
    Ft = F.transpose(-1, -2)
    eye = torch.eye(3, dtype=F.dtype, device=F.device).expand_as(F)
    E = 0.5 * (Ft @ F - eye)
    tr_E = E.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, T)
    frob_E2 = (E * E).sum(dim=(-2, -1))  # (B, T)
    L_elastic = ((mu * frob_E2 + 0.5 * lam * tr_E * tr_E) * volume).sum(dim=-1)  # (B,)

    L_gravity = -(mass[None, :, None] * x_next * gravity[None, None, :]).sum(dim=(-2, -1))
    L_ext = -(f_ext * x_next).sum(dim=(-2, -1))

    return (L_inertia + L_elastic + L_gravity + L_ext).sum()
```
(`potentials.py:39-55`)

Variational implicit Euler with StVK: its minimiser is one BE step at
`dt = 1/60`, while the data is 10 VBD substeps at `dt/10` — a known
objective mismatch (takeover review §2.3), which is why it now serves only
as a small off-manifold regulariser (`--phys-weight 1e-4`) rather than the
primary loss. Note the pin penalty is deliberately absent: pinned
particles carry zero mass, making the term identically zero, and the
decoder hard-pins anyway (`potentials.py:33-38`).

---

## 5. Data generation (`gen_train_data.py`)

```python
        # Random body force (per-vertex constant), magnitude in [0, 30] N total /n_verts.
        body_f_mag = rng.uniform(0.0, 25.0)
        body_f_dir = random_unit_vector(rng)
        body_force = (body_f_dir * body_f_mag).astype(np.float32)

        # Random point poke: choose a non-pinned vertex, magnitude up to 50 N, applied for first half.
        unpinned = np.where(particle_mass > 0)[0]
        poke_vert = int(rng.choice(unpinned))
        poke_force = (random_unit_vector(rng) * rng.uniform(0.0, 50.0)).astype(np.float32)
        poke_end_frame = rng.integers(2, n_frames_per // 2 + 1)
        ...
            for f in range(n_frames_per):
                # Compose f_ext for this frame.
                f_ext_frame = f_ext_np.copy()
                if f < poke_end_frame:
                    f_ext_frame[poke_vert] += poke_force

                for _ in range(args.substeps):
                    state_0.clear_forces()
                    # Add external force.
                    state_0.particle_f.assign(f_ext_frame)
                    model_t.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    state_0, state_1 = state_1, state_0
```
(`gen_train_data.py:102-111, 123-135`)

Reference = SolverVBD, 10 substeps x 10 iterations @ 60 fps, StVK, no
self-contact. The forcing has exactly the structure notes/01 worries
about: the **body force** is uniform and visible to every tet through
`f_ext_tet`; the **poke** is a single-vertex force visible only to the few
tets containing that vertex — every other tet must infer the disturbance
from its 1-ring stretch state as the wave arrives. The poke is released
mid-trajectory (`poke_end_frame`), adding a transient. Per-frame `f_ext`
is recorded so eval/rollout replay bit-identical schedules; the VBD replay
at the generation budget scoring exactly 0 in `bench_pareto` is the
determinism anchor for every error number we report.

---

## 6. Review findings

Ordered by how much they matter now.

- **R1 — dense Laplacian is the scale ceiling.** `(V, V)` fp64 storage and
  dense Cholesky put ~10k verts as a practical wall (4225 already means a
  122 MB factor and 3.9G of L). Anything real needs the sparse route
  (CHOLMOD-style factor or CG + AMG) or a Warp port. Not urgent for the
  current research questions; fatal for production claims.
- **R2 — pins are load-bearing.** `L_ff` invertibility, `inertial_predictor`
  pin restore, `pin_flag` feature, data gen (`fix_left=True`) all assume a
  pinned article. The momentum-anchoring plan (2026-08-09 decision)
  replaces exactly one of these four (the solve); the other three need
  coordinated changes — grep for `pinned` before assuming coverage.
- **R3 — polar backward is undefined on the SVD-fallback branch.** Backward
  computes `K = tr(S) I - S` from `S = R^T M`; on elements that took the
  `det M <= 0` fallback, S has a negative eigenvalue and K can be
  singular/indefinite. In-distribution this never fires (`det F > 0`, SPD
  `S*` by the tanh bound), but nothing *asserts* it: a silent `bad.any()`
  during training would produce garbage gradients with no error. Cheap fix:
  count fallback hits and warn/assert in training context.
- **R4 — `vert_to_tet_pin_flag` is defined twice** (`train.py:39`,
  `rollout.py:20`, eval imports the latter). Same body today; drift risk.
  Fold into `model.py` next time either is touched.
- **R5 — five dead feature dims** on current data (gravity 3, mu, lam —
  constant across every sample). Harmless now (z-scoring/std-clamp handles
  kNN; the MLP wastes a little capacity), but any "the net generalises
  across materials" claim is untested by construction until the data
  actually varies them.
- **R6 — `max_delta = 0.6` is a heuristic, not a guarantee.** Component-wise
  bound does not strictly bound eigenvalues (off-diagonal coupling);
  inherited from May, never ablated. Fine as a rail; do not read it as an
  SPD proof (that role belongs to R3's assert).
- **R7 — fp32/fp64 seams.** Net + features fp32, decoder + physics fp64,
  conversions at every block boundary (`train.py:200-205`). Correct but
  easy to get subtly wrong when editing; a Warp port should choose one
  story (fp32 decoder measured same-speed at toy scale — latency-bound).
- **R8 — the `--loss phys` objective is knowingly mismatched** with the data
  (1 BE step at dt vs 10 substeps at dt/10) — kept only as an ablation arm
  and regulariser. Anyone re-promoting it to the primary loss should first
  re-read takeover review §2.3.
- **R9 — no convergence telemetry in `solve()`.** Fixed `n_iters`, no
  residual logging; silent decoder truncation was *the* historical failure
  of this project (90% of May's Phase-2 error). A debug-mode
  `||x_k - x_{k-1}||` log (or a `diag_decoder_conv.py` run in CI fashion)
  would prevent a regression from hiding inside "model error" again.
- **R10 — window fallback can silently shorten the curriculum.** When fewer
  than `batch` windows have room for `k_target`, sampling falls back to any
  windows with `k_roll = min(room)` (`train.py:154-160`); at 20-frame
  trajectories with K=8 this triggers rarely, but it means late-curriculum
  batches are not guaranteed K=8. Log `k_roll` distribution if K ever grows.
- **R11 — batched-vs-per-sample equivalence of `solve()` is verified only
  ad-hoc** (session check, 1.5e-15 max diff) — not encoded as a test.
  Should join `test_torch_solver.py` (§7).
- **R12 — dead code/params:** `tikhonov` (no caller), `assemble_rhs` +
  `compute_F` (superseded by inlined batched versions in `solve` — still
  used by `potentials.incremental_potential` and tests), Phase-1 scripts
  (`run_*.py`, `recover_local_global.py`) untouched by the current
  pipeline. Harmless; listed so nobody mistakes them for load-bearing.

## 7. What is actually tested

| test | covers | verdict |
|---|---|---|
| `tests/test_polar.py` (8 tests) | Newton forward vs SVD reference across anisotropy (to 17:1); backward vs `gradcheck` and finite differences across the deformation range; fallback branch reached | solid for §2 |
| `test_torch_solver.py` | torch decoder vs the Phase-1 Warp implementation on a 60-frame recovery (3e-7 agreement); autograd smoke test (loss/grad finite) | pins §1's math + index conventions |
| ad-hoc (2026-08-04 session) | batched `solve()` == per-sample loop (1.5e-15) | should become a unit test (R11) |
| not tested | training step semantics (noise, curriculum, blocks bookkeeping), rollout/eval harness logic, data generator | relied on via results-level checks only (determinism anchor, known baselines) |

The results-level anchors partially compensate: VBD replay at the
generation budget scores exactly 0 in `bench_pareto` (whole-pipeline
determinism), and every experiment reports against persistence + oracle
floors, so a broken harness tends to show up as an impossible number
rather than a plausible one. But R11 and a `k_roll`/fallback-counter log
(R10, R3) are cheap and worth doing before the next round of training runs.
