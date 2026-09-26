# LIDO implementation walkthrough: the learned intrinsic deformation optimizer at commit f4298870

This walkthrough follows the code that trains and runs the learned optimizer for
clamped hexahedral bodies under `experiments/learned_intrinsic_solver/`. It
describes the implementation as it is at commit `f4298870` on
`ankac/learned-instrinic-solver` (the LIDO-v3 campaign source). Every excerpt is
copied verbatim from the files and carries its real line numbers; every
`file.py:NN` mention opens the embedded source at that line.

## Big picture: what one learned solver step does

The simulated object is a 10x10x40 grid of cubic cells with edge 0.025 m: 4,000
cells and 4,961 shared corners. The corners on the z-min face are prescribed
(clamped). One **physical timestep** (dt = 1/300 s) advances positions and
velocities with implicit Euler. Instead of a Newton solver, the timestep's
objective is minimized by a small number K of **learned optimizer queries**.

One query does the following:

- **Read the current candidate.** For every cell, the eight current corners give a
  3x3 deformation gradient F at the cell center. Its columns are the three
  transformed material axes. The closest right-handed rotation R to F is the
  cell's frame, and the network sees the axes in that frame, A = RᵀF (§8).
- **Assemble per-cell inputs.** Six 3x3 blocks per cell, all expressed in the
  cell's frame: current axes, inertial axis offset, physical deformation change,
  the current gradient of the physical objective with respect to the cell's axis
  increment, the previous such gradient, and the previous achieved axis update.
  The gradient blocks come from autograd of the physical energy followed by a
  transposed sparse solve through the assembly operator (§11, §12). Boundary
  flags, a log-RMS scalar and a history flag complete the 61 state values (§3).
- **Run the network.** One graph-transformer block attends over the 27-slot
  neighborhood of each cell with learned edge terms and material conditioning
  (§1, §2). The output is a bounded nine-value axis correction and one step size
  per cell (§3).
- **Fuse.** Cells cannot move shared corners independently. The world-space axis
  increments R(A_target − A) are fitted in a least-squares sense to one
  displacement per shared corner by a cached sparse factorization; prescribed
  corners are enforced exactly (§9).
- **Score.** The fused shape is evaluated by the implicit-Euler objective:
  stable Neo-Hookean elasticity at eight Gauss points per cell, the inertia term
  against the fixed inertial prediction Y, and metric damping anchored at the
  physical-step start (§4, §5, §6).
- **Learn.** The loss compares the energy after the update with the energy
  immediately before it on an asinh scale plus an increase penalty. Backpropagation
  runs from the energy through the fused positions and the transposed solve to
  the network weights; frames, the gradient feature and the history are constants
  for this backward pass (§7, §10).

Training samples fresh trajectories on the fly from a seeded augmenter and keeps a
pool of trajectories at different iteration and timestep ages (§13, §14, §15).

## One optimizer query through the pipeline (where each section lives)

```
PREPARE   CPU worker threads · once per trajectory reset, once per physical step
  InitialStateAugmenter.reset(seed): rest grid, multiscale shape and velocity,
    material, perturbation scale ................................................ §13, §14
  MixedHexSolverStep.prepare: Y = X_start + dt·V + dt²·a, pins, rigid initializer .. §4
  _TrajectoryFactory._candidate: inertial or perturbed-inertial start (50 / 50) ..... §15
  -> payload: candidate, Y, X_start, pins, context id, history (zeros at reset)

QUERY     GPU, one learned proposal per pool member · batch of 16 per rank
  _batch: stack payloads, build OptimizerHistory ..................................... §12
  assemble_inputs
    center F from eight corners, closest proper frame R, A = RᵀF ..................... §8
    dE/dX by autograd -> fusion adjoint -> Rᵀ -> RMS normalize -> state values ....... §11
    27-slot neighborhood, 24 edge values per slot in the receiver frame .............. §1
  IntrinsicSolverNetwork: encoders -> transformer block -> correction, per-cell step .. §2, §3
  world increment R(A_target − A) -> HexFusion.fuse: CPU PARDISO solve, exact pins ... §9
  energy on the fused shape: stable Neo-Hookean + inertia + metric damping ............ §5, §4, §6
  -> LearnedHexStepOutput: positions, energies, axis_gradient_world,
     achieved_axis_update_world, force_residual_norm, tie_mask

LEARN     once per mixed batch
  local_objective: asinh(E_after / scale) + relu((E_after − E_before) / scale) ....... §7
  loss.backward(): energy -> fused positions -> transposed solve -> weights ........... §7, §10
  Adam step; store_history into payloads; carried candidate detached .................. §12, §10

ADVANCE   after K queries · once per physical step (CPU worker)
  velocity = (X_solved − X_start) / dt with zero pinned velocity; prepare next Y ....... §15
  history carried unchanged into the new payload ...................................... §12
  after H steps: retire the trajectory and reset with the next seed ................... §15
```

## §1. Neighborhood topology and directed edge features

**Where this lives / what this part does.** Both functions live in experiments/learned_intrinsic_solver/network_geometry.py and have no learned parameters. `build_grid_neighborhood` (experiments/learned_intrinsic_solver/network_geometry.py:22) fixes who talks to whom: for every cell it returns a fixed-length list of neighbor slots and a boolean validity mask. `IntrinsicSolverNetwork.__init__` calls it once per distinct hop and registers the two tensors as buffers (experiments/learned_intrinsic_solver/network.py:263-266), so the topology is computed once per network construction and then travels with `state_dict()` and `.to(device)`. `build_edge_features` (experiments/learned_intrinsic_solver/network_geometry.py:80) converts the current geometry of every (receiver, neighbor) pair into a 24-value dimensionless descriptor. It runs once per network query and per distinct hop inside `assemble_inputs` (experiments/learned_intrinsic_solver/input_assembly.py:268-274), the input-assembly stage that precedes each network forward; one physical time step contains between 1 and 32 such queries (experiments/learned_intrinsic_solver/unrolled_solver.py:86-87).

Terms used in this section:

- **Cell**: one hexahedral voxel of the canonical cuboid. N is the number of cells, B the number of independent objects in the batch, S the number of neighbor slots.
- **z-fast ordering**: cell id `= x * (ny * nz) + y * nz + z`; consecutive ids walk along material z first.
- **Chebyshev distance**: `max(|dx|, |dy|, |dz|)` between two integer cell coordinates. Distance 1 covers the 26 cells that share a face, an edge or a corner with the center cell.
- **Slot**: a fixed position in each cell's neighbor list. Slot 0 is always the cell itself; slot s > 0 always means the same integer offset whichever cell is the receiver, so a given slot has the same meaning for every cell.
- **Frame R_i**: the proper rotation (3x3, determinant +1) closest to the cell-center deformation gradient F_i, computed and returned detached by `closest_proper_rotations` (experiments/learned_intrinsic_solver/frames.py:4-20 describes the rule). Its columns are the world directions of the cell's local axes.
- **Local axes A_i = R_i^T F_i**: the deformation gradient written in the cell's own frame, dimensionless, with the deformed material axes as columns. `R_i @ A_i = F_i` holds for every cell, including inverted ones.
- **Receiver**: the cell i that gathers messages from its slots. Every edge quantity below is expressed in the receiver's frame R_i.
- **Detach**: `tensor.detach()` cuts the autograd graph at that tensor, so backpropagation treats it as a constant.

**Topology.** After argument validation the function enumerates offsets once and gathers them for every cell at the same time:

```python
    nx, ny, nz = (int(count) for count in cell_counts)
    hop = int(hop)
    offsets = [(0, 0, 0)] + [
        (dx, dy, dz)
        for dx in range(-hop, hop + 1)
        for dy in range(-hop, hop + 1)
        for dz in range(-hop, hop + 1)
        if max(abs(dx), abs(dy), abs(dz)) == hop
    ]

    cell_ids = torch.arange(nx * ny * nz, dtype=torch.long, device=device)
    coordinates = torch.stack((cell_ids // (ny * nz), cell_ids // nz % ny, cell_ids % nz), dim=-1)
    neighbor_coordinates = coordinates[:, None, :] + torch.tensor(offsets, dtype=torch.long, device=device)
    counts = torch.tensor((nx, ny, nz), dtype=torch.long, device=device)
    valid = (neighbor_coordinates >= 0).all(dim=-1) & (neighbor_coordinates < counts).all(dim=-1)
    indices = (
        neighbor_coordinates[..., 0] * (ny * nz) + neighbor_coordinates[..., 1] * nz + neighbor_coordinates[..., 2]
    )
    return indices.masked_fill(~valid, 0), valid
```
(`experiments/learned_intrinsic_solver/network_geometry.py:59-77`)

- `offsets` starts with `(0, 0, 0)` (self) and then lists every integer offset at exactly Chebyshev distance `hop`, in lexicographic dx, dy, dz order. The slot count is `S = (2 hop + 1)^3 - (2 hop - 1)^3 + 1`: 27 for hop 1, 99 for hop 2, 387 for hop 4. The shell is exact: a hop-2 neighborhood does not contain the hop-1 cells.
- `coordinates` decodes each z-fast id into `(x, y, z)` with integer division; `neighbor_coordinates` is `[N, S, 3]` and `valid` marks the slots whose coordinates lie inside `[0, n)` on all three axes. Nothing wraps around: on a hop-1 grid a corner cell has 8 valid slots, a cell on one face 18, an interior cell 27.
- Output dtypes are `torch.long` ids and `torch.bool` masks, both `[N, S]`. Masked slots carry id 0 (an in-range sentinel), so callers may gather with them as long as they zero the result afterwards. Slot 0 is valid for every cell, so no row is ever fully masked on this topology.
- The same `[N, S]` tensors serve every batch entry: all objects in a batch share the cuboid dimensions. The module docstring restricts this to fully occupied cuboids (experiments/learned_intrinsic_solver/network_geometry.py:6-7); holes or disconnected occupancy are not representable.

**Directed edge descriptor.** After the shape, dtype and device checks (experiments/learned_intrinsic_solver/network_geometry.py:121-158), the geometry itself is nine tensor statements:

```python
    # Sanitize padding before gathering: sentinel IDs need not be in range.
    safe_indices = neighbor_indices.masked_fill(~neighbor_mask, 0)
    rotations = frames.detach()
    receiver_transpose = rotations.transpose(-1, -2).unsqueeze(2)
    relative_frames = receiver_transpose @ rotations[:, safe_indices]
    relative_axes = relative_frames @ local_axes[:, safe_indices]
    current_offsets = current_centers[:, safe_indices] - current_centers.unsqueeze(2)
    local_offsets = (receiver_transpose @ current_offsets.unsqueeze(-1)).squeeze(-1) / cell_size
    rest_offsets = (rest_centers[safe_indices] - rest_centers.unsqueeze(1)) / cell_size
    features = torch.cat(
        (
            rest_offsets.unsqueeze(0).expand(batch_count, -1, -1, -1),
            local_offsets,
            relative_frames.flatten(start_dim=-2),
            relative_axes.flatten(start_dim=-2),
        ),
        dim=-1,
    )
    return features.masked_fill(~neighbor_mask[None, :, :, None], 0)
```
(`experiments/learned_intrinsic_solver/network_geometry.py:160-178`)

The 24 values of one directed edge (receiver i, neighbor j), in order:

| index range | name | formula | notes |
|---|---|---|---|
| 0-2 | rest offset | `(X_j - X_i) / h` | material rest centers X [m] over the rest edge length h [m]; on a hop-1 grid every entry is -1, 0 or 1 up to float32 rounding: the zero components are bit-exact, but because the rest centers are stored in float32 the +-1 components differ from the integer by up to about 1.4e-6 when h = 0.025 m (they are exact only for power-of-two h such as 1.0 or 0.5); batch independent, expanded over B |
| 3-5 | current offset in the receiver frame | `R_i^T (x_j - x_i) / h` | world cell centers x [m]; the only channel built directly from the cell-center positions. Channels 6-23 also change with the current positions, since R and A are extracted from the current F(x); only the rest offset (0-2) is position independent |
| 6-14 | relative frame | `R_i^T R_j`, row-major | how neighbor j's frame is rotated relative to the receiver; identity on the self slot |
| 15-23 | transported neighbor axes | `R_i^T R_j A_j = R_i^T F_j`, row-major | neighbor j's deformation gradient seen from the receiver; equals A_i on the self slot |

Key implementation points:

- **Shapes.** `receiver_transpose` is `[B, N, 1, 3, 3]`; `rotations[:, safe_indices]` gathers `[B, N, S, 3, 3]` neighbor frames, so `relative_frames` and `relative_axes` are `[B, N, S, 3, 3]` and flatten to 9 values each. Both offsets are `[B, N, S, 3]`. The output is `[B, N, S, 24]`.
- **Detach boundary.** Only `frames` is detached (experiments/learned_intrinsic_solver/network_geometry.py:162). `current_centers`, `rest_centers` and `local_axes` keep their autograd paths, so the network's edge inputs stay differentiable in the candidate corner positions (through the cell centers and through A = R^T F) while the frame extraction is treated as a fixed change of basis. At the assembly call site the frames are already detached, so this call is a second guard rather than the primary cut.
- **Units and dtype.** Both offsets are divided by `cell_size` (a positive finite Python number in meters), so every channel is dimensionless. All floating inputs must share one dtype; float32 stays float32 with no promotion. `neighbor_indices` must be `torch.long` and `neighbor_mask` must be `torch.bool`.
- **Masks first, then zero.** `safe_indices` replaces every masked id with 0 before any gather (experiments/learned_intrinsic_solver/network_geometry.py:161), so arbitrary sentinel ids in masked slots never index out of range; the final `masked_fill` (experiments/learned_intrinsic_solver/network_geometry.py:178) zeroes all 24 values of every masked slot.
- **The self slot is real geometry, not a zero token.** For slot 0 both offsets are zero, the relative frame is the identity and the transported axes equal A_i. The layer in §2 can therefore treat self like any other slot without a special case.
- **Invariance.** A global rigid motion `x -> Q x + t`, `R -> Q R` leaves all 24 channels unchanged, because every quantity is either a rest-space difference or has been multiplied by `R_i^T` (the tests pin this to 2e-6 in float32). Material reference axes are shared by all cells, so `rest_offsets` is the only channel that says where in the grid an edge sits.

## §2. Transformer layer: masked attention with learned edge terms and FiLM

**Where this lives / what this part does.** `IntrinsicTransformerLayer` (experiments/learned_intrinsic_solver/network.py:36-183) is the one reusable block of the model: pre-norm multi-head attention over a cell's neighbor slots, plus a pre-norm residual feed-forward branch, both optionally modulated by FiLM. `IntrinsicSolverNetwork` instantiates one layer per entry of `hops` (experiments/learned_intrinsic_solver/network.py:280-289); the default is a single layer with hidden width 128, four heads and hop 1. The forward runs once per layer per network query on the whole batch, with the receiver cells processed in chunks of `query_chunk_size` (default 128).

Terms used in this section:

- **Token**: the hidden vector of one cell, `hidden_dim` = 128 channels. Hidden channels are not spatial vectors; all geometry enters through the edge features of §1.
- **LayerNorm**: per token, subtract the mean over channels and divide by the standard deviation, then apply a learned per-channel scale and shift. **Pre-norm** means the LayerNorm sits at the entrance of each residual branch, so the residual stream itself is never normalized.
- **FiLM (feature-wise linear modulation)**: `y = x * (1 + gamma) + beta`, where `gamma` and `beta` are produced from a conditioning vector by one linear layer. Zero-initialized weights and bias give `gamma = beta = 0`, so FiLM is the identity at the start of training.
- **Multi-head attention**: each token produces a query q, a key k and a value v per head, each of `head_dim = hidden_dim / num_heads` = 32 channels. The score of a neighbor is `q . k / sqrt(head_dim)`; a softmax over the S slots turns scores into weights; the message is the weighted sum of neighbor values, concatenated over heads and projected back to `hidden_dim`.
- **Edge bias / edge values**: learned per-edge terms computed from the encoded edge features. `edge_bias` adds one scalar per head to the score; `edge_val` adds a `hidden_dim` vector to the neighbor's value before weighting.
- **SiLU**: `x * sigmoid(x)`, the activation of the feed-forward branch.

**Parameters.** The constructor builds everything from `hidden_dim` and `edge_dim`:

```python
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        self.edge_val = nn.Linear(edge_dim, hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
            nn.SiLU(),
            nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
        )
        self.film = nn.Linear(conditioning_dim, 4 * hidden_dim) if conditioning_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
```
(`experiments/learned_intrinsic_solver/network.py:85-99`)

- `qkv` is one `Linear(128, 384)` producing query, key and value together; `edge_bias` maps the 64 encoded edge channels to `num_heads` = 4 scalars; `edge_val` maps them to 128 value channels; `out_projection` is the usual post-attention linear map.
- The feed-forward branch is `Linear(128, 512)`, SiLU, `Linear(512, 128)` (`ffn_multiplier` = 4).
- FiLM is a single `Linear(conditioning_dim, 4 * hidden_dim)` shared by both residual branches: its output is split into four chunks (scale and shift for the attention branch, scale and shift for the feed-forward branch). Weight and bias are zeroed at init (experiments/learned_intrinsic_solver/network.py:97-99).

**Conditioning, normalization and projections.** The forward first validates shapes (experiments/learned_intrinsic_solver/network.py:129-142), then:

```python
        modulation = None
        if self.film is not None:
            if conditioning is None:
                raise ValueError("conditioning is required when FiLM is enabled")
            conditioning = _expand_conditioning(conditioning, batch, cells, self.conditioning_dim)
            modulation = self.film(conditioning).chunk(4, dim=-1)
        elif conditioning is not None:
            raise ValueError("conditioning was supplied but this layer has no FiLM channels")

        normalized = self.attention_norm(features)
        if modulation is not None:
            normalized = normalized * (1 + modulation[0]) + modulation[1]
        query, key, value = self.qkv(normalized).reshape(batch, cells, 3, self.num_heads, self.head_dim).unbind(2)
```
(`experiments/learned_intrinsic_solver/network.py:144-156`)

- `conditioning` may be `[B, C]` (one vector per object) or `[B, N, C]`; `_expand_conditioning` (experiments/learned_intrinsic_solver/network.py:28) broadcasts the former to every cell. The four FiLM chunks are each `[B, N, hidden_dim]`.
- The attention branch input is `LayerNorm(features) * (1 + gamma_1) + beta_1`.
- `qkv(normalized)` is `[B, N, 384]`, reshaped to `[B, N, 3, num_heads, head_dim]` and unbound into three `[B, N, 4, 32]` tensors.

**The masked neighbor softmax and both residuals.**

```python
        safe_indices = torch.where(neighbor_mask, neighbor_indices, 0)
        chunks, attention_chunks = [], []
        for start in range(0, cells, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, cells)
            valid = neighbor_mask[None, start:stop, :, None]
            indices = safe_indices[start:stop]
            # Remove poisoned padding before learned projections, not just after softmax.
            edges = torch.where(valid, edge_features[:, start:stop], 0)
            scores = (query[:, start:stop, None] * key[:, indices]).sum(-1) * (self.head_dim**-0.5)
            scores = (scores + self.edge_bias(edges)).masked_fill(~valid, -torch.inf)
            # All-masked rows must not send NaNs through softmax or its backward.
            has_neighbor = valid.any(dim=2, keepdim=True)
            scores = torch.where(has_neighbor, scores, 0)
            weights = scores.softmax(dim=2).masked_fill(~valid, 0)
            edge_values = self.edge_val(edges).reshape(batch, stop - start, slots, self.num_heads, self.head_dim)
            message = (weights[..., None] * (value[:, indices] + edge_values)).sum(dim=2)
            chunks.append(message.reshape(batch, stop - start, self.hidden_dim))
            if return_attention:
                attention_chunks.append(weights.permute(0, 1, 3, 2))
        attended = features + self.out_projection(torch.cat(chunks, dim=1))
        normalized = self.ffn_norm(attended)
        if modulation is not None:
            normalized = normalized * (1 + modulation[2]) + modulation[3]
        output = attended + self.ffn(normalized)
        if return_attention:
            return output, torch.cat(attention_chunks, dim=1)
        return output
```
(`experiments/learned_intrinsic_solver/network.py:157-183`)

Shapes inside one chunk of `stop - start` receiver cells:

- `valid` is `[1, chunk, S, 1]`; `indices` is `[chunk, S]` with masked slots redirected to cell 0 (experiments/learned_intrinsic_solver/network.py:157), exactly as §1 does.
- `edges` is `[B, chunk, S, edge_dim]` with masked slots zeroed **before** `edge_bias` and `edge_val` see them (experiments/learned_intrinsic_solver/network.py:164). A NaN in padding would otherwise poison the linear layers' outputs and their backward even if its attention weight ends up zero; `torch.where` routes no gradient to the unselected branch, so masked padding receives exactly zero gradient.
- `key[:, indices]` gathers `[B, chunk, S, H, d]`; the elementwise product with `query[:, start:stop, None]` (shape `[B, chunk, 1, H, d]`) and the sum over d give scores `[B, chunk, S, H]`, scaled by `head_dim ** -0.5 = 1 / sqrt(32)`.
- `edge_bias(edges)` is `[B, chunk, S, H]` and is added to the scores; masked slots are then set to `-inf` so the softmax over `dim=2` (the slot axis) gives them weight exactly 0.
- **All-masked rows.** If a receiver had no valid slot, a row of all `-inf` would make softmax return NaN, and the NaN would flow into backward. `has_neighbor` (shape `[1, chunk, 1, 1]`) detects that case and replaces the whole row by zeros first; the following `masked_fill(~valid, 0)` then zeroes the uniform weights, so the message is exactly zero; the attention branch then adds only `out_projection`'s bias to the token (`nn.Linear` with its default bias), and the feed-forward branch (`ffn_norm`, FiLM, SiLU MLP) still runs on that result, so the token stays finite and well defined but is not literally unchanged. The canonical topology never produces such a row (self is always valid); this is a guard for hand-built topologies and tests.
- `edge_values` is `[B, chunk, S, H, d]`; the message is `sum_s w_s (v_j(s) + e_s)`, reduced over the slot axis to `[B, chunk, H, d]` and flattened to `[B, chunk, hidden_dim]`.
- After the loop, `out_projection` is applied to the concatenated `[B, N, hidden_dim]` messages and added to the **un-normalized** `features` (first residual). The second branch applies `ffn_norm`, the second FiLM pair and the SiLU MLP, and adds the result (second residual).
- With `return_attention=True` the weights are returned as `[B, N, num_heads, S]` (the `permute` at experiments/learned_intrinsic_solver/network.py:175 swaps the slot and head axes); invalid slots are exactly zero there.

Why chunk the queries? Each chunk allocates the gathered `key[:, indices]` and `value[:, indices]` tensors of size `B * chunk * S * hidden_dim` instead of `B * N * S * hidden_dim`. For the 4000-cell grid with S = 27 and 128 channels that is 128 x 27 x 128 = 442k floats per object per gather instead of 13.8M. The softmax is still computed over all S slots of each receiver, so the result is identical to an unchunked layer (the tests compare chunk size 1 against 16 to float32 tolerance, including gradients). During training autograd still saves every chunk's activations for backward, so chunking does not reduce the peak training memory of the layer; the class docstring says so at experiments/learned_intrinsic_solver/network.py:45-46.

## §3. Network module: encoders, bounded correction and the per-cell step

**Where this lives / what this part does.** `IntrinsicSolverNetwork` (experiments/learned_intrinsic_solver/network.py:197-356) wraps the layer of §2 with three small encoders (node, edge, conditioning), an output LayerNorm and two zero-initialized heads, and returns an `IntrinsicSolverOutput` (experiments/learned_intrinsic_solver/network.py:186-194) holding the local target axes `[B, N, 3, 3]`, the bounded correction `[B, N, 3, 3]` and a per-cell step `[B, N]`. The forward runs once per network query, called from experiments/learned_intrinsic_solver/solver_step.py:318 (single material) and experiments/learned_intrinsic_solver/mixed_physics.py:462 (mixed materials); the very next line in both places converts the result to a world-space axis increment `frames @ (local_target_axes - local_axes)` that is handed to fusion. The layout of what the network is fed is fixed by experiments/learned_intrinsic_solver/features.py, which holds the schema constants and the pure packing functions; `assemble_inputs` (experiments/learned_intrinsic_solver/input_assembly.py:185) calls them once per query.

Terms used in this section:

- **Encoder**: a two-layer MLP `Linear, SiLU, Linear` that maps raw inputs to a hidden width. There is one for cells (node), one for edges and one for the conditioning vector.
- **Registered buffer**: a tensor stored on the module with `register_buffer`, so it moves with `.to()` and is saved in `state_dict()` but is not a trainable parameter.
- **Correction head**: the linear map from the final 128-channel token to the nine entries of a 3x3 change of the local axes.
- **Step head**: the linear map from the same token to one scalar per cell, squashed by a sigmoid into `(0, max_step_size)`.
- **RMS (root mean square)**: `sqrt(mean(v^2))` over all cell matrix components of one object; used as a per-object scale.
- **Fusion adjoint**: fusion is the linear map from per-cell axis increments to shared corner displacements; its adjoint (transpose) maps a gradient with respect to corner positions back to a gradient with respect to per-cell axis increments. The gradient blocks of the state vector are produced by that adjoint.
- **Lamé parameters**: lambda and mu [Pa], the two elastic constants of the material law used by the physical objective; mu is the shear modulus.

**Construction.** Topology buffers, encoders, layers and heads:

```python
        for hop in dict.fromkeys(self.hops):
            indices, mask = build_grid_neighborhood(self.cell_counts, hop)
            self.register_buffer(f"neighbor_indices_{hop}", indices)
            self.register_buffer(f"neighbor_mask_{hop}", mask)
        self.state_feature_dim = state_feature_dim
        self.conditioning_dim = conditioning_dim
        self.edge_input_dim = edge_input_dim
        self.max_step_size = float(max_step_size)
        self.node_encoder = nn.Sequential(
            nn.Linear(9 + state_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.SiLU(), nn.Linear(edge_hidden_dim, edge_hidden_dim)
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList(
            IntrinsicTransformerLayer(
                hidden_dim,
                edge_hidden_dim,
                num_heads=num_heads,
                conditioning_dim=hidden_dim,
                query_chunk_size=query_chunk_size,
            )
            for _ in self.hops
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.correction_head = nn.Linear(hidden_dim, 9)
        self.step_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias)
```
(`experiments/learned_intrinsic_solver/network.py:263-296`)

- `dict.fromkeys(self.hops)` deduplicates hops while keeping order, so `hops=(1, 1, 1)` builds one topology buffer pair and three layers. `neighborhood(hop)` (experiments/learned_intrinsic_solver/network.py:298-302) is the accessor the input assembly uses to fetch the buffers.
- `node_encoder` takes `9 + state_feature_dim` inputs: the nine entries of the current local axes A (row-major) followed by the state vector. `state_feature_dim` is a required constructor argument with no default; every solver call site passes `features.STATE_FEATURE_DIM` = 61 (experiments/learned_intrinsic_solver/solver_step.py:130, experiments/learned_intrinsic_solver/newton_solver.py:246), giving a 70-channel node input. `edge_encoder` maps the 24 raw edge channels of §1 to `edge_hidden_dim` = 64. `condition_encoder` maps the six conditioning channels to 128.
- Each layer gets `conditioning_dim=hidden_dim` (experiments/learned_intrinsic_solver/network.py:285): FiLM is driven by the **encoded** 128-channel conditioning, not by the six raw scalars, so each layer's FiLM linear is `Linear(128, 512)`.
- Both heads start at exactly zero, weights and biases (experiments/learned_intrinsic_solver/network.py:293-296). With the default keyword arguments and `state_feature_dim` = 61 the module has 323,214 parameters (pinned by a test).

**Forward.** After the shape checks (experiments/learned_intrinsic_solver/network.py:329-334):

```python
        conditioning = _expand_conditioning(conditioning, batch, cells, self.conditioning_dim)
        condition = self.condition_encoder(conditioning)
        features = self.node_encoder(torch.cat((local_axes.flatten(-2), state_features), dim=-1))
        encoded_edges = {}
        for hop in dict.fromkeys(self.hops):
            indices, mask = self.neighborhood(hop)
            if hop not in edge_features:
                raise ValueError(f"edge_features is missing hop {hop}")
            edges = edge_features[hop]
            if edges.shape != (batch, cells, indices.shape[1], self.edge_input_dim):
                raise ValueError(f"edge_features[{hop}] has the wrong shape")
            edges = torch.where(mask[None, :, :, None], edges, 0)
            encoded_edges[hop] = self.edge_encoder(edges)
        for hop, layer in zip(self.hops, self.layers, strict=True):
            indices, mask = self.neighborhood(hop)
            features = layer(features, encoded_edges[hop], indices, mask, conditioning=condition)
        features = self.output_norm(features)
        raw = self.correction_head(features)
        correction = (raw / torch.sqrt(1 + raw.square().sum(dim=-1, keepdim=True))).reshape(batch, cells, 3, 3)
        step_size = self.max_step_size * self.step_head(features).squeeze(-1).sigmoid()
        target = local_axes + step_size[..., None, None] * correction
        return IntrinsicSolverOutput(target, correction, step_size)
```
(`experiments/learned_intrinsic_solver/network.py:335-356`)

- `condition` is `[B, N, 128]` and is passed unchanged to every layer's FiLM. `features` is `[B, N, 128]` after the node encoder.
- Edges are masked once more (`torch.where(mask[None, :, :, None], edges, 0)`, experiments/learned_intrinsic_solver/network.py:346) before the shared `edge_encoder`, for the same NaN-poisoning reason as in §2; the encoded `[B, N, S, 64]` tensor is computed once per distinct hop and reused by every layer with that hop.
- `output_norm` (a LayerNorm) is applied to the final token before both heads.
- **Bounded correction.** `raw` is `[B, N, 9]`; `correction = raw / sqrt(1 + |raw|^2)` where `|raw|` is the Euclidean norm of the nine values of one cell. This is a smooth squashing with `|correction| < 1` for every finite `raw`, slope 1 near zero (so early training behaves like an unbounded linear head) and no hard saturation. The nine values are reshaped row-major into `[B, N, 3, 3]`, matching how the input axes were flattened.
- **Per-cell step.** `step_size = max_step_size * sigmoid(step_head(features))`, shape `[B, N]`, strictly inside `(0, max_step_size)`. At zero initialization every cell gets `0.5 * max_step_size`; once the head has trained, each cell has its own step.
- **Target.** `target = local_axes + step_size[..., None, None] * correction`, so the Frobenius norm of each cell's axis change is strictly below `max_step_size`. At initialization `correction = 0` and the target equals the input axes exactly (a no-op proposal at the network level; the fusion that follows may still move corners).
- Everything is float32 by default (parameters and working tensors); nothing is detached inside the network, so gradients flow to `local_axes`, `state_features`, every `edge_features[hop]` and `conditioning`.

**Input contract: the 61-value state vector.** experiments/learned_intrinsic_solver/features.py:54-98 is the single source of truth for the layout. Its constants:

```python
MATRIX_BLOCKS = (
    "inertial_axis_offset",
    "physical_axis_change",
    "current_axis_gradient",
    "previous_axis_gradient",
    "previous_axis_update",
)
"""Nine-value matrix blocks in packing order; each is ``R^T M`` flattened row-major."""

MATRIX_FEATURE_DIM = 9 * len(MATRIX_BLOCKS)
"""Width of the five matrix blocks (45)."""

BOUNDARY_DIM = 14
"""Six exposed-face flags followed by eight fixed-corner flags in z-fast corner order."""

SCALAR_FEATURES = ("log_gradient_rms", "history_valid")
"""Trailing per-cell scalars: natural log of the object gradient RMS and the history flag."""

STATE_FEATURE_DIM = MATRIX_FEATURE_DIM + BOUNDARY_DIM + len(SCALAR_FEATURES)
"""Total state width (61): 45 matrix components, 14 boundary flags, 2 scalars."""
```
(`experiments/learned_intrinsic_solver/features.py:54-73`)

`pack_state_features` (experiments/learned_intrinsic_solver/features.py:261) concatenates the blocks along the last axis. With R the receiver's frame, F the current cell-center deformation gradient, F_Y the deformation gradient of the inertial prediction Y, F_prev that of the positions at the start of the physical step, G the fusion-adjoint gradient of the physical objective with respect to the world axis increment, and U_prev the previously achieved world axis change:

| index range | width | block | content |
|---|---|---|---|
| 0-8 | 9 | `inertial_axis_offset` | `R^T (F_Y - F)`, row-major; how far the axes are from the inertial guess |
| 9-17 | 9 | `physical_axis_change` | `R^T (F - F_prev)`, row-major; how far the axes have moved during this step so far |
| 18-26 | 9 | `current_axis_gradient` | `clip(R^T G / rms, -10, 10)`; rms is this object's gradient RMS |
| 27-35 | 9 | `previous_axis_gradient` | `clip(R^T G_prev / rms, -10, 10)`, same rms as the current gradient; zeros when no history |
| 36-44 | 9 | `previous_axis_update` | `clip(R^T U_prev / rms_own, -10, 10)`, normalized by its own RMS; zeros when no history |
| 45-50 | 6 | exposed-face flags | 1.0 where that material face of the cell lies on the object surface |
| 51-58 | 8 | fixed-corner flags | 1.0 where the cell's corner (z-fast corner order) is a prescribed corner |
| 59 | 1 | `log_gradient_rms` | `ln(rms)` of the current gradient, one value per object broadcast to all cells |
| 60 | 1 | `history_valid` | exactly 1.0 or 0.0 per object |

Every matrix block is in the receiver's frame: `to_local` (experiments/learned_intrinsic_solver/features.py:194-223) computes `R^T @ M`, and the block is flattened row-major like the axes and like the edge matrices, so state index `9 k + 3 i + j` is entry `(i, j)` of block k.

**Normalization.** `rms_normalize` (experiments/learned_intrinsic_solver/features.py:226-258) scales a matrix field by a per-object RMS and clips it:

```python
    _require_cell_matrices("values", values)
    batch_count = values.shape[0]
    if rms is None:
        scale = values.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    else:
        scale = _per_object("rms", rms, batch_count).to(dtype=values.dtype, device=values.device)
        scale = scale.reshape(batch_count, 1, 1, 1)
    scale = scale.clamp_min(RMS_FLOOR)
    return (values / scale).clamp(-CLIP, CLIP), scale
```
(`experiments/learned_intrinsic_solver/features.py:250-258`)

The RMS is taken over every cell and component of one object (`dim=(1, 2, 3)`), floored at `RMS_FLOOR = 1e-12`, and returned as `[B, 1, 1, 1]` so the caller can pass it on. The previous gradient is divided by the **current** gradient's RMS (so the network sees both gradients on one scale), while the previous update uses its own RMS. Components are then clipped to `+/-CLIP = +/-10`. A measured zero field stays zero and reports an RMS equal to the floor, so its log is `ln(1e-12) = -27.6`.

**History gating.** The two history blocks are zeroed for objects whose `history_valid` is false, whatever tensors were supplied, and the flag itself is written as 1.0 or 0.0:

```python
    mask = valid[:, None, None, None]
    columns = []
    for name in MATRIX_BLOCKS:
        block = blocks[name]
        if name in _HISTORY_BLOCKS:
            block = torch.where(mask, block, torch.zeros_like(block))
        columns.append(block.flatten(-2))
    columns.append(boundary)
    columns.append(log_rms[:, None, None].expand(-1, cell_count, 1))
    columns.append(valid.to(reference.dtype)[:, None, None].expand(-1, cell_count, 1))
    return torch.cat(columns, dim=-1)
```
(`experiments/learned_intrinsic_solver/features.py:335-345`)

A missing history is therefore always distinguishable from a measured zero. Further points of the packing:

- **Boundary flags** are a `[C, 14]` buffer shared by all objects of a step (built at experiments/learned_intrinsic_solver/solver_step.py:161-166: six exposed-face indicators from the rest geometry followed by the fixed flag of each of the cell's eight corners) and are broadcast over the batch by `pack_state_features`; a `[B, C, 14]` tensor is accepted too.
- **Dtype and detach.** All blocks must share one floating dtype and device; the output has that dtype and shape `[B, C, 61]`. The functions never detach anything. At the call site the frames and the gradient feature arrive detached, while `inertial_axis_offset` and `physical_axis_change` remain differentiable in the candidate positions.
- **The nine axis values are not in this vector.** The network prepends `local_axes.flatten(-2)` itself (experiments/learned_intrinsic_solver/network.py:337), so the complete node input is 70 values.

**Input contract: the six conditioning channels.** `conditioning_channels` (experiments/learned_intrinsic_solver/features.py:348) takes four 1-D material tensors and two Python scalars and builds one 6-value row per tensor entry: one row per object (`[B, 6]`) in the mixed-material step, one row per cell (`[C, 6]`) in the single-material step:

```python
    channels = (
        (lame_lambda / _REFERENCE_LAME).log1p(),
        (lame_mu / _REFERENCE_LAME).log1p(),
        (density / _REFERENCE_DENSITY).log(),
        torch.full_like(lame_mu, math.log(size / _REFERENCE_CELL_SIZE)),
        torch.full_like(lame_mu, math.log(step / _REFERENCE_TIME_STEP)),
        (damping / (lame_mu * step)).log1p(),
    )
    return torch.stack(channels, dim=-1)
```
(`experiments/learned_intrinsic_solver/features.py:399-407`)

| index | name | formula | reference value |
|---|---|---|---|
| 0 | `log1p_lame_lambda` | `log1p(lambda / 1e5)` | 1e5 Pa |
| 1 | `log1p_lame_mu` | `log1p(mu / 1e5)` | 1e5 Pa |
| 2 | `log_density` | `log(rho / 1000)` | 1000 kg/m^3 |
| 3 | `log_cell_size` | `log(h / 0.025)` | 0.025 m |
| 4 | `log_time_step` | `log(dt / (1/60))` | 1/60 s |
| 5 | `log1p_damping` | `log1p(eta / (mu dt))` | dimensionless ratio of the viscosity eta [Pa s] to mu dt |

- `log1p(x) = ln(1 + x)` keeps the Lamé and damping channels finite at zero (lambda = 0 or eta = 0 give exactly 0); plain `log` is used where zero is not a physical value (density, cell size, time step).
- Channels 3 and 4 are constants per call, filled with `torch.full_like(lame_mu, ...)` so they share dtype and device with the material tensors.
- Material tensors are not value-checked here; a nonpositive density or a zero mu produces a nonfinite channel visibly rather than silently. The step modules validate materials before calling.
- The single-material step computes the channels once at construction as a per-cell `[C, 6]` buffer (experiments/learned_intrinsic_solver/solver_step.py:167-171; its Lamé, density and damping tensors are per-cell `[C]`, expanded from scalars by `HexImplicitEulerLoss`, experiments/learned_intrinsic_solver/hex_energy.py:298-314) and expands it to `[B, C, 6]` on every query (experiments/learned_intrinsic_solver/solver_step.py:265); the mixed-material step recomputes a `[B, 6]` row per batch from each context's material and expands it to `[B, C, 6]` (experiments/learned_intrinsic_solver/mixed_physics.py:351-352).

**Constants summary.**

| constant | value | meaning |
|---|---|---|
| `MATRIX_FEATURE_DIM` | 45 | five 3x3 blocks |
| `BOUNDARY_DIM` | 14 | 6 face flags + 8 corner flags |
| `STATE_FEATURE_DIM` | 61 | 45 + 14 + 2 scalars |
| `CONDITIONING_DIM` | 6 | the channels above |
| `EDGE_FEATURE_DIM` | 24 | descriptor width of §1 |
| `FEATURE_SCHEMA_VERSION` | 3 | written into checkpoints so a loader can refuse another layout |
| `RMS_FLOOR` | 1e-12 | lower bound on every RMS before division |
| `CLIP` | 10.0 | symmetric clip on normalized local-frame components |

## §4. Implicit-Euler objective: inertial prediction, lumped masses and the three energy terms

**Where this lives / what this part does.** The objective that the learned optimizer minimizes is defined in two files. experiments/learned_intrinsic_solver/hex_energy.py:102 holds `make_inertial_prediction`, which builds the inertial target Y, and experiments/learned_intrinsic_solver/hex_energy.py:228 holds `HexImplicitEulerLoss`, the reference module that assembles lumped masses in its constructor and evaluates the three energy terms in `forward`. The trained path lives in experiments/learned_intrinsic_solver/mixed_physics.py: `MixedHexSolverStep.prepare` (experiments/learned_intrinsic_solver/mixed_physics.py:506) computes Y once per physical step on a CPU worker thread, `register_context` (experiments/learned_intrinsic_solver/mixed_physics.py:234) builds one `HexImplicitEulerLoss` per material only to harvest its mass and material buffers, and `MixedHexSolverStep._energy` (experiments/learned_intrinsic_solver/mixed_physics.py:389) is the batched GPU evaluation with one material per object. `_energy` runs three times per optimizer query: on the incoming candidate under `no_grad` to obtain E_before (§7), inside the autograd call that produces the gradient input (§7, §11), and on the fused positions to obtain E_after (experiments/learned_intrinsic_solver/mixed_physics.py:480).

Implicit Euler is written here as a minimization. Given the positions `X_n` and velocities `V_n` at the start of a physical step of length `dt`, the new positions `X_{n+1}` are the minimizer of an energy `E(X)`, and the learned optimizer is a procedure that lowers `E` from a starting candidate. Two definitions are needed first.

- **Lumped mass.** The continuous mass of each cubic cell, `rho h^3`, is split equally to its eight corners. The mass of corner `v` is `m_v = sum over incident cells of rho h^3 / 8`, so an interior corner shared by eight cells carries `rho h^3` and a corner of the clamped face carries half or less. "Lumped" means the mass matrix is diagonal; there is no consistent (coupled) mass matrix anywhere in this code.
- **Inertial prediction.** `Y = X_n + dt V_n + dt^2 a` is where each corner would land during the step under the known explicit acceleration `a` alone (gravity plus external force divided by lumped mass), with no elastic or viscous response. It is the target of the inertia term.

The objective is the sum of three terms, all in joules:

    E(X) = E_elastic + E_inertia + E_damping
    E_inertia = sum_v m_v |X_v - Y_v|^2 / (2 dt^2)
    E_elastic = sum_cells sum_q w_q psi(F_q(X))
    E_damping = sum_cells sum_q eta w_q ||F_q^T F_q - F_{n,q}^T F_{n,q}||_F^2 / (2 dt)

`F_q` is the deformation gradient at Gauss point `q` of a cell, `w_q = h^3 / 8` its quadrature weight (§5), `psi` the stable Neo-Hookean energy density (§5), `eta` the viscosity and `F_{n,q}` the deformation gradient of the physical-step start (§6). Why this is implicit Euler: setting `dE/dX_v = 0` gives `m_v (X_v - Y_v) / dt^2 = f_v(X)`, where `f = -dE_elastic/dX - dE_damping/dX` are the internal forces evaluated at the new positions. Substituting Y gives `X = X_n + dt V_n + dt^2 (a + f(X) / m)`, which is exactly `V_{n+1} = V_n + dt (a + f(X_{n+1}) / m)` followed by `X_{n+1} = X_n + dt V_{n+1}`. Gravity therefore enters only through Y and is not a separate potential term (experiments/learned_intrinsic_solver/hex_energy.py:112).

```python
    step = _time_step_tensor(time_step, previous_positions)
    prediction = previous_positions + step * previous_velocity
    if explicit_acceleration is not None:
        if (
            explicit_acceleration.dtype != previous_positions.dtype
            or explicit_acceleration.device != previous_positions.device
        ):
            raise TypeError("explicit_acceleration must match the positions dtype and device")
        if not torch.isfinite(explicit_acceleration).all().item():
            raise ValueError("explicit_acceleration must be finite")
        if torch.broadcast_shapes(explicit_acceleration.shape, previous_positions.shape) != previous_positions.shape:
            raise ValueError("explicit_acceleration must broadcast to the positions shape")
        prediction = prediction + step.square() * explicit_acceleration
    return prediction
```
(`experiments/learned_intrinsic_solver/hex_energy.py:135-148`)

The helper is plain tensor arithmetic after validation: positions and velocity `[B, P, 3]` in meters and m/s, `time_step` converted to a 0-d tensor of the same dtype and device, and an optional acceleration in m/s^2 that must broadcast to the positions. Gradients through all inputs are preserved, which matters for differentiable rollouts but not for training, where Y is a detached payload tensor.

```python
        with torch.no_grad(), context.lock:
            rigid = context.predictor.predict(x, velocity, force, self.time_step)
            fixed_positions = x[self._fixed_cpu].clone()
            base = x[None] @ rigid.rigid_delta_rotation.transpose(-1, -2) + rigid.rigid_delta_translation[:, None]
            zero_increment = x.new_zeros((1, len(self._rest.cell_corner_indices), 3, 3))
            candidate = context.fusion.fuse(base, zero_increment, fixed_positions[None])[0]
            inertial = make_inertial_prediction(
                x[None],
                velocity[None],
                self.time_step,
                explicit_acceleration=self._gravity_cpu + force / context.mass[:, None],
            )[0]
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:521-532`)

`prepare` snapshots one physical step for one object on the CPU in float32 (unbatched `[P, 3]` tensors), under `no_grad` and the context lock. `x` is the physical-step start `X_n`, stored in the payload as `physical_positions`; it is also the anchor of the damping term (§6). The rigid predictor and the zero-increment fusion solve build a rigid-guided `candidate`, which the trainer replaces with its own initializer (§15). The line that matters for the objective is `make_inertial_prediction(...)` with `explicit_acceleration = gravity + force / mass`: the per-corner force in newtons is divided by the lumped mass `[P] -> [P, 1]`, so every corner, including the prescribed ones, receives the same gravitational acceleration. The result is stored as `inertial_prediction` and is never modified by the K learned queries of the step; `advance` (experiments/learned_intrinsic_solver/mixed_physics.py:543) builds the next Y from the committed candidate with `velocity = (candidate - previous) / dt` and zero pinned velocity.

```python
        material(lame_lambda, "lame_lambda", 0, lower_inclusive=True)
        mu = material(lame_mu, "lame_mu", 0)
        rho = material(density, "density", 0)
        material(damping, "damping", 0, lower_inclusive=True)
        self.register_buffer("time_step", _time_step_tensor(time_step, mu).detach().clone())
        cell_mass_eighth = rho * self.quadrature_weights.sum() / 8
        mass = torch.zeros(self.particle_count, dtype=dtype)
        mass.index_add_(0, corners.reshape(-1), cell_mass_eighth[:, None].expand(-1, 8).reshape(-1))
        self.register_buffer("lumped_mass", mass)
```
(`experiments/learned_intrinsic_solver/hex_energy.py:311-319`)

Mass assembly happens once per material in the constructor of `HexImplicitEulerLoss`. The `material()` closure (experiments/learned_intrinsic_solver/hex_energy.py:298) turns a scalar or a per-cell array into a validated `[cell_count]` CPU buffer: `lame_lambda >= 0`, `lame_mu > 0`, `density > 0`, `damping >= 0`. Because the eight quadrature weights sum to `h^3`, `cell_mass_eighth` is `rho h^3 / 8` per cell, and `index_add_` over the flattened `[C, 8]` corner indices scatters it to the corners, giving `lumped_mass` of shape `[P]` in kilograms. Pinned corners keep their mass; `register_context` (experiments/learned_intrinsic_solver/mixed_physics.py:236) requires every mass to be positive, so pins participate in the inertia term as well as in the rigid predictor's momentum.

```python
    def _energy(self, positions: Tensor, inertial_prediction: Tensor, contexts, previous_positions) -> HexLossTerms:
        corners = positions[:, self.cell_corner_indices]
        deformation = torch.einsum("bcki,qkj->bcqij", corners - corners[:, :, :1], self.shape_gradients)
        material = torch.stack([context.material for context in contexts]).to(positions.device)
        lam, mu = material[:, 0, None, None], material[:, 1, None, None]
        density = stable_neo_hookean_density(deformation, mu, lam)
        elastic = (density * self.quadrature_weights[None, None]).sum((1, 2))
        masses = torch.stack([context.mass for context in contexts]).to(positions.device)
        step = positions.new_tensor(self.time_step)
        inertia = 0.5 * (masses[..., None] * (positions - inertial_prediction).square()).sum((1, 2)) / step.square()
        damping = torch.zeros_like(elastic)
        if any(context.specification["damping"] > 0 for context in contexts):
            difference = damping_metric_difference(
                positions, previous_positions, self.cell_corner_indices, self.shape_gradients
            )
            damping_density = material[:, 3, None, None] * difference.square().sum((-1, -2)) / (2 * step)
            damping = (damping_density * self.quadrature_weights[None, None]).sum((1, 2))
        return HexLossTerms(elastic + inertia + damping, elastic, inertia, damping)
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:389-406`)

This is the function whose value the optimizer minimizes and whose gradient the network reads. Key points:

- `corners = positions[:, self.cell_corner_indices]` gathers `[B, C, 8, 3]`. Subtracting the first corner of each cell (`corners - corners[:, :, :1]`) does not change F, because the shape gradients of the eight corners sum to zero (partition of unity, §5); it moves the float32 subtraction from world coordinates to cell-sized numbers and so reduces cancellation when the body is far from the origin.
- The einsum `bcki,qkj->bcqij` computes `F_q[i, j] = sum_k x_k[i] g_q[k, j]` for every batch item, cell and Gauss point: `deformation` has shape `[B, C, 8, 3, 3]` and is dimensionless.
- `material` is `[B, 4]` with the columns `(lambda, mu, rho, eta)` of each object's context; `lam` and `mu` are reshaped to `[B, 1, 1]` so they broadcast against the `[B, C, 8]` densities. Materials are constants (registered on the CPU, moved to the positions device per call), never parameters.
- `elastic` is `sum over cells and points of density * w_q`, shape `[B]`; `inertia` is the lumped-mass quadratic form divided by `dt^2` with `step` a 0-d float32 tensor on the positions device; `damping` is evaluated only if some object in the batch has `eta > 0` and is zero otherwise (§6).
- The per-context `HexImplicitEulerLoss` built in `register_context` receives no damping argument; the object's viscosity is kept as `material[3]` (experiments/learned_intrinsic_solver/mixed_physics.py:245) and applied only here.
- The return value `HexLossTerms(total, elastic, inertia, damping)` holds four `[B]` float32 tensors. Everything is plain differentiable PyTorch, so `dE/dX` is obtained by autograd (§7) rather than by hand-written force kernels. `HexImplicitEulerLoss` also supports float64 for the reference tests; the trained path is float32 only.

## §5. Stable Neo-Hookean density and eight-point quadrature

**Where this lives / what this part does.** `stable_neo_hookean_density` (experiments/learned_intrinsic_solver/hex_energy.py:151) is the elastic energy per unit rest volume `psi(F)` in J/m^3. It is called on the `[B, C, 8, 3, 3]` deformation gradients by `HexImplicitEulerLoss.forward` (experiments/learned_intrinsic_solver/hex_energy.py:382) and by `MixedHexSolverStep._energy` (experiments/learned_intrinsic_solver/mixed_physics.py:394), so it runs in every energy pass, three times per optimizer query. `hex_gauss_quadrature` (experiments/learned_intrinsic_solver/hex_energy.py:53) is a NumPy helper that runs once per `HexImplicitEulerLoss` construction (experiments/learned_intrinsic_solver/hex_energy.py:294) and once per `HexFusion` construction (experiments/learned_intrinsic_solver/fusion.py:181); `MixedHexSolverStep` copies its `shape_gradients` `[8, 8, 3]` and `quadrature_weights` `[8]` into its own buffers (experiments/learned_intrinsic_solver/mixed_physics.py:178).

Terms. The **deformation gradient** `F = dx/dX` is the 3x3 matrix mapping rest-space directions to current-space directions; its columns are the three deformed material axes. `J = det F` is the local volume ratio: 1 at rest, 0 for a collapsed cell, negative for an inverted one. The **Frobenius norm** `||F||_F^2` is the sum of the squared entries, equal to `tr(F^T F)`. The two **Lamé parameters** are the shear modulus `mu` and the first Lamé parameter `lambda`, both in pascals and both sampled per material (§14).

The elastic law is the stable Neo-Hookean density of Smith, De Goes and Kim (2018), the law used by Newton's VBD solver, in its rest-zero form:

    psi(F) = mu_NH / 2 (||F||_F^2 - 3) - mu_NH (J - 1) + lambda_NH / 2 (J - 1)^2
    P(F) = dpsi/dF = mu_NH F + (lambda_NH (J - 1) - mu_NH) cof(F)

with the mapping `mu_NH = mu` and `lambda_NH = lambda + mu`. `cof(F) = dJ/dF` is the **cofactor matrix**, whose entry `(i, j)` is the signed determinant of the 2x2 matrix left after deleting row `i` and column `j`; for invertible F it equals `J F^{-T}`, but it is a polynomial and exists for every F. The mapping is what makes the sampled parameters mean what they mean in linear elasticity: expanding `psi` to second order at `F = I`, the `mu/2 (||F||^2 - 3)` term contributes `mu` times the identity, the `-mu (J - 1)` term contributes `-mu` times the Hessian of the determinant, and `(lambda + mu)/2 (J - 1)^2` contributes `(lambda + mu)` times the outer product of `dJ/dF = I` with itself. At `F = I` the determinant's Hessian is `I (x) I - T` (`T` transposes its argument), so the three contributions sum to `mu Id - mu (I (x) I - T) + (lambda + mu) I (x) I = lambda I (x) I + mu (Id + T)`: the extra `mu` cancels and what remains is exactly the linear-elastic stiffness tensor `lambda delta_ij delta_kl + mu (delta_ik delta_jl + delta_il delta_jk)` (equivalently `lambda I (x) I + 2 mu I_sym`, i.e. `psi ~ lambda/2 (tr eps)^2 + mu ||eps||_F^2` with `eps = sym(H)`), the form asserted by experiments/learned_intrinsic_solver/tests/test_hex_energy.py:280. Its uniaxial stiffness is `lambda + 2 mu` and its shear stiffness is `mu`. Without the shift the material would behave like a linear solid with first Lamé parameter `lambda - mu`. `psi(I) = 0` and `P(I) = 0`, so the rest shape is stress free.

The density is **finite through J <= 0**. It is a polynomial in the entries of F: no `log J` barrier, no `F^{-1}`. At `F = 0` it equals `lambda / 2` with zero stress; at `F = -I` (a fully inverted cell) it equals `2 mu + 2 (lambda + mu)`. Because `lambda >= 0` and `mu > 0`, the volumetric stiffness `lambda + mu` is positive and the law is bounded below. That is why the solver accepts collapsed and inverted candidates without rejecting or shortening them (experiments/learned_intrinsic_solver/mixed_physics.py:92); the price is that nothing prevents inversion except the energy itself.

The implementation does not evaluate the formula above literally. It writes `H = F - I` and uses two exact identities:

    ||F||_F^2 - 3 = 2 tr(H) + ||H||_F^2
    J - 1 = tr(H) + s2(H) + det(H)

where `s2(H)` is the sum of the three **principal 2x2 minors** of H (the determinant of the 2x2 submatrix left after deleting row `i` and column `i`, for `i = 0, 1, 2`; equivalently the cofactors of the three diagonal entries, so the middle one is `h00 h22 - h02 h20`, not a contiguous block). The `mu tr(H)` terms cancel algebraically, leaving

    psi = mu / 2 ||H||_F^2 - mu (s2(H) + det(H)) + (lambda + mu) / 2 (tr(H) + s2(H) + det(H))^2

Why bother: near rest, `||F||^2 - 3` and `J - 1` are small differences of numbers of order one; in float32 each carries an absolute error around `1e-7`, while the true values are of order `|H|` and the energy of order `|H|^2`. For strains of `1e-3` that is a relative error of order one. Computing directly from H keeps the error relative to the small quantities themselves. The near-rest test (`test_near_rest_float32_energy_accuracy`) checks that the float32 energy is within `5e-8` J of the float64 value on a deformed grid.

```python
    increment = deformation - torch.eye(3, dtype=deformation.dtype, device=deformation.device)
    h00, h01, h02 = increment[..., 0, 0], increment[..., 0, 1], increment[..., 0, 2]
    h10, h11, h12 = increment[..., 1, 0], increment[..., 1, 1], increment[..., 1, 2]
    h20, h21, h22 = increment[..., 2, 0], increment[..., 2, 1], increment[..., 2, 2]
    trace = h00 + h11 + h22
    # First-row cofactors of H give det(H) without an inverse.
    cofactor_00 = h11 * h22 - h12 * h21
    cofactor_01 = h12 * h20 - h10 * h22
    cofactor_02 = h10 * h21 - h11 * h20
    principal_minors = cofactor_00 + (h00 * h22 - h02 * h20) + (h00 * h11 - h01 * h10)
    determinant = h00 * cofactor_00 + h01 * cofactor_01 + h02 * cofactor_02
    higher_order = principal_minors + determinant
    jacobian_minus_one = trace + higher_order
    squared_norm = increment.square().sum(dim=(-1, -2))
    return 0.5 * mu * squared_norm - mu * higher_order + 0.5 * (lam + mu) * jacobian_minus_one.square()
```
(`experiments/learned_intrinsic_solver/hex_energy.py:211-225`)

Reading the block: `cofactor_0j` are the three signed minors of the first row of H, so `determinant` is the Laplace expansion of `det H` along that row; `principal_minors` adds the two remaining diagonal minors to `cofactor_00`; `higher_order = s2 + det`; `jacobian_minus_one = tr(H) + higher_order`. Every `h_ab` has the leading shape `[B, C, 8]`, and `mu` and `lam` broadcast over it. There are no clamps, branches or `where` calls, so autograd yields the exact stress `P` (checked against the cofactor formula in `test_density_matches_naive_formula_and_cofactor_stress`). Inputs must be float32 or float64 and the material tensors must share F's dtype and device; scalars are converted.

Quadrature. Each cubic rest cell is integrated with the tensor-product two-point Gauss-Legendre rule: 2 x 2 x 2 = 8 points at `xi = +-1/sqrt(3)` in the reference cube `[-1, 1]^3`, exact for polynomials up to degree three per axis. The trilinear shape function of corner `k` with sign pattern `s_k` in `{-1, +1}^3` is `N_k(xi) = prod_a (1 + xi_a s_{k,a}) / 8`; its derivative with respect to the physical coordinate `x_a` is `s_{k,a} prod_{b != a} (1 + xi_b s_{k,b}) / (4 h)`, because `dxi/dx = 2 / h`. The Jacobian of the reference-to-physical map is `(h/2)^3`, so each weight is `h^3 / 8` and the eight weights sum to the cell volume. Corners and points are both ordered x/y/z with z varying fastest, the same order `generate_cuboid` uses for `cell_corner_indices`.

```python
    scalar = dtype.type
    signs = (2 * np.indices((2, 2, 2)).reshape(3, -1).T - 1).astype(dtype)
    points = signs / np.sqrt(scalar(3))
    factors = scalar(1) + points[:, None, :] * signs[None, :, :]
    shape_values = np.prod(factors, axis=-1, dtype=dtype) / scalar(8)
    gradients = np.empty((8, 8, 3), dtype=dtype)
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        gradients[:, :, axis] = (
            signs[None, :, axis] * np.prod(factors[:, :, other], axis=-1, dtype=dtype) / scalar(4 * cell_size)
        )
    weights = np.full(8, scalar(cell_size) ** 3 / scalar(8), dtype=dtype)
    return HexQuadrature(gradients, weights, shape_values, points)
```
(`experiments/learned_intrinsic_solver/hex_energy.py:65-77`)

`signs` is `[8, 3]`, `points` `[8, 3]`, `factors` `[8 points, 8 corners, 3]`, `shape_values` `[8, 8]` and `gradients` `[8, 8, 3]` in 1/m; the weights are `[8]` in m^3. The dtype is chosen by the caller so float64 reference modules and float32 training modules get matching rules. `shape_values` and `points` are not used by the energy; they are kept for tests and consumers that need positions of the Gauss points.

Two consequences of full integration are worth knowing. First, a single cell-center deformation gradient (which is what the frames and the network see, §8) cannot detect "hourglass" corner warps, non-affine deformations whose center gradient is exactly the identity; the eight-point rule penalizes them (`test_full_quadrature_sees_center_hourglass`), so no artificial hourglass stabilization is needed. Second, cost: every energy pass materializes several `[B, C, 8, 3, 3]` float32 intermediates; for the campaign grid (C = 4,000) and a batch of 16 that is 4.6 million entries per intermediate, and three passes per query.

## §6. Metric damping anchored at the physical-step start

**Where this lives / what this part does.** `damping_metric_difference` (experiments/learned_intrinsic_solver/damping.py:16) computes, at every Gauss point, the change of the metric tensor `C = F^T F` between the candidate and the physical-step start. `HexImplicitEulerLoss.forward` (experiments/learned_intrinsic_solver/hex_energy.py:392) and `MixedHexSolverStep._energy` (experiments/learned_intrinsic_solver/mixed_physics.py:401) turn it into the damping energy; it runs in every energy pass whenever any object in the batch has a positive damping coefficient. The anchor `previous_positions` is the payload's `physical_positions`, written by `prepare` (experiments/learned_intrinsic_solver/mixed_physics.py:535) and replaced only by `advance` (experiments/learned_intrinsic_solver/mixed_physics.py:543), so it is constant across the K inner queries of one physical step. The same module holds `pack_damping_features` (experiments/learned_intrinsic_solver/damping.py:47), a 48-value packing of the same metric change; it is used by tests only, the schema-3 network inputs (§3) do not include it.

Terms. The **metric tensor** (right Cauchy-Green tensor) `C = F^T F` is a symmetric, dimensionless 3x3 matrix whose entry `(i, j)` is the dot product of deformed material axes `i` and `j`. It records the lengths of and angles between the material axes and nothing about their orientation, because `(R F)^T (R F) = F^T F` for any rotation R. The **viscosity** `eta` in Pa s is one number per material in the mixed step (per cell in `HexImplicitEulerLoss`).

    E_damping = sum_cells sum_q eta w_q ||C_q(X) - C_q(X_n)||_F^2 / (2 dt)
              = (dt / 2) sum_cells sum_q eta w_q ||(C_q(X) - C_q(X_n)) / dt||_F^2

The second line shows the meaning: the rate of change of the metric over the step, squared, weighted by viscosity and rest volume and integrated over `dt`. This is a dissipation potential whose gradient is a viscous force proportional to the metric rate. Units: `Pa s * m^3 / s = J`. The resulting force matches the solid damping of Newton's VBD tetrahedral kernel after rest-volume and shape-gradient assembly (`test_native_vbd_tet_force_matches_hex_metric_stress`).

Why rigid motion gives zero. If the whole body moves rigidly from the anchor, `X = R X_n + t` with any finite rotation R and translation t, then `F_q(X) = R F_q(X_n)` at every Gauss point, so `C_q(X) = C_q(X_n)` exactly and both the damping energy and its force vanish. A damping term on position differences, `||X - X_n||^2`, would instead resist rotation and free fall, which are motions the step must allow. Because C is quadratic in F, the term is a fourth-degree polynomial in positions, finite for inverted and collapsed candidates like the elastic term.

Why the anchor is the physical-step start and nothing else. The K learned queries within one step are iterations on one fixed minimization problem, so the objective must be the same function of X for all of them; the damping term is the only one with a second position argument, and holding it at `X_n` keeps the landscape fixed. Passing the current candidate as the anchor would make the term vanish and remove the viscous force (`test_physical_anchor_is_required_and_changes_force`); passing Y would be wrong too, since Y already contains `dt V_n` (the validation test asserts the anchor differs from both).

```python
    corners = positions[:, cells]
    previous_corners = previous_positions[:, cells]
    # Partition of unity permits subtracting one corner to reduce cancellation
    # under world translation without changing either material gradient.
    deformation = torch.einsum("bcki,qkj->bcqij", corners - corners[:, :, :1], shape_gradients)
    previous_deformation = torch.einsum(
        "bcki,qkj->bcqij", previous_corners - previous_corners[:, :, :1], shape_gradients
    )
    return deformation.transpose(-1, -2) @ deformation - previous_deformation.transpose(-1, -2) @ previous_deformation
```
(`experiments/learned_intrinsic_solver/damping.py:36-44`)

The function reuses the first-corner subtraction and the einsum of §4 for both operands and returns `[B, N, 8, 3, 3]` symmetric matrices. Both operands keep their autograd history (needed for differentiable rollouts through several steps); in the learned step `previous_positions` is a detached payload tensor, so only the candidate side carries gradient.

In `_energy` (excerpt in §4, experiments/learned_intrinsic_solver/mixed_physics.py:400) the per-object coefficient `material[:, 3, None, None]` multiplies `difference.square().sum((-1, -2))`, the squared Frobenius norm over all nine entries (off-diagonal entries counted twice, as in Newton VBD), divided by `2 dt`, weighted by `w_q` and summed over cells and points. The check `any(context.specification["damping"] > 0 ...)` means the metric difference is computed for the whole batch as soon as one member is damped; undamped members add exactly zero because their `eta` is zero. `HexImplicitEulerLoss` raises if `previous_positions` is missing while its damping is positive (experiments/learned_intrinsic_solver/hex_energy.py:366); `MixedHexSolverStep.energy` applies the same rule per batch (experiments/learned_intrinsic_solver/mixed_physics.py:384), while `forward` and `prepare_inputs` always require the anchor because the physical axis-change input block (§3) needs it regardless of damping.

```python
    def advance(self, payload: dict) -> dict:
        """Commit candidate displacement to velocity and prepare the next step once."""
        candidate = self._cpu_snapshot(payload["candidate"], "candidate")
        previous = self._cpu_snapshot(payload["physical_positions"], "physical_positions")
        velocity = (candidate - previous) / self.time_step
        velocity[self._fixed_cpu] = 0
        return self.prepare(payload["context_id"], candidate, velocity, forces=payload.get("forces"))
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:543-549`)

`advance` is where the anchor moves: the committed candidate becomes the next `physical_positions`, the velocity is the displacement over `dt` with prescribed corners set to zero, and `prepare` recomputes Y and the rigid initializer. Until then, every query of the step differentiates against the same `X_n`. Two smaller points: `HexImplicitEulerLoss` can load checkpoints written before the `damping` buffer existed only if its configured damping is zero (experiments/learned_intrinsic_solver/hex_energy.py:321), and the viscosity also enters the loss scale floor as `eta / dt` (§7).

## §7. Two gradients: the position gradient by autograd and the weight gradient through the fusion backward; the LeCO loss and the energy floor

**Where this lives / what this part does.** Two different derivatives of the same objective are taken in every training query. The **position gradient** `dE/dX` in newtons is computed by `objective_gradient` (experiments/learned_intrinsic_solver/input_assembly.py:153), called from `assemble_inputs` (experiments/learned_intrinsic_solver/input_assembly.py:245) on behalf of `MixedHexSolverStep._prepare_inputs` (experiments/learned_intrinsic_solver/mixed_physics.py:353); it is then projected to per-cell axis-increment gradients by `HexFusion.project_gradient` (experiments/learned_intrinsic_solver/fusion.py:310), one object at a time (experiments/learned_intrinsic_solver/mixed_physics.py:345). Its norm is reported as `force_residual_norm` (experiments/learned_intrinsic_solver/mixed_physics.py:485), and `mixed_validation._free_force_residual_norms` (experiments/learned_intrinsic_solver/mixed_validation.py:67) recomputes the same quantity as the validation residual. The **weight gradient** starts at the loss `local_objective` (experiments/learned_intrinsic_solver/train_mixed.py:227) and runs, in `loss.backward()` (experiments/learned_intrinsic_solver/train_mixed.py:711), from E_after through `_energy` at the fused positions, through `_FusionSolve.backward` (experiments/learned_intrinsic_solver/fusion.py:62) per object, through the world increment (experiments/learned_intrinsic_solver/mixed_physics.py:463) into the network. `MixedHexSolverStep.energy_floor` (experiments/learned_intrinsic_solver/mixed_physics.py:408) supplies the floor of the loss scale. Cadence: the position gradient once per query per object (one transposed CPU solve each); the weight gradient once per batch (again one transposed solve per object).

Terms. In reverse-mode autodiff the **cotangent** of an intermediate quantity is the derivative of the final scalar with respect to that quantity; backpropagating through a linear map applies the map's **adjoint**, which for a real matrix is its transpose. For a linear solve `x = K^{-1} r` the adjoint is `K^{-T}`, a solve with the transposed matrix. The **force residual** is `dE/dX` itself: its units are J/m = N, it is zero at the exact implicit-Euler solution, and its norm over free corners measures how far a candidate is from solving the step.

**The position gradient.**

```python
    with torch.enable_grad():
        candidate = positions.detach().requires_grad_(True)
        total = energy_total(candidate, inertial_prediction.detach(), previous_positions.detach())
        gradient = torch.autograd.grad(total.sum(), candidate)[0]
    gradient = gradient.detach()
    gradient[:, fixed_indices] = 0
    return gradient
```
(`experiments/learned_intrinsic_solver/input_assembly.py:176-182`)

`torch.enable_grad()` makes this work inside the `no_grad` validation loop. The candidate is detached and re-marked as requiring grad, so the derivative is taken at the current shape and does not flow back into whatever produced the candidate (the previous query). Y and `X_n` are detached, which fixes the inertial and damping anchors. `total.sum()` over the batch gives each object its own gradient because the objects are independent. Rows of prescribed corners are zeroed: their positions are not unknowns, so the force acting on them carries no information for the optimizer. The result is a detached `[B, P, 3]` float32 tensor in newtons.

Projection through the fusion adjoint. Fusion (§9) solves `K_ff d_free = B D - K_fc d_fixed` for the free-corner displacements, where D `[C, 3, 3]` are the per-cell axis increments, `K_ff` the free block of the weighted gradient-fit stiffness and B the target operator. For a fixed base and fixed pins, `d_free` is linear in D with `d d_free / d D = K_ff^{-1} B`, so the gradient of any scalar function of the positions with respect to D is `G = B^T K_ff^{-T} g_free`.

```python
    def _adjoint_targets(self, free_gradient: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Pull free-corner cotangents back to axis increments through the transpose solve.

        Args:
            free_gradient: Free-corner position cotangents, shape [B, F, 3], in
                ``free_indices`` order, on any device. Autograd history is
                discarded.

        Returns:
            The packed adjoint columns ``K_ff^{-T} g_free`` with shape [F, 3B]
            and the axis-increment gradients ``unpack(B^T K_ff^{-T} g_free)``
            with shape [B, C, 3, 3] (axes in columns), both as CPU NumPy arrays
            in the working precision.
        """
        batch_count = free_gradient.shape[0]
        columns = _columns(free_gradient.detach().cpu().contiguous().numpy())
        adjoint = self._solve(columns, transpose=True)
        target_rows = _batch(self._target_operator.T @ adjoint, batch_count)
        return adjoint, target_rows.reshape(batch_count, self.cell_count, 3, 3).swapaxes(-1, -2)
```
(`experiments/learned_intrinsic_solver/fusion.py:239-257`)

`_columns` packs `[B, F, 3]` into a Fortran-ordered `[F, 3B]` matrix, one right-hand side per batch item and world axis. `_solve(columns, transpose=True)` runs the PARDISO transposed solve (`iparm[11] = 2`, experiments/learned_intrinsic_solver/pardiso.py:238) on the cached factor of `K_ff`; the matrix is symmetric, so the transposed solve equals the plain one mathematically, and differs from it only by roundoff because PARDISO factorizes `K_ff` as a nonsymmetric LU (matrix type 11, experiments/learned_intrinsic_solver/pardiso.py:161) whose transposed sweep is a different computation, but the code requests the exact adjoint anyway. `_target_operator.T @ adjoint` applies `B^T`; `_batch` and the final `swapaxes` restore `[B, C, 3, 3]` with material axes in columns, the layout of `world_axis_increments`. The pairing `dot(G, D) = dot(g_free, delta X_free)` has units N m = J, so G is an energy per unit dimensionless increment. In the mixed step each object has its own factor because the fit weights depend on the material, `mu (3 - mu / (lambda + mu)) h^3` (experiments/learned_intrinsic_solver/mixed_physics.py:239), so `project_gradient` is called per object with batch size one and the results are concatenated (experiments/learned_intrinsic_solver/mixed_physics.py:345). What happens to G afterwards, rotation into the cell frame, RMS normalization and clipping, is §11.

**The weight gradient.**

```python
        prediction = self.network(inputs.local_axes, inputs.state_features, inputs.edge_features, inputs.conditioning)
        world_increment = inputs.frames @ (prediction.local_target_axes - inputs.local_axes)
...
        fused = torch.cat(
            [
                context.fusion.fuse(positions[i : i + 1], world_increment[i : i + 1], fixed_positions[i : i + 1])
                for i, context in enumerate(contexts)
            ]
        )
        loss = self._energy(fused, inertial_prediction, contexts, previous_positions)
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:462-463, 474-480`)

The forward chain that the backward pass retraces: the network returns `local_target_axes` `[B, C, 3, 3]`; the world increment is `R (A_target - A)`. The frames R are constants: `closest_proper_rotations` detaches the deformation before decomposing it (experiments/learned_intrinsic_solver/frames.py:241) and the tests assert `frames.requires_grad` is False. The local axes `A = R^T F(candidate)` depend only on the candidate, a detached payload tensor, so in training the only live input of the increment is the network output. Fusion runs per object on a `[1, P, 3]` slice and the results are concatenated; `_energy` is evaluated on the fused shape and its `total` is E_after.

```python
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient_output):
        fusion = ctx.fusion
        batch_count = ctx.batch_count
        fixed, free = fusion._indices(ctx.device)
        adjoint, gradient_targets = fusion._adjoint_targets(gradient_output[:, free])
        boundary_transfer = _batch(fusion._fixed_coupling.T @ adjoint, batch_count)
        boundary_transfer = torch.from_numpy(np.ascontiguousarray(boundary_transfer)).to(device=ctx.device)
        gradient_base = gradient_output.clone()
        gradient_base[:, fixed] = boundary_transfer
        gradient_fixed = gradient_output[:, fixed] - boundary_transfer
        return (
            None,
            gradient_base,
            torch.from_numpy(np.ascontiguousarray(gradient_targets)).to(device=ctx.device),
            gradient_fixed,
        )
```
(`experiments/learned_intrinsic_solver/fusion.py:62-79`)

Backward from E_after: autograd first produces `dE_after/dX_fused`, the same stress-and-mass assembly as the position gradient but at the fused shape, then calls this function with it as `gradient_output` `[1, P, 3]`. `_adjoint_targets(gradient_output[:, free])` is the same transposed solve as `project_gradient`; its second output is the cotangent of the axis increments, `[1, C, 3, 3]`, which continues through `frames @ (...)` (autograd applies `R^T`) to `local_target_axes` and into every network parameter. The other two outputs handle the inputs that the increment does not cover: the base enters the solve only through `fixed_delta = fixed_positions - base[:, fixed]`, so `boundary_transfer = K_fc^T K_ff^{-T} g_free` is the whole cotangent of the base's fixed rows (their direct path is cut because `result[:, fixed]` is overwritten), while `fixed_positions` receives it with the opposite sign plus the direct pin gradient, `gradient_fixed = g_fixed - boundary_transfer`, because `X_fixed = fixed_positions` is assigned outright; free rows of the base get `g_free` unchanged because `X_free = base_free + delta`. In training both `positions` and `fixed_positions` are payload tensors without gradient, so those two outputs are computed and discarded. `once_differentiable` rules out second derivatives; the CPU factor is shared by the forward solve, the gradient input and this backward, and it is never differentiated (topology and weights are fixed at construction).

**The LeCO loss.** LeCO (LearnedClothOptimizer, the external learned cloth-optimizer codebase used as the reference implementation; see notes/decision-review.md Q8-Q9) is the source of the learned-optimizer conventions this project adopted: the loss below and the gradient-input normalization of §11 (experiments/learned_intrinsic_solver/features.py:14). The energy floor is the project's own material-aware choice, not LeCO's 1 J value. The loss compares the energy after one update with the energy immediately before it.

```python
def local_objective(after, before, floor, *, increase_weight=1.0):
    """Return per-member LeCO losses for one update; gradients reach only ``after``.

    ``scale = max(|before|, floor)`` is detached, where ``before`` is the
    energy of the candidate immediately before this update and ``floor`` the
    material-aware energy floor [J]. The loss is
    ``asinh(after / scale) + increase_weight * relu((after - before) / scale)``.

    Args:
        after: Energies after the update, shape [B]; the only differentiable input.
        before: Energies immediately before the update, shape [B]; detached.
        floor: Positive finite floor [J], shape [B] or scalar; detached.
        increase_weight: Nonnegative weight of the energy-increase penalty.
    """
    import torch

    before = before.detach()
    floor = torch.as_tensor(floor, dtype=before.dtype, device=before.device).detach()
    if not torch.isfinite(floor).all() or (floor <= 0).any():
        raise ValueError("energy floor must be finite and positive")
    scale = torch.maximum(before.abs(), floor)
    return torch.asinh(after / scale) + increase_weight * torch.relu((after - before) / scale)
```
(`experiments/learned_intrinsic_solver/train_mixed.py:227-248`)

`asinh(x) = ln(x + sqrt(x^2 + 1))` is odd, equals `x` for `|x| << 1` and grows like `sign(x) ln(2|x|)` for `|x| >> 1`. With `scale = max(|E_before|, floor)`, an update that leaves the energy unchanged scores `asinh(1) = 0.88`, one that halves it scores `0.48`, and one that multiplies it by ten scores `3.0` plus the increase penalty `9 w`. The derivative with respect to `E_after` is `1 / (scale sqrt(1 + (E_after / scale)^2))` plus `w / scale` when the energy went up: it is largest, `1 / scale`, when the energy is near zero, and decays like `1 / E_after` for large energies, so one exploded proposal cannot dominate the batch mean. Only `after` carries gradient; `before` and `floor` are detached, and the function raises on a non-positive or nonfinite floor.

```python
                    with torch.no_grad():
                        previous = step.energy(
                            batch["candidate"],
                            batch["inertial_prediction"],
                            batch["context_ids"],
                            previous_positions=batch["physical_positions"],
                        ).total
                        floor = step.energy_floor(batch["context_ids"])
                    if not torch.isfinite(previous).all():
                        raise ValueError("nonfinite input energy")
...
                try:
                    with torch.autocast(device_type=device.type, enabled=False):
                        result = _checked_forward(module, step, batch)
                        losses = local_objective(
                            result.loss.total, previous, floor, increase_weight=config.energy_increase_weight
                        )
```
(`experiments/learned_intrinsic_solver/train_mixed.py:678-687, 697-702`)

`previous` (E_before) is evaluated under `no_grad` on `batch["candidate"]` with the physical anchor. At inner iteration one the candidate is the trainer's initializer; at later iterations it is the previous update's fused output, stored by `record.payload["candidate"] = result.positions[i].detach()` (experiments/learned_intrinsic_solver/train_mixed.py:777). The scale is therefore the energy immediately before this update, not the energy at the start of the step (`test_later_inner_iterations_normalize_by_the_carried_candidate_energy`). A nonfinite E_before is reported as a preparation failure on all ranks (the error is exchanged through `_all_ranks_ok`) and raises `RuntimeError("mixed input preparation failed: ...")`, which is not caught inside the epoch loop: the run-level handler writes `failure_rank_{rank}.pt` on every rank and `failure.json` on rank 0, marks the report `failed`, and re-raises, so the whole training run ends (experiments/learned_intrinsic_solver/train_mixed.py:686-692 and experiments/learned_intrinsic_solver/train_mixed.py:914-939). `losses` is `[B]` and the optimizer minimizes its mean; `energy_increase_weight` defaults to 1.0 (experiments/learned_intrinsic_solver/train_mixed.py:64).

**The energy floor.**

```python
    def energy_floor(self, context_ids: tuple[str, ...]) -> Tensor:
        """Return the detached material-aware energy floor [J], shape [B] float32.

        ``floor = c * eps32 * V * (lambda + 2 mu + eta / dt + rho h^2 / dt^2)``
        with ``V`` the total rest volume, ``eps32 = 2**-23`` and ``c`` the
        constructor's ``energy_floor_scale`` (default 1). Evidence:
        ``generated/verification/energy_floor_calibration/SUMMARY.md``
        (provisional ``c = 1``).
        """
        if not isinstance(context_ids, tuple) or not context_ids:
            raise ValueError("context_ids must be a nonempty tuple of identifiers")
        contexts = self._lookup(context_ids, len(context_ids))
        material = torch.stack([context.material for context in contexts]).to(torch.float64)
        lam, mu, rho, damping = material.unbind(-1)
        step, size = self.time_step, self.cell_size
        modulus = lam + 2 * mu + damping / step + rho * size**2 / step**2
        floor = self.energy_floor_scale * _FLOAT32_EPSILON * self.rest_volume * modulus
        return floor.to(dtype=torch.float32, device=self.rest_positions.device)
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:408-425`)

Dividing by `|E_before|` alone would blow up for candidates whose energy is already near zero: an essentially undeformed body sitting on its inertial prediction with no metric change, where `E_before` is float32 rounding noise. Proximity to the minimizer is not the trigger; with the default gravity and pinned face the minimum of `E` is itself positive (the inertia term `m |X* - Y|^2 / 2 dt^2` alone is about 0.03 J for the campaign body, well above the floor). The floor is the float32 resolution of an energy of the body's natural size: `eps32 = 2^-23` is the float32 machine epsilon (the gap between 1 and the next representable float32, `torch.finfo(torch.float32).eps`; the code comment at experiments/learned_intrinsic_solver/mixed_physics.py:57 calls it the unit roundoff, which strictly is half that under round-to-nearest), `V` the total rest volume and the bracket a sum of moduli in pascals, `lambda + 2 mu` (uniaxial stiffness), `eta / dt` (viscous modulus) and `rho h^2 / dt^2` (inertial modulus, mass density times the square of one cell per step). `eps32 V modulus` is therefore the smallest energy difference that float32 can represent for that body; dividing by anything smaller would amplify rounding noise. The product is formed in float64 and cast to float32, detached, shape `[B]`, one value per object, multiplied by the constructor's `energy_floor_scale` (`c`, default 1). For scale: with `V = 0.0625 m^3` (the campaign grid), `lambda + 2 mu = 1e5 Pa` gives `7.5e-4 J`; `rho = 1000 kg/m^3`, `h = 0.025 m`, `dt = 1/300 s` gives an inertial modulus of `5.6e4 Pa` and another `4.2e-4 J`. Validation records the same floor per sample (experiments/learned_intrinsic_solver/mixed_validation.py:171) and recomputes the first-update loss from it in plain Python (experiments/learned_intrinsic_solver/mixed_validation.py:372).

**The position gradient as the validation residual.**

```python
    import torch

    with torch.enable_grad():
        positions = batch["candidate"].detach().clone().requires_grad_(True)
        energy = step.energy(
            positions,
            batch["inertial_prediction"].detach(),
            batch["context_ids"],
            previous_positions=batch["physical_positions"].detach(),
        ).total
        gradient = torch.autograd.grad(energy.sum(), positions)[0]
    gradient = gradient.detach()
    gradient[:, step.fixed_indices] = 0
    if not torch.isfinite(gradient).all():
        raise ValueError("nonfinite free-corner force residual")
    return torch.linalg.vector_norm(gradient.flatten(1).double(), dim=1).cpu().tolist()
```
(`experiments/learned_intrinsic_solver/mixed_validation.py:74-89`)

This is the same derivative as `objective_gradient`, computed through the public `step.energy` on a cloned candidate, with the pins zeroed and the norm accumulated in float64. The validation loop records it at iteration 0 (the initial candidate) and after each of the K queries (experiments/learned_intrinsic_solver/mixed_validation.py:180), always with the payload's `physical_positions` as anchor; the checkpoint-selection metric is the mean over samples of the last optimization-phase value (`SELECTION_AGGREGATION`, experiments/learned_intrinsic_solver/mixed_validation.py:29). The `force_residual_norm` returned by `forward` is the float32 norm of the same gradient at the pre-update candidate, so training reports and validation curves measure one quantity.

## §8. Center deformation and the closest proper frame with the clamped-face tie-break

**Where this lives / what this part does.** `center_deformation` in experiments/learned_intrinsic_solver/features.py:144 computes one deformation gradient per cell from the shared corner positions; the constant `center_gradients` buffer it uses is built once in `MixedHexSolverStep.__init__` at experiments/learned_intrinsic_solver/mixed_physics.py:184. The frame code is experiments/learned_intrinsic_solver/frames.py: `select_reference_corners` (experiments/learned_intrinsic_solver/frames.py:58) runs once per step module at construction (experiments/learned_intrinsic_solver/mixed_physics.py:192), while `reference_rotation` (experiments/learned_intrinsic_solver/frames.py:111) and `closest_proper_rotations` (experiments/learned_intrinsic_solver/frames.py:168) run once per network query for all B objects and C cells at once, called from `assemble_inputs` at experiments/learned_intrinsic_solver/input_assembly.py:233. This is the first stage of input assembly: it turns positions into the per-cell frame `R` and the local axes `A = R^T F` that every later feature block is expressed in.

**The cell-center deformation gradient.** The deformation gradient `F` is the 3x3 matrix that maps a material (rest) direction to its current direction: `dx = F dX`. For a trilinear hexahedron it varies inside the cell; this code uses its value at the cell center only. With the eight corner sign patterns `s_k` in `{-1, +1}^3` (z-fast order: the z sign flips fastest, then y, then x) and a cube of edge `h`, the trilinear shape function of corner `k` is `N_k = prod_i (1 + s_ki xi_i) / 8` on the reference cube, and its spatial gradient at the center is `s_kj / (4 h)`. That constant is the buffer `center_gradients` `G = signs / (4 h)`, shape `[8, 3]`, unit 1/m:

```python
        signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)
        self.register_buffer("center_gradients", signs / (4 * rest.cell_size))
        flags = torch.zeros(count, dtype=torch.float32)
        flags[self.fixed_indices] = 1
        boundary = torch.cat(
            (torch.tensor(rest.cell_exposed_faces, dtype=torch.float32), flags[self.cell_corner_indices]), -1
        )
        self.register_buffer("boundary_features", boundary)
        corners = select_reference_corners(rest.corner_rest_positions, fixed)
        reference = torch.zeros(0, dtype=torch.long) if corners is None else torch.as_tensor(corners, dtype=torch.long)
        self.register_buffer("reference_corners", reference)
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:184-194`)

Then `F_ij = sum_k (x_k - x_0)_i G_kj`, or in einsum form:

```python
def center_deformation(
    positions: torch.Tensor, cell_corner_indices: torch.Tensor, center_gradients: torch.Tensor
) -> torch.Tensor:
...
    corners = positions[:, cell_corner_indices]
    return torch.einsum("bcki,kj->bcij", corners - corners[:, :, :1], center_gradients)
```
(`experiments/learned_intrinsic_solver/features.py:144-146, 190-191`)

- Shapes: `positions [B, P, 3]` in meters, `cell_corner_indices [C, 8]` (`torch.long`), so `corners` is `[B, C, 8, 3]` and `F` is `[B, C, 3, 3]`, dimensionless. Column `j` of `F` is the image of the j-th material unit direction; it is not normalized.
- Subtracting corner zero is exact because the eight gradients of each column sum to zero (four `+1` and four `-1`), and it keeps the float32 sum well conditioned when the object is far from the origin.
- Inverted cells (`det F < 0`) and collapsed cells (`det F = 0`) give a finite `F`; nothing here rejects them. `center_deformation` is differentiable in `positions`, and nothing is detached inside it.
- The same constructor block also registers two more buffers: `boundary_features` (six exposed-face flags plus eight fixed-corner flags, `[C, 14]`) and `reference_corners`, the long `[3]` tensor of clamped-face corner IDs, or an empty long tensor when the prototype has fewer than three noncollinear pins.

**Closest proper rotation.** A "proper" rotation is an orthonormal matrix with determinant `+1` (right-handed). For a singular value decomposition `F = U diag(s1, s2, s3) Vh` with `s1 >= s2 >= s3 >= 0`, the rotation closest to `F` in the Frobenius norm is `U Vh` when `det(U Vh) > 0` (the classical polar factor), and `U diag(1, 1, -1) Vh` when `det(U Vh) < 0`, meaning the handedness flip is applied to the smallest singular direction. Both cases are one formula:

```python
def _proper_rotation(left: Tensor, right_transpose: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``U @ diag(1, 1, d) @ Vh`` and ``d = sign(det(U @ Vh))`` with zero treated as +1."""
    determinant = torch.linalg.det(left @ right_transpose)
    orientation = torch.where(determinant < 0, -1.0, 1.0).to(left.dtype)
    scale = torch.ones_like(left[..., 0, :])
    scale[..., 2] = orientation
    return (left * scale[..., None, :]) @ right_transpose, orientation
```
(`experiments/learned_intrinsic_solver/frames.py:159-165`)

`d = sign(det(U Vh))` with zero counted as `+1`; the sign is multiplied into the third column of `U` before the product. A cell with `det F < 0` therefore gets a right-handed frame whose local axes `A = R^T F` have `det A < 0`: the inversion is kept in `A`, not hidden in `R`.

**Ties and the clamped-face tie-break.** The minimizer is not always unique. If `F` has rank one (two singular values zero) every rotation mapping the leading right singular vector to the leading left one is equally close; if `F = 0` every rotation is; if `F` is inverted with `s2 = s3`, flipping either of the two smallest directions is equally close. In terms of squared Frobenius distance, the runner-up proper rotation is worse than the winner by exactly `4 gap`, where `gap = s2 + s3` for `d > 0` and `gap = s2 - s3` for `d < 0`. The code declares a tie when `gap <= tie_tolerance * max(s1, 1)`, with `tie_tolerance = 1e-4`: relative to the largest stretch for large cells, absolute below unit stretch so a tiny collapsed cell is still a tie.

```python
    with torch.no_grad():
        deformation = deformation.detach()
        left, singular_values, right_transpose = torch.linalg.svd(deformation)
        frames, orientation = _proper_rotation(left, right_transpose)
        scale = singular_values[..., 0].clamp_min(1.0)
        gap = torch.where(
            orientation > 0,
            singular_values[..., 1] + singular_values[..., 2],
            singular_values[..., 1] - singular_values[..., 2],
        )
        tie_mask = gap <= tolerance * scale
        if reference is not None and bool(tie_mask.any()):
            epsilon = (tolerance * scale)[..., None, None]
            tie_reference = reference[:, None].expand(-1, deformation.shape[1], -1, -1)[tie_mask].to(torch.float64)
            perturbed = deformation[tie_mask].to(torch.float64) + epsilon[tie_mask].to(torch.float64) * tie_reference
            tie_left, _, tie_right_transpose = torch.linalg.svd(perturbed)
            frames[tie_mask] = _proper_rotation(tie_left, tie_right_transpose)[0].to(deformation.dtype)
    return FrameResult(frames, tie_mask, singular_values)
```
(`experiments/learned_intrinsic_solver/frames.py:240-257`)

- Everything is under `torch.no_grad()` and `deformation` is detached first, so the returned `frames [B, C, 3, 3]`, `tie_mask [B, C]` (bool) and `singular_values [B, C, 3]` carry no autograd history.
- Tie cells with a reference are recomputed from the perturbed matrix `G = F + eps R_ref`, `eps = tie_tolerance * max(s1, 1)`, using the same formula. To first order this picks, among the equally close rotations, the one closest to `R_ref`; for `F = 0` it returns `R_ref` exactly. `eps` has the size of the tie threshold itself, so the perturbation is of the order of the ambiguity it resolves and stays far below the resolved singular values of a non-tie cell.
- The perturbed problem is conditioned like `1 / tie_tolerance = 1e4`. In float32 that would leave roughly three significant digits in the tie-breaking directions, so the tie cells are gathered (`deformation[tie_mask]`, `[T, 3, 3]`), promoted to float64, solved, and cast back to the working dtype. Only tie cells are written; `frames[tie_mask] = ...` leaves every other cell bit-identical to the plain formula.
- Without a reference (`reference is None`) tie cells keep the plain formula and the mask still reports them; this is the documented fallback for prototypes without three noncollinear prescribed corners.
- The reference is validated before use: shape `[B, 3, 3]`, same dtype and device as `F`, finite, `max |R_ref^T R_ref - I| <= 1e3 * eps_dtype` over all entries and batch items, where `eps_dtype` is the machine epsilon of the working dtype (`torch.finfo(dtype).eps`, so the bound is about 1.2e-4 for float32 and 2.2e-13 for float64; this is unrelated to the tie perturbation `eps` above), and `det R_ref > 0` for every batch item (experiments/learned_intrinsic_solver/frames.py:226-238).
- Caveat stated in the docstring at experiments/learned_intrinsic_solver/frames.py:194: the frame is discontinuous across the tie threshold, and for tie cells it moves with the reference. This is a coordinate choice for the network; `R A = F` holds exactly either way.

**Where the reference comes from.** The reference is a frame attached to the clamped face of the prototype, so it rotates with the whole problem. Three ordered pin IDs are chosen once from rest geometry:

```python
    pinned = positions[fixed]
    first = int(np.lexsort((pinned[:, 2], pinned[:, 1], pinned[:, 0]))[0])
    distances = np.linalg.norm(pinned - pinned[first], axis=1)
    farthest = float(distances.max())
    if farthest <= 0.0:
        return None
    second = int(np.flatnonzero(distances >= farthest * (1.0 - CORNER_TIE_RATIO))[0])
    areas = np.linalg.norm(np.cross(pinned[second] - pinned[first], pinned - pinned[first]), axis=1)
    largest = float(areas.max())
    if largest < COLLINEAR_AREA_RATIO * farthest**2:
        return None
    third = int(np.flatnonzero(areas >= largest * (1.0 - CORNER_TIE_RATIO))[0])
    return fixed[[first, second, third]].astype(np.int64)
```
(`experiments/learned_intrinsic_solver/frames.py:96-108`)

`p0` is the pin with the lexicographically smallest rest position (x, then y, then z; `np.lexsort` takes its keys last-first), `p1` the pin farthest from `p0`, `p2` the pin maximizing the parallelogram area `|(p1 - p0) x (p2 - p0)|`. Distances or areas equal within `CORNER_TIE_RATIO = 1e-9` relative resolve to the smallest ID, which is what the `flatnonzero(...)[0]` on the sorted `fixed` array gives. The function returns `None` (and the step stores an empty buffer) when fewer than three unique pins exist or the largest area is below `COLLINEAR_AREA_RATIO * |p1 - p0|^2`.

Each query then builds the current reference rotation from the current positions of those three corners:

```python
    with torch.no_grad():
        corners = positions.detach()[:, indices.to(positions.device)]
        if not torch.isfinite(corners).all():
            raise ValueError("reference corner positions must be finite")
        edge = corners[:, 1] - corners[:, 0]
        edge_norm = torch.linalg.vector_norm(edge, dim=-1, keepdim=True)
        if bool((edge_norm < DEGENERATE_NORM).any()):
            raise ValueError("reference corners 0 and 1 coincide")
        first = edge / edge_norm
        normal = torch.linalg.cross(first, corners[:, 2] - corners[:, 0])
        normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        if bool((normal_norm < DEGENERATE_NORM).any()):
            raise ValueError("reference corners are collinear")
        normal = normal / normal_norm
        second = torch.linalg.cross(normal, first)
        return torch.stack((first, second, normal), dim=-1)
```
(`experiments/learned_intrinsic_solver/frames.py:141-156`)

- `e1 = normalize(x1 - x0)`, `n = normalize(e1 x (x2 - x0))`, `e2 = n x e1`; columns `[e1, e2, n]` form a proper rotation by construction. Shape `[B, 3, 3]`, one reference per object, broadcast over cells inside `closest_proper_rotations`.
- Computed under `torch.no_grad()` from `positions.detach()`, so it is a constant for autograd. A degenerate edge or normal (norm below `DEGENERATE_NORM = 1e-12` m) raises rather than producing NaNs.
- Rigid equivariance: rotating all positions by `Q` maps `F` to `Q F` (so `U` to `Q U`) and `R_ref` to `Q R_ref`; both the plain and the perturbed formula then return `Q R`. Translation leaves both untouched.

**Assembly.** The three pieces meet at the top of `assemble_inputs`:

```python
    cells, gradients = step.cell_corner_indices, step.center_gradients
    deformation = center_deformation(positions, cells, gradients)
    tie_mask = None
    if frames is None:
        reference = None
        if step.reference_corners.numel():
            reference = reference_rotation(positions, step.reference_corners)
        result = closest_proper_rotations(deformation, reference)
        frames, tie_mask = result.frames, result.tie_mask
    axes = to_local(frames, deformation)
```
(`experiments/learned_intrinsic_solver/input_assembly.py:233-242`)

`to_local(frames, deformation)` is `R^T F` (experiments/learned_intrinsic_solver/features.py:223) and is evaluated outside any `no_grad` block, so the local axes `A` are differentiable in `positions` through `F` while `R` acts as a constant. When a caller replays supplied `frames` (validation and replay paths), the decomposition is skipped and `tie_mask` is `None`.

## §9. Fusion: least-squares assembly of shared corners from cell increments

**Where this lives / what this part does.** `HexFusion` in experiments/learned_intrinsic_solver/fusion.py:82 turns one 3x3 deformation-gradient increment per cell into a single consistent update of the shared corner positions. Its constructor builds the sparse operators and factorizes the free-corner normal matrix once per registered material context (experiments/learned_intrinsic_solver/mixed_physics.py:241). `_FusionSolve.forward` (experiments/learned_intrinsic_solver/fusion.py:41) runs once per object per query inside `MixedHexSolverStep.forward` (experiments/learned_intrinsic_solver/mixed_physics.py:474), and `_FusionSolve.backward` (experiments/learned_intrinsic_solver/fusion.py:64) once per object per training backward pass. The cached CPU LU factorization is `PardisoFactor` in experiments/learned_intrinsic_solver/pardiso.py:76, a ctypes bridge to oneMKL PARDISO. `fuse` with a zero increment is also used once per physical step by `prepare` (experiments/learned_intrinsic_solver/mixed_physics.py:526).

**The least-squares problem.** For every cell the solver step forms a target increment `D_c = R (A_target - A)` (3x3, world coordinates, material axes in columns) from the network's local target axes `A_target` (experiments/learned_intrinsic_solver/mixed_physics.py:463; §10); the network itself never emits a world-frame quantity. Eight cells share each interior corner, so eight increments compete for the same corner positions. Fusion resolves this by choosing the free-corner displacement `u` that best matches all increments at once. "Gauss quadrature" here is the standard eight-point rule for the cube (points at `+/- 1/sqrt(3)` along each axis, all weights equal); `dN_k/dX(q)` are the trilinear shape-function gradients at Gauss point `q`. The displacement gradient at `q` is `(grad u)_ij(q) = sum_k u_ki dN_k/dX_j(q)`. The objective is

    J(u) = sum_c w_c sum_q (omega_q / sum omega) || grad u_c(q) - D_c ||_F^2

with the fixed corners held at `fixed_positions - base_positions[fixed]`. The quadrature weights are normalized to sum to one, so the inner sum is an average over the eight points and `w_c` carries all the physical scaling. Since `J` is quadratic in `u`, the minimizer is a linear solve.

**Operators.** With `G` the sparse gradient operator (row `(c, q, j)` holds `dN_k/dX_j(q)` for the eight corners of `c`, shape `[24 C, P]`) and `W` the diagonal of row weights `w_c omega_q`, the normal matrix is `K = G^T W G` and the target operator is `T = G^T W Rep`, where `Rep` repeats one target row `(c, j)` over the eight Gauss points of its cell:

```python
        quadrature = hex_gauss_quadrature(rest.cell_size, dtype=self._numpy_dtype)
        gradients = np.asarray(quadrature.shape_gradients, dtype=self._numpy_dtype)
        quadrature_weights = np.asarray(quadrature.weights, dtype=self._numpy_dtype)
        quadrature_weights = quadrature_weights / quadrature_weights.sum(dtype=self._numpy_dtype)
        row_count = self.cell_count * 8 * 3
        rows = np.repeat(np.arange(row_count), 8)
        columns = np.broadcast_to(cells[:, None, None, :], (self.cell_count, 8, 3, 8)).reshape(-1)
        values = np.broadcast_to(gradients.transpose(0, 2, 1)[None], (self.cell_count, 8, 3, 8)).reshape(-1)
        gradient_operator = sparse.coo_matrix(
            (values, (rows, columns)), shape=(row_count, self.corner_count), dtype=self._numpy_dtype
        ).tocsr()
        row_weights = np.repeat((weights[:, None] * quadrature_weights[None]).reshape(-1), 3)
        stiffness = (gradient_operator.T @ gradient_operator.multiply(row_weights[:, None])).tocsc()
```
(`experiments/learned_intrinsic_solver/fusion.py:181-193`)

```python
        target_columns = np.broadcast_to(
            np.arange(self.cell_count * 3).reshape(self.cell_count, 1, 3), (self.cell_count, 8, 3)
        ).reshape(-1)
        weighted_repeat = sparse.coo_matrix(
            (row_weights, (np.arange(row_count), target_columns)),
            shape=(row_count, self.cell_count * 3),
            dtype=self._numpy_dtype,
        ).tocsr()
        target_operator = (gradient_operator.T @ weighted_repeat).tocsr()
        self._target_operator = target_operator[self._free].tocsr()
        self._fixed_coupling = stiffness[self._free][:, self._fixed].tocsr()
        self._free_stiffness = stiffness[self._free][:, self._free].tocsc()
        self._factor = None
        if len(self._free):
            try:
                self._factor = PardisoFactor(self._free_stiffness)
            except RuntimeError as error:
                raise ValueError(f"fusion factorization failed: {error}") from error
            if self._factor.dtype != self._numpy_dtype:
                raise RuntimeError("sparse factorization changed the requested working precision")
```
(`experiments/learned_intrinsic_solver/fusion.py:194-213`)

- `gradients` from `hex_gauss_quadrature` is `[8 q, 8 k, 3 j]`; `transpose(0, 2, 1)` makes it `[q, j, k]` so each row `(c, q, j)` receives its eight corner coefficients; `rows = repeat(arange(24 C), 8)` and `columns = cells[c, k]` place them.
- `row_weights` is `w_c * omega_q / sum(omega)` repeated three times for `j`; `stiffness` is `K` in CSC, `target_operator` is `T`. Both are assembled in the working NumPy dtype (float32 by default, float64 for reference checks).
- The free/fixed split is done by row and column slicing: `_target_operator = T[free]`, `_fixed_coupling = K[free][:, fixed]` (`K_fF`), `_free_stiffness = K[free][:, free]` (`K_ff`). The normal equations become `K_ff u_free = T_free d - K_fF u_fixed`, one right-hand side per world coordinate and batch member.
- `K` is symmetric positive semidefinite; its null space on a connected, fully integrated grid is the three translations, which one pin removes. That is why at least one fixed corner is mandatory (experiments/learned_intrinsic_solver/fusion.py:157-160) and why an unclamped translation gauge is not offered.
- `K_ff` is factorized exactly once here by `PardisoFactor`; the factor's dtype must equal the requested working precision or construction fails. When every corner is pinned (`len(self._free) == 0`) there is no factor and `_solve` returns zeros.
- The physical weights are chosen by the mixed step: `cell_weights = stiffness * h^3` with `stiffness = mu (3 - mu / (lambda + mu))` computed in the numerically stable form below. Units: Pa times m^3 gives joules, so `J(u)` is an energy-like quantity and rest volume enters exactly once (the quadrature weights are averages).

```python
            physical = HexImplicitEulerLoss(self._rest, lame_lambda, lame_mu, density, time_step=self.time_step)
            mass = physical.lumped_mass
            if not torch.isfinite(mass).all() or (mass <= 0).any():
                raise ValueError("all physical masses, including pins, must remain positive float32")
            lam, mu = physical.lame_lambda, physical.lame_mu
            scale = torch.maximum(lam, mu)
            stiffness = mu * (3 - (mu / scale) / (lam / scale + mu / scale))
            fusion = HexFusion(self._rest, self._fixed_cpu, cell_weights=stiffness * self.cell_size**3)
            predictor = RigidPosePredictor(mass, gravity=tuple(self._gravity_cpu.tolist()))
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:234-242`)

**Forward solve.** The autograd function packs targets and boundary displacements as multi-column right-hand sides, solves on CPU, and writes the answer back on the original device:

```python
    def forward(ctx, fusion, base_positions, world_axis_increments, fixed_positions):
        device = base_positions.device
        fixed, free = fusion._indices(device)
        increments = world_axis_increments.detach().cpu().contiguous().numpy()
        batch_count = base_positions.shape[0]
        target_rows = increments.swapaxes(-1, -2).reshape(batch_count, 3 * fusion.cell_count, 3)
        # The solve needs only targets and boundary displacement, not all base
        # coordinates. Keep the full position array on its original device.
        fixed_delta = (fixed_positions - base_positions[:, fixed]).detach().cpu().contiguous().numpy()
        rhs = fusion._target_operator @ _columns(target_rows) - fusion._fixed_coupling @ _columns(fixed_delta)
        free_delta = fusion._solve(rhs)
        delta = torch.from_numpy(np.ascontiguousarray(_batch(free_delta, batch_count))).to(device=device)
        result = base_positions.clone()
        result[:, free] += delta
        # Assign the supplied values directly rather than adding a cancellation.
        result[:, fixed] = fixed_positions
        ctx.fusion = fusion
        ctx.batch_count = batch_count
        ctx.device = device
        return result
```
(`experiments/learned_intrinsic_solver/fusion.py:41-60`)

- `world_axis_increments [B, C, 3, 3]` has material axes in columns, `D_c[i, j]`. The row `(c, j)` of the right-hand side for world coordinate `i` needs `D_c[i, j]`, hence `swapaxes(-1, -2)` before the reshape to `[B, 3 C, 3]`.
- `_columns` (experiments/learned_intrinsic_solver/fusion.py:29) turns a `[B, N, 3]` tensor into a Fortran-ordered `[N, 3 B]` matrix so all batch members and coordinates are one PARDISO call with `3 B` right-hand sides; `_batch` inverts it.
- Only the targets and `fixed_delta = fixed_positions - base[fixed]` cross to the CPU; the full position array stays on its device and receives `delta` on the free rows.
- Pins are exact by assignment: `result[:, fixed] = fixed_positions` overwrites rather than adding a cancelling correction, so prescribed rows equal the supplied values bit for bit. `_checked_forward` in the trainer relies on that with `torch.equal` (experiments/learned_intrinsic_solver/train_mixed.py:379).
- A zero increment with pins already satisfied is a no-op: both right-hand-side terms vanish, the solve returns zeros and `result` equals `base` exactly, including a warped, non-affine base. `prepare` uses `fuse(base, 0, fixed_positions)` on the rigidly moved base whose pins are then pulled back to their prescribed positions, which yields the minimum-weighted-gradient blend of that boundary correction into the free corners:

```python
        with torch.no_grad(), context.lock:
            rigid = context.predictor.predict(x, velocity, force, self.time_step)
            fixed_positions = x[self._fixed_cpu].clone()
            base = x[None] @ rigid.rigid_delta_rotation.transpose(-1, -2) + rigid.rigid_delta_translation[:, None]
            zero_increment = x.new_zeros((1, len(self._rest.cell_corner_indices), 3, 3))
            candidate = context.fusion.fuse(base, zero_increment, fixed_positions[None])[0]
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:521-526`)

**Adjoint (backward) solve.** The "adjoint" is the transpose of the linear map from inputs to outputs; for a solve with `K_ff` it is a solve with `K_ff^T`. Given the loss gradient `g` with respect to the fused positions, the backward pass computes `lambda = K_ff^{-T} g_free`, then the three input gradients:

```python
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient_output):
        fusion = ctx.fusion
        batch_count = ctx.batch_count
        fixed, free = fusion._indices(ctx.device)
        adjoint, gradient_targets = fusion._adjoint_targets(gradient_output[:, free])
        boundary_transfer = _batch(fusion._fixed_coupling.T @ adjoint, batch_count)
        boundary_transfer = torch.from_numpy(np.ascontiguousarray(boundary_transfer)).to(device=ctx.device)
        gradient_base = gradient_output.clone()
        gradient_base[:, fixed] = boundary_transfer
        gradient_fixed = gradient_output[:, fixed] - boundary_transfer
        return (
            None,
            gradient_base,
            torch.from_numpy(np.ascontiguousarray(gradient_targets)).to(device=ctx.device),
            gradient_fixed,
        )
```
(`experiments/learned_intrinsic_solver/fusion.py:62-79`)

```python
    def _solve(self, right_hand_side: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        if self._factor is None:
            return np.zeros_like(right_hand_side)
        return self._factor.solve(np.asfortranarray(right_hand_side), transpose=transpose)
...
        batch_count = free_gradient.shape[0]
        columns = _columns(free_gradient.detach().cpu().contiguous().numpy())
        adjoint = self._solve(columns, transpose=True)
        target_rows = _batch(self._target_operator.T @ adjoint, batch_count)
        return adjoint, target_rows.reshape(batch_count, self.cell_count, 3, 3).swapaxes(-1, -2)
```
(`experiments/learned_intrinsic_solver/fusion.py:225-228, 253-257`)

- Targets: `dL/dD = unpack(T_free^T lambda)`, reshaped to `[B, C, 3, 3]` with a final `swapaxes` that undoes the forward packing.
- Fixed positions: `dL/dx_fixed = g_fixed - K_fF^T lambda`; the first term is the direct assignment, the second comes through the right-hand side.
- Base positions: free rows receive `g_free` unchanged (`result_free = base_free + delta`), fixed rows receive `+ K_fF^T lambda` (the base enters `fixed_delta` with a minus sign), and `g_fixed` does not reach the base because the fixed rows of `result` were overwritten.
- `once_differentiable` marks the backward as first-order only; the factor, topology and weights are never differentiated. `ctx.fusion` keeps the `HexFusion` alive, so a context discarded from the registry can still run its adjoint while a graph references it (experiments/learned_intrinsic_solver/mixed_physics.py:253-258).
- `project_gradient` (experiments/learned_intrinsic_solver/fusion.py:310) exposes the same `_adjoint_targets` path as a detached operator; §10 uses it to build the gradient feature.

**The PARDISO factor and the transpose flag.** `PardisoFactor` copies the CSR matrix, calls phase 12 (analysis plus numerical factorization) once in the constructor, and thereafter only phase 33 (solve) per call, releasing with phase -1 on `close`. The relevant control parameters and the solve body:

```python
        self._iparm[0] = 1  # Preserve pardisoinit's matrix-type defaults.
        self._iparm[11] = 0  # A, not A.T, for the first solve.
        self._iparm[26] = 1  # Validate the CSR structure in the analysis phase.
        self._iparm[27] = 1 if self.dtype == np.float32 else 0
        self._iparm[34] = 1  # SciPy CSR is zero-based.
...
            vector = values.ndim == 1
            packed = np.asfortranarray(values[:, None] if vector else values)
            result = np.empty_like(packed, order="F")
            self._iparm[11] = 2 if transpose else 0
            self._call(33, packed, result)
            return result[:, 0].copy() if vector else result
```
(`experiments/learned_intrinsic_solver/pardiso.py:162-166, 235-240`)

- Matrix type 11 means real nonsymmetric: the full `K_ff` is factorized without exploiting symmetry, so `iparm[11] = 2` (solve with `A^T`) gives the exact adjoint of the forward solve even when float32 assembly rounding leaves `K_ff` slightly asymmetric. `iparm[27] = 1` keeps the factor and every solve in single precision for float32 matrices.
- Right-hand sides are a vector `[N]` or a matrix `[N, R]` in the factor's dtype, in any memory order (other dtypes or shapes, and non-finite values, raise); `solve` packs them Fortran-ordered itself and returns a new array of the same shape, Fortran-ordered in the matrix case.
- Thread count defaults to `MKL_NUM_THREADS` or 30 and is set per call with `MKL_Set_Num_Threads_Local` and restored afterwards; a lock serializes calls on one handle, and a PID check refuses reuse after a fork.

**Per-object factors in the mixed step.** Because `w_c` depends on the material, every registered context owns its own `HexFusion`, and the mixed forward pass fuses objects one at a time on one-element slices before concatenating:

```python
        fused = torch.cat(
            [
                context.fusion.fuse(positions[i : i + 1], world_increment[i : i + 1], fixed_positions[i : i + 1])
                for i, context in enumerate(contexts)
            ]
        )
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:474-479`)

Each slice is its own autograd node with its own synchronous GPU to CPU to GPU round trip; the batch loop is serial. The CPU bridge does not support CUDA graph capture.

## §10. What one query differentiates and what it freezes

**Where this lives / what this part does.** `assemble_inputs` and `objective_gradient` in experiments/learned_intrinsic_solver/input_assembly.py:153-284 compose the network inputs; `MixedHexSolverStep._prepare_inputs` (experiments/learned_intrinsic_solver/mixed_physics.py:337) supplies the per-batch energy and fusion-adjoint callables, and `MixedHexSolverStep.forward` (experiments/learned_intrinsic_solver/mixed_physics.py:427) runs the network, fusion and energy. The trainer loop in `run_training` (experiments/learned_intrinsic_solver/train_mixed.py:672-783) collates payloads with `_batch` (experiments/learned_intrinsic_solver/train_mixed.py:337), evaluates `local_objective` (experiments/learned_intrinsic_solver/train_mixed.py:227) and takes one Adam step per mixed batch; `store_history` in experiments/learned_intrinsic_solver/history.py:99 writes the detached history back. All of this runs once per query, which in training is once per optimizer update.

**The differentiable chain.** In training the only trainable leaves are the network parameters. Every position tensor entering a query is a detached leaf, so the parameter gradient flows through exactly this path: network output, local target axes, world increment `R (A_target - A)`, fusion, fused positions, energy, objective. Written out for one object:

- `A = R^T F(x)` and the two geometric blocks `R^T (F_Y - F)`, `R^T (F - F_prev)` are differentiable in the positions but, with detached positions, constants for the parameters. They are the network's view of the geometry.
- The network returns `A_target = A + step_size * correction` (experiments/learned_intrinsic_solver/network.py:355), with `step_size` in `(0, max_step_size)` per cell.
- `world_increment = R @ (A_target - A) = R @ (step_size * correction)`, shape `[B, C, 3, 3]`; `R` is a constant here.
- `fused = fuse(x, world_increment, fixed_positions)` per object (§9), whose backward is the transpose PARDISO solve.
- `energy = elastic + inertia + damping` at `fused` with `Y` and `X_start` fixed (experiments/learned_intrinsic_solver/mixed_physics.py:389-406): stable Neo-Hookean density at eight Gauss points, `0.5 sum_p m_p |x_p - Y_p|^2 / dt^2` with the lumped mass `m_p` (each cell's mass split equally over its eight corners, experiments/learned_intrinsic_solver/hex_energy.py:316-318), and the viscous term when the context has damping.
- `local_objective = asinh(after / scale) + w relu((after - before) / scale)`, `scale = max(|before|, floor)` detached, averaged over the batch.

**The gradient feature, frozen by construction.** The network is also told how the energy currently changes with respect to its own output. That feature is computed on a detached copy so it can never be differentiated:

```python
    with torch.enable_grad():
        candidate = positions.detach().requires_grad_(True)
        total = energy_total(candidate, inertial_prediction.detach(), previous_positions.detach())
        gradient = torch.autograd.grad(total.sum(), candidate)[0]
    gradient = gradient.detach()
    gradient[:, fixed_indices] = 0
    return gradient
```
(`experiments/learned_intrinsic_solver/input_assembly.py:176-182`)

`torch.enable_grad()` makes this work even when the caller runs the whole query under `no_grad`; the result is the position gradient of the full objective in newtons, `[B, P, 3]`, with prescribed rows zeroed. It is then pushed through each object's fusion adjoint (`project_gradient`, a NumPy transpose solve, so autograd history ends there regardless), giving the world axis gradient `G` in joules, `[B, C, 3, 3]`: the derivative of the fused energy with respect to a world axis increment at zero increment. The rest of the assembly normalizes and packs:

```python
    target_deformation = center_deformation(inertial_prediction, cells, gradients)
    previous_deformation = center_deformation(previous_positions, cells, gradients)
    position_gradient = objective_gradient(
        positions, inertial_prediction, previous_positions, energy_total, step.fixed_indices
    )
    world_gradient = project_gradient(position_gradient)
    current_gradient, gradient_rms = rms_normalize(to_local(frames, world_gradient))
    if history is None:
        previous_gradient = torch.zeros_like(current_gradient)
        previous_update = torch.zeros_like(current_gradient)
        history_valid = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    else:
        previous_gradient, _ = rms_normalize(to_local(frames, history.axis_gradient_world), rms=gradient_rms)
        previous_update, _ = rms_normalize(to_local(frames, history.axis_update_world))
        history_valid = history.valid
    state = pack_state_features(
        inertial_axis_offset=to_local(frames, target_deformation - deformation),
        physical_axis_change=to_local(frames, deformation - previous_deformation),
        current_axis_gradient=current_gradient,
        previous_axis_gradient=previous_gradient,
        previous_axis_update=previous_update,
        boundary_features=step.boundary_features,
        log_gradient_rms=gradient_rms.log(),
        history_valid=history_valid,
    )
```
(`experiments/learned_intrinsic_solver/input_assembly.py:243-267`)

- "RMS" is the root mean square over all cells and components of one object, `sqrt(mean_{c,i,j} v^2)`, floored at `RMS_FLOOR = 1e-12`; the normalized blocks are clipped to `+/- 10`. The current and previous gradient share the current RMS; the previous achieved update uses its own. `log_gradient_rms` is a per-object scalar input.
- History is detached in three places: `check_history` (experiments/learned_intrinsic_solver/input_assembly.py:150) before the query, `batch_history` (experiments/learned_intrinsic_solver/history.py:64) at collation and `store_history` (experiments/learned_intrinsic_solver/history.py:119-120) after the query. Objects with `valid = False` get zero history blocks and `history_valid = 0`.
- Edge descriptors (experiments/learned_intrinsic_solver/input_assembly.py:268-274) detach the frames again inside `build_edge_features` (experiments/learned_intrinsic_solver/network_geometry.py:162) but keep the current centers and local axes differentiable.

The mixed step provides the two callables and the conditioning, all material constants coming from the registered context rather than from parameters:

```python
        def energy_total(candidate: Tensor, target: Tensor, previous: Tensor) -> Tensor:
            return self._energy(candidate, target, contexts, previous).total

        def project_gradient(position_gradient: Tensor) -> Tensor:
            return torch.cat(
                [context.fusion.project_gradient(position_gradient[i : i + 1]) for i, context in enumerate(contexts)]
            )

        material = torch.stack([context.material for context in contexts]).to(positions.device)
        channels = conditioning_channels(*material.unbind(-1), self.cell_size, self.time_step)
        conditioning = channels[:, None].expand(-1, len(self.cell_corner_indices), -1)
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:342-352`)

**The forward pass.** The step evaluates the network once for the batch, forms the world increment, fuses per object, evaluates the energy on the fused positions and records detached diagnostics:

```python
        inputs = self._prepare_inputs(positions, inertial_prediction, contexts, previous_positions, history)
        prediction = self.network(inputs.local_axes, inputs.state_features, inputs.edge_features, inputs.conditioning)
        world_increment = inputs.frames @ (prediction.local_target_axes - inputs.local_axes)
        if fixed_positions is None:
            fixed_positions = self.rest_positions[self.fixed_indices][None].expand(len(contexts), -1, -1)
        if (
            not isinstance(fixed_positions, Tensor)
            or fixed_positions.shape != (len(contexts), len(self.fixed_indices), 3)
            or fixed_positions.dtype != positions.dtype
            or fixed_positions.device != positions.device
            or not torch.isfinite(fixed_positions).all()
        ):
            raise ValueError("fixed_positions must be finite [B,F,3] on the input dtype/device")
        fused = torch.cat(
            [
                context.fusion.fuse(positions[i : i + 1], world_increment[i : i + 1], fixed_positions[i : i + 1])
                for i, context in enumerate(contexts)
            ]
        )
        loss = self._energy(fused, inertial_prediction, contexts, previous_positions)
        with torch.no_grad():
            achieved = center_deformation(
                fused.detach(), self.cell_corner_indices, self.center_gradients
            ) - center_deformation(positions.detach(), self.cell_corner_indices, self.center_gradients)
            residual = torch.linalg.vector_norm(inputs.position_gradient.flatten(1), dim=1)
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:461-485`)

`achieved` (the world change of the center deformation produced by this update) and `residual` (the norm of the free-corner position gradient at the pre-update candidate) are computed under `no_grad` from detached tensors; they become the next query's history and the reported force residual. `fixed_positions` defaults to the rest positions of the pins.

**Trainer detach points.** The trainer never keeps a graph across queries. Payload tensors are detached when they are stacked into a batch:

```python
    payloads = [getattr(record, "payload", record) for record in records]
    values = {
        name: torch.stack([p[name].detach().to(device) for p in payloads])
        for name in ("candidate", "inertial_prediction", "fixed_positions", "physical_positions")
    }
    values["context_ids"] = tuple(p["context_id"] for p in payloads)
```
(`experiments/learned_intrinsic_solver/train_mixed.py:348-353`)

The energy before the update and the energy floor are evaluated under `no_grad`:

```python
                    records = pool.take_batch()
                    batch = _batch(records, device, cell_count=cell_count)
                    with torch.no_grad():
                        previous = step.energy(
                            batch["candidate"],
                            batch["inertial_prediction"],
                            batch["context_ids"],
                            previous_positions=batch["physical_positions"],
                        ).total
                        floor = step.energy_floor(batch["context_ids"])
                    if not torch.isfinite(previous).all():
                        raise ValueError("nonfinite input energy")
```
(`experiments/learned_intrinsic_solver/train_mixed.py:676-687`)

One forward, one backward and one Adam step per mixed batch, with autocast disabled explicitly (TF32 is also switched off at experiments/learned_intrinsic_solver/train_mixed.py:461-463):

```python
                optimizer.zero_grad(set_to_none=True)
                began = time.perf_counter()
                error = None
                try:
                    with torch.autocast(device_type=device.type, enabled=False):
                        result = _checked_forward(module, step, batch)
                        losses = local_objective(
                            result.loss.total, previous, floor, increase_weight=config.energy_increase_weight
                        )
                        loss = losses.mean()
...
                began = time.perf_counter()
                loss.backward()
                gradient_error = (
                    None
                    if all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
                    else "nonfinite gradient"
                )
                failures = _all_ranks_ok(gradient_error, device, world_size)
                if failures:
                    raise RuntimeError(f"backward failed: {failures}")
                optimizer.step()
                timings["backward_and_adam_seconds"] += time.perf_counter() - began
                after = result.loss.total.detach()
                residual = result.force_residual_norm.detach()
                step_size = result.step_size.detach()
```
(`experiments/learned_intrinsic_solver/train_mixed.py:694-703, 710-724`)

```python
    before = before.detach()
    floor = torch.as_tensor(floor, dtype=before.dtype, device=before.device).detach()
    if not torch.isfinite(floor).all() or (floor <= 0).any():
        raise ValueError("energy floor must be finite and positive")
    scale = torch.maximum(before.abs(), floor)
    return torch.asinh(after / scale) + increase_weight * torch.relu((after - before) / scale)
```
(`experiments/learned_intrinsic_solver/train_mixed.py:243-248`)

After the step, the query's detached history is written into the payloads and the fused positions become the next candidate, detached, so the next inner iteration starts a fresh graph:

```python
                payloads = [record.payload for record in records]
                store_history(payloads, result)
                for i, record in enumerate(records):
...
                    record.payload["candidate"] = result.positions[i].detach()
                pool.finish_batch(records)
```
(`experiments/learned_intrinsic_solver/train_mixed.py:762-764, 777-778`)

- `module` is `step` itself, or `DistributedDataParallel(step, broadcast_buffers=False)` when `WORLD_SIZE > 1` (experiments/learned_intrinsic_solver/train_mixed.py:597-601); the per-rank exception exchange `_all_ranks_ok` keeps ranks in lockstep so a failed proposal on one rank aborts all of them.
- In later inner iterations `before` is the energy of the carried candidate, which equals the previous update's `after`, so `scale` tracks the trajectory rather than the initial guess.
- Every tensor in the working chain is float32; the only float64 excursions are the tie-cell SVD (§8) and the energy floor formula (experiments/learned_intrinsic_solver/mixed_physics.py:420-425), both cast back.

| quantity | differentiated? | where the boundary is |
|---|---|---|
| network parameters | yes, the only trainable leaves | `optimizer.step()` at experiments/learned_intrinsic_solver/train_mixed.py:720 |
| candidate positions `x`, `Y`, `X_start`, `fixed_positions` | no in training (detached leaves) | `p[name].detach()` at experiments/learned_intrinsic_solver/train_mixed.py:350 |
| center deformation `F`, local axes `A = R^T F` | differentiable in positions; constant for the parameters once positions are detached | experiments/learned_intrinsic_solver/features.py:191 and experiments/learned_intrinsic_solver/input_assembly.py:242 |
| frames `R`, tie mask, singular values | no | `torch.no_grad()` at experiments/learned_intrinsic_solver/frames.py:240; reference at experiments/learned_intrinsic_solver/frames.py:141 |
| inertial offset and physical change blocks | differentiable in positions, `Y`, `X_start` (all detached in training) | experiments/learned_intrinsic_solver/input_assembly.py:259-260 |
| position gradient, world axis gradient `G`, RMS, `log rms` | no | `autograd.grad` on a detached copy at experiments/learned_intrinsic_solver/input_assembly.py:177-180; detach-to-NumPy at experiments/learned_intrinsic_solver/fusion.py:254 and the NumPy transpose solve at experiments/learned_intrinsic_solver/fusion.py:255 |
| history blocks and flag | no | experiments/learned_intrinsic_solver/input_assembly.py:150, experiments/learned_intrinsic_solver/history.py:64, experiments/learned_intrinsic_solver/history.py:119-120 |
| edge descriptors | centers and axes yes; frames no | experiments/learned_intrinsic_solver/network_geometry.py:162 |
| conditioning channels | no (context constants) | experiments/learned_intrinsic_solver/mixed_physics.py:350-351 |
| network outputs (target axes, correction, step size) | yes | experiments/learned_intrinsic_solver/network.py:354-356 |
| world increment `R (A_target - A)` | yes through the network output; `R` constant | experiments/learned_intrinsic_solver/mixed_physics.py:463 |
| fused positions | yes, first order, in increment, base and pins | `_FusionSolve.backward` at experiments/learned_intrinsic_solver/fusion.py:64; the factor is never differentiated (experiments/learned_intrinsic_solver/fusion.py:209) |
| energy after the update | yes | experiments/learned_intrinsic_solver/mixed_physics.py:480 |
| energy before, energy floor, `scale` | no | experiments/learned_intrinsic_solver/train_mixed.py:678-685 and experiments/learned_intrinsic_solver/train_mixed.py:243-247 |
| achieved axis update, force residual norm | no | `torch.no_grad()` at experiments/learned_intrinsic_solver/mixed_physics.py:481 |
| candidate carried to the next query | no | experiments/learned_intrinsic_solver/train_mixed.py:777 |

## §11. The gradient input: from the energy gradient at the corners to normalized axis coordinates

**Where this lives / what this part does.** `objective_gradient` at experiments/learned_intrinsic_solver/input_assembly.py:153 and the gradient lines of `assemble_inputs` at experiments/learned_intrinsic_solver/input_assembly.py:245, the two per-batch closures `energy_total` and `project_gradient` built by `MixedHexSolverStep._prepare_inputs` at experiments/learned_intrinsic_solver/mixed_physics.py:337, `HexFusion.project_gradient` and `HexFusion._adjoint_targets` at experiments/learned_intrinsic_solver/fusion.py:310 and experiments/learned_intrinsic_solver/fusion.py:239, and the pure feature functions `to_local`, `rms_normalize` and `pack_state_features` at experiments/learned_intrinsic_solver/features.py:194, experiments/learned_intrinsic_solver/features.py:226 and experiments/learned_intrinsic_solver/features.py:261. This is the QUERY stage of the pipeline map: it runs once per learned query for every object in the batch, before the network is evaluated, and it fills state columns 18 to 44 and column 59 of the 61-value per-cell state vector. It also produces the `force_residual_norm` diagnostic that `forward` returns at experiments/learned_intrinsic_solver/mixed_physics.py:485.

The network is an optimizer, so it must see the descent direction of the physical objective in the same coordinates as its own output. Its output is a 3x3 **local axis increment** per cell (the change of the axes `A = R^T F`, see §3 and §8); the objective is a function of the shared corner positions. Getting from one to the other takes three coordinate changes, applied in this order: corners to per-cell axis increments (the fusion adjoint), world frame to cell frame (`R^T`), and joules to dimensionless network inputs (RMS normalization). The result is constant for the backward pass of the training loss: the network is told where downhill is, but the loss is not differentiated through that information.

**Step 1: the position gradient of the physical objective.** The objective of one physical step is `E(X) = elastic(X) + inertia(X; Y) + damping(X; X_start)` in joules (§4, §5, §6), where `X` are the candidate corner positions [B, P, 3] in meters, `Y` is the inertial prediction and `X_start` the positions at the beginning of the physical step. Its gradient `g = dE/dX` has units J/m = N. It is the implicit-Euler force-balance residual of §4, `g_v = m_v (X_v - Y_v) / dt^2 - f_v(X)` with `f = -dE_elastic/dX - dE_damping/dX` the internal forces (gravity enters through `Y`): it vanishes exactly when the internal forces balance the inertial term, so the code calls its norm the force residual (§7).

```python
def objective_gradient(
    positions: Tensor,
    inertial_prediction: Tensor,
    previous_positions: Tensor,
    energy_total: Callable[[Tensor, Tensor, Tensor], Tensor],
    fixed_indices: Tensor,
) -> Tensor:
...
    with torch.enable_grad():
        candidate = positions.detach().requires_grad_(True)
        total = energy_total(candidate, inertial_prediction.detach(), previous_positions.detach())
        gradient = torch.autograd.grad(total.sum(), candidate)[0]
    gradient = gradient.detach()
    gradient[:, fixed_indices] = 0
    return gradient
```
(`experiments/learned_intrinsic_solver/input_assembly.py:153-159, 176-182`)

Implementation points:

- `torch.enable_grad()` makes the autograd call work even when the caller runs the whole query under `torch.no_grad()` (the validator at experiments/learned_intrinsic_solver/mixed_validation.py:260 and the rollout scripts do).
- `positions.detach().requires_grad_(True)` starts a fresh autograd graph rooted at the candidate. The returned gradient therefore has no connection to the graph that the training loss will differentiate later; this is the detach boundary of the gradient feature.
- `Y` and `X_start` are detached too: the gradient is the partial derivative with respect to the candidate only, with the inertial target and the damping anchor held fixed. Both are constants of the physical step anyway.
- `total.sum()` is differentiated once for the whole batch. Each object's energy depends only on its own positions, so the gradient of the sum equals the per-object gradients; this replaces B separate autograd calls with one.
- The rows of prescribed corners are zeroed in place. Pinned corners never move, so the force on them carries no information for the optimizer, and the fusion adjoint reads only free rows anyway (experiments/learned_intrinsic_solver/fusion.py:349).
- The returned tensor is float32 on the module device and is stored unchanged as `LearnedHexInputs.position_gradient` (experiments/learned_intrinsic_solver/input_assembly.py:282).

The `energy_total` callable is the mixed step's own `_energy` for the batch's material contexts, so a soft and a stiff object in one batch each get their own Lame parameters, lumped mass and damping coefficient (experiments/learned_intrinsic_solver/mixed_physics.py:389). A **lumped mass** is the per-corner mass obtained by distributing each cell's mass equally to its eight corners; it makes the inertia term a diagonal quadratic form.

```python
    def _prepare_inputs(
        self, positions: Tensor, inertial_prediction: Tensor, contexts, previous_positions: Tensor, history
    ) -> LearnedHexInputs:
        """Compose the shared schema-3 assembly with this batch's per-context energy and fusion adjoints."""

        def energy_total(candidate: Tensor, target: Tensor, previous: Tensor) -> Tensor:
            return self._energy(candidate, target, contexts, previous).total

        def project_gradient(position_gradient: Tensor) -> Tensor:
            return torch.cat(
                [context.fusion.project_gradient(position_gradient[i : i + 1]) for i, context in enumerate(contexts)]
            )
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:337-348`)

**Step 2: from corner forces to per-cell axis-increment gradients (the fusion adjoint).** Section §9 describes `HexFusion.fuse`: given a world axis increment `D` [B, C, 3, 3] per cell, it finds the one displacement per shared corner that fits `D` in a weighted least-squares sense with the prescribed corners held exactly. Writing the packed increment as a matrix with one row per (cell, material axis) and one column per world coordinate, the free-corner displacement is `delta_free = K_ff^-1 (B D_packed - K_fc delta_fixed)`, where `K = G^T W G` is the weighted normal matrix of the fit (`G` the Gauss-point shape gradients in 1/m, `W` the cell weights `stiffness * h^3` times the quadrature averages, so `K` has units J/m^2), `B = G^T W` (with each cell's target repeated at its eight Gauss points) is the target operator that maps increments to right-hand sides (units J/m), and the subscripts `ff`, `fc` select free and fixed corner rows and columns. This map from `D` to positions is affine. An **adjoint** of a linear map is its transpose: if `X = L D`, the chain rule gives `dE/dD = L^T dE/dX`. Here `L = K_ff^-1 B` on the free rows, so `dE/dD = B^T K_ff^-T g_free`, and the derivative has units (m per unit increment) times (J/m) = J. This is exactly what `_adjoint_targets` computes with one transposed sparse solve on the cached CPU factorization of that object's material context:

```python
    def _adjoint_targets(self, free_gradient: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Pull free-corner cotangents back to axis increments through the transpose solve.

        Args:
            free_gradient: Free-corner position cotangents, shape [B, F, 3], in
                ``free_indices`` order, on any device. Autograd history is
                discarded.

        Returns:
            The packed adjoint columns ``K_ff^{-T} g_free`` with shape [F, 3B]
            and the axis-increment gradients ``unpack(B^T K_ff^{-T} g_free)``
            with shape [B, C, 3, 3] (axes in columns), both as CPU NumPy arrays
            in the working precision.
        """
        batch_count = free_gradient.shape[0]
        columns = _columns(free_gradient.detach().cpu().contiguous().numpy())
        adjoint = self._solve(columns, transpose=True)
        target_rows = _batch(self._target_operator.T @ adjoint, batch_count)
        return adjoint, target_rows.reshape(batch_count, self.cell_count, 3, 3).swapaxes(-1, -2)
```
(`experiments/learned_intrinsic_solver/fusion.py:239-257`)

The same helper serves the autograd backward of `fuse` (experiments/learned_intrinsic_solver/fusion.py:68), so the gradient input the network sees and the gradient the training loss propagates through the fusion are produced by the same code path. `project_gradient` is the public, detached wrapper:

```python
    def project_gradient(self, position_gradient: torch.Tensor) -> torch.Tensor:
        """Return axis-increment gradients ``unpack(B^T K_ff^{-T} g_free)``.

        This is the adjoint of the increment-to-position map behind ``fuse``
        for a frozen base and frozen prescribed positions: for every increment
        ``D``, ``<project_gradient(g), D>`` equals
        ``<g_free, fuse(base, D, fixed) - fuse(base, 0, fixed)>``. It runs the
        cached factor's transpose solve through the same code path as the
        autograd backward of ``fuse``, so it matches
        ``autograd.grad(<g, fuse(base, D, fixed)>, D)`` for any ``D``. No extra
        stiffness or volume factor is applied.
...
        _, free = self._indices(position_gradient.device)
        _, gradient_targets = self._adjoint_targets(position_gradient[:, free])
        return torch.from_numpy(np.ascontiguousarray(gradient_targets)).to(device=position_gradient.device)
```
(`experiments/learned_intrinsic_solver/fusion.py:310-320, 348-350`)

Two consequences follow from the affine structure and are worth stating plainly.

- **It is the derivative with respect to the unscaled local axis increment, before the step multiplier.** The forward pass builds the fused input as `D_world = R (A_target - A)` (experiments/learned_intrinsic_solver/mixed_physics.py:463), and the network forms `A_target = A + s * correction` with the per-cell step size `s` (experiments/learned_intrinsic_solver/network.py:355). Call the raw local increment `Delta = A_target - A`. Because `dot(G_world, R Delta) = dot(R^T G_world, Delta)`, the local block `R^T G_world` that Step 3 produces, before the RMS normalization and clipping of Step 4, is `dE/dDelta`: the first-order change of the physical objective per unit change of a target axis component, in joules, in the cell's own frame. It is not divided by the step size and no other factor is applied to it; it describes the objective as a function of the quantity the network actually outputs. What the network receives in columns 18 to 26 is this block divided by its per-object RMS and clipped to +/-10 (Step 4). Because the increment-to-position map is affine, its Jacobian `L = K_ff^-1 B` and therefore the adjoint `B^T K_ff^-T` do not depend on the increment, so no linearization of the fusion is involved: the block is the exact gradient of `E(fuse(X, R Delta, fixed))` with respect to `Delta` at `Delta = 0`, that is, at the candidate. It is still a local, first-order quantity of the nonlinear objective `E`: at another increment the block would differ, because `g = dE/dX` would then be evaluated at the fused positions rather than at the candidate (experiments/learned_intrinsic_solver/input_assembly.py:176-179 evaluates it only there). The `project_gradient` docstring identity `dot(project_gradient(g), D) = dot(g_free, fuse(base, D, fixed) - fuse(base, 0, fixed))` for every `D` (experiments/learned_intrinsic_solver/fusion.py:313-319) holds for a fixed `g`; it does not say that `dE/dD` is constant in `D`.
- **No extra stiffness or volume factor is applied.** The cell weights `stiffness * h^3` already live inside `K` and `B` (experiments/learned_intrinsic_solver/mixed_physics.py:241), and the transposed solve applies their inverse. The identity `dot(project_gradient(g), D) = dot(g_free, fuse(base, D, fixed) - fuse(base, 0, fixed))` holds only for the bare `B^T K_ff^-T g_free`; any additional factor would make the block a different quantity than the derivative of the objective with respect to the network's output. The material scale still reaches the network, but through a separate channel: the RMS normalization below removes it from the block and reports its logarithm as one scalar.

In the mixed step each object has its own `HexFusion` with its own weights and factor, so `project_gradient` in experiments/learned_intrinsic_solver/mixed_physics.py:345 slices one object at a time and concatenates. The per-object shapes through this stage are:

| Stage | Tensor | Shape |
|---|---|---|
| `objective_gradient` output, whole batch | position gradient `g` [N] | `[B, P, 3]` |
| `project_gradient` closure, one object | `position_gradient[i : i + 1]` | `[1, P, 3]` |
| `_columns(free_gradient)` after selecting free rows | right-hand sides, one column per world coordinate | `[F, 3]` |
| `_solve(..., transpose=True)` | adjoint `K_ff^-T g_free` | `[F, 3]` |
| `_target_operator.T @ adjoint` | packed axis-increment gradient, rows (cell, axis) | `[3C, 3]` |
| `reshape(1, C, 3, 3).swapaxes(-1, -2)` | world axis gradient, axes in columns | `[1, C, 3, 3]` |
| `torch.cat` over objects | `world_gradient`, units J | `[B, C, 3, 3]` |
| `to_local`, `rms_normalize`, `flatten(-2)` | state columns 18 to 26 | `[B, C, 9]` |

Here `P` is the number of shared corners, `F` the number of free corners, `C` the number of cells. Every array on the CPU side is NumPy in the fusion's working precision (float32 in production), and the result is moved back to the input device with `torch.from_numpy(...).to(device=...)`; it never carries autograd history.

**Step 3: world to cell frame.** `to_local(frames, M)` returns `R^T @ M` for every cell (experiments/learned_intrinsic_solver/features.py:223). `R` is the cell's **proper rotation frame**: the closest rotation with determinant +1 to the center deformation gradient `F`, with world directions in its columns (§8). Axis differences, axis gradients and achieved axis updates all transform the same way under a change of frame, so one function serves all matrix blocks. The frames are detached before use, so this multiplication does not add anything to the autograd graph of the gradient blocks.

**Step 4: RMS normalization.** The **RMS** (root mean square) of a field is the square root of the mean of its squared components. `rms_normalize` takes it per object over every cell and every matrix component, floors it, divides, and clips:

```python
    _require_cell_matrices("values", values)
    batch_count = values.shape[0]
    if rms is None:
        scale = values.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    else:
        scale = _per_object("rms", rms, batch_count).to(dtype=values.dtype, device=values.device)
        scale = scale.reshape(batch_count, 1, 1, 1)
    scale = scale.clamp_min(RMS_FLOOR)
    return (values / scale).clamp(-CLIP, CLIP), scale
```
(`experiments/learned_intrinsic_solver/features.py:250-258`)

- The mean runs over dims `(1, 2, 3)`, that is over all `C * 9` components of one object, with `keepdim=True`, so the scale has shape `[B, 1, 1, 1]` and broadcasts.
- `RMS_FLOOR = 1e-12` (experiments/learned_intrinsic_solver/features.py:94) makes a measured zero finite: zero input gives zero output and an RMS equal to the floor, so `log(rms) = -27.6` instead of `-inf`. This happens for a fully prescribed grid and for a candidate that sits exactly at the minimum.
- `CLIP = 10.0` (experiments/learned_intrinsic_solver/features.py:97) bounds outliers after division. A component more than ten RMS from zero saturates, so the block carries direction and relative magnitude but not extreme ratios; the log-RMS scalar and the unclipped world tensor kept for the history are unaffected.
- The optional `rms` argument reuses another field's statistic; it is floored again and returned so the caller can log it.
- Nothing is detached here. The inputs to this function are already detached, which is what makes the outputs constants.

Which RMS each block uses (experiments/learned_intrinsic_solver/input_assembly.py:249, experiments/learned_intrinsic_solver/input_assembly.py:255 and experiments/learned_intrinsic_solver/input_assembly.py:256):

| Block | Input | Divided by | Why |
|---|---|---|---|
| current axis gradient (columns 18 to 26) | `R^T G` of this query | its own RMS `gradient_rms` | dimensionless direction and relative magnitude across cells |
| previous axis gradient (columns 27 to 35) | `R^T G_prev` from the history, re-expressed in the current frame | the current `gradient_rms` | the network can compare the two gradients; a shrinking gradient shows up as a previous block that is larger than the current one |
| previous achieved update (columns 36 to 44) | `R^T U_prev` from the history | its own RMS | a deformation change is dimensionless and unrelated in scale to a gradient in joules |
| log gradient RMS (column 59) | `gradient_rms.log()` | not normalized | the one scalar that tells the network how large the current gradient is in joules |

```python
    position_gradient = objective_gradient(
        positions, inertial_prediction, previous_positions, energy_total, step.fixed_indices
    )
    world_gradient = project_gradient(position_gradient)
    current_gradient, gradient_rms = rms_normalize(to_local(frames, world_gradient))
    if history is None:
        previous_gradient = torch.zeros_like(current_gradient)
        previous_update = torch.zeros_like(current_gradient)
        history_valid = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    else:
        previous_gradient, _ = rms_normalize(to_local(frames, history.axis_gradient_world), rms=gradient_rms)
        previous_update, _ = rms_normalize(to_local(frames, history.axis_update_world))
        history_valid = history.valid
    state = pack_state_features(
        inertial_axis_offset=to_local(frames, target_deformation - deformation),
        physical_axis_change=to_local(frames, deformation - previous_deformation),
        current_axis_gradient=current_gradient,
        previous_axis_gradient=previous_gradient,
        previous_axis_update=previous_update,
        boundary_features=step.boundary_features,
        log_gradient_rms=gradient_rms.log(),
        history_valid=history_valid,
    )
```
(`experiments/learned_intrinsic_solver/input_assembly.py:245-267`)

When `history` is `None` the two history blocks are zeros and `history_valid` is a `[B]` tensor of `False`; §12 explains how the history gets there.

**Step 5: packing.** `pack_state_features` validates every block, expands the shared boundary flags to the batch, zeroes the two history blocks of objects whose flag is false, and concatenates row-major flattened matrices with the scalars:

```python
    mask = valid[:, None, None, None]
    columns = []
    for name in MATRIX_BLOCKS:
        block = blocks[name]
        if name in _HISTORY_BLOCKS:
            block = torch.where(mask, block, torch.zeros_like(block))
        columns.append(block.flatten(-2))
    columns.append(boundary)
    columns.append(log_rms[:, None, None].expand(-1, cell_count, 1))
    columns.append(valid.to(reference.dtype)[:, None, None].expand(-1, cell_count, 1))
    return torch.cat(columns, dim=-1)
```
(`experiments/learned_intrinsic_solver/features.py:335-345`)

| Columns | Content | Differentiable in the candidate? |
|---|---|---|
| 0 to 8 | `R^T (F_Y - F)`, inertial axis offset | yes (through `F`) |
| 9 to 17 | `R^T (F - F_prev)`, physical axis change | yes (through `F`) |
| 18 to 26 | `clip(R^T G / rms)`, current axis gradient | no |
| 27 to 35 | `clip(R^T G_prev / rms)`, previous axis gradient | no |
| 36 to 44 | `clip(R^T U_prev / rms_own)`, previous achieved update | no |
| 45 to 50 | six exposed-face flags | no |
| 51 to 58 | eight fixed-corner flags | no |
| 59 | `log rms` of the current gradient | no |
| 60 | history flag, exactly 1.0 or 0.0 | no |

The output is `[B, C, 61]` in the dtype of the matrix blocks (float32). `flatten(-2)` on a `[B, C, 3, 3]` block writes the row-major order `M[0,0], M[0,1], M[0,2], M[1,0], ...`, so column `18 + 3 i + j` holds component `(i, j)` of the normalized local gradient. The `torch.where` on the history blocks is applied even when the caller already passed zeros, so an invalid flag is authoritative regardless of the tensors.

**The force residual output.** `forward` also returns the Euclidean norm of the zero-pinned position gradient, computed under `torch.no_grad()` at the pre-update candidate:

```python
        loss = self._energy(fused, inertial_prediction, contexts, previous_positions)
        with torch.no_grad():
            achieved = center_deformation(
                fused.detach(), self.cell_corner_indices, self.center_gradients
            ) - center_deformation(positions.detach(), self.cell_corner_indices, self.center_gradients)
            residual = torch.linalg.vector_norm(inputs.position_gradient.flatten(1), dim=1)
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:480-485`)

`force_residual_norm` has shape `[B]` and units N. It is the residual of the candidate the network was asked to improve, not of the fused result. Training reports its batch mean as `mean_force_residual_n` (experiments/learned_intrinsic_solver/train_mixed.py:748). The validator computes the same formula from a fresh autograd call, with the same zeroed pinned rows and the same damping anchor, but evaluates it at `batch["candidate"]` and accumulates the norm in float64 (experiments/learned_intrinsic_solver/mixed_validation.py:77-89): the first record is at the initial candidate (experiments/learned_intrinsic_solver/mixed_validation.py:180), and every later record is at the fused output of the preceding query, because `batch["candidate"]` is replaced by `result.positions` before recording (experiments/learned_intrinsic_solver/mixed_validation.py:184-186). The checkpoint-selection metric is the mean over samples of that value at the fused output of the last optimization query (experiments/learned_intrinsic_solver/mixed_validation.py:356-365), a residual that no `forward` call reports.

## §12. Optimizer history: what is stored, how it is carried across physical steps, when it is cleared

**Where this lives / what this part does.** The interface types are `OptimizerHistory` at experiments/learned_intrinsic_solver/input_assembly.py:92, `check_history` at experiments/learned_intrinsic_solver/input_assembly.py:113 and `LearnedHexStepOutput` at experiments/learned_intrinsic_solver/input_assembly.py:68. The payload plumbing is the history module: `empty_history` at experiments/learned_intrinsic_solver/history.py:32, `batch_history` at experiments/learned_intrinsic_solver/history.py:68, `store_history` at experiments/learned_intrinsic_solver/history.py:99 and `carry_history` at experiments/learned_intrinsic_solver/history.py:124. The trainer calls them from `_TrajectoryFactory.reset` and `_TrajectoryFactory.advance` (experiments/learned_intrinsic_solver/train_mixed.py:263 and experiments/learned_intrinsic_solver/train_mixed.py:291), from `_batch` (experiments/learned_intrinsic_solver/train_mixed.py:337) and from the update loop (experiments/learned_intrinsic_solver/train_mixed.py:763); the validator from `_store_history` (experiments/learned_intrinsic_solver/mixed_validation.py:152) inside its two loops. The values themselves are produced in `MixedHexSolverStep.forward` at experiments/learned_intrinsic_solver/mixed_physics.py:481. Cadence: `store_history` once per query; `batch_history` whenever a batch's history is collated (in `_batch` for every collated batch, and in the validator's `_store_history` after every query, which rebuilds `batch["history"]`); `check_history` once per query inside `prepare_inputs` or `forward`; `carry_history` once per physical-step advance inside `_TrajectoryFactory.advance` (on a CPU pool worker thread in training, where `finish_batch` submits the advance to the executor; on the calling thread in validation, where `factory.advance` is called directly); `empty_history` once per trajectory reset.

The history gives the network a one-step memory: the gradient it saw at the previous candidate and what its previous update actually did. Together with the current gradient this is the information a momentum or quasi-Newton style optimizer would use. Two blocks and a flag per object make up the whole history.

```python
class OptimizerHistory(NamedTuple):
    """Detached optimizer history from the previous learned query.

    World matrix coordinates keep the history independent of the frames, which
    are recomputed for every candidate; the step expresses both blocks in the
    current frame when it consumes them. Objects whose flag is False receive
    zero history blocks and ``history_valid = 0`` regardless of the tensors.

    Attributes:
        axis_gradient_world: Previous query's world axis gradient feature [J],
            shape [B, C, 3, 3].
        axis_update_world: Previous achieved world change of the center
            deformation (fused minus pre-update candidate), shape [B, C, 3, 3].
        valid: One boolean per object, shape [B].
    """

    axis_gradient_world: Tensor
    axis_update_world: Tensor
    valid: Tensor
```
(`experiments/learned_intrinsic_solver/input_assembly.py:92-110`)

- `axis_gradient_world` is the un-normalized world axis gradient `G` of the previous query, in joules, exactly the tensor §11 computed before `to_local` and `rms_normalize`.
- `axis_update_world` is the **achieved** change of the center deformation gradient, `F_center(fused) - F_center(candidate)`, dimensionless, in world coordinates.
- `valid` is one boolean per object. Objects flagged false receive zero blocks and `history_valid = 0` in the state, whatever their tensors hold, so a missing history is always distinguishable from a measured zero.

Both blocks are stored in **world** coordinates on purpose. Frames are recomputed for every candidate, and a block stored in the previous query's local frame would be expressed in the wrong basis one query later. Storing world matrices and applying the current `R^T` at consumption time (experiments/learned_intrinsic_solver/input_assembly.py:255) keeps the history consistent with the current blocks at the cost of one 3x3 product per cell.

**What the step produces.** `forward` returns the fused positions together with the two history candidates and two diagnostics:

```python
        return LearnedHexStepOutput(
            fused,
            prediction.local_target_axes,
            prediction.axis_correction,
            prediction.step_size,
            inputs.frames,
            loss,
            axis_gradient_world=inputs.axis_gradient_world,
            achieved_axis_update_world=achieved,
            force_residual_norm=residual,
            tie_mask=inputs.tie_mask,
        )
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:486-497`)

- `axis_gradient_world` is passed through from the inputs; it is the gradient at the pre-update candidate.
- `achieved_axis_update_world` is computed at experiments/learned_intrinsic_solver/mixed_physics.py:482 under `torch.no_grad()` from detached tensors as the difference of two `center_deformation` calls (the same eight-corner formula as §8). It is the realized update, not the requested one: fusion fits the requested increments `R (A_target - A)` in a least-squares sense with the pins held, so the actual change of each cell's `F` differs from what the network asked for. Storing the realized value tells the next query what happened rather than what was intended.
- `force_residual_norm` is described in §11; `tie_mask` (`[B, C]` boolean) marks cells whose closest rotation was ambiguous; with a clamped-face reference these are exactly the cells whose frame came from the tie-break (§8). Neither is part of the history; the trainer sums `tie_mask` into a per-update report count (experiments/learned_intrinsic_solver/train_mixed.py:725).

**Validation at the consumer.** `check_history` runs inside `prepare_inputs` and `forward` (experiments/learned_intrinsic_solver/mixed_physics.py:293) and refuses anything that could silently corrupt an input block:

```python
    expected = (batch, cell_count, 3, 3)
    dtype_name = str(dtype).removeprefix("torch.")
    for name in ("axis_gradient_world", "axis_update_world"):
        value = getattr(history, name)
        if not isinstance(value, Tensor) or value.shape != expected:
            raise ValueError(f"history.{name} must have shape [B, C, 3, 3] matching positions")
        if value.dtype != dtype or value.device != device:
            raise ValueError(f"history.{name} must use {dtype_name} on the module device")
        if not torch.isfinite(value).all():
            raise ValueError(f"history.{name} must be finite")
    valid = history.valid
    if not isinstance(valid, Tensor) or valid.shape != (batch,) or valid.dtype != torch.bool:
        raise ValueError("history.valid must be a boolean tensor with one flag per object")
    return OptimizerHistory(history.axis_gradient_world.detach(), history.axis_update_world.detach(), valid.to(device))
```
(`experiments/learned_intrinsic_solver/input_assembly.py:137-150`)

It requires both blocks to be finite `[B, C, 3, 3]` tensors in the working dtype on the module device, and the flag to be a `[B]` boolean tensor. It returns detached tensors (they share storage with the inputs but carry no autograd history; nothing is copied), so even a caller that hands in tensors with autograd history cannot connect two queries.

**Payload plumbing.** A trajectory lives in a plain dictionary (the payload) of detached tensors plus a context id, produced by `MixedHexSolverStep.prepare` and `advance`. The history module adds three keys to it, listed in `HISTORY_KEYS` at experiments/learned_intrinsic_solver/history.py:28: `history_axis_gradient_world` `[C, 3, 3]`, `history_axis_update_world` `[C, 3, 3]` and `history_valid`.

```python
    if isinstance(cell_count, bool) or not isinstance(cell_count, int) or cell_count < 1:
        raise ValueError("cell_count must be a positive integer")
    return {
        "history_axis_gradient_world": torch.zeros(cell_count, 3, 3, dtype=torch.float32),
        "history_axis_update_world": torch.zeros(cell_count, 3, 3, dtype=torch.float32),
        "history_valid": False,
    }
```
(`experiments/learned_intrinsic_solver/history.py:43-49`)

`empty_history` produces zero float32 CPU blocks and a false flag. `store_history` writes one detached per-object slice of the step output into each payload and sets the flag:

```python
    payloads = list(payloads)
    gradient = getattr(result, "axis_gradient_world", None)
    update = getattr(result, "achieved_axis_update_world", None)
    if gradient is None or update is None:
        raise ValueError("result must carry axis_gradient_world and achieved_axis_update_world")
    if gradient.shape != update.shape or gradient.ndim != 4 or gradient.shape[0] != len(payloads):
        raise ValueError("history tensors must have shape [B, C, 3, 3] with one entry per payload")
    for index, payload in enumerate(payloads):
        payload["history_axis_gradient_world"] = gradient[index].detach()
        payload["history_axis_update_world"] = update[index].detach()
        payload["history_valid"] = True
```
(`experiments/learned_intrinsic_solver/history.py:111-121`)

The slices keep the device of the step output; the pool's `finish_batch` detaches the whole payload again and the next collation moves them wherever the batch lives. `batch_history` is the inverse operation: it stacks the payload entries of one batch into an `OptimizerHistory` on the query device, treating a missing or false entry as zero blocks with a false flag, and refusing a true flag without stored blocks:

```python
def _blocks(payload: Mapping, cell_count: int, device) -> tuple[torch.Tensor, torch.Tensor, bool]:
    import torch

    valid = bool(payload.get("history_valid", False))
    if not valid:
        zeros = torch.zeros(cell_count, 3, 3, dtype=torch.float32, device=device)
        return zeros, zeros, False
    blocks = []
    for name in HISTORY_KEYS[:2]:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != (cell_count, 3, 3):
            raise ValueError(f"{name} must be a [C, 3, 3] tensor when history_valid is true")
        blocks.append(value.detach().to(device=device, dtype=torch.float32))
    return blocks[0], blocks[1], True
...
    return OptimizerHistory(
        torch.stack(gradients), torch.stack(updates), torch.tensor(flags, dtype=torch.bool, device=device)
    )
```
(`experiments/learned_intrinsic_solver/history.py:52-65, 94-96`)

`carry_history` copies the three keys, unchanged and by reference, from the finished payload into the payload of the next physical step; a missing source key is left missing:

```python
def carry_history(source: Mapping, target: MutableMapping) -> None:
    """Copy optimizer history unchanged across a physical timestep boundary.

    A missing source entry is left missing, which ``batch_history`` treats as
    no history. Initializer motion of the new candidate never enters history.
    """
    for name in HISTORY_KEYS:
        if name in source:
            target[name] = source[name]
```
(`experiments/learned_intrinsic_solver/history.py:124-132`)

**Lifecycle in training.** The `_TrajectoryFactory` owns the two boundaries where history is created or carried. `reset` prepares a new trajectory from the seeded augmenter and attaches an empty history; `advance` runs the physical step (velocity update, new inertial prediction, new rigid initializer) and then copies the history over:

```python
            payload = self.step.prepare(key, torch.from_numpy(initial.positions), torch.from_numpy(initial.velocities))
            payload.update(context_spec=specification, metadata=initial.metadata, seed=seed, physical_age=0)
            payload.update(empty_history(self.cell_count))
            return self._candidate(payload)
...
        payload = _cpu(payload)
        prepared = self.step.advance(payload)
        for key in ("context_spec", "metadata", "seed"):
            prepared[key] = payload[key]
        prepared["physical_age"] = payload["physical_age"] + 1
        carry_history(payload, prepared)
        return self._candidate(prepared)
```
(`experiments/learned_intrinsic_solver/train_mixed.py:283-286, 295-301`)

Neither `prepare` nor `_candidate` writes history: the initializer motion of a new candidate (inertial prediction, optional smooth noise) is not an optimizer update and never appears in `axis_update_world`. `_batch` collates a checked-out batch, infers `C` from the first stored block when not given, and yields `history=None` when no payload in the batch has any history entries:

```python
    if cell_count is None:
        blocks = [p.get(HISTORY_KEYS[0]) for p in payloads]
        blocks = [block for block in blocks if isinstance(block, torch.Tensor) and block.ndim == 3]
        if blocks:
            cell_count = int(blocks[0].shape[0])
        elif any(p.get("history_valid", False) for p in payloads):
            raise ValueError("payload history_valid is set without stored history blocks")
    values["history"] = batch_history(payloads, device, cell_count=cell_count) if cell_count is not None else None
    return values
```
(`experiments/learned_intrinsic_solver/train_mixed.py:354-362`)

In the update loop the order is: forward with `history=batch.get("history")` (experiments/learned_intrinsic_solver/train_mixed.py:375), loss, `backward`, `optimizer.step()`, then `store_history` and finally the detached fused positions become the next candidate:

```python
                payloads = [record.payload for record in records]
                store_history(payloads, result)
                for i, record in enumerate(records):
...
                    record.payload["candidate"] = result.positions[i].detach()
                pool.finish_batch(records)
```
(`experiments/learned_intrinsic_solver/train_mixed.py:762-764, 777-778`)

`pool.finish_batch` (experiments/learned_intrinsic_solver/trajectory_pool.py:256) then decides per trajectory: another inner iteration reuses the payload as is, so the next query of the same physical step sees this query's history; a physical-step boundary submits `factory.advance` to a worker thread, which carries the history into the new payload; and a completed step budget retires the trajectory and schedules `factory.reset` for the next seed, which starts with an empty history. History is therefore cleared only at reset.

**Lifecycle in validation.** The validator must feed held-out queries the same history the network was trained with, so it uses the same functions. `_store_history` writes the payloads and immediately rebuilds `batch["history"]` for the next query of the loop:

```python
def _store_history(step, payloads, batch, result, device) -> None:
    """Write this query's history into the payloads and refresh the batch for the next query."""
    from . import history as history_module  # noqa: PLC0415 -- Optional training boundary.

    history_module.store_history(payloads, result)
    batch["history"] = history_module.batch_history(payloads, device, cell_count=len(step.cell_corner_indices))
...
        for _ in range(config.validation_iterations):
            iteration += 1
            result = _checked_forward(step, step, batch)
            batch["candidate"] = result.positions.detach()
            _store_history(step, payloads, batch, result, device)
            _record_iteration(step, batch, start, samples, result.loss.total)
```
(`experiments/learned_intrinsic_solver/mixed_validation.py:152-157, 181-186`)

The frozen-problem optimization loop above records residuals after each query; the physical loop below runs `iterations` queries per physical step and lets `factory.advance` carry the history across steps exactly as training does:

```python
        for physical in range(physical_steps):
            batch = _batch(payloads, device)
            if start is None:
                start = batch["physical_positions"].clone()
            for _ in range(iterations):
                result = _checked_forward(step, step, batch)
                batch["candidate"] = result.positions.detach()
                _store_history(step, payloads, batch, result, device)
            _record_physical_step(step, batch, start, samples, result.loss.total, completed + 1)
            for index, payload in enumerate(payloads):
                payload["candidate"] = batch["candidate"][index].detach().cpu()
            completed += 1
            for sample in samples:
                sample["physical_steps"] = completed
            if physical + 1 < physical_steps:
                payloads = [factory.advance(payload) for payload in payloads]
```
(`experiments/learned_intrinsic_solver/mixed_validation.py:218-233`)

`validate_full_horizon` reuses `_physical` at the largest available K and H, so its history pattern is the same.

**The one-query detach rule.** Everything that crosses from one query into the next is a constant for autograd, so the loss of query k is differentiated only through query k's own network evaluation, local axes, fusion solve and energy.

- The gradient feature is computed on a fresh graph and detached (experiments/learned_intrinsic_solver/input_assembly.py:177 and experiments/learned_intrinsic_solver/input_assembly.py:180), and the fusion adjoint discards autograd history (experiments/learned_intrinsic_solver/fusion.py:254).
- The history blocks are detached three times: by `store_history` when written (experiments/learned_intrinsic_solver/history.py:119), by `batch_history` when stacked (experiments/learned_intrinsic_solver/history.py:64) and by `check_history` when consumed (experiments/learned_intrinsic_solver/input_assembly.py:150).
- The achieved update is computed under `torch.no_grad()` from detached positions (experiments/learned_intrinsic_solver/mixed_physics.py:481).
- The next candidate is `result.positions[i].detach()` (experiments/learned_intrinsic_solver/train_mixed.py:777), and `finish_batch` detaches the payload again.
- Frames are detached in §8, so the `R^T` applied to the history does not carry gradients either.

The consequence is that there is no backpropagation through time: the network is trained on a one-step objective (energy after this update versus before it, §7) while receiving the two-step context as data. It receives no learning signal from future queries at all: the positions it produces become the next candidate, and the history blocks derived from them are fed forward, only as detached data, so query k's parameters never see a gradient from query k+1. One query's backward pass costs one fusion adjoint solve per object, not one per query in the trajectory.

## §13. Rest grid and multiscale displacement fields

**Where this lives / what this part does.** `generate_cuboid` in experiments/learned_intrinsic_solver/data.py:62 builds the canonical rest grid: under the trainer defaults (experiments/learned_intrinsic_solver/train_mixed.py:53) a block of 10 by 10 by 40 cubic cells with edge length `cell_size = 0.025` m, that is 4000 cells and 4961 shared corners. `build_hierarchy`, `interpolate_control_grid` and `generate_multiscale` in experiments/learned_intrinsic_solver/multiscale.py:61, experiments/learned_intrinsic_solver/multiscale.py:115 and experiments/learned_intrinsic_solver/multiscale.py:215 turn that grid into a smooth random displacement of the corners with the material z-min face held in place. The rest grid is built once per training process at experiments/learned_intrinsic_solver/train_mixed.py:477 and shared by every material context, network buffer and factory. `generate_multiscale` runs on a CPU preparation worker once per trajectory reset (inside `InitialStateAugmenter.reset`, §14) and once more for every perturbed candidate (§15). Everything here is NumPy float64; nothing touches Torch or the network.

**The rest grid container.** `VoxelGridData` (experiments/learned_intrinsic_solver/data.py:23) is a frozen dataclass holding the shared rest geometry plus two per-cell fields that the training path never reads (`cell_deformation`, identity, and `cell_velocity`, zeros, as built by `generate_cuboid` at data.py:133; they are written by the per-cell augmentation `augment_grid` at data.py:148 and read by `CornerProjector.project` in experiments/learned_intrinsic_solver/vbd_samples.py:84 and :94). The fields that matter downstream:

- `corner_rest_positions`, float64 `[P, 3]` in meters, with P = (nx+1)(ny+1)(nz+1) = 11 * 11 * 41 = 4961 for the default grid.
- `cell_corner_indices`, int64 `[C, 8]` with C = 4000; this is the connectivity that the energy, the fusion solve and the network geometry all index with.
- `cell_neighbors`, int64 `[C, 6]` in the fixed face order (-x, +x, -y, +y, -z, +z); `-1` marks an exposed face. The derived `cell_exposed_faces` (`cell_neighbors == -1`) becomes the first six columns of the per-cell `boundary_features` buffer at experiments/learned_intrinsic_solver/mixed_physics.py:189.
- `cell_counts`, `cell_size` and `cell_rest_centers` (float64 `[C, 3]`, the centers `origin + h * (i + 0.5)`).

**Corner ordering.** Global and local indices both run in x, y, z order with z varying fastest. The global corner id is `ix * (ny+1)(nz+1) + iy * (nz+1) + iz`, so the corner strides of the default grid are (451, 41, 1). A cell's eight local corners are the offsets (dx, dy, dz) in {0, 1}^3 in the same order, so local corner `k = 4 dx + 2 dy + dz`: corner 0 is the minimum corner, corner 1 is its +z neighbor, corner 2 is +y and corner 4 is +x. Cell 0 of the default grid therefore lists corners `[0, 1, 41, 42, 451, 452, 492, 493]`. The face-neighbor table adds or subtracts the cell stride (400, 40, 1) along each axis wherever the cell is not on that boundary; cell 0 has neighbors `[-1, 400, -1, 40, -1, 1]`.

```python
    cell_coordinates = np.indices(counts, dtype=np.int64).reshape(3, -1).T
    corner_coordinates = np.indices(corner_counts, dtype=np.int64).reshape(3, -1).T
    local_corners = np.indices((2, 2, 2), dtype=np.int64).reshape(3, -1).T
    corner_strides = np.array([corner_counts[1] * corner_counts[2], corner_counts[2], 1], dtype=np.int64)
    cell_corner_indices = (cell_coordinates[:, None, :] + local_corners[None, :, :]) @ corner_strides

    with np.errstate(over="ignore", invalid="ignore"):
        corner_positions = origin_array + size * corner_coordinates
        cell_centers = origin_array + size * (cell_coordinates + 0.5)
    if not np.isfinite(corner_positions).all() or not np.isfinite(cell_centers).all():
        raise ValueError("grid coordinates exceed the supported numeric range")

    cell_count = prod(counts)
    cell_indices = np.arange(cell_count, dtype=np.int64)
    cell_strides = (counts[1] * counts[2], counts[2], 1)
    neighbors = np.full((cell_count, 6), -1, dtype=np.int64)
    for axis, stride in enumerate(cell_strides):
        lower = cell_coordinates[:, axis] > 0
        upper = cell_coordinates[:, axis] < counts[axis] - 1
        neighbors[lower, 2 * axis] = cell_indices[lower] - stride
        neighbors[upper, 2 * axis + 1] = cell_indices[upper] + stride
```
(`experiments/learned_intrinsic_solver/data.py:104-124`)

The `@ corner_strides` product turns the `[C, 8, 3]` integer coordinates into `[C, 8]` flat indices in one shot. All index arithmetic is int64, positions are `origin + size * coordinate` in float64, and the function rejects non-positive or boolean counts, corner counts beyond int64 and nonfinite positions.

**Control grids and trilinear interpolation.** A control grid is a coarse lattice of displacement vectors, shape `[nx, ny, nz, 3]`, whose point counts include both bounds along every axis and which spans exactly the rest bounds. `interpolate_control_grid` maps every query position to normalized coordinates `(x - origin) / extent` in [0, 1], multiplies by the number of intervals, and blends the eight surrounding control vectors with trilinear weights (the product over axes of `1 - f` or `f`, where `f` is the fractional position inside the interval). The displacement field is continuous and piecewise trilinear, so the displaced mesh is always compatible: shared corners move together and no cell detaches from its neighbors. Queries whose normalized coordinate `(x - origin) / extent` falls outside [0, 1] by more than `1e-12` raise; the tolerance is a fraction of the side length, not a distance in meters.

```python
    normalized = (positions - origin) / extent
    if (normalized < -1e-12).any() or (normalized > 1 + 1e-12).any():
        raise ValueError("query positions must lie inside the rest bounds")
    intervals = np.array(controls.shape[:3]) - 1
    coordinate = np.clip(normalized, 0, 1) * intervals
    lower = np.minimum(np.floor(coordinate).astype(int), intervals - 1)
    fraction = coordinate - lower
    result = np.zeros_like(positions)
    for bits in np.ndindex(2, 2, 2):
        offset = np.array(bits)
        weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
        index = lower + offset
        result += weight[:, None] * controls[index[:, 0], index[:, 1], index[:, 2]]
    return result
```
(`experiments/learned_intrinsic_solver/multiscale.py:139-152`)

**The automatic hierarchy.** `build_hierarchy` starts at the voxel spacing and doubles the target spacing until no axis has more than `coarse_max_intervals = 4` control intervals; the interval count at each spacing is `ceil(counts / factor)`, so coarse grids need not divide the voxel counts but still span the exact bounds. From that list it keeps `max_levels = 3` evenly spaced entries including the finest and the coarsest. For the default grid the candidate spacings are 0.025, 0.05, 0.1, 0.2 and 0.4 m, and the chosen levels are `coarse` with (2, 2, 4) control points at 0.4 m, `middle` with (4, 4, 11) points at 0.1 m, and `fine` with (11, 11, 41) points at the voxel spacing. The requested amplitude of a level (a uniform per-component half width in meters) is its share of a total budget `strength * shortest_side` split with weights `spacing ** 1.5`, so the coarse level carries most of the motion: with `strength = 0.5` and shortest side 0.25 m the amplitudes are 0.1096, 0.0137 and 0.0017 m and they always sum to 0.125 m regardless of the number of levels.

```python
    counts = np.asarray(rest.cell_counts)
    extent = counts * rest.cell_size
    candidates = [(counts.copy(), rest.cell_size)]
    factor = 1
    while candidates[-1][0].max() > coarse_max_intervals:
        factor *= 2
        intervals = (counts + factor - 1) // factor
        candidates.append((intervals, factor * rest.cell_size))
    take = np.unique(np.linspace(0, len(candidates) - 1, min(max_levels, len(candidates))).round().astype(int))
    chosen = [candidates[index] for index in reversed(take)]
    weights = np.array([spacing for _, spacing in chosen]) ** 1.5
    amplitudes = strength * extent.min() * weights / weights.sum()
```
(`experiments/learned_intrinsic_solver/multiscale.py:89-100`)

**Generating one sample.** `generate_multiscale` draws every level from its own random stream, `SeedSequence(seed, spawn_key=(index,))` feeding a PCG64 generator, so changing the amplitude or presence of one level never alters another level's pattern. Each control component is uniform in `[-amplitude_m, amplitude_m]`; then the whole z-index-0 plane of control vectors is zeroed (`vectors[:, :, 0] = 0`). Trilinear interpolation reproduces control values exactly on the grid faces, so the displacement is identically zero on the material z-min face, which is the face that §14 and the solver pin. The per-level fields are interpolated at the rest corners, stacked into `[L, P, 3]` and summed.

```python
    origin = rest.corner_rest_positions.min(axis=0)
    extent = np.array(rest.cell_counts) * rest.cell_size
    controls, fields = [], []
    for index, level in enumerate(levels):
        if len(level.control_counts) != 3 or any(not isinstance(n, int) or n < 2 for n in level.control_counts):
            raise ValueError("control counts must be three integers of at least two")
        if not np.isfinite(level.amplitude_m) or level.amplitude_m < 0:
            raise ValueError("level amplitudes must be finite and nonnegative")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(index,))))
        vectors = rng.uniform(-level.amplitude_m, level.amplitude_m, (*level.control_counts, 3))
        vectors[:, :, 0] = 0
        controls.append(vectors)
        fields.append(interpolate_control_grid(vectors, rest.corner_rest_positions, origin=origin, extent=extent))
    fields = np.stack(fields)
    total = fields.sum(axis=0)
    checker = _GeometryScreen(rest)
    scale = 1.0
    for backtracking_steps in range(25):
        positions = rest.corner_rest_positions + scale * total
        combined = checker.check(positions)
        individual = tuple(checker.check(rest.corner_rest_positions + scale * field) for field in fields)
        if min(value for screen in (combined, *individual) for value in screen.values()) >= min_volume_ratio:
            return MultiscaleSample(
                positions, levels, tuple(controls), fields, scale, backtracking_steps, combined, individual
            )
        scale *= 0.5
    raise ValueError("Could not produce a valid shape within 24 deterministic amplitude halvings")
```
(`experiments/learned_intrinsic_solver/multiscale.py:248-274`)

**Orientation screening and deterministic backtracking.** A random field can fold cells. `_GeometryScreen.check` computes two dimensionless minima over all cells: the smallest ratio of signed tetrahedron volume to rest volume over Newton's five-tetrahedron split of each hexahedron (the split alternates with cell parity in the corner order that `ModelBuilder.add_soft_grid` uses, experiments/learned_intrinsic_solver/multiscale.py:169), and the smallest determinant of the trilinear deformation gradient `F = dx/dX` at 17 points per cell (the eight corners, the eight two-point Gauss locations at `(1 +- 1/sqrt(3)) / 2` and the center). Shape derivatives are divided by `cell_size`, so `det F` is 1 at rest and equals the local volume ratio. Starting from `scale = 1`, the combined field and each level alone are screened at that scale; if any value falls below `min_volume_ratio = 0.2` the common scale is halved. Up to 24 halvings are tried before a `ValueError`; there is no resampling, so a seed always maps to the same shape.

```python
    def check(self, positions: np.ndarray) -> dict[str, float]:
        if positions.shape != self.rest.corner_rest_positions.shape or not np.isfinite(positions).all():
            raise ValueError("positions must be finite with the same shape as the rest corners")
        corners = positions[self.rest.cell_corner_indices]
        relative = corners - corners[:, :1]
        gradients = np.einsum("nki,qkj->nqij", relative, self.derivatives)
        return {
            "min_tet_volume_ratio": float(np.min(self._tet_determinants(positions) / self.rest_tet_determinants)),
            "min_sampled_jacobian": float(np.linalg.det(gradients).min()),
        }
```
(`experiments/learned_intrinsic_solver/multiscale.py:187-196`)

Key implementation points:

- `MultiscaleSample.positions` is float64 `[P, 3]`; `effective_scale` (a power of one half) and `backtracking_steps` record how far the requested amplitudes were reduced, and `level_displacements` keeps the unscaled per-level fields for audit.
- The screen is finite sampling only: it does not certify global injectivity or exclude self-contact between distant cells (module docstring, experiments/learned_intrinsic_solver/multiscale.py:8).
- `screen_geometry` (experiments/learned_intrinsic_solver/multiscale.py:199) is the public wrapper and rebuilds the `_GeometryScreen` cache (tet topology, shape derivatives) on every call; `generate_multiscale` constructs the checker once and reuses it across backtracking steps.
- `seed` must be a nonnegative integer (booleans rejected) and `min_volume_ratio` must lie strictly in (0, 1).

## §14. Initial state and material sampling

**Where this lives / what this part does.** `InitialStateAugmenter` in experiments/learned_intrinsic_solver/initial_state.py:68 (constructor at experiments/learned_intrinsic_solver/initial_state.py:79, `reset` at experiments/learned_intrinsic_solver/initial_state.py:141, `_float32_positions` at experiments/learned_intrinsic_solver/initial_state.py:123) produces the starting positions, velocities, pin indices and material of one trajectory. `sample_material` and `lame_from_youngs_modulus` in experiments/learned_intrinsic_solver/material_sampling.py:114 and experiments/learned_intrinsic_solver/material_sampling.py:82 draw the material. `_TrajectoryFactory.reset` constructs a fresh augmenter and calls `reset` once per trajectory reset (experiments/learned_intrinsic_solver/train_mixed.py:270) on a CPU preparation worker. The module is NumPy only and builds no Newton or Torch state; the trainer converts the returned float32 arrays with `torch.from_numpy`.

**Construction.** The constructor regenerates the canonical cuboid from `rest.cell_counts`, `rest.cell_size` and the first corner, refuses any grid whose corner positions or topology differ, and keeps that owned canonical copy so a caller can never mutate the source of future resets. The pinned corners are all corners whose rest z equals the minimum z (121 corners for the default grid, stored as int64). Ranges are validated as finite nonnegative `(lower, upper)` pairs. The defaults differ between the class and the trainer: the class defaults to `perturbation_scale_range = (1.0, 1.0)`, while `MixedTrainConfig` passes `(0.0, 1.0)` (experiments/learned_intrinsic_solver/train_mixed.py:99) together with `strength_range = (0.02, 0.1)`, `velocity_dt_range = (0.0, 0.1)` and `time_step = 1/300`.

**Seed streams.** Every random quantity comes from a `numpy.random.SeedSequence` keyed by `[master_seed, seed, tag]`, where `seed` is the trajectory's physical seed and the tag separates purposes:

- tag 701, the physical stream: draws `strength`, then the velocity control vectors, then the target displacement magnitude, in that order.
- tag 1301, the perturbation stream: draws one global multiplier `perturbation_scale` in `perturbation_scale_range`, uniform in [0, 1] under the trainer defaults.
- tag 1103, the material stream: its first 64-bit output becomes `material_seed`, which is passed to `sample_material`.
- The per-level shape noise inside `generate_multiscale` uses the bare `seed` with spawn keys (§13), so the direction pattern of the shape depends on the physical seed alone while its amplitude depends on `strength` from stream 701.

Because the streams are independent, changing the material ranges leaves positions and velocities bit-identical, and setting `strength_range = (0, 0)` leaves the material unchanged (`test_material_stream_independent_of_shape_and_within_bounds`).

```python
        current = self.seed if seed is None else _seed(seed, "seed")
        rest = self._rest
        physical_stream = [self.master_seed, current, 701]
        rng = np.random.default_rng(np.random.SeedSequence(physical_stream))
        strength = float(rng.uniform(*self.strength_range))
        sample = generate_multiscale(
            rest,
            seed=current,
            strength=strength,
            max_levels=_MULTISCALE_MAX_LEVELS,
            min_volume_ratio=_MIN_VOLUME_RATIO,
        )
        perturbation_stream = [self.master_seed, current, 1301]
        perturbation_scale = float(
            np.random.default_rng(np.random.SeedSequence(perturbation_stream)).uniform(*self.perturbation_scale_range)
        )
        if perturbation_scale == 1.0:
            sampled_positions = sample.positions  # Preserve the legacy float32 conversion bit for bit.
        elif perturbation_scale == 0.0:
            sampled_positions = rest.corner_rest_positions
        else:
            sampled_positions = rest.corner_rest_positions + perturbation_scale * (
                sample.positions - rest.corner_rest_positions
            )
        positions, screen, extra_halvings = self._float32_positions(sampled_positions)
```
(`experiments/learned_intrinsic_solver/initial_state.py:143-167`)

**Shape.** `strength` is uniform in `strength_range`, so the multiscale budget is 2 to 10 percent of the shortest side (0.5 to 2.5 cm for the default grid). The sampled displacement `sample.positions - rest` is multiplied by `perturbation_scale`; the two exact branches cast `sample.positions` itself at scale 1, which keeps the positions bit-identical to the earlier generator without `perturbation_scale` (the smoke sampler `_Sampler._physical` in experiments/learned_intrinsic_solver/train_smoke.py:193, enforced by `test_legacy_sampler_shape_and_velocity_stream_match_exactly` in experiments/learned_intrinsic_solver/tests/test_initial_state.py:54) without relying on `rest + 1.0 * (sample - rest)` round-tripping exactly in float64, and return the rest grid at scale 0. Under the trainer's `(0.0, 1.0)` range the scale is continuous, so about one trajectory in ten starts with less than a tenth of the sampled multiscale displacement, which gives the network near-rest starts without a separate mode.

```python
    def _float32_positions(self, sampled: np.ndarray) -> tuple[np.ndarray, dict[str, float], int]:
        rest = self._rest
        rest32 = rest.corner_rest_positions.astype(np.float32)
        for halvings in range(_MAX_FLOAT32_HALVINGS):
            if halvings == 0:
                positions = sampled.astype(np.float32)
            else:
                positions = (
                    rest.corner_rest_positions + (0.5**halvings) * (sampled - rest.corner_rest_positions)
                ).astype(np.float32)
            positions[self._fixed] = rest32[self._fixed]
            if not np.isfinite(positions).all():
                continue
            screen = screen_geometry(rest, positions)
            if min(screen.values()) >= _MIN_VOLUME_RATIO:
                return positions, screen, halvings
        raise ValueError("float32 augmented shape remained invalid after deterministic backtracking")
```
(`experiments/learned_intrinsic_solver/initial_state.py:123-139`)

**float32 positions.** The physical state is float32 everywhere downstream, so `_float32_positions` casts the float64 sample, snaps the pinned rows to the float32 rest positions and re-runs `screen_geometry` on the rounded result (the screen converts back to float64 internally, but it now sees the rounded values). Rounding plus the pin snap can push a marginal shape below the 0.2 margin, in which case the float64 displacement is halved and recast, up to 24 extra halvings. The count is reported as `float32_backtracking_steps`, and the combined factor `augmentation_scale = effective_scale * perturbation_scale * 0.5 ** extra_halvings` tells how far the final shape is from the requested amplitude.

```python
        counts = tuple(min(n + 1, cap) for n, cap in zip(rest.cell_counts, _VELOCITY_CONTROL_CAPS, strict=True))
        controls = rng.uniform(-1, 1, size=(*counts, 3))
        controls[:, :, 0] = 0
        velocity = interpolate_control_grid(
            controls,
            rest.corner_rest_positions,
            origin=rest.corner_rest_positions[0],
            extent=np.asarray(rest.cell_counts) * rest.cell_size,
        )
        velocity[self._fixed] = 0
        unscaled_target_displacement_rms = float(rng.uniform(*self.velocity_dt_range) * rest.cell_size)
        norm = _rms(velocity)
        velocity *= unscaled_target_displacement_rms / (self.time_step * norm) if norm else 0
        velocity = velocity.astype(np.float32)
        if perturbation_scale == 0.0:
            velocity.fill(0)
        elif perturbation_scale != 1.0:
            velocity *= np.float32(perturbation_scale)
        velocity[self._fixed] = 0
        if not np.isfinite(velocity).all():
            raise ValueError("float32 initial velocity is nonfinite")

        material_stream = [self.master_seed, current, 1103]
        material_seed = int(np.random.SeedSequence(material_stream).generate_state(1, dtype=np.uint64)[0])
        material = sample_material(material_seed, ranges=self.material_ranges)
```
(`experiments/learned_intrinsic_solver/initial_state.py:168-192`)

**Velocity.** The initial velocity is a second, much coarser control grid: `min(n + 1, cap)` points per axis with caps (3, 3, 5), which gives (3, 3, 5) for the default grid, uniform components in [-1, 1], zeroed on the z-index-0 plane and trilinearly interpolated at the rest corners; pinned corners are then set to zero explicitly. Its magnitude is set by a target on the displacement per step. RMS is the root mean square, `sqrt(mean over corners of |v_i|^2)`, computed by `_rms` at experiments/learned_intrinsic_solver/initial_state.py:64 over all corners including the zeroed pins. The RMS of `dt * V` is set equal to `uniform(velocity_dt_range) * cell_size`, 0 to 10 percent of the cell edge or 0 to 2.5 mm per step; with `dt = 1/300` s that is an RMS speed of at most 0.75 m/s. The line `velocity *= target / (dt * norm) if norm else 0` multiplies by zero when the field is identically zero, avoiding a division by zero. The result is cast to float32, multiplied by `perturbation_scale` as a float32 scalar (skipped at scale 1, replaced by `fill(0)` at scale 0), zeroed at the pins again and checked finite.

**Material and metadata.** `material_seed` is the first `uint64` state word of the tag-1103 sequence, so it is usually far larger than 2^32 (for master seed 73 and physical seed 0 it is 13568577045297897401); `numpy.random.default_rng` accepts it. `metadata` records every seed sequence, all ranges, `strength`, `perturbation_scale`, `augmentation_scale`, both backtracking counts, the achieved `velocity_dt_rms_m` next to the requested value, the screen minima, the material as a dict, its Young's modulus and Poisson's ratio under `material_parameters`, and `generator_version` (`initial_state_v4` when damping is nonzero, otherwise `initial_state_v3`). The training loop reads `metadata["material_parameters"]` and `metadata["perturbation_scale"]` for its per-epoch histograms (experiments/learned_intrinsic_solver/train_mixed.py:769). `reset` finally stores the seed on the augmenter, so a bare `reset()` repeats it, and returns `InitialState(positions, velocities, fixed_indices.copy(), material, seed, metadata)` with float32 `[P, 3]` positions and velocities and int64 pins.

**Material sampling.** Young's modulus `E` [Pa] is the stiffness of a bar under uniaxial stretch; Poisson's ratio `nu` (dimensionless, below 0.5) is how much the bar thins sideways when stretched; the Lame parameters `lambda` and `mu` [Pa] are the two coefficients of the isotropic elastic law that the stable Neo-Hookean energy in this project consumes directly (`mu` is the shear modulus). Log-uniform means uniform in `log(value)`, so every decade of a range is equally likely. `sample_material` takes three uniform draws from `default_rng(seed)` and spends them in a fixed order:

- `draws[0]`: `E = exp(log lo + u * (log hi - log lo))`, log-uniform in `youngs_modulus` (1e3 to 1e6 Pa by default).
- `draws[1]`: `nu = lo + u * (hi - lo)`, linear-uniform in `poissons_ratio` (0.2 to 0.49).
- `draws[2]`: `rho` log-uniform in `density` (100 to 10000 kg/m^3).
- The viscosity `eta` [Pa s] comes from a separate `SeedSequence([seed, 1709])` stream and is log-uniform in `damping` whenever that range is not `(0, 0)` (the trainer uses 10 to 1000 Pa s, experiments/learned_intrinsic_solver/train_mixed.py:93). Being a separate stream, enabling damping leaves `E`, `nu` and `rho` of a given seed unchanged; being absolute rather than a multiple of stiffness, changing the modulus range leaves `eta` unchanged.

```python
    draws = np.random.default_rng(seed).random(3)

    def log_uniform(bounds: tuple[float, float], draw: float) -> float:
        lower, upper = bounds
        if lower == upper:
            return float(lower)
        return math.exp(math.log(lower) + float(draw) * (math.log(upper) - math.log(lower)))

    youngs_modulus = log_uniform(ranges.youngs_modulus, draws[0])
    lower_nu, upper_nu = ranges.poissons_ratio
    poissons_ratio = lower_nu + float(draws[1]) * (upper_nu - lower_nu)
    lame_lambda, lame_mu = lame_from_youngs_modulus(youngs_modulus, poissons_ratio)
    damping = 0.0
    if ranges.damping != (0.0, 0.0):
        draw = np.random.default_rng(np.random.SeedSequence([seed, 1709])).random()
        damping = log_uniform(ranges.damping, draw)
    return MaterialSample(
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        density=log_uniform(ranges.density, draws[2]),
        damping=damping,
    )
```
(`experiments/learned_intrinsic_solver/material_sampling.py:133-154`)

The Lame mapping is the textbook one:

```python
    lame_lambda = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    lame_mu = youngs_modulus / (2 * (1 + poissons_ratio))
    return float(lame_lambda), float(lame_mu)
```
(`experiments/learned_intrinsic_solver/material_sampling.py:106-108`)

`lambda = E nu / ((1 + nu)(1 - 2 nu))` and `mu = E / (2 (1 + nu))`; `nu < 0.5` keeps `lambda` finite and `nu >= 0` keeps it nonnegative, which `MaterialRanges.__post_init__` and `lame_from_youngs_modulus` both enforce. `MaterialSample` stores `lame_lambda`, `lame_mu`, `density` and `damping` and recovers `E` and `nu` as properties (`E = mu (3 lambda + 2 mu) / (lambda + mu)`, `nu = lambda / (2 (lambda + mu))`); the metadata's `material_parameters` uses those properties, so the recorded provenance matches the solver inputs. The four field names are exactly the keyword arguments of `MixedHexSolverStep.register_context` (experiments/learned_intrinsic_solver/mixed_physics.py:203), which is why `asdict(initial.material)` is passed straight through at experiments/learned_intrinsic_solver/train_mixed.py:280.

## §15. Trajectories: candidates, the active pool, physical advance and validation seeds

**Where this lives / what this part does.** `_TrajectoryFactory` in experiments/learned_intrinsic_solver/train_mixed.py:251 supplies the three callbacks (`reset`, `advance`, `retire`) that `ActiveTrajectoryPool` in experiments/learned_intrinsic_solver/trajectory_pool.py:88 schedules on a thread pool; `MixedHexSolverStep.prepare` and `MixedHexSolverStep.advance` in experiments/learned_intrinsic_solver/mixed_physics.py:506 and experiments/learned_intrinsic_solver/mixed_physics.py:543 do the per-step physics bookkeeping; experiments/learned_intrinsic_solver/history.py:28 defines the optimizer-history payload keys that ride along. `run_training` builds one training factory and one validation factory per rank (experiments/learned_intrinsic_solver/train_mixed.py:499) and one pool per rank with capacity `pool_multiplier * batch_size = 64` and batch size 16 (experiments/learned_intrinsic_solver/train_mixed.py:573). `take_batch` and `finish_batch` run once per optimizer update on the training thread; `reset` and `advance` run on `preparation_workers = 2` worker threads; `retire` runs on the training thread after backward.

**Vocabulary.** A trajectory is one object with one fixed material integrated for `H` physical timesteps (its step budget), each timestep solved by `K` learned inner iterations (its iteration budget). A payload is a plain dict of detached CPU float32 tensors plus Python metadata; it is the only thing that crosses between the worker threads, the pool and the training loop. The candidate is the current guess for the end-of-step corner positions that the network refines; `physical_positions` is the start-of-step state that every inner iteration measures against. The inertial prediction `Y = X + dt V + dt^2 a` with `a = gravity + F / m` is where the corners would land with no elastic force; it is the target of the inertia term of the implicit Euler energy and stays fixed across the `K` inner iterations of a step.

```python
    def __init__(self, step, rest, config, *, rank, validation=False):
        self.step, self.rest, self.config = step, rest, config
        self.prefix = f"{'validation' if validation else 'train'}-{rank}"
        self.master_seed = config.seed + (1000000007 if validation else 0)
        self.seed_parity = int(validation)
        # Preparation workers use CPU topology without synchronizing CUDA.
        self.fixed_indices = step.fixed_indices.detach().cpu().clone()
        self.cell_count = len(step.cell_corner_indices)
```
(`experiments/learned_intrinsic_solver/train_mixed.py:254-261`)

**Seed namespaces.** The factory's `master_seed` is `config.seed` for training and `config.seed + 1000000007` for validation, and `seed_parity` is 0 for training and 1 for validation. `reset(seed)` hands `2 * seed + seed_parity` to the augmenter, so training trajectories use even physical seeds under one master seed and validation trajectories use odd physical seeds under another: the physical, perturbation and material streams of §14 can never coincide even where the integer seeds overlap. Context ids are `f"{prefix}-{seed}"`, for example `train-0-73`, with prefix `train-{rank}` or `validation-{rank}`. The training pool starts its reset seeds at `config.seed + rank * 100000000` (experiments/learned_intrinsic_solver/train_mixed.py:581) and increments by one per reset, so context ids stay unique within a run and ranks never collide. The cheap validation uses seeds `range(rank, validation_count, world_size)` (0 to 511 by default, experiments/learned_intrinsic_solver/mixed_validation.py:568), and the full-horizon check uses the next `validation_full_count` seeds after them (512 to 527), distributed round-robin across ranks:

```python
def _full_horizon_seeds(config, rank, world_size) -> list[int]:
    """Return this rank's share of the fixed full-horizon subset, disjoint from the cheap seeds."""
    count = config.validation_full_count
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("validation_full_count must be a non-negative integer")
    first = config.validation_count
    return list(range(first + rank, first + count, world_size))
```
(`experiments/learned_intrinsic_solver/mixed_validation.py:539-545`)

**Reset.** A fresh `InitialStateAugmenter` is constructed per reset (cheap: it regenerates and validates the canonical grid) and its `InitialState` provides `positions`, `velocities` and `material`. `register_context` builds the per-material CPU objects the solver needs: the `HexImplicitEulerLoss` with its lumped mass (each cell's mass `rho * volume` split equally onto its eight corners and summed at shared corners, experiments/learned_intrinsic_solver/hex_energy.py:316), the `HexFusion` PARDISO factor with cell weights of a stiffness scale times `h^3`, and the native `RigidPosePredictor`. Then `step.prepare` snapshots the physical step and the factory adds `context_spec` (the four material numbers), the full `metadata`, the pool `seed`, `physical_age = 0` and the empty optimizer history before drawing the first candidate. If anything fails after registration the context is discarded so the registry never leaks a factor.

```python
        c = self.config
        initial = InitialStateAugmenter(
            self.rest,
            master_seed=self.master_seed,
            time_step=c.time_step,
            material_ranges=c.material_ranges(),
            strength_range=c.strength_range,
            velocity_dt_range=c.velocity_dt_range,
            perturbation_scale_range=c.perturbation_scale_range,
        ).reset(2 * seed + self.seed_parity)
        key = f"{self.prefix}-{seed}"
        specification = asdict(initial.material)
        self.step.register_context(key, **specification)
        try:
            payload = self.step.prepare(key, torch.from_numpy(initial.positions), torch.from_numpy(initial.velocities))
            payload.update(context_spec=specification, metadata=initial.metadata, seed=seed, physical_age=0)
            payload.update(empty_history(self.cell_count))
            return self._candidate(payload)
        except BaseException:
            self.step.discard_context(key)
            raise
```
(`experiments/learned_intrinsic_solver/train_mixed.py:269-289`)

**What `prepare` stores.** With the context lock held, the rigid predictor integrates the corners as one rigid body, the resulting rotation and translation are applied to the positions, and `fusion.fuse` with a zero axis increment reconciles that rigid guess with the pinned corners; this becomes `payload["candidate"]`, but `_candidate` overwrites it immediately, so the rigid initializer never becomes the learned starting point. The inertial prediction uses the unmodified `X`, `V`, gravity and `F / m`. All tensors are float32 CPU clones, and all but `fixed_positions` are `[P, 3]` (`fixed_positions` is `[len(fixed), 3]`, 121 rows for the default grid); the payload holds only the context id string, never a model, factor or native handle, which is what lets it be pickled into checkpoints.

```python
        with torch.no_grad(), context.lock:
            rigid = context.predictor.predict(x, velocity, force, self.time_step)
            fixed_positions = x[self._fixed_cpu].clone()
            base = x[None] @ rigid.rigid_delta_rotation.transpose(-1, -2) + rigid.rigid_delta_translation[:, None]
            zero_increment = x.new_zeros((1, len(self._rest.cell_corner_indices), 3, 3))
            candidate = context.fusion.fuse(base, zero_increment, fixed_positions[None])[0]
            inertial = make_inertial_prediction(
                x[None],
                velocity[None],
                self.time_step,
                explicit_acceleration=self._gravity_cpu + force / context.mass[:, None],
            )[0]
        return {
            "context_id": context_id,
            "physical_positions": x,
            "velocities": velocity,
            "candidate": candidate.detach(),
            "inertial_prediction": inertial.detach(),
            "fixed_positions": fixed_positions,
            "forces": force,
        }
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:521-541`)

**Two candidate modes, no screening.** `_candidate` draws from `SeedSequence([master_seed, seed, physical_age, 911])`, so the mode and the noise of a given trajectory and physical step are reproducible and independent of worker timing. With probability `PERTURBED_CANDIDATE_PROBABILITY = 0.5` (experiments/learned_intrinsic_solver/train_mixed.py:43) the mode is `perturbed_inertial`, otherwise `inertial`:

- `inertial`: the candidate is a clone of the inertial prediction.
- `perturbed_inertial`: `generate_multiscale(rest, seed=rng.integers(2**32), strength=0.1)` produces a smooth compatible field that is zero on the pinned face (§13); its displacement from rest is rescaled to an RMS of `uniform(0.01, 0.1) * cell_size`, 1 to 10 percent of the cell edge (0.25 to 2.5 mm for the default grid), cast to float32 and added to the inertial prediction.

In both modes the pinned rows are overwritten with `fixed_positions`. The only check is finiteness: inverted or collapsed cells are accepted because the stable Neo-Hookean law is finite for any finite shape, and there is no backtracking, shortening or fallback mode (`test_inverted_perturbed_candidate_is_accepted_without_fallback`). The payload gains `candidate` (detached) and `candidate_mode`, which the epoch report tallies.

```python
        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, payload["seed"], payload["physical_age"], 911])
        )
        perturbed = rng.random() < PERTURBED_CANDIDATE_PROBABILITY
        candidate = payload["inertial_prediction"].clone()
        if perturbed:
            sample = generate_multiscale(self.rest, seed=int(rng.integers(2**32)), strength=0.1)
            noise = sample.positions - self.rest.corner_rest_positions
            rms = float(np.sqrt(np.mean(np.sum(noise**2, axis=-1))))
            noise *= float(rng.uniform(0.01, 0.1) * self.config.cell_size) / rms if rms else 0
            candidate = candidate + torch.from_numpy(noise.astype(np.float32))
        candidate[self.fixed_indices] = payload["fixed_positions"]
        if not torch.isfinite(candidate).all():
            raise ValueError("nonfinite candidate initialization")
        payload.update(candidate=candidate.detach(), candidate_mode="perturbed_inertial" if perturbed else "inertial")
        return payload
```
(`experiments/learned_intrinsic_solver/train_mixed.py:319-334`)

**Advance across a physical step.** When the last inner iteration of a step finishes, the pool submits `factory.advance(payload)` to a worker. `_cpu` (experiments/learned_intrinsic_solver/train_epochs.py:55) first deep-copies every tensor to detached CPU storage so the worker never touches a tensor the training thread still holds. `MixedHexSolverStep.advance` then reconstructs the velocity from the committed candidate, `V_new = (candidate - physical_positions) / dt`, zeroes it on the pinned corners, and calls `prepare` once with the candidate as the new physical positions and the same external forces. The result is a fresh step snapshot: new `physical_positions`, new inertial prediction, new rigid guess. No energy is evaluated and no repair is attempted; an inverted committed candidate is carried as is.

```python
    def advance(self, payload: dict) -> dict:
        """Commit candidate displacement to velocity and prepare the next step once."""
        candidate = self._cpu_snapshot(payload["candidate"], "candidate")
        previous = self._cpu_snapshot(payload["physical_positions"], "physical_positions")
        velocity = (candidate - previous) / self.time_step
        velocity[self._fixed_cpu] = 0
        return self.prepare(payload["context_id"], candidate, velocity, forces=payload.get("forces"))
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:543-549`)

The factory copies `context_spec`, `metadata` and `seed` forward, increments `physical_age`, carries the optimizer history and draws the new candidate for the new step:

```python
    def advance(self, payload):
        from .history import carry_history  # noqa: PLC0415 -- Optional training boundary.
        from .train_epochs import _cpu  # noqa: PLC0415 -- Optional training boundary.

        payload = _cpu(payload)
        prepared = self.step.advance(payload)
        for key in ("context_spec", "metadata", "seed"):
            prepared[key] = payload[key]
        prepared["physical_age"] = payload["physical_age"] + 1
        carry_history(payload, prepared)
        return self._candidate(prepared)

    def retire(self, payload):
        self.step.discard_context(payload["context_id"])
```
(`experiments/learned_intrinsic_solver/train_mixed.py:291-304`)

**Optimizer history keys.** The network sees two blocks from the previous learned query, the un-normalized world-frame axis gradient and the achieved world-frame axis update, both float32 `[C, 3, 3]`, plus a validity flag. `empty_history` (experiments/learned_intrinsic_solver/history.py:32) writes zeros and `history_valid = False` at reset; `store_history` (experiments/learned_intrinsic_solver/history.py:99) writes the detached blocks of every query from the training loop at experiments/learned_intrinsic_solver/train_mixed.py:763; `carry_history` copies the three keys unchanged across a physical boundary, so the initializer motion of the new candidate never enters the history and only a reset clears it.

```python
def carry_history(source: Mapping, target: MutableMapping) -> None:
    """Copy optimizer history unchanged across a physical timestep boundary.

    A missing source entry is left missing, which ``batch_history`` treats as
    no history. Initializer motion of the new candidate never enters history.
    """
    for name in HISTORY_KEYS:
        if name in source:
            target[name] = source[name]
```
(`experiments/learned_intrinsic_solver/history.py:124-132`)

**The active pool: budgets and dispatch.** `ActiveTrajectoryPool` keeps `capacity` records alive. Each `TrajectoryRecord` has a unique `id`, a unique reset `seed`, and immutable budgets `iteration_budget` (K) and `step_budget` (H) sampled once at scheduling time by `random.Random(seed).choice` from the currently available counts; this budget RNG is separate from the reset seed counter, and because scheduling order is deterministic the budget sequence does not depend on worker timing. The available counts come from the curriculum every epoch through `set_available_counts` (experiments/learned_intrinsic_solver/train_mixed.py:663) and change only future resets. Every scheduled record is appended to two queues: `_dispatch`, the FIFO that decides batch composition, and `_pending`, meaning a preparation job is outstanding.

```python
    def _schedule_reset(self) -> None:
        record = TrajectoryRecord(
            id=self._next_id,
            seed=self._next_seed,
            iteration_budget=self._rng.choice(self.iteration_counts),
            step_budget=self._rng.choice(self.physical_step_counts),
        )
        self._next_id += 1
        self._next_seed += 1
        self._records[record.id] = record
        self._dispatch.append(record.id)
        self._pending.append(record.id)
        self._jobs[record.id] = (self._executor.submit(_prepare, self._reset, record.seed), "reset")
        self.stats["resets"] += 1
```
(`experiments/learned_intrinsic_solver/trajectory_pool.py:185-198`)

`take_batch` takes the first `batch_size` ids of the dispatch FIFO in order, blocks on each one's future in `_resolve` (the wait is accounted in `stats["wait_seconds"]`, the worker time in `stats["preparation_seconds"]`), moves the ids out of `_ready` or `_pending`, and marks the batch as checked out; only one batch may be out at a time. A pending head stalls dispatch even if later members are ready. That is deliberate: it makes fairness, batch composition and checkpoint continuation depend only on the FIFO, not on thread scheduling. At 4B capacity a preparation can overlap roughly three other batches before its turn comes.

```python
    def take_batch(self) -> list[TrajectoryRecord]:
        """Return exactly B distinct ready records in deterministic FIFO order."""
        self._ensure_open()
        if self._batch is not None:
            raise RuntimeError("finish the checked-out batch before taking another batch")
        if len(self._dispatch) < self.batch_size:
            raise RuntimeError("trajectory pool cannot supply a full batch")
        identities = list(islice(self._dispatch, self.batch_size))
        # Resolve the selected batch before changing queue membership. A failed
        # preparation leaves every trajectory accounted for during cleanup.
        for identity in identities:
            self._resolve(identity)
        for identity in identities:
            self._dispatch.popleft()
            if identity in self._ready:
                self._ready.remove(identity)
            else:
                self._pending.remove(identity)
        self._batch = identities
        self.stats["batches"] += 1
        return [self._records[identity] for identity in self._batch]
```
(`experiments/learned_intrinsic_solver/trajectory_pool.py:216-236`)

**`finish_batch`: three exits per record.** The training loop stores the history (`store_history(payloads, result)`, experiments/learned_intrinsic_solver/train_mixed.py:763), then writes the network's output back into each payload (`record.payload["candidate"] = result.positions[i].detach()`, train_mixed.py:777), then calls `finish_batch` (train_mixed.py:778) with the same records in the same order. `_payload` re-detaches every tensor so no autograd graph survives an iteration boundary. Then each record takes one of three paths: another inner iteration (back to `_ready` and the dispatch tail), a physical boundary (inner counter reset, `physical_step + 1`, an `advance` job submitted, back to `_pending` and the dispatch tail), or retirement (context discarded, record deleted, and `_schedule_reset` immediately replaces it with a fresh id, seed and budgets). Final steps retire directly, so no unused `advance` is ever prepared.

```python
        for record in records:
            record.payload = _payload(record.payload)
            record.inner_iteration += 1
            self.stats["inner_iterations"] += 1
            if record.inner_iteration < record.iteration_budget:
                self._ready.append(record.id)
                self._dispatch.append(record.id)
            elif record.physical_step + 1 < record.step_budget:
                record.inner_iteration = 0
                record.physical_step += 1
                self._pending.append(record.id)
                self._dispatch.append(record.id)
                self._jobs[record.id] = (self._executor.submit(_prepare, self._advance, record.payload), "advance")
                self.stats["advances"] += 1
            else:
                if self._retire is not None:
                    self._retire(record.payload)
                self._prepared.remove(record.id)
                del self._records[record.id]
                self.stats["retired"] += 1
                self._schedule_reset()
        self._batch = None
```
(`experiments/learned_intrinsic_solver/trajectory_pool.py:252-273`)

**Checkpoint state.** `state_dict` refuses while a batch is checked out, calls `quiesce` to resolve every pending job, and snapshots the queue orders, the budget RNG state, the seed and id counters, the available counts and every record with its payload copied by `.detach().cpu().clone()`, so live payloads and the checkpoint never share storage. `from_state_dict` validates the membership invariants (every id in exactly one of ready or pending and also in dispatch, unique seeds, counters within range) and rebuilds the pool with `initialize=False`, calling no callback; all restored records count as prepared. Native factors are never serialized: the trainer re-registers each context from the saved `context_specs` before restoring the pool (experiments/learned_intrinsic_solver/train_mixed.py:527).

```python
        self.quiesce()
        return {
            "format_version": 2,
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "next_seed": self._next_seed,
            "next_id": self._next_id,
            "iteration_counts": self.iteration_counts,
            "physical_step_counts": self.physical_step_counts,
            "rng_state": self._rng.getstate(),
            "records": [_payload(vars(record), checkpoint=True) for record in self._records.values()],
            "dispatch": list(self._dispatch),
            "ready": list(self._ready),
            "pending": list(self._pending),
            "stats": dict(self.stats),
        }
```
(`experiments/learned_intrinsic_solver/trajectory_pool.py:292-308`)

Key implementation points:

- Pool construction in `run_training` submits `capacity` (64) resets at once onto two workers; each reset costs one multiscale generation from `InitialStateAugmenter.reset` (plus a second one from `_candidate` when the trajectory's first candidate mode is `perturbed_inertial`, probability 0.5), a PARDISO factorization and a rigid predictor build, so the first batches wait on the FIFO head.
- The pool `seed` (`config.seed + rank * 100000000`) is both the first reset seed and the seed of the budget RNG; the budgets come from `curriculum.available_counts` at construction.
- Payload tensors are CPU float32 `[P, 3]` (`physical_positions`, `velocities`, `candidate`, `inertial_prediction`, `forces`), `fixed_positions` is `[len(fixed), 3]`, history blocks are `[C, 3, 3]`; `_batch` (experiments/learned_intrinsic_solver/train_mixed.py:337) stacks them onto the device once per update.
- `retire` only drops the registry entry; an autograd graph that still references the context's PARDISO factor keeps it alive until backward completes (experiments/learned_intrinsic_solver/mixed_physics.py:253).
- `close` drains all jobs, shuts the executor down and retires every prepared context once, continuing past callback failures and re-raising the first error.

```python
            counts = curriculum.available_counts
            pool = ActiveTrajectoryPool(
                capacity=config.pool_multiplier * config.batch_size,
                batch_size=config.batch_size,
                reset=factory.reset,
                advance=factory.advance,
                retire=factory.retire,
                iteration_counts=counts[0],
                physical_step_counts=counts[1],
                seed=config.seed + rank * 100000000,
                workers=config.preparation_workers,
            )
```
(`experiments/learned_intrinsic_solver/train_mixed.py:572-583`)

## Findings and caveats

- **R1 — Every query pays for three CPU sparse solves per object.** The forward
  fusion solve, the transposed solve that produces the gradient input, and the
  transposed solve in the backward pass each run on the CPU PARDISO factor of
  that object's material context, with synchronous GPU to CPU transfers
  (`experiments/learned_intrinsic_solver/mixed_physics.py:345`,
  `experiments/learned_intrinsic_solver/fusion.py:239`). Measured on one L40 with
  the campaign configuration (full grid, batch 16): forward 0.27 to 0.46 s,
  backward plus Adam 0.24 to 0.30 s, peak 4.8 GiB.
- **R2 — Tie-cell frames are approximate in float32 and jump at the tie threshold.**
  Cells whose closest rotation is ambiguous (collapsed or inverted with equal small
  singular values) get their frame from a perturbed matrix; in float32 that frame
  is accurate to roughly 1e-3 rad, and it changes discontinuously when a cell crosses
  the tolerance (`experiments/learned_intrinsic_solver/frames.py:168`). This only
  changes the coordinate system the network sees; RᵀF always reconstructs F and all
  history is stored in world coordinates.
- **R3 — The energy floor multiplier is provisional.** The loss scale is bounded
  below by `c · 2^-23 · V · (λ + 2μ + η/dt + ρh²/dt²)` with c = 1
  (`experiments/learned_intrinsic_solver/mixed_physics.py:408`). The value comes from
  a CPU formula probe on near-rest float32 cases, not from the trained network path.
- **R4 — Legacy checkpoints cannot be resumed.** Only the 61/6/24 input schema is
  accepted; a configuration or network from the earlier 38/5 or 86/6 schemas raises
  an explicit error and is never reshaped
  (`experiments/learned_intrinsic_solver/train_mixed.py:203`).
- **R5 — The learning-rate controller halves on noise.** The plateau controller
  halves the learning rate after five epochs without a 0.1 percent improvement of
  the selection metric (`experiments/learned_intrinsic_solver/training_schedule.py:50`).
  The metric is the mean final force residual over 512 validation starts, which moves
  by tens of percent between epochs; in the LIDO-v3 campaign the rate fell from 1e-4
  to 3.1e-6 within 63 epochs.
- **R6 — The rigid initializer is computed but not used as a candidate.** `prepare`
  still performs one zero-increment fusion solve to build a rigid-guided shape
  (`experiments/learned_intrinsic_solver/mixed_physics.py:506`); the candidate
  initializer replaces it with the inertial or perturbed-inertial start.
- **R7 — A zero objective gradient feeds ln(1e-12) = -27.6 as the log-RMS scalar.** `rms_normalize` floors the per-object RMS at `RMS_FLOOR = 1e-12` (experiments/learned_intrinsic_solver/features.py:257) and the assembly writes `gradient_rms.log()` into state index 59 without any clipping (experiments/learned_intrinsic_solver/input_assembly.py:265). An object whose projected gradient is exactly zero therefore presents -27.6 in that channel together with all-zero normalized gradient blocks, a value far outside anything a nonzero gradient produces; the network has to learn that this combination means nothing to do rather than a very small step.
- **R8 — FiLM consumes the 128-channel encoded conditioning and is the largest single parameter block.** Each layer is built with `conditioning_dim=hidden_dim` (experiments/learned_intrinsic_solver/network.py:285), so its FiLM is `Linear(128, 512)` with 66,048 parameters, about 20 percent of the default 323,214, and FiLM and both heads start at exact zero (experiments/learned_intrinsic_solver/network.py:97-99 and experiments/learned_intrinsic_solver/network.py:293-296). A freshly initialized network is blind to material, cell size, time step and damping until the FiLM weights move, and the six raw channels never reach a layer directly.
- **R9 — Prescribed corners add a constant inertia offset to every energy.** `prepare` applies the explicit acceleration (gravity plus force/mass) to all corners, pins included (experiments/learned_intrinsic_solver/mixed_physics.py:531), while the candidate initializer and fusion hold pins at the step-start positions, so Y_pin = x_pin + dt^2 g and the inertia sum over all P corners (experiments/learned_intrinsic_solver/mixed_physics.py:398) contains the constant sum_pins m_pin dt^2 |g|^2 / 2 (about 5.3e-4 J per kilogram of pinned mass at dt = 1/300 s). Its gradient is zero, so forces and residuals are unaffected, but it is present in E_before and E_after and therefore in the loss scale max(|E_before|, floor); for the campaign grid it is of the same order as the energy floor, so the energy of a converged step never reaches zero and the asinh normalization saturates on this constant rather than on the floor.
- **R10 — Every optimizer query evaluates the full [B,C,8,3,3] energy three times.** E_before under no_grad (experiments/learned_intrinsic_solver/train_mixed.py:679), the autograd pass inside objective_gradient for the gradient input (experiments/learned_intrinsic_solver/input_assembly.py:178) and the fused energy that is backpropagated (experiments/learned_intrinsic_solver/mixed_physics.py:480). When any object in the batch is damped, the metric difference is additionally computed for the whole batch (experiments/learned_intrinsic_solver/mixed_physics.py:400), undamped members included; they contribute zero but pay the compute.
- **R11 — The previous-gradient block saturates as the optimizer converges.** The previous axis gradient is divided by the current query's gradient RMS, not its own (experiments/learned_intrinsic_solver/input_assembly.py:255); once the gradient shrinks by more than about 10x between two queries, most components of state columns 27 to 35 sit at the +/-10 clip (experiments/learned_intrinsic_solver/features.py:258) and the block carries only sign information. This is the LeCO convention by design, but it means the network sees a saturated, not a scaled, previous gradient near the minimum.
- **R12 — Perturbed candidates re-run the multiscale generator and then discard what its screen certified.** Half of all candidates call `generate_multiscale(rest, seed=..., strength=0.1)` at experiments/learned_intrinsic_solver/train_mixed.py:325, which builds a three-level field and runs the 17-point Jacobian screen with backtracking over all 4000 cells on a CPU worker; the field is then normalized to an RMS of 1 to 10 percent of the cell edge and added to the inertial prediction with no screen of the sum. The screen therefore only certified the unnormalized noise against the rest grid and says nothing about the actual candidate (inverted candidates are accepted by design, see test_inverted_perturbed_candidate_is_accepted_without_fallback). Cost: one full multiscale generation per perturbed candidate per physical step, on top of the one per reset, all on the two preparation workers.

## Test coverage

Every row names a test under `experiments/learned_intrinsic_solver/tests/` and what it pins down; run the suite from the worktree root with `uv run --no-sync python -m unittest discover -s experiments/learned_intrinsic_solver/tests` (453 tests at this commit).

| Topic | Test | What it checks |
|---|---|---|
| §1 hop-1 topology | tests/test_network_geometry.py::TestGridNeighborhood::test_hop_one_uses_literal_z_fast_neighbors | 27 slots on a (3,3,5) grid; literal z-fast neighbor ids for a corner and an interior cell; self in slot 0; masked ids are 0 |
| §1 exact hop-2 shell | tests/test_network_geometry.py::TestGridNeighborhood::test_hop_two_is_an_exact_shell | 99 slots; the hop-2 neighborhood excludes every hop-1 cell |
| §1 hop-4 shell | tests/test_network_geometry.py::TestGridNeighborhood::test_hop_four_keeps_the_full_shell | 387 slots; center cell has 387 distinct valid ids all at Chebyshev distance 4; corner cell keeps 62 |
| §1 singleton grid | tests/test_network_geometry.py::TestGridNeighborhood::test_singleton_retains_only_self | 1x1x1 grid keeps only the valid self slot at hops 1, 2 and 4 |
| §1 topology argument validation | tests/test_network_geometry.py::TestGridNeighborhood::test_reject_invalid_counts_and_hops | nonpositive, float, bool and short counts and hops raise ValueError |
| §1 edge descriptor values | tests/test_network_geometry.py::TestEdgeFeatures::test_known_receiver_frame_and_axis_transport | all 24 channels of one directed edge equal hand-computed values; float32 in gives float32 out |
| §1 self slot | tests/test_network_geometry.py::TestEdgeFeatures::test_self_keeps_identity_frame_and_own_axes | self slot has zero offsets, identity relative frame and the cell's own axes |
| §1 rigid invariance | tests/test_network_geometry.py::TestEdgeFeatures::test_global_rigid_transform_leaves_two_batch_features_unchanged | a common world rotation plus translation leaves all features unchanged to 2e-6 for two batch items |
| §1 masked sentinels | tests/test_network_geometry.py::TestEdgeFeatures::test_masked_sentinels_are_safe_and_zero | out-of-range ids in masked slots never index; masked slots are all zero; valid slots unchanged |
| §1 detach boundary | tests/test_network_geometry.py::TestEdgeFeatures::test_detach_frames_but_keep_geometry_gradients | frames.grad is None while rest centers, current centers and local axes get finite nonzero gradients |
| §1 edge input validation | tests/test_network_geometry.py::TestEdgeFeatures::test_reject_incompatible_shapes_and_dtypes | shape mismatches raise ValueError, dtype mismatches raise TypeError, bad cell_size raises ValueError |
| §2 edge bias and edge value | tests/test_network.py::TestIntrinsicTransformer::test_edge_bias_and_value | an edge bias of log(3) yields weights 0.25/0.75; the edge value delivers 0.75*log(3); zero edge_val makes the layer an identity |
| §2 masked edges and empty row | tests/test_network.py::TestIntrinsicTransformer::test_masked_edges_and_empty_row | NaN padding and sentinel ids give finite output and gradients; masked slots and an all-masked row have zero attention; padding gets zero gradient |
| §2 slot permutation | tests/test_network.py::TestIntrinsicTransformer::test_neighbor_slot_permutation | reordering slots together with their edge features leaves the output unchanged |
| §2 FiLM and backpropagation | tests/test_network.py::TestIntrinsicTransformer::test_conditioning_and_backpropagation | finite nonzero gradients to features, edges, conditioning and every parameter; shifting the conditioning changes the output |
| §2 query chunking | tests/test_network.py::TestIntrinsicTransformer::test_query_chunking_and_cell_permutation | chunk size 1 equals chunk size 16 in output, attention and input gradients; renumbering cells permutes the output |
| §3 zero init and state_dict | tests/test_network.py::TestIntrinsicSolverNetwork::test_initial_target_and_state_dict | initial target equals the input axes, correction is zero, step is 0.5*max for every cell [B,N]; save/load round trip is bitwise equal |
| §3 default architecture | tests/test_network.py::TestIntrinsicSolverNetwork::test_default_one_layer_radius_one_and_training_config | hops (1,), one layer, 323214 parameters, node encoder 9+61 inputs, condition encoder 6 inputs, 27 slots, 27 valid for the center cell and 8 for a corner |
| §3 explicit hop sequence | tests/test_network.py::TestIntrinsicSolverNetwork::test_saved_explicit_three_layer_config_restores_strictly | hops (1,1,1) builds three layers and loads strictly; the default network rejects that state_dict |
| §3 output bounds and batch independence | tests/test_network.py::TestIntrinsicSolverNetwork::test_batch_independence_and_output_bounds | batch entries are independent; nine-value correction norm below 1; step in (0, max_step_size); target = axes + step*correction |
| §3 per-cell step | tests/test_network.py::TestIntrinsicSolverNetwork::test_per_cell_step_varies_across_cells | uniform 0.1 at init with max 0.2; nonzero head weights give different finite steps per cell within bounds |
| §3 network gradients | tests/test_network.py::TestIntrinsicSolverNetwork::test_solver_network_gradients | finite gradients on every parameter including the step head; nonzero gradients reach state, conditioning and edge inputs |
| §3 edge masking before the encoder | tests/test_network.py::TestIntrinsicSolverNetwork::test_network_masks_before_edge_encoder | NaN in masked edge slots still gives finite targets and finite edge gradients |
| §3 full-grid inference | tests/test_network.py::TestIntrinsicSolverNetwork::test_full_grid_float32_inference | (10,10,40) grid returns [1,4000,3,3] float32 targets and [1,4000] steps with 27 slots |
| §3 schema constants | tests/test_features.py::TestSchemaConstants::test_dimensions_and_order | block order, widths 45/14/61/6/24, schema version 3, RMS_FLOOR 1e-12, CLIP 10 |
| §3 packing order | tests/test_features.py::TestPackStateFeatures::test_packing_order_and_width | state index 9k+3i+j is entry (i,j) of block k; 45-50 face flags, 51-58 corner flags, 59 log RMS, 60 flag; width 61 float32 |
| §3 history gating | tests/test_features.py::TestPackStateFeatures::test_history_flag_zeroes_history_blocks | indices 27-44 are zeroed and the flag is 0.0 for objects without history; the other blocks are untouched |
| §3 scalar shapes | tests/test_features.py::TestPackStateFeatures::test_numeric_flag_and_broadcast_scalar_shapes | [B,1,1,1] log RMS, numeric history flags and [B,C,14] boundary flags are accepted |
| §3 packing validation | tests/test_features.py::TestPackStateFeatures::test_rejects_malformed_inputs | wrong boundary width, block shape, scalar count raise ValueError; dtype mismatches raise TypeError |
| §3 RMS floor | tests/test_features.py::TestRmsNormalize::test_zero_input_uses_floor_and_stays_finite | a zero field gives zero output, RMS 1e-12 and a finite log |
| §3 RMS value | tests/test_features.py::TestRmsNormalize::test_rms_value_and_log | RMS is taken per object over all cells and components; its log is exposed |
| §3 clipping | tests/test_features.py::TestRmsNormalize::test_clips_to_plus_minus_ten | normalized components are clipped to +/-10 with the own or a supplied RMS |
| §3 shared RMS | tests/test_features.py::TestRmsNormalize::test_reuses_supplied_rms | the previous gradient is normalized by the current RMS; a supplied zero RMS is floored |
| §3 to_local | tests/test_features.py::TestToLocal::test_is_frame_transpose_times_matrix | to_local equals R^T @ M |
| §3 conditioning formulas | tests/test_features.py::TestConditioningChannels::test_channel_formulas | all six channel formulas including log1p(eta/(mu dt)) and the cell-size and time-step channels |
| §3 conditioning validation | tests/test_features.py::TestConditioningChannels::test_rejects_invalid_inputs | nonpositive or nonfinite scalars, mismatched shapes and dtypes are rejected |
| §4 inertial prediction Y and inertia gradient m (X - Y) / dt^2 | tests/test_hex_energy.py::TestHexEnergy::test_inertial_prediction_and_exact_gradient | Y = X + dt V + dt^2 a exactly; inertia gradient and energy match the lumped-mass formula |
| §4 lumped mass assembly | tests/test_hex_energy.py::TestHexEnergy::test_lumped_mass_and_material_arrays | each cell adds rho h^3 / 8 to its eight corners; total mass equals sum of rho h^3; buffers have no parameters |
| §4 prepare builds native Y and rigid candidate without network or energy | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_prepare_matches_native_rigid_candidate_and_original_inertia_target | payload Y and candidate match the native Newton problem; pins unchanged; all tensors CPU float32 detached |
| §4 batched _energy equals per-object HexImplicitEulerLoss | tests/test_mixed_damping.py::TestMixedDamping::test_mixed_damped_forward_matches_independent_energy_and_keeps_gradients | total, elastic, inertia and damping of a mixed batch equal independent single-material modules; gradients reach every parameter |
| §4 inverted and collapsed candidates give finite energies | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_inverted_and_collapsed_candidates_are_accepted_with_finite_energy_and_gradients | all four energy terms, outputs and parameter gradients finite for J < 0 and J = 0 batches |
| §5 quadrature rule | tests/test_hex_energy.py::TestHexEnergy::test_quadrature_reproduces_affine_geometry | shape values sum to 1, gradients sum to 0, affine F reproduced at all 8 points, weights sum to h^3 |
| §5 analytic energies | tests/test_hex_energy.py::TestHexEnergy::test_simple_shear_analytic_energy, test_uniform_dilation_analytic_energy, test_zero_lambda_dilation_energy_and_derivative | shear energy independent of lambda; dilation matches the mapped formula in float32 and float64; lambda = 0 keeps the mu volumetric terms |
| §5 H-form equals the naive density; autograd stress equals the cofactor formula | tests/test_hex_energy.py::TestHexEnergy::test_density_matches_naive_formula_and_cofactor_stress | random F including J < 0 and J = 0; psi to 1e-10 and P to 1e-10 |
| §5 Lamé mapping | tests/test_hex_energy.py::TestHexEnergy::test_rest_zero_stress_and_lame_small_strain_hessian | psi(I) = 0, P(I) = 0, Hessian at I is the Lamé tensor, uniaxial stiffness lambda + 2 mu |
| §5 finite through collapse and inversion | tests/test_hex_energy.py::TestHexEnergy::test_collapsed_and_inverted_cells_are_finite | F = diag(0,1,1), F = -I, folded corner and F = 0 give finite energies and gradients; psi(0) = lambda / 2 with zero stress |
| §5 near-rest float32 accuracy of the H form | tests/test_hex_energy.py::TestHexEnergy::test_near_rest_float32_energy_accuracy | float32 energy within 5e-8 J of float64; gradient within 2e-5 relative; directional finite difference within 1 percent |
| §5 Gauss-point stress assembles the position gradient | tests/test_hex_energy.py::TestHexEnergy::test_gauss_point_stress_assembles_position_gradient | autograd dE/dX equals sum_q w_q P_q g_qk scattered to corners, with inverted points present |
| §5 full quadrature sees hourglass modes; rigid motion is stress free | tests/test_hex_energy.py::TestHexEnergy::test_full_quadrature_sees_center_hourglass, test_rest_and_rigid_motion_zero_energy_force | non-affine warp with identity center gradient is penalized; rotated and translated grid has zero energy and force |
| §6 affine metric damping energy and force | tests/test_damping_energy.py::TestDampingEnergy::test_affine_energy_and_analytic_gradient | energy h^3 eta / (2 dt) ||C - I||_F^2 and its analytic force match to 1e-12 |
| §6 finite rigid motion of the anchor gives zero damping | tests/test_damping_energy.py::TestDampingEnergy::test_finite_rigid_motion_of_deformed_anchor | rotated and translated deformed anchor gives damping below 1e-24 and force below 1e-11 |
| §6 anchor must be the physical-step start | tests/test_damping_energy.py::TestDampingEnergy::test_physical_anchor_is_required_and_changes_force | missing anchor raises; candidate as anchor gives zero damping; physical anchor gives a force |
| §6 gradcheck through both operands | tests/test_damping_energy.py::TestDampingEnergy::test_nonaffine_position_and_anchor_gradcheck | float64 gradcheck of the damping term with respect to positions and anchor |
| §6 matches Newton VBD solid damping | tests/test_damping_energy.py::TestDampingEnergy::test_native_vbd_tet_force_matches_hex_metric_stress | integrated metric stress reproduces the native VBD tetrahedral damping force after rest-volume assembly |
| §6 1/dt scaling and per-cell coefficients | tests/test_damping_energy.py::TestDampingEnergy::test_inverse_timestep_and_per_cell_coefficients | doubling dt halves the damping; zero-viscosity cells add nothing |
| §6 per-object coefficients in a mixed batch | tests/test_mixed_damping.py::TestMixedDamping::test_uniform_stretch_damping_matches_integrated_metric_difference | two objects with eta 0.8 and 3.2 give eta V (0.21)^2 / (2 dt) each |
| §6 anchor held through inner queries, replaced by advance | tests/test_mixed_damping.py::TestMixedDamping::test_anchor_is_held_until_physical_advance_without_changing_context | physical_positions and Y unchanged over two queries; after advance the committed candidate is the anchor and has zero damping |
| §7 position gradient and its fusion-adjoint projection | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_gradient_feature_matches_float64_fused_energy_derivative | position_gradient equals dE/dX with zeroed pins; axis_gradient_world equals the float64 derivative of the fused energy with respect to the increment at zero |
| §7 gradient feature includes damping | tests/test_mixed_damping.py::TestMixedDamping::test_gradient_feature_includes_damping_and_matches_float64_twin | damped and undamped gradient inputs differ; damped one matches a float64 twin through fusion |
| §7 frozen inputs and live weight path; force_residual_norm | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_gradient_feature_is_detached_and_learned_path_reaches_all_parameters | frames, axis gradient and position gradient carry no grad; loss reaches every parameter; force_residual_norm equals the norm of position_gradient |
| §7 batched backward equals per-object backward | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_mixed_forward_and_all_parameter_gradients_match_independent_single_objects | one network call per batch; outputs, energies and all parameter gradients match independent single-object references |
| §7 fusion backward adjoint | tests/test_fusion.py::TestHexFusion::test_float32_adjoint_and_finite_difference, test_float64_reference_gradcheck | adjoint identity and central finite differences for base, increments and pins; float64 gradcheck |
| §7 project_gradient is the adjoint of fuse | tests/test_fusion.py::TestHexFusionProjectGradient::test_float64_adjoint_identity, test_float64_matches_autograd_of_fuse | dot(project_gradient(g), D) equals dot(g_free, fuse(D) - fuse(0)); equals autograd of fuse for any D |
| §7 LeCO loss formula | tests/test_train_mixed.py::TestMixedTraining::test_local_objective_matches_the_leco_formula_with_the_floor | asinh(after / scale) + w relu((after - before) / scale) with scale = max(|before|, floor); gradient only to after; invalid floor raises |
| §7 E_before is the carried candidate energy | tests/test_train_mixed.py::TestMixedTraining::test_later_inner_iterations_normalize_by_the_carried_candidate_energy | before_joule of inner iteration 2 equals after_joule of iteration 1 |
| §7 energy floor formula | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_energy_floor_matches_material_formula | c eps32 V (lambda + 2 mu + eta / dt + rho h^2 / dt^2) per object, float32, detached, scaled by energy_floor_scale |
| §7 validation residual anchor | tests/test_mixed_validation.py::TestMixedValidation::test_residual_anchor_is_the_physical_step_start_at_every_observation | every recorded energy and residual differentiates with the physical-step start as anchor, never Y or the candidate |
| §7 validator residual equals the real step's residual | tests/test_mixed_validation.py::TestRealStepResidual::test_validator_residual_matches_the_real_step_with_the_physical_anchor | _free_force_residual_norms equals forward's force_residual_norm with the physical anchor and differs for candidate or Y anchors |
| §8 center deformation | tests/test_features.py::TestCenterDeformation.test_affine_map_recovers_matrix | an affine map x -> M x + t gives F = M for every cell, including an inverted M |
| §8 center deformation | tests/test_features.py::TestCenterDeformation.test_translation_invariant_and_differentiable | F ignores rigid translation and propagates gradients whose corner sums vanish |
| §8 closest proper rotation | tests/test_frames.py::TestClosestProperRotations.test_inverted_flips_smallest_direction_and_is_closest_proper | det F < 0 gives U diag(1, 1, -1) Vh and it beats every other proper candidate in Frobenius distance |
| §8 tie threshold | tests/test_frames.py::TestClosestProperRotations.test_tie_detection_thresholds | gap = s2 + s3 (proper) or s2 - s3 (inverted) against 1e-4 max(s1, 1) on diagonal cases, plus a looser tolerance |
| §8 tie-break | tests/test_frames.py::TestClosestProperRotations.test_rank_one_tie_break_matches_closed_form | rank-one F: the chosen frame maps v1 to u1 and is the family member closest to R_ref (closed form and brute force) |
| §8 tie-break | tests/test_frames.py::TestClosestProperRotations.test_reference_only_modifies_tie_cells | non-tie frames are bit-identical with or without a reference; without one the plain formula is kept |
| §8 equivariance | tests/test_frames.py::TestClosestProperRotations.test_whole_problem_rotation_equivariance | rotating F and R_ref by Q rotates all frames by Q, tie cells included, in float32 and float64 |
| §8 reference corners | tests/test_frames.py::TestSelectReferenceCorners.test_canonical_grid_z_min_face | p0, p1, p2 are the origin, the far diagonal corner and the smallest-ID equal-area corner of the clamped face |
| §8 reference rotation | tests/test_frames.py::TestReferenceRotation.test_rigid_equivariance | the reference frame rotates with the corners and ignores translation |
| §8 step integration | tests/test_mixed_physics.py::TestMixedHexSolverStep.test_frames_are_proper_and_reconstruct_inverted_and_tied_deformation | frames are proper, detached, R A = F for regular, inverted and mirrored objects; tie mask matches frames.py with the step's reference |
| §8 fallback | tests/test_mixed_physics.py::TestMixedHexSolverStep.test_reference_corners_buffer_and_fallback_without_three_pins | reference_corners buffer holds the clamped-face IDs; with two pins it is empty and the plain formula is kept |
| §9 exact affine fit | tests/test_fusion.py::TestHexFusion.test_single_pin_reproduces_full_affine_increment | one affine increment on all cells plus a translation is reproduced exactly with a single pin |
| §9 zero increment | tests/test_fusion.py::TestHexFusion.test_zero_increment_preserves_a_warped_base_exactly | zero targets with satisfied pins return the warped base bit-exactly |
| §9 least squares | tests/test_fusion.py::TestHexFusion.test_incompatible_targets_satisfy_free_vertex_stationarity | the free-corner gradient of the weighted quadrature error vanishes while pins stay exact |
| §9 adjoint | tests/test_fusion.py::TestHexFusion.test_float32_adjoint_and_finite_difference | target, base and pin gradients through the sparse solve match finite differences in float32 |
| §9 adjoint | tests/test_fusion.py::TestHexFusion.test_float64_reference_gradcheck | every input derivative matches an independent finite difference in float64 |
| §9 device bridge | tests/test_fusion.py::TestHexFusion.test_cuda_forward_and_all_input_adjoints_match_cpu | CUDA inputs give the CPU results and gradients without changing device or precision |
| §9 project_gradient | tests/test_fusion.py::TestHexFusionProjectGradient.test_float64_matches_autograd_of_fuse | project_gradient equals autograd.grad(<g, fuse(base, D, fixed)>, D) for any D |
| §9 PARDISO transpose | tests/test_pardiso.py::TestPardisoFactor.test_float32_multirhs_forward_and_transpose | float32 preserved; A X = B and A^T X = B on a nonsymmetric matrix; only phase 33 after construction |
| §9 per-object factors | tests/test_mixed_physics.py::TestMixedHexSolverStep.test_mixed_forward_and_all_parameter_gradients_match_independent_single_objects | a mixed batch equals independently composed single-object queries, including parameter gradients |
| §9 prepare zero increment | tests/test_mixed_physics.py::TestMixedHexSolverStep.test_prepare_matches_native_rigid_candidate_and_original_inertia_target | prepare matches the native rigid candidate fused with a zero increment and leaves the inertia target unchanged |
| §10 detach boundaries | tests/test_mixed_physics.py::TestMixedHexSolverStep.test_gradient_feature_is_detached_and_learned_path_reaches_all_parameters | frames and gradient feature carry no grad; state columns 18:45 and 59:61 have no position gradient; every parameter and head receives a finite gradient |
| §10 gradient feature value | tests/test_mixed_physics.py::TestMixedHexSolverStep.test_gradient_feature_matches_float64_fused_energy_derivative | G equals the derivative of the fused energy with respect to a world axis increment at zero |
| §10 normalization and history | tests/test_mixed_physics.py::TestMixedHexSolverStep.test_state_features_follow_leco_normalization_and_history_contract | current RMS shared by both gradients, own RMS for the update, floor, clip and history flag zeroing |
| §10 history plumbing | tests/test_train_mixed.py::TestMixedTraining.test_history_is_stored_after_a_query_carried_by_advance_and_absent_after_reset | detached world gradient and achieved update are stored after a query, carried by advance, absent after reset |
| §10 batching | tests/test_train_mixed.py::TestMixedTraining.test_batch_infers_history_or_reports_none | _batch collates stored history blocks and reports None for historyless payloads |
| §10 objective | tests/test_train_mixed.py::TestMixedTraining.test_local_objective_matches_the_leco_formula_with_the_floor | scale = max(|E_before|, floor); everything but the new energy is detached |
| §10 later iterations | tests/test_train_mixed.py::TestMixedTraining.test_later_inner_iterations_normalize_by_the_carried_candidate_energy | E_before at inner iteration 2 is the previous update's E_after |
| §10 training loop | tests/test_train_mixed.py::TestMixedTraining.test_training_smoke_reports_residuals_history_and_selection | two tiny CPU epochs run end to end with the report fields, residuals, history and checkpoints |
| Gradient feature equals the fusion adjoint of the zero-pinned energy gradient (mixed step, two materials) | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_gradient_feature_matches_float64_fused_energy_derivative | axis_gradient_world matches float64 autograd of E(fuse(x, D)) at D = 0 and a central finite difference per object; position_gradient equals dE/dX with zeroed pinned rows |
| Gradient feature, state block and residual (single-material step) | tests/test_solver_step.py::TestLearnedHexSolverStep::test_gradient_feature_is_the_projected_objective_gradient | world gradient = project_gradient(zeroed dE/dX), columns 18 to 26 = clip(R^T G / rms), column 59 = log rms, force_residual_norm = norm of the zeroed gradient |
| LeCO normalization and history contract of the packed state | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_state_features_follow_leco_normalization_and_history_contract | columns 0 to 17 geometric blocks, 18 to 26 own-RMS gradient, previous gradient shares the current RMS and clips at 10, previous update uses its own RMS, invalid object gets zero history blocks and flag 0, all-pinned grid floors the RMS at 1e-12 with finite log |
| Detach boundaries and diagnostics of forward | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_gradient_feature_is_detached_and_learned_path_reaches_all_parameters | columns 18 to 44 and 59 to 60 have no gradient to positions, frames/axis_gradient_world/position_gradient detached, achieved_axis_update_world = F_center(fused) - F_center(candidate), force_residual_norm = norm(position_gradient), every network parameter receives a finite gradient |
| Frozen frames and gradient feature in the position derivative | tests/test_solver_step.py::TestLearnedHexSolverStep::test_position_gradient_with_frozen_frames_and_gradient_feature | analytic dLoss/dx with replayed frames matches a finite difference that holds objective_gradient fixed |
| project_gradient adjoint identity | tests/test_fusion.py::TestHexFusionProjectGradient::test_float64_adjoint_identity | dot(project_gradient(g), D) equals dot(g_free, fuse(D) - fuse(0)) per batch element; output detached |
| project_gradient equals autograd of fuse at any D | tests/test_fusion.py::TestHexFusionProjectGradient::test_float64_matches_autograd_of_fuse | equals autograd.grad(dot(g, fuse(base, D, fixed)), D) and is independent of D (affine map) |
| Fixed rows never reach the transposed solve | tests/test_fusion.py::TestHexFusionProjectGradient::test_fixed_rows_are_ignored | NaN in pinned rows gives the same result as zeroed rows; a gradient supported only on pins projects to zero |
| float32 production path of project_gradient | tests/test_fusion.py::TestHexFusionProjectGradient::test_float32_smoke_matches_reference_and_autograd | float32 result is finite, detached, matches float32 autograd of fuse and the float64 reference to 1e-4 relative |
| CUDA input to project_gradient | tests/test_fusion.py::TestHexFusionProjectGradient::test_cuda_input_matches_cpu_and_keeps_device | CUDA gradient returns on CUDA, bitwise equal to the CPU result (skipped without CUDA) |
| All-pinned grid | tests/test_fusion.py::TestHexFusionProjectGradient::test_all_pinned_projection_is_zero | no free corner yields a zero [B, C, 3, 3] projection |
| RMS floor | tests/test_features.py::TestRmsNormalize::test_zero_input_uses_floor_and_stays_finite | zero field gives zero output, rms = 1e-12 with shape [B, 1, 1, 1], finite log |
| RMS definition | tests/test_features.py::TestRmsNormalize::test_rms_value_and_log | RMS is the root mean square over all C*9 components of one object |
| Clip at +/-10 | tests/test_features.py::TestRmsNormalize::test_clips_to_plus_minus_ten | outliers saturate at +/-10 after division by own or supplied RMS |
| Shared RMS for the previous gradient | tests/test_features.py::TestRmsNormalize::test_reuses_supplied_rms | supplied rms in [B] or [B,1,1,1] form is reused and floored |
| to_local is R^T M | tests/test_features.py::TestToLocal::test_is_frame_transpose_times_matrix | R @ to_local(R, M) reconstructs M; to_local(R, R) is the identity |
| State packing order and width | tests/test_features.py::TestPackStateFeatures::test_packing_order_and_width | five row-major 9-blocks, flags at 45 to 58, log RMS at 59, flag at 60, width 61, float32 |
| History flag zeroes both history blocks | tests/test_features.py::TestPackStateFeatures::test_history_flag_zeroes_history_blocks | columns 27 to 44 are zero and column 60 is 0.0 for an invalid object even with nonzero tensors |
| History payload helpers | tests/test_history.py::TestHistoryPlumbing::test_empty_history_has_zero_blocks_and_false_flag | empty_history gives zero float32 [C, 3, 3] blocks and history_valid False; rejects cell_count 0 |
| batch_history tolerance | tests/test_history.py::TestHistoryPlumbing::test_batch_history_treats_missing_or_invalid_entries_as_zero | missing or false entries stack as zeros with False; a true flag without blocks raises; result detached |
| store, batch, carry round trip | tests/test_history.py::TestHistoryPlumbing::test_store_then_batch_round_trip_and_carry | store_history writes detached per-object slices and the flag, batch_history restores them, carry_history copies the three keys by reference and leaves a missing source untouched |
| History blocks follow the previous query | tests/test_solver_step.py::TestLearnedHexSolverStep::test_history_blocks_follow_the_previous_query | second query's columns 27 to 44 equal the normalized, re-framed first-query gradient and achieved update; invalid flag equals no history; history changes the proposal |
| History validation | tests/test_mixed_physics.py::TestMixedHexSolverStep::test_invalid_contexts_inputs_and_history_are_rejected_explicitly | wrong batch, float64, NaN, non-bool or wrong-length valid flag raise ValueError |
| History validation (single-material step) | tests/test_solver_step.py::TestLearnedHexSolverStep::test_previous_positions_and_history_are_validated | same malformed-history cases raise in prepare_inputs and forward |
| Trainer lifecycle: store, carry, clear | tests/test_train_mixed.py::TestMixedTraining::test_history_is_stored_after_a_query_carried_by_advance_and_absent_after_reset | reset payload has zero blocks and flag False (state column 60 = 0); store_history after a query writes the step's tensors; factory.advance carries them unchanged with physical_age 1 (column 60 = 1); a fresh reset has no history |
| _batch history inference | tests/test_train_mixed.py::TestMixedTraining::test_batch_infers_history_or_reports_none | cell count read from the first stored block, historyless payloads give None, a true flag without blocks raises |
| Validator feeds the same history as training | tests/test_mixed_validation.py::TestMixedValidation::test_history_is_stored_after_every_query_and_carried_across_physical_steps | flag pattern [F, T, T, F, T, T] over optimization and physical loops with 3 physical steps; markers show the previous query's gradient; advance carries history |
| Full-horizon validation carries history | tests/test_mixed_validation.py::TestFullHorizonValidation::test_complete_subset_reports_final_statistics_and_carries_history | history flags [F, F] then [T, T] across two physical steps of one query each |
| Validator residual anchor | tests/test_mixed_validation.py::TestMixedValidation::test_residual_anchor_is_the_physical_step_start_at_every_observation | every recorded residual differentiates the energy with X_start as the damping anchor, never the candidate or Y |
| Validator residual equals the step's own residual | tests/test_mixed_validation.py::TestRealStepResidual::test_validator_residual_matches_the_real_step_with_the_physical_anchor | _free_force_residual_norms matches forward's force_residual_norm on the real mixed step |
| rest grid: shared corners, bounds, local corner order (+x is corner 4, +y is 2, +z is 1) | test_data.py::TestCuboidGeneration::test_shared_corners_and_rest_geometry | two adjacent cells share exactly 4 corners; min/max bounds and cell centers; local corner axis order |
| rest grid: neighbor table and exposed faces | test_data.py::TestCuboidGeneration::test_face_connectivity_and_exposed_flags | (-x,+x,-y,+y,-z,+z) order, -1 marks exposed faces, interior cell of a 3x3x3 grid has none, 54 exposed faces total |
| rest grid: single voxel and defaults | test_data.py::TestCuboidGeneration::test_rest_state_and_single_voxel | identity deformation, zero cell velocity, all faces exposed for one voxel |
| rest grid: argument validation | test_data.py::TestCuboidGeneration::test_invalid_generation_parameters | rejects non-positive/boolean counts, bad cell size and origin |
| multiscale hierarchy on non-divisible grids | test_multiscale.py::TestMultiscaleDeformation::test_automatic_odd_hierarchy | finest level has corner counts, coarsest at most 4 intervals, amplitudes sum to strength times shortest side, z-min face exactly fixed, control values reproduced at the bounds |
| trilinear interpolation exactness | test_multiscale.py::TestMultiscaleDeformation::test_affine_interpolation | an affine control field is reproduced to 3e-16 on a nonmatching grid |
| multiscale seeding | test_multiscale.py::TestMultiscaleDeformation::test_seed_source_and_clamp | per-level PCG64 streams reproduce; source grid and global NumPy RNG untouched |
| multiscale independent level streams | test_multiscale.py::TestMultiscaleDeformation::test_zero_amplitude_and_independent_streams | zero amplitude returns rest; changing one level leaves the others unchanged |
| orientation screen and backtracking | test_multiscale.py::TestMultiscaleDeformation::test_orientation_and_backtracking | collapsed corner gives negative Jacobian; oversized amplitude backtracks with effective_scale = 0.5**steps and both minima at least 0.2 |
| default 10x10x40 grid shapes | test_multiscale.py::TestMultiscaleDeformation::test_twenty_default_initial_states | 20 seeds finite, pass the 0.2 screen combined and per level, RMS displacement above 1 cm |
| initial state reproducibility | test_initial_state.py::TestInitialStateAugmenter::test_repeated_reordered_and_fresh_instance_resets_are_exact | reset(seed) is bit-identical across call order and fresh instances |
| initial state owned copies | test_initial_state.py::TestInitialStateAugmenter::test_result_and_source_mutation_cannot_change_future_reset | mutating results or the source grid does not change later resets |
| physical stream 701 compatibility | test_initial_state.py::TestInitialStateAugmenter::test_legacy_sampler_shape_and_velocity_stream_match_exactly | positions/velocities match the smoke sampler bit for bit at perturbation scale 1 |
| material stream 1103 independence | test_initial_state.py::TestInitialStateAugmenter::test_material_stream_independent_of_shape_and_within_bounds | material_seed equals SeedSequence([73, seed, 1103]) first word; shape changes leave material unchanged and vice versa; bounds respected |
| perturbation scale 0 and fractional | test_initial_state.py::TestInitialStateAugmenter::test_zero_and_fractional_global_perturbation_scale | scale 0 gives float32 rest and zero velocity; fractional scale blends displacement and velocity |
| perturbation stream 1301 | test_initial_state.py::TestInitialStateAugmenter::test_random_global_scale_is_reproducible_independent_and_bounded | random global scale reproducible, independent of other streams, within range |
| float32 dtypes, pins, screen, metadata | test_initial_state.py::TestInitialStateAugmenter::test_float32_screen_pins_and_metadata | float32 X/V, int64 pins at rest32 with zero velocity, screen minima at least 0.2, seed sequences recorded |
| Lame provenance in metadata | test_initial_state.py::TestInitialStateAugmenter::test_youngs_and_poisson_provenance_matches_derived_solver_parameters | E=1000, nu=0.25 gives lambda=mu=400 and material_parameters match |
| damping in metadata | test_initial_state.py::TestInitialStateAugmenter::test_damping_reset_records_sample_without_changing_geometry | damping recorded with generator_version v4 while geometry is unchanged |
| global RNG isolation | test_initial_state.py::TestInitialStateAugmenter::test_reset_does_not_advance_global_numpy_random_state | reset leaves np.random global state untouched |
| invalid config and unrecoverable float32 shape | test_initial_state.py::TestInitialStateAugmenter::test_invalid_configuration_and_unrecoverable_float32_shape_fail | bad ranges/seeds raise; float32 backtracking failure raises |
| material draw order and distributions | test_material_sampling.py::TestMaterialSampling::test_known_rng_draws_map_to_log_and_linear_ranges | draws[0] log-uniform E, draws[1] linear nu, draws[2] log-uniform rho, Lame values to 12 places |
| Lame conversion and inverse properties | test_material_sampling.py::TestMaterialSampling::test_standard_lame_conversion_and_inverse_properties | (1000, 0.25) gives (400, 400); nu=0 gives lambda=0; youngs_modulus/poissons_ratio properties invert |
| damping stream [seed, 1709] | test_material_sampling.py::TestMaterialSampling::test_damping_is_independent_log_uniform_and_reproducible | enabling damping leaves E, nu, rho unchanged; damping independent of stiffness range; log-uniform median check over 1024 seeds |
| damping bounds | test_material_sampling.py::TestMaterialSampling::test_damping_bounds_and_fixed_reference | fixed (100, 100) returns 100; (0, 1), negative, inverted and infinite bounds rejected |
| material bounds and reproducibility | test_material_sampling.py::TestMaterialSampling::test_default_samples_stay_within_requested_bounds | default samples inside ranges |
| material bounds and reproducibility | test_material_sampling.py::TestMaterialSampling::test_same_seed_repeats_and_sample_is_model_ready | same seed repeats; fields are exactly the register_context keywords |
| candidate modes | test_train_mixed.py::TestMixedTraining::test_candidate_modes_are_equiprobable_and_deterministic | exactly two modes, 40 to 60 percent perturbed over 240 seeds, deterministic per (seed, physical_age), noise RMS at most 10 percent of h, pins exact, no fallback keys |
| inverted candidates accepted | test_train_mixed.py::TestMixedTraining::test_inverted_perturbed_candidate_is_accepted_without_fallback | inverted inertial prediction kept in both modes; only NaN raises |
| history across reset/advance in the trainer | test_train_mixed.py::TestMixedTraining::test_history_is_stored_after_a_query_carried_by_advance_and_absent_after_reset | history written after a query, carried by advance, cleared by reset |
| pool checkpoint in the trainer | test_train_mixed.py::TestMixedTraining::test_exact_resume_preserves_updates_and_active_trajectories | resume reproduces updates and active trajectories exactly |
| pool budgets and seeds | test_trajectory_pool.py::TestTrajectoryPool::test_mixed_budgets_and_seeds_are_deterministic | K and H fixed per record and identical for 1 or 3 workers; reset seeds consecutive from the pool seed |
| FIFO fairness | test_trajectory_pool.py::TestTrajectoryPool::test_dispatch_reaches_every_initial_member_before_reusing_a_batch | every member is dispatched once before any repeats, also for K=32 cohorts |
| overlap of preparation with ready records | test_trajectory_pool.py::TestTrajectoryPool::test_other_ready_records_run_while_advance_is_blocked | a blocked advance at the tail does not stop ready members from forming a batch |
| one advance per physical boundary | test_trajectory_pool.py::TestTrajectoryPool::test_advance_updates_velocity_once_at_each_physical_boundary | advance called once per boundary, velocity updated then, history_valid carried into the next step |
| maximum horizon | test_trajectory_pool.py::TestTrajectoryPool::test_maximum_horizon_runs_32_iterations_on_each_of_128_steps | 32 x 128 schedule completes without a 129th preparation |
| single-step retire | test_trajectory_pool.py::TestTrajectoryPool::test_single_step_trajectories_retire_without_advancing | H=1 records retire directly |
| curriculum only affects new resets | test_trajectory_pool.py::TestTrajectoryPool::test_curriculum_changes_only_new_trajectory_budgets | set_available_counts leaves active budgets unchanged |
| pool checkpoint round trip | test_trajectory_pool.py::TestTrajectoryPool::test_checkpoint_preserves_pending_queue_rng_and_payload | from_state_dict calls no reset and yields identical future batches for 30 updates |
| checkpoint storage isolation | test_trajectory_pool.py::TestTrajectoryPool::test_checkpoint_payloads_are_independent_cpu_detached_copies | serialized tensors do not alias live payloads |
| detach at iteration boundaries | test_trajectory_pool.py::TestTrajectoryPool::test_iteration_boundaries_detach_all_payload_tensors | finish_batch detaches every payload tensor |
| preparation failure handling | test_trajectory_pool.py::TestTrajectoryPool::test_preparation_failure_is_not_replaced_by_fresh_reset | a failed reset/advance raises and is not silently replaced |
| close semantics | test_trajectory_pool.py::TestTrajectoryPool::test_close_drains_preparation_and_retires_each_context_once | close drains jobs and retires each prepared context exactly once |
| batch checkout discipline | test_trajectory_pool.py::TestTrajectoryPool::test_checked_out_batches_cannot_be_lost_or_finished_twice | second take_batch or wrong finish_batch raises |
| restore validation | test_trajectory_pool.py::TestTrajectoryPool::test_restore_rejects_inconsistent_queue_membership | inconsistent ready/pending/dispatch membership is rejected |
| physical advance velocity reconstruction | test_mixed_physics.py::TestMixedHexSolverStep::test_advance_recomputes_native_problem_once_from_committed_candidate | V = (candidate - X)/dt with zero pins; prepare called once; candidate and inertial prediction match the native solver |
| advance with an inverted candidate | test_mixed_physics.py::TestMixedHexSolverStep::test_advance_carries_inverted_committed_candidate_without_repair | folded committed candidate advanced as is with finite payload |
| prepare contents | test_mixed_physics.py::TestMixedHexSolverStep::test_prepare_matches_native_rigid_candidate_and_original_inertia_target | rigid candidate and inertial target match the native reference |
| history helpers | test_history.py::TestHistoryPlumbing::test_store_then_batch_round_trip_and_carry | store_history then batch_history round trip; carry_history copies the three keys |
| validation seed namespaces | test_mixed_validation.py::TestFullHorizonValidation::test_summary_keys_failed_sample_visibility_and_disjoint_seeds | full-horizon seeds are disjoint from the cheap validation set |
| validation seed distribution | test_mixed_validation.py::TestFullHorizonValidation::test_seeds_are_distributed_round_robin_and_arguments_are_validated | full-horizon seeds split round-robin across ranks; argument validation |
