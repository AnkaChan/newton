# Old network (v4 checkpoint) parameter map and input layouts

Source: `generated/training_v4_20260928/checkpoints/best_validation.pt` (read only), key `network_state`
(60 entries, 903 372 float parameters), built by `experiments.learned_intrinsic_solver.network.IntrinsicSolverNetwork`
with the v4 config: `hidden_dim W = 192`, `num_heads = 6` (head dim 32), `edge_hidden_dim = 96`,
`target_modes = 7`, `state_feature_dim = 121`, `conditioning_dim = 7`, `hops = (1,)`,
`edge_network = True`, `contact_tokens = True`, `max_step_size = 0.05`. Node input width
`21 + 121 + 17 = 159`. `experiments/lido/tests/oracle.py::network_state_dict_summary` lists the same keys.

Two differences from the parameter sketch in the design spec (section 5.2) matter for a weight mapping:

1. **FiLM is fed by a conditioning encoder, not by the 7 channels directly.** `condition_encoder`
   maps 7 -> 192 -> 192 (SiLU between) and the block's `film` is `Linear(192, 4 W = 768)`.
   The spec sketch says `film 7->4W`; a mapping from the v4 weights must keep the encoder.
2. **Node and edge encoders are two-layer MLPs** (`159 -> 192 -> 192`, `24 -> 96 -> 96`, SiLU between),
   not single linears. The contact token encoder is likewise `19 -> 64 -> 64`.

## Parameter table (state_dict order)

Buffers (grid dependent, not parameters; the oracle rebuilds them for small grids):

| key | shape | note |
|---|---|---|
| `neighbor_indices_1` | (4000, 27) int64 | slot 0 self, slots 1..26 lexicographic (dx,dy,dz) offsets, Chebyshev distance 1; masked slots hold 0 |
| `neighbor_mask_1` | (4000, 27) bool | False for out-of-grid slots |

Contact encoder (`contact_network.ContactEncoder`, width 64, 2 heads, FFN x4, pool 16):

| key | shape | init |
|---|---|---|
| `contact_encoder.token_encoder.0.weight` / `.bias` | (64, 19) / (64,) | default |
| `contact_encoder.token_encoder.2.weight` / `.bias` | (64, 64) / (64,) | default |
| `contact_encoder.attention_norm.weight` / `.bias` | (64,) / (64,) | LayerNorm default |
| `contact_encoder.qkv.weight` / `.bias` | (192, 64) / (192,) | default; reshaped `[M, 3, heads=2, 32]` |
| `contact_encoder.out_projection.weight` / `.bias` | (64, 64) / (64,) | default |
| `contact_encoder.ffn_norm.weight` / `.bias` | (64,) / (64,) | LayerNorm default |
| `contact_encoder.ffn.0.weight` / `.bias` | (256, 64) / (256,) | default |
| `contact_encoder.ffn.2.weight` / `.bias` | (64, 256) / (64,) | default |
| `contact_encoder.pool_projection.weight` / `.bias` | (16, 128) / (16,) | **zero-init**; input `[masked mean (64), masked max (64)]` |

Encoders:

| key | shape | init |
|---|---|---|
| `node_encoder.0.weight` / `.bias` | (192, 159) / (192,) | default |
| `node_encoder.2.weight` / `.bias` | (192, 192) / (192,) | default |
| `edge_encoder.0.weight` / `.bias` | (96, 24) / (96,) | default |
| `edge_encoder.2.weight` / `.bias` | (96, 96) / (96,) | default |
| `condition_encoder.0.weight` / `.bias` | (192, 7) / (192,) | default |
| `condition_encoder.2.weight` / `.bias` | (192, 192) / (192,) | default |

Transformer block `layers.0` (`IntrinsicTransformerLayer`, pre-LayerNorm, FiLM on both branches):

| key | shape | init |
|---|---|---|
| `layers.0.attention_norm.weight` / `.bias` | (192,) / (192,) | LayerNorm default |
| `layers.0.ffn_norm.weight` / `.bias` | (192,) / (192,) | LayerNorm default |
| `layers.0.qkv.weight` / `.bias` | (576, 192) / (576,) | default; output reshaped `[3, heads=6, 32]` in that order |
| `layers.0.edge_bias.weight` / `.bias` | (6, 96) / (6,) | default; per-head score bias from the (updated) edge feature |
| `layers.0.edge_val.weight` / `.bias` | (192, 96) / (192,) | default; per-edge value addition, reshaped `[heads=6, 32]` |
| `layers.0.out_projection.weight` / `.bias` | (192, 192) / (192,) | default |
| `layers.0.ffn.0.weight` / `.bias` | (768, 192) / (768,) | default |
| `layers.0.ffn.2.weight` / `.bias` | (192, 768) / (192,) | default |
| `layers.0.film.weight` / `.bias` | (768, 192) / (768,) | **zero-init**; chunks `(gamma_attn, beta_attn, gamma_ffn, beta_ffn)`, applied as `x (1 + gamma) + beta` |
| `layers.0.edge_update.0.weight` / `.bias` | (192, 480) / (192,) | default; input `[h_i (192), h_j (192), e_ij (96)]` (receiver, sender, encoded edge) |
| `layers.0.edge_update.2.weight` / `.bias` | (96, 192) / (96,) | **zero-init** (A02 edge network) |

Heads:

| key | shape | init |
|---|---|---|
| `output_norm.weight` / `.bias` | (192,) / (192,) | LayerNorm default |
| `correction_head.weight` / `.bias` | (21, 192) / (21,) | **zero-init**; output bounded `raw / sqrt(1 + \|raw\|^2)`, reshaped `[3, 7]` |
| `step_head.weight` / `.bias` | (1, 192) / (1,) | **zero-init**; `step = 0.05 sigmoid(.)` = 0.025 at init |

## Forward pass (old code, `network.py`)

```
cond   = condition_encoder(c)                                   # [B, C, 192]  (c = 7 channels per object, expanded per cell)
x      = node_encoder(cat[local_axes.flatten (21), state (121), contact_encoder(tokens, mask) (17)])
e      = edge_encoder(edge_features masked to 0 on invalid slots)   # [B, C, 27, 96]
-- block --
g1,b1,g2,b2 = film(cond).chunk(4)
n      = LN_attn(x) (1 + g1) + b1
q,k,v  = qkv(n).reshape(B, C, 3, 6, 32)
e'     = e + edge_update(cat[n_i, n_j, e])          (masked slots -> 0)
score  = (q_i . k_j) / sqrt(32) + edge_bias(e')     (-inf on masked slots; softmax over the 27 slots)
msg    = sum_j softmax_j * (v_j + edge_val(e'))
x      = x + out_projection(msg)
n      = LN_ffn(x) (1 + g2) + b2
x      = x + ffn(n)                                   (192 -> 768 -> SiLU -> 192)
-- heads --
y      = output_norm(x)
corr   = raw / sqrt(1 + sum raw^2),  raw = correction_head(y)          # [B, C, 3, 7]
step   = 0.05 * sigmoid(step_head(y))                                  # [B, C]
target = local_axes + step * corr
```

Contact encoder: `token_encoder` -> pre-LN masked self-attention over the <= 24 tokens of the same cell
(2 heads, score scale 1/sqrt(32)) -> `out_projection` residual -> pre-LN FFN residual -> masked mean and
masked max over tokens -> `pool_projection` (zero-init) -> 16 channels, plus `count / 24`. Cells without
tokens output exact zeros in all 17 channels; a contact-free batch passes one all-masked slot.

## Node feature, 159 values per cell (`features.pack_state_features`, `network.forward`)

Every 21-block is a `[3, 7]` matrix `R^T V` (rows = local component i, columns = mode m in the order
`a1, a2, a3, w12, w13, w23, w123`) flattened row-major, value index `7 i + m`.

| slice | width | content |
|---|---|---|
| 0:21 | 21 | `local_axes`: `R^T V` of the current candidate (prepended by the network) |
| 21:42 | 21 | `inertial_axis_offset`: `R^T (V_Y - V)` |
| 42:63 | 21 | `physical_axis_change`: `R^T (V - V_prev)` |
| 63:84 | 21 | `current_axis_gradient`: `clip(R^T G / rms_G, +-10)`, `G` = fusion-adjoint projected gradient in units `mu h^3` |
| 84:105 | 21 | `previous_axis_gradient`: `clip(R^T G_prev / rms_G, +-10)` (shares the current RMS); zeros when history invalid |
| 105:126 | 21 | `previous_axis_update`: `clip(R^T U_prev / rms_U, +-10)` (own RMS); zeros when history invalid |
| 126:132 | 6 | exposed-face flags `-x, +x, -y, +y, -z, +z` |
| 132:140 | 8 | fixed-corner flags in local corner order (z fastest) |
| 140:141 | 1 | `log_gradient_rms = ln max(rms_G, 1e-12)` (per object) |
| 141:142 | 1 | `history_valid` (1.0 / 0.0) |
| 142:158 | 16 | contact pooled channels (`pool_projection`, zero-init) |
| 158:159 | 1 | contact `count / M`, `M = 24` |

RMS: `sqrt(mean over cells and all 21 components)` per object, floor `1e-12`, clip `+-10`.

## Edge feature, 24 values per directed edge (receiver i, sender j) (`network_geometry.build_edge_features`)

| slice | width | content |
|---|---|---|
| 0:3 | 3 | rest offset `(c_j - c_i)_rest / h` |
| 3:6 | 3 | current offset in the receiver frame `R_i^T (c_j - c_i) / h` (centres = mean of the 8 corners) |
| 6:15 | 9 | relative frame `R_i^T R_j`, row-major |
| 15:24 | 9 | transported axes `R_i^T R_j A_j = R_i^T F_j`, row-major (centre `F` only, not the warping vectors) |

Slots: 27 per cell, slot 0 = self (identity relative frame, own axes), slots 1..26 = offsets `(dx, dy, dz)`
in lexicographic order over `[-1, 1]^3` with Chebyshev distance 1; masked slots are zero.

## Contact token, 19 values (`contact_features.build_contact_tokens`, owner-cell frame, dimensionless)

| slice | width | content |
|---|---|---|
| 0:3 | 3 | contact point `R_i^T (x_s - c_i) / h` |
| 3:6 | 3 | partner point `R_i^T (p - c_i) / h` |
| 6:9 | 3 | partner normal `R_i^T n` |
| 9:10 | 1 | `gap / r`, `gap = (x_s - p) . n` |
| 10:11 | 1 | approach rate `-(n . (x_s - x_s0)) / r` over the current step |
| 11:12 | 1 | `min(r_p / r, 10)` lateral partner radius |
| 12:13 | 1 | `log1p(kappa)`, `kappa = ke / (E h)` |
| 13:14 | 1 | `beta = kd / (ke dt)` |
| 14:15 | 1 | `mu_friction` |
| 15:18 | 3 | kind one-hot: plane, point, self |
| 18:19 | 1 | self flag (always 0 in v1) |

Tokens are grouped per owning cell (the cell of the surface sample), first 24 valid pairs in pair order,
padded with zero rows and a False mask. `r = 0.5 h` (contact radius default).

## Conditioning, 7 channels per object (`features.conditioning_channels`)

`log1p(lambda / mu)`, `log(rho h^2 / (mu dt^2))`, `log1p(|g| dt^2 / h)`, `log1p(eta / (mu dt))`,
`log1p(kappa)`, `beta`, `mu_friction`; contact-free objects have zeros in the last three.
