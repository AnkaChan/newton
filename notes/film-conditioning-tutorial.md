# FiLM conditioning in the learned intrinsic solver, step by step

This tutorial explains how the learned intrinsic deformation optimizer (LIDO) tells its transformer block which material, grid and time step it is working on. It follows the code under `experiments/learned_intrinsic_solver/` on branch `ankac/learned-instrinic-solver` at commit `123a6be9` (2026-09-27). It is written for a reader who knows finite elements and optimization but is new to transformer conditioning: every formula is in LaTeX, every claim is tied to a line of code, and every `file.py:NN` mention opens the embedded source at that line. The embedded sources are the committed tree at that commit. At the time of this build the working tree already carried uncommitted edits to `features.py` that replace the nine channels of §1 by seven dimensionless ones (the plan of §9), so §1 and §2 describe the committed nine-channel version and will need a follow-up once that change lands. Read it top to bottom; the pipeline map after the big picture says where each step lives.

Notation: $B$ objects in a batch, $N$ cells per object ($N = 4000$ on the canonical $10 \times 10 \times 40$ grid), $S$ neighbour slots per cell ($S = 27$ for the default hop-1 block), $D = 128$ hidden channels, $H = 4$ heads of $d = 32$ channels each, indexed by $p$ (not $h$, which is the cell edge). Vectors are columns and $\odot$ is the element-wise product. Two symbols are reused because both meanings are standard, and the text always says which is meant: $\beta$ is the contact damping ratio $k_d/(k_e\,\Delta t)$ in §1, §2 and §9, and the FiLM shift $\beta(c)$, $\beta_1$, $\beta_2$ in §4 and §5. To avoid two further clashes, energies are written $U$ so that $E$ is always Young's modulus, and the LayerNorm affine pairs are $(a^\text{LN}_1, b^\text{LN}_1)$, $(a^\text{LN}_2, b^\text{LN}_2)$ while $b_1, b_2$ are the condition encoder's biases.

## Big picture: why the network must be told the material

**Where this lives / what this part does.** The network `IntrinsicSolverNetwork` (experiments/learned_intrinsic_solver/network.py:278) sees each cell through per-cell inputs that are deliberately scale-free: the current axes $A = R^\top F$ (a strain measure, independent of stiffness and density), the inertial offset and the physical change of the axes (pure geometry), the objective-gradient blocks divided by their own RMS (`rms_normalize`, experiments/learned_intrinsic_solver/features.py:244), boundary flags, one history flag and one per-object scalar, `log_gradient_rms` (experiments/learned_intrinsic_solver/features.py:78), the natural log of the gradient RMS in joules. That scalar is the only state input that depends on the material (it carries $\log(\mu h^3)$; §9 returns to it). Everything else is geometric: two objects with the same shape and the same candidate positions present *the same* per-cell inputs whether they are made of soft foam or stiff rubber, and one scalar cannot tell the network how stiffness, density and time step balance.

But the correct update is not the same. The implicit-Euler objective of one cell is built from terms that scale as

$$ U_\text{elastic} \propto \mu\,h^3, \qquad U_\text{inertia} \propto \frac{\rho\,h^3}{\Delta t^2}\,\lVert x - y \rVert^2, \qquad U_\text{damping} \propto \frac{\eta\,h^3}{\Delta t}, $$

which is exactly the sum the loss's energy floor is built from, $\lambda + 2\mu + \eta/\Delta t + \rho h^2/\Delta t^2$ times the rest volume (`energy_floor`, experiments/learned_intrinsic_solver/mixed_physics.py:648). The balance between "follow the inertial prediction $y$" and "relax the strain" is therefore set by the dimensionless group $\Lambda = \rho h^2 / (\mu\,\Delta t^2)$, and how far a normalised descent direction should be followed depends on $\mu$, $\rho$ and $\Delta t$. Training samples Young's modulus log-uniformly over three decades and density over two:

```python
    youngs_modulus: tuple[float, float] = (1e3, 1e6)
    """Log-uniform Young's modulus bounds [Pa]."""

    poissons_ratio: tuple[float, float] = (0.2, 0.49)
    """Linear-uniform Poisson's ratio bounds; equal values fix the ratio."""

    density: tuple[float, float] = (100.0, 10000.0)
    """Rest density bounds [kg/m³]."""
```
(`experiments/learned_intrinsic_solver/material_sampling.py:20-27`, the default sampling ranges)

so $\Lambda$ alone spans about five decades across the training set. A network that cannot see the material would have to predict one compromise update for all of them.

**Conditioning** is the general name for feeding such side information (here the material, the grid size, the time step and the contact coefficients) into a network so that it changes how the main input is processed. **Modulation** is the particular way this project does it: instead of appending the material as extra input columns, a vector computed from the material *rescales and shifts* the hidden features of every cell inside the transformer block. That mechanism is FiLM. The rest of this tutorial builds it up from the nine raw numbers to the block equations.

## One network call through the conditioning path (where each section lives)

```
REGISTER   CPU, once per material context (MixedHexSolverStep.register_context)
  context.material = (lambda, mu, rho, eta)   context.contact = (ke, kd, mu_f)
  -> plain tensors on the step; no network weights are read

PREPARE    once per network query, whole batch (MixedHexSolverStep._prepare_inputs)
  stack materials [B, 4] and contact coefficients [B, 3]
  features.contact_ratios: kappa = ke / (E h), beta = kd / (ke dt), mu_f ............ §1
  features.conditioning_channels: five logs, one log ratio, three contact -> c_raw [B, 9]  §1, §2
  expand to cells: c_raw[:, None].expand(B, N, 9)  (a view, no copy) ................. §2, §6
  -> assemble_inputs -> LearnedHexInputs.conditioning [B, N, 9]

NETWORK    IntrinsicSolverNetwork.forward, once per query
  _expand_conditioning: accept [B, 9] or [B, N, 9] ..................................... §6
  condition_encoder: Linear(9,128) -> SiLU -> Linear(128,128) -> c [B, N, 128] ......... §3, §7
  for each block (default one block, hop 1, S = 27 slots):
    IntrinsicTransformerLayer.forward(features, edges, indices, mask, conditioning=c)
      film: Linear(128, 512)(c).chunk(4) = (gamma_1, beta_1, gamma_2, beta_2) ........... §4, §7
      attention_norm -> (1 + gamma_1) * . + beta_1 -> qkv -> masked edge-biased softmax   §5
      -> residual -> ffn_norm -> (1 + gamma_2) * . + beta_2 -> SiLU MLP -> residual ..... §5
  output_norm -> correction head, step head (not conditioned)

CONTEXT    what the design implies
  relatives: concatenation, adaptive LayerNorm (DiT), cross-attention, LeCO ............ §8
  cell-normalisation plan: five of nine channels carry absolute scale ................... §9
```

---

## §1. The nine raw conditioning channels: formulas in features.conditioning_channels and contact_ratios

**Where this lives / what this part does.** `conditioning_channels` (experiments/learned_intrinsic_solver/features.py:438) turns the per-object material tensors and the two grid scalars into a $[B, 9]$ tensor; `contact_ratios` (experiments/learned_intrinsic_solver/features.py:380) supplies the three contact numbers it needs. Both are pure functions without module state, called once per network query from `MixedHexSolverStep._prepare_inputs` (§2). The channel order is fixed by the tuple `CONDITIONING_CHANNELS`, and its length is the schema constant `CONDITIONING_DIM = 9` from which the network's first conditioning layer is sized (§3).

```python

CONDITIONING_CHANNELS = (
    "log1p_lame_lambda",
    "log1p_lame_mu",
    "log_density",
    "log_cell_size",
    "log_time_step",
    "log1p_damping",
    "log1p_contact_kappa",
    "contact_beta",
    "contact_mu",
)
"""Per-object conditioning channels in order; see :func:`conditioning_channels`."""

CONDITIONING_DIM = len(CONDITIONING_CHANNELS)
"""Number of conditioning channels (9): six material channels and three contact channels."""
```
(`experiments/learned_intrinsic_solver/features.py:83-98`, channel names in order, and the width constant)

The inputs are the per-object material $(\lambda, \mu, \rho, \eta)$ (first and second Lamé parameters in Pa, density in kg/m³, viscosity in Pa s), the two grid scalars $h$ (cell edge in m) and $\Delta t$ (s), and the contact coefficients $(k_e, k_d, \mu_f)$ (penalty stiffness in N/m, penalty damping in N s/m, friction coefficient). The nine channels are

$$ c_\text{raw} = \begin{pmatrix} \log\!\left(1 + \lambda / 10^5\right) \\ \log\!\left(1 + \mu / 10^5\right) \\ \log\!\left(\rho / 1000\right) \\ \log\!\left(h / 0.025\right) \\ \log\!\left(60\,\Delta t\right) \\ \log\!\left(1 + \eta / (\mu\,\Delta t)\right) \\ \log\!\left(1 + \kappa\right) \\ \beta \\ \mu_f \end{pmatrix}, \qquad \kappa = \frac{k_e}{E\,h}, \quad \beta = \frac{k_d}{k_e\,\Delta t}, \quad E = \frac{\mu\,(3\lambda + 2\mu)}{\lambda + \mu}. $$

| # | name in `CONDITIONING_CHANNELS` | formula | value that gives 0 | dimensionless? |
|---|---|---|---|---|
| 1 | `log1p_lame_lambda` | $\log(1 + \lambda/10^5)$ | $\lambda = 0$ | no (absolute Pa) |
| 2 | `log1p_lame_mu` | $\log(1 + \mu/10^5)$ | $\mu = 0$ | no |
| 3 | `log_density` | $\log(\rho/1000)$ | $\rho = 1000$ kg/m³ | no |
| 4 | `log_cell_size` | $\log(h/0.025)$ | $h = 0.025$ m | no |
| 5 | `log_time_step` | $\log(60\,\Delta t)$ | $\Delta t = 1/60$ s | no |
| 6 | `log1p_damping` | $\log(1 + \eta/(\mu\,\Delta t))$ | $\eta = 0$ | yes |
| 7 | `log1p_contact_kappa` | $\log(1 + k_e/(E h))$ | $k_e = 0$ | yes |
| 8 | `contact_beta` | $k_d/(k_e\,\Delta t)$, defined as 0 where $k_e = 0$ | $k_d = 0$ | yes |
| 9 | `contact_mu` | $\mu_f$ | $\mu_f = 0$ | yes |

The reference constants sit at the top of the module:

```python
_REFERENCE_LAME = 1e5
_REFERENCE_DENSITY = 1000.0
_REFERENCE_CELL_SIZE = 0.025
_REFERENCE_TIME_STEP = 1.0 / 60.0
```
(`experiments/learned_intrinsic_solver/features.py:118-121`, the four reference values)

Three reading notes:

- **Why `log1p` for some channels and `log` for others.** $\log(1 + x)$ is $0$ at $x = 0$ and behaves like $x$ for small $x$, so a zero Lamé $\lambda$ (a legal material, tested at experiments/learned_intrinsic_solver/tests/test_solver_step.py:126), zero viscosity or no contact ($k_e = 0$) gives a finite channel of exactly $0$ instead of $-\infty$. Density is always positive, so a plain $\log$ is safe and symmetric: $\rho = 100$ and $\rho = 10^4$ map to $\mp 2.30$.
- **Why divide by a reference.** Each channel is $0$ for the reference case ($\rho = 1000$, $h = 0.025$, $\Delta t = 1/60$), so the encoder in §3 sees numbers of order one rather than $10^5$.
- **Why the last four are ratios.** $\eta/(\mu\,\Delta t)$, $k_e/(E h)$, $k_d/(k_e\,\Delta t)$ and $\mu_f$ are dimensionless: they do not change when the whole scene is rescaled in length, stiffness and time. The first five are not; §9 returns to this.

The contact ratios are computed once and shared with the contact tokens, so both use one definition:

```python
    youngs_modulus = lame_mu * (3 * lame_lambda + 2 * lame_mu) / (lame_lambda + lame_mu)
    kappa = contact_ke / (youngs_modulus * size)
    stiff = contact_ke > 0
    # Divide by one where ke = 0 so the unselected branch stays finite.
    denominator = torch.where(stiff, contact_ke * step, torch.ones_like(contact_ke))
    beta = torch.where(stiff, contact_kd / denominator, torch.zeros_like(contact_kd))
    return torch.stack((kappa, beta, contact_mu), dim=-1)
```
(`experiments/learned_intrinsic_solver/features.py:429-435`, inside contact_ratios)

- Line 429 recovers Young's modulus from the Lamé pair, $E = \mu(3\lambda + 2\mu)/(\lambda + \mu)$; line 430 forms $\kappa = k_e/(E h)$, the penalty stiffness measured in units of the cell's own stiffness times its size.
- Lines 431-434 form $\beta = k_d/(k_e\,\Delta t)$ with a `torch.where` guard: where $k_e = 0$ the divisor is replaced by one and the result by zero, so a contact-free object yields exactly $\beta = 0$ with finite gradients.

The channel assembly itself:

```python
    contact = [torch.zeros_like(lame_mu) if value is None else value for value in (contact_ke, contact_kd, contact_mu)]
    ratios = contact_ratios(lame_lambda, lame_mu, *contact, size, step)

    channels = (
        (lame_lambda / _REFERENCE_LAME).log1p(),
        (lame_mu / _REFERENCE_LAME).log1p(),
        (density / _REFERENCE_DENSITY).log(),
        torch.full_like(lame_mu, math.log(size / _REFERENCE_CELL_SIZE)),
        torch.full_like(lame_mu, math.log(step / _REFERENCE_TIME_STEP)),
        (damping / (lame_mu * step)).log1p(),
        ratios[:, 0].log1p(),
        ratios[:, 1],
        ratios[:, 2],
    )
    return torch.stack(channels, dim=-1)
```
(`experiments/learned_intrinsic_solver/features.py:492-506`, inside conditioning_channels)

Line 492 substitutes zeros for omitted contact tensors, lines 496-501 are channels 1-6 in the order above, and 502-504 are the three contact channels ($\log(1 + \kappa)$ from `ratios[:, 0]`, then $\beta$ and $\mu_f$ unchanged). Channels 4 and 5 are `torch.full_like` because $h$ and $\Delta t$ are Python floats shared by the whole batch.

## §2. Worked example on the canonical grid, and where mixed_physics._prepare_inputs computes and expands the channels

**Where this lives / what this part does.** `MixedHexSolverStep._prepare_inputs` (experiments/learned_intrinsic_solver/mixed_physics.py:466) is the production caller of `conditioning_channels` in the mixed-material step (the single-material `LearnedHexSolverStep.__init__` also calls it, once at construction, and stores the result as a `conditioning` buffer, experiments/learned_intrinsic_solver/solver_step.py:170; this tutorial follows the mixed step). It runs once per network query for the whole batch, stacks the registered contexts' materials, calls the two functions of §1, and broadcasts the $[B, 9]$ result to every cell before handing it to `assemble_inputs`, which stores it as `LearnedHexInputs.conditioning` (experiments/learned_intrinsic_solver/input_assembly.py:66); `forward` then passes it to the network (experiments/learned_intrinsic_solver/mixed_physics.py:697 and experiments/learned_intrinsic_solver/mixed_physics.py:703).

```python
        material = torch.stack([context.material for context in contexts]).to(positions.device)
        coefficients = self._contact_coefficients(contexts, positions.device)
        channels = conditioning_channels(
            *material.unbind(-1),
            self.cell_size,
            self.time_step,
            contact_ke=coefficients[:, 0],
            contact_kd=coefficients[:, 1],
            contact_mu=coefficients[:, 2],
        )
        conditioning = channels[:, None].expand(-1, len(self.cell_corner_indices), -1)
```
(`experiments/learned_intrinsic_solver/mixed_physics.py:485-495`, inside _prepare_inputs)

- `material` is $[B, 4]$, one row $(\lambda, \mu, \rho, \eta)$ per context (line 485); `coefficients` is $[B, 3]$, one row $(k_e, k_d, \mu_f)$ (line 486).
- `conditioning_channels(*material.unbind(-1), self.cell_size, self.time_step, ...)` returns $c_\text{raw} \in \mathbb{R}^{B \times 9}$ (lines 487-494).
- Line 495 inserts a cell axis and `expand`s to $[B, N, 9]$. `expand` creates a view with stride 0 along the new axis: nothing is copied, and every cell of an object reads the same nine numbers (§6).

**Worked example.** Take the canonical material used across the notes: $E = 10^5$ Pa, $\nu = 0.3$, $\rho = 1000$ kg/m³, $\eta = 10$ Pa s, contact $\kappa = 1$, $\beta = 0.5$, $\mu_f = 0.3$, on the canonical grid $h = 0.025$ m at $\Delta t = 1/300$ s. First the Lamé pair (`lame_from_youngs_modulus`, experiments/learned_intrinsic_solver/material_sampling.py:106-107):

$$ \lambda = \frac{E\,\nu}{(1+\nu)(1-2\nu)} = \frac{10^5 \cdot 0.3}{1.3 \cdot 0.4} = 57\,692.31 \text{ Pa}, \qquad \mu = \frac{E}{2(1+\nu)} = \frac{10^5}{2.6} = 38\,461.54 \text{ Pa}, $$

and the check $E = \mu(3\lambda + 2\mu)/(\lambda + \mu) = 10^5$ Pa recovers the input. The contact coefficients that realise $\kappa = 1$ and $\beta = 0.5$ are $k_e = \kappa E h = 2500$ N/m and $k_d = \beta\,k_e\,\Delta t = 4.167$ N s/m. The nine channels, computed with `features.conditioning_channels` in the project's environment (float64) and checked against the formulas by hand (they agree to $10^{-15}$):

| # | channel | expression | value |
|---|---|---|---|
| 1 | `log1p_lame_lambda` | $\log(1 + 57692.31/10^5) = \log 1.57692$ | $0.455476$ |
| 2 | `log1p_lame_mu` | $\log(1 + 38461.54/10^5) = \log 1.38462$ | $0.325422$ |
| 3 | `log_density` | $\log(1000/1000)$ | $0$ |
| 4 | `log_cell_size` | $\log(0.025/0.025)$ | $0$ |
| 5 | `log_time_step` | $\log(60/300) = \log 0.2$ | $-1.609438$ |
| 6 | `log1p_damping` | $\log(1 + 10/(38461.54/300)) = \log(1 + 0.078)$ | $0.075107$ |
| 7 | `log1p_contact_kappa` | $\log(1 + 1) = \log 2$ | $0.693147$ |
| 8 | `contact_beta` | $0.5$ | $0.5$ |
| 9 | `contact_mu` | $0.3$ | $0.3$ |

So the network receives $c_\text{raw} = (0.4555,\ 0.3254,\ 0,\ 0,\ -1.6094,\ 0.0751,\ 0.6931,\ 0.5,\ 0.3)$ for this object. Two things to notice: channel 5 is $-1.61$ for *every* object in the v3 campaign, because the campaign runs at $\Delta t = 1/300$ while the reference is $1/60$ (a constant offset the encoder absorbs into its bias), and channel 6 is $0$ in the default campaign because the default viscosity range is $(0, 0)$ (experiments/learned_intrinsic_solver/material_sampling.py:29).

To see the range the encoder must cope with, here are channels 1-6 at the four corners of the training ranges ($\nu = 0.3$, $\eta = 10$ Pa s, same grid):

| $E$ [Pa] | $\rho$ [kg/m³] | $\lambda$, $\mu$ [Pa] | channels 1-6 |
|---|---|---|---|
| $10^3$ | $100$ | $576.9$, $384.6$ | $(0.0058,\ 0.0038,\ -2.3026,\ 0,\ -1.6094,\ 2.1748)$ |
| $10^3$ | $10^4$ | $576.9$, $384.6$ | $(0.0058,\ 0.0038,\ +2.3026,\ 0,\ -1.6094,\ 2.1748)$ |
| $10^6$ | $100$ | $576\,923$, $384\,615$ | $(1.9124,\ 1.5782,\ -2.3026,\ 0,\ -1.6094,\ 0.0078)$ |
| $10^6$ | $10^4$ | $576\,923$, $384\,615$ | $(1.9124,\ 1.5782,\ +2.3026,\ 0,\ -1.6094,\ 0.0078)$ |

Because of the `log1p`, the soft end is compressed: three decades of $E$ move channel 1 only from $0.006$ to $1.91$, with the two decades from $10^3$ to $10^5$ squeezed into $0.006$ to $0.46$, while density is spread symmetrically to $\pm 2.30$. The viscosity channel runs the other way, because the *ratio* $\eta/(\mu\,\Delta t)$ is large for soft materials.

## §3. The condition encoder MLP in network.py: Linear(9, 128), SiLU, Linear(128, 128) gives c in R^128

**Where this lives / what this part does.** `IntrinsicSolverNetwork.__init__` builds `condition_encoder` (experiments/learned_intrinsic_solver/network.py:391-393) and `forward` applies it once per query, before the transformer blocks run (experiments/learned_intrinsic_solver/network.py:481-482). Its output `condition` is the conditioning vector $c$ that every block's FiLM layer reads (experiments/learned_intrinsic_solver/network.py:499).

```python
        self.condition_encoder = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
...
        conditioning = _expand_conditioning(conditioning, batch, cells, self.conditioning_dim)
        condition = self.condition_encoder(conditioning)
...
            features = layer(features, encoded_edges[hop], indices, mask, conditioning=condition)
```
(`experiments/learned_intrinsic_solver/network.py:391-393, 481-482, 499`, constructor, then forward)

The encoder is a small multilayer perceptron (MLP: linear maps with an element-wise nonlinearity between them), here two linear layers around one nonlinearity:

$$ c = W_2\,\operatorname{SiLU}\!\left(W_1\,c_\text{raw} + b_1\right) + b_2, \qquad W_1 \in \mathbb{R}^{128 \times 9},\; W_2 \in \mathbb{R}^{128 \times 128}, \qquad \operatorname{SiLU}(z) = z\,\sigma(z) = \frac{z}{1 + e^{-z}}. $$

- **One vector per object.** $c_\text{raw}$ is identical for all cells of an object, so $c \in \mathbb{R}^{128}$ is too. The code nevertheless evaluates the encoder *after* the expansion to cells (line 481 expands, line 482 encodes), on a $[B, N, 9]$ tensor, so it computes the same 128-vector $N$ times. That costs $B N\,(9 \cdot 128 + 128 \cdot 128)$ multiply-adds per query, small next to attention, and it keeps the per-cell $[B, N, C]$ option of §6 open. LeCO does the reverse (encode once per graph, then index the result to nodes; §8).
- **Why an encoder at all, instead of feeding the nine numbers straight into FiLM.** The FiLM layer of §4 is linear in its input. If it read $c_\text{raw}$ directly, the scale and shift would be *linear* functions of the nine channels, and the network could not express a rule such as "shrink the step when $\Lambda$ is large *and* contact is stiff". The SiLU between $W_1$ and $W_2$ is the one place where channels interact nonlinearly; $W_2$ then mixes the 128 hidden units into the shared $c$. Note that $W_2$ and the block's FiLM matrix $W_f$ (§4) are two linear maps in a row, so mathematically they could be one $512 \times 128$ matrix; keeping them separate lets several blocks (a `hops` tuple with more than one entry) share one $c$ while owning their own FiLM heads.
- **Initialisation.** The encoder uses PyTorch's default `nn.Linear` initialisation, so $c$ is a random nonlinear image of $c_\text{raw}$ from the first step. The FiLM layer that consumes it is zero-initialised, so this randomness has no effect on the output until training moves $W_f$ (§4).

## §4. FiLM itself: definition, the Perez et al. paper, the 1 + gamma identity-at-init trick, and the four chunks of the per-block film layer

**Where this lives / what this part does.** Each `IntrinsicTransformerLayer` owns one `film` linear layer (experiments/learned_intrinsic_solver/network.py:120-123) and evaluates it at the start of `forward` (experiments/learned_intrinsic_solver/network.py:220-226). Its 512 outputs are split into four 128-vectors that scale and shift the normalised features in front of the attention branch and in front of the feed-forward branch (§5).

**Definition.** Feature-wise Linear Modulation (Perez, Strub, de Vries, Dumoulin and Courville, "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018, arXiv:1709.07871) conditions a layer's activations $x \in \mathbb{R}^D$ on a vector $c$ through one affine map per feature channel:

$$ \operatorname{FiLM}(x \mid c) = \gamma(c) \odot x + \beta(c), \qquad \gamma(c),\ \beta(c) \in \mathbb{R}^D, $$

where $\gamma$ and $\beta$ are the outputs of a small network (the "FiLM generator") applied to $c$. "Feature-wise" means that the same $\gamma_k$ multiplies channel $k$ at *every* position (here: every cell), so the conditioning cannot single out individual cells; it can only re-weight and bias channels. That restriction is what makes it cheap, and it is what made it work in the original visual-reasoning setting, where a question modulates the channels of a convolutional network.

**The FiLM generator here.** One linear layer per block produces all four vectors at once:

```python
        self.film = nn.Linear(conditioning_dim, 4 * hidden_dim) if conditioning_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
```
(`experiments/learned_intrinsic_solver/network.py:120-123`, inside IntrinsicTransformerLayer.__init__)

$$ \begin{pmatrix} \gamma_1 \\ \beta_1 \\ \gamma_2 \\ \beta_2 \end{pmatrix} = W_f\,c + b_f, \qquad W_f \in \mathbb{R}^{512 \times 128},\; b_f \in \mathbb{R}^{512}, $$

and the forward splits the 512 outputs with `chunk(4, dim=-1)` into four consecutive blocks of 128 channels:

```python
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
(`experiments/learned_intrinsic_solver/network.py:220-231`, inside IntrinsicTransformerLayer.forward)

The assignment is fixed by how the chunks are consumed, so read it off the code rather than off names: `modulation[0]` (output channels 0-127) **multiplies** the attention-branch input and `modulation[1]` (128-255) **shifts** it (line 230); `modulation[2]` (256-383) **multiplies** the feed-forward input and `modulation[3]` (384-511) **shifts** it (experiments/learned_intrinsic_solver/network.py:260). In the notation above, $\gamma_1$ is `modulation[0]`, $\beta_1$ is `modulation[1]`, $\gamma_2$ is `modulation[2]` and $\beta_2$ is `modulation[3]`.

**The "1 + gamma" trick.** The code does not apply $\gamma \odot x + \beta$ but

$$ \tilde{x} = (1 + \gamma) \odot \operatorname{LN}(x) + \beta . $$

With `film.weight` and `film.bias` zeroed at construction (lines 122-123), $\gamma = \beta = 0$ for every material, so $\tilde{x} = \operatorname{LN}(x)$: a freshly built block is an ordinary pre-norm transformer block (one whose LayerNorm sits at the entry of each branch, before attention and before the feed-forward network; §5 writes the block out), and the material has no influence until training decides it should. Without the "1 +", zero weights would give $\tilde{x} = 0$ and silence the block, while PyTorch's default random initialisation would give random, material-dependent per-channel gains before any learning. The zeros do not block learning: with $L$ the loss, $\partial L/\partial W_f = (\partial L/\partial [\gamma_1;\beta_1;\gamma_2;\beta_2])\,c^\top$ is nonzero as soon as $c \neq 0$ and the downstream gradient is nonzero, so $W_f$ leaves zero at the first optimizer step. (The tests perturb `film.weight` with a small random `normal_` only to make the conditioning path visibly active in a forward comparison: `std=0.05` at experiments/learned_intrinsic_solver/tests/test_network.py:85, `std=0.01` at experiments/learned_intrinsic_solver/tests/test_solver_step.py:53.) This zero-initialised "$1 + \gamma$" form is the same device as adaLN-Zero in diffusion transformers (§8).

## §5. Where FiLM acts inside IntrinsicTransformerLayer.forward: the full block equations

**Where this lives / what this part does.** `IntrinsicTransformerLayer.forward` (experiments/learned_intrinsic_solver/network.py:176-264) is the whole transformer block. It consists of two *residual branches*: each branch takes a normalised copy of its input, computes a correction from it, and adds that correction back to the *unnormalised* input, so the token travels through the block as a running sum $x_i \to y_i \to z_i$ that the branches only add to; that running sum is the *residual stream*. Because the normalisation sits at the entry of each branch, the block is called *pre-norm*. The first branch is attention over the $S$ neighbour slots of each cell, the second a SiLU feed-forward network. FiLM enters exactly twice, immediately after each LayerNorm, so it acts on the *input* of each branch and never on the residual stream itself. The default network has one such block (`hops = (1,)`), so this runs once per query.

Below, $x_i \in \mathbb{R}^{128}$ is the token of cell $i$ entering the block, $c$ the object's conditioning vector from §3, and $j(i, s)$ the cell in neighbour slot $s$ of cell $i$, with a validity indicator $\chi_{is} \in \{0, 1\}$ (self is always a valid slot).

**Step 1: LayerNorm, then FiLM, then projections.** `attention_norm` (experiments/learned_intrinsic_solver/network.py:109) is

$$ \operatorname{LN}_1(x) = a^\text{LN}_1 \odot \frac{x - \operatorname{mean}(x)}{\sqrt{\operatorname{var}(x) + \varepsilon}} + b^\text{LN}_1, $$

with mean and variance over the 128 channels of one token and learned $a^\text{LN}_1, b^\text{LN}_1 \in \mathbb{R}^{128}$ (not the encoder biases $b_1, b_2$ of §3). FiLM follows (lines 229-230 of the excerpt in §4) and the modulated token is projected to queries, keys and values (line 231):

$$ \tilde{x}_i = (1 + \gamma_1) \odot \operatorname{LN}_1(x_i) + \beta_1, \qquad \begin{pmatrix} q_i \\ k_i \\ v_i \end{pmatrix} = W_{qkv}\,\tilde{x}_i + b_{qkv}, \qquad q_i, k_i, v_i \in \mathbb{R}^{H \times d}. $$

Note that $\gamma_1$ and $\beta_1$ do not depend on $i$: the material sets one gain and one bias per channel, and the same pair is applied to all $N$ cells. What differs between cells is $\operatorname{LN}_1(x_i)$, so the *effect* of the gain is cell-dependent even though the gain is not.

**Step 2: masked, edge-biased attention over the neighbour slots.** The core is `_attend_chunk`, called on chunks of `query_chunk_size` receiver cells (the chunking changes memory, not the result):

```python
        batch, count, slots, _ = edge_features.shape
        # Remove poisoned padding before learned projections, not just after softmax.
        edges = torch.where(valid, edge_features, 0)
        if self.edge_update is not None:
            sender = normalized[:, indices]
            pair = torch.cat((receiver[:, :, None].expand(batch, count, slots, self.hidden_dim), sender, edges), dim=-1)
            edges = torch.where(valid, edges + self.edge_update(pair), 0)
        scores = (query[:, :, None] * key[:, indices]).sum(-1) * (self.head_dim**-0.5)
        scores = (scores + self.edge_bias(edges)).masked_fill(~valid, -torch.inf)
        # All-masked rows must not send NaNs through softmax or its backward.
        has_neighbor = valid.any(dim=2, keepdim=True)
        scores = torch.where(has_neighbor, scores, 0)
        weights = scores.softmax(dim=2).masked_fill(~valid, 0)
        edge_values = self.edge_val(edges).reshape(batch, count, slots, self.num_heads, self.head_dim)
        message = (weights[..., None] * (value[:, indices] + edge_values)).sum(dim=2)
        return message.reshape(batch, count, self.hidden_dim), weights
```
(`experiments/learned_intrinsic_solver/network.py:159-174`, inside _attend_chunk)

With the encoded edge features $e_{is} \in \mathbb{R}^{64}$ of slot $s$ (zeroed on invalid slots, line 161), the learned edge terms are $b_{is} = W_b\,e_{is} + b_b \in \mathbb{R}^{H}$ (`edge_bias`) and $u_{is} = W_u\,e_{is} + b_u \in \mathbb{R}^{128}$ (`edge_val`, reshaped to $H \times d$). Per head $p$:

$$ s^{p}_{is} = \frac{q^{p}_i \cdot k^{p}_{j(i,s)}}{\sqrt{d}} + b^{p}_{is} \quad (\chi_{is} = 1), \qquad s^{p}_{is} = -\infty \quad (\chi_{is} = 0), $$

$$ w^{p}_{is} = \frac{\exp s^{p}_{is}}{\sum_{s'=1}^{S} \exp s^{p}_{is'}}, \qquad m^{p}_i = \sum_{s=1}^{S} w^{p}_{is}\,\bigl(v^{p}_{j(i,s)} + u^{p}_{is}\bigr). $$

Lines 166-167 are the score and the $-\infty$ mask, line 171 is the softmax over the slot axis (`dim=2`) followed by `masked_fill(~valid, 0)` so invalid slots have weight exactly $0$, and lines 168-170 guard a row with no valid slot, which the canonical topology never produces. Line 173 forms the message with the edge values added to the gathered neighbour values. When `edge_network` is on (lines 162-165), the edges are first updated from the *FiLM-modulated* tokens of both endpoints, $e'_{is} = e_{is} + \operatorname{MLP}([\tilde{x}_i;\ \tilde{x}_{j(i,s)};\ e_{is}])$, so the material reaches the edge terms too.

**Step 3: residual, second LayerNorm, second FiLM, feed-forward, residual.**

```python
        attended = features + self.out_projection(torch.cat(chunks, dim=1))
        normalized = self.ffn_norm(attended)
        if modulation is not None:
            normalized = normalized * (1 + modulation[2]) + modulation[3]
        output = attended + self.ffn(normalized)
        if return_attention:
            return output, torch.cat(attention_chunks, dim=1)
        return output
```
(`experiments/learned_intrinsic_solver/network.py:257-264`, end of IntrinsicTransformerLayer.forward)

$$ y_i = x_i + W_o\,\operatorname{concat}_p\!\left(m^{p}_i\right) + b_o, \qquad \tilde{y}_i = (1 + \gamma_2) \odot \operatorname{LN}_2(y_i) + \beta_2, $$

$$ z_i = y_i + W_4\,\operatorname{SiLU}\!\left(W_3\,\tilde{y}_i + b_3\right) + b_4, \qquad W_3 \in \mathbb{R}^{512 \times 128},\; W_4 \in \mathbb{R}^{128 \times 512}. $$

Line 257 is the first residual (it adds to the raw `features`, not to the normalised copy), lines 258-260 the second LayerNorm and FiLM, line 261 the feed-forward residual; $z_i$ is the block output that `output_norm` and the two heads read (experiments/learned_intrinsic_solver/network.py:500-503).

**What FiLM can and cannot do here, one sentence each.** It can scale channels up or down before the $q, k, v$ projections, which changes which neighbours a cell attends to and what it reads from them, per material. It can shift channels, which amounts to a material-dependent bias on the projections. It cannot produce a cell-specific effect on its own, because $\gamma$ and $\beta$ are shared by all cells of the object; cell-specific behaviour comes from the interaction with the cell's own $\operatorname{LN}(x_i)$ and from attention.

## §6. Shapes and broadcasting: per-object c to every cell, and the [B, N, C] per-cell option (_expand_conditioning)

**Where this lives / what this part does.** `_expand_conditioning` (experiments/learned_intrinsic_solver/network.py:36-41) is a six-line helper used twice: by the network on the raw channels (experiments/learned_intrinsic_solver/network.py:481) and by every layer on the encoded $c$ (experiments/learned_intrinsic_solver/network.py:223). It accepts either one vector per object or one vector per cell and returns the per-cell form.

```python
def _expand_conditioning(conditioning: Tensor, batch: int, cells: int, channels: int) -> Tensor:
    if conditioning.shape == (batch, channels):
        return conditioning[:, None, :].expand(batch, cells, channels)
    if conditioning.shape != (batch, cells, channels):
        raise ValueError("conditioning must have shape [batch, channels] or [batch, cells, channels]")
    return conditioning
```
(`experiments/learned_intrinsic_solver/network.py:36-41`)

- A $[B, C]$ input gets a cell axis inserted and is `expand`ed to $[B, N, C]$ (lines 37-38). `expand` returns a view whose stride along the new axis is zero, so the $N$ copies share memory, and PyTorch's automatic differentiation (autograd) sums the gradient of the expanded tensor back onto the single vector.
- A $[B, N, C]$ input passes through unchanged (line 41); any other shape raises (lines 39-40).

The shapes along the whole path:

| tensor | shape | where |
|---|---|---|
| per-object material $(\lambda, \mu, \rho, \eta)$ | $[B, 4]$ | experiments/learned_intrinsic_solver/mixed_physics.py:485 |
| contact coefficients $(k_e, k_d, \mu_f)$ | $[B, 3]$ | experiments/learned_intrinsic_solver/mixed_physics.py:486 |
| $c_\text{raw}$ | $[B, 9]$ | experiments/learned_intrinsic_solver/mixed_physics.py:487 |
| expanded to cells (a view, no copy) | $[B, N, 9]$ | experiments/learned_intrinsic_solver/mixed_physics.py:495 |
| `LearnedHexInputs.conditioning` | $[B, N, 9]$ | experiments/learned_intrinsic_solver/input_assembly.py:66 |
| after `_expand_conditioning` in the network (a no-op on this input) | $[B, N, 9]$ | experiments/learned_intrinsic_solver/network.py:481 |
| $c$, the output of `condition_encoder` | $[B, N, 128]$ | experiments/learned_intrinsic_solver/network.py:482 |
| layer input `conditioning=c`, with `conditioning_dim = hidden_dim = 128` | $[B, N, 128]$ | experiments/learned_intrinsic_solver/network.py:399 and experiments/learned_intrinsic_solver/network.py:499 |
| `film(c)` | $[B, N, 512]$ | experiments/learned_intrinsic_solver/network.py:224 |
| each of the four chunks $\gamma_1, \beta_1, \gamma_2, \beta_2$ | $[B, N, 128]$ | experiments/learned_intrinsic_solver/network.py:224 |
| `normalized`, before and after FiLM | $[B, N, 128]$ | experiments/learned_intrinsic_solver/network.py:228-230 |

The wiring that makes the layers' `conditioning_dim` equal to the hidden width:

```python
        self.layers = nn.ModuleList(
            IntrinsicTransformerLayer(
                hidden_dim,
                edge_hidden_dim,
                num_heads=num_heads,
                conditioning_dim=hidden_dim,
                query_chunk_size=query_chunk_size,
                edge_network=edge_network,
                checkpoint_chunks=checkpoint_chunks,
            )
            for _ in self.hops
        )
```
(`experiments/learned_intrinsic_solver/network.py:394-405`, inside IntrinsicSolverNetwork.__init__)

Line 399 passes `conditioning_dim=hidden_dim`: the layers never see the nine raw channels, only the 128-dimensional $c$. That is why `film` is `Linear(128, 512)` and not `Linear(9, 512)`.

**The per-cell option.** Nothing in the network requires all cells of an object to share $c$. If `LearnedHexInputs.conditioning` carried different rows per cell, for instance a per-cell $h$ in a mixed-resolution grid or a per-cell material, every formula above would hold unchanged with $\gamma_1(i), \beta_1(i), \ldots$ depending on $i$. Today `_prepare_inputs` always broadcasts one vector per object (§2), so the option is latent.

## §7. Parameter counts: film 66,048 per block, condition encoder 17,792, compared with the block's ffn and qkv

**Where this lives / what this part does.** All numbers below follow from the constructor lines already shown (experiments/learned_intrinsic_solver/network.py:109-123 for the block, experiments/learned_intrinsic_solver/network.py:385-412 for the network) and were checked by instantiating the default network on CPU and summing `p.numel()` over each module.

How to read them: an `nn.Linear(n_in, n_out)` has $n_\text{in}\,n_\text{out}$ weights plus $n_\text{out}$ biases; an `nn.LayerNorm(D)` has $2D$ parameters (its $a^\text{LN}$ and $b^\text{LN}$).

| module | construction | count | formula |
|---|---|---|---|
| `film` (per block) | `Linear(128, 512)` | 66,048 | $128 \cdot 512 + 512$ |
| `condition_encoder` (once) | `Linear(9, 128)`, SiLU, `Linear(128, 128)` | 17,792 | $(9 \cdot 128 + 128) + (128 \cdot 128 + 128) = 1280 + 16512$ |
| `qkv` (per block) | `Linear(128, 384)` | 49,536 | $128 \cdot 384 + 384$ |
| `out_projection` | `Linear(128, 128)` | 16,512 | $128 \cdot 128 + 128$ |
| `edge_bias` and `edge_val` | `Linear(64, 4)`, `Linear(64, 128)` | 8,580 | $(64 \cdot 4 + 4) + (64 \cdot 128 + 128)$ |
| `ffn` (per block) | `Linear(128, 512)`, SiLU, `Linear(512, 128)` | 131,712 | $(128 \cdot 512 + 512) + (512 \cdot 128 + 128)$ |
| two LayerNorms | `LayerNorm(128)` twice | 512 | $2 \cdot 256$ |
| **one block, total** | | **272,900** | sum of the block rows |
| `node_encoder` | `Linear(70, 128)`, SiLU, `Linear(128, 128)` | 25,600 | $70 = 9 + 61$ input channels |
| `edge_encoder` | `Linear(24, 64)`, SiLU, `Linear(64, 64)` | 5,760 | |
| `output_norm` and the two heads | `LayerNorm(128)`, `Linear(128, 9)`, `Linear(128, 1)` | 1,546 | |
| **default network, total** | one block, no edge network, no contact encoder | **323,598** | |

Reading the table:

- The conditioning path (`condition_encoder` plus `film`) is $17{,}792 + 66{,}048 = 83{,}840$ parameters, about 26 % of the default network. Almost all of it is the FiLM generator: 512 outputs, each a full linear function of the 128-dimensional $c$.
- `film` is just over half the size of `ffn` ($66{,}048 / 131{,}712 = 0.5015$: the weight matrices are exactly half, $4D^2$ against $8D^2$, and the biases, $512$ against $512 + 128$, break the equality) and exactly $4/3$ of `qkv`. This is typical for FiLM and adaLN blocks: producing four $D$-vectors from a $D$-vector costs $4D^2$, the same order as the attention projections ($3D^2$ for `qkv` plus $D^2$ for the output).
- Compute is a different story. Because the encoder and `film` are evaluated on the expanded $[B, N, \cdot]$ tensors (§3, §6), their multiply-adds per query are $B N\,(9 \cdot 128 + 128^2 + 128 \cdot 512)$, comparable to the first half of the feed-forward network. Evaluating them once per object and broadcasting the result would remove that cost without changing any output; LeCO does it that way (§8).

## §8. Relatives: plain concatenation, adaptive LayerNorm in DiT, cross-attention, and LeCO's FiLM in core/graph.py

**Where this lives / what this part does.** This section places the project's choice among the standard ways of conditioning a transformer, so that "FiLM" stops being a label and becomes one point in a small design space. The LeCO comparison reads the sibling repository at `/home/horde/Code/Graphics/study/LearnedClothOptimizer` (commit `1aa05c1`, 2026-08-19); those references are plain text because that file is outside this repository and is not embedded here.

**Plain concatenation (or an additive input embedding).** Append $c$ to every cell's input, $x^{(0)}_i = W_\text{in}\,[\,\text{state}_i;\ c\,] + b$, or equivalently add $W_c\,c$ to the input embedding. The material then enters as a *bias* of the first layer only; any multiplicative interaction between material and geometry has to be discovered by the nonlinearities downstream. It is cheap and often adequate, and this project uses exactly this pattern for the per-cell contact summary, which is concatenated to the node input (experiments/learned_intrinsic_solver/network.py:483-486), but not for the material.

**Adaptive LayerNorm (adaLN and adaLN-Zero).** Diffusion Transformers (Peebles and Xie, "Scalable Diffusion Models with Transformers", ICCV 2023) replace LayerNorm's learned affine pair $(a^\text{LN}, b^\text{LN})$ by functions of the conditioning (timestep and class embedding):

$$ \operatorname{adaLN}(x \mid c) = \gamma(c) \odot \frac{x - \operatorname{mean}(x)}{\sqrt{\operatorname{var}(x) + \varepsilon}} + \beta(c), $$

and adaLN-Zero additionally regresses a gate $\alpha(c)$ that multiplies each residual branch, with the regression layer zero-initialised so that every block is the identity at the start. FiLM applied right after a LayerNorm, as here, is algebraically adaLN: $(1 + \gamma) \odot (a^\text{LN} \odot \hat{x} + b^\text{LN}) + \beta$ is again an affine map of the normalised $\hat{x}$ with material-dependent coefficients. The differences are that this project keeps LayerNorm's own $(a^\text{LN}, b^\text{LN})$, uses the "$1 + \gamma$" form instead of a bare $\gamma$, and has no residual gate $\alpha$.

**Cross-attention.** Treat the conditioning as a set of tokens $\{c_1, \ldots, c_M\}$ and let every cell attend to them: $x_i \leftarrow x_i + \operatorname{Attn}(q = W_q x_i,\ k = W_k c_m,\ v = W_v c_m)$. This is the right tool when the conditioning is itself a sequence (a text prompt, a set of contact partners) because each cell can pick the tokens that matter to it. For a single material vector it degenerates to a per-cell weighted copy of one token, which FiLM provides far more cheaply. (The contact tokens of this project are pooled by a separate `ContactEncoder` rather than cross-attended; that is outside this tutorial.)

**LeCO (Learned Cloth Optimizer).** The `GraphTransformerBlock` in `core/graph.py` (class at line 394) uses the same FiLM recipe with three small differences. Its generator is `self.film = nn.Linear(width, 4 * width)` (line 444), and `conditioning()` returns `self.film(condition)[node_batch].chunk(4, dim=-1)` (line 455): the generator runs once per graph on the `[graphs, width]` condition and the four chunks are indexed to nodes through `node_batch` (`[node_batch]` picks, for every node, the row of the graph it belongs to), the once-per-object evaluation §7 mentioned. The forward unpacks `attention_scale, attention_shift, mlp_scale, mlp_shift` (line 478), the same order as here, and applies `normalized = self.norm1(x) * (1 + 0.1 * attention_scale) + attention_shift` (line 479) and `normalized = self.norm2(x) * (1 + 0.1 * mlp_scale) + mlp_shift` (line 506). So LeCO computes $(1 + 0.1\,\gamma) \odot \operatorname{LN}(x) + \beta$: the scale is damped by a factor $0.1$ and the `film` layer keeps PyTorch's default initialisation (there is no `zeros_` on it in `graph.py`), whereas LIDO uses an undamped $1 + \gamma$ with zero-initialised weights. LeCO's conditioning vector comes from four raw material features passed through a `RunningNormalizer(4)` (`core/model.py` line 1003) and an encoder `Linear(4, 2 * width), SiLU, Linear(2 * width, width)` (`core/model.py` lines 1006-1008); LIDO instead fixes the normalisation analytically with the nine log and ratio channels of §1 and uses a `Linear(9, 128), SiLU, Linear(128, 128)` encoder.

## §9. Consequence for the cell-normalisation plan: five channels carry absolute scale, replacing them changes only the encoder input

**Where this lives / what this part does.** The idea note `notes/ideas/idea-normalize-cells.md` (discussion log of 2026-09-27) proposes simulating every object in normalised units ($h = 1$, $\mu = 1$, $\Delta t = 1$) so that one network serves all sizes, stiffnesses and time steps. Its correction 3 and the tests in experiments/learned_intrinsic_solver/tests/test_scaling_invariance.py:743-819 establish which conditioning channels survive that change.

The physics is scale-similar when the following dimensionless groups are held fixed (the note's table, with the objective divided by $\mu h^3$):

$$ \frac{\lambda}{\mu}, \qquad \Lambda = \frac{\rho\,h^2}{\mu\,\Delta t^2}, \qquad \Gamma = \frac{g\,\Delta t^2}{h}, \qquad \Xi = \frac{\eta}{\mu\,\Delta t}, \qquad \kappa = \frac{k_e}{E\,h}, \quad \beta = \frac{k_d}{k_e\,\Delta t}, \quad \mu_f . $$

Compare with the nine channels of §1. Channels 6-9 are exactly $\log(1 + \Xi)$, $\log(1 + \kappa)$, $\beta$ and $\mu_f$: they coincide between a physical scene and its normalised copy. Channels 1-5 do not: $\log(1 + \lambda/10^5)$, $\log(1 + \mu/10^5)$, $\log(\rho/1000)$, $\log(h/0.025)$ and $\log(60\,\Delta t)$ each carry an absolute scale in pascals, kilograms, metres or seconds. The test pins this down by building each scene twice and comparing the channel tensors:

```python
        conditioning = channels(scenes, CELL_SIZE, TIME_STEP)
        unit_conditioning = channels(units, 1.0, 1.0)
        dimensionless = {"log1p_damping", "log1p_contact_kappa", "contact_beta", "contact_mu"}
        differing = set()
        for column, name in enumerate(features.CONDITIONING_CHANNELS):
            same = torch.allclose(conditioning[:, column], unit_conditioning[:, column], rtol=1e-9, atol=1e-12)
            if name in dimensionless:
                self.assertTrue(same, f"{name} is dimensionless and must coincide")
            elif not same:
                differing.add(name)
        print(f"conditioning channels that carry an absolute scale today: {sorted(differing)}")
        # Not part of the law: this records the current behaviour of features.conditioning_channels, whose
        # remaining channels are log1p(lambda / 1e5), log1p(mu / 1e5), log(rho / 1000), log(h / 0.025) and
        # log(60 dt). They are the to-do list of the note's implementation sketch (replace them by the groups
        # lambda / mu, Lambda, Gamma and Xi); this assertion flips, and should be deleted, when that lands.
        self.assertEqual(differing, set(features.CONDITIONING_CHANNELS) - dimensionless)
```
(`experiments/learned_intrinsic_solver/tests/test_scaling_invariance.py:804-819`, inside test_energy_floor_and_conditioning_channels)

The assertion at line 819 records that *exactly* the five absolute channels differ today, and its comment says it should flip, and be deleted, when the replacement lands. The replacement the note asks for is to feed $\lambda/\mu$, $\Lambda$ and $\Gamma$ (as logs, since $\Lambda$ alone spans about five decades over the training ranges) instead of channels 1-5, keeping $\Xi$, $\kappa$, $\beta$, $\mu_f$; gravity would then enter the conditioning for the first time, because $\Gamma$ contains $g$.

**What this changes in the network, and what it does not.** Everything in this tutorial from §3 onward is independent of *which* nine (or seven) numbers arrive:

- `conditioning_channels` (§1) and the tuple `CONDITIONING_CHANNELS` change; `CONDITIONING_DIM` becomes the new count, and `FEATURE_SCHEMA_VERSION` (experiments/learned_intrinsic_solver/features.py:109) must be bumped because old checkpoints assume nine inputs.
- The first layer of `condition_encoder`, `Linear(CONDITIONING_DIM, 128)` (experiments/learned_intrinsic_solver/network.py:392), gets a new `in_features`: that is the *only* weight tensor whose shape changes, $128 \times 9 \to 128 \times 7$.
- $c \in \mathbb{R}^{128}$, every `film` layer, the block equations of §5 and the parameter counts of §7 (except the encoder's first 1,280 entries) are untouched. The material still reaches the block as a per-channel gain and bias; it is merely described in scale-free coordinates.
- The other absolute scale the tests found, the `log_gradient_rms` state scalar from the big picture (experiments/learned_intrinsic_solver/features.py:78), is a state feature and not part of the conditioning path; the note treats it separately.

---

## Check your understanding

**1. Why do channels 1, 2, 6 and 7 use $\log(1 + x)$ while channel 3 uses $\log x$?**

Because $\lambda$, $\eta$ and $k_e$ can legitimately be zero ($\lambda = 0$ is a valid Lamé pair, $\eta = 0$ is the default, $k_e = 0$ means no contact), and $\log 0 = -\infty$ would poison the encoder; $\log(1 + 0) = 0$ keeps those cases finite and makes "absent" coincide with "reference". Density is strictly positive, so a plain log is safe and gives a symmetric spread ($\pm 2.30$ over the training range).

**2. A freshly constructed block has `film.weight = film.bias = 0`. What does it compute, and why does training still move those weights?**

With $\gamma = \beta = 0$, $\tilde{x} = (1 + 0) \odot \operatorname{LN}(x) + 0 = \operatorname{LN}(x)$: an ordinary pre-norm transformer block that ignores the material. The gradient with respect to $W_f$ is the outer product of the downstream gradient on $[\gamma; \beta]$ with $c$, which is nonzero whenever $c \neq 0$, so the first optimizer step already makes $W_f \neq 0$.

**3. $\gamma_1$ is the same for all 4000 cells of an object. How can the material then change the update of one cell differently from another?**

FiLM multiplies $\gamma_1$ into $\operatorname{LN}_1(x_i)$, which is cell-dependent, so the same gain amplifies different channels by different amounts in different cells; and the modulated tokens feed attention, which mixes cells. The material sets the rules (which channels count, and by how much); the cell's own state decides what those rules produce.

**4. Which of the 512 outputs of `film` shift the input of the feed-forward branch, and where in the code?**

Output channels 384-511, that is `modulation[3]` from `chunk(4, dim=-1)` (experiments/learned_intrinsic_solver/network.py:224), added at experiments/learned_intrinsic_solver/network.py:260 after the multiplication by `1 + modulation[2]` (channels 256-383).

**5. If the five absolute channels are replaced by $\lambda/\mu$, $\log \Lambda$ and $\log \Gamma$, which tensors in a saved checkpoint stop matching?**

Only `condition_encoder.0.weight` (shape $128 \times 9 \to 128 \times 7$); `condition_encoder.0.bias`, `condition_encoder.2.*`, every `layers.k.film.*` and everything else keep their shapes. The schema version must still be bumped, because equal shapes elsewhere do not make the old weights meaningful for new inputs.

## Test coverage of the conditioning path

**Where this lives / what this part does.** A case-insensitive grep for `conditioning` or `film` hits 20 of the test files under `experiments/learned_intrinsic_solver/tests/`. The table lists the tests among them that assert something about the conditioning path; the other hits (for example in test_gpu_solver.py, test_optimizer.py and test_damping_training.py) only pass `conditioning_dim=features.CONDITIONING_DIM` to a constructor or mention the word in passing, and are left out. They run on CPU with the project's virtual environment.

| test | what it checks about the conditioning path |
|---|---|
| `test_dimensions_and_order` (experiments/learned_intrinsic_solver/tests/test_features.py:34) | `CONDITIONING_DIM == 9` and the exact channel order |
| `TestConditioningChannels.test_channel_formulas` (experiments/learned_intrinsic_solver/tests/test_features.py:305) | the nine formulas against hand values, zero contact channels when contact is omitted, $\log(h/0.025)$ and $\log(60\,\Delta t)$ for a non-canonical grid |
| `test_contact_channels_and_ratios` (experiments/learned_intrinsic_solver/tests/test_features.py:339) | $\kappa$, $\beta$ (exactly zero where $k_e = 0$) and $\mu_f$; channels 7-9 equal `contact_ratios` |
| `test_rejects_invalid_inputs` (experiments/learned_intrinsic_solver/tests/test_features.py:371) | invalid $h$, $\Delta t$, shapes and dtypes raise |
| `test_conditioning_and_backpropagation` (experiments/learned_intrinsic_solver/tests/test_network.py:81) | gradients reach the conditioning input and every FiLM parameter; changing $c$ changes the output |
| `test_default_one_layer_radius_one_and_training_config` (experiments/learned_intrinsic_solver/tests/test_network.py:162) | `condition_encoder[0].in_features == 9` |
| `test_default_network_uses_revised_schema` (experiments/learned_intrinsic_solver/tests/test_newton_solver.py:82) | the Newton solver builds its network with `conditioning_dim == CONDITIONING_DIM`, and a network built with `conditioning_dim=5` is rejected as a legacy schema by `prepare_problem`; the shape change of §9 would be caught the same way |
| `test_solver_network_gradients` (experiments/learned_intrinsic_solver/tests/test_network.py:233) | with perturbed `film.weight`, `conditioning.grad` is nonzero through the full network |
| `test_zero_initialized_update_matches_baseline_exactly` (experiments/learned_intrinsic_solver/tests/test_network_edge_update.py:96) | a FiLM-modulated layer with and without the edge network is bit-identical at init, attention weights included |
| `test_checkpoint_chunks_matches_forward_and_gradients` (experiments/learned_intrinsic_solver/tests/test_network_edge_update.py:285) | conditioning gradients identical with and without chunk checkpointing |
| `test_per_cell_lame_conditioning_and_fusion_stiffness` (experiments/learned_intrinsic_solver/tests/test_solver_step.py:126) | exact nine channel values for two materials including $\lambda = 0$, and channel 6 with damping |
| `test_loss_backpropagates_through_fusion_and_attention` (experiments/learned_intrinsic_solver/tests/test_solver_step.py:437) | the physical loss gradient reaches `layers.0.film.weight` and `condition_encoder.0.weight` |
| `test_state_features_follow_leco_normalization_and_history_contract` (experiments/learned_intrinsic_solver/tests/test_mixed_physics.py:404) | `prepare_inputs` returns conditioning of shape $[B, N, 9]$ |
| `test_contact_free_context_reproduces_contact_less_objective` (experiments/learned_intrinsic_solver/tests/test_mixed_physics.py:857) | channels 7-9 are exactly zero for contact-free contexts |
| `test_penetrating_floor_raises_energy_and_pushes_bottom_corners_upward` (experiments/learned_intrinsic_solver/tests/test_mixed_physics.py:996) | channels 7-9 carry $\log(1 + \kappa)$, $\beta$ and $\mu_f$ of the registered contact |
| `test_physical_axis_change_block_and_damping_conditioning` (experiments/learned_intrinsic_solver/tests/test_mixed_damping.py:77) | channel 6 equals $\log(1 + \eta/(\mu\,\Delta t))$ and channels 1-5 are unaffected by damping |
| `test_damping_enters_energy_conditioning_and_physical_change_block` (experiments/learned_intrinsic_solver/tests/test_damping_solver.py:69) | the same for the single-material step |
| `test_energy_floor_and_conditioning_channels` (experiments/learned_intrinsic_solver/tests/test_scaling_invariance.py:743) | exactly four of the nine channels are invariant under the normalisation of §9 |
| `test_network_inputs_are_invariant_except_log_gradient_rms` (experiments/learned_intrinsic_solver/tests/test_scaling_invariance.py:821) | the same through `prepare_inputs`, plus the `log_gradient_rms` offset |
| `test_contact_scenes_follow_the_seeded_generator_and_the_sampled_material` (experiments/learned_intrinsic_solver/tests/test_train_mixed.py:497) | the training payload's $\kappa$, $\beta$, $\mu_f$ match the step's channels 7-9 |
| `test_contact_disabled_reproduces_the_schema_four_zero_path` (experiments/learned_intrinsic_solver/tests/test_train_mixed.py:672) | channels 7-9 are zero when contact is disabled in training |

Relied on implicitly rather than asserted: no test checks the FiLM zero-initialisation directly (`film.weight == 0` after construction) or the chunk order (that `modulation[0]` scales rather than shifts); both are exercised only indirectly, through the "fresh layers agree" comparisons and the gradient tests above.
