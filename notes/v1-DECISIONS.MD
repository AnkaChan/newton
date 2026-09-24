# V1 Technical Decisions: Learned Intrinsic Solver

Recorded from the design conversation through 2026-09-24. Early sections preserve the original proposals and open questions; the later implementation sections supersede their status claims. The current design, measured results, and next training configuration are consolidated in sections 22–24. This is an experimental solver, without a general convergence or contact guarantee. The short overview is [v1-PLAN.MD](v1-PLAN.MD).

**Latest clarification:** the global rigid pose target is used only in fusion. The physical implicit-Euler inertia, free-motion predictor, inertial-offset inputs, and physical gradients remain unchanged. Fusion constructs a candidate configuration; it does not replace the physical time-step objective.

**Status language:** “Chosen” identifies the current design direction accepted in the conversation. “Candidate” identifies an option discussed for evaluation. “Open” identifies a detail that still needs a precise definition or experiment. Numerical examples and formulas below explain the proposals; they do not imply implementation approval or experimental success.

## 1. Workspace and scope

The worktree is `/home/horde/Code/Graphics/newton-working-copies/learned-instrinic-solver`, on branch `ankac/learned-instrinic-solver`. It was created from the official `newton-physics/newton` upstream, initially at commit `f7a7f6f15c55fb3e889cc46d35a7a66cc693b69b`. The project spelling `learned-instrinic-solver` is intentional in these paths.

The initial scope is three-dimensional deformable solids represented on a material grid. Build a proof of concept before arbitrary boundaries, elaborate contact, or a large architecture search. This record uses the conversation and the specifically authorized references listed at the end; unrelated project notes are outside its source scope.

## 2. Material grid, graph, and control points

**Chosen:**

- Begin with actual voxel solids. A rest cell is full or empty; the physical surface follows voxel faces.
- One occupied cell is one graph node. Shared grid corners are control points for reconstruction, not graph nodes.
- Graph connectivity and shared degrees of freedom follow material connectivity. Spatially close disconnected pieces must not become glued together.
- Keep the reference grid attached to the material. Do not regenerate it from a fixed world grid each timestep.
- Use the same network for interior and boundary cells. Six exposed-face flags distinguish geometric surface faces.
- Keep geometric boundaries separate from physical conditions such as clamping.

Face and corner flags use fixed material-grid labels. Rotating the physical problem changes their world directions, not their ordering in the input.

A cell has eight corners. In an ordinary connected interior voxel grid, a face is shared by two cells, an edge by four, and a corner by eight. Shared coordinates are what make local predictions assemble into one compatible object. Coordinate coincidence alone must not create shared support between disconnected pieces.

## 3. Three-axis representation

**Chosen:** use the transformed unit material x, y, and z directions as three columns of a deformation matrix `F`. There are nine scalar components, not three scalar degrees of freedom.

```text
deformed_material_offset = F @ rest_material_offset
```

Adding the appropriate position supplies translation. Axis lengths encode stretch, angles encode shear, and orientation encodes rotation. Do not force the deformed axes to be orthogonal or unit length.

Normalization removes rest dimensions, not stretch:

```text
rest edge length = 2
deformed edge length = 3
deformation-axis length = 3 / 2 = 1.5
```

Rest size and shape remain separate, explicit information. Normalized directions plus three separate current lengths would contain equivalent information, but a single rest-size number cannot replace current stretch.

One matrix describes an affine deformation within a cell: a cube becomes a parallelepiped. Different matrices across cells describe object-scale bending and twisting. One matrix cannot describe every possible warped configuration of eight independent corners.

The network keeps the three-axis representation. The proposed addition of eight current corner offsets as a separate shape input was withdrawn. Eight corner positions remain the reconstruction unknowns. Eight-corner force/history vectors are different kinds of inputs and were not withdrawn by this decision.

## 4. Cell frames and equivariance

**Chosen:** use common rest-material axes within an object. For a valid, non-collapsed, non-inverted cell, use polar decomposition:

```text
F = R @ U
current_local_axes = R.T @ F
```

`R` is the cell's current local-to-world rotation. The cell's own local matrix is `U`, which is symmetric. It therefore has six independent values stored in nine entries. Keep nine entries for a direct, shared input/output representation.

The output uses this same current frame, held fixed for the evaluation. It may include a new rotation relative to that frame, so the output is not restricted to symmetric stretch. Do not remove its rotation by polar-decomposing the output.

**Chosen on 2026-09-24:** recompute the cell frames from the current candidate geometry before each learned update, then keep them fixed during that network evaluation and its reconstruction. Stop gradients through the polar frame extraction. This does not mean keeping the initial frames for the entire simulation. Gradients through predicted targets, fusion, and the physical loss remain part of the training design; this decision does not detach every other geometric quantity automatically. For several unrolled updates, omitting the dependence of later frames on earlier predictions gives an approximate gradient, which is accepted for this version.

**Chosen implementation direction:** use Warp for forward cell-frame extraction, with independent cells processed in parallel and data kept on the simulation device. Keep the existing NumPy calculation as a correctness reference. This is an implementation choice, not a measured speed claim; a Warp kernel and parity/performance checks remain to be implemented. A custom backward through the polar decomposition is not required for this version.

**Chosen numerical precision:** use float32 for the working shape, frame/axis extraction, and reconstruction solve. The requested 100-sample round-trip study must use this precision in its actual solver iterates, not only cast a float64 result afterward. Earlier float64 experiments remain reference results. Offline diagnostic accumulation may use float64 when explicitly identified; it does not change the precision of the tested pipeline.

For a message from cell `j` to cell `i`:

```text
relative_frame = R[i].T @ R[j]
neighbor_axes_in_receiver_frame =
    relative_frame @ neighbor_axes_in_its_own_frame
```

Only the spatial side is transformed because these axes map a shared rest basis into spatial vectors. A spatial stress or strain tensor would require a different, two-sided conversion. Different reference bases would also need a reference-side conversion; that complexity is avoided in V1.

Relative positions, inertial offsets, gradients, and updates are vectors: express them in the receiving cell's current frame. Re-express historical vectors from their stored physical frame into the current one; do not reuse components from an obsolete frame.

Under a global rotation, cell frames rotate with the object and local components remain unchanged. Translation is removed from local relative quantities. This requires transforming the whole physical problem, including forces, motion, constraints, and obstacles. Independent arbitrary relabeling of each local frame and reflection equivariance are stronger properties that have not been established. Collapsed and inverted cells need a separate handling rule.

## 5. Working input contract

The following captures the discussed interface. The rigid target affects fusion; it does not redefine the physical state inputs. Optional features remain provisional where marked.

| Block | Representation | Status and meaning |
|---|---|---|
| Current cell shape | `3 x 3` deformation axes | Chosen; current co-rotating frame |
| Rest size | Physical rest edge length `h` | Required explicitly for cubic V1 |
| Rest shape | Known cube reference geometry | Required conceptually; exact stored descriptor is open |
| Exposed faces | 6 flags | Chosen geometric boundary descriptor |
| Inertial offset | `8 x 3` vectors | Chosen; unchanged free inertial prediction minus current candidate position |
| Current physical-energy gradient | `8 x 3` corner vectors | Candidate learned-optimizer feature |
| Previous physical-energy gradient | `8 x 3` corner vectors | Candidate history feature |
| Previous accepted update | `8 x 3` corner vectors | Candidate history feature; actual accepted displacement |
| Fixed-corner conditions | 8 flags | Proposed prototype descriptor; prescribed motion needs more data |
| Material conditioning | Lamé lambda, Lamé mu, density | Direct per-cell isotropic material inputs |
| Size/time conditioning | Rest length and timestep | Chosen conditioning information |
| Magnitude/history conditioning | Gradient scale, update scale, previous log step | Candidate optimizer scalars |

An earlier explicit rest descriptor was `rest_corner_offsets[8,3] / h` plus `h`. For cubic voxels those normalized corners are a fixed pattern with coordinates `+/- 0.5`. It can be retained as a fixed descriptor or represented implicitly; do not confuse its redundancy with permission to omit the physical rest size. The final packed rest-geometry encoding is not frozen.

Lamé mu is the shear modulus. Lamé lambda controls the additional resistance to volume change in the chosen compressible Neo-Hookean law. Both have pressure units; density is mass per rest volume.

Candidate edge inputs are rest-relative cell-center offset (3), current relative offset in the receiver frame (3), and relative frame (9). Messages carry neighboring features in the receiver's frame.

Uniform gravity and explicit velocity channels were replaced in the initial elastic-network proposal by inertial offsets:

```text
Y = previous_physical_position
    + dt * previous_velocity
    + dt^2 * external_acceleration

inertial_offset = Y - current_candidate_position
```

This is a vector, never a scalar “distance.” Keep one per corner rather than only a center offset, so distributed motion is visible. The simulator still retains velocities and forces. Gravity included in a predictor must not be counted again as the same force contribution.

**Confirmed after the rigid-target clarification:** retain this original physical predictor, fixed during a physical timestep's optimizer iterations. Rotate `Y - current_candidate_position` into the receiving cell's current frame for network input; this coordinate conversion does not remove rigid-motion components. Candidate gradient inputs remain gradients of the unchanged physical objective, including its full inertia term. There is no requirement to replace these inputs with an internal-only predictor or to project out physical rigid-motion gradients. Friction, damping, and moving contacts may still require additional velocity information.

For candidate normalization, scale length offsets by rest length. Use one rotation-invariant object-level RMS gradient scale, and scale current and previous gradients using the same current scale. Preserve log magnitudes as scalar features. Previous accepted updates can likewise use an object-level scale plus its log magnitude. Exact floors, clipping, startup/reset values, and history persistence are open.

## 6. Graph transformer and FiLM

**Chosen architectural direction:** a graph transformer over material-connected cell tokens. The baseline uses three local attention blocks with hop schedule `1, 1, 1`. Each block attends to the cell itself and all cells sharing a face, edge, or corner. The baseline has no context jumping. Width 128 and four heads are configurable prototype defaults; the number of repeated solver updates remains open.

**Chosen attention design:** use a learned `edge_bias` added to each head's attention score and a learned `edge_val` added to the neighbor's value before weighted aggregation. Both come from encoded geometric edge features. Do not add a fixed geometric prior to the attention scores: no prescribed face-area, distance, or other geometric weighting. Geometry remains available as an input to the learned edge encoder.

**Chosen connectivity layout:** exploit the canonical grid with fixed neighbor slots and a validity mask for each slot. Each baseline block has 26 neighboring slots plus self, totaling 27. Boundary cells retain these slots but mask nonexistent edges. Set invalid attention logits to negative infinity before softmax so invalid slots receive exactly zero weight; zeroing their feature vectors alone is insufficient. Keep self valid for each active cell and use safe gather indices for invalid slots. Precompute the topology and masks and reuse them across network evaluations. This topological mask is separate from the excluded geometric attention prior; material connectivity must still prevent links between disconnected pieces. A general full-cuboid utility can construct larger exact Chebyshev-distance shells, but these are not used by the baseline.

**Chosen conditioning idea:** FiLM. A shared material/conditioning encoder produces a gain and bias for each hidden feature:

```text
conditioned_feature = gain * hidden_feature + bias
```

Use material, size, and timestep information to condition shared blocks rather than assigning a different network to every material or boundary type. These scalar conditions are compatible with the invariant local features. FiLM does not replace the actual material parameters in the physical energy.

The baseline shares its geometric edge encoder across the three local blocks; each block has its own learned edge-bias and edge-value projections. Wider neighborhoods and context jumping remain outside the baseline.

## 7. Targets and optimizer-style outputs

**Chosen:** outputs are preferred local deformation targets, analogous in role to local Projective Dynamics goals. They are not guaranteed next-step cell deformations; assembly must reconcile incompatible requests.

**Working output-head proposal:** predict a bounded nine-component correction per cell and a shared dimensionless deformation step per object:

```text
local_target_axes =
    current_local_axes + shared_deformation_step * local_correction
```

A candidate bounded correction is:

```text
local_correction = raw_correction / sqrt(1 + sum(raw_correction^2))
```

The norm here covers the nine correction components. It does not normalize each target axis to unit length. The correction is bounded, not necessarily a unit vector.

A second head can predict a change in the object's log step, yielding a positive shared scale. This borrows the direction/scale separation idea without copying length-valued displacement limits onto dimensionless deformation updates. A per-object scale is the starting proposal; per-cell step scales are not selected.

Neither bounded outputs nor a learned step guarantee descent, compatibility, non-inversion, or stability. Acceptance/backtracking against the actual physical objective is a candidate. Record the actual accepted update after any projection or line search.

**Implemented baseline on 2026-09-24:** incremental hex fusion fits the change in the local axes and adds a shared-corner displacement to the current shape. Zero network correction is an exact no-op when the current prescribed corners are satisfied and no rigid map is applied, including for warped cells. A changed rigid target or prescribed boundary can still move the candidate. The rigid predictor alone does not commit a physical state update.

## 8. Unchanged physical inertia; rigid pose guidance only in fusion

**Chosen:** keep the physical implicit-Euler problem unchanged. Separate the construction of an overall rigid pose target from the learned local deformation targets, then combine them in fusion.

For the earlier explicit-force predictor `Y`, the physical inertia is still:

```text
inertia_energy(X) = 0.5 * sum_over_corners(
    mass[a] / dt^2 * squared_length(X[a] - Y[a])
)

E_implicit(X) =
    inertia_energy(X) + elastic_energy(X) + other_physical_terms(X)
```

Keep the chosen physical contact model and any external-force accounting in this physical problem. Do not replace `Y` with the rigid target, change the mass weights, or remove the rigid components of this inertia. Contact behavior remains a separate modeling choice; generating a rigid target does not supply the entire physical contact law.

Fusion is a candidate-generation layer:

```text
learned local axes + global rigid pose target
    -> fusion
    -> candidate world configuration X
    -> original implicit-Euler objective / chosen training or acceptance rule
```

The precise training and acceptance rules remain open. A rigid-target predictor must not advance the physical state a second time or double-count external/contact impulses.

One fusion representation separates overall pose and internal shape:

```text
world_corner = translation + object_rotation @ internal_corner
```

The body origin is the mass center. A consistent convention must also separate overall rotation from internal shape; otherwise fusion can undo or duplicate its rigid pose guidance. Hard versus soft enforcement of the pose target is open. The target must not silently become a permanent restriction on the physical solution space.

The translational predictor has the form:

```text
predicted_velocity = old_velocity
    + dt * total_external_force / total_mass
    + total_contact_impulse / total_mass

translation_target = old_translation + dt * predicted_velocity
```

Within the predictor, estimate angular momentum from external torques and contact angular impulses, then predict orientation with the rigid integrator. These estimates supply the fusion target, not an independent committed trajectory. The precise predictor and impact law are open.

Under the working interpretation of the axes as internal deformation targets, transport them with the proposed rigid motion. Let `Q_input` be the object rotation when the cell frame `R_cell` and prediction `A_cell` were evaluated. A candidate fusion that enforces the pose target exactly would use:

```text
body_target_axes = Q_input.T @ R_cell @ A_cell

candidate_world_corner =
    translation_target + Q_target @ reconstructed_internal_corner
```

For the first predictor, `Q_input` is the old object rotation. Applying `Q_target` after this conversion intentionally carries the internal target with the predicted rigid motion; it is not merely a coordinate rewrite of an unchanged world target. If a target is instead meant to retain its absolute world orientation, its coordinates in the target body frame use `Q_target.T`. Choose the target interpretation consistently with training and refresh the input-state reference when iterating.

Under the internal-target interpretation, training targets must avoid counting the rigid motion twice. This does not remove any contribution from the physical inertia or its gradients.

**Open rotational convention:** zero mass-weighted internal translation is straightforward. For small corrections about internal positions `u0`, the linear conditions below remove infinitesimal rigid modes:

```text
sum(mass * delta_u) = 0
sum(mass * cross(u0, delta_u)) = 0
```

These are a candidate local convention, not an exact finite-rotation alignment rule. Large deformation needs a consistent finite convention and treatment of ambiguous alignments. Re-aligning a reconstructed shape must not silently change its pose, targets, or momentum.

Body coordinates do not change the physical inertia. If it is evaluated through the representation above, its residual is still `translation + object_rotation @ internal_corner - Y` at each corner. Any auxiliary reference or regularizer used to construct a fused proposal is a separate numerical choice, not a replacement physical inertia.

## 9. Direct axes fitting and global reconstruction

**Chosen fitting representation:** compute deformation axes from shared corner positions and minimize their difference from the proposed axes. The earlier edge-fitting explanation is an alternative, not the selected primary formulation.

For a rest cube of edge length `h`, the deformation gradient at its center has columns:

```text
x_axis = (average_of_positive_x_face - average_of_negative_x_face) / h
y_axis = (average_of_positive_y_face - average_of_negative_y_face) / h
z_axis = (average_of_positive_z_face - average_of_negative_z_face) / h
```

Each face average uses its four corners. In the split formulation, both computed and target axes are expressed in body coordinates.

```text
cell_error =
    squared_length(computed_x_axis - target_x_axis)
  + squared_length(computed_y_axis - target_y_axis)
  + squared_length(computed_z_axis - target_z_axis)
```

Shared corners automatically couple the cell requests. There can be more requested axis components than independent compatible corner motions; fitting resolves this intentionally overdetermined system.

Center-only fitting misses some warped corner motions. For example, alternate positive and negative corner displacements can cancel on every face average while changing the cell. A quadratic regularizer can make a fusion solve unique without supplying the missing deformation stiffness.

**Candidate numerical default:** evaluate axes at eight interior integration points (a `2 x 2 x 2` quadrature rule), comparing each to the same per-cell target. The network still outputs nine values per cell. Full quadrature or another justified stabilization is needed before relying on eight-corner reconstruction; the exact choice is not yet implemented.

The global deformation between solved corners is trilinear interpolation. For a rest point with voxel coordinates `u, v, w` between zero and one:

```text
wu = [1-u, u]
wv = [1-v, v]
ww = [1-w, w]

internal_position = sum over i,j,k in {0,1}:
    wu[i] * wv[j] * ww[k] * internal_corner[i,j,k]
```

Place this internal shape using the pose selected by fusion. Shared face corners give continuous interpolation across connected voxel faces. Compatibility does not guarantee non-inversion or contact safety; the resulting candidate is still evaluated against the original physical problem.

For completeness, the discussed edge alternative fits each of twelve cell edges to the target matrix times its rest edge vector. A shared edge can carry four different cell requests. They can be combined into a weighted average target with summed weights. Edge fitting and integrated gradient fitting penalize non-affine motions differently and should not be treated as identical discretizations.

## 10. Fitting weights

The proposed physical scaling is:

```text
fitting_weight = reconstruction_stiffness * rest_volume
```

Rest volume makes an equally deformed material region contribute approximately the same total fitting energy after grid refinement. Axes error is dimensionless; reconstruction stiffness with energy-per-volume units gives an energy-valued penalty.

Reconstruction stiffness controls target-fitting strength. It is not automatically Young's modulus, and one scalar squared-axis penalty is not a complete elastic constitutive law. Material behavior can enter the physical objective and the learned targets separately.

For identical voxels and a uniform coefficient, absorb the product into a single `target_weight`. A common factor does not change a pure fitting minimizer, but it matters relative to pose-fitting or other auxiliary fusion penalties. With quadrature, include the integration weights and count cell volume once. These weights do not rescale the physical implicit-Euler inertia.

## 11. Conditional quadratic solve and differentiation

The following is a candidate quadratic fusion layer, conditional on fixed linear pose/boundary constraints and an optional auxiliary reference. It is distinct from `E_implicit` in section 8 and does not settle hard versus soft pose guidance, finite rotation, or contact.

Flatten the unknown internal corner coordinates into `u`. Let:

| Symbol | Meaning |
|---|---|
| `G` | Fixed linear map from corner coordinates to sampled deformation-axis components |
| `a` | Stacked proposed body-frame axis components, repeated at integration points |
| `W` | Positive diagonal fitting weights, including volume and quadrature |
| `B` | Optional symmetric positive-semidefinite auxiliary fusion regularizer, possibly zero |
| `u_ref` | Optional reference used by that fusion regularizer; not the physical predictor `Y` |
| `C u = d` | Fixed linear conditions removing rigid modes or imposing compatible constraints |

With no factor of one-half in the fitting term:

```text
E_fuse(u) =
    (G @ u - a).T @ W @ (G @ u - a)
  + 0.5 * (u - u_ref).T @ B @ (u - u_ref)

gradient_u =
    2 * G.T @ W @ (G @ u - a)
  + B @ (u - u_ref)

K   = 2 * G.T @ W @ G + B
rhs = 2 * G.T @ W @ a + B @ u_ref
```

Without constraints solve `K @ u = rhs`, if nonsingular. With linear constraints:

```text
[ K    C.T ] [ u      ] = [ rhs ]
[ C     0  ] [ lambda ]   [ d   ]
```

This is a linear constrained system. A unique constrained minimum requires independent constraint rows and a `K` that is positive definite on allowed motions satisfying `C @ delta_u = 0`. A suitable gauge removes null modes; these conditions still need checking for the chosen discretization. Frames, targets, fitting weights, and constraint coefficients are fixed during this solve. A changing finite rotational gauge or general contact changes this problem.

The elementary chain rule is also direct. For one center axis with residual `r` and weight `w`, its energy gradient is `2*w*r`. Each corner contributes through a coefficient `+1/(4*h)` or `-1/(4*h)`. Sum the contributions of all axes and cells sharing that corner.

For training, suppose `L` depends on reconstructed world positions and its gradient has been propagated into `u` through the pose mapping. If `L` uses the physical objective, that gradient includes the unchanged implicit-Euler inertia. With the constrained fusion matrix above denoted `S`, solve:

```text
S.T @ [ z ; eta ] = [ gradient_u_L ; 0 ]
gradient_a_L = 2 * W @ G @ z
```

Thus backward propagation needs one adjoint linear solve and can reuse a factorization when the matrix and constraints are unchanged. Sum repeated integration-point contributions before mapping the target gradient back to each cell. For `body_target = Q_input.T @ R_cell @ A_cell`:

```text
gradient_A_cell =
    R_cell.T @ Q_input @ gradient_body_target
```

These expressions assume fixed input state, weights, and constraints for this call and no direct dependence of `L` on `a`. Add direct target derivatives or derivatives through changing matrices, constraints, rigid pose, or earlier iterations when those are part of the training graph. The chosen V1 frame extraction is excluded from that graph as described in Section 4; including its derivative would be a later change to the training design. Fixed active contact sets can sometimes use a similar adjoint; contact-status changes can be nonsmooth.

The fitting energy defines the fusion layer. It is not the physical implicit-Euler objective or, by itself, a sufficient training objective for correct physical behavior.

## 12. Deformation-dependent inertia and momentum

**Chosen requirement:** update the rigid predictor's inertia tensor from the deformed mass distribution. Total mass stays constant; its distribution and hence the tensor change as the shape deforms. The tensor is distinct from the mass weights in the physical implicit-Euler inertia, which remain unchanged.

For a lumped-mass discretization with centered internal positions `u`:

```text
I_body = sum over mass points:
    mass * (dot(u,u) * identity - outer(u,u))

I_world = Q @ I_body @ Q.T
```

Using corner masses defines a discrete approximation; the mass/quadrature choice must be consistent across dynamics and reconstruction.

When constructing the rigid target, use angular momentum rather than assuming unchanged angular velocity after a shape update. In consistent world coordinates:

```text
angular_velocity = solve(
    I_world,
    total_angular_momentum - internal_angular_momentum
)
```

Internal motion can carry angular momentum. Updating only the tensor is therefore insufficient. Use the current shape for the rigid-target predictor, refresh its tensor after reconstruction, and evaluate whether correcting the target helps fusion. The physical state is advanced through the implicit-Euler solver, not independently by this predictor. Exact momentum, energy, and frame bookkeeping remains open; fusion is not automatically conservative.

Separating stages does not remove centrifugal effects or physical coupling between angular motion and shape.

## 13. Contact and pose/deformation feedback

The earlier fusion explanation omitted contact. Finite repulsion penalties can permit penetration. A joint constrained fusion was discussed, but the user chose a separate rigid pose target as fusion guidance. This choice does not remove or replace the original implicit-Euler inertia or establish the full physical contact model.

In the current direction, estimated contact impulses can inform the rigid target. Physical contact response still belongs to the original physical problem, and local deformation needs appropriate contact information. A rigid prediction assumes a shape; reconstruction can change that shape, create another contact, or change the required impulse.

**Candidate split loop:**

```text
rigid pose target + learned deformation targets
    -> fuse a candidate configuration
    -> update inertia and contact geometry
    -> correct targets if needed
    -> evaluate the candidate using the original physical objective
```

The number and stopping rule of corrections are open. These are candidate-generation operations inside the solver, not separate physical time advances.

A useful check is a soft block hitting a floor. Its center of mass can continue descending as it compresses. A rigid target may stop that center too early; enforcing it permanently would restrict the physical solution incorrectly. Soft pose guidance, refreshed targets, or correction strategies remain options for fusion.

Nonpenetrating positions alone do not establish the intended impact velocity, restitution, or friction. General motion also needs collision detection that can catch crossings during a proposed update. Contact handling, velocity response, moving obstacles, self-contact, and differentiation through contact remain open; no collision guarantee has been established.

## 14. Future non-grid-aligned boundaries

**Deferred until after the actual-voxel proof of concept.** Both approaches keep one network for boundary and interior cells.

1. Subgrid occupancy fractions, for example a `4 x 4 x 4` pattern. This approximates the material distribution and need not align exactly with the surface. Exposed-face or neighboring occupancy/connectivity information may still be needed.
2. Actual surface samples carrying rest-relative position (3), outward normal (3), and represented area (1). Sample count may vary by cell.

**Agreed future experiment:** a small transformer reads the variable sample set and a learned summary token, producing a fixed-size geometry feature for the main graph transformer. Compare it with a shared point MLP followed by area-weighted pooling. Larger feature dimension does not by itself require a transformer; its motivation is interaction between samples before compression.

The encoder should ignore arbitrary sample order, use padding masks where needed, account for represented area, and define behavior for interior cells with no surface samples. Pooling can include total area and a surface-present flag. Static geometry features can be cached at inference with a fixed trained encoder.

These descriptors do not replace accurate occupied-volume integration, actual surface/contact geometry, or material connectivity. Grid corners outside the material can act as interpolation handles. Disconnected material fragments may require separate basis support even when they share background-grid coordinates.

Tiny occupied fragments can also cause poor conditioning. Aggregation or stabilization is a future option, but it must preserve connectivity rather than glue material across a gap.

## 15. Training and optimization ideas to evaluate

The following records ideas borrowed from the two supplied LCO summaries. FiLM is chosen; the other techniques remain candidates rather than adopted V1 requirements:

- Use an iterative learned optimizer within each physical timestep. Distinguish inner optimizer updates from physical velocities.
- Provide current/previous physical-energy gradients and previous accepted updates. Current geometry, frames, and gradients refresh as the iterate changes.
- Use rotation-invariant normalization with retained log scales and a recurrent shared log step.
- Use the chosen FiLM conditioning; test geometric attention biases and multi-hop neighborhoods as candidates.
- Train on states visited by the learned solver using persistent simulation environments. One-iteration backpropagation through time is a cost-saving option, not a long-horizon guarantee.
- Compare physical-energy training, an uphill penalty, and reference-motion supervision through reconstruction. Energy scaling and offsets must be handled deliberately; a nonlinear transform of absolute energy can depend on its arbitrary zero.
- Evaluate acceptance/rejection against the actual physical objective. Preserve actual accepted updates in history and define how rejection affects recurrent state.
- Cache fixed rest geometry, topology, and suitable matrix factorizations. Current frames, deformation, gradients, and changing constraints are not static.
- Delay collision-projection surrogate gradients and custom backward normalization until their need is established. The reference implementation's straight-through choices are not exact derivatives and are not automatically suitable here.
- Compare against classical optimization under the same physical objective, constraints, and wall-clock budget. Reproducible continuation should include simulation state and optimizer history.

The supplied LCO baseline is not evidence that this proposed architecture is rotation-equivariant. Its physical displacement limits also do not transfer directly to dimensionless deformation steps.

Physical energy and reference motion are both possible training signals. A preferred deformation target is generally not the final observed deformation, so naive target-equals-next-shape supervision is not justified. No final loss, data-generation procedure, architecture size, rollout length, or training schedule has been chosen.

## 16. Proposed validation

These experiments have not yet been run for this solver:

1. Known affine deformations: reconstruct translation-separated stretch, shear, and rotation-compatible targets.
2. Compatibility: conflicting neighboring targets still produce one shared-corner object.
3. Hidden corner modes: demonstrate the center-only failure and check the chosen integration/stabilization.
4. Equivariance: transform the complete problem and compare transformed outputs.
5. Optimizer behavior: accepted-update history, step units, actual descent, and the intended zero-update behavior.
6. Momentum and changing inertia: free translation, rotational motion, and deformation that redistributes mass.
7. Contact coupling: drop otherwise identical soft and stiff voxel blocks onto a floor; compare center-of-mass trajectories, compression, and contact impulses with a consistent reference.
8. Short trajectories and refinement: assess drift, timestep sensitivity, and fitting-weight scaling.

## 17. Decisions still needed before implementation

- Exact fusion body-frame convention, target transport semantics, and treatment of disconnected components.
- Hard versus soft rigid-pose guidance and optional fusion regularization; the physical inertia and inertial-offset definition are already fixed.
- Rigid-target prediction, angular-momentum bookkeeping, and target-feedback schedule.
- Internal fitting weights, mass discretization, quadrature/stabilization, and non-inversion handling.
- Physical contact and prescribed-boundary-condition handling, and their use when constructing fusion targets.
- Final input packing, neighborhood schedule, output-step controller, and no-op contract.
- Physical energy, training signal, differentiation scope, acceptance rule, and rollout schedule.

## 18. Authorized references

- [Initial idea](00_initial_idea.md): the user's original grid, local deformation, assembly, and separate rigid-step proposal.
- [LCO one-pager](References/lco-one-pager.md): optimizer features, output scaling, graph architecture, and training ideas discussed as candidates.
- [Learned-cloth study, unit 3](/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/learned-cloth-study/html/unit-3.html): the specifically requested local lesson, including FiLM, optimizer inputs, and optimization techniques.

The LCO implementation and linked secondary reports were not independently reviewed for this decision record.

## 17. Implemented differentiable hex training step — 2026-09-24

The physical loss now uses eight-node trilinear **hexahedra**, as requested,
with all eight points of 2×2×2 Gauss integration. Its material is compressible
logarithmic Neo-Hookean. Per-cell inputs are now direct Lamé coefficients:
`lambda >= 0`, `mu > 0`, with positive density. Lumped corner
masses accumulate `density * rest_cell_volume / 8` from every incident cell.
The implicit-Euler inertia and original predictor `Y` are unchanged. The loss
contains elasticity and inertia; physical contact and damping are not yet added.

`LearnedHexSolverStep` packs current axes, eight local inertial-offset vectors,
exposed faces, fixed corners, and five normalized material/size/time scalars,
runs the existing transformer, and reconstructs a candidate before evaluating
the physical loss. The baseline state packing is 38 channels plus 9 axes;
optimizer-gradient/history inputs remain optional future additions. Current
polar rotations are recomputed using Torch without gradients. Precomputed
frames can also be supplied. The planned Warp frame implementation is still
future work.

Fusion fits `gradient(corner_displacement)` to the world-frame axis **change**
at all eight Gauss points, then adds the displacement to the current corners.
A single predicted 3×3 change is repeated across the points of each cell.
Weights are `fusion_stiffness * rest_volume`, using the equivalent Young
stiffness `mu * (3 - mu / (lambda + mu))`, with Gauss weights normalized to
sum to one, so volume is counted once. Prescribed corners are imposed exactly
by elimination. This removes singularity from the pinned full-quadrature
solve and preserves unresolved current warping at zero correction. It does
not increase the expressiveness of the per-cell output: constant axis changes
still cannot independently correct every corner-warping mode, so training may
have a residual energy floor.

The CPU prototype caches a float32 SciPy sparse LU factorization. A Torch
autograd function solves the transpose system in backward, propagating
first-order gradients through axis increments, current positions, and
prescribed positions. Geometry, weights, and the constraint set are fixed
for that factorization. Float64 exists only as an explicit validation mode.
A fully GPU sparse solve, second derivatives, and free-body translation/rotation
constraints remain future work; at least one prescribed corner is required here.
Section 21 describes the implemented GPU tensor bridge to the CPU sparse solve.

An optional rigid map carries the current base and predicted axis increments
into the proposed pose before fusion. Pins are enforced inside the solve,
without changing `Y`. In this pinned baseline the pins determine translation;
the map's translation cancels from the exact gradient-fitting minimizer.
This does not settle the previously open free-body pose convention.

The energy rejects a nonpositive determinant at a Gauss point. There is no
inversion repair, line search, contact response, or convergence guarantee in
this first learned update. Gradient checks cover the loss, fusion adjoint,
network parameters, and current geometry with frozen frames. A reproducible
float32-versus-reference report is generated by `validate_gradients.py`.

## 18. Native Newton rigid predictor and solver integration

Implemented the user-requested reuse of Newton's architecture:

- `build_newton_hex_model()` returns an actual Newton Model. Shared corners
  are particles; rest geometry, hex indices, material parameters, and boundary
  flags use registered custom attributes under `learned_intrinsic`. Hex cell
  records use a registered custom frequency. No tetrahedral elasticity is added.
- Physical corner masses remain positive, including fixed corners. Newton
  ACTIVE flags and the explicit fusion constraints represent stationary pins.
- `SolverLearnedIntrinsic` derives from Newton's public SolverBase and exposes
  the standard state-in/state-out/control/contacts/dt step interface. The
  current adapter supports one pinned cuboid with CPU Newton States and
  float32 Torch computation on CPU or CUDA, as detailed in section 21.
- The rigid predictor uses a separate scratch Newton Model and two States.
  It recomputes current mass center, linear/angular momentum, and inertia
  from the physical corners each timestep. Both inertia and inverse inertia
  are refreshed, and angular velocity is inferred from current momentum.
  Newton's public `integrate_bodies()` performs the rigid update with zero
  angular damping; its equations are not duplicated in the experiment.
- The scratch proxy starts world-aligned at the current mass center on every
  call. Its predicted rotation and COM position define the current-to-target
  rigid map. It does not carry an independently advanced physical trajectory.
  Optional impulses in the standalone predictor enter once before integration.
- Compute both the rigid target and original physical Y once from state_in.
  Use rigid-guided fusion once to initialize the candidate before any learned
  iteration, then refine it with the same Y and boundary positions. Final velocities are
  computed once from total corner displacement over the physical dt.
- The rigid predictor is frozen for this prototype. Input Warp State arrays
  are snapshotted before Torch work. Training derivatives through all learned
  inner updates and fusion remain available in `last_result`; the committed
  output State is detached. Physical trajectory differentiation through these
  State boundaries and the predictor remains future work.
- The predictor does not solve contact or boundary reactions. Pins override
  its incompatible pose proposal in fusion. Nonempty contact buffers are
  rejected; external corner forces must exclude gravity to prevent counting
  it twice. The unchanged implicit-Euler loss still contains hex elasticity
  and physical inertia only.

## 19. Multi-iteration learned optimizer contract

The network is a shared optimizer: given the current candidate for one fixed
physical timestep, it proposes a local axis change that fusion converts into
a shared-corner displacement. It is not restricted to the original physical
state or candidates generated by its own earlier calls.

The native adapter now exposes this explicitly:

- `prepare_problem(state, dt)` snapshots the original state and unchanged
  implicit-Euler predictor Y, prescribed positions, and one rigid prediction.
  It retains its own material/dt/fusion context.
- `initialize_candidate(problem)` uses the rigid target in fusion once,
  without calling the network or changing Y.
- `propose_update(candidate, problem)` accepts any feasible float32 candidate
  with exactly satisfied pins, recomputes its frames/axes/features, and returns
  the proposed local axes and actual fused world displacement. The latter is
  the optimizer direction and is exactly zero at prescribed corners.
- `solve(problem, initial_positions=..., iterations=...)` repeatedly calls
  this same optimizer. The default is five learned updates; supplied initial
  candidates are used directly. Intermediate candidates stay attached to the
  graph, so the final physical loss backpropagates through every update.

Each iteration recomputes polar frames but excludes their derivatives. The
multi-iteration finite-difference check replays the recorded frame sequence
under both parameter perturbations to test this deliberately frozen-frame
gradient. A full derivative through changing polar frames is outside this
convention.

An uninitialized network's normal zero head returns zero directions; a small
random head was used to exercise all backward paths before training. The later
smoke-trained checkpoint is recorded in section 23. Diagnostics distinguish a locally descending direction
(`physical gradient dot fused direction < 0`) from actual energy decrease
after the finite update. No line search, fallback optimizer, or convergence
guarantee is added in this inference/backward test.

## 20. Direct Lamé material parameters

The user selected Lamé lambda and mu instead of Young's modulus and Poisson's
ratio. The native model, hex loss, and learned step now accept per-cell
`lame_lambda` and `lame_mu` in pascals. Lambda is nonnegative and mu is strictly
positive for the existing logarithmic compressible Neo-Hookean law, which is
unchanged. Density remains a separate positive per-cell input.

The five FiLM channels are `log1p(lambda/1e5)`, `log1p(mu/1e5)`,
`log(density/1000)`, `log(cell_size/0.025)`, and `log(dt/(1/60))` in SI units.
Using log1p for the Lamé channels supports zero lambda without a singular
feature. Fusion retains its previous equivalent Young stiffness times rest
volume weighting, computed from Lamé coefficients; this scalar does not
replace either coefficient in the physical energy or network conditioning.

## 21. GPU training and fixed sparse-solve bridge

The user requires GPU training. The network, frame extraction, geometric
features, hexahedral energy, and Adam updates run on CUDA in float32, without
TF32 or mixed precision. Native model/data setup and frozen rigid prediction
remain on CPU. Physical problem tensors move once to the network device.

The cached SciPy sparse factorization remains outside Torch on CPU. Its
factorization is constant and is not differentiated. The existing custom
first-order adjoint passes derivatives through fusion inputs: forward copies
axis increments and pin displacements to the sparse solve, and backward
copies free-corner cotangents to the transpose solve. Positions and returned
gradients are assembled on the original Torch device. This does not support
CUDA graph capture or eliminate CPU/GPU synchronization.

The initial training experiment uses one learned update per query, fixed
material/dt, deterministic multiscale physical states, and separately sampled
candidate perturbations around unchanged Y. It records loss curves and saves
initial, periodic, best-validation, and final resumable checkpoints. Validation
uses disjoint physical seeds and fixed candidates. Longer unrolls remain a
later curriculum stage, after measured improvement on held-out states.


## 22. Current design and research checkpoint

This section consolidates the implemented baseline as of 2026-09-24. Earlier
proposal sections are historical where they disagree with this snapshot.
The workspace is `learned-instrinic-solver`, branch
`ankac/learned-instrinic-solver`, based on official Newton commit
`f7a7f6f15c55fb3e889cc46d35a7a66cc693b69b`. All research code is isolated under
`experiments/learned_intrinsic_solver/`; it does not change Newton's core solver.

**Geometry and physics.** The proof of concept uses a canonical 10×10×40 grid
of 0.025 m cubic cells: 4,000 hexahedra, 4,961 shared corners, and a 0.25×0.25×1 m
rest object. The 121 corners on the material z-min face are fixed exactly.
Rest geometry stays canonical when the current shape is augmented. Three
transformed unit material axes describe each cell's center deformation;
their lengths retain stretch and their angles retain shear. One center
matrix does not encode every warped eight-corner shape.

The physical objective is logarithmic compressible Neo-Hookean elasticity
integrated at eight Gauss points per hex, plus the original implicit-Euler
inertia. Material inputs are Lamé lambda, Lamé mu, and density. Positive
physical mass is retained at prescribed corners. The free inertial prediction
Y is computed once from the original physical state, velocity, gravity, and
external forces. Candidate perturbations and optimizer iterations never
redefine it. The separate semi-implicit rigid pose target guides fusion only;
it does not replace Y or the physical inertia. Contact remains unimplemented.

**Input and output contract.** Here B is the number of independent objects,
N is the number of cells, and S=27 is the fixed number of neighbor slots.

| Field | Dimensions | Meaning |
|---|---|---|
| Current local axes | B×N×3×3 | Center deformation expressed in the cell's polar frame |
| Inertial offsets | B×N×8×3 | Original Y minus current corner positions, expressed in this cell's frame and divided by rest edge length |
| Boundary flags | B×N×6 | Exposed rest-material faces |
| Fixed-corner flags | B×N×8 | Which corners have prescribed positions |
| Packed state features | B×N×38 | The 24 offsets, six face flags, and eight fixed-corner flags |
| FiLM conditioning | B×N×5 | Normalized lambda, mu, density, rest edge length, and timestep |
| Edge features | B×N×27×24 | Rest offset (3), receiver-frame current offset (3), relative frame (9), and transported neighbor axes (9) |
| Proposed local axes | B×N×3×3 | Current axes plus a scaled learned correction, in the same input frame |
| Shared step size | B | One dimensionless step size per object |

Rest geometry is explicit through canonical topology, rest offsets, and cell
size. A scalar size suffices for the current cubic cells; arbitrary rest-cell
shapes are not implemented. Current/previous energy gradients and previous
updates remain candidate future inputs, not hidden channels in the baseline.

The transformer has three blocks with hops [1, 1, 1], width 128, four attention
heads, learned edge bias and edge value, FiLM, and query chunks of 128 cells.
Each cell attends to its 26 face/edge/corner neighbors and itself; missing
slots are masked. There is no fixed geometric attention prior or jumping
neighborhood schedule. The output separates a bounded nine-component
correction from a shared sigmoid step size. Training currently caps that
step at 0.05. There are **865,942 trainable parameters**, shared over cells
and repeated optimizer iterations.

**Fusion and differentiation.** Recompute the current polar frame before
each learned update, then freeze its derivatives. Torch currently performs
frame extraction on the network device; a Warp version is future work.
Transport predicted axis changes into world coordinates and fit displacement
gradients at all eight Gauss points, using one constant change per cell.
Add the solved displacement to the current corners. This preserves current
corner warping for zero correction. Fixed corners are eliminated exactly.
Fitting weights use equivalent Young stiffness times rest volume, with Gauss
weights summing to one. Fusion regularization is distinct from physical inertia.

The network, frames, features, physical energy, and Adam run on GPU in
float32; TF32 and mixed precision are disabled. The fixed SciPy sparse LU
factorization and forward/transpose solves remain on CPU. A custom adjoint
returns first-order derivatives through the axis changes, base positions,
and prescribed positions to the original device. The factorization itself
is not differentiated. Newton Model/State setup and the frozen rigid predictor
are also on CPU. Native State commits detach the physical trajectory; learned
inner iterations remain differentiable through fusion.

The optimizer accepts any valid, exactly pinned candidate for the fixed
physical problem. The native solver supports repeated updates with shared
weights. No line search, learned-output repair, physical contact, general
free-body fusion, or convergence guarantee is claimed. Constant per-cell
axis increments restrict the reachable corrections and may leave a residual
energy floor.

## 23. Completed training and capacity measurements

The completed GPU smoke run used 16 physical training states, eight fixed
held-out queries, batch size one, and 64 Adam updates: four passes through
the training state pool, with fresh candidate perturbations on revisits.
It trained one solver iteration per query. All 64 updates completed without
an invalid learned output. The final checkpoint lowered energy on **4/8**
validation queries. Mean normalized energy change was approximately -0.0641%,
but mean raw energy rose from 78.5971 J to 78.9811 J. Per-query normalization
explains why these two aggregates can have different signs. This is a working
training pipeline, not a reliably descending or converged optimizer.

The physical objective E contains elastic and inertial energy. The training
loss per query is `(E_after - E_before.detach()) / max(E_before.detach(), 1 J)`.
Both the baseline and denominator are fixed for backward. Negative values
mean lower energy for that query. This normalization changes the relative
weights of different queries, so raw energy and descent rate must also be
reported. There is no reference-simulation target in this training loss.

A separate L40 capacity test ran two complete forward/backward/Adam updates
per batch in isolated processes, with one full 4,000-cell object per sample,
float32, K=1, and no TF32 or mixed precision. Batch **72 passed**; batch **73
ran out of memory**. This test repeats existing physical examples to measure
memory; it is not a larger dataset or a multi-GPU scaling result. Batch 16
used 11.16 GB peak allocated memory (decimal) and processed approximately
25.5 samples/s in its second step. Throughput was similar for larger batches.
Capacity must be remeasured for longer unrolls or a changed architecture.

Compact measured results are preserved in [results/README.md](results/README.md).
The local full run and checkpoints remain in
`generated/training/lame_gpu_single_step_001/`. Browser artifacts are available
through the [project index](https://ankachen.com/artifacts/learned-intrinsic-solver/index.html)
and [smoke-training report](https://ankachen.com/artifacts/learned-intrinsic-solver/training/index.html).
Generated datasets, videos, and binary checkpoints are not Git source files.

The current trainer has no fresh-output overwrite guard: a fresh run must use
a new output directory; resume must be explicit. Do not reuse the completed
smoke-run directory for a new campaign.

## 24. Next training campaign: final configuration before implementation

The user requested a much larger dataset, more epochs, loss monitoring until
convergence, saved checkpoints, and GPU training. The following is the selected
next configuration. **It is recorded before implementation; the campaign has
not started.** Batched epoch training, DDP, and the scheduler/stop controller
still need implementation and validation.

| Setting | Selected value |
|---|---|
| Physical training states | 8,192, seeds 0–8191 |
| Fixed validation states | 512, seeds 10000–10511, disjoint from training |
| Master seed | 73; deterministic sample and perturbation seeds recorded |
| Initial unroll | One learned optimizer iteration per query |
| Per-GPU batch | 16 complete objects |
| GPU target | Four L40s, one DDP process per exclusively claimed GPU |
| Global batch | 64 with four GPUs; fall back to two GPUs (32) or one GPU (16), recording the actual world size |
| Epoch definition | One visit to each of the 8,192 physical training states |
| Updates per epoch | 128 at global batch 64, with no duplicate padding |
| Optimizer | Adam, initial learning rate 0.0001 |
| Monitoring | Full fixed validation every epoch, plus epoch-mean training loss |
| Minimum monitoring period | 30 epochs before a plateau can terminate the run |
| Initial run limit | 200 epochs; a limit exit is reported separately from convergence |
| Learning-rate schedule | Halve after five validation epochs without improvement; floor 0.000001 |
| Plateau threshold | Improvement of at least 0.0001 in mean normalized validation loss |
| Plateau stopping | 15 consecutive epochs without that improvement, after at least two LR reductions and the minimum monitoring period |
| Checkpoints | Initial, latest each epoch, best validation, every ten epochs, final, and failure |

Use only one, two, or four ranks so the full training pool divides evenly
into complete global batches without duplicate padding or a dropped tail.
The scheduler and plateau tracker use the same absolute improvement threshold.
A lower validation loss is better. Reaching a plateau only establishes that
this training configuration stalled. Label it successful measured convergence
only if the last five validations have no invalid outputs, at least 95% of
queries decrease energy, and mean raw post-update energy is below its fixed
pre-update baseline. Otherwise report stalled training and diagnose it; do
not call poor flat performance convergence. Even these criteria do not prove
descent for every possible state or that every query reached its minimum.
If loss is still improving at the run limit, preserve resumable state and
report that further training is needed.

Keep the current canonical rest shape, pin set, material, timestep, and gravity
for this first larger comparison: lambda=288461.53846 Pa, mu=192307.69231 Pa,
density=1000 kg/m³, dt=1/300 s, and gravity=(0,-9.81,0) m/s². A larger set of
shapes does not establish generalization across new materials or geometry.
Material/dt variation is a later controlled experiment.

Each physical state contains a deterministic multiscale deformation and a
smooth shared-corner velocity field. Preserve the existing strength range
0.02–0.1 and velocity displacement RMS range 0–0.1 cell widths per timestep.
On every epoch visit, freshly draw the optimizer candidate using the current
mixture: 50% pin-corrected Y, 35% Y plus multiscale displacement noise with
RMS 0.01–0.1 cell widths, 10% previous physical positions, and 5% rigid-guided
initialization. Restore exact pins and screen the final float32 candidate.
Noise may be reduced during initialization to obtain a valid sample; an
invalid learned proposal is a reported failure, without output repair.
Validation candidates and their physical objectives remain fixed across epochs.

Record both normalized loss and raw energies, descent fraction, invalid-query
count, learning rate, completed epochs, optimizer updates, sample count,
elapsed time, and GPU memory. Preserve CSV data and browser-viewable loss
curves with **epoch on the horizontal axis**. Separate pre-update training
batch statistics from post-epoch validation. Validation failures count against
the complete validation population, never disappear from success rates.

Wrap the complete learned step in DDP, shard physical identities without
repeated padding, and average equally sized per-rank batch losses. Each rank
owns its CPU sparse factorization. Coordinate failures before backward to
avoid leaving another rank inside a collective. Rank zero writes shared
reports/checkpoints; preserve every rank's RNG/sampler state, optimizer and
scheduler state, dataset identity, fixed validation candidates, and progress.
Keep large immutable dataset data separate from repeated checkpoints. Use
a fresh output directory and atomically replace report/checkpoint files.
Validate batched gradients, DDP updates, coordinated failure, and resume on
a small run before launching the full campaign. Multi-GPU speedup remains
unmeasured; CPU sparse solves and transfers may limit it.

Only increase the number of consecutive learned iterations after one-step
held-out descent is reliable. First evaluate two- and five-step sequences
with the same original Y and refreshed, backward-frozen frames. Require finite,
valid trajectories and sustained energy improvement before expanding training
unrolls. Put the full unroll inside one DDP forward and remeasure batch memory.
Increasing K changes the experiment and must be logged; it does not occur
merely because an epoch number has been reached.


## 25. Four-GPU path validation — 2026-09-24

The bounded distributed diagnostic is now implemented and tested. It wraps
`LearnedHexSolverStep` in PyTorch DistributedDataParallel, with four processes
on four distinct, exclusively claimed NVIDIA L40s. Each GPU processes 16
complete 10×10×40 objects, giving a global batch of 64. The 64 distinct physical
queries remain fixed across three Adam updates for numerical replay; this is
not the 8,192-state epoch campaign or a convergence result.

All three updates completed with finite outputs and gradients, exact pins,
and byte-identical parameter replicas. Serialized first-update checkpoints
were reloaded and updates two and three repeated: every rank reproduced the
weights, Adam state, and losses exactly. An independent single-GPU reference
accumulated the same four local batches with the global-mean loss weighting.
Its first gradient had relative L2 error 5.50616e-08;
the first Adam update had relative error 7.61026e-07.
Final checkpoint energy replay had zero absolute difference. First and final
Adam moment comparisons also passed. Working precision remains float32 with
TF32 and AMP disabled; each rank retains its CPU SciPy sparse solves.

Default NCCL peer-to-peer transport stalled in startup collectives on this
machine. The launcher timed out after 180 seconds and stopped all ranks.
Setting `NCCL_P2P_DISABLE=1` for the invocation made the same four-GPU path
work; NCCL logs selected shared host-memory transport. This is a measured
machine-specific workaround, not evidence that default P2P transport passed.
No global NCCL configuration was changed.

Peak allocated memory was 10.42 GiB per GPU.
Across updates two and three, the slowest-rank full iteration averaged
0.662 seconds; the equivalent single-GPU
serial accumulation averaged 2.482 seconds,
a 3.75× short-test speedup. Setup, warmup, and checkpoint
replay are excluded. These two timed updates are preliminary throughput
measurements, not a sustained epoch-training benchmark.

An intentional rank-2 failure during update two was reported by every rank
before backward. Only the first Adam update was applied, the launcher exited
without timeout, and all GPU reservations were released. The independent
reference rejects this failed probe instead of publishing a passing result.
The launcher also has real-process tests for timeout, worker failure,
SIGTERM-resistant descendants, and output-directory preservation.

All 133 experiment tests, including CUDA checks, passed. Reproduction commands
are in the [experiment README](../experiments/learned_intrinsic_solver/README.md#four-gpu-numerical-diagnostic).
Compact results are [four-gpu-summary.json](results/four-gpu-summary.json) and
[four-gpu-reference.json](results/four-gpu-reference.json); full local inputs,
checkpoints, and rank logs are under `generated/distributed_probe/full_001/`.
The [browser report](https://ankachen.com/artifacts/learned-intrinsic-solver/multi-gpu/index.html)
joins the existing project pages. Epoch sampling, validation, scheduling, and
the larger training campaign remain the next implementation step.

## 26. Larger epoch campaign implementation — 2026-09-24

The approved section 24 campaign now has separate dataset, trainer, schedule,
report, and launcher modules under `experiments/learned_intrinsic_solver/`.
Each rank owns a contiguous disjoint physical seed shard, with one shared
fusion factorization and immutable CPU dataset snapshot. Candidate randomness
is a deterministic function of master seed, physical seed, and epoch; local
shuffling visits each identity exactly once. Validation queries remain fixed.
Resume verifies dataset hashes, configuration, and rank count before loading
model, Adam, controller, and each rank's RNG state.

Initial validation sets the scheduler's baseline. Per-epoch loss reduction and
success rates include the full configured population; invalid validation loss
is unavailable rather than averaged over only successful cases. Training
coordinates input-transfer and forward failures before backward, and rejects
nonfinite gradients before Adam. Rank replicas are checked after every epoch.
A failure preserves the actual rank batches and a diagnostic checkpoint.

The launch configuration is 8,192 training states, 512 validation states,
10×10×40 cells, four GPUs, batch 16 per GPU, fresh seed-73 weights, and one
learned iteration. The curve uses completed epochs on its horizontal axis.
Reports and initial/latest/best/every-ten/final/failure checkpoints are atomic.
CPU sparse fusion forward/transpose solves remain unchanged; CUDA handles the
network, frames/features, energy, and Adam. Float32, TF32 off, no AMP.

The full-grid epoch path passed a four-GPU two-epoch run and a separate
one-epoch run resumed to epoch two. Each used 64 training and 16 validation
states, batch 16 per rank. Both epochs visited every state exactly once, with
finite proposals and gradients and equal replicas within each run. The two
trajectories matched within float32 tolerance: maximum absolute weight
difference 1.49e-08, relative update difference 1.65e-06, relative Adam-state
difference 1.81e-09, and validation-loss difference 6.17e-09. Dataset hashes,
controller state, and descent counts matched. Separate launches can have
slightly different float32 collective reduction order; this is not a claim of
bitwise replay across launches. CPU checkpoint resume is bitwise exact.
Compact measurements are [epoch-resume-summary.json](results/epoch-resume-summary.json).

The full campaign runs under a persistent session with automated plateau
monitoring and a browser report refreshed after every completed epoch. Its
runtime state and results are distinct from the previous 64-update smoke test.

Validation before launch: all 150 experiment tests passed, including CUDA and
checkpoint-resume checks. Repository pre-commit checks passed. The validation
fallback additionally handles per-query device transfer failures and retains
them in the complete-population failure count.

The larger campaign was launched as `generated/training/large_001` in tmux
session `learned-intrinsic-training`, using trainer commit `67f30ad4`. First
epoch: all 8,192 training states visited, 128 Adam updates, no invalid outputs.
Held-out one-step validation improved all 512/512 cases; normalized mean energy
change was -0.122199 and mean raw energy went from 91.9870 J to 81.6291 J.
The epoch took 229.8 seconds after preparation. Peak allocated memory on rank
zero was 10.68 GiB. This is an initial measurement, not convergence; epoch two
was running when recorded. Automatic scheduling/stopping and live publication
continue in the persistent process. The first-epoch report and published latest
and best checkpoints were verified against local files. Snapshot:
[large-training-start.json](results/large-training-start.json). Live results:
[training-large](https://ankachen.com/artifacts/learned-intrinsic-solver/training-large/index.html).

## 27. Extend the current campaign to 500 epochs — 2026-09-24

The user requested a full 200 epochs and then raised the target to 500. For
`large_001`, set `max_epochs=500` and disable plateau early stopping. Retain
the existing validation-driven learning-rate reductions, every-epoch
validation, latest/best checkpoints, and every-ten-epoch archives. Invalid
learned outputs or nonfinite training still stop with a diagnostic; reaching
500 is a budget completion, not a convergence claim.

Resume from the existing latest completed-epoch checkpoint, preserving model,
Adam, scheduler history, RNG states, all epoch metrics, and immutable dataset
hashes. Only the stopping policy and epoch cap change. The resume supervisor
gets a 72-hour timeout because 500 epochs exceed the original 24-hour limit
at the measured roughly 230 seconds per epoch. Four GPUs, global batch 64,
float32 precision, K=1, physics, training data, and fixed validation stay the
same. The live page reports the configured target and early-stopping policy.

The policy switch preserved epoch 15 / 1,920 Adam updates in
`checkpoints/before_500_epoch_0015.pt`. All four immutable dataset hashes matched
the saved checkpoint. Sixteen relevant CPU tests passed, including exact legacy
checkpoint resume with the stopping flag changed. A four-GPU full-grid smoke
run resumed an old epoch-two checkpoint through epoch three with the flag
disabled: all ranks exited successfully and validation had no invalid cases.
Repository pre-commit checks passed. The production restart uses the same
`large_001` directory and begins epoch 16 from the epoch-15 checkpoint.

## 28. Stop training and evaluate 100 solver iterations — 2026-09-24

The user stopped the campaign after observing a plateau. Training stopped
with epoch 198 / 25,344 completed Adam updates preserved; the partial epoch
199 is excluded from the final checkpoint. Latest and best checkpoints remain
available. All four GPU reservations were released. Final validation one-step
normalized energy change was -0.591373, with 512/512 states decreasing energy
and no invalid outputs. Raw mean energy was 91.9870 J before and 40.5051 J after
one update. The final resumable checkpoint has `status=stopped_by_user`.
[Stop snapshot](results/large-training-stop.json).

The user next requested 100 consecutive solver iterations on the fixed
validation set. Evaluate the saved epoch-198 final model on all 512 original
validation candidates, with weights fixed and no Adam updates. Hold each
original implicit Euler inertial target and pins unchanged, recompute frames
from the current candidate each iteration, and feed each fused global shape
into the next learned iteration. Physical time does not advance.

For each query and iteration, divide current implicit Euler energy by that
query's initial energy, using the exact positive initial value without a
1-joule floor. Plot maximum, arithmetic mean, and median of these per-query
ratios at iterations 0–100. This is a relative physical-energy measure, not an
error relative to the unknown minimum. All curves start at one. Initial
energies and first-step results must agree with the saved training validation.
Preserve per-query energies and ratios as downloadable data.

Report failed trajectories explicitly without repair, a line search, or
holding their last valid energy. Full-population statistics become undefined
if any query fails; any remaining-valid statistics must be separately labelled.
The bounded evaluation can use four independently claimed GPUs with float32,
TF32 and AMP off, and the existing CPU sparse fusion solves. Training remains
stopped throughout this evaluation.

The requested evaluation completed all 100 iterations for all 512 validation
states, without invalid geometry, nonfinite energies, or changed network
parameters. All four ranks exited successfully; evaluation took 134 seconds.
The first-step normalized energy change matched the saved epoch-198 validation
within 4.14e-09, validating the initial queries, checkpoint, and energy measure.
All initial energies were above 8.3 J, so the training normalization's 1 J floor
was inactive for this validation set.

The mean energy ratio fell to 0.408627 after one iteration and reached its
minimum, 0.368290, at iteration 3. It then rose: at iteration 100 the maximum
ratio was 2.607246, the mean 1.245757, and the median 1.184489. At that point
380/512 states had more energy than initially. Geometric validity therefore
coexists with failed repeated descent; the one-step-trained network does not
maintain an optimizing trajectory over 100 iterations. Training remains
stopped; no repair, line search, or additional training was added.

The [100-iteration report](https://ankachen.com/artifacts/learned-intrinsic-solver/validation-100/index.html)
contains linear/log curves, maximum/mean/median values, and all per-case data.
A compact archived result is [validation-100-summary.json](results/validation-100-summary.json).
All 159 experiment tests passed, including CUDA and evaluation regressions.

## 29. Ten physical rollouts with two learned iterations — 2026-09-24

The user requested ten initial states, each physically simulated for ten
seconds with two learned solver iterations per timestep and recorded using
Newton's renderer. Use the stopped epoch-198 final checkpoint, dt = 1/300 s,
and the existing native Newton Model/State/SolverLearnedIntrinsic integration.
Each timestep prepares a new original implicit Euler inertial target from
current positions and velocities, applies the existing rigid-guided candidate
initialization, and makes exactly two consecutive learned proposals against
that target. The resulting positions update velocity and become the next
physical state. This is distinct from the earlier fixed-objective validation.

Select ten saved held-out physical states without inspecting results, using
NumPy default_rng(73) to sample without replacement from seeds 10000–10511:
10015, 10112, 10136, 10153, 10248, 10356, 10395, 10411, 10488, and 10502.
Use their saved positions and velocities, with the original z-min pins. Keep
float32, trained materials, gravity, and the existing CPU sparse fusion solve;
network inference runs on independently claimed GPUs. Training stays stopped.

Save positions at 30 frames per simulated second, then render surface geometry
and grid lines with Newton ViewerGL. Keep each clip's camera fixed and frame
its full recorded motion. Preserve checkpoint/dataset identities, actual
proposal counts, physical times, and per-case completion or failure. If a
proposal becomes invalid, retain the valid prefix and report the failure;
do not change the solver, restart the state, or present a padded video as ten
seconds of completed simulation.


The ten attempts produced 23,733 completed physical timesteps and 47,474
attempted learned updates. Every accepted timestep executed exactly two
updates; each of the eight failed steps stopped in its first learned update.
Seeds 10248 and 10356 completed all 3,000 steps / 10 seconds. The other seeds
stopped when a proposed shape had a nonpositive hexahedral Gauss Jacobian:
10015 at 7.930 s, 10112 at 7.136667 s, 10136 at 6.230 s, 10153 at 8.546667 s,
10395 at 6.960 s, 10411 at 4.326667 s, 10488 at 8.273333 s, and 10502 at
9.706667 s. Every retained frame preserves all 121 pins exactly, and the
checkpoint parameter hash remains unchanged. These are interrupted attempts,
not ten successfully completed ten-second simulations.

Newton ViewerGL rendered all ten saved trajectories, with a fixed camera per
clip, surface voxel edges, orange pins, and physical timestamps. Complete
recordings contain 300 frames at 30 fps. Failed recordings stop at the last
valid state; an off-grid terminal sample can change playback length by less
than one frame. All ten MP4s were decoded and checked for frame counts and
nonuniform moving imagery. Desktop and mobile page layouts were inspected;
the public page returned HTTP 200 and video range requests returned HTTP 206.
All 167 experiment tests and repository pre-commit checks passed.

While these runs were in progress, the user pointed out that the optimizer
should represent and recover from inversion. The three-axis representation
can encode an inverted shape, but the present frame extraction explicitly
rejects inverted cell centers and the logarithmic Neo-Hookean energy rejects
nonpositive Gauss Jacobians. Recovery is therefore not implemented in this
baseline. Proper rotation frames with signed local axes, an energy defined
through inversion, and recovery training require a separate change; these
recordings preserve the existing checkpoint and physical formulation.

[Watch all ten recordings](https://ankachen.com/artifacts/learned-intrinsic-solver/learned-two-iterations/index.html).
The compact reproducibility record is
[learned-two-iterations-summary.json](results/learned-two-iterations-summary.json).

## 30. Train up to 32 iterations and consecutive physical steps — 2026-09-24

The user requested a maximum of 32 learned optimizer iterations per physical
timestep and training on consecutive physical timesteps. These are separate
counts: the inner solve holds its original inertial prediction fixed, while
each physical advance updates positions and velocities and constructs a new
inertial prediction. The previous physical positions must remain distinct
from the candidate shape used to initialize the optimizer.

The proposed curriculum starts from the existing epoch-198 weights and grows
through (inner iterations, physical steps) = (2,2), (4,2), (8,4), (16,4), and
(32,8), retaining some shorter queries. Progression must depend on held-out
per-iteration descent and physical-rollout results, and the actual counts and
curriculum stage must be recorded in curves and resumable checkpoints. A
stage that cannot advance must be reported explicitly. It must not silently
repeat the old one-iteration campaign. Later training windows should also
continue existing simulated trajectories to cover states beyond their first
few timesteps, alongside fresh augmented starts.

The user selected independent training at each physical timestep on the state
produced by the preceding timestep. The default gradient window is therefore
one physical step. Detach both positions and velocities at every physical
timestep boundary, preserving their values exactly. Within that timestep,
all learned optimizer iterations remain attached to the gradient graph, up
to 32 iterations. Each physical timestep has its own loss and backward pass;
weights can be updated after that solve before advancing to the next solve.
Weights must remain fixed throughout an inner solve and its backward pass.
Eight physical steps per forward rollout remains the proposed horizon.
Activation checkpointing recomputes the full learned step, including
frame/edge feature construction and fusion, to bound memory. Start measured
batch sizes small before sizing four-GPU training.

Each physical step averages the normalized energy change after all inner
updates and adds a penalty for energy increases. Its initial energy and
normalization scale are detached, as is the preceding energy used as each
descent comparison target. The physical loss treats that timestep's inertial
prediction as fixed, while its network-feature path remains differentiable.
This avoids rewarding changes to the energy reference itself. Frame
decomposition and native semi-implicit rigid guidance remain frozen as
previously agreed. The training tensor path reuses native Newton model
metadata and explicitly manages state continuity and the chosen gradient
boundaries.

New reusable modules implement the inner unroll and physical gradient
windows. This is design and feasibility work: the existing epoch trainer has
not yet been wired to this curriculum, and the stopped production campaign
has not resumed. Inversion-tolerant energy/frame handling remains a separate
requirement before claiming inversion recovery or robust long trajectories.


Before the user's one-timestep gradient choice, the CUDA feasibility check
used the full 10×10×40 grid and the
epoch-198 model on held-out physical seed 10000: eight physical steps,
32 learned iterations per step, two-step gradient windows, and activation
checkpointing. Positions and velocities carried forward exactly between
windows, pins remained exact, input and network gradients were finite and
nonzero, and parameters stayed unchanged (zero Adam updates). The batch-one
forward/backward run took 58.60 seconds with peak Torch allocated memory
742,385,664 bytes. All 177 experiment tests and repository pre-commit checks
passed. This is gradient-path feasibility, not improved solver quality: the
existing model increased energy sharply from the third physical timestep.
Results: [unroll32-physical8-probe.json](results/unroll32-physical8-probe.json).

After that choice, the rollout default was changed to one physical timestep.
A regression test failed under the old two-step default and passes with the
new policy. It verifies exact X/V continuity, independent backward passes,
and a parameter update between timesteps without a gradient connection back
to the preceding timestep. All eleven focused unroll/rollout tests passed.

## 31. Diagnose failure to preserve rest — 2026-09-24

The user observed difficulty reaching a resting state and asked how to improve
it. A force-free diagnostic used the epoch-198 checkpoint, the canonical
10×10×40 rest shape, zero velocity, zero gravity, and fixed inertial target
equal to that rest shape. The first proposal moved corners by 0.060 mm RMS
and raised energy from about 0.00014 J to 0.01546 J. All ten fixed-target
updates increased energy, reaching 0.03294 J. Pins remained exact and the
geometry stayed valid. The small nonzero initial energy is float32 evaluation
error. This test diagnoses drift from rest; it does not establish convergence
behavior for gravity-loaded equilibrium shapes.

The correction head can represent exactly zero, but this checkpoint's sigmoid
step controller is saturated at its configured maximum, 0.05. There is no
explicit rule preserving equilibrium. Recommended next changes are training
on verified equilibria and small perturbations around them, penalizing drift
from those equilibria, exposing the current implicit-Euler force residual,
and making corrections vanish as that residual vanishes. Those changes are
proposals, not implemented by this diagnostic. Consecutive optimizer and
physical-step training should include states near convergence as well as
large deformations.

Diagnostic: [epoch198-rest-drift.json](results/epoch198-rest-drift.json).

## 32. Retain physical states on disk and vary materials — 2026-09-24

The user approved implementing disk retention of previous physical states.
Keep every collected timestep input, with no automatic replay-buffer eviction.
Use float32 positions and velocities, the applicable loads and prescribed
positions, and a shared immutable physical context for rest geometry,
topology, material, mass and timestep. Replay recomputes the implicit-Euler
problem with the current network; historical predictions are not labels.

The implementation uses one SQLite archive per rank, atomic durable batch
appends, compressed deduplicated contexts, and lazy reads across rank files.
Commit a timestep's input before solving it, so failed solves retain their
starting states. Trajectory IDs and step indices cannot be overwritten, and
one trajectory cannot switch physical contexts. Sampling uses a seeded fixed
row-ID snapshot; explicit refresh admits newer data. The reader loads only
selected arrays. There is no storage-imposed ratio between replay and live
training, and no automatic trainer restart.

The user also requested material ranges: lambda and mu each from 1,000 to
1,000,000 Pa, density from 100 to 10,000 kg/m³. The new deterministic sampler
draws these independently and log-uniformly, once per object/trajectory.
Retained states restore their actual material and native masses; replaying a
state with another context is rejected. The old fixed-material training
dataset and trained checkpoint are unchanged.

This implements storage, the physical-rollout retention/replay adapter, and
the material sampler. The production epoch trainer still needs the separately
reviewed consecutive-iteration/physical-step integration before another
campaign. No new training campaign was authorized or launched.

During this work the user asked about inversion. The three signed axes can
represent it; a proper rotation frame can be obtained by sign-correcting an
SVD rotation and computing local axes from the original deformation matrix.
Reconstruction retains the inversion in those axes. Frame continuity near
degeneracy, an inversion-tolerant continuation of the elastic energy, and
recovery examples remain proposals. None of these physics changes is part of
the disk archive implementation.

Usage and storage contract: [DISK-REPLAY.MD](DISK-REPLAY.MD).
