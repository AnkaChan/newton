# `SolverCoupledProxy` — how the proxy coupling works

How Newton couples a **MuJoCo arm** with a **VBD cloth** in the `--solver proxy` experiment
(`python -m newton.exp --solver proxy`), mirroring IsaacLab `Isaac-Pick-Proxy-Cloth-Direct-v0`.
Everything below is from source:

- exp wiring: `newton/exp/solvers/proxy.py`, `newton/exp/runner.py`
- coupler: `newton/_src/solvers/coupled/solver_coupled_proxy.py`,
  `.../solver_coupled.py`, `.../interface.py`, `.../proxy_utils.py`
- sub-solvers: `newton/_src/solvers/{mujoco,vbd}/...`

---

## Contents

**Part I — Orientation**
- 1. Architecture — a partitioned multi-solver
- 2. The proxy concept

**Part II — How it runs**
- 3. Per-substep algorithm (`_step_proxy`, `lagged` mode)
- 4. `lagged` vs `staggered`
- 5. The force semantics
- 6. Key parameters
- 7. The outer `step()`

**Part III — Why it's correct**
- 8. The interface problem & the proxy fixed point (8.0-8.7; 8.8 "can we copy the exact arm inertia?")

**Part IV — Comparison with Genesis IPC**
- 9. Relation to Genesis `two_way_soft_constraint` (9.1-9.4)

**Part V — Practical**
- 10. How to run
- 11. Notes / limitations

**Appendices**
- 12. The per-solver implicit step
- 13. Effective (inverse) mass
- 14. When is the contact impulse a Lagrange multiplier? (what Eq. (1) becomes)

---

**Part I — Orientation**

## 1. Architecture — a partitioned multi-solver

`SolverCoupled` splits the single `Model` into **entries**, each a sub-solver over a *disjoint*
subset of the model (its own `ModelView` of bodies / particles / shapes). The proxy experiment uses
two:

| Entry | Solver | Owns | Notes |
|---|---|---|---|
| `mjc` | `SolverMuJoCo` | the **Franka arm** (bodies + joints + shapes) | IsaacLab gains: arm 4000/400, fingers 4e4/400, 1000 N effort, armature 1e-3; gravity on, **no** gravcomp |
| `vbd` | `SolverVBD` | the **cloth particles + static shapes** (here just the **ground plane**, `add_ground_plane`) | self-contact on, 20 iterations |

"Static shapes" = fixed, world-attached collision geometry (ground/table/walls); in the `shirt_pick`
scene that's only the ground plane.

The two solvers run independently on their partitions; **`SolverCoupledProxy` layers the cloth↔arm
interaction on top** via *proxies*. (There is also a monolithic alternative, `--solver avbd`, that
solves everything in one AVBD system — the analogue of a single coupled solve.)

**Disjoint collision worlds.** `_apply_entry_shape_visibility` (`solver_coupled.py`) clears the
collision flags on every shape *not* owned by an entry, so each solver only collides within its own
shape set (+ its proxies):

- `mjc` (arm) sees **only `robot_shapes`** — arm self-collision is possible, but there is **no
  arm↔ground and no direct arm↔cloth** collision.
- `vbd` (cloth) sees the **cloth + ground + gripper proxies**.

So arm↔cloth happens *only* through the proxies (§2), and **arm↔ground happens nowhere** — the ground
is masked out of the arm's view (the analogue of IsaacLab's `disable_robot_ground_collision`). If the
arm were driven into the floor, there would be no contact response.

---

## 2. The proxy concept

The **gripper bodies** (owned by `mjc`) are *also* registered as **virtual proxy bodies inside the
VBD world** — duplicate bodies the cloth can collide against. So the cloth "sees" the gripper, and
the interaction crosses the solver boundary by **(a) syncing the proxy to the arm's pose** and
**(b) harvesting the cloth's contact wrench back onto the arm**. Configured in `proxy.py`:

```python
SolverCoupledProxy.Proxy(
    source="mjc", destination="vbd",
    bodies=handles.gripper_bodies,        # which arm bodies become proxies
    mass_scale=5.0,                       # proxy inertia in VBD
    mode="lagged",                        # or "staggered"
    collision_pipeline=lambda m: newton.CollisionPipeline(m, broad_phase="explicit"),
    collide_interval=1,
)
```

---

**Part II — How it runs**

## 3. Per-substep algorithm (`_step_proxy`, **lagged** mode) — exact math

### Notation

For one `(mjc → vbd)` proxy group at substep $n$:

- $\mathbf q$ = body transform; $\dot{\mathbf q}=[\mathbf v;\,\boldsymbol\omega]$ = body **spatial velocity**
  (Warp `spatial_vector`: top = linear $\mathbf v$, bottom = angular $\boldsymbol\omega$).
- $\mathbf F=[\mathbf f;\,\boldsymbol\tau]$ = `coupling_forces`, a **spatial wrench** stored per proxy
  (global id). $\mathbf F^{\,n-1}$ is last substep's value.
- $m,\ \mathbf I$ = proxy mass and body-frame inertia **as seen by VBD**: $m=\texttt{mass\_scale}\cdot m_\text{real}$.
- $\mathbf R$ = proxy world rotation; $\mathbf g$ = per-body gravity acceleration; $\alpha=\texttt{proxy\_relaxation}$; $\Delta t$ = substep.
- superscript `arm` = the `mjc` solver's body; `proxy` = the matching virtual body in `vbd`.

The whole pass below repeats `proxy_iterations` times per physics step.

### Step 0 — Stash (`stash_proxy_body_forces_kernel`)
Save the current (lagged) feedback for the later blend:
$$\mathbf F^{\text{prev}} \leftarrow \mathbf F^{\,n-1}.$$

### Step 1 — Apply the lagged force, step the arm (`_add_body_force_input` → MuJoCo)
Add last substep's harvested cloth wrench to the gripper bodies' external force, then solve the arm:
$$\mathbf f^{\text{arm}}_\text{ext}\mathrel{+}=\mathbf F^{\,n-1},\qquad
(\mathbf q^{\text{arm}}_{n+1},\dot{\mathbf q}^{\text{arm}}_{n+1})=\mathrm{MuJoCo}\big(\mathbf q^{\text{arm}}_n,\dot{\mathbf q}^{\text{arm}}_n,\ \boldsymbol\tau_\text{PD},\ \mathbf f^{\text{arm}}_\text{ext},\ \Delta t\big).$$

### Step 2 — Sync the proxy + snapshot (`sync_proxy_states_kernel`)
Copy the arm's pose and velocity onto the proxy body in VBD, then record the pre-solve velocity:
$$\mathbf q^{\text{proxy}}\leftarrow \mathbf q^{\text{arm}}_{\star},\qquad
\dot{\mathbf q}^{\text{proxy}}\leftarrow \dot{\mathbf q}^{\text{arm}}_{n+1},\qquad
\dot{\mathbf q}^{\text{before}}\leftarrow \dot{\mathbf q}^{\text{proxy}},$$
where $\star=n$ (**begin** pose, `state_0.body_q`) for *lagged*, $\star=n{+}1$ (**end** pose,
`state_1.body_q`) for *staggered*; the velocity is the arm **end** velocity in both.

### Step 3 — Rewind (`subtract_proxy_body_forces_kernel`)
Overwrite the proxy's VBD force input so gravity and the lagged feedback are removed before the cloth solve:
$$\mathbf f^{\text{proxy}}_\text{ext}\leftarrow -\,\mathbf F^{\,n-1}-\big[\,m\,\mathbf g\,;\ \mathbf 0\,\big].$$

**What $\mathbf F^{\,n-1}$ is here.** It is `coupling_forces[global_id]` — the *exact same* spatial wrench
that Step 1 added to the gripper body inside MuJoCo. It is not a fresh quantity: it is the cloth→gripper
reaction harvested at the **previous** substep, and Step 1 of *this* substep already consumed it (pushed
the arm with it). The kernel writes (not `+=`) `dst_body_f = -F - [m g; 0]`, so whatever was in the
proxy's force slot is discarded and replaced by exactly this value.

**Why subtract it (the $-\mathbf F^{\,n-1}$ term).** The proxy is the *same physical gripper* as the arm
body, mirrored into VBD. The lagged wrench has therefore already been applied **once**, on the arm side,
in Step 1 — and Step 2 synced the arm's resulting end-velocity onto the proxy, so the proxy enters the
cloth solve already *carrying the momentum* that $\mathbf F^{\,n-1}$ imparted. If VBD then also kept
$\mathbf F^{\,n-1}$ on the proxy during its own solve, the one cloth reaction would be applied **twice**
within a single substep (once to the arm, once to the proxy), and — worse — it would leak into the
harvest: the momentum-change readout in Step 6, $m\,\Delta\dot{\mathbf q}/\Delta t$, would then measure
"new cloth contact **plus** the recycled $\mathbf F^{\,n-1}$," so the feedback loop would accumulate its
own past output instead of converging. Writing $-\mathbf F^{\,n-1}$ pre-cancels that contribution to the
proxy's momentum over the step, so what the solve adds to the proxy is *only* this substep's fresh cloth
contact — which is exactly what Step 6 should harvest.

**The $-m\mathbf g$ term** is independent: it cancels the gravity VBD would otherwise integrate onto the
proxy, so the heavy ($m=\texttt{mass\_scale}\cdot m_\text{real}$) proxy tracks the arm pose instead of
sagging during the solve. Gravity belongs to the *real* arm dynamics (resolved by MuJoCo), not to this
kinematic stand-in.

Net effect: during the cloth solve the proxy's velocity changes **only** due to cloth contact — which is
what makes the harvest clean. (For *staggered*, $\mathbf F$ is zeroed first, so this reduces to
$-[m\mathbf g;\mathbf 0]$ — there is no lagged term to remove.)

### Step 4 — Detect contacts
Run the proxy `CollisionPipeline` (`broad_phase="explicit"`, refreshed every `collide_interval`, =1) to
find cloth↔proxy contacts.

**What geometry the proxy collides with.** The "proxy" is *not* a simplified box or sphere — it is the
gripper bodies' **own authored collision shapes**. The proxy mapping is `bodies=handles.gripper_bodies`,
which resolves (`gripper_body_ids`) to the Franka **hand + two finger** bodies; those bodies' real
collision shapes are carried from the robot model into the `vbd` entry's view (as *non-owned* bodies — VBD
sees them for collision but does not integrate them as its own DOFs). So the cloth detects contacts
against the actual Franka finger/hand collision geometry. The word "proxy" refers only to the body's
**mass/kinematic** treatment (its pose is imposed from the arm and its mass is scaled), **not** to a
substitute shape. The contacts are cloth-particle↔shape soft contacts (`soft_contact_*` parameters), found
by this private pipeline — *not* by the global collision pass that the arm/MuJoCo entry uses.

### Step 5 — Step the cloth (`SolverVBD`)
Solve the cloth with the proxies present as collision bodies of mass $m=\texttt{mass\_scale}\cdot m_\text{real}$
(the same gripper shapes from Step 4, now with the scaled mass so the cloth's push registers as a finite
velocity change for the harvest):
$$\big(\mathbf q^{\text{cloth}}_{n+1},\ \dot{\mathbf q}^{\text{proxy}}_{\text{after}}\big)=\mathrm{VBD}\big(\text{cloth},\ \text{proxies},\ \text{contacts},\ \Delta t\big).$$

### Step 6 — Harvest the feedback wrench (`coupling_harvest_proxy_wrenches`)
The wrench the cloth applied to the proxy becomes the **new** $\mathbf F^{\,n}$. The two options below are
**alternatives for the same hook**, picked by polymorphic dispatch on the *destination* solver
(`dst.solver.coupling_harvest_proxy_wrenches`): the generic momentum estimate is the base-class **fallback**;
VBD **overrides** it (capability `BODY_PROXY_HARVEST`) with the explicit-contact sum. **In this MuJoCo + VBD
run the destination is VBD, so only the second path runs — the momentum path never executes here.**

- **Generic (momentum)** — `harvest_proxy_momentum_forces_kernel` (base-class fallback; *not used here*),
  from the proxy's velocity change:
  $$\mathbf f=\frac{m\,(\mathbf v_\text{after}-\mathbf v_\text{before})}{\Delta t},\qquad
  \boldsymbol\tau=\frac{\mathbf R\,\mathbf I\,\mathbf R^{\top}(\boldsymbol\omega_\text{after}-\boldsymbol\omega_\text{before})}{\Delta t},\qquad
  \mathbf F^{\,n}[\text{proxy}]=[\mathbf f;\,\boldsymbol\tau].$$
  (Because Step 3 removed gravity + lagged force, the velocity change is the contact impulse alone.)
- **VBD (explicit contact)** — `_harvest_vbd_*_kernel` (VBD's override; ***used here***): sum the actual contact forces on the proxy body,
  $$\mathbf F^{\,n}[\text{proxy}]=\sum_{\text{contacts }c}\Big[\,\mathbf f_c\ ;\ (\mathbf p_c-\mathbf c_\text{world})\times\mathbf f_c\,\Big],$$
  where $\mathbf f_c$ = the contact force on the proxy at contact $c$, $\mathbf p_c$ = its world contact
  point, and $\mathbf c_\text{world}$ = the proxy body's center of mass in world (`transform_point(body_q, body_com)`).
  So $(\mathbf p_c-\mathbf c_\text{world})\times\mathbf f_c$ is that force's torque about the CoM, and
  $\mathbf F^{\,n}$ is the total 6-DOF contact wrench (force; torque-about-CoM). Summed over
  cloth-particle↔proxy and rigid↔proxy contacts. VBD prefers this so it can keep proxy-vs-proxy
  collisions active inside the solve while feeding back *only* genuine contact forces.

### Step 7 — Blend (`blend_proxy_body_forces_kernel`)
Relax the new harvested wrench against the stashed previous one ($\mathbf F^{\text{prev}}$ = Step 0's stash):
$$\mathbf F^{\,n}\leftarrow \alpha\,\mathbf F^{\,n}_\text{harvested}+(1-\alpha)\,\mathbf F^{\text{prev}},\qquad \alpha=\texttt{proxy\_relaxation}.$$
This is the **relaxation parameter of the staggered coupling** (the under-relaxation / Aitken step of
partitioned FSI). It **does not change the fixed point** — at convergence $\mathbf F^{\,n}=\mathbf F^{\text{prev}}=\mathbf F^{*}$
so the blend returns $\mathbf F^{*}$ for any $\alpha\ne0$ — only the transient and stability: the lagged
iteration's contraction factor (§8.4's $r$) becomes $r_\alpha=(1-\alpha)+\alpha\,r$.

- $\alpha=1$ (**default**, and this scene) — no-op: $\mathbf F^{\,n}=\mathbf F^{\,n}_\text{harvested}$ (the harvest
  is fed back directly, $r_\alpha=r$).
- $\alpha<1$ (**under-relax**) — *shrinks* an oscillatory factor ($r<0$, which an over-scaled proxy $s>1$
  produces, §8.4) and **low-passes** the wrench across substeps ($\mathbf F^{\,n}=\alpha\mathbf F^{\,n}_\text{new}+(1-\alpha)\mathbf F^{\,n-1}$,
  an EMA since `proxy_iterations = 1`) — damps jitter under stiff contact, at the cost of responsiveness.
- $\alpha>1$ (**over-relax**) — SOR-style acceleration, risking $|r_\alpha|>1$ (divergence).

So the blend is a dormant knob (off at $\alpha=1$); turn it to $\alpha<1$ only if the feedback loop
jitters/oscillates.

$\mathbf F^{\,n}$ is then applied to the arm at **Step 1 of the next substep** — that one-substep delay is the *lag*.

```
F^{n-1} ─▶ [0] stash F^prev = F^{n-1}
           [1] arm f_ext += F^{n-1};  step MuJoCo ─▶ arm pose/vel
           [2] sync proxy(pose,vel) ⟵ arm;  snapshot qd_before
           [3] rewind:  proxy f_ext = -F^{n-1} - m g   (cancel gravity + lagged)
           [4] collide cloth ↔ proxy
           [5] step VBD cloth   (proxy mass = mass_scale·m_real)
           [6] harvest:  F^n = m·Δv/dt , R I Rᵀ·Δω/dt   (or Σ contact forces)
           [7] blend:  F^n = α·F^n + (1-α)·F^prev   ─▶  (next substep's lagged force)
```

### Where the "normal" sub-solver steps happen, and the loop nesting

The two real forward steps are the `_step_entry(...)` calls above — they are **not** separate from the
coupling, they are *wrapped* by it. `_step_entry(entry)` calls `entry.solver.step(state_0, state_1,
control, contacts, dt)` (the sub-solver's standard step, looped `entry.substeps`× — here 1):

- **Step 1** is `_step_entry("mjc")` → `SolverMuJoCo.step()` — the **arm forward step** (run *after* the
  lagged force is added to the gripper bodies).
- **Step 5** is `_step_entry("vbd")` → `SolverVBD.step()` — the **cloth forward step** (run *after* sync +
  rewind + proxy collision).

Full nesting per physics step (`runner.simulate`):

```
runner.simulate()                              # one physics step
 ├─ pre_substeps (VBD rebuild_bvh); collide once (outer contacts)
 └─ × num_substeps  (--substeps, =10)
       SolverCoupledProxy.step()               # the coupled substep
        └─ × proxy_iterations (=1)
             _step_proxy:  [1] MuJoCo.step()  …  [5] VBD.step()  + sync/rewind/harvest/blend
```

So per physics step the coupled `step()` runs `num_substeps` (10) times; each runs `proxy_iterations`
(1) `_step_proxy` passes; each pass calls **MuJoCo.step() once and VBD.step() once**. Note `SolverVBD(iterations=20)`
and `SolverMuJoCo(iterations=100, ls_iterations=20)` are *internal* iteration counts **within** one
`.step()` — not additional forward steps.

### Terminology

- **lagged force** $\mathbf F^{\,n-1}$ — the cloth→gripper contact wrench harvested at substep $n{-}1$,
  applied to the arm at substep $n$ (one step behind → "lagged"). It is the channel by which the cloth
  pushes the arm.
- **sync** — copy the arm's pose+velocity onto the proxy body inside the cloth solver (the proxy is a
  kinematic stand-in for the gripper).
- **rewind** — before the cloth solve, pre-subtract the forces already accounted for (gravity + the
  lagged force) from the proxy, so the proxy's motion during the solve reflects **only** new cloth
  contact and the harvest doesn't double-count.
- **harvest** — read off the cloth→proxy wrench after the cloth solve (from momentum change, or summed
  contact forces) — this is the reaction sent back to the arm.
- **blend** — relaxation mix $\alpha\,\text{new}+(1-\alpha)\,\text{old}$ of the harvested wrench with the
  previous one, to stabilize the lagged feedback loop.
- **stash** — save the previous `coupling_forces` before the solve so the blend has the "old" value.
- (**smooth teleportation** — a utility, `smooth_proxy_teleportation_kernel`, that re-encodes a proxy
  pose jump as a velocity so a penetration-free solver's finite-difference velocity isn't corrupted;
  *defined but not used* in this MuJoCo+VBD path.)

---

## 4. `lagged` vs `staggered`

- **`lagged`** — arm is stepped with the *previous* step's force; proxy synced at the **begin** pose;
  more decoupled, one-substep lag, cheapest.
- **`staggered`** — proxy synced at the **just-solved end** pose; `coupling_forces` is zeroed and
  re-harvested within the same step. Tighter (uses the current arm pose) at higher cost.

Both can be wrapped in `proxy_iterations` > 1 (fixed-point relaxation) to tighten the coupling
further.

---

## 5. The force semantics (why it's physical)

- The proxy is a **mass-bearing stand-in** for the gripper in the cloth world (`mass_scale·m`). The
  cloth pushes it during the VBD solve; the **harvested wrench is the contact load that push
  represents** (momentum change ÷ dt, or summed contact forces).
- By Newton's third law that load is what the gripper *should* feel, so it is fed back to the arm as
  an external wrench (mapped into joint torques by MuJoCo).
- The **rewind** step keeps the books straight: the lagged force and gravity already applied to the
  arm are removed from the proxy before the cloth solve, so the harvest captures only the *new* cloth
  interaction, not a re-application of forces already accounted for.

---

## 6. Key parameters (`proxy.py` defaults)

| Parameter | Default | Role |
|---|---|---|
| `mass_scale` | 5.0 | proxy inertia in VBD — how much the cloth perturbs it / how it resists; the proxy's effective mass to the cloth |
| `mode` | `lagged` | `lagged` (begin pose + prev force) vs `staggered` (end pose, re-harvest) |
| `proxy_iterations` | 1 | relaxation passes per physics step (tighter coupling) |
| `proxy_relaxation` | 1.0 | under/over-relaxation when blending harvested force |
| `collide_interval` | 1 | cloth↔proxy collision refresh interval |
| `--vbd-iterations` | 20 | VBD iterations per substep |
| `--mujoco-iterations` | 100 | MuJoCo Newton iterations |
| `substeps` (runner) | 10 | VBD sub-steps per physics step; `decimation` = 1 |

---

## 7. The outer `step()` — what wraps `_step_coupled`

The per-substep math in §3 all lives **inside** `_step_coupled`. The public entry point each substep is
`SolverCoupled.step()`, whose body is five lines. In our `mjc` (arm) + `vbd` (cloth) case:

```python
self._distribute_state(state_in, dt=dt)                        # 1
self._step_coupled(state_in, state_out, control, contacts, dt) # 2
_copy_state(state_in, state_out)                               # 3
self._reconcile_state(state_out)                               # 4
self._entry_output_state_valid = True                          # 5
```

1. **`_distribute_state(state_in, dt=dt)`** — fan the incoming parent `State` *into* each sub-solver's
   private input buffer `entry.state_0`. `_copy_state_to_entry` remaps the global body/particle arrays
   to the entry's local index space (arm bodies → `mjc.state_0`; cloth particles → `vbd.state_0`), then
   `_notify_input_state_update` fires the coupling hook so proxy mappings pick up gravity/`dt` for this
   substep. This is the **scatter-in**: each solver now sees its slice of the world as its `state_0`.

2. **`_step_coupled(...)`** — the template method, **overridden by the proxy solver** to loop
   `proxy_iterations` (=1 here) times over `_step_proxy` — i.e. the entire Step 0–7 algorithm of §3.
   (On iteration `k>0` it re-`_distribute_state`s with `iteration_restart=True` so a relaxation
   restart reuses the original input and only carries the harvested feedback buffers forward.) `state_out`
   is unused here — each entry writes its own `state_1`.

3. **`_copy_state(state_in, state_out)`** — bulk-copy the *whole* input state into the output state, so
   every field **not owned** by any entry (and the unchanged baseline of owned fields) is carried through
   verbatim. Without this, `state_out` would only contain whatever the scatter in step 4 writes.

4. **`_reconcile_state(state_out)`** — the **gather-out**: scatter each entry's post-step `state_1` back
   into the global `state_out` via the local→global index maps (`_scatter_body_state_mapped` for
   `mjc.state_1.body_q/qd` → arm transforms; `_scatter_particle_state_mapped` for `vbd.state_1.particle_q/qd`
   → cloth nodes). After this line `state_out` is the merged result of both solvers.

5. **`_entry_output_state_valid = True`** — bookkeeping flag recording that every entry's `state_1` now
   holds valid post-step data (lets later reads/caches trust the entry buffers instead of recomputing).

So the data path per substep is: **`state_in` → (distribute) → `entry.state_0` → (`_step_proxy`) →
`entry.state_1` → (reconcile) → `state_out`.** The math in §3 is what happens in that middle arrow.

---

**Part III — Why it's correct**

## 8. Why the coupling is correct — the interface problem and the proxy fixed point

§3 says *what* happens; this section says *why it lands on the right answer*. It follows the
"Multi-Solver Coupling Strategies" design note (`Coupling.pdf`, §2–§3.3). Each claim is tagged
**[theory]** (from the note), **[exact]** (verified in our code), or **[approx]** (a deliberate
simplification). This mirrors the structure of the Genesis `ipc_two_way_coupling_math.md` doc.

**The big picture (read this first).** Everything in this section rests on **one** system — the monolithic
interface problem, whose bilateral form is Eq. (1) below:

$$
\underbrace{\mathbf M_a\mathbf v_a=\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda}_{\text{row a — MuJoCo arm}},\qquad
\underbrace{\mathbf M_b\mathbf v_b=\mathbf f_b+\mathbf J_b^\top\boldsymbol\lambda}_{\text{row b — VBD cloth}},\qquad
\underbrace{g(\mathbf u,\boldsymbol\lambda)=\mathbf 0}_{\text{row }g\text{ — interface law}} .
$$

Three facts make the proxy scheme work; the subsections below develop each in detail.

1. **The two dynamics rows are *linearized implicit steps*, not exact dynamics** (Appendix §12). $\mathbf M_a$
   is MuJoCo's `implicitfast` **step matrix** $\mathbf M(\mathbf q)+h(\mathbf K_d+\mathbf D)$ — not the bare
   mass — and $\mathbf M_b$ is VBD's lumped vertex mass plus the elastic Hessian. Eq. (1) is *each solver's own
   local quadratic model, glued at a shared $\boldsymbol\lambda$*, and their **linearity is load-bearing**.
2. **Differencing two iterates cancels $\mathbf f_a$** (§8.3). Because row a is affine with a frozen operator,
   evaluating it at the target impulse and at the lagged one and subtracting removes the entire internal force
   $\mathbf f_a$, leaving a clean relation between the velocity *increment* and the impulse *increment* (Eqs.
   (3)$-$(4)$=$(5)). That is what lets VBD substitute its approximate proxy inertia $\hat{\mathbf M}_a$ into the
   *increment only* — changing the convergence **rate** but never the **fixed point** (§8.4).
3. **The closure $g$ is never assembled** (§8.1). No KKT/LCP/cone solve lives in the coupler; each contact's
   law is produced *inside* the solver that owns it (VBD's penalty contact for gripper $\leftrightarrow$ cloth,
   MuJoCo's constraint solver for its own contacts). Only the products $\mathbf J^\top\boldsymbol\lambda$ cross
   the interface — never $\boldsymbol\lambda$ or $g$ — so the correctness argument is, of necessity,
   **indifferent to $g$** (§8.4).

Operationally this is a **two-tier** coupling (§9.1): a *monolithic* inner solve (cloth $+$ rigid proxy, one
`SolverVBD.step()`) wrapped in a *partitioned, lagged* outer exchange with MuJoCo — the arm runs row a with its
true $\mathbf M_a$ (Eq. (4)), VBD runs row b plus a replica of row a with the approximate $\hat{\mathbf M}_a$
(Eq. (7)), and the shared $\boldsymbol\lambda$ is the only thing crossing, one substep late.

### 8.0 Notation (interface level)

| Symbol | Meaning |
|---|---|
| side $a$ / side $b$ | **source** = `mjc` arm / **destination** = `vbd` cloth |
| $\mathbf v_a,\mathbf v_b$ | generalized velocities owned by each sub-solver this substep |
| $\mathbf M_a,\mathbf M_b$ | each side's step inertia (the local quadratic model it linearizes to) |
| $\mathbf f_a,\mathbf f_b$ | explicit forces / controls / internal terms after time-integration linearization |
| $\mathbf J_a,\mathbf J_b$ | contact/attachment Jacobians at the gripper $\leftrightarrow$ cloth interface |
| $\mathbf u=\mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b$ | interface relative velocity; target $\bar{\mathbf u}$ |
| $\boldsymbol\lambda$ | the **interface impulse** = the contact wrench (impulse units) |
| $\hat{\mathbf M}_a$ | **proxy inertia** of side $a$ used inside side $b$ ($=\texttt{mass\_scale}\cdot m$ on the gripper bodies) |
| superscript $k$ | outer-iteration / substep index of the lagged loop |

#### What these symbols are, concretely (this scene)

The two solvers do **not** share a coordinate system: MuJoCo works in **reduced joint coordinates**, VBD
in **maximal nodal (Cartesian) coordinates**. The interface Jacobians are precisely what reconcile them at
the contact.

| symbol | side $a$ = MuJoCo Franka arm | side $b$ = VBD shirt cloth |
|---|---|---|
| DOFs | 9 joint coords $\mathbf q^{\text{jnt}}$ (7 arm + 2 finger), fixed base | $3N$ nodal coords, $N=6436$ shirt vertices |
| $\mathbf v$ | $\dot{\mathbf q}^{\text{jnt}}\in\mathbb R^{9}$ (joint velocity) | $[\dot{\mathbf x}_1;\dots;\dot{\mathbf x}_N]\in\mathbb R^{19308}$ (vertex velocities) |
| $\mathbf M$ (bare inertia) | **dense** articulated inertia $\mathbf M(\mathbf q)\in\mathbb R^{9\times9}$ (MuJoCo `qM`; couples all joints down the chain) | **diagonal** lumped vertex mass $\operatorname{diag}(m_i\mathbf I_3)\in\mathbb R^{19308\times19308}$ |
| $\mathbf f$ | PD torque $k_p(\mathbf q^{*}-\mathbf q^{\text{jnt}})-k_d\dot{\mathbf q}^{\text{jnt}}$ (gains 4000/400 arm, 4e4/400 finger) + gravity/Coriolis bias | gravity $m_i\mathbf g$ + membrane (Baraff–Witkin) + bending + self-contact internal forces |
| $\mathbf J$ | geometric Jacobian of the gripper contact point(s), $\partial(\text{contact-pt vel})/\partial\dot{\mathbf q}^{\text{jnt}}\in\mathbb R^{c\times9}$ | barycentric pick of the contacting vertices, in the contact frame, $\in\mathbb R^{c\times 19308}$ |

with $c=3\times(\text{number of active gripper}\leftrightarrow\text{cloth contacts})$, interface relative
velocity $\mathbf u\in\mathbb R^{c}$, and impulse $\boldsymbol\lambda\in\mathbb R^{c}$ (the wrench the cloth
puts on the fingers).

> **$\mathbf M$ (bare inertia) vs. $\mathbf M_a$ (step inertia) — don't conflate them.** The row above is the
> *bare* inertia $\mathbf M$. The symbol in the interface system Eq. (1) / §8.1 is the sub-solver's **step
> inertia $\mathbf M_a$** (the *local quadratic model*), which is **built from** $\mathbf M$ but is **not
> equal** to it: for the arm $\mathbf M_a=\mathbf M(\mathbf q)+h(\mathbf K_d+\mathbf D)$ (dense `qM` *plus*
> the $h$-weighted PD/damping Jacobian, §12.2), and for the cloth $\mathbf M_a$ is per-vertex
> $\tfrac{m_i}{\Delta t^{2}}\mathbf I_3+\nabla^2\Psi_i$ (lumped mass *plus* stiffness Hessian, §12.3). To
> **leading order $\mathbf M_a\approx\mathbf M$** — which is why this table lists $\mathbf M$ as the headline
> inertia and why the §8.4 mass-ratio intuition ($s=\hat{\mathbf M}_a/\mathbf M_a$) reads as a *mass* ratio —
> but the exact step matrix carries the extra $h$-damping / stiffness terms (full forms in §12.2/§12.3).

#### The proxy's inertia $\hat{\mathbf M}_a$ — top-down

**The one question it answers:** the proxy is a rigid body inside VBD standing in for the arm — *what inertia
should it carry so the cloth pushes it the way it would push the real, whole, articulated arm?* Three moves:

1. **What we want** — the arm's inertia **reflected to the gripper**: how hard the gripper is to accelerate
   given the whole chain behind it (*not* a finger's local mass).
2. **Why it isn't a copy — the coordinate bridge.** The arm's inertia $\mathbf M_a$ is **joint-space**
   ($9\times9$); the proxy is a **maximal-coordinate** rigid body ($6\times6$) in VBD. You cannot drop one
   into the other — you **pull** the joint inertia back to the body through its Jacobian: a body's effective
   *inverse* inertia is $\mathbf J_\ell\mathbf M^{-1}\mathbf J_\ell^\top$, and its inverse is "the mass felt at
   that body."
3. **What the code does.** MuJoCo already precomputes that pullback (as `body_invweight0`); the proxy turns
   it into a per-body effective mass $M_a^{\text{eff}}$, multiplies by `mass_scale`, and installs the result
   as the proxy body's rigid mass $+$ inertia in VBD — **once, at setup**.

$$
\underbrace{\mathbf M_a}_{\text{arm inertia, joint }9\times9}\ \longrightarrow\ \underbrace{M_a^{\text{eff}}}_{\text{mass felt at a gripper body}}\ \longrightarrow\ \underbrace{\hat{\mathbf M}_a=\texttt{mass\_scale}\cdot M_a^{\text{eff}}}_{\text{proxy inertia in VBD, }6\times6/\text{body}}
\qquad(\text{pull back, then scale}).
$$

**Three caveats** (used in §8.4/§8.6; not re-derived here):

- **A deliberate estimate.** `body_invweight0` uses the **bare** mass at the **home** pose (no PD gains, no
  reconfiguration), so it *undercounts* the stiff PD-held gripper; `mass_scale = 5` inflates it. Since the
  converged answer is **mass-independent** (§8.4), $\hat{\mathbf M}_a$ is a **convergence knob**, not a value
  that must be exact — so the nominal `mass_scale`$=5$ is *not* the theory's contraction scale $s$ (which is
  measured against the true $\mathbf M_a$; §8.6, item 5).
- **Per body, not per contact.** One $6\times6$ inertia per gripper body (hand + 2 fingers); the many cloth
  contacts all act on that one body and are **summed** in the harvest (§8.5).
- **One-shot.** Computed at setup and baked into the VBD proxy body; the per-substep loop just *uses* it.

#### The inertias in play — a glossary

Five distinct inertia objects appear in this doc; they are **not** interchangeable. Keep them straight:

| symbol | name | precisely | space / shape | role |
|---|---|---|---|---|
| $\mathbb M_\ell$ | body spatial inertia | $\operatorname{diag}(\mathbf I_\ell,\,m_\ell\mathbf I_3)$ for one rigid body $\ell$ | maximal, $6\times6$/body | ingredient of $\mathbf M(\mathbf q)$ (§12.2) |
| $\mathbf M(\mathbf q)$ | **bare** articulated mass | $\sum_\ell\mathbf J_\ell^\top\mathbb M_\ell\mathbf J_\ell$ (MuJoCo `qM`); *pure* inertia, **no** actuator gains, config-dependent | joint, $9\times9$ | building block of $\mathbf M_a$ |
| $\mathbf M_a$ | side-$a$ **step inertia** (local quadratic model) | $\mathbf M(\mathbf q)+h(\mathbf K_d+\mathbf D)$ — bare mass **+** implicit PD/damping; the operator in Eqs. (1)–(9) | joint, $9\times9$ | the **true** side-$a$ inertia in the theory |
| $M_a^{\text{eff}}$ | MuJoCo effective-mass **estimate** | pullback of the **bare** $\mathbf M$ to the body, from `body_invweight0` (bare $\mathbf M$, home pose); $=1/\text{inv\_mass}$ here since $\mathbf r{=}0$ — see the note below | **per body**: a scalar mass $+$ a $3\times3$ inertia | what the code feeds the proxy |
| $\hat{\mathbf M}_a$ | **proxy / virtual** inertia | $\texttt{mass\_scale}\cdot M_a^{\text{eff}}$ | maximal, $6\times6$ per gripper body (×3) | the replica's inertia inside VBD |
| $\mathbf M_b$ | side-$b$ (cloth) step inertia | $\operatorname{diag}(m_i\mathbf I_3)$ (+ stiffness Hessian, §12.3) | nodal, $3N\times3N$ | the cloth's inertia |

*(The table's "space" column is the crux of move 2: $\mathbf M_a$ is joint-space $9\times9$; $\hat{\mathbf M}_a$
and $M_a^{\text{eff}}$ are maximal per-body $6\times6$. They are only comparable **after** the Jacobian
pullback — never as raw $9\times9$ vs $6\times6$.)

> *The formula, briefly (for the curious).* `body_invweight0`$=(\text{inv\_mass},\text{inv\_rot})$ is the
> **trace-averaged diagonal** of the pullback $\mathbf J_\ell\mathbf M^{-1}\mathbf J_\ell^\top$ (translational,
> rotational), over the **bare** $\mathbf M$ at the home pose — verified in `mujoco_warp/_src/io.py`
> (`_finalize_body_invweight0`), matching MuJoCo's `mjModel.body_invweight0`. The effective mass **at a point
> offset $\mathbf r$** is $1/\big(\text{inv\_mass}+\tfrac23\,\text{inv\_rot}\lVert\mathbf r\rVert^2\big)$
> (verbatim in `eval_mujoco_coupling_effective_mass_block_kernel`): the second term is that point's *extra*
> translational mobility because an off-center push also spins the body — it does **not** change $\mathbf I$;
> the $\tfrac23$ is a direction average (my reconstruction of the constant). The coupler queries each gripper
> body at its **own origin**, so **$\mathbf r=\mathbf 0$** and it reduces to $1/\text{inv\_mass}$, paired with
> a $3\times3$ rotational inertia (body $\mathbf I_\ell$ rescaled to mean $1/\text{inv\_rot}$) — together the
> proxy body's $6\times6$ (the "scalar $+3\times3$" in the table).
>
> **$\mathbf r=\mathbf 0$ is not an approximation of the lever arm.** It queries the effective inertia at the
> body's *own* frame — precisely what a rigid body's mass+inertia are defined at. The proxy is installed as a
> **full rigid body** with that $(m,\mathbf I)$, so a cloth contact at an offset $\mathbf r_c$ during the VBD
> solve automatically gets the right point response $1/(\tfrac1m+(\mathbf r_c\times\mathbf n)^\top\mathbf I^{-1}(\mathbf r_c\times\mathbf n))$
> — with the *actual* $\mathbf r_c$, produced by the proxy's rigid-body dynamics. The offset lever is handled
> **downstream by VBD**, not by this formula; the $\tfrac23\lVert\mathbf r\rVert^2$ term is only for collapsing
> to a single scalar at an offset, which the proxy never needs. (The real approximations are the trace/3
> isotropization, the home pose, and the missing PD — §8.6, item 5 — plus modeling the reflected arm inertia
> as a free rigid body; *not* $\mathbf r=\mathbf 0$.)

**The Jacobians are never assembled.** Neither $\mathbf J_a$ nor $\mathbf J_b$ is materialized. MuJoCo
receives $\boldsymbol\lambda$ as a 6-DOF wrench **deposited on the gripper body** (`body_f`) and computes
the $\mathbf J_a^\top\boldsymbol\lambda$ joint-torque pullback *inside its own articulated solve*; VBD
receives the contact force **added directly to the contacting cloth vertices**, which is the
$\mathbf J_b^\top\boldsymbol\lambda$ action by construction. So "$\mathbf J^\top\boldsymbol\lambda$" in
§8.1 is realized by each native solver applying a force in its own coordinates — the abstract Jacobians
are only a bookkeeping device for the shared contact space (the note's "Jacobian actions: apply
$\mathbf J^\top\mathbf f$ without materializing $\mathbf J$").

### 8.1 The exact thing being approximated — the monolithic coupled solve [theory]

> *Equation numbering.* §8.1–§8.4 number their milestone equations **(1)–(9)** (shown at the right
> margin); cross-references use those. Where one coincides with the design note's equation, that is noted as
> "(≡ note Eq. M)". References to note-only equations not displayed here keep the explicit "note Eq. M".

If arm + cloth were solved as **one** system, one Newton step couples **three relations** — the two sides'
linearized dynamics, sharing a single interface impulse $\boldsymbol\lambda$, plus an **interface law** $g$
that closes them:

$$
\mathbf M_a\mathbf v_a=\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda,\qquad
\mathbf M_b\mathbf v_b=\mathbf f_b+\mathbf J_b^\top\boldsymbol\lambda,\qquad
\underbrace{g(\mathbf u,\boldsymbol\lambda)=\mathbf 0}_{\text{interface law}},\quad \mathbf u=\mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b .
$$

The first two rows are **linear**: each side integrates under its own forces *plus the same interface
impulse $\boldsymbol\lambda$*, opposite sign. They are identical no matter what the contact is, and carry
**all** the proxy/mass content of §8.3–§8.4. The third row, the **interface law $g$**, is the *only* place
the contact model enters; it subsumes every case we need:

| interface law | $g(\mathbf u,\boldsymbol\lambda)=\mathbf 0$ | where |
|---|---|---|
| bilateral attachment | $\mathbf u-\bar{\mathbf u}$ | stick / weld |
| frictionless unilateral | $0\le(u_n-\bar u_n)\perp\lambda_n\ge0$ | Signorini (cloth can only push) |
| compliant / penalty | $\mathbf u-\bar{\mathbf u}+\mathbf C\boldsymbol\lambda$ | finite stiffness, compliance $\mathbf C$ (note Eq. 4) — *what VBD does* |
| Coulomb friction | $\boldsymbol\lambda\in\mathcal K(\lambda_n),\ \mathbf u_t\!\perp\!\partial\mathcal K$ | max-dissipation on the friction cone $\mathcal K$ |

**What $\mathbf u$ and $\bar{\mathbf u}$ are.** $\mathbf u=\mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b$ is the
**interface relative velocity** — gripper (proxy) surface velocity *minus* cloth velocity at a contact,
resolved in the contact frame: a normal $u_n$ (approach/separation rate) and two tangential $u_t$
(sliding); the signs in $\mathbf J_a,\mathbf J_b$ make it that *relative* quantity. $\bar{\mathbf u}$ is a
**per-law bias** appearing inside specific $g$'s: $\mathbf 0$ for a stick weld, or a Baumgarte/restitution
term $\bar u_n=-\tfrac{\beta}{\Delta t}(\text{penetration})$ for contact. The complementarity/cone rows are
not a smooth $g$; the unifying statement is the **inclusion** $-\boldsymbol\lambda\in\partial\phi(\mathbf u)$
for a convex contact potential $\phi$, with $g(\mathbf u,\boldsymbol\lambda)=\mathbf 0$ the smooth-case
shorthand. (Pure $g(\mathbf u)=0$ reaches only the bilateral row; contact and friction couple $\mathbf u$
**and** $\boldsymbol\lambda$.)

**Bilateral special case $\Rightarrow$ the KKT matrix.** When $g=\mathbf u-\bar{\mathbf u}$ — and *only*
then — all three rows are linear and assemble into the saddle system (≡ note Eq. 2):

$$
\begin{bmatrix}\mathbf M_a & \mathbf 0 & -\mathbf J_a^\top\\ \mathbf 0 & \mathbf M_b & -\mathbf J_b^\top\\ -\mathbf J_a & -\mathbf J_b & \mathbf 0\end{bmatrix}
\begin{bmatrix}\mathbf v_a\\ \mathbf v_b\\ \boldsymbol\lambda\end{bmatrix}
=\begin{bmatrix}\mathbf f_a\\ \mathbf f_b\\ -\bar{\mathbf u}\end{bmatrix}. \tag{1}
$$

**Why the row $\mathbf M_a\mathbf v_a=\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda$ holds.** It is just
Newton's second law in generalized coordinates over one implicit step, with the contact reaction added by
the principle of virtual work:

1. *Unconstrained step.* Solved alone, side $a$'s linearized step is $\mathbf M_a\mathbf v_a=\mathbf f_a$,
   i.e. $\mathbf v_a=\mathbf M_a^{-1}\mathbf f_a$ — the end-of-step velocity produced by all of side $a$'s
   own terms (PD torque, gravity/Coriolis bias, carried-in momentum) against its step inertia
   $\mathbf M_a$. These $\mathbf M_a,\mathbf f_a$ are the local quadratic model the solver *already* builds
   for its own implicit integration.
2. *Constraint reaction by duality.* The contact acts in **contact space** with impulse
   $\boldsymbol\lambda$, doing work only through the interface velocity: power
   $=\boldsymbol\lambda^\top\mathbf u=\boldsymbol\lambda^\top(\mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b)$.
   The generalized force this exerts on side $a$ (conjugate to $\mathbf v_a$) is therefore
   $\partial(\boldsymbol\lambda^\top\mathbf u)/\partial\mathbf v_a=\mathbf J_a^\top\boldsymbol\lambda$. The
   **transpose is not a coincidence**: it is the adjoint of the velocity map
   $\mathbf v_a\mapsto\mathbf J_a\mathbf v_a$, and using it guarantees the impulse injects exactly the right
   energy across the interface (no spurious work). Adding this to (1) gives the row.

The virtual-work argument is **independent of $g$**: $\mathbf J_a^\top\boldsymbol\lambda$ is the generalized
force any interface impulse $\boldsymbol\lambda$ exerts on side $a$, whatever law sets $\boldsymbol\lambda$.
The closure $g$ only decides *which* $\boldsymbol\lambda$. In the **bilateral** case it is the **KKT
stationarity** of a constrained minimization: minimize each side's incremental energy
$\tfrac12\mathbf v_a^\top\mathbf M_a\mathbf v_a-\mathbf v_a^\top\mathbf f_a$ (plus side $b$'s) subject to
$\mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b=\bar{\mathbf u}$; the multiplier of that equality is
$\boldsymbol\lambda$, and $\partial/\partial\boldsymbol\lambda$ recovers the closure
$\mathbf u=\bar{\mathbf u}$ — i.e. **bilateral $g$ makes $\boldsymbol\lambda$ a Lagrange multiplier**. For a
general $g$ (Signorini, friction, penalty) $\boldsymbol\lambda$ is instead the impulse selected by
$g(\mathbf u,\boldsymbol\lambda)=\mathbf 0$ (a complementarity/cone solution, or the gradient of a contact
potential), but the two dynamics rows — and hence everything in §8.2–§8.4 — are untouched.

Row 1, $\mathbf M_a\mathbf v_a=\mathbf f_a$, is **one (linearized) implicit Euler step of the arm**, and
$\mathbf M_a$ is *not* the bare mass matrix but the solver's **step matrix** (mass + $\Delta t$-weighted
stiffness/damping). For the full derivation — what exactly is being linearized, the Newton step, the
variational/incremental-potential view, and the **detailed MuJoCo `implicitfast` form** (total force,
$\partial\mathbf f/\partial\dot{\mathbf q}$, $\partial\mathbf f/\partial\mathbf v$, $\partial\mathbf f/\partial\mathbf q$) —
see **§12**.

The shared $\boldsymbol\lambda$ is exactly what makes action $=$ reaction. Eliminating
$\mathbf v_a,\mathbf v_b$ from the two linear dynamics rows gives the contact-space (Delassus) form
(Eq. (2) below; ≡ note Eq. 3)

$$
\mathbf u=\mathbf D\boldsymbol\lambda+\mathbf u^0,\quad
\mathbf D=\mathbf J_a\mathbf M_a^{-1}\mathbf J_a^\top+\mathbf J_b\mathbf M_b^{-1}\mathbf J_b^\top,\quad
\mathbf u^0=\mathbf J_a\mathbf M_a^{-1}\mathbf f_a+\mathbf J_b\mathbf M_b^{-1}\mathbf f_b, \tag{2}
$$

so the **monolithic contact problem** is this relation **closed by the interface law**:

$$
\mathbf u=\mathbf D\boldsymbol\lambda+\mathbf u^0
\quad\text{together with}\quad
g(\mathbf u,\boldsymbol\lambda)=\mathbf 0.
$$

(Bilateral $g=\mathbf u-\bar{\mathbf u}$ closes it explicitly: $\bar{\mathbf u}=\mathbf D\boldsymbol\lambda+\mathbf u^0\Rightarrow\boldsymbol\lambda=\mathbf D^{-1}(\bar{\mathbf u}-\mathbf u^0)$; Signorini gives an LCP, friction a cone solve, penalty a smooth root-find.) The Delassus form $\mathbf D,\mathbf u^0$ comes
**entirely from the linear dynamics** and is independent of $g$ — $g$ enters only as the closure. **This
monolithic solve is the gold standard.** The proxy scheme reaches the same
$(\mathbf v_a,\mathbf v_b,\boldsymbol\lambda)$ using two *separate* native solvers that never assemble it —
and, as §8.4 shows, the correctness of that is **indifferent to which $g$ is used**.

> **What kind of contact model is this — and is it IPC?** §8.1 works at the **constraint / impulse level**:
> the interface law $g(\mathbf u,\boldsymbol\lambda)=\mathbf 0$ is a velocity-level closure selecting the
> impulse $\boldsymbol\lambda$ (bilateral $\Rightarrow$ a Lagrange multiplier; Signorini $\Rightarrow$ an
> LCP; etc.). It is **not** an IPC-style **log-barrier** contact — there is no interior-point barrier, no
> penetration-free invariant, and no CCD line search
> anywhere in this proxy path (that machinery is the *Genesis/libuipc* approach, contrasted in §9; here
> interpenetration is possible and merely generates a restoring force). And the actual implementation is not
> even a *hard* multiplier: the gripper $\leftrightarrow$ cloth $\boldsymbol\lambda$ is realized by **VBD's
> penalty / compliant contact** — a one-sided spring that switches on only once `penetration_depth > 0`,
> with force $=$ `contact_ke`$\cdot$penetration $+$ `kd` damping $+$ Coulomb $\mu$ (and MuJoCo's *own*
> internal contacts $\mathbf J_c^\top\boldsymbol\lambda_c$ use its regularized-constraint solver). So the
> $\boldsymbol\lambda$ that is harvested and fed back is a **penalty contact force**, closer to the note's
> finite-stiffness form (note Eq. 4, $E_c=\tfrac{\kappa}{2}\lVert\mathbf u-\bar{\mathbf u}\rVert^2$) than to
> the hard Eq. (1). Three distinct models, then: **hard multiplier** (§8.1's idealization, for reasoning about the
> fixed point) — **penalty/compliant** (what VBD actually solves) — **log-barrier IPC** (Genesis, *not* used
> here). The hard-constraint picture is used only because the correctness/convergence argument (§8.3–§8.5) is
> cleanest in it; the penalty realization approximates that same $\boldsymbol\lambda$.

### 8.2 Why not just exchange colliders [theory]

The cheapest scheme (note §3.1) lets each solver see the other as kinematic geometry and run its own
contact solve. Then each side computes its **own, private** $\boldsymbol\lambda$ — the two are not
equal-and-opposite, momentum is not conserved across the interface, and the combined complementarity is
never one problem. The proxy scheme fixes this by making the gripper a **finite-mass replica inside the
cloth solver** and harvesting the cloth's *actual* contact wrench, so a single $\boldsymbol\lambda$ flows
to both sides.

### 8.3 The proxy / virtual-inertia construction [theory]

**The problem.** We want the monolithic solution of §8.1, but the destination solver (side $b$ = VBD) does
**not** have side $a$'s true articulated inertia $\mathbf M_a$ — it cannot evaluate the side-$a$ row
$\mathbf M_a\mathbf v_a=\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda$ itself. The proxy idea: represent
side-$a$ bodies inside side $b$ by **replica DOFs with an approximate inertia $\hat{\mathbf M}_a$** (note
§3.3), solve side $b$ with that replica present, and iterate so the replica's interface impulse converges
to the true one. The intermediate steps:

> **The essential picture: two solvers run two *separate* systems in parallel, coupled only through the
> shared impulse $\boldsymbol\lambda$.** There is *no* combined matrix — each coupler iteration $k$ runs two
> independent native solves:
>
> | solver | system it actually solves | inertia | driven by | output |
> |---|---|---|---|---|
> | **MuJoCo** | **Eq. (4)**, the *real* arm | true $\mathbf M_a$ | lagged $\boldsymbol\lambda^{k}$ (applied, §3 Step 1) | arm velocity $\mathbf v_a^{k}$ |
> | **VBD** | **Eq. (7)**, cloth $+$ proxy replica | approximate $\hat{\mathbf M}_a$ | closure $g$ (§3 Steps 4–6) | cloth $\mathbf v_b$, **new** $\boldsymbol\lambda^{k+1}$ |
>
> The two systems are never assembled together and never solved simultaneously. Their *only* handshake is
> $\boldsymbol\lambda$: VBD **produces** $\boldsymbol\lambda^{k+1}$ (harvested, §3 Step 6), MuJoCo
> **consumes** $\boldsymbol\lambda^{k}$ (applied, §3 Step 1) — the one-substep lag. Everything below (Eqs.
> (5)–(6)) is **not** a third system; it is the *bridge* derived from (3)/(4) that tells the VBD-side replica
> how to respond, so that at a fixed point the two independent solves agree
> ($\boldsymbol\lambda^{k+1}=\boldsymbol\lambda^{k}$) and jointly reproduce the monolithic §8.1 solution.

**Step 1 — the two side-$a$ rows.** The *target* side-$a$ row (from the monolithic system §8.1) is

$$
\mathbf M_a\mathbf v_a=\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda. \tag{3}
$$

But side $a$ has *already been advanced* once this iteration, **by MuJoCo, its own native solver** — with
the arm's true inertia $\mathbf M_a$, under the **previous** interface impulse $\boldsymbol\lambda^{k}$ (the
lagged wrench applied in §3 Step 1) — and that is what produced its current velocity (≡ note Eq. 6):

$$
\mathbf M_a\mathbf v_a^{k}=\mathbf f_a+\mathbf J_{a}^\top\boldsymbol\lambda^{k}. \tag{4}
$$

**This is the first of the two parallel systems: (4) is exactly what MuJoCo solves**, on its own, for the
real arm. It is a *given* by the time the VBD side runs — $\mathbf v_a^{k}$ is already on the table, not an
unknown of the system below.

**Step 2 — work with the *increment*, to cancel $\mathbf f_a$.** Subtract (4) from (3). The
solver-internal force $\mathbf f_a$ (gravity, PD, bias) drops out, leaving a relation purely between the
velocity change and the **change in interface impulse**:

$$
\mathbf M_a\big(\mathbf v_a-\mathbf v_a^{k}\big)=\mathbf J_a^\top\boldsymbol\lambda-\mathbf J_{a}^\top\boldsymbol\lambda^{k}. \tag{5}
$$

**Step 3 — swap in the proxy inertia (the one approximation).** Side $b$ does not have $\mathbf M_a$, so
replace it by the replica inertia $\hat{\mathbf M}_a$ in (5) — this is the *only* substitution (≡ note
Eq. 7–8), and §8.4 shows it affects the convergence rate but not the fixed point:

$$
\hat{\mathbf M}_a\big(\mathbf v_a-\mathbf v_a^{k}\big)=\mathbf J_a^\top\boldsymbol\lambda-\mathbf J_{a}^\top\boldsymbol\lambda^{k}
\qquad\Longleftrightarrow\qquad
\mathbf v_a=\mathbf v_a^{k}+\hat{\mathbf M}_a^{-1}\big(\mathbf J_a^\top\boldsymbol\lambda-\mathbf J_{a}^\top\boldsymbol\lambda^{k}\big) \tag{6}
$$

Read (6): the replica starts at the velocity side $a$ *actually* reached, $\mathbf v_a^{k}$ (which already
contains $\boldsymbol\lambda^{k}$), and moves only by the response, through $\hat{\mathbf M}_a$, to the
**increment** $\mathbf J_a^\top\boldsymbol\lambda-\mathbf J_a^\top\boldsymbol\lambda^{k}$. If we instead used
the full $\mathbf J_a^\top\boldsymbol\lambda$ here, $\boldsymbol\lambda^{k}$ would be applied **twice** — once
on the real arm in (4) and again on the replica — so the $-\mathbf J_a^\top\boldsymbol\lambda^{k}$ is the
bookkeeping that removes the double count.

**Step 4 — assemble the destination system: proxy dynamics + the *same* closure $g$.** Move the knowns to
the right in (6), and keep side $b$'s own row and the **interface law** $g$ exactly as in §8.1 — only the
side-$a$ row changed ($\mathbf M_a\!\to\!\hat{\mathbf M}_a$, plus the rewind on the RHS):

$$
\underbrace{\hat{\mathbf M}_a\mathbf v_a-\mathbf J_a^\top\boldsymbol\lambda=\hat{\mathbf M}_a\mathbf v_a^{k}-\mathbf J_a^\top\boldsymbol\lambda^{k}}_{\text{proxy side-}a\text{ row}},\qquad
\mathbf M_b\mathbf v_b=\mathbf f_b+\mathbf J_b^\top\boldsymbol\lambda,\qquad
g(\mathbf u,\boldsymbol\lambda)=\mathbf 0 . \tag{7}
$$

**This is the second of the two parallel systems: (7) is exactly what VBD solves**, and it is a *different*
system from MuJoCo's (4) — a different inertia ($\hat{\mathbf M}_a$, not $\mathbf M_a$), a different set of
DOFs (cloth vertices $+$ a gripper *replica*, not the articulated arm), and a different driver (the contact
closure $g$, not a prescribed wrench). The two share nothing but $\boldsymbol\lambda$; here VBD **outputs** a
fresh $\boldsymbol\lambda^{k+1}$ that will drive MuJoCo's (4) at the next iteration.

The closure $g$ is **untouched** — only the dynamics inertia changed. When $g$ is the bilateral law the three
rows are linear and this is the saddle system side $b$ (VBD) solves (≡ note Eq. 9):

$$
\begin{bmatrix}\hat{\mathbf M}_a & \mathbf 0 & -\mathbf J_a^\top\\ \mathbf 0 & \mathbf M_b & -\mathbf J_b^\top\\ -\mathbf J_a & -\mathbf J_b & \mathbf 0\end{bmatrix}
\begin{bmatrix}\mathbf v_a\\ \mathbf v_b\\ \boldsymbol\lambda\end{bmatrix}
=\begin{bmatrix}\hat{\mathbf M}_a\mathbf v_a^{k}-\mathbf J_a^\top\boldsymbol\lambda^{k}\\ \mathbf f_b\\ -\bar{\mathbf u}\end{bmatrix}.
$$

So the proxy problem is the §8.1 monolithic one with **only two changes**, both in the dynamics:
$\mathbf M_a\!\to\!\hat{\mathbf M}_a$ and the matching rewind RHS. The solved $\boldsymbol\lambda$ becomes
$\boldsymbol\lambda^{k+1}$, fed to side $a$ next iteration. **The $-\mathbf J_a^\top\boldsymbol\lambda^{k}$
term is the "rewind"** (§3 Step 3); at a fixed point
$\boldsymbol\lambda^{k+1}=\boldsymbol\lambda^{k}\Rightarrow\mathbf v_a=\mathbf v_a^{k}$, so (5)/(6)
collapse back to the exact row (3) — the converged solution is the monolithic one **regardless of
$\hat{\mathbf M}_a$ and regardless of $g$** (§8.4/§8.5).

> *What the superscript $k$ is.* $k$ indexes the **coupler (outer) iteration** — *not* the physics time
> step, and *not* the internal MuJoCo/VBD iterations. One coupler iteration is one `_step_proxy` pass
> (§3, Steps 0–7), and each such pass contains a **full** MuJoCo solve (its `mujoco_iterations = 100`
> internal Newton iterations) **and** a **full** VBD solve (`vbd_iterations = 20` Gauss–Seidel sweeps) — so
> $\mathbf v_a^{k}$ is the *converged* output of the previous MuJoCo solve, not a single iterate. The
> nesting, innermost to outermost: MuJoCo/VBD internal iterations $\subset$ one coupler iteration
> (`_step_proxy`) $\subset$ substep (`num_substeps = 10`) $\subset$ physics step. With the default
> `proxy_iterations = 1` there is exactly **one** coupler iteration per substep, so the "previous outer
> iteration" of Step 1 *is* the previous substep — i.e. $k$ effectively indexes substeps and
> $\boldsymbol\lambda^{k}$ is literally last substep's harvested wrench (the one-substep *lag* of §3/§8.6).
> The note writes "previous outer iteration **or** previous time step" precisely because the two coincide
> when `proxy_iterations = 1`.

> *Frozen-Jacobian note.* The note writes the previous-step Jacobian as $\mathbf J_{a,k}$ and the current
> as $\mathbf J_a$; here (as in the implementation) the interface geometry is held fixed across the one
> iteration, $\mathbf J_{a,k}=\mathbf J_a$, which is what lets $\mathbf f_a$ cancel cleanly in Step 2.

### 8.4 Convergence — the proxy mass is a relaxation knob, not a physical mass [theory]

Iterating §8.3 is a fixed-point map $\boldsymbol\lambda^{k}\mapsto\boldsymbol\lambda^{k+1}$: eliminate
$\mathbf v_a,\mathbf v_b$ from the proxy **dynamics** rows (giving a contact-space relation with **no $g$**),
then **close with the interface law $g$** and read off the new impulse. Because the elimination is pure
dynamics, $g$ enters only at the closure — the source of the law-independence below.

**Step 1 — eliminate the velocities → the proxy contact-space relation (no $g$ yet).** From the proxy
side-$a$ row (§8.3, iv), $\mathbf v_a=\mathbf v_a^{k}+\hat{\mathbf M}_a^{-1}\mathbf J_a^\top(\boldsymbol\lambda-\boldsymbol\lambda^{k})$;
from side $b$, $\mathbf v_b=\mathbf M_b^{-1}(\mathbf f_b+\mathbf J_b^\top\boldsymbol\lambda)$. Form
$\mathbf u=\mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b$ and substitute
$\mathbf v_a^{k}=\mathbf M_a^{-1}(\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda^{k})$ (Eq. (4)):

$$
\mathbf u=\mathbf J_a\mathbf M_a^{-1}(\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda^{k})+\mathbf J_a\hat{\mathbf M}_a^{-1}\mathbf J_a^\top(\boldsymbol\lambda-\boldsymbol\lambda^{k})+\mathbf J_b\mathbf M_b^{-1}(\mathbf f_b+\mathbf J_b^\top\boldsymbol\lambda).
$$

Now collect the three groups — the coefficient of $\boldsymbol\lambda$, the coefficient of $\boldsymbol\lambda^{k}$, and the constant force term:

- **of $\boldsymbol\lambda$:** $\mathbf J_a\hat{\mathbf M}_a^{-1}\mathbf J_a^\top+\mathbf J_b\mathbf M_b^{-1}\mathbf J_b^\top=\hat{\mathbf D}$ (the proxy Delassus);
- **of $\boldsymbol\lambda^{k}$:** $+\mathbf J_a\mathbf M_a^{-1}\mathbf J_a^\top$ from $\mathbf v_a^{k}$ (which carries the *true* $\mathbf M_a$) *minus* $\mathbf J_a\hat{\mathbf M}_a^{-1}\mathbf J_a^\top$ from the increment's $-\boldsymbol\lambda^{k}$, i.e. $\mathbf J_a(\mathbf M_a^{-1}-\hat{\mathbf M}_a^{-1})\mathbf J_a^\top$ — the one term where the true and proxy inertias **fail to cancel** (it vanishes iff $\hat{\mathbf M}_a=\mathbf M_a$);
- **constant:** $\mathbf J_a\mathbf M_a^{-1}\mathbf f_a+\mathbf J_b\mathbf M_b^{-1}\mathbf f_b=\mathbf u^0$.

That is exactly

$$
\mathbf u=\hat{\mathbf D}\,\boldsymbol\lambda+\mathbf b^{k},\qquad
\hat{\mathbf D}=\mathbf J_a\hat{\mathbf M}_a^{-1}\mathbf J_a^\top+\mathbf J_b\mathbf M_b^{-1}\mathbf J_b^\top,\qquad
\mathbf b^{k}=\mathbf u^0+\mathbf J_a\big(\mathbf M_a^{-1}-\hat{\mathbf M}_a^{-1}\big)\mathbf J_a^\top\boldsymbol\lambda^{k},
$$

with $\mathbf u^0=\mathbf J_a\mathbf M_a^{-1}\mathbf f_a+\mathbf J_b\mathbf M_b^{-1}\mathbf f_b$ (as in §8.1).
This is a **local linear compliance model** — proxy Delassus $\hat{\mathbf D}$, offset $\mathbf b^{k}$
carrying the previous impulse — and it is **independent of the interface law**.

**Step 2 — close with $g$ to get the update map.** Solve the Step-1 relation together with the closure:

$$
\boldsymbol\lambda^{k+1}=\Phi(\boldsymbol\lambda^{k}):\quad\text{solve}\ \big\{\,\mathbf u=\hat{\mathbf D}\boldsymbol\lambda+\mathbf b^{k},\ \ g(\mathbf u,\boldsymbol\lambda)=\mathbf 0\,\big\}.
$$

(Bilateral $\Rightarrow$ a linear solve; Signorini $\Rightarrow$ an LCP; friction $\Rightarrow$ a cone
solve; the penalty contact VBD uses $\Rightarrow$ a smooth root-find.) For the **bilateral** case
$g=\mathbf u-\bar{\mathbf u}$ (the weld: the two attached points move together, so their **relative** velocity
equals the prescribed $\bar{\mathbf u}$ — zero for a fixed weld) the solve is explicit. **Substitute the
closure $\mathbf u=\bar{\mathbf u}$ into the Step-1 mobility relation** to eliminate $\mathbf u$, then invert:

$$
\bar{\mathbf u}=\hat{\mathbf D}\,\boldsymbol\lambda^{k+1}+\mathbf b^{k}
\;\Longrightarrow\;
\hat{\mathbf D}\,\boldsymbol\lambda^{k+1}=\bar{\mathbf u}-\mathbf b^{k}
\;\Longrightarrow\;
\boldsymbol\lambda^{k+1}=\hat{\mathbf D}^{-1}\big(\bar{\mathbf u}-\mathbf b^{k}\big),
$$

the last step using that $\hat{\mathbf D}=\mathbf J_a\hat{\mathbf M}_a^{-1}\mathbf J_a^\top+\mathbf J_b\mathbf M_b^{-1}\mathbf J_b^\top$
is symmetric positive-definite — hence invertible — on the active-contact directions. (This $\boldsymbol\lambda^{k+1}$
is the impulse that makes the proxy dynamics deliver exactly the weld velocity $\bar{\mathbf u}$; it is fed to
the arm next iteration.) Now expand $\mathbf b^{k}=\mathbf u^0+\mathbf J_a(\mathbf M_a^{-1}-\hat{\mathbf M}_a^{-1})\mathbf J_a^\top\boldsymbol\lambda^{k}$
and flip the sign of the $\boldsymbol\lambda^{k}$ coefficient
($-(\mathbf M_a^{-1}-\hat{\mathbf M}_a^{-1})=\hat{\mathbf M}_a^{-1}-\mathbf M_a^{-1}$) to get the **affine map**

$$
\boldsymbol\lambda^{k+1}=\underbrace{\hat{\mathbf D}^{-1}\mathbf J_a\big(\hat{\mathbf M}_a^{-1}-\mathbf M_a^{-1}\big)\mathbf J_a^\top}_{=\ \mathbf G\ \text{(iteration matrix; }\equiv\text{ note Eq. 10)}}\,\boldsymbol\lambda^{k}+\hat{\mathbf D}^{-1}(\bar{\mathbf u}-\mathbf u^0), \tag{8}
$$

which converges iff the spectral radius $\rho(\mathbf G)<1$. For a general $g$, $\Phi$ is nonlinear and
$\mathbf G$ is its linearization on a fixed active set (Step 4).

**Step 3 — the fixed point is the monolithic solution, for *any* $g$ and *any* $\hat{\mathbf M}_a$.** At a
fixed point $\boldsymbol\lambda^{k+1}=\boldsymbol\lambda^{k}=\boldsymbol\lambda^{*}$, the offset $\mathbf b^{k}$
uses $\boldsymbol\lambda^{*}$, so the Step-1 relation gives

$$
\mathbf u^{*}=\hat{\mathbf D}\boldsymbol\lambda^{*}+\mathbf u^0+\mathbf J_a(\mathbf M_a^{-1}-\hat{\mathbf M}_a^{-1})\mathbf J_a^\top\boldsymbol\lambda^{*}
=\mathbf u^0+\underbrace{\big(\mathbf J_a\mathbf M_a^{-1}\mathbf J_a^\top+\mathbf J_b\mathbf M_b^{-1}\mathbf J_b^\top\big)}_{=\ \mathbf D\ \text{(true Delassus)}}\boldsymbol\lambda^{*},
$$

the **$\hat{\mathbf M}_a$ terms cancelling** ($\hat{\mathbf D}+\mathbf J_a(\mathbf M_a^{-1}-\hat{\mathbf M}_a^{-1})\mathbf J_a^\top=\mathbf D$). Together with the **same** closure, this is

$$
\mathbf u^{*}=\mathbf u^0+\mathbf D\boldsymbol\lambda^{*}\qquad\text{and}\qquad g(\mathbf u^{*},\boldsymbol\lambda^{*})=\mathbf 0,
$$

**exactly the monolithic contact problem of §8.1** (true Delassus $+$ true closure). So $\hat{\mathbf M}_a$
sits in the *rate* $\mathbf G$ but **cancels in the answer** $\boldsymbol\lambda^{*}$ — and the cancellation
lives in the $\mathbf u$–$\boldsymbol\lambda$ relation, *not* in $g$, so it holds for **every** interface
law. (Bilateral special case: $\boldsymbol\lambda^{*}=\mathbf D^{-1}(\bar{\mathbf u}-\mathbf u^0)$.) This is
the precise sense in which closing with $\mathbf u=\bar{\mathbf u}$ in the derivation costs nothing: the
converged $\boldsymbol\lambda$ is correct for whatever $g$ the solver actually uses.

**Step 4 — what $g$ changes: the rate, not the fixed point.** With a general $g$ the update $\Phi$ is
nonlinear. On a **fixed active set** with smooth/linearized $g$, its local factor is $\mathbf G$ restricted
to the active contacts (frictionless Signorini active contacts behave bilaterally; inactive ones decouple
with $\lambda=0$), so the linear analysis — and $r(s,q)$ below — describes the local contraction there. The
genuinely new ingredient for general $g$ is **active-set / stick–slip switching** (a contact opening or
closing, friction flipping stick $\leftrightarrow$ slip) — nonsmooth and combinatorial, not captured by the
spectral radius of any single $\mathbf G$ (the same flavor as the moving-$\mathbf J$ caveat below).

**Step 5 — scalar reduction.** For a scalar interface ($\mathbf J_a=\mathbf J_b=1$, all inertias scalars),
$\mathbf G$ is the number

$$
r=\frac{\hat{\mathbf M}_a^{-1}-\mathbf M_a^{-1}}{\hat{\mathbf M}_a^{-1}+\mathbf M_b^{-1}}.
$$

Write the **virtual-inertia scale** $s=\hat{\mathbf M}_a/\mathbf M_a$ and the **mass ratio**
$q=\mathbf M_a/\mathbf M_b$; factoring $\mathbf M_a^{-1}$ from numerator and denominator,
$r=\dfrac{1/s-1}{1/s+q}$, and multiplying through by $s$ (≡ note Eq. 11):

$$
r(s,q)=\frac{1-s}{1+qs}. \tag{9}
$$

This is the heart of the correctness argument:

- $s=1$ (proxy inertia $=$ source's *true* effective inertia at the interface) $\Rightarrow r=0$:
  converges in **one** iteration — exactly the monolithic solution.
- $0<s<1$ (proxy lighter than truth) $\Rightarrow 0<r<1$ for any $q$: **contractive**, geometric
  convergence; $r\to1$ (slow) as $s\to0$.
- $s>1$ (proxy heavier) $\Rightarrow r<0$; **$|r|<1$ iff $s(1-q)<2$**. So for $q=M_a/M_b\ge1$ (cloth
  lighter than the arm's effective mass) a heavy proxy is contractive for **every** $s>1$; it goes
  non-contractive only when $q<1$ (cloth as heavy as / heavier than the arm) and $s$ is large.

The crucial point: **$\hat{\mathbf M}_a$ enters only the error-propagation matrix, never the fixed
point.** Any contractive $\hat{\mathbf M}_a$ converges to the *same* $\boldsymbol\lambda$ — the
monolithic solution of §8.1. So `mass_scale` is a **convergence / relaxation knob, not a physical
quantity that biases the answer** (note: "treat the proxy mass scale as a relaxation parameter rather
than a precise physical mass").

> **Then why `mass_scale = 5` ($s>1$) rather than $0<s\le1$?** $s=1$ is the optimal ($r=0$, one step) and
> $0<s<1$ is also contractive, so the choice looks paradoxical — but four things make over-scaling the right
> call here:
> 1. **No instability in this regime.** $s>1$ is unstable only for $q<1$ (a heavy counterpart); the shirt is
>    light, so $q\gg1$ and $s=5$ gives $r=-4/(1+5q)\approx0$ — firmly contractive (see the $s>1$ bullet).
> 2. **One *lagged* iteration, not the asymptotic limit.** With `proxy_iterations = 1` we take a single step
>    per substep, so "$0<s<1$ eventually converges" barely applies — what matters is the quality of *one*
>    step. A **light** proxy ($s<1$) gets shoved by the cloth *within* that solve: it drifts off the synced
>    (correct) arm pose and under-reads the contact. A **heavy** proxy stays glued to the synced gripper and
>    returns a clean contact wrench. With one shot, over-damped beats under-damped.
> 3. **The fixed point is mass-independent** (explicit-contact harvest, §8.5/Step 3), so $s$ cannot bias the
>    answer — only the rate/stability — and erring heavy is the safe direction when $q\gg1$.
> 4. **$M_a^{\text{eff}}$ undercounts the real resistance.** `body_invweight0` is the *bare mechanism*
>    inertia (home config, **no actuator gains**), yet the gripper is held by a very stiff PD ($k_p$ up to
>    $4\times10^4$, $k_d=400$) and resists motion far more than its bare inertia. So `mass_scale > 1` is
>    partly *compensation* toward the gripper's true controlled stiffness — the effective $s$ measured
>    against the real (PD-laden) resistance is well below the nominal $5$. `mass_scale = 5` is an empirical
>    robustness margin, not a dive into the unstable regime.

> *Are $\mathbf J_a,\mathbf J_b$ constant across the map? — the frozen-Jacobian assumption.* The derivation
> above treats $\mathbf J_a,\mathbf J_b$ — and the active contact set, hence the dimension $c$ of
> $\boldsymbol\lambda$ — as **fixed** while iterating. That is the *explicit premise* behind Eq. (8)
> (the note's "for frozen Jacobians and bilateral constraints"); without it $\hat{\mathbf D}$ and $\mathbf G$ would
> change every step and there would be no single linear map to take a spectral radius of. In reality they
> are **not** constant: $\mathbf J_a=\mathbf J_a(\mathbf q)$ depends on the arm configuration, $\mathbf J_b$
> on the cloth positions and the contact normals / barycentric weights, and the **contact set itself forms
> and breaks** (changing $c$). The proxy pipeline re-detects contacts **every substep**
> (`collide_interval = 1`), so $\mathbf J_a,\mathbf J_b$ are rebuilt each substep. Two regimes:
> - **Within one substep:** with `proxy_iterations = 1` there is a single pass, so $\mathbf J$ is trivially
>   fixed for that pass ($\mathbf J_{a,k}=\mathbf J_a$, §8.3 note) — the linearization holds exactly there.
> - **Across substeps:** $\mathbf J_a,\mathbf J_b$ and the contact set drift as the geometry moves. So
>   $r(s,q)$ is a **per-substep, local** contraction estimate around the current contact configuration, not
>   a global rate — and because we take only **one** step of the map per substep while its fixed point is
>   itself moving, the scheme **chases a drifting target** rather than settling on a static one.
>
> The frozen-$\mathbf J$ picture is a good local model when the contact configuration changes slowly
> relative to $\Delta t$ (small steps, stable contact); it degrades under fast contact-set changes —
> making/breaking contact, rapid sliding — which is exactly where the one-substep lag (§8.6, item 4) also
> hurts. (This is *separate* from the $\hat{\mathbf M}_a$-vs-$\mathbf M_a$ approximation: even with a perfect
> $s=1$, a moving $\mathbf J$ means the per-substep fixed point is not the previous one.)

### 8.5 Mapping the §3 steps onto the theory [exact]

Side $a=$ `mjc`, side $b=$ `vbd`, proxies $=$ Franka hand+finger bodies,
$\hat{\mathbf M}_a=\texttt{mass\_scale}\cdot M_a^{\text{eff}}$ ($=5\,M_a^{\text{eff}}$, the arm's *articulated*
effective mass at the gripper — see §8.6, item 5). Per substep $k\to k{+}1$:

| §3 step | code | theory |
|---|---|---|
| 0 stash | $\mathbf F^{\text{prev}}\!\leftarrow\!\mathbf F^{k}$ | save $\boldsymbol\lambda^{k}$ for the blend |
| 1 apply + arm step | `f_ext += F^k`; MuJoCo | side-$a$ solve $\mathbf M_a\mathbf v_a^{k}=\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda^{k}$ (Eq. (4)) |
| 2 sync | proxy $\mathbf q,\dot{\mathbf q}\!\leftarrow\!$ arm | set the proxy's pre-solve state $=\mathbf v_a^{k}$ (the RHS $\hat{\mathbf M}_a\mathbf v_a^{k}$) |
| 3 rewind | `body_f = -F^k - mg` | the **$-\mathbf J_a^\top\boldsymbol\lambda^{k}$** subtraction of Eq. (7) (plus removing gravity) |
| 4–5 collide + VBD | cloth solve w/ proxy | side-$b$ block of Eq. (7) $\Rightarrow$ new $\boldsymbol\lambda^{k+1}$ |
| 6 harvest | $\sum_c$ contact forces on proxy | read the **body** wrench $\mathbf F^{k+1}=\sum_c[\mathbf f_c;(\mathbf p_c{-}\mathbf c_\text{world})\times\mathbf f_c]=\mathbf J_c^\top\boldsymbol\lambda^{k+1}$ (contact impulses summed onto the proxy body, torque about the CoM $\mathbf c_\text{world}$ — §3 Step 6; the joint pull-back $\mathbf J_a^\top=\mathbf J_{\text{body}}^\top\mathbf J_c^\top$ is completed by MuJoCo at the next Step 1) |
| 7 blend | $\alpha\,\text{new}+(1{-}\alpha)\,\text{old}$ | relaxation; $\alpha=\texttt{proxy\_relaxation}=1$ here |
| next 1 | apply $\mathbf F^{k+1}$ to arm | feed $\boldsymbol\lambda^{k+1}$ to side $a$ (the outer iteration) |

*The interface Jacobians.* Four Jacobians appear ($c=3\times$# active contacts = contact-space rows). Each
maps some solver's velocity DOFs to the **contact-space** relative velocity; the interface relative velocity
is $\mathbf u=\mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b$, and the arm side factors
$\mathbf J_a=\mathbf J_c\,\mathbf J_{\text{body}}$.

| symbol | maps (velocity → velocity) | shape | what a row is |
|---|---|---|---|
| $\mathbf J_a$ | arm joint rates $\dot{\mathbf q}\to$ contact-point velocity (arm side) | $c\times9$ | $=\mathbf J_c\mathbf J_{\text{body}}$ (whole chain to the contact) |
| $\mathbf J_{\text{body}}$ | arm joint rates $\to$ gripper-**body** spatial velocity | $6\times9$ | the gripper body's geometric Jacobian |
| $\mathbf J_c$ | gripper-body spatial velocity $\to$ contact-point velocity | $c\times6$ | per contact $[\,\mathbf I_3\ \ -[\mathbf r_c]_\times\,]$ (offset $\mathbf r_c$ on the body), in the contact frame |
| $\mathbf J_b$ | cloth nodal velocities $\to$ contact-point velocity (cloth side) | $c\times3N$ | per contact, **selects the single contacting particle** — nonzero only in that vertex's 3 columns (a sparse pick, not a dense Jacobian) |

**$\mathbf J_b$ exactly:** cloth contacts here are *particle*-level (a cloth vertex touching the proxy shape),
so the cloth side of contact $c$ *is* one particle $p$; the row of $\mathbf J_b$ picks that particle's velocity
(in the contact frame). Hence $\mathbf J_b^\top\boldsymbol\lambda$ just **scatters the contact force back onto
that vertex** — no interpolation.

Now, which solver applies which. **$\boldsymbol\lambda$ is never assembled** — the code only forms
$\mathbf J^\top\boldsymbol\lambda$ products (and never materializes a $\mathbf J$ either): the per-contact
forces $\mathbf f_c$ (VBD's contact solve — the only place $\boldsymbol\lambda$'s entries transiently live) are
consumed immediately into these:

| solver / role | Jacobian | applies $\mathbf J^\top\boldsymbol\lambda$ as | how (in code) |
|---|---|---|---|
| **VBD — deformable cloth** (side $b$) | $\mathbf J_b$ | particle forces $\mathbf J_b^\top\boldsymbol\lambda$ | contact force added directly to the contacting vertices |
| **VBD — rigid proxy** (side $a$ in VBD) | $\mathbf J_c$ | body wrench $\mathbf F=\mathbf J_c^\top\boldsymbol\lambda$ | per-contact $\mathbf f_c$ summed as (force; torque about CoM) — the harvest, `_harvest_vbd_*` |
| **MuJoCo — arm** (side $a$) | $\mathbf J_{\text{body}}$ | joint torques $\mathbf J_{\text{body}}^\top\mathbf F$ | the harvested $\mathbf F$ on the gripper `body_f`, pulled back to joints inside the articulated solve |

So $\mathbf J_a$ is **split** across two solvers ($\mathbf J_c$ in VBD-rigid, $\mathbf J_{\text{body}}$ in
MuJoCo), while $\mathbf J_b$ lives entirely in VBD-deformable. The chain of a single contact impulse:
VBD produces $\mathbf f_c$ → summed by $\mathbf J_c^\top$ into the body wrench $\mathbf F$ (VBD-rigid, harvest)
→ deposited on `body_f` and pulled back by $\mathbf J_{\text{body}}^\top$ to joint torques (MuJoCo, next Step 1)
→ and, on the cloth, scattered by $\mathbf J_b^\top$ onto the contacting vertex (VBD-deformable).

The coordinate spaces and the maps between them (**velocities push forward through $\mathbf J$; forces/impulses
pull back through $\mathbf J^\top$**; contact space is the hub):

$$
\textbf{velocities}\ (\text{forward }\mathbf J):\qquad
\underset{\text{arm joints}}{\mathbf v_a\in\mathbb R^{9}}\ \xrightarrow{\ \mathbf J_{\text{body}}\ }\ \underset{\text{gripper body}}{\boldsymbol\xi\in\mathbb R^{6}}\ \xrightarrow{\ \mathbf J_c\ }\ \underset{\text{contact}}{\mathbf u\in\mathbb R^{c}}\ \xleftarrow{\ \mathbf J_b\ }\ \underset{\text{cloth nodes}}{\mathbf v_b\in\mathbb R^{3N}}
$$

$$
\textbf{forces}\ (\text{adjoint }\mathbf J^\top):\qquad
\underset{\text{joint torque}}{\boldsymbol\tau\in\mathbb R^{9}}\ \xleftarrow{\ \mathbf J_{\text{body}}^\top\ }\ \underset{\text{body wrench}}{\mathbf F\in\mathbb R^{6}}\ \xleftarrow{\ \mathbf J_c^\top\ }\ \underset{\text{contact impulse}}{\boldsymbol\lambda\in\mathbb R^{c}}\ \xrightarrow{\ \mathbf J_b^\top\ }\ \underset{\text{cloth force}}{\mathbf f\in\mathbb R^{3N}}
$$

$$
\mathbf J_a=\mathbf J_c\mathbf J_{\text{body}},\quad
\mathbf u=\mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b,\quad
\mathbf F=\mathbf J_c^\top\boldsymbol\lambda,\quad
\boldsymbol\tau=\mathbf J_{\text{body}}^\top\mathbf F=\mathbf J_a^\top\boldsymbol\lambda,\quad
\mathbf f=\mathbf J_b^\top\boldsymbol\lambda .
$$

Read it as one duality: in the **velocity** chain each solver's velocity is mapped *into* contact space by
its $\mathbf J$ (the two arm hops compose to $\mathbf J_a$), converging on the relative velocity $\mathbf u$;
in the **force** chain the single contact impulse $\boldsymbol\lambda$ is mapped *out* to each solver's
coordinates by the corresponding $\mathbf J^\top$ — $\mathbf F$ (body wrench) being its arm-side image,
carried across the VBD→MuJoCo boundary. Both $\boldsymbol\lambda$ and every $\mathbf J$ stay implicit — only
the $\mathbf J^\top(\cdot)$ products ($\mathbf F$, $\boldsymbol\tau$, $\mathbf f$) are ever formed.

**How the proxy row's RHS $\hat{\mathbf M}_a\mathbf v_a^{k}$ enters the VBD solve.** It is *not* assembled as
an explicit vector — it is the proxy body's **incoming momentum**, supplied through VBD's ordinary
**inertial term**. The proxy is a genuine *dynamic* rigid body in the VBD view: its inverse mass is kept
**finite** (installed as $\hat{\mathbf M}_a$ via `_apply_body_inertia_override`; *not* zeroed like the other
non-owned bodies), so VBD integrates it implicitly alongside the cloth. That body's per-step kinetic term is
$\tfrac{1}{2\Delta t^{2}}\lVert\mathbf q-\tilde{\mathbf q}\rVert^2_{\hat{\mathbf M}_a}$ with inertial predictor
$\tilde{\mathbf q}=\mathbf q^{n}+\Delta t\,\mathbf v_a^{k}+\Delta t^{2}\hat{\mathbf M}_a^{-1}\mathbf f_\text{ext}$.
The pre-solve velocity $\mathbf v_a^{k}$ there is exactly what **sync** (Step 2) wrote into
`state_0.body_qd`. Differentiating the kinetic term gives, at the velocity level,
$\tfrac{\hat{\mathbf M}_a}{\Delta t}(\mathbf v_a-\mathbf v_a^{k})$ — so $\hat{\mathbf M}_a\mathbf v_a^{k}$
appears on the RHS of the proxy's local Newton equation as its **starting momentum**. With the rewind force
$\mathbf f_\text{ext}=-\mathbf J_a^\top\boldsymbol\lambda^{k}/\Delta t$ (minus gravity) and the cloth contact
impulse $\mathbf J_a^\top\boldsymbol\lambda$, stationarity reads
$\hat{\mathbf M}_a(\mathbf v_a-\mathbf v_a^{k})=\mathbf J_a^\top\boldsymbol\lambda-\mathbf J_a^\top\boldsymbol\lambda^{k}$
— exactly Eq. (6)/(7). Since $\hat{\mathbf M}_a$ is large (mass-scaled), the kinetic term strongly resists
leaving $\mathbf v_a^{k}$, so the proxy **tracks the synced arm velocity** while the cloth contact perturbs
it — the intended behavior of a heavy virtual-inertia proxy.

**Why our fixed point is exactly right even though $\hat{\mathbf M}_a$ is only approximate [exact].**
Two implementation facts conspire:

1. The **sync** imposes the arm's *true* interface pose + velocity on the proxy before the cloth solve,
   so the cloth always sees the real gripper state — not a drifted proxy state.
2. The **harvest reads the explicit contact wrench** ($\mathbf F^{k+1}=\sum_c[\mathbf f_c;(\mathbf p_c-\mathbf c_\text{world})\times\mathbf f_c]=\mathbf J_c^\top\boldsymbol\lambda^{k+1}$,
   `_harvest_vbd_proxy_wrenches_kernel`) — the interface impulse as a body wrench, *directly* — **not** the
   proxy momentum change $\hat{\mathbf M}_a\Delta\dot{\mathbf q}$. So $\hat{\mathbf M}_a$ never enters the
   harvested value; it only sets how far the proxy moved during the solve (a transient), which washes out
   at the fixed point.

At a fixed point $\mathbf F^{k+1}=\mathbf F^{k}=\mathbf F^{*}$: the arm integrated with $\mathbf F^{*}$
produces a pose; the cloth solved against the gripper at that pose produces contact wrench $\mathbf F^{*}$;
consistent. That joint $(\mathbf v_a,\mathbf v_b,\boldsymbol\lambda=\mathbf F^{*})$ satisfies **both rows**
of §8.1 — **so the converged proxy loop equals the monolithic coupled solve**, and `mass_scale` only
decides whether/how fast we get there.

### 8.6 What is exact, what is approximate, and what the design note glosses over

**Sound / [exact]:**

1. The rewind is *exactly* Eq. (7)'s $-\mathbf J_a^\top\boldsymbol\lambda^{k}$ — verified:
   `subtract_proxy_body_forces_kernel` overwrites `body_f = -F - mg`.
2. The explicit-contact harvest reads the true interface wrench, making the fixed point
   **mass-independent** (§8.5) — strictly better than the note's generic momentum harvest (note Eq. 8),
   which *does* drag $\hat{\mathbf M}_a$ into the estimate.
3. The reaction is applied as a 6-DOF wrench on the gripper bodies and mapped to joint torques by
   MuJoCo's own dynamics — i.e. the $\mathbf J_a^\top\boldsymbol\lambda$ row of §8.1 is realized by the
   native solver, with no Jacobian assembled by hand.

**Caveats / [approx] — where it is *not* the monolithic answer:**

4. **`proxy_iterations = 1` ⟹ one iteration per substep — feedback is strictly lagged by one substep.**
   We never iterate to the fixed point *within* a step; correctness relies on the contact state changing
   slowly across the 10 substeps so the lag is small. Under fast/stiff contact transients the interface
   is *not* converged and the arm responds a substep late (the note's "lagged unless fixed-point
   iterations are used"). This is the single biggest gap from the monolithic ideal.
5. **$\hat{\mathbf M}_a=\texttt{mass\_scale}\cdot M_a^{\text{eff}}$ uses the arm's *articulated* effective
   mass — `mass_scale = 5` is a deliberate over-scaling, not a blind guess.** The proxy coupling *does* call
   the effective-mass hook: MuJoCo's `coupling_eval_effective_mass_block` returns
   $M_a^{\text{eff}}=1/\big(\text{body\_invweight0}+\tfrac23\,\text{inv\_rot}\,\lVert\mathbf r\rVert^2\big)$ —
   the articulated effective (inverse) inertia felt at the gripper, from MuJoCo's `body_invweight0` (the
   whole-chain inverse inertia), with a point-offset term — though the coupler evaluates it at the body
   origin ($\mathbf r=\mathbf 0$, glossary §8.0), so in practice $M_a^{\text{eff}}=1/\text{body\_invweight0}[0]$
   and the offset term is inactive. So the **nominal** scale is
   $\hat{\mathbf M}_a/M_a^{\text{eff}}=\texttt{mass\_scale}=5$ — known, and $>1$ on purpose. **Mind the two
   readings of the scale** (glossary, §8.0): the theory's $s=\hat{\mathbf M}_a\big/(\mathbf J_a\mathbf M_a^{-1}\mathbf J_a^\top)^{-1}$
   is measured against the *true* step inertia $\mathbf M_a$, **not** against $M_a^{\text{eff}}$ — and since
   $M_a^{\text{eff}}$ **undercounts** $\mathbf M_a$ (it omits the PD damping $h\mathbf K_d$ and uses the home
   pose), the true $s$ is **smaller** than the nominal $5$. Either way the map is contractive here: by
   Eq. (9), for $q=M_a/M_b\gg1$ (the shirt is light vs. the arm's effective mass) any $s>1$ gives
   $|r|=|1-s|/(1+qs)\ll1$. *Two residual approximations:* (i) `body_invweight0` is the **home-configuration**
   inverse inertia, not recomputed as the arm moves; and (ii) it is a translational + $\lVert\mathbf r\rVert^2$
   scalar/diagonal reduction, not the full $\mathbf J_a\mathbf M_a^{-1}\mathbf J_a^\top$ block. `mass_scale = 5`
   is the safety margin against those approximations, the PD undercount, and the one-substep lag.
6. **The cloth feels the replica inertia $\hat{\mathbf M}_a$, not the arm's real articulated mass.** Even
   at the fixed point the *transient* contact response (impact, fast grasp) is shaped by
   $\hat{\mathbf M}_a$: the arm's true off-diagonal articulated inertia is replaced by a scalar-scaled
   rigid mass on 3 bodies. Steady force is right; impact fidelity is approximate.
7. **Only the gripper (hand + fingers) is a proxy** — the rest of the arm never collides with the cloth
   (the same structural limit as the Genesis two-way fingers). Whole-arm / cloth contact is not
   represented.
8. **Lagged sync uses the *begin*-of-substep arm pose** (`state_0.body_q`), so the pose the cloth sees is
   itself a half-step stale in `lagged` mode; `staggered` uses the end pose and is tighter (§4).

**What the note glosses over (my read):** its §3.3 derives convergence for the *momentum* harvest
(note Eq. 8, $\hat{\mathbf M}_a$ in the estimate), but the shipped VBD path uses the *explicit-contact* harvest,
which is cleaner (mass-independent fixed point) and the note mentions only in passing (p. 21,
`BODY_PROXY_HARVEST`). The careful $s,q$ contraction analysis (Eq. (9); note Fig. 1) applies only loosely:
`proxy_iterations = 1` means there is no inner fixed-point loop to contract — the contraction instead plays
out *across substeps*, against a moving contact configuration (the frozen-$\mathbf J$ caveat above). The
implementation *does* use the analysis to size the proxy, though: it sets
$\hat{\mathbf M}_a=\texttt{mass\_scale}\cdot M_a^{\text{eff}}$ from MuJoCo's articulated effective mass
(item 5), so $s=5$ is a chosen (and, for a light cloth, contractive) over-scale rather than a blind
constant — the loop is closed at the level of *one* per-substep step, with the small $\Delta t$ absorbing
the rest.

### 8.7 Does this theory cover IPC-style log-barrier contact? No — structurally

The §8 fixed-point theory is intrinsically a **penalty / constraint-level, velocity, frozen-Jacobian**
theory. It does **not** extend to a log-barrier (interior-point) contact, for two compounding reasons —
this is a *scope* statement about the theory, independent of how contact happens to be implemented here.

1. **The convergence analysis has no place for the barrier stiffness.** $r(s,q)=\tfrac{1-s}{1+qs}$ (Eq. (9))
   is a *linear* analysis whose contact-space operator is the **inertia Delassus**
   $\hat{\mathbf D}=\mathbf J_a\hat{\mathbf M}_a^{-1}\mathbf J_a^\top+\mathbf J_b\mathbf M_b^{-1}\mathbf J_b^\top$,
   with the mass ratio $s=\hat{\mathbf M}_a/\mathbf M_a$ the only knob. A log-barrier's defining feature is a
   **position-level potential whose Hessian $\nabla^2 B$ diverges as the gap $d\to0$** and swings by orders
   of magnitude within one contact event. Near contact *that* stiffness — not the mass ratio — governs
   stability, and it is nowhere in $r(s,q)$. Moreover the lagged feedback is an **explicit** treatment of
   the interface, and explicitly integrating a near-singular stiff barrier across a one-substep lag is
   precisely the unstable regime (it would force $\Delta t\to0$). IPC's barrier is built for a **monolithic
   implicit Newton** solve with $\nabla^2 B$ *in* the system matrix — the opposite of a partitioned lagged
   exchange.

2. **The scheme imposes the proxy pose; a barrier forbids that.** The proxy pose is **synced
   (kinematically imposed)** from a contact-*unaware* solver (MuJoCo). Penalty contact tolerates this: a
   penetrating imposed pose merely yields a large restoring force. A **log-barrier cannot** — it is
   undefined/infinite at $d\le0$ and **requires a penetration-free start state plus a CCD-filtered line
   search over the whole trajectory**. An externally driven pose can start in penetration and has **no lag
   mechanism** to stay out, so it breaks the barrier's precondition outright.

This is exactly why two-way coupling of a barrier solver uses a **different architecture** — the Genesis
`two_way_soft_constraint` one (§9): the contact body is kept **dynamic inside the IPC solve** (its pose is
IPC's own penetration-free output, carried across steps) and is pulled toward the external command by a
**soft spring**, so it can *lag* a penetrating aim while the barrier holds it out; only the spring force is
fed back. The proxy/virtual-inertia theory here is the **penalty-world counterpart** of that, and the two
are **not interchangeable**: you cannot drop a log-barrier into the "sync the pose → solve contact →
harvest → lag" loop. So §8 answers convergence for compliant/constraint contact; for barrier contact the
correct object is the monolithic IPC incremental potential, not this fixed-point map.

---

### 8.8 Could we just copy the exact $\mathbf M_a$ into VBD every step? (feasible; barely changes the answer)

A natural instinct: the proxy inertia $\hat{\mathbf M}_a$ is only an *estimate* (bare mass, home pose,
isotropic scalar, per-body — §8.6 item 5), and $s=\hat{\mathbf M}_a/\mathbf M_a^{\text{eff}}=1$ gives $r=0$,
one-shot convergence (§8.4). So why not compute the arm's **true** inertia in MuJoCo and hand it to VBD every
step? It is worth being precise about what this buys, because the answer is *"yes, feasible; no, it barely
moves the converged result."*

**First, the right object is not $\mathbf M_a$.** VBD has no arm DOFs to hold a $9\times9$ joint-space
matrix. What sets $s=1$ is matching the arm's **operational-space (interface) inertia** — the block
$(\mathbf J_a\mathbf M_a^{-1}\mathbf J_a^\top)^{-1}$ seen *at the gripper bodies* (§13.4; the Delassus
contribution of side $a$). "Copying $\mathbf M_a$" really means: give each proxy the effective $6\times6$
inertia the articulated arm presents at that body, *now*.

**It is genuinely computable.** MuJoCo/`mujoco_warp` already forms $\mathbf M(\mathbf q)$ (CRBA) and the
$\mathbf J\mathbf M^{-1}\mathbf J^\top$ products its constraint solver needs. The current
`coupling_eval_effective_mass_block` deliberately takes the *cheap* version of exactly this — `body_invweight0`,
which is the trace-averaged block at the **home pose**, on the **bare** mass matrix, with **no PD term**
(§13.4, §8.6 item 5). A faithful version would instead, per step: (i) evaluate at the **current $\mathbf q$**;
(ii) use the **step matrix** $\mathbf M(\mathbf q)+h(\mathbf K_d+\mathbf D)$, not bare $\mathbf M$ (§12.2);
(iii) keep the full $6\times6$ block instead of the isotropic scalar. This is moderate engineering — the
pieces exist in `mujoco_warp` — and it would drive $s\to1$ and let us **retire `mass_scale`** as a hand-tuned
knob (`mass_scale`$=5$ exists precisely to inflate the *undercounted* bare-mass estimate back up toward the
true step inertia; §8.6 item 5).

**But four things stop it from being the exact monolithic answer in one shot — and only one is an
approximation you could engineer away:**

1. **Per-body vs. the arm's articulation coupling — structural, not a tuning error.** VBD installs the
   proxies as **independent rigid bodies**: each gets *one* scalar mass and *one* $3\times3$ inertia
   (`_apply_body_inertia_override`), with **no joints linking hand and fingers**. The true operational-space
   inertia of the three gripper bodies is a **coupled $18\times18$** operator — pushing one finger accelerates
   the hand *through the shared arm*. Three independent diagonal blocks cannot represent those off-diagonal
   terms. Fixing this would require replicating the arm's **joints** inside VBD too — i.e. re-porting the
   articulation into the cloth solver, which is exactly the monolithic assembly the partition exists to avoid.
   So even a "perfect" per-body inertia is a *block-diagonal approximation* of the real interface response.
2. **The response is configuration-dependent and nonlinear.** A single inertia block is a linearization about
   the current state; the arm's actual reply is $\mathbf q$-dependent and is itself produced by ~100 internal
   MuJoCo Newton iterations, not one linear solve. Matching the block makes $r\approx0$ *locally at that
   instant*, not identically.
3. **The lag and frozen Jacobian are independent of $\hat{\mathbf M}_a$.** The one-substep impulse lag and the
   held-fixed $\mathbf J$ (§8.6) are separate error sources; perfect inertia removes neither. And with the
   default `proxy_iterations = 1` there is **no inner loop** for a small $r$ to act on — $r=0$ would mean "one
   lagged pass already equals this substep's frozen-data monolithic solve," but the *inter-substep* lag
   remains. A better $\hat{\mathbf M}_a$ only pays off in the transient/stability, or if you also raise
   `proxy_iterations`.
4. **Active-set / stick–slip switching** (§8.4 Step 4) is outside any single linear $\mathbf G$, so no inertia
   choice linearizes it away.

**What actually changes, then.** The **fixed point is already exact regardless of $\hat{\mathbf M}_a$**
(§8.4 Step 3) — so copying the true inertia changes the *rate, stability and transient fidelity*, **not the
converged physics**. The clearest practical win is the **stiff-PD fingers** (`FINGER_STIFFNESS`$=4\times10^4$):
their step inertia $\mathbf M(\mathbf q)+h\mathbf K_d$ is *dominated* by the $h\mathbf K_d$ term that the
current bare-mass estimate omits entirely, so $s$ is worst there and `mass_scale` is compensating most
crudely during exactly the grasp phase that matters. A PD-inclusive, current-pose estimate would target that
regime correctly and remove a fragile tuning knob.

**Verdict.** Feasible with moderate effort, and modestly worthwhile — smoother, more robust transients and one
fewer hand-tuned parameter, with the biggest effect during stiff grasping. But **not transformative**: it
cannot reach true one-shot convergence (limits 1–4), it leaves the converged answer unchanged (that was
already correct), and with `proxy_iterations = 1` the dominant residual error is the one-substep *lag*, not
the inertia mismatch. The high-value version of the idea is narrow — "include the PD/current-pose term in the
per-body effective mass so `mass_scale` can go away" — whereas the *full* fidelity gain (coupled articulation
inertia) would mean rebuilding the arm inside VBD, i.e. giving up the partition. This is the same trade the
whole document turns on: $\hat{\mathbf M}_a$ is a **convergence knob, not a physical mass** (§8.4), so effort
spent making it exact improves *how fast and how smoothly* we reach the answer, never *which* answer.

**Part IV — Comparison with Genesis IPC**

## 9. Relation to the Genesis IPC `two_way_soft_constraint` coupling

Same *family* — partitioned co-simulation with the gripper as a proxy in the cloth solver and lagged
force feedback — but different primitives:

| | Genesis IPC `two_way` | Newton `SolverCoupledProxy` |
|---|---|---|
| Solvers | Genesis-rigid + libuipc-IPC | MuJoCo (arm) + VBD (cloth) |
| Coupling primitive | **soft spring** (`SoftTransformConstraint`, stiffness η) to an inertialess proxy | **direct pose/vel sync** of a mass-scaled proxy **+ harvested contact wrench** |
| Proxy inertia | inertialess (`external_kinetic`: no kinetic term; §9.2) | **has mass** (`mass_scale`) → cloth feels a tunable inertia |
| Force fed back | spring gradient (approximate reconstruction) | actual contact wrench (momentum-change or explicit contact) |
| Coupling iterations | single pass | **iterable** (`proxy_iterations` + relaxation) |
| Monolithic alternative | `external_articulation` (joints inside IPC) | `--solver avbd` (one AVBD solve) |

So the proxy coupling directly mitigates two IPC-coupling drawbacks: the **massless-proxy** issue
(here the proxy carries `mass_scale` inertia, so the cloth feels something closer to the arm's
effective mass) and the **single-pass** issue (here you can relax with `proxy_iterations`). It still
shares the **lagged-feedback** limitation — `lagged` is explicitly one step behind; `staggered`
tightens it.

### 9.1 What each of the three solvers sees — and why the coupling is *two-tier*

"MuJoCo $+$ VBD" is the entry-level count, but `SolverVBD` is documented as *"Vertex Block Descent (VBD)
for particles **and Augmented VBD (AVBD) for rigid bodies**"* — so the single cloth entry internally runs
**two** algorithms. So there are really **three views** of the world — and Genesis IPC's
`two_way_soft_constraint` has a **direct counterpart for each** (plus the same two coupling tiers), which is
the clearest way to see how the two frameworks line up:

| the view | Newton `SolverCoupledProxy` | Genesis IPC `two_way_soft_constraint` |
|---|---|---|
| **Arm** (articulated robot) | **MuJoCo**, own `mjc` entry — a *separate* solver: full Franka, 9 joint DOF, PD actuators, gravity; true step inertia $\mathbf M_a=\mathbf M(\mathbf q)+h(\mathbf K_d+\mathbf D)$; solves its own arm/table + self contacts; never sees the cloth | **Genesis rigid solver** — *separate* from IPC: full Franka, PD/IK control, real articulated dynamics. The arm is **not** in the IPC solve; only the finger poses are exported as targets |
| **Gripper proxy** (rigid, living in the cloth solver) | **AVBD-rigid**, in the `vbd` entry: 3 replicas (hand + 2 fingers), 6 DOF each, no joints (§8.8); **finite** inertia $\hat{\mathbf M}_a=\texttt{mass\_scale}\cdot M_a^{\text{eff}}$ — a genuine dynamic body the cloth accelerates | **libuipc affine bodies**: **only the 2 fingers**, 12 affine DOF; **inertialess** (`external_kinetic`, §9.2), pulled to the arm's `aim` by a soft-transform spring (strength $\eta$) |
| **Cloth** (deformable) | **VBD**, in the `vbd` entry: unisex_shirt, $3N$ nodal DOF ($N=6436$); Baraff–Witkin membrane + discrete-shell bending + self-contact; lumped nodal mass (§12.3) | **libuipc FEM cloth**: Baraff–Witkin shell + discrete-shell bending + self-contact, all under the IPC log-barrier with CCD line search |
| **Rigid $\leftrightarrow$ cloth** (inner coupling) | **monolithic** — AVBD + VBD in *one* `SolverVBD.step()`, one incremental potential: tight, no lag; contact via VBD penalty | **monolithic** — cloth + affine fingers + contact in *one* libuipc solve: tight, no lag; contact via IPC log-barrier |
| **Arm $\leftrightarrow$ cloth-solver** (outer coupling) | **partitioned, lagged** — pose + velocity down (sync), **harvested** contact wrench $\mathbf J_c^\top\boldsymbol\lambda$ up, one-substep lag | **partitioned, lagged** — arm pose $\to$ held `aim` down (a frozen snapshot), **reconstructed** spring-gradient reaction up, single pass (§9.2) |

The right mental model is **two tiers of coupling of very different tightness:**

**Inner tier — monolithic (tight).** AVBD-rigid and VBD-cloth are **not** two coupled solvers exchanging
messages — they are two blocks of **one** `SolverVBD.step()`, descending one incremental potential
together. The proxy $\leftrightarrow$ cloth contact impulse $\boldsymbol\lambda$ is resolved *simultaneously*
with the cloth's own elasticity and self-contact; it is never harvested, blended, or lagged, and it lives
entirely inside the Warp solve. This tier is as tight as IPC's — everything in it converges jointly.

**Outer tier — partitioned (lagged).** MuJoCo $\leftrightarrow$ `SolverVBD` is the actual proxy coupling of
§3–§8. That boundary is exactly where the two frameworks differ, so it is worth tabulating **every channel
that crosses the arm-solver ↔ contact-solver boundary in both** — the down-sync, the up-feedback, and what
each omits:

| channel | Newton `SolverCoupledProxy` | Genesis `two_way_soft_constraint` |
|---|---|---|
| **↓ arm pose → contact solver** | `body_q` **hard-set** on the proxy — kinematic sync, every substep (§3 Step 2) | finger pose written as `aim_transform` — a **soft target**, pose only |
| **↓ arm velocity → contact solver** | `body_qd` **hard-set** on the proxy (§3 Step 2) | **none** (no velocity target) |
| **↓ arm inertia → contact solver** | $\hat{\mathbf M}_a=\texttt{mass\_scale}\cdot M_a^{\text{eff}}$, **finite**; set on model change | **none** — the body is `external_kinetic` (inertialess, §9.2); its $\mathbf M$ is reused only as the spring metric $\eta\mathbf M$ |
| **↑ contact solver → arm force** | **harvested true wrench** $\mathbf F=\mathbf J_c^\top\boldsymbol\lambda$; lagged one substep (§3 Step 6 $\to$ next Step 1) | **reconstructed** spring gradient $\mathbf F=\tfrac{\eta}{\Delta t^{2}}\mathbf M(\mathbf q_{\text{ipc}}-\mathbf q_{\text{aim}})$ (§9.2) |
| **↑ contact solver → arm pose/velocity** | **none** — the proxy's solved pose/velocity is discarded | **none** in `two_way` (only the separate `ipc_only` mode writes pose + velocity back, for IPC-owned free bodies) |
| **lag / iteration** | one-substep lag; **iterable** (`proxy_iterations` + relaxation) | one-step staggered; **single pass** |

The two asymmetries this exposes: **down**, Newton sends *more* (hard pose **+** velocity **+** finite inertia)
while Genesis sends *less and softer* (a pose-only soft target, inertialess); **up**, both send force only, but
Newton's is the *true harvested* reaction whereas Genesis's is a *reconstruction*.

For Newton that is the entire conversation — and note what *never* crosses: never $\boldsymbol\lambda$
itself, never the cloth or arm state directly (§8.5, §8.1). Everything in §8 — the fixed-point theory,
$\hat{\mathbf M}_a$ as a relaxation knob, the one-substep lag — is about **this outer tier only**; the inner
tier is exact.

**Contrast with Genesis IPC — same two tiers, not "monolithic vs. partitioned."** It is tempting to call
Genesis "fully monolithic," but for `two_way_soft_constraint` only the *inner* tier is: cloth + the
affine-body fingers + contact are one libuipc `advance()`, tight and lag-free (like Newton's
`SolverVBD.step()`). The **outer** tier is **partitioned and lagged too** — the Genesis rigid solver runs the
*full articulated arm* separately (just as MuJoCo does), and IPC sees it only through a **held `aim`**: a
snapshot of the arm's finger pose taken at the top of `couple()` (`_store_gs_rigid_states`) and **frozen for
the whole solve**, with the reaction *reconstructed* afterward and pushed to the arm for its next step. That
`aim` is precisely a **lagged proxy target for the rigid body** — the direct analogue of Newton's synced
proxy. (Genesis's genuinely monolithic mode is a *different* one, `external_articulation`, which puts the
arm's joints inside IPC — the §9-table "monolithic alternative.")

So the two frameworks are structurally **the same**: a monolithic cloth/contact core wrapped in a
partitioned, **lagged** coupling to a full-fidelity articulated arm living in its own solver. The genuine
differences are local (see the table): Newton's proxy carries **finite inertia**, syncs **pose + velocity**,
feeds back the **actual harvested wrench** $\mathbf J_c^\top\boldsymbol\lambda$, and can **iterate** the outer
loop; Genesis's finger is **inertialess**, follows a **pose-only aim**, feeds back a **reconstructed spring
gradient**, in a **single pass** — with contact resolved by an IPC log-barrier rather than a VBD penalty.

### 9.2 Why the Genesis proxy is *inertialess* (what "no kinetic term" means)

In `two_way_soft_constraint` the coupled fingers contribute **no kinetic/inertia term** to the IPC incremental
potential. The cloth nodes get the usual
inertia $\tfrac12\lVert\mathbf z-\tilde{\mathbf z}\rVert^2_{\mathcal M}$; the fingers get **none** — their
real rigid-body dynamics are computed *externally* (by the Genesis arm), and IPC is handed only a **target
pose** (the `aim`) for them to follow. That is what "inertialess" means, and what the §9 table's `K=0` was
loose shorthand for: the coefficient on the body's own kinetic (inertia) term is zero.

So inside the IPC solve a coupled finger has **no mass to accelerate**. The only things acting on its 12
affine DOF are (i) the **soft-transform spring**
$\Psi^{\text{stc}}_\ell=\tfrac12\,\delta\mathbf q_\ell^\top\tilde{\mathbf M}_\ell\,\delta\mathbf q_\ell$,
$\delta\mathbf q_\ell=\mathbf q_\ell-\hat{\mathbf q}_\ell$, pulling its pose toward the aim, and (ii) the ABD
rigidity penalty (keeps it a rigid shape) $+$ the contact barrier (the cloth pushing back). Its finger-block
stationarity is a **quasi-static** balance $\tilde{\mathbf M}_\ell\,\delta\mathbf q_\ell+\mathbf J^\top\mathbf f_{\text{contact}}=\mathbf 0$
— spring force vs. contact force — with **no $\mathbf M/\Delta t^{2}$ inertia** in it.

**$\tilde{\mathbf M}_\ell$ is a stiffness, not an inertia.** The penalty weight is built from the body's *own*
mass matrix, $\tilde{\mathbf M}_\ell=\eta_p\,\mathbf M_{cm}+\eta_a\,\mathbf M_{rot}$ (translation weight
$\eta_p$, rotation weight $\eta_a$, both in $[0,100]$). $\mathbf M$ is reused **only as the metric of the
pose penalty** — a heavier body is pulled toward the aim proportionally harder — *not* as dynamical inertia.
Raising $\eta$ makes the finger track the arm more stiffly; in the limit it is **position-prescribed**
(infinite apparent stiffness, still zero apparent inertia). ($\eta_p=\eta_a=100$ in the shirt-pick/teleop
scenes; libuipc's `SoftTransformConstraint.apply_to` receives the *raw* strength, while the reaction below
uses $\eta/\Delta t^{2}$.)

**The fed-back reaction is a reconstructed spring gradient — verified in code.** Genesis does **not** harvest
libuipc's solved constraint/contact force. After the IPC solve, `_apply_abd_coupling_forces` $\to$
`update_coupling_forces` (`ipc_coupler/{coupler,utils}.py`) *recomputes* the reaction on each finger from how
far IPC dragged it off its aim, weighting translation by mass $m$ and rotation by the world-frame inertia
$\mathbf I_{\text{world}}=\mathbf R_{\text{ipc}}\mathbf I_i\mathbf R_{\text{ipc}}^\top$, both scaled by
$1/\Delta t^{2}$:

$$
\mathbf F=\frac{\eta_p}{\Delta t^{2}}\,m\,\big(\mathbf p^{\,n+1}_{\text{ipc}}-\mathbf p^{\,n}_{\text{aim}}\big),
\qquad
\boldsymbol\tau=\frac{\eta_a}{\Delta t^{2}}\,\mathbf I_{\text{world}}\,\operatorname{log}\!\big(\mathbf R^{\,n+1}_{\text{ipc}}(\mathbf R^{\,n}_{\text{aim}})^{\top}\big).
$$

This is exactly the gradient $\tilde{\mathbf M}_\ell\,\delta\mathbf q$ of the soft-transform spring evaluated
at the converged pose — the code comments it as enforcing *"action-reaction consistency,"*
$\mathbf F_{\text{genesis}}=\mathbf M(\mathbf q^{\,n+1}_{\text{ipc}}-\mathbf q^{\,n}_{\text{genesis}})$. So the
feedback is a **penalty-spring reconstruction** rather than a harvested impulse — but for this *inertialess*
body that is far less lossy than it sounds. The finger's rigid-mode stationarity carries **no inertia term**
(`external_kinetic`), so it reduces to a pure balance
$\tilde{\mathbf M}_\ell\,\delta\mathbf q=-\mathbf J^\top\mathbf f_{\text{contact}}$: the spring force needed to
hold the finger *is* the contact reaction (**Newton's third law** — exactly the "action-reaction consistency"
the code comments). The reconstruction is therefore **$\approx$ exact**; its genuine errors are only the
$12\to6$ affine-to-rigid projection, the $\eta/\Delta t^2$ scaling, and the raw (un-lerped) aim — *not* the
concept. If IPC holds the finger near its aim ($\delta\mathbf q\approx\mathbf 0$) that means the cloth is
barely pushing (the balance forbids a large force at $\delta\mathbf q\approx\mathbf 0$), so a near-zero
reported reaction is *correct*, not a miss. This is the content of the §9 table's "spring gradient (approximate
reconstruction)"; the equality breaks only once the body is given inertia (§9.4).

**Why it matters — and why Newton's two changes come as a package.** The *primary* drawback is **no inertia**:
the cloth feels the gripper as a *stiff kinematic target*, not a mass — a fast finger $\to$ cloth impact
transfers no momentum through a finger-inertia term. The reconstructed feedback is *not* a separate defect in
Genesis, because (above) the inertialess balance makes $\tilde{\mathbf M}_\ell\,\delta\mathbf q$ ≈equal the true
reaction. But the two are **linked**: the moment you fix the inertia by giving the proxy **finite mass**, that
balance gains an inertial term and the spring gradient **stops** equalling the contact force — so you must
*also* switch to a **directly harvested** wrench. Newton's `SolverCoupledProxy` does exactly this pair: the
proxy carries a finite $\hat{\mathbf M}_a=\texttt{mass\_scale}\cdot M_a^{\text{eff}}$ (§9.1) — a **genuine
dynamic body** the cloth feels the inertia of — *and* the feedback is the **actual summed contact wrench**
$\mathbf F=\mathbf J_c^\top\boldsymbol\lambda$, *harvested* from the VBD contact solve (§3 Step 6, §8.5), which
stays correct with mass present. The price is the lagged outer coupling (§9.1).

### 9.3 Which design is conceptually better — and why the choice isn't free

Since §9.1 established that the two frameworks are **structurally the same** (monolithic cloth/contact core
$+$ partitioned-lagged coupling to an articulated arm), the interesting question is which set of *localized*
choices is better. A subjective read, with the reasoning made explicit.

**Where the Newton coupler is cleaner (as a coupling scheme).**

- **The feedback is read directly, not reconstructed** — *conditionally* an advantage. Newton harvests the
  actual contact wrench $\mathbf J_c^\top\boldsymbol\lambda$; Genesis reconstructs it as the spring gradient
  $\tilde{\mathbf M}_\ell\,\delta\mathbf q$. For Genesis's *inertialess* finger those are ≈equal by the force
  balance $\tilde{\mathbf M}_\ell\,\delta\mathbf q=-\mathbf J^\top\mathbf f_{\text{contact}}$ (Newton's third
  law, §9.2) — so on its own this is a **weak** edge, mostly avoiding the reconstruction's projection/scaling
  approximations. It becomes **decisive only because of the next point**: once the proxy has inertia the spring
  gradient no longer equals the contact force, while a direct harvest still does (§9.4). The honest feedback is
  thus *downstream of* finite mass, not an independent win.
- **The proxy mass is a provable convergence knob, not a bias.** The §8 theory shows the converged solution
  is the true monolithic one *for any* $\hat{\mathbf M}_a$; the mass only sets the rate $r(s,q)$. That is a
  clean separation of "what is exact" from "what is a knob." Genesis's inertialess proxy is instead a
  standing *physical* compromise — the cloth never feels the gripper as a mass, so impact momentum transfer
  is unmodelled.
- **It is iterable.** `proxy_iterations` $+$ relaxation can tighten the lag toward the monolithic answer;
  Genesis is single-pass.

**Where Genesis is stronger (as a contact solver).** Its inner core is a real IPC barrier solve —
**guaranteed penetration-free** (log-barrier $+$ CCD line search). For cloth, tunneling through the fingers
is catastrophic and obvious, and non-penetration here is a *hard guarantee*, not a stiffness-dependent knob.
Newton's VBD **penalty** contact offers no such guarantee: thin cloth against rigid fingers can interpenetrate
and relies on stiffness to push back.

**The point that ties it together: the contact model dictates the coupling design — you cannot mix-and-match.**

- **Penalty contact tolerates an imposed pose** (a penetrating kinematic pose just yields a large restoring
  force). *That* is what lets Newton sync a proxy pose from a contact-unaware solver and harvest the true
  reaction. The finite proxy mass and the honest feedback are **downstream of the penalty choice**.
- **A log-barrier cannot accept an imposed pose** — it is infinite at $d\le0$ and needs a penetration-free
  start plus a CCD-filtered trajectory (§9's "why not IPC here" note). So Genesis is *forced* to keep the
  finger **dynamic inside IPC** and pull it with a spring toward a lagged target — and that, in turn, is what
  *forces* the inertialess proxy and the reconstructed feedback. Those are **downstream of the barrier
  choice**, not independent design mistakes.

So "which do I prefer" largely reduces to **penalty vs. barrier for the gripper $\leftrightarrow$ cloth
contact**, and each framework is internally coherent given that root choice. My take: as a *coupling scheme*
Newton's is the more elegant and physically honest one (true-force feedback, finite mass, a fixed-point
theory that says exactly what is exact); as a *contact guarantee* Genesis's barrier is the property I would
most want to keep. The dream — Newton's harvest-and-iterate coupling wrapped around an IPC barrier core — is
exactly what the incompatibility above rules out without a redesign (you would have to feed the barrier a
*dynamic* proxy and read a *true* Lagrange/contact force out of it, rather than syncing a pose and harvesting
a penalty force). For raw elegance: the Newton proxy coupler. For "I trust the contact will never be wrong":
Genesis.

### 9.4 Could you couple MuJoCo with IPC the Newton way?

Not *absolutely* impossible — but the thing that is impossible is the **literal mechanism**, not the
coupling. Newton's scheme bundles a *philosophy* with a specific *implementation*, and only the
implementation collides with a barrier.

**What breaks.** Newton's §3 Step 2 **kinematically imposes** the proxy pose (hard-sets `body_q` from
MuJoCo each substep). That single act is what an IPC barrier cannot accept: MuJoCo is contact-unaware of the
cloth, so the imposed pose can start *in penetration*, and the log-barrier is infinite/undefined at
$d\le0$ and needs a penetration-free start plus a CCD-filtered trajectory (§9's "why not IPC here" note).
There is no soft escape — "hard-set the pose and let the barrier sort it out" is self-contradictory. Even
projecting/CCD-snapping the imposed pose out of penetration means it is *no longer an exact sync*. So **exact
hard-sync is genuinely incompatible** — that, precisely, is what "impossible" refers to.

**What partly transfers — and what does not.** Newton's coupler has two ideas: **(A)** represent the arm
inside the contact solver as a **finite-mass proxy** (so the cloth feels inertia), and **(B)** **harvest the
raw contact reaction** and feed it back to MuJoCo. Only **(A)** survives the barrier; **(B) does not** — momentum
consistency forces *spring-force* feedback instead (point 3). A barrier-compatible MuJoCo $\leftrightarrow$
IPC coupler looks like:

1. **Make the proxy a *dynamic* IPC body with finite mass** — drop Genesis's `external_kinetic` (§9.2). IPC
   solves for *its own* penetration-free pose, carried across steps from IPC's output, so the barrier
   precondition is never violated.
2. **Drive it toward MuJoCo's exported finger pose with a *stiff soft-constraint*** (a soft-transform aim),
   **not** a hard weld — stiff enough to track, soft enough to yield to the barrier when tracking and
   non-penetration conflict. That "pressure-relief valve" is exactly why a soft constraint works where the
   hard sync cannot.
3. **What you feed back is the *coupling (spring) force* — not the raw contact force.** Intuition:
   *a massless body is a perfect force gauge.* Genesis's proxy finger has only two forces on it — the tracking
   spring and the cloth contact — and, being massless, it can never accelerate, so the two cancel exactly:
   $$\text{spring} + \text{contact} = m\mathbf a = 0 \quad\Rightarrow\quad \underbrace{\tilde{\mathbf M}_\ell\,\delta\mathbf q}_{\text{spring}} = -\,\mathbf f_{\text{contact}} .$$
   The **spring is what couples the finger to the arm**, so the force transmitted to the arm is the spring
   reaction $-\text{spring}$ — and for a massless finger that *equals* the contact force, which is why
   Genesis's spring-force feedback already *is* the contact reaction (§9.2). Now give the proxy **real mass**:
   the balance gains an inertial term, so the transmitted force is the contact force **minus what the proxy
   absorbed**,
   $$\text{spring} + \text{contact} = m\mathbf a \quad\Rightarrow\quad -\text{spring} = \mathbf f_{\text{contact}} - m\mathbf a .$$
   Tempting to say "then just harvest the raw $\mathbf f_{\text{contact}}$ instead" — but you **can't**: the
   spring is a *two-sided internal force*, and momentum balances only if the arm receives its reaction
   $-\text{spring}$ (over a substep: arm $-\mathbf P_s$, proxy $\mathbf P_c{+}\mathbf P_s$, cloth $-\mathbf P_c$,
   summing to zero). Injecting the raw $\mathbf f_{\text{contact}}$ while a *persistent* proxy *also* takes up
   $m\mathbf a$ **leaks momentum**. So the feedback must stay the **spring force**. Newton gets away with
   harvesting the raw contact wrench only because its proxy is **disposable** — hard-reset each substep,
   untethered during the solve, its momentum discarded (a one-shot gauge). The barrier forbids that reset, so
   it forces a persistent, spring-tethered proxy — hence spring-force feedback. The same hard-sync
   incompatibility that blocks the *down* direction (step 2) blocks the raw-contact harvest on the *up*
   direction; one root cause. **So the barrier collapses the up-direction to Genesis too:** giving the proxy
   mass merely turns the fed-back spring force from $\mathbf f_{\text{contact}}$ (massless) into the
   mass-filtered $\mathbf f_{\text{contact}}-m\mathbf a$ — it does *not* buy a true-contact harvest.
4. **The MuJoCo side is trivial**: it only needs to export the finger pose (it does) and accept an external
   wrench on the finger bodies (it does). MuJoCo $\leftrightarrow$ IPC is no harder than MuJoCo $\leftrightarrow$ VBD.

Notably, **Genesis already proves most of this plumbing works** — `two_way_soft_constraint` *is* an
articulated-arm $\leftrightarrow$ IPC proxy coupler; swapping the arm solver from Genesis-rigid to MuJoCo is
the easy part. But the only Newton-flavored delta that survives is **finite proxy mass** (step 1); the
raw-contact harvest (step 3) does **not** — momentum consistency forces Genesis's spring-force feedback.

**What you must give up (the irreducible cost).**

- **Exact per-substep kinematic tracking.** The proxy becomes a *dynamic* body tracking MuJoCo with spring
  compliance and yielding to the cloth — not a hard-driven one. This concession to the barrier is
  unavoidable; it is the one thing Newton's design assumed it could skip.
- **Barrier bookkeeping** — penetration-free start, CCD line search, `d_hat` compliance: the harvested force
  is the (slightly compliant) barrier force, not a hard multiplier. Physical, but it is IPC's cost.
- **Stability tuning** of a stiff spring $+$ finite mass $+$ lag, plus the extra mass knob — same class as
  Genesis's, one dial larger.

**Bottom line.** You cannot port Newton's **hard kinematic sync**, and — less obviously — you cannot port its
**raw-contact harvest** either: *both* rely on the disposable, per-substep-reset proxy that the barrier
forbids (points 2–3). The barrier collapses *both* coupling directions to Genesis's design — soft-spring drive
*down*, spring-force feedback *up*. What survives is only the **finite proxy mass** (step 1), realized on
Genesis's persistent, spring-coupled body — with the reaction still fed back as the **spring force**, now
mass-filtered to $\mathbf f_{\text{contact}}-m\mathbf a$. So this is weaker than §9.3's "dream hybrid": it is
*Genesis with a heavier, spring-transmitted proxy*, not Newton-on-IPC. The one assumption Newton's
penalty-based design was built on — a disposable proxy you can hard-reset and read the raw contact force from
— is exactly what the barrier denies, on **both** the sync and the harvest.

---

**Part V — Practical**

## 10. How to run

```bash
PY=/home/donglaix/Workspace/tools/venvs/env_isaaclab_uv_cursor/bin/python
cd /mnt/nvme1/Workspace/robotics/newton_coupled          # so `newton` resolves to this repo

# headless smoke test
DISPLAY=:1 $PY -m newton.exp --solver proxy --control state_machine --viewer null --num-frames 30

# interactive (GL viewer)
DISPLAY=:1 $PY -m newton.exp --solver proxy --control interactive

# monolithic AVBD baseline for comparison
DISPLAY=:1 $PY -m newton.exp --solver avbd  --control state_machine
```

Verified: the headless smoke test runs cleanly (MuJoCo arm + VBD cloth + proxy coupling; the scripted
state machine steps REST → MOVE_DOWN → ...).

---

## 11. Notes / limitations

- **Lagged feedback** (one-substep delay in `lagged` mode) → reduced accuracy/stability under stiff
  contact; `staggered` and `proxy_iterations` tighten it at a cost.
- **Two solvers + per-substep state marshaling** (sync poses, rewind, collide, harvest) every step —
  not free; MuJoCo + VBD both run.
- **`mass_scale` over-scales the arm's articulated effective mass** — the proxy inertia is
  $\texttt{mass\_scale}\cdot M_a^{\text{eff}}$ with $M_a^{\text{eff}}$ from MuJoCo's `body_invweight0`
  (§8.6, item 5); it sets how the cloth perturbs the proxy. The full articulated dynamics still live in
  MuJoCo; `mass_scale` is the convergence/relaxation knob.
- **Proxy ↔ arm can diverge** (the proxy is synced from the arm but solved against the cloth), as in
  any partitioned scheme.
- The coupler lives under `newton.solvers.experimental.coupled` — an experimental API.

---

**Appendices**

## 12. Appendix — the per-solver implicit step (the "$\mathbf M_a\mathbf v_a=\mathbf f_a$" model in detail)

§8.1 treats "$\mathbf M_a\mathbf v_a=\mathbf f_a$" as each sub-solver's local response. This appendix unpacks
it: what is being linearized in general (§12.1), the **exact MuJoCo `implicitfast` form** with its total
force and the derivatives $\partial\mathbf f/\partial\dot{\mathbf q},\ \partial\mathbf f/\partial\mathbf v,\
\partial\mathbf f/\partial\mathbf q$ (§12.2), the VBD per-vertex model (§12.3), and why the coupling needs
only the *effective* response (§12.4).

### 12.1 What is being linearized (generic implicit step)

$\mathbf M_a\mathbf v_a=\mathbf f_a$ is **one Newton iteration of the sub-solver's own implicit time step**
(equivalently, minimizing the local *quadratic model* of its incremental potential). Implicit Euler must
solve the nonlinear residual

$$
\mathbf r(\mathbf v)=\mathbf M\,\frac{\mathbf v-\mathbf v^{-}}{\Delta t}-\mathbf f\big(\mathbf q(\mathbf v),\mathbf v\big)=\mathbf 0,
$$

where $\mathbf f$ bundles gravity, actuator/PD, internal elastic, and bias forces. It is **nonlinear**
because the elastic force depends on the (implicit) end position $\mathbf q=\mathbf q^{-}+\Delta t\,\mathbf v$
and the damping/Coriolis bias depends on $\mathbf v$.

*Linearizing $\mathbf r$ about the current iterate — Newton's method, step by step.* Newton solves
$\mathbf r(\mathbf v)=\mathbf 0$ by repeated first-order corrections. Given the current guess
$\mathbf v^{(i)}$, expand the residual to first order in an update $\Delta\mathbf v$ and demand the result
vanish:

$$
\mathbf r(\mathbf v^{(i)}+\Delta\mathbf v)\approx\mathbf r(\mathbf v^{(i)})+\frac{\partial\mathbf r}{\partial\mathbf v}\Big|_{\mathbf v^{(i)}}\Delta\mathbf v\;\overset{!}{=}\;\mathbf 0
\quad\Longrightarrow\quad
\frac{\partial\mathbf r}{\partial\mathbf v}\,\Delta\mathbf v=-\,\mathbf r(\mathbf v^{(i)}).
$$

The system matrix $\partial\mathbf r/\partial\mathbf v$ follows from the **chain rule**, since $\mathbf f$
depends on $\mathbf v$ both directly and through $\mathbf q(\mathbf v)=\mathbf q^{-}+\Delta t\,\mathbf v$
(so $\partial\mathbf q/\partial\mathbf v=\Delta t\,\mathbf I$):

$$
\frac{\partial\mathbf r}{\partial\mathbf v}
=\frac{\mathbf M}{\Delta t}
-\underbrace{\frac{\partial\mathbf f}{\partial\mathbf q}\,\frac{\partial\mathbf q}{\partial\mathbf v}}_{\Delta t\,\partial\mathbf f/\partial\mathbf q}
-\frac{\partial\mathbf f}{\partial\mathbf v}
=\frac{\mathbf M}{\Delta t}-\Delta t\,\frac{\partial\mathbf f}{\partial\mathbf q}-\frac{\partial\mathbf f}{\partial\mathbf v}.
$$

Write the internal/elastic force as $\mathbf f_{\text{int}}=-\nabla\Psi(\mathbf q)$ and the dissipative
force as velocity-linear. The two derivatives are then exactly the **stiffness** and **damping** matrices

$$
\mathbf K=\nabla^2\Psi=-\frac{\partial\mathbf f}{\partial\mathbf q}\ \succeq 0,
\qquad
\mathbf C=-\frac{\partial\mathbf f}{\partial\mathbf v}\ \succeq 0,
$$

so the system matrix is the symmetric positive-definite **step matrix** (mass, stiffness, and damping all
add):

$$
\frac{\partial\mathbf r}{\partial\mathbf v}=\underbrace{\frac{\mathbf M}{\Delta t}+\Delta t\,\mathbf K+\mathbf C}_{=\ \mathbf M_a\ \text{(step matrix / Hessian)}}\ \succ 0 .
$$

One Newton step is therefore $\mathbf M_a\,\Delta\mathbf v=-\mathbf r(\mathbf v^{(i)})$, followed by
$\mathbf v^{(i+1)}=\mathbf v^{(i)}+\Delta\mathbf v$; iterating until $\mathbf r\to\mathbf 0$ *is* the implicit
solve. The compact $\mathbf M_a\mathbf v_a=\mathbf f_a$ of §8.1 is just this relation rolled up — $\mathbf M_a$
is the converged step matrix and $\mathbf f_a$ collects $-\mathbf r$ plus the $\mathbf M_a\mathbf v^{(i)}$
shift, i.e. the inertial predictor and all force terms evaluated at the linearization point.

> **Two different "$\mathbf v$"s — which is the unknown?** In the *coupling* equation
> $\mathbf M_a\mathbf v_a=\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda$ (§8.1), the unknown $\mathbf v_a$ is
> the **absolute end-of-step velocity** — the quantity each sub-solver owns and reconciles back into the
> global state. That works only because $\mathbf f_a$ is *defined* to absorb the inertial carry-over
> (the $\mathbf M_a\mathbf v^{-}$ momentum term), so $\mathbf v_a=\mathbf M_a^{-1}\mathbf f_a$ comes out
> absolute. In the *linear systems the integrators actually solve* — the Newton step
> $\mathbf M_a\,\Delta\mathbf v=-\mathbf r$ here, and MuJoCo's
> $(\mathbf M-h\,\partial\mathbf f/\partial\dot{\mathbf q})\,\Delta\dot{\mathbf q}=h\mathbf f$ in §12.2 — the
> unknown is the **velocity increment** $\Delta\mathbf v$, *not* the absolute velocity. The two are the same
> equation shifted by the carry-over: $\mathbf v_a=\mathbf v^{-}+\Delta\mathbf v$, with
> $\mathbf f_a$ holding the $\mathbf M_a\mathbf v^{-}$ term that the increment form keeps on the left. So:
> abstract $\mathbf v_a$ = absolute; implemented unknown = update.

*Equivalent variational view — why this is "the local quadratic model."* When the forces are conservative,
implicit Euler is the stationarity of an **incremental potential**

$$
E(\mathbf q)=\frac{1}{2\Delta t^{2}}\lVert\mathbf q-\tilde{\mathbf q}\rVert_{\mathbf M}^{2}+\Psi(\mathbf q),
\qquad
\tilde{\mathbf q}=\mathbf q^{-}+\Delta t\,\mathbf v^{-}+\Delta t^{2}\,\mathbf M^{-1}\mathbf f_{\text{ext}},
$$

where the inertial predictor $\tilde{\mathbf q}$ is where the body would drift under explicit forces alone;
$\nabla E=\mathbf M(\mathbf q-\tilde{\mathbf q})/\Delta t^{2}+\nabla\Psi=\mathbf 0$ is the same equation as
$\mathbf r=\mathbf 0$. Taylor-expanding $E$ to **second order** about $\mathbf q^{(i)}$,

$$
E(\mathbf q^{(i)}+\Delta\mathbf q)\approx E^{(i)}+\nabla E^{\top}\Delta\mathbf q+\tfrac12\,\Delta\mathbf q^{\top}\underbrace{\big(\mathbf M/\Delta t^{2}+\mathbf K\big)}_{\nabla^2 E\ =\ \text{Hessian}}\Delta\mathbf q,
$$

and **minimizing this quadratic model** gives $(\mathbf M/\Delta t^{2}+\mathbf K)\,\Delta\mathbf q=-\nabla E$
— identical to the Newton row above after substituting $\Delta\mathbf q=\Delta t\,\Delta\mathbf v$. So
"linearizing the residual $\mathbf r$" and "minimizing the local quadratic (second-order) model of the
energy $E$" are the same operation, and $\mathbf M_a=\nabla^2 E$ is that model's Hessian. **This view is
exact for the cloth (VBD/IPC, §12.3); for the arm it only partly applies** — see the caveat in §12.2.

### 12.2 MuJoCo arm — `implicitfast`, derived

**Where the equation of motion comes from — the Lagrangian derivation.** *(Notation: the index $\ell$ runs
over individual rigid bodies/links, kept distinct from the coupling's side $a$/side $b$.)* The arm has
$n=9$ joint coordinates $\mathbf q$. Forward kinematics places each rigid body $\ell$, and its spatial
velocity $\boldsymbol\xi_\ell=(\boldsymbol\omega_\ell;\mathbf v_\ell)$ is **linear in the joint rates**,
$\boldsymbol\xi_\ell=\mathbf J_\ell(\mathbf q)\,\dot{\mathbf q}$, where
$\mathbf J_\ell(\mathbf q)=\partial\boldsymbol\xi_\ell/\partial\dot{\mathbf q}$ is body $\ell$'s geometric
Jacobian (it depends on $\mathbf q$ because the geometry rotates/translates as the joints move). The
**kinetic energy** is the sum of each body's $\tfrac12\boldsymbol\xi_\ell^\top\mathbb M_\ell\boldsymbol\xi_\ell$ with
$\mathbb M_\ell=\operatorname{diag}(\mathbf I_\ell,\,m_\ell\mathbf I_3)$ the body's $6\times6$ spatial inertia — here
$\mathbf I_\ell$ is body $\ell$'s $3\times3$ **rotational inertia tensor** (about its CoM, paired with
$\boldsymbol\omega_\ell$) and $\mathbf I_3$ is the $3\times3$ **identity**, so $m_\ell\mathbf I_3$ is the
translational block (paired with $\mathbf v_\ell$); hence $\tfrac12\boldsymbol\xi_\ell^\top\mathbb M_\ell\boldsymbol\xi_\ell=\tfrac12(\boldsymbol\omega_\ell^\top\mathbf I_\ell\boldsymbol\omega_\ell+m_\ell\lVert\mathbf v_\ell\rVert^2)$:

$$
T(\mathbf q,\dot{\mathbf q})=\tfrac12\sum_\ell\boldsymbol\xi_\ell^\top\mathbb M_\ell\boldsymbol\xi_\ell
=\tfrac12\,\dot{\mathbf q}^\top\underbrace{\Big(\sum_\ell\mathbf J_\ell(\mathbf q)^\top\mathbb M_\ell\,\mathbf J_\ell(\mathbf q)\Big)}_{=\ \mathbf M(\mathbf q)}\dot{\mathbf q}.
$$

> *Where $\mathbf I_\ell$ (and $m_\ell$) come from — a density $\rho(\mathbf x)$ over the body's shape.* With
> mass $m_\ell=\int_V\rho\,dV$ and center of mass $\mathbf x_{cm}=\tfrac1{m_\ell}\int_V\rho\,\mathbf x\,dV$, set
> $\mathbf r=\mathbf x-\mathbf x_{cm}$. The CoM inertia tensor (body frame) is
> $$
> \mathbf I_\ell=\int_V\rho(\mathbf x)\,\big(\lVert\mathbf r\rVert^2\mathbf I_3-\mathbf r\,\mathbf r^\top\big)\,dV
> =\begin{bmatrix}
> \int\rho(y^2{+}z^2) & -\int\rho\,xy & -\int\rho\,xz\\
> -\int\rho\,xy & \int\rho(x^2{+}z^2) & -\int\rho\,yz\\
> -\int\rho\,xz & -\int\rho\,yz & \int\rho(x^2{+}y^2)
> \end{bmatrix},
> $$
> with $\mathbf r=(x,y,z)$ from the CoM — diagonal = moments of inertia, off-diagonal = (negative) products
> of inertia; symmetric PSD. *Derivation:* a point at $\mathbf r$ spinning at $\boldsymbol\omega$ moves at
> $\boldsymbol\omega\times\mathbf r$, and $\lVert\boldsymbol\omega\times\mathbf r\rVert^2=\boldsymbol\omega^\top(\lVert\mathbf r\rVert^2\mathbf I_3-\mathbf r\mathbf r^\top)\boldsymbol\omega$, so
> $\tfrac12\int_V\rho\lVert\boldsymbol\omega\times\mathbf r\rVert^2dV=\tfrac12\boldsymbol\omega^\top\mathbf I_\ell\boldsymbol\omega$
> — the rotational term above. *Shortcut:* with the second-moment matrix $\mathbf C=\int_V\rho\,\mathbf r\mathbf r^\top dV$,
> $\mathbf I_\ell=\operatorname{tr}(\mathbf C)\mathbf I_3-\mathbf C$. *Reference point:* about a point offset
> $\mathbf d$ from the CoM use parallel-axis $\mathbf I=\mathbf I_\ell+m_\ell(\lVert\mathbf d\rVert^2\mathbf I_3-\mathbf d\mathbf d^\top)$;
> in the world frame it rotates as $\mathbf R\mathbf I_\ell\mathbf R^\top$. For a **mesh** (uniform $\rho$) these
> volume integrals reduce to triangle surface integrals (Mirtich's polyhedral algorithm) or a sum of
> closed-form per-tetrahedron terms — how MuJoCo/Newton fill `body_inertia` from geometry.

This **defines the joint-space mass matrix** $\mathbf M(\mathbf q)=\sum_\ell\mathbf J_\ell^\top\mathbb M_\ell\mathbf J_\ell$:
it is the pull-back of the bodies' spatial inertias through their Jacobians (MuJoCo computes it with the
Composite-Rigid-Body Algorithm, `qM`; `joint_armature` adds rotor inertia to the diagonal). It is symmetric
positive-definite and **configuration-dependent precisely because the $\mathbf J_b(\mathbf q)$ are** — i.e.
$\mathbf M$ is a (Riemannian) *metric* on configuration space, not a constant.

With potential energy $V(\mathbf q)=-\sum_\ell m_\ell\,\mathbf g^\top\mathbf p_\ell(\mathbf q)$ (gravity; plus any joint
springs), the Lagrangian is $L=T-V$ and the **Euler–Lagrange** equations
$\tfrac{d}{dt}\big(\partial L/\partial\dot{\mathbf q}\big)-\partial L/\partial\mathbf q=\boldsymbol\tau$ give,
term by term:

$$
\frac{\partial L}{\partial\dot{\mathbf q}}=\mathbf M(\mathbf q)\dot{\mathbf q},
\qquad
\frac{d}{dt}\big(\mathbf M\dot{\mathbf q}\big)=\mathbf M\ddot{\mathbf q}+\dot{\mathbf M}\dot{\mathbf q},
\qquad
\frac{\partial L}{\partial\mathbf q}=\tfrac12\dot{\mathbf q}^\top\frac{\partial\mathbf M}{\partial\mathbf q}\dot{\mathbf q}-\frac{\partial V}{\partial\mathbf q},
$$

so

$$
\mathbf M(\mathbf q)\ddot{\mathbf q}+\underbrace{\Big(\dot{\mathbf M}\dot{\mathbf q}-\tfrac12\nabla_{\mathbf q}\big(\dot{\mathbf q}^\top\mathbf M\dot{\mathbf q}\big)\Big)}_{=\ \mathbf C(\mathbf q,\dot{\mathbf q})\,\dot{\mathbf q}\ \text{(Coriolis/centrifugal)}}+\underbrace{\nabla_{\mathbf q}V}_{\text{gravity }\mathbf g(\mathbf q)}=\boldsymbol\tau .
$$

The velocity-quadratic middle term is the **Coriolis/centrifugal** force; its $i$-th component is
$\big(\mathbf C\dot{\mathbf q}\big)_i=\sum_{j,k}\Gamma_{ijk}\,\dot q_j\dot q_k$ with the Christoffel symbols

$$
\Gamma_{ijk}=\tfrac12\Big(\frac{\partial M_{ij}}{\partial q_k}+\frac{\partial M_{ik}}{\partial q_j}-\frac{\partial M_{jk}}{\partial q_i}\Big),
$$

where $M_{ij}$ is the scalar $(i,j)$ **entry of the joint-space mass matrix $\mathbf M(\mathbf q)$** derived
above ($\mathbf M=\sum_\ell\mathbf J_\ell^\top\mathbb M_\ell\mathbf J_\ell\in\mathbb R^{9\times9}$ — *not* the per-body
spatial inertia $\mathbb M_\ell$), and $\partial M_{ij}/\partial q_k$ is that entry's derivative w.r.t. joint
coordinate $q_k$. So the Coriolis term **comes entirely from the configuration-dependence of $\mathbf M$**: if $\mathbf M$
were constant ($\partial\mathbf M/\partial\mathbf q=0$) it would vanish. It is a *fictitious* force from the
non-constant metric — quadratic in $\dot{\mathbf q}$, and **not the gradient of any potential** (this is the
precise reason the arm has no single incremental potential, §12.1). MuJoCo lumps gravity + Coriolis +
centrifugal into one **bias** vector $\mathbf c(\mathbf q,\dot{\mathbf q})=\mathbf C\dot{\mathbf q}+\mathbf g$,
which it evaluates cheaply by the Recursive Newton–Euler Algorithm at zero acceleration,
$\mathbf c=\mathrm{RNE}(\mathbf q,\dot{\mathbf q},\ddot{\mathbf q}=\mathbf 0)$ (`qfrc_bias`).

**The total smooth force.** Writing the EOM as $\mathbf M(\mathbf q)\ddot{\mathbf q}+\mathbf c(\mathbf q,\dot{\mathbf q})=\boldsymbol\tau_{\text{act}}+\boldsymbol\tau_{\text{passive}}+\boldsymbol\tau_{\text{applied}}$
(the right side is the *applied* generalized force $\boldsymbol\tau$):

$$
\mathbf M(\mathbf q)\,\ddot{\mathbf q}+\mathbf c(\mathbf q,\dot{\mathbf q})=\boldsymbol\tau_{\text{act}}+\boldsymbol\tau_{\text{passive}}+\boldsymbol\tau_{\text{applied}},
$$

with $\mathbf M(\mathbf q)$ and $\mathbf c=\mathbf C\dot{\mathbf q}+\mathbf g$ as just derived. Collect the
**total smooth generalized force** so that $\mathbf M\ddot{\mathbf q}=\mathbf f$:

$$
\mathbf f(\mathbf q,\dot{\mathbf q})=\underbrace{\mathbf K_p(\mathbf q^{*}-\mathbf q)-\mathbf K_d\dot{\mathbf q}}_{\boldsymbol\tau_{\text{act}}\ (\text{PD servo})}+\boldsymbol\tau_{\text{passive}}+\boldsymbol\tau_{\text{applied}}-\mathbf c(\mathbf q,\dot{\mathbf q}),
$$

with our IsaacLab gains $\mathbf K_p=\operatorname{diag}(4000{\times}7,\ 4{\cdot}10^{4}{\times}2)$ and
$\mathbf K_d=\operatorname{diag}(400{\times}9)$, and $\mathbf q^{*}$ the IK joint target.

> **Note — the arm has no single incremental potential.** Unlike the cloth (§12.1/§12.3), MuJoCo's dynamics
> are *not* the gradient of one energy: gravity and joint springs are conservative, but the
> **Coriolis/centrifugal bias $\mathbf c$ is not a gradient of any potential**, and contacts/limits are
> solved as a separate **convex constraint problem** (a QP over accelerations in the friction cone), not a
> smooth potential. So the correct object for the arm is the **force-balance residual** below, not an
> energy $E$ — the "minimize $E$" picture of §12.1 is exact only on the cloth side.

**The `implicitfast` velocity update.** MuJoCo integrates velocity with a backward-Euler step that is
implicit **only in velocity**: linearize $\mathbf f$ in $\dot{\mathbf q}$ about $\dot{\mathbf q}^{-}$,

$$
\mathbf M\,\frac{\dot{\mathbf q}^{+}-\dot{\mathbf q}^{-}}{h}=\mathbf f(\mathbf q^{-},\dot{\mathbf q}^{+})\approx\mathbf f(\mathbf q^{-},\dot{\mathbf q}^{-})+\frac{\partial\mathbf f}{\partial\dot{\mathbf q}}\,\Delta\dot{\mathbf q},
$$

$$
\boxed{\ \Big(\mathbf M-h\,\frac{\partial\mathbf f}{\partial\dot{\mathbf q}}\Big)\,\Delta\dot{\mathbf q}=h\,\mathbf f(\mathbf q^{-},\dot{\mathbf q}^{-})\ },\qquad \Delta\dot{\mathbf q}=\dot{\mathbf q}^{+}-\dot{\mathbf q}^{-},
$$

then position updates semi-implicitly $\mathbf q^{+}=\mathbf q^{-}+h\,\dot{\mathbf q}^{+}$. Matching §8.1: the
step matrix is $\mathbf M_a=\mathbf M-h\,\partial\mathbf f/\partial\dot{\mathbf q}$ and the right-hand side is
$h\,\mathbf f$ — the coupling wrench $\mathbf J_a^\top\boldsymbol\lambda$ (gripper $\leftrightarrow$ cloth) and
MuJoCo's own constraint impulses $\mathbf J_c^\top\boldsymbol\lambda_c$ (its internal contacts/limits) are
added to that RHS and solved together.

**$\partial\mathbf f/\partial\dot{\mathbf q}$ — what is in it, what `implicitfast` drops.** Differentiating
the total force in velocity,

$$
\frac{\partial\mathbf f}{\partial\dot{\mathbf q}}=\underbrace{\frac{\partial\boldsymbol\tau_{\text{act}}}{\partial\dot{\mathbf q}}}_{-\mathbf K_d}+\underbrace{\frac{\partial\boldsymbol\tau_{\text{passive}}}{\partial\dot{\mathbf q}}}_{-\mathbf D\ (\text{joint/fluid damping})}-\underbrace{\frac{\partial\mathbf c}{\partial\dot{\mathbf q}}}_{\text{Coriolis/centrifugal derivative}}.
$$

- **`implicit`** keeps all three (the $-\partial\mathbf c/\partial\dot{\mathbf q}$ is the expensive analytic
  RNE velocity-derivative).
- **`implicitfast`** (what we run) **drops $-\partial\mathbf c/\partial\dot{\mathbf q}$** — the
  Coriolis/centrifugal derivative — keeping only the cheap diagonal damping terms:

$$
\frac{\partial\mathbf f}{\partial\dot{\mathbf q}}\Big|_{\text{implicitfast}}\approx-\mathbf K_d-\mathbf D\quad(\text{here }\approx-\operatorname{diag}(400{\times}9),\ \text{passive }\mathbf D\approx\mathbf 0),
$$

giving the SPD step matrix

$$
\mathbf M_a=\mathbf M(\mathbf q)+h(\mathbf K_d+\mathbf D)\approx\mathbf M(\mathbf q)+h\,\operatorname{diag}(400).
$$

The damping gain $\mathbf K_d$ enters $\mathbf M_a$ (stabilizing) — that is the *point* of the implicit
treatment: it lets a large $\mathbf K_d$ be used without explicit-integration blow-up.

**$\partial\mathbf f/\partial\dot{\mathbf q}$ vs $\partial\mathbf f/\partial\mathbf v$ — the same matrix here.**
MuJoCo's generalized velocity is `qvel` $=\mathbf v$, and for hinge/slide joints (all 9 of the Franka's) it
*is* the time-derivative of `qpos`, $\mathbf v=\dot{\mathbf q}$ — so $\partial\mathbf f/\partial\dot{\mathbf q}\equiv\partial\mathbf f/\partial\mathbf v$.
The distinction matters only for **ball/free** joints, where `qpos` is a quaternion (dim 4 / 7) but `qvel` is
an angular/spatial velocity (dim 3 / 6); then $\dot{\mathbf q}$ and $\mathbf v$ live in different spaces tied
by the joint's tangent map and MuJoCo always differentiates w.r.t. `qvel` $=\mathbf v$. The Franka has none.

**$\partial\mathbf f/\partial\mathbf q$ (the stiffness) is *not* in the matrix.** The position-derivative

$$
\frac{\partial\mathbf f}{\partial\mathbf q}=\underbrace{\frac{\partial\boldsymbol\tau_{\text{act}}}{\partial\mathbf q}}_{-\mathbf K_p}+\frac{\partial\boldsymbol\tau_{\text{passive}}}{\partial\mathbf q}-\frac{\partial\mathbf c}{\partial\mathbf q}\qquad(\text{dominant term }-\mathbf K_p=-\operatorname{diag}(4000,\,4{\cdot}10^{4}))
$$

is **omitted** by both `implicit` and `implicitfast`: neither folds $\partial\mathbf f/\partial\mathbf q$ into
$\mathbf M_a$ (contrast the cloth, §12.1, where $\Delta t\,\mathbf K$ *is* in the Hessian). The PD position
stiffness $\mathbf K_p$ is thus integrated **explicitly** (the actuator force uses the start-of-step
$\mathbf q$). This is why a very stiff $\mathbf K_p$ (the $4{\cdot}10^{4}$ finger gain) needs the
implicit-velocity damping and a small $h$ to stay stable, and it is the technical reason §8.1's arm
$\mathbf M_a$ carries **no** stiffness term while the cloth's does.

### 12.3 VBD cloth — block Gauss–Seidel local model

For the cloth the variational picture of §12.1 is **exact**: VBD minimizes the incremental potential

$$
E(\mathbf x)=\frac{1}{2\Delta t^{2}}\lVert\mathbf x-\tilde{\mathbf x}\rVert_{\mathbf M}^{2}+\Psi_{\text{mem}}(\mathbf x)+\Psi_{\text{bend}}(\mathbf x)+B_{\text{self}}(\mathbf x)
$$

over the $3N$ nodal positions ($N=6436$): $\Psi_{\text{mem}}$ = Baraff–Witkin membrane, $\Psi_{\text{bend}}$ =
discrete-shell bending, $B_{\text{self}}$ = self-contact penalty. VBD does **block coordinate descent** — it
sweeps vertices, and for each vertex $i$ solves a local $3\times3$ Newton step

$$
\mathbf H_i\,\Delta\mathbf x_i=-\nabla_{\!i}E,\qquad
\mathbf H_i=\frac{m_i}{\Delta t^{2}}\mathbf I_3+\sum_{e\ni i}\nabla^2_{\!i}\Psi_e,
$$

(lumped vertex mass $+$ the Hessians of the membrane/bending elements incident to $i$), for
`vbd_iterations = 20` sweeps per substep. Here the side-$b$ inertia $\mathbf M_b=\operatorname{diag}(m_i\mathbf I_3)$
is genuinely **diagonal**, and the per-vertex $\mathbf H_i$ is the local analog of the step matrix
$\mathbf M_a$ — *with* the stiffness $\nabla^2\Psi$ folded in (cloth is stiffness-implicit, unlike the arm).

### 12.4 Why the coupling needs only the effective response

Each sub-solver runs **many** internal iterations (`mujoco_iterations = 100`, `vbd_iterations = 20`), so
neither literally solves a single linear $\mathbf M_a\mathbf v_a=\mathbf f_a$. The coupling of §8 does not
care: it needs only the **input $\to$ output map** — "inject generalized force $\mathbf f_a$ (for the arm, the
wrench $\mathbf J_a^\top\boldsymbol\lambda$ deposited on the gripper body; for the cloth, the contact force on
the vertices) and read back the converged velocity $\mathbf v_a$." The **step inertia $\mathbf M_a$**
(glossary, §8.0) is the effective local inertia of that map, $\mathbf v_a\approx\mathbf M_a^{-1}\mathbf f_a$
— exactly the quantity that sets the convergence factor $r(s,q)$ of §8.4, with the theory scale
$s=\hat{\mathbf M}_a\big/(\mathbf J_a\mathbf M_a^{-1}\mathbf J_a^\top)^{-1}$ measured against **this true
$\mathbf M_a$** — not the bare mass $\mathbf M(\mathbf q)$, and not the code's home-pose estimate
$M_a^{\text{eff}}$ (which is what the proxy is actually built from; see §8.6, item 5). That is what lets §8
treat each native solver as a black box and still reason about both the fixed point and the rate.

---

## 13. Appendix — effective (inverse) mass

The effective inverse-inertia operator is the object behind §8.1's Delassus $\mathbf D$, §8.0's
`body_invweight0`/$M_a^{\text{eff}}$, and the scale $s$ (§8.4). This appendix defines it strictly and derives
its formulas.

### 13.1 Definitions

Fix a mechanical system with generalized velocity $\mathbf v\in\mathbb R^{n}$ and symmetric positive-definite
generalized inertia $\mathbf M\in\mathbb R^{n\times n}$ (for the arm, the step inertia $\mathbf M_a$ of §12).
An **interface** is a point/direction set specified by a Jacobian $\mathbf J\in\mathbb R^{d\times n}$ whose
rows map $\mathbf v$ to the interface velocity $\mathbf u=\mathbf J\mathbf v\in\mathbb R^{d}$ ($d=1$ for a
single normal, $3$ for a point, $6$ for a body wrench, $c$ for $c$ stacked contact rows).

> **Definition 1 (effective inverse *generalized inertia* operator).**
> $$\mathbf A:=\mathbf J\,\mathbf M^{-1}\mathbf J^\top\in\mathbb R^{d\times d}.$$
> $\mathbf A$ is symmetric and positive semidefinite (positive definite iff $\mathbf J$ has full row rank $d$).
> Its rows/columns carry the units of the interface DOFs: for a $6$-DOF body interface it has a
> **translational** block (units of inverse *mass*), a **rotational** block (inverse *moment of inertia*),
> and their coupling — the $\mathbf A_{tt},\mathbf A_{rr},\mathbf A_{tr}$ of §13.3.
>
> **Definition 2 (effective inverse inertia along a unit direction $\mathbf n\in\mathbb R^{d}$).**
> $$\frac{1}{\iota_{\text{eff}}(\mathbf n)}:=\mathbf n^\top\mathbf A\,\mathbf n\ \ge 0,\qquad \iota_{\text{eff}}(\mathbf n):=\big(\mathbf n^\top\mathbf A\,\mathbf n\big)^{-1}.$$
> The scalar $\iota_{\text{eff}}$ is an **effective mass** $m_{\text{eff}}$ when $\mathbf n$ is a
> *translational* direction, and an **effective moment of inertia** $I_{\text{eff}}$ when $\mathbf n$ is a
> *rotational* one.

The **operator** $\mathbf A$ is a *matrix*; the **effective (inverse) inertia** is the *scalar* obtained from
it after fixing a direction $\mathbf n$. Because translational and rotational directions carry **different
units**, a $6$-vector $\mathbf n$ mixing the two is dimensionally inhomogeneous — the two subspaces are kept
separate (which is exactly why `body_invweight0` reports *two* scalars, §13.4). These are the only objects;
nothing below is used loosely.

### 13.2 Operational meaning (derivation of Def. 1)

> **Claim.** $\mathbf A$ is exactly the linear map from an applied interface impulse to the resulting
> interface velocity change: $\Delta\mathbf u=\mathbf A\,\boldsymbol\pi$.

*Proof.* Apply an interface impulse $\boldsymbol\pi\in\mathbb R^{d}$ (an impulse conjugate to $\mathbf u$).

1. **Pull-back (virtual work).** An interface impulse does work only through $\mathbf u=\mathbf J\mathbf v$, so
   it injects the generalized impulse $\mathbf J^\top\boldsymbol\pi$ (the adjoint of the velocity map).
2. **Impulse–momentum.** Integrating $\mathbf M\dot{\mathbf v}=(\text{force})$ over the impulse,
   $\mathbf M\,\Delta\mathbf v=\mathbf J^\top\boldsymbol\pi$, hence $\Delta\mathbf v=\mathbf M^{-1}\mathbf J^\top\boldsymbol\pi$.
3. **Push-forward.** $\Delta\mathbf u=\mathbf J\,\Delta\mathbf v=\mathbf J\mathbf M^{-1}\mathbf J^\top\boldsymbol\pi=\mathbf A\,\boldsymbol\pi.$ $\qquad\blacksquare$

Along a unit $\mathbf n$ with $\boldsymbol\pi=j\,\mathbf n$: $\ \mathbf n^\top\Delta\mathbf u=(\mathbf n^\top\mathbf A\,\mathbf n)\,j=\dfrac{j}{\iota_{\text{eff}}(\mathbf n)}$,
i.e. $\ j=\iota_{\text{eff}}(\mathbf n)\,\big(\mathbf n^\top\Delta\mathbf u\big)$ — impulse $=$ (effective inertia)
$\times$ (velocity change). That is precisely why $\mathbf n^\top\mathbf A\mathbf n$ is the effective *inverse
inertia* along $\mathbf n$: for a translational $\mathbf n$ it reads $j=m_{\text{eff}}\,\Delta v$ (a point
mass's $\Delta v=j/m$); for a rotational $\mathbf n$, $\ell=I_{\text{eff}}\,\Delta\omega$.

*Why the along-$\mathbf n$ projection?* The concept is **not** contact-specific: $\mathbf n^\top\mathbf A\mathbf n$
is the effective inverse inertia **in the one-dimensional channel spanned by $\mathbf n$**. Restrict any
interaction to that channel — apply an impulse confined to it, $\boldsymbol\pi=j\,\mathbf n$ (scalar $j$), and
read the response in the same channel, $\mathbf n^\top\Delta\mathbf u$ — and Def. 1 gives one scalar law,
$$
\mathbf n^\top\Delta\mathbf u=(\mathbf n^\top\mathbf A\,\mathbf n)\,j\ \Longrightarrow\ j=\frac{\mathbf n^\top\Delta\mathbf u}{\mathbf n^\top\mathbf A\,\mathbf n}=\iota_{\text{eff}}(\mathbf n)\,\big(\mathbf n^\top\Delta\mathbf u\big).
$$
A scalar drive matched to a scalar response admits only this one equation — the projection onto $\mathbf n$ —
so $\mathbf n^\top\mathbf A\mathbf n$ is what closes it. This holds for *any* single scalar interface along
$\mathbf n$ (a 1-DOF joint reaction, a prescribed-direction actuator, a scalar attachment), independent of
contact.

**Contact is the archetype** (and this doc's use, §8.1). There the channel is *forced*: a frictionless
contact can push only along the normal and its law constrains only $u_n=\mathbf n^\top\mathbf u$
(Signorini/restitution), giving $j=\iota_{\text{eff}}(\mathbf n)\,\Delta u_n$. The impulse direction and the
constrained-velocity direction are the *same* $\mathbf n$ by the ideal-constraint structure: gap $g$,
$\mathbf n\propto\nabla g$, velocity row $\dot g=\mathbf n^\top\mathbf u$ (so $\mathbf J_c=\mathbf n^\top$), and
by d'Alembert the reaction is $\boldsymbol\pi=\mathbf J_c^\top\lambda=\mathbf n\,\lambda$. (In every case
$\mathbf n$ is **not** the direction of $\Delta\mathbf u=j\,\mathbf A\mathbf n$ — off-axis for anisotropic
$\mathbf A$; and in the bare Def. 2, $\mathbf n$ is simply *any* query direction.)

### 13.3 Worked case — one rigid body (the full $6\times6$ operator)

Take a single free rigid body: $\mathbf v=(\mathbf v_c;\boldsymbol\omega)$ (CoM linear velocity, angular
velocity), $\mathbf M=\operatorname{diag}(m\mathbf I_3,\ \mathbf I)$ with $\mathbf I$ the CoM inertia tensor.
Its **6-DOF** interface $\mathbf u=(\mathbf u_P;\boldsymbol\omega)=\mathbf J_b\mathbf v$ has operator
$$
\mathbf A=\mathbf J_b\mathbf M^{-1}\mathbf J_b^\top=\begin{bmatrix}\mathbf A_{tt}&\mathbf A_{tr}\\ \mathbf A_{tr}^\top&\mathbf A_{rr}\end{bmatrix}
\quad\Big(\mathbf A_{tt}=\text{inverse \textbf{mass}},\ \ \mathbf A_{rr}=\text{inverse \textbf{rotational inertia}},\ \ \mathbf A_{tr}=\text{coupling}\Big).
$$
We derive both diagonal blocks.

**Translational block $\mathbf A_{tt}$ — two ways.** A material point $P$ at offset $\mathbf r$ from the CoM
has $\mathbf u_P=\mathbf v_c+\boldsymbol\omega\times\mathbf r=\mathbf J_P\mathbf v$ with
$\mathbf J_P=[\,\mathbf I_3\ \ -[\mathbf r]_\times\,]$ (using $\boldsymbol\omega\times\mathbf r=-[\mathbf r]_\times\boldsymbol\omega$).

*(a) Operator route (Def. 1):*
$$
\mathbf A_{tt}=\mathbf J_P\mathbf M^{-1}\mathbf J_P^\top=\tfrac1m\mathbf I_3+[\mathbf r]_\times\mathbf I^{-1}[\mathbf r]_\times^\top,\qquad
\frac{1}{m_{\text{eff}}(\mathbf n)}=\mathbf n^\top\mathbf A_{tt}\,\mathbf n=\frac1m+(\mathbf r\times\mathbf n)^\top\mathbf I^{-1}(\mathbf r\times\mathbf n),
$$
using $[\mathbf r]_\times^\top\mathbf n=-(\mathbf r\times\mathbf n)$.

*(b) Newton–Euler check:* an impulse $j\,\mathbf n$ at $P$ gives $\Delta\mathbf v_c=\tfrac{j}{m}\mathbf n$ and
$\Delta\boldsymbol\omega=j\,\mathbf I^{-1}(\mathbf r\times\mathbf n)$; with $\Delta\mathbf u_P=\Delta\mathbf v_c+\Delta\boldsymbol\omega\times\mathbf r$,
$$
\mathbf n^\top\Delta\mathbf u_P=j\Big[\tfrac1m+(\mathbf r\times\mathbf n)^\top\mathbf I^{-1}(\mathbf r\times\mathbf n)\Big]
$$
via $\mathbf n\cdot(\mathbf w\times\mathbf r)=(\mathbf r\times\mathbf n)\cdot\mathbf w$. Same result. The second
term is the rotational contribution through the lever $\mathbf r$ — it enlarges the *point's* mobility but
does **not** alter $\mathbf I$.

**Rotational block $\mathbf A_{rr}$.** Take the interface to be the angular velocity, $\boldsymbol\omega=\mathbf J_r\mathbf v$
with $\mathbf J_r=[\,\mathbf 0\ \ \mathbf I_3\,]$ for a free body. Then
$$
\mathbf A_{rr}=\mathbf J_r\mathbf M^{-1}\mathbf J_r^\top=[\,\mathbf 0\ \ \mathbf I_3\,]\operatorname{diag}(\tfrac1m\mathbf I_3,\ \mathbf I^{-1})[\,\mathbf 0\ \ \mathbf I_3\,]^\top=\mathbf I^{-1},
$$
i.e. a torque-impulse $\boldsymbol\ell$ gives $\Delta\boldsymbol\omega=\mathbf I^{-1}\boldsymbol\ell$; the
effective inverse rotational inertia is simply $\mathbf I^{-1}$ (along an axis $\mathbf a$,
$1/I_{\text{eff}}(\mathbf a)=\mathbf a^\top\mathbf I^{-1}\mathbf a$). It is *trivial* for a free body — which is
why §13.3(a,b) spent the effort on the translational block — but for the **articulated** arm it is the
non-trivial $\mathbf J_r\mathbf M^{-1}\mathbf J_r^\top$ (the whole chain), and it is exactly what
`body_invweight0`'s $\text{inv\_rot}$ captures.

At the CoM ($\mathbf r=0$) the whole operator is block-diagonal,
$\mathbf A=\mathbf M^{-1}=\operatorname{diag}(\tfrac1m\mathbf I_3,\ \mathbf I^{-1})$ — inverse mass and inverse
rotational inertia, no coupling; the lever term (a) and the coupling $\mathbf A_{tr}$ appear only for
$\mathbf r\ne0$. The proxy needs **both** diagonal blocks: the scalar mass from $\mathbf A_{tt}$ and the
$3\times3$ inertia from $\mathbf A_{rr}$ (§13.4, §8.0).

### 13.4 Specializing $\mathbf A$: the Delassus sum (§8.1) and the `body_invweight0` scalars (§8.0)

**Additivity (reduced mass).** For a contact between systems $A$ and $B$ with relative interface velocity
$\mathbf u=\mathbf J_A\mathbf v_A+\mathbf J_B\mathbf v_B$, the relative-motion operator is the **sum**
$\mathbf A=\mathbf A_A+\mathbf A_B$ (inverse inertias add). Here $\mathbf n$ is the **contact normal** (§13.2 —
the direction the impulse acts and the law constrains; a *tangent* $\mathbf n$ gives a friction row). Along
it, $1/m_{\text{eff}}=1/m_{\text{eff},A}+1/m_{\text{eff},B}$ (the two-body **reduced mass**), and the impulse
removing a relative approach velocity is $j=-m_{\text{eff}}\,(\mathbf n^\top\Delta\mathbf u_{\text{rel}})$.
This sum is exactly §8.1's Delassus $\mathbf D=\mathbf J_a\mathbf M_a^{-1}\mathbf J_a^\top+\mathbf J_b\mathbf M_b^{-1}\mathbf J_b^\top$.

**Scalar reduction.** Replacing the $d\times d$ operator $\mathbf A$ by a single scalar requires either a
fixed direction ($\mathbf n^\top\mathbf A\mathbf n$) or an isotropy assumption. MuJoCo's `body_invweight0`
takes the latter: for a body's $6\times6$ $\mathbf A$ it stores the two block trace-averages
$\text{inv\_mass}=\tfrac13\sum_{i=0}^{2}A_{ii}$, $\text{inv\_rot}=\tfrac13\sum_{i=3}^{5}A_{ii}$, discarding
$\mathbf A$'s anisotropy and translation–rotation coupling. This is the $M_a^{\text{eff}}$ of §8.0.

**Where the $\tfrac23$ in $M_a^{\text{eff}}$ comes from.** Start from the exact §13.3 per-direction inverse
mass at offset $\mathbf r$, push direction $\mathbf n$, and apply the code's isotropic reduction
$\mathbf I^{-1}=\tfrac1I\mathbf I_3$ (only scalars are stored):
$$
\frac{1}{m_{\text{eff}}(\mathbf n)}=\frac1m+(\mathbf r\times\mathbf n)^\top\mathbf I^{-1}(\mathbf r\times\mathbf n)=\frac1m+\frac1I\,\lVert\mathbf r\times\mathbf n\rVert^2 .
$$
No push direction is supplied, so **average this whole scalar over $\mathbf n$** ($\mathbb E_{\mathbf n}[f]:=\tfrac{1}{4\pi}\int_{S^2}f\,d\Omega$):
$$
\mathbb E_{\mathbf n}\!\left[\frac{1}{m_{\text{eff}}(\mathbf n)}\right]=\frac1m+\frac1I\,\mathbb E_{\mathbf n}\!\big[\lVert\mathbf r\times\mathbf n\rVert^2\big]=\frac1m+\frac{2}{3}\frac{\lVert\mathbf r\rVert^2}{I},
$$
since $1/m$ is constant in $\mathbf n$ and
$\mathbb E_{\mathbf n}\!\big[\lVert\mathbf r\times\mathbf n\rVert^2\big]=\mathbb E_{\mathbf n}\!\big[\lVert\mathbf r\rVert^2-(\mathbf r\!\cdot\!\mathbf n)^2\big]=\lVert\mathbf r\rVert^2-\tfrac13\lVert\mathbf r\rVert^2=\tfrac23\lVert\mathbf r\rVert^2$
(uniform second moment $\mathbb E_{\mathbf n}[\,n_in_j\,]=\tfrac13\delta_{ij}$). Here $1/m$ and $1/I$ are the
arm's **articulated effective** inverse mass and inverse rotational inertia — the trace-averaged blocks above
(`body_invweight0`, i.e. the code's `inv_mass`, `inv_rot`) — and $M_a^{\text{eff}}$ is the reciprocal of the
last line. The $\tfrac23$ is the sphere-average of $\sin^2\theta$ between $\mathbf r$ and $\mathbf n$; the
averaging is *why* the formula uses only $\lVert\mathbf r\rVert^2$, not the actual contact normal. (For the
gripper proxy $\mathbf r=0$, so the term drops; §8.0.)

So $\mathbf A$ is one object at two granularities: the **interface sum** is §8.1's $\mathbf D$; the
**arm-side per-body** value (via `body_invweight0`) is §8.0's $M_a^{\text{eff}}$. (Standard names for
$\mathbf A$: Delassus operator; operational-space inverse inertia.)

---


## 14. Appendix — when is $\boldsymbol\lambda$ a Lagrange multiplier? (and what Eq. (1) becomes)

This expands the remark in §8.1 that "bilateral $g$ makes $\boldsymbol\lambda$ a Lagrange multiplier." Short
answer: $\boldsymbol\lambda$ is a **classical Lagrange multiplier** exactly when the interface closure $g$ is
a smooth **bilateral equality that constrains $\mathbf u$ alone** — i.e. $g(\mathbf u,\boldsymbol\lambda)=\mathbf u-\bar{\mathbf u}$,
always active, with no $\boldsymbol\lambda$-dependence and no inequality/complementarity. Then
$\boldsymbol\lambda$ is the multiplier that enforces that equality in a constrained energy minimization.

### 14.1 The variational picture

In the bilateral case the monolithic solve is the **KKT stationarity of one equality-constrained QP**:

$$
\min_{\mathbf v_a,\mathbf v_b}\ \tfrac12\mathbf v_a^\top\mathbf M_a\mathbf v_a-\mathbf v_a^\top\mathbf f_a+\tfrac12\mathbf v_b^\top\mathbf M_b\mathbf v_b-\mathbf v_b^\top\mathbf f_b
\qquad\text{s.t.}\qquad \mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b=\bar{\mathbf u}.
$$

Introduce the Lagrangian, subtracting the constraint weighted by $\boldsymbol\lambda$:

$$
\mathcal L=\tfrac12\mathbf v_a^\top\mathbf M_a\mathbf v_a-\mathbf v_a^\top\mathbf f_a+\tfrac12\mathbf v_b^\top\mathbf M_b\mathbf v_b-\mathbf v_b^\top\mathbf f_b-\boldsymbol\lambda^\top\big(\mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b-\bar{\mathbf u}\big).
$$

Its stationarity conditions are **exactly the three rows of the monolithic system**:

$$
\begin{aligned}
\partial_{\mathbf v_a}\mathcal L=\mathbf 0 &\;\Rightarrow\; \mathbf M_a\mathbf v_a=\mathbf f_a+\mathbf J_a^\top\boldsymbol\lambda,\\
\partial_{\mathbf v_b}\mathcal L=\mathbf 0 &\;\Rightarrow\; \mathbf M_b\mathbf v_b=\mathbf f_b+\mathbf J_b^\top\boldsymbol\lambda,\\
\partial_{\boldsymbol\lambda}\mathcal L=\mathbf 0 &\;\Rightarrow\; \mathbf J_a\mathbf v_a+\mathbf J_b\mathbf v_b=\bar{\mathbf u}.
\end{aligned}
$$

So "$\boldsymbol\lambda$ is a Lagrange multiplier" is precisely the statement that the closure row is
$\partial\mathcal L/\partial\boldsymbol\lambda=\mathbf 0$ — which only parses if that row is a plain equality
independent of $\boldsymbol\lambda$.

### 14.2 What Eq. (1) becomes

The abstract three-row system (with the opaque closure $g(\mathbf u,\boldsymbol\lambda)=\mathbf 0$) collapses
into a **single linear symmetric saddle system** — the matrix form written as Eq. (1) in §8.1:

$$
\begin{bmatrix}\mathbf M_a & \mathbf 0 & -\mathbf J_a^\top\\ \mathbf 0 & \mathbf M_b & -\mathbf J_b^\top\\ -\mathbf J_a & -\mathbf J_b & \mathbf 0\end{bmatrix}
\begin{bmatrix}\mathbf v_a\\ \mathbf v_b\\ \boldsymbol\lambda\end{bmatrix}
=\begin{bmatrix}\mathbf f_a\\ \mathbf f_b\\ -\bar{\mathbf u}\end{bmatrix}.
$$

The nonlinear closure has become the bottom linear block. Eliminating $\mathbf v_a,\mathbf v_b$ through the
Delassus form (Eq. (2), $\mathbf u=\mathbf D\boldsymbol\lambda+\mathbf u^0$) reduces it to a **single linear
solve**:

$$
\boldsymbol\lambda=\mathbf D^{-1}(\bar{\mathbf u}-\mathbf u^0),\qquad
\mathbf D=\mathbf J_a\mathbf M_a^{-1}\mathbf J_a^\top+\mathbf J_b\mathbf M_b^{-1}\mathbf J_b^\top .
$$

That is the whole payoff of the bilateral case: no complementarity, no cone, no root-find — one linear solve
for $\boldsymbol\lambda$ (and then $\mathbf u=\bar{\mathbf u}$).

### 14.3 Where it stops being a clean multiplier

The other closures of §8.0 sit on a spectrum:

| closure | is $\boldsymbol\lambda$ a multiplier? | what the system becomes |
|---|---|---|
| **bilateral** $\mathbf u=\bar{\mathbf u}$ | **yes** — free equality multiplier | linear symmetric KKT; one linear solve |
| **frictionless Signorini** $0\le(u_n-\bar u_n)\perp\lambda_n\ge0$ | yes, but of an *inequality* (KKT of a convex QP; $\lambda_n\ge0$ $+$ complementarity) | a **linear complementarity problem** (LCP), not a single solve |
| **compliant / penalty** $\mathbf u-\bar{\mathbf u}+\mathbf C\boldsymbol\lambda=\mathbf 0$ | **no** — $\boldsymbol\lambda=\mathbf C^{-1}(\bar{\mathbf u}-\mathbf u)$ is a *determined* force, not a free multiplier | a plain (regularized) system with $\boldsymbol\lambda$ substituted out — *what VBD actually does* |
| **Coulomb friction** (cone $+$ max-dissipation) | not from a single potential — *non-associated* | a cone / variational-inequality problem; no symmetric-KKT structure |

The unifying view: $\boldsymbol\lambda$ is a Lagrange/KKT multiplier exactly when the closure is the
**optimality condition of a convex constrained minimization** — cleanest (a free equality multiplier, linear
KKT) for the bilateral weld; still a sign-constrained, complementary multiplier for frictionless contact; and
it degenerates to "just a force law" for compliance and to a non-associated variational inequality for
friction.

Finally, recall §8.3–§8.4: the proxy scheme's correctness **never requires $\boldsymbol\lambda$ to be a
multiplier**. The fixed-point argument is indifferent to which $g$ is used; the Lagrange-multiplier reading is
only the cleanest special case in which to reason, which is why §8.1 develops Eq. (1) there.
