# `SolverCoupledProxy` — how the proxy coupling works

How Newton couples a **MuJoCo arm** with a **VBD cloth** in the `--solver proxy` experiment
(`python -m newton.exp --solver proxy`), mirroring IsaacLab `Isaac-Pick-Proxy-Cloth-Direct-v0`.
Everything below is from source:

- exp wiring: `newton/exp/solvers/proxy.py`, `newton/exp/runner.py`
- coupler: `newton/_src/solvers/coupled/solver_coupled_proxy.py`,
  `.../solver_coupled.py`, `.../interface.py`, `.../proxy_utils.py`
- sub-solvers: `newton/_src/solvers/{mujoco,vbd}/...`

---

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
The $-m\mathbf g$ cancels the gravity VBD would apply to the proxy (so it tracks the arm instead of
falling); the $-\mathbf F^{\,n-1}$ removes the already-applied lagged force. The net effect: during the
cloth solve the proxy's velocity changes **only** due to cloth contact — which is what makes the
harvest clean. (For *staggered*, $\mathbf F$ is zeroed first, so this reduces to $-[m\mathbf g;\mathbf 0]$.)

### Step 4 — Detect contacts
Run the proxy `CollisionPipeline` (refreshed every `collide_interval`) to find cloth↔proxy contacts.

### Step 5 — Step the cloth (`SolverVBD`)
Solve the cloth with the proxies present as collision bodies of mass $m=\texttt{mass\_scale}\cdot m_\text{real}$:
$$\big(\mathbf q^{\text{cloth}}_{n+1},\ \dot{\mathbf q}^{\text{proxy}}_{\text{after}}\big)=\mathrm{VBD}\big(\text{cloth},\ \text{proxies},\ \text{contacts},\ \Delta t\big).$$

### Step 6 — Harvest the feedback wrench (`coupling_harvest_proxy_wrenches`)
The wrench the cloth applied to the proxy becomes the **new** $\mathbf F^{\,n}$:

- **Generic (momentum)** — `harvest_proxy_momentum_forces_kernel`, from the proxy's velocity change:
  $$\mathbf f=\frac{m\,(\mathbf v_\text{after}-\mathbf v_\text{before})}{\Delta t},\qquad
  \boldsymbol\tau=\frac{\mathbf R\,\mathbf I\,\mathbf R^{\top}(\boldsymbol\omega_\text{after}-\boldsymbol\omega_\text{before})}{\Delta t},\qquad
  \mathbf F^{\,n}[\text{proxy}]=[\mathbf f;\,\boldsymbol\tau].$$
  (Because Step 3 removed gravity + lagged force, the velocity change is the contact impulse alone.)
- **VBD (explicit contact)** — `_harvest_vbd_*_kernel`: sum the actual contact forces on the proxy,
  with torque about its center of mass $\mathbf c$:
  $$\mathbf F^{\,n}[\text{proxy}]=\sum_{\text{contacts }c}\Big[\,\mathbf f_c\ ;\ (\mathbf p_c-\mathbf c_\text{world})\times\mathbf f_c\,\Big],$$
  over cloth-particle↔proxy and rigid↔proxy contacts. VBD prefers this so it can keep proxy-vs-proxy
  collisions active inside the solve while feeding back *only* genuine contact forces.

### Step 7 — Blend (`blend_proxy_body_forces_kernel`)
Relax the new harvested wrench against the stashed previous one:
$$\mathbf F^{\,n}\leftarrow \alpha\,\mathbf F^{\,n}_\text{harvested}+(1-\alpha)\,\mathbf F^{\text{prev}}.$$
$\alpha=1$ keeps the harvest as-is; $\alpha<1$ under-relaxes (smooths the lagged-feedback loop); $\alpha>1$ over-relaxes.

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

## 7. Relation to the Genesis IPC `two_way_soft_constraint` coupling

Same *family* — partitioned co-simulation with the gripper as a proxy in the cloth solver and lagged
force feedback — but different primitives:

| | Genesis IPC `two_way` | Newton `SolverCoupledProxy` |
|---|---|---|
| Solvers | Genesis-rigid + libuipc-IPC | MuJoCo (arm) + VBD (cloth) |
| Coupling primitive | **soft spring** (`SoftTransformConstraint`, stiffness η) to an inertialess proxy | **direct pose/vel sync** of a mass-scaled proxy **+ harvested contact wrench** |
| Proxy inertia | inertialess (`K=0`) | **has mass** (`mass_scale`) → cloth feels a tunable inertia |
| Force fed back | spring gradient (approximate reconstruction) | actual contact wrench (momentum-change or explicit contact) |
| Coupling iterations | single pass | **iterable** (`proxy_iterations` + relaxation) |
| Monolithic alternative | `external_articulation` (joints inside IPC) | `--solver avbd` (one AVBD solve) |

So the proxy coupling directly mitigates two IPC-coupling drawbacks: the **massless-proxy** issue
(here the proxy carries `mass_scale` inertia, so the cloth feels something closer to the arm's
effective mass) and the **single-pass** issue (here you can relax with `proxy_iterations`). It still
shares the **lagged-feedback** limitation — `lagged` is explicitly one step behind; `staggered`
tightens it.

---

## 8. How to run

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

## 9. Notes / limitations

- **Lagged feedback** (one-substep delay in `lagged` mode) → reduced accuracy/stability under stiff
  contact; `staggered` and `proxy_iterations` tighten it at a cost.
- **Two solvers + per-substep state marshaling** (sync poses, rewind, collide, harvest) every step —
  not free; MuJoCo + VBD both run.
- **`mass_scale` is a tuning knob, not the true arm inertia** — it sets the proxy's effective mass to
  the cloth; the real articulated inertia still lives in MuJoCo.
- **Proxy ↔ arm can diverge** (the proxy is synced from the arm but solved against the cloth), as in
  any partitioned scheme.
- The coupler lives under `newton.solvers.experimental.coupled` — an experimental API.
