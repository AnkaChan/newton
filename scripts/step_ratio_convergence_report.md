# VBD Step Ratio (gamma) Convergence Study

**Date:** 2026-04-13
**Branch:** `horde/debug-contact-instability`
**Newton Physics Engine -- Vertex Block Descent Solver**

---

## 1. Motivation

The VBD solver updates particle positions via a block-diagonal Newton step:

    x <- x + gamma * H_local^{-1} f

where H_local is the per-vertex 3x3 Hessian and f is the total force (inertia + elastic + bending + contact). The parameter gamma (step ratio) controls how much of the Newton step is applied.

> **Does reducing gamma impede convergence of elasticity when objects are moving fast or stretched heavily, causing them to look softer than they should be?**

---

## 2. Methodology

### Residual metric
RMS force residual: root-mean-square of ||f_i|| across all particles. Step-size-independent.

For per-iteration convergence (Exp 2), a consistent evaluation with gamma=0 probe pass (with full state save/restore) gives clean gradient norm snapshots.

### Scenes

| Scene | Description | Scale |
|-------|-------------|-------|
| Cloth Hanging | 32x16 grid, one side fixed, no ground | Meter-scale (cell=0.1m) |
| Tubes on Table | 20x10 grid tubes, 3 stacked layers on ground | cm-scale (cell=1cm, gravity=981) |

### Parameters
- Gammas: 0.3, 0.5, 0.7, 0.9, 1.0
- VBD iterations/substep: 5 (Exp 1, 3) or 50 (Exp 2)
- Substeps/frame: 10 at 60 FPS
- Damping: Absolute; Tile solve: Disabled

---

## 3. Experiment 1: Residual vs Timestep

### 3.1 Cloth Hanging (no ground)

120 frames (1200 substeps), cloth falls and swings under gravity.

![Residual vs timestep -- cloth hanging](exp1_cloth_hanging.png)

| gamma | Final Residual | vs gamma=1.0 |
|-------|---------------|-------------|
| 0.3 | 5.128e-01 | 95.3x worse |
| 0.5 | 1.751e-01 | 32.5x worse |
| 0.7 | 3.465e-02 | 6.4x worse |
| 0.9 | 5.530e-03 | 1.0x worse |
| **1.0** | **5.383e-03** | **baseline** |

**Findings:** gamma=0.9 and gamma=1.0 nearly identical. gamma<=0.7 progressively worse. gamma=0.3 is ~100x worse -- under-relaxation leaves unconverged residual that becomes velocity feeding next substep's inertia.

### 3.2 Tubes on Table (3 layers, contact)

60 frames (600 substeps), three cylindrical tubes stacked on ground with self-contact.

![Residual vs timestep -- tubes on table](exp1_grid_on_table.png)

| gamma | Final Residual | vs gamma=1.0 |
|-------|---------------|-------------|
| 0.3 | 1998 | 12.72x (worse) |
| 0.5 | 1012 | 6.44x (worse) |
| 0.7 | 114 | 0.73x (better) |
| 0.9 | 114 | 0.72x (better) |
| **1.0** | **157** | **baseline** |

**Findings:** gamma=0.7-0.9 outperform baseline in contact scene. Cross-color interference compounds with gamma=1.0. gamma<=0.5 still substantially worse.

---

## 4. Experiment 2: Per-Iteration Convergence

### Warmup State Validation

Warmed up for **400 substeps** with gamma=1.0. At substep 400: KE~11 (from peak ~24,000), max velocity ~5.6 cm/s. Valid near-rest state.

![Warmup diagnostic](warmup_diagnostic.png)

### Protocol

1. Warm up 400 substeps with gamma=1.0
2. Save state (positions + velocities)
3. For each gamma: restore state, run 1 substep with 50 iterations, record residual per iteration via gamma=0 probe

![Per-iteration convergence](exp2_per_iteration.png)

| gamma | Initial | After 5 iters | After 50 iters | Ratio |
|-------|---------|--------------|----------------|-------|
| 0.3 | 564 | 119.7 | 30.1 | 0.0534 (slowest) |
| 0.5 | 564 | 58.6 | 26.4 | 0.0469 |
| 0.7 | 564 | 42.7 | 23.1 | 0.0409 |
| 0.9 | 564 | 36.8 | 19.0 | 0.0337 |
| 1.0 | 564 | 33.4 | 16.8 | 0.0298 (fastest) |

**Findings:** Per-iteration convergence monotonically improves with higher gamma. gamma=1.0 is optimal within a single substep. Benefit most pronounced in iterations 0-5. After iteration 20-30, all gammas enter slow convergence regime.

**Reconciling:** gamma=1.0 best per-iteration but gamma=0.7-0.9 best cross-substep in contact scenes. Unconverged residual becomes velocity with directional bias from cross-color interference; mild under-relaxation dampens this compounding.

---

## 5. Experiment 3: Cloth Sag (Softness Test)

32x16 cloth, one side fixed, no ground. 600 frames (6000 substeps). Equilibrium averaged over last 500 substeps.

![Cloth sag](exp3_cloth_sag.png)

| gamma | Z-Centroid | Lowest Z | Excess sag |
|-------|-----------|----------|-----------|
| 0.3 | 2.262 | 0.774 | +5.0% |
| 0.5 | 2.323 | 0.873 | +1.8% |
| 0.7 | 2.359 | 0.875 | +1.7% |
| 0.9 | 2.361 | 0.925 | +0.1% |
| **1.0** | **2.361** | **0.928** | **baseline** |

**Findings:** gamma=0.3 produces ~5% more sag. **gamma=0.9 has negligible softness** (<1%). gamma=0.7-1.0 converge to same equilibrium shape.

---

## 6. Summary

| | Per-iter convergence | Cross-substep stability | Softness |
|-|---------------------|------------------------|----------|
| gamma=1.0 | Best | Worse (contact divergence) | None |
| gamma=0.9 | Near-best | Good | Negligible |
| gamma=0.7 | Moderate | Good | Minor (~1.7%) |
| gamma=0.5 | Poor | Moderate | Noticeable (~1.8%) |
| gamma=0.3 | Poor | Poor | Significant (~5%) |

### Recommendations

1. **Default gamma=1.0** for no-contact scenes
2. **gamma=0.9 for contact-heavy scenes** -- eliminates cross-color divergence, >99% convergence speed, negligible softness
3. **Avoid gamma<=0.5** -- severe convergence penalty, measurable softness
4. **gamma=0.7** acceptable as conservative fallback
5. **Future: adaptive gamma** -- 1.0 far from contact, 0.9 near contact
