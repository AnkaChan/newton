# Is the learned local-global step a zero-stable integrator?

Fixed points, fusion and truncation for the LIDO inner loop.

Date: 2026-09-28. Status: derivation note (mathematics only, no code).
Related notes: `notes/epoch-regime-20260927.md` (the fixed-state epoch regime the campaign is trained under), `notes/ideas/idea-normalize-cells.md` (cell normalisation and the dimensionless groups referred to in Section 7).

## 0. Summary

The learned solver replaces the Newton or gradient iteration inside a variational implicit-Euler step by a network that proposes, for every hex cell, a target for the cell's local deformation axes. The proposals are fused into a single displacement of the shared corners by a weighted least-squares fit. The question of this note is whether that replacement changes the *integrator*, that is, whether the time-stepping scheme obtained when the inner loop converges is still backward Euler, and therefore zero-stable, and what a truncated inner loop does to the trajectory.

The answer, in order of the sections below:

- The fusion introduces no spurious stationary points: the projected gradient the network receives vanishes if and only if $\nabla\Phi_n$ vanishes (Proposition 5.1). Given Assumption A2 on the network (its fused step vanishes if and only if that input does), the fixed points of the inner loop coincide with the stationary points of the implicit-Euler objective $\Phi_n$. The fusion changes the path to the fixed point, not the fixed point. This holds for a connected body with at least one pinned corner, because the fusion operator is then a projection with trivial null space (Corollary 4.2).
- The time-stepping map at convergence is therefore backward Euler on the first-order system: a one-step method with characteristic polynomial $\rho(\zeta)=\zeta-1$, hence zero-stable, first-order consistent and convergent (Section 6.1); unconditionally stable and dissipative on the linear test problem with per-mode amplification $|\zeta| = (1+\omega^2\Delta t^2)^{-1/2}$ (Proposition 6.1).
- Truncating the inner loop at $K_{\mathrm{it}}$ iterations leaves a gradient residual $r_n$ whose effect on the position is at most $\Delta t^2\,\|r_n\|/m_{\min}$ per step and accumulates linearly, giving a drift of order $T\,\Delta t\,\max_n\|r_n\|/m_{\min}$ over a horizon $T$ (Section 6.3). This is what the free-corner force-residual metric measures and what the long-horizon drift in the rollouts is.
- For an unpinned body the fusion has a three-dimensional translation null space (Section 7). Blending the fit toward the inertial prediction $y_n$ pins the mass-weighted centroid to free fall regardless of contact, so the body sinks by $\Delta t^2 F_{\mathrm{con}}/M_{\mathrm{tot}}$ per step relative to backward Euler and no deformation target can correct it (Section 7.3). A rigid target recomputed from the current iterate's contact force restores the translation component of $\nabla\Phi_n=0$ and is a Picard iteration with contraction constant $\Delta t^2\sum k_e/M_{\mathrm{tot}}$, about $0.44$ at $E=10^5\,$Pa and about $4.4$ at $E=10^6\,$Pa for the campaign numbers; a $3\times 3$ Newton step on the centroid removes the condition (Sections 7.4 to 7.6).
- Extending the per-cell output from 9 values (axes) to 21 values (axes plus four warping vectors) leaves the fusion matrix unchanged: it is literally the same $K=B^{\top}WB$, so the cached factor is reused and only the right-hand side of the normal equations changes (Section 8).

What is proven here and what is not is collected in Section 9.

## 1. Setup and notation

**Mesh and degrees of freedom.** The body is a hexahedral mesh with $C$ cells and $P$ corners. The corner positions are stacked in $x\in\mathbb{R}^{3P}$. The corner index set is split into free corners $f$ and pinned corners $p$, with $P_{\mathrm{free}}+P_{\mathrm{pin}}=P$; we write $x=(x_f,x_p)$ and treat $x_p$ as prescribed (in the campaign the pinned corners do not move, $x_p(t)\equiv x_p$). Gradients with respect to $x$ are understood as gradients with respect to $x_f$ unless the pinned block is explicitly named.

**Mass, time step, external force.** The lumped mass matrix is $M=\operatorname{diag}(m_i I_3)_{i=1}^{P}$ with $m_i>0$; $m_{\min}=\min_i m_i$ and $M_{\mathrm{tot}}=\sum_i m_i$. The time step is $\Delta t$. Gravity is $g\in\mathbb{R}^3$; with $\mathbf{1}\in\mathbb{R}^P$ the all-ones vector and $\otimes$ the Kronecker product, the stacked gravity vector is $\mathbf{1}\otimes g\in\mathbb{R}^{3P}$ and the external force is

$$ f_{\mathrm{ext}} = M\,(\mathbf{1}\otimes g), \qquad M^{-1} f_{\mathrm{ext}} = \mathbf{1}\otimes g . \tag{1.1} $$

We also use the three translation vectors $\mathbf{1}_a=\mathbf{1}\otimes e_a\in\mathbb{R}^{3P}$, $a=1,2,3$, and the $3P\times 3$ matrix $Z=\mathbf{1}\otimes I_3=[\mathbf{1}_1\ \mathbf{1}_2\ \mathbf{1}_3]$, so that a rigid translation by $t\in\mathbb{R}^3$ is the displacement $Z t=\mathbf{1}\otimes t$. Index convention: $c$ always indexes cells, and $a,b,d\in\{1,2,3\}$ index coordinate directions (superscripts on $\xi$ and on vector components).

**Reference cell and shape functions.** Each cell $c$ has eight corners $x_{c,k}\in\mathbb{R}^3$, $k=1,\dots,8$, and is the image of a reference cube of side $h$. The corner parametric coordinates are $\xi_k\in\{-1,+1\}^3$, the reference map is $X(\xi)=\tfrac{h}{2}\xi$, and the trilinear shape functions are

$$ N_k(\xi)=\frac18\prod_{a=1}^{3}\bigl(1+\xi_k^a\,\xi^a\bigr), \qquad \sum_{k=1}^{8}N_k(\xi)=1 \ \text{ for all } \xi . \tag{1.2} $$

The second identity (partition of unity) follows from expanding the product: every term containing at least one factor $\xi_k^a\xi^a$ sums to zero over the eight sign patterns $\xi_k$. The gradient with respect to the reference coordinates is $\nabla N_k(\xi)=\tfrac{2}{h}\nabla_\xi N_k(\xi)\in\mathbb{R}^3$, and by (1.2)

$$ \sum_{k=1}^{8}\nabla N_k(\xi)=0 \qquad\text{for all } \xi . \tag{1.3} $$

**Deformation gradient.** The interpolated position in cell $c$ is $x_c(\xi)=\sum_k x_{c,k}N_k(\xi)$, and the deformation gradient at $\xi$ is

$$ F_c(x;\xi)=\sum_{k=1}^{8} x_{c,k}\,\nabla N_k(\xi)^{\!\top}\in\mathbb{R}^{3\times 3}, \tag{1.4} $$

which is linear in $x$. The eight Gauss points are $\xi_q\in\{-1/\sqrt3,+1/\sqrt3\}^3$, $q=1,\dots,8$, with weights $w_q=h^3/8$ (so $\sum_q w_q=h^3$, the cell volume). Writing $\operatorname{vec}F\in\mathbb{R}^9$ for the stacked entries of $F$, the Gauss-point deformation gradient is a linear map of the corner positions,

$$ \operatorname{vec}F_{c,q}(x)=G_{c,q}\,x, \qquad G_{c,q}\in\mathbb{R}^{9\times 3P}, \tag{1.5} $$

with $G_{c,q}$ nonzero only in the 24 columns belonging to the corners of cell $c$. We write $F_c(x)=F_c(x;0)$ for the deformation gradient at the cell centre.

**Cell frame and local axes.** The per-cell frame $R_c\in SO(3)$ is the rotation closest to the centre deformation in the Frobenius norm, $R_c=\arg\min_{R\in SO(3)}\|F_c-R\|_F$, i.e. the rotation factor of the polar decomposition $F_c=R_cS_c$ (with the sign convention that keeps $R_c$ proper when $\det F_c<0$). The frame is *detached*: it is evaluated at the current iterate and no derivative is taken through it. The local axes are

$$ A_c=R_c^{\top}F_c . \tag{1.6} $$

**Energy.** The incremental potential at step $n$ is

$$ E(x;x_n)=E_{\mathrm{el}}(x)+E_{\mathrm{visc}}(x;x_n)+E_{\mathrm{con}}(x;x_n), \tag{1.7} $$

where $E_{\mathrm{el}}(x)=\sum_c\sum_q w_q\,\psi\bigl(F_{c,q}(x)\bigr)$ with $\psi$ the stable Neo-Hookean density (eight-point quadrature); $E_{\mathrm{visc}}$ is the viscous damping term, a function of the rate $\bigl(F_{c,q}(x)-F_{c,q}(x_n)\bigr)/\Delta t$ and hence of $x_n$; and $E_{\mathrm{con}}$ is the contact penalty for the pair set (corner, obstacle) collected at $x_n$ and frozen for the step, using the Newton engine's penalty law (normal stiffness $k_e$, damping $k_d$, friction). For a pair $i$ with unit obstacle normal $n_i$ and signed gap $g_i(x)=n_i\cdot(x_i-o_i)$, $o_i$ a point on the obstacle, the normal part of the penalty is $\tfrac{k_e}{2}\max(0,-g_i(x))^2$; only this part enters the derivations of Section 7.

**Assumption A1.** For every $x_n$, $E(\,\cdot\,;x_n)$ is bounded below and continuously differentiable on $\mathbb{R}^{3P}$. Beyond A1, the derivations use the following properties of $E$ where indicated: Lipschitz continuity of $\nabla E$ on the region visited (Section 6.1(b)); existence and positive semidefiniteness of the Hessian $H_n=\nabla^2E$ at the point of linearisation (Section 6.3); and, in Section 7, the translation invariance (1.8) and the structure of the normal penalty, (1.9) and (7.13).

**Translation invariance.** The elastic and viscous energies depend on $x$ only through deformation gradients, which by (1.3) are unchanged by a common translation, so

$$ \begin{aligned} E_{\mathrm{el}}(x+Zt)&=E_{\mathrm{el}}(x),\qquad E_{\mathrm{visc}}(x+Zt;x_n)=E_{\mathrm{visc}}(x;x_n),\\ \text{hence}\qquad \mathbf{1}_a^{\top}\nabla E_{\mathrm{el}}&=\mathbf{1}_a^{\top}\nabla E_{\mathrm{visc}}=0 \qquad (a=1,2,3). \end{aligned} \tag{1.8} $$

The contact energy is not translation invariant (the obstacles are fixed in space). Its translation component is minus the total contact force on the body:

$$ \mathbf{1}_a^{\top}\nabla E_{\mathrm{con}}(x)=\sum_i \frac{\partial E_{\mathrm{con}}}{\partial x_i^{a}} = -F_{\mathrm{con}}^{a}(x), \qquad F_{\mathrm{con}}(x):=-\sum_{i}\nabla_{x_i}E_{\mathrm{con}}(x)\in\mathbb{R}^3 . \tag{1.9} $$

**Centroid.** The mass-weighted centroid is

$$ c(x)=\frac{1}{M_{\mathrm{tot}}}\sum_{i=1}^{P}m_i\,x_i=\frac{1}{M_{\mathrm{tot}}}Z^{\top}Mx, \qquad c(x+Zt)=c(x)+t, \tag{1.10} $$

using $Z^{\top}MZ=(\sum_i m_i)\,I_3=M_{\mathrm{tot}}I_3$. We write $c_n=c(x_n)$ and $\dot c_n=c(v_n)$ (the same linear map applied to the velocities).

## 2. Variational implicit Euler

Given $(x_n,v_n)$, define the inertial prediction and the incremental objective

$$ y_n=x_n+\Delta t\,v_n+\Delta t^2 M^{-1}f_{\mathrm{ext}} = x_n+\Delta t\,v_n+\Delta t^2\,(\mathbf{1}\otimes g), \tag{2.1} $$

$$ \Phi_n(x)=\frac{1}{2\Delta t^2}(x-y_n)^{\top}M(x-y_n)+E(x;x_n). \tag{2.2} $$

> **Proposition 2.1 (stationary points are backward-Euler steps).** Let $x_{n+1}$ be a stationary point of $\Phi_n$, $\nabla\Phi_n(x_{n+1})=0$, and set $v_{n+1}=(x_{n+1}-x_n)/\Delta t$. Then
> $$ x_{n+1}=x_n+\Delta t\,v_{n+1}, \qquad M\,\frac{v_{n+1}-v_n}{\Delta t}=-\nabla E(x_{n+1};x_n)+f_{\mathrm{ext}}, \tag{2.3} $$
> i.e. $(x_{n+1},v_{n+1})$ is the backward-Euler step of the first-order system $\dot x=v$, $M\dot v=-\nabla E(x)+f_{\mathrm{ext}}$, with the velocity-dependent (viscous) force evaluated at the new velocity.

*Proof.* The gradient of (2.2) is

$$ \nabla\Phi_n(x)=\frac{1}{\Delta t^2}M(x-y_n)+\nabla E(x;x_n). \tag{2.4} $$

Substituting (2.1) and the definition of $v_{n+1}$,

$$ x_{n+1}-y_n = (x_{n+1}-x_n)-\Delta t\,v_n-\Delta t^2M^{-1}f_{\mathrm{ext}} = \Delta t\,(v_{n+1}-v_n)-\Delta t^2M^{-1}f_{\mathrm{ext}}, $$

so that

$$ \frac{1}{\Delta t^2}M(x_{n+1}-y_n)=M\,\frac{v_{n+1}-v_n}{\Delta t}-f_{\mathrm{ext}} . $$

Setting (2.4) to zero at $x_{n+1}$ gives the second equation of (2.3); the first is the definition of $v_{n+1}$. The viscous term of $E$ depends on $(x_{n+1}-x_n)/\Delta t=v_{n+1}$, so the damping force is evaluated implicitly. $\blacksquare$

> **Proposition 2.2 (existence).** Under Assumption A1, $\Phi_n$ has a global minimiser, which is a stationary point.

*Proof.* With $E_{\inf}=\inf_x E(x;x_n)>-\infty$,

$$ \Phi_n(x)\ \ge\ \frac{m_{\min}}{2\Delta t^2}\,\|x-y_n\|^2+E_{\inf}\ \longrightarrow\ \infty \quad (\|x\|\to\infty), $$

so $\Phi_n$ is coercive. Its sublevel set $\{\Phi_n\le\Phi_n(y_n)\}$ is therefore bounded, closed (continuity), hence compact, and the continuous function $\Phi_n$ attains its minimum on it; this is a global minimiser. Since $\Phi_n$ is $C^1$, the gradient vanishes there. $\blacksquare$

**Remark 2.3 (nonconvexity).** $E_{\mathrm{el}}$ is not convex in $x$ (stable Neo-Hookean is polyconvex at best), and the frozen-normal friction and damping terms of the contact law are not convex in $x$ either; the normal contact penalty $\tfrac{k_e}{2}\max(0,-g_i(x))^2$, with $g_i$ affine in $x$, is convex in $x$ and is not a source of nonconvexity. So $\Phi_n$ may have several stationary points: local minima, saddles, and a global minimiser. Each of them satisfies (2.3) and is therefore a valid backward-Euler step. Which one an iterative method reaches depends on the method and the starting point. Nothing in this note claims uniqueness.

**Remark 2.4 (pinned corners).** With $x_p$ prescribed, $\Phi_n$ is minimised over $x_f$ only. All statements above hold with $\nabla$ read as $\nabla_{x_f}$ and $M$ as its free block. The reaction forces on the pinned corners are the pinned components of $\nabla\Phi_n$ and are not required to vanish.

## 3. The learned local-global step

We now fix the outer step $n$ and describe one *query*, i.e. one iteration of the inner loop that seeks a stationary point of $\Phi_n$. The inner iterate is $x_k$, $k=0,1,\dots$, with $x_0$ a starting candidate (in the campaign, $x_0=y_n$ or the previous iterate). To keep indices apart, $n$ always counts time steps and $k$ inner iterations.

**Frozen frames and current axes.** At the iterate $x_k$ the frames $R_c=R_c(x_k)$ are evaluated and held fixed for the whole query. The current local axes are

$$ \hat A_c=R_c^{\top}F_c(x_k). \tag{3.1} $$

**Network output and world increment.** The network returns for every cell target axes $\hat A_c^{\mathrm{target}}\in\mathbb{R}^{3\times3}$, expressed in the same frozen frame (and possibly a scalar per-cell step length in $(0,1]$ that scales the increment; it is absorbed into the definition below and plays no role in the analysis). The requested increment of the cell's centre deformation gradient, rotated back to world coordinates, is

$$ \Delta F_c=R_c\bigl(\hat A_c^{\mathrm{target}}-\hat A_c\bigr)=R_c\hat A_c^{\mathrm{target}}-F_c(x_k). \tag{3.2} $$

**Fusion.** The per-cell increments are fused into one corner displacement $d\in\mathbb{R}^{3P}$ by the weighted least-squares problem

$$ d=\arg\min_{d}\ \frac12\sum_{c=1}^{C}\sum_{q=1}^{8} w_q\,\bigl\|G_{c,q}\,d-\operatorname{vec}\Delta F_c\bigr\|^2 \quad\text{subject to } d_p=d_p^{\mathrm{presc}}, \tag{3.3} $$

where $d_p^{\mathrm{presc}}$ is the prescribed displacement of the pinned corners over the step (zero in the campaign). In matrix form, stack the $8C$ blocks $G_{c,q}$ into $B\in\mathbb{R}^{72C\times 3P}$ (nine rows per Gauss point), let $W=\operatorname{diag}(w_q I_9)$ be the $72C\times72C$ weight matrix, and let $\Delta F\in\mathbb{R}^{72C}$ be the stacked targets, with $\operatorname{vec}\Delta F_c$ repeated in the eight Gauss-point slots of cell $c$. Then (3.3) reads

$$ d=\arg\min_{d_p=d_p^{\mathrm{presc}}}\ J(d), \qquad J(d)=\frac12\bigl\|W^{1/2}(Bd-\Delta F)\bigr\|^2 . \tag{3.4} $$

Split the columns of $B$ into free and pinned, $B=[B_f\ B_p]$, and define

$$ K=B^{\top}WB=\begin{pmatrix}K_{ff}&K_{fp}\\K_{pf}&K_{pp}\end{pmatrix}, \qquad K_{ff}=B_f^{\top}WB_f,\quad K_{fp}=B_f^{\top}WB_p . \tag{3.5} $$

Setting $\nabla_{d_f}J=B_f^{\top}W(B_fd_f+B_pd_p-\Delta F)=0$ gives the normal equations

$$ K_{ff}\,d_f=B_f^{\top}W\,\Delta F-K_{fp}\,d_p, \qquad x_{k+1}=x_k+d . \tag{3.6} $$

$K$ is symmetric positive semidefinite by construction; Corollary 4.2 shows when $K_{ff}$ is definite.

**Remark 3.1 (one increment per cell asks for an affine change).** The same $\Delta F_c$ is imposed at all eight Gauss points of cell $c$. An affine displacement of the cell's corners, $d_{c,k}=L\,X(\xi_k)+t$ with $L\in\mathbb{R}^{3\times3}$, produces by (1.4) the increment $\sum_k(LX(\xi_k)+t)\nabla N_k^{\top}=L\sum_kX(\xi_k)\nabla N_k(\xi)^{\top}=L$ at every $\xi$ (the identity $\sum_kX(\xi_k)\nabla N_k(\xi)^{\top}=I$ is the exactness of the trilinear interpolation for linear fields, and the $t$ term vanishes by (1.3)). Hence the per-cell fit is exact for the affine displacement $L=\Delta F_c$ and penalises any non-affine (warping) part of the increment, which would make $G_{c,q}d$ vary across Gauss points. Warping changes of a cell arise in the fused solution only through the consistency of shared corners with neighbouring cells. Section 8 removes this restriction.

**Remark 3.2 (linearity within a query).** Because the frames $R_c$ are frozen during the query, the map $\Delta F\mapsto d$ defined by (3.6) is linear and does not depend on the network. The nonlinearity of the inner loop is entirely in the network's map $x_k\mapsto\hat A^{\mathrm{target}}$ and in the re-evaluation of the frames between queries.

**Remark 3.3 (interpretation).** The scheme is a local-global iteration in the sense of projective dynamics: the *local* step produces per-cell target deformations; the *global* step solves one sparse SPD system with a fixed matrix to make them consistent at shared corners. In projective dynamics the local step is a projection onto a constraint manifold (for instance the closest rotation); here it is the network. The global matrix in projective dynamics is $M/\Delta t^2+\sum_c w_c\mathsf{S}_c^{\top}\mathsf{A}_c^{\top}\mathsf{A}_c\mathsf{S}_c$, with $\mathsf{S}_c$ the selection matrix of the element's degrees of freedom and $\mathsf{A}_c$ the constraint's differential operator (sans-serif to avoid a clash with the local axes $A_c$ and the stretch $S_c$ of Section 1); the fusion matrix $K$ is the second term alone with $\mathsf{A}_c\mathsf{S}_c$ the Gauss-point deformation-gradient operators. The inertia term is absent because the network is meant to account for it through the gradient input (Section 5).

## 4. Null space and reachability

> **Lemma 4.1 (null space of $B$).** For a mesh that is connected through shared corners (any two cells are joined by a chain of cells in which consecutive cells share at least one corner), $\operatorname{null}(B)=\{Zt:\ t\in\mathbb{R}^3\}$, the rigid translations. Rotations are not in the null space.

*Proof.* (i) *Translations are in the null space.* For $d=Zt$ every corner of every cell is displaced by the same $t$, and by (1.3)

$$ \sum_k t\,\nabla N_k(\xi_q)^{\top}=t\Bigl(\sum_k\nabla N_k(\xi_q)\Bigr)^{\!\top}=0 . $$

(ii) *A single cell has no other null vectors.* Expand the product in (1.2). For corner displacements $d_k$ the interpolated displacement is

$$ \sum_k d_kN_k(\xi)=t+\sum_{a}\alpha^{a}\xi^{a}+\sum_{a<b}\omega^{ab}\xi^{a}\xi^{b}+\omega^{123}\xi^1\xi^2\xi^3, \tag{4.1} $$

with the 24 coefficients (each a vector in $\mathbb{R}^3$)

$$ \begin{aligned} t&=\frac18\sum_kd_k, &\qquad \alpha^{a}&=\frac18\sum_k\xi_k^{a}\,d_k,\\ \omega^{ab}&=\frac18\sum_k\xi_k^{a}\xi_k^{b}\,d_k, &\qquad \omega^{123}&=\frac18\sum_k\xi_k^1\xi_k^2\xi_k^3\,d_k . \end{aligned} \tag{4.2} $$

(This is the mode decomposition of Section 8; the eight functions $1,\xi^a,\xi^a\xi^b,\xi^1\xi^2\xi^3$ on the eight corners form an orthogonal basis with squared norm 8, so (4.2) is the inverse of (4.1).) The reference-coordinate gradient of the displacement is $\tfrac2h\,\partial_\xi$ of (4.1). Its first column, the derivative with respect to $\xi^1$, is

$$ \frac{\partial}{\partial\xi^1}\sum_kd_kN_k=\alpha^{1}+\omega^{12}\xi^2+\omega^{13}\xi^3+\omega^{123}\xi^2\xi^3, \tag{4.3} $$

a bilinear function of $(\xi^2,\xi^3)$ that does not depend on $\xi^1$. If $G_{c,q}d_c=0$ for all eight Gauss points, then (4.3) vanishes at the four sign patterns $(\xi^2,\xi^3)\in\{\pm s\}^2$, $s=1/\sqrt3$. The $4\times4$ matrix with rows $(1,\ \xi^2,\ \xi^3,\ \xi^2\xi^3)$ at these four points is a Hadamard matrix times $\operatorname{diag}(1,s,s,s^2)$ (column scaling) and is invertible, so $\alpha^1=\omega^{12}=\omega^{13}=\omega^{123}=0$. The same argument on the other two columns gives $\alpha^2=\alpha^3=\omega^{23}=0$. Hence (4.1) reduces to the constant $t$: $d_k=t$ for all $k$, a translation of the cell.

(iii) *Assembly.* If $Bd=0$ then every cell's 24 corner displacements are a translation $t_c$ of that cell by (ii). Two cells sharing a corner have $t_c=t_{c'}$; connectedness propagates one common $t$ to the whole mesh.

(iv) *Rotations.* An infinitesimal rotation $d_k=\omega\times X(\xi_k)$ is affine with $L=[\omega]_\times\neq0$, and by Remark 3.1 it produces $G_{c,q}d=\operatorname{vec}[\omega]_\times\neq0$. A finite rotation $x_k=RX(\xi_k)$ of the reference cell gives $F=R\neq I$: the deformation gradient does change. $\blacksquare$

> **Corollary 4.2 (pinned body).** Let the mesh be connected with at least one pinned corner and take $d_p=0$. Then $\operatorname{null}(B_f)=\{0\}$, $K_{ff}=B_f^{\top}WB_f$ is symmetric positive definite, $B_f^{\top}W$ maps $\mathbb{R}^{72C}$ onto $\mathbb{R}^{3P_{\mathrm{free}}}$, and for every free displacement $d_f$ the target $\Delta F=B_fd_f$ is reproduced exactly by the fusion:
> $$ K_{ff}^{-1}B_f^{\top}W\,(B_fd_f)=d_f . \tag{4.4} $$

*Proof.* If $B_fd_f=0$, extend $d_f$ by zeros on the pinned corners to $d=(d_f,0)$; then $Bd=B_fd_f=0$, so $d=Zt$ by Lemma 4.1, and $d_p=0$ on at least one corner forces $t=0$, hence $d_f=0$. Thus $B_f$ has full column rank, $d_f^{\top}K_{ff}d_f=\|W^{1/2}B_fd_f\|^2>0$ for $d_f\neq0$ ($W\succ0$), and $B_f^{\top}$, having full row rank, is onto; composing with the invertible $W$ keeps it onto. (4.4) is $K_{ff}^{-1}K_{ff}d_f=d_f$. $\blacksquare$

**Dimension count.** The fusion is the linear map $\Pi=K_{ff}^{-1}B_f^{\top}W:\mathbb{R}^{72C}\to\mathbb{R}^{3P_{\mathrm{free}}}$ with $\Pi B_f=I$, so $B_f\Pi$ is the $W$-orthogonal projector onto $\operatorname{range}(B_f)$: the fusion projects the requested Gauss-point increments onto those realisable by a corner displacement and loses no free-corner mode. The requests are heavily over-determined. For the campaign beam, $10\times10\times40$ cells of side $h=0.025\,$m,

$$ \begin{aligned} &C=4000,\qquad 9C=36\,000 \text{ target values},\qquad P=11\cdot11\cdot41=4961,\\ &P_{\mathrm{pin}}=121,\qquad 3P_{\mathrm{free}}=3\,(4961-121)=14\,520 . \end{aligned} $$

(The residual in (3.4) has $72C=288\,000$ components, but the target carries only $9C$ independent values because each $\Delta F_c$ is repeated at the eight Gauss points.)

## 5. Stationarity with the fusion

Fix the query at iterate $x_k$, assume the setting of Corollary 4.2 and $d_p=0$, and write $\Pi=K_{ff}^{-1}B_f^{\top}W$. The composite objective seen through the fusion is

$$ \Psi(\Delta F)=\Phi_n\bigl(x_k+\Pi\,\Delta F\bigr), \qquad \Delta F\in\mathbb{R}^{72C}. \tag{5.1} $$

By the chain rule and the symmetry of $K_{ff}$ and $W$,

$$ \nabla\Psi(\Delta F)=\Pi^{\top}\,\nabla\Phi_n(x)\big|_{x=x_k+\Pi\Delta F}=W\,B_f\,K_{ff}^{-1}\,\nabla\Phi_n(x). \tag{5.2} $$

This is the *projected gradient*: for each cell and Gauss point it is the $3\times3$ block of $W B_fK_{ff}^{-1}\nabla\Phi_n$, i.e. the deformation-gradient increment that the fused corner displacement $K_{ff}^{-1}\nabla\Phi_n$ would produce, weighted by $w_q$. It is what the network receives as its gradient input.

> **Proposition 5.1 (the fusion introduces no spurious stationary points).** Under Corollary 4.2, $\nabla\Psi(\Delta F)=0$ if and only if $\nabla\Phi_n(x_k+\Pi\Delta F)=0$.

*Proof.* Write $\gamma=\nabla\Phi_n(x)$. If $\gamma=0$ then $\nabla\Psi=0$ by (5.2). Conversely, if $WB_fK_{ff}^{-1}\gamma=0$ then, $W$ being invertible, $B_f(K_{ff}^{-1}\gamma)=0$, so $K_{ff}^{-1}\gamma\in\operatorname{null}(B_f)=\{0\}$, and since $K_{ff}$ is invertible, $\gamma=0$. $\blacksquare$

**Consequence for the fixed points of the inner loop.** A fixed point of the inner iteration $x_{k+1}=x_k+\Pi\,\Delta F(x_k)$ is a point $x^\ast$ with $\Pi\,\Delta F(x^\ast)=0$. Two facts follow from Proposition 5.1 and Corollary 4.2 without any assumption on the network:

- If $\nabla\Phi_n(x^\ast)\neq0$, then $\nabla\Psi\neq0$ at $x^\ast$, so there exist targets $\Delta F$ for which $\Pi\Delta F$ is a descent direction of $\Phi_n$ (any $\Delta F$ with $\langle\nabla\Psi,\Delta F\rangle<0$). The fusion never hides a nonzero gradient from a network that reads $\nabla\Psi$.
- If $\nabla\Phi_n(x^\ast)=0$, then $\nabla\Psi=0$: the network's gradient input is identically zero, and a network whose output vanishes when its gradient input vanishes returns $\Delta F=0$, hence $d=0$.

**Assumption A2 (network consistency).** The fused step $\Pi\,\Delta F(x)$ vanishes if and only if $\nabla\Psi(x)=0$. Under A2, the fixed points of the inner loop are *exactly* the stationary points of $\Phi_n$: the fusion changes the path of the iteration, not its fixed points. A2 is a property of the trained network and is not proven here; what is proven is that the fusion itself neither adds nor removes stationary points, and that the information reaching the network vanishes exactly at the stationary points of $\Phi_n$.

**Remark 5.2 (gradient step through the fusion).** If the network returned the scaled negative projected image of the gradient in target space, $\Delta F=-\alpha\,B_fK_{ff}^{-1}\nabla\Phi_n(x_k)$, then by (4.4)

$$ d=\Pi\,\Delta F=-\alpha\,K_{ff}^{-1}B_f^{\top}WB_f\,K_{ff}^{-1}\nabla\Phi_n=-\alpha\,K_{ff}^{-1}\nabla\Phi_n(x_k), \tag{5.3} $$

a gradient step preconditioned by the fusion matrix. This is precisely the role of the global matrix in projective dynamics, where the local step is a rotation projection and the global solve is a preconditioned descent step on the same objective. The learned local step can therefore be read as a learned replacement for the projection, with the same global solve.

## 6. Zero-stability and the effect of truncation

### 6.1 The scheme at inner convergence is a one-step method

Suppose that at every step the inner loop converges to a stationary point $x_{n+1}$ of $\Phi_n$ (Section 5) and set $v_{n+1}=(x_{n+1}-x_n)/\Delta t$. The map $(x_n,v_n)\mapsto(x_{n+1},v_{n+1})$ depends only on the fixed point reached and not on the path of the inner iteration, so with $z=(x,v)$ and

$$ \dot z=\varphi(z), \qquad \varphi(x,v)=\bigl(v,\ M^{-1}(-\nabla E(x)+f_{\mathrm{ext}})\bigr), \tag{6.1} $$

Proposition 2.1 says $z_{n+1}=z_n+\Delta t\,\varphi(z_{n+1})$: backward Euler, a linear one-step method with first characteristic polynomial $\rho(\zeta)=\zeta-1$. Its only root $\zeta=1$ lies on the unit circle and is simple, so the root condition holds and the method is zero-stable. The local truncation error of backward Euler is $-\tfrac12\Delta t^2\ddot z+O(\Delta t^3)$, so the method is consistent of order one. By the Dahlquist equivalence theorem (consistency and zero-stability imply convergence), the scheme converges to the solution of (6.1) as $\Delta t\to0$ with $N\Delta t=T$ fixed, with global error $O(\Delta t)$.

*Assumptions behind this statement.* (a) The inner loop converges (Section 5; Assumption A2 and a convergent network). (b) $\varphi$ is Lipschitz on the region visited, i.e. $\nabla E$ is Lipschitz there, which holds for the stable Neo-Hookean density (smooth for all $F$, including $\det F\le0$) and for the penalty contact law ($C^{1,1}$). (c) The implicit equation selects a branch of stationary points that depends continuously on $z_n$, which holds where $M/\Delta t^2+\nabla^2E$ is invertible by the implicit function theorem. Where $\Phi_n$ has several stationary points (Remark 2.3), the scheme is a one-step method on each branch; jumping between branches is a discontinuity of the discrete flow and is not covered by the theorem.

*Zero-stability is a statement about the recurrence, not about $\Delta t$.* It says that a perturbation of size $\varepsilon$ introduced at one step grows at most by a constant factor (bounded by $e^{LT}$, $L$ the Lipschitz constant, in the classical nonstiff estimate) and not geometrically with the number of steps. Section 6.3 uses the sharper, $\Delta t$-independent form of this property that backward Euler enjoys on dissipative systems (B-stability).

### 6.2 Linear test problem: unconditional stability and dissipation

Take $E(x)=\tfrac12x^{\top}K_ex$ with $K_e$ symmetric positive semidefinite, no external force and no damping: $M\ddot x=-K_ex$. Let $K_e\phi_j=\omega_j^2M\phi_j$ with $\phi_j^{\top}M\phi_l=\delta_{jl}$ be the generalised eigenpairs and $x=\sum_jq_j\phi_j$. Each modal coordinate obeys the scalar oscillator $\ddot q=-\omega^2q$, i.e. the first-order system $\dot q=p$, $\dot p=-\omega^2q$. Backward Euler on it reads

$$ q_{n+1}=q_n+\Delta t\,p_{n+1},\qquad p_{n+1}=p_n-\Delta t\,\omega^2q_{n+1}, \tag{6.2} $$

or, in matrix form,

$$ \begin{aligned} \begin{pmatrix}1&-\Delta t\\ \omega^2\Delta t&1\end{pmatrix}\begin{pmatrix}q_{n+1}\\p_{n+1}\end{pmatrix}&=\begin{pmatrix}q_n\\p_n\end{pmatrix},\\[4pt] \begin{pmatrix}q_{n+1}\\p_{n+1}\end{pmatrix}=S\begin{pmatrix}q_n\\p_n\end{pmatrix},\qquad S&=\frac{1}{1+\omega^2\Delta t^2}\begin{pmatrix}1&\Delta t\\-\omega^2\Delta t&1\end{pmatrix}. \end{aligned} \tag{6.3} $$

> **Proposition 6.1 (amplification factor).** The eigenvalues of the amplification matrix $S$ are $\zeta_\pm=\bigl(1\pm i\,\omega\Delta t\bigr)^{-1}$, with
> $$ |\zeta_\pm|=\frac{1}{\sqrt{1+\omega^2\Delta t^2}}<1 \quad\text{for all } \omega\Delta t>0 . \tag{6.4} $$
> Backward Euler is therefore unconditionally stable on the linear problem, and dissipative: each mode loses the fraction $1-(1+\omega^2\Delta t^2)^{-1/2}$ of its amplitude per step, with high-frequency modes ($\omega\Delta t\gg1$) damped almost completely.

*Proof.* The matrix on the left of (6.3) has eigenvalues $1\pm i\,\omega\Delta t$ (its characteristic polynomial is $(1-\zeta)^2+\omega^2\Delta t^2$), so its inverse $S$ has eigenvalues $(1\pm i\,\omega\Delta t)^{-1}$, a complex-conjugate pair, and $|1\pm i\,\omega\Delta t|^2=1+\omega^2\Delta t^2$. The determinant check $\det S=(1+\omega^2\Delta t^2)^{-2}\,(1+\omega^2\Delta t^2)=(1+\omega^2\Delta t^2)^{-1}=|\zeta_+|^2$ agrees. $\blacksquare$

In the variational form this is the statement that $\Phi_n$ is strictly convex for the linear problem ($M/\Delta t^2+K_e\succ0$), so the stationary point is unique and the map $z_n\mapsto z_{n+1}$ is the linear contraction $S$ mode by mode.

### 6.3 Truncation at $K_{\mathrm{it}}$ iterations

In practice the inner loop is stopped after $K_{\mathrm{it}}$ queries and the returned iterate $\tilde x_{n+1}$ is not a stationary point. Define the *residual*

$$ r_n=\nabla\Phi_n(\tilde x_{n+1})=\frac{1}{\Delta t^2}M(\tilde x_{n+1}-y_n)+\nabla E(\tilde x_{n+1};x_n), \tag{6.5} $$

a force (in newtons) on the free corners; this is the free-corner force-residual metric of the campaign. Let $x_{n+1}^\ast$ be the stationary point that the loop was converging to and $\delta_n=\tilde x_{n+1}-x_{n+1}^\ast$ the position error of the step. Linearising $\nabla\Phi_n$ about $x_{n+1}^\ast$, where it vanishes,

$$ r_n=\nabla\Phi_n(\tilde x_{n+1})-\nabla\Phi_n(x^\ast_{n+1})\approx\Bigl(\frac{1}{\Delta t^2}M+H_n\Bigr)\delta_n, \qquad H_n=\nabla^2E(x_{n+1}^\ast;x_n). \tag{6.6} $$

> **Proposition 6.2 (per-step error from the residual).** If $H_n\succeq0$, then to first order
> $$ \|\delta_n\|\ \le\ \frac{\Delta t^2}{m_{\min}}\,\|r_n\|, \qquad \|\tilde v_{n+1}-v^\ast_{n+1}\|=\frac{\|\delta_n\|}{\Delta t}\ \le\ \frac{\Delta t}{m_{\min}}\,\|r_n\| . \tag{6.7} $$

*Proof.* $M/\Delta t^2+H_n\succeq M/\Delta t^2\succeq(m_{\min}/\Delta t^2)I$, so all eigenvalues of the symmetric matrix in (6.6) are at least $m_{\min}/\Delta t^2$ and the inverse has spectral norm at most $\Delta t^2/m_{\min}$. The velocity statement is the definition $\tilde v_{n+1}=(\tilde x_{n+1}-x_n)/\Delta t$. $\blacksquare$

**Accumulation over a horizon.** Measure the state error in the norm $\|(\delta x,\delta v)\|_{\Delta t}=\|\delta x\|+\Delta t\|\delta v\|$, in which one step's perturbation is $\|\delta_n\|+\Delta t\cdot\|\delta_n\|/\Delta t=2\|\delta_n\|\le 2\Delta t^2\|r_n\|/m_{\min}$. The classical zero-stability constant $e^{L(N-j)\Delta t}$ of Section 6.1 does not justify a useful bound here: $L$ is of the order of the highest elastic or contact frequency and $\Delta t\,L\gg1$ in the stiff regime. The correct statement is B-stability. Backward Euler is B-stable (Dahlquist; Hairer and Wanner, *Solving Ordinary Differential Equations II*, Section IV.12): on a dissipative (monotone) system, $\langle\varphi(z)-\varphi(\tilde z),\,z-\tilde z\rangle\le0$ in a suitable inner product (for (6.1) the energy inner product, when $\nabla^2E\succeq0$ and the damping is dissipative), any two backward-Euler trajectories satisfy $\|z_{n+1}-\tilde z_{n+1}\|\le\|z_n-\tilde z_n\|$ for every $\Delta t>0$. The propagation of a perturbation introduced at step $j$ to step $N$ is therefore non-expansive: the stability constant is $C_{\mathrm{stab}}=1$ in that norm, independent of $\Delta t$, $L$ and $N$ (the constant of passing between that norm and $\|\cdot\|_{\Delta t}$ is absorbed in $\lesssim$ below). Summing the $N$ per-step perturbations,

$$ \begin{aligned} \|x_N-x_N^{\mathrm{exact}}\|\ &\lesssim\ C_{\mathrm{stab}}\sum_{n<N}\frac{2\Delta t^2}{m_{\min}}\|r_n\| \ \le\ 2\,C_{\mathrm{stab}}\,\frac{N\Delta t^2}{m_{\min}}\max_n\|r_n\|\\ &=\ 2\,C_{\mathrm{stab}}\,\frac{T\,\Delta t}{m_{\min}}\max_n\|r_n\|, \end{aligned} \tag{6.8} $$

with $T=N\Delta t$. The drift is linear in the horizon and in the residual: a solver that leaves a residual $r$ behaves like backward Euler driven by a spurious force $r_n$ at every step. This is the long-horizon drift observed in the rollouts, and it is why the residual, not the per-step energy decrease, is the quantity to control. Where $H_n$ has negative eigenvalues (buckling, near-inverted cells), let $\lambda^-=\max\bigl(0,-\lambda_{\min}(H_n)\bigr)$; the bound (6.7) weakens to $\|\delta_n\|\le\Delta t^2\|r_n\|/(m_{\min}-\Delta t^2\lambda^-)$, which is still finite as long as $\Delta t^2\lambda^-<m_{\min}$; monotonicity, and with it $C_{\mathrm{stab}}=1$, is lost there, and $C_{\mathrm{stab}}$ grows with the strength and duration of the negative curvature.

## 7. Unpinned bodies: the translation null space

### 7.1 Singular fusion matrix

Without pinned corners, Lemma 4.1 gives $\operatorname{null}(K)=\operatorname{null}(B)=\{Zt\}$, three translations, and $K$ is singular. Moreover, the right-hand side of the normal equations is always orthogonal to the null space, whatever the network outputs:

$$ Z^{\top}B^{\top}W\Delta F=(BZ)^{\top}W\Delta F=0 \qquad\text{for every } \Delta F . \tag{7.1} $$

So the fusion can never move the centroid: the deformation-gradient targets carry no information about where the body is. Rotations are *not* in the null space, so no similar problem arises for rotations.

### 7.2 Blended fusion

A standard remedy is to blend the fit toward a target configuration $x_t$ in the mass norm:

$$ \begin{aligned} d&=\arg\min_d\ \frac12\bigl\|W^{1/2}(Bd-\Delta F)\bigr\|^2+\frac{\lambda}{2}\,\bigl\|x_k+d-x_t\bigr\|_M^2,\\ &\qquad \|u\|_M^2=u^{\top}Mu,\quad \lambda>0, \end{aligned} \tag{7.2} $$

with normal equations

$$ (K+\lambda M)\,d=B^{\top}W\Delta F+\lambda M(x_t-x_k). \tag{7.3} $$

$K+\lambda M$ is symmetric positive definite ($K\succeq0$, $M\succ0$), so $d$ is unique.

**Translation component.** Multiply (7.3) on the left by $Z^{\top}$ and use $Z^{\top}K=0$, (7.1) and (1.10):

$$ \lambda\,Z^{\top}Md=\lambda\,Z^{\top}M(x_t-x_k) \ \iff\ c(x_k+d)=c(x_t). \tag{7.4} $$

After *every* fusion the centroid of the new iterate equals the centroid of the target, for any $\lambda>0$ and any network output. The blend parameter $\lambda$ does not influence the translation at all.

**Shape component and fixed points.** Decompose $\mathbb{R}^{3P}=\operatorname{range}(B^{\top})\oplus\operatorname{null}(B)$ (an orthogonal sum, $\operatorname{null}(B)^\perp=\operatorname{range}(B^{\top})$). A fixed point $d=0$ of (7.3) requires

$$ B^{\top}W\Delta F=\lambda M(x_k-x_t). \tag{7.5} $$

Its null-space component is (7.4) with $d=0$: $c(x_k)=c(x_t)$. Its range component is a condition on the network: it must output the $\Delta F$ whose fused pull $B^{\top}W\Delta F$ cancels the blend's pull $\lambda M(x_k-x_t)$ in the shape directions. This is always *possible*, because once $c(x_k)=c(x_t)$ the vector $\lambda M(x_k-x_t)$ has zero translation component ($Z^{\top}M(x_k-x_t)=M_{\mathrm{tot}}(c(x_k)-c(x_t))=0$), so it lies in $\operatorname{range}(B^{\top})$, onto which $B^{\top}W$ maps. But it means that at a stationary point of $\Phi_n$ the network's natural output (zero projected gradient, hence $\Delta F=0$) is *not* a fixed point unless $x^\ast=x_t$ in every shape direction: the blend biases the shape unless $x_t$ has the right shape. Section 7.6 removes this bias by blending the centroid only.

### 7.3 Target $x_t=y_n$: free fall of the centroid, also in contact

Take the inertial prediction as target, $x_t=y_n$. By (2.1) and (1.10),

$$ c(y_n)=c_n+\Delta t\,\dot c_n+\Delta t^2g, \tag{7.6} $$

and by (7.4) every iterate after the first fusion, hence the returned $x_{n+1}$ whether or not the inner loop has converged, satisfies $c(x_{n+1})=c(y_n)$:

$$ M_{\mathrm{tot}}\,\frac{c_{n+1}-c_n-\Delta t\,\dot c_n}{\Delta t^2}=M_{\mathrm{tot}}\,g, \qquad \dot c_{n+1}=\frac{c_{n+1}-c_n}{\Delta t}=\dot c_n+\Delta t\,g . \tag{7.7} $$

This is momentum balance *without* contact forces: the centroid is in free fall at every step, whatever the contact state.

Compare with the translation component of the true stationarity condition. Multiplying (2.4) by $Z^{\top}$ and using (1.8), (1.9) and (1.10),

$$ Z^{\top}\nabla\Phi_n(x)=\frac{M_{\mathrm{tot}}}{\Delta t^2}\bigl(c(x)-c(y_n)\bigr)-F_{\mathrm{con}}(x), \tag{7.8} $$

so a stationary point $x^\ast$ of $\Phi_n$ has

$$ c(x^\ast)=c(y_n)+\frac{\Delta t^2}{M_{\mathrm{tot}}}F_{\mathrm{con}}(x^\ast), \qquad M_{\mathrm{tot}}\,\frac{c^\ast-c_n-\Delta t\,\dot c_n}{\Delta t^2}=M_{\mathrm{tot}}\,g+F_{\mathrm{con}}(x^\ast). \tag{7.9} $$

At the blend's fixed point $x^{\mathrm{blend}}_{n+1}$, (7.8) evaluates to $Z^{\top}\nabla\Phi_n(x^{\mathrm{blend}}_{n+1})=-F_{\mathrm{con}}(x^{\mathrm{blend}}_{n+1})\neq0$ whenever the body is in contact: the fixed point is *not* a stationary point of $\Phi_n$. Its centroid is $c^{\mathrm{blend}}_{n+1}=c(y_n)$; the centroid $c^{\mathrm{BE}}_{n+1}$ of a stationary point $x^{\mathrm{BE}}_{n+1}$ of $\Phi_n$ (a backward-Euler step) is given by (7.9), so the two differ by

$$ c^{\mathrm{blend}}_{n+1}-c^{\mathrm{BE}}_{n+1}=-\frac{\Delta t^2}{M_{\mathrm{tot}}}F_{\mathrm{con}}\bigl(x^{\mathrm{BE}}_{n+1}\bigr) : \tag{7.10} $$

the body sinks by $\Delta t^2F_{\mathrm{con}}(x^{\mathrm{BE}}_{n+1})/M_{\mathrm{tot}}$ per step relative to backward Euler, and by (7.7) its centroid velocity never receives the contact impulse, so the sinking is not a one-off offset but an unbounded free fall through the obstacle while the contact force (which grows with the penetration) deforms the shape. By (7.1) no choice of $\Delta F$ can correct this: the translation is decided entirely by the blend target. The inconsistency is structural, not a matter of training.

### 7.4 Target from a rigid semi-implicit step with the current contact force

Let the target's centroid be the rigid-body semi-implicit step computed with the contact force of the *current iterate*,

$$ c_{\mathrm{rig}}(x_k)=c_n+\Delta t\,\dot c_n+\Delta t^2g+\frac{\Delta t^2}{M_{\mathrm{tot}}}F_{\mathrm{con}}(x_k), \tag{7.11} $$

and let $x_t$ be any configuration with $c(x_t)=c_{\mathrm{rig}}(x_k)$ (for the full blend (7.2) the shape of $x_t$ also matters, see Section 7.2; for the centroid blend of Section 7.6 only $c_t=c_{\mathrm{rig}}(x_k)$ enters). By (7.4) the iteration imposes $c(x_{k+1})=c_{\mathrm{rig}}(x_k)$, and at a fixed point $x^\ast$

$$ \begin{aligned} c^\ast=c_{\mathrm{rig}}(x^\ast) \ &\iff\ M_{\mathrm{tot}}\,\frac{c^\ast-c_n-\Delta t\,\dot c_n}{\Delta t^2}=M_{\mathrm{tot}}\,g+F_{\mathrm{con}}(x^\ast)\\ &\iff\ Z^{\top}\nabla\Phi_n(x^\ast)=0, \end{aligned} \tag{7.12} $$

which is exactly (7.9), the translation component of stationarity with the *implicit* contact force. For the centroid blend (7.17)/(7.19), whose fixed points in the shape directions are those of the pinned analysis (Proposition 7.3(c)), stationarity in the shape directions (Section 5, Assumption A2) then gives $\nabla\Phi_n(x^\ast)=0$ in full, so the scheme at inner convergence is backward Euler in contact and everything in Section 6 applies. For the full blend (7.2) this is not automatic: by (7.5) the shape directions carry the $\lambda$-dependent bias $\lambda M(x^\ast-x_t)$, which the network must cancel exactly for the fixed point to be a stationary point of $\Phi_n$ (Assumption A2 then refers to that biased fixed-point condition rather than to $\nabla\Psi=0$). The translation update $c(x_{k+1})=c_{\mathrm{rig}}(x_k)$ is a Picard (fixed-point) iteration for the implicit translation equation (7.9).

> **Proposition 7.1 (Picard contraction constant).** Freeze the shape of the body and consider the centroid $c$ as the only unknown, so that $F_{\mathrm{con}}$ is a function of $c$ alone. With the normal penalty law, $f_i(c)=k_e\max\bigl(0,-g_i(c)\bigr)n_i$ and $g_i(c)=g_i^{0}+n_i\cdot(c-c^{0})$, the Picard map $T(c)=c_n+\Delta t\,\dot c_n+\Delta t^2g+\Delta t^2F_{\mathrm{con}}(c)/M_{\mathrm{tot}}$ has
> $$ \frac{\partial F_{\mathrm{con}}}{\partial c}=-\sum_{i\ \mathrm{active}}k_e\,n_in_i^{\top}, \qquad \operatorname{Lip}(T)\ \le\ \frac{\Delta t^2}{M_{\mathrm{tot}}}\sum_{i\ \mathrm{active}}k_e, \tag{7.13} $$
> and the Picard iteration converges (to the unique solution of (7.9) with frozen shape) when the right-hand side is below 1.

*Proof.* For an active pair ($g_i<0$), $\partial_c\bigl[k_e(-g_i)n_i\bigr]=-k_e\,n_i\,(\partial_cg_i)^{\top}=-k_en_in_i^{\top}$; inactive pairs contribute zero; the sum is negative semidefinite with norm at most $\sum_{\mathrm{active}}k_e$ (each $n_in_i^{\top}$ has norm one). $F_{\mathrm{con}}$ is piecewise linear and continuous, hence Lipschitz with this constant, and $T$ is Lipschitz with constant $\Delta t^2/M_{\mathrm{tot}}$ times it. The Banach fixed-point theorem gives convergence and uniqueness when the constant is below one. $\blacksquare$

**Campaign numbers.** $\Delta t=1/300\,$s, $h=0.025\,$m, penalty stiffness $k_e=\kappa Eh$ with $\kappa=10$, 100 active pairs, body volume $0.25\times0.25\times1.0\,\mathrm{m}^3=0.0625\,\mathrm{m}^3$, density $\rho=1000\,\mathrm{kg/m^3}$, so $M_{\mathrm{tot}}=62.5\,$kg:

$$ \begin{aligned} E=10^5\,\mathrm{Pa}:&\quad k_e=2.5\cdot10^4\,\mathrm{N/m},\quad \sum k_e=2.5\cdot10^6\,\mathrm{N/m},\\ &\quad \operatorname{Lip}(T)\le\frac{(1/300)^2\cdot2.5\cdot10^6}{62.5}=\frac{27.8}{62.5}\approx0.44 \quad(\text{converges});\\[6pt] E=10^6\,\mathrm{Pa}:&\quad \sum k_e=2.5\cdot10^7\,\mathrm{N/m},\quad \operatorname{Lip}(T)\le\frac{277.8}{62.5}\approx4.4 \quad(\text{diverges}). \end{aligned} $$

The constant scales as $E\Delta t^2\kappa\,h\,N_{\mathrm{active}}/(\rho\,V)$: it is the ratio of the contact stiffness to the inertial stiffness $M_{\mathrm{tot}}/\Delta t^2$ of the whole body, the same dimensionless group ($\kappa$ times $\Lambda^{-1}$ times a count ratio, in the notation of the cell-normalisation note) that decides whether penalty contact is stiff relative to the step.

### 7.5 Newton step on the centroid

Define the translation residual (a function of $c$ with the shape frozen)

$$ \begin{aligned} r_{\mathrm{tr}}(c)&=\frac{M_{\mathrm{tot}}}{\Delta t^2}\bigl(c-c_n-\Delta t\,\dot c_n\bigr)-M_{\mathrm{tot}}\,g-F_{\mathrm{con}}(c),\\ \frac{\partial r_{\mathrm{tr}}}{\partial c}&=\frac{M_{\mathrm{tot}}}{\Delta t^2}I_3+\sum_{i\ \mathrm{active}}k_e\,n_in_i^{\top}\ \succ\ 0 . \end{aligned} \tag{7.14} $$

Replacing the Picard target (7.11) by the Newton target $c_t=c(x_k)+\Delta c$ with

$$ \Bigl(\frac{M_{\mathrm{tot}}}{\Delta t^2}I_3+\sum_{i\ \mathrm{active}}k_e\,n_in_i^{\top}\Bigr)\Delta c=-\,r_{\mathrm{tr}}\bigl(c(x_k)\bigr) \tag{7.15} $$

keeps the same fixed point ($\Delta c=0\iff r_{\mathrm{tr}}=0\iff$ (7.9)) and removes the contraction condition:

> **Proposition 7.2 (convergence of the centroid Newton step).** $r_{\mathrm{tr}}$ is the gradient of the strongly convex, piecewise quadratic potential
> $$ \phi_{\mathrm{tr}}(c)=\frac{M_{\mathrm{tot}}}{2\Delta t^2}\|c-c_n-\Delta t\,\dot c_n\|^2-M_{\mathrm{tot}}\,g\cdot c+\sum_i\frac{k_e}{2}\max\bigl(0,-g_i(c)\bigr)^2, \tag{7.16} $$
> so (7.9) has a unique solution for any contact state. (i) If all active normals are equal (a single floor), the undamped Newton iteration converges from any start in finitely many steps. (ii) In general, (7.15) is a semismooth Newton step: it is exact (one step) as soon as the active set at the iterate equals the active set at the solution, and the iteration with an Armijo backtracking line search on $\phi_{\mathrm{tr}}$ converges globally.

*Proof.* Each term of (7.16) is convex ($\max(0,\cdot)^2$ of an affine function), the first is strongly convex with modulus $M_{\mathrm{tot}}/\Delta t^2$, and $\nabla\phi_{\mathrm{tr}}=r_{\mathrm{tr}}$ since $\partial_c\tfrac{k_e}{2}\max(0,-g_i)^2=-k_e\max(0,-g_i)n_i=-f_i(c)$. Strong convexity gives uniqueness. (i) With a common normal $n$, the components of $c$ orthogonal to $n$ enter $r_{\mathrm{tr}}$ linearly and are solved exactly in the first step. Along $n$, writing $c=\sigma n+\ldots$, the scalar residual $\sigma\mapsto\tfrac{M_{\mathrm{tot}}}{\Delta t^2}(\sigma-\sigma_{\mathrm{pred}})-\sum_ik_e\max(0,\sigma_i^{\mathrm{floor}}-\sigma)$ is increasing (slope at least $M_{\mathrm{tot}}/\Delta t^2$), piecewise linear and *concave* (it is a linear function minus a sum of convex functions $\max(0,\cdot)$). For an increasing concave function the tangent lies above the graph, so the first Newton iterate is at or below the root; from below the root the iterates increase monotonically and cannot overshoot (Newton–Fourier), and since the function is piecewise linear with finitely many breakpoints, the iterate lands exactly on the root once it is in the root's linear piece. (ii) On the region where the active set is that of the solution, $r_{\mathrm{tr}}$ is affine and one Newton step is exact. Global convergence of damped Newton on a strongly convex function with a piecewise-constant, uniformly positive definite generalised Hessian is the standard semismooth Newton result. $\blacksquare$

Both the Picard and the Newton variant leave the fixed point untouched; they differ only in how the translation is updated between queries. Note that in the actual algorithm the shape also changes between queries, so Propositions 7.1 and 7.2 describe the translation subproblem with the shape frozen; they are the conditions under which the translation does not, by itself, destabilise the inner loop.

### 7.6 Restricting the blend to the centroid

The full blend (7.2) pulls every corner toward $x_t$ and therefore biases the shape (Section 7.2). Since the only purpose of the blend is to fix the three translations, replace it by a penalty on the centroid alone:

$$ d=\arg\min_d\ \frac12\bigl\|W^{1/2}(Bd-\Delta F)\bigr\|^2+\frac{\lambda M_{\mathrm{tot}}}{2}\,\bigl\|c(x_k+d)-c_t\bigr\|^2 . \tag{7.17} $$

Let $m=(m_1,\dots,m_P)\in\mathbb{R}^{P}$ be the vector of corner masses, so that $M=\operatorname{diag}(m)\otimes I_3$ and $M_{\mathrm{tot}}=\mathbf{1}^{\top}m$, and let $V=MZ=M(\mathbf{1}\otimes I_3)=m\otimes I_3\in\mathbb{R}^{3P\times3}$, whose columns are the mass-weighted translation vectors $v_a=M\mathbf{1}_a=M(\mathbf{1}\otimes e_a)\in\mathbb{R}^{3P}$, $a=1,2,3$. Then $c(x)=V^{\top}x/M_{\mathrm{tot}}$, so the gradient and Hessian of the penalty in $d$ are

$$ \begin{aligned} \nabla_d\Bigl[\frac{\lambda M_{\mathrm{tot}}}{2}\|c(x_k+d)-c_t\|^2\Bigr]&=\lambda\,V\bigl(c(x_k+d)-c_t\bigr)\ \ (=\lambda\,MZ\bigl(c(x_k+d)-c_t\bigr)),\\ \nabla_d^2\Bigl[\frac{\lambda M_{\mathrm{tot}}}{2}\|c(x_k+d)-c_t\|^2\Bigr]&=\frac{\lambda}{M_{\mathrm{tot}}}VV^{\top}, \end{aligned} \tag{7.18} $$

a rank-3 matrix: $VV^{\top}=\sum_{a=1}^{3}v_av_a^{\top}=(mm^{\top})\otimes I_3$, whose $(i,j)$ corner block is $m_im_jI_3$, so the penalty couples every pair of corners through $\lambda\,m_im_j\,I_3/M_{\mathrm{tot}}$. The normal equations are

$$ \Bigl(K+\frac{\lambda}{M_{\mathrm{tot}}}VV^{\top}\Bigr)d=B^{\top}W\Delta F+\lambda\,MZ\bigl(c_t-c(x_k)\bigr). \tag{7.19} $$

> **Proposition 7.3.** (a) $K+\tfrac{\lambda}{M_{\mathrm{tot}}}VV^{\top}$ is symmetric positive definite for every $\lambda>0$. (b) The solution of (7.19) satisfies $c(x_k+d)=c_t$ exactly, for every $\lambda>0$ and every $\Delta F$. (c) At $c(x_k)=c_t$ the penalty's pull vanishes identically, so $d=0$ if and only if $B^{\top}W\Delta F=0$: no shape direction is biased, and the fixed points in the shape directions are those of the pinned analysis (Section 5).

*Proof.* (a) If $d^{\top}Kd+\tfrac{\lambda}{M_{\mathrm{tot}}}\|V^{\top}d\|^2=0$ then $Bd=0$, so $d=Zt$ by Lemma 4.1, and $V^{\top}Zt=Z^{\top}MZt=M_{\mathrm{tot}}t$, so $\lambda M_{\mathrm{tot}}\|t\|^2=0$ and $t=0$. (b) Multiply (7.19) by $Z^{\top}$: $Z^{\top}K=0$, $Z^{\top}V=M_{\mathrm{tot}}I_3$, $Z^{\top}B^{\top}=0$, $Z^{\top}MZ=M_{\mathrm{tot}}I_3$, giving $\lambda V^{\top}d=\lambda M_{\mathrm{tot}}(c_t-c(x_k))$, i.e. $c(d)=c_t-c(x_k)$. (c) is read off (7.19) with $c(x_k)=c_t$: the right-hand side reduces to $B^{\top}W\Delta F$, and $d=0$ iff it vanishes. $\blacksquare$

**Solving (7.19) with the cached factor.** Because $KZ=0$ and $V=MZ$, the rank-3 system splits without a Woodbury correction. Choose one reference corner $r$ and let $f$ denote all other corners; $K_{ff}$ is SPD by Corollary 4.2 (a body with one artificially fixed corner) and its factor is cached. Write $d=\hat d+Zt$ with $\hat d_r=0$. Multiplying (7.19) by $Z^{\top}$ as in (b) gives $V^{\top}d=V_f^{\top}\hat d_f+M_{\mathrm{tot}}t=M_{\mathrm{tot}}\bigl(c_t-c(x_k)\bigr)$; substituting back removes the rank-3 term, leaving

$$ K\hat d=b-\frac{1}{M_{\mathrm{tot}}}V\,Z^{\top}b, \qquad\text{where}\quad b:=B^{\top}W\Delta F+\lambda MZ\bigl(c_t-c(x_k)\bigr), \tag{7.20} $$

whose right-hand side has zero translation component ($Z^{\top}V=M_{\mathrm{tot}}I_3$), so it lies in $\operatorname{range}(K)$. In fact it simplifies: by (7.1) and $Z^{\top}MZ=M_{\mathrm{tot}}I_3$, $Z^{\top}b=\lambda M_{\mathrm{tot}}\bigl(c_t-c(x_k)\bigr)$, so $\frac{1}{M_{\mathrm{tot}}}VZ^{\top}b=\lambda MZ\bigl(c_t-c(x_k)\bigr)$ is exactly the blend term of $b$, and the right-hand side of (7.20) equals $B^{\top}W\Delta F$. Thus $\lambda$ drops out of the shape solve entirely and enters only through the constraint $c(x_k+d)=c_t$ that fixes $t$. The unique solution with $\hat d_r=0$ is obtained from the free block alone:

$$ K_{ff}\,\hat d_f=\Bigl(b-\frac{1}{M_{\mathrm{tot}}}VZ^{\top}b\Bigr)_f=\bigl(B^{\top}W\Delta F\bigr)_f, \qquad t=c_t-c(x_k)-\frac{1}{M_{\mathrm{tot}}}V_f^{\top}\hat d_f . \tag{7.21} $$

One solve with the cached factor and a $3\times3$ computation; this is the Schur-complement form of the rank-3 update. (A direct numerical check on a $2\times2\times2$ cell block confirms that (7.21) reproduces the solution of (7.19) to round-off, that $c(x_k+d)=c_t$ to $10^{-15}$ for $\lambda\in\{10^{-3},1,10^{3}\}$, and that the smallest eigenvalue of the matrix in (7.19) is positive.)

### 7.7 Alternative: per-cell translation targets

Instead of blending, the network could output a full displacement proposal for all 24 corner coordinates of each cell (3 translation values in addition to the 21 shape values of Section 8). The per-cell operator is then the identity on the cell's corners, $B$ is the gather matrix (one row block per cell copying its 8 corners) and is injective on all $3P$ degrees of freedom without pins. The fusion $K=B^{\top}WB$ becomes diagonal, $K_{ii}=\sum_{c\ni i}w_c\,I_3$, and the fused displacement is the weighted average of the proposals of the cells sharing each corner, exactly as in projective dynamics with per-element position projections. The projected gradient $WBK^{-1}\nabla\Phi_n$ then delivers to each cell the valence-weighted gradient of its eight corners, whose sum is the net force on the cell; the network sees translation information and Section 5 applies with $\operatorname{null}(B)=\{0\}$. The price is 24 outputs per cell and the loss of the exact translation invariance of the targets, which is what makes the axes representation frame-independent.

## 8. Extension to 21 targets per cell (axes and four warping vectors)

**Mode basis.** For a trilinear hex the eight corner positions $x_{c,k}$ decompose linearly into translation (3 values), affine (9) and warping (12) modes. With the corner coordinates $\xi_k\in\{-1,+1\}^3$ the mode functions on the corners are

$$ \begin{aligned} &\text{translation (1 function):}&& 1,\\ &\text{affine (3):}&& \xi_k^{a},\quad a=1,2,3,\\ &\text{warping (4):}&& \xi_k^{a}\xi_k^{b}\ (a<b)\quad\text{and}\quad \xi_k^{1}\xi_k^{2}\xi_k^{3}, \end{aligned} \tag{8.1} $$

eight functions that are mutually orthogonal on the eight corners, $\sum_k\mu(\xi_k)\nu(\xi_k)=8\,\delta_{\mu\nu}$ (a Hadamard basis). The corresponding coefficients, each a vector in $\mathbb{R}^3$, are

$$ \begin{aligned} t_c&=\frac18\sum_kx_{c,k}, &\qquad \alpha_c^{a}&=\frac18\sum_k\xi_k^{a}\,x_{c,k},\\ \omega_c^{ab}&=\frac18\sum_k\xi_k^{a}\xi_k^{b}\,x_{c,k}, &\qquad \omega_c^{123}&=\frac18\sum_k\xi_k^{1}\xi_k^{2}\xi_k^{3}\,x_{c,k}, \end{aligned} \tag{8.2} $$

and the inverse relation (the trilinear interpolant itself) is

$$ x_c(\xi)=t_c+\sum_{a}\alpha_c^{a}\,\xi^{a}+\sum_{a<b}\omega_c^{ab}\,\xi^{a}\xi^{b}+\omega_c^{123}\,\xi^{1}\xi^{2}\xi^{3}, \qquad x_{c,k}=x_c(\xi_k). \tag{8.3} $$

$3+9+9+3=24$ values, a change of basis of the 24 corner coordinates.

**Deformation gradient in the mode basis.** Since $\nabla N_k=\tfrac2h\nabla_\xi N_k$, the deformation gradient is $F_c(\xi)=\tfrac2h\,\partial x_c/\partial\xi$, and differentiating (8.3) column by column gives the explicit linear function of the 21 non-translation values

$$ F_c(\xi)=\frac{2}{h}\bigl[\,\varphi_c^{1}(\xi)\ \ \varphi_c^{2}(\xi)\ \ \varphi_c^{3}(\xi)\,\bigr], \qquad \varphi_c^{a}(\xi)=\alpha_c^{a}+\omega_c^{ab}\,\xi^{b}+\omega_c^{ad}\,\xi^{d}+\omega_c^{123}\,\xi^{b}\xi^{d}, \tag{8.4} $$

where for each column $a$ the direction indices $b<d$ are the two other coordinate directions (the subscript $c$ remains the cell index, by the convention of Section 1) and $\omega_c^{ab}$ is read with its indices sorted ($\omega_c^{ba}=\omega_c^{ab}$). Written out, the first column is $\alpha_c^{1}+\omega_c^{12}\xi^{2}+\omega_c^{13}\xi^{3}+\omega_c^{123}\xi^{2}\xi^{3}$, the second $\alpha_c^{2}+\omega_c^{12}\xi^{1}+\omega_c^{23}\xi^{3}+\omega_c^{123}\xi^{1}\xi^{3}$, the third $\alpha_c^{3}+\omega_c^{13}\xi^{1}+\omega_c^{23}\xi^{2}+\omega_c^{123}\xi^{1}\xi^{2}$; each is evaluated at $\xi=\xi_q$ for the Gauss points. The translation $t_c$ does not appear. At the centre, $F_c(0)=\tfrac2h[\alpha_c^1\ \alpha_c^2\ \alpha_c^3]$: the nine affine values are the centre deformation gradient, and the local axes of (1.6) are $A_c=R_c^{\top}F_c(0)$. The four warping vectors in the local frame are $R_c^{\top}\omega_c^{12},R_c^{\top}\omega_c^{13},R_c^{\top}\omega_c^{23},R_c^{\top}\omega_c^{123}$ (scaled by $2/h$ to match the axes). A numerical check of (8.4) against the direct evaluation (1.4) at all eight Gauss points agrees to $10^{-14}$.

> **Lemma 8.1.** The linear map from the 21 non-translation mode values to the eight Gauss-point deformation gradients $(F_c(\xi_q))_{q=1}^{8}\in\mathbb{R}^{72}$ is injective.

*Proof.* This is step (ii) of the proof of Lemma 4.1: each column of (8.4) is bilinear in two of the $\xi$'s and is determined by its values at the four Gauss sign patterns of those two coordinates. $\blacksquare$

**Per-Gauss-point targets.** Let the network output, per cell, increments of the 21 values: nine for the axes and twelve for the four warping vectors, all in the frozen frame. Rotating back with $R_c$ and inserting into (8.4) gives per-Gauss-point target increments $\Delta F_{c,q}$, $q=1,\dots,8$, and the fusion becomes

$$ d=\arg\min_d\ \frac12\sum_c\sum_qw_q\bigl\|G_{c,q}d-\operatorname{vec}\Delta F_{c,q}\bigr\|^2 \quad\text{subject to } d_p=d_p^{\mathrm{presc}} . \tag{8.5} $$

By construction the eight targets of a cell are consistent (they come from one set of 21 values), so the per-cell fit is exact: for a single isolated cell the residual of (8.5) is zero, with the corner displacement given by (8.3) up to the cell's translation. In the assembled problem the only residual comes from shared-corner agreement between neighbouring cells, and the network now prescribes the warping increment instead of having it penalised (Remark 3.1).

**Equivalent mode-space form and the fusion matrix.** Let $P_c\in\mathbb{R}^{21\times24}$ be the matrix that extracts the 21 mode values from a cell's 24 corner coordinates by (8.2), and $\Gamma_q\in\mathbb{R}^{9\times21}$ the matrix of (8.4) at $\xi_q$, so that $G_{c,q}d=\Gamma_qP_cd_c$ and $\operatorname{vec}\Delta F_{c,q}=\Gamma_q\,\Delta m_c$ with $\Delta m_c$ the 21 target increments. Then

$$ \begin{aligned} \sum_qw_q\bigl\|G_{c,q}d-\operatorname{vec}\Delta F_{c,q}\bigr\|^2&=\bigl(P_cd_c-\Delta m_c\bigr)^{\top}Q_c\bigl(P_cd_c-\Delta m_c\bigr),\\ Q_c&=\sum_qw_q\,\Gamma_q^{\top}\Gamma_q\ \succ\ 0, \end{aligned} \tag{8.6} $$

where $Q_c$ is positive definite by Lemma 8.1. The per-cell operator is thus the $21\times24$ matrix $P_c$, with $\operatorname{null}(P_c)$ = the cell's translations (the translation row is the one omitted), and the assembled fusion matrix

$$ K=\sum_cP_c^{\top}Q_cP_c \tag{8.7} $$

is not merely analogous to the fusion matrix of Section 3: it is literally the same matrix. Since $G_{c,q}=\Gamma_qP_c$ on the columns of cell $c$ and $W=\operatorname{diag}(w_qI_9)$,

$$ \sum_cP_c^{\top}Q_cP_c=\sum_c\sum_qw_q\,P_c^{\top}\Gamma_q^{\top}\Gamma_qP_c=\sum_c\sum_qw_q\,G_{c,q}^{\top}G_{c,q}=B^{\top}WB=K \tag{8.8} $$

(a direct numerical check on a $2\times2\times2$ block gives $\|K_{8}-K_{3}\|/\|K_{3}\|=7\cdot10^{-16}$ for the two assemblies (8.7) and (3.5)). Hence the null space (Lemma 4.1), the positive definiteness of $K_{ff}$ for a pinned connected body (Corollary 4.2), the size $3P_{\mathrm{free}}$, the sparsity pattern (two corners are coupled iff they share a cell) and the cached factor are all reused unchanged. What changes is only the right-hand side $B^{\top}W\Delta F$ of the normal equations (3.6): the $72C$-vector $\Delta F$ now holds the per-Gauss-point $\operatorname{vec}\Delta F_{c,q}$ in the eight slots of cell $c$ instead of the repeated per-cell $\operatorname{vec}\Delta F_c$. The energy path is unchanged: $E$ is still evaluated from the eight Gauss-point deformation gradients computed from the corners by (1.4), and the projected gradient (5.2) is still $WBK^{-1}\nabla\Phi_n$; $B$ already had its $72C$ Gauss-point rows in Section 3. What is new is that the network can output per-Gauss-point targets, so the warping part of the projected gradient, which was present in (5.2) all along but could not be acted on by a per-cell affine target (Remark 3.1), becomes actionable.

## 9. Summary

| | Physical objective | Solver choice | Assumption |
|---|---|---|---|
| Time integrator | Backward Euler on $\dot x=v$, $M\dot v=-\nabla E+f_{\mathrm{ext}}$, as the stationarity condition of $\Phi_n$ (2.2) | The inner loop that finds the stationary point: network step plus fusion | Inner convergence; if truncated, residual $r_n$ (6.5) |
| Constitutive law | Stable Neo-Hookean, eight-point quadrature; viscous damping evaluated at $v_{n+1}$ | None (the energy path is not learned) | $E$ bounded below, $C^1$ (A1); Hessian PSD for the bound (6.7) |
| Contact law | Newton engine penalty, pair set frozen at $x_n$, force implicit in $x_{n+1}$ | Translation handling for unpinned bodies: blend target (7.11) or Newton target (7.15), centroid-only blend (7.17) | Picard constant $<1$ (7.13) unless the Newton step is used |
| Local step | none | Network output: 9 axes (Section 3) or 21 values (Section 8); frozen frames $R_c$ | Network consistency (A2): zero step iff zero projected gradient |
| Global step | none | Fusion metric $W$ (Gauss weights), least squares in $F$ | Connected body, at least one pin (Corollary 4.2) or centroid blend (Prop. 7.3) |
| Starting candidate | none | $x_0=y_n$ or previous iterate | Selects the branch of stationary points reached (Remark 2.3) |

**Proven in this note.**

- Stationary points of $\Phi_n$ are backward-Euler steps (Proposition 2.1) and exist (Proposition 2.2).
- For a connected pinned body, the fusion is a projection with trivial null space (Lemma 4.1, Corollary 4.2), and the projected gradient vanishes iff $\nabla\Phi_n$ vanishes (Proposition 5.1): the fusion changes the path, not the fixed points.
- At inner convergence the scheme is backward Euler, a one-step method, hence zero-stable and convergent (Section 6.1); on the linear problem it is unconditionally stable with amplification $(1+\omega^2\Delta t^2)^{-1/2}$ (Proposition 6.1).
- A residual $r_n$ perturbs the step by at most $\Delta t^2\|r_n\|/m_{\min}$ and accumulates linearly over the horizon (Proposition 6.2, (6.8)).
- For unpinned bodies, the blend toward $y_n$ forces free fall of the centroid in contact, and no target can correct it (Section 7.3); the rigid target with the current contact force restores the implicit translation equation (7.12) with Picard constant (7.13); the centroid Newton step has the same fixed point and, with the shape frozen, converges unconditionally for a single floor (finite termination) and globally with Armijo backtracking for several normals (Proposition 7.2); the centroid-only blend does not bias the shape (Proposition 7.3) and is solvable with the cached factor, with $\lambda$ dropping out of the shape solve (7.21).
- The 21-value extension leaves the fusion matrix unchanged, $K=B^{\top}WB$ exactly (8.8), so size, sparsity, definiteness and the cached factor are preserved; only the right-hand side changes (Section 8).

**Not proven (and not claimed).**

- Convergence of the learned inner iteration to a stationary point, or its rate. Assumption A2 (the network's step vanishes only at zero projected gradient) is a property of the trained network.
- Uniqueness of the stationary point of $\Phi_n$ (nonconvex elasticity; Remark 2.3), and which branch a truncated iteration follows.
- Anything about the size of the residual left after $K_{\mathrm{it}}$ iterations; Section 6.3 only converts a measured residual into a trajectory error.
- Convergence of the coupled shape-and-translation inner loop for unpinned bodies; Propositions 7.1 and 7.2 treat the translation with the shape frozen.

## Appendix A. Numerical checks behind the derivation

The following identities were verified in double precision on the reference cube ($h=0.025\,$m) and on a $2\times2\times2$-cell unpinned block with random corner masses; they are checks of the algebra, not of the trained solver.

- $\operatorname{rank}G_c=21$ for the $72\times24$ single-cell operator; the three translation vectors are in its null space to machine precision; a rotation of the reference cell gives $F=R$ with $\|F-I\|_F=0.42$ for a $0.3\,$rad rotation (Lemma 4.1).
- An affine corner displacement produces the same $F$ increment at all eight Gauss points, spread $2\cdot10^{-16}$ (Remark 3.1).
- The mode coefficients (8.2) reconstruct random corner positions through (8.3) to $2\cdot10^{-16}$, and (8.4) matches the direct Gauss-point evaluation to $2\cdot10^{-14}$; $F_c(0)=\tfrac2h[\alpha^1\ \alpha^2\ \alpha^3]$ exactly.
- The eigenvalues of the backward-Euler amplification matrix (6.3) have modulus $(1+\omega^2\Delta t^2)^{-1/2}$ at $\omega\Delta t\in\{0.1,1,10\}$: $0.9950,\ 0.7071,\ 0.0995$ (Proposition 6.1).
- Picard constants: $0.444$ at $E=10^5\,$Pa and $4.444$ at $E=10^6\,$Pa (Proposition 7.1); $3P_{\mathrm{free}}=14\,520$, $9C=36\,000$, $72C=288\,000$.
- The 21-target assembly (8.7) and the Section 3 assembly $B^{\top}WB$ agree to $7\cdot10^{-16}$ in relative Frobenius norm on the $2\times2\times2$ block (identity (8.8)).
- Unpinned $K$ has exactly three zero eigenvalues; for the full blend (7.3) and the centroid blend (7.19) with $\lambda\in\{10^{-3},1,10^{3}\}$ and random $\Delta F$, $\|c(x_k+d)-c(x_t)\|\le1.1\cdot10^{-15}$ (Section 7.2, Proposition 7.3(b)); the split solve (7.21) agrees with the direct solve of (7.19) to $10^{-10}$ or better and the omitted reference-corner rows of (7.20) are satisfied to $10^{-12}$.
- Single-floor translation problem with 100 pairs at random touching heights, $\Delta t=1/300$, $M_{\mathrm{tot}}=62.5\,$kg, predicted centroid 2 cm below the floor: the undamped Newton iteration (7.15) reaches $|r_{\mathrm{tr}}|<10^{-9}\,$N in 1 to 4 steps from starts at $\pm1\,$m and $0$, for both $E=10^5$ and $E=10^6\,$Pa (Proposition 7.2(i)); the Picard iteration (7.11) converges at $E=10^5\,$Pa (oscillating, ratio about $0.44$) and settles into a period-2 cycle far from the root at $E=10^6\,$Pa (Proposition 7.1).
