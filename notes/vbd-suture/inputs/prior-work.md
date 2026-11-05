# Prior work: suture and needle insertion simulation, and what Newton has today

Date: 2026-11-05. Author: Claude (Fable 5.1) for Anka. Phase 1 of the task in `task-prompt.md`. Read-only study; no code was changed.

How to read this document. Part A covers the literature and SOFA. Part B covers Newton at `upstream/main` 84242814c with file and line citations. Part C lists what we can borrow. Three research agents collected the literature with verbatim quotes; their full reports with every URL are in `lit-needle-insertion-raw.md`, `lit-suture-thread-raw.md`, and `lit-sofa-pbd-rods-raw.md` in this folder. I checked the items marked "[verified]" myself against the primary source. Every other literature claim is agent-reported with a cited source. Sentences marked "(my inference)" are my judgement, not a source.

## Part A. Literature and SOFA

### A.1 Family 1: conforming mesh plus Lagrange multipliers (remeshing)

**A.1.1 Chentanez et al., SIGGRAPH 2009.** "Interactive simulation of surgical needle insertion and steering", ACM TOG 28(3). Quote: "an efficient method for coupling a 3D finite element simulation with a 1D inextensible rod with stick-slip friction". Coupling: the tet mesh is remeshed so nodes lie on the needle path ("the node snap, the edge split, the face split, and the tetrahedron split"), and each such node is tied to the rod by three Lagrange multipliers. Tunnel: the node chain is the tunnel; the tip adds a cutting force `f_cut` that varies by tissue type. Friction: per-node stick-slip with a friction force per unit length; "Currently, we treat only sliding friction." Reusable with VBD (agent assessment, I agree): the per-node stick/slip state and the tip cutting force. Not reusable: the remeshing (recoloring and adjacency rebuild in VBD) and the LCP solve.

**A.1.2 DiMaio and Salcudean, IEEE T-RA 2003 and IEEE TBME 2005.** 2D linear elastostatic FEM. A measured shaft force profile is applied to the tissue nodes as boundary conditions; the system is condensed to the shaft DOFs; boundary-condition switches use low-rank matrix updates. 2005 adds a large-strain flexible needle and haptics. Puncture criterion and friction law: not stated in the abstracts the agent read. Reusable: the idea that a tissue node switches between free, stuck to the needle, and sliding on the needle. Not reusable: the global matrix tricks.

**A.1.3 Goksel et al., MICCAI 2005 / Computer Aided Surgery 2006; IEEE Trans. Haptics 2011.** Quote: "node repositioning and node addition are the two methods of mesh modification examined for coarse meshes." Tissue nodes are moved onto the needle path or added, then treated as contact nodes; stiffness updates use the Woodbury formula; later work runs the mesh adaptation on the GPU. Reusable: node repositioning in material space is tolerable in VBD. Node addition changes topology and coloring.

### A.2 Family 2: constraints without remeshing (the SOFA lineage)

**A.2.1 Duriez, Guebert, Marchal, Cotin, Grisoni, MICCAI 2009 [verified, PubMed abstract].** "Interactive simulation of flexible needle insertions based on constraint models." Quote: "the interactions between soft tissues and flexible instruments are reproduced using a set of dedicated complementarity constraints. Each constraint is positionned and applied to the deformable models without requiring any remeshing. Our method allows for the 3D simulation of different physical phenomena such as puncture, cutting, static and dynamic friction at interactive frame rate." The Guebert thesis (2010) describes three constraint types: membrane penetration with a force threshold `f_p`, tip cutting, and friction along the whole shaft that keeps the needle on its path. Reusable: this constraint set is the cleanest template for a non-remeshing VBD needle. Not reusable: the dual Gauss-Seidel on the compliance matrix.

**A.2.2 Guebert, Duriez, Grisoni, VRIPHYS 2008; Guebert et al., SCA 2009 poster.** The 2008 paper gives "a generic way of dealing with a set of different constraints (bilateral, unilateral, dry friction)" and demonstrates a point-sliding-on-curve constraint with friction "depending on the force pulling the object on the curve". The 2009 poster "Suturing Simulation Based on Complementarity Constraints" has no retrievable text; the thesis is the only record.

**A.2.3 Courtecuisse group, 2014 to 2024.** Courtecuisse et al. 2014 (MedIA) is a contact and cutting paper with an asynchronous preconditioner, not a needle paper. Adagolodjo et al. 2019 (IEEE T-RO) uses "constraint-based interaction models allowing for the simulation of needle insertion and complex nonlinear phenomena (friction, puncture, and insertion)" for robot control. Perrusi, Baksic, Courtecuisse, EG 2021 short: "Atomic constraints phi_i prevent lateral motion of the needle along the inserted trajectory, and psi_i adds friction along the needle shaft." with parameters "The penetration force fp, the cutting force fc, the tearing force ft, the friction at the surface mu_p and during the insertion mu_i." Martin, Zeng, Courtecuisse, EG 2023 short [verified, PDF]: "we generate a new constraint at each intersection point between the edges of the needle and the triangles of the tetrahedral volume mesh. Constraints are then transferred to the mechanical DOFs of both models using barycentric coordinates." Each group has two bilateral rows orthogonal to the shaft and one friction row along it. Reusable: constraint placement by geometry and barycentric transfer to tet DOFs; the two-plus-one row layout. Not reusable: the hybrid direct and Gauss-Seidel dual solver.

**A.2.4 Lenoir et al., 2002, 2004, 2006.** 2002 (ESAIM): a dynamic spline thread with Lagrange multipliers and penalty self-collision, for knots. 2004 ISMS "A Suture Model for Surgical Simulation" [verified, HAL abstract]: "The suture is simulated via socalled 'sliding point' constraints, which allow the spline to move freely while constrained to pass through specific piercing points. The direction of the spline at these points can also be imposed. Moreover, to enhance realism, an adapted model of friction is proposed, which allows the thread to remain fixed at the piercing point or slides through it." 2006 (Computers and Graphics): catheter and guidewire as one composite rod whose material changes locally. Reusable: the sliding point is the cleanest statement of a tunnel primitive, one point fixed in tissue and a free parameter along the thread, with stick or slide friction.

### A.3 Suturing and thread models

**A.3.1 Berkley et al., IEEE TVCG 2004 [verified, author PDF].** Linear FE tissue with precomputation. Puncture: "If this value exceeds a prescribed needle puncture force, the needle 'pops' through the skin", where the value is the dot product of needle direction and surface normal times the applied force. "After puncture, the contact point is secured to the shaft of the needle. Some simple rules that incorporate friction and needle rotation determine how the contact point slides along the needle." Suture: "Sutures can be modeled as stiff springs that only support tension." "The second anchor point is treated as a 'pulley' that allows the suture to slide through, but will also bear a load." Limit: "In the current simulator, the needle enters and exits at the same point." Reusable: the orientation-weighted puncture threshold, the anchor as a load-bearing pulley.

**A.3.2 Brown et al., 2002 and 2004.** Kinematic follow-the-leader rope for knot tying at up to 140 Hz, collision management with friction so knots can be pulled tight; tissue coupling not stated. Reusable: knot tightening as a test case for a rod, not the method.

**A.3.3 Payandeh group: LeDuc 2003, Marshall 2005, Payandeh and Shi 2010.** Mass-spring tissue and a geometric rope. "When the needle pierces the deformable model, the surface mesh is subdivided at the contact point." The pierced node hops from one suture node to the next when the pulling force exceeds a friction threshold (agent paraphrase of the 2003 body). A tearing rule: "if needle insertion points are too close from each other or to the edge of the wound, when the suture is pulled, the suture will tear the soft tissue instead of suturing the incision together." Reusable: the tearing criterion as an evaluation rule; subdivision at the pierce point for coarse surfaces.

**A.3.4 Choi, Chan, Pang, J. Medical Systems 2012.** PhysX: spring-connected boxes for tissue, chained rigid segments with spherical joints for needle and thread. Mechanism of insertion and friction: not stated. Lesson: force readback from a black-box engine is hard.

**A.3.5 Jia and Pan, ICALIP 2010.** Follow-the-leader thread. "Sliding constraints are formed when the suture is being pulled through the object." "At each sliding constraint point, internal tension force of the suture is applied to the object, while the deformation of the object affects the position of the constraint point conversely." This is the two-way sliding point of A.2.4 with a kinematic thread.

**A.3.6 Spillmann and Teschner, SCA 2007 (CORDE) and Eurographics 2008.** Cosserat rod finite elements with per-element energies; knots at 26 fps with 2 ms steps. 2008: adaptive control points by energy minimisation, impact zones, "a position-based approximative solution for frictional effects". Reusable: the rod energies; the impact-zone position correction is close to VBD contact energies. Adaptive insertion would force recoloring.

### A.4 SOFA components today

**A.4.1 BeamAdapter plugin, `AdaptiveBeamSlidingConstraint`.** Registration string: "Constrain a rigid to be attached to a beam (only in position, not the orientation)." Convention: "object1 is the beam model, object2 is the list of point constraints"; each point is projected on the closest beam segment (Newton with a brute-force fallback) and tracked by curvilinear abscissa. Three rows per point in the local frame of the projection; the two transverse rows receive force, the tangent row receives none. Forces and torques go to the two end frames of the containing beam through the spline weights. `AdaptiveBeamLengthConstraint` is a unilateral elongation limit. `WireRestShape` returns the rest transform at an abscissa. `InterventionalRadiologyController` deploys the wire through a fixed insertion point by resampling a fixed-size topology with pinned inactive nodes. Reusable: the two-transverse-rows-plus-free-tangent layout, the abscissa-cached projection, the fixed-size topology.

**A.4.2 SOFA core `SlidingLagrangianConstraint` and `BilateralLagrangianConstraint`.** Sliding: point P on object1 against segment A-B on object2; barycentric projection clamped to [0,1]; transverse rows bilateral with weights (1-bary) on A and bary on B; two axial rows unilateral so P moves freely between A and B and stops at the ends ("(forbidden)|-> (free) <-|(forbidden)" in the source comments). Bilateral: equality of two points, three rows, optional rest vector, `load` factor for incremental loading. Reusable: both map to small VBD force elements compatible with coloring.

**A.4.3 SofaDefrost Cosserat plugin, needle insertion examples.** `PointsManager` adds a constraint point at the beam tip as it advances; the points live under a `BarycentricMapping` into the tet FEM. `CosseratNeedleSlidingConstraint` constrains local y and z only (`useDirections` default (0,1,1)); "It 's also important to say that the x direction is not taken into account". The controller punctures when "the contact force is large enough to go through the tissue" (threshold 3.0 in the example), then deactivates the contact constraint, adds a constraint point each `constraintDistance` of tip travel, removes points on retraction, and removes the entry point when the needle leaves. Contact uses `FrictionContactConstraint` with mu 0.1. Reusable: this is a complete recipe for Option 1 in `options.md`, point by point.

**A.4.4 InfinyTech3D `CollisionAlgorithm` and `ConstraintGeometry` (SOFA v25.06, GPL-3.0).** "Needle insertion is simulated in phases: puncture, insertion, retraction". `InsertionAlgorithm` has `punctureForceThreshold`, `tipDistThreshold`, `enablePuncture`, `enableInsertion`, `enableShaftCollision`; `ConstraintInsertion` has a friction coefficient. The example uses a Rigid3 beam needle and a tetrahedral corotational tissue. Which rows are bilateral or unilateral is not stated in the sources the agent read. Reusable: the phase machine and the explicit shaft-collision switch.

**A.4.5 Other SOFA items.** SoftRobots `CableConstraint`: barycentric via-points plus one length row, "the others will represent sliding points for the cable"; the cheapest rod-through-soft-body model but one dense row. `SuturePointPerformer`: a mouse tool that adds one spring between two picked vertices. No `SutureConstraint` component exists. Forum threads (2020, 2021) point users to springs or to the Cosserat needle scene.

### A.5 Position-based and VBD-family work

**A.5.1 Rod models.** Kugelstadt and Schomer, SCA 2016: stretch-shear constraint between two positions and a quaternion, bend-twist constraint between two quaternions, "quaternions are treated in the exact same way as positions". Deul et al., CGF 2018: rigid segments with Cosserat constraints, a direct tree solve, and "we alternate the computation of our rod solver with the computations of the non-linear Gauss-Seidel solver for loop-closing constraints and collision constraints." XPBD (Macklin et al. 2016): `U = 1/2 C^T alpha^-1 C` with compliance alpha. Reusable: the energies; the direct-plus-local alternation matches Newton issue #4794.

**A.5.2 Yu, Pan, Qin, Hao, Wang, CAVW 2020 [verified, Semantic Scholar abstract].** "External interaction involves the interplay between needle/suture and the soft tissue, which are both deformed by position-based dynamics (PBD) with different constraints." "once the force exceeds a threshold, the needle tip will puncture and penetrate into the soft tissue and generate a path." "we propose a novel coupling method by matching and generating the constraints among needle, suture, and penetration path." Third-party wording (Mussin et al. 2026): suture particles "may slide along the needle path but cannot move away in orthogonal directions". Friction: not stated. How the path follows the deforming tissue: not stated (my inference: it must be attached barycentrically). CRESSim-Neo 2026 reuses this method as "path-following constraints".

**A.5.3 Xu and Liu, Int J CARS 2018.** PBD Cosserat thread; "the direct solution of the distance constraint based on tridiagonal matrix algorithm is used to enhance stretching resistance in every constraint projection iteration." Thread to tissue by collision only. Reusable: a direct line solve for inextensibility along a chain, again the idea of issue #4794.

**A.5.4 Needle embedding in PBD and XPBD systems.** iMSTK (Kitware): "Once punctured the embedding constraints kick in and constraint the needle to the original puncture locations" "via the initial barycentric coordinates when it entered that triangle"; a puncture record holds "the ids of the puncturable, ids of the cell punctured, and barycentric weights"; `PBDStaticSuture` embeds an arc-shaped needle. Westergaard et al., Computers and Graphics 2025/2026: XPBD tissue and thread; the nearest tissue particle is attached to the tip; "the haptic model does not incorporate frictional forces". Vantari patent US 12,676,083 (2026): per-step transient distance constraints between needle-shaft particles and nearby tissue particles, viscous friction only. VirtaMed patent US 2022/0151701 [verified, Google Patents]: the Cosserat thread "will then be constrained to pass through the suture fixation point within the tissue" and "slide through it if the user pulls from the needle"; the fixation is "parametrized using barycentric coordinates". CRESSim-MPM (Ou and Tavakoli 2025): a rigid needle moves inside MPM tissue with "only the tangent component is kept". Mussin, Ou, Tavakoli, BioRob 2026: PBD suture and MPM tissue with contact coupling. Reusable: the VirtaMed and iMSTK designs are the closest published specifications of a barycentric tunnel constraint; none reports Coulomb stick-slip along the path.

**A.5.5 VBD lineage and related rod work.** VBD (Chen, Liu, Yang, Yuksel 2024): "Since our method directly manipulates the position of each vertex, managing a constraint on a vertex becomes straightforward." Rods were "linear springs for elastic rods." AVBD (Giles, Diaz, Yuksel 2025): augmented Lagrangian rows for infinite stiffness, `E = 1/2 k C^2 + lambda C`, dual update `lambda += k C`. OGC (Chen et al. 2025) and DAT (Chen, Hsu, Ayman, Macklin 2026) give the penetration-free truncation; DAT's planar variant "enables smooth layer separation and tangential motion". Stable Cosserat Rods (Hsu, Wang, Wu, Yuksel, SIGGRAPH 2025) [verified, PDF]: "Our key contribution is a split position and rotation optimization scheme with a closed-form Gauss-Seidel quasi-static orientation update." and "VBD fails to converge to the correct result even with 8 iterations, as it cannot enforce the unit quaternion constraint." This limit applies to a particle-plus-quaternion rod; Newton's rods are rigid segments with full orientation blocks, so it does not apply directly (my inference). Sueda, Jones, Levin, Pai, SIGGRAPH 2011 [verified, UBC page], "Large-scale dynamic simulation of highly constrained strands": Eulerian-on-Lagrangian strands, where each node carries an extra Eulerian sliding coordinate; the basis of Option 3 in `options.md`. No paper combines VBD with needle insertion or suturing (three agents searched; none found one).

### A.6 Summary table

| Work | Rod to tissue coupling | Tunnel or puncture | Friction and sliding | What transfers to VBD |
| --- | --- | --- | --- | --- |
| Chentanez 2009 | remesh nodes onto path, Lagrange multipliers | node chain; tip cutting force | per-node stick-slip, sliding only | stick-slip state, cutting-force gate |
| DiMaio 2003/2005 | boundary-condition switching, condensed FEM | threshold (not stated) | force profile along shaft | free / stuck / sliding node states |
| Goksel 2005-2011 | node repositioning or addition | remesh per element entered | stick-slip decision | repositioning in material space |
| Duriez 2009, Guebert | complementarity constraints, no remesh | penetration threshold f_p, cutting | static and dynamic Coulomb | the constraint set |
| Perrusi 2021, Martin 2023 | constraints at needle-edge and tet-face intersections, barycentric | f_p at surface, f_c cutting | mu_p surface, mu_i inside; 2 bilateral + 1 friction row | row layout, barycentric transfer |
| Lenoir 2004 | sliding point constraints on a spline | piercing points | stick or slide at the point | the sliding point primitive |
| Berkley 2004 | prescribed displacement at contact point | orientation-weighted force threshold | simple sliding rules; pulley anchor | puncture rule, pulley |
| BeamAdapter sliding | 2 transverse rows, tangent free, forces to 2 end frames | n/a | n/a | row layout, abscissa projection |
| SOFA Cosserat needle | barycentric points created at the tip; y,z rows | contact-force threshold; contact deactivated | FrictionContactConstraint | complete recipe for Option 1 |
| InfinyTech3D | proximity pairs shaft to tets | punctureForceThreshold; phases | friction coefficient | phase machine, shaft-collision switch |
| Yu 2020, CRESSim-Neo | PBD path constraints | threshold; stored path | not stated | path as barycentric chain |
| iMSTK, VirtaMed | barycentric embedding at puncture | at puncture | iMSTK mu 0.1; VirtaMed generic | barycentric tunnel spec |
| Stable Cosserat Rods, Deul 2018, Xu 2018 | rod solvers | n/a | spring or projection friction | split solves, direct chain solve (#4794) |

### A.7 Verification status

I checked these against the primary source myself: Duriez 2009 (PubMed E-utilities), Martin 2023 (Eurographics PDF), Yu 2020 (Semantic Scholar), Lenoir 2004 (HAL API), Berkley 2004 (author PDF), Stable Cosserat Rods 2025 (PDF), VirtaMed patent (Google Patents), Sueda 2011 (UBC page). The agents report their sources per entry in the raw files. Springer, IEEE Xplore, HAL HTML pages and the Eurographics library blocked automated reads, so several entries rest on abstracts from PubMed, HAL, Crossref or Semantic Scholar. The agent corrected two venues: Payandeh and Shi 2010 is in Virtual Reality, not The Visual Computer; the Lenoir catheter MMVR paper is 2006, not 2004.

## Part B. What Newton has today

Worktree: `/home/horde/Code/Graphics/newton-working-copies/vbd-suture-study`, detached at `upstream/main` 84242814c (2026-11-05). All paths below are relative to the repository root. Line numbers refer to that commit.

### B.1 Step structure of SolverVBD

- `newton/_src/solvers/vbd/solver_vbd.py:2898` `step()`. Order per substep: pre-step collision detection, `_initialize_rigid_bodies`, `_initialize_particles`, attachment set-up (line 2977), a detection pass (2983). Then for each iteration (2993): detection if due (2996), `_solve_rigid_body_iteration` (3002), `_solve_particle_iteration` (3003), attachment dual update (3006). Then finalize (3047).
- `solver_vbd.py:4280` `_solve_particle_iteration`. For each color group: particle-body attachments, rigid-soft contact (gather or scatter kernel), springs, self-contact, then `solve_elasticity`, then `_penetration_free_truncation` (4523).
- `newton/_src/solvers/vbd/particle_vbd_kernels.py:2948` `solve_elasticity`. One thread per vertex of the color. It forms inertia force and Hessian, adds the elasticity terms, adds the pre-accumulated `particle_forces` and `particle_hessians`, and applies one 3x3 solve: `displacement += h_inv * f` (line 3030). Any new per-vertex term must write its force and 3x3 Hessian into those two arrays for the vertices of the current color before this kernel runs.
- Coloring: `newton/_src/sim/builder.py:13124` `color()` builds the particle graph with `construct_particle_graph` (`newton/_src/sim/graph_coloring.py:202`) from triangles, optional bending edges, and tets. Springs are not in the graph. The spring kernel (`particle_vbd_kernels.py:1799`) adds forces only to endpoints of the current color, so two same-colored endpoints update in Jacobi style. Particle-body attachments and rigid-soft contacts need no particle graph edge, because the body side is solved in the separate rigid pass.

### B.2 Soft-body tet FEM

- Neo-Hookean tet energy: `particle_vbd_kernels.py:184` `evaluate_volumetric_neo_hookean_force_and_hessian`. Per-vertex assembly: `particle_vbd_kernels.py:1853` `accumulate_elasticity_force_and_hessian`. Tiled variant: `particle_vbd_kernels.py:2729`.
- Authoring: `builder.py:12441` `add_soft_mesh`, `builder.py:12256` `add_soft_grid`. Model arrays `tet_indices`, `tet_poses`, `tet_materials` (`newton/_src/sim/model.py:1331-1337`).
- Example: `newton/examples/softbody/example_softbody_franka.py:109-119` (tet mesh, density 100, k_mu = k_lambda = 1e6, particle radius 5 mm) with `SolverVBD` at lines 159-163.

### B.3 Rods and cables

- `newton/_src/sim/rod.py:112` `class Rod`: centerline points, edges, one material frame per segment, optional radius and material. Points may form any curve, so a curved needle centerline is expressible.
- `builder.py:10481` `add_rod`: "Adds a rod composed of capsule bodies connected by rod joints." Each segment is a capsule rigid body. Incident segments connect by ROD joints with stretch, shear, bend, and twist response. `builder.py:7006` `add_joint_rod`.
- Solver side: ROD joints are four compliant ALM rows. Stretch and shear: `newton/_src/solvers/vbd/rigid_vbd_kernels.py:1802`. Bend and twist: `rigid_vbd_kernels.py:1584`. Dahl hysteresis for angular response: `rigid_vbd_kernels.py:5157`. Each body solves one 6x6 system per color (`docs/solvers/vbd.rst:441-567`).
- There is no particle-based rod in VBD. Bending exists only as the dihedral angle between two triangles (`particle_vbd_kernels.py:700`). Particle springs exist (`particle_vbd_kernels.py:1709`) and carry no bending.
- GitHub issue #4794 (open, 2026-10-19, by Codnay): "[REQ] SolverVBD: opt-in direct solve for stiff, slender rods". It proposes a block tree elimination per connected ROD chain inside each rigid iteration. A stiff needle-plus-thread chain would benefit. No PR yet.
- Examples: `newton/examples/cable/` (7 examples, pulleys and contact) and `newton/examples/vbd/example_cable_*.py` (7 examples, bend, twist, Dahl).

### B.4 Rigid-soft contact

- Detection: `newton/_src/geometry/kernels.py:1184` `create_soft_contacts`. One thread per (particle, shape) pair. It skips the pair if the particle is not ACTIVE or the shape lacks `COLLIDE_PARTICLES`. It evaluates the shape SDF at the particle (capsule at line 1289) and emits a record when `d < margin + s_margin + radius` (line 1353). Each record stores the shape, the closest surface point in body frame, and the normal.
- Full-surface option: `newton/_src/geometry/soft_contacts_mesh.py:1-15`. Vertex-face, face-vertex, and edge-edge pairs for mesh shapes. Only SolverVBD consumes them (`newton/_src/sim/contacts.py:703`).
- Force law: `rigid_vbd_kernels.py:2140` `_compute_body_particle_contact_force` = normal penalty or log-barrier (`rigid_vbd_kernels.py:2108`) + damping on approach + `compute_projected_isotropic_friction` (`rigid_vbd_kernels.py:2426`). Geometry resolution: `rigid_vbd_kernels.py:2177` `_eval_body_particle_contact`.
- Particle side per color: `particle_vbd_kernels.py:2467` `gather_particle_body_contact_force_and_hessian` (deterministic, one thread per particle) or `particle_vbd_kernels.py:2598` scatter variant. Body side per body: `rigid_vbd_kernels.py:5945` `accumulate_body_particle_contacts_per_body`.
- Dynamic filter: `soft_contacts_mesh.py:1593` `_soft_contact_force_enabled`. Analytic SDF rows always produce force. Mesh rows need a resolved sign, at least one ACTIVE particle, and `mesh_contact_valid`.
- Flags: `newton/_src/geometry/flags.py:8` `ParticleFlags` has only ACTIVE and PROXY. `ShapeFlags.COLLIDE_PARTICLES` (`flags.py:36`) is per shape and static. There is no per-pair or per-region contact gate today.
- My inference: interior tets have no surface triangles. A thin needle inside the body therefore meets only the vertices within one radius plus margin of its axis, plus the surface faces at entry and exit. Contact alone cannot hold a needle on a path inside the body. A tunnel constraint must do that.

### B.5 Divide and Truncate (DAT)

- Particle side: planar truncation per detected pair `particle_vbd_kernels.py:2278` `apply_planar_truncation_parallel_by_collision`, radius truncation `particle_vbd_kernels.py:2416` `apply_truncation_ts`. Driver `solver_vbd.py:3605` `_penetration_free_truncation`, called after every color (4523).
- Rigid-soft DAT: constructor flag `rigid_soft_enable_dat` (`solver_vbd.py:491`, default False). `rigid_vbd_kernels.py:8252` `apply_rigid_soft_truncation` builds one division plane per contact record from the detection-time configuration and truncates both the soft vertices (straight rays) and the rigid primitive (curved trajectories). `rigid_vbd_kernels.py:8503` `apply_body_truncation_ts` scales the whole pose update of a body by its smallest factor. Driver `solver_vbd.py:3519`.
- Docs `docs/solvers/vbd.rst:825-886`: "A pair for which no separating plane can be certified freezes." "Kinematic bodies move outside the solver and are not truncated." The motion budget is about one half of the relaxation factor times the query radius per detection cycle.
- Consequence for suturing: a DAT-protected pair can never penetrate. The needle tip must be excluded from DAT against the tissue it punctures, or the puncture is impossible. The needle shaft, the thread, and the tissue away from the tunnel can keep DAT.
- Example: `newton/examples/multiphysics/example_vbd_dat_rigid_soft.py:144-157`.

### B.6 Particle-body attachments (closest existing template)

- Authoring: `builder.py:11237` `add_attachment_particle_body(particle, body, body_point, stiffness, damping, enabled)`. `math.inf` stiffness gives a hard point constraint. Model arrays `model.py:1282-1294`.
- Solver: compliant ALM row. Once per step `particle_vbd_kernels.py:1964` `step_particle_body_attachment_lambda_rho` sets the metric rho from both endpoints' support (particle inverse mass reduced by its material Hessian, plus body inverse mass) and retains lambda. Per color on the particle side: `particle_vbd_kernels.py:2070` calls `rigid_vbd_kernels.py:1734` `evaluate_body_particle_attachment_particle_force_hessian`: `C = x_p - anchor`, force `-(k_eff C + lambda_eff)`, Hessian `k_eff I`, optional damping. Per body on the rigid side: `rigid_vbd_kernels.py:5866` `accumulate_body_particle_attachments_per_body` converts the point force to a body wrench and 6x6 Hessian blocks. Duals after both passes: `rigid_vbd_kernels.py:1767` `update_duals_particle_body_attachments`, launched at `solver_vbd.py:3006`. CSR adjacency body-to-attachments: `solver_vbd.py:2708`.
- Limits (`solver_vbd.py:305-315`): translational only; both endpoints must be integrated by SolverVBD; finite iterations can leave residual error.
- Reusable: the full pipeline (rho, lambda, particle side per color, rigid side per body, dual update) is the template for a sliding point-on-curve constraint. What changes is the residual `C` (remove the tangent component) and the soft anchor (a barycentric point inside a tet instead of one particle).

### B.7 Soft contact friction

- `particle_vbd_kernels.py:1353` `compute_friction`: IPC-style smoothed Coulomb friction with a constant normal, transition width `friction_epsilon * dt`. `rigid_vbd_kernels.py:2426` `compute_projected_isotropic_friction` is the rigid-soft counterpart. Both take a normal force magnitude and a relative displacement, so they can serve tunnel friction with the constraint force as the normal force and the tunnel tangent as the sliding direction.

### B.8 Tendons

- `newton/_src/sim/tendon.py:28` `TendonGuideType`: ROLLER, ANCHOR, PINHOLE. `newton/_src/solvers/vbd/tendon_kernels.py` holds the VBD kernels. A tendon is a length row between guides on rigid bodies and slides through the guides. It has no particle endpoints. It is the mirror image of a suture: here the path is attached to rigid bodies, in a suture the path is attached to the soft body.

### B.9 XFEM cutting branch (read-only, other agent)

- Branch `ankac/vbd-xfem-cut`, head 68a571b85, 37 files, +8779/-264 lines. Task folder `/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/vbd-xfem-surgical-cut/design.md`.
- `newton/_src/solvers/vbd/tet_cutting.py:27` `VBDTetCutting`: a companion that owns a copy of the model, state, and solver. `cut()` (line 282) takes a finite blade sweep. `_rebuild` (361) duplicates DOFs by phantom nodes, rebuilds adjacency and coloring on the host. Cut events cannot run inside a CUDA graph. `newton/_src/solvers/vbd/xfem_geometry.py:330` `build_dof_map`, `394` `extract_surfaces`.
- `newton/_src/solvers/vbd/particle_contact_kernels.py:4`: "Pull native triangle-mesh contact terms back through a convex embedding." `_scatter_embedded_contact` (line 52) distributes the force and Hessian of a contact on an embedded surface point to the simulation DOFs with weight `w` for the force and `w*w` for the Hessian, honoring colors. `build_embedded_body_contact_adjacency` (94), `gather_embedded_body_contact_force_and_hessian` (120), `apply_embedded_planar_truncation` (487).
- Limits that matter here: particle attachments are unsupported with the companion (design.md, "Current architecture and scope"); rigid-soft DAT is rejected with an embedded contact mesh (branch `solver_vbd.py:1056`); the full cut, separate, and release sequence does not pass yet (design.md, table of final experiments).
- Reusable: the embedding scatter is the exact mechanism a barycentric tunnel anchor needs. The host-rebuild companion is the cost model for a real tunnel by local remeshing.

### B.10 GitHub search

- `gh search issues` and a GraphQL discussion search on `newton-physics/newton` for suture, suturing, needle, sewing, stitch, catheter, guidewire, needle insertion, thread, surgical (2026-11-05): no matching issue, discussion, or PR. The only related item is #4794 (stiff slender rod direct solve).

### B.11 Gap table

| Feature from the request | State in Newton | Where |
| --- | --- | --- |
| Curved needle as a rigid body | exists | capsule or mesh shapes on one body; `Rod` for a curved chain |
| Thread attached to the needle and following it | exists | `add_rod` thread + ROD or FIXED joint to the needle body |
| Needle and thread collide with the body (no penetration) | exists | rigid-soft contact B.4, DAT B.5 |
| Only the tip can puncture | missing | no per-pair contact gate; DAT forbids crossing |
| Tunnel between entry and exit (virtual) | missing | template: attachments B.6 + embedded scatter B.9 |
| Tunnel (real geometry change) | missing | cost model: XFEM companion host rebuild B.9 |
| Sliding along the tunnel with friction | missing | friction law B.7 reusable |
| Exit through the surface again; thread stays in the tunnel | missing | needs a puncture state machine |
| Pull the thread to close the wound | follows from the tunnel | thread tension through ROD stretch rows |

## Part C. What we can borrow, and what we cannot

1. **Tunnel as a chain of barycentric points created at the tip.** SOFA Cosserat plugin (A.4.3), Yu et al. 2020 (A.5.2), iMSTK and VirtaMed (A.5.4), Lenoir's sliding point (A.2.4). Newton has the embedding scatter pattern on the XFEM branch (B.9) and tet arrays (B.2).
2. **Constraint rows: two transverse, tangent free.** BeamAdapter (A.4.1), Martin 2023 (A.2.3), SOFA SlidingLagrangianConstraint (A.4.2). Newton's compliant ALM attachment row (B.6) is the template; only the residual projector and the anchor change.
3. **Puncture by a force threshold with an orientation test.** Berkley 2004 (A.3.1), Cosserat plugin and InfinyTech3D (A.4.3, A.4.4), Yu 2020 (A.5.2), Duriez's `f_p` (A.2.1). Newton exposes rigid-soft contact forces per record (B.4, observables in the SolverVBD docstring).
4. **Contact off between the needle group and the tunnel region after puncture.** Cosserat plugin deactivates the contact constraint; InfinyTech3D has `enableShaftCollision`. Newton has no per-pair gate today (B.4); DAT would otherwise forbid the crossing (B.5).
5. **Coulomb stick-slip per anchor along the tangent.** Duriez 2009 and Perrusi 2021 (`mu_p`, `mu_i`), Chentanez per unit length. Newton's `compute_friction` (B.7) takes a normal force and a tangential displacement, so it fits with the constraint force plus a grip force as the normal force.
6. **Hard rows through augmented Lagrangian duals.** AVBD (A.5.5) is already Newton's rigid path (B.1, B.6).
7. **A direct solve along a stiff chain.** Deul 2018, Xu and Liu 2018 (A.5.1, A.5.3) and Newton issue #4794 (B.3). Useful if a needle-plus-thread chain converges too slowly under per-body descent.
8. **Not transferable.** The dual Gauss-Seidel on a compliance matrix (A.2), conforming remeshing with Lagrange LCP (A.1.1), Woodbury or condensation updates (A.1.2, A.1.3), and PBD projection order. VBD has no global matrix and keeps per-vertex and per-body blocks.
9. **Gap in the literature.** No published work combines VBD with needle insertion or suturing (A.5.5). The closest PBD systems have no Coulomb friction along the tunnel (A.5.4).
