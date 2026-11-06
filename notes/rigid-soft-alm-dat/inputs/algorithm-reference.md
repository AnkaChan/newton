# ALM-DAT step pseudocode (Anka's reference of 2026-10-14, minor fixes applied)

```text
# x_start: penetration-free start state of the step (the detection pose); y: inertial target
# contacts: compact global buffer from the triangle-mesh collision detector, arranged by the
#           vertex side's id; each vertex's rows sorted by the three triangle indices (easy matching)
contacts = InitialSphericalQuery(x_start, motion_budgets)

# p, n share the layout of contacts: point and normal of every division plane
p, n = ComputeContactConstraintsAndNormals(contacts, x_start)

# Initial stage
x_anchor = x_start
x_trial = y
x_anchor = DAT_truncate(x_anchor, x_trial, p, n)          # motion budget measured from x_start
p, n = ComputeContactConstraintsAndNormals(contacts, x_anchor)

# multipliers: zero, or carried from the previous step by matching the sorted rows
planes_lambdas = zeros(contact_count)
if contacts_history:
    planes_lambdas = retrieve_lambda_from_history(contacts, contacts_history, lambda_history)

# Stage 1: ALM/VBD sweeps on the trial state (the trial may penetrate)
for iteration in range(FIXED_ALM_ITERATIONS):
    x_trial = ALM_VBDSweep(x_trial, p, n, planes_lambdas)
    if iteration % PLANE_UPDATE_INTERVAL == 0:
        x_anchor = DAT_truncate(x_anchor, x_trial, p, n)
        p, n = ComputeContactConstraintsAndNormals(contacts, x_anchor)
    # the multipliers learn from the untruncated trial state
    planes_lambdas = UpdateALMMultipliers(x_trial, p, n, planes_lambdas)

# Stage 2: iterative truncation
# x_anchor is penetration-free, x_trial may penetrate; p, n are valid at x_anchor
for iteration in range(FIXED_TRUNCATION_ITERATIONS):
    # truncate every particle toward x_trial against its planes (motion budget measured from x_start)
    x_anchor = DAT_truncate(x_anchor, x_trial, p, n)
    # slide each plane along its normal: sides that still want to move get the room
    # (computed from the truncated x_anchor and the remaining displacement x_trial - x_anchor)
    p = SlidePlanes(contacts, x_anchor, x_trial, p, n)
    # occasionally re-pick the normals, at the penetration-free x_anchor
    if iteration % NORMAL_UPDATE_INTERVAL == 0:
        p, n = ComputeContactConstraintsAndNormals(contacts, x_anchor, x_trial)

x_out = x_anchor
v_out = (x_out - x_start) / dt
lambda_history, contacts_history = planes_lambdas, contacts

SlidePlanes(contacts, x_anchor, x_trial, p, n), for every pair c:
    # heights of the truncated positions, rates from the remaining displacement
    h_i = n . x_anchor_i                       for every vertex i of the pair
    r_i = x_trial_i - x_anchor_i
    a_i = approach rate toward the plane:      max(0, +n . r_i) on side A, max(0, -n . r_i) on side B

    # where the plane may be right now (both primitives stay separated with margin m)
    b_lo = max over A of h_i + m
    b_hi = min over B of h_j - m               # m shrunk to half the gap if the gap is thinner than 2m

    # who still wants room?
    if nobody approaches:        b = midpoint(b_lo, b_hi)
    elif only B approaches:      b = b_lo     # A needs nothing: give B everything
    elif only A approaches:      b = b_hi
    else:
        # balance the slowest vertex on each side: equal fraction of the remaining motion
        # tau_i(b) = (b - m - h_i) / a_i on A,  tau_j(b) = (h_j - m - b) / a_j on B
        for each approaching pair (i in A, j in B):       # 3 pairs for vertex-triangle, 4 for edge-edge
            b_ij = (a_j (h_i + m) + a_i (h_j - m)) / (a_i + a_j)    # where tau_i == tau_j
            clamp b_ij to [b_lo, b_hi]
            score_ij = min over all approaching vertices of tau(b_ij)
        b = the b_ij with the largest score

    p_c = p_c + (b - n . p_c) * n              # slide the stored point along the normal
```
