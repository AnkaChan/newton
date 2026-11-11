# Fast cut in SolverVBD

The words of the cut design for tetrahedral tissue in Newton's SolverVBD. One word per thing. The suture words (tunnel, tunnel point, puncture, contact gate) keep their meaning from the suture CONTEXT.md.

## Language

### The tool

**Blade**:
The dynamic rigid body that cuts the tissue.
_Avoid_: knife, scalpel, cutter

**Holder**:
The kinematic body that drives the blade through a joint with drives.
_Avoid_: handle, arm

**Blade edge**:
The round front of the blade, a capsule, that presses on the tissue.
_Avoid_: tip, cutting line

**Blade plane**:
The plane that holds the blade edge and the cutting direction.
_Avoid_: blade plate, cut plane (for the tool)

**Cutting direction**:
The direction, fixed in the blade body, from the blade body out through the blade edge.
_Avoid_: blade direction, feed direction

**Side A, side B**:
The two half spaces of the blade plane; side B is the side that the plane normal points to.
_Avoid_: left and right, plus and minus side

**Contact record**:
One rigid-soft contact that the collision pipeline found between a shape and the tissue.
_Avoid_: record (alone), contact (for the data entry)

**Blade force**:
The sum, over the contact records of the blade edge, of the normal force times the positive part along the cutting direction.
_Avoid_: edge force, tip force (that is the needle)

**Cut force**:
The tissue value that the blade force must reach before the tissue under the edge is cut.
_Avoid_: puncture force (that is the needle), cut threshold

**Cuttable**:
Said of a tissue particle with a finite cut force.

**Force range**:
The distance from the blade edge axis within which a contact record gives force.

**Detection range**:
The distance from the blade edge axis within which the collision pipeline makes a contact record; the force range plus the pipeline gap.

**Contact range**:
The distance from the blade edge within which a tissue vertex counts as pressed; the proposed value is the detection range.
_Avoid_: reach, gate radius (that is the needle)

**Pressed vertex**:
A cuttable tissue vertex within the contact range of the blade edge.

**Cut update**:
One run of the cut kernels at the end of a solver step.

**Cut advance**:
A cut update in which the blade force reached the cut force, so one tet row is cut.
_Avoid_: cut event, trigger

**Stroke**:
The time during which the blade edge has at least one contact with the tissue.
_Avoid_: armed period, pass

**Side label**:
The side of the blade plane that a tet was on when the stroke first needed it; kept until the stroke ends.

### The cut

**Free cut**:
A cut that can happen at any place in the body, along the existing tet faces nearest to the blade.
_Avoid_: arbitrary cut, XFEM cut

**Cut face**:
One interior tet face that a cut separates into two triangles, one on each side.
_Avoid_: split face, crack face

**Cut vertex**:
A tissue vertex on a cut face.

**Split vertex**:
A cut vertex that has at least one copy.

**Original**:
The particle that a split vertex had before the cut; it stays with one tet group.

**Wound face**:
All new triangles on one side of a cut.
_Avoid_: cut surface (for one side), lip

**Wound edge**:
A mesh edge that lies on a wound face.

**Wound lip**:
The split vertices and their copies along one side of a wound.
_Avoid_: rim, flap

**Contact gate**:
The suture word: one bit per tool shape and per tissue particle; where a shape and a particle share a bit, their contact is still detected but gives no force and no truncation plane.
_Avoid_: filter (that is the self-contact filter)

**Through cut**:
A cut that separates the whole cross-section, so the body falls into two parts.
_Avoid_: full cut, complete cut

**Partial cut**:
A cut that ends inside the tissue.
_Avoid_: half cut, incision

**Cut front**:
The line where a partial cut ends inside the tissue; vertices on it are not split.
_Avoid_: rim, crack tip, crack front

**Stepped cut**:
A cut whose wound faces follow tet faces that do not lie in one plane, because the blade is off the mesh planes.
_Avoid_: jagged cut, staircase

**Tet row**:
The layer of tets that one cut advance opens.
_Avoid_: row (alone; a row is a line of an index array)

**Lead of the cut**:
The distance, along the cutting direction, from the blade edge axis to the farthest particle of a face cut in one cut advance.

**Start gap**:
The distance between the two wound faces right after a split, made by moving each original and its copy apart.
_Avoid_: birth gap, birth offset, kerf

**Opening**:
How far a start gap direction moves a new particle away from one cut face; positive means away.

**Crossed pair**:
A cut face and one of its split vertices whose opening is zero or less, so the particle starts on the wrong side of that face.

**Stage 1**:
The first implementation stage: a flat through cut on a grid plane, a 4 mm pull, three stitches, release.

### The data

**Spare particle**:
A particle that exists from the start, does nothing, and waits on the parking lattice until a cut makes it a copy.
_Avoid_: reserve particle, phantom particle

**Copy**:
A spare particle that took the place of a cut vertex for the tets on one side.
_Avoid_: duplicate, twin, phantom node, virtual node

**Pool**:
The spare particles of one copy color.

**Copy color**:
The second color group of an original color; the copies of vertices with that color live there.
_Avoid_: twin color, shadow color

**Parking particle**:
A particle that never becomes a copy; an unused surface row or the idle tet points at it.

**Parking lattice**:
The set of places away from the tissue and from every tool path where spare particles and parking particles sit.
_Avoid_: graveyard, pool position

**Spare row**:
An unused triangle row or edge row of the collision surface.

**Idle tet**:
The one tet with zero stiffness that unused tet records point at.
_Avoid_: null tet, dummy tet

**Tet record**:
One entry (tet, corner) of the vertex-to-tet table.
_Avoid_: record (alone; a contact record is another thing)

**Tet group**:
Tets around one vertex that are still joined through uncut faces.

**Fan**:
The run of tets around one mesh edge that are joined through uncut faces.
_Avoid_: wedge, ring

## Relationships

- A **Holder** drives one **Blade**; a **Blade** has one **Blade edge** and one **Blade plane**
- When the **Blade force** reaches the **Cut force**, the **Cut update** is a **Cut advance**: the faces at the **Pressed vertices** whose two tets have different **Side labels** become **Cut faces**, one **Tet row**
- A **Free cut** is a set of **Cut faces**; a **Cut face** gives one triangle to each of the two **Wound faces**
- A **Split vertex** keeps its **Original** for one **Tet group** and gets one **Copy** for each other **Tet group**
- A **Copy** comes from the **Pool** of the **Copy color** of its **Original**
- An **Original** and its **Copy** start one **Start gap** apart
- A **Cut vertex** on the **Cut front** has one **Tet group** and is not a **Split vertex**
- A **Partial cut** has a **Cut front**; a **Through cut** has none
- A wound has two **Wound lips**; in run P0 the **Blade edge** rests on them and does not reach the **Cut front**
- Each **Fan** at a **Wound edge** has one edge row

## Flagged ambiguities

- "copy" was also used for a duplicated tet (the XFEM form) and for saved arrays. Resolved: **Copy** means a particle only. A duplicated tet is a "tet copy"; saved arrays are "saved arrays".
- "row" was used for a layer of tets and for a line of an index array. Resolved: **Tet row** for the layer; "triangle row" and "edge row" for the arrays.
- "record" was used for a contact record and for an entry of the vertex-to-tet table. Resolved: "contact record" and **Tet record**.
- "Blade" was first written as a kinematic body. Resolved by D7: the **Blade** is dynamic; the **Holder** is kinematic.
- "armed" in the options meant a host switch for the cut period. Resolved by D6: `cut_armed` means that the first contact with force cuts, as `suture_armed`.
