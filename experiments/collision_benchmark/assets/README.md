# Collision benchmark assets

The PNG files show static rest geometry. They are not simulation results. A smoke scene can rescale, rotate, or deform an asset. Use that scene's metadata for the simulated dimensions and material settings.

The original library has ten cloth assets: `patch`, `towel`, `sheet`, `trousers`, `dress`, `tshirt`, `skirt`, `tote`, `pouch`, and `gusset_bag`. It has six solid assets: `block`, `pad`, `sphere`, `capsule`, `torus`, and `branched`. All positions use metres and Z up. Original assets have their lowest vertex at Z=0.

Garments use two original cut panels. Shared vertices sew the side, shoulder, and inner-leg seams. Waist, cuff, neck, armhole, and hem edges remain open as applicable. The rest panels have a small declared separation to avoid an initial double layer at the same location. This is a prescribed rest shape, not a simulated drape. Bags use shared panel vertices, closed bottoms, open rims, and attached handle strips. The gusset bag has concave side folds. The pouch has no handle.

Soft solids have conforming tetrahedra. The sphere comes from a radial cube-to-sphere map. The capsule is an axially stretched rounded surrogate, not an exact analytic spherocylinder. The torus is a periodic swept disk. The branched solid is an original four-arm cross with rectangular arms. It is not an octopus or a TetWild dataset object.

The library supports `coarse`, `medium`, and `fine` resolutions. The exported files use `medium`. Resolution changes mesh spacing. It does not change object dimensions or contact radius. Coarse meshes have 81–824 particles for cloth and 60–315 particles for solids. Medium and fine meshes support geometry studies; their contact behavior still needs simulation validation.

```python
import newton
from experiments.collision_benchmark.assets import (
    add_cloth_asset,
    add_solid_asset,
    cloth_mesh,
    solid_mesh,
    validate_asset,
)

mesh = cloth_mesh("trousers", resolution="coarse")
assert validate_asset(mesh)["valid"]
# mesh.vertices: float32 [N, 3], metres
# mesh.triangles: int32 [M, 3]
# mesh.tetrahedra: int32 [K, 4] for solid_mesh(), else None
# mesh.landmarks: local indices for top, bottom, sides, seams, rims, handles
# mesh.metadata: dimensions, geometry hash, generator settings, synthetic preset

builder = newton.ModelBuilder()
cloth = add_cloth_asset(builder, "towel", resolution="coarse", pos=(0, 0, 0.5))
solid = add_solid_asset(builder, "sphere", resolution="coarse", pos=(0.2, 0, 0.1))
# cloth.particle_start: inclusive global particle index
# cloth.particle_end: exclusive global particle index
# cloth.landmark_indices: global particle indices in this builder
```

The add helpers accept `pos`, `rot`, `scale`, `density`, and `particle_radius`. Extra keyword arguments pass to the Newton cloth or soft-mesh builder. They set material constants, velocity, and display color. They do not pin particles. Cloth density is kg/m²; solid density is kg/m³. Cloth uses Newton's triangle and bending coefficients. Solids use Lamé coefficients in Pa and viscous damping in Pa·s. The defaults are synthetic presets, not measured fabric or rubber. Physical shell thickness is not calibrated; the declared particle radius is the rigid-contact radius and must not be described as a measured thickness. VBD self-contact distances belong to the scene configuration.

`validate_asset()` checks finite coordinates, positive triangle area, one connected component, manifold surface edges, consistent face winding, and positive tet volume. Revision 2 coarse sewn garments have minimum triangle angles above 26 degrees, with no angles below 10 degrees. All 48 generated type/resolution combinations pass. All sixteen coarse assets load into Newton with positive particle masses. These checks do not prove a stable grasp, self-intersection freedom under motion, mesh convergence, or material accuracy.

```bash
uv run --no-sync python -m experiments.collision_benchmark.assets \
  --output experiments/collision_benchmark/outputs/asset-previews --cached-previews
```

The original generator needs NumPy and the existing SciPy dependency. Revision 2 uses a triangular lattice and Delaunay triangulation for cut panels. It checks that all panel boundary segments are preserved. This replaces the thin triangles in revision 1; the previous exports remain in the separate `assets-v1` archive. Newton and Warp are loaded only by builder helpers. Static previews also use Matplotlib. Cached previews use USD and trimesh packages. The explicit `--cached-previews` option can download pinned source assets. Ordinary cache inventory does not download them. See the parent [README](../README.md) for installation.

The shipped metadata retains the original export paths as historical provenance. Runtime loaders resolve assets through pinned downloads and package-relative files; those old machine paths are not required. Write new exports to a separate output directory to preserve the original geometry records.

`manifest.json` records original geometry hashes, dimensions, mesh counts, landmarks, generator parameters, material defaults, and validation. Each NPZ contains geometry arrays. Each OBJ contains surface geometry. A solid OBJ does not contain its tetrahedra. Each asset also has a separate `*.metadata.json` record. The manifest's generator SHA-256 hashes `assets.py`; each geometry SHA-256 hashes little-endian vertex/triangle/tet arrays.

The cached inventory contains Franka FR3 with its hand, Robotiq 2F-85, Allegro, LEAP, Shadow, the Newton rubber duck, and five Style3D garments including `h1_jacket`. Full Git revisions, all source-file hashes, a source-tree hash, and license notices are retained in the manifest. Robot previews use their imported rest joint configuration and forward kinematics only. Style3D coordinates and stage units are inspected; transfer of its seam and material models into VBD is not claimed.

Original geometry and the generator use Apache-2.0. The included `LICENSE.md` contains that license. Cached Franka, Shadow, and duck assets use Apache-2.0; Allegro and Robotiq use BSD-2-Clause; LEAP uses MIT; Style3D uses BSD-3-Clause. Source licenses and notices remain in each cached record. No source geometry is copied into the original NPZ or OBJ files.

The local LIDO TetWild/Thingi10K candidate subset, `bunny.vtk`, and `octopus.vtk` remain excluded. Their individual semantic identities or source reuse terms have not been established. A mesher's license does not establish a mesh's reuse terms.

The separate `jacket_scene.py` prepares and loads a BSD-3-Clause derivative of the cached `h1_jacket`. The 830-vertex / 1,599-triangle mesh keeps one component, three boundary loops, and Euler characteristic -1. The source notice is in `STYLE3D-LICENSE.txt`. Its `cached_jacket_smoke.metadata.json` records source hashes, decimation settings, and the remaining thin triangles. The original acquisition attempt and one corrected free-settle / side-grip attempt both completed with finite states, but neither established a held lift. Their videos and metrics remain visible as exploratory failures. No deformable particle was pinned.
