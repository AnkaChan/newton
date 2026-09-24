I found that transformers are very good at learning dynamics.
I want to create a equivariant solver that generates to the entire Euclidean space.

Idea:

- drop the orignal mesh, sample the object using grid (galerkin grid, avoid gluing close but separate things), each grid cell represented as nodes
- graph transformers architecture: each node see its topological neighbors (how many rings?)
- equivariant inputs: instrinsic measurements of how elements locally deforms; size? material?
- equivariant outputs: instrinsic measurements of how elements locally deforms
- local-global deformation assemble: ARAP-like local grobal fusing (maybe called as strechy as possible)
- global rotation-translation: solve from a global semi-implicit rigid step
