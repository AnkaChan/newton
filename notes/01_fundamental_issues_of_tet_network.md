I think there is a fundamental issues of the current method.
Take a long beam for example. When one side of the beam is pressed, not only the tets that are in touch will be compressed. But also all the tets along the beam will be touched (in a implicit integratio step). In real world this deformation will propagate along the material in the speed of sound in the material.

However the current way of implementation, a tet has no idea whether a tet on the other side is compressed. The same input for it, can have a variety of expected outputs depends on the conditions of the other side, which it does not see. This is fundanmentally a ill defined problem.
The global step, though capable of combining all the local information, cannot control how each local states is computated, and as far as I can tell the global shape is not fed back to the local networks. Therefore it does not solve this fundanmental problem.

I have the following ideas:
1. Multi-res problem. Use a octree or VDB data structure to decompose the object into a hariechy of coarse to fine grids. Each node predict a deformation. the child node's deformation will ** be added on top of ** parent's deformation prediction.
2. Graph neural networks. Maybe connect those nodes using graph networks. to achieve across node/hierachy communication.

Those ideas can be combined.
