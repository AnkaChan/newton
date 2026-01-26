# Blender Python script to load crusher simulation initial meshes from Newton
# Run this script in Blender's Script Editor or via command line:
#   blender --python blender_load_crusher_output.py

import bpy
import os
import numpy as np
from mathutils import Vector

# ============================================================================
# CONFIGURATION - Set this to your simulation output folder path
# ============================================================================
OUTPUT_DIR = r"D:\Data\DAT_Sim\crusher\2026-01-19_18-27-36"  # Update this!
SCALE = 1.0  # Simulation is already in meters
# ============================================================================


def create_mesh_from_arrays(name, vertices, faces, scale=1.0):
    """Create a Blender mesh from vertex and face arrays."""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    bpy.context.collection.objects.link(obj)
    
    scaled_vertices = (vertices * scale).tolist()
    mesh.from_pydata(scaled_vertices, [], faces.tolist())
    mesh.update()
    
    return obj


def set_material(obj, color):
    """Create and assign a material with given color to object."""
    mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
    mat.use_nodes = True
    
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
    
    obj.data.materials.append(mat)


def transform_vertices(verts, pos, quat):
    """Transform vertices by quaternion rotation and translation."""
    qx, qy, qz, qw = quat
    
    rot_mat = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    
    return (verts @ rot_mat.T + pos).astype(np.float32)


def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Directory not found: {OUTPUT_DIR}")
        return
    
    print(f"Loading from: {OUTPUT_DIR}")
    
    # Load mesh data
    roller_verts = np.load(os.path.join(OUTPUT_DIR, "roller_verts.npy"))
    roller_faces = np.load(os.path.join(OUTPUT_DIR, "roller_faces.npy"))
    tri_faces = np.load(os.path.join(OUTPUT_DIR, "tri_faces.npy"))
    particles = np.load(os.path.join(OUTPUT_DIR, "particles_initial.npy"))
    body_q = np.load(os.path.join(OUTPUT_DIR, "body_q_initial.npy"))
    
    # Create collection
    collection = bpy.data.collections.new("Crusher")
    bpy.context.scene.collection.children.link(collection)
    
    # Create soft body
    softbody = create_mesh_from_arrays("SoftBody", particles, tri_faces, SCALE)
    bpy.context.collection.objects.unlink(softbody)
    collection.objects.link(softbody)
    set_material(softbody, (0.8, 0.5, 0.3))
    print(f"  Created SoftBody: {len(particles)} verts")
    
    # Create rollers
    for i in range(2):
        pos = body_q[i, :3]
        quat = body_q[i, 3:7]
        roller_world = transform_vertices(roller_verts, pos, quat)
        
        roller = create_mesh_from_arrays(f"Roller_{i+1}", roller_world, roller_faces, SCALE)
        bpy.context.collection.objects.unlink(roller)
        collection.objects.link(roller)
        set_material(roller, (0.5, 0.5, 0.55))
        print(f"  Created Roller_{i+1}")
    
    print("Done!")


if __name__ == "__main__":
    main()
