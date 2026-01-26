# Blender Python script to load initial meshes from Newton simulation
# Run this script in Blender's Script Editor or via command line:
#   blender --python blender_load_initial_meshes.py
#
# Before running, set MESH_DIR to your simulation output folder's initial_meshes path

import bpy
import json
import os
import numpy as np

# ============================================================================
# CONFIGURATION - Set this to your initial_meshes folder path
# ============================================================================
MESH_DIR = r"D:\Data\DAT_Sim\multiphysics_drop\4x5_20260119_004809\initial_meshes"
SCALE = 0.01  # Convert cm to meters
# ============================================================================


def clear_scene():
    """Remove all mesh objects from the scene."""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()


def create_mesh_from_arrays(name, vertices, faces, scale=1.0):
    """Create a Blender mesh from vertex and face arrays."""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    # Link object to scene
    bpy.context.collection.objects.link(obj)
    
    # Scale vertices (convert cm to meters)
    scaled_vertices = (vertices * scale).tolist()
    
    # Create mesh from vertices and faces
    mesh.from_pydata(scaled_vertices, [], faces.tolist())
    mesh.update()
    
    return obj


def load_ply(filepath):
    """Load a PLY file and return vertices and faces."""
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    i = 0
    num_vertices = 0
    num_faces = 0
    in_header = True
    
    while i < len(lines) and in_header:
        line = lines[i].strip()
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
        elif line.startswith('element face'):
            num_faces = int(line.split()[-1])
        elif line == 'end_header':
            in_header = False
        i += 1
    
    # Read vertices
    for j in range(num_vertices):
        parts = lines[i + j].strip().split()
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    i += num_vertices
    
    # Read faces
    for j in range(num_faces):
        parts = lines[i + j].strip().split()
        # First number is vertex count (should be 3 for triangles)
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def set_material(obj, color, name=None):
    """Create and assign a material with given color to object."""
    mat_name = name or f"Mat_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    # Get the principled BSDF node
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def load_initial_meshes(mesh_dir):
    """Load all initial meshes from the specified directory."""
    
    # Load mesh info
    info_path = os.path.join(mesh_dir, "mesh_info.json")
    if not os.path.exists(info_path):
        print(f"Error: mesh_info.json not found in {mesh_dir}")
        return
    
    with open(info_path, 'r') as f:
        mesh_info = json.load(f)
    
    print(f"Loading meshes from: {mesh_dir}")
    
    # Create collections for organization
    soft_collection = bpy.data.collections.new("SoftBodies")
    rigid_collection = bpy.data.collections.new("RigidBodies")
    cloth_collection = bpy.data.collections.new("Cloth")
    bpy.context.scene.collection.children.link(soft_collection)
    bpy.context.scene.collection.children.link(rigid_collection)
    bpy.context.scene.collection.children.link(cloth_collection)
    
    # Load soft bodies
    for soft_info in mesh_info.get("soft_bodies", []):
        name = soft_info["name"]
        ply_path = os.path.join(mesh_dir, f"{name}.ply")
        
        if os.path.exists(ply_path):
            vertices, faces = load_ply(ply_path)
            obj = create_mesh_from_arrays(name, vertices, faces, scale=SCALE)
            
            # Move to soft bodies collection
            bpy.context.collection.objects.unlink(obj)
            soft_collection.objects.link(obj)
            
            # Set color based on type
            if "hippo" in name:
                set_material(obj, (0.2, 0.6, 0.9))  # Light blue
            elif "bunny" in name:
                set_material(obj, (0.3, 0.7, 0.4))  # Green
            else:
                set_material(obj, (0.5, 0.5, 0.8))  # Purple
            
            print(f"  Loaded soft body: {name}")
    
    # Load rigid bodies
    for rigid_info in mesh_info.get("rigid_bodies", []):
        name = rigid_info["name"]
        ply_path = os.path.join(mesh_dir, f"{name}.ply")
        
        if os.path.exists(ply_path):
            vertices, faces = load_ply(ply_path)
            obj = create_mesh_from_arrays(name, vertices, faces, scale=SCALE)
            
            # Move to rigid bodies collection
            bpy.context.collection.objects.unlink(obj)
            rigid_collection.objects.link(obj)
            
            # Set color based on type
            if "box" in name:
                set_material(obj, (0.9, 0.4, 0.2))  # Orange
            elif "gear" in name:
                set_material(obj, (0.7, 0.7, 0.2))  # Gold
            else:
                set_material(obj, (0.6, 0.3, 0.3))  # Brown
            
            print(f"  Loaded rigid body: {name}")
    
    # Load cloth
    cloth_info = mesh_info.get("cloth")
    if cloth_info:
        name = cloth_info["name"]
        ply_path = os.path.join(mesh_dir, f"{name}.ply")
        
        if os.path.exists(ply_path):
            vertices, faces = load_ply(ply_path)
            obj = create_mesh_from_arrays(name, vertices, faces, scale=SCALE)
            
            # Move to cloth collection
            bpy.context.collection.objects.unlink(obj)
            cloth_collection.objects.link(obj)
            
            set_material(obj, (0.8, 0.8, 0.9))  # Light gray/white
            
            print(f"  Loaded cloth: {name}")
    
    print("Done loading meshes!")
    
    # Select all loaded objects
    bpy.ops.object.select_all(action='SELECT')
    
    # Try to frame view (only works if running from 3D viewport)
    try:
        bpy.ops.view3d.view_selected(use_all_regions=False)
    except RuntimeError:
        # Not in 3D view context, skip framing
        pass


def main():
    # Check if MESH_DIR exists
    if not os.path.exists(MESH_DIR):
        print(f"Error: Directory not found: {MESH_DIR}")
        print("Please update MESH_DIR at the top of this script to point to your initial_meshes folder.")
        return
    
    # Clear existing meshes (optional - comment out if you want to keep existing objects)
    # clear_scene()
    
    # Load meshes
    load_initial_meshes(MESH_DIR)


if __name__ == "__main__":
    main()
