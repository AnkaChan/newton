"""
Blender Rendering Setup Script for Unroll Cloth Simulation

Sets up a static rendering scene with lighting, camera, materials, and initial meshes.
Update vertex coordinates yourself after running this script.

Usage:
    1. Open Blender
    2. Go to Scripting workspace
    3. Open this script
    4. Adjust the OUTPUT_DIR path to your simulation output
    5. Run the script
"""

import bpy
import os
import math

# =============================================================================
# Configuration
# =============================================================================

# Path to simulation output directory (adjust this!)
OUTPUT_DIR = r"D:\Data\DAT_Sim\unroll_cloth"

# Rendering settings
RENDER_RESOLUTION_X = 1920
RENDER_RESOLUTION_Y = 1080
RENDER_SAMPLES = 128
RENDER_ENGINE = "CYCLES"  # "CYCLES" or "BLENDER_EEVEE"
USE_GPU = True

# Camera settings
CAMERA_LOCATION = (150, -300, 200)
CAMERA_LOOK_AT = (0, 100, 50)

# Materials
CLOTH_COLOR = (0.8, 0.3, 0.3, 1.0)  # Reddish
COLLIDER_COLOR = (0.3, 0.3, 0.3, 1.0)  # Gray
GROUND_COLOR = (0.9, 0.9, 0.9, 1.0)  # Light gray


# =============================================================================
# Utility Functions
# =============================================================================


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def create_material(name, color, roughness=0.5, metallic=0.0, subsurface=0.0):
    """Create a PBR material."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = metallic
    if "Subsurface Weight" in principled.inputs:
        principled.inputs["Subsurface Weight"].default_value = subsurface
    elif "Subsurface" in principled.inputs:
        principled.inputs["Subsurface"].default_value = subsurface

    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    return mat


def setup_camera():
    """Set up the camera."""
    import mathutils

    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    camera = bpy.context.object
    camera.name = "SimCamera"

    # Point camera at target
    look_at = mathutils.Vector(CAMERA_LOOK_AT)
    cam_loc = mathutils.Vector(CAMERA_LOCATION)
    direction = look_at - cam_loc
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = camera
    camera.data.lens = 50
    camera.data.clip_start = 0.1
    camera.data.clip_end = 10000

    return camera


def setup_lighting():
    """Set up three-point lighting."""
    # Key light
    bpy.ops.object.light_add(type="AREA", location=(200, -200, 300))
    key_light = bpy.context.object
    key_light.name = "KeyLight"
    key_light.data.energy = 50000
    key_light.data.size = 100
    key_light.data.color = (1.0, 0.98, 0.95)

    # Fill light
    bpy.ops.object.light_add(type="AREA", location=(-200, -100, 150))
    fill_light = bpy.context.object
    fill_light.name = "FillLight"
    fill_light.data.energy = 20000
    fill_light.data.size = 150
    fill_light.data.color = (0.9, 0.95, 1.0)

    # Rim light
    bpy.ops.object.light_add(type="AREA", location=(0, 200, 250))
    rim_light = bpy.context.object
    rim_light.name = "RimLight"
    rim_light.data.energy = 30000
    rim_light.data.size = 80

    # Environment
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.05, 0.05, 0.08, 1.0)
        bg.inputs["Strength"].default_value = 0.5


def setup_ground_plane():
    """Create a ground plane."""
    bpy.ops.mesh.primitive_plane_add(size=2000, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"
    mat = create_material("GroundMaterial", GROUND_COLOR, roughness=0.9)
    ground.data.materials.append(mat)
    return ground


def import_ply(filepath, name, color, roughness=0.5, metallic=0.0, subsurface=0.0):
    """Import a PLY file with material."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    bpy.ops.wm.ply_import(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    obj.name = name

    mat = create_material(f"{name}Material", color, roughness, metallic, subsurface)
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()

    print(f"Loaded: {filepath}")
    return obj


def setup_render_settings():
    """Configure render settings."""
    scene = bpy.context.scene

    scene.render.engine = RENDER_ENGINE
    if RENDER_ENGINE == "CYCLES":
        scene.cycles.samples = RENDER_SAMPLES
        scene.cycles.use_denoising = True

        if USE_GPU:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"
            for device in prefs.devices:
                device.use = True
            scene.cycles.device = "GPU"

    scene.render.resolution_x = RENDER_RESOLUTION_X
    scene.render.resolution_y = RENDER_RESOLUTION_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"


def setup_scene(output_dir=None):
    """Main function to set up the scene."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print(f"\n{'='*60}")
    print("Blender Scene Setup for Cloth Simulation")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    clear_scene()
    setup_render_settings()
    setup_lighting()
    setup_camera()
    setup_ground_plane()

    # Load initial meshes
    collider_path = os.path.join(output_dir, "initial_collider.ply")
    cloth_path = os.path.join(output_dir, "initial_cloth.ply")

    collider = import_ply(
        collider_path, "Collider",
        color=COLLIDER_COLOR, roughness=0.3, metallic=0.5
    )

    cloth = import_ply(
        cloth_path, "Cloth",
        color=CLOTH_COLOR, roughness=0.8, subsurface=0.1
    )

    bpy.context.scene.frame_set(0)

    print(f"\n{'='*60}")
    print("Scene setup complete!")
    print("Objects created: Collider, Cloth, Ground, Lights, Camera")
    print(f"{'='*60}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    setup_scene()
