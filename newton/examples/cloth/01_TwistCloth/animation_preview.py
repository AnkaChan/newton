"""
Animation Preview Script

Quickly review npy animation sequences at 60fps using polyscope.
Press SPACE to play/pause, LEFT/RIGHT arrows to step frames.
"""

import glob
import os
import time
from os.path import join
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# =============================================================================
# Configuration
# =============================================================================

in_folder = r"D:\Data\DAT_Sim\cloth_twist_release\truncation_1_iter_100_20260105_164500_filtered_butter_o2_c025"
mesh_ply_path = None  # Optional: path to PLY file, or auto-detect from folder

fps = 60
loop = True

# =============================================================================
# Global State
# =============================================================================

class AnimState:
    frames: np.ndarray = None  # (num_frames, num_vertices, 3)
    faces: np.ndarray = None   # (num_faces, 3) or None for point cloud
    current_frame: int = 0
    num_frames: int = 0
    is_playing: bool = False
    last_frame_time: float = 0
    frame_duration: float = 1.0 / 60.0
    mesh_name: str = "animation"

state = AnimState()

# =============================================================================
# Loading
# =============================================================================

def load_ply_faces(ply_path: str) -> tuple:
    """Load vertices and faces from a PLY file."""
    import pyvista as pv
    mesh = pv.read(ply_path)
    
    verts = np.array(mesh.points)
    
    # pyvista faces format: [n, v0, v1, v2, n, v0, v1, v2, ...]
    faces_raw = mesh.faces
    if faces_raw is None or len(faces_raw) == 0:
        return verts, None
    
    # Convert to (N, 3) array
    faces = []
    i = 0
    while i < len(faces_raw):
        n = faces_raw[i]
        if n == 3:
            faces.append([faces_raw[i+1], faces_raw[i+2], faces_raw[i+3]])
        i += n + 1
    return verts, np.array(faces) if faces else None


def find_all_mesh_plys(folder: str) -> list:
    """Find all mesh PLY files in folder, sorted."""
    # Look for numbered mesh files first (for multi-layer simulations)
    ply_files = sorted(glob.glob(join(folder, "initial_mesh_*.ply")))
    if ply_files:
        return ply_files
    
    # Fallback to any PLY
    ply_files = sorted(glob.glob(join(folder, "*.ply")))
    return ply_files


def load_animation(folder: str, ply_path: str = None):
    """Load all npy frames and optionally mesh topology."""
    global state
    
    npy_files = sorted(glob.glob(join(folder, "*.npy")))
    if len(npy_files) == 0:
        print(f"Error: No npy files found in {folder}")
        return False
    
    print(f"Loading {len(npy_files)} frames...")
    frames = []
    for f in npy_files:
        frames.append(np.load(f))
    
    state.frames = np.stack(frames, axis=0)
    state.num_frames = len(frames)
    state.current_frame = 0
    state.frame_duration = 1.0 / fps
    
    print(f"Loaded: {state.frames.shape} (frames, vertices, 3)")
    
    # Load all mesh PLY files and combine faces with offset indices
    ply_files = find_all_mesh_plys(folder)
    if ply_files:
        print(f"Found {len(ply_files)} mesh files")
        
        all_faces = []
        vertex_offset = 0
        
        for ply_file in ply_files:
            verts, faces = load_ply_faces(ply_file)
            if faces is not None:
                # Offset face indices by cumulative vertex count
                offset_faces = faces + vertex_offset
                all_faces.append(offset_faces)
            vertex_offset += len(verts)
        
        if all_faces:
            state.faces = np.vstack(all_faces)
            print(f"Combined mesh: {state.faces.shape[0]} faces, {vertex_offset} vertices")
    else:
        print("No mesh topology found - displaying as point cloud")
    
    return True

# =============================================================================
# Polyscope Callbacks
# =============================================================================

def update_mesh():
    """Update the displayed mesh/point cloud."""
    verts = state.frames[state.current_frame]
    
    if state.faces is not None:
        ps.register_surface_mesh(state.mesh_name, verts, state.faces)
    else:
        ps.register_point_cloud(state.mesh_name, verts)


def gui_callback():
    """ImGui callback for controls."""
    global state
    
    current_time = time.time()
    
    # Auto-advance if playing
    if state.is_playing:
        if current_time - state.last_frame_time >= state.frame_duration:
            state.current_frame += 1
            if state.current_frame >= state.num_frames:
                if loop:
                    state.current_frame = 0
                else:
                    state.current_frame = state.num_frames - 1
                    state.is_playing = False
            state.last_frame_time = current_time
            update_mesh()
    
    # GUI Controls
    psim.TextUnformatted(f"Frame: {state.current_frame} / {state.num_frames - 1}")
    psim.TextUnformatted(f"FPS: {fps} | {'PLAYING' if state.is_playing else 'PAUSED'}")
    
    psim.Separator()
    
    # Play/Pause button
    if psim.Button("Play" if not state.is_playing else "Pause"):
        state.is_playing = not state.is_playing
        state.last_frame_time = current_time
    
    psim.SameLine()
    
    # Step buttons
    if psim.Button("<< Prev"):
        state.current_frame = max(0, state.current_frame - 1)
        update_mesh()
    
    psim.SameLine()
    
    if psim.Button("Next >>"):
        state.current_frame = min(state.num_frames - 1, state.current_frame + 1)
        update_mesh()
    
    # Frame slider
    changed, new_frame = psim.SliderInt("Frame", state.current_frame, 0, state.num_frames - 1)
    if changed:
        state.current_frame = new_frame
        update_mesh()
    
    # Jump controls
    psim.Separator()
    if psim.Button("Start"):
        state.current_frame = 0
        update_mesh()
    
    psim.SameLine()
    
    if psim.Button("End"):
        state.current_frame = state.num_frames - 1
        update_mesh()
    
    psim.Separator()
    psim.TextUnformatted("Keyboard: SPACE=play/pause")


# =============================================================================
# Main
# =============================================================================

def main():
    # Load animation
    if not load_animation(in_folder, mesh_ply_path):
        return
    
    # Initialize polyscope
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    
    # Register initial frame
    update_mesh()
    
    # Set callback
    ps.set_user_callback(gui_callback)
    
    print("\n" + "=" * 50)
    print("Animation Preview")
    print(f"  Frames: {state.num_frames}")
    print(f"  FPS: {fps}")
    print("  Controls:")
    print("    - Click Play/Pause or press buttons")
    print("    - Use slider to scrub")
    print("=" * 50 + "\n")
    
    # Show
    ps.show()


if __name__ == "__main__":
    main()

