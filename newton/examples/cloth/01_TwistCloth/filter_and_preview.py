"""
Filter and Preview Script

Applies temporal low-pass filter to npy sequences, then launches animation preview.
All-in-one workflow for cleaning up and reviewing simulation results.
"""

import glob
import os
import shutil
import time
from os.path import join
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import tqdm
from scipy.signal import butter, filtfilt


# =============================================================================
# Configuration
# =============================================================================

in_folder = r"D:\Data\DAT_Sim\cloth_twist_release\truncation_1_iter_100_20260105_164500"

# Filter parameters
cutoff_freq = 0.15    # Normalized frequency (0-1), higher = less smoothing
filter_order = 2      # Butterworth filter order

# Preview parameters
fps = 60
loop = True


# =============================================================================
# Temporal Filter
# =============================================================================

def butter_lowpass_filter(data: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    """Apply Butterworth low-pass filter along axis 0 (time)."""
    b, a = butter(order, cutoff, btype='low', analog=False)
    
    min_samples = 3 * max(len(a), len(b))
    if data.shape[0] < min_samples:
        print(f"Warning: Not enough frames ({data.shape[0]}) for filter. Need at least {min_samples}.")
        return data
    
    return filtfilt(b, a, data, axis=0)


def copy_mesh_files(in_folder: str, out_folder: str):
    """Copy PLY mesh files from input to output folder."""
    ply_files = glob.glob(join(in_folder, "*.ply"))
    for ply_file in ply_files:
        dst = join(out_folder, Path(ply_file).name)
        shutil.copy2(ply_file, dst)
        print(f"Copied mesh: {Path(ply_file).name}")


def temporal_filter_npy(in_folder: str, cutoff_freq: float, filter_order: int) -> str:
    """Apply temporal filter and return output folder path."""
    
    # Generate output folder as subfolder of input
    cutoff_str = str(cutoff_freq).replace(".", "")
    out_folder = join(in_folder, f"filtered_butter_o{filter_order}_c{cutoff_str}")
    
    # Check if already filtered
    if os.path.exists(out_folder):
        existing_files = glob.glob(join(out_folder, "*.npy"))
        if existing_files:
            print(f"Found existing filtered data: {out_folder}")
            print(f"  ({len(existing_files)} files)")
            return out_folder
    
    os.makedirs(out_folder, exist_ok=True)
    copy_mesh_files(in_folder, out_folder)
    
    # Get input files
    npy_files = sorted(glob.glob(join(in_folder, "*.npy")))
    num_frames = len(npy_files)
    
    if num_frames == 0:
        print(f"Error: No npy files found in {in_folder}")
        return None
    
    print("=" * 60)
    print("Temporal Low-Pass Filter")
    print(f"  Input:         {in_folder}")
    print(f"  Output:        {out_folder}")
    print(f"  Num frames:    {num_frames}")
    print(f"  Filter:        Butterworth order {filter_order}")
    print(f"  Cutoff freq:   {cutoff_freq}")
    print("=" * 60)
    
    # Load all frames
    print("Loading frames...")
    frames = []
    for f in tqdm.tqdm(npy_files, desc="Loading"):
        frames.append(np.load(f))
    
    all_data = np.stack(frames, axis=0)
    print(f"Data shape: {all_data.shape}")
    
    # Apply filter
    print("Filtering...")
    filtered_data = np.zeros_like(all_data)
    for dim in range(3):
        filtered_data[:, :, dim] = butter_lowpass_filter(all_data[:, :, dim], cutoff_freq, filter_order)
    
    # Save
    print("Saving...")
    for i, npy_path in enumerate(tqdm.tqdm(npy_files, desc="Saving")):
        out_path = join(out_folder, Path(npy_path).name)
        np.save(out_path, filtered_data[i].astype(np.float32))
    
    print(f"Done! Filtered data saved to: {out_folder}")
    return out_folder


# =============================================================================
# Animation Preview
# =============================================================================

class AnimState:
    frames: np.ndarray = None
    faces: np.ndarray = None
    current_frame: int = 0
    num_frames: int = 0
    is_playing: bool = False
    last_frame_time: float = 0
    frame_duration: float = 1.0 / 60.0
    mesh_name: str = "animation"

state = AnimState()


def load_ply_faces(ply_path: str) -> tuple:
    """Load vertices and faces from PLY file."""
    import pyvista as pv
    mesh = pv.read(ply_path)
    
    verts = np.array(mesh.points)
    
    faces_raw = mesh.faces
    if faces_raw is None or len(faces_raw) == 0:
        return verts, None
    
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


def load_animation(folder: str):
    """Load animation frames for preview."""
    global state
    
    npy_files = sorted(glob.glob(join(folder, "*.npy")))
    if len(npy_files) == 0:
        return False
    
    print(f"Loading {len(npy_files)} frames for preview...")
    frames = [np.load(f) for f in npy_files]
    
    state.frames = np.stack(frames, axis=0)
    state.num_frames = len(frames)
    state.current_frame = 0
    state.frame_duration = 1.0 / fps
    
    # Load all mesh PLY files and combine faces with offset indices
    ply_files = find_all_mesh_plys(folder)
    if ply_files:
        print(f"Found {len(ply_files)} mesh files")
        
        all_faces = []
        vertex_offset = 0
        
        for ply_path in ply_files:
            verts, faces = load_ply_faces(ply_path)
            if faces is not None:
                # Offset face indices by cumulative vertex count
                offset_faces = faces + vertex_offset
                all_faces.append(offset_faces)
            vertex_offset += len(verts)
        
        if all_faces:
            state.faces = np.vstack(all_faces)
            print(f"Combined mesh: {state.faces.shape[0]} faces, {vertex_offset} vertices")
    
    return True


def update_mesh():
    """Update displayed mesh."""
    verts = state.frames[state.current_frame]
    if state.faces is not None:
        ps.register_surface_mesh(state.mesh_name, verts, state.faces)
    else:
        ps.register_point_cloud(state.mesh_name, verts)


def gui_callback():
    """ImGui controls."""
    global state
    
    current_time = time.time()
    
    if state.is_playing:
        if current_time - state.last_frame_time >= state.frame_duration:
            state.current_frame += 1
            if state.current_frame >= state.num_frames:
                state.current_frame = 0 if loop else state.num_frames - 1
                if not loop:
                    state.is_playing = False
            state.last_frame_time = current_time
            update_mesh()
    
    psim.TextUnformatted(f"Frame: {state.current_frame} / {state.num_frames - 1}")
    psim.TextUnformatted(f"FPS: {fps} | {'PLAYING' if state.is_playing else 'PAUSED'}")
    
    psim.Separator()
    
    if psim.Button("Play" if not state.is_playing else "Pause"):
        state.is_playing = not state.is_playing
        state.last_frame_time = current_time
    
    psim.SameLine()
    if psim.Button("<< Prev"):
        state.current_frame = max(0, state.current_frame - 1)
        update_mesh()
    
    psim.SameLine()
    if psim.Button("Next >>"):
        state.current_frame = min(state.num_frames - 1, state.current_frame + 1)
        update_mesh()
    
    changed, new_frame = psim.SliderInt("Frame", state.current_frame, 0, state.num_frames - 1)
    if changed:
        state.current_frame = new_frame
        update_mesh()
    
    psim.Separator()
    if psim.Button("Start"):
        state.current_frame = 0
        update_mesh()
    psim.SameLine()
    if psim.Button("End"):
        state.current_frame = state.num_frames - 1
        update_mesh()


def launch_preview(folder: str):
    """Launch animation preview."""
    if not load_animation(folder):
        print("Failed to load animation")
        return
    
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    
    update_mesh()
    ps.set_user_callback(gui_callback)
    
    print("\n" + "=" * 50)
    print("Animation Preview - Filtered Result")
    print(f"  Frames: {state.num_frames}")
    print(f"  FPS: {fps}")
    print("=" * 50 + "\n")
    
    ps.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Step 1: Filter
    filtered_folder = temporal_filter_npy(in_folder, cutoff_freq, filter_order)
    
    if filtered_folder:
        # Step 2: Preview
        print("\nLaunching preview...")
        launch_preview(filtered_folder)

