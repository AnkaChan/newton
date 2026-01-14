"""
Filter and Preview Script

Applies temporal filtering to npy sequences, then launches animation preview.
Supports two filtering modes:
  - Butterworth: Low-pass filter to remove high-frequency jitter
  - Linear Stabilizer: Blends toward a target frame to settle endings

Both can be applied together (Butterworth first, then stabilizer).
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

in_folder = r"D:\Data\DAT_Sim\ClothDrop\200_layers\20260108_020623"
# in_folder = r"D:\Data\DAT_Sim\ClothDrop\300_layers\20260108_020850"

# -----------------------------------------------------------------------------
# Filter Selection (set to True/False to enable/disable each)
# -----------------------------------------------------------------------------
use_butterworth = True
use_linear_stabilizer = True

# -----------------------------------------------------------------------------
# Butterworth Filter Parameters
# -----------------------------------------------------------------------------
cutoff_freq = 0.05    # Normalized frequency (0-1), higher = less smoothing
filter_order = 2      # Butterworth filter order

# -----------------------------------------------------------------------------
# Linear Stabilizer Parameters
# -----------------------------------------------------------------------------
stabilize_start_frame = 160     # Frame to start blending
stabilize_target_frame = 300    # Frame to blend toward (stable reference)
stabilize_min_blend = 0.02      # Minimum weight for current state (0.02 = 2%)

# -----------------------------------------------------------------------------
# Preview Parameters
# -----------------------------------------------------------------------------
fps = 60
loop = True


# =============================================================================
# Butterworth Filter
# =============================================================================

def butter_lowpass_filter(data: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    """Apply Butterworth low-pass filter along axis 0 (time)."""
    b, a = butter(order, cutoff, btype='low', analog=False)
    
    min_samples = 3 * max(len(a), len(b))
    if data.shape[0] < min_samples:
        print(f"Warning: Not enough frames ({data.shape[0]}) for filter. Need at least {min_samples}.")
        return data
    
    return filtfilt(b, a, data, axis=0)


def apply_butterworth(all_data: np.ndarray, cutoff_freq: float, filter_order: int) -> np.ndarray:
    """Apply Butterworth filter to loaded data. Shape: (frames, vertices, 3)."""
    print("Applying Butterworth filter...")
    filtered_data = np.zeros_like(all_data)
    for dim in range(3):
        filtered_data[:, :, dim] = butter_lowpass_filter(all_data[:, :, dim], cutoff_freq, filter_order)
    return filtered_data


# =============================================================================
# Linear Stabilizer
# =============================================================================

def apply_linear_stabilizer(all_data: np.ndarray, start_frame: int, target_frame: int, min_blend: float) -> np.ndarray:
    """Apply linear stabilization to loaded data. Shape: (frames, vertices, 3).
    
    Gradually blends frames toward target_frame starting from start_frame.
    By the time we reach target_frame, blend is at min_blend (e.g., 2% current, 98% target).
    """
    print(f"Applying linear stabilizer (frames {start_frame} -> {target_frame})...")
    
    num_frames = all_data.shape[0]
    
    if target_frame >= num_frames:
        print(f"Warning: target_frame ({target_frame}) >= num_frames ({num_frames}). Clamping.")
        target_frame = num_frames - 1
    
    if start_frame >= target_frame:
        print(f"Warning: start_frame ({start_frame}) >= target_frame ({target_frame}). Skipping stabilization.")
        return all_data
    
    stable_pos = all_data[target_frame].copy()
    stabilizing_speed = (1 - min_blend) / (target_frame - start_frame)
    
    result = all_data.copy()
    
    for i in tqdm.tqdm(range(start_frame, num_frames), desc="Stabilizing"):
        steps = i - start_frame
        w_current = max(1 - steps * stabilizing_speed, min_blend)
        w_stable = 1 - w_current
        
        result[i] = all_data[i] * w_current + stable_pos * w_stable
    
    return result


# =============================================================================
# Combined Filter Pipeline
# =============================================================================

def copy_mesh_files(in_folder: str, out_folder: str):
    """Copy PLY mesh files from input to output folder."""
    ply_files = glob.glob(join(in_folder, "*.ply"))
    for ply_file in ply_files:
        dst = join(out_folder, Path(ply_file).name)
        shutil.copy2(ply_file, dst)
        print(f"Copied mesh: {Path(ply_file).name}")


def generate_output_folder_name(
    use_butter: bool, butter_order: int, butter_cutoff: float,
    use_linear: bool, linear_start: int, linear_target: int, linear_min: float
) -> str:
    """Generate output folder name based on filter settings."""
    parts = ["filtered"]
    
    if use_butter:
        cutoff_str = str(butter_cutoff).replace(".", "")
        parts.append(f"butter_o{butter_order}_c{cutoff_str}")
    
    if use_linear:
        min_pct = int(linear_min * 100)
        parts.append(f"linear_s{linear_start}_t{linear_target}_m{min_pct}")
    
    return "_".join(parts)


def filter_npy_pipeline(
    in_folder: str,
    use_butter: bool = True,
    butter_cutoff: float = 0.45,
    butter_order: int = 2,
    use_linear: bool = False,
    linear_start: int = 200,
    linear_target: int = 250,
    linear_min: float = 0.02
) -> str:
    """Apply filtering pipeline and return output folder path."""
    
    if not use_butter and not use_linear:
        print("No filters enabled. Returning input folder.")
        return in_folder
    
    # Generate output folder name
    folder_name = generate_output_folder_name(
        use_butter, butter_order, butter_cutoff,
        use_linear, linear_start, linear_target, linear_min
    )
    out_folder = join(in_folder, folder_name)
    
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
    
    # Print config
    print("=" * 60)
    print("Filter Pipeline")
    print(f"  Input:         {in_folder}")
    print(f"  Output:        {out_folder}")
    print(f"  Num frames:    {num_frames}")
    print("-" * 60)
    if use_butter:
        print(f"  [1] Butterworth:  order={butter_order}, cutoff={butter_cutoff}")
    if use_linear:
        print(f"  [2] Linear Stab:  start={linear_start}, target={linear_target}, min={linear_min*100:.0f}%")
    print("=" * 60)
    
    # Load all frames
    print("Loading frames...")
    frames = []
    for f in tqdm.tqdm(npy_files, desc="Loading"):
        frames.append(np.load(f))
    
    all_data = np.stack(frames, axis=0)
    print(f"Data shape: {all_data.shape}")
    
    # Apply filters in sequence
    if use_butter:
        all_data = apply_butterworth(all_data, butter_cutoff, butter_order)
    
    if use_linear:
        all_data = apply_linear_stabilizer(all_data, linear_start, linear_target, linear_min)
    
    # Save
    print("Saving...")
    for i, npy_path in enumerate(tqdm.tqdm(npy_files, desc="Saving")):
        out_path = join(out_folder, Path(npy_path).name)
        np.save(out_path, all_data[i].astype(np.float32))
    
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
    # Step 1: Apply filter pipeline
    filtered_folder = filter_npy_pipeline(
        in_folder,
        use_butter=use_butterworth,
        butter_cutoff=cutoff_freq,
        butter_order=filter_order,
        use_linear=use_linear_stabilizer,
        linear_start=stabilize_start_frame,
        linear_target=stabilize_target_frame,
        linear_min=stabilize_min_blend
    )
    
    if filtered_folder:
        # Step 2: Preview
        print("\nLaunching preview...")
        launch_preview(filtered_folder)

