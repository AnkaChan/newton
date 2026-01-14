"""
Temporal Low-Pass Filter for Motion Sequences

Applies a Butterworth low-pass filter to smooth out high-frequency jittering
in npy motion sequences (Nx3 vertex coordinates per frame).
"""

import glob
import os
from os.path import join
from pathlib import Path

import numpy as np
import tqdm
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to data along axis 0 (time).
    
    Args:
        data: Array of shape (num_frames, ...) - filter applied along first axis
        cutoff: Normalized cutoff frequency (0 < cutoff < 1), higher = less smoothing
        order: Filter order (default 2)
    
    Returns:
        Filtered data with same shape as input
    """
    # Design Butterworth filter
    b, a = butter(order, cutoff, btype='low', analog=False)
    
    # Apply filter forward and backward to avoid phase shift
    # filtfilt requires at least 3*max(len(a), len(b)) samples
    min_samples = 3 * max(len(a), len(b))
    if data.shape[0] < min_samples:
        print(f"Warning: Not enough frames ({data.shape[0]}) for filter. Need at least {min_samples}.")
        return data
    
    return filtfilt(b, a, data, axis=0)


def copy_mesh_files(in_folder: str, out_folder: str):
    """Copy PLY mesh files from input to output folder."""
    import shutil
    
    ply_files = glob.glob(join(in_folder, "*.ply"))
    for ply_file in ply_files:
        dst = join(out_folder, Path(ply_file).name)
        shutil.copy2(ply_file, dst)
        print(f"Copied mesh: {Path(ply_file).name}")


def temporal_filter_npy(
    in_folder: str,
    cutoff_freq: float = 0.35,
    filter_order: int = 2,
    out_folder: str = None,
):
    """
    Apply temporal low-pass filter to all npy files in a folder.
    
    Args:
        in_folder: Input folder containing npy files (Nx3 vertex coordinates)
        cutoff_freq: Normalized cutoff frequency (0-1), higher = less smoothing
        filter_order: Butterworth filter order
        out_folder: Output folder (auto-generated if None)
    """
    # Generate output folder as subfolder of input
    if out_folder is None:
        cutoff_str = str(cutoff_freq).replace(".", "")
        out_folder = join(in_folder, f"filtered_butter_o{filter_order}_c{cutoff_str}")
    
    os.makedirs(out_folder, exist_ok=True)
    
    # Copy mesh files so animation_preview.py can find them
    copy_mesh_files(in_folder, out_folder)
    
    # Get input files
    npy_files = sorted(glob.glob(join(in_folder, "*.npy")))
    num_frames = len(npy_files)
    
    if num_frames == 0:
        print(f"Error: No npy files found in {in_folder}")
        return
    
    print("=" * 60)
    print("Temporal Low-Pass Filter Configuration:")
    print(f"  Input:         {in_folder}")
    print(f"  Output:        {out_folder}")
    print(f"  Num frames:    {num_frames}")
    print(f"  Filter:        Butterworth order {filter_order}")
    print(f"  Cutoff freq:   {cutoff_freq} (normalized)")
    print("=" * 60)
    
    # Load all frames into memory
    print("Loading all frames...")
    frames = []
    for npy_path in tqdm.tqdm(npy_files, desc="Loading"):
        frames.append(np.load(npy_path))
    
    # Stack into (num_frames, num_vertices, 3)
    all_data = np.stack(frames, axis=0)
    num_vertices = all_data.shape[1]
    print(f"Data shape: {all_data.shape} (frames, vertices, 3)")
    
    # Apply filter to each coordinate dimension
    print("Applying Butterworth low-pass filter...")
    filtered_data = np.zeros_like(all_data)
    
    for dim in range(3):  # x, y, z
        dim_name = ['X', 'Y', 'Z'][dim]
        print(f"  Filtering {dim_name} coordinates...")
        # Shape: (num_frames, num_vertices) -> filter along axis 0
        filtered_data[:, :, dim] = butter_lowpass_filter(
            all_data[:, :, dim], 
            cutoff_freq, 
            filter_order
        )
    
    # Save filtered frames
    print("Saving filtered frames...")
    for i, npy_path in enumerate(tqdm.tqdm(npy_files, desc="Saving")):
        out_path = join(out_folder, Path(npy_path).name)
        np.save(out_path, filtered_data[i].astype(np.float32))
    
    print("=" * 60)
    print(f"Done! Output saved to: {out_folder}")
    print(f"Total frames processed: {num_frames}")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    in_folder = r"D:\Data\DAT_Sim\cloth_twist_release\truncation_1_iter_100_20260105_164500"
    
    # Filter parameters
    cutoff_freq = 0.25    # Normalized frequency (0-1), higher = less smoothing
    filter_order = 2      # Butterworth filter order
    
    # -------------------------------------------------------------------------
    # Run filter
    # -------------------------------------------------------------------------
    
    temporal_filter_npy(
        in_folder=in_folder,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
    )

