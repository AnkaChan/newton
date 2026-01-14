import pyvista as pv
from pathlib import Path
import tqdm
import glob
from os.path import join
import os
import numpy as np
import array


# =============================================================================
# NPY Stabilizers (Nx3 coordinates)
# =============================================================================

def copy_mesh_files(inFolder: str, outFolder: str):
    """Copy PLY mesh files from input to output folder."""
    import shutil
    
    ply_files = glob.glob(join(inFolder, "*.ply"))
    for ply_file in ply_files:
        dst = join(outFolder, Path(ply_file).name)
        shutil.copy2(ply_file, dst)
        print(f"Copied mesh: {Path(ply_file).name}")


def linearStabilizerNpy(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder, copyUnchanged=True):
    """Linear stabilizer for npy files with Nx3 coordinates.
    
    Args:
        copyUnchanged: If True, copy frames before startFrame to output folder unchanged.
    """
    os.makedirs(outFolder, exist_ok=True)
    
    # Copy mesh files (PLY) so animation_preview.py can find them
    if inFiles:
        inFolder = str(Path(inFiles[0]).parent)
        copy_mesh_files(inFolder, outFolder)

    # Copy unchanged frames (0 to startFrame-1)
    if copyUnchanged:
        print(f"Copying {startFrame} unchanged frames...")
        for iFile in tqdm.tqdm(range(min(startFrame, len(inFiles))), desc="Copying"):
            meshFile = inFiles[iFile]
            meshFP = Path(meshFile)
            outFile = join(outFolder, meshFP.stem + ".npy")
            
            # Load and save (to ensure consistent format)
            data = np.load(meshFile)
            np.save(outFile, data.astype(np.float32))

    stablePos = np.load(inFiles[stabilizingFrame])  # Shape: (N, 3)
    
    print(f"Stabilizing frames {startFrame} to {len(inFiles)-1}...")
    for iFile in tqdm.tqdm(range(startFrame, len(inFiles)), desc="Stabilizing"):
        meshFile = inFiles[iFile]
        steps = iFile - startFrame

        wCurState = 1 - steps * stabilizingSpeed
        if wCurState < minBlending:
            wCurState = minBlending

        wStabilizer = 1 - wCurState

        curPos = np.load(meshFile)  # Shape: (N, 3)
        
        blendedPos = curPos * wCurState + stablePos * wStabilizer

        meshFP = Path(meshFile)
        outFile = join(outFolder, meshFP.stem + ".npy")
        np.save(outFile, blendedPos.astype(np.float32))


def exponentialStabilizerNpy(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder, copyUnchanged=True):
    """Exponential stabilizer for npy files with Nx3 coordinates.
    
    Args:
        copyUnchanged: If True, copy frames before startFrame to output folder unchanged.
    """
    os.makedirs(outFolder, exist_ok=True)
    
    # Copy mesh files (PLY) so animation_preview.py can find them
    if inFiles:
        inFolder = str(Path(inFiles[0]).parent)
        copy_mesh_files(inFolder, outFolder)

    # Copy unchanged frames (0 to startFrame-1)
    if copyUnchanged:
        print(f"Copying {startFrame} unchanged frames...")
        for iFile in tqdm.tqdm(range(min(startFrame, len(inFiles))), desc="Copying"):
            meshFile = inFiles[iFile]
            meshFP = Path(meshFile)
            outFile = join(outFolder, meshFP.stem + ".npy")
            
            # Load and save (to ensure consistent format)
            data = np.load(meshFile)
            np.save(outFile, data.astype(np.float32))

    stablePos = np.load(inFiles[stabilizingFrame])  # Shape: (N, 3)
    
    print(f"Stabilizing frames {startFrame} to {len(inFiles)-1}...")
    for iFile in tqdm.tqdm(range(startFrame, len(inFiles)), desc="Stabilizing"):
        meshFile = inFiles[iFile]
        steps = iFile - startFrame

        wCurState = np.power(stabilizingSpeed, steps)
        if wCurState < minBlending:
            wCurState = minBlending

        wStabilizer = 1 - wCurState

        curPos = np.load(meshFile)  # Shape: (N, 3)
        
        blendedPos = curPos * wCurState + stablePos * wStabilizer

        meshFP = Path(meshFile)
        outFile = join(outFolder, meshFP.stem + ".npy")
        np.save(outFile, blendedPos.astype(np.float32))


# =============================================================================
# PLY Stabilizers (using pyvista)
# =============================================================================

def linearStabilizer(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder):
    """Linear stabilizer for PLY files."""
    os.makedirs(outFolder, exist_ok=True)

    meshStable = pv.PolyData(inFiles[stabilizingFrame])
    stablePos = np.array(meshStable.points)
    
    for iFile in tqdm.tqdm(range(startFrame, len(inFiles))):
        meshFile = inFiles[iFile]
        steps = iFile - startFrame

        wCurState = 1 - steps * stabilizingSpeed
        if wCurState < minBlending:
            wCurState = minBlending

        wStabilizer = 1 - wCurState

        meshCurFrame = pv.PolyData(meshFile)

        print(f"Frame: {iFile}, wCurState: {wCurState:.4f}, wStabilizer: {wStabilizer:.4f}")
        meshCurFrame.points = np.array(meshCurFrame.points) * wCurState + stablePos * wStabilizer

        meshFP = Path(meshFile)
        outFile = join(outFolder, meshFP.stem + ".ply")
        meshCurFrame.save(outFile, binary=True)


def exponentialStabilizer(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder):
    """Exponential stabilizer for PLY files."""
    os.makedirs(outFolder, exist_ok=True)

    meshStable = pv.PolyData(inFiles[stabilizingFrame])
    stablePos = np.array(meshStable.points)
    
    for iFile in tqdm.tqdm(range(startFrame, len(inFiles))):
        meshFile = inFiles[iFile]
        steps = iFile - startFrame

        wCurState = np.power(stabilizingSpeed, steps)
        if wCurState < minBlending:
            wCurState = minBlending

        wStabilizer = 1 - wCurState

        meshCurFrame = pv.PolyData(meshFile)

        print(f"Frame: {iFile}, wCurState: {wCurState:.4f}, wStabilizer: {wStabilizer:.4f}")
        meshCurFrame.points = np.array(meshCurFrame.points) * wCurState + stablePos * wStabilizer

        meshFP = Path(meshFile)
        outFile = join(outFolder, meshFP.stem + ".ply")
        meshCurFrame.save(outFile, binary=True)

# =============================================================================
# Binary Stabilizers (flat float32 arrays)
# =============================================================================

def linearStabilizerBinary(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder):
    """Linear stabilizer for raw binary float32 files."""
    os.makedirs(outFolder, exist_ok=True)

    stablePos = np.fromfile(inFiles[stabilizingFrame], dtype=np.float32)
    
    for iFile in tqdm.tqdm(range(startFrame, len(inFiles))):
        meshFile = inFiles[iFile]
        steps = iFile - startFrame

        wCurState = 1 - steps * stabilizingSpeed
        if wCurState < minBlending:
            wCurState = minBlending

        wStabilizer = 1 - wCurState

        meshCurFrame = np.fromfile(meshFile, dtype=np.float32)

        print(f"Frame: {iFile}, wCurState: {wCurState:.4f}, wStabilizer: {wStabilizer:.4f}")
        meshCurFrame = meshCurFrame * wCurState + stablePos * wStabilizer

        meshFP = Path(meshFile)
        outFile = join(outFolder, meshFP.stem + ".bin")
        
        meshCurFrameArray = array.array('f', meshCurFrame)
        with open(outFile, 'wb') as of:
            meshCurFrameArray.tofile(of)


def exponentialStabilizerBinary(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder):
    """Exponential stabilizer for raw binary float32 files."""
    os.makedirs(outFolder, exist_ok=True)

    stablePos = np.fromfile(inFiles[stabilizingFrame], dtype=np.float32)
    
    for iFile in tqdm.tqdm(range(startFrame, len(inFiles))):
        meshFile = inFiles[iFile]
        steps = iFile - startFrame

        wCurState = np.power(stabilizingSpeed, steps)
        if wCurState < minBlending:
            wCurState = minBlending

        wStabilizer = 1 - wCurState

        meshCurFrame = np.fromfile(meshFile, dtype=np.float32)

        print(f"Frame: {iFile}, wCurState: {wCurState:.4f}, wStabilizer: {wStabilizer:.4f}")
        meshCurFrame = meshCurFrame * wCurState + stablePos * wStabilizer

        meshFP = Path(meshFile)
        outFile = join(outFolder, meshFP.stem + ".bin")
        
        meshCurFrameArray = array.array('f', meshCurFrame)
        with open(outFile, 'wb') as of:
            meshCurFrameArray.tofile(of)

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    # Input folder
    inFolder = r"D:\Data\DAT_Sim\ClothDrop\200_layers\20260106_052909"
    
    # File format: "npy", "bin", or "ply"
    file_format = "npy"
    
    # Stabilization parameters
    startFrame = 200          # Frame to start blending
    stabilizingFrame = 250    # Frame to use as stable target (fully converged to this by stabilizingFrame)
    minBlending = 0.02        # Minimum weight for current state (0.02 = 2%)
    
    # Stabilization mode: "linear" or "exponential"
    mode = "linear"
    
    # Copy unchanged frames to make output self-contained
    copyUnchanged = True
    
    # -------------------------------------------------------------------------
    # Generate output folder name with parameters
    # -------------------------------------------------------------------------
    
    # Output folder name: {inFolder}_stabilized_{mode}_s{start}_t{target}_min{minBlend}
    outFolderName = f"stabilized_{mode}_s{startFrame}_t{stabilizingFrame}_min{int(minBlending*100)}pct"
    outFolder = join(Path(inFolder).parent, Path(inFolder).name + "_" + outFolderName)
    
    print("=" * 60)
    print("Stabilization Configuration:")
    print(f"  Input:           {inFolder}")
    print(f"  Output:          {outFolder}")
    print(f"  Mode:            {mode}")
    print(f"  Start frame:     {startFrame}")
    print(f"  Target frame:    {stabilizingFrame}")
    print(f"  Min blending:    {minBlending*100:.1f}%")
    print(f"  Copy unchanged:  {copyUnchanged}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Compute stabilizing speed and run
    # -------------------------------------------------------------------------
    
    # For linear: stabilizingSpeed = (1 - minBlending) / (stabilizingFrame - startFrame)
    # For exponential: stabilizingSpeed = minBlending^(1 / (stabilizingFrame - startFrame))
    if mode == "linear":
        stabilizingSpeed = (1 - minBlending) / (stabilizingFrame - startFrame)
    else:
        stabilizingSpeed = np.power(minBlending, 1 / (stabilizingFrame - startFrame))
    
    # Get input files
    inFiles = sorted(glob.glob(join(inFolder, f"*.{file_format}")))
    print(f"Found {len(inFiles)} {file_format} files.")
    
    if len(inFiles) == 0:
        print(f"Error: No {file_format} files found in {inFolder}")
        exit(1)
    
    if stabilizingFrame >= len(inFiles):
        print(f"Error: stabilizingFrame ({stabilizingFrame}) >= number of files ({len(inFiles)})")
        exit(1)
    
    # Run appropriate stabilizer
    if file_format == "npy":
        if mode == "linear":
            linearStabilizerNpy(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder, copyUnchanged)
        else:
            exponentialStabilizerNpy(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder, copyUnchanged)
    elif file_format == "bin":
        if mode == "linear":
            linearStabilizerBinary(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder)
        else:
            exponentialStabilizerBinary(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder)
    elif file_format == "ply":
        if mode == "linear":
            linearStabilizer(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder)
        else:
            exponentialStabilizer(inFiles, startFrame, stabilizingFrame, stabilizingSpeed, minBlending, outFolder)
    else:
        print(f"Error: Unknown file format '{file_format}'")
        exit(1)
    
    print("=" * 60)
    print(f"Done! Output saved to: {outFolder}")
    print(f"Total files: {len(inFiles)}")
    print("=" * 60)