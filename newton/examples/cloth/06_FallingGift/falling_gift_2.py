# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Falling Gift Demo - Comparison Study

Runs three experiments with different collision handling modes:
1. Truncation mode 0 (Isotropic-DAT)
2. Truncation mode 1 (Planar-DAT)
3. No collision handling

Exports individual videos and merges them into a 1x3 side-by-side comparison.
"""

import os
import subprocess
import numpy as np
import warp as wp
import cv2

from os.path import join

import sys
from pathlib import Path

# Add parent directory to path for M01_Simulator import
sys.path.insert(0, str(Path(__file__).parent.parent))
from M01_Simulator import Simulator, default_config


# =============================================================================
# Geometry Helpers
# =============================================================================

def cloth_loop_around_box(
    hx=1.6,            # half-size in X (box width / 2)
    hz=2.0,            # half-size in Z (box height / 2)
    width=0.25,        # strap width (along Y)
    center_y=0.0,      # Y position of the strap center
    nu=120,            # resolution along loop
    nv=6,              # resolution across strap width
):
    """
    Vertical closed cloth loop wrapped around a cuboid.
    Loop lies in X-Z plane, strap width is along Y.
    Z is up.
    """
    verts = []
    faces = []

    # Rectangle perimeter length
    P = 4.0 * (hx + hz)

    for i in range(nu):
        s = (i / nu) * P

        # Walk rectangle in X–Z plane (counter-clockwise)
        if s < 2 * hx:
            x = -hx + s
            z = -hz
        elif s < 2 * hx + 2 * hz:
            x = hx
            z = -hz + (s - 2 * hx)
        elif s < 4 * hx + 2 * hz:
            x = hx - (s - (2 * hx + 2 * hz))
            z = hz
        else:
            x = -hx
            z = hz - (s - (4 * hx + 2 * hz))

        for j in range(nv):
            v = (j / (nv - 1) - 0.5) * width
            y = center_y + v
            verts.append([x, y, z])

    def idx(i, j):
        return (i % nu) * nv + j

    # Triangulation
    for i in range(nu):
        for j in range(nv - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            faces.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    return (
        np.array(verts, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )


PYRAMID_TET_INDICES = np.array(
    [
        [0, 1, 3, 9],
        [1, 4, 3, 13],
        [1, 3, 9, 13],
        [3, 9, 13, 12],
        [1, 9, 10, 13],
        [1, 2, 4, 10],
        [2, 5, 4, 14],
        [2, 4, 10, 14],
        [4, 10, 14, 13],
        [2, 10, 11, 14],
        [3, 4, 6, 12],
        [4, 7, 6, 16],
        [4, 6, 12, 16],
        [6, 12, 16, 15],
        [4, 12, 13, 16],
        [4, 5, 7, 13],
        [5, 8, 7, 17],
        [5, 7, 13, 17],
        [7, 13, 17, 16],
        [5, 13, 14, 17],
    ],
    dtype=np.int32,
)

PYRAMID_PARTICLES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (2.0, 1.0, 0.0),
    (0.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
    (2.0, 2.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (2.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0),
    (0.0, 2.0, 1.0),
    (1.0, 2.0, 1.0),
    (2.0, 2.0, 1.0),
]

# Number of vertices per soft body block
VERTS_PER_BLOCK = len(PYRAMID_PARTICLES)  # 18


# =============================================================================
# Falling Gift Simulator
# =============================================================================

class FallingGiftSimulator(Simulator):
    """
    Simulation of four stacked soft body blocks with two cloth straps.
    """

    def __init__(self, config: dict | None = None):
        # Store geometry for later use
        self.base_height = 30.0
        self.spacing = 1.01  # small gap to avoid initial penetration
        
        # Generate cloth geometry
        self.strap1_verts, self.strap1_faces = cloth_loop_around_box(
            hx=1.01, hz=2.02, width=0.6
        )
        self.strap2_verts, self.strap2_faces = cloth_loop_around_box(
            hx=1.015, hz=2.025, width=0.6
        )
        
        self.strap1_count = len(self.strap1_verts)
        self.strap2_count = len(self.strap2_verts)
        
        # Call parent init (this calls custom_init)
        super().__init__(config)

    def custom_init(self):
        """Add soft body blocks and cloth straps to the simulation."""
        
        # Add 4 stacked soft body blocks
        for i in range(4):
            self.builder.add_soft_mesh(
                pos=wp.vec3(0.0, 0.0, self.base_height + i * self.spacing),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=PYRAMID_PARTICLES,
                indices=PYRAMID_TET_INDICES.flatten().tolist(),
                density=100,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=1e-5,
            )
        
        # Add first cloth strap
        self.builder.add_cloth_mesh(
            pos=wp.vec3(1.0, 1.0, self.base_height + 1.5 * self.spacing + 0.5),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=self.strap1_verts,
            indices=self.strap1_faces.flatten().tolist(),
            density=0.02,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-5,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=0.05,
        )
        
        # Add second cloth strap (rotated 90 degrees)
        self.builder.add_cloth_mesh(
            pos=wp.vec3(1.0, 1.0, self.base_height + 1.5 * self.spacing + 0.5),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=self.strap2_verts,
            indices=self.strap2_faces.flatten().tolist(),
            density=0.02,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-5,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=0.05,
        )

    def custom_finalize(self):
        """Extract face indices for each mesh type."""
        # Get box faces (first portion of tri_indices, excluding cloth)
        all_faces = self.model.tri_indices.numpy()
        cloth_face_count = len(self.strap1_faces) + len(self.strap2_faces)
        box_faces_all = all_faces[:len(all_faces) - cloth_face_count]
        
        # Each box has the same faces (just first quarter)
        self.box_faces = box_faces_all[:len(box_faces_all) // 4]
        
        # Vertex layout:
        # [0:18]    - Box 1
        # [18:36]   - Box 2
        # [36:54]   - Box 3
        # [54:72]   - Box 4
        # [72:72+strap1_count] - Strap 1
        # [72+strap1_count:]   - Strap 2
        self.box_start = 0
        self.strap1_start = 4 * VERTS_PER_BLOCK
        self.strap2_start = self.strap1_start + self.strap1_count

    def setup_polyscope_meshes(self):
        """Register individual meshes for visualization."""
        if not self.do_rendering:
            return
        
        import polyscope as ps
        
        all_verts = self.model.particle_q.numpy()
        
        # Register boxes as volume meshes
        self.ps_box1 = ps.register_volume_mesh(
            "Box1", all_verts[0:18], tets=PYRAMID_TET_INDICES
        )
        self.ps_box2 = ps.register_volume_mesh(
            "Box2", all_verts[18:36], tets=PYRAMID_TET_INDICES
        )
        self.ps_box3 = ps.register_volume_mesh(
            "Box3", all_verts[36:54], tets=PYRAMID_TET_INDICES
        )
        self.ps_box4 = ps.register_volume_mesh(
            "Box4", all_verts[54:72], tets=PYRAMID_TET_INDICES
        )
        
        # Register cloth straps as surface meshes
        self.register_ps_mesh(
            name="Strap1",
            vertices=all_verts[self.strap1_start:self.strap1_start + self.strap1_count],
            faces=self.strap1_faces,
            vertex_indices=slice(self.strap1_start, self.strap1_start + self.strap1_count),
            color=(1.0, 0.0, 0.0),
        )
        self.register_ps_mesh(
            name="Strap2",
            vertices=all_verts[self.strap2_start:],
            faces=self.strap2_faces,
            vertex_indices=slice(self.strap2_start, None),
            color=(1.0, 0.0, 0.0),
        )
        
        # Set box colors
        box_color = (0.0, 0.2, 0.125)
        self.ps_box1.set_color(box_color)
        self.ps_box2.set_color(box_color)
        self.ps_box3.set_color(box_color)
        self.ps_box4.set_color(box_color)

    def update_ps_meshes(self):
        """Update all meshes with current positions."""
        all_verts = self.state_0.particle_q.numpy()
        
        # Update boxes
        self.ps_box1.update_vertex_positions(all_verts[0:18])
        self.ps_box2.update_vertex_positions(all_verts[18:36])
        self.ps_box3.update_vertex_positions(all_verts[36:54])
        self.ps_box4.update_vertex_positions(all_verts[54:72])
        
        # Update cloth straps (handled by parent class via ps_meshes registry)
        super().update_ps_meshes()

    def save_initial_meshes(self):
        """Save initial mesh topology for each component separately."""
        if self.output_path is None:
            return
        
        all_verts = self.model.particle_q.numpy()
        
        def write_ply(path, vertices, faces):
            with open(path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                for v in vertices:
                    f.write(f"{v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        # Save each mesh separately
        write_ply(join(self.output_path, "initial_cloth1.ply"),
                  all_verts[self.strap1_start:self.strap1_start + self.strap1_count],
                  self.strap1_faces)
        write_ply(join(self.output_path, "initial_cloth2.ply"),
                  all_verts[self.strap2_start:],
                  self.strap2_faces)
        write_ply(join(self.output_path, "initial_box1.ply"), all_verts[0:18], self.box_faces)
        write_ply(join(self.output_path, "initial_box2.ply"), all_verts[18:36], self.box_faces)
        write_ply(join(self.output_path, "initial_box3.ply"), all_verts[36:54], self.box_faces)
        write_ply(join(self.output_path, "initial_box4.ply"), all_verts[54:72], self.box_faces)
        
        print(f"Initial meshes saved to: {self.output_path}")


# =============================================================================
# Experiment Runner
# =============================================================================

def get_base_config():
    """Get base configuration shared across all experiments."""
    return {
        **default_config,
        "name": "falling_gift",
        "up_axis": "z",
        "gravity": -10,
        "fps": 60,
        "sim_substeps": 10,
        "iterations": 15,
        "sim_num_frames": 800,
        # Self-contact settings (will be overridden per experiment)
        "handle_self_contact": True,
        "self_contact_radius": 0.04,
        "self_contact_margin": 0.06,
        "topological_contact_filter_threshold": 1,
        "truncation_mode": 1,
        # Contact physics
        "soft_contact_ke": 1.0e5,
        "soft_contact_kd": 1e-5,
        "soft_contact_mu": 0.5,
        # Ground
        "has_ground": True,
        "ground_height": 0.0,
        "show_ground_plane": True,
        # Camera (from Polyscope Ctrl+C)
        "camera_json": {
            "farClipRatio": 20.0,
            "fov": 45.0,
            "nearClipRatio": 0.005,
            "projectionMode": "Perspective",
            "viewMat": [
                0.00993161741644144, -0.999950230121613, 2.50074894125873e-09, 1.13012027740479,
                0.218161955475807, 0.00216674711555243, 0.97590959072113, -7.91851997375488,
                -0.975861549377441, -0.00969239696860313, 0.218173131346703, -61.6613006591797,
                0.0, 0.0, 0.0, 1.0
            ],
        },
        # Output (will be configured per experiment)
        "output_path": None,
        "output_ext": "npy",
        "write_output": False,  # Disabled for now
        "write_video": True,
        # Visualization
        "do_rendering": True,
        "is_initially_paused": False,
    }


def run_experiment(experiment_name: str, config_overrides: dict, output_dir: str):
    """
    Run a single experiment with the given configuration.
    
    Args:
        experiment_name: Name for this experiment (used in output folder)
        config_overrides: Configuration overrides for this experiment
        output_dir: Base output directory
    
    Returns:
        Path to the output video file
    """
    import polyscope as ps
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*60}")
    
    config = get_base_config()
    config.update(config_overrides)
    config["experiment_name"] = experiment_name
    config["output_path"] = output_dir
    config["output_timestamp"] = False  # Use fixed folder names for merging
    
    try:
        # Create and run simulator
        sim = FallingGiftSimulator(config)
        sim.finalize()
        
        print(f"[INFO] Starting {experiment_name}...")
        
        sim.simulate()
    except KeyboardInterrupt:
        print(f"\n[INFO] {experiment_name} interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {experiment_name} failed with exception:")
        import traceback
        traceback.print_exc()
    
    # Return path to the video
    video_path = join(output_dir, experiment_name, "video.mp4")
    return video_path


def merge_videos_side_by_side(video_paths: list, labels: list, output_path: str):
    """
    Merge multiple videos side by side with labels.
    
    Args:
        video_paths: List of paths to input videos
        labels: List of labels for each video
        output_path: Path for the output merged video
    """
    print(f"\n{'='*60}")
    print("Merging videos side by side...")
    print(f"{'='*60}")
    
    # Check all videos exist
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Warning: Video not found: {path}")
            return
    
    # Try using ffmpeg first (better quality)
    try:
        # Build ffmpeg filter for horizontal stack with labels
        n = len(video_paths)
        inputs = " ".join([f'-i "{p}"' for p in video_paths])
        
        # Create filter complex for horizontal stacking with padding and labels
        filter_parts = []
        
        # Add labels to each video
        for i, label in enumerate(labels):
            filter_parts.append(
                f"[{i}:v]drawtext=text='{label}':fontsize=24:fontcolor=white:"
                f"x=(w-text_w)/2:y=20:box=1:boxcolor=black@0.5:boxborderw=5[v{i}]"
            )
        
        # Stack horizontally
        stack_inputs = "".join([f"[v{i}]" for i in range(n)])
        filter_parts.append(f"{stack_inputs}hstack=inputs={n}[out]")
        
        filter_complex = ";".join(filter_parts)
        
        cmd = f'ffmpeg -y {inputs} -filter_complex "{filter_complex}" -map "[out]" -c:v libx264 -crf 18 "{output_path}"'
        
        print(f"Running ffmpeg command...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Merged video saved to: {output_path}")
            return
        else:
            print(f"ffmpeg failed: {result.stderr}")
            print("Falling back to OpenCV...")
    except Exception as e:
        print(f"ffmpeg error: {e}")
        print("Falling back to OpenCV...")
    
    # Fallback: use OpenCV
    merge_videos_opencv(video_paths, labels, output_path)


def merge_videos_opencv(video_paths: list, labels: list, output_path: str):
    """
    Merge videos side by side using OpenCV (fallback method).
    """
    # Open all videos
    caps = [cv2.VideoCapture(p) for p in video_paths]
    
    # Get video properties from first video
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    total_width = width * len(video_paths)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, height))
    
    print(f"Merging {len(video_paths)} videos ({width}x{height} each) into {total_width}x{height}...")
    
    for frame_idx in range(frame_count):
        frames = []
        all_read = True
        
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                all_read = False
                break
            frames.append(frame)
        
        if not all_read:
            break
        
        # Add labels to frames
        for i, (frame, label) in enumerate(zip(frames, labels)):
            cv2.putText(frame, label, (width//2 - len(label)*6, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, label, (width//2 - len(label)*6, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Stack horizontally
        merged = np.hstack(frames)
        out.write(merged)
    
    # Cleanup
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"Merged video saved to: {output_path}")


def run_comparison_study(output_dir: str, write_results: bool = False):
    """
    Run all three experiments and merge videos.
    
    Args:
        output_dir: Base output directory for all experiments
        write_results: Whether to export result files (ply/npy)
    """
    import polyscope as ps
    from datetime import datetime
    
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped subfolder for this comparison run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = join(output_dir, f"comparison_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Saving experiments to: {run_dir}")
    
    # Define experiments
    experiments = [
        {
            "name": "truncation_mode_0",
            "label": "Isotropic-DAT (Mode 0)",
            "config": {
                "handle_self_contact": True,
                "truncation_mode": 0,
                "write_output": write_results,
            }
        },
        {
            "name": "truncation_mode_1",
            "label": "Planar-DAT (Mode 1)",
            "config": {
                "handle_self_contact": True,
                "truncation_mode": 1,
                "write_output": write_results,
            }
        },
        {
            "name": "no_collision",
            "label": "No Collision",
            "config": {
                "handle_self_contact": False,
                "truncation_mode": 1,  # Doesn't matter when collision is off
                "write_output": write_results,
            }
        },
    ]
    
    video_paths = []
    labels = []
    
    # Run each experiment
    for i, exp in enumerate(experiments):
        print(f"\n[INFO] Experiment {i+1}/{len(experiments)}: {exp['name']}")
        
        # Shutdown polyscope before starting new experiment (except for first)
        if i > 0:
            try:
                ps.shutdown()
            except Exception as e:
                print(f"[WARN] Could not shutdown polyscope: {e}")
        
        video_path = run_experiment(exp["name"], exp["config"], run_dir)
        video_paths.append(video_path)
        labels.append(exp["label"])
    
    # Shutdown polyscope after all experiments
    try:
        ps.shutdown()
    except:
        pass
    
    # Merge videos
    merged_output = join(run_dir, "comparison_1x3.mp4")
    merge_videos_side_by_side(video_paths, labels, merged_output)
    
    print(f"\n{'='*60}")
    print("Comparison study complete!")
    print(f"{'='*60}")
    print(f"Output folder: {run_dir}")
    print(f"Individual videos:")
    for path, label in zip(video_paths, labels):
        print(f"  - {label}: {path}")
    print(f"Merged video: {merged_output}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Falling Gift Comparison Study")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="falling_gift_comparison",
        help="Output directory for videos and results"
    )
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="Export result files (ply/npy) for each experiment"
    )
    parser.add_argument(
        "--single",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Run only a single experiment (0=isotropic, 1=planar, 2=no collision)"
    )
    
    args = parser.parse_args()
    
    if args.single is not None:
        # Run single experiment
        import polyscope as ps
        from datetime import datetime
        
        experiments = [
            ("truncation_mode_0", {"handle_self_contact": True, "truncation_mode": 0}),
            ("truncation_mode_1", {"handle_self_contact": True, "truncation_mode": 1}),
            ("no_collision", {"handle_self_contact": False, "truncation_mode": 1}),
        ]
        
        name, config = experiments[args.single]
        config["write_output"] = args.write_results
        
        # Create base output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create timestamped subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = join(args.output_dir, f"{name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        run_experiment(name, config, run_dir)
        ps.shutdown()
    else:
        # Run full comparison study
        try:
            run_comparison_study(args.output_dir, write_results=args.write_results)
        except Exception as e:
            print(f"\n[ERROR] Comparison study failed:")
            import traceback
            traceback.print_exc()
