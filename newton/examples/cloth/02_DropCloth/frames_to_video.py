import cv2
import os
import glob


frames_dir = r"C:\Users\youss\Downloads\newton-divide_and_truncate\newton-divide_and_truncate\ply_frames\Rendering_subdiv"
output_video = "planar_cloth_drop.mp4"
fps = 60


frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

if not frame_paths:
    raise RuntimeError("No PNG frames found!")


first_frame = cv2.imread(frame_paths[0])
height, width, _ = first_frame.shape



fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
video = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)


for path in frame_paths:
    frame = cv2.imread(path)

    if frame is None:
        print(f"Warning: could not read {path}")
        continue

    if frame.shape[0] != height or frame.shape[1] != width:
        raise ValueError(f"Frame size mismatch: {path}")

    video.write(frame)

video.release()
print(f"Video saved as {output_video}")
