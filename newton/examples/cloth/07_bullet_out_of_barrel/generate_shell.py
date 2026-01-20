"""
Generate a bullet shell casing mesh - simple version without neck.
"""

import numpy as np


def generate_shell_ply(
    shell_radius: float = 0.0105,  # Outer radius of shell body
    shell_length: float = 0.032,  # Length of shell casing
    rim_radius: float = 0.011,  # Radius of rim at base (smaller)
    rim_thickness: float = 0.002,  # Thickness of rim
    wall_thickness: float = 0.0008,  # Wall thickness of shell
    top_radius: float = 0.0096,  # Radius at top (to embrace bullet)
    taper_length: float = 0.008,  # Length of taper transition at top
    radial_segments: int = 60,
    length_segments: int = 25,
    z_offset: float = -0.036,
    output_path: str = "shell.ply",
):
    """
    Generate a simple shell casing mesh (rim + straight body) and save as PLY.
    """

    vertices = []
    faces = []

    base_z = z_offset
    top_z = z_offset + shell_length
    rim_top_z = base_z + rim_thickness
    body_length = shell_length - rim_thickness
    inner_radius = shell_radius - wall_thickness
    inner_top_radius = top_radius - wall_thickness

    # ===== OUTER SURFACE =====

    # Rim bottom ring (outer edge)
    rim_bottom_outer_start = len(vertices)
    for j in range(radial_segments):
        theta = (j / radial_segments) * 2 * np.pi
        vertices.append((rim_radius * np.cos(theta), rim_radius * np.sin(theta), base_z))

    # Rim top ring (outer edge)
    rim_top_outer_start = len(vertices)
    for j in range(radial_segments):
        theta = (j / radial_segments) * 2 * np.pi
        vertices.append((rim_radius * np.cos(theta), rim_radius * np.sin(theta), rim_top_z))

    # Body rings (from rim top to shell top) with smooth taper at top
    body_start_idx = len(vertices)
    taper_start_z = top_z - taper_length

    for i in range(length_segments + 1):
        t = i / length_segments
        z = rim_top_z + t * body_length

        # Smooth taper at the top
        if z > taper_start_z:
            taper_t = (z - taper_start_z) / taper_length
            # Smoothstep for nice curve
            smooth_t = taper_t * taper_t * (3 - 2 * taper_t)
            r = shell_radius + (top_radius - shell_radius) * smooth_t
        else:
            r = shell_radius

        for j in range(radial_segments):
            theta = (j / radial_segments) * 2 * np.pi
            vertices.append((r * np.cos(theta), r * np.sin(theta), z))

    # ===== INNER SURFACE =====

    # Inner top ring
    inner_top_start = len(vertices)
    for j in range(radial_segments):
        theta = (j / radial_segments) * 2 * np.pi
        vertices.append((inner_top_radius * np.cos(theta), inner_top_radius * np.sin(theta), top_z))

    # Inner body rings (from top to rim) with taper
    inner_body_start = len(vertices)
    for i in range(length_segments, -1, -1):
        t = i / length_segments
        z = rim_top_z + t * body_length

        # Match outer taper
        if z > taper_start_z:
            taper_t = (z - taper_start_z) / taper_length
            smooth_t = taper_t * taper_t * (3 - 2 * taper_t)
            r = inner_radius + (inner_top_radius - inner_radius) * smooth_t
        else:
            r = inner_radius

        for j in range(radial_segments):
            theta = (j / radial_segments) * 2 * np.pi
            vertices.append((r * np.cos(theta), r * np.sin(theta), z))

    # Inner bottom ring (at rim_top_z level)
    inner_bottom_ring_start = len(vertices)
    inner_bottom_z = rim_top_z + 0.001
    for j in range(radial_segments):
        theta = (j / radial_segments) * 2 * np.pi
        vertices.append((inner_radius * np.cos(theta), inner_radius * np.sin(theta), inner_bottom_z))

    # Inner bottom center
    vertices.append((0.0, 0.0, inner_bottom_z))
    inner_center_idx = len(vertices) - 1

    # Rim bottom inner ring (for bottom face)
    rim_bottom_inner_start = len(vertices)
    for j in range(radial_segments):
        theta = (j / radial_segments) * 2 * np.pi
        vertices.append((shell_radius * np.cos(theta), shell_radius * np.sin(theta), base_z))

    # Rim bottom center
    vertices.append((0.0, 0.0, base_z))
    rim_center_idx = len(vertices) - 1

    # ===== GENERATE FACES =====

    def idx(start, ring, j):
        return start + ring * radial_segments + (j % radial_segments)

    # Rim outer side
    for j in range(radial_segments):
        v0 = rim_bottom_outer_start + j
        v1 = rim_bottom_outer_start + (j + 1) % radial_segments
        v2 = rim_top_outer_start + (j + 1) % radial_segments
        v3 = rim_top_outer_start + j
        faces.append((v0, v1, v2))
        faces.append((v0, v2, v3))

    # Rim top face (from rim edge to body)
    for j in range(radial_segments):
        v0 = rim_top_outer_start + j
        v1 = rim_top_outer_start + (j + 1) % radial_segments
        v2 = body_start_idx + (j + 1) % radial_segments
        v3 = body_start_idx + j
        faces.append((v0, v2, v1))
        faces.append((v0, v3, v2))

    # Body outer faces
    for i in range(length_segments):
        for j in range(radial_segments):
            v0 = idx(body_start_idx, i, j)
            v1 = idx(body_start_idx, i, j + 1)
            v2 = idx(body_start_idx, i + 1, j + 1)
            v3 = idx(body_start_idx, i + 1, j)
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))

    # Top lip (outer top to inner top)
    outer_top_ring = body_start_idx + length_segments * radial_segments
    for j in range(radial_segments):
        v0 = outer_top_ring + j
        v1 = outer_top_ring + (j + 1) % radial_segments
        v2 = inner_top_start + (j + 1) % radial_segments
        v3 = inner_top_start + j
        faces.append((v0, v1, v2))
        faces.append((v0, v2, v3))

    # Inner surface faces
    total_inner_rings = length_segments + 2
    for i in range(total_inner_rings - 1):
        for j in range(radial_segments):
            v0 = idx(inner_top_start, i, j)
            v1 = idx(inner_top_start, i, j + 1)
            v2 = idx(inner_top_start, i + 1, j + 1)
            v3 = idx(inner_top_start, i + 1, j)
            faces.append((v0, v2, v1))
            faces.append((v0, v3, v2))

    # Inner bottom face (last ring to bottom ring)
    last_inner_ring_start = inner_top_start + (total_inner_rings - 1) * radial_segments
    for j in range(radial_segments):
        v0 = last_inner_ring_start + j
        v1 = last_inner_ring_start + (j + 1) % radial_segments
        v2 = inner_bottom_ring_start + (j + 1) % radial_segments
        v3 = inner_bottom_ring_start + j
        faces.append((v0, v2, v1))
        faces.append((v0, v3, v2))

    # Inner bottom to center
    for j in range(radial_segments):
        v0 = inner_bottom_ring_start + j
        v1 = inner_bottom_ring_start + (j + 1) % radial_segments
        v2 = inner_center_idx
        faces.append((v0, v2, v1))

    # Rim bottom face (outer to inner)
    for j in range(radial_segments):
        v0 = rim_bottom_outer_start + j
        v1 = rim_bottom_outer_start + (j + 1) % radial_segments
        v2 = rim_bottom_inner_start + (j + 1) % radial_segments
        v3 = rim_bottom_inner_start + j
        faces.append((v0, v3, v2))
        faces.append((v0, v2, v1))

    # Rim bottom center
    for j in range(radial_segments):
        v0 = rim_bottom_inner_start + j
        v1 = rim_bottom_inner_start + (j + 1) % radial_segments
        v2 = rim_center_idx
        faces.append((v0, v2, v1))

    # Write PLY
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Shell casing mesh (no neck)\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Generated shell: {output_path}")
    print(f"  - Length: {shell_length * 1000:.1f}mm, Radius: {shell_radius * 1000:.2f}mm")
    print(f"  - Vertices: {len(vertices)}, Triangles: {len(faces)}")

    return vertices, faces


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))

    generate_shell_ply(
        shell_radius=0.0105,  # Larger shell body
        shell_length=0.032,
        rim_radius=0.011,  # Smaller rim
        rim_thickness=0.002,
        wall_thickness=0.0008,
        top_radius=0.0096,  # Tapers to embrace bullet (9.5mm radius)
        taper_length=0.008,  # Smooth 8mm transition
        radial_segments=60,
        length_segments=30,  # More segments for smooth taper
        z_offset=-0.036,
        output_path=os.path.join(script_dir, "shell.ply"),
    )
