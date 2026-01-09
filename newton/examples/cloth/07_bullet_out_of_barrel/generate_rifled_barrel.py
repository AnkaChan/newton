"""
Generate a rifled barrel mesh with helical grooves.
The barrel has 6 rifling grooves that spiral along its length.
"""

import numpy as np


def generate_rifled_barrel(
    length: float = 0.3,  # Barrel length in meters
    outer_radius: float = 0.02,  # Outer radius
    inner_radius: float = 0.01,  # Inner bore radius (lands)
    groove_depth: float = 0.002,  # Depth of rifling grooves
    num_grooves: int = 6,  # Number of rifling grooves
    twist_rate: float = 2.0,  # Number of full rotations over barrel length
    groove_width_ratio: float = 0.4,  # Ratio of groove width to land width
    radial_segments: int = 72,  # Segments around circumference
    length_segments: int = 100,  # Segments along length
    output_path: str = "rifled_barrel.obj",
):
    """
    Generate a rifled barrel mesh and save as OBJ.

    The rifling creates a helical pattern of grooves (deeper cuts) and lands
    (raised portions) on the interior surface of the barrel.
    """

    vertices = []
    faces = []

    # Generate vertices
    for i in range(length_segments + 1):
        z = (i / length_segments) * length
        twist_angle = (i / length_segments) * twist_rate * 2 * np.pi

        for j in range(radial_segments):
            theta = (j / radial_segments) * 2 * np.pi

            # Outer surface vertex
            x_outer = outer_radius * np.cos(theta)
            y_outer = outer_radius * np.sin(theta)
            vertices.append((x_outer, y_outer, z))

            # Inner surface vertex with rifling (distinct grooves cut into cylinder)
            effective_angle = theta + twist_angle

            # Create distinct grooves: flat cylinder with flat-bottomed rectangular cuts
            segment_angle = 2 * np.pi / num_grooves
            pos_in_segment = (effective_angle % segment_angle) / segment_angle

            # Groove occupies middle 50% of each segment
            groove_start = 0.25
            groove_end = 0.75
            transition = 0.08  # Wider transition for smooth rounded edges

            if pos_in_segment < groove_start - transition:
                depth_factor = 0.0  # Land (cylinder wall)
            elif pos_in_segment < groove_start + transition:
                t = (pos_in_segment - groove_start + transition) / (2 * transition)
                # Smoother quintic interpolation instead of cubic
                depth_factor = t * t * t * (t * (t * 6 - 15) + 10)
            elif pos_in_segment < groove_end - transition:
                depth_factor = 1.0  # Groove bottom (flat)
            elif pos_in_segment < groove_end + transition:
                t = (pos_in_segment - groove_end + transition) / (2 * transition)
                depth_factor = 1.0 - t * t * t * (t * (t * 6 - 15) + 10)
            else:
                depth_factor = 0.0  # Land (cylinder wall)

            current_inner_radius = inner_radius + groove_depth * depth_factor

            x_inner = current_inner_radius * np.cos(theta)
            y_inner = current_inner_radius * np.sin(theta)
            vertices.append((x_inner, y_inner, z))

    # Generate faces
    # Each length segment has radial_segments quads for outer and inner surfaces
    # Plus end caps

    def get_vertex_index(length_idx, radial_idx, is_inner):
        """Get vertex index for given position."""
        radial_idx = radial_idx % radial_segments
        base = length_idx * radial_segments * 2 + radial_idx * 2
        return base + (1 if is_inner else 0)

    # Outer surface faces (normals pointing outward)
    for i in range(length_segments):
        for j in range(radial_segments):
            v0 = get_vertex_index(i, j, False)
            v1 = get_vertex_index(i, j + 1, False)
            v2 = get_vertex_index(i + 1, j + 1, False)
            v3 = get_vertex_index(i + 1, j, False)
            faces.append((v0, v1, v2, v3))

    # Inner surface faces (normals pointing inward, so reverse winding)
    for i in range(length_segments):
        for j in range(radial_segments):
            v0 = get_vertex_index(i, j, True)
            v1 = get_vertex_index(i, j + 1, True)
            v2 = get_vertex_index(i + 1, j + 1, True)
            v3 = get_vertex_index(i + 1, j, True)
            faces.append((v0, v3, v2, v1))  # Reversed winding

    # Front end cap (z = 0)
    for j in range(radial_segments):
        v0_outer = get_vertex_index(0, j, False)
        v1_outer = get_vertex_index(0, j + 1, False)
        v1_inner = get_vertex_index(0, j + 1, True)
        v0_inner = get_vertex_index(0, j, True)
        faces.append((v0_outer, v0_inner, v1_inner, v1_outer))

    # Back end cap (z = length)
    for j in range(radial_segments):
        v0_outer = get_vertex_index(length_segments, j, False)
        v1_outer = get_vertex_index(length_segments, j + 1, False)
        v1_inner = get_vertex_index(length_segments, j + 1, True)
        v0_inner = get_vertex_index(length_segments, j, True)
        faces.append((v0_outer, v1_outer, v1_inner, v0_inner))

    # Write OBJ file
    with open(output_path, "w") as f:
        f.write("# Rifled barrel mesh\n")
        f.write(f"# {num_grooves} rifling grooves, {twist_rate} twist(s) over {length}m\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        for face in faces:
            # OBJ uses 1-based indexing
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1} {face[3] + 1}\n")

    print(f"Generated rifled barrel: {output_path}")
    print(f"  - Length: {length}m")
    print(f"  - Outer radius: {outer_radius}m")
    print(f"  - Inner radius (bore): {inner_radius}m")
    print(f"  - Groove depth: {groove_depth}m")
    print(f"  - Number of grooves: {num_grooves}")
    print(f"  - Twist rate: {twist_rate} rotations")
    print(f"  - Vertices: {len(vertices)}")
    print(f"  - Faces: {len(faces)}")

    return vertices, faces


def generate_rifled_barrel_ply(
    length: float = 0.3,
    outer_radius: float = 0.02,
    inner_radius: float = 0.01,
    groove_depth: float = 0.002,
    num_grooves: int = 6,
    twist_rate: float = 2.0,
    groove_width_ratio: float = 0.4,
    radial_segments: int = 72,
    length_segments: int = 100,
    output_path: str = "rifled_barrel.ply",
):
    """
    Generate a rifled barrel mesh with triangulated faces and save as PLY.
    """

    vertices = []
    faces = []

    # Generate vertices (same as quad version)
    for i in range(length_segments + 1):
        z = (i / length_segments) * length
        twist_angle = (i / length_segments) * twist_rate * 2 * np.pi

        for j in range(radial_segments):
            theta = (j / radial_segments) * 2 * np.pi

            # Outer surface vertex
            x_outer = outer_radius * np.cos(theta)
            y_outer = outer_radius * np.sin(theta)
            vertices.append((x_outer, y_outer, z))

            # Inner surface vertex with rifling (distinct grooves cut into cylinder)
            effective_angle = theta + twist_angle

            # Create distinct grooves: flat cylinder with flat-bottomed rectangular cuts
            segment_angle = 2 * np.pi / num_grooves
            pos_in_segment = (effective_angle % segment_angle) / segment_angle

            # Groove occupies middle 50% of each segment
            groove_start = 0.25
            groove_end = 0.75
            transition = 0.08  # Wider transition for smooth rounded edges

            if pos_in_segment < groove_start - transition:
                depth_factor = 0.0  # Land (cylinder wall)
            elif pos_in_segment < groove_start + transition:
                t = (pos_in_segment - groove_start + transition) / (2 * transition)
                # Smoother quintic interpolation instead of cubic
                depth_factor = t * t * t * (t * (t * 6 - 15) + 10)
            elif pos_in_segment < groove_end - transition:
                depth_factor = 1.0  # Groove bottom (flat)
            elif pos_in_segment < groove_end + transition:
                t = (pos_in_segment - groove_end + transition) / (2 * transition)
                depth_factor = 1.0 - t * t * t * (t * (t * 6 - 15) + 10)
            else:
                depth_factor = 0.0  # Land (cylinder wall)

            current_inner_radius = inner_radius + groove_depth * depth_factor

            x_inner = current_inner_radius * np.cos(theta)
            y_inner = current_inner_radius * np.sin(theta)
            vertices.append((x_inner, y_inner, z))

    def get_vertex_index(length_idx, radial_idx, is_inner):
        radial_idx = radial_idx % radial_segments
        base = length_idx * radial_segments * 2 + radial_idx * 2
        return base + (1 if is_inner else 0)

    # Outer surface faces (triangulated)
    for i in range(length_segments):
        for j in range(radial_segments):
            v0 = get_vertex_index(i, j, False)
            v1 = get_vertex_index(i, j + 1, False)
            v2 = get_vertex_index(i + 1, j + 1, False)
            v3 = get_vertex_index(i + 1, j, False)
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))

    # Inner surface faces (triangulated, reversed winding)
    for i in range(length_segments):
        for j in range(radial_segments):
            v0 = get_vertex_index(i, j, True)
            v1 = get_vertex_index(i, j + 1, True)
            v2 = get_vertex_index(i + 1, j + 1, True)
            v3 = get_vertex_index(i + 1, j, True)
            faces.append((v0, v2, v1))
            faces.append((v0, v3, v2))

    # Front end cap (triangulated)
    for j in range(radial_segments):
        v0_outer = get_vertex_index(0, j, False)
        v1_outer = get_vertex_index(0, j + 1, False)
        v1_inner = get_vertex_index(0, j + 1, True)
        v0_inner = get_vertex_index(0, j, True)
        faces.append((v0_outer, v0_inner, v1_inner))
        faces.append((v0_outer, v1_inner, v1_outer))

    # Back end cap (triangulated)
    for j in range(radial_segments):
        v0_outer = get_vertex_index(length_segments, j, False)
        v1_outer = get_vertex_index(length_segments, j + 1, False)
        v1_inner = get_vertex_index(length_segments, j + 1, True)
        v0_inner = get_vertex_index(length_segments, j, True)
        faces.append((v0_outer, v1_outer, v1_inner))
        faces.append((v0_outer, v1_inner, v0_inner))

    # Write PLY file
    with open(output_path, "w") as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Rifled barrel mesh\n")
        f.write(f"comment {num_grooves} rifling grooves, {twist_rate} twist(s) over {length}m\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertices
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Faces (PLY uses 0-based indexing)
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Generated rifled barrel PLY: {output_path}")
    print(f"  - Length: {length}m")
    print(f"  - Outer radius: {outer_radius}m")
    print(f"  - Inner radius (bore): {inner_radius}m")
    print(f"  - Groove depth: {groove_depth}m")
    print(f"  - Number of grooves: {num_grooves}")
    print(f"  - Twist rate: {twist_rate} rotations")
    print(f"  - Vertices: {len(vertices)}")
    print(f"  - Triangles: {len(faces)}")

    return vertices, faces


if __name__ == "__main__":
    import os

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate PLY version (triangulated, best for physics simulations)
    # Higher radial_segments = smoother inner surface
    generate_rifled_barrel_ply(
        length=0.3,
        outer_radius=0.02,
        inner_radius=0.01,
        groove_depth=0.002,  # Depth of the grooves cut into the bore
        num_grooves=6,
        twist_rate=2.0,
        radial_segments=120,  # High resolution for smooth surface
        length_segments=400,  # High resolution along length for smooth helix
        output_path=os.path.join(script_dir, "rifled_barrel.ply"),
    )
