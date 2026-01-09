"""
Generate a bullet mesh that matches the rifled barrel caliber.
"""

import numpy as np


def generate_bullet_ply(
    bullet_radius: float = 0.0098,    # Slightly less than bore (0.01) to fit
    body_length: float = 0.015,       # Cylindrical body length
    nose_length: float = 0.012,       # Ogive nose length
    base_taper: float = 0.002,        # Slight taper at base
    radial_segments: int = 60,        # Segments around circumference
    nose_segments: int = 30,          # Segments along nose
    body_segments: int = 10,          # Segments along body
    z_offset: float = -0.015,         # Position: negative = nose inside barrel
    output_path: str = "bullet.ply"
):
    """
    Generate a bullet mesh with ogive nose and save as PLY.
    
    The bullet is positioned so its nose is inside the barrel (z=0 is barrel entrance).
    """
    
    vertices = []
    faces = []
    
    total_length = nose_length + body_length
    tip_z = z_offset + total_length
    
    # Single continuous smooth nose curve (no separate dome)
    tip_rings = 8  # Rings at the very tip for good topology
    total_nose_rings = tip_rings + nose_segments
    
    # Tip center point
    vertices.append((0.0, 0.0, tip_z))
    tip_center_idx = 0
    
    # Generate entire nose as one continuous elliptical/round profile
    # Using parametric ellipse: from tip (t=0) to body junction (t=1)
    for i in range(1, total_nose_rings + 1):
        t = i / total_nose_rings  # 0 to 1
        
        # Elliptical profile for smooth round nose
        # r = bullet_radius * sin(t * pi/2)
        # z = tip_z - nose_length * (1 - cos(t * pi/2))
        angle = t * (np.pi / 2)
        r = bullet_radius * np.sin(angle)
        z = tip_z - nose_length * (1 - np.cos(angle))
        
        for j in range(radial_segments):
            theta = (j / radial_segments) * 2 * np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            vertices.append((x, y, z))
    
    # Generate body (cylinder with slight base taper)
    for i in range(body_segments + 1):
        t = i / body_segments  # 0 to 1 from nose-body junction to base
        z = z_offset + body_length - t * body_length
        
        # Slight taper at the very base
        if t > 0.7:
            taper_t = (t - 0.7) / 0.3
            r = bullet_radius * (1.0 - taper_t * (base_taper / bullet_radius))
        else:
            r = bullet_radius
        
        for j in range(radial_segments):
            theta = (j / radial_segments) * 2 * np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            vertices.append((x, y, z))
    
    # Helper to get vertex index for nose rings (0 = first ring after tip center)
    def get_nose_ring_vertex(ring_idx, radial_idx):
        """Get vertex for nose rings (0 = first ring after center)"""
        radial_idx = radial_idx % radial_segments
        return 1 + ring_idx * radial_segments + radial_idx
    
    # Helper to get vertex index for body rings (after nose)
    def get_body_ring_vertex(ring_idx, radial_idx):
        """Get vertex index for body ring (0 = first body ring after nose)"""
        radial_idx = radial_idx % radial_segments
        return 1 + total_nose_rings * radial_segments + ring_idx * radial_segments + radial_idx
    
    total_body_rings = body_segments + 1
    
    # Generate base cap with more concentric rings (better topology)
    base_z = z_offset
    base_rings = 8  # More rings for better topology
    last_body_ring = total_body_rings - 1
    
    # Get the radius at the base (accounting for taper)
    base_radius = bullet_radius * (1.0 - base_taper / bullet_radius)
    
    # Store where base vertices start
    base_start_idx = len(vertices)
    
    # Add concentric ring vertices for base (from outer to inner)
    for i in range(1, base_rings):
        ring_radius = base_radius * (1.0 - i / base_rings)
        for j in range(radial_segments):
            theta = (j / radial_segments) * 2 * np.pi
            x = ring_radius * np.cos(theta)
            y = ring_radius * np.sin(theta)
            vertices.append((x, y, base_z))
    
    # Base center point
    vertices.append((0.0, 0.0, base_z))
    base_center_idx = len(vertices) - 1
    
    def get_base_ring_vertex(ring_idx, radial_idx):
        """Get vertex for base rings (0 = outermost = last body ring)"""
        radial_idx = radial_idx % radial_segments
        if ring_idx == 0:
            return get_body_ring_vertex(last_body_ring, radial_idx)
        else:
            return base_start_idx + (ring_idx - 1) * radial_segments + radial_idx
    
    # Tip center to first nose ring
    for j in range(radial_segments):
        v0 = tip_center_idx
        v1 = get_nose_ring_vertex(0, j)
        v2 = get_nose_ring_vertex(0, j + 1)
        faces.append((v0, v1, v2))
    
    # Nose rings (continuous curve)
    for i in range(total_nose_rings - 1):
        for j in range(radial_segments):
            v0 = get_nose_ring_vertex(i, j)
            v1 = get_nose_ring_vertex(i, j + 1)
            v2 = get_nose_ring_vertex(i + 1, j + 1)
            v3 = get_nose_ring_vertex(i + 1, j)
            faces.append((v0, v2, v1))
            faces.append((v0, v3, v2))
    
    # Connect nose to body (reversed winding for outward normals)
    for j in range(radial_segments):
        v0 = get_nose_ring_vertex(total_nose_rings - 1, j)
        v1 = get_nose_ring_vertex(total_nose_rings - 1, j + 1)
        v2 = get_body_ring_vertex(0, j + 1)
        v3 = get_body_ring_vertex(0, j)
        faces.append((v0, v2, v1))
        faces.append((v0, v3, v2))
    
    # Body triangles (connect consecutive rings) - reversed winding for outward normals
    for i in range(total_body_rings - 1):
        for j in range(radial_segments):
            v0 = get_body_ring_vertex(i, j)
            v1 = get_body_ring_vertex(i, j + 1)
            v2 = get_body_ring_vertex(i + 1, j + 1)
            v3 = get_body_ring_vertex(i + 1, j)
            faces.append((v0, v2, v1))
            faces.append((v0, v3, v2))
    
    # Base cap with concentric rings (winding for -z facing outward normals)
    for i in range(base_rings - 1):
        for j in range(radial_segments):
            v0 = get_base_ring_vertex(i, j)
            v1 = get_base_ring_vertex(i, j + 1)
            v2 = get_base_ring_vertex(i + 1, j + 1)
            v3 = get_base_ring_vertex(i + 1, j)
            # Different triangle split for correct winding
            faces.append((v0, v1, v3))
            faces.append((v1, v2, v3))
    
    # Inner ring to center point (small triangles at very center)
    inner_ring = base_rings - 1
    for j in range(radial_segments):
        v0 = get_base_ring_vertex(inner_ring, j)
        v1 = get_base_ring_vertex(inner_ring, j + 1)
        v2 = base_center_idx
        faces.append((v1, v2, v0))
    
    # Write PLY file
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Bullet mesh - caliber {bullet_radius * 2 * 1000:.1f}mm\n")
        f.write(f"comment Total length: {total_length * 1000:.1f}mm\n")
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
    
    print(f"Generated bullet PLY: {output_path}")
    print(f"  - Bullet radius: {bullet_radius * 1000:.2f}mm (diameter: {bullet_radius * 2 * 1000:.2f}mm)")
    print(f"  - Bore diameter: 20mm (inner_radius=0.01)")
    print(f"  - Total length: {total_length * 1000:.1f}mm")
    print(f"  - Nose in barrel: {-z_offset * 1000:.1f}mm")
    print(f"  - Vertices: {len(vertices)}")
    print(f"  - Triangles: {len(faces)}")
    
    return vertices, faces


if __name__ == "__main__":
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate handgun bullet - round nose style
    generate_bullet_ply(
        bullet_radius=0.0095,     # 9.5mm radius = 19mm diameter (fits in 20mm bore)
        body_length=0.016,        # 16mm body
        nose_length=0.010,        # 10mm smooth round nose (shorter, rounder)
        base_taper=0.0008,        # Slight base taper
        radial_segments=60,
        nose_segments=25,         # Smooth nose resolution
        body_segments=10,
        z_offset=-0.018,          # Position with nose inside barrel
        output_path=os.path.join(script_dir, "bullet.ply")
    )
