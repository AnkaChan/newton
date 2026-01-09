"""
Tetrahedralize the bullet mesh using TetGen.
Outputs a tetrahedral mesh that can be used for FEM simulation.
"""

import numpy as np
import os
import subprocess


def load_ply(filepath):
    """Load vertices and faces from a PLY file."""
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        assert line == "ply", f"Not a PLY file: {filepath}"
        
        vertex_count = 0
        face_count = 0
        in_header = True
        
        while in_header:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line == "end_header":
                in_header = False
        
        for _ in range(vertex_count):
            line = f.readline().strip()
            coords = [float(x) for x in line.split()[:3]]
            vertices.append(coords)
        
        for _ in range(face_count):
            line = f.readline().strip()
            parts = [int(x) for x in line.split()]
            n = parts[0]
            face_indices = parts[1:n+1]
            if n == 3:
                faces.append(face_indices)
            elif n == 4:
                faces.append([face_indices[0], face_indices[1], face_indices[2]])
                faces.append([face_indices[0], face_indices[2], face_indices[3]])
    
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def write_poly_file(filepath, vertices, faces):
    """Write TetGen .poly file format."""
    with open(filepath, 'w') as f:
        # Part 1: Node list
        f.write(f"{len(vertices)} 3 0 0\n")
        for i, v in enumerate(vertices):
            f.write(f"{i+1} {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        
        # Part 2: Facet list
        f.write(f"{len(faces)} 0\n")
        for face in faces:
            f.write(f"1\n")  # 1 polygon per facet
            f.write(f"3 {face[0]+1} {face[1]+1} {face[2]+1}\n")  # 1-indexed
        
        # Part 3: Hole list
        f.write("0\n")
        
        # Part 4: Region list
        f.write("0\n")
    
    print(f"Wrote {filepath}")


def read_tetgen_output(base_path):
    """Read TetGen output files (.node, .ele)."""
    # Read nodes
    node_file = base_path + ".1.node"
    vertices = []
    with open(node_file, 'r') as f:
        header = f.readline().split()
        num_nodes = int(header[0])
        for _ in range(num_nodes):
            parts = f.readline().split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    vertices = np.array(vertices, dtype=np.float64)
    
    # Read elements (tetrahedra)
    ele_file = base_path + ".1.ele"
    tets = []
    with open(ele_file, 'r') as f:
        header = f.readline().split()
        num_tets = int(header[0])
        for _ in range(num_tets):
            parts = f.readline().split()
            # TetGen uses 1-indexed, convert to 0-indexed
            tets.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1, int(parts[4])-1])
    tets = np.array(tets, dtype=np.int32)
    
    return vertices, tets


def save_tetmesh_npz(filepath, vertices, tets, surface_faces=None):
    """Save tetrahedral mesh as NPZ file."""
    data = {'vertices': vertices, 'tetrahedra': tets}
    if surface_faces is not None:
        data['surface_faces'] = surface_faces
    np.savez(filepath, **data)
    print(f"Saved tetmesh to {filepath}")


def save_tetmesh_vtk(filepath, vertices, tets):
    """Save tetrahedral mesh as VTK file for visualization."""
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Tetrahedral mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {len(vertices)} float\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        num_tets = len(tets)
        f.write(f"CELLS {num_tets} {num_tets * 5}\n")
        for tet in tets:
            f.write(f"4 {tet[0]} {tet[1]} {tet[2]} {tet[3]}\n")
        
        f.write(f"CELL_TYPES {num_tets}\n")
        for _ in range(num_tets):
            f.write("10\n")  # VTK_TETRA = 10
    
    print(f"Saved VTK to {filepath}")


def save_tetmesh_ply(filepath, vertices, tets):
    """Save tetrahedral mesh as PLY with tet info in comments."""
    # Extract surface faces
    face_count = {}
    for tet in tets:
        faces_of_tet = [
            (tet[0], tet[2], tet[1]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[3], tet[2]),
            (tet[1], tet[2], tet[3]),
        ]
        for face in faces_of_tet:
            key = tuple(sorted(face))
            if key not in face_count:
                face_count[key] = [0, face]
            face_count[key][0] += 1
    
    surface_faces = [face for key, (count, face) in face_count.items() if count == 1]
    
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Tetrahedral mesh: {len(tets)} tetrahedra\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(surface_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        for face in surface_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Saved surface PLY to {filepath}")
    return np.array(surface_faces, dtype=np.int32)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    input_ply = os.path.join(script_dir, "bullet.ply")
    poly_file = os.path.join(script_dir, "bullet.poly")
    tetgen_exe = os.path.join(script_dir, "tetgen.exe")
    output_npz = os.path.join(script_dir, "bullet_tetmesh.npz")
    output_ply = os.path.join(script_dir, "bullet_tetmesh.ply")
    output_vtk = os.path.join(script_dir, "bullet_tetmesh.vtk")
    
    # Load surface mesh
    print(f"Loading {input_ply}...")
    vertices, faces = load_ply(input_ply)
    print(f"Input: {len(vertices)} vertices, {len(faces)} triangles")
    
    # Write .poly file for TetGen
    write_poly_file(poly_file, vertices, faces)
    
    # Run TetGen
    # -p: tetrahedralize PLC
    # -q: quality mesh (adds Steiner points)
    # -a: max tet volume constraint (optional)
    print(f"\nRunning TetGen...")
    cmd = [tetgen_exe, "-pq1.2", poly_file]
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        # Read TetGen output
        base_path = os.path.join(script_dir, "bullet")
        tet_vertices, tet_indices = read_tetgen_output(base_path)
        print(f"\nOutput: {len(tet_vertices)} vertices, {len(tet_indices)} tetrahedra")
        
        # Save outputs
        surface_faces = save_tetmesh_ply(output_ply, tet_vertices, tet_indices)
        save_tetmesh_npz(output_npz, tet_vertices, tet_indices, surface_faces)
        save_tetmesh_vtk(output_vtk, tet_vertices, tet_indices)
        
        print("\nDone! Files created:")
        print(f"  - {output_npz}")
        print(f"  - {output_ply}")
        print(f"  - {output_vtk}")
    else:
        print(f"TetGen failed with return code {result.returncode}")
