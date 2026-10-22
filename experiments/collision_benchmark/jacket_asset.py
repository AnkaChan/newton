# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Load the traced native Style3D jacket without lossy mesh simplification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.collision_benchmark.asset_paths import STYLE3D_REVISION, newton_asset_path
from experiments.collision_benchmark.assets import AssetMesh, _cached_mesh, _style3d_record, validate_asset
from experiments.collision_benchmark.jacket_diagnostics import mesh_statistics

__all__ = ["load_native_jacket"]


def load_native_jacket():
    """Return original source triangles and metre, Z-up vertices, with provenance."""
    folder = Path(__file__).with_name("assets")
    path = folder / "cached_jacket_native.npz"
    meta_path = path.with_suffix(".metadata.json")
    if path.exists() and meta_path.exists():
        data = np.load(path)
        metadata = json.loads(meta_path.read_text())
        return AssetMesh("cached_jacket_native", data["vertices"], data["triangles"], metadata=metadata)
    directory = newton_asset_path("style3d", revision=STYLE3D_REVISION)
    record = _style3d_record(directory, "h1_jacket")
    source = _cached_mesh(record)
    vertices = np.asarray(source.vertices, dtype=np.float32)
    triangles = np.asarray(source.triangles, dtype=np.int32)
    digest = hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest()
    metadata = {
        "name": "cached_jacket_native",
        "generator_revision": "cached-jacket-native-v2",
        "geometry_sha256": digest,
        "source": record["source"],
        "source_revision": record["source_revision"],
        "source_tree_sha256": record["source_tree_sha256"],
        "source_entrypoint": record["entrypoint"],
        "license": record["license"],
        "license_notice": record["license_notice"],
        "preparation": "Native authored vertices and triangles; local-to-world transform, Y-up to Z-up, bottom translated to zero. No decimation, welding, hole filling, or smoothing.",
        "units": "metres",
        "statistics": mesh_statistics(vertices, triangles),
        "color_rgb": [0.68, 0.27, 0.18],
        "material_mapping": "Synthetic Newton coefficients; Style3D material data are not transferred.",
    }
    result = AssetMesh("cached_jacket_native", vertices, triangles, metadata=metadata)
    metadata["validation"] = validate_asset(result)
    if not metadata["validation"]["valid"]:
        raise ValueError("Native cached jacket failed topology validation")
    np.savez_compressed(path, vertices=vertices, triangles=triangles)
    metadata["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return result
