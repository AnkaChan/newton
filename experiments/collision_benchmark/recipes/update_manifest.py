# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Refresh packaged recipe byte hashes after formatting, preserving source semantics.

Run ``python -m experiments.collision_benchmark.recipes.update_manifest`` from
the checkout. This refuses changes to Python syntax trees (apart from import
ordering) or JSON values. Original source hashes remain unchanged.
"""

import ast
import hashlib
import json
from pathlib import Path


def semantic_digest(path):
    """Hash recipe syntax or JSON values independently of formatting."""
    path = Path(path)
    if path.suffix == ".py":
        tree = ast.parse(path.read_text())
        # Ruff can sort an import block; it must not alter executable recipes.
        index = 0
        while index < len(tree.body):
            if not isinstance(tree.body[index], (ast.Import, ast.ImportFrom)):
                index += 1
                continue
            end = index + 1
            while end < len(tree.body) and isinstance(tree.body[end], (ast.Import, ast.ImportFrom)):
                end += 1
            tree.body[index:end] = sorted(tree.body[index:end], key=ast.dump)
            index = end
        payload = ast.dump(tree, include_attributes=False)
    else:
        payload = json.dumps(json.loads(path.read_text()), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest_path = root / "h1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for case_id, record in manifest["cases"].items():
        for relative in record["files"]:
            path = root / "h1" / case_id / relative
            if semantic_digest(path) != record["original_semantics_sha256"][relative]:
                raise ValueError(f"Recipe semantics changed: {path}; byte hashes were not refreshed")
            record["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    registry_path = root.parent / "cases.json"
    registry = json.loads(registry_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for case in registry["cases"]:
        if case["case_id"] in manifest["cases"]:
            case["replay"]["manifest.json"]["sha256"] = digest
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    print("Recipe byte hashes refreshed; original syntax trees and JSON values preserved")


if __name__ == "__main__":
    main()
