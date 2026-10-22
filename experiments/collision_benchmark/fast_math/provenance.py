# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the frozen physics while permitting benchmark packaging commits."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash a source or evidence file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics_paths(root: Path) -> list[Path]:
    """List Newton Python core files and the shipped Franka recipe used here."""
    paths = [
        path
        for path in (root / "newton").rglob("*.py")
        if not {"tests", "examples"}.intersection(path.relative_to(root / "newton").parts)
    ]
    for name in (
        "newton/examples/__init__.py",
        "newton/examples/softbody/example_softbody_franka.py",
        "newton/tests/unittest_utils.py",
    ):
        recipe = root / name
        if recipe.is_file():
            paths.append(recipe)
    return sorted(paths)


def verify_physics(root: Path, manifest: dict, *, warp_version: str) -> dict:
    """Reject changed physics, compiler versions, or an unrelated Git history."""
    if warp_version != manifest["warp_version"]:
        raise ValueError(f"Warp {manifest['warp_version']} is required; found {warp_version}")
    base = manifest["base_revision"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestor.returncode:
        raise ValueError(f"Frozen physics base {base} must be an ancestor of HEAD; retain the repository history")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    actual = {str(path.relative_to(root)): sha256(path) for path in physics_paths(root)}
    expected = manifest["files_sha256"]
    if actual.keys() != expected.keys():
        raise ValueError(f"Frozen physics inventory changed: {sorted(actual.keys() ^ expected.keys())}")
    changed = [path for path, digest in actual.items() if expected[path] != digest]
    if changed:
        raise ValueError(f"Frozen physics source changed: {changed}")
    return {"revision": revision, "base_revision": base, "files_sha256": actual, "warp_version": warp_version}


def validate_output(out: Path) -> None:
    """Accept any empty output directory without replacing prior evidence."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {out}")
