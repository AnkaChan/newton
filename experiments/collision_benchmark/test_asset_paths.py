# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test portable, revision-pinned asset resolution without network access."""

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestAssetPaths(unittest.TestCase):
    """Keep asset locations independent of the original author's cache."""

    def setUp(self):
        """Require the portable resolver before exercising its consumers."""
        spec = importlib.util.find_spec("experiments.collision_benchmark.asset_paths")
        self.assertIsNotNone(spec, "The portable asset resolver is not implemented")
        self.paths = importlib.import_module("experiments.collision_benchmark.asset_paths")

    def test_newton_download_uses_pinned_revision_and_returned_directory(self):
        """Read the downloaded tree from an arbitrary cache location."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "relocated" / "franka_emika_panda"
            folder.mkdir(parents=True)
            (folder / "LICENSE").write_text("source notice", encoding="utf-8")
            with patch("newton.utils.download_asset", return_value=folder) as download:
                result = self.paths.newton_asset_path("franka_emika_panda")
            self.assertEqual((result / "LICENSE").read_text(), "source notice")
            download.assert_called_once_with("franka_emika_panda", ref="f8fb7abcbeba2318814a74f3eeb02780ad7925d6")

    def test_style3d_keeps_the_native_jacket_source_revision(self):
        """Use the jacket's older authored source instead of Newton's default."""
        with patch("newton.utils.download_asset", return_value=Path("portable/style3d")) as download:
            self.paths.newton_asset_path("style3d", revision=self.paths.STYLE3D_REVISION)
        download.assert_called_once_with("style3d", ref="26c7f5ccaadd9a163f4d6c8f867c789b3ac57d83")

    def test_optional_menagerie_download_preserves_repository_and_revision(self):
        """Resolve the original optional gallery source through the public downloader."""
        with patch("newton.examples.download_external_git_folder", return_value=Path("portable/leap_hand")) as download:
            result = self.paths.menagerie_asset_path("leap_hand")
        self.assertEqual(result, Path("portable/leap_hand"))
        download.assert_called_once_with(
            "https://github.com/google-deepmind/mujoco_menagerie.git",
            "leap_hand",
            ref="da76818e269b82289eba39808e2fb91d679d6994",
        )

    def test_inventory_respects_explicit_cache_and_newton_environment(self):
        """Use the explicit cache before the documented environment override."""
        with patch.dict(os.environ, {"NEWTON_CACHE_PATH": "/temporary/newton-cache"}):
            self.assertEqual(self.paths.asset_cache_directory(), Path("/temporary/newton-cache"))
            self.assertEqual(self.paths.asset_cache_directory("/explicit/cache"), Path("/explicit/cache"))

    def test_inventory_does_not_substitute_another_garment_revision(self):
        """Keep the authored jacket when another Style3D revision is cached."""
        cached_inventory = importlib.import_module("experiments.collision_benchmark.assets").cached_inventory

        with tempfile.TemporaryDirectory() as temporary:
            for suffix, contents in (("26c7f5cc", "original jacket"), ("f8fb7abc", "different jacket")):
                checkout = Path(temporary) / f"newton-assets_style3d_identity_{suffix}"
                (checkout / ".git").mkdir(parents=True)
                (checkout / ".git/HEAD").write_text(suffix + "\n", encoding="utf-8")
                folder = checkout / "style3d"
                (folder / "garments").mkdir(parents=True)
                (folder / "LICENSE").write_text("source notice", encoding="utf-8")
                (folder / "garments/h1_jacket.usd").write_text(contents, encoding="utf-8")
            record = next(r for r in cached_inventory(temporary) if r["name"] == "h1_jacket")
            self.assertEqual(Path(record["entrypoint"]["path"]).read_text(), "original jacket")

    def test_importing_hand_adapters_does_not_download(self):
        """Keep module import free of external asset access."""
        with patch("newton.utils.download_asset", side_effect=AssertionError("Unexpected download during import")):
            for name in ("robot_gripper", "allegro_gripper", "h1_gripper"):
                importlib.reload(importlib.import_module(f"experiments.collision_benchmark.{name}"))


if __name__ == "__main__":
    unittest.main()
