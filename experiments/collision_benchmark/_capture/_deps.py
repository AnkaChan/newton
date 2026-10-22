# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check optional recording dependencies without changing the environment."""

import importlib.util


def ensure_imageio_ffmpeg() -> bool:
    """Return whether the ffmpeg Python package is installed."""
    return importlib.util.find_spec("imageio_ffmpeg") is not None
