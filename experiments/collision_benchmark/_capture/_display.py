# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manage Xvfb virtual display for headless rendering."""

import atexit
import os
import signal
import subprocess
import time


class VirtualDisplay:
    """Start and manage an Xvfb virtual display.

    Used as a context manager or standalone. Automatically cleans up the Xvfb
    process on exit.
    """

    def __init__(self, width: int = 1920, height: int = 1080, depth: int = 24):
        self.width = width
        self.height = height
        self.depth = depth
        self._process: subprocess.Popen | None = None
        self._display: str | None = None
        self._old_display: str | None = None

    @property
    def display(self) -> str | None:
        return self._display

    def start(self) -> str:
        """Start Xvfb on a free display number. Returns the display string (e.g. ':99')."""
        if self._process is not None:
            return self._display

        display_num = self._find_free_display()
        self._display = f":{display_num}"

        self._process = subprocess.Popen(
            [
                "Xvfb",
                self._display,
                "-screen",
                "0",
                f"{self.width}x{self.height}x{self.depth}",
                "-ac",  # disable access control
                "+extension",
                "GLX",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for Xvfb to be ready
        for _ in range(50):
            if self._process.poll() is not None:
                raise RuntimeError(f"Xvfb exited immediately (display {self._display})")
            lockfile = f"/tmp/.X{display_num}-lock"
            if os.path.exists(lockfile):
                break
            time.sleep(0.1)

        self._old_display = os.environ.get("DISPLAY")
        os.environ["DISPLAY"] = self._display
        atexit.register(self.stop)
        return self._display

    def stop(self):
        """Stop Xvfb and restore the original DISPLAY."""
        if self._process is None:
            return

        try:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                self._process.kill()
                self._process.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

        self._process = None

        # Restore original DISPLAY
        if self._old_display is not None:
            os.environ["DISPLAY"] = self._old_display
        elif "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]
        self._old_display = None

        try:
            atexit.unregister(self.stop)
        except Exception:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    @staticmethod
    def _find_free_display(start: int = 99, end: int = 199) -> int:
        for num in range(start, end):
            lockfile = f"/tmp/.X{num}-lock"
            socket = f"/tmp/.X11-unix/X{num}"
            if not os.path.exists(lockfile) and not os.path.exists(socket):
                return num
        raise RuntimeError(f"No free display found in :{start}-:{end}")


def needs_virtual_display() -> bool:
    """Check whether we need to start Xvfb."""
    return not os.environ.get("DISPLAY")
