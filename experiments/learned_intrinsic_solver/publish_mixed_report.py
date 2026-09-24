# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental durable report-only mirror for a separately launched campaign.

Run this module in its own tmux session. It never starts training and never
copies checkpoints, raw rank logs or trajectory state into the public tree.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .mixed_report import write_mixed_report

__all__ = ["prepare_public_report"]


def prepare_public_report(run, output, *, training_running=None, seen_training=False):
    """Render only allowlisted metrics into a publication staging directory.

    Experimental. The training directory is read-only and may not yet exist.
    Publication heartbeats and source training timestamps remain distinct.

    Args:
        run: Private training output directory.
        output: Staging directory exclusively for public report files.
        training_running: Whether the independently managed tmux session exists.
        seen_training: Whether this mirror has observed that session running.
    """
    run, output = Path(run), Path(output)
    source = run / "report.json"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if source.is_file():
        report = json.loads(source.read_text())
        metrics_time = report.get("updated_at") or datetime.fromtimestamp(
            source.stat().st_mtime, timezone.utc
        ).isoformat(timespec="seconds")
    else:
        report = {
            "format": "mixed_pool_v2",
            "config": {"max_epochs": 500, "validation_iterations": 100},
            "completed_epochs": 0,
            "completed_updates": 0,
            "epochs": [],
            "updates": [],
            "status": "initializing" if training_running else "preparing",
        }
        metrics_time = "Waiting for first epoch metrics"
    progress_file = run / "progress.json"
    report["progress"] = json.loads(progress_file.read_text()) if progress_file.is_file() else {}
    failure_file = run / "failure.json"
    if failure_file.is_file():
        report["failure"] = json.loads(failure_file.read_text())
        report["status"] = "failed"
    elif seen_training and training_running is False and report["status"] in ("running", "initializing", "preparing"):
        report["status"] = "interrupted"
        report["failure"] = {
            "error": "Training session stopped before reporting completion. Check the private rank logs."
        }
    report["publication"] = {"updated_at": now, "training_session_running": training_running}
    write_mixed_report(output, report, updated_at=metrics_time)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--slug", default="learned-intrinsic-training-v2")
    parser.add_argument("--base-url", default="https://ankachen.com")
    parser.add_argument("--training-session", default="learned-intrinsic-v2-damping")
    parser.add_argument("--interval", type=float, default=45.0)
    parser.add_argument("--kanna-dist")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--publisher-script",
        type=Path,
        default=Path.home() / ".codex/skills/publish-artifact/scripts/publish_artifact.py",
    )
    args = parser.parse_args()
    if not 10 <= args.interval <= 300:
        parser.error("interval must be between 10 and 300 seconds")
    spec = importlib.util.spec_from_file_location("_artifact_publisher", args.publisher_script)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    if not publisher.SLUG_RE.fullmatch(args.slug):
        parser.error("slug must contain only letters, digits, dots, underscores and hyphens")
    target = publisher.discover_kanna_dist(args.kanna_dist) / "artifacts" / args.slug
    url = publisher.artifact_url(args.base_url.rstrip("."), args.slug, "index.html")
    print(f"Public report: {url}", flush=True)
    seen_training = False
    while True:
        try:
            training_running = (
                subprocess.run(
                    ["tmux", "has-session", "-t", f"={args.training_session}"], capture_output=True, check=False
                ).returncode
                == 0
            )
            seen_training |= training_running
            with tempfile.TemporaryDirectory(prefix="mixed-training-public-") as directory:
                stage = Path(directory)
                report = prepare_public_report(
                    args.run, stage, training_running=training_running, seen_training=seen_training
                )
                publisher.copy_source(stage, target, "index.html", False, False)
            print(
                f"{report['publication']['updated_at']} published status={report['status']} epoch={report['completed_epochs']} updates={report['progress'].get('completed_updates', report['completed_updates'])}",
                flush=True,
            )
        except Exception as error:
            if args.once:
                raise
            print(f"Publication retry: {type(error).__name__}: {error}", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
