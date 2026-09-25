# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental read-only campaign watchdog with queued Codex investigation.

Run in a persistent tmux session. This module never kills workers, changes
training settings, or restarts a checkpoint. It queues checks for the existing
agent conversation, which investigates failures under the user's instructions.
Queued messages wait for that conversation's active turn to finish.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["check_training_health", "has_pending_monitor_message"]

_MESSAGE_MARKER = "[LIDO-v2 monitor]"
_TERMINAL_STATUSES = {"complete", "completed", "plateau_converged", "stalled", "epoch_limit", "max_epochs", "finished"}
_PROC = Path("/proc")


def _read_object(path, issues):
    try:
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError("expected an object")
        return result
    except (OSError, ValueError) as error:
        issues.append(f"Cannot read {path.name}: {error}")
        return {}


def _age(path, now):
    try:
        return max(0.0, now - path.stat().st_mtime)
    except OSError:
        return None


def check_training_health(run, *, workers, dashboard, now=None, expected_workers=4, stale_seconds=2700):
    """Inspect campaign files and supplied live-process observations without writes.

    Experimental. A stale observation requests investigation; it never proves
    a deadlock. Checkpoint age adapts to three recent epoch durations, allowing
    expensive validation and preparation. All time quantities are seconds.
    """
    run = Path(run)
    now = time.time() if now is None else now
    issues = []
    progress = _read_object(run / "progress.json", issues)
    report = _read_object(run / "report.json", issues)
    status = progress.get("status", report.get("status", "unknown"))
    health = "paused" if status == "paused" else "completed" if status in _TERMINAL_STATUSES else "healthy"
    progress_age = _age(run / "progress.json", now)
    checkpoint_age = _age(run / "checkpoints/latest.pt", now)
    failure_path = run / "failure.json"
    if failure_path.exists():
        failure = _read_object(failure_path, issues)
        issues.append(f"Training failure: {failure.get('error', 'unreadable failure record')}")
    if status in {"failed", "interrupted"}:
        issues.append(f"Training reports status={status}")
    live = [worker for worker in workers if worker.get("state") not in {"T", "t", "Z", "X"}]
    ranks = {worker.get("rank") for worker in live}
    epoch_seconds = [float(epoch.get("seconds", 0)) for epoch in report.get("epochs", [])[-3:]]
    checkpoint_threshold = max(stale_seconds, 3 * max(epoch_seconds, default=0))
    if health == "healthy":
        if len(live) != expected_workers or ranks != set(range(expected_workers)):
            issues.append(f"Live worker ranks: {len(live)}/{expected_workers}; ranks={sorted(ranks, key=str)}")
        if progress_age is None or progress_age > stale_seconds:
            issues.append(f"Training progress heartbeat is stale or missing: age={progress_age} s")
        if checkpoint_age is None or checkpoint_age > checkpoint_threshold:
            issues.append(f"Latest checkpoint is stale or missing: age={checkpoint_age} s")
    if not dashboard.get("ok"):
        issues.append(f"Dashboard check failed: {dashboard.get('error', 'unknown error')}")
    if issues:
        health = "attention"
    return {
        "checked_at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
        "health": health,
        "training_status": status,
        "phase": progress.get("phase", "unknown"),
        "epoch": progress.get("epoch", report.get("completed_epochs")),
        "completed_updates": progress.get("completed_updates", report.get("completed_updates")),
        "available_K": progress.get("available_K"),
        "available_H": progress.get("available_H"),
        "workers": workers,
        "progress_age_seconds": progress_age,
        "checkpoint_age_seconds": checkpoint_age,
        "checkpoint_stale_threshold_seconds": checkpoint_threshold,
        "dashboard": dashboard,
        "issues": issues,
    }


def _workers(run):
    result = []
    for directory in _PROC.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / "cmdline").read_bytes().split(b"\0")
            if not Path(os.fsdecode(argv[0])).name.startswith("python"):
                continue
            module_index = argv.index(b"-m")
            if argv[module_index + 1] != b"experiments.learned_intrinsic_solver.train_mixed":
                continue
            output = Path(os.fsdecode(argv[argv.index(b"--output") + 1]))
            if not output.is_absolute():
                output = (directory / "cwd").resolve() / output
            if output.resolve() != run.resolve():
                continue
            fields = dict(line.split(":", 1) for line in (directory / "status").read_text().splitlines())
            rank = next(
                int(entry[5:])
                for entry in (directory / "environ").read_bytes().split(b"\0")
                if entry.startswith(b"RANK=")
            )
            result.append({"pid": int(directory.name), "rank": rank, "state": fields["State"].strip()[0]})
        except (OSError, ValueError, IndexError, StopIteration, KeyError):
            continue
    return sorted(result, key=lambda item: (item["rank"], item["pid"]))


def _dashboard(url, run, now):
    try:
        response = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--max-time", "20", f"{url}?monitor={int(now)}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=25,
        )
        report = json.loads(response.stdout)
        publication = report.get("publication", {}).get("updated_at")
        age = None if not publication else now - datetime.fromisoformat(publication).timestamp()
        if age is None or age > 180:
            raise ValueError(f"publication heartbeat age={age} s")
        local_progress = json.loads((run / "progress.json").read_text())
        remote_progress = report.get("progress", {})
        local_update = local_progress.get("completed_updates", 0)
        remote_update = remote_progress.get("completed_updates", 0)
        if remote_update < local_update and (_age(run / "progress.json", now) or 0) > 180:
            raise ValueError(f"published updates={remote_update}, local updates={local_update}")
        return {"ok": True, "publication_age_seconds": max(0, age), "completed_updates": remote_update}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"ok": False, "error": str(error)[:600]}


def has_pending_monitor_message(database, thread):
    """Check only this thread's queue, without modifying Codex-owned storage.

    Experimental. This read-only deduplication uses the installed Codex queue
    schema. A schema change raises an error instead of accumulating messages.
    """
    with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True, timeout=2) as connection:
        return any(
            _MESSAGE_MARKER in row[0]
            for row in connection.execute("SELECT payload_json FROM queued_items WHERE thread_id = ?", (thread,))
        )


def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _notify(args, health):
    if has_pending_monitor_message(args.queue_database, args.thread):
        return "existing monitor message pending; latest health remains on disk"
    message = (
        f"{_MESSAGE_MARKER} Scheduled 15-minute check requested by the user. "
        f"{health['checked_at']}: {health['health']}; epoch={health['epoch']}; "
        f"updates={health['completed_updates']}; phase={health['phase']}; "
        f"worker PIDs={[worker['pid'] for worker in health['workers']]}. "
        f"Latest health and issues: {args.output / 'health.json'}. "
        f"Issues: {'; '.join(health['issues']) or 'none detected'}. "
        "Read the latest health, inspect training progress and relevant logs, and investigate/fix failures "
        "within the user's existing authorization. Do not blindly restart or change the objective, "
        "architecture, sampling or curriculum. Respect a later user pause/stop request. "
        "If healthy, keep training running; report only useful changes. "
        f"To disable this monitor after a user stop request, create {args.output / 'STOP'}."
    )
    result = subprocess.run(
        [args.codex, "queue", "--thread", args.thread, "--message", message],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--interval", type=float, default=900)
    parser.add_argument("--stale-seconds", type=float, default=2700)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--codex", default="/usr/bin/codex")
    parser.add_argument("--queue-database", type=Path, default=Path.home() / ".codex/queue_1.sqlite")
    parser.add_argument(
        "--dashboard-url", default="https://ankachen.com/artifacts/learned-intrinsic-training-v2/report.json"
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()
    if args.interval < 60 or args.stale_seconds < args.interval or args.workers < 1:
        parser.error("interval must be at least 60 s, stale threshold at least interval, and workers positive")
    args.run, args.output = args.run.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (args.output / "STOP").exists():
            started = time.time()
            try:
                health = check_training_health(
                    args.run,
                    workers=_workers(args.run),
                    dashboard=_dashboard(args.dashboard_url, args.run, started),
                    now=started,
                    expected_workers=args.workers,
                    stale_seconds=args.stale_seconds,
                )
                health.update(monitor_pid=os.getpid(), interval_seconds=args.interval, thread=args.thread)
                _save(args.output / "health.json", health)
                if not args.no_notify:
                    try:
                        health["notification"] = _notify(args, health)
                    except (OSError, sqlite3.Error, subprocess.SubprocessError) as error:
                        health["notification_error"] = repr(error)
                health["next_check_at"] = datetime.fromtimestamp(started + args.interval, timezone.utc).isoformat()
                _save(args.output / "health.json", health)
                with (args.output / "checks.jsonl").open("a") as history:
                    history.write(json.dumps(health, allow_nan=False) + "\n")
                print(json.dumps(health, allow_nan=False), flush=True)
            except Exception as error:
                print(f"Monitor check failed; retrying at next interval: {error!r}", flush=True)
            if args.once:
                break
            remaining = max(0, started + args.interval - time.time())
            while remaining > 0 and not (args.output / "STOP").exists():
                time.sleep(min(5, remaining))
                remaining = max(0, started + args.interval - time.time())


if __name__ == "__main__":
    main()
