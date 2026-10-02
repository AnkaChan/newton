# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run report files of the LIDO trainer: report.json, progress.json, epochs.csv and failure.json.

The key set is the one the existing dashboard reads (see docs/dashboard_keys.md). Rank 0 owns one
RunReport. Every value is converted to plain Python types before writing; non-finite floats become null.

    python -m experiments.lido.report <run_dir>    prints the headline of a run
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from experiments.lido.config import TrainConfig

STATUSES = ("initializing", "preparing", "running", "completed", "epoch_limit", "failed", "interrupted")

# epochs.csv columns, in the dashboard's order (docs/dashboard_keys.md section 5).
EPOCH_COLUMNS = (
    "epoch",
    "loss",
    "query_count",
    "seconds",
    "mean_force_residual_n",
    "step_size_mean",
    "step_size_min",
    "step_size_max",
    "tie_cell_count",
    "gradient_norm_mean",
    "gradient_norm_max",
    "selection_metric",
    "selection_eligible",
    "physical_survivors",
    "sample_count",
    "contact_scene_fraction",
    "contact_realized_fraction",
    "contact_max_penetration_r",
    "validation_final_max_penetration_r",
    "regime_stage",
    "regime_k_max",
    "regime_h_max",
    "regime_updates",
    "full_horizon_selection_metric",
    "full_horizon_selection_eligible",
)

# progress.json keys (docs/dashboard_keys.md section 4).
PROGRESS_KEYS = (
    "updated_at",
    "status",
    "phase",
    "epoch",
    "max_epochs",
    "completed_epochs",
    "completed_updates",
    "latest_batch_loss",
    "available_K",
    "available_H",
    "regime",
)

_REGIME_KEYS = (
    "stage",
    "k_max",
    "h_max",
    "queries",
    "filler_queries",
    "updates",
    "step_cap",  # the step-cap curriculum's cap of the epoch (None before the curriculum existed)
    "pinned_fraction",  # the v5 scene curriculum's mix of the epoch (None in body mode)
    "resting_fraction",
)


# ----------------------------------------------------------------------------- plain values and files


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def plain(value):
    """Return ``value`` as plain JSON-safe Python data.

    Tensors, numpy arrays and numpy scalars become lists or scalars, tuples become lists, dataclasses
    become dicts, and non-finite floats become None.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value))
    if hasattr(value, "tolist"):  # torch tensors, numpy arrays and numpy scalars
        return plain(value.tolist())
    if hasattr(value, "item"):
        return plain(value.item())
    return str(value)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values: list):
    finite = [float(v) for v in values if _finite(v)]
    return statistics.fmean(finite) if finite else None


def _stats(values: list, keys=("mean", "median", "max")) -> dict:
    finite = [float(v) for v in values if _finite(v)]
    full = {"mean": None, "median": None, "max": None}
    if finite:
        full = {"mean": statistics.fmean(finite), "median": statistics.median(finite), "max": max(finite)}
    return {k: full[k] for k in keys}


def _at(seq, i):
    """``seq[i]`` or None when ``seq`` is not a list or the index is out of range."""
    return seq[i] if isinstance(seq, list) and -len(seq) <= i < len(seq) else None


def dumps(value) -> str:
    """JSON text of ``plain(value)``; raises instead of writing NaN or inf."""
    return json.dumps(plain(value), indent=2, allow_nan=False) + "\n"


def atomic_write(path: str | Path, text: str) -> None:
    """Write ``text`` to a temporary file next to ``path`` and rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ----------------------------------------------------------------------------- summaries


def build_epoch_record(
    *,
    epoch,
    loss,
    query_count,
    updates,
    seconds,
    lr,
    grad_norm_mean,
    grad_norm_max,
    step_mean,
    step_min,
    step_max,
    tie_cell_count,
    mean_force_residual_n,
    resets,
    failures: list,
    regime: dict,
    available_K: list,
    available_H: list,
    contact_scene_fraction,
    contact_realized_fraction,
    contact_max_penetration_r,
    material_histograms: dict | None = None,
    validation: dict | None = None,
    full_horizon_validation: dict | None = None,
    rank_diagnostics: list | None = None,
    scene_regime: dict | None = None,
) -> dict:
    """One row of ``report["epochs"]`` with the dashboard's key names (regime named ``fixed_states``).

    ``failures`` may hold FailureRecord dataclasses or dicts; ``updates``, ``resets``, ``failures``,
    ``material_histograms`` and ``rank_diagnostics`` are carried through, the dashboard does not read them.
    ``scene_regime`` (the v5 scene statistics of ``SceneRunner.epoch_summary``) is added as an extra key when given;
    the dashboard tolerates extra keys.
    """
    regime = plain(regime) or {}
    record = {
        "epoch": epoch,
        "loss": loss,
        "query_count": query_count,
        "updates": updates,
        "seconds": seconds,
        "learning_rate": lr,
        "gradient_norm_mean": grad_norm_mean,
        "gradient_norm_max": grad_norm_max,
        "step_size_mean": step_mean,
        "step_size_min": step_min,
        "step_size_max": step_max,
        "tie_cell_count": tie_cell_count,
        "mean_force_residual_n": mean_force_residual_n,
        "resets": resets,
        "failures": failures,
        "regime": {"name": "fixed_states", **{k: regime.get(k) for k in _REGIME_KEYS}},
        "available_K": list(available_K),
        "available_H": list(available_H),
        "contact_scene_fraction": contact_scene_fraction,
        "contact_realized_fraction": contact_realized_fraction,
        "contact_max_penetration_r": contact_max_penetration_r,
        "material_histograms": material_histograms,
        "validation": validation,
        "full_horizon_validation": full_horizon_validation,
        "rank_diagnostics": rank_diagnostics,
    }
    if scene_regime is not None:
        record["scene_regime"] = scene_regime
    return plain(record)


# v5 scene records (validation.py): per-iteration curves summarised like the penetration, and final values.
SCENE_CURVE_KEYS = ("interbody_penetration_r", "plane_penetration_r")
PAIR_KEYS = ("total", "plane", "static", "body")  # the summarised pair counts (static = the scene's static faces)


def _scene_curves(samples: list, iterations: int) -> dict:
    """{key: [K+1 x {iteration, mean, max}]} for the v5 scene curves present in the samples (empty otherwise)."""
    out = {}
    for key in SCENE_CURVE_KEYS:
        if any(isinstance(s.get(key), list) for s in samples):
            out[key] = [
                {"iteration": i, **_stats([_at(s.get(key), i) for s in samples], ("mean", "max"))}
                for i in range(int(iterations) + 1)
            ]
    pairs = [s.get("contact_pairs") for s in samples if isinstance(s.get("contact_pairs"), dict)]
    if pairs:
        out["contact_pairs"] = {k: _mean([p.get(k) for p in pairs]) for k in PAIR_KEYS}
    return out


def summarize_cheap_validation(samples: list, iterations: int, floor_scale_joule: float | None = None) -> dict:
    """Summary of the cheap validation (K iterations on fixed states) in the dashboard's ``validation`` shape.

    Each sample: ``{seed, residual_n: [K+1], energy_joule: [K+1], penetration_r: [K+1], inverted_cells: [K+1],
    survived, failed_first_update, scale_joule, contact}``; the curves hold the value before each query and
    after the last. ``scale_joule`` is the loss scale max(|E_before|, floor) of the first query; when a sample
    lacks it, max(|E_0|, floor_scale_joule) is used. Samples whose |E_0| / scale < 1e-6 are "near zero" and
    excluded from the relative-energy ratios.
    """
    samples = [plain(s) for s in samples]
    survivors = [s for s in samples if s.get("survived")]
    first, last, normalized, near_zero, descended = [], [], [], [], 0
    for s in samples:
        e0, e_last = _at(s.get("energy_joule"), 0), _at(s.get("energy_joule"), -1)
        scale = s.get("scale_joule")
        if not (_finite(scale) and scale > 0):
            scale = max(abs(e0) if _finite(e0) else 0.0, floor_scale_joule or 0.0)
        near_zero.append(not (_finite(e0) and scale > 0 and abs(e0) / scale >= 1e-6))
        first.append(e0)
        last.append(e_last)
        if _finite(e_last) and scale > 0:
            normalized.append(e_last / scale)
        if _finite(e0) and _finite(e_last) and e_last < e0:
            descended += 1
    relative, residual, penetration = [], [], []
    for i in range(int(iterations) + 1):
        ratios = [
            _at(s["energy_joule"], i) / _at(s["energy_joule"], 0)
            for s, nz in zip(samples, near_zero, strict=True)
            if not nz and _finite(_at(s.get("energy_joule"), i))
        ]
        relative.append({"iteration": i, **_stats(ratios), "near_zero_count": sum(near_zero)})
        residual.append({"iteration": i, **_stats([_at(s.get("residual_n"), i) for s in samples])})
        penetration.append(
            {"iteration": i, **_stats([_at(s.get("penetration_r"), i) for s in samples], ("mean", "max"))}
        )
    return {
        "mean_normalized_loss": _mean(normalized),
        "descent_rate": descended / len(samples) if samples else None,
        "mean_before_joule": _mean(first),
        "mean_after_joule": _mean(last),
        "selection": {
            "metric": _mean([_at(s.get("residual_n"), -1) for s in survivors]),
            "eligible": bool(samples) and len(survivors) == len(samples),
        },
        "physical_survivors": len(survivors),
        "sample_count": len(samples),
        "failed_count": len(samples) - len(survivors),
        "first_update_failed_count": sum(1 for s in samples if s.get("failed_first_update")),
        "failures": [s.get("seed") for s in samples if not s.get("survived")],
        "relative_energy": relative,
        "force_residual": residual,
        "penetration": penetration,
        **_scene_curves(samples, iterations),
        "samples": samples,
    }


def summarize_full_horizon(samples: list, iterations: int, physical_steps: int, seconds: float) -> dict:
    """Summary of the held-out full-horizon check in the dashboard's ``full_horizon_validation`` shape.

    Each sample: ``{seed, physical_records: [H x {residual_n, energy_joule, penetration_r, inverted_cells}],
    survived}``. Final statistics are taken over the survivors' last physical record. v5 scene samples
    (validation.py) add ``interbody_penetration_r``, ``plane_penetration_r`` and ``contact_pairs`` to the records
    and ``momentum_drift`` to the sample; their final statistics are added when present.
    """
    samples = [plain(s) for s in samples]
    survivors = [s for s in samples if s.get("survived")]
    finals = [(s.get("physical_records") or [{}])[-1] for s in survivors]
    residual = _stats([r.get("residual_n") for r in finals])
    out = {
        "iterations": int(iterations),
        "physical_steps": int(physical_steps),
        "physical_survivors": len(survivors),
        "sample_count": len(samples),
        "seconds": plain(float(seconds)),
        "final_free_force_residual_norm_n": residual,
        "final_energy_joule": _stats([r.get("energy_joule") for r in finals], ("mean",)),
        "final_max_penetration_r": _stats([r.get("penetration_r") for r in finals], ("mean", "max")),
        "selection": {"metric": residual["mean"], "eligible": bool(samples) and len(survivors) == len(samples)},
    }
    for key in SCENE_CURVE_KEYS:
        if any(key in r for r in finals):
            out[f"final_{key}"] = _stats([r.get(key) for r in finals], ("mean", "max"))
    pairs = [r.get("contact_pairs") for r in finals if isinstance(r.get("contact_pairs"), dict)]
    if pairs:
        out["final_contact_pairs"] = {k: _mean([p.get(k) for p in pairs]) for k in PAIR_KEYS}
    if any("momentum_drift" in s for s in samples):
        out["momentum_drift"] = _mean([s.get("momentum_drift") for s in samples])
    out["samples"] = samples
    return out


def selection_better(candidate: dict, best: dict | None) -> bool:
    """True when ``candidate`` should replace ``best``: eligible beats ineligible, then the lower metric wins.

    Both arguments may be a summary (with a nested ``selection``) or a flat record with ``metric`` and
    ``eligible``. A candidate without a finite metric never wins; any candidate beats a missing best.
    """
    c = plain(candidate)
    c = c.get("selection", c)
    if not _finite(c.get("metric")):
        return False
    b = plain(best) or {}
    b = b.get("selection", b) or {}
    if not _finite(b.get("metric")):
        return True
    if bool(c.get("eligible")) != bool(b.get("eligible")):
        return bool(c.get("eligible"))
    return c["metric"] < b["metric"]


def _best_record(summary: dict, epoch: int, source: str, completed_updates: int) -> dict:
    summary = plain(summary)
    selection = summary.get("selection", summary)
    return {
        "metric": selection.get("metric"),
        "eligible": selection.get("eligible"),
        "epoch": int(epoch),
        "source": source,
        "iterations": summary.get("iterations"),
        "physical_steps": summary.get("physical_steps"),
        "sample_count": summary.get("sample_count"),
        "physical_survivors": summary.get("physical_survivors"),
        "final_energy_joule": summary.get("final_energy_joule"),
        "final_max_penetration_r": summary.get("final_max_penetration_r"),
        "completed_updates": int(completed_updates),
    }


def _csv_row(row: dict) -> dict:
    validation = row.get("validation") or {}
    full = row.get("full_horizon_validation") or {}
    regime = row.get("regime") or {}
    penetration = validation.get("penetration") or []
    last_penetration = penetration[-1] if penetration and isinstance(penetration[-1], dict) else {}
    return {
        **row,
        "selection_metric": (validation.get("selection") or {}).get("metric"),
        "selection_eligible": (validation.get("selection") or {}).get("eligible"),
        "physical_survivors": validation.get("physical_survivors"),
        "sample_count": validation.get("sample_count"),
        "validation_final_max_penetration_r": last_penetration.get("max"),
        "regime_stage": regime.get("stage"),
        "regime_k_max": regime.get("k_max"),
        "regime_h_max": regime.get("h_max"),
        "regime_updates": regime.get("updates"),
        "full_horizon_selection_metric": (full.get("selection") or {}).get("metric"),
        "full_horizon_selection_eligible": (full.get("selection") or {}).get("eligible"),
    }


# ----------------------------------------------------------------------------- the writer


class RunReport:
    """Rank-0 writer of ``report.json``, ``progress.json``, ``epochs.csv`` and ``failure.json`` in ``run_dir``.

    ``log_update`` only refreshes ``progress.json`` (cheap); ``log_epoch`` and ``write`` write all three
    files; ``set_status`` writes ``progress.json`` and ``report.json``. Call ``set_best_selection`` before
    ``log_epoch`` so the epoch's files carry the new record.
    """

    def __init__(
        self,
        run_dir: str | Path,
        cfg: TrainConfig,
        world_size: int,
        git_sha: str = "",
        initialized_from: dict | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self._phase = "initializing"
        self._epoch = 0
        self._regime = {"name": "fixed_states"}
        self._latest_loss = None
        self._report = {
            "status": "initializing",
            "updated_at": _utc_now(),
            "config": plain(cfg.to_dict()),
            "world_size": int(world_size),
            "epochs": [],
            "updates": [],
            "completed_epochs": 0,
            "completed_updates": 0,
            "best_selection": None,
            "best_selection_history": [],
            "parameter_count": None,
            "feature_schema_version": int(cfg.feature_schema_version),
            "git_sha": str(git_sha),
            "initialized_from": plain(initialized_from),
        }

    @property
    def report(self) -> dict:
        return self._report

    def set_parameter_count(self, n: int) -> None:
        self._report["parameter_count"] = int(n)

    def log_update(self, epoch, update, loss, grad_norm, lr, step_mean, active, resets, seconds) -> None:
        """Append one update row and refresh ``progress.json``.

        ``update`` is the global count of optimizer updates done so far (across epochs); it becomes
        ``completed_updates``. At most ``cfg.updates_history_limit`` rows are kept, oldest dropped.
        """
        row = plain(
            {
                "update": update,
                "epoch": epoch,
                "loss": loss,
                "gradient_norm": grad_norm,
                "learning_rate": lr,
                "step_size_mean": step_mean,
                "active": active,
                "resets": resets,
                "seconds": seconds,
            }
        )
        updates = self._report["updates"]
        updates.append(row)
        del updates[: max(0, len(updates) - int(self.cfg.updates_history_limit))]
        self._report["completed_updates"] = max(self._report["completed_updates"], int(row["update"] or 0))
        self._latest_loss = row["loss"]
        self._epoch = int(row["epoch"] or self._epoch)
        self._write_progress()

    def log_epoch(self, record: dict) -> None:
        """Append a ``build_epoch_record`` row, count the epoch and write all three files."""
        record = plain(record)
        self._report["epochs"].append(record)
        self._report["completed_epochs"] += 1
        self._epoch = int(record.get("epoch") or self._epoch)
        if isinstance(record.get("regime"), dict):
            self._regime = record["regime"]
        self.write()

    def set_best_selection(self, record: dict, epoch: int, reset_reason: str | None = None) -> None:
        """Replace ``best_selection`` with ``record`` (a validation summary or a flat selection record).

        With ``reset_reason`` (for example after a budget change) the superseded record is pushed onto
        ``best_selection_history`` first. Does not write; ``log_epoch`` or ``write`` does.
        """
        if reset_reason is not None:
            self._report["best_selection_history"].append(
                {"reset_at_epoch": int(epoch), "reason": str(reset_reason), "record": self._report["best_selection"]}
            )
        self._report["best_selection"] = _best_record(
            record, epoch, self.cfg.selection_source, self._report["completed_updates"]
        )

    def set_status(
        self, status: str, phase: str | None = None, *, epoch: int | None = None, regime: dict | None = None
    ) -> None:
        """Set the run status (and heartbeat phase, defaulting to the status); writes progress and report.

        ``epoch`` is the epoch now in progress and ``regime`` its growth-stage block; both are optional and
        otherwise follow the latest ``log_update`` / ``log_epoch``.
        """
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
        self._report["status"] = status
        self._phase = phase or status
        if epoch is not None:
            self._epoch = int(epoch)
        if regime is not None:
            self._regime = {"name": "fixed_states", **plain(regime)}
        self._write_progress()
        self._write_report()

    def write(self) -> None:
        """Atomically write ``report.json``, ``progress.json`` and ``epochs.csv``."""
        self._write_report()
        self._write_progress()
        self._write_csv()

    def write_failure(self, record: dict) -> None:
        """Write ``failure.json`` (a dict with an ``error`` string, or a FailureRecord)."""
        atomic_write(self.run_dir / "failure.json", dumps(record))

    # -- files

    def _write_report(self) -> None:
        self._report["updated_at"] = _utc_now()
        atomic_write(self.run_dir / "report.json", dumps(self._report))

    def _write_progress(self) -> None:
        atomic_write(self.run_dir / "progress.json", dumps(self.progress()))

    def _write_csv(self) -> None:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=EPOCH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in self._report["epochs"])
        atomic_write(self.run_dir / "epochs.csv", buffer.getvalue())

    def progress(self) -> dict:
        """The ``progress.json`` heartbeat (the 11 documented keys)."""
        latest = self._report["epochs"][-1] if self._report["epochs"] else {}
        return {
            "updated_at": _utc_now(),
            "status": self._report["status"],
            "phase": self._phase,
            "epoch": self._epoch,
            "max_epochs": int(self.cfg.max_epochs),
            "completed_epochs": self._report["completed_epochs"],
            "completed_updates": self._report["completed_updates"],
            "latest_batch_loss": self._latest_loss,
            "available_K": self._available(self.cfg.K_values, self._regime.get("k_max"), latest.get("available_K")),
            "available_H": self._available(
                self.cfg.physical_step_counts, self._regime.get("h_max"), latest.get("available_H")
            ),
            "regime": dict(self._regime),
        }

    @staticmethod
    def _available(values, cap, fallback) -> list:
        """Values within the current stage cap, else the latest epoch's list, else the smallest value."""
        if _finite(cap):
            return [int(v) for v in values if v <= cap] or [int(min(values))]
        return list(fallback) if fallback else [int(min(values))]


# ----------------------------------------------------------------------------- CLI


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the headline of a LIDO run directory.")
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    report = json.loads((run / "report.json").read_text())
    progress_file = run / "progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.is_file() else {}
    config = report.get("config") or {}
    print(
        f"status {report.get('status')} (phase {progress.get('phase', '-')}), "
        f"epoch {progress.get('epoch', '-')} in progress, "
        f"{report.get('completed_epochs', 0)} / {config.get('max_epochs', '-')} epochs completed, "
        f"{progress.get('completed_updates', report.get('completed_updates', 0))} updates"
    )
    best = report.get("best_selection") or {}
    if _finite(best.get("metric")):
        state = "eligible" if best.get("eligible") else "not eligible"
        print(
            f"best selection: {best['metric']:.6g} N at epoch {best.get('epoch')} ({state}, source {best.get('source')})"
        )
    else:
        print("best selection: none yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
