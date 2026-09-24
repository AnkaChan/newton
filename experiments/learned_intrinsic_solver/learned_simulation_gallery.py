# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build an offline HTML gallery for ten saved learned-solver simulations."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

__all__ = ["build_gallery"]


def _seconds(value):
    return f"{float(value):g} s"


def _stage_asset(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.samefile(source):
            raise FileExistsError(f"gallery asset differs from rendered source: {target}")
    else:
        target.hardlink_to(source)


def build_gallery(
    simulation_dir: Path,
    renders_dir: Path,
    *,
    output: Path | None = None,
    expected_seeds: tuple[int, ...] | None = None,
) -> dict:
    """Write a gallery that distinguishes requested duration from valid time."""
    simulation_dir, renders_dir = Path(simulation_dir), Path(renders_dir)
    output = renders_dir if output is None else Path(output)
    if expected_seeds is None:
        seeds = sorted(
            int(path.parent.name.removeprefix("seed_")) for path in simulation_dir.glob("seed_*/report.json")
        )
        if len(seeds) != 10 or len(set(seeds)) != 10:
            raise ValueError("the simulation gallery requires ten distinct seed reports")
    else:
        seeds = list(expected_seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("expected_seeds must be nonempty and distinct")
    rows = []
    cards = []
    for seed in seeds:
        stem = f"seed_{seed}"
        simulation = json.loads((simulation_dir / stem / "report.json").read_text())
        render = json.loads((renders_dir / stem / "render.json").read_text())
        if simulation["seed"] != seed or simulation["optimizer_iterations_per_step"] != 2:
            raise ValueError(f"{stem} has mismatched seed or learned iteration count")
        if abs(float(simulation["time_step"]) - 1 / 300) > 1e-10:
            raise ValueError(f"{stem} has an unexpected physical time step")
        if render.get("status") != "complete":
            raise ValueError(f"{stem} render did not complete")
        for key in ("video", "initial_image", "final_image"):
            source = renders_dir / stem / render[key]
            if not source.is_file():
                raise FileNotFoundError(f"{stem} is missing {render[key]}")
            _stage_asset(source, output / stem / render[key])
        _stage_asset(simulation_dir / stem / "report.json", output / stem / "simulation_report.json")
        _stage_asset(renders_dir / stem / "render.json", output / stem / "render.json")
        requested = float(simulation["requested_duration_seconds"])
        actual = float(simulation["actual_duration_seconds"])
        if requested <= 0 or actual < 0 or actual > requested + 1e-8:
            raise ValueError(f"{stem} has invalid requested or actual physical duration")
        if "last_physical_time_seconds" in render and abs(actual - float(render["last_physical_time_seconds"])) > 1e-6:
            raise ValueError(f"{stem} render time differs from the simulated duration")
        status = simulation["status"]
        complete = status == "complete" and actual >= requested - 1e-8
        failure = simulation.get("failure")
        if not complete and failure is None:
            raise ValueError(f"{stem} stopped early without a failure record")
        velocity_rms = simulation.get("initial_velocity_rms_m_per_s")
        velocity_max = simulation.get("initial_velocity_max_m_per_s")
        row = {
            "seed": seed,
            "checkpoint_epoch": int(simulation["checkpoint_epoch"]),
            "status": "complete" if complete else "partial",
            "requested_duration_seconds": requested,
            "actual_duration_seconds": actual,
            "time_step": float(simulation["time_step"]),
            "optimizer_iterations_per_step": 2,
            "initial_velocity_rms_m_per_s": velocity_rms,
            "initial_velocity_max_m_per_s": velocity_max,
            "failure": failure,
            "video": f"{stem}/{render['video']}",
        }
        rows.append(row)
        title = "Completed" if complete else "Stopped at last valid state"
        failure_html = (
            ""
            if complete
            else (
                '<p class="failure">Stopped when a learned update failed physical validity checks.</p>'
                f"<details><summary>Failure diagnostics</summary><pre>{html.escape(json.dumps(failure, indent=2, ensure_ascii=False))}</pre></details>"
            )
        )
        video = html.escape(row["video"], quote=True)
        initial = html.escape(f"{stem}/{render['initial_image']}", quote=True)
        final = html.escape(f"{stem}/{render['final_image']}", quote=True)
        speed = (
            f"Initial velocity RMS {float(velocity_rms):.4g} m/s, max {float(velocity_max):.4g} m/s"
            if velocity_rms is not None and velocity_max is not None
            else "Initial velocity unavailable"
        )
        cards.append(
            f'<article class="card"><h2>Seed {seed} <span class="badge">{title}</span></h2>'
            f"<p>Epoch {row['checkpoint_epoch']} checkpoint · 2 learned iterations per physical step · "
            f"dt = 1/300 s</p><p><strong>{_seconds(actual)} simulated</strong> · {_seconds(requested)} requested. "
            f"{html.escape(speed)}.</p>"
            f'<video controls preload="metadata" poster="{initial}"><source src="{video}" type="video/mp4">'
            f'<a href="{video}">Download video</a></video><div class="thumbnails">'
            f'<figure><img src="{initial}" alt="Initial state for seed {seed}"><figcaption>Initial state</figcaption></figure>'
            f'<figure><img src="{final}" alt="Last valid state for seed {seed}"><figcaption>'
            f"{'Final state' if complete else 'Last valid state'}</figcaption></figure></div>{failure_html}"
            f'<p><a href="{stem}/simulation_report.json">Simulation details</a> · '
            f'<a href="{stem}/render.json">Render details</a></p></article>'
        )
    epochs = {row["checkpoint_epoch"] for row in rows}
    if len(epochs) != 1:
        raise ValueError("gallery simulations use different checkpoint epochs")
    complete_count = sum(row["status"] == "complete" for row in rows)
    summary = {
        "checkpoint_epoch": rows[0]["checkpoint_epoch"],
        "sample_count": len(rows),
        "complete_count": complete_count,
        "failed_count": len(rows) - complete_count,
        "samples": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "gallery.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "gallery.json")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Learned intrinsic solver · physical simulation videos</title><style>
body{{font:17px/1.5 system-ui,sans-serif;max-width:1200px;margin:28px auto;padding:0 18px;color:#223844;background:#f5f8fa}}
h1{{line-height:1.2}} .lead{{max-width:85ch}} .card{{background:white;padding:22px;margin:24px 0;border-radius:12px;box-shadow:0 2px 14px #d8e2e8}}
.badge{{font-size:.7em;color:#85501e;margin-left:1em}} video{{width:100%;max-height:650px;background:#14252c;border-radius:8px}}
.thumbnails{{display:flex;gap:18px}} figure{{flex:1;margin:12px 0}} figure img{{width:100%;border-radius:6px}} figcaption{{font-size:.85em;color:#526976}}
.failure{{background:#fff0df;padding:12px;overflow-wrap:anywhere}} a{{color:#087b91}}
</style></head><body><p><a href="../index.html">← All solver experiments</a></p><h1>{len(rows)} learned-solver physical simulations</h1>
<p class="lead">Epoch {summary["checkpoint_epoch"]} checkpoint. Each physical step uses two consecutive learned optimizer calls at dt = 1/300 s. The requested duration is 10 s per state; a failed simulation ends at its last valid physical state. Clips contain no repeated or padded frames. The camera stays fixed within each clip and shows the saved shape trajectory, surface voxel grid, and pinned corners.</p>
<p>{complete_count} completed · {summary["failed_count"]} stopped early. Physical time appears in each video frame. Initial and final thumbnails are below each clip.</p>
{"".join(cards)}<p><a href="gallery.json">Machine-readable results</a></p></body></html>"""
    temporary = output / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(output / "index.html")
    return summary


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--simulation-dir", type=Path, required=True)
    parser.add_argument("--renders-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_gallery(args.simulation_dir, args.renders_dir, output=args.output)
    print(
        json.dumps({key: result[key] for key in ("checkpoint_epoch", "sample_count", "complete_count", "failed_count")})
    )


if __name__ == "__main__":
    _main()
