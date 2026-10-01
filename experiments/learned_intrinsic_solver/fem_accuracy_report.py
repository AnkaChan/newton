# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the FEM accuracy comparison page for the learned solver and Newton VBD.

Experimental. The page compares the four beam scenarios of
:mod:`.fem_accuracy_scenarios` as run by ``fem_accuracy_learned`` and
``fem_accuracy_vbd``. Each scenario section shows the side-by-side clip
(learned left, Newton VBD right) composed with ffmpeg from the two
``render_learned`` clips, or the two clips next to each other when no
composite exists, a plain-language description of the schedule, and a
metrics table with learned, Newton VBD, analytic (where one exists) and
difference columns. The extension scenario gets a per-frame tip-displacement
plot. The closing summary states the verdict per scenario from the recorded
metrics alone: agreement, deviation, cell inversion, truncation, or a
nonfinite failure. ``--publish`` copies the site into Kanna's static
artifacts route and ``--index-card`` adds a card to the project index there.

Input layout under the run root::

    <root>/learned/<scenario>/{metrics.json,run.json}
    <root>/vbd/<scenario>/{metrics.json,run.json}
    <root>/vbd_iter<K>/extension/metrics.json                 optional references
    <root>/renders/<solver>_<scenario>/simulation.mp4         render_learned output
    <root>/renders/<solver>_<scenario>.mp4                    staged clip
    <root>/renders/side_by_side_<scenario>.mp4                optional composite
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import fem_accuracy_scenarios as scenarios

__all__ = [
    "add_index_card",
    "build_report",
    "compose_side_by_side",
    "homogeneous_stretch_reference",
    "publish_site",
    "stage_renders",
]

TITLE = "Volumetric correctness scenarios: learned intrinsic solver vs Newton VBD"
PULL_REQUEST_URL = "https://github.com/newton-physics/newton/pull/2901"
SOLVERS = (("learned", "Learned intrinsic solver"), ("vbd", "Newton VBD"))
"""Directory name and display name of the two compared solvers, left to right in the composites."""

DEFAULT_PUBLISHER_SCRIPT = Path.home() / ".codex/skills/publish-artifact/scripts/publish_artifact.py"
PROJECT_SLUG = "learned-intrinsic-solver"

AGREEMENT_TOLERANCE = 0.05
"""Largest relative learned-to-VBD difference over the table metrics still reported as agreement."""

LARGE_DEVIATION = 0.25
"""Relative difference above which the verdict says the solvers disagree rather than deviate."""

_SERIES_COLORS = {
    "learned": "#2a78d6",
    "vbd": "#eb6834",
    "vbd_iter100": "#1baf7a",
    "vbd_iter400": "#eda100",
}
"""Validated categorical palette in fixed slot order (dataviz reference palette, light surface)."""

_INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
_SURFACE = "#fcfcfb"


@dataclass(frozen=True)
class _Metric:
    """One row of a scenario metrics table.

    Attributes:
        key: Key in ``metrics.json``.
        label: Row label.
        scale: Factor applied to the stored value for display.
        unit: Display unit after scaling.
        digits: Decimal places shown.
        analytic: Reference value in display units, or ``None``.
        analytic_note: Where the reference comes from.
        jacobian: Whether the row is a Jacobian ratio whose sign marks inversion.
    """

    key: str
    label: str
    scale: float = 1.0
    unit: str = ""
    digits: int = 4
    analytic: float | None = None
    analytic_note: str | None = None
    jacobian: bool = False


def homogeneous_stretch_reference(axial_stretch: float) -> tuple[float, float]:
    """Return ``(J, lateral)`` of a homogeneous uniaxial stretch of the shared material.

    Both solvers minimise the stable Neo-Hookean density
    ``mu/2 (|F|^2 - 3) - mu (J - 1) + (lambda + mu)/2 (J - 1)^2``. For
    ``F = diag(s, s, a)`` with free lateral faces the stationary point is
    ``J = 1 + (a - 1) mu / (a (lambda + mu))`` and ``s = sqrt(J / a)``. The
    clamped end faces of the beam are ignored.
    """
    if axial_stretch <= 0:
        raise ValueError("axial stretch must be positive")
    ratio = scenarios.LAME_MU / (scenarios.LAME_LAMBDA + scenarios.LAME_MU)
    volume_ratio = 1.0 + (axial_stretch - 1.0) / axial_stretch * ratio
    return volume_ratio, math.sqrt(volume_ratio / axial_stretch)


def _extension_volume_reference() -> float:
    """Return the small-strain V/V0 of the hanging bar, ``1 + (1 - 2 nu) rho g L / (2 E)``."""
    mean_strain = (
        scenarios.DENSITY * scenarios.GRAVITY_MAGNITUDE * scenarios.BEAM_LENGTH / (2.0 * scenarios.YOUNG_MODULUS)
    )
    return 1.0 + (1.0 - 2.0 * scenarios.POISSON_RATIO) * mean_strain


def _extension_tip_cell_jacobian() -> float:
    """Return the small-strain centre Jacobian of the cell at the free end, half a cell from the tip."""
    strain = scenarios.DENSITY * scenarios.GRAVITY_MAGNITUDE * (0.5 * scenarios.CELL_SIZE) / scenarios.YOUNG_MODULUS
    return 1.0 + (1.0 - 2.0 * scenarios.POISSON_RATIO) * strain


def _metric_rows(name: str) -> tuple[_Metric, ...]:
    if name == "extension":
        return (
            _Metric(
                "tip_displacement_final",
                "Tip displacement at the final frame",
                scale=1e3,
                unit="mm",
                digits=2,
                analytic=1e3 * scenarios.analytic_extension_tip_displacement(),
                analytic_note="small-strain hanging bar, rho g L^2 / (2 E)",
            ),
            _Metric(
                "bulk_volume_ratio",
                "Volume ratio V/V0 at the final frame",
                analytic=_extension_volume_reference(),
                analytic_note="small strain, 1 + (1 - 2 nu) rho g L / (2 E)",
            ),
            _Metric(
                "min_centre_jacobian_ratio",
                "Minimum cell centre Jacobian ratio at the final frame",
                analytic=_extension_tip_cell_jacobian(),
                analytic_note="small strain in the free-end cell, half a cell from the tip",
                jacobian=True,
            ),
        )
    if name == "stretch":
        volume_ratio, lateral = homogeneous_stretch_reference(2.0)
        note = "homogeneous stable Neo-Hookean stretch to 2 L, clamped ends ignored"
        return (
            _Metric(
                "bulk_volume_ratio", "Volume ratio V/V0 at the final frame", analytic=volume_ratio, analytic_note=note
            ),
            _Metric(
                "lateral_contraction",
                "Mid-length width over rest width at the final frame",
                analytic=lateral,
                analytic_note=note,
            ),
            _Metric(
                "min_centre_jacobian_ratio",
                "Minimum cell centre Jacobian ratio at the final frame",
                analytic=volume_ratio,
                analytic_note=note,
                jacobian=True,
            ),
        )
    if name == "twist":
        return (
            _Metric("bulk_volume_ratio_peak", "Volume ratio V/V0 at peak twist (frame 200)"),
            _Metric(
                "min_centre_jacobian_ratio_peak", "Minimum cell centre Jacobian ratio at peak twist", jacobian=True
            ),
            _Metric("bulk_volume_ratio_final", "Volume ratio V/V0 at the final frame"),
            _Metric(
                "min_centre_jacobian_ratio_final",
                "Minimum cell centre Jacobian ratio at the final frame",
                jacobian=True,
            ),
        )
    if name == "compression_release":
        note = "elastic rest state once the released beam stops swinging"
        return (
            _Metric(
                "min_centre_jacobian_ratio_compression",
                "Minimum cell centre Jacobian ratio during the driven phase (frames 0 to 150)",
                jacobian=True,
            ),
            _Metric(
                "length_recovery_ratio",
                "Length recovery at the final frame (far-face z over L)",
                analytic=1.0,
                analytic_note=note,
            ),
            _Metric(
                "bulk_volume_ratio_final", "Volume ratio V/V0 at the final frame", analytic=1.0, analytic_note=note
            ),
        )
    raise ValueError(f"unknown scenario {name!r}")


def _schedule_text(scenario: scenarios.Scenario) -> str:
    fps = scenarios.FPS
    total = f"{scenario.frame_count} frames ({scenario.frame_count / fps:.3g} s)"
    if scenario.name == "extension":
        return (
            f"Gravity of {scenarios.GRAVITY_MAGNITUDE} m/s^2 points along the beam axis away from the clamp, so the "
            f"{scenarios.BEAM_LENGTH:g} m beam hangs from its clamped end and stretches under its own weight. The far face is "
            f"free. {total}. The small-strain estimate of the tip extension is rho g L^2 / (2 E) = "
            f"{1e3 * scenarios.analytic_extension_tip_displacement():.2f} mm."
        )
    ramp = f"{scenario.ramp_frames} frames ({scenario.ramp_seconds:.3g} s)"
    if scenario.name == "stretch":
        return (
            f"No gravity. The far face is pulled along the beam axis by d(t) = L min(t / T, 1) with T = {ramp}, doubling the "
            f"beam length, and then held. {total}; metrics at the final frame."
        )
    if scenario.name == "twist":
        return (
            f"No gravity. The far face is rotated rigidly about the beam axis through its centroid by theta(t) = 2 pi "
            f"min(t / T, 1) with T = {ramp}, one full turn, and then held. {total}; metrics at the peak-twist frame "
            f"{scenario.ramp_frames} and at the final frame."
        )
    if scenario.name == "compression_release":
        release = scenario.release_frame
        return (
            f"No gravity. The far face is pushed toward the clamp by d(t) = -L/2 min(t / T, 1) with T = {ramp}, compressing "
            f"the beam to half its length, held through frame {release} ({release / fps:.3g} s), and then released so the far "
            f"face is free. {total}. A slender beam buckles under this compression, so the metrics are the smallest cell "
            f"centre Jacobian during the driven phase and the length and volume recovered by the final frame."
        )
    raise ValueError(f"unknown scenario {scenario.name!r}")


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _format(value, metric: _Metric) -> str:
    if value is None:
        return "not reached"
    return f"{float(value) * metric.scale:.{metric.digits}f}{' ' + metric.unit if metric.unit else ''}"


def _relative(a: float, b: float) -> float | None:
    if abs(b) < 1e-12:
        return None
    return (a - b) / abs(b)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.1f}%"


def _difference(learned, vbd, metric: _Metric) -> tuple[str, float | None]:
    """Return the display string of learned minus VBD and the relative difference to VBD."""
    if learned is None or vbd is None:
        return "n/a", None
    scaled_learned, scaled_vbd = float(learned) * metric.scale, float(vbd) * metric.scale
    relative = _relative(scaled_learned, scaled_vbd)
    text = f"{scaled_learned - scaled_vbd:+.{metric.digits}f}{' ' + metric.unit if metric.unit else ''}"
    if relative is not None:
        text += f" ({_percent(relative)})"
    return text, relative


def _solver_state(metrics, run) -> dict:
    """Summarise the completion state of one solver run."""
    if metrics is None:
        return {"available": False, "completed": False, "failure": None, "frame_count": 0, "text": "no run on disk"}
    failure = metrics.get("failure") or (run or {}).get("failure")
    completed = bool(metrics.get("completed"))
    frames = int(metrics.get("frame_count", 0))
    if failure:
        frame = failure.get("frame")
        text = f"nonfinite proposal at frame {frame}" + (
            f" ({frame / scenarios.FPS:.2f} s)" if frame is not None else ""
        )
        text += f"; {frames} frames recorded, metrics from the last finite frame"
    elif not completed:
        text = f"stopped after {frames} frames without a failure record"
    else:
        text = f"complete, {frames} frames"
    return {"available": True, "completed": completed, "failure": failure, "frame_count": frames, "text": text}


def _lower_first(text: str) -> str:
    """Return ``text`` with only its first character lowercased, keeping names such as Jacobian."""
    return text[:1].lower() + text[1:] if text else text


def _grade(deviation: float) -> str:
    """Map a relative deviation to ``agreement``, ``deviation`` or ``disagreement``."""
    if deviation <= AGREEMENT_TOLERANCE:
        return "agreement"
    return "deviation" if deviation <= LARGE_DEVIATION else "disagreement"


def _verdict(
    scenario: scenarios.Scenario, rows: list[dict], states: dict, references: dict, vbd_iterations=None
) -> dict:
    """Derive the per-scenario verdict from the metric rows and the completion states.

    The learned-to-VBD comparison sets the status. For the extension scenario
    with ``vbd_iter<K>`` reference runs on disk, the comparison against the
    most-iterated reference sets it instead, because the tabulated VBD run is
    known not to be converged there; the sentences state both comparisons.

    Returns:
        ``{"status": ..., "label": ..., "sentences": [...]}`` where ``status``
        is one of ``missing``, ``failure``, ``inversion``, ``disagreement``,
        ``deviation`` or ``agreement``.
    """
    labels = {
        "missing": "Not available",
        "failure": "Failure",
        "inversion": "Cell inversion",
        "disagreement": "Disagreement",
        "deviation": "Deviation",
        "agreement": "Agreement",
    }
    sentences: list[str] = []
    status = None
    for key, name in SOLVERS:
        if not states[key]["available"]:
            sentences.append(f"{name}: {states[key]['text']}.")
            status = "missing"
    if status == "missing":
        return {"status": status, "label": labels[status], "sentences": sentences}
    for key, name in SOLVERS:
        if states[key]["failure"] or not states[key]["completed"]:
            sentences.append(f"{name}: {states[key]['text']}.")
            status = "failure"
    inverted = []
    for row in rows:
        if not row["jacobian"]:
            continue
        for key, name in SOLVERS:
            value = row["values"][key]
            if value is not None and float(value) < 0:
                inverted.append(
                    f"{name} reaches a negative centre Jacobian ({float(value):.3f}) in the {_lower_first(row['label'])}"
                )
    if inverted:
        sentences.append("; ".join(inverted) + ".")
        status = status or "inversion"
    comparable = [row for row in rows if row["relative"] is not None and not row["both_negative"]]
    excluded = sum(1 for row in rows if row["relative"] is not None and row["both_negative"])
    exclusion = f"; the {excluded} row where both solvers invert is not compared by percentage" if excluded else ""
    if comparable:
        worst = max(comparable, key=lambda row: abs(row["relative"]))
        deviation = abs(worst["relative"])
        comparison = _grade(deviation)
        if comparison == "agreement":
            sentences.append(
                f"The two solvers agree within {100 * deviation:.1f}% on all {len(comparable)} compared metrics{exclusion}."
            )
        else:
            sentences.append(
                f"The largest learned-to-VBD difference is {100 * deviation:.1f}% in the {_lower_first(worst['label'])} "
                f"(learned {worst['display']['learned']} vs Newton VBD {worst['display']['vbd']}){exclusion}."
            )
        if scenario.name == "extension" and references:
            best = max(references)
            deviations = []
            for row in rows:
                learned_value, reference_value = row["values"]["learned"], references[best].get(row["key"])
                if learned_value is None or reference_value is None:
                    continue
                relative = _relative(float(learned_value) * row["scale"], float(reference_value) * row["scale"])
                if relative is not None:
                    deviations.append((abs(relative), row, float(reference_value)))
            if deviations:
                deviation, row, reference_value = max(deviations, key=lambda item: item[0])
                comparison = _grade(deviation)
                shown = f"{reference_value * row['scale']:.{row['digits']}f}{' ' + row['unit'] if row['unit'] else ''}"
                sentences.append(
                    f"Against Newton VBD at {best} iterations per substep, its converged reference, the largest learned "
                    f"difference is {100 * deviation:.1f}% in the {_lower_first(row['label'])} (learned {row['display']['learned']} vs "
                    f"{shown}); this comparison sets the verdict because the "
                    f"{vbd_iterations if vbd_iterations is not None else 'tabulated'}-iteration VBD run is not converged for "
                    f"this gravity-driven case."
                )
        status = status or comparison
    elif status is None:
        status = "deviation"
        sentences.append("No metric could be compared between the two solvers.")
    analytic_rows = [row for row in rows if row["analytic"] is not None and row["values"]["learned"] is not None]
    for row in analytic_rows:
        parts = []
        for key, name in SOLVERS:
            value = row["values"][key]
            if value is None:
                continue
            relative = _relative(float(value) * row["scale"], row["analytic"])
            parts.append(f"{name} {row['display'][key]} ({_percent(relative)})")
        sentences.append(
            f"{row['label']} against the analytic reference {row['display']['analytic']}: " + ", ".join(parts) + "."
        )
    if scenario.name == "extension" and references:
        listed = ", ".join(
            f"{iterations} iterations {1e3 * float(entry['tip_displacement_final']):.2f} mm"
            for iterations, entry in sorted(references.items())
        )
        sentences.append(
            f"Newton VBD reruns with more Gauss-Seidel iterations per substep give tip displacements of {listed}; the "
            f"20-iteration VBD reference is therefore not converged for this gravity-driven case and its bias, not the "
            f"material model, explains most of its distance from the analytic value."
        )
    return {"status": status, "label": labels[status], "sentences": sentences}


def _plot_extension(series: list[dict], analytic_mm: float, output: Path) -> None:
    """Write the per-frame tip-displacement plot with matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 5.2), dpi=150)
    figure.patch.set_facecolor(_SURFACE)
    axis.set_facecolor(_SURFACE)
    end = 0.0
    for entry in series:
        axis.plot(
            entry["times"],
            entry["values_mm"],
            color=entry["color"],
            linewidth=2.0,
            solid_joinstyle="round",
            solid_capstyle="round",
            label=entry["label"],
        )
        end = max(end, float(entry["times"][-1]))
    axis.axhline(analytic_mm, color=_INK["muted"], linewidth=2.0, label=f"Analytic small strain {analytic_mm:.2f} mm")
    axis.text(
        1.01 * end,
        analytic_mm,
        f"analytic {analytic_mm:.2f} mm",
        color=_INK["secondary"],
        fontsize=9,
        va="bottom",
        ha="left",
        clip_on=False,
    )
    axis.set_xlabel("Physical time [s]", color=_INK["secondary"])
    axis.set_ylabel("Mean far-face tip displacement [mm]", color=_INK["secondary"])
    axis.set_title("Extension under gravity: tip displacement per recorded frame", color=_INK["primary"], loc="left")
    axis.grid(axis="y", color=_INK["grid"], linewidth=1.0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(_INK["axis"])
    axis.tick_params(colors=_INK["muted"], labelsize=9)
    axis.set_xlim(0.0, end * 1.12)
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=_INK["secondary"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, facecolor=_SURFACE)
    plt.close(figure)


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.samefile(source):
        return
    shutil.copy2(source, target)


def stage_renders(root: Path, names=None) -> dict:
    """Link ``renders/<solver>_<scenario>/simulation.mp4`` to ``renders/<solver>_<scenario>.mp4``.

    Existing staged clips are kept. Returns the staged clip path per
    ``(solver, scenario)`` for the clips that exist.
    """
    root = Path(root)
    staged = {}
    for name in names or scenarios.SCENARIOS:
        for solver, _ in SOLVERS:
            clip = root / "renders" / f"{solver}_{name}.mp4"
            source = root / "renders" / f"{solver}_{name}" / "simulation.mp4"
            if not clip.is_file() and source.is_file():
                try:
                    clip.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, clip)
            if clip.is_file():
                staged[(solver, name)] = clip
    return staged


def _ffmpeg_executable(explicit=None) -> str | None:
    if explicit:
        return str(explicit)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - Optional composition boundary.

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _frame_count(path: Path) -> int:
    import imageio_ffmpeg  # noqa: PLC0415 - Optional composition boundary.

    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    return int(frames)


def _label_image(text: str, path: Path) -> None:
    """Write a rounded dark label strip in the style of the renderer's time stamp."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 25)
    except OSError:
        font = ImageFont.load_default()
    probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    width, height = right - left + 28, bottom - top + 18
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=8, fill=(13, 29, 39, 235))
    draw.text((14 - left, 9 - top), text, font=font, fill=(244, 249, 251, 255))
    image.save(path)


def compose_side_by_side(left: Path, right: Path, output: Path, *, labels=None, ffmpeg=None) -> dict:
    """Stack two clips horizontally with ffmpeg, holding the last frame of the shorter clip.

    Args:
        left: Left clip (the learned solver by convention).
        right: Right clip (Newton VBD by convention).
        output: Output MP4 path; overwritten.
        labels: Two label strings drawn at the top right of each half;
            defaults to the display names in :data:`SOLVERS`.
        ffmpeg: ffmpeg executable; defaults to the one on PATH or the
            ``imageio_ffmpeg`` binary.

    Returns:
        ``{"frames": ..., "left_frames": ..., "right_frames": ..., "padded": ...}``.

    Raises:
        RuntimeError: If no ffmpeg executable is available or encoding fails.
    """
    left, right, output = Path(left), Path(right), Path(output)
    executable = _ffmpeg_executable(ffmpeg)
    if executable is None:
        raise RuntimeError("ffmpeg is not available")
    labels = tuple(labels) if labels is not None else tuple(name for _, name in SOLVERS)
    if len(labels) != 2:
        raise ValueError("labels must hold exactly two strings")
    left_frames, right_frames = _frame_count(left), _frame_count(right)
    if left_frames < 1 or right_frames < 1:
        raise RuntimeError("both clips must contain at least one frame")
    frames = max(left_frames, right_frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="side-by-side-") as directory:
        stage = Path(directory)
        _label_image(labels[0], stage / "left.png")
        _label_image(labels[1], stage / "right.png")
        graph = (
            f"[0:v]tpad=stop={frames - left_frames}:stop_mode=clone[l0];"
            f"[1:v]tpad=stop={frames - right_frames}:stop_mode=clone[r0];"
            "[l0][2:v]overlay=W-w-20:18[l];"
            "[r0][3:v]overlay=W-w-20:18[r];"
            "[l][r]hstack=inputs=2,format=yuv420p[v]"
        )
        command = [
            executable,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(left),
            "-i",
            str(right),
            "-i",
            str(stage / "left.png"),
            "-i",
            str(stage / "right.png"),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(stage / "composite.mp4"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}")
        shutil.move(str(stage / "composite.mp4"), output)
    return {
        "frames": frames,
        "left_frames": left_frames,
        "right_frames": right_frames,
        "padded": left_frames != right_frames,
    }


def compose_all(root: Path, names=None, *, ffmpeg=None) -> dict:
    """Compose ``renders/side_by_side_<scenario>.mp4`` for every scenario with both clips."""
    root = Path(root)
    staged = stage_renders(root, names)
    results = {}
    for name in names or scenarios.SCENARIOS:
        left, right = staged.get(("learned", name)), staged.get(("vbd", name))
        if left is None or right is None:
            continue
        results[name] = compose_side_by_side(left, right, root / "renders" / f"side_by_side_{name}.mp4", ffmpeg=ffmpeg)
        print(f"composed {name}: {results[name]}", flush=True)
    return results


def _reference_runs(root: Path) -> dict:
    """Return extension metrics of ``vbd_iter<K>`` reference runs keyed by iteration count."""
    references = {}
    for directory in sorted(Path(root).glob("vbd_iter*")):
        metrics = _read_json(directory / "extension" / "metrics.json")
        suffix = directory.name.removeprefix("vbd_iter")
        if metrics and suffix.isdigit() and metrics.get("tip_displacement_final") is not None:
            run = _read_json(directory / "extension" / "run.json") or {}
            metrics.setdefault("wall_seconds", run.get("wall_seconds"))
            references[int(suffix)] = metrics
    return references


_STYLE = """
body{font:17px/1.5 system-ui,sans-serif;max-width:1200px;margin:28px auto;padding:0 18px;color:#223844;background:#f5f8fa}
h1{line-height:1.2} .lead{max-width:85ch} .card{background:white;padding:22px;margin:24px 0;border-radius:12px;box-shadow:0 2px 14px #d8e2e8}
.badge{font-size:.7em;color:#85501e;margin-left:1em} .badge.agreement{color:#1f6f43} .badge.failure,.badge.inversion,.badge.disagreement{color:#a13a2f} .badge.missing{color:#526976}
video{width:100%;max-height:650px;background:#14252c;border-radius:8px}
.videos{display:flex;gap:18px} .videos figure{flex:1;margin:12px 0} figcaption,.caption{font-size:.85em;color:#526976}
table{border-collapse:collapse;width:100%;margin:14px 0;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #e3ebf0;vertical-align:top} th:first-child,td:first-child{text-align:left}
thead th{color:#526976;font-weight:600;font-size:.85em} td small{color:#526976}
.note{background:#f2f6f8;padding:12px 14px;border-radius:8px;font-size:.95em} .plot img{width:100%;border-radius:6px}
.failure{background:#fff0df;padding:12px;overflow-wrap:anywhere} a{color:#087b91} .verdicts li{margin:10px 0}
"""


def build_report(root: Path, *, site: Path | None = None, names=None, campaign: str = "v4") -> dict:
    """Write ``index.html`` and ``report.json`` into ``site`` and return the report summary.

    Args:
        root: Run root with the ``learned``, ``vbd`` and ``renders`` directories.
        site: Output directory; defaults to ``root / "site"``.
        names: Scenario names in page order; defaults to every registered scenario.
        campaign: Training campaign label used when naming the checkpoint.

    Returns:
        The machine-readable report also written to ``report.json``.
    """
    root = Path(root)
    site = root / "site" if site is None else Path(site)
    names = list(names or scenarios.SCENARIOS)
    site.mkdir(parents=True, exist_ok=True)
    staged = stage_renders(root, names)
    references = _reference_runs(root)
    learned_runs = [_read_json(root / "learned" / name / "run.json") for name in names]
    vbd_runs = [_read_json(root / "vbd" / name / "run.json") for name in names]
    learned_run = next((run for run in learned_runs if run), None) or {}
    vbd_run = next((run for run in vbd_runs if run), None) or {}
    checkpoint = {
        "campaign": campaign,
        "epoch": learned_run.get("checkpoint_epoch"),
        "path": learned_run.get("checkpoint"),
        "sha256": learned_run.get("checkpoint_sha256"),
        "iterations_per_substep": learned_run.get("iterations_per_substep"),
    }
    vbd_settings = {
        "iterations_per_substep": (vbd_run.get("solver") or {}).get("iterations_per_substep"),
        "newton_version": (vbd_run.get("environment") or {}).get("newton_version"),
        "newton_git_revision": (vbd_run.get("environment") or {}).get("newton_git_revision"),
    }
    sections = []
    report_scenarios = []
    verdict_items = []
    for index, name in enumerate(names):
        scenario = scenarios.SCENARIOS[name]
        metrics = {solver: _read_json(root / solver / name / "metrics.json") for solver, _ in SOLVERS}
        runs = {"learned": learned_runs[index], "vbd": vbd_runs[index]}
        states = {solver: _solver_state(metrics[solver], runs[solver]) for solver, _ in SOLVERS}
        rows = []
        for metric in _metric_rows(name):
            values = {solver: (metrics[solver] or {}).get(metric.key) for solver, _ in SOLVERS}
            difference, relative = _difference(values["learned"], values["vbd"], metric)
            both_negative = all(value is not None and float(value) < 0 for value in values.values())
            rows.append(
                {
                    "key": metric.key,
                    "label": metric.label,
                    "scale": metric.scale,
                    "unit": metric.unit,
                    "digits": metric.digits,
                    "jacobian": metric.jacobian,
                    "values": {solver: (None if value is None else float(value)) for solver, value in values.items()},
                    "analytic": metric.analytic,
                    "analytic_note": metric.analytic_note,
                    "display": {
                        **{
                            solver: ("no run" if metrics[solver] is None else _format(value, metric))
                            for solver, value in values.items()
                        },
                        "analytic": (
                            ""
                            if metric.analytic is None
                            else f"{metric.analytic:.{metric.digits}f}{' ' + metric.unit if metric.unit else ''}"
                        ),
                        "difference": difference,
                    },
                    "relative": relative,
                    "both_negative": both_negative,
                }
            )
        verdict = _verdict(scenario, rows, states, references, vbd_settings["iterations_per_substep"])
        videos = {}
        for solver, _ in SOLVERS:
            clip = staged.get((solver, name))
            if clip is not None:
                target = site / "videos" / f"{solver}_{name}.mp4"
                _copy(clip, target)
                videos[solver] = target.relative_to(site).as_posix()
                poster = root / "renders" / f"{solver}_{name}" / "initial.png"
                if poster.is_file():
                    _copy(poster, site / "posters" / f"{solver}_{name}.png")
                    videos[f"{solver}_poster"] = f"posters/{solver}_{name}.png"
        composite = root / "renders" / f"side_by_side_{name}.mp4"
        if composite.is_file() and len([solver for solver, _ in SOLVERS if solver in videos]) == 2:
            _copy(composite, site / "videos" / composite.name)
            videos["side_by_side"] = f"videos/{composite.name}"
        data_links = []
        for solver, label in SOLVERS:
            for kind in ("metrics", "run"):
                source = root / solver / name / f"{kind}.json"
                if source.is_file():
                    target = site / "data" / f"{solver}_{name}_{kind}.json"
                    _copy(source, target)
                    data_links.append((f"{label} {kind}", target.relative_to(site).as_posix()))
        plot = None
        if name == "extension":
            series = []
            for solver, label in SOLVERS:
                entry = metrics[solver]
                if entry and entry.get("tip_displacement_series") and entry.get("series_times"):
                    iterations = (
                        checkpoint["iterations_per_substep"]
                        if solver == "learned"
                        else vbd_settings["iterations_per_substep"]
                    )
                    suffix = f" ({iterations} iterations)" if iterations is not None else ""
                    series.append(
                        {
                            "label": label + suffix,
                            "times": [float(value) for value in entry["series_times"]],
                            "values_mm": [1e3 * float(value) for value in entry["tip_displacement_series"]],
                            "color": _SERIES_COLORS[solver],
                        }
                    )
            for iterations, entry in sorted(references.items()):
                color = _SERIES_COLORS.get(f"vbd_iter{iterations}")
                if color and entry.get("tip_displacement_series") and entry.get("series_times"):
                    series.append(
                        {
                            "label": f"Newton VBD ({iterations} iterations)",
                            "times": [float(value) for value in entry["series_times"]],
                            "values_mm": [1e3 * float(value) for value in entry["tip_displacement_series"]],
                            "color": color,
                        }
                    )
            if series:
                plot = "plots/extension_tip_displacement.png"
                _plot_extension(series, 1e3 * scenarios.analytic_extension_tip_displacement(), site / plot)
        sections.append(
            _section_html(
                scenario, rows, states, verdict, videos, data_links, plot, checkpoint, vbd_settings, references
            )
        )
        report_scenarios.append(
            {
                "name": name,
                "status": verdict["status"],
                "verdict": verdict["sentences"],
                "states": states,
                "rows": [
                    {
                        key: row[key]
                        for key in ("key", "label", "values", "analytic", "analytic_note", "display", "relative")
                    }
                    for row in rows
                ],
                "videos": videos,
                "plot": plot,
            }
        )
        verdict_items.append((scenario, verdict))
    counts = {}
    for _, verdict in verdict_items:
        counts[verdict["status"]] = counts.get(verdict["status"], 0) + 1
    report = {
        "title": TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pull_request": PULL_REQUEST_URL,
        "checkpoint": checkpoint,
        "vbd": vbd_settings,
        "reference_iterations": sorted(references),
        "scenarios": report_scenarios,
        "verdict_counts": counts,
    }
    page = _page_html(sections, verdict_items, checkpoint, vbd_settings, references)
    temporary = site / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(site / "report.json")
    temporary = site / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(site / "index.html")
    return report


def _video_html(videos: dict, scenario_name: str, checkpoint: dict, vbd_settings: dict) -> str:
    learned_iterations = checkpoint.get("iterations_per_substep")
    vbd_iterations = vbd_settings.get("iterations_per_substep")
    left = "Learned intrinsic solver" + (f" (K = {learned_iterations} per step)" if learned_iterations else "")
    right = "Newton VBD" + (f" ({vbd_iterations} iterations per substep)" if vbd_iterations else "")
    if "side_by_side" in videos:
        source = html.escape(videos["side_by_side"], quote=True)
        return (
            f'<video controls preload="metadata"><source src="{source}" type="video/mp4"><a href="{source}">Download video</a></video>'
            f'<p class="caption">Left: {html.escape(left)}. Right: {html.escape(right)}. Both halves show the same recorded '
            f"frames at 30 fps with the physical time stamped in the corner; the shorter clip, if any, holds its last frame.</p>"
        )
    figures = []
    for solver, label in SOLVERS:
        if solver not in videos:
            figures.append(
                f'<figure><div class="failure">No {html.escape(label)} clip for {html.escape(scenario_name)}.</div>'
                f"<figcaption>{html.escape(label)}</figcaption></figure>"
            )
            continue
        source = html.escape(videos[solver], quote=True)
        poster = videos.get(f"{solver}_poster")
        poster_attribute = f' poster="{html.escape(poster, quote=True)}"' if poster else ""
        caption = left if solver == "learned" else right
        figures.append(
            f'<figure><video controls preload="metadata"{poster_attribute}><source src="{source}" type="video/mp4">'
            f'<a href="{source}">Download video</a></video><figcaption>{html.escape(caption)}</figcaption></figure>'
        )
    return f'<div class="videos">{"".join(figures)}</div>'


def _reference_table(references: dict, rows: list[dict]) -> str:
    """Return the table of Newton VBD extension reruns with more iterations per substep."""
    if not references:
        return ""
    header = "".join(f"<th>{html.escape(row['label'])}</th>" for row in rows)
    body = []
    for iterations, entry in sorted(references.items()):
        cells = []
        for row in rows:
            value = entry.get(row["key"])
            cells.append(
                "<td>n/a</td>"
                if value is None
                else f"<td>{float(value) * row['scale']:.{row['digits']}f}{' ' + row['unit'] if row['unit'] else ''}</td>"
            )
        wall = entry.get("wall_seconds")
        cells.append("<td>n/a</td>" if wall is None else f"<td>{float(wall):.0f} s</td>")
        body.append(f"<tr><td>{iterations}</td>{''.join(cells)}</tr>")
    return (
        "<h3>Newton VBD with more iterations per substep</h3>"
        f"<table><thead><tr><th>Iterations</th>{header}<th>Wall time</th></tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _section_html(
    scenario, rows, states, verdict, videos, data_links, plot, checkpoint, vbd_settings, references
) -> str:
    title = scenario.name.replace("_", " and ") if scenario.name == "compression_release" else scenario.name
    table_rows = []
    for row in rows:
        analytic = row["display"]["analytic"]
        if analytic and row["analytic_note"]:
            analytic = f"{html.escape(analytic)}<br><small>{html.escape(row['analytic_note'])}</small>"
        table_rows.append(
            f"<tr><td>{html.escape(row['label'])}</td><td>{html.escape(row['display']['learned'])}</td>"
            f"<td>{html.escape(row['display']['vbd'])}</td><td>{analytic or '<small>none</small>'}</td>"
            f"<td>{html.escape(row['display']['difference'])}</td></tr>"
        )
    state_lines = "".join(
        f"<li>{html.escape(name)}: {html.escape(states[solver]['text'])}.</li>" for solver, name in SOLVERS
    )
    plot_html = (
        f'<div class="plot"><img src="{html.escape(plot, quote=True)}" alt="Tip displacement per frame for the extension scenario">'
        f'<p class="caption">Mean z displacement of the far face per recorded frame. The grey line is the small-strain analytic '
        f"value; additional Newton VBD series, when present, rerun the same scenario with more iterations per substep.</p></div>"
        if plot
        else ""
    )
    links = " · ".join(
        f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>' for label, href in data_links
    )
    clip_links = " · ".join(
        f'<a href="{html.escape(videos[solver], quote=True)}">{html.escape(label)} clip</a>'
        for solver, label in SOLVERS
        if solver in videos
    )
    downloads = " · ".join(part for part in (clip_links, links) if part)
    return (
        f'<article class="card" id="{html.escape(scenario.name)}"><h2>{html.escape(title.capitalize())} '
        f'<span class="badge {html.escape(verdict["status"])}">{html.escape(verdict["label"])}</span></h2>'
        f"<p>{html.escape(_schedule_text(scenario))}</p>"
        f"{_video_html(videos, scenario.name, checkpoint, vbd_settings)}"
        f"<table><thead><tr><th>Metric</th><th>Learned</th><th>Newton VBD</th><th>Analytic</th><th>Difference (learned minus VBD)</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>"
        f"{_reference_table(references, rows) if scenario.name == 'extension' else ''}"
        f"<ul>{state_lines}</ul>{plot_html}"
        f'<p class="note">{" ".join(html.escape(sentence) for sentence in verdict["sentences"])}</p>'
        + (f"<p>{downloads}</p>" if downloads else "")
        + "</article>"
    )


def _page_html(sections, verdict_items, checkpoint, vbd_settings, references) -> str:
    epoch = checkpoint.get("epoch")
    checkpoint_text = (
        f"epoch {epoch} checkpoint of the {checkpoint['campaign']} training campaign"
        if epoch is not None
        else f"{checkpoint['campaign']} checkpoint (epoch unknown until a learned run.json is on disk)"
    )
    if checkpoint.get("sha256"):
        checkpoint_text += f" (sha256 {checkpoint['sha256'][:12]})"
    learned_iterations = checkpoint.get("iterations_per_substep")
    vbd_iterations = vbd_settings.get("iterations_per_substep")
    newton = vbd_settings.get("newton_version")
    revision = vbd_settings.get("newton_git_revision")
    newton_text = f"Newton {newton}" + (f" at commit {revision[:10]}" if revision else "") if newton else "Newton"
    grid = " x ".join(str(count) for count in scenarios.CELL_COUNTS)
    intro = (
        f'<p class="lead">Newton <a href="{PULL_REQUEST_URL}">pull request #2901</a> adds volumetric Neo-Hookean correctness '
        f"tests for the VBD solver. The four that apply to a clamped beam are run here on the beam the learned intrinsic hex "
        f"solver was trained on, once with the learned solver and once with {html.escape(newton_text)} SolverVBD, using identical "
        f"schedules, material and time stepping. Beam: {grid} hex cells of edge {scenarios.CELL_SIZE:g} m "
        f"({scenarios.BEAM_WIDTH:g} x {scenarios.BEAM_WIDTH:g} x {scenarios.BEAM_LENGTH:g} m, {math.prod(scenarios.CELL_COUNTS)} cells), "
        f"clamped on the face z = 0. Material: E = {scenarios.YOUNG_MODULUS:.1e} Pa, nu = {scenarios.POISSON_RATIO}, "
        f"lambda = {scenarios.LAME_LAMBDA / 1e3:.1f} kPa, mu = {scenarios.LAME_MU / 1e3:.1f} kPa, rho = {scenarios.DENSITY:g} kg/m^3, "
        f"damping {scenarios.DAMPING:g} Pa s. Time step dt = 1/{scenarios.FPS * scenarios.SUBSTEPS} s, {scenarios.SUBSTEPS} substeps "
        f"per recorded frame at {scenarios.FPS} fps. The learned solver uses the {html.escape(checkpoint_text)} with "
        f"K = {learned_iterations if learned_iterations is not None else '?'} learned iterations per substep; Newton VBD uses "
        f"{vbd_iterations if vbd_iterations is not None else '?'} Gauss-Seidel iterations per substep.</p>"
        "<p>Each clip shows the beam surface, its voxel grid and the clamped corners in orange; the driven far face is not "
        "drawn as pinned because it moves. Metric definitions: V/V0 is the exactly integrated trilinear hex volume over the "
        "rest volume; the centre Jacobian ratio is det F at the cell centre, one at rest and negative for an inverted "
        "cell; the tip displacement is the mean z displacement of the far-face corners.</p>"
    )
    verdict_list = "".join(
        f"<li><strong>{html.escape(scenario.name.replace('_', ' and ') if scenario.name == 'compression_release' else scenario.name)}"
        f'</strong> <span class="badge {html.escape(verdict["status"])}">{html.escape(verdict["label"])}</span>: '
        f"{' '.join(html.escape(sentence) for sentence in verdict['sentences'])}</li>"
        for scenario, verdict in verdict_items
    )
    counts = {}
    for _, verdict in verdict_items:
        counts[verdict["label"]] = counts.get(verdict["label"], 0) + 1
    count_text = ", ".join(f"{count} {label.lower()}" for label, count in counts.items())
    reference_note = (
        f"<p>Extra Newton VBD extension runs at {', '.join(str(k) for k in sorted(references))} iterations per substep are "
        f"included as references in the extension plot and verdict.</p>"
        if references
        else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(TITLE)}</title><style>{_STYLE}</style></head><body><p><a href="../index.html">← All solver experiments</a></p>
<h1>{html.escape(TITLE)}</h1>
{intro}{reference_note}
{"".join(sections)}
<article class="card"><h2>Summary</h2><p>Verdicts over {len(verdict_items)} scenarios: {html.escape(count_text)}. A verdict of agreement means the learned and Newton VBD metrics differ by at most {100 * AGREEMENT_TOLERANCE:.0f}% relative to VBD; deviation means up to {100 * LARGE_DEVIATION:.0f}%; disagreement means more. Cell inversion and failure are reported whenever either solver shows them, regardless of how close the remaining metrics are.</p>
<ul class="verdicts">{verdict_list}</ul></article>
<p><a href="report.json">Machine-readable results</a></p></body></html>
"""


def _load_publisher(script: Path):
    script = Path(script)
    if not script.is_file():
        raise FileNotFoundError(f"publisher script not found: {script}")
    spec = importlib.util.spec_from_file_location("_artifact_publisher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish_site(
    site: Path,
    *,
    slug: str,
    project: str = PROJECT_SLUG,
    kanna_dist=None,
    base_url: str = "https://ankachen.com",
    publisher_script: Path = DEFAULT_PUBLISHER_SCRIPT,
) -> dict:
    """Copy ``site`` to ``<kanna_dist>/artifacts/<project>/<slug>`` and return the public URLs.

    Uses ``discover_kanna_dist`` and ``copy_source`` from the publish-artifact
    helper script, exactly like ``publish_mixed_report``. The returned
    dictionary holds the target path, the index URL, the video URLs and the
    HTTP status of each URL as reported by the helper's ``verify_url``.
    """
    site = Path(site)
    if not (site / "index.html").is_file():
        raise FileNotFoundError(f"{site} does not contain index.html")
    publisher = _load_publisher(publisher_script)
    if not publisher.SLUG_RE.fullmatch(slug) or "/" in slug:
        raise ValueError("slug must contain only letters, digits, dots, underscores and hyphens")
    root = publisher.discover_kanna_dist(kanna_dist)
    target = root / "artifacts" / project / slug
    publisher.copy_source(site, target, "index.html", False, False)
    urls = {"index.html": publisher.artifact_url(base_url, f"{project}/{slug}", "index.html")}
    for video in sorted((site / "videos").glob("*.mp4")):
        relative = video.relative_to(site).as_posix()
        urls[relative] = publisher.artifact_url(base_url, f"{project}/{slug}", relative)
    return {
        "target": str(target),
        "kanna_dist": str(root),
        "urls": urls,
        "status": {name: publisher.verify_url(url) for name, url in urls.items()},
    }


def add_index_card(index_path: Path, *, href: str, badge: str, title: str, description: str, marker: str) -> bool:
    """Insert a card at the top of the project index unless one with ``marker`` exists.

    The deployed project index is a hand-maintained page; new galleries are
    added as the first ``<a class="card">`` inside the cards section, each
    with a ``data-<marker>`` attribute so a rerun does not duplicate it.

    Returns:
        ``True`` when the card was inserted, ``False`` when it was already present.
    """
    index_path = Path(index_path)
    text = index_path.read_text(encoding="utf-8")
    attribute = f"data-{marker}"
    if f" {attribute}>" in text or f" {attribute} " in text:
        return False
    anchor = '<section class="cards" aria-label="Project webpages">'
    if anchor not in text:
        raise ValueError("project index does not contain the cards section")
    card = (
        f'<a class="card" href="{html.escape(href, quote=True)}" {attribute}><span class="badge">{html.escape(badge)}</span>'
        f'<h2>{html.escape(title)}</h2><p>{html.escape(description)}</p><span class="open">Open the comparison →</span></a>'
    )
    text = text.replace(anchor, anchor + card, 1)
    temporary = index_path.with_name(index_path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(index_path)
    return True


def _card_description(report: dict) -> str:
    verdicts = ", ".join(f"{entry['name'].replace('_', ' and ')}: {entry['status']}" for entry in report["scenarios"])
    return (
        "Four beam scenarios from Newton PR #2901 (gravity extension, stretch to 2 L, one full twist, compression to L/2 "
        "and release) run on the training beam with the learned solver and Newton SolverVBD under identical schedules. "
        f"Side-by-side clips, metrics tables with analytic references, tip-displacement plot. Verdicts: {verdicts}."
    )


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True, help="run root, e.g. generated/fem_accuracy_20260930")
    parser.add_argument("--site", type=Path, help="site output directory (default <root>/site)")
    parser.add_argument(
        "--scenarios", nargs="+", choices=sorted(scenarios.SCENARIOS), help="scenario subset in page order"
    )
    parser.add_argument("--campaign", default="v4")
    parser.add_argument("--compose", action="store_true", help="compose side-by-side clips with ffmpeg first")
    parser.add_argument("--ffmpeg", help="ffmpeg executable (default PATH or imageio_ffmpeg)")
    parser.add_argument("--publish", action="store_true", help="copy the site into Kanna's artifacts route")
    parser.add_argument("--slug", default="fem-accuracy-20260930")
    parser.add_argument("--kanna-dist")
    parser.add_argument("--base-url", default="https://ankachen.com")
    parser.add_argument("--publisher-script", type=Path, default=DEFAULT_PUBLISHER_SCRIPT)
    parser.add_argument("--index-card", action="store_true", help="add a card to the project index in the Kanna dist")
    args = parser.parse_args()
    names = args.scenarios or list(scenarios.SCENARIOS)
    if args.compose:
        compose_all(args.root, names, ffmpeg=args.ffmpeg)
    report = build_report(args.root, site=args.site, names=names, campaign=args.campaign)
    site = args.root / "site" if args.site is None else args.site
    print(
        json.dumps({"site": str(site), "verdicts": {entry["name"]: entry["status"] for entry in report["scenarios"]}})
    )
    if args.publish:
        published = publish_site(
            site,
            slug=args.slug,
            kanna_dist=args.kanna_dist,
            base_url=args.base_url,
            publisher_script=args.publisher_script,
        )
        print(json.dumps(published, indent=2))
        if args.index_card:
            index_path = Path(published["kanna_dist"]) / "artifacts" / PROJECT_SLUG / "index.html"
            epoch = report["checkpoint"].get("epoch")
            learned_iterations = report["checkpoint"].get("iterations_per_substep")
            vbd_iterations = report["vbd"].get("iterations_per_substep")
            badge = (
                f"FEM accuracy · {args.campaign}"
                + (f" · epoch {epoch}" if epoch is not None else "")
                + (f" · learned K = {learned_iterations}" if learned_iterations is not None else "")
                + (f" vs Newton VBD {vbd_iterations} iterations" if vbd_iterations is not None else " vs Newton VBD")
            )
            inserted = add_index_card(
                index_path,
                href=f"{args.slug}/index.html",
                badge=badge,
                title="Volumetric correctness: learned solver vs Newton VBD",
                description=_card_description(report),
                marker=args.slug,
            )
            print(json.dumps({"index_card": "inserted" if inserted else "already present", "index": str(index_path)}))


if __name__ == "__main__":
    _main()
