# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render accepted free-body reference shards as a small HTML video gallery.

The renderer is deliberately a replay tool: it never rebuilds or advances a
solver.  Every rendered surface comes from an exact ``q`` state stored in an
accepted reference shard.  ``render-all`` launches ``render-one`` in a fresh
subprocess for each MP4 because consecutive headless ViewerGL contexts in one
process can produce black frames.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Sequence

import numpy as np

_INDEX_SCHEMA = "pss-free-body-reference-index-v1"
_SHARD_SCHEMA = "pss-free-body-reference-shard-v1"
_EVIDENCE_SCHEMA = "pss-free-body-reference-evidence-v1"
_GALLERY_SCHEMA = "pss-free-body-reference-gallery-v1"
_EXPECTED_STEP_COUNT = 8
_EXPECTED_STORED_STATE_COUNT = _EXPECTED_STEP_COUNT + 1
_EXPECTED_DT_EXPRESSION = "1/300"
_DEFAULT_FPS = 30
_DEFAULT_HOLD_FRAMES = 6
_DEFAULT_FOV_DEGREES = 34.0
_VIEW_DIRECTION = np.array((1.35, -1.75, 1.15), dtype=np.float64)


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    dtype = array.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    return np.array(array, dtype=canonical_dtype, order="C", copy=True)


def array_sha256(value: np.ndarray) -> str:
    """Hash an array with the producer's canonical dtype/shape contract."""
    array = _canonical_array(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_scalar(value: np.ndarray, dtype: np.dtype, name: str) -> tuple[float, tuple[int, ...], bool]:
    """Read one exact scalar and report legacy singleton-vector storage.

    Canonical reference shards store scalar fields with shape ``()``.  The
    already accepted preview corpus predates that correction and stores them
    with shape ``(1,)``.  Both representations are authenticated exactly;
    every other shape and dtype is rejected.
    """
    array = np.asarray(value)
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
    if array.shape == ():
        legacy_singleton_vector = False
    elif array.shape == (1,):
        legacy_singleton_vector = True
    else:
        raise ValueError(f"{name} must have canonical shape () or legacy shape (1,), got {array.shape}")
    scalar = float(array.item())
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar, tuple(array.shape), legacy_singleton_vector


def _require_file_hash(path: pathlib.Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def ping_pong_stored_state_indices(step_count: int, hold_frames: int) -> tuple[int, ...]:
    """Return a loop schedule containing stored-state indices only.

    For eight physical transitions this traverses ``q[0]..q[8]`` and then
    ``q[7]..q[1]``.  Repeating each index slows playback without generating
    interpolated geometry.  The next loop begins at ``q[0]`` without a jump.
    """
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 1:
        raise ValueError("step_count must be a positive integer")
    if isinstance(hold_frames, bool) or not isinstance(hold_frames, int) or hold_frames < 1:
        raise ValueError("hold_frames must be a positive integer")
    one_cycle = tuple(range(step_count + 1)) + tuple(range(step_count - 1, 0, -1))
    return tuple(index for index in one_cycle for _ in range(hold_frames))


@dataclasses.dataclass(frozen=True)
class Camera:
    """One fixed camera derived from the bounds of every stored state."""

    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_degrees: float
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable camera record."""
        return dataclasses.asdict(self)


def camera_from_sequence_bounds(q: np.ndarray, *, fov_degrees: float = _DEFAULT_FOV_DEGREES) -> Camera:
    """Fit a deterministic isometric camera to the complete sequence bounds."""
    positions = np.asarray(q, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 2:
        raise ValueError("q must have shape (stored_state_count, vertex_count, 3)")
    if positions.shape[1] < 4 or not np.isfinite(positions).all():
        raise ValueError("q must contain at least four finite vertices")
    if not math.isfinite(fov_degrees) or not 10.0 <= fov_degrees <= 80.0:
        raise ValueError("fov_degrees must lie in [10, 80]")

    bounds_min = positions.min(axis=(0, 1))
    bounds_max = positions.max(axis=(0, 1))
    extent = bounds_max - bounds_min
    radius = 0.5 * float(np.linalg.norm(extent))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("full-sequence bounds must have positive extent")

    target = 0.5 * (bounds_min + bounds_max)
    view_direction = _VIEW_DIRECTION / np.linalg.norm(_VIEW_DIRECTION)
    # A bounding sphere fit is independent of object orientation.  The 1.18
    # margin leaves room for the q0 outline and the video overlay.
    distance = 1.18 * radius / math.sin(math.radians(fov_degrees) / 2.0)
    position = target + view_direction * distance
    return Camera(
        position=tuple(float(value) for value in position),
        target=tuple(float(value) for value in target),
        fov_degrees=float(fov_degrees),
        bounds_min=tuple(float(value) for value in bounds_min),
        bounds_max=tuple(float(value) for value in bounds_max),
    )


@dataclasses.dataclass(frozen=True)
class SourceSequence:
    """Authenticated paths and metadata for one accepted reference sequence."""

    asset_id: str
    sequence_id: str
    vertex_count: int
    tet_count: int
    deformation_seed: int
    velocity_seed: int
    source_sha256: str
    topology_sha256: str
    material_sha256: str
    operator_sha256: str
    protocol_sha256: str
    static_npz: pathlib.Path
    static_npz_sha256: str
    sequence_npz: pathlib.Path
    sequence_npz_sha256: str
    manifest_json: pathlib.Path
    manifest_json_sha256: str
    evidence_json: pathlib.Path
    evidence_json_sha256: str
    q_sha256: str
    dt_seconds: float
    dt_storage_shape: tuple[int, ...]
    dt_legacy_singleton_vector: bool
    step_count: int

    @property
    def stem(self) -> str:
        """Return a collision-free media stem."""
        return f"{self.asset_id}--{self.sequence_id}"


def _load_json(path: pathlib.Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_source_sequences(source_root: pathlib.Path) -> tuple[str, tuple[SourceSequence, ...]]:
    """Authenticate and enumerate every accepted sequence in a source index."""
    source_root = source_root.resolve()
    index_path = source_root / "index.json"
    index = _load_json(index_path)
    if index.get("schema") != _INDEX_SCHEMA:
        raise ValueError(f"unsupported source index schema: {index.get('schema')!r}")
    assets = index.get("assets")
    if not isinstance(assets, list):
        raise ValueError("source index assets must be a list")

    records: list[SourceSequence] = []
    seen: set[tuple[str, str]] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("every source asset must be a JSON object")
        asset_id = str(asset["asset_id"])
        identities = asset["identities"]
        if not isinstance(identities, dict):
            raise ValueError(f"{asset_id} identities must be a JSON object")
        static_info = asset["static_npz"]
        if not isinstance(static_info, dict):
            raise ValueError(f"{asset_id} static_npz must be a JSON object")
        static_path = source_root / str(static_info["path"])
        static_hash = str(static_info["sha256"])
        _require_file_hash(static_path, static_hash, f"{asset_id} static_npz")

        sequences = asset.get("sequences")
        if not isinstance(sequences, list):
            raise ValueError(f"{asset_id} sequences must be a list")
        for sequence in sequences:
            if not isinstance(sequence, dict):
                raise ValueError(f"{asset_id} sequence must be a JSON object")
            sequence_id = str(sequence["sequence_id"])
            key = (asset_id, sequence_id)
            if key in seen:
                raise ValueError(f"duplicate source sequence {asset_id}/{sequence_id}")
            seen.add(key)

            manifest_info = sequence["manifest"]
            sequence_info = sequence["sequence_npz"]
            if not isinstance(manifest_info, dict) or not isinstance(sequence_info, dict):
                raise ValueError(f"{asset_id}/{sequence_id} file records must be JSON objects")
            manifest_path = source_root / str(manifest_info["path"])
            sequence_path = source_root / str(sequence_info["path"])
            manifest_hash = str(manifest_info["sha256"])
            sequence_hash = str(sequence_info["sha256"])
            _require_file_hash(manifest_path, manifest_hash, f"{asset_id}/{sequence_id} manifest")
            _require_file_hash(sequence_path, sequence_hash, f"{asset_id}/{sequence_id} sequence_npz")

            manifest = _load_json(manifest_path)
            if manifest.get("schema") != _SHARD_SCHEMA or manifest.get("reference_accepted") is not True:
                raise ValueError(f"{asset_id}/{sequence_id} is not an accepted reference shard")
            if manifest.get("asset_id") != asset_id or manifest.get("sequence_id") != sequence_id:
                raise ValueError(f"{asset_id}/{sequence_id} manifest identity mismatch")
            if int(manifest.get("step_count", -1)) != _EXPECTED_STEP_COUNT:
                raise ValueError(f"{asset_id}/{sequence_id} must contain exactly {_EXPECTED_STEP_COUNT} steps")
            protocol = manifest.get("protocol")
            if not isinstance(protocol, dict) or protocol.get("requested_dt_expression") != _EXPECTED_DT_EXPRESSION:
                raise ValueError(f"{asset_id}/{sequence_id} does not use dt={_EXPECTED_DT_EXPRESSION}")

            files = manifest.get("files")
            if not isinstance(files, dict):
                raise ValueError(f"{asset_id}/{sequence_id} manifest files must be a JSON object")
            manifest_sequence = files.get("sequence_npz")
            manifest_static = files.get("static_npz")
            evidence_info = files.get("evidence_json")
            if not all(isinstance(value, dict) for value in (manifest_sequence, manifest_static, evidence_info)):
                raise ValueError(f"{asset_id}/{sequence_id} manifest file records are incomplete")
            assert isinstance(manifest_sequence, dict)
            assert isinstance(manifest_static, dict)
            assert isinstance(evidence_info, dict)
            if manifest_sequence.get("sha256") != sequence_hash or manifest_static.get("sha256") != static_hash:
                raise ValueError(f"{asset_id}/{sequence_id} manifest/index file hash mismatch")

            evidence_path = manifest_path.parent / str(evidence_info["path"])
            evidence_hash = str(evidence_info["sha256"])
            _require_file_hash(evidence_path, evidence_hash, f"{asset_id}/{sequence_id} evidence")
            evidence = _load_json(evidence_path)
            steps = evidence.get("steps")
            if evidence.get("schema") != _EVIDENCE_SCHEMA or not isinstance(steps, list):
                raise ValueError(f"{asset_id}/{sequence_id} has invalid evidence")
            if len(steps) != _EXPECTED_STEP_COUNT or any(
                not isinstance(step, dict)
                or step.get("reference_accepted") is not True
                or step.get("reference_failures") != []
                for step in steps
            ):
                raise ValueError(f"{asset_id}/{sequence_id} evidence does not accept every step")

            arrays = manifest_sequence.get("arrays")
            if (
                not isinstance(arrays, dict)
                or not isinstance(arrays.get("q"), dict)
                or not isinstance(arrays.get("dt"), dict)
            ):
                raise ValueError(f"{asset_id}/{sequence_id} manifest lacks q or dt inventory")
            q_record = arrays["q"]
            dt_record = arrays["dt"]
            assert isinstance(q_record, dict) and isinstance(dt_record, dict)
            with np.load(sequence_path, allow_pickle=False) as sequence_data:
                q = np.asarray(sequence_data["q"])
                dt = np.asarray(sequence_data["dt"])
            if q.shape != (_EXPECTED_STORED_STATE_COUNT, int(asset["vertex_count"]), 3) or q.dtype != np.float64:
                raise ValueError(f"{asset_id}/{sequence_id} q must be float64 with nine stored states")
            if not np.array_equal(q.astype(np.float32).astype(np.float64), q):
                raise ValueError(f"{asset_id}/{sequence_id} q is not a lossless promotion of SolverVBD float32")
            q_hash = array_sha256(q)
            if q_hash != q_record.get("sha256"):
                raise ValueError(f"{asset_id}/{sequence_id} q inventory hash mismatch")
            dt_seconds, dt_storage_shape, dt_legacy = _exact_scalar(dt, np.float32, "dt")
            if (
                dt_record.get("dtype") != dt.dtype.str
                or dt_record.get("shape") != list(dt.shape)
                or dt_record.get("nbytes") != dt.nbytes
                or dt_record.get("sha256") != array_sha256(dt)
            ):
                raise ValueError(f"{asset_id}/{sequence_id} dt inventory mismatch")
            if dt_seconds != float(np.float32(1.0 / 300.0)):
                raise ValueError(f"{asset_id}/{sequence_id} has an unexpected execution dt")

            records.append(
                SourceSequence(
                    asset_id=asset_id,
                    sequence_id=sequence_id,
                    vertex_count=int(asset["vertex_count"]),
                    tet_count=int(asset["tet_count"]),
                    deformation_seed=int(sequence["deformation_seed"]),
                    velocity_seed=int(sequence["velocity_seed"]),
                    source_sha256=str(asset["source_sha256"]),
                    topology_sha256=str(identities["topology_sha256"]),
                    material_sha256=str(identities["material_sha256"]),
                    operator_sha256=str(identities["operator_sha256"]),
                    protocol_sha256=str(identities["protocol_sha256"]),
                    static_npz=static_path,
                    static_npz_sha256=static_hash,
                    sequence_npz=sequence_path,
                    sequence_npz_sha256=sequence_hash,
                    manifest_json=manifest_path,
                    manifest_json_sha256=manifest_hash,
                    evidence_json=evidence_path,
                    evidence_json_sha256=evidence_hash,
                    q_sha256=q_hash,
                    dt_seconds=dt_seconds,
                    dt_storage_shape=dt_storage_shape,
                    dt_legacy_singleton_vector=dt_legacy,
                    step_count=_EXPECTED_STEP_COUNT,
                )
            )

    expected_count = int(index.get("accepted_sequence_count", -1))
    if expected_count != len(records) or int(index.get("asset_count", -1)) != len(assets):
        raise ValueError("source index counts do not match its authenticated contents")
    return file_sha256(index_path), tuple(records)


def _find_sequence(records: Sequence[SourceSequence], asset_id: str, sequence_id: str) -> SourceSequence:
    matches = [record for record in records if record.asset_id == asset_id and record.sequence_id == sequence_id]
    if len(matches) != 1:
        raise ValueError(f"expected one source sequence for {asset_id}/{sequence_id}, found {len(matches)}")
    return matches[0]


def _q0_outline_edges(q0: np.ndarray, triangles: np.ndarray, camera_position: Sequence[float]) -> np.ndarray:
    """Select q0 silhouette, crease, and open-boundary edges.

    Drawing every triangulation edge obscures the solid surface on dense
    assets.  This deterministic view-dependent selection retains the useful
    before/after outline while leaving PBR shading visible.
    """
    positions = np.asarray(q0, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int32)
    camera = np.asarray(camera_position, dtype=np.float64)
    raw_normals = np.cross(
        positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]]
    )
    lengths = np.linalg.norm(raw_normals, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("q0 surface contains a degenerate triangle")
    normals = raw_normals / lengths[:, None]
    centroids = positions[faces].mean(axis=1)
    facing = np.einsum("ij,ij->i", normals, camera[None, :] - centroids)

    adjacent_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(first, second)), int(max(first, second)))
            adjacent_faces.setdefault(edge, []).append(face_index)

    crease_cosine = math.cos(math.radians(35.0))
    selected: list[tuple[int, int]] = []
    for edge, adjacency in sorted(adjacent_faces.items()):
        if len(adjacency) != 2:
            selected.append(edge)
            continue
        first, second = adjacency
        is_silhouette = (facing[first] >= 0.0) != (facing[second] >= 0.0)
        is_crease = float(np.dot(normals[first], normals[second])) < crease_cosine
        if is_silhouette or is_crease:
            selected.append(edge)
    if not selected:
        raise ValueError("q0 outline selection unexpectedly produced no edges")
    return np.asarray(selected, dtype=np.int32)


class _MP4Writer:
    """Small imageio-ffmpeg writer that does not require the imageio package."""

    def __init__(self, path: pathlib.Path, *, width: int, height: int, fps: int):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self._generator = None

    def __enter__(self):
        import imageio_ffmpeg  # noqa: PLC0415 -- installed by the capture skill

        self._generator = imageio_ffmpeg.write_frames(
            self.path,
            (self.width, self.height),
            fps=self.fps,
            quality=9,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=2,
            output_params=["-movflags", "+faststart"],
        )
        self._generator.send(None)
        return self

    def write_frame(self, frame: np.ndarray) -> None:
        array = np.asarray(frame)
        if array.shape != (self.height, self.width, 3) or array.dtype != np.uint8:
            raise ValueError("video frame does not match the declared RGB uint8 dimensions")
        assert self._generator is not None
        self._generator.send(np.ascontiguousarray(array))

    def __exit__(self, *_exc) -> None:
        if self._generator is not None:
            self._generator.close()
            self._generator = None


def _font(size: int):
    from PIL import ImageFont  # noqa: PLC0415 -- capture-only dependency

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _overlay_frame(
    frame: np.ndarray,
    *,
    asset_id: str,
    sequence_id: str,
    state_index: int,
    step_count: int,
    dt_seconds: float,
) -> np.ndarray:
    from PIL import Image, ImageDraw  # noqa: PLC0415 -- capture-only dependency

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(max(18, frame.shape[0] // 28))
    detail_font = _font(max(14, frame.shape[0] // 40))
    pad = max(12, frame.shape[0] // 45)
    box_height = max(72, frame.shape[0] // 7)
    draw.rounded_rectangle(
        (pad, pad, min(frame.shape[1] - pad, pad + frame.shape[1] * 0.57), pad + box_height),
        radius=10,
        fill=(8, 14, 24, 196),
    )
    draw.text((pad * 2, pad * 1.55), f"{asset_id}  /  {sequence_id}", font=title_font, fill=(240, 247, 255, 255))
    physical_time = state_index * dt_seconds
    detail = f"stored q[{state_index}]  ·  physical t={physical_time:.6f} s"
    draw.text((pad * 2, pad * 1.55 + title_font.size + 5), detail, font=detail_font, fill=(157, 224, 255, 255))
    warning = f"SLOW PLAYBACK — NOT REAL TIME  ·  {step_count} steps at dt=1/300 s"
    warning_box = draw.textbbox((0, 0), warning, font=detail_font)
    text_width = warning_box[2] - warning_box[0]
    text_height = warning_box[3] - warning_box[1]
    x0 = max(pad, (frame.shape[1] - text_width) // 2 - pad)
    y0 = frame.shape[0] - text_height - pad * 2.2
    draw.rounded_rectangle((x0, y0, x0 + text_width + 2 * pad, frame.shape[0] - pad), radius=8, fill=(8, 14, 24, 205))
    draw.text((x0 + pad, y0 + pad * 0.45), warning, font=detail_font, fill=(255, 213, 120, 255))
    return np.asarray(image.convert("RGB"))


def render_one(
    *,
    source_root: pathlib.Path,
    output_root: pathlib.Path,
    asset_id: str,
    sequence_id: str,
    width: int,
    height: int,
    fps: int,
    hold_frames: int,
    device: str,
) -> pathlib.Path:
    """Render one MP4 and poster in the current process."""
    # Imports that initialize Warp/GL stay inside this one-video subprocess.
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415

    tools_dir = pathlib.Path(os.environ.get("AI_LOGS", "/home/horde/Code/AI-Docs/AI-Logs")) / "Newton/tools"
    sys.path.insert(0, str(tools_dir))
    from newton_capture import Capture  # noqa: PLC0415

    _, records = load_source_sequences(source_root)
    record = _find_sequence(records, asset_id, sequence_id)
    with np.load(record.static_npz, allow_pickle=False) as static_data:
        rest_q = np.asarray(static_data["rest_q"], dtype=np.float64)
        triangles = np.asarray(static_data["boundary_triangles"], dtype=np.int32)
    with np.load(record.sequence_npz, allow_pickle=False) as sequence_data:
        q = np.asarray(sequence_data["q"], dtype=np.float64)
    camera = camera_from_sequence_bounds(q)
    schedule = ping_pong_stored_state_indices(record.step_count, hold_frames)

    wp.init()
    builder = newton.ModelBuilder(gravity=0.0)
    vertex_count = q.shape[1]
    builder.add_particles(
        pos=[wp.vec3(*position) for position in rest_q.astype(np.float32)],
        vel=[wp.vec3(0.0, 0.0, 0.0)] * vertex_count,
        mass=[1.0] * vertex_count,
        radius=[0.0] * vertex_count,
        flags=[int(newton.ParticleFlags.ACTIVE)] * vertex_count,
    )
    zeros = np.zeros(triangles.shape[0], dtype=np.float32)
    areas = builder.add_triangles(
        triangles[:, 0],
        triangles[:, 1],
        triangles[:, 2],
        tri_ke=zeros,
        tri_ka=zeros,
        tri_kd=zeros,
        tri_drag=zeros,
        tri_lift=zeros,
    )
    if len(areas) != triangles.shape[0] or np.any(np.asarray(areas) <= 0.0):
        raise RuntimeError("surface reconstruction dropped or inverted a boundary triangle")
    model = builder.finalize(device=device)
    state = model.state()

    edges = _q0_outline_edges(q[0], triangles, camera.position)
    q0 = q[0].astype(np.float32)
    edge_starts = wp.array(q0[edges[:, 0]], dtype=wp.vec3, device=device)
    edge_ends = wp.array(q0[edges[:, 1]], dtype=wp.vec3, device=device)
    edge_colors = wp.full(edges.shape[0], wp.vec3(0.16, 0.86, 1.0), dtype=wp.vec3, device=device)

    videos_dir = output_root / "videos"
    posters_dir = output_root / "posters"
    records_dir = output_root / "records"
    videos_dir.mkdir(parents=True, exist_ok=True)
    posters_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    video_path = videos_dir / f"{record.stem}.mp4"
    poster_path = posters_dir / f"{record.stem}.png"
    record_path = records_dir / f"{record.stem}.json"
    if any(path.exists() for path in (video_path, poster_path, record_path)):
        raise FileExistsError(f"refusing to overwrite an existing render for {record.stem}")

    poster_state_index = max(
        range(q.shape[0]),
        key=lambda index: float(np.sqrt(np.mean(np.square(q[index] - q[0])))),
    )
    poster_written = False
    with Capture(
        out_dir=str(output_root),
        width=width,
        height=height,
        camera_pos=camera.position,
        camera_target=camera.target,
        camera_fov=camera.fov_degrees,
        shading_style="studio",
    ) as capture:
        viewer = capture._get_viewer(model)
        capture._apply_camera(viewer)
        viewer.show_particles = False
        viewer.show_triangles = True
        viewer.renderer.draw_wireframe = False
        viewer.renderer.draw_shadows = True
        viewer.renderer.line_width = 2.0
        with _MP4Writer(video_path, width=width, height=height, fps=fps) as writer:
            for output_frame, stored_state_index in enumerate(schedule):
                state.particle_q.assign(q[stored_state_index].astype(np.float32))
                viewer.begin_frame(float(stored_state_index) * record.dt_seconds)
                viewer.log_state(state)
                viewer.log_lines("/reference/q0-outline", edge_starts, edge_ends, edge_colors)
                viewer.end_frame()
                frame = viewer.get_frame().numpy()
                frame = _overlay_frame(
                    frame,
                    asset_id=record.asset_id,
                    sequence_id=record.sequence_id,
                    state_index=stored_state_index,
                    step_count=record.step_count,
                    dt_seconds=record.dt_seconds,
                )
                writer.write_frame(frame)
                if stored_state_index == poster_state_index and not poster_written:
                    from PIL import Image  # noqa: PLC0415 -- capture-only dependency

                    Image.fromarray(frame).save(poster_path)
                    poster_written = True
                if output_frame == 0 or (output_frame + 1) % 32 == 0:
                    print(f"[{record.stem}] frame {output_frame + 1}/{len(schedule)}", flush=True)

    if not poster_written or not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"rendering {record.stem} did not produce complete media")
    media_record = {
        "asset_id": record.asset_id,
        "camera": camera.as_dict(),
        "contract": _GALLERY_SCHEMA,
        "fps": fps,
        "frame_count": len(schedule),
        "geometry_contract": "exact stored q float32 states only; no interpolation; fixed q0 cyan surface outline",
        "hold_frames_per_stored_state": hold_frames,
        "playback": "forward then reverse ping-pong; deliberately slowed; not real time",
        "poster": {
            "bytes": poster_path.stat().st_size,
            "path": f"posters/{poster_path.name}",
            "sha256": file_sha256(poster_path),
            "stored_state_index": poster_state_index,
        },
        "sequence_id": record.sequence_id,
        "source": {
            "dt_storage": {
                "legacy_singleton_vector": record.dt_legacy_singleton_vector,
                "shape": list(record.dt_storage_shape),
            },
            "evidence_json_sha256": record.evidence_json_sha256,
            "manifest_json_sha256": record.manifest_json_sha256,
            "q_sha256": record.q_sha256,
            "sequence_npz_sha256": record.sequence_npz_sha256,
            "static_npz_sha256": record.static_npz_sha256,
        },
        "stored_state_schedule": list(schedule),
        "video": {
            "bytes": video_path.stat().st_size,
            "path": f"videos/{video_path.name}",
            "sha256": file_sha256(video_path),
        },
    }
    record_path.write_text(json.dumps(media_record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {video_path}", flush=True)
    return record_path


def _copy_exact(source: pathlib.Path, destination: pathlib.Path, expected_sha256: str) -> None:
    _require_file_hash(source, expected_sha256, str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    _require_file_hash(destination, expected_sha256, str(destination))


def _generate_gallery_files(
    *,
    source_root: pathlib.Path,
    output_root: pathlib.Path,
    source_index_sha256: str,
    records: Sequence[SourceSequence],
) -> None:
    provenance_root = output_root / "provenance"
    _copy_exact(source_root / "index.json", provenance_root / "source-index.json", source_index_sha256)
    cards: list[dict[str, object]] = []
    for record in records:
        media_path = output_root / "records" / f"{record.stem}.json"
        media = _load_json(media_path)
        video = media.get("video")
        poster = media.get("poster")
        source = media.get("source")
        if not isinstance(video, dict) or not isinstance(poster, dict) or not isinstance(source, dict):
            raise ValueError(f"{media_path} has incomplete media records")
        expected_dt_storage = {
            "legacy_singleton_vector": record.dt_legacy_singleton_vector,
            "shape": list(record.dt_storage_shape),
        }
        stored_dt_storage = source.get("dt_storage")
        if stored_dt_storage is not None and stored_dt_storage != expected_dt_storage:
            raise ValueError(f"{record.stem} media dt storage record disagrees with its source shard")
        if stored_dt_storage is None:
            source["dt_storage"] = expected_dt_storage
            media_path.write_text(json.dumps(media, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        _require_file_hash(output_root / str(video["path"]), str(video["sha256"]), f"{record.stem} video")
        _require_file_hash(output_root / str(poster["path"]), str(poster["sha256"]), f"{record.stem} poster")

        manifest_copy = provenance_root / record.asset_id / f"{record.sequence_id}.manifest.json"
        evidence_copy = provenance_root / record.asset_id / f"{record.sequence_id}.evidence.json"
        _copy_exact(record.manifest_json, manifest_copy, record.manifest_json_sha256)
        _copy_exact(record.evidence_json, evidence_copy, record.evidence_json_sha256)
        cards.append(
            {
                "asset_id": record.asset_id,
                "deformation_seed": record.deformation_seed,
                "dt_seconds": record.dt_seconds,
                "dt_storage": expected_dt_storage,
                "evidence": {
                    "path": manifest_copy.relative_to(output_root)
                    .as_posix()
                    .replace(".manifest.json", ".evidence.json"),
                    "sha256": record.evidence_json_sha256,
                },
                "identities": {
                    "material_sha256": record.material_sha256,
                    "operator_sha256": record.operator_sha256,
                    "protocol_sha256": record.protocol_sha256,
                    "source_sha256": record.source_sha256,
                    "topology_sha256": record.topology_sha256,
                },
                "manifest": {
                    "path": manifest_copy.relative_to(output_root).as_posix(),
                    "sha256": record.manifest_json_sha256,
                },
                "media": media,
                "q_sha256": record.q_sha256,
                "sequence_id": record.sequence_id,
                "sequence_npz_sha256": record.sequence_npz_sha256,
                "step_count": record.step_count,
                "tet_count": record.tet_count,
                "velocity_seed": record.velocity_seed,
                "vertex_count": record.vertex_count,
            }
        )

    gallery_manifest = {
        "accepted_reference_sequence_count": len(cards),
        "contract": _GALLERY_SCHEMA,
        "physical_duration_seconds": _EXPECTED_STEP_COUNT / 300.0,
        "render_policy": {
            "camera": "one fixed camera fitted to the complete q[0:9] bounds of each sequence",
            "geometry": "exact stored q states only; no interpolation and no solver replay",
            "material": "single solid PBR surface; no displacement colormap",
            "outline": "fixed q0 cyan surface-edge outline",
            "playback": "forward/reverse ping-pong, slowed for inspection, not real time",
        },
        "source_storage": {
            "canonical_dt_scalar_sequence_count": sum(not record.dt_legacy_singleton_vector for record in records),
            "legacy_dt_singleton_vector_sequence_count": sum(record.dt_legacy_singleton_vector for record in records),
        },
        "sequences": cards,
        "source_index": {"path": "provenance/source-index.json", "sha256": source_index_sha256},
        "step_count": _EXPECTED_STEP_COUNT,
        "time_step": _EXPECTED_DT_EXPRESSION,
    }
    manifest_path = output_root / "gallery-manifest.json"
    manifest_path.write_text(
        json.dumps(gallery_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "index.html").write_text(_gallery_html(gallery_manifest), encoding="utf-8")


def _short_hash(value: str) -> str:
    return f"{value[:12]}…{value[-8:]}"


def _gallery_html(manifest: dict[str, object]) -> str:
    sequences = manifest["sequences"]
    assert isinstance(sequences, list)
    card_html: list[str] = []
    for card in sequences:
        assert isinstance(card, dict)
        media = card["media"]
        evidence = card["evidence"]
        source_manifest = card["manifest"]
        assert isinstance(media, dict) and isinstance(evidence, dict) and isinstance(source_manifest, dict)
        video = media["video"]
        poster = media["poster"]
        assert isinstance(video, dict) and isinstance(poster, dict)
        asset_id = html.escape(str(card["asset_id"]))
        sequence_id = html.escape(str(card["sequence_id"]))
        card_html.append(
            f"""
            <article class="card">
              <video autoplay muted loop playsinline controls preload="metadata"
                     poster="{html.escape(str(poster["path"]))}">
                <source src="{html.escape(str(video["path"]))}" type="video/mp4">
              </video>
              <div class="body">
                <div class="title"><strong>{asset_id}</strong><span>{sequence_id}</span></div>
                <div class="stats">{card["vertex_count"]:,} vertices · {card["tet_count"]:,} tets · seeds D{card["deformation_seed"]} / V{card["velocity_seed"]}</div>
                <div class="links">
                  <a href="{html.escape(str(source_manifest["path"]))}">manifest</a>
                  <a href="{html.escape(str(evidence["path"]))}">accepted evidence</a>
                  <a href="{html.escape(str(media["video"]["path"]))}">MP4</a>
                </div>
                <details><summary>exact hashes</summary>
                  <dl>
                    <dt>stored q</dt><dd title="{card["q_sha256"]}">{_short_hash(str(card["q_sha256"]))}</dd>
                    <dt>sequence NPZ</dt><dd title="{card["sequence_npz_sha256"]}">{_short_hash(str(card["sequence_npz_sha256"]))}</dd>
                    <dt>evidence JSON</dt><dd title="{evidence["sha256"]}">{_short_hash(str(evidence["sha256"]))}</dd>
                    <dt>rendered MP4</dt><dd title="{video["sha256"]}">{_short_hash(str(video["sha256"]))}</dd>
                  </dl>
                </details>
              </div>
            </article>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Accepted free-body references · 5 assets x 3 samples</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07101d; --panel:#0f1b2b; --line:#263a51; --text:#eef6ff; --muted:#9bb0c8; --cyan:#3cd9ff; --amber:#ffd27a; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top,#152842 0,#07101d 38rem); color:var(--text); font:15px/1.45 system-ui,sans-serif; }}
    header, main, footer {{ width:min(1480px,calc(100% - 32px)); margin:auto; }}
    header {{ padding:42px 0 24px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(28px,4vw,48px); letter-spacing:-.035em; }}
    .lede {{ max-width:940px; margin:0; color:var(--muted); font-size:17px; }}
    .notice {{ margin-top:18px; padding:13px 16px; border:1px solid #5a4725; border-radius:12px; background:#2b2110cc; color:var(--amber); }}
    .legend {{ display:flex; flex-wrap:wrap; gap:10px 20px; margin-top:16px; color:var(--muted); }}
    .legend b {{ color:var(--text); }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; padding:8px 0 40px; }}
    .card {{ overflow:hidden; border:1px solid var(--line); border-radius:15px; background:linear-gradient(160deg,#132237,#0b1523); box-shadow:0 18px 50px #0005; }}
    video {{ display:block; width:100%; aspect-ratio:16/9; background:#03070d; object-fit:cover; }}
    .body {{ padding:14px 16px 16px; }}
    .title {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; font-size:18px; }}
    .title span,.stats {{ color:var(--muted); }}
    .stats {{ margin-top:5px; font-size:13px; }}
    .links {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:12px; }}
    a {{ color:var(--cyan); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    details {{ margin-top:11px; color:var(--muted); font-size:12px; }}
    summary {{ cursor:pointer; }} dl {{ display:grid; grid-template-columns:auto 1fr; gap:4px 10px; margin:8px 0 0; }} dt {{ color:#7890aa; }} dd {{ margin:0; font-family:ui-monospace,monospace; overflow-wrap:anywhere; }}
    footer {{ padding:0 0 48px; color:var(--muted); }}
    footer code {{ overflow-wrap:anywhere; word-break:break-all; }}
    @media(max-width:980px) {{ .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media(max-width:640px) {{ .grid {{ grid-template-columns:1fr; }} header,main,footer {{ width:min(100% - 20px,1480px); }} }}
  </style>
</head>
<body>
  <header>
    <h1>Accepted reference dynamics</h1>
    <p class="lede">Five assets x three independently sampled initial states. The solid surface is the accepted SolverVBD reference; the cyan outline is the fixed initial state q[0]. There is no displacement colormap.</p>
    <div class="notice"><strong>Slow inspection playback — not real time.</strong> Each clip contains 8 physical steps at dt=1/300 s (0.026667 s total), then plays the exact stored states forward and backward as a loop. No frames are interpolated.</div>
    <div class="legend"><span><b>Camera:</b> fixed from full-sequence bounds</span><span><b>Surface:</b> solid PBR</span><span><b>Boundary:</b> free body, no pins</span><span><b>Evidence:</b> all 15 x 8 steps accepted</span></div>
  </header>
  <main><section class="grid">{"".join(card_html)}</section></main>
  <footer>Source index SHA-256: <code>{manifest["source_index"]["sha256"]}</code> · <a href="gallery-manifest.json">gallery manifest</a> · <a href="provenance/source-index.json">source index</a></footer>
</body>
</html>
"""


def render_all(
    *,
    source_root: pathlib.Path,
    output_root: pathlib.Path,
    width: int,
    height: int,
    fps: int,
    hold_frames: int,
    device: str,
) -> pathlib.Path:
    """Render all accepted shards, one fresh subprocess per MP4, atomically."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing output directory: {output_root}")
    source_index_sha256, records = load_source_sequences(source_root)
    if len(records) != 15:
        raise ValueError(f"the 5x3 gallery requires exactly 15 accepted sequences, got {len(records)}")

    temporary_root = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temporary_root.exists():
        raise FileExistsError(f"temporary output already exists: {temporary_root}")
    temporary_root.mkdir(parents=True)
    for number, record in enumerate(records, start=1):
        print(f"rendering {number:02d}/{len(records)} {record.stem}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.principal_stretch.render_reference_gallery",
                "render-one",
                "--source-root",
                str(source_root),
                "--output-root",
                str(temporary_root),
                "--asset-id",
                record.asset_id,
                "--sequence-id",
                record.sequence_id,
                "--width",
                str(width),
                "--height",
                str(height),
                "--fps",
                str(fps),
                "--hold-frames",
                str(hold_frames),
                "--device",
                device,
            ],
            check=True,
        )
    _generate_gallery_files(
        source_root=source_root,
        output_root=temporary_root,
        source_index_sha256=source_index_sha256,
        records=records,
    )
    os.replace(temporary_root, output_root)
    return output_root / "index.html"


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line renderer."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("render-one", "render-all"):
        child = subparsers.add_parser(command)
        child.add_argument("--source-root", type=pathlib.Path, required=True)
        child.add_argument("--output-root", type=pathlib.Path, required=True)
        child.add_argument("--width", type=_positive_int, default=960)
        child.add_argument("--height", type=_positive_int, default=540)
        child.add_argument("--fps", type=_positive_int, default=_DEFAULT_FPS)
        child.add_argument("--hold-frames", type=_positive_int, default=_DEFAULT_HOLD_FRAMES)
        child.add_argument("--device", default="cuda:0")
    one = subparsers.choices["render-one"]
    one.add_argument("--asset-id", required=True)
    one.add_argument("--sequence-id", required=True)

    args = parser.parse_args(argv)
    if args.command == "render-one":
        render_one(
            source_root=args.source_root,
            output_root=args.output_root,
            asset_id=args.asset_id,
            sequence_id=args.sequence_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
            hold_frames=args.hold_frames,
            device=args.device,
        )
    else:
        index_path = render_all(
            source_root=args.source_root,
            output_root=args.output_root,
            width=args.width,
            height=args.height,
            fps=args.fps,
            hold_frames=args.hold_frames,
            device=args.device,
        )
        print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
