"""Run an Isaac Lab profiling script with recorded or replayed PyTetWild output."""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


CACHE_FORMAT_VERSION = 1


def _manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".manifest.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_parameter(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"type": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        return {"type": "float", "value": float(value), "hex": float(value).hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(f"Unsupported tetrahedralize parameter type: {type(value).__name__}")


def _array_metadata(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def _input_hash(vertices: np.ndarray, faces: np.ndarray, parameters: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, array in (("vertices", vertices), ("faces", faces)):
        contiguous = np.ascontiguousarray(array)
        header = {"name": name, "dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        digest.update(contiguous.tobytes())
    digest.update(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _output_hash(vertices: np.ndarray, tetrahedra: np.ndarray) -> str:
    return _input_hash(vertices, tetrahedra, {})


class FrozenTetrahedralization:
    """Record one PyTetWild call or replay its exact output."""

    def __init__(
        self,
        original: Callable,
        mode: str,
        cache_path: Path,
        *,
        overwrite: bool = False,
        invocation: dict[str, Any] | None = None,
        pytetwild_version: str | None = None,
    ) -> None:
        self.original = original
        self.signature = inspect.signature(original)
        self.mode = mode
        self.cache_path = cache_path.resolve()
        self.manifest_path = _manifest_path(self.cache_path)
        self.overwrite = overwrite
        self.invocation = invocation or {}
        self.pytetwild_version = pytetwild_version
        self.call_count = 0
        self.expected_input_hash: str | None = None
        self.parameters: dict[str, dict[str, Any]] | None = None
        self.output_vertices: np.ndarray | None = None
        self.output_tetrahedra: np.ndarray | None = None

        if mode == "record":
            if (self.cache_path.exists() or self.manifest_path.exists()) and not overwrite:
                raise FileExistsError(
                    f"Frozen tetrahedralization cache already exists: {self.cache_path}. "
                    "Pass --pytetwild-overwrite to replace it."
                )
        elif mode == "replay":
            self._load()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict]:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        vertices = np.asarray(bound.arguments.pop("vertices"))
        faces = np.asarray(bound.arguments.pop("faces"))
        parameters = {name: _json_parameter(value) for name, value in bound.arguments.items()}
        return vertices, faces, parameters

    def _record(
        self,
        input_vertices: np.ndarray,
        input_faces: np.ndarray,
        parameters: dict[str, dict[str, Any]],
        output_vertices: np.ndarray,
        output_tetrahedra: np.ndarray,
        input_hash: str,
    ) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        output_hash = _output_hash(output_vertices, output_tetrahedra)
        np.savez_compressed(
            self.cache_path,
            format_version=np.asarray(CACHE_FORMAT_VERSION, dtype=np.int64),
            input_vertices=np.ascontiguousarray(input_vertices),
            input_faces=np.ascontiguousarray(input_faces),
            output_vertices=np.ascontiguousarray(output_vertices),
            output_tetrahedra=np.ascontiguousarray(output_tetrahedra),
            input_hash=np.asarray(input_hash),
            output_hash=np.asarray(output_hash),
        )
        manifest = {
            "format_version": CACHE_FORMAT_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "mode": "record",
            "pytetwild_version": self.pytetwild_version,
            "tetrahedralize_signature": str(self.signature),
            "parameters": parameters,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "input_vertices": _array_metadata(input_vertices),
            "input_faces": _array_metadata(input_faces),
            "output_vertices": _array_metadata(output_vertices),
            "output_tetrahedra": _array_metadata(output_tetrahedra),
            "cache_npz": str(self.cache_path),
            "cache_npz_sha256": _file_sha256(self.cache_path),
            "invocation": self.invocation,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def _load(self) -> None:
        if not self.cache_path.is_file() or not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Replay requires both {self.cache_path} and {self.manifest_path}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported cache format {manifest.get('format_version')}; expected {CACHE_FORMAT_VERSION}"
            )
        actual_file_hash = _file_sha256(self.cache_path)
        if actual_file_hash != manifest.get("cache_npz_sha256"):
            raise ValueError(
                f"Cache artifact SHA mismatch: expected {manifest.get('cache_npz_sha256')}, got {actual_file_hash}"
            )

        with np.load(self.cache_path, allow_pickle=False) as cache:
            version = int(cache["format_version"])
            if version != CACHE_FORMAT_VERSION:
                raise ValueError(f"NPZ cache format {version} does not match {CACHE_FORMAT_VERSION}")
            input_vertices = cache["input_vertices"]
            input_faces = cache["input_faces"]
            self.output_vertices = cache["output_vertices"].copy()
            self.output_tetrahedra = cache["output_tetrahedra"].copy()
            stored_input_hash = str(cache["input_hash"])
            stored_output_hash = str(cache["output_hash"])

        parameters = manifest["parameters"]
        actual_input_hash = _input_hash(input_vertices, input_faces, parameters)
        actual_output_hash = _output_hash(self.output_vertices, self.output_tetrahedra)
        if actual_input_hash != stored_input_hash or stored_input_hash != manifest.get("input_hash"):
            raise ValueError("Recorded PyTetWild input or parameters failed content-hash validation")
        if actual_output_hash != stored_output_hash or stored_output_hash != manifest.get("output_hash"):
            raise ValueError("Recorded PyTetWild output failed content-hash validation")
        self.expected_input_hash = stored_input_hash
        self.parameters = parameters

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        input_vertices, input_faces, parameters = self._bind(args, kwargs)
        input_hash = _input_hash(input_vertices, input_faces, parameters)
        self.call_count += 1

        if self.mode == "record" and self.expected_input_hash is None:
            output_vertices, output_tetrahedra = self.original(*args, **kwargs)
            self.output_vertices = np.asarray(output_vertices).copy()
            self.output_tetrahedra = np.asarray(output_tetrahedra).copy()
            self.expected_input_hash = input_hash
            self.parameters = parameters
            self._record(
                input_vertices.copy(),
                input_faces.copy(),
                parameters,
                self.output_vertices,
                self.output_tetrahedra,
                input_hash,
            )
            return output_vertices, output_tetrahedra

        if input_hash != self.expected_input_hash:
            raise ValueError(
                "PyTetWild replay input/parameter mismatch: "
                f"expected {self.expected_input_hash}, got {input_hash}"
            )
        assert self.output_vertices is not None and self.output_tetrahedra is not None
        return self.output_vertices.copy(), self.output_tetrahedra.copy()


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytetwild-mode", choices=("record", "replay"), required=True)
    parser.add_argument("--pytetwild-cache", type=Path, required=True)
    parser.add_argument("--pytetwild-overwrite", action="store_true")
    parser.add_argument("target", type=Path)
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    target_args = args.target_args[1:] if args.target_args[:1] == ["--"] else args.target_args
    target = args.target.resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Target script does not exist: {target}")

    import pytetwild

    original = pytetwild.tetrahedralize
    invocation = {"target": str(target), "target_args": target_args}
    controller = FrozenTetrahedralization(
        original,
        args.pytetwild_mode,
        args.pytetwild_cache,
        overwrite=args.pytetwild_overwrite,
        invocation=invocation,
        pytetwild_version=getattr(pytetwild, "__version__", None),
    )
    pytetwild.tetrahedralize = functools.wraps(original)(controller)
    try:
        sys.argv = [str(target), *target_args]
        runpy.run_path(str(target), run_name="__main__")
    finally:
        pytetwild.tetrahedralize = original

    if controller.call_count == 0:
        raise RuntimeError("Target completed without calling pytetwild.tetrahedralize")
    print(
        json.dumps(
            {
                "mode": args.pytetwild_mode,
                "call_count": controller.call_count,
                "input_hash": controller.expected_input_hash,
                "cache": str(controller.cache_path),
                "manifest": str(controller.manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
