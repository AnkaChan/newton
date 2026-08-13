#!/usr/bin/env python3
"""Run process-isolated ABBA benchmarks for Newton VBD changes.

Execution is opt-in via ``--execute``. The default command performs all static
checks and prints the exact schedule without launching Isaac Lab. Each measured
entry starts a fresh ``uv run --no-sync python`` process, validates the imported
Newton checkout, snapshots GPU/process metadata, and journals its result before
the next process starts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analyze_vbd_abba import parse_runtime_result

LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOT = Path(r"D:\Code\Graphics\newton-working-copies\codex-vbd-baseline")
DEFAULT_CANDIDATE_ROOT = Path(r"D:\Code\Graphics\newton-working-copies\codex-vbd-profile")
DEFAULT_UV = Path(r"C:\Users\ankac\AppData\Local\anaconda3\Scripts\uv.exe")
DEFAULT_FROZEN_VOLUME_CACHE = LAB_ROOT / "profiling" / "validation" / "volume_tet_cache.npz"
DEFAULT_FROZEN_VOLUME_WRAPPER = LAB_ROOT / "profiling" / "run_with_frozen_tetrahedralization.py"
DEFAULT_OVERRIDES = ("presets=newton_mjwarp_vbd_proxy", "env.sim.physics.num_substeps=1")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".json", ".py", ".toml"}
WORKLOADS = {
    "cloth": {
        "task": "Isaac-Lift-Cloth-Franka",
        "config": Path("source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/franka_cloth_env_cfg.py"),
        "contact_capacity": 1024,
    },
    "volume": {
        "task": "Isaac-Lift-Soft-Franka",
        "config": Path("source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/franka_soft_env_cfg.py"),
        "contact_capacity": 256,
    },
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(root: Path) -> dict[str, Any]:
    root = root.resolve()
    package = root / "newton"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"Newton package not found under {root}")
    files = [path for path in package.rglob("*") if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        files.append(pyproject)
    digest = hashlib.sha256()
    byte_count = 0
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
        byte_count += len(contents)
    return {
        "root": str(root),
        "source_hash": digest.hexdigest(),
        "source_file_count": len(files),
        "source_byte_count": byte_count,
        **_git_metadata(root),
    }


def _isaaclab_fingerprint() -> dict[str, Any]:
    roots = (
        LAB_ROOT / "source" / "isaaclab" / "isaaclab" / "benchmark",
        LAB_ROOT / "source" / "isaaclab_newton" / "isaaclab_newton",
        LAB_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "core" / "lift" / "config" / "franka_soft",
    )
    files = [LAB_ROOT / "scripts" / "benchmarks" / "runtime.py"]
    for root in roots:
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    digest = hashlib.sha256()
    byte_count = 0
    for path in sorted(set(files), key=lambda item: item.relative_to(LAB_ROOT).as_posix()):
        relative = path.relative_to(LAB_ROOT).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
        byte_count += len(contents)
    head = _run_capture(["git", "-C", str(LAB_ROOT), "rev-parse", "HEAD"])
    if head.returncode != 0:
        raise RuntimeError(f"failed to inspect IsaacLab Git HEAD under {LAB_ROOT}")
    return {
        "root": str(LAB_ROOT),
        "git_head": head.stdout.strip(),
        "relevant_source_hash": digest.hexdigest(),
        "source_file_count": len(files),
        "source_byte_count": byte_count,
    }


def _run_capture(
    argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _git_metadata(root: Path) -> dict[str, Any]:
    head = _run_capture(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _run_capture(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"])
    diff = _run_capture(["git", "-C", str(root), "diff", "--no-ext-diff", "--binary", "HEAD", "--"])
    if head.returncode != 0 or status.returncode != 0 or diff.returncode != 0:
        raise RuntimeError(f"failed to inspect Git state under {root}")
    return {
        "git_head": head.stdout.strip(),
        "git_dirty_tracked": bool(status.stdout.strip()),
        "git_status_tracked": status.stdout.splitlines(),
        "git_diff_sha256": hashlib.sha256(diff.stdout.encode("utf-8")).hexdigest(),
    }


def _file_evidence(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"evidence file not found: {path}")
    return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256_file(path)}


def _lab_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (LAB_ROOT / path).resolve()


def _task_config_evidence(path: Path, expected_capacity: int) -> dict[str, Any]:
    evidence = _file_evidence(path)
    text = path.read_text(encoding="utf-8")
    iterations = [int(value) for value in re.findall(r"VBDSolverCfg\s*\(\s*iterations\s*=\s*(\d+)", text)]
    capacities = [int(value) for value in re.findall(r"rigid_body_particle_contact_buffer_size\s*=\s*(\d+)", text)]
    substeps = [int(value) for value in re.findall(r"num_substeps\s*=\s*(\d+)", text)]
    if iterations != [10]:
        raise ValueError(f"expected exactly one ten-iteration VBD config in {path}, found {iterations}")
    if capacities != [expected_capacity]:
        raise ValueError(f"expected contact capacity {expected_capacity} in {path}, found {capacities}")
    if substeps != [2]:
        raise ValueError(f"expected source default of two substeps in {path}, found {substeps}")
    return {
        **evidence,
        "extracted_vbd_iterations": iterations[0],
        "extracted_contact_capacity": capacities[0],
        "extracted_default_num_substeps": substeps[0],
        "resolved_num_substeps_override": 1,
    }


def _frozen_cache_evidence(cache_path: Path) -> dict[str, Any]:
    cache = _file_evidence(cache_path)
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    manifest_evidence = _file_evidence(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError(f"unsupported frozen tetrahedralization cache format in {manifest_path}")
    if manifest.get("cache_npz_sha256") != cache["sha256"]:
        raise ValueError(f"frozen tetrahedralization cache SHA does not match {manifest_path}")
    for field in ("input_hash", "output_hash"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(field, ""))):
            raise ValueError(f"invalid {field} in frozen tetrahedralization manifest {manifest_path}")
    return {
        "cache": cache,
        "manifest": manifest_evidence,
        "input_hash": manifest.get("input_hash"),
        "output_hash": manifest.get("output_hash"),
        "output_vertices": manifest.get("output_vertices"),
        "output_tetrahedra": manifest.get("output_tetrahedra"),
        "parameters": manifest.get("parameters"),
    }


def _expected_reported_presets(overrides: list[str]) -> list[str]:
    expected = []
    for override in overrides:
        if override.startswith("presets="):
            expected.extend(item for item in override.split("=", 1)[1].split(",") if item)
        else:
            expected.append(override)
    return expected


def _validate_extra_overrides(overrides: list[str]) -> None:
    if overrides:
        raise ValueError(
            "the fixed implementation-only acceptance harness does not accept extra Hydra overrides; "
            f"received {overrides!r}"
        )


def _validate_isolated_root(root: Path, *, allow_non_isolated: bool) -> None:
    if allow_non_isolated:
        return
    if "newton-working-copies" not in [part.casefold() for part in root.resolve().parts]:
        raise ValueError(
            f"refusing Newton root outside newton-working-copies: {root}; "
            "pass --allow-non-isolated-roots only for an intentional read-only comparison"
        )


def _resolve_uv(value: Path) -> Path:
    if value.is_file():
        return value.resolve()
    discovered = shutil.which(str(value)) or shutil.which("uv")
    if discovered is None:
        raise ValueError(f"uv executable not found: {value}")
    return Path(discovered).resolve()


def _preflight_newton(uv: Path, newton_root: Path, environment: dict[str, str]) -> dict[str, Any]:
    marker = "VBD_PREFLIGHT_JSON="
    code = (
        "import json,newton,pathlib,sys;"
        "print('VBD_PREFLIGHT_JSON='+json.dumps({"
        "'newton_file':str(pathlib.Path(newton.__file__).resolve()),"
        "'newton_version':getattr(newton,'__version__',None),"
        "'python_executable':sys.executable,"
        "'python_version':sys.version}))"
    )
    completed = _run_capture([str(uv), "run", "--no-sync", "python", "-c", code], cwd=LAB_ROOT, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Newton import preflight failed ({completed.returncode})\n{completed.stdout}\n{completed.stderr}"
        )
    lines = [line[len(marker) :] for line in completed.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"Newton preflight emitted {len(lines)} JSON markers\n{completed.stdout}")
    result = json.loads(lines[0])
    imported = Path(result["newton_file"]).resolve()
    try:
        imported.relative_to(newton_root.resolve())
    except ValueError as error:
        raise RuntimeError(f"imported Newton from {imported}, expected it under {newton_root}") from error
    result["preflight_stdout"] = completed.stdout.splitlines()
    result["preflight_stderr"] = completed.stderr.splitlines()
    return result


def _nvidia_smi_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "pstate",
        "temperature.gpu",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.draw",
        "power.limit",
    ]
    gpu = _run_capture([executable, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    apps = _run_capture(
        [
            executable,
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, str]] = []
    if gpu.returncode == 0:
        for row in csv.reader(gpu.stdout.splitlines(), skipinitialspace=True):
            if len(row) == len(fields):
                rows.append(dict(zip(fields, row, strict=True)))
    app_rows: list[dict[str, str]] = []
    if apps.returncode == 0:
        for row in csv.reader(apps.stdout.splitlines(), skipinitialspace=True):
            if len(row) == 3:
                app_rows.append(dict(zip(("pid", "process_name", "used_gpu_memory"), row, strict=True)))
    return {
        "available": gpu.returncode == 0,
        "executable": executable,
        "gpus": rows,
        "compute_processes": app_rows,
        "gpu_query_returncode": gpu.returncode,
        "gpu_query_stderr": gpu.stderr.splitlines(),
        "compute_query_returncode": apps.returncode,
        "compute_query_stderr": apps.stderr.splitlines(),
    }


def _gpu_busy_reasons(snapshot: dict[str, Any], threshold: float) -> list[str]:
    reasons = []
    for gpu in snapshot.get("gpus") or []:
        try:
            utilization = float(gpu["utilization.gpu"])
        except (KeyError, ValueError):
            continue
        if utilization > threshold:
            reasons.append(f"GPU {gpu.get('index')} utilization is {utilization:.1f}% (limit {threshold:.1f}%)")
    # On WDDM, query-compute-apps includes ordinary desktop graphics contexts
    # and reports their memory as N/A. Keep the complete list as provenance but
    # only classify recognizable Python/CUDA processes as competing work.
    competing = []
    for process in snapshot.get("compute_processes") or []:
        name = Path(process.get("process_name", "")).name.casefold()
        memory = process.get("used_gpu_memory", "").strip()
        numeric_memory = bool(re.fullmatch(r"\d+(?:\.\d+)?", memory))
        if name in {"python", "python.exe", "pythonw.exe", "ncu.exe", "nsys.exe"} or numeric_memory:
            competing.append(process)
    if competing:
        reasons.append(f"nvidia-smi reports {len(competing)} competing Python/CUDA process(es)")
    return reasons


def _format_argv(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def _plan_runs(block_count: int) -> list[dict[str, Any]]:
    runs = []
    ordinal = 0
    for block in range(1, block_count + 1):
        for position, (variant, pair_number) in enumerate(
            (("baseline", 1), ("candidate", 1), ("candidate", 2), ("baseline", 2)), start=1
        ):
            ordinal += 1
            runs.append(
                {
                    "ordinal": ordinal,
                    "block": block,
                    "position": position,
                    "pair_id": f"block_{block:02d}_pair_{pair_number}",
                    "variant": variant,
                    "attempt": 0,
                    "status": "pending",
                }
            )
    return runs


def _benchmark_argv(config: dict[str, Any], run_dir: Path) -> list[str]:
    target_args = [
        "--task",
        config["task"],
        "--num_envs",
        str(config["num_envs"]),
        "--num_steps",
        str(config["num_steps"]),
        "--warmup_steps",
        str(config["warmup_steps"]),
        "--seed",
        str(config["seed"]),
        "--device",
        config["device"],
        "--visualizer",
        "none",
        "--benchmark_formatter",
        "schema",
        "--output_path",
        str(run_dir),
        *config["hydra_overrides"],
    ]
    python_args = [config["benchmark_script"], *target_args]
    if config["frozen_volume_cache"] is not None:
        python_args = [
            config["frozen_volume_wrapper"],
            "--pytetwild-mode",
            "replay",
            "--pytetwild-cache",
            config["frozen_volume_cache"],
            config["benchmark_script"],
            "--",
            *target_args,
        ]
    return [config["uv"], "run", "--no-sync", "python", *python_args]


def _process_environment(newton_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(newton_root.resolve())
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["UV_NO_SYNC"] = "1"
    return environment


def _find_result(run_dir: Path) -> Path:
    candidates = []
    for path in run_dir.glob("benchmark_runtime*.json"):
        try:
            parsed = parse_runtime_result(path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if parsed["format"] == "schema":
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one schema runtime result in {run_dir}, found {len(candidates)}")
    return candidates[0]


def _validate_parsed_result(result: dict[str, Any], config: dict[str, Any]) -> None:
    expected = {
        "task": config["task"],
        "num_envs": config["num_envs"],
        "num_steps": config["num_steps"],
        "warmup_steps": config["warmup_steps"],
        "measurement_mode": "host_return",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"result {key}={result.get(key)!r}, expected {value!r}")
    if result.get("seed") is not None and result["seed"] != config["seed"]:
        raise RuntimeError(f"result seed={result['seed']!r}, expected {config['seed']!r}")
    missing = set(config["expected_reported_presets"]).difference(result["presets"])
    if missing:
        raise RuntimeError(f"result metadata is missing preset/overrides {sorted(missing)}")


def _assert_immutable_sources(manifest: dict[str, Any]) -> None:
    for variant, expected in manifest["sources"].items():
        actual = _source_fingerprint(Path(expected["root"]))
        if actual["source_hash"] != expected["source_hash"]:
            raise RuntimeError(f"{variant} Newton source changed during the suite")
    actual_lab = _isaaclab_fingerprint()
    expected_lab = manifest["isaaclab_source"]
    for key in ("git_head", "relevant_source_hash"):
        if actual_lab[key] != expected_lab[key]:
            raise RuntimeError("IsaacLab tracked source changed during the suite")
    for expected in manifest["immutable_files"]:
        actual = _file_evidence(Path(expected["path"]))
        if actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"immutable suite input changed: {expected['path']}")


def _build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    workload = WORKLOADS[args.workload]
    baseline_root = args.baseline_root.resolve()
    candidate_root = args.candidate_root.resolve()
    _validate_isolated_root(baseline_root, allow_non_isolated=args.allow_non_isolated_roots)
    _validate_isolated_root(candidate_root, allow_non_isolated=args.allow_non_isolated_roots)
    if baseline_root == candidate_root:
        raise ValueError("baseline and candidate Newton roots must differ")
    uv = _resolve_uv(args.uv)
    benchmark_script = _lab_path(args.benchmark_script)
    if not benchmark_script.is_file():
        raise ValueError(f"benchmark script not found: {benchmark_script}")
    task_config = (LAB_ROOT / workload["config"]).resolve()
    task_evidence = _task_config_evidence(task_config, workload["contact_capacity"])
    frozen_cache = None
    frozen_wrapper = None
    frozen_cache_evidence = None
    if args.workload == "volume" and not args.allow_unfrozen_volume:
        frozen_cache = _lab_path(args.frozen_volume_cache or DEFAULT_FROZEN_VOLUME_CACHE)
        frozen_wrapper = _lab_path(args.frozen_volume_wrapper)
        if not frozen_wrapper.is_file():
            raise ValueError(f"frozen tetrahedralization wrapper not found: {frozen_wrapper}")
        frozen_cache_evidence = _frozen_cache_evidence(frozen_cache)
    elif args.frozen_volume_cache is not None:
        raise ValueError("--frozen-volume-cache is only valid for a frozen volume workload")
    additional_evidence = [_file_evidence(_lab_path(path)) for path in args.immutable_evidence]
    _validate_extra_overrides(args.hydra_override)
    overrides = [*DEFAULT_OVERRIDES, *args.hydra_override]
    sources = {
        "baseline": _source_fingerprint(baseline_root),
        "candidate": _source_fingerprint(candidate_root),
    }
    if sources["baseline"]["git_dirty_tracked"] and not args.allow_dirty_baseline:
        raise ValueError("baseline Newton checkout has tracked changes; pass --allow-dirty-baseline to override")
    suite_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_id = f"vbd_abba_{args.workload}_{suite_stamp}"
    output_root = args.output_root or (LAB_ROOT / "profiling" / "final_abba" / suite_id)
    immutable_files = [
        task_evidence,
        _file_evidence(benchmark_script),
        _file_evidence(Path(__file__).resolve()),
        _file_evidence(Path(__file__).resolve().with_name("analyze_vbd_abba.py")),
        *additional_evidence,
    ]
    if frozen_cache_evidence is not None:
        immutable_files.extend(
            (
                frozen_cache_evidence["cache"],
                frozen_cache_evidence["manifest"],
                _file_evidence(frozen_wrapper),
            )
        )
    uv_version = _run_capture([str(uv), "--version"])
    config = {
        "workload": args.workload,
        "task": workload["task"],
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "device": args.device,
        "abba_blocks": args.abba_blocks,
        "runs_per_variant": args.abba_blocks * 2,
        "fresh_process_count": args.abba_blocks * 4,
        "cooldown_seconds": args.cooldown_seconds,
        "max_start_gpu_util_pct": args.max_start_gpu_util_pct,
        "busy_gpu_policy": args.busy_gpu_policy,
        "uv": str(uv),
        "benchmark_script": str(benchmark_script),
        "frozen_volume_wrapper": None if frozen_wrapper is None else str(frozen_wrapper),
        "frozen_volume_cache": None if frozen_cache is None else str(frozen_cache),
        "hydra_overrides": overrides,
        "expected_reported_presets": _expected_reported_presets(overrides),
        "benchmark_formatter": "schema",
        "timing_mode": "host_return",
        "vbd_iterations": task_evidence["extracted_vbd_iterations"],
        "contact_capacity": task_evidence["extracted_contact_capacity"],
        "num_substeps": task_evidence["resolved_num_substeps_override"],
        "volume_topology": (
            "frozen-replay"
            if frozen_cache is not None
            else "unfrozen-topology-confounded"
            if args.workload == "volume"
            else "not-applicable"
        ),
    }
    return {
        "schema_version": "vbd-abba-suite-v1",
        "suite_id": suite_id,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "status": "planned",
        "output_root": str(output_root.resolve()),
        "config": config,
        "sources": sources,
        "isaaclab_source": _isaaclab_fingerprint(),
        "task_config_evidence": task_evidence,
        "frozen_volume_evidence": frozen_cache_evidence,
        "additional_immutable_evidence": additional_evidence,
        "immutable_files": immutable_files,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "orchestrator_python": sys.executable,
            "orchestrator_python_version": sys.version,
            "uv_version": uv_version.stdout.strip(),
            "initial_nvidia_smi": _nvidia_smi_snapshot(),
        },
        "runs": _plan_runs(args.abba_blocks),
    }


def _compatible_resume(existing: dict[str, Any], planned: dict[str, Any]) -> None:
    output_root = existing["output_root"]
    if existing["config"] != planned["config"]:
        raise ValueError("resume arguments do not match the existing manifest config")
    for variant in ("baseline", "candidate"):
        if existing["sources"][variant]["source_hash"] != planned["sources"][variant]["source_hash"]:
            raise ValueError(f"cannot resume: {variant} source hash changed")
    for key in ("git_head", "relevant_source_hash"):
        if existing["isaaclab_source"][key] != planned["isaaclab_source"][key]:
            raise ValueError("cannot resume: IsaacLab tracked source changed")
    old_files = {item["path"]: item["sha256"] for item in existing["immutable_files"]}
    new_files = {item["path"]: item["sha256"] for item in planned["immutable_files"]}
    if old_files != new_files:
        raise ValueError("cannot resume: an immutable suite input changed")
    if not Path(output_root).is_dir():
        raise ValueError(f"cannot resume: output root does not exist: {output_root}")


def _run_suite(manifest: dict[str, Any], manifest_path: Path, *, resume: bool) -> None:
    config = manifest["config"]
    output_root = manifest_path.parent
    manifest["status"] = "running"
    manifest["updated_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    for run in manifest["runs"]:
        if run["status"] == "completed":
            continue
        _assert_immutable_sources(manifest)
        if config["cooldown_seconds"] > 0 and any(item.get("status") == "completed" for item in manifest["runs"]):
            print(f"cooldown: {config['cooldown_seconds']:.1f}s", flush=True)
            time.sleep(config["cooldown_seconds"])
        variant = run["variant"]
        source = manifest["sources"][variant]
        newton_root = Path(source["root"])
        environment = _process_environment(newton_root)
        preflight = _preflight_newton(Path(config["uv"]), newton_root, environment)
        gpu_before = _nvidia_smi_snapshot()
        busy_reasons = _gpu_busy_reasons(gpu_before, config["max_start_gpu_util_pct"])
        if busy_reasons and config["busy_gpu_policy"] == "error":
            raise RuntimeError("GPU is not idle: " + "; ".join(busy_reasons))
        if busy_reasons and config["busy_gpu_policy"] == "warn":
            print("WARNING: GPU is not idle: " + "; ".join(busy_reasons), flush=True)
        run["attempt"] = int(run.get("attempt", 0)) + 1
        suffix = "" if run["attempt"] == 1 else f"_attempt_{run['attempt']:02d}"
        run_dir = output_root / "runs" / f"{run['ordinal']:03d}_{variant}{suffix}"
        run_dir.mkdir(parents=True, exist_ok=False)
        argv = _benchmark_argv(config, run_dir)
        log_path = run_dir / "process.log"
        metadata_path = run_dir / "run_metadata.json"
        metadata = {
            "suite_id": manifest["suite_id"],
            "ordinal": run["ordinal"],
            "block": run["block"],
            "position": run["position"],
            "pair_id": run["pair_id"],
            "variant": variant,
            "attempt": run["attempt"],
            "source": source,
            "preflight": preflight,
            "gpu_before": gpu_before,
            "busy_gpu_reasons": busy_reasons,
            "argv": argv,
            "command": _format_argv(argv),
            "cwd": str(LAB_ROOT),
            "environment": {
                key: environment.get(key)
                for key in ("PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONHASHSEED", "UV_NO_SYNC", "CUDA_VISIBLE_DEVICES")
            },
            "started_utc": _utc_now(),
            "status": "running",
        }
        _atomic_json(metadata_path, metadata)
        run.update(
            {
                "status": "running",
                "run_dir": os.path.relpath(run_dir, output_root),
                "source_hash": source["source_hash"],
                "metadata_path": os.path.relpath(metadata_path, output_root),
                "started_utc": metadata["started_utc"],
            }
        )
        manifest["updated_utc"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        print(
            f"[{run['ordinal']:02d}/{len(manifest['runs']):02d}] {variant} "
            f"block={run['block']} position={run['position']} pair={run['pair_id']}",
            flush=True,
        )
        print(_format_argv(argv), flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                argv,
                cwd=LAB_ROOT,
                env=environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            metadata["launcher_pid"] = process.pid
            _atomic_json(metadata_path, metadata)
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                returncode = process.wait(timeout=30)
                raise
        elapsed = time.perf_counter() - started
        gpu_after = _nvidia_smi_snapshot()
        metadata.update(
            {
                "finished_utc": _utc_now(),
                "elapsed_s": elapsed,
                "returncode": returncode,
                "gpu_after": gpu_after,
                "status": "completed" if returncode == 0 else "failed",
            }
        )
        _atomic_json(metadata_path, metadata)
        run.update(
            {
                "finished_utc": metadata["finished_utc"],
                "elapsed_s": elapsed,
                "returncode": returncode,
                "log_path": os.path.relpath(log_path, output_root),
            }
        )
        if returncode != 0:
            run["status"] = "failed"
            manifest["status"] = "failed"
            manifest["updated_utc"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            raise RuntimeError(f"benchmark process failed with exit code {returncode}; see {log_path}")
        _assert_immutable_sources(manifest)
        result_path = _find_result(run_dir)
        parsed = parse_runtime_result(result_path)
        _validate_parsed_result(parsed, config)
        run.update(
            {
                "status": "completed",
                "result_path": os.path.relpath(result_path, output_root),
                "environment_step_fps": parsed["environment_step_fps"],
                "iteration_time_ms": parsed["iteration_time_ms"],
                "reported_config": {
                    key: parsed[key]
                    for key in (
                        "format",
                        "task",
                        "seed",
                        "num_envs",
                        "num_steps",
                        "warmup_steps",
                        "measurement_mode",
                        "presets",
                        "reported_physics_backend",
                        "status",
                    )
                },
            }
        )
        manifest["updated_utc"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        print(f"completed: {parsed['environment_step_fps']:,.3f} FPS ({elapsed:.1f}s process wall time)", flush=True)
    manifest["status"] = "completed"
    manifest["completed_utc"] = _utc_now()
    manifest["updated_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    analyzer = Path(__file__).resolve().with_name("analyze_vbd_abba.py")
    completed = subprocess.run([sys.executable, str(analyzer), str(manifest_path)], cwd=LAB_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"suite completed but analysis failed with exit code {completed.returncode}")


def _print_plan(manifest: dict[str, Any]) -> None:
    config = manifest["config"]
    print(f"suite: {manifest['suite_id']}")
    print(f"output: {manifest['output_root']}")
    print(
        f"workload: {config['task']}, {config['num_envs']} envs, "
        f"{config['warmup_steps']} warmup + {config['num_steps']} measured steps"
    )
    print(
        f"solver invariants: {config['vbd_iterations']} VBD iterations, {config['num_substeps']} substep, "
        f"contact capacity {config['contact_capacity']}"
    )
    if config["workload"] == "volume":
        print(f"volume topology: {config['volume_topology']}")
        if manifest["frozen_volume_evidence"] is not None:
            evidence = manifest["frozen_volume_evidence"]
            print(
                f"    cache={evidence['cache']['path']} ({evidence['cache']['sha256'][:16]}) "
                f"input={evidence['input_hash'][:16]} output={evidence['output_hash'][:16]}"
            )
    print(
        f"schedule: {config['abba_blocks']} ABBA blocks, {config['runs_per_variant']} runs/variant, "
        f"{config['fresh_process_count']} fresh benchmark processes"
    )
    for variant, source in manifest["sources"].items():
        print(
            f"{variant}: {source['root']} @ {source['git_head'][:12]}, "
            f"source {source['source_hash'][:16]}, tracked_dirty={source['git_dirty_tracked']}"
        )
    for run in manifest["runs"]:
        root = Path(manifest["sources"][run["variant"]]["root"])
        preview_dir = Path(manifest["output_root"]) / "runs" / f"{run['ordinal']:03d}_{run['variant']}"
        argv = _benchmark_argv(config, preview_dir)
        print(
            f"{run['ordinal']:02d}: block {run['block']} pos {run['position']} {run['variant']} "
            f"({run['pair_id']}); PYTHONPATH={root}"
        )
        print(f"    {_format_argv(argv)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="cloth")
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--uv", type=Path, default=DEFAULT_UV)
    parser.add_argument("--benchmark-script", type=Path, default=Path("scripts/benchmarks/runtime.py"))
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--abba-blocks", type=int, choices=range(1, 33), default=4)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--max-start-gpu-util-pct", type=float, default=20.0)
    parser.add_argument("--busy-gpu-policy", choices=("error", "warn", "ignore"), default="error")
    parser.add_argument("--hydra-override", action="append", default=[])
    parser.add_argument("--frozen-volume-cache", type=Path)
    parser.add_argument("--frozen-volume-wrapper", type=Path, default=DEFAULT_FROZEN_VOLUME_WRAPPER)
    parser.add_argument("--immutable-evidence", action="append", type=Path, default=[])
    parser.add_argument("--allow-unfrozen-volume", action="store_true")
    parser.add_argument("--allow-dirty-baseline", action="store_true")
    parser.add_argument("--allow-non-isolated-roots", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Launch the long benchmark suite")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.num_steps <= 0 or args.warmup_steps < 0:
        parser.error("num-envs and num-steps must be positive; warmup-steps must be non-negative")
    if args.cooldown_seconds < 0.0:
        parser.error("cooldown-seconds must be non-negative")
    if not 0.0 <= args.max_start_gpu_util_pct <= 100.0:
        parser.error("max-start-gpu-util-pct must be between 0 and 100")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.resume and args.output_root is None:
        parser.error("--resume requires the existing suite's --output-root")
    return args


def main() -> None:
    args = _parse_args()
    planned = _build_manifest(args)
    output_root = Path(planned["output_root"])
    manifest_path = output_root / "manifest.json"
    if not args.execute:
        _print_plan(planned)
        print("dry run only; pass --execute to create the suite and launch processes")
        return
    if manifest_path.exists():
        if not args.resume:
            raise ValueError(f"manifest already exists: {manifest_path}; pass --resume to continue it")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        _compatible_resume(existing, planned)
        manifest = existing
    else:
        if args.resume:
            raise ValueError(f"cannot resume because manifest does not exist: {manifest_path}")
        output_root.mkdir(parents=True, exist_ok=False)
        manifest = planned
        _atomic_json(manifest_path, manifest)
    _print_plan(manifest)
    _run_suite(manifest, manifest_path, resume=args.resume)


if __name__ == "__main__":
    main()
