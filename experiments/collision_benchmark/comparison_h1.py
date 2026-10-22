# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay accepted H1 recipes from a shared, standard-elasticity preparation.

The cache is an initial condition, not a reference solution or scoring target.
It includes both state buffers and the original robot commands. Replays rebuild
the model, restore the common preparation before planning, retain the source
geometry guards, and start with a fresh solver. No recorded smoke result is
used as the comparison's initial material state.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import warp as wp

import newton.solvers

from .asset_paths import newton_asset_path

__all__ = ["H1_CASE_IDS", "build_h1", "install_h1_solver"]

H1_CASE_IDS = ("j3_h1_jacket", "r7_h1_sheet", "r7_h1_towel", "r7_h1_gusset_bag")
_RECORDED = Path(__file__).with_name("recipes") / "h1"
_PACKAGE = "experiments.collision_benchmark"
_CONTACT_OPTIONS = {
    "particle_self_contact_method": "penalty_dat",
    "rigid_compliant_alm": False,
    "rigid_contact_hard": False,
}
_PREPARATION_FIELDS = (
    "times",
    "hold_window",
    "duration",
    "time",
    "settling_time",
    "settled_targets",
    "patch_indices",
    "patch_anchor_indices",
    "patch_normals",
    "patch_eigenvalues",
    "patch_plane_spans",
    "grasp",
    "initial",
    "authored_initial",
    "authored_state",
    "free_settle_history",
    "settle_history",
    "free_settle_duration",
    "pre_lift_extraction",
    "planned_path_audit",
    "path_audits",
    "minimum_pair_x_gap",
    "open_start_evidence",
    "recorded_start_audits",
    "near_tip_layer_audits",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _digest_arrays(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _array_slots(scene, *, drivers):
    slots = {"clock": scene.clock}
    for name in ("state0", "state1", "control"):
        for attr, value in vars(getattr(scene, name)).items():
            if isinstance(value, wp.array):
                slots[f"{name}.{attr}"] = value
    if drivers:
        for index, driver in enumerate(scene.drivers):
            for attr, value in vars(driver).items():
                if isinstance(value, wp.array):
                    slots[f"driver{index}.{attr}"] = value
    return slots


def _snapshot(scene, *, drivers):
    return {key: value.numpy().copy() for key, value in _array_slots(scene, drivers=drivers).items()}


def _restore(scene, arrays, *, drivers):
    slots = _array_slots(scene, drivers=drivers)
    if slots.keys() != arrays.keys():
        raise ValueError("Common H1 cache has different state/driver array fields")
    for key, target in slots.items():
        value = arrays[key]
        current = target.numpy()
        if current.shape != value.shape or current.dtype != value.dtype:
            raise ValueError(f"Common H1 array layout changed: {key}")
        target.assign(value)


def _restore_python(value, template):
    if isinstance(template, np.ndarray):
        return np.asarray(value, dtype=template.dtype)
    if isinstance(template, tuple):
        return tuple(_restore_python(v, t) for v, t in zip(value, template, strict=True))
    if isinstance(template, list) and len(value) == len(template):
        return [_restore_python(v, t) for v, t in zip(value, template, strict=True)]
    if isinstance(template, dict):
        return {k: _restore_python(v, template.get(k)) for k, v in value.items()}
    return copy.deepcopy(value)


def _verify(path, expected):
    actual = _sha(path)
    if actual != expected:
        raise ValueError(f"Recorded H1 source changed: {path}: expected {expected}, found {actual}")
    return actual


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _uncaptured_jacket_step(module):
    """Retain the recorded observers while exposing each physical substep."""
    tree = ast.parse(Path(module.__file__).read_text())
    scene_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "H1JacketScene")
    step = copy.deepcopy(
        next(node for node in scene_class.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    )
    if ast.unparse(step.body[0]) != "wp.capture_launch(self.graph)":
        raise ValueError("Saved jacket step no longer has the expected capture-only dispatch")
    step.body[0] = ast.parse("self.simulate()").body[0]
    tree = ast.fix_missing_locations(ast.Module(body=[step], type_ignores=[]))
    scope = {}
    exec(compile(tree, module.__file__, "exec"), module.__dict__, scope)
    module.H1JacketScene.step = scope["step"]


def _recorded_modules(case_id):
    """Load exact recipe versions without replacing shared package modules."""
    folder = _RECORDED / case_id
    namespace = f"{_PACKAGE}._h1_replay_{case_id}"
    package = types.ModuleType(namespace)
    package.__path__ = [str(Path(__file__).parent)]
    sys.modules[namespace] = package
    manifest_path = _RECORDED / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    recipe = manifest["cases"][case_id]
    provenance = {str(manifest_path): _sha(manifest_path)}
    for name, digest in recipe["files"].items():
        path = folder / name
        provenance[str(path)] = _verify(path, digest)
    # The asset pin is portable; verify the original source mesh bytes as well.
    asset_root = newton_asset_path("unitree_h1")
    for name, digest in manifest["robot_asset_files"].items():
        path = asset_root / name
        provenance[str(path)] = _verify(path, digest)
    for name in ("asset_paths", "assets", "cloth_robot_scenes", "h1_gripper", "jacket_asset", "jacket_diagnostics"):
        module = importlib.import_module(f"{_PACKAGE}.{name}")
        sys.modules[f"{namespace}.{name}"] = module
        provenance[str(Path(module.__file__))] = _sha(module.__file__)
    if case_id == "j3_h1_jacket":
        module = _load_module(f"{namespace}.jacket_h1_scene", folder / "scene-source.py")
        _uncaptured_jacket_step(module)
        return module, recipe, provenance
    if case_id in ("r7_h1_sheet", "r7_h1_towel"):
        base = _load_module(f"{namespace}.h1_bag_scenes", folder / "h1_bag_scenes.py")
        module = _load_module(f"{namespace}.h1_open_start_scene", folder / "h1_open_start_scene.py")
        factory = _load_module(f"{namespace}.cloth_h1_scenes", folder / "cloth_h1_scenes.py")
        return module, (base, factory, recipe), provenance
    base = _load_module(f"{namespace}.h1_bag_scenes", folder / "h1_bag_scenes.py")
    # Keep the exact class, but exclude CLI dispatch, mkdir and main() side effects.
    path = folder / "record-scene.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            if isinstance(node, ast.ImportFrom) and node.module == f"{_PACKAGE}.h1_bag_scenes":
                node.module = f"{namespace}.h1_bag_scenes"
            body.append(node)
    module = types.ModuleType(f"{namespace}.prepared_body_rim")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module, (base, None, recipe), provenance


@contextlib.contextmanager
def _clean_preparation_environment():
    keys = [
        key
        for key in os.environ
        if key.startswith("NEWTON_H1_") or key in ("NEWTON_CLOTH_STATE_DIR", "NEWTON_JACKET_STATE_DIR")
    ]
    saved = {key: os.environ.pop(key) for key in keys}
    try:
        yield
    finally:
        os.environ.update(saved)


def _build_standard(case_id, device, cached, scratch):
    module, dependencies, source_hashes = _recorded_modules(case_id)
    kwargs_record = {}

    def standard_solver(model, **kwargs):
        kwargs.update(_CONTACT_OPTIONS)
        kwargs.update(
            particle_elasticity_alm=False,
            particle_elasticity_alm_deviatoric=True,
            particle_elasticity_alm_rho_scale=1.0,
        )
        kwargs_record.update(kwargs)
        result = newton.solvers.SolverVBD(model, **kwargs)
        if result.particle_elasticity_alm:
            raise ValueError("H1 preparation requires the global comparison solver mode to be standard elasticity")
        return result

    preparation = {}
    if case_id == "j3_h1_jacket":
        module.SolverVBD = standard_solver
        base_class = module.H1JacketScene

        class CommonJacket(base_class):
            def _capture(self):
                if cached is not None and not hasattr(self, "grasp"):
                    # The saved constructor launches its preparation graph 60 times.
                    # Replaying copies the shared prepared state, never another solve.
                    slots = _array_slots(self, drivers=False)
                    buffers = {
                        k: wp.array(v, dtype=slots[k].dtype, device=self.model.device)
                        for k, v in cached["prepare"].items()
                    }
                    with wp.ScopedCapture(device=self.model.device) as capture:
                        for key, target in slots.items():
                            wp.copy(target, buffers[key])
                    self.graph = capture.graph
                    self._comparison_preparation_buffers = buffers
                else:
                    super()._capture()

            def _configure_grips(self, q):
                preparation.update(_snapshot(self, drivers=False))
                super()._configure_grips(q)

        scene = CommonJacket(device=device, substeps=10, iterations=100)
        mesh = scene.mesh
        geometry_sha = hashlib.sha256(mesh.vertices.tobytes() + mesh.triangles.tobytes()).hexdigest()
        if geometry_sha != dependencies["native_geometry_sha256"]:
            raise ValueError("Packaged native jacket geometry differs from the recorded recipe")
    elif case_id in ("r7_h1_sheet", "r7_h1_towel"):
        base, factory, manifest = dependencies
        base.SolverVBD = standard_solver
        base_class = module.H1OpenStartScene

        class CommonPlanar(base_class):
            def _simulate(self):
                if cached is not None and getattr(self, "_planning_untrimmed", False):
                    _restore(self, cached["prepare"], drivers=False)
                    return
                super()._simulate()

            def _plan_from_material(self, q):
                preparation.update(_snapshot(self, drivers=False))
                super()._plan_from_material(q)

        name = case_id.removeprefix("r7_h1_")
        grasps = json.loads((_RECORDED / case_id / "grasp-recipe.json").read_text())
        scene = CommonPlanar(
            case_id,
            asset_name=name,
            device=device,
            substeps=10,
            iterations=30,
            grasps=grasps,
            outward_offset_m=manifest["open_offset_m"],
            rest_grip_positions=factory._front_grips(name),
            table_thickness=0.025,
            use_cuda_graph=False,
        )
    else:
        base, _, _ = dependencies
        base.SolverVBD = standard_solver
        module.OUT = scratch
        scene = module.PreparedBodyRimScene(case_id, device=device, substeps=10, iterations=100)
        preparation.update(_snapshot(scene, drivers=False))
    scene.comparison_solver_kwargs = copy.deepcopy(kwargs_record)
    scene.comparison_source_hashes = source_hashes
    return scene, preparation


def _cache_identity(case_id, scene):
    model_arrays = {}
    for key in (
        "particle_q",
        "particle_mass",
        "particle_flags",
        "tri_indices",
        "tri_poses",
        "edge_indices",
        "edge_rest_angle",
        "body_q",
        "body_com",
        "shape_transform",
        "shape_body",
    ):
        array = getattr(scene.model, key, None)
        if isinstance(array, wp.array):
            model_arrays[key] = array.numpy()
    core = Path(newton.__file__).parent
    core_hashes = {
        str(path): _sha(path)
        for relative in (
            "_src/solvers/vbd/solver_vbd.py",
            "_src/solvers/vbd/particle_alm_kernels.py",
            "_src/solvers/vbd/particle_vbd_kernels.py",
            "_src/solvers/vbd/rigid_vbd_kernels.py",
            "_src/solvers/vbd/vbd_coupling_kernels.py",
            "_src/geometry/tri_mesh_collision.py",
            "_src/sim/collide.py",
        )
        for path in (core / relative,)
    }
    return {
        "schema": 1,
        "case_id": case_id,
        "adapter_sha256": _sha(__file__),
        "source_hashes": scene.comparison_source_hashes,
        "newton_source_hashes": core_hashes,
        "solver_kwargs": scene.comparison_solver_kwargs,
        "model_sha256": _digest_arrays(model_arrays),
        "scope": "Shared standard-elasticity task preparation only; never a scoring reference",
    }


def build_h1(case_id, device=None, *, cache_dir=None):
    """Build a saved H1 recipe with exactly shared initial state and commands.

    Args:
        case_id: One of H1_CASE_IDS.
        device: Newton/Warp device for construction and preparation.
        cache_dir: Shared directory for both arms, independent of their output
            directories. Defaults to the user cache under newton/collision-benchmark.
            Set NEWTON_COMPARE_H1_CACHE to choose another shared cache.

    Returns:
        Original scene with comparison_start and comparison_solver_kwargs.
        Call install_h1_solver after construction and before any recorded step.
    """
    if case_id not in H1_CASE_IDS:
        raise ValueError(f"Unknown accepted H1 case: {case_id}")
    user_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    default_cache = user_cache / "newton/collision-benchmark/common-starts"
    root = Path(cache_dir or os.environ.get("NEWTON_COMPARE_H1_CACHE", default_cache))
    folder = root / case_id
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata_path, arrays_path = folder / "common-start.json", folder / "common-start.npz"
        cached, metadata = None, None
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            identity = metadata["identity"]
            if identity["adapter_sha256"] != _sha(__file__):
                raise ValueError("Common H1 cache uses a different adapter. Use a new cache directory.")
            for group in ("source_hashes", "newton_source_hashes"):
                for name, digest in identity[group].items():
                    _verify(name, digest)
            _verify(arrays_path, metadata["npz_sha256"])
            with np.load(arrays_path, allow_pickle=False) as data:
                cached = {
                    stage: {
                        key.removeprefix(stage + "/"): data[key].copy()
                        for key in data.files
                        if key.startswith(stage + "/")
                    }
                    for stage in ("prepare", "start")
                }
        with _clean_preparation_environment(), tempfile.TemporaryDirectory(prefix="prepare-", dir=folder) as tmp:
            scene, preparation = _build_standard(case_id, device, cached, Path(tmp))
        identity = _cache_identity(case_id, scene)
        if metadata is not None:
            if identity != metadata["identity"]:
                raise ValueError(
                    "Common H1 cache source/model/config differs. Use a new cache directory; do not overwrite it."
                )
            for name, value in metadata["python_fields"].items():
                setattr(scene, name, _restore_python(value, getattr(scene, name, None)))
            _restore(scene, cached["start"], drivers=True)
            start = _snapshot(scene, drivers=True)
        else:
            start = _snapshot(scene, drivers=True)
            metadata = {
                "identity": identity,
                "python_fields": {
                    name: _jsonable(getattr(scene, name)) for name in _PREPARATION_FIELDS if hasattr(scene, name)
                },
                "state_sha256": _digest_arrays(start),
                "preparation_sha256": _digest_arrays(preparation),
                "command_sha256": _digest_arrays({k: v for k, v in start.items() if k.startswith("driver")}),
            }
            payload = {f"prepare/{k}": v for k, v in preparation.items()}
            payload.update({f"start/{k}": v for k, v in start.items()})
            temporary = folder / "common-start.npz.tmp"
            with temporary.open("wb") as output:
                np.savez_compressed(output, **payload)
            os.replace(temporary, arrays_path)
            metadata["npz_sha256"] = _sha(arrays_path)
            temporary = folder / "common-start.json.tmp"
            temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            os.replace(temporary, metadata_path)
        if _digest_arrays(start) != metadata["state_sha256"]:
            raise AssertionError("Common H1 start-state replay is not exact")
        scene.graph = None
        scene.comparison_start = {
            "cache": str(folder),
            "replayed": cached is not None,
            "state_sha256": metadata["state_sha256"],
            "command_sha256": metadata["command_sha256"],
            "preparation_sha256": metadata["preparation_sha256"],
            "npz_sha256": metadata["npz_sha256"],
            "scope": identity["scope"],
            "preparation_elasticity_alm": False,
            "history_reset_required": True,
        }
        return scene


def install_h1_solver(scene, *, particle_elasticity_alm):
    """Install either treatment with fresh history at the shared start boundary.

    Captured graphs are discarded. The comparison runner must execute its own
    observed substeps or capture a new graph after installing the solver.
    """
    kwargs = copy.deepcopy(scene.comparison_solver_kwargs)
    kwargs.update(_CONTACT_OPTIONS)
    kwargs["particle_elasticity_alm"] = bool(particle_elasticity_alm)
    kwargs["particle_elasticity_alm_deviatoric"] = True
    kwargs["particle_elasticity_alm_rho_scale"] = 1.0
    scene.solver = newton.solvers.SolverVBD(scene.model, **kwargs)
    if scene.solver.particle_elasticity_alm != bool(particle_elasticity_alm):
        raise ValueError("Global comparison solver mode disagrees with requested H1 treatment")
    scene.solver.reset(scene.state, flags=0)
    scene.graph = None
    scene.comparison_start["history_reset_required"] = False
    scene.comparison_start["installed_particle_elasticity_alm"] = bool(particle_elasticity_alm)
    return scene.solver
