# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe installed Warp SVD conventions and capture without changing the solver."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

print("[kernels] version: alm_svd_warp_probe_v1")
wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[2] / ".cache" / "svd-warp-probe")


@wp.kernel
def decompose32(a: wp.array[wp.mat33], u: wp.array[wp.mat33], s: wp.array[wp.vec3], v: wp.array[wp.mat33]):
    i = wp.tid()
    ui, si, vi = wp.svd3(a[i])
    u[i] = ui
    s[i] = si
    v[i] = vi


@wp.kernel
def decompose64(a: wp.array[wp.mat33d], u: wp.array[wp.mat33d], s: wp.array[wp.vec3d], v: wp.array[wp.mat33d]):
    i = wp.tid()
    ui, si, vi = wp.svd3(a[i])
    u[i] = ui
    s[i] = si
    v[i] = vi


def rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q[:, -1] *= np.linalg.det(q)
    return q


def cases():
    rng = np.random.default_rng(713)
    left, right = rotation(rng), rotation(rng)
    result = {
        "identity": np.eye(3),
        "rotation": left,
        "reflection": np.diag([-1.0, 1.0, 1.0]),
        "inverted_distinct": left @ np.diag([2.0, 1.0, -0.5]) @ right.T,
        "zero": np.zeros((3, 3)),
        "rank_one": left @ np.diag([1.0, 0.0, 0.0]) @ right.T,
        "rank_two": left @ np.diag([2.0, 1.0, 0.0]) @ right.T,
    }
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        result[f"repeated_plus_{eps:g}"] = left @ np.diag([1 + eps, 1 - eps, 0.5]) @ right.T
        result[f"repeated_minus_{eps:g}"] = left @ np.diag([1 - eps, 1 + eps, 0.5]) @ right.T
        result[f"small_min_{eps:g}"] = left @ np.diag([2.0, 1.0, eps]) @ right.T
    for scale in (1e-8, 1e-6, 1e-4, 1e4, 1e8):
        result[f"scaled_{scale:g}"] = scale * left @ np.diag([2.0, 1.0, 0.5]) @ right.T
    for i in range(1000):
        result[f"random_{i}"] = rng.normal(size=(3, 3))
    return result


def run(device, capture):
    matrices = cases()
    names = list(matrices)
    result = {"warp_version": wp.__version__, "device": str(device), "source": wp.__file__, "precision": {}}
    for name, dtype, mat_type, vec_type, kernel in (
        ("float32", np.float32, wp.mat33, wp.vec3, decompose32),
        ("float64", np.float64, wp.mat33d, wp.vec3d, decompose64),
    ):
        data = np.asarray(list(matrices.values()), dtype=dtype)
        a = wp.array(data, dtype=mat_type, device=device)
        u, s, v = [wp.empty(len(data), dtype=t, device=device) for t in (mat_type, vec_type, mat_type)]
        wp.launch(kernel, dim=len(data), inputs=[a, u, s, v], device=device)
        expected_u, expected_s, expected_v = u.numpy(), s.numpy(), v.numpy()
        captured_bitwise = None
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                wp.launch(kernel, dim=len(data), inputs=[a, u, s, v], device=device)
            u.zero_()
            s.zero_()
            v.zero_()
            for _ in range(3):
                wp.capture_launch(cap.graph)
            captured_bitwise = all(
                np.array_equal(actual.numpy(), expected)
                for actual, expected in ((u, expected_u), (s, expected_s), (v, expected_v))
            )
        uu, ss, vv = expected_u.astype(float), expected_s.astype(float), expected_v.astype(float)
        reconstruction = (uu * ss[:, None, :]) @ vv.transpose(0, 2, 1)
        true_s = np.linalg.svd(data.astype(float), compute_uv=False)
        errors = np.linalg.norm(reconstruction - data, axis=(1, 2)) / np.maximum(
            np.linalg.norm(data, axis=(1, 2)), np.finfo(dtype).tiny
        )
        orth_u = np.linalg.norm(uu.transpose(0, 2, 1) @ uu - np.eye(3), axis=(1, 2))
        orth_v = np.linalg.norm(vv.transpose(0, 2, 1) @ vv - np.eye(3), axis=(1, 2))
        per_case = {}
        for i, label in enumerate(names):
            if label.startswith("random_"):
                continue
            per_case[label] = {
                "sigma": ss[i].tolist(),
                "reference_abs_sigma": true_s[i].tolist(),
                "det_u": float(np.linalg.det(uu[i])),
                "det_v": float(np.linalg.det(vv[i])),
                "reconstruction_relative_error": float(errors[i]),
                "u_orthogonality_error": float(orth_u[i]),
                "v_orthogonality_error": float(orth_v[i]),
            }
        result["precision"][name] = {
            "captured_replay_bitwise_matches_eager": captured_bitwise,
            "random_1000": {
                "max_reconstruction_relative_error": float(np.max(errors[-1000:])),
                "median_reconstruction_relative_error": float(np.median(errors[-1000:])),
                "max_u_orthogonality_error": float(np.max(orth_u[-1000:])),
                "max_v_orthogonality_error": float(np.max(orth_v[-1000:])),
                "det_u_min_max": [float(x) for x in (np.linalg.det(uu[-1000:]).min(), np.linalg.det(uu[-1000:]).max())],
                "det_v_min_max": [float(x) for x in (np.linalg.det(vv[-1000:]).min(), np.linalg.det(vv[-1000:]).max())],
            },
            "cases": per_case,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    wp.init()
    result = run(wp.get_device(args.device), args.capture)
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
