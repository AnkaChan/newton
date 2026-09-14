# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exploratory float64 reference calculations; not a Newton implementation.

Run: uv run --no-sync python notes/svd-exploration/numpy_probe.py
Finite differences hold history fixed, as a primal vertex solve must.
"""

import json
from functools import partial
from pathlib import Path

import numpy as np

MU = 10000.0
RHO = 1000.0
RETENTION = MU / (MU + RHO)
K_EFF = MU * RHO / (MU + RHO)


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def stretch_data(F, epsilon=0.0):
    _, singular, vt = np.linalg.svd(F)
    stretch = np.hypot(singular, epsilon)
    if stretch.min() == 0:
        raise ValueError("Unregularized derivative is undefined at rank loss")
    v = vt.T
    S = (v * stretch) @ vt

    def sylvester(B):
        return v @ ((vt @ B @ v) / (stretch[:, None] + stretch[None, :])) @ vt

    return S, sylvester


def tensor_energy(F, history, epsilon=0.0):
    S, _ = stretch_data(F, epsilon)
    return 0.5 * K_EFF * np.sum(S * S) + RETENTION * np.sum(history * S)


def tensor_stress(F, history, epsilon=0.0):
    _, solve = stretch_data(F, epsilon)
    X = solve(history)
    return K_EFF * F + 2.0 * RETENTION * F @ X


def tensor_hessian_action(F, history, D, epsilon=0.0):
    _, solve = stretch_data(F, epsilon)
    X = solve(history)
    dS = solve(F.T @ D + D.T @ F)
    dX = solve(-dS @ X - X @ dS)
    return K_EFF * D + 2.0 * RETENTION * (D @ X + F @ dX)


def tensor_hessian_upper_action(F, history, D, epsilon=0.0):
    _, solve = stretch_data(F, epsilon)
    X = solve(history)
    return K_EFF * D + 2.0 * RETENTION * D @ X


def sorted_stress(F, history):
    u, singular, vt = np.linalg.svd(F)
    return (u * (K_EFF * singular + RETENTION * history)) @ vt


def matrix_stress(F, history):
    return K_EFF * F + RETENTION * history


def cofactor(F):
    return np.stack([np.cross(F[:, 1], F[:, 2]), np.cross(F[:, 2], F[:, 0]), np.cross(F[:, 0], F[:, 1])], axis=1)


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(b)))


def vertex_hessian(action, weight):
    return np.column_stack([action(np.outer(axis, weight)) @ weight for axis in np.eye(3)])


def run():
    rng = np.random.default_rng(420)
    history = MU * np.array([[1.5, 0.2, 0.1], [0.2, 0.5, 0.07], [0.1, 0.07, 0.8]])
    F = rotation(0.37) @ np.array([[1.2, 0.1, -0.05], [0.0, 0.8, 0.13], [0.0, 0.0, 1.1]])
    Q = rotation(1.2)
    delta = 1.0e-6
    energy_gradient = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            D = np.zeros((3, 3))
            D[i, j] = delta
            energy_gradient[i, j] = (tensor_energy(F + D, history) - tensor_energy(F - D, history)) / (2.0 * delta)
    weight = np.array([-0.7, 0.3, 1.1])
    H = vertex_hessian(lambda D: tensor_hessian_action(F, history, D), weight)
    H_fd = vertex_hessian(
        lambda D: (tensor_stress(F + delta * D, history) - tensor_stress(F - delta * D, history)) / (2.0 * delta),
        weight,
    )
    results = {
        "scope": "Float64 constitutive probes, fixed history; no solver integration or scene speedup measurement",
        "mu_pa": MU,
        "rho_pa": RHO,
        "tensor_energy_gradient_relative_error": relative_error(tensor_stress(F, history), energy_gradient),
        "tensor_vertex_hessian_relative_error": relative_error(H, H_fd),
        "tensor_vertex_hessian_symmetry_relative_error": relative_error(H, H.T),
        "objectivity_relative_error": relative_error(tensor_stress(Q @ F, history), Q @ tensor_stress(F, history)),
        "stress_torque_symmetry_relative_error": relative_error(
            tensor_stress(F, history) @ F.T, F @ tensor_stress(F, history).T
        ),
    }
    S, _ = stretch_data(F)
    R = F @ np.linalg.inv(S)
    shortcut_stress = R @ (K_EFF * S + RETENTION * history)
    results["rotate_history_shortcut_stress_relative_error"] = relative_error(
        shortcut_stress, tensor_stress(F, history)
    )
    results["rotate_history_shortcut_torque_error_relative"] = relative_error(
        shortcut_stress @ F.T, F @ shortcut_stress.T
    )
    results["rest_rotation_total_stress_norm_pa"] = {}
    for angle in (0.0, 0.2, 1.0, 2.5):
        R = rotation(angle)
        pressure_stress = -MU * cofactor(R)
        results["rest_rotation_total_stress_norm_pa"][str(angle)] = {
            "old_matrix": float(np.linalg.norm(matrix_stress(R, MU * np.eye(3)) + pressure_stress)),
            "sorted_svd": float(np.linalg.norm(sorted_stress(R, MU * np.ones(3)) + pressure_stress)),
            "stretch_tensor": float(np.linalg.norm(tensor_stress(R, MU * np.eye(3)) + pressure_stress)),
        }
    results["stretch_crossing_stress_jump_pa"] = []
    scalar_history = MU * np.array([1.5, 0.5, 0.4])
    # Match material directions before the crossing: y owns the largest stretch.
    material_history = MU * np.diag([0.5, 1.5, 0.4])
    for step in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8):
        left, right = np.diag([1.0 - step, 1.0 + step, 0.7]), np.diag([1.0 + step, 1.0 - step, 0.7])
        results["stretch_crossing_stress_jump_pa"].append(
            {
                "half_interval": step,
                "sorted_svd": float(
                    np.linalg.norm(sorted_stress(right, scalar_history) - sorted_stress(left, scalar_history))
                ),
                "stretch_tensor": float(
                    np.linalg.norm(tensor_stress(right, material_history) - tensor_stress(left, material_history))
                ),
            }
        )
    results["repeated_stretch_basis_stress_error_relative"] = []
    # At S=I every orthogonal V is a valid singular basis. Transform the *same* history.
    reference = tensor_stress(np.eye(3), history)
    for _ in range(10):
        v, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        X = v @ (0.5 * (v.T @ history @ v)) @ v.T
        results["repeated_stretch_basis_stress_error_relative"].append(
            relative_error(K_EFF * np.eye(3) + 2 * RETENTION * X, reference)
        )
    results["rank_crossing"] = []
    rank_history = MU * np.diag([1.0, 1.0, 0.8])
    for thickness in (-1.0e-2, -1.0e-4, -1.0e-6, 0.0, 1.0e-6, 1.0e-4, 1.0e-2):
        deformation = np.diag([1.0, 1.0, thickness])
        exact = float(tensor_stress(deformation, rank_history)[2, 2]) if thickness != 0.0 else None
        regularized = float(tensor_stress(deformation, rank_history, epsilon=1.0e-3)[2, 2])
        results["rank_crossing"].append(
            {"signed_thickness": thickness, "unregularized_P33_pa": exact, "regularized_P33_pa": regularized}
        )
    results["regularized_equilibrium_stress_error_relative"] = []
    results["regularized_hessian_fd_error_relative"] = []
    for thickness in (1.0, 1.0e-3, 0.0, -1.0e-3, -1.0):
        deformation = np.diag([1.0, 1.1, thickness])
        stretch, _ = stretch_data(deformation, epsilon=1.0e-3)
        results["regularized_equilibrium_stress_error_relative"].append(
            relative_error(tensor_stress(deformation, MU * stretch, epsilon=1.0e-3), MU * deformation)
        )
        D = np.diag([0.0, 0.0, 1.0])
        fd_step = 1.0e-7
        finite_difference = (
            tensor_stress(deformation + fd_step * D, rank_history, 1.0e-3)
            - tensor_stress(deformation - fd_step * D, rank_history, 1.0e-3)
        ) / (2.0 * fd_step)
        results["regularized_hessian_fd_error_relative"].append(
            relative_error(tensor_hessian_action(deformation, rank_history, D, 1.0e-3), finite_difference)
        )
    at_collapse = np.diag([1.0, 1.0, 0.0])
    results["regularized_collapse_H3333_pa"] = float(
        tensor_hessian_action(at_collapse, rank_history, np.diag([0.0, 0.0, 1.0]), 1.0e-3)[2, 2]
    )
    minimum_exact, minimum_upper, minimum_difference = np.inf, np.inf, np.inf
    worst = None
    for _ in range(200):
        deformation = np.eye(3) + 0.3 * rng.normal(size=(3, 3))
        A = rng.normal(size=(3, 3))
        remembered = MU * (A.T @ A + 0.01 * np.eye(3))
        w = rng.normal(size=3)
        hessian = vertex_hessian(partial(tensor_hessian_action, deformation, remembered, epsilon=1.0e-3), w)
        upper = vertex_hessian(partial(tensor_hessian_upper_action, deformation, remembered, epsilon=1.0e-3), w)
        eigenvalue = np.linalg.eigvalsh(hessian).min()
        if eigenvalue < minimum_exact:
            minimum_exact = eigenvalue
            worst = {
                "F": deformation.tolist(),
                "history_pa": remembered.tolist(),
                "weight": w.tolist(),
                "hessian": hessian.tolist(),
            }
        minimum_upper = min(minimum_upper, np.linalg.eigvalsh(upper).min())
        minimum_difference = min(minimum_difference, np.linalg.eigvalsh(upper - hessian).min())
    results["random_vertex_hessian_scan"] = {
        "count": 200,
        "minimum_exact_eigenvalue": float(minimum_exact),
        "minimum_upper_eigenvalue": float(minimum_upper),
        "minimum_upper_minus_exact_eigenvalue": float(minimum_difference),
        "negative_curvature_case": worst,
    }
    assert results["tensor_energy_gradient_relative_error"] < 1.0e-7
    assert results["tensor_vertex_hessian_relative_error"] < 1.0e-7
    assert results["objectivity_relative_error"] < 1.0e-12
    assert max(results["regularized_equilibrium_stress_error_relative"]) < 1.0e-12
    assert max(results["regularized_hessian_fd_error_relative"]) < 1.0e-5
    assert minimum_upper > 0.0 and minimum_difference > -1.0e-7
    return results


if __name__ == "__main__":
    result = run()
    output = Path(__file__).with_name("numpy_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "random_vertex_hessian_scan"}, indent=2))
    print(
        "Hessian scan:",
        {key: value for key, value in result["random_vertex_hessian_scan"].items() if key != "negative_curvature_case"},
    )
