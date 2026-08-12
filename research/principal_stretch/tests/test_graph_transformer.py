# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the multiresolution principal-stretch graph transformer."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from research.principal_stretch import torch_solver as ts
from research.principal_stretch.graph_transformer import (
    EDGE_FEATURE_DIM,
    GraphTransformerConfig,
    PrincipalStretchGraphTransformer,
    RelationAttentionBlock,
    covariant_observation_frame,
)
from research.principal_stretch.hierarchy import build_hierarchy
from research.principal_stretch.predictor import (
    build_stretch_predictor,
    checkpoint_predictor_config,
    load_stretch_predictor_state,
)
from research.principal_stretch.spd_log import spd_floor, sym_log


def _chain_mesh(n_tets: int, offset=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """A tet chain: consecutive tets share exactly one face."""
    count = n_tets + 3
    u = np.linspace(-1.0, 1.0, count)
    index = np.arange(count)
    rest = np.stack(
        [
            u,
            0.3 * u * u + 0.1 * np.sin(1.7 * index),
            0.15 * u * u * u + 0.1 * np.cos(1.3 * index),
        ],
        axis=1,
    )
    rest += np.asarray(offset)
    tets = np.stack([np.arange(i, i + 4) for i in range(n_tets)]).astype(np.int64)
    for tet in tets:
        Dm = np.stack([rest[tet[1]] - rest[tet[0]], rest[tet[2]] - rest[tet[0]], rest[tet[3]] - rest[tet[0]]], axis=1)
        if np.linalg.det(Dm) < 0.0:
            tet[2], tet[3] = tet[3], tet[2]
    return rest, tets


def _disconnected_chains(n_tets: int) -> tuple[np.ndarray, np.ndarray]:
    rest_a, tets_a = _chain_mesh(n_tets)
    rest_b, tets_b = _chain_mesh(n_tets, offset=(0.0, 4.0, 0.0))
    return np.concatenate([rest_a, rest_b]), np.concatenate([tets_a, tets_b + rest_a.shape[0]])


def _tet_poses(rest: np.ndarray, tets: np.ndarray) -> np.ndarray:
    corners = rest[tets]
    Dm = np.stack(
        [corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]],
        axis=-1,
    )
    return np.linalg.inv(Dm)


def _model_and_state(rest: np.ndarray, tets: np.ndarray, pinned: np.ndarray | None = None):
    poses = _tet_poses(rest, tets)
    if pinned is None:
        pinned = np.array([0, 1, 2], dtype=np.int64)
    state = ts.build_solver(
        rest,
        tets,
        poses,
        pinned,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    config = GraphTransformerConfig(hidden_dim=32, num_heads=4, n_levels=8, cluster_size=2)
    hierarchy = build_hierarchy(tets, rest, n_levels=config.n_levels, target=config.cluster_size)
    model = PrincipalStretchGraphTransformer(hierarchy, tets, rest.shape[0], config)
    return model, state, hierarchy


def _inputs(rest: np.ndarray, tets: np.ndarray):
    x_previous = torch.as_tensor(rest, dtype=torch.float64)
    phase = torch.linspace(0.0, math.pi, rest.shape[0], dtype=torch.float64)
    displacement = torch.stack(
        [0.01 * torch.sin(phase), 0.02 * torch.sin(phase) ** 2, 0.015 * torch.cos(phase)], dim=-1
    )
    displacement[:3] = 0.0
    x_current = x_previous + displacement
    force = torch.zeros_like(x_current)
    force[-1] = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    gravity = torch.tensor([0.4, -9.81, 0.7], dtype=torch.float64)
    mu = torch.linspace(2.0e4, 4.0e4, tets.shape[0])
    lam = torch.linspace(3.0e4, 5.0e4, tets.shape[0])
    pin = torch.as_tensor(np.isin(tets, [0, 1, 2]).any(axis=1), dtype=torch.float32)
    return x_current, x_previous, force, gravity, mu, lam, pin


def _randomize_output(model: PrincipalStretchGraphTransformer):
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        model.output_head[-1].weight.normal_(std=2.0e-3, generator=generator)


def _rotation() -> torch.Tensor:
    axis = torch.tensor([0.2, 0.4, 0.7], dtype=torch.float64)
    axis = axis / axis.norm()
    x, y, z = axis
    K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float64)
    angle = 0.8
    return torch.eye(3, dtype=torch.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


class TestRelationAttention(unittest.TestCase):
    def test_sparse_mask_rows_and_content_dependence(self):
        torch.manual_seed(1)
        block = RelationAttentionBlock(16, 4, EDGE_FEATURE_DIM, 0.0)
        hidden = torch.randn(2, 5, 16)
        adjacency = torch.tensor([[1, 2, -1], [0, 2, 3], [0, 1, 4], [1, 4, -1], [2, 3, -1]], dtype=torch.int64)
        weight = (adjacency >= 0).to(torch.float32)
        edge = torch.randn(2, 5, 3, EDGE_FEATURE_DIM)
        attention = block.attention_weights(hidden, edge, adjacency, weight)

        self.assertLess((attention.sum(dim=2) - 1.0).abs().max().item(), 2.0e-7)
        mask = torch.cat([torch.ones(5, 1, dtype=torch.bool), adjacency >= 0], dim=1)
        self.assertEqual(attention.masked_select(~mask[None, :, :, None]).abs().max().item(), 0.0)

        changed = hidden.clone()
        changed[:, 2] += torch.linspace(-4.0, 4.0, 16)
        attention_changed = block.attention_weights(changed, edge, adjacency, weight)
        self.assertGreater((attention_changed - attention).abs().max().item(), 1.0e-4)


class TestPrincipalStretchGraphTransformer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rest, cls.tets = _chain_mesh(12)

    def test_zero_initialization_is_persistent_and_spd(self):
        model, state, _hierarchy = _model_and_state(self.rest, self.tets)
        inputs = _inputs(self.rest, self.tets)
        target = model(state, *inputs)

        x_current = inputs[0]
        F = torch.einsum("tac,tad->tdc", state.J, x_current[state.tets])
        C = F.transpose(-1, -2) @ F
        self.assertLess((target - target.transpose(-1, -2)).abs().max().item(), 2.0e-7)
        self.assertGreater(torch.linalg.eigvalsh(target).min().item(), 0.0)
        self.assertLess((target.double() @ target.double() - C).abs().max().item(), 2.0e-5)

    def test_full_active_se3_and_decoder_equivariance(self):
        model, state, _hierarchy = _model_and_state(self.rest, self.tets)
        _randomize_output(model)
        model.eval()
        x_current, x_previous, force, gravity, mu, lam, pin = _inputs(self.rest, self.tets)
        target = model(state, x_current, x_previous, force, gravity, mu, lam, pin)

        Q = _rotation()
        translation = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)

        def rotate(vector):
            return vector @ Q.T

        target_transformed = model(
            state,
            rotate(x_current) + translation,
            rotate(x_previous) + translation,
            rotate(force),
            rotate(gravity),
            mu,
            lam,
            pin,
        )
        self.assertLess((target_transformed - target).abs().max().item(), 3.0e-6)

        pinned = x_current[state.pinned]
        decoded = ts.solve(state, target.double(), pinned, x_init=x_current, n_iters=3)
        transformed_pinned = rotate(pinned) + translation
        decoded_transformed = ts.solve(
            state,
            target_transformed.double(),
            transformed_pinned,
            x_init=rotate(x_current) + translation,
            n_iters=3,
        )
        self.assertLess((decoded_transformed - (rotate(decoded) + translation)).abs().max().item(), 2.0e-10)

    def test_batch_matches_loop_and_gradients_reach_attention(self):
        model, state, _hierarchy = _model_and_state(self.rest, self.tets)
        _randomize_output(model)
        model.eval()
        inputs = list(_inputs(self.rest, self.tets))
        batched = [torch.stack([value, value]) if value.dim() > 1 else value for value in inputs]
        # Material and pin arrays are mesh constants, while gravity is shared.
        batched[3] = inputs[3]
        batched[4] = inputs[4]
        batched[5] = inputs[5]
        batched[6] = inputs[6]
        out_batch = model(state, *batched)
        out_loop = torch.stack([model(state, *inputs), model(state, *inputs)])
        self.assertLess((out_batch - out_loop).abs().max().item(), 2.0e-6)

        x_current = inputs[0].clone().requires_grad_(True)
        out = model(state, x_current, *inputs[1:])
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(x_current.grad).all())
        query_grad = model.down_attention[0].query.weight.grad
        self.assertIsNotNone(query_grad)
        self.assertGreater(query_grad.abs().max().item(), 0.0)

    def test_inverted_and_singular_states_have_finite_gradients(self):
        model, state, _hierarchy = _model_and_state(self.rest, self.tets)
        _randomize_output(model)
        model.train()
        inputs = list(_inputs(self.rest, self.tets))

        transforms = (
            torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
            torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)),
        )
        for transform in transforms:
            with self.subTest(transform=transform.diagonal().tolist()):
                model.zero_grad(set_to_none=True)
                x_current = (torch.as_tensor(self.rest, dtype=torch.float64) @ transform.T).requires_grad_(True)
                target = model(state, x_current, *inputs[1:])
                self.assertTrue(torch.isfinite(target).all())
                Q = _rotation()
                translation = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64)
                transformed_target = model(
                    state,
                    x_current.detach() @ Q.T + translation,
                    inputs[1] @ Q.T + translation,
                    inputs[2] @ Q.T,
                    inputs[3] @ Q.T,
                    *inputs[4:],
                )
                self.assertLess((transformed_target - target.detach()).abs().max().item(), 4.0e-6)
                target.square().mean().backward()
                self.assertIsNotNone(x_current.grad)
                self.assertTrue(torch.isfinite(x_current.grad).all())
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_checkpoint_contains_only_learned_state(self):
        model, _state, _hierarchy = _model_and_state(self.rest, self.tets)
        state_dict = model.state_dict()
        topology_names = (
            "tets",
            "corner_force_weight",
            "adjacency_",
            "edge_weight_",
            "volume_",
            "rest_length_",
            "rest_direction_",
            "log_edge_weight_",
            "assign_",
            "child_volume_",
            "pou_index_",
            "pou_weight_",
            "representative_",
        )
        self.assertFalse(any(name.startswith(topology_names) for name in state_dict))

        rebuilt, _state, _hierarchy = _model_and_state(self.rest, self.tets)
        rebuilt.load_state_dict(state_dict)

    def test_v2_state_is_independent_of_actual_hierarchy_depth(self):
        short_rest, short_tets = _chain_mesh(2)
        long_rest, long_tets = _chain_mesh(40)
        short, _state, short_hierarchy = _model_and_state(short_rest, short_tets)
        long, _state, long_hierarchy = _model_and_state(long_rest, long_tets)
        self.assertNotEqual(len(short_hierarchy.levels), len(long_hierarchy.levels))

        short_state = short.state_dict()
        long_state = long.state_dict()
        self.assertEqual(short_state.keys(), long_state.keys())
        self.assertEqual(
            {name: value.shape for name, value in short_state.items()},
            {name: value.shape for name, value in long_state.items()},
        )
        short.load_state_dict(long_state)

    def test_conservative_load_and_multires_far_field(self):
        model, state, hierarchy = _model_and_state(self.rest, self.tets)
        self.assertEqual(hierarchy.levels[-1].vol.shape[0], 1)
        _randomize_output(model)
        model.eval()
        x_current, x_previous, force, gravity, mu, lam, pin = _inputs(self.rest, self.tets)
        random_force = torch.randn_like(force)
        self.assertLess(
            (model.conservative_tet_load(random_force).sum(dim=0) - random_force.sum(dim=0)).abs().max().item(),
            2.0e-6,
        )

        target_zero = model(state, x_current, x_previous, torch.zeros_like(force), gravity, mu, lam, pin)
        target_loaded = model(state, x_current, x_previous, force, gravity, mu, lam, pin)
        self.assertGreater((target_loaded[0] - target_zero[0]).abs().max().item(), 1.0e-9)

    def test_disconnected_components_do_not_exchange_state(self):
        rest, tets = _disconnected_chains(8)
        offset = rest.shape[0] // 2
        pinned = np.array([0, 1, 2, offset, offset + 1, offset + 2], dtype=np.int64)
        model, state, hierarchy = _model_and_state(rest, tets, pinned)
        self.assertEqual(hierarchy.levels[-1].vol.shape[0], 2)
        _randomize_output(model)
        model.eval()
        x_current, x_previous, force, gravity, mu, lam, pin = _inputs(rest, tets)
        zero = torch.zeros_like(force)
        force[:] = 0.0
        force[-1] = torch.tensor([2.0, 1.0, -0.5], dtype=torch.float64)
        target_zero = model(state, x_current, x_previous, zero, gravity, mu, lam, pin)
        target_force = model(state, x_current, x_previous, force, gravity, mu, lam, pin)
        self.assertLess((target_force[:8] - target_zero[:8]).abs().max().item(), 1.0e-7)


class TestCovariantObservationFrame(unittest.TestCase):
    def test_rotation_covariance_and_gradients_on_bad_states(self):
        Q = _rotation()
        matrices = torch.stack(
            [
                torch.eye(3, dtype=torch.float64),
                torch.diag(torch.tensor([1.5, 0.8, 0.3], dtype=torch.float64)),
                torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
                torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)),
            ]
        ).requires_grad_(True)
        C = matrices.transpose(-1, -2) @ matrices
        # Use the same floored Hencky construction as the model.
        H = 0.5 * sym_log(spd_floor(C, lam_min=0.05**2))
        frame = covariant_observation_frame(matrices, H)
        transformed = Q @ matrices.detach()
        transformed_C = transformed.transpose(-1, -2) @ transformed
        transformed_H = 0.5 * sym_log(spd_floor(transformed_C, lam_min=0.05**2))
        transformed_frame = covariant_observation_frame(transformed, transformed_H)
        self.assertLess((transformed_frame - Q @ frame.detach()).abs().max().item(), 2.0e-12)

        frame.square().sum().backward()
        self.assertTrue(torch.isfinite(matrices.grad).all())
        self.assertGreater(matrices.grad.abs().max().item(), 0.0)


class TestGraphCheckpointCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rest, cls.tets = _chain_mesh(12)

    def _predictor(self, version: int):
        return build_stretch_predictor(
            "graph-transformer",
            self.rest,
            self.tets,
            torch.device("cpu"),
            torch.float32,
            residual=True,
            graph_config=GraphTransformerConfig(
                hidden_dim=32,
                num_heads=4,
                n_levels=8,
                cluster_size=2,
                architecture_version=version,
            ),
        )

    def test_unversioned_current_checkpoint_is_v1(self):
        source = self._predictor(1)
        config = source.checkpoint_config()
        del config["graph_transformer"]["architecture_version"]
        checkpoint = {"predictor_config": config, "state_dict": source.model.state_dict()}
        normalized = checkpoint_predictor_config(checkpoint)
        self.assertEqual(normalized["graph_transformer"]["architecture_version"], 1)

        rebuilt = self._predictor(1)
        load_stretch_predictor_state(rebuilt, checkpoint)
        for name, value in source.model.state_dict().items():
            self.assertTrue(torch.equal(value, rebuilt.model.state_dict()[name]), name)

    def test_static_topology_keys_select_v0_and_are_stripped(self):
        source = self._predictor(0)
        config = source.checkpoint_config()
        del config["graph_transformer"]["architecture_version"]
        state_dict = dict(source.model.state_dict())
        state_dict["tets"] = source.model.tets.clone()
        state_dict["adjacency_0"] = source.model.adjacency_0.clone()
        checkpoint = {"predictor_config": config, "state_dict": state_dict}
        normalized = checkpoint_predictor_config(checkpoint)
        self.assertEqual(normalized["graph_transformer"]["architecture_version"], 0)

        rebuilt = self._predictor(0)
        load_stretch_predictor_state(rebuilt, checkpoint)
        checkpoint["state_dict"]["unexpected_learned_key"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            load_stretch_predictor_state(rebuilt, checkpoint)


if __name__ == "__main__":
    unittest.main()
