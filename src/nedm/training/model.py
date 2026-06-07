from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nedm.training.model_transformer import ContinuousTransformer, TransformerConfig


class HMMWVDynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        target_dim: int,
        transformer_cfg: dict[str, Any],
        normalization: dict[str, list[float]],
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_dim = target_dim
        self.backbone = ContinuousTransformer(
            TransformerConfig(
                input_dim=state_dim + action_dim,
                block_size=int(transformer_cfg["block_size"]),
                n_layer=int(transformer_cfg["n_layer"]),
                n_head=int(transformer_cfg["n_head"]),
                n_embd=int(transformer_cfg["n_embd"]),
                dropout=float(transformer_cfg["dropout"]),
                bias=bool(transformer_cfg["bias"]),
            )
        )
        hidden_dim = int(transformer_cfg.get("head_hidden_dim", self.backbone.config.n_embd))
        self.head = nn.Sequential(
            nn.Linear(self.backbone.config.n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.register_buffer("state_mean", torch.tensor(normalization["state_mean"], dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(normalization["state_std"], dtype=torch.float32))
        self.register_buffer("action_mean", torch.tensor(normalization["action_mean"], dtype=torch.float32))
        self.register_buffer("action_std", torch.tensor(normalization["action_std"], dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(normalization["target_mean"], dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(normalization["target_std"], dtype=torch.float32))

    def normalize_state(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.state_mean) / self.state_std

    def normalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    def normalize_target(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.target_mean) / self.target_std

    def denormalize_target(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        return normalized_targets * self.target_std + self.target_mean

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.normalize_state(states), self.normalize_action(actions)], dim=-1)
        features = self.backbone(tokens)
        return self.head(features)

    def predict_delta(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.denormalize_target(self.forward(states, actions))

