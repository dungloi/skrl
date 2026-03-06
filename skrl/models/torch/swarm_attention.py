from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from .base import Model
from .deterministic import DeterministicMixin
from .gaussian import GaussianMixin

try:
    import gymnasium

    SpaceType = gymnasium.Space
except Exception:
    SpaceType = Any


def _build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "leaky_relu":
        return nn.LeakyReLU()
    if key == "selu":
        return nn.SELU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _build_layernorm_mlp(input_dim: int, hidden_layers: Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_build_activation(activation))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class _SwarmAttentionBackbone(nn.Module):
    def __init__(
        self,
        *,
        observation_dim: int,
        history_length: int,
        self_observation_dim: int,
        relative_observation_dim: int,
        embed_dim: int,
        num_attention_heads: int,
        attention_dropout: float,
        activation: str,
    ) -> None:
        super().__init__()

        if history_length <= 0:
            raise ValueError(f"history_length must be positive, got {history_length}")
        if observation_dim % history_length != 0:
            raise ValueError("Observation dimension is incompatible with history_length: " f"{observation_dim} % {history_length} != 0")
        if self_observation_dim <= 0 or relative_observation_dim <= 0:
            raise ValueError("self_observation_dim and relative_observation_dim must be positive")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if embed_dim % num_attention_heads != 0:
            raise ValueError("embed_dim must be divisible by num_attention_heads: " f"{embed_dim} % {num_attention_heads} != 0")

        transient_obs_dim = observation_dim // history_length
        relative_total_dim = transient_obs_dim - self_observation_dim
        if relative_total_dim < 0:
            raise ValueError("Transient observation dimension is smaller than self_observation_dim: " f"{transient_obs_dim} < {self_observation_dim}")

        inferred_num_other = relative_total_dim // relative_observation_dim
        if relative_total_dim % relative_observation_dim != 0:
            raise ValueError(
                "Cannot infer number of neighbors from observation shape. "
                f"relative_total_dim={relative_total_dim}, relative_observation_dim={relative_observation_dim}"
            )
        if inferred_num_other <= 0:
            raise ValueError(
                "num_other_agents inferred from observation shape must be positive. "
                f"Got inferred_num_other={inferred_num_other} (relative_total_dim={relative_total_dim}, "
                f"relative_observation_dim={relative_observation_dim})"
            )

        self.history_length = history_length
        self.self_observation_dim = self_observation_dim
        self.relative_observation_dim = relative_observation_dim
        self.transient_observation_dim = transient_obs_dim
        self.num_other_agents = inferred_num_other
        self.embed_dim = embed_dim

        self.ego_embedding = nn.Sequential(
            nn.Linear(self_observation_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            _build_activation(activation),
        )
        self.other_embedding = nn.Sequential(
            nn.Linear(relative_observation_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            _build_activation(activation),
        )

        self.neighbor_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.step_fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            _build_activation(activation),
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, history_length, embed_dim, dtype=torch.float32))
        self.temporal_norm = nn.LayerNorm(embed_dim)

        self.output_dim = 2 * embed_dim

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 2:
            raise ValueError(f"states must be rank-2 [batch, dim], got shape={tuple(states.shape)}")

        batch_size = states.shape[0]
        states = states.reshape(batch_size, self.history_length, self.transient_observation_dim)  # [batch_size, history_length, transient_observation_dim]

        ego = states[:, :, : self.self_observation_dim]  # [batch_size, history_length, self_observation_dim]
        ego_emb = self.ego_embedding(ego)  # [batch_size, history_length, embed_dim]

        other_flat = states[:, :, self.self_observation_dim :]  # [batch_size, history_length, num_other_agents * relative_observation_dim]
        other = other_flat.reshape(batch_size, self.history_length, self.num_other_agents, self.relative_observation_dim)  # [batch_size, history_length, num_other_agents, relative_observation_dim]
        other_emb = self.other_embedding(other)  # [batch_size, history_length, num_other_agents, embed_dim]

        query = ego_emb.reshape(batch_size * self.history_length, 1, self.embed_dim)  # [batch_size * history_length, 1, embed_dim]
        key_value = other_emb.reshape(batch_size * self.history_length, self.num_other_agents, self.embed_dim)  # [batch_size * history_length, num_other_agents, embed_dim]
        attention_output, _ = self.neighbor_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )
        neighbor_context = attention_output.squeeze(1)  # [batch_size * history_length, embed_dim]
        fused = torch.cat([ego_emb.reshape(batch_size * self.history_length, self.embed_dim), neighbor_context], dim=-1)  # [batch_size * history_length, 2 * embed_dim]

        step_tokens = self.step_fusion(fused).reshape(batch_size, self.history_length, self.embed_dim)  # [batch_size, history_length, embed_dim]
        step_tokens = step_tokens + self.temporal_pos_embedding  # [batch_size, history_length, embed_dim]

        temporal_output, _ = self.temporal_attention(
            query=step_tokens,
            key=step_tokens,
            value=step_tokens,
            need_weights=False,
        )
        step_tokens = self.temporal_norm(step_tokens + temporal_output)  # [batch_size, history_length, embed_dim]

        mean_token = step_tokens.mean(dim=1)  # [batch_size, embed_dim]
        last_token = step_tokens[:, -1]  # [batch_size, embed_dim]
        return torch.cat([mean_token, last_token], dim=-1)  # [batch_size, 2 * embed_dim]


class _SwarmAttentionGaussianModel(GaussianMixin, Model):
    def __init__(
        self,
        *,
        history_length: int,
        self_observation_dim: int,
        relative_observation_dim: int,
        observation_space: Optional[Union[int, tuple[int], SpaceType]] = None,
        action_space: Optional[Union[int, tuple[int], SpaceType]] = None,
        device: Optional[Union[str, torch.device]] = None,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        initial_log_std: float = 0.0,
        fixed_log_std: bool = False,
        embed_dim: int = 64,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.0,
        mlp_hidden_layers: Sequence[int] = (128, 64, 32),
        activation: str = "elu",
        role: str = "",
        **kwargs,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role=role,
        )

        self.encoder = _SwarmAttentionBackbone(
            observation_dim=self.num_observations,
            history_length=history_length,
            self_observation_dim=self_observation_dim,
            relative_observation_dim=relative_observation_dim,
            embed_dim=embed_dim,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            activation=activation,
        )
        self.action_head = _build_layernorm_mlp(
            input_dim=self.encoder.output_dim,
            hidden_layers=mlp_hidden_layers,
            output_dim=self.num_actions,
            activation=activation,
        )
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(initial_log_std), dtype=torch.float32),
            requires_grad=not fixed_log_std,
        )

    def compute(self, inputs: dict[str, Any], role: str = ""):
        features = self.encoder(inputs["states"])
        mean_actions = self.action_head(features)
        return mean_actions, self.log_std_parameter, {}


class _SwarmAttentionDeterministicModel(DeterministicMixin, Model):
    def __init__(
        self,
        *,
        history_length: int,
        self_observation_dim: int,
        relative_observation_dim: int,
        observation_space: Optional[Union[int, tuple[int], SpaceType]] = None,
        action_space: Optional[Union[int, tuple[int], SpaceType]] = None,
        device: Optional[Union[str, torch.device]] = None,
        clip_actions: bool = False,
        embed_dim: int = 64,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.0,
        mlp_hidden_layers: Sequence[int] = (128, 64, 32),
        activation: str = "elu",
        role: str = "",
        **kwargs,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)

        self.encoder = _SwarmAttentionBackbone(
            observation_dim=self.num_observations,
            history_length=history_length,
            self_observation_dim=self_observation_dim,
            relative_observation_dim=relative_observation_dim,
            embed_dim=embed_dim,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            activation=activation,
        )
        self.value_head = _build_layernorm_mlp(
            input_dim=self.encoder.output_dim,
            hidden_layers=mlp_hidden_layers,
            output_dim=1,
            activation=activation,
        )

    def compute(self, inputs: dict[str, Any], role: str = ""):
        features = self.encoder(inputs["states"])
        values = self.value_head(features)
        return values, {}


def swarm_attention_gaussian_model(
    observation_space: Optional[Union[int, tuple[int], SpaceType]] = None,
    action_space: Optional[Union[int, tuple[int], SpaceType]] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_source: bool = False,
    *args,
    **kwargs,
):
    if return_source:
        return inspect.getsource(_SwarmAttentionGaussianModel)
    return _SwarmAttentionGaussianModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **kwargs,
    )


def swarm_attention_deterministic_model(
    observation_space: Optional[Union[int, tuple[int], SpaceType]] = None,
    action_space: Optional[Union[int, tuple[int], SpaceType]] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_source: bool = False,
    *args,
    **kwargs,
):
    if return_source:
        return inspect.getsource(_SwarmAttentionDeterministicModel)
    return _SwarmAttentionDeterministicModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **kwargs,
    )
