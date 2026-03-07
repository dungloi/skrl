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
        attention_chunk_size: int,
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
        if attention_chunk_size <= 0:
            raise ValueError(f"attention_chunk_size must be positive, got {attention_chunk_size}")

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
        self.neighbor_attention_chunk_size = attention_chunk_size
        self.ego_stack_dim = history_length * self_observation_dim
        self.other_stack_dim = history_length * relative_observation_dim

        self.ego_embedding = nn.Sequential(
            nn.Linear(self.ego_stack_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            _build_activation(activation),
        )
        self.other_embedding = nn.Sequential(
            nn.Linear(self.other_stack_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            _build_activation(activation),
        )

        self.neighbor_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.output_dim = 2 * embed_dim

    @staticmethod
    def _run_attention_in_chunks(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        batch_tokens = query.shape[0]
        if batch_tokens <= chunk_size:
            output, _ = attention(query=query, key=key, value=value, need_weights=False)
            return output

        outputs: list[torch.Tensor] = []
        for start in range(0, batch_tokens, chunk_size):
            end = min(start + chunk_size, batch_tokens)
            output_chunk, _ = attention(
                query=query[start:end],
                key=key[start:end],
                value=value[start:end],
                need_weights=False,
            )
            outputs.append(output_chunk)
        return torch.cat(outputs, dim=0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 2:
            raise ValueError(f"states must be rank-2 [batch, dim], got shape={tuple(states.shape)}")

        batch_size = states.shape[0]
        states = states.reshape(batch_size, self.history_length, self.transient_observation_dim)  # [batch_size, history_length, transient_observation_dim]

        ego = states[:, :, : self.self_observation_dim]  # [batch_size, history_length, self_observation_dim]
        ego_stacked = ego.reshape(batch_size, self.ego_stack_dim)  # [batch_size, history_length * self_observation_dim]
        ego_emb = self.ego_embedding(ego_stacked)  # [batch_size, embed_dim]

        other_flat = states[:, :, self.self_observation_dim :]  # [batch_size, history_length, num_other_agents * relative_observation_dim]
        other = other_flat.reshape(
            batch_size,
            self.history_length,
            self.num_other_agents,
            self.relative_observation_dim,
        )  # [batch_size, history_length, num_other_agents, relative_observation_dim]
        other_stacked = other.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_other_agents,
            self.other_stack_dim,
        )  # [batch_size, num_other_agents, history_length * relative_observation_dim]
        other_emb = self.other_embedding(other_stacked)  # [batch_size, num_other_agents, embed_dim]

        query = ego_emb.unsqueeze(1)  # [batch_size, 1, embed_dim]
        key_value = other_emb  # [batch_size, num_other_agents, embed_dim]
        attention_output = self._run_attention_in_chunks(
            self.neighbor_attention,
            query=query,
            key=key_value,
            value=key_value,
            chunk_size=self.neighbor_attention_chunk_size,
        )
        neighbor_context = attention_output.squeeze(1)  # [batch_size, embed_dim]
        return torch.cat([ego_emb, neighbor_context], dim=-1)  # [batch_size, 2 * embed_dim]


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
        attention_chunk_size: int = 8192,
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
            attention_chunk_size=attention_chunk_size,
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
        attention_chunk_size: int = 8192,
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
            attention_chunk_size=attention_chunk_size,
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
