"""Execution-level invariants for the recurrent Torch PPO implementation.

These tests deliberately use an actual ``torch.nn.GRU`` rather than a mocked
recurrence.  The largest case mirrors the deployed rollout geometry
(``T=80``, ``sequence_length=40``, seven environments) while keeping the
network tiny enough for the CPU test suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import gymnasium
import pytest

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.models.torch.custom_models import CNNGRUAttentionMLPPolicy, CNNGRUAttentionMLPValue
from skrl.resources.preprocessors.torch import RunningStandardScaler, SelectiveRunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.spaces.torch import flatten_tensorized_space


class _RealGRUCore:
    """Small recurrent core implementing skrl's flattened-sequence contract."""

    def _build_gru_core(
        self,
        *,
        input_key: str,
        input_size: int,
        output_size: int,
        sequence_length: int,
        num_envs: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        self.input_key = input_key
        self.sequence_length = sequence_length
        self.spec_num_envs = num_envs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, output_size)

    def get_specification(self) -> dict[str, Any]:
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.spec_num_envs, self.hidden_size)],
            }
        }

    def _forward_gru(self, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        x = inputs[self.input_key]
        hidden = inputs["rnn"][0]

        if not self.training:
            output, hidden = self.gru(x.unsqueeze(1), hidden)
            return self.head(output.squeeze(1)), hidden

        if x.shape[0] % self.sequence_length:
            raise ValueError(
                f"batch size {x.shape[0]} is not divisible by sequence length {self.sequence_length}"
            )
        sequence_count = x.shape[0] // self.sequence_length
        x = x.view(sequence_count, self.sequence_length, x.shape[-1])
        hidden = hidden.view(
            self.num_layers, sequence_count, self.sequence_length, self.hidden_size
        )[:, :, 0, :].contiguous()

        terminated = inputs.get("terminated")
        truncated = inputs.get("truncated")
        if terminated is None and truncated is None:
            done = torch.zeros(
                (sequence_count, self.sequence_length), dtype=torch.bool, device=x.device
            )
        else:
            if terminated is None:
                terminated = torch.zeros_like(truncated)
            if truncated is None:
                truncated = torch.zeros_like(terminated)
            done = (terminated | truncated).view(sequence_count, self.sequence_length)

        outputs = []
        for step in range(self.sequence_length):
            output, hidden = self.gru(x[:, step : step + 1], hidden)
            outputs.append(output)
            # A done transition produces the current output, then clears the
            # state consumed by the following transition.
            hidden = hidden * (~done[:, step]).view(1, -1, 1)
        output = torch.cat(outputs, dim=1).flatten(0, 1)
        return self.head(output), hidden


class RealGRUPolicy(_RealGRUCore, GaussianMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        state_space: gymnasium.Space,
        action_space: gymnasium.Space,
        *,
        sequence_length: int,
        num_envs: int,
        hidden_size: int = 5,
        num_layers: int = 1,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        GaussianMixin.__init__(self, reduction="sum")
        self._build_gru_core(
            input_key="observations",
            input_size=observation_space.shape[0],
            output_size=action_space.shape[0],
            sequence_length=sequence_length,
            num_envs=num_envs,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.log_std_parameter = torch.nn.Parameter(torch.full(action_space.shape, -0.4))

    def compute(self, inputs: dict[str, Any], role: str = ""):
        mean, hidden = self._forward_gru(inputs)
        return mean, {"log_std": self.log_std_parameter, "rnn": [hidden]}


class RealGRUValue(_RealGRUCore, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        state_space: gymnasium.Space,
        action_space: gymnasium.Space,
        *,
        sequence_length: int,
        num_envs: int,
        hidden_size: int = 7,
        num_layers: int = 1,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self._build_gru_core(
            input_key="states",
            input_size=state_space.shape[0],
            output_size=1,
            sequence_length=sequence_length,
            num_envs=num_envs,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def compute(self, inputs: dict[str, Any], role: str = ""):
        value, hidden = self._forward_gru(inputs)
        return value, {"rnn": [hidden]}


class _RealLSTMCore:
    """Two-state recurrent core covering skrl's LSTM hidden/cell contract."""

    def _build_lstm_core(
        self,
        *,
        input_key: str,
        input_size: int,
        output_size: int,
        sequence_length: int,
        num_envs: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        self.input_key = input_key
        self.sequence_length = sequence_length
        self.spec_num_envs = num_envs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, output_size)

    def get_specification(self) -> dict[str, Any]:
        state_size = (self.num_layers, self.spec_num_envs, self.hidden_size)
        return {"rnn": {"sequence_length": self.sequence_length, "sizes": [state_size, state_size]}}

    def _forward_lstm(self, inputs: dict[str, Any]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = inputs[self.input_key]
        hidden, cell = inputs["rnn"]
        if not self.training:
            output, (hidden, cell) = self.lstm(x.unsqueeze(1), (hidden, cell))
            return self.head(output.squeeze(1)), (hidden, cell)

        if x.shape[0] % self.sequence_length:
            raise ValueError(
                f"batch size {x.shape[0]} is not divisible by sequence length {self.sequence_length}"
            )
        sequence_count = x.shape[0] // self.sequence_length
        x = x.view(sequence_count, self.sequence_length, x.shape[-1])

        def initial_chunk_state(state: torch.Tensor) -> torch.Tensor:
            return state.view(
                self.num_layers, sequence_count, self.sequence_length, self.hidden_size
            )[:, :, 0, :].contiguous()

        hidden, cell = initial_chunk_state(hidden), initial_chunk_state(cell)
        terminated = inputs.get("terminated")
        truncated = inputs.get("truncated")
        if terminated is None and truncated is None:
            done = torch.zeros(
                (sequence_count, self.sequence_length), dtype=torch.bool, device=x.device
            )
        else:
            if terminated is None:
                terminated = torch.zeros_like(truncated)
            if truncated is None:
                truncated = torch.zeros_like(terminated)
            done = (terminated | truncated).view(sequence_count, self.sequence_length)

        outputs = []
        for step in range(self.sequence_length):
            output, (hidden, cell) = self.lstm(x[:, step : step + 1], (hidden, cell))
            outputs.append(output)
            keep = (~done[:, step]).view(1, -1, 1)
            hidden, cell = hidden * keep, cell * keep
        output = torch.cat(outputs, dim=1).flatten(0, 1)
        return self.head(output), (hidden, cell)


class RealLSTMPolicy(_RealLSTMCore, GaussianMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        state_space: gymnasium.Space,
        action_space: gymnasium.Space,
        *,
        sequence_length: int,
        num_envs: int,
        hidden_size: int = 4,
        num_layers: int = 2,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        GaussianMixin.__init__(self, reduction="sum")
        self._build_lstm_core(
            input_key="observations",
            input_size=observation_space.shape[0],
            output_size=action_space.shape[0],
            sequence_length=sequence_length,
            num_envs=num_envs,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.log_std_parameter = torch.nn.Parameter(torch.full(action_space.shape, -0.4))

    def compute(self, inputs: dict[str, Any], role: str = ""):
        mean, states = self._forward_lstm(inputs)
        return mean, {"log_std": self.log_std_parameter, "rnn": list(states)}


class RealLSTMValue(_RealLSTMCore, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        state_space: gymnasium.Space,
        action_space: gymnasium.Space,
        *,
        sequence_length: int,
        num_envs: int,
        hidden_size: int = 6,
        num_layers: int = 2,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self._build_lstm_core(
            input_key="states",
            input_size=state_space.shape[0],
            output_size=1,
            sequence_length=sequence_length,
            num_envs=num_envs,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def compute(self, inputs: dict[str, Any], role: str = ""):
        value, states = self._forward_lstm(inputs)
        return value, {"rnn": list(states)}


class SharedRealGRUActorCritic(_RealGRUCore, GaussianMixin, DeterministicMixin, Model):
    """A role-dispatched recurrent model with one genuinely shared GRU trunk."""

    def __init__(
        self,
        observation_space: gymnasium.Space,
        state_space: gymnasium.Space,
        action_space: gymnasium.Space,
        *,
        sequence_length: int,
        num_envs: int,
        hidden_size: int = 5,
        num_layers: int = 2,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        GaussianMixin.__init__(self, reduction="sum", role="policy")
        DeterministicMixin.__init__(self, role="value")
        # Both roles consume observations, so a single recurrent state has an
        # unambiguous meaning during policy and value evaluation.
        self._build_gru_core(
            input_key="observations",
            input_size=observation_space.shape[0],
            output_size=hidden_size,
            sequence_length=sequence_length,
            num_envs=num_envs,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.policy_head = torch.nn.Linear(hidden_size, action_space.shape[0])
        self.value_head = torch.nn.Linear(hidden_size, 1)
        self.log_std_parameter = torch.nn.Parameter(torch.full(action_space.shape, -0.4))

    def act(self, inputs: dict[str, Any], *, role: str = ""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        raise ValueError(f"Unsupported role: {role}")

    def compute(self, inputs: dict[str, Any], role: str = ""):
        features, hidden = self._forward_gru(inputs)
        if role == "policy":
            return self.policy_head(features), {"log_std": self.log_std_parameter, "rnn": [hidden]}
        if role == "value":
            return self.value_head(features), {"rnn": [hidden]}
        raise ValueError(f"Unsupported role: {role}")


def _agent_cfg(
    *,
    rollouts: int,
    learning_epochs: int = 1,
    mini_batches: int = 1,
    learning_rate: float = 0.0,
    kl_threshold: float = 0.0,
    random_timesteps: int = 0,
) -> dict[str, Any]:
    return {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "learning_rate": learning_rate,
        "observation_preprocessor": None,
        "state_preprocessor": None,
        "value_preprocessor": None,
        "random_timesteps": random_timesteps,
        "learning_starts": 0,
        "grad_norm_clip": 0.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "entropy_loss_scale": 0.0,
        "value_loss_scale": 0.5,
        "kl_threshold": kl_threshold,
        "mixed_precision": False,
        "experiment": {
            "directory": "",
            "experiment_name": "",
            "write_interval": 0,
            "checkpoint_interval": 0,
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        },
    }


def _make_agent_from_models(
    policy: Model,
    value: Model,
    *,
    rollouts: int,
    num_envs: int,
    learning_epochs: int = 1,
    mini_batches: int = 1,
    learning_rate: float = 0.0,
) -> PPO_RNN:
    agent = PPO_RNN(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=rollouts, num_envs=num_envs, device="cpu"),
        observation_space=policy.observation_space,
        state_space=policy.state_space,
        action_space=policy.action_space,
        device="cpu",
        cfg=_agent_cfg(
            rollouts=rollouts,
            learning_epochs=learning_epochs,
            mini_batches=mini_batches,
            learning_rate=learning_rate,
        ),
    )
    agent.init()
    return agent


def _make_agent(
    *,
    rollouts: int,
    num_envs: int,
    policy_sequence_length: int,
    value_sequence_length: int | None = None,
    policy_spec_num_envs: int | None = None,
    value_spec_num_envs: int | None = None,
    policy_hidden_size: int = 5,
    value_hidden_size: int = 7,
    policy_num_layers: int = 1,
    value_num_layers: int = 1,
    learning_epochs: int = 1,
    mini_batches: int = 1,
    learning_rate: float = 0.0,
    kl_threshold: float = 0.0,
    random_timesteps: int = 0,
    initialize: bool = True,
) -> PPO_RNN:
    torch.manual_seed(1234)
    observation_space = gymnasium.spaces.Box(low=-10_000, high=10_000, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-10_000, high=10_000, shape=(5,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,))
    policy = RealGRUPolicy(
        observation_space,
        state_space,
        action_space,
        sequence_length=policy_sequence_length,
        num_envs=policy_spec_num_envs or num_envs,
        hidden_size=policy_hidden_size,
        num_layers=policy_num_layers,
    )
    value = RealGRUValue(
        observation_space,
        state_space,
        action_space,
        sequence_length=value_sequence_length or policy_sequence_length,
        num_envs=value_spec_num_envs or num_envs,
        hidden_size=value_hidden_size,
        num_layers=value_num_layers,
    )
    agent = PPO_RNN(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=rollouts, num_envs=num_envs, device="cpu"),
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=_agent_cfg(
            rollouts=rollouts,
            learning_epochs=learning_epochs,
            mini_batches=mini_batches,
            learning_rate=learning_rate,
            kl_threshold=kl_threshold,
            random_timesteps=random_timesteps,
        ),
    )
    if initialize:
        agent.init()
    return agent


def _transition_tensors(timestep: int, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    env = torch.arange(num_envs, dtype=torch.float32)
    sample_id = timestep * num_envs + env
    observations = torch.stack(
        (sample_id, env / 10, torch.sin(sample_id / 7), torch.cos(sample_id / 11)), dim=1
    )
    states = torch.stack(
        (sample_id, env / 13, torch.sin(sample_id / 5), torch.cos(sample_id / 9), (sample_id % 3) / 3), dim=1
    )
    return observations, states


DoneFactory = Callable[[int, int], tuple[torch.Tensor, torch.Tensor]]


def _no_done(timestep: int, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    del timestep
    shape = (num_envs, 1)
    return torch.zeros(shape, dtype=torch.bool), torch.zeros(shape, dtype=torch.bool)


def _representative_async_done(timestep: int, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    terminated = torch.zeros((num_envs, 1), dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    terminated_events = {(5, 0), (39, 2), (40, 5), (78, 6)}
    truncated_events = {(0, 1), (17, 3), (39, 4), (55, 0)}
    for event_timestep, env in terminated_events:
        if timestep == event_timestep and env < num_envs:
            terminated[env] = True
    for event_timestep, env in truncated_events:
        if timestep == event_timestep and env < num_envs:
            truncated[env] = True
    return terminated, truncated


def _collect(agent: PPO_RNN, *, rollouts: int, num_envs: int, done_factory: DoneFactory = _no_done) -> None:
    for timestep in range(rollouts):
        observations, states = _transition_tensors(timestep, num_envs)
        next_observations, next_states = _transition_tensors(timestep + 1, num_envs)
        terminated, truncated = done_factory(timestep, num_envs)
        with torch.no_grad():
            actions, _ = agent.act(observations, states, timestep=timestep, timesteps=rollouts)
            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=torch.zeros((num_envs, 1)),
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos={},
                timestep=timestep,
                timesteps=rollouts,
            )


def _assert_complete_sequences(ids: torch.Tensor, *, sequence_length: int, num_envs: int) -> None:
    ids = ids.to(dtype=torch.long).flatten()
    assert ids.numel() % sequence_length == 0
    for sequence in ids.view(-1, sequence_length):
        assert torch.unique(sequence % num_envs).numel() == 1
        torch.testing.assert_close(
            sequence[1:] // num_envs,
            sequence[:-1] // num_envs + 1,
            rtol=0,
            atol=0,
        )


def test_t80_l40_seven_env_async_rollout_exactly_matches_sequence_replay():
    """The deployed geometry must replay every policy/value sample exactly."""

    rollouts, sequence_length, num_envs = 80, 40, 7
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        learning_epochs=1,
        mini_batches=5,
    )
    _collect(
        agent,
        rollouts=rollouts,
        num_envs=num_envs,
        done_factory=_representative_async_done,
    )

    replay_ids: list[torch.Tensor] = []
    replay_log_probs: list[torch.Tensor] = []
    replay_value_ids: list[torch.Tensor] = []
    replay_values: list[torch.Tensor] = []
    original_policy_act = agent.policy.act
    original_value_act = agent.value.act

    def recording_policy_act(inputs: dict[str, Any], *, role: str = ""):
        actions, outputs = original_policy_act(inputs, role=role)
        if "taken_actions" in inputs:
            replay_ids.append(inputs["observations"][:, 0].detach().clone())
            replay_log_probs.append(outputs["log_prob"].detach().clone())
        return actions, outputs

    def recording_value_act(inputs: dict[str, Any], *, role: str = ""):
        values, outputs = original_value_act(inputs, role=role)
        if "terminated" in inputs:
            replay_value_ids.append(inputs["states"][:, 0].detach().clone())
            replay_values.append(values.detach().clone())
        return values, outputs

    agent.policy.act = recording_policy_act
    agent.value.act = recording_value_act
    agent.enable_models_training_mode(True)
    agent.update(timestep=rollouts - 1, timesteps=rollouts)

    ids = torch.cat(replay_ids).long()
    log_probs = torch.cat(replay_log_probs)
    value_ids = torch.cat(replay_value_ids).long()
    replayed_values = torch.cat(replay_values)
    expected_ids = torch.arange(rollouts * num_envs)
    assert torch.equal(ids.sort().values, expected_ids)
    assert torch.equal(value_ids.sort().values, expected_ids)

    stored_log_probs = agent.memory.get_tensor_by_name("log_prob").flatten(0, 1)
    stored_values = agent.memory.get_tensor_by_name("values").flatten(0, 1)
    torch.testing.assert_close(log_probs, stored_log_probs[ids], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(replayed_values, stored_values[value_ids], rtol=1e-5, atol=1e-6)
    assert agent.tracking_data["Policy / Approx KL"][-1] == pytest.approx(0.0, abs=1e-7)
    assert agent.tracking_data["Policy / IS ratio (mean)"][-1] == pytest.approx(1.0, abs=1e-7)
    assert agent.tracking_data["Policy / Clip fraction"][-1] == 0.0


def test_sequences_are_complete_shuffled_each_epoch_and_cover_every_sample_once():
    rollouts, sequence_length, num_envs = 8, 2, 3
    learning_epochs, mini_batches = 3, 4
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        learning_epochs=learning_epochs,
        mini_batches=mini_batches,
    )
    _collect(agent, rollouts=rollouts, num_envs=num_envs)

    calls: list[torch.Tensor] = []
    original_policy_act = agent.policy.act

    def recording_policy_act(inputs: dict[str, Any], *, role: str = ""):
        actions, outputs = original_policy_act(inputs, role=role)
        if "taken_actions" in inputs:
            calls.append(inputs["observations"][:, 0].detach().clone())
        return actions, outputs

    agent.policy.act = recording_policy_act
    agent.enable_models_training_mode(True)
    torch.manual_seed(77)
    agent.update(timestep=rollouts - 1, timesteps=rollouts)

    assert len(calls) == learning_epochs * mini_batches
    expected = torch.arange(rollouts * num_envs)
    epoch_orders = []
    epoch_partitions = []
    for epoch in range(learning_epochs):
        epoch_calls = calls[epoch * mini_batches : (epoch + 1) * mini_batches]
        for ids in epoch_calls:
            _assert_complete_sequences(ids, sequence_length=sequence_length, num_envs=num_envs)
        epoch_ids = torch.cat(epoch_calls).long()
        assert torch.equal(epoch_ids.sort().values, expected)
        epoch_orders.append(tuple(epoch_ids.tolist()))
        # Shuffling only the four already-built minibatches leaves the same
        # sequence groups together forever. Require sequence-level shuffling,
        # so early stopping cannot repeatedly expose a fixed group composition.
        epoch_partitions.append(
            frozenset(
                frozenset(ids.long().view(-1, sequence_length)[:, 0].tolist())
                for ids in epoch_calls
            )
        )
    assert len(set(epoch_orders)) == learning_epochs
    assert len(set(epoch_partitions)) > 1


def test_sequence_kl_early_stop_preserves_boundaries_and_ends_the_complete_update():
    rollouts, sequence_length, num_envs = 8, 2, 3
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        learning_epochs=5,
        mini_batches=4,
        kl_threshold=0.01,
    )
    _collect(agent, rollouts=rollouts, num_envs=num_envs)

    calls: list[torch.Tensor] = []
    original_policy_act = agent.policy.act

    def policy_act_with_second_batch_kl(inputs: dict[str, Any], *, role: str = ""):
        actions, outputs = original_policy_act(inputs, role=role)
        if "taken_actions" in inputs:
            calls.append(inputs["observations"][:, 0].detach().clone())
            if len(calls) == 2:
                outputs["log_prob"] = outputs["log_prob"] + 5
        return actions, outputs

    agent.policy.act = policy_act_with_second_batch_kl
    agent.enable_models_training_mode(True)
    agent.update(timestep=rollouts - 1, timesteps=rollouts)

    assert len(calls) == 2
    for ids in calls:
        _assert_complete_sequences(ids, sequence_length=sequence_length, num_envs=num_envs)
    assert not set(calls[0].long().tolist()) & set(calls[1].long().tolist())
    assert agent.tracking_data["Optimization / Observed minibatches"][-1] == 2
    assert agent.tracking_data["Optimization / Effective minibatches"][-1] == 1
    assert agent.tracking_data["Optimization / KL early-stop count"][-1] == 1


def test_policy_and_value_sequence_length_mismatch_fails_during_init():
    agent = _make_agent(
        rollouts=8,
        num_envs=3,
        policy_sequence_length=2,
        value_sequence_length=4,
        initialize=False,
    )
    with pytest.raises(ValueError, match=r"(?i)policy.*value.*sequence|sequence.*policy.*value"):
        agent.init()


@pytest.mark.parametrize(("policy_envs", "value_envs"), [(2, 3), (3, 4)])
def test_recurrent_spec_environment_count_mismatch_fails_during_init(policy_envs: int, value_envs: int):
    agent = _make_agent(
        rollouts=4,
        num_envs=3,
        policy_sequence_length=2,
        policy_spec_num_envs=policy_envs,
        value_spec_num_envs=value_envs,
        initialize=False,
    )
    with pytest.raises(ValueError, match=r"(?i)environment|num_envs"):
        agent.init()


def test_policy_and_value_may_have_different_hidden_topologies():
    agent = _make_agent(
        rollouts=4,
        num_envs=3,
        policy_sequence_length=2,
        policy_hidden_size=3,
        value_hidden_size=6,
        policy_num_layers=1,
        value_num_layers=2,
        mini_batches=2,
    )
    _collect(agent, rollouts=4, num_envs=3, done_factory=_representative_async_done)
    agent.enable_models_training_mode(True)
    agent.update(timestep=3, timesteps=4)
    assert agent.tracking_data["Policy / Approx KL"][-1] == pytest.approx(0.0, abs=1e-7)


def test_random_warmup_advances_hidden_but_only_a_later_complete_on_policy_rollout_is_trained():
    rollouts, sequence_length, num_envs, warmup_steps = 4, 2, 3, 3
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        mini_batches=2,
        random_timesteps=warmup_steps,
    )
    total_timesteps = warmup_steps + rollouts
    agent.pre_interaction(timestep=0, timesteps=total_timesteps)

    # Uniform actions are not samples from the Gaussian behavior policy. They
    # may warm up the environment and recurrent state, but must never enter an
    # on-policy PPO batch or advance its rollout counter.
    for timestep in range(warmup_steps):
        observations, states = _transition_tensors(timestep, num_envs)
        next_observations, next_states = _transition_tensors(timestep + 1, num_envs)
        terminated = torch.zeros((num_envs, 1), dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        if timestep == 0:
            terminated[0] = True
        with torch.no_grad():
            actions, _ = agent.act(observations, states, timestep=timestep, timesteps=total_timesteps)
            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=torch.zeros((num_envs, 1)),
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos={},
                timestep=timestep,
                timesteps=total_timesteps,
            )
        agent.post_interaction(timestep=timestep, timesteps=total_timesteps)
        assert len(agent.memory) == 0
        assert agent._rollout == 0
        if timestep == 0:
            assert torch.count_nonzero(agent._rnn_initial_states["policy"][0][:, 0]) == 0
            assert torch.count_nonzero(agent._rnn_initial_states["value"][0][:, 0]) == 0
            assert torch.count_nonzero(agent._rnn_initial_states["policy"][0][:, 1:]) > 0
            assert torch.count_nonzero(agent._rnn_initial_states["value"][0][:, 1:]) > 0

    warm_policy_hidden = [state.clone() for state in agent._rnn_initial_states["policy"]]
    warm_value_hidden = [state.clone() for state in agent._rnn_initial_states["value"]]

    # Now collect exactly one complete on-policy rollout. post_interaction must
    # count these transitions and trigger one ordinary PPO update at the end.
    for rollout_step in range(rollouts):
        timestep = warmup_steps + rollout_step
        observations, states = _transition_tensors(timestep, num_envs)
        next_observations, next_states = _transition_tensors(timestep + 1, num_envs)
        terminated = torch.zeros((num_envs, 1), dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        if rollout_step == 1:
            truncated[2] = True
        with torch.no_grad():
            actions, _ = agent.act(observations, states, timestep=timestep, timesteps=total_timesteps)
            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=torch.zeros((num_envs, 1)),
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos={},
                timestep=timestep,
                timesteps=total_timesteps,
            )
        if rollout_step == 0:
            for index, expected in enumerate(warm_policy_hidden):
                stored = agent.memory.get_tensor_by_name(f"rnn_policy_{index}")[0].transpose(0, 1)
                torch.testing.assert_close(stored, expected, rtol=0, atol=0)
            for index, expected in enumerate(warm_value_hidden):
                stored = agent.memory.get_tensor_by_name(f"rnn_value_{index}")[0].transpose(0, 1)
                torch.testing.assert_close(stored, expected, rtol=0, atol=0)
        agent.post_interaction(timestep=timestep, timesteps=total_timesteps)

    assert len(agent.memory) == rollouts * num_envs
    assert agent._rollout == rollouts
    assert torch.isfinite(agent.memory.get_tensor_by_name("log_prob")).all()
    assert agent.tracking_data["Policy / Approx KL"][-1] == pytest.approx(0.0, abs=1e-7)
    assert agent.tracking_data["Policy / IS ratio (mean)"][-1] == pytest.approx(1.0, abs=1e-7)


def test_async_hidden_lifecycle_stores_pre_action_state_resets_only_done_envs_and_survives_update():
    rollouts, sequence_length, num_envs = 4, 2, 3
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        mini_batches=2,
    )

    for timestep in range(rollouts):
        observations, states = _transition_tensors(timestep, num_envs)
        next_observations, next_states = _transition_tensors(timestep + 1, num_envs)
        terminated = torch.zeros((num_envs, 1), dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        if timestep == 0:
            terminated[0] = True
        if timestep == 1:
            truncated[2] = True

        pre_policy = [state.detach().clone() for state in agent._rnn_initial_states["policy"]]
        pre_value = [state.detach().clone() for state in agent._rnn_initial_states["value"]]
        with torch.no_grad():
            actions, policy_outputs = agent.act(observations, states, timestep=timestep, timesteps=rollouts)
            # record_transition resets the tensors returned by act in-place;
            # preserve the unmasked result as the independent reset oracle.
            expected_policy_final = [state.detach().clone() for state in policy_outputs["rnn"]]
            _, expected_value_outputs = agent.value.act(
                {"observations": observations, "states": states, "rnn": pre_value}, role="value"
            )
            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=torch.zeros((num_envs, 1)),
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos={},
                timestep=timestep,
                timesteps=rollouts,
            )

        for index, expected in enumerate(pre_policy):
            stored = agent.memory.get_tensor_by_name(f"rnn_policy_{index}")[timestep].transpose(0, 1)
            torch.testing.assert_close(stored, expected, rtol=0, atol=0)
        for index, expected in enumerate(pre_value):
            stored = agent.memory.get_tensor_by_name(f"rnn_value_{index}")[timestep].transpose(0, 1)
            torch.testing.assert_close(stored, expected, rtol=0, atol=0)

        done = (terminated | truncated).flatten()
        for expected_final, actual in zip(expected_policy_final, agent._rnn_initial_states["policy"]):
            expected_final[:, done] = 0
            torch.testing.assert_close(actual, expected_final, rtol=0, atol=0)
        for expected_final, actual in zip(expected_value_outputs["rnn"], agent._rnn_initial_states["value"]):
            expected_final = expected_final.detach().clone()
            expected_final[:, done] = 0
            torch.testing.assert_close(actual, expected_final, rtol=0, atol=0)
        assert agent._rnn_initial_states["policy"][0].data_ptr() != agent._rnn_initial_states["value"][0].data_ptr()

    live_policy = [state.detach().clone() for state in agent._rnn_initial_states["policy"]]
    live_value = [state.detach().clone() for state in agent._rnn_initial_states["value"]]
    agent.enable_models_training_mode(True)
    agent.update(timestep=rollouts - 1, timesteps=rollouts)
    for expected, actual in zip(live_policy, agent._rnn_initial_states["policy"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(live_value, agent._rnn_initial_states["value"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_multilayer_lstm_hidden_and_cell_rollout_replay_done_reset_and_nonzero_update():
    rollouts, sequence_length, num_envs = 8, 4, 3
    observation_space = gymnasium.spaces.Box(low=-10_000, high=10_000, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-10_000, high=10_000, shape=(5,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,))
    torch.manual_seed(314)
    policy = RealLSTMPolicy(
        observation_space,
        state_space,
        action_space,
        sequence_length=sequence_length,
        num_envs=num_envs,
        hidden_size=4,
        num_layers=2,
    )
    value = RealLSTMValue(
        observation_space,
        state_space,
        action_space,
        sequence_length=sequence_length,
        num_envs=num_envs,
        hidden_size=6,
        num_layers=2,
    )
    agent = _make_agent_from_models(
        policy,
        value,
        rollouts=rollouts,
        num_envs=num_envs,
        learning_epochs=1,
        mini_batches=1,
        learning_rate=2e-3,
    )

    def lstm_done(timestep: int, count: int):
        terminated, truncated = _no_done(timestep, count)
        if timestep == 0:
            truncated[1] = True
        if timestep == 3:
            terminated[0] = True
        if timestep == 5:
            truncated[2] = True
        return terminated, truncated

    _collect(agent, rollouts=rollouts, num_envs=num_envs, done_factory=lstm_done)
    assert agent._rnn_tensors_names == [
        "rnn_policy_0",
        "rnn_policy_1",
        "rnn_value_0",
        "rnn_value_1",
    ]
    # Each named tensor is a pre-action state. Both hidden and cell, in both
    # models and both layers, must be zero immediately after each async done.
    for name in agent._rnn_tensors_names:
        state = agent.memory.get_tensor_by_name(name)
        assert state.shape[2] == 2
        assert torch.count_nonzero(state) > 0
        assert torch.count_nonzero(state[1, 1]) == 0
        assert torch.count_nonzero(state[4, 0]) == 0
        assert torch.count_nonzero(state[6, 2]) == 0

    reward = (0.3 + torch.sin(torch.arange(rollouts * num_envs) * 0.41)).view(rollouts, num_envs, 1)
    agent.memory.set_tensor_by_name("rewards", reward)
    stored_log_prob = agent.memory.get_tensor_by_name("log_prob").flatten(0, 1).clone()
    stored_values = agent.memory.get_tensor_by_name("values").flatten(0, 1).clone()
    replay_ids: list[torch.Tensor] = []
    replay_log_probs: list[torch.Tensor] = []
    value_ids: list[torch.Tensor] = []
    replay_values: list[torch.Tensor] = []
    original_policy_act = policy.act
    original_value_act = value.act

    def recording_policy_act(inputs: dict[str, Any], *, role: str = ""):
        actions, outputs = original_policy_act(inputs, role=role)
        if "taken_actions" in inputs:
            replay_ids.append(inputs["observations"][:, 0].clone())
            replay_log_probs.append(outputs["log_prob"].detach().clone())
        return actions, outputs

    def recording_value_act(inputs: dict[str, Any], *, role: str = ""):
        values, outputs = original_value_act(inputs, role=role)
        if "terminated" in inputs:
            value_ids.append(inputs["states"][:, 0].clone())
            replay_values.append(values.detach().clone())
        return values, outputs

    policy.act = recording_policy_act
    value.act = recording_value_act
    policy_before = [parameter.detach().clone() for parameter in policy.parameters()]
    value_before = [parameter.detach().clone() for parameter in value.parameters()]
    agent.enable_models_training_mode(True)
    agent.update(timestep=rollouts - 1, timesteps=rollouts)

    ids = torch.cat(replay_ids).long()
    replayed_log_prob = torch.cat(replay_log_probs)
    replayed_value_ids = torch.cat(value_ids).long()
    replayed_values = torch.cat(replay_values)
    expected_ids = torch.arange(rollouts * num_envs)
    assert torch.equal(ids.sort().values, expected_ids)
    assert torch.equal(replayed_value_ids.sort().values, expected_ids)
    torch.testing.assert_close(replayed_log_prob, stored_log_prob[ids], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(replayed_values, stored_values[replayed_value_ids], rtol=1e-5, atol=1e-6)
    assert any(not torch.equal(old, new) for old, new in zip(policy_before, policy.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(value_before, value.parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in value.parameters())
    assert agent.tracking_data["Policy / IS ratio (mean)"][-1] == pytest.approx(1.0, abs=1e-6)


def test_shared_recurrent_actor_critic_rollout_replay_done_reset_and_nonzero_update():
    rollouts, sequence_length, num_envs = 8, 2, 3
    observation_space = gymnasium.spaces.Box(low=-10_000, high=10_000, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-10_000, high=10_000, shape=(5,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,))
    torch.manual_seed(2718)
    shared = SharedRealGRUActorCritic(
        observation_space,
        state_space,
        action_space,
        sequence_length=sequence_length,
        num_envs=num_envs,
        hidden_size=5,
        num_layers=2,
    )
    agent = _make_agent_from_models(
        shared,
        shared,
        rollouts=rollouts,
        num_envs=num_envs,
        learning_epochs=1,
        mini_batches=1,
        learning_rate=2e-3,
    )

    _collect(
        agent,
        rollouts=rollouts,
        num_envs=num_envs,
        done_factory=_representative_async_done,
    )
    assert agent.policy is agent.value
    assert agent._rnn_initial_states["policy"] is agent._rnn_initial_states["value"]
    assert "rnn_policy_0" in agent.memory.tensors
    assert "rnn_value_0" not in agent.memory.tensors
    shared_hidden = agent.memory.get_tensor_by_name("rnn_policy_0")
    assert shared_hidden.shape[2] == 2
    # _representative_async_done truncates env 1 at t=0.
    assert torch.count_nonzero(shared_hidden[1, 1]) == 0
    assert torch.count_nonzero(shared_hidden) > 0

    optimizer_parameters = [parameter for group in agent.optimizer.param_groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in optimizer_parameters}) == len(optimizer_parameters)
    assert len(optimizer_parameters) == len(list(shared.parameters()))

    reward = (0.4 + torch.cos(torch.arange(rollouts * num_envs) * 0.29)).view(rollouts, num_envs, 1)
    agent.memory.set_tensor_by_name("rewards", reward)
    stored_log_prob = agent.memory.get_tensor_by_name("log_prob").flatten(0, 1).clone()
    stored_values = agent.memory.get_tensor_by_name("values").flatten(0, 1).clone()
    policy_ids: list[torch.Tensor] = []
    replay_log_probs: list[torch.Tensor] = []
    value_ids: list[torch.Tensor] = []
    replay_values: list[torch.Tensor] = []
    original_act = shared.act

    def recording_act(inputs: dict[str, Any], *, role: str = ""):
        outputs = original_act(inputs, role=role)
        if role == "policy" and "taken_actions" in inputs:
            policy_ids.append(inputs["observations"][:, 0].clone())
            replay_log_probs.append(outputs[1]["log_prob"].detach().clone())
        if role == "value" and "terminated" in inputs:
            value_ids.append(inputs["observations"][:, 0].clone())
            replay_values.append(outputs[0].detach().clone())
        return outputs

    shared.act = recording_act
    parameters_before = [parameter.detach().clone() for parameter in shared.parameters()]
    agent.enable_models_training_mode(True)
    agent.update(timestep=rollouts - 1, timesteps=rollouts)

    ids = torch.cat(policy_ids).long()
    replayed_value_ids = torch.cat(value_ids).long()
    expected_ids = torch.arange(rollouts * num_envs)
    assert torch.equal(ids.sort().values, expected_ids)
    assert torch.equal(replayed_value_ids.sort().values, expected_ids)
    torch.testing.assert_close(torch.cat(replay_log_probs), stored_log_prob[ids], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.cat(replay_values), stored_values[replayed_value_ids], rtol=1e-5, atol=1e-6)
    assert any(not torch.equal(old, new) for old, new in zip(parameters_before, shared.parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in shared.parameters())
    assert agent.tracking_data["Policy / IS ratio (mean)"][-1] == pytest.approx(1.0, abs=1e-6)


def test_pre_interaction_zero_discards_a_partial_rollout_and_resets_the_run_lifecycle():
    """A new trainer run must not splice a reset environment into an old sequence."""

    rollouts, sequence_length, num_envs = 4, 2, 3
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        mini_batches=2,
    )
    agent.pre_interaction(timestep=0, timesteps=100)

    observations, states = _transition_tensors(0, num_envs)
    next_observations, next_states = _transition_tensors(1, num_envs)
    terminated, truncated = _no_done(0, num_envs)
    with torch.no_grad():
        actions, _ = agent.act(observations, states, timestep=0, timesteps=100)
        agent.record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=torch.zeros((num_envs, 1)),
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos={},
            timestep=0,
            timesteps=100,
        )
    agent.post_interaction(timestep=0, timesteps=100)

    assert len(agent.memory) == num_envs
    assert agent._rollout == 1
    assert any(torch.count_nonzero(state) for state in agent._rnn_initial_states["policy"])
    live_before = [state.clone() for state in agent._rnn_initial_states["policy"]]
    agent.pre_interaction(timestep=1, timesteps=100)
    for expected, actual in zip(live_before, agent._rnn_initial_states["policy"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # This models stopping a trainer mid-rollout, resetting the environment,
    # then starting a fresh train/eval call whose local timestep is zero.
    agent.pre_interaction(timestep=0, timesteps=100)
    assert len(agent.memory) == 0
    assert agent._rollout == 0
    assert agent._current_next_observations is None
    assert agent._current_next_states is None
    assert agent._current_log_prob is None
    for role in ("policy", "value"):
        assert all(torch.count_nonzero(state) == 0 for state in agent._rnn_initial_states[role])
        assert all(torch.count_nonzero(state) == 0 for state in agent._rnn_final_states[role])


def test_explicit_rnn_reset_selects_environments_and_discards_the_partial_rollout():
    rollouts, sequence_length, num_envs = 4, 2, 3
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        mini_batches=2,
    )
    _collect(agent, rollouts=2, num_envs=num_envs)
    assert len(agent.memory) > 0

    snapshots = {}
    for group_name, state_group in (
        ("initial", agent._rnn_initial_states),
        ("final", agent._rnn_final_states),
    ):
        for role, states in state_group.items():
            for index, state in enumerate(states):
                snapshots[(group_name, role, index)] = state.clone()

    agent.reset_rnn_states(env_ids=[1])
    for group_name, state_group in (
        ("initial", agent._rnn_initial_states),
        ("final", agent._rnn_final_states),
    ):
        for role, states in state_group.items():
            for index, state in enumerate(states):
                expected = snapshots[(group_name, role, index)].clone()
                expected[:, 1] = 0
                torch.testing.assert_close(state, expected, rtol=0, atol=0)
    assert len(agent.memory) == 0
    assert agent._rollout == 0
    assert agent._current_next_observations is None
    assert agent._current_next_states is None
    assert agent._current_log_prob is None

    agent.reset_rnn_states()
    for state_group in (agent._rnn_initial_states, agent._rnn_final_states):
        for states in state_group.values():
            assert all(torch.count_nonzero(state) == 0 for state in states)
    assert len(agent.memory) == 0


def test_nonzero_learning_rate_remains_finite_across_multiple_rollout_update_cycles():
    rollouts, sequence_length, num_envs = 8, 2, 3
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        learning_epochs=2,
        mini_batches=3,
        learning_rate=3e-3,
    )
    metric_names = (
        "Loss / Policy loss",
        "Loss / Value loss",
        "Policy / Approx KL",
        "Policy / IS ratio (mean)",
        "Value / Explained variance",
        "Optimization / Grad norm",
    )

    for update_index in range(5):
        agent.enable_models_training_mode(False)
        _collect(
            agent,
            rollouts=rollouts,
            num_envs=num_envs,
            done_factory=_representative_async_done,
        )
        reward = (
            0.2
            + torch.sin(torch.arange(rollouts * num_envs, dtype=torch.float32) * 0.37 + update_index)
        ).view(rollouts, num_envs, 1)
        agent.memory.set_tensor_by_name("rewards", reward)
        before = [
            parameter.detach().clone()
            for model in (agent.policy, agent.value)
            for parameter in model.parameters()
        ]

        agent.enable_models_training_mode(True)
        agent.update(timestep=(update_index + 1) * rollouts - 1, timesteps=5 * rollouts)

        after = [
            parameter.detach()
            for model in (agent.policy, agent.value)
            for parameter in model.parameters()
        ]
        assert any(not torch.equal(old, new) for old, new in zip(before, after))
        assert all(torch.isfinite(parameter).all() for parameter in after)
        for name in metric_names:
            assert math.isfinite(agent.tracking_data[name][-1]), (update_index, name, agent.tracking_data[name][-1])


def test_stored_hidden_drift_after_parameter_change_is_finite_and_a_done_restores_exact_replay(
    record_property,
):
    """Quantify the known stale-hidden approximation without assuming zero burn-in error."""

    rollouts, sequence_length, num_envs = 12, 4, 1
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        mini_batches=3,
    )
    terminated = torch.zeros((rollouts, 1, 1), dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    terminated[9, 0] = True

    for timestep in range(rollouts):
        scalar = torch.tensor(float(timestep) / 20)
        next_scalar = torch.tensor(float(timestep + 1) / 20)
        observations = torch.stack(
            (scalar, torch.sin(scalar), torch.cos(scalar), scalar.square())
        ).view(1, -1)
        states = torch.cat((observations, scalar.view(1, 1)), dim=1)
        next_observations = torch.stack(
            (next_scalar, torch.sin(next_scalar), torch.cos(next_scalar), next_scalar.square())
        ).view(1, -1)
        next_states = torch.cat((next_observations, next_scalar.view(1, 1)), dim=1)
        with torch.no_grad():
            actions, _ = agent.act(observations, states, timestep=timestep, timesteps=rollouts)
            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=torch.zeros((1, 1)),
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated[timestep],
                truncated=truncated[timestep],
                infos={},
                timestep=timestep,
                timesteps=rollouts,
            )

    # Apply a deterministic, bounded parameter update after the old hidden
    # states have been stored. This isolates the stale-hidden effect from PPO's
    # stochastic minibatch optimization while exercising the same replay path.
    with torch.no_grad():
        for parameter_index, parameter in enumerate(agent.policy.gru.parameters()):
            direction = torch.linspace(-1, 1, parameter.numel()).view_as(parameter)
            parameter.add_(0.025 * (parameter_index + 1) * direction)

    observations = agent.memory.get_tensor_by_name("observations").flatten(0, 1)
    actions = agent.memory.get_tensor_by_name("actions").flatten(0, 1)
    stored_hidden = agent.memory.get_tensor_by_name("rnn_policy_0").flatten(0, 1)
    flat_terminated = terminated.flatten(0, 1)
    flat_truncated = truncated.flatten(0, 1)

    oracle_log_probs = []
    hidden = torch.zeros((1, 1, agent.policy.hidden_size))
    agent.policy.eval()
    with torch.no_grad():
        for timestep in range(rollouts):
            _, outputs = agent.policy.act(
                {
                    "observations": observations[timestep : timestep + 1],
                    "taken_actions": actions[timestep : timestep + 1],
                    "rnn": [hidden],
                },
                role="policy",
            )
            oracle_log_probs.append(outputs["log_prob"])
            hidden = outputs["rnn"][0]
            if terminated[timestep, 0] or truncated[timestep, 0]:
                hidden.zero_()
    oracle_log_probs = torch.cat(oracle_log_probs)

    agent.policy.train()
    with torch.no_grad():
        _, replay_outputs = agent.policy.act(
            {
                "observations": observations,
                "taken_actions": actions,
                "rnn": [stored_hidden.transpose(0, 1)],
                "terminated": flat_terminated,
                "truncated": flat_truncated,
            },
            role="policy",
        )
    error = (replay_outputs["log_prob"] - oracle_log_probs).abs().flatten()
    stale_error = error[4:10]
    recovered_error = error[10:]
    record_property("stored_hidden_max_log_prob_error", stale_error.max().item())
    record_property("stored_hidden_mean_log_prob_error", stale_error.mean().item())
    record_property("post_done_max_log_prob_error", recovered_error.max().item())

    assert torch.isfinite(error).all()
    torch.testing.assert_close(error[:4], torch.zeros(4), rtol=0, atol=1e-6)
    assert stale_error.max() > 1e-6
    # For the fixed 2.5%-per-index perturbation above, stale state should be a
    # measurable approximation error, not a discontinuous/catastrophic jump.
    assert stale_error.max() < 0.1
    torch.testing.assert_close(recovered_error, torch.zeros_like(recovered_error), rtol=0, atol=1e-6)


def test_repeated_kl_early_stops_do_not_lock_observation_to_fixed_environment_prefix():
    rollouts, sequence_length, num_envs = 8, 2, 4
    agent = _make_agent(
        rollouts=rollouts,
        num_envs=num_envs,
        policy_sequence_length=sequence_length,
        learning_epochs=4,
        mini_batches=4,
        kl_threshold=0.01,
    )
    first_batches: list[torch.Tensor] = []
    original_policy_act = agent.policy.act

    def policy_act_with_forced_kl(inputs: dict[str, Any], *, role: str = ""):
        actions, outputs = original_policy_act(inputs, role=role)
        if "taken_actions" in inputs:
            first_batches.append(inputs["observations"][:, 0].detach().clone())
            outputs["log_prob"] = outputs["log_prob"] + 5
        return actions, outputs

    agent.policy.act = policy_act_with_forced_kl
    torch.manual_seed(2026)
    for update_index in range(8):
        agent.enable_models_training_mode(False)
        _collect(agent, rollouts=rollouts, num_envs=num_envs)
        agent.enable_models_training_mode(True)
        agent.update(timestep=(update_index + 1) * rollouts - 1, timesteps=8 * rollouts)
        assert agent.tracking_data["Optimization / Observed minibatches"][-1] == 1
        assert agent.tracking_data["Optimization / Effective minibatches"][-1] == 0

    assert len(first_batches) == 8
    sequence_start_groups = []
    observed_envs = set()
    for ids in first_batches:
        _assert_complete_sequences(ids, sequence_length=sequence_length, num_envs=num_envs)
        sequence_starts = ids.long().view(-1, sequence_length)[:, 0]
        sequence_start_groups.append(tuple(sorted(sequence_starts.tolist())))
        observed_envs.update((sequence_starts % num_envs).tolist())
    assert len(set(sequence_start_groups)) > 1
    assert observed_envs == set(range(num_envs))


def _multimodal_space() -> gymnasium.spaces.Dict:
    return gymnasium.spaces.Dict(
        {
            "image": gymnasium.spaces.Box(-10, 10, shape=(1, 3, 3), dtype=float),
            "ego": gymnasium.spaces.Box(-10, 10, shape=(2,), dtype=float),
            "other_0": gymnasium.spaces.Box(-10, 10, shape=(1, 2), dtype=float),
            "other_1": gymnasium.spaces.Box(-10, 10, shape=(1, 2), dtype=float),
            "others_mask": gymnasium.spaces.Box(0, 1, shape=(2,), dtype=float),
        }
    )


def _multimodal_network(sequence_length: int) -> dict[str, Any]:
    return {
        "cnn": {
            "channels": [2],
            "kernels": [[1, 1]],
            "strides": [[1, 1]],
            "paddings": [[0, 0]],
            "activation": "relu",
        },
        "attention": {"embed_dim": 2, "num_heads": 1},
        "gru": {"hidden_size": 4, "num_layers": 1, "sequence_length": sequence_length},
        "mlp": {"hidden_dims": [4], "use_layernorm": False, "activation": "relu"},
    }


def _multimodal_samples(rollouts: int, num_envs: int, *, offset: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(900 + int(offset * 10))
    count = rollouts * num_envs
    native = {
        "image": torch.randn((count, 1, 3, 3), generator=generator) + offset,
        "ego": torch.randn((count, 2), generator=generator) + offset,
        "other_0": torch.randn((count, 1, 2), generator=generator),
        "other_1": torch.randn((count, 1, 2), generator=generator),
        "others_mask": torch.ones((count, 2)),
    }
    return flatten_tensorized_space(native).view(rollouts, num_envs, -1)


def test_attention_fusion_receives_valid_ratio_derived_from_mask():
    observation_space = _multimodal_space()
    action_space = gymnasium.spaces.Box(-1, 1, shape=(2,))
    policy = CNNGRUAttentionMLPPolicy(
        observation_space=observation_space,
        state_space=observation_space,
        action_space=action_space,
        device="cpu",
        num_envs=2,
        network=_multimodal_network(sequence_length=1),
        reduction="sum",
    )
    identical_other = torch.tensor([[[1.0, -1.0]], [[1.0, -1.0]]])
    native = {
        "image": torch.zeros((2, 1, 3, 3)),
        "ego": torch.zeros((2, 2)),
        "other_0": identical_other.clone(),
        "other_1": identical_other.clone(),
        "others_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
    }
    preprocessor = SelectiveRunningStandardScaler(
        size=observation_space,
        exclude_keys=["image", "others_mask"],
        device="cpu",
    )
    observations = preprocessor(flatten_tensorized_space(native), train=True)
    fusion_inputs = []
    hook = policy.fusion_layer[0].register_forward_pre_hook(
        lambda _module, args: fusion_inputs.append(args[0].detach().clone())
    )
    try:
        policy.eval()
        with torch.no_grad():
            policy.compute({"observations": observations}, role="policy")
    finally:
        hook.remove()

    assert len(fusion_inputs) == 1
    torch.testing.assert_close(fusion_inputs[0][:, -1], torch.tensor([0.5, 1.0]))


def test_separate_feature_projection_bypasses_fusion_and_preserves_gru_input_size():
    observation_space = _multimodal_space()
    action_space = gymnasium.spaces.Box(-1, 1, shape=(2,))
    network = _multimodal_network(sequence_length=1)
    network["cnn"]["output_dim"] = 2
    network["gru"]["hidden_size"] = 4
    network["gru"]["separate_feature_projection"] = True
    policy = CNNGRUAttentionMLPPolicy(
        observation_space=observation_space,
        state_space=observation_space,
        action_space=action_space,
        device="cpu",
        num_envs=2,
        network=network,
        reduction="sum",
    )

    assert policy.fusion_layer is None
    assert policy.separate_feature_projection_dims == (2, 1, 1)
    assert sum(policy.separate_feature_projection_dims) == policy.hidden_size
    assert policy.depth_projection.in_features == 2
    assert policy.ego_projection.in_features == 2
    assert policy.others_projection.in_features == 2

    native = {
        "image": torch.zeros((2, 1, 3, 3)),
        "ego": torch.zeros((2, 2)),
        "other_0": torch.zeros((2, 1, 2)),
        "other_1": torch.zeros((2, 1, 2)),
        "others_mask": torch.ones((2, 2)),
    }
    observations = flatten_tensorized_space(native)
    policy.eval()
    with torch.no_grad():
        actions, _ = policy.compute({"observations": observations}, role="policy")
    assert actions.shape == (2, 2)


def test_actual_cnn_gru_attention_models_replay_t80_l40_async_done_exactly():
    """Integration oracle for the actual model classes used by the swarm config."""

    rollouts, sequence_length, num_envs = 80, 40, 7
    observation_space = _multimodal_space()
    state_space = _multimodal_space()
    action_space = gymnasium.spaces.Box(-1, 1, shape=(2,))
    network = _multimodal_network(sequence_length)
    torch.manual_seed(99)
    policy = CNNGRUAttentionMLPPolicy(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        num_envs=num_envs,
        network=network,
        reduction="sum",
    )
    value = CNNGRUAttentionMLPValue(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        num_envs=num_envs,
        network=network,
    )
    observations = _multimodal_samples(rollouts, num_envs, offset=0.0)
    states = _multimodal_samples(rollouts, num_envs, offset=0.5)
    terminated = torch.zeros((rollouts, num_envs, 1), dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    for timestep in range(rollouts):
        step_terminated, step_truncated = _representative_async_done(timestep, num_envs)
        terminated[timestep] = step_terminated
        truncated[timestep] = step_truncated

    policy.eval()
    value.eval()
    policy_hidden = torch.zeros((1, num_envs, 4))
    value_hidden = torch.zeros((1, num_envs, 4))
    policy_hidden_before = []
    value_hidden_before = []
    actions = []
    log_probs = []
    values = []
    with torch.no_grad():
        for timestep in range(rollouts):
            policy_hidden_before.append(policy_hidden.clone())
            value_hidden_before.append(value_hidden.clone())
            action, policy_outputs = policy.act(
                {"observations": observations[timestep], "rnn": [policy_hidden]}, role="policy"
            )
            value_output, value_outputs = value.act(
                {"states": states[timestep], "rnn": [value_hidden]}, role="value"
            )
            actions.append(action)
            log_probs.append(policy_outputs["log_prob"])
            values.append(value_output)
            policy_hidden = policy_outputs["rnn"][0]
            value_hidden = value_outputs["rnn"][0]
            done = (terminated[timestep] | truncated[timestep]).flatten()
            policy_hidden[:, done] = 0
            value_hidden[:, done] = 0

    # Match Memory.all_sequence_indexes: environment-major, then split each
    # environment's trajectory into two complete length-40 sequences.
    sequence_indexes = torch.cat(
        [torch.arange(env, rollouts * num_envs, num_envs) for env in range(num_envs)]
    )
    flat_observations = observations.flatten(0, 1)
    flat_states = states.flatten(0, 1)
    flat_actions = torch.stack(actions).flatten(0, 1)
    flat_log_probs = torch.stack(log_probs).flatten(0, 1)
    flat_values = torch.stack(values).flatten(0, 1)
    flat_terminated = terminated.flatten(0, 1)
    flat_truncated = truncated.flatten(0, 1)
    stored_policy_hidden = torch.stack(policy_hidden_before).permute(0, 2, 1, 3).flatten(0, 1)
    stored_value_hidden = torch.stack(value_hidden_before).permute(0, 2, 1, 3).flatten(0, 1)

    policy.train()
    value.train()
    with torch.no_grad():
        _, replay_policy_outputs = policy.act(
            {
                "observations": flat_observations[sequence_indexes],
                "taken_actions": flat_actions[sequence_indexes],
                "rnn": [stored_policy_hidden[sequence_indexes].transpose(0, 1)],
                "terminated": flat_terminated[sequence_indexes],
                "truncated": flat_truncated[sequence_indexes],
            },
            role="policy",
        )
        replay_values, _ = value.act(
            {
                "states": flat_states[sequence_indexes],
                "rnn": [stored_value_hidden[sequence_indexes].transpose(0, 1)],
                "terminated": flat_terminated[sequence_indexes],
                "truncated": flat_truncated[sequence_indexes],
            },
            role="value",
        )

    torch.testing.assert_close(
        replay_policy_outputs["log_prob"], flat_log_probs[sequence_indexes], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(replay_values, flat_values[sequence_indexes], rtol=1e-5, atol=1e-6)


def _deployment_multimodal_space() -> gymnasium.spaces.Dict:
    """Body-rate deployment shape: 16x5 ego and 4x5 history per neighbor."""

    return gymnasium.spaces.Dict(
        {
            "image": gymnasium.spaces.Box(-float("inf"), float("inf"), shape=(1, 32, 32), dtype="float32"),
            "ego": gymnasium.spaces.Box(-float("inf"), float("inf"), shape=(80,), dtype="float32"),
            "other_0": gymnasium.spaces.Box(-float("inf"), float("inf"), shape=(1, 20), dtype="float32"),
            "other_1": gymnasium.spaces.Box(-float("inf"), float("inf"), shape=(1, 20), dtype="float32"),
            "other_2": gymnasium.spaces.Box(-float("inf"), float("inf"), shape=(1, 20), dtype="float32"),
        }
    )


def _deployment_network() -> dict[str, Any]:
    return {
        "cnn": {
            "channels": [6, 8, 8],
            "kernels": [[3, 3], [3, 3], [3, 3]],
            "strides": [[2, 2], [2, 2], [2, 2]],
            "paddings": [[1, 1], [1, 1], [0, 0]],
            "output_dim": 48,
            "activation": "relu",
        },
        "attention": {"embed_dim": 64, "num_heads": 4},
        "gru": {"hidden_size": 64, "num_layers": 1, "sequence_length": 40},
        "mlp": {"hidden_dims": [32], "use_layernorm": True, "activation": "relu"},
    }


def _deployment_multimodal_samples(rollouts: int, num_envs: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    count = rollouts * num_envs
    native = {
        "image": torch.rand((count, 1, 32, 32), generator=generator),
        "ego": torch.randn((count, 80), generator=generator) * 0.25,
        "other_0": torch.randn((count, 1, 20), generator=generator) * 0.2,
        "other_1": torch.randn((count, 1, 20), generator=generator) * 0.2,
        "other_2": torch.randn((count, 1, 20), generator=generator) * 0.2,
    }
    return flatten_tensorized_space(native).view(rollouts, num_envs, -1)


def test_deployment_shape_full_ppo_rnn_update_smoke(record_property):
    """Run the body-rate YAML geometry and optimizer settings on CPU end to end."""

    rollouts, sequence_length, num_envs = 80, 40, 3
    observation_space = _deployment_multimodal_space()
    state_space = _deployment_multimodal_space()
    action_space = gymnasium.spaces.Box(-1, 1, shape=(4,), dtype="float32")
    network = _deployment_network()
    torch.manual_seed(4242)
    policy = CNNGRUAttentionMLPPolicy(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        num_envs=num_envs,
        network=network,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-1.6,
        max_log_std=0.26,
        initial_log_std=0.0,
        reduction="sum",
    )
    value = CNNGRUAttentionMLPValue(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        num_envs=num_envs,
        network=network,
        clip_actions=False,
    )
    for model in (policy, value):
        assert model._cnn_backbone_out_shape == (8, 3, 3)
        assert model._cnn_projection_flat_dim == 72
        assert model._cnn_out_dim == 48
        projection = model.cnn_projection
        assert isinstance(projection[0], nn.Flatten)
        assert isinstance(projection[1], nn.Linear)
        assert projection[1].in_features == 72
        assert projection[1].out_features == 48
    cfg = _agent_cfg(
        rollouts=rollouts,
        learning_epochs=5,
        mini_batches=5,
        learning_rate=1e-3,
        kl_threshold=0.1,
    )
    cfg.update(
        {
            "learning_rate_scheduler": KLAdaptiveLR,
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.01, "max_lr": 1e-3},
            "observation_preprocessor": SelectiveRunningStandardScaler,
            "observation_preprocessor_kwargs": {
                "size": observation_space,
                "exclude_keys": ["image"],
                "device": "cpu",
            },
            "state_preprocessor": SelectiveRunningStandardScaler,
            "state_preprocessor_kwargs": {
                "size": state_space,
                "exclude_keys": ["image"],
                "device": "cpu",
            },
            "value_preprocessor": RunningStandardScaler,
            "value_preprocessor_kwargs": {"size": 1, "device": "cpu"},
            "grad_norm_clip": 1.0,
            "value_loss_scale": 1.0,
            "time_limit_bootstrap": False,
        }
    )
    agent = PPO_RNN(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=rollouts, num_envs=num_envs, device="cpu"),
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    observations = _deployment_multimodal_samples(rollouts + 1, num_envs, seed=7001)
    states = _deployment_multimodal_samples(rollouts + 1, num_envs, seed=7002)
    terminated = torch.zeros((rollouts, num_envs, 1), dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    terminated[17, 0] = True
    terminated[55, 1] = True
    truncated[0, 1] = True
    truncated[39, 2] = True
    truncated[78, 0] = True

    for timestep in range(rollouts):
        reward = (
            0.15
            + torch.sin(torch.arange(num_envs, dtype=torch.float32).view(-1, 1) + timestep * 0.17)
        )
        with torch.no_grad():
            actions, _ = agent.act(
                observations[timestep], states[timestep], timestep=timestep, timesteps=rollouts
            )
            agent.record_transition(
                observations=observations[timestep],
                states=states[timestep],
                actions=actions,
                rewards=reward,
                next_observations=observations[timestep + 1],
                next_states=states[timestep + 1],
                terminated=terminated[timestep],
                truncated=truncated[timestep],
                infos={},
                timestep=timestep,
                timesteps=rollouts,
            )

    assert agent._observation_preprocessor._selected_size == 140
    assert agent._state_preprocessor._selected_size == 140
    # Dict flattening is sorted: ego occupies [0:80], then image [80:1104].
    scaled_probe = agent._observation_preprocessor(observations[0])
    torch.testing.assert_close(scaled_probe[:, 80:1104], observations[0, :, 80:1104], rtol=0, atol=0)
    for name in ("rnn_policy_0", "rnn_value_0"):
        hidden = agent.memory.get_tensor_by_name(name)
        assert torch.count_nonzero(hidden[1, 1]) == 0
        assert torch.count_nonzero(hidden[18, 0]) == 0
        assert torch.count_nonzero(hidden[40, 2]) == 0

    policy_before = [parameter.detach().clone() for parameter in policy.parameters()]
    value_before = [parameter.detach().clone() for parameter in value.parameters()]
    agent.enable_models_training_mode(True)
    agent.update(timestep=rollouts - 1, timesteps=rollouts)

    record_property("configured_learning_epochs", agent.cfg.learning_epochs)
    record_property(
        "deployment_effective_minibatches",
        agent.tracking_data["Optimization / Effective minibatches"][-1],
    )
    record_property(
        "deployment_observed_minibatches",
        agent.tracking_data["Optimization / Observed minibatches"][-1],
    )
    record_property(
        "deployment_kl_early_stop_count",
        agent.tracking_data["Optimization / KL early-stop count"][-1],
    )
    record_property(
        "deployment_initial_replay_max_abs_log_ratio",
        agent.tracking_data["Policy / Initial replay max abs log-ratio"][-1],
    )
    assert agent.tracking_data["Policy / Initial replay max abs log-ratio"][-1] < 2e-5
    assert agent.tracking_data["Optimization / Effective minibatches"][-1] > 0
    assert any(not torch.equal(old, new) for old, new in zip(policy_before, policy.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(value_before, value.parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in value.parameters())
    assert agent._observation_preprocessor.current_count.item() == 1 + rollouts * num_envs
    assert agent._state_preprocessor.current_count.item() == 1 + rollouts * num_envs
    assert agent._value_preprocessor.current_count.item() == 1 + rollouts * num_envs
    for name in (
        "Loss / Policy loss",
        "Loss / Value loss",
        "Policy / Approx KL",
        "Policy / IS ratio (mean)",
        "Optimization / Grad norm",
        "Learning / Learning rate",
    ):
        assert math.isfinite(agent.tracking_data[name][-1]), (name, agent.tracking_data[name][-1])
