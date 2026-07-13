from __future__ import annotations

from dataclasses import dataclass

import gymnasium
import pytest
import torch

from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model


_GAMMA = 0.91
_LAMBDA = 0.83
_VALUE_CLIP = 1.0e-6
_VALUE_LOSS_SCALE = 0.5


class RecordingValueScaler(RunningStandardScaler):
    """Running scaler that exposes which raw populations train its statistics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats_inputs: list[torch.Tensor] = []

    def update_stats(self, x):
        if x is not None:
            self.stats_inputs.append(x.detach().clone())
        return super().update_stats(x)


class RecordingRandomMemory(RandomMemory):
    """Memory that records the exact replay order selected by PPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_sample_indexes: list[torch.Tensor] = []

    def sample_by_index(self, names, *, indexes, mini_batches=1):
        if "values" in names:
            self.value_sample_indexes.append(torch.as_tensor(indexes).detach().cpu().clone())
        return super().sample_by_index(names, indexes=indexes, mini_batches=mini_batches)


@dataclass(frozen=True)
class _Stats:
    mean: torch.Tensor
    variance: torch.Tensor
    count: float


@dataclass(frozen=True)
class _Batch:
    observations: torch.Tensor
    states: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    next_observation: torch.Tensor
    next_state: torch.Tensor


def _snapshot(scaler: RunningStandardScaler) -> _Stats:
    return _Stats(
        mean=scaler.running_mean.detach().double().clone(),
        variance=scaler.running_variance.detach().double().clone(),
        count=float(scaler.current_count.item()),
    )


def _oracle_normalize(x: torch.Tensor, stats: _Stats, *, epsilon: float, clip: float) -> torch.Tensor:
    """Independent normalization formula (does not call the production scaler)."""

    output = (x.double() - stats.mean) / (stats.variance.sqrt() + epsilon)
    return output.clamp(min=-clip, max=clip).to(dtype=x.dtype)


def _oracle_inverse(x: torch.Tensor, stats: _Stats, *, clip: float) -> torch.Tensor:
    """Independent inverse-normalization formula (does not call the production scaler)."""

    output = stats.variance.sqrt() * x.double().clamp(min=-clip, max=clip) + stats.mean
    return output.to(dtype=x.dtype)


def _oracle_merge(stats: _Stats, population: torch.Tensor) -> _Stats:
    """Parallel population-variance merge, independently mirroring the defining equations."""

    population = population.detach().double().reshape(-1, population.shape[-1])
    population_count = population.shape[0]
    population_mean = population.mean(dim=0)
    population_variance = population.var(dim=0, unbiased=False)

    total_count = stats.count + population_count
    delta = population_mean - stats.mean
    merged_mean = stats.mean + delta * population_count / total_count
    merged_m2 = (
        stats.variance * stats.count
        + population_variance * population_count
        + delta.square() * stats.count * population_count / total_count
    )
    return _Stats(mean=merged_mean, variance=merged_m2 / total_count, count=total_count)


def _oracle_returns(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
) -> torch.Tensor:
    """Small fp64 GAE oracle deliberately independent of either PPO module's compute_gae."""

    rewards = rewards.double()
    terminated = terminated.bool()
    values = values.double()
    next_value = next_value.double()
    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros_like(next_value)

    for index in range(rewards.shape[0] - 1, -1, -1):
        bootstrap_value = values[index + 1] if index + 1 < rewards.shape[0] else next_value
        continuation = (~terminated[index]).to(dtype=torch.float64)
        delta = rewards[index] + _GAMMA * continuation * bootstrap_value - values[index]
        running_advantage = delta + _GAMMA * _LAMBDA * continuation * running_advantage
        advantages[index] = running_advantage
    return (advantages + values).float()


def _make_batch(kind: str) -> _Batch:
    if kind == "single":
        observations = torch.tensor([[0.25, -0.50]])
        states = torch.tensor([[-0.75, 0.50, 1.25]])
        rewards = torch.tensor([[2.50]])
        terminated = torch.tensor([[True]])
    elif kind == "constant":
        observations = torch.tensor([[0.40, -0.20]]).repeat(4, 1)
        states = torch.tensor([[-0.60, 0.30, 1.10]]).repeat(4, 1)
        rewards = torch.full((4, 1), 3.25)
        # Every transition is a complete one-step episode, so the oracle return is
        # exactly the same constant regardless of the critic prediction.
        terminated = torch.ones((4, 1), dtype=torch.bool)
    elif kind == "varied":
        observations = torch.tensor(
            [[-0.80, 0.10], [-0.20, 0.70], [0.35, -0.45], [0.90, 0.25]]
        )
        states = torch.tensor(
            [[-1.20, 0.10, 0.50], [-0.40, 0.80, -0.20], [0.30, -0.60, 1.40], [1.10, 0.20, -0.70]]
        )
        rewards = torch.tensor([[-0.75], [1.50], [0.25], [2.75]])
        terminated = torch.tensor([[False], [True], [False], [False]])
    else:  # pragma: no cover - test author error
        raise ValueError(f"Unknown batch kind: {kind}")

    return _Batch(
        observations=observations,
        states=states,
        rewards=rewards,
        terminated=terminated,
        next_observation=observations[-1:] + torch.tensor([[0.15, -0.05]]),
        next_state=states[-1:] + torch.tensor([[0.20, -0.10, 0.05]]),
    )


def _make_agent(agent_class, *, rollouts: int, learning_epochs: int = 1):
    torch.manual_seed(731)
    observation_space = gymnasium.spaces.Box(low=-10, high=10, shape=(2,))
    state_space = gymnasium.spaces.Box(low=-10, high=10, shape=(3,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))

    policy = gaussian_model(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [5], "activations": "tanh"}],
        output="ACTIONS",
    )
    value = deterministic_model(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        network=[{"name": "net", "input": "STATES", "layers": [5], "activations": "tanh"}],
        output="ONE",
    )
    memory = RecordingRandomMemory(memory_size=rollouts, num_envs=1, device="cpu")
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": 1,
        "discount_factor": _GAMMA,
        "lambda_": _LAMBDA,
        "learning_rate": 0.0,
        "observation_preprocessor": None,
        "state_preprocessor": None,
        "value_preprocessor": RecordingValueScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": "cpu", "clip_threshold": 100.0},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 0.0,
        "ratio_clip": 0.2,
        "value_clip": _VALUE_CLIP,
        "entropy_loss_scale": 0.0,
        "value_loss_scale": _VALUE_LOSS_SCALE,
        "kl_threshold": 0.0,
        "time_limit_bootstrap": False,
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
    agent = agent_class(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()

    # Start from a deliberately non-identity value scale. This makes an
    # accidental statistics update before clipping observably change domains.
    agent._value_preprocessor.update_stats(torch.tensor([[-6.0], [2.0], [11.0]]))
    agent._value_preprocessor.stats_inputs.clear()
    return agent


def _model_value(agent, state: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output, _ = agent.value.act({"observations": state[:, :2], "states": state}, role="value")
    return output.detach().clone()


def _collect_fixed_batch(agent, batch: _Batch) -> tuple[torch.Tensor, torch.Tensor]:
    rollout_outputs = []
    for timestep in range(batch.rewards.shape[0]):
        observations = batch.observations[timestep : timestep + 1]
        states = batch.states[timestep : timestep + 1]
        rollout_outputs.append(_model_value(agent, states))
        with torch.no_grad():
            actions, _ = agent.act(
                observations,
                states,
                timestep=timestep,
                timesteps=batch.rewards.shape[0],
            )
            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=batch.rewards[timestep : timestep + 1],
                next_observations=(
                    batch.observations[timestep + 1 : timestep + 2]
                    if timestep + 1 < batch.rewards.shape[0]
                    else batch.next_observation
                ),
                next_states=(
                    batch.states[timestep + 1 : timestep + 2]
                    if timestep + 1 < batch.rewards.shape[0]
                    else batch.next_state
                ),
                terminated=batch.terminated[timestep : timestep + 1],
                truncated=torch.zeros((1, 1), dtype=torch.bool),
                infos={},
                timestep=timestep,
                timesteps=batch.rewards.shape[0],
            )
    return torch.cat(rollout_outputs), _model_value(agent, batch.next_state)


def _run_fixed_update(agent, batch: _Batch, stats_before: _Stats) -> _Stats:
    scaler = agent._value_preprocessor
    scaler.stats_inputs.clear()
    standardized_values, standardized_next_value = _collect_fixed_batch(agent, batch)

    raw_values = _oracle_inverse(standardized_values, stats_before, clip=scaler.clip_threshold)
    raw_next_value = _oracle_inverse(standardized_next_value, stats_before, clip=scaler.clip_threshold)
    raw_returns = _oracle_returns(
        rewards=batch.rewards,
        terminated=batch.terminated,
        values=raw_values,
        next_value=raw_next_value,
    )
    expected_old_values = _oracle_normalize(
        raw_values,
        stats_before,
        epsilon=scaler.epsilon,
        clip=scaler.clip_threshold,
    )
    expected_returns = _oracle_normalize(
        raw_returns,
        stats_before,
        epsilon=scaler.epsilon,
        clip=scaler.clip_threshold,
    )
    expected_value_loss = _VALUE_LOSS_SCALE * torch.mean((expected_returns - standardized_values).square())

    # Verify the rollout itself used the frozen inverse scale assumed by the oracle.
    torch.testing.assert_close(
        agent.memory.get_tensor_by_name("values").flatten(0, 1), raw_values, rtol=2e-6, atol=2e-6
    )

    parameters_before = [parameter.detach().clone() for parameter in agent.value.parameters()]
    replay_outputs = []
    original_value_act = agent.value.act
    agent.memory.value_sample_indexes.clear()

    def recording_value_act(inputs, *, role=""):
        output, extra = original_value_act(inputs, role=role)
        # update() evaluates the bootstrap in eval mode and every optimization
        # replay in training mode, giving an implementation-independent split.
        if role == "value" and agent.value.training:
            replay_outputs.append(output.detach().clone())
        return output, extra

    agent.value.act = recording_value_act
    agent.enable_models_training_mode(True)
    try:
        agent.update(timestep=batch.rewards.shape[0] - 1, timesteps=batch.rewards.shape[0])
    finally:
        agent.value.act = original_value_act

    # lr=0 is the counterfactual required by value clipping: the exact same
    # critic must replay the exact same old values, in the exact same domain.
    assert replay_outputs
    assert len(replay_outputs) == len(agent.memory.value_sample_indexes)
    stored_values = agent.memory.get_tensor_by_name("values").flatten(0, 1)
    stored_returns = agent.memory.get_tensor_by_name("returns").flatten(0, 1)
    for replay_output, indexes in zip(replay_outputs, agent.memory.value_sample_indexes):
        torch.testing.assert_close(replay_output, standardized_values[indexes], rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(
            replay_output - stored_values[indexes],
            torch.zeros_like(replay_output),
            rtol=0,
            atol=2e-6,
        )
    torch.testing.assert_close(stored_values, expected_old_values, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(stored_returns, expected_returns, rtol=2e-6, atol=2e-6)

    assert agent.tracking_data["Value / Clip fraction"][-1] == 0.0
    assert agent.tracking_data["Loss / Value loss"][-1] == pytest.approx(
        expected_value_loss.item(), rel=2e-5, abs=2e-6
    )
    assert all(
        torch.equal(parameter_before, parameter_after)
        for parameter_before, parameter_after in zip(parameters_before, agent.value.parameters())
    )

    # A rollout's return population must train the normalizer exactly once and
    # only after all epochs that consume the frozen snapshot have completed.
    assert len(scaler.stats_inputs) == 1
    torch.testing.assert_close(scaler.stats_inputs[0].reshape_as(raw_returns), raw_returns, rtol=2e-6, atol=2e-6)
    expected_stats = _oracle_merge(stats_before, raw_returns)
    stats_after = _snapshot(scaler)
    torch.testing.assert_close(stats_after.mean, expected_stats.mean, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(stats_after.variance, expected_stats.variance, rtol=1e-6, atol=1e-6)
    assert stats_after.count == expected_stats.count
    return expected_stats


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN])
@pytest.mark.parametrize("kind", ["single", "constant", "varied"])
def test_value_scaler_uses_one_frozen_domain_for_a_fixed_batch(agent_class, kind):
    batch = _make_batch(kind)
    agent = _make_agent(agent_class, rollouts=batch.rewards.shape[0])

    _run_fixed_update(agent, batch, _snapshot(agent._value_preprocessor))


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN])
def test_value_scaler_stays_consistent_across_multiple_zero_lr_updates(agent_class):
    agent = _make_agent(agent_class, rollouts=4, learning_epochs=3)
    expected_stats = _snapshot(agent._value_preprocessor)

    # The middle round has zero target variance and the surrounding rounds use
    # different value/return populations. Reusing one unchanged critic across
    # all three snapshots catches both stale-scale replay and double counting.
    for kind in ("varied", "constant", "varied"):
        expected_stats = _run_fixed_update(agent, _make_batch(kind), expected_stats)
