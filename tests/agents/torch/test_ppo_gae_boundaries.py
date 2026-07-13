"""Mathematical and episode-boundary contracts shared by PPO and PPO-RNN.

These tests deliberately use an independent, float64 GAE oracle. In particular,
the oracle does not call either production ``compute_gae`` implementation.

Time-limit bootstrap contract
-----------------------------
``rewards`` must already contain ``gamma * V(final_observation)`` at a
truncated transition when time-limit bootstrapping is enabled. ``compute_gae``
must then treat both ``terminated`` and ``truncated`` as episode boundaries so
that rewards/values from the automatically-reset next episode cannot leak
backwards. ``next_values`` is only the bootstrap for environments whose last
rollout transition is not an episode boundary.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.agents.torch.ppo.ppo import compute_gae as ppo_compute_gae
from skrl.agents.torch.ppo.ppo_rnn import compute_gae as ppo_rnn_compute_gae

from tests.agents.torch.test_ppo_update_invariants import _collect_rollout, _make_agent


ComputeGAE = Callable[..., tuple[torch.Tensor, torch.Tensor]]


@pytest.fixture(params=[ppo_compute_gae, ppo_rnn_compute_gae], ids=["ppo", "ppo_rnn"])
def compute_gae(request) -> ComputeGAE:
    return request.param


def _fp64_gae_oracle(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount_factor: float,
    lambda_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (returns, normalized advantages, raw advantages) in float64.

    The implementation first constructs one-step TD residuals and then applies
    the discounted lambda filter. This keeps the reference calculation
    structurally separate from the production reverse-loop expression.
    """

    rewards = rewards.detach().to(dtype=torch.float64)
    values = values.detach().to(dtype=torch.float64)
    next_values = next_values.detach().to(dtype=torch.float64)
    episode_continues = ~(terminated.detach().bool() | truncated.detach().bool())

    following_values = torch.cat((values[1:], next_values.unsqueeze(0)), dim=0)
    td_residuals = (
        rewards
        + float(discount_factor) * episode_continues.to(torch.float64) * following_values
        - values
    )

    raw_advantages = torch.empty_like(td_residuals)
    filtered_residual = torch.zeros_like(td_residuals[0])
    continuation_discount = float(discount_factor) * float(lambda_coefficient)
    for index in range(td_residuals.shape[0] - 1, -1, -1):
        filtered_residual = (
            td_residuals[index]
            + continuation_discount
            * episode_continues[index].to(torch.float64)
            * filtered_residual
        )
        raw_advantages[index] = filtered_residual

    returns = raw_advantages + values
    normalized_advantages = (raw_advantages - raw_advantages.mean()) / (
        raw_advantages.std(unbiased=False) + 1e-8
    )
    return returns, normalized_advantages, raw_advantages


def _call_compute_gae(
    compute_gae: ComputeGAE,
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount_factor: float,
    lambda_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the required public contract explicitly, including truncations."""

    return compute_gae(
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        next_values=next_values,
        discount_factor=discount_factor,
        lambda_coefficient=lambda_coefficient,
    )


def test_gae_matches_independent_fp64_oracle_for_asynchronous_episode_boundaries(compute_gae):
    # Four environments deliberately finish at different positions and by
    # different mechanisms. Timeout rewards below have already been augmented
    # with gamma * V(final_observation), per the module-level contract.
    rewards = torch.tensor(
        [
            [[0.2], [-0.4], [1.3], [2.1]],
            [[1.1], [0.5], [-0.7], [0.3]],
            [[-0.2], [3.2], [0.8], [-1.5]],
            [[2.0], [-0.3], [1.7], [0.6]],
            [[0.4], [1.2], [-2.1], [0.9]],
            [[-0.8], [0.7], [0.1], [1.5]],
        ],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [
            [[0.1], [0.4], [-0.3], [0.8]],
            [[0.6], [-0.2], [0.5], [0.7]],
            [[-0.4], [0.9], [0.2], [-0.1]],
            [[1.0], [0.3], [-0.8], [0.4]],
            [[0.2], [-0.5], [1.1], [0.6]],
            [[-0.7], [0.8], [0.4], [-0.2]],
        ],
        dtype=torch.float64,
    )
    terminated = torch.zeros((6, 4, 1), dtype=torch.bool)
    terminated[1, 0] = True
    terminated[3, 2] = True
    terminated[5, 3] = True
    truncated = torch.zeros_like(terminated)
    truncated[0, 3] = True
    truncated[2, 1] = True
    truncated[4, 0] = True
    next_values = torch.tensor([[0.9], [-0.6], [0.25], [4.0]], dtype=torch.float64)
    gamma, gae_lambda = 0.91, 0.73

    expected_returns, expected_advantages, _ = _fp64_gae_oracle(
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        next_values=next_values,
        discount_factor=gamma,
        lambda_coefficient=gae_lambda,
    )
    actual_returns, actual_advantages = _call_compute_gae(
        compute_gae,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        next_values=next_values,
        discount_factor=gamma,
        lambda_coefficient=gae_lambda,
    )

    assert actual_returns.dtype == torch.float64
    assert actual_advantages.dtype == torch.float64
    torch.testing.assert_close(actual_returns, expected_returns, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual_advantages, expected_advantages, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("boundary_kind", ["terminated", "truncated"])
def test_gae_never_leaks_a_new_episode_reward_across_a_boundary(compute_gae, boundary_kind):
    rewards = torch.tensor([0.0, 0.0, 10.0, 0.0], dtype=torch.float64).view(4, 1, 1)
    values = torch.zeros_like(rewards)
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    (terminated if boundary_kind == "terminated" else truncated)[1] = True

    returns, _ = _call_compute_gae(
        compute_gae,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        next_values=torch.zeros((1, 1), dtype=torch.float64),
        discount_factor=1.0,
        lambda_coefficient=1.0,
    )

    torch.testing.assert_close(
        returns.flatten(),
        torch.tensor([0.0, 0.0, 10.0, 0.0], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("time_limit_bootstrap", [False, True])
def test_time_limit_bootstrap_uses_final_state_value_but_still_cuts_the_trace(
    compute_gae, time_limit_bootstrap
):
    gamma = 0.9
    final_state_value = 7.0
    rewards = torch.tensor([0.0, 0.0, 100.0, 0.0], dtype=torch.float64).view(4, 1, 1)
    if time_limit_bootstrap:
        # This shaping must be performed from V(final_observation), never from
        # V(current_state) or the reset observation of the following episode.
        rewards[1] += gamma * final_state_value

    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    terminated[3] = True
    truncated = torch.zeros_like(terminated)
    truncated[1] = True
    returns, _ = _call_compute_gae(
        compute_gae,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=torch.zeros_like(rewards),
        next_values=torch.tensor([[999.0]], dtype=torch.float64),
        discount_factor=gamma,
        lambda_coefficient=1.0,
    )

    timeout_return = gamma * final_state_value if time_limit_bootstrap else 0.0
    expected = torch.tensor(
        [gamma * timeout_return, timeout_return, 100.0, 0.0], dtype=torch.float64
    )
    torch.testing.assert_close(returns.flatten(), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
def test_agent_time_limit_bootstrap_targets_the_provided_final_next_state(agent_class):
    """Exercise the record/update path, not only the pure GAE helper.

    ``next_states`` is explicitly a non-reset final state in this test. The
    timeout return must use its value; using the current state's value is an
    off-by-one bootstrap error. Environments that auto-reset before returning
    must equivalently pass their final observation/state via the integration
    layer.
    """

    gamma = 0.83
    base_reward = 1.75
    agent = _make_agent(
        agent_class,
        learning_epochs=1,
        learning_rate=0.0,
        rollouts=1,
        mini_batches=1,
        recurrent=False,
    )
    agent.cfg.discount_factor = gamma
    agent.cfg.time_limit_bootstrap = True

    observation = torch.tensor([[1.0, -2.0]])
    state = torch.tensor([[-3.0, 0.5, 2.0]])
    final_next_observation = torch.tensor([[4.0, 1.0]])
    final_next_state = torch.tensor([[8.0, -5.0, 1.0]])
    with torch.no_grad():
        current_value, _ = agent.value.act(
            {
                "observations": agent._observation_preprocessor(observation),
                "states": agent._state_preprocessor(state),
            },
            role="value",
        )
        final_next_value, _ = agent.value.act(
            {
                "observations": agent._observation_preprocessor(final_next_observation),
                "states": agent._state_preprocessor(final_next_state),
            },
            role="value",
        )
        current_value = agent._value_preprocessor(current_value, inverse=True)
        final_next_value = agent._value_preprocessor(final_next_value, inverse=True)
    assert not torch.allclose(current_value, final_next_value, rtol=1e-5, atol=1e-5)

    with torch.no_grad():
        action, _ = agent.act(observation, state, timestep=0, timesteps=1)
        agent.record_transition(
            observations=observation,
            states=state,
            actions=action,
            rewards=torch.tensor([[base_reward]]),
            next_observations=final_next_observation,
            next_states=final_next_state,
            terminated=torch.tensor([[False]]),
            truncated=torch.tensor([[True]]),
            infos={},
            timestep=0,
            timesteps=1,
        )
    agent.enable_models_training_mode(True)
    agent.update(timestep=0, timesteps=1)

    expected_timeout_return = base_reward + gamma * final_next_value.item()
    actual_timeout_return = agent.memory.get_tensor_by_name("returns")[0, 0, 0].item()
    assert actual_timeout_return == pytest.approx(expected_timeout_return, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
def test_agent_time_limit_bootstrap_rejects_reset_observations_from_autoreset_env(agent_class):
    agent = _make_agent(
        agent_class,
        learning_epochs=1,
        learning_rate=0.0,
        rollouts=1,
        mini_batches=1,
        recurrent=False,
    )
    agent.cfg.time_limit_bootstrap = True
    observation = torch.tensor([[1.0, -2.0]])
    state = torch.tensor([[-3.0, 0.5, 2.0]])
    with torch.no_grad():
        action, _ = agent.act(observation, state, timestep=0, timesteps=1)
        with pytest.raises(RuntimeError, match=r"(?i)pre-reset|auto-reset"):
            agent.record_transition(
                observations=observation,
                states=state,
                actions=action,
                rewards=torch.ones((1, 1)),
                next_observations=torch.zeros_like(observation),
                next_states=torch.zeros_like(state),
                terminated=torch.zeros((1, 1), dtype=torch.bool),
                truncated=torch.ones((1, 1), dtype=torch.bool),
                infos={"_skrl_autoreset": True},
                timestep=0,
                timesteps=1,
            )


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
def test_true_termination_is_never_time_limit_bootstrapped_when_both_flags_are_set(agent_class):
    agent = _make_agent(
        agent_class,
        learning_epochs=1,
        learning_rate=0.0,
        rollouts=1,
        mini_batches=1,
        recurrent=False,
    )
    agent.cfg.time_limit_bootstrap = True
    observation = torch.tensor([[1.0, -2.0]])
    state = torch.tensor([[-3.0, 0.5, 2.0]])
    reward = torch.tensor([[1.25]])
    with torch.no_grad():
        action, _ = agent.act(observation, state, timestep=0, timesteps=1)
        agent.record_transition(
            observations=observation,
            states=state,
            actions=action,
            rewards=reward,
            next_observations=torch.full_like(observation, float("inf")),
            next_states=torch.full_like(state, float("inf")),
            terminated=torch.ones((1, 1), dtype=torch.bool),
            truncated=torch.ones((1, 1), dtype=torch.bool),
            infos={"_skrl_autoreset": True},
            timestep=0,
            timesteps=1,
        )
    torch.testing.assert_close(agent.memory.get_tensor_by_name("rewards")[0], reward, rtol=0, atol=0)


@pytest.mark.parametrize("boundary_kind", ["terminated", "truncated"])
def test_rollout_tail_value_is_ignored_when_the_last_transition_ends_an_episode(compute_gae, boundary_kind):
    rewards = torch.tensor([[[3.0]]], dtype=torch.float64)
    values = torch.tensor([[[1.25]]], dtype=torch.float64)
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    (terminated if boundary_kind == "terminated" else truncated)[0] = True

    returns, advantages = _call_compute_gae(
        compute_gae,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        next_values=torch.tensor([[1e9]], dtype=torch.float64),
        discount_factor=0.99,
        lambda_coefficient=0.95,
    )

    torch.testing.assert_close(returns, rewards, rtol=0, atol=0)
    assert torch.equal(advantages, torch.zeros_like(advantages))
    assert torch.isfinite(advantages).all()


@pytest.mark.parametrize(
    ("gamma", "gae_lambda"),
    [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.99, 0.95)],
)
def test_gae_discount_and_lambda_extremes_match_fp64_oracle(compute_gae, gamma, gae_lambda):
    rewards = torch.tensor(
        [[[1.0], [-2.0]], [[0.5], [3.0]], [[-1.0], [0.25]]], dtype=torch.float64
    )
    values = torch.tensor(
        [[[0.2], [0.3]], [[-0.4], [0.6]], [[0.8], [-0.1]]], dtype=torch.float64
    )
    terminated = torch.tensor(
        [[[False], [False]], [[True], [False]], [[False], [False]]], dtype=torch.bool
    )
    truncated = torch.tensor(
        [[[False], [True]], [[False], [False]], [[False], [False]]], dtype=torch.bool
    )
    next_values = torch.tensor([[0.7], [-0.9]], dtype=torch.float64)
    expected_returns, expected_advantages, _ = _fp64_gae_oracle(
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        next_values=next_values,
        discount_factor=gamma,
        lambda_coefficient=gae_lambda,
    )
    actual_returns, actual_advantages = _call_compute_gae(
        compute_gae,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        next_values=next_values,
        discount_factor=gamma,
        lambda_coefficient=gae_lambda,
    )

    torch.testing.assert_close(actual_returns, expected_returns, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual_advantages, expected_advantages, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    ("agent_class", "rollouts", "mini_batches", "recurrent"),
    [(PPO, 5, 2, False), (PPO_RNN, 6, 2, True)],
    ids=["ppo-transition-remainder", "ppo-rnn-sequence-remainder"],
)
def test_each_learning_epoch_replays_every_rollout_sample_exactly_once(
    agent_class, rollouts, mini_batches, recurrent
):
    learning_epochs = 3
    agent = _make_agent(
        agent_class,
        learning_epochs=learning_epochs,
        learning_rate=0.0,
        rollouts=rollouts,
        mini_batches=mini_batches,
        recurrent=recurrent,
    )
    observations, _ = _collect_rollout(agent, sample_count=rollouts)

    replayed_observation_ids: list[torch.Tensor] = []
    original_policy_act = agent.policy.act

    def recording_policy_act(inputs, *, role=""):
        if "taken_actions" in inputs:
            replayed_observation_ids.append(inputs["observations"][:, 0].detach().cpu())
        return original_policy_act(inputs, role=role)

    agent.policy.act = recording_policy_act
    agent.enable_models_training_mode(True)
    agent.update(timestep=rollouts - 1, timesteps=rollouts)

    observed = torch.cat(replayed_observation_ids)
    expected = observations[:, 0].repeat(learning_epochs)
    torch.testing.assert_close(observed.sort().values, expected.sort().values, rtol=0, atol=0)
    for observation_id in observations[:, 0]:
        assert torch.count_nonzero(observed == observation_id).item() == learning_epochs
