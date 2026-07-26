"""Fixed-batch numerical oracle and fail-fast contracts for PPO/PPO-RNN.

The loss oracle below intentionally does not call the production PPO loss,
``compute_gae``, model mixin log-probability, or entropy helpers. It evaluates
the scalar Gaussian and clipped PPO equations directly in float64, then uses
autograd only to differentiate those independent equations.
"""

from __future__ import annotations

import math

import gymnasium
import pytest
import torch

from skrl import config
from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.agents.torch.ppo._utils import (
    compute_value_loss_fp32,
    require_finite,
    validate_scalar_output,
)
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


ROLLOUTS = 4
LEARNING_RATE = 0.05
GAMMA = 0.9
GAE_LAMBDA = 0.8
RATIO_CLIP = 0.2
VALUE_CLIP = 0.12
ENTROPY_SCALE = 0.07
VALUE_SCALE = 0.6


class LinearGaussianPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        GaussianMixin.__init__(self, clip_log_std=False, reduction="sum")
        self.weight = torch.nn.Parameter(torch.tensor([0.35, -0.2]))
        self.bias = torch.nn.Parameter(torch.tensor([0.1]))
        self.log_std = torch.nn.Parameter(torch.tensor([-0.35]))

    def compute(self, inputs, role=""):
        mean = (inputs["observations"] * self.weight).sum(dim=-1, keepdim=True) + self.bias
        return mean, {"log_std": self.log_std.expand_as(mean)}


class LinearValue(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor([0.25, -0.15]))
        self.bias = torch.nn.Parameter(torch.tensor([0.05]))

    def compute(self, inputs, role=""):
        value = (inputs["states"] * self.weight).sum(dim=-1, keepdim=True) + self.bias
        return value, {}


def _spaces():
    observation_space = gymnasium.spaces.Box(low=-100, high=100, shape=(2,))
    state_space = gymnasium.spaces.Box(low=-100, high=100, shape=(2,))
    action_space = gymnasium.spaces.Box(low=-100, high=100, shape=(1,))
    return observation_space, state_space, action_space


def _base_cfg(*, rollouts=ROLLOUTS, **overrides):
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": 1,
        "mini_batches": 1,
        "discount_factor": GAMMA,
        "lambda_": GAE_LAMBDA,
        "learning_rate": LEARNING_RATE,
        "optimizer": torch.optim.SGD,
        "optimizer_kwargs": {},
        "learning_rate_scheduler": None,
        "learning_rate_scheduler_kwargs": {},
        "observation_preprocessor": None,
        "observation_preprocessor_kwargs": {},
        "state_preprocessor": None,
        "state_preprocessor_kwargs": {},
        "value_preprocessor": None,
        "value_preprocessor_kwargs": {},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 0.0,
        "ratio_clip": RATIO_CLIP,
        "value_clip": VALUE_CLIP,
        "entropy_loss_scale": ENTROPY_SCALE,
        "value_loss_scale": VALUE_SCALE,
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
    cfg.update(overrides)
    return cfg


def _make_agent(
    agent_class,
    *,
    cfg_overrides: dict | None = None,
    memory_size: int = ROLLOUTS,
    initialize: bool = True,
    models: dict | None = None,
):
    observation_space, state_space, action_space = _spaces()
    if models is None:
        models = {
            "policy": LinearGaussianPolicy(observation_space, state_space, action_space),
            "value": LinearValue(observation_space, state_space, action_space),
        }
    cfg = _base_cfg(**(cfg_overrides or {}))
    memory = RandomMemory(memory_size=memory_size, num_envs=1, device="cpu")
    agent = agent_class(
        models=models,
        memory=memory,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    if initialize:
        agent.init()
    return agent


def _normal_log_prob(mean: torch.Tensor, log_std: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Scalar Normal log-density, written without torch.distributions/skrl mixins."""

    return -0.5 * ((actions - mean) * torch.exp(-log_std)).square() - log_std - 0.5 * math.log(
        2 * math.pi
    )


def _fixed_rollout(agent):
    observations = torch.tensor(
        [[1.0, -0.5], [-0.3, 0.8], [0.7, 1.2], [-1.0, 0.4]], dtype=torch.float32
    )
    states = torch.tensor(
        [[0.2, 1.0], [-1.0, 0.5], [0.6, -0.4], [1.2, 0.3]], dtype=torch.float32
    )
    actions = torch.tensor([[0.2], [0.4], [-0.3], [0.1]], dtype=torch.float32)
    rewards = torch.tensor([[0.7], [-0.2], [1.1], [0.3]], dtype=torch.float32)
    old_values = torch.tensor([[0.15], [-0.25], [0.4], [-0.1]], dtype=torch.float32)
    terminated = torch.tensor([[False], [False], [False], [True]])
    truncated = torch.zeros_like(terminated)

    # Construct valid finite behavior log-probabilities that force ratios on
    # both sides of both clipping limits under the current policy.
    with torch.no_grad():
        current_mean = (observations * agent.policy.weight).sum(dim=-1, keepdim=True) + agent.policy.bias
        current_log_prob = _normal_log_prob(current_mean, agent.policy.log_std, actions)
        target_ratios = torch.tensor([[1.5], [0.6], [1.1], [0.6]])
        old_log_prob = current_log_prob - torch.log(target_ratios)

    agent.memory.add_samples(
        observations=observations,
        states=states,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        log_prob=old_log_prob,
        values=old_values,
    )
    agent._current_next_observations = torch.tensor([[0.25, -0.75]], dtype=torch.float32)
    agent._current_next_states = torch.tensor([[0.5, 0.25]], dtype=torch.float32)
    return {
        "observations": observations,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "old_values": old_values,
        "terminated": terminated,
        "truncated": truncated,
        "old_log_prob": old_log_prob,
    }


def _reference_gae(batch, tail_value: torch.Tensor):
    """Independent TD-residual/filter form of GAE, evaluated in float64."""

    rewards = batch["rewards"].double()
    old_values = batch["old_values"].double()
    continues = ~(batch["terminated"] | batch["truncated"])
    following_values = torch.cat((old_values[1:], tail_value.detach().double().reshape(1, 1)))
    residuals = rewards + GAMMA * continues.double() * following_values - old_values

    raw_advantages = torch.empty_like(residuals)
    accumulator = torch.zeros_like(residuals[0])
    for index in range(ROLLOUTS - 1, -1, -1):
        accumulator = residuals[index] + GAMMA * GAE_LAMBDA * continues[index].double() * accumulator
        raw_advantages[index] = accumulator

    returns = raw_advantages + old_values
    advantages = (raw_advantages - raw_advantages.mean()) / (
        raw_advantages.std(unbiased=False) + 1e-8
    )
    # PPO stores these tensors in a float32 rollout memory before optimizing.
    return returns.float().double(), advantages.float().double()


def _independent_loss_and_gradient(agent, batch):
    observations = batch["observations"].double()
    states = batch["states"].double()
    actions = batch["actions"].double()
    old_log_prob = batch["old_log_prob"].double()
    old_values = batch["old_values"].double()

    policy_weight = agent.policy.weight.detach().double().clone().requires_grad_(True)
    policy_bias = agent.policy.bias.detach().double().clone().requires_grad_(True)
    log_std = agent.policy.log_std.detach().double().clone().requires_grad_(True)
    value_weight = agent.value.weight.detach().double().clone().requires_grad_(True)
    value_bias = agent.value.bias.detach().double().clone().requires_grad_(True)

    tail_value = (
        agent._current_next_states.double() * value_weight
    ).sum(dim=-1, keepdim=True) + value_bias
    returns, advantages = _reference_gae(batch, tail_value)

    mean = (observations * policy_weight).sum(dim=-1, keepdim=True) + policy_bias
    new_log_prob = _normal_log_prob(mean, log_std, actions)
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1 - RATIO_CLIP, 1 + RATIO_CLIP)
    policy_loss = -torch.minimum(advantages * ratio, advantages * clipped_ratio).mean()

    entropy = (log_std + 0.5 * (1 + math.log(2 * math.pi))).mean()
    entropy_loss = -ENTROPY_SCALE * entropy

    raw_predictions = (states * value_weight).sum(dim=-1, keepdim=True) + value_bias
    value_delta = raw_predictions - old_values
    clipped_predictions = old_values + torch.clamp(value_delta, -VALUE_CLIP, VALUE_CLIP)
    value_errors_unclipped = (returns - raw_predictions).square()
    value_errors_clipped = (returns - clipped_predictions).square()
    # PPO's value clipping is a pessimistic objective, exactly like its policy
    # clipping: clipping must never make a bad new prediction look artificially
    # better. Taking only the clipped-prediction MSE violates that contract.
    value_loss = VALUE_SCALE * torch.maximum(value_errors_unclipped, value_errors_clipped).mean()

    total_loss = policy_loss + entropy_loss + value_loss
    parameters = [policy_weight, policy_bias, log_std, value_weight, value_bias]
    total_loss.backward()
    gradients = {
        "policy.weight": policy_weight.grad.detach(),
        "policy.bias": policy_bias.grad.detach(),
        "policy.log_std": log_std.grad.detach(),
        "value.weight": value_weight.grad.detach(),
        "value.bias": value_bias.grad.detach(),
    }
    grad_norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients.values()))

    return {
        "returns": returns,
        "advantages": advantages,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "entropy_loss": entropy_loss.detach(),
        "approx_kl": ((ratio - 1) - log_ratio).mean().detach(),
        "ratio": ratio.detach(),
        "policy_clip_fraction": (torch.abs(ratio - 1) > RATIO_CLIP).double().mean().detach(),
        "value_clip_fraction": (torch.abs(value_delta) > VALUE_CLIP).double().mean().detach(),
        "value_errors_unclipped": value_errors_unclipped.detach(),
        "value_errors_clipped": value_errors_clipped.detach(),
        "gradients": gradients,
        "grad_norm": grad_norm.detach(),
        "parameters": parameters,
    }


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn_feedforward"])
def test_fixed_batch_losses_gradients_and_one_sgd_step_match_independent_oracle(agent_class):
    agent = _make_agent(agent_class)
    batch = _fixed_rollout(agent)
    expected = _independent_loss_and_gradient(agent, batch)
    before = {
        **{f"policy.{name}": value.detach().clone() for name, value in agent.policy.named_parameters()},
        **{f"value.{name}": value.detach().clone() for name, value in agent.value.named_parameters()},
    }

    # Sanity-check that this fixture genuinely exercises both clipping paths.
    assert expected["policy_clip_fraction"].item() == pytest.approx(0.75, abs=1e-6)
    assert expected["value_clip_fraction"].item() == pytest.approx(0.75, abs=1e-6)
    assert torch.any(expected["value_errors_unclipped"] > expected["value_errors_clipped"])
    assert torch.any(expected["value_errors_clipped"] > expected["value_errors_unclipped"])

    agent.enable_models_training_mode(True)
    agent.update(timestep=ROLLOUTS - 1, timesteps=ROLLOUTS)

    torch.testing.assert_close(
        agent.memory.get_tensor_by_name("returns").flatten(0, 1).double(),
        expected["returns"],
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        agent.memory.get_tensor_by_name("advantages").flatten(0, 1).double(),
        expected["advantages"],
        rtol=2e-6,
        atol=2e-6,
    )

    tracked_expectations = {
        "Loss / Policy loss": expected["policy_loss"],
        "Loss / Value loss": expected["value_loss"],
        "Loss / Entropy loss": expected["entropy_loss"],
        "Policy / Approx KL": expected["approx_kl"],
        "Policy / Entropy": expected["entropy"],
        "Policy / Clip fraction": expected["policy_clip_fraction"],
        "Value / Clip fraction": expected["value_clip_fraction"],
        "Optimization / Grad norm": expected["grad_norm"],
    }
    for key, expected_value in tracked_expectations.items():
        assert agent.tracking_data[key][-1] == pytest.approx(
            expected_value.item(), rel=3e-5, abs=3e-6
        ), key

    after = {
        **{f"policy.{name}": value.detach().clone() for name, value in agent.policy.named_parameters()},
        **{f"value.{name}": value.detach().clone() for name, value in agent.value.named_parameters()},
    }
    actual_gradients = {
        **{f"policy.{name}": value.grad.detach().clone() for name, value in agent.policy.named_parameters()},
        **{f"value.{name}": value.grad.detach().clone() for name, value in agent.value.named_parameters()},
    }
    for name, expected_gradient in expected["gradients"].items():
        torch.testing.assert_close(
            actual_gradients[name].double(), expected_gradient, rtol=3e-5, atol=3e-6
        )
        expected_delta = -LEARNING_RATE * expected_gradient
        actual_delta = after[name].double() - before[name].double()
        torch.testing.assert_close(actual_delta, expected_delta, rtol=4e-5, atol=4e-6)


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn_feedforward"])
def test_separate_policy_and_value_learning_rates_control_their_own_sgd_deltas(agent_class):
    policy_lr, value_lr = 0.03, 0.08
    agent = _make_agent(agent_class, cfg_overrides={"learning_rate": (policy_lr, value_lr)})
    batch = _fixed_rollout(agent)
    expected = _independent_loss_and_gradient(agent, batch)
    before = {
        **{f"policy.{name}": value.detach().clone() for name, value in agent.policy.named_parameters()},
        **{f"value.{name}": value.detach().clone() for name, value in agent.value.named_parameters()},
    }

    assert [group["lr"] for group in agent.optimizer.param_groups] == [policy_lr, value_lr]
    agent.enable_models_training_mode(True)
    agent.update(timestep=ROLLOUTS - 1, timesteps=ROLLOUTS)
    assert agent.tracking_data["Learning / Policy learning rate"][-1] == pytest.approx(policy_lr)
    assert agent.tracking_data["Learning / Value learning rate"][-1] == pytest.approx(value_lr)

    after = {
        **{f"policy.{name}": value.detach() for name, value in agent.policy.named_parameters()},
        **{f"value.{name}": value.detach() for name, value in agent.value.named_parameters()},
    }
    for name, expected_gradient in expected["gradients"].items():
        learning_rate = policy_lr if name.startswith("policy.") else value_lr
        expected_delta = -learning_rate * expected_gradient
        actual_delta = after[name].double() - before[name].double()
        torch.testing.assert_close(actual_delta, expected_delta, rtol=4e-5, atol=4e-6)


def test_value_loss_is_promoted_before_squaring_large_half_precision_residuals():
    predicted = torch.tensor([[300.0]], dtype=torch.float16, requires_grad=True)
    loss, unscaled_loss, clip_fraction = compute_value_loss_fp32(
        predicted_values=predicted,
        sampled_values=torch.zeros_like(predicted),
        sampled_returns=torch.zeros_like(predicted),
        value_clip=0.0,
        value_loss_scale=1.0,
    )

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert unscaled_loss.item() == pytest.approx(90_000.0)
    assert clip_fraction.item() == 0.0
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn_feedforward"])
def test_critic_guard_freezes_value_continues_policy_and_backs_off_value_lr(agent_class):
    policy_lr, value_lr = 0.03, 0.08
    agent = _make_agent(
        agent_class,
        cfg_overrides={
            "learning_rate": (policy_lr, value_lr),
            "learning_epochs": 2,
            "mini_batches": 2,
            "value_loss_guard": 1e-12,
            "value_lr_backoff_factor": 0.5,
            "value_lr_backoff_min": 1e-4,
        },
    )
    _fixed_rollout(agent)
    policy_before = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    value_before = [parameter.detach().clone() for parameter in agent.value.parameters()]

    agent.enable_models_training_mode(True)
    agent.update(timestep=ROLLOUTS - 1, timesteps=ROLLOUTS)

    assert any(
        not torch.equal(before, after)
        for before, after in zip(policy_before, agent.policy.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(value_before, agent.value.parameters())
    )
    assert agent.tracking_data["Optimization / Critic guard activations"][-1] == 1
    assert agent.tracking_data["Optimization / Critic skipped minibatches"][-1] == 4
    assert agent.tracking_data["Value / Maximum unscaled loss"][-1] > 0
    assert [group["lr"] for group in agent.optimizer.param_groups] == pytest.approx(
        [policy_lr, value_lr * 0.5]
    )
    assert agent.tracking_data["Learning / Value LR backoff ratio"][-1] == pytest.approx(0.5)


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn_feedforward"])
@pytest.mark.parametrize("grad_norm_clip", [0.0, 0.5], ids=["measured", "clipped"])
def test_nonfinite_gradient_fails_before_optimizer_can_corrupt_parameters(agent_class, grad_norm_clip):
    agent = _make_agent(agent_class, cfg_overrides={"grad_norm_clip": grad_norm_clip})
    _fixed_rollout(agent)
    before = {
        name: parameter.detach().clone()
        for model_name, model in (("policy", agent.policy), ("value", agent.value))
        for name, parameter in model.named_parameters(prefix=model_name)
    }
    hook = agent.policy.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    try:
        agent.enable_models_training_mode(True)
        with pytest.raises(FloatingPointError, match="gradient norm"):
            agent.update(timestep=ROLLOUTS - 1, timesteps=ROLLOUTS)
    finally:
        hook.remove()

    after = {
        name: parameter.detach()
        for model_name, model in (("policy", agent.policy), ("value", agent.value))
        for name, parameter in model.named_parameters(prefix=model_name)
    }
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn_feedforward"])
@pytest.mark.parametrize("malformed_output", ["log_prob", "value"])
def test_replay_scalar_output_shape_fails_before_broadcast_or_optimizer_step(
    agent_class, malformed_output, monkeypatch
):
    agent = _make_agent(agent_class)
    _fixed_rollout(agent)
    before = [parameter.detach().clone() for model in (agent.policy, agent.value) for parameter in model.parameters()]

    if malformed_output == "log_prob":
        original_act = agent.policy.act

        def malformed_policy_act(inputs, role=""):
            actions, outputs = original_act(inputs, role=role)
            if "taken_actions" in inputs:
                outputs = dict(outputs)
                outputs["log_prob"] = outputs["log_prob"].squeeze(-1)
            return actions, outputs

        monkeypatch.setattr(agent.policy, "act", malformed_policy_act)
    else:
        original_act = agent.value.act

        def malformed_value_act(inputs, role=""):
            values, outputs = original_act(inputs, role=role)
            if agent.value.training:
                values = values.squeeze(-1)
            return values, outputs

        monkeypatch.setattr(agent.value, "act", malformed_value_act)

    agent.enable_models_training_mode(True)
    with pytest.raises(ValueError, match="shape"):
        agent.update(timestep=ROLLOUTS - 1, timesteps=ROLLOUTS)

    after = [parameter.detach() for model in (agent.policy, agent.value) for parameter in model.parameters()]
    for old, new in zip(before, after):
        torch.testing.assert_close(new, old, rtol=0, atol=0)


def test_distributed_finite_check_propagates_remote_failure_and_local_mode_skips_collective(monkeypatch):
    monkeypatch.setattr(config.torch, "_is_distributed", True)
    calls = []

    def remote_failure(tensor, op):
        calls.append(op)
        tensor.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", remote_failure)
    with pytest.raises(FloatingPointError, match="another distributed rank"):
        require_finite("synchronized value", torch.ones(1), synchronize=True)
    assert calls == [torch.distributed.ReduceOp.MAX]

    # Callers can still request a strictly local validation when they have not first
    # synchronized their control flow.
    calls.clear()
    validate_scalar_output("conditional timeout value", torch.ones((1, 1)), 1)
    assert calls == []


def _override_id(overrides):
    field, value = next(iter(overrides.items()))
    return f"{field}={value!r}"


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
@pytest.mark.parametrize(
    "overrides",
    [
        {"rollouts": 0},
        {"rollouts": -1},
        {"rollouts": True},
        {"rollouts": 1.5},
        {"learning_epochs": 0},
        {"mini_batches": 0},
        {"random_timesteps": -1},
        {"random_timesteps": True},
        {"random_timesteps": 1.5},
        {"learning_starts": -1},
        {"learning_starts": True},
        {"learning_starts": 1.5},
        {"discount_factor": -0.01},
        {"discount_factor": 1.01},
        {"discount_factor": float("nan")},
        {"lambda_": -0.01},
        {"lambda_": 1.01},
        {"lambda_": float("nan")},
        {"learning_rate": -1e-3},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"learning_rate": ()},
        {"learning_rate": (1e-3,)},
        {"learning_rate": (1e-3, 2e-3, 3e-3)},
        {"ratio_clip": -0.1},
        {"ratio_clip": float("nan")},
        {"ratio_clip": float("inf")},
        {"value_clip": float("nan")},
        {"value_mixed_precision": True},
        {"value_loss_guard": -1.0},
        {"value_prediction_guard": -1.0},
        {"value_lr_backoff_factor": 0.0},
        {"value_lr_backoff_factor": 1.01},
        {"value_lr_backoff_min": -1.0},
        {"value_clip": float("inf")},
        {"kl_threshold": float("nan")},
        {"kl_threshold": float("inf")},
        {"grad_norm_clip": float("nan")},
        {"grad_norm_clip": float("inf")},
        {"entropy_loss_scale": -0.1},
        {"entropy_loss_scale": float("nan")},
        {"entropy_loss_scale": float("inf")},
        {"value_loss_scale": -0.1},
        {"value_loss_scale": float("nan")},
        {"value_loss_scale": float("inf")},
    ],
    ids=_override_id,
)
def test_invalid_ppo_configuration_fails_during_construction(agent_class, overrides):
    # Keep memory construction valid even when the deliberately-invalid rollout
    # count cannot represent a buffer size.
    memory_size = overrides.get("rollouts", ROLLOUTS)
    if not isinstance(memory_size, int) or isinstance(memory_size, bool) or memory_size < 1:
        memory_size = ROLLOUTS
    with pytest.raises(ValueError):
        _make_agent(
            agent_class,
            cfg_overrides=overrides,
            memory_size=memory_size,
            initialize=False,
        )


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
def test_negative_value_clip_preserves_documented_disable_clipping_contract(agent_class):
    agent = _make_agent(agent_class, cfg_overrides={"value_clip": -1.0})
    assert agent.cfg.value_clip == -1.0


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
def test_rollout_configuration_must_match_memory_capacity(agent_class):
    with pytest.raises(ValueError, match="memory size|rollout length"):
        _make_agent(
            agent_class,
            cfg_overrides={"rollouts": ROLLOUTS},
            memory_size=ROLLOUTS + 1,
            initialize=False,
        )


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
@pytest.mark.parametrize("missing_role", ["policy", "value"])
def test_policy_and_value_models_are_both_required(agent_class, missing_role):
    observation_space, state_space, action_space = _spaces()
    models = {
        "policy": LinearGaussianPolicy(observation_space, state_space, action_space),
        "value": LinearValue(observation_space, state_space, action_space),
    }
    del models[missing_role]
    with pytest.raises(ValueError, match="policy|value"):
        _make_agent(agent_class, models=models, initialize=False)


def _add_partial_rollout(agent, sample_count: int):
    # Add one environment step at a time, matching the normal collection path.
    # Bulk insertion of fewer samples than capacity has distinct circular-buffer
    # semantics and is not the behavior under test here.
    for _ in range(sample_count):
        agent.memory.add_samples(
            observations=torch.zeros((1, 2)),
            states=torch.zeros((1, 2)),
            actions=torch.zeros((1, 1)),
            rewards=torch.zeros((1, 1)),
            terminated=torch.zeros((1, 1), dtype=torch.bool),
            truncated=torch.zeros((1, 1), dtype=torch.bool),
            log_prob=torch.zeros((1, 1)),
            values=torch.zeros((1, 1)),
        )
    agent._current_next_observations = torch.zeros((1, 2))
    agent._current_next_states = torch.zeros((1, 2))


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
@pytest.mark.parametrize("sample_count", [0, 1, ROLLOUTS - 1])
def test_update_rejects_an_incomplete_rollout_before_changing_parameters(agent_class, sample_count):
    agent = _make_agent(agent_class)
    _add_partial_rollout(agent, sample_count)
    before = [parameter.detach().clone() for model in (agent.policy, agent.value) for parameter in model.parameters()]

    with pytest.raises(RuntimeError, match="full rollout"):
        agent.update(timestep=sample_count, timesteps=ROLLOUTS)

    after = [parameter.detach() for model in (agent.policy, agent.value) for parameter in model.parameters()]
    assert all(torch.equal(old, new) for old, new in zip(before, after))


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
def test_update_without_rollout_memory_fails_explicitly(agent_class):
    observation_space, state_space, action_space = _spaces()
    policy = LinearGaussianPolicy(observation_space, state_space, action_space)
    value = LinearValue(observation_space, state_space, action_space)
    agent = agent_class(
        models={"policy": policy, "value": value},
        memory=None,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=_base_cfg(),
    )
    agent.init()

    with pytest.raises(RuntimeError, match="memory"):
        agent.update(timestep=0, timesteps=ROLLOUTS)


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
def test_rollout_cannot_be_reused_or_partially_overwritten_before_manual_update(agent_class):
    agent = _make_agent(agent_class, cfg_overrides={"learning_rate": 0.0})
    _fixed_rollout(agent)
    agent.enable_models_training_mode(True)
    agent.update(timestep=ROLLOUTS - 1, timesteps=ROLLOUTS)

    with pytest.raises(RuntimeError, match="already been consumed"):
        agent.update(timestep=ROLLOUTS, timesteps=2 * ROLLOUTS)

    observation = torch.zeros((1, 2))
    state = torch.zeros((1, 2))
    agent.enable_models_training_mode(False)
    with torch.no_grad():
        action, _ = agent.act(observation, state, timestep=0, timesteps=ROLLOUTS)
        agent.record_transition(
            observations=observation,
            states=state,
            actions=action,
            rewards=torch.zeros((1, 1)),
            next_observations=observation,
            next_states=state,
            terminated=torch.zeros((1, 1), dtype=torch.bool),
            truncated=torch.zeros((1, 1), dtype=torch.bool),
            infos={},
            timestep=0,
            timesteps=ROLLOUTS,
        )
    assert len(agent.memory) == ROLLOUTS
    assert agent.memory.memory_index == 1
    with pytest.raises(RuntimeError, match="fresh full rollout"):
        agent.update(timestep=0, timesteps=ROLLOUTS)


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN], ids=["ppo", "ppo_rnn"])
@pytest.mark.parametrize(
    "corruption_target",
    [
        "observations",
        "states",
        "actions",
        "rewards",
        "log_prob",
        "values",
        "next_observations",
        "next_states",
        "policy_weight",
        "value_weight",
    ],
)
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_nonfinite_rollout_data_fails_before_the_optimizer_step(
    agent_class, corruption_target, nonfinite
):
    agent = _make_agent(agent_class)
    _fixed_rollout(agent)
    if corruption_target == "next_observations":
        agent._current_next_observations[0, 0] = nonfinite
    elif corruption_target == "next_states":
        agent._current_next_states[0, 0] = nonfinite
    elif corruption_target == "policy_weight":
        agent.policy.weight.data[0] = nonfinite
    elif corruption_target == "value_weight":
        agent.value.weight.data[0] = nonfinite
    else:
        tensor = agent.memory.get_tensor_by_name(corruption_target)
        tensor.view(-1, tensor.shape[-1])[0, 0] = nonfinite
    before = [parameter.detach().clone() for model in (agent.policy, agent.value) for parameter in model.parameters()]

    with pytest.raises(FloatingPointError, match="NaN|Inf|finite"):
        agent.update(timestep=ROLLOUTS - 1, timesteps=ROLLOUTS)

    after = [parameter.detach() for model in (agent.policy, agent.value) for parameter in model.parameters()]
    for old, new in zip(before, after):
        torch.testing.assert_close(old, new, rtol=0, atol=0, equal_nan=True)
