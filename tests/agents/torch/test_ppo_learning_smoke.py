"""Fast end-to-end learnability tests for PPO and recurrent PPO.

The environments are deliberately tiny and analytically interpretable. These
tests complement update-level oracles: they fail when all local tensor shapes
look valid but the complete data-collection/update loop cannot improve a policy.
"""

from __future__ import annotations

import math

import gymnasium
import pytest
import torch

from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler


_GAUSSIAN_TARGET = 0.70


class _BanditPolicy(CategoricalMixin, Model):
    def __init__(self, observation_space, state_space, action_space):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        CategoricalMixin.__init__(self)
        self.logits = torch.nn.Parameter(torch.zeros(2))

    def compute(self, inputs, role=""):
        return self.logits.unsqueeze(0).expand(inputs["observations"].shape[0], -1), {}


class _BanditValue(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self.value = torch.nn.Parameter(torch.zeros(1))

    def compute(self, inputs, role=""):
        return self.value.view(1, 1).expand(inputs["states"].shape[0], 1), {}


class _GaussianBanditPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        GaussianMixin.__init__(self, min_log_std=-4.0, max_log_std=1.0, reduction="sum")
        self.mean = torch.nn.Parameter(torch.zeros(1))
        self.log_std = torch.nn.Parameter(torch.full((1,), -0.20))

    def compute(self, inputs, role=""):
        batch_size = inputs["observations"].shape[0]
        return self.mean.view(1, 1).expand(batch_size, 1), {
            "log_std": self.log_std.view(1, 1).expand(batch_size, 1)
        }


class _GRUCore:
    def _build_core(self, *, input_key, output_size, sequence_length, num_envs, hidden_size):
        self.input_key = input_key
        self.sequence_length = sequence_length
        self.num_envs = num_envs
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(3, hidden_size, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, output_size)

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(1, self.num_envs, self.hidden_size)],
            }
        }

    def _recurrent_output(self, inputs):
        x = inputs[self.input_key]
        hidden = inputs["rnn"][0]
        if not self.training:
            output, hidden = self.gru(x.unsqueeze(1), hidden)
            return self.head(output.squeeze(1)), hidden

        sequence_count = x.shape[0] // self.sequence_length
        assert sequence_count * self.sequence_length == x.shape[0]
        x = x.view(sequence_count, self.sequence_length, x.shape[-1])
        hidden = hidden.view(1, sequence_count, self.sequence_length, self.hidden_size)[:, :, 0].contiguous()
        terminated = inputs["terminated"].view(sequence_count, self.sequence_length)
        truncated = inputs["truncated"].view(sequence_count, self.sequence_length)
        done = terminated | truncated

        outputs = []
        for step in range(self.sequence_length):
            output, hidden = self.gru(x[:, step : step + 1], hidden)
            outputs.append(output)
            hidden = hidden * (~done[:, step]).view(1, -1, 1)
        return self.head(torch.cat(outputs, dim=1).flatten(0, 1)), hidden


class _CuePolicy(_GRUCore, CategoricalMixin, Model):
    def __init__(self, observation_space, state_space, action_space, *, sequence_length, num_envs):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        CategoricalMixin.__init__(self)
        self._build_core(
            input_key="observations",
            output_size=2,
            sequence_length=sequence_length,
            num_envs=num_envs,
            hidden_size=12,
        )

    def compute(self, inputs, role=""):
        logits, hidden = self._recurrent_output(inputs)
        return logits, {"rnn": [hidden]}


class _CueValue(_GRUCore, DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, *, sequence_length, num_envs):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self._build_core(
            input_key="states",
            output_size=1,
            sequence_length=sequence_length,
            num_envs=num_envs,
            hidden_size=10,
        )

    def compute(self, inputs, role=""):
        value, hidden = self._recurrent_output(inputs)
        return value, {"rnn": [hidden]}


class _CueMemoryEnv:
    """Vectorized POMDP with independently resetting two-to-five-step episodes."""

    def __init__(self, num_envs: int, *, seed: int):
        self.num_envs = num_envs
        self.generator = torch.Generator().manual_seed(seed)
        self.cue = torch.empty(num_envs, dtype=torch.long)
        self.length = torch.empty(num_envs, dtype=torch.long)
        self.step_index = torch.zeros(num_envs, dtype=torch.long)
        self._reset(torch.arange(num_envs))

    def _reset(self, indexes: torch.Tensor):
        count = indexes.numel()
        self.cue[indexes] = torch.randint(0, 2, (count,), generator=self.generator)
        self.length[indexes] = torch.randint(2, 6, (count,), generator=self.generator)
        self.step_index[indexes] = 0

    def observe(self):
        cue_phase = self.step_index == 0
        query_phase = self.step_index == self.length - 1
        cue_sign = self.cue.float().mul(2).sub(1)
        return torch.stack(
            (cue_sign * cue_phase.float(), cue_phase.float(), query_phase.float()), dim=1
        )

    def advance(self, actions: torch.Tensor):
        actions = actions.view(-1).long()
        done = self.step_index == self.length - 1
        correct = actions == self.cue
        rewards = torch.where(done, torch.where(correct, 1.0, -1.0), 0.0).unsqueeze(-1)
        self.step_index += 1
        if done.any():
            self._reset(done.nonzero(as_tuple=False).view(-1))
        return rewards, done.unsqueeze(-1), self.observe(), correct & done


def _base_cfg(*, rollouts, learning_rate, learning_epochs, mini_batches, entropy_loss_scale):
    return {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "discount_factor": 0.97,
        "lambda_": 0.95,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "optimizer_kwargs": {},
        "observation_preprocessor": None,
        "state_preprocessor": None,
        "value_preprocessor": None,
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "entropy_loss_scale": entropy_loss_scale,
        "value_loss_scale": 0.5,
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


def _assert_finite_update(agent):
    for name, values in agent.tracking_data.items():
        assert values and math.isfinite(values[-1]), f"non-finite tracked value: {name}={values[-1]}"
    for name, tensor in agent.memory.tensors.items():
        assert torch.isfinite(tensor).all(), f"non-finite rollout tensor: {name}"
    for model in (agent.policy, agent.value):
        for name, parameter in model.named_parameters():
            assert torch.isfinite(parameter).all(), f"non-finite parameter: {name}"
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"


def _parameters_changed(model, initial_parameters):
    return any(
        not torch.equal(parameter.detach(), initial)
        for parameter, initial in zip(model.parameters(), initial_parameters)
    )


def _bandit_probability(policy):
    return torch.softmax(policy.logits.detach(), dim=-1)[1].item()


def _train_bandit(agent, *, updates):
    num_envs = agent.memory.num_envs
    observations = torch.zeros((num_envs, 1))
    states = torch.zeros((num_envs, 1))
    qualities = []
    for update in range(updates):
        agent.enable_models_training_mode(False)
        for step in range(agent.cfg.rollouts):
            with torch.no_grad():
                actions, _ = agent.act(observations, states, timestep=step, timesteps=agent.cfg.rollouts)
                rewards = torch.where(actions == 1, 1.0, -1.0)
                agent.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=observations,
                    next_states=states,
                    terminated=torch.ones((num_envs, 1), dtype=torch.bool),
                    truncated=torch.zeros((num_envs, 1), dtype=torch.bool),
                    infos={},
                    timestep=step,
                    timesteps=agent.cfg.rollouts,
                )
        agent.enable_models_training_mode(True)
        agent.update(timestep=update, timesteps=updates)
        _assert_finite_update(agent)
        qualities.append(_bandit_probability(agent.policy))
    agent.enable_models_training_mode(False)
    return qualities


def _make_bandit_agent(*, seed=4103):
    torch.manual_seed(seed)
    num_envs = 48
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
    action_space = gymnasium.spaces.Discrete(2)
    policy = _BanditPolicy(observation_space, state_space, action_space)
    value = _BanditValue(observation_space, state_space, action_space)
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=4, num_envs=num_envs, device="cpu"),
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=_base_cfg(
            rollouts=4,
            learning_rate=0.035,
            learning_epochs=4,
            mini_batches=4,
            entropy_loss_scale=0.0,
        ),
    )
    agent.init()
    return agent


def _make_gaussian_bandit_agent(*, seed, use_scalers=False):
    torch.manual_seed(seed)
    num_envs = 48
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
    action_space = gymnasium.spaces.Box(low=-4, high=4, shape=(1,))
    policy = _GaussianBanditPolicy(observation_space, state_space, action_space)
    value = _BanditValue(observation_space, state_space, action_space)
    cfg = _base_cfg(
        rollouts=4,
        learning_rate=0.025,
        learning_epochs=4,
        mini_batches=4,
        entropy_loss_scale=0.0,
    )
    if use_scalers:
        cfg.update(
            {
                "observation_preprocessor": RunningStandardScaler,
                "observation_preprocessor_kwargs": {"size": observation_space, "device": "cpu"},
                "state_preprocessor": RunningStandardScaler,
                "state_preprocessor_kwargs": {"size": state_space, "device": "cpu"},
                "value_preprocessor": RunningStandardScaler,
                "value_preprocessor_kwargs": {"size": 1, "device": "cpu"},
            }
        )
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=4, num_envs=num_envs, device="cpu"),
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    return agent


def _gaussian_policy_quality(policy):
    mean = policy.mean.detach().item()
    std = policy.log_std.detach().clamp(min=-4.0, max=1.0).exp().item()
    # E[(A - target)^2] for A ~ Normal(mean, std), computed analytically.
    expected_squared_error = (mean - _GAUSSIAN_TARGET) ** 2 + std**2
    return mean, std, expected_squared_error


def _train_gaussian_bandit(agent, *, updates):
    num_envs = agent.memory.num_envs
    observations = torch.zeros((num_envs, 1))
    states = torch.zeros((num_envs, 1))
    qualities = []
    for update in range(updates):
        agent.enable_models_training_mode(False)
        for step in range(agent.cfg.rollouts):
            with torch.no_grad():
                actions, _ = agent.act(observations, states, timestep=step, timesteps=agent.cfg.rollouts)
                rewards = 1.0 - (actions - _GAUSSIAN_TARGET).square()
                agent.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=observations,
                    next_states=states,
                    terminated=torch.ones((num_envs, 1), dtype=torch.bool),
                    truncated=torch.zeros((num_envs, 1), dtype=torch.bool),
                    infos={},
                    timestep=step,
                    timesteps=agent.cfg.rollouts,
                )
        agent.enable_models_training_mode(True)
        agent.update(timestep=update, timesteps=updates)
        _assert_finite_update(agent)
        qualities.append(_gaussian_policy_quality(agent.policy))
    agent.enable_models_training_mode(False)
    return qualities


def _make_cue_agent(*, seed=8127, use_scalers=False):
    torch.manual_seed(seed)
    num_envs = 16
    rollouts = 8
    sequence_length = 4
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))
    action_space = gymnasium.spaces.Discrete(2)
    policy = _CuePolicy(
        observation_space,
        state_space,
        action_space,
        sequence_length=sequence_length,
        num_envs=num_envs,
    )
    value = _CueValue(
        observation_space,
        state_space,
        action_space,
        sequence_length=sequence_length,
        num_envs=num_envs,
    )
    cfg = _base_cfg(
        rollouts=rollouts,
        learning_rate=0.012,
        learning_epochs=4,
        mini_batches=4,
        entropy_loss_scale=0.005,
    )
    if use_scalers:
        cfg.update(
            {
                "observation_preprocessor": RunningStandardScaler,
                "observation_preprocessor_kwargs": {"size": observation_space, "device": "cpu"},
                "state_preprocessor": RunningStandardScaler,
                "state_preprocessor_kwargs": {"size": state_space, "device": "cpu"},
                "value_preprocessor": RunningStandardScaler,
                "value_preprocessor_kwargs": {"size": 1, "device": "cpu"},
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
    return agent


def _evaluate_cue_policy(policy, *, seed, episode_target=512, reset_hidden_every_step=False):
    num_envs = 64
    environment = _CueMemoryEnv(num_envs, seed=seed)
    hidden = torch.zeros((1, num_envs, policy.hidden_size))
    probability_sum = 0.0
    greedy_successes = 0
    episode_count = 0
    policy.eval()

    with torch.no_grad():
        while episode_count < episode_target:
            observations = environment.observe()
            logits, outputs = policy.compute(
                {"observations": observations, "states": observations, "rnn": [hidden]}, role="policy"
            )
            hidden = outputs["rnn"][0]
            query = (environment.step_index == environment.length - 1).nonzero(as_tuple=False).view(-1)
            if query.numel():
                probabilities = torch.softmax(logits[query], dim=-1)
                cues = environment.cue[query]
                probability_sum += probabilities[torch.arange(query.numel()), cues].sum().item()
                greedy_successes += (probabilities.argmax(dim=-1) == cues).sum().item()
                episode_count += query.numel()
            actions = logits.argmax(dim=-1, keepdim=True)
            _, done, _, _ = environment.advance(actions)
            hidden[:, done.view(-1)] = 0
            if reset_hidden_every_step:
                hidden.zero_()

    return probability_sum / episode_count, greedy_successes / episode_count


def _train_cue_agent(agent, *, updates):
    environment = _CueMemoryEnv(agent.memory.num_envs, seed=991)
    query_successes = 0
    query_count = 0
    asynchronous_reset_steps = 0
    for update in range(updates):
        agent.enable_models_training_mode(False)
        for step in range(agent.cfg.rollouts):
            observations = environment.observe()
            with torch.no_grad():
                actions, _ = agent.act(
                    observations,
                    observations,
                    timestep=step,
                    timesteps=agent.cfg.rollouts,
                )
                rewards, terminated, next_observations, correct_queries = environment.advance(actions)
                query_successes += correct_queries.sum().item()
                query_count += terminated.sum().item()
                asynchronous_reset_steps += int(terminated.any() and (~terminated).any())
                agent.record_transition(
                    observations=observations,
                    states=observations,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_observations,
                    terminated=terminated,
                    truncated=torch.zeros_like(terminated),
                    infos={},
                    timestep=step,
                    timesteps=agent.cfg.rollouts,
                )
        agent.enable_models_training_mode(True)
        agent.update(timestep=update, timesteps=updates)
        _assert_finite_update(agent)
    agent.enable_models_training_mode(False)
    return query_successes / query_count, asynchronous_reset_steps


@pytest.mark.parametrize("seed", [4103, 5119, 8127])
def test_ppo_learns_the_analytically_optimal_bandit_action(seed):
    agent = _make_bandit_agent(seed=seed)
    initial_policy = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    initial_value = [parameter.detach().clone() for parameter in agent.value.parameters()]
    initial_probability = _bandit_probability(agent.policy)

    probabilities = _train_bandit(agent, updates=10)

    assert initial_probability == pytest.approx(0.5, abs=1e-8)
    assert probabilities[4] > 0.80
    assert probabilities[-1] > 0.97
    assert probabilities[-1] > initial_probability + 0.45
    assert _parameters_changed(agent.policy, initial_policy)
    assert _parameters_changed(agent.value, initial_value)
    assert agent.tracking_data["Optimization / Grad norm"][-1] > 0
    assert "Policy / Maximum action probability" in agent.tracking_data


@pytest.mark.parametrize("seed", [101, 509, 997])
def test_ppo_gaussian_bandit_converges_to_the_analytic_target(seed):
    agent = _make_gaussian_bandit_agent(seed=seed)
    initial_policy = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    initial_value = [parameter.detach().clone() for parameter in agent.value.parameters()]
    initial_mean, initial_std, initial_error = _gaussian_policy_quality(agent.policy)

    qualities = _train_gaussian_bandit(agent, updates=12)
    middle_mean, middle_std, middle_error = qualities[5]
    final_mean, final_std, final_error = qualities[-1]

    assert initial_mean == pytest.approx(0.0, abs=1e-8)
    assert math.isfinite(initial_std) and initial_std > 0
    assert math.isfinite(middle_std) and middle_std > 0
    assert abs(middle_mean - _GAUSSIAN_TARGET) < abs(initial_mean - _GAUSSIAN_TARGET)
    assert middle_error < initial_error * 0.40
    assert final_mean == pytest.approx(_GAUSSIAN_TARGET, abs=0.08)
    assert 0 < final_std < 0.25
    assert final_error < initial_error * 0.08
    assert final_error < middle_error
    assert _parameters_changed(agent.policy, initial_policy)
    assert _parameters_changed(agent.value, initial_value)
    assert agent.tracking_data["Optimization / Grad norm"][-1] > 0
    assert "Policy / Standard deviation" in agent.tracking_data
    reported_std = agent.tracking_data["Policy / Standard deviation"][-1]
    assert math.isfinite(reported_std) and reported_std > 0
    # The tracked distribution is the final minibatch's pre-optimizer-step
    # distribution, so it may lag the post-update parameter by one small step.
    assert reported_std == pytest.approx(final_std, rel=0.05)


def test_ppo_with_all_running_scalers_still_learns_gaussian_bandit():
    agent = _make_gaussian_bandit_agent(seed=509, use_scalers=True)
    _, _, initial_error = _gaussian_policy_quality(agent.policy)
    final_mean, final_std, final_error = _train_gaussian_bandit(agent, updates=12)[-1]

    assert final_mean == pytest.approx(_GAUSSIAN_TARGET, abs=0.10)
    assert 0 < final_std < 0.25
    assert final_error < initial_error * 0.10
    assert agent._observation_preprocessor.current_count.item() > 1
    assert agent._state_preprocessor.current_count.item() > 1
    assert agent._value_preprocessor.current_count.item() > 1


@pytest.mark.parametrize("seed", [8127, 9131, 10243])
def test_ppo_rnn_learns_cue_memory_with_asynchronous_resets(seed):
    agent = _make_cue_agent(seed=seed)
    initial_policy = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    initial_value = [parameter.detach().clone() for parameter in agent.value.parameters()]
    initial_probability, initial_success = _evaluate_cue_policy(agent.policy, seed=121)

    sampled_training_success, asynchronous_reset_steps = _train_cue_agent(agent, updates=14)
    final_probability, final_success = _evaluate_cue_policy(agent.policy, seed=121)
    memoryless_probability, memoryless_success = _evaluate_cue_policy(
        agent.policy, seed=121, reset_hidden_every_step=True
    )

    assert 0.35 < initial_probability < 0.65
    assert final_probability > 0.82
    assert final_probability > initial_probability + 0.25
    assert final_success > 0.90
    assert final_success > initial_success + 0.30
    assert sampled_training_success > 0.60
    assert asynchronous_reset_steps > 0
    assert memoryless_probability < 0.60
    assert memoryless_success < 0.60
    assert final_probability > memoryless_probability + 0.30
    assert _parameters_changed(agent.policy, initial_policy)
    assert _parameters_changed(agent.value, initial_value)
    assert agent.tracking_data["Optimization / Grad norm"][-1] > 0
    assert "Policy / Maximum action probability" in agent.tracking_data


def test_ppo_rnn_with_all_running_scalers_still_learns_cue_memory():
    agent = _make_cue_agent(seed=9131, use_scalers=True)
    initial_probability, _ = _evaluate_cue_policy(agent.policy, seed=121)
    _train_cue_agent(agent, updates=14)
    final_probability, final_success = _evaluate_cue_policy(agent.policy, seed=121)
    memoryless_probability, memoryless_success = _evaluate_cue_policy(
        agent.policy, seed=121, reset_hidden_every_step=True
    )

    assert final_probability > 0.85
    assert final_probability > initial_probability + 0.30
    assert final_success > 0.95
    assert memoryless_probability < 0.60
    assert memoryless_success < 0.60
    assert final_probability > memoryless_probability + 0.30
    assert agent._observation_preprocessor.current_count.item() > 1
    assert agent._state_preprocessor.current_count.item() > 1
    assert agent._value_preprocessor.current_count.item() > 1
